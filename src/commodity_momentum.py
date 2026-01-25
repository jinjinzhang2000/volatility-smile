#!/usr/bin/env python3
"""
Commodity Momentum Ranking for Chinese Futures Markets

Ranks commodities by various momentum metrics:
- Price momentum (returns over different periods)
- Trend strength (price vs moving averages)
- Volume momentum (volume changes)
- Open interest momentum

Supports all major Chinese futures exchanges:
- SHFE (Shanghai Futures Exchange)
- DCE (Dalian Commodity Exchange)
- CZCE (Zhengzhou Commodity Exchange)
- CFFEX (China Financial Futures Exchange)
- INE (Shanghai International Energy Exchange)
"""

import tushare as ts
import pandas as pd
import numpy as np
import os
from datetime import datetime, timedelta
import matplotlib.pyplot as plt
import matplotlib

matplotlib.rcParams['font.sans-serif'] = ['SimHei', 'Arial Unicode MS', 'Microsoft YaHei']
matplotlib.rcParams['axes.unicode_minus'] = False

# Tushare token
TUSHARE_TOKEN = os.environ.get("TUSHARE_TOKEN", "a70287c82208760b640d7f08525b97181166b817e0d9ff5f8f244bc2")

# Commodity info: code -> (name, exchange)
COMMODITIES = {
    # SHFE - Shanghai Futures Exchange
    'RB': ('螺纹钢', 'SHFE'),
    'HC': ('热卷', 'SHFE'),
    'SS': ('不锈钢', 'SHFE'),
    'CU': ('铜', 'SHFE'),
    'AL': ('铝', 'SHFE'),
    'ZN': ('锌', 'SHFE'),
    'PB': ('铅', 'SHFE'),
    'NI': ('镍', 'SHFE'),
    'SN': ('锡', 'SHFE'),
    'AU': ('黄金', 'SHFE'),
    'AG': ('白银', 'SHFE'),
    'RU': ('天然橡胶', 'SHFE'),
    'FU': ('燃料油', 'SHFE'),
    'BU': ('沥青', 'SHFE'),
    'SP': ('纸浆', 'SHFE'),
    'WR': ('线材', 'SHFE'),
    'AO': ('氧化铝', 'SHFE'),
    'BR': ('丁二烯橡胶', 'SHFE'),

    # DCE - Dalian Commodity Exchange
    'I': ('铁矿石', 'DCE'),
    'J': ('焦炭', 'DCE'),
    'JM': ('焦煤', 'DCE'),
    'A': ('豆一', 'DCE'),
    'B': ('豆二', 'DCE'),
    'M': ('豆粕', 'DCE'),
    'Y': ('豆油', 'DCE'),
    'P': ('棕榈油', 'DCE'),
    'C': ('玉米', 'DCE'),
    'CS': ('玉米淀粉', 'DCE'),
    'JD': ('鸡蛋', 'DCE'),
    'L': ('塑料', 'DCE'),
    'PP': ('聚丙烯', 'DCE'),
    'V': ('PVC', 'DCE'),
    'EB': ('苯乙烯', 'DCE'),
    'EG': ('乙二醇', 'DCE'),
    'PG': ('液化石油气', 'DCE'),
    'LH': ('生猪', 'DCE'),

    # CZCE - Zhengzhou Commodity Exchange
    'SR': ('白糖', 'CZCE'),
    'CF': ('棉花', 'CZCE'),
    'CY': ('棉纱', 'CZCE'),
    'TA': ('PTA', 'CZCE'),
    'MA': ('甲醇', 'CZCE'),
    'FG': ('玻璃', 'CZCE'),
    'SA': ('纯碱', 'CZCE'),
    'RM': ('菜粕', 'CZCE'),
    'OI': ('菜油', 'CZCE'),
    'AP': ('苹果', 'CZCE'),
    'CJ': ('红枣', 'CZCE'),
    'PK': ('花生', 'CZCE'),
    'SF': ('硅铁', 'CZCE'),
    'SM': ('锰硅', 'CZCE'),
    'UR': ('尿素', 'CZCE'),
    'PX': ('PX', 'CZCE'),
    'SH': ('烧碱', 'CZCE'),
    'PF': ('涤纶短纤', 'CZCE'),

    # INE - Shanghai International Energy Exchange
    'SC': ('原油', 'INE'),
    'LU': ('低硫燃料油', 'INE'),
    'NR': ('20号胶', 'INE'),
    'BC': ('国际铜', 'INE'),
    'EC': ('集运指数', 'INE'),
}


def init_tushare():
    """Initialize Tushare API"""
    ts.set_token(TUSHARE_TOKEN)
    return ts.pro_api()


def get_main_contract(pro, symbol, exchange, trade_date):
    """
    Get the main (most liquid) contract for a commodity
    Uses open interest to determine the main contract
    """
    # Map exchange codes for Tushare
    exchange_map = {
        'SHFE': 'SHFE',
        'DCE': 'DCE',
        'CZCE': 'CZCE',
        'INE': 'INE',
        'CFFEX': 'CFFEX'
    }
    ts_exchange = exchange_map.get(exchange, exchange)

    try:
        # Get all contracts for this symbol
        df = pro.fut_daily(
            exchange=ts_exchange,
            trade_date=trade_date,
            fields='ts_code,trade_date,close,settle,vol,oi'
        )

        if df is None or df.empty:
            return None

        # Filter by symbol prefix
        df = df[df['ts_code'].str.upper().str.startswith(symbol.upper())]

        if df.empty:
            return None

        # Get contract with highest open interest (main contract)
        main_contract = df.loc[df['oi'].idxmax()]
        return main_contract['ts_code']

    except Exception as e:
        print(f"  Error getting main contract for {symbol}: {e}")
        return None


def download_exchange_data(pro, exchange, start_date, end_date):
    """
    Download all futures data for an exchange
    Makes multiple API calls to get sufficient historical data
    Returns cached data for efficiency
    """
    import time

    cache_key = f"{exchange}_{start_date}_{end_date}"

    # Use a simple cache mechanism
    if not hasattr(download_exchange_data, 'cache'):
        download_exchange_data.cache = {}

    if cache_key in download_exchange_data.cache:
        return download_exchange_data.cache[cache_key]

    try:
        # Make calls for each trading day to get complete data
        # This is slower but more reliable
        all_data = []
        current_date = datetime.strptime(end_date, '%Y%m%d')
        min_date = datetime.strptime(start_date, '%Y%m%d')

        # Download day by day (more reliable for getting all contracts)
        days_fetched = 0
        while current_date >= min_date and days_fetched < 100:
            date_str = current_date.strftime('%Y%m%d')

            df = pro.fut_daily(
                exchange=exchange,
                trade_date=date_str,
                fields='ts_code,trade_date,open,high,low,close,settle,vol,oi'
            )

            if df is not None and not df.empty:
                all_data.append(df)
                days_fetched += 1

            current_date -= timedelta(days=1)
            time.sleep(0.1)  # Rate limiting

        if not all_data:
            return None

        result = pd.concat(all_data, ignore_index=True)
        download_exchange_data.cache[cache_key] = result
        return result

    except Exception as e:
        print(f"  Error downloading {exchange} data: {e}")
        return None


def get_continuous_prices(pro, symbol, exchange, end_date, days=120):
    """
    Get continuous price series for a commodity using main contracts
    Returns daily prices for the specified number of trading days
    """
    # Calculate start date (add buffer for non-trading days)
    start_date = (datetime.strptime(end_date, '%Y%m%d') - timedelta(days=days * 2)).strftime('%Y%m%d')

    # Download all data for this exchange (cached)
    df = download_exchange_data(pro, exchange, start_date, end_date)

    if df is None or df.empty:
        return None

    # Filter by symbol prefix (handle special cases like 'I' for iron ore)
    if symbol.upper() == 'I':
        # Iron ore: match I followed by 4 digits
        mask = df['ts_code'].str.match(r'^I\d{4}\.', case=False)
    else:
        mask = df['ts_code'].str.upper().str.startswith(symbol.upper())

    df = df[mask]

    if df.empty:
        return None

    # Exclude index contracts (like RBL.SHF)
    df = df[~df['ts_code'].str.contains('L\\.', regex=True)]

    # For each date, use the contract with highest OI (main contract)
    continuous_data = []
    for date in sorted(df['trade_date'].unique()):
        day_data = df[df['trade_date'] == date]
        if not day_data.empty and day_data['oi'].max() > 0:
            main = day_data.loc[day_data['oi'].idxmax()]
            continuous_data.append({
                'trade_date': date,
                'ts_code': main['ts_code'],
                'open': main['open'],
                'high': main['high'],
                'low': main['low'],
                'close': main['close'],
                'settle': main['settle'],
                'volume': main['vol'],
                'oi': main['oi']
            })

    if not continuous_data:
        return None

    result = pd.DataFrame(continuous_data)
    result = result.sort_values('trade_date').tail(days)
    return result


def calculate_momentum_metrics(df):
    """
    Calculate various momentum metrics from price data

    Returns dict with:
    - ret_1d, ret_5d, ret_10d, ret_20d, ret_60d: Returns over different periods
    - ma_ratio_5, ma_ratio_20, ma_ratio_60: Price vs MA ratios
    - trend_score: Composite trend strength
    - vol_change_5d, vol_change_20d: Volume changes
    - oi_change_5d, oi_change_20d: Open interest changes
    """
    if df is None or len(df) < 20:
        return None

    df = df.sort_values('trade_date').reset_index(drop=True)

    # Use close price (or settle if close not available)
    prices = df['close'].fillna(df['settle'])
    volumes = df['volume']
    oi = df['oi']

    current_price = prices.iloc[-1]

    metrics = {}

    # Price returns over different periods
    for period, label in [(1, '1d'), (5, '5d'), (10, '10d'), (20, '20d'), (60, '60d')]:
        if len(prices) > period:
            ret = (current_price / prices.iloc[-period-1] - 1) * 100
            metrics[f'ret_{label}'] = ret
        else:
            metrics[f'ret_{label}'] = np.nan

    # Moving average ratios (trend indicators)
    for period, label in [(5, '5'), (20, '20'), (60, '60')]:
        if len(prices) >= period:
            ma = prices.tail(period).mean()
            metrics[f'ma_ratio_{label}'] = (current_price / ma - 1) * 100
        else:
            metrics[f'ma_ratio_{label}'] = np.nan

    # Trend score: composite of MA ratios
    ma_scores = []
    for label in ['5', '20', '60']:
        if not np.isnan(metrics.get(f'ma_ratio_{label}', np.nan)):
            ma_scores.append(metrics[f'ma_ratio_{label}'])
    metrics['trend_score'] = np.mean(ma_scores) if ma_scores else np.nan

    # EWMAC (Exponentially Weighted Moving Average Crossover)
    # Raw EWMAC = EMA(fast) - EMA(slow)
    # Normalized EWMAC = Raw EWMAC / volatility (for cross-asset comparison)
    # Common pairs: (2,8), (4,16), (8,32), (16,64), (32,128)
    ewmac_pairs = [(2, 8), (4, 16), (8, 32), (16, 64)]

    # Calculate volatility for normalization (using 20-day rolling std of returns)
    if len(prices) >= 20:
        returns = prices.pct_change().dropna()
        volatility = returns.tail(20).std()
    else:
        volatility = np.nan

    ewmac_values = []
    for fast, slow in ewmac_pairs:
        if len(prices) >= slow:
            # Calculate EMAs using pandas ewm
            # span = 2 * n - 1 gives same decay as simple n-day lookback
            ema_fast = prices.ewm(span=fast, adjust=False).mean().iloc[-1]
            ema_slow = prices.ewm(span=slow, adjust=False).mean().iloc[-1]

            raw_ewmac = ema_fast - ema_slow

            # Normalize by price * volatility to get comparable signal across assets
            if volatility > 0 and current_price > 0:
                # Normalize: divide by (price * volatility) to get unit-less signal
                normalized_ewmac = raw_ewmac / (current_price * volatility)
                metrics[f'ewmac_{fast}_{slow}'] = normalized_ewmac
                ewmac_values.append(normalized_ewmac)
            else:
                metrics[f'ewmac_{fast}_{slow}'] = np.nan
        else:
            metrics[f'ewmac_{fast}_{slow}'] = np.nan

    # Composite EWMAC score (average of all EWMAC signals)
    metrics['ewmac_score'] = np.mean(ewmac_values) if ewmac_values else np.nan

    # Volume momentum
    for period, label in [(5, '5d'), (20, '20d')]:
        if len(volumes) > period:
            recent_vol = volumes.tail(period).mean()
            prev_vol = volumes.iloc[-period*2:-period].mean() if len(volumes) >= period*2 else volumes.iloc[:-period].mean()
            if prev_vol > 0:
                metrics[f'vol_change_{label}'] = (recent_vol / prev_vol - 1) * 100
            else:
                metrics[f'vol_change_{label}'] = np.nan
        else:
            metrics[f'vol_change_{label}'] = np.nan

    # Open interest momentum
    for period, label in [(5, '5d'), (20, '20d')]:
        if len(oi) > period:
            current_oi = oi.iloc[-1]
            prev_oi = oi.iloc[-period-1]
            if prev_oi > 0:
                metrics[f'oi_change_{label}'] = (current_oi / prev_oi - 1) * 100
            else:
                metrics[f'oi_change_{label}'] = np.nan
        else:
            metrics[f'oi_change_{label}'] = np.nan

    # Current values
    metrics['last_price'] = current_price
    metrics['volume'] = volumes.iloc[-1]
    metrics['oi'] = oi.iloc[-1]

    return metrics


def rank_commodities(trade_date=None, lookback_days=120):
    """
    Rank all commodities by momentum metrics

    Args:
        trade_date: Date to calculate rankings (YYYYMMDD format), default today
        lookback_days: Number of days of price history to use

    Returns:
        DataFrame with momentum rankings
    """
    if trade_date is None:
        trade_date = datetime.now().strftime('%Y%m%d')

    print(f"=" * 60)
    print(f"Commodity Momentum Ranking - {trade_date}")
    print(f"=" * 60)

    pro = init_tushare()

    results = []

    for symbol, (name, exchange) in COMMODITIES.items():
        print(f"Processing {symbol} ({name})...")

        # Get price data
        df = get_continuous_prices(pro, symbol, exchange, trade_date, lookback_days)

        if df is None or df.empty:
            print(f"  No data available")
            continue

        # Calculate metrics
        metrics = calculate_momentum_metrics(df)

        if metrics is None:
            print(f"  Insufficient data for metrics")
            continue

        metrics['symbol'] = symbol
        metrics['name'] = name
        metrics['exchange'] = exchange
        metrics['data_points'] = len(df)

        results.append(metrics)
        print(f"  OK - {len(df)} days, ret_20d={metrics.get('ret_20d', 0):.2f}%")

    if not results:
        print("No data collected!")
        return None

    # Create DataFrame
    df = pd.DataFrame(results)

    # Reorder columns
    cols = ['symbol', 'name', 'exchange', 'last_price',
            'ret_1d', 'ret_5d', 'ret_10d', 'ret_20d', 'ret_60d',
            'ma_ratio_5', 'ma_ratio_20', 'ma_ratio_60', 'trend_score',
            'ewmac_2_8', 'ewmac_4_16', 'ewmac_8_32', 'ewmac_16_64', 'ewmac_score',
            'vol_change_5d', 'vol_change_20d', 'oi_change_5d', 'oi_change_20d',
            'volume', 'oi', 'data_points']
    df = df[[c for c in cols if c in df.columns]]

    return df


def print_rankings(df, metric='ret_20d', top_n=15):
    """Print top and bottom commodities by a given metric"""
    if df is None or df.empty:
        print("No data to display")
        return

    valid_df = df.dropna(subset=[metric])

    # Determine format based on metric type
    is_ewmac = 'ewmac' in metric
    fmt = '{:+7.2f}' if is_ewmac else '{:+7.2f}%'
    unit = '' if is_ewmac else '%'

    print(f"\n{'='*60}")
    print(f"TOP {top_n} by {metric}")
    print(f"{'='*60}")

    top = valid_df.nlargest(top_n, metric)
    for i, (_, row) in enumerate(top.iterrows(), 1):
        val = fmt.format(row[metric])
        print(f"{i:2d}. {row['symbol']:4s} {row['name']:8s} | {metric}: {val}{unit} | Price: {row['last_price']:.2f}")

    print(f"\n{'='*60}")
    print(f"BOTTOM {top_n} by {metric}")
    print(f"{'='*60}")

    bottom = valid_df.nsmallest(top_n, metric)
    for i, (_, row) in enumerate(bottom.iterrows(), 1):
        val = fmt.format(row[metric])
        print(f"{i:2d}. {row['symbol']:4s} {row['name']:8s} | {metric}: {val}{unit} | Price: {row['last_price']:.2f}")


def plot_momentum_heatmap(df, save_path=None):
    """Create a heatmap of momentum metrics"""
    if df is None or df.empty:
        return

    # Select numeric columns for heatmap
    metrics = ['ret_1d', 'ret_5d', 'ret_10d', 'ret_20d', 'ret_60d',
               'ma_ratio_5', 'ma_ratio_20', 'ma_ratio_60', 'trend_score']

    # Filter to available metrics
    metrics = [m for m in metrics if m in df.columns]

    # Sort by 20-day return
    plot_df = df.dropna(subset=['ret_20d']).sort_values('ret_20d', ascending=False)

    if len(plot_df) < 5:
        print("Not enough data for heatmap")
        return

    # Create labels
    labels = [f"{row['symbol']} {row['name']}" for _, row in plot_df.iterrows()]

    # Extract data
    data = plot_df[metrics].values

    # Create figure
    fig, ax = plt.subplots(figsize=(14, max(8, len(labels) * 0.4)))

    # Create heatmap
    im = ax.imshow(data, cmap='RdYlGn', aspect='auto', vmin=-10, vmax=10)

    # Labels
    ax.set_yticks(range(len(labels)))
    ax.set_yticklabels(labels, fontsize=9)
    ax.set_xticks(range(len(metrics)))
    ax.set_xticklabels(['1日', '5日', '10日', '20日', '60日',
                        'MA5比', 'MA20比', 'MA60比', '趋势分'][:len(metrics)],
                       fontsize=10, rotation=45, ha='right')

    # Add colorbar
    cbar = plt.colorbar(im, ax=ax, shrink=0.8)
    cbar.set_label('涨跌幅 (%)', fontsize=11)

    # Add value annotations
    for i in range(len(labels)):
        for j in range(len(metrics)):
            val = data[i, j]
            if not np.isnan(val):
                color = 'white' if abs(val) > 5 else 'black'
                ax.text(j, i, f'{val:.1f}', ha='center', va='center',
                       fontsize=7, color=color)

    ax.set_title(f'商品期货动量热图 - {datetime.now().strftime("%Y-%m-%d")}',
                 fontsize=14, fontweight='bold', pad=10)

    plt.tight_layout()

    if save_path:
        plt.savefig(save_path, dpi=150, bbox_inches='tight')
        print(f"Saved: {save_path}")

    plt.close()


def plot_ewmac_chart(df, save_path=None):
    """Create a horizontal bar chart for EWMAC signals"""
    if df is None or df.empty:
        return

    # Check if EWMAC columns exist
    ewmac_cols = ['ewmac_2_8', 'ewmac_4_16', 'ewmac_8_32', 'ewmac_16_64', 'ewmac_score']
    available_cols = [c for c in ewmac_cols if c in df.columns]

    if not available_cols or 'ewmac_score' not in df.columns:
        print("No EWMAC data available for chart")
        return

    # Sort by EWMAC score and filter valid data
    plot_df = df.dropna(subset=['ewmac_score']).sort_values('ewmac_score', ascending=True)

    if len(plot_df) < 5:
        print("Not enough data for EWMAC chart")
        return

    # Create figure with two subplots
    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(16, max(10, len(plot_df) * 0.35)))

    # Left plot: EWMAC Score bar chart
    labels = [f"{row['symbol']} {row['name']}" for _, row in plot_df.iterrows()]
    scores = plot_df['ewmac_score'].values

    # Color bars based on positive/negative
    colors = ['#d73027' if s < 0 else '#1a9850' for s in scores]

    bars = ax1.barh(range(len(labels)), scores, color=colors, edgecolor='none', height=0.7)

    ax1.set_yticks(range(len(labels)))
    ax1.set_yticklabels(labels, fontsize=9)
    ax1.axvline(x=0, color='black', linewidth=0.8)
    ax1.set_xlabel('EWMAC Score (标准化)', fontsize=11)
    ax1.set_title('EWMAC综合信号排名', fontsize=14, fontweight='bold')

    # Add value labels on bars
    for i, (bar, val) in enumerate(zip(bars, scores)):
        x_pos = val + 0.05 if val >= 0 else val - 0.05
        ha = 'left' if val >= 0 else 'right'
        ax1.text(x_pos, i, f'{val:.2f}', va='center', ha=ha, fontsize=8)

    # Right plot: EWMAC heatmap by timeframe
    ewmac_metrics = ['ewmac_2_8', 'ewmac_4_16', 'ewmac_8_32', 'ewmac_16_64']
    ewmac_metrics = [m for m in ewmac_metrics if m in df.columns]

    if ewmac_metrics:
        # Re-sort for heatmap (descending order for top-down reading)
        heatmap_df = plot_df.sort_values('ewmac_score', ascending=False)
        heatmap_labels = [f"{row['symbol']} {row['name']}" for _, row in heatmap_df.iterrows()]
        heatmap_data = heatmap_df[ewmac_metrics].values

        # Determine color scale based on data range
        vmax = max(abs(np.nanmin(heatmap_data)), abs(np.nanmax(heatmap_data)))
        vmax = min(vmax, 5)  # Cap at 5 for better visualization

        im = ax2.imshow(heatmap_data, cmap='RdYlGn', aspect='auto', vmin=-vmax, vmax=vmax)

        ax2.set_yticks(range(len(heatmap_labels)))
        ax2.set_yticklabels(heatmap_labels, fontsize=9)
        ax2.set_xticks(range(len(ewmac_metrics)))
        ax2.set_xticklabels(['快(2,8)', '中快(4,16)', '中慢(8,32)', '慢(16,64)'][:len(ewmac_metrics)],
                           fontsize=10, rotation=45, ha='right')

        # Add colorbar
        cbar = plt.colorbar(im, ax=ax2, shrink=0.8)
        cbar.set_label('EWMAC信号强度', fontsize=11)

        # Add value annotations
        for i in range(len(heatmap_labels)):
            for j in range(len(ewmac_metrics)):
                val = heatmap_data[i, j]
                if not np.isnan(val):
                    color = 'white' if abs(val) > vmax * 0.5 else 'black'
                    ax2.text(j, i, f'{val:.1f}', ha='center', va='center',
                            fontsize=7, color=color)

        ax2.set_title('EWMAC多周期信号热图', fontsize=14, fontweight='bold')

    plt.suptitle(f'EWMAC趋势跟踪信号 - {datetime.now().strftime("%Y-%m-%d")}',
                 fontsize=16, fontweight='bold', y=1.02)

    plt.tight_layout()

    if save_path:
        plt.savefig(save_path, dpi=150, bbox_inches='tight')
        print(f"Saved: {save_path}")

    plt.close()


def create_momentum_report(df, output_dir='output'):
    """Create a complete momentum report with charts and CSV"""
    os.makedirs(output_dir, exist_ok=True)

    today = datetime.now().strftime('%Y%m%d')

    # Save CSV
    csv_path = os.path.join(output_dir, f'commodity_momentum_{today}.csv')
    df.to_csv(csv_path, index=False, encoding='utf-8-sig')
    print(f"Saved: {csv_path}")

    # Save heatmap
    heatmap_path = os.path.join(output_dir, 'charts', 'commodity_momentum_heatmap.png')
    os.makedirs(os.path.dirname(heatmap_path), exist_ok=True)
    plot_momentum_heatmap(df, heatmap_path)

    # Save EWMAC chart
    ewmac_path = os.path.join(output_dir, 'charts', 'commodity_ewmac.png')
    plot_ewmac_chart(df, ewmac_path)

    return csv_path


def main():
    """Main entry point"""
    import sys

    # Parse arguments
    trade_date = None
    if len(sys.argv) > 1:
        trade_date = sys.argv[1]

    # Calculate rankings
    df = rank_commodities(trade_date=trade_date)

    if df is None or df.empty:
        print("Failed to calculate rankings")
        return

    # Print rankings for different metrics
    print_rankings(df, 'ret_20d', top_n=10)
    print_rankings(df, 'trend_score', top_n=10)
    print_rankings(df, 'ewmac_score', top_n=10)

    # Create report
    create_momentum_report(df)

    print("\n" + "=" * 60)
    print("Done!")


if __name__ == '__main__':
    main()
