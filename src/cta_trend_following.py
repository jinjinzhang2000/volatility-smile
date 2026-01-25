#!/usr/bin/env python3
"""
CTA-Style Trend Following Strategy for Chinese Commodity Futures

Implements a classic 1-3-12 month momentum strategy with:
- Direction score based on sign(R_1m) + sign(R_3m) + sign(R_12m)
- Volatility-targeting position sizing
- Sector and single-market risk caps
- Drawdown throttle for risk management
- Monthly rebalancing (end-of-day)

Supports all major Chinese futures exchanges:
- SHFE, DCE, CZCE, INE
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

# Universe: Liquid commodities by sector
# Format: symbol -> (name, exchange, sector)
UNIVERSE = {
    # Energy
    'SC': ('原油', 'INE', 'Energy'),
    'FU': ('燃料油', 'SHFE', 'Energy'),
    'LU': ('低硫燃料油', 'INE', 'Energy'),
    'BU': ('沥青', 'SHFE', 'Energy'),
    'PG': ('液化石油气', 'DCE', 'Energy'),

    # Metals
    'AU': ('黄金', 'SHFE', 'Metals'),
    'AG': ('白银', 'SHFE', 'Metals'),
    'CU': ('铜', 'SHFE', 'Metals'),
    'AL': ('铝', 'SHFE', 'Metals'),
    'ZN': ('锌', 'SHFE', 'Metals'),
    'NI': ('镍', 'SHFE', 'Metals'),

    # Ferrous
    'RB': ('螺纹钢', 'SHFE', 'Ferrous'),
    'HC': ('热卷', 'SHFE', 'Ferrous'),
    'I': ('铁矿石', 'DCE', 'Ferrous'),
    'J': ('焦炭', 'DCE', 'Ferrous'),
    'JM': ('焦煤', 'DCE', 'Ferrous'),

    # Agriculture
    'M': ('豆粕', 'DCE', 'Agriculture'),
    'Y': ('豆油', 'DCE', 'Agriculture'),
    'P': ('棕榈油', 'DCE', 'Agriculture'),
    'C': ('玉米', 'DCE', 'Agriculture'),
    'CF': ('棉花', 'CZCE', 'Agriculture'),
    'SR': ('白糖', 'CZCE', 'Agriculture'),
    'AP': ('苹果', 'CZCE', 'Agriculture'),

    # Chemicals
    'TA': ('PTA', 'CZCE', 'Chemicals'),
    'MA': ('甲醇', 'CZCE', 'Chemicals'),
    'L': ('塑料', 'DCE', 'Chemicals'),
    'PP': ('聚丙烯', 'DCE', 'Chemicals'),
    'EG': ('乙二醇', 'DCE', 'Chemicals'),
    'SA': ('纯碱', 'CZCE', 'Chemicals'),

    # Livestock
    'LH': ('生猪', 'DCE', 'Livestock'),
    'JD': ('鸡蛋', 'DCE', 'Livestock'),
}

# Risk parameters
RISK_PARAMS = {
    'target_vol': 0.10,           # 10% annualized target volatility
    'vol_lookback': 60,           # Days for volatility calculation
    'sector_cap': 0.40,           # Max 40% risk per sector
    'single_market_cap': 0.15,    # Max 15% risk per commodity
    'gross_leverage_cap': 2.0,    # Max 2x notional exposure
    'drawdown_throttle': 0.10,    # Cut risk at 10% drawdown
    'throttle_reduction': 0.50,   # Reduce target vol by 50% when throttled
    'no_trade_band': 1/3,         # Skip weak signals |S| < 1/3
}


def init_tushare():
    """Initialize Tushare API"""
    ts.set_token(TUSHARE_TOKEN)
    return ts.pro_api()


def get_trading_days(pro, start_date, end_date):
    """Get list of trading days"""
    df = pro.trade_cal(
        exchange='SHFE',
        start_date=start_date,
        end_date=end_date,
        is_open='1'
    )
    return sorted(df['cal_date'].tolist())


def get_month_end_dates(trading_days):
    """Get month-end trading dates"""
    df = pd.DataFrame({'date': trading_days})
    df['date'] = pd.to_datetime(df['date'])
    df['year_month'] = df['date'].dt.to_period('M')
    month_ends = df.groupby('year_month')['date'].max()
    return [d.strftime('%Y%m%d') for d in month_ends]


def download_exchange_data(pro, exchange, start_date, end_date):
    """Download all futures data for an exchange with caching"""
    import time

    cache_key = f"{exchange}_{start_date}_{end_date}"

    if not hasattr(download_exchange_data, 'cache'):
        download_exchange_data.cache = {}

    if cache_key in download_exchange_data.cache:
        return download_exchange_data.cache[cache_key]

    try:
        all_data = []
        current_date = datetime.strptime(end_date, '%Y%m%d')
        min_date = datetime.strptime(start_date, '%Y%m%d')

        days_fetched = 0
        while current_date >= min_date and days_fetched < 300:
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
            time.sleep(0.08)

        if not all_data:
            return None

        result = pd.concat(all_data, ignore_index=True)
        download_exchange_data.cache[cache_key] = result
        return result

    except Exception as e:
        print(f"  Error downloading {exchange} data: {e}")
        return None


def get_continuous_prices(pro, symbol, exchange, end_date, days=300):
    """Get continuous price series using main contracts (highest OI)"""
    start_date = (datetime.strptime(end_date, '%Y%m%d') - timedelta(days=days * 2)).strftime('%Y%m%d')

    df = download_exchange_data(pro, exchange, start_date, end_date)

    if df is None or df.empty:
        return None

    # Filter by symbol
    if symbol.upper() == 'I':
        mask = df['ts_code'].str.match(r'^I\d{4}\.', case=False)
    else:
        mask = df['ts_code'].str.upper().str.startswith(symbol.upper())

    df = df[mask]

    if df.empty:
        return None

    # Exclude index contracts
    df = df[~df['ts_code'].str.contains('L\\.', regex=True)]

    # For each date, use contract with highest OI
    continuous_data = []
    for date in sorted(df['trade_date'].unique()):
        day_data = df[df['trade_date'] == date]
        if not day_data.empty and day_data['oi'].max() > 0:
            main = day_data.loc[day_data['oi'].idxmax()]
            continuous_data.append({
                'trade_date': date,
                'close': main['close'],
                'settle': main['settle'],
                'volume': main['vol'],
                'oi': main['oi']
            })

    if not continuous_data:
        return None

    result = pd.DataFrame(continuous_data)
    result = result.sort_values('trade_date').tail(days)
    result['price'] = result['close'].fillna(result['settle'])
    return result


def calculate_momentum_returns(prices_df):
    """
    Calculate 1-month, 3-month, and 12-month returns

    Returns dict with R_1m, R_3m, R_12m
    """
    if prices_df is None or len(prices_df) < 22:
        return None

    prices = prices_df['price'].values
    n = len(prices)

    returns = {}

    # 1-month return (~22 trading days)
    if n >= 22:
        returns['R_1m'] = (prices[-1] / prices[-22] - 1)
    else:
        returns['R_1m'] = np.nan

    # 3-month return (~66 trading days)
    if n >= 66:
        returns['R_3m'] = (prices[-1] / prices[-66] - 1)
    else:
        returns['R_3m'] = np.nan

    # 12-month return (~252 trading days)
    if n >= 252:
        returns['R_12m'] = (prices[-1] / prices[-252] - 1)
    elif n >= 200:
        # Use available data if close to 12 months
        returns['R_12m'] = (prices[-1] / prices[0] - 1)
    else:
        returns['R_12m'] = np.nan

    return returns


def calculate_direction_score(returns):
    """
    Calculate direction score S_i = (sign(R_1m) + sign(R_3m) + sign(R_12m)) / 3

    S_i ∈ {-1, -1/3, +1/3, +1}
    """
    if returns is None:
        return np.nan

    signs = []
    for key in ['R_1m', 'R_3m', 'R_12m']:
        if key in returns and not np.isnan(returns[key]):
            signs.append(np.sign(returns[key]))

    if len(signs) == 0:
        return np.nan

    return np.mean(signs)


def calculate_volatility(prices_df, lookback=60):
    """
    Calculate annualized volatility from daily returns
    """
    if prices_df is None or len(prices_df) < lookback:
        return np.nan

    prices = prices_df['price'].tail(lookback + 1)
    returns = prices.pct_change().dropna()

    if len(returns) < lookback // 2:
        return np.nan

    # Annualize: daily std * sqrt(252)
    return returns.std() * np.sqrt(252)


def calculate_positions(signals_df, risk_params):
    """
    Calculate position weights with risk controls:
    - Volatility targeting
    - Sector caps
    - Single market caps
    - Gross leverage cap
    """
    df = signals_df.copy()

    # Filter out weak signals (no-trade band)
    df['trade'] = df['direction_score'].abs() >= risk_params['no_trade_band']

    # Raw weight: S_i / sigma_i
    df['raw_weight'] = np.where(
        df['trade'] & (df['volatility'] > 0),
        df['direction_score'] / df['volatility'],
        0
    )

    # Normalize to target volatility
    # Portfolio vol ≈ sum of |w_i| * sigma_i (simplified, ignoring correlations)
    total_risk = (df['raw_weight'].abs() * df['volatility']).sum()

    if total_risk > 0:
        scale_factor = risk_params['target_vol'] / total_risk
        df['scaled_weight'] = df['raw_weight'] * scale_factor
    else:
        df['scaled_weight'] = 0

    # Apply single market cap
    single_cap = risk_params['single_market_cap']
    df['capped_weight'] = df['scaled_weight'].clip(-single_cap, single_cap)

    # Apply sector caps
    sector_cap = risk_params['sector_cap']
    for sector in df['sector'].unique():
        sector_mask = df['sector'] == sector
        sector_risk = (df.loc[sector_mask, 'capped_weight'].abs() * df.loc[sector_mask, 'volatility']).sum()

        if sector_risk > sector_cap * risk_params['target_vol']:
            reduction = (sector_cap * risk_params['target_vol']) / sector_risk
            df.loc[sector_mask, 'capped_weight'] *= reduction

    # Apply gross leverage cap
    gross_exposure = df['capped_weight'].abs().sum()
    if gross_exposure > risk_params['gross_leverage_cap']:
        df['capped_weight'] *= risk_params['gross_leverage_cap'] / gross_exposure

    df['final_weight'] = df['capped_weight']

    return df


def apply_drawdown_throttle(current_nav, peak_nav, risk_params):
    """
    Reduce target volatility if in drawdown
    """
    drawdown = (peak_nav - current_nav) / peak_nav

    if drawdown > risk_params['drawdown_throttle']:
        return risk_params['target_vol'] * (1 - risk_params['throttle_reduction'])
    return risk_params['target_vol']


def generate_signals(pro, rebalance_date, risk_params=RISK_PARAMS):
    """
    Generate trading signals for all commodities on a given date
    """
    print(f"\n{'='*60}")
    print(f"Generating signals for {rebalance_date}")
    print(f"{'='*60}")

    results = []

    for symbol, (name, exchange, sector) in UNIVERSE.items():
        print(f"Processing {symbol} ({name})...")

        # Get price data (need ~300 days for 12-month returns)
        prices_df = get_continuous_prices(pro, symbol, exchange, rebalance_date, days=300)

        if prices_df is None or len(prices_df) < 22:
            print(f"  Insufficient data")
            continue

        # Calculate momentum returns
        returns = calculate_momentum_returns(prices_df)

        # Calculate direction score
        direction_score = calculate_direction_score(returns)

        # Calculate volatility
        volatility = calculate_volatility(prices_df, risk_params['vol_lookback'])

        if np.isnan(direction_score) or np.isnan(volatility):
            print(f"  Missing metrics")
            continue

        result = {
            'symbol': symbol,
            'name': name,
            'exchange': exchange,
            'sector': sector,
            'last_price': prices_df['price'].iloc[-1],
            'R_1m': returns.get('R_1m', np.nan) * 100,
            'R_3m': returns.get('R_3m', np.nan) * 100,
            'R_12m': returns.get('R_12m', np.nan) * 100 if returns.get('R_12m') else np.nan,
            'direction_score': direction_score,
            'volatility': volatility,
            'data_points': len(prices_df)
        }

        results.append(result)
        print(f"  OK - S={direction_score:+.2f}, vol={volatility:.1%}")

    if not results:
        return None

    df = pd.DataFrame(results)

    # Calculate positions
    df = calculate_positions(df, risk_params)

    return df


def print_portfolio(df):
    """Print portfolio summary"""
    if df is None or df.empty:
        print("No positions")
        return

    print(f"\n{'='*60}")
    print("PORTFOLIO POSITIONS")
    print(f"{'='*60}")

    # Sort by weight
    sorted_df = df[df['final_weight'] != 0].sort_values('final_weight', ascending=False)

    print("\nLONG POSITIONS:")
    print("-" * 60)
    longs = sorted_df[sorted_df['final_weight'] > 0]
    for _, row in longs.iterrows():
        print(f"  {row['symbol']:4s} {row['name']:8s} | Weight: {row['final_weight']:+6.1%} | "
              f"S: {row['direction_score']:+.2f} | Vol: {row['volatility']:.1%}")

    print("\nSHORT POSITIONS:")
    print("-" * 60)
    shorts = sorted_df[sorted_df['final_weight'] < 0]
    for _, row in shorts.iterrows():
        print(f"  {row['symbol']:4s} {row['name']:8s} | Weight: {row['final_weight']:+6.1%} | "
              f"S: {row['direction_score']:+.2f} | Vol: {row['volatility']:.1%}")

    print("\nNO POSITION (weak signals):")
    print("-" * 60)
    flat = df[df['final_weight'] == 0]
    for _, row in flat.iterrows():
        print(f"  {row['symbol']:4s} {row['name']:8s} | S: {row['direction_score']:+.2f}")

    # Summary stats
    print(f"\n{'='*60}")
    print("PORTFOLIO SUMMARY")
    print(f"{'='*60}")

    gross = df['final_weight'].abs().sum()
    net = df['final_weight'].sum()
    long_exposure = df[df['final_weight'] > 0]['final_weight'].sum()
    short_exposure = df[df['final_weight'] < 0]['final_weight'].sum()

    print(f"  Gross Exposure: {gross:.1%}")
    print(f"  Net Exposure:   {net:+.1%}")
    print(f"  Long Exposure:  {long_exposure:+.1%}")
    print(f"  Short Exposure: {short_exposure:+.1%}")

    # Sector breakdown
    print(f"\nSECTOR EXPOSURE:")
    print("-" * 40)
    sector_exp = df.groupby('sector')['final_weight'].agg(['sum', lambda x: x.abs().sum()])
    sector_exp.columns = ['Net', 'Gross']
    for sector, row in sector_exp.iterrows():
        print(f"  {sector:12s} | Net: {row['Net']:+6.1%} | Gross: {row['Gross']:6.1%}")


def plot_signals(df, save_path=None):
    """Create visualization of signals and positions"""
    if df is None or df.empty:
        return

    fig, axes = plt.subplots(2, 2, figsize=(16, 12))

    # 1. Direction Score Bar Chart
    ax1 = axes[0, 0]
    sorted_df = df.sort_values('direction_score')
    colors = ['#d73027' if s < 0 else '#1a9850' for s in sorted_df['direction_score']]
    labels = [f"{row['symbol']} {row['name']}" for _, row in sorted_df.iterrows()]

    ax1.barh(range(len(labels)), sorted_df['direction_score'], color=colors, height=0.7)
    ax1.set_yticks(range(len(labels)))
    ax1.set_yticklabels(labels, fontsize=8)
    ax1.axvline(x=0, color='black', linewidth=0.8)
    ax1.axvline(x=1/3, color='gray', linewidth=0.5, linestyle='--', alpha=0.5)
    ax1.axvline(x=-1/3, color='gray', linewidth=0.5, linestyle='--', alpha=0.5)
    ax1.set_xlabel('Direction Score (S)')
    ax1.set_title('1-3-12 Momentum Direction Score', fontweight='bold')

    # 2. Position Weights Bar Chart
    ax2 = axes[0, 1]
    sorted_df2 = df.sort_values('final_weight')
    colors2 = ['#d73027' if w < 0 else '#1a9850' for w in sorted_df2['final_weight']]
    labels2 = [f"{row['symbol']} {row['name']}" for _, row in sorted_df2.iterrows()]

    ax2.barh(range(len(labels2)), sorted_df2['final_weight'] * 100, color=colors2, height=0.7)
    ax2.set_yticks(range(len(labels2)))
    ax2.set_yticklabels(labels2, fontsize=8)
    ax2.axvline(x=0, color='black', linewidth=0.8)
    ax2.set_xlabel('Position Weight (%)')
    ax2.set_title('Risk-Adjusted Position Weights', fontweight='bold')

    # 3. Momentum Returns Heatmap
    ax3 = axes[1, 0]
    heatmap_df = df.sort_values('direction_score', ascending=False)
    heatmap_data = heatmap_df[['R_1m', 'R_3m', 'R_12m']].values
    heatmap_labels = [f"{row['symbol']} {row['name']}" for _, row in heatmap_df.iterrows()]

    vmax = min(50, np.nanmax(np.abs(heatmap_data)))
    im = ax3.imshow(heatmap_data, cmap='RdYlGn', aspect='auto', vmin=-vmax, vmax=vmax)

    ax3.set_yticks(range(len(heatmap_labels)))
    ax3.set_yticklabels(heatmap_labels, fontsize=8)
    ax3.set_xticks([0, 1, 2])
    ax3.set_xticklabels(['1个月', '3个月', '12个月'], fontsize=10)

    cbar = plt.colorbar(im, ax=ax3, shrink=0.8)
    cbar.set_label('收益率 (%)')

    for i in range(len(heatmap_labels)):
        for j in range(3):
            val = heatmap_data[i, j]
            if not np.isnan(val):
                color = 'white' if abs(val) > vmax * 0.5 else 'black'
                ax3.text(j, i, f'{val:.1f}', ha='center', va='center', fontsize=7, color=color)

    ax3.set_title('1-3-12 月收益率', fontweight='bold')

    # 4. Sector Exposure Pie Chart
    ax4 = axes[1, 1]
    sector_exp = df.groupby('sector')['final_weight'].apply(lambda x: x.abs().sum())
    sector_exp = sector_exp[sector_exp > 0]

    if len(sector_exp) > 0:
        colors_pie = plt.cm.Set3(np.linspace(0, 1, len(sector_exp)))
        wedges, texts, autotexts = ax4.pie(
            sector_exp.values,
            labels=sector_exp.index,
            autopct='%1.1f%%',
            colors=colors_pie,
            startangle=90
        )
        ax4.set_title('Sector Gross Exposure', fontweight='bold')
    else:
        ax4.text(0.5, 0.5, 'No Positions', ha='center', va='center', fontsize=14)
        ax4.set_title('Sector Exposure', fontweight='bold')

    plt.suptitle(f'CTA Trend Following Signals - {datetime.now().strftime("%Y-%m-%d")}',
                 fontsize=16, fontweight='bold', y=1.02)

    plt.tight_layout()

    if save_path:
        plt.savefig(save_path, dpi=150, bbox_inches='tight')
        print(f"Saved: {save_path}")

    plt.close()


def create_report(df, output_dir='output'):
    """Create full report with CSV and charts"""
    os.makedirs(output_dir, exist_ok=True)
    os.makedirs(os.path.join(output_dir, 'charts'), exist_ok=True)

    today = datetime.now().strftime('%Y%m%d')

    # Save CSV
    csv_path = os.path.join(output_dir, f'cta_signals_{today}.csv')
    df.to_csv(csv_path, index=False, encoding='utf-8-sig')
    print(f"Saved: {csv_path}")

    # Save chart
    chart_path = os.path.join(output_dir, 'charts', 'cta_trend_signals.png')
    plot_signals(df, chart_path)

    return csv_path


def main():
    """Main entry point"""
    import sys

    # Parse arguments
    trade_date = None
    if len(sys.argv) > 1:
        trade_date = sys.argv[1]

    if trade_date is None:
        trade_date = datetime.now().strftime('%Y%m%d')

    pro = init_tushare()

    # Generate signals
    df = generate_signals(pro, trade_date)

    if df is None or df.empty:
        print("Failed to generate signals")
        return

    # Print portfolio
    print_portfolio(df)

    # Create report
    create_report(df)

    print("\n" + "=" * 60)
    print("Done!")


if __name__ == '__main__':
    main()
