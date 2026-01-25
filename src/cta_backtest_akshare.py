#!/usr/bin/env python3
"""
CTA Trend Following Strategy Backtest using AKShare

Uses AKShare for free historical commodity futures data (2009-present)
Backtests the 1-3-12 momentum strategy with monthly rebalancing.
"""

import akshare as ak
import pandas as pd
import numpy as np
import os
from datetime import datetime, timedelta
import matplotlib.pyplot as plt
import matplotlib
import time

matplotlib.rcParams['font.sans-serif'] = ['SimHei', 'Arial Unicode MS', 'Microsoft YaHei']
matplotlib.rcParams['axes.unicode_minus'] = False

# Universe: symbol -> (Chinese name, AKShare symbol, sector)
# Expanded to all available liquid commodity futures
UNIVERSE = {
    # === SHFE (Shanghai Futures Exchange) ===
    # Precious Metals
    'AU': ('黄金', 'AU0', 'Precious'),
    'AG': ('白银', 'AG0', 'Precious'),

    # Base Metals
    'CU': ('铜', 'CU0', 'Base Metals'),
    'AL': ('铝', 'AL0', 'Base Metals'),
    'ZN': ('锌', 'ZN0', 'Base Metals'),
    'NI': ('镍', 'NI0', 'Base Metals'),
    'PB': ('铅', 'PB0', 'Base Metals'),
    'SN': ('锡', 'SN0', 'Base Metals'),
    'SS': ('不锈钢', 'SS0', 'Base Metals'),
    'BC': ('国际铜', 'BC0', 'Base Metals'),
    'AO': ('氧化铝', 'AO0', 'Base Metals'),

    # Ferrous
    'RB': ('螺纹钢', 'RB0', 'Ferrous'),
    'HC': ('热卷', 'HC0', 'Ferrous'),
    'WR': ('线材', 'WR0', 'Ferrous'),

    # Energy - SHFE
    'BU': ('沥青', 'BU0', 'Energy'),
    'FU': ('燃料油', 'FU0', 'Energy'),
    'LU': ('低硫燃油', 'LU0', 'Energy'),
    'SC': ('原油', 'SC0', 'Energy'),
    'EC': ('集运指数', 'EC0', 'Energy'),

    # Rubber
    'RU': ('橡胶', 'RU0', 'Rubber'),
    'NR': ('20号胶', 'NR0', 'Rubber'),
    'BR': ('丁二烯橡胶', 'BR0', 'Rubber'),

    # Paper
    'SP': ('纸浆', 'SP0', 'Paper'),

    # === CZCE (Zhengzhou Commodity Exchange) ===
    # Agriculture - Grains & Oilseeds
    'CF': ('棉花', 'CF0', 'Agriculture'),
    'SR': ('白糖', 'SR0', 'Agriculture'),
    'OI': ('菜油', 'OI0', 'Agriculture'),
    'RM': ('菜粕', 'RM0', 'Agriculture'),
    'AP': ('苹果', 'AP0', 'Agriculture'),
    'CJ': ('红枣', 'CJ0', 'Agriculture'),
    'PK': ('花生', 'PK0', 'Agriculture'),
    'CY': ('棉纱', 'CY0', 'Agriculture'),

    # Chemicals - CZCE
    'TA': ('PTA', 'TA0', 'Chemicals'),
    'MA': ('甲醇', 'MA0', 'Chemicals'),
    'FG': ('玻璃', 'FG0', 'Chemicals'),
    'SA': ('纯碱', 'SA0', 'Chemicals'),
    'UR': ('尿素', 'UR0', 'Chemicals'),
    'PF': ('短纤', 'PF0', 'Chemicals'),
    'PX': ('对二甲苯', 'PX0', 'Chemicals'),
    'SH': ('烧碱', 'SH0', 'Chemicals'),

    # Building Materials
    'SF': ('硅铁', 'SF0', 'Ferroalloy'),
    'SM': ('锰硅', 'SM0', 'Ferroalloy'),

    # === GFEX (Guangzhou Futures Exchange) ===
    'SI': ('工业硅', 'SI0', 'Ferroalloy'),
    'LC': ('碳酸锂', 'LC0', 'Battery'),
}

# Risk parameters
RISK_PARAMS = {
    'target_vol': 0.10,
    'vol_lookback': 60,
    'sector_cap': 0.40,
    'single_market_cap': 0.15,
    'gross_leverage_cap': 2.0,
    'drawdown_throttle': 0.10,
    'throttle_reduction': 0.50,
    'no_trade_band': 1/3,
}


def download_all_data(use_weekly=True, years=10):
    """
    Download all historical data using AKShare get_futures_daily.
    Returns dict: symbol -> DataFrame with prices (daily or weekly)
    """
    freq = "weekly" if use_weekly else "daily"
    print(f"Downloading historical data from AKShare ({freq})...", flush=True)

    # Map symbols to their exchanges
    EXCHANGE_MAP = {
        # SHFE (Shanghai Futures Exchange)
        'AU': 'SHFE', 'AG': 'SHFE',  # Precious
        'CU': 'SHFE', 'AL': 'SHFE', 'ZN': 'SHFE', 'NI': 'SHFE', 'PB': 'SHFE',
        'SN': 'SHFE', 'SS': 'SHFE', 'BC': 'SHFE', 'AO': 'SHFE',  # Base Metals
        'RB': 'SHFE', 'HC': 'SHFE', 'WR': 'SHFE',  # Ferrous
        'BU': 'SHFE', 'FU': 'SHFE', 'LU': 'SHFE', 'SC': 'SHFE', 'EC': 'SHFE',  # Energy
        'RU': 'SHFE', 'NR': 'SHFE', 'BR': 'SHFE',  # Rubber
        'SP': 'SHFE',  # Paper

        # CZCE (Zhengzhou Commodity Exchange)
        'CF': 'CZCE', 'SR': 'CZCE', 'OI': 'CZCE', 'RM': 'CZCE', 'AP': 'CZCE',
        'CJ': 'CZCE', 'PK': 'CZCE', 'CY': 'CZCE',  # Agriculture
        'TA': 'CZCE', 'MA': 'CZCE', 'FG': 'CZCE', 'SA': 'CZCE', 'UR': 'CZCE',
        'PF': 'CZCE', 'PX': 'CZCE', 'SH': 'CZCE',  # Chemicals
        'SF': 'CZCE', 'SM': 'CZCE',  # Ferroalloy

        # GFEX (Guangzhou Futures Exchange)
        'SI': 'GFEX', 'LC': 'GFEX',
    }

    # Download by exchange to minimize API calls
    all_data = {}
    end_date = datetime.now().strftime('%Y%m%d')
    start_date = (datetime.now() - timedelta(days=365*years)).strftime('%Y%m%d')

    exchanges = set(EXCHANGE_MAP.values())

    for exchange in exchanges:
        print(f"\n  Downloading {exchange} data...", flush=True)
        try:
            # Download all data for this exchange
            df = ak.get_futures_daily(start_date=start_date, end_date=end_date, market=exchange)

            if df is not None and len(df) > 0:
                print(f"    Got {len(df)} rows", flush=True)

                # Process each symbol in this exchange
                for symbol, (name, ak_symbol, sector) in UNIVERSE.items():
                    if EXCHANGE_MAP.get(symbol) != exchange:
                        continue

                    # Filter for main contract (symbol + dominant month)
                    # Get the continuous data by finding rows where variety == symbol
                    symbol_df = df[df['variety'] == symbol].copy()

                    if len(symbol_df) == 0:
                        print(f"    {symbol}: No data", flush=True)
                        continue

                    # Group by date and use the most liquid contract (highest volume)
                    symbol_df = symbol_df.sort_values(['date', 'volume'], ascending=[True, False])
                    symbol_df = symbol_df.groupby('date').first().reset_index()

                    # Standardize columns
                    symbol_df = symbol_df.rename(columns={
                        'date': 'trade_date',
                        'close': 'price',
                    })
                    symbol_df['trade_date'] = pd.to_datetime(symbol_df['trade_date'].astype(str))

                    if use_weekly:
                        # Resample to weekly (Friday close)
                        symbol_df = symbol_df.set_index('trade_date')
                        weekly = symbol_df.resample('W-FRI').agg({
                            'price': 'last',
                            'volume': 'sum',
                        }).dropna()
                        weekly = weekly.reset_index()
                        weekly['trade_date'] = weekly['trade_date'].dt.strftime('%Y%m%d')
                        symbol_df = weekly
                    else:
                        symbol_df['trade_date'] = symbol_df['trade_date'].dt.strftime('%Y%m%d')

                    symbol_df = symbol_df.sort_values('trade_date')
                    all_data[symbol] = symbol_df
                    print(f"    {symbol} ({name}): {len(symbol_df)} {freq}", flush=True)

            time.sleep(0.5)  # Rate limiting

        except Exception as e:
            print(f"    Error downloading {exchange}: {e}", flush=True)

    return all_data


def get_month_ends(all_data, start_date, end_date):
    """Get all month-end dates from the data"""
    all_dates = set()
    for symbol, df in all_data.items():
        all_dates.update(df['trade_date'].tolist())

    all_dates = sorted(all_dates)
    all_dates = [d for d in all_dates if start_date <= d <= end_date]

    # Get month ends
    df = pd.DataFrame({'date': all_dates})
    df['date_dt'] = pd.to_datetime(df['date'])
    df['year_month'] = df['date_dt'].dt.to_period('M')
    month_ends = df.groupby('year_month')['date'].max().tolist()

    return month_ends


def calculate_signals_at_date(all_data, calc_date, risk_params, use_weekly=True):
    """Calculate signals for all commodities at a given date."""
    results = []

    # Weekly: 4, 13, 52 weeks for 1m, 3m, 12m
    # Daily: 22, 66, 252 days
    # Using 12-1 momentum: skip the most recent month to avoid short-term reversal
    if use_weekly:
        p1m, p3m, p12m = 4, 13, 52
        skip_period = 4  # Skip 1 month (4 weeks)
        vol_lookback = 12  # ~3 months of weekly data
    else:
        p1m, p3m, p12m = 22, 66, 252
        skip_period = 22  # Skip 1 month (22 days)
        vol_lookback = 60

    for symbol, (name, ak_symbol, sector) in UNIVERSE.items():
        if symbol not in all_data:
            continue

        df = all_data[symbol]
        df = df[df['trade_date'] <= calc_date].tail(100)

        if len(df) < p3m:  # Need at least 3 months
            continue

        prices = df['price'].values
        n = len(prices)

        # Calculate returns using 12-1 momentum (skip most recent month)
        # R_1m: return from t-1m to t (kept for short-term signal)
        R_1m = (prices[-1] / prices[-p1m] - 1) if n >= p1m else np.nan

        # R_3m: return from t-3m to t-1m (skip last month)
        if n >= p3m + skip_period:
            R_3m = (prices[-1-skip_period] / prices[-p3m-skip_period] - 1)
        elif n >= p3m:
            R_3m = (prices[-1] / prices[-p3m] - 1)  # Fallback if not enough data
        else:
            R_3m = np.nan

        # R_12m: return from t-12m to t-1m (skip last month) - classic 12-1 momentum
        if n >= p12m + skip_period:
            R_12m = (prices[-1-skip_period] / prices[-p12m-skip_period] - 1)
        elif n >= p12m:
            R_12m = (prices[-1] / prices[-p12m] - 1)  # Fallback if not enough data
        else:
            R_12m = np.nan

        # Direction score (using 12-1 momentum)
        signs = []
        if not np.isnan(R_1m): signs.append(np.sign(R_1m))
        if not np.isnan(R_3m): signs.append(np.sign(R_3m))
        if not np.isnan(R_12m): signs.append(np.sign(R_12m))

        if len(signs) == 0:
            continue

        direction_score = np.mean(signs)

        # Volatility (annualized)
        if n >= vol_lookback:
            returns = pd.Series(prices).pct_change().dropna().tail(vol_lookback)
            ann_factor = np.sqrt(52) if use_weekly else np.sqrt(252)
            volatility = returns.std() * ann_factor
        else:
            continue

        if volatility <= 0 or np.isnan(volatility):
            continue

        results.append({
            'symbol': symbol,
            'name': name,
            'sector': sector,
            'price': prices[-1],
            'R_1m': R_1m,
            'R_3m': R_3m,
            'R_12m': R_12m,
            'direction_score': direction_score,
            'volatility': volatility,
        })

    if not results:
        return None

    df = pd.DataFrame(results)

    # Calculate positions
    df['trade'] = df['direction_score'].abs() >= risk_params['no_trade_band']
    df['raw_weight'] = np.where(
        df['trade'] & (df['volatility'] > 0),
        df['direction_score'] / df['volatility'],
        0
    )

    # Normalize to target vol
    total_risk = (df['raw_weight'].abs() * df['volatility']).sum()
    if total_risk > 0:
        scale = risk_params['target_vol'] / total_risk
        df['weight'] = df['raw_weight'] * scale
    else:
        df['weight'] = 0

    # Apply caps
    single_cap = risk_params['single_market_cap']
    df['weight'] = df['weight'].clip(-single_cap, single_cap)

    # Sector cap
    for sector in df['sector'].unique():
        mask = df['sector'] == sector
        sector_risk = (df.loc[mask, 'weight'].abs() * df.loc[mask, 'volatility']).sum()
        if sector_risk > risk_params['sector_cap'] * risk_params['target_vol']:
            reduction = (risk_params['sector_cap'] * risk_params['target_vol']) / sector_risk
            df.loc[mask, 'weight'] *= reduction

    # Leverage cap
    gross = df['weight'].abs().sum()
    if gross > risk_params['gross_leverage_cap']:
        df['weight'] *= risk_params['gross_leverage_cap'] / gross

    return df


def run_backtest(all_data, month_ends, risk_params, use_weekly=True):
    """Run backtest with monthly rebalancing."""
    print(f"\nRunning backtest over {len(month_ends)} months...", flush=True)

    # Get all trading dates
    all_dates = set()
    for symbol, df in all_data.items():
        all_dates.update(df['trade_date'].tolist())
    all_dates = sorted(all_dates)

    # Filter to backtest period
    all_dates = [d for d in all_dates if d >= month_ends[0]]

    # Initialize
    nav = 1.0
    peak_nav = 1.0
    current_weights = {}

    daily_navs = []

    # Build price lookup
    price_lookup = {}
    for symbol, df in all_data.items():
        price_lookup[symbol] = df.set_index('trade_date')['price'].to_dict()

    prev_month_end_idx = -1

    for i, date in enumerate(all_dates):
        # Check if we need to rebalance
        current_month_end_idx = -1
        for j, me in enumerate(month_ends):
            if date > me:
                current_month_end_idx = j

        if current_month_end_idx > prev_month_end_idx and current_month_end_idx >= 0:
            # Rebalance
            rebal_date = month_ends[current_month_end_idx]

            # Apply drawdown throttle
            effective_params = risk_params.copy()
            drawdown = (peak_nav - nav) / peak_nav if peak_nav > 0 else 0
            if drawdown > risk_params['drawdown_throttle']:
                effective_params['target_vol'] = risk_params['target_vol'] * (1 - risk_params['throttle_reduction'])

            signals_df = calculate_signals_at_date(all_data, rebal_date, effective_params, use_weekly)

            if signals_df is not None:
                current_weights = dict(zip(signals_df['symbol'], signals_df['weight']))

            prev_month_end_idx = current_month_end_idx

        # Calculate daily return
        daily_return = 0
        for symbol, weight in current_weights.items():
            if symbol not in price_lookup:
                continue

            prev_date = all_dates[i-1] if i > 0 else None

            if prev_date and prev_date in price_lookup[symbol] and date in price_lookup[symbol]:
                price_today = price_lookup[symbol][date]
                price_prev = price_lookup[symbol][prev_date]

                if price_prev > 0:
                    ret = (price_today / price_prev - 1)
                    daily_return += weight * ret

        # Update NAV
        nav *= (1 + daily_return)
        peak_nav = max(peak_nav, nav)

        daily_navs.append({
            'date': date,
            'nav': nav,
            'daily_return': daily_return,
            'drawdown': (peak_nav - nav) / peak_nav
        })

    return pd.DataFrame(daily_navs)


def calculate_stats(daily_df, use_weekly=True):
    """Calculate performance statistics"""
    if daily_df is None or daily_df.empty:
        return {}

    returns = daily_df['daily_return']
    nav = daily_df['nav']

    total_return = (nav.iloc[-1] / nav.iloc[0] - 1) * 100

    # Annualized (52 weeks or 252 days per year)
    periods_per_year = 52 if use_weekly else 252
    n_years = len(returns) / periods_per_year
    ann_return = ((nav.iloc[-1] / nav.iloc[0]) ** (1/n_years) - 1) * 100 if n_years > 0 else 0
    ann_vol = returns.std() * np.sqrt(periods_per_year) * 100
    sharpe = ann_return / ann_vol if ann_vol > 0 else 0

    # Max drawdown
    max_dd = daily_df['drawdown'].max() * 100
    calmar = ann_return / max_dd if max_dd > 0 else 0

    # Win rate
    win_rate = (returns > 0).sum() / len(returns) * 100 if len(returns) > 0 else 0

    # Monthly stats
    daily_df_copy = daily_df.copy()
    daily_df_copy['date'] = pd.to_datetime(daily_df_copy['date'])
    monthly_rets = daily_df_copy.set_index('date')['daily_return'].resample('ME').sum()
    positive_months = (monthly_rets > 0).sum()
    total_months = len(monthly_rets)

    return {
        'Total Return': f'{total_return:.1f}%',
        'Annualized Return': f'{ann_return:.1f}%',
        'Annualized Vol': f'{ann_vol:.1f}%',
        'Sharpe Ratio': f'{sharpe:.2f}',
        'Max Drawdown': f'{max_dd:.1f}%',
        'Calmar Ratio': f'{calmar:.2f}',
        'Win Rate (daily)': f'{win_rate:.1f}%',
        'Positive Months': f'{positive_months}/{total_months} ({positive_months/total_months*100:.0f}%)',
        'Trading Days': f'{len(daily_df)}',
        'Years': f'{n_years:.1f}',
    }


def calculate_yearly_stats(daily_df, use_weekly=True):
    """Calculate performance statistics by year"""
    if daily_df is None or daily_df.empty:
        return {}

    periods_per_year = 52 if use_weekly else 252

    df = daily_df.copy()
    df['date'] = pd.to_datetime(df['date'])
    df['year'] = df['date'].dt.year

    yearly_stats = {}

    for year in sorted(df['year'].unique()):
        year_df = df[df['year'] == year].copy()

        if len(year_df) < 4:  # Need at least a few data points
            continue

        returns = year_df['daily_return']
        nav_start = year_df['nav'].iloc[0] / (1 + year_df['daily_return'].iloc[0])
        nav_end = year_df['nav'].iloc[-1]

        year_return = (nav_end / nav_start - 1) * 100
        year_vol = returns.std() * np.sqrt(periods_per_year) * 100
        year_sharpe = year_return / year_vol if year_vol > 0 else 0

        # Max drawdown for this year
        peak = year_df['nav'].expanding().max()
        drawdown = (peak - year_df['nav']) / peak
        max_dd = drawdown.max() * 100

        # Count positive/negative weeks
        pos_weeks = (returns > 0).sum()
        neg_weeks = (returns <= 0).sum()

        yearly_stats[year] = {
            'return': year_return,
            'vol': year_vol,
            'sharpe': year_sharpe,
            'max_dd': max_dd,
            'pos_weeks': pos_weeks,
            'neg_weeks': neg_weeks,
            'win_rate': pos_weeks / len(returns) * 100 if len(returns) > 0 else 0
        }

    return yearly_stats


def plot_backtest(daily_df, title_suffix="", save_path=None):
    """Plot backtest results"""
    if daily_df is None or daily_df.empty:
        return

    fig, axes = plt.subplots(3, 1, figsize=(14, 10), height_ratios=[2, 1, 1])

    dates = pd.to_datetime(daily_df['date'])

    # NAV curve
    ax1 = axes[0]
    ax1.plot(dates, daily_df['nav'], 'b-', linewidth=1.2, label='Strategy NAV')
    ax1.axhline(y=1, color='gray', linestyle='--', alpha=0.5)
    ax1.set_ylabel('NAV')
    ax1.set_title(f'CTA Trend Following Strategy Backtest {title_suffix}', fontsize=14, fontweight='bold')
    ax1.legend(loc='upper left')
    ax1.grid(True, alpha=0.3)

    ax1.fill_between(dates, 1, daily_df['nav'], where=daily_df['nav'] >= 1,
                     color='green', alpha=0.3)
    ax1.fill_between(dates, 1, daily_df['nav'], where=daily_df['nav'] < 1,
                     color='red', alpha=0.3)

    # Drawdown
    ax2 = axes[1]
    ax2.fill_between(dates, 0, -daily_df['drawdown'] * 100, color='red', alpha=0.5)
    ax2.set_ylabel('Drawdown (%)')
    ax2.set_ylim(ax2.get_ylim()[0], 5)
    ax2.grid(True, alpha=0.3)

    # Monthly returns
    ax3 = axes[2]
    temp_df = daily_df.copy()
    temp_df['date'] = pd.to_datetime(temp_df['date'])
    monthly_rets = temp_df.set_index('date')['daily_return'].resample('ME').sum() * 100
    colors = ['green' if r >= 0 else 'red' for r in monthly_rets]
    ax3.bar(monthly_rets.index, monthly_rets.values, width=20, color=colors, alpha=0.7)
    ax3.axhline(y=0, color='black', linewidth=0.5)
    ax3.set_ylabel('Monthly Return (%)')
    ax3.set_xlabel('Date')
    ax3.grid(True, alpha=0.3)

    plt.tight_layout()

    if save_path:
        plt.savefig(save_path, dpi=150, bbox_inches='tight')
        print(f"Saved: {save_path}")

    plt.close()


def main():
    """Run backtest"""
    import sys

    # Parse years from command line (default 3)
    years = 3
    if len(sys.argv) > 1:
        years = int(sys.argv[1])

    print(f"=" * 60)
    print(f"CTA Trend Following Backtest - {years} Years")
    print(f"=" * 60)

    # Use weekly data for faster backtest
    use_weekly = True

    # Download data
    all_data = download_all_data(use_weekly=use_weekly, years=years)

    if not all_data:
        print("Failed to download data")
        return

    print(f"\nLoaded data for {len(all_data)} commodities")

    # Set backtest period
    end_date = datetime.now().strftime('%Y%m%d')
    start_date = (datetime.now() - timedelta(days=365*years)).strftime('%Y%m%d')

    print(f"Backtest period: {start_date} to {end_date}")

    # Get month ends
    month_ends = get_month_ends(all_data, start_date, end_date)
    print(f"Found {len(month_ends)} month-end rebalancing dates")

    # Run backtest
    daily_df = run_backtest(all_data, month_ends, RISK_PARAMS, use_weekly)

    # Calculate stats
    stats = calculate_stats(daily_df, use_weekly)
    yearly_stats = calculate_yearly_stats(daily_df, use_weekly)

    print("\n" + "=" * 60)
    print(f"BACKTEST RESULTS ({years} Years)")
    print("=" * 60)
    for key, value in stats.items():
        print(f"  {key}: {value}")

    # Print yearly breakdown
    print("\n" + "=" * 60)
    print("PERFORMANCE BY YEAR")
    print("=" * 60)
    print(f"{'Year':<8} {'Return':<10} {'Vol':<10} {'Sharpe':<10} {'MaxDD':<10} {'WinRate':<10}")
    print("-" * 60)
    for year, ys in sorted(yearly_stats.items()):
        print(f"{year:<8} {ys['return']:>+7.1f}%  {ys['vol']:>7.1f}%  {ys['sharpe']:>7.2f}   {ys['max_dd']:>7.1f}%  {ys['win_rate']:>7.1f}%")

    # Save results
    os.makedirs('output/charts', exist_ok=True)

    daily_df.to_csv(f'output/cta_backtest_{years}y_daily.csv', index=False)
    print(f"\nSaved: output/cta_backtest_{years}y_daily.csv")

    plot_backtest(daily_df, f"({years} Years)", f'output/charts/cta_backtest_{years}y.png')

    print("\n" + "=" * 60)
    print("Done!")

    return daily_df, stats


if __name__ == '__main__':
    main()
