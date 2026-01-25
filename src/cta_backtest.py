#!/usr/bin/env python3
"""
CTA Trend Following Strategy Backtest

Backtests the 1-3-12 momentum strategy with monthly rebalancing
over historical data.
"""

import tushare as ts
import pandas as pd
import numpy as np
import os
from datetime import datetime, timedelta
import matplotlib.pyplot as plt
import matplotlib
import time

matplotlib.rcParams['font.sans-serif'] = ['SimHei', 'Arial Unicode MS', 'Microsoft YaHei']
matplotlib.rcParams['axes.unicode_minus'] = False

# Tushare token
TUSHARE_TOKEN = os.environ.get("TUSHARE_TOKEN", "a70287c82208760b640d7f08525b97181166b817e0d9ff5f8f244bc2")

# Universe (reduced for faster backtest)
UNIVERSE = {
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

    # Chemicals
    'TA': ('PTA', 'CZCE', 'Chemicals'),
    'MA': ('甲醇', 'CZCE', 'Chemicals'),
    'L': ('塑料', 'DCE', 'Chemicals'),
    'PP': ('聚丙烯', 'DCE', 'Chemicals'),

    # Energy
    'BU': ('沥青', 'SHFE', 'Energy'),
    'FU': ('燃料油', 'SHFE', 'Energy'),
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


def init_tushare():
    ts.set_token(TUSHARE_TOKEN)
    return ts.pro_api()


def download_all_data(pro, start_date, end_date):
    """
    Download all historical data needed for backtest.
    Returns dict: symbol -> DataFrame with daily prices
    """
    import sys
    print("Downloading historical data...", flush=True)

    all_data = {}
    exchanges = set(info[1] for info in UNIVERSE.values())

    # Download data by exchange - use date range query (faster)
    exchange_data = {}
    for exchange in exchanges:
        print(f"  Downloading {exchange} data...", flush=True)
        sys.stdout.flush()

        all_exchange_data = []

        # Download in chunks of ~60 days to avoid API limits
        current_end = datetime.strptime(end_date, '%Y%m%d')
        min_date = datetime.strptime(start_date, '%Y%m%d')

        while current_end >= min_date:
            chunk_start = max(current_end - timedelta(days=60), min_date)

            try:
                df = pro.fut_daily(
                    exchange=exchange,
                    start_date=chunk_start.strftime('%Y%m%d'),
                    end_date=current_end.strftime('%Y%m%d'),
                    fields='ts_code,trade_date,close,settle,vol,oi'
                )

                if df is not None and not df.empty:
                    all_exchange_data.append(df)
                    print(f"    {chunk_start.strftime('%Y%m%d')}-{current_end.strftime('%Y%m%d')}: {len(df)} records", flush=True)

            except Exception as e:
                print(f"    Error: {e}", flush=True)
                time.sleep(2)
                continue

            current_end = chunk_start - timedelta(days=1)
            time.sleep(0.3)  # Rate limit

        if all_exchange_data:
            exchange_data[exchange] = pd.concat(all_exchange_data, ignore_index=True)
            print(f"    Total {exchange}: {len(exchange_data[exchange])} records", flush=True)

    # Extract continuous series for each commodity
    for symbol, (name, exchange, sector) in UNIVERSE.items():
        if exchange not in exchange_data:
            continue

        df = exchange_data[exchange]

        # Filter by symbol
        if symbol == 'I':
            mask = df['ts_code'].str.match(r'^I\d{4}\.', case=False)
        else:
            mask = df['ts_code'].str.upper().str.startswith(symbol.upper())

        symbol_df = df[mask].copy()

        if symbol_df.empty:
            continue

        # Exclude index contracts
        symbol_df = symbol_df[~symbol_df['ts_code'].str.contains('L\\.', regex=True)]

        # For each date, use main contract (highest OI)
        continuous = []
        for date in symbol_df['trade_date'].unique():
            day_data = symbol_df[symbol_df['trade_date'] == date]
            if not day_data.empty and day_data['oi'].max() > 0:
                main = day_data.loc[day_data['oi'].idxmax()]
                continuous.append({
                    'trade_date': date,
                    'price': main['close'] if pd.notna(main['close']) else main['settle'],
                    'volume': main['vol'],
                    'oi': main['oi']
                })

        if continuous:
            result = pd.DataFrame(continuous).sort_values('trade_date')
            all_data[symbol] = result
            print(f"  {symbol}: {len(result)} days")

    return all_data


def get_month_ends(all_data, start_date, end_date):
    """Get all month-end dates from the data"""
    # Get all unique dates across all symbols
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


def calculate_signals_at_date(all_data, calc_date, risk_params):
    """
    Calculate signals for all commodities at a given date.
    Returns DataFrame with positions.
    """
    results = []

    for symbol, (name, exchange, sector) in UNIVERSE.items():
        if symbol not in all_data:
            continue

        df = all_data[symbol]
        df = df[df['trade_date'] <= calc_date].tail(300)

        if len(df) < 66:  # Need at least 3 months of data
            continue

        prices = df['price'].values
        n = len(prices)

        # Calculate returns
        R_1m = (prices[-1] / prices[-22] - 1) if n >= 22 else np.nan
        R_3m = (prices[-1] / prices[-66] - 1) if n >= 66 else np.nan
        R_12m = (prices[-1] / prices[-252] - 1) if n >= 252 else (prices[-1] / prices[0] - 1) if n >= 200 else np.nan

        # Direction score
        signs = []
        if not np.isnan(R_1m): signs.append(np.sign(R_1m))
        if not np.isnan(R_3m): signs.append(np.sign(R_3m))
        if not np.isnan(R_12m): signs.append(np.sign(R_12m))

        if len(signs) == 0:
            continue

        direction_score = np.mean(signs)

        # Volatility
        if n >= 60:
            returns = pd.Series(prices).pct_change().dropna().tail(60)
            volatility = returns.std() * np.sqrt(252)
        else:
            continue

        if volatility <= 0 or np.isnan(volatility):
            continue

        results.append({
            'symbol': symbol,
            'name': name,
            'exchange': exchange,
            'sector': sector,
            'price': prices[-1],
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


def run_backtest(all_data, month_ends, risk_params):
    """
    Run backtest with monthly rebalancing.
    Returns DataFrame with daily NAV and monthly returns.
    """
    print(f"\nRunning backtest over {len(month_ends)} months...")

    # Get all trading dates
    all_dates = set()
    for symbol, df in all_data.items():
        all_dates.update(df['trade_date'].tolist())
    all_dates = sorted(all_dates)

    # Initialize
    nav = 1.0
    peak_nav = 1.0
    current_weights = {}

    daily_navs = []
    monthly_returns = []

    # Build price lookup
    price_lookup = {}
    for symbol, df in all_data.items():
        price_lookup[symbol] = df.set_index('trade_date')['price'].to_dict()

    prev_month_end_idx = -1

    for i, date in enumerate(all_dates):
        if date < month_ends[0]:
            continue

        # Check if we need to rebalance (new month)
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

            signals_df = calculate_signals_at_date(all_data, rebal_date, effective_params)

            if signals_df is not None:
                current_weights = dict(zip(signals_df['symbol'], signals_df['weight']))

            prev_month_end_idx = current_month_end_idx

        # Calculate daily return
        daily_return = 0
        for symbol, weight in current_weights.items():
            if symbol not in price_lookup:
                continue

            # Get today and yesterday prices
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

        # Track monthly returns
        if date in month_ends:
            monthly_returns.append({
                'date': date,
                'nav': nav
            })

    return pd.DataFrame(daily_navs), pd.DataFrame(monthly_returns)


def calculate_stats(daily_df):
    """Calculate performance statistics"""
    if daily_df is None or daily_df.empty:
        return {}

    returns = daily_df['daily_return']
    nav = daily_df['nav']

    total_return = (nav.iloc[-1] / nav.iloc[0] - 1) * 100

    # Annualized return
    n_years = len(returns) / 252
    ann_return = ((nav.iloc[-1] / nav.iloc[0]) ** (1/n_years) - 1) * 100 if n_years > 0 else 0

    # Volatility
    ann_vol = returns.std() * np.sqrt(252) * 100

    # Sharpe (assuming 0 risk-free rate)
    sharpe = ann_return / ann_vol if ann_vol > 0 else 0

    # Max drawdown
    max_dd = daily_df['drawdown'].max() * 100

    # Calmar ratio
    calmar = ann_return / max_dd if max_dd > 0 else 0

    # Win rate
    win_rate = (returns > 0).sum() / len(returns) * 100 if len(returns) > 0 else 0

    return {
        'Total Return': f'{total_return:.1f}%',
        'Annualized Return': f'{ann_return:.1f}%',
        'Annualized Vol': f'{ann_vol:.1f}%',
        'Sharpe Ratio': f'{sharpe:.2f}',
        'Max Drawdown': f'{max_dd:.1f}%',
        'Calmar Ratio': f'{calmar:.2f}',
        'Win Rate (daily)': f'{win_rate:.1f}%',
    }


def plot_backtest(daily_df, save_path=None):
    """Plot backtest results"""
    if daily_df is None or daily_df.empty:
        return

    fig, axes = plt.subplots(3, 1, figsize=(14, 10), height_ratios=[2, 1, 1])

    dates = pd.to_datetime(daily_df['date'])

    # NAV curve
    ax1 = axes[0]
    ax1.plot(dates, daily_df['nav'], 'b-', linewidth=1.5, label='Strategy NAV')
    ax1.axhline(y=1, color='gray', linestyle='--', alpha=0.5)
    ax1.set_ylabel('NAV')
    ax1.set_title('CTA Trend Following Strategy - 3 Year Backtest', fontsize=14, fontweight='bold')
    ax1.legend(loc='upper left')
    ax1.grid(True, alpha=0.3)

    # Fill above/below 1
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
    """Run 3-year backtest"""
    pro = init_tushare()

    # 3 years of data (plus buffer for 12-month lookback)
    end_date = datetime.now().strftime('%Y%m%d')
    start_date = (datetime.now() - timedelta(days=365*4)).strftime('%Y%m%d')  # 4 years for lookback
    backtest_start = (datetime.now() - timedelta(days=365*3)).strftime('%Y%m%d')  # Actual backtest start

    print(f"Backtest period: {backtest_start} to {end_date}", flush=True)
    print(f"Data download: {start_date} to {end_date}", flush=True)

    # Download data
    all_data = download_all_data(pro, start_date, end_date)

    if not all_data:
        print("Failed to download data")
        return

    # Get month ends
    month_ends = get_month_ends(all_data, backtest_start, end_date)
    print(f"\nFound {len(month_ends)} month-end rebalancing dates")

    # Run backtest
    daily_df, monthly_df = run_backtest(all_data, month_ends, RISK_PARAMS)

    # Calculate stats
    stats = calculate_stats(daily_df)

    print("\n" + "=" * 60)
    print("BACKTEST RESULTS (3 Years)")
    print("=" * 60)
    for key, value in stats.items():
        print(f"  {key}: {value}")

    # Save results
    os.makedirs('output/charts', exist_ok=True)

    daily_df.to_csv('output/cta_backtest_daily.csv', index=False)
    print(f"\nSaved: output/cta_backtest_daily.csv")

    plot_backtest(daily_df, 'output/charts/cta_backtest.png')

    print("\n" + "=" * 60)
    print("Done!")


if __name__ == '__main__':
    main()
