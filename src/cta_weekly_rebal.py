#!/usr/bin/env python3
"""
CTA MA Strategy with Weekly Rebalancing and Transaction Costs
==============================================================

Comparing monthly vs weekly rebalancing with realistic transaction costs.

Transaction Cost Assumptions:
- Commission: 0.01% (1bp) per trade
- Slippage: 0.02% (2bp) per trade
- Total one-way cost: 0.03% (3bp)
- Round-trip cost: 0.06% (6bp)

For Chinese commodity futures, typical costs are:
- Exchange fee: ~0.5-2bp
- Broker fee: ~0.5-1bp
- Slippage: ~1-3bp depending on liquidity
"""

import pandas as pd
import numpy as np
import akshare as ak
from datetime import datetime, timedelta
import warnings
warnings.filterwarnings('ignore')

# =============================================================================
# COMMODITY UNIVERSE
# =============================================================================

COMMODITY_UNIVERSE = {
    # 贵金属 (Precious)
    'AU': {'name': '黄金', 'exchange': 'SHFE', 'sector': 'precious'},
    'AG': {'name': '白银', 'exchange': 'SHFE', 'sector': 'precious'},

    # 有色金属 (Base Metals)
    'CU': {'name': '铜', 'exchange': 'SHFE', 'sector': 'base_metal'},
    'AL': {'name': '铝', 'exchange': 'SHFE', 'sector': 'base_metal'},
    'ZN': {'name': '锌', 'exchange': 'SHFE', 'sector': 'base_metal'},
    'NI': {'name': '镍', 'exchange': 'SHFE', 'sector': 'base_metal'},
    'PB': {'name': '铅', 'exchange': 'SHFE', 'sector': 'base_metal'},
    'SN': {'name': '锡', 'exchange': 'SHFE', 'sector': 'base_metal'},

    # 黑色 (Ferrous)
    'RB': {'name': '螺纹钢', 'exchange': 'SHFE', 'sector': 'ferrous'},
    'HC': {'name': '热卷', 'exchange': 'SHFE', 'sector': 'ferrous'},

    # 能源 (Energy - SHFE)
    'BU': {'name': '沥青', 'exchange': 'SHFE', 'sector': 'energy'},
    'FU': {'name': '燃料油', 'exchange': 'SHFE', 'sector': 'energy'},
    'SC': {'name': '原油', 'exchange': 'SHFE', 'sector': 'energy'},

    # 橡胶
    'RU': {'name': '橡胶', 'exchange': 'SHFE', 'sector': 'rubber'},
    'SP': {'name': '纸浆', 'exchange': 'SHFE', 'sector': 'paper'},

    # 农产品 CZCE
    'CF': {'name': '棉花', 'exchange': 'CZCE', 'sector': 'agriculture'},
    'SR': {'name': '白糖', 'exchange': 'CZCE', 'sector': 'agriculture'},
    'OI': {'name': '菜油', 'exchange': 'CZCE', 'sector': 'agriculture'},
    'RM': {'name': '菜粕', 'exchange': 'CZCE', 'sector': 'agriculture'},
    'AP': {'name': '苹果', 'exchange': 'CZCE', 'sector': 'agriculture'},

    # 化工 CZCE
    'TA': {'name': 'PTA', 'exchange': 'CZCE', 'sector': 'chemicals'},
    'MA': {'name': '甲醇', 'exchange': 'CZCE', 'sector': 'chemicals'},
    'FG': {'name': '玻璃', 'exchange': 'CZCE', 'sector': 'chemicals'},
    'SA': {'name': '纯碱', 'exchange': 'CZCE', 'sector': 'chemicals'},

    # 铁合金 CZCE
    'SF': {'name': '硅铁', 'exchange': 'CZCE', 'sector': 'ferroalloy'},
    'SM': {'name': '锰硅', 'exchange': 'CZCE', 'sector': 'ferroalloy'},
}

# Transaction cost parameters
TRANSACTION_COSTS = {
    'commission': 0.0001,  # 1bp commission
    'slippage': 0.0002,    # 2bp slippage
}


def load_all_data(years=10):
    """Load weekly data for all commodities using get_futures_daily."""
    print(f"Downloading {years}-year historical data...")

    all_data = {}
    end_date = datetime.now().strftime('%Y%m%d')
    start_date = (datetime.now() - timedelta(days=365*years)).strftime('%Y%m%d')

    for exchange in ['SHFE', 'CZCE']:
        print(f"\n  Downloading {exchange}...")

        try:
            df = ak.get_futures_daily(start_date=start_date, end_date=end_date, market=exchange)

            if df is None or len(df) == 0:
                print(f"    No data for {exchange}")
                continue

            print(f"    Got {len(df)} rows")

            for sym, info in COMMODITY_UNIVERSE.items():
                if info['exchange'] != exchange:
                    continue

                sym_df = df[df['variety'] == sym].copy()
                if len(sym_df) == 0:
                    continue

                sym_df = sym_df.sort_values(['date', 'volume'], ascending=[True, False])
                sym_df = sym_df.groupby('date').first().reset_index()
                sym_df['date'] = pd.to_datetime(sym_df['date'].astype(str))
                sym_df = sym_df.set_index('date')

                weekly = pd.DataFrame()
                weekly['close'] = sym_df['close'].resample('W-FRI').last()
                weekly['high'] = sym_df['high'].resample('W-FRI').max()
                weekly['low'] = sym_df['low'].resample('W-FRI').min()
                weekly = weekly.dropna()

                if len(weekly) >= 52:
                    all_data[sym] = weekly
                    print(f"    {sym} ({info['name']}): {len(weekly)} weeks")

        except Exception as e:
            print(f"    Error: {e}")

    print(f"\nLoaded {len(all_data)} commodities")
    return all_data


def calculate_volatility(prices: pd.Series, lookback: int = 12) -> float:
    """Calculate annualized volatility."""
    returns = prices.pct_change().dropna().tail(lookback)
    if len(returns) < 4:
        return np.nan
    return returns.std() * np.sqrt(52)


def apply_risk_limits(weights: dict, max_sector: float = 0.40,
                     max_single: float = 0.15, max_leverage: float = 1.5) -> dict:
    """Apply risk limits."""
    for sym in weights:
        if weights[sym] > max_single:
            weights[sym] = max_single

    sectors = {}
    for sym in weights:
        sector = COMMODITY_UNIVERSE.get(sym, {}).get('sector', 'other')
        if sector not in sectors:
            sectors[sector] = 0
        sectors[sector] += weights[sym]

    for sector, total in sectors.items():
        if total > max_sector:
            scale = max_sector / total
            for sym in weights:
                if COMMODITY_UNIVERSE.get(sym, {}).get('sector') == sector:
                    weights[sym] *= scale

    gross = sum(weights.values())
    if gross > max_leverage:
        scale = max_leverage / gross
        weights = {k: v * scale for k, v in weights.items()}

    return weights


def calculate_turnover(old_weights: dict, new_weights: dict) -> float:
    """
    Calculate portfolio turnover.
    Turnover = sum of absolute weight changes / 2
    """
    all_symbols = set(old_weights.keys()) | set(new_weights.keys())

    total_change = 0.0
    for sym in all_symbols:
        old_w = old_weights.get(sym, 0.0)
        new_w = new_weights.get(sym, 0.0)
        total_change += abs(new_w - old_w)

    return total_change / 2  # One-way turnover


def calculate_transaction_cost(turnover: float, costs: dict = TRANSACTION_COSTS) -> float:
    """
    Calculate transaction cost based on turnover.
    Cost = turnover * (commission + slippage) * 2 (buy and sell)
    """
    one_way_cost = costs['commission'] + costs['slippage']
    return turnover * one_way_cost * 2  # Round-trip


def run_backtest(all_data: dict, ma_period: int = 10, rebal_freq: str = 'weekly',
                 target_vol: float = 0.10, include_costs: bool = True) -> dict:
    """
    Run backtest with specified rebalancing frequency and transaction costs.

    Args:
        all_data: Dictionary of price DataFrames
        ma_period: MA period for signal generation
        rebal_freq: 'weekly' or 'monthly'
        target_vol: Target volatility for position sizing
        include_costs: Whether to include transaction costs

    Returns:
        Dictionary with results DataFrame and statistics
    """
    # Get date range
    all_dates = set()
    for sym, df in all_data.items():
        all_dates.update(df.index)
    all_dates = sorted(all_dates)

    start_date = pd.Timestamp('2016-01-01')
    end_date = pd.Timestamp('2026-01-24')
    dates = [d for d in all_dates if start_date <= d <= end_date]

    # Set rebalancing dates
    dates_series = pd.Series(dates)

    if rebal_freq == 'weekly':
        rebal_dates = dates  # Every week
    else:  # monthly
        month_ends = dates_series.groupby([dates_series.dt.year, dates_series.dt.month]).max().values
        rebal_dates = [pd.Timestamp(d) for d in month_ends]

    # Track portfolio
    portfolio_values = [1.0]
    portfolio_dates = [rebal_dates[0]]
    prev_weights = {}

    total_turnover = 0.0
    total_costs = 0.0
    num_rebalances = 0

    for i in range(1, len(rebal_dates)):
        current_date = rebal_dates[i]
        prev_date = rebal_dates[i-1]

        # Calculate signals
        signals = {}
        vols = {}

        for sym, df in all_data.items():
            df_up_to_date = df[df.index <= prev_date]

            if len(df_up_to_date) < max(52, ma_period):
                continue

            prices = df_up_to_date['close']
            ma = prices.rolling(ma_period).mean()

            # MA Long-Only signal
            if prices.iloc[-1] > ma.iloc[-1]:
                signal = 1.0
            else:
                signal = 0.0

            if signal <= 0:
                continue

            vol = calculate_volatility(df_up_to_date['close'], lookback=12)

            if not np.isnan(vol) and vol > 0:
                signals[sym] = signal
                vols[sym] = vol

        # Calculate weights
        if len(signals) == 0:
            new_weights = {}
        else:
            raw_weights = {}
            for sym in signals:
                raw_weights[sym] = signals[sym] * (target_vol / vols[sym])
            new_weights = apply_risk_limits(raw_weights)

        # Calculate turnover and costs
        turnover = calculate_turnover(prev_weights, new_weights)
        period_cost = calculate_transaction_cost(turnover) if include_costs else 0.0

        total_turnover += turnover
        total_costs += period_cost
        num_rebalances += 1

        # Calculate return
        period_return = 0.0
        for sym, weight in new_weights.items():
            df = all_data[sym]
            prev_prices = df[df.index <= prev_date]
            curr_prices = df[df.index <= current_date]

            if len(prev_prices) > 0 and len(curr_prices) > 0:
                p0 = prev_prices['close'].iloc[-1]
                p1 = curr_prices['close'].iloc[-1]
                ret = (p1 / p0) - 1
                period_return += weight * ret

        # Subtract transaction costs
        net_return = period_return - period_cost

        new_value = portfolio_values[-1] * (1 + net_return)
        portfolio_values.append(new_value)
        portfolio_dates.append(current_date)

        prev_weights = new_weights.copy()

    results = pd.DataFrame({
        'date': portfolio_dates,
        'value': portfolio_values
    })
    results.set_index('date', inplace=True)

    # Calculate statistics
    returns = results['value'].pct_change().dropna()
    total_return = (results['value'].iloc[-1] / results['value'].iloc[0] - 1) * 100
    n_years = len(rebal_dates) / 52 if rebal_freq == 'weekly' else len(rebal_dates) / 12
    cagr = ((results['value'].iloc[-1] / results['value'].iloc[0]) ** (1/n_years) - 1) * 100

    if rebal_freq == 'weekly':
        ann_vol = returns.std() * np.sqrt(52) * 100
    else:
        ann_vol = returns.std() * np.sqrt(12) * 100

    sharpe = cagr / ann_vol if ann_vol > 0 else 0
    peak = results['value'].expanding().max()
    max_dd = ((results['value'] - peak) / peak).min() * 100
    win_rate = (returns > 0).mean() * 100

    avg_turnover = total_turnover / num_rebalances if num_rebalances > 0 else 0
    annual_turnover = avg_turnover * (52 if rebal_freq == 'weekly' else 12)
    annual_cost = total_costs / n_years * 100  # As percentage

    return {
        'results': results,
        'total_return': total_return,
        'cagr': cagr,
        'volatility': ann_vol,
        'sharpe': sharpe,
        'max_dd': max_dd,
        'win_rate': win_rate,
        'total_turnover': total_turnover * 100,  # As percentage
        'annual_turnover': annual_turnover * 100,
        'total_costs': total_costs * 100,
        'annual_costs': annual_cost,
        'num_rebalances': num_rebalances,
    }


def main():
    print("="*70)
    print("CTA MA Strategy: Weekly vs Monthly Rebalancing with Transaction Costs")
    print("="*70)

    all_data = load_all_data(years=10)

    if len(all_data) < 10:
        print("Not enough data. Exiting.")
        return

    # Test configurations
    configs = [
        {'ma_period': 10, 'rebal_freq': 'monthly', 'include_costs': False, 'name': 'MA10 Monthly (No Costs)'},
        {'ma_period': 10, 'rebal_freq': 'monthly', 'include_costs': True, 'name': 'MA10 Monthly (With Costs)'},
        {'ma_period': 10, 'rebal_freq': 'weekly', 'include_costs': False, 'name': 'MA10 Weekly (No Costs)'},
        {'ma_period': 10, 'rebal_freq': 'weekly', 'include_costs': True, 'name': 'MA10 Weekly (With Costs)'},
        {'ma_period': 40, 'rebal_freq': 'monthly', 'include_costs': True, 'name': 'MA40 Monthly (With Costs)'},
        {'ma_period': 40, 'rebal_freq': 'weekly', 'include_costs': True, 'name': 'MA40 Weekly (With Costs)'},
    ]

    results = {}

    for config in configs:
        print(f"\n{'='*60}")
        print(f"Testing: {config['name']}")
        print(f"  MA Period: {config['ma_period']}")
        print(f"  Rebalancing: {config['rebal_freq']}")
        print(f"  Transaction Costs: {'Yes' if config['include_costs'] else 'No'}")
        print(f"{'='*60}")

        result = run_backtest(
            all_data,
            ma_period=config['ma_period'],
            rebal_freq=config['rebal_freq'],
            include_costs=config['include_costs']
        )

        results[config['name']] = result

        print(f"\nResults:")
        print(f"  Total Return: {result['total_return']:.1f}%")
        print(f"  CAGR: {result['cagr']:.1f}%")
        print(f"  Volatility: {result['volatility']:.1f}%")
        print(f"  Sharpe: {result['sharpe']:.2f}")
        print(f"  Max Drawdown: {result['max_dd']:.1f}%")
        print(f"  Win Rate: {result['win_rate']:.1f}%")
        print(f"  Rebalances: {result['num_rebalances']}")
        print(f"  Annual Turnover: {result['annual_turnover']:.1f}%")
        print(f"  Annual Costs: {result['annual_costs']:.2f}%")

    # Summary comparison
    print("\n" + "="*70)
    print("COMPARISON SUMMARY")
    print("="*70)
    print(f"{'Strategy':<30} {'Return':>8} {'CAGR':>7} {'Sharpe':>7} {'MaxDD':>8} {'AnnCost':>8} {'AnnTurn':>8}")
    print("-"*70)

    for name, result in results.items():
        print(f"{name:<30} {result['total_return']:>+7.1f}% {result['cagr']:>+6.1f}% "
              f"{result['sharpe']:>7.2f} {result['max_dd']:>+7.1f}% "
              f"{result['annual_costs']:>7.2f}% {result['annual_turnover']:>7.1f}%")

    # Cost impact analysis
    print("\n" + "="*70)
    print("TRANSACTION COST IMPACT ANALYSIS")
    print("="*70)

    ma10_monthly_no = results.get('MA10 Monthly (No Costs)', {})
    ma10_monthly_yes = results.get('MA10 Monthly (With Costs)', {})
    ma10_weekly_no = results.get('MA10 Weekly (No Costs)', {})
    ma10_weekly_yes = results.get('MA10 Weekly (With Costs)', {})

    if ma10_monthly_no and ma10_monthly_yes:
        cost_impact_monthly = ma10_monthly_no['cagr'] - ma10_monthly_yes['cagr']
        print(f"MA10 Monthly - Cost Impact: -{cost_impact_monthly:.2f}% CAGR")

    if ma10_weekly_no and ma10_weekly_yes:
        cost_impact_weekly = ma10_weekly_no['cagr'] - ma10_weekly_yes['cagr']
        print(f"MA10 Weekly - Cost Impact: -{cost_impact_weekly:.2f}% CAGR")

    if ma10_monthly_yes and ma10_weekly_yes:
        weekly_vs_monthly = ma10_weekly_yes['cagr'] - ma10_monthly_yes['cagr']
        print(f"\nWeekly vs Monthly (MA10 with costs):")
        print(f"  CAGR difference: {weekly_vs_monthly:+.2f}%")
        print(f"  Sharpe difference: {ma10_weekly_yes['sharpe'] - ma10_monthly_yes['sharpe']:+.2f}")
        print(f"  Turnover difference: {ma10_weekly_yes['annual_turnover'] - ma10_monthly_yes['annual_turnover']:+.1f}%")


if __name__ == "__main__":
    main()
