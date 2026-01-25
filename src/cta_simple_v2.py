#!/usr/bin/env python3
"""
Simple CTA Strategies - Using get_futures_daily API
====================================================

Testing simple, robust CTA strategies for Chinese commodity futures.
Key insight: Long-only works better than long/short in China market.
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


# =============================================================================
# DATA LOADING
# =============================================================================

def load_all_data(years=10):
    """Load weekly data for all commodities using get_futures_daily."""
    print(f"Downloading {years}-year historical data...")

    all_data = {}
    end_date = datetime.now().strftime('%Y%m%d')
    start_date = (datetime.now() - timedelta(days=365*years)).strftime('%Y%m%d')

    # Download by exchange
    for exchange in ['SHFE', 'CZCE']:
        print(f"\n  Downloading {exchange}...")

        try:
            df = ak.get_futures_daily(start_date=start_date, end_date=end_date, market=exchange)

            if df is None or len(df) == 0:
                print(f"    No data for {exchange}")
                continue

            print(f"    Got {len(df)} rows")

            # Process each commodity
            for sym, info in COMMODITY_UNIVERSE.items():
                if info['exchange'] != exchange:
                    continue

                # Filter for this variety
                sym_df = df[df['variety'] == sym].copy()

                if len(sym_df) == 0:
                    continue

                # Use the most liquid contract each day
                sym_df = sym_df.sort_values(['date', 'volume'], ascending=[True, False])
                sym_df = sym_df.groupby('date').first().reset_index()

                # Convert to weekly
                sym_df['date'] = pd.to_datetime(sym_df['date'].astype(str))
                sym_df = sym_df.set_index('date')

                weekly = pd.DataFrame()
                weekly['close'] = sym_df['close'].resample('W-FRI').last()
                weekly['high'] = sym_df['high'].resample('W-FRI').max()
                weekly['low'] = sym_df['low'].resample('W-FRI').min()
                weekly = weekly.dropna()

                if len(weekly) >= 52:  # At least 1 year
                    all_data[sym] = weekly
                    print(f"    {sym} ({info['name']}): {len(weekly)} weeks")

        except Exception as e:
            print(f"    Error: {e}")

    print(f"\nLoaded {len(all_data)} commodities")
    return all_data


# =============================================================================
# STRATEGIES
# =============================================================================

def strategy_ma_long_only(df: pd.DataFrame, period: int = 40) -> float:
    """
    Simple MA Long-Only Strategy.
    - Price > MA → Long (1.0)
    - Price < MA → Flat (0.0)
    """
    if len(df) < period:
        return 0.0

    prices = df['close']
    ma = prices.rolling(period).mean()

    if prices.iloc[-1] > ma.iloc[-1]:
        return 1.0
    return 0.0


def strategy_dual_ma_long_only(df: pd.DataFrame, fast: int = 10, slow: int = 40) -> float:
    """
    Dual MA Long-Only Strategy.
    - Fast MA > Slow MA → Long (1.0)
    - Fast MA < Slow MA → Flat (0.0)
    """
    if len(df) < slow:
        return 0.0

    prices = df['close']
    ma_fast = prices.rolling(fast).mean()
    ma_slow = prices.rolling(slow).mean()

    if ma_fast.iloc[-1] > ma_slow.iloc[-1]:
        return 1.0
    return 0.0


def strategy_breakout(df: pd.DataFrame, period: int = 20) -> float:
    """
    Donchian Breakout Long-Only Strategy.
    - Price at 20-week high → Long (1.0)
    - Otherwise → Flat (0.0)
    """
    if len(df) < period:
        return 0.0

    prices = df['close']
    high_channel = df['high'].rolling(period).max()

    # Long if price >= recent high (breakout)
    if prices.iloc[-1] >= high_channel.iloc[-2]:
        return 1.0
    return 0.0


def strategy_multi_momentum(df: pd.DataFrame) -> float:
    """
    Multi-Timeframe Momentum Long-Only.
    Score = average of sign(returns) over 4w, 13w, 26w, 52w
    - Score > 0.5 → Long (1.0)
    - Score > 0 → Half (0.5)
    - Score <= 0 → Flat (0.0)
    """
    if len(df) < 52:
        return 0.0

    prices = df['close']
    periods = [4, 13, 26, 52]

    score = 0
    for p in periods:
        if len(prices) >= p:
            ret = prices.iloc[-1] / prices.iloc[-p] - 1
            score += 1 if ret > 0 else 0

    score = score / len(periods)

    if score > 0.5:
        return 1.0
    elif score > 0.25:
        return 0.5
    return 0.0


def strategy_combined(df: pd.DataFrame) -> float:
    """
    Combined Strategy: Average of MA, Dual MA, and Momentum.
    """
    s1 = strategy_ma_long_only(df, 40)
    s2 = strategy_dual_ma_long_only(df, 10, 40)
    s3 = strategy_multi_momentum(df)

    return (s1 + s2 + s3) / 3


# =============================================================================
# BACKTEST
# =============================================================================

def calculate_volatility(prices: pd.Series, lookback: int = 12) -> float:
    """Calculate annualized volatility."""
    returns = prices.pct_change().dropna().tail(lookback)
    if len(returns) < 4:
        return np.nan
    return returns.std() * np.sqrt(52)


def apply_risk_limits(weights: dict, max_sector: float = 0.40,
                     max_single: float = 0.15, max_leverage: float = 1.5) -> dict:
    """Apply risk limits."""

    # Single position limit
    for sym in weights:
        if weights[sym] > max_single:
            weights[sym] = max_single

    # Sector limits
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

    # Gross leverage limit
    gross = sum(weights.values())
    if gross > max_leverage:
        scale = max_leverage / gross
        weights = {k: v * scale for k, v in weights.items()}

    return weights


def run_backtest(all_data: dict, strategy_func, strategy_name: str,
                 target_vol: float = 0.10) -> pd.DataFrame:
    """Run backtest."""
    print(f"\n{'='*60}")
    print(f"Strategy: {strategy_name}")
    print(f"{'='*60}")

    # Get date range
    all_dates = set()
    for sym, df in all_data.items():
        all_dates.update(df.index)
    all_dates = sorted(all_dates)

    start_date = pd.Timestamp('2016-01-01')
    end_date = pd.Timestamp('2026-01-24')
    dates = [d for d in all_dates if start_date <= d <= end_date]

    # Monthly rebalancing
    dates_series = pd.Series(dates)
    month_ends = dates_series.groupby([dates_series.dt.year, dates_series.dt.month]).max().values
    rebal_dates = [pd.Timestamp(d) for d in month_ends]

    print(f"Period: {dates[0].strftime('%Y-%m-%d')} to {dates[-1].strftime('%Y-%m-%d')}")
    print(f"Rebalancing: {len(rebal_dates)} months")

    # Track portfolio
    portfolio_values = [1.0]
    portfolio_dates = [rebal_dates[0]]

    for i in range(1, len(rebal_dates)):
        current_date = rebal_dates[i]
        prev_date = rebal_dates[i-1]

        # Calculate signals (use prev_date to avoid look-ahead)
        signals = {}
        vols = {}

        for sym, df in all_data.items():
            df_up_to_date = df[df.index <= prev_date]

            if len(df_up_to_date) < 52:
                continue

            signal = strategy_func(df_up_to_date)

            if signal <= 0:
                continue

            vol = calculate_volatility(df_up_to_date['close'], lookback=12)

            if not np.isnan(vol) and vol > 0:
                signals[sym] = signal
                vols[sym] = vol

        if len(signals) == 0:
            portfolio_values.append(portfolio_values[-1])
            portfolio_dates.append(current_date)
            continue

        # Calculate weights (inverse vol)
        raw_weights = {}
        for sym in signals:
            raw_weights[sym] = signals[sym] * (target_vol / vols[sym])

        weights = apply_risk_limits(raw_weights)

        # Calculate return
        period_return = 0.0
        for sym, weight in weights.items():
            df = all_data[sym]
            prev_prices = df[df.index <= prev_date]
            curr_prices = df[df.index <= current_date]

            if len(prev_prices) > 0 and len(curr_prices) > 0:
                p0 = prev_prices['close'].iloc[-1]
                p1 = curr_prices['close'].iloc[-1]
                ret = (p1 / p0) - 1
                period_return += weight * ret

        new_value = portfolio_values[-1] * (1 + period_return)
        portfolio_values.append(new_value)
        portfolio_dates.append(current_date)

    results = pd.DataFrame({
        'date': portfolio_dates,
        'value': portfolio_values
    })
    results.set_index('date', inplace=True)

    # Stats
    returns = results['value'].pct_change().dropna()
    total_return = (results['value'].iloc[-1] / results['value'].iloc[0] - 1) * 100
    n_years = len(returns) / 12
    cagr = ((results['value'].iloc[-1] / results['value'].iloc[0]) ** (1/n_years) - 1) * 100
    ann_vol = returns.std() * np.sqrt(12) * 100
    sharpe = cagr / ann_vol if ann_vol > 0 else 0
    peak = results['value'].expanding().max()
    max_dd = ((results['value'] - peak) / peak).min() * 100
    win_rate = (returns > 0).mean() * 100

    print(f"\nResults:")
    print(f"  Total Return: {total_return:.1f}%")
    print(f"  CAGR: {cagr:.1f}%")
    print(f"  Volatility: {ann_vol:.1f}%")
    print(f"  Sharpe: {sharpe:.2f}")
    print(f"  Max Drawdown: {max_dd:.1f}%")
    print(f"  Win Rate: {win_rate:.1f}%")

    return results


# =============================================================================
# MAIN
# =============================================================================

def main():
    print("="*60)
    print("Simple CTA Strategies - Long Only")
    print("="*60)

    all_data = load_all_data(years=10)

    if len(all_data) < 10:
        print("Not enough data. Exiting.")
        return

    results = {}

    # Test strategies
    results['MA40_Long'] = run_backtest(
        all_data, strategy_ma_long_only,
        "40-Week MA Long-Only"
    )

    results['DualMA_Long'] = run_backtest(
        all_data, strategy_dual_ma_long_only,
        "Dual MA (10/40) Long-Only"
    )

    results['Breakout'] = run_backtest(
        all_data, strategy_breakout,
        "20-Week Breakout Long-Only"
    )

    results['MultiMom'] = run_backtest(
        all_data, strategy_multi_momentum,
        "Multi-TF Momentum Long-Only"
    )

    results['Combined'] = run_backtest(
        all_data, strategy_combined,
        "Combined Strategy"
    )

    # Summary
    print("\n" + "="*60)
    print("STRATEGY COMPARISON")
    print("="*60)

    for name, df in results.items():
        returns = df['value'].pct_change().dropna()
        total_ret = (df['value'].iloc[-1] / df['value'].iloc[0] - 1) * 100
        n_years = len(returns) / 12
        cagr = ((df['value'].iloc[-1] / df['value'].iloc[0]) ** (1/n_years) - 1) * 100
        ann_vol = returns.std() * np.sqrt(12) * 100
        sharpe = cagr / ann_vol if ann_vol > 0 else 0
        peak = df['value'].expanding().max()
        max_dd = ((df['value'] - peak) / peak).min() * 100

        print(f"{name:15s}: Return={total_ret:6.1f}%, CAGR={cagr:5.1f}%, "
              f"Vol={ann_vol:5.1f}%, Sharpe={sharpe:5.2f}, MaxDD={max_dd:6.1f}%")


if __name__ == "__main__":
    main()
