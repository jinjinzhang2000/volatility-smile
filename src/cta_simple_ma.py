#!/usr/bin/env python3
"""
Simple CTA Strategies - Moving Average Based
=============================================

Implements classic, robust CTA strategies:
1. 200-day (or 40-week) Moving Average
2. 50/200 Dual Moving Average Crossover
3. EWMAC (Exponential Weighted Moving Average Crossover)

Using AKShare for Chinese commodity futures data.
"""

import pandas as pd
import numpy as np
import akshare as ak
from datetime import datetime, timedelta
import warnings
warnings.filterwarnings('ignore')

# =============================================================================
# COMMODITY UNIVERSE (Same as before)
# =============================================================================

COMMODITY_UNIVERSE = {
    # 农产品 (Agriculture)
    'CF': {'name': '棉花', 'exchange': 'CZCE', 'sector': 'agriculture', 'mult': 5},
    'SR': {'name': '白糖', 'exchange': 'CZCE', 'sector': 'agriculture', 'mult': 10},
    'OI': {'name': '菜油', 'exchange': 'CZCE', 'sector': 'agriculture', 'mult': 10},
    'RM': {'name': '菜粕', 'exchange': 'CZCE', 'sector': 'agriculture', 'mult': 10},
    'AP': {'name': '苹果', 'exchange': 'CZCE', 'sector': 'agriculture', 'mult': 10},
    'CJ': {'name': '红枣', 'exchange': 'CZCE', 'sector': 'agriculture', 'mult': 5},
    'PK': {'name': '花生', 'exchange': 'CZCE', 'sector': 'agriculture', 'mult': 5},
    'CY': {'name': '棉纱', 'exchange': 'CZCE', 'sector': 'agriculture', 'mult': 5},

    # 能源化工 (Energy & Chemicals)
    'TA': {'name': 'PTA', 'exchange': 'CZCE', 'sector': 'energy', 'mult': 5},
    'MA': {'name': '甲醇', 'exchange': 'CZCE', 'sector': 'energy', 'mult': 10},
    'FG': {'name': '玻璃', 'exchange': 'CZCE', 'sector': 'energy', 'mult': 20},
    'SA': {'name': '纯碱', 'exchange': 'CZCE', 'sector': 'energy', 'mult': 20},
    'UR': {'name': '尿素', 'exchange': 'CZCE', 'sector': 'energy', 'mult': 20},
    'PF': {'name': '短纤', 'exchange': 'CZCE', 'sector': 'energy', 'mult': 5},
    'PX': {'name': '对二甲苯', 'exchange': 'CZCE', 'sector': 'energy', 'mult': 5},
    'SH': {'name': '烧碱', 'exchange': 'CZCE', 'sector': 'energy', 'mult': 30},

    # 黑色金属 (Ferrous Metals)
    'SF': {'name': '硅铁', 'exchange': 'CZCE', 'sector': 'ferrous', 'mult': 5},
    'SM': {'name': '锰硅', 'exchange': 'CZCE', 'sector': 'ferrous', 'mult': 5},

    # SHFE - 贵金属 (Precious Metals)
    'AU': {'name': '黄金', 'exchange': 'SHFE', 'sector': 'precious', 'mult': 1000},
    'AG': {'name': '白银', 'exchange': 'SHFE', 'sector': 'precious', 'mult': 15},

    # SHFE - 有色金属 (Base Metals)
    'CU': {'name': '铜', 'exchange': 'SHFE', 'sector': 'base_metal', 'mult': 5},
    'AL': {'name': '铝', 'exchange': 'SHFE', 'sector': 'base_metal', 'mult': 5},
    'ZN': {'name': '锌', 'exchange': 'SHFE', 'sector': 'base_metal', 'mult': 5},
    'NI': {'name': '镍', 'exchange': 'SHFE', 'sector': 'base_metal', 'mult': 1},
    'PB': {'name': '铅', 'exchange': 'SHFE', 'sector': 'base_metal', 'mult': 5},
    'SN': {'name': '锡', 'exchange': 'SHFE', 'sector': 'base_metal', 'mult': 1},
    'SS': {'name': '不锈钢', 'exchange': 'SHFE', 'sector': 'base_metal', 'mult': 5},
    'BC': {'name': '国际铜', 'exchange': 'SHFE', 'sector': 'base_metal', 'mult': 5},
    'AO': {'name': '氧化铝', 'exchange': 'SHFE', 'sector': 'base_metal', 'mult': 20},

    # SHFE - 黑色 (Ferrous)
    'RB': {'name': '螺纹钢', 'exchange': 'SHFE', 'sector': 'ferrous', 'mult': 10},
    'HC': {'name': '热卷', 'exchange': 'SHFE', 'sector': 'ferrous', 'mult': 10},
    'WR': {'name': '线材', 'exchange': 'SHFE', 'sector': 'ferrous', 'mult': 10},

    # SHFE - 能源 (Energy)
    'BU': {'name': '沥青', 'exchange': 'SHFE', 'sector': 'energy', 'mult': 10},
    'FU': {'name': '燃料油', 'exchange': 'SHFE', 'sector': 'energy', 'mult': 10},
    'LU': {'name': '低硫燃油', 'exchange': 'SHFE', 'sector': 'energy', 'mult': 10},
    'SC': {'name': '原油', 'exchange': 'SHFE', 'sector': 'energy', 'mult': 1000},
    'EC': {'name': '集运指数', 'exchange': 'SHFE', 'sector': 'energy', 'mult': 50},

    # SHFE - 其他 (Others)
    'RU': {'name': '橡胶', 'exchange': 'SHFE', 'sector': 'agriculture', 'mult': 10},
    'NR': {'name': '20号胶', 'exchange': 'SHFE', 'sector': 'agriculture', 'mult': 10},
    'BR': {'name': '丁二烯橡胶', 'exchange': 'SHFE', 'sector': 'agriculture', 'mult': 5},
    'SP': {'name': '纸浆', 'exchange': 'SHFE', 'sector': 'agriculture', 'mult': 10},
}

# =============================================================================
# DATA LOADING FUNCTIONS
# =============================================================================

def get_weekly_data_from_exchange(exchange: str, symbols: list) -> dict:
    """Get weekly data for all symbols from an exchange."""
    result = {}

    try:
        if exchange == 'CZCE':
            df = ak.futures_main_sina(symbol="CF0", start_date="20150101", end_date="20260201")
            df_all = ak.futures_zh_daily_sina(symbol="CF0")
        elif exchange == 'SHFE':
            df_all = ak.futures_zh_daily_sina(symbol="CU0")
        elif exchange == 'GFEX':
            df_all = ak.futures_zh_daily_sina(symbol="LC0")

        # Get data for each symbol
        for sym in symbols:
            try:
                symbol_code = f"{sym}0"  # Main contract
                df = ak.futures_zh_daily_sina(symbol=symbol_code)

                if df is not None and len(df) > 0:
                    df['date'] = pd.to_datetime(df['date'])
                    df = df.sort_values('date')
                    df.set_index('date', inplace=True)

                    # Resample to weekly
                    weekly = df['close'].resample('W-FRI').last().dropna()

                    if len(weekly) >= 52:  # At least 1 year
                        result[sym] = weekly
                        print(f"    {sym} ({COMMODITY_UNIVERSE[sym]['name']}): {len(weekly)} weekly")
            except Exception as e:
                pass

    except Exception as e:
        print(f"    Error downloading {exchange}: {e}")

    return result


def load_all_data() -> dict:
    """Load weekly price data for all commodities."""
    print("Downloading historical data from AKShare (weekly)...")

    all_data = {}

    # Group by exchange
    exchanges = {}
    for sym, info in COMMODITY_UNIVERSE.items():
        ex = info['exchange']
        if ex not in exchanges:
            exchanges[ex] = []
        exchanges[ex].append(sym)

    for exchange, symbols in exchanges.items():
        print(f"\n  Downloading {exchange} data...")

        for sym in symbols:
            try:
                symbol_code = f"{sym}0"
                df = ak.futures_zh_daily_sina(symbol=symbol_code)

                if df is not None and len(df) > 0:
                    df['date'] = pd.to_datetime(df['date'])
                    df = df.sort_values('date')
                    df.set_index('date', inplace=True)

                    # Resample to weekly
                    weekly = df['close'].resample('W-FRI').last().dropna()

                    if len(weekly) >= 52:
                        all_data[sym] = weekly
                        print(f"    {sym} ({COMMODITY_UNIVERSE[sym]['name']}): {len(weekly)} weekly")
            except Exception as e:
                pass

    print(f"\nLoaded data for {len(all_data)} commodities")
    return all_data


# =============================================================================
# STRATEGY: 40-Week (200-day) Moving Average
# =============================================================================

def strategy_ma200(prices: pd.Series, lookback: int = 40) -> pd.Series:
    """
    Simple 200-day (40-week) Moving Average Strategy.

    Signal:
    - Price > MA40 → Long (+1)
    - Price < MA40 → Short (-1)

    Args:
        prices: Weekly price series
        lookback: MA period in weeks (40 weeks ≈ 200 days)

    Returns:
        Signal series: +1 (long), -1 (short)
    """
    ma = prices.rolling(window=lookback).mean()
    signal = np.where(prices > ma, 1, -1)
    return pd.Series(signal, index=prices.index)


# =============================================================================
# STRATEGY: Dual Moving Average (10/40 weeks = 50/200 days)
# =============================================================================

def strategy_dual_ma(prices: pd.Series, fast: int = 10, slow: int = 40) -> pd.Series:
    """
    Dual Moving Average Crossover Strategy.

    Signal:
    - Fast MA > Slow MA → Long (+1)
    - Fast MA < Slow MA → Short (-1)

    Args:
        prices: Weekly price series
        fast: Fast MA period (10 weeks ≈ 50 days)
        slow: Slow MA period (40 weeks ≈ 200 days)

    Returns:
        Signal series: +1 (long), -1 (short)
    """
    ma_fast = prices.rolling(window=fast).mean()
    ma_slow = prices.rolling(window=slow).mean()
    signal = np.where(ma_fast > ma_slow, 1, -1)
    return pd.Series(signal, index=prices.index)


# =============================================================================
# STRATEGY: EWMAC (Exponential Weighted Moving Average Crossover)
# =============================================================================

def strategy_ewmac(prices: pd.Series, fast: int = 8, slow: int = 32) -> pd.Series:
    """
    EWMAC (Exponential Weighted Moving Average Crossover) Strategy.

    This is the core signal used by many systematic CTA funds.

    Signal:
    - Fast EMA > Slow EMA → Long (+1)
    - Fast EMA < Slow EMA → Short (-1)

    Args:
        prices: Weekly price series
        fast: Fast EMA span (8 weeks ≈ 40 days)
        slow: Slow EMA span (32 weeks ≈ 160 days)

    Returns:
        Signal series: +1 (long), -1 (short)
    """
    ema_fast = prices.ewm(span=fast, adjust=False).mean()
    ema_slow = prices.ewm(span=slow, adjust=False).mean()
    signal = np.where(ema_fast > ema_slow, 1, -1)
    return pd.Series(signal, index=prices.index)


# =============================================================================
# BACKTEST ENGINE
# =============================================================================

def calculate_volatility(prices: pd.Series, lookback: int = 12) -> float:
    """Calculate annualized volatility from weekly prices."""
    returns = prices.pct_change().dropna().tail(lookback)
    if len(returns) < 4:
        return np.nan
    return returns.std() * np.sqrt(52)


def run_backtest(all_data: dict, strategy_func, strategy_name: str,
                 target_vol: float = 0.10,
                 max_sector_weight: float = 0.40,
                 max_single_weight: float = 0.15,
                 max_leverage: float = 2.0) -> pd.DataFrame:
    """
    Run backtest for a given strategy.

    Monthly rebalancing with volatility targeting.
    """
    print(f"\n{'='*60}")
    print(f"Strategy: {strategy_name}")
    print(f"{'='*60}")

    # Get common date range
    all_dates = set()
    for sym, prices in all_data.items():
        all_dates.update(prices.index)
    all_dates = sorted(all_dates)

    start_date = pd.Timestamp('2016-01-01')
    end_date = pd.Timestamp('2026-01-24')

    dates = [d for d in all_dates if start_date <= d <= end_date]

    # Monthly rebalancing dates
    dates_series = pd.Series(dates)
    month_ends = dates_series.groupby([dates_series.dt.year, dates_series.dt.month]).max().values
    rebal_dates = [pd.Timestamp(d) for d in month_ends]

    print(f"Backtest period: {dates[0].strftime('%Y%m%d')} to {dates[-1].strftime('%Y%m%d')}")
    print(f"Found {len(rebal_dates)} month-end rebalancing dates")

    # Track portfolio
    portfolio_values = [1.0]
    portfolio_dates = [rebal_dates[0]]

    print(f"\nRunning backtest over {len(rebal_dates)} months...")

    for i in range(1, len(rebal_dates)):
        current_date = rebal_dates[i]
        prev_date = rebal_dates[i-1]

        # Calculate signals and weights for each commodity
        # IMPORTANT: Use data up to prev_date to avoid look-ahead bias
        # Signal is generated at month-end, then we hold until next month-end
        signals = {}
        vols = {}

        for sym, prices in all_data.items():
            # Get prices up to PREVIOUS date (avoid look-ahead bias)
            prices_up_to_date = prices[prices.index <= prev_date]

            if len(prices_up_to_date) < 52:  # Need at least 1 year
                continue

            # Calculate signal based on data available at prev_date
            signal_series = strategy_func(prices_up_to_date)
            signal = signal_series.iloc[-1] if len(signal_series) > 0 else 0

            # Calculate volatility
            vol = calculate_volatility(prices_up_to_date, lookback=12)

            if not np.isnan(vol) and vol > 0:
                signals[sym] = signal
                vols[sym] = vol

        if len(signals) == 0:
            portfolio_values.append(portfolio_values[-1])
            portfolio_dates.append(current_date)
            continue

        # Calculate raw weights (inverse vol weighted)
        raw_weights = {}
        for sym in signals:
            raw_weights[sym] = signals[sym] * (target_vol / vols[sym])

        # Apply risk limits
        weights = apply_risk_limits(raw_weights, max_sector_weight,
                                   max_single_weight, max_leverage)

        # Calculate period return
        period_return = 0.0
        for sym, weight in weights.items():
            if sym in all_data:
                prices = all_data[sym]

                # Get prices for current and previous date
                prev_prices = prices[prices.index <= prev_date]
                curr_prices = prices[prices.index <= current_date]

                if len(prev_prices) > 0 and len(curr_prices) > 0:
                    p0 = prev_prices.iloc[-1]
                    p1 = curr_prices.iloc[-1]
                    ret = (p1 / p0) - 1
                    period_return += weight * ret

        # Update portfolio
        new_value = portfolio_values[-1] * (1 + period_return)
        portfolio_values.append(new_value)
        portfolio_dates.append(current_date)

    # Create results DataFrame
    results = pd.DataFrame({
        'date': portfolio_dates,
        'value': portfolio_values
    })
    results.set_index('date', inplace=True)

    # Calculate statistics
    calc_and_print_stats(results, strategy_name)

    return results


def apply_risk_limits(weights: dict, max_sector: float, max_single: float,
                     max_leverage: float) -> dict:
    """Apply risk limits to portfolio weights."""

    # 1. Single position limit
    for sym in weights:
        if abs(weights[sym]) > max_single:
            weights[sym] = np.sign(weights[sym]) * max_single

    # 2. Sector limits
    sectors = {}
    for sym in weights:
        sector = COMMODITY_UNIVERSE.get(sym, {}).get('sector', 'other')
        if sector not in sectors:
            sectors[sector] = 0
        sectors[sector] += weights[sym]

    for sector, total in sectors.items():
        if abs(total) > max_sector:
            scale = max_sector / abs(total)
            for sym in weights:
                if COMMODITY_UNIVERSE.get(sym, {}).get('sector') == sector:
                    weights[sym] *= scale

    # 3. Gross leverage limit
    gross = sum(abs(w) for w in weights.values())
    if gross > max_leverage:
        scale = max_leverage / gross
        weights = {k: v * scale for k, v in weights.items()}

    return weights


def calc_and_print_stats(results: pd.DataFrame, strategy_name: str):
    """Calculate and print performance statistics."""

    returns = results['value'].pct_change().dropna()

    total_return = (results['value'].iloc[-1] / results['value'].iloc[0] - 1) * 100
    n_years = len(returns) / 12
    cagr = ((results['value'].iloc[-1] / results['value'].iloc[0]) ** (1/n_years) - 1) * 100

    ann_vol = returns.std() * np.sqrt(12) * 100
    sharpe = (cagr / ann_vol) if ann_vol > 0 else 0

    # Drawdown
    peak = results['value'].expanding().max()
    drawdown = (results['value'] - peak) / peak
    max_dd = drawdown.min() * 100

    # Win rate
    win_rate = (returns > 0).mean() * 100

    print(f"\n{strategy_name} Results:")
    print(f"  Total Return: {total_return:.1f}%")
    print(f"  CAGR: {cagr:.1f}%")
    print(f"  Annual Volatility: {ann_vol:.1f}%")
    print(f"  Sharpe Ratio: {sharpe:.2f}")
    print(f"  Max Drawdown: {max_dd:.1f}%")
    print(f"  Monthly Win Rate: {win_rate:.1f}%")


# =============================================================================
# MAIN
# =============================================================================

def main():
    print("="*60)
    print("Simple CTA Strategies Comparison")
    print("="*60)

    # Load data
    all_data = load_all_data()

    if len(all_data) < 10:
        print("Not enough data loaded. Exiting.")
        return

    # Run all strategies
    results = {}

    # Strategy 1: 40-Week (200-day) Moving Average
    results['MA200'] = run_backtest(
        all_data,
        lambda p: strategy_ma200(p, lookback=40),
        "40-Week Moving Average (≈200天均线)"
    )

    # Strategy 2: Dual MA (10/40 weeks = 50/200 days)
    results['DualMA'] = run_backtest(
        all_data,
        lambda p: strategy_dual_ma(p, fast=10, slow=40),
        "Dual MA Crossover (10/40周 ≈ 50/200天)"
    )

    # Strategy 3: EWMAC (8/32 weeks)
    results['EWMAC'] = run_backtest(
        all_data,
        lambda p: strategy_ewmac(p, fast=8, slow=32),
        "EWMAC (8/32周)"
    )

    # Strategy 4: Long-only MA200 (只做多)
    def ma200_long_only(prices, lookback=40):
        ma = prices.rolling(window=lookback).mean()
        signal = np.where(prices > ma, 1, 0)  # Long or flat
        return pd.Series(signal, index=prices.index)

    results['MA200_Long'] = run_backtest(
        all_data,
        lambda p: ma200_long_only(p, lookback=40),
        "40-Week MA Long-Only (只做多)"
    )

    # Print comparison
    print("\n" + "="*60)
    print("STRATEGY COMPARISON")
    print("="*60)

    for name, df in results.items():
        total_ret = (df['value'].iloc[-1] / df['value'].iloc[0] - 1) * 100
        returns = df['value'].pct_change().dropna()
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
