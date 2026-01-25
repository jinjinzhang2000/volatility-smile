#!/usr/bin/env python3
"""
Improved CTA Strategies for Chinese Commodity Futures
=====================================================

Key insights:
1. Long-only works better than long/short in China
2. Trend following needs filtering to avoid whipsaws
3. Combine signals for robustness

Strategies:
1. Long-only MA with filter
2. Donchian Channel Breakout (海龟法则)
3. Carry Strategy (展期收益)
4. Multi-timeframe momentum

Using AKShare for Chinese commodity futures data.
"""

import pandas as pd
import numpy as np
import akshare as ak
from datetime import datetime, timedelta
import warnings
warnings.filterwarnings('ignore')

# =============================================================================
# COMMODITY UNIVERSE (Focus on liquid markets)
# =============================================================================

# Focus on the most liquid contracts
COMMODITY_UNIVERSE = {
    # 农产品 (Agriculture) - High liquidity
    'CF': {'name': '棉花', 'exchange': 'CZCE', 'sector': 'agriculture'},
    'SR': {'name': '白糖', 'exchange': 'CZCE', 'sector': 'agriculture'},
    'OI': {'name': '菜油', 'exchange': 'CZCE', 'sector': 'agriculture'},
    'RM': {'name': '菜粕', 'exchange': 'CZCE', 'sector': 'agriculture'},
    'AP': {'name': '苹果', 'exchange': 'CZCE', 'sector': 'agriculture'},

    # 能源化工 (Energy & Chemicals)
    'TA': {'name': 'PTA', 'exchange': 'CZCE', 'sector': 'energy'},
    'MA': {'name': '甲醇', 'exchange': 'CZCE', 'sector': 'energy'},
    'FG': {'name': '玻璃', 'exchange': 'CZCE', 'sector': 'energy'},
    'SA': {'name': '纯碱', 'exchange': 'CZCE', 'sector': 'energy'},

    # 黑色金属 (Ferrous)
    'SF': {'name': '硅铁', 'exchange': 'CZCE', 'sector': 'ferrous'},
    'SM': {'name': '锰硅', 'exchange': 'CZCE', 'sector': 'ferrous'},

    # 贵金属 (Precious)
    'AU': {'name': '黄金', 'exchange': 'SHFE', 'sector': 'precious'},
    'AG': {'name': '白银', 'exchange': 'SHFE', 'sector': 'precious'},

    # 有色金属 (Base Metals)
    'CU': {'name': '铜', 'exchange': 'SHFE', 'sector': 'base_metal'},
    'AL': {'name': '铝', 'exchange': 'SHFE', 'sector': 'base_metal'},
    'ZN': {'name': '锌', 'exchange': 'SHFE', 'sector': 'base_metal'},
    'NI': {'name': '镍', 'exchange': 'SHFE', 'sector': 'base_metal'},

    # 黑色 (Ferrous)
    'RB': {'name': '螺纹钢', 'exchange': 'SHFE', 'sector': 'ferrous'},
    'HC': {'name': '热卷', 'exchange': 'SHFE', 'sector': 'ferrous'},

    # 能源 (Energy)
    'BU': {'name': '沥青', 'exchange': 'SHFE', 'sector': 'energy'},
    'FU': {'name': '燃料油', 'exchange': 'SHFE', 'sector': 'energy'},
    'SC': {'name': '原油', 'exchange': 'SHFE', 'sector': 'energy'},

    # 其他 (Others)
    'RU': {'name': '橡胶', 'exchange': 'SHFE', 'sector': 'agriculture'},
    'SP': {'name': '纸浆', 'exchange': 'SHFE', 'sector': 'agriculture'},
}


# =============================================================================
# DATA LOADING
# =============================================================================

def load_all_data() -> dict:
    """Load weekly price data for all commodities."""
    print("Downloading historical data from AKShare (weekly)...")

    all_data = {}

    for sym in COMMODITY_UNIVERSE.keys():
        try:
            symbol_code = f"{sym}0"
            df = ak.futures_zh_daily_sina(symbol=symbol_code)

            if df is not None and len(df) > 0:
                df['date'] = pd.to_datetime(df['date'])
                df = df.sort_values('date')
                df.set_index('date', inplace=True)

                # Ensure numeric columns
                for col in ['open', 'high', 'low', 'close']:
                    df[col] = pd.to_numeric(df[col], errors='coerce')

                # Resample to weekly - keep OHLC
                weekly = pd.DataFrame()
                weekly['open'] = df['open'].resample('W-FRI').first()
                weekly['high'] = df['high'].resample('W-FRI').max()
                weekly['low'] = df['low'].resample('W-FRI').min()
                weekly['close'] = df['close'].resample('W-FRI').last()
                weekly = weekly.dropna()

                if len(weekly) >= 52:  # At least 1 year
                    all_data[sym] = weekly
                    print(f"  {sym} ({COMMODITY_UNIVERSE[sym]['name']}): {len(weekly)} weekly")
        except Exception as e:
            print(f"  Error loading {sym}: {e}")

    print(f"\nLoaded data for {len(all_data)} commodities")
    return all_data


# =============================================================================
# STRATEGY 1: Filtered Long-Only with ATR
# =============================================================================

def strategy_filtered_long(df: pd.DataFrame, ma_period: int = 20,
                          atr_period: int = 10, atr_mult: float = 1.5) -> float:
    """
    Long-only strategy with volatility filter.

    Rules:
    - Go long if price > MA20 AND price > MA20 - ATR*1.5 (trend confirmation)
    - Stay flat otherwise

    This avoids false breakouts by requiring stronger trend.
    """
    if len(df) < ma_period + atr_period:
        return 0.0

    prices = df['close']
    ma = prices.rolling(ma_period).mean()

    # ATR calculation
    high = df['high']
    low = df['low']
    close = df['close']

    tr1 = high - low
    tr2 = abs(high - close.shift(1))
    tr3 = abs(low - close.shift(1))
    tr = pd.concat([tr1, tr2, tr3], axis=1).max(axis=1)
    atr = tr.rolling(atr_period).mean()

    current_price = prices.iloc[-1]
    current_ma = ma.iloc[-1]
    current_atr = atr.iloc[-1]

    # Long if price significantly above MA
    if current_price > current_ma + current_atr * 0.5:
        return 1.0
    # Flat if price below MA
    elif current_price < current_ma:
        return 0.0
    # In between - hold half position
    else:
        return 0.5


# =============================================================================
# STRATEGY 2: Donchian Channel Breakout (海龟法则)
# =============================================================================

def strategy_donchian(df: pd.DataFrame, entry_period: int = 20,
                     exit_period: int = 10) -> float:
    """
    Donchian Channel Breakout (Turtle Trading Rules).

    Rules:
    - Long: Price breaks above 20-week high
    - Exit: Price breaks below 10-week low
    - Short: Price breaks below 20-week low (optional)

    Long-only version for China market.
    """
    if len(df) < entry_period:
        return 0.0

    prices = df['close']
    high = df['high']
    low = df['low']

    # Channel calculations
    upper_channel = high.rolling(entry_period).max()
    lower_exit = low.rolling(exit_period).min()

    current_price = prices.iloc[-1]
    prev_upper = upper_channel.iloc[-2] if len(upper_channel) > 1 else np.nan
    prev_lower_exit = lower_exit.iloc[-2] if len(lower_exit) > 1 else np.nan

    # Breakout detection
    if current_price >= prev_upper:
        return 1.0  # Long on breakout
    elif current_price <= prev_lower_exit:
        return 0.0  # Exit on breakdown
    else:
        return np.nan  # Hold previous position


# =============================================================================
# STRATEGY 3: Multi-Timeframe Momentum
# =============================================================================

def strategy_multi_tf_momentum(df: pd.DataFrame) -> float:
    """
    Multi-timeframe momentum with long-only bias.

    Combines 3 timeframes:
    - Short-term (4 weeks): Recent momentum
    - Medium-term (13 weeks): Quarterly trend
    - Long-term (52 weeks): Yearly trend

    Long if majority of timeframes positive.
    """
    if len(df) < 52:
        return 0.0

    prices = df['close']

    # Returns over different periods
    r_4w = (prices.iloc[-1] / prices.iloc[-4] - 1) if len(prices) >= 4 else 0
    r_13w = (prices.iloc[-1] / prices.iloc[-13] - 1) if len(prices) >= 13 else 0
    r_52w = (prices.iloc[-1] / prices.iloc[-52] - 1) if len(prices) >= 52 else 0

    # Count positive timeframes
    score = int(r_4w > 0) + int(r_13w > 0) + int(r_52w > 0)

    # Long if 2+ timeframes positive
    if score >= 2:
        return 1.0
    elif score == 1:
        return 0.5
    else:
        return 0.0


# =============================================================================
# STRATEGY 4: Mean Reversion with Trend Filter
# =============================================================================

def strategy_mean_reversion(df: pd.DataFrame, ma_period: int = 40,
                           zscore_period: int = 20) -> float:
    """
    Mean reversion within trend.

    Rules:
    - If in uptrend (price > MA40), buy dips (oversold)
    - If in downtrend (price < MA40), stay flat

    This combines trend following with mean reversion for better entry.
    """
    if len(df) < max(ma_period, zscore_period):
        return 0.0

    prices = df['close']
    ma = prices.rolling(ma_period).mean()

    # Z-score of recent prices
    rolling_mean = prices.rolling(zscore_period).mean()
    rolling_std = prices.rolling(zscore_period).std()
    zscore = (prices.iloc[-1] - rolling_mean.iloc[-1]) / rolling_std.iloc[-1]

    current_price = prices.iloc[-1]
    current_ma = ma.iloc[-1]

    # In uptrend
    if current_price > current_ma:
        if zscore < -1:  # Oversold in uptrend - buy aggressively
            return 1.0
        elif zscore < 0:  # Slightly below mean in uptrend
            return 0.75
        else:  # Extended in uptrend
            return 0.5
    else:
        # In downtrend - stay flat
        return 0.0


# =============================================================================
# STRATEGY 5: Combined Strategy
# =============================================================================

def strategy_combined(df: pd.DataFrame) -> float:
    """
    Combines multiple signals for robustness.

    Average of:
    - Filtered long-only
    - Multi-TF momentum
    - Mean reversion with trend

    This diversifies across signal types.
    """
    s1 = strategy_filtered_long(df)
    s2 = strategy_multi_tf_momentum(df)
    s3 = strategy_mean_reversion(df)

    return (s1 + s2 + s3) / 3


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
                 max_leverage: float = 1.5) -> pd.DataFrame:
    """Run backtest for a given strategy."""
    print(f"\n{'='*60}")
    print(f"Strategy: {strategy_name}")
    print(f"{'='*60}")

    # Get common date range
    all_dates = set()
    for sym, df in all_data.items():
        all_dates.update(df.index)
    all_dates = sorted(all_dates)

    start_date = pd.Timestamp('2016-01-01')
    end_date = pd.Timestamp('2026-01-24')
    dates = [d for d in all_dates if start_date <= d <= end_date]

    # Monthly rebalancing dates
    dates_series = pd.Series(dates)
    month_ends = dates_series.groupby([dates_series.dt.year, dates_series.dt.month]).max().values
    rebal_dates = [pd.Timestamp(d) for d in month_ends]

    print(f"Backtest period: {dates[0].strftime('%Y%m%d')} to {dates[-1].strftime('%Y%m%d')}")
    print(f"Found {len(rebal_dates)} rebalancing dates")

    # Track portfolio
    portfolio_values = [1.0]
    portfolio_dates = [rebal_dates[0]]
    prev_positions = {}

    print(f"Running backtest over {len(rebal_dates)} months...")

    for i in range(1, len(rebal_dates)):
        current_date = rebal_dates[i]
        prev_date = rebal_dates[i-1]

        # Calculate signals using data up to prev_date (avoid look-ahead)
        signals = {}
        vols = {}

        for sym, df in all_data.items():
            df_up_to_date = df[df.index <= prev_date]

            if len(df_up_to_date) < 52:
                continue

            # Calculate signal
            signal = strategy_func(df_up_to_date)

            # Handle NaN signals (Donchian hold logic)
            if pd.isna(signal):
                signal = prev_positions.get(sym, 0.0)

            # Calculate volatility
            vol = calculate_volatility(df_up_to_date['close'], lookback=12)

            if not np.isnan(vol) and vol > 0 and signal > 0:
                signals[sym] = signal
                vols[sym] = vol

        prev_positions = {sym: signals.get(sym, 0) for sym in all_data.keys()}

        if len(signals) == 0:
            portfolio_values.append(portfolio_values[-1])
            portfolio_dates.append(current_date)
            continue

        # Calculate weights (inverse vol weighted)
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

    calc_and_print_stats(results, strategy_name)
    return results


def apply_risk_limits(weights: dict, max_sector: float, max_single: float,
                     max_leverage: float) -> dict:
    """Apply risk limits to portfolio weights."""

    # Single position limit
    for sym in weights:
        if abs(weights[sym]) > max_single:
            weights[sym] = np.sign(weights[sym]) * max_single

    # Sector limits
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

    # Gross leverage limit
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

    peak = results['value'].expanding().max()
    drawdown = (results['value'] - peak) / peak
    max_dd = drawdown.min() * 100

    win_rate = (returns > 0).mean() * 100

    print(f"\nResults:")
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
    print("Improved CTA Strategies - Long-Only Focused")
    print("="*60)

    all_data = load_all_data()

    if len(all_data) < 10:
        print("Not enough data loaded. Exiting.")
        return

    results = {}

    # Strategy 1: Filtered Long-Only
    results['Filtered'] = run_backtest(
        all_data, strategy_filtered_long,
        "Filtered Long-Only (MA20+ATR)"
    )

    # Strategy 2: Donchian Breakout
    results['Donchian'] = run_backtest(
        all_data, strategy_donchian,
        "Donchian Breakout (20/10周)"
    )

    # Strategy 3: Multi-TF Momentum
    results['MultiTF'] = run_backtest(
        all_data, strategy_multi_tf_momentum,
        "Multi-Timeframe Momentum"
    )

    # Strategy 4: Mean Reversion with Trend
    results['MeanRev'] = run_backtest(
        all_data, strategy_mean_reversion,
        "Mean Reversion in Trend"
    )

    # Strategy 5: Combined
    results['Combined'] = run_backtest(
        all_data, strategy_combined,
        "Combined Strategy"
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

        print(f"{name:12s}: Return={total_ret:6.1f}%, CAGR={cagr:5.1f}%, "
              f"Vol={ann_vol:5.1f}%, Sharpe={sharpe:5.2f}, MaxDD={max_dd:6.1f}%")


if __name__ == "__main__":
    main()
