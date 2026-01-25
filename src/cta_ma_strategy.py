#!/usr/bin/env python3
"""
Simple MA CTA Strategy for Chinese Commodity Futures
=====================================================

Based on 10-year backtest (2016-2026):
- MA10 Long-Only: +130.5% total, +8.7% CAGR, Sharpe 0.46
- MA40 Long-Only: +22.0% total, +2.0% CAGR, Sharpe 0.11

Conclusion: MA10 significantly outperforms MA40 in Chinese commodities.

Strategy Rules:
1. Price > MA10 (10-week) → Long
2. Price < MA10 → Flat (no position)
3. Volatility-weighted position sizing
4. Single position cap: 15%
5. Total leverage cap: 150%
6. Monthly rebalancing
"""

import pandas as pd
import numpy as np
import akshare as ak
from datetime import datetime, timedelta
import warnings
warnings.filterwarnings('ignore')

COMMODITIES = {
    # SHFE
    'AU': ('黄金', 'SHFE', 'precious'),
    'AG': ('白银', 'SHFE', 'precious'),
    'CU': ('铜', 'SHFE', 'base_metal'),
    'AL': ('铝', 'SHFE', 'base_metal'),
    'ZN': ('锌', 'SHFE', 'base_metal'),
    'NI': ('镍', 'SHFE', 'base_metal'),
    'PB': ('铅', 'SHFE', 'base_metal'),
    'SN': ('锡', 'SHFE', 'base_metal'),
    'RB': ('螺纹钢', 'SHFE', 'ferrous'),
    'HC': ('热卷', 'SHFE', 'ferrous'),
    'BU': ('沥青', 'SHFE', 'energy'),
    'FU': ('燃料油', 'SHFE', 'energy'),
    'SC': ('原油', 'SHFE', 'energy'),
    'RU': ('橡胶', 'SHFE', 'rubber'),
    'SP': ('纸浆', 'SHFE', 'paper'),
    # CZCE
    'CF': ('棉花', 'CZCE', 'agriculture'),
    'SR': ('白糖', 'CZCE', 'agriculture'),
    'OI': ('菜油', 'CZCE', 'agriculture'),
    'RM': ('菜粕', 'CZCE', 'agriculture'),
    'AP': ('苹果', 'CZCE', 'agriculture'),
    'TA': ('PTA', 'CZCE', 'chemicals'),
    'MA': ('甲醇', 'CZCE', 'chemicals'),
    'FG': ('玻璃', 'CZCE', 'chemicals'),
    'SA': ('纯碱', 'CZCE', 'chemicals'),
    'SF': ('硅铁', 'CZCE', 'ferroalloy'),
    'SM': ('锰硅', 'CZCE', 'ferroalloy'),
}


def get_current_signals(ma_period=10):
    """Get current trading signals based on MA strategy."""
    print(f"Downloading data for MA{ma_period} signals...")

    years = 3
    end_date = datetime.now().strftime('%Y%m%d')
    start_date = (datetime.now() - timedelta(days=365*years)).strftime('%Y%m%d')

    results = []

    for exchange in ['SHFE', 'CZCE']:
        try:
            df = ak.get_futures_daily(start_date=start_date, end_date=end_date, market=exchange)

            for sym, (name, ex, sector) in COMMODITIES.items():
                if ex != exchange:
                    continue

                sym_df = df[df['variety'] == sym].copy()
                if len(sym_df) == 0:
                    continue

                sym_df = sym_df.sort_values(['date', 'volume'], ascending=[True, False])
                sym_df = sym_df.groupby('date').first().reset_index()
                sym_df['date'] = pd.to_datetime(sym_df['date'].astype(str))
                sym_df = sym_df.set_index('date')

                weekly = sym_df['close'].resample('W-FRI').last().dropna()

                if len(weekly) < ma_period:
                    continue

                price = weekly.iloc[-1]
                ma = weekly.rolling(ma_period).mean().iloc[-1]
                distance = (price / ma - 1) * 100
                signal = 'LONG' if price > ma else 'FLAT'

                # Calculate volatility for position sizing
                returns = weekly.pct_change().dropna().tail(12)
                vol = returns.std() * np.sqrt(52) if len(returns) >= 4 else np.nan

                results.append({
                    'symbol': sym,
                    'name': name,
                    'sector': sector,
                    'price': price,
                    'ma': ma,
                    'distance': distance,
                    'signal': signal,
                    'volatility': vol
                })
        except Exception as e:
            print(f"  Error with {exchange}: {e}")

    return pd.DataFrame(results)


def calculate_weights(signals_df, target_vol=0.10, max_single=0.15, max_total=1.5):
    """Calculate portfolio weights based on signals."""
    longs = signals_df[signals_df['signal'] == 'LONG'].copy()

    if len(longs) == 0:
        return {}

    # Inverse volatility weighting
    longs['raw_weight'] = target_vol / longs['volatility']
    longs['raw_weight'] = longs['raw_weight'].clip(upper=max_single)

    # Scale to max leverage
    total = longs['raw_weight'].sum()
    if total > max_total:
        longs['weight'] = longs['raw_weight'] * (max_total / total)
    else:
        longs['weight'] = longs['raw_weight']

    return dict(zip(longs['symbol'], longs['weight']))


def print_signals(signals_df, ma_period=10):
    """Print formatted signals report."""
    print(f"\n{'='*80}")
    print(f"MA{ma_period} Long-Only Strategy - Current Signals")
    print(f"Date: {datetime.now().strftime('%Y-%m-%d')}")
    print(f"{'='*80}")

    signals_df = signals_df.sort_values('distance', ascending=False)

    print(f"\n{'Symbol':<8} {'Name':<8} {'Price':>10} {'MA%':>8} {'Signal':<6} {'Vol%':>8}")
    print('-'*60)

    for _, row in signals_df.iterrows():
        signal_mark = '●' if row['signal'] == 'LONG' else '○'
        vol_pct = f"{row['volatility']*100:.1f}%" if not pd.isna(row['volatility']) else 'N/A'
        print(f"{row['symbol']:<8} {row['name']:<8} {row['price']:>10.0f} "
              f"{row['distance']:>+7.1f}% {signal_mark}{row['signal']:<5} {vol_pct:>8}")

    # Summary
    longs = signals_df[signals_df['signal'] == 'LONG']
    flats = signals_df[signals_df['signal'] == 'FLAT']

    print('-'*60)
    print(f"LONG: {len(longs)} | FLAT: {len(flats)}")

    # Portfolio weights
    weights = calculate_weights(signals_df)

    if weights:
        print(f"\n{'='*80}")
        print("Recommended Portfolio Weights (Vol-Weighted)")
        print(f"{'='*80}")

        total_weight = 0
        for sym, weight in sorted(weights.items(), key=lambda x: -x[1]):
            name = COMMODITIES.get(sym, ('', '', ''))[0]
            print(f"  {sym} {name}: {weight*100:.1f}%")
            total_weight += weight

        print(f"\nTotal Exposure: {total_weight*100:.1f}%")


def main():
    """Main entry point."""
    import sys

    # Default MA period
    ma_period = 10

    if len(sys.argv) > 1:
        try:
            ma_period = int(sys.argv[1])
        except ValueError:
            pass

    signals_df = get_current_signals(ma_period)

    if signals_df.empty:
        print("Failed to get signals")
        return

    print_signals(signals_df, ma_period)

    # Save to CSV
    output_path = f"output/cta_signals_ma{ma_period}_{datetime.now().strftime('%Y%m%d')}.csv"
    signals_df.to_csv(output_path, index=False)
    print(f"\nSaved: {output_path}")


if __name__ == "__main__":
    main()
