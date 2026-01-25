"""
Tail Strategy Backtester
Tests systematic tail hedging strategies using options

Strategies:
1. Systematic OTM Put Buying - buy puts when skew is low
2. Skew Mean Reversion - trade risk reversals when skew is extreme
3. Vol Spike Protection - buy straddles when vol is low
4. Cross-Asset Tail Hedge - use correlated assets for cheaper hedges
"""

import pandas as pd
import numpy as np
from scipy.stats import norm
import matplotlib.pyplot as plt
import matplotlib
import os
from datetime import datetime, timedelta
from skew_analyzer import (
    load_skew_history, calculate_skew_percentile,
    ASSET_GROUPS, calculate_skew_metrics
)

matplotlib.rcParams['font.sans-serif'] = ['SimHei', 'Arial Unicode MS', 'Microsoft YaHei']
matplotlib.rcParams['axes.unicode_minus'] = False

# Output directory
OUTPUT_DIR = "output/backtest"


def black_scholes_price(S, K, T, r, sigma, option_type='put'):
    """Calculate Black-Scholes option price"""
    if T <= 0 or sigma <= 0:
        return max(0, K - S) if option_type == 'put' else max(0, S - K)

    d1 = (np.log(S / K) + (r + 0.5 * sigma ** 2) * T) / (sigma * np.sqrt(T))
    d2 = d1 - sigma * np.sqrt(T)

    if option_type == 'call':
        return S * norm.cdf(d1) - K * np.exp(-r * T) * norm.cdf(d2)
    else:
        return K * np.exp(-r * T) * norm.cdf(-d2) - S * norm.cdf(-d1)


class TailStrategyBacktester:
    """Backtester for tail hedging strategies"""

    def __init__(self, product_code, underlying_data=None):
        """
        Initialize backtester

        Args:
            product_code: Asset code (e.g., 'rb', 'fg', 'io')
            underlying_data: DataFrame with columns ['date', 'close']
        """
        self.product = product_code
        self.underlying = underlying_data
        self.skew_history = load_skew_history()
        self.results = {}

    def load_underlying_from_tushare(self, start_date, end_date):
        """Load underlying price data from Tushare"""
        try:
            import tushare as ts
            token = os.environ.get("TUSHARE_TOKEN", "")
            if not token:
                print("Warning: TUSHARE_TOKEN not set")
                return None

            ts.set_token(token)
            pro = ts.pro_api()

            # Map product codes to Tushare codes
            code_map = {
                'rb': 'RB.SHF',
                'fg': 'FG.ZCE',
                'ag': 'AG.SHF',
                'au': 'AU.SHF',
                'cu': 'CU.SHF',
            }

            if self.product not in code_map:
                print(f"No mapping for {self.product}")
                return None

            df = pro.fut_daily(
                ts_code=code_map[self.product],
                start_date=start_date,
                end_date=end_date,
                fields='trade_date,close'
            )

            if df is not None and not df.empty:
                df['date'] = pd.to_datetime(df['trade_date'], format='%Y%m%d')
                df = df.sort_values('date')
                self.underlying = df[['date', 'close']]
                return df

        except Exception as e:
            print(f"Error loading data: {e}")

        return None

    def strategy_systematic_put_buying(self, moneyness=0.95, holding_days=20,
                                        entry_percentile=30, position_size=0.02):
        """
        Strategy: Buy OTM puts when skew is low (cheap protection)

        Args:
            moneyness: Strike as fraction of spot (0.95 = 5% OTM)
            holding_days: Days to hold position
            entry_percentile: Enter when skew below this percentile
            position_size: Fraction of portfolio to spend on puts

        Returns:
            DataFrame with strategy results
        """
        if self.skew_history.empty:
            print("No skew history available")
            return None

        hist = self.skew_history[self.skew_history['product'] == self.product].copy()
        if hist.empty:
            print(f"No history for {self.product}")
            return None

        # Get front-month data
        front_month = hist.groupby('date').first().reset_index()
        front_month = front_month.sort_values('date')

        trades = []
        r = 0.025  # Risk-free rate

        for i in range(len(front_month) - holding_days):
            row = front_month.iloc[i]
            date = row['date']
            atm_iv = row['atm_iv'] / 100
            skew = row['skew_25d']

            # Calculate percentile
            lookback = front_month[front_month['date'] < date].tail(60)
            if len(lookback) < 20:
                continue

            percentile = (lookback['skew_25d'] < skew).mean() * 100

            # Entry signal: skew is low (cheap puts)
            if percentile <= entry_percentile:
                # Simulate put purchase
                # Use ATM IV + some adjustment for OTM
                put_iv = atm_iv * (1 + (1 - moneyness) * 2)  # Higher IV for OTM

                T = holding_days / 365
                S = 100  # Normalized spot
                K = S * moneyness

                entry_price = black_scholes_price(S, K, T, r, put_iv, 'put')

                # Get exit data
                exit_row = front_month.iloc[i + holding_days]
                exit_atm_iv = exit_row['atm_iv'] / 100

                # Simulate underlying move (use random for now, replace with actual data)
                # In practice, use actual underlying returns
                np.random.seed(int(date.timestamp()) % 2**31)
                daily_vol = atm_iv / np.sqrt(252)
                underlying_return = np.random.normal(0, daily_vol * np.sqrt(holding_days))
                S_exit = S * (1 + underlying_return)

                T_exit = 0.001  # Near expiry
                exit_price = black_scholes_price(S_exit, K, T_exit, r, exit_atm_iv, 'put')

                # P&L
                pnl = (exit_price - entry_price) / entry_price
                pnl_dollar = pnl * position_size * 100  # Assuming $100 portfolio

                trades.append({
                    'entry_date': date,
                    'exit_date': exit_row['date'],
                    'entry_skew_pct': percentile,
                    'entry_iv': atm_iv * 100,
                    'exit_iv': exit_atm_iv * 100,
                    'underlying_return': underlying_return * 100,
                    'option_return': pnl * 100,
                    'pnl_dollar': pnl_dollar
                })

        if not trades:
            print("No trades generated")
            return None

        results = pd.DataFrame(trades)
        self.results['systematic_put'] = results
        return results

    def strategy_skew_mean_reversion(self, entry_high_pct=90, entry_low_pct=10,
                                      holding_days=10):
        """
        Strategy: Trade risk reversals when skew is extreme

        - High skew (>90th pct): Sell puts, buy calls (expect skew to normalize)
        - Low skew (<10th pct): Buy puts, sell calls (expect skew to rise)

        Returns:
            DataFrame with strategy results
        """
        if self.skew_history.empty:
            return None

        hist = self.skew_history[self.skew_history['product'] == self.product].copy()
        if hist.empty:
            return None

        front_month = hist.groupby('date').first().reset_index()
        front_month = front_month.sort_values('date')

        trades = []

        for i in range(len(front_month) - holding_days):
            row = front_month.iloc[i]
            date = row['date']
            skew = row['skew_25d']

            lookback = front_month[front_month['date'] < date].tail(60)
            if len(lookback) < 20:
                continue

            percentile = (lookback['skew_25d'] < skew).mean() * 100

            direction = None
            if percentile >= entry_high_pct:
                direction = 'short_skew'  # Expect skew to fall
            elif percentile <= entry_low_pct:
                direction = 'long_skew'   # Expect skew to rise

            if direction:
                exit_row = front_month.iloc[i + holding_days]
                skew_change = exit_row['skew_25d'] - skew

                if direction == 'short_skew':
                    pnl = -skew_change  # Profit when skew falls
                else:
                    pnl = skew_change   # Profit when skew rises

                trades.append({
                    'entry_date': date,
                    'exit_date': exit_row['date'],
                    'direction': direction,
                    'entry_skew': skew,
                    'exit_skew': exit_row['skew_25d'],
                    'skew_change': skew_change,
                    'pnl_bps': pnl * 100  # In basis points
                })

        if not trades:
            return None

        results = pd.DataFrame(trades)
        self.results['skew_mean_reversion'] = results
        return results

    def strategy_tail_event_detector(self, vol_spike_threshold=1.3,
                                      skew_spike_threshold=80):
        """
        Strategy: Detect potential tail events based on vol/skew signals

        Returns list of detected events with characteristics
        """
        if self.skew_history.empty:
            return None

        hist = self.skew_history[self.skew_history['product'] == self.product].copy()
        if hist.empty:
            return None

        front_month = hist.groupby('date').first().reset_index()
        front_month = front_month.sort_values('date')

        events = []

        for i in range(20, len(front_month)):
            row = front_month.iloc[i]
            date = row['date']

            lookback = front_month.iloc[i-20:i]
            avg_iv = lookback['atm_iv'].mean()
            avg_skew = lookback['skew_25d'].mean()

            iv_ratio = row['atm_iv'] / avg_iv
            skew_percentile = (lookback['skew_25d'] < row['skew_25d']).mean() * 100

            # Detect tail event conditions
            is_vol_spike = iv_ratio >= vol_spike_threshold
            is_skew_spike = skew_percentile >= skew_spike_threshold

            if is_vol_spike or is_skew_spike:
                event_type = []
                if is_vol_spike:
                    event_type.append('VOL_SPIKE')
                if is_skew_spike:
                    event_type.append('SKEW_SPIKE')

                events.append({
                    'date': date,
                    'event_type': '+'.join(event_type),
                    'atm_iv': row['atm_iv'],
                    'iv_ratio': iv_ratio,
                    'skew': row['skew_25d'],
                    'skew_percentile': skew_percentile
                })

        return pd.DataFrame(events) if events else None

    def analyze_results(self, strategy_name):
        """Analyze and print strategy results"""
        if strategy_name not in self.results:
            print(f"No results for {strategy_name}")
            return

        results = self.results[strategy_name]

        print(f"\n{'='*60}")
        print(f"Strategy: {strategy_name}")
        print(f"Product: {self.product.upper()}")
        print(f"{'='*60}")

        print(f"\nTrade Statistics:")
        print(f"  Total trades: {len(results)}")

        if 'pnl_dollar' in results.columns:
            total_pnl = results['pnl_dollar'].sum()
            avg_pnl = results['pnl_dollar'].mean()
            win_rate = (results['pnl_dollar'] > 0).mean() * 100

            print(f"  Total P&L: ${total_pnl:.2f}")
            print(f"  Avg P&L per trade: ${avg_pnl:.2f}")
            print(f"  Win rate: {win_rate:.1f}%")

            # Analyze tail events
            big_wins = results[results['pnl_dollar'] > results['pnl_dollar'].std() * 2]
            print(f"  Large winning trades (>2σ): {len(big_wins)}")

        if 'option_return' in results.columns:
            print(f"\n  Avg option return: {results['option_return'].mean():.1f}%")
            print(f"  Max option return: {results['option_return'].max():.1f}%")
            print(f"  Min option return: {results['option_return'].min():.1f}%")

        if 'underlying_return' in results.columns:
            # Correlation with underlying moves
            corr = results['option_return'].corr(results['underlying_return'])
            print(f"\n  Correlation with underlying: {corr:.2f}")

            # Performance in down markets
            down_markets = results[results['underlying_return'] < -5]
            if len(down_markets) > 0:
                print(f"\n  Performance in down markets (>5% drop):")
                print(f"    Number of periods: {len(down_markets)}")
                print(f"    Avg option return: {down_markets['option_return'].mean():.1f}%")

    def plot_results(self, strategy_name, save_path=None):
        """Plot strategy results"""
        if strategy_name not in self.results:
            return

        results = self.results[strategy_name]

        fig, axes = plt.subplots(2, 2, figsize=(14, 10))

        # Cumulative P&L
        if 'pnl_dollar' in results.columns:
            results['cum_pnl'] = results['pnl_dollar'].cumsum()
            axes[0, 0].plot(results['entry_date'], results['cum_pnl'], 'b-', linewidth=2)
            axes[0, 0].fill_between(results['entry_date'], results['cum_pnl'],
                                     alpha=0.3, color='blue')
            axes[0, 0].set_title('Cumulative P&L', fontsize=12, fontweight='bold')
            axes[0, 0].set_ylabel('P&L ($)')
            axes[0, 0].grid(True, alpha=0.3)

        # P&L distribution
        if 'pnl_dollar' in results.columns:
            axes[0, 1].hist(results['pnl_dollar'], bins=30, edgecolor='black', alpha=0.7)
            axes[0, 1].axvline(x=0, color='red', linestyle='--', linewidth=2)
            axes[0, 1].set_title('P&L Distribution', fontsize=12, fontweight='bold')
            axes[0, 1].set_xlabel('P&L ($)')
            axes[0, 1].grid(True, alpha=0.3)

        # Option return vs underlying return
        if 'option_return' in results.columns and 'underlying_return' in results.columns:
            axes[1, 0].scatter(results['underlying_return'], results['option_return'],
                               alpha=0.5, s=50)
            axes[1, 0].axhline(y=0, color='gray', linestyle='-', linewidth=0.5)
            axes[1, 0].axvline(x=0, color='gray', linestyle='-', linewidth=0.5)
            axes[1, 0].set_title('Option vs Underlying Returns', fontsize=12, fontweight='bold')
            axes[1, 0].set_xlabel('Underlying Return (%)')
            axes[1, 0].set_ylabel('Option Return (%)')
            axes[1, 0].grid(True, alpha=0.3)

        # Entry skew percentile
        if 'entry_skew_pct' in results.columns:
            axes[1, 1].hist(results['entry_skew_pct'], bins=20, edgecolor='black', alpha=0.7)
            axes[1, 1].set_title('Entry Skew Percentile Distribution', fontsize=12, fontweight='bold')
            axes[1, 1].set_xlabel('Skew Percentile')
            axes[1, 1].grid(True, alpha=0.3)

        plt.suptitle(f'{self.product.upper()} - {strategy_name}',
                     fontsize=14, fontweight='bold')
        plt.tight_layout()

        if save_path:
            os.makedirs(os.path.dirname(save_path), exist_ok=True)
            plt.savefig(save_path, dpi=300, bbox_inches='tight')
            print(f"Saved: {save_path}")

        plt.close()


def run_group_backtest(group_name, strategy='systematic_put'):
    """Run backtest across an asset group"""
    if group_name not in ASSET_GROUPS:
        print(f"Unknown group: {group_name}")
        return

    group = ASSET_GROUPS[group_name]
    print(f"\n{'='*60}")
    print(f"Group Backtest: {group['name']}")
    print(f"Strategy: {strategy}")
    print(f"{'='*60}")

    all_results = {}

    for asset in group['assets']:
        print(f"\nProcessing {asset}...")
        bt = TailStrategyBacktester(asset)

        if strategy == 'systematic_put':
            results = bt.strategy_systematic_put_buying()
        elif strategy == 'skew_mean_reversion':
            results = bt.strategy_skew_mean_reversion()
        else:
            continue

        if results is not None:
            bt.analyze_results(strategy)
            all_results[asset] = results

            # Save plot
            save_path = f"{OUTPUT_DIR}/{asset}_{strategy}.png"
            bt.plot_results(strategy, save_path)

    return all_results


if __name__ == '__main__':
    print("="*60)
    print("Tail Strategy Backtester")
    print("="*60)

    # Example: Run backtest on real estate group
    print("\nAvailable asset groups:")
    for name, group in ASSET_GROUPS.items():
        print(f"  {name}: {group['assets']}")

    print("\nTo run a backtest:")
    print("  from tail_strategy_backtest import TailStrategyBacktester, run_group_backtest")
    print("  run_group_backtest('real_estate', 'systematic_put')")
