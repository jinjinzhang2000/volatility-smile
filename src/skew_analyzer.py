"""
Volatility Skew Analyzer for Tail Risk Strategies
Calculates skew metrics, tracks history, and generates alerts

Supports:
- 25-delta put-call skew (risk reversal)
- ATM volatility tracking
- Skew percentile ranking
- Tail risk alerts
"""

import pandas as pd
import numpy as np
from scipy.stats import norm
from scipy import optimize
import matplotlib.pyplot as plt
import matplotlib
import os
import json
from datetime import datetime, timedelta

# Try to use WenQuanYi fonts for better support in Linux/GitHub Actions
matplotlib.rcParams['font.sans-serif'] = ['WenQuanYi Micro Hei', 'WenQuanYi Zen Hei', 'SimHei', 'Arial Unicode MS', 'Microsoft YaHei']
matplotlib.rcParams['axes.unicode_minus'] = False

# Configuration
SKEW_HISTORY_FILE = "output/data/skew_history.csv"
ALERT_THRESHOLDS = {
    'skew_percentile_high': 90,  # Alert when skew > 90th percentile
    'skew_percentile_low': 10,   # Alert when skew < 10th percentile
    'atm_iv_spike': 1.5,         # Alert when ATM IV > 1.5x 20-day average
}


def black_scholes_delta(S, K, T, r, sigma, option_type='call'):
    """Calculate Black-Scholes delta"""
    if T <= 0 or sigma <= 0:
        return 0
    d1 = (np.log(S / K) + (r + 0.5 * sigma ** 2) * T) / (sigma * np.sqrt(T))
    if option_type == 'call':
        return norm.cdf(d1)
    else:
        return norm.cdf(d1) - 1


def find_strike_for_delta(S, T, r, sigma, target_delta, option_type='call'):
    """Find strike price that gives target delta"""
    def objective(K):
        delta = black_scholes_delta(S, K, T, r, sigma, option_type)
        return (delta - target_delta) ** 2

    # Search bounds
    K_low = S * 0.5
    K_high = S * 1.5

    try:
        result = optimize.minimize_scalar(objective, bounds=(K_low, K_high), method='bounded')
        return result.x
    except:
        return None


def calculate_skew_metrics(smile_df, underlying_price=None):
    """
    Calculate skew metrics from volatility smile data

    Returns:
        dict with skew metrics:
        - atm_iv: ATM implied volatility
        - put_25d_iv: 25-delta put IV
        - call_25d_iv: 25-delta call IV
        - skew_25d: 25-delta risk reversal (put IV - call IV)
        - butterfly_25d: 25-delta butterfly (wing avg - ATM)
        - skew_slope: Linear slope of smile
    """
    if smile_df is None or smile_df.empty:
        return None

    results = {}

    for maturity in smile_df['maturity'].unique():
        df_mat = smile_df[smile_df['maturity'] == maturity].copy()

        if len(df_mat) < 5:
            continue

        # Get underlying price
        if 'underlying' in df_mat.columns:
            S = df_mat['underlying'].iloc[0]
        elif 'forward' in df_mat.columns:
            S = df_mat['forward'].iloc[0]
        elif underlying_price:
            S = underlying_price
        else:
            continue

        days = df_mat['days'].iloc[0]

        # Find ATM IV (moneyness closest to 1.0)
        df_mat['dist_from_atm'] = abs(df_mat['moneyness'] - 1.0)
        atm_row = df_mat.loc[df_mat['dist_from_atm'].idxmin()]
        atm_iv = atm_row['iv']

        # Interpolate to get 25-delta put and call IV
        # 25-delta put is typically around 0.92-0.95 moneyness
        # 25-delta call is typically around 1.05-1.08 moneyness

        # For puts (OTM puts have moneyness < 1)
        otm_puts = df_mat[df_mat['moneyness'] < 0.98].sort_values('moneyness')
        if len(otm_puts) >= 2:
            # Target ~0.93 moneyness for 25-delta put
            target_m = 0.93
            closest_put = otm_puts.iloc[(otm_puts['moneyness'] - target_m).abs().argsort()[:1]]
            put_25d_iv = closest_put['iv'].values[0]
        else:
            put_25d_iv = atm_iv

        # For calls (OTM calls have moneyness > 1)
        otm_calls = df_mat[df_mat['moneyness'] > 1.02].sort_values('moneyness')
        if len(otm_calls) >= 2:
            # Target ~1.07 moneyness for 25-delta call
            target_m = 1.07
            closest_call = otm_calls.iloc[(otm_calls['moneyness'] - target_m).abs().argsort()[:1]]
            call_25d_iv = closest_call['iv'].values[0]
        else:
            call_25d_iv = atm_iv

        # Calculate metrics
        skew_25d = put_25d_iv - call_25d_iv  # Risk reversal
        butterfly_25d = (put_25d_iv + call_25d_iv) / 2 - atm_iv  # Butterfly

        # Linear slope of smile
        try:
            coeffs = np.polyfit(df_mat['moneyness'], df_mat['iv'], deg=1)
            skew_slope = coeffs[0]  # Negative slope = put skew
        except:
            skew_slope = 0

        results[maturity] = {
            'days': days,
            'underlying': S,
            'atm_iv': atm_iv,
            'put_25d_iv': put_25d_iv,
            'call_25d_iv': call_25d_iv,
            'skew_25d': skew_25d,
            'butterfly_25d': butterfly_25d,
            'skew_slope': skew_slope
        }

    return results


def load_skew_history():
    """Load historical skew data"""
    if os.path.exists(SKEW_HISTORY_FILE):
        return pd.read_csv(SKEW_HISTORY_FILE, parse_dates=['date'])
    return pd.DataFrame()


def save_skew_history(history_df):
    """Save skew history to file"""
    os.makedirs(os.path.dirname(SKEW_HISTORY_FILE), exist_ok=True)
    history_df.to_csv(SKEW_HISTORY_FILE, index=False)


def update_skew_history(product_code, trade_date, skew_metrics):
    """Add new skew data to history"""
    history = load_skew_history()

    new_rows = []
    for maturity, metrics in skew_metrics.items():
        new_rows.append({
            'date': pd.to_datetime(trade_date, format='%Y%m%d'),
            'product': product_code,
            'maturity': maturity,
            'days': metrics['days'],
            'atm_iv': metrics['atm_iv'],
            'skew_25d': metrics['skew_25d'],
            'butterfly_25d': metrics['butterfly_25d'],
            'skew_slope': metrics['skew_slope']
        })

    new_df = pd.DataFrame(new_rows)

    # Remove duplicates and append
    if not history.empty:
        history = history[~((history['date'] == new_df['date'].iloc[0]) &
                           (history['product'] == product_code))]

    history = pd.concat([history, new_df], ignore_index=True)
    history = history.sort_values(['product', 'date', 'maturity'])

    save_skew_history(history)
    return history


def calculate_skew_percentile(product_code, current_skew, lookback_days=60):
    """Calculate percentile rank of current skew vs history"""
    history = load_skew_history()

    if history.empty:
        return None

    # Filter to product and lookback period
    cutoff_date = datetime.now() - timedelta(days=lookback_days)
    hist = history[(history['product'] == product_code) &
                   (history['date'] >= cutoff_date)]

    if len(hist) < 10:
        return None

    # Use front-month skew for percentile
    front_month = hist.groupby('date').first().reset_index()

    percentile = (front_month['skew_25d'] < current_skew).mean() * 100
    return percentile


def generate_alerts(product_code, skew_metrics, trade_date):
    """Generate alerts based on skew thresholds"""
    alerts = []

    if not skew_metrics:
        return alerts

    # Get front-month metrics
    front_month = min(skew_metrics.keys())
    metrics = skew_metrics[front_month]

    # Check skew percentile
    percentile = calculate_skew_percentile(product_code, metrics['skew_25d'])

    if percentile is not None:
        if percentile >= ALERT_THRESHOLDS['skew_percentile_high']:
            alerts.append({
                'type': 'HIGH_SKEW',
                'product': product_code,
                'message': f"Put skew at {percentile:.0f}th percentile - elevated tail risk pricing",
                'skew': metrics['skew_25d'],
                'percentile': percentile
            })
        elif percentile <= ALERT_THRESHOLDS['skew_percentile_low']:
            alerts.append({
                'type': 'LOW_SKEW',
                'product': product_code,
                'message': f"Put skew at {percentile:.0f}th percentile - potential cheap tail hedge",
                'skew': metrics['skew_25d'],
                'percentile': percentile
            })

    # Check ATM IV spike
    history = load_skew_history()
    if not history.empty:
        recent = history[(history['product'] == product_code) &
                        (history['date'] >= datetime.now() - timedelta(days=20))]
        if len(recent) >= 5:
            avg_atm = recent.groupby('date')['atm_iv'].first().mean()
            if metrics['atm_iv'] > avg_atm * ALERT_THRESHOLDS['atm_iv_spike']:
                alerts.append({
                    'type': 'IV_SPIKE',
                    'product': product_code,
                    'message': f"ATM IV spike: {metrics['atm_iv']:.1f}% vs 20d avg {avg_atm:.1f}%",
                    'current_iv': metrics['atm_iv'],
                    'avg_iv': avg_atm
                })

    return alerts


def plot_skew_history(product_code, save_path=None):
    """Plot historical skew for a product"""
    history = load_skew_history()

    if history.empty:
        print(f"No history for {product_code}")
        return

    hist = history[history['product'] == product_code]
    if hist.empty:
        print(f"No history for {product_code}")
        return

    # Use front-month data
    front_month = hist.groupby('date').first().reset_index()

    fig, axes = plt.subplots(2, 1, figsize=(14, 10), sharex=True)

    # Plot ATM IV
    axes[0].plot(front_month['date'], front_month['atm_iv'],
                 'b-', linewidth=2, label='ATM IV')
    axes[0].fill_between(front_month['date'], front_month['atm_iv'],
                         alpha=0.3)
    axes[0].set_ylabel('ATM IV (%)', fontsize=12)
    axes[0].legend(loc='upper right')
    axes[0].grid(True, alpha=0.3)
    axes[0].set_title(f'{product_code.upper()} - Volatility History', fontsize=14, fontweight='bold')

    # Plot Skew
    colors = ['red' if x > 0 else 'green' for x in front_month['skew_25d']]
    axes[1].bar(front_month['date'], front_month['skew_25d'],
                color=colors, alpha=0.7, label='25-Delta Skew')
    axes[1].axhline(y=0, color='black', linestyle='-', linewidth=0.5)
    axes[1].set_ylabel('25-Delta Skew (%)', fontsize=12)
    axes[1].set_xlabel('Date', fontsize=12)
    axes[1].legend(loc='upper right')
    axes[1].grid(True, alpha=0.3)

    plt.tight_layout()

    if save_path:
        plt.savefig(save_path, dpi=300, bbox_inches='tight')
        print(f"Saved: {save_path}")

    plt.close()


def print_skew_report(product_code, skew_metrics, trade_date):
    """Print formatted skew report"""
    if not skew_metrics:
        print(f"  No skew data for {product_code}")
        return

    print(f"\n{'='*60}")
    print(f"{product_code.upper()} Skew Report - {trade_date}")
    print(f"{'='*60}")

    for maturity in sorted(skew_metrics.keys()):
        m = skew_metrics[maturity]
        print(f"\n  {maturity} ({m['days']}d):")
        print(f"    ATM IV:      {m['atm_iv']:6.2f}%")
        print(f"    25d Put IV:  {m['put_25d_iv']:6.2f}%")
        print(f"    25d Call IV: {m['call_25d_iv']:6.2f}%")
        print(f"    Skew (RR):   {m['skew_25d']:+6.2f}%  {'⚠️ Put premium' if m['skew_25d'] > 2 else ''}")
        print(f"    Butterfly:   {m['butterfly_25d']:+6.2f}%")
        print(f"    Slope:       {m['skew_slope']:+6.2f}")

    # Check for alerts
    alerts = generate_alerts(product_code, skew_metrics, trade_date)
    if alerts:
        print(f"\n  ⚠️  ALERTS:")
        for alert in alerts:
            print(f"    [{alert['type']}] {alert['message']}")


# Asset groupings for tail strategies
ASSET_GROUPS = {
    'real_estate': {
        'name': 'Real Estate / Construction',
        'assets': ['rb', 'fg'],
        'description': 'RB (螺纹钢) and FG (玻璃) as proxies for real estate demand'
    },
    'precious_metals': {
        'name': 'Precious Metals',
        'assets': ['au', 'ag'],
        'description': 'AU (黄金) and AG (白银) as safe haven assets'
    },
    'industrial': {
        'name': 'Industrial Metals',
        'assets': ['cu'],
        'description': 'CU (铜) as global macro indicator'
    },
    'index': {
        'name': 'Index Options',
        'assets': ['50etf', '300etf', '500etf', 'io', 'mo', 'ho'],
        'description': 'Equity index options for market tail risk'
    }
}


def analyze_group_skew(group_name):
    """Analyze skew across an asset group"""
    if group_name not in ASSET_GROUPS:
        print(f"Unknown group: {group_name}")
        return

    group = ASSET_GROUPS[group_name]
    print(f"\n{'='*60}")
    print(f"Group Analysis: {group['name']}")
    print(f"{'='*60}")
    print(f"Assets: {', '.join(group['assets'])}")
    print(f"Description: {group['description']}")

    history = load_skew_history()
    if history.empty:
        print("No historical data available")
        return

    for asset in group['assets']:
        hist = history[history['product'] == asset]
        if hist.empty:
            continue

        recent = hist[hist['date'] == hist['date'].max()]
        if recent.empty:
            continue

        front = recent.iloc[0]
        percentile = calculate_skew_percentile(asset, front['skew_25d'])

        print(f"\n  {asset.upper()}:")
        print(f"    ATM IV: {front['atm_iv']:.2f}%")
        print(f"    Skew:   {front['skew_25d']:+.2f}%", end='')
        if percentile:
            print(f" ({percentile:.0f}th percentile)")
        else:
            print()


if __name__ == '__main__':
    import sys

    print("="*60)
    print("Volatility Skew Analyzer")
    print("="*60)

    # Example usage
    print("\nAsset Groups for Tail Strategies:")
    for group_name, group in ASSET_GROUPS.items():
        print(f"  {group_name}: {group['name']}")
        print(f"    Assets: {', '.join(group['assets'])}")

    print("\nRun with commodity_volatility_smile.py or index_volatility_smile.py")
    print("to generate smile data, then use this analyzer for skew metrics.")
