#!/usr/bin/env python3
"""
25-Delta Skew Ranking for Commodity Options

Calculates 25-delta put-call skew (risk reversal) for ~90 day expiration
and ranks all commodities.

Skew = 25d Put IV - 25d Call IV
- Positive skew: puts more expensive (downside protection premium)
- Negative skew: calls more expensive (upside demand)
"""

import os
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from datetime import datetime

# Configure matplotlib for Chinese fonts - include WenQuanYi for Linux support
plt.rcParams['font.sans-serif'] = ['WenQuanYi Micro Hei', 'WenQuanYi Zen Hei', 'Arial Unicode MS', 'SimHei', 'STHeiti', 'Heiti TC']
plt.rcParams['axes.unicode_minus'] = False

# Commodity configurations with Chinese names
COMMODITIES = {
    # SHFE
    'rb': ('螺纹钢', 'SHFE'),
    'ag': ('白银', 'SHFE'),
    'au': ('黄金', 'SHFE'),
    'cu': ('铜', 'SHFE'),
    'al': ('铝', 'SHFE'),
    'zn': ('锌', 'SHFE'),
    'pb': ('铅', 'SHFE'),
    'ni': ('镍', 'SHFE'),
    'sn': ('锡', 'SHFE'),
    'ru': ('天然橡胶', 'SHFE'),
    'ao': ('氧化铝', 'SHFE'),
    'br': ('丁二烯橡胶', 'SHFE'),
    'fu': ('燃料油', 'SHFE'),
    'bu': ('沥青', 'SHFE'),
    'sp': ('纸浆', 'SHFE'),
    # DCE
    'i': ('铁矿石', 'DCE'),
    'jm': ('焦煤', 'DCE'),
    'j': ('焦炭', 'DCE'),
    'm': ('豆粕', 'DCE'),
    'y': ('豆油', 'DCE'),
    'a': ('豆一', 'DCE'),
    'b': ('豆二', 'DCE'),
    'p': ('棕榈油', 'DCE'),
    'c': ('玉米', 'DCE'),
    'l': ('聚乙烯', 'DCE'),
    'v': ('PVC', 'DCE'),
    'pp': ('聚丙烯', 'DCE'),
    'eg': ('乙二醇', 'DCE'),
    'eb': ('苯乙烯', 'DCE'),
    'pg': ('液化石油气', 'DCE'),
    'lh': ('生猪', 'DCE'),
    'jd': ('鸡蛋', 'DCE'),
    # CZCE
    'fg': ('玻璃', 'CZCE'),
    'sa': ('纯碱', 'CZCE'),
    'sr': ('白糖', 'CZCE'),
    'cf': ('棉花', 'CZCE'),
    'ta': ('PTA', 'CZCE'),
    'ma': ('甲醇', 'CZCE'),
    'rm': ('菜粕', 'CZCE'),
    'oi': ('菜油', 'CZCE'),
    'ap': ('苹果', 'CZCE'),
    'cj': ('红枣', 'CZCE'),
    'pk': ('花生', 'CZCE'),
    'sf': ('硅铁', 'CZCE'),
    'sm': ('锰硅', 'CZCE'),
    'ur': ('尿素', 'CZCE'),
    'px': ('PX', 'CZCE'),
    'sh': ('烧碱', 'CZCE'),
    'pf': ('涤纶短纤', 'CZCE'),
}


def calculate_25d_skew(smile_df, target_days=90):
    """
    Calculate 25-delta skew from smile data.

    Uses contract closest to target_days.
    25d put IV approximated at ~0.93 moneyness
    25d call IV approximated at ~1.07 moneyness

    Returns:
        tuple: (skew, atm_iv, put_25d_iv, call_25d_iv, days) or (None, None, None, None, None)
    """
    if smile_df is None or smile_df.empty:
        return None, None, None, None, None

    # Get unique expiries and find closest to target_days
    unique_days = smile_df['days'].unique()
    closest_days = min(unique_days, key=lambda x: abs(x - target_days))
    target_df = smile_df[smile_df['days'] == closest_days].copy()

    if target_df.empty or len(target_df) < 5:
        return None, None, None, None, None

    # Find ATM IV (moneyness closest to 1.0)
    target_df['atm_dist'] = abs(target_df['moneyness'] - 1.0)
    atm_row = target_df.loc[target_df['atm_dist'].idxmin()]
    atm_iv = atm_row['iv']

    # Find 25-delta put IV (moneyness ~0.93)
    otm_puts = target_df[target_df['moneyness'] < 0.98].copy()
    if len(otm_puts) >= 1:
        otm_puts['put_dist'] = abs(otm_puts['moneyness'] - 0.93)
        put_row = otm_puts.loc[otm_puts['put_dist'].idxmin()]
        put_25d_iv = put_row['iv']
    else:
        put_25d_iv = atm_iv

    # Find 25-delta call IV (moneyness ~1.07)
    otm_calls = target_df[target_df['moneyness'] > 1.02].copy()
    if len(otm_calls) >= 1:
        otm_calls['call_dist'] = abs(otm_calls['moneyness'] - 1.07)
        call_row = otm_calls.loc[otm_calls['call_dist'].idxmin()]
        call_25d_iv = call_row['iv']
    else:
        call_25d_iv = atm_iv

    # 25d skew = Put IV - Call IV (risk reversal)
    skew_25d = put_25d_iv - call_25d_iv

    return skew_25d, atm_iv, put_25d_iv, call_25d_iv, closest_days


def load_all_skews(data_dir, target_days=90):
    """Load all smile data files and calculate 25d skew."""
    results = []

    for code, (name, exchange) in COMMODITIES.items():
        filepath = os.path.join(data_dir, f'{code}_smile_data.csv')
        if not os.path.exists(filepath):
            continue

        try:
            df = pd.read_csv(filepath)
            skew, atm_iv, put_iv, call_iv, days = calculate_25d_skew(df, target_days)

            if skew is not None:
                results.append({
                    'code': code.upper(),
                    'name': name,
                    'exchange': exchange,
                    'skew_25d': skew,
                    'atm_iv': atm_iv,
                    'put_25d_iv': put_iv,
                    'call_25d_iv': call_iv,
                    'days': days,
                })
        except Exception as e:
            print(f"  Error loading {code}: {e}")
            continue

    return pd.DataFrame(results)


def plot_skew_ranking(df, save_path, date_str):
    """Create horizontal bar chart of 25d skew ranking."""
    if df.empty:
        print("No data to plot")
        return

    # Sort by skew (most negative to most positive)
    df = df.sort_values('skew_25d', ascending=True)

    # Color by skew direction
    colors = ['#e74c3c' if s > 0 else '#3498db' for s in df['skew_25d']]

    # Create figure
    fig, ax = plt.subplots(figsize=(12, max(8, len(df) * 0.35)))

    # Create bars
    y_pos = range(len(df))
    bars = ax.barh(y_pos, df['skew_25d'], color=colors, edgecolor='white', linewidth=0.5)

    # Labels
    labels = [f"{row['name']} ({row['code']})" for _, row in df.iterrows()]
    ax.set_yticks(y_pos)
    ax.set_yticklabels(labels)

    # Add value labels on bars
    for i, (bar, skew) in enumerate(zip(bars, df['skew_25d'])):
        x_pos = bar.get_width()
        ha = 'left' if x_pos >= 0 else 'right'
        offset = 0.3 if x_pos >= 0 else -0.3
        ax.text(x_pos + offset, bar.get_y() + bar.get_height()/2,
                f'{skew:+.1f}%', va='center', ha=ha, fontsize=9)

    # Vertical line at zero
    ax.axvline(x=0, color='black', linewidth=0.8, linestyle='-')

    # Formatting
    ax.set_xlabel('25-Delta Skew (Put IV - Call IV, %)', fontsize=12)
    ax.set_title(f'商品期权25Delta偏度排名 (~90天到期)\n25-Delta Skew Ranking ({date_str})',
                 fontsize=14, fontweight='bold')

    # Add legend
    from matplotlib.patches import Patch
    legend_elements = [
        Patch(facecolor='#e74c3c', label='Positive (Put Premium)'),
        Patch(facecolor='#3498db', label='Negative (Call Premium)'),
    ]
    ax.legend(handles=legend_elements, loc='lower right')

    # Grid
    ax.xaxis.grid(True, linestyle='--', alpha=0.7)
    ax.set_axisbelow(True)

    # Adjust layout
    plt.tight_layout()
    plt.savefig(save_path, dpi=150, bbox_inches='tight', facecolor='white')
    plt.close()

    print(f"Saved: {save_path}")


def main():
    """Main function to generate 25d skew ranking."""
    # Paths
    script_dir = os.path.dirname(os.path.abspath(__file__))
    project_dir = os.path.dirname(script_dir)
    data_dir = os.path.join(project_dir, 'output', 'data')
    chart_dir = os.path.join(project_dir, 'output', 'charts')

    os.makedirs(chart_dir, exist_ok=True)

    date_str = datetime.now().strftime('%Y-%m-%d')

    print("=" * 60)
    print(f"25-Delta Skew Ranking - {date_str}")
    print("=" * 60)

    # Load data
    print("\nCalculating 25d skew for ~90 day expiration...")
    df = load_all_skews(data_dir, target_days=90)

    if df.empty:
        print("No smile data found!")
        return

    # Sort and display
    df_sorted = df.sort_values('skew_25d', ascending=False)

    print(f"\nFound {len(df)} commodities with skew data\n")
    print("=" * 70)
    print("25-Delta Skew Ranking (Put IV - Call IV)")
    print("=" * 70)
    print(f"{'Name':10s} {'Code':5s} | {'Skew':>8s} | {'ATM IV':>8s} | {'Put 25d':>8s} | {'Call 25d':>8s} | {'Days':>4s}")
    print("-" * 70)

    for _, row in df_sorted.iterrows():
        print(f"{row['name']:10s} {row['code']:5s} | {row['skew_25d']:>+7.2f}% | {row['atm_iv']:>7.2f}% | {row['put_25d_iv']:>7.2f}% | {row['call_25d_iv']:>7.2f}% | {row['days']:>4d}d")

    # Save CSV
    csv_path = os.path.join(data_dir, 'commodity_skew_ranking.csv')
    df_sorted.to_csv(csv_path, index=False)
    print(f"\nSaved: {csv_path}")

    # Plot
    chart_path = os.path.join(chart_dir, 'commodity_skew_ranking.png')
    plot_skew_ranking(df_sorted, chart_path, date_str)

    # Summary stats
    print("\n" + "=" * 60)
    print("Summary")
    print("=" * 60)
    print(f"Average skew: {df['skew_25d'].mean():+.2f}%")
    print(f"Most positive (put premium): {df_sorted.iloc[0]['name']} ({df_sorted.iloc[0]['skew_25d']:+.2f}%)")
    print(f"Most negative (call premium): {df_sorted.iloc[-1]['name']} ({df_sorted.iloc[-1]['skew_25d']:+.2f}%)")

    print("\nDone!")


if __name__ == '__main__':
    main()
