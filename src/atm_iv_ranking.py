#!/usr/bin/env python3
"""
ATM Implied Volatility Ranking for Commodity Options

Reads smile data from all commodities and creates a bar chart
ranking them by ATM implied volatility (front month).
"""

import os
import pandas as pd
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


def get_atm_iv(smile_df, target_days=90):
    """
    Extract ATM IV from smile data.
    Uses contract closest to target_days (default 90) with moneyness closest to 1.0.
    """
    if smile_df is None or smile_df.empty:
        return None, None

    # Get unique expiries and find closest to target_days
    unique_days = smile_df['days'].unique()
    closest_days = min(unique_days, key=lambda x: abs(x - target_days))
    target_df = smile_df[smile_df['days'] == closest_days].copy()

    if target_df.empty:
        return None, None

    # Find strike closest to ATM (moneyness = 1.0)
    target_df['atm_dist'] = abs(target_df['moneyness'] - 1.0)
    atm_row = target_df.loc[target_df['atm_dist'].idxmin()]

    return atm_row['iv'], closest_days  # Already in percentage


def load_smile_data(data_dir):
    """Load all smile data files and extract ATM IVs."""
    results = []

    for code, (name, exchange) in COMMODITIES.items():
        filepath = os.path.join(data_dir, f'{code}_smile_data.csv')
        if not os.path.exists(filepath):
            continue

        try:
            df = pd.read_csv(filepath)
            atm_iv, days = get_atm_iv(df)

            if atm_iv is not None:
                results.append({
                    'code': code.upper(),
                    'name': name,
                    'exchange': exchange,
                    'atm_iv': atm_iv,
                    'days': days,
                })
        except Exception as e:
            print(f"  Error loading {code}: {e}")
            continue

    return pd.DataFrame(results)


def plot_atm_iv_ranking(df, save_path, date_str):
    """Create horizontal bar chart of ATM IV ranking."""
    if df.empty:
        print("No data to plot")
        return

    # Sort by ATM IV descending
    df = df.sort_values('atm_iv', ascending=True)

    # Color by exchange
    colors = {
        'SHFE': '#e74c3c',  # Red
        'DCE': '#3498db',   # Blue
        'CZCE': '#2ecc71',  # Green
    }
    bar_colors = [colors.get(ex, '#95a5a6') for ex in df['exchange']]

    # Create figure
    fig, ax = plt.subplots(figsize=(12, max(8, len(df) * 0.35)))

    # Create bars
    y_pos = range(len(df))
    bars = ax.barh(y_pos, df['atm_iv'], color=bar_colors, edgecolor='white', linewidth=0.5)

    # Labels
    labels = [f"{row['name']} ({row['code']})" for _, row in df.iterrows()]
    ax.set_yticks(y_pos)
    ax.set_yticklabels(labels)

    # Add value labels on bars
    for i, (bar, iv) in enumerate(zip(bars, df['atm_iv'])):
        ax.text(bar.get_width() + 0.5, bar.get_y() + bar.get_height()/2,
                f'{iv:.1f}%', va='center', fontsize=9)

    # Formatting
    ax.set_xlabel('ATM Implied Volatility (%)', fontsize=12)
    ax.set_title(f'商品期权ATM隐含波动率排名\nCommodity Options ATM IV Ranking ({date_str})',
                 fontsize=14, fontweight='bold')

    # Add legend for exchanges
    from matplotlib.patches import Patch
    legend_elements = [
        Patch(facecolor=colors['SHFE'], label='SHFE (上期所)'),
        Patch(facecolor=colors['DCE'], label='DCE (大商所)'),
        Patch(facecolor=colors['CZCE'], label='CZCE (郑商所)'),
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
    """Main function to generate ATM IV ranking."""
    # Paths
    script_dir = os.path.dirname(os.path.abspath(__file__))
    project_dir = os.path.dirname(script_dir)
    data_dir = os.path.join(project_dir, 'output', 'data')
    chart_dir = os.path.join(project_dir, 'output', 'charts')

    os.makedirs(chart_dir, exist_ok=True)

    date_str = datetime.now().strftime('%Y-%m-%d')

    print("=" * 60)
    print(f"ATM IV Ranking - {date_str}")
    print("=" * 60)

    # Load data
    print("\nLoading smile data...")
    df = load_smile_data(data_dir)

    if df.empty:
        print("No smile data found!")
        return

    # Sort and display
    df_sorted = df.sort_values('atm_iv', ascending=False)

    print(f"\nFound {len(df)} commodities with ATM IV data\n")
    print("=" * 60)
    print("ATM IV Ranking (Front Month)")
    print("=" * 60)

    for i, row in df_sorted.iterrows():
        print(f"{row['name']:8s} ({row['code']:3s}) | ATM IV: {row['atm_iv']:6.2f}% | {row['days']:3d}d | {row['exchange']}")

    # Save CSV
    csv_path = os.path.join(data_dir, 'commodity_atm_iv_ranking.csv')
    df_sorted.to_csv(csv_path, index=False)
    print(f"\nSaved: {csv_path}")

    # Plot
    chart_path = os.path.join(chart_dir, 'commodity_atm_iv_ranking.png')
    plot_atm_iv_ranking(df_sorted, chart_path, date_str)

    print("\nDone!")


if __name__ == '__main__':
    main()
