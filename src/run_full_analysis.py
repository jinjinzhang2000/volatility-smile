#!/usr/bin/env python3
"""
Full Analysis Runner
Runs volatility smile, skew analysis, and generates comprehensive report

This is the main entry point for GitHub Actions
"""

import os
import sys
from datetime import datetime
import numpy as np
import pandas as pd
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt

# Try to use WenQuanYi fonts for better support in Linux/GitHub Actions
matplotlib.rcParams['font.sans-serif'] = ['WenQuanYi Micro Hei', 'WenQuanYi Zen Hei', 'SimHei', 'Arial Unicode MS', 'Microsoft YaHei']
matplotlib.rcParams['axes.unicode_minus'] = False

# Ensure we're in the right directory
SCRIPT_DIR = os.path.dirname(os.path.abspath(__file__))
BASE_DIR = os.path.dirname(SCRIPT_DIR)
os.chdir(BASE_DIR)

# Display names
NAMES = {
    'rb': '螺纹钢 RB', 'fg': '玻璃 FG', 'ag': '白银 AG', 'au': '黄金 AU',
    'cu': '铜 CU', 'ru': '橡胶 RU', 'al': '铝 AL', 'zn': '锌 ZN',
    'i': '铁矿 I', 'jm': '焦煤 JM', 'j': '焦炭 J',
    'sr': '白糖 SR', 'cf': '棉花 CF', 'ta': 'PTA', 'ma': '甲醇 MA',
    'pp': 'PP', 'eg': 'EG', 'sa': '纯碱 SA',
    'io': '沪深300 IO', 'mo': '中证1000 MO', 'ho': '上证50 HO',
    '50etf': '50ETF', '300etf': '300ETF', '500etf': '500ETF',
}

def plot_all_smiles(all_smiles, trade_date_str, save_path):
    """Plot all underlyings' vol surface (all maturities) on one figure."""
    smile_data = {}
    for code, df in all_smiles.items():
        if df is None or df.empty:
            continue
        # Keep all maturities with enough data
        valid_mats = []
        for mat in sorted(df['maturity'].unique()):
            sub = df[df['maturity'] == mat]
            if len(sub) >= 3:
                valid_mats.append(mat)
        if valid_mats:
            smile_data[code] = df[df['maturity'].isin(valid_mats)]

    if not smile_data:
        print("  No smile data to plot")
        return

    n = len(smile_data)
    cols = min(4, n)
    rows = (n + cols - 1) // cols

    fig, axes = plt.subplots(rows, cols, figsize=(4.5 * cols, 3.5 * rows))
    if n == 1:
        axes = np.array([axes])
    axes = axes.flatten()

    # Color palette for maturities
    mat_colors = ['#264653', '#2a9d8f', '#e9c46a', '#f4a261', '#e63946', '#6a4c93']

    for idx, (code, df) in enumerate(sorted(smile_data.items())):
        ax = axes[idx]
        maturities = sorted(df['maturity'].unique())

        for mi, mat in enumerate(maturities):
            sub = df[df['maturity'] == mat].sort_values('moneyness')
            x = sub['moneyness'].values
            y = sub['iv'].values
            days = int(sub['days'].iloc[0])
            color = mat_colors[mi % len(mat_colors)]

            ax.scatter(x, y, s=12, alpha=0.4, color=color, zorder=5)

            # Quadratic fit per maturity
            try:
                coeffs = np.polyfit(x, y, min(2, len(x) - 1))
                poly = np.poly1d(coeffs)
                x_sm = np.linspace(max(x.min(), 0.85), min(x.max(), 1.15), 60)
                ax.plot(x_sm, poly(x_sm), color=color, linewidth=1.8,
                        label=f'{mat} ({days}d)', zorder=10)
            except:
                ax.plot(x, y, color=color, linewidth=1.2, label=f'{mat} ({days}d)')

        ax.axvline(1.0, color='gray', linestyle='--', alpha=0.3, linewidth=1)

        name = NAMES.get(code, code.upper())
        ax.set_title(name, fontsize=10, fontweight='bold')
        ax.set_xlim(0.85, 1.15)
        ax.tick_params(labelsize=8)
        ax.grid(True, alpha=0.15)
        ax.spines['top'].set_visible(False)
        ax.spines['right'].set_visible(False)
        ax.legend(fontsize=6, loc='upper center', ncol=min(3, len(maturities)),
                  handlelength=1.5, framealpha=0.6)

    for idx in range(n, len(axes)):
        axes[idx].set_visible(False)

    fig.supxlabel('Moneyness (K/F)', fontsize=12, fontweight='bold')
    fig.supylabel('Implied Volatility (%)', fontsize=12, fontweight='bold')

    date_fmt = f"{trade_date_str[:4]}-{trade_date_str[4:6]}-{trade_date_str[6:8]}"
    fig.suptitle(f'波动率曲面 Volatility Surface — {date_fmt}', fontsize=14, fontweight='bold', y=1.01)
    plt.tight_layout()
    plt.savefig(save_path, dpi=200, bbox_inches='tight')
    print(f"  Saved combined surface chart: {save_path}")
    plt.close()


def plot_skew_summary(all_skew_metrics, trade_date_str, save_path):
    """Plot front-end vs back-end 25-delta skew for all underlyings."""
    rows_list = []
    for code, metrics in all_skew_metrics.items():
        if not metrics:
            continue
        sorted_mats = sorted(metrics.keys())
        if len(sorted_mats) < 1:
            continue

        front_mat = sorted_mats[0]
        front_m = metrics[front_mat]

        # Back-end: use the longest maturity if ≥2 maturities, else same as front
        back_mat = sorted_mats[-1] if len(sorted_mats) >= 2 else sorted_mats[0]
        back_m = metrics[back_mat]

        rows_list.append({
            'code': code,
            'name': NAMES.get(code, code.upper()),
            'front_skew': front_m['skew_25d'],
            'front_days': front_m['days'],
            'front_mat': front_mat,
            'back_skew': back_m['skew_25d'],
            'back_days': back_m['days'],
            'back_mat': back_mat,
            'has_back': len(sorted_mats) >= 2,
        })

    if not rows_list:
        print("  No skew data to plot")
        return

    df = pd.DataFrame(rows_list).sort_values('front_skew')

    n = len(df)
    bar_h = 0.35
    fig, ax = plt.subplots(figsize=(11, max(4, n * 0.6 + 1)))

    y_pos = np.arange(n)

    # Front-end bars
    front_colors = ['#e63946' if s < 0 else '#2a9d8f' for s in df['front_skew']]
    ax.barh(y_pos + bar_h/2, df['front_skew'].values, height=bar_h,
            color=front_colors, alpha=0.85, label='Front-end')

    # Back-end bars
    back_colors = ['#c1121f' if s < 0 else '#1b998b' for s in df['back_skew']]
    ax.barh(y_pos - bar_h/2, df['back_skew'].values, height=bar_h,
            color=back_colors, alpha=0.55, label='Back-end', edgecolor='gray', linewidth=0.5)

    # Labels
    ax.set_yticks(y_pos)
    ax.set_yticklabels([r['name'] for _, r in df.iterrows()], fontsize=10)

    # Value annotations
    for i, (_, r) in enumerate(df.iterrows()):
        # Front label
        s_f = r['front_skew']
        offset = -0.08 if s_f < 0 else 0.08
        ha = 'right' if s_f < 0 else 'left'
        ax.text(s_f + offset, i + bar_h/2, f"{s_f:+.1f}% ({int(r['front_days'])}d)",
                va='center', ha=ha, fontsize=7.5, fontweight='bold', color='#333')

        # Back label (only if different from front)
        if r['has_back']:
            s_b = r['back_skew']
            offset = -0.08 if s_b < 0 else 0.08
            ha = 'right' if s_b < 0 else 'left'
            ax.text(s_b + offset, i - bar_h/2, f"{s_b:+.1f}% ({int(r['back_days'])}d)",
                    va='center', ha=ha, fontsize=7.5, color='#666')

    ax.axvline(0, color='black', linewidth=0.8)
    ax.set_xlabel('25-Delta Skew (Put IV − Call IV) %', fontsize=12, fontweight='bold')

    date_fmt = f"{trade_date_str[:4]}-{trade_date_str[4:6]}-{trade_date_str[6:8]}"
    ax.set_title(f'25-Delta Skew: Front vs Back — {date_fmt}\n'
                 f'(负值 = Put更贵, 正值 = Call更贵)',
                 fontsize=13, fontweight='bold')

    ax.legend(loc='lower right', fontsize=10, framealpha=0.8)
    ax.grid(True, axis='x', alpha=0.2)
    ax.spines['top'].set_visible(False)
    ax.spines['right'].set_visible(False)

    plt.tight_layout()
    plt.savefig(save_path, dpi=200, bbox_inches='tight')
    print(f"  Saved combined skew chart: {save_path}")
    plt.close()

def run_analysis():
    """Run full analysis pipeline"""
    print("="*60)
    print("Full Volatility Analysis Pipeline")
    print(f"Time: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}")
    print("="*60)

    results = {
        'commodity_smiles': {},
        'index_smiles': {},
        'skew_metrics': {},
        'alerts': []
    }

    # 1. Generate commodity volatility smiles
    print("\n[1/4] Generating commodity volatility smiles...")
    try:
        from commodity_volatility_smile import process_commodity, COMMODITIES
        for code in COMMODITIES.keys():
            smile_df = process_commodity(code)
            results['commodity_smiles'][code] = smile_df
    except Exception as e:
        print(f"Error in commodity smile: {e}")

    # 2. Generate index volatility smiles
    print("\n[2/4] Generating index volatility smiles...")
    try:
        from index_volatility_smile import process_all_index_options
        index_results = process_all_index_options()
        results['index_smiles'] = index_results
    except Exception as e:
        print(f"Error in index smile: {e}")

    # 3. Calculate skew metrics and update history
    print("\n[3/4] Calculating skew metrics...")
    try:
        from skew_analyzer import (
            calculate_skew_metrics, update_skew_history,
            print_skew_report, generate_alerts, plot_skew_history,
            ASSET_GROUPS
        )

        all_alerts = []

        # Process commodities
        for code, smile_df in results['commodity_smiles'].items():
            if smile_df is not None and not smile_df.empty:
                skew_metrics = calculate_skew_metrics(smile_df)
                if skew_metrics:
                    trade_date = datetime.now().strftime('%Y%m%d')
                    results['skew_metrics'][code] = skew_metrics
                    update_skew_history(code, trade_date, skew_metrics)
                    print_skew_report(code, skew_metrics, trade_date)

                    # Generate alerts
                    alerts = generate_alerts(code, skew_metrics, trade_date)
                    all_alerts.extend(alerts)



        # Process indices
        for code, smile_df in results['index_smiles'].items():
            if smile_df is not None and not smile_df.empty:
                skew_metrics = calculate_skew_metrics(smile_df)
                if skew_metrics:
                    trade_date = datetime.now().strftime('%Y%m%d')
                    results['skew_metrics'][code] = skew_metrics
                    update_skew_history(code, trade_date, skew_metrics)
                    print_skew_report(code, skew_metrics, trade_date)

                    alerts = generate_alerts(code, skew_metrics, trade_date)
                    all_alerts.extend(alerts)

        results['alerts'] = all_alerts

        # Print all alerts
        if all_alerts:
            print("\n" + "="*60)
            print("⚠️  ALERTS SUMMARY")
            print("="*60)
            for alert in all_alerts:
                print(f"  [{alert['type']}] {alert['product'].upper()}: {alert['message']}")

    except Exception as e:
        print(f"Error in skew analysis: {e}")
        import traceback
        traceback.print_exc()

    # 4. Generate group analysis
    print("\n[4/6] Generating group analysis...")
    try:
        from skew_analyzer import analyze_group_skew
        for group_name in ['real_estate', 'precious_metals', 'industrial']:
            analyze_group_skew(group_name)
    except Exception as e:
        print(f"Error in group analysis: {e}")

    # 5. Generate ATM IV ranking
    print("\n[5/6] Generating ATM IV ranking...")
    try:
        from atm_iv_ranking import main as atm_iv_main
        atm_iv_main()
    except Exception as e:
        print(f"Error in ATM IV ranking: {e}")

    # 6. Generate 25d skew ranking
    print("\n[6/6] Generating 25-delta skew ranking...")
    try:
        from skew_ranking import main as skew_ranking_main
        skew_ranking_main()
    except Exception as e:
        print(f"Error in skew ranking: {e}")

    # 7. Generate combined charts (the two main output charts)
    print("\n[7/7] Generating combined charts...")
    trade_date = datetime.now().strftime('%Y%m%d')

    # Clear old charts so only the 2 combined charts remain
    charts_dir = 'output/charts'
    if os.path.exists(charts_dir):
        import glob as _glob
        for old_png in _glob.glob(os.path.join(charts_dir, '*.png')):
            os.remove(old_png)
    os.makedirs(charts_dir, exist_ok=True)

    # Collect all smile DataFrames
    all_smiles = {}
    all_smiles.update(results.get('commodity_smiles', {}))
    all_smiles.update(results.get('index_smiles', {}))

    try:
        plot_all_smiles(all_smiles, trade_date, 'output/charts/all_volatility_smiles.png')
    except Exception as e:
        print(f"  Error generating smile chart: {e}")
        import traceback; traceback.print_exc()

    try:
        plot_skew_summary(results.get('skew_metrics', {}), trade_date, 'output/charts/all_skew_summary.png')
    except Exception as e:
        print(f"  Error generating skew chart: {e}")
        import traceback; traceback.print_exc()

    print("\n" + "="*60)
    print("Analysis complete!")
    print("="*60)

    return results


def generate_html_report(results):
    """Generate HTML report with all analysis"""
    today = datetime.now().strftime("%Y-%m-%d")

    alerts_html = ""
    if results.get('alerts'):
        alerts_html = "<h3>⚠️ Alerts</h3><ul>"
        for alert in results['alerts']:
            color = 'red' if 'HIGH' in alert['type'] or 'SPIKE' in alert['type'] else 'orange'
            alerts_html += f"<li style='color:{color}'><strong>[{alert['type']}]</strong> {alert['product'].upper()}: {alert['message']}</li>"
        alerts_html += "</ul>"

    skew_html = ""
    if results.get('skew_metrics'):
        skew_html = "<h3>Skew Summary (Front Month)</h3><table border='1' cellpadding='8' style='border-collapse:collapse'>"
        skew_html += "<tr style='background:#4CAF50;color:white'><th>Product</th><th>ATM IV</th><th>25d Skew</th><th>Signal</th></tr>"

        for code, metrics in results['skew_metrics'].items():
            if metrics:
                front_month = min(metrics.keys())
                m = metrics[front_month]
                signal = ""
                if m['skew_25d'] > 3:
                    signal = "🔴 High put premium"
                elif m['skew_25d'] < 0:
                    signal = "🟢 Cheap puts"
                else:
                    signal = "⚪ Normal"

                skew_html += f"<tr><td>{code.upper()}</td><td>{m['atm_iv']:.1f}%</td><td>{m['skew_25d']:+.2f}%</td><td>{signal}</td></tr>"

        skew_html += "</table>"

    html = f"""
    <html>
    <head>
        <style>
            body {{ font-family: Arial, sans-serif; margin: 20px; }}
            h1 {{ color: #333; }}
            h2 {{ color: #666; border-bottom: 1px solid #ddd; padding-bottom: 5px; }}
            h3 {{ color: #888; }}
            table {{ margin: 15px 0; }}
            th, td {{ text-align: left; }}
            .group {{ margin: 20px 0; padding: 15px; background: #f5f5f5; border-radius: 5px; }}
            .alert {{ color: red; font-weight: bold; }}
        </style>
    </head>
    <body>
        <h1>📊 Options Volatility Analysis Report</h1>
        <h2>Date: {today}</h2>

        {alerts_html}

        <h2>Asset Groups</h2>

        <div class="group">
            <h3>🏗️ Real Estate / Construction</h3>
            <p>RB (螺纹钢) and FG (玻璃) - proxies for real estate demand</p>
        </div>

        <div class="group">
            <h3>🥇 Precious Metals</h3>
            <p>AU (黄金) and AG (白银) - safe haven assets</p>
        </div>

        <div class="group">
            <h3>🏭 Industrial</h3>
            <p>CU (铜) - global macro indicator</p>
        </div>

        {skew_html}

        <h2>Products Analyzed</h2>
        <ul>
            <li><strong>Commodity Options:</strong> RB (螺纹钢), FG (玻璃), AG (白银), AU (黄金), CU (铜)</li>
            <li><strong>Index Options:</strong> 50ETF, 300ETF, 500ETF, IO (沪深300), MO (中证1000), HO (上证50)</li>
        </ul>

        <p>Volatility smile and skew history charts are attached.</p>

        <hr>
        <p style="color: #888; font-size: 11px;">
            Generated by Volatility Analysis Pipeline<br>
            Report time: {datetime.now().strftime("%Y-%m-%d %H:%M:%S")} UTC
        </p>
    </body>
    </html>
    """
    return html


if __name__ == '__main__':
    results = run_analysis()

    # Generate HTML report
    html_report = generate_html_report(results)

    # Save report
    os.makedirs('output', exist_ok=True)
    with open('output/report.html', 'w') as f:
        f.write(html_report)
    print(f"\nHTML report saved to output/report.html")
