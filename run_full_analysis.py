#!/usr/bin/env python3
"""
Full Analysis Runner
Runs volatility smile, skew analysis, and generates comprehensive report

This is the main entry point for GitHub Actions
"""

import os
import sys
from datetime import datetime

# Ensure we're in the right directory
SCRIPT_DIR = os.path.dirname(os.path.abspath(__file__))
os.chdir(SCRIPT_DIR)

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

                    # Plot history if we have enough data
                    try:
                        plot_skew_history(code, f"output/charts/{code}_skew_history.png")
                    except:
                        pass

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
    print("\n[4/4] Generating group analysis...")
    try:
        from skew_analyzer import analyze_group_skew
        for group_name in ['real_estate', 'precious_metals', 'industrial']:
            analyze_group_skew(group_name)
    except Exception as e:
        print(f"Error in group analysis: {e}")

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
