#!/usr/bin/env python3
"""
Send volatility smile report via email using Resend API
Designed for GitHub Actions
"""

import os
import sys
from datetime import datetime

# Configuration
RECIPIENT_EMAIL = "jingjin.zhang@gmail.com"
RESEND_API_KEY = os.environ.get("RESEND_API_KEY", "")


def create_email_body():
    """Create HTML email body"""
    today = datetime.now().strftime("%Y-%m-%d")

    html = f"""
    <html>
    <head>
        <style>
            body {{ font-family: Arial, sans-serif; }}
            h1 {{ color: #333; }}
            h2 {{ color: #666; }}
            .summary {{ margin: 20px 0; }}
        </style>
    </head>
    <body>
        <h1>Options Volatility Smile Report</h1>
        <h2>Date: {today}</h2>

        <h3>Included Products:</h3>
        <ul>
            <li><strong>Index Options:</strong> 50ETF, 300ETF, 500ETF, IO (沪深300), MO (中证1000), HO (上证50)</li>
            <li><strong>Commodity Options:</strong> RB (螺纹钢), FG (玻璃), AG (白银), AU (黄金), CU (铜)</li>
        </ul>

        <p>Volatility smile charts are attached to this email.</p>

        <hr>
        <p style="color: #888; font-size: 11px;">
            Generated automatically by GitHub Actions<br>
            Report time: {datetime.now().strftime("%Y-%m-%d %H:%M:%S")} UTC
        </p>
    </body>
    </html>
    """
    return html


def send_email():
    """Send email with charts attached via Resend API"""
    if not RESEND_API_KEY:
        print("ERROR: RESEND_API_KEY not set!")
        print("Please add RESEND_API_KEY to GitHub Secrets")
        sys.exit(1)

    try:
        import resend
        resend.api_key = RESEND_API_KEY
    except ImportError:
        print("ERROR: resend package not installed")
        sys.exit(1)

    today = datetime.now().strftime("%Y-%m-%d")

    # Collect chart attachments
    attachments = []
    charts_dir = "output/charts"

    if os.path.exists(charts_dir):
        for img_file in os.listdir(charts_dir):
            if img_file.endswith('.png'):
                img_path = os.path.join(charts_dir, img_file)
                with open(img_path, 'rb') as f:
                    img_data = f.read()
                attachments.append({
                    "filename": img_file,
                    "content": list(img_data),
                })
                print(f"  Attached: {img_file}")

    if not attachments:
        print("WARNING: No charts found to attach")

    # Create HTML body
    html_body = create_email_body()

    # Send via Resend
    try:
        print("Sending email via Resend API...")
        params = {
            "from": "Volatility Report <onboarding@resend.dev>",
            "to": [RECIPIENT_EMAIL],
            "subject": f"Options Volatility Smile Report - {today}",
            "html": html_body,
            "attachments": attachments
        }

        email = resend.Emails.send(params)
        print(f"Email sent successfully to {RECIPIENT_EMAIL}")
        print(f"Email ID: {email.get('id', 'N/A')}")
        return True
    except Exception as e:
        print(f"ERROR sending email: {e}")
        sys.exit(1)


if __name__ == '__main__':
    print("=" * 60)
    print("Sending Volatility Smile Report")
    print("=" * 60)
    send_email()
    print("Done!")
