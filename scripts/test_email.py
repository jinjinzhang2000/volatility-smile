#!/usr/bin/env python3
"""
Test script to verify Resend email service works
Run this locally or in GitHub Actions to test email delivery
"""

import os
import sys
from datetime import datetime

# Configuration
RECIPIENT_EMAIL = "jingjin.zhang@gmail.com"
RESEND_API_KEY = os.environ.get("RESEND_API_KEY", "")

# Fallback: read from config file if env var not set
if not RESEND_API_KEY:
    config_file = os.path.join(os.path.dirname(__file__), "..", "config", ".email_config")
    if os.path.exists(config_file):
        with open(config_file, 'r') as f:
            RESEND_API_KEY = f.read().strip()


def test_email():
    """Send a test email to verify the service works"""
    if not RESEND_API_KEY:
        print("ERROR: RESEND_API_KEY not set!")
        print("Set it via: export RESEND_API_KEY='re_xxxxx'")
        print("Or add it to GitHub Secrets")
        sys.exit(1)

    try:
        import resend
        resend.api_key = RESEND_API_KEY
    except ImportError:
        print("ERROR: resend package not installed. Run: pip install resend")
        sys.exit(1)

    now = datetime.now().strftime("%Y-%m-%d %H:%M:%S")

    html_body = f"""
    <html>
    <body>
        <h1>Email Service Test</h1>
        <p>This is a test email from the Volatility Smile GitHub Action.</p>
        <p><strong>Timestamp:</strong> {now} UTC</p>
        <p>If you receive this email, the email service is working correctly!</p>
        <hr>
        <p style="color: #888; font-size: 11px;">
            Sent from: GitHub Actions test script
        </p>
    </body>
    </html>
    """

    try:
        print(f"Sending test email to {RECIPIENT_EMAIL}...")
        params = {
            "from": "Volatility Report <onboarding@resend.dev>",
            "to": [RECIPIENT_EMAIL],
            "subject": f"[TEST] Email Service Test - {now}",
            "html": html_body,
        }

        email = resend.Emails.send(params)
        print(f"SUCCESS! Email sent to {RECIPIENT_EMAIL}")
        print(f"Email ID: {email.get('id', 'N/A')}")
        return True
    except Exception as e:
        print(f"ERROR sending email: {e}")
        sys.exit(1)


if __name__ == '__main__':
    print("=" * 50)
    print("Testing Resend Email Service")
    print("=" * 50)
    test_email()
    print("=" * 50)
    print("Test complete!")
