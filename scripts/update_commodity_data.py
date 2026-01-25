#!/usr/bin/env python3
"""
Update commodity and index options data via Tushare
Downloads latest data for volatility smile analysis
"""

import tushare as ts
import pandas as pd
import os
from datetime import datetime, timedelta
import time

# Tushare token - use environment variable for security
TUSHARE_TOKEN = os.environ.get("TUSHARE_TOKEN", "a70287c82208760b640d7f08525b97181166b817e0d9ff5f8f244bc2")

# Output directories
BASE_DIR = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
COMMODITY_DIR = os.path.join(BASE_DIR, "data", "commodity")
INDEX_OPTIONS_DIR = os.path.join(BASE_DIR, "data", "index_options")

# Today's date
TODAY = datetime.now().strftime("%Y%m%d")

# Get data from last 10 trading days to ensure we have latest
START_DATE = (datetime.now() - timedelta(days=10)).strftime("%Y%m%d")


def init_tushare():
    ts.set_token(TUSHARE_TOKEN)
    return ts.pro_api()


def get_latest_trade_date(pro):
    """Get the most recent trading date"""
    cal = pro.trade_cal(
        exchange='SSE',
        start_date=START_DATE,
        end_date=TODAY,
        is_open='1'
    )
    if not cal.empty:
        return cal['cal_date'].max()
    return TODAY


def download_shfe_options(pro, code, name):
    """Download SHFE options (RB, AG, AU, CU, RU, etc.)"""
    print(f"\n[{name}] Downloading {code.upper()} options...")

    try:
        df = pro.opt_daily(
            exchange='SHFE',
            start_date=START_DATE,
            end_date=TODAY,
            fields='ts_code,trade_date,pre_settle,pre_close,open,high,low,close,settle,vol,amount,oi'
        )

        if df is None or df.empty:
            print(f"  No data returned from API")
            return None

        # Filter by commodity code
        df = df[df['ts_code'].str.upper().str.startswith(code.upper())]

        if df.empty:
            print(f"  No {code.upper()} options found")
            return None

        print(f"  Downloaded {len(df)} records")

        # Convert to Chinese format to match existing data
        df_out = pd.DataFrame()
        df_out['合约代码'] = df['ts_code'].str.replace('.SHFE', '', regex=False)
        df_out['开盘价'] = df['open']
        df_out['最高价'] = df['high']
        df_out['最低价'] = df['low']
        df_out['收盘价'] = df['close']
        df_out['前结算价'] = df['pre_settle']
        df_out['结算价'] = df['settle']
        df_out['涨跌1'] = df['close'] - df['pre_close']
        df_out['涨跌2'] = df['settle'] - df['pre_settle']
        df_out['成交量'] = df['vol']
        df_out['持仓量'] = df['oi']
        df_out['持仓量变化'] = 0
        df_out['成交额'] = df['amount']
        df_out['德尔塔'] = ''
        df_out['行权量'] = 0
        df_out['trade_date'] = df['trade_date']

        # Save to commodity directory
        output_dir = os.path.join(COMMODITY_DIR, code.lower())
        os.makedirs(output_dir, exist_ok=True)

        latest_date = str(df['trade_date'].max())
        month = latest_date[:6]
        output_file = os.path.join(output_dir, f"{code.lower()}_option_{month}.csv")

        if os.path.exists(output_file):
            existing = pd.read_csv(output_file, encoding='utf-8-sig')
            existing_dates = set(existing['trade_date'].astype(str).unique())
            new_only = df_out[~df_out['trade_date'].astype(str).isin(existing_dates)]
            if not new_only.empty:
                combined = pd.concat([existing, new_only], ignore_index=True)
                combined.to_csv(output_file, index=False, encoding='utf-8-sig')
                print(f"  Updated: {output_file} (+{len(new_only)} records)")
            else:
                print(f"  No new data to add")
        else:
            df_out.to_csv(output_file, index=False, encoding='utf-8-sig')
            print(f"  Created: {output_file}")

        return df_out

    except Exception as e:
        print(f"  Error: {e}")
        return None


def download_czce_options(pro, code, name):
    """Download CZCE options (FG, SR, CF, etc.)"""
    print(f"\n[{name}] Downloading {code.upper()} options...")

    try:
        df = pro.opt_daily(
            exchange='CZCE',
            start_date=START_DATE,
            end_date=TODAY,
            fields='ts_code,trade_date,pre_settle,pre_close,open,high,low,close,settle,vol,amount,oi'
        )

        if df is None or df.empty:
            print(f"  No data returned from API")
            return None

        # Filter by commodity code
        df = df[df['ts_code'].str.upper().str.startswith(code.upper())]

        if df.empty:
            print(f"  No {code.upper()} options found")
            return None

        print(f"  Downloaded {len(df)} records")

        # Save in ts_code format
        output_dir = os.path.join(COMMODITY_DIR, code.lower())
        os.makedirs(output_dir, exist_ok=True)

        output_file = os.path.join(output_dir, f"{code.lower()}_option_2026.csv")

        if os.path.exists(output_file):
            existing = pd.read_csv(output_file)
            existing_dates = set(existing['trade_date'].astype(str).unique())
            new_only = df[~df['trade_date'].astype(str).isin(existing_dates)]
            if not new_only.empty:
                combined = pd.concat([existing, new_only], ignore_index=True)
                combined.to_csv(output_file, index=False)
                print(f"  Updated: {output_file} (+{len(new_only)} records)")
            else:
                print(f"  No new data to add")
        else:
            df.to_csv(output_file, index=False)
            print(f"  Created: {output_file}")

        return df

    except Exception as e:
        print(f"  Error: {e}")
        return None


def download_dce_options(pro, code, name):
    """Download DCE options (I, JM, etc.) - 大连商品交易所"""
    print(f"\n[{name}] Downloading {code.upper()} options...")

    try:
        df = pro.opt_daily(
            exchange='DCE',
            start_date=START_DATE,
            end_date=TODAY,
            fields='ts_code,trade_date,pre_settle,pre_close,open,high,low,close,settle,vol,amount,oi'
        )

        if df is None or df.empty:
            print(f"  No data returned from API")
            return None

        # Filter by commodity code
        df = df[df['ts_code'].str.upper().str.startswith(code.upper())]

        if df.empty:
            print(f"  No {code.upper()} options found")
            return None

        print(f"  Downloaded {len(df)} records")

        # Save raw data with ts_code format (DCE format: JM2604-C-1000.DCE)
        output_dir = os.path.join(COMMODITY_DIR, code.lower())
        os.makedirs(output_dir, exist_ok=True)

        latest_date = str(df['trade_date'].max())
        month = latest_date[:6]
        output_file = os.path.join(output_dir, f"{code.lower()}_option_{month}.csv")

        if os.path.exists(output_file):
            existing = pd.read_csv(output_file)
            existing_dates = set(existing['trade_date'].astype(str).unique())
            new_only = df[~df['trade_date'].astype(str).isin(existing_dates)]
            if not new_only.empty:
                combined = pd.concat([existing, new_only], ignore_index=True)
                combined.to_csv(output_file, index=False)
                print(f"  Updated: {output_file} (+{len(new_only)} records)")
            else:
                print(f"  No new data to add")
        else:
            df.to_csv(output_file, index=False)
            print(f"  Created: {output_file}")

        return df

    except Exception as e:
        print(f"  Error: {e}")
        return None


def download_index_options(pro, code, name):
    """Download index options (IO, MO, HO)"""
    print(f"\n[{name}] Downloading {code.upper()} options...")

    try:
        df = pro.opt_daily(
            exchange='CFFEX',
            start_date=START_DATE,
            end_date=TODAY,
            fields='ts_code,trade_date,pre_settle,pre_close,open,high,low,close,settle,vol,amount,oi'
        )

        if df is None or df.empty:
            print(f"  No data returned from API")
            return None

        # Filter by index code
        df = df[df['ts_code'].str.upper().str.startswith(code.upper())]

        if df.empty:
            print(f"  No {code.upper()} options found")
            return None

        print(f"  Downloaded {len(df)} records")

        # Save to index options directory
        os.makedirs(INDEX_OPTIONS_DIR, exist_ok=True)
        output_file = os.path.join(INDEX_OPTIONS_DIR, f"{code.lower()}_options_daily.csv")

        if os.path.exists(output_file):
            existing = pd.read_csv(output_file)
            existing_dates = set(existing['trade_date'].astype(str).unique())
            new_only = df[~df['trade_date'].astype(str).isin(existing_dates)]
            if not new_only.empty:
                combined = pd.concat([existing, new_only], ignore_index=True)
                combined.to_csv(output_file, index=False)
                print(f"  Updated: {output_file} (+{len(new_only)} records)")
            else:
                print(f"  No new data to add")
        else:
            df.to_csv(output_file, index=False)
            print(f"  Created: {output_file}")

        return df

    except Exception as e:
        print(f"  Error: {e}")
        return None


def download_etf_options(pro, etf_code, name, exchange):
    """Download ETF options"""
    print(f"\n[{name}] Downloading ETF options...")

    try:
        df = pro.opt_daily(
            exchange=exchange,
            start_date=START_DATE,
            end_date=TODAY,
            fields='ts_code,trade_date,pre_settle,pre_close,open,high,low,close,settle,vol,amount,oi'
        )

        if df is None or df.empty:
            print(f"  No options found for {etf_code}")
            return None

        # Filter by ETF code
        df = df[df['ts_code'].str.contains(etf_code)]

        if df.empty:
            print(f"  No options found for {etf_code}")
            return None

        print(f"  Downloaded {len(df)} records")

        # Save to index options directory
        os.makedirs(INDEX_OPTIONS_DIR, exist_ok=True)
        output_file = os.path.join(INDEX_OPTIONS_DIR, f"etf_{etf_code}_options_daily.csv")

        if os.path.exists(output_file):
            existing = pd.read_csv(output_file)
            existing_dates = set(existing['trade_date'].astype(str).unique())
            new_only = df[~df['trade_date'].astype(str).isin(existing_dates)]
            if not new_only.empty:
                combined = pd.concat([existing, new_only], ignore_index=True)
                combined.to_csv(output_file, index=False)
                print(f"  Updated: {output_file} (+{len(new_only)} records)")
            else:
                print(f"  No new data to add")
        else:
            df.to_csv(output_file, index=False)
            print(f"  Created: {output_file}")

        return df

    except Exception as e:
        print(f"  Error: {e}")
        return None


def main():
    print("=" * 60)
    print("Options Data Update Tool")
    print(f"Date range: {START_DATE} to {TODAY}")
    print("=" * 60)

    pro = init_tushare()

    latest_date = get_latest_trade_date(pro)
    print(f"Latest trading date: {latest_date}")

    # Download SHFE commodities (上海期货交易所)
    shfe_commodities = [
        ('rb', '螺纹钢'),
        ('ag', '白银'),
        ('au', '黄金'),
        ('cu', '铜'),
        ('ru', '天然橡胶'),
    ]

    for code, name in shfe_commodities:
        download_shfe_options(pro, code, name)
        time.sleep(0.5)

    # Download CZCE commodities (郑州商品交易所)
    czce_commodities = [
        ('fg', '玻璃'),
        ('sr', '白糖'),
    ]

    for code, name in czce_commodities:
        download_czce_options(pro, code, name)
        time.sleep(0.5)

    # Download DCE commodities (大连商品交易所)
    dce_commodities = [
        ('i', '铁矿石'),
        ('jm', '焦煤'),
    ]

    for code, name in dce_commodities:
        download_dce_options(pro, code, name)
        time.sleep(0.5)

    # Download index options (CFFEX)
    index_options = [
        ('IO', '沪深300指数'),
        ('MO', '中证1000指数'),
        ('HO', '上证50指数'),
    ]

    for code, name in index_options:
        download_index_options(pro, code, name)
        time.sleep(0.5)

    # Download ETF options
    etf_options = [
        ('510050', '50ETF', 'SSE'),
        ('510300', '300ETF', 'SSE'),
        ('510500', '500ETF', 'SSE'),
    ]

    for etf_code, name, exchange in etf_options:
        download_etf_options(pro, etf_code, name, exchange)
        time.sleep(0.5)

    print("\n" + "=" * 60)
    print("Update complete!")
    print("=" * 60)


if __name__ == '__main__':
    main()
