#!/usr/bin/env python3
"""
Update commodity and index options data via Tushare
Downloads latest data for volatility smile analysis
"""

import tushare as ts
import pandas as pd
import os
import sys
from datetime import datetime, timedelta
import time

# Tushare token - environment only. Do not keep a fallback in the repo.
TUSHARE_TOKEN = os.environ.get("TUSHARE_TOKEN", "").strip()

SCRIPT_DIR = os.path.dirname(os.path.abspath(__file__))
BASE_DIR = os.path.dirname(SCRIPT_DIR)
sys.path.insert(0, os.path.join(BASE_DIR, "src"))
from market_codes import classify_etf, filter_contracts, normalize_call_put  # noqa: E402

# Data directories - support DATA_DIR env variable for shared data
# Default to a local 'data' directory in the project root
DATA_DIR = os.environ.get("DATA_DIR", os.path.join(BASE_DIR, "data"))

COMMODITY_DIR = os.path.join(DATA_DIR, "commodity")
INDEX_OPTIONS_DIR = os.path.join(DATA_DIR, "index_options")

# Today's date
TODAY = datetime.now().strftime("%Y%m%d")

# Get data from last 10 trading days to ensure we have latest
START_DATE = (datetime.now() - timedelta(days=10)).strftime("%Y%m%d")


_OPT_DAILY_CACHE = {}
_OPT_BASIC_CACHE = {}


def init_tushare():
    if not TUSHARE_TOKEN:
        raise RuntimeError("TUSHARE_TOKEN is not set")
    ts.set_token(TUSHARE_TOKEN)
    return ts.pro_api(TUSHARE_TOKEN)


def fetch_exchange_options(pro, exchange):
    """Download one exchange's option book once per run."""
    if exchange in _OPT_DAILY_CACHE:
        return _OPT_DAILY_CACHE[exchange]

    print(f"\nFetching {exchange} option daily ({START_DATE} to {TODAY})...")
    df = pro.opt_daily(
        exchange=exchange,
        start_date=START_DATE,
        end_date=TODAY,
        fields='ts_code,trade_date,pre_settle,pre_close,open,high,low,close,settle,vol,amount,oi'
    )
    if df is None:
        df = pd.DataFrame()
    _OPT_DAILY_CACHE[exchange] = df
    time.sleep(0.4)
    return df


def fetch_opt_basic(pro, exchange):
    """Contract metadata (name, strike, call/put, delist) for one exchange."""
    if exchange in _OPT_BASIC_CACHE:
        return _OPT_BASIC_CACHE[exchange]

    fields = 'ts_code,name,call_put,exercise_price,delist_date,maturity_date'
    try:
        basic = pro.opt_basic(exchange=exchange, fields=fields)
    except Exception as e:
        print(f"  opt_basic with fields failed ({e}); retrying without a field list")
        basic = pro.opt_basic(exchange=exchange)
    if basic is None:
        basic = pd.DataFrame()
    _OPT_BASIC_CACHE[exchange] = basic
    time.sleep(0.3)
    return basic


def _append_new_rows(output_file, df_out, encoding=None):
    """Append rows whose trade_date is not already in the file."""
    os.makedirs(os.path.dirname(output_file), exist_ok=True)
    write_kwargs = {'index': False}
    if encoding:
        write_kwargs['encoding'] = encoding
    read_kwargs = {'encoding': encoding} if encoding else {}

    if os.path.exists(output_file):
        existing = pd.read_csv(output_file, **read_kwargs)
        existing_dates = set(existing['trade_date'].astype(str).unique())
        new_only = df_out[~df_out['trade_date'].astype(str).isin(existing_dates)]
        if new_only.empty:
            print(f"  No new data to add")
            return
        combined = pd.concat([existing, new_only], ignore_index=True)
        combined.to_csv(output_file, **write_kwargs)
        print(f"  Updated: {output_file} (+{len(new_only)} records)")
    else:
        df_out.to_csv(output_file, **write_kwargs)
        print(f"  Created: {output_file}")


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
        df = fetch_exchange_options(pro, 'SHFE')

        if df is None or df.empty:
            print(f"  No data returned from API")
            return None

        # Digit anchor so a short code cannot swallow a longer one
        df = filter_contracts(df, code)

        if df.empty:
            print(f"  No {code.upper()} options found")
            return None

        print(f"  Downloaded {len(df)} records")

        # Convert to Chinese format to match existing data
        df_out = pd.DataFrame()
        df_out['合约代码'] = df['ts_code'].str.replace(r'\.SHF(E)?$', '', regex=True)
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

        _append_new_rows(output_file, df_out, encoding='utf-8-sig')

        return df_out

    except Exception as e:
        print(f"  Error: {e}")
        return None


def download_czce_options(pro, code, name):
    """Download CZCE options (FG, SR, CF, etc.)"""
    print(f"\n[{name}] Downloading {code.upper()} options...")

    try:
        df = fetch_exchange_options(pro, 'CZCE')

        if df is None or df.empty:
            print(f"  No data returned from API")
            return None

        df = filter_contracts(df, code)

        if df.empty:
            print(f"  No {code.upper()} options found")
            return None

        print(f"  Downloaded {len(df)} records")

        # Save in ts_code format
        output_dir = os.path.join(COMMODITY_DIR, code.lower())
        os.makedirs(output_dir, exist_ok=True)

        latest_date = str(df['trade_date'].max())
        month = latest_date[:6]
        output_file = os.path.join(output_dir, f"{code.lower()}_option_{month}.csv")
        _append_new_rows(output_file, df)

        return df

    except Exception as e:
        print(f"  Error: {e}")
        return None


def download_dce_options(pro, code, name):
    """Download DCE options (I, JM, etc.) - 大连商品交易所"""
    print(f"\n[{name}] Downloading {code.upper()} options...")

    try:
        df = fetch_exchange_options(pro, 'DCE')

        if df is None or df.empty:
            print(f"  No data returned from API")
            return None

        # ^I\\d keeps iron ore off other I* products; ^J\\d does not include JM
        df = filter_contracts(df, code)

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

        _append_new_rows(output_file, df)

        return df

    except Exception as e:
        print(f"  Error: {e}")
        return None


def download_index_options(pro, code, name):
    """Download index options (IO, MO, HO)"""
    print(f"\n[{name}] Downloading {code.upper()} options...")

    try:
        df = fetch_exchange_options(pro, 'CFFEX')

        if df is None or df.empty:
            print(f"  No data returned from API")
            return None

        df = filter_contracts(df, code)

        if df.empty:
            print(f"  No {code.upper()} options found")
            return None

        print(f"  Downloaded {len(df)} records")

        # Save to index options directory
        os.makedirs(INDEX_OPTIONS_DIR, exist_ok=True)
        output_file = os.path.join(INDEX_OPTIONS_DIR, f"{code.lower()}_options_daily.csv")

        _append_new_rows(output_file, df)

        return df

    except Exception as e:
        print(f"  Error: {e}")
        return None


def _attach_etf_contract_fields(daily, basic, target_name):
    """Join opt_basic onto daily rows so the smile loader can read strikes."""
    if basic is None or basic.empty or 'name' not in basic.columns:
        return daily.iloc[0:0].copy()

    meta = basic.copy()
    meta['etf_type'] = meta['name'].apply(classify_etf)
    meta = meta[meta['etf_type'] == target_name]
    if meta.empty:
        return daily.iloc[0:0].copy()

    keep = [c for c in ('ts_code', 'name', 'call_put', 'exercise_price', 'delist_date', 'maturity_date')
            if c in meta.columns]
    meta = meta[keep].drop_duplicates('ts_code')
    merged = daily[daily['ts_code'].isin(set(meta['ts_code']))].merge(meta, on='ts_code', how='left')
    if merged.empty:
        return merged

    names = merged['name'] if 'name' in merged.columns else ''
    if 'call_put' in merged.columns:
        merged['call_put'] = [
            normalize_call_put(cp, name)
            for cp, name in zip(merged['call_put'], names)
        ]
    else:
        merged['call_put'] = [normalize_call_put('', name) for name in names]

    if 'delist_date' not in merged.columns and 'maturity_date' in merged.columns:
        merged['delist_date'] = merged['maturity_date']
    return merged


def download_etf_options(pro, etf_code, name, exchange):
    """Download ETF options"""
    print(f"\n[{name}] Downloading ETF options...")

    try:
        basic = fetch_opt_basic(pro, exchange)
        if basic is None or basic.empty:
            print(f"  Could not fetch basic options info for {exchange}")
            return None

        daily = fetch_exchange_options(pro, exchange)
        if daily is None or daily.empty:
            print(f"  No daily data returned from API for {exchange}")
            return None

        # classify_etf keeps 科创50 out of 50ETF and checks 500ETF before 50ETF.
        df = _attach_etf_contract_fields(daily, basic, name)
        if df.empty:
            print(f"  No {name} ({etf_code}) options data found for the date range")
            return None

        print(f"  Downloaded {len(df)} records")

        os.makedirs(INDEX_OPTIONS_DIR, exist_ok=True)
        output_file = os.path.join(INDEX_OPTIONS_DIR, f"etf_{etf_code}_options_daily.csv")
        _append_new_rows(output_file, df)

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

    # Download CZCE commodities (郑州商品交易所)
    czce_commodities = [
        ('fg', '玻璃'),
        ('sr', '白糖'),
    ]

    for code, name in czce_commodities:
        download_czce_options(pro, code, name)

    # Download DCE commodities (大连商品交易所)
    dce_commodities = [
        ('i', '铁矿石'),
        ('jm', '焦煤'),
    ]

    for code, name in dce_commodities:
        download_dce_options(pro, code, name)

    # Download index options (CFFEX)
    index_options = [
        ('IO', '沪深300指数'),
        ('MO', '中证1000指数'),
        ('HO', '上证50指数'),
    ]

    for code, name in index_options:
        download_index_options(pro, code, name)

    # Download ETF options
    etf_options = [
        ('510050', '50ETF', 'SSE'),
        ('510300', '300ETF', 'SSE'),
        ('510500', '500ETF', 'SSE'),
    ]

    for etf_code, name, exchange in etf_options:
        download_etf_options(pro, etf_code, name, exchange)

    print("\n" + "=" * 60)
    print("Update complete!")
    print("=" * 60)


if __name__ == '__main__':
    main()
