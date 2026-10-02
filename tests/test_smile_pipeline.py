"""Regression tests for smile parsing bugs that showed up in the daily report."""

import os
import sys
from datetime import datetime

import pandas as pd

sys.path.insert(0, os.path.join(os.path.dirname(__file__), '..', 'src'))

from market_codes import (  # noqa: E402
    classify_etf,
    czce_contract_month,
    dominant_trade_date,
    filter_contracts,
    normalize_option_columns,
)
import commodity_volatility_smile as smile  # noqa: E402
import index_volatility_smile as index_smile  # noqa: E402


def test_czce_year_digit_uses_the_near_year():
    assert czce_contract_month('611', 2026) == '2611'
    assert czce_contract_month('701', 2026) == '2701'
    assert czce_contract_month('512', 2026) == '2512'
    assert czce_contract_month('001', 2029) == '3001'


def test_contract_filter_does_not_mix_prefixes():
    df = pd.DataFrame({
        'ts_code': [
            'J2601-C-2000.DCE',
            'JM2601-C-1200.DCE',
            'I2601-C-800.DCE',
            'RB2611C3100.SHF',
            'FG611C1100.ZCE',
        ]
    })
    assert list(filter_contracts(df, 'j')['ts_code']) == ['J2601-C-2000.DCE']
    assert list(filter_contracts(df, 'jm')['ts_code']) == ['JM2601-C-1200.DCE']
    assert list(filter_contracts(df, 'i')['ts_code']) == ['I2601-C-800.DCE']
    assert list(filter_contracts(df, 'fg')['ts_code']) == ['FG611C1100.ZCE']


def test_etf_classifier_keeps_50_and_500_apart():
    assert classify_etf('华夏上证50ETF期权2603认购2.63') == '50ETF'
    assert classify_etf('南方中证500ETF期权2603认购5.50') == '500ETF'
    assert classify_etf('华泰柏瑞沪深300ETF期权2603认购3.80') == '300ETF'
    assert classify_etf('华夏科创50ETF期权2603认购1.20') is None


def test_vol_column_is_treated_as_volume():
    raw = pd.DataFrame({
        'close': [1.2],
        'vol': [80],
        'trade_date': [20260924],
    })
    out = normalize_option_columns(raw)
    assert out['volume'].iloc[0] == 80
    assert out['trade_date'].iloc[0] == '20260924'


def test_futures_code_uses_ticker_not_display_name():
    assert smile.futures_ts_code('ag', '2611') == 'AG2611.SHF'
    assert smile.futures_ts_code('fg', '2611') == 'FG611.ZCE'
    assert smile.futures_ts_code('i', '2611') == 'I2611.DCE'
    assert smile.futures_ts_code('白银', '2611') is None


def test_czce_smile_survives_vol_column_and_2027_month(tmp_path, monkeypatch):
    trade_date = '20260924'
    forward = 1000.0
    sigma = 0.25
    r = 0.025
    expiry = datetime(2026, 10, 26)
    asof = datetime.strptime(trade_date, '%Y%m%d')
    t_years = (expiry - asof).days / 365.0

    rows = []
    for strike in range(900, 1120, 20):
        call = smile.black76_call(forward, strike, t_years, r, sigma)
        put = smile.black76_put(forward, strike, t_years, r, sigma)
        rows.append({
            'ts_code': f'FG611C{strike}.ZCE',
            'trade_date': int(trade_date),
            'close': call,
            'settle': call,
            'vol': 200,
        })
        rows.append({
            'ts_code': f'FG611P{strike}.ZCE',
            'trade_date': int(trade_date),
            'close': put,
            'settle': put,
            'vol': 180,
        })
    # A 2027 month must not be decoded as 2017 and dropped as expired.
    rows.append({
        'ts_code': 'FG701C1000.ZCE',
        'trade_date': int(trade_date),
        'close': 50.0,
        'settle': 50.0,
        'vol': 10,
    })

    data_dir = tmp_path / 'fg'
    data_dir.mkdir()
    pd.DataFrame(rows).to_csv(data_dir / 'fg_option_202609.csv', index=False)

    loaded = smile.load_czce_data(str(data_dir), 'fg')
    assert set(loaded['maturity'].dropna()) >= {'2611', '2701'}
    assert loaded['volume'].min() >= 10

    monkeypatch.setattr(smile, 'get_futures_price', lambda *args, **kwargs: forward)
    result = smile.calculate_volatility_smile(
        loaded, trade_date, '玻璃', code='fg', min_volume=50
    )
    assert result is not None and not result.empty
    assert (result['maturity'] == '2611').any()
    assert result['trade_date'].iloc[0] == trade_date
    atm = result.iloc[(result['moneyness'] - 1).abs().argmin()]
    assert abs(atm['iv'] - sigma * 100) < 1.5


def test_short_chinese_name_does_not_become_the_futures_ticker(monkeypatch):
    seen = {}

    def fake_price(prefix, maturity, trade_date):
        seen['prefix'] = prefix
        return 1000.0

    monkeypatch.setattr(smile, 'get_futures_price', fake_price)
    df = pd.DataFrame({
        '合约代码': ['AG2611C1000.SHF', 'AG2611P1000.SHF'],
        'trade_date': ['20260924', '20260924'],
        'close': [30.0, 30.0],
        'volume': [100, 100],
        'maturity': ['2611', '2611'],
        'call_put': ['C', 'P'],
        'exercise_price': [1000, 1000],
    })
    smile.calculate_volatility_smile(df, '20260924', '白银', code='ag', min_volume=1)
    assert seen['prefix'] == 'ag'


def test_history_date_follows_the_smile_not_the_run_clock():
    frame = pd.DataFrame({'trade_date': ['20260924'], 'iv': [20.0]})
    other = pd.DataFrame({'trade_date': ['20260925'], 'iv': [21.0]})
    assert dominant_trade_date([frame, frame, other], '20260926') == '20260924'
    assert dominant_trade_date([pd.DataFrame()], '20260926') == '20260926'


def test_etf_loader_reads_downloader_filenames(tmp_path):
    rows = pd.DataFrame([
        {
            'ts_code': '10001.SH',
            'name': '华夏上证50ETF期权2610认购2.60',
            'call_put': 'C',
            'exercise_price': 2.6,
            'delist_date': '20261028',
            'trade_date': '20260924',
            'close': 0.05,
            'settle': 0.05,
            'vol': 100,
        },
        {
            'ts_code': '10002.SH',
            'name': '华夏科创50ETF期权2610认购1.20',
            'call_put': 'C',
            'exercise_price': 1.2,
            'delist_date': '20261028',
            'trade_date': '20260924',
            'close': 0.04,
            'settle': 0.04,
            'vol': 100,
        },
        {
            'ts_code': '10003.SH',
            'name': '南方中证500ETF期权2610认购5.50',
            'call_put': '认购',
            'exercise_price': 5.5,
            'delist_date': '20261028',
            'trade_date': '20260924',
            'close': 0.08,
            'settle': 0.08,
            'vol': 80,
        },
    ])
    rows.to_csv(tmp_path / 'etf_510050_options_daily.csv', index=False)

    loaded = index_smile.load_sse_etf_options(str(tmp_path))
    assert set(loaded['etf_type']) == {'50ETF', '500ETF'}
    assert (loaded['call_put'] == 'C').all()
    assert loaded['maturity'].iloc[0] == '2610'


def test_etf_loader_reads_abbreviated_tushare_names(tmp_path):
    """Live opt_basic names look like '50ETF购3月2750', not '期权2603认购'."""
    rows = pd.DataFrame([
        {
            'ts_code': '10012435.SH',
            'name': '50ETF购3月2750',
            'call_put': 'C',
            'exercise_price': 2.75,
            'delist_date': '20270324',
            'maturity_date': '20270324',
            'trade_date': '20260924',
            'close': 0.12,
            'settle': 0.12,
            'vol': 100,
        },
        {
            'ts_code': '10012436.SH',
            'name': '科创50沽3月1400',
            'call_put': 'P',
            'exercise_price': 1.4,
            'delist_date': '20270324',
            'trade_date': '20260924',
            'close': 0.08,
            'settle': 0.08,
            'vol': 40,
        },
    ])
    rows.to_csv(tmp_path / 'etf_510050_options_daily.csv', index=False)
    loaded = index_smile.load_sse_etf_options(str(tmp_path))
    assert list(loaded['etf_type']) == ['50ETF']
    assert loaded['maturity'].iloc[0] == '2703'
    assert loaded['exercise_price'].iloc[0] == 2.75
