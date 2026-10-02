"""Pure helpers for exchange contract codes and option frames.

Kept free of network and plotting imports so download, smile, and tests
can share the same parsing rules.
"""

import re

import pandas as pd


def czce_contract_month(three_digit, ref_year):
    """Map a CZCE 3-digit month code (YMM) to a 4-digit YYMM.

    CZCE prints only the last digit of the year: FG701 is July 2027, not 2017.
    The year digit is resolved to the calendar year closest to ``ref_year``.
    """
    three_digit = str(three_digit)
    if len(three_digit) != 3 or not three_digit.isdigit():
        return None
    month = int(three_digit[1:])
    if not 1 <= month <= 12:
        return None
    year_digit = int(three_digit[0])
    decade = (int(ref_year) // 10) * 10
    candidates = [
        decade - 10 + year_digit,
        decade + year_digit,
        decade + 10 + year_digit,
    ]
    # On an exact tie, prefer the later year (a listed back month, not a long-expired one).
    year = min(candidates, key=lambda y: (abs(y - int(ref_year)), y < int(ref_year)))
    return f"{year % 100:02d}{month:02d}"


def filter_contracts(df, code, column='ts_code'):
    """Keep rows whose contract id starts with ``code`` followed by a digit.

    ``startswith('J')`` also matches JM. Anchoring on a digit keeps coke (J)
    and coking coal (JM) apart, and keeps iron ore (I) from matching anything else.
    """
    if df is None or df.empty or column not in df.columns:
        return df.iloc[0:0].copy() if df is not None else df
    pattern = rf'(?i)^{re.escape(str(code))}\d'
    mask = df[column].astype(str).str.contains(pattern, regex=True, na=False)
    return df.loc[mask].copy()


def classify_etf(name):
    """Return 50ETF / 300ETF / 500ETF, or None for other SSE options.

    500ETF is tested before 50ETF. 科创50 and A50 are not the 50ETF complex.
    """
    text = str(name)
    if '500ETF' in text or '中证500ETF' in text:
        return '500ETF'
    if '300ETF' in text or '沪深300ETF' in text:
        return '300ETF'
    if '科创' in text or 'A50' in text:
        return None
    if '上证50ETF' in text or '50ETF' in text:
        return '50ETF'
    return None


def normalize_call_put(value, name=''):
    """Return 'C' or 'P' from an exchange field or a Chinese contract name."""
    text = str(value).strip().upper()
    if text in ('C', 'CALL'):
        return 'C'
    if text in ('P', 'PUT'):
        return 'P'
    label = str(name)
    if '认购' in label or text in ('认购',):
        return 'C'
    if '认沽' in label or text in ('认沽',):
        return 'P'
    return None


def ref_year_from_dates(series, default_year):
    """Year of the latest trade date in a column, else ``default_year``."""
    if series is None:
        return int(default_year)
    years = []
    for value in series.dropna().astype(str):
        digits = re.sub(r'\D', '', value)
        if len(digits) >= 4:
            years.append(int(digits[:4]))
    return max(years) if years else int(default_year)


def normalize_option_columns(df):
    """Align close / volume / trade_date names across SHFE, DCE, CZCE, and ETF files."""
    if df is None or df.empty:
        return df
    out = df.copy()
    if 'close' not in out.columns and '收盘价' in out.columns:
        out['close'] = pd.to_numeric(out['收盘价'], errors='coerce')
    elif 'close' in out.columns:
        out['close'] = pd.to_numeric(out['close'], errors='coerce')

    if 'volume' not in out.columns:
        if 'vol' in out.columns:
            out['volume'] = pd.to_numeric(out['vol'], errors='coerce')
        elif '成交量' in out.columns:
            out['volume'] = pd.to_numeric(out['成交量'], errors='coerce')
    else:
        out['volume'] = pd.to_numeric(out['volume'], errors='coerce')

    if 'trade_date' in out.columns:
        out['trade_date'] = (
            out['trade_date']
            .astype(str)
            .str.replace(r'\.0$', '', regex=True)
            .str.replace('-', '', regex=False)
        )
    return out


def dominant_trade_date(frames, fallback):
    """Most common trade_date across smile frames, else ``fallback``."""
    dates = []
    for df in frames:
        if df is None or getattr(df, 'empty', True):
            continue
        if 'trade_date' not in df.columns:
            continue
        value = str(df['trade_date'].iloc[0]).replace('-', '')
        if len(value) >= 8 and value[:8].isdigit():
            dates.append(value[:8])
    if not dates:
        return fallback
    return max(set(dates), key=dates.count)
