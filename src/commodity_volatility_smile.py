"""
Commodity Options Volatility Smile Calculator
Calculates and plots implied volatility vs moneyness for commodity options
Supports: 螺纹钢 (RB), 玻璃 (FG), and other commodities
"""

import pandas as pd
import numpy as np
from scipy import optimize
from scipy.stats import norm
import matplotlib.pyplot as plt
import matplotlib
import os
import re
import time
from datetime import datetime

from market_codes import (
    czce_contract_month,
    normalize_option_columns,
    ref_year_from_dates,
)

# Try to use WenQuanYi fonts for better support in Linux/GitHub Actions
matplotlib.rcParams['font.sans-serif'] = ['WenQuanYi Micro Hei', 'WenQuanYi Zen Hei', 'SimHei', 'Arial Unicode MS', 'Microsoft YaHei']
matplotlib.rcParams['axes.unicode_minus'] = False


def black76_call(F, K, T, r, sigma):
    """Black-76 call option price for futures options

    Black-76 is the correct model for options on futures.
    Unlike Black-Scholes, d1 uses only 0.5*sigma^2*T (no drift term).
    """
    if T <= 0 or sigma <= 0:
        return 0
    d1 = (np.log(F / K) + 0.5 * sigma ** 2 * T) / (sigma * np.sqrt(T))
    d2 = d1 - sigma * np.sqrt(T)
    return np.exp(-r * T) * (F * norm.cdf(d1) - K * norm.cdf(d2))


def black76_put(F, K, T, r, sigma):
    """Black-76 put option price for futures options"""
    if T <= 0 or sigma <= 0:
        return 0
    d1 = (np.log(F / K) + 0.5 * sigma ** 2 * T) / (sigma * np.sqrt(T))
    d2 = d1 - sigma * np.sqrt(T)
    return np.exp(-r * T) * (K * norm.cdf(-d2) - F * norm.cdf(-d1))


def implied_volatility(option_price, F, K, T, r, option_type='call'):
    """Calculate implied volatility using Black-76 model and Brent's method

    Args:
        option_price: Market price of the option
        F: Forward/futures price (NOT spot price)
        K: Strike price
        T: Time to expiry in years
        r: Risk-free rate
        option_type: 'call' or 'put'
    """
    if option_price <= 0 or F <= 0 or K <= 0 or T <= 0:
        return np.nan

    def objective(sigma):
        if sigma <= 0:
            return float('inf')
        if option_type == 'call':
            return black76_call(F, K, T, r, sigma) - option_price
        else:
            return black76_put(F, K, T, r, sigma) - option_price

    try:
        return optimize.brentq(objective, 0.01, 5.0, maxiter=200)
    except:
        return np.nan


_FUTURES_CACHE = {}
_TUSHARE_PRO = None


def _tushare_pro():
    """Reuse one Tushare client. Token comes only from the environment."""
    global _TUSHARE_PRO
    if _TUSHARE_PRO is not None:
        return _TUSHARE_PRO
    token = os.environ.get("TUSHARE_TOKEN", "").strip()
    if not token:
        raise RuntimeError("TUSHARE_TOKEN is not set")
    import tushare as ts
    ts.set_token(token)
    _TUSHARE_PRO = ts.pro_api(token)
    return _TUSHARE_PRO


def futures_ts_code(prefix, maturity):
    """Tushare futures code for a commodity month (YYMM)."""
    prefix = str(prefix).lower()
    exchange_map = {
        'rb': 'SHFE', 'ag': 'SHFE', 'au': 'SHFE', 'cu': 'SHFE', 'ru': 'SHFE',
        'zn': 'SHFE', 'pb': 'SHFE', 'ni': 'SHFE', 'sn': 'SHFE', 'al': 'SHFE',
        'ao': 'SHFE', 'fu': 'SHFE', 'bu': 'SHFE', 'sp': 'SHFE', 'br': 'SHFE',
        'fg': 'CZCE', 'sr': 'CZCE', 'cf': 'CZCE', 'ta': 'CZCE', 'ma': 'CZCE',
        'rm': 'CZCE', 'oi': 'CZCE', 'sa': 'CZCE', 'pf': 'CZCE', 'pk': 'CZCE',
        'ur': 'CZCE', 'ap': 'CZCE', 'cj': 'CZCE', 'sf': 'CZCE', 'sm': 'CZCE',
        'px': 'CZCE', 'sh': 'CZCE',
        'i': 'DCE', 'jm': 'DCE', 'j': 'DCE', 'jd': 'DCE', 'm': 'DCE', 'y': 'DCE',
        'a': 'DCE', 'b': 'DCE', 'p': 'DCE', 'c': 'DCE', 'v': 'DCE', 'l': 'DCE',
        'pp': 'DCE', 'eg': 'DCE', 'eb': 'DCE', 'pg': 'DCE', 'lh': 'DCE',
    }
    exchange = exchange_map.get(prefix)
    if exchange is None or not maturity or len(str(maturity)) < 4:
        return None
    maturity = str(maturity)
    if exchange == 'CZCE':
        # CZCE uses 3-digit month: FG605.ZCE
        return f"{prefix.upper()}{maturity[1:]}.ZCE"
    if exchange == 'DCE':
        return f"{prefix.upper()}{maturity}.DCE"
    return f"{prefix.upper()}{maturity}.SHF"


def get_futures_price(prefix, maturity, trade_date):
    """
    Get futures price for the underlying contract from Tushare.

    Returns the close price, or settlement if close is missing. None when the
    contract cannot be resolved or the API has no row for that session.
    """
    ts_code = futures_ts_code(prefix, maturity)
    if ts_code is None:
        return None

    cache_key = (ts_code, str(trade_date))
    if cache_key in _FUTURES_CACHE:
        return _FUTURES_CACHE[cache_key]

    try:
        pro = _tushare_pro()
    except Exception as e:
        print(f"    Warning: Could not fetch futures price for {prefix}{maturity}: {e}")
        _FUTURES_CACHE[cache_key] = None
        return None

    last_error = None
    for attempt in range(3):
        try:
            df = pro.fut_daily(
                ts_code=ts_code,
                start_date=trade_date,
                end_date=trade_date,
                fields='close,settle'
            )
            if df is not None and not df.empty:
                close = df['close'].iloc[0]
                settle = df['settle'].iloc[0] if 'settle' in df.columns else None
                price = close if pd.notna(close) else settle
                if pd.notna(price):
                    price = float(price)
                    _FUTURES_CACHE[cache_key] = price
                    return price
            # An empty book is a real miss, not a transient error.
            _FUTURES_CACHE[cache_key] = None
            return None
        except Exception as e:
            last_error = e
            time.sleep(0.4 * (attempt + 1))

    if last_error is not None:
        print(f"    Warning: Could not fetch futures price for {ts_code}: {last_error}")
    _FUTURES_CACHE[cache_key] = None
    return None


def list_option_files(data_dir, prefix):
    """Monthly and yearly option CSVs for one product. Skips ``*_full`` dumps."""
    files = []
    prefix = prefix.lower()
    for name in sorted(os.listdir(data_dir)):
        lower = name.lower()
        if not lower.endswith('.csv'):
            continue
        if not lower.startswith(f'{prefix}_option_'):
            continue
        if '_full' in lower:
            continue
        files.append(name)
    return files


def load_shfe_data(data_dir, prefix):
    """
    Load SHFE options data with Chinese column names
    Used for: RB (螺纹钢), AG (白银), AU (黄金), CU (铜), etc.
    """
    all_data = []
    for f in list_option_files(data_dir, prefix):
        try:
            df = pd.read_csv(os.path.join(data_dir, f), encoding='utf-8-sig')
            all_data.append(df)
        except Exception as e:
            print(f"  Warning: Could not load {f}: {e}")

    if not all_data:
        return None

    df = pd.concat(all_data, ignore_index=True)
    if '合约代码' in df.columns and 'trade_date' in df.columns:
        df = df.drop_duplicates(subset=['合约代码', 'trade_date'], keep='last')

    # Parse contract code: rb2601C2650 -> maturity=2601, type=C, strike=2650
    def parse_code(code):
        pattern = rf'{prefix}(\d{{4}})([CP])(\d+)'
        match = re.match(pattern, str(code), re.IGNORECASE)
        if match:
            return match.group(1), match.group(2).upper(), int(match.group(3))
        return None, None, None

    parsed = df['合约代码'].apply(parse_code)
    df['maturity'] = parsed.apply(lambda x: x[0])
    df['call_put'] = parsed.apply(lambda x: x[1])
    df['exercise_price'] = parsed.apply(lambda x: x[2])
    df['settle'] = pd.to_numeric(df['结算价'], errors='coerce')
    df['close'] = pd.to_numeric(df['收盘价'], errors='coerce')
    df['volume'] = pd.to_numeric(df['成交量'], errors='coerce')

    return df


def load_dce_data(data_dir, prefix):
    """
    Load DCE options data (大连商品交易所)
    Used for: I (铁矿石), JM (焦煤), etc.
    Format: JM2604-C-1000.DCE or I2604-C-800.DCE
    """
    all_data = []
    for f in list_option_files(data_dir, prefix):
        try:
            df = pd.read_csv(os.path.join(data_dir, f), encoding='utf-8-sig')
            all_data.append(df)
        except Exception as e:
            print(f"  Warning: Could not load {f}: {e}")

    if not all_data:
        return None

    df = pd.concat(all_data, ignore_index=True)
    if 'ts_code' in df.columns and 'trade_date' in df.columns:
        df = df.drop_duplicates(subset=['ts_code', 'trade_date'], keep='last')

    # Parse ts_code: JM2604-C-1000.DCE or I2604-C-800.DCE
    def parse_code(code):
        # Pattern: PREFIX + 4-digit maturity + dash + C/P + dash + strike + .DCE
        pattern = rf'{prefix.upper()}(\d{{4}})-([CP])-(\d+)\.DCE'
        match = re.match(pattern, str(code), re.IGNORECASE)
        if match:
            return match.group(1), match.group(2).upper(), int(match.group(3))
        return None, None, None

    parsed = df['ts_code'].apply(parse_code)
    df['maturity'] = parsed.apply(lambda x: x[0])
    df['call_put'] = parsed.apply(lambda x: x[1])
    df['exercise_price'] = parsed.apply(lambda x: x[2])
    df['settle'] = pd.to_numeric(df['settle'], errors='coerce')
    df['close'] = pd.to_numeric(df['close'], errors='coerce')
    df['volume'] = pd.to_numeric(df['vol'], errors='coerce')

    return df


def load_czce_data(data_dir, prefix):
    """
    Load CZCE options data with English column names (ts_code format)
    Used for: FG (玻璃), SR (白糖), CF (棉花), etc.
    """
    all_data = []
    for f in list_option_files(data_dir, prefix):
        try:
            df = pd.read_csv(os.path.join(data_dir, f))
            all_data.append(df)
        except Exception as e:
            print(f"  Warning: Could not load {f}: {e}")

    if not all_data:
        return None

    df = pd.concat(all_data, ignore_index=True)
    if 'ts_code' in df.columns and 'trade_date' in df.columns:
        df = df.drop_duplicates(subset=['ts_code', 'trade_date'], keep='last')
    df = normalize_option_columns(df)
    ref_year = ref_year_from_dates(df['trade_date'] if 'trade_date' in df.columns else None,
                                   datetime.now().year)

    # Parse ts_code: FG611C1000.ZCE -> maturity=2611, type=C, strike=1000
    # FG701 is July 2027 (2701), not 1701.
    def parse_code(code):
        pattern = rf'{prefix.upper()}(\d{{3}})([CP])(\d+)\.ZCE'
        match = re.match(pattern, str(code), re.IGNORECASE)
        if match:
            maturity = czce_contract_month(match.group(1), ref_year)
            return maturity, match.group(2).upper(), int(match.group(3))
        return None, None, None

    parsed = df['ts_code'].apply(parse_code)
    df['maturity'] = parsed.apply(lambda x: x[0])
    df['call_put'] = parsed.apply(lambda x: x[1])
    df['exercise_price'] = parsed.apply(lambda x: x[2])

    return df


def _product_code_from_frame(df_mat):
    """Ticker from a contract id. Display names are not tickers."""
    for column in ('合约代码', 'ts_code'):
        if column in df_mat.columns and len(df_mat):
            match = re.match(r'([a-zA-Z]+)', str(df_mat[column].iloc[0]))
            if match:
                return match.group(1).lower()
    return None


def calculate_volatility_smile(df, trade_date_str, name, risk_free_rate=0.025, min_volume=50, code=None):
    """
    Calculate volatility smile for a given date

    Args:
        df: DataFrame with options data
        trade_date_str: Date string in YYYYMMDD format
        name: Commodity name for display
        risk_free_rate: Risk-free rate (default 2.5%)
        min_volume: Minimum volume filter (default 50, excludes illiquid options)

    Returns:
        DataFrame with volatility smile data
    """
    df = normalize_option_columns(df)
    trade_date_str = str(trade_date_str)
    df = df[df['trade_date'] == trade_date_str].copy()

    # Filter by minimum volume to exclude illiquid options with stale prices.
    # A missing volume column is not the same as zero volume: CZCE files store
    # size in ``vol``, and dropping those rows used to erase the whole smile.
    if 'volume' in df.columns:
        df = df[df['volume'].fillna(0) >= min_volume]

    if df.empty:
        print(f"  No data for {trade_date_str}")
        return None

    trade_date = datetime.strptime(trade_date_str, '%Y%m%d')
    results = []

    for maturity in sorted(df['maturity'].dropna().unique()):
        df_mat = df[df['maturity'] == maturity].copy()

        # Calculate days to expiry
        # Chinese commodity options expire ~5 trading days before contract month
        # e.g., AG2602 (Feb contract) expires around Jan 26, not Feb 15
        try:
            mat_year = 2000 + int(maturity[:2])
            mat_month = int(maturity[2:])
            # Expiry is around the 26th of the PRIOR month
            if mat_month == 1:
                expiry = datetime(mat_year - 1, 12, 26)
            else:
                expiry = datetime(mat_year, mat_month - 1, 26)
            days = (expiry - trade_date).days
        except:
            continue

        if days <= 5:
            continue

        T = days / 365.0

        # Futures lookup needs the ticker (ag), not the display name (白银).
        # Short Chinese names used to be sent to Tushare as the contract prefix.
        product_code = str(code).lower() if code else _product_code_from_frame(df_mat)

        F = None
        if product_code:
            F = get_futures_price(product_code, maturity, trade_date_str)
            if F:
                print(f"  {name} {maturity}: F={F:.0f} (futures), T={T:.3f} ({days}d)")

        # Fallback: Calculate implied forward from put-call parity
        if F is None:
            call_cols = ['exercise_price', 'close']
            put_cols = ['exercise_price', 'close']
            if 'volume' in df_mat.columns:
                call_cols.append('volume')
                put_cols.append('volume')
            calls = df_mat[df_mat['call_put'] == 'C'][call_cols].copy()
            puts = df_mat[df_mat['call_put'] == 'P'][put_cols].copy()

            merged = calls.merge(puts, on='exercise_price', suffixes=('_c', '_p'))
            merged = merged[(merged['close_c'] > 0) & (merged['close_p'] > 0)]

            if not merged.empty:
                # Proper put-call parity: F = K + (C - P) * exp(r*T)
                merged['F_implied'] = merged['exercise_price'] + (merged['close_c'] - merged['close_p']) * np.exp(risk_free_rate * T)
                F = merged['F_implied'].median()
                print(f"  {name} {maturity}: F={F:.0f} (implied), T={T:.3f} ({days}d)")
            else:
                # Last fallback: use F from nearby maturity or median strike
                if results and len(results) > 0:
                    F = results[-1]['forward']
                    print(f"  {name} {maturity}: F={F:.0f} (prev mat), T={T:.3f} ({days}d)")
                else:
                    F = df_mat['exercise_price'].median()
                    print(f"  {name} {maturity}: F={F:.0f} (median K), T={T:.3f} ({days}d)")

        # Group by strike - calculate IV for BOTH put and call at each strike
        # This allows comparison of put vs call IV to check put-call parity
        strikes = df_mat['exercise_price'].dropna().unique()
        for K in strikes:
            strike_opts = df_mat[df_mat['exercise_price'] == K]
            moneyness = K / F

            if not (0.8 < moneyness < 1.2):
                continue

            # Calculate IV for both call and put at this strike
            call_iv = None
            put_iv = None
            call_vol = 0
            put_vol = 0

            for _, row in strike_opts.iterrows():
                # Use close/trading price (收盘价) instead of settlement price (结算价)
                price = row['close'] if pd.notna(row['close']) and row['close'] > 0 else None
                if price is None or price <= 0:
                    continue

                vol = row['volume'] if 'volume' in row.index and pd.notna(row['volume']) else 0
                if vol < min_volume:
                    continue

                if row['call_put'] == 'C':
                    iv = implied_volatility(price, F, K, T, risk_free_rate, 'call')
                    if pd.notna(iv) and 0.05 < iv < 3.0:
                        call_iv = iv * 100
                        call_vol = vol
                else:
                    iv = implied_volatility(price, F, K, T, risk_free_rate, 'put')
                    if pd.notna(iv) and 0.05 < iv < 3.0:
                        put_iv = iv * 100
                        put_vol = vol

            # Determine which option type to use for the main smile (OTM convention)
            if moneyness < 0.98:
                # Use put (OTM)
                primary_type = 'put'
                primary_iv = put_iv
                primary_vol = put_vol
            elif moneyness > 1.02:
                # Use call (OTM)
                primary_type = 'call'
                primary_iv = call_iv
                primary_vol = call_vol
            else:
                # Near ATM - use the more liquid one
                if call_vol >= put_vol and call_iv is not None:
                    primary_type = 'call'
                    primary_iv = call_iv
                    primary_vol = call_vol
                elif put_iv is not None:
                    primary_type = 'put'
                    primary_iv = put_iv
                    primary_vol = put_vol
                else:
                    continue

            if primary_iv is not None:
                results.append({
                    'trade_date': trade_date_str,
                    'maturity': maturity,
                    'days': days,
                    'strike': K,
                    'forward': F,
                    'moneyness': moneyness,
                    'option_type': primary_type,
                    'iv': primary_iv,
                    'volume': primary_vol,
                    'call_iv': call_iv,
                    'put_iv': put_iv,
                    'call_volume': call_vol,
                    'put_volume': put_vol
                })

    if not results:
        return None

    result_df = pd.DataFrame(results)

    # Filter IV outliers within each maturity using IQR method
    cleaned_results = []
    for mat in result_df['maturity'].unique():
        mat_df = result_df[result_df['maturity'] == mat].copy()
        days = mat_df['days'].iloc[0] if len(mat_df) > 0 else 0

        # Require more data points for short-dated options (more prone to noise)
        min_points = 7 if days < 30 else 5

        if len(mat_df) < min_points:
            print(f"  {mat}: Skipping (only {len(mat_df)} points, need {min_points})")
            continue

        if len(mat_df) >= 5:
            q1 = mat_df['iv'].quantile(0.25)
            q3 = mat_df['iv'].quantile(0.75)
            iqr = q3 - q1
            lower_bound = q1 - 2.0 * iqr  # More lenient for vol smile
            upper_bound = q3 + 2.0 * iqr

            before_count = len(mat_df)
            mat_df = mat_df[(mat_df['iv'] >= lower_bound) & (mat_df['iv'] <= upper_bound)]
            after_count = len(mat_df)

            if before_count > after_count:
                print(f"  {mat}: Removed {before_count - after_count} IV outliers")

            # After filtering, check if we still have enough points
            if len(mat_df) < min_points:
                print(f"  {mat}: Skipping after outlier removal (only {len(mat_df)} points)")
                continue

        cleaned_results.append(mat_df)

    if not cleaned_results:
        return None

    result_df = pd.concat(cleaned_results, ignore_index=True)

    # Filter outlier maturities (entire maturity has bad data)
    median_iv = result_df['iv'].median()
    valid = []
    for mat in result_df['maturity'].unique():
        mat_med = result_df[result_df['maturity'] == mat]['iv'].median()
        if mat_med > median_iv * 0.4:
            valid.append(mat)
        else:
            print(f"  Warning: Excluding {mat} (median IV {mat_med:.2f}% too low)")

    return result_df[result_df['maturity'].isin(valid)]


def plot_volatility_smile(smile_df, name, trade_date_str, save_path):
    """Plot volatility smile curves with polynomial fit"""
    if smile_df is None or smile_df.empty:
        print(f"  No data to plot for {name}")
        return

    fig, ax = plt.subplots(figsize=(14, 8))
    maturities = sorted(smile_df['maturity'].unique())
    colors = plt.cm.viridis(np.linspace(0, 0.9, len(maturities)))

    for idx, mat in enumerate(maturities):
        df_m = smile_df[smile_df['maturity'] == mat]
        days = df_m['days'].iloc[0]

        # Combine calls and puts for fitting
        all_data = df_m.sort_values('moneyness')
        x = all_data['moneyness'].values
        y = all_data['iv'].values

        if len(x) < 3:
            continue

        # Plot scatter points (single marker style - one point per strike)
        ax.scatter(all_data['moneyness'], all_data['iv'], marker='o', color=colors[idx],
                   s=30, alpha=0.5, zorder=5)

        # Fit polynomial curve
        try:
            # Remove outliers using IQR method for better curve fitting
            q1, q3 = np.percentile(y, [25, 75])
            iqr = q3 - q1
            lower_bound = q1 - 1.5 * iqr
            upper_bound = q3 + 1.5 * iqr
            mask = (y >= lower_bound) & (y <= upper_bound)
            x_clean = x[mask]
            y_clean = y[mask]

            if len(x_clean) < 3:
                x_clean, y_clean = x, y

            # Check if the data is essentially flat (range < 1%)
            # If flat, use degree 1 (linear); otherwise use degree 2 (quadratic)
            iv_range = y_clean.max() - y_clean.min()
            fit_degree = 1 if iv_range < 1.0 else 2

            coeffs = np.polyfit(x_clean, y_clean, deg=fit_degree)
            poly = np.poly1d(coeffs)

            # Generate smooth curve
            x_smooth = np.linspace(x.min(), x.max(), 100)
            y_smooth = poly(x_smooth)

            ax.plot(x_smooth, y_smooth, color=colors[idx], linewidth=2.5,
                    label=f"{mat} ({days}d)", alpha=0.9, zorder=10)
        except:
            # Fallback: just connect dots if fitting fails
            ax.plot(x, y, color=colors[idx], linewidth=1.5,
                    label=f"{mat} ({days}d)", alpha=0.7)

    ax.axvline(x=1.0, color='gray', linestyle='--', alpha=0.5, linewidth=1.5, label='ATM')
    ax.set_xlabel('Moneyness (K/F)', fontsize=13, fontweight='bold')
    ax.set_ylabel('Implied Volatility (%)', fontsize=13, fontweight='bold')

    date_fmt = f"{trade_date_str[:4]}-{trade_date_str[4:6]}-{trade_date_str[6:8]}"
    ax.set_title(f'{name}期权波动率微笑 - {date_fmt}',
                 fontsize=14, fontweight='bold', pad=15)
    ax.legend(loc='upper right', fontsize=10, ncol=2)
    ax.grid(True, alpha=0.3, linestyle='--')
    ax.set_xlim(0.85, 1.15)

    # Set Y-axis to show meaningful range (at least 2% spread centered on data)
    all_ivs = smile_df['iv'].values
    iv_min, iv_max = all_ivs.min(), all_ivs.max()
    iv_center = (iv_min + iv_max) / 2
    iv_range = iv_max - iv_min

    # Ensure minimum 2% range for readability
    min_range = 2.0
    if iv_range < min_range:
        y_low = iv_center - min_range / 2
        y_high = iv_center + min_range / 2
    else:
        # Add 10% padding
        padding = iv_range * 0.1
        y_low = iv_min - padding
        y_high = iv_max + padding

    ax.set_ylim(y_low, y_high)

    ax.spines['top'].set_visible(False)
    ax.spines['right'].set_visible(False)

    plt.tight_layout()
    plt.savefig(save_path, dpi=300, bbox_inches='tight')
    print(f"  Saved: {save_path}")
    plt.close()


def print_summary(smile_df, name):
    """Print summary statistics"""
    if smile_df is None or smile_df.empty:
        return

    print(f"\n  {name} Summary:")
    for mat in sorted(smile_df['maturity'].unique()):
        df_m = smile_df[smile_df['maturity'] == mat]
        atm = df_m[abs(df_m['moneyness'] - 1.0) < 0.03]['iv']
        atm_iv = atm.mean() if len(atm) > 0 else df_m['iv'].mean()
        print(f"    {mat} ({df_m['days'].iloc[0]:3d}d): ATM IV={atm_iv:5.2f}%, "
              f"Range={df_m['iv'].min():.2f}%-{df_m['iv'].max():.2f}%")


# Commodity configurations - All available options
COMMODITIES = {
    # ==================== SHFE (上海期货交易所) ====================
    'rb': {'name': '螺纹钢', 'name_en': 'Rebar', 'exchange': 'SHFE', 'loader': load_shfe_data},
    'ag': {'name': '白银', 'name_en': 'Silver', 'exchange': 'SHFE', 'loader': load_shfe_data},
    'au': {'name': '黄金', 'name_en': 'Gold', 'exchange': 'SHFE', 'loader': load_shfe_data},
    'cu': {'name': '铜', 'name_en': 'Copper', 'exchange': 'SHFE', 'loader': load_shfe_data},
    'ru': {'name': '天然橡胶', 'name_en': 'Natural Rubber', 'exchange': 'SHFE', 'loader': load_shfe_data},
    'zn': {'name': '锌', 'name_en': 'Zinc', 'exchange': 'SHFE', 'loader': load_shfe_data},
    'pb': {'name': '铅', 'name_en': 'Lead', 'exchange': 'SHFE', 'loader': load_shfe_data},
    'ni': {'name': '镍', 'name_en': 'Nickel', 'exchange': 'SHFE', 'loader': load_shfe_data},
    'sn': {'name': '锡', 'name_en': 'Tin', 'exchange': 'SHFE', 'loader': load_shfe_data},
    'al': {'name': '铝', 'name_en': 'Aluminum', 'exchange': 'SHFE', 'loader': load_shfe_data},
    'ao': {'name': '氧化铝', 'name_en': 'Alumina', 'exchange': 'SHFE', 'loader': load_shfe_data},
    'fu': {'name': '燃料油', 'name_en': 'Fuel Oil', 'exchange': 'SHFE', 'loader': load_shfe_data},
    'bu': {'name': '沥青', 'name_en': 'Bitumen', 'exchange': 'SHFE', 'loader': load_shfe_data},
    'sp': {'name': '纸浆', 'name_en': 'Pulp', 'exchange': 'SHFE', 'loader': load_shfe_data},
    'br': {'name': '丁二烯橡胶', 'name_en': 'BR Rubber', 'exchange': 'SHFE', 'loader': load_shfe_data},

    # ==================== CZCE (郑州商品交易所) ====================
    'fg': {'name': '玻璃', 'name_en': 'Glass', 'exchange': 'CZCE', 'loader': load_czce_data},
    'sr': {'name': '白糖', 'name_en': 'Sugar', 'exchange': 'CZCE', 'loader': load_czce_data},
    'cf': {'name': '棉花', 'name_en': 'Cotton', 'exchange': 'CZCE', 'loader': load_czce_data},
    'ta': {'name': 'PTA', 'name_en': 'PTA', 'exchange': 'CZCE', 'loader': load_czce_data},
    'ma': {'name': '甲醇', 'name_en': 'Methanol', 'exchange': 'CZCE', 'loader': load_czce_data},
    'rm': {'name': '菜粕', 'name_en': 'Rapeseed Meal', 'exchange': 'CZCE', 'loader': load_czce_data},
    'oi': {'name': '菜油', 'name_en': 'Rapeseed Oil', 'exchange': 'CZCE', 'loader': load_czce_data},
    'sa': {'name': '纯碱', 'name_en': 'Soda Ash', 'exchange': 'CZCE', 'loader': load_czce_data},
    'pf': {'name': '短纤', 'name_en': 'Staple Fiber', 'exchange': 'CZCE', 'loader': load_czce_data},
    'pk': {'name': '花生', 'name_en': 'Peanut', 'exchange': 'CZCE', 'loader': load_czce_data},
    'ur': {'name': '尿素', 'name_en': 'Urea', 'exchange': 'CZCE', 'loader': load_czce_data},
    'ap': {'name': '苹果', 'name_en': 'Apple', 'exchange': 'CZCE', 'loader': load_czce_data},
    'cj': {'name': '红枣', 'name_en': 'Red Date', 'exchange': 'CZCE', 'loader': load_czce_data},
    'sf': {'name': '硅铁', 'name_en': 'Ferrosilicon', 'exchange': 'CZCE', 'loader': load_czce_data},
    'sm': {'name': '锰硅', 'name_en': 'Silicon Manganese', 'exchange': 'CZCE', 'loader': load_czce_data},
    'px': {'name': 'PX', 'name_en': 'PX', 'exchange': 'CZCE', 'loader': load_czce_data},
    'sh': {'name': '烧碱', 'name_en': 'Caustic Soda', 'exchange': 'CZCE', 'loader': load_czce_data},

    # ==================== DCE (大连商品交易所) ====================
    'i': {'name': '铁矿石', 'name_en': 'Iron Ore', 'exchange': 'DCE', 'loader': load_dce_data},
    'jm': {'name': '焦煤', 'name_en': 'Coking Coal', 'exchange': 'DCE', 'loader': load_dce_data},
    'j': {'name': '焦炭', 'name_en': 'Coke', 'exchange': 'DCE', 'loader': load_dce_data},
    'jd': {'name': '鸡蛋', 'name_en': 'Egg', 'exchange': 'DCE', 'loader': load_dce_data},
    'm': {'name': '豆粕', 'name_en': 'Soybean Meal', 'exchange': 'DCE', 'loader': load_dce_data},
    'y': {'name': '豆油', 'name_en': 'Soybean Oil', 'exchange': 'DCE', 'loader': load_dce_data},
    'a': {'name': '豆一', 'name_en': 'Soybean No.1', 'exchange': 'DCE', 'loader': load_dce_data},
    'b': {'name': '豆二', 'name_en': 'Soybean No.2', 'exchange': 'DCE', 'loader': load_dce_data},
    'p': {'name': '棕榈油', 'name_en': 'Palm Oil', 'exchange': 'DCE', 'loader': load_dce_data},
    'c': {'name': '玉米', 'name_en': 'Corn', 'exchange': 'DCE', 'loader': load_dce_data},
    'v': {'name': '聚氯乙烯', 'name_en': 'PVC', 'exchange': 'DCE', 'loader': load_dce_data},
    'l': {'name': '聚乙烯', 'name_en': 'LLDPE', 'exchange': 'DCE', 'loader': load_dce_data},
    'pp': {'name': '聚丙烯', 'name_en': 'PP', 'exchange': 'DCE', 'loader': load_dce_data},
    'eg': {'name': '乙二醇', 'name_en': 'Ethylene Glycol', 'exchange': 'DCE', 'loader': load_dce_data},
    'eb': {'name': '苯乙烯', 'name_en': 'Styrene', 'exchange': 'DCE', 'loader': load_dce_data},
    'pg': {'name': '液化石油气', 'name_en': 'LPG', 'exchange': 'DCE', 'loader': load_dce_data},
    'lh': {'name': '生猪', 'name_en': 'Live Hog', 'exchange': 'DCE', 'loader': load_dce_data},
}


def process_commodity(code, trade_date=None):
    """
    Process a single commodity

    Args:
        code: Commodity code (e.g., 'rb', 'fg')
        trade_date: Optional date string (YYYYMMDD). If None, uses latest.

    Returns:
        DataFrame with volatility smile data
    """
    code = code.lower()
    if code not in COMMODITIES:
        print(f"Unknown commodity: {code}")
        print(f"Available: {list(COMMODITIES.keys())}")
        return None

    config = COMMODITIES[code]

    # Check multiple possible data directories
    # Support DATA_DIR env variable for shared data
    SCRIPT_DIR = os.path.dirname(os.path.abspath(__file__))
    BASE_DIR = os.path.dirname(SCRIPT_DIR)
    data_root = os.environ.get("DATA_DIR", os.path.join(BASE_DIR, "data"))

    possible_dirs = [
        os.path.join(data_root, 'commodity', code),
        os.path.join(BASE_DIR, 'data', 'commodity', code),
        os.path.join(BASE_DIR, 'Commodity', code),
    ]
    data_dir = None
    for d in possible_dirs:
        if os.path.exists(d):
            data_dir = d
            break

    if data_dir is None:
        print(f"Data directory not found: {code}")
        return None

    print(f"\n[{config['name']}] {config['name_en']} ({code.upper()})")

    # Load data
    df = config['loader'](data_dir, code)
    if df is None or df.empty:
        print(f"  No data found")
        return None

    print(f"  Loaded {len(df)} options")

    # Get trade date. CSV loads may store the column as int or str.
    dates = sorted({
        str(d).replace('-', '').removesuffix('.0')
        for d in df['trade_date'].dropna().unique()
    }, reverse=True)
    if trade_date is None:
        trade_date = dates[0]
    else:
        trade_date = str(trade_date).replace('-', '').removesuffix('.0')
        if trade_date not in dates:
            print(f"  Date {trade_date} not found, using {dates[0]}")
            trade_date = dates[0]

    print(f"  Using date: {trade_date}")

    # Calculate smile. Pass the ticker so futures lookup is not inferred from the Chinese name.
    smile_df = calculate_volatility_smile(df, trade_date, config['name'], code=code)

    # Ensure output directories exist
    os.makedirs('output/charts', exist_ok=True)
    os.makedirs('output/data', exist_ok=True)

    # Plot and save
    save_path = f"output/charts/{code}_volatility_smile.png"
    plot_volatility_smile(smile_df, config['name'], trade_date, save_path)

    # Print summary
    print_summary(smile_df, config['name'])

    # Save data
    if smile_df is not None and not smile_df.empty:
        csv_path = f"output/data/{code}_smile_data.csv"
        smile_df.to_csv(csv_path, index=False)
        print(f"  Data saved: {csv_path}")

    return smile_df


if __name__ == '__main__':
    import sys

    print("=" * 60)
    print("Commodity Options Volatility Smile Calculator")
    print("=" * 60)

    # Parse command line arguments
    if len(sys.argv) > 1:
        # Process specific commodities from command line
        codes = sys.argv[1:]
        for code in codes:
            process_commodity(code)
    else:
        # Default: process RB and FG
        process_commodity('rb')
        process_commodity('fg')

    print("\n" + "=" * 60)
    print("Done!")
