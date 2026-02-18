"""
Index Options Volatility Smile Calculator
Calculates and plots implied volatility vs moneyness for index options

Supports:
- SSE ETF Options: 50ETF, 300ETF, 500ETF
- CFFEX Index Options: IO (沪深300), MO (中证1000), HO (上证50)
"""

import pandas as pd
import numpy as np
from scipy import optimize
from scipy.stats import norm
import matplotlib.pyplot as plt
import matplotlib
import os
import re
from datetime import datetime

# Try to use WenQuanYi fonts for better support in Linux/GitHub Actions
matplotlib.rcParams['font.sans-serif'] = ['WenQuanYi Micro Hei', 'WenQuanYi Zen Hei', 'SimHei', 'Arial Unicode MS', 'Microsoft YaHei']
matplotlib.rcParams['axes.unicode_minus'] = False


def black76_call(F, K, T, r, sigma):
    """Black-76 call option price (forward-based)"""
    if T <= 0 or sigma <= 0:
        return 0
    d1 = (np.log(F / K) + 0.5 * sigma ** 2 * T) / (sigma * np.sqrt(T))
    d2 = d1 - sigma * np.sqrt(T)
    return np.exp(-r * T) * (F * norm.cdf(d1) - K * norm.cdf(d2))


def black76_put(F, K, T, r, sigma):
    """Black-76 put option price (forward-based)"""
    if T <= 0 or sigma <= 0:
        return 0
    d1 = (np.log(F / K) + 0.5 * sigma ** 2 * T) / (sigma * np.sqrt(T))
    d2 = d1 - sigma * np.sqrt(T)
    return np.exp(-r * T) * (K * norm.cdf(-d2) - F * norm.cdf(-d1))


def implied_volatility_black76(option_price, F, K, T, r, option_type='call'):
    """Calculate implied volatility using Black-76 model and Brent's method"""
    if option_price <= 0 or F <= 0 or K <= 0 or T <= 0:
        return np.nan

    # Check for arbitrage bounds
    discount = np.exp(-r * T)
    if option_type == 'call':
        intrinsic = max(0, discount * (F - K))
        max_price = discount * F
    else:
        intrinsic = max(0, discount * (K - F))
        max_price = discount * K

    if option_price < intrinsic * 0.99 or option_price > max_price * 1.01:
        return np.nan

    def objective(sigma):
        if sigma <= 0:
            return float('inf')
        if option_type == 'call':
            return black76_call(F, K, T, r, sigma) - option_price
        else:
            return black76_put(F, K, T, r, sigma) - option_price

    try:
        return optimize.brentq(objective, 0.001, 5.0, maxiter=200)
    except:
        return np.nan


def calculate_implied_forward(calls_df, puts_df, T, r):
    """
    Calculate implied forward price from put-call parity
    F = K + (C - P) * exp(r * T)

    Returns median forward across all strike pairs
    """
    merged = calls_df.merge(puts_df, on='strike', suffixes=('_c', '_p'))
    merged = merged[(merged['price_c'] > 0) & (merged['price_p'] > 0)]

    if merged.empty:
        return None

    # Implied forward from put-call parity: C - P = (F - K) * exp(-rT)
    # So: F = K + (C - P) * exp(rT)
    merged['F_implied'] = merged['strike'] + (merged['price_c'] - merged['price_p']) * np.exp(r * T)

    # Use median to be robust to outliers
    return merged['F_implied'].median()


def load_sse_etf_options(data_dir):
    """
    Load SSE ETF options data
    Handles: 50ETF, 300ETF, 500ETF
    """
    all_data = []
    for f in sorted(os.listdir(data_dir), reverse=True):
        # Support both old format (sse_etf_option_*) and new format (sse_etf_options_daily)
        if (f.startswith('sse_etf_option') and f.endswith('.csv')):
            try:
                df = pd.read_csv(os.path.join(data_dir, f))
                all_data.append(df)
            except Exception as e:
                print(f"  Warning: Could not load {f}: {e}")

    if not all_data:
        return None

    df = pd.concat(all_data, ignore_index=True)

    # Identify ETF type from name
    def get_etf_type(name):
        if '上证50ETF' in str(name) or '50ETF' in str(name):
            return '50ETF'
        elif '沪深300ETF' in str(name) or '300ETF' in str(name):
            return '300ETF'
        elif '中证500ETF' in str(name) or '500ETF' in str(name):
            return '500ETF'
        return None

    df['etf_type'] = df['name'].apply(get_etf_type)
    df = df[df['etf_type'].notna()]

    # Parse maturity from name (e.g., "华夏上证50ETF期权2603认购2.63")
    def parse_maturity(name):
        match = re.search(r'期权(\d{4})', str(name))
        if match:
            return match.group(1)
        return None

    df['maturity'] = df['name'].apply(parse_maturity)

    return df


def load_cffex_index_options(data_dir, prefix):
    """
    Load CFFEX index options data
    prefix: 'io' for 沪深300, 'mo' for 中证1000, 'ho' for 上证50
    """
    all_data = []
    for f in sorted(os.listdir(data_dir), reverse=True):
        # Support both old format (io_option_*) and new format (io_options_daily)
        if f.startswith(f'{prefix}_option') and f.endswith('.csv'):
            try:
                df = pd.read_csv(os.path.join(data_dir, f))
                all_data.append(df)
            except Exception as e:
                print(f"  Warning: Could not load {f}: {e}")

    if not all_data:
        return None

    df = pd.concat(all_data, ignore_index=True)

    # Parse ts_code: IO2602-C-4000.CFX -> maturity=2602, type=C, strike=4000
    def parse_code(code):
        pattern = rf'{prefix.upper()}(\d{{4}})-([CP])-(\d+)\.CFX'
        match = re.match(pattern, str(code))
        if match:
            return match.group(1), match.group(2), float(match.group(3))
        return None, None, None

    parsed = df['ts_code'].apply(parse_code)
    df['maturity'] = parsed.apply(lambda x: x[0])
    df['call_put'] = parsed.apply(lambda x: x[1])
    df['strike'] = parsed.apply(lambda x: x[2])

    return df


def calculate_etf_volatility_smile(df, etf_type, trade_date_str, risk_free_rate=0.025):
    """
    Calculate volatility smile for ETF options using Black-76 model

    Args:
        df: DataFrame with ETF options data
        etf_type: '50ETF', '300ETF', or '500ETF'
        trade_date_str: Date string in YYYYMMDD format
        risk_free_rate: Risk-free rate (default 2.5%)

    Returns:
        DataFrame with volatility smile data
    """
    df = df[(df['etf_type'] == etf_type) & (df['trade_date'] == int(trade_date_str))].copy()

    if df.empty:
        print(f"  No data for {etf_type} on {trade_date_str}")
        return None

    trade_date = datetime.strptime(trade_date_str, '%Y%m%d')
    results = []

    for maturity in sorted(df['maturity'].dropna().unique()):
        df_mat = df[df['maturity'] == maturity].copy()

        # Calculate days to expiry from delist_date
        try:
            delist_date_str = str(df_mat['delist_date'].iloc[0])
            expiry = datetime.strptime(delist_date_str, '%Y%m%d')
            days = (expiry - trade_date).days
        except:
            # Fallback: approximate expiry as 4th Wednesday of maturity month
            try:
                mat_year = 2000 + int(maturity[:2])
                mat_month = int(maturity[2:])
                expiry = datetime(mat_year, mat_month, 22)
                days = (expiry - trade_date).days
            except:
                continue

        if days <= 5:
            continue

        T = days / 365.0

        # Get calls and puts
        calls = df_mat[df_mat['call_put'] == 'C'][['exercise_price', 'settle', 'close']].copy()
        puts = df_mat[df_mat['call_put'] == 'P'][['exercise_price', 'settle', 'close']].copy()

        # Use close (trading price), fallback to settle
        calls['price'] = calls['close'].fillna(calls['settle'])
        puts['price'] = puts['close'].fillna(puts['settle'])

        calls = calls[calls['price'] > 0]
        puts = puts[puts['price'] > 0]

        # Rename for forward calculation
        calls_for_fwd = calls.rename(columns={'exercise_price': 'strike'})
        puts_for_fwd = puts.rename(columns={'exercise_price': 'strike'})

        # Calculate implied forward using put-call parity
        F = calculate_implied_forward(calls_for_fwd, puts_for_fwd, T, risk_free_rate)

        if F is None or F <= 0:
            continue

        print(f"  {etf_type} {maturity}: F={F:.4f}, T={T:.3f} ({days}d)")

        # Group by strike - only use the more liquid contract (call or put) at each strike
        strikes = df_mat['exercise_price'].dropna().unique()
        for K in strikes:
            strike_opts = df_mat[df_mat['exercise_price'] == K]

            # Find the most liquid option at this strike (use vol column)
            best_opt = None
            best_vol = -1
            for _, row in strike_opts.iterrows():
                price = row['close'] if pd.notna(row['close']) and row['close'] > 0 else row['settle']
                if pd.isna(price) or price <= 0:
                    continue
                vol = row.get('vol', 0) if 'vol' in row else 0
                if pd.isna(vol):
                    vol = 0
                if vol > best_vol:
                    best_vol = vol
                    best_opt = row

            if best_opt is None:
                continue

            price = best_opt['close'] if pd.notna(best_opt['close']) and best_opt['close'] > 0 else best_opt['settle']
            opt_type = 'call' if best_opt['call_put'] == 'C' else 'put'
            iv = implied_volatility_black76(price, F, K, T, risk_free_rate, opt_type)

            if pd.notna(iv) and 0.05 < iv < 3.0:
                moneyness = K / F
                if 0.8 < moneyness < 1.2:
                    results.append({
                        'maturity': maturity,
                        'days': days,
                        'strike': K,
                        'forward': F,
                        'moneyness': moneyness,
                        'option_type': opt_type,
                        'iv': iv * 100,
                        'volume': best_vol
                    })

    if not results:
        return None

    return pd.DataFrame(results)


def calculate_index_volatility_smile(df, name, trade_date_str, risk_free_rate=0.025):
    """
    Calculate volatility smile for CFFEX index options using Black-76 model

    Args:
        df: DataFrame with CFFEX index options data
        name: Index name for display
        trade_date_str: Date string in YYYYMMDD format
        risk_free_rate: Risk-free rate (default 2.5%)

    Returns:
        DataFrame with volatility smile data
    """
    df = df[df['trade_date'] == int(trade_date_str)].copy()

    if df.empty:
        print(f"  No data for {name} on {trade_date_str}")
        return None

    trade_date = datetime.strptime(trade_date_str, '%Y%m%d')
    results = []

    for maturity in sorted(df['maturity'].dropna().unique()):
        df_mat = df[df['maturity'] == maturity].copy()

        # Calculate days to expiry from delist_date
        try:
            delist_date_str = str(df_mat['delist_date'].iloc[0])
            expiry = datetime.strptime(delist_date_str, '%Y%m%d')
            days = (expiry - trade_date).days
        except:
            # Fallback: 3rd Friday of maturity month
            try:
                mat_year = 2000 + int(maturity[:2])
                mat_month = int(maturity[2:])
                expiry = datetime(mat_year, mat_month, 20)
                days = (expiry - trade_date).days
            except:
                continue

        if days <= 5:
            continue

        T = days / 365.0

        # Get calls and puts
        calls = df_mat[df_mat['call_put'] == 'C'][['strike', 'settle', 'close']].copy()
        puts = df_mat[df_mat['call_put'] == 'P'][['strike', 'settle', 'close']].copy()

        # Use close (trading price), fallback to settle
        calls['price'] = calls['close'].fillna(calls['settle'])
        puts['price'] = puts['close'].fillna(puts['settle'])

        calls = calls[calls['price'] > 0]
        puts = puts[puts['price'] > 0]

        # Calculate implied forward using put-call parity
        F = calculate_implied_forward(calls, puts, T, risk_free_rate)

        if F is None or F <= 0:
            continue

        print(f"  {name} {maturity}: F={F:.2f}, T={T:.3f} ({days}d)")

        # Group by strike - only use the more liquid contract (call or put) at each strike
        strikes = df_mat['strike'].dropna().unique()
        for K in strikes:
            strike_opts = df_mat[df_mat['strike'] == K]

            # Find the most liquid option at this strike (use vol column)
            best_opt = None
            best_vol = -1
            for _, row in strike_opts.iterrows():
                price = row['close'] if pd.notna(row['close']) and row['close'] > 0 else row['settle']
                if pd.isna(price) or price <= 0:
                    continue
                vol = row.get('vol', 0) if 'vol' in row else 0
                if pd.isna(vol):
                    vol = 0
                if vol > best_vol:
                    best_vol = vol
                    best_opt = row

            if best_opt is None:
                continue

            price = best_opt['close'] if pd.notna(best_opt['close']) and best_opt['close'] > 0 else best_opt['settle']
            opt_type = 'call' if best_opt['call_put'] == 'C' else 'put'
            iv = implied_volatility_black76(price, F, K, T, risk_free_rate, opt_type)

            if pd.notna(iv) and 0.05 < iv < 3.0:
                moneyness = K / F
                if 0.8 < moneyness < 1.2:
                    results.append({
                        'maturity': maturity,
                        'days': days,
                        'strike': K,
                        'forward': F,
                        'moneyness': moneyness,
                        'option_type': opt_type,
                        'iv': iv * 100,
                        'volume': best_vol
                    })

    if not results:
        return None

    return pd.DataFrame(results)


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

        # Fit polynomial curve (degree 2 for smile shape)
        try:
            # Remove outliers using IQR method for better curve fitting
            q1, q3 = np.percentile(y, [25, 75])
            iqr = q3 - q1
            lower_bound = q1 - 1.5 * iqr
            upper_bound = q3 + 1.5 * iqr
            mask = (y >= lower_bound) & (y <= upper_bound)
            x_clean = x[mask]
            y_clean = y[mask]

            if len(x_clean) >= 3:
                coeffs = np.polyfit(x_clean, y_clean, deg=2)
                poly = np.poly1d(coeffs)
                x_smooth = np.linspace(x.min(), x.max(), 100)
                y_smooth = poly(x_smooth)
                ax.plot(x_smooth, y_smooth, color=colors[idx], linewidth=2.5,
                        label=f"{mat} ({days}d)", alpha=0.9, zorder=10)
            else:
                coeffs = np.polyfit(x, y, deg=2)
                poly = np.poly1d(coeffs)
                x_smooth = np.linspace(x.min(), x.max(), 100)
                y_smooth = poly(x_smooth)
                ax.plot(x_smooth, y_smooth, color=colors[idx], linewidth=2.5,
                        label=f"{mat} ({days}d)", alpha=0.9, zorder=10)
        except:
            ax.plot(x, y, color=colors[idx], linewidth=1.5,
                    label=f"{mat} ({days}d)", alpha=0.7)

    ax.axvline(x=1.0, color='gray', linestyle='--', alpha=0.5, linewidth=1.5, label='ATM')
    ax.set_xlabel('Moneyness (K/S)', fontsize=13, fontweight='bold')
    ax.set_ylabel('Implied Volatility (%)', fontsize=13, fontweight='bold')

    date_fmt = f"{trade_date_str[:4]}-{trade_date_str[4:6]}-{trade_date_str[6:8]}"
    ax.set_title(f'{name}期权波动率微笑 - {date_fmt}',
                 fontsize=14, fontweight='bold', pad=15)
    ax.legend(loc='upper right', fontsize=10, ncol=2)
    ax.grid(True, alpha=0.3, linestyle='--')
    ax.set_xlim(0.85, 1.15)
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


# Product configurations
INDEX_OPTIONS = {
    '50etf': {
        'name': '上证50ETF',
        'type': 'etf',
        'etf_type': '50ETF'
    },
    '300etf': {
        'name': '沪深300ETF',
        'type': 'etf',
        'etf_type': '300ETF'
    },
    '500etf': {
        'name': '中证500ETF',
        'type': 'etf',
        'etf_type': '500ETF'
    },
    'io': {
        'name': '沪深300指数',
        'type': 'cffex',
        'prefix': 'io'
    },
    'mo': {
        'name': '中证1000指数',
        'type': 'cffex',
        'prefix': 'mo'
    },
    'ho': {
        'name': '上证50指数',
        'type': 'cffex',
        'prefix': 'ho'
    }
}


def process_index_option(code, data_dir=None, trade_date=None):
    """
    Process a single index option product

    Args:
        code: Product code (e.g., '50etf', 'io', 'mo')
        data_dir: Directory containing options data
        trade_date: Optional date string (YYYYMMDD). If None, uses latest.

    Returns:
        DataFrame with volatility smile data
    """
    code = code.lower()
    if code not in INDEX_OPTIONS:
        print(f"Unknown product: {code}")
        print(f"Available: {list(INDEX_OPTIONS.keys())}")
        return None

    config = INDEX_OPTIONS[code]

    # Default data directory - support DATA_DIR env variable for shared data
    if data_dir is None:
        SCRIPT_DIR = os.path.dirname(os.path.abspath(__file__))
        BASE_DIR = os.path.dirname(SCRIPT_DIR)
        data_root = os.environ.get("DATA_DIR", os.path.join(BASE_DIR, "data"))
        data_dir = os.path.join(data_root, 'index_options')
        if not os.path.exists(data_dir):
            # Fallback to local path
            data_dir = os.path.join(BASE_DIR, 'data', 'index_options')

    if not os.path.exists(data_dir):
        print(f"Data directory not found: {data_dir}")
        return None

    print(f"\n[{config['name']}] ({code.upper()})")

    # Load data based on type
    if config['type'] == 'etf':
        df = load_sse_etf_options(data_dir)
        if df is None or df.empty:
            print(f"  No ETF data found")
            return None
        # Filter to specific ETF type
        df_filtered = df[df['etf_type'] == config['etf_type']]
        if df_filtered.empty:
            print(f"  No data for {config['etf_type']}")
            return None
        print(f"  Loaded {len(df_filtered)} {config['etf_type']} options")
    else:
        df = load_cffex_index_options(data_dir, config['prefix'])
        if df is None or df.empty:
            print(f"  No CFFEX data found for {config['prefix']}")
            return None
        df_filtered = df
        print(f"  Loaded {len(df_filtered)} options")

    # Get trade date
    dates = sorted(df_filtered['trade_date'].unique(), reverse=True)
    if trade_date is None:
        trade_date = str(dates[0])
    elif int(trade_date) not in dates:
        print(f"  Date {trade_date} not found, using {dates[0]}")
        trade_date = str(dates[0])

    print(f"  Using date: {trade_date}")

    # Calculate smile
    if config['type'] == 'etf':
        smile_df = calculate_etf_volatility_smile(df, config['etf_type'], trade_date)
    else:
        smile_df = calculate_index_volatility_smile(df_filtered, config['name'], trade_date)

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


def process_all_index_options(data_dir=None, trade_date=None):
    """Process all available index options"""
    results = {}

    # Process ETF options
    for code in ['50etf', '300etf', '500etf']:
        results[code] = process_index_option(code, data_dir, trade_date)

    # Process CFFEX index options
    for code in ['io', 'mo', 'ho']:
        results[code] = process_index_option(code, data_dir, trade_date)

    return results


if __name__ == '__main__':
    import sys

    print("=" * 60)
    print("Index Options Volatility Smile Calculator")
    print("=" * 60)

    # Parse command line arguments
    if len(sys.argv) > 1:
        # Process specific products from command line
        codes = sys.argv[1:]
        for code in codes:
            process_index_option(code)
    else:
        # Default: process all index options
        process_all_index_options()

    print("\n" + "=" * 60)
    print("Done!")
