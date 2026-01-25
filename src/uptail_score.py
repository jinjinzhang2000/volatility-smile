"""
UpTailScore Calculator for Commodity Options
Identifies attractive opportunities for buying OTM calls (upside tail bets)

UpTailScore Formula (Optimized):
- 35% × JumpRisk_rank       -- Higher historical jump risk is better (core predictor)
- 25% × Skew_rank           -- Higher skew is better (calls cheaper vs puts)
- 20% × (100 - IV_pct)      -- Lower IV is better (cheaper options)
- 15% × (100 - Wing_rank)   -- Lower wing premium is better (deep OTM not overpriced)
- 5% × Liquidity_rank       -- Better liquidity is better (execution)

Key Settings:
- ATM IV, Skew, Wing: Uses ~90-day expiry options (not front month) for stability
- Jump Risk: Uses 2-year (504 trading days) historical data

Author: Forrest
Date: 2025-01-25
"""

import pandas as pd
import numpy as np
import os
from datetime import datetime, timedelta
from scipy.stats import percentileofscore
import warnings
warnings.filterwarnings('ignore')

# Import from existing modules
from commodity_volatility_smile import (
    COMMODITIES, process_commodity, load_shfe_data, load_czce_data, load_dce_data
)
from skew_analyzer import calculate_skew_metrics


# Use the expanded COMMODITIES from commodity_volatility_smile.py
ALL_COMMODITIES = COMMODITIES


def load_smile_data(code):
    """Load existing smile data from CSV"""
    csv_path = f"output/data/{code}_smile_data.csv"
    if os.path.exists(csv_path):
        return pd.read_csv(csv_path)
    return None


def select_best_maturity(smile_df, target_days=90, min_records=10):
    """
    Select the best maturity for IV calculation.
    Prefers target_days but falls back to more liquid maturities if data quality is poor.

    Returns: (selected_days, target_month_df)
    """
    if smile_df is None or smile_df.empty:
        return None, None

    available_days = sorted(smile_df['days'].unique())
    if len(available_days) == 0:
        return None, None

    # Sort by closeness to target_days
    sorted_days = sorted(available_days, key=lambda x: abs(x - target_days))

    for days in sorted_days:
        target_month = smile_df[smile_df['days'] == days].drop_duplicates().copy()

        # Check data quality: need enough records with reasonable volume
        if len(target_month) >= min_records:
            # Check if we have both calls and puts near ATM
            atm_range = target_month[(target_month['moneyness'] >= 0.95) &
                                      (target_month['moneyness'] <= 1.05)]
            if len(atm_range) >= 3:
                return days, target_month

    # Fallback: use the maturity with most records
    best_days = max(available_days, key=lambda d: len(smile_df[smile_df['days'] == d]))
    return best_days, smile_df[smile_df['days'] == best_days].drop_duplicates().copy()


def calculate_atm_iv(smile_df, target_days=90):
    """
    Calculate ATM IV from smile data
    Uses contracts closest to target_days (default 90 days) for more stable IV
    Falls back to more liquid maturities if target maturity has poor data quality
    """
    if smile_df is None or smile_df.empty:
        return None

    selected_days, target_month = select_best_maturity(smile_df, target_days)

    if target_month is None or target_month.empty:
        return None

    # Find ATM (moneyness closest to 1.0)
    target_month['atm_dist'] = abs(target_month['moneyness'] - 1.0)
    atm_row = target_month.loc[target_month['atm_dist'].idxmin()]

    atm_iv = atm_row['iv']

    # Sanity check: IV should be between 3% and 200%
    if atm_iv < 3 or atm_iv > 200:
        # Try to find a more reasonable ATM IV from nearby strikes
        atm_range = target_month[(target_month['moneyness'] >= 0.98) &
                                  (target_month['moneyness'] <= 1.02)]
        if len(atm_range) > 0:
            atm_iv = atm_range['iv'].median()

    return atm_iv if 3 <= atm_iv <= 200 else None


def calculate_skew_25d(smile_df, target_days=90):
    """
    Calculate 25-delta risk reversal (put IV - call IV)
    Positive = put premium (downside fear) = calls are CHEAPER
    Negative = call premium (upside expectation) = calls are MORE EXPENSIVE

    For UpTailScore, we want HIGHER skew (more positive = calls cheaper relative to puts)
    This means Put IV > Call IV, so OTM calls are relatively cheap

    Uses contracts closest to target_days (default 90 days) for more stable calculation
    Falls back to more liquid maturities if needed
    """
    if smile_df is None or smile_df.empty:
        return None

    selected_days, target_month = select_best_maturity(smile_df, target_days, min_records=8)

    if target_month is None or len(target_month) < 5:
        return None

    # Find 25-delta put (~0.93 moneyness) and call (~1.07 moneyness)
    otm_puts = target_month[(target_month['moneyness'] < 0.98) &
                            (target_month['option_type'] == 'put')]
    otm_calls = target_month[(target_month['moneyness'] > 1.02) &
                             (target_month['option_type'] == 'call')]

    # Fallback: if no explicit type, use moneyness only
    if len(otm_puts) == 0:
        otm_puts = target_month[target_month['moneyness'] < 0.98]
    if len(otm_calls) == 0:
        otm_calls = target_month[target_month['moneyness'] > 1.02]

    if len(otm_puts) == 0 or len(otm_calls) == 0:
        return None

    # Target moneyness for 25-delta
    put_target = 0.93
    call_target = 1.07

    put_iv = otm_puts.iloc[(otm_puts['moneyness'] - put_target).abs().argsort()[:1]]['iv'].values[0]
    call_iv = otm_calls.iloc[(otm_calls['moneyness'] - call_target).abs().argsort()[:1]]['iv'].values[0]

    # Risk reversal: put IV - call IV
    skew_25d = put_iv - call_iv

    # Sanity check: skew should be within reasonable range (-20% to +20%)
    if abs(skew_25d) > 20:
        return None

    return skew_25d


def calculate_wing_score(smile_df, target_days=90):
    """
    Calculate wing (deep OTM) premium
    Wing = average IV of deep OTM options relative to ATM

    For UpTailScore, we want LOWER wing (deep OTM calls not overpriced)
    Uses contracts closest to target_days (default 90 days) for more stable calculation
    Falls back to more liquid maturities if needed
    """
    if smile_df is None or smile_df.empty:
        return None

    selected_days, target_month = select_best_maturity(smile_df, target_days, min_records=8)

    if target_month is None or len(target_month) < 5:
        return None

    # Find ATM IV
    target_month['atm_dist'] = abs(target_month['moneyness'] - 1.0)
    atm_iv = target_month.loc[target_month['atm_dist'].idxmin(), 'iv']

    # Sanity check ATM IV
    if atm_iv < 3 or atm_iv > 200:
        atm_range = target_month[(target_month['moneyness'] >= 0.98) &
                                  (target_month['moneyness'] <= 1.02)]
        if len(atm_range) > 0:
            atm_iv = atm_range['iv'].median()
        else:
            return None

    # Deep OTM calls (moneyness > 1.10)
    deep_otm_calls = target_month[(target_month['moneyness'] > 1.10) &
                                   (target_month['option_type'] == 'call')]

    if len(deep_otm_calls) == 0:
        # Fallback: any deep OTM
        deep_otm_calls = target_month[target_month['moneyness'] > 1.10]

    if len(deep_otm_calls) == 0:
        # Fallback to moderate OTM
        deep_otm_calls = target_month[target_month['moneyness'] > 1.05]

    if len(deep_otm_calls) == 0:
        return None

    # Wing = deep OTM IV - ATM IV (positive = wing premium)
    wing_premium = deep_otm_calls['iv'].mean() - atm_iv

    return wing_premium


def calculate_jump_risk(code, lookback_days=504):
    """
    Calculate historical jump risk (large positive moves)
    Based on frequency and magnitude of upside jumps in futures prices

    For UpTailScore, we want HIGHER jump risk (more potential for big moves)
    """
    try:
        import tushare as ts
        token = os.environ.get("TUSHARE_TOKEN", "a70287c82208760b640d7f08525b97181166b817e0d9ff5f8f244bc2")
        ts.set_token(token)
        pro = ts.pro_api()

        # Map to exchange
        exchange_map = {
            'rb': 'SHFE', 'ag': 'SHFE', 'au': 'SHFE', 'cu': 'SHFE', 'ru': 'SHFE',
            'zn': 'SHFE', 'pb': 'SHFE', 'ni': 'SHFE', 'sn': 'SHFE', 'al': 'SHFE',
            'fu': 'SHFE', 'bu': 'SHFE', 'ss': 'SHFE', 'sp': 'SHFE',
            'fg': 'CZCE', 'sr': 'CZCE', 'cf': 'CZCE', 'ta': 'CZCE', 'ma': 'CZCE',
            'rm': 'CZCE', 'sa': 'CZCE', 'pf': 'CZCE', 'pk': 'CZCE', 'oi': 'CZCE',
            'ur': 'CZCE', 'zc': 'CZCE',
            'i': 'DCE', 'jm': 'DCE', 'j': 'DCE', 'm': 'DCE', 'y': 'DCE',
            'p': 'DCE', 'c': 'DCE', 'v': 'DCE', 'l': 'DCE', 'pp': 'DCE',
            'eg': 'DCE', 'pg': 'DCE',
            'sc': 'INE',
        }

        exchange = exchange_map.get(code.lower(), 'SHFE')

        # Get main contract data
        end_date = datetime.now().strftime('%Y%m%d')
        start_date = (datetime.now() - timedelta(days=lookback_days * 2)).strftime('%Y%m%d')

        # Get index (continuous contract) data
        if exchange == 'CZCE':
            ts_code = f"{code.upper()}L.ZCE"  # L = main contract
        elif exchange == 'DCE':
            ts_code = f"{code.upper()}L.DCE"
        elif exchange == 'INE':
            ts_code = f"{code.upper()}L.INE"
        else:
            ts_code = f"{code.upper()}L.SHF"

        df = pro.fut_daily(
            ts_code=ts_code,
            start_date=start_date,
            end_date=end_date,
            fields='trade_date,close,pre_close'
        )

        if df is None or len(df) < 60:
            # Try alternative: use specific contract
            return None

        df = df.sort_values('trade_date')
        df['return'] = df['close'].pct_change()

        # Calculate jump metrics
        returns = df['return'].dropna()

        if len(returns) < 60:
            return None

        # Jump risk score based on:
        # 1. Frequency of large positive moves (> 2 std)
        # 2. Magnitude of top positive moves
        # 3. Kurtosis (fat tails)

        std = returns.std()
        mean = returns.mean()

        # Count big up moves (> 2 std above mean)
        big_up_moves = (returns > mean + 2 * std).sum()
        big_up_freq = big_up_moves / len(returns) * 252  # Annualized frequency

        # Average magnitude of top 5% positive returns
        top_returns = returns[returns > returns.quantile(0.95)]
        avg_top_return = top_returns.mean() if len(top_returns) > 0 else 0

        # Kurtosis (higher = fatter tails)
        from scipy.stats import kurtosis
        kurt = kurtosis(returns)

        # Combine into jump risk score
        jump_score = (big_up_freq * 10) + (avg_top_return * 100) + (max(0, kurt) * 2)

        return jump_score

    except Exception as e:
        print(f"    Warning: Could not calculate jump risk for {code}: {e}")
        return None


def calculate_liquidity(smile_df):
    """
    Calculate liquidity score based on trading volume

    For UpTailScore, we want HIGHER liquidity
    """
    if smile_df is None or smile_df.empty:
        return None

    # Total volume across all strikes and maturities
    total_volume = smile_df['volume'].sum()

    # Average volume per strike
    avg_volume = smile_df['volume'].mean()

    # Number of liquid strikes (volume > 100)
    liquid_strikes = (smile_df['volume'] > 100).sum()

    # Combined liquidity score
    liquidity_score = np.log1p(total_volume) * 10 + liquid_strikes

    return liquidity_score


def calculate_uptail_scores():
    """
    Calculate UpTailScore for all available commodities

    Returns DataFrame with scores and rankings
    """
    print("=" * 70)
    print("UpTailScore Calculator - Commodity Options")
    print("=" * 70)
    print(f"\nDate: {datetime.now().strftime('%Y-%m-%d %H:%M')}\n")

    results = []

    # Collect metrics for all commodities
    for code, config in ALL_COMMODITIES.items():
        print(f"Processing {code.upper()} ({config['name']})...", end=" ")

        # Load smile data
        smile_df = load_smile_data(code)

        if smile_df is None or smile_df.empty:
            # Try to generate fresh data
            try:
                data_dir = f"Commodity/{code}"
                if not os.path.exists(data_dir):
                    print("No data")
                    continue
                smile_df = process_commodity(code)
            except:
                pass

        if smile_df is None or smile_df.empty:
            print("No smile data")
            continue

        # Calculate metrics
        atm_iv = calculate_atm_iv(smile_df)
        skew_25d = calculate_skew_25d(smile_df)
        wing = calculate_wing_score(smile_df)
        jump_risk = calculate_jump_risk(code)
        liquidity = calculate_liquidity(smile_df)

        if atm_iv is None:
            print("Incomplete metrics")
            continue

        results.append({
            'code': code,
            'name': config['name'],
            'exchange': config['exchange'],
            'atm_iv': atm_iv,
            'skew_25d': skew_25d,
            'wing': wing,
            'jump_risk': jump_risk,
            'liquidity': liquidity
        })

        print(f"ATM IV={atm_iv:.1f}%")

    if not results:
        print("\nNo valid data found!")
        return None

    # Convert to DataFrame
    df = pd.DataFrame(results)

    print(f"\n{'='*70}")
    print(f"Collected data for {len(df)} commodities")
    print(f"{'='*70}\n")

    # Calculate percentile ranks (0-100)
    # For IV: lower is better, so we use (100 - percentile)
    df['iv_pct'] = df['atm_iv'].rank(pct=True) * 100

    # For Skew: HIGHER (more positive) is better for buying calls
    # Positive skew means Put IV > Call IV, so OTM calls are relatively cheap
    # We rank high skew = high rank (good for calls)
    df['skew_rank'] = df['skew_25d'].rank(pct=True) * 100  # Higher skew = higher rank

    # For Wing: lower is better (deep OTM not overpriced)
    df['wing_rank'] = df['wing'].rank(pct=True) * 100

    # For Jump Risk: higher is better
    df['jump_rank'] = df['jump_risk'].rank(pct=True) * 100

    # For Liquidity: higher is better
    df['liquidity_rank'] = df['liquidity'].rank(pct=True) * 100

    # Fill NaN with 50 (neutral)
    df['skew_rank'] = df['skew_rank'].fillna(50)
    df['wing_rank'] = df['wing_rank'].fillna(50)
    df['jump_rank'] = df['jump_rank'].fillna(50)
    df['liquidity_rank'] = df['liquidity_rank'].fillna(50)

    # Calculate UpTailScore (Optimized weights)
    # UpTailScore = 35% × Jump_rank + 25% × Skew_rank + 20% × (100 - IV_pct) +
    #               15% × (100 - Wing_rank) + 5% × Liquidity_rank
    #
    # Weight rationale:
    # - Jump 35%: Core logic - historical jumps predict future tail events
    # - Skew 25%: Direct pricing advantage - cheap calls relative to puts
    # - IV 20%: Lower IV means cheaper options, but low IV may be justified
    # - Wing 15%: Less critical as we typically buy moderate OTM (5-10%), not deep OTM
    # - Liquidity 5%: Execution concern, minimal weight
    df['uptail_score'] = (
        0.35 * df['jump_rank'] +           # Higher jump risk is better
        0.25 * df['skew_rank'] +           # Higher skew is better (calls cheaper)
        0.20 * (100 - df['iv_pct']) +      # Lower IV is better
        0.15 * (100 - df['wing_rank']) +   # Lower wing is better
        0.05 * df['liquidity_rank']        # Higher liquidity is better
    )

    # Sort by UpTailScore
    df = df.sort_values('uptail_score', ascending=False)

    # Add rank column
    df['rank'] = range(1, len(df) + 1)

    return df


def print_uptail_report(df):
    """Print formatted UpTailScore report"""
    if df is None or df.empty:
        return

    print("\n" + "=" * 90)
    print("                    UpTailScore 排名 - 看涨尾部风险机会")
    print("=" * 90)
    print("\n公式: 35%×JumpRisk + 25%×Skew + 20%×(100-IV) + 15%×(100-Wing) + 5%×Liquidity")
    print("\n设置: ATM IV/Skew/Wing使用~90天到期期权, Jump Risk使用2年历史数据")
    print("说明: IV低=好, Skew正(Put>Call)=好, Wing低=好, Jump高=好, 流动性高=好\n")

    print("-" * 90)
    print(f"{'Rank':<5} {'Code':<6} {'Name':<10} {'Exchange':<8} {'ATM IV':<8} {'Skew25d':<8} "
          f"{'Wing':<8} {'Jump':<8} {'Score':<8}")
    print("-" * 90)

    for _, row in df.head(20).iterrows():
        skew_str = f"{row['skew_25d']:+.1f}" if pd.notna(row['skew_25d']) else "N/A"
        wing_str = f"{row['wing']:+.1f}" if pd.notna(row['wing']) else "N/A"
        jump_str = f"{row['jump_risk']:.1f}" if pd.notna(row['jump_risk']) else "N/A"

        print(f"{row['rank']:<5} {row['code'].upper():<6} {row['name']:<10} {row['exchange']:<8} "
              f"{row['atm_iv']:>6.1f}% {skew_str:>7} {wing_str:>7} {jump_str:>7} "
              f"{row['uptail_score']:>6.1f}")

    print("-" * 90)

    # Top 5 recommendations
    print("\n" + "=" * 90)
    print("                         TOP 5 看涨尾部机会")
    print("=" * 90)

    for idx, row in df.head(5).iterrows():
        print(f"\n#{row['rank']} {row['code'].upper()} {row['name']} ({row['exchange']})")
        print(f"   UpTailScore: {row['uptail_score']:.1f}")
        print(f"   ATM IV: {row['atm_iv']:.1f}% (排名: {100-row['iv_pct']:.0f}%ile - {'低' if row['iv_pct'] < 50 else '高'})")

        if pd.notna(row['skew_25d']):
            print(f"   25d Skew: {row['skew_25d']:+.2f}% ({'Call便宜' if row['skew_25d'] > 0 else 'Call贵'})")

        if pd.notna(row['wing']):
            print(f"   Wing Premium: {row['wing']:+.2f}% ({'翘尾' if row['wing'] > 0 else '平坦'})")

        if pd.notna(row['jump_risk']):
            print(f"   Jump Risk: {row['jump_risk']:.1f} (排名: {row['jump_rank']:.0f}%ile)")

        print(f"   流动性排名: {row['liquidity_rank']:.0f}%ile")

    # Bottom 5 (avoid)
    print("\n" + "=" * 90)
    print("                      BOTTOM 5 (避免买入Call)")
    print("=" * 90)

    for idx, row in df.tail(5).iloc[::-1].iterrows():
        print(f"\n#{row['rank']} {row['code'].upper()} {row['name']}")
        print(f"   UpTailScore: {row['uptail_score']:.1f}")
        print(f"   ATM IV: {row['atm_iv']:.1f}% ({'高波动率溢价' if row['iv_pct'] > 70 else ''})")


def save_uptail_results(df):
    """Save UpTailScore results to CSV"""
    if df is None or df.empty:
        return

    # Ensure output directory exists
    os.makedirs('output/data', exist_ok=True)

    # Select columns to save
    output_df = df[[
        'rank', 'code', 'name', 'exchange',
        'uptail_score', 'atm_iv', 'skew_25d', 'wing',
        'jump_risk', 'liquidity',
        'iv_pct', 'skew_rank', 'wing_rank', 'jump_rank', 'liquidity_rank'
    ]].copy()

    # Save to CSV
    csv_path = 'output/data/uptail_score_ranking.csv'
    output_df.to_csv(csv_path, index=False, encoding='utf-8-sig')
    print(f"\nResults saved to: {csv_path}")

    return csv_path


if __name__ == '__main__':
    import sys

    # Calculate scores
    df = calculate_uptail_scores()

    if df is not None and not df.empty:
        # Print report
        print_uptail_report(df)

        # Save results
        save_uptail_results(df)

        print("\n" + "=" * 90)
        print("Done!")
        print("=" * 90)
