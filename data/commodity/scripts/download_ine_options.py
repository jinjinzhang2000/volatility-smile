"""
使用Tushare下载上海国际能源交易中心(INE)商品期权数据
包括：原油期权(SC)、低硫燃料油期权(LU)、20号胶期权(NR)、国际铜期权(BC)

使用前请确保：
pip install tushare pandas openpyxl
"""

import tushare as ts
import pandas as pd
import os
from datetime import datetime, timedelta
import time

# ============ 配置区域 ============
TUSHARE_TOKEN = "a70287c82208760b640d7f08525b97181166b817e0d9ff5f8f244bc2"
OUTPUT_DIR = "./ine_options_data"
START_DATE = "20250101"  # 只下载2025年数据
END_DATE = datetime.now().strftime("%Y%m%d")
# ==================================

def init_tushare():
    """初始化tushare"""
    ts.set_token(TUSHARE_TOKEN)
    pro = ts.pro_api()
    return pro

def create_output_dir():
    """创建输出目录"""
    os.makedirs(OUTPUT_DIR, exist_ok=True)
    print(f"输出目录: {OUTPUT_DIR}")

# =====================================================
# Part 1: 获取期权合约基本信息
# =====================================================

def get_option_basic(pro, exchange='INE'):
    """
    获取期权合约基本信息
    exchange: INE-上海国际能源交易中心
    """
    print(f"\n正在获取 {exchange} 期权合约基本信息...")
    
    try:
        df = pro.opt_basic(
            exchange=exchange,
            fields='ts_code,exchange,name,per_unit,opt_code,opt_type,call_put,exercise_type,exercise_price,s_month,maturity_date,list_price,list_date,delist_date,last_edate,last_ddate,quote_unit,min_price_chg'
        )
        if df is not None and len(df) > 0:
            print(f"  获取 {len(df)} 个期权合约")
            return df
    except Exception as e:
        print(f"  获取失败: {e}")
    
    return pd.DataFrame()

# =====================================================
# Part 2: 按日期批量获取期权日线
# =====================================================

def get_option_daily_by_date(pro, trade_date, exchange='INE', max_retries=3):
    """获取某一天所有期权的日线数据"""
    for attempt in range(max_retries):
        try:
            df = pro.opt_daily(
                trade_date=trade_date,
                exchange=exchange,
                fields='ts_code,trade_date,exchange,pre_settle,pre_close,open,high,low,close,settle,vol,amount,oi'
            )
            return df
        except Exception as e:
            if attempt < max_retries - 1:
                time.sleep(1)
            else:
                return None
    return None

def get_all_daily_by_date_range(pro, start_date, end_date, exchange='INE'):
    """
    按日期范围批量获取期权日线数据
    """
    all_data = []
    
    # 获取交易日历
    print(f"\n正在获取交易日历...")
    cal = pro.trade_cal(
        exchange='SSE',
        start_date=start_date,
        end_date=end_date,
        is_open='1'
    )
    trade_dates = sorted(cal['cal_date'].tolist())
    total_days = len(trade_dates)
    
    print(f"共 {total_days} 个交易日")
    print(f"开始下载 {exchange} 期权数据...\n")
    
    for i, trade_date in enumerate(trade_dates, 1):
        if i % 20 == 0 or i == 1:
            print(f"进度: {i}/{total_days} ({i/total_days*100:.1f}%) - {trade_date}")
        
        df = get_option_daily_by_date(pro, trade_date, exchange)
        
        if df is not None and len(df) > 0:
            all_data.append(df)
        
        time.sleep(0.1)
    
    if all_data:
        result = pd.concat(all_data, ignore_index=True)
        return result
    return pd.DataFrame()

# =====================================================
# Part 3: 获取期货日线数据（标的物）
# =====================================================

def get_futures_daily(pro, exchange='INE', start_date='20210701', end_date=None):
    """获取INE期货日线数据"""
    if end_date is None:
        end_date = datetime.now().strftime("%Y%m%d")
    
    print(f"\n正在获取 {exchange} 期货日线数据...")
    
    all_data = []
    
    cal = pro.trade_cal(
        exchange='SSE',
        start_date=start_date,
        end_date=end_date,
        is_open='1'
    )
    trade_dates = sorted(cal['cal_date'].tolist())
    total_days = len(trade_dates)
    
    print(f"共 {total_days} 个交易日")
    
    for i, trade_date in enumerate(trade_dates, 1):
        if i % 50 == 0 or i == 1:
            print(f"进度: {i}/{total_days} ({i/total_days*100:.1f}%)")
        
        try:
            df = pro.fut_daily(
                trade_date=trade_date,
                exchange=exchange,
                fields='ts_code,trade_date,pre_close,pre_settle,open,high,low,close,settle,change1,change2,vol,amount,oi,oi_chg'
            )
            if df is not None and len(df) > 0:
                all_data.append(df)
        except:
            pass
        
        time.sleep(0.05)
    
    if all_data:
        result = pd.concat(all_data, ignore_index=True)
        print(f"  共获取 {len(result)} 条期货数据")
        return result
    return pd.DataFrame()

# =====================================================
# Part 4: 保存数据
# =====================================================

def save_data(df, filename, description=""):
    """保存数据到CSV"""
    if df is None or len(df) == 0:
        print(f"  {description} 无数据，跳过保存")
        return
    
    filepath = os.path.join(OUTPUT_DIR, filename)
    df.to_csv(filepath, index=False, encoding='utf-8-sig')
    print(f"  已保存: {filepath} ({len(df)} 条)")

def save_to_excel(data_dict, filename):
    """保存多个DataFrame到Excel的不同sheet"""
    filepath = os.path.join(OUTPUT_DIR, filename)
    
    with pd.ExcelWriter(filepath, engine='openpyxl') as writer:
        for sheet_name, df in data_dict.items():
            if df is not None and len(df) > 0:
                # Excel sheet名最多31字符
                sheet_name = sheet_name[:31]
                df.to_excel(writer, sheet_name=sheet_name, index=False)
                print(f"  Sheet '{sheet_name}': {len(df)} 条")
    
    print(f"  已保存: {filepath}")

# =====================================================
# 主函数
# =====================================================

def main():
    print("=" * 60)
    print("INE商品期权数据下载工具 (Tushare)")
    print("=" * 60)
    
    # 检查token
    if TUSHARE_TOKEN == "your_token_here":
        print("\n❌ 错误: 请先设置Tushare Token!")
        return
    
    # 初始化
    pro = init_tushare()
    create_output_dir()
    
    # 选择下载内容
    print("\n请选择下载内容:")
    print("1. 仅下载期权合约基本信息")
    print("2. 下载期权日线数据（按日期，推荐）")
    print("3. 下载期权+期货日线数据（完整）")
    print("4. 全部下载")
    
    choice = input("\n请输入选择 (1/2/3/4，默认4): ").strip() or "4"
    
    results = {}
    
    # 1. 期权合约基本信息
    if choice in ['1', '4']:
        print("\n" + "=" * 40)
        print("下载期权合约基本信息")
        print("=" * 40)
        
        option_basic = get_option_basic(pro, exchange='INE')
        if option_basic is not None and len(option_basic) > 0:
            results['option_basic'] = option_basic
            save_data(option_basic, 'ine_option_basic.csv', '期权合约信息')
            
            # 分品种统计
            print("\n期权品种统计:")
            for code_prefix in ['SC', 'LU', 'NR', 'BC']:
                count = len(option_basic[option_basic['ts_code'].str.startswith(code_prefix)])
                if count > 0:
                    name = {'SC': '原油', 'LU': '低硫燃料油', 'NR': '20号胶', 'BC': '国际铜'}[code_prefix]
                    print(f"  {name}期权({code_prefix}): {count} 个合约")
    
    # 2. 期权日线数据
    if choice in ['2', '3', '4']:
        print("\n" + "=" * 40)
        print("下载期权日线数据")
        print("=" * 40)
        
        option_daily = get_all_daily_by_date_range(pro, START_DATE, END_DATE, exchange='INE')
        if option_daily is not None and len(option_daily) > 0:
            results['option_daily'] = option_daily
            save_data(option_daily, f'ine_option_daily_{START_DATE}_{END_DATE}.csv', '期权日线')
            
            # 分品种保存
            print("\n按品种分别保存:")
            for code_prefix in ['SC', 'LU', 'NR', 'BC']:
                df_subset = option_daily[option_daily['ts_code'].str.startswith(code_prefix)]
                if len(df_subset) > 0:
                    name = {'SC': 'crude_oil', 'LU': 'fuel_oil', 'NR': 'rubber', 'BC': 'copper'}[code_prefix]
                    save_data(df_subset, f'ine_{name}_option_daily.csv', f'{code_prefix}期权')
    
    # 3. 期货日线数据
    if choice in ['3', '4']:
        print("\n" + "=" * 40)
        print("下载期货日线数据（标的物）")
        print("=" * 40)
        
        futures_daily = get_futures_daily(pro, exchange='INE', start_date=START_DATE, end_date=END_DATE)
        if futures_daily is not None and len(futures_daily) > 0:
            results['futures_daily'] = futures_daily
            save_data(futures_daily, f'ine_futures_daily_{START_DATE}_{END_DATE}.csv', '期货日线')
    
    # 汇总保存到Excel
    if len(results) > 0:
        print("\n" + "=" * 40)
        print("保存汇总Excel文件")
        print("=" * 40)
        save_to_excel(results, f'ine_options_complete_{END_DATE}.xlsx')
    
    # 完成
    print("\n" + "=" * 60)
    print("✅ 下载完成!")
    print("=" * 60)
    
    # 打印数据统计
    if 'option_daily' in results:
        df = results['option_daily']
        print(f"\n期权数据统计:")
        print(f"  总记录数: {len(df):,}")
        print(f"  合约数量: {df['ts_code'].nunique()}")
        print(f"  日期范围: {df['trade_date'].min()} 至 {df['trade_date'].max()}")
        
        print(f"\n按品种统计:")
        for code_prefix in ['SC', 'LU', 'NR', 'BC']:
            subset = df[df['ts_code'].str.startswith(code_prefix)]
            if len(subset) > 0:
                name = {'SC': '原油(SC)', 'LU': '低硫燃料油(LU)', 'NR': '20号胶(NR)', 'BC': '国际铜(BC)'}[code_prefix]
                print(f"  {name}: {len(subset):,} 条, {subset['ts_code'].nunique()} 个合约")

if __name__ == "__main__":
    main()
