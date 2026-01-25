#!/usr/bin/env python3
"""
量化交易AI助手 - 使用AppleRouter连接多个AI模型
专为中国商品期货和股票期权市场设计
"""

from openai import OpenAI
import os
from datetime import datetime

class QuantAIAssistant:
    """量化交易AI助手"""
    
    def __init__(self, api_key: str):
        self.client = OpenAI(
            api_key=api_key,
            base_url="https://api.applerouter.ai/v1"
        )
        
        # 不同任务使用不同模型
        self.models = {
            "market_analysis": "deepseek-chat",      # DeepSeek擅长中文市场分析
            "strategy_dev": "claude-3-5-sonnet-20241022",  # Claude擅长复杂推理
            "code_gen": "gpt-4o",                    # GPT-4o擅长代码生成
            "quick_query": "gpt-4o-mini"             # 快速查询用mini版本
        }
    
    def analyze_market(self, market: str, asset: str, context: str = "") -> str:
        """
        市场分析
        
        Args:
            market: 市场类型 (商品期货/股票期权/A股)
            asset: 标的资产 (如：螺纹钢、铁矿石、50ETF等)
            context: 额外背景信息
        """
        prompt = f"""作为量化交易专家，请分析{market}市场中{asset}的当前状况。

分析要点：
1. 当前市场环境和宏观因素
2. 该标的的供需基本面
3. 技术面信号和趋势
4. 潜在的尾部风险
5. 交易建议和风险控制

{f'额外背景：{context}' if context else ''}

请基于2026年1月的市场环境进行分析。"""

        messages = [
            {"role": "system", "content": "你是一位经验丰富的量化交易专家，专注于中国商品期货和股票期权市场，擅长尾部风险管理和波动率交易。"},
            {"role": "user", "content": prompt}
        ]
        
        response = self.client.chat.completions.create(
            model=self.models["market_analysis"],
            messages=messages,
            temperature=0.3  # 分析类任务用较低温度
        )
        
        return response.choices[0].message.content
    
    def design_strategy(self, strategy_type: str, description: str) -> str:
        """
        策略设计
        
        Args:
            strategy_type: 策略类型 (期权套利/波动率交易/趋势跟踪等)
            description: 策略描述和要求
        """
        prompt = f"""请设计一个{strategy_type}策略：

{description}

请提供：
1. 策略逻辑和理论基础
2. 入场和出场条件
3. 仓位管理和风险控制
4. 预期收益特征和风险点
5. 适用市场环境
6. 回测和优化建议"""

        messages = [
            {"role": "system", "content": "你是策略研发专家，擅长设计稳健的量化交易策略，特别关注风险调整后收益。"},
            {"role": "user", "content": prompt}
        ]
        
        response = self.client.chat.completions.create(
            model=self.models["strategy_dev"],
            messages=messages,
            temperature=0.5
        )
        
        return response.choices[0].message.content
    
    def generate_code(self, task: str, language: str = "Python") -> str:
        """
        代码生成
        
        Args:
            task: 任务描述
            language: 编程语言
        """
        prompt = f"""请用{language}实现以下功能：

{task}

要求：
1. 代码清晰易读，有完整注释
2. 包含错误处理
3. 如果涉及数据分析，使用pandas/numpy
4. 如果涉及金融计算，考虑使用tushare或其他金融库
5. 提供使用示例"""

        messages = [
            {"role": "system", "content": f"你是{language}编程专家，擅长金融量化开发，熟悉pandas、numpy、tushare等库。"},
            {"role": "user", "content": prompt}
        ]
        
        response = self.client.chat.completions.create(
            model=self.models["code_gen"],
            messages=messages,
            temperature=0.2
        )
        
        return response.choices[0].message.content
    
    def quick_query(self, question: str) -> str:
        """快速查询"""
        messages = [
            {"role": "system", "content": "你是量化交易助手，简洁准确地回答问题。"},
            {"role": "user", "content": question}
        ]
        
        response = self.client.chat.completions.create(
            model=self.models["quick_query"],
            messages=messages,
            temperature=0.3
        )
        
        return response.choices[0].message.content
    
    def compare_models(self, question: str) -> dict:
        """
        使用多个模型回答同一问题并对比
        
        Args:
            question: 问题
            
        Returns:
            各模型的回答字典
        """
        results = {}
        
        messages = [{"role": "user", "content": question}]
        
        for name, model in self.models.items():
            try:
                response = self.client.chat.completions.create(
                    model=model,
                    messages=messages,
                    temperature=0.5
                )
                results[f"{name} ({model})"] = response.choices[0].message.content
            except Exception as e:
                results[f"{name} ({model})"] = f"错误: {e}"
        
        return results


def demo_market_analysis(assistant: QuantAIAssistant):
    """演示市场分析功能"""
    print("\n" + "="*80)
    print("📊 市场分析演示")
    print("="*80)
    
    analysis = assistant.analyze_market(
        market="商品期货",
        asset="螺纹钢",
        context="考虑中国房地产市场的恢复预期和政府政策支持"
    )
    
    print(analysis)


def demo_strategy_design(assistant: QuantAIAssistant):
    """演示策略设计功能"""
    print("\n" + "="*80)
    print("🎯 策略设计演示")
    print("="*80)
    
    strategy = assistant.design_strategy(
        strategy_type="深度虚值期权买入",
        description="""
设计一个系统性买入深度虚值期权的策略，用于捕捉2026年中国大宗商品市场的尾部风险。

标的：螺纹钢、铁矿石、原油、PTA、玻璃期权
目标：在控制单次亏损的前提下，捕捉极端行情带来的非对称收益
约束：需要考虑流动性限制和交易成本
"""
    )
    
    print(strategy)


def demo_code_generation(assistant: QuantAIAssistant):
    """演示代码生成功能"""
    print("\n" + "="*80)
    print("💻 代码生成演示")
    print("="*80)
    
    code = assistant.generate_code(
        task="""
使用tushare API获取A股某只股票的历史数据，并计算：
1. 最近20日的历史波动率
2. 布林带上下轨
3. RSI指标
并绘制K线图和指标图
"""
    )
    
    print(code)


def demo_model_comparison(assistant: QuantAIAssistant):
    """演示多模型对比"""
    print("\n" + "="*80)
    print("🔄 多模型对比演示")
    print("="*80)
    
    question = "为什么期权隐含波动率曲面会出现波动率微笑现象？"
    
    results = assistant.compare_models(question)
    
    for model_name, answer in results.items():
        print(f"\n【{model_name}】")
        print("-" * 80)
        print(answer)
        print("-" * 80)


def interactive_mode(assistant: QuantAIAssistant):
    """交互式模式"""
    print("\n" + "="*80)
    print("💬 交互式量化交易助手")
    print("="*80)
    print("\n功能选择:")
    print("1 - 市场分析")
    print("2 - 策略设计")
    print("3 - 代码生成")
    print("4 - 快速查询")
    print("5 - 多模型对比")
    print("quit - 退出\n")
    
    while True:
        choice = input("请选择功能 (1-5 或 quit): ").strip()
        
        if choice.lower() in ['quit', 'exit', 'q']:
            print("再见！")
            break
        
        if choice == '1':
            market = input("市场类型 (商品期货/股票期权/A股): ").strip()
            asset = input("标的资产: ").strip()
            context = input("额外背景 (可选): ").strip()
            
            print("\n分析中...\n")
            result = assistant.analyze_market(market, asset, context)
            print(result)
            
        elif choice == '2':
            strategy_type = input("策略类型: ").strip()
            description = input("策略描述和要求: ").strip()
            
            print("\n设计中...\n")
            result = assistant.design_strategy(strategy_type, description)
            print(result)
            
        elif choice == '3':
            task = input("任务描述: ").strip()
            
            print("\n生成中...\n")
            result = assistant.generate_code(task)
            print(result)
            
        elif choice == '4':
            question = input("你的问题: ").strip()
            
            print("\n查询中...\n")
            result = assistant.quick_query(question)
            print(result)
            
        elif choice == '5':
            question = input("你的问题: ").strip()
            
            print("\n对比中...\n")
            results = assistant.compare_models(question)
            for model_name, answer in results.items():
                print(f"\n【{model_name}】")
                print("-" * 80)
                print(answer)
        
        print("\n" + "="*80 + "\n")


def main():
    # 从环境变量获取API密钥
    api_key = os.getenv("APPLEROUTER_API_KEY", "YOUR_API_KEY")
    
    if api_key == "YOUR_API_KEY":
        print("⚠️  请设置你的AppleRouter API密钥！")
        print("\n方法: export APPLEROUTER_API_KEY='your-key-here'\n")
        return
    
    assistant = QuantAIAssistant(api_key)
    
    print("🤖 量化交易AI助手")
    print(f"📅 当前日期: {datetime.now().strftime('%Y-%m-%d')}")
    
    # 选择模式
    print("\n运行模式:")
    print("1 - 市场分析演示")
    print("2 - 策略设计演示")
    print("3 - 代码生成演示")
    print("4 - 多模型对比演示")
    print("5 - 交互式模式")
    
    choice = input("\n请选择 (1-5): ").strip()
    
    if choice == '1':
        demo_market_analysis(assistant)
    elif choice == '2':
        demo_strategy_design(assistant)
    elif choice == '3':
        demo_code_generation(assistant)
    elif choice == '4':
        demo_model_comparison(assistant)
    elif choice == '5':
        interactive_mode(assistant)
    else:
        print("无效选择")


if __name__ == "__main__":
    main()
