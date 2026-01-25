#!/usr/bin/env python3
"""
简单测试：查询AppleRouter实际支持的模型
"""

from openai import OpenAI

api_key = "sk-IJBDTS344JZloC5dyLTgzlsD6jlxAN53xkQi2aWUcoP9p1pq"

client = OpenAI(
    api_key=api_key,
    base_url="https://api.applerouter.ai/v1"
)

print("="*80)
print("🔍 查询AppleRouter可用模型")
print("="*80)

# 方法1：尝试获取模型列表
try:
    print("\n方法1：调用models.list() API...")
    models = client.models.list()
    
    if hasattr(models, 'data'):
        print(f"\n✅ 找到 {len(models.data)} 个模型:")
        print("-"*80)
        for i, model in enumerate(models.data, 1):
            model_id = model.id if hasattr(model, 'id') else str(model)
            print(f"{i}. {model_id}")
        print("-"*80)
    else:
        print("⚠️ API返回了数据但格式不同")
        print(f"返回内容: {models}")
        
except Exception as e:
    print(f"❌ 无法获取模型列表: {e}")

# 方法2：尝试一些可能的模型名称变体
print("\n方法2：测试常见模型名称...")
print("-"*80)

# 可能的模型名称（AppleRouter可能使用不同的命名）
test_models = [
    # 简化名称
    "gpt-4",
    "gpt-3.5",
    "claude",
    "deepseek",
    
    # 完整名称
    "gpt-4o-2024-05-13",
    "gpt-4o-mini-2024-07-18",
    "claude-3-5-sonnet-20241022",
    "deepseek-chat",
    
    # 可能的别名
    "openai/gpt-4o",
    "anthropic/claude-3-5-sonnet",
    "deepseek/deepseek-chat",
    
    # 通用名称
    "default",
    "auto",
]

available = []

for model_name in test_models:
    try:
        response = client.chat.completions.create(
            model=model_name,
            messages=[{"role": "user", "content": "test"}],
            max_tokens=5
        )
        print(f"✅ {model_name}")
        available.append(model_name)
    except Exception as e:
        error_str = str(e)
        if "model_not_found" in error_str:
            print(f"❌ {model_name} - 未配置")
        elif "rate_limit" in error_str:
            print(f"⚠️ {model_name} - 速率限制（可能可用）")
            available.append(model_name)
        else:
            # 只显示简短错误
            short_error = error_str.split('\n')[0][:60]
            print(f"❌ {model_name} - {short_error}...")

print("-"*80)

if available:
    print(f"\n✅ 找到 {len(available)} 个可用模型:")
    for model in available:
        print(f"  • {model}")
    
    print("\n你可以使用这些模型进行测试！")
else:
    print("\n❌ 没有找到任何可用模型")
    print("\n可能的原因:")
    print("  1. '测试组001' 是空的试用分组")
    print("  2. 需要在AppleRouter后台配置模型")
    print("  3. 需要升级到付费套餐")
    print("  4. 联系AppleRouter客服获取帮助")

print("\n" + "="*80)
print("💡 建议:")
print("  1. 登录AppleRouter后台查看你的套餐")
print("  2. 查看'测试组001'的配置详情")
print("  3. 查阅AppleRouter的文档了解模型命名规则")
print("="*80)
