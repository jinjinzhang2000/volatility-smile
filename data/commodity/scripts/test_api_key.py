#!/usr/bin/env python3
"""
快速测试AppleRouter API密钥
"""

from openai import OpenAI
import sys

def test_api_key(api_key: str):
    """测试API密钥是否有效"""
    
    print("🔍 测试AppleRouter API密钥...")
    print(f"密钥: {api_key[:20]}...{api_key[-10:]}")
    print("-" * 80)
    
    client = OpenAI(
        api_key=api_key,
        base_url="https://api.applerouter.ai/v1"
    )
    
    # 测试模型列表
    test_models = [
        ("GPT-4o Mini", "gpt-4o-mini"),
        ("GPT-4o", "gpt-4o"),
        ("Claude 3.5 Sonnet", "claude-3-5-sonnet-20241022"),
        ("DeepSeek Chat", "deepseek-chat"),
    ]
    
    test_message = [
        {"role": "user", "content": "请用一句话介绍你自己"}
    ]
    
    success_count = 0
    failed_models = []
    
    for model_name, model_id in test_models:
        try:
            print(f"\n✅ 测试 {model_name} ({model_id})...")
            
            response = client.chat.completions.create(
                model=model_id,
                messages=test_message,
                max_tokens=100,
                temperature=0.7
            )
            
            result = response.choices[0].message.content
            print(f"   回复: {result[:100]}...")
            success_count += 1
            
        except Exception as e:
            print(f"   ❌ 失败: {str(e)}")
            failed_models.append((model_name, model_id, str(e)))
    
    # 总结
    print("\n" + "=" * 80)
    print(f"📊 测试结果: {success_count}/{len(test_models)} 个模型可用")
    print("=" * 80)
    
    if success_count > 0:
        print(f"✅ API密钥有效！成功连接 {success_count} 个模型")
    else:
        print("❌ API密钥可能无效或网络连接有问题")
    
    if failed_models:
        print("\n⚠️ 以下模型无法访问:")
        for name, model_id, error in failed_models:
            print(f"   • {name} ({model_id})")
            print(f"     错误: {error}")
    
    print("\n" + "=" * 80)
    
    if success_count == len(test_models):
        print("🎉 完美！所有模型都可以正常使用")
        return True
    elif success_count > 0:
        print("⚠️ 部分模型可用，可以开始使用")
        return True
    else:
        print("❌ 请检查API密钥或网络连接")
        return False


if __name__ == "__main__":
    # 你的API密钥
    api_key = "sk-IJBDTS344JZloC5dyLTgzlsD6jlxAN53xkQi2aWUcoP9p1pq"
    
    print("="*80)
    print("🤖 AppleRouter API 测试工具")
    print("="*80)
    
    success = test_api_key(api_key)
    
    if success:
        print("\n✅ 测试通过！你可以开始使用以下脚本：")
        print("   • python applerouter_ai_client.py  (通用客户端)")
        print("   • python quant_ai_assistant.py     (量化交易助手)")
    else:
        print("\n❌ 测试失败，请检查:")
        print("   1. API密钥是否正确")
        print("   2. 网络连接是否正常")
        print("   3. AppleRouter服务是否可用")
    
    sys.exit(0 if success else 1)
