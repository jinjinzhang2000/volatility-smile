"""
测试 AppleRouter 模型可用性
"""

from openai import OpenAI
import time

API_KEY = "sk-2vCcIePwaiAumoZnTeLCJS24KTd9jJZ5EAhjbNPJk28EyMMO"

client = OpenAI(
    api_key=API_KEY,
    base_url="https://api.applerouter.ai/v1"
)

# 要测试的模型列表
MODELS_TO_TEST = [
    # GPT 系列
    "gpt-5-pro",
    "gpt-5.1",
    "gpt-5",
    "gpt-5-chat",
    "gpt-5-mini",
    "gpt-4o",
    "gpt-4o-mini",
    "gpt-4.5-preview",
    "gpt-4.1",
    
    # Claude 系列
    "claude-opus-4-5",
    "claude-opus-4",
    "claude-sonnet-4-5",
    "claude-sonnet-4",
    "claude-haiku-4-5",
    
    # O 系列 (推理模型)
    "o4-mini-deep-research-2025-06-26",
    "o4-mini",
    "o3-deep-research",
    "o3",
    "o3-mini",
    "o1",
    "o1-mini",
    
    # DeepSeek
    "deepseek-v3.1-250821",
    "deepseek-v3.1",
    "deepseek-r1",
    "deepseek-chat",
    
    # Gemini
    "gemini-2.5-pro",
    "gemini-2.5-flash",
    "gemini-3-pro-preview",
    
    # Qwen
    "qwen-max",
    "qwen3-max-preview",
]


def test_model(model_id: str) -> tuple[bool, str]:
    """测试单个模型是否可用"""
    try:
        response = client.chat.completions.create(
            model=model_id,
            messages=[{"role": "user", "content": "Hi"}],
            max_tokens=5,
            timeout=30
        )
        return True, "✅ 可用"
    except Exception as e:
        error_msg = str(e)[:80]
        if "404" in error_msg:
            return False, "❌ 404 不存在或需验证"
        elif "503" in error_msg:
            return False, "❌ 503 无可用渠道"
        elif "429" in error_msg:
            return False, "⚠️ 429 限流"
        elif "401" in error_msg:
            return False, "❌ 401 认证失败"
        else:
            return False, f"❌ {error_msg[:40]}"


def main():
    print("=" * 60)
    print("AppleRouter 模型可用性测试")
    print("=" * 60)
    print()
    
    available = []
    unavailable = []
    
    for i, model in enumerate(MODELS_TO_TEST, 1):
        print(f"[{i}/{len(MODELS_TO_TEST)}] 测试: {model}...", end=" ", flush=True)
        success, msg = test_model(model)
        print(msg)
        
        if success:
            available.append(model)
        else:
            unavailable.append((model, msg))
        
        # 避免请求过快
        time.sleep(0.5)
    
    # 汇总结果
    print()
    print("=" * 60)
    print(f"测试完成: {len(available)} 可用 / {len(unavailable)} 不可用")
    print("=" * 60)
    
    print("\n✅ 可用模型:")
    for m in available:
        print(f"  - {m}")
    
    print("\n❌ 不可用模型:")
    for m, reason in unavailable:
        print(f"  - {m}: {reason}")
    
    # 生成推荐配置
    print("\n" + "=" * 60)
    print("推荐配置 (复制到 applerouter_client.py):")
    print("=" * 60)
    print("MODELS = {")
    for i, m in enumerate(available[:8], 1):  # 最多8个
        print(f'    "{i}": ("{m}", "{m}"),')
    print("}")


if __name__ == "__main__":
    main()
