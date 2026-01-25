#!/usr/bin/env python3
"""
AppleRouter AI API 客户端
通过统一接口连接多个AI提供商：OpenAI, Anthropic, Google, DeepSeek等
"""

from openai import OpenAI
import os
from typing import List, Dict, Optional

class AppleRouterClient:
    """AppleRouter统一API客户端"""
    
    def __init__(self, api_key: str):
        """
        初始化AppleRouter客户端
        
        Args:
            api_key: 你的AppleRouter API密钥
        """
        self.client = OpenAI(
            api_key=api_key,
            base_url="https://api.applerouter.ai/v1"
        )
    
    def chat(self, 
             model: str, 
             messages: List[Dict[str, str]], 
             temperature: float = 0.7,
             max_tokens: Optional[int] = None,
             stream: bool = False) -> str:
        """
        发送聊天请求
        
        Args:
            model: 模型名称，如 "gpt-4o", "claude-3-5-sonnet-20241022", "gemini-pro"
            messages: 消息列表
            temperature: 温度参数 (0-1)
            max_tokens: 最大token数
            stream: 是否流式输出
        
        Returns:
            模型响应内容
        """
        kwargs = {
            "model": model,
            "messages": messages,
            "temperature": temperature,
            "stream": stream
        }
        
        if max_tokens:
            kwargs["max_tokens"] = max_tokens
        
        if stream:
            return self._stream_response(kwargs)
        else:
            response = self.client.chat.completions.create(**kwargs)
            return response.choices[0].message.content
    
    def _stream_response(self, kwargs: dict) -> str:
        """处理流式响应"""
        full_response = ""
        stream = self.client.chat.completions.create(**kwargs)
        
        for chunk in stream:
            if chunk.choices[0].delta.content:
                content = chunk.choices[0].delta.content
                print(content, end="", flush=True)
                full_response += content
        
        print()  # 换行
        return full_response


def demo_multiple_providers(api_key: str):
    """演示连接多个AI提供商"""
    
    client = AppleRouterClient(api_key)
    
    # 测试消息
    test_message = [
        {"role": "user", "content": "用一句话介绍你自己"}
    ]
    
    # 支持的模型列表（根据AppleRouter的实际支持情况）
    models = {
        "OpenAI GPT-4o": "gpt-4o",
        "OpenAI GPT-4o-mini": "gpt-4o-mini",
        "Anthropic Claude 3.5 Sonnet": "claude-3-5-sonnet-20241022",
        "Anthropic Claude 3.5 Haiku": "claude-3-5-haiku-20241022",
        "Google Gemini Pro": "gemini-pro",
        "DeepSeek V3": "deepseek-chat",
    }
    
    print("=" * 80)
    print("通过AppleRouter测试多个AI模型")
    print("=" * 80)
    
    for name, model_id in models.items():
        try:
            print(f"\n🤖 {name} ({model_id}):")
            print("-" * 80)
            response = client.chat(
                model=model_id,
                messages=test_message,
                temperature=0.7
            )
            print(response)
            print("-" * 80)
        except Exception as e:
            print(f"❌ 错误: {e}")
            print("-" * 80)


def interactive_chat(api_key: str, model: str = "gpt-4o"):
    """交互式聊天"""
    
    client = AppleRouterClient(api_key)
    messages = []
    
    print(f"🤖 开始与 {model} 对话 (输入 'quit' 退出)\n")
    
    while True:
        user_input = input("你: ").strip()
        
        if user_input.lower() in ['quit', 'exit', 'q']:
            print("再见！")
            break
        
        if not user_input:
            continue
        
        messages.append({"role": "user", "content": user_input})
        
        try:
            print(f"\n{model}: ", end="", flush=True)
            response = client.chat(
                model=model,
                messages=messages,
                stream=True  # 使用流式输出
            )
            messages.append({"role": "assistant", "content": response})
            print()  # 额外换行
            
        except Exception as e:
            print(f"\n❌ 错误: {e}")
            messages.pop()  # 移除导致错误的用户消息


def main():
    """主函数"""
    
    # 从环境变量获取API密钥，或者直接在这里填写
    api_key = os.getenv("APPLEROUTER_API_KEY", "YOUR_API_KEY")
    
    if api_key == "YOUR_API_KEY":
        print("⚠️  请设置你的AppleRouter API密钥！")
        print("\n方法1: 在代码中直接修改 api_key 变量")
        print("方法2: 设置环境变量 APPLEROUTER_API_KEY")
        print("\n示例: export APPLEROUTER_API_KEY='your-key-here'\n")
        return
    
    # 选择运行模式
    print("选择运行模式:")
    print("1. 测试多个AI模型")
    print("2. 交互式对话")
    
    choice = input("\n请选择 (1/2): ").strip()
    
    if choice == "1":
        demo_multiple_providers(api_key)
    elif choice == "2":
        print("\n可用模型:")
        print("- gpt-4o")
        print("- gpt-4o-mini")
        print("- claude-3-5-sonnet-20241022")
        print("- deepseek-chat")
        print("- gemini-pro")
        
        model = input("\n输入模型名称 (默认: gpt-4o): ").strip() or "gpt-4o"
        interactive_chat(api_key, model)
    else:
        print("无效选择")


if __name__ == "__main__":
    main()
