"""
AppleRouter API Client
使用 OpenAI 兼容接口调用 AppleRouter
"""

import json
import os
from datetime import datetime
from openai import OpenAI


# 可选模型 (AppleRouter 可用)
MODELS = {
    "1": ("gpt-5-pro", "GPT-5 Pro"),
    "2": ("claude-opus-4-5", "Claude Opus 4.5"),
    "3": ("o4-mini-deep-research-2025-06-26", "O4 Mini Deep Research"),
    "4": ("deepseek-v3.1-250821", "DeepSeek V3.1"),
    "5": ("o3-deep-research", "O3 Deep Research"),
    "6": ("gpt-5.1", "GPT-5.1"),
    "7": ("deepseek-r1", "DeepSeek R1 (推理)"),
    "8": ("gemini-2.5-pro", "Gemini 2.5 Pro"),
}

# 历史记录文件
HISTORY_FILE = os.path.expanduser("~/.applerouter_history.json")


def create_client(api_key: str) -> OpenAI:
    """创建 AppleRouter 客户端"""
    return OpenAI(
        api_key=api_key,
        base_url="https://api.applerouter.ai/v1"
    )


def list_available_models(client: OpenAI):
    """列出 API 支持的所有模型"""
    try:
        models = client.models.list()
        print("\n=== AppleRouter 可用模型 ===")
        for m in sorted([m.id for m in models.data]):
            print(f"  {m}")
        print("=" * 30)
    except Exception as e:
        print(f"[获取模型列表失败: {e}]")


def select_model() -> str:
    """让用户选择模型"""
    print("\n选择模型:")
    for key, (model_id, name) in MODELS.items():
        print(f"  {key}. {name}")
    
    choice = input("输入数字 (默认1): ").strip() or "1"
    model_id, name = MODELS.get(choice, MODELS["1"])
    print(f"→ 使用: {name}\n")
    return model_id


def save_history(history: list, model: str):
    """保存历史到文件"""
    data = {
        "model": model,
        "updated": datetime.now().isoformat(),
        "history": history
    }
    try:
        with open(HISTORY_FILE, 'w', encoding='utf-8') as f:
            json.dump(data, f, ensure_ascii=False, indent=2)
    except Exception as e:
        print(f"[保存失败: {e}]")


def load_history() -> tuple[list, str]:
    """从文件加载历史"""
    try:
        if os.path.exists(HISTORY_FILE):
            with open(HISTORY_FILE, 'r', encoding='utf-8') as f:
                data = json.load(f)
            return data.get("history", []), data.get("model", "")
    except Exception as e:
        print(f"[加载历史失败: {e}]")
    return [], ""


def show_history(history: list):
    """显示历史记录"""
    if not history:
        print("[没有历史记录]")
        return
    
    print(f"\n=== 历史记录 ({len(history)//2} 轮对话) ===")
    for i, msg in enumerate(history):
        role = "你" if msg["role"] == "user" else "AI"
        content = msg["content"][:80] + "..." if len(msg["content"]) > 80 else msg["content"]
        print(f"{role}: {content}")
    print("=" * 40)


def chat(client: OpenAI, message: str, model: str) -> str:
    """发送聊天请求"""
    response = client.chat.completions.create(
        model=model,
        messages=[
            {"role": "user", "content": message}
        ]
    )
    return response.choices[0].message.content


def chat_stream(client: OpenAI, message: str, model: str, history: list = None):
    """流式输出，带错误处理"""
    messages = history.copy() if history else []
    messages.append({"role": "user", "content": message})
    
    try:
        response = client.chat.completions.create(
            model=model,
            messages=messages,
            stream=True,
            timeout=120  # 2分钟超时
        )
        
        full_response = ""
        for chunk in response:
            if chunk.choices and chunk.choices[0].delta.content:
                content = chunk.choices[0].delta.content
                print(content, end="", flush=True)
                full_response += content
        print()
        return full_response
        
    except Exception as e:
        print(f"\n\n[错误: {type(e).__name__}] {str(e)[:100]}")
        print("[提示: 输入 'r' 重试]")
        return None


def interactive_chat(client: OpenAI):
    """交互式聊天，带历史记录"""
    print("=" * 50)
    print("AppleRouter 交互式聊天")
    print("命令: q退出 | m切换模型 | c清空历史 | r重试 | h查看历史 | l列出所有模型")
    print("=" * 50)
    
    # 尝试加载上次的历史
    history, saved_model = load_history()
    
    if history:
        print(f"\n[发现上次对话记录 ({len(history)//2} 轮)]")
        choice = input("继续上次对话? (y/n, 默认y): ").strip().lower()
        if choice == 'n':
            history = []
            model = select_model()
        else:
            print("[已加载历史]")
            # 找到保存的模型
            model = saved_model
            if model:
                model_name = next((name for mid, name in MODELS.values() if mid == model), model)
                print(f"→ 继续使用: {model_name}")
            else:
                model = select_model()
    else:
        model = select_model()
    
    last_input = ""
    
    while True:
        try:
            user_input = input("\n你: ").strip()
            
            if not user_input:
                continue
            elif user_input.lower() == 'q':
                save_history(history, model)
                print("[历史已保存] 再见!")
                break
            elif user_input.lower() == 'm':
                model = select_model()
                continue
            elif user_input.lower() == 'c':
                history = []
                save_history(history, model)
                print("[历史已清空]")
                continue
            elif user_input.lower() == 'h':
                show_history(history)
                continue
            elif user_input.lower() == 'l':
                list_available_models(client)
                continue
            elif user_input.lower() == 'r':
                if last_input:
                    user_input = last_input
                    print(f"[重试: {user_input[:50]}...]" if len(user_input) > 50 else f"[重试: {user_input}]")
                else:
                    print("[没有可重试的问题]")
                    continue
            
            last_input = user_input
            print("\nAI: ", end="")
            response = chat_stream(client, user_input, model, history)
            
            if response:
                history.append({"role": "user", "content": user_input})
                history.append({"role": "assistant", "content": response})
                # 保留最近10轮对话
                if len(history) > 20:
                    history = history[-20:]
                # 自动保存
                save_history(history, model)
            
        except KeyboardInterrupt:
            save_history(history, model)
            print("\n[历史已保存] 再见!")
            break


if __name__ == "__main__":
    # 你的 API Key
    API_KEY = "sk-2vCcIePwaiAumoZnTeLCJS24KTd9jJZ5EAhjbNPJk28EyMMO"
    
    client = create_client(API_KEY)
    interactive_chat(client)
