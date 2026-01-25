#!/usr/bin/env python3

from openai import OpenAI
import os
from dotenv import load_dotenv

load_dotenv()

def main():
    api_key = os.getenv("OPENAI_API_KEY")
    
    if not api_key:
        print("错误：请设置 OPENAI_API_KEY 环境变量")
        return
    
    client = OpenAI(
        api_key=api_key,
        base_url="https://api.applerouter.ai/v1"
    )
    
    # 列出可用模型
    print("查询可用模型...")
    try:
        models = client.models.list()
        print("\n可用模型列表：")
        for model in models.data:
            print(f"  - {model.id}")
    except Exception as e:
        print(f"无法列出模型: {e}")
        return

if __name__ == "__main__":
    main()