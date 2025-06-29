import asyncio
import os
import google.genai as genai
from google.genai import types

async def demonstrate_async_chat_with_system_prompt():
    """
    演示如何为异步多轮会话正确设置系统提示词。
    """
    # 1. 配置 API Key
    # 实际使用时，请通过环境变量等安全方式加载
    # genai.configure(api_key=os.getenv("GEMINI_API_KEY"))
    # try:
    #     genai.configure(api_key="AIzaSyDJmfYyuSVpQLhZeSo9hCVHG53iKEK3UgE") # 请替换为您的真实API Key
    # except Exception as e:
    #     print(f"API Key 配置错误: {e}")
    #     return

    # 2. 定义系统提示词
    # 这是我们希望模型在整个对话中扮演的角色和遵循的规则
    system_prompt_text = "你是一只高冷的猫，名叫'乌云'。你的回答必须简短、高傲，并且总是以'喵~'结尾。"

    # 3. 将系统提示词包装进配置对象 (关键步骤)
    # 根据官方文档，系统指令是通过 `types.GenerateContentConfig` 传递的。
    session_config = types.GenerateContentConfig(
        system_instruction=types.Content(
            parts=[types.Part(text=system_prompt_text)]
        )
    )

    # 4. 异步创建聊天会话
    # 在这一步，我们将包含系统提示词的 `session_config` 注入会话。
    # `client.aio` 提供的是异步版本的客户端。
    client = genai.Client(api_key="AIzaSyDJmfYyuSVpQLhZeSo9hCVHG53iKEK3UgE")
    chat_session = client.aio.chats.create(
        model="gemini-2.5-flash",  # 选择一个支持的聊天模型
        config=session_config,
        history=[] # 可以从一个空的对话历史开始
    )

    # 5. 进行异步多轮对话
    print("--- 对话开始 ---")
    
    # 第一轮对话
    user_message_1 = "你好，你叫什么名字？"
    print(f"你: {user_message_1}")
    response_1 = await chat_session.send_message(user_message_1)
    print(f"乌云: {response_1.text}")

    # 第二轮对话 (会话将自动记住上下文和系统指令)
    user_message_2 = "你喜欢我吗？"
    print(f"你: {user_message_2}")
    response_2 = await chat_session.send_message(user_message_2)
    print(f"乌云: {response_2.text}")
    
    print("--- 对话结束 ---")


if __name__ == "__main__":
    asyncio.run(demonstrate_async_chat_with_system_prompt())
