from openai import OpenAI
import logging

# 启用详细日志
logging.basicConfig(level=logging.DEBUG)

try:
    # 创建OpenAI客户端实例
    client = OpenAI(
        api_key="bf9dd506-1205-447e-9662-291ee9c159fe",  # 确保没有多余的空格或制表符
        base_url="https://api.doubao.com/chat/completions"  # 请确认这个地址是否正确
    )
    
    # 测试请求
    response = client.chat.completions.create(
        model="Doubao-4",  # 确认豆包平台支持的模型名称
        messages=[
            {"role": "system", "content": "你是一个帮助用户的助手。"},
            {"role": "user", "content": "你好，我是用户，这是一个测试请求。"}
        ],
        temperature=0.7,
        max_tokens=100
    )
    
    # 打印响应结果
    print("测试成功！响应内容：")
    print(response.choices[0].message.content)
    
except Exception as e:
    print(f"测试失败：{type(e).__name__} - {e}")