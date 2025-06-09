import gradio as gr
import httpx
from loguru import logger

# 您 FastAPI 应用的本地地址
# 如果您的端口不是 8000，请修改这里
FASTAPI_BASE_URL = "http://127.0.0.1:8000"

# 这是“桥梁函数”，它负责调用您已有的 FastAPI 端点
async def call_ask_api(query_text: str):
    """
    一个异步函数，用于向 FastAPI 的 /query/ask 端点发送请求。
    Gradio 将会调用这个函数。
    """
    if not query_text:
        return "请输入问题后再提交。"

    # API 端点的完整 URL
    api_url = f"{FASTAPI_BASE_URL}/query/ask"
    
    # 准备要发送的 JSON 数据
    payload = {"query": query_text}
    
    logger.info(f"Gradio 界面正在调用 API: {api_url}，负载: {payload}")

    try:
        # 使用 httpx 异步地发送 POST 请求
        async with httpx.AsyncClient(timeout=600.0) as client: # 设置60秒超时
            response = await client.post(api_url, json=payload)
            response.raise_for_status()  # 如果 API 返回错误状态码 (如 4xx, 5xx)，则会抛出异常
            
            # 解析返回的 JSON 数据
            data = response.json()
            
            # 从返回的数据中提取最终答案并返回
            # 请根据您的 AskQueryResponse 模型结构调整这里的键名
            answer = data.get("answer", "未能从响应中解析出答案。")
            retrieved_context_texts = data.get("retrieved_context_texts") or []

            # 格式化输出，使其更美观
            formatted_output = f"### 回答:\n{answer}\n\n"
            # 检查列表是否为空
            if retrieved_context_texts:
                formatted_output += "--- \n### 参考资料:\n"
                # 直接遍历字符串列表，因为 retrieved_context_texts 就是一个 list[str]
                for i, text_chunk in enumerate(retrieved_context_texts, 1):
                    # 每个 'text_chunk' 就是一段纯文本字符串
                    formatted_output += f"**[{i}]** {text_chunk}\n\n"
            
            return formatted_output

    except httpx.HTTPStatusError as e:
        error_message = f"API 请求失败，状态码: {e.response.status_code}，响应: {e.response.text}"
        logger.error(error_message)
        return error_message
    except Exception as e:
        error_message = f"调用 API 时发生未知错误: {e}"
        logger.error(error_message, exc_info=True)
        return error_message


# 使用 gr.Blocks() 创建一个自定义的 Gradio 界面
with gr.Blocks(title="RAG 本地知识库应用", theme=gr.themes.Soft()) as rag_demo_ui:
    gr.Markdown("# RAG 本地知识库应用问答")
    gr.Markdown("在这里输入您的问题，系统将调用后端的 RAG 流程来生成答案。")
    
    with gr.Row():
        # 输入组件
        question_input = gr.Textbox(label="您的问题", placeholder="例如：请介绍一下 Qwen2.5 模型有什么新特性？", lines=3)
    
    with gr.Row():
        # 提交按钮
        submit_button = gr.Button("提交问题", variant="primary")
        
    with gr.Row():
        # 输出组件，使用 Markdown 格式以支持富文本显示
        answer_output = gr.Markdown(label="生成的回答")

    # 设置按钮的点击事件
    # 当按钮被点击时，调用 call_ask_api 函数
    # 函数的输入来自于 question_input 组件
    # 函数的输出将更新到 answer_output 组件
    submit_button.click(
        fn=call_ask_api,
        inputs=[question_input],
        outputs=[answer_output]
    )