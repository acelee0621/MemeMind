import gradio as gr
import httpx
from loguru import logger
import pandas as pd

# 您 FastAPI 应用的本地地址
FASTAPI_BASE_URL = "http://127.0.0.1:8000"

# ===================================================================
# 桥梁函数 (Bridge Functions)
# ===================================================================


async def call_ask_api(query_text: str):
    """【问答模块】的桥梁函数"""
    if not query_text or not query_text.strip():
        return "请输入有效的问题后再提交。"
    api_url = f"{FASTAPI_BASE_URL}/query/ask"
    payload = {"query": query_text}
    logger.info(f"Gradio 界面正在调用 API: {api_url}，负载: {payload}")
    try:
        async with httpx.AsyncClient(timeout=300.0) as client:
            response = await client.post(api_url, json=payload)
            response.raise_for_status()
            data = response.json()
            answer = data.get("answer", "未能从响应中解析出答案。")
            source_texts = data.get("retrieved_context_texts") or []
            formatted_output = f"### 答案:\n{answer}\n\n"
            if source_texts:
                formatted_output += "--- \n### 参考资料:\n"
                for i, text_chunk in enumerate(source_texts, 1):
                    formatted_output += f"**[{i}]** {text_chunk or '无内容'}\n\n"
            return formatted_output
    except Exception as e:
        logger.error(f"调用问答 API 时发生错误: {e}", exc_info=True)
        return f"处理问答请求时出错: {e}"


async def get_all_docs_bridge():
    """【文档管理】获取所有文档列表的桥梁函数"""
    api_url = f"{FASTAPI_BASE_URL}/documents"
    logger.info(f"Gradio 正在刷新文档列表，调用 API: {api_url}")
    try:
        async with httpx.AsyncClient() as client:
            response = await client.get(api_url, params={"limit": 100, "offset": 0})
            response.raise_for_status()
            docs_list = response.json()
            if not docs_list:
                return pd.DataFrame(
                    columns=["ID", "文件名", "状态", "块数量", "创建时间"]
                )
            df = pd.DataFrame(docs_list)
            df_display = df[
                ["id", "original_filename", "status", "number_of_chunks", "created_at"]
            ].copy()
            df_display.rename(
                columns={
                    "id": "ID",
                    "original_filename": "文件名",
                    "status": "处理状态",
                    "number_of_chunks": "块数量",
                    "created_at": "上传时间",
                },
                inplace=True,
            )
            return df_display
    except Exception as e:
        logger.error(f"刷新文档列表时出错: {e}", exc_info=True)
        gr.Error(f"无法加载文档列表: {e}")
        return pd.DataFrame(columns=["ID", "文件名", "状态", "块数量", "创建时间"])


async def upload_doc_bridge(file_obj):
    """【文档管理】上传文件的桥梁函数"""
    if file_obj is None:
        return "未选择任何文件"
    api_url = f"{FASTAPI_BASE_URL}/documents"
    files = {
        "file": (file_obj.name, open(file_obj.name, "rb"), "application/octet-stream")
    }
    logger.info(f"Gradio 正在上传文件: {file_obj.name} 到 {api_url}")
    try:
        async with httpx.AsyncClient(timeout=60.0) as client:
            response = await client.post(api_url, files=files)
            response.raise_for_status()
            data = response.json()
            success_message = (
                f"文件 '{data.get('original_filename')}' 上传成功！ID: {data.get('id')}"
            )
            logger.info(success_message)
            return success_message
    except Exception as e:
        error_message = f"文件上传失败: {e}"
        logger.error(error_message, exc_info=True)
        gr.Error(error_message)
        return error_message


async def delete_doc_bridge(doc_id_str: str):
    """【文档管理】删除指定ID文档的桥梁函数"""
    if not doc_id_str or not doc_id_str.strip().isdigit():
        message = "请输入有效的纯数字文档ID"
        gr.Warning(message)
        return message
    doc_id = int(doc_id_str)
    api_url = f"{FASTAPI_BASE_URL}/documents/{doc_id}"
    logger.info(f"Gradio 正在删除文档 ID: {doc_id}，调用 API: {api_url}")
    try:
        async with httpx.AsyncClient() as client:
            response = await client.delete(api_url)
            response.raise_for_status()
            success_message = f"文档 ID: {doc_id} 已成功删除。"
            logger.info(success_message)
            return success_message
    except Exception as e:
        error_message = f"删除过程中发生错误: {e}"
        logger.error(error_message, exc_info=True)
        gr.Error(error_message)
        return error_message


async def retrieve_chunks_bridge(query: str, top_k: int):
    """【检索测试】的桥梁函数"""
    if not query or not query.strip():
        gr.Warning("请输入检索关键词！")
        return None  # 返回 None 以保持输出区域不变

    api_url = f"{FASTAPI_BASE_URL}/query/retrieve-chunks"
    payload = {"query": query, "top_k": int(top_k)}
    logger.info(f"Gradio 正在调用检索 API: {api_url}，负载: {payload}")

    try:
        async with httpx.AsyncClient(timeout=120.0) as client:
            response = await client.post(api_url, json=payload)
            response.raise_for_status()
            chunks_list = response.json()

            if not chunks_list:
                return pd.DataFrame(columns=["ID", "来源文档ID", "文本块内容", "顺序"])

            # 将返回的JSON列表转换为DataFrame以供展示
            df = pd.DataFrame(chunks_list)
            df_display = df[
                ["id", "source_document_id", "chunk_text", "sequence_in_document"]
            ].copy()
            df_display.rename(
                columns={
                    "id": "块ID",
                    "source_document_id": "来源文档ID",
                    "chunk_text": "文本块内容",
                    "sequence_in_document": "块顺序",
                },
                inplace=True,
            )
            return df_display

    except Exception as e:
        error_message = f"检索文本块时出错: {e}"
        logger.error(error_message, exc_info=True)
        gr.Error(error_message)
        return pd.DataFrame(columns=["ID", "来源文档ID", "文本块内容", "顺序"])


# ===================================================================
# Gradio UI 界面定义
# ===================================================================

with gr.Blocks(title="RAG 应用控制台", theme=gr.themes.Soft()) as rag_demo_ui:
    gr.Markdown("# RAG 应用控制台")

    with gr.Tabs():
        # --- 选项卡一：智能问答 ---
        with gr.TabItem("智能问答"):
            gr.Markdown("在这里输入您的问题，系统将调用后端的 RAG 流程来生成答案。")
            with gr.Row():
                question_input = gr.Textbox(
                    label="您的问题",
                    placeholder="例如：这里提你自己的问题？",
                    lines=3,
                    scale=4,
                )
                ask_submit_button = gr.Button("提交问题", variant="primary", scale=1)
            with gr.Row():
                answer_output = gr.Markdown(label="生成的回答")

        # --- 选项卡二：文档管理 ---
        with gr.TabItem("文档管理") as doc_management_tab:
            gr.Markdown(
                "管理您的知识库文档：上传新文件、查看已处理文件、删除不需要的文件。"
            )
            with gr.Row():
                with gr.Column(scale=1):
                    upload_file_button = gr.File(
                        label="上传新文档", file_count="single"
                    )
                    upload_status_text = gr.Textbox(label="上传状态", interactive=False)
                    gr.Markdown("---")
                    delete_doc_id_input = gr.Textbox(label="输入要删除的文档ID")
                    delete_button = gr.Button("确认删除", variant="stop")
                    delete_status_text = gr.Textbox(label="删除状态", interactive=False)
                with gr.Column(scale=3):
                    refresh_docs_button = gr.Button("刷新文档列表")
                    file_list_df = gr.DataFrame(
                        label="已上传文档列表", interactive=False
                    )

        # --- 选项卡三：检索测试 ---
        with gr.TabItem("检索测试"):
            gr.Markdown(
                "在此测试您的 Embedding+Reranker 模型的检索效果，无需调用 LLM。"
            )
            with gr.Row():
                with gr.Column(scale=3):
                    retrieve_query_input = gr.Textbox(
                        label="输入检索关键词", placeholder="例如：关键词", lines=2
                    )
                with gr.Column(scale=1):
                    retrieve_top_k_input = gr.Number(label="返回数量 (Top K)", value=5)
                    retrieve_button = gr.Button("执行检索", variant="primary")
            with gr.Row():
                retrieved_chunks_df = gr.DataFrame(
                    label="检索到的文本块", interactive=False
                )

    # ===================================================================
    # Gradio 事件绑定
    # ===================================================================

    # 问答选项卡的事件
    ask_submit_button.click(
        fn=call_ask_api, inputs=[question_input], outputs=[answer_output]
    )

    # 文档管理选项卡的事件
    doc_management_tab.select(fn=get_all_docs_bridge, inputs=[], outputs=[file_list_df])
    refresh_docs_button.click(fn=get_all_docs_bridge, inputs=[], outputs=[file_list_df])
    upload_file_button.upload(
        fn=upload_doc_bridge, inputs=[upload_file_button], outputs=[upload_status_text]
    ).then(fn=get_all_docs_bridge, inputs=[], outputs=[file_list_df])
    delete_button.click(
        fn=delete_doc_bridge, inputs=[delete_doc_id_input], outputs=[delete_status_text]
    ).then(fn=get_all_docs_bridge, inputs=[], outputs=[file_list_df])

    # 检索测试选项卡的事件
    retrieve_button.click(
        fn=retrieve_chunks_bridge,
        inputs=[retrieve_query_input, retrieve_top_k_input],
        outputs=[retrieved_chunks_df],
    )
