import os
import time
import httpx
import gradio as gr
import pandas as pd

# FastAPI 的基础URL，确保它指向你的应用地址
FASTAPI_BASE_URL = "http://127.0.0.1:8000"

# ===================================================================
# Gradio 回调函数
# ===================================================================


async def stream_chat_gradio(query: str, history: list[dict]):
    """【问答模块】通过 httpx 调用流式 API"""
    if not query or not query.strip():
        gr.Warning("请输入有效的问题！")
        return
    
    api_url = f"{FASTAPI_BASE_URL}/query/ask/stream"
    payload = {"query": query}
    history.append({"role": "user", "content": query})
    history.append({"role": "assistant", "content": ""})

    try:
        async with httpx.AsyncClient(timeout=300.0) as client:
            # 使用 client.stream 发起流式请求
            async with client.stream("POST", api_url, json=payload) as response:
                response.raise_for_status()
                async for chunk in response.aiter_text():
                    if chunk:
                        history[-1]["content"] += chunk
                        yield "", history
    except Exception as e:
        history[-1]["content"] = f"请求出错: {e}"
        yield "", history


async def get_all_docs_gradio():
    """【文档管理】通过 httpx 调用 API 获取所有文档列表"""
    api_url = f"{FASTAPI_BASE_URL}/documents"
    try:
        async with httpx.AsyncClient() as client:
            response = await client.get(api_url, params={"limit": 100, "offset": 0})
            response.raise_for_status()
            docs_list = response.json()
        
        if not docs_list:
            return pd.DataFrame(columns=["ID", "文件名", "状态", "块数量", "上传时间"])
        
        df = pd.DataFrame(docs_list)
        df_display = df[["id", "original_filename", "status", "number_of_chunks", "created_at"]].copy()
        df_display['created_at'] = pd.to_datetime(df_display['created_at']).dt.strftime('%Y-%m-%d %H:%M:%S')
        df_display.rename(columns={
            "id": "ID", "original_filename": "文件名", "status": "处理状态",
            "number_of_chunks": "块数量", "created_at": "上传时间"
        }, inplace=True)
        return df_display
    except Exception as e:
        gr.Error(f"无法加载文档列表: {e}")
        return pd.DataFrame()


async def upload_doc_gradio(file_obj: gr.File):
    """【文档管理】通过 httpx 上传文件"""
    if file_obj is None: 
        return "未选择文件"
    
    api_url = f"{FASTAPI_BASE_URL}/documents"
    original_filename = getattr(file_obj, 'orig_name', os.path.basename(file_obj.name))
    
    files = {'file': (original_filename, open(file_obj.name, 'rb'), 'application/octet-stream')}
    
    try:
        async with httpx.AsyncClient(timeout=60.0) as client:
            response = await client.post(api_url, files=files)
            response.raise_for_status()
        success_message = f"文件 '{original_filename}' 上传成功！"
        gr.Info(success_message)
        return success_message
    except Exception as e:
        error_message = f"文件上传失败: {e}"
        gr.Error(error_message)
        return error_message


async def delete_doc_gradio(doc_id_str: str):
    """【文档管理】通过 httpx 删除指定ID文档"""
    if not doc_id_str or not doc_id_str.strip().isdigit():
        return "请输入有效的纯数字文档ID"
    
    doc_id = int(doc_id_str)
    api_url = f"{FASTAPI_BASE_URL}/documents/{doc_id}"
    
    try:
        async with httpx.AsyncClient() as client:
            response = await client.delete(api_url)
            response.raise_for_status()
        success_message = f"文档 ID: {doc_id} 已成功删除。"
        gr.Info(success_message)
        return success_message
    except Exception as e:
        error_message = f"删除失败: {e}"
        gr.Error(error_message)
        return error_message

async def retrieve_chunks_gradio(query: str, top_k: int):
    """【检索测试】通过 httpx 调用 API"""
    if not query or not query.strip():
        return pd.DataFrame(), "请输入查询"

    top_k = int(top_k)
    t0 = time.monotonic()
    api_url = f"{FASTAPI_BASE_URL}/query/retrieve-chunks"
    payload = {"query": query, "top_k": top_k}

    try:
        async with httpx.AsyncClient(timeout=120.0) as client:
            response = await client.post(api_url, json=payload)
            response.raise_for_status()
            retrieved_docs = response.json()

        if not retrieved_docs:
            return pd.DataFrame(), "未检索到任何相关内容。"

        data = [
            {
                "相关度分数": f"{doc['metadata'].get('relevance_score', 0):.4f}",
                "文本块内容": doc['page_content'],
                "来源文件名": doc['metadata'].get('original_filename', '未知来源'),
            } for doc in retrieved_docs
        ]
        
        df = pd.DataFrame(data)
        t1 = time.monotonic()
        duration_str = f"检索完成，总耗时: {t1 - t0:.2f} 秒"
        return df, duration_str
    except Exception as e:
        error_message = f"检索时出错: {e}"
        gr.Error(error_message)
        return pd.DataFrame(), error_message


# ===================================================================
# Gradio UI 界面定义
# ===================================================================

with gr.Blocks(title="RAG 应用控制台", theme=gr.themes.Soft()) as rag_demo_ui:
    gr.Markdown("# MemeMind RAG 应用控制台")
    with gr.Tabs():
        with gr.TabItem("智能问答"):
            # ... (智能问答 Tab 保持不变)
            with gr.Row():
                with gr.Column(scale=4):
                    chatbot = gr.Chatbot(label="对话窗口", height=500, type="messages")
                    query_input = gr.Textbox(
                        label="您的问题",
                        placeholder="在这里输入你的问题...",
                        show_label=False,
                        container=False,
                    )
                with gr.Column(scale=1):
                    clear_button = gr.Button("清除对话", variant="secondary")

        with gr.TabItem("文档管理") as doc_management_tab:
            # ... (文档管理 Tab 保持不变)
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

        with gr.TabItem("检索测试"):
            gr.Markdown(
                "在此测试您的 Embedding+Reranker 模型的检索效果，无需调用 LLM。"
            )
            with gr.Row():
                retrieve_query_input = gr.Textbox(
                    label="输入检索关键词", lines=2, scale=3
                )
                # 添加一个用于输入 top_k 的数字输入框
                retrieve_top_k_input = gr.Number(
                    label="返回数量 (Top K)", value=5, minimum=1, maximum=50, step=1
                )
                retrieve_button = gr.Button("执行检索", variant="primary", scale=1)
            retrieve_timer_text = gr.Textbox(label="处理耗时", interactive=False)
            retrieved_chunks_df = gr.DataFrame(
                label="检索到的文本块", interactive=False, wrap=True
            )

    # --- 事件绑定 ---
    query_input.submit(
        fn=stream_chat_gradio,
        inputs=[query_input, chatbot],
        outputs=[query_input, chatbot],
    )
    clear_button.click(lambda: (None, []), outputs=[query_input, chatbot])

    doc_management_tab.select(fn=get_all_docs_gradio, outputs=[file_list_df])
    refresh_docs_button.click(fn=get_all_docs_gradio, outputs=[file_list_df])
    upload_file_button.upload(
        fn=upload_doc_gradio, inputs=[upload_file_button], outputs=[upload_status_text]
    ).then(fn=get_all_docs_gradio, outputs=[file_list_df])
    delete_button.click(
        fn=delete_doc_gradio, inputs=[delete_doc_id_input], outputs=[delete_status_text]
    ).then(fn=get_all_docs_gradio, outputs=[file_list_df])

    retrieve_button.click(
        fn=retrieve_chunks_gradio,
        inputs=[retrieve_query_input, retrieve_top_k_input],
        outputs=[retrieved_chunks_df, retrieve_timer_text],
    )
