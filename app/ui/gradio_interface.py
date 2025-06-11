# app/ui/gradio_interface.py

import time
import gradio as gr
import pandas as pd
from loguru import logger
from typing import List

# 直接从后端导入我们的链和数据库/服务创建工具
from app.chains.qa_chain import create_rag_qa_chain, get_qwen_contextual_retriever
from app.core.database import create_engine_and_session_for_celery
from app.services.doc_service import SourceDocumentService
from app.repository.doc_repository import SourceDocumentRepository

# ===================================================================
# Gradio 回调函数 (不再是调用API的桥梁，而是直接执行业务逻辑)
# ===================================================================

async def stream_chat_gradio(query: str, history: List[list]):
    """【问答模块】的流式回调函数"""
    if not query or not query.strip():
        gr.Warning("请输入有效的问题！")
        return
    
    logger.info(f"Gradio Chatbot 收到问题: '{query}'")
    
    # 获取 RAG 链并以流式方式调用
    rag_chain = create_rag_qa_chain()
    full_response = ""
    history.append([query, ""]) # 先在界面上显示用户的问题
    
    # .astream() 返回一个异步生成器，我们在这里迭代它
    async for chunk in rag_chain.astream(query):
        full_response += chunk
        history[-1][1] = full_response # 实时更新 Gradio Chatbot 的显示
        yield "", history # 返回空字符串和更新后的历史记录

# --- 用于文档管理的辅助函数 ---
async def doc_service_provider():
    """一个异步生成器，提供一次性的数据库会话和服务实例"""
    engine, SessionLocal = create_engine_and_session_for_celery()
    async with SessionLocal() as db:
        repo = SourceDocumentRepository(db)
        service = SourceDocumentService(repo)
        yield service
    await engine.dispose()

async def get_all_docs_gradio():
    """【文档管理】获取所有文档列表"""
    logger.info("Gradio 正在刷新文档列表...")
    async for service in doc_service_provider():
        docs_list = await service.get_documents(limit=100, offset=0, order_by="created_at desc")
        if not docs_list:
            return pd.DataFrame(columns=["ID", "文件名", "状态", "块数量", "上传时间"])
        
        df = pd.DataFrame([doc.model_dump() for doc in docs_list])
        df_display = df[["id", "original_filename", "status", "number_of_chunks", "created_at"]].copy()
        df_display.rename(columns={
            "id": "ID", "original_filename": "文件名", "status": "处理状态",
            "number_of_chunks": "块数量", "created_at": "上传时间"
        }, inplace=True)
        return df_display

async def upload_doc_gradio(file_obj):
    """【文档管理】上传文件"""
    if file_obj is None: 
        return "未选择文件"
    
    logger.info(f"Gradio 正在上传文件: {file_obj.name}")
    try:
        # 读取文件内容
        with open(file_obj.name, "rb") as f:
            content_bytes = f.read()
        
        # 直接调用 service
        async for service in doc_service_provider():
            doc_response = await service.add_document(
                file_content=content_bytes,
                filename=file_obj.name,
                content_type=file_obj.type
            )
        success_message = f"文件 '{doc_response.original_filename}' 上传成功！ID: {doc_response.id}"
        logger.info(success_message)
        gr.Info(success_message)
        return success_message
    except Exception as e:
        error_message = f"文件上传失败: {e}"
        logger.error(error_message, exc_info=True)
        gr.Error(error_message)
        return error_message

async def delete_doc_gradio(doc_id_str: str):
    """【文档管理】删除文档"""
    if not doc_id_str or not doc_id_str.strip().isdigit():
        message = "请输入有效的纯数字文档ID"
        gr.Warning(message)
        return message
    
    doc_id = int(doc_id_str)
    logger.info(f"Gradio 正在删除文档 ID: {doc_id}")
    try:
        async for service in doc_service_provider():
            await service.delete_document(doc_id)
        success_message = f"文档 ID: {doc_id} 已成功删除。"
        gr.Info(success_message)
        return success_message
    except Exception as e:
        error_message = f"删除失败: {e}"
        logger.error(error_message, exc_info=True)
        gr.Error(error_message)
        return error_message

async def retrieve_chunks_gradio(query: str):
    """【检索测试】的回调函数"""
    if not query or not query.strip():
        gr.Warning("请输入检索关键词！")
        return pd.DataFrame(), "请输入查询"
        
    t0 = time.monotonic()
    logger.info(f"Gradio 正在执行调试检索: '{query}'")
    try:
        retriever = get_qwen_contextual_retriever()
        retrieved_docs = await retriever.ainvoke(query)
        
        if not retrieved_docs:
            return pd.DataFrame(), "未检索到任何相关内容。"

        data = [
            {
                "相关度分数": doc.metadata.get('relevance_score', 'N/A'),
                "文本块内容": doc.page_content,
                "来源文件名": doc.metadata.get('source', 'N/A').split('/')[-1],
            } for doc in retrieved_docs
        ]
        df = pd.DataFrame(data)
        t1 = time.monotonic()
        duration_str = f"检索完成，总耗时: {t1 - t0:.2f} 秒"
        logger.info(duration_str)
        return df, duration_str
    except Exception as e:
        error_message = f"检索时出错: {e}"
        logger.error(error_message, exc_info=True)
        gr.Error(error_message)
        return pd.DataFrame(), error_message

# ===================================================================
# Gradio UI 界面定义 (使用 gr.Chatbot 优化)
# ===================================================================

with gr.Blocks(title="RAG 应用控制台", theme=gr.themes.Soft()) as rag_demo_ui:
    gr.Markdown("# MemeMind RAG 应用控制台")

    with gr.Tabs():
        with gr.TabItem("智能问答"):
            with gr.Row():
                with gr.Column(scale=4):
                    chatbot = gr.Chatbot(label="对话窗口", height=500)
                    query_input = gr.Textbox(label="您的问题", placeholder="在这里输入你的问题...", show_label=False, container=False)
                with gr.Column(scale=1):
                    clear_button = gr.Button("清除对话", variant="secondary")

        with gr.TabItem("文档管理") as doc_management_tab:
            # (文档管理UI布局保持不变)
            gr.Markdown("管理您的知识库文档：上传新文件、查看已处理文件、删除不需要的文件。")
            with gr.Row():
                with gr.Column(scale=1):
                    upload_file_button = gr.File(label="上传新文档", file_count="single")
                    upload_status_text = gr.Textbox(label="上传状态", interactive=False)
                    gr.Markdown("---")
                    delete_doc_id_input = gr.Textbox(label="输入要删除的文档ID")
                    delete_button = gr.Button("确认删除", variant="stop")
                    delete_status_text = gr.Textbox(label="删除状态", interactive=False)
                with gr.Column(scale=3):
                    refresh_docs_button = gr.Button("刷新文档列表")
                    file_list_df = gr.DataFrame(label="已上传文档列表", interactive=False, height=500)

        with gr.TabItem("检索测试"):
            # (检索测试UI布局保持不变)
            gr.Markdown("在此测试您的 Embedding+Reranker 模型的检索效果，无需调用 LLM。")
            with gr.Row():
                retrieve_query_input = gr.Textbox(label="输入检索关键词", lines=2, scale=3)
                retrieve_button = gr.Button("执行检索", variant="primary", scale=1)
            retrieve_timer_text = gr.Textbox(label="处理耗时", interactive=False)
            retrieved_chunks_df = gr.DataFrame(label="检索到的文本块", interactive=False, wrap=True)

    # --- 事件绑定 ---
    query_input.submit(fn=stream_chat_gradio, inputs=[query_input, chatbot], outputs=[query_input, chatbot])
    clear_button.click(lambda: (None, None), outputs=[query_input, chatbot])
    
    doc_management_tab.select(fn=get_all_docs_gradio, outputs=[file_list_df])
    refresh_docs_button.click(fn=get_all_docs_gradio, outputs=[file_list_df])
    upload_file_button.upload(fn=upload_doc_gradio, inputs=[upload_file_button], outputs=[upload_status_text]).then(fn=get_all_docs_gradio, outputs=[file_list_df])
    delete_button.click(fn=delete_doc_gradio, inputs=[delete_doc_id_input], outputs=[delete_status_text]).then(fn=get_all_docs_gradio, outputs=[file_list_df])

    retrieve_button.click(fn=retrieve_chunks_gradio, inputs=[retrieve_query_input], outputs=[retrieved_chunks_df, retrieve_timer_text])