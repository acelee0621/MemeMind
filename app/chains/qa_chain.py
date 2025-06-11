# app/chains/qa_chain.py

from typing import List
from loguru import logger
from langchain_core.documents import Document
from langchain_core.runnables import RunnableParallel, RunnablePassthrough
from langchain_core.prompts import ChatPromptTemplate
from langchain_core.output_parsers import StrOutputParser

from app.core.config import settings
from app.chains.vector_store import get_chroma_vector_store
from app.chains.reranker_loader import get_qwen_reranker
from app.chains.llm_loader import get_qwen_llm
from langchain.retrievers import ContextualCompressionRetriever

# --- 1. 创建我们的核心组件：一个带精排功能的检索器 ---

def get_qwen_contextual_retriever():
    """
    创建一个集成了向量检索和 Reranker 精排的 LangChain 检索器。
    """
    logger.info("初始化两阶段检索器 (Vector Search + Reranker)...")
    
    # a. 创建基础的向量检索器
    vector_retriever = get_chroma_vector_store().as_retriever(
        search_kwargs={"k": settings.INITIAL_RETRIEVAL_TOP_K}
    )
    
    # b. 加载我们的 Reranker 组件
    reranker = get_qwen_reranker(top_n=settings.FINAL_CONTEXT_TOP_N)
    
    # c. 使用 ContextualCompressionRetriever 将两者结合
    #    这个检索器会自动完成“先召回、后精排”的流程
    contextual_retriever = ContextualCompressionRetriever(
        base_compressor=reranker,
        base_retriever=vector_retriever,
    )
    logger.success("两阶段检索器初始化成功。")
    return contextual_retriever

# --- 2. 创建完整的 RAG 问答链 ---

def create_rag_qa_chain():
    """
    使用 LCEL (LangChain Expression Language) 组装完整的 RAG 问答链。
    """
    logger.info("正在创建 RAG 问答链...")

    # a. 定义一个函数，用于将检索到的文档列表格式化为字符串上下文
    def format_docs(docs: List[Document]) -> str:
        return "\n---\n".join(doc.page_content for doc in docs)

    # b. 定义我们的提示词模板
    prompt_template = ChatPromptTemplate.from_messages([
        ("system", settings.LLM_SYSTEM_PROMPT),
        ("user", """【指令】根据下面提供的上下文信息来回答用户提出的问题。如果上下文中没有足够的信息来回答问题，请明确说明你无法从已知信息中找到答案，不要编造。请使用中文回答。

【上下文信息】
{context}

【用户问题】
{question}

【回答】""")
    ])

    # c. 加载我们的核心组件
    retriever = get_qwen_contextual_retriever()
    llm = get_qwen_llm()

    # d. 使用 LCEL 组装链
    rag_chain = (
        # RunnableParallel 允许我们并行处理，这里我们将用户的原始问题 (question)
        # 一路直接传递下去，另一路通过检索器 (retriever) 获取上下文 (context)。
        RunnableParallel(
            context=retriever,
            question=RunnablePassthrough()
        )
        # 将检索器输出的 context (文档列表) 使用 format_docs 函数格式化为字符串
        .assign(context=lambda inputs: format_docs(inputs["context"]))
        # 将 context 和 question 填入提示词模板
        | prompt_template
        # 将格式化后的提示词传递给 LLM
        | llm
        # 将 LLM 的输出 (AIMessage) 转换为纯字符串
        | StrOutputParser()
    )
    
    logger.success("RAG 问答链创建成功。")
    return rag_chain