from loguru import logger
from langchain.retrievers import ContextualCompressionRetriever
from langchain_core.documents import Document
from langchain_core.prompts import PromptTemplate
from langchain_core.runnables import (
    RunnableParallel,
    RunnablePassthrough,
    RunnableLambda,
)
from langchain_core.output_parsers import StrOutputParser
from langchain_core.tracers import ConsoleCallbackHandler

from app.core.config import settings
from app.chains.vector_store import get_chroma_vector_store
from app.chains.reranker_loader import get_bge_reranker
from app.chains.llm_loader import get_qwen_llm

# --- RAG 链的核心构建函数 ---


def format_docs(docs: list[Document]) -> str:
    if not docs:
        return "根据已知信息，无法回答该问题。"

    formatted_docs = []
    for i, doc in enumerate(docs):
        # 暂时忽略分数为0的问题，让它显示出来即可
        score_str = f"{doc.metadata.get('relevance_score', 0.0):.4f}"
        source_str = doc.metadata.get("original_filename", "未知来源")

        formatted_docs.append(
            f"--- 相关文档 {i + 1} (来源: {source_str}, 相关度: {score_str}) ---\n"
            f"{doc.page_content}"
        )
    return "\n\n".join(formatted_docs)


def clean_repetition(text: str) -> str:
    lines = text.strip().splitlines()
    seen = set()
    final = []
    for line in lines:
        if line not in seen:
            seen.add(line)
            final.append(line)
    return "\n".join(final)[:300]  # 限制长度


async def create_rag_qa_chain():
    task_logger = logger.bind(chain="rag_qa_simple")
    task_logger.info("正在创建RAG问答链...")

    prompt = PromptTemplate.from_template(
        "你是一位知识助手。请仅基于下列参考资料回答问题，要求语言简洁明确，避免分析、复述、评论。"
        "若资料不足，请直接回答“无法确定”。\n\n参考资料：\n{context}\n\n问题：{question}\n\n答案："
    )

    llm = get_qwen_llm()
    base_retriever = get_chroma_vector_store().as_retriever(
        search_kwargs={"k": settings.INITIAL_RETRIEVAL_TOP_K}
    )
    reranker_compressor = get_bge_reranker()
    compression_retriever = ContextualCompressionRetriever(
        base_compressor=reranker_compressor,
        base_retriever=base_retriever,
        callbacks=[ConsoleCallbackHandler()],
    )

    rag_chain = (
        {
            "context": compression_retriever | format_docs,
            "question": RunnablePassthrough(),
        }
        | prompt
        | llm
        | StrOutputParser()
        # | RunnableLambda(clean_repetition)
    )
    task_logger.success("最简版RAG问答链创建成功！")
    return rag_chain


async def get_standalone_retriever(
    query: str, top_k: int, reranker=None
) -> list[Document]:
    base_retriever = get_chroma_vector_store().as_retriever(search_kwargs={"k": top_k})
    reranker = reranker or get_bge_reranker()
    compression_retriever = ContextualCompressionRetriever(
        base_compressor=reranker, base_retriever=base_retriever
    )
    return await compression_retriever.ainvoke(query)
