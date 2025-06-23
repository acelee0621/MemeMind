# app/chains/qa_chain.py

from loguru import logger
from langchain_core.documents import Document
from langchain_core.prompts import ChatPromptTemplate
from langchain_core.runnables import RunnablePassthrough
from langchain_core.output_parsers import StrOutputParser

from app.core.config import settings
from app.chains.vector_store import get_chroma_vector_store
from app.chains.llm_loader import get_qwen_llm


def format_docs(docs: list[Document]) -> str:
    """将检索到的文档格式化为字符串，作为上下文。"""
    if not docs:
        return "没有找到相关信息。"

    # 我们不再有 reranker 的 relevance_score，所以简化格式
    return "\n\n".join(
        f"--- 相关文档 {i + 1} (来源: {doc.metadata.get('original_filename', '未知来源')}) ---\n{doc.page_content}"
        for i, doc in enumerate(docs)
    )


async def create_rag_qa_chain():
    """
    创建一个基于 Ollama 的、无 Reranker 的 RAG 问答链。
    """
    task_logger = logger.bind(chain="rag_qa_ollama")
    task_logger.info("正在创建基于 Ollama 的 RAG 问答链...")

    llm = get_qwen_llm()
    #  retriever 直接从 vector store 创建，不再需要压缩或精排
    base_retriever = get_chroma_vector_store().as_retriever(
        search_kwargs={"k": settings.FINAL_CONTEXT_TOP_K}
    )

    # 创建一个更通用的 Prompt 模板
    # ChatOllama 模型能很好地处理这种角色分明的对话格式
    prompt = ChatPromptTemplate.from_template(
        """
你是一个严格按照指令执行的问答机器人。
你的唯一任务是根据下面提供的“参考资料”来回答用户提出的“问题”。

**严格遵守以下规则**:
1.  **直接回答问题**：不要进行任何形式的自我介绍、打招呼或说开场白（例如不要说“根据参考资料...”）。
2.  **内容完全基于资料**：你的回答必须 **100%** 来自“参考资料”，不得包含任何外部知识或个人推断。
3.  **禁止内心戏**：**绝对禁止**在你的回答中输出任何思考过程、分析或类似 `<think>` 的标签。你的输出应该是干净、直接的最终答案。
4.  **资料不足则坦白**：如果“参考资料”中没有足够的信息来回答“问题”，你的唯一回答应该是：“根据提供的资料，无法回答该问题。”
5.  **保持简洁**：在确保回答准确的前提下，语言应尽可能简洁明了。

---
**参考资料**:
{context}
---
**问题**:
{question}
---
**你的回答**:
"""
    )

    rag_chain = (
        {"context": base_retriever | format_docs, "question": RunnablePassthrough()}
        | prompt
        | llm
        | StrOutputParser()
    )

    task_logger.success("基于 Ollama 的 RAG 问答链创建成功！")
    return rag_chain


async def get_standalone_retriever(query: str, top_k: int) -> list[Document]:
    """
    一个独立的检索器，仅执行向量检索，用于调试。
    """
    base_retriever = get_chroma_vector_store().as_retriever(search_kwargs={"k": top_k})
    return await base_retriever.ainvoke(query)
