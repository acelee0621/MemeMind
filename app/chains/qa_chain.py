# app/chains/qa_chain.py
import re
from loguru import logger
from langchain_core.documents import Document
from langchain_core.prompts import ChatPromptTemplate
from langchain_core.runnables import RunnablePassthrough, RunnableLambda
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


def clean_llm_output(text: str) -> str:
    """
    使用正则表达式移除 <think>...</think> 标签及其内容。
    """
    # re.DOTALL 使得 '.' 可以匹配包括换行在内的任意字符
    cleaned_text = re.sub(r"<think>.*?</think>\s*", "", text, flags=re.DOTALL)
    return cleaned_text.strip()


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
你是一位非常严谨且专业的知识助手。
你的任务是基于下面提供的“参考资料”，为用户提出的“问题”提供一个全面、深入且易于理解的回答。

**你的回答必须遵循以下结构和规则**:

**1. 核心摘要**:
首先，用一两句话清晰、直接地概括问题的核心答案。

**2. 详细解读**:
接下来，在“详细解读”部分，你需要：
   a. **引用原文**: 从“参考资料”中找出与问题最相关的关键句子或段落，并使用 Markdown 的引用格式（以 `>` 开头）来呈现它们。
   b. **深入分析**: 在每一段引用的下方，用你自己的话对引用的内容进行详细、清晰的解释。说明这些规定或步骤在整个流程中的具体作用、前提条件以及上下文。
   c. **综合信息**: 如果多份资料都提到了相关内容，鼓励你将它们结合起来，提供一个更全面的视角。

**基本要求**:
- **忠于原文**: 你所有的分析和解读都必须严格基于“参考资料”，禁止引入任何外部信息或个人推测。
- **格式清晰**: 请严格遵守“核心摘要”和“详细解读”的输出格式。
- **无需赘述**: 不要重复问题，也不要说“根据参考资料...”之类的开场白，直接开始回答。

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
        | RunnableLambda(clean_llm_output)
    )

    task_logger.success("基于 Ollama 的 RAG 问答链创建成功！")
    return rag_chain


async def get_standalone_retriever(query: str, top_k: int) -> list[Document]:
    """
    一个独立的检索器，仅执行向量检索，用于调试。
    """
    base_retriever = get_chroma_vector_store().as_retriever(search_kwargs={"k": top_k})
    return await base_retriever.ainvoke(query)
