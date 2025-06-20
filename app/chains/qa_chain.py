import re

from loguru import logger
from langchain.retrievers import ContextualCompressionRetriever
from langchain_core.documents import Document

# from langchain_core.prompts import PromptTemplate
from langchain_core.runnables import RunnablePassthrough, RunnableLambda
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


# 将 LangChain prompt 转换为 Qwen3 聊天模板格式
def apply_qwen3_chat_template(input_dict: dict) -> str:
    # 这里的 tokenizer 应该和 get_qwen_llm 里的是同一个实例
    # 由于 get_qwen_llm 做了 lru_cache，我们可以安全地再次调用它来获取 tokenizer
    llm_instance = get_qwen_llm()
    # 访问 HuggingFacePipeline 内部的 tokenizer
    tokenizer = llm_instance.pipeline.tokenizer

    context = input_dict["context"]
    question = input_dict["question"]

    # 定义 Qwen3 的 Prompt 结构，包含 RAG 指令
    # 这里的指令需要融入到 system role 或 user role 中
    # 我们可以将 RAG 指令作为 system role，或者作为 user role 的一部分

    # 这里使用 system role 来承载 RAG 指令，这样更清晰
    system_instruction = (
        "你是一位知识助手。请严格按照“参考资料”回答“问题”。\n"
        "你的回答必须满足以下要求：\n"
        "1. 基于参考资料，**提供全面且直接的答案**。\n"  # 修改这里，强调全面
        "2. **避免多余的寒暄或分析**，直接进入答案内容。\n"  # 放松禁止分析，改为避免多余的寒暄和分析
        "3. 如果“参考资料”无法回答“问题”，请直接回复“无法确定”。\n"
        "### 参考资料 ###\n"
        f"{context}\n"
    )

    # 用户的实际问题
    user_message = f"### 问题 ###\n{question}\n\n### 答案 ###\n"

    messages = [
        {"role": "system", "content": system_instruction},
        {"role": "user", "content": user_message},
    ]

    # 使用 Qwen3 的 apply_chat_template
    # enable_thinking=False 禁用思考模式，更适合RAG直接出答案
    # add_generation_prompt=True 让模型知道它应该开始生成回答
    # 文档说：For non-thinking mode, the model will not generate any think content and will not include a <think>...</think> block.
    # 所以这里不再需要 `<|STOP|>` 这种额外的停止词，依靠模型内部的 `eos_token_id` 和上下文停止。
    # 如果仍然有幻觉或停止问题，再考虑重新引入自定义停止符。
    formatted_text = tokenizer.apply_chat_template(
        messages,
        tokenize=False,
        add_generation_prompt=True,
        enable_thinking=False,  # 关键：禁用思考模式
    )

    # Qwen3 的 chat template 在 messages 末尾添加了 prompt for generation
    # 通常会是 "<|im_start|>assistant\n" 或类似。
    # 所以不再需要 clean_after_answer_tag 了，因为我们直接控制了输入格式。

    return formatted_text


# Qwen3在 non-thinking mode 下，不会生成 <think>...</think>，
# 并且会按照聊天格式来生成。
# 所以这里我们不再需要特殊的 `clean_after_answer_tag` 来处理 <|answer|> 或 <|STOP|>
# 相反，我们需要处理 Qwen3 聊天模板可能带来的额外前缀，例如 "<|im_start|>assistant\n"
def clean_qwen3_output(text: str) -> str:
    # 移除 Qwen3 聊天模板生成回答时可能添加的前缀
    # 例如："<|im_start|>assistant\n"
    # 或者其他可能的起始标记

    # 假设模型的回答以 <|im_start|>assistant 或类似开头
    # 并且以 <|im_end|> 结束（如果设置了 eos_token_id）

    # 尝试匹配 Qwen3 assistant 回答的常见前缀并移除
    match = re.search(r"<\|im_start\|>assistant\n(.*)", text, re.DOTALL)
    if match:
        cleaned_text = match.group(1).strip()
    else:
        cleaned_text = text.strip()

    # 移除可能的 <|im_end|> 标记
    cleaned_text = cleaned_text.replace("<|im_end|>", "").strip()
    return cleaned_text


# 不得已才需要用的输出答案修剪
def clean_after_answer_tag(text: str) -> str:
    # 删除 <|answer|> 后所有内容
    result = re.split(r"<\|answer\|>", text, maxsplit=1)
    return result[1].strip() if len(result) > 1 else text.strip()


async def create_rag_qa_chain():
    task_logger = logger.bind(chain="rag_qa_simple")
    task_logger.info("正在创建RAG问答链...")

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
        # | prompt
        | RunnableLambda(apply_qwen3_chat_template)
        | llm
        | StrOutputParser()
        | RunnableLambda(clean_after_answer_tag)  # 用于输出答案修剪
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
