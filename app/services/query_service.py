from loguru import logger
from app.chains.qa_chain import create_rag_qa_chain


class QueryService:
    """
    新版的查询服务层。
    它的职责是加载并调用预先构建好的 LangChain RAG 链。
    """

    def __init__(self):
        # 避免在 __init__ 中直接调用异步函数
        self.rag_chain = None
        logger.info("QueryService 已初始化，等待异步加载 RAG 链。")

    @classmethod
    async def create(cls):
        """异步创建 QueryService 实例"""
        instance = cls()
        instance.rag_chain = await create_rag_qa_chain()  # 异步调用
        logger.info("QueryService 已异步加载 RAG 链。")
        return instance

    async def stream_answer(self, query: str):
        """
        使用 RAG 链处理查询，并以流式方式返回答案。

        Args:
            query (str): 用户的查询问题。

        Yields:
            str: LLM 生成的答案片段 (token)。
        """
        logger.info(f"开始流式处理查询: '{query}'")
        if self.rag_chain is None:
            raise RuntimeError("RAG 链未初始化，请使用 QueryService.create() 创建实例")
        async for chunk in self.rag_chain.astream(query):
            yield chunk
