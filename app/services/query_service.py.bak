from loguru import logger
from app.chains.qa_chain import create_rag_qa_chain

class QueryService:
    """
    新版的查询服务层。
    它的职责是加载并调用预先构建好的 LangChain RAG 链。
    """
    def __init__(self):
        # 在服务实例化时，直接创建并持有 RAG 链
        self.rag_chain = create_rag_qa_chain()
        logger.info("QueryService 已初始化，并成功创建 RAG 链。")

    async def stream_answer(self, query: str):
        """
        使用 RAG 链处理查询，并以流式方式返回答案。
        
        Args:
            query (str): 用户的查询问题。
            
        Yields:
            str: LLM 生成的答案片段 (token)。
        """
        logger.info(f"开始流式处理查询: '{query}'")
        # LangChain 的 .astream() 方法会返回一个异步生成器
        async for chunk in self.rag_chain.astream(query):
            yield chunk