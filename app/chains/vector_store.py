import chromadb
from loguru import logger
from langchain_chroma import Chroma
from app.core.config import settings
from app.chains.embedding_loader import get_bge_embeddings


def get_chroma_vector_store() -> Chroma:
    """
    连接到 ChromaDB 并返回一个 LangChain 兼容的 VectorStore 实例。

    这个函数现在是一个同步的工厂，它内部配置了一个异步客户端。
    LangChain的Chroma类足够智能，可以处理同步和异步操作。
    """
    logger.info("开始初始化 ChromaDB 向量存储组件...")

    try:
        # --- 1. 获取配置 ---
        host = settings.CHROMA_HOST
        port = settings.CHROMA_PORT
        if not host or not port:
            raise ValueError("无效的 ChromaDB 配置, 请检查 CHROMA_HOST 和 CHROMA_PORT")

        logger.info(f"配置 ChromaDB 连接: Host={host}, Port={port}")

        # --- 2. 创建 ChromaDB 异步 HTTP 客户端 ---
        # 改用 chromadb.HttpClient，因为最新版的Langchain-chroma可以智能处理
        # 同步和异步操作。直接使用同步客户端初始化可以简化启动流程。
        # Langchain在执行.ainvoke()时，其内部会高效地处理与数据库的异步通信。
        chroma_client = chromadb.HttpClient(host=host, port=port)

        # --- 3. 获取嵌入函数 ---
        embedding_function = get_bge_embeddings()

        # --- 4. 创建 LangChain 的 Chroma 实例 ---
        vector_store = Chroma(
            client=chroma_client,
            collection_name=settings.CHROMA_COLLECTION_NAME,
            embedding_function=embedding_function,
        )

        logger.success(
            f"ChromaDB 向量存储组件初始化成功。集合: '{settings.CHROMA_COLLECTION_NAME}'"
        )
        return vector_store

    except Exception as e:
        logger.error(f"初始化 ChromaDB 向量存储组件失败: {e}", exc_info=True)
        raise
