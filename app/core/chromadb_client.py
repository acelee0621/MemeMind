import chromadb

from app.core.logging import get_logger
from app.core.config import settings


logger = get_logger(__name__)


chroma_client = None


def get_chroma_collection():
    """获取或创建 ChromaDB 集合的辅助函数。"""
    global chroma_client
    if chroma_client is None:
        try:
            # 如果 Celery worker 与 ChromaDB Docker 容器在同一个 Docker 网络中，
            # 可以直接使用容器名和容器端口，例如 http://chromadb:8000
            # 如果 Celery worker 运行在宿主机，则使用宿主机IP/localhost 和映射的端口 5500
            # settings.CHROMA_HTTP_ENDPOINT = "http://localhost:5500"
            chroma_client = chromadb.HttpClient(settings.CHROMA_HTTP_ENDPOINT)
            logger.info(f"ChromaDB 客户端已连接到: {settings.CHROMA_HTTP_ENDPOINT}")
        except Exception as e:
            logger.error(
                f"连接 ChromaDB 失败 ({settings.CHROMA_HTTP_ENDPOINT}): {e}",
                exc_info=True,
            )
            raise RuntimeError("无法连接到 ChromaDB") from e

    try:
        # settings.CHROMA_COLLECTION_NAME 是你在配置文件中定义的集合名称，例如 "rag_collection"
        # 你也可以为 bce-embedding 模型指定 embedding_function，但由于我们手动生成，可以不指定        
        # 对于 BCE，其维度是 768
        collection = chroma_client.get_or_create_collection(
            name=settings.CHROMA_COLLECTION_NAME,            
            metadata={
                "hnsw:space": "cosine",  # 指定距离度量方法
                "embedding_dimensions": 768,  # 显式声明嵌入维度
            },
        )
        logger.info(f"已获取或创建 ChromaDB 集合: {settings.CHROMA_COLLECTION_NAME}")
        return collection
    except Exception as e:
        logger.error(
            f"获取或创建 ChromaDB 集合 '{settings.CHROMA_COLLECTION_NAME}' 失败: {e}",
            exc_info=True,
        )        
        raise RuntimeError("无法获取或创建 ChromaDB 集合") from e
