from functools import lru_cache
from loguru import logger
from langchain_ollama.embeddings import OllamaEmbeddings
from app.core.config import settings

@lru_cache(maxsize=1)
def get_bge_embeddings() -> OllamaEmbeddings:
    """
    加载并缓存通过 Ollama 服务的 BGE embedding 模型。
    """
    logger.info(f"开始初始化 Ollama Embedding 模型: {settings.OLLAMA_EMBEDDING_MODEL}...")

    try:
        # 直接初始化 OllamaEmbeddings，指向配置好的模型
        embeddings = OllamaEmbeddings(
            model=settings.OLLAMA_EMBEDDING_MODEL,
            base_url=settings.OLLAMA_BASE_URL,
        )
        logger.success(f"Ollama Embedding 模型 '{settings.OLLAMA_EMBEDDING_MODEL}' 初始化成功。")
        return embeddings
    except Exception as e:
        logger.error(f"初始化 Ollama Embedding 模型失败: {e}", exc_info=True)
        raise