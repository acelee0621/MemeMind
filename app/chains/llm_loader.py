from functools import lru_cache
from loguru import logger
from langchain_ollama.chat_models import ChatOllama
from app.core.config import settings

@lru_cache(maxsize=1)
def get_qwen_llm() -> ChatOllama:
    """
    获取并缓存通过 Ollama 服务的 LLM 实例。
    """
    logger.info(f"开始初始化 Ollama LLM: {settings.OLLAMA_LLM_MODEL}...")

    try:
        # 使用 ChatOllama 类连接到 Ollama 服务
        llm = ChatOllama(
            model=settings.OLLAMA_LLM_MODEL,
            base_url=settings.OLLAMA_BASE_URL,
            temperature=0.7,
            # 你可以在这里添加更多参数, 例如 top_p, top_k 等
        )
        logger.success(f"Ollama LLM '{settings.OLLAMA_LLM_MODEL}' 初始化成功。")
        return llm

    except Exception as e:
        logger.error(f"初始化 Ollama LLM 失败: {e}", exc_info=True)
        raise