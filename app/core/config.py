from pydantic_settings import BaseSettings, SettingsConfigDict
from functools import lru_cache


class Settings(BaseSettings):
    app_name: str = "MemeMind"
    BASE_URL: str = "http://localhost:8000"
    DEBUG: bool = False

    # PostgreSQL 配置
    POSTGRES_HOST: str = "localhost"
    POSTGRES_PORT: str = "5432"
    POSTGRES_DB: str = "mememind"
    POSTGRES_USER: str = "postgres"
    POSTGRES_PASSWORD: str = "postgres"

    # RabbitMQ 配置
    RABBITMQ_HOST: str = "localhost:5672"
    RABBITMQ_USER: str = "user"
    RABBITMQ_PASSWORD: str = "bitnami"

    # Redis 配置
    REDIS_HOST: str = "localhost:6379"

    # 上传文件路径配置
    LOCAL_STORAGE_PATH: str = "source_documents/"

    # ChromaDB 配置
    CHROMA_HOST: str = "localhost"
    CHROMA_PORT: int = 5500
    CHROMA_COLLECTION_NAME: str = "mememind_rag_collection"

    # --- RAG 核心配置 ---

    # Embedding 模型 (BAAI BGE)
    EMBEDDING_MODEL_PATH: str = "local_models/embedding/bge-large-zh-v1.5"

    # Reranker 模型 (BAAI BGE)
    RERANKER_MODEL_PATH: str = "local_models/reranker/bge-reranker-v2-m3"

    # LLM 模型 (Qwen)
    LLM_MODEL_PATH: str = "local_models/llm/Qwen2.5-1.5B-Instruct"

    # 检索参数
    CHUNK_SIZE: int = 800
    CHUNK_OVERLAP: int = 100
    INITIAL_RETRIEVAL_TOP_K: int = 50  # 向量库粗召回返回的文档数量
    FINAL_CONTEXT_TOP_N: int = 5  # Reranker精排后最终提供给LLM的文档数量

    # LLM 系统提示
    LLM_SYSTEM_PROMPT: str = (
        "你是一个精通知识管理和个人知识库的智能助手。"
        "请根据下面提供的上下文信息，用清晰、结构化、准确的语言回答问题，并聚焦于主题、概念和可靠信息。"
    )

    model_config = SettingsConfigDict(
        env_file=(".env", ".env.local"), env_file_encoding="utf-8"
    )


@lru_cache()
def get_settings():
    return Settings()


settings = get_settings()
