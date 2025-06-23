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

    # --- RAG 核心配置 (已更新为 Ollama) ---
    OLLAMA_BASE_URL: str = "http://localhost:11434"  # Ollama API 地址
    OLLAMA_EMBEDDING_MODEL: str = "bge-m3:latest"  # Embedding 模型
    OLLAMA_LLM_MODEL: str = "deepseek-r1:1.5b"  # LLM 模型

    # 检索参数
    CHUNK_SIZE: int = 800
    CHUNK_OVERLAP: int = 100
    FINAL_CONTEXT_TOP_K: int = 10

    model_config = SettingsConfigDict(
        env_file=(".env", ".env.local"), env_file_encoding="utf-8"
    )


@lru_cache()
def get_settings():
    return Settings()


settings = get_settings()
