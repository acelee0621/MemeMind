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

    # ChromaDB 配置 ...
    CHROMA_HOST: str = "http://localhost"
    CHROMA_PORT: int = 5500 
    CHROMA_COLLECTION_NAME: str = "mememind_rag_collection"

    # Embedding 模型相关
    EMBEDDING_MODEL_PATH: str = "local_models/embedding/Qwen3-Embedding-0.6B"
    EMBEDDING_INSTRUCTION_FOR_RETRIEVAL: str = "为这个句子生成表示以用于检索相关文章"
    EMBEDDING_DIMENSIONS: int = 1024  # 嵌入维度, Qwen 0.6B为1024 Qwen 4B为2560
    CHUNK_SIZE: int = 512
    CHUNK_OVERLAP: int = 50

    # Reranker 相关配置
    RERANKER_MODEL_PATH: str = "local_models/reranker/Qwen3-Reranker-0.6B"
    INITIAL_RETRIEVAL_TOP_K: int = 50  # 第一阶段向量召回的数量
    FINAL_CONTEXT_TOP_N: int = 5  # Rerank 后最终选取的数量
    RERANKER_INSTRUCTION: str = "给定一个网页搜索查询，检索回答该查询的相关段落"

    # LLM 相关配置
    LLM_MODEL_PATH: str = "local_models/llm/Qwen2.5-1.5B-Instruct"
    LLM_SYSTEM_PROMPT: str = "You are a helpful assistant." 

    model_config = SettingsConfigDict(
        env_file=(".env", ".env.local"), env_file_encoding="utf-8"
    )


@lru_cache()
def get_settings():
    return Settings()  # type: ignore[attr-defined]


settings = get_settings()
