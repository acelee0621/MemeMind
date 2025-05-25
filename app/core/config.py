from pydantic_settings import BaseSettings, SettingsConfigDict
from functools import lru_cache


class Settings(BaseSettings):
    app_name: str = "MemeMind"
    BASE_URL: str = "http://localhost:8000"
    JWT_SECRET: str = "your-jwt-secret"
    JWT_ALGORITHM: str = "HS256"
    JWT_EXPIRATION: int = 30
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

    # S3/MinIO 配置
    MINIO_ENDPOINT: str = "localhost:9000"
    MINIO_ACCESS_KEY: str = "minio"
    MINIO_SECRET_KEY: str = "miniosecret"
    MINIO_USE_SSL: bool = False
    MINIO_BUCKET: str = "mememind"
    
    # ChromaDB 配置 ...
    CHROMA_HTTP_ENDPOINT: str = "http://localhost:5500" # ChromaDB HTTP 访问地址
    CHROMA_COLLECTION_NAME: str = "mememind_rag_collection" # ChromaDB 集合名称
    
    # Embedding 模型相关
    EMBEDDING_INSTRUCTION_FOR_RETRIEVAL: str = "为这个句子生成表示以用于检索相关文章："
    CHUNK_SIZE: int = 1000 
    CHUNK_OVERLAP: int = 100
    
    model_config = SettingsConfigDict(
        env_file=(".env", ".env.local"), env_file_encoding="utf-8"
    )


@lru_cache()
def get_settings():
    return Settings()  # type: ignore[attr-defined]


settings = get_settings()
