from pydantic_settings import BaseSettings, SettingsConfigDict
from typing import Optional


class Settings(BaseSettings):
    """
    Application settings loaded from environment variables or .env file.
    Pydantic will automatically validate types.
    """
    PROJECT_NAME: str = "Financial RAG API"
    API_V1_STR: str = "/api/v1"

    # JWT
    JWT_SECRET_KEY: str
    JWT_ALGORITHM: str = "HS256"
    ACCESS_TOKEN_EXPIRE_MINUTES: int = 1440

    #  Database Connections
    POSTGRES_URL: Optional[str] = None

    #  Qdrant
    QDRANT_URL: Optional[str] = None
    QDRANT_API_KEY: Optional[str] = None

    # S3
    S3_BUCKET_NAME: Optional[str] = None
    S3_ACCESS_KEY: Optional[str] = None
    S3_SECRET_KEY: Optional[str] = None
    S3_ENDPOINT_URL: Optional[str] = None

    # LLM
    OPENAI_API_KEY: Optional[str] = None

    #----RAG SETUP-----
    # RAG storage
    RAG_COLLECTION_NAME: str = "finance_documents"
    RAG_DUCKDB_PATH: str = "/data/finance.duckdb"
    RAG_GLOBAL_WORKSPACE_ID: Optional[str] = None

    # Chunking
    RAG_CHUNK_MAX_TOKENS: int = 1000
    RAG_CHUNK_OVERLAP_TOKENS: int = 120
    RAG_CHUNK_MIN_TOKENS: int = 25
    RAG_CHUNK_MERGE_RATIO: float = 0.85

    # Embeddings
    RAG_USE_OPENAI_EMBEDDINGS: bool = True
    RAG_EMBEDDING_MODEL: str = "text-embedding-3-large"
    RAG_EMBEDDING_DIMENSIONS: int = 3072
    RAG_EMBEDDING_BATCH_SIZE: int = 64
    RAG_EMBEDDING_MAX_INPUT_TOKENS: int = 7500
    RAG_LOCAL_EMBEDDING_MODEL: str = "BAAI/bge-small-en-v1.5"
    RAG_LOCAL_EMBEDDING_BATCH_SIZE: int = 32


    # Retrieval
    RAG_RETRIEVAL_TOP_K: int = 8
    RAG_PREFETCH_MIN: int = 40
    RAG_FUSION_MIN: int = 30
    RAG_FIRST_STAGE_MULTIPLIER: int = 5
    RAG_FUSION_MULTIPLIER: int = 3

    # Reranker
    RAG_RERANKER_MODE: str = "auto"  # off | cpu_light | gpu_heavy | auto
    RAG_CPU_RERANKER_MODEL: str = "cross-encoder/ms-marco-MiniLM-L-6-v2"
    RAG_GPU_RERANKER_MODEL: str = "BAAI/bge-reranker-v2-m3"
    RAG_RERANKER_MAX_LENGTH: int = 1024

    # Runtime
    RAG_INGESTION_CONCURRENCY: int = 1
    RAG_THREAD_POOL_WORKERS: int = 8

    model_config = SettingsConfigDict(
        env_file=".env",
        env_file_encoding="utf-8",
        extra="ignore"
    )


# Global settings instance
settings = Settings()
