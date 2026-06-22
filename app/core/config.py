from pydantic_settings import BaseSettings, SettingsConfigDict
from typing import Optional
import os


class Settings(BaseSettings):
  """
  Application settings loaded from environment variables or .env file.
  Pydantic will automatically validate types.
  """
  PROJECT_NAME: str = "Financial RAG API"
  API_V1_STR: str = "/api/v1"

  # JWT
  JWT_SECRET_KEY: str = "change-me-extremely-secret"
  JWT_ALGORITHM: str = "HS256"
  ACCESS_TOKEN_EXPIRE_MINUTES: int = 1440

  #  Database Connections
  POSTGRES_URL: Optional[str] = None

  #  Qdrant
  QDRANT_URL: Optional[str] = None
  QDRANT_API_KEY: Optional[str] = None

  # LLM
  OPENAI_API_KEY: Optional[str] = None

  model_config = SettingsConfigDict(
    env_file=".env",
    env_file_encoding="utf-8",
    extra="ignore"
  )


# Global settings instance
settings = Settings()