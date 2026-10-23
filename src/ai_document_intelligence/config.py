from typing import Literal

from pydantic_settings import BaseSettings, SettingsConfigDict
from functools import lru_cache


class Settings(BaseSettings):
    embedding_model: str = "sentence-transformers/all-MiniLM-L6-v2"
    chroma_collection: str = "multi_document_test"
    chroma_persist_directory: str = "./chroma_multi_db"

    llm_provider: Literal["ollama", "groq"] = "ollama"

    ollama_model: str = "qwen3"
    ollama_base_url: str = "http://localhost:11434"

    groq_model: str = "openai/gpt-oss-20b"
    groq_api_key: str | None = None

    model_config = SettingsConfigDict(
        env_file=".env",
        extra="ignore"
    )

@lru_cache
def get_settings() -> Settings:
    return Settings()