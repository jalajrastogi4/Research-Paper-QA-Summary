from typing import Literal, Optional
from pathlib import Path
from pydantic_settings import BaseSettings, SettingsConfigDict
from pydantic import Field, validator, field_validator
import os

from core.logging import get_logger

logger = get_logger()

class Settings(BaseSettings):
    OPENAI_API_KEY: str = ""
    LANGFUSE_PUBLIC_KEY: str = ""
    LANGFUSE_SECRET_KEY: str = ""
    LANGFUSE_BASE_URL: str = "https://cloud.langfuse.com"
    
    LLM_MODEL: str = "gpt-4-turbo"
    LLM_TEMPERATURE: float = 0.0
    EMBEDDINGS_MODEL: str = "text-embedding-3-small"
    
    CHUNK_SIZE: int = 1000
    CHUNK_OVERLAP: int = 100
    RETRIEVAL_DOCS: int = 3

    CHROMADB_DIR: str = "data/chromadb"
    LOGS_DIR: str = "logs"
    ARXIV_DIR: str = "data/arxiv"

    SECTION_PARSER_LIMIT: int = 5000

    CITATION_SCORE: float = 0.4
    LLM_SCORE: float = 0.4
    CONSISTENCY_SCORE: float = 0.2

    model_config = SettingsConfigDict(
        env_file=".envs/.env.local",
        env_ignore_empty=True,
        extra="ignore",
        case_sensitive=True
    )

    @property
    def chromadb_path(self) -> Path:
        return Path(self.CHROMADB_DIR)

    @field_validator('OPENAI_API_KEY')
    def validate_openai_key(cls, v):
        if not v:
            raise ValueError("OpenAI API key is required")
        if not v.startswith('sk-'):
            raise ValueError("OpenAI API key must start with 'sk-'")
        if len(v) < 20:
            raise ValueError("OpenAI API key must be at least 20 characters long")
        return v


settings = Settings()