from typing import Literal, Optional
from pathlib import Path
from pydantic_settings import BaseSettings, SettingsConfigDict
from pydantic import Field, validator, field_validator
import os

from core.logging import get_logger

logger = get_logger()

class Settings(BaseSettings):
    # ===== OpenAI & LLM Configuration =====
    OPENAI_API_KEY: str = ""
    LLM_MODEL: str = "gpt-4-turbo"
    LLM_TEMPERATURE: float = 0.0
    EMBEDDINGS_MODEL: str = "text-embedding-3-small"
    
    # ===== Langfuse Observability =====
    LANGFUSE_PUBLIC_KEY: str = ""
    LANGFUSE_SECRET_KEY: str = ""
    LANGFUSE_BASE_URL: str = "https://cloud.langfuse.com"
    
    # ===== Database Configuration =====
    DATABASE_URL: str = ""
    DB_POOL_SIZE: int = 5
    DB_MAX_OVERFLOW: int = 10
    DB_POOL_RECYCLE: int = 3600  # 1 hour
    DB_POOL_TIMEOUT: int = 30
    
    # ===== Redis & Celery Configuration =====
    REDIS_URL: str = ""
    CELERY_BROKER_URL: str = ""
    CELERY_RESULT_BACKEND_URL: str = ""
    
    # ===== Pinecone Configuration =====
    PINECONE_API_KEY: str = ""
    PINECONE_INDEX_NAME: str = "research-papers"
    PINECONE_ENVIRONMENT: str = "us-east-1"
    
    # ===== Retrieval & Chunking Settings =====
    CHUNK_SIZE: int = 1000
    CHUNK_OVERLAP: int = 100
    RETRIEVAL_DOCS: int = 3
    SECTION_PARSER_LIMIT: int = 5000
    
    # ===== Hallucination Detection Weights =====
    CITATION_SCORE: float = 0.4
    LLM_SCORE: float = 0.4
    CONSISTENCY_SCORE: float = 0.2
    
    # ===== Directory Configuration =====
    CHROMADB_DIR: str = "data/chromadb"  # Legacy, will be replaced by Pinecone
    LOGS_DIR: str = "logs"
    ARXIV_DIR: str = "data/arxiv"
    RENDER_DISK_PATH: str = "/data"  # Render persistent disk mount path
    
    # ===== Application Settings =====
    ENVIRONMENT: str = "development"  # development, staging, production
    
    model_config = SettingsConfigDict(
        env_file=".envs/.env.local",
        env_ignore_empty=True,
        extra="ignore",
        case_sensitive=True
    )
    
    # ===== Computed Properties =====
    
    @property
    def chromadb_path(self) -> Path:
        """Legacy property for ChromaDB path"""
        return Path(self.CHROMADB_DIR)
    
    @property
    def database_url_async_path(self) -> str:
        """
        Async database URL for SQLAlchemy.
        Converts postgresql:// to postgresql+asyncpg:// if needed.
        """
        if not self.DATABASE_URL:
            return ""
        
        if self.DATABASE_URL.startswith("postgresql://"):
            return self.DATABASE_URL.replace("postgresql://", "postgresql+asyncpg://", 1)
        elif self.DATABASE_URL.startswith("postgresql+asyncpg://"):
            return self.DATABASE_URL
        else:
            logger.warning(f"Unexpected DATABASE_URL format: {self.DATABASE_URL}")
            return self.DATABASE_URL
    
    @property
    def celery_broker_url(self) -> str:
        """Celery broker URL (defaults to REDIS_URL if not set)"""
        return self.CELERY_BROKER_URL or self.REDIS_URL
    
    @property
    def celery_result_backend_url(self) -> str:
        """Celery result backend URL (defaults to REDIS_URL if not set)"""
        return self.CELERY_RESULT_BACKEND_URL or self.REDIS_URL
    
    @property
    def is_production(self) -> bool:
        """Check if running in production environment"""
        return self.ENVIRONMENT.lower() == "production"
    
    @property
    def is_development(self) -> bool:
        """Check if running in development environment"""
        return self.ENVIRONMENT.lower() == "development"
    
    # ===== Validators =====
    
    @field_validator('OPENAI_API_KEY')
    def validate_openai_key(cls, v):
        if not v:
            raise ValueError("OpenAI API key is required")
        if not v.startswith('sk-'):
            raise ValueError("OpenAI API key must start with 'sk-'")
        if len(v) < 20:
            raise ValueError("OpenAI API key must be at least 20 characters long")
        return v
    
    @field_validator('DATABASE_URL')
    def validate_database_url(cls, v):
        if not v:
            logger.warning("DATABASE_URL not set - database features will be disabled")
            return v
        
        if not (v.startswith('postgresql://') or v.startswith('postgresql+asyncpg://')):
            raise ValueError("DATABASE_URL must start with 'postgresql://' or 'postgresql+asyncpg://'")
        
        return v
    
    @field_validator('PINECONE_API_KEY')
    def validate_pinecone_key(cls, v):
        if not v:
            logger.warning("PINECONE_API_KEY not set - vector store features will be disabled")
            return v
        
        if len(v) < 20:
            raise ValueError("PINECONE_API_KEY appears to be invalid (too short)")
        
        return v


settings = Settings()