import asyncio
import importlib
from typing import AsyncGenerator

from sqlalchemy import text
from sqlalchemy.ext.asyncio import async_sessionmaker, create_async_engine
from sqlalchemy.pool import AsyncAdaptedQueuePool
from sqlmodel import SQLModel
from sqlmodel.ext.asyncio.session import AsyncSession

from core.config import settings
from core.logging import get_logger

logger = get_logger()



class DatabaseManager:

    def __init__(self):

        self.async_engine = None
        self.async_session_factory = None

    def initialize(self) -> None:
        """
        Initialize the async engine and session factory.
        """
        try:
            self.async_engine = create_async_engine(
                settings.database_url_async_path,
                poolclass=AsyncAdaptedQueuePool,
                pool_pre_ping=True,
                pool_size=settings.DB_POOL_SIZE,
                max_overflow=settings.DB_MAX_OVERFLOW,
                pool_recycle=settings.DB_POOL_RECYCLE,
                pool_timeout=settings.DB_POOL_TIMEOUT,
            )

            self.async_session_factory = async_sessionmaker(
                self.async_engine, 
                class_=AsyncSession,
                expire_on_commit=False)

            logger.info("Async engine and session factory initialized successfully")
        except Exception as e:
            logger.error(f"Error initializing async engine: {e}")
            raise

    async def create_tables(self) -> None:
        """
        Create the tables for research papers, QA cache, and job status.
        """
        try:
            # Import models to register them with SQLModel
            importlib.import_module("api.models.research_paper")
            importlib.import_module("api.models.paper_qa_cache")
            importlib.import_module("api.models.job_status")
            logger.info("Models loaded successfully")

            async with self.async_engine.begin() as conn:
                await conn.run_sync(SQLModel.metadata.create_all)

            logger.info("Tables created successfully")
        except Exception as e:
            logger.error(f"Error loading models: {e}")
            raise



db_manager = DatabaseManager()
db_manager.initialize()


async def init_db() -> None:
    """
    Initialize the database.
    """
    try:
        
        logger.info("Loading Models and creating tables...")
        await db_manager.create_tables()

        max_retries = 3
        retry_delay = 2

        for attempt in range(max_retries):
            try:
                async with db_manager.async_engine.begin() as conn:
                    await conn.execute(text("SELECT 1"))
                logger.info("Database connection verified successfully")
                break
            except Exception:
                if attempt == max_retries - 1:
                    logger.error(f"Failed to verify database connection after {max_retries} attempts")
                    raise
                logger.warning(f"Database connection attempt {attempt + 1} failed, retrying...")
                await asyncio.sleep(retry_delay * (attempt + 1))

    except Exception as e:
        logger.error(f"Error initializing database: {e}")
        raise


async def get_session() -> AsyncGenerator[AsyncSession, None]:
    """
    Get async session for the database for dependency in Fastapi.
    """
    session = db_manager.async_session_factory()
    try:
        yield session
    except Exception as e:
        logger.error(f"Database session error: {e}")
        if session:
            try:
                await session.rollback()
                logger.info("successfully rolled back session after error")
            except Exception as rollback_error:
                logger.error(f"Error during session rollback: {rollback_error}")
        raise
    finally:
        if session:
            try:
                await session.close()
                logger.debug("Database session closed successfully")
            except Exception as close_error:
                logger.error(f"Error closing database session: {close_error}")