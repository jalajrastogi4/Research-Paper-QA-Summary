from fastapi import APIRouter, FastAPI, status
from fastapi.responses import JSONResponse
from contextlib import asynccontextmanager

from api.routers import papers, jobs, health
from core.db import db_manager, init_db
from core.config import settings
from core.logging import get_logger
from core.health import health_checker, ServiceStatus

import asyncio
import time

logger = get_logger()

async def startup_health_check(timeout: float = 90.0) -> bool:
    try:
        async with asyncio.timeout(timeout):
            retry_intervals = [1, 2, 5, 10, 15]
            start_time = time.time()

            while True:
                is_healthy = await health_checker.wait_for_services()
                if is_healthy:
                    return True
                elapsed = time.time() - start_time
                if elapsed >= timeout:
                    logger.error("Services failed health check during startup")
                    return False
                wait_time = retry_intervals[
                    min(len(retry_intervals) - 1, int(elapsed / 10))
                ]
                logger.warning(
                    f"Services not healthy, waiting {wait_time}s before retry"
                )
                await asyncio.sleep(wait_time)
    except asyncio.TimeoutError:
        logger.error(f"Health check timed out after {timeout} seconds")
        return False
    except Exception as e:
        logger.error(f"Error during startup health check: {e}")
        return False


@asynccontextmanager
async def lifespan(app: FastAPI):
    try:
        logger.info("Initializing database...")
        await init_db()
        logger.info("Database initialized successfully")

        await health_checker.add_service("database", health_checker.check_database)
        await health_checker.add_service("celery", health_checker.check_celery)
        await health_checker.add_service("redis", health_checker.check_redis)
        await health_checker.add_service("pinecone", health_checker.check_pinecone)

        # if not await startup_health_check():
        #     # raise RuntimeError("Critical services failed to start")
        #     logger.error(
        #         "Startup health check failed — starting API anyway"
        #     )
        # logger.info("All services initialized and healthy")
        
        yield
    except Exception as e:
        logger.error(f"Error initializing services: {e}")
        if db_manager.async_engine is not None:
            await db_manager.async_engine.dispose()
        await health_checker.cleanup()
        raise
    finally:
        logger.info("Shutting down database and services...")
        if db_manager.async_engine is not None:
            await db_manager.async_engine.dispose()
        await health_checker.cleanup()


app = FastAPI(
    title="Research Agent API", 
    description="API for the Research Agent",
    docs_url=f"/docs",
    redoc_url=f"/redoc",
    version="0.1.0",
    lifespan=lifespan,
)


app.include_router(health.router)
app.include_router(papers.router)
app.include_router(jobs.router)

if __name__ == "__main__":
    import uvicorn
    uvicorn.run(app, host="0.0.0.0", port=8000)