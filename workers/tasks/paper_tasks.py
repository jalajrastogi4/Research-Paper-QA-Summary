# import nest_asyncio
# nest_asyncio.apply()

from celery import Task
from typing import Dict, Any, List, Optional
from uuid import uuid4
import asyncio

from workers.celery_config import celery_app
from api.services.paper_service import PaperService
from api.services.evaluation_service import EvaluationService
from api.services.crud import create_job, update_job_status
from core.db import db_manager
from core.logging import get_logger

logger = get_logger()


def run_async(coro):
    try:
        loop = asyncio.get_running_loop()
    except RuntimeError:
        loop = None

    if loop is not None and loop.is_running():
        raise RuntimeError(
            "Celery task is running inside an active event loop. "
            "Async Celery pools are not supported with this setup."
        )

    return asyncio.run(coro)


@celery_app.task(bind=True, name="analyze_paper", queue="paper_analysis")
def analyze_paper(
    self,
    job_id: str,
    arxiv_id: str,
    question: str,
    use_cache: bool = True
) -> Dict[str, Any]:
    """
    Celery task wrapper fpr Paper Q&A.
    """
    try:
        # import asyncio
        return run_async(
            _process_paper_analysis(
                job_id=job_id,
                arxiv_id=arxiv_id,
                question=question,
                use_cache=use_cache
            )
        )
        # return asyncio.run(_process_paper_analysis(job_id, arxiv_id, question, use_cache))
    except Exception as e:
        logger.error(f"Q&A failed for job {job_id}: {str(e)}")
        raise


async def _process_paper_analysis(
    job_id: str,
    arxiv_id: str,
    question: str,
    use_cache: bool = True
) -> Dict[str, Any]:
    """
    Celery based async implementation of paper analysis with job status updates.
    """
    db_manager.initialize()
    async with db_manager.async_session_factory() as session:
        try:
            paper_service = PaperService(session)

            await update_job_status(
                session=session,
                job_id=job_id,
                status="processing"
            )

            logger.info(f"[Task {job_id}] Starting analysis for {arxiv_id}")

            result = await paper_service.analyze_paper(arxiv_id, question, use_cache)

            await update_job_status(
                session=session,
                job_id=job_id,
                status="completed",
                result=result
            )

            await session.commit()

            logger.info(f"[Task {job_id}] Completed analysis for {arxiv_id}")
            return {"status": "completed", "job_id": job_id}

        except Exception as e:
            logger.error(f"[Task {job_id}] Failed analysis for {arxiv_id}: {str(e)}")
            await session.rollback()
            await update_job_status(
                session=session,
                job_id=job_id,
                status="failed",
                error=str(e)
            )
            await session.commit()
            raise
        finally:
            await db_manager.async_engine.dispose()
            db_manager.async_engine = None
            db_manager.async_session_factory = None



@celery_app.task(bind=True, name="evaluate_research_agent", queue="paper_analysis")
def evaluate_research_agent(
    self,
    job_id: str,
    test_cases: Optional[List[Dict]] = None
) -> Dict[str, Any]:
    """
    Celery task wrapper fpr Agent Evaluation.
    """
    try:
        # import asyncio
        return run_async(
            _process_evaluation(
                job_id=job_id,
                test_cases=test_cases
            )
        )
        # return asyncio.run(_process_evaluation(job_id, test_cases))
    except Exception as e:
        logger.error(f"Evaluation failed for job {job_id}: {str(e)}")
        raise


async def _process_evaluation(job_id: str, test_cases: Optional[List[Dict]] = None) -> Dict[str, Any]:
    """
    Celery based async implementation of Agent Evaluation.
    """
    db_manager.initialize()
    async with db_manager.async_session_factory() as session:
        try:
            evaluation_service = EvaluationService()

            await update_job_status(
                session=session,
                job_id=job_id,
                status="processing"
            )

            logger.info(f"[Task {job_id}] Starting evaluation")

            result = await evaluation_service.run_evaluation(test_cases)

            await update_job_status(
                session=session,
                job_id=job_id,
                status="completed",
                result=result
            )

            await session.commit()

            logger.info(f"[Task {job_id}] Completed evaluation")
            return {"status": "completed", "job_id": job_id}
        
        except Exception as e:
            logger.error(f"[Task {job_id}] Failed evaluation: {str(e)}")
            await session.rollback()
            await update_job_status(
                session=session,
                job_id=job_id,
                status="failed",
                error=str(e)
            )
            await session.commit()
            raise
        finally:
            await db_manager.async_engine.dispose()
            db_manager.async_engine = None
            db_manager.async_session_factory = None
