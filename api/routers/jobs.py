from fastapi import APIRouter, HTTPException, Depends, status
from sqlmodel.ext.asyncio.session import AsyncSession
from typing import Union, List
from datetime import datetime, timedelta
import uuid
from api.schemas import (
    AnalyzeResponse,
    EvaluationResponse,
    JobStatusRequest,
    JobStatusResponse,

)
from api.schemas.job import JobStatus, JobTaskType
from api.services.paper_service import PaperService
from api.services.crud import create_job, get_job_status, cleanup_jobs, get_jobs_by_arxiv_id
from core.db import get_session
from core.logging import get_logger
from workers.tasks.paper_tasks import analyze_paper, evaluate_research_agent

logger = get_logger()
router = APIRouter(prefix="/jobs", tags=["Jobs"])


@router.get("/{job_id}", response_model=Union[AnalyzeResponse, EvaluationResponse, JobStatusResponse])
async def job_status_check(
    job_id: str,
    session: AsyncSession = Depends(get_session)
) -> Union[AnalyzeResponse, EvaluationResponse, JobStatusResponse]:
    """
    Get the status of a job.
    """
    try:
        job = await get_job_status(session=session, job_id=job_id)
        if job:
            if job.status == JobStatus.completed:
                if job.task_name == JobTaskType.analyze_paper:
                    result = job.result
                    return AnalyzeResponse(**result)
                elif job.task_name == JobTaskType.evaluate_research_agent:
                    result = job.result
                    return EvaluationResponse(**result)
            elif job.status == JobStatus.processing or job.status == JobStatus.pending:
                return JobStatusResponse(
                    job_id=job_id,
                    status=job.status,
                    task_name=job.task_name,
                    created_at=job.created_at
                )
            elif job.status == JobStatus.failed:
                return JobStatusResponse(
                    job_id=job_id,
                    status=job.status,
                    task_name=job.task_name,
                    error=job.error,
                    created_at=job.created_at
                )
        else:
            raise HTTPException(
                status_code=status.HTTP_404_NOT_FOUND,
                detail=f"Job not found: {job_id}"
            )
    except Exception as e:
        logger.error(f"Error getting job status: {e}")
        raise HTTPException(
            status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
            detail=str(e)
        )


@router.get("/{arxiv_id}/jobs", response_model=List[JobStatusResponse])
async def get_arxiv_id_jobs(
    arxiv_id: str,
    session: AsyncSession = Depends(get_session)
) -> List[JobStatusResponse]:
    """
    Get the jobs for a specific arXiv ID.
    """
    try:
        jobs = await get_jobs_by_arxiv_id(session=session, arxiv_id=arxiv_id)
        return [JobStatusResponse(
            job_id=job.job_id,
            status=job.status,
            task_name=job.task_name,
            created_at=job.created_at
        ) for job in jobs]
    except Exception as e:
        logger.error(f"Error getting jobs for arXiv ID: {e}")
        raise HTTPException(
            status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
            detail=str(e)
        )


@router.post("/cleanup", response_model=dict)
async def cleanup_old_jobs(
    days_old: int = 30,
    session: AsyncSession = Depends(get_session)
) -> dict:
    cutoff_date = datetime.now() - timedelta(days=days_old)
    try:
        count = await cleanup_jobs(session=session, cutoff_date=cutoff_date)
        return {"message": f"Cleaned up {count} old jobs"}
    except Exception as e:
        logger.error(f"Error cleaning up old jobs: {e}")
        raise HTTPException(
            status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
            detail=str(e)
        )