from fastapi import APIRouter, HTTPException, Depends, status
from sqlmodel.ext.asyncio.session import AsyncSession
import uuid
from api.schemas import (
    AnalyzeRequest,
    EvaluationRequest,
    JobStatusResponse,
    PaperStatusResponse
)
from api.services.paper_service import PaperService
from api.services.crud import create_job
from core.db import get_session
from core.logging import get_logger
from workers.tasks.paper_tasks import analyze_paper, evaluate_research_agent

logger = get_logger()
router = APIRouter(prefix="/papers", tags=["Papers"])


@router.post("/analyze", response_model=JobStatusResponse, status_code=status.HTTP_200_OK)
async def analyze_paper_async(
    request: AnalyzeRequest,
    session: AsyncSession = Depends(get_session)
) -> JobStatusResponse:
    """
    Analyze a paper asynchronously and return a job ID.
    """
    try:
        job_id = str(uuid.uuid4())
        job = await create_job(
            session=session,
            job_id=job_id,
            arxiv_id=request.arxiv_id,
            task_name="analyze_paper",
            status="pending"
        )

        analyze_paper.delay(
            job_id=job_id,
            arxiv_id=request.arxiv_id,
            question=request.question,
            use_cache=request.use_cache
        )

        logger.info(f"Job created: {job_id} for paper {request.arxiv_id}")
        return JobStatusResponse(
            job_id=job_id,
            arxiv_id=request.arxiv_id,
            task_name="analyze_paper",
            status="pending",
            created_at=job.created_at
        )
    except Exception as e:
        logger.error(f"Error in sync analysis: {e}")
        raise HTTPException(
            status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
            detail=str(e)
        )


@router.get("/{arxiv_id}/status", response_model=PaperStatusResponse)
async def get_paper_status(
    arxiv_id: str,
    session: AsyncSession = Depends(get_session)
) -> PaperStatusResponse:
    """
    Check if a paper is cached and get its status.
    """
    try:
        paper_service = PaperService(session)

        result = await paper_service.get_paper_status(arxiv_id)
        if result:
            return PaperStatusResponse(**result)
        else:
            return PaperStatusResponse(
                arxiv_id=arxiv_id,
                cached=False,
            )
        
    except Exception as e:
        logger.error(f"Error getting paper status: {e}")
        raise HTTPException(
            status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
            detail=str(e)
        )


@router.post("/evaluate", response_model=JobStatusResponse, status_code=status.HTTP_200_OK)
async def evaluate_agent_async(
    request: EvaluationRequest,
    session: AsyncSession = Depends(get_session)
) -> JobStatusResponse:
    """
    Evaluate the research agent asynchronously and return a job ID.
    """
    try:
        job_id = str(uuid.uuid4())
        job = await create_job(
            session=session,
            job_id=job_id,
            task_name="evaluate_research_agent",
            status="pending"
        )

        evaluate_research_agent.delay(job_id=job_id, test_cases=request.test_cases)

        logger.info(f"Job created: {job_id} for evaluation")
        return JobStatusResponse(
            job_id=job_id,
            task_name="evaluate_research_agent",
            status="pending",
            created_at=job.created_at
        )
    except Exception as e:
        logger.error(f"Error in evaluation: {e}")
        raise HTTPException(
            status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
            detail=str(e)
        )
