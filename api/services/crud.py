"""
CRUD operations for database models.

Provides async functions for:
- Research paper management
- Q&A cache operations
- Job status tracking
"""

from datetime import datetime
from typing import Optional, List, Dict, Any
from sqlmodel.ext.asyncio.session import AsyncSession
from sqlmodel import select
import hashlib

from api.models.research_paper import ResearchPaper
from api.models.paper_qa_cache import PaperQACache
from api.models.job_status import JobStatus
from core.logging import get_logger

logger = get_logger()


# ===== Research Paper CRUD =====

async def get_paper_by_arxiv_id(
    session: AsyncSession,
    arxiv_id: str
) -> Optional[ResearchPaper]:
    """Get a research paper by arXiv ID."""
    statement = select(ResearchPaper).where(ResearchPaper.arxiv_id == arxiv_id)
    result = await session.execute(statement)
    return result.scalar_one_or_none()


async def create_or_update_paper(
    session: AsyncSession,
    arxiv_id: str,
    title: str,
    authors: List[str],
    last_modified_date: datetime,
    published_date: Optional[datetime] = None,
    abstract: Optional[str] = None,
    summary_json: Optional[dict] = None,
    chunk_count: Optional[int] = None,
    vector_store_status: str = "pending"
) -> ResearchPaper:
    """Create or update a research paper."""
    existing_paper = await get_paper_by_arxiv_id(session, arxiv_id)
    
    if existing_paper:
        # Update existing paper
        existing_paper.title = title
        existing_paper.authors = authors
        existing_paper.last_modified_date = last_modified_date
        existing_paper.published_date = published_date
        existing_paper.abstract = abstract
        existing_paper.summary_json = summary_json
        existing_paper.chunk_count = chunk_count
        existing_paper.vector_store_status = vector_store_status
        existing_paper.cache_updated_at = datetime.utcnow()
        
        session.add(existing_paper)
        await session.commit()
        await session.refresh(existing_paper)
        
        logger.info(f"Updated paper: {arxiv_id}")
        return existing_paper
    else:
        # Create new paper
        new_paper = ResearchPaper(
            arxiv_id=arxiv_id,
            title=title,
            authors=authors,
            published_date=published_date,
            last_modified_date=last_modified_date,
            abstract=abstract,
            summary_json=summary_json,
            chunk_count=chunk_count,
            vector_store_status=vector_store_status
        )
        
        session.add(new_paper)
        await session.commit()
        await session.refresh(new_paper)
        
        logger.info(f"Created paper: {arxiv_id}")
        return new_paper


# ===== Q&A Cache CRUD =====

def generate_question_hash(question: str) -> str:
    """Generate MD5 hash of normalized question for cache lookup."""
    normalized = question.lower().strip()
    return hashlib.md5(normalized.encode()).hexdigest()


async def get_cached_qa(
    session: AsyncSession,
    arxiv_id: str,
    question: str
) -> Optional[Dict[str, Any]]:
    """Get cached Q&A result."""
    question_hash = generate_question_hash(question)
    
    statement = select(PaperQACache).where(
        PaperQACache.arxiv_id == arxiv_id,
        PaperQACache.question_hash == question_hash
    )
    result = await session.execute(statement)
    cached_qa = result.scalar_one_or_none()
    
    if cached_qa:
        # Update access tracking
        cached_qa.last_accessed_at = datetime.utcnow()
        cached_qa.access_count += 1
        session.add(cached_qa)
    
        result = {
            "arxiv_id": cached_qa.arxiv_id,
            "question": cached_qa.question,
            "answer": cached_qa.answer,
            "citations": cached_qa.citations,
            "hallucination_score": cached_qa.hallucination_score,
            "hallucination_risk": cached_qa.hallucination_risk,
            "answer_relevancy_score": cached_qa.answer_relevancy_score,
            "cache_timestamp": cached_qa.created_at.isoformat() if cached_qa.created_at else None,
            "cached": True
        }

        await session.commit()
        logger.info(f"Cache hit for {arxiv_id}: {question[:50]}...")
        return result
    
    return None


async def cache_qa_result(
    session: AsyncSession,
    arxiv_id: str,
    question: str,
    answer: str,
    citations: Optional[str] = None,
    hallucination_score: Optional[float] = None,
    hallucination_risk: Optional[str] = None,
    faithfulness_score: Optional[float] = None,
    answer_relevancy_score: Optional[float] = None,
    response_metadata: Optional[dict] = None
) -> PaperQACache:
    """Cache a Q&A result."""
    question_hash = generate_question_hash(question)
    
    # Check if already exists
    existing = await get_cached_qa(session, arxiv_id, question)
    
    if existing:
        # Update existing cache
        existing.answer = answer
        existing.citations = citations
        existing.hallucination_score = hallucination_score
        existing.hallucination_risk = hallucination_risk
        existing.faithfulness_score = faithfulness_score
        existing.answer_relevancy_score = answer_relevancy_score
        existing.response_metadata = response_metadata
        existing.last_accessed_at = datetime.utcnow()
        
        session.add(existing)
        await session.commit()
        await session.refresh(existing)
        
        logger.info(f"Updated cache for {arxiv_id}")
        return existing
    else:
        # Create new cache entry
        new_cache = PaperQACache(
            arxiv_id=arxiv_id,
            question_hash=question_hash,
            question=question,
            answer=answer,
            citations=citations,
            hallucination_score=hallucination_score,
            hallucination_risk=hallucination_risk,
            faithfulness_score=faithfulness_score,
            answer_relevancy_score=answer_relevancy_score,
            response_metadata=response_metadata
        )
        
        session.add(new_cache)
        await session.commit()
        await session.refresh(new_cache)
        
        logger.info(f"Cached Q&A for {arxiv_id}")
        return new_cache


# ===== Job Status CRUD =====

async def create_job(
    session: AsyncSession,
    job_id: str,
    arxiv_id: Optional[str] = None,
    task_name: Optional[str] = None,
    status: str = "pending"
) -> JobStatus:
    """Create a new job status entry."""
    job = JobStatus(
        job_id=job_id,
        arxiv_id=arxiv_id,
        task_name=task_name,
        status=status
    )
    
    session.add(job)
    await session.commit()
    await session.refresh(job)
    
    logger.info(f"Created job: {job_id}")
    return job


async def get_job_status(
    session: AsyncSession,
    job_id: str
) -> Optional[JobStatus]:
    """Get job status by ID."""
    statement = select(JobStatus).where(JobStatus.job_id == job_id)
    result = await session.execute(statement)
    return result.scalar_one_or_none()


async def update_job_status(
    session: AsyncSession,
    job_id: str,
    status: str,
    result: Optional[dict] = None,
    error: Optional[str] = None,
    progress: Optional[int] = None
) -> Optional[JobStatus]:
    """Update job status."""
    job = await get_job_status(session, job_id)
    
    if not job:
        logger.warning(f"Job not found: {job_id}")
        return None
    
    job.status = status
    job.updated_at = datetime.utcnow()
    
    if result is not None:
        job.result = result
    
    if error is not None:
        job.error = error
    
    if progress is not None:
        job.progress = progress
    
    # Update timestamps based on status
    if status == "processing" and not job.started_at:
        job.started_at = datetime.utcnow()
    
    if status in ("completed", "failed") and not job.completed_at:
        job.completed_at = datetime.utcnow()
    
    session.add(job)
    await session.commit()
    await session.refresh(job)
    
    logger.info(f"Updated job {job_id}: {status}")
    return job


async def get_jobs_by_arxiv_id(
    session: AsyncSession,
    arxiv_id: str,
    limit: int = 10
) -> List[JobStatus]:
    """Get recent jobs for a specific arXiv ID."""
    statement = (
        select(JobStatus)
        .where(JobStatus.arxiv_id == arxiv_id)
        .order_by(JobStatus.created_at.desc())
        .limit(limit)
    )
    result = await session.execute(statement)
    return list(result.scalars().all())


async def cleanup_jobs(
    session: AsyncSession,
    cutoff_date: datetime
) -> int:
    """Cleanup old jobs."""
    statement = select(JobStatus).where(
        JobStatus.status.in_(["completed", "failed"]),
        JobStatus.created_at < cutoff_date)
    result = await session.execute(statement)
    jobs = result.scalars().all()
    for job in jobs:
        session.delete(job)
    await session.commit()
    return len(jobs)
