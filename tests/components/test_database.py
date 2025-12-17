import pytest
import pytest_asyncio
from sqlalchemy import text
from sqlmodel import select
from datetime import datetime

from core.db import db_manager, get_session
from core.config import settings
from api.models.paper_qa_cache import PaperQACache
from api.services.crud import generate_question_hash
from api.services.crud import (
    create_or_update_paper,
    get_paper_by_arxiv_id,
    cache_qa_result,
    get_cached_qa,
    create_job,
    get_job_status,
    update_job_status
)
from core.logging import get_logger

logger = get_logger()

TEST_ARXIV_ID = "9999.99999"
TEST_JOB_ID = "test-job-pytest-001"


@pytest_asyncio.fixture
async def db_session_with_cleanup():
    """Fixture that provides session and handles cleanup."""
    # Cleanup before test
    async with db_manager.async_engine.begin() as conn:
        await conn.execute(text(f"DELETE FROM paper_qa_cache WHERE arxiv_id = '{TEST_ARXIV_ID}'"))
        await conn.execute(text(f"DELETE FROM job_status WHERE job_id = '{TEST_JOB_ID}'"))
        await conn.execute(text(f"DELETE FROM research_papers WHERE arxiv_id = '{TEST_ARXIV_ID}'"))
    
    # Provide session for test
    async with db_manager.async_session_factory() as session:
        yield session
    
    # Cleanup after test (still in same async context)
    async with db_manager.async_engine.begin() as conn:
        await conn.execute(text(f"DELETE FROM paper_qa_cache WHERE arxiv_id = '{TEST_ARXIV_ID}'"))
        await conn.execute(text(f"DELETE FROM job_status WHERE job_id = '{TEST_JOB_ID}'"))
        await conn.execute(text(f"DELETE FROM research_papers WHERE arxiv_id = '{TEST_ARXIV_ID}'"))


@pytest.mark.asyncio
async def test_db_pipeline(db_session_with_cleanup):
    """Test Paper,Q&A and job operations."""
    # First, create the paper (required for foreign key)
    test_paper = await create_or_update_paper(
        session=db_session_with_cleanup,
        arxiv_id=TEST_ARXIV_ID,
        title="Attention Is All You Need",
        authors=["Vaswani, Ashish", "Shazeer, Noam"],
        published_date=datetime(2017, 6, 12),
        last_modified_date=datetime(2017, 12, 6),
        abstract="The dominant sequence transduction models...",
        summary_json={
            "main_contribution": "Transformer architecture",
            "key_findings": ["Self-attention", "Parallelization"]
        },
        chunk_count=45,
        vector_store_status="created"
    )
    
    # Now cache a Q&A result
    cached = await cache_qa_result(
        session=db_session_with_cleanup,
        arxiv_id=TEST_ARXIV_ID,
        question="What is the main contribution?",
        answer="The main contribution is the Transformer architecture...",
        citations="[Chunk 5, Chunk 12]",
        hallucination_score=15.56,
        hallucination_risk="LOW",
        faithfulness_score=0.92,
        answer_relevancy_score=0.88,
        response_metadata={
            "citation_verification": 0.0,
            "nli_verification": 0.0
        }
    )

    assert test_paper.arxiv_id == TEST_ARXIV_ID
    retrieved_paper = await get_paper_by_arxiv_id(db_session_with_cleanup, TEST_ARXIV_ID)
    assert retrieved_paper is not None, "Failed to retrieve paper"
    assert retrieved_paper.title == "Attention Is All You Need"
    assert retrieved_paper.chunk_count == 45

    assert cached.id is not None

    # Retrieve from cache using the SAME session
    question_hash = generate_question_hash("What is the main contribution?")
    statement = select(PaperQACache).where(
        PaperQACache.arxiv_id == TEST_ARXIV_ID,
        PaperQACache.question_hash == question_hash
    )
    result = await db_session_with_cleanup.execute(statement)
    retrieved = result.scalar_one_or_none()

    assert retrieved is not None, "Failed to retrieve cached Q&A"
    assert retrieved.hallucination_risk == "LOW"

    retrieved.access_count += 1
    db_session_with_cleanup.add(retrieved)
    await db_session_with_cleanup.commit()
    await db_session_with_cleanup.refresh(retrieved)
    assert retrieved.access_count >= 1

    # Create a job to test job status operations
    job = await create_job(
        session=db_session_with_cleanup,
        job_id=TEST_JOB_ID,
        arxiv_id=TEST_ARXIV_ID,
        task_name="process_paper_analysis",
        status="pending"
    )
    assert job.job_id == TEST_JOB_ID
    assert job.status == "pending"
        
    # Update job to processing
    updated = await update_job_status(
        session=db_session_with_cleanup,
        job_id=TEST_JOB_ID,
        status="processing",
        progress=50
    )
    
    assert updated.status == "processing"
    assert updated.progress == 50
        
    # Complete the job
    completed = await update_job_status(
        session=db_session_with_cleanup,
        job_id=TEST_JOB_ID,
        status="completed",
        result={"answer": "Test answer"},
        progress=100
    )
    assert completed.status == "completed"
    assert completed.progress == 100
    assert completed.completed_at is not None