"""
Cache Service for managing paper and QA result caching.

Provides:
- QA result caching (question + answer pairs)
- Paper metadata caching
- Cache validation
"""

from datetime import datetime
from typing import Optional, Dict, Any
from sqlmodel.ext.asyncio.session import AsyncSession

from core.logging import get_logger
from api.services.crud import (
    generate_question_hash, 
    get_cached_qa,
    cache_qa_result,
    get_paper_by_arxiv_id,
    create_or_update_paper
    )

logger = get_logger()


class CacheService:
    """
    Service for managing caching operations.
    
    Tier 1: QA Cache (fastest - instant return)
    Tier 2: Paper Cache (medium - skip processing)
    """
    
    async def check_qa_cache(
        self,
        session: AsyncSession,
        arxiv_id: str,
        question: str
    ) -> Optional[Dict[str, Any]]:
        """
        Check if QA result exists in cache.
        
        Returns cached result immediately if found (Tier 1 cache).
        """
        try:
            cached_qa = await get_cached_qa(session, arxiv_id, question)
            if cached_qa:
                logger.info(f"QA cache HIT for {arxiv_id}")
                logger.info(f"[DEBUG] Returning dict: {cached_qa}")
                return cached_qa
            else:
                logger.info(f"QA cache MISS for {arxiv_id}")
                return None
            
        except Exception as e:
            logger.error(f"Error checking QA cache: {e}")
            return None
    
    async def store_qa_result(
        self,
        session: AsyncSession,
        arxiv_id: str,
        question: str,
        answer: str,
        citations: str,
        hallucination_score: float,
        hallucination_risk: str,
        faithfulness_score: Optional[float] = None,
        answer_relevancy_score: Optional[float] = None,
        retrieved_chunks: Optional[list] = None,
        metadata: Optional[Dict[str, Any]] = None
    ) -> bool:
        """Store QA result in cache."""
        try:
            qa_to_cache = await cache_qa_result(
                session=session,
                arxiv_id=arxiv_id,
                question=question,
                answer=answer,
                citations=citations,
                hallucination_score=hallucination_score,
                hallucination_risk=hallucination_risk,
                faithfulness_score=faithfulness_score,
                answer_relevancy_score=answer_relevancy_score,
                response_metadata=metadata
            )

            logger.info(f"Stored QA result for {arxiv_id}")
            return True
            
        except Exception as e:
            logger.error(f"Error storing QA result: {e}")
            return False
    
    async def check_paper_cache(
        self,
        session: AsyncSession,
        arxiv_id: str,
        current_modified_date: datetime
    ) -> Optional[Dict[str, Any]]:
        """
        Check if paper exists in cache and is current.
        
        Returns paper metadata if cached and current (Tier 2 cache).
        """
        try:
            paper = await get_paper_by_arxiv_id(session, arxiv_id)
            if not paper:
                logger.info(f"Paper cache MISS for {arxiv_id}")
                return None

            result = {
                    "arxiv_id": paper.arxiv_id,
                    "title": paper.title,
                    "authors": paper.authors,
                    "abstract": paper.abstract,
                    "summary_json": paper.summary_json,
                    "chunk_count": paper.chunk_count,
                    "vector_store_status": paper.vector_store_status,
                    "cached": True
                }
            if paper.last_modified_date >= current_modified_date:
                logger.info(f"Paper cache HIT for {arxiv_id} (current)")
                result["is_current"] = True
                return result
            else:
                logger.info(f"Paper cache HIT for {arxiv_id} (outdated - needs update)")
                result["is_current"] = False
                return result
            
        except Exception as e:
            logger.error(f"Error checking paper cache: {e}")
            return None
    
    async def store_paper_result(
        self,
        session: AsyncSession,
        arxiv_id: str,
        title: str,
        authors: list,
        last_modified_date: datetime,
        published_date: Optional[datetime] = None,
        abstract: Optional[str] = None,
        summary_json: Optional[dict] = None,
        chunk_count: Optional[int] = None,
        vector_store_status: str = "completed"
    ) -> bool:
        """Store or update paper in cache."""
        try:
            paper = await create_or_update_paper(
                session=session,
                arxiv_id=arxiv_id,
                title=title,
                authors=authors,
                last_modified_date=last_modified_date,
                published_date=published_date,
                abstract=abstract,
                summary_json=summary_json,
                chunk_count=chunk_count,
                vector_store_status=vector_store_status
            )

            logger.info(f"Stored paper {arxiv_id} in cache")
            return True
            
        except Exception as e:
            logger.error(f"Error storing paper: {e}")
            return False

