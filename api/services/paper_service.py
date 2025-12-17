from datetime import datetime
from sqlmodel.ext.asyncio.session import AsyncSession
from typing import Dict, Any, Optional
from uuid import uuid4

from agents.research_agent import ResearchAssistant
from api.services.cache_service import CacheService
from api.services.crud import get_paper_by_arxiv_id
from core.db import get_session
from core.logging import get_logger

logger = get_logger()


class PaperService:
    """
    Main service for paper analysis with postgres db caching.
    """
    
    def __init__(self, session: AsyncSession):
        self.session = session
        self.cache_service = CacheService()
        # self.assistant = ResearchAssistant(session)
    
    async def analyze_paper(
        self,
        arxiv_id: str,
        question: str,
        use_cache: bool = True
    ) -> Dict[str, Any]:
        """
        Analyze paper with question using db based caching.
        """
        job_id = str(uuid4())
        logger.info(f"[{job_id}] Starting paper analysis for {arxiv_id}")
        
        try:
            last_modified_date = None
            if use_cache:
                paper = await get_paper_by_arxiv_id(self.session, arxiv_id)
                cached_qa = await self.cache_service.check_qa_cache(self.session, arxiv_id, question)
                logger.info(f"[DEBUG] cached_qa type: {type(cached_qa)}, value: {cached_qa}")
                if cached_qa:
                    logger.info(f"[{job_id}] QA cache HIT for {arxiv_id}")
                    logger.info(f"[DEBUG] Returning cached result {cached_qa}")
                    if not paper:
                        logger.warning(f"Paper NOT found for {arxiv_id} when {cached_qa} EXISTS - INTEGRITY CHECK REQUIRED. FK CONSTRAINT may NOT be enforced")
                    return {
                        **cached_qa,
                        "title": paper.title if paper else "Unknown",
                        "authors": paper.authors if paper else [],
                        "job_id": job_id,
                    }

                logger.info(f"[DEBUG] cached_qa was falsy, continuing to Tier 2")

                from utils.arxiv_fetcher import fetch_arxiv_paper
                _, _, _, last_modified_date = await fetch_arxiv_paper(arxiv_id)
                cached_paper = await self.cache_service.check_paper_cache(self.session, arxiv_id, last_modified_date)
                if cached_paper and cached_paper["is_current"]:
                    logger.info(f"[{job_id}] Paper cache HIT for {arxiv_id}")
                    result = await self._run_qa_only_workflow(arxiv_id, question, cached_paper)
                    await self.cache_service.store_qa_result(
                        session=self.session,
                        arxiv_id=arxiv_id,
                        question=question,
                        answer=result["answer"],
                        citations=result["citations"],
                        hallucination_score=result["hallucination_score"],
                        hallucination_risk=result["hallucination_risk"],
                        faithfulness_score=result["faithfulness_score"],
                        answer_relevancy_score=result["answer_relevancy_score"],
                        retrieved_chunks=result["retrieved_chunks"],
                        metadata=result["metadata"]
                    )
                    result["job_id"] = job_id
                    result["cached"] = True
                    result["title"] = cached_paper.get("title", "Unknown")
                    result["authors"] = cached_paper.get("authors", [])
                    return result

                logger.info(f"[{job_id}] Paper cache MISS for {arxiv_id} - Running full pipeline")
            
            if last_modified_date is None:
                from utils.arxiv_fetcher import fetch_arxiv_paper
                _, _, _, last_modified_date = await fetch_arxiv_paper(arxiv_id)
            research_assistant = ResearchAssistant(self.session)
            result = await research_assistant.run(arxiv_id, question)
            await self.cache_service.store_paper_result(
                session=self.session,
                arxiv_id=arxiv_id,
                title=result.get('title', 'Unknown'),
                authors=result.get('authors', []),
                last_modified_date=result.get('last_modified_date', last_modified_date),
                published_date=result.get('published_date', last_modified_date),
                abstract=result.get('abstract', "No abstract available"),
                summary_json=result.get('summary', {}),
                chunk_count=result.get('metadata', {}).get('chunk_count', 0),
                vector_store_status="completed"
            )
            await self.cache_service.store_qa_result(
                session=self.session,
                arxiv_id=arxiv_id,
                question=question,
                answer=result.get("answer", ""),
                citations=result.get("citations", ""),
                hallucination_score=result.get("comprehensive_hallucination_check", {}).get("overall_score", 0),
                hallucination_risk=result.get("comprehensive_hallucination_check", {}).get("overall_risk", "UNKNOWN"),
                answer_relevancy_score=result.get("metadata", {}).get("avg_relevance_score", 0),
                retrieved_chunks=result.get("retrieved_chunks", []),
                metadata=result.get("metadata", {})
            )
            result["job_id"] = job_id
            return self._normalize_response(result)
            
        except Exception as e:
            logger.error(f"[{job_id}] Error analyzing paper {arxiv_id}: {e}")
            return {
                "error": str(e),
                "arxiv_id": arxiv_id,
                "question": question,
                "answer": "No answer generated",
                "citations": "No citations mentioned",
                "hallucination_score": 0,
                "hallucination_risk": "UNKNOWN",
                "cached": False,
                "job_id": job_id
            }

    
    async def _run_qa_only_workflow(
        self,
        arxiv_id: str,
        question: str,
        cached_paper: Dict[str, Any]
    ) -> Dict[str, Any]:
        """
        Run QA workflow using cached paper data (skip fetch/parse/summarize/store).
        
        This is for Tier 2 cache hits - paper exists but question is new.
        """
        logger.info(f"Running QA-only workflow for {arxiv_id} (cached paper)")
        
        try:
            research_assistant = ResearchAssistant(self.session)
            result = await research_assistant.run(arxiv_id, question)
            return {
                "question": question,
                "arxiv_id": arxiv_id,
                "answer":result.get("answer", "No answer generated"),
                "citations":result.get("citations", ""),
                "hallucination_score":result.get("comprehensive_hallucination_check", {}).get("overall_score", 0),
                "hallucination_risk":result.get("comprehensive_hallucination_check", {}).get("overall_risk", "UNKNOWN"),
                "faithfulness_score":None,
                "answer_relevancy_score":result.get("metadata", {}).get("avg_relevance_score", 0),
                "retrieved_chunks":result.get("retrieved_chunks", []),
                "metadata":result.get("metadata", {})
            }
            
        except Exception as e:
            logger.error(f"Error in QA-only workflow: {e}")
            raise
    
    async def get_paper_status(
        self,
        arxiv_id: str
    ) -> Optional[Dict[str, Any]]:
        """
        Get cached paper status without running analysis.
        """
        try:
            paper = await get_paper_by_arxiv_id(self.session, arxiv_id)
            if paper:
                return {
                    "arxiv_id": paper.arxiv_id,
                    "title": paper.title,
                    "cached": True,
                    "vector_store_status": paper.vector_store_status,
                    "cache_updated_at": paper.cache_updated_at.isoformat() if paper.cache_updated_at else None,
                    "chunk_count": paper.chunk_count
                }
            else:
                return None
            
        except Exception as e:
            logger.error(f"Error getting paper status: {e}")
            return None

    def _normalize_response(self, result: Dict[str, Any]) -> Dict[str, Any]:
        if "comprehensive_hallucination_check" in result:
            check = result.pop("comprehensive_hallucination_check")
            result["hallucination_score"] = check.get("overall_score", 0)
            result["hallucination_risk"] = check.get("overall_risk", "UNKNOWN")
        
        # Handle nested metadata.avg_relevance_score
        if "metadata" in result and "avg_relevance_score" in result["metadata"]:
            result["answer_relevancy_score"] = result["metadata"]["avg_relevance_score"]
        
        return result