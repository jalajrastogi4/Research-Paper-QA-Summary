"""
Cache Loader Agent for LangGraph workflow.

Checks if paper exists in database and loads cached data to skip redundant processing.
"""

from datetime import datetime
from typing import Dict, Any
from sqlalchemy.ext.asyncio import AsyncSession
from agents.state import PaperState, create_error_state
from api.services.crud import get_paper_by_arxiv_id
from core.logging import get_logger
from core.db import get_session
from core.config import settings

logger = get_logger()


class CacheLoaderAgent:
    """
    Agent to check and load cached paper data from database.
    
    This agent:
    1. Checks if paper exists in research_papers table
    2. Verifies if cached paper is still current (not modified on arXiv)
    3. Loads cached data (summary, chunks, vectors) into state
    """
    def __init__(self, session: AsyncSession) -> None:
        self.session = session
    
    async def check_and_load(self, state: PaperState) -> PaperState:
        """
        Check if paper exists in cache and if it's current.
        
        Sets flags in state to route workflow:
        - paper_cached: True if paper exists in DB
        - paper_current: True if DB version matches arXiv version
        """
        arxiv_id = state["arxiv_id"]
        logger.info(f"Checking cache for paper: {arxiv_id}")
        
        try:
            if state.get("error"):
                logger.error(f"Upstream error detected in cache_loader for {arxiv_id}: {state['error']}")
                return {
                    "paper_cached": False,
                    "paper_current": False,
                    "metadata": {
                        **state.get("metadata", {}),
                        "cache_check_error": state.get("error")
                    }
                }

            if not state.get("last_modified_date"):
                logger.error(f"Missing last_modified_date in state for {arxiv_id}")
                return {
                    "paper_cached": False,
                    "paper_current": False,
                    "metadata": {
                        **state.get("metadata", {}),
                        "cache_check_error": "Missing last_modified_date in state"
                    }
                }

            paper = await get_paper_by_arxiv_id(self.session, arxiv_id)
            if not paper:
                return {
                    "paper_cached": False,
                    "paper_current": False,
                }
                
            is_current = paper.last_modified_date >= state["last_modified_date"]

            return {
                "paper_cached": True,
                "paper_current": is_current
            }
            
        except Exception as e:
            logger.error(f"Error checking cache for {arxiv_id}: {e}")
            return {
                "paper_cached": False,
                "paper_current": False,
                "metadata": {
                    **state.get("metadata", {}),
                    "cache_check_error": str(e)
                }
            }

    
    async def load_cached_paper(self, state: PaperState) -> PaperState:
        """
        Load cached paper data into state to skip processing.
        
        Populates state with:
        - title, authors, abstract (from cached_paper)
        - summary (from cached_paper.summary_json)
        - raw_text (from local PDF or re-fetch)
        
        Skips: parse, summarize, store_vector nodes
        """
        logger.info(f"Loading cached paper data for {state['arxiv_id']}")
        arxiv_id = state["arxiv_id"]
        
        try:
            paper = await get_paper_by_arxiv_id(self.session, arxiv_id)

            if not paper:
                return create_error_state("Paper not found in cache", "cache_loader")

            result = {
                "title": paper.title,
                "authors": paper.authors,
                "abstract": paper.abstract if paper.abstract else "",
                "summary": paper.summary_json,
                "metadata": {
                    **state.get("metadata", {}),
                    "paper_loaded_from_cache": True,
                    "cache_updated_at": paper.cache_updated_at.isoformat(),
                    "chunk_count": paper.chunk_count
                }
            }
            
        except Exception as e:
            logger.error(f"Error loading cached paper: {e}")
            return {
                "error": f"Failed to load cached paper: {e}",
                "metadata": {
                    **state.get("metadata", {}),
                    "cache_load_error": str(e)
                }
            }


        try:
            raw_text = await self._get_paper_text(arxiv_id)
            result["raw_text"] = raw_text
        except Exception as e:
            logger.error(f"Error getting paper text: {e}")
            return create_error_state(f"Failed to get paper text: {e}", "cache_loader")

        return result


    async def _get_paper_text(self, arxiv_id: str) -> str:
        """Get paper text from local PDF or re-fetch if not available."""
        from utils.arxiv_fetcher import fetch_arxiv_paper
        from pathlib import Path

        pdf_files = list(Path(settings.ARXIV_DIR).glob(f"{arxiv_id}*.pdf"))

        if pdf_files:
            import pypdf
            with open(pdf_files[0], "rb") as file:
                reader = pypdf.PdfReader(file)
                return " ".join([page.extract_text() for page in reader.pages])
        else:
            text, _, _, _ = await fetch_arxiv_paper(arxiv_id)
            return text

