from langgraph.graph import StateGraph, END
from datetime import datetime

from agents.state import PaperState, create_error_state 
from core.logging import get_logger

logger = get_logger()


class FetcherAgent:
    def __init__(self):
        self.name = "fetcher_agent"


    async def fetch_paper(self, state: PaperState) -> PaperState:
        """Agent node: Downloads and extracts basic paper info"""
        try:
            from utils.arxiv_fetcher import fetch_arxiv_paper

            text, title, authors = await fetch_arxiv_paper(state["arxiv_id"])
            logger.info(f"Fetched paper: {title}")
            return {
                "raw_text": text,
                "title": title,
                "authors": authors,
                "metadata": {
                    "fetched_at": datetime.now().isoformat(),
                    "status": "success"
                }
            }
        except Exception as e:
            logger.error(f"Failed to fetch paper: {e}")
            return create_error_state(f"Fetch failed: {str(e)}", "fetch")

    async def check_fetch_success(self, state: PaperState) -> str:
        """Conditional edge: Decide next step based on fetch success"""
        if state.get("error"):
            return "handle_error"
        return "to_parser"

    
def create_fetcher_graph():

    fetcher = FetcherAgent()
    workflow = StateGraph(PaperState)

    workflow.add_node("fetch_paper", fetcher.fetch_paper)
    workflow.add_node("handle_error", lambda state: state)

    workflow.set_entry_point("fetch_paper")
    workflow.add_conditional_edges(
        "fetch_paper",
        fetcher.check_fetch_success,
        {"handle_error": "handle_error", "to_parser": END}
    )

    logger.info("Fetcher graph created")
    return workflow.compile()