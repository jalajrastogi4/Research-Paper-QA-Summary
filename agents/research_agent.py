from langgraph.graph import StateGraph, END
from langfuse import observe, get_client
from datetime import datetime
import os

from sqlalchemy.ext.asyncio import AsyncSession

from core.config import settings
from agents.fetcher import create_fetcher_graph
from agents.parser import create_parser_graph
from agents.summarizer import SummarizerAgent
from agents.qa_agent import QAAgent
from agents.cache_loader_agent import CacheLoaderAgent
from agents.hallucination_detector import HallucinationDetector
from agents.vectorstore_agent import VectorStoreAgent
from agents.state import PaperState
from agents.graph_input import GraphInput

from core.logging import get_logger

logger = get_logger()

# Configure Langfuse from settings
os.environ["LANGFUSE_PUBLIC_KEY"] = settings.LANGFUSE_PUBLIC_KEY
os.environ["LANGFUSE_SECRET_KEY"] = settings.LANGFUSE_SECRET_KEY
os.environ["LANGFUSE_HOST"] = settings.LANGFUSE_BASE_URL


class ResearchAssistant:
    def __init__(self, session: AsyncSession):
        self.session = session
        self.fetcher = create_fetcher_graph()
        self.parser = create_parser_graph()
        self.summarizer = SummarizerAgent()
        self.qa = QAAgent()
        self.detector = HallucinationDetector()
        self.vectorstore_agent = VectorStoreAgent()
        self.cache_loader = CacheLoaderAgent(session)

        self.workflow = StateGraph(PaperState)

        self.workflow.add_node("fetch", self.fetcher)
        self.workflow.add_node("parse", self.parser)
        self.workflow.add_node("summarize", self.summarizer.generate_summary)
        self.workflow.add_node("store_vector", self.vectorstore_agent.store_in_vector_db)
        self.workflow.add_node("check_cache", self.cache_loader.check_and_load)
        self.workflow.add_node("load_from_cache", self.cache_loader.load_cached_paper)
        self.workflow.add_node("retrieve_context", self.qa.retrieve_context)
        self.workflow.add_node("generate_answer", self.qa.generate_answer)
        self.workflow.add_node("check_hallucination", self.detector.verify_citations)
        self.workflow.add_node("check_consistency", self.detector.cross_check_answer)
        self.workflow.add_node("comprehensive_check", self.detector.comprehensive_check)

        self.workflow.set_entry_point("fetch")
        self.workflow.add_edge("fetch", "check_cache")
        self.workflow.add_conditional_edges(
            "check_cache",
            self._route_based_on_cache,
            {
                "use_cache": "load_from_cache",
                "process_new": "parse"
            }
        )
        self.workflow.add_edge("load_from_cache", "retrieve_context")

        self.workflow.add_edge("parse", "summarize")
        self.workflow.add_edge("summarize", "store_vector")
        self.workflow.add_edge("store_vector", "retrieve_context")
        
        self.workflow.add_edge("retrieve_context", "generate_answer")
        self.workflow.add_edge("generate_answer", "check_hallucination")
        self.workflow.add_edge("check_hallucination", "check_consistency")
        self.workflow.add_edge("check_consistency", "comprehensive_check")
        self.workflow.add_edge("comprehensive_check", END)

        self.graph = self.workflow.compile()

    def _route_based_on_cache(self, state: PaperState) -> str:
        """
        Decide whether to use cached paper or process new.
        
        Returns:
            "use_cache" if paper is cached and current
            "process_new" if paper needs processing
        """
        paper_cached = state.get("paper_cached", False)
        paper_current = state.get("paper_current", False)
        if paper_cached and paper_current:
            logger.info(f"Using cached paper for {state['arxiv_id']}")
            return "use_cache"
        else:
            logger.info(f"Processing new paper for {state['arxiv_id']}")
            return "process_new"

    @observe
    async def run(self, arxiv_id: str, question: str = None):
        """Main execution with Langfuse tracing"""
        langfuse = get_client()
        langfuse.update_current_trace(
            name="research_assistant",
            input={"arxiv_id": arxiv_id, "question": question}
        )

        validated_input = GraphInput(
            arxiv_id=arxiv_id,
            question=question or "What are the main contributions of this paper?",
            metadata={"start_time": datetime.now().isoformat()}
        )

        initial_state = validated_input.dict()

        result = await self.graph.ainvoke(initial_state)

        langfuse.score_current_trace(
            name="execution_success",
            value=0 if result.get("error") else 1
        )

        return result