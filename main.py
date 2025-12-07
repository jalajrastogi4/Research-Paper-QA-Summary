from langgraph.graph import StateGraph, END
from langfuse import observe, get_client
from datetime import datetime
import os

from core.config import settings
from agents.fetcher import create_fetcher_graph
from agents.parser import create_parser_graph
from agents.summarizer import SummarizerAgent
from agents.qa_agent import QAAgent
from agents.hallucination_detector import HallucinationDetector
from agents.vectorstore_agent import VectorStoreAgent
from agents.state import PaperState
from agents.graph_input import GraphInput

# Configure Langfuse from settings
os.environ["LANGFUSE_PUBLIC_KEY"] = settings.LANGFUSE_PUBLIC_KEY
os.environ["LANGFUSE_SECRET_KEY"] = settings.LANGFUSE_SECRET_KEY
os.environ["LANGFUSE_HOST"] = settings.LANGFUSE_BASE_URL



class ResearchAssistant:
    def __init__(self):
        self.fetcher = create_fetcher_graph()
        self.parser = create_parser_graph()
        self.summarizer = SummarizerAgent()
        self.qa = QAAgent()
        self.detector = HallucinationDetector()
        self.vectorstore_agent = VectorStoreAgent()

        self.workflow = StateGraph(PaperState)

        self.workflow.add_node("fetch", self.fetcher)
        self.workflow.add_node("parse", self.parser)
        self.workflow.add_node("summarize", self.summarizer.generate_summary)
        self.workflow.add_node("store_vector", self.vectorstore_agent.store_in_vector_db)
        self.workflow.add_node("retrieve_context", self.qa.retrieve_context)
        self.workflow.add_node("generate_answer", self.qa.generate_answer)
        self.workflow.add_node("check_hallucination", self.detector.verify_citations)
        self.workflow.add_node("check_consistency", self.detector.cross_check_answer)
        self.workflow.add_node("comprehensive_check", self.detector.comprehensive_check)

        self.workflow.set_entry_point("fetch")
        self.workflow.add_edge("fetch", "parse")
        self.workflow.add_edge("parse", "summarize")
        self.workflow.add_edge("summarize", "store_vector")
        self.workflow.add_edge("store_vector", "retrieve_context")
        self.workflow.add_edge("retrieve_context", "generate_answer")
        self.workflow.add_edge("generate_answer", "check_hallucination")
        self.workflow.add_edge("check_hallucination", "check_consistency")
        self.workflow.add_edge("check_consistency", "comprehensive_check")
        self.workflow.add_edge("comprehensive_check", END)

        self.graph = self.workflow.compile()

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



if __name__ == "__main__":
    import argparse
    parser = argparse.ArgumentParser(description="Research Paper Assistant")
    parser.add_argument("--arxiv-id", required=True, help="arXiv ID of the paper")
    parser.add_argument("--question", help="Question about the paper")

    args = parser.parse_args()

    assistant = ResearchAssistant()
    
    import asyncio
    result = asyncio.run(assistant.run(args.arxiv_id, args.question))

    if result.get("error"):
        print(f"\n Error: {result['error']}")
        print(f"\n Stage: {result.get('metadata', {}).get('error_stage', 'unknown')}")
    else:
        print(f"\n Paper: {result.get('title', 'Unknown')}")
        print(f"\n Authors: {', '.join(result.get('authors', []))}")
        print(f"\n Question: {result.get('question')}")
        print(f"\n Answer : {result.get('answer', 'No answer generated')}")
        print(f"\n Citations : {result.get('citations', 'None')}")

        if "hallucination_check" in result:
            check = result["hallucination_check"]
            print(f"\n Hallucination Risk: {check.get('score', 0):.2f} ({check.get('status')})")
        
        if "consistency_check" in result:
            check = result["consistency_check"]
            print(f"\n Answer Consistency: {check.get('average_similarity', 0):.2f}")