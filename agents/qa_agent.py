import re
from langchain_core.output_parsers import StrOutputParser

from agents.state import PaperState, create_error_state 
from utils.prompts import qa_prompt
from utils.llm import llm_model
from utils.vector_store import VectorStoreManager
from core.config import settings
from core.logging import get_logger

logger = get_logger()


class QAAgent:
    def __init__(self):
        self.llm = llm_model.get_llm()
        self.vector_store = VectorStoreManager()
        self.qa_prompt = qa_prompt()

    async def retrieve_context(self, state: PaperState) -> PaperState:
        """Retrieve relevant chunks using vector similarity"""
        question = state["question"]
        arxiv_id = state["arxiv_id"]

        try:
            relevant_chunks = self.vector_store.get_relevant_chunks(
                arxiv_id=arxiv_id,
                query=question,
                k=settings.RETRIEVAL_DOCS
            )

            avg_score = 0
            if relevant_chunks:
                avg_score = sum(c["relevance_score"] for c in relevant_chunks) / len(relevant_chunks)

            return {
                "retrieved_chunks": relevant_chunks,
                "metadata": {
                    **state.get("metadata", {}),
                    "chunks_retrieved": len(relevant_chunks),
                    "avg_relevance_score": avg_score,
                    "retrieval_method": "vector_similarity"
                }
            }
        except Exception as e:
            logger.error(f"Failed to retrieve context: {e}")
            return create_error_state(f"Retrieve failed: {str(e)}", "retrieve", retrieved_chunks=[])

    async def generate_answer(self, state: PaperState) -> PaperState:
        """Generate answer with citations"""
        retrieved_chunks = state.get("retrieved_chunks", [])

        if not retrieved_chunks:
            return {
                "answer": "Could not find relevant information in the paper.",
                "citations": "No relevant sections found",
                "metadata": {
                    **state.get("metadata", {}),
                    "answer_generated": False,
                    "error": "No relevant chunks retrieved"
                }
            }

        try:
            context_parts = []
            for chunk in retrieved_chunks:
                content = chunk.get("content", "")
                metadata = chunk.get("metadata", {})
                chunk_idx = metadata.get("chunk_index", "unknown")

                context_parts.append(f"[Chunk {chunk_idx}]: {content}")

            context = "\n\n".join(context_parts)
            
            chain = self.qa_prompt | self.llm | StrOutputParser()
            response = await chain.ainvoke({
                "question": state["question"],
                "context": context
            })
            
            answer_match = re.search(r"Answer:\s*(.*?)(?=\n*Citations:|$)", response, re.DOTALL)
            citation_match = re.search(r"Citations:\s*(.*)", response)
            
            answer = answer_match.group(1).strip() if answer_match else response
            citations = citation_match.group(1).strip() if citation_match else "Not provided"

            used_chunk_indices = [c.get("metadata", {}).get("chunk_index") for c in retrieved_chunks]
            
            return {
                "answer": answer,
                "citations": citations,
                "retrieved_chunks": retrieved_chunks,
                "metadata": {
                    **state.get("metadata", {}),
                    "answer_generated": True,
                    "chunks_used": used_chunk_indices,
                    "chunk_relevance_scores": [c.get("relevance_score", 0) for c in retrieved_chunks]
                }
            }
        except Exception as e:
            logger.error(f"Failed to generate answer: {e}")
            return create_error_state(f"Generate failed: {str(e)}", "generate", answer="", citations="", retrieved_chunks=retrieved_chunks)
