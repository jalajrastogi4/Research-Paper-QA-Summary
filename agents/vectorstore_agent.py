from agents.state import PaperState
from core.logging import get_logger

logger = get_logger()


class VectorStoreAgent:
    async def store_in_vector_db(self, state: PaperState) -> PaperState:
        """Store chunks in vector database and return store reference"""
        from utils.vector_store import VectorStoreManager
        
        vector_manager = VectorStoreManager()
        
        try:
            vector_store = vector_manager.create_vector_store(
                chunks=state["chunks"],
                arxiv_id=state["arxiv_id"]
            )
            logger.info(f"Vector store created for arxiv_id: {state['arxiv_id']}")
            return {
                "metadata": {
                    **state.get("metadata", {}),
                    "vector_store_created": True,
                    "chunk_count": len(state["chunks"])
                }
            }
        except Exception as e:
            logger.error(f"Failed to store in vector DB: {e}")
            raise RuntimeError(f"Failed to store in vector DB: {e}")