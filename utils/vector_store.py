from langchain_chroma import Chroma
from langchain_core.documents import Document
from typing import List, Optional
import chromadb

from utils.llm import llm_model
from core.config import settings
from core.logging import get_logger

logger = get_logger()


class VectorStoreManager:
    def __init__(self, persist_dir: str = settings.CHROMADB_DIR):
        self.persist_dir = persist_dir
        self.embeddings = llm_model.get_embeddings()
        self.vector_stores = {}

    def create_vector_store(self, chunks: List[str], arxiv_id: str) -> Chroma:
        """Create and persist a vector store for this paper"""
        documents = [
            Document(
                page_content=chunk,
                metadata={
                    "arxiv_id": arxiv_id,
                    "chunk_index": i,
                    "source": f"paper_{arxiv_id}"
                }
            )
            for i, chunk in enumerate(chunks)
        ]
        try:
            vector_store = Chroma.from_documents(
                documents=documents,
                embedding=self.embeddings,
                collection_name=f"paper_{arxiv_id}",
                persist_directory=self.persist_dir
            )

            logger.info(f"Vector store created for arxiv_id: {arxiv_id}")
            
            self.vector_stores[arxiv_id] = vector_store
            return vector_store
        except Exception as e:
            logger.error(f"Failed to create vector store: {e}")
            raise RuntimeError(f"Failed to create vector store: {e}")

    def get_relevant_chunks(self, arxiv_id: str, query: str, k: int = 3) -> List[Document]:
        """Retrieve top-k relevant chunks for a query"""
        if arxiv_id not in self.vector_stores:
            try:
                vector_store = Chroma(
                    collection_name=f"paper_{arxiv_id}",
                    persist_directory=self.persist_dir,
                    embedding_function=self.embeddings,
                )
                self.vector_stores[arxiv_id] = vector_store
            except Exception as e:
                logger.error(f"Failed to load vector store for {arxiv_id}: {e}")
                return []

        vector_store = self.vector_stores[arxiv_id]

        results = vector_store.similarity_search_with_score(
            query=query,
            k=k
        )
        
        relevant_chunks = []
        for doc, score in results:
            relevant_chunks.append({
                "content": doc.page_content,
                "metadata": doc.metadata,
                "relevance_score": float(score)
            })

        return relevant_chunks
