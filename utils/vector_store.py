"""
Vector store manager using Pinecone for embeddings storage.

Migrated from ChromaDB to Pinecone for:
- Better scalability
- No persistent disk requirements
- Managed infrastructure
"""

from typing import List, Dict, Any
from pinecone import Pinecone, ServerlessSpec

from utils.llm import llm_model
from core.config import settings
from core.logging import get_logger

logger = get_logger()


class VectorStoreManager:
    """
    Manages vector embeddings using Pinecone.
    
    Each paper gets its own namespace within a single Pinecone index.
    """
    
    def __init__(self):
        """Initialize Pinecone client and index."""
        self.pc = Pinecone(api_key=settings.PINECONE_API_KEY)
        self.index_name = settings.PINECONE_INDEX_NAME
        self.embeddings = llm_model.get_embeddings()
        
        # Create index if it doesn't exist
        self._ensure_index_exists()
        
        # Connect to the index
        self.index = self.pc.Index(self.index_name)
        logger.info(f"Connected to Pinecone index: {self.index_name}")
    
    def _ensure_index_exists(self):
        """Create Pinecone index if it doesn't exist."""
        try:
            existing_indexes = [idx.name for idx in self.pc.list_indexes()]
            
            if self.index_name not in existing_indexes:
                logger.info(f"Creating Pinecone index: {self.index_name}")
                self.pc.create_index(
                    name=self.index_name,
                    dimension=1536,  # OpenAI text-embedding-3-small dimension
                    metric="cosine",
                    spec=ServerlessSpec(
                        cloud="aws",
                        region=settings.PINECONE_ENVIRONMENT
                    )
                )
                logger.info(f"Pinecone index created: {self.index_name}")
            else:
                logger.info(f"Pinecone index already exists: {self.index_name}")
        except Exception as e:
            logger.error(f"Error ensuring Pinecone index exists: {e}")
            raise
    
    def create_vector_store(self, chunks: List[str], arxiv_id: str) -> int:
        """
        Create vector embeddings for paper chunks and store in Pinecone.
        
        Args:
            chunks: List of text chunks from the paper
            arxiv_id: arXiv paper ID (used as namespace)
        
        Returns:
            Number of vectors upserted
        """
        try:
            # Generate embeddings for all chunks
            logger.info(f"Generating embeddings for {len(chunks)} chunks...")
            embeddings = self.embeddings.embed_documents(chunks)
            
            # Prepare vectors for upsert
            vectors = []
            for i, (chunk, embedding) in enumerate(zip(chunks, embeddings)):
                vector_id = f"{arxiv_id}_{i}"
                
                # Pinecone metadata has a size limit (~40KB), truncate to 10,000 chars to be safe
                # This allows full chunk storage even with overlap
                truncated_text = chunk[:10000] if len(chunk) > 10000 else chunk
                
                vectors.append({
                    "id": vector_id,
                    "values": embedding,
                    "metadata": {
                        "arxiv_id": arxiv_id,
                        "chunk_index": i,
                        "text": truncated_text,
                        "source": f"paper_{arxiv_id}",
                        "full_text_length": len(chunk)
                    }
                })
            
            # Upsert vectors in batches of 100 (Pinecone recommendation)
            batch_size = 100
            total_upserted = 0
            
            for i in range(0, len(vectors), batch_size):
                batch = vectors[i:i + batch_size]
                self.index.upsert(
                    vectors=batch,
                    namespace=arxiv_id  # Each paper gets its own namespace
                )
                total_upserted += len(batch)
                logger.info(f"Upserted batch {i//batch_size + 1}: {len(batch)} vectors")
            
            logger.info(f"Successfully upserted {total_upserted} vectors for {arxiv_id}")
            return total_upserted
            
        except Exception as e:
            logger.error(f"Failed to create vector store for {arxiv_id}: {e}")
            raise RuntimeError(f"Failed to create vector store: {e}")
    
    def get_relevant_chunks(
        self,
        arxiv_id: str,
        query: str,
        k: int = 3
    ) -> List[Dict[str, Any]]:
        """
        Retrieve top-k relevant chunks for a query.
        
        Args:
            arxiv_id: arXiv paper ID (namespace)
            query: Query text
            k: Number of results to return
        
        Returns:
            List of dicts with content, metadata, and relevance_score
        """
        try:
            # Generate query embedding
            query_embedding = self.embeddings.embed_query(query)
            
            # Query Pinecone
            results = self.index.query(
                vector=query_embedding,
                top_k=k,
                namespace=arxiv_id,
                include_metadata=True
            )
            
            # Format results
            relevant_chunks = []
            for match in results.matches:
                relevant_chunks.append({
                    "content": match.metadata.get("text", ""),
                    "metadata": {
                        "arxiv_id": match.metadata.get("arxiv_id"),
                        "chunk_index": match.metadata.get("chunk_index"),
                        "source": match.metadata.get("source"),
                        "vector_id": match.id
                    },
                    "relevance_score": float(match.score)
                })
            
            logger.info(f"Retrieved {len(relevant_chunks)} chunks for query in {arxiv_id}")
            return relevant_chunks
            
        except Exception as e:
            logger.error(f"Failed to retrieve chunks for {arxiv_id}: {e}")
            return []
    
    def delete_paper_vectors(self, arxiv_id: str) -> bool:
        """
        Delete all vectors for a specific paper.
        
        Args:
            arxiv_id: arXiv paper ID (namespace to delete)
        
        Returns:
            True if successful, False otherwise
        """
        try:
            self.index.delete(delete_all=True, namespace=arxiv_id)
            logger.info(f"Deleted all vectors for {arxiv_id}")
            return True
        except Exception as e:
            logger.error(f"Failed to delete vectors for {arxiv_id}: {e}")
            return False
    
    def get_index_stats(self) -> Dict[str, Any]:
        """Get statistics about the Pinecone index."""
        try:
            stats = self.index.describe_index_stats()
            return {
                "total_vector_count": stats.total_vector_count,
                "dimension": stats.dimension,
                "index_fullness": stats.index_fullness,
                "namespaces": stats.namespaces
            }
        except Exception as e:
            logger.error(f"Failed to get index stats: {e}")
            return {}
