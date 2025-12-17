import pytest
from utils.vector_store import VectorStoreManager

@pytest.mark.asyncio
async def test_pinecone_operations():
    """Test Pinecone-specific operations"""
    manager = VectorStoreManager()

    stats = manager.get_index_stats()
    assert stats is not None, "Index stats are None"

    test_chunks = [
        "This is a test chunk about machine learning.",
        "Another test chunk about neural networks.",
        "Final test chunk about transformers."
    ]
    test_arxiv_id = "test-paper-001"

    vector_count = manager.create_vector_store(test_chunks, test_arxiv_id)
    assert vector_count is not None, "Vector count is None"
    assert vector_count > 0, "Vector count is less than 0"

    query = "What is machine learning?"
    chunks = manager.get_relevant_chunks(test_arxiv_id, query, k=2)
    assert chunks is not None, "Chunks are None"
    assert len(chunks) > 0, "Chunks are empty"
    for i, chunk in enumerate(chunks, 1):
        assert chunk is not None, "Chunk is None"
        assert chunk.get("content") is not None, "Content is None"
        assert chunk.get("metadata") is not None, "Metadata is None"
        assert chunk.get("relevance_score") is not None, "Relevance score is None"


    # Test 4: Delete test vectors (cleanup)
    success = manager.delete_paper_vectors(test_arxiv_id)
    assert success is not None, "Success is None"
    assert success, "Failed to delete vectors"