import pytest
from agents.research_agent_evaluate import ResearchAssistantEvaluate


@pytest.mark.asyncio
async def test_full_pipeline():
    """Test complete pipeline with Pinecone"""
    assistant = ResearchAssistantEvaluate()

    arxiv_id = "1706.03762"
    question = "What is the main contribution of the Transformer architecture?"

    result = await assistant.run(arxiv_id, question)
    assert result is not None, "Result is None"
    assert result.get("error") is None, f"Error: {result['error']}"
    assert result.get("title") is not None, "Title is None"
    assert result.get("answer") is not None, "Answer is None"
    assert result.get("citations") is not None, "Citations are None"

    metadata = result.get('metadata', {})
    assert metadata is not None, "Metadata is None"
    assert metadata.get("vector_count") is not None, "Vector count is None"
    assert metadata.get("chunks_retrieved") is not None, "Chunks retrieved is None"
    assert metadata.get("avg_relevance_score") is not None, "Avg relevance score is None"

    comprehensive_hallucination_check = result.get("comprehensive_hallucination_check", {})
    assert comprehensive_hallucination_check is not None, "Comprehensive hallucination check is None"
    assert comprehensive_hallucination_check.get("overall_score") is not None, "Overall score is None"
    assert comprehensive_hallucination_check.get("overall_risk") is not None, "Overall risk is None"