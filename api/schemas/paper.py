from typing import Optional, List, Dict, Any
from pydantic import BaseModel, Field
from datetime import datetime


class AnalyzeRequest(BaseModel):
    """Request schema for paper analysis."""
    
    arxiv_id: str = Field(
        ...,
        description="arXiv paper ID (e.g., '1706.03762')",
        example="1706.03762"
    )
    question: str = Field(
        ...,
        description="Question to answer about the paper",
        example="What is the main contribution of this paper?"
    )
    use_cache: bool = Field(
        default=True,
        description="Whether to use caching for faster responses"
    )
    
    class Config:
        json_schema_extra = {
            "example": {
                "arxiv_id": "1706.03762",
                "question": "What is the Transformer architecture?",
                "use_cache": True
            }
        }


class RetrievedChunk(BaseModel):
    """Schema for retrieved document chunks."""
    content: str = Field(..., description="Chunk text content")
    metadata: Dict[str, Any] = Field(..., description="Chunk metadata (arxiv_id, chunk_index, etc.)")
    relevance_score: float = Field(..., description="Similarity/relevance score (0-1)")


class AnalyzeResponse(BaseModel):
    """Response schema for paper analysis."""
    
    job_id: Optional[str] = Field(None, description="Job ID for async requests")
    arxiv_id: str = Field(..., description="arXiv paper ID")
    title: Optional[str] = Field(None, description="Paper title")
    authors: Optional[List[str]] = Field(None, description="Paper authors")
    question: str = Field(..., description="Question asked")
    answer: str = Field(..., description="Generated answer")
    citations: str = Field(..., description="Citations and references")
    
    # Metrics
    hallucination_score: float = Field(..., description="Hallucination risk score")
    hallucination_risk: str = Field(..., description="Risk level")
    answer_relevancy_score: Optional[float] = Field(None, description="Avg retrieval relevance")
    
    # Retrieved chunks
    retrieved_chunks: Optional[List[RetrievedChunk]] = Field(None, description="Retrieved document chunks")
    
    # Cache info
    cached: bool = Field(default=False, description="Whether result was cached")
    cache_timestamp: Optional[datetime] = Field(None, description="Cache timestamp if cached")
    
    # Metadata
    metadata: Optional[Dict[str, Any]] = Field(None, description="Additional metadata")
    error: Optional[str] = Field(None, description="Error message if evaluation failed")
    
    class Config:
        json_schema_extra = {
            "example": {
                "job_id": "550e8400-e29b-41d4-a716-446655440000",
                "arxiv_id": "1706.03762",
                "title": "Attention Is All You Need",
                "authors": ["Vaswani et al."],
                "question": "What is the Transformer?",
                "answer": "The Transformer is a novel architecture...",
                "citations": "[Section 3.1], [Figure 1]",
                "hallucination_score": 0.15,
                "hallucination_risk": "LOW",
                "answer_relevancy_score": 0.85,
                "cached": False
            }
        }


class PaperStatusRequest(BaseModel):
    arxiv_id: str = Field(..., description="arXiv paper ID")


class PaperStatusResponse(BaseModel):
    arxiv_id: str = Field(..., description="arXiv paper ID")
    title: Optional[str] = Field(None, description="Paper title")
    cached: bool = Field(default=False, description="Whether result was cached")
    vector_store_status: Optional[str] = Field(None, description="Vector store status")
    cache_updated_at: Optional[datetime] = Field(None, description="Cache timestamp if cached")
    chunk_count: Optional[int] = Field(None, description="Number of chunks")


class EvaluationRequest(BaseModel):
    test_cases: Optional[List[Dict]] = Field(None, description="Test cases")
    class Config:
        json_schema_extra = {
            "example": {
                "test_cases": [
                    {
                        "arxiv_id": "1706.03762",
                        "questions": [
                            {
                                "question": "What is the main contribution of the Transformer architecture?",
                                "expected_answer": "Introduces attention mechanism without recurrence or convolution",
                                "expected_citations": ["section 1", "abstract"]
                            },
                            {
                                "question": "What datasets were used for evaluation?",
                                "expected_answer": "WMT 2014 English-German and English-French translation tasks",
                                "expected_citations": ["section 5", "section 6"]
                            }
                        ]
                    }
                ]
            }
        }


class EvaluationResponse(BaseModel):
    job_id: Optional[str] = Field(None, description="Job ID for async requests")
    results: Optional[List[Dict]] = Field(None, description="Evaluation results")
    ragas_metrics: Optional[Dict[str, Any]] = Field(None, description="RAGAS metrics")
    summary: Optional[Dict[str, Any]] = Field(None, description="Summary of evaluation results")
    error: Optional[str] = Field(None, description="Error message if evaluation failed")