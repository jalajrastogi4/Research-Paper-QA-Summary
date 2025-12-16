"""
SQLModel for paper_qa_cache table.

Stores cached Q&A results with hallucination detection metrics.
"""

from datetime import datetime
from typing import Optional
from sqlmodel import SQLModel, Field, Column, JSON, UniqueConstraint, Index
from sqlalchemy import TIMESTAMP, ForeignKey


class PaperQACache(SQLModel, table=True):
    """
    Cached Q&A results with hallucination metrics.
    
    This table stores:
    - Question-answer pairs for papers
    - Hallucination detection scores
    - RAGAS evaluation metrics
    - Access tracking for cache management
    """
    
    __tablename__ = "paper_qa_cache"
    
    # Primary Key
    id: Optional[int] = Field(
        default=None,
        primary_key=True,
        description="Auto-incrementing primary key"
    )
    
    # Foreign Key to research_papers
    arxiv_id: str = Field(
        foreign_key="research_papers.arxiv_id",
        max_length=50,
        index=True,
        description="Reference to research paper"
    )
    
    # Question Hashing for Cache Lookup
    question_hash: str = Field(
        max_length=64,
        description="MD5/SHA256 hash of normalized question for exact match"
    )
    
    question: str = Field(
        description="Original question text"
    )
    
    # Answer and Citations
    answer: str = Field(
        description="Generated answer from QA agent"
    )
    
    citations: Optional[str] = Field(
        default=None,
        description="Citation references (e.g., '[Chunk 5, Chunk 12]')"
    )
    
    # Hallucination Detection Metrics
    hallucination_score: Optional[float] = Field(
        default=None,
        description="Overall hallucination score (0-100, lower is better)"
    )
    
    hallucination_risk: Optional[str] = Field(
        default=None,
        max_length=20,
        description="Risk level: 'LOW', 'MEDIUM', 'HIGH'"
    )
    
    # RAGAS Evaluation Metrics
    faithfulness_score: Optional[float] = Field(
        default=None,
        description="RAGAS faithfulness score (0-1, higher is better)"
    )
    
    answer_relevancy_score: Optional[float] = Field(
        default=None,
        description="RAGAS answer relevancy score (0-1, higher is better)"
    )
    
    # Response Metadata (store additional metrics as JSON)
    response_metadata: Optional[dict] = Field(
        default=None,
        sa_column=Column(JSON),
        description="Additional metadata (retrieval scores, NLI results, etc.)"
    )
    
    # Cache Management
    created_at: datetime = Field(
        default_factory=datetime.utcnow,
        sa_column=Column(TIMESTAMP(timezone=True)),
        description="When this Q&A was first cached"
    )
    
    last_accessed_at: datetime = Field(
        default_factory=datetime.utcnow,
        sa_column=Column(TIMESTAMP(timezone=True)),
        description="Last time this cache entry was accessed"
    )
    
    access_count: int = Field(
        default=1,
        description="Number of times this cache entry has been accessed"
    )
    
    # Table constraints
    __table_args__ = (
        UniqueConstraint('arxiv_id', 'question_hash', name='uq_arxiv_question'),
        Index('idx_qa_access', 'access_count', 'last_accessed_at'),
        Index('idx_qa_arxiv', 'arxiv_id'),
        Index('idx_qa_created', 'created_at'),
    )
    
    class Config:
        json_schema_extra = {
            "example": {
                "id": 1,
                "arxiv_id": "1706.03762",
                "question_hash": "5d41402abc4b2a76b9719d911017c592",
                "question": "What is the main contribution of the Transformer?",
                "answer": "The main contribution is the self-attention mechanism...",
                "citations": "[Chunk 5, Chunk 12]",
                "hallucination_score": 15.56,
                "hallucination_risk": "LOW",
                "faithfulness_score": 0.92,
                "answer_relevancy_score": 0.88,
                "response_metadata": {
                    "citation_verification": 0.0,
                    "nli_verification": 0.0,
                    "consistency_check": 77.79
                },
                "access_count": 5,
                "created_at": "2025-12-10T10:00:00Z",
                "last_accessed_at": "2025-12-10T14:00:00Z"
            }
        }
