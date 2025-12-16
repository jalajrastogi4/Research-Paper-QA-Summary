"""
SQLModel for research_papers table.

Stores metadata and cached data for research papers from arXiv.
"""

from datetime import datetime
from typing import Optional, List
from sqlmodel import SQLModel, Field, Column, JSON
from sqlalchemy import ARRAY, String, TIMESTAMP, CheckConstraint, Index


class ResearchPaper(SQLModel, table=True):
    """
    Research paper metadata and cached processing results.
    
    This table stores:
    - Paper metadata (title, authors, dates, abstract)
    - Cached summary (structured JSON)
    - Vector store status
    - Cache timestamps
    """
    
    __tablename__ = "research_papers"
    
    # Primary Key
    arxiv_id: str = Field(
        primary_key=True,
        max_length=50,
        description="arXiv paper ID (e.g., '1706.03762')"
    )
    
    # Paper Metadata
    title: str = Field(
        index=True,
        description="Paper title"
    )
    
    authors: List[str] = Field(
        sa_column=Column(ARRAY(String)),
        description="List of paper authors"
    )
    
    published_date: Optional[datetime] = Field(
        default=None,
        sa_column=Column(TIMESTAMP(timezone=True)),
        description="Original publication date from arXiv"
    )
    
    last_modified_date: datetime = Field(
        sa_column=Column(TIMESTAMP(timezone=True)),
        description="Last modified date from arXiv"
    )
    
    abstract: Optional[str] = Field(
        default=None,
        description="Paper abstract"
    )
    
    # Cached Data (stored as JSON to avoid large TEXT fields)
    summary_json: Optional[dict] = Field(
        default=None,
        sa_column=Column(JSON),
        description="Structured summary from summarizer agent"
    )
    
    # Metadata
    chunk_count: Optional[int] = Field(
        default=None,
        description="Number of chunks created for RAG"
    )
    
    vector_store_status: str = Field(
        default="pending",
        max_length=20,
        description="Status: 'pending', 'created', 'failed'"
    )
    
    # Timestamps
    cached_at: datetime = Field(
        default_factory=datetime.utcnow,
        sa_column=Column(TIMESTAMP(timezone=True)),
        description="When paper was first cached"
    )
    
    cache_updated_at: datetime = Field(
        default_factory=datetime.utcnow,
        sa_column=Column(TIMESTAMP(timezone=True)),
        description="When cache was last updated"
    )
    
    # Table constraints
    __table_args__ = (
        CheckConstraint(
            'last_modified_date <= cache_updated_at',
            name='check_dates'
        ),
        Index('idx_papers_modified', 'last_modified_date'),
        Index('idx_papers_cached', 'cached_at'),
        Index('idx_papers_status', 'vector_store_status'),
    )
    
    class Config:
        json_schema_extra = {
            "example": {
                "arxiv_id": "1706.03762",
                "title": "Attention Is All You Need",
                "authors": ["Vaswani, Ashish", "Shazeer, Noam"],
                "published_date": "2017-06-12T00:00:00Z",
                "last_modified_date": "2017-12-06T00:00:00Z",
                "abstract": "The dominant sequence transduction models...",
                "summary_json": {
                    "main_contribution": "Transformer architecture",
                    "key_findings": ["Self-attention mechanism", "Parallelization"]
                },
                "chunk_count": 45,
                "vector_store_status": "created"
            }
        }
