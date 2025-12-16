"""
SQLModel for job_status table.

Tracks async Celery job execution status and results.
"""

from datetime import datetime
from typing import Optional
from sqlmodel import SQLModel, Field, Column, JSON
from sqlalchemy import TIMESTAMP
import uuid


class JobStatus(SQLModel, table=True):
    """
    Async job status tracking for Celery tasks.
    
    This table stores:
    - Job execution status (pending, processing, completed, failed)
    - Job results or error messages
    - Timestamps for monitoring
    """
    
    __tablename__ = "job_status"
    
    # Primary Key (UUID for distributed systems)
    job_id: str = Field(
        primary_key=True,
        max_length=36,
        description="UUID for the job"
    )
    
    # Job Metadata
    arxiv_id: Optional[str] = Field(
        default=None,
        max_length=50,
        index=True,
        description="arXiv ID being processed (if applicable)"
    )
    
    task_name: Optional[str] = Field(
        default=None,
        max_length=100,
        description="Celery task name (e.g., 'process_paper_analysis')"
    )
    
    # Job Status
    status: str = Field(
        default="pending",
        max_length=20,
        index=True,
        description="Status: 'pending', 'processing', 'completed', 'failed'"
    )
    
    # Results and Errors
    result: Optional[dict] = Field(
        default=None,
        sa_column=Column(JSON),
        description="Job result data (stored as JSON)"
    )
    
    error: Optional[str] = Field(
        default=None,
        description="Error message if job failed"
    )
    
    # Progress Tracking (optional)
    progress: Optional[int] = Field(
        default=0,
        description="Progress percentage (0-100)"
    )
    
    # Timestamps
    created_at: datetime = Field(
        default_factory=datetime.utcnow,
        sa_column=Column(TIMESTAMP(timezone=True)),
        description="When job was created"
    )
    
    updated_at: datetime = Field(
        default_factory=datetime.utcnow,
        sa_column=Column(TIMESTAMP(timezone=True)),
        description="Last update timestamp"
    )
    
    started_at: Optional[datetime] = Field(
        default=None,
        sa_column=Column(TIMESTAMP(timezone=True)),
        description="When job started processing"
    )
    
    completed_at: Optional[datetime] = Field(
        default=None,
        sa_column=Column(TIMESTAMP(timezone=True)),
        description="When job completed (success or failure)"
    )
    
    class Config:
        json_schema_extra = {
            "example": {
                "job_id": "550e8400-e29b-41d4-a716-446655440000",
                "arxiv_id": "1706.03762",
                "task_name": "process_paper_analysis",
                "status": "completed",
                "result": {
                    "answer": "The main contribution is...",
                    "hallucination_score": 15.56,
                    "citations": "[Chunk 5]"
                },
                "error": None,
                "progress": 100,
                "created_at": "2025-12-10T10:00:00Z",
                "updated_at": "2025-12-10T10:05:00Z",
                "started_at": "2025-12-10T10:00:30Z",
                "completed_at": "2025-12-10T10:05:00Z"
            }
        }
    
    def is_terminal(self) -> bool:
        """Check if job is in a terminal state (completed or failed)."""
        return self.status in ("completed", "failed")
    
    def is_running(self) -> bool:
        """Check if job is currently running."""
        return self.status == "processing"
    
    def is_pending(self) -> bool:
        """Check if job is pending."""
        return self.status == "pending"
