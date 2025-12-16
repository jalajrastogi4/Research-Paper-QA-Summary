from typing import Optional, Dict, Any
from pydantic import BaseModel, Field
from datetime import datetime
from enum import Enum

class JobTaskType(str, Enum):
    analyze_paper = "analyze_paper"
    evaluate_research_agent = "evaluate_research_agent"

class JobStatus(str, Enum):
    pending = "pending"
    processing = "processing"
    completed = "completed"
    failed = "failed"


class JobStatusRequest(BaseModel):
    job_id: str = Field(..., description="Unique job identifier")

class JobStatusResponse(BaseModel):
    """Response schema for job status."""
    
    job_id: str = Field(..., description="Unique job identifier")
    status: JobStatus = Field(..., description="Job status: pending, processing, completed, failed")
    task_name: Optional[JobTaskType] = Field(None, description="Task name (eg. analyze_paper, evaluate_research_agent)")
    arxiv_id: Optional[str] = Field(None, description="arXiv paper ID")
    result: Optional[Dict[str, Any]] = Field(None, description="Analysis result")
    error: Optional[str] = Field(None, description="Error message if failed")
    created_at: datetime = Field(..., description="Job creation time")
    completed_at: Optional[datetime] = Field(None, description="Completion time")
    
    class Config:
        json_schema_extra = {
            "example": {
                "job_id": "550e8400-e29b-41d4-a716-446655440000",
                "status": "completed",
                "task_name": "analyze_paper",
                "arxiv_id": "1706.03762",
                "result": {"answer": "...", "citations": "..."},
                "created_at": "2025-12-15T10:00:00",
                "completed_at": "2025-12-15T10:05:00",
            }
        }