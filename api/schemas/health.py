from typing import Dict, Any, Optional
from pydantic import BaseModel, Field
from datetime import datetime


class ServiceHealth(BaseModel):
    """Health status for individual service."""
    status: str = Field(..., description="Service status: HEALTHY, DEGRADED, UNHEALTHY")
    last_check: datetime = Field(..., description="Last health check timestamp")
    error: Optional[str] = Field(None, description="Error message if unhealthy")

class HealthCheckResponse(BaseModel):
    """Overall system health check response."""
    status: str = Field(..., description="Overall status: HEALTHY, DEGRADED, UNHEALTHY")
    timestamp: str = Field(..., description="Health check timestamp (ISO format)")
    services: Dict[str, ServiceHealth] = Field(..., description="Individual service health statuses")
    
    class Config:
        json_schema_extra = {
            "example": {
                "status": "HEALTHY",
                "timestamp": "2025-12-15T10:00:00Z",
                "services": {
                    "database": {"status": "HEALTHY", "last_check": "2025-12-15T09:59:55Z"},
                    "redis": {"status": "HEALTHY", "last_check": "2025-12-15T09:59:56Z"},
                    "celery": {"status": "HEALTHY", "last_check": "2025-12-15T09:59:57Z"},
                    "pinecone": {"status": "HEALTHY", "last_check": "2025-12-15T09:59:58Z"}
                }
            }
        }