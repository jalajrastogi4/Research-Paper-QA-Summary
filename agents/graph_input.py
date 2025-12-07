from pydantic import BaseModel, Field, field_validator
from typing import Optional, Dict
import re

class GraphInput(BaseModel):
    arxiv_id: str = Field(..., description="ArXiv identifier of the paper")
    question: Optional[str] = Field(
        None, description="User's research question about the paper"
    )
    metadata: Dict[str, str] = Field(
        default_factory=dict, description="Metadata about the run"
    )

    @field_validator('arxiv_id')
    def validate_arxiv_id(cls, v):
        # ArXiv IDs: YYMM.NNNNN or archive/YYMMNNN
        pattern = r'^(\d{4}\.\d{4,5}|[a-z\-]+/\d{7})$'
        if not re.match(pattern, v):
            raise ValueError(
                f"Invalid arXiv ID format: {v}. "
                "Expected format: YYMM.NNNNN (e.g., 1706.03762)"
            )
        return v
