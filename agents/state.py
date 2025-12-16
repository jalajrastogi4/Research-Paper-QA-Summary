from typing import TypedDict, List, Optional, Any, Dict
from datetime import datetime

class PaperState(TypedDict):
    arxiv_id: str
    question: str

    raw_text: Optional[str]
    title: Optional[str]
    authors: Optional[List[str]]
    chunks: Optional[List[str]]
    sections: Optional[Dict[str, str]]
    summary: Optional[Dict[str, Any]]  # Changed from str to Dict for structured data
    abstract: Optional[str]

    paper_cached: Optional[bool]
    paper_current: Optional[bool]
    last_modified_date: Optional[datetime]

    answer: Optional[str]
    citations: Optional[str]
    retrieved_chunks: Optional[List[Dict[str, Any]]]

    hallucination_check: Optional[Dict[str, Any]]
    llm_verification: Optional[Dict[str, Any]]
    consistency_check: Optional[Dict[str, Any]]
    comprehensive_hallucination_check: Optional[Dict[str, Any]]

    error: Optional[str]
    
    metadata: Dict[str, Any]


def create_error_state(error_msg: str, stage: str, **kwargs) -> dict:
    """Helper to create consistent error state"""
    return {
        "error": error_msg,
        "metadata": {
            "status": "failed",
            "error_stage": stage,
            **kwargs
        }
    }    