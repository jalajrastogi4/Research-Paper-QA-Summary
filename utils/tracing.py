from langfuse.langchain import CallbackHandler
from core.config import settings

def get_langfuse_handler(trace_name: str, metadata: dict = None):
    """Get Langfuse handler for a specific trace/span"""
    return CallbackHandler(
        public_key=settings.LANGFUSE_PUBLIC_KEY,
        secret_key=settings.LANGFUSE_SECRET_KEY,
        host=settings.LANGFUSE_HOST,
        trace_name=trace_name,
        metadata=metadata or {}
    )