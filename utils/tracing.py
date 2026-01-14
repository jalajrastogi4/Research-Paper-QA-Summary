from langfuse.langchain import CallbackHandler
from core.config import settings

def get_langfuse_handler():
    """Get Langfuse handler for a specific trace/span"""
    return CallbackHandler()