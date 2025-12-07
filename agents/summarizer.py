from langchain_core.output_parsers import PydanticOutputParser
from pydantic import BaseModel, Field
from typing import List

from agents.state import PaperState, create_error_state 
from utils.prompts import summary_prompt
from utils.llm import llm_model
from core.logging import get_logger

logger = get_logger()


class PaperSummary(BaseModel):
    key_contributions: List[str] = Field(description="3-5 key contributions")
    methodology: str = Field(description="Brief methodology description")
    results: str = Field(description="Main results/findings")
    limitations: List[str] = Field(description="2-3 limitations mentioned")
    future_work: List[str] = Field(description="Suggested future work")


class SummarizerAgent:
    def __init__(self):
        self.llm = llm_model.get_llm()
        self.parser = PydanticOutputParser(pydantic_object=PaperSummary)
        self.prompt = summary_prompt()

    async def generate_summary(self, state: PaperState) -> PaperState:
        """Generate structured summary using LLM"""

        chain = self.prompt | self.llm | self.parser

        content = state.get("sections", {})
        if not content:
            # Fallback to raw_text if sections not available
            raw_text = state.get("raw_text", "")
            if not raw_text:
                logger.error("No content available for summarization (missing sections and raw_text)")
                return create_error_state(
                    "Summarization failed: No content available", 
                    "summarize", 
                    summary="", 
                    summary_generated=False
                )
            content = raw_text[:5000]

        try:
            summary = await chain.ainvoke({
                "title": state["title"],
                "authors": ", ".join(state["authors"]),
                "content": str(content),
                "format_instructions": self.parser.get_format_instructions()
            })
            logger.info(f"Generated summary: {summary}")
            return {
                "summary": summary.model_dump_json(),
                "metadata": {
                    **state.get("metadata", {}),
                    "summary_generated": True
                }
            }

        except Exception as e:
            logger.error(f"Failed to generate summary: {e}")
            return create_error_state(f"Generate failed: {str(e)}", "generate", summary="", summary_generated=False)
            