import json
import re
from langgraph.graph import StateGraph, END
from langchain_core.output_parsers import JsonOutputParser, StrOutputParser
from typing import Dict

from agents.state import PaperState, create_error_state 
from utils.llm import llm_model
from utils.prompts import parsing_prompt
from core.config import settings
from core.logging import get_logger

logger = get_logger()


class ParserAgent:
    def __init__(self):
        self.name = "parser_agent"
        self.llm = llm_model.get_llm()
        self.parsing_prompt = parsing_prompt()


    async def _identify_sections_with_llm(self, text: str) -> Dict[str, str]:
        """Identify section start strings using LLM"""
        chain = self.parsing_prompt | self.llm | JsonOutputParser()
        try:
            response = await chain.ainvoke({"text_start": text})
            logger.info(f"Sections identified with LLM: {response}")
            return response
        except Exception as e:
            logger.info(f"JsonOutputParser failed: {e}. Retrying with StrOutputParser + manual json.loads")
            fallback_chain = self.parsing_prompt | self.llm | StrOutputParser()
            try:
                raw = await fallback_chain.ainvoke({"text_start": text})
                logger.info(f"Sections identified with StrOutputParser: {raw}")
                return json.loads(raw.strip())
            except Exception as inner_e:
                logger.error(f"Failed to identify sections with LLM: {inner_e}")
                raise RuntimeError(f"Failed to identify sections with LLM: {inner_e}")


    async def parse_sections(self, state: PaperState) -> PaperState:
        """Extract structured sections from raw text"""
        try:
            raw_text = state["raw_text"]
            text_start = raw_text[:settings.SECTION_PARSER_LIMIT]

            response = await self._identify_sections_with_llm(text_start)

            section_names = ["abstract", "introduction", "methodology", "results", "conclusion"]

            extracted_sections = {}

            for name in section_names:
                start_phrase = response.get(f"{name}_start")
                if not start_phrase:
                    extracted_sections[name] = ""
                    continue
                
                idx = raw_text.find(start_phrase)
                if idx == -1:
                    logger.warning(f"Start phrase for {name} not found in raw_text")
                    extracted_sections[name] = ""
                    continue
                
                next_indices = [
                    raw_text.find(response.get(f"{other}_start")) 
                    for other in section_names 
                    if other != name and response.get(f"{other}_start")
                ]
                next_indices = [i for i in next_indices if i != -1 and i > idx]
                end_idx = min(next_indices) if next_indices else len(raw_text)

                extracted_sections[name] = raw_text[idx:end_idx].strip()

            return {"sections": extracted_sections}
        except Exception as e:
            logger.error(f"Failed to parse sections: {e}")
            return create_error_state(f"Parse failed: {str(e)}", "parse", sections={})


    async def chunk_content(self, state: PaperState) -> PaperState:
        """Chunk the paper for vector storage"""
        from utils.chunker import chunk_text

        try:
            text_to_chunk = state["raw_text"]
            chunks = chunk_text(text_to_chunk)
            logger.info(f"Text chunked into {len(chunks)} chunks")
            return {
                "chunks": chunks,
                "metadata": {
                    **state.get("metadata", {}),
                    "chunk_count": len(chunks),
                    "avg_chunk_length": sum(len(c) for c in chunks) / len(chunks) if chunks else 0
                }
            }
        except Exception as e:
            logger.error(f"Failed to chunk content: {e}")
            return create_error_state(f"Chunk failed: {str(e)}", "chunk", chunks=[])



def create_parser_graph():
    parser = ParserAgent()
    workflow = StateGraph(PaperState)

    workflow.add_node("parse_sections", parser.parse_sections)
    workflow.add_node("chunk_content", parser.chunk_content)

    workflow.set_entry_point("parse_sections")
    workflow.add_edge("parse_sections", "chunk_content")
    workflow.add_edge("chunk_content", END)

    logger.info("Parser graph created")
    return workflow.compile()