import asyncio
import json
from datetime import datetime
from typing import Dict, Any, List
from agents.research_agent_evaluate import ResearchAssistantEvaluate
from tests.evaluate import AsyncEvaluator
from core.logging import get_logger

logger = get_logger()

class EvaluationService:
    def __init__(self) -> None:
        self.assistant = ResearchAssistantEvaluate()
        self.evaluator = AsyncEvaluator()

    async def run_evaluation(self, test_cases: List[Dict] = None) -> Dict[str, Any]:
        """Run evaluation on benchmark questions"""
        return await self.evaluator.evaluate_agent(self.assistant, test_cases)