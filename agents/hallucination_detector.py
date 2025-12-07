import re
from typing import List, Dict, Any
from difflib import SequenceMatcher
from langchain_core.output_parsers import StrOutputParser, JsonOutputParser

from agents.qa_agent import PaperState
from utils.llm import llm_model
from utils.prompts import verification_prompt, claim_extraction_prompt, nli_verification_prompt
from utils.vector_store import VectorStoreManager
from core.config import settings
from core.logging import get_logger

logger = get_logger()


class HallucinationDetector:
    def __init__(self):
        self.llm = llm_model.get_llm()
        self.verification_prompt = verification_prompt()
        self.claim_prompt = claim_extraction_prompt()
        self.nli_prompt = nli_verification_prompt()

    async def verify_citations(self, state: PaperState) -> PaperState:
        """Citation verification"""
        citations = state.get("citations", "")
        text = state["raw_text"]
        retrieved_chunks = state.get("retrieved_chunks", [])

        verification_results = []

        section_patterns = r'(?:section|sec\.?|§)\s*(\d+(?:\.\d+)*)'
        section_matches = re.finditer(section_patterns, citations, re.IGNORECASE)

        page_patterns = r'(?:p\.|pp\.|page)\s*(\d+(?:\s*[–-]\s*\d+)?)'
        page_matches = re.finditer(page_patterns, citations, re.IGNORECASE)

        figure_pattern = r'(?:fig\.|figure|table|tbl\.)\s*(\d+(?:\.\d+)*)'
        figure_matches = re.finditer(figure_pattern, citations, re.IGNORECASE)

        all_matches = list(section_matches) + list(page_matches) + list(figure_matches)

        for curr_match in all_matches:
            ref = curr_match.group(1)
            ref_type = curr_match.group(0).split('.')[0].lower() if '.' in curr_match.group(0) else curr_match.group(0).lower()

            found = False
            confidence = 'low'

            if ref and re.search(rf'\b{re.escape(ref)}\b', text, re.IGNORECASE):
                found = True
                confidence = "medium"

            for chunk in retrieved_chunks:
                chunk_content = chunk.get("content", "").lower()
                if ref and ref.lower() in chunk_content:
                    found = True
                    confidence = "high"
                    break

            verification_results.append({
                "reference": f"{ref_type} {ref}",
                "found": found,
                "confidence": confidence,
                "type": ref_type
            })

        hallucination_score = 0
        if verification_results:
            unfound_count = sum(1 for r in verification_results if not r["found"])
            hallucination_score = unfound_count / len(verification_results)

        citation_present = bool(citations and citations.lower() not in ["not provided", "none", "no citations", "no relevant sections", "not explicitly stated in paper"])
        if not citation_present and state.get("answer", ""):
            hallucination_score = max(hallucination_score, 0.7)

        return {
            "hallucination_check": {
                "score": hallucination_score,
                "verified_citations": verification_results,
                "citation_present": citation_present,
                "status": "high_risk" if hallucination_score > 0.5 else "medium_risk" if hallucination_score > 0.2 else "low_risk",
                "metadata": {
                    **state.get("metadata", {}),
                    "hallucination_score": hallucination_score,
                    "citations_verified": len(verification_results)
                }
            }
        }

    async def verify_claims_with_nli(self, state: PaperState) -> PaperState:
        """Using NLI to verify claims"""
        answer = state.get("answer", "")
        retrieved_chunks = state.get("retrieved_chunks", [])
        if not answer or not retrieved_chunks:
            return {
                "llm_verification": {
                    "status": "skipped",
                    "reason": "Missing answer or context"
                }
            }

        context = "\n".join([c.get("content", "") for c in retrieved_chunks])

        claims_chain = self.claim_prompt | self.llm | JsonOutputParser()
        nli_chain = self.nli_prompt | self.llm | JsonOutputParser()
        verifications = []

        try:
            claims = await claims_chain.ainvoke({"answer": answer})
            logger.info(f"Extracted claims: {claims}")
            for claim in claims:
                nli_result = await nli_chain.ainvoke({"claim": claim, "context": context})
                logger.info(f"NLI result: {nli_result}")
                verifications.append({
                    "claim": claim,
                    "verdict": nli_result["verdict"],
                    "explanation": nli_result["explanation"]
                })
            
            supported_count = sum(1 for v in verifications if v["verdict"] == "SUPPORTED")
            llm_hallucination_score = 1 - (supported_count / len(verifications)) if verifications else 0.5
            logger.info(f"LLM Hallucination Score: {llm_hallucination_score}")

            return {
                "llm_verification": {
                    "verifications": verifications,
                    "supported_claims": supported_count,
                    "total_claims": len(verifications),
                    "hallucination_score": llm_hallucination_score
                },
                "metadata": {
                    **state.get("metadata", {}),
                    "llm_verification_score": llm_hallucination_score
                }
            }

        except Exception as e:
            logger.error(f"Failed to extract claims or claim verification: {e}")
            return {"llm_verification": {"status": "failed", "reason": str(e)}}


    async def cross_check_answer(self, state: PaperState) -> PaperState:
        """Generate answer variations"""
        from agents.qa_agent import QAAgent

        original_answer = state.get("answer", "")
        retrieved_chunks = state.get("retrieved_chunks", [])

        if not original_answer or not retrieved_chunks:
            return {
                "consistency_check": {
                    "average_similarity": 0,
                    "status": "skipped",
                    "reason": "Missing answer or context"
                }
            }

        qa_agent1 = QAAgent()
        qa_agent2 = QAAgent()

        qa_agent1.llm = llm_model.get_llm(temperature=0.3)

        variation_state = {
            "question": state["question"],
            "retrieved_chunks": retrieved_chunks[:1]
        }

        try:
            var1_result = await qa_agent1.generate_answer(state)
            var2_result = await qa_agent2.generate_answer(variation_state)

            var1_answer = var1_result.get("answer", "")
            var2_answer = var2_result.get("answer", "")
        except Exception as e:
            return {
                "consistency_check": {
                    "average_similarity": 0,
                    "status": "error",
                    "error": str(e)
                }
            }

        answers = [original_answer, var1_answer, var2_answer]
        valid_answers = [a for a in answers if a and len(a) > 10]

        if len(valid_answers) < 2:
            return {
                "consistency_check": {
                    "average_similarity": 0,
                    "status": "insufficient_data",
                    "valid_answers": len(valid_answers)
                }
            }

        similarities = []
        for i in range(len(valid_answers)):
            for j in range(i + 1, len(valid_answers)):
                sim = SequenceMatcher(None, valid_answers[i].lower(), valid_answers[j].lower()).ratio()
                similarities.append(sim)

        avg_similarity = sum(similarities) / len(similarities) if similarities else 0

        if avg_similarity > 0.8:
            status = "highly_consistent"
        elif avg_similarity > 0.6:
            status = "moderately_consistent"
        elif avg_similarity > 0.4:
            status = "low_consistency"
        else:
            status = "inconsistent"

        return {
            "consistency_check": {
                "average_similarity": avg_similarity,
                "status": status,
                "variation_count": len(valid_answers),
                "pairwise_similarities": similarities,
                "original_answer_length": len(original_answer),
                "variation_1_length": len(var1_answer),
                "variation_2_length": len(var2_answer)
            },
            "metadata": {
                **state.get("metadata", {}),
                "answer_consistency": avg_similarity
            }
        }

    
    async def comprehensive_check(self, state: PaperState) -> PaperState:
        """Run all hallucination checks and combine results"""
        citation_check = await self.verify_citations(state)
        llm_check = await self.verify_claims_with_nli(state)
        consistency_check = await self.cross_check_answer(state)

        citation_score = citation_check["hallucination_check"]["score"]
        llm_score = llm_check.get("llm_verification", {}).get("hallucination_score", 0.5)
        consistency_score = 1 - consistency_check["consistency_check"]["average_similarity"]

        weights = {"citation": settings.CITATION_SCORE, "llm": settings.LLM_SCORE, "consistency": settings.CONSISTENCY_SCORE}
        final_score = (citation_score * weights["citation"] + llm_score * weights["llm"] + consistency_score * weights["consistency"])

        if final_score > 0.7:
            overall_risk = "CRITICAL"
        elif final_score > 0.5:
            overall_risk = "HIGH"
        elif final_score > 0.3:
            overall_risk = "MEDIUM"
        else:
            overall_risk = "LOW"

        return {
            "comprehensive_hallucination_check": {
                "overall_score": final_score,
                "overall_risk": overall_risk,
                "component_scores": {
                    "citation_verification": citation_score,
                    "llm_verification": llm_score,
                    "consistency_check": consistency_score
                },
                "citation_details": citation_check["hallucination_check"],
                "llm_details": llm_check.get("llm_verification", {}),
                "consistency_details": consistency_check["consistency_check"]
            },
            "metadata": {
                **state.get("metadata", {}),
                **citation_check.get("metadata", {}),
                **llm_check.get("metadata", {}),
                **consistency_check.get("metadata", {}),
                "final_hallucination_score": final_score,
                "hallucination_risk": overall_risk
            }
        }