import re
from typing import List, Dict, Any
from difflib import SequenceMatcher
from langchain_core.output_parsers import StrOutputParser, JsonOutputParser
from sentence_transformers import SentenceTransformer, util
from langfuse import get_client

from agents.qa_agent import PaperState
from utils.llm import llm_model
from utils.prompts import (
    verification_prompt, 
    claim_extraction_prompt, 
    nli_verification_prompt, 
    batch_nli_verification_prompt
    )
from utils.vector_store import VectorStoreManager
from utils.tracing import get_langfuse_handler
from core.config import settings
from core.logging import get_logger

logger = get_logger()


class HallucinationDetector:
    def __init__(self):
        self.llm = llm_model.get_llm()
        self.verification_prompt = verification_prompt()
        self.claim_prompt = claim_extraction_prompt()
        self.nli_prompt = nli_verification_prompt()
        self.batch_nli_prompt = batch_nli_verification_prompt()
        self.similarity_model = SentenceTransformer('all-MiniLM-L6-v2')

    def _semantic_cross_check(self, ans1: str, ans2: str) -> float:
        emb1 = self.similarity_model.encode(ans1, convert_to_tensor=True)
        emb2 = self.similarity_model.encode(ans2, convert_to_tensor=True)
        
        return util.pytorch_cos_sim(emb1, emb2).item()

    def _categorize_answer(self,ans: str) -> str:
        if not ans or len(ans) < 2:
            return "empty"
        
        if ans.lower() in ["yes", "no", "unknown", "not specified", "n/a"]:
            return "valid_short"
        
        if any(word in ans.lower() for word in ["error", "failed", "could not"]):
            return "error"
        
        if len(ans) < 10:
            return "too_short"
        
        return "valid_long"

    def _get_score_safe(self, layer_result, path, default=0.5):
        try:
            current = layer_result
            for key in path:
                current = current[key]
            return current
        except (KeyError, TypeError):
            logger.warning(f"Failed to extract score from {path}, using default {default}")
            return default

    async def verify_citations(self, state: PaperState) -> PaperState:
        """Citation verification"""
        citations = state.get("citations", "")
        text = state["raw_text"]
        chunk_citations = state.get("chunk_citations", [])
        retrieved_chunks = state.get("retrieved_chunks", [])

        valid_indices = {c.get("metadata", {}).get("chunk_index", -999) for c in retrieved_chunks}

        verification_results = []

        for idx in chunk_citations:
            is_valid = idx in valid_indices and idx != -999
            verification_results.append({
                "reference": f"Chunk {idx}",
                "found": is_valid,
                "confidence": "high" if is_valid else "invalid"
            })

        # section_patterns = r'(?:section|sec\.?|§)\s*(\d+(?:\.\d+)*)'
        # section_matches = re.finditer(section_patterns, citations, re.IGNORECASE)

        # page_patterns = r'(?:p\.|pp\.|page)\s*(\d+(?:\s*[–-]\s*\d+)?)'
        # page_matches = re.finditer(page_patterns, citations, re.IGNORECASE)

        # figure_pattern = r'(?:fig\.|figure|table|tbl\.)\s*(\d+(?:\.\d+)*)'
        # figure_matches = re.finditer(figure_pattern, citations, re.IGNORECASE)

        # all_matches = list(section_matches) + list(page_matches) + list(figure_matches)

        # for curr_match in all_matches:
        #     ref = curr_match.group(1)
        #     ref_type = curr_match.group(0).split('.')[0].lower() if '.' in curr_match.group(0) else curr_match.group(0).lower()

        #     found = False
        #     confidence = 'low'

        #     if ref and re.search(rf'\b{re.escape(ref)}\b', text, re.IGNORECASE):
        #         found = True
        #         confidence = "medium"

        #     for chunk in retrieved_chunks:
        #         chunk_content = chunk.get("content", "").lower()
        #         if ref and ref.lower() in chunk_content:
        #             found = True
        #             confidence = "high"
        #             break

        #     verification_results.append({
        #         "reference": f"{ref_type} {ref}",
        #         "found": found,
        #         "confidence": confidence,
        #         "type": ref_type
        #     })

        if not chunk_citations:
            # hallucination_score = 0.5
            # citation_present = False
            answer_lower = state.get("answer", "").lower()
            citations_lower = citations.lower()

            uncertainty_markers = [
                "doesn't specify", "does not specify", "not specified",
                "not mentioned", "doesn't mention", "does not mention", 
                "not stated", "not explicitly stated", "unclear",
                "not provided", "doesn't provide", "not discussed",
                "no mention", "doesn't say", "does not say",
                "unclear from the paper", "not clear from"
            ]

            is_uncertain = any(marker in answer_lower or marker in citations_lower 
                      for marker in uncertainty_markers)

            if is_uncertain:
                hallucination_score = 0.0
                citation_present = False
            else:
                holistic_markers = ["throughout", "overall", "entire paper", 
                           "whole paper", "multiple sections"]
                is_holistic = any(marker in citations_lower for marker in holistic_markers)

                if is_holistic:
                    hallucination_score = 0.4
                    citation_present = False
                else:
                    hallucination_score = 0.7
                    citation_present = False
        else:
            unfound_count = sum(1 for r in verification_results if not r["found"])
            hallucination_score = unfound_count / len(chunk_citations)
            citation_present = True

        # hallucination_score = 0
        # if verification_results:
        #     unfound_count = sum(1 for r in verification_results if not r["found"])
        #     hallucination_score = unfound_count / len(verification_results)

        # citation_present = bool(citations and citations.lower() not in ["not provided", "none", "no citations", "no relevant sections", "not explicitly stated in paper"])
        # if not citation_present and state.get("answer", ""):
        #     hallucination_score = max(hallucination_score, 0.7)

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
        claim_handler = get_langfuse_handler()

        claims_chain = self.claim_prompt | self.llm | JsonOutputParser()
        nli_chain = self.nli_prompt | self.llm | JsonOutputParser()
        verifications = []

        try:
            # claims = await claims_chain.ainvoke({"answer": answer}, config={"callbacks": [claim_handler]})
            claims = await claims_chain.ainvoke(
                {"answer": answer},
                config = {
                    "callbacks": [claim_handler],
                    "tags": ["claim_extraction"],
                    "metadata": {"arxiv_id": state["arxiv_id"]}
                }
            )
            logger.info(f"Extracted claims: {claims}")

            nli_handler = get_langfuse_handler()
            batch_nli_chain = self.batch_nli_prompt | self.llm | JsonOutputParser()
            claims_text = "\n".join([f"{i+1}. {claim}" for i, claim in enumerate(claims)])
            batch_results = await batch_nli_chain.ainvoke(
                {"context": context, "claims": claims_text},
                config={
                        "callbacks": [nli_handler],
                        "tags": ["batch_nli_verification"],
                        "metadata": {
                            "claim_count": len(claims), 
                            "arxiv_id": state["arxiv_id"]
                        }
                    }
                )
            for result in batch_results:
                verifications.append({
                    "claim": result["claim"],
                    "verdict": result["verdict"],
                    "explanation": result["explanation"]
                })
            # for claim in claims:
            #     nli_result = await nli_chain.ainvoke({"claim": claim, "context": context})
            #     logger.info(f"NLI result: {nli_result}")
            #     verifications.append({
            #         "claim": claim,
            #         "verdict": nli_result["verdict"],
            #         "explanation": nli_result["explanation"]
            #     })
            
            supported_count = sum(1 for v in verifications if v["verdict"] == "SUPPORTED")
            llm_hallucination_score = 1 - (supported_count / len(verifications)) if verifications else 0.5
            logger.info(f"LLM Hallucination Score: {llm_hallucination_score}")

            return {
                "llm_verification": {
                    "verifications": verifications,
                    "supported_claims": supported_count,
                    "total_claims": len(verifications),
                    "hallucination_score": llm_hallucination_score,
                    "status": "verification task performed"
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
            "retrieved_chunks": retrieved_chunks
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
        categories = [self._categorize_answer(a) for a in answers]
        

        if categories.count("error") >= 2 or categories.count("too_short") >= 2:
            return {
                "consistency_check": {
                    "average_similarity": 0,
                    "status": "generation_failed",
                    "valid_answers": 0,
                    "error_count": categories.count("error"),
                    "too_short_count": categories.count("too_short")
                }
            }

        if categories.count("valid_short") >= 2 and categories.count("error") == 0:
            valid_short_answers = [a for a, cat in zip(answers, categories) if cat == "valid_short"]
            if all(a.lower() == valid_short_answers[0].lower() for a in valid_short_answers):
                return {
                    "consistency_check": {
                        "average_similarity": 1.0,
                        "status": "fully_consistent_short",
                        "valid_answers": len(valid_short_answers),
                        "answer_type": "short"
                    }
                }
            else:
                similarities = []
                for i in range(len(valid_short_answers)):
                    for j in range(i + 1, len(valid_short_answers)):
                        sim = self._semantic_cross_check(valid_short_answers[i], valid_short_answers[j])
                        similarities.append(sim)
                
                avg_similarity = sum(similarities) / len(similarities) if similarities else 0
                return {
                    "consistency_check": {
                        "average_similarity": avg_similarity,
                        "status": "short_answers_compared",
                        "valid_answers": len(valid_short_answers)
                    }
                }

        valid_answers = [a for a, cat in zip(answers, categories) if cat in ["valid_short", "valid_long"]]
        
        if len(valid_answers) < 2:
            return {
                "consistency_check": {
                    "average_similarity": 0,
                    "status": "insufficient_valid_data",
                    "valid_answers": len(valid_answers),
                    "error_count": categories.count("error"),
                    "too_short_count": categories.count("too_short")
                }
            }
        
        similarities = []
        for i in range(len(valid_answers)):
            for j in range(i + 1, len(valid_answers)):
                sim = self._semantic_cross_check(valid_answers[i], valid_answers[j])
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
                "variation_2_length": len(var2_answer),
                "answer_categories": categories
            },
            "metadata": {
                **state.get("metadata", {}),
                "answer_consistency": avg_similarity
            }
        }

    
    async def comprehensive_check(self, state: PaperState) -> PaperState:
        """Run all hallucination checks and combine results"""
        # citation_check = await self.verify_citations(state)
        # llm_check = await self.verify_claims_with_nli(state)
        # consistency_check = await self.cross_check_answer(state)
        citation_check = state.get("hallucination_check", {})
        llm_check = state.get("llm_verification", {})
        consistency_check = state.get("consistency_check", {})

        # citation_score = citation_check["hallucination_check"]["score"]
        citation_score = self._get_score_safe(citation_check, ["score"], default=0.5)
        # llm_score = llm_check.get("llm_verification", {}).get("hallucination_score", 0.5)
        llm_score = self._get_score_safe(llm_check, ["hallucination_score"], default=0.5)
        
        failed_layers = []
        consistency_result = consistency_check
        consistency_status = consistency_result.get("status", "unknown")

        if consistency_status in ["skipped", "error", "insufficient_valid_data", "generation_failed"]:
            consistency_score = 0.5
            failed_layers.append("consistency")
        else:
            avg_sim = consistency_result.get("average_similarity", 0.5)
            consistency_score = 1 - avg_sim

        llm_status = llm_check.get("status", "unknown")
        if llm_status in ["skipped", "failed"]:
            failed_layers.append("llm_verification")
        
        # if citation_score == 0.5 and citation_check.get("hallucination_check", {}).get("status") == "error":
        #     failed_layers.append("citation")
        
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

        try:

            langfuse = get_client()
            if overall_risk in ["HIGH", "CRITICAL"]:
                langfuse.update_current_trace(tags=["HIGH_RISK", "REVIEW_REQUIRED"])
            elif overall_risk == "MEDIUM":
                langfuse.update_current_trace(tags=["MEDIUM_RISK"])
            else:
                langfuse.update_current_trace(tags=["LOW_RISK"])

            langfuse.score_current_trace(
                name="hallucination_score",
                value=final_score
            )
            logger.info(f"Langfuse trace updated successfully")
        except Exception as e:
            logger.error(f"Failed to update Langfuse trace: {e}")

        return {
            "comprehensive_hallucination_check": {
                "overall_score": final_score,
                "overall_risk": overall_risk,
                "component_scores": {
                    "citation_verification": citation_score,
                    "llm_verification": llm_score,
                    "consistency_check": consistency_score
                },
                "citation_details": citation_check,
                "llm_details": llm_check,
                "consistency_details": consistency_check,
                "failed_layers": failed_layers
            },
            "metadata": {
                **state.get("metadata", {}),
                # **citation_check.get("metadata", {}),
                # **llm_check.get("metadata", {}),
                # **consistency_check.get("metadata", {}),
                "final_hallucination_score": final_score,
                "hallucination_risk": overall_risk
            }
        }