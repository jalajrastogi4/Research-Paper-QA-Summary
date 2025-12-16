import json
from typing import Dict, List
from datasets import Dataset 
from ragas import evaluate
from ragas.llms import LangchainLLMWrapper
from ragas.embeddings import LangchainEmbeddingsWrapper
from ragas.metrics import faithfulness, answer_relevancy

from utils.llm import llm_model
from agents.research_agent_evaluate import ResearchAssistantEvaluate

class AsyncEvaluator:
    """Evaluator for RAGAS metrics"""
    
    def __init__(self):
        self.benchmark = self.load_benchmark()
        self.ragas_llm = LangchainLLMWrapper(llm_model.get_llm())
        self.ragas_embedding = LangchainEmbeddingsWrapper(llm_model.get_embeddings())    
    
    def load_benchmark(self) -> List[Dict]:
        """Load test cases with ground truth"""
        return [
            {
                "arxiv_id": "1706.03762",  # Attention Is All You Need
                "questions": [
                    {
                        "question": "What is the main contribution of the Transformer architecture?",
                        "expected_answer": "Introduces attention mechanism without recurrence or convolution",
                        "expected_citations": ["section 1", "abstract"]
                    },
                    {
                        "question": "What datasets were used for evaluation?",
                        "expected_answer": "WMT 2014 English-German and English-French translation tasks",
                        "expected_citations": ["section 5", "section 6"]
                    }
                ]
            }
        ]
    
    async def evaluate_agent(self, assistant: ResearchAssistantEvaluate, test_cases=None):
        """Run evaluation on benchmark questions"""
        test_cases = test_cases or self.benchmark
        
        results = []
        questions = []
        answers = []
        contexts = []
        ground_truths = []
        
        print(f"\nEvaluating {sum(len(t['questions']) for t in test_cases)} questions...")
        
        for test_idx, test in enumerate(test_cases, 1):
            arxiv_id = test["arxiv_id"]
            print(f"\n[Paper {test_idx}] ArXiv ID: {arxiv_id}")
            
            for q_idx, qa in enumerate(test["questions"], 1):
                question = qa["question"]
                print(f"  [Q{q_idx}] {question[:60]}...")
                
                try:
                    result = await assistant.run(arxiv_id, question)
                    
                    if result.get("error"):
                        print(f"     Error: {result['error']}")
                        continue
                    
                    answer = result.get("answer", "")
                    retrieved_chunks = result.get("retrieved_chunks", [])
                    context_list = [c.get("content", "") for c in retrieved_chunks]
                    
                    questions.append(question)
                    answers.append(answer)
                    contexts.append(context_list)
                    ground_truths.append(qa["expected_answer"])
                    
                    results.append({
                        "arxiv_id": arxiv_id,
                        "question": question,
                        "answer": answer,
                        "expected_answer": qa["expected_answer"],
                        "retrieved_chunks_count": len(retrieved_chunks),
                        "hallucination_score": result.get("comprehensive_hallucination_check", {}).get("overall_score", 0),
                        "hallucination_risk": result.get("comprehensive_hallucination_check", {}).get("overall_risk", "UNKNOWN"),
                        "avg_relevance_score": result.get("metadata", {}).get("avg_relevance_score", 0)
                    })
                    
                    print(f"    Completed")
                    
                except Exception as e:
                    print(f"    Error: {e}")
                    import traceback
                    traceback.print_exc()
        
        if not results:
            print("\n No successful evaluations. Cannot compute RAGAS metrics.")
            return None
        
        print(f"\n{'='*80}")
        print("Computing RAGAS Metrics...")
        print(f"{'='*80}")
        
        data_samples = {
            "question": questions,
            "answer": answers,
            "contexts": contexts,
            "ground_truth": ground_truths
        }
        
        dataset = Dataset.from_dict(data_samples)
        
        try:
            ragas_results = evaluate(dataset, metrics=[faithfulness, answer_relevancy],
                                     llm=self.ragas_llm, embeddings=self.ragas_embedding)
            ragas_dict = ragas_results.to_pandas().to_dict('records')[0]
        except Exception as e:
            print(f"RAGAS evaluation failed: {e}")
            import traceback
            traceback.print_exc()
            ragas_dict = {"faithfulness": 0, "answer_relevancy": 0}
        
        avg_hallucination = sum(r["hallucination_score"] for r in results) / len(results)
        avg_relevance = sum(r["avg_relevance_score"] for r in results) / len(results)
        
        return {
            "ragas_metrics": ragas_dict,
            "summary": {
                "total_questions": len(results),
                "avg_hallucination_score": avg_hallucination,
                "avg_retrieval_relevance": avg_relevance,
                "successful_evaluations": len(results)
            },
            "detailed_results": results
        }