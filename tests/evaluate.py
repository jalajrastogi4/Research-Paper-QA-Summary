import json
from typing import Dict, List
from datasets import Dataset 
from ragas import evaluate
from ragas.metrics import faithfulness, answer_relevancy

class Evaluator:
    def __init__(self):
        self.benchmark = self.load_benchmark()
    
    def load_benchmark(self) -> List[Dict]:
        """Load test cases with ground truth"""
        return [
            {
                "arxiv_id": "1706.03762",
                "questions": [
                    {
                        "question": "What is the main contribution of the Transformer architecture?",
                        "expected_answer": "Introduces attention mechanism without recurrence or convolution",
                        "expected_citations": ["section 1", "abstract"]
                    },
                    {
                        "question": "What datasets were used for evaluation?",
                        "expected_answer": "WMT 2014 English-German and English-French",
                        "expected_citations": ["section 5"]
                    }
                ]
            }
        ]
    
    def evaluate_agent(self, assistant, test_cases=None):
        """Run evaluation on benchmark"""
        test_cases = test_cases or self.benchmark
        
        results = []
        for test in test_cases:
            for qa in test["questions"]:
                result = assistant.run(test["arxiv_id"], qa["question"])
                
                contexts = [c.get("content", "") for c in result.get("retrieved_chunks", [])]

                metrics = self.calculate_metrics(
                    result["answer"],
                    qa["expected_answer"],
                    result.get("hallucination_check", {}).get("score", 1)
                )
                
                results.append({
                    "arxiv_id": test["arxiv_id"],
                    "question": qa["question"],
                    "answer": result["answer"],
                    "expected": qa["expected_answer"],
                    "metrics": metrics,
                    "contexts": contexts,
                    "hallucination_score": result.get("hallucination_check", {}).get("score", 1),
                    "consistency": result.get("consistency_check", {}).get("average_similarity", 0)
                })
        
        avg_accuracy = sum(r["metrics"]["accuracy"] for r in results) / len(results)
        avg_hallucination = sum(r["hallucination_score"] for r in results) / len(results)
        
        questions = [r["question"] for r in results]
        answers = [r["answer"] for r in results]
        contexts = [r["contexts"] for r in results]
        ground_truths = [r["expected"] for r in results]

        data_samples = {
            "question": questions,
            "answer": answers,
            "contexts": contexts,
            "ground_truth": ground_truths
        }
        dataset = Dataset.from_dict(data_samples)
        ragas_results = evaluate(dataset, metrics=[faithfulness, answer_relevancy])
        
        return {
            "results": results,
            "ragas_results": ragas_results,
            "summary": {
                "average_accuracy": avg_accuracy,
                "average_hallucination_score": avg_hallucination,
                "total_tests": len(results)
            }
        }
    
    def calculate_metrics(self, answer, expected, hallucination_score):
        """Calculate evaluation metrics"""
        from difflib import SequenceMatcher
        
        similarity = SequenceMatcher(None, answer.lower(), expected.lower()).ratio()
        
        accuracy = similarity * (1 - hallucination_score)
        
        return {
            "similarity": similarity,
            "accuracy": accuracy,
            "hallucination_penalty": hallucination_score
        }