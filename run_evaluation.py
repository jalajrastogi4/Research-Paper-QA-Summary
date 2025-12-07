"""
RAGAS Evaluation Script for LangGraph Multi-Agent System

This script runs comprehensive evaluation using RAGAS metrics:
- Faithfulness: Measures factual consistency with retrieved context
- Answer Relevancy: Measures how relevant the answer is to the question
"""

import asyncio
import json
from datetime import datetime

from main import ResearchAssistant
from tests.evaluate import AsyncEvaluator


async def main():
    """Main evaluation function"""
    print("="*80)
    print("RAGAS Evaluation - LangGraph Research Paper Summarizer and Q&A")
    print("="*80)
    
    print("\n[1/3] Initializing Research Assistant...")
    assistant = ResearchAssistant()
    
    print("\n[2/3] Initializing Evaluator...")
    evaluator = AsyncEvaluator()
    
    print("\n[3/3] Running Evaluation...")
    
    start_time = datetime.now()
    
    results = await evaluator.evaluate_agent(assistant)
    
    end_time = datetime.now()
    duration = (end_time - start_time).total_seconds()
    
    if not results:
        print("\n Evaluation failed. No results to display.")
        return
    
    print("\n" + "="*80)
    print("RAGAS Evaluation Results")
    print("="*80)
    
    ragas = results["ragas_metrics"]
    summary = results["summary"]
    
    print(f"\n RAGAS Metrics:")
    print(f"  • Faithfulness:      {ragas.get('faithfulness', 0):.2%}")
    print(f"  • Answer Relevancy:  {ragas.get('answer_relevancy', 0):.2%}")
    
    print(f"\n System Metrics:")
    print(f"  • Questions Evaluated:     {summary['total_questions']}")
    print(f"  • Avg Hallucination Risk:  {summary['avg_hallucination_score']:.2%}")
    print(f"  • Avg Retrieval Relevance: {summary['avg_retrieval_relevance']:.2%}")
    
    print(f"\n Performance:")
    print(f"  • Total Duration:          {duration:.1f}s")
    print(f"  • Avg Time per Question:   {duration/summary['total_questions']:.1f}s")
    
    print(f"\n Detailed Results:")
    for i, result in enumerate(results["detailed_results"], 1):
        print(f"\n  [{i}] {result['question'][:60]}...")
        print(f"      Hallucination Risk: {result['hallucination_score']:.2%} ({result['hallucination_risk']})")
        print(f"      Retrieval Relevance: {result['avg_relevance_score']:.2%}")
        print(f"      Retrieved Chunks: {result['retrieved_chunks_count']}")
    
    # Save results
    output_file = f"evaluation_results_{datetime.now().strftime('%Y%m%d_%H%M%S')}.json"
    with open(output_file, 'w') as f:
        json.dump({
            "ragas_metrics": ragas,
            "summary_metrics": summary,
            "detailed_results": results["detailed_results"],
            "evaluation_metadata": {
                "duration_seconds": duration,
                "timestamp": datetime.now().isoformat()
            }
        }, f, indent=2)
    
    print(f"\n Results saved to: {output_file}")
    
    print("\n" + "="*80)
    print("Evaluation Complete!")
    print("="*80)
    
    return results


if __name__ == "__main__":
    asyncio.run(main())
