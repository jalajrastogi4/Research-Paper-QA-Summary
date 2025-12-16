"""
Demo script to run LangGraph system and capture results for README
"""
import asyncio
import json
from datetime import datetime
from agents.research_agent_evaluate import ResearchAssistantEvaluate

async def run_demo():
    """Run demo on sample paper"""
    print("=" * 80)
    print("LangGraph Multi-Agent Research Paper Analysis - Demo")
    print("=" * 80)
    
    # Initialize assistant
    print("\n[1/3] Initializing Research Assistant...")
    assistant = ResearchAssistantEvaluate()
    
    # Test paper - using well-known Transformer paper
    test_paper = {
        "arxiv_id": "1706.03762",  # Attention Is All You Need
        "question": "What is the main contribution of the Transformer architecture?"
    }
    
    print(f"\n[2/3] Processing Paper: {test_paper['arxiv_id']}")
    print(f"Question: {test_paper['question']}")
    print("-" * 80)
    
    try:
        result = await assistant.run(
            arxiv_id=test_paper['arxiv_id'],
            question=test_paper['question']
        )
        
        # Check for errors
        if result.get("error"):
            print(f"\n Error: {result['error']}")
            print(f"   Stage: {result.get('metadata', {}).get('error_stage', 'unknown')}")
            return None
        
        # Display results
        print(f"\n Paper: {result.get('title', 'Unknown')}")
        print(f" Authors: {', '.join(result.get('authors', [])[:3])}...")
        print(f"\n Question: {result.get('question')}")
        print(f"\n Answer: {result.get('answer', 'No answer')[:500]}...")
        print(f"\n Citations: {result.get('citations', 'None')}")
        
        if "hallucination_check" in result:
            check = result["hallucination_check"]
            print(f"\n Hallucination Risk: {check.get('score', 0):.2%} ({check.get('status')})")
        
        if "consistency_check" in result:
            check = result["consistency_check"]
            print(f" Answer Consistency: {check.get('average_similarity', 0):.2%})")
        
        # Save result
        print(f"\n[3/3] Saving results...")
        demo_result = {
            "arxiv_id": test_paper['arxiv_id'],
            "title": result.get('title'),
            "question": test_paper['question'],
            "answer": result.get('answer'),
            "citations": result.get('citations'),
            "hallucination_score": result.get('hallucination_check', {}).get('score', 0),
            "consistency": result.get('consistency_check', {}).get('average_similarity', 0),
            "metadata": result.get('metadata', {})
        }
        
        output_file = f"demo_results_{datetime.now().strftime('%Y%m%d_%H%M%S')}.json"
        with open(output_file, 'w') as f:
            json.dump(demo_result, f, indent=2)
        print(f" Results saved to: {output_file}")
        
        print("\n" + "=" * 80)
        print("Demo Complete!")
        print("=" * 80)
        print(f"\n Summary:")
        print(f"  - Paper processed: {test_paper['arxiv_id']}")
        print(f"  - Hallucination score: {demo_result['hallucination_score']:.2%}")
        print(f"  - Consistency score: {demo_result['consistency']:.2%}")
        print(f"\n Check Langfuse dashboard: https://cloud.langfuse.com")
        
        return demo_result
        
    except Exception as e:
        print(f"\n Error processing {test_paper['arxiv_id']}: {e}")
        import traceback
        traceback.print_exc()
        return None

if __name__ == "__main__":
    asyncio.run(run_demo())
