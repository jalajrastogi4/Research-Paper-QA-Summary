from agents.research_agent_evaluate import ResearchAssistantEvaluate


if __name__ == "__main__":
    import argparse
    parser = argparse.ArgumentParser(description="Research Paper Assistant")
    parser.add_argument("--arxiv-id", required=True, help="arXiv ID of the paper")
    parser.add_argument("--question", help="Question about the paper")

    args = parser.parse_args()

    assistant = ResearchAssistantEvaluate()
    
    import asyncio
    result = asyncio.run(assistant.run(args.arxiv_id, args.question))

    if result.get("error"):
        print(f"\n Error: {result['error']}")
        print(f"\n Stage: {result.get('metadata', {}).get('error_stage', 'unknown')}")
    else:
        print(f"\n Paper: {result.get('title', 'Unknown')}")
        print(f"\n Authors: {', '.join(result.get('authors', []))}")
        print(f"\n Question: {result.get('question')}")
        print(f"\n Answer : {result.get('answer', 'No answer generated')}")
        print(f"\n Citations : {result.get('citations', 'None')}")

        if "hallucination_check" in result:
            check = result["hallucination_check"]
            print(f"\n Hallucination Risk: {check.get('score', 0):.2f} ({check.get('status')})")
        
        if "consistency_check" in result:
            check = result["consistency_check"]
            print(f"\n Answer Consistency: {check.get('average_similarity', 0):.2f}")