from langchain_core.prompts import ChatPromptTemplate

from core.logging import get_logger

logger = get_logger()

def summary_prompt():
    try:
        prompt = ChatPromptTemplate.from_template("""
            Summarize this research paper in structured format.
            
            Paper Title: {title}
            Authors: {authors}
            
            Paper Content:
            {content}
            
            {format_instructions}
        """)
        return prompt
    except Exception as e:
        logger.error(f"Failed to create summary prompt: {e}")
        raise RuntimeError(f"Failed to create summary prompt: {e}")


def qa_prompt():
    try:
        prompt = ChatPromptTemplate.from_template("""
        Answer the question based on the research paper context.
        You MUST cite specific sections or page numbers where possible.
        You MUST ALSO cite the specific chunks you used
        
        Question: {question}
        
        Context from paper:
        {context}
        
        Format your answer as JSON:
        {{
            "answer": "your answer",
            "citations": "section/page references or 'Not explicitly stated in paper'",
            "chunk_citations": [0, 2, 5]
        }}

        Where:
        - answer: Your answer to the question
        - citations: Human-readable references like "Section 3.1", "Abstract", or "Figure 2" and if no references are found then use 'Not explicitly stated in paper'
        - chunk_citations: List of integers representing chunk indices you used (e.g., [0, 2, 5])
        
        If the question cannot be answered from the context, say so and use empty list for chunk_citations.
        """)
        return prompt
    except Exception as e:
        logger.error(f"Failed to create QA prompt: {e}")
        raise RuntimeError(f"Failed to create QA prompt: {e}")


def verification_prompt():
    try:
        prompt = ChatPromptTemplate.from_template("""
        Verify if the following claim is supported by the provided context.
        
        Claim: {claim}
        
        Context from research paper:
        {context}
        
        Instructions:
        1. Check if the claim is DIRECTLY stated in the context
        2. Check if it's IMPLIED by the context
        3. Check if it CONTRADICTS the context
        4. Check if there's NO EVIDENCE in the context
        
        Respond ONLY with one word: SUPPORTED | IMPLIED | CONTRADICTED | NO_EVIDENCE
        """)
        return prompt
    except Exception as e:
        logger.error(f"Failed to create verification prompt: {e}")
        raise RuntimeError(f"Failed to create verification prompt: {e}")


def parsing_prompt():
    try:
        prompt = ChatPromptTemplate.from_template("""
            Analyze the first 5000 characters of a research paper and identify the STARTING phrases 
            for the following sections: Abstract, Introduction, Methodology, Results, Conclusion.
            
            Paper Text Start:
            {text_start}
            
            Return JSON format:
            {{
                "abstract_start": "text snippet...",
                "introduction_start": "text snippet...",
                "methodology_start": "text snippet...",
                "results_start": "text snippet...",
                "conclusion_start": "text snippet..."
            }}
            If a section is not found, return null.
        """)
        return prompt
    except Exception as e:
        logger.error(f"Failed to create parsing prompt: {e}")
        raise RuntimeError(f"Failed to create parsing prompt: {e}")


def claim_extraction_prompt():
    try:
        prompt = ChatPromptTemplate.from_template("""
            Extract 3-5 key factual claims from the following answer.
            Return as a JSON list of strings.
            
            Answer:
            {answer}
            
            Output format: ["claim 1", "claim 2", "claim 3"]
        """)
        return prompt
    except Exception as e:
        logger.error(f"Failed to create claim extraction prompt: {e}")
        raise RuntimeError(f"Failed to create claim extraction prompt: {e}")


def nli_verification_prompt():
    try:
        prompt = ChatPromptTemplate.from_template("""
            Verify if the claim is supported by the context.
            
            Context:
            {context}
            
            Claim:
            {claim}
            
            Return JSON:
            {{
                "verdict": "SUPPORTED" | "CONTRADICTED" | "NOT_MENTIONED",
                "explanation": "brief reason"
            }}
        """)
        return prompt
    except Exception as e:
        logger.error(f"Failed to create NLI prompt: {e}")
        raise RuntimeError(f"Failed to create NLI prompt: {e}")


def batch_nli_verification_prompt():
    try:
        prompt = ChatPromptTemplate.from_template("""
            Verify if the list of claims are supported by the context.
            
            Context:
            {context}
            
            Claims:
            {claims}
            
            For EACH claim listed above, verify it against the context.
            Return a JSON array with one object per claim.
            
            Return format:
            [
                {{"claim": "first claim text", "verdict": "SUPPORTED" | "CONTRADICTED" | "NOT_MENTIONED", "explanation": "brief reason"}},
                {{"claim": "second claim text", "verdict": "SUPPORTED" | "CONTRADICTED" | "NOT_MENTIONED", "explanation": "brief reason"}},
                {{"claim": "third claim text", "verdict": "SUPPORTED" | "CONTRADICTED" | "NOT_MENTIONED", "explanation": "brief reason"}}
            ]
            
            Note: The array length must match the number of claims provided.
        """)
        return prompt
    except Exception as e:
        logger.error(f"Failed to create batch NLI verification prompt: {e}")
        raise RuntimeError(f"Failed to create batch NLI verification prompt: {e}")