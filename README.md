# LangGraph Multi-Agent Research Paper Analysis System

A production-ready multi-agent system built with LangGraph for analyzing research papers from arXiv, featuring advanced hallucination detection and comprehensive observability.

## Key Features

- **Multi-Agent Architecture**: Specialized agents for fetching, parsing, summarizing, Q&A, and verification
- **Advanced Hallucination Detection**: Multi-layered verification combining:
  - Citation verification
  - NLI-based claim checking
  - Consistency analysis across answer variations
- **Production Observability**: Full Langfuse integration for tracing and monitoring
- **Async Architecture**: Built with async/await for optimal performance
- **RAGAS Evaluation**: Automated evaluation with faithfulness and answer relevancy metrics
- **Vector-based RAG**: ChromaDB for efficient semantic search and retrieval

## Architecture

```mermaid
graph TD
    A[User Query] --> B[Fetch Agent]
    B --> C[Parser Agent]
    C --> D[Summarizer Agent]
    D --> E[Vector Store Agent]
    E --> F[QA Agent]
    F --> G[Hallucination Detector]
    G --> H[Final Answer]
    
    G --> G1[Citation Verification]
    G --> G2[NLI Claim Verification]
    G --> G3[Consistency Check]
    
    style G fill:#f9f,stroke:#333,stroke-width:4px
    style H fill:#9f9,stroke:#333,stroke-width:2px
```

### Agent Responsibilities

| Agent | Purpose | Key Features |
|-------|---------|--------------|
| **Fetcher** | Downloads papers from arXiv | SSL bypass, PDF extraction |
| **Parser** | Extracts structured sections | LLM-based section identification |
| **Summarizer** | Generates structured summaries | Pydantic output parsing |
| **Vector Store** | Creates embeddings database | ChromaDB persistence |
| **QA Agent** | Answers questions with citations | RAG with top-k retrieval |
| **Hallucination Detector** | Verifies answer accuracy | Multi-layered verification |

## Quick Start

### Prerequisites

```bash
Python 3.10+
OpenAI API key
Langfuse account (optional, for observability)
```

### Installation

```bash
# Clone repository
git clone <your-repo-url>
cd Langgraph_project

# Create virtual environment
python -m venv .venv
source .venv/bin/activate  # On Windows: .venv\Scripts\activate

# Install dependencies
pip install -r requirements.txt
```

### Configuration

Create `.envs/.env.local`:

```bash
# Required
OPENAI_API_KEY="sk-your-key-here"

# Optional (for observability)
LANGFUSE_PUBLIC_KEY="pk-lf-..."
LANGFUSE_SECRET_KEY="sk-lf-..."
LANGFUSE_BASE_URL="https://cloud.langfuse.com"

# Model Configuration
LLM_MODEL="gpt-4-turbo"
EMBEDDINGS_MODEL="text-embedding-3-small"
LLM_TEMPERATURE=0.0

# Retrieval Settings
CHUNK_SIZE=1000
CHUNK_OVERLAP=100
RETRIEVAL_DOCS=3

# Hallucination Detection Weights
CITATION_SCORE=0.4
LLM_SCORE=0.4
CONSISTENCY_SCORE=0.2
```

### Usage

```bash
# Run on a paper
python main.py --arxiv-id 1706.03762 --question "What is the main contribution?"

# Run demo (quick demonstration)
python run_demo.py

# Run RAGAS evaluation
python run_evaluation.py
```

### RAGAS Evaluation

The system includes comprehensive evaluation using RAGAS metrics:

```bash
python run_evaluation.py
```

**Metrics Computed:**
- **Faithfulness**: Measures factual consistency with retrieved context
- **Answer Relevancy**: Measures how relevant the answer is to the question
- **Hallucination Risk**: Multi-layered verification (citation + NLI + consistency)
- **Retrieval Quality**: Average relevance scores of retrieved chunks

**Expected Output:**
```
RAGAS Metrics:
  • Faithfulness:      
  • Answer Relevancy:  

System Metrics:
  • Questions Evaluated:
  • Avg Hallucination Risk:  
  • Avg Retrieval Relevance: 
```

## Demo Results

### Test Paper: "Attention Is All You Need" (1706.03762)

**Question**: What is the main contribution of the Transformer architecture?

**Answer**:
> The main contribution of the Transformer architecture is its reliance solely on an attention mechanism, specifically designed to draw global dependencies between input and output, which allows for significant parallelization of the training process. This design eschews traditional recurrence and convolutional methods used in other models, enabling the Transformer to achieve state-of-the-art results in translation quality with considerably reduced training time.

**Citations**: [Chunk 5]

**Metrics**:
- **Hallucination Risk**: 15.56% (LOW)
- **Citation Verification**: 0% (Perfect - all citations verified)
- **NLI Claim Verification**: 0% (Perfect - 3/3 claims supported)
- **Retrieval Quality**: 87.75% average relevance score

### Hallucination Detection Breakdown

| Component | Score | Weight | Contribution |
|-----------|-------|--------|--------------|
| Citation Verification | 0% | 40% | 0% |
| NLI Claim Verification | 0% | 40% | 0% |
| Consistency Check | 77.79% | 20% | 15.56% |
| **Final Score** | **15.56%** | - | **LOW RISK** |

**NLI Verification Details**:

All 3 extracted claims were verified as SUPPORTED:

1. "The Transformer architecture primarily relies on an attention mechanism to draw global dependencies between input and output."
2. "The design of the Transformer allows for significant parallelization of the training process."
3. "The Transformer achieves state-of-the-art results in translation quality with reduced training time."

## Observability with Langfuse

### Trace Overview

![Langfuse Traces](screenshots/langfuse_traces.png)

### Detailed Trace Execution

![Langfuse Trace Detail](screenshots/langfuse_trace_detail.png)

### Hallucination Detection Insights

![Hallucination Check](screenshots/hallucination_check3.png)
![Hallucination Check](screenshots/hallucination_check4.png)

## 🛠️ Technical Stack

- **Framework**: LangGraph, LangChain
- **LLM**: OpenAI GPT-4 Turbo
- **Embeddings**: OpenAI text-embedding-3-small
- **Vector Store**: ChromaDB
- **Observability**: Langfuse
- **Evaluation**: RAGAS
- **Async**: asyncio

## Project Structure

```
Langgraph_project/
├── agents/              # Agent implementations
│   ├── fetcher.py      # ArXiv paper fetcher
│   ├── parser.py       # Section parser
│   ├── summarizer.py   # Summary generator
│   ├── qa_agent.py     # Q&A with RAG
│   ├── hallucination_detector.py  # Multi-layered verification
│   └── vectorstore_agent.py       # Vector DB management
├── core/               # Core configuration
│   ├── config.py       # Settings management
│   └── logging.py      # Logging setup
├── utils/              # Utilities
│   ├── arxiv_fetcher.py   # ArXiv API wrapper
│   ├── vector_store.py    # ChromaDB manager
│   ├── prompts.py         # LLM prompts
│   └── chunker.py         # Text chunking
├── tests/              # Evaluation
│   └── evaluate.py     # RAGAS evaluation
├── main.py             # Main entry point
├── run_demo.py         # Demo script
└── requirements.txt    # Dependencies
```

## License

MIT License
