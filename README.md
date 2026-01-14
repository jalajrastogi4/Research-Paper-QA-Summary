# LangGraph Multi-Agent Research Paper Analysis System

A production-ready multi-agent system built with LangGraph for analyzing research papers from arXiv, featuring advanced hallucination detection, distributed task processing and comprehensive observability.

## Key Features

- **Multi-Agent Architecture**: Specialized agents for fetching, parsing, summarizing, Q&A, and verification
- **🚀 3-Tier Intelligent Caching System**: 
  - **Tier 1 (QA Cache)**: Question already answered → **4-7s response** (instant DB lookup)
  - **Tier 2 (Paper Cache)**: Paper processed, new question → **40-50s response** (skip embeddings)
  - **Tier 3 (Full Pipeline)**: New/outdated paper → **60-70s response** (complete processing)
- **Advanced Hallucination Detection**: Multi-layered verification combining:
  - Citation verification
  - NLI-based claim checking
  - Consistency analysis across answer variations
- **Production Observability**: Full Langfuse integration for tracing and monitoring
- **Async Architecture**: Built with async/await for optimal performance
- **RAGAS Evaluation**: Automated evaluation with faithfulness and answer relevancy metrics
- **Vector-based RAG**: ChromaDB for efficient semantic search and retrieval
- **RESTful API**: Production-ready FastAPI with async endpoints
- **Distributed Task Processing**: Celery workers with Redis for scalable async operations
- **PostgreSQL Database**: Persistent storage for papers, summaries, QA cache, and job status

## Architecture

### System Overview

```mermaid
graph TB
    Client[Client/User] --> API[FastAPI Server]
    API --> Cache{Check Cache}
    API --> Jobs[Job Status API]
    
    Cache -->|Hit| DB[(PostgreSQL)]
    Cache -->|Miss| Celery[Celery Worker]
    
    Celery --> Graph[LangGraph Agent System]
    
    Graph --> Fetch[Fetch Agent]
    Fetch --> Parse[Parser Agent]
    Parse --> Summarize[Summarizer Agent]
    Summarize --> Vector[Vector Store Agent]
    Vector --> QA[QA Agent]
    QA --> Hallucination[Hallucination Detector]
    
    Vector --> Pinecone[(Pinecone)]
    Vector --> ChromaDB[(ChromaDB)]
    
    Graph --> DB
    
    Hallucination --> Result[Final Answer]
    Result --> DB
    
    API --> Redis[(Redis)]
    Celery --> Redis
    
    Graph -.-> Langfuse[Langfuse Observability]
    
    style API fill:#4A90E2,stroke:#2E5C8A,color:#fff
    style Graph fill:#7B68EE,stroke:#5A4FCF,color:#fff
    style Hallucination fill:#FF6B6B,stroke:#E85555,color:#fff
    style DB fill:#50C878,stroke:#3BA05F,color:#fff
    style Pinecone fill:#FF9E3D,stroke:#E88825,color:#fff
```

### 🎯 3-Tier Intelligent Caching System

The system implements a sophisticated caching strategy that dramatically reduces response time and API costs:

#### Tier 1: QA Cache (Fastest - 4-7s)
**Scenario**: Exact question already answered for this paper

**Flow**:
1. Request arrives with `arxiv_id` + `question`
2. Hash question and lookup in `paper_qa_cache` table
3. If found: **Instant return** from database
4. No agent processing, no LLM calls, no embeddings

**Performance**:
- Response Time: **4-7 seconds**
- Cost: **$0** (no API calls)
- Cache Key: `arxiv_id + question_hash`

**Example**:
```python
# First request: 90s
POST /papers/analyze {"arxiv_id": "1706.03762", "question": "What is the Transformer?"}

# Same question later: 5s
POST /papers/analyze {"arxiv_id": "1706.03762", "question": "What is the Transformer?"}
```

#### Tier 2: Paper Cache (Medium - 40-50s)
**Scenario**: Paper processed before, but question is new

**Flow**:
1. QA cache miss → Check `research_papers` table
2. If paper exists and is current (dates match):
   - Load cached summary and metadata
   - **Skip**: Fetch, Parse, Summarize, Vector Store agents
   - **Run**: QA + Hallucination Detection only
3. Retrieve from existing vector embeddings
4. Generate answer and store in QA cache

**Performance**:
- Response Time: **40-50 seconds**
- Cost: **~40% of full pipeline** (QA + verification only)
- Savings: Skip PDF processing, parsing, summarization, embedding creation

**Cache Validation**:
- Checks if arXiv paper was modified since last cache
- If outdated → Falls through to Tier 3

**Example**:
```python
# First question: ~60s (full pipeline)
POST /papers/analyze {"arxiv_id": "1706.03762", "question": "What is attention?"}

# New question, same paper: ~40s (QA only)
POST /papers/analyze {"arxiv_id": "1706.03762", "question": "What datasets were used?"}
```

#### Tier 3: Full Pipeline (Slowest - 60-70s)
**Scenario**: New paper or outdated cache

**Flow**:
1. QA cache miss + Paper cache miss/outdated
2. **Run complete pipeline**:
   - Fetch paper from arXiv
   - Parse sections with LLM
   - Generate structured summary
   - Create text chunks
   - Generate embeddings (OpenAI)
   - Store in vector database (Pinecone)
   - Run QA with RAG
   - Perform hallucination detection
3. Cache everything for future requests

**Performance**:
- Response Time: **60-70 seconds**
- Cost: **Full** (all LLM + embedding calls)
- Operations: ~15-20 LLM calls + embeddings for all chunks

**Triggers**:
- Paper never processed before
- Paper modified on arXiv (date check)
- Cache manually invalidated

**Example**:
```python
# Brand new paper: ~60s (full processing)
POST /papers/analyze {"arxiv_id": "2312.00752", "question": "What is this about?"}
```

### Cache Performance Comparison

| Tier | Scenario | Response Time | Cost | LLM Calls | Embeddings |
|------|----------|---------------|------|-----------|------------|
| **Tier 1** | QA Cache Hit | **4-7s** | $0 | 0 | 0 |
| **Tier 2** | Paper Cached | **40-50s** | ~$0.15 | 3-5 | 0 |
| **Tier 3** | Full Pipeline | **60-70s** | ~$0.40 | 15-20 | All chunks |

### Cache Implementation Details

**Database Tables**:
- `paper_qa_cache`: Stores question-answer pairs with MD5 hash lookup
- `research_papers`: Stores paper metadata, summaries, and vector status
- `job_status`: Tracks async job execution

**Cache Validation**:
```python
# Freshness check
if cached_paper.last_modified_date >= arxiv_paper.last_modified_date:
    use_cache = True  # Paper hasn't changed
else:
    invalidate_cache()  # Paper updated on arXiv
```

**Smart Routing** (in `PaperService.analyze_paper`):
```python
1. Check QA cache (Tier 1)
   ├─ Hit → Return immediately
   └─ Miss → Continue

2. Check paper cache + fetch arXiv date (Tier 2)
   ├─ Hit + Current → Run QA only
   └─ Miss or Outdated → Continue

3. Run full pipeline (Tier 3)
   └─ Cache all results
```


### Agent Responsibilities

| Agent | Purpose | Key Features |
|-------|---------|--------------|
| **Cache Loader** | Manages database caching | Paper freshness validation, cache routing decisions |
| **Fetcher** | Downloads papers from arXiv | SSL bypass, PDF extraction, metadata parsing |
| **Parser** | Extracts structured sections | LLM-based section identification |
| **Summarizer** | Generates structured summaries | Pydantic output parsing, key findings extraction |
| **Vector Store** | Creates embeddings database | Pinecone/ChromaDB persistence, chunking strategy |
| **QA Agent** | Answers questions with citations | RAG with top-k retrieval, citation tracking |
| **Hallucination Detector** | Verifies answer accuracy | Citation check + NLI verification + consistency |

## Quick Start

### Prerequisites

- Python 3.10+
- PostgreSQL database (local or cloud)
- Redis instance (local or cloud)
- OpenAI API key
- Pinecone API key
- Langfuse account (optional, for observability)

### Installation

```bash
# Clone repository
git clone <your-repo-url>
cd Temp_LanggraphProject2

# Create virtual environment
python -m venv .venv

# Activate virtual environment
# On Windows:
.venv\Scripts\activate
# On macOS/Linux:
source .venv/bin/activate

# Install dependencies
pip install -r requirements.txt
```

### Configuration

1. Copy environment template:
```bash
cp .envs/env.template .envs/.env.local
```

2. Edit `.envs/.env.local` with your credentials:

```bash
# ===== Required: OpenAI =====
OPENAI_API_KEY="sk-your-key-here"

# ===== Required: Database =====
DATABASE_URL="postgresql+asyncpg://user:password@host:5432/database"

# ===== Required: Redis & Celery =====
REDIS_URL="redis://default:password@host:6379"
CELERY_BROKER_URL=""  # Leave empty to use REDIS_URL
CELERY_RESULT_BACKEND_URL=""  # Leave empty to use REDIS_URL

# ===== Required: Pinecone Vector Store =====
PINECONE_API_KEY="your-pinecone-key"
PINECONE_INDEX_NAME="research-papers"
PINECONE_ENVIRONMENT="us-east-1"

# ===== Optional: Langfuse (Observability) =====
LANGFUSE_PUBLIC_KEY="pk-lf-..."
LANGFUSE_SECRET_KEY="sk-lf-..."
LANGFUSE_BASE_URL="https://cloud.langfuse.com"

# ===== Model Configuration =====
LLM_MODEL="gpt-4-turbo"
EMBEDDINGS_MODEL="text-embedding-3-small"
LLM_TEMPERATURE=0.0

# ===== Retrieval Settings =====
CHUNK_SIZE=1000
CHUNK_OVERLAP=100
RETRIEVAL_DOCS=3

# ===== Hallucination Detection Weights =====
CITATION_SCORE=0.4
LLM_SCORE=0.4
CONSISTENCY_SCORE=0.2
```

### Database Setup

```bash
# Run database migrations
alembic upgrade head
```

### Running the Application

#### Option 1: Full Stack (API + Worker)

**Terminal 1 - Start FastAPI Server:**
```bash
uvicorn api.main:app --host 0.0.0.0 --port 8000 --reload
```

**Terminal 2 - Start Celery Worker:**
```bash
celery -A workers.celery_config.celery_app worker --loglevel=info --pool=solo -Q paper_analysis,celery
```

**Access API:**
- API Docs: http://localhost:8000/docs
- Health Check: http://localhost:8000/health

#### Option 2: CLI Mode (Direct Execution)

```bash
# Run on a paper
python main.py --arxiv-id 1706.03762 --question "What is the main contribution?"
```

```bash
# Run demo (quick demonstration)
python run_demo.py
```
Sample output : [results/demo_results_20260101_135354.json]

```bash
# Run RAGAS evaluation
python run_evaluation.py
```
Sample output : [results/evaluation_results_20251207_150517.json]

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

### Run Unit Tests

```bash
pytest tests/
```

## 🔧 Technology Stack

| Category | Technology |
|----------|-----------|
| **Framework** | FastAPI, LangGraph, LangChain |
| **Language** | Python 3.10+ |
| **LLM** | OpenAI GPT-4 Turbo |
| **Embeddings** | OpenAI text-embedding-3-small |
| **Vector Stores** | Pinecone (primary), ChromaDB (legacy) |
| **Database** | PostgreSQL with SQLModel |
| **ORM** | SQLModel (SQLAlchemy 2.0) |
| **Migrations** | Alembic |
| **Task Queue** | Celery with Redis |
| **Caching** | Redis |
| **Observability** | Langfuse |
| **Evaluation** | RAGAS |
| **Async** | asyncio, aiofiles |
| **HTTP Client** | httpx |
| **Testing** | pytest, pytest-asyncio |

## 📁 Project Structure

```
Temp_LanggraphProject2/
├── agents/                     # LangGraph Agent Implementations
│   ├── cache_loader_agent.py  # Database cache management
│   ├── fetcher.py              # ArXiv paper fetcher
│   ├── graph_input.py          # Pydantic model for graph input
│   ├── hallucination_detector.py  # Multi-layered verification
│   ├── parser.py               # Section parser
│   ├── qa_agent.py             # Q&A with RAG
│   ├── research_agent.py       # Main agent orchestrator (with DB)
│   ├── research_agent_evaluate.py  # Agent for evaluation (without DB)
│   ├── state.py                # State for LangGraph
│   ├── summarizer.py           # Summary generator
│   └── vectorstore_agent.py    # Vector DB management
│
├── api/                        # FastAPI Application
│   ├── main.py                 # FastAPI app setup & lifespan
│   ├── models/                 # SQLModel database models
│   │   ├── job_status.py       # Job status tracking
│   │   ├── paper_qa_cache.py   # QA cache model
│   │   └── research_paper.py   # Research paper model
│   ├── routers/                # API route handlers
│   │   ├── health.py           # Health check endpoints
│   │   ├── jobs.py             # Job status endpoints
│   │   └── papers.py           # Paper analysis endpoints
│   ├── schemas/                # Pydantic request/response schemas
│   │   ├── health.py           # Health check schemas
│   │   ├── job.py              # Job schemas
│   │   └── paper.py            # Paper analysis schemas
│   └── services/               # Business logic services
│       ├── cache_service.py    # QA cache operations
│       ├── crud.py             # Database CRUD operations
│       ├── evaluation_service.py  # Evaluation logic
│       └── paper_service.py    # Paper processing logic
│
├── core/                       # Core Configuration
│   ├── config.py               # Settings management with Pydantic
│   ├── db.py                   # Database connection & session management
│   ├── health.py               # Health check utilities
│   └── logging.py              # Logging setup with Loguru
│
├── workers/                    # Celery Task Queue
│   ├── celery_config.py        # Celery app configuration
│   └── tasks/
│       └── paper_tasks.py      # Paper analysis Celery tasks
│
├── utils/                      # Utility Modules
│   ├── arxiv_fetcher.py        # ArXiv API wrapper
│   ├── chunker.py              # Text chunking strategy
│   ├── llm.py                  # LLM wrapper (OpenAI)
│   ├── prompts.py              # LLM prompts for all agents
│   ├── qa.py                   # QA utilities
│   ├── tracing.py              # Langfuse tracing setup
│   └── vector_store.py         # Vector store manager (Pinecone/ChromaDB)
│
├── tests/                      # Testing & Evaluation
│   ├── components/             # Component tests
│   │   ├── test_all_health_checks.py
│   │   ├── test_database.py
│   │   └── test_pinecone_*.py
│   └── evaluate.py             # RAGAS evaluation
│
├── alembic/                    # Database Migrations
│   ├── versions/               # Migration scripts
│   └── env.py                  # Alembic environment
│
├── data/                       # Data Storage
│   ├── arxiv/                  # Downloaded PDF papers
│   └── chromadb/               # ChromaDB persistence (legacy)
│
├── logs/                       # Application Logs
│   ├── debug.log
│   └── error.log
│
├── results/                    # Evaluation Results
│   ├── demo_results_*.json
│   └── evaluation_results_*.json
│
├── screenshots/                # Documentation Screenshots
│
├── .envs/                      # Environment Variables
│   └── env.template            # Environment template
│
├── main.py                     # CLI entry point
├── run_demo.py                 # Demo script
├── run_evaluation.py           # Evaluation script
├── requirements.txt            # Python dependencies
├── alembic.ini                 # Alembic configuration
├── pytest.ini                  # Pytest configuration
├── render.yaml                 # Render.com deployment config
└── README.md                   # This file
```

## License

MIT License
