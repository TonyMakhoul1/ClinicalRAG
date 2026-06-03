# ClinicalRAG

A production-ready Retrieval-Augmented Generation (RAG) system for clinical document Q&A. Ask questions in natural language, and get grounded answers with confidence scores and source citations — streamed in real time.

---

## Features

- **Hybrid retrieval** — BM25 keyword search + dense vector search combined via Reciprocal Rank Fusion
- **Cross-encoder reranking** — `ms-marco-MiniLM-L-6-v2` reranks candidates before generation
- **Multi-agent pipeline** — CrewAI planner/researcher/synthesizer decomposes complex questions into focused sub-queries
- **LLM guardrails** — input relevance check + output grounding check using a fast 8B model
- **Semantic cache** — Redis vector cache; similar questions return instantly with zero LLM calls
- **JWT authentication** — bcrypt-hashed credentials, HS256 tokens, configurable expiry
- **SSE streaming** — answers stream token by token directly to the UI
- **Confidence scoring** — sigmoid of the top reranked chunk score, displayed as a color-coded badge
- **LangSmith observability** — full trace of every agent call and LLM interaction
- **RAGAS evaluation** — automated pipeline scoring faithfulness, answer relevancy, context precision, context recall
- **CI/CD** — GitHub Actions builds and deploys to AWS EC2 on every push to `main`

---

## Architecture

```
┌─────────────────────────────────────────────────────────┐
│                    Streamlit UI :8501                   │
│         Login → Chat → Streaming → Sources             │
└──────────────────────┬──────────────────────────────────┘
                       │ HTTPS  JWT
┌──────────────────────▼──────────────────────────────────┐
│                  FastAPI Backend :8000                  │
│   /auth/login   /chat/answer   /chat/stream  /health   │
└──────┬─────────────────────────────────┬────────────────┘
       │                                 │
┌──────▼──────┐                 ┌────────▼────────┐
│ Redis Cache │                 │  CrewAI Agents  │
│ Semantic    │                 │                 │
│ Vector      │                 │ Planner  (8B)   │
│ Search      │                 │ Researcher(70B) │
└─────────────┘                 │ Synthesizer(70B)│
                                └────────┬────────┘
                                         │
                          ┌──────────────▼──────────────┐
                          │      Retrieval Pipeline      │
                          │  BM25 + Dense → Reranker    │
                          │  ChromaDB  BAAI/bge-small   │
                          └─────────────────────────────┘
```

---

## Tech Stack

| Layer | Technology |
|---|---|
| LLM | Groq — `llama-3.3-70b-versatile` (synthesis) + `llama-3.1-8b-instant` (guardrails/planning) |
| Embeddings | `BAAI/bge-small-en-v1.5` via HuggingFace |
| Vector Store | ChromaDB (persistent) |
| Reranker | `cross-encoder/ms-marco-MiniLM-L-6-v2` |
| Agents | CrewAI |
| Cache | Redis (`redis-stack-server`) with redisvl |
| Backend | FastAPI + uvicorn |
| Frontend | Streamlit |
| Observability | LangSmith + structlog |
| Evaluation | RAGAS |
| Infra | Docker Compose, AWS ECR + EC2 |
| CI/CD | GitHub Actions |

---

## Project Structure

```
├── src/
│   ├── rag_ingestion/
│   │   ├── rag_ingestion.py          # PDF → chunks → ChromaDB
│   │   ├── retrieval.py              # hybrid search + reranking + generation
│   │   └── config/rag_ingestion_settings.py
│   ├── agents_src/
│   │   ├── agents.py                 # planner, researcher, synthesizer
│   │   ├── crew.py                   # CrewAI orchestration
│   │   ├── tasks.py                  # task prompts
│   │   └── tools/rag_tool.py         # wraps retrieve() as a CrewAI tool
│   ├── backend_src/
│   │   ├── api/app.py                # FastAPI routes + middleware
│   │   ├── auth/auth.py              # JWT + bcrypt
│   │   ├── services/
│   │   │   ├── get_rag_resp.py       # pipeline: cache → guardrails → crew → stream
│   │   │   ├── guardrails.py         # input + output LLM guards
│   │   │   └── semantic_cache.py     # Redis vector cache
│   │   └── config/backend_settings.py
│   ├── frontend_src/
│   │   ├── app.py                    # Streamlit UI
│   │   └── config/frontend_settings.py
│   └── evaluation/
│       └── evaluate.py               # RAGAS evaluation suite
├── docs_dir/                         # PDF documents to ingest
├── .github/workflows/
│   └── ci-cd.yml                     # lint → build → deploy
├── docker-compose.yml                # all services
├── docker-compose.prod.yml           # ECR image overrides for production
├── Dockerfile
└── pyproject.toml
```

---

## Getting Started (Local)

### Prerequisites

- Python 3.12+
- [uv](https://docs.astral.sh/uv/)
- Docker Desktop
- A [Groq](https://console.groq.com) API key

### 1. Clone and install dependencies

```bash
git clone https://github.com/<your-username>/ClinicalRAG.git
cd ClinicalRAG
uv sync
```

### 2. Configure environment

```bash
cp .env.example .env
```

Fill in the required values in `.env`:

```env
GROQ_API_KEY=your_groq_api_key

SECRET_KEY=<generate with: python -c "import secrets; print(secrets.token_hex(32))">
ADMIN_USERNAME=admin
ADMIN_PASSWORD_HASH=<generate with: python -c "import bcrypt; print(bcrypt.hashpw('yourpassword'.encode(), bcrypt.gensalt()).decode())">
```

### 3. Add your documents

Place PDF files in the `docs_dir/` folder.

### 4. Start Redis

```bash
docker compose up redis -d
```

### 5. Run ingestion

```bash
uv run python -c "from src.rag_ingestion.rag_ingestion import ingestion; ingestion()"
```

### 6. Start the backend and frontend

```bash
# Terminal 1
uv run uvicorn src.backend_src.api.app:app --reload

# Terminal 2
uv run streamlit run src/frontend_src/app.py
```

Open `http://localhost:8501`, log in, and start asking questions.

---

## Docker (Full Stack)

```bash
docker compose up --build
```

Services started: `redis`, `backend` (:8000), `streamlit` (:8501).

To ingest documents into the Docker volume:

```bash
docker compose --profile tools run --rm ingestion
```

To run the RAGAS evaluation suite:

```bash
docker compose --profile tools run --rm evaluation
```

---

## Environment Variables

| Variable | Required | Description |
|---|---|---|
| `GROQ_API_KEY` | Yes | Groq API key |
| `MODEL_NAME` | No | Main LLM (default: `llama-3.3-70b-versatile`) |
| `GUARDRAIL_MODEL` | No | Fast LLM for guardrails/planning (default: `llama-3.1-8b-instant`) |
| `MODEL_TEMPERATURE` | No | LLM temperature (default: `0.0`) |
| `VECTOR_STORE_DIR` | Yes | Absolute path to ChromaDB persistence directory |
| `DOCUMENTS_DIR` | Yes | Absolute path to PDF documents directory |
| `COLLECTION_NAME` | No | ChromaDB collection name (default: `my_collection1`) |
| `REDIS_URL` | No | Redis connection URL (default: `redis://localhost:6379`) |
| `SECRET_KEY` | Yes | JWT signing key |
| `JWT_ALGORITHM` | No | JWT algorithm (default: `HS256`) |
| `JWT_EXPIRE_MINUTES` | No | Token expiry in minutes (default: `60`) |
| `ADMIN_USERNAME` | Yes | Login username |
| `ADMIN_PASSWORD_HASH` | Yes | bcrypt hash of the admin password |
| `LANGCHAIN_TRACING_V2` | No | Enable LangSmith tracing (`true`/`false`) |
| `LANGCHAIN_API_KEY` | No | LangSmith API key |
| `LOG_FORMAT` | No | `console` (local) or `json` (Docker) |

---

## CI/CD Pipeline

Every push to `main` triggers the full pipeline automatically.

```
push to main
     │
     ▼
┌─────────────┐     ┌──────────────────┐     ┌──────────────────────┐
│    Lint     │────▶│  Build & Push    │────▶│       Deploy         │
│             │     │                  │     │                      │
│ ruff check  │     │ docker build     │     │ write .env           │
│ src/        │     │ push to ECR      │     │ ECR login            │
│             │     │ (sha + latest)   │     │ docker compose pull  │
└─────────────┘     └──────────────────┘     │ run ingestion        │
GitHub-hosted        GitHub-hosted           │ docker compose up -d │
                                             └──────────────────────┘
                                              EC2 self-hosted runner
```

### GitHub Secrets required

| Secret | Description |
|---|---|
| `AWS_ACCESS_KEY_ID` | IAM user with ECR push permissions |
| `AWS_SECRET_ACCESS_KEY` | IAM user secret |
| `ECR_REGISTRY` | e.g. `<account>.dkr.ecr.eu-west-2.amazonaws.com` |
| `ECR_REPO` | ECR repository name (e.g. `rag-app`) |
| `GROQ_API_KEY` | Groq API key |
| `SECRET_KEY` | JWT signing key |
| `ADMIN_PASSWORD_HASH` | bcrypt hash (no surrounding quotes) |
| `LANGCHAIN_API_KEY` | LangSmith API key (optional) |
| `LANGSMITH_API_KEY` | LangSmith API key (optional) |

---

## Request Flow

```
User question
      │
      ▼
JWT authentication
      │
      ▼
Rate limit check (5 req/min)
      │
      ▼
Semantic cache lookup ──── HIT ───▶ stream cached answer
      │ MISS
      ▼
Input guardrail (8B) ──── REJECT ──▶ error message
      │ PASS
      ▼
Planner agent (8B)
  └─▶ decomposes into 1–4 sub-questions
      │
      ▼
Researcher agent (70B) × N sub-questions
  └─▶ BM25 + Dense retrieval → Cross-encoder reranking → top 5 chunks
  └─▶ 70B generates answer per sub-question
      │
      ▼
Synthesizer agent (70B)
  └─▶ combines all sub-answers into final response
      │
      ▼
Output guardrail (8B) ──── REJECT ──▶ retraction message
      │ PASS
      ▼
Stream to UI (SSE) + store in Redis cache
```

---

## Evaluation

Run the RAGAS evaluation suite to benchmark retrieval and generation quality:

```bash
uv run python -m src.evaluation.evaluate
```

Metrics scored across the test set:

| Metric | Description | Target |
|---|---|---|
| `faithfulness` | Answer is grounded in retrieved chunks | ≥ 0.8 |
| `answer_relevancy` | Answer addresses the question | ≥ 0.8 |
| `context_precision` | Most useful chunks ranked highest | ≥ 0.8 |
| `context_recall` | All needed information was retrieved | ≥ 0.8 |

Results are pushed to LangSmith for tracking across runs.

---

## 👨‍💻 Author

Tony Makhoul – Computer Engineering Student, AI Engineer

📧 Contact: tmakhoul2002@gmail.com

🔗 [LinkedIn](https://www.linkedin.com/in/tony-makhoul-05b6b7243)  

🔗 [GitHub](https://github.com/TonyMakhoul1)