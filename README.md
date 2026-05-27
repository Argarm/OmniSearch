# DocuStream

Internal RAG (Retrieval-Augmented Generation) system for organizational knowledge bases.

## Stack

| Component | Technology |
|---|---|
| Vector DB | Qdrant (self-hosted, Docker) |
| Embeddings | `BAAI/bge-large-en-v1.5` (HuggingFace) |
| Orchestration | LangChain + LCEL |
| LLM | OpenAI-compatible (default: `gpt-4o-mini`) |
| Backend | FastAPI + SSE streaming |
| Frontend | Chainlit |
| CI/CD | GitHub Actions |

## Quick Start

### 1. Configure environment

```bash
cp .env.example .env
# Edit .env with your API keys
```

### 2. Start the full stack

```bash
docker compose up --build
# Frontend available at http://localhost:8501
```

### 3. Run the ingestion pipeline

```bash
# Drop PDFs into data/sources/ then:
docker compose --profile ingestion run ingestion

# Or trigger manually via GitHub Actions → workflow_dispatch
```

## Development

```bash
# Install all dependencies
pip install -e ".[ingestion,backend,frontend,dev]"

# Start services with hot reload
docker compose -f docker-compose.yml -f docker-compose.dev.yml up

# Run tests
pytest tests/unit/ -v
pytest tests/integration/ -v   # Requires Qdrant running
```

## Architecture

```
GitHub Actions (indexer.yml)
  └── ingestion/pipeline.py
        ├── connectors/ (PDF, Notion, Confluence)
        ├── chunker.py  (RecursiveCharacterTextSplitter, token-based)
        ├── embedder.py (BAAI/bge-large-en-v1.5, BGE asymmetric)
        └── vector_store.py (Qdrant upsert, idempotent)
              │
              ▼ (persisted vectors)
         Qdrant collection "docustream"
              │
              ▼ (at query time)
         backend/main.py (FastAPI)
           └── /api/v1/query
                 ├── retriever.py  (cosine similarity search)
                 ├── chain.py      (LCEL: retrieve → prompt → LLM)
                 └── SSE streaming
                       │
                       ▼
               frontend/app.py (Chainlit)
                 ├── Streaming token display
                 └── Source document panel
```

## Data Sources

| Source | Connector | Env Vars Required |
|---|---|---|
| Local PDFs | `PdfConnector` | `PDF_SOURCE_DIR` |
| Notion | `NotionConnector` | `NOTION_TOKEN`, `NOTION_ROOT_PAGE_ID` |
| Confluence | `ConfluenceConnector` | `CONFLUENCE_URL`, `CONFLUENCE_USER_EMAIL`, `CONFLUENCE_TOKEN`, `CONFLUENCE_SPACE_KEY` |

## Ingestion Automation

The indexer runs automatically:
- **Nightly** at 2AM UTC (cron schedule)
- **On push** when files are added to `data/sources/`
- **Manually** via GitHub Actions → Run workflow (choose source type)

## Security

- Only the frontend port (8501) is exposed to the host
- All API keys are stored as GitHub Secrets / Docker env vars
- Qdrant and the backend communicate over an internal Docker network
- The LLM is instructed to cite sources and refuse to answer outside the knowledge base

## Stress Testing

```bash
# Place complex PDFs in tests/stress/fixtures/ then:
RUN_STRESS_TESTS=1 pytest tests/stress/ -v
```
