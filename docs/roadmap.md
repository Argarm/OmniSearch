# OmniSearch — Roadmap

Status of the project and what's planned next. For the architectural rationale
behind individual decisions, see [docs/adr/](adr/).

## Done

- **RAG core** — LCEL chain with query rewriting, Qdrant retrieval, SSE streaming.
- **Ingestion pipeline** — PDF / Notion / Confluence connectors, chunking,
  BGE embeddings, vector store upsert.
- **Multi-provider LLM** — swappable OpenAI / Anthropic backends behind the
  `LLMProvider` interface, selected via `LLM_PROVIDER` (see
  [ADR 0001](adr/0001-provider-abstraction.md)).
- **CI** — GitHub Actions running `ruff`, `mypy`, and `pytest` (unit + integration
  against an ephemeral Qdrant) on every PR.
- **Ingestion automation** — nightly cron + push-triggered + manual `workflow_dispatch`.
- **Containerization** — `docker compose` stack (Qdrant + backend + frontend) with
  a dev override for hot reload.

## In progress

- **Cost & latency benchmarking** — published-pricing estimates in the README; a
  `scripts/bench_providers.py` helper to measure real numbers per provider locally.

## Planned

### RAG quality tuning (once real documents are indexed)
Adjust in `config/settings.yaml`:
- `chunking.chunk_size` / `chunk_overlap` — raise for long-paragraph documents or
  when answers lose cross-chunk context.
- `retrieval.top_k` (try 8–10 for incomplete answers) and `score_threshold`
  (lower for sparse results, raise to cut noise).
- `embedding.device: cuda` when a GPU is available (~10× faster indexing).
- Lighter embedding model option: `BAAI/bge-base-en-v1.5` (768-dim, half the RAM);
  requires recreating the collection and re-indexing.

### Local / offline LLM (no cost)
Use Ollama through the OpenAI-compatible path — no code changes:
```bash
ollama pull llama3.2
# .env:
OPENAI_BASE_URL=http://localhost:11434/v1
OPENAI_MODEL=llama3.2
OPENAI_API_KEY=ollama   # placeholder; Ollama ignores it
```

### Production deployment
- Linux VM with Docker (≥ 8 GB RAM for the embedding model).
- Persistent Qdrant volume (`qdrant_data`) on durable storage; point the nightly
  `indexer.yml` at a `QDRANT_URL` secret instead of the ephemeral CI service.
- Reverse proxy (Nginx / Traefik) with TLS in front of port 8501.
- Chainlit auth via `CHAINLIT_AUTH_SECRET` for access-gated deployments.

### Operational hardening
- Stress-test suite over complex PDFs with grounding checks
  (`RUN_STRESS_TESTS=1 pytest tests/stress/ -v`).
- Collection reset / re-index tooling (`scripts/reset_collection.py`).

## Quick command reference

| Action | Command |
|---|---|
| Full stack | `docker compose up --build` |
| Qdrant only (local) | `docker run -d -p 6333:6333 qdrant/qdrant` |
| Index PDFs | `python -m ingestion.pipeline --source pdf` |
| Index everything | `python -m ingestion.pipeline --source all` |
| Unit tests | `pytest tests/unit/ -v` |
| Integration tests | `QDRANT_URL=http://localhost:6333 pytest tests/integration/ -v` |
| Backend logs | `docker compose logs -f backend` |
| Swagger UI (dev) | `http://localhost:8000/docs` |
| Qdrant dashboard (dev) | `http://localhost:6333/dashboard` |
