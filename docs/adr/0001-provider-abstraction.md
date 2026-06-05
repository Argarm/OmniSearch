# ADR 0001 — LLM Provider Abstraction

- **Status:** Accepted
- **Date:** 2026-06-05
- **Deciders:** Aarón García Marrero

## Context

OmniSearch generates answers with a chat LLM at the end of the RAG chain. We want
to demonstrate that the model backend is interchangeable — specifically OpenAI
(GPT) and Anthropic (Claude) — for three reasons:

1. **Avoid vendor lock-in.** Pricing, latency, and availability differ between
   providers and change over time; the system should not be welded to one vendor.
2. **Cost/quality tuning.** A cheap model (`gpt-4o-mini`, `claude-3-5-haiku`) is
   fine for most queries, but we may want to escalate to a stronger model without
   touching the chain logic.
3. **Local/offline runs.** The OpenAI path is OpenAI-compatible, so an Ollama
   endpoint (`OPENAI_BASE_URL`) can stand in with zero code changes.

LangChain already exposes a common `BaseChatModel` type that both `ChatOpenAI`
and `ChatAnthropic` implement, so a bare factory returning `BaseChatModel` would
technically work. The open question was whether to expose that LangChain type
directly across the codebase or hide it behind our own boundary.

## Decision

Introduce a small in-house `LLMProvider` interface (`backend/llm_factory.py`):

- An abstract base class `LLMProvider` with `model_id` and `build() -> BaseChatModel`.
- Two concrete implementations: `OpenAIProvider` and `AnthropicProvider`, each
  reading its own config from the environment.
- A registry + `get_provider()` selector driven by the `LLM_PROVIDER` env var,
  which raises `ValueError` on an unknown value (fail fast instead of silently
  defaulting).
- `get_llm()` remains the public entry point (`get_provider().build()`) so existing
  consumers like `RAGChain` need no changes.

Provider selection and model overrides are configuration, not code:

```bash
LLM_PROVIDER=anthropic
ANTHROPIC_MODEL=claude-3-5-haiku-20241022
```

## Consequences

**Positive**
- Swapping GPT ↔ Claude is a one-line env change; no chain or retrieval edits.
- Provider selection is unit-testable without API keys or network — tests assert
  the selected class and `model_id` without calling `build()`.
- A clean seam: if we ever drop LangChain, only the `build()` methods change; the
  rest of the app depends on our `LLMProvider`/`get_llm()` boundary.
- Unknown configuration fails loudly at startup rather than silently using OpenAI.

**Negative / costs**
- One thin extra layer over what LangChain already provides — a deliberate trade
  of a little indirection for an explicit, owned boundary.
- Each provider's heavy SDK import is deferred into `build()` to keep the module
  import-light; provider classes themselves stay dependency-free.

## Alternatives considered

- **Bare factory returning `BaseChatModel`** (the previous state): fewer lines, but
  leaks LangChain across the codebase and offers no place to attach provider
  metadata (cost, model id) or fail-fast validation.
- **Runtime per-request provider selection**: unnecessary for current needs; the
  provider is a deploy-time choice. Easy to add later behind the same interface.
