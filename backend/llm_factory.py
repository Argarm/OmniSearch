"""
LLM provider abstraction.

A small `LLMProvider` interface wraps each supported chat backend so the rest of
the application depends on our own boundary rather than directly on LangChain's
`BaseChatModel`. Swapping providers requires only an env var change
(`LLM_PROVIDER`), with no modifications to the chain or retrieval logic.

Supported providers:
  - "openai"    → OpenAIProvider     (ChatOpenAI; OpenAI-compatible, works with Ollama)
  - "anthropic" → AnthropicProvider  (ChatAnthropic; Claude models)

See docs/adr/0001-provider-abstraction.md for the rationale.
"""
from __future__ import annotations

import os
from abc import ABC, abstractmethod

from langchain_core.language_models import BaseChatModel


class LLMProvider(ABC):
    """Common interface for swappable chat LLM backends.

    Implementations read their own config from the environment so the selection
    layer stays provider-agnostic.
    """

    #: Stable key used for `LLM_PROVIDER` selection.
    name: str

    @property
    @abstractmethod
    def model_id(self) -> str:
        """The concrete model identifier this provider will instantiate."""

    @abstractmethod
    def build(self) -> BaseChatModel:
        """Construct the underlying LangChain chat model."""

    def __repr__(self) -> str:  # pragma: no cover - debugging aid
        return f"{type(self).__name__}(model_id={self.model_id!r})"


class OpenAIProvider(LLMProvider):
    """OpenAI (and OpenAI-compatible, e.g. Ollama via OPENAI_BASE_URL) backend."""

    name = "openai"

    @property
    def model_id(self) -> str:
        return os.getenv("OPENAI_MODEL", "gpt-4o-mini")

    def build(self) -> BaseChatModel:
        from langchain_openai import ChatOpenAI

        return ChatOpenAI(
            model=self.model_id,
            api_key=os.getenv("OPENAI_API_KEY"),
            base_url=os.getenv("OPENAI_BASE_URL") or None,
            max_tokens=int(os.getenv("LLM_MAX_TOKENS", "1024")),
            streaming=True,
        )


class AnthropicProvider(LLMProvider):
    """Anthropic Claude backend."""

    name = "anthropic"

    @property
    def model_id(self) -> str:
        return os.getenv("ANTHROPIC_MODEL", "claude-3-5-haiku-20241022")

    def build(self) -> BaseChatModel:
        from langchain_anthropic import ChatAnthropic

        return ChatAnthropic(
            model=self.model_id,
            api_key=os.getenv("ANTHROPIC_API_KEY"),
            max_tokens=int(os.getenv("LLM_MAX_TOKENS", "1024")),
            streaming=True,
        )


_PROVIDERS: dict[str, type[LLMProvider]] = {
    "openai": OpenAIProvider,
    "anthropic": AnthropicProvider,
}


def get_provider() -> LLMProvider:
    """Resolve the configured provider from the `LLM_PROVIDER` env var.

    Raises:
        ValueError: if `LLM_PROVIDER` is set to an unknown value.
    """
    key = os.getenv("LLM_PROVIDER", "openai").lower()
    try:
        return _PROVIDERS[key]()
    except KeyError:
        supported = ", ".join(sorted(_PROVIDERS))
        raise ValueError(
            f"Unknown LLM_PROVIDER={key!r}. Supported providers: {supported}."
        ) from None


def get_llm() -> BaseChatModel:
    """Build the chat model for the configured provider.

    Kept as the public entry point so consumers (e.g. RAGChain) stay decoupled
    from the provider classes.
    """
    return get_provider().build()
