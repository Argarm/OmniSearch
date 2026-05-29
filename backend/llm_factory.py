"""
LLM provider factory.

Supported providers (set via LLM_PROVIDER env var):
  - "openai"    → ChatOpenAI (default, OpenAI-compatible, works with Ollama too)
  - "anthropic" → ChatAnthropic (Claude models)

This abstraction means swapping providers requires only an env var change,
with no modifications to the chain or retrieval logic.
"""
import os

from langchain_core.language_models import BaseChatModel


def get_llm() -> BaseChatModel:
    provider = os.getenv("LLM_PROVIDER", "openai").lower()

    if provider == "anthropic":
        from langchain_anthropic import ChatAnthropic

        return ChatAnthropic(
            model=os.getenv("ANTHROPIC_MODEL", "claude-3-5-haiku-20241022"),
            api_key=os.getenv("ANTHROPIC_API_KEY"),
            max_tokens=int(os.getenv("LLM_MAX_TOKENS", "1024")),
            streaming=True,
        )
    else:  # default: openai
        from langchain_openai import ChatOpenAI

        return ChatOpenAI(
            model=os.getenv("OPENAI_MODEL", "gpt-4o-mini"),
            api_key=os.getenv("OPENAI_API_KEY"),
            base_url=os.getenv("OPENAI_BASE_URL") or None,
            max_tokens=int(os.getenv("LLM_MAX_TOKENS", "1024")),
            streaming=True,
        )
