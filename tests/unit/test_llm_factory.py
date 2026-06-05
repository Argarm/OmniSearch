"""Unit tests for the LLM provider abstraction.

These tests verify provider *selection* and config resolution only. They never
call `.build()`, so they require neither API keys nor the provider SDKs to be
installed.
"""
from __future__ import annotations

import pytest

from backend.llm_factory import (
    AnthropicProvider,
    LLMProvider,
    OpenAIProvider,
    get_provider,
)


def test_defaults_to_openai(monkeypatch):
    monkeypatch.delenv("LLM_PROVIDER", raising=False)
    provider = get_provider()
    assert isinstance(provider, OpenAIProvider)
    assert isinstance(provider, LLMProvider)


def test_selects_openai_explicitly(monkeypatch):
    monkeypatch.setenv("LLM_PROVIDER", "openai")
    monkeypatch.setenv("OPENAI_MODEL", "gpt-4o")
    provider = get_provider()
    assert isinstance(provider, OpenAIProvider)
    assert provider.model_id == "gpt-4o"


def test_selects_anthropic(monkeypatch):
    monkeypatch.setenv("LLM_PROVIDER", "Anthropic")  # case-insensitive
    provider = get_provider()
    assert isinstance(provider, AnthropicProvider)
    assert provider.model_id.startswith("claude")


def test_anthropic_model_override(monkeypatch):
    monkeypatch.setenv("LLM_PROVIDER", "anthropic")
    monkeypatch.setenv("ANTHROPIC_MODEL", "claude-3-5-sonnet-20241022")
    assert get_provider().model_id == "claude-3-5-sonnet-20241022"


def test_unknown_provider_raises(monkeypatch):
    monkeypatch.setenv("LLM_PROVIDER", "bogus")
    with pytest.raises(ValueError, match="Unknown LLM_PROVIDER"):
        get_provider()
