"""Unit tests for RAG prompt formatting."""
from __future__ import annotations

from backend.rag.prompts import format_context


def test_format_context_includes_chunk_numbers():
    sources = [
        {"title": "Budget 2024", "page": 3, "url": None, "text": "Budget is $4.2M."},
    ]
    result = format_context(sources)
    assert "[Chunk 1]" in result
    assert "Budget 2024" in result
    assert "Page: 3" in result
    assert "Budget is $4.2M." in result


def test_format_context_includes_url_when_no_page():
    sources = [
        {
            "title": "Onboarding Guide",
            "page": None,
            "url": "https://notion.so/abc",
            "text": "Onboarding takes 4 weeks.",
        }
    ]
    result = format_context(sources)
    assert "https://notion.so/abc" in result
    assert "Onboarding takes 4 weeks." in result


def test_format_context_multiple_chunks_numbered():
    sources = [
        {"title": "Doc A", "page": 1, "url": None, "text": "Content A"},
        {"title": "Doc B", "page": 2, "url": None, "text": "Content B"},
    ]
    result = format_context(sources)
    assert "[Chunk 1]" in result
    assert "[Chunk 2]" in result


def test_format_context_empty_returns_empty():
    result = format_context([])
    assert result == ""
