"""Integration tests for the FastAPI /query endpoint."""
from __future__ import annotations

from unittest.mock import AsyncMock, MagicMock, patch

import pytest
from fastapi.testclient import TestClient
from httpx import AsyncClient

from backend.main import create_app
from backend.models.schemas import SourceDocument


@pytest.fixture
def mock_rag_chain():
    chain = MagicMock()
    chain.ainvoke = AsyncMock(
        return_value=(
            "The 2024 budget is $4.2M [Source: Budget Report 2024, Page 3].",
            [
                SourceDocument(
                    source="/data/budget.pdf",
                    source_type="pdf",
                    title="Budget Report 2024",
                    page=3,
                    url=None,
                    chunk_text="The 2024 annual budget is $4.2 million.",
                    score=0.87,
                )
            ],
        )
    )

    async def mock_astream(*args, **kwargs):
        yield "The 2024 budget is $4.2M"
        yield [
            SourceDocument(
                source="/data/budget.pdf",
                source_type="pdf",
                title="Budget Report 2024",
                page=3,
                url=None,
                chunk_text="The 2024 annual budget is $4.2 million.",
                score=0.87,
            )
        ]

    chain.astream = mock_astream
    return chain


@pytest.fixture
def test_app(mock_rag_chain):
    app = create_app()
    app.state.rag_chain = mock_rag_chain
    return app


def test_non_streaming_query_returns_valid_schema(test_app):
    with TestClient(test_app) as client:
        resp = client.post(
            "/api/v1/query",
            json={"query": "What is the 2024 budget?", "stream": False},
        )
    assert resp.status_code == 200
    data = resp.json()
    assert "answer" in data
    assert "sources" in data
    assert "query_id" in data
    assert len(data["sources"]) == 1
    assert data["sources"][0]["source_type"] == "pdf"


def test_streaming_query_returns_sse_events(test_app):
    with TestClient(test_app) as client:
        with client.stream(
            "POST",
            "/api/v1/query",
            json={"query": "What is the 2024 budget?", "stream": True},
        ) as resp:
            assert resp.status_code == 200
            assert "text/event-stream" in resp.headers["content-type"]
            lines = list(resp.iter_lines())

    data_lines = [l for l in lines if l.startswith("data: ")]
    assert any("[SOURCES]" in l for l in data_lines), "SSE stream must include a [SOURCES] event"
    assert any("[DONE]" in l for l in data_lines), "SSE stream must end with [DONE]"


def test_health_endpoint(test_app):
    test_app.state.rag_chain.retriever = MagicMock()
    test_app.state.rag_chain.retriever.collection_name = "docustream"
    test_app.state.rag_chain.retriever.client.get_collection.side_effect = Exception("no qdrant")

    with TestClient(test_app) as client:
        resp = client.get("/health")
    assert resp.status_code == 200
    # Degraded is acceptable — health should not 500
    assert resp.json()["status"] in ("ok", "degraded") or "degraded" in resp.json()["status"]


def test_empty_query_rejected(test_app):
    with TestClient(test_app) as client:
        resp = client.post("/api/v1/query", json={"query": "", "stream": False})
    assert resp.status_code == 422
