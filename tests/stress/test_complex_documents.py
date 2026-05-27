"""Stress tests for complex document types.

These tests validate that the ingestion pipeline handles edge cases without
exceptions and that the grounding check passes for known factual content.

Run after seeding the test Qdrant collection:
    pytest tests/stress/ -v
"""
from __future__ import annotations

import os
from pathlib import Path

import pytest

QDRANT_URL = os.getenv("QDRANT_URL", "http://localhost:6333")
TEST_COLLECTION = "docustream_stress_test"

pytestmark = pytest.mark.skipif(
    not os.getenv("RUN_STRESS_TESTS"),
    reason="Set RUN_STRESS_TESTS=1 to run stress tests (they need real Qdrant + models)",
)


@pytest.fixture(scope="module")
def stress_store():
    from ingestion.vector_store import VectorStore

    store = VectorStore(url=QDRANT_URL, collection_name=TEST_COLLECTION, vector_size=1024)
    try:
        store.client.delete_collection(TEST_COLLECTION)
    except Exception:
        pass
    store.ensure_collection()
    yield store
    try:
        store.client.delete_collection(TEST_COLLECTION)
    except Exception:
        pass


@pytest.fixture(scope="module")
def embedder():
    from ingestion.embedder import Embedder

    return Embedder(model_name="BAAI/bge-large-en-v1.5", device="cpu", batch_size=8)


def _ingest_fixture(name: str, stress_store, embedder) -> int:
    """Ingest a fixture PDF and return the number of chunks created."""
    from ingestion.chunker import split_documents
    from ingestion.connectors.pdf_connector import PdfConnector

    fixture_path = Path(__file__).parent / "fixtures"
    pdf_path = fixture_path / name

    if not pdf_path.exists():
        pytest.skip(f"Fixture file not found: {pdf_path}")

    connector = PdfConnector(fixture_path)
    documents = [d for d in connector.load() if name.split(".")[0] in d.metadata["source"]]
    assert len(documents) > 0, f"No pages loaded from {name}"

    chunks = split_documents(documents)
    embeddings = embedder.embed_documents(chunks)
    stress_store.upsert_chunks(chunks, embeddings)
    return len(chunks)


def test_table_heavy_pdf_ingests_without_error(stress_store, embedder):
    chunk_count = _ingest_fixture("table_heavy.pdf", stress_store, embedder)
    assert chunk_count > 0


def test_mixed_format_pdf_ingests_without_error(stress_store, embedder):
    chunk_count = _ingest_fixture("mixed_format.pdf", stress_store, embedder)
    assert chunk_count > 0


def test_grounding_check(stress_store, embedder):
    """Verify that a known fact from the ingested document can be retrieved.

    This is the key correctness test: ingest a document with a known fact,
    query for that fact, and assert the retrieved chunk contains it.
    The fact must be present in the fixture PDF as literal text.
    """
    known_fact = "The 2024 budget is $4.2M"
    known_fixture = "table_heavy.pdf"

    _ingest_fixture(known_fixture, stress_store, embedder)

    query_vector = embedder.embed_query(known_fact)
    results = stress_store.client.search(
        collection_name=TEST_COLLECTION,
        query_vector=query_vector,
        limit=5,
        with_payload=True,
    )

    assert len(results) > 0, "Should retrieve at least one result for a known fact"
    top_text = results[0].payload.get("text", "")
    assert "$4.2" in top_text or "4.2M" in top_text or "4,200" in top_text, (
        f"Top result should contain the known fact. Got: {top_text[:200]}"
    )
