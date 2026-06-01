"""Integration tests for the ingestion pipeline against a real Qdrant instance.

These tests require QDRANT_URL to be set (defaults to http://localhost:6333).
They are designed to run in CI with a Qdrant service container.
"""
from __future__ import annotations

import os

import pytest

QDRANT_URL = os.getenv("QDRANT_URL", "http://localhost:6333")
TEST_COLLECTION = "omnisearch_test"


@pytest.fixture(scope="module")
def vector_store():
    """Create and clean up a test Qdrant collection."""
    from ingestion.vector_store import VectorStore

    store = VectorStore(
        url=QDRANT_URL,
        collection_name=TEST_COLLECTION,
        vector_size=1024,
    )

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


def test_ensure_collection_creates_it(vector_store):
    from qdrant_client.http.models import CollectionStatus

    info = vector_store.client.get_collection(TEST_COLLECTION)
    assert info.status == CollectionStatus.GREEN or info.points_count >= 0


def test_upsert_and_count(vector_store, embedder, sample_documents):
    from ingestion.chunker import split_documents

    chunks = split_documents(sample_documents, chunk_size=200, chunk_overlap=20)
    embeddings = embedder.embed_documents(chunks)

    vector_store.upsert_chunks(chunks, embeddings)

    info = vector_store.client.get_collection(TEST_COLLECTION)
    assert info.points_count == len(chunks), (
        f"Expected {len(chunks)} vectors, got {info.points_count}"
    )


def test_upsert_is_idempotent(vector_store, embedder, sample_documents):
    """Re-ingesting the same documents should not increase the vector count."""
    from ingestion.chunker import split_documents

    chunks = split_documents(sample_documents, chunk_size=200, chunk_overlap=20)
    embeddings = embedder.embed_documents(chunks)

    count_before = vector_store.client.get_collection(TEST_COLLECTION).points_count
    vector_store.upsert_chunks(chunks, embeddings)
    count_after = vector_store.client.get_collection(TEST_COLLECTION).points_count

    assert count_before == count_after, "Re-ingesting same docs should not create duplicates"


def test_delete_by_source(vector_store, sample_documents):
    source_to_delete = sample_documents[0].metadata["source"]
    vector_store.delete_by_source(source_to_delete)

    results = vector_store.client.scroll(
        collection_name=TEST_COLLECTION,
        scroll_filter={
            "must": [{"key": "source", "match": {"value": source_to_delete}}]
        },
        limit=10,
    )
    assert len(results[0]) == 0, "All points for the deleted source should be gone"
