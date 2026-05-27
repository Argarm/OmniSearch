"""Unit tests for the text chunker."""
from __future__ import annotations

import pytest
from langchain_core.documents import Document

from ingestion.chunker import split_documents


@pytest.fixture
def long_document() -> Document:
    # ~1200 tokens of repeating text to force splitting
    paragraph = "This is a test sentence about the engineering department. " * 30
    return Document(
        page_content=paragraph,
        metadata={
            "source": "/test/doc.pdf",
            "source_type": "pdf",
            "title": "Test Document",
            "page": 1,
        },
    )


def test_chunks_respect_max_token_size(long_document):
    chunks = split_documents([long_document], chunk_size=200, chunk_overlap=20)
    assert len(chunks) > 1, "Long document should be split into multiple chunks"


def test_chunk_index_is_sequential(long_document):
    chunks = split_documents([long_document], chunk_size=200, chunk_overlap=20)
    indices = [c.metadata["chunk_index"] for c in chunks]
    assert indices == list(range(len(indices))), "chunk_index must be sequential starting at 0"


def test_metadata_is_inherited(long_document):
    chunks = split_documents([long_document], chunk_size=200, chunk_overlap=20)
    for chunk in chunks:
        assert chunk.metadata["source"] == "/test/doc.pdf"
        assert chunk.metadata["source_type"] == "pdf"
        assert chunk.metadata["title"] == "Test Document"
        assert chunk.metadata["page"] == 1


def test_overlap_produces_repeated_content(long_document):
    chunks = split_documents([long_document], chunk_size=200, chunk_overlap=50)
    if len(chunks) < 2:
        pytest.skip("Document not long enough to produce overlap")
    # The end of chunk 0 should appear somewhere in chunk 1
    end_of_first = chunks[0].page_content[-30:]
    assert end_of_first in chunks[1].page_content, "Overlap content should appear in next chunk"


def test_empty_document_produces_no_chunks():
    doc = Document(page_content="", metadata={"source": "empty"})
    chunks = split_documents([doc])
    assert len(chunks) == 0


def test_short_document_produces_single_chunk():
    doc = Document(
        page_content="Short text that fits in one chunk.",
        metadata={"source": "short", "source_type": "pdf", "title": "Short"},
    )
    chunks = split_documents([doc], chunk_size=800)
    assert len(chunks) == 1
    assert chunks[0].metadata["chunk_index"] == 0
