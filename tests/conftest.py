"""Shared pytest fixtures for DocuStream tests."""
from __future__ import annotations

from pathlib import Path
from unittest.mock import MagicMock

import pytest
from langchain_core.documents import Document


@pytest.fixture
def sample_documents() -> list[Document]:
    return [
        Document(
            page_content="The 2024 annual budget is $4.2 million. This includes R&D costs of $1.5M.",
            metadata={
                "source": "/data/budget_2024.pdf",
                "source_type": "pdf",
                "title": "Budget Report 2024",
                "page": 3,
                "url": "file:///data/budget_2024.pdf",
                "last_modified": "2024-01-15T10:00:00+00:00",
            },
        ),
        Document(
            page_content="Engineering onboarding takes 4 weeks and includes a buddy system pairing.",
            metadata={
                "source": "abc123",
                "source_type": "notion",
                "title": "Engineering Onboarding Guide",
                "url": "https://notion.so/abc123",
                "last_modified": "2024-03-01T08:00:00+00:00",
            },
        ),
    ]


@pytest.fixture
def mock_qdrant_client():
    client = MagicMock()
    client.get_collections.return_value = MagicMock(collections=[])
    client.upsert.return_value = None
    client.search.return_value = []
    client.delete.return_value = None
    return client


@pytest.fixture
def fixtures_dir() -> Path:
    return Path(__file__).parent / "stress" / "fixtures"
