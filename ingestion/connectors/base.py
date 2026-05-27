from __future__ import annotations

from abc import ABC, abstractmethod
from typing import TypedDict

from langchain_core.documents import Document


class DocumentMetadata(TypedDict, total=False):
    source: str           # unique identifier (file path, page URL, etc.)
    source_type: str      # "pdf" | "notion" | "confluence"
    title: str
    page: int             # page number for PDFs
    url: str              # canonical URL for web-based sources
    last_modified: str    # ISO 8601 string


class BaseConnector(ABC):
    """Abstract base for all DocuStream data source connectors.

    Every connector returns a list of LangChain Documents with standardized
    metadata so downstream chunking and upsert logic stays source-agnostic.
    """

    @abstractmethod
    def load(self) -> list[Document]:
        """Load documents from the source and return them as LangChain Documents."""
        ...
