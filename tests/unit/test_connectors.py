"""Unit tests for data source connectors using API mocks."""
from __future__ import annotations

from pathlib import Path
from unittest.mock import MagicMock, patch

import pytest
from langchain_core.documents import Document


class TestPdfConnector:
    def test_raises_on_missing_directory(self):
        from ingestion.connectors.pdf_connector import PdfConnector

        with pytest.raises(FileNotFoundError):
            PdfConnector("/nonexistent/path/to/pdfs")

    def test_loads_pages_with_correct_metadata(self, tmp_path):
        from ingestion.connectors.pdf_connector import PdfConnector

        # Create a minimal PDF using PyMuPDF for the test
        try:
            import fitz
        except ImportError:
            pytest.skip("PyMuPDF not installed")

        pdf_path = tmp_path / "test.pdf"
        doc = fitz.open()
        page = doc.new_page()
        page.insert_text((50, 50), "Hello world test content for OmniSearch.")
        doc.save(str(pdf_path))
        doc.close()

        connector = PdfConnector(tmp_path)
        documents = connector.load()

        assert len(documents) >= 1
        first = documents[0]
        assert first.metadata["source_type"] == "pdf"
        assert first.metadata["page"] == 1
        assert "Hello world" in first.page_content

    def test_skips_empty_pages(self, tmp_path):
        from ingestion.connectors.pdf_connector import PdfConnector

        try:
            import fitz
        except ImportError:
            pytest.skip("PyMuPDF not installed")

        pdf_path = tmp_path / "empty.pdf"
        doc = fitz.open()
        doc.new_page()  # Empty page — no text
        doc.save(str(pdf_path))
        doc.close()

        connector = PdfConnector(tmp_path)
        documents = connector.load()
        assert len(documents) == 0, "Empty pages should be skipped"


class TestNotionConnector:
    def test_metadata_shape(self):
        from ingestion.connectors.notion_connector import NotionConnector

        connector = NotionConnector(token="test_token", root_page_id="page123")

        mock_client = MagicMock()
        mock_client.blocks.retrieve.return_value = {
            "id": "page123",
            "type": "page",
            "url": "https://notion.so/page123",
            "last_edited_time": "2024-01-01T00:00:00.000Z",
            "properties": {
                "title": {"title": [{"plain_text": "Test Page"}]}
            },
        }
        mock_client.blocks.children.list.return_value = {
            "results": [
                {
                    "id": "block1",
                    "type": "paragraph",
                    "paragraph": {
                        "rich_text": [{"plain_text": "This is test content."}]
                    },
                }
            ],
            "has_more": False,
        }

        connector._client = mock_client
        docs = []
        connector._collect_pages("page123", docs)

        assert len(docs) == 1
        assert docs[0].metadata["source_type"] == "notion"
        assert docs[0].metadata["title"] == "Test Page"
        assert "test content" in docs[0].page_content

    def test_table_to_markdown(self):
        from ingestion.connectors.notion_connector import NotionConnector

        connector = NotionConnector.__new__(NotionConnector)

        mock_client = MagicMock()
        mock_client.blocks.children.list.return_value = {
            "results": [
                {
                    "id": "row1",
                    "type": "table_row",
                    "table_row": {
                        "cells": [
                            [{"plain_text": "Name"}],
                            [{"plain_text": "Value"}],
                        ]
                    },
                },
                {
                    "id": "row2",
                    "type": "table_row",
                    "table_row": {
                        "cells": [
                            [{"plain_text": "Budget"}],
                            [{"plain_text": "$4.2M"}],
                        ]
                    },
                },
            ],
            "has_more": False,
        }
        connector._client = mock_client

        result = connector._table_to_markdown("table_id")
        assert "| Name | Value |" in result
        assert "| Budget | $4.2M |" in result
        assert "---" in result


class TestConfluenceConnector:
    def test_html_to_markdown_table(self):
        from ingestion.connectors.confluence_connector import ConfluenceConnector

        connector = ConfluenceConnector.__new__(ConfluenceConnector)

        try:
            from bs4 import BeautifulSoup
            connector._bs4 = BeautifulSoup
        except ImportError:
            pytest.skip("beautifulsoup4 not installed")

        html = """
        <table>
          <tr><th>Column A</th><th>Column B</th></tr>
          <tr><td>Value 1</td><td>Value 2</td></tr>
        </table>
        """
        result = connector._html_to_markdown(html)
        assert "| Column A | Column B |" in result
        assert "| Value 1 | Value 2 |" in result

    def test_html_strips_tags(self):
        from ingestion.connectors.confluence_connector import ConfluenceConnector

        connector = ConfluenceConnector.__new__(ConfluenceConnector)

        try:
            from bs4 import BeautifulSoup
            connector._bs4 = BeautifulSoup
        except ImportError:
            pytest.skip("beautifulsoup4 not installed")

        html = "<p>Hello <strong>world</strong></p>"
        result = connector._html_to_markdown(html)
        assert "<" not in result
        assert "Hello" in result
        assert "world" in result
