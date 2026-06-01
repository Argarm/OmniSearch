from __future__ import annotations

import os
from datetime import UTC, datetime

from langchain_core.documents import Document

from ingestion.connectors.base import BaseConnector


class ConfluenceConnector(BaseConnector):
    """Loads pages from a Confluence space using atlassian-python-api.

    HTML content is converted to clean Markdown (preserving table structure)
    before being returned as Documents, ensuring chunking operates on
    human-readable text rather than raw HTML.
    """

    def __init__(
        self,
        url: str | None = None,
        user_email: str | None = None,
        token: str | None = None,
        space_key: str | None = None,
    ) -> None:
        self.url = url or os.environ["CONFLUENCE_URL"]
        self.user_email = user_email or os.environ["CONFLUENCE_USER_EMAIL"]
        self.token = token or os.environ["CONFLUENCE_TOKEN"]
        self.space_key = space_key or os.environ["CONFLUENCE_SPACE_KEY"]

    def load(self) -> list[Document]:
        try:
            from atlassian import Confluence
        except ImportError as e:
            raise ImportError(
                "atlassian-python-api is required: pip install atlassian-python-api"
            ) from e

        try:
            from bs4 import BeautifulSoup
        except ImportError as e:
            raise ImportError("beautifulsoup4 is required: pip install beautifulsoup4") from e

        self._confluence = Confluence(
            url=self.url,
            username=self.user_email,
            password=self.token,
            cloud=True,
        )
        self._bs4 = BeautifulSoup

        documents: list[Document] = []
        start = 0
        limit = 50

        while True:
            pages = self._confluence.get_all_pages_from_space(
                space=self.space_key,
                start=start,
                limit=limit,
                expand="body.storage,version,history.lastUpdated",
            )
            if not pages:
                break

            for page in pages:
                doc = self._page_to_document(page)
                if doc:
                    documents.append(doc)

            if len(pages) < limit:
                break
            start += limit

        return documents

    def _page_to_document(self, page: dict) -> Document | None:
        html = page.get("body", {}).get("storage", {}).get("value", "")
        if not html.strip():
            return None

        text = self._html_to_markdown(html)
        if not text.strip():
            return None

        page_id = page["id"]
        title = page.get("title", page_id)
        url = f"{self.url.rstrip('/')}/wiki/spaces/{self.space_key}/pages/{page_id}"
        last_modified = (
            page.get("history", {})
            .get("lastUpdated", {})
            .get("when", datetime.now(UTC).isoformat())
        )

        return Document(
            page_content=text,
            metadata={
                "source": page_id,
                "source_type": "confluence",
                "title": title,
                "url": url,
                "last_modified": last_modified,
            },
        )

    def _html_to_markdown(self, html: str) -> str:
        soup = self._bs4(html, "html.parser")

        # Convert tables to Markdown before stripping HTML
        for table in soup.find_all("table"):
            md_table = self._table_to_markdown(table)
            table.replace_with(soup.new_string(f"\n{md_table}\n"))

        # Preserve heading levels
        for level in range(1, 7):
            for tag in soup.find_all(f"h{level}"):
                tag.replace_with(soup.new_string(f"\n{'#' * level} {tag.get_text()}\n"))

        # Preserve list items
        for tag in soup.find_all("li"):
            tag.replace_with(soup.new_string(f"\n- {tag.get_text()}"))

        # Preserve code blocks
        for tag in soup.find_all("code"):
            tag.replace_with(soup.new_string(f"`{tag.get_text()}`"))

        # Strip remaining HTML tags and normalize whitespace
        text = soup.get_text(separator="\n")
        lines = [line.strip() for line in text.splitlines()]
        # Collapse more than two consecutive blank lines
        result: list[str] = []
        blank_count = 0
        for line in lines:
            if not line:
                blank_count += 1
                if blank_count <= 2:
                    result.append("")
            else:
                blank_count = 0
                result.append(line)

        return "\n".join(result).strip()

    @staticmethod
    def _table_to_markdown(table) -> str:
        rows: list[list[str]] = []
        for tr in table.find_all("tr"):
            cells = [td.get_text(strip=True) for td in tr.find_all(["th", "td"])]
            if cells:
                rows.append(cells)

        if not rows:
            return ""

        col_count = max(len(row) for row in rows)
        # Pad rows to equal width
        padded = [row + [""] * (col_count - len(row)) for row in rows]

        header = "| " + " | ".join(padded[0]) + " |"
        separator = "| " + " | ".join(["---"] * col_count) + " |"
        body = "\n".join("| " + " | ".join(row) + " |" for row in padded[1:])
        return "\n".join(filter(None, [header, separator, body]))
