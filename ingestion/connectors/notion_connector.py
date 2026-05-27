from __future__ import annotations

import os
from datetime import datetime, timezone

from langchain_core.documents import Document

from ingestion.connectors.base import BaseConnector


class NotionConnector(BaseConnector):
    """Loads pages recursively from a Notion workspace.

    Uses the official notion-client SDK. Traverses from a root page/database ID
    and flattens all block content (paragraphs, headings, lists, code, tables)
    into plain text with Markdown table formatting preserved.
    """

    BLOCK_TEXT_TYPES = {
        "paragraph",
        "heading_1",
        "heading_2",
        "heading_3",
        "bulleted_list_item",
        "numbered_list_item",
        "toggle",
        "quote",
        "callout",
        "code",
    }

    def __init__(
        self,
        token: str | None = None,
        root_page_id: str | None = None,
    ) -> None:
        self.token = token or os.environ["NOTION_TOKEN"]
        self.root_page_id = root_page_id or os.environ["NOTION_ROOT_PAGE_ID"]

    def load(self) -> list[Document]:
        try:
            from notion_client import Client
        except ImportError as e:
            raise ImportError("notion-client is required: pip install notion-client") from e

        self._client = Client(auth=self.token)
        documents: list[Document] = []
        self._collect_pages(self.root_page_id, documents)
        return documents

    def _collect_pages(self, block_id: str, documents: list[Document]) -> None:
        try:
            block = self._client.blocks.retrieve(block_id)  # type: ignore[attr-defined]
        except Exception:
            return

        block_type = block.get("type", "")

        # If this is a database, iterate its child pages
        if block_type == "child_database":
            self._collect_database_pages(block_id, documents)
            return

        # Otherwise treat it as a page
        page_text, page_url, page_title, last_modified = self._extract_page(block_id, block)
        if page_text.strip():
            documents.append(
                Document(
                    page_content=page_text,
                    metadata={
                        "source": block_id,
                        "source_type": "notion",
                        "title": page_title,
                        "url": page_url,
                        "last_modified": last_modified,
                    },
                )
            )

        # Recurse into child pages/databases
        for child in self._paginate_children(block_id):
            child_type = child.get("type", "")
            child_id = child["id"]
            if child_type in ("child_page", "child_database"):
                self._collect_pages(child_id, documents)

    def _collect_database_pages(self, db_id: str, documents: list[Document]) -> None:
        has_more = True
        cursor = None
        while has_more:
            resp = self._client.databases.query(  # type: ignore[attr-defined]
                database_id=db_id,
                **{"start_cursor": cursor} if cursor else {},
            )
            for page in resp.get("results", []):
                self._collect_pages(page["id"], documents)
            has_more = resp.get("has_more", False)
            cursor = resp.get("next_cursor")

    def _extract_page(
        self, block_id: str, block: dict
    ) -> tuple[str, str, str, str]:
        props = block.get("properties", {})
        title = self._extract_title_from_properties(props) or block_id
        url = block.get("url", f"https://notion.so/{block_id.replace('-', '')}")
        last_edited = block.get("last_edited_time", datetime.now(timezone.utc).isoformat())

        text_parts: list[str] = []
        for child in self._paginate_children(block_id):
            part = self._block_to_text(child)
            if part:
                text_parts.append(part)

        return "\n".join(text_parts), url, title, last_edited

    def _block_to_text(self, block: dict) -> str:
        block_type = block.get("type", "")
        content = block.get(block_type, {})

        if block_type in self.BLOCK_TEXT_TYPES:
            rich_texts = content.get("rich_text", [])
            text = self._rich_text_to_plain(rich_texts)
            if block_type == "heading_1":
                return f"# {text}"
            if block_type == "heading_2":
                return f"## {text}"
            if block_type == "heading_3":
                return f"### {text}"
            if block_type == "code":
                lang = content.get("language", "")
                return f"```{lang}\n{text}\n```"
            return text

        if block_type == "table":
            return self._table_to_markdown(block["id"])

        return ""

    def _table_to_markdown(self, table_id: str) -> str:
        rows: list[list[str]] = []
        for row_block in self._paginate_children(table_id):
            if row_block.get("type") != "table_row":
                continue
            cells = row_block["table_row"].get("cells", [])
            rows.append([self._rich_text_to_plain(cell) for cell in cells])

        if not rows:
            return ""

        header = "| " + " | ".join(rows[0]) + " |"
        separator = "| " + " | ".join(["---"] * len(rows[0])) + " |"
        body = "\n".join("| " + " | ".join(row) + " |" for row in rows[1:])
        return "\n".join(filter(None, [header, separator, body]))

    def _paginate_children(self, block_id: str) -> list[dict]:
        results: list[dict] = []
        has_more = True
        cursor = None
        while has_more:
            resp = self._client.blocks.children.list(  # type: ignore[attr-defined]
                block_id=block_id,
                **{"start_cursor": cursor} if cursor else {},
            )
            results.extend(resp.get("results", []))
            has_more = resp.get("has_more", False)
            cursor = resp.get("next_cursor")
        return results

    @staticmethod
    def _rich_text_to_plain(rich_texts: list[dict]) -> str:
        return "".join(rt.get("plain_text", "") for rt in rich_texts)

    @staticmethod
    def _extract_title_from_properties(props: dict) -> str:
        for key in ("title", "Name", "Title"):
            if key in props:
                rich_texts = props[key].get("title", [])
                text = "".join(rt.get("plain_text", "") for rt in rich_texts)
                if text:
                    return text
        return ""
