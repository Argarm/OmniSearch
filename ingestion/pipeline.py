"""OmniSearch ingestion pipeline entry point.

Usage:
    python -m ingestion.pipeline --source all
    python -m ingestion.pipeline --source pdf
    python -m ingestion.pipeline --source notion
    python -m ingestion.pipeline --source confluence
"""
from __future__ import annotations

import argparse
import os
from pathlib import Path

from langchain_core.documents import Document

from ingestion.chunker import split_documents
from ingestion.connectors.confluence_connector import ConfluenceConnector
from ingestion.connectors.notion_connector import NotionConnector
from ingestion.connectors.pdf_connector import PdfConnector
from ingestion.embedder import Embedder
from ingestion.vector_store import VectorStore


def _load_documents(source_type: str) -> list[Document]:
    documents: list[Document] = []

    if source_type in ("pdf", "all"):
        pdf_dir = Path(os.getenv("PDF_SOURCE_DIR", "data/sources"))
        if pdf_dir.exists():
            print(f"[Pipeline] Loading PDFs from {pdf_dir}")
            docs = PdfConnector(pdf_dir).load()
            print(f"[Pipeline] Loaded {len(docs)} PDF pages")
            documents.extend(docs)

    if source_type in ("notion", "all"):
        token = os.getenv("NOTION_TOKEN", "")
        root_id = os.getenv("NOTION_ROOT_PAGE_ID", "")
        if token and root_id:
            print("[Pipeline] Loading Notion pages")
            docs = NotionConnector(token=token, root_page_id=root_id).load()
            print(f"[Pipeline] Loaded {len(docs)} Notion pages")
            documents.extend(docs)
        elif source_type == "notion":
            print("[Pipeline] NOTION_TOKEN or NOTION_ROOT_PAGE_ID not set — skipping")

    if source_type in ("confluence", "all"):
        required = [
            "CONFLUENCE_URL",
            "CONFLUENCE_USER_EMAIL",
            "CONFLUENCE_TOKEN",
            "CONFLUENCE_SPACE_KEY",
        ]
        if all(os.getenv(k) for k in required):
            print("[Pipeline] Loading Confluence pages")
            docs = ConfluenceConnector().load()
            print(f"[Pipeline] Loaded {len(docs)} Confluence pages")
            documents.extend(docs)
        elif source_type == "confluence":
            print("[Pipeline] Confluence env vars not set — skipping")

    return documents


def run_pipeline(
    source_type: str = "all",
    qdrant_url: str | None = None,
    qdrant_api_key: str | None = None,
    collection_name: str | None = None,
) -> None:
    # Configuration from env with argument overrides
    qdrant_url = qdrant_url or os.getenv("QDRANT_URL", "http://localhost:6333")
    qdrant_api_key = qdrant_api_key or os.getenv("QDRANT_API_KEY", "")
    collection_name = collection_name or os.getenv("QDRANT_COLLECTION_NAME", "omnisearch")

    chunk_size = int(os.getenv("CHUNKING_CHUNK_SIZE", "800"))
    chunk_overlap = int(os.getenv("CHUNKING_CHUNK_OVERLAP", "150"))
    embed_model = os.getenv("EMBEDDING_MODEL_NAME", "BAAI/bge-large-en-v1.5")
    embed_device = os.getenv("EMBEDDING_DEVICE", "cpu")
    embed_batch = int(os.getenv("EMBEDDING_BATCH_SIZE", "32"))

    # 1. Load
    documents = _load_documents(source_type)
    if not documents:
        print("[Pipeline] No documents loaded — nothing to ingest.")
        return

    print(f"[Pipeline] Total documents loaded: {len(documents)}")

    # 2. Chunk
    chunks = split_documents(documents, chunk_size=chunk_size, chunk_overlap=chunk_overlap)
    print(f"[Pipeline] Total chunks after splitting: {len(chunks)}")

    # 3. Embed
    embedder = Embedder(
        model_name=embed_model,
        device=embed_device,
        batch_size=embed_batch,
    )
    print(f"[Pipeline] Generating embeddings with {embed_model} on {embed_device}...")
    embeddings = embedder.embed_documents(chunks)
    print(f"[Pipeline] Generated {len(embeddings)} embeddings (dim={len(embeddings[0])})")

    # 4. Upsert
    store = VectorStore(
        url=qdrant_url,
        api_key=qdrant_api_key,
        collection_name=collection_name,
    )
    store.ensure_collection()
    store.upsert_chunks(chunks, embeddings)

    info = store.collection_info()
    print(f"[Pipeline] Collection '{info['name']}' now has {info['vectors_count']} vectors.")
    print("[Pipeline] Ingestion complete.")


def main() -> None:
    parser = argparse.ArgumentParser(description="OmniSearch ingestion pipeline")
    parser.add_argument(
        "--source",
        choices=["all", "pdf", "notion", "confluence"],
        default="all",
        help="Data source to ingest (default: all)",
    )
    args = parser.parse_args()
    run_pipeline(source_type=args.source)


if __name__ == "__main__":
    main()
