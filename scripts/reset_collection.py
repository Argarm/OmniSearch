"""Utility: wipe and recreate the Qdrant collection.

Usage:
    python scripts/reset_collection.py

WARNING: This permanently deletes all indexed vectors. Use with caution.
"""
from __future__ import annotations

import os
import sys

QDRANT_URL = os.getenv("QDRANT_URL", "http://localhost:6333")
COLLECTION = os.getenv("QDRANT_COLLECTION_NAME", "docustream")


def main() -> None:
    try:
        from qdrant_client import QdrantClient
    except ImportError:
        print("qdrant-client is required: pip install qdrant-client")
        sys.exit(1)

    confirm = input(
        f"This will DELETE all vectors in collection '{COLLECTION}' at {QDRANT_URL}.\n"
        "Type 'yes' to confirm: "
    )
    if confirm.strip().lower() != "yes":
        print("Aborted.")
        sys.exit(0)

    client = QdrantClient(url=QDRANT_URL, api_key=os.getenv("QDRANT_API_KEY") or None)

    try:
        client.delete_collection(COLLECTION)
        print(f"Deleted collection '{COLLECTION}'")
    except Exception as exc:
        print(f"Collection did not exist or could not be deleted: {exc}")

    from ingestion.vector_store import VectorStore

    store = VectorStore(url=QDRANT_URL, collection_name=COLLECTION)
    store.ensure_collection()
    print(f"Recreated empty collection '{COLLECTION}'")


if __name__ == "__main__":
    main()
