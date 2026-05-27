from __future__ import annotations

import hashlib
import uuid
from typing import Any

from langchain_core.documents import Document


def _deterministic_uuid(source: str, chunk_index: int) -> str:
    """Generate a stable UUID from source + chunk_index for idempotent upserts."""
    key = f"{source}::{chunk_index}"
    digest = hashlib.sha256(key.encode()).hexdigest()
    # Format as a valid UUID (use first 32 hex chars)
    return str(uuid.UUID(digest[:32]))


class VectorStore:
    """Manages the Qdrant collection for OmniSearch.

    Uses deterministic point IDs so re-ingesting the same document overwrites
    existing vectors rather than duplicating them (idempotent upserts).
    """

    def __init__(
        self,
        url: str = "http://localhost:6333",
        api_key: str = "",
        collection_name: str = "omnisearch",
        vector_size: int = 1024,
        distance: str = "Cosine",
        on_disk_payload: bool = True,
    ) -> None:
        self.url = url
        self.api_key = api_key or None
        self.collection_name = collection_name
        self.vector_size = vector_size
        self.distance = distance
        self.on_disk_payload = on_disk_payload
        self._client = None

    @property
    def client(self):
        if self._client is None:
            try:
                from qdrant_client import QdrantClient
            except ImportError as e:
                raise ImportError(
                    "qdrant-client is required: pip install qdrant-client"
                ) from e
            self._client = QdrantClient(
                url=self.url,
                api_key=self.api_key,
                timeout=30,
            )
        return self._client

    def ensure_collection(self) -> None:
        """Create the collection if it does not exist; no-op if it already does."""
        from qdrant_client.http.models import Distance, VectorParams

        _distance_map = {
            "Cosine": Distance.COSINE,
            "Dot": Distance.DOT,
            "Euclidean": Distance.EUCLID,
        }
        distance = _distance_map.get(self.distance, Distance.COSINE)

        existing = [c.name for c in self.client.get_collections().collections]
        if self.collection_name not in existing:
            self.client.create_collection(
                collection_name=self.collection_name,
                vectors_config=VectorParams(size=self.vector_size, distance=distance),
                on_disk_payload=self.on_disk_payload,
            )
            print(f"[VectorStore] Created collection '{self.collection_name}'")

    def upsert_chunks(
        self,
        chunks: list[Document],
        embeddings: list[list[float]],
    ) -> None:
        """Upsert chunk embeddings into Qdrant.

        Point IDs are deterministic (hash of source + chunk_index), so calling
        this method twice with the same documents is safe and idempotent.
        """
        from qdrant_client.http.models import PointStruct

        if len(chunks) != len(embeddings):
            raise ValueError(
                f"Mismatch: {len(chunks)} chunks but {len(embeddings)} embeddings"
            )

        points = [
            PointStruct(
                id=_deterministic_uuid(
                    chunk.metadata.get("source", ""),
                    chunk.metadata.get("chunk_index", idx),
                ),
                vector=embedding,
                payload={
                    "text": chunk.page_content,
                    **chunk.metadata,
                },
            )
            for idx, (chunk, embedding) in enumerate(zip(chunks, embeddings))
        ]

        # Upload in batches to avoid large request payloads
        batch_size = 100
        for i in range(0, len(points), batch_size):
            self.client.upsert(
                collection_name=self.collection_name,
                points=points[i : i + batch_size],
            )

        print(f"[VectorStore] Upserted {len(points)} points into '{self.collection_name}'")

    def delete_by_source(self, source: str) -> None:
        """Delete all vectors whose payload.source matches the given value.

        Enables per-document re-indexing without rebuilding the entire collection.
        """
        from qdrant_client.http.models import FieldCondition, Filter, MatchValue

        self.client.delete(
            collection_name=self.collection_name,
            points_selector=Filter(
                must=[FieldCondition(key="source", match=MatchValue(value=source))]
            ),
        )
        print(f"[VectorStore] Deleted points for source='{source}'")

    def collection_info(self) -> dict[str, Any]:
        info = self.client.get_collection(self.collection_name)
        return {
            "name": self.collection_name,
            "vectors_count": info.vectors_count,
            "status": str(info.status),
        }
