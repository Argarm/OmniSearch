from __future__ import annotations

from functools import lru_cache

from backend.models.schemas import SourceDocument


@lru_cache(maxsize=1)
def _get_embedder(model_name: str, device: str, query_prefix: str):
    """Load and cache the embedding model once per process."""
    from ingestion.embedder import Embedder

    return Embedder(model_name=model_name, device=device, query_prefix=query_prefix)


@lru_cache(maxsize=1)
def _get_qdrant_client(url: str, api_key: str):
    from qdrant_client import QdrantClient

    return QdrantClient(url=url, api_key=api_key or None, timeout=10)


class Retriever:
    """Performs similarity search against the Qdrant collection.

    Applies the BGE query prefix before embedding and filters results by the
    configured score threshold to avoid returning low-confidence chunks.
    """

    def __init__(
        self,
        qdrant_url: str,
        qdrant_api_key: str,
        collection_name: str,
        embed_model: str,
        embed_device: str,
        query_prefix: str,
        top_k: int = 6,
        score_threshold: float = 0.35,
    ) -> None:
        self.qdrant_url = qdrant_url
        self.qdrant_api_key = qdrant_api_key
        self.collection_name = collection_name
        self.embed_model = embed_model
        self.embed_device = embed_device
        self.query_prefix = query_prefix
        self.top_k = top_k
        self.score_threshold = score_threshold

    @property
    def embedder(self):
        return _get_embedder(self.embed_model, self.embed_device, self.query_prefix)

    @property
    def client(self):
        return _get_qdrant_client(self.qdrant_url, self.qdrant_api_key)

    def retrieve(self, query: str, top_k: int | None = None) -> list[SourceDocument]:
        """Embed the query and return the top-k matching source documents."""
        query_vector = self.embedder.embed_query(query)

        results = self.client.search(
            collection_name=self.collection_name,
            query_vector=query_vector,
            limit=top_k or self.top_k,
            score_threshold=self.score_threshold,
            with_payload=True,
        )

        sources: list[SourceDocument] = []
        for hit in results:
            payload = hit.payload or {}
            sources.append(
                SourceDocument(
                    source=payload.get("source", ""),
                    source_type=payload.get("source_type", ""),
                    title=payload.get("title", ""),
                    page=payload.get("page"),
                    url=payload.get("url"),
                    chunk_text=payload.get("text", ""),
                    score=round(hit.score, 4),
                )
            )

        return sources
