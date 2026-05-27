from __future__ import annotations

from functools import lru_cache

from langchain_core.documents import Document


@lru_cache(maxsize=1)
def _load_model(model_name: str, device: str):
    """Load the SentenceTransformer model once and cache it for the process lifetime."""
    try:
        from sentence_transformers import SentenceTransformer
    except ImportError as e:
        raise ImportError(
            "sentence-transformers is required: pip install sentence-transformers"
        ) from e
    return SentenceTransformer(model_name, device=device)


class Embedder:
    """Generates embeddings using a HuggingFace SentenceTransformer model.

    BGE models use an asymmetric embedding strategy:
    - Documents are embedded WITHOUT any prefix at index time.
    - Queries are embedded WITH a prefix at retrieval time to improve ranking.
    This asymmetry is part of the BGE design and should not be changed.
    """

    def __init__(
        self,
        model_name: str = "BAAI/bge-large-en-v1.5",
        device: str = "cpu",
        batch_size: int = 32,
        query_prefix: str = "Represent this sentence for searching relevant passages: ",
    ) -> None:
        self.model_name = model_name
        self.device = device
        self.batch_size = batch_size
        self.query_prefix = query_prefix

    @property
    def model(self):
        return _load_model(self.model_name, self.device)

    def embed_documents(self, chunks: list[Document]) -> list[list[float]]:
        """Embed document chunks — no prefix applied (BGE asymmetric design)."""
        texts = [chunk.page_content for chunk in chunks]
        return self._encode(texts, apply_prefix=False)

    def embed_query(self, query: str) -> list[float]:
        """Embed a retrieval query — applies the BGE query prefix."""
        return self._encode([query], apply_prefix=True)[0]

    def _encode(self, texts: list[str], apply_prefix: bool) -> list[list[float]]:
        if apply_prefix:
            texts = [self.query_prefix + t for t in texts]

        all_embeddings: list[list[float]] = []
        for i in range(0, len(texts), self.batch_size):
            batch = texts[i : i + self.batch_size]
            embeddings = self.model.encode(
                batch,
                normalize_embeddings=True,
                show_progress_bar=False,
            )
            all_embeddings.extend(embeddings.tolist())

        return all_embeddings
