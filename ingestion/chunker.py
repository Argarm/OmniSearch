from __future__ import annotations

from langchain_core.documents import Document
from langchain_text_splitters import RecursiveCharacterTextSplitter


def build_splitter(
    chunk_size: int = 800,
    chunk_overlap: int = 150,
    separators: list[str] | None = None,
) -> RecursiveCharacterTextSplitter:
    """Build a token-aware text splitter using tiktoken cl100k_base encoding.

    Measuring in tokens (not characters) ensures chunks fit within the embedding
    model's context window regardless of text density.
    """
    if separators is None:
        separators = ["\n\n", "\n", ". ", " ", ""]

    return RecursiveCharacterTextSplitter.from_tiktoken_encoder(
        encoding_name="cl100k_base",
        chunk_size=chunk_size,
        chunk_overlap=chunk_overlap,
        separators=separators,
    )


def split_documents(
    documents: list[Document],
    chunk_size: int = 800,
    chunk_overlap: int = 150,
    separators: list[str] | None = None,
) -> list[Document]:
    """Split a list of Documents into chunks, preserving and extending metadata.

    Each chunk inherits all parent metadata plus a ``chunk_index`` field that
    records its position within the parent document (used for source ordering).
    """
    splitter = build_splitter(chunk_size, chunk_overlap, separators)

    all_chunks: list[Document] = []

    # Process per source so chunk_index resets for each document
    for doc in documents:
        chunks = splitter.split_documents([doc])
        for idx, chunk in enumerate(chunks):
            chunk.metadata["chunk_index"] = idx
        all_chunks.extend(chunks)

    return all_chunks
