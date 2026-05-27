from __future__ import annotations

from pydantic import BaseModel, Field


class Message(BaseModel):
    role: str  # "user" | "assistant"
    content: str


class QueryRequest(BaseModel):
    query: str = Field(..., min_length=1, max_length=2000)
    conversation_history: list[Message] = Field(default_factory=list)
    top_k: int = Field(default=6, ge=1, le=20)
    stream: bool = True


class SourceDocument(BaseModel):
    source: str
    source_type: str
    title: str
    page: int | None = None
    url: str | None = None
    chunk_text: str
    score: float


class QueryResponse(BaseModel):
    answer: str
    sources: list[SourceDocument]
    query_id: str


class HealthResponse(BaseModel):
    status: str
    collection: dict | None = None
