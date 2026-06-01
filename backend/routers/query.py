from __future__ import annotations

import json
import uuid
from collections.abc import AsyncIterator

from fastapi import APIRouter, Request
from fastapi.responses import StreamingResponse

from backend.models.schemas import QueryRequest, QueryResponse

router = APIRouter()


async def _event_stream(
    request: Request,
    query_request: QueryRequest,
) -> AsyncIterator[str]:
    """Generate SSE events for a streaming RAG response.

    Event format:
      data: <token>          — individual answer token
      data: [SOURCES] <json> — final event with source document list
      data: [DONE]           — stream termination signal
    """
    chain = request.app.state.rag_chain

    async for item in chain.astream(
        query=query_request.query,
        history=query_request.conversation_history,
        top_k=query_request.top_k,
    ):
        if isinstance(item, str):
            yield f"data: {item}\n\n"
        elif isinstance(item, list):
            # Final item: source documents
            sources_json = json.dumps([s.model_dump() for s in item])
            yield f"data: [SOURCES] {sources_json}\n\n"

    yield "data: [DONE]\n\n"


@router.post("/query")
async def query(request: Request, body: QueryRequest):
    """Query the knowledge base.

    When ``stream=true`` (default), returns a text/event-stream SSE response
    where tokens arrive incrementally followed by a [SOURCES] event.

    When ``stream=false``, returns a standard JSON QueryResponse.
    """
    if body.stream:
        return StreamingResponse(
            _event_stream(request, body),
            media_type="text/event-stream",
            headers={
                "Cache-Control": "no-cache",
                "X-Accel-Buffering": "no",
            },
        )

    # Non-streaming path
    chain = request.app.state.rag_chain
    answer, sources = await chain.ainvoke(
        query=body.query,
        history=body.conversation_history,
        top_k=body.top_k,
    )

    return QueryResponse(
        answer=answer,
        sources=sources,
        query_id=str(uuid.uuid4()),
    )
