"""DocuStream Chainlit frontend.

Run with:
    chainlit run frontend/app.py --port 8501
"""
from __future__ import annotations

import json
import os

import chainlit as cl
import httpx

from frontend.components.source_panel import build_source_elements

BACKEND_URL = os.getenv("BACKEND_URL", "http://localhost:8000")
QUERY_ENDPOINT = f"{BACKEND_URL}/api/v1/query"


@cl.on_chat_start
async def on_chat_start():
    """Initialize session state when a new conversation starts."""
    cl.user_session.set("history", [])
    await cl.Message(
        content="Hello! I'm DocuStream, your internal knowledge base assistant. What would you like to know?",
        author="DocuStream",
    ).send()


@cl.on_message
async def on_message(message: cl.Message):
    """Handle incoming user messages, stream the RAG response, and show sources."""
    history: list[dict] = cl.user_session.get("history", [])

    # Start the response message (will be streamed into)
    response_msg = cl.Message(content="", author="DocuStream")
    await response_msg.send()

    sources: list[dict] = []
    full_answer_tokens: list[str] = []

    payload = {
        "query": message.content,
        "conversation_history": history,
        "stream": True,
    }

    try:
        async with httpx.AsyncClient(timeout=60.0) as client:
            async with client.stream("POST", QUERY_ENDPOINT, json=payload) as response:
                response.raise_for_status()

                async for line in response.aiter_lines():
                    if not line.startswith("data: "):
                        continue

                    data = line[len("data: "):]

                    if data == "[DONE]":
                        break

                    if data.startswith("[SOURCES] "):
                        sources = json.loads(data[len("[SOURCES] "):])
                        continue

                    # Regular token — stream it into the message
                    await response_msg.stream_token(data)
                    full_answer_tokens.append(data)

    except httpx.HTTPStatusError as exc:
        error_text = f"\n\n⚠ Backend error {exc.response.status_code}: {exc.response.text}"
        await response_msg.stream_token(error_text)
    except Exception as exc:
        error_text = f"\n\n⚠ Connection error: {exc}"
        await response_msg.stream_token(error_text)

    # Finalize the streamed message
    await response_msg.update()

    # Attach source documents as side-panel elements if any were returned
    if sources:
        source_elements = build_source_elements(sources)
        response_msg.elements = source_elements
        await response_msg.update()

    # Update conversation history for follow-up question support
    full_answer = "".join(full_answer_tokens)
    history.append({"role": "user", "content": message.content})
    history.append({"role": "assistant", "content": full_answer})
    cl.user_session.set("history", history)
