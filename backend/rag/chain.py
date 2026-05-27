from __future__ import annotations

from typing import AsyncIterator

from langchain_core.messages import AIMessage, HumanMessage
from langchain_core.output_parsers import StrOutputParser
from langchain_core.runnables import RunnablePassthrough
from langchain_openai import ChatOpenAI

from backend.models.schemas import Message, SourceDocument
from backend.rag.prompts import RAG_PROMPT, REWRITE_PROMPT, format_context
from backend.rag.retriever import Retriever


def _to_langchain_messages(history: list[Message]):
    """Convert Pydantic Message objects to LangChain message objects."""
    lc_messages = []
    for msg in history:
        if msg.role == "user":
            lc_messages.append(HumanMessage(content=msg.content))
        else:
            lc_messages.append(AIMessage(content=msg.content))
    return lc_messages


class RAGChain:
    """LCEL-based RAG chain with streaming support and optional query rewriting.

    Flow:
    1. Rewrite the query if conversation history exists (resolves follow-ups).
    2. Retrieve relevant chunks from Qdrant.
    3. Format context and run the RAG prompt through the LLM.
    4. Stream tokens back to the caller; send source documents as a final event.
    """

    def __init__(
        self,
        retriever: Retriever,
        openai_api_key: str,
        openai_base_url: str,
        model: str = "gpt-4o-mini",
        temperature: float = 0.0,
        max_tokens: int = 2048,
    ) -> None:
        self.retriever = retriever
        self._llm = ChatOpenAI(
            model=model,
            temperature=temperature,
            max_tokens=max_tokens,
            streaming=True,
            api_key=openai_api_key,
            base_url=openai_base_url,
        )

    async def _rewrite_query(
        self, query: str, history: list[Message]
    ) -> str:
        """Use a cheap LLM call to expand a follow-up question into a standalone query."""
        if not history:
            return query

        chain = REWRITE_PROMPT | self._llm | StrOutputParser()
        rewritten = await chain.ainvoke({
            "query": query,
            "history": _to_langchain_messages(history),
        })
        return rewritten.strip() or query

    async def astream(
        self,
        query: str,
        history: list[Message],
        top_k: int = 6,
    ) -> AsyncIterator[str | list[SourceDocument]]:
        """Stream answer tokens, then yield the source document list as the last item.

        Yields:
            str: individual answer tokens during generation
            list[SourceDocument]: final item containing the retrieved sources
        """
        effective_query = await self._rewrite_query(query, history)

        sources = self.retriever.retrieve(effective_query, top_k=top_k)
        context = format_context([s.model_dump() for s in sources])

        chain = RAG_PROMPT | self._llm | StrOutputParser()

        async for token in chain.astream({
            "query": query,
            "context": context,
            "history": _to_langchain_messages(history) if history else [],
        }):
            yield token

        # Signal end of tokens — caller should send sources separately
        yield sources

    async def ainvoke(
        self,
        query: str,
        history: list[Message],
        top_k: int = 6,
    ) -> tuple[str, list[SourceDocument]]:
        """Non-streaming invocation. Returns (answer, sources)."""
        effective_query = await self._rewrite_query(query, history)

        sources = self.retriever.retrieve(effective_query, top_k=top_k)
        context = format_context([s.model_dump() for s in sources])

        chain = RAG_PROMPT | self._llm | StrOutputParser()
        answer = await chain.ainvoke({
            "query": query,
            "context": context,
            "history": _to_langchain_messages(history) if history else [],
        })

        return answer, sources
