from __future__ import annotations

from contextlib import asynccontextmanager

from fastapi import FastAPI
from fastapi.middleware.cors import CORSMiddleware

from backend.core.config import get_settings
from backend.models.schemas import HealthResponse
from backend.rag.chain import RAGChain
from backend.rag.retriever import Retriever
from backend.routers.query import router as query_router


@asynccontextmanager
async def lifespan(app: FastAPI):
    """Initialize expensive resources once at startup and clean up on shutdown."""
    # Respect a pre-injected chain (e.g. a test mock); only build the real one otherwise.
    if getattr(app.state, "rag_chain", None) is None:
        settings = get_settings()

        retriever = Retriever(
            qdrant_url=settings.qdrant.url,
            qdrant_api_key=settings.qdrant.api_key,
            collection_name=settings.qdrant.collection_name,
            embed_model=settings.embedding.model_name,
            embed_device=settings.embedding.device,
            query_prefix=settings.embedding.query_prefix,
            top_k=settings.retrieval.top_k,
            score_threshold=settings.retrieval.score_threshold,
        )

        app.state.rag_chain = RAGChain(retriever=retriever)

    print("[OmniSearch] Backend ready.")
    yield
    print("[OmniSearch] Backend shutting down.")


def create_app() -> FastAPI:
    settings = get_settings()

    app = FastAPI(
        title="OmniSearch API",
        description="RAG query backend for internal knowledge bases",
        version="0.1.0",
        lifespan=lifespan,
    )

    app.add_middleware(
        CORSMiddleware,
        allow_origins=settings.cors_origins,
        allow_credentials=True,
        allow_methods=["GET", "POST"],
        allow_headers=["*"],
    )

    app.include_router(query_router, prefix="/api/v1", tags=["query"])

    @app.get("/health", response_model=HealthResponse, tags=["health"])
    async def health():
        try:
            chain = app.state.rag_chain
            info = chain.retriever.client.get_collection(
                chain.retriever.collection_name
            )
            collection_info = {
                "name": chain.retriever.collection_name,
                "vectors_count": info.points_count,
                "status": str(info.status),
            }
            return HealthResponse(status="ok", collection=collection_info)
        except Exception as exc:
            return HealthResponse(status=f"degraded: {exc}")

    return app


app = create_app()

if __name__ == "__main__":
    import uvicorn

    settings = get_settings()
    uvicorn.run(
        "backend.main:app",
        host=settings.backend_host,
        port=settings.backend_port,
        reload=True,
    )
