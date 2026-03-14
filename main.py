"""
Medical Knowledge RAG System - FastAPI Application Entry Point
"""
import structlog
from contextlib import asynccontextmanager
from fastapi import FastAPI
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import JSONResponse

from app.config import settings
from app.api.routes import router

logger = structlog.get_logger()


@asynccontextmanager
async def lifespan(app: FastAPI):
    """Application startup and shutdown events."""
    logger.info(
        "Starting Medical RAG System",
        llm_provider=settings.llm_provider,
        embedding_model=settings.embedding_model,
    )

    # Pre-load embedding model and vector store on startup
    try:
        from app.core.embeddings import embedding_service
        from app.core.vector_store import vector_store
        from app.graph.pipeline import get_rag_graph

        # Warm up embedding model
        _ = embedding_service.embed_text("warmup")
        logger.info("Embedding model ready")

        # Compile LangGraph
        _ = get_rag_graph()
        logger.info("LangGraph pipeline compiled")

        stats = vector_store.get_collection_stats()
        logger.info("Vector store ready", stats=stats)

    except Exception as e:
        logger.error("Startup initialization error", error=str(e))

    yield
    logger.info("Medical RAG System shutting down")


app = FastAPI(
    title="Medical Knowledge RAG System",
    description=(
        "Production-ready Medical RAG System using LangChain + LangGraph. "
        "Retrieves evidence-based answers from trusted medical documents with citations, "
        "hallucination detection, safety filtering, and temporal knowledge ranking."
    ),
    version="1.0.0",
    lifespan=lifespan,
    docs_url="/docs",
    redoc_url="/redoc",
)

# CORS
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

# Routes
app.include_router(router)


@app.get("/")
async def root():
    return {
        "message": "Medical Knowledge RAG System",
        "version": "1.0.0",
        "docs": "/docs",
        "health": "/api/v1/health",
    }


@app.exception_handler(Exception)
async def global_exception_handler(request, exc):
    logger.error("Unhandled exception", error=str(exc))
    return JSONResponse(
        status_code=500,
        content={"detail": "An internal error occurred. Please try again."},
    )


if __name__ == "__main__":
    import uvicorn
    uvicorn.run(
        "main:app",
        host=settings.app_host,
        port=settings.app_port,
        reload=True,
        log_level=settings.log_level.lower(),
    )