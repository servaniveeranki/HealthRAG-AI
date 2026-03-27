"""
FastAPI routes for the Medical RAG System.

Endpoints:
  POST /api/v1/query          - Ask a medical question
  POST /api/v1/ingest/file    - Upload and ingest a document
  POST /api/v1/ingest/text    - Ingest raw text
  GET  /api/v1/conversation/{id} - Get conversation history
  POST /api/v1/conversation/new  - Create new conversation
  GET  /api/v1/health         - Health check
  GET  /api/v1/stats          - Vector store stats
"""
import uuid
import os
import tempfile
from typing import Optional

from fastapi import APIRouter, UploadFile, File, Form, HTTPException, BackgroundTasks
from fastapi.responses import JSONResponse
import structlog

from app.config import settings
from app.graph.pipeline import run_medical_rag
from app.core.document_processor import document_processor
from app.core.vector_store import vector_store
from app.core.memory import memory_store
from app.models.schemas import (
    QueryRequest,
    MedicalAnswer,
    IngestResponse,
    SourceCitation,
)

router = APIRouter(prefix="/api/v1", tags=["Medical RAG"])
logger = structlog.get_logger()

DISCLAIMER = (
    "⚠️ This information is for educational purposes only and should not replace "
    "professional medical advice. Always consult a qualified healthcare provider."
)

ALLOWED_EXTENSIONS = {".pdf", ".txt", ".md", ".png", ".jpg", ".jpeg", ".tiff"}


# ─────────────────────────────────────────────────────────────────────────────
# Query Endpoint
# ─────────────────────────────────────────────────────────────────────────────
@router.post("/query", response_model=MedicalAnswer)
async def query_medical_knowledge(request: QueryRequest):
    """
    Ask a medical question and get an evidence-based answer with citations.
    """
    logger.info("Query received", query=request.query[:100])

    try:
        # Run the LangGraph pipeline
        state = await run_medical_rag(
            query=request.query,
            conversation_id=request.conversation_id,
            top_k=request.top_k,
            include_temporal_ranking=request.include_temporal_ranking,
        )

        # Handle errors
        if state.get("error") and not state.get("generated_answer"):
            raise HTTPException(status_code=500, detail=state["error"])

        answer_text = state.get("generated_answer") or "No answer could be generated."
        citations = state.get("citations", [])
        confidence = state.get("confidence_score", 0.0)
        is_safe = state.get("is_safe", True)
        hallucination_detected = state.get("hallucination_detected", False)
        highlighted = state.get("highlighted_contexts", [])

        # Append warning if hallucination detected
        if hallucination_detected:
            answer_text += (
                "\n\n⚠️ Note: Some claims in this response may not be fully "
                "supported by the retrieved documents. Please verify with primary sources."
            )

        return MedicalAnswer(
            accuracy_score=round(state.get("accuracy_score", 0.0), 3),
            support_level=state.get("support_level", "medium"),
            unsupported_claims=state.get("unsupported_claims", []),
            well_supported_claims=state.get("well_supported_claims", []),
            accuracy_explanation=state.get("accuracy_explanation", ""),
            safety_severity=state.get("safety_severity", "none"),
            safety_concerns=state.get("safety_concerns", []),
            web_fallback_used=state.get("web_fallback_used", False),
            answer=answer_text,
            citations=citations,
            confidence_score=round(confidence, 3),
            is_safe=is_safe,
            safety_disclaimer=DISCLAIMER,
            hallucination_detected=hallucination_detected,
            highlighted_contexts=highlighted,
        )

    except HTTPException:
        raise
    except Exception as e:
        logger.error("Query failed", error=str(e))
        raise HTTPException(status_code=500, detail=f"Query processing failed: {str(e)}")


# ─────────────────────────────────────────────────────────────────────────────
# Document Ingestion - File Upload
# ─────────────────────────────────────────────────────────────────────────────
@router.post("/ingest/file", response_model=IngestResponse)
async def ingest_document_file(
    background_tasks: BackgroundTasks,
    file: UploadFile = File(...),
    source_type: str = Form(default="Other"),
    document_date: Optional[str] = Form(default=None),
    guideline_version: Optional[str] = Form(default=None),
    title: Optional[str] = Form(default=None),
):
    """
    Upload and ingest a medical document (PDF, TXT, image).
    Supports text extraction, table parsing, and OCR.
    """
    ext = os.path.splitext(file.filename or "")[1].lower()
    if ext not in ALLOWED_EXTENSIONS:
        raise HTTPException(
            status_code=400,
            detail=f"Unsupported file type: {ext}. Allowed: {ALLOWED_EXTENSIONS}",
        )

    doc_id = str(uuid.uuid4())

    # Save to temp file
    with tempfile.NamedTemporaryFile(delete=False, suffix=ext) as tmp:
        content = await file.read()
        tmp.write(content)
        tmp_path = tmp.name

    try:
        metadata = {
            "document_id": doc_id,
            "source": source_type or document_processor.auto_detect_source(file.filename or ""),
            "title": title or file.filename,
            "document_date": document_date or "",
            "guideline_version": guideline_version or "",
            "filename": file.filename,
        }

        # Process based on file type
        if ext == ".pdf":
            chunks = document_processor.process_pdf(tmp_path, metadata)
        elif ext in {".png", ".jpg", ".jpeg", ".tiff"}:
            chunks = document_processor.process_image(tmp_path, metadata)
        else:
            chunks = document_processor.process_text(tmp_path, metadata)

        if not chunks:
            raise HTTPException(
                status_code=422,
                detail="No content could be extracted from the document.",
            )

        # Store in vector database
        vector_store.add_documents(chunks)
        os.unlink(tmp_path)

        logger.info("Document ingested", doc_id=doc_id, chunks=len(chunks))
        return IngestResponse(
            success=True,
            document_id=doc_id,
            chunks_created=len(chunks),
            message=f"Successfully ingested '{file.filename}' into {len(chunks)} chunks.",
        )

    except HTTPException:
        raise
    except Exception as e:
        logger.error("Ingestion failed", error=str(e))
        if os.path.exists(tmp_path):
            os.unlink(tmp_path)
        raise HTTPException(status_code=500, detail=f"Ingestion failed: {str(e)}")


# ─────────────────────────────────────────────────────────────────────────────
# Document Ingestion - Raw Text
# ─────────────────────────────────────────────────────────────────────────────
@router.post("/ingest/text", response_model=IngestResponse)
async def ingest_raw_text(
    text: str = Form(...),
    source_type: str = Form(default="Other"),
    document_date: Optional[str] = Form(default=None),
    guideline_version: Optional[str] = Form(default=None),
    title: Optional[str] = Form(default=None),
):
    """Ingest raw text content directly into the knowledge base."""
    doc_id = str(uuid.uuid4())

    metadata = {
        "document_id": doc_id,
        "source": source_type,
        "title": title or "Untitled Document",
        "document_date": document_date or "",
        "guideline_version": guideline_version or "",
    }

    chunks = document_processor.process_raw_text(text, metadata)
    if not chunks:
        raise HTTPException(status_code=422, detail="No content extracted.")

    vector_store.add_documents(chunks)
    return IngestResponse(
        success=True,
        document_id=doc_id,
        chunks_created=len(chunks),
        message=f"Ingested text into {len(chunks)} chunks.",
    )


# ─────────────────────────────────────────────────────────────────────────────
# Conversation Management
# ─────────────────────────────────────────────────────────────────────────────
@router.post("/conversation/new")
async def create_conversation():
    """Create a new conversation session."""
    conv_id = memory_store.create_conversation()
    return {"conversation_id": conv_id}


@router.get("/conversation/{conversation_id}")
async def get_conversation(conversation_id: str):
    """Get conversation history."""
    conv = memory_store.get_conversation(conversation_id)
    if not conv:
        raise HTTPException(status_code=404, detail="Conversation not found.")
    return conv


@router.delete("/conversation/{conversation_id}")
async def delete_conversation(conversation_id: str):
    """Delete a conversation."""
    deleted = memory_store.delete_conversation(conversation_id)
    if not deleted:
        raise HTTPException(status_code=404, detail="Conversation not found.")
    return {"success": True}


# ─────────────────────────────────────────────────────────────────────────────
# Health & Stats
# ─────────────────────────────────────────────────────────────────────────────
@router.get("/health")
async def health_check():
    """System health check."""
    try:
        stats = vector_store.get_collection_stats()
        return {
            "status": "healthy",
            "vector_store": stats,
            "llm_provider": settings.llm_provider,
            "embedding_model": settings.embedding_model,
        }
    except Exception as e:
        return JSONResponse(
            status_code=503,
            content={"status": "unhealthy", "error": str(e)},
        )


@router.get("/stats")
async def get_stats():
    """Get vector store and system statistics."""
    return {
        "vector_store": vector_store.get_collection_stats(),
        "active_conversations": len(memory_store.list_conversations()),
        "config": {
            "chunk_size": settings.chunk_size,
            "chunk_overlap": settings.chunk_overlap,
            "retrieval_top_k": settings.retrieval_top_k,
            "min_confidence_threshold": settings.min_confidence_threshold,
            "safety_filter_enabled": settings.enable_safety_filter,
        },
    }