"""
LangGraph nodes for the Medical RAG pipeline.

Pipeline:
  query_embedding → document_retrieval → temporal_ranking
  → context_filtering → answer_generation → hallucination_check
  → safety_filter → finalize
"""
import json
from typing import Dict, Any, List
import structlog

from langchain_core.messages import HumanMessage, SystemMessage

from app.config import settings
from app.core.embeddings import embedding_service
from app.core.vector_store import vector_store
from app.core.memory import memory_store
from app.core.llm import get_llm, MEDICAL_QA_PROMPT, HALLUCINATION_CHECK_PROMPT, SAFETY_CHECK_PROMPT
from app.models.schemas import MedicalRAGState, SourceCitation

logger = structlog.get_logger()


# ─────────────────────────────────────────────────────────────────────────────
# NODE 1: Query Embedding
# ─────────────────────────────────────────────────────────────────────────────
def query_embedding_node(state: Dict[str, Any]) -> Dict[str, Any]:
    """Embed the user's query for semantic retrieval."""
    logger.info("Node: query_embedding", query=state["query"][:100])
    try:
        embedding = embedding_service.embed_text(state["query"])
        return {"query_embedding": embedding}
    except Exception as e:
        logger.error("Embedding failed", error=str(e))
        return {"error": f"Embedding failed: {str(e)}"}


# ─────────────────────────────────────────────────────────────────────────────
# NODE 2: Document Retrieval
# ─────────────────────────────────────────────────────────────────────────────
def document_retrieval_node(state: Dict[str, Any]) -> Dict[str, Any]:
    """Retrieve semantically similar documents from ChromaDB."""
    logger.info("Node: document_retrieval")
    try:
        top_k = state.get("top_k", settings.retrieval_top_k)
        # Increase k on retry
        if state.get("retrieval_attempts", 0) > 0:
            top_k = min(top_k * 2, 20)

        docs = vector_store.similarity_search(
            query=state["query"],
            top_k=top_k,
        )

        return {
            "retrieved_docs": docs,
            "retrieval_attempts": state.get("retrieval_attempts", 0) + 1,
        }
    except Exception as e:
        logger.error("Retrieval failed", error=str(e))
        return {
            "retrieved_docs": [],
            "retrieval_attempts": state.get("retrieval_attempts", 0) + 1,
            "error": f"Retrieval failed: {str(e)}",
        }


# ─────────────────────────────────────────────────────────────────────────────
# NODE 3: Temporal Relevance Ranking
# ─────────────────────────────────────────────────────────────────────────────
def temporal_ranking_node(state: Dict[str, Any]) -> Dict[str, Any]:
    """Re-rank docs by combining semantic similarity + document recency."""
    logger.info("Node: temporal_ranking", doc_count=len(state.get("retrieved_docs", [])))

    docs = state.get("retrieved_docs", [])
    if not docs:
        return {"ranked_docs": []}

    use_temporal = state.get("include_temporal_ranking", True)
    if use_temporal:
        ranked = vector_store.temporal_rerank(docs, recency_weight=0.2)
    else:
        ranked = sorted(docs, key=lambda x: x["similarity_score"], reverse=True)

    return {"ranked_docs": ranked}


# ─────────────────────────────────────────────────────────────────────────────
# NODE 4: Context Filtering
# ─────────────────────────────────────────────────────────────────────────────
def context_filtering_node(state: Dict[str, Any]) -> Dict[str, Any]:
    """
    Filter out low-relevance documents and compute confidence score.
    Also highlights relevant document excerpts.
    """
    logger.info("Node: context_filtering")

    ranked_docs = state.get("ranked_docs", [])
    min_score = settings.min_confidence_threshold

    # Filter by minimum similarity threshold
    filtered = [
        d for d in ranked_docs
        if d.get("similarity_score", 0) >= min_score
    ]

    # Cap at top_k
    top_k = state.get("top_k", settings.retrieval_top_k)
    filtered = filtered[:top_k]

    # Compute confidence from query embedding vs doc embeddings
    confidence = 0.0
    if state.get("query_embedding") and filtered:
        doc_embeddings = [
            embedding_service.embed_text(d["text"]) for d in filtered[:3]
        ]
        confidence = embedding_service.compute_confidence(
            state["query_embedding"], doc_embeddings
        )

    # Build highlighted contexts
    highlighted = [d["text"][:300] + "..." if len(d["text"]) > 300 else d["text"]
                   for d in filtered[:3]]

    # Flag if we need more retrieval
    needs_more = len(filtered) == 0 or confidence < 0.4

    logger.info(
        "Context filtered",
        filtered_count=len(filtered),
        confidence=round(confidence, 3),
        needs_more=needs_more,
    )

    return {
        "filtered_docs": filtered,
        "confidence_score": confidence,
        "highlighted_contexts": highlighted,
        "needs_more_retrieval": needs_more,
    }


# ─────────────────────────────────────────────────────────────────────────────
# NODE 5: Answer Generation
# ─────────────────────────────────────────────────────────────────────────────
def answer_generation_node(state: Dict[str, Any]) -> Dict[str, Any]:
    """Generate grounded medical answer from filtered context."""
    logger.info("Node: answer_generation")

    filtered_docs = state.get("filtered_docs", [])

    # Build context string with source labels
    context_parts = []
    for i, doc in enumerate(filtered_docs, 1):
        meta = doc.get("metadata", {})
        source_label = (
            f"[Source {i}: {meta.get('source', 'Unknown')} "
            f"| Page {meta.get('page', 'N/A')} "
            f"| Date: {meta.get('document_date', 'Unknown')}]"
        )
        context_parts.append(f"{source_label}\n{doc['text']}")
    context = "\n\n---\n\n".join(context_parts) if context_parts else "No relevant documents found."

    # Build conversation history
    conv_id = state.get("conversation_id")
    if conv_id:
        history = memory_store.get_history_as_string(conv_id)
    else:
        history = "No previous conversation."

    # Format prompt
    prompt = MEDICAL_QA_PROMPT.format(
        context=context,
        history=history,
        question=state["query"],
    )

    try:
        llm = get_llm(temperature=0.1)
        response = llm.invoke([HumanMessage(content=prompt)])
        answer = response.content

        # Build citations
        citations = []
        for i, doc in enumerate(filtered_docs, 1):
            meta = doc.get("metadata", {})
            citations.append(
                SourceCitation(
                    document_id=f"doc_{i}",
                    source=meta.get("source", "Unknown"),
                    page=meta.get("page"),
                    section=meta.get("section"),
                    document_date=meta.get("document_date"),
                    guideline_version=meta.get("guideline_version"),
                    excerpt=doc["text"][:200],
                    relevance_score=doc.get("similarity_score", 0.0),
                )
            )

        # Update conversation memory
        if conv_id:
            memory_store.add_message(conv_id, "user", state["query"])
            memory_store.add_message(conv_id, "assistant", answer)

        return {"generated_answer": answer, "citations": citations}

    except Exception as e:
        logger.error("Answer generation failed", error=str(e))
        fallback = (
            "I was unable to generate an answer from the available medical documents. "
            "Please consult a healthcare professional for your query."
        )
        return {"generated_answer": fallback, "citations": [], "error": str(e)}


# ─────────────────────────────────────────────────────────────────────────────
# NODE 6: Hallucination Detection
# ─────────────────────────────────────────────────────────────────────────────
def hallucination_check_node(state: Dict[str, Any]) -> Dict[str, Any]:
    """Verify the generated answer is grounded in retrieved context."""
    logger.info("Node: hallucination_check")

    answer = state.get("generated_answer", "")
    filtered_docs = state.get("filtered_docs", [])

    if not filtered_docs or not answer:
        return {"hallucination_detected": False}

    context = "\n\n".join([d["text"][:500] for d in filtered_docs[:3]])
    prompt = HALLUCINATION_CHECK_PROMPT.format(context=context, answer=answer)

    try:
        llm = get_llm(temperature=0.0)
        response = llm.invoke([HumanMessage(content=prompt)])
        raw = response.content.strip()

        # Parse JSON response
        if "```json" in raw:
            raw = raw.split("```json")[1].split("```")[0]
        elif "```" in raw:
            raw = raw.split("```")[1].split("```")[0]

        result = json.loads(raw)
        hallucination_detected = not result.get("is_supported", True)

        if hallucination_detected:
            logger.warning(
                "Hallucination detected",
                unsupported=result.get("unsupported_claims", []),
            )

        return {
            "hallucination_detected": hallucination_detected,
            "confidence_score": min(
                state.get("confidence_score", 0.0),
                result.get("confidence", 1.0),
            ),
        }

    except Exception as e:
        logger.error("Hallucination check failed", error=str(e))
        return {"hallucination_detected": False}


# ─────────────────────────────────────────────────────────────────────────────
# NODE 7: Safety Filter
# ─────────────────────────────────────────────────────────────────────────────
def safety_filter_node(state: Dict[str, Any]) -> Dict[str, Any]:
    """Check for potentially unsafe medical advice."""
    logger.info("Node: safety_filter")

    if not settings.enable_safety_filter:
        return {"is_safe": True}

    answer = state.get("generated_answer", "")
    if not answer:
        return {"is_safe": True}

    prompt = SAFETY_CHECK_PROMPT.format(answer=answer)

    try:
        llm = get_llm(temperature=0.0)
        response = llm.invoke([HumanMessage(content=prompt)])
        raw = response.content.strip()

        if "```json" in raw:
            raw = raw.split("```json")[1].split("```")[0]
        elif "```" in raw:
            raw = raw.split("```")[1].split("```")[0]

        result = json.loads(raw)
        is_safe = result.get("is_safe", True)
        severity = result.get("severity", "none")

        if not is_safe and severity in ("high", "medium"):
            logger.warning("Safety concern detected", severity=severity, concerns=result.get("concerns"))

        return {"is_safe": is_safe}

    except Exception as e:
        logger.error("Safety filter failed", error=str(e))
        return {"is_safe": True}  # Default to safe on error


# ─────────────────────────────────────────────────────────────────────────────
# CONDITIONAL EDGE: Check if more retrieval needed
# ─────────────────────────────────────────────────────────────────────────────
def should_retrieve_more(state: Dict[str, Any]) -> str:
    """Decide whether to retry retrieval or proceed to generation."""
    needs_more = state.get("needs_more_retrieval", False)
    attempts = state.get("retrieval_attempts", 0)
    max_attempts = 2

    if needs_more and attempts < max_attempts:
        logger.info("Retrying retrieval", attempt=attempts)
        return "retrieve_more"
    return "generate"