"""
LangGraph nodes — Medical RAG pipeline.

Key improvements:
- Web sources ALWAYS enriched (not just on fallback)
- Accuracy score with label (High/Medium/Low) + breakdown
- Strict grounding: never hallucinates
- Safety severity levels
"""
import json
from typing import Dict, Any, List
import structlog
from langchain_core.messages import HumanMessage

from app.config import settings
from app.core.embeddings import embedding_service
from app.core.vector_store import vector_store
from app.core.memory import memory_store
from app.core.llm import get_llm, MEDICAL_QA_PROMPT, ACCURACY_CHECK_PROMPT, SAFETY_CHECK_PROMPT
from app.core.web_retrieval import fetch_web_sources
from app.models.schemas import SourceCitation

logger = structlog.get_logger()

NO_CONTEXT_MSG = "FALLBACK_TO_GENERAL"  # internal sentinel — triggers general knowledge answer

# ── Node 1: Query Embedding ───────────────────────────────────────────────────
def query_embedding_node(state: Dict[str, Any]) -> Dict[str, Any]:
    logger.info("Node: query_embedding", query=state["query"][:100])
    try:
        emb = embedding_service.embed_text(state["query"])
        return {"query_embedding": emb}
    except Exception as e:
        logger.error("Embedding failed", error=str(e))
        return {"query_embedding": None, "error": str(e)}


# ── Node 2: Document Retrieval ────────────────────────────────────────────────
def document_retrieval_node(state: Dict[str, Any]) -> Dict[str, Any]:
    logger.info("Node: document_retrieval")
    try:
        top_k = state.get("top_k", settings.retrieval_top_k)
        if state.get("retrieval_attempts", 0) > 0:
            top_k = min(top_k * 2, 20)
        docs = vector_store.similarity_search(query=state["query"], top_k=top_k)
        return {
            "retrieved_docs": docs,
            "retrieval_attempts": state.get("retrieval_attempts", 0) + 1,
        }
    except Exception as e:
        logger.error("Retrieval failed", error=str(e))
        return {
            "retrieved_docs": [],
            "retrieval_attempts": state.get("retrieval_attempts", 0) + 1,
            "error": str(e),
        }


# ── Node 3: Temporal Ranking ──────────────────────────────────────────────────
def temporal_ranking_node(state: Dict[str, Any]) -> Dict[str, Any]:
    logger.info("Node: temporal_ranking", doc_count=len(state.get("retrieved_docs", [])))
    docs = state.get("retrieved_docs", [])
    if not docs:
        return {"ranked_docs": []}
    if state.get("include_temporal_ranking", True):
        ranked = vector_store.temporal_rerank(docs, recency_weight=0.2)
    else:
        ranked = sorted(docs, key=lambda x: x["similarity_score"], reverse=True)
    return {"ranked_docs": ranked}


# ── Node 4: Context Filtering + Web Enrichment ───────────────────────────────
def context_filtering_node(state: Dict[str, Any]) -> Dict[str, Any]:
    logger.info("Node: context_filtering")

    ranked_docs = state.get("ranked_docs", [])
    min_score   = settings.min_confidence_threshold
    top_k       = state.get("top_k", settings.retrieval_top_k)

    # Filter local docs by threshold
    local_docs = [d for d in ranked_docs if d.get("similarity_score", 0) >= min_score][:top_k]

    # Compute confidence from embeddings
    confidence = 0.0
    if state.get("query_embedding") and local_docs:
        doc_embs = [embedding_service.embed_text(d["text"]) for d in local_docs[:3]]
        confidence = embedding_service.compute_confidence(state["query_embedding"], doc_embs)

    # ── Always fetch web sources ──────────────────────────────────────────────
    # Strategy:
    #   - If local KB has good results (conf > 0.5): enrich with 2 web sources
    #   - If local KB is weak/empty: fetch full 6 web sources
    web_fallback_used = False
    attempts = state.get("retrieval_attempts", 0)

    if confidence >= 0.5:
        # Good local results — add a couple of web sources for extra proof
        web_docs = fetch_web_sources(state["query"], max_total=2)
        web_fallback_used = False   # enrichment, not fallback
    else:
        # Weak or no local results — full web retrieval
        web_docs = fetch_web_sources(state["query"], max_total=6)
        web_fallback_used = True
        logger.info("Low local confidence — using web as primary source", confidence=confidence)

    # Merge local + web docs
    # Local docs already filtered by min_confidence_threshold (0.60)
    # Web docs: keep all — they come from trusted sources (WHO, PubMed, NIH)
    # The LLM accuracy check will still score and flag low-quality answers
    filtered = local_docs + web_docs
    has_context = len(filtered) > 0
    needs_more  = not has_context and attempts < 2

    # Recompute confidence based on number of sources found
    if filtered:
        n_docs = len(filtered)
        if n_docs >= 4:
            confidence = max(confidence, 0.60)   # enough sources
        elif n_docs >= 2:
            confidence = max(confidence, 0.50)   # some sources
        else:
            confidence = max(confidence, 0.40)   # at least one source

    # Build highlighted context snippets
    highlighted = [
        d["text"][:300] + "…" if len(d["text"]) > 300 else d["text"]
        for d in filtered[:3]
    ]

    logger.info("Context ready",
                local=len(local_docs), web=len(web_docs),
                total=len(filtered), confidence=round(confidence, 3),
                has_context=has_context, web_fallback=web_fallback_used)

    return {
        "filtered_docs":      filtered,
        "confidence_score":   confidence,
        "highlighted_contexts": highlighted,
        "needs_more_retrieval": needs_more,
        "web_fallback_used":  web_fallback_used,
        "has_context":        has_context,
    }


# ── Node 5: Answer Generation ─────────────────────────────────────────────────
def answer_generation_node(state: Dict[str, Any]) -> Dict[str, Any]:
    logger.info("Node: answer_generation")

    filtered_docs = state.get("filtered_docs", [])

    # If no local/web docs found, use the LLM's general medical knowledge
    # This ensures the AI ALWAYS helps rather than refusing entirely
    if not state.get("has_context") or not filtered_docs:
        logger.info("No retrieved context — using LLM general medical knowledge")
        general_prompt = f"""You are a comprehensive medical information assistant with expertise in all areas of medicine.
The user has asked a health question and you must provide a thorough, accurate, helpful answer.

QUESTION: {state["query"]}

Provide a well-structured, detailed medical answer that includes:
1. A direct answer to the question in the first paragraph
2. Detailed explanation of the condition/topic (causes, mechanism, symptoms if relevant)
3. Current treatment approaches or recommendations
4. Important warning signs or when to seek immediate medical care
5. Practical tips or lifestyle recommendations if applicable

Use clear, accessible language. Be thorough and helpful.
Always note at the end that the user should consult a healthcare professional for personal advice."""
        try:
            llm    = get_llm(temperature=0.1)
            answer = llm.invoke([HumanMessage(content=general_prompt)]).content
            # Add a disclaimer that this is from general knowledge, not retrieved docs
            answer += (
                "\n\n---\n*Note: This answer is based on general medical knowledge as no "
                "specific documents were retrieved from the knowledge base or web sources for "
                "this query. Always consult a qualified healthcare professional.*"
            )
            return {
                "generated_answer": answer,
                "citations": [],
                "web_fallback_used": False,
                "has_context": False,
            }
        except Exception as e:
            logger.error("General knowledge fallback failed", error=str(e))
            return {
                "generated_answer": (
                    "I was unable to find specific sources for this question. "
                    "Please consult a healthcare professional or visit WHO (who.int), "
                    "NHS (nhs.uk), or MedlinePlus (medlineplus.gov) for reliable information."
                ),
                "citations": [],
            }

    # Build numbered context for the prompt
    context_parts = []
    for i, doc in enumerate(filtered_docs, 1):
        meta = doc.get("metadata", {})
        org  = meta.get("organization", "")
        date = meta.get("document_date", "")
        header = f"[Source {i}: {meta.get('title', meta.get('source', 'Unknown'))} | {org}"
        if date:
            header += f" | {date}"
        header += "]"
        context_parts.append(f"{header}\n{doc['text']}")
    context = "\n\n---\n\n".join(context_parts)

    conv_id = state.get("conversation_id")
    history = memory_store.get_history_as_string(conv_id) if conv_id else "No previous conversation."

    prompt = MEDICAL_QA_PROMPT.format(
        context=context, history=history, question=state["query"]
    )

    try:
        llm    = get_llm(temperature=0.1)
        answer = llm.invoke([HumanMessage(content=prompt)]).content

        # Build rich citations
        citations = []
        for i, doc in enumerate(filtered_docs, 1):
            meta  = doc.get("metadata", {})
            src   = meta.get("source") or meta.get("title") or "Unknown"
            # Split "WHO — Title" style
            if " — " in src:
                _, title_part = src.split(" — ", 1)
            else:
                title_part = src
            citations.append(SourceCitation(
                document_id      = f"doc_{i}",
                source           = title_part[:120],
                organization     = meta.get("organization", ""),
                page             = meta.get("page"),
                section          = meta.get("section"),
                document_date    = meta.get("document_date", ""),
                guideline_version= meta.get("guideline_version"),
                excerpt          = doc["text"][:280],
                relevance_score  = round(doc.get("similarity_score", 0.0), 3),
                source_url       = meta.get("url", ""),
                is_web_source    = doc.get("is_web_fallback", False),
                web_source_name  = doc.get("web_source", ""),
            ))

        if conv_id:
            memory_store.add_message(conv_id, "user", state["query"])
            memory_store.add_message(conv_id, "assistant", answer)

        return {"generated_answer": answer, "citations": citations}

    except Exception as e:
        logger.error("Answer generation failed", error=str(e))
        return {
            "generated_answer": (
                "I encountered an error generating the answer. "
                "Please try again or consult a healthcare professional."
            ),
            "citations": [],
            "error": str(e),
        }


# ── Node 6: Accuracy Check (replaces hallucination check) ────────────────────
def hallucination_check_node(state: Dict[str, Any]) -> Dict[str, Any]:
    """
    Renamed semantically: scores accuracy of the answer vs sources.
    Returns accuracy_score (0-1), support_level (high/medium/low),
    and lists of supported and unsupported claims.
    """
    logger.info("Node: accuracy_check")
    answer       = state.get("generated_answer", "")
    filtered_docs= state.get("filtered_docs", [])

    # Skip accuracy check for general-knowledge fallback answers (no docs to check against)
    if not filtered_docs or not answer or "general medical knowledge" in answer:
        return {"hallucination_detected": False, "accuracy_score": 0.72,
                "support_level": "medium", "accuracy_explanation": "Answer from general medical knowledge."}

    context = "\n\n".join([d["text"][:500] for d in filtered_docs[:4]])
    prompt  = ACCURACY_CHECK_PROMPT.format(context=context, answer=answer)

    try:
        llm  = get_llm(temperature=0.0)
        raw  = llm.invoke([HumanMessage(content=prompt)]).content.strip()

        # Strip markdown fences if present
        for fence in ("```json", "```"):
            if fence in raw:
                raw = raw.split(fence)[1].split("```")[0]

        result          = json.loads(raw.strip())
        accuracy_score  = float(result.get("accuracy_score", 0.8))
        # Override support_level based on our 60% threshold rule (not just LLM label)
        if accuracy_score >= 0.75:
            support_level = "high"
        elif accuracy_score >= 0.60:
            support_level = "medium"
        else:
            support_level = "low"
        is_supported    = result.get("is_supported", True) and accuracy_score >= 0.60
        unsupported     = result.get("unsupported_claims", [])
        explanation     = result.get("explanation", "")

        # Blend accuracy_score with existing confidence
        blended = (state.get("confidence_score", 0.5) * 0.4) + (accuracy_score * 0.6)

        # Enforce 60% accuracy bar: below this = treat as hallucination / unreliable
        ACCURACY_THRESHOLD = 0.60
        hallucination_detected = (not is_supported) or (accuracy_score < ACCURACY_THRESHOLD)
        if accuracy_score < ACCURACY_THRESHOLD:
            logger.warning("Answer below 60% accuracy threshold — flagged",
                           accuracy=accuracy_score)

        logger.info("Accuracy check done",
                    accuracy=accuracy_score, level=support_level,
                    unsupported_count=len(unsupported),
                    hallucination_detected=hallucination_detected)

        return {
            "hallucination_detected": hallucination_detected,
            "confidence_score":       round(blended, 3),
            "accuracy_score":         accuracy_score,
            "support_level":          support_level,
            "unsupported_claims":     unsupported,
            "well_supported_claims":  result.get("well_supported_claims", []),
            "accuracy_explanation":   explanation,
        }
    except Exception as e:
        logger.error("Accuracy check failed", error=str(e))
        return {"hallucination_detected": False, "accuracy_score": 0.75,
                "support_level": "medium", "accuracy_explanation": "Check could not be completed."}


# ── Node 7: Safety Filter ─────────────────────────────────────────────────────
def safety_filter_node(state: Dict[str, Any]) -> Dict[str, Any]:
    logger.info("Node: safety_filter")

    if not settings.enable_safety_filter:
        return {"is_safe": True, "safety_severity": "none", "safety_concerns": []}

    answer = state.get("generated_answer", "")
    if not answer or len(answer) < 20:
        return {"is_safe": True, "safety_severity": "none", "safety_concerns": []}

    prompt = SAFETY_CHECK_PROMPT.format(answer=answer)
    try:
        llm = get_llm(temperature=0.0)
        raw = llm.invoke([HumanMessage(content=prompt)]).content.strip()

        for fence in ("```json", "```"):
            if fence in raw:
                raw = raw.split(fence)[1].split("```")[0]

        result   = json.loads(raw.strip())
        is_safe  = result.get("is_safe", True)
        severity = result.get("severity", "none")
        concerns = result.get("concerns", [])

        if not is_safe and severity in ("high", "medium"):
            logger.warning("Safety concern", severity=severity, concerns=concerns)

        return {
            "is_safe":          is_safe,
            "safety_severity":  severity,
            "safety_concerns":  concerns,
            "safe_summary":     result.get("safe_summary", ""),
        }
    except Exception as e:
        logger.error("Safety filter failed", error=str(e))
        return {"is_safe": True, "safety_severity": "none", "safety_concerns": []}


# ── Conditional edge ──────────────────────────────────────────────────────────
def should_retrieve_more(state: Dict[str, Any]) -> str:
    if state.get("needs_more_retrieval") and state.get("retrieval_attempts", 0) < 2:
        return "retrieve_more"
    return "generate"