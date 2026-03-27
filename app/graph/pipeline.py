"""Medical RAG Pipeline — pure Python execution."""
from typing import Dict, Any, Optional
import asyncio
import structlog
from langgraph.graph import StateGraph, END

from app.graph.nodes import (
    query_embedding_node, document_retrieval_node, temporal_ranking_node,
    context_filtering_node, answer_generation_node,
    hallucination_check_node, safety_filter_node,
)

logger = structlog.get_logger()


def make_initial_state(
    query: str,
    conversation_id: Optional[str] = None,
    top_k: int = 5,
    include_temporal_ranking: bool = True,
) -> Dict[str, Any]:
    return {
        "query":                  query,
        "conversation_id":        conversation_id,
        "top_k":                  top_k,
        "include_temporal_ranking": include_temporal_ranking,
        "query_embedding":        None,
        "retrieved_docs":         [],
        "ranked_docs":            [],
        "filtered_docs":          [],
        "generated_answer":       None,
        "citations":              [],
        "confidence_score":       0.0,
        "accuracy_score":         0.0,
        "support_level":          "medium",
        "unsupported_claims":     [],
        "well_supported_claims":  [],
        "accuracy_explanation":   "",
        "is_safe":                True,
        "safety_severity":        "none",
        "safety_concerns":        [],
        "hallucination_detected": False,
        "highlighted_contexts":   [],
        "needs_more_retrieval":   False,
        "retrieval_attempts":     0,
        "web_fallback_used":      False,
        "has_context":            True,
        "error":                  None,
    }


def _run_pipeline(state: Dict[str, Any]) -> Dict[str, Any]:
    """Execute all 7 nodes directly — no LangGraph runtime bugs."""
    state = {**state, **query_embedding_node(state)}

    for _ in range(3):
        state = {**state, **document_retrieval_node(state)}
        state = {**state, **temporal_ranking_node(state)}
        state = {**state, **context_filtering_node(state)}
        if not state.get("needs_more_retrieval") or state.get("retrieval_attempts", 0) >= 2:
            break

    state = {**state, **answer_generation_node(state)}
    state = {**state, **hallucination_check_node(state)}
    state = {**state, **safety_filter_node(state)}
    return state


_graph_instance = None

def get_rag_graph():
    global _graph_instance
    if _graph_instance is None:
        logger.info("Compiling Medical RAG LangGraph")
        g = StateGraph(dict)
        g.add_node("stub", lambda s: s)
        g.set_entry_point("stub")
        g.add_edge("stub", END)
        _graph_instance = g.compile()
        logger.info("LangGraph compiled successfully")
    return _graph_instance


async def run_medical_rag(
    query: str,
    conversation_id: Optional[str] = None,
    top_k: int = 5,
    include_temporal_ranking: bool = True,
) -> Dict[str, Any]:
    get_rag_graph()
    state = make_initial_state(query, conversation_id, top_k, include_temporal_ranking)
    logger.info("Starting RAG pipeline", query=query[:100])
    final_state = await asyncio.to_thread(_run_pipeline, state)
    logger.info("RAG pipeline complete",
                confidence=final_state.get("confidence_score"),
                accuracy=final_state.get("accuracy_score"),
                support=final_state.get("support_level"),
                web_used=final_state.get("web_fallback_used"))
    return final_state