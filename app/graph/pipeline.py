"""
LangGraph pipeline — Medical RAG.

StateGraph(dict) replaces state entirely on each node return,
so every node wrapper must return a full merged copy of state.
"""
from typing import Dict, Any, Optional
from langgraph.graph import StateGraph, END
import structlog

from app.graph.nodes import (
    query_embedding_node,
    document_retrieval_node,
    temporal_ranking_node,
    context_filtering_node,
    answer_generation_node,
    hallucination_check_node,
    safety_filter_node,
    should_retrieve_more,
)

logger = structlog.get_logger()


def make_initial_state(
    query: str,
    conversation_id: Optional[str] = None,
    top_k: int = 5,
    include_temporal_ranking: bool = True,
) -> Dict[str, Any]:
    return {
        "query": query,
        "conversation_id": conversation_id,
        "top_k": top_k,
        "include_temporal_ranking": include_temporal_ranking,
        "query_embedding": None,
        "retrieved_docs": [],
        "ranked_docs": [],
        "filtered_docs": [],
        "generated_answer": None,
        "citations": [],
        "confidence_score": 0.0,
        "is_safe": True,
        "hallucination_detected": False,
        "highlighted_contexts": [],
        "needs_more_retrieval": False,
        "retrieval_attempts": 0,
        "error": None,
    }


# ── Node wrappers: always return {**state, **updates} ────────────────────────
# StateGraph(dict) REPLACES state with the node's return value.
# Without merging, every node would wipe out all other keys.

def _embed(state: Dict[str, Any]) -> Dict[str, Any]:
    result = query_embedding_node(state)
    # query_embedding_node returns {"query_embedding": [...]}
    return {**state, **result}


def _retrieve(state: Dict[str, Any]) -> Dict[str, Any]:
    result = document_retrieval_node(state)
    return {**state, **result}


def _rank(state: Dict[str, Any]) -> Dict[str, Any]:
    result = temporal_ranking_node(state)
    return {**state, **result}


def _filter(state: Dict[str, Any]) -> Dict[str, Any]:
    result = context_filtering_node(state)
    return {**state, **result}


def _generate(state: Dict[str, Any]) -> Dict[str, Any]:
    result = answer_generation_node(state)
    return {**state, **result}


def _hallucinate(state: Dict[str, Any]) -> Dict[str, Any]:
    result = hallucination_check_node(state)
    return {**state, **result}


def _safety(state: Dict[str, Any]) -> Dict[str, Any]:
    result = safety_filter_node(state)
    return {**state, **result}


def _route(state: Dict[str, Any]) -> str:
    """
    Conditional edge — decide whether to retry retrieval or proceed.
    Hard cap at 2 attempts to prevent infinite loops.
    """
    needs_more = state.get("needs_more_retrieval", False)
    attempts = state.get("retrieval_attempts", 0)
    if needs_more and attempts < 2:
        logger.info("Retrying retrieval", attempt=attempts)
        return "retrieve_more"
    return "generate"


# ── Build graph ───────────────────────────────────────────────────────────────
def build_medical_rag_graph():
    graph = StateGraph(dict)

    graph.add_node("embed",             _embed)
    graph.add_node("retrieve",          _retrieve)
    graph.add_node("rank",              _rank)
    graph.add_node("filter_ctx",        _filter)
    graph.add_node("generate",          _generate)
    graph.add_node("hallucinate_check", _hallucinate)
    graph.add_node("safety_check",      _safety)

    graph.set_entry_point("embed")
    graph.add_edge("embed",    "retrieve")
    graph.add_edge("retrieve", "rank")
    graph.add_edge("rank",     "filter_ctx")

    graph.add_conditional_edges(
        "filter_ctx",
        _route,
        {"retrieve_more": "retrieve", "generate": "generate"},
    )

    graph.add_edge("generate",          "hallucinate_check")
    graph.add_edge("hallucinate_check", "safety_check")
    graph.add_edge("safety_check",      END)

    return graph.compile()


_graph_instance = None


def get_rag_graph():
    global _graph_instance
    if _graph_instance is None:
        logger.info("Compiling Medical RAG LangGraph")
        _graph_instance = build_medical_rag_graph()
        logger.info("LangGraph compiled successfully")
    return _graph_instance


async def run_medical_rag(
    query: str,
    conversation_id: Optional[str] = None,
    top_k: int = 5,
    include_temporal_ranking: bool = True,
) -> Dict[str, Any]:
    graph = get_rag_graph()
    state = make_initial_state(query, conversation_id, top_k, include_temporal_ranking)
    logger.info("Starting RAG pipeline", query=query[:100])
    final_state = await graph.ainvoke(state)
    logger.info("RAG pipeline complete", confidence=final_state.get("confidence_score"))
    return final_state