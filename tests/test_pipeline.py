"""
Unit tests for the Medical RAG pipeline nodes.
Run: pytest tests/ -v
"""
import pytest
from unittest.mock import patch, MagicMock
import sys
import os

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))


# ─────────────────────────────────────────────────────────────────────────────
# Test Document Processor
# ─────────────────────────────────────────────────────────────────────────────
def test_document_processor_raw_text():
    from app.core.document_processor import DocumentProcessor
    processor = DocumentProcessor(chunk_size=100, chunk_overlap=10)
    chunks = processor.process_raw_text(
        "Diabetes is a metabolic disease characterized by high blood sugar. "
        "It affects millions of people worldwide. Treatment includes insulin therapy.",
        {"source": "test", "document_date": "2024-01-01"}
    )
    assert len(chunks) > 0
    assert all("text" in c for c in chunks)
    assert all("metadata" in c for c in chunks)


def test_document_processor_table_to_text():
    from app.core.document_processor import DocumentProcessor
    processor = DocumentProcessor()
    table = [
        ["Test", "Normal Range", "Unit"],
        ["Glucose", "70-100", "mg/dL"],
        ["HbA1c", "< 5.7", "%"],
    ]
    text = processor._table_to_text(table)
    assert "Glucose" in text
    assert "70-100" in text
    assert "mg/dL" in text


def test_clean_text():
    from app.core.document_processor import DocumentProcessor
    processor = DocumentProcessor()
    dirty = "  Hello   World  \n\n  Extra   spaces  "
    clean = processor._clean_text(dirty)
    assert "  " not in clean
    assert clean.strip() == clean


# ─────────────────────────────────────────────────────────────────────────────
# Test Embedding Service
# ─────────────────────────────────────────────────────────────────────────────
def test_cosine_similarity():
    from app.core.embeddings import EmbeddingService
    svc = EmbeddingService()
    # Same vector = 1.0
    v = [1.0, 0.0, 0.0]
    assert abs(svc.cosine_similarity(v, v) - 1.0) < 0.001
    # Orthogonal = 0.0
    v2 = [0.0, 1.0, 0.0]
    assert abs(svc.cosine_similarity(v, v2)) < 0.001


# ─────────────────────────────────────────────────────────────────────────────
# Test Temporal Ranking
# ─────────────────────────────────────────────────────────────────────────────
def test_temporal_rerank_prefers_recent():
    from app.core.vector_store import VectorStore
    vs = VectorStore.__new__(VectorStore)
    vs._initialized = True

    docs = [
        {"text": "old doc", "similarity_score": 0.9, "metadata": {"document_date": "2010-01-01"}},
        {"text": "new doc", "similarity_score": 0.8, "metadata": {"document_date": "2024-01-01"}},
    ]
    ranked = vs.temporal_rerank(docs, recency_weight=0.3)
    # New doc should rank higher despite lower semantic score
    assert ranked[0]["text"] == "new doc"


# ─────────────────────────────────────────────────────────────────────────────
# Test Graph Nodes
# ─────────────────────────────────────────────────────────────────────────────
def test_query_embedding_node():
    from app.graph.nodes import query_embedding_node
    state = {"query": "What are the symptoms of diabetes?"}
    result = query_embedding_node(state)
    assert "query_embedding" in result
    assert isinstance(result["query_embedding"], list)
    assert len(result["query_embedding"]) > 0


def test_should_retrieve_more_logic():
    from app.graph.nodes import should_retrieve_more
    # Should retry if needs_more and attempts < max
    state = {"needs_more_retrieval": True, "retrieval_attempts": 1}
    assert should_retrieve_more(state) == "retrieve_more"

    # Should not retry if max attempts reached
    state = {"needs_more_retrieval": True, "retrieval_attempts": 2}
    assert should_retrieve_more(state) == "generate"

    # Should not retry if not needed
    state = {"needs_more_retrieval": False, "retrieval_attempts": 0}
    assert should_retrieve_more(state) == "generate"


# ─────────────────────────────────────────────────────────────────────────────
# Test Schemas
# ─────────────────────────────────────────────────────────────────────────────
def test_source_citation_schema():
    from app.models.schemas import SourceCitation
    citation = SourceCitation(
        document_id="doc_1",
        source="WHO Guidelines",
        page=12,
        excerpt="Diabetes symptoms include...",
        relevance_score=0.87,
    )
    assert citation.source == "WHO Guidelines"
    assert citation.relevance_score == 0.87


def test_medical_answer_schema():
    from app.models.schemas import MedicalAnswer
    answer = MedicalAnswer(
        answer="Diabetes symptoms include polydipsia and polyuria.",
        citations=[],
        confidence_score=0.87,
    )
    assert answer.confidence_score == 0.87
    assert "educational purposes" in answer.safety_disclaimer


if __name__ == "__main__":
    pytest.main([__file__, "-v"])