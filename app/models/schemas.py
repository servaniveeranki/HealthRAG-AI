"""
Pydantic schemas for request/response models.
"""
from pydantic import BaseModel, Field
from typing import Optional, List, Dict, Any
from datetime import datetime
from enum import Enum


class DocumentSource(str, Enum):
    WHO_GUIDELINES = "WHO Guidelines"
    PUBMED = "PubMed"
    MEDICAL_TEXTBOOK = "Medical Textbook"
    CLINICAL_RESEARCH = "Clinical Research"
    LAB_REPORT = "Lab Report"
    OTHER = "Other"


class SourceCitation(BaseModel):
    document_id: str
    source: str
    page: Optional[int] = None
    section: Optional[str] = None
    document_date: Optional[str] = None
    guideline_version: Optional[str] = None
    excerpt: str
    relevance_score: float


class MedicalAnswer(BaseModel):
    answer: str
    citations: List[SourceCitation]
    confidence_score: float = Field(ge=0.0, le=1.0)
    is_safe: bool = True
    safety_disclaimer: str = (
        "⚠️ This information is for educational purposes only and should not "
        "replace professional medical advice. Always consult a qualified healthcare provider."
    )
    hallucination_detected: bool = False
    highlighted_contexts: List[str] = []
    query_timestamp: datetime = Field(default_factory=datetime.utcnow)


class QueryRequest(BaseModel):
    query: str = Field(..., min_length=3, max_length=2000)
    conversation_id: Optional[str] = None
    top_k: int = Field(default=5, ge=1, le=20)
    include_temporal_ranking: bool = True
    language: str = "en"


class IngestRequest(BaseModel):
    source_type: DocumentSource = DocumentSource.OTHER
    document_date: Optional[str] = None
    guideline_version: Optional[str] = None
    title: Optional[str] = None
    tags: List[str] = []


class IngestResponse(BaseModel):
    success: bool
    document_id: str
    chunks_created: int
    message: str


class ConversationMessage(BaseModel):
    role: str  # "user" or "assistant"
    content: str
    timestamp: datetime = Field(default_factory=datetime.utcnow)


class ConversationHistory(BaseModel):
    conversation_id: str
    messages: List[ConversationMessage] = []


# LangGraph State
class MedicalRAGState(BaseModel):
    query: str
    conversation_id: Optional[str] = None
    conversation_history: List[Dict[str, Any]] = []
    query_embedding: Optional[List[float]] = None
    retrieved_docs: List[Dict[str, Any]] = []
    ranked_docs: List[Dict[str, Any]] = []
    filtered_docs: List[Dict[str, Any]] = []
    generated_answer: Optional[str] = None
    citations: List[SourceCitation] = []
    confidence_score: float = 0.0
    is_safe: bool = True
    hallucination_detected: bool = False
    highlighted_contexts: List[str] = []
    error: Optional[str] = None
    needs_more_retrieval: bool = False
    retrieval_attempts: int = 0

    class Config:
        arbitrary_types_allowed = True