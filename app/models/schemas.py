"""Pydantic schemas for request/response models."""
from pydantic import BaseModel, Field
from typing import Optional, List, Dict, Any
from datetime import datetime
from enum import Enum


class DocumentSource(str, Enum):
    WHO_GUIDELINES   = "WHO Guidelines"
    PUBMED           = "PubMed"
    MEDICAL_TEXTBOOK = "Medical Textbook"
    CLINICAL_RESEARCH= "Clinical Research"
    LAB_REPORT       = "Lab Report"
    OTHER            = "Other"


class SourceCitation(BaseModel):
    document_id:       str
    source:            str
    organization:      Optional[str] = None
    page:              Optional[int] = None
    section:           Optional[str] = None
    document_date:     Optional[str] = None
    guideline_version: Optional[str] = None
    excerpt:           str
    relevance_score:   float
    source_url:        Optional[str] = None
    is_web_source:     bool = False
    web_source_name:   Optional[str] = None   # "PubMed", "WHO", "MedlinePlus", "FDA"


class MedicalAnswer(BaseModel):
    answer:              str
    citations:           List[SourceCitation]
    confidence_score:    float = Field(ge=0.0, le=1.0)
    # Accuracy breakdown
    accuracy_score:      float = 0.0
    support_level:       str   = "medium"    # high / medium / low / none
    unsupported_claims:  List[str] = []
    well_supported_claims: List[str] = []
    accuracy_explanation:  str = ""
    # Safety
    is_safe:             bool = True
    safety_severity:     str  = "none"       # none / low / medium / high
    safety_concerns:     List[str] = []
    safety_disclaimer:   str = (
        "⚠️ This information is for educational purposes only. "
        "Always consult a qualified healthcare professional for personal medical advice."
    )
    # Hallucination
    hallucination_detected: bool = False
    # Context
    highlighted_contexts: List[str] = []
    web_fallback_used:    bool = False
    query_timestamp:      datetime = Field(default_factory=datetime.utcnow)


class QueryRequest(BaseModel):
    query:                   str  = Field(..., min_length=3, max_length=2000)
    conversation_id:         Optional[str] = None
    top_k:                   int  = Field(default=5, ge=1, le=20)
    include_temporal_ranking:bool = True
    language:                str  = "en"


class IngestRequest(BaseModel):
    source_type:       DocumentSource = DocumentSource.OTHER
    document_date:     Optional[str]  = None
    guideline_version: Optional[str]  = None
    title:             Optional[str]  = None
    tags:              List[str]      = []


class IngestResponse(BaseModel):
    success:       bool
    document_id:   str
    chunks_created:int
    message:       str


class ConversationMessage(BaseModel):
    role:      str
    content:   str
    timestamp: datetime = Field(default_factory=datetime.utcnow)


class ConversationHistory(BaseModel):
    conversation_id: str
    messages:        List[ConversationMessage] = []


class MedicalRAGState(BaseModel):
    query:              str
    conversation_id:    Optional[str]        = None
    query_embedding:    Optional[List[float]]= None
    retrieved_docs:     List[Dict[str, Any]] = []
    ranked_docs:        List[Dict[str, Any]] = []
    filtered_docs:      List[Dict[str, Any]] = []
    generated_answer:   Optional[str]        = None
    citations:          List[SourceCitation] = []
    confidence_score:   float = 0.0
    accuracy_score:     float = 0.0
    support_level:      str   = "medium"
    unsupported_claims: List[str] = []
    well_supported_claims: List[str] = []
    accuracy_explanation:  str   = ""
    is_safe:            bool  = True
    safety_severity:    str   = "none"
    safety_concerns:    List[str] = []
    hallucination_detected: bool = False
    highlighted_contexts: List[str] = []
    needs_more_retrieval: bool = False
    retrieval_attempts:   int  = 0
    web_fallback_used:    bool = False
    has_context:          bool = True
    error:                Optional[str] = None

    class Config:
        arbitrary_types_allowed = True