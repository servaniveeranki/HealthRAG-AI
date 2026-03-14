"""
LLM provider abstraction — supports OpenAI, Ollama, and Groq.
"""
import structlog
from app.config import settings

logger = structlog.get_logger()


def get_llm(temperature: float = 0.1):
    """Return LLM instance based on LLM_PROVIDER in .env"""

    provider = settings.llm_provider.lower()

    if provider == "openai":
        from langchain_openai import ChatOpenAI
        return ChatOpenAI(
            model=settings.llm_model,
            temperature=temperature,
            openai_api_key=settings.openai_api_key,
        )

    elif provider == "ollama":
        from langchain_community.chat_models import ChatOllama
        return ChatOllama(
            model=settings.ollama_model,
            base_url=settings.ollama_base_url,
            temperature=temperature,
        )

    elif provider == "groq":
        from langchain_groq import ChatGroq
        return ChatGroq(
            model=settings.llm_model or "llama3-8b-8192",
            temperature=temperature,
            api_key=settings.groq_api_key,
        )

    else:
        raise ValueError(
            f"Unsupported LLM_PROVIDER: '{provider}'. "
            "Choose from: openai, ollama, groq"
        )


# ── Prompt templates ──────────────────────────────────────────────────────────

MEDICAL_QA_PROMPT = """You are a knowledgeable medical information assistant. Answer the user's medical question using ONLY the provided context from verified medical documents.

STRICT RULES:
1. Base your answer ONLY on the provided context. Do not add information from outside.
2. If the context does not contain enough information, say so clearly.
3. Always mention which source supports each claim.
4. Use clear, accessible language while maintaining medical accuracy.
5. Do NOT recommend specific medications or dosages without noting they come from the source.

CONTEXT FROM MEDICAL DOCUMENTS:
{context}

CONVERSATION HISTORY:
{history}

USER QUESTION: {question}

Provide a comprehensive, evidence-based answer with source references:"""


HALLUCINATION_CHECK_PROMPT = """You are a medical fact-checker. Determine whether the generated answer is fully supported by the retrieved context.

RETRIEVED CONTEXT:
{context}

GENERATED ANSWER:
{answer}

Analyze whether every claim in the answer is supported by the context.
Respond with ONLY a JSON object (no extra text):
{{
  "is_supported": true,
  "unsupported_claims": [],
  "confidence": 0.9,
  "explanation": "brief explanation"
}}"""


SAFETY_CHECK_PROMPT = """You are a medical safety reviewer. Check if this response contains dangerous medical advice.

RESPONSE TO CHECK:
{answer}

Respond with ONLY a JSON object (no extra text):
{{
  "is_safe": true,
  "concerns": [],
  "severity": "none"
}}"""