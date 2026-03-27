"""
LLM provider abstraction — supports OpenAI, Ollama, and Groq.
"""
import structlog
from app.config import settings

logger = structlog.get_logger()


def get_llm(temperature: float = 0.1):
    provider = settings.llm_provider.lower()
    if provider == "openai":
        from langchain_openai import ChatOpenAI
        return ChatOpenAI(model=settings.llm_model, temperature=temperature,
                          openai_api_key=settings.openai_api_key)
    elif provider == "ollama":
        from langchain_community.chat_models import ChatOllama
        return ChatOllama(model=settings.ollama_model, base_url=settings.ollama_base_url,
                          temperature=temperature)
    elif provider == "groq":
        from langchain_groq import ChatGroq
        return ChatGroq(model=settings.llm_model or "llama-3.3-70b-versatile",
                        temperature=temperature, api_key=settings.groq_api_key)
    else:
        raise ValueError(f"Unsupported LLM_PROVIDER: '{provider}'. Choose: openai, ollama, groq")


# ── Prompts ───────────────────────────────────────────────────────────────────

MEDICAL_QA_PROMPT = """You are a trusted medical information assistant backed by WHO, PubMed, and NIH sources.

Answer the user's health question clearly, thoroughly, and helpfully using the provided sources.

STRICT RULES:
1. Use the provided SOURCE DOCUMENTS as your primary evidence.
2. Structure your answer:
   - Direct answer in the first sentence
   - Detailed explanation with sections if needed
   - Key facts as bullet points
   - Recommended next steps / when to see a doctor
3. Cite sources inline e.g. (Source 1), (Source 2) for key claims.
4. If sources are partial, supplement with your medical knowledge but LABEL IT clearly as "Based on medical knowledge:".
5. Always be helpful — never refuse to answer a health question.
6. Use plain, accessible language.
7. Do NOT recommend specific dosages as personal medical advice.

SOURCE DOCUMENTS:
{context}

CONVERSATION HISTORY:
{history}

USER QUESTION: {question}

Give a comprehensive, well-structured medical answer:"""


ACCURACY_CHECK_PROMPT = """You are a medical accuracy evaluator. Analyse this health answer and score it.

SOURCES USED:
{context}

GENERATED ANSWER:
{answer}

Evaluate carefully and respond ONLY with a JSON object — no extra text, no markdown:
{{
  "is_supported": true,
  "accuracy_score": 0.85,
  "support_level": "high",
  "unsupported_claims": [],
  "well_supported_claims": ["claim 1", "claim 2"],
  "explanation": "Brief explanation of accuracy assessment"
}}

Rules for scoring:
- accuracy_score: 0.0-1.0 (how well the answer is backed by sources)
- support_level: "high" (>0.75), "medium" (0.5-0.75), "low" (<0.5)
- is_supported: true if all major claims have source backing
- unsupported_claims: list any claims NOT found in sources
- well_supported_claims: list 2-3 strongest supported claims"""


SAFETY_CHECK_PROMPT = """You are a medical safety reviewer.

ANSWER TO CHECK:
{answer}

Check for dangerous content and respond ONLY with a JSON object — no markdown, no extra text:
{{
  "is_safe": true,
  "severity": "none",
  "concerns": [],
  "safe_summary": "Answer appears safe for general health information purposes."
}}

Severity levels:
- "none": No safety concerns
- "low": Minor caveats (e.g. general advice that should be personalised)
- "medium": Contains specific dosage recommendations or treatment instructions
- "high": Could cause direct harm if followed without medical supervision"""