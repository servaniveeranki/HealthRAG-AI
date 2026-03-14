"""
Configuration management for Medical RAG System.
"""
from pydantic_settings import BaseSettings
from typing import Optional


class Settings(BaseSettings):
    # LLM
    openai_api_key: Optional[str] = None
    ollama_base_url: str = "http://localhost:11434"
    ollama_model: str = "mistral"
    llm_provider: str = "openai"
    llm_model: str = "gpt-4o-mini"
    groq_api_key: Optional[str] = None
    # ChromaDB
    chroma_persist_dir: str = "./data/chroma_db"
    chroma_collection_name: str = "medical_knowledge"

    # Embeddings
    embedding_model: str = "all-MiniLM-L6-v2"

    # App
    app_host: str = "0.0.0.0"
    app_port: int = 8000
    log_level: str = "INFO"

    # Safety
    enable_safety_filter: bool = True
    min_confidence_threshold: float = 0.3

    # RAG params
    retrieval_top_k: int = 5
    chunk_size: int = 500
    chunk_overlap: int = 50

    # Cache
    redis_url: Optional[str] = None
    enable_cache: bool = False

    class Config:
        env_file = ".env"
        extra = "ignore"


settings = Settings()