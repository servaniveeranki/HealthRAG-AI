"""
Embedding generation using SentenceTransformers.
"""
import numpy as np
from typing import List, Union
from sentence_transformers import SentenceTransformer
import structlog

from app.config import settings

logger = structlog.get_logger()


class EmbeddingService:
    """Manages embedding generation with SentenceTransformers."""

    _instance = None
    _model = None

    def __new__(cls):
        if cls._instance is None:
            cls._instance = super().__new__(cls)
        return cls._instance

    def __init__(self):
        if self._model is None:
            self._load_model()

    def _load_model(self):
        logger.info("Loading embedding model", model=settings.embedding_model)
        self._model = SentenceTransformer(settings.embedding_model)
        logger.info("Embedding model loaded successfully")

    def embed_text(self, text: str) -> List[float]:
        """Generate embedding for a single text."""
        embedding = self._model.encode(text, convert_to_numpy=True)
        return embedding.tolist()

    def embed_batch(self, texts: List[str]) -> List[List[float]]:
        """Generate embeddings for a batch of texts."""
        embeddings = self._model.encode(texts, convert_to_numpy=True, batch_size=32)
        return embeddings.tolist()

    def cosine_similarity(self, vec1: List[float], vec2: List[float]) -> float:
        """Compute cosine similarity between two vectors."""
        a = np.array(vec1)
        b = np.array(vec2)
        norm_a = np.linalg.norm(a)
        norm_b = np.linalg.norm(b)
        if norm_a == 0 or norm_b == 0:
            return 0.0
        return float(np.dot(a, b) / (norm_a * norm_b))

    def compute_confidence(
        self, query_embedding: List[float], doc_embeddings: List[List[float]]
    ) -> float:
        """Compute average confidence from top doc similarities."""
        if not doc_embeddings:
            return 0.0
        similarities = [
            self.cosine_similarity(query_embedding, doc_emb)
            for doc_emb in doc_embeddings
        ]
        top_k = min(3, len(similarities))
        top_scores = sorted(similarities, reverse=True)[:top_k]
        return float(np.mean(top_scores))


embedding_service = EmbeddingService()