"""
ChromaDB vector store with temporal ranking support.
"""
import uuid
from datetime import datetime
from typing import List, Dict, Any, Optional
import chromadb
from chromadb.config import Settings as ChromaSettings
import structlog

from app.config import settings
from app.core.embeddings import embedding_service

logger = structlog.get_logger()


class VectorStore:
    """ChromaDB-backed vector store with metadata and temporal ranking."""

    _instance = None

    def __new__(cls):
        if cls._instance is None:
            cls._instance = super().__new__(cls)
            cls._instance._initialized = False
        return cls._instance

    def __init__(self):
        if not self._initialized:
            self._setup_client()
            self._initialized = True

    def _setup_client(self):
        logger.info("Initializing ChromaDB", persist_dir=settings.chroma_persist_dir)
        self.client = chromadb.PersistentClient(
            path=settings.chroma_persist_dir,
            settings=ChromaSettings(anonymized_telemetry=False),
        )
        self.collection = self.client.get_or_create_collection(
            name=settings.chroma_collection_name,
            metadata={"hnsw:space": "cosine"},
        )
        logger.info(
            "ChromaDB initialized",
            collection=settings.chroma_collection_name,
            count=self.collection.count(),
        )

    def add_documents(self, chunks: List[Dict[str, Any]]) -> List[str]:
        """
        Add document chunks to the vector store.

        Each chunk: {text, metadata: {source, page, section, document_date, ...}}
        """
        ids = [str(uuid.uuid4()) for _ in chunks]
        texts = [c["text"] for c in chunks]
        metadatas = [c.get("metadata", {}) for c in chunks]

        # Normalize metadata - ChromaDB requires string/int/float/bool
        clean_metadatas = []
        for m in metadatas:
            clean = {}
            for k, v in m.items():
                if v is None:
                    clean[k] = ""
                elif isinstance(v, (str, int, float, bool)):
                    clean[k] = v
                else:
                    clean[k] = str(v)
            clean_metadatas.append(clean)

        embeddings = embedding_service.embed_batch(texts)

        self.collection.add(
            ids=ids,
            documents=texts,
            embeddings=embeddings,
            metadatas=clean_metadatas,
        )
        logger.info("Added documents to vector store", count=len(chunks))
        return ids

    def similarity_search(
        self,
        query: str,
        top_k: int = 5,
        filters: Optional[Dict] = None,
    ) -> List[Dict[str, Any]]:
        """Semantic similarity search."""
        query_embedding = embedding_service.embed_text(query)

        kwargs = {
            "query_embeddings": [query_embedding],
            "n_results": min(top_k * 2, self.collection.count() or 1),
            "include": ["documents", "metadatas", "distances"],
        }
        if filters:
            kwargs["where"] = filters

        results = self.collection.query(**kwargs)

        docs = []
        for i, (doc, meta, dist) in enumerate(
            zip(
                results["documents"][0],
                results["metadatas"][0],
                results["distances"][0],
            )
        ):
            # Convert cosine distance to similarity
            similarity = 1 - dist
            docs.append(
                {
                    "text": doc,
                    "metadata": meta,
                    "similarity_score": similarity,
                    "rank": i + 1,
                }
            )
        return docs

    def temporal_rerank(
        self, docs: List[Dict[str, Any]], recency_weight: float = 0.2
    ) -> List[Dict[str, Any]]:
        """
        Re-rank documents combining semantic similarity + temporal recency.

        Score = (1 - recency_weight) * semantic_score + recency_weight * recency_score
        """
        now = datetime.utcnow()

        for doc in docs:
            semantic_score = doc["similarity_score"]
            doc_date_str = doc["metadata"].get("document_date", "")

            recency_score = 0.5  # default neutral
            if doc_date_str:
                try:
                    doc_date = datetime.fromisoformat(doc_date_str[:10])
                    age_days = (now - doc_date).days
                    # Normalize: 0 days = 1.0, 3650 days (10 years) = 0.0
                    recency_score = max(0.0, 1.0 - (age_days / 3650))
                except (ValueError, TypeError):
                    recency_score = 0.5

            combined = (
                (1 - recency_weight) * semantic_score + recency_weight * recency_score
            )
            doc["recency_score"] = recency_score
            doc["combined_score"] = combined

        return sorted(docs, key=lambda x: x["combined_score"], reverse=True)

    def get_collection_stats(self) -> Dict[str, Any]:
        return {
            "total_documents": self.collection.count(),
            "collection_name": settings.chroma_collection_name,
        }

    def delete_collection(self):
        """Reset collection (for testing)."""
        self.client.delete_collection(settings.chroma_collection_name)
        self.collection = self.client.get_or_create_collection(
            name=settings.chroma_collection_name,
            metadata={"hnsw:space": "cosine"},
        )


vector_store = VectorStore()