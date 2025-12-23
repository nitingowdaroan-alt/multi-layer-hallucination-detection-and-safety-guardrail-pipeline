"""
Embedding Service for Clinical Document Retrieval

Provides embedding generation for clinical text chunks using
OpenAI or other embedding models.
"""

import hashlib
from typing import List, Optional, Dict, Any
from dataclasses import dataclass
import numpy as np

from ..core.config import ModelConfig


@dataclass
class EmbeddingResult:
    """Result of embedding generation."""
    text: str
    embedding: List[float]
    model: str
    dimension: int
    cache_key: str


class EmbeddingService:
    """
    Service for generating text embeddings.

    Supports:
    - OpenAI embeddings (text-embedding-3-small, text-embedding-3-large)
    - Caching for repeated queries
    - Batch processing for efficiency
    """

    def __init__(
        self,
        config: Optional[ModelConfig] = None,
        openai_client: Optional[Any] = None
    ):
        self.config = config or ModelConfig()
        self.model = self.config.embedding_model
        self._client = openai_client
        self._cache: Dict[str, List[float]] = {}

    @property
    def client(self):
        """Lazy initialization of OpenAI client."""
        if self._client is None:
            try:
                from openai import OpenAI
                self._client = OpenAI()
            except ImportError:
                raise ImportError("OpenAI package required. Install with: pip install openai")
        return self._client

    def get_embedding(self, text: str, use_cache: bool = True) -> EmbeddingResult:
        """
        Generate embedding for a single text.

        Args:
            text: Text to embed
            use_cache: Whether to use cached embeddings

        Returns:
            EmbeddingResult with the embedding vector
        """
        cache_key = self._get_cache_key(text)

        if use_cache and cache_key in self._cache:
            return EmbeddingResult(
                text=text,
                embedding=self._cache[cache_key],
                model=self.model,
                dimension=len(self._cache[cache_key]),
                cache_key=cache_key
            )

        # Call OpenAI API
        response = self.client.embeddings.create(
            model=self.model,
            input=text,
            encoding_format="float"
        )

        embedding = response.data[0].embedding

        if use_cache:
            self._cache[cache_key] = embedding

        return EmbeddingResult(
            text=text,
            embedding=embedding,
            model=self.model,
            dimension=len(embedding),
            cache_key=cache_key
        )

    def get_embeddings_batch(
        self,
        texts: List[str],
        use_cache: bool = True,
        batch_size: int = 100
    ) -> List[EmbeddingResult]:
        """
        Generate embeddings for multiple texts efficiently.

        Args:
            texts: List of texts to embed
            use_cache: Whether to use cached embeddings
            batch_size: Maximum texts per API call

        Returns:
            List of EmbeddingResults
        """
        results = []
        uncached_texts = []
        uncached_indices = []

        # Check cache first
        for i, text in enumerate(texts):
            cache_key = self._get_cache_key(text)
            if use_cache and cache_key in self._cache:
                results.append((i, EmbeddingResult(
                    text=text,
                    embedding=self._cache[cache_key],
                    model=self.model,
                    dimension=len(self._cache[cache_key]),
                    cache_key=cache_key
                )))
            else:
                uncached_texts.append(text)
                uncached_indices.append(i)

        # Batch process uncached texts
        for batch_start in range(0, len(uncached_texts), batch_size):
            batch_texts = uncached_texts[batch_start:batch_start + batch_size]
            batch_indices = uncached_indices[batch_start:batch_start + batch_size]

            response = self.client.embeddings.create(
                model=self.model,
                input=batch_texts,
                encoding_format="float"
            )

            for j, embedding_data in enumerate(response.data):
                text = batch_texts[j]
                embedding = embedding_data.embedding
                cache_key = self._get_cache_key(text)

                if use_cache:
                    self._cache[cache_key] = embedding

                results.append((batch_indices[j], EmbeddingResult(
                    text=text,
                    embedding=embedding,
                    model=self.model,
                    dimension=len(embedding),
                    cache_key=cache_key
                )))

        # Sort by original index
        results.sort(key=lambda x: x[0])
        return [r[1] for r in results]

    def compute_similarity(
        self,
        embedding1: List[float],
        embedding2: List[float]
    ) -> float:
        """
        Compute cosine similarity between two embeddings.

        Args:
            embedding1: First embedding vector
            embedding2: Second embedding vector

        Returns:
            Cosine similarity score (0-1)
        """
        vec1 = np.array(embedding1)
        vec2 = np.array(embedding2)

        # Cosine similarity
        dot_product = np.dot(vec1, vec2)
        norm1 = np.linalg.norm(vec1)
        norm2 = np.linalg.norm(vec2)

        if norm1 == 0 or norm2 == 0:
            return 0.0

        return float(dot_product / (norm1 * norm2))

    def compute_similarities_batch(
        self,
        query_embedding: List[float],
        document_embeddings: List[List[float]]
    ) -> List[float]:
        """
        Compute similarities between a query and multiple documents.

        Args:
            query_embedding: Query embedding vector
            document_embeddings: List of document embedding vectors

        Returns:
            List of similarity scores
        """
        query_vec = np.array(query_embedding)
        query_norm = np.linalg.norm(query_vec)

        if query_norm == 0:
            return [0.0] * len(document_embeddings)

        similarities = []
        for doc_embedding in document_embeddings:
            doc_vec = np.array(doc_embedding)
            doc_norm = np.linalg.norm(doc_vec)

            if doc_norm == 0:
                similarities.append(0.0)
            else:
                similarity = float(np.dot(query_vec, doc_vec) / (query_norm * doc_norm))
                similarities.append(similarity)

        return similarities

    def _get_cache_key(self, text: str) -> str:
        """Generate cache key for text."""
        return hashlib.md5(f"{self.model}:{text}".encode()).hexdigest()

    def clear_cache(self):
        """Clear the embedding cache."""
        self._cache.clear()

    def get_cache_stats(self) -> Dict[str, int]:
        """Get cache statistics."""
        return {
            "cached_embeddings": len(self._cache),
            "estimated_memory_mb": len(self._cache) * 1536 * 4 / (1024 * 1024)  # Rough estimate
        }
