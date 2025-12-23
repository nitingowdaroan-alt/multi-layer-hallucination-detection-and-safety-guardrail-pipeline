"""
Clinical Document Retriever

Retrieves relevant chunks from clinical documents based on
extraction queries with clinical-aware ranking.
"""

from typing import List, Optional, Dict, Any, Tuple
from dataclasses import dataclass
import heapq

from ..core.models import RetrievedChunk, ExtractionField, FieldType
from ..core.config import RAGConfig
from .embeddings import EmbeddingService


@dataclass
class RetrievalQuery:
    """Query for retrieving clinical evidence."""
    field: ExtractionField
    query_text: str
    context_hints: Optional[List[str]] = None
    required_section_types: Optional[List[str]] = None


@dataclass
class RetrievalResult:
    """Result of a retrieval operation."""
    query: RetrievalQuery
    chunks: List[RetrievedChunk]
    total_chunks_searched: int
    retrieval_time_ms: float


class ClinicalRetriever:
    """
    Retrieves relevant clinical text chunks for extraction.

    Implements:
    - Semantic similarity search
    - Section-type filtering for clinical relevance
    - Multi-query retrieval for comprehensive coverage
    - Re-ranking based on clinical context
    """

    # Mapping of field types to preferred section types
    FIELD_SECTION_MAPPING = {
        FieldType.MEDICATION: ["medications", "plan", "discharge"],
        FieldType.DOSAGE: ["medications", "plan"],
        FieldType.DIAGNOSIS: ["diagnosis", "assessment", "history"],
        FieldType.LAB_VALUE: ["laboratory", "examination"],
        FieldType.PROCEDURE: ["plan", "history"],
        FieldType.ALLERGY: ["allergies", "medications"],
        FieldType.VITAL_SIGN: ["examination", "vitals"],
        FieldType.DATE: ["general", "history", "plan"],
        FieldType.PROVIDER: ["general"],
        FieldType.FACILITY: ["general"],
    }

    def __init__(
        self,
        embedding_service: EmbeddingService,
        config: Optional[RAGConfig] = None
    ):
        self.embedding_service = embedding_service
        self.config = config or RAGConfig()
        self._chunk_index: Dict[str, List[RetrievedChunk]] = {}
        self._chunk_embeddings: Dict[str, List[float]] = {}

    def index_chunks(
        self,
        document_id: str,
        chunks: List[RetrievedChunk]
    ) -> int:
        """
        Index document chunks for retrieval.

        Args:
            document_id: ID of the source document
            chunks: List of chunks to index

        Returns:
            Number of chunks indexed
        """
        # Store chunks
        self._chunk_index[document_id] = chunks

        # Generate embeddings for all chunks
        chunk_texts = [chunk.content for chunk in chunks]
        embedding_results = self.embedding_service.get_embeddings_batch(chunk_texts)

        # Store embeddings
        for chunk, embedding_result in zip(chunks, embedding_results):
            chunk.embedding = embedding_result.embedding
            self._chunk_embeddings[chunk.chunk_id] = embedding_result.embedding

        return len(chunks)

    def retrieve(
        self,
        document_id: str,
        query: RetrievalQuery,
        top_k: Optional[int] = None
    ) -> RetrievalResult:
        """
        Retrieve relevant chunks for a query.

        Args:
            document_id: ID of the document to search
            query: Retrieval query with field information
            top_k: Number of chunks to retrieve

        Returns:
            RetrievalResult with ranked chunks
        """
        import time
        start_time = time.time()

        top_k = top_k or self.config.top_k_chunks
        chunks = self._chunk_index.get(document_id, [])

        if not chunks:
            return RetrievalResult(
                query=query,
                chunks=[],
                total_chunks_searched=0,
                retrieval_time_ms=0.0
            )

        # Generate query embedding
        query_embedding = self.embedding_service.get_embedding(query.query_text)

        # Filter chunks by section type if specified
        candidate_chunks = self._filter_by_section(
            chunks,
            query.field.field_type,
            query.required_section_types
        )

        # Compute similarities
        chunk_embeddings = [
            self._chunk_embeddings.get(c.chunk_id, c.embedding)
            for c in candidate_chunks
        ]

        similarities = self.embedding_service.compute_similarities_batch(
            query_embedding.embedding,
            chunk_embeddings
        )

        # Score and rank chunks
        scored_chunks = []
        for chunk, similarity in zip(candidate_chunks, similarities):
            # Apply section boost
            section_boost = self._get_section_boost(
                chunk.metadata.get("section_type"),
                query.field.field_type
            )

            # Apply content boost
            content_boost = self._get_content_boost(chunk, query.field.field_type)

            # Final score
            final_score = similarity * (1 + section_boost + content_boost)
            chunk.similarity_score = final_score
            scored_chunks.append((final_score, chunk))

        # Get top-k chunks
        top_chunks = heapq.nlargest(top_k, scored_chunks, key=lambda x: x[0])
        result_chunks = [chunk for score, chunk in top_chunks]

        # Filter by minimum threshold
        result_chunks = [
            chunk for chunk in result_chunks
            if chunk.similarity_score >= self.config.min_similarity_threshold
        ]

        end_time = time.time()

        return RetrievalResult(
            query=query,
            chunks=result_chunks,
            total_chunks_searched=len(candidate_chunks),
            retrieval_time_ms=(end_time - start_time) * 1000
        )

    def retrieve_multi_query(
        self,
        document_id: str,
        queries: List[RetrievalQuery],
        top_k_per_query: Optional[int] = None,
        deduplicate: bool = True
    ) -> Dict[str, RetrievalResult]:
        """
        Retrieve chunks for multiple queries.

        Args:
            document_id: ID of the document to search
            queries: List of retrieval queries
            top_k_per_query: Chunks per query
            deduplicate: Whether to deduplicate across queries

        Returns:
            Dictionary mapping field names to results
        """
        results = {}
        seen_chunk_ids = set()

        for query in queries:
            result = self.retrieve(document_id, query, top_k_per_query)

            if deduplicate:
                # Remove already-seen chunks
                unique_chunks = [
                    chunk for chunk in result.chunks
                    if chunk.chunk_id not in seen_chunk_ids
                ]
                seen_chunk_ids.update(c.chunk_id for c in unique_chunks)
                result.chunks = unique_chunks

            results[query.field.name] = result

        return results

    def retrieve_for_field(
        self,
        document_id: str,
        field: ExtractionField,
        additional_context: Optional[str] = None
    ) -> RetrievalResult:
        """
        Retrieve chunks for a specific extraction field.

        Generates an optimized query based on field type.
        """
        # Generate query text based on field type
        query_text = self._generate_field_query(field, additional_context)

        query = RetrievalQuery(
            field=field,
            query_text=query_text,
            required_section_types=self.FIELD_SECTION_MAPPING.get(field.field_type)
        )

        return self.retrieve(document_id, query)

    def _filter_by_section(
        self,
        chunks: List[RetrievedChunk],
        field_type: FieldType,
        required_sections: Optional[List[str]] = None
    ) -> List[RetrievedChunk]:
        """Filter chunks by section type relevance."""
        if required_sections:
            preferred_sections = required_sections
        else:
            preferred_sections = self.FIELD_SECTION_MAPPING.get(field_type, [])

        if not preferred_sections:
            return chunks

        # Prioritize preferred sections but include all
        preferred = []
        other = []

        for chunk in chunks:
            section_type = chunk.metadata.get("section_type", "general")
            if section_type in preferred_sections:
                preferred.append(chunk)
            else:
                other.append(chunk)

        return preferred + other

    def _get_section_boost(
        self,
        section_type: Optional[str],
        field_type: FieldType
    ) -> float:
        """Get section relevance boost."""
        if not section_type:
            return 0.0

        preferred_sections = self.FIELD_SECTION_MAPPING.get(field_type, [])

        if section_type in preferred_sections:
            # Higher boost for more specific matches
            position = preferred_sections.index(section_type)
            return 0.2 - (position * 0.05)

        return 0.0

    def _get_content_boost(
        self,
        chunk: RetrievedChunk,
        field_type: FieldType
    ) -> float:
        """Get content-based relevance boost."""
        boost = 0.0
        metadata = chunk.metadata

        if field_type == FieldType.MEDICATION and metadata.get("has_medications"):
            boost += 0.1
        elif field_type == FieldType.LAB_VALUE and metadata.get("has_lab_values"):
            boost += 0.1
        elif field_type == FieldType.DIAGNOSIS and metadata.get("has_diagnoses"):
            boost += 0.1
        elif field_type == FieldType.DATE and metadata.get("has_dates"):
            boost += 0.05

        return boost

    def _generate_field_query(
        self,
        field: ExtractionField,
        additional_context: Optional[str] = None
    ) -> str:
        """Generate an optimized query for a field."""
        base_queries = {
            FieldType.MEDICATION: f"medications prescribed {field.name}",
            FieldType.DOSAGE: f"medication dosage amount {field.name}",
            FieldType.DIAGNOSIS: f"diagnosis assessment {field.name}",
            FieldType.LAB_VALUE: f"laboratory test result {field.name}",
            FieldType.PROCEDURE: f"procedure performed {field.name}",
            FieldType.ALLERGY: f"patient allergies {field.name}",
            FieldType.VITAL_SIGN: f"vital signs {field.name}",
            FieldType.DATE: f"date {field.name}",
            FieldType.PROVIDER: f"physician provider {field.name}",
            FieldType.FACILITY: f"hospital facility {field.name}",
        }

        query = base_queries.get(field.field_type, field.name)

        if additional_context:
            query = f"{query} {additional_context}"

        return query

    def clear_index(self, document_id: Optional[str] = None):
        """Clear indexed chunks."""
        if document_id:
            self._chunk_index.pop(document_id, None)
            # Clear associated embeddings
            if document_id in self._chunk_index:
                for chunk in self._chunk_index[document_id]:
                    self._chunk_embeddings.pop(chunk.chunk_id, None)
        else:
            self._chunk_index.clear()
            self._chunk_embeddings.clear()

    def get_index_stats(self) -> Dict[str, Any]:
        """Get indexing statistics."""
        total_chunks = sum(len(chunks) for chunks in self._chunk_index.values())
        return {
            "indexed_documents": len(self._chunk_index),
            "total_chunks": total_chunks,
            "cached_embeddings": len(self._chunk_embeddings),
        }
