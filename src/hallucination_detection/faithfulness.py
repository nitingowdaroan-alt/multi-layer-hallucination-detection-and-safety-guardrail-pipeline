"""
Layer 1: Retrieval Faithfulness Scoring

Evaluates how well extracted values are grounded in the retrieved evidence.
Uses semantic similarity and textual matching to score faithfulness.
"""

import re
from typing import List, Optional, Tuple
from dataclasses import dataclass

from ..core.models import RetrievedChunk, FaithfulnessResult, FieldType
from ..core.config import HallucinationDetectionConfig
from ..rag.embeddings import EmbeddingService


@dataclass
class FaithfulnessMetrics:
    """Detailed faithfulness metrics."""
    semantic_similarity: float
    lexical_overlap: float
    exact_match_found: bool
    partial_match_score: float
    grounding_strength: str  # "strong", "moderate", "weak", "none"


class FaithfulnessScorer:
    """
    Scores the faithfulness of extracted values to source evidence.

    Faithfulness scoring determines if an extracted value can be traced
    back to the retrieved clinical text. High faithfulness means the
    value is explicitly supported by the evidence.

    Scoring Components:
    1. Semantic similarity between extracted value and evidence
    2. Lexical/token overlap
    3. Exact match detection
    4. Partial match scoring for numeric values
    """

    def __init__(
        self,
        embedding_service: EmbeddingService,
        config: Optional[HallucinationDetectionConfig] = None
    ):
        self.embedding_service = embedding_service
        self.config = config or HallucinationDetectionConfig()

    def score_faithfulness(
        self,
        extracted_value: str,
        supporting_chunks: List[RetrievedChunk],
        field_name: str,
        field_type: FieldType
    ) -> FaithfulnessResult:
        """
        Score the faithfulness of an extracted value.

        Args:
            extracted_value: The value extracted by the LLM
            supporting_chunks: Chunks used as evidence
            field_name: Name of the extracted field
            field_type: Type of the field

        Returns:
            FaithfulnessResult with scores and evidence
        """
        if not extracted_value or not supporting_chunks:
            return FaithfulnessResult(
                field_name=field_name,
                faithfulness_score=0.0,
                supporting_chunks=[],
                max_chunk_similarity=0.0,
                avg_chunk_similarity=0.0,
                grounding_evidence="",
                is_grounded=False
            )

        # Get detailed metrics for each chunk
        chunk_metrics = []
        for chunk in supporting_chunks:
            metrics = self._compute_chunk_metrics(
                extracted_value,
                chunk,
                field_type
            )
            chunk_metrics.append((chunk, metrics))

        # Find best matching chunk
        best_chunk, best_metrics = max(
            chunk_metrics,
            key=lambda x: x[1].semantic_similarity
        )

        # Compute aggregate scores
        similarities = [m.semantic_similarity for _, m in chunk_metrics]
        max_similarity = max(similarities)
        avg_similarity = sum(similarities) / len(similarities)

        # Compute final faithfulness score
        faithfulness_score = self._compute_faithfulness_score(
            best_metrics,
            field_type
        )

        # Determine if value is grounded
        is_grounded = (
            faithfulness_score >= self.config.faithfulness_threshold or
            best_metrics.exact_match_found
        )

        # Extract grounding evidence
        grounding_evidence = self._extract_grounding_evidence(
            extracted_value,
            best_chunk.content,
            field_type
        )

        return FaithfulnessResult(
            field_name=field_name,
            faithfulness_score=faithfulness_score,
            supporting_chunks=supporting_chunks,
            max_chunk_similarity=max_similarity,
            avg_chunk_similarity=avg_similarity,
            grounding_evidence=grounding_evidence,
            is_grounded=is_grounded
        )

    def _compute_chunk_metrics(
        self,
        extracted_value: str,
        chunk: RetrievedChunk,
        field_type: FieldType
    ) -> FaithfulnessMetrics:
        """Compute faithfulness metrics for a single chunk."""
        content = chunk.content.lower()
        value_lower = extracted_value.lower()

        # 1. Semantic similarity
        value_embedding = self.embedding_service.get_embedding(extracted_value)
        chunk_embedding = chunk.embedding or self.embedding_service.get_embedding(
            chunk.content
        ).embedding

        semantic_similarity = self.embedding_service.compute_similarity(
            value_embedding.embedding,
            chunk_embedding
        )

        # 2. Lexical overlap
        lexical_overlap = self._compute_lexical_overlap(value_lower, content)

        # 3. Exact match detection
        exact_match_found = self._check_exact_match(
            extracted_value,
            chunk.content,
            field_type
        )

        # 4. Partial match scoring (especially for numbers)
        partial_match_score = self._compute_partial_match(
            extracted_value,
            chunk.content,
            field_type
        )

        # Determine grounding strength
        if exact_match_found:
            grounding_strength = "strong"
        elif partial_match_score > 0.8:
            grounding_strength = "moderate"
        elif semantic_similarity > 0.7:
            grounding_strength = "moderate"
        elif semantic_similarity > 0.5:
            grounding_strength = "weak"
        else:
            grounding_strength = "none"

        return FaithfulnessMetrics(
            semantic_similarity=semantic_similarity,
            lexical_overlap=lexical_overlap,
            exact_match_found=exact_match_found,
            partial_match_score=partial_match_score,
            grounding_strength=grounding_strength
        )

    def _compute_lexical_overlap(self, value: str, content: str) -> float:
        """Compute token overlap between value and content."""
        value_tokens = set(re.findall(r'\b\w+\b', value))
        content_tokens = set(re.findall(r'\b\w+\b', content))

        if not value_tokens:
            return 0.0

        overlap = len(value_tokens & content_tokens)
        return overlap / len(value_tokens)

    def _check_exact_match(
        self,
        value: str,
        content: str,
        field_type: FieldType
    ) -> bool:
        """Check if value appears exactly in content."""
        value_normalized = self._normalize_value(value, field_type)
        content_normalized = self._normalize_value(content, field_type)

        return value_normalized in content_normalized

    def _normalize_value(self, text: str, field_type: FieldType) -> str:
        """Normalize text for matching based on field type."""
        text = text.lower().strip()

        if field_type in [FieldType.DOSAGE, FieldType.LAB_VALUE]:
            # Normalize numeric representations
            text = re.sub(r'\s+', '', text)  # Remove spaces
            text = re.sub(r'(\d),(\d)', r'\1\2', text)  # Remove numeric commas

        return text

    def _compute_partial_match(
        self,
        value: str,
        content: str,
        field_type: FieldType
    ) -> float:
        """Compute partial match score, especially for numeric values."""
        if field_type in [FieldType.DOSAGE, FieldType.LAB_VALUE, FieldType.VITAL_SIGN]:
            # Extract numbers from both
            value_numbers = re.findall(r'\d+\.?\d*', value)
            content_numbers = re.findall(r'\d+\.?\d*', content)

            if not value_numbers:
                return 0.0

            matches = 0
            for vn in value_numbers:
                if vn in content_numbers:
                    matches += 1

            return matches / len(value_numbers)

        elif field_type == FieldType.MEDICATION:
            # Check if medication name appears in content
            words = value.lower().split()
            content_lower = content.lower()

            matches = sum(1 for w in words if w in content_lower)
            return matches / len(words) if words else 0.0

        else:
            # General partial match
            return self._compute_lexical_overlap(value.lower(), content.lower())

    def _compute_faithfulness_score(
        self,
        metrics: FaithfulnessMetrics,
        field_type: FieldType
    ) -> float:
        """Compute final faithfulness score from metrics."""
        # Weights depend on field type
        if field_type in [FieldType.DOSAGE, FieldType.LAB_VALUE]:
            # For numeric fields, exact/partial match is most important
            weights = {
                "semantic": 0.2,
                "lexical": 0.2,
                "exact": 0.4,
                "partial": 0.2
            }
        elif field_type in [FieldType.MEDICATION, FieldType.DIAGNOSIS]:
            # For clinical terms, lexical and semantic both matter
            weights = {
                "semantic": 0.35,
                "lexical": 0.35,
                "exact": 0.15,
                "partial": 0.15
            }
        else:
            # Default weights
            weights = {
                "semantic": 0.3,
                "lexical": 0.3,
                "exact": 0.2,
                "partial": 0.2
            }

        exact_score = 1.0 if metrics.exact_match_found else 0.0

        score = (
            metrics.semantic_similarity * weights["semantic"] +
            metrics.lexical_overlap * weights["lexical"] +
            exact_score * weights["exact"] +
            metrics.partial_match_score * weights["partial"]
        )

        return min(1.0, max(0.0, score))

    def _extract_grounding_evidence(
        self,
        value: str,
        content: str,
        field_type: FieldType
    ) -> str:
        """Extract the specific text that grounds the value."""
        value_lower = value.lower()
        content_lower = content.lower()

        # Try to find exact occurrence
        idx = content_lower.find(value_lower)
        if idx >= 0:
            # Return with context
            start = max(0, idx - 50)
            end = min(len(content), idx + len(value) + 50)
            return content[start:end]

        # Try to find key terms
        if field_type in [FieldType.DOSAGE, FieldType.LAB_VALUE]:
            numbers = re.findall(r'\d+\.?\d*', value)
            for num in numbers:
                idx = content.find(num)
                if idx >= 0:
                    start = max(0, idx - 30)
                    end = min(len(content), idx + len(num) + 30)
                    return content[start:end]

        # Return first relevant sentence
        sentences = content.split('.')
        for sentence in sentences:
            if any(word in sentence.lower() for word in value_lower.split()[:3]):
                return sentence.strip()

        return content[:200] if content else ""

    def score_multiple_fields(
        self,
        extractions: List[Tuple[str, str, List[RetrievedChunk], FieldType]],
    ) -> List[FaithfulnessResult]:
        """
        Score faithfulness for multiple extracted fields.

        Args:
            extractions: List of (field_name, extracted_value, chunks, field_type)

        Returns:
            List of FaithfulnessResults
        """
        results = []
        for field_name, value, chunks, field_type in extractions:
            result = self.score_faithfulness(value, chunks, field_name, field_type)
            results.append(result)
        return results
