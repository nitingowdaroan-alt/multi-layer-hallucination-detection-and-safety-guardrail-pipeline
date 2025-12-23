"""
Confidence Scoring System

Computes final confidence scores for extracted fields based on
hallucination detection results and applies decision thresholds.
"""

from typing import Optional, Dict, List
from dataclasses import dataclass

from ..core.models import (
    ConfidenceScore,
    HallucinationDetectionResult,
    DecisionType,
    FieldType,
    SeverityLevel,
)
from ..core.config import ConfidenceConfig


@dataclass
class ConfidenceBreakdown:
    """Detailed breakdown of confidence components."""
    retrieval_component: float
    verifier_component: float
    consistency_component: float
    rule_component: float
    severity_adjustment: float
    final_score: float


class ConfidenceScorer:
    """
    Computes confidence scores for clinical extractions.

    The confidence score is computed from multiple signals:
    1. Retrieval faithfulness - How well is the value grounded?
    2. Verifier agreement - Does the verifier confirm the value?
    3. Self-consistency - Is the extraction stable?
    4. Rule validation - Do clinical rules pass?

    Critical fields (medications, allergies, diagnoses) have
    stricter thresholds applied.
    """

    def __init__(self, config: Optional[ConfidenceConfig] = None):
        self.config = config or ConfidenceConfig()

    def compute_confidence(
        self,
        detection_result: HallucinationDetectionResult,
        field_type: FieldType,
        severity: SeverityLevel
    ) -> ConfidenceScore:
        """
        Compute confidence score from detection results.

        Args:
            detection_result: Results from hallucination detection
            field_type: Type of the extracted field
            severity: Severity level of the field

        Returns:
            ConfidenceScore with computed values and decision
        """
        # Extract component scores
        retrieval_score = self._compute_retrieval_component(detection_result)
        verifier_score = self._compute_verifier_component(detection_result)
        consistency_score = self._compute_consistency_component(detection_result)
        rule_score = self._compute_rule_component(detection_result)

        # Create confidence score object
        confidence = ConfidenceScore(
            field_name=detection_result.field_name,
            retrieval_faithfulness=retrieval_score,
            verifier_agreement=verifier_score,
            self_consistency=consistency_score,
            rule_validation=rule_score,
            weights=self.config.weights,
            accept_threshold=self._get_adjusted_accept_threshold(severity),
            review_threshold=self._get_adjusted_review_threshold(severity),
        )

        # Compute final score
        confidence.compute_final_score()

        # Determine decision
        confidence.determine_decision()

        # Apply severity override
        confidence = self._apply_severity_overrides(
            confidence, detection_result, severity
        )

        return confidence

    def _compute_retrieval_component(
        self,
        result: HallucinationDetectionResult
    ) -> float:
        """Compute retrieval faithfulness component."""
        faithfulness = result.faithfulness

        # Base score from faithfulness
        score = faithfulness.faithfulness_score

        # Bonus for strong grounding
        if faithfulness.is_grounded:
            score = min(1.0, score + 0.1)

        # Penalty for low similarity
        if faithfulness.max_chunk_similarity < 0.3:
            score *= 0.8

        return min(1.0, max(0.0, score))

    def _compute_verifier_component(
        self,
        result: HallucinationDetectionResult
    ) -> float:
        """Compute verifier agreement component."""
        contradiction = result.contradiction

        if contradiction.contradiction_detected:
            # Contradiction is a strong negative signal
            if contradiction.evidence_alignment == "contradicted":
                return 0.0
            else:  # unsupported
                return 0.2
        else:
            # No contradiction - use verifier confidence
            if contradiction.evidence_alignment == "aligned":
                return contradiction.verifier_confidence
            else:
                return 0.5  # Neutral

    def _compute_consistency_component(
        self,
        result: HallucinationDetectionResult
    ) -> float:
        """Compute self-consistency component."""
        consistency = result.consistency

        # Base score from agreement rate
        score = consistency.agreement_rate

        # Penalty for high variance
        if consistency.variance_score > 0.3:
            score *= (1 - consistency.variance_score)

        # Penalty for confidence decay
        score *= (1 - consistency.confidence_decay)

        return min(1.0, max(0.0, score))

    def _compute_rule_component(
        self,
        result: HallucinationDetectionResult
    ) -> float:
        """Compute rule validation component."""
        rules = result.rule_validation

        if not rules.passed:
            # Rule violation - score based on severity
            if rules.violations:
                return 0.0  # Hard violations
            else:
                return 0.3  # Only warnings
        else:
            # Rules passed
            if rules.warnings:
                return 0.8  # Passed with warnings
            else:
                return 1.0  # Clean pass

    def _get_adjusted_accept_threshold(self, severity: SeverityLevel) -> float:
        """Get accept threshold adjusted for field severity."""
        base = self.config.accept_threshold
        boost = self.config.critical_field_threshold_boost

        if severity == SeverityLevel.CRITICAL:
            return min(0.95, base + boost * 2)
        elif severity == SeverityLevel.HIGH:
            return min(0.92, base + boost)
        else:
            return base

    def _get_adjusted_review_threshold(self, severity: SeverityLevel) -> float:
        """Get review threshold adjusted for field severity."""
        base = self.config.review_threshold

        if severity == SeverityLevel.CRITICAL:
            return min(0.75, base + 0.10)
        elif severity == SeverityLevel.HIGH:
            return min(0.70, base + 0.05)
        else:
            return base

    def _apply_severity_overrides(
        self,
        confidence: ConfidenceScore,
        detection_result: HallucinationDetectionResult,
        severity: SeverityLevel
    ) -> ConfidenceScore:
        """Apply severity-based decision overrides."""
        # For critical fields, any hallucination indicator triggers rejection
        if severity == SeverityLevel.CRITICAL:
            if detection_result.is_hallucinated:
                confidence.decision = DecisionType.REJECT

            # Rule violations on critical fields are always rejected
            if not detection_result.rule_validation.passed:
                confidence.decision = DecisionType.REJECT

            # Even accepted critical fields should be flagged if not perfect
            if (confidence.decision == DecisionType.ACCEPT and
                    confidence.final_score < 0.95):
                confidence.decision = DecisionType.FLAG_FOR_REVIEW

        # For high severity, contradictions trigger rejection
        elif severity == SeverityLevel.HIGH:
            if detection_result.contradiction.contradiction_detected:
                if detection_result.contradiction.evidence_alignment == "contradicted":
                    confidence.decision = DecisionType.REJECT

        return confidence

    def compute_batch(
        self,
        results: List[HallucinationDetectionResult],
        field_types: List[FieldType],
        severities: List[SeverityLevel]
    ) -> List[ConfidenceScore]:
        """Compute confidence scores for multiple fields."""
        scores = []
        for result, field_type, severity in zip(results, field_types, severities):
            score = self.compute_confidence(result, field_type, severity)
            scores.append(score)
        return scores

    def get_confidence_explanation(
        self,
        confidence: ConfidenceScore
    ) -> str:
        """Generate human-readable explanation of confidence score."""
        explanation_parts = []

        # Overall assessment
        if confidence.final_score >= 0.9:
            explanation_parts.append("HIGH CONFIDENCE: Strong evidence support.")
        elif confidence.final_score >= 0.7:
            explanation_parts.append("MODERATE CONFIDENCE: Reasonable evidence support.")
        elif confidence.final_score >= 0.5:
            explanation_parts.append("LOW CONFIDENCE: Weak evidence support.")
        else:
            explanation_parts.append("VERY LOW CONFIDENCE: Insufficient evidence.")

        # Component breakdown
        explanation_parts.append("\nComponent scores:")
        explanation_parts.append(
            f"  - Retrieval faithfulness: {confidence.retrieval_faithfulness:.2f}"
        )
        explanation_parts.append(
            f"  - Verifier agreement: {confidence.verifier_agreement:.2f}"
        )
        explanation_parts.append(
            f"  - Self-consistency: {confidence.self_consistency:.2f}"
        )
        explanation_parts.append(
            f"  - Rule validation: {confidence.rule_validation:.2f}"
        )

        # Decision explanation
        explanation_parts.append(f"\nDecision: {confidence.decision.value}")
        explanation_parts.append(
            f"Thresholds: Accept >= {confidence.accept_threshold:.2f}, "
            f"Review >= {confidence.review_threshold:.2f}"
        )

        return "\n".join(explanation_parts)
