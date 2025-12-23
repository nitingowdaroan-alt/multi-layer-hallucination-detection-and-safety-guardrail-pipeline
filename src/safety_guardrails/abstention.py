"""
Abstention Engine

Implements abstention logic for clinical data extraction.
Determines when to abstain from providing a value and generates
clear, auditable abstention reasons.
"""

from typing import Optional, Dict, List, Any
from dataclasses import dataclass
from enum import Enum

from ..core.models import (
    ConfidenceScore,
    HallucinationDetectionResult,
    DecisionType,
    FieldType,
    SeverityLevel,
)
from ..core.config import ConfidenceConfig


class AbstentionReason(Enum):
    """Enumeration of abstention reasons."""
    INSUFFICIENT_EVIDENCE = "insufficient_evidence"
    CONTRADICTION_DETECTED = "contradiction_detected"
    RULE_VIOLATION = "rule_violation"
    LOW_CONFIDENCE = "low_confidence"
    HIGH_VARIANCE = "high_variance"
    NO_GROUNDING = "no_grounding"
    AMBIGUOUS_SOURCE = "ambiguous_source"
    SAFETY_CONCERN = "safety_concern"


@dataclass
class AbstentionResult:
    """Result of abstention decision."""
    should_abstain: bool
    reason: Optional[AbstentionReason]
    reason_detail: str
    confidence_score: float
    evidence_summary: str
    recommendation: str


class AbstentionEngine:
    """
    Determines when to abstain from providing extracted values.

    Abstention is a SAFETY MECHANISM for clinical AI:
    - It's better to say "I don't know" than to hallucinate
    - Clear abstention reasons help clinicians understand limitations
    - All abstentions are logged for audit purposes

    Abstention triggers:
    1. Insufficient evidence in source text
    2. Contradiction between extraction and evidence
    3. Clinical rule violations
    4. Low confidence scores
    5. High extraction variance (instability)
    6. Safety concerns for critical fields
    """

    def __init__(self, config: Optional[ConfidenceConfig] = None):
        self.config = config or ConfidenceConfig()

        # Abstention thresholds
        self.min_confidence_threshold = config.review_threshold if config else 0.60
        self.min_faithfulness_threshold = 0.40
        self.max_variance_threshold = 0.50
        self.min_agreement_threshold = 0.50

    def should_abstain(
        self,
        confidence: ConfidenceScore,
        detection_result: HallucinationDetectionResult,
        field_type: FieldType,
        severity: SeverityLevel
    ) -> AbstentionResult:
        """
        Determine if abstention is warranted.

        Args:
            confidence: Computed confidence score
            detection_result: Hallucination detection results
            field_type: Type of the field
            severity: Severity level of the field

        Returns:
            AbstentionResult with decision and reasoning
        """
        # Check each abstention condition in priority order
        abstention_checks = [
            self._check_rule_violation(detection_result, severity),
            self._check_contradiction(detection_result, severity),
            self._check_insufficient_evidence(detection_result),
            self._check_no_grounding(detection_result),
            self._check_high_variance(detection_result),
            self._check_low_confidence(confidence, severity),
            self._check_safety_concerns(detection_result, field_type, severity),
        ]

        for result in abstention_checks:
            if result.should_abstain:
                return result

        # No abstention warranted
        return AbstentionResult(
            should_abstain=False,
            reason=None,
            reason_detail="Value meets all confidence and safety thresholds",
            confidence_score=confidence.final_score,
            evidence_summary=self._get_evidence_summary(detection_result),
            recommendation="accept"
        )

    def _check_rule_violation(
        self,
        result: HallucinationDetectionResult,
        severity: SeverityLevel
    ) -> AbstentionResult:
        """Check for clinical rule violations."""
        rules = result.rule_validation

        if not rules.passed:
            violations_str = "; ".join(rules.violations[:3])
            return AbstentionResult(
                should_abstain=True,
                reason=AbstentionReason.RULE_VIOLATION,
                reason_detail=f"Clinical rule violations: {violations_str}",
                confidence_score=0.0,
                evidence_summary="Rule validation failed",
                recommendation="abstain_due_to_rule_violation"
            )

        return AbstentionResult(
            should_abstain=False,
            reason=None,
            reason_detail="",
            confidence_score=0.0,
            evidence_summary="",
            recommendation=""
        )

    def _check_contradiction(
        self,
        result: HallucinationDetectionResult,
        severity: SeverityLevel
    ) -> AbstentionResult:
        """Check for contradictions with evidence."""
        contradiction = result.contradiction

        if contradiction.contradiction_detected:
            if contradiction.evidence_alignment == "contradicted":
                return AbstentionResult(
                    should_abstain=True,
                    reason=AbstentionReason.CONTRADICTION_DETECTED,
                    reason_detail=f"Extracted value contradicts source: {contradiction.explanation}",
                    confidence_score=0.0,
                    evidence_summary="Contradiction detected",
                    recommendation="abstain_due_to_contradiction"
                )
            elif contradiction.evidence_alignment == "unsupported":
                # For critical fields, unsupported = abstain
                if severity in [SeverityLevel.CRITICAL, SeverityLevel.HIGH]:
                    return AbstentionResult(
                        should_abstain=True,
                        reason=AbstentionReason.INSUFFICIENT_EVIDENCE,
                        reason_detail="Value not supported by source evidence",
                        confidence_score=0.0,
                        evidence_summary="No supporting evidence found",
                        recommendation="abstain_due_to_lack_of_evidence"
                    )

        return AbstentionResult(
            should_abstain=False,
            reason=None,
            reason_detail="",
            confidence_score=0.0,
            evidence_summary="",
            recommendation=""
        )

    def _check_insufficient_evidence(
        self,
        result: HallucinationDetectionResult
    ) -> AbstentionResult:
        """Check for insufficient evidence."""
        faithfulness = result.faithfulness

        if not faithfulness.supporting_chunks:
            return AbstentionResult(
                should_abstain=True,
                reason=AbstentionReason.INSUFFICIENT_EVIDENCE,
                reason_detail="No supporting evidence chunks found",
                confidence_score=0.0,
                evidence_summary="No evidence retrieved",
                recommendation="abstain_no_evidence"
            )

        if faithfulness.faithfulness_score < self.min_faithfulness_threshold:
            return AbstentionResult(
                should_abstain=True,
                reason=AbstentionReason.INSUFFICIENT_EVIDENCE,
                reason_detail=f"Low faithfulness score: {faithfulness.faithfulness_score:.2f}",
                confidence_score=faithfulness.faithfulness_score,
                evidence_summary=faithfulness.grounding_evidence[:200],
                recommendation="abstain_weak_evidence"
            )

        return AbstentionResult(
            should_abstain=False,
            reason=None,
            reason_detail="",
            confidence_score=0.0,
            evidence_summary="",
            recommendation=""
        )

    def _check_no_grounding(
        self,
        result: HallucinationDetectionResult
    ) -> AbstentionResult:
        """Check for lack of textual grounding."""
        faithfulness = result.faithfulness

        if not faithfulness.is_grounded:
            return AbstentionResult(
                should_abstain=True,
                reason=AbstentionReason.NO_GROUNDING,
                reason_detail="Extracted value not grounded in source text",
                confidence_score=faithfulness.faithfulness_score,
                evidence_summary="Value not found in evidence",
                recommendation="abstain_ungrounded"
            )

        return AbstentionResult(
            should_abstain=False,
            reason=None,
            reason_detail="",
            confidence_score=0.0,
            evidence_summary="",
            recommendation=""
        )

    def _check_high_variance(
        self,
        result: HallucinationDetectionResult
    ) -> AbstentionResult:
        """Check for high extraction variance."""
        consistency = result.consistency

        if consistency.variance_score > self.max_variance_threshold:
            unique_count = len(consistency.unique_values)
            return AbstentionResult(
                should_abstain=True,
                reason=AbstentionReason.HIGH_VARIANCE,
                reason_detail=f"High extraction variance: {unique_count} different values across runs",
                confidence_score=consistency.agreement_rate,
                evidence_summary=f"Values varied: {', '.join(consistency.unique_values[:3])}",
                recommendation="abstain_unstable"
            )

        if consistency.agreement_rate < self.min_agreement_threshold:
            return AbstentionResult(
                should_abstain=True,
                reason=AbstentionReason.HIGH_VARIANCE,
                reason_detail=f"Low agreement rate: {consistency.agreement_rate:.2f}",
                confidence_score=consistency.agreement_rate,
                evidence_summary="Extraction results inconsistent",
                recommendation="abstain_low_agreement"
            )

        return AbstentionResult(
            should_abstain=False,
            reason=None,
            reason_detail="",
            confidence_score=0.0,
            evidence_summary="",
            recommendation=""
        )

    def _check_low_confidence(
        self,
        confidence: ConfidenceScore,
        severity: SeverityLevel
    ) -> AbstentionResult:
        """Check for low overall confidence."""
        threshold = self.min_confidence_threshold

        # Stricter threshold for critical fields
        if severity == SeverityLevel.CRITICAL:
            threshold = max(threshold, 0.70)
        elif severity == SeverityLevel.HIGH:
            threshold = max(threshold, 0.65)

        if confidence.final_score < threshold:
            return AbstentionResult(
                should_abstain=True,
                reason=AbstentionReason.LOW_CONFIDENCE,
                reason_detail=f"Confidence {confidence.final_score:.2f} below threshold {threshold:.2f}",
                confidence_score=confidence.final_score,
                evidence_summary="Overall confidence too low",
                recommendation="abstain_low_confidence"
            )

        return AbstentionResult(
            should_abstain=False,
            reason=None,
            reason_detail="",
            confidence_score=0.0,
            evidence_summary="",
            recommendation=""
        )

    def _check_safety_concerns(
        self,
        result: HallucinationDetectionResult,
        field_type: FieldType,
        severity: SeverityLevel
    ) -> AbstentionResult:
        """Check for safety concerns on critical fields."""
        # For critical fields, extra safety checks
        if severity == SeverityLevel.CRITICAL:
            # Medication and allergy fields
            if field_type in [FieldType.MEDICATION, FieldType.ALLERGY, FieldType.DOSAGE]:
                # Require higher confidence
                if result.composite_score < 0.80:
                    return AbstentionResult(
                        should_abstain=True,
                        reason=AbstentionReason.SAFETY_CONCERN,
                        reason_detail=f"Safety-critical field requires higher confidence",
                        confidence_score=result.composite_score,
                        evidence_summary="Critical field below safety threshold",
                        recommendation="abstain_safety_critical"
                    )

        return AbstentionResult(
            should_abstain=False,
            reason=None,
            reason_detail="",
            confidence_score=0.0,
            evidence_summary="",
            recommendation=""
        )

    def _get_evidence_summary(
        self,
        result: HallucinationDetectionResult
    ) -> str:
        """Get summary of supporting evidence."""
        evidence = result.faithfulness.grounding_evidence
        if evidence:
            return evidence[:300]
        return "Evidence available in source text"

    def get_abstention_message(
        self,
        abstention_result: AbstentionResult,
        field_name: str
    ) -> str:
        """Generate user-facing abstention message."""
        if not abstention_result.should_abstain:
            return ""

        reason_messages = {
            AbstentionReason.INSUFFICIENT_EVIDENCE:
                f"Unable to extract '{field_name}': Insufficient evidence in source document.",
            AbstentionReason.CONTRADICTION_DETECTED:
                f"Unable to extract '{field_name}': Detected contradiction in source.",
            AbstentionReason.RULE_VIOLATION:
                f"Unable to extract '{field_name}': Value failed clinical validation.",
            AbstentionReason.LOW_CONFIDENCE:
                f"Unable to extract '{field_name}': Confidence below acceptable threshold.",
            AbstentionReason.HIGH_VARIANCE:
                f"Unable to extract '{field_name}': Extraction results unstable.",
            AbstentionReason.NO_GROUNDING:
                f"Unable to extract '{field_name}': Value not found in source text.",
            AbstentionReason.AMBIGUOUS_SOURCE:
                f"Unable to extract '{field_name}': Source text is ambiguous.",
            AbstentionReason.SAFETY_CONCERN:
                f"Unable to extract '{field_name}': Safety threshold not met.",
        }

        base_message = reason_messages.get(
            abstention_result.reason,
            f"Unable to extract '{field_name}': Extraction not possible."
        )

        return f"{base_message}\nDetail: {abstention_result.reason_detail}"
