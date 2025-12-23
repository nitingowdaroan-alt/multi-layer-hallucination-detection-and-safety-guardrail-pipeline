"""
Explainability Generator

Generates human-readable explanations for extraction decisions,
confidence scores, and hallucination detections.
"""

from typing import Optional, List, Dict, Any
from dataclasses import dataclass

from ..core.models import (
    SafetyDecision,
    ConfidenceScore,
    HallucinationDetectionResult,
    DecisionType,
    FieldType,
)


@dataclass
class ExplanationOutput:
    """Structured explanation output."""
    summary: str
    confidence_explanation: str
    evidence_summary: str
    decision_rationale: str
    recommendations: List[str]
    technical_details: Dict[str, Any]


class ExplainabilityGenerator:
    """
    Generates human-readable explanations for system outputs.

    All outputs include:
    - Summary of the decision
    - Confidence score breakdown
    - Supporting evidence
    - Recommendations for action
    """

    def explain_decision(
        self,
        decision: SafetyDecision,
        detection_result: Optional[HallucinationDetectionResult] = None
    ) -> ExplanationOutput:
        """
        Generate a comprehensive explanation for a decision.

        Args:
            decision: The safety decision made
            detection_result: Optional hallucination detection result

        Returns:
            ExplanationOutput with detailed explanation
        """
        # Generate summary
        summary = self._generate_summary(decision)

        # Confidence explanation
        confidence_explanation = self._explain_confidence(decision.confidence)

        # Evidence summary
        evidence_summary = self._summarize_evidence(decision.evidence_snippets)

        # Decision rationale
        decision_rationale = self._explain_rationale(decision, detection_result)

        # Recommendations
        recommendations = self._generate_recommendations(decision)

        # Technical details
        technical_details = self._compile_technical_details(decision, detection_result)

        return ExplanationOutput(
            summary=summary,
            confidence_explanation=confidence_explanation,
            evidence_summary=evidence_summary,
            decision_rationale=decision_rationale,
            recommendations=recommendations,
            technical_details=technical_details
        )

    def _generate_summary(self, decision: SafetyDecision) -> str:
        """Generate a summary statement."""
        decision_descriptions = {
            DecisionType.ACCEPT: f"✓ ACCEPTED: The extracted value '{decision.extracted_value}' has been accepted with high confidence.",
            DecisionType.FLAG_FOR_REVIEW: f"⚠ FLAGGED: The extraction requires human review before use.",
            DecisionType.ABSTAIN: f"○ ABSTAINED: Unable to reliably extract this field.",
            DecisionType.REJECT: f"✗ REJECTED: The extraction was blocked due to safety concerns.",
        }
        return decision_descriptions.get(decision.decision, "Decision status unknown.")

    def _explain_confidence(self, confidence: ConfidenceScore) -> str:
        """Explain the confidence score breakdown."""
        lines = [
            f"Overall Confidence: {confidence.final_score:.1%}",
            "",
            "Component Breakdown:",
            f"  • Evidence Grounding: {confidence.retrieval_faithfulness:.1%}",
            f"    (How well the value is supported by source text)",
            "",
            f"  • Verification Score: {confidence.verifier_agreement:.1%}",
            f"    (Independent verification of the extraction)",
            "",
            f"  • Consistency Score: {confidence.self_consistency:.1%}",
            f"    (Stability across multiple extraction attempts)",
            "",
            f"  • Rule Compliance: {confidence.rule_validation:.1%}",
            f"    (Adherence to clinical validation rules)",
        ]

        # Add threshold context
        lines.extend([
            "",
            "Decision Thresholds:",
            f"  • Accept: ≥ {confidence.accept_threshold:.0%}",
            f"  • Review: ≥ {confidence.review_threshold:.0%}",
            f"  • Abstain: < {confidence.review_threshold:.0%}",
        ])

        return "\n".join(lines)

    def _summarize_evidence(self, evidence_snippets: List[str]) -> str:
        """Summarize the supporting evidence."""
        if not evidence_snippets:
            return "No supporting evidence was found in the source document."

        lines = ["Supporting Evidence from Source Document:", ""]

        for i, snippet in enumerate(evidence_snippets[:3], 1):
            # Clean and truncate snippet
            clean_snippet = snippet.strip()[:200]
            if len(snippet) > 200:
                clean_snippet += "..."
            lines.append(f"  [{i}] \"{clean_snippet}\"")
            lines.append("")

        if len(evidence_snippets) > 3:
            lines.append(f"  ... and {len(evidence_snippets) - 3} more evidence snippets")

        return "\n".join(lines)

    def _explain_rationale(
        self,
        decision: SafetyDecision,
        detection_result: Optional[HallucinationDetectionResult]
    ) -> str:
        """Explain the rationale for the decision."""
        lines = ["Decision Rationale:", ""]

        if decision.decision == DecisionType.ACCEPT:
            lines.extend([
                "The extraction was accepted because:",
                "  • Value was found explicitly in the source document",
                "  • Independent verification confirmed the extraction",
                "  • Multiple extraction attempts produced consistent results",
                "  • All clinical validation rules were satisfied",
            ])

        elif decision.decision == DecisionType.FLAG_FOR_REVIEW:
            lines.extend([
                "The extraction was flagged for review because:",
                f"  • Confidence score ({decision.confidence.final_score:.1%}) is below acceptance threshold",
            ])
            if decision.confidence.retrieval_faithfulness < 0.7:
                lines.append("  • Evidence grounding is weak")
            if decision.confidence.self_consistency < 0.8:
                lines.append("  • Extraction results showed some variation")
            if decision.is_critical_field:
                lines.append("  • This is a critical field requiring higher confidence")

        elif decision.decision == DecisionType.ABSTAIN:
            lines.extend([
                "The system abstained from extraction because:",
            ])
            if decision.abstention_reason:
                lines.append(f"  • {decision.abstention_reason}")
            else:
                lines.append("  • Insufficient evidence to support a reliable extraction")

        elif decision.decision == DecisionType.REJECT:
            lines.extend([
                "The extraction was rejected because:",
            ])
            if decision.abstention_reason:
                lines.append(f"  • {decision.abstention_reason}")
            if detection_result and detection_result.is_hallucinated:
                lines.append(f"  • Hallucination detected: {detection_result.hallucination_type}")

        return "\n".join(lines)

    def _generate_recommendations(self, decision: SafetyDecision) -> List[str]:
        """Generate actionable recommendations."""
        recommendations = []

        if decision.decision == DecisionType.ACCEPT:
            recommendations.append("The extracted value can be used as-is.")
            if decision.is_critical_field:
                recommendations.append(
                    "As this is a critical field, consider a quick visual verification."
                )

        elif decision.decision == DecisionType.FLAG_FOR_REVIEW:
            recommendations.append("A clinician should review this extraction before use.")
            recommendations.append("Check the source document for the relevant section.")
            if decision.evidence_snippets:
                recommendations.append(
                    "Review the provided evidence snippets against the original document."
                )

        elif decision.decision == DecisionType.ABSTAIN:
            recommendations.append("Manual extraction from the source document is required.")
            recommendations.append(
                "The system could not find reliable information for this field."
            )

        elif decision.decision == DecisionType.REJECT:
            recommendations.append("Do NOT use any value for this field from this extraction.")
            recommendations.append("Manual extraction from the source document is required.")
            recommendations.append("Review the rejection reason to understand the issue.")

        return recommendations

    def _compile_technical_details(
        self,
        decision: SafetyDecision,
        detection_result: Optional[HallucinationDetectionResult]
    ) -> Dict[str, Any]:
        """Compile technical details for debugging/audit."""
        details = {
            "field_name": decision.field_name,
            "decision_type": decision.decision.value,
            "is_critical_field": decision.is_critical_field,
            "requires_human_review": decision.requires_human_review,
            "timestamp": decision.decision_timestamp.isoformat(),
            "confidence": {
                "final_score": decision.confidence.final_score,
                "retrieval_faithfulness": decision.confidence.retrieval_faithfulness,
                "verifier_agreement": decision.confidence.verifier_agreement,
                "self_consistency": decision.confidence.self_consistency,
                "rule_validation": decision.confidence.rule_validation,
                "accept_threshold": decision.confidence.accept_threshold,
                "review_threshold": decision.confidence.review_threshold,
            },
        }

        if detection_result:
            details["hallucination_detection"] = {
                "is_hallucinated": detection_result.is_hallucinated,
                "hallucination_type": detection_result.hallucination_type,
                "composite_score": detection_result.composite_score,
                "faithfulness_score": detection_result.faithfulness.faithfulness_score,
                "is_grounded": detection_result.faithfulness.is_grounded,
                "contradiction_detected": detection_result.contradiction.contradiction_detected,
                "consistency_agreement": detection_result.consistency.agreement_rate,
                "rule_validation_passed": detection_result.rule_validation.passed,
            }

        return details

    def generate_batch_summary(
        self,
        decisions: List[SafetyDecision]
    ) -> str:
        """Generate a summary for multiple decisions."""
        total = len(decisions)
        accepted = sum(1 for d in decisions if d.decision == DecisionType.ACCEPT)
        flagged = sum(1 for d in decisions if d.decision == DecisionType.FLAG_FOR_REVIEW)
        abstained = sum(1 for d in decisions if d.decision == DecisionType.ABSTAIN)
        rejected = sum(1 for d in decisions if d.decision == DecisionType.REJECT)

        lines = [
            "=" * 60,
            "EXTRACTION SUMMARY",
            "=" * 60,
            "",
            f"Total Fields Processed: {total}",
            "",
            f"  ✓ Accepted:    {accepted:3d} ({accepted/total*100:5.1f}%)",
            f"  ⚠ Flagged:     {flagged:3d} ({flagged/total*100:5.1f}%)",
            f"  ○ Abstained:   {abstained:3d} ({abstained/total*100:5.1f}%)",
            f"  ✗ Rejected:    {rejected:3d} ({rejected/total*100:5.1f}%)",
            "",
        ]

        # Add critical field summary
        critical_fields = [d for d in decisions if d.is_critical_field]
        if critical_fields:
            critical_accepted = sum(1 for d in critical_fields if d.decision == DecisionType.ACCEPT)
            lines.extend([
                f"Critical Fields: {len(critical_fields)}",
                f"  Accepted: {critical_accepted} / {len(critical_fields)}",
                "",
            ])

        # Add fields requiring review
        review_fields = [d for d in decisions if d.requires_human_review]
        if review_fields:
            lines.extend([
                "Fields Requiring Human Review:",
            ])
            for d in review_fields[:5]:
                lines.append(f"  • {d.field_name}")
            if len(review_fields) > 5:
                lines.append(f"  ... and {len(review_fields) - 5} more")
            lines.append("")

        lines.append("=" * 60)
        return "\n".join(lines)

    def format_for_clinical_review(
        self,
        decision: SafetyDecision
    ) -> str:
        """Format explanation for clinical review interface."""
        lines = [
            f"Field: {decision.field_name}",
            f"Value: {decision.extracted_value or '[No value extracted]'}",
            f"Status: {decision.decision.value.upper()}",
            f"Confidence: {decision.confidence.final_score:.0%}",
            "",
        ]

        if decision.abstention_reason:
            lines.append(f"Reason: {decision.abstention_reason}")
            lines.append("")

        if decision.evidence_snippets:
            lines.append("Evidence:")
            for snippet in decision.evidence_snippets[:2]:
                lines.append(f"  \"{snippet[:150]}...\"")

        return "\n".join(lines)
