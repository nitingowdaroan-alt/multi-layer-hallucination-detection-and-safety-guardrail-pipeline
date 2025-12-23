"""
Safety Decision Engine

Orchestrates all safety components to make final decisions
about extracted values.
"""

from typing import Optional, List, Dict, Any
from dataclasses import dataclass, field
from datetime import datetime

from ..core.models import (
    SafetyDecision,
    ConfidenceScore,
    HallucinationDetectionResult,
    DecisionType,
    FieldType,
    SeverityLevel,
    ExtractionField,
    AuditLog,
)
from ..core.config import SystemConfig
from .confidence import ConfidenceScorer
from .abstention import AbstentionEngine, AbstentionResult
from .healthcare_safety import HealthcareSafetyGuard, SafetyCheckResult


@dataclass
class DecisionContext:
    """Context for making a safety decision."""
    field: ExtractionField
    extracted_value: Optional[str]
    detection_result: HallucinationDetectionResult
    source_text: str
    extraction_reasoning: Optional[str] = None


@dataclass
class DecisionOutcome:
    """Complete outcome of the decision process."""
    decision: SafetyDecision
    confidence: ConfidenceScore
    abstention: AbstentionResult
    safety_check: SafetyCheckResult
    audit_log: AuditLog
    processing_notes: List[str] = field(default_factory=list)


class SafetyDecisionEngine:
    """
    Makes final safety decisions for clinical extractions.

    This engine:
    1. Computes confidence scores
    2. Checks abstention conditions
    3. Runs healthcare safety checks
    4. Makes final accept/review/abstain/reject decision
    5. Generates audit-ready documentation

    Decision priority:
    1. Safety violations → REJECT
    2. Abstention conditions → ABSTAIN
    3. Low confidence → FLAG_FOR_REVIEW
    4. All checks pass → ACCEPT
    """

    def __init__(self, config: Optional[SystemConfig] = None):
        self.config = config or SystemConfig()

        self.confidence_scorer = ConfidenceScorer(self.config.confidence)
        self.abstention_engine = AbstentionEngine(self.config.confidence)
        self.safety_guard = HealthcareSafetyGuard(self.config.safety)

    def make_decision(
        self,
        context: DecisionContext,
        request_id: str,
        document_id: str
    ) -> DecisionOutcome:
        """
        Make a safety decision for an extracted field.

        Args:
            context: Decision context with all relevant data
            request_id: Request ID for audit
            document_id: Document ID for audit

        Returns:
            DecisionOutcome with complete decision information
        """
        processing_notes = []

        # Step 1: Compute confidence score
        confidence = self.confidence_scorer.compute_confidence(
            context.detection_result,
            context.field.field_type,
            context.field.severity
        )
        processing_notes.append(f"Confidence computed: {confidence.final_score:.3f}")

        # Step 2: Check abstention conditions
        abstention = self.abstention_engine.should_abstain(
            confidence,
            context.detection_result,
            context.field.field_type,
            context.field.severity
        )
        processing_notes.append(f"Abstention check: {'should abstain' if abstention.should_abstain else 'no abstention'}")

        # Step 3: Run safety checks
        safety_result = self.safety_guard.check_safety(
            context.field.name,
            context.field.field_type,
            context.extracted_value,
            context.source_text,
            context.extraction_reasoning
        )
        processing_notes.append(f"Safety check: {'passed' if safety_result.passed else 'failed'}")

        # Step 4: Determine final decision
        final_decision = self._determine_final_decision(
            confidence,
            abstention,
            safety_result,
            context
        )
        processing_notes.append(f"Final decision: {final_decision.value}")

        # Step 5: Build safety decision
        safety_decision = self._build_safety_decision(
            context,
            final_decision,
            confidence,
            abstention,
            safety_result
        )

        # Step 6: Create audit log
        audit_log = self._create_audit_log(
            context,
            final_decision,
            confidence,
            abstention,
            safety_result,
            request_id,
            document_id
        )

        return DecisionOutcome(
            decision=safety_decision,
            confidence=confidence,
            abstention=abstention,
            safety_check=safety_result,
            audit_log=audit_log,
            processing_notes=processing_notes
        )

    def _determine_final_decision(
        self,
        confidence: ConfidenceScore,
        abstention: AbstentionResult,
        safety_result: SafetyCheckResult,
        context: DecisionContext
    ) -> DecisionType:
        """Determine the final decision type."""
        # Priority 1: Safety violations
        if not safety_result.passed:
            if safety_result.severity == "critical":
                return DecisionType.REJECT
            else:
                return DecisionType.FLAG_FOR_REVIEW

        # Priority 2: Abstention conditions
        if abstention.should_abstain:
            return DecisionType.ABSTAIN

        # Priority 3: Confidence-based decision
        return confidence.decision

    def _build_safety_decision(
        self,
        context: DecisionContext,
        decision: DecisionType,
        confidence: ConfidenceScore,
        abstention: AbstentionResult,
        safety_result: SafetyCheckResult
    ) -> SafetyDecision:
        """Build the complete safety decision object."""
        # Get evidence snippets
        evidence_snippets = []
        if context.detection_result.faithfulness.grounding_evidence:
            evidence_snippets.append(
                context.detection_result.faithfulness.grounding_evidence[:500]
            )

        for chunk in context.detection_result.faithfulness.supporting_chunks[:3]:
            evidence_snippets.append(chunk.content[:200])

        # Build explanation
        explanation = self._build_explanation(
            decision,
            confidence,
            abstention,
            safety_result,
            context.detection_result
        )

        # Determine abstention reason
        abstention_reason = None
        if decision == DecisionType.ABSTAIN:
            abstention_reason = self.abstention_engine.get_abstention_message(
                abstention,
                context.field.name
            )
        elif decision == DecisionType.REJECT:
            abstention_reason = f"Rejected: {safety_result.violation_description}"

        return SafetyDecision(
            field_name=context.field.name,
            decision=decision,
            confidence=confidence,
            extracted_value=context.extracted_value if decision == DecisionType.ACCEPT else None,
            evidence_snippets=evidence_snippets,
            explanation=explanation,
            abstention_reason=abstention_reason,
            requires_human_review=(decision == DecisionType.FLAG_FOR_REVIEW),
            is_critical_field=(context.field.severity == SeverityLevel.CRITICAL),
            decision_timestamp=datetime.utcnow()
        )

    def _build_explanation(
        self,
        decision: DecisionType,
        confidence: ConfidenceScore,
        abstention: AbstentionResult,
        safety_result: SafetyCheckResult,
        detection_result: HallucinationDetectionResult
    ) -> str:
        """Build human-readable explanation."""
        parts = []

        # Decision summary
        decision_summaries = {
            DecisionType.ACCEPT: "Value accepted with high confidence.",
            DecisionType.FLAG_FOR_REVIEW: "Value flagged for human review.",
            DecisionType.ABSTAIN: "Abstaining from extraction.",
            DecisionType.REJECT: "Value rejected due to safety concerns.",
        }
        parts.append(decision_summaries[decision])

        # Confidence info
        parts.append(f"Confidence score: {confidence.final_score:.2f}")

        # Key factors
        if detection_result.is_hallucinated:
            parts.append(f"⚠️ Hallucination detected: {detection_result.hallucination_type}")

        if not safety_result.passed:
            parts.append(f"⚠️ Safety violation: {safety_result.violation_description}")

        if abstention.should_abstain:
            parts.append(f"ℹ️ Abstention reason: {abstention.reason.value if abstention.reason else 'N/A'}")

        # Component scores
        parts.append("\nComponent scores:")
        parts.append(f"  - Faithfulness: {confidence.retrieval_faithfulness:.2f}")
        parts.append(f"  - Verifier: {confidence.verifier_agreement:.2f}")
        parts.append(f"  - Consistency: {confidence.self_consistency:.2f}")
        parts.append(f"  - Rules: {confidence.rule_validation:.2f}")

        return "\n".join(parts)

    def _create_audit_log(
        self,
        context: DecisionContext,
        decision: DecisionType,
        confidence: ConfidenceScore,
        abstention: AbstentionResult,
        safety_result: SafetyCheckResult,
        request_id: str,
        document_id: str
    ) -> AuditLog:
        """Create audit log entry."""
        # Build reasoning
        reasoning_parts = []
        reasoning_parts.append(f"Confidence: {confidence.final_score:.3f}")
        reasoning_parts.append(f"Thresholds: accept={confidence.accept_threshold:.2f}, review={confidence.review_threshold:.2f}")

        if abstention.should_abstain:
            reasoning_parts.append(f"Abstention: {abstention.reason.value if abstention.reason else 'N/A'}")

        if not safety_result.passed:
            reasoning_parts.append(f"Safety: {safety_result.violation_description}")

        # Build evidence list
        evidence = []
        if context.detection_result.faithfulness.grounding_evidence:
            evidence.append(context.detection_result.faithfulness.grounding_evidence[:200])

        return AuditLog(
            log_id="",  # Will be auto-generated
            request_id=request_id,
            document_id=document_id,
            event_type="decision",
            event_description=f"Safety decision for field '{context.field.name}'",
            actor_type="system",
            field_name=context.field.name,
            new_value=context.extracted_value,
            decision=decision,
            reasoning=" | ".join(reasoning_parts),
            evidence=evidence,
            model_version=self.config.version,
            config_hash=self.config.get_config_hash()
        )

    def make_decisions_batch(
        self,
        contexts: List[DecisionContext],
        request_id: str,
        document_id: str
    ) -> List[DecisionOutcome]:
        """Make decisions for multiple fields."""
        outcomes = []
        for context in contexts:
            outcome = self.make_decision(context, request_id, document_id)
            outcomes.append(outcome)
        return outcomes

    def get_decision_summary(
        self,
        outcomes: List[DecisionOutcome]
    ) -> Dict[str, Any]:
        """Get summary of decision outcomes."""
        total = len(outcomes)

        decision_counts = {
            DecisionType.ACCEPT: 0,
            DecisionType.FLAG_FOR_REVIEW: 0,
            DecisionType.ABSTAIN: 0,
            DecisionType.REJECT: 0,
        }

        for outcome in outcomes:
            decision_counts[outcome.decision.decision] += 1

        return {
            "total_fields": total,
            "accepted": decision_counts[DecisionType.ACCEPT],
            "flagged": decision_counts[DecisionType.FLAG_FOR_REVIEW],
            "abstained": decision_counts[DecisionType.ABSTAIN],
            "rejected": decision_counts[DecisionType.REJECT],
            "acceptance_rate": decision_counts[DecisionType.ACCEPT] / total if total > 0 else 0,
            "safety_rate": (total - decision_counts[DecisionType.REJECT]) / total if total > 0 else 1,
        }
