"""
Multi-Layer Hallucination Detector

Orchestrates all four hallucination detection layers and combines
their results into a comprehensive detection outcome.
"""

from typing import List, Optional, Dict, Any
from dataclasses import dataclass
import time

from ..core.models import (
    RetrievedChunk,
    HallucinationDetectionResult,
    FaithfulnessResult,
    ContradictionResult,
    ConsistencyResult,
    RuleValidationResult,
    FieldType,
    ExtractionField,
)
from ..core.config import (
    SystemConfig,
    HallucinationDetectionConfig,
    ModelConfig,
)
from ..rag.embeddings import EmbeddingService
from .faithfulness import FaithfulnessScorer
from .contradiction import ContradictionDetector
from .consistency import SelfConsistencyChecker
from .clinical_rules import ClinicalRuleValidator


@dataclass
class DetectionInput:
    """Input for hallucination detection."""
    field: ExtractionField
    extracted_value: Optional[str]
    supporting_chunks: List[RetrievedChunk]
    extraction_context: str
    additional_context: Optional[Dict[str, Any]] = None


class HallucinationDetector:
    """
    Multi-layer hallucination detection system.

    Combines four detection layers:
    1. Retrieval Faithfulness Scoring - Is the value grounded in evidence?
    2. Contradiction Detection - Does the value contradict the evidence?
    3. Self-Consistency Checks - Is the extraction stable across runs?
    4. Rule-Based Clinical Validation - Does the value pass clinical rules?

    Each layer produces a score/result that contributes to the final
    hallucination detection outcome.
    """

    def __init__(
        self,
        config: Optional[SystemConfig] = None,
        embedding_service: Optional[EmbeddingService] = None,
        llm_client: Optional[Any] = None
    ):
        self.config = config or SystemConfig()
        self.detection_config = self.config.hallucination_detection
        self.model_config = self.config.model

        # Initialize layer components
        self.embedding_service = embedding_service or EmbeddingService(
            self.model_config
        )

        self.faithfulness_scorer = FaithfulnessScorer(
            self.embedding_service,
            self.detection_config
        )

        self.contradiction_detector = ContradictionDetector(
            self.model_config,
            self.detection_config,
            llm_client
        )

        self.consistency_checker = SelfConsistencyChecker(
            self.model_config,
            self.detection_config,
            llm_client
        )

        self.rule_validator = ClinicalRuleValidator(
            self.config.safety
        )

    def detect(
        self,
        detection_input: DetectionInput,
        run_all_layers: bool = True,
        skip_layers: Optional[List[str]] = None
    ) -> HallucinationDetectionResult:
        """
        Run hallucination detection on an extracted field.

        Args:
            detection_input: Input containing field, value, and evidence
            run_all_layers: Whether to run all layers
            skip_layers: List of layer names to skip

        Returns:
            HallucinationDetectionResult with all layer results
        """
        skip_layers = skip_layers or []

        # Handle null/missing values
        if detection_input.extracted_value is None:
            return self._create_null_result(detection_input.field.name)

        # Layer 1: Faithfulness Scoring
        if "faithfulness" not in skip_layers:
            faithfulness_result = self._run_faithfulness_layer(detection_input)
        else:
            faithfulness_result = self._create_default_faithfulness(
                detection_input.field.name
            )

        # Layer 2: Contradiction Detection
        if "contradiction" not in skip_layers:
            contradiction_result = self._run_contradiction_layer(detection_input)
        else:
            contradiction_result = self._create_default_contradiction(
                detection_input.field.name
            )

        # Layer 3: Self-Consistency
        if "consistency" not in skip_layers:
            consistency_result = self._run_consistency_layer(detection_input)
        else:
            consistency_result = self._create_default_consistency(
                detection_input.field.name
            )

        # Layer 4: Rule Validation
        if "rules" not in skip_layers:
            rule_result = self._run_rule_layer(detection_input)
        else:
            rule_result = self._create_default_rules(
                detection_input.field.name,
                detection_input.field.field_type
            )

        # Combine results
        return self._combine_results(
            detection_input.field.name,
            faithfulness_result,
            contradiction_result,
            consistency_result,
            rule_result
        )

    def _run_faithfulness_layer(
        self,
        detection_input: DetectionInput
    ) -> FaithfulnessResult:
        """Run Layer 1: Faithfulness Scoring."""
        return self.faithfulness_scorer.score_faithfulness(
            extracted_value=detection_input.extracted_value,
            supporting_chunks=detection_input.supporting_chunks,
            field_name=detection_input.field.name,
            field_type=detection_input.field.field_type
        )

    def _run_contradiction_layer(
        self,
        detection_input: DetectionInput
    ) -> ContradictionResult:
        """Run Layer 2: Contradiction Detection."""
        return self.contradiction_detector.detect_contradiction(
            field_name=detection_input.field.name,
            extracted_value=detection_input.extracted_value,
            supporting_chunks=detection_input.supporting_chunks,
            field_type=detection_input.field.field_type
        )

    def _run_consistency_layer(
        self,
        detection_input: DetectionInput
    ) -> ConsistencyResult:
        """Run Layer 3: Self-Consistency Checks."""
        return self.consistency_checker.check_consistency(
            field=detection_input.field,
            context=detection_input.extraction_context,
            initial_value=detection_input.extracted_value
        )

    def _run_rule_layer(
        self,
        detection_input: DetectionInput
    ) -> RuleValidationResult:
        """Run Layer 4: Rule-Based Validation."""
        additional_context = detection_input.additional_context or {}

        return self.rule_validator.validate(
            field_name=detection_input.field.name,
            field_type=detection_input.field.field_type,
            extracted_value=detection_input.extracted_value,
            context=additional_context
        )

    def _combine_results(
        self,
        field_name: str,
        faithfulness: FaithfulnessResult,
        contradiction: ContradictionResult,
        consistency: ConsistencyResult,
        rules: RuleValidationResult
    ) -> HallucinationDetectionResult:
        """Combine all layer results into final detection result."""
        # Compute composite score
        composite_score = self._compute_composite_score(
            faithfulness, contradiction, consistency, rules
        )

        # Determine if hallucinated
        is_hallucinated, hallucination_type = self._determine_hallucination(
            faithfulness, contradiction, consistency, rules, composite_score
        )

        return HallucinationDetectionResult(
            field_name=field_name,
            faithfulness=faithfulness,
            contradiction=contradiction,
            consistency=consistency,
            rule_validation=rules,
            composite_score=composite_score,
            is_hallucinated=is_hallucinated,
            hallucination_type=hallucination_type
        )

    def _compute_composite_score(
        self,
        faithfulness: FaithfulnessResult,
        contradiction: ContradictionResult,
        consistency: ConsistencyResult,
        rules: RuleValidationResult
    ) -> float:
        """Compute weighted composite hallucination score."""
        # Weights for each component
        weights = {
            "faithfulness": 0.35,
            "contradiction": 0.25,
            "consistency": 0.20,
            "rules": 0.20
        }

        # Convert each layer result to a 0-1 score (higher = more reliable)
        faithfulness_score = faithfulness.faithfulness_score

        # Contradiction: 0 if contradiction detected, 1 if aligned
        if contradiction.contradiction_detected:
            contradiction_score = 0.0
        elif contradiction.evidence_alignment == "aligned":
            contradiction_score = contradiction.verifier_confidence
        else:
            contradiction_score = 0.3  # Unsupported

        consistency_score = consistency.agreement_rate

        rules_score = 1.0 if rules.passed else 0.0

        # Weighted combination
        composite = (
            faithfulness_score * weights["faithfulness"] +
            contradiction_score * weights["contradiction"] +
            consistency_score * weights["consistency"] +
            rules_score * weights["rules"]
        )

        return min(1.0, max(0.0, composite))

    def _determine_hallucination(
        self,
        faithfulness: FaithfulnessResult,
        contradiction: ContradictionResult,
        consistency: ConsistencyResult,
        rules: RuleValidationResult,
        composite_score: float
    ) -> tuple:
        """Determine if the extraction is hallucinated and the type."""
        # Rule violations are automatic hallucination
        if not rules.passed:
            return True, "rule_violation"

        # Contradiction is strong evidence
        if contradiction.contradiction_detected:
            if contradiction.evidence_alignment == "contradicted":
                return True, "contradicted"
            else:
                return True, "unsupported"

        # Low faithfulness
        if faithfulness.faithfulness_score < self.detection_config.faithfulness_threshold:
            if not faithfulness.is_grounded:
                return True, "ungrounded"

        # Low consistency
        if not consistency.is_stable:
            if consistency.agreement_rate < 0.5:
                return True, "unstable"

        # Composite score threshold
        if composite_score < 0.4:
            return True, "low_confidence"

        return False, None

    def _create_null_result(self, field_name: str) -> HallucinationDetectionResult:
        """Create result for null/missing values."""
        return HallucinationDetectionResult(
            field_name=field_name,
            faithfulness=FaithfulnessResult(
                field_name=field_name,
                faithfulness_score=0.0,
                supporting_chunks=[],
                max_chunk_similarity=0.0,
                avg_chunk_similarity=0.0,
                grounding_evidence="",
                is_grounded=False
            ),
            contradiction=ContradictionResult(
                field_name=field_name,
                contradiction_detected=False,
                contradiction_reason=None,
                verifier_confidence=1.0,
                evidence_alignment="unsupported",
                explanation="No value to verify"
            ),
            consistency=ConsistencyResult(
                field_name=field_name,
                num_runs=0,
                unique_values=[],
                agreement_rate=0.0,
                variance_score=1.0,
                most_common_value=None,
                confidence_decay=0.0,
                is_stable=False
            ),
            rule_validation=RuleValidationResult(
                field_name=field_name,
                field_type=FieldType.FREE_TEXT,
                passed=True,
                rules_applied=["null_check"],
                violations=[],
                warnings=[]
            ),
            composite_score=0.0,
            is_hallucinated=False,
            hallucination_type=None
        )

    def _create_default_faithfulness(self, field_name: str) -> FaithfulnessResult:
        """Create default faithfulness result when layer is skipped."""
        return FaithfulnessResult(
            field_name=field_name,
            faithfulness_score=0.5,
            supporting_chunks=[],
            max_chunk_similarity=0.0,
            avg_chunk_similarity=0.0,
            grounding_evidence="Layer skipped",
            is_grounded=True
        )

    def _create_default_contradiction(self, field_name: str) -> ContradictionResult:
        """Create default contradiction result when layer is skipped."""
        return ContradictionResult(
            field_name=field_name,
            contradiction_detected=False,
            contradiction_reason=None,
            verifier_confidence=0.5,
            evidence_alignment="unsupported",
            explanation="Layer skipped"
        )

    def _create_default_consistency(self, field_name: str) -> ConsistencyResult:
        """Create default consistency result when layer is skipped."""
        return ConsistencyResult(
            field_name=field_name,
            num_runs=1,
            unique_values=[],
            agreement_rate=1.0,
            variance_score=0.0,
            most_common_value=None,
            confidence_decay=0.0,
            is_stable=True
        )

    def _create_default_rules(
        self,
        field_name: str,
        field_type: FieldType
    ) -> RuleValidationResult:
        """Create default rule result when layer is skipped."""
        return RuleValidationResult(
            field_name=field_name,
            field_type=field_type,
            passed=True,
            rules_applied=["layer_skipped"],
            violations=[],
            warnings=[]
        )

    def detect_batch(
        self,
        inputs: List[DetectionInput],
        parallel: bool = False
    ) -> List[HallucinationDetectionResult]:
        """
        Run detection on multiple inputs.

        Args:
            inputs: List of detection inputs
            parallel: Whether to run in parallel (not implemented)

        Returns:
            List of detection results
        """
        results = []
        for inp in inputs:
            result = self.detect(inp)
            results.append(result)
        return results

    def quick_detect(
        self,
        field_name: str,
        field_type: FieldType,
        extracted_value: str,
        evidence_text: str
    ) -> Dict[str, Any]:
        """
        Quick detection without full layer analysis.

        Useful for real-time checking or high-volume scenarios.
        """
        # Quick faithfulness check
        is_present = extracted_value.lower() in evidence_text.lower()

        # Quick contradiction check
        has_contradiction = self.contradiction_detector.quick_contradiction_check(
            extracted_value, evidence_text
        )

        # Quick rule check
        rule_result = self.rule_validator.validate(
            field_name, field_type, extracted_value
        )

        is_suspicious = (
            not is_present or
            has_contradiction or
            not rule_result.passed
        )

        return {
            "field_name": field_name,
            "is_suspicious": is_suspicious,
            "present_in_evidence": is_present,
            "possible_contradiction": has_contradiction,
            "rule_violations": rule_result.violations,
            "recommendation": "full_analysis" if is_suspicious else "likely_valid"
        }
