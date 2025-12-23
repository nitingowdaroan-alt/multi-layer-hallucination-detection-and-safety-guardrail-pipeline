"""
System Evaluator

Runs comprehensive evaluation of the hallucination detection
and safety guardrails system.
"""

from typing import List, Dict, Any, Optional, Callable
from dataclasses import dataclass
from datetime import datetime
import json

from ..core.models import (
    ClinicalDocument,
    ExtractionSchema,
    ExtractionField,
    DecisionType,
    FieldType,
    SeverityLevel,
)
from ..core.config import SystemConfig
from .metrics import (
    HallucinationMetrics,
    AbstentionMetrics,
    ExtractionMetrics,
    SafetyMetrics,
    ConfidenceCalibrationMetrics,
    MetricsAggregator,
)
from .golden_dataset import GoldenDataset, GoldenExample, GoldenLabel


@dataclass
class EvaluationResult:
    """Result of an evaluation run."""
    evaluation_id: str
    dataset_name: str
    started_at: str
    completed_at: str
    total_examples: int
    total_fields: int

    # Metrics
    hallucination_metrics: Dict[str, Any]
    abstention_metrics: Dict[str, Any]
    extraction_metrics: Dict[str, Any]
    safety_metrics: Dict[str, Any]
    calibration_metrics: Dict[str, Any]

    # Detailed results
    example_results: List[Dict[str, Any]]

    # Summary
    summary: Dict[str, float]


class SystemEvaluator:
    """
    Evaluates the complete hallucination detection and safety system.

    Supports:
    - Offline evaluation against golden datasets
    - Online evaluation during production
    - Threshold tuning experiments
    - Model comparison
    """

    def __init__(
        self,
        config: Optional[SystemConfig] = None,
        pipeline: Optional[Any] = None  # The main extraction pipeline
    ):
        self.config = config or SystemConfig()
        self.pipeline = pipeline
        self.metrics = MetricsAggregator()

    def evaluate_dataset(
        self,
        dataset: GoldenDataset,
        extraction_fn: Callable,
        verbose: bool = False
    ) -> EvaluationResult:
        """
        Evaluate system on a golden dataset.

        Args:
            dataset: Golden dataset with ground truth
            extraction_fn: Function that takes (document_text, field) and returns extraction result
            verbose: Whether to print progress

        Returns:
            EvaluationResult with comprehensive metrics
        """
        import uuid

        evaluation_id = str(uuid.uuid4())[:8]
        started_at = datetime.utcnow().isoformat()

        example_results = []
        total_fields = 0

        for i, example in enumerate(dataset.examples):
            if verbose:
                print(f"Evaluating example {i+1}/{len(dataset.examples)}: {example.example_id}")

            example_result = self._evaluate_example(example, extraction_fn)
            example_results.append(example_result)
            total_fields += len(example.labels)

        completed_at = datetime.utcnow().isoformat()

        return EvaluationResult(
            evaluation_id=evaluation_id,
            dataset_name=dataset.name,
            started_at=started_at,
            completed_at=completed_at,
            total_examples=len(dataset.examples),
            total_fields=total_fields,
            hallucination_metrics=self.metrics.hallucination_metrics.to_dict(),
            abstention_metrics=self.metrics.abstention_metrics.to_dict(),
            extraction_metrics=self.metrics.extraction_metrics.to_dict(),
            safety_metrics=self.metrics.safety_metrics.to_dict(),
            calibration_metrics=self.metrics.calibration_metrics.to_dict(),
            example_results=example_results,
            summary=self._compute_summary()
        )

    def _evaluate_example(
        self,
        example: GoldenExample,
        extraction_fn: Callable
    ) -> Dict[str, Any]:
        """Evaluate a single example."""
        field_results = []

        for label in example.labels:
            # Get extraction result
            try:
                result = extraction_fn(example.document_text, label.field_name, label.field_type)
            except Exception as e:
                result = {
                    "extracted_value": None,
                    "decision": DecisionType.ABSTAIN,
                    "confidence": 0.0,
                    "is_hallucinated": False,
                    "error": str(e)
                }

            # Evaluate against ground truth
            field_result = self._evaluate_field(label, result)
            field_results.append(field_result)

            # Update metrics
            self._update_metrics(label, result, field_result)

        return {
            "example_id": example.example_id,
            "document_type": example.document_type,
            "field_results": field_results,
        }

    def _evaluate_field(
        self,
        label: GoldenLabel,
        result: Dict[str, Any]
    ) -> Dict[str, Any]:
        """Evaluate a single field extraction."""
        extracted_value = result.get("extracted_value")
        decision = result.get("decision", DecisionType.ABSTAIN)
        confidence = result.get("confidence", 0.0)
        is_hallucinated = result.get("is_hallucinated", False)

        # Determine correctness
        is_correct = False
        evaluation_notes = []

        if label.should_abstain:
            # Should have abstained
            if decision == DecisionType.ABSTAIN:
                is_correct = True
                evaluation_notes.append("Correctly abstained")
            elif extracted_value is None:
                is_correct = True
                evaluation_notes.append("Correctly returned no value")
            else:
                is_correct = False
                evaluation_notes.append(f"Should have abstained but extracted: {extracted_value}")
        else:
            # Should have extracted
            if extracted_value is None or decision == DecisionType.ABSTAIN:
                is_correct = False
                evaluation_notes.append("Incorrectly abstained when value was present")
            elif self._values_match(extracted_value, label.expected_value):
                is_correct = True
                evaluation_notes.append("Correct extraction")
            else:
                is_correct = False
                evaluation_notes.append(
                    f"Incorrect: extracted '{extracted_value}' expected '{label.expected_value}'"
                )

        # Check for hallucination
        is_actual_hallucination = False
        if extracted_value and not label.is_present_in_source:
            is_actual_hallucination = True
            evaluation_notes.append("HALLUCINATION: Extracted value not in source")
        elif extracted_value and label.is_hallucination_if_different:
            if not self._values_match(extracted_value, label.expected_value):
                is_actual_hallucination = True
                evaluation_notes.append("HALLUCINATION: Value differs from ground truth")

        # Detection accuracy
        hallucination_detected_correctly = (is_hallucinated == is_actual_hallucination)

        return {
            "field_name": label.field_name,
            "field_type": label.field_type,
            "expected_value": label.expected_value,
            "extracted_value": extracted_value,
            "decision": decision.value if isinstance(decision, DecisionType) else str(decision),
            "confidence": confidence,
            "is_correct": is_correct,
            "is_actual_hallucination": is_actual_hallucination,
            "hallucination_detected": is_hallucinated,
            "hallucination_detection_correct": hallucination_detected_correctly,
            "should_abstain": label.should_abstain,
            "severity": label.severity,
            "notes": evaluation_notes,
        }

    def _values_match(
        self,
        extracted: Optional[str],
        expected: Optional[str]
    ) -> bool:
        """Check if extracted value matches expected."""
        if extracted is None and expected is None:
            return True
        if extracted is None or expected is None:
            return False

        # Normalize for comparison
        extracted_norm = extracted.lower().strip()
        expected_norm = expected.lower().strip()

        # Exact match
        if extracted_norm == expected_norm:
            return True

        # Substring match for longer values
        if len(expected_norm) > 5:
            if expected_norm in extracted_norm or extracted_norm in expected_norm:
                return True

        # Numeric comparison
        import re
        extracted_nums = re.findall(r'\d+\.?\d*', extracted)
        expected_nums = re.findall(r'\d+\.?\d*', expected)
        if extracted_nums and expected_nums:
            if extracted_nums == expected_nums:
                return True

        return False

    def _update_metrics(
        self,
        label: GoldenLabel,
        result: Dict[str, Any],
        field_result: Dict[str, Any]
    ):
        """Update aggregated metrics based on field result."""
        # Hallucination metrics
        self.metrics.hallucination_metrics.total_extractions += 1

        if field_result["is_actual_hallucination"]:
            if field_result["hallucination_detected"]:
                self.metrics.hallucination_metrics.true_positives += 1
                self.metrics.hallucination_metrics.detected_hallucinations += 1
            else:
                self.metrics.hallucination_metrics.false_negatives += 1
        else:
            if field_result["hallucination_detected"]:
                self.metrics.hallucination_metrics.false_positives += 1
                self.metrics.hallucination_metrics.detected_hallucinations += 1
            else:
                self.metrics.hallucination_metrics.true_negatives += 1

        # Abstention metrics
        self.metrics.abstention_metrics.total_extractions += 1
        decision = result.get("decision")
        if decision == DecisionType.ABSTAIN or result.get("extracted_value") is None:
            self.metrics.abstention_metrics.abstentions += 1
            if label.should_abstain:
                self.metrics.abstention_metrics.correct_abstentions += 1
            else:
                self.metrics.abstention_metrics.incorrect_abstentions += 1
        else:
            if label.should_abstain:
                self.metrics.abstention_metrics.missed_abstentions += 1

        # Extraction metrics
        is_critical = label.severity == "critical"
        self.metrics.extraction_metrics.add_field_result(
            label.field_type,
            field_result["is_correct"],
            is_critical
        )

        # Safety metrics
        safety_passed = not field_result["is_actual_hallucination"]
        self.metrics.safety_metrics.add_check_result(
            safety_passed,
            "hallucination" if not safety_passed else None,
            label.severity
        )

        # Calibration metrics
        confidence = result.get("confidence", 0.0)
        self.metrics.calibration_metrics.add_prediction(
            confidence,
            field_result["is_correct"]
        )

    def _compute_summary(self) -> Dict[str, float]:
        """Compute summary metrics."""
        return {
            "hallucination_rate": self.metrics.hallucination_metrics.compute_hallucination_rate(),
            "hallucination_detection_f1": self.metrics.hallucination_metrics.compute_f1_score(),
            "abstention_accuracy": self.metrics.abstention_metrics.compute_abstention_accuracy(),
            "abstention_rate": self.metrics.abstention_metrics.compute_abstention_rate(),
            "extraction_precision": self.metrics.extraction_metrics.compute_overall_precision(),
            "critical_precision": self.metrics.extraction_metrics.compute_critical_precision(),
            "safety_rate": self.metrics.safety_metrics.compute_safety_rate(),
            "calibration_error": self.metrics.calibration_metrics.compute_calibration_error(),
        }

    def reset_metrics(self):
        """Reset all metrics for new evaluation."""
        self.metrics.reset()

    def save_results(self, result: EvaluationResult, file_path: str):
        """Save evaluation results to JSON."""
        with open(file_path, 'w') as f:
            json.dump({
                "evaluation_id": result.evaluation_id,
                "dataset_name": result.dataset_name,
                "started_at": result.started_at,
                "completed_at": result.completed_at,
                "total_examples": result.total_examples,
                "total_fields": result.total_fields,
                "hallucination_metrics": result.hallucination_metrics,
                "abstention_metrics": result.abstention_metrics,
                "extraction_metrics": result.extraction_metrics,
                "safety_metrics": result.safety_metrics,
                "calibration_metrics": result.calibration_metrics,
                "summary": result.summary,
                "example_results": result.example_results,
            }, f, indent=2)

    def compare_runs(
        self,
        results: List[EvaluationResult]
    ) -> Dict[str, Any]:
        """Compare multiple evaluation runs."""
        comparison = {
            "runs": [],
            "metric_comparison": {}
        }

        metrics_to_compare = [
            "hallucination_rate",
            "hallucination_detection_f1",
            "abstention_accuracy",
            "extraction_precision",
            "critical_precision",
            "safety_rate",
        ]

        for metric in metrics_to_compare:
            comparison["metric_comparison"][metric] = {}

        for result in results:
            run_info = {
                "evaluation_id": result.evaluation_id,
                "dataset_name": result.dataset_name,
                "completed_at": result.completed_at,
            }
            comparison["runs"].append(run_info)

            for metric in metrics_to_compare:
                value = result.summary.get(metric, 0.0)
                comparison["metric_comparison"][metric][result.evaluation_id] = value

        # Find best for each metric
        comparison["best_by_metric"] = {}
        for metric in metrics_to_compare:
            values = comparison["metric_comparison"][metric]
            if metric in ["hallucination_rate", "calibration_error"]:
                # Lower is better
                best_id = min(values, key=values.get)
            else:
                # Higher is better
                best_id = max(values, key=values.get)
            comparison["best_by_metric"][metric] = {
                "evaluation_id": best_id,
                "value": values[best_id]
            }

        return comparison
