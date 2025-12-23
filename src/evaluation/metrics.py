"""
Evaluation Metrics for Healthcare AI Safety System

Defines all metrics used to evaluate system performance:
- Hallucination rate
- Unsupported claim rate
- Abstention accuracy
- Critical field precision
"""

from typing import List, Dict, Any, Optional
from dataclasses import dataclass, field
from enum import Enum
import statistics


class MetricType(Enum):
    """Types of evaluation metrics."""
    HALLUCINATION_RATE = "hallucination_rate"
    UNSUPPORTED_CLAIM_RATE = "unsupported_claim_rate"
    ABSTENTION_ACCURACY = "abstention_accuracy"
    ABSTENTION_RATE = "abstention_rate"
    PRECISION = "precision"
    RECALL = "recall"
    F1_SCORE = "f1_score"
    SAFETY_VIOLATION_RATE = "safety_violation_rate"
    CONFIDENCE_CALIBRATION = "confidence_calibration"


@dataclass
class MetricResult:
    """Result of a metric computation."""
    metric_type: MetricType
    value: float
    sample_size: int
    confidence_interval: Optional[tuple] = None
    breakdown: Optional[Dict[str, float]] = None


@dataclass
class HallucinationMetrics:
    """Metrics for hallucination detection."""
    # Counts
    total_extractions: int = 0
    detected_hallucinations: int = 0
    true_positives: int = 0  # Correctly detected hallucinations
    false_positives: int = 0  # Incorrectly flagged as hallucination
    true_negatives: int = 0  # Correctly accepted extractions
    false_negatives: int = 0  # Missed hallucinations

    # By type
    fabricated_count: int = 0
    contradicted_count: int = 0
    unsupported_count: int = 0
    ungrounded_count: int = 0

    def compute_hallucination_rate(self) -> float:
        """Compute overall hallucination rate."""
        if self.total_extractions == 0:
            return 0.0
        return self.detected_hallucinations / self.total_extractions

    def compute_detection_precision(self) -> float:
        """Precision: Of detected hallucinations, how many were correct?"""
        total_detected = self.true_positives + self.false_positives
        if total_detected == 0:
            return 1.0
        return self.true_positives / total_detected

    def compute_detection_recall(self) -> float:
        """Recall: Of actual hallucinations, how many did we detect?"""
        total_actual = self.true_positives + self.false_negatives
        if total_actual == 0:
            return 1.0
        return self.true_positives / total_actual

    def compute_f1_score(self) -> float:
        """F1 score for hallucination detection."""
        precision = self.compute_detection_precision()
        recall = self.compute_detection_recall()
        if precision + recall == 0:
            return 0.0
        return 2 * (precision * recall) / (precision + recall)

    def compute_unsupported_rate(self) -> float:
        """Rate of unsupported claims."""
        if self.total_extractions == 0:
            return 0.0
        return self.unsupported_count / self.total_extractions

    def to_dict(self) -> Dict[str, Any]:
        """Convert to dictionary."""
        return {
            "total_extractions": self.total_extractions,
            "hallucination_rate": self.compute_hallucination_rate(),
            "detection_precision": self.compute_detection_precision(),
            "detection_recall": self.compute_detection_recall(),
            "f1_score": self.compute_f1_score(),
            "unsupported_rate": self.compute_unsupported_rate(),
            "breakdown": {
                "fabricated": self.fabricated_count,
                "contradicted": self.contradicted_count,
                "unsupported": self.unsupported_count,
                "ungrounded": self.ungrounded_count,
            }
        }


@dataclass
class AbstentionMetrics:
    """Metrics for abstention behavior."""
    total_extractions: int = 0
    abstentions: int = 0

    # Abstention quality
    correct_abstentions: int = 0  # Abstained when should have
    incorrect_abstentions: int = 0  # Abstained when shouldn't have
    missed_abstentions: int = 0  # Didn't abstain when should have

    # By reason
    abstention_by_reason: Dict[str, int] = field(default_factory=dict)

    def compute_abstention_rate(self) -> float:
        """Overall abstention rate."""
        if self.total_extractions == 0:
            return 0.0
        return self.abstentions / self.total_extractions

    def compute_abstention_accuracy(self) -> float:
        """Accuracy of abstention decisions."""
        total_decisions = self.correct_abstentions + self.incorrect_abstentions
        if total_decisions == 0:
            return 1.0
        return self.correct_abstentions / total_decisions

    def compute_abstention_precision(self) -> float:
        """Precision: Of abstentions, how many were correct?"""
        if self.abstentions == 0:
            return 1.0
        return self.correct_abstentions / self.abstentions

    def compute_abstention_recall(self) -> float:
        """Recall: Of cases that needed abstention, how many did we abstain?"""
        should_abstain = self.correct_abstentions + self.missed_abstentions
        if should_abstain == 0:
            return 1.0
        return self.correct_abstentions / should_abstain

    def to_dict(self) -> Dict[str, Any]:
        """Convert to dictionary."""
        return {
            "total_extractions": self.total_extractions,
            "abstention_rate": self.compute_abstention_rate(),
            "abstention_accuracy": self.compute_abstention_accuracy(),
            "abstention_precision": self.compute_abstention_precision(),
            "abstention_recall": self.compute_abstention_recall(),
            "by_reason": self.abstention_by_reason,
        }


@dataclass
class ExtractionMetrics:
    """Metrics for extraction quality."""
    total_fields: int = 0
    extracted_fields: int = 0
    correct_extractions: int = 0

    # By field type
    metrics_by_type: Dict[str, Dict[str, int]] = field(default_factory=dict)

    # Critical fields (medications, allergies, diagnoses)
    critical_total: int = 0
    critical_correct: int = 0

    def compute_overall_precision(self) -> float:
        """Overall extraction precision."""
        if self.extracted_fields == 0:
            return 0.0
        return self.correct_extractions / self.extracted_fields

    def compute_overall_recall(self) -> float:
        """Overall extraction recall."""
        if self.total_fields == 0:
            return 0.0
        return self.correct_extractions / self.total_fields

    def compute_critical_precision(self) -> float:
        """Precision on critical fields."""
        if self.critical_total == 0:
            return 0.0
        return self.critical_correct / self.critical_total

    def compute_f1_score(self) -> float:
        """F1 score for extractions."""
        precision = self.compute_overall_precision()
        recall = self.compute_overall_recall()
        if precision + recall == 0:
            return 0.0
        return 2 * (precision * recall) / (precision + recall)

    def add_field_result(
        self,
        field_type: str,
        is_correct: bool,
        is_critical: bool = False
    ):
        """Add a field result."""
        self.total_fields += 1
        self.extracted_fields += 1

        if is_correct:
            self.correct_extractions += 1

        if is_critical:
            self.critical_total += 1
            if is_correct:
                self.critical_correct += 1

        if field_type not in self.metrics_by_type:
            self.metrics_by_type[field_type] = {"total": 0, "correct": 0}

        self.metrics_by_type[field_type]["total"] += 1
        if is_correct:
            self.metrics_by_type[field_type]["correct"] += 1

    def to_dict(self) -> Dict[str, Any]:
        """Convert to dictionary."""
        by_type = {}
        for ft, counts in self.metrics_by_type.items():
            precision = counts["correct"] / counts["total"] if counts["total"] > 0 else 0
            by_type[ft] = {
                "total": counts["total"],
                "correct": counts["correct"],
                "precision": precision
            }

        return {
            "total_fields": self.total_fields,
            "overall_precision": self.compute_overall_precision(),
            "overall_recall": self.compute_overall_recall(),
            "f1_score": self.compute_f1_score(),
            "critical_precision": self.compute_critical_precision(),
            "by_field_type": by_type,
        }


@dataclass
class SafetyMetrics:
    """Metrics for safety system performance."""
    total_checks: int = 0
    passed_checks: int = 0

    # Violations by type
    violations_by_type: Dict[str, int] = field(default_factory=dict)

    # By severity
    critical_violations: int = 0
    high_violations: int = 0
    medium_violations: int = 0

    def compute_safety_rate(self) -> float:
        """Rate of extractions passing safety checks."""
        if self.total_checks == 0:
            return 1.0
        return self.passed_checks / self.total_checks

    def compute_violation_rate(self) -> float:
        """Rate of safety violations."""
        return 1.0 - self.compute_safety_rate()

    def compute_critical_violation_rate(self) -> float:
        """Rate of critical safety violations."""
        if self.total_checks == 0:
            return 0.0
        return self.critical_violations / self.total_checks

    def add_check_result(
        self,
        passed: bool,
        violation_type: Optional[str] = None,
        severity: str = "medium"
    ):
        """Add a safety check result."""
        self.total_checks += 1

        if passed:
            self.passed_checks += 1
        else:
            if violation_type:
                self.violations_by_type[violation_type] = \
                    self.violations_by_type.get(violation_type, 0) + 1

            if severity == "critical":
                self.critical_violations += 1
            elif severity == "high":
                self.high_violations += 1
            else:
                self.medium_violations += 1

    def to_dict(self) -> Dict[str, Any]:
        """Convert to dictionary."""
        return {
            "total_checks": self.total_checks,
            "safety_rate": self.compute_safety_rate(),
            "violation_rate": self.compute_violation_rate(),
            "critical_violation_rate": self.compute_critical_violation_rate(),
            "violations_by_type": self.violations_by_type,
            "by_severity": {
                "critical": self.critical_violations,
                "high": self.high_violations,
                "medium": self.medium_violations,
            }
        }


@dataclass
class ConfidenceCalibrationMetrics:
    """Metrics for confidence score calibration."""
    # Binned accuracy
    confidence_bins: Dict[str, Dict[str, float]] = field(default_factory=dict)

    # Calibration error
    expected_calibration_error: float = 0.0
    max_calibration_error: float = 0.0

    def add_prediction(
        self,
        confidence: float,
        is_correct: bool
    ):
        """Add a prediction for calibration analysis."""
        # Bin the confidence
        bin_name = self._get_bin_name(confidence)

        if bin_name not in self.confidence_bins:
            self.confidence_bins[bin_name] = {
                "count": 0,
                "correct": 0,
                "confidence_sum": 0.0
            }

        self.confidence_bins[bin_name]["count"] += 1
        self.confidence_bins[bin_name]["confidence_sum"] += confidence
        if is_correct:
            self.confidence_bins[bin_name]["correct"] += 1

    def _get_bin_name(self, confidence: float) -> str:
        """Get bin name for confidence value."""
        if confidence >= 0.9:
            return "0.9-1.0"
        elif confidence >= 0.8:
            return "0.8-0.9"
        elif confidence >= 0.7:
            return "0.7-0.8"
        elif confidence >= 0.6:
            return "0.6-0.7"
        elif confidence >= 0.5:
            return "0.5-0.6"
        else:
            return "0.0-0.5"

    def compute_calibration_error(self) -> float:
        """Compute expected calibration error."""
        total_samples = sum(b["count"] for b in self.confidence_bins.values())
        if total_samples == 0:
            return 0.0

        ece = 0.0
        for bin_data in self.confidence_bins.values():
            if bin_data["count"] == 0:
                continue

            accuracy = bin_data["correct"] / bin_data["count"]
            avg_confidence = bin_data["confidence_sum"] / bin_data["count"]
            weight = bin_data["count"] / total_samples

            ece += weight * abs(accuracy - avg_confidence)

        self.expected_calibration_error = ece
        return ece

    def to_dict(self) -> Dict[str, Any]:
        """Convert to dictionary."""
        bin_analysis = {}
        for bin_name, data in self.confidence_bins.items():
            if data["count"] > 0:
                bin_analysis[bin_name] = {
                    "count": data["count"],
                    "accuracy": data["correct"] / data["count"],
                    "avg_confidence": data["confidence_sum"] / data["count"],
                }

        return {
            "expected_calibration_error": self.compute_calibration_error(),
            "bins": bin_analysis,
        }


class MetricsAggregator:
    """Aggregates metrics across multiple evaluation runs."""

    def __init__(self):
        self.hallucination_metrics = HallucinationMetrics()
        self.abstention_metrics = AbstentionMetrics()
        self.extraction_metrics = ExtractionMetrics()
        self.safety_metrics = SafetyMetrics()
        self.calibration_metrics = ConfidenceCalibrationMetrics()

    def get_full_report(self) -> Dict[str, Any]:
        """Get comprehensive metrics report."""
        return {
            "hallucination": self.hallucination_metrics.to_dict(),
            "abstention": self.abstention_metrics.to_dict(),
            "extraction": self.extraction_metrics.to_dict(),
            "safety": self.safety_metrics.to_dict(),
            "calibration": self.calibration_metrics.to_dict(),
            "summary": {
                "hallucination_rate": self.hallucination_metrics.compute_hallucination_rate(),
                "abstention_accuracy": self.abstention_metrics.compute_abstention_accuracy(),
                "extraction_precision": self.extraction_metrics.compute_overall_precision(),
                "critical_precision": self.extraction_metrics.compute_critical_precision(),
                "safety_rate": self.safety_metrics.compute_safety_rate(),
            }
        }

    def reset(self):
        """Reset all metrics."""
        self.hallucination_metrics = HallucinationMetrics()
        self.abstention_metrics = AbstentionMetrics()
        self.extraction_metrics = ExtractionMetrics()
        self.safety_metrics = SafetyMetrics()
        self.calibration_metrics = ConfidenceCalibrationMetrics()
