"""
Evaluation Framework for Healthcare AI Safety System

Provides comprehensive metrics and evaluation for:
- Hallucination detection accuracy
- Abstention appropriateness
- Clinical field precision
- System reliability
"""

from .metrics import (
    HallucinationMetrics,
    AbstentionMetrics,
    ExtractionMetrics,
    SafetyMetrics,
)
from .evaluator import SystemEvaluator
from .golden_dataset import GoldenDataset, GoldenExample
from .callbacks import EvaluationCallback, LangChainCallback

__all__ = [
    "HallucinationMetrics",
    "AbstentionMetrics",
    "ExtractionMetrics",
    "SafetyMetrics",
    "SystemEvaluator",
    "GoldenDataset",
    "GoldenExample",
    "EvaluationCallback",
    "LangChainCallback",
]
