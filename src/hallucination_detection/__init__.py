"""
Multi-Layer Hallucination Detection System

Provides comprehensive hallucination detection for clinical data extraction:
- Layer 1: Retrieval Faithfulness Scoring
- Layer 2: Contradiction Detection
- Layer 3: Self-Consistency Checks
- Layer 4: Rule-Based Clinical Validation
"""

from .faithfulness import FaithfulnessScorer
from .contradiction import ContradictionDetector
from .consistency import SelfConsistencyChecker
from .clinical_rules import ClinicalRuleValidator
from .detector import HallucinationDetector

__all__ = [
    "FaithfulnessScorer",
    "ContradictionDetector",
    "SelfConsistencyChecker",
    "ClinicalRuleValidator",
    "HallucinationDetector",
]
