"""
Healthcare-Specific Safety Guard

Implements hard safety rules for healthcare AI:
- Never infer diagnoses
- Never hallucinate missing values
- Never normalize ambiguous medical abbreviations
- Never override clinician-written text
"""

import re
from typing import Optional, List, Dict, Any, Tuple
from dataclasses import dataclass
from enum import Enum

from ..core.models import FieldType, SeverityLevel
from ..core.config import SafetyGuardrailConfig


class SafetyViolationType(Enum):
    """Types of safety violations."""
    INFERRED_DIAGNOSIS = "inferred_diagnosis"
    HALLUCINATED_VALUE = "hallucinated_value"
    NORMALIZED_ABBREVIATION = "normalized_abbreviation"
    OVERRIDDEN_CLINICIAN_TEXT = "overridden_clinician_text"
    FABRICATED_MEDICATION = "fabricated_medication"
    ALTERED_DOSAGE = "altered_dosage"
    INVENTED_LAB_VALUE = "invented_lab_value"
    UNSAFE_RECOMMENDATION = "unsafe_recommendation"


@dataclass
class SafetyCheckResult:
    """Result of a safety check."""
    passed: bool
    violation_type: Optional[SafetyViolationType]
    violation_description: str
    severity: str  # "critical", "high", "medium"
    remediation: str


@dataclass
class SafetyAuditEntry:
    """Audit entry for safety checks."""
    field_name: str
    check_performed: str
    result: str
    details: str
    timestamp: str


class HealthcareSafetyGuard:
    """
    Healthcare-specific safety guardrails.

    HARD RULES - These CANNOT be overridden:
    1. Never infer diagnoses from symptoms or test results
    2. Never hallucinate missing values - use abstention instead
    3. Never normalize ambiguous medical abbreviations
    4. Never override or modify clinician-written text
    5. Never fabricate medication names or dosages
    6. Never invent lab values or vital signs

    These rules are designed to prevent patient harm from AI errors.
    """

    # Ambiguous medical abbreviations that should not be normalized
    AMBIGUOUS_ABBREVIATIONS = {
        "MS": ["multiple sclerosis", "mitral stenosis", "morphine sulfate", "magnesium sulfate"],
        "BS": ["blood sugar", "bowel sounds", "breath sounds"],
        "DC": ["discontinue", "discharge", "Doctor of Chiropractic"],
        "OD": ["right eye", "overdose", "once daily"],
        "OS": ["left eye", "overall survival"],
        "OU": ["both eyes"],
        "IU": ["international units"],  # Often confused with IV
        "U": ["units"],  # Can be confused with 0
        "QD": ["once daily"],  # Can be confused with QID
        "QOD": ["every other day"],
        "SC": ["subcutaneous"],  # Can be confused with SL
        "SL": ["sublingual"],  # Can be confused with SC
        "HS": ["at bedtime", "half strength"],
        "SS": ["sliding scale", "one-half"],
        "AU": ["each ear"],
        "AD": ["right ear"],
        "AS": ["left ear"],
        "CC": ["cubic centimeter", "chief complaint"],
        "D/C": ["discontinue", "discharge"],
        "μg": ["microgram"],  # Can be misread as mg
        "TID": ["three times daily"],
        "BID": ["twice daily"],
        "QID": ["four times daily"],
        "PRN": ["as needed"],
    }

    # Diagnosis-related keywords that indicate inference
    DIAGNOSIS_INFERENCE_PATTERNS = [
        r"(?i)\b(likely|probably|possibly|suggests?|indicates?|consistent with|suspicious for)\b.*\b(diagnosis|diagnoses)\b",
        r"(?i)\b(may have|might have|could have|appears to have)\b",
        r"(?i)\b(differential|rule out|r/o)\b",
        r"(?i)\b(impression|assessment):\s*(?!.*\bdiagnosed\b)",
    ]

    # Patterns indicating fabrication
    FABRICATION_PATTERNS = [
        r"(?i)\b(assuming|assume|estimated|approximately|about|around|roughly)\s+\d",
        r"(?i)\b(typical|usual|standard|normal)\s+(dose|dosage|value)",
        r"(?i)\b(based on|according to)\s+(guidelines|protocols|standards)",
    ]

    def __init__(self, config: Optional[SafetyGuardrailConfig] = None):
        self.config = config or SafetyGuardrailConfig()
        self._audit_log: List[SafetyAuditEntry] = []

    def check_safety(
        self,
        field_name: str,
        field_type: FieldType,
        extracted_value: Optional[str],
        source_text: str,
        extraction_reasoning: Optional[str] = None
    ) -> SafetyCheckResult:
        """
        Perform comprehensive safety check on an extraction.

        Args:
            field_name: Name of the extracted field
            field_type: Type of the field
            extracted_value: The extracted value
            source_text: Original source text
            extraction_reasoning: Optional reasoning from the LLM

        Returns:
            SafetyCheckResult with check outcome
        """
        if extracted_value is None:
            return SafetyCheckResult(
                passed=True,
                violation_type=None,
                violation_description="No value to check",
                severity="low",
                remediation=""
            )

        # Run all safety checks
        checks = [
            self._check_diagnosis_inference(field_type, extracted_value, extraction_reasoning),
            self._check_hallucinated_value(extracted_value, source_text, field_type),
            self._check_abbreviation_normalization(extracted_value, source_text),
            self._check_clinician_override(extracted_value, source_text, field_type),
            self._check_fabricated_values(extracted_value, source_text, field_type),
        ]

        # Return first violation found
        for result in checks:
            if not result.passed:
                self._log_audit(field_name, result)
                return result

        return SafetyCheckResult(
            passed=True,
            violation_type=None,
            violation_description="All safety checks passed",
            severity="low",
            remediation=""
        )

    def _check_diagnosis_inference(
        self,
        field_type: FieldType,
        value: str,
        reasoning: Optional[str]
    ) -> SafetyCheckResult:
        """Check if a diagnosis was inferred rather than extracted."""
        if field_type != FieldType.DIAGNOSIS:
            return SafetyCheckResult(passed=True, violation_type=None,
                                     violation_description="", severity="low", remediation="")

        if not self.config.never_infer_diagnoses:
            return SafetyCheckResult(passed=True, violation_type=None,
                                     violation_description="", severity="low", remediation="")

        # Check reasoning for inference language
        check_text = (reasoning or "") + " " + value

        for pattern in self.DIAGNOSIS_INFERENCE_PATTERNS:
            if re.search(pattern, check_text):
                return SafetyCheckResult(
                    passed=False,
                    violation_type=SafetyViolationType.INFERRED_DIAGNOSIS,
                    violation_description="Diagnosis appears to be inferred rather than explicitly stated",
                    severity="critical",
                    remediation="Only extract diagnoses that are explicitly stated in the source text"
                )

        return SafetyCheckResult(passed=True, violation_type=None,
                                 violation_description="", severity="low", remediation="")

    def _check_hallucinated_value(
        self,
        value: str,
        source_text: str,
        field_type: FieldType
    ) -> SafetyCheckResult:
        """Check if value is hallucinated (not in source)."""
        if not self.config.never_hallucinate_missing:
            return SafetyCheckResult(passed=True, violation_type=None,
                                     violation_description="", severity="low", remediation="")

        value_lower = value.lower().strip()
        source_lower = source_text.lower()

        # For numeric values, check if the number appears in source
        if field_type in [FieldType.DOSAGE, FieldType.LAB_VALUE, FieldType.VITAL_SIGN]:
            numbers = re.findall(r'\d+\.?\d*', value)
            for num in numbers:
                if num not in source_text:
                    return SafetyCheckResult(
                        passed=False,
                        violation_type=SafetyViolationType.HALLUCINATED_VALUE,
                        violation_description=f"Numeric value '{num}' not found in source text",
                        severity="critical",
                        remediation="Only extract numeric values that appear in the source document"
                    )

        # For medications, check if name appears
        elif field_type == FieldType.MEDICATION:
            # Extract medication name (first word usually)
            med_name = value.split()[0].lower() if value.split() else ""
            if len(med_name) > 3 and med_name not in source_lower:
                return SafetyCheckResult(
                    passed=False,
                    violation_type=SafetyViolationType.FABRICATED_MEDICATION,
                    violation_description=f"Medication '{med_name}' not found in source text",
                    severity="critical",
                    remediation="Only extract medications that are mentioned in the source document"
                )

        return SafetyCheckResult(passed=True, violation_type=None,
                                 violation_description="", severity="low", remediation="")

    def _check_abbreviation_normalization(
        self,
        value: str,
        source_text: str
    ) -> SafetyCheckResult:
        """Check if ambiguous abbreviations were normalized."""
        if not self.config.never_normalize_abbreviations:
            return SafetyCheckResult(passed=True, violation_type=None,
                                     violation_description="", severity="low", remediation="")

        value_upper = value.upper()

        for abbrev, expansions in self.AMBIGUOUS_ABBREVIATIONS.items():
            # Check if value contains an expansion of an ambiguous abbreviation
            for expansion in expansions:
                if expansion.lower() in value.lower():
                    # Check if the abbreviation (not expansion) is in source
                    if abbrev in source_text.upper() and expansion.lower() not in source_text.lower():
                        return SafetyCheckResult(
                            passed=False,
                            violation_type=SafetyViolationType.NORMALIZED_ABBREVIATION,
                            violation_description=f"Ambiguous abbreviation '{abbrev}' was normalized to '{expansion}'",
                            severity="high",
                            remediation=f"Keep abbreviation '{abbrev}' as-is; it has multiple meanings: {', '.join(expansions)}"
                        )

        return SafetyCheckResult(passed=True, violation_type=None,
                                 violation_description="", severity="low", remediation="")

    def _check_clinician_override(
        self,
        value: str,
        source_text: str,
        field_type: FieldType
    ) -> SafetyCheckResult:
        """Check if clinician-written text was overridden."""
        if not self.config.never_override_clinician_text:
            return SafetyCheckResult(passed=True, violation_type=None,
                                     violation_description="", severity="low", remediation="")

        # This is a heuristic check - look for modifications
        # If value is significantly different from any matching text in source
        value_words = set(value.lower().split())
        source_words = set(source_text.lower().split())

        # Check for invented words not in source
        invented_words = value_words - source_words

        # Allow common connector words
        common_words = {"the", "a", "an", "is", "are", "was", "were", "of", "in", "for", "to", "with"}
        invented_meaningful = invented_words - common_words

        if len(invented_meaningful) > len(value_words) * 0.5:
            return SafetyCheckResult(
                passed=False,
                violation_type=SafetyViolationType.OVERRIDDEN_CLINICIAN_TEXT,
                violation_description="Extracted value contains significant content not in source",
                severity="high",
                remediation="Extract exact text from source document without modification"
            )

        return SafetyCheckResult(passed=True, violation_type=None,
                                 violation_description="", severity="low", remediation="")

    def _check_fabricated_values(
        self,
        value: str,
        source_text: str,
        field_type: FieldType
    ) -> SafetyCheckResult:
        """Check for fabricated or estimated values."""
        # Check for fabrication language patterns
        for pattern in self.FABRICATION_PATTERNS:
            if re.search(pattern, value):
                return SafetyCheckResult(
                    passed=False,
                    violation_type=SafetyViolationType.HALLUCINATED_VALUE,
                    violation_description="Value appears to be estimated or fabricated",
                    severity="critical",
                    remediation="Only extract explicit values from source text"
                )

        return SafetyCheckResult(passed=True, violation_type=None,
                                 violation_description="", severity="low", remediation="")

    def _log_audit(self, field_name: str, result: SafetyCheckResult):
        """Log safety check for audit trail."""
        from datetime import datetime

        entry = SafetyAuditEntry(
            field_name=field_name,
            check_performed=result.violation_type.value if result.violation_type else "safety_check",
            result="failed" if not result.passed else "passed",
            details=result.violation_description,
            timestamp=datetime.utcnow().isoformat()
        )
        self._audit_log.append(entry)

    def get_audit_log(self) -> List[SafetyAuditEntry]:
        """Get the audit log."""
        return self._audit_log.copy()

    def clear_audit_log(self):
        """Clear the audit log."""
        self._audit_log.clear()

    def check_batch(
        self,
        extractions: List[Tuple[str, FieldType, str, str]]
    ) -> List[SafetyCheckResult]:
        """Check multiple extractions for safety."""
        results = []
        for field_name, field_type, value, source in extractions:
            result = self.check_safety(field_name, field_type, value, source)
            results.append(result)
        return results

    def get_safety_summary(self) -> Dict[str, Any]:
        """Get summary of safety check results."""
        total = len(self._audit_log)
        failures = sum(1 for e in self._audit_log if e.result == "failed")

        violation_counts = {}
        for entry in self._audit_log:
            if entry.result == "failed":
                vtype = entry.check_performed
                violation_counts[vtype] = violation_counts.get(vtype, 0) + 1

        return {
            "total_checks": total,
            "passed": total - failures,
            "failed": failures,
            "violation_types": violation_counts,
            "safety_rate": (total - failures) / total if total > 0 else 1.0
        }
