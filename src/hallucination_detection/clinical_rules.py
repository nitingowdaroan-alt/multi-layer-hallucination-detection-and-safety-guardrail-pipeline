"""
Layer 4: Rule-Based Clinical Validation

Applies deterministic clinical validation rules to extracted values.
Rules can override LLM outputs when violations are detected.
"""

import re
from typing import List, Optional, Dict, Any, Tuple
from dataclasses import dataclass, field
from datetime import datetime, timedelta
from abc import ABC, abstractmethod

from ..core.models import RuleValidationResult, FieldType
from ..core.config import SafetyGuardrailConfig


@dataclass
class ValidationRule:
    """Base class for validation rules."""
    rule_id: str
    rule_name: str
    description: str
    severity: str  # "error", "warning"
    applicable_field_types: List[FieldType]


@dataclass
class RuleViolation:
    """Details of a rule violation."""
    rule_id: str
    rule_name: str
    violation_message: str
    severity: str
    suggested_correction: Optional[str] = None


class ClinicalRuleValidator:
    """
    Applies deterministic clinical validation rules.

    Rules include:
    - Medication dosage within safe bounds
    - Lab units match expected formats
    - Date/time consistency
    - Vital signs within physiological ranges
    - Allergy format validation

    Rules are AUTHORITATIVE and override LLM outputs when violated.
    """

    # Safe dosage ranges for common medications (typical adult doses)
    MEDICATION_SAFE_RANGES = {
        "metformin": {"min": 250, "max": 2550, "unit": "mg"},
        "lisinopril": {"min": 2.5, "max": 80, "unit": "mg"},
        "amlodipine": {"min": 2.5, "max": 10, "unit": "mg"},
        "metoprolol": {"min": 25, "max": 400, "unit": "mg"},
        "atorvastatin": {"min": 10, "max": 80, "unit": "mg"},
        "omeprazole": {"min": 10, "max": 40, "unit": "mg"},
        "levothyroxine": {"min": 12.5, "max": 300, "unit": "mcg"},
        "aspirin": {"min": 81, "max": 650, "unit": "mg"},
        "warfarin": {"min": 0.5, "max": 15, "unit": "mg"},
        "insulin": {"min": 1, "max": 100, "unit": "units"},
        "furosemide": {"min": 20, "max": 600, "unit": "mg"},
        "prednisone": {"min": 1, "max": 80, "unit": "mg"},
    }

    # Lab value normal ranges
    LAB_NORMAL_RANGES = {
        "glucose": {"min": 70, "max": 400, "unit": "mg/dL"},
        "hemoglobin": {"min": 4, "max": 20, "unit": "g/dL"},
        "hematocrit": {"min": 15, "max": 60, "unit": "%"},
        "wbc": {"min": 1000, "max": 50000, "unit": "cells/μL"},
        "platelets": {"min": 10000, "max": 1000000, "unit": "cells/μL"},
        "creatinine": {"min": 0.1, "max": 20, "unit": "mg/dL"},
        "bun": {"min": 2, "max": 150, "unit": "mg/dL"},
        "sodium": {"min": 110, "max": 170, "unit": "mEq/L"},
        "potassium": {"min": 2, "max": 8, "unit": "mEq/L"},
        "chloride": {"min": 80, "max": 130, "unit": "mEq/L"},
        "co2": {"min": 10, "max": 45, "unit": "mEq/L"},
        "alt": {"min": 0, "max": 2000, "unit": "U/L"},
        "ast": {"min": 0, "max": 2000, "unit": "U/L"},
        "bilirubin": {"min": 0, "max": 30, "unit": "mg/dL"},
        "albumin": {"min": 1, "max": 6, "unit": "g/dL"},
        "hba1c": {"min": 4, "max": 15, "unit": "%"},
        "tsh": {"min": 0.01, "max": 100, "unit": "mIU/L"},
        "inr": {"min": 0.5, "max": 10, "unit": ""},
    }

    # Vital sign ranges (physiological limits)
    VITAL_SIGN_RANGES = {
        "heart_rate": {"min": 20, "max": 250, "unit": "bpm"},
        "systolic_bp": {"min": 40, "max": 300, "unit": "mmHg"},
        "diastolic_bp": {"min": 20, "max": 200, "unit": "mmHg"},
        "respiratory_rate": {"min": 4, "max": 60, "unit": "/min"},
        "temperature": {"min": 86, "max": 113, "unit": "°F"},
        "temperature_c": {"min": 30, "max": 45, "unit": "°C"},
        "oxygen_saturation": {"min": 50, "max": 100, "unit": "%"},
        "weight_kg": {"min": 0.5, "max": 500, "unit": "kg"},
        "weight_lb": {"min": 1, "max": 1100, "unit": "lb"},
        "height_cm": {"min": 30, "max": 275, "unit": "cm"},
        "height_in": {"min": 12, "max": 108, "unit": "in"},
    }

    def __init__(self, config: Optional[SafetyGuardrailConfig] = None):
        self.config = config or SafetyGuardrailConfig()

    def validate(
        self,
        field_name: str,
        field_type: FieldType,
        extracted_value: Optional[str],
        context: Optional[Dict[str, Any]] = None
    ) -> RuleValidationResult:
        """
        Validate an extracted value against clinical rules.

        Args:
            field_name: Name of the field
            field_type: Type of the field
            extracted_value: The extracted value to validate
            context: Additional context (e.g., medication name for dosage)

        Returns:
            RuleValidationResult with validation outcome
        """
        context = context or {}
        rules_applied = []
        violations = []
        warnings = []
        corrected_value = None

        if extracted_value is None or extracted_value in ["NOT_FOUND", "AMBIGUOUS"]:
            return RuleValidationResult(
                field_name=field_name,
                field_type=field_type,
                passed=True,  # No value to validate
                rules_applied=["null_check"],
                violations=[],
                warnings=[]
            )

        # Apply rules based on field type
        if field_type == FieldType.DOSAGE:
            result = self._validate_dosage(extracted_value, context)
            rules_applied.extend(result["rules"])
            violations.extend(result["violations"])
            warnings.extend(result["warnings"])
            corrected_value = result.get("corrected")

        elif field_type == FieldType.LAB_VALUE:
            result = self._validate_lab_value(extracted_value, field_name)
            rules_applied.extend(result["rules"])
            violations.extend(result["violations"])
            warnings.extend(result["warnings"])

        elif field_type == FieldType.VITAL_SIGN:
            result = self._validate_vital_sign(extracted_value, field_name)
            rules_applied.extend(result["rules"])
            violations.extend(result["violations"])
            warnings.extend(result["warnings"])

        elif field_type == FieldType.DATE:
            result = self._validate_date(extracted_value)
            rules_applied.extend(result["rules"])
            violations.extend(result["violations"])
            warnings.extend(result["warnings"])

        elif field_type == FieldType.MEDICATION:
            result = self._validate_medication(extracted_value)
            rules_applied.extend(result["rules"])
            violations.extend(result["violations"])
            warnings.extend(result["warnings"])

        elif field_type == FieldType.ALLERGY:
            result = self._validate_allergy(extracted_value)
            rules_applied.extend(result["rules"])
            violations.extend(result["violations"])
            warnings.extend(result["warnings"])

        # General format validation
        format_result = self._validate_format(extracted_value, field_type)
        rules_applied.extend(format_result["rules"])
        violations.extend(format_result["violations"])
        warnings.extend(format_result["warnings"])

        # Determine if passed
        passed = len(violations) == 0

        return RuleValidationResult(
            field_name=field_name,
            field_type=field_type,
            passed=passed,
            rules_applied=rules_applied,
            violations=violations,
            warnings=warnings,
            corrected_value=corrected_value
        )

    def _validate_dosage(
        self,
        value: str,
        context: Dict[str, Any]
    ) -> Dict[str, Any]:
        """Validate medication dosage."""
        result = {"rules": [], "violations": [], "warnings": [], "corrected": None}

        # Extract numeric value and unit
        parsed = self._parse_dosage(value)
        if not parsed:
            result["rules"].append("dosage_format")
            result["violations"].append("Invalid dosage format: could not parse")
            return result

        numeric_value, unit = parsed
        result["rules"].append("dosage_format")

        # Check against known medication ranges
        medication_name = context.get("medication_name", "").lower()

        if medication_name and medication_name in self.MEDICATION_SAFE_RANGES:
            safe_range = self.MEDICATION_SAFE_RANGES[medication_name]
            result["rules"].append("medication_safe_range")

            if unit.lower() != safe_range["unit"].lower():
                result["warnings"].append(
                    f"Unit mismatch: expected {safe_range['unit']}, got {unit}"
                )
            else:
                if numeric_value < safe_range["min"]:
                    result["warnings"].append(
                        f"Dosage {numeric_value}{unit} below typical minimum "
                        f"({safe_range['min']}{unit}) for {medication_name}"
                    )
                elif numeric_value > safe_range["max"]:
                    result["violations"].append(
                        f"Dosage {numeric_value}{unit} exceeds maximum safe dose "
                        f"({safe_range['max']}{unit}) for {medication_name}"
                    )

        # General dosage sanity checks
        result["rules"].append("dosage_sanity")
        if numeric_value <= 0:
            result["violations"].append("Dosage must be positive")
        elif numeric_value > 10000 and unit.lower() == "mg":
            result["violations"].append(
                f"Extremely high dosage: {numeric_value}mg"
            )

        return result

    def _parse_dosage(self, value: str) -> Optional[Tuple[float, str]]:
        """Parse dosage string into (value, unit)."""
        # Pattern: number followed by unit
        pattern = r'(\d+\.?\d*)\s*(mg|mcg|g|ml|units?|tabs?|capsules?|cc|iu)'
        match = re.search(pattern, value, re.IGNORECASE)

        if match:
            return float(match.group(1)), match.group(2)
        return None

    def _validate_lab_value(
        self,
        value: str,
        field_name: str
    ) -> Dict[str, Any]:
        """Validate laboratory value."""
        result = {"rules": [], "violations": [], "warnings": []}

        # Parse lab value
        parsed = self._parse_lab_value(value)
        if not parsed:
            result["rules"].append("lab_format")
            result["warnings"].append("Could not parse lab value format")
            return result

        numeric_value, unit = parsed
        result["rules"].append("lab_format")

        # Normalize field name for lookup
        lab_key = self._normalize_lab_name(field_name)

        if lab_key in self.LAB_NORMAL_RANGES:
            lab_range = self.LAB_NORMAL_RANGES[lab_key]
            result["rules"].append("lab_physiological_range")

            # Check if value is physiologically possible
            if numeric_value < lab_range["min"]:
                result["violations"].append(
                    f"Lab value {numeric_value} below physiological minimum "
                    f"({lab_range['min']})"
                )
            elif numeric_value > lab_range["max"]:
                result["violations"].append(
                    f"Lab value {numeric_value} above physiological maximum "
                    f"({lab_range['max']})"
                )

        return result

    def _parse_lab_value(self, value: str) -> Optional[Tuple[float, str]]:
        """Parse lab value string."""
        # Pattern: number optionally followed by unit
        pattern = r'(\d+\.?\d*)\s*([a-zA-Z/%μ]+)?'
        match = re.search(pattern, value)

        if match:
            num = float(match.group(1))
            unit = match.group(2) or ""
            return num, unit
        return None

    def _normalize_lab_name(self, name: str) -> str:
        """Normalize lab name for lookup."""
        name_lower = name.lower()

        # Common aliases
        aliases = {
            "blood_glucose": "glucose",
            "fasting_glucose": "glucose",
            "hgb": "hemoglobin",
            "hct": "hematocrit",
            "white_blood_cell": "wbc",
            "plt": "platelets",
            "cre": "creatinine",
            "blood_urea_nitrogen": "bun",
            "na": "sodium",
            "k": "potassium",
            "cl": "chloride",
            "bicarb": "co2",
            "sgpt": "alt",
            "sgot": "ast",
            "total_bilirubin": "bilirubin",
            "alb": "albumin",
            "a1c": "hba1c",
            "hemoglobin_a1c": "hba1c",
        }

        for alias, canonical in aliases.items():
            if alias in name_lower:
                return canonical

        # Direct match
        for key in self.LAB_NORMAL_RANGES:
            if key in name_lower:
                return key

        return name_lower

    def _validate_vital_sign(
        self,
        value: str,
        field_name: str
    ) -> Dict[str, Any]:
        """Validate vital sign value."""
        result = {"rules": [], "violations": [], "warnings": []}

        parsed = self._parse_vital_sign(value)
        if not parsed:
            result["rules"].append("vital_format")
            result["warnings"].append("Could not parse vital sign format")
            return result

        numeric_value, unit = parsed
        result["rules"].append("vital_format")

        # Normalize field name
        vital_key = self._normalize_vital_name(field_name)

        if vital_key in self.VITAL_SIGN_RANGES:
            vital_range = self.VITAL_SIGN_RANGES[vital_key]
            result["rules"].append("vital_physiological_range")

            if numeric_value < vital_range["min"]:
                result["violations"].append(
                    f"Vital sign {numeric_value} below physiological minimum "
                    f"({vital_range['min']})"
                )
            elif numeric_value > vital_range["max"]:
                result["violations"].append(
                    f"Vital sign {numeric_value} above physiological maximum "
                    f"({vital_range['max']})"
                )

        return result

    def _parse_vital_sign(self, value: str) -> Optional[Tuple[float, str]]:
        """Parse vital sign value."""
        # Handle blood pressure format (120/80)
        bp_pattern = r'(\d+)\s*/\s*(\d+)'
        bp_match = re.search(bp_pattern, value)
        if bp_match:
            # Return systolic for validation
            return float(bp_match.group(1)), "mmHg"

        # Standard pattern
        pattern = r'(\d+\.?\d*)\s*([a-zA-Z°/%]+)?'
        match = re.search(pattern, value)
        if match:
            return float(match.group(1)), match.group(2) or ""
        return None

    def _normalize_vital_name(self, name: str) -> str:
        """Normalize vital sign name."""
        name_lower = name.lower()

        aliases = {
            "hr": "heart_rate",
            "pulse": "heart_rate",
            "bp": "systolic_bp",
            "blood_pressure": "systolic_bp",
            "sbp": "systolic_bp",
            "dbp": "diastolic_bp",
            "rr": "respiratory_rate",
            "resp": "respiratory_rate",
            "temp": "temperature",
            "spo2": "oxygen_saturation",
            "o2_sat": "oxygen_saturation",
            "wt": "weight_kg",
            "ht": "height_cm",
        }

        for alias, canonical in aliases.items():
            if alias in name_lower:
                return canonical

        return name_lower

    def _validate_date(self, value: str) -> Dict[str, Any]:
        """Validate date value."""
        result = {"rules": [], "violations": [], "warnings": []}
        result["rules"].append("date_format")

        # Try to parse date
        parsed_date = self._parse_date(value)
        if not parsed_date:
            result["warnings"].append("Could not parse date format")
            return result

        result["rules"].append("date_logic")

        # Check for future dates
        if not self.config.future_date_allowed:
            if parsed_date > datetime.now():
                result["violations"].append("Future date not allowed in clinical context")

        # Check for very old dates
        min_date = datetime.now() - timedelta(days=self.config.max_past_years * 365)
        if parsed_date < min_date:
            result["violations"].append(
                f"Date is more than {self.config.max_past_years} years in the past"
            )

        return result

    def _parse_date(self, value: str) -> Optional[datetime]:
        """Parse date string."""
        formats = [
            "%m/%d/%Y",
            "%m-%d-%Y",
            "%Y-%m-%d",
            "%m/%d/%y",
            "%B %d, %Y",
            "%b %d, %Y",
            "%d %B %Y",
        ]

        for fmt in formats:
            try:
                return datetime.strptime(value.strip(), fmt)
            except ValueError:
                continue

        return None

    def _validate_medication(self, value: str) -> Dict[str, Any]:
        """Validate medication name."""
        result = {"rules": [], "violations": [], "warnings": []}
        result["rules"].append("medication_format")

        # Check for suspicious patterns
        if re.search(r'\d{5,}', value):  # Long number sequences
            result["warnings"].append("Medication name contains unusual number sequence")

        if len(value) < 3:
            result["warnings"].append("Medication name unusually short")

        if len(value) > 100:
            result["violations"].append("Medication name unusually long")

        return result

    def _validate_allergy(self, value: str) -> Dict[str, Any]:
        """Validate allergy value."""
        result = {"rules": [], "violations": [], "warnings": []}
        result["rules"].append("allergy_format")

        # NKDA check
        if value.upper() in ["NKDA", "NKA", "NO KNOWN ALLERGIES", "NO KNOWN DRUG ALLERGIES"]:
            return result

        if len(value) < 2:
            result["warnings"].append("Allergy name unusually short")

        return result

    def _validate_format(
        self,
        value: str,
        field_type: FieldType
    ) -> Dict[str, Any]:
        """General format validation."""
        result = {"rules": [], "violations": [], "warnings": []}
        result["rules"].append("general_format")

        # Check for empty or whitespace only
        if not value or not value.strip():
            result["violations"].append("Value is empty or whitespace only")
            return result

        # Check for placeholder text
        placeholders = ["TBD", "N/A", "UNKNOWN", "???", "XXX", "PLACEHOLDER"]
        if value.upper() in placeholders:
            result["warnings"].append(f"Value appears to be placeholder: {value}")

        # Check for unusual characters
        if re.search(r'[<>{}|\\]', value):
            result["warnings"].append("Value contains unusual characters")

        return result

    def validate_batch(
        self,
        extractions: List[Tuple[str, FieldType, str, Dict[str, Any]]]
    ) -> List[RuleValidationResult]:
        """Validate multiple extractions."""
        results = []
        for field_name, field_type, value, context in extractions:
            result = self.validate(field_name, field_type, value, context)
            results.append(result)
        return results
