"""
Verifier Prompt Templates

Prompt templates for the contradiction detection verifier LLM.
"""

from typing import Optional
from dataclasses import dataclass

from ..core.models import FieldType


@dataclass
class VerifierPromptTemplate:
    """Templates for verifier/contradiction detection prompts."""

    @staticmethod
    def get_system_prompt() -> str:
        """
        System prompt for the verifier LLM.

        The verifier checks if extracted values are supported by evidence.
        """
        return """You are a clinical data verification system. Your ONLY task is to verify
whether an extracted value is supported by the provided evidence text.

═══════════════════════════════════════════════════════════════════════════════
                           VERIFICATION RULES
═══════════════════════════════════════════════════════════════════════════════

1. WHAT YOU CAN DO:
   ✓ Check if a value appears in the evidence text
   ✓ Check if a value is explicitly stated
   ✓ Identify when evidence contradicts the extracted value
   ✓ Identify when evidence does not mention the value

2. WHAT YOU CANNOT DO:
   ✗ Use medical knowledge to infer relationships
   ✗ Assume equivalence between similar terms
   ✗ Interpret abbreviations
   ✗ Judge clinical correctness
   ✗ Use external knowledge

3. VERIFICATION OUTCOMES:

   ALIGNED - The extracted value is explicitly stated in the evidence
   Example: Evidence says "Metformin 500mg", extraction is "500mg" → ALIGNED

   CONTRADICTED - The evidence states something different
   Example: Evidence says "500mg", extraction is "50mg" → CONTRADICTED
   Example: Evidence says "aspirin", extraction is "ibuprofen" → CONTRADICTED

   UNSUPPORTED - The evidence does not contain this information
   Example: Evidence discusses labs, extraction is about medications → UNSUPPORTED

4. BE LITERAL:
   - "Metformin" and "metformin HCl" are different unless source shows equivalence
   - "500mg" and "500 mg" are the same (spacing doesn't matter)
   - "QD" and "once daily" require source to show they're equivalent

═══════════════════════════════════════════════════════════════════════════════

OUTPUT FORMAT:
Respond with JSON only:

{
    "evidence_alignment": "aligned" | "contradicted" | "unsupported",
    "contradiction_detected": true | false,
    "confidence": <0.0 to 1.0>,
    "explanation": "<brief explanation>",
    "specific_issue": "<specific problem if any, null otherwise>"
}"""

    @staticmethod
    def get_user_prompt(
        field_name: str,
        field_type: FieldType,
        extracted_value: str,
        evidence: str
    ) -> str:
        """
        Generate user prompt for verification.

        Args:
            field_name: Name of the field
            field_type: Type of the field
            extracted_value: Value to verify
            evidence: Evidence text to check against

        Returns:
            Formatted verification prompt
        """
        field_guidance = VerifierPromptTemplate._get_field_guidance(field_type)

        return f"""Verify if the following extracted value is supported by the evidence.

╔══════════════════════════════════════════════════════════════════════════════╗
║  VERIFICATION REQUEST                                                        ║
╚══════════════════════════════════════════════════════════════════════════════╝

Field Name: {field_name}
Field Type: {field_type.value}
Extracted Value: {extracted_value}

{field_guidance}

╔══════════════════════════════════════════════════════════════════════════════╗
║  EVIDENCE TEXT                                                               ║
╚══════════════════════════════════════════════════════════════════════════════╝

{evidence}

╔══════════════════════════════════════════════════════════════════════════════╗
║  VERIFICATION TASK                                                           ║
╚══════════════════════════════════════════════════════════════════════════════╝

Determine if "{extracted_value}" is:
1. ALIGNED - Explicitly stated in the evidence
2. CONTRADICTED - Evidence says something different
3. UNSUPPORTED - Evidence doesn't mention this

Respond with JSON only:
{{
    "evidence_alignment": "aligned" | "contradicted" | "unsupported",
    "contradiction_detected": true | false,
    "confidence": <0.0 to 1.0>,
    "explanation": "<brief explanation>",
    "specific_issue": "<specific problem, or null>"
}}"""

    @staticmethod
    def _get_field_guidance(field_type: FieldType) -> str:
        """Get field-specific verification guidance."""
        guidance = {
            FieldType.DOSAGE: """
┌─ DOSAGE VERIFICATION ─────────────────────────────────────────────────────┐
│ • Check if EXACT numeric value appears in evidence                        │
│ • "500mg" vs "50mg" = CONTRADICTION (different numbers)                   │
│ • "500mg" vs "500 mg" = ALIGNED (same, spacing irrelevant)               │
│ • Missing dosage in evidence = UNSUPPORTED                                │
└───────────────────────────────────────────────────────────────────────────┘""",

            FieldType.MEDICATION: """
┌─ MEDICATION VERIFICATION ─────────────────────────────────────────────────┐
│ • Check if medication name appears in evidence                            │
│ • "Aspirin" vs "Acetaminophen" = CONTRADICTION (different drugs)         │
│ • Case differences are acceptable (aspirin = Aspirin)                     │
│ • Do NOT assume generic/brand equivalence                                 │
└───────────────────────────────────────────────────────────────────────────┘""",

            FieldType.DIAGNOSIS: """
┌─ DIAGNOSIS VERIFICATION ──────────────────────────────────────────────────┐
│ • Check if diagnosis is explicitly stated in evidence                     │
│ • Symptoms are NOT diagnoses                                              │
│ • "Rule out X" does NOT confirm X                                         │
│ • Different diagnoses = CONTRADICTION                                     │
└───────────────────────────────────────────────────────────────────────────┘""",

            FieldType.LAB_VALUE: """
┌─ LAB VALUE VERIFICATION ──────────────────────────────────────────────────┐
│ • Check if EXACT value and unit appear in evidence                        │
│ • "7.8%" vs "8.7%" = CONTRADICTION                                        │
│ • Same number, different units = CONTRADICTION                            │
│ • Value not in evidence = UNSUPPORTED                                     │
└───────────────────────────────────────────────────────────────────────────┘""",

            FieldType.ALLERGY: """
┌─ ALLERGY VERIFICATION ────────────────────────────────────────────────────┐
│ • Check if allergen is explicitly stated                                  │
│ • Similar substances are NOT the same (penicillin ≠ amoxicillin)         │
│ • "NKDA" = No Known Drug Allergies                                        │
│ • Allergy not mentioned = UNSUPPORTED                                     │
└───────────────────────────────────────────────────────────────────────────┘""",
        }

        return guidance.get(field_type, """
┌─ GENERAL VERIFICATION ────────────────────────────────────────────────────┐
│ • Check for exact or near-exact match in evidence                         │
│ • Different values = CONTRADICTION                                        │
│ • Value not mentioned = UNSUPPORTED                                       │
└───────────────────────────────────────────────────────────────────────────┘""")

    @staticmethod
    def get_batch_verification_prompt(
        verifications: list,
        evidence: str
    ) -> str:
        """
        Generate prompt for verifying multiple values at once.

        Args:
            verifications: List of (field_name, field_type, extracted_value) tuples
            evidence: Evidence text

        Returns:
            Formatted batch verification prompt
        """
        items = "\n".join([
            f"  {i+1}. {name} ({ftype}): \"{value}\""
            for i, (name, ftype, value) in enumerate(verifications)
        ])

        return f"""Verify the following extracted values against the evidence:

VALUES TO VERIFY:
{items}

EVIDENCE TEXT:
{evidence}

For each value, determine: ALIGNED, CONTRADICTED, or UNSUPPORTED

Respond with JSON:
{{
    "verifications": [
        {{
            "field_name": "<name>",
            "evidence_alignment": "aligned" | "contradicted" | "unsupported",
            "contradiction_detected": true | false,
            "explanation": "<brief reason>"
        }}
    ]
}}"""
