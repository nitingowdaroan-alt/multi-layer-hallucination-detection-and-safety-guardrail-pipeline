"""
Extraction Prompt Templates

Prompt templates for the clinical data extraction LLM.
These prompts enforce strict grounding and safety rules.
"""

from typing import Optional, List, Dict, Any
from dataclasses import dataclass

from ..core.models import ExtractionField, FieldType


@dataclass
class ExtractionPromptTemplate:
    """Templates for extraction prompts."""

    @staticmethod
    def get_system_prompt() -> str:
        """
        System prompt for the extraction LLM.

        This prompt establishes the safety rules that the LLM must follow.
        """
        return """You are a clinical data extraction system designed for healthcare environments.
Your task is to extract specific structured information from clinical documents.

═══════════════════════════════════════════════════════════════════════════════
                           CRITICAL SAFETY RULES
═══════════════════════════════════════════════════════════════════════════════

You MUST follow these rules without exception. Patient safety depends on your accuracy.

1. EXTRACTION SCOPE
   ✓ ONLY extract information that is EXPLICITLY stated in the provided context
   ✗ NEVER use external medical knowledge or training data
   ✗ NEVER infer, assume, or deduce information
   ✗ NEVER fill in "typical" or "standard" values

2. DIAGNOSIS HANDLING
   ✗ NEVER infer diagnoses from symptoms, lab results, or medications
   ✓ ONLY extract diagnoses that are explicitly labeled as such
   ✗ NEVER interpret "rule out" or "differential" as confirmed diagnoses

3. MEDICATION & DOSAGE
   ✗ NEVER guess dosages that aren't explicitly stated
   ✗ NEVER normalize medication names to generic equivalents
   ✓ Extract exact medication names and dosages as written

4. ABBREVIATIONS
   ✗ NEVER expand ambiguous medical abbreviations
   ✗ NEVER assume what an abbreviation means
   ✓ Keep abbreviations exactly as they appear in the source

5. WHEN INFORMATION IS MISSING
   ✓ Respond with "NOT_FOUND" if the field is not in the text
   ✓ Respond with "AMBIGUOUS: [reason]" if information is unclear
   ✓ Respond with "PARTIAL: [value]" if only part is available
   ✗ NEVER fabricate or guess missing information

6. CLINICIAN TEXT
   ✗ NEVER modify, correct, or "improve" clinician-written text
   ✓ Preserve exact wording, spelling, and formatting

═══════════════════════════════════════════════════════════════════════════════

OUTPUT FORMAT:
Respond with valid JSON only. Include source citations.

{
    "extracted_value": "<value | NOT_FOUND | AMBIGUOUS: reason | PARTIAL: value>",
    "source_citations": [<source numbers where found>],
    "confidence_note": "<any relevant notes>"
}

Remember: When in doubt, abstain. It is safer to say "NOT_FOUND" than to hallucinate."""

    @staticmethod
    def get_user_prompt(
        field: ExtractionField,
        context: str,
        additional_instructions: Optional[str] = None
    ) -> str:
        """
        Generate user prompt for field extraction.

        Args:
            field: The field to extract
            context: Retrieved context from the document
            additional_instructions: Optional additional instructions

        Returns:
            Formatted user prompt
        """
        field_specific = ExtractionPromptTemplate._get_field_specific_instructions(
            field.field_type
        )

        prompt = f"""Extract the following field from the clinical text:

╔══════════════════════════════════════════════════════════════════════════════╗
║  FIELD TO EXTRACT                                                            ║
╚══════════════════════════════════════════════════════════════════════════════╝

Field Name: {field.name}
Field Type: {field.field_type.value}
Description: {field.description}
{f"Expected Format: {field.expected_format}" if field.expected_format else ""}

{field_specific}

╔══════════════════════════════════════════════════════════════════════════════╗
║  CLINICAL TEXT (Retrieved Context)                                           ║
╚══════════════════════════════════════════════════════════════════════════════╝

{context}

╔══════════════════════════════════════════════════════════════════════════════╗
║  INSTRUCTIONS                                                                ║
╚══════════════════════════════════════════════════════════════════════════════╝

1. Search the provided context for information about "{field.name}"
2. Extract ONLY what is explicitly stated in the text
3. Include the source number(s) where you found the information
4. If the information is not found, respond with: NOT_FOUND

{additional_instructions or ""}

Respond with JSON only:
{{
    "extracted_value": "<value or NOT_FOUND or AMBIGUOUS: reason or PARTIAL: value>",
    "source_citations": [<source numbers>],
    "confidence_note": "<notes>"
}}"""

        return prompt

    @staticmethod
    def _get_field_specific_instructions(field_type: FieldType) -> str:
        """Get field-type specific extraction instructions."""
        instructions = {
            FieldType.MEDICATION: """
┌─ MEDICATION EXTRACTION RULES ─────────────────────────────────────────────┐
│ • Extract the exact medication name as written                            │
│ • Do NOT substitute generic for brand names or vice versa                 │
│ • Include route of administration if stated (PO, IV, etc.)               │
│ • Do NOT expand abbreviations                                             │
└───────────────────────────────────────────────────────────────────────────┘""",

            FieldType.DOSAGE: """
┌─ DOSAGE EXTRACTION RULES ─────────────────────────────────────────────────┐
│ • Extract the exact numeric value and unit as stated                      │
│ • Include frequency if stated (e.g., "twice daily", "BID")               │
│ • Do NOT convert units or calculate doses                                 │
│ • If multiple dosages exist, extract the most recent/current              │
└───────────────────────────────────────────────────────────────────────────┘""",

            FieldType.DIAGNOSIS: """
┌─ DIAGNOSIS EXTRACTION RULES ──────────────────────────────────────────────┐
│ • ONLY extract explicitly stated diagnoses                                │
│ • Look for sections: "Diagnosis", "Assessment", "Impression"              │
│ • Do NOT extract from "Rule out" or "Differential" lists                  │
│ • Do NOT infer diagnoses from symptoms or treatments                      │
└───────────────────────────────────────────────────────────────────────────┘""",

            FieldType.LAB_VALUE: """
┌─ LAB VALUE EXTRACTION RULES ──────────────────────────────────────────────┐
│ • Extract the exact numeric value and unit                                │
│ • Include reference range flags if present (H, L, etc.)                   │
│ • Extract the most recent value if multiple exist                         │
│ • Do NOT calculate or derive values                                       │
└───────────────────────────────────────────────────────────────────────────┘""",

            FieldType.ALLERGY: """
┌─ ALLERGY EXTRACTION RULES ────────────────────────────────────────────────┐
│ • Extract allergen name exactly as stated                                 │
│ • Include reaction type if documented (e.g., "rash", "anaphylaxis")      │
│ • "NKDA" or "NKA" = No Known Drug Allergies                              │
│ • Do NOT assume cross-allergies                                           │
└───────────────────────────────────────────────────────────────────────────┘""",

            FieldType.VITAL_SIGN: """
┌─ VITAL SIGN EXTRACTION RULES ─────────────────────────────────────────────┐
│ • Extract exact numeric value with unit                                   │
│ • For BP, extract as "systolic/diastolic" format                         │
│ • Include time of measurement if available                                │
│ • Extract most recent if multiple readings exist                          │
└───────────────────────────────────────────────────────────────────────────┘""",

            FieldType.DATE: """
┌─ DATE EXTRACTION RULES ───────────────────────────────────────────────────┐
│ • Extract date in the format it appears in the source                     │
│ • Do NOT convert date formats                                             │
│ • Include time if specified                                               │
│ • Be careful with relative dates ("yesterday", "last week")              │
└───────────────────────────────────────────────────────────────────────────┘""",

            FieldType.PROCEDURE: """
┌─ PROCEDURE EXTRACTION RULES ──────────────────────────────────────────────┐
│ • Extract procedure name as stated                                        │
│ • Include date/time if available                                          │
│ • Note if procedure is planned vs. completed                              │
│ • Do NOT expand procedure abbreviations                                   │
└───────────────────────────────────────────────────────────────────────────┘""",
        }

        return instructions.get(field_type, """
┌─ GENERAL EXTRACTION RULES ────────────────────────────────────────────────┐
│ • Extract information exactly as it appears in the source                 │
│ • Do NOT interpret, normalize, or transform the data                      │
│ • Preserve original formatting and terminology                            │
└───────────────────────────────────────────────────────────────────────────┘""")

    @staticmethod
    def get_batch_extraction_prompt(
        fields: List[ExtractionField],
        context: str
    ) -> str:
        """
        Generate prompt for extracting multiple fields at once.

        Args:
            fields: List of fields to extract
            context: Retrieved context

        Returns:
            Formatted prompt for batch extraction
        """
        field_list = "\n".join([
            f"  {i+1}. {f.name} ({f.field_type.value}): {f.description}"
            for i, f in enumerate(fields)
        ])

        return f"""Extract the following fields from the clinical text:

FIELDS TO EXTRACT:
{field_list}

CLINICAL TEXT:
{context}

RULES:
- Extract ONLY what is explicitly stated
- Use "NOT_FOUND" for missing fields
- Do NOT infer or guess

Respond with JSON:
{{
    "extractions": [
        {{
            "field_name": "<name>",
            "extracted_value": "<value or NOT_FOUND>",
            "source_citation": "<where found>"
        }}
    ]
}}"""
