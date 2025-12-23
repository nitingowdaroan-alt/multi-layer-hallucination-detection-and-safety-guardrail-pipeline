"""
Layer 2: Contradiction Detection

Uses a secondary verifier LLM or NLI model to detect contradictions
between extracted values and source evidence.
"""

import json
from typing import List, Optional, Any
from dataclasses import dataclass

from ..core.models import RetrievedChunk, ContradictionResult, FieldType
from ..core.config import HallucinationDetectionConfig, ModelConfig


@dataclass
class VerifierResponse:
    """Response from the verifier model."""
    contradiction_detected: bool
    evidence_alignment: str  # "aligned", "contradicted", "unsupported"
    confidence: float
    explanation: str
    specific_issue: Optional[str] = None


class ContradictionDetector:
    """
    Detects contradictions between extracted values and source evidence.

    Uses a separate verifier LLM to:
    1. Check if the extracted value contradicts the evidence
    2. Detect fabricated values not in the source
    3. Identify altered dosages or invented diagnoses
    4. Provide explanations for detected issues

    CRITICAL: The verifier must NOT use medical inference.
    It can only check textual support/contradiction.
    """

    def __init__(
        self,
        model_config: Optional[ModelConfig] = None,
        detection_config: Optional[HallucinationDetectionConfig] = None,
        llm_client: Optional[Any] = None
    ):
        self.model_config = model_config or ModelConfig()
        self.detection_config = detection_config or HallucinationDetectionConfig()
        self._llm_client = llm_client

    @property
    def llm_client(self):
        """Lazy initialization of LLM client."""
        if self._llm_client is None:
            try:
                from openai import OpenAI
                self._llm_client = OpenAI()
            except ImportError:
                raise ImportError("OpenAI package required.")
        return self._llm_client

    def detect_contradiction(
        self,
        field_name: str,
        extracted_value: str,
        supporting_chunks: List[RetrievedChunk],
        field_type: FieldType
    ) -> ContradictionResult:
        """
        Detect if extracted value contradicts the source evidence.

        Args:
            field_name: Name of the extracted field
            extracted_value: Value extracted by the extraction LLM
            supporting_chunks: Evidence chunks used for extraction
            field_type: Type of the field

        Returns:
            ContradictionResult with detection outcome
        """
        if not extracted_value or not supporting_chunks:
            return ContradictionResult(
                field_name=field_name,
                contradiction_detected=True,
                contradiction_reason="No value or evidence provided",
                verifier_confidence=1.0,
                evidence_alignment="unsupported",
                explanation="Cannot verify without value and evidence"
            )

        # Format evidence for verifier
        evidence_text = self._format_evidence(supporting_chunks)

        # Call verifier
        verifier_response = self._call_verifier(
            field_name,
            extracted_value,
            evidence_text,
            field_type
        )

        return ContradictionResult(
            field_name=field_name,
            contradiction_detected=verifier_response.contradiction_detected,
            contradiction_reason=verifier_response.specific_issue,
            verifier_confidence=verifier_response.confidence,
            evidence_alignment=verifier_response.evidence_alignment,
            explanation=verifier_response.explanation
        )

    def _format_evidence(self, chunks: List[RetrievedChunk]) -> str:
        """Format evidence chunks for the verifier."""
        evidence_parts = []
        for i, chunk in enumerate(chunks, 1):
            section = chunk.metadata.get("section_name", "Unknown")
            evidence_parts.append(
                f"[Evidence {i}] (Section: {section})\n{chunk.content}"
            )
        return "\n\n".join(evidence_parts)

    def _call_verifier(
        self,
        field_name: str,
        extracted_value: str,
        evidence: str,
        field_type: FieldType
    ) -> VerifierResponse:
        """Call the verifier LLM to check for contradictions."""
        system_prompt = self._get_verifier_system_prompt()
        user_prompt = self._get_verifier_user_prompt(
            field_name,
            extracted_value,
            evidence,
            field_type
        )

        response = self.llm_client.chat.completions.create(
            model=self.model_config.verifier_model,
            messages=[
                {"role": "system", "content": system_prompt},
                {"role": "user", "content": user_prompt}
            ],
            temperature=self.model_config.verifier_temperature,
            max_tokens=1024
        )

        return self._parse_verifier_response(response.choices[0].message.content)

    def _get_verifier_system_prompt(self) -> str:
        """Get system prompt for the verifier LLM."""
        return """You are a clinical data verification system. Your ONLY task is to verify
if an extracted value is supported by the provided evidence text.

CRITICAL RULES - YOU MUST FOLLOW THESE:
1. You can ONLY check if the value appears in or is supported by the evidence
2. You CANNOT use any medical knowledge or inference
3. You CANNOT assume relationships between medical concepts
4. You CANNOT normalize or interpret medical terms
5. You must be LITERAL in your verification

VERIFICATION TYPES:
1. ALIGNED - The extracted value is explicitly stated in the evidence
2. CONTRADICTED - The evidence states something different from the extracted value
3. UNSUPPORTED - The evidence does not contain information about this value

EXAMPLES OF CONTRADICTIONS:
- Dosage: Evidence says "500mg" but extraction says "50mg"
- Medication: Evidence mentions "aspirin" but extraction says "acetaminophen"
- Value: Evidence says "normal" but extraction gives a specific abnormal number

EXAMPLES OF UNSUPPORTED:
- The field is not mentioned anywhere in the evidence
- The evidence discusses something else entirely

You must respond with a JSON object only. No additional text."""

    def _get_verifier_user_prompt(
        self,
        field_name: str,
        extracted_value: str,
        evidence: str,
        field_type: FieldType
    ) -> str:
        """Get user prompt for verification."""
        field_context = self._get_field_verification_context(field_type)

        return f"""Verify if the following extracted value is supported by the evidence.

FIELD NAME: {field_name}
FIELD TYPE: {field_type.value}
EXTRACTED VALUE: {extracted_value}

EVIDENCE TEXT:
{evidence}

{field_context}

Respond ONLY with this JSON format:
{{
    "evidence_alignment": "aligned" | "contradicted" | "unsupported",
    "contradiction_detected": true | false,
    "confidence": <0.0 to 1.0>,
    "explanation": "<brief explanation>",
    "specific_issue": "<specific problem if contradiction/unsupported, null otherwise>"
}}"""

    def _get_field_verification_context(self, field_type: FieldType) -> str:
        """Get verification context based on field type."""
        contexts = {
            FieldType.DOSAGE: """
DOSAGE VERIFICATION:
- Check if the exact numeric value and unit appear in the evidence
- A different number is a CONTRADICTION
- Missing dosage entirely is UNSUPPORTED""",

            FieldType.MEDICATION: """
MEDICATION VERIFICATION:
- Check if the medication name appears in the evidence
- A different medication name is a CONTRADICTION
- Similar drug names are NOT the same medication""",

            FieldType.DIAGNOSIS: """
DIAGNOSIS VERIFICATION:
- Check if the diagnosis is explicitly stated in the evidence
- Symptoms are NOT diagnoses
- Different diagnoses are CONTRADICTIONS""",

            FieldType.LAB_VALUE: """
LAB VALUE VERIFICATION:
- Check if the exact value and unit appear in the evidence
- Different values are CONTRADICTIONS
- Missing lab results are UNSUPPORTED""",

            FieldType.ALLERGY: """
ALLERGY VERIFICATION:
- Check if the allergy is explicitly stated
- Similar substances are NOT the same allergy
- "NKDA" means No Known Drug Allergies"""
        }

        return contexts.get(field_type, """
GENERAL VERIFICATION:
- Check for exact or near-exact match in evidence
- Different values are CONTRADICTIONS
- Missing information is UNSUPPORTED""")

    def _parse_verifier_response(self, response: str) -> VerifierResponse:
        """Parse the verifier's JSON response."""
        try:
            # Clean up response
            response_clean = response.strip()
            if response_clean.startswith("```json"):
                response_clean = response_clean[7:]
            if response_clean.startswith("```"):
                response_clean = response_clean[3:]
            if response_clean.endswith("```"):
                response_clean = response_clean[:-3]

            data = json.loads(response_clean.strip())

            return VerifierResponse(
                contradiction_detected=data.get("contradiction_detected", False),
                evidence_alignment=data.get("evidence_alignment", "unsupported"),
                confidence=float(data.get("confidence", 0.5)),
                explanation=data.get("explanation", ""),
                specific_issue=data.get("specific_issue")
            )

        except (json.JSONDecodeError, KeyError, TypeError):
            # Conservative fallback - treat as unsupported
            return VerifierResponse(
                contradiction_detected=True,
                evidence_alignment="unsupported",
                confidence=0.5,
                explanation="Failed to parse verifier response",
                specific_issue="Verification parsing error"
            )

    def detect_contradictions_batch(
        self,
        extractions: List[tuple]
    ) -> List[ContradictionResult]:
        """
        Detect contradictions for multiple extractions.

        Args:
            extractions: List of (field_name, value, chunks, field_type)

        Returns:
            List of ContradictionResults
        """
        results = []
        for field_name, value, chunks, field_type in extractions:
            result = self.detect_contradiction(
                field_name, value, chunks, field_type
            )
            results.append(result)
        return results

    def quick_contradiction_check(
        self,
        extracted_value: str,
        evidence_text: str
    ) -> bool:
        """
        Quick heuristic check for obvious contradictions.

        Useful as a pre-filter before calling the full verifier.
        """
        value_lower = extracted_value.lower().strip()
        evidence_lower = evidence_text.lower()

        # Check for exact presence
        if value_lower in evidence_lower:
            return False  # Likely not a contradiction

        # Check for numeric contradictions
        import re
        value_numbers = re.findall(r'\d+\.?\d*', extracted_value)
        evidence_numbers = re.findall(r'\d+\.?\d*', evidence_text)

        if value_numbers:
            # If the extracted value has numbers not in evidence, suspicious
            for num in value_numbers:
                if num not in evidence_numbers:
                    return True  # Potential contradiction

        return False  # No obvious contradiction
