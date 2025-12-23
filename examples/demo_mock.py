#!/usr/bin/env python3
"""
Mock Demo - Tests pipeline logic without real API calls.

This demonstrates the full pipeline flow using simulated LLM responses.
"""

import os
import sys
sys.path.insert(0, os.path.join(os.path.dirname(__file__), '..'))

from src.core.models import (
    ClinicalDocument,
    ExtractionSchema,
    ExtractionField,
    ExtractedField,
    RetrievedChunk,
    FieldType,
    SeverityLevel,
    DecisionType,
    HallucinationDetectionResult,
    FaithfulnessResult,
    ContradictionResult,
    ConsistencyResult,
    RuleValidationResult,
)
from src.rag.document_processor import ClinicalDocumentProcessor
from src.hallucination_detection.clinical_rules import ClinicalRuleValidator
from src.safety_guardrails.confidence import ConfidenceScorer
from src.safety_guardrails.abstention import AbstentionEngine
from src.safety_guardrails.healthcare_safety import HealthcareSafetyGuard
from src.utils.explainability import ExplainabilityGenerator


def create_mock_detection_result(
    field_name: str,
    faithfulness_score: float,
    contradiction_detected: bool,
    agreement_rate: float,
    rule_passed: bool
) -> HallucinationDetectionResult:
    """Create a mock hallucination detection result."""
    return HallucinationDetectionResult(
        field_name=field_name,
        faithfulness=FaithfulnessResult(
            field_name=field_name,
            faithfulness_score=faithfulness_score,
            supporting_chunks=[],
            max_chunk_similarity=faithfulness_score,
            avg_chunk_similarity=faithfulness_score * 0.9,
            grounding_evidence="Mock evidence from source document",
            is_grounded=faithfulness_score > 0.6
        ),
        contradiction=ContradictionResult(
            field_name=field_name,
            contradiction_detected=contradiction_detected,
            contradiction_reason="Contradiction found" if contradiction_detected else None,
            verifier_confidence=0.9,
            evidence_alignment="contradicted" if contradiction_detected else "aligned",
            explanation="Mock verification result"
        ),
        consistency=ConsistencyResult(
            field_name=field_name,
            num_runs=5,
            unique_values=["value1"] if agreement_rate > 0.8 else ["value1", "value2", "value3"],
            agreement_rate=agreement_rate,
            variance_score=1 - agreement_rate,
            most_common_value="1000mg",
            confidence_decay=0.05,
            is_stable=agreement_rate > 0.8
        ),
        rule_validation=RuleValidationResult(
            field_name=field_name,
            field_type=FieldType.DOSAGE,
            passed=rule_passed,
            rules_applied=["dosage_format", "medication_safe_range"],
            violations=[] if rule_passed else ["Violation detected"],
            warnings=[]
        ),
        composite_score=faithfulness_score * 0.4 + (0 if contradiction_detected else 0.3) + agreement_rate * 0.2 + (0.1 if rule_passed else 0),
        is_hallucinated=faithfulness_score < 0.5 or contradiction_detected,
        hallucination_type="fabricated" if faithfulness_score < 0.5 else None
    )


def run_mock_pipeline():
    """Run the full pipeline with mock data."""
    print("=" * 70)
    print("  MOCK PIPELINE DEMONSTRATION")
    print("  (Simulates full pipeline without API calls)")
    print("=" * 70)

    # Sample document
    document = ClinicalDocument(
        document_id="mock_001",
        document_type="discharge_summary",
        content="""
DISCHARGE SUMMARY

Patient: Test Patient
Date: 01/20/2024

DISCHARGE MEDICATIONS:
1. Metformin 1000mg twice daily
2. Lisinopril 20mg once daily
3. Atorvastatin 40mg at bedtime

ALLERGIES: Penicillin (anaphylaxis)

DISCHARGE DIAGNOSIS:
1. Type 2 Diabetes Mellitus
2. Hypertension

LABORATORY VALUES:
- HbA1c: 8.2%
- Glucose: 186 mg/dL
        """
    )

    # Process document
    print("\n[1] DOCUMENT PROCESSING")
    print("-" * 70)
    processor = ClinicalDocumentProcessor()
    chunks = processor.process_document(document)
    print(f"    Document ID: {document.document_id}")
    print(f"    Chunks created: {len(chunks)}")
    for chunk in chunks[:3]:
        section = chunk.metadata.get("section_name", "Unknown")
        print(f"      - {section}: {len(chunk.content)} chars")

    # Define extraction fields
    fields = [
        ("metformin_dose", FieldType.DOSAGE, "1000mg", 0.92, False, 0.95, True),
        ("lisinopril_dose", FieldType.DOSAGE, "20mg", 0.88, False, 0.90, True),
        ("allergy", FieldType.ALLERGY, "Penicillin", 0.95, False, 1.0, True),
        ("aspirin_dose", FieldType.DOSAGE, "325mg", 0.25, True, 0.40, True),  # HALLUCINATION
        ("glucose", FieldType.LAB_VALUE, "186 mg/dL", 0.90, False, 0.95, True),
        ("unsafe_dose", FieldType.DOSAGE, "50000mg", 0.85, False, 0.90, False),  # RULE VIOLATION
    ]

    # Initialize components
    confidence_scorer = ConfidenceScorer()
    abstention_engine = AbstentionEngine()
    safety_guard = HealthcareSafetyGuard()
    explainer = ExplainabilityGenerator()

    print("\n[2] EXTRACTION & HALLUCINATION DETECTION")
    print("-" * 70)

    results = []
    for field_name, field_type, value, faith, contra, agree, rule_pass in fields:
        # Create mock detection result
        detection = create_mock_detection_result(
            field_name, faith, contra, agree, rule_pass
        )

        # Compute confidence
        confidence = confidence_scorer.compute_confidence(
            detection, field_type, SeverityLevel.HIGH
        )

        # Check abstention
        abstention = abstention_engine.should_abstain(
            confidence, detection, field_type, SeverityLevel.HIGH
        )

        # Determine final decision
        if not rule_pass:
            decision = DecisionType.REJECT
        elif abstention.should_abstain:
            decision = DecisionType.ABSTAIN
        elif confidence.final_score >= 0.85:
            decision = DecisionType.ACCEPT
        else:
            decision = DecisionType.FLAG_FOR_REVIEW

        results.append({
            "field": field_name,
            "value": value,
            "decision": decision,
            "confidence": confidence.final_score,
            "hallucinated": detection.is_hallucinated,
            "abstention_reason": abstention.reason.value if abstention.reason else None
        })

        # Print result
        status_icons = {
            DecisionType.ACCEPT: "✓",
            DecisionType.FLAG_FOR_REVIEW: "⚠",
            DecisionType.ABSTAIN: "○",
            DecisionType.REJECT: "✗"
        }
        icon = status_icons[decision]
        print(f"\n    {icon} {field_name}: '{value}'")
        print(f"      Decision: {decision.value}")
        print(f"      Confidence: {confidence.final_score:.1%}")
        print(f"      Hallucinated: {detection.is_hallucinated}")
        if detection.is_hallucinated:
            print(f"      Hallucination Type: {detection.hallucination_type}")
        if abstention.reason:
            print(f"      Abstention Reason: {abstention.reason.value}")

    # Summary
    print("\n" + "=" * 70)
    print("[3] EXTRACTION SUMMARY")
    print("=" * 70)

    accepted = sum(1 for r in results if r["decision"] == DecisionType.ACCEPT)
    flagged = sum(1 for r in results if r["decision"] == DecisionType.FLAG_FOR_REVIEW)
    abstained = sum(1 for r in results if r["decision"] == DecisionType.ABSTAIN)
    rejected = sum(1 for r in results if r["decision"] == DecisionType.REJECT)
    hallucinated = sum(1 for r in results if r["hallucinated"])

    print(f"""
    Total Fields:     {len(results)}
    ─────────────────────────────
    ✓ Accepted:       {accepted}
    ⚠ Flagged:        {flagged}
    ○ Abstained:      {abstained}
    ✗ Rejected:       {rejected}
    ─────────────────────────────
    Hallucinations:   {hallucinated}
    Safety Rate:      {(len(results) - rejected) / len(results):.1%}
    """)

    print("\n[4] DETAILED RESULTS TABLE")
    print("-" * 70)
    print(f"{'Field':<20} {'Value':<15} {'Decision':<15} {'Confidence':<12} {'Halluc.'}")
    print("-" * 70)
    for r in results:
        print(f"{r['field']:<20} {r['value']:<15} {r['decision'].value:<15} {r['confidence']:.1%}        {'Yes' if r['hallucinated'] else 'No'}")

    print("\n" + "=" * 70)
    print("  MOCK PIPELINE COMPLETE")
    print("=" * 70)
    print("""
    This demonstration showed:

    1. Document Processing
       - Clinical-aware chunking by section

    2. Hallucination Detection (4 Layers)
       - Faithfulness scoring
       - Contradiction detection
       - Self-consistency checks
       - Clinical rule validation

    3. Safety Decisions
       - Confidence scoring
       - Abstention logic
       - Accept/Flag/Abstain/Reject outcomes

    4. Results
       - 'aspirin_dose' was ABSTAINED (not in source = hallucination)
       - 'unsafe_dose' was REJECTED (violates clinical rules)
       - Valid extractions were ACCEPTED

    For real LLM-powered extraction, run locally with:
        export OPENAI_API_KEY="your-key"
        python examples/demo.py
    """)


if __name__ == "__main__":
    run_mock_pipeline()
