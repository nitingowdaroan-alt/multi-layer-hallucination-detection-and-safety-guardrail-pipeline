#!/usr/bin/env python3
"""
Demo Script for Healthcare AI Safety Guardrails Pipeline

This script demonstrates the hallucination detection and safety guardrails
system using sample clinical documents.

Usage:
    export OPENAI_API_KEY="your-api-key"
    python examples/demo.py
"""

import os
import sys

# Add src to path
sys.path.insert(0, os.path.join(os.path.dirname(__file__), '..'))

from src.core.models import (
    ClinicalDocument,
    ExtractionSchema,
    ExtractionField,
    FieldType,
    SeverityLevel,
)
from src.core.config import SystemConfig
from src.rag.document_processor import ClinicalDocumentProcessor
from src.hallucination_detection.clinical_rules import ClinicalRuleValidator
from src.safety_guardrails.healthcare_safety import HealthcareSafetyGuard
from src.evaluation.golden_dataset import create_sample_golden_dataset


def demo_document_processing():
    """Demonstrate clinical document processing."""
    print("=" * 60)
    print("DEMO 1: Clinical Document Processing")
    print("=" * 60)

    # Sample discharge summary
    document = ClinicalDocument(
        document_id="demo_001",
        document_type="discharge_summary",
        content="""
DISCHARGE SUMMARY

Patient: John Smith
MRN: 123456
Date of Discharge: 01/15/2024

CHIEF COMPLAINT:
Chest pain and shortness of breath

HISTORY OF PRESENT ILLNESS:
65-year-old male with history of hypertension and type 2 diabetes presents
with 2 days of chest pain. Pain is substernal, non-radiating, worse with exertion.

PAST MEDICAL HISTORY:
- Hypertension (diagnosed 2015)
- Type 2 Diabetes Mellitus (diagnosed 2018)
- Hyperlipidemia

MEDICATIONS ON ADMISSION:
1. Metformin 500mg twice daily
2. Lisinopril 10mg once daily
3. Atorvastatin 20mg at bedtime
4. Aspirin 81mg daily

ALLERGIES:
Penicillin - causes rash
Sulfa drugs - anaphylaxis

VITAL SIGNS:
- Blood Pressure: 145/92 mmHg
- Heart Rate: 78 bpm
- Temperature: 98.6°F
- Respiratory Rate: 16/min
- O2 Saturation: 97% on room air

LABORATORY DATA:
- Glucose: 142 mg/dL (H)
- HbA1c: 7.8%
- Creatinine: 1.1 mg/dL
- BUN: 18 mg/dL
- Sodium: 139 mEq/L
- Potassium: 4.2 mEq/L
- Troponin: <0.01 ng/mL (normal)

ASSESSMENT:
1. Chest pain - likely musculoskeletal, cardiac workup negative
2. Type 2 Diabetes Mellitus - suboptimally controlled
3. Hypertension - controlled on current regimen

DISCHARGE MEDICATIONS:
1. Metformin 500mg twice daily (continue)
2. Lisinopril 10mg once daily (continue)
3. Atorvastatin 40mg at bedtime (increased from 20mg)
4. Aspirin 81mg daily (continue)
5. Omeprazole 20mg daily (new - for GI protection)

DISCHARGE INSTRUCTIONS:
- Follow up with PCP in 1 week
- Follow up with Cardiology in 2 weeks
- Low sodium, diabetic diet
- Return to ED if chest pain recurs

DISCHARGE DIAGNOSIS:
- Atypical chest pain, non-cardiac
- Type 2 Diabetes Mellitus
- Hypertension
- Hyperlipidemia
        """
    )

    # Process document
    processor = ClinicalDocumentProcessor()
    chunks = processor.process_document(document)

    print(f"\nDocument ID: {document.document_id}")
    print(f"Document Type: {document.document_type}")
    print(f"Total chunks created: {len(chunks)}")
    print("\nChunk breakdown by section:")

    section_counts = {}
    for chunk in chunks:
        section = chunk.metadata.get("section_name", "Unknown")
        section_counts[section] = section_counts.get(section, 0) + 1

    for section, count in section_counts.items():
        print(f"  - {section}: {count} chunks")

    print("\nSample chunk content:")
    if chunks:
        sample = chunks[0]
        print(f"  Section: {sample.metadata.get('section_name')}")
        print(f"  Has medications: {sample.metadata.get('has_medications')}")
        print(f"  Has lab values: {sample.metadata.get('has_lab_values')}")
        print(f"  Content preview: {sample.content[:200]}...")

    return document, chunks


def demo_clinical_rule_validation():
    """Demonstrate clinical rule validation."""
    print("\n" + "=" * 60)
    print("DEMO 2: Clinical Rule Validation")
    print("=" * 60)

    validator = ClinicalRuleValidator()

    # Test cases
    test_cases = [
        # Valid cases
        ("metformin_dose", FieldType.DOSAGE, "500mg", {"medication_name": "metformin"}),
        ("glucose", FieldType.LAB_VALUE, "142 mg/dL", {}),
        ("heart_rate", FieldType.VITAL_SIGN, "78 bpm", {}),
        ("admission_date", FieldType.DATE, "01/15/2024", {}),

        # Invalid cases (should trigger violations)
        ("metformin_dose_unsafe", FieldType.DOSAGE, "50000mg", {"medication_name": "metformin"}),
        ("glucose_impossible", FieldType.LAB_VALUE, "5000 mg/dL", {}),
        ("heart_rate_impossible", FieldType.VITAL_SIGN, "500 bpm", {}),
        ("future_date", FieldType.DATE, "12/31/2099", {}),
    ]

    print("\nValidation Results:")
    print("-" * 60)

    for field_name, field_type, value, context in test_cases:
        result = validator.validate(field_name, field_type, value, context)
        status = "✓ PASSED" if result.passed else "✗ FAILED"
        print(f"\n{field_name}: {value}")
        print(f"  Status: {status}")
        print(f"  Rules applied: {', '.join(result.rules_applied)}")
        if result.violations:
            print(f"  Violations: {'; '.join(result.violations)}")
        if result.warnings:
            print(f"  Warnings: {'; '.join(result.warnings)}")


def demo_safety_guardrails():
    """Demonstrate healthcare safety guardrails."""
    print("\n" + "=" * 60)
    print("DEMO 3: Healthcare Safety Guardrails")
    print("=" * 60)

    safety_guard = HealthcareSafetyGuard()

    source_text = """
    DISCHARGE MEDICATIONS:
    1. Metformin 500mg twice daily
    2. Lisinopril 10mg once daily

    ALLERGIES: Penicillin (rash)

    DIAGNOSIS: Type 2 Diabetes Mellitus
    """

    # Test cases
    test_cases = [
        # Safe extractions (values in source)
        ("metformin_dose", FieldType.DOSAGE, "500mg", "Value explicitly in source"),
        ("allergy", FieldType.ALLERGY, "Penicillin", "Value explicitly in source"),

        # Potentially hallucinated (values NOT in source)
        ("aspirin_dose", FieldType.DOSAGE, "325mg", "Aspirin not mentioned in source"),
        ("fabricated_diagnosis", FieldType.DIAGNOSIS, "Coronary artery disease", "Not stated in source"),
    ]

    print("\nSafety Check Results:")
    print("-" * 60)

    for field_name, field_type, value, description in test_cases:
        result = safety_guard.check_safety(
            field_name=field_name,
            field_type=field_type,
            extracted_value=value,
            source_text=source_text
        )

        status = "✓ SAFE" if result.passed else "⚠ VIOLATION"
        print(f"\n{field_name}: '{value}'")
        print(f"  Description: {description}")
        print(f"  Status: {status}")
        if not result.passed:
            print(f"  Violation: {result.violation_type.value if result.violation_type else 'N/A'}")
            print(f"  Detail: {result.violation_description}")
            print(f"  Remediation: {result.remediation}")


def demo_extraction_schema():
    """Demonstrate extraction schema definition."""
    print("\n" + "=" * 60)
    print("DEMO 4: Extraction Schema Definition")
    print("=" * 60)

    schema = ExtractionSchema(
        schema_id="discharge_extraction_v1",
        name="Discharge Summary Extraction",
        description="Extract key information from discharge summaries",
        fields=[
            ExtractionField(
                name="primary_diagnosis",
                field_type=FieldType.DIAGNOSIS,
                description="Primary discharge diagnosis",
                severity=SeverityLevel.CRITICAL,
                required=True,
            ),
            ExtractionField(
                name="medications",
                field_type=FieldType.MEDICATION,
                description="List of discharge medications",
                severity=SeverityLevel.CRITICAL,
                required=True,
            ),
            ExtractionField(
                name="allergies",
                field_type=FieldType.ALLERGY,
                description="Patient allergies",
                severity=SeverityLevel.CRITICAL,
                required=True,
            ),
            ExtractionField(
                name="glucose_level",
                field_type=FieldType.LAB_VALUE,
                description="Blood glucose level",
                severity=SeverityLevel.HIGH,
                expected_format="XXX mg/dL",
            ),
            ExtractionField(
                name="blood_pressure",
                field_type=FieldType.VITAL_SIGN,
                description="Blood pressure reading",
                severity=SeverityLevel.HIGH,
                expected_format="XXX/XX mmHg",
            ),
            ExtractionField(
                name="follow_up_date",
                field_type=FieldType.DATE,
                description="Follow-up appointment date",
                severity=SeverityLevel.MEDIUM,
            ),
        ]
    )

    print(f"\nSchema: {schema.name}")
    print(f"ID: {schema.schema_id}")
    print(f"Total fields: {len(schema.fields)}")

    print("\nField definitions:")
    print("-" * 60)
    for field in schema.fields:
        print(f"\n  {field.name}")
        print(f"    Type: {field.field_type.value}")
        print(f"    Severity: {field.severity.value}")
        print(f"    Required: {field.required}")
        print(f"    Description: {field.description}")

    print(f"\nCritical fields: {len(schema.get_critical_fields())}")
    for field in schema.get_critical_fields():
        print(f"  - {field.name}")


def demo_golden_dataset():
    """Demonstrate golden dataset for evaluation."""
    print("\n" + "=" * 60)
    print("DEMO 5: Golden Dataset for Evaluation")
    print("=" * 60)

    dataset = create_sample_golden_dataset()
    stats = dataset.get_statistics()

    print(f"\nDataset: {stats['name']}")
    print(f"Total examples: {stats['total_examples']}")
    print(f"Total labels: {stats['total_labels']}")
    print(f"Expected abstention rate: {stats['abstention_rate']:.1%}")

    print("\nBy document type:")
    for doc_type, count in stats['by_document_type'].items():
        print(f"  - {doc_type}: {count}")

    print("\nBy field type:")
    for field_type, count in stats['by_field_type'].items():
        print(f"  - {field_type}: {count}")

    print("\nSample example:")
    if dataset.examples:
        example = dataset.examples[0]
        print(f"  ID: {example.example_id}")
        print(f"  Type: {example.document_type}")
        print(f"  Labels: {len(example.labels)}")
        for label in example.labels[:3]:
            print(f"    - {label.field_name}: '{label.expected_value}' (abstain: {label.should_abstain})")


def demo_full_pipeline_simulation():
    """Simulate the full pipeline flow (without LLM calls)."""
    print("\n" + "=" * 60)
    print("DEMO 6: Full Pipeline Simulation")
    print("=" * 60)

    print("""
    ┌─────────────────────────────────────────────────────────────┐
    │                    PIPELINE FLOW                            │
    ├─────────────────────────────────────────────────────────────┤
    │                                                             │
    │  1. DOCUMENT INPUT                                          │
    │     └─▶ Clinical document (EHR, discharge summary, etc.)   │
    │                                                             │
    │  2. RAG PIPELINE                                           │
    │     └─▶ Chunk document                                     │
    │     └─▶ Generate embeddings                                │
    │     └─▶ Retrieve relevant chunks for each field            │
    │                                                             │
    │  3. EXTRACTION LLM                                         │
    │     └─▶ Extract values using strict prompts                │
    │     └─▶ Return "NOT_FOUND" for missing values              │
    │                                                             │
    │  4. HALLUCINATION DETECTION (4 Layers)                     │
    │     └─▶ L1: Faithfulness scoring (semantic similarity)     │
    │     └─▶ L2: Contradiction detection (verifier LLM)         │
    │     └─▶ L3: Self-consistency (N runs, variance check)      │
    │     └─▶ L4: Clinical rule validation                       │
    │                                                             │
    │  5. SAFETY DECISION                                        │
    │     └─▶ Compute confidence score                           │
    │     └─▶ Apply safety guardrails                            │
    │     └─▶ Decide: ACCEPT / FLAG / ABSTAIN / REJECT          │
    │                                                             │
    │  6. OUTPUT                                                  │
    │     └─▶ Extracted values with confidence                   │
    │     └─▶ Supporting evidence                                │
    │     └─▶ Audit trail                                        │
    │                                                             │
    └─────────────────────────────────────────────────────────────┘
    """)

    print("\nTo run the full pipeline with LLM calls:")
    print("-" * 60)
    print("""
    1. Set your OpenAI API key:
       export OPENAI_API_KEY="sk-your-key-here"

    2. Use the pipeline:

       from src.pipeline import HealthcareExtractionPipeline
       from src.core.models import ClinicalDocument, ExtractionSchema

       pipeline = HealthcareExtractionPipeline()
       result = pipeline.extract(document, schema, verbose=True)

       for decision in result.decisions:
           print(f"{decision.field_name}: {decision.extracted_value}")
           print(f"  Decision: {decision.decision.value}")
           print(f"  Confidence: {decision.confidence.final_score:.2%}")
    """)


def main():
    """Run all demos."""
    print("\n" + "=" * 60)
    print("  HEALTHCARE AI SAFETY GUARDRAILS - DEMONSTRATION")
    print("=" * 60)

    # Run demos
    demo_document_processing()
    demo_clinical_rule_validation()
    demo_safety_guardrails()
    demo_extraction_schema()
    demo_golden_dataset()
    demo_full_pipeline_simulation()

    print("\n" + "=" * 60)
    print("  DEMO COMPLETE")
    print("=" * 60)
    print("\nThis framework provides:")
    print("  ✓ Multi-layer hallucination detection")
    print("  ✓ Healthcare-specific safety guardrails")
    print("  ✓ Confidence scoring with abstention logic")
    print("  ✓ Comprehensive audit logging")
    print("  ✓ HIPAA-compliant design")
    print("\nFor full LLM-powered extraction, set OPENAI_API_KEY and run the pipeline.")
    print("=" * 60 + "\n")


if __name__ == "__main__":
    main()
