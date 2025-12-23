"""
Golden Dataset Management

Provides structures for clinician-labeled ground truth datasets
used for evaluation.
"""

import json
from typing import List, Dict, Any, Optional
from dataclasses import dataclass, field
from pathlib import Path
from datetime import datetime


@dataclass
class GoldenLabel:
    """Ground truth label for a single field."""
    field_name: str
    field_type: str
    expected_value: Optional[str]  # None means field should not be extracted
    is_present_in_source: bool
    should_abstain: bool
    is_hallucination_if_different: bool
    severity: str  # "critical", "high", "medium", "low"
    notes: str = ""


@dataclass
class GoldenExample:
    """A single example in the golden dataset."""
    example_id: str
    document_text: str
    document_type: str  # EHR, discharge summary, radiology, etc.
    labels: List[GoldenLabel]
    source: str  # Where this example came from
    annotator_id: Optional[str] = None
    annotation_date: Optional[str] = None
    metadata: Dict[str, Any] = field(default_factory=dict)

    def get_field_label(self, field_name: str) -> Optional[GoldenLabel]:
        """Get label for a specific field."""
        for label in self.labels:
            if label.field_name == field_name:
                return label
        return None

    def get_critical_fields(self) -> List[GoldenLabel]:
        """Get critical field labels."""
        return [l for l in self.labels if l.severity == "critical"]


class GoldenDataset:
    """
    Collection of clinician-labeled ground truth examples.

    Used for:
    - Offline evaluation
    - Regression testing
    - Model comparison
    - Threshold tuning
    """

    def __init__(self, name: str, description: str = ""):
        self.name = name
        self.description = description
        self.examples: List[GoldenExample] = []
        self.created_at = datetime.utcnow().isoformat()
        self.version = "1.0"

    def add_example(self, example: GoldenExample):
        """Add an example to the dataset."""
        self.examples.append(example)

    def add_examples(self, examples: List[GoldenExample]):
        """Add multiple examples."""
        self.examples.extend(examples)

    def get_example(self, example_id: str) -> Optional[GoldenExample]:
        """Get example by ID."""
        for example in self.examples:
            if example.example_id == example_id:
                return example
        return None

    def filter_by_document_type(self, doc_type: str) -> List[GoldenExample]:
        """Filter examples by document type."""
        return [e for e in self.examples if e.document_type == doc_type]

    def filter_by_field_type(self, field_type: str) -> List[GoldenExample]:
        """Filter examples that have a specific field type."""
        filtered = []
        for example in self.examples:
            if any(l.field_type == field_type for l in example.labels):
                filtered.append(example)
        return filtered

    def get_statistics(self) -> Dict[str, Any]:
        """Get dataset statistics."""
        total_examples = len(self.examples)
        total_labels = sum(len(e.labels) for e in self.examples)

        # By document type
        doc_types = {}
        for example in self.examples:
            doc_types[example.document_type] = doc_types.get(example.document_type, 0) + 1

        # By field type
        field_types = {}
        for example in self.examples:
            for label in example.labels:
                field_types[label.field_type] = field_types.get(label.field_type, 0) + 1

        # By severity
        severities = {}
        for example in self.examples:
            for label in example.labels:
                severities[label.severity] = severities.get(label.severity, 0) + 1

        # Abstention expectations
        abstain_count = sum(
            1 for e in self.examples
            for l in e.labels if l.should_abstain
        )

        return {
            "name": self.name,
            "total_examples": total_examples,
            "total_labels": total_labels,
            "by_document_type": doc_types,
            "by_field_type": field_types,
            "by_severity": severities,
            "expected_abstentions": abstain_count,
            "abstention_rate": abstain_count / total_labels if total_labels > 0 else 0,
        }

    def save(self, file_path: str):
        """Save dataset to JSON file."""
        data = {
            "name": self.name,
            "description": self.description,
            "version": self.version,
            "created_at": self.created_at,
            "examples": [
                {
                    "example_id": e.example_id,
                    "document_text": e.document_text,
                    "document_type": e.document_type,
                    "source": e.source,
                    "annotator_id": e.annotator_id,
                    "annotation_date": e.annotation_date,
                    "metadata": e.metadata,
                    "labels": [
                        {
                            "field_name": l.field_name,
                            "field_type": l.field_type,
                            "expected_value": l.expected_value,
                            "is_present_in_source": l.is_present_in_source,
                            "should_abstain": l.should_abstain,
                            "is_hallucination_if_different": l.is_hallucination_if_different,
                            "severity": l.severity,
                            "notes": l.notes,
                        }
                        for l in e.labels
                    ]
                }
                for e in self.examples
            ]
        }

        with open(file_path, 'w') as f:
            json.dump(data, f, indent=2)

    @classmethod
    def load(cls, file_path: str) -> "GoldenDataset":
        """Load dataset from JSON file."""
        with open(file_path, 'r') as f:
            data = json.load(f)

        dataset = cls(
            name=data["name"],
            description=data.get("description", "")
        )
        dataset.version = data.get("version", "1.0")
        dataset.created_at = data.get("created_at", "")

        for example_data in data["examples"]:
            labels = [
                GoldenLabel(
                    field_name=l["field_name"],
                    field_type=l["field_type"],
                    expected_value=l["expected_value"],
                    is_present_in_source=l["is_present_in_source"],
                    should_abstain=l["should_abstain"],
                    is_hallucination_if_different=l["is_hallucination_if_different"],
                    severity=l["severity"],
                    notes=l.get("notes", ""),
                )
                for l in example_data["labels"]
            ]

            example = GoldenExample(
                example_id=example_data["example_id"],
                document_text=example_data["document_text"],
                document_type=example_data["document_type"],
                labels=labels,
                source=example_data["source"],
                annotator_id=example_data.get("annotator_id"),
                annotation_date=example_data.get("annotation_date"),
                metadata=example_data.get("metadata", {}),
            )
            dataset.add_example(example)

        return dataset

    def split(
        self,
        train_ratio: float = 0.8,
        seed: int = 42
    ) -> tuple:
        """Split dataset into train and test sets."""
        import random
        random.seed(seed)

        examples = self.examples.copy()
        random.shuffle(examples)

        split_idx = int(len(examples) * train_ratio)

        train_dataset = GoldenDataset(
            f"{self.name}_train",
            f"Training split of {self.name}"
        )
        train_dataset.add_examples(examples[:split_idx])

        test_dataset = GoldenDataset(
            f"{self.name}_test",
            f"Test split of {self.name}"
        )
        test_dataset.add_examples(examples[split_idx:])

        return train_dataset, test_dataset


def create_sample_golden_dataset() -> GoldenDataset:
    """Create a sample golden dataset for testing."""
    dataset = GoldenDataset(
        name="sample_clinical_dataset",
        description="Sample dataset for testing hallucination detection"
    )

    # Example 1: Discharge summary with medications
    example1 = GoldenExample(
        example_id="sample_001",
        document_text="""
DISCHARGE SUMMARY

Patient: John Doe
Date: 01/15/2024

DISCHARGE MEDICATIONS:
1. Metformin 500mg twice daily
2. Lisinopril 10mg once daily
3. Atorvastatin 20mg at bedtime

DISCHARGE DIAGNOSIS:
- Type 2 Diabetes Mellitus
- Hypertension

ALLERGIES: Penicillin (rash)

FOLLOW-UP: PCP in 2 weeks
        """,
        document_type="discharge_summary",
        labels=[
            GoldenLabel(
                field_name="metformin_dose",
                field_type="dosage",
                expected_value="500mg",
                is_present_in_source=True,
                should_abstain=False,
                is_hallucination_if_different=True,
                severity="critical",
            ),
            GoldenLabel(
                field_name="aspirin_dose",
                field_type="dosage",
                expected_value=None,
                is_present_in_source=False,
                should_abstain=True,
                is_hallucination_if_different=True,
                severity="critical",
                notes="Aspirin is not mentioned in the document"
            ),
            GoldenLabel(
                field_name="primary_diagnosis",
                field_type="diagnosis",
                expected_value="Type 2 Diabetes Mellitus",
                is_present_in_source=True,
                should_abstain=False,
                is_hallucination_if_different=True,
                severity="critical",
            ),
            GoldenLabel(
                field_name="allergy",
                field_type="allergy",
                expected_value="Penicillin",
                is_present_in_source=True,
                should_abstain=False,
                is_hallucination_if_different=True,
                severity="critical",
            ),
        ],
        source="synthetic",
        annotator_id="clinical_expert_1",
        annotation_date="2024-01-20",
    )

    # Example 2: Lab results
    example2 = GoldenExample(
        example_id="sample_002",
        document_text="""
LABORATORY RESULTS

Date: 01/16/2024

CHEMISTRY:
- Glucose: 142 mg/dL (H)
- BUN: 18 mg/dL
- Creatinine: 1.1 mg/dL
- Sodium: 139 mEq/L
- Potassium: 4.2 mEq/L

HEMATOLOGY:
- WBC: 7,500 cells/μL
- Hemoglobin: 13.5 g/dL
- Hematocrit: 40%
- Platelets: 250,000 cells/μL

HbA1c: 7.8%
        """,
        document_type="lab_report",
        labels=[
            GoldenLabel(
                field_name="glucose",
                field_type="lab_value",
                expected_value="142 mg/dL",
                is_present_in_source=True,
                should_abstain=False,
                is_hallucination_if_different=True,
                severity="high",
            ),
            GoldenLabel(
                field_name="hba1c",
                field_type="lab_value",
                expected_value="7.8%",
                is_present_in_source=True,
                should_abstain=False,
                is_hallucination_if_different=True,
                severity="high",
            ),
            GoldenLabel(
                field_name="ldl_cholesterol",
                field_type="lab_value",
                expected_value=None,
                is_present_in_source=False,
                should_abstain=True,
                is_hallucination_if_different=True,
                severity="medium",
                notes="LDL cholesterol is not in this lab report"
            ),
        ],
        source="synthetic",
        annotator_id="clinical_expert_1",
        annotation_date="2024-01-20",
    )

    dataset.add_examples([example1, example2])
    return dataset
