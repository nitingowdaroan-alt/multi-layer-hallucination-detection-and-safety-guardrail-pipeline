"""
Core Data Models for Healthcare AI Safety Guardrails System

These models define the data structures used throughout the pipeline for
clinical document processing, extraction, hallucination detection, and audit logging.
"""

from dataclasses import dataclass, field
from datetime import datetime
from enum import Enum
from typing import Any, Dict, List, Optional
import uuid


class DecisionType(Enum):
    """Decision types for safety guardrails."""
    ACCEPT = "accept"  # High confidence, safe to use
    FLAG_FOR_REVIEW = "flag_for_review"  # Low confidence, needs human review
    ABSTAIN = "abstain"  # Insufficient evidence, explicit abstention
    REJECT = "reject"  # Unsafe or contradictory, blocked


class FieldType(Enum):
    """Types of clinical fields for extraction."""
    DIAGNOSIS = "diagnosis"
    MEDICATION = "medication"
    DOSAGE = "dosage"
    LAB_VALUE = "lab_value"
    PROCEDURE = "procedure"
    ALLERGY = "allergy"
    VITAL_SIGN = "vital_sign"
    DATE = "date"
    PROVIDER = "provider"
    FACILITY = "facility"
    FREE_TEXT = "free_text"


class SeverityLevel(Enum):
    """Severity levels for clinical fields."""
    CRITICAL = "critical"  # Medications, allergies, diagnoses
    HIGH = "high"  # Dosages, lab values
    MEDIUM = "medium"  # Procedures, dates
    LOW = "low"  # Provider names, facilities


@dataclass
class ClinicalDocument:
    """Represents a clinical document for processing."""
    document_id: str
    document_type: str  # EHR note, discharge summary, radiology report, etc.
    content: str
    metadata: Dict[str, Any] = field(default_factory=dict)
    created_at: datetime = field(default_factory=datetime.utcnow)
    source_system: Optional[str] = None

    def __post_init__(self):
        if not self.document_id:
            self.document_id = str(uuid.uuid4())


@dataclass
class ExtractionField:
    """Defines a field to be extracted from clinical text."""
    name: str
    field_type: FieldType
    description: str
    required: bool = True
    severity: SeverityLevel = SeverityLevel.HIGH
    validation_rules: List[str] = field(default_factory=list)
    expected_format: Optional[str] = None


@dataclass
class ExtractionSchema:
    """Schema defining what fields to extract from clinical documents."""
    schema_id: str
    name: str
    description: str
    fields: List[ExtractionField]
    version: str = "1.0"

    def get_critical_fields(self) -> List[ExtractionField]:
        """Return fields marked as critical severity."""
        return [f for f in self.fields if f.severity == SeverityLevel.CRITICAL]


@dataclass
class RetrievedChunk:
    """A chunk of text retrieved from the clinical document."""
    chunk_id: str
    content: str
    start_position: int
    end_position: int
    embedding: Optional[List[float]] = None
    similarity_score: float = 0.0
    metadata: Dict[str, Any] = field(default_factory=dict)


@dataclass
class ExtractedField:
    """Represents an extracted field with its value and supporting evidence."""
    field_name: str
    field_type: FieldType
    extracted_value: Optional[str]
    raw_value: Optional[str] = None  # Original text from document
    supporting_chunks: List[RetrievedChunk] = field(default_factory=list)
    extraction_timestamp: datetime = field(default_factory=datetime.utcnow)

    # Hallucination detection scores
    faithfulness_score: float = 0.0
    contradiction_detected: bool = False
    contradiction_reason: Optional[str] = None
    self_consistency_score: float = 0.0
    self_consistency_variance: float = 0.0
    rule_validation_passed: bool = True
    rule_validation_errors: List[str] = field(default_factory=list)

    # Final decision
    confidence_score: float = 0.0
    decision: DecisionType = DecisionType.ABSTAIN
    abstention_reason: Optional[str] = None


@dataclass
class FaithfulnessResult:
    """Result of faithfulness scoring for a field."""
    field_name: str
    faithfulness_score: float
    supporting_chunks: List[RetrievedChunk]
    max_chunk_similarity: float
    avg_chunk_similarity: float
    grounding_evidence: str
    is_grounded: bool


@dataclass
class ContradictionResult:
    """Result of contradiction detection for a field."""
    field_name: str
    contradiction_detected: bool
    contradiction_reason: Optional[str]
    verifier_confidence: float
    evidence_alignment: str  # "aligned", "contradicted", "unsupported"
    explanation: str


@dataclass
class ConsistencyResult:
    """Result of self-consistency checks for a field."""
    field_name: str
    num_runs: int
    unique_values: List[str]
    agreement_rate: float
    variance_score: float
    most_common_value: Optional[str]
    confidence_decay: float  # How much confidence drops across runs
    is_stable: bool


@dataclass
class RuleValidationResult:
    """Result of rule-based clinical validation."""
    field_name: str
    field_type: FieldType
    passed: bool
    rules_applied: List[str]
    violations: List[str]
    warnings: List[str]
    corrected_value: Optional[str] = None  # If rule correction was applied


@dataclass
class HallucinationDetectionResult:
    """Complete hallucination detection result for a field."""
    field_name: str
    faithfulness: FaithfulnessResult
    contradiction: ContradictionResult
    consistency: ConsistencyResult
    rule_validation: RuleValidationResult

    # Aggregated scores
    composite_score: float = 0.0
    is_hallucinated: bool = False
    hallucination_type: Optional[str] = None  # fabricated, altered, unsupported
    detection_timestamp: datetime = field(default_factory=datetime.utcnow)


@dataclass
class ConfidenceScore:
    """Detailed confidence score breakdown."""
    field_name: str

    # Component scores (0-1 scale)
    retrieval_faithfulness: float
    verifier_agreement: float
    self_consistency: float
    rule_validation: float

    # Weights used
    weights: Dict[str, float] = field(default_factory=dict)

    # Final score
    final_score: float = 0.0

    # Thresholds applied
    accept_threshold: float = 0.85
    review_threshold: float = 0.60

    # Decision
    decision: DecisionType = DecisionType.ABSTAIN

    def compute_final_score(self) -> float:
        """Compute weighted final confidence score."""
        default_weights = {
            "retrieval_faithfulness": 0.35,
            "verifier_agreement": 0.25,
            "self_consistency": 0.20,
            "rule_validation": 0.20,
        }
        weights = self.weights or default_weights

        self.final_score = (
            self.retrieval_faithfulness * weights["retrieval_faithfulness"] +
            self.verifier_agreement * weights["verifier_agreement"] +
            self.self_consistency * weights["self_consistency"] +
            self.rule_validation * weights["rule_validation"]
        )
        return self.final_score

    def determine_decision(self) -> DecisionType:
        """Determine decision based on thresholds."""
        if self.rule_validation < 1.0:  # Any rule violation
            self.decision = DecisionType.REJECT
        elif self.final_score >= self.accept_threshold:
            self.decision = DecisionType.ACCEPT
        elif self.final_score >= self.review_threshold:
            self.decision = DecisionType.FLAG_FOR_REVIEW
        else:
            self.decision = DecisionType.ABSTAIN
        return self.decision


@dataclass
class SafetyDecision:
    """Final safety decision for an extracted field."""
    field_name: str
    decision: DecisionType
    confidence: ConfidenceScore
    extracted_value: Optional[str]

    # Supporting evidence
    evidence_snippets: List[str] = field(default_factory=list)

    # Explanation
    explanation: str = ""
    abstention_reason: Optional[str] = None

    # Flags
    requires_human_review: bool = False
    is_critical_field: bool = False

    # Timestamp
    decision_timestamp: datetime = field(default_factory=datetime.utcnow)


@dataclass
class ExtractionResult:
    """Complete extraction result for a clinical document."""
    request_id: str
    document_id: str
    schema_id: str

    # Extracted fields with decisions
    fields: List[SafetyDecision] = field(default_factory=list)

    # Summary statistics
    total_fields: int = 0
    accepted_fields: int = 0
    flagged_fields: int = 0
    abstained_fields: int = 0
    rejected_fields: int = 0

    # Processing metadata
    processing_time_ms: float = 0.0
    model_used: str = ""
    retrieval_chunks_used: int = 0

    # Timestamps
    started_at: datetime = field(default_factory=datetime.utcnow)
    completed_at: Optional[datetime] = None

    def __post_init__(self):
        if not self.request_id:
            self.request_id = str(uuid.uuid4())

    def compute_summary(self):
        """Compute summary statistics."""
        self.total_fields = len(self.fields)
        self.accepted_fields = sum(1 for f in self.fields if f.decision == DecisionType.ACCEPT)
        self.flagged_fields = sum(1 for f in self.fields if f.decision == DecisionType.FLAG_FOR_REVIEW)
        self.abstained_fields = sum(1 for f in self.fields if f.decision == DecisionType.ABSTAIN)
        self.rejected_fields = sum(1 for f in self.fields if f.decision == DecisionType.REJECT)


@dataclass
class AuditLog:
    """Audit log entry for compliance and traceability."""
    log_id: str
    request_id: str
    document_id: str

    # Event details
    event_type: str  # extraction, validation, decision, review
    event_description: str

    # Actor
    actor_type: str  # system, model, human_reviewer
    actor_id: Optional[str] = None

    # Data
    field_name: Optional[str] = None
    previous_value: Optional[str] = None
    new_value: Optional[str] = None
    decision: Optional[DecisionType] = None

    # Reasoning
    reasoning: str = ""
    evidence: List[str] = field(default_factory=list)

    # Metadata
    timestamp: datetime = field(default_factory=datetime.utcnow)
    session_id: Optional[str] = None
    model_version: Optional[str] = None
    config_hash: Optional[str] = None

    def __post_init__(self):
        if not self.log_id:
            self.log_id = str(uuid.uuid4())

    def to_dict(self) -> Dict[str, Any]:
        """Convert to dictionary for JSON serialization."""
        return {
            "log_id": self.log_id,
            "request_id": self.request_id,
            "document_id": self.document_id,
            "event_type": self.event_type,
            "event_description": self.event_description,
            "actor_type": self.actor_type,
            "actor_id": self.actor_id,
            "field_name": self.field_name,
            "previous_value": self.previous_value,
            "new_value": self.new_value,
            "decision": self.decision.value if self.decision else None,
            "reasoning": self.reasoning,
            "evidence": self.evidence,
            "timestamp": self.timestamp.isoformat(),
            "session_id": self.session_id,
            "model_version": self.model_version,
            "config_hash": self.config_hash,
        }
