"""
Main Pipeline Orchestrator

Orchestrates the complete extraction pipeline with hallucination
detection and safety guardrails.
"""

import time
import uuid
from typing import Optional, List, Dict, Any
from dataclasses import dataclass

from .core.models import (
    ClinicalDocument,
    ExtractionSchema,
    ExtractionField,
    ExtractionResult,
    SafetyDecision,
    DecisionType,
    FieldType,
    SeverityLevel,
)
from .core.config import SystemConfig, load_config
from .rag import RAGPipeline
from .hallucination_detection import HallucinationDetector, DetectionInput
from .safety_guardrails import SafetyDecisionEngine, DecisionContext
from .evaluation import TracingCallback, MetricsCallback
from .utils import AuditLogger, ExplainabilityGenerator


@dataclass
class PipelineResult:
    """Complete result from the extraction pipeline."""
    extraction_result: ExtractionResult
    decisions: List[SafetyDecision]
    explanations: Dict[str, str]
    metrics: Dict[str, Any]
    audit_trail: List[str]  # Log IDs


class HealthcareExtractionPipeline:
    """
    Production-grade extraction pipeline for healthcare LLMs.

    This pipeline orchestrates:
    1. RAG-based context retrieval
    2. LLM-based structured extraction
    3. Multi-layer hallucination detection
    4. Safety guardrail enforcement
    5. Confidence scoring and abstention
    6. Audit logging and explainability

    All extractions are traced and auditable for clinical compliance.
    """

    def __init__(
        self,
        config: Optional[SystemConfig] = None,
        llm_client: Optional[Any] = None
    ):
        """
        Initialize the extraction pipeline.

        Args:
            config: System configuration (loads default if None)
            llm_client: Optional LLM client to use
        """
        self.config = config or load_config()

        # Initialize components
        self.rag_pipeline = RAGPipeline(self.config, llm_client)
        self.hallucination_detector = HallucinationDetector(
            self.config,
            self.rag_pipeline.embedding_service,
            llm_client
        )
        self.decision_engine = SafetyDecisionEngine(self.config)
        self.audit_logger = AuditLogger(
            log_directory=self.config.audit.log_directory,
            model_version=self.config.version
        )
        self.explainability = ExplainabilityGenerator()

        # Set config hash for audit
        self.audit_logger.set_config_hash(self.config.get_config_hash())

        # Callbacks
        self.tracer = TracingCallback()
        self.metrics = MetricsCallback()

    def extract(
        self,
        document: ClinicalDocument,
        schema: ExtractionSchema,
        verbose: bool = False
    ) -> PipelineResult:
        """
        Execute the complete extraction pipeline.

        Args:
            document: Clinical document to extract from
            schema: Schema defining fields to extract
            verbose: Whether to print progress

        Returns:
            PipelineResult with all extractions and decisions
        """
        request_id = str(uuid.uuid4())
        start_time = time.time()

        # Start audit trail
        self.audit_logger.log_extraction_start(
            request_id, document.document_id, schema.schema_id
        )
        self.tracer.on_extraction_start(document.document_id, schema.schema_id)
        self.metrics.on_extraction_start(document.document_id, schema.schema_id)

        if verbose:
            print(f"Starting extraction for document: {document.document_id}")
            print(f"Schema: {schema.name} with {len(schema.fields)} fields")

        # Process document through RAG pipeline
        if verbose:
            print("Processing document...")
        chunks = self.rag_pipeline.process_document(document)

        # Extract each field
        decisions = []
        explanations = {}
        audit_trail = []

        for field in schema.fields:
            if verbose:
                print(f"  Extracting: {field.name}")

            decision, explanation = self._extract_field(
                document=document,
                field=field,
                request_id=request_id
            )
            decisions.append(decision)
            explanations[field.name] = explanation

            # Log to audit trail
            log = self.audit_logger.log_field_decision(
                request_id=request_id,
                document_id=document.document_id,
                field_name=field.name,
                decision=decision.decision,
                extracted_value=decision.extracted_value,
                confidence=decision.confidence.final_score,
                reasoning=decision.explanation[:500],
                evidence=decision.evidence_snippets[:3]
            )
            audit_trail.append(log.log_id)

            # Callbacks
            self.tracer.on_field_extracted(
                field.name,
                decision.extracted_value,
                decision.confidence.final_score
            )
            self.metrics.on_field_extracted(
                field.name,
                decision.extracted_value,
                decision.confidence.final_score
            )

            if decision.decision == DecisionType.ABSTAIN:
                self.tracer.on_abstention(field.name, decision.abstention_reason or "")
                self.metrics.on_abstention(field.name, decision.abstention_reason or "")

        # Build extraction result
        extraction_result = ExtractionResult(
            request_id=request_id,
            document_id=document.document_id,
            schema_id=schema.schema_id,
            fields=decisions,
            processing_time_ms=(time.time() - start_time) * 1000,
            model_used=self.config.model.extraction_model,
            retrieval_chunks_used=len(chunks),
        )
        extraction_result.compute_summary()

        # Log completion
        self.audit_logger.log_extraction_complete(
            request_id=request_id,
            document_id=document.document_id,
            summary={
                "total_fields": extraction_result.total_fields,
                "accepted": extraction_result.accepted_fields,
                "flagged": extraction_result.flagged_fields,
                "abstained": extraction_result.abstained_fields,
                "rejected": extraction_result.rejected_fields,
            }
        )

        self.tracer.on_extraction_end(document.document_id, {
            "total_fields": extraction_result.total_fields,
            "accepted_fields": extraction_result.accepted_fields,
            "abstained_fields": extraction_result.abstained_fields,
        })

        if verbose:
            print(f"\nExtraction complete:")
            print(f"  Accepted: {extraction_result.accepted_fields}")
            print(f"  Flagged: {extraction_result.flagged_fields}")
            print(f"  Abstained: {extraction_result.abstained_fields}")
            print(f"  Rejected: {extraction_result.rejected_fields}")
            print(f"  Time: {extraction_result.processing_time_ms:.1f}ms")

        return PipelineResult(
            extraction_result=extraction_result,
            decisions=decisions,
            explanations=explanations,
            metrics=self.metrics.get_metrics(),
            audit_trail=audit_trail
        )

    def _extract_field(
        self,
        document: ClinicalDocument,
        field: ExtractionField,
        request_id: str
    ) -> tuple:
        """Extract a single field with full pipeline."""
        # Step 1: RAG extraction
        rag_result = self.rag_pipeline.extract_single_field(document, field)

        # Step 2: Hallucination detection
        detection_input = DetectionInput(
            field=field,
            extracted_value=rag_result.extracted_value,
            supporting_chunks=rag_result.supporting_chunks,
            extraction_context=rag_result.context_used,
        )
        detection_result = self.hallucination_detector.detect(detection_input)

        # Log if hallucination detected
        if detection_result.is_hallucinated:
            self.audit_logger.log_hallucination_detection(
                request_id=request_id,
                document_id=document.document_id,
                field_name=field.name,
                hallucination_type=detection_result.hallucination_type or "unknown",
                detection_details={
                    "composite_score": detection_result.composite_score,
                    "faithfulness": detection_result.faithfulness.faithfulness_score,
                    "contradiction": detection_result.contradiction.contradiction_detected,
                }
            )
            self.tracer.on_hallucination_detected(field.name, {
                "type": detection_result.hallucination_type,
                "composite_score": detection_result.composite_score,
                "faithfulness_score": detection_result.faithfulness.faithfulness_score,
            })

        # Step 3: Safety decision
        decision_context = DecisionContext(
            field=field,
            extracted_value=rag_result.extracted_value,
            detection_result=detection_result,
            source_text=document.content,
            extraction_reasoning=rag_result.raw_response,
        )

        decision_outcome = self.decision_engine.make_decision(
            decision_context,
            request_id,
            document.document_id
        )

        # Log safety violations
        if not decision_outcome.safety_check.passed:
            self.audit_logger.log_safety_violation(
                request_id=request_id,
                document_id=document.document_id,
                field_name=field.name,
                violation_type=decision_outcome.safety_check.violation_type.value if decision_outcome.safety_check.violation_type else "unknown",
                severity=decision_outcome.safety_check.severity,
                description=decision_outcome.safety_check.violation_description
            )
            self.tracer.on_safety_violation(field.name, {
                "type": str(decision_outcome.safety_check.violation_type),
                "severity": decision_outcome.safety_check.severity,
                "description": decision_outcome.safety_check.violation_description,
            })

        # Step 4: Generate explanation
        explanation_output = self.explainability.explain_decision(
            decision_outcome.decision,
            detection_result
        )

        return decision_outcome.decision, explanation_output.summary

    def get_audit_trail(self, request_id: str) -> List[Dict[str, Any]]:
        """Get the complete audit trail for a request."""
        logs = self.audit_logger.get_request_audit_trail(request_id)
        return [log.to_dict() for log in logs]

    def export_audit_report(self, request_id: str, output_path: str) -> str:
        """Export audit trail as a report."""
        return self.audit_logger.export_audit_report(request_id, output_path)

    def get_metrics(self) -> Dict[str, Any]:
        """Get current pipeline metrics."""
        return self.metrics.get_metrics()

    def get_trace(self) -> List[Dict[str, Any]]:
        """Get the execution trace."""
        return self.tracer.get_events()

    def reset_metrics(self):
        """Reset pipeline metrics."""
        self.metrics.reset()
        self.tracer.clear()


def create_pipeline(
    config_path: Optional[str] = None,
    **kwargs
) -> HealthcareExtractionPipeline:
    """
    Factory function to create a configured pipeline.

    Args:
        config_path: Optional path to configuration file
        **kwargs: Additional configuration overrides

    Returns:
        Configured HealthcareExtractionPipeline
    """
    config = load_config(config_path)
    return HealthcareExtractionPipeline(config, **kwargs)
