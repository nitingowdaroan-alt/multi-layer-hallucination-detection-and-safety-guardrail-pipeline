"""
Audit Logger for Healthcare AI Safety System

Provides comprehensive audit logging for HIPAA compliance and
clinical review requirements.
"""

import json
import os
from typing import List, Optional, Dict, Any
from dataclasses import dataclass, asdict
from datetime import datetime
from pathlib import Path
import uuid
import hashlib
from abc import ABC, abstractmethod

from ..core.models import AuditLog, DecisionType


class AuditLogStorage(ABC):
    """Abstract base class for audit log storage."""

    @abstractmethod
    def store(self, log: AuditLog):
        """Store an audit log entry."""
        pass

    @abstractmethod
    def retrieve(self, log_id: str) -> Optional[AuditLog]:
        """Retrieve a specific log entry."""
        pass

    @abstractmethod
    def query(self, filters: Dict[str, Any]) -> List[AuditLog]:
        """Query logs by filters."""
        pass


class FileAuditLogStorage(AuditLogStorage):
    """File-based audit log storage."""

    def __init__(self, log_directory: str):
        self.log_directory = Path(log_directory)
        self.log_directory.mkdir(parents=True, exist_ok=True)

    def store(self, log: AuditLog):
        """Store log to file."""
        date_str = datetime.utcnow().strftime("%Y-%m-%d")
        log_file = self.log_directory / f"audit_{date_str}.jsonl"

        with open(log_file, 'a') as f:
            f.write(json.dumps(log.to_dict()) + "\n")

    def retrieve(self, log_id: str) -> Optional[AuditLog]:
        """Retrieve log by ID (searches recent files)."""
        for log_file in sorted(self.log_directory.glob("audit_*.jsonl"), reverse=True):
            with open(log_file, 'r') as f:
                for line in f:
                    data = json.loads(line)
                    if data.get("log_id") == log_id:
                        return self._dict_to_audit_log(data)
        return None

    def query(self, filters: Dict[str, Any]) -> List[AuditLog]:
        """Query logs by filters."""
        results = []
        for log_file in sorted(self.log_directory.glob("audit_*.jsonl"), reverse=True):
            with open(log_file, 'r') as f:
                for line in f:
                    data = json.loads(line)
                    if self._matches_filters(data, filters):
                        results.append(self._dict_to_audit_log(data))
        return results

    def _matches_filters(self, data: Dict, filters: Dict) -> bool:
        """Check if data matches all filters."""
        for key, value in filters.items():
            if data.get(key) != value:
                return False
        return True

    def _dict_to_audit_log(self, data: Dict) -> AuditLog:
        """Convert dictionary to AuditLog."""
        decision = None
        if data.get("decision"):
            decision = DecisionType(data["decision"])

        return AuditLog(
            log_id=data.get("log_id", ""),
            request_id=data.get("request_id", ""),
            document_id=data.get("document_id", ""),
            event_type=data.get("event_type", ""),
            event_description=data.get("event_description", ""),
            actor_type=data.get("actor_type", ""),
            actor_id=data.get("actor_id"),
            field_name=data.get("field_name"),
            previous_value=data.get("previous_value"),
            new_value=data.get("new_value"),
            decision=decision,
            reasoning=data.get("reasoning", ""),
            evidence=data.get("evidence", []),
            timestamp=datetime.fromisoformat(data.get("timestamp", datetime.utcnow().isoformat())),
            session_id=data.get("session_id"),
            model_version=data.get("model_version"),
            config_hash=data.get("config_hash"),
        )


class AuditLogger:
    """
    Comprehensive audit logger for healthcare AI system.

    Provides:
    - Immutable audit trail
    - Request tracing
    - Decision logging
    - Evidence capture
    - HIPAA-compliant logging

    All logs include:
    - Unique identifiers
    - Timestamps
    - Actor information
    - Evidence and reasoning
    - Configuration state
    """

    def __init__(
        self,
        storage: Optional[AuditLogStorage] = None,
        log_directory: str = "./audit_logs",
        model_version: str = "1.0.0"
    ):
        self.storage = storage or FileAuditLogStorage(log_directory)
        self.model_version = model_version
        self._session_id = str(uuid.uuid4())
        self._config_hash: Optional[str] = None

    def set_config_hash(self, config_hash: str):
        """Set the configuration hash for audit trail."""
        self._config_hash = config_hash

    def log_extraction_start(
        self,
        request_id: str,
        document_id: str,
        schema_id: str
    ) -> AuditLog:
        """Log the start of an extraction request."""
        log = AuditLog(
            log_id=str(uuid.uuid4()),
            request_id=request_id,
            document_id=document_id,
            event_type="extraction_start",
            event_description=f"Started extraction with schema {schema_id}",
            actor_type="system",
            reasoning=f"Schema: {schema_id}",
            session_id=self._session_id,
            model_version=self.model_version,
            config_hash=self._config_hash,
        )
        self.storage.store(log)
        return log

    def log_extraction_complete(
        self,
        request_id: str,
        document_id: str,
        summary: Dict[str, Any]
    ) -> AuditLog:
        """Log completion of an extraction request."""
        log = AuditLog(
            log_id=str(uuid.uuid4()),
            request_id=request_id,
            document_id=document_id,
            event_type="extraction_complete",
            event_description="Extraction completed",
            actor_type="system",
            reasoning=json.dumps(summary),
            session_id=self._session_id,
            model_version=self.model_version,
            config_hash=self._config_hash,
        )
        self.storage.store(log)
        return log

    def log_field_decision(
        self,
        request_id: str,
        document_id: str,
        field_name: str,
        decision: DecisionType,
        extracted_value: Optional[str],
        confidence: float,
        reasoning: str,
        evidence: List[str]
    ) -> AuditLog:
        """Log a field-level decision."""
        log = AuditLog(
            log_id=str(uuid.uuid4()),
            request_id=request_id,
            document_id=document_id,
            event_type="field_decision",
            event_description=f"Decision for field '{field_name}': {decision.value}",
            actor_type="system",
            field_name=field_name,
            new_value=extracted_value,
            decision=decision,
            reasoning=f"Confidence: {confidence:.3f}. {reasoning}",
            evidence=evidence[:5],  # Limit evidence snippets
            session_id=self._session_id,
            model_version=self.model_version,
            config_hash=self._config_hash,
        )
        self.storage.store(log)
        return log

    def log_hallucination_detection(
        self,
        request_id: str,
        document_id: str,
        field_name: str,
        hallucination_type: str,
        detection_details: Dict[str, Any]
    ) -> AuditLog:
        """Log a hallucination detection event."""
        log = AuditLog(
            log_id=str(uuid.uuid4()),
            request_id=request_id,
            document_id=document_id,
            event_type="hallucination_detected",
            event_description=f"Hallucination detected for '{field_name}': {hallucination_type}",
            actor_type="system",
            field_name=field_name,
            reasoning=json.dumps(detection_details),
            session_id=self._session_id,
            model_version=self.model_version,
            config_hash=self._config_hash,
        )
        self.storage.store(log)
        return log

    def log_safety_violation(
        self,
        request_id: str,
        document_id: str,
        field_name: str,
        violation_type: str,
        severity: str,
        description: str
    ) -> AuditLog:
        """Log a safety violation."""
        log = AuditLog(
            log_id=str(uuid.uuid4()),
            request_id=request_id,
            document_id=document_id,
            event_type="safety_violation",
            event_description=f"Safety violation for '{field_name}': {violation_type}",
            actor_type="system",
            field_name=field_name,
            reasoning=f"Severity: {severity}. {description}",
            session_id=self._session_id,
            model_version=self.model_version,
            config_hash=self._config_hash,
        )
        self.storage.store(log)
        return log

    def log_abstention(
        self,
        request_id: str,
        document_id: str,
        field_name: str,
        reason: str,
        evidence_summary: str
    ) -> AuditLog:
        """Log an abstention event."""
        log = AuditLog(
            log_id=str(uuid.uuid4()),
            request_id=request_id,
            document_id=document_id,
            event_type="abstention",
            event_description=f"Abstained from extracting '{field_name}'",
            actor_type="system",
            field_name=field_name,
            decision=DecisionType.ABSTAIN,
            reasoning=reason,
            evidence=[evidence_summary] if evidence_summary else [],
            session_id=self._session_id,
            model_version=self.model_version,
            config_hash=self._config_hash,
        )
        self.storage.store(log)
        return log

    def log_human_review(
        self,
        request_id: str,
        document_id: str,
        field_name: str,
        reviewer_id: str,
        original_value: Optional[str],
        reviewed_value: Optional[str],
        review_notes: str
    ) -> AuditLog:
        """Log a human review action."""
        log = AuditLog(
            log_id=str(uuid.uuid4()),
            request_id=request_id,
            document_id=document_id,
            event_type="human_review",
            event_description=f"Human review for '{field_name}'",
            actor_type="human_reviewer",
            actor_id=reviewer_id,
            field_name=field_name,
            previous_value=original_value,
            new_value=reviewed_value,
            reasoning=review_notes,
            session_id=self._session_id,
            model_version=self.model_version,
            config_hash=self._config_hash,
        )
        self.storage.store(log)
        return log

    def get_request_audit_trail(self, request_id: str) -> List[AuditLog]:
        """Get complete audit trail for a request."""
        return self.storage.query({"request_id": request_id})

    def get_document_audit_trail(self, document_id: str) -> List[AuditLog]:
        """Get complete audit trail for a document."""
        return self.storage.query({"document_id": document_id})

    def export_audit_report(
        self,
        request_id: str,
        output_path: str
    ) -> str:
        """Export audit trail as a report."""
        logs = self.get_request_audit_trail(request_id)

        report = {
            "request_id": request_id,
            "generated_at": datetime.utcnow().isoformat(),
            "session_id": self._session_id,
            "model_version": self.model_version,
            "config_hash": self._config_hash,
            "total_events": len(logs),
            "events": [log.to_dict() for log in logs]
        }

        with open(output_path, 'w') as f:
            json.dump(report, f, indent=2)

        return output_path
