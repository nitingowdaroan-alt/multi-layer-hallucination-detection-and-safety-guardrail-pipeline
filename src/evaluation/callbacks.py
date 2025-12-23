"""
Evaluation Callbacks

Provides callbacks for tracing and monitoring during extraction,
compatible with LangChain callback system.
"""

from typing import Any, Dict, List, Optional, Union
from dataclasses import dataclass, field
from datetime import datetime
from abc import ABC, abstractmethod
import json
import uuid


@dataclass
class CallbackEvent:
    """Event captured by callback."""
    event_id: str
    event_type: str
    timestamp: str
    data: Dict[str, Any]
    duration_ms: Optional[float] = None


class EvaluationCallback(ABC):
    """Base class for evaluation callbacks."""

    @abstractmethod
    def on_extraction_start(self, document_id: str, schema_id: str):
        """Called when extraction starts."""
        pass

    @abstractmethod
    def on_extraction_end(self, document_id: str, results: Dict[str, Any]):
        """Called when extraction ends."""
        pass

    @abstractmethod
    def on_field_extracted(self, field_name: str, value: Any, confidence: float):
        """Called when a field is extracted."""
        pass

    @abstractmethod
    def on_hallucination_detected(self, field_name: str, details: Dict[str, Any]):
        """Called when hallucination is detected."""
        pass

    @abstractmethod
    def on_abstention(self, field_name: str, reason: str):
        """Called when system abstains."""
        pass

    @abstractmethod
    def on_safety_violation(self, field_name: str, violation: Dict[str, Any]):
        """Called when safety violation occurs."""
        pass


class TracingCallback(EvaluationCallback):
    """Callback that traces all events for debugging and analysis."""

    def __init__(self):
        self.events: List[CallbackEvent] = []
        self.current_extraction_id: Optional[str] = None
        self._start_times: Dict[str, datetime] = {}

    def on_extraction_start(self, document_id: str, schema_id: str):
        self.current_extraction_id = str(uuid.uuid4())[:8]
        self._start_times[self.current_extraction_id] = datetime.utcnow()

        self._add_event("extraction_start", {
            "extraction_id": self.current_extraction_id,
            "document_id": document_id,
            "schema_id": schema_id,
        })

    def on_extraction_end(self, document_id: str, results: Dict[str, Any]):
        duration = None
        if self.current_extraction_id in self._start_times:
            start = self._start_times[self.current_extraction_id]
            duration = (datetime.utcnow() - start).total_seconds() * 1000

        self._add_event("extraction_end", {
            "extraction_id": self.current_extraction_id,
            "document_id": document_id,
            "total_fields": results.get("total_fields", 0),
            "accepted_fields": results.get("accepted_fields", 0),
            "abstained_fields": results.get("abstained_fields", 0),
        }, duration)

        self.current_extraction_id = None

    def on_field_extracted(self, field_name: str, value: Any, confidence: float):
        self._add_event("field_extracted", {
            "extraction_id": self.current_extraction_id,
            "field_name": field_name,
            "value": str(value)[:100] if value else None,
            "confidence": confidence,
        })

    def on_hallucination_detected(self, field_name: str, details: Dict[str, Any]):
        self._add_event("hallucination_detected", {
            "extraction_id": self.current_extraction_id,
            "field_name": field_name,
            "hallucination_type": details.get("type"),
            "composite_score": details.get("composite_score"),
            "faithfulness_score": details.get("faithfulness_score"),
        })

    def on_abstention(self, field_name: str, reason: str):
        self._add_event("abstention", {
            "extraction_id": self.current_extraction_id,
            "field_name": field_name,
            "reason": reason,
        })

    def on_safety_violation(self, field_name: str, violation: Dict[str, Any]):
        self._add_event("safety_violation", {
            "extraction_id": self.current_extraction_id,
            "field_name": field_name,
            "violation_type": violation.get("type"),
            "severity": violation.get("severity"),
            "description": violation.get("description"),
        })

    def _add_event(
        self,
        event_type: str,
        data: Dict[str, Any],
        duration_ms: Optional[float] = None
    ):
        event = CallbackEvent(
            event_id=str(uuid.uuid4())[:8],
            event_type=event_type,
            timestamp=datetime.utcnow().isoformat(),
            data=data,
            duration_ms=duration_ms,
        )
        self.events.append(event)

    def get_events(self) -> List[Dict[str, Any]]:
        """Get all events as dictionaries."""
        return [
            {
                "event_id": e.event_id,
                "event_type": e.event_type,
                "timestamp": e.timestamp,
                "data": e.data,
                "duration_ms": e.duration_ms,
            }
            for e in self.events
        ]

    def get_summary(self) -> Dict[str, Any]:
        """Get summary of traced events."""
        event_counts = {}
        for event in self.events:
            event_counts[event.event_type] = event_counts.get(event.event_type, 0) + 1

        return {
            "total_events": len(self.events),
            "event_counts": event_counts,
            "hallucinations_detected": event_counts.get("hallucination_detected", 0),
            "abstentions": event_counts.get("abstention", 0),
            "safety_violations": event_counts.get("safety_violation", 0),
        }

    def clear(self):
        """Clear all events."""
        self.events.clear()
        self._start_times.clear()


class MetricsCallback(EvaluationCallback):
    """Callback that collects metrics during extraction."""

    def __init__(self):
        self.metrics = {
            "extractions": 0,
            "fields_extracted": 0,
            "hallucinations": 0,
            "abstentions": 0,
            "safety_violations": 0,
            "confidence_sum": 0.0,
            "confidence_count": 0,
        }

    def on_extraction_start(self, document_id: str, schema_id: str):
        self.metrics["extractions"] += 1

    def on_extraction_end(self, document_id: str, results: Dict[str, Any]):
        pass

    def on_field_extracted(self, field_name: str, value: Any, confidence: float):
        self.metrics["fields_extracted"] += 1
        self.metrics["confidence_sum"] += confidence
        self.metrics["confidence_count"] += 1

    def on_hallucination_detected(self, field_name: str, details: Dict[str, Any]):
        self.metrics["hallucinations"] += 1

    def on_abstention(self, field_name: str, reason: str):
        self.metrics["abstentions"] += 1

    def on_safety_violation(self, field_name: str, violation: Dict[str, Any]):
        self.metrics["safety_violations"] += 1

    def get_metrics(self) -> Dict[str, Any]:
        """Get current metrics."""
        avg_confidence = 0.0
        if self.metrics["confidence_count"] > 0:
            avg_confidence = self.metrics["confidence_sum"] / self.metrics["confidence_count"]

        return {
            **self.metrics,
            "average_confidence": avg_confidence,
            "hallucination_rate": (
                self.metrics["hallucinations"] / self.metrics["fields_extracted"]
                if self.metrics["fields_extracted"] > 0 else 0.0
            ),
            "abstention_rate": (
                self.metrics["abstentions"] / self.metrics["fields_extracted"]
                if self.metrics["fields_extracted"] > 0 else 0.0
            ),
        }

    def reset(self):
        """Reset metrics."""
        for key in self.metrics:
            if isinstance(self.metrics[key], float):
                self.metrics[key] = 0.0
            else:
                self.metrics[key] = 0


class LangChainCallback:
    """
    LangChain-compatible callback handler for integration with
    LangChain tracing and evaluation tools.
    """

    def __init__(self, tracer: Optional[TracingCallback] = None):
        self.tracer = tracer or TracingCallback()
        self.run_id: Optional[str] = None

    def on_llm_start(
        self,
        serialized: Dict[str, Any],
        prompts: List[str],
        **kwargs
    ):
        """Called when LLM starts."""
        self.run_id = kwargs.get("run_id", str(uuid.uuid4())[:8])
        self.tracer._add_event("llm_start", {
            "run_id": self.run_id,
            "model": serialized.get("name", "unknown"),
            "prompt_length": len(prompts[0]) if prompts else 0,
        })

    def on_llm_end(self, response: Any, **kwargs):
        """Called when LLM ends."""
        self.tracer._add_event("llm_end", {
            "run_id": self.run_id,
            "response_length": len(str(response)),
        })

    def on_llm_error(self, error: Exception, **kwargs):
        """Called when LLM errors."""
        self.tracer._add_event("llm_error", {
            "run_id": self.run_id,
            "error": str(error),
        })

    def on_chain_start(
        self,
        serialized: Dict[str, Any],
        inputs: Dict[str, Any],
        **kwargs
    ):
        """Called when chain starts."""
        self.tracer._add_event("chain_start", {
            "chain_name": serialized.get("name", "unknown"),
        })

    def on_chain_end(self, outputs: Dict[str, Any], **kwargs):
        """Called when chain ends."""
        self.tracer._add_event("chain_end", {
            "output_keys": list(outputs.keys()) if outputs else [],
        })

    def on_retriever_start(self, query: str, **kwargs):
        """Called when retriever starts."""
        self.tracer._add_event("retriever_start", {
            "query_length": len(query),
        })

    def on_retriever_end(self, documents: List[Any], **kwargs):
        """Called when retriever ends."""
        self.tracer._add_event("retriever_end", {
            "num_documents": len(documents),
        })

    def get_trace(self) -> List[Dict[str, Any]]:
        """Get the full trace."""
        return self.tracer.get_events()


class CompositeCallback(EvaluationCallback):
    """Combines multiple callbacks."""

    def __init__(self, callbacks: List[EvaluationCallback]):
        self.callbacks = callbacks

    def on_extraction_start(self, document_id: str, schema_id: str):
        for cb in self.callbacks:
            cb.on_extraction_start(document_id, schema_id)

    def on_extraction_end(self, document_id: str, results: Dict[str, Any]):
        for cb in self.callbacks:
            cb.on_extraction_end(document_id, results)

    def on_field_extracted(self, field_name: str, value: Any, confidence: float):
        for cb in self.callbacks:
            cb.on_field_extracted(field_name, value, confidence)

    def on_hallucination_detected(self, field_name: str, details: Dict[str, Any]):
        for cb in self.callbacks:
            cb.on_hallucination_detected(field_name, details)

    def on_abstention(self, field_name: str, reason: str):
        for cb in self.callbacks:
            cb.on_abstention(field_name, reason)

    def on_safety_violation(self, field_name: str, violation: Dict[str, Any]):
        for cb in self.callbacks:
            cb.on_safety_violation(field_name, violation)
