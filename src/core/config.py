"""
Configuration Management for Healthcare AI Safety System

Provides centralized configuration for all system components with
support for environment-specific overrides and validation.
"""

import os
from dataclasses import dataclass, field
from typing import Any, Dict, List, Optional
import yaml
import hashlib
import json


@dataclass
class ModelConfig:
    """Configuration for LLM models."""
    extraction_model: str = "gpt-4"
    verifier_model: str = "gpt-4"
    embedding_model: str = "text-embedding-3-small"

    # Model parameters
    extraction_temperature: float = 0.0  # Deterministic for extraction
    verifier_temperature: float = 0.0
    consistency_temperature: float = 0.3  # Slight variation for consistency checks

    max_tokens: int = 4096
    timeout_seconds: int = 60


@dataclass
class RAGConfig:
    """Configuration for Retrieval-Augmented Generation."""
    chunk_size: int = 512
    chunk_overlap: int = 50
    top_k_chunks: int = 5

    # Embedding settings
    embedding_dimension: int = 1536

    # Similarity thresholds
    min_similarity_threshold: float = 0.3
    high_similarity_threshold: float = 0.7


@dataclass
class HallucinationDetectionConfig:
    """Configuration for hallucination detection layers."""

    # Layer 1: Faithfulness
    faithfulness_threshold: float = 0.6
    min_evidence_chunks: int = 1

    # Layer 2: Contradiction
    contradiction_confidence_threshold: float = 0.8

    # Layer 3: Self-Consistency
    consistency_num_runs: int = 5
    consistency_agreement_threshold: float = 0.8
    consistency_variance_threshold: float = 0.2

    # Layer 4: Rule Validation
    enable_rule_validation: bool = True
    strict_rule_enforcement: bool = True


@dataclass
class ConfidenceConfig:
    """Configuration for confidence scoring and abstention."""

    # Thresholds
    accept_threshold: float = 0.85
    review_threshold: float = 0.60

    # Component weights
    weights: Dict[str, float] = field(default_factory=lambda: {
        "retrieval_faithfulness": 0.35,
        "verifier_agreement": 0.25,
        "self_consistency": 0.20,
        "rule_validation": 0.20,
    })

    # Critical field multiplier (stricter for critical fields)
    critical_field_threshold_boost: float = 0.05


@dataclass
class SafetyGuardrailConfig:
    """Configuration for safety guardrails."""

    # Hard guardrails
    never_infer_diagnoses: bool = True
    never_hallucinate_missing: bool = True
    never_normalize_abbreviations: bool = True
    never_override_clinician_text: bool = True

    # Medication safety
    medication_dose_validation: bool = True
    max_safe_dose_multiplier: float = 10.0  # Flag if > 10x typical dose

    # Lab value validation
    lab_value_range_validation: bool = True

    # Date validation
    future_date_allowed: bool = False
    max_past_years: int = 100


@dataclass
class AuditConfig:
    """Configuration for audit logging."""
    enable_audit_logging: bool = True
    log_level: str = "INFO"
    log_format: str = "json"

    # Storage
    log_directory: str = "./audit_logs"
    max_log_size_mb: int = 100
    log_retention_days: int = 365

    # What to log
    log_extractions: bool = True
    log_decisions: bool = True
    log_evidence: bool = True
    log_model_responses: bool = True


@dataclass
class EvaluationConfig:
    """Configuration for evaluation framework."""
    enable_online_evaluation: bool = True
    enable_offline_evaluation: bool = True

    # Metrics to track
    track_hallucination_rate: bool = True
    track_unsupported_claim_rate: bool = True
    track_abstention_accuracy: bool = True
    track_critical_field_precision: bool = True

    # Thresholds for alerts
    max_hallucination_rate: float = 0.05
    min_critical_field_precision: float = 0.95


@dataclass
class SystemConfig:
    """Main system configuration."""
    environment: str = "development"
    version: str = "1.0.0"

    # Sub-configurations
    model: ModelConfig = field(default_factory=ModelConfig)
    rag: RAGConfig = field(default_factory=RAGConfig)
    hallucination_detection: HallucinationDetectionConfig = field(
        default_factory=HallucinationDetectionConfig
    )
    confidence: ConfidenceConfig = field(default_factory=ConfidenceConfig)
    safety: SafetyGuardrailConfig = field(default_factory=SafetyGuardrailConfig)
    audit: AuditConfig = field(default_factory=AuditConfig)
    evaluation: EvaluationConfig = field(default_factory=EvaluationConfig)

    # API Keys (should be loaded from environment)
    openai_api_key: Optional[str] = None
    anthropic_api_key: Optional[str] = None

    def __post_init__(self):
        # Load API keys from environment
        self.openai_api_key = os.getenv("OPENAI_API_KEY")
        self.anthropic_api_key = os.getenv("ANTHROPIC_API_KEY")

    def get_config_hash(self) -> str:
        """Generate hash of configuration for audit purposes."""
        config_str = json.dumps(self.to_dict(), sort_keys=True)
        return hashlib.sha256(config_str.encode()).hexdigest()[:16]

    def to_dict(self) -> Dict[str, Any]:
        """Convert configuration to dictionary."""
        return {
            "environment": self.environment,
            "version": self.version,
            "model": {
                "extraction_model": self.model.extraction_model,
                "verifier_model": self.model.verifier_model,
                "embedding_model": self.model.embedding_model,
                "extraction_temperature": self.model.extraction_temperature,
                "verifier_temperature": self.model.verifier_temperature,
                "consistency_temperature": self.model.consistency_temperature,
                "max_tokens": self.model.max_tokens,
            },
            "rag": {
                "chunk_size": self.rag.chunk_size,
                "chunk_overlap": self.rag.chunk_overlap,
                "top_k_chunks": self.rag.top_k_chunks,
                "min_similarity_threshold": self.rag.min_similarity_threshold,
            },
            "hallucination_detection": {
                "faithfulness_threshold": self.hallucination_detection.faithfulness_threshold,
                "consistency_num_runs": self.hallucination_detection.consistency_num_runs,
                "consistency_agreement_threshold": self.hallucination_detection.consistency_agreement_threshold,
            },
            "confidence": {
                "accept_threshold": self.confidence.accept_threshold,
                "review_threshold": self.confidence.review_threshold,
                "weights": self.confidence.weights,
            },
            "safety": {
                "never_infer_diagnoses": self.safety.never_infer_diagnoses,
                "never_hallucinate_missing": self.safety.never_hallucinate_missing,
                "medication_dose_validation": self.safety.medication_dose_validation,
            },
        }

    @classmethod
    def from_yaml(cls, file_path: str) -> "SystemConfig":
        """Load configuration from YAML file."""
        with open(file_path, "r") as f:
            config_dict = yaml.safe_load(f)
        return cls._from_dict(config_dict)

    @classmethod
    def _from_dict(cls, config_dict: Dict[str, Any]) -> "SystemConfig":
        """Create config from dictionary."""
        config = cls()

        if "environment" in config_dict:
            config.environment = config_dict["environment"]

        if "model" in config_dict:
            model_cfg = config_dict["model"]
            config.model = ModelConfig(
                extraction_model=model_cfg.get("extraction_model", "gpt-4"),
                verifier_model=model_cfg.get("verifier_model", "gpt-4"),
                embedding_model=model_cfg.get("embedding_model", "text-embedding-3-small"),
                extraction_temperature=model_cfg.get("extraction_temperature", 0.0),
                verifier_temperature=model_cfg.get("verifier_temperature", 0.0),
                consistency_temperature=model_cfg.get("consistency_temperature", 0.3),
            )

        if "rag" in config_dict:
            rag_cfg = config_dict["rag"]
            config.rag = RAGConfig(
                chunk_size=rag_cfg.get("chunk_size", 512),
                chunk_overlap=rag_cfg.get("chunk_overlap", 50),
                top_k_chunks=rag_cfg.get("top_k_chunks", 5),
                min_similarity_threshold=rag_cfg.get("min_similarity_threshold", 0.3),
            )

        if "hallucination_detection" in config_dict:
            hd_cfg = config_dict["hallucination_detection"]
            config.hallucination_detection = HallucinationDetectionConfig(
                faithfulness_threshold=hd_cfg.get("faithfulness_threshold", 0.6),
                consistency_num_runs=hd_cfg.get("consistency_num_runs", 5),
                consistency_agreement_threshold=hd_cfg.get("consistency_agreement_threshold", 0.8),
            )

        if "confidence" in config_dict:
            conf_cfg = config_dict["confidence"]
            config.confidence = ConfidenceConfig(
                accept_threshold=conf_cfg.get("accept_threshold", 0.85),
                review_threshold=conf_cfg.get("review_threshold", 0.60),
                weights=conf_cfg.get("weights", config.confidence.weights),
            )

        return config


def load_config(config_path: Optional[str] = None) -> SystemConfig:
    """Load system configuration from file or environment."""
    if config_path and os.path.exists(config_path):
        return SystemConfig.from_yaml(config_path)

    # Check for environment variable
    env_config_path = os.getenv("HEALTHCARE_AI_CONFIG")
    if env_config_path and os.path.exists(env_config_path):
        return SystemConfig.from_yaml(env_config_path)

    # Return default configuration
    return SystemConfig()
