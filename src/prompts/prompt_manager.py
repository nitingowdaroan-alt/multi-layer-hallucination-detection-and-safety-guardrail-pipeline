"""
Prompt Manager

Manages prompt templates and provides a unified interface for
generating prompts throughout the system.
"""

from typing import Optional, List, Dict, Any
from dataclasses import dataclass

from ..core.models import ExtractionField, FieldType
from .extraction_prompts import ExtractionPromptTemplate
from .verifier_prompts import VerifierPromptTemplate


@dataclass
class PromptConfig:
    """Configuration for prompt generation."""
    include_examples: bool = False
    max_context_length: int = 4000
    include_field_specific_rules: bool = True
    strict_mode: bool = True  # Enforce all safety rules


class PromptManager:
    """
    Centralized prompt management for the healthcare AI system.

    Provides:
    - Consistent prompt generation
    - Template versioning
    - Prompt customization
    - Context length management
    """

    def __init__(self, config: Optional[PromptConfig] = None):
        self.config = config or PromptConfig()
        self.extraction_template = ExtractionPromptTemplate()
        self.verifier_template = VerifierPromptTemplate()

    def get_extraction_system_prompt(self) -> str:
        """Get the system prompt for extraction."""
        return self.extraction_template.get_system_prompt()

    def get_extraction_user_prompt(
        self,
        field: ExtractionField,
        context: str,
        additional_instructions: Optional[str] = None
    ) -> str:
        """
        Get user prompt for field extraction.

        Args:
            field: Field to extract
            context: Retrieved context
            additional_instructions: Optional extra instructions

        Returns:
            Formatted user prompt
        """
        # Truncate context if needed
        if len(context) > self.config.max_context_length:
            context = self._truncate_context(context)

        return self.extraction_template.get_user_prompt(
            field=field,
            context=context,
            additional_instructions=additional_instructions
        )

    def get_verifier_system_prompt(self) -> str:
        """Get the system prompt for verification."""
        return self.verifier_template.get_system_prompt()

    def get_verifier_user_prompt(
        self,
        field_name: str,
        field_type: FieldType,
        extracted_value: str,
        evidence: str
    ) -> str:
        """
        Get user prompt for verification.

        Args:
            field_name: Name of the field
            field_type: Type of the field
            extracted_value: Value to verify
            evidence: Evidence text

        Returns:
            Formatted verification prompt
        """
        if len(evidence) > self.config.max_context_length:
            evidence = self._truncate_context(evidence)

        return self.verifier_template.get_user_prompt(
            field_name=field_name,
            field_type=field_type,
            extracted_value=extracted_value,
            evidence=evidence
        )

    def get_consistency_prompt(
        self,
        field: ExtractionField,
        context: str
    ) -> str:
        """
        Get prompt for consistency checking runs.

        Uses a simplified extraction prompt for faster runs.
        """
        return f"""Extract the following field from the clinical text.

FIELD: {field.name}
TYPE: {field.field_type.value}
DESCRIPTION: {field.description}

CLINICAL TEXT:
{context[:self.config.max_context_length]}

RULES:
- Extract ONLY what is explicitly stated
- If not found, respond with "NOT_FOUND"
- Do NOT infer or guess

Respond with JSON: {{"value": "<extracted value or NOT_FOUND>"}}"""

    def _truncate_context(self, context: str) -> str:
        """Truncate context while preserving structure."""
        if len(context) <= self.config.max_context_length:
            return context

        # Try to truncate at a natural break point
        truncated = context[:self.config.max_context_length]

        # Find last complete sentence or paragraph
        last_period = truncated.rfind('.')
        last_newline = truncated.rfind('\n')
        break_point = max(last_period, last_newline)

        if break_point > self.config.max_context_length * 0.8:
            truncated = truncated[:break_point + 1]

        return truncated + "\n\n[Context truncated...]"

    def format_context_with_citations(
        self,
        chunks: List[Dict[str, Any]]
    ) -> str:
        """
        Format retrieved chunks with source citations.

        Args:
            chunks: List of chunks with content and metadata

        Returns:
            Formatted context string with citations
        """
        formatted_parts = []

        for i, chunk in enumerate(chunks, 1):
            section = chunk.get("metadata", {}).get("section_name", "Unknown")
            similarity = chunk.get("similarity_score", 0.0)
            content = chunk.get("content", "")

            formatted_parts.append(
                f"[Source {i}] (Section: {section}, Relevance: {similarity:.2f})\n"
                f"{content}"
            )

        return "\n\n---\n\n".join(formatted_parts)

    def get_prompt_templates(self) -> Dict[str, str]:
        """Get all prompt templates for documentation/debugging."""
        return {
            "extraction_system": self.get_extraction_system_prompt(),
            "verifier_system": self.get_verifier_system_prompt(),
        }

    def validate_prompt_length(
        self,
        prompt: str,
        max_tokens: int = 8000
    ) -> tuple:
        """
        Validate that prompt is within token limits.

        Returns:
            Tuple of (is_valid, estimated_tokens)
        """
        # Rough estimate: 4 characters per token
        estimated_tokens = len(prompt) // 4
        is_valid = estimated_tokens < max_tokens
        return is_valid, estimated_tokens
