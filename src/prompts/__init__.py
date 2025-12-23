"""
Prompt Templates for Healthcare AI Safety System

Contains all prompt templates used by the system:
- Extraction LLM prompts
- Verifier LLM prompts
- Consistency check prompts
"""

from .extraction_prompts import ExtractionPromptTemplate
from .verifier_prompts import VerifierPromptTemplate
from .prompt_manager import PromptManager

__all__ = [
    "ExtractionPromptTemplate",
    "VerifierPromptTemplate",
    "PromptManager",
]
