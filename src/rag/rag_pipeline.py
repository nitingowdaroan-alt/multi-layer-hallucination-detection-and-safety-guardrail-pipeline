"""
RAG Pipeline for Clinical Document Extraction

Orchestrates the retrieval-augmented generation process for
extracting structured information from clinical documents.
"""

from typing import List, Optional, Dict, Any, Tuple
from dataclasses import dataclass, field
import json
import time

from ..core.models import (
    ClinicalDocument,
    ExtractionSchema,
    ExtractionField,
    RetrievedChunk,
    ExtractedField,
    FieldType,
)
from ..core.config import SystemConfig, RAGConfig
from .document_processor import ClinicalDocumentProcessor
from .embeddings import EmbeddingService
from .retriever import ClinicalRetriever, RetrievalQuery


@dataclass
class RAGExtractionContext:
    """Context for RAG-based extraction."""
    document: ClinicalDocument
    field: ExtractionField
    retrieved_chunks: List[RetrievedChunk]
    formatted_context: str
    retrieval_time_ms: float


@dataclass
class RAGExtractionResult:
    """Result of RAG extraction for a field."""
    field_name: str
    extracted_value: Optional[str]
    raw_response: str
    supporting_chunks: List[RetrievedChunk]
    context_used: str
    extraction_time_ms: float
    model_used: str


class RAGPipeline:
    """
    RAG Pipeline for Clinical Document Extraction.

    This pipeline:
    1. Processes clinical documents into chunks
    2. Indexes chunks with embeddings
    3. Retrieves relevant context for each extraction field
    4. Generates structured extractions using LLM
    5. Returns extractions with supporting evidence

    CRITICAL: The LLM must NOT use external medical knowledge.
    All extractions must be grounded in retrieved context.
    """

    def __init__(
        self,
        config: Optional[SystemConfig] = None,
        llm_client: Optional[Any] = None
    ):
        self.config = config or SystemConfig()
        self._llm_client = llm_client

        # Initialize components
        self.document_processor = ClinicalDocumentProcessor(self.config.rag)
        self.embedding_service = EmbeddingService(self.config.model)
        self.retriever = ClinicalRetriever(
            self.embedding_service,
            self.config.rag
        )

        # Document cache
        self._processed_documents: Dict[str, List[RetrievedChunk]] = {}

    @property
    def llm_client(self):
        """Lazy initialization of LLM client."""
        if self._llm_client is None:
            try:
                from openai import OpenAI
                self._llm_client = OpenAI()
            except ImportError:
                raise ImportError("OpenAI package required. Install with: pip install openai")
        return self._llm_client

    def process_document(self, document: ClinicalDocument) -> List[RetrievedChunk]:
        """
        Process and index a clinical document.

        Args:
            document: Clinical document to process

        Returns:
            List of processed chunks
        """
        # Check cache
        if document.document_id in self._processed_documents:
            return self._processed_documents[document.document_id]

        # Process into chunks
        chunks = self.document_processor.process_document(document)

        # Index for retrieval
        self.retriever.index_chunks(document.document_id, chunks)

        # Cache
        self._processed_documents[document.document_id] = chunks

        return chunks

    def extract_fields(
        self,
        document: ClinicalDocument,
        schema: ExtractionSchema,
        temperature: Optional[float] = None
    ) -> List[RAGExtractionResult]:
        """
        Extract all fields from a document using RAG.

        Args:
            document: Clinical document
            schema: Extraction schema defining fields
            temperature: LLM temperature override

        Returns:
            List of extraction results
        """
        # Ensure document is processed
        self.process_document(document)

        results = []
        for extraction_field in schema.fields:
            result = self.extract_single_field(
                document,
                extraction_field,
                temperature
            )
            results.append(result)

        return results

    def extract_single_field(
        self,
        document: ClinicalDocument,
        extraction_field: ExtractionField,
        temperature: Optional[float] = None
    ) -> RAGExtractionResult:
        """
        Extract a single field using RAG.

        Args:
            document: Clinical document
            field: Field to extract
            temperature: LLM temperature override

        Returns:
            RAGExtractionResult with extracted value and evidence
        """
        start_time = time.time()

        # Ensure document is processed
        if document.document_id not in self._processed_documents:
            self.process_document(document)

        # Retrieve relevant chunks
        retrieval_result = self.retriever.retrieve_for_field(
            document.document_id,
            extraction_field
        )

        # Build extraction context
        context = self._build_extraction_context(
            document,
            extraction_field,
            retrieval_result.chunks
        )

        # Generate extraction
        raw_response = self._call_extraction_llm(
            context,
            extraction_field,
            temperature or self.config.model.extraction_temperature
        )

        # Parse response
        extracted_value = self._parse_extraction_response(
            raw_response,
            extraction_field
        )

        extraction_time = (time.time() - start_time) * 1000

        return RAGExtractionResult(
            field_name=extraction_field.name,
            extracted_value=extracted_value,
            raw_response=raw_response,
            supporting_chunks=retrieval_result.chunks,
            context_used=context.formatted_context,
            extraction_time_ms=extraction_time,
            model_used=self.config.model.extraction_model
        )

    def _build_extraction_context(
        self,
        document: ClinicalDocument,
        field: ExtractionField,
        chunks: List[RetrievedChunk]
    ) -> RAGExtractionContext:
        """Build formatted context for extraction."""
        # Format chunks with citations
        formatted_parts = []
        for i, chunk in enumerate(chunks, 1):
            section = chunk.metadata.get("section_name", "Unknown")
            similarity = chunk.similarity_score
            formatted_parts.append(
                f"[Source {i}] (Section: {section}, Relevance: {similarity:.2f})\n"
                f"{chunk.content}"
            )

        formatted_context = "\n\n---\n\n".join(formatted_parts)

        return RAGExtractionContext(
            document=document,
            field=field,
            retrieved_chunks=chunks,
            formatted_context=formatted_context,
            retrieval_time_ms=0.0
        )

    def _call_extraction_llm(
        self,
        context: RAGExtractionContext,
        field: ExtractionField,
        temperature: float
    ) -> str:
        """Call LLM for extraction with strict grounding instructions."""
        system_prompt = self._get_extraction_system_prompt()

        user_prompt = self._get_extraction_user_prompt(context, field)

        response = self.llm_client.chat.completions.create(
            model=self.config.model.extraction_model,
            messages=[
                {"role": "system", "content": system_prompt},
                {"role": "user", "content": user_prompt}
            ],
            temperature=temperature,
            max_tokens=self.config.model.max_tokens
        )

        return response.choices[0].message.content

    def _get_extraction_system_prompt(self) -> str:
        """Get system prompt for extraction LLM."""
        return """You are a clinical data extraction system. Your ONLY task is to extract
specific information from the provided clinical text.

CRITICAL SAFETY RULES - YOU MUST FOLLOW THESE:
1. ONLY extract information that is EXPLICITLY stated in the provided context
2. NEVER use external medical knowledge or inference
3. NEVER guess, assume, or fabricate information
4. NEVER normalize medical abbreviations unless they are defined in the text
5. NEVER infer diagnoses - only extract explicitly stated diagnoses
6. If information is not found, respond with "NOT_FOUND"
7. If information is ambiguous, respond with "AMBIGUOUS: [reason]"
8. If information is partially available, respond with "PARTIAL: [what was found]"

OUTPUT FORMAT:
- For single values: Provide the extracted value exactly as it appears in the text
- For multiple values: Provide a JSON array of values
- Always cite the source number where you found the information

REMEMBER: Patient safety depends on your accuracy. When in doubt, abstain."""

    def _get_extraction_user_prompt(
        self,
        context: RAGExtractionContext,
        field: ExtractionField
    ) -> str:
        """Get user prompt for extraction."""
        return f"""Extract the following field from the clinical text:

FIELD TO EXTRACT: {field.name}
FIELD TYPE: {field.field_type.value}
DESCRIPTION: {field.description}
{f"EXPECTED FORMAT: {field.expected_format}" if field.expected_format else ""}

CLINICAL TEXT CONTEXT:
{context.formatted_context}

INSTRUCTIONS:
1. Search the provided context for information about "{field.name}"
2. Extract ONLY what is explicitly stated
3. Include the source number(s) where you found the information
4. If not found, respond with exactly: NOT_FOUND

Respond in this JSON format:
{{
    "extracted_value": "<value or NOT_FOUND or AMBIGUOUS: reason or PARTIAL: value>",
    "source_citations": [<source numbers used>],
    "confidence_note": "<any relevant notes about the extraction>"
}}"""

    def _parse_extraction_response(
        self,
        response: str,
        field: ExtractionField
    ) -> Optional[str]:
        """Parse extraction response from LLM."""
        try:
            # Try to parse as JSON
            response_clean = response.strip()
            if response_clean.startswith("```json"):
                response_clean = response_clean[7:]
            if response_clean.startswith("```"):
                response_clean = response_clean[3:]
            if response_clean.endswith("```"):
                response_clean = response_clean[:-3]

            data = json.loads(response_clean.strip())
            value = data.get("extracted_value", "NOT_FOUND")

            if value in ["NOT_FOUND", None]:
                return None
            if value.startswith("AMBIGUOUS:") or value.startswith("PARTIAL:"):
                return value

            return str(value)

        except json.JSONDecodeError:
            # Fallback: try to extract value directly
            if "NOT_FOUND" in response:
                return None
            return response.strip()

    def get_retrieval_stats(self) -> Dict[str, Any]:
        """Get retrieval statistics."""
        return {
            "processed_documents": len(self._processed_documents),
            "retriever_stats": self.retriever.get_index_stats(),
            "embedding_cache_stats": self.embedding_service.get_cache_stats(),
        }

    def clear_cache(self, document_id: Optional[str] = None):
        """Clear processed document cache."""
        if document_id:
            self._processed_documents.pop(document_id, None)
            self.retriever.clear_index(document_id)
        else:
            self._processed_documents.clear()
            self.retriever.clear_index()
