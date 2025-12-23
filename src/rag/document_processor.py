"""
Clinical Document Processor

Handles chunking and preprocessing of clinical documents for RAG.
Preserves clinical context and section boundaries.
"""

import re
import uuid
from typing import List, Tuple, Optional
from dataclasses import dataclass, field

from ..core.models import ClinicalDocument, RetrievedChunk
from ..core.config import RAGConfig


@dataclass
class ChunkMetadata:
    """Metadata for a document chunk."""
    section_name: Optional[str] = None
    section_type: Optional[str] = None
    has_medications: bool = False
    has_lab_values: bool = False
    has_diagnoses: bool = False
    has_dates: bool = False
    sentence_count: int = 0


class ClinicalDocumentProcessor:
    """
    Processes clinical documents into chunks suitable for RAG.

    Implements clinical-aware chunking that:
    - Preserves section boundaries (HPI, Assessment, Plan, etc.)
    - Maintains context around critical information
    - Handles clinical formatting patterns
    """

    # Common clinical section headers
    SECTION_PATTERNS = [
        r"(?i)^(HISTORY OF PRESENT ILLNESS|HPI)[:\s]",
        r"(?i)^(CHIEF COMPLAINT|CC)[:\s]",
        r"(?i)^(PAST MEDICAL HISTORY|PMH)[:\s]",
        r"(?i)^(MEDICATIONS|CURRENT MEDICATIONS|MEDS)[:\s]",
        r"(?i)^(ALLERGIES)[:\s]",
        r"(?i)^(FAMILY HISTORY|FH)[:\s]",
        r"(?i)^(SOCIAL HISTORY|SH)[:\s]",
        r"(?i)^(REVIEW OF SYSTEMS|ROS)[:\s]",
        r"(?i)^(PHYSICAL EXAM|PHYSICAL EXAMINATION|PE)[:\s]",
        r"(?i)^(VITAL SIGNS|VITALS)[:\s]",
        r"(?i)^(LABORATORY|LAB|LABS|LABORATORY DATA)[:\s]",
        r"(?i)^(IMAGING|RADIOLOGY)[:\s]",
        r"(?i)^(ASSESSMENT|IMPRESSION)[:\s]",
        r"(?i)^(PLAN|TREATMENT PLAN)[:\s]",
        r"(?i)^(DISCHARGE INSTRUCTIONS)[:\s]",
        r"(?i)^(DISCHARGE MEDICATIONS)[:\s]",
        r"(?i)^(FOLLOW-UP|FOLLOW UP)[:\s]",
    ]

    # Patterns indicating clinical data
    MEDICATION_PATTERN = re.compile(
        r"\b\d+\s*(mg|mcg|g|ml|units?|tabs?|capsules?)\b",
        re.IGNORECASE
    )
    LAB_VALUE_PATTERN = re.compile(
        r"\b\d+\.?\d*\s*(mg/dL|mmol/L|mEq/L|g/dL|%|cells/μL)\b",
        re.IGNORECASE
    )
    DIAGNOSIS_PATTERN = re.compile(
        r"\b(diagnosed with|diagnosis of|assessment:|impression:)\b",
        re.IGNORECASE
    )
    DATE_PATTERN = re.compile(
        r"\b\d{1,2}[/-]\d{1,2}[/-]\d{2,4}\b|\b(January|February|March|April|May|June|July|August|September|October|November|December)\s+\d{1,2},?\s+\d{4}\b",
        re.IGNORECASE
    )

    def __init__(self, config: Optional[RAGConfig] = None):
        self.config = config or RAGConfig()
        self.chunk_size = self.config.chunk_size
        self.chunk_overlap = self.config.chunk_overlap

    def process_document(self, document: ClinicalDocument) -> List[RetrievedChunk]:
        """
        Process a clinical document into chunks.

        Args:
            document: The clinical document to process

        Returns:
            List of document chunks with metadata
        """
        # First, identify sections
        sections = self._identify_sections(document.content)

        # Chunk each section
        chunks = []
        for section_name, section_content, start_pos in sections:
            section_chunks = self._chunk_section(
                content=section_content,
                section_name=section_name,
                base_position=start_pos,
                document=document
            )
            chunks.extend(section_chunks)

        return chunks

    def _identify_sections(self, content: str) -> List[Tuple[str, str, int]]:
        """
        Identify clinical sections in the document.

        Returns:
            List of (section_name, section_content, start_position)
        """
        sections = []
        lines = content.split("\n")

        current_section = "GENERAL"
        current_content = []
        current_start = 0

        position = 0
        for line in lines:
            # Check if this line is a section header
            section_match = None
            for pattern in self.SECTION_PATTERNS:
                if re.match(pattern, line.strip()):
                    # Save current section
                    if current_content:
                        sections.append((
                            current_section,
                            "\n".join(current_content),
                            current_start
                        ))

                    # Start new section
                    current_section = line.strip().rstrip(":").upper()
                    current_content = [line]
                    current_start = position
                    section_match = True
                    break

            if not section_match:
                current_content.append(line)

            position += len(line) + 1  # +1 for newline

        # Don't forget the last section
        if current_content:
            sections.append((
                current_section,
                "\n".join(current_content),
                current_start
            ))

        return sections

    def _chunk_section(
        self,
        content: str,
        section_name: str,
        base_position: int,
        document: ClinicalDocument
    ) -> List[RetrievedChunk]:
        """
        Chunk a section into smaller pieces.

        Uses sentence-aware chunking to avoid breaking mid-sentence.
        """
        if len(content) <= self.chunk_size:
            # Section fits in one chunk
            metadata = self._analyze_chunk(content, section_name)
            return [RetrievedChunk(
                chunk_id=str(uuid.uuid4()),
                content=content,
                start_position=base_position,
                end_position=base_position + len(content),
                metadata={
                    "section_name": section_name,
                    "document_id": document.document_id,
                    "document_type": document.document_type,
                    **self._metadata_to_dict(metadata)
                }
            )]

        # Split into sentences for smarter chunking
        sentences = self._split_sentences(content)

        chunks = []
        current_chunk = []
        current_length = 0
        chunk_start = base_position

        for sentence in sentences:
            sentence_len = len(sentence)

            if current_length + sentence_len > self.chunk_size and current_chunk:
                # Create chunk from current content
                chunk_content = " ".join(current_chunk)
                metadata = self._analyze_chunk(chunk_content, section_name)

                chunks.append(RetrievedChunk(
                    chunk_id=str(uuid.uuid4()),
                    content=chunk_content,
                    start_position=chunk_start,
                    end_position=chunk_start + len(chunk_content),
                    metadata={
                        "section_name": section_name,
                        "document_id": document.document_id,
                        "document_type": document.document_type,
                        **self._metadata_to_dict(metadata)
                    }
                ))

                # Start new chunk with overlap
                overlap_sentences = self._get_overlap_sentences(
                    current_chunk,
                    self.chunk_overlap
                )
                current_chunk = overlap_sentences
                current_length = sum(len(s) for s in overlap_sentences)
                chunk_start = chunk_start + len(chunk_content) - current_length

            current_chunk.append(sentence)
            current_length += sentence_len

        # Handle remaining content
        if current_chunk:
            chunk_content = " ".join(current_chunk)
            metadata = self._analyze_chunk(chunk_content, section_name)

            chunks.append(RetrievedChunk(
                chunk_id=str(uuid.uuid4()),
                content=chunk_content,
                start_position=chunk_start,
                end_position=chunk_start + len(chunk_content),
                metadata={
                    "section_name": section_name,
                    "document_id": document.document_id,
                    "document_type": document.document_type,
                    **self._metadata_to_dict(metadata)
                }
            ))

        return chunks

    def _split_sentences(self, text: str) -> List[str]:
        """Split text into sentences, preserving clinical patterns."""
        # Basic sentence splitting, being careful with abbreviations
        # Common in clinical text: Dr., pt., q.d., b.i.d., etc.
        abbrev_pattern = r"(?<!\b(?:Dr|Mr|Mrs|Ms|pt|q\.d|b\.i\.d|t\.i\.d|p\.r\.n|h\.s|a\.c|p\.c|stat))\.(?=\s+[A-Z])"

        # Split on sentence boundaries
        sentences = re.split(r"(?<=[.!?])\s+(?=[A-Z])", text)

        # Clean up
        sentences = [s.strip() for s in sentences if s.strip()]

        return sentences

    def _get_overlap_sentences(
        self,
        sentences: List[str],
        target_overlap: int
    ) -> List[str]:
        """Get sentences for overlap from the end of the current chunk."""
        overlap_sentences = []
        current_length = 0

        for sentence in reversed(sentences):
            if current_length >= target_overlap:
                break
            overlap_sentences.insert(0, sentence)
            current_length += len(sentence)

        return overlap_sentences

    def _analyze_chunk(self, content: str, section_name: str) -> ChunkMetadata:
        """Analyze chunk content for metadata."""
        return ChunkMetadata(
            section_name=section_name,
            section_type=self._categorize_section(section_name),
            has_medications=bool(self.MEDICATION_PATTERN.search(content)),
            has_lab_values=bool(self.LAB_VALUE_PATTERN.search(content)),
            has_diagnoses=bool(self.DIAGNOSIS_PATTERN.search(content)),
            has_dates=bool(self.DATE_PATTERN.search(content)),
            sentence_count=len(self._split_sentences(content))
        )

    def _categorize_section(self, section_name: str) -> str:
        """Categorize section by type."""
        section_upper = section_name.upper()

        if any(kw in section_upper for kw in ["MEDICATION", "MEDS", "DRUG"]):
            return "medications"
        elif any(kw in section_upper for kw in ["LAB", "LABORATORY"]):
            return "laboratory"
        elif any(kw in section_upper for kw in ["ASSESSMENT", "IMPRESSION", "DIAGNOSIS"]):
            return "diagnosis"
        elif any(kw in section_upper for kw in ["VITAL", "PE", "PHYSICAL"]):
            return "examination"
        elif any(kw in section_upper for kw in ["HISTORY", "HPI", "PMH"]):
            return "history"
        elif any(kw in section_upper for kw in ["PLAN", "TREATMENT"]):
            return "plan"
        elif any(kw in section_upper for kw in ["ALLERG"]):
            return "allergies"
        else:
            return "general"

    def _metadata_to_dict(self, metadata: ChunkMetadata) -> dict:
        """Convert ChunkMetadata to dictionary."""
        return {
            "section_type": metadata.section_type,
            "has_medications": metadata.has_medications,
            "has_lab_values": metadata.has_lab_values,
            "has_diagnoses": metadata.has_diagnoses,
            "has_dates": metadata.has_dates,
            "sentence_count": metadata.sentence_count,
        }
