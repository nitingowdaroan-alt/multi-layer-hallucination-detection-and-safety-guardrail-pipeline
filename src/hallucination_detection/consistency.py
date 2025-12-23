"""
Layer 3: Self-Consistency Checks

Runs extraction multiple times with controlled randomness and
compares outputs to detect unstable or unreliable extractions.
"""

import json
import re
from typing import List, Optional, Any, Dict, Tuple
from dataclasses import dataclass, field
from collections import Counter
import statistics

from ..core.models import RetrievedChunk, ConsistencyResult, FieldType, ExtractionField
from ..core.config import HallucinationDetectionConfig, ModelConfig


@dataclass
class ConsistencyRun:
    """Result of a single extraction run."""
    run_id: int
    extracted_value: Optional[str]
    raw_response: str
    temperature: float


@dataclass
class ValueVariance:
    """Variance analysis for extracted values."""
    unique_values: List[str]
    value_counts: Dict[str, int]
    most_common: Tuple[str, int]
    entropy: float
    normalized_variance: float


class SelfConsistencyChecker:
    """
    Checks self-consistency of extractions by running multiple times.

    This layer:
    1. Runs the extraction N times with slight temperature variation
    2. Compares outputs across runs
    3. Flags fields with high variance (unstable extractions)
    4. Detects confidence decay across runs

    High consistency = reliable extraction
    Low consistency = potential hallucination or ambiguous source
    """

    def __init__(
        self,
        model_config: Optional[ModelConfig] = None,
        detection_config: Optional[HallucinationDetectionConfig] = None,
        llm_client: Optional[Any] = None
    ):
        self.model_config = model_config or ModelConfig()
        self.detection_config = detection_config or HallucinationDetectionConfig()
        self._llm_client = llm_client
        self.num_runs = self.detection_config.consistency_num_runs

    @property
    def llm_client(self):
        """Lazy initialization of LLM client."""
        if self._llm_client is None:
            try:
                from openai import OpenAI
                self._llm_client = OpenAI()
            except ImportError:
                raise ImportError("OpenAI package required.")
        return self._llm_client

    def check_consistency(
        self,
        field: ExtractionField,
        context: str,
        initial_value: Optional[str] = None,
        num_runs: Optional[int] = None
    ) -> ConsistencyResult:
        """
        Check extraction consistency by running multiple times.

        Args:
            field: Field being extracted
            context: The retrieval context text
            initial_value: The initial extraction (counts as run 1)
            num_runs: Number of additional runs

        Returns:
            ConsistencyResult with consistency metrics
        """
        num_runs = num_runs or self.num_runs
        runs = []

        # Include initial value as first run
        if initial_value:
            runs.append(ConsistencyRun(
                run_id=0,
                extracted_value=initial_value,
                raw_response=initial_value,
                temperature=0.0
            ))
            num_runs -= 1

        # Run additional extractions with varied temperature
        temperatures = self._generate_temperatures(num_runs)

        for i, temp in enumerate(temperatures):
            run_result = self._run_extraction(
                field=field,
                context=context,
                temperature=temp,
                run_id=len(runs)
            )
            runs.append(run_result)

        # Analyze consistency
        return self._analyze_consistency(field.name, runs)

    def _generate_temperatures(self, num_runs: int) -> List[float]:
        """Generate temperature values for runs."""
        base_temp = self.model_config.consistency_temperature
        temps = []

        for i in range(num_runs):
            # Slight variation around base temperature
            variation = (i / max(num_runs - 1, 1)) * 0.2
            temps.append(min(base_temp + variation, 1.0))

        return temps

    def _run_extraction(
        self,
        field: ExtractionField,
        context: str,
        temperature: float,
        run_id: int
    ) -> ConsistencyRun:
        """Run a single extraction with specified temperature."""
        prompt = self._build_extraction_prompt(field, context)

        response = self.llm_client.chat.completions.create(
            model=self.model_config.extraction_model,
            messages=[
                {"role": "system", "content": self._get_system_prompt()},
                {"role": "user", "content": prompt}
            ],
            temperature=temperature,
            max_tokens=512
        )

        raw_response = response.choices[0].message.content
        extracted_value = self._parse_response(raw_response)

        return ConsistencyRun(
            run_id=run_id,
            extracted_value=extracted_value,
            raw_response=raw_response,
            temperature=temperature
        )

    def _get_system_prompt(self) -> str:
        """Get system prompt for consistency runs."""
        return """You are a clinical data extraction system. Extract ONLY the requested
field from the provided clinical text.

RULES:
1. Extract ONLY what is explicitly stated
2. If not found, respond with "NOT_FOUND"
3. Do not infer or guess
4. Be precise and consistent

Respond with JSON: {"value": "<extracted value or NOT_FOUND>"}"""

    def _build_extraction_prompt(
        self,
        field: ExtractionField,
        context: str
    ) -> str:
        """Build extraction prompt for consistency run."""
        return f"""Extract the following field:

FIELD: {field.name}
TYPE: {field.field_type.value}
DESCRIPTION: {field.description}

CLINICAL TEXT:
{context}

Respond with JSON only: {{"value": "<extracted value or NOT_FOUND>"}}"""

    def _parse_response(self, response: str) -> Optional[str]:
        """Parse extraction response."""
        try:
            response_clean = response.strip()
            if response_clean.startswith("```"):
                response_clean = re.sub(r'^```\w*\n?', '', response_clean)
                response_clean = re.sub(r'\n?```$', '', response_clean)

            data = json.loads(response_clean)
            value = data.get("value")

            if value in ["NOT_FOUND", None, ""]:
                return None
            return str(value)

        except json.JSONDecodeError:
            # Try to extract value directly
            if "NOT_FOUND" in response:
                return None
            return response.strip()

    def _analyze_consistency(
        self,
        field_name: str,
        runs: List[ConsistencyRun]
    ) -> ConsistencyResult:
        """Analyze consistency across multiple runs."""
        # Extract all non-None values
        values = [r.extracted_value for r in runs if r.extracted_value]
        all_values = [r.extracted_value for r in runs]

        if not values:
            return ConsistencyResult(
                field_name=field_name,
                num_runs=len(runs),
                unique_values=[],
                agreement_rate=0.0,
                variance_score=1.0,
                most_common_value=None,
                confidence_decay=1.0,
                is_stable=False
            )

        # Normalize values for comparison
        normalized_values = [self._normalize_value(v) for v in values]

        # Count unique values
        value_counts = Counter(normalized_values)
        unique_values = list(value_counts.keys())

        # Most common value
        most_common, most_common_count = value_counts.most_common(1)[0]

        # Agreement rate
        agreement_rate = most_common_count / len(runs)

        # Variance score (0 = no variance, 1 = all different)
        variance_score = 1 - (most_common_count / len(runs))

        # Confidence decay (how much confidence drops as we add runs)
        confidence_decay = self._compute_confidence_decay(all_values)

        # Determine stability
        is_stable = (
            agreement_rate >= self.detection_config.consistency_agreement_threshold and
            variance_score <= self.detection_config.consistency_variance_threshold
        )

        # Map back to original value format
        original_most_common = None
        for v in values:
            if self._normalize_value(v) == most_common:
                original_most_common = v
                break

        return ConsistencyResult(
            field_name=field_name,
            num_runs=len(runs),
            unique_values=unique_values,
            agreement_rate=agreement_rate,
            variance_score=variance_score,
            most_common_value=original_most_common,
            confidence_decay=confidence_decay,
            is_stable=is_stable
        )

    def _normalize_value(self, value: str) -> str:
        """Normalize value for comparison."""
        if not value:
            return ""

        # Lowercase and strip whitespace
        normalized = value.lower().strip()

        # Normalize numbers
        normalized = re.sub(r'(\d),(\d)', r'\1\2', normalized)

        # Normalize units
        unit_map = {
            'milligrams': 'mg',
            'grams': 'g',
            'micrograms': 'mcg',
            'milliliters': 'ml',
            'liters': 'l',
        }
        for full, abbr in unit_map.items():
            normalized = normalized.replace(full, abbr)

        return normalized

    def _compute_confidence_decay(
        self,
        values: List[Optional[str]]
    ) -> float:
        """
        Compute confidence decay across runs.

        Measures how much the "consensus" changes as we add more runs.
        High decay = unstable, low confidence
        """
        if len(values) < 2:
            return 0.0

        # Track agreement ratio as we add runs
        agreement_ratios = []

        for i in range(2, len(values) + 1):
            subset = [v for v in values[:i] if v]
            if subset:
                counts = Counter([self._normalize_value(v) for v in subset])
                most_common_count = counts.most_common(1)[0][1]
                ratio = most_common_count / i
                agreement_ratios.append(ratio)

        if len(agreement_ratios) < 2:
            return 0.0

        # Compute how much agreement drops
        initial_agreement = agreement_ratios[0]
        final_agreement = agreement_ratios[-1]

        decay = max(0, initial_agreement - final_agreement)
        return decay

    def quick_consistency_check(
        self,
        value1: str,
        value2: str,
        field_type: FieldType
    ) -> bool:
        """Quick check if two values are consistent."""
        norm1 = self._normalize_value(value1)
        norm2 = self._normalize_value(value2)

        if norm1 == norm2:
            return True

        # For numeric fields, check numeric equivalence
        if field_type in [FieldType.DOSAGE, FieldType.LAB_VALUE]:
            nums1 = re.findall(r'\d+\.?\d*', value1)
            nums2 = re.findall(r'\d+\.?\d*', value2)
            if nums1 and nums2 and nums1 == nums2:
                return True

        return False

    def compute_consensus_value(
        self,
        runs: List[ConsistencyRun]
    ) -> Tuple[Optional[str], float]:
        """
        Compute the consensus value from multiple runs.

        Returns:
            Tuple of (consensus_value, confidence)
        """
        values = [r.extracted_value for r in runs if r.extracted_value]

        if not values:
            return None, 0.0

        normalized = [self._normalize_value(v) for v in values]
        counts = Counter(normalized)
        most_common_norm, count = counts.most_common(1)[0]

        # Find original value
        for v in values:
            if self._normalize_value(v) == most_common_norm:
                confidence = count / len(runs)
                return v, confidence

        return None, 0.0
