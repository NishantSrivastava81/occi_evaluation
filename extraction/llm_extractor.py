"""
LLM-Based Structural Causal Model Extraction for OCCI Framework.

This module implements the LLM extraction component of the Ontology-Constrained
Causal Induction (OCCI) pipeline. It uses Azure OpenAI GPT-5 to propose candidate
variables and causal edges from legal contract text, subject to ontology constraints.

The extraction follows the methodology described in Section 4 of the paper:
1. Structured prompting with ontology type definitions
2. JSON-formatted output for reliable parsing
3. Multi-prompt consensus for robustness (optional)
"""

import json
import re
import logging
from typing import Dict, List, Optional, Tuple, Any
from dataclasses import dataclass
import os

# Azure OpenAI SDK (primary)
try:
    from openai import AzureOpenAI, OpenAI
except ImportError:
    AzureOpenAI = None
    OpenAI = None

# Anthropic SDK (for ablation studies)
try:
    from anthropic import Anthropic
except ImportError:
    Anthropic = None

import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).parent.parent))

from ontology.legal_ontology import Variable, Edge, OntologyType, LEGAL_ONTOLOGY
from utils.extraction_logger import get_extraction_logger
from config.settings import (
    # Azure OpenAI (primary)
    AZURE_OPENAI_ENDPOINT,
    AZURE_OPENAI_API_KEY,
    AZURE_OPENAI_API_VERSION,
    AZURE_GPT5_DEPLOYMENT,
    AZURE_GPT4_DEPLOYMENT,
    # Fallback providers
    OPENAI_API_KEY,
    ANTHROPIC_API_KEY,
    # Model parameters
    TEMPERATURE,
    MAX_TOKENS,
    TOP_P,
    FREQUENCY_PENALTY,
    PRESENCE_PENALTY,
    # Extraction settings
    ORIENTATION_PATTERNS,
    VALIDATOR_CONFIG,
)

# Configure module logger
logger = logging.getLogger(__name__)


# =============================================================================
# EXTRACTION PROMPT TEMPLATE
# =============================================================================
# This prompt implements the structured extraction methodology from Section 4.2.
# The prompt explicitly encodes ontology constraints to guide LLM extraction.

EXTRACTION_PROMPT = """You are an expert legal analyst specializing in causal structure extraction from commercial contracts.

TASK: Extract the Structural Causal Model (SCM) from the provided contract text.

## ONTOLOGY SPECIFICATION

### Type Hierarchy (Partial Order ≺):
The following types form a directed acyclic order for valid causal edges:
  Clause ≺ Obligation ≺ ComplianceEvent ≺ Outcome ≺ Remedy or Cost

### Type Definitions:
- **Clause**: Contractual provisions, terms, or sections (e.g., "Insurance Requirements", "Section 5.2")
- **Obligation**: Duties imposed on parties, typically indicated by modal verbs ("shall", "must", "will")
- **ComplianceEvent**: Events indicating fulfillment or breach of obligations ("failure to maintain", "breach of")
- **Outcome**: Legal consequences triggered by compliance events ("material breach", "termination event", "default")
- **Remedy**: Actions available to non-breaching party ("may terminate", "entitled to cure", "right to damages")
- **Cost**: Financial consequences ("liable for damages", "penalty of $X", "indemnification")

### Allowed Edge Types (Relation Set R):
Only the following directed edges are valid in the SCM:
  - Clause → Obligation
  - Obligation → ComplianceEvent  
  - ComplianceEvent → Outcome
  - Outcome → Remedy
  - Outcome → Cost

## EXTRACTION REQUIREMENTS

1. **Span Grounding**: Every variable MUST include the exact text span from the contract
2. **Orientation Markers**: Every edge MUST include the linguistic marker indicating causation
   (e.g., "shall constitute", "results in", "upon", "in the event of", "triggers")
3. **Determinism**: Mark edges as deterministic only if the text uses definitive language
   ("shall constitute", "shall be deemed") vs. conditional ("may result in", "could lead to")
4. **No Hallucination**: Extract ONLY relationships explicitly stated in the text

## OUTPUT FORMAT (JSON)
```json
{{
  "variables": [
    {{
      "name": "UniqueVariableName",
      "type": "Clause|Obligation|ComplianceEvent|Outcome|Remedy|Cost",
      "span_text": "exact quoted text from contract",
      "span_start": 0,
      "span_end": 50,
      "attributes": {{"key": "value"}}
    }}
  ],
  "edges": [
    {{
      "source": "SourceVariableName",
      "source_type": "SourceType",
      "target": "TargetVariableName", 
      "target_type": "TargetType",
      "orientation_marker": "shall constitute",
      "span_evidence": "exact text containing the causal relationship",
      "is_deterministic": true
    }}
  ],
  "metadata": {{
    "n_variables": 5,
    "n_edges": 4,
    "extraction_confidence": 0.85
  }}
}}
```

## CONTRACT TEXT FOR ANALYSIS:
---
{contract_text}
---

Extract the complete causal structure. Return ONLY valid JSON."""


# =============================================================================
# PROMPT VARIATIONS FOR MULTI-PROMPT CONSENSUS
# =============================================================================
# Alternative prompts for robustness testing (Section 4.3)

PROMPT_VARIATIONS = [
    # Variation 1: Focus on obligations
    "Focus primarily on identifying OBLIGATIONS (shall/must/will statements) and their consequences.",
    # Variation 2: Focus on failures/breaches
    "Focus primarily on FAILURE conditions and their resulting OUTCOMES.",
    # Variation 3: Focus on remedies
    "Focus primarily on REMEDIES and COSTS that arise from contract breaches.",
    # Variation 4: Bottom-up extraction
    "Start by identifying COSTS and REMEDIES, then trace back to their causal antecedents.",
    # Variation 5: Clause-first extraction
    "Start by identifying all CLAUSES, then enumerate the obligations each clause imposes.",
]


# =============================================================================
# LLM EXTRACTOR CLASS
# =============================================================================


class LLMExtractor:
    """
    Ontology-Constrained LLM Extractor for Structural Causal Models.

    This class implements the LLM-based extraction component of the OCCI framework.
    It uses Azure OpenAI GPT-5 as the primary model, with fallback support for
    GPT-4o and Claude for ablation studies.

    The extractor:
    1. Sends structured prompts with ontology constraints
    2. Parses JSON responses into typed Variable and Edge objects
    3. Optionally uses multi-prompt consensus for robustness

    Attributes:
        model: Model identifier (deployment name for Azure)
        provider: Backend provider ('azure', 'openai', 'anthropic')
        temperature: Sampling temperature (0.0 for determinism)
        max_tokens: Maximum response tokens

    Example:
        >>> extractor = LLMExtractor(model="gpt-5", provider="azure")
        >>> variables, edges = extractor.extract(contract_text)
    """

    def __init__(
        self,
        model: str = None,
        provider: str = "azure",
        temperature: float = TEMPERATURE,
        max_tokens: int = MAX_TOKENS,
        ontology: Any = None,
    ):
        """
        Initialize the LLM extractor.

        Args:
            model: Model name or Azure deployment name. Defaults to GPT-5.
            provider: Backend provider ('azure', 'openai', 'anthropic')
            temperature: Sampling temperature (default: 0.0 for reproducibility)
            max_tokens: Maximum tokens in response
            ontology: Ontology instance (defaults to LEGAL_ONTOLOGY)
        """
        self.provider = provider.lower()
        self.temperature = temperature
        self.max_tokens = max_tokens
        self.ontology = ontology or LEGAL_ONTOLOGY

        # Set default model based on provider
        if model is None:
            if self.provider == "azure":
                model = AZURE_GPT5_DEPLOYMENT
            else:
                model = "gpt-4o"
        self.model = model

        # Initialize client based on provider
        self._init_client()

        logger.info(
            f"Initialized LLMExtractor: provider={self.provider}, model={self.model}"
        )

    def _init_client(self):
        """Initialize the appropriate API client."""
        if self.provider == "azure":
            if AzureOpenAI is None:
                raise ImportError("openai package required: pip install openai>=1.0.0")
            if not AZURE_OPENAI_ENDPOINT or not AZURE_OPENAI_API_KEY:
                raise ValueError(
                    "Azure OpenAI credentials not configured. "
                    "Set AZURE_OPENAI_ENDPOINT and AZURE_OPENAI_API_KEY in environment."
                )
            self.client = AzureOpenAI(
                azure_endpoint=AZURE_OPENAI_ENDPOINT,
                api_key=AZURE_OPENAI_API_KEY,
                api_version=AZURE_OPENAI_API_VERSION,
            )

        elif self.provider == "openai":
            if OpenAI is None:
                raise ImportError("openai package required: pip install openai>=1.0.0")
            self.client = OpenAI(api_key=OPENAI_API_KEY)

        elif self.provider == "anthropic":
            if Anthropic is None:
                raise ImportError("anthropic package required: pip install anthropic")
            self.client = Anthropic(api_key=ANTHROPIC_API_KEY)

        else:
            raise ValueError(
                f"Unknown provider: {self.provider}. Use 'azure', 'openai', or 'anthropic'."
            )

    def extract(
        self, contract_text: str, max_length: int = 12000
    ) -> Tuple[List[Variable], List[Edge]]:
        """
        Extract candidate SCM structure from contract text.

        This method implements the main extraction step of the OCCI pipeline.
        It sends a structured prompt to the LLM and parses the response into
        typed Variable and Edge objects.

        Args:
            contract_text: Raw contract text to analyze
            max_length: Maximum text length to process (truncated if exceeded)

        Returns:
            Tuple of (variables, edges) extracted from the contract

        Raises:
            Exception: If LLM call fails after retries
        """
        # Truncate if needed (preserve sentence boundaries)
        if len(contract_text) > max_length:
            truncate_point = contract_text.rfind(".", 0, max_length)
            if truncate_point > max_length * 0.8:
                contract_text = contract_text[: truncate_point + 1]
            else:
                contract_text = contract_text[:max_length]
            logger.warning(
                f"Contract text truncated to {len(contract_text)} characters"
            )

        prompt = EXTRACTION_PROMPT.format(contract_text=contract_text)

        # Call LLM with retry logic and logging
        import time as time_module
        start_time = time_module.time()
        response_text = self._call_llm(prompt)
        latency = time_module.time() - start_time

        # Parse response into structured objects
        variables, edges = self._parse_response(response_text)

        # Log the LLM call for transparency
        try:
            extraction_logger = get_extraction_logger()
            extraction_logger.log_llm_call(
                provider=self.provider,
                model=self.model,
                contract_id=getattr(self, '_current_contract_id', 'unknown'),
                prompt=prompt,
                response=response_text,
                parsed_variables=len(variables),
                parsed_edges=len(edges),
                latency=latency,
                success=True
            )
        except Exception as log_error:
            logger.debug(f"Logging skipped: {log_error}")

        logger.debug(f"Extracted {len(variables)} variables and {len(edges)} edges")

        return variables, edges

    def extract_with_consensus(
        self, contract_text: str, num_prompts: int = None, threshold: int = None
    ) -> Tuple[List[Variable], List[Edge]]:
        """
        Extract using multiple prompt variations with consensus voting.

        This implements the multi-prompt robustness mechanism from Section 4.3.
        An edge/variable is accepted only if it appears in >= threshold extractions.

        Args:
            contract_text: Contract text to analyze
            num_prompts: Number of prompt variations (default from config)
            threshold: Minimum agreement count to accept (default from config)

        Returns:
            Tuple of (variables, edges) passing consensus threshold
        """
        num_prompts = num_prompts or VALIDATOR_CONFIG.get("num_prompts", 5)
        threshold = threshold or VALIDATOR_CONFIG.get("multi_prompt_threshold", 3)

        all_variables = []
        all_edges = []
        successful_extractions = 0

        # Run extraction with prompt variations
        for i in range(num_prompts):
            try:
                # Add variation instruction if available
                variation = PROMPT_VARIATIONS[i] if i < len(PROMPT_VARIATIONS) else ""

                if variation:
                    modified_text = f"{variation}\n\n{contract_text}"
                    vars_i, edges_i = self.extract(modified_text)
                else:
                    vars_i, edges_i = self.extract(contract_text)

                all_variables.extend(vars_i)
                all_edges.extend(edges_i)
                successful_extractions += 1

            except Exception as e:
                logger.warning(f"Extraction attempt {i+1}/{num_prompts} failed: {e}")
                continue

        if successful_extractions == 0:
            logger.error("All extraction attempts failed")
            return [], []

        # Aggregate and count occurrences
        var_counts = {}
        for var in all_variables:
            key = (var.name, var.var_type.value)
            if key not in var_counts:
                var_counts[key] = {"count": 0, "var": var}
            var_counts[key]["count"] += 1

        edge_counts = {}
        for edge in all_edges:
            key = (edge.source.name, edge.target.name)
            if key not in edge_counts:
                edge_counts[key] = {"count": 0, "edge": edge}
            edge_counts[key]["count"] += 1

        # Filter by consensus threshold
        consensus_vars = [
            v["var"] for v in var_counts.values() if v["count"] >= threshold
        ]
        consensus_edges = [
            e["edge"] for e in edge_counts.values() if e["count"] >= threshold
        ]

        logger.info(
            f"Consensus extraction: {len(consensus_vars)} variables, "
            f"{len(consensus_edges)} edges from {successful_extractions} runs"
        )

        return consensus_vars, consensus_edges

    def _call_llm(self, prompt: str, max_retries: int = 3) -> str:
        """
        Call the LLM API with retry logic.

        Args:
            prompt: The prompt to send
            max_retries: Maximum number of retry attempts

        Returns:
            Response text from the model

        Raises:
            Exception: If all retries fail
        """
        import time

        last_error = None

        for attempt in range(max_retries):
            try:
                if self.provider == "azure" or self.provider == "openai":
                    response = self.client.chat.completions.create(
                        model=self.model,
                        messages=[
                            {
                                "role": "system",
                                "content": "You are an expert legal analyst specializing in causal structure extraction. Output valid JSON only.",
                            },
                            {"role": "user", "content": prompt},
                        ],
                        temperature=self.temperature,
                        max_completion_tokens=self.max_tokens,  # GPT-5 uses max_completion_tokens
                        top_p=TOP_P,
                        frequency_penalty=FREQUENCY_PENALTY,
                        presence_penalty=PRESENCE_PENALTY,
                        response_format={"type": "json_object"},
                    )
                    # GPT-5 may return content in different ways
                    content = response.choices[0].message.content
                    if content:
                        return content
                    # Try to get from parsed field if using structured outputs
                    if (
                        hasattr(response.choices[0].message, "parsed")
                        and response.choices[0].message.parsed
                    ):
                        return json.dumps(response.choices[0].message.parsed)
                    # Fallback to checking reasoning/refusal
                    if (
                        hasattr(response.choices[0].message, "refusal")
                        and response.choices[0].message.refusal
                    ):
                        logger.warning(
                            f"Model refused: {response.choices[0].message.refusal}"
                        )
                        return "{}"
                    logger.warning(
                        f"Empty response from model. Full response: {response}"
                    )
                    return "{}"

                elif self.provider == "anthropic":
                    response = self.client.messages.create(
                        model=self.model,
                        max_tokens=self.max_tokens,
                        messages=[{"role": "user", "content": prompt}],
                    )
                    return response.content[0].text

                else:
                    raise ValueError(f"Unknown provider: {self.provider}")

            except Exception as e:
                last_error = e
                logger.warning(f"LLM call attempt {attempt + 1} failed: {e}")
                if attempt < max_retries - 1:
                    wait_time = 2**attempt  # Exponential backoff
                    time.sleep(wait_time)

        raise Exception(f"LLM call failed after {max_retries} attempts: {last_error}")

    def _parse_response(self, response_text: str) -> Tuple[List[Variable], List[Edge]]:
        """
        Parse LLM JSON response into Variable and Edge objects.

        Args:
            response_text: Raw JSON response from LLM

        Returns:
            Tuple of (variables, edges) lists
        """
        variables = []
        edges = []

        # Extract JSON from response (handle markdown code blocks)
        try:
            # Remove markdown code blocks if present
            cleaned = re.sub(r"^```json\s*", "", response_text.strip())
            cleaned = re.sub(r"\s*```$", "", cleaned)

            # Try to find JSON block
            json_match = re.search(r"\{[\s\S]*\}", cleaned)
            if json_match:
                data = json.loads(json_match.group())
            else:
                data = json.loads(cleaned)

        except json.JSONDecodeError as e:
            logger.error(f"JSON parse error: {e}")
            logger.error(
                f"Response text (first 1000 chars): {response_text[:1000] if response_text else 'EMPTY RESPONSE'}"
            )
            return variables, edges

        # Parse variables
        var_map = {}
        for var_data in data.get("variables", []):
            try:
                var_type = self.ontology.string_to_type(var_data.get("type", ""))
                if var_type is None:
                    logger.warning(f"Unknown type: {var_data.get('type')}")
                    continue

                var = Variable(
                    name=var_data.get("name", ""),
                    var_type=var_type,
                    span_text=var_data.get("span_text", ""),
                    span_start=var_data.get("span_start", 0),
                    span_end=var_data.get("span_end", 0),
                    attributes=var_data.get("attributes", {}),
                )
                variables.append(var)
                var_map[var.name] = var
            except Exception as e:
                logger.warning(f"Error parsing variable: {e}")
                continue

        # Parse edges
        for edge_data in data.get("edges", []):
            try:
                source_name = edge_data.get("source", "")
                target_name = edge_data.get("target", "")

                # Get or create source variable
                if source_name in var_map:
                    source = var_map[source_name]
                else:
                    source_type = self.ontology.string_to_type(
                        edge_data.get("source_type", "")
                    )
                    if source_type is None:
                        logger.warning(f"Unknown source type for edge: {source_name}")
                        continue
                    source = Variable(
                        name=source_name,
                        var_type=source_type,
                        span_text=edge_data.get("span_evidence", "")[:100],
                    )
                    var_map[source_name] = source
                    variables.append(source)

                # Get or create target variable
                if target_name in var_map:
                    target = var_map[target_name]
                else:
                    target_type = self.ontology.string_to_type(
                        edge_data.get("target_type", "")
                    )
                    if target_type is None:
                        logger.warning(f"Unknown target type for edge: {target_name}")
                        continue
                    target = Variable(
                        name=target_name,
                        var_type=target_type,
                        span_text=edge_data.get("span_evidence", "")[:100],
                    )
                    var_map[target_name] = target
                    variables.append(target)

                edge = Edge(
                    source=source,
                    target=target,
                    orientation_marker=edge_data.get("orientation_marker", ""),
                    span_evidence=edge_data.get("span_evidence", ""),
                    is_deterministic=edge_data.get("is_deterministic", False),
                )
                edges.append(edge)

            except Exception as e:
                logger.warning(f"Error parsing edge: {e}")
                continue

        logger.debug(f"Parsed {len(variables)} variables and {len(edges)} edges")
        return variables, edges


# =============================================================================
# BASELINE: UNCONSTRAINED LLM EXTRACTOR
# =============================================================================


class UnconstrainedLLMExtractor(LLMExtractor):
    """
    Unconstrained LLM Extraction Baseline (Ablation Study).

    This baseline uses the same LLM (Azure OpenAI GPT-5) but WITHOUT
    ontology constraints. It serves as an ablation to measure the
    contribution of ontological guidance to extraction quality.

    The prompt asks for generic causal relationships without:
    - Type hierarchy constraints
    - Allowed edge set restrictions
    - Orientation marker requirements

    This allows direct comparison with OCCI to isolate the effect
    of ontology constraints on structural correctness and FPR.
    """

    UNCONSTRAINED_PROMPT = """Extract all causal relationships from this legal contract.

For each causal relationship you identify:
1. CAUSE: The event, condition, or action that triggers the effect
2. EFFECT: The resulting consequence or outcome
3. EVIDENCE: The exact text from the contract supporting this relationship
4. STRENGTH: Whether this is a definite/deterministic relationship or conditional/probabilistic

Output Format (JSON):
{{
  "relationships": [
    {{
      "cause": "failure to maintain insurance",
      "effect": "material breach",
      "evidence": "Failure to maintain insurance shall constitute material breach",
      "strength": "deterministic"
    }}
  ],
  "metadata": {{
    "n_relationships": 5
  }}
}}

CONTRACT TEXT:
---
{contract_text}
---

Extract ALL causal relationships. Return valid JSON only."""

    def extract(
        self, contract_text: str, max_length: int = 12000
    ) -> Tuple[List[Variable], List[Edge]]:
        """
        Extract causal structure without ontology constraints.

        This method extracts generic cause-effect relationships and maps
        them to default types for comparison with ontology-constrained methods.

        Args:
            contract_text: Contract text to analyze
            max_length: Maximum text length

        Returns:
            Tuple of (variables, edges) with default type assignments
        """
        if len(contract_text) > max_length:
            truncate_point = contract_text.rfind(".", 0, max_length)
            if truncate_point > max_length * 0.8:
                contract_text = contract_text[: truncate_point + 1]
            else:
                contract_text = contract_text[:max_length]

        prompt = self.UNCONSTRAINED_PROMPT.format(contract_text=contract_text)
        response_text = self._call_llm(prompt)

        # Parse into structured format (assign default types)
        variables = []
        edges = []
        var_id = 0

        try:
            # Clean and parse JSON
            cleaned = re.sub(r"^```json\s*", "", response_text.strip())
            cleaned = re.sub(r"\s*```$", "", cleaned)

            json_match = re.search(r"\{[\s\S]*\}", cleaned)
            if json_match:
                data = json.loads(json_match.group())
            else:
                data = json.loads(cleaned)

            for rel in data.get("relationships", []):
                cause_name = f"unconstrained_cause_{var_id}"
                var_id += 1
                effect_name = f"unconstrained_effect_{var_id}"
                var_id += 1

                # Assign default types (no ontology guidance)
                # This represents what an LLM produces without type constraints
                cause_var = Variable(
                    name=cause_name,
                    var_type=OntologyType.CLAUSE,  # Default type
                    span_text=rel.get("cause", ""),
                )
                effect_var = Variable(
                    name=effect_name,
                    var_type=OntologyType.OUTCOME,  # Default type
                    span_text=rel.get("effect", ""),
                )

                variables.extend([cause_var, effect_var])

                # Determine if deterministic
                is_deterministic = rel.get("strength", "").lower() == "deterministic"

                edge = Edge(
                    source=cause_var,
                    target=effect_var,
                    orientation_marker="",  # No orientation marker required
                    span_evidence=rel.get("evidence", ""),
                    is_deterministic=is_deterministic,
                )
                edges.append(edge)

        except Exception as e:
            logger.error(f"Unconstrained extraction parse error: {e}")

        logger.debug(
            f"Unconstrained extraction: {len(variables)} vars, {len(edges)} edges"
        )
        return variables, edges
