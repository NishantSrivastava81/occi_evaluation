"""
Pattern-based causal extraction baseline.

Uses regex patterns to extract causal relationships without LLMs.
"""

import re
import time
from typing import List, Tuple
from dataclasses import dataclass

import sys
from pathlib import Path

sys.path.append(str(Path(__file__).parent.parent))

from ontology.legal_ontology import Variable, Edge, OntologyType, LEGAL_ONTOLOGY
from utils.extraction_logger import get_extraction_logger


class PatternExtractor:
    """
    Pattern-based extraction baseline.

    Uses predefined regex patterns to extract causal relations.
    Does not use LLMs or enforce ontology typing.
    """

    # Causal patterns: (regex, source_type, target_type)
    CAUSAL_PATTERNS = [
        # Obligation patterns
        (r"(?P<actor>\w+)\s+shall\s+(?P<obligation>[^.]+)", "Clause", "Obligation"),
        (r"(?P<actor>\w+)\s+must\s+(?P<obligation>[^.]+)", "Clause", "Obligation"),
        (
            r"(?P<actor>\w+)\s+agrees?\s+to\s+(?P<obligation>[^.]+)",
            "Clause",
            "Obligation",
        ),
        # Breach/failure patterns
        (
            r"failure\s+to\s+(?P<action>[^.]+)\s+(shall\s+)?(constitute|be\s+deemed)\s+(?P<outcome>[^.]+)",
            "ComplianceEvent",
            "Outcome",
        ),
        (
            r"breach\s+of\s+(?P<obligation>[^.]+)\s+(shall\s+)?(result|lead)\s+in\s+(?P<outcome>[^.]+)",
            "ComplianceEvent",
            "Outcome",
        ),
        # Consequence patterns
        (
            r"(?P<trigger>[^.]+)\s+shall\s+constitute\s+(?P<outcome>[^.]+)",
            "ComplianceEvent",
            "Outcome",
        ),
        (
            r"(?P<trigger>[^.]+)\s+shall\s+be\s+deemed\s+(?P<outcome>[^.]+)",
            "ComplianceEvent",
            "Outcome",
        ),
        # Remedy patterns
        (
            r"upon\s+(?P<event>[^,]+),?\s+(?P<party>\w+)\s+may\s+(?P<remedy>[^.]+)",
            "Outcome",
            "Remedy",
        ),
        (
            r"in\s+the\s+event\s+of\s+(?P<event>[^,]+),?\s+(?P<party>\w+)\s+(shall\s+be\s+entitled|may)\s+(?P<remedy>[^.]+)",
            "Outcome",
            "Remedy",
        ),
        # Damages patterns
        (
            r"(?P<party>\w+)\s+(shall\s+be\s+)?liable\s+for\s+(?P<damages>[^.]+)",
            "Outcome",
            "Cost",
        ),
        (
            r"(?P<party>\w+)\s+shall\s+(pay|reimburse|compensate)\s+(?P<amount>[^.]+)",
            "Outcome",
            "Cost",
        ),
        (
            r"damages?\s+(not\s+to\s+exceed|limited\s+to|capped\s+at)\s+(?P<amount>[^.]+)",
            "Outcome",
            "Cost",
        ),
    ]

    def __init__(self):
        self.compiled_patterns = [
            (re.compile(p, re.IGNORECASE), src, tgt)
            for p, src, tgt in self.CAUSAL_PATTERNS
        ]

    def extract(self, contract_text: str) -> Tuple[List[Variable], List[Edge]]:
        """
        Extract causal structure using patterns.

        Returns:
            (variables, edges) tuple
        """
        variables = []
        edges = []
        var_id = 0
        var_map = {}

        for pattern, src_type_str, tgt_type_str in self.compiled_patterns:
            for match in pattern.finditer(contract_text):
                try:
                    groups = match.groupdict()
                    matched_text = match.group()

                    # Create source variable
                    src_name = f"pattern_src_{var_id}"
                    var_id += 1
                    src_type = LEGAL_ONTOLOGY.string_to_type(src_type_str)

                    if src_type:
                        src_var = Variable(
                            name=src_name,
                            var_type=src_type,
                            span_text=matched_text[:100],
                            span_start=match.start(),
                            span_end=match.end(),
                        )
                        variables.append(src_var)
                        var_map[src_name] = src_var

                    # Create target variable
                    tgt_name = f"pattern_tgt_{var_id}"
                    var_id += 1
                    tgt_type = LEGAL_ONTOLOGY.string_to_type(tgt_type_str)

                    if tgt_type:
                        tgt_var = Variable(
                            name=tgt_name,
                            var_type=tgt_type,
                            span_text=matched_text[:100],
                            span_start=match.start(),
                            span_end=match.end(),
                        )
                        variables.append(tgt_var)
                        var_map[tgt_name] = tgt_var

                    # Create edge
                    if src_type and tgt_type:
                        # Find orientation marker
                        markers = LEGAL_ONTOLOGY.find_orientation_markers(matched_text)
                        marker = markers[0][0] if markers else ""

                        edge = Edge(
                            source=src_var,
                            target=tgt_var,
                            orientation_marker=marker,
                            span_evidence=matched_text,
                            is_deterministic=LEGAL_ONTOLOGY.is_deterministic_relation(
                                matched_text
                            ),
                        )
                        edges.append(edge)

                except Exception as e:
                    print(f"Pattern extraction error: {e}")
                    continue

        # Log extraction for transparency
        try:
            extraction_logger = get_extraction_logger()
            extraction_logger.log_extraction(
                method="pattern",
                contract_id=getattr(self, '_current_contract_id', 'unknown'),
                input_text=contract_text[:1000],  # First 1000 chars
                variables=variables,
                edges=edges,
                extraction_time=0.0  # Pattern extraction is fast
            )
        except Exception as log_error:
            pass  # Logging should not break extraction

        return variables, edges

    def get_pattern_stats(self, contract_text: str) -> dict:
        """Get statistics about pattern matches."""
        stats = {"total_matches": 0, "pattern_counts": {}}

        for i, (pattern, src_type, tgt_type) in enumerate(self.compiled_patterns):
            matches = list(pattern.finditer(contract_text))
            pattern_key = f"{src_type}->{tgt_type}"

            if pattern_key not in stats["pattern_counts"]:
                stats["pattern_counts"][pattern_key] = 0

            stats["pattern_counts"][pattern_key] += len(matches)
            stats["total_matches"] += len(matches)

        return stats
