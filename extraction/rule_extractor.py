"""
Rule-based extraction baseline.

Uses strict ontology rules without LLM to extract structure.
"""

import re
import time
from typing import List, Tuple, Dict
from dataclasses import dataclass

import sys
from pathlib import Path

sys.path.append(str(Path(__file__).parent.parent))

from ontology.legal_ontology import Variable, Edge, OntologyType, LEGAL_ONTOLOGY
from utils.extraction_logger import get_extraction_logger


class RuleExtractor:
    """
    Rule-only extraction baseline.

    Uses strict definitional templates from the ontology
    without any LLM proposals. Tests ontology coverage.
    """

    # Strict definitional rules: (pattern, variable_type, variable_name_template)
    VARIABLE_RULES = [
        # Clause detection
        (
            r"(?:Section|Article|Clause)\s+(\d+(?:\.\d+)?)[:\s]+([^.]+)",
            OntologyType.CLAUSE,
            "Section_{0}",
        ),
        # Obligation detection (strict "shall" pattern)
        (
            r"(\w+)\s+shall\s+maintain\s+([^.]+)",
            OntologyType.OBLIGATION,
            "Obligation_Maintain_{1}",
        ),
        (
            r"(\w+)\s+shall\s+provide\s+([^.]+)",
            OntologyType.OBLIGATION,
            "Obligation_Provide_{1}",
        ),
        (
            r"(\w+)\s+shall\s+pay\s+([^.]+)",
            OntologyType.OBLIGATION,
            "Obligation_Pay_{1}",
        ),
        (
            r"(\w+)\s+shall\s+deliver\s+([^.]+)",
            OntologyType.OBLIGATION,
            "Obligation_Deliver_{1}",
        ),
        (
            r"(\w+)\s+shall\s+comply\s+with\s+([^.]+)",
            OntologyType.OBLIGATION,
            "Obligation_Comply_{1}",
        ),
        # ComplianceEvent detection
        (
            r"[Ff]ailure\s+to\s+(\w+)\s+([^.]+)",
            OntologyType.COMPLIANCE_EVENT,
            "Failure_{0}",
        ),
        (r"[Bb]reach\s+of\s+([^.]+)", OntologyType.COMPLIANCE_EVENT, "Breach_{0}"),
        (r"[Dd]efault\s+under\s+([^.]+)", OntologyType.COMPLIANCE_EVENT, "Default_{0}"),
        # Outcome detection
        (
            r"shall\s+constitute\s+(?:a\s+)?(?:material\s+)?([^.]+)",
            OntologyType.OUTCOME,
            "Outcome_{0}",
        ),
        (
            r"shall\s+be\s+deemed\s+(?:a\s+)?([^.]+)",
            OntologyType.OUTCOME,
            "Outcome_{0}",
        ),
        # Remedy detection
        (r"may\s+terminate\s+([^.]+)", OntologyType.REMEDY, "Remedy_Terminate"),
        (r"entitled\s+to\s+([^.]+)", OntologyType.REMEDY, "Remedy_{0}"),
        (r"right\s+to\s+cure\s+([^.]+)", OntologyType.REMEDY, "Remedy_Cure"),
        # Cost detection
        (
            r"(?:direct\s+)?damages?\s+(?:not\s+to\s+exceed|capped\s+at|limited\s+to)\s+([^.]+)",
            OntologyType.COST,
            "Cost_Damages",
        ),
        (r"liable\s+for\s+([^.]+)", OntologyType.COST, "Cost_Liability"),
        (r"indemnif(?:y|ication)\s+([^.]+)", OntologyType.COST, "Cost_Indemnification"),
    ]

    # Edge rules: (source_pattern, target_pattern, edge_type)
    EDGE_RULES = [
        # Clause -> Obligation: clause containing "shall"
        (
            r"(?:Section|Article)\s+\d+",
            r"\w+\s+shall\s+\w+",
            (OntologyType.CLAUSE, OntologyType.OBLIGATION),
        ),
        # Obligation -> ComplianceEvent: obligation failure
        (
            r"shall\s+(\w+)",
            r"[Ff]ailure\s+to\s+\w+",
            (OntologyType.OBLIGATION, OntologyType.COMPLIANCE_EVENT),
        ),
        # ComplianceEvent -> Outcome: definitional
        (
            r"[Ff]ailure|[Bb]reach|[Dd]efault",
            r"shall\s+constitute|shall\s+be\s+deemed",
            (OntologyType.COMPLIANCE_EVENT, OntologyType.OUTCOME),
        ),
        # Outcome -> Remedy
        (
            r"material\s+breach|[Dd]efault|[Tt]ermination\s+event",
            r"may\s+terminate|entitled\s+to",
            (OntologyType.OUTCOME, OntologyType.REMEDY),
        ),
        # Outcome -> Cost
        (
            r"material\s+breach|[Dd]efault",
            r"liable|damages|indemnif",
            (OntologyType.OUTCOME, OntologyType.COST),
        ),
    ]

    def __init__(self):
        self.compiled_var_rules = [
            (re.compile(p, re.IGNORECASE), vtype, template)
            for p, vtype, template in self.VARIABLE_RULES
        ]
        self.compiled_edge_rules = [
            (re.compile(sp, re.IGNORECASE), re.compile(tp, re.IGNORECASE), etypes)
            for sp, tp, etypes in self.EDGE_RULES
        ]

    def extract(self, contract_text: str) -> Tuple[List[Variable], List[Edge]]:
        """
        Extract using strict ontology rules only.

        Returns:
            (variables, edges) tuple
        """
        variables = []
        edges = []
        var_map = {}
        var_id = 0

        # Split into sentences for local context
        sentences = re.split(r"[.!?]\s+", contract_text)

        # Extract variables
        for pattern, var_type, name_template in self.compiled_var_rules:
            for match in pattern.finditer(contract_text):
                try:
                    groups = match.groups()

                    # Generate unique name
                    try:
                        var_name = (
                            name_template.format(*groups) if groups else name_template
                        )
                    except:
                        var_name = f"var_{var_id}"

                    # Clean name
                    var_name = re.sub(r"[^a-zA-Z0-9_]", "_", var_name)[:50]
                    var_name = f"{var_name}_{var_id}"
                    var_id += 1

                    var = Variable(
                        name=var_name,
                        var_type=var_type,
                        span_text=match.group()[:150],
                        span_start=match.start(),
                        span_end=match.end(),
                    )

                    variables.append(var)
                    var_map[var_name] = var

                except Exception as e:
                    continue

        # Extract edges based on co-occurrence in sentences
        for sentence in sentences:
            # Find all variables mentioned in this sentence
            sentence_vars = []
            for var in variables:
                # Check if variable's span overlaps with sentence
                if var.span_text[:30].lower() in sentence.lower():
                    sentence_vars.append(var)

            # Create edges between compatible types
            for i, src_var in enumerate(sentence_vars):
                for tgt_var in sentence_vars[i + 1 :]:
                    # Check if edge type is allowed
                    if LEGAL_ONTOLOGY.is_valid_edge_type(
                        src_var.var_type, tgt_var.var_type
                    ):
                        # Find orientation marker
                        markers = LEGAL_ONTOLOGY.find_orientation_markers(sentence)
                        marker = markers[0][0] if markers else ""

                        edge = Edge(
                            source=src_var,
                            target=tgt_var,
                            orientation_marker=marker,
                            span_evidence=sentence[:200],
                            is_deterministic=LEGAL_ONTOLOGY.is_deterministic_relation(
                                sentence
                            ),
                        )
                        edges.append(edge)

                    # Also check reverse direction
                    elif LEGAL_ONTOLOGY.is_valid_edge_type(
                        tgt_var.var_type, src_var.var_type
                    ):
                        markers = LEGAL_ONTOLOGY.find_orientation_markers(sentence)
                        marker = markers[0][0] if markers else ""

                        edge = Edge(
                            source=tgt_var,
                            target=src_var,
                            orientation_marker=marker,
                            span_evidence=sentence[:200],
                            is_deterministic=LEGAL_ONTOLOGY.is_deterministic_relation(
                                sentence
                            ),
                        )
                        edges.append(edge)

        # Log extraction for transparency
        try:
            extraction_logger = get_extraction_logger()
            extraction_logger.log_extraction(
                method="rule",
                contract_id=getattr(self, "_current_contract_id", "unknown"),
                input_text=contract_text[:1000],  # First 1000 chars
                variables=variables,
                edges=edges,
                extraction_time=0.0,  # Rule extraction is fast
            )
        except Exception as log_error:
            pass  # Logging should not break extraction

        return variables, edges

    def get_coverage_stats(self, contract_text: str) -> Dict:
        """Get rule coverage statistics."""
        stats = {
            "total_variables": 0,
            "total_edges": 0,
            "variables_by_type": {},
            "edges_by_type": {},
        }

        variables, edges = self.extract(contract_text)
        stats["total_variables"] = len(variables)
        stats["total_edges"] = len(edges)

        for var in variables:
            type_name = var.var_type.value
            stats["variables_by_type"][type_name] = (
                stats["variables_by_type"].get(type_name, 0) + 1
            )

        for edge in edges:
            edge_type = f"{edge.source.var_type.value}->{edge.target.var_type.value}"
            stats["edges_by_type"][edge_type] = (
                stats["edges_by_type"].get(edge_type, 0) + 1
            )

        return stats
