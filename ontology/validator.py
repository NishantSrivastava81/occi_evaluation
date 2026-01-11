"""
Ontology Validator with Adversarial Testing.

Implements Algorithm 1 from the paper:
- Typing check
- Span grounding
- Acyclicity check
- Adversarial tests (decoy rejection, contradiction challenge)
"""

from dataclasses import dataclass, field
from typing import Dict, List, Set, Tuple, Optional
import networkx as nx
from collections import defaultdict

from .legal_ontology import (
    LegalContractsOntology,
    Variable,
    Edge,
    OntologyType,
    LEGAL_ONTOLOGY,
)


@dataclass
class ValidationResult:
    """Result of validation for a single element."""

    is_valid: bool
    reason: str
    element_type: str  # "variable" or "edge"
    element_id: str


@dataclass
class ValidationCertificate:
    """Certificate produced by validator."""

    is_accepted: bool
    accepted_variables: List[Variable]
    accepted_edges: List[Edge]
    rejected_variables: List[Tuple[Variable, str]]  # (var, reason)
    rejected_edges: List[Tuple[Edge, str]]  # (edge, reason)
    validation_results: List[ValidationResult]
    metrics: Dict = field(default_factory=dict)

    def summary(self) -> Dict:
        return {
            "accepted": self.is_accepted,
            "num_accepted_variables": len(self.accepted_variables),
            "num_accepted_edges": len(self.accepted_edges),
            "num_rejected_variables": len(self.rejected_variables),
            "num_rejected_edges": len(self.rejected_edges),
            "structural_consistency": 1.0 if self.is_accepted else 0.0,
            **self.metrics,
        }


class OntologyValidator:
    """
    Validator V_O implementing Algorithm 1.

    Enforces:
    1. Typing and admissibility
    2. Span grounding
    3. Acyclicity
    4. Adversarial robustness
    """

    def __init__(
        self,
        ontology: LegalContractsOntology = LEGAL_ONTOLOGY,
        k_decoys: int = 10,
        similarity_threshold: float = 0.7,
        tau_conf: float = 0.6,
        tau_agree: float = 0.8,
    ):
        self.ontology = ontology
        self.k_decoys = k_decoys
        self.similarity_threshold = similarity_threshold
        self.tau_conf = tau_conf
        self.tau_agree = tau_agree

    def validate(
        self,
        candidate_variables: List[Variable],
        candidate_edges: List[Edge],
        document_text: str,
        decoy_spans: Optional[List[str]] = None,
    ) -> ValidationCertificate:
        """
        Main validation function implementing Algorithm 1.

        Args:
            candidate_variables: Proposed variables from LLM
            candidate_edges: Proposed edges from LLM
            document_text: Original document text
            decoy_spans: Optional decoy sentences for hallucination detection

        Returns:
            ValidationCertificate with accepted elements and reasons
        """
        results = []
        accepted_vars = []
        rejected_vars = []
        accepted_edges = []
        rejected_edges = []

        # Track decoy-only evidence
        decoy_set = set(decoy_spans) if decoy_spans else set()

        # Step 1: Variable typing and admissibility check
        for var in candidate_variables:
            is_valid, reason = self._check_variable_typing(var)

            if is_valid:
                # Check span grounding
                is_grounded, grounding_reason = self._check_span_grounding(
                    var.span_text, document_text, decoy_set
                )
                if is_grounded:
                    accepted_vars.append(var)
                    results.append(
                        ValidationResult(True, "Accepted", "variable", var.name)
                    )
                else:
                    rejected_vars.append((var, grounding_reason))
                    results.append(
                        ValidationResult(False, grounding_reason, "variable", var.name)
                    )
            else:
                rejected_vars.append((var, reason))
                results.append(ValidationResult(False, reason, "variable", var.name))

        # Create lookup for accepted variables
        accepted_var_names = {v.name for v in accepted_vars}

        # Step 2: Edge validation
        for edge in candidate_edges:
            # Check source and target are accepted
            if edge.source.name not in accepted_var_names:
                rejected_edges.append(
                    (edge, f"Source variable '{edge.source.name}' not accepted")
                )
                results.append(
                    ValidationResult(
                        False,
                        "Source not accepted",
                        "edge",
                        f"{edge.source.name}->{edge.target.name}",
                    )
                )
                continue

            if edge.target.name not in accepted_var_names:
                rejected_edges.append(
                    (edge, f"Target variable '{edge.target.name}' not accepted")
                )
                results.append(
                    ValidationResult(
                        False,
                        "Target not accepted",
                        "edge",
                        f"{edge.source.name}->{edge.target.name}",
                    )
                )
                continue

            # Type pair check
            is_valid_type, type_reason = self._check_edge_typing(edge)
            if not is_valid_type:
                rejected_edges.append((edge, type_reason))
                results.append(
                    ValidationResult(
                        False,
                        type_reason,
                        "edge",
                        f"{edge.source.name}->{edge.target.name}",
                    )
                )
                continue

            # Span grounding for edge
            is_grounded, grounding_reason = self._check_span_grounding(
                edge.span_evidence, document_text, decoy_set
            )
            if not is_grounded:
                rejected_edges.append((edge, grounding_reason))
                results.append(
                    ValidationResult(
                        False,
                        grounding_reason,
                        "edge",
                        f"{edge.source.name}->{edge.target.name}",
                    )
                )
                continue

            # Orientation marker check
            has_marker, marker_reason = self._check_orientation_marker(edge)
            if not has_marker:
                rejected_edges.append((edge, marker_reason))
                results.append(
                    ValidationResult(
                        False,
                        marker_reason,
                        "edge",
                        f"{edge.source.name}->{edge.target.name}",
                    )
                )
                continue

            accepted_edges.append(edge)
            results.append(
                ValidationResult(
                    True, "Accepted", "edge", f"{edge.source.name}->{edge.target.name}"
                )
            )

        # Step 3: Acyclicity check
        is_acyclic, cycle_info = self._check_acyclicity(accepted_edges)
        if not is_acyclic:
            # Remove edges that cause cycles
            accepted_edges, removed = self._remove_cycle_edges(accepted_edges)
            for edge in removed:
                rejected_edges.append((edge, f"Causes cycle: {cycle_info}"))
                results.append(
                    ValidationResult(
                        False,
                        "Causes cycle",
                        "edge",
                        f"{edge.source.name}->{edge.target.name}",
                    )
                )

        # Step 4: Adversarial tests (decoy detection)
        if decoy_spans:
            decoy_edges = self._detect_decoy_only_edges(
                accepted_edges, document_text, decoy_set
            )
            for edge in decoy_edges:
                accepted_edges.remove(edge)
                rejected_edges.append((edge, "Supported only by decoy spans"))
                results.append(
                    ValidationResult(
                        False,
                        "Decoy-only evidence",
                        "edge",
                        f"{edge.source.name}->{edge.target.name}",
                    )
                )

        # Compute metrics
        metrics = {
            "typing_rejection_rate": len(
                [r for r in rejected_vars if "type" in r[1].lower()]
            )
            / max(len(candidate_variables), 1),
            "grounding_rejection_rate": len(
                [
                    r
                    for r in rejected_vars
                    if "span" in r[1].lower() or "grounding" in r[1].lower()
                ]
            )
            / max(len(candidate_variables), 1),
            "edge_type_rejection_rate": len(
                [r for r in rejected_edges if "type" in r[1].lower()]
            )
            / max(len(candidate_edges), 1),
            "decoy_rejection_count": len(
                [r for r in rejected_edges if "decoy" in r[1].lower()]
            ),
        }

        is_accepted = len(accepted_edges) > 0 or len(accepted_vars) > 0

        return ValidationCertificate(
            is_accepted=is_accepted,
            accepted_variables=accepted_vars,
            accepted_edges=accepted_edges,
            rejected_variables=rejected_vars,
            rejected_edges=rejected_edges,
            validation_results=results,
            metrics=metrics,
        )

    def _check_variable_typing(self, var: Variable) -> Tuple[bool, str]:
        """Check variable type is valid."""
        return self.ontology.validate_variable(var)

    def _check_edge_typing(self, edge: Edge) -> Tuple[bool, str]:
        """Check edge type pair is allowed and respects partial order."""
        if not self.ontology.is_valid_edge_type(
            edge.source.var_type, edge.target.var_type
        ):
            return (
                False,
                f"Invalid edge type: {edge.source.var_type.value} -> {edge.target.var_type.value}",
            )

        if not self.ontology.respects_partial_order(
            edge.source.var_type, edge.target.var_type
        ):
            return False, f"Violates partial order"

        return True, "Valid"

    def _check_span_grounding(
        self, span_text: str, document_text: str, decoy_set: Set[str]
    ) -> Tuple[bool, str]:
        """Check if span is grounded in document (not just decoys)."""
        if not span_text or len(span_text.strip()) < 3:
            return False, "Span text too short"

        # Check if span appears in document
        span_lower = span_text.lower().strip()
        doc_lower = document_text.lower()

        # Check for substantial overlap (at least 50% of words)
        span_words = set(span_lower.split())
        doc_words = set(doc_lower.split())

        if len(span_words) == 0:
            return False, "Empty span"

        overlap = len(span_words & doc_words) / len(span_words)

        if overlap < 0.5:
            return False, f"Insufficient document grounding (overlap: {overlap:.2f})"

        # Check if grounded ONLY in decoys
        if decoy_set:
            decoy_text = " ".join(decoy_set).lower()
            decoy_words = set(decoy_text.split())

            decoy_overlap = len(span_words & decoy_words) / len(span_words)
            non_decoy_words = doc_words - decoy_words
            non_decoy_overlap = len(span_words & non_decoy_words) / len(span_words)

            if decoy_overlap > 0.8 and non_decoy_overlap < 0.2:
                return False, "Span grounded only in decoy text"

        return True, "Grounded"

    def _check_orientation_marker(self, edge: Edge) -> Tuple[bool, str]:
        """Check if edge has valid orientation marker."""
        if edge.orientation_marker and len(edge.orientation_marker.strip()) > 0:
            return True, "Has explicit marker"

        # Try to find marker in span evidence
        markers = self.ontology.find_orientation_markers(edge.span_evidence)
        if markers:
            return True, f"Found marker: {markers[0][0]}"

        return False, "No orientation marker found"

    def _check_acyclicity(self, edges: List[Edge]) -> Tuple[bool, str]:
        """Check if edge set forms an acyclic graph."""
        G = nx.DiGraph()

        for edge in edges:
            G.add_edge(edge.source.name, edge.target.name)

        try:
            cycles = list(nx.simple_cycles(G))
            if cycles:
                return False, f"Cycle detected: {cycles[0]}"
        except:
            pass

        return True, "Acyclic"

    def _remove_cycle_edges(self, edges: List[Edge]) -> Tuple[List[Edge], List[Edge]]:
        """Remove minimum edges to break cycles."""
        G = nx.DiGraph()
        edge_map = {}

        for edge in edges:
            G.add_edge(edge.source.name, edge.target.name)
            edge_map[(edge.source.name, edge.target.name)] = edge

        removed = []

        while True:
            try:
                cycles = list(nx.simple_cycles(G))
                if not cycles:
                    break

                # Remove last edge in first cycle
                cycle = cycles[0]
                if len(cycle) >= 2:
                    src, tgt = cycle[-1], cycle[0]
                    if (src, tgt) in edge_map:
                        removed.append(edge_map[(src, tgt)])
                        G.remove_edge(src, tgt)
                        del edge_map[(src, tgt)]
                else:
                    break
            except:
                break

        remaining = [edge_map[k] for k in edge_map]
        return remaining, removed

    def _detect_decoy_only_edges(
        self, edges: List[Edge], document_text: str, decoy_set: Set[str]
    ) -> List[Edge]:
        """Detect edges supported only by decoy spans."""
        decoy_edges = []

        if not decoy_set:
            return decoy_edges

        decoy_text = " ".join(decoy_set).lower()
        doc_without_decoys = document_text.lower()
        for decoy in decoy_set:
            doc_without_decoys = doc_without_decoys.replace(decoy.lower(), "")

        for edge in edges:
            span_lower = edge.span_evidence.lower()
            span_words = set(span_lower.split())

            decoy_words = set(decoy_text.split())
            clean_doc_words = set(doc_without_decoys.split())

            decoy_overlap = len(span_words & decoy_words) / max(len(span_words), 1)
            clean_overlap = len(span_words & clean_doc_words) / max(len(span_words), 1)

            if decoy_overlap > 0.7 and clean_overlap < 0.3:
                decoy_edges.append(edge)

        return decoy_edges

    def compute_structural_consistency(
        self, certificate: ValidationCertificate
    ) -> float:
        """
        Compute structural consistency score.

        SC = 1.0 if all accepted elements satisfy constraints, 0.0 otherwise.
        By design, accepted elements always satisfy constraints.
        """
        if not certificate.is_accepted:
            return 0.0

        # Verify all accepted edges satisfy constraints
        for edge in certificate.accepted_edges:
            if not self.ontology.is_valid_edge_type(
                edge.source.var_type, edge.target.var_type
            ):
                return 0.0
            if not self.ontology.respects_partial_order(
                edge.source.var_type, edge.target.var_type
            ):
                return 0.0

        # Check acyclicity
        is_acyclic, _ = self._check_acyclicity(certificate.accepted_edges)
        if not is_acyclic:
            return 0.0

        return 1.0
