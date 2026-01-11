"""
Legal Contracts Ontology for OCCI Framework.

Defines the type hierarchy, allowed edges, and constraint rules.
"""

from dataclasses import dataclass, field
from typing import Dict, List, Set, Tuple, Optional
from enum import Enum
import re


class OntologyType(Enum):
    """Variable types in the legal contracts ontology."""

    CLAUSE = "Clause"
    OBLIGATION = "Obligation"
    COMPLIANCE_EVENT = "ComplianceEvent"
    OUTCOME = "Outcome"
    REMEDY = "Remedy"
    COST = "Cost"


@dataclass
class Variable:
    """A variable in the induced SCM."""

    name: str
    var_type: OntologyType
    span_text: str
    span_start: int = 0
    span_end: int = 0
    attributes: Dict = field(default_factory=dict)
    confidence: float = 1.0

    def __hash__(self):
        return hash((self.name, self.var_type.value))

    def __eq__(self, other):
        if not isinstance(other, Variable):
            return False
        return self.name == other.name and self.var_type == other.var_type


@dataclass
class Edge:
    """A directed causal edge in the induced SCM."""

    source: Variable
    target: Variable
    orientation_marker: str
    span_evidence: str
    confidence: float = 1.0
    is_deterministic: bool = False

    def __hash__(self):
        return hash((self.source.name, self.target.name))

    def __eq__(self, other):
        if not isinstance(other, Edge):
            return False
        return (
            self.source.name == other.source.name
            and self.target.name == other.target.name
        )

    def to_tuple(self) -> Tuple[str, str]:
        return (self.source.name, self.target.name)


class LegalContractsOntology:
    """
    Ontology for legal contracts domain.

    Defines:
    - Type classes T with partial order ≺
    - Allowed edge types R
    - Orientation patterns
    - Cardinality and scope constraints
    """

    # Partial order over types (lower number = earlier in causal chain)
    TYPE_ORDER: Dict[OntologyType, int] = {
        OntologyType.CLAUSE: 0,
        OntologyType.OBLIGATION: 1,
        OntologyType.COMPLIANCE_EVENT: 2,
        OntologyType.OUTCOME: 3,
        OntologyType.REMEDY: 4,
        OntologyType.COST: 4,  # Same level as Remedy (both follow Outcome)
    }

    # Allowed edge type pairs (source_type, target_type)
    ALLOWED_EDGES: Set[Tuple[OntologyType, OntologyType]] = {
        (OntologyType.CLAUSE, OntologyType.OBLIGATION),
        (OntologyType.OBLIGATION, OntologyType.COMPLIANCE_EVENT),
        (OntologyType.COMPLIANCE_EVENT, OntologyType.OUTCOME),
        (OntologyType.OUTCOME, OntologyType.REMEDY),
        (OntologyType.OUTCOME, OntologyType.COST),
    }

    # Orientation patterns that indicate causal direction
    ORIENTATION_PATTERNS: List[Tuple[str, str]] = [
        # (pattern, direction_hint)
        (r"\bshall\s+constitute\b", "forward"),
        (r"\bshall\s+be\s+deemed\b", "forward"),
        (r"\bshall\s+be\s+considered\b", "forward"),
        (r"\bresults?\s+in\b", "forward"),
        (r"\bleads?\s+to\b", "forward"),
        (r"\btriggers?\b", "forward"),
        (r"\bcauses?\b", "forward"),
        (r"\bupon\b", "forward"),
        (r"\bif\s+", "forward"),
        (r"\bprovided\s+that\b", "forward"),
        (r"\bsubject\s+to\b", "forward"),
        (r"\bin\s+the\s+event\s+of\b", "forward"),
        (r"\bfailure\s+to\b", "forward"),
        (r"\bbreach\s+of\b", "forward"),
        (r"\bmay\s+terminate\b", "forward"),
        (r"\bentitled\s+to\b", "forward"),
        (r"\bliable\s+for\b", "forward"),
        (r"\bresponsible\s+for\b", "forward"),
        (r"\bshall\b", "obligation"),
        (r"\bmust\b", "obligation"),
        (r"\bwill\b", "obligation"),
        (r"\bagrees?\s+to\b", "obligation"),
    ]

    # Deterministic relation patterns
    DETERMINISTIC_PATTERNS: List[str] = [
        r"\bshall\s+constitute\b",
        r"\bshall\s+be\s+deemed\b",
        r"\bis\s+defined\s+as\b",
        r"\bmeans\b",
    ]

    def __init__(self):
        self.compiled_patterns = [
            (re.compile(p, re.IGNORECASE), hint)
            for p, hint in self.ORIENTATION_PATTERNS
        ]
        self.deterministic_patterns = [
            re.compile(p, re.IGNORECASE) for p in self.DETERMINISTIC_PATTERNS
        ]

    def get_type_order(self, var_type: OntologyType) -> int:
        """Get the order value for a type in the partial order."""
        return self.TYPE_ORDER.get(var_type, 999)

    def is_valid_edge_type(
        self, source_type: OntologyType, target_type: OntologyType
    ) -> bool:
        """Check if edge type pair is allowed."""
        return (source_type, target_type) in self.ALLOWED_EDGES

    def respects_partial_order(
        self, source_type: OntologyType, target_type: OntologyType
    ) -> bool:
        """Check if edge respects the partial order (source ≺ target)."""
        return self.get_type_order(source_type) < self.get_type_order(target_type)

    def find_orientation_markers(self, text: str) -> List[Tuple[str, str, int, int]]:
        """
        Find orientation markers in text.

        Returns:
            List of (marker_text, direction_hint, start, end)
        """
        markers = []
        for pattern, hint in self.compiled_patterns:
            for match in pattern.finditer(text):
                markers.append((match.group(), hint, match.start(), match.end()))
        return markers

    def is_deterministic_relation(self, text: str) -> bool:
        """Check if text indicates a deterministic (definitional) relation."""
        return any(p.search(text) for p in self.deterministic_patterns)

    def validate_variable(self, var: Variable) -> Tuple[bool, str]:
        """
        Validate a variable against ontology constraints.

        Returns:
            (is_valid, reason)
        """
        # Check type is valid
        if var.var_type not in OntologyType:
            return False, f"Invalid type: {var.var_type}"

        # Check span evidence exists
        if not var.span_text or len(var.span_text.strip()) < 3:
            return False, "Insufficient span evidence"

        # Check name is not empty
        if not var.name or len(var.name.strip()) < 1:
            return False, "Empty variable name"

        return True, "Valid"

    def validate_edge(self, edge: Edge) -> Tuple[bool, str]:
        """
        Validate an edge against ontology constraints.

        Returns:
            (is_valid, reason)
        """
        # Check type pair is allowed
        if not self.is_valid_edge_type(edge.source.var_type, edge.target.var_type):
            return (
                False,
                f"Invalid edge type pair: {edge.source.var_type.value} -> {edge.target.var_type.value}",
            )

        # Check partial order
        if not self.respects_partial_order(edge.source.var_type, edge.target.var_type):
            return (
                False,
                f"Violates partial order: {edge.source.var_type.value} must precede {edge.target.var_type.value}",
            )

        # Check span evidence
        if not edge.span_evidence or len(edge.span_evidence.strip()) < 5:
            return False, "Insufficient span evidence for edge"

        # Check orientation marker
        if not edge.orientation_marker:
            # Try to find marker in evidence
            markers = self.find_orientation_markers(edge.span_evidence)
            if not markers:
                return False, "No orientation marker found"

        return True, "Valid"

    def string_to_type(self, type_str: str) -> Optional[OntologyType]:
        """Convert string to OntologyType."""
        type_map = {
            "clause": OntologyType.CLAUSE,
            "obligation": OntologyType.OBLIGATION,
            "complianceevent": OntologyType.COMPLIANCE_EVENT,
            "compliance_event": OntologyType.COMPLIANCE_EVENT,
            "outcome": OntologyType.OUTCOME,
            "remedy": OntologyType.REMEDY,
            "cost": OntologyType.COST,
        }
        return type_map.get(type_str.lower().replace(" ", ""))


# Singleton instance
LEGAL_ONTOLOGY = LegalContractsOntology()
