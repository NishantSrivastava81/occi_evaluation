"""
Evaluation metrics for OCCI framework.

Implements:
- Structural Correctness (SC): Valid edges / total edges
- Cycle Score (CS): 1 - (graphs with cycles / total graphs)
- False Positive Rate (FPR): Decoy edges / decoy opportunities
- True Positive Rate (TPR): Correctly identified causal edges
- False-Positive Causal Rate (FPCR): Spurious non-deterministic edges
- Decoy Rejection Rate (DRR): Successfully rejected decoy sentences
- Invariance Accuracy (IA): Paraphrase consistency
- Inversion Rate (IR): Contradiction detection
"""

import json
from typing import List, Dict, Tuple, Optional
from dataclasses import dataclass, asdict
from collections import defaultdict
import networkx as nx

import sys
from pathlib import Path

sys.path.append(str(Path(__file__).parent.parent))

from ontology.legal_ontology import Variable, Edge, LEGAL_ONTOLOGY, OntologyType
from ontology.validator import (
    OntologyValidator,
    ValidationResult,
    ValidationCertificate,
)


@dataclass
class ExtractionResult:
    """Result from an extraction method."""

    method_name: str
    contract_id: str
    variables: List[Variable]
    edges: List[Edge]
    extraction_time: float
    validation_result: Optional[ValidationCertificate] = None


@dataclass
class MetricResult:
    """Result for a single metric."""

    name: str
    value: float
    numerator: int
    denominator: int
    details: Dict = None


@dataclass
class EvaluationSummary:
    """Summary of all metrics for a method."""

    method_name: str
    n_contracts: int
    metrics: Dict[str, MetricResult]
    per_contract_results: List[Dict] = None


class OCCIMetrics:
    """
    Compute all evaluation metrics for the OCCI framework.
    """

    def __init__(self, validator: OntologyValidator = None):
        self.validator = validator or OntologyValidator(LEGAL_ONTOLOGY)

    def compute_structural_correctness(
        self, edges: List[Edge], validation_result: ValidationResult = None
    ) -> MetricResult:
        """
        Compute Structural Correctness (SC).

        SC = (edges respecting partial order AND typing) / total edges
        """
        if not edges:
            return MetricResult(
                name="structural_correctness", value=1.0, numerator=0, denominator=0
            )

        valid_edges = 0
        invalid_details = []

        for edge in edges:
            # Check type validity
            type_valid = LEGAL_ONTOLOGY.is_valid_edge_type(
                edge.source.var_type, edge.target.var_type
            )

            # Check partial order
            order_valid = LEGAL_ONTOLOGY.respects_partial_order(
                edge.source.var_type, edge.target.var_type
            )

            if type_valid and order_valid:
                valid_edges += 1
            else:
                invalid_details.append(
                    {
                        "source_type": edge.source.var_type.value,
                        "target_type": edge.target.var_type.value,
                        "type_valid": type_valid,
                        "order_valid": order_valid,
                    }
                )

        sc = valid_edges / len(edges)

        return MetricResult(
            name="structural_correctness",
            value=sc,
            numerator=valid_edges,
            denominator=len(edges),
            details={"invalid_edges": invalid_details[:10]},  # Keep first 10
        )

    def compute_cycle_score(
        self, extraction_results: List[ExtractionResult]
    ) -> MetricResult:
        """
        Compute Cycle Score (CS).

        CS = 1 - (number of graphs with cycles / total graphs)
        """
        if not extraction_results:
            return MetricResult(
                name="cycle_score", value=1.0, numerator=0, denominator=0
            )

        acyclic_count = 0
        cyclic_contracts = []

        for result in extraction_results:
            # Build graph
            G = nx.DiGraph()

            for edge in result.edges:
                src_id = f"{edge.source.name}_{edge.source.var_type.value}"
                tgt_id = f"{edge.target.name}_{edge.target.var_type.value}"
                G.add_edge(src_id, tgt_id)

            # Check acyclicity
            if nx.is_directed_acyclic_graph(G):
                acyclic_count += 1
            else:
                cycles = list(nx.simple_cycles(G))
                cyclic_contracts.append(
                    {
                        "contract_id": result.contract_id,
                        "n_cycles": len(cycles),
                        "first_cycle": cycles[0] if cycles else None,
                    }
                )

        cs = acyclic_count / len(extraction_results)

        return MetricResult(
            name="cycle_score",
            value=cs,
            numerator=acyclic_count,
            denominator=len(extraction_results),
            details={"cyclic_contracts": cyclic_contracts[:5]},
        )

    def compute_false_positive_rate(
        self, accepted_decoy_edges: int, total_decoy_edges: int
    ) -> MetricResult:
        """
        Compute False Positive Rate (FPR) for decoy detection.

        FPR = accepted_decoy_edges / total_decoy_edges
        
        This measures hallucination rate: what fraction of decoy-induced
        edges were incorrectly accepted by the system.
        """
        if total_decoy_edges == 0:
            return MetricResult(
                name="false_positive_rate", value=0.0, numerator=0, denominator=0
            )

        fpr = accepted_decoy_edges / total_decoy_edges

        return MetricResult(
            name="false_positive_rate",
            value=fpr,
            numerator=accepted_decoy_edges,
            denominator=total_decoy_edges,
        )

    def compute_true_positive_rate(
        self, extracted_edges: List[Edge], ground_truth_edges: List[Tuple[str, str]]
    ) -> MetricResult:
        """
        Compute True Positive Rate (TPR) / Recall.

        TPR = correctly identified causal edges / ground truth edges

        Ground truth is based on clause type mappings from CUAD.
        """
        if not ground_truth_edges:
            return MetricResult(
                name="true_positive_rate", value=0.0, numerator=0, denominator=0
            )

        # Create signature set from extracted edges
        extracted_sigs = set()
        for edge in extracted_edges:
            sig = (edge.source.var_type.value, edge.target.var_type.value)
            extracted_sigs.add(sig)

        # Create signature set from ground truth edges
        gt_sigs = set()
        for gt_edge in ground_truth_edges:
            if isinstance(gt_edge, dict):
                # Ground truth from create_ground_truth_edges() returns dicts
                sig = (gt_edge.get("source_type", ""), gt_edge.get("target_type", ""))
            else:
                # Already a tuple
                sig = gt_edge
            gt_sigs.add(sig)

        # Count matches
        matches = len(extracted_sigs.intersection(gt_sigs))

        tpr = matches / len(gt_sigs) if gt_sigs else 0.0

        return MetricResult(
            name="true_positive_rate",
            value=tpr,
            numerator=matches,
            denominator=len(gt_sigs),
        )

    def compute_false_positive_causal_rate(
        self, edges: List[Edge], deterministic_evidence: List[str]
    ) -> MetricResult:
        """
        Compute False-Positive Causal Rate (FPCR).

        FPCR = spurious non-deterministic edges / total edges

        An edge is spuriously marked deterministic if the evidence
        doesn't contain strong causal language.
        """
        if not edges:
            return MetricResult(
                name="false_positive_causal_rate", value=0.0, numerator=0, denominator=0
            )

        spurious_count = 0

        for edge in edges:
            if edge.is_deterministic:
                # Check if evidence supports determinism
                evidence_text = edge.span_evidence.lower()
                has_strong_causation = any(
                    marker in evidence_text
                    for marker in [
                        "shall constitute",
                        "shall be deemed",
                        "results in",
                        "causes",
                    ]
                )

                if not has_strong_causation:
                    spurious_count += 1

        fpcr = spurious_count / len(edges)

        return MetricResult(
            name="false_positive_causal_rate",
            value=fpcr,
            numerator=spurious_count,
            denominator=len(edges),
        )

    def compute_decoy_rejection_rate(
        self, rejected_decoy_edges: int, accepted_decoy_edges: int
    ) -> MetricResult:
        """
        Compute Decoy Rejection Rate (DRR) as per paper Section 8.2.

        DRR = (R_decoy - A_decoy) / (R_decoy + A_decoy)
        
        Where:
        - R_decoy = number of decoy-induced edges rejected by the validator
        - A_decoy = number of decoy-induced edges accepted (hallucinations)
        
        DRR ∈ [-1, 1]:
        - DRR = 1.0: Perfect rejection (all decoy edges rejected)
        - DRR = 0.0: Equal rejection and acceptance
        - DRR < 0: System accepts more decoy edges than it rejects (bad)
        """
        total = rejected_decoy_edges + accepted_decoy_edges
        
        if total == 0:
            # No decoy-induced edges at all - perfect performance
            return MetricResult(
                name="decoy_rejection_rate", value=1.0, numerator=0, denominator=0
            )

        drr = (rejected_decoy_edges - accepted_decoy_edges) / total

        return MetricResult(
            name="decoy_rejection_rate",
            value=drr,
            numerator=rejected_decoy_edges,
            denominator=total,
            details={
                "rejected_decoy_edges": rejected_decoy_edges,
                "accepted_decoy_edges": accepted_decoy_edges,
                "interpretation": "positive=good, negative=hallucinating"
            }
        )

    def compute_invariance_accuracy(
        self,
        original_results: List[ExtractionResult],
        paraphrase_results: List[ExtractionResult],
    ) -> MetricResult:
        """
        Compute Invariance Accuracy (IA).

        IA = pairs with same structure / total pairs
        """
        if len(original_results) != len(paraphrase_results):
            raise ValueError("Must have matching original and paraphrase results")

        if not original_results:
            return MetricResult(
                name="invariance_accuracy", value=1.0, numerator=0, denominator=0
            )

        matches = 0

        for orig, para in zip(original_results, paraphrase_results):
            # Compare edge type signatures
            orig_sigs = set(
                (e.source.var_type.value, e.target.var_type.value) for e in orig.edges
            )
            para_sigs = set(
                (e.source.var_type.value, e.target.var_type.value) for e in para.edges
            )

            # Check if structures match (allow for minor differences)
            common = orig_sigs & para_sigs
            total = orig_sigs | para_sigs

            if not total or len(common) / len(total) >= 0.8:
                matches += 1

        ia = matches / len(original_results)

        return MetricResult(
            name="invariance_accuracy",
            value=ia,
            numerator=matches,
            denominator=len(original_results),
        )

    def compute_inversion_rate(
        self,
        original_results: List[ExtractionResult],
        contradiction_results: List[ExtractionResult],
    ) -> MetricResult:
        """
        Compute Inversion Rate (IR).

        IR = pairs where contradiction changed structure / total pairs
        """
        if len(original_results) != len(contradiction_results):
            raise ValueError("Must have matching original and contradiction results")

        if not original_results:
            return MetricResult(
                name="inversion_rate", value=0.0, numerator=0, denominator=0
            )

        inversions = 0

        for orig, contra in zip(original_results, contradiction_results):
            orig_sigs = set(
                (e.source.var_type.value, e.target.var_type.value) for e in orig.edges
            )
            contra_sigs = set(
                (e.source.var_type.value, e.target.var_type.value) for e in contra.edges
            )

            # Contradiction should produce different structure
            if orig_sigs != contra_sigs:
                inversions += 1

        ir = inversions / len(original_results)

        return MetricResult(
            name="inversion_rate",
            value=ir,
            numerator=inversions,
            denominator=len(original_results),
        )

    def compute_all_metrics(
        self,
        extraction_results: List[ExtractionResult],
        rejected_decoy_edges: int = 0,
        accepted_decoy_edges: int = 0,
        ground_truth_edges: List[Tuple[str, str]] = None,
        paraphrase_original: List[ExtractionResult] = None,
        paraphrase_results: List[ExtractionResult] = None,
        contradiction_original: List[ExtractionResult] = None,
        contradiction_results: List[ExtractionResult] = None,
    ) -> EvaluationSummary:
        """
        Compute all metrics for a method.
        """
        if not extraction_results:
            return EvaluationSummary(method_name="unknown", n_contracts=0, metrics={})

        method_name = extraction_results[0].method_name

        # Aggregate all edges
        all_edges = []
        for result in extraction_results:
            all_edges.extend(result.edges)

        metrics = {}

        # Core metrics
        sc_result = self.compute_structural_correctness(all_edges)
        metrics[sc_result.name] = sc_result

        cs_result = self.compute_cycle_score(extraction_results)
        metrics[cs_result.name] = cs_result

        # Adversarial metrics (decoy rejection as per paper Section 8.2)
        total_decoy_related = rejected_decoy_edges + accepted_decoy_edges
        if total_decoy_related > 0:
            # FPR based on accepted (hallucinated) decoy edges
            fpr_result = self.compute_false_positive_rate(
                accepted_decoy_edges, total_decoy_related
            )
            metrics[fpr_result.name] = fpr_result

            # DRR = (R_decoy - A_decoy) / (R_decoy + A_decoy) as per paper
            drr_result = self.compute_decoy_rejection_rate(
                rejected_decoy_edges, accepted_decoy_edges
            )
            metrics[drr_result.name] = drr_result

        # Ground truth metrics
        if ground_truth_edges:
            tpr_result = self.compute_true_positive_rate(all_edges, ground_truth_edges)
            metrics[tpr_result.name] = tpr_result

        # Causal precision
        fpcr_result = self.compute_false_positive_causal_rate(all_edges, [])
        metrics[fpcr_result.name] = fpcr_result

        # Paraphrase invariance - use paraphrase_original for comparison
        if paraphrase_original and paraphrase_results:
            ia_result = self.compute_invariance_accuracy(
                paraphrase_original, paraphrase_results
            )
            metrics[ia_result.name] = ia_result

        # Contradiction detection - use contradiction_original for comparison
        if contradiction_original and contradiction_results:
            ir_result = self.compute_inversion_rate(
                contradiction_original, contradiction_results
            )
            metrics[ir_result.name] = ir_result

        # Per-contract results
        per_contract = []
        for result in extraction_results:
            pc = {
                "contract_id": result.contract_id,
                "n_variables": len(result.variables),
                "n_edges": len(result.edges),
                "extraction_time": result.extraction_time,
            }
            if result.validation_result:
                # ValidationCertificate uses is_accepted, not is_valid
                pc["validation_passed"] = result.validation_result.is_accepted
                # Count errors from validation_results (List[ValidationResult])
                n_errors = len(
                    [
                        r
                        for r in result.validation_result.validation_results
                        if not r.is_valid
                    ]
                )
                pc["n_errors"] = n_errors
            per_contract.append(pc)

        return EvaluationSummary(
            method_name=method_name,
            n_contracts=len(extraction_results),
            metrics=metrics,
            per_contract_results=per_contract,
        )

    def format_results(self, summary: EvaluationSummary) -> str:
        """Format results for display."""
        lines = [
            f"\n{'='*60}",
            f"Evaluation Results: {summary.method_name}",
            f"{'='*60}",
            f"Contracts evaluated: {summary.n_contracts}",
            f"\nMetrics:",
            f"{'-'*40}",
        ]

        for name, result in summary.metrics.items():
            lines.append(
                f"  {name}: {result.value:.4f} "
                f"({result.numerator}/{result.denominator})"
            )

        lines.append(f"{'='*60}\n")

        return "\n".join(lines)

    def to_dict(self, summary: EvaluationSummary) -> Dict:
        """Convert summary to dictionary for JSON export."""
        return {
            "method_name": summary.method_name,
            "n_contracts": summary.n_contracts,
            "metrics": {
                name: {
                    "value": r.value,
                    "numerator": r.numerator,
                    "denominator": r.denominator,
                }
                for name, r in summary.metrics.items()
            },
        }
