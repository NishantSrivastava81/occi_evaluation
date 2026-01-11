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
        self,
        edges: List[Edge],
        validation_result: ValidationResult = None,
        use_accepted_only: bool = True,
    ) -> MetricResult:
        """
        Compute Structural Correctness (SC).

        Per IEEE paper Section 7.2:
        - For OCCI: SC = 1.0 for accepted graphs BY CONSTRUCTION
          (validator ensures all accepted edges satisfy constraints)
        - For baselines: SC = (valid edges) / (total edges)

        Args:
            edges: List of edges to evaluate
            validation_result: Optional validation certificate
            use_accepted_only: If True and validation_result provided,
                              compute SC only on accepted edges (should be 1.0)
        """
        # If validation result provided and we should use accepted only,
        # SC is 1.0 by construction for OCCI method
        if use_accepted_only and validation_result is not None:
            # For accepted graphs, SC = 1.0 by design (paper claim)
            # Validator already ensured all accepted edges are valid
            accepted_edges = getattr(validation_result, "accepted_edges", edges)
            if len(accepted_edges) == 0 and len(edges) == 0:
                return MetricResult(
                    name="structural_correctness", value=1.0, numerator=0, denominator=0
                )
            # Verify the guarantee holds
            all_valid = True
            for edge in accepted_edges:
                type_valid = LEGAL_ONTOLOGY.is_valid_edge_type(
                    edge.source.var_type, edge.target.var_type
                )
                order_valid = LEGAL_ONTOLOGY.respects_partial_order(
                    edge.source.var_type, edge.target.var_type
                )
                if not (type_valid and order_valid):
                    all_valid = False
                    break
            if all_valid:
                return MetricResult(
                    name="structural_correctness",
                    value=1.0,
                    numerator=len(accepted_edges),
                    denominator=len(accepted_edges),
                    details={"note": "SC=1.0 by construction for accepted graphs"},
                )

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

        FPR = min(1.0, accepted_decoy_edges / total_decoy_edges)

        This measures hallucination rate: what fraction of decoy-induced
        edges were incorrectly accepted by the system.

        Note: FPR is capped at 1.0. Values >1 can occur when multiple edges
        are generated from the same decoy text, but this still indicates
        complete failure to reject decoys.
        """
        if total_decoy_edges == 0:
            return MetricResult(
                name="false_positive_rate", value=0.0, numerator=0, denominator=0
            )

        # Cap at 1.0 - more than 100% acceptance is still 100% failure
        fpr = min(1.0, accepted_decoy_edges / total_decoy_edges)

        return MetricResult(
            name="false_positive_rate",
            value=fpr,
            numerator=accepted_decoy_edges,
            denominator=total_decoy_edges,
            details=(
                {"raw_ratio": accepted_decoy_edges / total_decoy_edges}
                if accepted_decoy_edges > total_decoy_edges
                else None
            ),
        )

    def compute_true_positive_rate(
        self,
        extracted_edges: List[Edge],
        ground_truth_edges: List[Tuple[str, str]],
        use_semantic_matching: bool = False,
    ) -> MetricResult:
        """
        Compute True Positive Rate (TPR) / Recall.

        TPR = correctly identified causal edges / ground truth edges

        Args:
            extracted_edges: Edges extracted by the method
            ground_truth_edges: Ground truth edges from CUAD annotations
            use_semantic_matching: If True, use semantic text similarity instead of
                                   type-pair matching. This is fairer for methods
                                   that don't use ontology types.

        For ontology-constrained methods (OCCI, pattern, rule):
            - Match by type pairs: (source_type, target_type)

        For unconstrained methods:
            - Match by semantic content overlap in span text
        """
        if not ground_truth_edges:
            return MetricResult(
                name="true_positive_rate", value=0.0, numerator=0, denominator=0
            )

        if use_semantic_matching:
            # Semantic matching: check if extracted edges capture similar concepts
            # This is fairer for unconstrained LLM which doesn't use types
            matches = 0

            # Define key causal concepts to look for
            causal_concepts = [
                ("insurance", "breach"),
                ("failure", "breach"),
                ("breach", "terminat"),
                ("default", "remedy"),
                ("violat", "damage"),
                ("confidential", "breach"),
                ("non-compete", "breach"),
                ("indemn", "cost"),
                ("shall", "breach"),
                ("obligation", "breach"),
            ]

            # Check if extracted edges capture any ground truth concepts
            extracted_text = " ".join(
                [
                    f"{e.source.span_text} {e.target.span_text} {e.span_evidence}"
                    for e in extracted_edges
                ]
            ).lower()

            for cause_kw, effect_kw in causal_concepts:
                if cause_kw in extracted_text and effect_kw in extracted_text:
                    matches += 1
                    break  # Count as found if any concept pair is present

            # Normalize: did we find at least some causal content?
            # This gives unconstrained LLM credit for finding relationships
            tpr = min(1.0, matches) if extracted_edges else 0.0

            return MetricResult(
                name="true_positive_rate",
                value=tpr,
                numerator=matches,
                denominator=1,  # Binary: found causal content or not
                details={"matching_mode": "semantic"},
            )

        # Type-based matching (default for ontology-constrained methods)
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
            details={"matching_mode": "type_signature"},
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
        Compute Decoy Rejection Rate (DRR) as per paper Section 7.2.

        DRR = (R_decoy - A_decoy) / (R_decoy + A_decoy)

        Where:
        - R_decoy = number of decoy-induced edges rejected by the validator
        - A_decoy = number of decoy-induced edges accepted (hallucinations)

        DRR ∈ [-1, 1]:
        - DRR = 1.0: Perfect rejection (all decoy edges rejected)
        - DRR = 0.0: Equal rejection and acceptance
        - DRR < 0: System accepts more decoy edges than it rejects (bad)

        Note: For methods without validators (unconstrained LLM), R_decoy=0
        so DRR = -A_decoy/A_decoy = -1.0 when any decoys are accepted.
        """
        total = rejected_decoy_edges + accepted_decoy_edges

        if total == 0:
            # No decoy-induced edges at all - perfect performance
            return MetricResult(
                name="decoy_rejection_rate", value=1.0, numerator=0, denominator=0
            )

        drr = (rejected_decoy_edges - accepted_decoy_edges) / total
        # Clamp to valid range (should be mathematically guaranteed, but defensive)
        drr = max(-1.0, min(1.0, drr))

        return MetricResult(
            name="decoy_rejection_rate",
            value=drr,
            numerator=rejected_decoy_edges,
            denominator=total,
            details={
                "rejected_decoy_edges": rejected_decoy_edges,
                "accepted_decoy_edges": accepted_decoy_edges,
                "interpretation": "positive=good, negative=hallucinating",
            },
        )

    def compute_invariance_accuracy(
        self,
        original_results: List[ExtractionResult],
        paraphrase_results: List[ExtractionResult],
    ) -> MetricResult:
        """
        Compute Counterfactual Stability (CS) / Invariance Accuracy.

        Per IEEE paper Section 7.2:
        CS = |E ∩ E'| / |E ∪ E'| (Jaccard similarity)

        Where E and E' are edge sets from original and paraphrased documents.

        This measures structural stability: the same causal structure should
        be extracted from semantically equivalent paraphrased text.
        """
        if len(original_results) != len(paraphrase_results):
            raise ValueError("Must have matching original and paraphrase results")

        if not original_results:
            return MetricResult(
                name="invariance_accuracy", value=1.0, numerator=0, denominator=0
            )

        # Compute Jaccard similarity for each pair, then average
        jaccard_scores = []

        for orig, para in zip(original_results, paraphrase_results):
            # Create edge signatures using type pairs (more robust than names)
            orig_sigs = set(
                (e.source.var_type.value, e.target.var_type.value) for e in orig.edges
            )
            para_sigs = set(
                (e.source.var_type.value, e.target.var_type.value) for e in para.edges
            )

            # Jaccard similarity: |intersection| / |union|
            intersection = orig_sigs & para_sigs
            union = orig_sigs | para_sigs

            if not union:
                # Both empty - perfect match
                jaccard = 1.0
            else:
                jaccard = len(intersection) / len(union)

            jaccard_scores.append(jaccard)

        # Average Jaccard across all pairs
        avg_jaccard = sum(jaccard_scores) / len(jaccard_scores)

        # Also compute strict match count (for paper's IA metric)
        matches = sum(1 for j in jaccard_scores if j >= 0.8)

        return MetricResult(
            name="invariance_accuracy",
            value=avg_jaccard,  # Use Jaccard as CS per paper
            numerator=matches,
            denominator=len(original_results),
            details={
                "avg_jaccard_similarity": avg_jaccard,
                "strict_matches": matches,
                "total_pairs": len(original_results),
                "per_pair_jaccard": jaccard_scores[:10],  # First 10 for debugging
            },
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
        n_decoys_tested: int = 0,
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

        # Determine if this is an unconstrained method (needs semantic matching)
        is_unconstrained = "unconstrained" in method_name.lower()

        # Aggregate all edges
        all_edges = []
        for result in extraction_results:
            all_edges.extend(result.edges)

        metrics = {}

        # Core metrics
        # For unconstrained LLM, SC is not meaningful (doesn't use ontology)
        if not is_unconstrained:
            sc_result = self.compute_structural_correctness(all_edges)
            metrics[sc_result.name] = sc_result
        else:
            # Report SC but note it's expected to be low for unconstrained
            sc_result = self.compute_structural_correctness(all_edges)
            sc_result.details = {"note": "SC not meaningful for unconstrained methods"}
            metrics[sc_result.name] = sc_result

        cs_result = self.compute_cycle_score(extraction_results)
        metrics[cs_result.name] = cs_result

        # Adversarial metrics (decoy rejection as per paper Section 7.2)
        # Report metrics if decoy test was run (n_decoys_tested > 0) even if no edges triggered
        total_decoy_related = rejected_decoy_edges + accepted_decoy_edges
        if n_decoys_tested > 0 or total_decoy_related > 0:
            # FPR based on accepted (hallucinated) decoy edges
            # If no decoy edges triggered (perfect result), FPR = 0
            fpr_result = self.compute_false_positive_rate(
                accepted_decoy_edges, max(total_decoy_related, n_decoys_tested)
            )
            metrics[fpr_result.name] = fpr_result

            # DRR = (R_decoy - A_decoy) / (R_decoy + A_decoy) as per paper
            # If no decoy edges at all, DRR = 1.0 (perfect rejection)
            if total_decoy_related == 0 and n_decoys_tested > 0:
                # Perfect: all decoys were ignored - no edges extracted from them
                drr_result = MetricResult(
                    name="decoy_rejection_rate",
                    value=1.0,
                    numerator=n_decoys_tested,
                    denominator=n_decoys_tested,
                    details={
                        "note": "All decoys correctly ignored (no edges extracted)"
                    },
                )
            else:
                drr_result = self.compute_decoy_rejection_rate(
                    rejected_decoy_edges, accepted_decoy_edges
                )
            metrics[drr_result.name] = drr_result

        # Ground truth metrics
        # Use semantic matching for unconstrained LLM (fairer comparison)
        if ground_truth_edges:
            tpr_result = self.compute_true_positive_rate(
                all_edges, ground_truth_edges, use_semantic_matching=is_unconstrained
            )
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
