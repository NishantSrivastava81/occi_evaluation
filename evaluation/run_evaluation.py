"""
Main evaluation runner for OCCI framework.

Orchestrates:
1. Data loading from CUAD
2. Extraction using multiple methods
3. Adversarial challenge testing
4. Metric computation and reporting
"""

import os
import sys
import json
import time
import logging
from pathlib import Path
from typing import List, Dict, Tuple, Optional
from dataclasses import dataclass, asdict
from datetime import datetime
import random

# Add project root to path
sys.path.insert(0, str(Path(__file__).parent.parent))

from config.settings import (
    DATA_DIR,
    RESULTS_DIR,
    CUAD_CLAUSE_MAPPING,
    VALIDATOR_CONFIG,
    EVALUATION_CONFIG,
)
from data.download_cuad import download_cuad, load_contracts, create_ground_truth_edges
from ontology.legal_ontology import LEGAL_ONTOLOGY, Variable, Edge, OntologyType
from ontology.validator import OntologyValidator
from extraction.llm_extractor import LLMExtractor, UnconstrainedLLMExtractor
from extraction.pattern_extractor import PatternExtractor
from extraction.rule_extractor import RuleExtractor
from adversarial.decoy_generator import DecoyGenerator
from adversarial.perturbation import PerturbationChallenge
from evaluation.metrics import OCCIMetrics, ExtractionResult, EvaluationSummary
from utils.extraction_logger import ExtractionLogger, get_extraction_logger


# Configure logging
logging.basicConfig(
    level=logging.INFO, format="%(asctime)s - %(name)s - %(levelname)s - %(message)s"
)
logger = logging.getLogger(__name__)


@dataclass
class EvaluationRun:
    """Record of an evaluation run."""

    run_id: str
    timestamp: str
    n_contracts: int
    methods_evaluated: List[str]
    summaries: Dict[str, Dict]
    config: Dict


class EvaluationRunner:
    """
    Main Evaluation Orchestrator for OCCI Framework.

    Coordinates the full experimental evaluation pipeline:
    1. Dataset loading from CUAD
    2. Multi-method extraction (OCCI, baselines)
    3. Adversarial challenge testing
    4. Metric computation and reporting

    Attributes:
        llm_provider: Backend provider ('azure', 'openai', 'anthropic')
        n_contracts: Number of contracts to evaluate
        include_adversarial: Whether to run adversarial tests
    """

    def __init__(
        self,
        llm_provider: str = "azure",
        n_contracts: int = 100,
        include_adversarial: bool = True,
    ):
        """
        Initialize the evaluation runner.

        Args:
            llm_provider: LLM backend ('azure', 'openai', or 'anthropic')
                          Default is 'azure' for Azure OpenAI GPT-5
            n_contracts: Number of contracts to evaluate
            include_adversarial: Whether to run adversarial challenge tests
        """
        self.llm_provider = llm_provider
        self.n_contracts = n_contracts
        self.include_adversarial = include_adversarial

        # Initialize evaluation components
        self.validator = OntologyValidator(LEGAL_ONTOLOGY)
        self.metrics = OCCIMetrics(self.validator)
        self.decoy_generator = DecoyGenerator()
        self.perturbation = PerturbationChallenge()

        # Initialize extraction methods
        self.extractors = {}
        self._init_extractors()

        # Ensure output directories exist
        RESULTS_DIR.mkdir(parents=True, exist_ok=True)

        logger.info(
            f"EvaluationRunner initialized: provider={llm_provider}, contracts={n_contracts}"
        )

    def _init_extractors(self):
        """Initialize all extraction methods for comparison."""
        logger.info("Initializing extraction methods...")

        # OCCI: Ontology-constrained LLM extraction (proposed method)
        self.extractors["occi"] = LLMExtractor(
            provider=self.llm_provider, ontology=LEGAL_ONTOLOGY
        )

        # Baseline: Unconstrained LLM (same model, no ontology)
        self.extractors["llm_unconstrained"] = UnconstrainedLLMExtractor(
            provider=self.llm_provider
        )

        # Baseline: Pattern-based extraction (regex, no LLM)
        self.extractors["pattern"] = PatternExtractor()

        # Baseline: Rule-only extraction (ontology templates, no LLM)
        self.extractors["rule"] = RuleExtractor()

        logger.info(f"Initialized {len(self.extractors)} extraction methods")

    def load_evaluation_data(self) -> List[Dict]:
        """Load and prepare evaluation data from CUAD."""
        logger.info("Loading CUAD contracts...")

        # Download if needed
        download_cuad()

        # Load contracts
        contracts = load_contracts(num_contracts=self.n_contracts * 2)

        # Sample for evaluation
        if len(contracts) > self.n_contracts:
            contracts = random.sample(contracts, self.n_contracts)

        logger.info(f"Loaded {len(contracts)} contracts for evaluation")
        return contracts

    def run_extraction(
        self, method_name: str, contracts: List[Dict]
    ) -> List[ExtractionResult]:
        """
        Run extraction with a specific method.

        Returns list of ExtractionResult objects.
        """
        logger.info(f"Running extraction with method: {method_name}")

        extractor = self.extractors.get(method_name)
        if not extractor:
            raise ValueError(f"Unknown method: {method_name}")

        results = []

        for i, contract in enumerate(contracts):
            try:
                start_time = time.time()
                contract_id = contract["id"]

                # Set contract ID on extractor for logging
                extractor._current_contract_id = contract_id

                # Extract structure
                if method_name in ["occi", "llm_unconstrained"]:
                    variables, edges = extractor.extract(contract["context"])
                else:
                    variables, edges = extractor.extract(contract["context"])

                extraction_time = time.time() - start_time

                # Validate (for OCCI only)
                validation_result = None
                validation_errors = None
                if method_name == "occi":
                    validation_result = self.validator.validate(
                        variables, edges, contract["context"]
                    )
                    # Collect validation errors for logging
                    validation_errors = [
                        r.reason
                        for r in validation_result.validation_results
                        if not r.is_valid
                    ]

                # Log extraction result for transparency
                try:
                    extraction_logger = get_extraction_logger()
                    extraction_logger.log_extraction(
                        method=method_name,
                        contract_id=contract_id,
                        input_text=contract["context"][:1000],
                        variables=variables,
                        edges=edges,
                        extraction_time=extraction_time,
                        validation_passed=(
                            validation_result.is_accepted if validation_result else None
                        ),
                        validation_errors=validation_errors,
                    )
                except Exception as log_error:
                    logger.debug(f"Extraction logging skipped: {log_error}")

                result = ExtractionResult(
                    method_name=method_name,
                    contract_id=contract_id,
                    variables=variables,
                    edges=edges,
                    extraction_time=extraction_time,
                    validation_result=validation_result,
                )
                results.append(result)

                if (i + 1) % 10 == 0:
                    logger.info(f"  Processed {i+1}/{len(contracts)} contracts")

            except Exception as e:
                logger.error(f"Error processing contract {contract['id']}: {e}")
                # Add empty result
                results.append(
                    ExtractionResult(
                        method_name=method_name,
                        contract_id=contract["id"],
                        variables=[],
                        edges=[],
                        extraction_time=0.0,
                    )
                )

        return results

    def run_decoy_test(
        self, method_name: str, contracts: List[Dict], n_decoys_per_contract: int = 5
    ) -> Tuple[int, int, int]:
        """
        Run decoy rejection test as per paper Section 7 methodology.

        Implements "Negative control insertion": Insert decoy clauses INTO
        the document text and verify that edges are not spuriously extracted
        from them. The validator detects edges "supported only by decoy spans".

        This aligns with the paper's specification:
        - DRR = (R_decoy - A_decoy) / (R_decoy + A_decoy)
        - where R_decoy = decoy-induced edges rejected by validator
        - and A_decoy = decoy-induced edges accepted (hallucinations)

        Args:
            method_name: Extraction method to test
            contracts: List of contracts to inject decoys into
            n_decoys_per_contract: Number of decoy sentences per contract

        Returns:
            (rejected_decoy_edges, accepted_decoy_edges, total_decoy_sentences)
        """
        logger.info(f"Running decoy injection test for {method_name}")
        logger.info(f"  Injecting {n_decoys_per_contract} decoys per contract")

        extractor = self.extractors.get(method_name)
        total_rejected = 0
        total_accepted = 0
        total_decoy_sentences = 0

        for i, contract in enumerate(contracts):
            try:
                # Get causal sentences from contract
                contract_text = contract["context"]
                contract_sentences = [
                    s.strip() for s in contract_text.split(".") if len(s.strip()) > 10
                ]

                # Generate decoys
                decoys = self.decoy_generator.generate_decoy_set(n_decoys_per_contract)
                decoy_texts = [d.text for d in decoys]

                # Mix decoys into contract (as per paper's "negative control insertion")
                # Pass decoy_texts to ensure consistency between injection and detection
                mixed_doc, decoy_indices, actual_decoy_texts = (
                    self.decoy_generator.create_mixed_document(
                        contract_sentences, decoy_ratio=0.3, decoy_texts=decoy_texts
                    )
                )
                total_decoy_sentences += len(actual_decoy_texts)

                # Extract from mixed document
                variables, edges = extractor.extract(mixed_doc)

                # Validate with decoy spans - validator will detect decoy-only edges
                validation_result = self.validator.validate(
                    variables, edges, mixed_doc, decoy_spans=actual_decoy_texts
                )

                # Count decoy-related rejections from validation certificate
                decoy_rejections = len(
                    [
                        (edge, reason)
                        for edge, reason in validation_result.rejected_edges
                        if "decoy" in reason.lower()
                    ]
                )
                total_rejected += decoy_rejections

                # Count edges that were accepted but grounded in decoys (hallucinations)
                # These are edges the validator should have caught
                decoy_accepted = self._count_decoy_grounded_edges(
                    validation_result.accepted_edges, actual_decoy_texts, mixed_doc
                )
                total_accepted += decoy_accepted

                # Log adversarial test for transparency
                try:
                    extraction_logger = get_extraction_logger()
                    extraction_logger.log_adversarial_test(
                        test_type="decoy_injection",
                        contract_id=contract.get("id", f"contract_{i}"),
                        details={
                            "n_decoys_injected": len(actual_decoy_texts),
                            "n_edges_extracted": len(edges),
                            "n_decoy_rejections": decoy_rejections,
                            "n_decoy_accepted": decoy_accepted,
                            "decoy_types": [d.decoy_type for d in decoys],
                        },
                    )
                except Exception as log_error:
                    pass

                if (i + 1) % 10 == 0:
                    logger.info(f"  Processed {i+1}/{len(contracts)} contracts")

            except Exception as e:
                logger.warning(
                    f"Decoy test error for contract {contract.get('id', i)}: {e}"
                )

        logger.info(
            f"  Decoy test complete: {total_rejected} rejected, {total_accepted} accepted (hallucinations)"
        )
        return total_rejected, total_accepted, total_decoy_sentences

    def _count_decoy_grounded_edges(
        self, edges: List[Edge], decoy_texts: List[str], document_text: str
    ) -> int:
        """
        Count edges that are grounded primarily in decoy text.
        These represent hallucinations that slipped through validation.
        """
        decoy_grounded = 0
        decoy_text_lower = " ".join(decoy_texts).lower()
        decoy_words = set(decoy_text_lower.split())

        # Get non-decoy words from document
        doc_without_decoys = document_text.lower()
        for decoy in decoy_texts:
            doc_without_decoys = doc_without_decoys.replace(decoy.lower(), "")
        non_decoy_words = set(doc_without_decoys.split())

        for edge in edges:
            if not edge.span_evidence:
                continue
            span_words = set(edge.span_evidence.lower().split())
            if not span_words:
                continue

            decoy_overlap = len(span_words & decoy_words) / len(span_words)
            non_decoy_overlap = len(span_words & non_decoy_words) / len(span_words)

            # Edge is decoy-grounded if mostly overlaps with decoy text
            if decoy_overlap > 0.7 and non_decoy_overlap < 0.3:
                decoy_grounded += 1

        return decoy_grounded

    def run_paraphrase_test(
        self,
        method_name: str,
        contracts: List[Dict],
        extraction_results: List[ExtractionResult],
        n_samples: int = 2,
    ) -> Tuple[List[ExtractionResult], List[ExtractionResult]]:
        """
        Run paraphrase invariance test.

        Tests structural stability: the same causal structure should be
        extracted from semantically equivalent paraphrased text.

        OPTIMIZED: Reuses existing extraction_results instead of re-extracting.

        Returns (original_results, paraphrase_results)
        """
        n_samples = min(n_samples, len(contracts))
        logger.info(f"Running paraphrase test for {method_name} (n={n_samples})")

        extractor = self.extractors.get(method_name)

        # Build lookup of existing results by contract_id
        results_by_id = {r.contract_id: r for r in extraction_results}

        original_results = []
        paraphrase_results = []

        for contract in contracts[:n_samples]:
            try:
                # Reuse existing extraction result instead of re-extracting
                contract_id = contract["id"]
                if contract_id in results_by_id:
                    original_results.append(results_by_id[contract_id])
                else:
                    # Fallback: extract if not found
                    start_time = time.time()
                    orig_vars, orig_edges = extractor.extract(contract["context"])
                    orig_time = time.time() - start_time
                    original_results.append(
                        ExtractionResult(
                            method_name=method_name,
                            contract_id=contract_id,
                            variables=orig_vars,
                            edges=orig_edges,
                            extraction_time=orig_time,
                        )
                    )

                # Generate paraphrase
                perturbation = self.perturbation.generate_paraphrase(
                    contract["context"]
                )

                # Paraphrase extraction (this is the only API call needed)
                start_time = time.time()
                para_vars, para_edges = extractor.extract(perturbation.perturbed_text)
                para_time = time.time() - start_time

                paraphrase_results.append(
                    ExtractionResult(
                        method_name=method_name,
                        contract_id=f"{contract_id}_para",
                        variables=para_vars,
                        edges=para_edges,
                        extraction_time=para_time,
                    )
                )

            except Exception as e:
                logger.warning(f"Paraphrase test error: {e}")
                # Remove the original if paraphrase failed to keep lists aligned
                if (
                    original_results
                    and original_results[-1].contract_id == contract["id"]
                ):
                    original_results.pop()

        return original_results, paraphrase_results

    def run_contradiction_test(
        self,
        method_name: str,
        contracts: List[Dict],
        extraction_results: List[ExtractionResult],
        n_samples: int = 2,
    ) -> Tuple[List[ExtractionResult], List[ExtractionResult]]:
        """
        Run contradiction detection test.

        Tests sensitivity to semantic changes: edges should change when
        causal markers are inverted (e.g., 'X causes Y' -> 'X does not cause Y').

        OPTIMIZED: Reuses existing extraction_results instead of re-extracting.

        Returns (original_results, contradiction_results)
        """
        n_samples = min(n_samples, len(contracts))
        logger.info(f"Running contradiction test for {method_name} (n={n_samples})")

        extractor = self.extractors.get(method_name)

        # Build lookup of existing results by contract_id
        results_by_id = {r.contract_id: r for r in extraction_results}

        original_results = []
        contradiction_results = []

        for contract in contracts[:n_samples]:
            try:
                # Reuse existing extraction result instead of re-extracting
                contract_id = contract["id"]
                if contract_id in results_by_id:
                    orig_result = results_by_id[contract_id]
                else:
                    # Fallback: extract if not found
                    start_time = time.time()
                    orig_vars, orig_edges = extractor.extract(contract["context"])
                    orig_time = time.time() - start_time
                    orig_result = ExtractionResult(
                        method_name=method_name,
                        contract_id=contract_id,
                        variables=orig_vars,
                        edges=orig_edges,
                        extraction_time=orig_time,
                    )

                # Generate contradiction
                perturbation = self.perturbation.generate_contradiction(
                    contract["context"]
                )
                if not perturbation:
                    continue

                # Contradiction extraction (this is the only API call needed)
                start_time = time.time()
                contra_vars, contra_edges = extractor.extract(
                    perturbation.perturbed_text
                )
                contra_time = time.time() - start_time

                # Add both results only if contradiction succeeded
                original_results.append(orig_result)
                contradiction_results.append(
                    ExtractionResult(
                        method_name=method_name,
                        contract_id=f"{contract_id}_contra",
                        variables=contra_vars,
                        edges=contra_edges,
                        extraction_time=contra_time,
                    )
                )

            except Exception as e:
                logger.warning(f"Contradiction test error: {e}")

        return original_results, contradiction_results

    def run_full_evaluation(self) -> EvaluationRun:
        """
        Run full evaluation on all methods.

        Returns EvaluationRun with all results.
        """
        run_id = datetime.now().strftime("%Y%m%d_%H%M%S")
        logger.info(f"Starting evaluation run: {run_id}")

        # Initialize extraction logger for this run
        ExtractionLogger.reset_instance()  # Clear any previous instance
        extraction_logger = ExtractionLogger.get_instance(run_id)
        logger.info(f"Extraction logs will be saved to: {extraction_logger.log_dir}")

        # Load data
        contracts = self.load_evaluation_data()

        # Create ground truth edges from all contracts
        ground_truth = []
        for contract in contracts:
            ground_truth.extend(create_ground_truth_edges(contract))

        summaries = {}

        for method_name in self.extractors.keys():
            logger.info(f"\n{'='*60}")
            logger.info(f"Evaluating method: {method_name}")
            logger.info(f"{'='*60}")

            # Main extraction
            extraction_results = self.run_extraction(method_name, contracts)

            # Adversarial tests (if enabled and LLM-based)
            rejected_decoy_edges = 0
            accepted_decoy_edges = 0
            n_decoy_sentences = 0
            paraphrase_original = None
            paraphrase_results = None
            contradiction_original = None
            contradiction_results = None

            # Only run adversarial tests for LLM-based methods (not pattern/rule)
            is_llm_method = method_name in ["occi", "llm_unconstrained"]

            if self.include_adversarial and is_llm_method:
                # Decoy test - inject decoys into contracts as per IEEE paper Section 7
                # Uses "Negative control insertion" methodology
                rejected_decoy_edges, accepted_decoy_edges, n_decoy_sentences = (
                    self.run_decoy_test(method_name, contracts, n_decoys_per_contract=5)
                )

                # Paraphrase test - run on all contracts, reuse extraction results
                orig_para, para_results = self.run_paraphrase_test(
                    method_name, contracts, extraction_results, n_samples=len(contracts)
                )
                # Store both original and paraphrase results for comparison
                paraphrase_original = orig_para
                paraphrase_results = para_results

                # Contradiction test - run on all contracts, reuse extraction results
                orig_contra, contra_results = self.run_contradiction_test(
                    method_name, contracts, extraction_results, n_samples=len(contracts)
                )
                # Store both original and contradiction results for comparison
                contradiction_original = orig_contra
                contradiction_results = contra_results

            # Compute metrics
            summary = self.metrics.compute_all_metrics(
                extraction_results=extraction_results,
                rejected_decoy_edges=rejected_decoy_edges,
                accepted_decoy_edges=accepted_decoy_edges,
                n_decoys_tested=n_decoy_sentences,
                ground_truth_edges=ground_truth,
                paraphrase_original=paraphrase_original,
                paraphrase_results=paraphrase_results,
                contradiction_original=contradiction_original,
                contradiction_results=contradiction_results,
            )

            # Display results
            print(self.metrics.format_results(summary))

            # Store summary
            summaries[method_name] = self.metrics.to_dict(summary)

        # Finalize extraction logger
        try:
            extraction_logger = get_extraction_logger()
            extraction_logger.finalize()
            logger.info(f"Extraction logs finalized: {extraction_logger.log_dir}")
        except Exception as e:
            logger.warning(f"Could not finalize extraction logger: {e}")

        # Create run record
        run = EvaluationRun(
            run_id=run_id,
            timestamp=datetime.now().isoformat(),
            n_contracts=len(contracts),
            methods_evaluated=list(self.extractors.keys()),
            summaries=summaries,
            config={
                "llm_provider": self.llm_provider,
                "include_adversarial": self.include_adversarial,
            },
        )

        # Save results
        self.save_results(run)

        return run

    def save_results(self, run: EvaluationRun):
        """Save evaluation results to JSON."""
        output_path = RESULTS_DIR / f"evaluation_{run.run_id}.json"

        with open(output_path, "w") as f:
            json.dump(asdict(run), f, indent=2, default=str)

        logger.info(f"Results saved to: {output_path}")

    def generate_comparison_table(self, run: EvaluationRun) -> str:
        """Generate a comparison table for all methods."""
        lines = ["\n" + "=" * 80, "EVALUATION COMPARISON TABLE", "=" * 80, ""]

        # Header
        methods = list(run.summaries.keys())
        metric_names = [
            "structural_correctness",
            "cycle_score",
            "false_positive_rate",
            "decoy_rejection_rate",
            "true_positive_rate",
        ]

        header = "Metric".ljust(30) + "".join(m.center(15) for m in methods)
        lines.append(header)
        lines.append("-" * 80)

        # Rows
        for metric in metric_names:
            row = metric.ljust(30)
            for method in methods:
                if metric in run.summaries[method]["metrics"]:
                    value = run.summaries[method]["metrics"][metric]["value"]
                    row += f"{value:.4f}".center(15)
                else:
                    row += "N/A".center(15)
            lines.append(row)

        lines.append("=" * 80 + "\n")

        return "\n".join(lines)


def run_quick_test():
    """Run a quick test with minimal data."""
    logger.info("Running quick test...")

    runner = EvaluationRunner(
        llm_provider="openai", n_contracts=5, include_adversarial=False
    )

    # Just test with pattern extractor (no API calls)
    sample_text = """
    Section 4.1: Insurance Requirements.
    The Vendor shall maintain comprehensive liability insurance with coverage 
    not less than $2,000,000 per occurrence. Failure to maintain such insurance 
    shall constitute a material breach of this Agreement. Upon such breach, 
    the Company may terminate this Agreement immediately.
    """

    # Test pattern extractor
    pattern_extractor = PatternExtractor()
    variables, edges = pattern_extractor.extract(sample_text)

    logger.info(f"Pattern extraction: {len(variables)} variables, {len(edges)} edges")

    # Test rule extractor
    rule_extractor = RuleExtractor()
    variables, edges = rule_extractor.extract(sample_text)

    logger.info(f"Rule extraction: {len(variables)} variables, {len(edges)} edges")

    # Test validator
    validator = OntologyValidator(LEGAL_ONTOLOGY)
    result = validator.validate(variables, edges, sample_text)

    logger.info(f"Validation passed: {result.is_valid}")
    logger.info(f"Errors: {len(result.errors)}, Warnings: {len(result.warnings)}")

    return True


if __name__ == "__main__":
    import argparse

    parser = argparse.ArgumentParser(description="OCCI Evaluation Framework")
    parser.add_argument("--quick", action="store_true", help="Run quick test")
    parser.add_argument("--full", action="store_true", help="Run full evaluation")
    parser.add_argument("--contracts", type=int, default=50, help="Number of contracts")
    parser.add_argument(
        "--provider",
        type=str,
        default="openai",
        choices=["openai", "anthropic"],
        help="LLM provider",
    )
    parser.add_argument(
        "--no-adversarial", action="store_true", help="Skip adversarial tests"
    )

    args = parser.parse_args()

    if args.quick:
        run_quick_test()
    elif args.full:
        runner = EvaluationRunner(
            llm_provider=args.provider,
            n_contracts=args.contracts,
            include_adversarial=not args.no_adversarial,
        )
        run = runner.run_full_evaluation()
        print(runner.generate_comparison_table(run))
    else:
        print("Use --quick for a quick test or --full for full evaluation")
        print(
            "Example: python run_evaluation.py --full --contracts 50 --provider openai"
        )
