"""
Download and prepare CUAD (Contract Understanding Atticus Dataset).

CUAD dataset from HuggingFace: https://huggingface.co/datasets/kenlevine/CUAD
Contains 510 commercial legal contracts with 41 clause type annotations.

Reference:
    Hendrycks et al. (2021) "CUAD: An Expert-Annotated NLP Dataset for
    Legal Contract Review" - https://arxiv.org/abs/2103.06268
"""

import json
from pathlib import Path
from typing import Dict, List, Optional
from tqdm import tqdm

import sys

sys.path.append(str(Path(__file__).parent.parent))
from config.settings import DATA_DIR, CUAD_CLAUSE_MAPPING


def download_cuad(force_download: bool = False) -> Path:
    """
    Download CUAD dataset from HuggingFace.

    Uses the kenlevine/CUAD dataset which has the SQuAD format:
        from datasets import load_dataset
        ds = load_dataset("kenlevine/CUAD")

    Args:
        force_download: If True, re-download even if exists

    Returns:
        Path to the processed data directory
    """
    from datasets import load_dataset

    cuad_dir = DATA_DIR / "cuad"
    processed_file = cuad_dir / "contracts_processed.json"

    if processed_file.exists() and not force_download:
        print(f"CUAD dataset already exists at {cuad_dir}")
        return cuad_dir

    cuad_dir.mkdir(parents=True, exist_ok=True)

    print("Downloading CUAD dataset from HuggingFace...")
    print("  Using: load_dataset('kenlevine/CUAD')")

    # Load from HuggingFace - kenlevine/CUAD has the SQuAD format
    dataset = load_dataset("kenlevine/CUAD")

    # The dataset has a single row with 'data' containing all contracts
    raw_data = dataset["train"][0]["data"]
    print(f"  Loaded {len(raw_data)} contracts")

    # Process and structure the data
    contracts = {}

    print("Processing contracts...")

    for doc in tqdm(raw_data, desc="Processing contracts"):
        title = doc.get("title", "")
        paragraphs = doc.get("paragraphs", [])

        for para in paragraphs:
            context = para.get("context", "")
            qas = para.get("qas", [])

            # Generate contract ID from title
            contract_id = title.replace(" ", "_").replace("/", "_")[:50]

            if not contract_id or not context:
                continue

            if contract_id not in contracts:
                contracts[contract_id] = {
                    "id": contract_id,
                    "title": title,
                    "context": context,
                    "clauses": {},
                    "annotations": [],
                }
            elif len(context) > len(contracts[contract_id].get("context", "")):
                # Keep the longer context
                contracts[contract_id]["context"] = context

            # Process QA pairs
            for qa in qas:
                question = qa.get("question", "")
                answers = qa.get("answers", [])

                # Extract clause type from question
                clause_type = (
                    question.replace(
                        "Highlight the parts (if any) of this contract related to ",
                        "",
                    )
                    .replace(".", "")
                    .strip()
                )

                if clause_type not in contracts[contract_id]["clauses"]:
                    contracts[contract_id]["clauses"][clause_type] = []

                # Process answers
                for ans in answers:
                    answer_text = ans.get("text", "")
                    answer_start = ans.get("answer_start", 0)

                    if answer_text:
                        contracts[contract_id]["clauses"][clause_type].append(
                            {
                                "text": answer_text,
                                "start": answer_start,
                                "ontology_type": CUAD_CLAUSE_MAPPING.get(
                                    clause_type, "Clause"
                                ),
                            }
                        )

    # Save processed data
    with open(processed_file, "w", encoding="utf-8") as f:
        json.dump(contracts, f, indent=2, ensure_ascii=False)

    print(f"Processed {len(contracts)} unique contracts")
    print(f"Saved to {processed_file}")

    # Create summary statistics
    stats = {
        "total_contracts": len(contracts),
        "clause_types": {},
        "avg_clauses_per_contract": 0,
    }

    total_clauses = 0
    for contract in contracts.values():
        for clause_type, instances in contract["clauses"].items():
            if clause_type not in stats["clause_types"]:
                stats["clause_types"][clause_type] = 0
            stats["clause_types"][clause_type] += len(instances)
            total_clauses += len(instances)

    stats["avg_clauses_per_contract"] = (
        total_clauses / len(contracts) if contracts else 0
    )

    with open(cuad_dir / "stats.json", "w") as f:
        json.dump(stats, f, indent=2)

    return cuad_dir


def load_contracts(
    num_contracts: Optional[int] = None, split: str = "all"
) -> List[Dict]:
    """
    Load processed contracts.

    Args:
        num_contracts: Number of contracts to load (None for all)
        split: "train", "test", or "all"

    Returns:
        List of contract dictionaries
    """
    cuad_dir = DATA_DIR / "cuad"
    processed_file = cuad_dir / "contracts_processed.json"

    if not processed_file.exists():
        download_cuad()

    with open(processed_file, "r", encoding="utf-8") as f:
        contracts = json.load(f)

    contract_list = list(contracts.values())

    if num_contracts:
        contract_list = contract_list[:num_contracts]

    return contract_list


def get_contract_text(contract_id: str) -> Optional[str]:
    """
    Get the full text of a specific contract.

    Args:
        contract_id: The contract identifier

    Returns:
        Contract text or None if not found
    """
    cuad_dir = DATA_DIR / "cuad"
    processed_file = cuad_dir / "contracts_processed.json"

    if not processed_file.exists():
        return None

    with open(processed_file, "r", encoding="utf-8") as f:
        contracts = json.load(f)

    if contract_id in contracts:
        return contracts[contract_id].get("context", "")
    return None


def get_contract_clauses(contract_id: str) -> Dict:
    """
    Get clause annotations for a contract.

    Args:
        contract_id: The contract identifier

    Returns:
        Dictionary of clause type -> list of clause instances
    """
    cuad_dir = DATA_DIR / "cuad"
    processed_file = cuad_dir / "contracts_processed.json"

    if not processed_file.exists():
        return {}

    with open(processed_file, "r", encoding="utf-8") as f:
        contracts = json.load(f)

    if contract_id in contracts:
        return contracts[contract_id].get("clauses", {})
    return {}


def get_dataset_stats() -> Dict:
    """
    Get dataset statistics.

    Returns:
        Dictionary with dataset statistics
    """
    stats_file = DATA_DIR / "cuad" / "stats.json"

    if not stats_file.exists():
        download_cuad()

    with open(stats_file, "r") as f:
        return json.load(f)


def create_ground_truth_edges(contract: Dict) -> List[Dict]:
    """
    Create CUAD-derived ground truth causal edges from contract clause annotations.

    This implements the CUAD-Derived Annotation Methodology from Section 7.1:
    1. Clause-to-Variable Mapping: Map CUAD's 41 clause types to ontology types
    2. Edge Derivation Rules: Create edges using ontology partial order constraints
    3. Automated Extraction: Pattern matching on CUAD annotations

    Args:
        contract: Contract dictionary with clauses

    Returns:
        List of edge dictionaries with source/target types
    """
    edges = []
    clauses = contract.get("clauses", {})

    # Ontology hierarchy for edge creation:
    # Clause -> Obligation -> ComplianceEvent -> Outcome -> {Remedy, Cost}

    # Map CUAD types to ontology types
    type_mapping = CUAD_CLAUSE_MAPPING

    # Create edges based on common legal patterns
    for clause_type, instances in clauses.items():
        if not instances:
            continue

        ont_type = type_mapping.get(clause_type, "Clause")

        for instance in instances:
            # Clause -> Obligation edges
            if ont_type == "Clause":
                edges.append(
                    {
                        "source_type": "Clause",
                        "target_type": "Obligation",
                        "source_text": instance.get("text", "")[:100],
                        "confidence": 0.9,
                    }
                )

            # Obligation patterns
            elif ont_type == "Obligation":
                edges.append(
                    {
                        "source_type": "Obligation",
                        "target_type": "ComplianceEvent",
                        "source_text": instance.get("text", "")[:100],
                        "confidence": 0.85,
                    }
                )

            # Termination/Breach -> Outcome edges
            elif (
                "termination" in clause_type.lower() or "breach" in clause_type.lower()
            ):
                edges.append(
                    {
                        "source_type": "ComplianceEvent",
                        "target_type": "Outcome",
                        "source_text": instance.get("text", "")[:100],
                        "confidence": 0.9,
                    }
                )

            # Indemnification/Damages -> Cost edges
            elif (
                "indemnification" in clause_type.lower()
                or "damage" in clause_type.lower()
            ):
                edges.append(
                    {
                        "source_type": "Outcome",
                        "target_type": "Cost",
                        "source_text": instance.get("text", "")[:100],
                        "confidence": 0.85,
                    }
                )

            # Remedies -> Remedy edges
            elif "remedy" in clause_type.lower() or "cure" in clause_type.lower():
                edges.append(
                    {
                        "source_type": "Outcome",
                        "target_type": "Remedy",
                        "source_text": instance.get("text", "")[:100],
                        "confidence": 0.85,
                    }
                )

    return edges


if __name__ == "__main__":
    # Download and process dataset
    download_cuad(force_download=True)

    # Print summary
    stats = get_dataset_stats()
    print("\nDataset Statistics:")
    print(f"  Total contracts: {stats['total_contracts']}")
    print(f"  Avg clauses per contract: {stats['avg_clauses_per_contract']:.1f}")
    print(f"  Clause types: {len(stats['clause_types'])}")
