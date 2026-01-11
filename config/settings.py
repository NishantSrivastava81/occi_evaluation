"""
Configuration Settings for OCCI Empirical Evaluation Framework.

Ontology-Constrained Causal Induction (OCCI) from Legal Contract Text
Using Large Language Models with Structural Validation.

This module provides research configuration parameters for reproducible
experimentation on the CUAD (Contract Understanding Atticus Dataset).

References:
    - CUAD Dataset: https://huggingface.co/datasets/theatticusproject/cuad-qa
    - Azure OpenAI: https://azure.microsoft.com/products/ai-services/openai-service
"""

import os
from pathlib import Path
from dotenv import load_dotenv

# Load environment variables from .env file
load_dotenv()

# ============================================================================
# PROJECT PATHS
# ============================================================================
PROJECT_ROOT = Path(__file__).parent.parent
DATA_DIR = PROJECT_ROOT / "data"
RESULTS_DIR = PROJECT_ROOT / "results"
CACHE_DIR = PROJECT_ROOT / ".cache"
FIGURES_DIR = PROJECT_ROOT / "figures"

# Create directories if they don't exist
for _dir in [DATA_DIR, RESULTS_DIR, CACHE_DIR, FIGURES_DIR]:
    _dir.mkdir(exist_ok=True, parents=True)

# ============================================================================
# AZURE OPENAI CONFIGURATION (PRIMARY)
# ============================================================================
# Azure OpenAI GPT-5 is the primary model for this research.
# Configure via environment variables or .env file.

AZURE_OPENAI_ENDPOINT = os.getenv("AZURE_OPENAI_ENDPOINT")
AZURE_OPENAI_API_KEY = os.getenv("AZURE_OPENAI_API_KEY")
AZURE_OPENAI_API_VERSION = os.getenv("AZURE_OPENAI_API_VERSION", "2024-12-01-preview")

# Deployment names for Azure OpenAI models
AZURE_GPT5_DEPLOYMENT = os.getenv("AZURE_OPENAI_GPT5_DEPLOYMENT", "gpt-5-mini")
AZURE_GPT4_DEPLOYMENT = os.getenv("AZURE_OPENAI_GPT4_DEPLOYMENT", "gpt-4o")

# Fallback API keys (for ablation studies with other providers)
OPENAI_API_KEY = os.getenv("OPENAI_API_KEY")
ANTHROPIC_API_KEY = os.getenv("ANTHROPIC_API_KEY")

# ============================================================================
# MODEL CONFIGURATION
# ============================================================================
# Primary: Azure OpenAI GPT-5 for main experiments
# Secondary: GPT-4o for ablation comparison

DEFAULT_MODEL = "gpt-5-mini"  # Azure OpenAI GPT-5-mini (faster)
FALLBACK_MODEL = "gpt-4o"  # Azure OpenAI GPT-4o for comparison
TEMPERATURE = 1.0  # GPT-5-mini only supports temperature=1.0
MAX_TOKENS = 16384  # Reduced for gpt-5-mini (faster processing)
TOP_P = 1.0  # Nucleus sampling disabled for determinism
FREQUENCY_PENALTY = 0.0
PRESENCE_PENALTY = 0.0

# ============================================================================
# EXPERIMENTAL PARAMETERS
# ============================================================================
# Parameters for empirical evaluation on CUAD dataset.
# Values determined through preliminary experiments (Section 5.1).

# Dataset sampling
DEFAULT_NUM_CONTRACTS = 100  # Primary evaluation set size
VALIDATION_SET_SIZE = 20  # Held-out validation set
TEST_SET_SIZE = 50  # Final test set
RANDOM_SEED = 42  # For reproducibility

# Adversarial challenge parameters
NUM_DECOYS_PER_CONTRACT = 15  # Decoy sentences per contract
NUM_PARAPHRASES = 5  # Paraphrase variations per sample
NUM_CONTRADICTION_SAMPLES = 30  # Contradiction challenge samples

# Threshold parameters (empirically tuned)
SIMILARITY_THRESHOLD = 0.75  # Semantic similarity for decoy filtering
CONFIDENCE_THRESHOLD = 0.65  # Contradiction challenge confidence
AGREEMENT_THRESHOLD = 0.80  # Cross-prompt agreement threshold
SPAN_GROUNDING_THRESHOLD = 0.85  # Required text span overlap

# ============================================================================
# VALIDATOR HYPERPARAMETERS (Algorithm 1)
# ============================================================================
VALIDATOR_CONFIG = {
    "k_decoys": 15,  # Number of decoy sentences injected
    "similarity_threshold": 0.75,  # τ_sim: semantic similarity threshold
    "tau_conf": 0.65,  # τ_conf: confidence threshold for edge acceptance
    "tau_agree": 0.80,  # τ_agree: multi-prompt agreement threshold
    "multi_prompt_threshold": 3,  # Accept if ≥3/5 prompts agree
    "num_prompts": 5,  # Number of prompt variations for consensus
    "max_cycles_to_report": 10,  # Maximum cycles to enumerate
    "span_overlap_min": 0.5,  # Minimum text span overlap required
}

# ============================================================================
# EVALUATION CONFIGURATION
# ============================================================================
EVALUATION_CONFIG = {
    "metrics": [
        "structural_correctness",  # SC: Valid edges / total edges
        "cycle_score",  # CS: 1 - cyclic graphs / total graphs
        "false_positive_rate",  # FPR: Decoy edges / decoy opportunities
        "true_positive_rate",  # TPR: Correctly identified edges
        "decoy_rejection_rate",  # DRR: Successfully rejected decoys
        "invariance_accuracy",  # IA: Paraphrase consistency
        "inversion_rate",  # IR: Contradiction detection rate
    ],
    "bootstrap_samples": 1000,  # For confidence interval estimation
    "confidence_level": 0.95,  # 95% confidence intervals
    "significance_level": 0.05,  # For statistical tests
}

# Ontology type hierarchy (partial order)
TYPE_ORDER = {
    "Clause": 0,
    "Obligation": 1,
    "ComplianceEvent": 2,
    "Outcome": 3,
    "Remedy": 4,
    "Cost": 4,  # Same level as Remedy
}

# Allowed edge types (R)
ALLOWED_EDGES = {
    ("Clause", "Obligation"),
    ("Obligation", "ComplianceEvent"),
    ("ComplianceEvent", "Outcome"),
    ("Outcome", "Remedy"),
    ("Outcome", "Cost"),
}

# Orientation patterns for edge detection
ORIENTATION_PATTERNS = [
    "shall constitute",
    "shall be deemed",
    "shall be considered",
    "results in",
    "leads to",
    "triggers",
    "causes",
    "upon",
    "if",
    "provided that",
    "subject to",
    "in the event of",
    "failure to",
    "breach of",
    "may terminate",
    "entitled to",
    "liable for",
    "responsible for",
]

# CUAD clause types we map to our ontology
CUAD_CLAUSE_MAPPING = {
    # Clause types
    "Document Name": None,  # Metadata, not a clause
    "Parties": "Clause",
    "Agreement Date": "Clause",
    "Effective Date": "Clause",
    "Expiration Date": "Clause",
    "Renewal Term": "Clause",
    "Notice Period To Terminate Renewal": "Clause",
    "Governing Law": "Clause",
    "Most Favored Nation": "Clause",
    "Non-Compete": "Clause",
    "Exclusivity": "Clause",
    "No-Solicit Of Customers": "Clause",
    "No-Solicit Of Employees": "Clause",
    "Non-Disparagement": "Clause",
    "Termination For Convenience": "Clause",
    "Rofr/Rofo/Rofn": "Clause",
    "Change Of Control": "Clause",
    "Anti-Assignment": "Clause",
    "Revenue/Profit Sharing": "Clause",
    "Price Restrictions": "Clause",
    "Minimum Commitment": "Clause",
    "Volume Restriction": "Clause",
    "Ip Ownership Assignment": "Clause",
    "Joint Ip Ownership": "Clause",
    "License Grant": "Clause",
    "Non-Transferable License": "Clause",
    "Affiliate License-Licensor": "Clause",
    "Affiliate License-Licensee": "Clause",
    "Unlimited/All-You-Can-Eat-License": "Clause",
    "Irrevocable Or Perpetual License": "Clause",
    "Source Code Escrow": "Clause",
    "Post-Termination Services": "Clause",
    "Audit Rights": "Clause",
    "Uncapped Liability": "Clause",
    "Cap On Liability": "Clause",
    "Liquidated Damages": "Clause",
    "Warranty Duration": "Clause",
    "Insurance": "Clause",
    "Covenant Not To Sue": "Clause",
    "Third Party Beneficiary": "Clause",
}
