# OCCI: Ontology-Constrained Causal Induction from Legal Text

**Empirical Evaluation Framework for Structural Causal Model Extraction**

This repository contains the experimental framework for evaluating the OCCI (Ontology-Constrained Causal Induction) methodology on the CUAD (Contract Understanding Atticus Dataset) using Azure OpenAI GPT-5.

## Dataset Configuration

- **Training set:** 250 contracts
- **Validation set:** 50 contracts  
- **Test set:** 150 contracts
- **CUAD-derived causal edges:** 3,847 edges across 450 contracts
- **Decoy insertions:** 10 per contract (1,500 total for test set)

## Abstract

Large Language Models can extract causal relationships from text but often produce structurally invalid graphs (cycles, type violations). OCCI addresses this by constraining LLM extraction with a domain ontology that enforces:
- Type hierarchy: Clause ≺ Obligation ≺ ComplianceEvent ≺ Outcome ≺ {Remedy, Cost}
- Allowed edge types: Only ontology-permitted causal connections
- Span grounding: Variables must be grounded in text spans
- Acyclicity: The resulting SCM must be a DAG

## Dataset Configuration

- **Training set:** 250 contracts
- **Validation set:** 50 contracts  
- **Test set:** 150 contracts
- **CUAD-derived causal edges:** 3,847 edges across 450 contracts
- **Decoy insertions:** 10 per contract (1,500 total for test set)

## Project Structure

```
occi_evaluation/
├── config/
│   └── settings.py              # Configuration, Azure OpenAI settings, hyperparameters
├── data/
│   └── download_cuad.py         # CUAD dataset download and preprocessing
├── ontology/
│   ├── legal_ontology.py        # Legal contracts ontology (O = ⟨T, R, ≺⟩)
│   └── validator.py             # Algorithm 1: Ontology constraint validation
├── extraction/
│   ├── llm_extractor.py         # Azure OpenAI GPT-5 based extraction
│   ├── pattern_extractor.py     # Pattern-based baseline (regex)
│   └── rule_extractor.py        # Rule-only baseline (no LLM)
├── adversarial/
│   ├── decoy_generator.py       # Semantically neutral decoy generation
│   └── perturbation.py          # Paraphrase invariance & contradiction tests
├── evaluation/
│   ├── metrics.py               # Metrics: SC, CS, FPR, TPR, DRR, IA, IR
│   └── run_evaluation.py        # Evaluation orchestration
├── results/                     # Experimental results (JSON)
├── figures/                     # Publication figures
├── requirements.txt
└── main.py                      # CLI entry point
```

## Solution Overview & Workflow

The OCCI framework processes legal documents to robustly identify causal chains (e.g., "Failure to pay" -> "Late Fee"). It follows this pipeline when you run `python main.py --run`:

1.  **Data Loading (`data/`)**: Loads a stratified sample of legal contracts from the CUAD dataset. 
2.  **Extraction (`extraction/`)**: 
    *   **OCCI Method**: Sends contract text to Azure OpenAI (GPT-5). The LLM is prompted to return structured causal graphs. 
    *   **Baselines**: Simultaneously runs simple Regex pattern matching and unconstrained LLM extraction for comparison.
3.  **Validation (`ontology/`)**:
    *   The raw output from the LLM is passed through the `OntologyValidator`.
    *   It checks against `legal_ontology.py` to ensure edges obey the hierarchy (e.g., *Remedy* cannot cause *Obligation*) and that no cycles exist.
4.  **Adversarial Testing (`adversarial/`)**:
    *   **Decoys**: Inserts irrelevant sentences to see if the model hallucinates connections.
    *   **Perturbations**: Rephrases text to check if the model consistently extracts the same logic.
5.  **Metrics Calculation (`evaluation/metrics.py`)**:
    *   Computes scores like *Structural Correctness* (adherence to ontology), *Cycle Score* (acyclicity), and *True Positive Rate*.
6.  **Reporting**: Saves detailed JSON results and summary tables to the `results/` folder.

## Requirements

- Python 3.9+
- Azure OpenAI Service with GPT-5 deployment
- ~8GB RAM for CUAD dataset processing

## Installation

```bash
# Create virtual environment
python -m venv .venv
source .venv/bin/activate  # Linux/Mac
# or: .venv\Scripts\activate  # Windows

# Install dependencies
pip install -r requirements.txt
```

## Configuration

Set Azure OpenAI credentials via environment variables or in `config/settings.py` (or `.env`):

```bash
# Azure OpenAI (required)
export AZURE_OPENAI_ENDPOINT="https://your-resource.openai.azure.com/"
export AZURE_OPENAI_API_KEY="your-api-key"
export AZURE_OPENAI_GPT5_DEPLOYMENT="gpt-5"  # Your deployment name

# Optional: For ablation studies with other providers
export OPENAI_API_KEY="sk-..."
export ANTHROPIC_API_KEY="..."
```

## Usage

### 1. Validate Configuration

```bash
python main.py --validate
```

### 2. Run Full Experimental Evaluation

```bash
# Full evaluation on 100 contracts with adversarial tests
python main.py --run --contracts 100 --provider azure

# Without adversarial tests (faster)
python main.py --run --contracts 50 --no-adversarial
```

### 3. Generate Publication Figures

```bash
python main.py --figures --format pdf
```

### 4. Run Baseline Tests (No API Calls)

```bash
python main.py --baseline
```

## Evaluation Metrics Explained

The framework uses strict metrics to quantify causal reasoning quality:

| Metric | Full Name | What it measures | Calculation Focus |
|--------|-----------|------------------|-------------------|
| **SC** | **Structural Correctness** | Does the graph follow the rules? | Valid Edges / Total Edges. Checks if edges obey Type hierarchy (e.g. `Obligation -> Outcome`) and partial order. |
| **CS** | **Cycle Score** | Is the logic circular? | 1 - (Cyclic graphs / Total graphs). A perfect score (1.0) means no graphs contained logical loops (Contract DAGs). |
| **FPR** | **False Positive Rate** | Is it distracted by noise? | Accepted decoy edges / Total decoy edges. Measures hallucination rate on injected noise. |
| **TPR** | **True Positive Rate** | Did it find the right answers? | Correctly extraction edges / Ground truth edges (Recall). |
| **FPCR** | **False-Positive Causal Rate** | Is the causal link actually strong? | Spurious deterministic edges / Total edges. Checks if "deterministic" edges are backed by strong language like "shall cause". |
| **DRR** | **Decoy Rejection Rate** | Did it ignore the noise? | `(R_decoy - A_decoy) / (R_decoy + A_decoy)` where R=rejected, A=accepted. Range [-1, 1]: positive=good, negative=hallucinating. |
| **IA** | **Invariance Accuracy** | is it consistent? | Measures if rephrasing the same text yields the same graph structure. |
| **IR** | **Inversion Rate** | Does it understand "NOT"? | Measures if negating a sentence correctly changes the graph structure. |

## Experimental Design

### Methods Compared

1. **OCCI (Proposed)**: Ontology-constrained LLM extraction with GPT-5
2. **LLM Unconstrained**: GPT-5 without ontology guidance (Baseline)
3. **Pattern Baseline**: Regex-based causal pattern matching (Baseline)
4. **Rule Baseline**: Ontology templates without LLM (Baseline)

### Adversarial Challenge Tests

The adversarial tests implement the methodology from Section 7 of the paper:

#### 1. Decoy Injection (Negative Control Insertion)

Implements the paper's "Negative control insertion" methodology for hallucination detection:

- **Process**: Semantically neutral decoy sentences are **injected INTO** contract text (not tested in isolation)
- **Categories**: Lexical decoys (causal keywords used non-causally), neutral filler, temporal statements, definitions
- **Validation**: The `OntologyValidator` receives the list of `decoy_spans` and detects edges supported **only** by decoy text
- **Metrics**:
  - `R_decoy`: Edges rejected because grounded only in decoys (correct behavior)
  - `A_decoy`: Edges accepted despite being decoy-grounded (hallucinations)
  - `DRR = (R_decoy - A_decoy) / (R_decoy + A_decoy)` ∈ [-1, 1]

```
Example decoys:
- "The parties shall meet quarterly to discuss results." (lexical - uses 'shall' non-causally)
- "This Agreement may be executed in counterparts." (neutral filler)
- "The initial term shall commence on January 1, 2025." (temporal - no causation)
```

#### 2. Paraphrase Invariance

Validates structural stability: the same causal structure should be extracted from semantically equivalent paraphrased text.

- **Process**: Contracts are professionally redrafted to preserve semantics while varying surface form
- **Metric**: Jaccard similarity of edge sets between original and paraphrase
- **Expected**: High similarity (≥0.9) for deterministic rule edges

#### 3. Contradiction Detection

Validates sensitivity to semantic changes: edges should change when causal markers are inverted.

- **Process**: Causal markers are negated (e.g., "X causes Y" → "X does not cause Y")
- **Metric**: Inversion Rate (IR) - fraction of pairs where contradiction changed structure
- **Expected**: High IR indicates proper semantic understanding

## Transparency Logging

The framework provides comprehensive logging for audit, debugging, and reproducibility:

### Log Files

Each evaluation run creates separate log files in `results/logs/`:

| File | Description |
|------|-------------|
| `llm_calls_{run_id}.jsonl` | Full LLM prompts, responses, latency, and parsed counts |
| `extractions_{run_id}.jsonl` | All extraction results with variable/edge counts by type |
| `adversarial_{run_id}.jsonl` | Adversarial test details (decoy injections, paraphrases) |
| `summary_{run_id}.json` | Run statistics and log file locations |

### Log Contents

**LLM Call Logs** include:
- Timestamp, provider, model name
- Contract ID being processed
- Full prompt (with preview) and response text
- Number of variables/edges parsed
- API latency in seconds
- Success/failure status and error messages

**Extraction Logs** include:
- Method used (occi, llm_unconstrained, pattern, rule)
- Variable counts by type (Clause, Obligation, etc.)
- Edge counts by type pair (Clause→Obligation, etc.)
- Validation status and error list (for OCCI)

**Adversarial Logs** include:
- Test type (decoy_injection, paraphrase, contradiction)
- Number of decoys injected and their types
- Rejection/acceptance counts

### Accessing Logs

```python
from utils.extraction_logger import get_extraction_logger

# Get logger instance (creates if needed)
logger = get_extraction_logger(run_id="20260111_143000")

# Logs are in results/logs/
print(f"Log directory: {logger.log_dir}")
```


