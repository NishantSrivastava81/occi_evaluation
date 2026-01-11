"""
Extraction Logger for OCCI Framework.

Provides transparent logging of all LLM calls and extractor outputs
for audit, debugging, and reproducibility purposes.

Each evaluation run creates a separate log file with:
- Timestamps for all operations
- Full LLM prompts and responses
- Extracted variables and edges
- Validation results
- Error traces
"""

import json
import logging
from datetime import datetime
from pathlib import Path
from typing import Any, Dict, List, Optional
from dataclasses import dataclass, asdict
import threading

# Thread-safe singleton pattern for logger instance
_logger_lock = threading.Lock()
_extraction_logger_instance: Optional["ExtractionLogger"] = None


@dataclass
class LLMCallLog:
    """Record of a single LLM API call."""

    timestamp: str
    provider: str
    model: str
    contract_id: str
    prompt_length: int
    prompt_preview: str  # First 500 chars
    response_length: int
    response_text: str
    parsed_variables: int
    parsed_edges: int
    latency_seconds: float
    success: bool
    error_message: Optional[str] = None


@dataclass
class ExtractorLog:
    """Record of an extraction operation (any method)."""

    timestamp: str
    method: str  # 'occi', 'llm_unconstrained', 'pattern', 'rule'
    contract_id: str
    input_length: int
    num_variables: int
    num_edges: int
    variable_types: Dict[str, int]  # Count per type
    edge_types: Dict[str, int]  # Count per type pair
    extraction_time: float
    validation_passed: Optional[bool] = None
    validation_errors: Optional[List[str]] = None


@dataclass
class AdversarialTestLog:
    """Record of an adversarial test."""

    timestamp: str
    test_type: str  # 'decoy', 'paraphrase', 'contradiction'
    contract_id: str
    details: Dict[str, Any]


class ExtractionLogger:
    """
    Centralized logger for all extraction operations.

    Creates structured JSON log files per evaluation run for:
    - Full transparency of LLM interactions
    - Reproducibility verification
    - Debugging extraction issues
    - Audit trail for research

    Usage:
        logger = ExtractionLogger.get_instance(run_id="20260111_143000")
        logger.log_llm_call(...)
        logger.log_extraction(...)
        logger.finalize()  # Write summary and close
    """

    def __init__(self, run_id: str, log_dir: Path = None):
        """
        Initialize the extraction logger.

        Args:
            run_id: Unique identifier for this evaluation run
            log_dir: Directory for log files (defaults to results/logs/)
        """
        self.run_id = run_id
        self.start_time = datetime.now()

        # Set up log directory
        if log_dir is None:
            from config.settings import RESULTS_DIR

            log_dir = RESULTS_DIR / "logs"
        self.log_dir = Path(log_dir)
        self.log_dir.mkdir(parents=True, exist_ok=True)

        # Log file paths
        self.llm_log_path = self.log_dir / f"llm_calls_{run_id}.jsonl"
        self.extraction_log_path = self.log_dir / f"extractions_{run_id}.jsonl"
        self.adversarial_log_path = self.log_dir / f"adversarial_{run_id}.jsonl"
        self.summary_path = self.log_dir / f"summary_{run_id}.json"

        # Counters for summary
        self.llm_call_count = 0
        self.extraction_count = 0
        self.adversarial_test_count = 0
        self.error_count = 0

        # Initialize log files with headers
        self._init_log_files()

        logging.getLogger(__name__).info(
            f"ExtractionLogger initialized: run_id={run_id}, log_dir={self.log_dir}"
        )

    def _init_log_files(self):
        """Initialize log files with metadata headers."""
        metadata = {
            "run_id": self.run_id,
            "start_time": self.start_time.isoformat(),
            "log_version": "1.0",
            "framework": "OCCI Evaluation",
        }

        for path in [
            self.llm_log_path,
            self.extraction_log_path,
            self.adversarial_log_path,
        ]:
            with open(path, "w", encoding="utf-8") as f:
                f.write(json.dumps({"_metadata": metadata}) + "\n")

    @classmethod
    def get_instance(
        cls, run_id: str = None, log_dir: Path = None
    ) -> "ExtractionLogger":
        """
        Get or create the singleton logger instance.

        Args:
            run_id: Run identifier (required for first call)
            log_dir: Optional log directory

        Returns:
            ExtractionLogger instance
        """
        global _extraction_logger_instance

        with _logger_lock:
            if _extraction_logger_instance is None:
                if run_id is None:
                    run_id = datetime.now().strftime("%Y%m%d_%H%M%S")
                _extraction_logger_instance = cls(run_id, log_dir)
            return _extraction_logger_instance

    @classmethod
    def reset_instance(cls):
        """Reset the singleton instance (for new evaluation runs)."""
        global _extraction_logger_instance
        with _logger_lock:
            if _extraction_logger_instance is not None:
                _extraction_logger_instance.finalize()
            _extraction_logger_instance = None

    def log_llm_call(
        self,
        provider: str,
        model: str,
        contract_id: str,
        prompt: str,
        response: str,
        parsed_variables: int,
        parsed_edges: int,
        latency: float,
        success: bool = True,
        error: str = None,
    ):
        """
        Log an LLM API call with full details.

        Args:
            provider: LLM provider (azure, openai, anthropic)
            model: Model name/deployment
            contract_id: Identifier for the contract being processed
            prompt: Full prompt sent to LLM
            response: Full response from LLM
            parsed_variables: Number of variables extracted
            parsed_edges: Number of edges extracted
            latency: API call duration in seconds
            success: Whether the call succeeded
            error: Error message if failed
        """
        log_entry = LLMCallLog(
            timestamp=datetime.now().isoformat(),
            provider=provider,
            model=model,
            contract_id=contract_id,
            prompt_length=len(prompt),
            prompt_preview=prompt[:500] + "..." if len(prompt) > 500 else prompt,
            response_length=len(response),
            response_text=response,
            parsed_variables=parsed_variables,
            parsed_edges=parsed_edges,
            latency_seconds=latency,
            success=success,
            error_message=error,
        )

        with open(self.llm_log_path, "a", encoding="utf-8") as f:
            f.write(json.dumps(asdict(log_entry), default=str) + "\n")

        self.llm_call_count += 1
        if not success:
            self.error_count += 1

    def log_extraction(
        self,
        method: str,
        contract_id: str,
        input_text: str,
        variables: List[Any],
        edges: List[Any],
        extraction_time: float,
        validation_passed: bool = None,
        validation_errors: List[str] = None,
    ):
        """
        Log an extraction operation result.

        Args:
            method: Extraction method name
            contract_id: Contract identifier
            input_text: Input contract text
            variables: Extracted Variable objects
            edges: Extracted Edge objects
            extraction_time: Time taken for extraction
            validation_passed: Whether validation passed (for OCCI)
            validation_errors: List of validation error messages
        """
        # Count variable types
        var_types = {}
        for var in variables:
            type_name = (
                var.var_type.value if hasattr(var, "var_type") else str(type(var))
            )
            var_types[type_name] = var_types.get(type_name, 0) + 1

        # Count edge types
        edge_types = {}
        for edge in edges:
            if hasattr(edge, "source") and hasattr(edge, "target"):
                src_type = (
                    edge.source.var_type.value
                    if hasattr(edge.source, "var_type")
                    else "unknown"
                )
                tgt_type = (
                    edge.target.var_type.value
                    if hasattr(edge.target, "var_type")
                    else "unknown"
                )
                key = f"{src_type}->{tgt_type}"
            else:
                key = "unknown"
            edge_types[key] = edge_types.get(key, 0) + 1

        log_entry = ExtractorLog(
            timestamp=datetime.now().isoformat(),
            method=method,
            contract_id=contract_id,
            input_length=len(input_text),
            num_variables=len(variables),
            num_edges=len(edges),
            variable_types=var_types,
            edge_types=edge_types,
            extraction_time=extraction_time,
            validation_passed=validation_passed,
            validation_errors=validation_errors,
        )

        with open(self.extraction_log_path, "a", encoding="utf-8") as f:
            f.write(json.dumps(asdict(log_entry), default=str) + "\n")

        self.extraction_count += 1

    def log_adversarial_test(
        self, test_type: str, contract_id: str, details: Dict[str, Any]
    ):
        """
        Log an adversarial test result.

        Args:
            test_type: Type of test (decoy, paraphrase, contradiction)
            contract_id: Contract identifier
            details: Test-specific details
        """
        log_entry = AdversarialTestLog(
            timestamp=datetime.now().isoformat(),
            test_type=test_type,
            contract_id=contract_id,
            details=details,
        )

        with open(self.adversarial_log_path, "a", encoding="utf-8") as f:
            f.write(json.dumps(asdict(log_entry), default=str) + "\n")

        self.adversarial_test_count += 1

    def finalize(self):
        """
        Finalize logging and write summary file.

        Call this at the end of an evaluation run.
        """
        end_time = datetime.now()
        duration = (end_time - self.start_time).total_seconds()

        summary = {
            "run_id": self.run_id,
            "start_time": self.start_time.isoformat(),
            "end_time": end_time.isoformat(),
            "duration_seconds": duration,
            "statistics": {
                "llm_calls": self.llm_call_count,
                "extractions": self.extraction_count,
                "adversarial_tests": self.adversarial_test_count,
                "errors": self.error_count,
            },
            "log_files": {
                "llm_calls": str(self.llm_log_path),
                "extractions": str(self.extraction_log_path),
                "adversarial": str(self.adversarial_log_path),
            },
        }

        with open(self.summary_path, "w", encoding="utf-8") as f:
            json.dump(summary, f, indent=2)

        logging.getLogger(__name__).info(
            f"ExtractionLogger finalized: {self.llm_call_count} LLM calls, "
            f"{self.extraction_count} extractions, {self.error_count} errors. "
            f"Logs saved to {self.log_dir}"
        )


def get_extraction_logger(run_id: str = None) -> ExtractionLogger:
    """
    Convenience function to get the extraction logger.

    Args:
        run_id: Optional run identifier

    Returns:
        ExtractionLogger instance
    """
    return ExtractionLogger.get_instance(run_id)
