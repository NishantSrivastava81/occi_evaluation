"""
Perturbation challenge tests for robustness evaluation.

Tests:
1. Paraphrase consistency - Same meaning, different wording
2. Contradiction detection - Claims that contradict extracted structure
3. Noise injection - Irrelevant content insertion
"""

import re
import random
from typing import List, Tuple, Dict, Optional
from dataclasses import dataclass

import sys
from pathlib import Path

sys.path.append(str(Path(__file__).parent.parent))

from ontology.legal_ontology import Variable, Edge


@dataclass
class PerturbationResult:
    """Result of a perturbation challenge."""

    original_text: str
    perturbed_text: str
    perturbation_type: str
    expected_change: str  # 'none', 'modified', 'contradicted'
    description: str


class PerturbationChallenge:
    """
    Generates perturbations for robustness testing.

    Tests two key properties:
    1. Invariance: Paraphrases should yield same SCM structure
    2. Contradiction: Negations should yield different/no edges
    """

    # Paraphrase mappings for legal language
    PARAPHRASE_RULES = [
        # Modal verbs
        (r"\bshall\b", ["must", "will", "is required to", "has the obligation to"]),
        (r"\bmay\b", ["is permitted to", "has the right to", "can", "is entitled to"]),
        (r"\bshall not\b", ["must not", "is prohibited from", "may not", "cannot"]),
        # Causal connectives
        (r"\btherefore\b", ["consequently", "as a result", "thus", "hence"]),
        (r"\bbecause\b", ["since", "as", "due to the fact that", "given that"]),
        (r"\bif\b", ["in the event that", "should", "where", "in case"]),
        # Legal terms
        (
            r"\bbreach\b",
            ["violation", "default", "non-compliance", "failure to comply"],
        ),
        (r"\bterminate\b", ["end", "cancel", "discontinue", "cease"]),
        (r"\bliable\b", ["responsible", "accountable", "answerable"]),
        (r"\bdamages\b", ["compensation", "monetary relief", "financial recovery"]),
        # Time references
        (r"\bupon\b", ["on", "when", "at the time of", "following"]),
        (r"\bprior to\b", ["before", "in advance of", "preceding"]),
        (r"\bsubsequent to\b", ["after", "following", "succeeding"]),
    ]

    # Contradiction patterns
    CONTRADICTION_TEMPLATES = [
        # Negate obligation
        (r"(\w+)\s+shall\s+(\w+)", r"\1 shall not \2"),
        # Negate consequence
        (r"shall\s+constitute\s+(?:a\s+)?(\w+)", r"shall not constitute a \1"),
        # Reverse causation
        (r"if\s+([^,]+),\s+then\s+([^.]+)", r"if \2, then \1"),
        # Negate liability
        (r"shall\s+be\s+liable", r"shall not be liable"),
        (r"is\s+entitled\s+to", r"is not entitled to"),
    ]

    def __init__(self):
        self.compiled_paraphrases = [
            (re.compile(p, re.IGNORECASE), alternatives)
            for p, alternatives in self.PARAPHRASE_RULES
        ]
        self.compiled_contradictions = [
            (re.compile(p, re.IGNORECASE), repl)
            for p, repl in self.CONTRADICTION_TEMPLATES
        ]

    def generate_paraphrase(
        self, text: str, n_substitutions: int = 2
    ) -> PerturbationResult:
        """
        Generate a paraphrased version of the text.

        The paraphrase should yield the SAME SCM structure.
        """
        perturbed = text
        substitutions_made = 0

        for pattern, alternatives in self.compiled_paraphrases:
            if substitutions_made >= n_substitutions:
                break

            match = pattern.search(perturbed)
            if match:
                # Choose random alternative
                replacement = random.choice(alternatives)

                # Preserve case
                if match.group().isupper():
                    replacement = replacement.upper()
                elif match.group()[0].isupper():
                    replacement = replacement.capitalize()

                perturbed = pattern.sub(replacement, perturbed, count=1)
                substitutions_made += 1

        return PerturbationResult(
            original_text=text,
            perturbed_text=perturbed,
            perturbation_type="paraphrase",
            expected_change="none",
            description=f"Applied {substitutions_made} paraphrase substitutions",
        )

    def generate_contradiction(self, text: str) -> Optional[PerturbationResult]:
        """
        Generate a contradicted version of the text.

        The contradiction should yield DIFFERENT or NO edges.
        """
        for pattern, replacement in self.compiled_contradictions:
            if pattern.search(text):
                perturbed = pattern.sub(replacement, text, count=1)
                return PerturbationResult(
                    original_text=text,
                    perturbed_text=perturbed,
                    perturbation_type="contradiction",
                    expected_change="contradicted",
                    description="Applied negation/reversal to causal structure",
                )

        return None

    def inject_noise(self, text: str, noise_ratio: float = 0.2) -> PerturbationResult:
        """
        Inject irrelevant noise into the text.

        Structure should be preserved despite noise.
        """
        noise_sentences = [
            "The weather was particularly pleasant that day.",
            "Several attendees noted the comfortable temperature.",
            "The conference room had been recently renovated.",
            "Coffee and refreshments were provided.",
            "The document was printed on recycled paper.",
            "A brief recess was taken at 10:30 AM.",
        ]

        sentences = re.split(r"([.!?]\s+)", text)
        n_noise = max(1, int(len(sentences) * noise_ratio))

        noise_to_insert = random.sample(
            noise_sentences, min(n_noise, len(noise_sentences))
        )

        # Insert noise at random positions
        result_parts = []
        noise_idx = 0

        for i, part in enumerate(sentences):
            result_parts.append(part)
            if noise_idx < len(noise_to_insert) and random.random() < noise_ratio:
                result_parts.append(" " + noise_to_insert[noise_idx] + " ")
                noise_idx += 1

        perturbed = "".join(result_parts)

        return PerturbationResult(
            original_text=text,
            perturbed_text=perturbed,
            perturbation_type="noise",
            expected_change="none",
            description=f"Injected {noise_idx} noise sentences",
        )

    def evaluate_invariance(
        self, original_edges: List[Edge], perturbed_edges: List[Edge]
    ) -> Dict:
        """
        Evaluate if paraphrase maintained same structure.

        Returns invariance metrics.
        """

        def edge_signature(e: Edge) -> Tuple:
            return (
                e.source.var_type.value,
                e.target.var_type.value,
                e.is_deterministic,
            )

        orig_signatures = set(edge_signature(e) for e in original_edges)
        pert_signatures = set(edge_signature(e) for e in perturbed_edges)

        common = orig_signatures & pert_signatures

        # Invariance metrics
        precision = len(common) / len(pert_signatures) if pert_signatures else 1.0
        recall = len(common) / len(orig_signatures) if orig_signatures else 1.0
        f1 = (
            2 * precision * recall / (precision + recall)
            if (precision + recall) > 0
            else 0.0
        )

        return {
            "invariance_precision": precision,
            "invariance_recall": recall,
            "invariance_f1": f1,
            "original_edges": len(original_edges),
            "perturbed_edges": len(perturbed_edges),
            "common_edges": len(common),
        }

    def evaluate_contradiction_detection(
        self, original_edges: List[Edge], contradicted_edges: List[Edge]
    ) -> Dict:
        """
        Evaluate if contradiction was properly detected.

        Returns contradiction detection metrics.
        """

        def edge_signature(e: Edge) -> Tuple:
            return (e.source.var_type.value, e.target.var_type.value)

        orig_sigs = set(edge_signature(e) for e in original_edges)
        cont_sigs = set(edge_signature(e) for e in contradicted_edges)

        # Contradiction should produce different edges
        # Higher difference = better detection
        changed = orig_sigs.symmetric_difference(cont_sigs)
        total = orig_sigs.union(cont_sigs)

        change_ratio = len(changed) / len(total) if total else 0.0

        return {
            "change_ratio": change_ratio,
            "edges_changed": len(changed),
            "edges_preserved": len(orig_sigs & cont_sigs),
            "contradiction_detected": change_ratio > 0.5,
        }

    def generate_challenge_set(
        self,
        texts: List[str],
        include_paraphrase: bool = True,
        include_contradiction: bool = True,
        include_noise: bool = True,
    ) -> List[PerturbationResult]:
        """
        Generate a full challenge set for multiple texts.
        """
        challenges = []

        for text in texts:
            if include_paraphrase:
                challenges.append(self.generate_paraphrase(text))

            if include_contradiction:
                contra = self.generate_contradiction(text)
                if contra:
                    challenges.append(contra)

            if include_noise:
                challenges.append(self.inject_noise(text))

        return challenges
