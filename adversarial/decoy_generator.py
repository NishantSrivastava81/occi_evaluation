"""
Decoy sentence generator for adversarial testing.

Generates semantically neutral sentences that should NOT produce
causal edges if the ontology-constrained extractor works correctly.
"""

import random
from typing import List, Tuple
from dataclasses import dataclass


@dataclass
class DecoySentence:
    """A decoy sentence with metadata."""

    text: str
    decoy_type: str
    expected_edges: int  # Should be 0 for valid decoys
    description: str


class DecoyGenerator:
    """
    Generates decoy sentences for FPR testing.

    Categories:
    1. Lexical decoys: Use causal keywords non-causally
    2. Neutral filler: Business-neutral statements
    3. Temporal statements: Timeline mentions without causation
    4. Definitions: Legal definitions (no causal content)
    """

    # Lexical decoys - use causal words non-causally
    LEXICAL_DECOYS = [
        (
            "The parties shall meet quarterly to discuss results.",
            "lexical",
            "Uses 'shall' but non-causal",
        ),
        (
            "Failure to receive notice shall not affect the validity of this agreement.",
            "lexical",
            "Uses 'failure' but negated causation",
        ),
        (
            "The term 'breach' means any violation as defined herein.",
            "lexical",
            "Definition using 'breach' keyword",
        ),
        (
            "As a result of the foregoing recitals, the parties agree as follows.",
            "lexical",
            "Uses 'result' non-causally",
        ),
        (
            "The insurance shall cover all equipment on the premises.",
            "lexical",
            "Uses 'shall' but describes coverage, not causation",
        ),
        (
            "Therefore, this Agreement supersedes all prior agreements.",
            "lexical",
            "Uses 'therefore' but for supersession, not causation",
        ),
        (
            "Notwithstanding the foregoing, the parties may amend this agreement.",
            "lexical",
            "Legal boilerplate with causal-like structure",
        ),
        (
            "Consequently, all notices shall be in writing.",
            "lexical",
            "Uses 'consequently' but for procedural requirement",
        ),
    ]

    # Neutral filler - pure administrative content
    NEUTRAL_FILLER = [
        (
            "This Agreement is entered into as of the date first written above.",
            "neutral",
            "Date statement",
        ),
        (
            "The headings in this Agreement are for convenience only.",
            "neutral",
            "Formatting statement",
        ),
        (
            "This Agreement may be executed in counterparts.",
            "neutral",
            "Execution mechanics",
        ),
        (
            "All exhibits attached hereto are incorporated by reference.",
            "neutral",
            "Incorporation by reference",
        ),
        (
            "This Agreement shall be governed by the laws of the State of Delaware.",
            "neutral",
            "Choice of law",
        ),
        (
            "Any dispute shall be resolved through binding arbitration.",
            "neutral",
            "Dispute resolution mechanism",
        ),
        (
            "Neither party may assign this Agreement without prior written consent.",
            "neutral",
            "Assignment restriction",
        ),
        (
            "The waiver of any breach shall not constitute a waiver of any subsequent breach.",
            "neutral",
            "Waiver clause",
        ),
    ]

    # Temporal statements - time-related without causation
    TEMPORAL_DECOYS = [
        (
            "The initial term shall commence on January 1, 2025.",
            "temporal",
            "Start date only",
        ),
        (
            "This Agreement shall continue for a period of three years.",
            "temporal",
            "Duration statement",
        ),
        ("The renewal term shall be one year.", "temporal", "Renewal period"),
        (
            "Notice must be provided at least 30 days prior to termination.",
            "temporal",
            "Timing requirement",
        ),
        (
            "Payment shall be due on the first day of each month.",
            "temporal",
            "Payment timing",
        ),
        (
            "The warranty period shall extend for 12 months following delivery.",
            "temporal",
            "Warranty duration",
        ),
        (
            "Quarterly reports shall be submitted within 15 business days.",
            "temporal",
            "Reporting timeline",
        ),
    ]

    # Definitions - legal definitions without causal content
    DEFINITION_DECOYS = [
        (
            "'Affiliate' means any entity that controls, is controlled by, or is under common control with a party.",
            "definition",
            "Affiliate definition",
        ),
        (
            "'Confidential Information' means any information marked as confidential.",
            "definition",
            "Confidentiality definition",
        ),
        (
            "'Effective Date' means the date first written above.",
            "definition",
            "Effective date definition",
        ),
        (
            "'Material Adverse Change' has the meaning set forth in Section 5.2.",
            "definition",
            "Cross-reference definition",
        ),
        (
            "'Territory' means the United States and Canada.",
            "definition",
            "Geographic definition",
        ),
        (
            "'Force Majeure Event' includes acts of God, war, terrorism, and natural disasters.",
            "definition",
            "Force majeure definition",
        ),
    ]

    def __init__(self):
        self.all_decoys = (
            self.LEXICAL_DECOYS
            + self.NEUTRAL_FILLER
            + self.TEMPORAL_DECOYS
            + self.DEFINITION_DECOYS
        )

    def generate_decoy_set(
        self, n: int = 10, category: str = None
    ) -> List[DecoySentence]:
        """
        Generate a set of decoy sentences.

        Args:
            n: Number of decoys to generate
            category: Specific category ('lexical', 'neutral', 'temporal', 'definition')
                     or None for mixed

        Returns:
            List of DecoySentence objects
        """
        if category:
            pool = [d for d in self.all_decoys if d[1] == category]
        else:
            pool = self.all_decoys

        # Sample with replacement if needed
        if n > len(pool):
            selected = pool * (n // len(pool) + 1)
            selected = selected[:n]
        else:
            selected = random.sample(pool, min(n, len(pool)))

        return [
            DecoySentence(
                text=text, decoy_type=dtype, expected_edges=0, description=desc
            )
            for text, dtype, desc in selected
        ]

    def create_mixed_document(
        self,
        causal_sentences: List[str],
        decoy_ratio: float = 0.3,
        decoy_texts: List[str] = None,
    ) -> Tuple[str, List[int], List[str]]:
        """
        Create a document mixing causal and decoy sentences.

        Args:
            causal_sentences: List of sentences with actual causal content
            decoy_ratio: Ratio of decoys to total sentences
            decoy_texts: Optional pre-generated decoy texts to use

        Returns:
            (mixed_document, decoy_indices, decoy_texts) tuple
        """
        n_causal = len(causal_sentences)
        n_decoys = int(n_causal * decoy_ratio / (1 - decoy_ratio))

        # Use provided decoys or generate new ones
        if decoy_texts is None:
            decoys = self.generate_decoy_set(n_decoys)
            decoy_texts = [d.text for d in decoys]
        else:
            # Adjust n_decoys to match provided decoys
            n_decoys = len(decoy_texts)

        # Interleave causal and decoy sentences
        all_sentences = []
        decoy_indices = []

        causal_idx = 0
        decoy_idx = 0

        for i in range(n_causal + n_decoys):
            if decoy_idx < n_decoys and (
                causal_idx >= n_causal or random.random() < decoy_ratio
            ):
                all_sentences.append(decoy_texts[decoy_idx])
                decoy_indices.append(i)
                decoy_idx += 1
            else:
                all_sentences.append(causal_sentences[causal_idx])
                causal_idx += 1

        mixed_doc = " ".join(all_sentences)
        return mixed_doc, decoy_indices, decoy_texts

    def get_all_decoys(self) -> List[DecoySentence]:
        """Get all predefined decoy sentences."""
        return [
            DecoySentence(
                text=text, decoy_type=dtype, expected_edges=0, description=desc
            )
            for text, dtype, desc in self.all_decoys
        ]

    def generate_domain_specific_decoys(
        self, domain: str = "legal"
    ) -> List[DecoySentence]:
        """
        Generate domain-specific decoy sentences.

        Args:
            domain: Domain type ('legal', 'insurance', 'real_estate')
        """
        if domain == "insurance":
            return [
                DecoySentence(
                    text="The policy period shall be from 12:01 AM on the effective date.",
                    decoy_type="temporal",
                    expected_edges=0,
                    description="Insurance timing",
                ),
                DecoySentence(
                    text="Coverage applies to the Named Insured and its subsidiaries.",
                    decoy_type="neutral",
                    expected_edges=0,
                    description="Coverage scope",
                ),
                DecoySentence(
                    text="The deductible amount shall be $10,000 per occurrence.",
                    decoy_type="neutral",
                    expected_edges=0,
                    description="Deductible statement",
                ),
            ]
        elif domain == "real_estate":
            return [
                DecoySentence(
                    text="The premises shall include all fixtures and improvements.",
                    decoy_type="definition",
                    expected_edges=0,
                    description="Premises definition",
                ),
                DecoySentence(
                    text="The lease term shall commence on the delivery date.",
                    decoy_type="temporal",
                    expected_edges=0,
                    description="Lease timing",
                ),
            ]
        else:  # default legal
            return self.generate_decoy_set(10)
