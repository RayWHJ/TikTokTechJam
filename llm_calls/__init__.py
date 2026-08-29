"""
llm_calls: the LLM-calling layer for the autonomous ML research agent.

Public contract (frozen — do not change signatures without team sign-off):
    diagnose(node_context: dict) -> dict
    ground_in_literature(bottleneck: str) -> dict
    generate_hypothesis(diagnosis: dict, evidence_card: dict) -> list[dict]
    audit(diff: str, checklist: dict) -> dict
    dedup_fingerprint_match(candidate_fingerprint: tuple, memory_entries: list) -> bool

This module is fully self-contained: it only talks to a model API and
validates JSON. It has no knowledge of the dataset, the search tree, or
the execution sandbox, and can be developed/tested in isolation.
"""

from .exceptions import LLMSchemaError
from .diagnose import diagnose
from .literature import ground_in_literature
from .hypothesis import generate_hypothesis
from .audit import audit
from .dedup import dedup_fingerprint_match

__all__ = [
    "LLMSchemaError",
    "diagnose",
    "ground_in_literature",
    "generate_hypothesis",
    "audit",
    "dedup_fingerprint_match",
]
