"""
llm_calls: the LLM-calling layer for the autonomous ML research agent.

Public contract (frozen — do not change signatures without team sign-off):
    diagnose(node_context: dict) -> dict
    ground_in_literature(bottleneck: str) -> dict
    generate_hypothesis(diagnosis: dict, evidence_card: dict) -> list[dict]
    refine(component, component_source, ablations, iter_history,
           improvement_score, prior_refines=None) -> dict
    audit(diff: str, checklist: dict) -> dict
    verdict(hypothesis: dict, measured: dict, context: dict) -> dict

This module is fully self-contained: it only talks to a model API and
validates JSON. It has no knowledge of the dataset, the search tree, or
the execution sandbox, and can be developed/tested in isolation.
"""

from .exceptions import LLMSchemaError, LLMTruncatedError
from .diagnose import diagnose
from .literature import ground_in_literature
from .hypothesis import generate_hypothesis
from .audit import audit
from .refine import refine
from .verdict import verdict

__all__ = [
    "LLMSchemaError",
    "LLMTruncatedError",
    "diagnose",
    "ground_in_literature",
    "generate_hypothesis",
    "refine",
    "audit",
    "verdict",
]
