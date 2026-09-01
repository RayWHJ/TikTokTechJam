from __future__ import annotations

import json
from typing import Dict, List, Optional

from .client import call_model_text
from .routing import effort_for, model_for
from .usage import KIND_REFINE
from .personas import REFINER_SYSTEM_PROMPT
from .retry import call_with_schema_retry
from .schemas import validate_refinement


def _build_prompt(component: str, component_source: str,
                  ablations: Dict[str, float],
                  iter_history: List[float],
                  improvement_score: Optional[float],
                  prior_refines: List[dict]) -> str:
    return (
        f"Target component: {component}\n\n"
        f"Ablation deltas across the registered components (parent minus "
        f"parent_without_component; smaller means pipeline depends on it less):\n"
        f"{json.dumps(ablations, indent=2)}\n\n"
        f"iter_history (running-best primary per iteration; index 0 = baseline):\n"
        f"{json.dumps(iter_history, indent=2)}\n\n"
        f"improvement_score (iter_history[-1] - iter_history[-4], the ε/N "
        f"plateau signal; null if <3 iterations have completed): "
        f"{improvement_score}\n\n"
        f"Current implementation of the {component} component (verbatim):\n"
        f"```python\n{component_source}\n```\n\n"
        f"Prior refinement attempts on this component in this run "
        f"(may be empty):\n"
        f"{json.dumps(prior_refines, indent=2)}\n\n"
        f"Produce one refinement as a JSON object matching the required schema."
    )


def refine(component: str, component_source: str,
           ablations: Dict[str, float],
           iter_history: List[float],
           improvement_score: Optional[float],
           prior_refines: Optional[List[dict]] = None) -> Dict:
    """Ask the refiner persona for one component-scoped hypothesis."""
    prompt = _build_prompt(component, component_source, ablations,
                           iter_history, improvement_score,
                           prior_refines or [])

    def call_fn(p: str) -> str:
        return call_model_text(REFINER_SYSTEM_PROMPT, p,
                               model=model_for(KIND_REFINE),
                               effort=effort_for(KIND_REFINE),
                               kind=KIND_REFINE)

    return call_with_schema_retry(call_fn, prompt, validate_refinement)
