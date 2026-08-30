"""Run the fixed ablation set against a node, cache per (node, component).

An ablation is a candidate run with a hand-crafted `instruction` in place of
an LLM-generated hypothesis. Reuses the same write→stage→execute machinery,
so a delta measured here is on the same footing as a normal candidate delta.

Cache key: (node.id, component). Ablations are stable transformations, so
the same (node, component) never needs to run twice — the writer prompt is
byte-identical and the underlying code_dir is byte-identical.
"""
from __future__ import annotations
import json
import os
import time
from typing import Dict, Optional

from codegen.ablations import ABLATIONS

ABLATIONS_LOG_PATH = "orchestrator/_state/ablations.jsonl"


def _cache_key(node_id: str, component: str) -> str:
    return f"{node_id}:{component}"


def load_cache(path: str | None = None) -> Dict[str, float]:
    """Rebuild {'node_id:component' -> delta} from the JSONL log.

    Cheap at hackathon scale to call each iteration. Guarded so a corrupt
    line does not crash the read.

    `path` resolves against the module global at CALL time rather than
    defaulting to it in the signature — same rule as driver._append_nodes_log:
    a signature default freezes the repo path at import, so a test that
    redirects ABLATIONS_LOG_PATH would read the live file anyway.
    """
    path = path or ABLATIONS_LOG_PATH
    if not os.path.exists(path):
        return {}
    out: Dict[str, float] = {}
    with open(path, encoding="utf-8") as fh:
        for line in fh:
            try:
                r = json.loads(line)
                if "delta" in r:
                    out[_cache_key(r["node_id"], r["component"])] = r["delta"]
            except (ValueError, KeyError):
                continue
    return out


def _append(record: dict, path: str | None = None) -> None:
    path = path or ABLATIONS_LOG_PATH
    os.makedirs(os.path.dirname(path), exist_ok=True)
    with open(path, "a", encoding="utf-8") as fh:
        fh.write(json.dumps(record, default=str) + "\n")


def run_ablations(node, node_primary: float, *,
                  codegen_mod, stage_fn, apply_diff_fn,
                  execute_seed: int = 0,
                  wallclock_cap_seconds: int = 300,
                  data_dir: str,
                  counters=None,
                  log_path: str | None = None) -> Dict[str, float]:
    """Ablate each registered component. Return {component -> delta}.

    delta = node_primary - ablated_primary. A small positive delta means
    the pipeline barely leans on that component.

    codegen_mod / stage_fn / apply_diff_fn are injected so the driver can
    pass its real bindings without importing driver here (which would
    circular-import). Callers pass driver.codegen, driver._apply_diff_and_stage,
    driver._apply_diff_to_dir.

    `apply_diff_fn` is accepted but unused: staging via `stage_fn` already
    applies the diff. It stays in the signature because an ablation that has
    to patch an already-staged dir (the repair path's shape) would need it,
    and the driver call site passes it today.

    `counters` is optional; when given, each completed ablation run is billed
    to `ablation_runs` and to the valid_search scorer budget. An ablation pass
    is three real scorer queries, and a budget report that doesn't see them
    understates what the search spent.

    Silent-skip an ablation on any error — a failed ablation is a bug in
    the strategy, not evidence about the pipeline.
    """
    cache = load_cache(log_path)
    deltas: Dict[str, float] = {}
    for name, abl in ABLATIONS.items():
        key = _cache_key(node.id, name)
        if key in cache:
            deltas[name] = cache[key]
            continue
        try:
            diff = codegen_mod.write_fix(
                hypothesis={"mechanism": abl.instruction,
                            "implementation_sketch": abl.instruction},
                target_component=abl.target,
                root=node.code_dir,
            )
            abl_dir = stage_fn(
                diff, root=node.code_dir,
                candidate_id=f"ablation_{node.id}_{name}",
            )
            if abl_dir is None:
                continue
            code_path = os.path.join(abl_dir, "baseline.py")
            t0 = time.time()
            res = codegen_mod.execute(
                code_path, seed=execute_seed, split="valid_search",
                wallclock_cap_seconds=wallclock_cap_seconds,
                root=abl_dir, data_dir=data_dir,
            )
            if counters is not None:
                counters.bump("ablation_runs")
                counters.bump_scorer("valid_search")
            if res["status"] != "ok":
                continue
            ablated_primary = res["metrics"]["primary"]
            delta = node_primary - ablated_primary
            deltas[name] = delta
            _append({
                "node_id": node.id,
                "component": name,
                "delta": delta,
                "ablated_primary": ablated_primary,
                "node_primary": node_primary,
                "wallclock_s": round(time.time() - t0, 1),
                "description": abl.description,
            }, log_path)
        except Exception as exc:  # pragma: no cover — defensive
            _append({
                "node_id": node.id, "component": name,
                "error": type(exc).__name__ + ": " + str(exc)[:200],
            }, log_path)
    return deltas


def pick_weakest_component(deltas: Dict[str, float],
                           min_meaningful_delta: float = 0.005
                           ) -> Optional[str]:
    """Return the component with the smallest delta; None when no delta is
    meaningful.

    A component's delta is `parent_primary - parent_without_component_primary`.
    Small = the pipeline barely depends on that component. Under MLE-STAR that
    is normally the strongest target for refinement — but "small" is only a
    signal when at least ONE component's delta is materially above noise.
    When max(deltas) <= min_meaningful_delta, the ablation set is telling us
    the pipeline is already well-tuned along every measured axis; there is no
    lever here. Return None so the driver falls through to a normal improve
    rather than refining a component picked essentially at random.

    The 0.005 default is set against the observed noise floor: the baseline's
    own 5-seed std on primary is ~0.0008, so a real component contribution
    should clear ~5σ ≈ 0.004. Rounded up for headroom.

    Ties broken deterministically by name so nodes.jsonl is reproducible.
    """
    if not deltas:
        return None
    if max(deltas.values()) <= min_meaningful_delta:
        return None
    return min(sorted(deltas.items()), key=lambda kv: kv[1])[0]
