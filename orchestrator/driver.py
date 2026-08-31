"""Main orchestrator loop. Swap the mock imports below when real PRs land."""
from dotenv import load_dotenv
load_dotenv()

import os
import json
import hashlib
import re
import time
import uuid
import tempfile
import subprocess
import shutil
from collections import Counter
from dataclasses import asdict
from typing import List

# ── ONE-LINE SWAP POINT ───────────────────────────────────────────────
# from .mocks import harness, llm, codegen

import harness
from llm_calls import (diagnose, ground_in_literature, generate_hypothesis,
                       refine, audit, verdict)
from llm_calls.families import (ALL_FAMILIES as _FAMILY_NAMES,
                                MECHANISM_FAMILIES, OTHER,
                                family_from_text, normalise_declaration)
import codegen
from codegen.ablations import ABLATIONS


class _LLM:
    diagnose = staticmethod(diagnose)
    ground_in_literature = staticmethod(ground_in_literature)
    generate_hypothesis = staticmethod(generate_hypothesis)
    refine = staticmethod(refine)
    audit = staticmethod(audit)
    verdict = staticmethod(verdict)
llm = _LLM()

from .node import Node
from .memory import Memory, EvidenceEntry
from .selection import select
from .triage import rank
# should_continue_locally is deliberately NOT imported any more. It still lives
# in promotion.py and is still the right rule for a significance claim, but the
# driver's only two decisions are parent acceptance (a hill-climb step, hence
# should_expand_as_parent) and promotion (checked inline against
# PROMOTE_TRIGGER_P_POS). Leaving the import in place would suggest a third
# call site that no longer exists.
from .promotion import (bootstrap_delta, should_expand_as_parent,
                        should_promote_globally)
from .convergence import local_plateau, global_should_stop
from .counters import Counters
from . import ablation_harness
from .ablation_harness import (run_ablations, pick_weakest_component,
                               load_cache as _load_ablation_cache,
                               _cache_key as _ablation_cache_key)

# Toggle from CLI. When False, diff bodies aren't printed each iteration —
# keeps run.log readable across a 50-iter run.
SHOW_DIFFS = False

# Per-iteration score record, rewritten after every iteration so an overnight
# run can be inspected mid-flight instead of waiting for run() to return.
PROGRESS_PATH = "orchestrator/_state/progress.json"

# Append-only per-node record: hypothesis, diagnosis, resulting metrics and
# error/recovery events, one JSON object per line. Kept separate from
# progress.json (which is rewritten whole each iteration and holds aggregates).
NODES_LOG_PATH = "orchestrator/_state/nodes.jsonl"

# Measured once per machine and reused: the unmodified baseline's per-user scores,
# which every candidate is paired against. ~20s per seed, so this is cheap.
ROOT_BASELINE_PATH = "orchestrator/_state/root_baseline.json"

# The Devpost-style write-up, synthesised from the run log at the end of a run.
# `codegen.synthesize_report` existed, was tested, and was wired into nothing
# (T3.5) — so a required deliverable was being produced by hand from a file the
# search wrote. One model call at the end of a run is the cheapest of the four
# dead call sites to make real, and it is the only one that produces a
# deliverable.
REPORT_PATH = "orchestrator/_state/report.md"
ROOT_SEEDS = (0, 1, 2)

# The baseline scored on the SEALED valid_confirm split, measured lazily on the
# first promotion attempt and cached. Promotion used to compare a candidate's
# valid_confirm primary against a global_best carried on valid_search — two
# different splits, so the "delta" mixed a real effect with the level difference
# between splits and meant nothing. A promotion test needs same-split baseline
# per-user scores to pair against.
CONFIRM_BASELINE_PATH = "orchestrator/_state/root_confirm_baseline.json"
CONFIRM_SEEDS = (0,)

# Promotion trigger: spend a sealed valid_confirm query only when the candidate
# is significantly better than its parent on the paired per-user bootstrap AND
# its scalar clears the current champion. The old trigger was
# `local_best_score > global_best + 0.003` — four times larger than the best
# delta this search has ever produced (+0.0011), so no candidate was ever
# eligible for a confirm run at all.
PROMOTE_TRIGGER_P_POS = 0.9

# Parent acceptance (see promotion.should_expand_as_parent). Separate from the
# promotion gate above on purpose: promotion is a claim tested on a sealed
# split, parent acceptance is one step of a hill climb. The old code used the
# significance test for both, so a +0.00059 candidate that missed p_pos=0.8 by
# 0.024 was discarded as a parent and the tree never left the root.
HILLCLIMB_MIN_MEAN_DELTA = 0.0
HILLCLIMB_MIN_P_POSITIVE = 0.5

# Cap on the frontier. Unbounded open_nodes makes select()'s UCT term diffuse
# over branches that already lost; keeping the best few by scalar primary
# concentrates the remaining iterations on the live ones.
MAX_OPEN_NODES = 4

# How many candidates may reach _attempt() in one iteration, after dedup, the
# numpy-feasibility check and family diversity have filtered the batch.
#
# The asymmetry this exploits: a hypothesis costs ~300 output tokens, a candidate
# costs a writer call (~5.5k in / 4.3k out), an audit call (~4.5k in) and a triage
# run. So the PROPOSAL distribution can be widened two orders of magnitude more
# cheaply than the compute bill. Propose 6-8, execute at most 4.
#
# Budget check: 4 triage runs plus 3 survivors x 2 seeds at ~60s each is roughly
# 10 minutes per iteration, about 36 iterations inside the 6h ceiling.
MAX_CANDIDATES_PER_ITER = 4

# Mechanisms whose named primitive cannot be written in numpy on one core inside
# the triage cap. Checked against the mechanism AND the implementation sketch.
#
# Not a style preference: 61 of the 65 stored proposals from earlier runs named
# something in this list — MAML and other meta-learning, ColdNAS and neural
# architecture search, DeepFM / xDeepFM, contrastive objectives, SAM/ASAM,
# frequency-decomposed state-space models, a small LLM for token augmentation —
# a ~94% unimplementable rate WITH the constraint block already in the persona.
# Rejecting them before a writer call is spent costs nothing and saves ~5.5k
# input + 4.3k output tokens each.
_INFEASIBLE_TOKENS: tuple[str, ...] = (
    "maml", "meta-learning", "meta learning", "coldnas",
    "neural architecture search", "architecture search",
    "deepfm", "xdeepfm", "transformer", "attention layer", "self-attention",
    "state-space", "state space model", "contrastive", "sam optimizer", "asam",
    "torch", "pytorch", "tensorflow", "keras", "sklearn", "scikit-learn",
    "pandas", "huggingface", "pretrained", "pre-trained", "embedding model",
    "large language model", "llm-based", "gnn", "graph neural",
    "variational autoencoder", "diffusion model", "reinforcement learning",
)

# Where the best-scoring candidate's source is archived. The staged candidate
# dirs live under tempfile.gettempdir() and are not durable — on the run that
# produced orchestrator/_state/ they were /var/folders/... which macOS clears.
# Nothing copied the winner out, so even a candidate that beat the baseline
# left no artifact to generate a submission from.
CHAMPION_DIR = os.path.join("orchestrator", "_state", "champions")

# Two candidate per-user vectors closer than this are the same computation.
# A rewrite that applies cleanly but leaves the executed code path untouched
# reproduces its parent's score exactly, to the last bit of the float.
NOOP_EPSILON = 1e-9

# Repair attempts per candidate. A candidate that fails, gets repaired to a
# still-broken state and fails again would otherwise loop forever; two attempts
# is enough at hackathon scale, since the second retry rarely helps if the
# first didn't.
MAX_FIX_ATTEMPTS = 2

# Repair attempts against the SMOKE stage, which costs ~0.06s on 200 synthetic
# rows instead of 240s on 1.4M. Higher than MAX_FIX_ATTEMPTS because the two
# budgets buy different things: this one buys attempts at a nearly-free signal,
# and only code that already runs reaches the expensive one. Raising the attempt
# budget without raising the bill is the entire point of the stage.
MAX_SMOKE_FIX_ATTEMPTS = 5

# Wall-clock cap for the 1-seed triage run. The unmodified baseline trains and
# scores valid_search in 18s measured on this machine (early-stopping at epoch
# 11 at ~1.1s/epoch), so the old 120s was only 6.7x the baseline — enough to
# kill any candidate that trains more epochs, runs a second backward pass
# (pairwise/listwise losses do), or adds a per-user aggregation. A correct but
# 8x-slower mechanism was being recorded as a failed implementation and its
# hypothesis thrown away. 240s is 13x the baseline and still well under the
# 600s full-seed cap, so a candidate that clears triage cannot then time out
# on seeds 1 and 2.
# ADAPTIVE triage cap (T2.11). Replaces a flat TRIAGE_WALLCLOCK_CAP_S = 240. cap = clamp(TRIAGE_RUNTIME_MULTIPLE * parent's own
# last clean runtime, TRIAGE_CAP_MIN_S, TRIAGE_CAP_MAX_S).
#
# The flat constant measured tolerance against the ROOT, forever. The baseline
# runs in ~40s, so a flat 240s already tolerates a 6x slowdown and a flat 500s
# tolerates 12x — raising the constant buys tolerance for inefficiency rather
# than for genuinely expensive mechanisms. Scaling off the parent's own cost is
# what starts to matter once a promoted candidate is itself slower than the root:
# under a flat cap, every child of it inherits a budget measured against
# something cheaper than its own parent.
#
# Paired with the T1.6 smoke stage the cap is only ever PAID by code already
# proven to run correctly on 200 rows, so a raise costs measurement time instead
# of debugging time.
#
# The ceiling stays at the full-run cap (600s) so nothing can clear triage and
# then time out at higher fidelity — the failure mode that would waste the most.
TRIAGE_RUNTIME_MULTIPLE = 4
TRIAGE_CAP_MIN_S = 300
TRIAGE_CAP_MAX_S = 600

# Wall-clock cap for a full-seed run. Also the ceiling on the adaptive triage
# cap, for the reason above.
FULL_RUN_WALLCLOCK_CAP_S = 600

# Plateau stop calibration. Deliberately NOT convergence.local_plateau's own
# defaults (ε=0.002, N=3), which are unreachable on this task.
#
# iter_history is a MONOTONE running best, so local_plateau's
# `max(h[-N:]) - max(h[:-N])` collapses to exactly `h[-1] - h[-4]` at N=3. The
# rule therefore demands a BRAND-NEW >0.002 jump in the running best every three
# iterations, forever — a gain can never count twice, because three iterations
# later it is itself the `h[-4]` being subtracted. Against a baseline whose
# 5-seed std is 0.0008 and a search whose largest single-candidate gain ever is
# +0.0031, that bar cannot be held: the 4-iteration run cleared the iteration-3
# check by 8.8e-6 and then stopped with the window delta at exactly 0.0, having
# used 28 minutes of a 6-hour budget.
#
# ε=0.0005 sits below the 0.0008 seed std, so the rule now reads "no improvement
# larger than measurement noise". N=8 widens the window so a branch gets several
# iterations to pay off before the run is judged, and moves the earliest possible
# stop from iteration 3 to iteration 8.
#
# Note this inverts the relationship with PLATEAU_REFINE_THRESHOLD (0.001): the
# stop bar is now BELOW the refine trigger, so a plateauing run reaches an
# ablation-guided refine instead of being killed before it. That was the intent
# all along — tests/test_mlestar.py::_tighter_stop_bar and
# tests/test_refine_trap_fixes.py both had to monkeypatch ε=0.0005 to observe it.
#
# Kept as driver constants rather than changed in convergence.py because
# local_plateau is a generic utility whose defaults
# orchestrator/tests/test_smoke.py::test_local_plateau_rule pins directly.
PLATEAU_STOP_EPSILON = 0.0005
PLATEAU_STOP_WINDOW_N = 8

# Phase 2 (MLE-STAR ablation-guided refinement) kill switch.
#
# OFF: the search measurably regressed once refine was in the loop, so the
# driver is back to the Phase 1 behaviour — improve every iteration, no
# ablation passes, no refine nodes, and no ablation evidence in the
# diagnostician's context. Nothing is deleted: the registry, the harness, the
# refiner prompt and their tests are all still here, and flipping this to True
# restores Phase 2 exactly as it was. The Phase 2 test modules skip themselves
# while this is False.
#
# Everything below this line under "Refine ..." is inert until it flips.
REFINE_ENABLED = False

# Refine cadence. Fixed floor: every K improve iterations, the next
# iteration turns into an ablation pass + one component-scoped refine.
REFINE_EVERY_K_IMPROVES = 3

# Refine ceiling: if improvement_score (iter_history[-1] - iter_history[-4])
# drops at or below this threshold, refine fires on the NEXT iteration
# regardless of how many improves have accumulated. The threshold is well
# below ε=0.002 (the local_plateau bar) so refine reacts before the run
# actually converges and stops.
PLATEAU_REFINE_THRESHOLD = 0.001

# Refine gate: a node only earns an ablation pass once its own evolution has
# produced measurable headroom over the baseline. Matched to
# PLATEAU_REFINE_THRESHOLD's scale — comfortably above the baseline's ~0.0008
# 5-seed std, low enough that a marginal-but-real improvement unlocks refine.
REFINE_TARGET_MIN_IMPROVEMENT = 0.0005

# The registry's component names. Imported from codegen.ablations rather than
# read off the `codegen` binding above, which --mock swaps for a stub that has
# no registry.
_ABLATION_COMPONENTS = tuple(ABLATIONS)

# codegen.execute defaults data_dir to <root>/KuaiRand-Pure/data, and we pass the
# per-candidate STAGED dir as root — which holds only the patched .py files, no
# dataset. Without passing this explicitly every staged candidate dies with
# FileNotFoundError on video_features_basic_pure.csv before running a line of
# the modification.
DATA_DIR = os.environ.get("CODEGEN_DATA_DIR",
                          os.path.abspath(os.path.join("KuaiRand-Pure", "data")))

# Published FM baseline, used only as a last-resort fallback when the root
# measurement fails. It is the TEST primary from the README, not valid_search.
FALLBACK_ROOT_PRIMARY = 0.5946


class _NullLedger:
    """Stand-in when llm_calls.usage is unavailable. Reports nothing, honestly."""

    def reset(self):
        pass

    def totals(self):
        return {"calls": 0, "tokens_in": 0, "tokens_cached": 0, "tokens_out": 0,
                "tokens_reasoning": 0, "web_searches": 0,
                "calls_without_usage": 0, "tokens_total": 0,
                "estimated_cost_usd": 0.0, "by_kind": {}}


def _usage_ledger():
    """The shared token ledger, or a no-op stand-in.

    Resolved at CALL time rather than imported at module scope so accounting can
    never make this module unimportable, and so a test can swap the ledger.
    """
    try:
        from llm_calls.usage import LEDGER
        return LEDGER
    except Exception:                           # noqa: BLE001 — see docstring
        return _NullLedger()


def _using_mocks() -> bool:
    """True when --mock (or a test) swapped the module-level bindings.

    Read off the bound module's own name rather than a flag, so it is correct
    however the swap was performed — the CLI rebinds these globals and eleven
    test modules monkeypatch them.
    """
    return getattr(codegen, "__name__", "").startswith("orchestrator.mocks")


def _model_report() -> dict:
    """Everything about model routing this run will bill to, as data.

    Structured rather than printed so three callers can share it: the startup
    banner, `--check-models`, and progress.json (T2.4). Every field is read
    defensively — this is reporting, and reporting must never be the reason a
    6-hour unattended run fails to start.
    """
    rep: dict = {"using_mocks": _using_mocks()}

    # WRITER. Two values, deliberately: what the environment ASKS for, and what
    # the backend will ACTUALLY call. `_auto_backend` silently returns
    # FakeBackend when OPENAI_API_KEY is unset or the SDK import fails, so
    # reporting only the first announces a frontier model while canned
    # single-token edits are served — which is strictly worse than reporting
    # nothing, because it reads like the run is working.
    try:
        from codegen.llm_client import (DEFAULT_WRITER_MODEL,
                                        resolved_writer_model)
        rep["writer_requested"] = resolved_writer_model()
        rep["writer_source"] = ("env CODEGEN_LLM_MODEL"
                                if os.environ.get("CODEGEN_LLM_MODEL")
                                else f"code default {DEFAULT_WRITER_MODEL}")
    except Exception as e:                      # noqa: BLE001 — see docstring
        rep["writer_requested"] = f"unknown ({type(e).__name__}: {e})"
        rep["writer_source"] = "unknown"

    rep["backend_env"] = os.environ.get("CODEGEN_LLM_BACKEND") or "auto"
    rep["api_key_set"] = bool(os.environ.get("OPENAI_API_KEY")
                              or os.environ.get("LLM_CALLS_API_KEY"))
    try:
        from codegen.llm_client import get_default_client
        client = get_default_client()
        rep["backend"] = client.backend_name
        rep["is_fake"] = client.is_fake
        rep["writer_effective"] = client.backend_model
    except Exception as e:                      # noqa: BLE001
        # Selecting "openai" with no key raises here. That is a real
        # misconfiguration and --check-models must report it, not hide it.
        rep["backend"] = f"UNAVAILABLE ({type(e).__name__}: {e})"
        rep["is_fake"] = None
        rep["writer_effective"] = None

    # REASONER. A separate knob on purpose: the writer produces diffs (a 60%
    # failure rate in the recorded run), the reasoner produces diagnoses and
    # hypotheses. One number for "the model" cannot say which to spend on.
    try:
        from llm_calls import client as _lc
        rep["reasoner"] = _lc.DEFAULT_MODEL
        rep["cheap"] = _lc.DEFAULT_CHEAP_MODEL
        rep["max_output_tokens"] = _lc.MAX_OUTPUT_TOKENS
    except Exception as e:                      # noqa: BLE001
        rep["reasoner"] = rep["cheap"] = f"unknown ({type(e).__name__}: {e})"
        rep["max_output_tokens"] = None

    # Per-persona routing, once T2.4 provides it.
    try:
        from llm_calls.routing import resolved_table
        rep["routing"] = resolved_table()
    except Exception:                           # noqa: BLE001
        rep["routing"] = None
    return rep


def _print_model_banner(rep: dict | None = None) -> None:
    """The startup block: every model, its source, and the backend actually used."""
    rep = rep or _model_report()
    eff = rep.get("writer_effective")
    if rep.get("is_fake"):
        eff_str = "FakeBackend — CANNED OUTPUT, NOT THIS MODEL"
    elif eff:
        eff_str = f"backend will call {eff}"
    else:
        eff_str = "backend model unknown"
    print(f"[models] writer={rep.get('writer_requested')} "
          f"(from {rep.get('writer_source')}; {eff_str})")
    print(f"[models] reasoner={rep.get('reasoner')} (LLM_CALLS_MODEL) | "
          f"dedup={rep.get('cheap')} (LLM_CALLS_CHEAP_MODEL) | "
          f"max_output_tokens={rep.get('max_output_tokens')}")
    print(f"[models] backend={rep.get('backend')} "
          f"(CODEGEN_LLM_BACKEND={rep.get('backend_env')}, "
          f"api_key_set={rep.get('api_key_set')}, "
          f"mocks={rep.get('using_mocks')})")
    for persona, cfg in (rep.get("routing") or {}).items():
        print(f"[models]   {persona:22s} {cfg.get('model')} "
              f"effort={cfg.get('effort')}")


class FakeBackendInRealRunError(RuntimeError):
    """Raised when a real run would silently write canned edits.

    `codegen/llm_client.py::_auto_backend` returns FakeBackend whenever
    OPENAI_API_KEY is unset or the SDK import fails, with no warning — so a run
    with a bad key produces gate-clean single-token edits, scores them, and looks
    like it is working. Every candidate is then a measurement of the fake
    backend. This is the one place that turns that into a stop.
    """


def _require_real_models(rep: dict | None = None) -> dict:
    """Refuse to start a real search on a fake backend. Returns the report."""
    rep = rep or _model_report()
    if rep["using_mocks"]:
        return rep                      # --mock is an explicit, honest choice
    if rep.get("is_fake"):
        raise FakeBackendInRealRunError(
            f"codegen would use FakeBackend for a REAL run: canned edits, "
            f"scored as if measured. backend_env="
            f"{rep.get('backend_env')!r}, api_key_set={rep.get('api_key_set')}. "
            f"Set OPENAI_API_KEY (and CODEGEN_LLM_BACKEND=openai), or pass "
            f"--mock if a mocked run is what you wanted.")
    if rep.get("is_fake") is None:
        raise FakeBackendInRealRunError(
            f"codegen could not build an LLM backend: {rep.get('backend')}. "
            f"api_key_set={rep.get('api_key_set')}.")
    return rep


def _new_id() -> str:
    return uuid.uuid4().hex[:8]


def _new_root() -> Node:
    """An unmeasured root. _measure_root() fills in the real numbers.

    Deliberately carries no per-user data: it used to be seeded with a synthetic
    {u0..u9: 0.5946}, which shares no user ids with a real candidate, so every
    paired delta list came back empty (see _measure_root).
    """
    return Node(id=_new_id(), parent_id=None, code_path="baseline.py",
                operation="draft",
                local_best_score=FALLBACK_ROOT_PRIMARY)


def _measure_root(root: Node, counters: Counters, *, seeds=ROOT_SEEDS,
                  split: str = "valid_search", cache_path: str | None = None,
                  wallclock_cap_s: int = 1800, verbose: bool = True) -> bool:
    """Score the unmodified baseline so the root carries REAL per-user data.

    bootstrap_delta pairs candidate-against-parent per user id, so the parent's
    per-user scores have to come from an actual run. With the old synthetic root
    the intersection of user ids was always empty, every paired delta list came
    back empty, p_positive was always 0.0, should_continue_locally always False —
    so no candidate ever entered open_nodes and the tree could not grow past the
    root. This is what makes improvements able to compound.

    Returns True if the root now carries measured per-user data.
    """
    blob = None
    if cache_path and os.path.exists(cache_path):
        try:
            with open(cache_path) as fh:
                cached = json.load(fh)
            if cached.get("split") == split and cached.get("seeds"):
                blob = cached
                if verbose:
                    print(f"[root] reusing cached baseline from {cache_path}")
        except (OSError, ValueError):
            blob = None

    if blob is None:
        measured = {}
        for s in seeds:
            _t0 = time.time()
            r = codegen.execute(root.code_path, seed=s, split=split,
                                wallclock_cap_seconds=wallclock_cap_s, root=".",
                                data_dir=DATA_DIR)
            _elapsed = time.time() - _t0
            counters.bump("full_runs")
            counters.bump_scorer(split)
            if r["status"] != "ok":
                if verbose:
                    print(f"[root] baseline seed={s} {r['status']}; skipping seed")
                continue
            measured[str(s)] = {"primary": r["metrics"]["primary"],
                                # Measured runtime, cached alongside the scores,
                                # so the adaptive triage cap has a real number to
                                # scale off from the very first iteration.
                                "runtime_s": _elapsed,
                                # Both components, so the root has per-metric
                                # values for candidates to be compared against.
                                # A cache written before T2.8 has neither, which
                                # is handled below.
                                "GAUC": r["metrics"].get("GAUC"),
                                "nDCG@5": r["metrics"].get("nDCG@5"),
                                "per_user": r["metrics"].get("per_user", {})}
        if not measured:
            if verbose:
                print("[root] WARNING: baseline never scored. Root keeps the "
                      "published fallback and has no per-user data, so "
                      "should_continue_locally can never pass and the tree "
                      "cannot grow.")
            return False
        blob = {"split": split, "seeds": measured}
        if cache_path:
            _write_json_atomic(cache_path, blob)

    per_user = {int(s): v.get("per_user", {}) for s, v in blob["seeds"].items()}
    if not any(per_user.values()):
        if verbose:
            print("[root] WARNING: baseline reported no per_user block; the "
                  "tree cannot grow. Does the candidate emit "
                  "##CODEGEN_METRICS## with per_user?")
        return False

    root.per_user_by_seed = per_user
    root.seeds_run = sorted(per_user)
    root.per_seed_primary = {int(s): v["primary"] for s, v in blob["seeds"].items()}
    # `.get`, not `[...]`: a root_baseline.json written before T2.8 has only
    # `primary`, and re-measuring the baseline to backfill two reporting fields
    # would cost three full training runs. A cache from before the change simply
    # reports no per-metric split for the root.
    root.per_seed_gauc = {int(s): v["GAUC"] for s, v in blob["seeds"].items()
                          if v.get("GAUC") is not None}
    root.per_seed_ndcg5 = {int(s): v["nDCG@5"] for s, v in blob["seeds"].items()
                           if v.get("nDCG@5") is not None}
    _runtimes = [v["runtime_s"] for v in blob["seeds"].values()
                 if v.get("runtime_s")]
    # MEDIAN, not mean: a single cold-cache first seed should not inflate every
    # child's cap for the rest of the run.
    if _runtimes:
        _runtimes.sort()
        root.clean_runtime_s = _runtimes[len(_runtimes) // 2]
    # MEAN, not max. See _scalar_primary: the root runs 3 seeds and most
    # candidates finish only a 1-seed triage run, so a max-vs-max comparison
    # gave the root a free E[max of 3] - E[max of 1] ~= 0.85*sigma head start.
    root.local_best_score = _scalar_primary(root)
    return True


def _measure_confirm_baseline(counters: Counters, *, seeds=CONFIRM_SEEDS,
                              cache_path: str | None,
                              wallclock_cap_s: int = 1800,
                              verbose: bool = True) -> dict[int, dict[str, float]]:
    """Per-user baseline scores on valid_confirm, measured once and cached.

    Promotion pairs a candidate against this, on the same split. One seed is
    enough: the paired bootstrap runs over thousands of users, and every query
    here spends the sealed split's budget.

    Returns {seed: {user: primary}}, empty if the baseline could not be scored.
    """
    if cache_path and os.path.exists(cache_path):
        try:
            with open(cache_path) as fh:
                cached = json.load(fh)
            if cached.get("split") == "valid_confirm" and cached.get("seeds"):
                return {int(s): v.get("per_user", {})
                        for s, v in cached["seeds"].items()}
        except (OSError, ValueError):
            pass

    measured = {}
    for s in seeds:
        r = codegen.execute("baseline.py", seed=s, split="valid_confirm",
                            wallclock_cap_seconds=wallclock_cap_s, root=".",
                            data_dir=DATA_DIR)
        counters.bump("full_runs")
        counters.bump_scorer("valid_confirm")
        if r["status"] != "ok":
            if verbose:
                print(f"[confirm] baseline seed={s} {r['status']}; skipping seed")
            continue
        measured[str(s)] = {"primary": r["metrics"]["primary"],
                            "per_user": r["metrics"].get("per_user", {})}
    if not measured:
        if verbose:
            print("[confirm] WARNING: baseline never scored on valid_confirm; "
                  "no candidate can be promoted.")
        return {}
    if cache_path:
        _write_json_atomic(cache_path, {"split": "valid_confirm", "seeds": measured})
    return {int(s): v.get("per_user", {}) for s, v in measured.items()}


#: The mechanism-family taxonomy now lives in llm_calls/families.py, because
#: T2.6 makes the family a DECLARED schema field and the schema layer has to
#: validate against the same list. Re-exported under the historical names so
#: nothing that reads them from here has to change.
_MECHANISM_FAMILIES = MECHANISM_FAMILIES
ALL_FAMILIES = _FAMILY_NAMES


def _fingerprint(h: dict):
    """Semantic fingerprint of a hypothesis.

    Three branches, in precedence order.

    1. A DECLARED `mechanism_family`. This is what production now emits (T2.6),
       and it makes the fingerprint exact. The substring branch below could not
       be: family assignment depended on hand-ordered table position, so "a
       pairwise loss over user history" resolved to `generic_pairwise` rather
       than `sequence_features` purely because the loss families are checked
       first, and the bare token `sequence` swallowed any hypothesis containing
       that word anywhere.

    2. Structured fields, which keep the hand-authored preseeds in memory.py
       meaningful. Note `_reject_unknown_keys` forbids all four on a real
       hypothesis, so this branch is reachable only from a preseed or a test —
       which is exactly why the family collision was unreachable in every test
       before Tier 1: the mock supplied these keys with a per-iteration digest.

    3. The substring fallback, then a prose hash. Retained for a hypothesis that
       declared nothing (an older mock, a hand-built dict) and permissive for
       genuinely novel ideas.
    """
    declared = normalise_declaration(h.get("mechanism_family"))
    if declared and declared != OTHER:
        return ("mechanism_family", declared, "", "")
    if declared == OTHER:
        # The escape hatch is never a family, so it can never be blocked as one.
        # Hash the prose instead, which gives each `other` its own identity.
        mech = (h.get("mechanism") or "").strip().lower()
        digest = hashlib.sha1(mech.encode("utf-8")).hexdigest()[:16]
        return ("mechanism_hash", digest, "", "")

    structured_keys = ("loss_type", "sampler", "feature_set", "dataset_tier")
    if any(k in h for k in structured_keys):
        return (h.get("loss_type", "pointwise_logloss"),
                h.get("sampler", "uniform"),
                h.get("feature_set", "5field_baseline"),
                h.get("dataset_tier", "pure"))

    mech = (h.get("mechanism") or "").strip().lower()
    sketch = (h.get("implementation_sketch") or "").strip().lower()
    family = family_from_text(f"{mech} {sketch}")
    if family is not None:
        return ("mechanism_family", family, "", "")
    digest = hashlib.sha1(mech.encode("utf-8")).hexdigest()[:16]
    return ("mechanism_hash", digest, "", "")


#: Aliases onto the shared taxonomy (llm_calls/families.py), kept so the local
#: reads stay short. `other` is the escape hatch: a mechanism declaring it gets a
#: prose hash rather than a family, so no family entry can ever block it.
_ALL_FAMILIES: tuple[str, ...] = _FAMILY_NAMES
_OTHER_FAMILY = OTHER


def _family_of(fp) -> str | None:
    """The mechanism-family name a fingerprint names, or None if it names none.

    Only `_fingerprint`'s family branch carries a family; the structured
    preseeds and the prose-hash fallback do not.
    """
    return fp[1] if fp and fp[0] == "mechanism_family" else None


def _family_standing(memory: Memory) -> tuple[List[str], List[str], List[str]]:
    """(refuted, probationary, legal) mechanism families, per the evidence store.

    THE missing input on the proposal side. `_ask`'s context carried parent,
    history, iter_history, improvement_score, ablations, tried,
    component_ledger and exhausted_components — and nothing whatsoever about
    which mechanism families memory had already retired. So the diagnostician
    was being asked to avoid a constraint it could not see: it proposed into a
    banned family, the filter below silently deleted the result, and neither
    side ever learned anything. Iteration 4 of the recorded run did this 3 for 3
    and returned zero candidates.

    `refuted` are hard blocks. `probationary` have one scored refutation and are
    still proposable — the discount is rendered into the prompt rather than
    enforced as a filter, because generalising a single categorical-field
    experiment to a whole family is the unsound step, not the measurement.
    """
    refuted, probationary = [], []
    on_probation = set(memory.probationary_families())   # hoisted: O(n^2) each
    for fam in _ALL_FAMILIES:
        fp = ("mechanism_family", fam, "", "")
        if _memory_blocks(memory, fp):
            refuted.append(fam)
        elif fam in on_probation:
            probationary.append(fam)
    legal = [f for f in _ALL_FAMILIES if f not in refuted] + [_OTHER_FAMILY]
    return refuted, probationary, legal


def _infeasible_reason(h: dict) -> str | None:
    """The named primitive that cannot be written here, or None.

    Deterministic and free, run BEFORE a writer call is spent. The persona
    already carries a constraint block naming these, and 61 of 65 stored
    proposals violated it anyway — so a prompt request is demonstrably not
    enough, and this is the same "enforce it in the loop, not only in the
    prompt" move as the component-exhaustion budget.
    """
    text = (f"{h.get('mechanism') or ''} "
            f"{h.get('implementation_sketch') or ''}").lower()
    return next((t for t in _INFEASIBLE_TOKENS if t in text), None)


def _memory_blocks(memory: Memory, fp) -> bool:
    """Does the evidence store refuse a proposal at this fingerprint?

    The POLICY question, kept separate from `Memory.is_duplicate`'s EVIDENCE
    question ("is a refutation recorded here"). T1.4 makes the policy require
    corroboration; this indirection is the single place the driver asks.
    """
    return memory.is_blocked(fp)


def _diff_hash(diff: str) -> str:
    return hashlib.sha1(diff.encode("utf-8")).hexdigest()


def _best_primary(c: Node) -> float:
    """Best primary score observed for a candidate.

    Survivors get local_best_score from the full-seed runs; candidates that
    only completed a triage run keep local_best_score == -inf and carry their
    score in partial_scores. Take the max of whatever exists.

    Retained for nodes that carry no per-seed breakdown; _scalar_primary is the
    score the search actually compares on.
    """
    scores = list(c.partial_scores)
    if c.local_best_score > float("-inf"):
        scores.append(c.local_best_score)
    return max(scores) if scores else float("-inf")


#: Per-metric ceilings on this dataset. The oracle primary is 0.8645, and it is
#: NOT the mean of two equal halves: a perfect ranking gives GAUC 1.0 but nDCG@5
#: only 0.7289, because users with zero positives score 0 and are counted in the
#: average. So the headroom is lopsided — 0.3390 on GAUC against 0.2007 on
#: nDCG@5 from the baseline — and a diagnostician that sees one scalar cannot
#: know which gap is bigger.
GAUC_CEILING = 1.0
NDCG5_CEILING = 0.7289

#: Baseline hidden-test values, for the distance-to-ceiling report.
BASELINE_GAUC = 0.6610
BASELINE_NDCG5 = 0.5282


def _triage_cap_for(parent: Node) -> int:
    """The wall-clock cap for a triage run of one of `parent`'s children.

    Derived from the PARENT's own last clean runtime, clamped to
    [TRIAGE_CAP_MIN_S, TRIAGE_CAP_MAX_S]. Falls back to the minimum when the
    parent has no measured runtime — which is strictly more generous than the old
    flat 240s, and the smoke stage means only correct code ever reaches it.
    """
    rt = getattr(parent, "clean_runtime_s", None)
    if not rt or rt <= 0:
        return TRIAGE_CAP_MIN_S
    return int(max(TRIAGE_CAP_MIN_S,
                   min(TRIAGE_CAP_MAX_S, TRIAGE_RUNTIME_MULTIPLE * rt)))


def _mean_metric(per_seed: dict) -> float | None:
    """Mean over the seeds a node actually ran, or None if it ran none."""
    if not per_seed:
        return None
    return sum(per_seed.values()) / len(per_seed)


def _metric_block(c: Node) -> dict:
    """Both metrics for one node, each with its own distance to its own ceiling.

    THE resolution the diagnostician was missing. `gauc_to_ceiling` and
    `ndcg5_to_ceiling` are the two numbers that say where the headroom actually
    is; a scalar primary averages them and hides the asymmetry.
    """
    gauc = _mean_metric(c.per_seed_gauc)
    ndcg = _mean_metric(c.per_seed_ndcg5)
    return {
        "GAUC": gauc,
        "nDCG@5": ndcg,
        "gauc_to_ceiling": (GAUC_CEILING - gauc) if gauc is not None else None,
        "ndcg5_to_ceiling": (NDCG5_CEILING - ndcg) if ndcg is not None else None,
        "n_seeds": len(c.per_seed_primary),
    }


def _record_metrics(c: Node, seed: int, metrics: dict) -> None:
    """Store primary AND both components for one seed of one node."""
    c.per_seed_primary[seed] = metrics["primary"]
    if metrics.get("GAUC") is not None:
        c.per_seed_gauc[seed] = float(metrics["GAUC"])
    if metrics.get("nDCG@5") is not None:
        c.per_seed_ndcg5[seed] = float(metrics["nDCG@5"])
    c.per_user_by_seed[seed] = metrics.get("per_user", {})


def _scalar_primary(c: Node) -> float:
    """The one number the search ranks a node by: MEAN primary over its seeds.

    Deliberately the mean. The root is measured on 3 seeds while most candidates
    finish only the 1-seed triage run, so a max-vs-max comparison hands whichever
    side ran more seeds a free E[max of n] head start. Measured on this repo's
    own cached root baseline (per-seed 0.594505 / 0.594966 / 0.594448): max
    0.594966 against mean 0.594640, a bias of +0.00033 — a third of the largest
    delta the search has ever produced (+0.0011), paid by the candidate every
    time. A mean is unbiased in the number of seeds, so a 1-seed candidate and a
    3-seed root are compared on equal terms: noisier, but not tilted.
    """
    if c.per_seed_primary:
        return sum(c.per_seed_primary.values()) / len(c.per_seed_primary)
    return _best_primary(c)


def _is_no_op(cand: Node, parent: Node, seed: int) -> bool:
    """True if the candidate reproduced its parent's per-user scores exactly.

    A rewrite can apply cleanly, pass the gate, run to completion and still not
    touch the code that executes — the classic shape is a new helper function
    that nothing calls. Then every per-user score matches the parent bit for bit.
    Detecting that here turns a silent "no improvement" into a named, actionable
    outcome, and stops the memory store from recording the mechanism as
    genuinely refuted when it was never actually tried.

    Returns False when there is nothing to compare against, so a missing parent
    measurement is never reported as a no-op.
    """
    cu = cand.per_user_by_seed.get(seed) or {}
    pu = parent.per_user_by_seed.get(seed) or {}
    shared = set(cu) & set(pu)
    if not shared:
        return False
    return all(abs(cu[u] - pu[u]) < NOOP_EPSILON for u in shared)


def _write_json_atomic(path: str, payload: dict) -> None:
    os.makedirs(os.path.dirname(path), exist_ok=True)
    tmp = path + ".tmp"
    with open(tmp, "w") as fh:
        json.dump(payload, fh, indent=2)
    os.replace(tmp, path)   # atomic: tailing mid-run never sees a partial write


def _write_text_atomic(path: str, text: str) -> None:
    os.makedirs(os.path.dirname(path) or ".", exist_ok=True)
    tmp = path + ".tmp"
    with open(tmp, "w", encoding="utf-8") as fh:
        fh.write(text)
    os.replace(tmp, path)


def _append_nodes_log(candidates: List[Node], iter_no: int,
                      path: str | None = None) -> None:
    """Append one JSON object per candidate to nodes.jsonl.

    Each line stands alone: a later iteration's failure never corrupts prior
    records, and `tail -f nodes.jsonl | jq .` streams live. Diagnosis and
    hypothesis go in verbatim because they are the record judges read to
    assess Autonomy.

    `path` resolves against the module global at CALL time rather than defaulting
    to it in the signature: a default would freeze the repo path at import, so a
    test redirecting NODES_LOG_PATH would clear its tmp file and then write into
    the live orchestrator/_state/ anyway.
    """
    path = path or NODES_LOG_PATH
    os.makedirs(os.path.dirname(path), exist_ok=True)
    with open(path, "a", encoding="utf-8") as fh:
        for c in candidates:
            fh.write(json.dumps({
                "iter": iter_no,
                "id": c.id,
                "parent_id": c.parent_id,
                "operation": c.operation,
                "status": c.status,
                "evidence_type": c.evidence_type,
                "code_dir": c.code_dir,
                "hypothesis": c.hypothesis,
                "diagnosis": c.diagnosis,
                "per_seed_primary": c.per_seed_primary,
                # Both components per seed. Section 2.5 requires them in the
                # submitted run log, and a scalar primary cannot show that a
                # change gained on GAUC and lost on nDCG@5.
                "per_seed_gauc": c.per_seed_gauc,
                "per_seed_ndcg5": c.per_seed_ndcg5,
                "GAUC": _mean_metric(c.per_seed_gauc),
                "nDCG@5": _mean_metric(c.per_seed_ndcg5),
                "mean_delta": c.mean_delta,
                "p_positive": c.p_positive,
                "lower_95": c.lower_95,
                "fix_attempts": c.fix_attempts,
                "wallclock_used_s": round(c.wallclock_used_s, 2),
                # Why it failed and how the criterion it declared was judged.
                # Both were previously unrecoverable after the run: the failure
                # reason lived only in the run log, and the verdict did not
                # exist, so "refuted" and "never implemented" and "criterion
                # uncalibrated by 8x" were one indistinguishable record.
                # The leak control's result, alongside the real score. The pair
                # is the evidence; either number alone says nothing.
                "confirm_primary": c.confirm_primary,
                "permuted_primary": c.permuted_primary,
                "last_error_excerpt": c.last_error_excerpt,
                "verdict": c.verdict,
                "verdict_reason": c.verdict_reason,
                "next_action": c.next_action,
                "criterion_was_calibrated": c.criterion_was_calibrated,
            }, default=str) + "\n")


def _archive_champion(node: Node, primary: float, *,
                      champion_dir: str | None = None) -> str | None:
    """Copy the best-scoring candidate's staged source somewhere durable.

    Candidate dirs come from _apply_diff_and_stage, which stages under
    tempfile.gettempdir(). On the machine that produced orchestrator/_state/
    that was /var/folders/hr/..., which macOS clears — and nothing ever copied
    the winner out. So a run whose best candidate beat the baseline still left
    no source tree to regenerate a submission from once the temp dir went away.

    Writes <champion_dir>/<node id>/ containing the staged modules plus a
    manifest naming the score and the hypothesis, so the archive is
    self-describing without needing progress.json alongside it. Returns the
    directory, or None if the node has nothing to archive.
    """
    champion_dir = champion_dir or CHAMPION_DIR
    if not node.code_dir or not os.path.isdir(node.code_dir):
        return None
    dest = os.path.join(champion_dir, node.id)
    os.makedirs(dest, exist_ok=True)
    for name in _STAGED_MODULES:
        src = os.path.join(node.code_dir, name)
        if os.path.exists(src):
            shutil.copy2(src, os.path.join(dest, name))
    _write_json_atomic(os.path.join(dest, "manifest.json"), {
        "id": node.id,
        "parent_id": node.parent_id,
        "operation": node.operation,
        "primary": primary,
        "per_seed_primary": node.per_seed_primary,
        "mean_delta": node.mean_delta,
        "p_positive": node.p_positive,
        "lower_95": node.lower_95,
        "confirm_primary": node.confirm_primary,
        "hypothesis": node.hypothesis,
        "diagnosis": node.diagnosis,
        "staged_from": node.code_dir,
        "archived_at": time.time(),
    })
    return dest


#: How many past attempts to show a proposal operator. Bounded because the
#: ledger is rendered into every diagnose/hypothesis prompt and an unbounded
#: history is exactly the context blowup ML-Master's scoped memory avoids.
LEDGER_MAX_ENTRIES = 12


def _attempt_ledger(nodes: List[Node], parent: Node,
                    max_entries: int = LEDGER_MAX_ENTRIES) -> List[dict]:
    """What has already been tried, for the proposal operators to read.

    THE missing input. diagnose() used to receive only
    {parent, history, iter_history, improvement_score, ablations} and
    generate_hypothesis() only (diagnosis, evidence_card) — neither could see
    a single prior attempt. With the parent frozen at the root the diagnosis
    was therefore identical every iteration ("objective mismatch: pointwise
    loss on a ranking metric"), and all 11 proposals across the 5-iteration
    run were BPR / listwise-softmax / RankNet / LambdaRank reworded. The
    memory store recorded every one of them and no prompt ever read it.

    Scoping follows the two findings this is taken from: siblings of the node
    about to be expanded, for diversity (AIRA), and every attempt's measured
    outcome as feedback, for the inner refinement loop (MLE-STAR). Newest
    last, so the most recent attempts sit closest to the instruction.
    """
    out: List[dict] = []
    for n in nodes:
        if n.hypothesis is None:
            continue          # the root draft has no proposal to report
        out.append({
            "id": n.id,
            "sibling_of_selected_parent": n.parent_id == parent.id,
            "component": (n.diagnosis or {}).get("component"),
            "mechanism": (n.hypothesis or {}).get("mechanism"),
            "outcome": n.evidence_type or ("scored" if n.per_seed_primary
                                           else "unresolved"),
            "primary": _scalar_primary(n) if n.per_seed_primary else None,
            # Both metrics, not just their mean. A change that gained on GAUC and
            # lost on nDCG@5 was indistinguishable from one that did nothing, and
            # the headroom is lopsided (0.3390 on GAUC vs 0.2007 on nDCG@5), so
            # the proposer could not aim at the larger gap.
            "GAUC": _mean_metric(n.per_seed_gauc),
            "nDCG@5": _mean_metric(n.per_seed_ndcg5),
            "mean_delta_vs_parent": n.mean_delta,
            # The grader's read on the criterion this attempt declared, and what
            # it recommends doing with the mechanism family. Without these the
            # ledger says an attempt scored +0.0006 but not whether that counted
            # as success against what it promised — and "+0.0006 missed a
            # miscalibrated +0.005" invites a cheaper retry, while
            # "abandon_mechanism" forbids the family outright.
            "verdict": n.verdict,
            "next_action": n.next_action,
        })
    return out[-max_entries:]


def _apply_verdict(c: Node, parent: Node, root: Node, verdict_llm,
                   counters: Counters, *, verbose: bool = True) -> None:
    """Grade one scored candidate against its own declared criterion and write
    the result onto the node.

    Swallows every exception on purpose. A grader is commentary on a
    measurement, never the measurement itself, so a schema failure or a network
    error must not lose a candidate that already cost three full training runs.
    On failure the node keeps verdict=None and the search proceeds exactly as it
    did before this step existed.
    """
    if not c.hypothesis:
        return
    try:
        v = verdict_llm.verdict(
            c.hypothesis,
            {"mean_delta": c.mean_delta,
             "p_positive": c.p_positive,
             "lower_95": c.lower_95,
             "per_seed_primary": dict(c.per_seed_primary),
             "candidate_primary": _scalar_primary(c),
             "parent_primary": _scalar_primary(parent),
             "evidence_type": c.evidence_type,
             "n_seeds": len(c.per_seed_primary)},
            {"baseline_primary": root.local_best_score,
             "parent_id": parent.id,
             "component": (c.diagnosis or {}).get("component"),
             "paired_noise_floor": 0.0012,
             "baseline_seed_std": 0.0008})
    except Exception as e:                      # noqa: BLE001 — see docstring
        if verbose:
            print(f"  {c.id} verdict unavailable ({type(e).__name__}: {e}); "
                  f"continuing without one")
        return
    c.verdict = v["verdict"]
    c.verdict_reason = v["reason"]
    c.next_action = v["next_action"]
    c.criterion_was_calibrated = v["criterion_was_calibrated"]
    if verbose:
        print(f"  {c.id} verdict={c.verdict} "
              f"(criterion {'calibrated' if c.criterion_was_calibrated else 'UNCALIBRATED'}"
              f", next: {c.next_action}) — {c.verdict_reason}")


#: Canonical component names, and the substrings that map onto them. Checked in
#: order, first match wins, so the SPECIFIC buckets come before the generic ones
#: they would otherwise be swallowed by ("sequence features" contains "feature").
#:
#: This normalisation is what makes the attempt budget below able to fire at all.
#: `component` is a free-form string the diagnostician invents — the mock returns
#: "loss", a real diagnosis returned "loss_function", and a third phrasing would
#: be "objective mismatch". Counted verbatim, three attempts at the same
#: bottleneck under three spellings look like one attempt each and the budget
#: never trips.
_COMPONENT_ALIASES: tuple[tuple[str, tuple[str, ...]], ...] = (
    ("sequence_features", ("sequence", "history", "behaviour", "behavior",
                           "lag", "session")),
    ("auxiliary_targets", ("auxiliar", "multi-task", "multitask", "multi task")),
    ("loss_function", ("loss", "objective", "criterion", "ranking metric "
                       "mismatch")),
    ("sampling", ("sampl", "negative")),
    ("regularization", ("regulari", "dropout", "weight decay", "l2 ", "overfit")),
    # Plain substrings, not regexes — these are matched with `in`, so a token
    # like r"\blr\b" would silently never fire.
    ("optimization", ("optimiz", "learning rate", "learning_rate", "adam",
                      "schedule", "epoch", "convergence")),
    ("feature_engineering", ("feature", "encoding", "field", "data.py",
                             "representation", "input")),
    ("architecture", ("architecture", "capacity", "embedding dim", "model "
                      "family", "interaction order")),
)


def _canonical_component(component: str | None) -> str | None:
    """Map a free-form diagnosis component onto a canonical bucket.

    Unmatched names fall through to their own lowercased selves rather than a
    catch-all, so a genuinely novel bottleneck gets its own budget instead of
    inheriting an unrelated one's exhaustion.
    """
    if not component:
        return None
    text = str(component).strip().lower()
    for canon, tokens in _COMPONENT_ALIASES:
        if any(t in text for t in tokens):
            return canon
    return text


#: A component is exhausted after this many SCORED attempts that failed to beat
#: COMPONENT_EXHAUSTED_DELTA. Attempts that never produced a paired delta
#: (failed_implementation, timeout, no_op) are deliberately excluded: they are
#: evidence about the WRITER, not about the component, and counting them would
#: retire a bottleneck that was never actually tested. In the recorded run only
#: 5 of 11 candidates ever produced a paired delta, so the distinction decides
#: whether the budget measures anything real.
COMPONENT_ATTEMPT_BUDGET = 3

#: The bar a component's best paired delta must clear to stay live. Sits below
#: the 0.0008 baseline seed std on purpose: "no attempt on this component has
#: produced a gain even the size of measurement noise".
COMPONENT_EXHAUSTED_DELTA = 0.0005

#: The README's ranked unexplored directions, as canonical component names, used
#: only as the deterministic fallback when the diagnostician insists on an
#: exhausted component twice. Ordered by the starter kit's own order of promise.
UNEXPLORED_PRIORITY = ("loss_function", "sequence_features",
                       "auxiliary_targets", "feature_engineering")


def _component_ledger(all_nodes: List[Node]) -> dict:
    """Per-component tally of what has been attempted and what it measured.

    Keyed by canonical component. `scored` counts only attempts that produced a
    paired delta; `attempts` counts every candidate ever created for that
    component, so the two together say whether a component looks bad or merely
    looks untested.
    """
    out: dict = {}
    for n in all_nodes:
        if n.hypothesis is None:
            continue                 # the root draft has no component
        comp = _canonical_component((n.diagnosis or {}).get("component"))
        if comp is None:
            continue
        rec = out.setdefault(comp, {"attempts": 0, "scored": 0,
                                    "best_mean_delta": None, "verdicts": []})
        rec["attempts"] += 1
        if n.mean_delta is not None:
            rec["scored"] += 1
            if (rec["best_mean_delta"] is None
                    or n.mean_delta > rec["best_mean_delta"]):
                rec["best_mean_delta"] = n.mean_delta
        if n.verdict:
            rec["verdicts"].append(n.verdict)
    return out


def _exhausted_components(ledger: dict) -> set:
    """Components with enough SCORED attempts and nothing to show for them."""
    return {comp for comp, r in ledger.items()
            if r["scored"] >= COMPONENT_ATTEMPT_BUDGET
            and (r["best_mean_delta"] is None
                 or r["best_mean_delta"] < COMPONENT_EXHAUSTED_DELTA)}


#: How much of a failed ancestor's log tail to carry. ~600 chars is enough for a
#: Python traceback's final frame plus its exception line, which is the part that
#: says what to fix.
ANCESTOR_ERROR_CHARS = 600

#: How far up the tree the debug operator is shown. Bounded for the same reason
#: the ledger is: this text goes into a prompt alongside a whole source file.
ANCESTOR_MAX_DEPTH = 4


def _ancestor_chain(node: Node, all_nodes: List[Node],
                    max_depth: int = ANCESTOR_MAX_DEPTH) -> List[dict]:
    """The chain from `node` up toward the root, newest first, excluding `node`.

    AIRA's "ancestral memories for Debug" (arXiv:2507.02554). The repair
    operator used to receive a traceback and a file and nothing else — not what
    the edit was trying to do, not that its ancestors had already failed the
    same way. It handled 6 of the 11 candidates in the recorded run (5
    failed_implementation + 1 timeout, per progress.json), so the modal path
    was the blind one.

    Each entry carries what the two branches of the debug instruction need to be
    distinguishable: the mechanism and evidence_type say whether the ancestor
    failed to RUN (try a different implementation of the same idea) or ran and
    scored WORSE (the mechanism is suspect, do not resurrect it), and
    last_error_excerpt says how it broke.

    Stops at the root, and is safe against a parent_id cycle — a malformed tree
    must not hang an unattended run.
    """
    by_id = {n.id: n for n in all_nodes}
    out: List[dict] = []
    seen = {node.id}
    cur = by_id.get(node.parent_id) if node.parent_id else None
    while cur is not None and len(out) < max_depth and cur.id not in seen:
        seen.add(cur.id)
        out.append({
            "id": cur.id,
            "operation": cur.operation,
            "mechanism": (cur.hypothesis or {}).get("mechanism"),
            "evidence_type": cur.evidence_type,
            "mean_delta": cur.mean_delta,
            "last_error_excerpt": cur.last_error_excerpt,
        })
        cur = by_id.get(cur.parent_id) if cur.parent_id else None
    return out


# --------------------------------------------------------------------------- #
#  Run-wide failure digest (T3.2)                                              #
# --------------------------------------------------------------------------- #
# WHY IT IS NOT THE ANCESTOR CHAIN. `_ancestor_chain` walks ancestors only, by
# design — a descendant inherits its parent's staged code, so its parent's
# failures are the relevant ones. But the two identical `encode` crashes in the
# recorded run were SIBLINGS: both children of the root in iteration 2, two
# candidates that extended raw(x) without extending FIELDS and died with the same
# IndexError at the same line. Neither could see the other's traceback, because
# neither is an ancestor of the other.
#
# Two failures with the same signature are the cheapest learning signal a search
# can get — the second one already knows the answer if it can see the first. This
# indexes by (file, exception_type, line) rather than by tree position, because
# that is how the failures actually cluster.
#
# After T1.6 the digest also fills from SMOKE rejections, which cost ~0.06s each
# instead of 240s, so the signal accumulates at near-zero price.

#: Cap, matching LEDGER_MAX_ENTRIES' reasoning: this text goes into a repair
#: prompt alongside a whole source file.
DIGEST_MAX_PER_SIGNATURE = 4

#: Last frame of a traceback: `File "...", line N, in name` then the exception.
_TB_FRAME_RE = re.compile(r'File "([^"]+)", line (\d+)')
_TB_EXC_RE = re.compile(r"^(\w+(?:Error|Exception|Warning))\s*:", re.MULTILINE)


def _failure_signature(log: str) -> tuple | None:
    """(file, exception_type, line) for the DEEPEST frame of a traceback.

    The deepest frame is the one that says what to fix; the outer frames are the
    call path that got there and are identical for every candidate. Returns None
    when the log carries no recognisable traceback — a timeout, for instance,
    which has no signature and must not be filed under a fake one.
    """
    if not log:
        return None
    frames = _TB_FRAME_RE.findall(log)
    excs = _TB_EXC_RE.findall(log)
    if not frames or not excs:
        return None
    path, line = frames[-1]
    return (os.path.basename(path), excs[-1], int(line))


class FailureDigest:
    """Prior failures and repairs, indexed by signature, for the whole run."""

    def __init__(self, max_per_signature: int = DIGEST_MAX_PER_SIGNATURE):
        self.max_per_signature = max_per_signature
        self.by_signature: dict = {}

    def record(self, log: str, *, node_id: str, mechanism: str | None,
               repair_attempted: str | None = None,
               repaired: bool | None = None,
               stage: str = "execute") -> tuple | None:
        """File one failure. Returns its signature, or None if it has none."""
        sig = _failure_signature(log)
        if sig is None:
            return None
        entries = self.by_signature.setdefault(sig, [])
        entries.append({
            "node_id": node_id,
            "mechanism": (mechanism or "")[:160],
            "stage": stage,
            "repair_attempted": (repair_attempted or "")[:400] or None,
            # None = not yet known, True/False = the rerun's verdict. This is the
            # field that makes the digest worth more than a list of tracebacks:
            # "three candidates hit this and none of the repairs worked" is
            # actionable in a way "three candidates hit this" is not.
            "repaired": repaired,
        })
        del entries[:-self.max_per_signature]
        return sig

    def note_repair(self, sig, *, node_id: str, diff: str) -> None:
        """Attach the repair diff that was tried for `node_id`'s failure.

        Truncated hard: this goes into a prompt that already carries a whole
        source file, and the first few hunks are what say what was attempted.
        """
        for e in reversed(self.by_signature.get(sig) or []):
            if e["node_id"] == node_id:
                e["repair_attempted"] = diff[:400]
                return

    def resolve(self, sig, *, node_id: str, repaired: bool) -> None:
        """Record whether the repair for `node_id`'s failure at `sig` worked."""
        for e in reversed(self.by_signature.get(sig) or []):
            if e["node_id"] == node_id:
                e["repaired"] = repaired
                return

    def matching(self, log: str, *, exclude_node: str | None = None) -> List[dict]:
        """Prior failures sharing this log's signature, excluding `exclude_node`.

        Empty when the signature is new, which is the common case early in a run
        and is exactly when there is nothing useful to say.
        """
        sig = _failure_signature(log)
        if sig is None:
            return []
        return [e for e in (self.by_signature.get(sig) or [])
                if e["node_id"] != exclude_node]

    def summary(self) -> List[dict]:
        """Signatures seen more than once, for the run log. The repeats are the
        interesting part: a signature seen once is a bug, seen three times is a
        systematic gap in what the writer is being told."""
        return sorted(
            ({"file": sig[0], "exception": sig[1], "line": sig[2],
              "count": len(entries),
              "repairs_that_worked": sum(1 for e in entries if e["repaired"]),
              "nodes": [e["node_id"] for e in entries]}
             for sig, entries in self.by_signature.items() if len(entries) > 1),
            key=lambda r: -r["count"])


# --------------------------------------------------------------------------- #
#  Leak controls on the promotion path (T3.4)                                  #
# --------------------------------------------------------------------------- #
#: Highest primary a LABEL-PERMUTED run may score before the candidate is treated
#: as reading the label.
#:
#: Calibrated by measurement, not guessed. Running the unmodified baseline on
#: valid_search with each user's labels shuffled among that user's own rows:
#:
#:     real      primary 0.5938   GAUC 0.6631   nDCG@5 0.5246
#:     permuted  primary 0.4840   GAUC 0.4998   nDCG@5 0.4682
#:
#: GAUC lands on 0.4998 against a theoretical 0.5, which is the control working
#: exactly as intended. 0.55 sits 0.066 above that measured null and 0.044 below
#: the real baseline, so it separates the two cleanly with room on both sides.
PERMUTATION_MAX_PRIMARY = 0.55

#: The measured null, kept for the log line so a reader can see how far a
#: control run was from where it should be.
PERMUTED_BASELINE_PRIMARY = 0.4840


def _leak_check(c: Node, cand_dir: str, counters: Counters, *,
                confirm_primary: float | None, history: List[float] | None = None,
                verbose: bool = True) -> str | None:
    """Two deterministic leak controls. Returns a reason to REFUSE, or None.

    Run only on the promotion path, which is the one place worth paying an extra
    run for: a promotion is what a submission gets generated from, and the
    advisory LLM auditor cannot be relied on — it flagged 5 of 5 candidates in
    the recorded run, including "y is being used within the step function", which
    is the training loop. Its signal-to-noise is zero at ~4,500 input tokens per
    candidate; a permutation control is deterministic and costs one run.

    1. ORACLE CEILING. A primary above `ORACLE_PRIMARY_CEILING` (0.8645) is
       physically impossible on this dataset, so it is not a result. The constant
       already existed and was reachable ONLY through `codegen/debug.py`'s sanity
       branch, which the driver never triggers because it never passes
       `observed_score` — so nothing in the loop ever checked it.

    2. LABEL PERMUTATION. Re-score the candidate with each user's labels shuffled
       among that user's own rows. A model carrying no label information collapses
       to chance; one reading the label keeps its score. The shuffle is
       within-user so every user's positive count is preserved, which leaves
       GAUC's user filter and nDCG's ideal DCG unchanged — see
       `codegen/sandbox.py::_PERMUTE_SHIM`.
    """
    from codegen.constants import ORACLE_PRIMARY_CEILING

    if confirm_primary is not None and confirm_primary > ORACLE_PRIMARY_CEILING:
        return (f"confirm primary {confirm_primary:.4f} exceeds the oracle "
                f"ceiling {ORACLE_PRIMARY_CEILING} — physically impossible on "
                f"this dataset, so it is a leak or a scoring bug, not a result")

    if verbose:
        print(f"  {c.id} running the label-permutation control on "
              f"valid_confirm ...")
    r = codegen.execute(c.code_path, seed=0, split="valid_confirm",
                        wallclock_cap_seconds=FULL_RUN_WALLCLOCK_CAP_S,
                        root=cand_dir, data_dir=DATA_DIR,
                        permute_labels=True)
    counters.bump("permutation_runs")
    if r["status"] != "ok":
        # Inconclusive, NOT a pass. A control that could not run has told us
        # nothing, and promoting on "the check crashed" is how a leak ships.
        return (f"the label-permutation control did not run "
                f"({r['status']}), so the candidate is unverified")
    permuted = r["metrics"].get("primary")
    c.permuted_primary = permuted
    if permuted is None:
        return "the label-permutation control reported no primary"
    if permuted > PERMUTATION_MAX_PRIMARY:
        return (f"label-permutation control scored {permuted:.4f} on SHUFFLED "
                f"labels (bar {PERMUTATION_MAX_PRIMARY}, measured null "
                f"{PERMUTED_BASELINE_PRIMARY}). A model with no label "
                f"information cannot do this — the score is coming from the "
                f"label, not from the mechanism")
    # ADVISORY, and free unless the score is implausible: sanity_check returns
    # None without a model call unless the result is above the oracle ceiling or
    # a >0.02 leap over the best prior. It adds the one judgement the
    # deterministic controls above cannot make — whether the diff implements the
    # stated mechanism at all. It does NOT veto: the auditor's demonstrated
    # signal-to-noise on this repo is zero, so an LLM opinion is recorded, not
    # obeyed.
    try:
        opinion = codegen.sanity_check(
            None, hypothesis=c.hypothesis, observed_score=confirm_primary,
            history=list(history or []), threshold=None)
    except Exception as e:                          # noqa: BLE001
        opinion = {"reasoning": f"sanity check unavailable ({e})"}
    if opinion:
        c.diagnosis = {**(c.diagnosis or {}), "sanity": opinion}
        if verbose:
            print(f"  {c.id} sanity (advisory): "
                  f"leak_suspected={opinion.get('leak_suspected')} — "
                  f"{str(opinion.get('reasoning'))[:120]}")

    if verbose:
        print(f"  {c.id} permutation control OK: {permuted:.4f} on shuffled "
              f"labels vs {confirm_primary:.4f} real")
    return None


def _prior_refines_for_component(component: str,
                                 path: str | None = None) -> List[dict]:
    """Scan nodes.jsonl for prior refine attempts on this component.

    Returns a list of {mechanism, mean_delta} dicts, oldest first. Empty
    if nodes.jsonl doesn't exist or has no matching entries.

    `path` resolves at CALL time for the same reason _append_nodes_log's does.
    """
    path = path or NODES_LOG_PATH
    if not os.path.exists(path):
        return []
    out = []
    with open(path, encoding="utf-8") as fh:
        for line in fh:
            try:
                r = json.loads(line)
                if (r.get("operation") == "refine"
                        and (r.get("diagnosis") or {}).get("component") == component):
                    out.append({
                        "mechanism": (r.get("hypothesis") or {}).get("mechanism"),
                        "mean_delta": r.get("mean_delta"),
                    })
            except ValueError:
                continue
    return out


def _refine_triggers(iter_history: List[float], improves_since_refine: int
                     ) -> tuple[bool, bool, float | None]:
    """Should this iteration refine instead of improve?

    Returns (cadence_trigger, plateau_trigger, recent_improvement). Pulled out
    of run()'s loop as a pure function so both triggers can be tested without
    a test-only hook for seeding the closure's iter_history.

    recent_improvement is iter_history[-1] - iter_history[-4] evaluated at the
    START of an iteration, so it equals the improvement_score that
    _record_iteration wrote for the PREVIOUS iteration. None until four
    entries exist (baseline + three completed iterations).
    """
    recent_improvement = (iter_history[-1] - iter_history[-4]
                          if len(iter_history) >= 4 else None)
    cadence_trigger = improves_since_refine >= REFINE_EVERY_K_IMPROVES
    plateau_trigger = (recent_improvement is not None
                       and recent_improvement <= PLATEAU_REFINE_THRESHOLD)
    return cadence_trigger, plateau_trigger, recent_improvement


def _build_improve_candidates(parent: Node, *,
                              diag_llm, memory: Memory,
                              counters: Counters,
                              history: List[float],
                              iter_history: List[float],
                              improvement_score: float | None,
                              tried: List[dict] | None = None,
                              component_ledger: dict | None = None,
                              verbose: bool = True
                              ) -> tuple[dict, List[Node]]:
    """The pre-Phase-2 improve path, factored out and enriched with
    trajectory signals. Returns (diagnosis, the dedup'd list of new Nodes).

    The diagnosis comes back alongside the candidates because the caller's
    _attempt() closure needs it to route the writer at target_component. It
    cannot be recovered from an empty candidate list.
    """
    # With Phase 2 off, no ablation evidence reaches the diagnostician — a
    # stale ablations.jsonl from an earlier run would otherwise keep feeding
    # it, and this path is meant to be the pre-Phase-2 one exactly.
    cached_ablations: dict = {}
    if REFINE_ENABLED:
        _abl_cache = _load_ablation_cache()
        cached_ablations = {
            name: _abl_cache.get(_ablation_cache_key(parent.id, name))
            for name in _ABLATION_COMPONENTS
        }
        cached_ablations = {k: v for k, v in cached_ablations.items()
                            if v is not None}

    tried = tried or []
    component_ledger = component_ledger or {}
    exhausted = _exhausted_components(component_ledger)
    refuted_families, probationary_families, legal_families = _family_standing(
        memory)

    def _ask(refusal: str | None = None) -> dict:
        ctx = {
            "parent": parent.id,
            "history": history,                     # promotion ladder (unchanged)
            "iter_history": list(iter_history),     # iteration-level trajectory
            "improvement_score": improvement_score, # current ε/N plateau signal
            "ablations": cached_ablations or None,
            # Every prior attempt and what it measured. Without this the
            # diagnostician re-derived the same bottleneck from the same
            # trajectory every iteration; see _attempt_ledger.
            "tried": tried or None,
            # Per-component tallies, and the components that have spent their
            # budget without producing a gain.
            "component_ledger": component_ledger or None,
            "exhausted_components": sorted(exhausted) or None,
            # Where the headroom IS, per metric. The oracle primary of 0.8645 is
            # not two equal halves: a perfect ranking gives GAUC 1.0 but nDCG@5
            # only 0.7289, because users with zero positives score 0 and are
            # counted in the average. So from the baseline there is 0.3390 to
            # gain on GAUC and 0.2007 on nDCG@5, and a diagnostician shown one
            # averaged number cannot aim at the larger gap.
            "metric_ceilings": {"GAUC": GAUC_CEILING,
                                "nDCG@5": NDCG5_CEILING,
                                "baseline_GAUC": BASELINE_GAUC,
                                "baseline_nDCG@5": BASELINE_NDCG5,
                                "baseline_gauc_to_ceiling": round(
                                    GAUC_CEILING - BASELINE_GAUC, 4),
                                "baseline_ndcg5_to_ceiling": round(
                                    NDCG5_CEILING - BASELINE_NDCG5, 4)},
            "parent_metrics": _metric_block(parent),
            # Which MECHANISM FAMILIES the evidence store has retired, which
            # ones carry unconfirmed refuting evidence, and what is left. See
            # _family_standing: without these the proposer is asked to respect a
            # constraint it cannot see, and its proposals are silently deleted
            # for violating it.
            "refuted_families": refuted_families or None,
            "probationary_families": probationary_families or None,
            "legal_families": legal_families,
        }
        if refusal:
            ctx["refusal"] = refusal
        return diag_llm.diagnose(ctx)

    def _propose(refusal: str | None = None) -> tuple[dict, List[dict]]:
        """diagnose (exhaustion budget enforced) → literature → hypotheses.

        Factored out of the straight-line body so the dedup-starvation path
        below can run the whole proposal step a second time with a refusal,
        which is what the exhausted-component path a few lines down has always
        done for its own constraint.
        """
        diag = _ask(refusal)

        # Enforce the exhaustion budget HERE, not only in the prompt. A prompt
        # request is what failed: the diagnostician kept re-deriving the same
        # bottleneck from the same flat trajectory because the trajectory really
        # did look flat, and nothing in the loop could tell it to stop.
        named = _canonical_component(diag.get("component"))
        if named in exhausted:
            if verbose:
                print(f"  diagnosis named exhausted component {named!r} "
                      f"({component_ledger[named]['scored']} scored attempts, best "
                      f"delta {component_ledger[named]['best_mean_delta']}) — "
                      f"re-asking once")
            diag = _ask(refusal=(
                f"You named {diag.get('component')!r}, which is in "
                f"exhausted_components. That component has already had "
                f"{component_ledger[named]['scored']} scored attempts whose best "
                f"paired delta was "
                f"{component_ledger[named]['best_mean_delta']}, below the "
                f"{COMPONENT_EXHAUSTED_DELTA} bar. Name the next most load-bearing "
                f"component instead, and do not name any component listed in "
                f"exhausted_components."))
            named = _canonical_component(diag.get("component"))

        if named in exhausted:
            # It insisted twice. Substitute deterministically rather than spend a
            # third call, and record that this happened so the run log shows the
            # diagnosis was overridden rather than produced.
            fallback = next((c for c in UNEXPLORED_PRIORITY
                             if c not in exhausted), None)
            if fallback is None:
                if verbose:
                    print(f"  every priority component is exhausted; keeping "
                          f"{diag.get('component')!r}")
                diag["exhaustion_note"] = ("all priority components exhausted; "
                                           "model's choice kept")
            else:
                if verbose:
                    print(f"  diagnosis insisted on exhausted {named!r}; "
                          f"falling back to {fallback!r}")
                diag["exhaustion_fallback"] = {"refused": diag.get("component"),
                                               "substituted": fallback}
                diag["component"] = fallback

        # Everything below reads diag["component"], so the substitution above
        # has to land before this line — the writer is routed at the
        # replacement, not at the component that was refused.
        evidence_card = diag_llm.ground_in_literature(diag["bottleneck"])
        # blocked_families: enforced at the SCHEMA layer, where a retry loop
        # already exists, so a proposal into a refuted family bounces back to the
        # model with the reason and the legal set appended. The driver's own
        # dedup filter below stays as the last line of defence, but it is no
        # longer the FIRST thing that notices.
        hypotheses = diag_llm.generate_hypothesis(
            diag, evidence_card, tried=tried,
            blocked_families=refuted_families or None)
        counters.bump("proposals", len(hypotheses))
        return diag, hypotheses

    def _node_for(diag: dict, h: dict, probation: dict | None = None) -> Node:
        # Each node gets its own copy of the diagnosis so a per-node annotation
        # (audit concerns, the probation marker below) cannot leak sideways into
        # its siblings' records.
        d = dict(diag)
        if probation:
            d["dedup_probation"] = probation
        return Node(id=_new_id(), parent_id=parent.id,
                    code_path=parent.code_path, code_dir=parent.code_dir,
                    operation="improve", diagnosis=d, hypothesis=h)

    def _filter(diag: dict, hypotheses: List[dict]
                ) -> tuple[List[Node], List[dict], List[dict]]:
        """Filter a proposal batch down to the candidates worth executing.

        Three deterministic gates, all free, all BEFORE a writer call is spent:

          1. DEDUP — the family carries a corroborated refutation.
          2. FEASIBILITY — the mechanism names a primitive that cannot be
             written in numpy on one core. 61 of 65 stored proposals did.
          3. DIVERSITY — one candidate per family per iteration. 4 of the 5
             candidates in the recorded run were the same idea
             (prev_video_id / prev_author_id / prev_long_view / session_depth),
             so the iteration measured one thing four times.

        Then a hard cap at MAX_CANDIDATES_PER_ITER. Returns
        (candidates, drop records, dropped hypotheses).
        """
        kept: List[Node] = []
        records: List[dict] = []
        rejects: List[dict] = []
        seen_families: set = set()

        def _drop(h, fp, reason):
            records.append({"mechanism": (h.get("mechanism") or "")[:240],
                            "family": _family_of(fp) or fp[0],
                            "fingerprint": list(fp),
                            "reason": reason})
            rejects.append(h)

        for h in hypotheses:
            fp = _fingerprint(h)
            # A family is retired only by a corroborated REFUTATION, not by any
            # prior sighting: every scored candidate is recorded here, most as
            # `inconclusive`, so an unconditional match retired a family after
            # one indecisive result even when the verdict step said
            # retry_cheaper or build_on_it. See _memory_blocks.
            if _memory_blocks(memory, fp):
                _drop(h, fp, "refuted_family")
                continue
            infeasible = _infeasible_reason(h)
            if infeasible:
                _drop(h, fp, f"infeasible: names {infeasible!r}")
                if verbose:
                    print(f"  dropped (infeasible: {infeasible!r}): "
                          f"{(h.get('mechanism') or '')[:70]}")
                continue
            fam = _family_of(fp)
            # `other`/prose-hash proposals have no family, so they never collide
            # on diversity — two novel ideas are two ideas.
            if fam is not None and fam in seen_families:
                _drop(h, fp, "duplicate_family_in_batch")
                continue
            if len(kept) >= MAX_CANDIDATES_PER_ITER:
                # Not a failure — the batch was wider than the compute budget,
                # which is the intended shape. Recorded so the log shows what was
                # proposed but deferred rather than what was rejected.
                _drop(h, fp, "over_execution_cap")
                continue
            if fam is not None:
                seen_families.add(fam)
            kept.append(_node_for(diag, h))
        return kept, records, rejects

    diag, hypotheses = _propose()
    candidates, dropped, rejects = _filter(diag, hypotheses)

    if hypotheses and not candidates:
        # STARVATION. Every proposal fingerprinted into a blocked family, so the
        # filter emptied the list. Returning [] here is what killed the recorded
        # run: zero candidates closed the parent, the parent was the root, and
        # the frontier emptied.
        #
        # The recovery is the pattern that already exists ~60 lines above for
        # exhausted components — enumerate the constraint into the context,
        # re-ask once, then substitute deterministically. The re-ask alone is not
        # sufficient and the run proved it: iteration 4's ledger already carried
        # `next_action: abandon_mechanism` for both families and
        # llm_calls/hypothesis.py's prompt already said
        # `abandon_mechanism: FORBIDDEN`, and the model proposed into them 3 for
        # 3 anyway. So the last-resort branch below is what actually guarantees
        # liveness.
        counters.bump("dedup_starved")
        blocked_now = sorted({d["family"] for d in dropped})
        # WHY the batch emptied, tallied. It is no longer necessarily dedup: the
        # feasibility and diversity gates can empty it too, and those call for
        # completely different corrections — "that primitive cannot be written
        # here" versus "you proposed the same family repeatedly".
        why = Counter(d.get("reason", "refuted_family") for d in dropped)
        why_str = ", ".join(f"{n}x {r}" for r, n in why.most_common())
        if verbose:
            print(f"  all {len(hypotheses)} hypotheses dropped ({why_str}) "
                  f"— widening the ask")
        reasons_block = ""
        if any(r.startswith("infeasible") for r in why):
            named = sorted({d["reason"].split("names ")[-1]
                            for d in dropped
                            if d.get("reason", "").startswith("infeasible")})
            reasons_block += (
                f" Some named a primitive that CANNOT be written here "
                f"({', '.join(named)}): there is no torch, tensorflow, sklearn "
                f"or pandas, only numpy and lightgbm on one core. Name the numpy "
                f"operations that implement your mechanism — array indexing, "
                f"np.add.at, a matmul, a bincount, a searchsorted — or propose "
                f"something else.")
        if why.get("duplicate_family_in_batch"):
            reasons_block += (
                " Some declared a family another hypothesis in the same batch "
                "had already claimed, which measures one idea several times.")
        diag, hyp2 = _propose(refusal=(
            f"All {len(hypotheses)} of your previous hypotheses were discarded "
            f"without being run ({why_str}), so that ask measured nothing."
            f"{reasons_block}"
            + (f" The refuted families are: {', '.join(refuted_families)}."
               if refuted_families else "")
            + f" The legal families are: {', '.join(legal_families)} — where "
              f"'{_OTHER_FAMILY}' means any mechanism matching none of the named "
              f"families, which is always legal. Propose in a DIFFERENT family "
              f"from the ones just refused, with an implementation you can name "
              f"the numpy operations for."))
        cands2, dropped2, rejects2 = _filter(diag, hyp2)
        dropped += dropped2
        rejects += rejects2
        if cands2:
            candidates = cands2
        else:
            # It insisted twice. Emit ONE probation node rather than [], so the
            # iteration measures something instead of closing a node. This
            # re-tests a blocked family on purpose: the ban's unsound step is its
            # RESOLUTION, not its measurement — `sequence_features` matches the
            # bare token "sequence" and so covers DIN, target attention, history
            # pooling and session features, none of which ever ran. A fresh
            # member of the family is genuinely untested evidence, and measuring
            # it beats measuring nothing.
            probation = {
                "reason": ("every hypothesis in this iteration was dropped by "
                           "the dedup filter, twice"),
                "blocked_families": blocked_now,
                "note": ("emitted under probation to keep the iteration live; "
                         "the family's evidence is a refutation of specific "
                         "implementations, not of the family"),
            }
            candidates = [_node_for(diag, rejects[-1], probation=probation)]
            if verbose:
                print(f"  proposer insisted on {', '.join(blocked_now)} twice — "
                      f"emitting 1 probation node so the iteration still "
                      f"measures something")

    # Attached to the returned diagnosis, NOT to the candidate nodes: run()
    # forwards it into the iteration record, and it has no business in the
    # writer's prompt. Before this the drop left no node, no ledger entry and no
    # log line — the only surviving evidence in the recorded run was the
    # arithmetic gap between counters.proposals (8) and sum(n_candidates) (5).
    if dropped:
        diag["dropped_by_dedup"] = dropped

    if verbose:
        print(f"  hypotheses={len(hypotheses)} candidates={len(candidates)}"
              f"{f' dropped_by_dedup={len(dropped)}' if dropped else ''}")
    return diag, candidates


#: The four root modules a candidate stages and the sandbox runs.
_STAGED_MODULES = ("data.py", "evaluate.py", "baseline.py", "submit.py")


def _apply_diff_to_dir(diff: str, target_dir: str) -> bool:
    """Apply a unified diff to files under target_dir. Returns True on success.

    Try `patch` first (present on macOS), fall back to `git apply`. Both are
    tried at -p1 and -p0: models emit headers with (`a/data.py`) and without
    (`data.py`) the prefix, and the wrong strip level makes patch report
    "can't find file to patch" on a diff that is otherwise fine.

    stdin=DEVNULL is load-bearing: with no strip level that resolves, `patch`
    interactively prompts "File to patch:" and would hang an unattended run.
    """
    diff_path = os.path.join(target_dir, "_patch.diff")
    with open(diff_path, "w", encoding="utf-8") as fh:
        fh.write(diff)
    for cmd in (["patch", "-p1", "-i", "_patch.diff"],
                ["patch", "-p0", "-i", "_patch.diff"],
                ["git", "apply", "--unsafe-paths", "_patch.diff"],
                ["git", "apply", "--unsafe-paths", "-p0", "_patch.diff"]):
        try:
            subprocess.run(cmd, cwd=target_dir, check=True,
                           stdin=subprocess.DEVNULL,
                           capture_output=True, text=True)
            return True
        except (subprocess.CalledProcessError, FileNotFoundError):
            continue
    return False


def _dir_sha256(directory: str,
                files: tuple[str, ...] = _STAGED_MODULES) -> str:
    """Hash the contents of the named files under directory, in order.

    Used to verify that applying a repair diff actually changed something —
    a repair may edit data.py rather than baseline.py, so hashing just
    c.code_path would miss real changes.
    """
    h = hashlib.sha256()
    for name in files:
        path = os.path.join(directory, name)
        if os.path.exists(path):
            h.update(name.encode())
            with open(path, "rb") as fh:
                h.update(fh.read())
    return h.hexdigest()


def _apply_diff_and_stage(diff: str, root: str, candidate_id: str) -> str | None:
    """Stage a candidate: copy root modules to a per-candidate dir and apply
    the diff to whichever file the diff header targets. Returns the candidate
    directory path (containing patched files), or None if the patch failed to
    apply.

    The returned directory is later passed as `root=` to codegen.execute so
    the sandbox copies FROM this patched dir instead of the pristine repo
    root — that's what makes the modification actually run.
    """
    cand_dir = os.path.join(tempfile.gettempdir(), f"candidate_{candidate_id}")
    if os.path.exists(cand_dir):
        shutil.rmtree(cand_dir)
    os.makedirs(cand_dir)
    for name in _STAGED_MODULES:
        src = os.path.join(root, name)
        if os.path.exists(src):
            shutil.copy2(src, os.path.join(cand_dir, name))
    return cand_dir if _apply_diff_to_dir(diff, cand_dir) else None


def run(max_iters: int = 50, wallclock_cap_s: int = 6 * 3600, verbose: bool = True,
        progress_path: str | None = PROGRESS_PATH,
        root_baseline_path: str | None = ROOT_BASELINE_PATH,
        confirm_baseline_path: str | None = CONFIRM_BASELINE_PATH,
        memory_path: str | None = None,
        champion_dir: str | None = None,
        # Defaults to True because the failure it prevents is silent. Mocked runs
        # and mocked tests are exempted automatically by _using_mocks(), so this
        # costs them nothing and no test has to opt out.
        require_real_models: bool = True):
    """`champion_dir` is a parameter rather than only the CHAMPION_DIR global
    for the same isolation reason progress_path and memory_path are: a mocked
    or tested run must not archive its candidates into the live
    orchestrator/_state/champions/. It defaults to the global at call time, so
    a real run needs to pass nothing.

    That default mattered immediately: the first mocked 8-iteration run and the
    driver-level tests together left 27 mock champions in the live directory,
    every one scored ~0.60 on the mocks' 10-user synthetic split. Since that
    directory is what a submission gets generated from, a real winner would
    have been indistinguishable from a mock artifact.
    """
    champion_archive_dir = champion_dir or CHAMPION_DIR
    memory = Memory(path=memory_path) if memory_path else Memory()
    counters = Counters()
    t_start = time.time()
    # The ledger is a PROCESS global, so a second run in the same process would
    # otherwise inherit the first one's tokens and report a total that belongs to
    # neither run.
    _usage_ledger().reset()

    # nodes.jsonl is append-only within a run, so a fresh run must start clean
    # or its records interleave with the previous run's under the same iter
    # numbers. progress_path=None means "write nothing", log included.
    if progress_path and os.path.exists(NODES_LOG_PATH):
        os.remove(NODES_LOG_PATH)

    # Model routing, reported once and then CHECKED. The check is not optional
    # on a real run: a FakeBackend run produces canned gate-clean edits, scores
    # them, promotes them and archives a champion, so nothing downstream can tell
    # it from a real search. See FakeBackendInRealRunError.
    model_report = _model_report()
    if verbose:
        _print_model_banner(model_report)
    if require_real_models:
        _require_real_models(model_report)

    root = _new_root()
    if verbose:
        print(f"[root] measuring unmodified baseline on valid_search "
              f"(seeds {list(ROOT_SEEDS)}) ...")
    root_measured = _measure_root(root, counters, cache_path=root_baseline_path,
                                  verbose=verbose)
    if verbose:
        n_users = len(next(iter(root.per_user_by_seed.values()), {}))
        print(f"[root] primary={root.local_best_score:.4f} "
              f"seeds={root.seeds_run} per_user_users={n_users} "
              f"measured={root_measured}")

    open_nodes: List[Node] = [root]
    # Every node ever created, in creation order. open_nodes is the frontier and
    # gets pruned; this is the full record the attempt ledger reads from, so a
    # branch that was abandoned still warns the proposal operators off it.
    all_nodes: List[Node] = [root]
    global_best = root.local_best_score
    global_best_node = root
    history: List[float] = [root.local_best_score]

    # The best candidate by scalar primary, regardless of whether it ever
    # cleared the sealed-split promotion gate. THE run this repo's state was
    # captured from promoted nothing, so global_best_node stayed the root and
    # the deliverable was the unmodified baseline — a 5-iteration search whose
    # output was its own starting point. A greedy champion is what makes the
    # run produce something; promotion stays the stronger, separate claim.
    champion = root
    champion_primary = root.local_best_score
    champion_archive: str | None = None

    # The root is line 0 of the log: the draft seed every later node descends
    # from, emitted before any iteration so the tree reads top-down.
    if progress_path:
        _append_nodes_log([root], iter_no=0)

    # Baseline per-user scores on the sealed valid_confirm split. Measured lazily
    # on the first promotion attempt so a run that never triggers one spends
    # nothing from that split's budget.
    confirm_baseline: dict | None = None

    # One entry per iteration. Kept separate from `history`, which stays a
    # promotion-only ladder because local_plateau() and llm.diagnose() both
    # read it — padding it per-iteration would trip the plateau break at ~4.
    iter_records: List[dict] = []
    iter_history: List[float] = [root.local_best_score]   # index 0 = baseline

    # Refine scheduling. Private to the run — deliberately not persisted to
    # nodes.jsonl, which records what was tried, not how it was scheduled.
    improves_since_refine = 0

    # Run-wide failure digest, keyed on (file, exception, line). See
    # FailureDigest: the two identical `encode` crashes in the recorded run were
    # SIBLINGS, so the ancestor chain structurally could not show either one the
    # other's traceback.
    digest = FailureDigest()

    # Diff-hash dedup — spans the whole run, not just one iteration. Prevents
    # burning execute calls on codegen outputs we've already tried.
    seen_diff_hashes: set = set()

    def _record_iteration(it: int, candidates: List[Node], promoted_ids: List[str],
                          global_best_at_start: float,
                          dropped_by_dedup: List[dict] | None = None) -> None:
        """Snapshot this iteration's scores, print them, rewrite the progress file.

        curr_vs_baseline is measured against the champion as it stood when the
        iteration BEGAN. Using the live value made the one iteration that
        actually promoted report a delta of +0.000000, since the candidate had
        just become the thing it was being compared to.
        """
        pairs = [(c, _scalar_primary(c)) for c in candidates]
        scored = [(c, s) for c, s in pairs if s > float("-inf")]
        best_node, iter_primary = max(scored, key=lambda cs: cs[1],
                                      default=(None, None))
        # The paired delta is the honest read on whether this iteration found
        # anything. iter_primary compares two noisy absolute numbers; mean_delta
        # is per-user paired against the parent, so a real +0.0007 shows up here
        # as significant instead of being lost in the seed spread.
        delta_pairs = [(c, c.mean_delta) for c, _ in pairs
                       if c.mean_delta is not None]
        best_delta_node, best_mean_delta = max(
            delta_pairs, key=lambda cd: cd[1], default=(None, None))

        prev = iter_history[-1]
        # Carry-forward on None: iterations where nothing scored count as
        # "no improvement," not "no data." Otherwise 3 all-failed iterations
        # would never accumulate toward the ε/N plateau signal.
        running_best = max(prev, iter_primary) if iter_primary is not None else prev
        iter_history.append(running_best)
        # Exactly the quantity local_plateau() compares against ε: the running
        # best now minus the running best three iterations ago.
        improvement_score = (iter_history[-1] - iter_history[-4]
                             if len(iter_history) >= 4 else None)

        iter_records.append({
            "iter": it,
            "elapsed_s": round(time.time() - t_start, 1),
            "baseline": global_best,          # renamed from global_best; still updates on
                                              # a confirmed promotion, but on runs without
                                              # one it equals baseline_primary
            "iter_primary": iter_primary,     # score after THIS iteration's amendment
            "running_best": iter_history[-1],
            "improvement_score": improvement_score,
            "iter_primary_node": best_node.id if best_node else None,
            "curr_vs_baseline": (iter_primary - global_best_at_start)
                                if iter_primary is not None else None,
            "best_mean_delta": best_mean_delta,
            "best_mean_delta_node": best_delta_node.id if best_delta_node else None,
            "n_candidates": len(candidates),
            "n_scored": len(scored),
            # The adaptive triage cap this iteration actually used, and the
            # parent runtime it was derived from. A cap that does not appear in
            # the log cannot be told from a timeout that was the mechanism's
            # fault.
            "triage_cap_s": _triage_cap_for(parent) if parent else None,
            "parent_clean_runtime_s": (round(parent.clean_runtime_s, 2)
                                       if parent is not None
                                       and parent.clean_runtime_s else None),
            "n_open_nodes": len(open_nodes),
            "promoted": promoted_ids,
            # The widening event, per iteration. Section 2.5 of the problem
            # statement requires per-iteration error and recovery events, and a
            # dedup drop is both: it is why an iteration produced fewer
            # candidates than proposals, and the re-ask is the recovery. Without
            # it an iteration recording n_candidates=0 gives a reader no way to
            # tell starvation from a proposer that returned nothing.
            "dropped_by_dedup": dropped_by_dedup or [],
            "dedup_starved": counters.dedup_starved,
            # Every candidate, not just the scored ones: when iter_primary is
            # null it's the unscored candidates' evidence_type that says why.
            # n_seeds is here because a 1-seed and a 3-seed primary are not the
            # same measurement, and reading the file without it hides that.
            "candidates": [{"id": c.id,
                            "primary": s if s > float("-inf") else None,
                            "n_seeds": len(c.per_seed_primary),
                            # Section 2.5 requires per-iteration GAUC / nDCG@5
                            # in the submitted run log, which a primary-only
                            # schema could not produce.
                            "GAUC": _mean_metric(c.per_seed_gauc),
                            "nDCG@5": _mean_metric(c.per_seed_ndcg5),
                            "per_seed_gauc": dict(c.per_seed_gauc),
                            "per_seed_ndcg5": dict(c.per_seed_ndcg5),
                            "mean_delta": c.mean_delta,
                            "p_positive": c.p_positive,
                            "lower_95": c.lower_95,
                            "confirm_primary": c.confirm_primary,
                            "permuted_primary": c.permuted_primary,
                            "status": c.status,
                            "evidence_type": c.evidence_type} for c, s in pairs],
            # The best candidate's two metrics and each one's distance to its OWN
            # ceiling, which is where the headroom actually is.
            "iter_metrics": _metric_block(best_node) if best_node else None,
        })

        if verbose:
            rec = iter_records[-1]
            if rec["n_scored"]:
                scores_str = ", ".join(
                    f"{c['id']}={c['primary']:.4f}/{c['n_seeds']}s"
                    for c in sorted((c for c in rec["candidates"]
                                     if c["primary"] is not None),
                                    key=lambda c: -c["primary"]))
                # The paired delta, when there is one, is the line to read: it
                # resolves gains an absolute-score comparison buries in noise.
                delta_str = ("" if rec["best_mean_delta"] is None else
                             f" | best paired delta {rec['best_mean_delta']:+.4f}")
                print(f"[iter {it}] iter_primary={rec['iter_primary']:.4f} "
                      f"({rec['curr_vs_baseline']:+.4f} vs baseline)"
                      f"{delta_str} | candidates: {scores_str}")
            else:
                # Say why there's no score, else "n/a" is unreadable in run.log.
                tally = Counter(c["evidence_type"] or "unresolved"
                                for c in rec["candidates"])
                why = ", ".join(f"{n} {k}" for k, n in tally.most_common())
                print(f"[iter {it}] iter_primary=n/a "
                      f"(0/{rec['n_candidates']} scored"
                      f"{': ' + why if why else ''})")

        if progress_path:
            _append_nodes_log(candidates, iter_no=it)

        # Real numbers, refreshed BEFORE the write rather than after the loop.
        # wallclock_s was assigned once, after the loop that writes this file, so
        # every persisted progress.json reports 0.0 — and `tokens` was
        # hand-incremented constants totalling a fabricated 13200. Both are what
        # Feasibility & Practicality is scored on.
        counters.wallclock_s = time.time() - t_start
        counters.sync_usage(_usage_ledger())

        if progress_path:
            _write_json_atomic(progress_path, {
                "updated_at": time.time(),
                "metric": "primary = (GAUC + nDCG@5) / 2 on valid_search",
                "baseline_primary": root.local_best_score,
                # The baseline's own two metrics, so every candidate's per-metric
                # numbers in `iterations` have something to be read against.
                "baseline_metrics": _metric_block(root),
                "metric_ceilings": {"GAUC": GAUC_CEILING,
                                    "nDCG@5": NDCG5_CEILING},
                # False means the root has no per-user data, so no candidate can
                # pass should_continue_locally and the tree stays flat.
                "root_measured": root_measured,
                "iters_completed": it,
                "global_best": global_best,
                "history": list(history),
                "iter_history": list(iter_history),
                "counters": asdict(counters),
                # Failure signatures seen MORE THAN ONCE. A signature seen
                # once is a bug; seen three times it is a systematic gap in what
                # the writer is being told, and that is the actionable read.
                "repeated_failures": digest.summary(),
                # Which models actually served this run, and the backend that
                # served them. Without it a reader cannot tell a real run from a
                # FakeBackend run, and cannot interpret the cost figures.
                "models": model_report,
                "iterations": iter_records,
            })

    for it in range(1, max_iters + 1):
        elapsed = time.time() - t_start
        if verbose:
            # Leading blank line, so each iteration is one visually separate
            # block in run.log. Placed here rather than at the end of the
            # iteration because the loop has several break/continue exits — one
            # leading newline gives exactly one blank line between consecutive
            # iterations without needing every exit path to remember to print it.
            #
            # current_best is the running best on valid_search (iter_history[-1]),
            # NOT global_best: global_best only moves on a confirmed promotion, so
            # printing it here would show a constant equal to `baseline` for any
            # run that never promotes, which is every run so far. The two are
            # reported separately in the final summary.
            print(f"\n[iter {it}] open_nodes={len(open_nodes)} "
                  f"current_best={iter_history[-1]:.4f} "
                  f"baseline={root.local_best_score:.4f} "
                  f"elapsed={elapsed:.0f}s")

        # The liveness invariant, asserted rather than described. It was already
        # stated in prose on the prune path below and held on only ONE of the two
        # paths that can remove a node, which is how iteration 4 of the recorded
        # run reached `n_open_nodes=0` and had "global convergence" declared over
        # it. Both removal paths now satisfy this.
        assert open_nodes, "frontier emptied — liveness invariant violated"

        if elapsed > wallclock_cap_s:
            if verbose:
                print(f"[stop] wall-clock cap at iter {it}")
            break
        if global_should_stop(open_nodes, max_iters - it + 1, global_best):
            if verbose:
                print(f"[stop] global convergence at iter {it}")
            break

        promoted_ids: List[str] = []
        # Frozen here so curr_vs_baseline reports against the champion this
        # iteration had to beat, not the one it may itself have just become.
        global_best_at_start = global_best

        # 1. Route the iteration: refine (MLE-STAR, ablation-guided) or improve.
        #
        # Two triggers, either sufficient. Cadence is the floor — every K
        # improves, look at what the pipeline is actually leaning on. The
        # plateau signal is the ceiling: it fires whenever the run needs it
        # most, which may be well before K accumulates.
        #
        # REFINE_ENABLED is off, so this whole block short-circuits and every
        # iteration takes the improve path. recent_improvement is still needed
        # downstream — it is the ε/N plateau signal the Phase 1 progress
        # schema and the diagnostician's context both report.
        if REFINE_ENABLED:
            cadence_trigger, plateau_trigger, recent_improvement = _refine_triggers(
                iter_history, improves_since_refine)
        else:
            cadence_trigger = plateau_trigger = False
            _, _, recent_improvement = _refine_triggers(iter_history, 0)

        # Ablation deltas are only meaningful against a node with measured
        # per-user data — an unmeasured node has no primary to subtract from.
        # Additionally, refine only earns its cost against a node whose
        # evolution has produced measurable headroom over baseline. On the
        # pristine root every ablation just reports the baseline's own
        # component-utility profile and picking a weakest is essentially random.
        refine_target = max(
            (n for n in open_nodes
             if n.per_user_by_seed
             and n.local_best_score > root.local_best_score
                                      + REFINE_TARGET_MIN_IMPROVEMENT),
            key=lambda n: n.local_best_score, default=None,
        ) if REFINE_ENABLED else None
        if (cadence_trigger or plateau_trigger) and refine_target is None:
            if verbose:
                print(f"[iter {it}] refine trigger fired but no target above "
                      f"baseline+{REFINE_TARGET_MIN_IMPROVEMENT} — "
                      f"improving instead")

        diag = None
        candidates: List[Node] = []
        if (cadence_trigger or plateau_trigger) and refine_target is not None:
            if verbose:
                # Name every trigger that fired, not just the first one that
                # matched: "plateau" alone would hide that the cadence floor
                # had also come due, which is the thing you want to know when
                # tuning K against the threshold.
                why = "+".join(w for w, on in (("cadence", cadence_trigger),
                                               ("plateau", plateau_trigger))
                               if on)
                print(f"[iter {it}] refine triggered by {why} "
                      f"(improvement_score={recent_improvement}, "
                      f"improves_since_refine={improves_since_refine})")

            # The refine node descends from the ablated node, not from
            # select()'s pick — so `parent` is rebound here and everything
            # downstream (writer root, no-op check, paired bootstrap) pairs
            # against the tree the ablation actually measured.
            parent = refine_target
            ablations = run_ablations(
                refine_target, refine_target.local_best_score,
                codegen_mod=codegen,
                stage_fn=_apply_diff_and_stage,
                apply_diff_fn=_apply_diff_to_dir,
                data_dir=DATA_DIR,
                counters=counters,
            )
            component = pick_weakest_component(ablations)
            if component is not None:
                abl = ABLATIONS[component]
                with open(os.path.join(refine_target.code_dir, abl.file)) as fh:
                    component_source = fh.read()
                if verbose:
                    print(f"  weakest component: {component} "
                          f"(deltas {ablations})")
                prior_refines = _prior_refines_for_component(component)
                hypothesis = llm.refine(component, component_source, ablations,
                                        list(iter_history), recent_improvement,
                                        prior_refines)
                counters.bump("proposals")
                diag = {"component": component,
                        "bottleneck": f"weakest by ablation: {component}",
                        "ablation_deltas": ablations,
                        "improvement_score": recent_improvement}
                candidates = [Node(
                    id=_new_id(), parent_id=refine_target.id,
                    code_path=refine_target.code_path,
                    code_dir=refine_target.code_dir,
                    operation="refine",
                    diagnosis=diag,
                    hypothesis=hypothesis,
                )]
                improves_since_refine = 0
            elif verbose:
                # Every ablation failed to stage or run: no evidence either
                # way, so fall through to a normal improve rather than
                # refining a component picked at random.
                print("  no ablation produced a delta — falling back to improve")

        if not candidates:
            parent = select(open_nodes)
            diag, candidates = _build_improve_candidates(
                parent, diag_llm=llm, memory=memory,
                counters=counters, history=history,
                iter_history=iter_history,
                improvement_score=recent_improvement,
                tried=_attempt_ledger(all_nodes, parent),
                component_ledger=_component_ledger(all_nodes),
                verbose=verbose,
            )
            improves_since_refine += 1

        parent.n_visits += 1
        all_nodes.extend(candidates)

        if not candidates:
            # The ROOT is never closed and never leaves the frontier.
            #
            # orchestrator/selection.py::select raises RuntimeError on an empty
            # list, so before this guard the only thing between the run and a
            # crash was that global_should_stop (convergence.py) reached the
            # empty frontier first and called it convergence. Fixing the stop
            # condition alone would therefore convert a false convergence into a
            # crash, which is why the guarantee has to live at the frontier
            # instead.
            #
            # Nothing anywhere re-opens a closed node — open_nodes.append
            # appears exactly once in this module, on the parent-acceptance
            # path — so the frontier is a ratchet and needs an explicit floor.
            # The prune path below already has this guard; this was the other of
            # the two paths that can remove a node, and it did not.
            if parent is not root:
                parent.status = "closed"
                parent.evidence_type = "invariant"
                if parent in open_nodes:
                    open_nodes.remove(parent)
            elif verbose:
                print("  parent is the root — kept on the frontier "
                      "(liveness floor)")
            # Record even here, so dedup-only iterations aren't a gap in the file.
            _record_iteration(it, [], promoted_ids, global_best_at_start,
                              dropped_by_dedup=(diag or {}).get(
                                  "dropped_by_dedup"))
            continue

        # 3. write → diff-hash dedup → gate (hard) → audit (advisory) → partial run
        def _attempt(c: Node, semantic_feedback: str | None = None) -> str:
            """One write→stage→triage-run pass. Mutates c; returns why it ended.

            Split out of the loop so a candidate rejected as a no-op can be
            re-written once with that fact fed back, instead of being recorded as
            a refuted mechanism it never actually implemented.
            """
            # Per-attempt, not per-node: a no-op rewrite gets a fresh smoke
            # budget because it is a different piece of code.
            smoke_fixes = 0
            if c.operation == "refine":
                # Route through the registry so the component name resolves to
                # the file it actually lives in. "capacity" and
                # "regularization" both mean baseline.py, which write_fix's
                # substring heuristic gets right only by accident.
                diff = codegen.write_refine(
                    c.hypothesis, component=diag["component"],
                    root=parent.code_dir,
                    semantic_feedback=semantic_feedback)
            else:
                diff = codegen.write_fix(c.hypothesis,
                                         target_component=diag["component"],
                                         root=parent.code_dir,
                                         semantic_feedback=semantic_feedback)

            # Diff-hash dedup: skip if this exact diff has already been tried.
            dh = _diff_hash(diff)
            if dh in seen_diff_hashes:
                if verbose:
                    print(f"  {c.id} skipped: identical diff already tried ({dh[:8]})")
                return "dedup"
            seen_diff_hashes.add(dh)

            if SHOW_DIFFS and verbose:
                print(f"\n--- diff for {c.id} ---")
                print(diff)
                print("--- end diff ---")

            # Hard gate: deterministic static scan.
            gate = codegen.pre_execution_gate(diff)
            if not gate["pass"]:
                if verbose:
                    print(f"  {c.id} blocked by gate: {gate['reasons']}")
                return "gate"

            # Audit is ADVISORY — concerns logged to diagnosis, do not veto.
            # The blind LLM auditor hallucinates leaks from thin context; the
            # deterministic gate above is the real gatekeeper.
            audit_res = llm.audit(diff, checklist={
                "test_label_access": True, "external_data_rule": True,
                "temporal_causality": True, "same_row_auxiliary_as_input": True})
            if not audit_res["pass"]:
                c.diagnosis = {**(c.diagnosis or {}),
                               "audit_concerns": audit_res.get("violations", [])}
                if verbose:
                    print(f"  {c.id} audit flagged (advisory): "
                          f"{audit_res.get('violations', [])}")

            # Apply the diff to a per-candidate staged directory so execute()
            # actually runs the MODIFIED code, not the pristine baseline.
            cand_dir = _apply_diff_and_stage(diff, root=parent.code_dir,
                                             candidate_id=c.id)
            if cand_dir is None:
                if verbose:
                    print(f"  {c.id} patch failed to apply (malformed diff)")
                return "patch"
            c.code_dir = cand_dir
            c.code_path = os.path.join(cand_dir, "baseline.py")

            # 3a. SMOKE STAGE — the real candidate contract on 200 synthetic
            # rows, in well under a second. Every execution failure in the
            # recorded run was detectable here: a shape mismatch, an IndexError
            # in encode(), and a mechanism whose per-row cost was immediately
            # obvious. Those cost 240s each plus up to two further 240s repair
            # runs, so one bad edit could burn 12 minutes to learn what this
            # says instantly.
            #
            # Because the signal is nearly free, the repair budget here is
            # MAX_SMOKE_FIX_ATTEMPTS (5) rather than MAX_FIX_ATTEMPTS (2): more
            # attempts at no extra cost, and only code already proven to RUN
            # ever reaches the expensive path below.
            smoke = codegen.smoke_check(cand_dir)
            counters.bump("smoke_runs")
            while not smoke["ok"] and smoke_fixes < MAX_SMOKE_FIX_ATTEMPTS:
                smoke_fixes += 1
                c.fix_attempts += 1
                c.last_error_excerpt = smoke["error"][-ANCESTOR_ERROR_CHARS:]
                if verbose:
                    print(f"  {c.id} smoke failed at {smoke.get('stage')!r} in "
                          f"{smoke.get('seconds', 0):.2f}s — repair "
                          f"{smoke_fixes}/{MAX_SMOKE_FIX_ATTEMPTS}")
                _sig = digest.record(
                    smoke["error"], node_id=c.id,
                    mechanism=(c.hypothesis or {}).get("mechanism"),
                    stage="smoke")
                repair = codegen.debug_and_retry(
                    c.code_path, smoke["error"], root=cand_dir,
                    hypothesis=c.hypothesis,
                    ancestors=_ancestor_chain(c, all_nodes),
                    prior_failures=digest.matching(smoke["error"],
                                                   exclude_node=c.id))
                if repair.get("is_semantic_change"):
                    counters.bump("semantic_retries")
                repair_diff = (repair.get("code_diff") or "").strip()
                if not repair_diff:
                    break
                digest.note_repair(_sig, node_id=c.id, diff=repair_diff)
                pre_hash = _dir_sha256(cand_dir)
                if not _apply_diff_to_dir(repair_diff, cand_dir):
                    break
                if _dir_sha256(cand_dir) == pre_hash:
                    break
                smoke = codegen.smoke_check(cand_dir)
                counters.bump("smoke_runs")
                # Whether THIS repair worked is the field that makes the digest
                # worth more than a list of tracebacks: "three candidates hit
                # this and none of the repairs worked" is actionable.
                digest.resolve(_sig, node_id=c.id, repaired=smoke["ok"])
            if not smoke["ok"]:
                counters.bump("smoke_rejects")
                # Never reaches codegen.execute. This is the whole point: the
                # 240s triage run is not spent on code that cannot survive 200
                # rows.
                c.last_error_excerpt = smoke["error"][-ANCESTOR_ERROR_CHARS:]
                if verbose:
                    print(f"  {c.id} rejected by smoke at "
                          f"{smoke.get('stage')!r} after {smoke_fixes} repair "
                          f"attempt(s); no triage run spent")
                return "smoke"
            if smoke_fixes and verbose:
                print(f"  {c.id} smoke clean after {smoke_fixes} repair "
                      f"attempt(s) ({smoke.get('seconds', 0):.2f}s)")

            # 3b. Triage run — one seed, cap scaled off the PARENT's own cost.
            triage_cap = _triage_cap_for(parent)
            _t_triage = time.time()
            res = codegen.execute(c.code_path, seed=0, split="valid_search",
                                  wallclock_cap_seconds=triage_cap,
                                  root=cand_dir, data_dir=DATA_DIR)
            counters.bump("triage_runs")
            counters.bump_scorer("valid_search")
            if res["status"] == "ok":
                # Measured, so this node's own children get a cap scaled off it.
                c.clean_runtime_s = time.time() - _t_triage

            # Record HOW it failed before repairing, so this node's own chain
            # entry is informative to its descendants even if every repair
            # attempt below also fails. The TAIL of the log, because a Python
            # traceback puts its cause at the end.
            if res["status"] != "ok":
                c.last_error_excerpt = (res.get("logs") or "")[-ANCESTOR_ERROR_CHARS:]

            # The EXECUTION repair budget stays at MAX_FIX_ATTEMPTS. Measured
            # from wherever the smoke repairs left c.fix_attempts, so a
            # candidate that needed 5 cheap repairs still gets its 2 expensive
            # ones — the two budgets buy different things and must not share a
            # counter.
            exec_fix_ceiling = c.fix_attempts + MAX_FIX_ATTEMPTS
            while res["status"] != "ok" and c.fix_attempts < exec_fix_ceiling:
                c.fix_attempts += 1
                # hypothesis and ancestors: without them the repair model saw a
                # traceback and a file, with no idea what the edit was trying to
                # do or that its ancestors had already failed the same way. This
                # operator handled 6 of the 11 candidates in the recorded run.
                _sig = digest.record(
                    res["logs"], node_id=c.id,
                    mechanism=(c.hypothesis or {}).get("mechanism"),
                    stage="execute")
                repair = codegen.debug_and_retry(
                    c.code_path, res["logs"], root=cand_dir,
                    hypothesis=c.hypothesis,
                    ancestors=_ancestor_chain(c, all_nodes),
                    prior_failures=digest.matching(res["logs"],
                                                   exclude_node=c.id))
                if repair.get("is_semantic_change"):
                    counters.bump("semantic_retries")
                repair_diff = (repair.get("code_diff") or "").strip()
                if not repair_diff:
                    break
                pre_hash = _dir_sha256(cand_dir)
                if not _apply_diff_to_dir(repair_diff, cand_dir):
                    break
                if _dir_sha256(cand_dir) == pre_hash:
                    # Patch reported success but no file changed — a common
                    # failure mode when the diff targeted the wrong path or had
                    # only empty hunks.
                    break
                digest.note_repair(_sig, node_id=c.id, diff=repair_diff)
                _t_triage = time.time()
                res = codegen.execute(c.code_path, seed=0, split="valid_search",
                                      wallclock_cap_seconds=triage_cap,
                                      root=cand_dir, data_dir=DATA_DIR)
                counters.bump("triage_runs")
                counters.bump_scorer("valid_search")
                if res["status"] == "ok":
                    c.clean_runtime_s = time.time() - _t_triage
                # Keep the excerpt on the LAST failure, not the first: after a
                # repair the crash is usually somewhere else, and that later
                # error is the one a descendant needs to avoid.
                digest.resolve(_sig, node_id=c.id,
                               repaired=res["status"] == "ok")
                if res["status"] != "ok":
                    c.last_error_excerpt = (
                        res.get("logs") or "")[-ANCESTOR_ERROR_CHARS:]
            if res["status"] == "timeout":
                # Distinct from "exec": the code was syntactically fine and ran,
                # it just did not finish inside the cap. That calls for a cheaper
                # implementation of the SAME mechanism (fewer epochs, vectorised
                # inner loop), not for abandoning the hypothesis — so it must not
                # be filed as a failed implementation.
                if verbose:
                    print(f"  {c.id} timed out at {triage_cap}s "
                          f"(4x parent runtime "
                          f"{getattr(parent, 'clean_runtime_s', None)}, clamped) "
                          f"(after {c.fix_attempts} repair attempt(s))")
                return "timeout"
            if res["status"] != "ok":
                return "exec"

            # Repaired successfully, so this node did NOT fail. Clearing the
            # excerpt keeps _ancestor_chain from telling a descendant that a
            # working ancestor is broken.
            c.last_error_excerpt = None
            c.partial_scores.append(res["metrics"]["primary"])
            _record_metrics(c, 0, res["metrics"])
            return "ok"

        #: How a failed _attempt is recorded. A no-op is NOT one of these — it
        #: ran fine and gets its own evidence type, because "the writer missed"
        #: and "the mechanism doesn't work" call for completely different fixes.
        _ATTEMPT_EVIDENCE = {"dedup": "refuted_under_context",
                             "gate": "failed_implementation",
                             "patch": "failed_implementation",
                             "exec": "failed_implementation",
                             # Failed the 200-row smoke stage after
                             # MAX_SMOKE_FIX_ATTEMPTS repairs, so it never
                             # reached a triage run. Same class as "exec" — the
                             # writer could not implement it — and deliberately
                             # NOT a statement about the mechanism.
                             "smoke": "failed_implementation",
                             # Ran, but not inside TRIAGE_WALLCLOCK_CAP_S. Its own
                             # type so the per-iteration tally in
                             # _record_iteration distinguishes "the writer can't
                             # implement this" from "this is too slow to measure",
                             # which need opposite responses.
                             "timeout": "timeout"}

        for c in candidates:
            outcome = _attempt(c)
            if outcome == "ok" and _is_no_op(c, parent, seed=0):
                # Applied, passed the gate, ran to completion — and reproduced
                # the parent's per-user scores exactly, so the executed code path
                # was never touched. Rewrite once with that fed back.
                if verbose:
                    print(f"  {c.id} scored identically to parent "
                          f"({c.per_seed_primary[0]:.6f}) — no-op, rewriting")
                counters.bump("no_op_rewrites")
                c.partial_scores.clear()
                c.per_seed_primary.clear()
                c.per_seed_gauc.clear()
                c.per_seed_ndcg5.clear()
                c.per_user_by_seed.clear()
                outcome = _attempt(c, semantic_feedback=codegen.NO_SEMANTIC_CHANGE)
                if outcome == "ok" and _is_no_op(c, parent, seed=0):
                    outcome = "no_op"
                elif outcome == "dedup":
                    # The rewrite came back byte-identical: nothing new to run.
                    outcome = "no_op"

            if outcome == "no_op":
                c.status = "closed"
                c.evidence_type = "no_op"
                if verbose:
                    print(f"  {c.id} no_op: rewrite never changed the executed path")
                continue
            if outcome != "ok":
                c.status = "closed"
                c.evidence_type = _ATTEMPT_EVIDENCE[outcome]
                continue

        # 4. triage-rank, run full seeds on survivors
        survivors = rank([c for c in candidates if c.status == "open"],
                         keep=3, wildcard=True)
        for c in survivors:
            cand_dir = os.path.dirname(c.code_path)
            for seed in (1, 2):
                r = codegen.execute(c.code_path, seed=seed, split="valid_search",
                                    wallclock_cap_seconds=FULL_RUN_WALLCLOCK_CAP_S,
                                    root=cand_dir, data_dir=DATA_DIR)
                counters.bump("full_runs")
                counters.bump_scorer("valid_search")
                if r["status"] == "ok":
                    c.seeds_run.append(seed)
                    _record_metrics(c, seed, r["metrics"])
            c.local_best_score = _scalar_primary(c)

            mean_d, p_pos, lower_95 = bootstrap_delta(
                c.per_user_by_seed, parent.per_user_by_seed)
            c.mean_delta, c.p_positive, c.lower_95 = mean_d, p_pos, lower_95

            # Grade the measurement against the criterion this hypothesis
            # declared before it was written. Nothing used to do this, so a
            # +0.0006 against a stated +0.005 closed the node with exactly the
            # same record as a mechanism that was never implemented.
            _apply_verdict(c, parent, root, llm, counters, verbose=verbose)

            # Parent acceptance is a HILL-CLIMB step, not a significance test.
            # should_continue_locally (p_pos > 0.8 and lower_95 > 0) is kept in
            # the module and still gates the sealed-split promotion below, but
            # using it here is what froze the tree: iteration 1's +0.00059
            # winner came back p_pos=0.776 / lower_95=-0.00067, was refused as a
            # parent, and every later iteration re-expanded the pristine root.
            if should_expand_as_parent(mean_d, p_pos, lower_95,
                                       min_mean_delta=HILLCLIMB_MIN_MEAN_DELTA,
                                       min_p_positive=HILLCLIMB_MIN_P_POSITIVE):
                if c not in open_nodes:
                    open_nodes.append(c)
                    if verbose:
                        print(f"  {c.id} accepted as parent "
                              f"(paired delta {mean_d:+.5f}, p_pos {p_pos:.3f})")

            # Greedy champion, independent of promotion. Archived immediately —
            # the staged dir is temporary and the run that produced
            # orchestrator/_state/ lost every candidate tree it built.
            if c.local_best_score > champion_primary:
                champion, champion_primary = c, c.local_best_score
                champion_archive = _archive_champion(
                    c, champion_primary, champion_dir=champion_archive_dir)
                if verbose:
                    print(f"[champion] {c.id} primary={champion_primary:.4f} "
                          f"({champion_primary - root.local_best_score:+.4f} vs "
                          f"baseline) archived to {champion_archive}")

            # 5. Promotion — sealed valid_confirm scorer, only when trigger clears.
            #
            # The trigger is the PAIRED bootstrap computed just above (significant
            # vs the parent, over thousands of users) plus a scalar check that the
            # candidate clears the champion. It used to be
            # `local_best_score > global_best + 0.003`, a naive compare between
            # two noisy maxima with a bar four times the largest delta this
            # search has ever produced — so a confirm query was never spent.
            if (p_pos >= PROMOTE_TRIGGER_P_POS and lower_95 > 0
                    and c.local_best_score > global_best):
                r_conf = codegen.execute(c.code_path, seed=0, split="valid_confirm",
                                         wallclock_cap_seconds=FULL_RUN_WALLCLOCK_CAP_S,
                                         root=cand_dir, data_dir=DATA_DIR)
                counters.bump_scorer("valid_confirm")
                if r_conf["status"] == "ok":
                    c.confirm_primary = r_conf["metrics"]["primary"]
                    if confirm_baseline is None:
                        confirm_baseline = _measure_confirm_baseline(
                            counters, cache_path=confirm_baseline_path,
                            verbose=verbose)
                    # Paired against the BASELINE ON THE SAME SPLIT. Comparing a
                    # valid_confirm primary against a valid_search global_best,
                    # as this did, mixes the real effect with the level
                    # difference between two different splits.
                    conf_mean, _conf_p, conf_lower = bootstrap_delta(
                        {0: r_conf["metrics"].get("per_user", {})},
                        confirm_baseline)
                    # LEAK CONTROLS, checked before the promotion is accepted.
                    # Both cost nothing when the candidate is clean, and a
                    # promotion is the ONE place worth paying for certainty: it
                    # is what a submission gets generated from.
                    leak = _leak_check(
                        c, cand_dir, counters,
                        confirm_primary=c.confirm_primary,
                        history=history, verbose=verbose)
                    if leak:
                        c.status = "closed"
                        c.evidence_type = "leak_suspected"
                        c.verdict_reason = leak
                        if verbose:
                            print(f"  {c.id} PROMOTION REFUSED — {leak}")
                    elif should_promote_globally(conf_mean, conf_lower):
                        c.status = "promoted"
                        c.evidence_type = "invariant"
                        # global_best stays on valid_search — the currency the
                        # whole search ranks and reports in. The confirm run is
                        # the gate, not the scoreboard.
                        global_best = c.local_best_score
                        global_best_node = c
                        history.append(global_best)
                        promoted_ids.append(c.id)
                        if verbose:
                            print(f"[promote] {c.id} → {global_best:.4f} "
                                  f"(valid_confirm paired delta {conf_mean:+.4f}, "
                                  f"lower95 {conf_lower:+.4f})")
                    elif verbose:
                        print(f"  {c.id} confirm failed: paired delta "
                              f"{conf_mean:+.4f} lower95 {conf_lower:+.4f}")

            # The verdict decides whether this entry RETIRES the mechanism
            # family or merely records it. Only "abandon_mechanism" is a hard
            # block (Memory.BLOCKING_EVIDENCE); "retry_cheaper" and
            # "build_on_it" must leave the family proposable, which is why the
            # dedup lookup above passes blocking_only=True instead of this
            # branch simply skipping the record. The evidence is worth keeping
            # either way — Tier 3A renders it back into the prompts.
            _evidence = c.evidence_type or "inconclusive"
            if c.next_action == "abandon_mechanism":
                _evidence = "refuted_under_context"
            memory.record(EvidenceEntry(
                fingerprint=_fingerprint(c.hypothesis),
                architecture="FM",
                loss=c.hypothesis.get("mechanism", ""),
                sampler=c.hypothesis.get("sampler", "uniform"),
                split="valid_search",
                seed_count=len(c.seeds_run),
                confidence_interval=(c.local_best_score - 0.005,
                                     c.local_best_score + 0.005),
                code_hash=c.id,
                evidence_type=_evidence,
                note=(c.verdict_reason or c.hypothesis.get("mechanism", ""))))

        # Prune the frontier to the best MAX_OPEN_NODES by scalar primary. The
        # root is never dropped, so the search can always fall back to a fresh
        # draft if every branch dies. Without a cap, select()'s UCT term spreads
        # visits over branches that already lost — the exploration bonus is
        # sqrt(log(total)/n_visits), so an untried loser outranks a proven
        # parent indefinitely.
        if len(open_nodes) > MAX_OPEN_NODES:
            keep = sorted(open_nodes, key=_scalar_primary,
                          reverse=True)[:MAX_OPEN_NODES]
            if root not in keep:
                keep[-1] = root
            for n in open_nodes:
                if n not in keep and n.status == "open":
                    n.status = "closed"
                    n.evidence_type = n.evidence_type or "inconclusive"
            open_nodes = keep

        _record_iteration(it, candidates, promoted_ids, global_best_at_start,
                          dropped_by_dedup=(diag or {}).get("dropped_by_dedup"))

        # iter_history, not history: `history` only grows on promotion, so it
        # is almost always shorter than local_plateau()'s N+1 minimum and the
        # stop condition never fires. iter_history is the per-iteration
        # running-best trajectory the plateau rule is defined against.
        if local_plateau(iter_history, epsilon=PLATEAU_STOP_EPSILON,
                         N=PLATEAU_STOP_WINDOW_N):
            if verbose:
                print(f"[stop] local plateau at iter {it} "
                      f"(no gain > {PLATEAU_STOP_EPSILON} across the last "
                      f"{PLATEAU_STOP_WINDOW_N} iterations)")
            break

    # Kept, as the authoritative end-of-run values. _record_iteration also writes
    # both per iteration so a killed run still leaves real numbers.
    counters.wallclock_s = time.time() - t_start
    counters.sync_usage(_usage_ledger())
    if verbose:
        print(f"\n[champion] {champion.id} primary={champion_primary:.4f} "
              f"({champion_primary - root.local_best_score:+.4f} vs baseline)"
              f"{'' if champion is not root else ' — the unmodified baseline'}")
        if champion_archive:
            print(f"[champion] source archived at {champion_archive}")

    # The best a candidate ever reached on valid_search, promoted or not. Kept
    # separate from global_best because they answer different questions: "did the
    # search find anything" versus "did anything survive the sealed confirm
    # split". A run where those two differ is a run whose gains did not replicate,
    # which is the single most important thing a reader needs to see.
    scored_iters = [r for r in iter_records if r["iter_primary"] is not None]
    best_iter = max(scored_iters, key=lambda r: r["iter_primary"], default=None)

    result = {"global_best": global_best,
            "global_best_node_id": global_best_node.id,
            # Per-metric, for print_final_summary. A scalar primary cannot say
            # which of the two halves a run actually moved.
            "champion_metrics": _metric_block(champion),
            "baseline_metrics": _metric_block(root),
            # The greedy best, which is what a submission should be generated
            # from when nothing cleared the sealed-split promotion gate.
            "champion_primary": champion_primary,
            "champion_node_id": champion.id,
            "champion_dir": champion_archive,
            "champion_is_baseline": champion is root,
            "baseline_primary": root.local_best_score,
            "best_valid_search": best_iter["iter_primary"] if best_iter else None,
            "best_valid_search_node_id": (best_iter["iter_primary_node"]
                                          if best_iter else None),
            "iters_completed": len(iter_records),
            "history": history,
            "counters": counters}

    # The write-up, from the run's own record rather than from memory. Written
    # only when this run persists state at all (progress_path=None means "write
    # nothing"), and never allowed to fail the run: a report is the last thing
    # that happens and losing it must not lose the numbers.
    if progress_path:
        try:
            report = codegen.synthesize_report({
                "task": "KuaiRand-Pure within-user ranking (TechJam Track 2)",
                "baseline_primary": root.local_best_score,
                "baseline_metrics": _metric_block(root),
                "metric_ceilings": {"GAUC": GAUC_CEILING,
                                    "nDCG@5": NDCG5_CEILING},
                "global_best": {"node_id": global_best_node.id,
                                "primary": global_best,
                                "mechanism": (global_best_node.hypothesis
                                              or {}).get("mechanism"),
                                "split": "valid_confirm"},
                "champion": {"node_id": champion.id,
                             "primary": champion_primary,
                             "is_baseline": champion is root,
                             "mechanism": (champion.hypothesis
                                           or {}).get("mechanism"),
                             "metrics": _metric_block(champion)},
                "promotions": max(len(history) - 1, 0),
                "iters_completed": len(iter_records),
                "counters": asdict(counters),
                "models": model_report,
                "repeated_failures": digest.summary(),
                "iterations": iter_records,
            })
            # Beside progress.json, NOT at the module-global path: a mocked run
            # and every driver-level test pass their own progress_path, and a
            # fixed global would have them all overwrite the live
            # orchestrator/_state/report.md — the same isolation trap
            # champion_dir already documents.
            report_path = os.path.join(
                os.path.dirname(progress_path) or ".",
                os.path.basename(REPORT_PATH))
            _write_text_atomic(report_path, report)
            if verbose:
                print(f"[report] written to {report_path}")
        except Exception as e:                      # noqa: BLE001 — see above
            if verbose:
                print(f"[report] unavailable ({type(e).__name__}: {e})")

    return result


def print_final_summary(result: dict) -> None:
    """The end-of-run block: best score, whether it is still the baseline, the
    baseline itself, then the counters.

    The baseline flag is the point of this function. `global_best` only advances
    on a promotion confirmed against the sealed valid_confirm split, so a run
    that found nothing prints a "best" numerically identical to the baseline —
    and read quickly, 0.5946 looks like a result rather than the absence of one.
    Saying so outright is the difference between a log that reports progress and
    a log that reports the truth.
    """
    best = result["global_best"]
    baseline = result.get("baseline_primary")
    print("\n=== final ===")

    is_baseline = baseline is not None and abs(best - baseline) < NOOP_EPSILON
    node = result.get("global_best_node_id")
    if is_baseline:
        print(f"best primary:     {best:.6f}   "
              f"** STILL THE BASELINE — nothing was promoted **")
    elif baseline is None:
        # The root measurement failed, so there is no baseline to compare to and
        # claiming a delta against the published fallback would be misleading.
        print(f"best primary:     {best:.6f}   (node {node}; "
              f"baseline UNMEASURED)")
    else:
        print(f"best primary:     {best:.6f}   "
              f"({best - baseline:+.6f} vs baseline, node {node})")
    if baseline is not None:
        print(f"baseline primary: {baseline:.6f}")

    # Only worth a line when it disagrees with global_best: that gap is exactly
    # "a candidate beat the baseline on valid_search and then failed to
    # replicate on the sealed split", which the two numbers above cannot show.
    seen = result.get("best_valid_search")
    if seen is not None and baseline is not None and seen > best + NOOP_EPSILON:
        print(f"best seen on valid_search (UNCONFIRMED): {seen:.6f} "
              f"({seen - baseline:+.6f} vs baseline, "
              f"node {result.get('best_valid_search_node_id')}) — "
              f"did not survive promotion")

    # Both metrics, with each one's own remaining headroom. The primary averages
    # two quantities whose ceilings differ (GAUC 1.0, nDCG@5 0.7289), so a run
    # reported only as a primary cannot say which half it moved or which half
    # still has room.
    champ_m = result.get("champion_metrics") or {}
    base_m = result.get("baseline_metrics") or {}
    if champ_m.get("GAUC") is not None or base_m.get("GAUC") is not None:
        print("per-metric:")
        for name, ceiling in (("GAUC", GAUC_CEILING), ("nDCG@5", NDCG5_CEILING)):
            cv, bv = champ_m.get(name), base_m.get(name)
            if cv is None and bv is None:
                continue
            line = f"  {name:8s}"
            if bv is not None:
                line += f" baseline {bv:.4f}"
            if cv is not None:
                line += f"  champion {cv:.4f}"
                if bv is not None:
                    line += f"  ({cv - bv:+.4f})"
                line += f"  {ceiling - cv:.4f} to ceiling {ceiling}"
            elif bv is not None:
                line += f"  {ceiling - bv:.4f} to ceiling {ceiling}"
            print(line)

    promotions = max(len(result.get("history") or []) - 1, 0)
    print(f"iterations: {result.get('iters_completed', '?')} | "
          f"promotions: {promotions}")

    c = result["counters"]
    print("counters:")
    by_kind = None
    for field, value in asdict(c).items():
        if field == "tokens_by_kind":
            by_kind = value            # printed as its own block below
            continue
        if isinstance(value, dict):
            value = "  ".join(f"{k}={v}" for k, v in value.items())
        elif field == "estimated_cost_usd":
            value = f"${value:.2f} (estimate — see llm_calls/usage.py)"
        elif field == "wallclock_s":
            # Enough precision that a fast mocked run reads as 0.03s rather than
            # as "0.0", which is indistinguishable from the pre-T2.3 bug where
            # this field was never written at all.
            value = (f"{value:.2f}" if value < 10 else
                     f"{value:.0f}  ({value / 60:.1f} min)")
        elif isinstance(value, float):
            value = f"{value:.1f}"
        print(f"  {field:22s} {value}")

    # The per-operator breakdown, which is the actionable half. A single scalar
    # cannot show that the writer reproduces a whole file at the output rate
    # while the auditor burns input tokens for a signal that fired on 5 of 5.
    if by_kind:
        print("tokens by operator:")
        rows = sorted(by_kind.items(),
                      key=lambda kv: -kv[1].get("estimated_cost_usd", 0))
        for kind, u in rows:
            print(f"  {kind:14s} calls={u.get('calls', 0):4d} "
                  f"in={u.get('tokens_in', 0):8d} "
                  f"(cached {u.get('tokens_cached', 0):7d}) "
                  f"out={u.get('tokens_out', 0):7d} "
                  f"reasoning={u.get('tokens_reasoning', 0):7d} "
                  f"~${u.get('estimated_cost_usd', 0):.3f}")
        if c.calls_without_usage:
            print(f"  NOTE: {c.calls_without_usage} call(s) reported no usage "
                  f"block, so these totals are an undercount")


if __name__ == "__main__":
    import argparse
    ap = argparse.ArgumentParser()
    ap.add_argument("--mock", action="store_true")
    ap.add_argument("--max-iters", type=int, default=5)
    ap.add_argument("--show-diffs", action="store_true",
                    help="print each generated diff (verbose, useful for debugging)")
    ap.add_argument("--check-models", action="store_true",
                    help="print the resolved model routing and exit. Non-zero "
                         "exit if a real run would fall back to FakeBackend. "
                         "Makes no API call.")
    ap.add_argument("--check-schemas", action="store_true",
                    help="make ONE minimal real API call per enforced JSON "
                         "schema and exit non-zero if any is rejected. Costs a "
                         "few cents; catches a 400 that would otherwise surface "
                         "several paid calls into a run.")
    args = ap.parse_args()

    SHOW_DIFFS = args.show_diffs

    if args.check_models:
        # Deliberately BEFORE the --mock swap, so this reports what a REAL run
        # would use. Costs no API call: the backends are constructed but never
        # invoked, which is exactly the check — OpenAIBackend's constructor is
        # what validates the key and the SDK import.
        _rep = _model_report()
        _print_model_banner(_rep)
        try:
            _require_real_models(_rep)
        except FakeBackendInRealRunError as e:
            print(f"\n[check-models] FAIL: {e}")
            raise SystemExit(1)
        print("\n[check-models] OK: a real run would call real models.")
        raise SystemExit(0)

    if args.check_schemas:
        # ONE minimal real call per enforced schema, to prove the API accepts
        # them. Exists because a malformed `strict` schema is a 400 at the FIRST
        # call that uses it — and in the real driver that lands several paid
        # calls into a run, after the baseline measurement, diagnose and
        # literature grounding have already been spent. A root-array schema cost
        # exactly that. Structural tests catch the known rules
        # (tests/test_model_routing.py); only the API can confirm the rest.
        #
        # Deliberately opt-in and deliberately tiny: a handful of ~50-token
        # calls, no search tool, effort "none" where the model allows it.
        from llm_calls import client as _lc
        from llm_calls.schemas import (DIAGNOSIS_JSON_SCHEMA,
                                       VERDICT_JSON_SCHEMA,
                                       hypothesis_json_schema)
        _rep = _model_report()
        _print_model_banner(_rep)
        _require_real_models(_rep)
        _cases = [("diagnosis", DIAGNOSIS_JSON_SCHEMA),
                  ("verdict", VERDICT_JSON_SCHEMA),
                  ("hypotheses", hypothesis_json_schema()),
                  ("hypotheses (narrowed)",
                   hypothesis_json_schema(["bpr_pairwise"]))]
        print(f"\n[check-schemas] {len(_cases)} minimal live calls ...")
        _failed = []
        for _name, _schema in _cases:
            try:
                _lc.call_model_text(
                    "Return the smallest valid object for the given schema.",
                    "Fill every required field with a short placeholder.",
                    max_tokens=2000, effort="none", kind="schema_check",
                    text_format=_schema)
                print(f"  OK    {_name}")
            except Exception as e:                 # noqa: BLE001
                _failed.append((_name, e))
                print(f"  FAIL  {_name}: {type(e).__name__}: "
                      f"{str(e)[:300]}")
        if _failed:
            print(f"\n[check-schemas] {len(_failed)} schema(s) rejected — a real "
                  f"run would crash at the first call using them.")
            raise SystemExit(1)
        print("\n[check-schemas] OK: every enforced schema is accepted.")
        raise SystemExit(0)

    run_kwargs = {}
    if args.mock:
        from .mocks import harness as _h, llm as _l, codegen as _c
        harness, llm, codegen = _h, _l, _c

        # A mocked run gets its own state dir. Two reasons, both load-bearing:
        #
        # 1. Correctness. With the shared cache the mock reads the REAL
        #    root_baseline.json, whose 10894 measured user ids share nothing
        #    with the mock's u0..u9 — so paired_user_deltas returns [] for
        #    every candidate, mean_delta is 0.0, should_continue_locally never
        #    passes, and the tree cannot grow past the root. The run stalls at
        #    iteration 3 and nothing past it is exercisable. This is the same
        #    trap orchestrator/tests/test_smoke.py::_isolated_caches guards.
        # 2. Isolation. A mocked run used to overwrite the live progress.json,
        #    nodes.jsonl and memory.json an overnight run had produced.
        mock_state = os.path.join("orchestrator", "_state", "mock")
        os.makedirs(mock_state, exist_ok=True)
        NODES_LOG_PATH = os.path.join(mock_state, "nodes.jsonl")
        ablation_harness.ABLATIONS_LOG_PATH = os.path.join(mock_state,
                                                           "ablations.jsonl")
        run_kwargs = {
            "progress_path": os.path.join(mock_state, "progress.json"),
            "root_baseline_path": os.path.join(mock_state, "root_baseline.json"),
            "confirm_baseline_path": os.path.join(mock_state,
                                                  "confirm_baseline.json"),
            "memory_path": os.path.join(mock_state, "memory.json"),
            # Isolated for the same reason as the paths above. Without it a
            # mocked run archives its fake champions into the live
            # orchestrator/_state/champions/ — which is the directory a
            # submission gets generated from, so a real winner would sit there
            # indistinguishable from mock source scored on 10 synthetic users.
            "champion_dir": os.path.join(mock_state, "champions"),
        }
        print(f"[mock] state dir: {mock_state}")

    result = run(max_iters=args.max_iters, **run_kwargs)
    print_final_summary(result)