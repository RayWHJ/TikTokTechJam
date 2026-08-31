"""Main orchestrator loop. Swap the mock imports below when real PRs land."""
from dotenv import load_dotenv
load_dotenv()

import os
import json
import hashlib
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

# Wall-clock cap for the 1-seed triage run. The unmodified baseline trains and
# scores valid_search in 18s measured on this machine (early-stopping at epoch
# 11 at ~1.1s/epoch), so the old 120s was only 6.7x the baseline — enough to
# kill any candidate that trains more epochs, runs a second backward pass
# (pairwise/listwise losses do), or adds a per-user aggregation. A correct but
# 8x-slower mechanism was being recorded as a failed implementation and its
# hypothesis thrown away. 240s is 13x the baseline and still well under the
# 600s full-seed cap, so a candidate that clears triage cannot then time out
# on seeds 1 and 2.
TRIAGE_WALLCLOCK_CAP_S = 240

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
            r = codegen.execute(root.code_path, seed=s, split=split,
                                wallclock_cap_seconds=wallclock_cap_s, root=".",
                                data_dir=DATA_DIR)
            counters.bump("full_runs")
            counters.bump_scorer(split)
            if r["status"] != "ok":
                if verbose:
                    print(f"[root] baseline seed={s} {r['status']}; skipping seed")
                continue
            measured[str(s)] = {"primary": r["metrics"]["primary"],
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


#: Mechanism families for fingerprinting, checked in order — first match wins,
#: so put the specific surrogates ahead of the generic "pairwise/listwise"
#: bucket they belong to. Deliberately coarse: the point is that two proposals
#: which would produce substantially the same edit collapse to one entry, not
#: that the taxonomy is complete. Unmatched mechanisms fall through to a prose
#: hash, which is the old behaviour and stays permissive for genuinely novel
#: ideas.
_MECHANISM_FAMILIES: tuple[tuple[str, tuple[str, ...]], ...] = (
    ("lambdarank_surrogate", ("lambdarank", "lambda rank", "lambda-weight",
                              "lambda weight")),
    ("ranknet_pairwise", ("ranknet", "rank net")),
    ("bpr_pairwise", ("bpr", "bayesian personalized ranking",
                      "bayesian personalised ranking")),
    ("listwise_softmax", ("listwise", "list-wise", "within-user softmax",
                          "softmax over these scores", "softmax loss")),
    ("generic_pairwise", ("pairwise loss", "pairwise ranking", "pair-wise")),
    ("multitask_auxiliary", ("multi-task", "multitask", "auxiliary task",
                             "auxiliary loss", "esmm")),
    ("sequence_features", ("sequence", "behaviour history", "behavior history",
                           "user history", "din", "target attention")),
    ("watchtime_censored", ("censored", "watch time", "watch-time",
                            "play_time")),
    ("capacity_or_regularization", ("embedding dimension", "embedding dim",
                                    "increase k", "weight decay", "dropout",
                                    "l2 regularization", "l2 regularisation")),
    ("static_feature_domains", ("add feature", "additional feature",
                               "more feature", "feature domain",
                               "extra categorical")),
    ("negative_sampling", ("negative sampling", "sample negatives",
                           "hard negative")),
    ("gbdt_swap", ("lightgbm", "gbdt", "gradient boost")),
    ("ensemble_blend", ("ensemble", "blend", "stack")),
)


def _fingerprint(h: dict):
    """Semantic fingerprint of a hypothesis.

    If the hypothesis declares structured fields, use them (this keeps the
    hand-authored preseeds in memory.py meaningful). Otherwise fall back to
    a hash of the mechanism string, so distinct proposals don't all collapse
    to the same default 4-tuple.
    """
    structured_keys = ("loss_type", "sampler", "feature_set", "dataset_tier")
    if any(k in h for k in structured_keys):
        return (h.get("loss_type", "pointwise_logloss"),
                h.get("sampler", "uniform"),
                h.get("feature_set", "5field_baseline"),
                h.get("dataset_tier", "pure"))
    # No structured fields: classify the mechanism into a FAMILY rather than
    # hashing its prose. Hashing prose made dedup dead code — every proposal in
    # the 5-iteration run was a loss swap, but each was worded differently, so
    # each hashed differently and memory.is_duplicate (an exact tuple match)
    # never fired once across 11 candidates. Matching on family means the
    # second "replace pointwise logloss with BPR" is recognised as the first
    # one no matter how it is phrased.
    mech = (h.get("mechanism") or "").strip().lower()
    sketch = (h.get("implementation_sketch") or "").strip().lower()
    text = f"{mech} {sketch}"
    family = next((fam for fam, tokens in _MECHANISM_FAMILIES
                   if any(t in text for t in tokens)), None)
    if family is not None:
        return ("mechanism_family", family, "", "")
    digest = hashlib.sha1(mech.encode("utf-8")).hexdigest()[:16]
    return ("mechanism_hash", digest, "", "")


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


#: Token cost charged per verdict call. One call per SCORED candidate only —
#: the unscored ones never reach the survivors loop — so at 3 survivors an
#: iteration this is a rounding error against the 800 the writer already spends.
VERDICT_TOKENS = 400


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
    counters.bump("tokens", VERDICT_TOKENS)
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
        }
        if refusal:
            ctx["refusal"] = refusal
        return diag_llm.diagnose(ctx)

    diag = _ask()
    counters.bump("tokens", 500)

    # Enforce the exhaustion budget HERE, not only in the prompt. A prompt
    # request is what failed: the diagnostician kept re-deriving the same
    # bottleneck from the same flat trajectory because the trajectory really did
    # look flat, and nothing in the loop could tell it to stop.
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
        counters.bump("tokens", 500)
        named = _canonical_component(diag.get("component"))

    if named in exhausted:
        # It insisted twice. Substitute deterministically rather than spend a
        # third call, and record that this happened so the run log shows the
        # diagnosis was overridden rather than produced.
        fallback = next((c for c in UNEXPLORED_PRIORITY if c not in exhausted),
                        None)
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

    # Everything below reads diag["component"], so the substitution above has to
    # land before this line — the writer is routed at the replacement, not at
    # the component that was refused.
    evidence_card = diag_llm.ground_in_literature(diag["bottleneck"])
    counters.bump("tokens", 500)
    hypotheses = diag_llm.generate_hypothesis(diag, evidence_card, tried=tried)
    counters.bump("proposals", len(hypotheses))
    counters.bump("tokens", 300 * len(hypotheses))

    candidates: List[Node] = []
    for h in hypotheses:
        fp = _fingerprint(h)
        # blocking_only: a family is retired only by a REFUTED verdict, not by
        # any prior sighting. See Memory.is_duplicate — every scored candidate is
        # recorded, most as `inconclusive`, so an unconditional match retired a
        # family after one indecisive result even when the verdict step said
        # retry_cheaper or build_on_it.
        if memory.is_duplicate(fp, blocking_only=True):
            continue
        candidates.append(Node(
            id=_new_id(), parent_id=parent.id, code_path=parent.code_path,
            code_dir=parent.code_dir,
            operation="improve",
            diagnosis=diag, hypothesis=h,
        ))
    if verbose:
        print(f"  hypotheses={len(hypotheses)} candidates={len(candidates)}")
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
        champion_dir: str | None = None):
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

    # nodes.jsonl is append-only within a run, so a fresh run must start clean
    # or its records interleave with the previous run's under the same iter
    # numbers. progress_path=None means "write nothing", log included.
    if progress_path and os.path.exists(NODES_LOG_PATH):
        os.remove(NODES_LOG_PATH)

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

    # Diff-hash dedup — spans the whole run, not just one iteration. Prevents
    # burning execute calls on codegen outputs we've already tried.
    seen_diff_hashes: set = set()

    def _record_iteration(it: int, candidates: List[Node], promoted_ids: List[str],
                          global_best_at_start: float) -> None:
        """Snapshot this iteration's scores, print them, rewrite the progress file.

        curr_vs_baseline is measured against the champion as it stood when the
        iteration BEGAN. Using the live value made the one iteration that
        actually promoted report a delta of +0.000000, since the candidate had
        just become the thing it was being compared to.
        """
        nonlocal iter_history
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
            "n_open_nodes": len(open_nodes),
            "promoted": promoted_ids,
            # Every candidate, not just the scored ones: when iter_primary is
            # null it's the unscored candidates' evidence_type that says why.
            # n_seeds is here because a 1-seed and a 3-seed primary are not the
            # same measurement, and reading the file without it hides that.
            "candidates": [{"id": c.id,
                            "primary": s if s > float("-inf") else None,
                            "n_seeds": len(c.per_seed_primary),
                            "mean_delta": c.mean_delta,
                            "p_positive": c.p_positive,
                            "lower_95": c.lower_95,
                            "confirm_primary": c.confirm_primary,
                            "status": c.status,
                            "evidence_type": c.evidence_type} for c, s in pairs],
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

        if progress_path:
            _write_json_atomic(progress_path, {
                "updated_at": time.time(),
                "metric": "primary = (GAUC + nDCG@5) / 2 on valid_search",
                "baseline_primary": root.local_best_score,
                # False means the root has no per-user data, so no candidate can
                # pass should_continue_locally and the tree stays flat.
                "root_measured": root_measured,
                "iters_completed": it,
                "global_best": global_best,
                "history": list(history),
                "iter_history": list(iter_history),
                "counters": asdict(counters),
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
                counters.bump("tokens", 800)
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
            parent.status = "closed"
            parent.evidence_type = "invariant"
            if parent in open_nodes:
                open_nodes.remove(parent)
            # Record even here, so dedup-only iterations aren't a gap in the file.
            _record_iteration(it, [], promoted_ids, global_best_at_start)
            continue

        # 3. write → diff-hash dedup → gate (hard) → audit (advisory) → partial run
        def _attempt(c: Node, semantic_feedback: str | None = None) -> str:
            """One write→stage→triage-run pass. Mutates c; returns why it ended.

            Split out of the loop so a candidate rejected as a no-op can be
            re-written once with that fact fed back, instead of being recorded as
            a refuted mechanism it never actually implemented.
            """
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
            counters.bump("tokens", 800)

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
            counters.bump("tokens", 400)
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

            # 3a. Triage run — one seed, short cap.
            res = codegen.execute(c.code_path, seed=0, split="valid_search",
                                  wallclock_cap_seconds=TRIAGE_WALLCLOCK_CAP_S,
                                  root=cand_dir, data_dir=DATA_DIR)
            counters.bump("triage_runs")
            counters.bump_scorer("valid_search")

            # Record HOW it failed before repairing, so this node's own chain
            # entry is informative to its descendants even if every repair
            # attempt below also fails. The TAIL of the log, because a Python
            # traceback puts its cause at the end.
            if res["status"] != "ok":
                c.last_error_excerpt = (res.get("logs") or "")[-ANCESTOR_ERROR_CHARS:]

            while res["status"] != "ok" and c.fix_attempts < MAX_FIX_ATTEMPTS:
                c.fix_attempts += 1
                # hypothesis and ancestors: without them the repair model saw a
                # traceback and a file, with no idea what the edit was trying to
                # do or that its ancestors had already failed the same way. This
                # operator handled 6 of the 11 candidates in the recorded run.
                repair = codegen.debug_and_retry(
                    c.code_path, res["logs"], root=cand_dir,
                    hypothesis=c.hypothesis,
                    ancestors=_ancestor_chain(c, all_nodes))
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
                res = codegen.execute(c.code_path, seed=0, split="valid_search",
                                      wallclock_cap_seconds=TRIAGE_WALLCLOCK_CAP_S,
                                      root=cand_dir, data_dir=DATA_DIR)
                counters.bump("triage_runs")
                counters.bump_scorer("valid_search")
                # Keep the excerpt on the LAST failure, not the first: after a
                # repair the crash is usually somewhere else, and that later
                # error is the one a descendant needs to avoid.
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
                    print(f"  {c.id} timed out at {TRIAGE_WALLCLOCK_CAP_S}s "
                          f"(after {c.fix_attempts} repair attempt(s))")
                return "timeout"
            if res["status"] != "ok":
                return "exec"

            # Repaired successfully, so this node did NOT fail. Clearing the
            # excerpt keeps _ancestor_chain from telling a descendant that a
            # working ancestor is broken.
            c.last_error_excerpt = None
            c.partial_scores.append(res["metrics"]["primary"])
            c.per_seed_primary[0] = res["metrics"]["primary"]
            c.per_user_by_seed[0] = res["metrics"].get("per_user", {})
            return "ok"

        #: How a failed _attempt is recorded. A no-op is NOT one of these — it
        #: ran fine and gets its own evidence type, because "the writer missed"
        #: and "the mechanism doesn't work" call for completely different fixes.
        _ATTEMPT_EVIDENCE = {"dedup": "refuted_under_context",
                             "gate": "failed_implementation",
                             "patch": "failed_implementation",
                             "exec": "failed_implementation",
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
                                    wallclock_cap_seconds=600,
                                    root=cand_dir, data_dir=DATA_DIR)
                counters.bump("full_runs")
                counters.bump_scorer("valid_search")
                if r["status"] == "ok":
                    c.seeds_run.append(seed)
                    c.per_seed_primary[seed] = r["metrics"]["primary"]
                    c.per_user_by_seed[seed] = r["metrics"].get("per_user", {})
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
                                         wallclock_cap_seconds=600,
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
                    if should_promote_globally(conf_mean, conf_lower):
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

        _record_iteration(it, candidates, promoted_ids, global_best_at_start)

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

    counters.wallclock_s = time.time() - t_start
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

    return {"global_best": global_best,
            "global_best_node_id": global_best_node.id,
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

    promotions = max(len(result.get("history") or []) - 1, 0)
    print(f"iterations: {result.get('iters_completed', '?')} | "
          f"promotions: {promotions}")

    c = result["counters"]
    print("counters:")
    for field, value in asdict(c).items():
        if isinstance(value, dict):
            value = "  ".join(f"{k}={v}" for k, v in value.items())
        elif isinstance(value, float):
            value = f"{value:.1f}"
        print(f"  {field:18s} {value}")


if __name__ == "__main__":
    import argparse
    ap = argparse.ArgumentParser()
    ap.add_argument("--mock", action="store_true")
    ap.add_argument("--max-iters", type=int, default=5)
    ap.add_argument("--show-diffs", action="store_true",
                    help="print each generated diff (verbose, useful for debugging)")
    args = ap.parse_args()

    SHOW_DIFFS = args.show_diffs

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