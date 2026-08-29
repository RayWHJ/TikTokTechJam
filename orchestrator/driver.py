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
from llm_calls import diagnose, ground_in_literature, generate_hypothesis, audit
import codegen


class _LLM:
    diagnose = staticmethod(diagnose)
    ground_in_literature = staticmethod(ground_in_literature)
    generate_hypothesis = staticmethod(generate_hypothesis)
    audit = staticmethod(audit)
llm = _LLM()

from .node import Node
from .memory import Memory, EvidenceEntry
from .selection import select
from .triage import rank
from .promotion import bootstrap_delta, should_continue_locally, should_promote_globally
from .convergence import local_plateau, global_should_stop
from .counters import Counters

# Toggle from CLI. When False, diff bodies aren't printed each iteration —
# keeps run.log readable across a 50-iter run.
SHOW_DIFFS = False

# Per-iteration score record, rewritten after every iteration so an overnight
# run can be inspected mid-flight instead of waiting for run() to return.
PROGRESS_PATH = "orchestrator/_state/progress.json"

# Measured once per machine and reused: the unmodified baseline's per-user scores,
# which every candidate is paired against. ~20s per seed, so this is cheap.
ROOT_BASELINE_PATH = "orchestrator/_state/root_baseline.json"
ROOT_SEEDS = (0, 1, 2)

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
    root.local_best_score = max(v["primary"] for v in blob["seeds"].values())
    return True


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
    mech = (h.get("mechanism") or "").strip().lower()
    digest = hashlib.sha1(mech.encode("utf-8")).hexdigest()[:16]
    return ("mechanism_hash", digest, "", "")


def _diff_hash(diff: str) -> str:
    return hashlib.sha1(diff.encode("utf-8")).hexdigest()


def _best_primary(c: Node) -> float:
    """Best primary score observed for a candidate.

    Survivors get local_best_score from the full-seed runs; candidates that
    only completed a triage run keep local_best_score == -inf and carry their
    score in partial_scores. Take the max of whatever exists.
    """
    scores = list(c.partial_scores)
    if c.local_best_score > float("-inf"):
        scores.append(c.local_best_score)
    return max(scores) if scores else float("-inf")


def _write_json_atomic(path: str, payload: dict) -> None:
    os.makedirs(os.path.dirname(path), exist_ok=True)
    tmp = path + ".tmp"
    with open(tmp, "w") as fh:
        json.dump(payload, fh, indent=2)
    os.replace(tmp, path)   # atomic: tailing mid-run never sees a partial write


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
    for name in ("data.py", "evaluate.py", "baseline.py", "submit.py"):
        src = os.path.join(root, name)
        if os.path.exists(src):
            shutil.copy2(src, os.path.join(cand_dir, name))
    diff_path = os.path.join(cand_dir, "_patch.diff")
    with open(diff_path, "w") as fh:
        fh.write(diff)
    # Try `patch` first (present on macOS), fall back to `git apply`. Both are
    # tried at -p1 and -p0: models emit headers with (`a/data.py`) and without
    # (`data.py`) the prefix, and the wrong strip level makes patch report
    # "can't find file to patch" on a diff that is otherwise fine.
    #
    # stdin=DEVNULL is load-bearing: with no strip level that resolves, `patch`
    # interactively prompts "File to patch:" and would hang an unattended run.
    for cmd in (["patch", "-p1", "-i", "_patch.diff"],
                ["patch", "-p0", "-i", "_patch.diff"],
                ["git", "apply", "--unsafe-paths", "_patch.diff"],
                ["git", "apply", "--unsafe-paths", "-p0", "_patch.diff"]):
        try:
            subprocess.run(cmd, cwd=cand_dir, check=True,
                           stdin=subprocess.DEVNULL,
                           capture_output=True, text=True)
            return cand_dir
        except (subprocess.CalledProcessError, FileNotFoundError):
            continue
    return None


def run(max_iters: int = 50, wallclock_cap_s: int = 6 * 3600, verbose: bool = True,
        progress_path: str | None = PROGRESS_PATH,
        root_baseline_path: str | None = ROOT_BASELINE_PATH):
    memory = Memory()
    counters = Counters()
    t_start = time.time()

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
    global_best = root.local_best_score
    global_best_node = root
    history: List[float] = [root.local_best_score]

    # One entry per iteration. Kept separate from `history`, which stays a
    # promotion-only ladder because local_plateau() and llm.diagnose() both
    # read it — padding it per-iteration would trip the plateau break at ~4.
    iter_records: List[dict] = []

    # Diff-hash dedup — spans the whole run, not just one iteration. Prevents
    # burning execute calls on codegen outputs we've already tried.
    seen_diff_hashes: set = set()

    def _record_iteration(it: int, candidates: List[Node],
                          promoted_ids: List[str]) -> None:
        """Snapshot this iteration's scores, print them, rewrite the progress file."""
        pairs = [(c, _best_primary(c)) for c in candidates]
        scored = [(c, s) for c, s in pairs if s > float("-inf")]
        best_node, iter_primary = max(scored, key=lambda cs: cs[1],
                                      default=(None, None))
        iter_records.append({
            "iter": it,
            "elapsed_s": round(time.time() - t_start, 1),
            "global_best": global_best,       # confirmed champion (valid_confirm-gated)
            "iter_primary": iter_primary,     # score after THIS iteration's amendment
            "iter_primary_node": best_node.id if best_node else None,
            "delta_vs_global": (iter_primary - global_best)
                               if iter_primary is not None else None,
            "n_candidates": len(candidates),
            "n_scored": len(scored),
            "n_open_nodes": len(open_nodes),
            "promoted": promoted_ids,
            # Every candidate, not just the scored ones: when iter_primary is
            # null it's the unscored candidates' evidence_type that says why.
            "candidates": [{"id": c.id,
                            "primary": s if s > float("-inf") else None,
                            "status": c.status,
                            "evidence_type": c.evidence_type} for c, s in pairs],
        })

        if verbose:
            rec = iter_records[-1]
            if rec["n_scored"]:
                scores_str = ", ".join(
                    f"{c['id']}={c['primary']:.4f}"
                    for c in sorted((c for c in rec["candidates"]
                                     if c["primary"] is not None),
                                    key=lambda c: -c["primary"]))
                print(f"[iter {it}] iter_primary={rec['iter_primary']:.4f} "
                      f"({rec['delta_vs_global']:+.4f} vs global_best) "
                      f"| candidates: {scores_str}")
            else:
                # Say why there's no score, else "n/a" is unreadable in run.log.
                tally = Counter(c["evidence_type"] or "unresolved"
                                for c in rec["candidates"])
                why = ", ".join(f"{n} {k}" for k, n in tally.most_common())
                print(f"[iter {it}] iter_primary=n/a "
                      f"(0/{rec['n_candidates']} scored"
                      f"{': ' + why if why else ''})")

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
                "counters": asdict(counters),
                "iterations": iter_records,
            })

    for it in range(1, max_iters + 1):
        elapsed = time.time() - t_start
        if verbose:
            print(f"[iter {it}] open_nodes={len(open_nodes)} "
                  f"global_best={global_best:.4f} elapsed={elapsed:.0f}s")

        if elapsed > wallclock_cap_s:
            if verbose:
                print(f"[stop] wall-clock cap at iter {it}")
            break
        if global_should_stop(open_nodes, max_iters - it + 1, global_best):
            if verbose:
                print(f"[stop] global convergence at iter {it}")
            break

        parent = select(open_nodes)
        parent.n_visits += 1
        promoted_ids: List[str] = []

        # 1. diagnose + hypothesize
        diag = llm.diagnose({"parent": parent.id, "history": history})
        counters.bump("tokens", 500)
        evidence_card = llm.ground_in_literature(diag["bottleneck"])
        counters.bump("tokens", 500)
        hypotheses = llm.generate_hypothesis(diag, evidence_card)
        counters.bump("proposals", len(hypotheses))
        counters.bump("tokens", 300 * len(hypotheses))

        # 2. fingerprint dedup against memory
        candidates: List[Node] = []
        for h in hypotheses:
            fp = _fingerprint(h)
            if memory.is_duplicate(fp):
                continue
            candidates.append(Node(
                id=_new_id(), parent_id=parent.id, code_path=parent.code_path,
                diagnosis=diag, hypothesis=h,
            ))

        if verbose:
            print(f"[iter {it}] hypotheses={len(hypotheses)} candidates={len(candidates)}")

        if not candidates:
            parent.status = "closed"
            parent.evidence_type = "invariant"
            if parent in open_nodes:
                open_nodes.remove(parent)
            # Record even here, so dedup-only iterations aren't a gap in the file.
            _record_iteration(it, [], promoted_ids)
            continue

        # 3. write → diff-hash dedup → gate (hard) → audit (advisory) → partial run
        for c in candidates:
            diff = codegen.write_fix(c.hypothesis, target_component=diag["component"])
            counters.bump("tokens", 800)

            # Diff-hash dedup: skip if this exact diff has already been tried.
            dh = _diff_hash(diff)
            if dh in seen_diff_hashes:
                c.status = "closed"
                c.evidence_type = "refuted_under_context"
                if verbose:
                    print(f"  {c.id} skipped: identical diff already tried ({dh[:8]})")
                continue
            seen_diff_hashes.add(dh)

            if SHOW_DIFFS and verbose:
                print(f"\n--- diff for {c.id} ---")
                print(diff)
                print("--- end diff ---")

            # Hard gate: deterministic static scan.
            gate = codegen.pre_execution_gate(diff)
            if not gate["pass"]:
                c.status = "closed"
                c.evidence_type = "failed_implementation"
                if verbose:
                    print(f"  {c.id} blocked by gate: {gate['reasons']}")
                continue

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
            cand_dir = _apply_diff_and_stage(diff, root=".", candidate_id=c.id)
            if cand_dir is None:
                c.status = "closed"
                c.evidence_type = "failed_implementation"
                if verbose:
                    print(f"  {c.id} patch failed to apply (malformed diff)")
                continue
            c.code_path = os.path.join(cand_dir, "baseline.py")

            # 3a. Triage run — one seed, short cap.
            res = codegen.execute(c.code_path, seed=0, split="valid_search",
                                  wallclock_cap_seconds=120,
                                  root=cand_dir, data_dir=DATA_DIR)
            counters.bump("triage_runs")
            counters.bump_scorer("valid_search")

            if res["status"] != "ok":
                repair = codegen.debug_and_retry(c.code_path, res["logs"])
                if repair.get("is_semantic_change"):
                    counters.bump("semantic_retries")
                res = codegen.execute(c.code_path, seed=0, split="valid_search",
                                      wallclock_cap_seconds=120,
                                      root=cand_dir, data_dir=DATA_DIR)
                counters.bump("triage_runs")
                counters.bump_scorer("valid_search")
                if res["status"] != "ok":
                    c.status = "closed"
                    c.evidence_type = "failed_implementation"
                    continue

            c.partial_scores.append(res["metrics"]["primary"])
            c.per_user_by_seed[0] = res["metrics"].get("per_user", {})

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
                    c.local_best_score = max(c.local_best_score, r["metrics"]["primary"])
                    c.per_user_by_seed[seed] = r["metrics"].get("per_user", {})

            mean_d, p_pos, lower_95 = bootstrap_delta(
                c.per_user_by_seed, parent.per_user_by_seed)
            upper_bound = mean_d + 2 * abs(mean_d - lower_95)

            if should_continue_locally(mean_d, p_pos, upper_bound):
                if c not in open_nodes:
                    open_nodes.append(c)

            # 5. Promotion — sealed valid_confirm scorer, only when trigger clears.
            if c.local_best_score > global_best + 0.003:
                r_conf = codegen.execute(c.code_path, seed=0, split="valid_confirm",
                                         wallclock_cap_seconds=600,
                                         root=cand_dir, data_dir=DATA_DIR)
                counters.bump_scorer("valid_confirm")
                if r_conf["status"] == "ok":
                    confirm_delta = r_conf["metrics"]["primary"] - global_best
                    confirm_lower = confirm_delta - 0.002
                    if should_promote_globally(confirm_delta, confirm_lower):
                        c.status = "promoted"
                        c.evidence_type = "invariant"
                        global_best = r_conf["metrics"]["primary"]
                        global_best_node = c
                        history.append(global_best)
                        promoted_ids.append(c.id)
                        if verbose:
                            print(f"[promote] {c.id} → {global_best:.4f}")

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
                evidence_type=c.evidence_type or "inconclusive",
                note=c.hypothesis.get("mechanism", "")))

        _record_iteration(it, candidates, promoted_ids)

        if local_plateau(history):
            if verbose:
                print(f"[stop] local plateau at iter {it}")
            break

    counters.wallclock_s = time.time() - t_start
    return {"global_best": global_best,
            "global_best_node_id": global_best_node.id,
            "history": history,
            "counters": counters}


if __name__ == "__main__":
    import argparse
    ap = argparse.ArgumentParser()
    ap.add_argument("--mock", action="store_true")
    ap.add_argument("--max-iters", type=int, default=5)
    ap.add_argument("--show-diffs", action="store_true",
                    help="print each generated diff (verbose, useful for debugging)")
    args = ap.parse_args()

    SHOW_DIFFS = args.show_diffs

    if args.mock:
        from .mocks import harness as _h, llm as _l, codegen as _c
        harness, llm, codegen = _h, _l, _c

    result = run(max_iters=args.max_iters)
    print("\n=== final ===")
    print("best:", result["global_best"])
    print("history:", result["history"])
    print("counters:", result["counters"])