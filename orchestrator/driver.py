"""Main orchestrator loop. Swap the mock imports below when real PRs land."""
import hashlib
import time
import uuid
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


def _new_id() -> str:
    return uuid.uuid4().hex[:8]


def _new_root() -> Node:
    root = Node(id=_new_id(), parent_id=None, code_path="baseline.py",
                local_best_score=0.5946)
    root.per_user_by_seed = {s: {f"u{i}": 0.5946 for i in range(10)} for s in range(3)}
    return root


def _fingerprint(h: dict):
    return (h.get("loss_type", "pointwise_logloss"),
            h.get("sampler", "uniform"),
            h.get("feature_set", "5field_baseline"),
            h.get("dataset_tier", "pure"))


def _diff_hash(diff: str) -> str:
    return hashlib.sha1(diff.encode("utf-8")).hexdigest()


def run(max_iters: int = 50, wallclock_cap_s: int = 6 * 3600, verbose: bool = True):
    memory = Memory()
    counters = Counters()
    t_start = time.time()

    root = _new_root()
    open_nodes: List[Node] = [root]
    global_best = root.local_best_score
    global_best_node = root
    history: List[float] = [root.local_best_score]

    # Diff-hash dedup — spans the whole run, not just one iteration. Prevents
    # burning execute calls on codegen outputs we've already tried.
    seen_diff_hashes: set = set()

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

            # 3a. Triage run — one seed, short cap.
            res = codegen.execute(c.code_path, seed=0, split="valid_search",
                                  wallclock_cap_seconds=120)
            counters.bump("triage_runs")
            counters.bump_scorer("valid_search")

            if res["status"] != "ok":
                repair = codegen.debug_and_retry(c.code_path, res["logs"])
                if repair.get("is_semantic_change"):
                    counters.bump("semantic_retries")
                res = codegen.execute(c.code_path, seed=0, split="valid_search",
                                      wallclock_cap_seconds=120)
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
            for seed in (1, 2):
                r = codegen.execute(c.code_path, seed=seed, split="valid_search",
                                    wallclock_cap_seconds=600)
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
                                         wallclock_cap_seconds=600)
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