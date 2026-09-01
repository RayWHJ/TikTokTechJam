"""Fill parts of the deliverable templates automatically from state files.
 
Reads orchestrator/_state/{nodes.jsonl,progress.json} and
.llm_calls_cache/calls.jsonl, prints ready-to-paste markdown for the
summary table, evidence distribution, champion detail, LLM tokens by kind,
and wall-clock.
 
Usage:  python fill_deliverables.py
"""
 
from __future__ import annotations
 
import collections
import json
import os
import sys
 
 
NODES_PATH = "orchestrator/_state/nodes.jsonl"
PROGRESS_PATH = "orchestrator/_state/progress.json"
LLM_LOG_PATH = ".llm_calls_cache/calls.jsonl"
 
 
def _load_nodes():
    if not os.path.exists(NODES_PATH):
        sys.exit(f"missing {NODES_PATH}")
    # Same id can appear multiple times (once per iter it was written); last
    # write has the freshest status/scores. Dedupe by id, keep last.
    by_id = {}
    with open(NODES_PATH, encoding="utf-8") as fh:
        for line in fh:
            try:
                r = json.loads(line)
            except ValueError:
                continue
            by_id[r["id"]] = r
    return list(by_id.values())
 
 
def _load_progress():
    if not os.path.exists(PROGRESS_PATH):
        sys.exit(f"missing {PROGRESS_PATH}")
    with open(PROGRESS_PATH, encoding="utf-8") as fh:
        return json.load(fh)
 
 
def _load_llm_calls():
    if not os.path.exists(LLM_LOG_PATH):
        return []
    calls = []
    with open(LLM_LOG_PATH, encoding="utf-8") as fh:
        for line in fh:
            try:
                calls.append(json.loads(line))
            except ValueError:
                continue
    return calls
 
 
def _mean(d):
    vs = [v for v in d.values() if v is not None]
    return sum(vs) / len(vs) if vs else None
 
 
def print_summary_table(nodes):
    print("=== SUMMARY TABLE ROWS (paste into deliverable 3) ===\n")
    print("| Iter | Parent | Hypothesis | Component | GAUC / nDCG@5 / primary | "
          "delta paired | p_pos / lower_95 | Status | Notes |")
    print("|------|--------|-----------|-----------|-------------------------|"
          "----------|-------------------|--------|-------|")
    for n in sorted(nodes, key=lambda n: (n.get("iter", 0), n.get("id", ""))):
        it = n.get("iter", "?")
        parent = (n.get("parent_id") or "root")[:8]
        mech = (n.get("hypothesis") or {}).get("mechanism", "")
        hyp_short = mech[:80] + ("..." if len(mech) > 80 else "")
        comp = (n.get("diagnosis") or {}).get("component", "-")
 
        g = _mean(n.get("per_seed_gauc") or {})
        ns5 = _mean(n.get("per_seed_ndcg5") or {})
        p = _mean(n.get("per_seed_primary") or {})
        metrics = "-"
        if p is not None:
            g_str = f"{g:.4f}" if g is not None else "-"
            n_str = f"{ns5:.4f}" if ns5 is not None else "-"
            metrics = f"{g_str} / {n_str} / {p:.4f}"
 
        md = n.get("mean_delta")
        md_str = f"{md:+.4f}" if md is not None else "-"
 
        pp, l95 = n.get("p_positive"), n.get("lower_95")
        stats = f"{pp:.3f} / {l95:+.4f}" if pp is not None and l95 is not None else "-"
 
        status = n.get("status", "?")
        ev = n.get("evidence_type") or ""
        notes = ev
        if n.get("confirm_primary") is not None:
            notes = (notes + " ; " if notes else "") + f"confirm={n['confirm_primary']:.4f}"
 
        print(f"| {it} | {parent} | {hyp_short} | {comp} | {metrics} | "
              f"{md_str} | {stats} | {status} | {notes} |")
    print()
 
 
def print_evidence_distribution(nodes):
    print("=== EVIDENCE DISTRIBUTION (paste into deliverable 3) ===\n")
    print("| Status | Evidence Type | Count | IDs |")
    print("|--------|---------------|-------|-----|")
    grouped = collections.defaultdict(list)
    for n in nodes:
        key = (n.get("status", "?"), n.get("evidence_type") or "-")
        grouped[key].append(n["id"][:8])
    for (status, ev), ids in sorted(grouped.items(), key=lambda kv: -len(kv[1])):
        joined = ", ".join(ids[:5]) + ("..." if len(ids) > 5 else "")
        print(f"| {status} | {ev} | {len(ids)} | {joined} |")
    print()
 
 
def print_champion(nodes, progress):
    print("=== CHAMPION (paste into deliverable 4) ===\n")
    promoted = [n for n in nodes if n.get("status") == "promoted"]
    if not promoted:
        print("No candidates promoted this run. global_best = baseline.\n")
        return
    champ = max(promoted, key=lambda n: n.get("local_best_score") or 0)
    print(f"Node:      {champ['id']}")
    print(f"Iter:      {champ.get('iter')}")
    print(f"Parent:    {(champ.get('parent_id') or 'root')[:8]}")
    print(f"Mechanism: {(champ.get('hypothesis') or {}).get('mechanism', '')}")
    print(f"Component: {(champ.get('diagnosis') or {}).get('component', '-')}\n")
 
    g = _mean(champ.get("per_seed_gauc") or {})
    ns5 = _mean(champ.get("per_seed_ndcg5") or {})
    p = _mean(champ.get("per_seed_primary") or {})
    baseline_p = progress.get("baseline_primary")
    print("valid_search:")
    if g is not None:
        print(f"  GAUC:    {g:.4f}   (baseline 0.6674, delta {g - 0.6674:+.4f})")
    if ns5 is not None:
        print(f"  nDCG@5:  {ns5:.4f}   (baseline 0.5357, delta {ns5 - 0.5357:+.4f})")
    if p is not None and baseline_p is not None:
        print(f"  primary: {p:.4f}   (baseline {baseline_p:.4f}, delta {p - baseline_p:+.4f})")
    print("\nvalid_confirm:")
    print(f"  primary:      {champ.get('confirm_primary') or 'not measured'}")
    md = champ.get('mean_delta')
    print(f"  paired delta:     {md:+.4f}" if md is not None else "  paired delta:     -")
    l95 = champ.get('lower_95')
    print(f"  lower 95%:    {l95:+.4f}" if l95 is not None else "  lower 95%:    -")
    print(f"  p_positive:   {champ.get('p_positive')}\n")
 
 
def print_llm_totals(calls):
    print("=== LLM TOKENS (paste into deliverable 4) ===\n")
    if not calls:
        print(f"No log at {LLM_LOG_PATH}. Falling back to progress.counters.tokens\n"
              "(hardcoded estimates, understates real usage).\n")
        return
    by_kind = collections.defaultdict(lambda: {"input": 0, "output": 0, "n": 0})
    total_in = total_out = 0
    for r in calls:
        kind = r.get("kind", "unknown")
        ti = r.get("input_tokens") or 0
        to = r.get("output_tokens") or 0
        by_kind[kind]["input"] += ti
        by_kind[kind]["output"] += to
        by_kind[kind]["n"] += 1
        total_in += ti
        total_out += to
 
    print("| Kind             | Calls | Input      | Output     | Total      |")
    print("|------------------|------:|-----------:|-----------:|-----------:|")
    for kind, d in sorted(by_kind.items(),
                          key=lambda kv: -(kv[1]["input"] + kv[1]["output"])):
        print(f"| {kind:16} | {d['n']:5} | {d['input']:10,} | "
              f"{d['output']:10,} | {d['input'] + d['output']:10,} |")
    total_calls = sum(d["n"] for d in by_kind.values())
    print(f"| **TOTAL**        | {total_calls:5} | {total_in:10,} | "
          f"{total_out:10,} | {total_in + total_out:10,} |\n")
 
 
def print_wallclock(progress):
    print("=== WALL-CLOCK (paste into deliverable 4) ===\n")
    iters = progress.get("iterations", [])
    if not iters:
        print("No iteration records.\n")
        return
    total_s = iters[-1].get("elapsed_s") or 0
    n_iters = progress.get("iters_completed", len(iters))
    mean_s = total_s / n_iters if n_iters else 0
    print(f"Elapsed:      {total_s:.0f} s   ({total_s / 3600:.2f} h)")
    print(f"Iterations:   {n_iters} of 50")
    print(f"Mean / iter:  {mean_s:.0f} s   ({mean_s / 60:.1f} min)\n")
 
 
def main():
    nodes = _load_nodes()
    progress = _load_progress()
    calls = _load_llm_calls()
 
    print(f"Loaded {len(nodes)} nodes, "
          f"{len(progress.get('iterations', []))} iter records, "
          f"{len(calls)} LLM calls\n")
    print("=" * 72 + "\n")
 
    print_summary_table(nodes)
    print_evidence_distribution(nodes)
    print_champion(nodes, progress)
    print_llm_totals(calls)
    print_wallclock(progress)
 
 
if __name__ == "__main__":
    main()
 