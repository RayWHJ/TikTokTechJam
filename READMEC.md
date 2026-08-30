# TikTok TechJam 2026 — Team Build Plan

**Repo:** https://github.com/RayWHJ/TikTokTechJam
**Team:** 4 people, working fully in parallel from hour zero (see "Frozen interface
contract" below — it removes the need for a live kickoff meeting).

The starter kit is already in this repo at the root: `data.py`, `evaluate.py`,
`baseline.py`, `submit.py`, `baseline_scores.json`, `ablation_features.py`, and the
downloaded dataset under `KuaiRand-Pure/data/`. **None of these get modified** —
new work adds *new* top-level packages that wrap/call these files, per the layout
below. The original starter-kit documentation is preserved unchanged further down
this README (in Chinese) — that content is the organizer's authoritative reference
for the task definition, baseline scores, and submission format; don't restate or
translate it, just read it once.

---

## Repo hygiene note (do this before anyone pushes a branch)

`KuaiRand-Pure.tar.gz` (47MB), the extracted `KuaiRand-Pure/data/` folder, and
`__pycache__/` are currently committed directly. Add a `.gitignore` before more
branches get pushed, or every push re-uploads the dataset:

```
KuaiRand-Pure.tar.gz
KuaiRand-Pure/data/
__pycache__/
*.pyc
```

---

## Team roles and new packages added on top of this repo

| Person | Role | New package (added at repo root) | Wraps |
|---|---|---|---|
| **A** | Harness Engineer | `harness/` | `data.py`, `evaluate.py`, `baseline.py` |
| **B** | Orchestrator Engineer | `orchestrator/` | calls A, C, D |
| **C** | Reasoning Engineer | `llm_calls/` | none — standalone |
| **D** | Execution Engineer | `codegen/` | `baseline.py`, `data.py`, `submit.py` |

**Git workflow:** each person works on their own branch off `main`
(`a-harness`, `b-orchestrator`, `c-llm-calls`, `d-codegen`), commits their package
only, opens a PR into `main` as soon as their own unit tests pass. Person B merges
last and owns final integration on the shared branch, since `orchestrator/` is the
only package that imports all three others.

---

## Frozen interface contract v1 — implement exactly this, don't renegotiate mid-build

```
── Person A owns: harness/  (wraps existing root files, does not edit them) ──

harness.validated_evaluate(user_ids: list, labels: list[int], scores: list[float],
                            split_name: str) -> dict
    calls the real, unmodified evaluate.py:evaluate() after validating input
    returns: {"GAUC": float, "nDCG@5": float, "primary": float, "users": int, "rows": int}
    raises: ValueError on unequal lengths, NaN/Inf, non-binary labels, empty input,
            or a row count that doesn't match split_name's expected size

harness.get_split(name: str) -> tuple[X, y, user_ids]
    name in {"train", "valid_search", "valid_confirm", "test"}
    built on top of data.py's load()/encode() — do not modify data.py itself
    valid_search  = dates 20220422–20220426 (5 days)  — touched every iteration
    valid_confirm = dates 20220427–20220428 (2 days)  — SEALED, touched only when
                    promoting a new global-best candidate
    test          = ONE-SHOT — callable exactly once per process;
                    raises RuntimeError on any 2nd call

harness.check_provenance(column_names: list[str], point_in_time: bool = False) -> None
    raises ValueError if any name is in NON_CAUSAL_COLUMNS and point_in_time is False
    NON_CAUSAL_COLUMNS (every column in video_features_statistic_pure.csv except
    video_id — this file is under KuaiRand-Pure/data/):
        show_cnt, show_user_num, play_cnt, play_user_num, play_duration,
        complete_play_cnt, complete_play_user_num, valid_play_cnt, valid_play_user_num,
        long_time_play_cnt, long_time_play_user_num, short_time_play_cnt,
        short_time_play_user_num, play_progress, comment_stay_duration, like_cnt,
        like_user_num, click_like_cnt, double_click_cnt, cancel_like_cnt,
        cancel_like_user_num, comment_cnt, comment_user_num, direct_comment_cnt,
        reply_comment_cnt, delete_comment_cnt, delete_comment_user_num,
        comment_like_cnt, comment_like_user_num, follow_cnt, follow_user_num,
        cancel_follow_cnt, cancel_follow_user_num, share_cnt, share_user_num,
        download_cnt, download_user_num, report_cnt, report_user_num,
        reduce_similar_cnt, reduce_similar_user_num, collect_cnt, collect_user_num,
        cancel_collect_cnt, cancel_collect_user_num, direct_comment_user_num,
        reply_comment_user_num, share_all_cnt, share_all_user_num, outsite_share_all_cnt

── Person B owns: orchestrator/ ────────────────────────────────────────

orchestrator.Node  (dataclass)
    id: str, parent_id: str | None, code_path: str, diagnosis: dict | None,
    hypothesis: dict | None, status: str  # "open"|"closed"|"promoted"
    local_best_score: float, seeds_run: list[int],
    evidence_type: str | None  # "invariant"|"refuted_under_context"|
                                 # "failed_implementation"|"inconclusive"

── Person C owns: llm_calls/ ───────────────────────────────────────────

llm.diagnose(node_context: dict) -> dict
    returns: {"bottleneck": str, "evidence": str, "confidence": float,
              "component": str, "edit_radius": "small"|"large",
              "expected_cost": str, "incompatibilities": list[str], "uncertainty": float}

llm.ground_in_literature(bottleneck: str) -> dict
    returns: {"mechanism": str, "assumptions": list[str],
              "contradictory_findings": list[str], "dataset_compatibility": list[str],
              "implementation_cost": str, "primary_citation": str}

llm.generate_hypothesis(diagnosis: dict, evidence_card: dict) -> list[dict]
    returns: list of {"mechanism": str, "success_criterion_paired": str,
                       "implementation_sketch": str}

llm.audit(diff: str, checklist: dict) -> dict
    returns: {"pass": bool, "violations": list[str], "notes": str}

── Person D owns: codegen/  (wraps baseline.py/data.py/submit.py, doesn't edit them) ──

codegen.write_fix(hypothesis: dict, target_component: str) -> str    # diff text
codegen.pre_execution_gate(code_diff: str) -> dict
    returns: {"pass": bool, "reasons": list[str]}
codegen.execute(code_path: str, seed: int, split: str, wallclock_cap_seconds: int) -> dict
    returns: {"status": "ok"|"error"|"timeout"|"diverged", "metrics": dict, "logs": str}
codegen.debug_and_retry(code_path: str, error_context: str) -> dict
    returns: {"code_diff": str, "is_semantic_change": bool}
codegen.check_submission(path: str, split: str) -> bool
    wraps submit.py's --check exactly — never reads labels
```

---

## Person A — Harness Engineer

### Prompt for Person A

```
You are helping me build the data & evaluation foundation for our team's hackathon
repo: https://github.com/RayWHJ/TikTokTechJam (TikTok TechJam 2026, Track 2:
Autonomous ML Research Agent, KuaiRand-Pure dataset, metrics GAUC + nDCG@5). I'm
Person A on a 4-person team, working on branch a-harness. My teammates are building
against a FROZEN interface contract, so I need to implement exactly the signatures
below — no renegotiating them.

The repo already contains, at its root, the files I'll paste: data.py, evaluate.py,
baseline.py, submit.py, baseline_scores.json. The actual dataset is already
downloaded under KuaiRand-Pure/data/ in the repo. I will NOT modify these root
files — I'm adding a new package harness/ that wraps them.

Build me a Python package harness/ implementing exactly this frozen contract:

1. harness.validated_evaluate(user_ids, labels, scores, split_name) -> dict
   Returns {"GAUC": float, "nDCG@5": float, "primary": float, "users": int, "rows": int}.
   Calls the real, UNMODIFIED evaluate.py:evaluate(), but only after validating:
   equal-length inputs, no NaN/Inf, binary 0/1 labels, non-empty, and row count
   matches what's expected for split_name. Raise ValueError with a clear message on
   any violation — never silently return a wrong number.

2. harness.get_split(name) -> (X, y, user_ids)
   Built on top of data.py's load()/encode() — do not modify data.py itself.
   name in {"train", "valid_search", "valid_confirm", "test"}.
   - valid_search  = dates 20220422–20220426 (5 days), freely callable any number of
     times.
   - valid_confirm = dates 20220427–20220428 (2 days), freely callable but meant to
     be used sparingly by callers — just implement the date split correctly, don't
     add extra restriction on your end.
   - test = ONE-SHOT: implement a module-level flag so this raises RuntimeError on
     any call after the first. Log the timestamp and calling stack on the one
     successful call.

3. harness.check_provenance(column_names, point_in_time=False) -> None
   Raise ValueError if any name in column_names is in this frozen NON_CAUSAL_COLUMNS
   set unless point_in_time=True (every column in
   KuaiRand-Pure/data/video_features_statistic_pure.csv except video_id):
   show_cnt, show_user_num, play_cnt, play_user_num, play_duration,
   complete_play_cnt, complete_play_user_num, valid_play_cnt, valid_play_user_num,
   long_time_play_cnt, long_time_play_user_num, short_time_play_cnt,
   short_time_play_user_num, play_progress, comment_stay_duration, like_cnt,
   like_user_num, click_like_cnt, double_click_cnt, cancel_like_cnt,
   cancel_like_user_num, comment_cnt, comment_user_num, direct_comment_cnt,
   reply_comment_cnt, delete_comment_cnt, delete_comment_user_num, comment_like_cnt,
   comment_like_user_num, follow_cnt, follow_user_num, cancel_follow_cnt,
   cancel_follow_user_num, share_cnt, share_user_num, download_cnt, download_user_num,
   report_cnt, report_user_num, reduce_similar_cnt, reduce_similar_user_num,
   collect_cnt, collect_user_num, cancel_collect_cnt, cancel_collect_user_num,
   direct_comment_user_num, reply_comment_user_num, share_all_cnt, share_all_user_num,
   outsite_share_all_cnt

4. Golden fixture tests (pytest) for validated_evaluate covering: AUC with tied
   scores, an all-positive user, an all-negative user, mismatched-length input (must
   raise), NaN input (must raise), non-binary label input (must raise).

5. A baseline sanity check: run baseline.py --model fm for 5 seeds, average the
   result, and compare that MEAN against baseline_scores.json's 5-seed mean (not a
   single seed) within a tolerance derived from the reported std of 0.0008.

Do NOT wait for or ask about anything from teammates — this contract is final. Give
me working Python with docstrings matching these exact signatures, ready to commit
on branch a-harness and open a PR into main.
```

---

## Person B — Orchestrator Engineer

### Prompt for Person B

```
You are helping me build the core decision loop for our team's hackathon repo:
https://github.com/RayWHJ/TikTokTechJam (TikTok TechJam 2026, Track 2: Autonomous ML
Research Agent, KuaiRand-Pure, metrics GAUC + nDCG@5). I'm Person B on a 4-person
team, working on branch b-orchestrator. My teammates are building modules called
harness (Person A), llm_calls (Person C), and codegen (Person D) against a FROZEN
interface contract. I will NOT have their real code today — I need to build my own
logic against exact mock stubs matching that contract, so I can fully test my own
module in isolation right now. I'm the one who does final integration once all four
branches are ready, so keep everything cleanly swappable.

Here is the frozen contract for the 3 modules I depend on — implement lightweight
mock/stub versions of ALL of these first, with hardcoded plausible return values,
so I have something to call while building:

harness.validated_evaluate(user_ids, labels, scores, split_name) -> dict
    returns {"GAUC": float, "nDCG@5": float, "primary": float, "users": int, "rows": int}
harness.get_split(name) -> (X, y, user_ids)   # name in train/valid_search/valid_confirm/test
harness.check_provenance(column_names, point_in_time=False) -> None  # raises or no-op

llm.diagnose(node_context: dict) -> dict
    returns {"bottleneck": str, "evidence": str, "confidence": float, "component": str,
             "edit_radius": "small"|"large", "expected_cost": str,
             "incompatibilities": list[str], "uncertainty": float}
llm.ground_in_literature(bottleneck: str) -> dict
    returns {"mechanism": str, "assumptions": list[str], "contradictory_findings": list[str],
             "dataset_compatibility": list[str], "implementation_cost": str,
             "primary_citation": str}
llm.generate_hypothesis(diagnosis: dict, evidence_card: dict) -> list[dict]
    returns list of {"mechanism": str, "success_criterion_paired": str,
                      "implementation_sketch": str}
llm.audit(diff: str, checklist: dict) -> dict
    returns {"pass": bool, "violations": list[str], "notes": str}

codegen.write_fix(hypothesis: dict, target_component: str) -> str      # diff text
codegen.pre_execution_gate(code_diff: str) -> dict
    returns {"pass": bool, "reasons": list[str]}
codegen.execute(code_path: str, seed: int, split: str, wallclock_cap_seconds: int) -> dict
    returns {"status": "ok"|"error"|"timeout"|"diverged", "metrics": dict, "logs": str}
codegen.debug_and_retry(code_path: str, error_context: str) -> dict
    returns {"code_diff": str, "is_semantic_change": bool}
codegen.check_submission(path: str, split: str) -> bool

Now build my real module, orchestrator/, implementing exactly this contract for MY
part:

orchestrator.Node dataclass: id, parent_id, code_path, diagnosis, hypothesis,
    status ("open"|"closed"|"promoted"), local_best_score, seeds_run, evidence_type
    ("invariant"|"refuted_under_context"|"failed_implementation"|"inconclusive")

1. memory.py — a persisted store (JSON or SQLite) of typed evidence entries with the
   evidence_type values above, each carrying architecture, loss, sampler, split,
   seed_count, confidence_interval, code_hash. A dedup lookup keyed on a structured
   fingerprint tuple (loss_type, sampler, feature_set, dataset_tier) — not embedding
   similarity. Pre-seed this memory store with two already-known dead ends from the
   repo's README: adding the full CWM 13-feature set gave no gain over the 5-field
   baseline, and increasing FM embedding dim (k=8/16/32) gave no gain either — mark
   both "refuted_under_context" so the dedup check blocks re-trying them for free.

2. selection.py — node selection across open Nodes: expected value + an exploration
   bonus for under-tried branches + a cost-normalization term (score gain per unit
   wall-clock spent).

3. triage.py — multi-fidelity ranking of partial-run candidates using posterior
   uncertainty over the likely final score, not just an extrapolated mean from fixed
   early epochs. Include one reserved "wildcard" slot per round.

4. promotion.py — candidate-minus-parent delta on matched seeds, bootstrapped over
   user blocks. Continue locally if P(delta>0) > 0.8 and the upper bound could still
   clear a practical margin. Promote to global-best only via harness.get_split(
   "valid_confirm") with mean delta > 0.002 AND a positive one-sided 95% lower bound.

5. convergence.py — local: loose plateau heuristic. Global: stop when the estimated
   probability that any remaining open node can still clear the practical margin
   within remaining budget drops below a threshold.

6. driver.py — the main loop calling the mock stubs from harness/llm/codegen above,
   plus honest counters kept SEPARATELY: proposals generated, triage/partial runs,
   full runs, semantic retries (each as a new trial id via is_semantic_change),
   scorer queries per split, wall-clock, tokens.

Give me working, type-hinted, docstringed Python for both the mocks and the real
orchestrator/ package, structured so swapping a mock for a teammate's real module
later (after their PRs land) is a one-line import change. Ready to commit on branch
b-orchestrator.
```

---

## Person C — Reasoning Engineer

### Prompt for Person C

```
You are helping me build the LLM-calling layer for our team's hackathon repo:
https://github.com/RayWHJ/TikTokTechJam (TikTok TechJam 2026, Track 2: Autonomous ML
Research Agent, KuaiRand-Pure dataset, task = within-user ranking, metrics
GAUC + nDCG@5, FM baseline test primary 0.5946). I'm Person C on a 4-person team,
working on branch c-llm-calls. My module is fully independent — it only needs to
call a model API and validate JSON, it doesn't touch the dataset, tree, or execution
sandbox at all — so I can build and test it completely standalone.

Build me a Python package llm_calls/ implementing exactly this frozen contract:

1. diagnose(node_context: dict) -> dict
   Persona: an ML-engineering diagnostician ("MLE-STAR"). Given a node's metric
   history and dataset context (KuaiRand-Pure: 5 categorical fields — user_id,
   video_id, author_id, tab, dur_bucket — long_view label, FM baseline scoring
   0.5946 test primary), identify the single biggest bottleneck. Return STRICT JSON:
   {"bottleneck": str, "evidence": str, "confidence": float 0-1, "component": str,
    "edit_radius": "small"|"large", "expected_cost": str,
    "incompatibilities": list[str], "uncertainty": float 0-1}

2. ground_in_literature(bottleneck: str) -> dict
   Use a retrieval-enabled model to find published techniques relevant to the
   bottleneck (e.g. pairwise/listwise ranking losses, sequential user-history models
   like DIN/SIM, multi-task learning for recommenders like ESMM). Return STRICT JSON:
   {"mechanism": str, "assumptions": list[str], "contradictory_findings": list[str],
    "dataset_compatibility": list[str], "implementation_cost": str,
    "primary_citation": str}
   Cache results by bottleneck string so repeat lookups don't re-call the API.

3. generate_hypothesis(diagnosis: dict, evidence_card: dict) -> list[dict]
   Persona: "AI-Scientist". Produce 1 hypothesis by default, up to 3 if
   diagnosis["confidence"] is low or context suggests a plateau. Each as STRICT JSON:
   {"mechanism": str, "success_criterion_paired": str — phrased as a
    candidate-minus-parent delta claim on a NAMED validation tier, never a flat
    absolute threshold, "implementation_sketch": str}

4. audit(diff: str, checklist: dict) -> dict
   BLIND persona — receives ONLY the code diff string and a fixed checklist dict
   (keys like: test_label_access, external_data_rule, temporal_causality,
   same_row_auxiliary_as_input). It must NOT receive the hypothesis or rationale
   that motivated the diff. Return STRICT JSON:
   {"pass": bool, "violations": list[str], "notes": str}

5. dedup_fingerprint_match(candidate_fingerprint: tuple, memory_entries: list) -> bool
   Mostly deterministic tuple comparison; escalate only genuinely ambiguous
   near-duplicate cases to a cheap model call for a yes/no with one-line reasoning.

For all 5 functions: validate the model's JSON response against a schema; on
malformed output, retry up to 2 times with the parse error appended to the prompt;
if it still fails, raise a clear typed exception (e.g. LLMSchemaError) rather than
crashing or returning garbage — a teammate's orchestrator needs to catch this
cleanly.

Since I have no teammate code to test against yet, also write a small standalone
test harness that calls each function with a few hand-written fake node_context /
diff / checklist inputs, so I can verify the schema validation and retry logic work
correctly on their own, independent of anyone else's module.

Write out the full system prompt text for each of the 5 personas as string
constants I can tune later, plus the shared retry/validation wrapper they all use.
Ready to commit on branch c-llm-calls.
```

---

## Person D — Execution Engineer

### Prompt for Person D

```
You are helping me build the code-generation and execution layer for our team's
hackathon repo: https://github.com/RayWHJ/TikTokTechJam (TikTok TechJam 2026,
Track 2: Autonomous ML Research Agent, KuaiRand-Pure dataset, baseline is a
Factorization Machine in baseline.py, feature encoding in data.py). I'm Person D on
a 4-person team, working on branch d-codegen. The repo already has everything I
need at its root: data.py, evaluate.py, baseline.py, submit.py, and I have a frozen
list of non-causal columns (below) — I don't need to wait on any teammate's code. I
will NOT modify baseline.py/data.py/submit.py directly — codegen/ wraps and
generates diffs against them.

I'll paste baseline.py, data.py, evaluate.py, and submit.py for context.

Frozen NON_CAUSAL_COLUMNS list (every column in
KuaiRand-Pure/data/video_features_statistic_pure.csv except video_id — these must
not be used as model inputs unless explicitly marked point_in_time-safe):
show_cnt, show_user_num, play_cnt, play_user_num, play_duration, complete_play_cnt,
complete_play_user_num, valid_play_cnt, valid_play_user_num, long_time_play_cnt,
long_time_play_user_num, short_time_play_cnt, short_time_play_user_num,
play_progress, comment_stay_duration, like_cnt, like_user_num, click_like_cnt,
double_click_cnt, cancel_like_cnt, cancel_like_user_num, comment_cnt,
comment_user_num, direct_comment_cnt, reply_comment_cnt, delete_comment_cnt,
delete_comment_user_num, comment_like_cnt, comment_like_user_num, follow_cnt,
follow_user_num, cancel_follow_cnt, cancel_follow_user_num, share_cnt,
share_user_num, download_cnt, download_user_num, report_cnt, report_user_num,
reduce_similar_cnt, reduce_similar_user_num, collect_cnt, collect_user_num,
cancel_collect_cnt, cancel_collect_user_num, direct_comment_user_num,
reply_comment_user_num, share_all_cnt, share_all_user_num, outsite_share_all_cnt

Build me a Python package codegen/ implementing exactly this frozen contract:

1. writer.py — write_fix(hypothesis: dict, target_component: str) -> str (a code
   diff/patch as text). Calls a code-generation model with the relevant existing
   file's content as context plus hypothesis["mechanism"] and
   hypothesis["implementation_sketch"]. Route by target_component: if it's about
   features/history/auxiliary signals, frame the prompt as "extend data.py's feature
   encoding"; otherwise frame it as "modify baseline.py's model/loss/training loop."

2. gate.py — pre_execution_gate(code_diff: str) -> dict, returning
   {"pass": bool, "reasons": list[str]}. A STATIC, deterministic (no LLM) AST/regex
   scanner run BEFORE any compute is spent. Block if the diff:
   - references test-split files/variables (anything with "test" combined with data
     loading patterns),
   - uses any name from NON_CAUSAL_COLUMNS above without an explicit
     point_in_time=True marker nearby,
   - imports external pretrained weights or datasets,
   - feeds a same-row auxiliary signal (is_like, is_follow, is_comment, is_forward,
     play_time_ms) into the model as an input feature array rather than only as a
     loss target.
   Bias toward blocking on ambiguity.

3. sandbox.py — execute(code_path: str, seed: int, split: str,
   wallclock_cap_seconds: int) -> dict, returning
   {"status": "ok"|"error"|"timeout"|"diverged", "metrics": dict, "logs": str}.
   Run the candidate as an isolated subprocess. If feasible, make any file with
   "test" in its name physically absent from the subprocess's working directory, so
   an attempted leak fails with FileNotFoundError rather than relying only on the
   static scanner. Enforce the wall-clock cap; watch stdout/stderr for NaN/inf or
   loss-divergence patterns.

4. debug.py — debug_and_retry(code_path: str, error_context: str) -> dict, returning
   {"code_diff": str, "is_semantic_change": bool}. On failure, call the code model
   with the traceback to produce a fix, retried up to 2 times. Also implement a
   sanity-check path: given a result that scores implausibly well relative to a
   provided history/threshold argument, ask the model "does this diff actually
   implement the stated hypothesis, or could this be a bug/leak producing a
   misleadingly good number?" and fold its verdict into the return. Set
   is_semantic_change=True whenever the repair changes model/loss/features/schedule
   rather than just fixing a crash.

5. submission.py — check_submission(path: str, split: str) -> bool. Wrap submit.py's
   --check logic (format/alignment ONLY, never reads labels) as a hard gate.

6. report.py — synthesize_report(run_log: dict) -> str (markdown). Calls a writing
   model to turn a structured run log into the Devpost-style written project
   description.

Since I have no teammate code yet, also write a tiny fake `hypothesis` dict and a
fake `run_log` dict by hand so I can test writer.py and report.py end-to-end right
now without anyone else's module existing.

Give me working, documented Python for all of this. Ready to commit on branch
d-codegen.
```

---

## Integration checklist (once all 4 branches have open PRs)

- [ ] A's `harness/` passes its own fixture tests, PR opened against `main`
- [ ] B's `orchestrator/` runs one full mock iteration end-to-end (stubs only), PR opened
- [ ] C's `llm_calls/` passes its standalone fake-input test harness, PR opened
- [ ] D's `codegen/` passes its standalone fake-input test, PR opened
- [ ] B merges A first (foundation), then D, then C, swapping mocks for real imports
- [ ] Confirm test-label one-shot gate raises correctly on a second call
- [ ] Confirm `pre_execution_gate` blocks a deliberately-planted test-access line
- [ ] Run 3-5 real iterations against the FM baseline node; confirm promotion only
      fires on sealed `valid_confirm`
- [ ] Run `submit.py --check` on a generated submission before designating anything final
- [ ] 5-seed re-check on the final candidate matches the promotion claim
- [ ] Assemble run/iteration logs into the required deliverable format
- [ ] Record demo video, finalize Devpost write-up, submit before **1 Sep 12pm**

---

## What this buys you, and what it doesn't

All 4 people can start immediately and build a fully working, fully tested version
of their own package today, with zero blocking. What this doesn't remove: a short
reconciliation check once real modules replace mocks — if someone's actual code
subtly drifts from the frozen contract (a field renamed, a type changed), that
surfaces on first real integration, not before. Budget a short check-in for that,
not a redesign session.

---


## Appendix — Original Starter Kit README (translated, organizer-provided reference)

# KuaiRand-Pure Starter Kit

## Dependencies

Python 3.9+ and numpy. **Nothing else.** No torch, pandas, or sklearn required.

## Data

Download from https://kuairand.com (direct Zenodo link, no registration needed):

```bash
# Run from inside the Starter Kit directory; extracting gives ./KuaiRand-Pure/
wget https://zenodo.org/records/10439422/files/KuaiRand-Pure.tar.gz
tar xzf KuaiRand-Pure.tar.gz
```

## Run

```bash
python3 baseline.py --model fm
```

`--data_dir` defaults to `./KuaiRand-Pure/data`; pass it explicitly if the data lives
elsewhere.

`--model` accepts `fm` (official baseline) / `pop` (trivial baseline) / `random`
(lower bound, used to sanity-check the evaluation code). The full FM run takes about
40 seconds (CPU, single core).

## Task definition (the conventions below are pinned — do not change them)

| | |
|---|---|
| Task | **Within-user ranking** — each user's own logged impressions in the evaluation set are ranked against each other; this is not full-catalog retrieval |
| Relevance label | `long_view` (native column, 0/1) |
| Metrics | `GAUC`, `nDCG@5`; **primary = mean of both** |
| Data split | train `20220408–20220421` / valid `20220422–20220428` / test `20220429–20220508` |
| Zero-positive users | nDCG is recorded as 0.0 and included in the average; GAUC only counts users with `0 < positive count < impression count`, weighted by positive count |
| nDCG gain | `2^rel − 1` (equivalent to identity under binary labels) |

Implementation is in `evaluate.py`; every convention is documented in that file's
header comments.

## Baseline ladder

Scores on the test set. **FM is the row you need to beat.**

| | GAUC | nDCG@5 | primary |
|---|---|---|---|
| random (lower bound, sanity check) | 0.4996 | 0.4511 | 0.4753 |
| item popularity (trivial) | 0.6308 | 0.5121 | 0.5715 |
| **FM (official baseline)** | **0.6610** | **0.5282** | **0.5946** |

### ⚠️ The real range of the metrics: nDCG@5's ceiling is 0.729, not 1.0

Among the 23,875 users in the test set:

| | Share | Effect on the metric |
|---|---|---|
| All-negative users (none of that user's impressions is `long_view`) | **27.1%** | nDCG is always **0**, no model can fix this; excluded from GAUC |
| All-positive users | **9.2%** | nDCG is always **1**; excluded from GAUC |
| Discriminative users | **63.7%** | The actual sample GAUC is computed over |

So even scoring with the true labels (oracle, perfect ranking) only reaches:

| | random | FM baseline | **oracle ceiling** | Range FM has already captured |
|---|---|---|---|---|
| GAUC | 0.4996 | 0.6610 | **1.0000** | 32.3% |
| nDCG@5 | 0.4511 | 0.5282 | **0.7289** | 27.8% |
| **primary** | 0.4753 | **0.5946** | **0.8645** | **30.7%** |

**Measure progress against the oracle ceiling, not against 1.0.** Seeing 0.5946 and
concluding "still far from a perfect score of 1.0" is a misread — the baseline has
already captured about three-tenths of the available range, so the remaining
headroom is 0.27, not 0.41.

FM's std across 5 random seeds is **0.0008**. Based on that, the convergence rule is
**ε = 0.002 (≈2.5σ), N = 3**: a run is considered converged once the validation
primary score fails to improve by more than 0.002 over 3 consecutive iterations.

> Sanity check: if your evaluation code doesn't get primary ≈ 0.475 (±0.001) when run
> with `--model random`, something is wrong with the harness — fix that first.

## Submission format

CSV with a header row, one line per row of the evaluation set:

```
row_id,user_id,video_id,score
0,0,7531,-3.34176
1,0,4214,-1.4955
...
```

| Field | Description |
|---|---|
| `row_id` | 0-based, strictly increasing, matching the row order of `data.load()[split]` (deterministic: read `log_standard_4_08_to_4_21_pure.csv` first, then `log_standard_4_22_to_5_08_pure.csv`, filter by date, and keep original file order) |
| `user_id` / `video_id` | Redundant fields, used only to verify alignment |
| `score` | Your model's score for that row, any real number, only relative order matters; NaN/Inf are not allowed |

> **Why `row_id` is required:** `(user_id, video_id)` is **not unique** in the
> evaluation set — 3.06% of test-set pairs are repeated, up to 12 times. So it can't
> serve as a primary key.

Generation and validation:

```bash
python3 submit.py --make  --split test  submission.csv    # generate an example submission using the official FM baseline
python3 submit.py --check --split test  submission.csv    # validate format and alignment
python3 submit.py --score --split valid submission.csv    # validate and score (only available locally on valid)
```

`--check` rejects: a wrong header, a row-count mismatch, gaps in `row_id`,
`user_id`/`video_id` misaligned with the evaluation set, and non-numeric or NaN/Inf
`score` values. **Run `--check` yourself before submitting.**

## Where to start improving things

The ordering below is **based on actual measurements**, not guesses. The dead ends
the organizers already tried are marked explicitly — don't spend iterations
re-discovering them.

### Already measured: these two give no benefit — don't waste iterations here

| Tried | Result |
|---|---|
| **Adding static features** — bringing in all 13 of CWM's feature fields (+`music_id`/`video_type`/`upload_type` + 6 coarse user-side buckets) | primary **0.5940** vs. **0.5950** for the 5-field version — no difference beyond noise, if anything slightly worse |
| **Adding model capacity** — embedding dimension k = 8 / 16 / 32 | 0.5895 / 0.5902 / 0.5887 — essentially flat |

Why: the `user_id × video_id` cross term already absorbs most of the learnable
signal. Coarse buckets like `follow_user_num_range` are redundant once `user_id` is
present, and 1.14 million rows can't support much more capacity anyway. **The
bottleneck is not features or capacity.**

⚠️ Also note: **first-order terms on purely user-side features contribute exactly
zero to the score.** Because ranking happens within each user, any term that's
constant within a user doesn't change the within-group order (measured: `item_pop ×
user bias` and plain `item_pop` score identically, to the last digit). User-side
features can only matter through **cross terms with item-side features**.

### Unexplored: the headroom is probably here

Ordered by our best guess at likelihood (**these have not been tested by the
organizers — they're left for you**):

1. **Change the loss function.** Training currently uses pointwise logloss, but the
   metrics (GAUC / nDCG) are **ranking metrics**. Switching to pairwise (BPR) or
   listwise (softmax over a given user's impressions) aligns the training objective
   with the evaluation criteria — this is the direction we think is most likely to
   help.
2. **User history sequences.** The current features make **no use at all** of
   behavioral sequences. Each KuaiRand user has hundreds to thousands of
   interactions in train — DIN/SIM-style interest modeling is a completely
   unexplored direction here.
3. **Multi-task learning.** The logs also contain `is_click`, `is_like`,
   `is_follow`, `is_comment`, `is_forward`, `play_time_ms` — these could serve as
   auxiliary tasks supporting the main `long_view` objective.
4. **Modeling watch duration.** This is exactly [CWM](https://github.com/hyz20/CWM)'s
   contribution: it treats watch duration as a **censored regression** problem
   (once a video finishes playing, the true watch time is truncated by the video's
   length, so a one-sided loss is used instead of squared error). This is a
   direction with real research depth.
5. **Change the model.** DeepFM / DCN / xDeepFM. Since capacity was measured not to
   be the bottleneck, **prioritize this after 1–4**.
6. **Temporal features and distribution shift.** `hourmin`, `date`, and the drift
   between train and test.
7. **Unbiased validation (advanced).** `log_random_4_22_to_5_08_pure.csv` is a
   randomized-exposure log (1.18 million rows) that can serve as an additional
   unbiased validation set, to check whether a model is only overfitting to biased
   traffic.

## Using your own model (including CWM)

`evaluate.py` is completely decoupled from any specific model — it only needs three
equal-length arrays:

```python
from evaluate import evaluate
print(evaluate(user_ids, labels, scores))   # scores can come from any model
```

- `user_ids`: the user_id for each row of the evaluation set
- `labels`: that row's `long_view` value (0/1)
- `scores`: your model's score for that row (any real number, only relative order matters)

So you don't have to use `baseline.py` at all — you can swap in PyTorch, LightGBM, or
[CWM](https://github.com/hyz20/CWM)'s xDeepFM, as long as the final `scores` are
handed to `evaluate()`. **The scoring convention is determined solely by
`evaluate.py`.**

> A note on using CWM: it depends on `torch==1.6.0` (a 2020-era release, which
> probably won't install on newer GPUs), and its loss optimizes counterfactual watch
> time while its evaluation label is a self-reconstructed `long_view2`. It's the
> research code behind a watch-time-debiasing paper — useful as an **advanced
> reference**, but not recommended as a starting point.

## Files

| | |
|---|---|
| `evaluate.py` | Metric implementation + every scoring convention. **Do not modify.** |
| `data.py` | Data loading, the official split, feature encoding. Add features here. |
| `baseline.py` | The three baselines. FM is the one to beat. |
| `baseline_scores.json` | Officially published scores + seed variance + convergence parameters. |
| `submit.py` | Generate / validate submission files. |
| `ablation_features.py` | Feature ablation experiment; reproduces the "adding features gives no benefit" numbers. |
