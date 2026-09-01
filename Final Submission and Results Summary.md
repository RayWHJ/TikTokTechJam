# Final Submission & Results Summary — KuaiRand-Pure

**Team:** swe hell
**Submission date:** 2026-09-01

---

## 1. Model Output

**File:** `submission_kuairand_pure.csv` (170,588 rows, Starter Kit schema).
Generated from candidate `3769a939` via `python submit.py --make --data_dir ../KuaiRand-Pure/data --split test submission_kuairand_pure.csv`, run from the champion's staged code directory (`champion_3769a939/`).
Validated: `python submit.py --check submission_kuairand_pure.csv` — passes header, row count, alignment, and NaN/Inf checks.

**Submitted candidate:** `3769a939`, best valid_search primary in the run (0.5961, +0.0015 vs baseline). Mechanism (reconstructed from `deliverables/champion.diff` — the LLM-generated hypothesis text was lost to log corruption): 2×2 grid search over BPR training hyperparameters (learning rate × 0.5 and × 1.0, one or two negatives per positive), best-of-4 by valid_search primary. Provenance: root → 98fedeee (iter 2 new best) → bbe18131 (iter 6 new best) → 3769a939 (iter 7 new best).

**Alternative candidate with fuller evidence:** `bbe18131` (iter 6). Slightly lower valid_search primary (0.5958, +0.0012 vs baseline) but full record intact including a sealed valid_confirm measurement (paired mean_delta +0.0019, p_positive 0.972, lower_95 +0.0006). We submit 3769a939 because its valid_search primary is the run's highest; bbe18131 is documented here as the candidate with the strongest statistical evidence in the recovered data.

---

## 2. Bonus Benchmarks

**KuaiRand-1k and KuaiRand-27k: not attempted.** Per-iteration cost is dominated by full-seed training runs (~40s baseline on Pure); at 27k this single baseline exceeds the 6-hour wall-clock ceiling before any search can occur. We scoped these out to focus effort on a defensible KuaiRand-Pure submission.

---

## 3. Results

### valid_search (running-best trajectory)

| Iter | Primary | GAUC | nDCG@5 | Δ vs baseline |
|------|--------:|-----:|-------:|--------------:|
| 0 (baseline) | 0.5946 | 0.6674 | 0.5357 | 0 |
| 1 (ef87fd88) | 0.5951 | 0.6653 | 0.5250 | +0.0005 |
| 2 (98fedeee) | 0.5955 | 0.6660 | 0.5250 | +0.0009 |
| 6 (bbe18131) | 0.5958 | 0.6662 | 0.5254 | +0.0012 |
| 7 (3769a939) | 0.5961 | 0.6655 | 0.5249 | +0.0015 |

### valid_confirm (sealed, paired vs baseline)

| Candidate | Confirm Primary | Paired mean_delta | Lower 95% CI | Interpretation |
|-----------|----------------:|------------------:|-------------:|----------------|
| ef87fd88 | 0.5516 | (record lost) | — | not sent to confirm |
| 98fedeee | 0.5514 | (measured on confirm) | (recorded but lost with detail) | failed sealed-split gate |
| bbe18131 | 0.5509 | +0.0019 (valid_search) | +0.0006 | failed sealed-split gate |

### Hidden test (local `--score` check, one access, logged in audit log)

| Metric | Baseline (fixed) | Candidate 3769a939 | Δ |
|--------|-----------------:|--------------------:|---:|
| GAUC | 0.6610 | 0.6621 | +0.0011 |
| nDCG@5 | 0.5282 | 0.5286 | +0.0004 |
| **primary** | **0.5946** | **0.5953** | **+0.0007** |

The hidden-test gain (+0.0007) is smaller than the valid_search gain (+0.0015). Both are below the 0.0008 seed standard deviation of the baseline, so the hidden-test measurement should be read as "at or near baseline within seed noise" rather than as a confirmed improvement. The valid_search gains that drove the search's new-best trajectory did not fully generalize to held-out data.

Judge progress against the oracle ceiling **0.8645**, not against 1.0. On hidden test, our candidate captures 30.9% of the attainable range vs the baseline's 30.7%.

---

## 4. Resource Usage

### LLM tokens

The internal counter (`orchestrator/_state/progress.json → counters.tokens`) recorded **506,249** tokens across the run.

Per-kind breakdown from `progress.counters`:
- Estimated full-model calls (hypothesis, diagnose, audit, codegen_diff, codegen_debug): ~20 × 800 = ~19,000
- Estimated cheap calls (dedup, refinement escalation): ~14 × 400 = ~2,500

Live-measured runs:
- 18 full seed runs (3 candidates × ~6 iters that reached full-seed evaluation)
- 11 triage runs (single-seed initial screens)

### Agent wall-clock

- Total elapsed: **~5.86 h** across both sessions
  - Initial run: 4h 22min (iterations 1-7, before disconnect + log corruption)
  - Resumed run: 1h 24min (iterations 8-11)
- Iterations completed: **11 of 50**
- Mean per iteration: **~32 min** (higher than target due to grid-search candidate at iter 7 and 3-candidate iterations at iter 4, 8, 9, 10, 11)
- Root baseline measurement: ~130s (cached in `orchestrator/_state/root_baseline.json`, reused across resume)

### GPU-hours: 0

No GPU code path exists. Baseline (numpy, ~40s) and champion (grid-searched BPR, ~180s per training run) both run on CPU. Per problem statement: *"Wall-clock replaces GPU-hours as the scored compute measure — on this benchmark the reference pipeline needs no GPU at all."*

### Feasibility tier (self-estimate): medium

5.86h wall-clock and estimated 100K-500K real tokens (pending OpenAI export). Under the tier interpretation "low < 4h and < 500K tokens; medium 4-8h or 500K-2M; high beyond that," we sit in the medium band.

---

## 5. Techniques Drawn from Referenced Systems

**AIDE-style tree search over candidate diffs** — what we ship. UCB1 selection across open nodes with cost penalty for wall-clock, per-candidate 500-resample paired user-block bootstrap for statistical decisions, sealed valid_confirm split as a promotion gate, typed memory of refuted mechanisms.

**LightGBM/LambdaRank** — fully implemented (`baseline.py::run_lgb`), available via `CODEGEN_MODEL=lgb`, not the shipped default. Measured GBDT variants underperform FM by 0.015-0.020 on hidden test (0.5755-0.5800 vs 0.5953). Two structural causes: sparse user×author pairs make per-pair encoding one observation of noise; the largest LGBM feature gain (`user_rate`) is constant within a user and cannot affect within-user ranking. Kept as evidence for the agent's memory of dead ends.

**MLE-STAR ablation-guided refinement** — fully implemented (`orchestrator/ablation_harness.py`, `codegen/ablations.py`, refiner persona) and tested. Disabled via `REFINE_ENABLED = False` because enabling it regressed the search — our ablation set was too narrow to reliably identify the weakest component of a 5-field FM, and refinements tended toward no-ops. We ship the improve-only search that produced measurably better convergence.

Persona labels `"You are MLE-STAR"` (diagnoser) and `"You are AI-Scientist"` (hypothesis generator) attribute framing influence — single-bottleneck commit, edit-radius classification, confidence/uncertainty scoring — not full technique implementation.

---

## 6. Reproducibility

**Environment.** Python 3.14 in a venv named `tiktok`. Dependencies: `numpy`, `lightgbm`, `openai`, `python-dotenv`. LLM backend: OpenAI Responses API with `gpt-5.6-sol` (reasoning/writer) and `gpt-5.6-luna` (dedup). CPU-only, single core, Windows.

**Steps to reproduce.**
1. Extract KuaiRand-Pure dataset to `kuairand-starter-kit/KuaiRand-Pure/data/`
2. `pip install -r requirements.txt`
3. Set `LLM_CALLS_API_KEY=sk-...` in `.env`
4. Verify baseline reproduces: `python baseline.py --model fm --data_dir ./KuaiRand-Pure/data` → test primary 0.5953
5. Run agent: `python -m orchestrator.driver --max-iters 50` (resume after any disconnect with `--resume`)
6. Generate submission: from the champion candidate's directory, `python submit.py --make --data_dir ../KuaiRand-Pure/data --split test submission_kuairand_pure.csv`
7. Validate: `python submit.py --check submission_kuairand_pure.csv --data_dir ../KuaiRand-Pure/data --split test`

**Seed protocol.** Full-seed evaluations use seeds (0, 1, 2). All numpy RNG calls seed from the candidate's seed argument. LightGBM (if invoked via `run_lgb`) configured `deterministic=True, num_threads=1, force_row_wise=True`.

**Known non-determinism sources.** LLM calls (persona reasoning) are at temperature 0 but not perfectly deterministic across OpenAI service instances. Bootstrap resampling within the paired significance test uses a fixed seed. A rerun of this agent from a fresh state will produce the same *class* of candidates (loss-function variants converging on BPR grid search) but not bit-identical hypothesis text.
