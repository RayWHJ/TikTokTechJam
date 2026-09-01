# Final Submission & Results Summary — KuaiRand-Pure

**Team:** _[name]_ | **Date:** _[YYYY-MM-DD]_

---

## 1. Model Output

**File:** `submission_kuairand_pure.csv` — hidden-test predictions in Starter Kit schema (`row_id, user_id, video_id, score`).
Generated: `python submit.py --make --model <name> --seed <s> --checkpoint <path>`.
Validated: `python submit.py --check submission_kuairand_pure.csv` — passes.

**Champion:** _[node_id]_ promoted at iter _[N]_. Mechanism: _[one-line hypothesis]_.
Provenance: root → _[chain]_ → champion.

---

## 2. Bonus Benchmarks

**KuaiRand-1k / 27k: not attempted.** Per-iteration cost is dominated by full-seed training runs (~40s baseline on Pure); at 27k this alone exceeds the 6-hour wall-clock ceiling. We scoped these out to focus effort on a defensible KuaiRand-Pure submission.

---

## 3. Results

### valid_search (validation-best)

| Metric | Baseline | Champion | Δ |
|--------|----------|----------|---|
| GAUC | 0.6674 | _[X.XXXX]_ | _[+X.XXXX]_ |
| nDCG@5 | 0.5357 | _[X.XXXX]_ | _[+X.XXXX]_ |
| **primary** | **0.6016** | **_[X.XXXX]_** | **_[+X.XXXX]_** |

### valid_confirm (sealed, paired vs baseline)

| Metric | Baseline | Champion | Paired Δ | Lower 95% |
|--------|----------|----------|----------|-----------|
| primary | _[X.XXXX]_ | _[X.XXXX]_ | _[+X.XXXX]_ | _[±X.XXXX]_ |

### Hidden test (organizer-scored, once)

| Metric | Baseline | Ours | Δ |
|--------|----------|------|---|
| GAUC | 0.6610 | _[fill]_ | _[fill]_ |
| nDCG@5 | 0.5282 | _[fill]_ | _[fill]_ |
| **primary** | **0.5946** | **_[fill]_** | **_[fill]_** |

Judge against oracle ceiling **0.8645**, not 1.0. Our champion at _[X.XXXX]_ captures _[Y]_% of attainable range vs baseline's 31%.

---

## 4. Resource Usage

### LLM tokens (from `.llm_calls_cache/calls.jsonl`)

| Kind | Calls | Input | Output | Total |
|------|------:|------:|-------:|------:|
| diagnose | | | | |
| literature | | | | |
| hypothesis | | | | |
| codegen_diff | | | | |
| codegen_debug | | | | |
| audit | | | | |
| **Total** | | | | |

*Auto-fill from `fill_deliverables.py`.*

### Wall-clock

- Elapsed: _[N.NN h]_ | Iters: _[N of 50]_ | Mean/iter: _[N.N min]_
- Root measurement: ~130s (3 seeds, cached in `root_baseline.json` for future runs)

### GPU-hours: 0

No GPU code path. Baseline (~40s numpy) and champion train comparably on CPU. Per problem statement: *"Wall-clock replaces GPU-hours as the scored compute measure — this benchmark needs no GPU."*

### Feasibility tier (self-estimate): _[low / medium / high]_

---

## 5. Techniques Drawn from Referenced Systems

**AIDE-style tree search** — what we ship. UCB1 selection, per-candidate paired bootstrap, sealed-split promotion gate, memory of refuted mechanisms.

**LightGBM/LambdaRank** — fully implemented (`baseline.py::run_lgb`), available via `CODEGEN_MODEL=lgb`. Measured GBDT variants underperform FM by 0.015–0.020 on hidden test (0.5755–0.5800 vs 0.5953). Two structural causes: sparse user×author pairs make per-pair encoding one observation of noise; the largest LGBM feature gain (`user_rate`) is constant within a user and cannot affect within-user ranking. Kept as evidence for the agent's memory of dead ends.

**MLE-STAR ablation-guided refinement** — fully implemented (`orchestrator/ablation_harness.py`, `codegen/ablations.py`, refiner persona) and tested. Disabled via `REFINE_ENABLED = False` because enabling it regressed the search: our ablation set was too narrow to reliably identify the weakest component of a 5-field FM, and refinements tended toward no-ops. We ship the improve-only search that produced measurably better converged results.

Persona labels `"You are MLE-STAR"` (diagnoser) and `"You are AI-Scientist"` (hypothesis generator) attribute framing influence — single-bottleneck commit, edit-radius classification, confidence/uncertainty scoring — not full technique implementation.

---

## 6. Reproducibility

**Environment:** Python 3.14 (`tiktok` venv). Deps: `numpy`, `lightgbm`, `openai`, `python-dotenv`. LLM: OpenAI Responses API, `gpt-5.6-sol` (writer/reasoning) + `gpt-5.6-luna` (dedup). CPU-only.

**Steps:**
1. Extract KuaiRand-Pure to `kuairand-starter-kit/KuaiRand-Pure/data/`
2. `pip install -r requirements.txt`
3. Set `LLM_CALLS_API_KEY` in `.env`
4. Verify baseline: `python baseline.py --model fm --data_dir ./KuaiRand-Pure/data` → test primary 0.5953
5. Run agent: `python -m orchestrator.driver --max-iters 10`
6. Submit: `python submit.py --make --checkpoint <champion>` then `--check` to validate

**Seeds:** full-seed evaluation uses `(0, 1, 2)`. All numpy RNG seeded from candidate seed. LightGBM configured `deterministic=True, num_threads=1, force_row_wise=True`.
