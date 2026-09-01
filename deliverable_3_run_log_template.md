# Run & Iteration Log — KuaiRand-Pure

**Team:** _[name]_ | **Run date:** _[YYYY-MM-DD]_
**Wall-clock:** _[N.N h]_ | **Iterations:** _[N of 50]_ | **Stop reason:** _[plateau / cap / wall-clock]_

---

## Summary Table

| Iter | Parent | Hypothesis (≤1 line) | Component | GAUC / nDCG@5 / primary | Δ paired | p_pos / lower_95 | Status | Notes |
|------|--------|----------------------|-----------|--------------------------|----------|------------------|--------|-------|
| 1 | root | | loss_function | / / | | / | | |
| 2 | | | | / / | | / | | |
| … | | | | | | | | |

*Auto-fill from `fill_deliverables.py`.*

---

## Per-iteration Detail

*One block per iteration.*

### Iteration N

**Hypothesis.** _[hypothesis.mechanism, one sentence]_

**Why.** _[diagnosis.bottleneck + evidence, two sentences]_

**Diff.** `diffs/iter_NN_<id>.diff` — _[1 sentence: what changed]_

**Result.**
- valid_search: GAUC _[X.XXXX]_ / nDCG@5 _[X.XXXX]_ / primary _[X.XXXX]_
- Paired Δ: _[+X.XXXX]_ (p_pos _[X.XXX]_, lower_95 _[±X.XXXX]_)
- Wall-clock: _[NNN]_ s, seeds run: _[3]_

**Confirm (if triggered).** primary _[X.XXXX]_, paired Δ _[+X.XXXX]_ (lower_95 _[±X.XXXX]_) — _[promoted / failed]_

**Errors / recovery.** _[pick one:]_
- No errors.
- _[N]_ of 5 fix attempts used. Error: _[type]_. Recovered by _[how]_.
- Audit flagged _[concern]_ — advisory, false positive on select_on scope.
- No-op on first write; rewritten with feedback → _[ran / still no-op]_.
- Timeout at 420s — _[recovered / closed as timeout]_.

**Status:** _[promoted / open / closed]_

---

## Manual Interventions

**Total during this run: _[N]_**

| # | Iter | Description | Reason |
|---|------|-------------|--------|
| … | | | |

**Between-run development (not counted):** exhausted_families patch, MAX_FIX_ATTEMPTS 2→5, TRIAGE_WALLCLOCK_CAP_S 240→420, --resume support, display fixes.

---

## Evidence Distribution

| Status | Evidence Type | Count |
|--------|---------------|-------|
| … | … | … |

*Auto-fill from `fill_deliverables.py`.*

---

## Techniques Implemented But Disabled by Measurement

**LightGBM/LambdaRank (`baseline.py::run_lgb`).** Implemented, available via `CODEGEN_MODEL=lgb`, not the shipped default. Every measured GBDT variant on KuaiRand-Pure test primary underperforms the FM by 0.015–0.020: LambdaRank 0.5755, small-capacity 0.5795, binary 0.5800, binary + OOF FM score 0.5797, vs FM 0.5953. Two structural causes: user×author pairs occur 1.07× in train on average (per-pair encoding is one observation of noise), and the largest LGBM feature gain is `user_rate` which is constant within a user (cannot change within-user ranking). Kept in the codebase as evidence for the agent's memory of dead ends.

**MLE-STAR ablation-guided refinement.** Fully implemented in `orchestrator/ablation_harness.py`, `codegen/ablations.py`, and `llm_calls/personas.py::REFINER_SYSTEM_PROMPT`. Disabled via `REFINE_ENABLED = False` because enabling it measurably regressed the search: our ablation set (features, regularization, capacity) is too narrow to reliably identify the weakest component of a 5-field FM, and refinements tended to produce no-ops. Persona labels `"MLE-STAR"` (diagnoser) and `"AI-Scientist"` (hypothesis generator) attribute framing influence, not full technique implementation.
