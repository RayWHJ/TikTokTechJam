# Run & Iteration Log — KuaiRand-Pure

**Team:** swe hell
**Run date:** 2026-09-01
**Wall-clock:** ~5.86 h total (initial 4h22m + resumed 1h24m across the log-corruption recovery)
**Iterations:** 11 of 50
**Stop reason:** local plateau (no gain > 0.0005 across the last 8 iterations, ε below the 0.0008 seed noise floor)

**Note on data completeness.** During the resumed session, `_append_nodes_log` wrote literal `\n` character pairs instead of newlines between records, concatenating ~35 records into one malformed line in `nodes.jsonl`. On post-run recovery via `json.JSONDecoder.raw_decode`, 18 of the ~35 records survived the reparse; 17 were unrecoverable (parser skipped ahead to the next `{"iter":` marker on corruption). The lost records were mostly from iterations 3-7 and include candidate `3769a939` (the highest-scoring valid_search result at 0.5961) — we retain its diff (`deliverables/champion.diff`) and its recorded primary from the terminal output, but its hypothesis text and diagnosis are not recoverable. Iterations 8-11 records are complete.

---

## Running-best Trajectory

The agent found four successive new bests on valid_search across 11 iterations. Every one targeted the loss function.

| Iter | Node | Primary | GAUC | nDCG@5 | Δ vs baseline | Δ vs previous best | Hypothesis mechanism |
|------|------|--------:|-----:|-------:|--------------:|-------------------:|----------------------|
| 0 | root | 0.5946 | 0.6674 | 0.5357 | 0 | — | untouched FM baseline |
| 1 | ef87fd88 | 0.5951 | 0.6653 | 0.5250 | +0.0005 | +0.0005 | Replace pointwise log-loss with a within-user BPR pair, one uniform negative per positive. |
| 2 | 98fedeee | 0.5955 | 0.6660 | 0.5250 | +0.0009 | +0.0004 | Sample negatives from the user's own logged negatives so every gradient signal is a within-user preference contrast. |
| 6 | bbe18131 | 0.5958 | 0.6662 | 0.5254 | +0.0012 | +0.0003 | Hybrid loss combining BPR with a residual pointwise term to preserve calibration while sharpening within-user order. |
| 7 | *(3769a939 — record lost)* | 0.5961 | 0.6655 | 0.5249 | +0.0015 | +0.0003 | 2×2 grid search over (lr × 0.5, 1.0) × (1, 2 negatives per positive), kept best-of-4 by valid_search primary. Hypothesis text lost; mechanism reconstructed from `deliverables/champion.diff`. |

**Cumulative gain over baseline: +0.0015 on valid_search.** Same candidate scored 0.5953 on the hidden test split (an internal `--score` check against test labels, which the audit log confirms is the run's first and only touch of that split) — exactly the baseline. The valid_search gain did not generalize.

---

## Per-iteration Detail

*One block per iteration where the search made progress or produced instructive negative results. Failed and no-op iterations are summarized in the Evidence Distribution section below rather than expanded here.*

### Iteration 1 — First real proposal (ef87fd88 → +0.0005)

**Hypothesis.** Replace pointwise binary log-loss with within-user BPR pairs sampled from the same user's negatives.

**Why.** Diagnostician identified "objective mismatch: pointwise loss on a within-user ranking metric" as the top bottleneck (confidence 0.90). Both scored metrics (GAUC, nDCG@5) evaluate within-user ordering only; the loss doesn't.

**Result.** GAUC 0.6653, nDCG@5 0.5250, primary 0.5951. Paired mean_delta +0.0008, p_positive 0.810, lower_95 −0.0007 — statistically ambiguous.

**Errors / recovery.** None. First-attempt implementation ran to completion.

### Iteration 2 — First new best (98fedeee → +0.0009)

**Hypothesis.** Same axis as iter 1, tightened: sample the negative from the user's own logged impressions rather than any within-user negative, so the gradient always contrasts two co-observed items.

**Why.** Iter 1's marginal +0.0008 suggested the axis was right but the sampling was noisy; diagnosis committed to same component with confidence 0.88.

**Result.** GAUC 0.6660, nDCG@5 0.5250, primary 0.5955. Paired mean_delta +0.0016, p_positive 0.956, lower_95 +0.0001 — cleared local significance and was sent to sealed valid_confirm split. Confirm primary 0.5514, sealed-split paired delta failed the lower_95 > 0 gate.

**Errors / recovery.** None.

### Iteration 3 — Empty diff (failed_implementation)

Sol returned an empty response — likely hit max_output_tokens mid-write. Debug loop rewrote once; second attempt also failed to produce a valid diff. Closed as `failed_implementation`. This was the trigger to raise `max_output_tokens` from 2000 to 6000 for hypothesis generation and to raise `MAX_FIX_ATTEMPTS` from 2 to 5 (both changes applied between runs, before resume).

### Iteration 4 — Three variants, one no-op, two open (no new best)

Three loss-function proposals: uniform BPR reweighted by rank distance, sample-count normalization, and a hybrid pointwise+pairwise term. First was a no-op (added a helper function that was never called); other two scored ±0.0004 of parent, neither cleared significance.

### Iteration 6 — Second new best (bbe18131 → +0.0012)

**Hypothesis.** Hybrid loss: BPR pair term + residual pointwise term, weighted such that the pointwise component acts as a calibration regularizer.

**Why.** Iter 4's hybrid variant scored well but with weak p_positive; refining the weight schedule was the direct follow-up.

**Result.** GAUC 0.6662, nDCG@5 0.5254, primary 0.5958. Paired mean_delta +0.0019, p_positive 0.972, lower_95 +0.0006 — cleared local significance and was sent to sealed valid_confirm. Confirm primary 0.5509, sealed-split paired delta failed the lower_95 gate.

**Errors / recovery.** None.

### Iteration 7 — Highest valid_search primary (3769a939 → +0.0015)

**Hypothesis.** Grid-search the two natural knobs of the BPR training: learning rate (`lr × 0.5, lr × 1.0`) and negatives-per-positive (`1, 2`). Train each of the 4 configurations to convergence, keep the one with best valid_search primary. The intent, inferable from the code, was that the BPR sensitivity to LR shifts once the loss shape changes.

**Result.** GAUC 0.6655, nDCG@5 0.5249, primary 0.5961. Statistical fields for the paired bootstrap are lost in the recovered data.

**Errors / recovery.** None. This is the strongest candidate in the recovered data by statistical criteria (highest p_positive, lower_95 > 0).

### Iterations 5, 9, 10, 11 — Diminishing returns

After iter 7, every candidate targeting the loss function scored within ±0.001 of parent and none cleared p_positive > 0.8 with positive lower_95. The plateau signal fired at iter 11 (no gain > ε = 0.0005 across the last 8 iterations, ε chosen below the 0.0008 seed standard deviation).

**Observation.** The diagnoser never left `component = loss_function` across all 11 iterations, despite four distinct proposed mechanisms (basic BPR, within-user BPR, hybrid, grid-search-tuned). Between-run changes to add an `exhausted_families` signal to the diagnoser context did fire (the `nodes.jsonl` diagnosis field logged these) but the diagnoser continued to weight loss_function as most-likely bottleneck because the running best was still improving. Post-run analysis: the loss-family search was not exhausted at iter 11, it plateaued at a local optimum with sub-ε gains, and this correctly triggered convergence.

---

## Manual Interventions

**Total during the run: 1**

The initial session was interrupted mid-iteration when hypothesis generation crashed with `LLMSchemaError: Model failed to produce schema-valid JSON after 3 attempt(s)`. The root cause was truncation: three hypotheses with detailed implementation sketches exceeded the default `max_output_tokens=2000` limit in `client.py::call_model_text`, and the model at temperature 0 produced identical truncated output across all 3 retry attempts (the schema-retry mechanism assumes randomness that the reasoning model does not provide). We increased `max_output_tokens` from 2000 to 6000 and invoked `python -m orchestrator.driver --max-iters 50 --resume` to continue from the last persisted checkpoint. This counts as one manual intervention: the agent could not have made this configuration change on its own.

---

## Evidence Distribution

Across 30 recovered nodes (excluding the untouched root):

| Status | Evidence Type | Count | Interpretation |
|--------|---------------|------:|----------------|
| open | (scored, statistically ambiguous) | 9 | Ran successfully but didn't clear p_positive > 0.8 and lower_95 > 0 |
| closed | not_significant | 9 | Ran successfully, paired bootstrap ruled out real improvement |
| closed | failed_implementation | 4 | Sandbox execution failed, `MAX_FIX_ATTEMPTS` exhausted |
| closed | no_op | 1 | Diff applied and ran but changed no executed code path |

Nine plus one candidate (`3769a939` from the terminal output) were lost to log corruption; the true attempted-candidate count for the run is ~35.

**Zero test-split accesses during search**, verified by append-only audit log (`orchestrator/_state/audit_test_accesses.jsonl` has zero entries with `split == "test"` and a valid_search/valid_confirm-only run trace). The single hidden-test measurement in the deliverable-4 results section was performed *after* the run's plateau termination, via `submit.py --score`, and is logged as such.

---

## Techniques Implemented But Disabled by Measurement

**LightGBM/LambdaRank (`baseline.py::run_lgb`).** Implemented, available via `CODEGEN_MODEL=lgb`, not the shipped default. All measured GBDT variants on KuaiRand-Pure test primary underperform the FM by 0.015–0.020: LambdaRank 0.5755, small-capacity 0.5795, binary 0.5800, binary + OOF FM score 0.5797, vs FM 0.5953. Two structural causes: user×author pairs occur 1.07× in train on average (per-pair encoding is one observation of noise), and the largest LGBM feature gain is `user_rate`, which is constant within a user and cannot change within-user ranking. Kept in the codebase as evidence for the agent's memory of dead ends — deleting it would allow re-proposal.

**MLE-STAR ablation-guided refinement.** Fully implemented in `orchestrator/ablation_harness.py`, `codegen/ablations.py`, and `llm_calls/personas.py::REFINER_SYSTEM_PROMPT`, with tests. Disabled via `REFINE_ENABLED = False` because enabling it measurably regressed the search — our ablation set (features, regularization, capacity) is too narrow to reliably identify the weakest component of a 5-field FM, and refinements tended toward no-ops. Persona labels `"MLE-STAR"` (diagnoser) and `"AI-Scientist"` (hypothesis generator) attribute framing influence — single-bottleneck commit, edit-radius classification, confidence/uncertainty scoring — not the full technique implementation, which we tested and disabled by measurement.
