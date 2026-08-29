# `harness/` — data & evaluation foundation (Person A)

Wraps the organizer's `data.py`, `evaluate.py`, and `baseline.py` **without
modifying them**. Those files define the scoring conventions; this package's only
job is to make them impossible to use wrongly.

```python
import harness

X, y, users = harness.get_split('valid_search')
harness.check_provenance(['user_id', 'video_id', 'tab'])
metrics = harness.validated_evaluate(users, y, my_scores, 'valid_search')
# {'GAUC': ..., 'nDCG@5': ..., 'primary': ..., 'users': ..., 'rows': ...}
```

## The three contract functions

### `validated_evaluate(user_ids, labels, scores, split_name) -> dict`

Calls the real `evaluate.py:evaluate()` unmodified, but only after the input
survives every check. Raises `ValueError` — never a wrong number — on:

- unequal lengths, or empty input
- non-numeric or non-finite (NaN/Inf) scores
- non-binary or NaN labels
- 2-D input
- null/NaN `user_ids`
- an unrecognized `split_name`
- a row count that doesn't match `split_name`'s expected size

Accepts `float32` arrays straight out of `data.py`/`baseline.py`, and plain
`bool` labels. Labels are cast to `int` before the call because `evaluate.py`'s
nDCG gain term is `2**rel`, and `2**np.float32(1)` would quietly return a float.

`tests/test_baseline_repro.py` asserts the wrapper returns **bit-identical**
metrics to calling `evaluate()` directly on a real split — the coercion above
cannot shift a metric in the fourth decimal, which is the decimal the whole
promotion rule (ε = 0.002) turns on.

### `get_split(name) -> (X, y, user_ids)`

| name | dates | days | rows | access |
|---|---|---|---|---|
| `train` | 20220408–20220421 | 14 | 1,141,112 | unrestricted |
| `valid_search` | 20220422–20220426 | 5 | 96,609 | unrestricted — the search tier |
| `valid_confirm` | 20220427–20220428 | 2 | 28,300 | sealed by convention; **every access logged** |
| `test` | 20220429–20220508 | 10 | 170,588 | **one-shot per process**; 2nd call raises `RuntimeError` |

The two `valid_*` tiers exactly partition `data.py`'s official `valid`
(20220422–20220428): contiguous, no gap, no overlap. `data.py`'s `encode()` is
called unmodified, which is safe because it derives both its vocabularies and its
duration-bucket quantile edges from `splits['train']` only — partitioning `valid`
cannot shift the encoding.

### `check_provenance(column_names, point_in_time=False) -> None`

Raises `ValueError` naming every offending column if any name is one of the 50
`NON_CAUSAL_COLUMNS` — every column of `video_features_statistic_pure.csv` except
`video_id`. Those are post-hoc engagement aggregates computed over the full log,
*including rows after the impression being predicted*, so using them as model
inputs is a label leak. `point_in_time=True` is the caller's explicit assertion
that the value was reconstructed as of the prediction timestamp.

## Audit helpers (additive, not part of the frozen contract)

```python
harness.count_sealed_accesses('test')   # -> int, across ALL processes
harness.read_sealed_accesses()          # -> list of entries, oldest first
```

**Read this before trusting the one-shot gate.** The gate is module state, so it
resets in every subprocess — and `codegen.execute()` runs each candidate as a
subprocess by design. A child *can* pull `test` after the parent already has, and
no library can prevent that from the inside. What it can do is make it
undeniable: every `test` and `valid_confirm` access appends one JSON line
(timestamp, pid, calling stack) to `$HARNESS_AUDIT_LOG`, default
`.harness_cache/sealed_access.jsonl`.

So the real end-of-run check is not "did the gate raise?" but:

```python
assert harness.count_sealed_accesses('test') == 1
```

Writes never raise — by the time `record_access` is called the caller already has
the data, so losing a log line beats crashing a run. Failures surface on the
`harness.audit` logger.

## Caveat for downstream consumers

> A harness `valid_search` score is **not** comparable to
> `baseline_scores.json`'s `fm_official.valid` number (0.6016), which was
> measured on the full 7-day `valid` range. `baseline.py:run_fm` also early-stops
> on that full range.

Compare candidate-minus-parent deltas measured *through this harness on the same
tier*, never a harness score against the published `valid` row. This matters most
for `orchestrator/promotion.py`.

## Costs

| operation | cold | warm |
|---|---|---|
| `import harness` | 0.4s | 0.4s |
| `validated_evaluate` row-count check | ~2s | ~1ms |
| `get_split` (any name; loads + encodes all four) | ~34s | free |

Row counts come from a date-column-only scan of the two log CSVs, memoized
in-process and cached to `<data_dir>/.harness_cache/split_sizes.json` keyed on the
logs' size+mtime. `tests/test_splits.py::test_cheap_sizes_match_load_derived_sizes`
is what licenses that shortcut — it asserts the cheap counts equal the
`data.load()`-derived ones.

Set `HARNESS_DATA_DIR` to point elsewhere than `./KuaiRand-Pure/data`.

## Tests

```bash
pytest              # 31 tests, ~2.5s — the gate for every change
pytest -m slow      # 10 split-integrity + 7 baseline-repro tests, ~5 min
```

The fast tier covers metric conventions on hand-built fixtures (tied scores,
all-positive user, all-negative user), every rejection path, and the audit log.

The slow tier is where the real claims live:

- **`test_random_baseline_matches_published`** — the README's first-line check.
  Random scores have no signal, so a deviation from 0.4753 means the *evaluation
  path* is broken, not the model. Nothing else is meaningful until this passes.
- **`test_fm_five_seed_mean_matches_published`** — 5-seed **mean** vs. the
  published 5-seed mean, tolerance `2.5 · 0.0008 / √5`. Deliberately not a
  single-seed comparison: with σ = 0.0008 one seed can sit 2σ off and be
  perfectly healthy, so a per-seed assertion would flake.
- **`test_no_row_appears_in_two_splits`** — the leak check. Rows are compared as
  whole tuples including date, because `(user_id, video_id)` is deliberately
  non-unique here (3.06% of test pairs repeat, up to 12×), so a pair-level
  comparison would report false leaks.

Baseline reproduction reads test labels only to reproduce a published number, and
goes through `baseline.py` on raw `data.load()` output — never through
`get_split('test')`, so a test run never burns the one-shot budget.
