# llm_calls

The LLM-calling layer for the autonomous ML research agent (branch `c-llm-calls`).
Self-contained: only talks to a model API and validates JSON, no dependency on
the dataset/tree/sandbox modules.

## Install

```bash
pip install -r requirements.txt
export ANTHROPIC_API_KEY=sk-...   # or LLM_CALLS_API_KEY if you want a separate key
```

## Usage

```python
from llm_calls import diagnose, ground_in_literature, generate_hypothesis, audit, dedup_fingerprint_match
from llm_calls import LLMSchemaError

try:
    diagnosis = diagnose(node_context)
    evidence = ground_in_literature(diagnosis["bottleneck"])
    hypotheses = generate_hypothesis(diagnosis, evidence)
except LLMSchemaError as e:
    # e.attempts, e.last_error, e.last_raw_response are available for logging
    ...
```

## Layout

- `personas.py` — the 5 system prompts as string constants. **This is the
  main file to edit when tuning behavior** — the other files are plumbing
  that shouldn't need to change.
- `client.py` — thin Anthropic SDK wrapper. Swap providers/models here only.
- `schemas.py` — hand-rolled validation for each function's JSON shape (no
  external schema library dependency).
- `retry.py` — shared `call_with_schema_retry`: extracts JSON from a raw
  response (tolerating markdown fences / stray commentary), validates it,
  and on failure retries up to twice with the error appended to the prompt.
  Raises `LLMSchemaError` if all attempts fail.
- `diagnose.py`, `literature.py`, `hypothesis.py`, `audit.py`, `dedup.py` —
  one file per contract function.
- `exceptions.py` — `LLMSchemaError`, the one exception type teammates'
  orchestrator code needs to catch.

## Design notes / things to know

- **`audit()` is blind by construction, not by convention** — its function
  signature only accepts `(diff, checklist)`. There's no parameter for
  hypothesis/rationale, so it's structurally impossible to leak that in,
  even if the caller has it in scope.
- **`ground_in_literature()` caches to disk** at `.llm_calls_cache/literature_cache.json`
  (configurable via `LLM_CALLS_CACHE_DIR`), keyed by a normalized (trimmed,
  lowercased, whitespace-collapsed) bottleneck string. Delete that file to
  force a fresh lookup.
- **`generate_hypothesis()`'s success-criterion check is a heuristic lint**,
  not a guarantee — it checks for comparison language + a named validation
  tier and rejects flat absolute thresholds. If you see false positives/
  negatives in practice, tune `_DELTA_INDICATORS` / `_TIER_INDICATORS` in
  `schemas.py`.
- **`dedup_fingerprint_match()` only escalates to a model call when exactly
  one fingerprint position differs** from an existing memory entry (see
  `_AMBIGUOUS_MAX_DIFF_POSITIONS` in `dedup.py`). Widen that if two-field
  differences should also count as ambiguous in practice.
- Model names default to `claude-sonnet-4-5` (main calls) and
  `claude-haiku-4-5` (cheap dedup escalation) — override via
  `LLM_CALLS_MODEL` / `LLM_CALLS_CHEAP_MODEL` env vars if your team is
  using something else.

## Testing

```bash
pytest tests/test_harness.py -v
```

All 17 tests run offline — they monkeypatch the model-call functions to
return hand-written fake responses (including deliberately malformed ones),
so you can verify schema validation and the retry/escalation logic without
an API key or any teammate code. Once the sandbox/orchestrator exists,
these are a good starting point for higher-level integration tests too.

**Known gotcha if you add more monkeypatch-based tests:** `llm_calls/__init__.py`
imports `diagnose` and `audit` functions whose names collide with their own
submodule names (`diagnose.py` exports a function called `diagnose`, same
for `audit.py`). This means `from llm_calls import diagnose` binds to the
*function*, not the module — monkeypatching that reference is a silent
no-op. Use `importlib.import_module("llm_calls.diagnose")` instead, as the
test harness does, to get the actual module object.
