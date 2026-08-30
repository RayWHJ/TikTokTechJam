# codegen/ — Person D (Execution Engineer)

Code-generation and execution layer for the TikTok TechJam Track 2 autonomous ML
research agent. Wraps the repo-root starter kit (`baseline.py`, `data.py`,
`submit.py`, `evaluate.py`) **without modifying it** — all edits are produced as
diffs and run in an isolated sandbox.

## Frozen contract (implemented exactly)

```python
codegen.write_fix(hypothesis: dict, target_component: str) -> str          # diff text
codegen.pre_execution_gate(code_diff: str) -> {"pass": bool, "reasons": list[str]}
codegen.execute(code_path, seed, split, wallclock_cap_seconds) -> {"status","metrics","logs"}
codegen.debug_and_retry(code_path, error_context) -> {"code_diff","is_semantic_change"}
codegen.check_submission(path: str, split: str) -> bool
codegen.synthesize_report(run_log: dict) -> str                            # markdown
```

`status` ∈ `"ok" | "error" | "timeout" | "diverged"`.

## Runs today with no API key and no teammate code

Every model-calling function (`write_fix`, `debug_and_retry`, `synthesize_report`)
uses `llm_client.LLMClient`, which auto-selects:

- **AnthropicBackend** — only when the `anthropic` SDK is importable **and**
  `ANTHROPIC_API_KEY` is set **and** `CODEGEN_LLM_BACKEND != "fake"`.
- **FakeBackend** — deterministic, offline, the default otherwise. Produces
  gate-clean diffs / a sanity JSON / a Devpost report so the whole package is
  demoable and testable without credentials.

Inject the real client later from the orchestrator with a one-liner — no change in
`writer`/`debug`/`report`:

```python
from codegen.llm_client import LLMClient, AnthropicBackend
client = LLMClient(AnthropicBackend())            # needs ANTHROPIC_API_KEY
diff = codegen.write_fix(hyp, "loss", client=client)
```

Or just set env: `CODEGEN_LLM_BACKEND=anthropic`, `CODEGEN_LLM_MODEL=<id>`,
`ANTHROPIC_API_KEY=<key>`.

## The safety gate (gate.py) — static, deterministic, bias-to-block

Runs before any compute. Scans the **added** lines of a diff and blocks if it:

1. references the **test** split / a test-named file in a data-loading context;
2. uses a **non-causal statistic column** (the `video_features_statistic_pure.csv`
   columns) without an explicit `point_in_time=True` marker nearby;
3. imports **external pretrained weights / datasets** or downloads anything;
4. feeds a **same-row auxiliary signal** (`is_like`, `is_follow`, …) in as an
   input feature array instead of only a loss target (ambiguous usage also blocks).

Word-boundary matching means `follow_user_num_range` (a legit user bucket) is not
confused with the `follow_user_num` statistic, and the feature-matrix tokens are
case-sensitive so `aux_target` is not mistaken for an `X_` matrix.

## The sandbox (sandbox.py) — defence in depth against test leakage

`execute` copies the unchanged modules into a fresh temp working directory
**omitting any test-named file**, runs the candidate as a subprocess with a hard
wall-clock cap (killed on expiry), passes the target split via `CODEGEN_SPLIT`
(never a bare `test` path), and scans stdout/stderr for NaN/inf/divergence. A
candidate that tries to open a relative test-named file fails with
`FileNotFoundError` instead of leaking.

**Candidate contract** (writer-generated candidates honour it): invoked as
`python <candidate> --data_dir <dir> --seed <seed>`, reads `CODEGEN_SPLIT`, and
prints a `##CODEGEN_METRICS## {json}` line (baseline-style `primary 0.59…` output
is also parsed as a fallback).

## debug_and_retry (debug.py)

Repairs a failed candidate (model call with the traceback, up to 2 retries) and
classifies `is_semantic_change` (True when the fix touches model/loss/features/
schedule, not just a crash). Passing `observed_score=` triggers the **leak
sanity-check**: a result above the 0.8645 oracle ceiling — or jumping >0.02 in one
edit — is sent to a skeptical reviewer persona; the verdict is folded in as
`sanity` / `leak_suspected`.

## Run the tests

```bash
pip install pytest
pytest codegen/tests -q          # 25 tests, fully offline
```

## Quick demo

```bash
python -c "import codegen, codegen.fixtures as f; print(codegen.write_fix(f.FAKE_HYPOTHESIS,'loss'))"
python -c "import codegen, codegen.fixtures as f; print(codegen.synthesize_report(f.FAKE_RUN_LOG))"
```

## Notes for integration (Person B)

- Import surface matches the frozen contract names exactly, so swapping the mock
  for `import codegen` is a one-line change.
- All functions take optional `root=` (repo root) and, where relevant, `data_dir=`
  / `client=`. Defaults assume the repo root and `./KuaiRand-Pure/data`.
- `check_submission` maps the harness's `valid_search`/`valid_confirm` tiers onto
  submit.py's `valid` window for alignment (submit.py only knows `valid`/`test`);
  it never reads labels.
- `fixtures.py` holds a hand-written `FAKE_HYPOTHESIS`, `FAKE_HYPOTHESIS_FEATURE`,
  and `FAKE_RUN_LOG` for standalone testing.
