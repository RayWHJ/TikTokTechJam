"""Tier 2A — the client fixes that make a reasoning-model swap safe and
measurable.

T2.1 is the load-bearing one. Reasoning tokens count against
`max_output_tokens` in the Responses API, and every `llm_calls` call site used the
2000 default (diagnose, hypothesis, audit, verdict) or 3000 (literature) with no
override anywhere. So the moment a reasoning model is configured, four personas
hit the ceiling at once, return an empty `output_text`, and the symptom —
`LLMSchemaError` after three identical retries — reads as "the model is bad"
rather than "the constant is wrong".
"""
import types

import pytest

from llm_calls import client as client_mod
from llm_calls.exceptions import LLMSchemaError, LLMTruncatedError
from llm_calls.retry import call_with_schema_retry


# --------------------------------------------------------------------------- #
#  Stubs shaped like a Responses API response                                 #
# --------------------------------------------------------------------------- #
def _resp(status="completed", output_text="ok", reason=None,
          output_tokens=None, reasoning_tokens=None):
    details = types.SimpleNamespace(reason=reason) if reason else None
    usage = None
    if output_tokens is not None or reasoning_tokens is not None:
        usage = types.SimpleNamespace(
            output_tokens=output_tokens,
            output_tokens_details=types.SimpleNamespace(
                reasoning_tokens=reasoning_tokens))
    return types.SimpleNamespace(status=status, output_text=output_text,
                                 incomplete_details=details, usage=usage)


@pytest.fixture
def captured_call(monkeypatch):
    """Replace the SDK client, and record the kwargs each call was made with."""
    calls = []
    box = {"resp": _resp()}

    def _create(**kw):
        calls.append(kw)
        return box["resp"]

    fake = types.SimpleNamespace(responses=types.SimpleNamespace(create=_create))
    monkeypatch.setattr(client_mod, "get_client", lambda: fake)
    return calls, box


# --------------------------------------------------------------------------- #
#  T2.1 — the ceiling                                                         #
# --------------------------------------------------------------------------- #
def test_the_output_ceiling_is_no_longer_2000_or_3000(captured_call):
    """The two constants that would have bitten four personas at once."""
    calls, _box = captured_call
    assert client_mod.MAX_OUTPUT_TOKENS == 16000

    client_mod.call_model_text("sys", "user")
    client_mod.call_model_with_search("sys", "user")

    assert [c["max_output_tokens"] for c in calls] == [16000, 16000]


def test_every_llm_calls_persona_gets_the_raised_ceiling(captured_call):
    """Not one of the five call sites passed max_tokens, so raising the DEFAULT
    is what actually moves them. A test on the constant alone would not catch a
    call site that hard-codes 2000."""
    import inspect

    from llm_calls import audit, diagnose, hypothesis, literature, verdict
    for mod in (diagnose, hypothesis, audit, verdict, literature):
        src = inspect.getsource(mod)
        assert "max_tokens=2000" not in src and "max_tokens=3000" not in src, \
            f"{mod.__name__} pins a stale output ceiling"


# --------------------------------------------------------------------------- #
#  T2.1 — status checking                                                     #
# --------------------------------------------------------------------------- #
def test_an_incomplete_response_raises_truncated_not_schema_error(captured_call):
    """THE acceptance criterion: status="incomplete", output_text="" must raise
    LLMTruncatedError naming the reason — not LLMSchemaError."""
    calls, box = captured_call
    box["resp"] = _resp(status="incomplete", output_text="",
                        reason="max_output_tokens", output_tokens=16000)

    with pytest.raises(LLMTruncatedError) as e:
        client_mod.call_model_text("sys", "user", model="gpt-5.6-sol")

    assert e.value.reason == "max_output_tokens"
    assert e.value.status == "incomplete"
    assert e.value.model == "gpt-5.6-sol"
    assert e.value.max_tokens == 16000
    msg = str(e.value)
    assert "max_output_tokens" in msg
    assert "gpt-5.6-sol" in msg
    # The message has to say what to DO, or it is just a different mystery.
    assert "reasoning" in msg.lower()
    assert "configuration failure" in msg.lower()


def test_an_empty_output_with_a_completed_status_also_raises(captured_call):
    """The quieter version of the same disease: the API says it finished, but the
    whole budget went to reasoning tokens and nothing was emitted. Retrying
    cannot help, so it must not be reported as a schema problem."""
    calls, box = captured_call
    box["resp"] = _resp(status="completed", output_text="   ",
                        reasoning_tokens=15980)

    with pytest.raises(LLMTruncatedError) as e:
        client_mod.call_model_text("sys", "user")
    assert e.value.reason == "empty_output_text"
    assert "15980" in str(e.value)
    assert "reasoning tokens" in str(e.value)


def test_the_search_call_is_checked_too(captured_call):
    calls, box = captured_call
    box["resp"] = _resp(status="incomplete", output_text="", reason="content_filter")
    with pytest.raises(LLMTruncatedError) as e:
        client_mod.call_model_with_search("sys", "user")
    assert e.value.reason == "content_filter"
    assert "web_search" in str(e.value)


def test_a_good_response_is_returned_unchanged(captured_call):
    calls, box = captured_call
    box["resp"] = _resp(status="completed", output_text='{"a": 1}')
    assert client_mod.call_model_text("sys", "user") == '{"a": 1}'


def test_a_response_object_without_status_still_works(captured_call):
    """Stubbed responses in the existing test suite, and older SDK versions, have
    no `status`. A status check that raises AttributeError on those would be
    worse than no check at all."""
    calls, box = captured_call
    box["resp"] = types.SimpleNamespace(output_text="fine")
    assert client_mod.call_model_text("sys", "user") == "fine"


# --------------------------------------------------------------------------- #
#  T2.1 — the retry loop must not convert it back                             #
# --------------------------------------------------------------------------- #
def test_the_retry_loop_does_not_retry_a_truncation():
    """Three identical calls at the same ceiling cost three times as much and
    tell you nothing new, then raise the wrong exception."""
    n = {"calls": 0}

    def _call(prompt):
        n["calls"] += 1
        raise LLMTruncatedError("truncated", reason="max_output_tokens")

    with pytest.raises(LLMTruncatedError):
        call_with_schema_retry(_call, "prompt", lambda parsed: parsed,
                               max_retries=2)
    assert n["calls"] == 1, \
        f"a truncation must fail fast, made {n['calls']} calls"


def test_the_retry_loop_still_retries_a_real_schema_failure():
    """The distinction has to cut both ways — malformed JSON is exactly what the
    retry loop is for."""
    n = {"calls": 0}

    def _call(prompt):
        n["calls"] += 1
        return "not json at all"

    with pytest.raises(LLMSchemaError):
        call_with_schema_retry(_call, "prompt", lambda parsed: parsed,
                               max_retries=2)
    assert n["calls"] == 3


def test_a_truncation_is_distinguishable_from_a_schema_error_by_type():
    """The orchestrator catches these differently: a schema error is worth
    continuing past, a truncation means every subsequent call will fail the same
    way and the run should say so loudly."""
    assert not issubclass(LLMTruncatedError, LLMSchemaError)
    assert not issubclass(LLMTruncatedError, ValueError), \
        "subclassing ValueError would let call_with_schema_retry swallow it"


def test_the_exception_is_exported():
    import llm_calls
    assert llm_calls.LLMTruncatedError is LLMTruncatedError


# --------------------------------------------------------------------------- #
#  T2.2 — fail loud instead of silently faking                                #
# --------------------------------------------------------------------------- #
@pytest.fixture
def fresh_codegen_client(monkeypatch):
    """Clear codegen's process-wide `_DEFAULT` singleton around each test.

    Load-bearing: `_DEFAULT` is built once per process and `conftest.py` forces
    CODEGEN_LLM_BACKEND=fake at import, so without this every test in the run
    shares whichever backend the first one happened to build.
    """
    from codegen import llm_client
    monkeypatch.setattr(llm_client, "_DEFAULT", None)
    yield llm_client
    llm_client._DEFAULT = None


def test_a_fake_backend_reports_itself_as_fake_not_as_the_env_model(
        fresh_codegen_client, monkeypatch):
    """The trap: `_auto_backend` silently returns FakeBackend when the key is
    missing, so a banner reading the env var announces a frontier model while
    canned single-token edits are served. Reading the BACKEND is the fix."""
    from orchestrator import driver

    monkeypatch.setenv("CODEGEN_LLM_BACKEND", "fake")
    monkeypatch.setenv("CODEGEN_LLM_MODEL", "gpt-5.6-sol")

    rep = driver._model_report()
    assert rep["writer_requested"] == "gpt-5.6-sol"   # what the env asks for
    assert rep["is_fake"] is True                     # what will actually run
    assert rep["writer_effective"] is None
    assert rep["backend"] == "FakeBackend"


def test_the_banner_says_canned_output_when_the_backend_is_fake(
        fresh_codegen_client, monkeypatch, capsys):
    from orchestrator import driver

    monkeypatch.setenv("CODEGEN_LLM_BACKEND", "fake")
    monkeypatch.setenv("CODEGEN_LLM_MODEL", "gpt-5.6-sol")
    driver._print_model_banner()
    out = capsys.readouterr().out
    assert "CANNED OUTPUT" in out, \
        "announcing a frontier model while serving canned diffs is worse than " \
        "printing nothing — it reads like the run is working"
    assert "FakeBackend" in out


def test_a_real_run_on_a_fake_backend_is_refused(fresh_codegen_client,
                                                 monkeypatch):
    """A FakeBackend run produces gate-clean canned edits, scores them, and
    archives a champion. Nothing downstream can tell it from a real search, so
    the refusal has to happen at the start."""
    from orchestrator import driver

    monkeypatch.setenv("CODEGEN_LLM_BACKEND", "fake")
    monkeypatch.setattr(driver, "_using_mocks", lambda: False)

    with pytest.raises(driver.FakeBackendInRealRunError) as e:
        driver._require_real_models()
    msg = str(e.value)
    assert "canned" in msg
    assert "--mock" in msg, "the message must name the honest alternative"


def test_a_mocked_run_is_exempt(fresh_codegen_client, monkeypatch):
    """--mock is an explicit, honest choice and must not be blocked."""
    from orchestrator import driver

    monkeypatch.setenv("CODEGEN_LLM_BACKEND", "fake")
    monkeypatch.setattr(driver, "_using_mocks", lambda: True)
    rep = driver._require_real_models()
    assert rep["using_mocks"] is True


def test_using_mocks_detects_the_module_swap(monkeypatch):
    from orchestrator import driver
    from orchestrator.mocks import codegen as mock_codegen

    monkeypatch.setattr(driver, "codegen", mock_codegen)
    assert driver._using_mocks() is True

    import codegen as real_codegen
    monkeypatch.setattr(driver, "codegen", real_codegen)
    assert driver._using_mocks() is False


def test_an_unbuildable_backend_is_reported_not_swallowed(fresh_codegen_client,
                                                          monkeypatch):
    """Selecting 'openai' with no key raises in OpenAIBackend's constructor.
    --check-models must surface that rather than falling through to fake."""
    from codegen import llm_client
    from orchestrator import driver

    def _boom():
        raise llm_client.LLMError("OPENAI_API_KEY not set")

    monkeypatch.setattr(llm_client.LLMClient, "_auto_backend",
                        staticmethod(_boom))
    monkeypatch.setattr(driver, "_using_mocks", lambda: False)

    rep = driver._model_report()
    assert rep["is_fake"] is None
    assert "UNAVAILABLE" in rep["backend"]
    with pytest.raises(driver.FakeBackendInRealRunError):
        driver._require_real_models(rep)


def test_check_models_exits_zero_with_a_real_backend_and_nonzero_without():
    """End to end through the CLI, which is how it will actually be used."""
    import os
    import subprocess
    import sys

    # conftest.py forces CODEGEN_LLM_BACKEND=fake for the whole test session, and
    # a subprocess inherits it — so the "should pass" case has to clear it and
    # let auto-detection find the key from .env.
    real_env = {k: v for k, v in os.environ.items()
                if k != "CODEGEN_LLM_BACKEND"}
    if not (real_env.get("OPENAI_API_KEY") or os.path.exists(".env")):
        pytest.skip("no API key available to exercise the passing path")

    ok = subprocess.run([sys.executable, "-m", "orchestrator.driver",
                         "--check-models"],
                        capture_output=True, text=True, env=real_env)
    assert ok.returncode == 0, ok.stdout + ok.stderr
    assert "[models] writer=" in ok.stdout
    assert "backend=" in ok.stdout

    env = dict(os.environ, CODEGEN_LLM_BACKEND="fake")
    bad = subprocess.run([sys.executable, "-m", "orchestrator.driver",
                          "--check-models"],
                         capture_output=True, text=True, env=env)
    assert bad.returncode == 1, bad.stdout + bad.stderr
    assert "FAIL" in bad.stdout


def test_check_models_does_not_start_a_search():
    """It has to be free to run. A flag that trains the baseline is a flag
    nobody uses before a 6-hour run."""
    import subprocess
    import sys
    import time

    t0 = time.time()
    r = subprocess.run([sys.executable, "-m", "orchestrator.driver",
                        "--check-models"], capture_output=True, text=True)
    assert time.time() - t0 < 20
    assert "[root] measuring" not in r.stdout
    assert "[iter 1]" not in r.stdout


# --------------------------------------------------------------------------- #
#  T2.3 — the counters have to be real                                        #
# --------------------------------------------------------------------------- #
def _usage_resp(in_tok=1000, cached=800, out_tok=500, reasoning=300):
    return types.SimpleNamespace(
        status="completed", output_text="ok", incomplete_details=None,
        usage=types.SimpleNamespace(
            input_tokens=in_tok,
            input_tokens_details=types.SimpleNamespace(cached_tokens=cached),
            output_tokens=out_tok,
            output_tokens_details=types.SimpleNamespace(
                reasoning_tokens=reasoning)))


@pytest.fixture
def ledger():
    from llm_calls.usage import LEDGER
    LEDGER.reset()
    yield LEDGER
    LEDGER.reset()


def test_usage_is_pulled_off_the_response_and_keyed_by_operator(ledger,
                                                               captured_call):
    """A single scalar cannot show that the writer dominates the bill. The
    breakdown is the actionable half."""
    calls, box = captured_call
    box["resp"] = _usage_resp()

    client_mod.call_model_text("sys", "u", model="gpt-5.6-sol", kind="diagnose")
    client_mod.call_model_text("sys", "u", model="gpt-5.6-luna", kind="verdict")

    t = ledger.totals()
    assert t["calls"] == 2
    assert t["tokens_in"] == 2000
    assert t["tokens_cached"] == 1600
    assert t["tokens_out"] == 1000
    assert t["tokens_reasoning"] == 600
    assert t["tokens_total"] == 3000
    assert set(t["by_kind"]) == {"diagnose", "verdict"}
    assert t["by_kind"]["diagnose"]["models"] == {"gpt-5.6-sol": 1}


def test_every_persona_reports_under_its_own_kind():
    """The seven call sites each had to pass their own `kind`; the default would
    collapse them all into one "text" bucket and lose the whole point of the
    breakdown."""
    import inspect

    from llm_calls import (audit, diagnose, hypothesis, literature,
                           refine, verdict)
    from llm_calls import usage as usage_mod

    expected = {
        diagnose: usage_mod.KIND_DIAGNOSE,
        hypothesis: usage_mod.KIND_HYPOTHESIS,
        audit: usage_mod.KIND_AUDIT,
        verdict: usage_mod.KIND_VERDICT,
        refine: usage_mod.KIND_REFINE,
        literature: usage_mod.KIND_LITERATURE,
    }
    for mod, kind in expected.items():
        src = inspect.getsource(mod)
        const = {v: k for k, v in vars(usage_mod).items()
                 if k.startswith("KIND_")}[kind]
        assert f"kind={const}" in src, (
            f"{mod.__name__} must tag its model call with kind={const}, or its "
            f"tokens land in the untagged default bucket")


def test_the_writer_and_debug_kinds_reach_the_same_ledger():
    """codegen and llm_calls are separate packages with one shared ledger, so a
    run's cost report covers both. The writer is the largest line item."""
    from codegen.llm_client import _USAGE_KINDS, KIND_DEBUG, KIND_DIFF
    from llm_calls.usage import ALL_KINDS
    assert _USAGE_KINDS[KIND_DIFF] == "writer"
    assert _USAGE_KINDS[KIND_DEBUG] == "debug"
    for mapped in _USAGE_KINDS.values():
        assert mapped in ALL_KINDS, f"{mapped} is not a known ledger kind"


def test_a_truncated_call_is_still_billed(ledger, captured_call):
    """It burned the tokens. A ledger that counts only successes understates
    exactly the failure mode that costs the most."""
    calls, box = captured_call
    box["resp"] = types.SimpleNamespace(
        status="incomplete", output_text="", 
        incomplete_details=types.SimpleNamespace(reason="max_output_tokens"),
        usage=types.SimpleNamespace(
            input_tokens=5000,
            input_tokens_details=types.SimpleNamespace(cached_tokens=0),
            output_tokens=16000,
            output_tokens_details=types.SimpleNamespace(reasoning_tokens=16000)))

    with pytest.raises(LLMTruncatedError):
        client_mod.call_model_text("sys", "u", model="gpt-5.6-sol", kind="diagnose")

    t = ledger.totals()
    assert t["tokens_out"] == 16000, "a truncated call still cost 16k output tokens"
    assert t["calls"] == 1


def test_a_response_with_no_usage_block_is_reported_as_an_undercount(
        ledger, captured_call):
    """Silently contributing zero would make the total look complete when it
    is not. A cost figure that cannot say how complete it is cannot be checked."""
    calls, box = captured_call
    box["resp"] = _resp(status="completed", output_text="ok")   # no usage

    client_mod.call_model_text("sys", "u", kind="diagnose")
    t = ledger.totals()
    assert t["calls_without_usage"] == 1
    assert t["tokens_in"] == 0


def test_cost_is_estimated_from_verified_prices(ledger):
    """Cached input is 10x cheaper than fresh on Sol, and this repo re-sends the
    same ~3.5k-token personas every call, so the split matters."""
    from llm_calls.usage import PRICES_USD_PER_MTOK
    # Verified against OpenAI's pricing page on 2026-08-31.
    assert PRICES_USD_PER_MTOK["gpt-5.6-sol"] == {"input": 4.00, "cached": 0.40,
                                                  "output": 20.00}
    assert PRICES_USD_PER_MTOK["gpt-5.6-luna"]["output"] == 1.20

    ledger.record("writer", model="gpt-5.6-sol", tokens_in=1_000_000,
                  tokens_cached=0, tokens_out=1_000_000)
    t = ledger.totals()
    assert t["estimated_cost_usd"] == pytest.approx(24.0)  # $4 in + $20 out


def test_an_unknown_model_yields_no_cost_estimate_rather_than_a_guess(ledger):
    ledger.record("writer", model="some-future-model", tokens_in=1_000_000,
                  tokens_out=1_000_000)
    assert ledger.totals()["estimated_cost_usd"] == 0.0


def test_web_search_is_billed_per_call_not_only_per_token(ledger, captured_call):
    calls, box = captured_call
    box["resp"] = _usage_resp(in_tok=0, cached=0, out_tok=0, reasoning=0)
    client_mod.call_model_with_search("sys", "u", model="gpt-5.6-sol",
                                      kind="literature")
    t = ledger.totals()
    assert t["web_searches"] == 1
    assert t["estimated_cost_usd"] == pytest.approx(0.01)   # $10 / 1k calls


def test_progress_json_carries_real_wallclock_and_tokens(monkeypatch, tmp_path,
                                                         ledger):
    """THE acceptance criterion. `wallclock_s` was 0.0 in every persisted file
    because it was assigned after the loop that writes the file, and `tokens` was
    a fabricated 13200."""
    import json
    import random

    from orchestrator import driver
    from orchestrator.mocks import codegen as mock_codegen
    from orchestrator.mocks import harness as mock_harness
    from orchestrator.mocks import llm as mock_llm

    monkeypatch.setattr(driver, "harness", mock_harness)
    monkeypatch.setattr(driver, "llm", mock_llm)
    monkeypatch.setattr(driver, "codegen", mock_codegen)
    monkeypatch.setattr(driver, "NODES_LOG_PATH", str(tmp_path / "nodes.jsonl"))

    # The mocks make no API calls, so stand in for the usage a real run reports.
    real_diagnose = mock_llm.diagnose

    def _diagnose(ctx):
        ledger.record("diagnose", model="gpt-5.6-sol", tokens_in=4000,
                      tokens_cached=3500, tokens_out=600, tokens_reasoning=400)
        return real_diagnose(ctx)

    monkeypatch.setattr(mock_llm, "diagnose", _diagnose)

    random.seed(0)
    progress = tmp_path / "progress.json"
    result = driver.run(max_iters=2, verbose=False, progress_path=str(progress),
                        memory_path=str(tmp_path / "memory.json"),
                        champion_dir=str(tmp_path / "champions"),
                        root_baseline_path=str(tmp_path / "rb.json"),
                        confirm_baseline_path=str(tmp_path / "cb.json"))

    blob = json.loads(progress.read_text())
    c = blob["counters"]
    assert c["wallclock_s"] > 0.0, \
        "wallclock_s is 0.0 in every pre-T2.3 persisted file"
    # Proportional to the number of diagnose calls the run actually made, not a
    # hard-coded total: the exhaustion re-ask can add a call, and T2.7's wider
    # batch changes how quickly a component exhausts.
    n_diag = c["tokens_by_kind"]["diagnose"]["calls"]
    assert n_diag >= 2
    assert c["tokens_in"] == 4000 * n_diag
    assert c["tokens_out"] == 600 * n_diag
    assert c["tokens"] == c["tokens_in"] + c["tokens_out"], \
        "the legacy `tokens` field must be the real total"
    assert c["tokens_reasoning"] == 400 * n_diag
    assert c["tokens_cached"] == 3500 * n_diag
    assert c["estimated_cost_usd"] > 0
    assert "diagnose" in c["tokens_by_kind"]
    # And the resolved model routing, so the cost figures are interpretable.
    assert blob["models"]["using_mocks"] is True
    # Same numbers on the returned object.
    assert result["counters"].tokens == c["tokens"]


def test_the_ledger_is_reset_between_runs(monkeypatch, tmp_path, ledger):
    """The ledger is a process global; a second run inheriting the first's
    tokens would report a total belonging to neither."""
    import random

    from orchestrator import driver
    from orchestrator.mocks import codegen as mock_codegen
    from orchestrator.mocks import harness as mock_harness
    from orchestrator.mocks import llm as mock_llm

    monkeypatch.setattr(driver, "harness", mock_harness)
    monkeypatch.setattr(driver, "llm", mock_llm)
    monkeypatch.setattr(driver, "codegen", mock_codegen)
    monkeypatch.setattr(driver, "NODES_LOG_PATH", str(tmp_path / "nodes.jsonl"))

    ledger.record("writer", model="gpt-5.6-sol", tokens_in=999_999,
                  tokens_out=999_999)
    random.seed(0)
    result = driver.run(max_iters=1, verbose=False,
                        progress_path=str(tmp_path / "p.json"),
                        memory_path=str(tmp_path / "m.json"),
                        champion_dir=str(tmp_path / "ch"),
                        root_baseline_path=str(tmp_path / "rb.json"),
                        confirm_baseline_path=str(tmp_path / "cb.json"))
    assert result["counters"].tokens == 0, \
        "the previous run's 2M tokens leaked into this one"


# --------------------------------------------------------------------------- #
#  T2.4 — reasoning effort and per-persona routing                            #
# --------------------------------------------------------------------------- #
def test_the_effort_parameter_reaches_the_api(captured_call):
    calls, _box = captured_call
    client_mod.call_model_text("sys", "u", effort="high")
    assert calls[-1]["reasoning"] == {"effort": "high"}

    client_mod.call_model_with_search("sys", "u", effort="low")
    assert calls[-1]["reasoning"] == {"effort": "low"}


def test_no_effort_omits_the_parameter_entirely(captured_call):
    """A non-reasoning model rejects `reasoning` outright, so it has to be
    absent rather than null — the same reasoning that already keeps
    `temperature` out of these calls."""
    calls, _box = captured_call
    client_mod.call_model_text("sys", "u")
    assert "reasoning" not in calls[-1]


def test_the_table_splits_the_expensive_and_cheap_personas():
    """One shared DEFAULT_MODEL forced the auditor and the diagnostician onto the
    same tier. The auditor is the second-largest input consumer in the run and
    flagged 5 of 5 candidates including the training loop's own `y`."""
    from llm_calls import routing
    from llm_calls.usage import (KIND_AUDIT, KIND_DIAGNOSE, KIND_HYPOTHESIS,
                                 KIND_VERDICT, KIND_WRITER)

    assert routing.TABLE[KIND_WRITER][0] == routing.STRONG
    assert routing.TABLE[KIND_DIAGNOSE][0] == routing.STRONG
    assert routing.TABLE[KIND_HYPOTHESIS][0] == routing.STRONG
    assert routing.TABLE[KIND_AUDIT][0] == routing.CHEAP
    assert routing.TABLE[KIND_VERDICT][0] == routing.CHEAP
    # The writer gets the most effort: its task is the coupled multi-site edit
    # that 60% of candidates got wrong.
    assert routing.TABLE[KIND_WRITER][1] == "high"
    assert routing.TABLE[KIND_AUDIT][1] == "low"


def test_every_effort_in_the_table_is_a_value_the_api_accepts():
    """Verified against OpenAI's model docs on 2026-08-31. A rejected parameter
    fails the whole call, so a typo here would break the run at the first
    diagnose."""
    from llm_calls import routing
    assert routing.EFFORTS == ("none", "low", "medium", "high", "xhigh", "max")
    for persona, (_tier, effort) in routing.TABLE.items():
        assert effort in routing.EFFORTS, f"{persona} has effort={effort!r}"


def test_the_writer_keeps_its_own_env_var(monkeypatch):
    """CODEGEN_LLM_MODEL has configured the writer since before the table
    existed and is what the deployment .env sets. Centralising routing must not
    silently move the writer onto LLM_CALLS_MODEL."""
    from llm_calls.routing import model_for
    from llm_calls.usage import KIND_DEBUG, KIND_DIAGNOSE, KIND_WRITER

    monkeypatch.setenv("CODEGEN_LLM_MODEL", "writer-model")
    monkeypatch.setenv("LLM_CALLS_MODEL", "reasoner-model")
    assert model_for(KIND_WRITER) == "writer-model"
    assert model_for(KIND_DEBUG) == "writer-model"
    assert model_for(KIND_DIAGNOSE) == "reasoner-model"


def test_report_and_sanity_do_not_inherit_the_writer_knob(monkeypatch):
    """They run through codegen's client too, but CODEGEN_LLM_MODEL means "the
    model that writes code" — one knob controlling two unrelated things is the
    coupling the table exists to remove."""
    from llm_calls.routing import model_for
    from llm_calls.usage import KIND_REPORT, KIND_SANITY

    monkeypatch.setenv("CODEGEN_LLM_MODEL", "writer-model")
    monkeypatch.setenv("LLM_CALLS_CHEAP_MODEL", "cheap-model")
    assert model_for(KIND_REPORT) == "cheap-model"
    assert model_for(KIND_SANITY) == "cheap-model"


def test_a_single_persona_can_be_moved_without_touching_the_others(monkeypatch):
    from llm_calls.routing import effort_for, model_for
    from llm_calls.usage import KIND_AUDIT, KIND_VERDICT

    monkeypatch.setenv("LLM_ROUTE_AUDIT", "some-other-model")
    monkeypatch.setenv("LLM_EFFORT_AUDIT", "xhigh")
    monkeypatch.setenv("LLM_CALLS_CHEAP_MODEL", "cheap-model")
    assert model_for(KIND_AUDIT) == "some-other-model"
    assert effort_for(KIND_AUDIT) == "xhigh"
    assert model_for(KIND_VERDICT) == "cheap-model"
    assert effort_for(KIND_VERDICT) == "low"


def test_a_typo_in_an_effort_env_var_is_ignored_not_sent(monkeypatch):
    """An unrecognised effort is rejected by the API and fails the whole call. A
    typo in an env var must not take a 6-hour run down at the first diagnose."""
    from llm_calls.routing import effort_for
    from llm_calls.usage import KIND_DIAGNOSE

    monkeypatch.setenv("LLM_EFFORT_DIAGNOSE", "maximum")   # not a real value
    assert effort_for(KIND_DIAGNOSE) is None


def test_routing_resolves_at_call_time_not_import_time(monkeypatch):
    """`DEFAULT_MODEL` being an import-time constant is why a typo in an env var
    used to fail quietly."""
    from llm_calls.routing import model_for
    from llm_calls.usage import KIND_DIAGNOSE

    monkeypatch.setenv("LLM_CALLS_MODEL", "first")
    assert model_for(KIND_DIAGNOSE) == "first"
    monkeypatch.setenv("LLM_CALLS_MODEL", "second")
    assert model_for(KIND_DIAGNOSE) == "second"


def test_the_resolved_table_is_recorded_in_progress_json(monkeypatch, tmp_path):
    """A cost total is meaningless without knowing which model produced it, and
    a reader cannot reconstruct routing that lived only in env vars."""
    import json
    import random

    from orchestrator import driver
    from orchestrator.mocks import codegen as mock_codegen
    from orchestrator.mocks import harness as mock_harness
    from orchestrator.mocks import llm as mock_llm

    monkeypatch.setattr(driver, "harness", mock_harness)
    monkeypatch.setattr(driver, "llm", mock_llm)
    monkeypatch.setattr(driver, "codegen", mock_codegen)
    monkeypatch.setattr(driver, "NODES_LOG_PATH", str(tmp_path / "nodes.jsonl"))
    monkeypatch.setenv("LLM_CALLS_MODEL", "gpt-5.6-sol")
    monkeypatch.setenv("LLM_CALLS_CHEAP_MODEL", "gpt-5.6-luna")

    random.seed(0)
    progress = tmp_path / "progress.json"
    driver.run(max_iters=1, verbose=False, progress_path=str(progress),
               memory_path=str(tmp_path / "m.json"),
               champion_dir=str(tmp_path / "ch"),
               root_baseline_path=str(tmp_path / "rb.json"),
               confirm_baseline_path=str(tmp_path / "cb.json"))

    routing = json.loads(progress.read_text())["models"]["routing"]
    assert routing, "the resolved routing table must be in progress.json"
    assert routing["diagnose"]["model"] == "gpt-5.6-sol"
    assert routing["diagnose"]["effort"] == "medium"
    assert routing["audit"]["model"] == "gpt-5.6-luna"
    assert routing["audit"]["effort"] == "low"
    # And which env var each one answers to, so the run is reproducible.
    assert routing["writer"]["env"] == "CODEGEN_LLM_MODEL"
    assert routing["diagnose"]["env"] == "LLM_CALLS_MODEL"


def test_the_banner_lists_a_model_and_effort_per_persona(capsys, monkeypatch):
    from orchestrator import driver
    monkeypatch.setenv("LLM_CALLS_MODEL", "gpt-5.6-sol")
    monkeypatch.setenv("LLM_CALLS_CHEAP_MODEL", "gpt-5.6-luna")
    driver._print_model_banner()
    out = capsys.readouterr().out
    assert "diagnose" in out and "effort=medium" in out
    assert "audit" in out and "effort=low" in out
    assert "writer" in out and "effort=high" in out


# --------------------------------------------------------------------------- #
#  T2.6 — the hypothesis declares its own family                              #
# --------------------------------------------------------------------------- #
def test_the_declared_family_beats_the_substring_guess():
    """THE bug the declaration removes. Family assignment was substring matching
    in hand-ordered table position, first match wins — so "a pairwise loss over
    user history" landed in `generic_pairwise` rather than `sequence_features`
    purely because the loss families are listed first."""
    from orchestrator import driver

    mech = "a pairwise loss over user history"
    guessed = driver._fingerprint({"mechanism": mech})
    declared = driver._fingerprint({"mechanism": mech,
                                    "mechanism_family": "sequence_features"})
    assert guessed == ("mechanism_family", "generic_pairwise", "", "")
    assert declared == ("mechanism_family", "sequence_features", "", "")


def test_the_bare_token_sequence_no_longer_claims_a_family_by_itself():
    """`sequence_features`' token list includes the bare word "sequence", so it
    swallowed any hypothesis containing it anywhere — including DIN, target
    attention, history pooling and session features, none of which had run when
    the family was banned off one categorical-field experiment."""
    from orchestrator import driver

    h = {"mechanism": "recalibrate scores per tab; the exposure sequence is "
                      "randomised so no correction is needed",
         "mechanism_family": "other"}
    assert driver._fingerprint(h)[0] == "mechanism_hash", \
        "an `other` declaration must never resolve to a blockable family"


def test_other_can_never_be_blocked():
    """The escape hatch has to be real: the taxonomy is a dedup aid, not a menu
    of permitted ideas."""
    from orchestrator import driver

    fp = driver._fingerprint({"mechanism": "a genuinely novel idea",
                              "mechanism_family": "other"})
    assert fp[0] == "mechanism_hash"
    assert "other" not in driver.ALL_FAMILIES


def test_an_unrecognised_family_gets_its_own_fingerprint_not_a_bucket():
    """Bucketing an unknown value into the nearest known family is how a novel
    mechanism inherits a ban it has nothing to do with."""
    from orchestrator import driver
    fp = driver._fingerprint({"mechanism": "x",
                              "mechanism_family": "exposure_debiasing"})
    assert fp == ("mechanism_family", "exposure_debiasing", "", "")


def test_the_schema_requires_a_declared_family():
    from llm_calls.schemas import validate_hypothesis_item

    base = {"mechanism": "swap the loss",
            "success_criterion_paired": "paired delta > +0.001 on the valid_search split",
            "implementation_sketch": "in baseline.py FM.step"}
    with pytest.raises(ValueError, match="mechanism_family"):
        validate_hypothesis_item(base)

    ok = validate_hypothesis_item({**base, "mechanism_family": "bpr_pairwise"})
    assert ok["mechanism_family"] == "bpr_pairwise"


def test_the_schema_rejects_a_blocked_family_so_the_retry_loop_re_asks():
    """Enforcement moves to the layer where a retry loop already exists. The
    driver's dedup filter had none — it silently deleted the hypothesis, which is
    how iteration 4 returned zero candidates from three proposals."""
    from llm_calls.schemas import validate_hypothesis_item

    h = {"mechanism": "swap the loss",
         "success_criterion_paired": "paired delta > +0.001 on the valid_search split",
         "implementation_sketch": "in baseline.py FM.step",
         "mechanism_family": "bpr_pairwise"}
    with pytest.raises(ValueError) as e:
        validate_hypothesis_item(h, blocked_families=["bpr_pairwise"])
    msg = str(e.value)
    assert "REFUTED" in msg
    # The message goes back to the model, so it must name the legal set too.
    assert "Legal:" in msg and "sequence_features" in msg


def test_the_schema_still_forbids_the_four_non_production_keys():
    """Their absence is why the mock's fingerprints were unique forever and the
    family collision was unreachable in every test."""
    from llm_calls.schemas import validate_hypothesis_item

    h = {"mechanism": "x",
         "success_criterion_paired": "paired delta > +0.001 on the valid_search split",
         "implementation_sketch": "y", "mechanism_family": "bpr_pairwise",
         "loss_type": "bpr"}
    with pytest.raises(ValueError, match="loss_type"):
        validate_hypothesis_item(h)


def test_a_batch_must_span_distinct_families():
    """4 of the 5 candidates in the recorded run were the same idea. A batch that
    is one family wide is one experiment run several times."""
    from llm_calls.schemas import validate_hypothesis_list

    def _h(fam):
        return {"mechanism": f"m {fam}",
                "success_criterion_paired": "paired delta > +0.001 on the valid_search split",
                "implementation_sketch": "s", "mechanism_family": fam}

    with pytest.raises(ValueError, match="DISTINCT"):
        validate_hypothesis_list([_h("bpr_pairwise"), _h("bpr_pairwise")],
                                 expected_count=2,
                                 require_distinct_families=True)
    ok = validate_hypothesis_list([_h("bpr_pairwise"), _h("sequence_features")],
                                  expected_count=2,
                                  require_distinct_families=True)
    assert len(ok) == 2
    # `other` may repeat: it is free text, so two `other`s need not be one idea.
    ok2 = validate_hypothesis_list([_h("other"), _h("other")], expected_count=2,
                                   require_distinct_families=True)
    assert len(ok2) == 2


def test_the_legal_families_go_into_the_enforced_schema_enum():
    """Enum enforcement makes the model structurally unable to declare a blocked
    family, rather than told not to and rejected for doing it anyway."""
    from llm_calls.schemas import hypothesis_json_schema

    from llm_calls.schemas import HYPOTHESES_KEY

    schema = hypothesis_json_schema(["sequence_features", "gbdt_swap"])
    # The array is wrapped in an object because structured outputs rejects a root
    # array with 400 invalid_json_schema.
    item = schema["schema"]["properties"][HYPOTHESES_KEY]["items"]
    enum = item["properties"]["mechanism_family"]["enum"]
    assert set(enum) == {"sequence_features", "gbdt_swap", "other"}
    assert "bpr_pairwise" not in enum
    # And the Responses API shape, not the Chat Completions one.
    assert schema["type"] == "json_schema" and schema["strict"] is True
    assert item["additionalProperties"] is False


def test_the_three_personas_send_an_enforced_schema(captured_call):
    import importlib
    diagnose_mod = importlib.import_module("llm_calls.diagnose")
    calls, box = captured_call
    import json as _json
    box["resp"] = _resp(output_text=_json.dumps({
        "bottleneck": "b", "evidence": "e", "confidence": 0.5,
        "component": "loss", "edit_radius": "small", "expected_cost": "low",
        "incompatibilities": [], "uncertainty": 0.5}))
    diagnose_mod.diagnose({"parent": "root"})
    # `text={"format": ...}` — the Responses API spelling.
    assert calls[-1]["text"]["format"]["name"] == "diagnosis"
    assert calls[-1]["text"]["format"]["strict"] is True


def test_the_driver_and_the_schema_share_one_family_list():
    """`llm_calls` must not import `orchestrator.driver` — that would invert the
    dependency and break this package's standalone test harness. So the taxonomy
    lives below both, and the driver re-exports it."""
    from llm_calls.families import ALL_FAMILIES as shared
    from orchestrator import driver
    assert driver.ALL_FAMILIES == shared
    import inspect
    src = inspect.getsource(__import__("llm_calls.schemas",
                                       fromlist=["schemas"]))
    assert "orchestrator" not in src


# --------------------------------------------------------------------------- #
#  T2.7 — propose 6-8, filter deterministically, execute at most 4            #
# --------------------------------------------------------------------------- #
def _h(fam, mech=None, sketch="in baseline.py FM.step, vectorised with np.add.at"):
    return {"mechanism": mech or f"attack {fam} with a within-user objective",
            "success_criterion_paired": "paired delta > +0.001 on the valid_search split",
            "implementation_sketch": sketch,
            "mechanism_family": fam}


def _build_with(hypotheses, tmp_path, blocked=None):
    from orchestrator import driver
    from orchestrator.memory import Memory
    from orchestrator.node import Node

    class _LLM:
        def __init__(self):
            self.contexts = []

        def diagnose(self, ctx):
            self.contexts.append(ctx)
            return {"bottleneck": "b", "evidence": "e", "confidence": 0.75,
                    "component": "loss", "edit_radius": "small",
                    "expected_cost": "low", "incompatibilities": [],
                    "uncertainty": 0.25}

        def ground_in_literature(self, b):
            return {"mechanism": "m", "assumptions": [],
                    "contradictory_findings": [], "dataset_compatibility": [],
                    "implementation_cost": "low", "primary_citation": "c"}

        def generate_hypothesis(self, d, e, tried=None, **kw):
            return [dict(x) for x in hypotheses]

    mem = Memory(path=str(tmp_path / "m.json"))
    for fam in (blocked or []):
        from orchestrator.memory import EvidenceEntry
        for ch in ("a", "b"):
            mem.record(EvidenceEntry(
                fingerprint=("mechanism_family", fam, "", ""), architecture="FM",
                loss="x", sampler="uniform", split="valid_search", seed_count=3,
                confidence_interval=None, code_hash=f"{fam}_{ch}",
                evidence_type="refuted_under_context", note="negative"))
    root = Node(id="root", parent_id=None, code_path="baseline.py")
    counters = driver.Counters()
    diag, cands = driver._build_improve_candidates(
        root, diag_llm=_LLM(), memory=mem, counters=counters, history=[0.59],
        iter_history=[0.59], improvement_score=None, tried=[],
        component_ledger={}, verbose=False)
    return diag, cands, counters


def test_a_batch_of_eight_is_capped_at_four_candidates(tmp_path):
    """The asymmetry: proposing is ~300 output tokens, executing is a writer
    call plus an audit call plus a triage run. Widen the proposals, cap the
    compute."""
    from orchestrator import driver

    fams = ["bpr_pairwise", "sequence_features", "multitask_auxiliary",
            "listwise_softmax", "gbdt_swap", "ensemble_blend",
            "negative_sampling", "watchtime_censored"]
    diag, cands, _c = _build_with([_h(f) for f in fams], tmp_path)

    assert driver.MAX_CANDIDATES_PER_ITER == 4
    assert len(cands) == 4
    # The deferred four are recorded, not silently discarded.
    deferred = [d for d in diag["dropped_by_dedup"]
                if d["reason"] == "over_execution_cap"]
    assert len(deferred) == 4


def test_the_batch_that_reaches_attempt_spans_distinct_families(tmp_path):
    """T2.7's acceptance. 4 of the 5 candidates in the recorded run were the
    same idea (prev_video_id / prev_author_id / prev_long_view /
    session_depth), so the iteration measured one thing four times."""
    diag, cands, _c = _build_with(
        [_h("sequence_features", "the user's previous video_id"),
         _h("sequence_features", "the user's previous author_id"),
         _h("sequence_features", "whether the previous impression was long_view"),
         _h("sequence_features", "the session depth, bucketed"),
         _h("bpr_pairwise"), _h("multitask_auxiliary"), _h("gbdt_swap")],
        tmp_path)

    fams = [c.hypothesis["mechanism_family"] for c in cands]
    assert len(cands) >= 4, f"expected >= 4 candidates, got {len(cands)}"
    assert len(set(fams)) >= 3, f"expected >= 3 distinct families, got {fams}"
    assert fams.count("sequence_features") == 1, \
        "the four restatements of one idea must collapse to one candidate"
    dropped = {d["reason"] for d in diag["dropped_by_dedup"]}
    assert "duplicate_family_in_batch" in dropped


def test_an_unwritable_mechanism_is_dropped_before_a_writer_call(tmp_path):
    """61 of 65 stored proposals named MAML, ColdNAS, DeepFM, contrastive
    objectives or an LLM — a ~94% unimplementable rate WITH the constraint block
    already in the persona. So a prompt request is demonstrably not enough."""
    diag, cands, _c = _build_with([
        _h("other", "use MAML to meta-learn per-user initialisations"),
        _h("other", "add a DeepFM branch", sketch="a torch nn.Module"),
        _h("other", "contrastive pretraining over video embeddings"),
        _h("bpr_pairwise"),
    ], tmp_path)

    assert len(cands) == 1
    assert cands[0].hypothesis["mechanism_family"] == "bpr_pairwise"
    reasons = [d["reason"] for d in diag["dropped_by_dedup"]]
    assert sum(r.startswith("infeasible") for r in reasons) == 3
    assert any("maml" in r for r in reasons)
    assert any("deepfm" in r for r in reasons)
    assert any("contrastive" in r for r in reasons)

    # A library named only in the SKETCH is caught too — that is where the
    # unbuildable dependency usually shows up.
    from orchestrator import driver
    assert driver._infeasible_reason(
        {"mechanism": "a within-user ranking objective",
         "implementation_sketch": "wrap it in a torch.nn.Module"}) == "torch"


def test_a_feasible_numpy_mechanism_is_not_dropped(tmp_path):
    """The gate must not be so broad it rejects the direction the prompts
    actively recommend."""
    diag, cands, _c = _build_with([
        _h("sequence_features",
           "add the user's previous video_id as a categorical field",
           sketch="append to data.EXTRA_FIELDS; one pass with a dict, "
                  "np.searchsorted for the bucket edges"),
    ], tmp_path)
    assert len(cands) == 1
    assert not [d for d in (diag.get("dropped_by_dedup") or [])
                if d["reason"].startswith("infeasible")]


def test_full_runs_per_iteration_is_unchanged_by_the_wider_batch(monkeypatch,
                                                                tmp_path):
    """The whole point: more measurements per iteration, same compute bill.
    `triage.rank(keep=3, wildcard=True)` still decides who gets full seeds."""
    import itertools
    import random

    from orchestrator import driver
    from orchestrator.mocks import codegen as mock_codegen
    from orchestrator.mocks import harness as mock_harness
    from orchestrator.mocks import llm as mock_llm

    monkeypatch.setattr(driver, "harness", mock_harness)
    monkeypatch.setattr(driver, "llm", mock_llm)
    monkeypatch.setattr(driver, "codegen", mock_codegen)
    monkeypatch.setattr(driver, "NODES_LOG_PATH", str(tmp_path / "nodes.jsonl"))
    ids = itertools.count()
    monkeypatch.setattr(driver, "_new_id", lambda: f"n{next(ids):04d}")

    fams = ["bpr_pairwise", "sequence_features", "multitask_auxiliary",
            "listwise_softmax", "gbdt_swap", "ensemble_blend"]

    def _gen(diagnosis, evidence_card, tried=None, **kw):
        blocked = set(kw.get("blocked_families") or ())
        return [_h(f) for f in fams if f not in blocked]

    monkeypatch.setattr(mock_llm, "generate_hypothesis", _gen)

    random.seed(0)
    progress = tmp_path / "progress.json"
    res = driver.run(max_iters=1, verbose=False, progress_path=str(progress),
                     memory_path=str(tmp_path / "m.json"),
                     champion_dir=str(tmp_path / "ch"),
                     root_baseline_path=str(tmp_path / "rb.json"),
                     confirm_baseline_path=str(tmp_path / "cb.json"))

    import json
    rec = json.loads(progress.read_text())["iterations"][0]
    assert rec["n_candidates"] == 4, "the execution cap must hold"
    # 3 root seeds + at most 3 survivors x 2 seeds. rank(keep=3, wildcard=True)
    # is untouched, so this is the same full-run bill as a 1-candidate iteration
    # with 3 survivors.
    assert res["counters"].full_runs <= 3 + 3 * 2
    assert res["counters"].triage_runs == 4


# --------------------------------------------------------------------------- #
#  T2.8 — GAUC and nDCG@5 carried separately, end to end                      #
# --------------------------------------------------------------------------- #
def test_the_two_ceilings_are_not_equal_halves_of_the_oracle():
    """The fact that makes the split worth carrying. The oracle primary is
    0.8645 but a PERFECT ranking gives GAUC 1.0 and nDCG@5 only 0.7289, because
    users with zero positives score 0 and are counted in the average. So the
    headroom is lopsided and a scalar primary averages the asymmetry away."""
    from orchestrator import driver

    assert driver.GAUC_CEILING == 1.0
    assert driver.NDCG5_CEILING == 0.7289
    # 0.3390 on GAUC against 0.2007 on nDCG@5, from the baseline.
    assert driver.GAUC_CEILING - driver.BASELINE_GAUC == pytest.approx(0.3390)
    assert driver.NDCG5_CEILING - driver.BASELINE_NDCG5 == pytest.approx(0.2007)
    # And the oracle primary really is the mean of the two ceilings.
    assert (driver.GAUC_CEILING + driver.NDCG5_CEILING) / 2 == \
        pytest.approx(0.86445, abs=1e-4)


def test_a_gauc_gain_offset_by_an_ndcg_loss_is_now_visible():
    """The failure being fixed: a change that gained on GAUC and lost on nDCG@5
    was indistinguishable from a change that did nothing, because both produce
    the same primary."""
    from orchestrator import driver
    from orchestrator.node import Node

    flat = Node(id="a", parent_id="root", code_path="b.py")
    driver._record_metrics(flat, 0, {"primary": 0.5946, "GAUC": 0.6610,
                                     "nDCG@5": 0.5282, "per_user": {}})
    traded = Node(id="b", parent_id="root", code_path="b.py")
    driver._record_metrics(traded, 0, {"primary": 0.5946, "GAUC": 0.6910,
                                       "nDCG@5": 0.4982, "per_user": {}})

    assert driver._scalar_primary(flat) == driver._scalar_primary(traded), \
        "identical primary — which is exactly why the split is needed"
    a, b = driver._metric_block(flat), driver._metric_block(traded)
    assert a["GAUC"] != b["GAUC"] and a["nDCG@5"] != b["nDCG@5"]
    assert b["gauc_to_ceiling"] < a["gauc_to_ceiling"]
    assert b["ndcg5_to_ceiling"] > a["ndcg5_to_ceiling"]


def test_a_node_that_never_scored_reports_none_not_zero():
    """Zero would read as a measured catastrophe rather than as no measurement."""
    from orchestrator import driver
    from orchestrator.node import Node
    block = driver._metric_block(Node(id="x", parent_id=None, code_path="b.py"))
    assert block["GAUC"] is None and block["nDCG@5"] is None
    assert block["gauc_to_ceiling"] is None


def test_a_pre_t28_root_baseline_cache_degrades_gracefully(tmp_path,
                                                          monkeypatch):
    """A root_baseline.json written before this change has only `primary`.
    Re-measuring to backfill two REPORTING fields would cost three full training
    runs, so a stale cache reports no per-metric split rather than crashing."""
    import json

    from orchestrator import driver
    from orchestrator.node import Node

    cache = tmp_path / "root.json"
    cache.write_text(json.dumps({"split": "valid_search", "seeds": {
        "0": {"primary": 0.5946, "per_user": {"u0": 0.59}},
        "1": {"primary": 0.5950, "per_user": {"u0": 0.60}}}}))

    root = Node(id="root", parent_id=None, code_path="baseline.py")
    ok = driver._measure_root(root, driver.Counters(), cache_path=str(cache),
                              verbose=False)
    assert ok
    assert root.per_seed_primary == {0: 0.5946, 1: 0.5950}
    assert root.per_seed_gauc == {}, "no per-metric data to invent"
    assert driver._metric_block(root)["GAUC"] is None


def test_the_diagnose_context_carries_the_per_metric_headroom(tmp_path):
    """This is the concrete version of "let the model reason about the
    bottleneck": more resolution on what was measured, not more instructions."""
    diag, cands, _c = _build_with([_h("bpr_pairwise")], tmp_path)
    # _build_with's stub records contexts; re-run capturing it.
    from orchestrator import driver
    from orchestrator.memory import Memory
    from orchestrator.node import Node

    seen = []

    class _LLM:
        def diagnose(self, ctx):
            seen.append(ctx)
            return {"bottleneck": "b", "evidence": "e", "confidence": 0.75,
                    "component": "loss", "edit_radius": "small",
                    "expected_cost": "low", "incompatibilities": [],
                    "uncertainty": 0.25}

        def ground_in_literature(self, b):
            return {"mechanism": "m", "assumptions": [],
                    "contradictory_findings": [], "dataset_compatibility": [],
                    "implementation_cost": "low", "primary_citation": "c"}

        def generate_hypothesis(self, d, e, tried=None, **kw):
            return [_h("bpr_pairwise")]

    parent = Node(id="root", parent_id=None, code_path="baseline.py")
    driver._record_metrics(parent, 0, {"primary": 0.5946, "GAUC": 0.6610,
                                       "nDCG@5": 0.5282, "per_user": {}})
    driver._build_improve_candidates(
        parent, diag_llm=_LLM(), memory=Memory(path=str(tmp_path / "m2.json")),
        counters=driver.Counters(), history=[0.59], iter_history=[0.59],
        improvement_score=None, tried=[], component_ledger={}, verbose=False)

    ctx = seen[0]
    assert ctx["metric_ceilings"]["GAUC"] == 1.0
    assert ctx["metric_ceilings"]["nDCG@5"] == 0.7289
    assert ctx["metric_ceilings"]["baseline_gauc_to_ceiling"] == pytest.approx(0.3390)
    assert ctx["parent_metrics"]["GAUC"] == pytest.approx(0.6610)
    assert ctx["parent_metrics"]["ndcg5_to_ceiling"] == pytest.approx(0.2007)


def test_progress_json_and_nodes_jsonl_carry_both_metrics_per_candidate(
        monkeypatch, tmp_path):
    """T2.8's acceptance, and a Section 2.5 deliverable."""
    import itertools
    import json
    import random

    from orchestrator import driver
    from orchestrator.mocks import codegen as mock_codegen
    from orchestrator.mocks import harness as mock_harness
    from orchestrator.mocks import llm as mock_llm

    nodes_log = tmp_path / "nodes.jsonl"
    monkeypatch.setattr(driver, "harness", mock_harness)
    monkeypatch.setattr(driver, "llm", mock_llm)
    monkeypatch.setattr(driver, "codegen", mock_codegen)
    monkeypatch.setattr(driver, "NODES_LOG_PATH", str(nodes_log))
    ids = itertools.count()
    monkeypatch.setattr(driver, "_new_id", lambda: f"n{next(ids):04d}")

    random.seed(0)
    progress = tmp_path / "progress.json"
    driver.run(max_iters=2, verbose=False, progress_path=str(progress),
               memory_path=str(tmp_path / "m.json"),
               champion_dir=str(tmp_path / "ch"),
               root_baseline_path=str(tmp_path / "rb.json"),
               confirm_baseline_path=str(tmp_path / "cb.json"))

    blob = json.loads(progress.read_text())
    assert blob["metric_ceilings"] == {"GAUC": 1.0, "nDCG@5": 0.7289}
    assert blob["baseline_metrics"]["GAUC"] is not None

    scored = [c for r in blob["iterations"] for c in r["candidates"]
              if c["primary"] is not None]
    assert scored, "the mocked run must score something"
    for c in scored:
        assert c["GAUC"] is not None, f"candidate {c['id']} has no GAUC"
        assert c["nDCG@5"] is not None
        assert c["per_seed_gauc"], "per-SEED values, not just the mean"

    lines = [json.loads(l) for l in nodes_log.read_text().splitlines()]
    scored_nodes = [n for n in lines if n["per_seed_primary"]]
    assert scored_nodes
    for n in scored_nodes:
        assert n["per_seed_gauc"] and n["per_seed_ndcg5"]
        assert set(n["per_seed_gauc"]) == set(n["per_seed_primary"])


def test_the_final_summary_reports_both_metrics_and_their_own_ceilings(capsys):
    from orchestrator import driver
    driver.print_final_summary({
        "global_best": 0.5946, "baseline_primary": 0.5946,
        "global_best_node_id": "root", "iters_completed": 3, "history": [0.5946],
        "counters": driver.Counters(),
        "champion_metrics": {"GAUC": 0.6700, "nDCG@5": 0.5300},
        "baseline_metrics": {"GAUC": 0.6610, "nDCG@5": 0.5282},
    })
    out = capsys.readouterr().out
    assert "per-metric:" in out
    assert "GAUC" in out and "nDCG@5" in out
    assert "to ceiling 1.0" in out
    assert "to ceiling 0.7289" in out
    assert "+0.0090" in out, "the per-metric delta must be shown"


# --------------------------------------------------------------------------- #
#  T2.9 — the bottleneck menu is gone, the hard constraints stay              #
# --------------------------------------------------------------------------- #
def test_the_fixed_bottleneck_menu_is_gone():
    """A fixed menu of six bottlenecks anchored the model to the same six every
    iteration — the recorded run's diagnosis was near-identical for four
    iterations running, which is the failure the attempt ledger was built to
    fix."""
    from llm_calls import personas
    p = personas.DIAGNOSTICIAN_SYSTEM_PROMPT
    assert "Bottlenecks worth considering" not in p
    assert "not an exhaustive list" not in p
    # And it says to derive the answer from the evidence instead.
    assert "DERIVE IT FROM THE MEASUREMENTS" in p


def test_the_per_metric_table_replaced_the_menu():
    from llm_calls import personas
    p = personas.DIAGNOSTICIAN_SYSTEM_PROMPT
    assert "0.3390 of headroom" in p and "0.2007 of headroom" in p
    assert "metric_ceilings" in p and "parent_metrics" in p
    # The reason the two halves differ has to be stated, or the numbers look
    # arbitrary.
    assert "zero positives" in p


def test_every_hard_constraint_survives():
    """T2.9 removes suggestions, not limits. 61 stored proposals naming MAML,
    ColdNAS, DeepFM and contrastive objectives are the evidence that these lines
    are load-bearing."""
    from llm_calls import personas
    ctx = personas._DATASET_CONTEXT
    for constraint in (
            "numpy and lightgbm",           # library limit
            "NOTHING ELSE",
            "single-file edit",             # one-file rule
            "VARIABLE-LENGTH",              # no ragged tensors
            "TARGET, never an INPUT",       # aux signals
            "EXTRA_FIELDS",                 # the T1.5 registry
            "name the numpy operations",    # the constructive requirement
            "One CPU core",
            "No external data",
            "OWN generator",                # the RNG rule
    ):
        assert constraint in ctx, f"hard constraint lost: {constraint!r}"


def test_the_ranked_priority_ordering_is_no_longer_asserted():
    """`UNEXPLORED_PRIORITY` survives as the DETERMINISTIC last-resort
    substitution, but must not also be handed to the model as an ordering — that
    is a second anchor doing the same damage as the menu."""
    from llm_calls import personas
    ctx = personas._DATASET_CONTEXT
    assert "in the starter kit's own order of promise" not in ctx
    assert "this paragraph is\nnot an ordering" in ctx or \
        "not an ordering" in ctx
    # The measured dead ends are still stated — those are evidence, not hints.
    assert "MEASURED DEAD ENDS" in ctx
    assert "1.07 times" in ctx


def test_unexplored_priority_is_only_the_deterministic_fallback():
    import inspect

    from orchestrator import driver
    src = inspect.getsource(driver)
    # Exactly two mentions: the definition and the fallback lookup.
    assert src.count("UNEXPLORED_PRIORITY") == 2
    # It must not reach the diagnose context.
    assert "\"unexplored_priority\"" not in src
    assert "'unexplored_priority'" not in src


def test_the_effect_size_constraint_is_kept_as_a_measurement_fact():
    """Not a suggestion: it says which mechanisms are DETECTABLE at this sample
    size, which is a limit on what the search can learn."""
    from llm_calls import personas
    ctx = personas._DATASET_CONTEXT
    assert "0.0012" in ctx and "+0.0006" in ctx
    assert "EASIER TO DETECT" in ctx


# --------------------------------------------------------------------------- #
#  T2.11 — the triage cap scales off the parent, not off the root forever     #
# --------------------------------------------------------------------------- #
def test_the_cap_scales_with_the_parents_own_runtime():
    """A flat 240s measured tolerance against the ROOT forever. The baseline runs
    in ~40s, so a flat 240 already tolerates 6x and a flat 500 tolerates 12x —
    raising the constant buys tolerance for inefficiency, not for genuinely
    expensive mechanisms."""
    from orchestrator import driver
    from orchestrator.node import Node

    p = Node(id="p", parent_id=None, code_path="b.py")
    p.clean_runtime_s = 100.0
    assert driver._triage_cap_for(p) == 400      # 4 x 100, inside the clamp

    p.clean_runtime_s = 40.0                     # the baseline's own cost
    assert driver._triage_cap_for(p) == driver.TRIAGE_CAP_MIN_S   # 160 -> 300


def test_the_cap_is_clamped_at_both_ends():
    from orchestrator import driver
    from orchestrator.node import Node

    p = Node(id="p", parent_id=None, code_path="b.py")
    p.clean_runtime_s = 1.0
    assert driver._triage_cap_for(p) == driver.TRIAGE_CAP_MIN_S == 300
    p.clean_runtime_s = 10_000.0
    assert driver._triage_cap_for(p) == driver.TRIAGE_CAP_MAX_S == 600


def test_the_ceiling_equals_the_full_run_cap():
    """Nothing may clear triage and then time out at higher fidelity — that
    wastes the triage run AND the full-seed run."""
    from orchestrator import driver
    assert driver.TRIAGE_CAP_MAX_S == driver.FULL_RUN_WALLCLOCK_CAP_S == 600


def test_an_unmeasured_parent_gets_the_floor_not_a_crash():
    """Strictly more generous than the old flat 240, and the smoke stage means
    only correct code ever reaches the cap at all."""
    from orchestrator import driver
    from orchestrator.node import Node
    p = Node(id="p", parent_id=None, code_path="b.py")
    assert p.clean_runtime_s is None
    assert driver._triage_cap_for(p) == driver.TRIAGE_CAP_MIN_S


def test_the_flat_constant_is_gone():
    from orchestrator import driver
    assert not hasattr(driver, "TRIAGE_WALLCLOCK_CAP_S"), \
        "a flat cap alongside the adaptive one is two sources of truth"


def test_a_candidate_three_times_slower_than_its_parent_completes(monkeypatch,
                                                                 tmp_path):
    """T2.11's acceptance. Under the old flat cap derived from a 100s parent, a
    3x-slower child at 300s would have been killed at 240s and filed as a
    timeout — i.e. as unmeasurable rather than as merely expensive."""
    import itertools
    import random

    from orchestrator import driver
    from orchestrator.mocks import codegen as mock_codegen
    from orchestrator.mocks import harness as mock_harness
    from orchestrator.mocks import llm as mock_llm

    monkeypatch.setattr(driver, "harness", mock_harness)
    monkeypatch.setattr(driver, "llm", mock_llm)
    monkeypatch.setattr(driver, "codegen", mock_codegen)
    monkeypatch.setattr(driver, "NODES_LOG_PATH", str(tmp_path / "nodes.jsonl"))
    ids = itertools.count()
    monkeypatch.setattr(driver, "_new_id", lambda: f"n{next(ids):04d}")

    caps = []
    real_execute = mock_codegen.execute

    def _execute(code_path, seed, split, wallclock_cap_seconds, root=None,
                 data_dir=None):
        if code_path != "baseline.py":
            caps.append(wallclock_cap_seconds)
        return real_execute(code_path, seed, split, wallclock_cap_seconds,
                            root=root, data_dir=data_dir)

    monkeypatch.setattr(mock_codegen, "execute", _execute)
    # Pin the root's measured cost at 100s, so children get 4 x 100 = 400s.
    real_measure = driver._measure_root

    def _measure(root, counters, **kw):
        ok = real_measure(root, counters, **kw)
        root.clean_runtime_s = 100.0
        return ok

    monkeypatch.setattr(driver, "_measure_root", _measure)

    random.seed(0)
    driver.run(max_iters=1, verbose=False,
               progress_path=str(tmp_path / "p.json"),
               memory_path=str(tmp_path / "m.json"),
               champion_dir=str(tmp_path / "ch"),
               root_baseline_path=str(tmp_path / "rb.json"),
               confirm_baseline_path=str(tmp_path / "cb.json"))

    assert caps, "no candidate reached execute"
    # Every candidate triage run got 4x the parent's 100s, not a flat 240.
    assert set(caps) <= {400, driver.FULL_RUN_WALLCLOCK_CAP_S}, \
        f"unexpected caps: {sorted(set(caps))}"
    assert 400 in caps, "the adaptive triage cap never reached execute"
    # A 3x-slower child costs 300s, which fits inside 400 and would NOT have
    # fitted inside the retired flat 240.
    assert 3 * 100.0 < 400
    assert 3 * 100.0 > 240


def test_the_cap_and_its_basis_appear_in_the_iteration_log(monkeypatch,
                                                          tmp_path):
    """A cap that does not appear in the log cannot be told apart from a timeout
    that was the mechanism's own fault."""
    import itertools
    import json
    import random

    from orchestrator import driver
    from orchestrator.mocks import codegen as mock_codegen
    from orchestrator.mocks import harness as mock_harness
    from orchestrator.mocks import llm as mock_llm

    monkeypatch.setattr(driver, "harness", mock_harness)
    monkeypatch.setattr(driver, "llm", mock_llm)
    monkeypatch.setattr(driver, "codegen", mock_codegen)
    monkeypatch.setattr(driver, "NODES_LOG_PATH", str(tmp_path / "nodes.jsonl"))
    ids = itertools.count()
    monkeypatch.setattr(driver, "_new_id", lambda: f"n{next(ids):04d}")

    random.seed(0)
    progress = tmp_path / "progress.json"
    driver.run(max_iters=2, verbose=False, progress_path=str(progress),
               memory_path=str(tmp_path / "m.json"),
               champion_dir=str(tmp_path / "ch"),
               root_baseline_path=str(tmp_path / "rb.json"),
               confirm_baseline_path=str(tmp_path / "cb.json"))

    for rec in json.loads(progress.read_text())["iterations"]:
        assert rec["triage_cap_s"] is not None
        assert driver.TRIAGE_CAP_MIN_S <= rec["triage_cap_s"] \
            <= driver.TRIAGE_CAP_MAX_S
        assert "parent_clean_runtime_s" in rec


def test_a_clean_triage_run_records_its_runtime_on_the_node(monkeypatch,
                                                            tmp_path):
    """Without this the cap has nothing to scale off after the root, and every
    generation inherits the root's cost."""
    import itertools
    import json
    import random

    from orchestrator import driver
    from orchestrator.mocks import codegen as mock_codegen
    from orchestrator.mocks import harness as mock_harness
    from orchestrator.mocks import llm as mock_llm

    monkeypatch.setattr(driver, "harness", mock_harness)
    monkeypatch.setattr(driver, "llm", mock_llm)
    monkeypatch.setattr(driver, "codegen", mock_codegen)
    monkeypatch.setattr(driver, "NODES_LOG_PATH", str(tmp_path / "nodes.jsonl"))
    ids = itertools.count()
    monkeypatch.setattr(driver, "_new_id", lambda: f"n{next(ids):04d}")

    random.seed(0)
    driver.run(max_iters=2, verbose=False,
               progress_path=str(tmp_path / "p.json"),
               memory_path=str(tmp_path / "m.json"),
               champion_dir=str(tmp_path / "ch"),
               root_baseline_path=str(tmp_path / "rb.json"),
               confirm_baseline_path=str(tmp_path / "cb.json"))

    # Iteration 2's cap must be derived from a MEASURED parent, not the floor
    # fallback, once iteration 1 produced a clean run.
    recs = json.loads((tmp_path / "p.json").read_text())["iterations"]
    measured = [r for r in recs if r["parent_clean_runtime_s"] is not None]
    assert measured, "no iteration recorded a measured parent runtime"


# --------------------------------------------------------------------------- #
#  Enforced-schema structure — guards a whole class of 400                    #
# --------------------------------------------------------------------------- #
# These are the rules OpenAI's structured outputs imposes on a `strict: True`
# schema. Getting one wrong is a 400 at the FIRST call that uses it, which on the
# real driver means a crash several paid API calls into a run:
#
#   openai.BadRequestError: 400 invalid_json_schema — schema must be a JSON
#   Schema of 'type: "object"', got 'type: "array"'
#
# That is what happened to `hypotheses`, which had an array at the root. Asserted
# structurally here because the alternative — finding out from the API — costs a
# diagnose call and a web_search call every time.
def _all_enforced_schemas():
    from llm_calls.schemas import (DIAGNOSIS_JSON_SCHEMA, VERDICT_JSON_SCHEMA,
                                   hypothesis_json_schema)
    return {
        "diagnosis": DIAGNOSIS_JSON_SCHEMA,
        "verdict": VERDICT_JSON_SCHEMA,
        "hypotheses": hypothesis_json_schema(),
        "hypotheses_narrowed": hypothesis_json_schema(["bpr_pairwise"]),
    }


def _walk_objects(node, path="schema"):
    """Yield (path, subschema) for every object-typed subschema."""
    if isinstance(node, dict):
        if node.get("type") == "object":
            yield path, node
        for key in ("items", "additionalItems"):
            if key in node:
                yield from _walk_objects(node[key], f"{path}.{key}")
        for name, sub in (node.get("properties") or {}).items():
            yield from _walk_objects(sub, f"{path}.{name}")


@pytest.mark.parametrize("name", sorted(_all_enforced_schemas()))
def test_every_enforced_schema_has_an_object_at_the_root(name):
    """THE bug. Structured outputs rejects a root array outright."""
    s = _all_enforced_schemas()[name]
    assert s["schema"]["type"] == "object", (
        f"{name} has a {s['schema']['type']!r} at the schema root; structured "
        f"outputs requires 'object' and returns 400 invalid_json_schema "
        f"otherwise")


@pytest.mark.parametrize("name", sorted(_all_enforced_schemas()))
def test_every_enforced_schema_is_wellformed_for_strict_mode(name):
    """`strict: True` requires every object to forbid additional properties and
    to list every one of its properties in `required`."""
    s = _all_enforced_schemas()[name]
    assert s["type"] == "json_schema"
    assert s["strict"] is True
    assert s["name"], "the format needs a name"

    for path, obj in _walk_objects(s["schema"]):
        props = set(obj.get("properties") or {})
        assert obj.get("additionalProperties") is False, \
            f"{name} at {path}: strict mode needs additionalProperties: false"
        assert set(obj.get("required") or []) == props, \
            (f"{name} at {path}: strict mode requires every property in "
             f"`required`; missing {sorted(props - set(obj.get('required') or []))}")


def test_the_hypothesis_batch_is_wrapped_not_a_bare_array():
    from llm_calls.schemas import HYPOTHESES_KEY, hypothesis_json_schema

    s = hypothesis_json_schema(["bpr_pairwise", "gbdt_swap"])["schema"]
    assert list(s["properties"]) == [HYPOTHESES_KEY]
    arr = s["properties"][HYPOTHESES_KEY]
    assert arr["type"] == "array"
    # The enum still narrows to the legal set — the wrapper must not lose it.
    assert set(arr["items"]["properties"]["mechanism_family"]["enum"]) == \
        {"bpr_pairwise", "gbdt_swap", "other"}


def test_the_validator_accepts_both_the_wrapped_and_bare_shapes():
    """The enforced path returns {"hypotheses": [...]}; a stub or a model without
    schema support returns a bare array. Both must validate, or the fix trades
    one crash for another."""
    from llm_calls.schemas import validate_hypothesis_list

    h = {"mechanism": "swap the loss",
         "success_criterion_paired": "paired delta > +0.001 on the valid_search split",
         "implementation_sketch": "in baseline.py FM.step",
         "mechanism_family": "bpr_pairwise"}

    wrapped = validate_hypothesis_list({"hypotheses": [h]}, expected_count=1)
    bare = validate_hypothesis_list([h], expected_count=1)
    assert wrapped == bare == [h]


def test_the_validator_still_rejects_a_wrong_count_inside_the_wrapper():
    """Unwrapping must not skip the count check — that check is what makes the
    retry loop able to ask for the batch size again."""
    from llm_calls.schemas import validate_hypothesis_list

    h = {"mechanism": "m",
         "success_criterion_paired": "paired delta > +0.001 on the valid_search split",
         "implementation_sketch": "s", "mechanism_family": "bpr_pairwise"}
    with pytest.raises(ValueError, match="exactly 3"):
        validate_hypothesis_list({"hypotheses": [h]}, expected_count=3)


def test_an_object_without_the_hypotheses_key_is_a_clear_error():
    from llm_calls.schemas import validate_hypothesis_list
    with pytest.raises(ValueError, match="hypotheses"):
        validate_hypothesis_list({"items": []}, expected_count=1)


def test_generate_hypothesis_round_trips_the_wrapped_shape(captured_call):
    """End to end through the real call path: schema out, wrapped JSON back,
    validated list returned."""
    import json as _json

    from llm_calls import hypothesis as hyp

    calls, box = captured_call
    n = hyp.HYPOTHESES_MIN
    fams = ["bpr_pairwise", "sequence_features", "multitask_auxiliary",
            "listwise_softmax", "gbdt_swap", "ensemble_blend"][:n]
    payload = {"hypotheses": [
        {"mechanism": f"attack {f}",
         "success_criterion_paired": "paired delta > +0.001 on the valid_search split",
         "implementation_sketch": "np.add.at over the index arrays",
         "mechanism_family": f} for f in fams]}
    box["resp"] = _resp(output_text=_json.dumps(payload))

    out = hyp.generate_hypothesis({"confidence": 0.9}, {"mechanism": "m"})
    assert len(out) == n
    assert [h["mechanism_family"] for h in out] == fams
    # And the request carried an object-rooted schema.
    assert calls[-1]["text"]["format"]["schema"]["type"] == "object"
