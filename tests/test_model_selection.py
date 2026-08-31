"""T1.7 — the writer and the reasoner are separate knobs, and the run log says
which is which.

The writer had a 60% failure rate in the recorded run: of 5 candidates, 3 never
produced a paired delta, two of them dying with the identical `IndexError` at the
same line of data.py. Its task is a multi-site coupled edit delivered as a whole
reproduced file, which is the shape small models get wrong — so it needs a
stronger model than the reasoner, whose output is a few hundred tokens of
structured JSON.

The trap this module also pins: `.env` is loaded at the top of
`orchestrator/driver.py`, and `CODEGEN_LLM_MODEL` set there silently beats the
code default. So changing the default is necessary but not sufficient, and the
banner has to report which one actually won.
"""
import os

from codegen import llm_client
from orchestrator import driver


#: The deployment configuration, mirrored from `.env`. The code defaults must
#: AGREE with it: a default that contradicts the env var means a run made without
#: `.env` silently uses a different model than one made with it, and the recorded
#: run is the proof — the code default said `gpt-4o-mini` while `.env` said
#: `gpt-4o`, so every conclusion drawn about "the writer model" named the wrong
#: one. `.env` is gitignored, so these constants are the only checked-in record of
#: what a run actually uses.
EXPECTED_DEFAULTS = {
    "CODEGEN_LLM_MODEL": "gpt-5.6-sol",
    "LLM_CALLS_MODEL": "gpt-5.6-sol",
    "LLM_CALLS_CHEAP_MODEL": "gpt-5.6-luna",
}


def test_the_code_defaults_match_the_deployment_env():
    """A default that disagrees with `.env` is a trap, not a fallback."""
    from llm_calls import client as reasoner_client

    assert llm_client.DEFAULT_WRITER_MODEL == \
        EXPECTED_DEFAULTS["CODEGEN_LLM_MODEL"]
    assert reasoner_client.DEFAULT_MODEL == EXPECTED_DEFAULTS["LLM_CALLS_MODEL"]
    assert reasoner_client.DEFAULT_CHEAP_MODEL == \
        EXPECTED_DEFAULTS["LLM_CALLS_CHEAP_MODEL"]


def test_the_writer_default_is_not_a_small_model():
    assert llm_client.DEFAULT_WRITER_MODEL != "gpt-4o-mini"
    assert "mini" not in llm_client.DEFAULT_WRITER_MODEL
    assert "nano" not in llm_client.DEFAULT_WRITER_MODEL


def test_every_default_is_a_model_the_cost_estimator_knows():
    """An unpriced model yields no cost estimate rather than a wrong one, so a
    default the estimator has never heard of silently reports $0.00 on the
    criterion Feasibility & Practicality is scored from."""
    from llm_calls import client as reasoner_client
    from llm_calls.usage import PRICES_USD_PER_MTOK

    for model in (llm_client.DEFAULT_WRITER_MODEL,
                  reasoner_client.DEFAULT_MODEL,
                  reasoner_client.DEFAULT_CHEAP_MODEL):
        assert model in PRICES_USD_PER_MTOK, \
            f"{model} has no price entry, so its spend reports as $0.00"


def test_the_cheap_tier_is_actually_cheaper_than_the_strong_tier():
    """The split only buys anything if the tiers differ in price. Both tiers
    resolving to the same model would make the routing table decorative."""
    from llm_calls import client as reasoner_client
    from llm_calls.usage import PRICES_USD_PER_MTOK

    strong = PRICES_USD_PER_MTOK[reasoner_client.DEFAULT_MODEL]
    cheap = PRICES_USD_PER_MTOK[reasoner_client.DEFAULT_CHEAP_MODEL]
    assert cheap["output"] < strong["output"]
    assert cheap["input"] < strong["input"]


def test_the_writer_and_reasoner_knobs_are_independent():
    """They resolve to the SAME model today, which is a configuration choice —
    not the same knob. Asserted on the env vars rather than on the values,
    because comparing the values would only test today's config."""
    from llm_calls import client as reasoner_client

    src = open(os.path.join(os.path.dirname(os.path.dirname(
        os.path.abspath(__file__))), "llm_calls", "client.py")).read()
    assert "LLM_CALLS_MODEL" in src
    assert "CODEGEN_LLM_MODEL" not in src, \
        "the reasoner must not read the writer's env var"

    # And moving one really does not move the other.
    import os as _os
    from unittest import mock
    with mock.patch.dict(_os.environ, {"CODEGEN_LLM_MODEL": "writer-only"}):
        assert llm_client.resolved_writer_model() == "writer-only"
        assert reasoner_client.DEFAULT_MODEL == \
            EXPECTED_DEFAULTS["LLM_CALLS_MODEL"]


def test_env_overrides_the_default_and_resolution_is_read_at_call_time(
        monkeypatch):
    monkeypatch.delenv("CODEGEN_LLM_MODEL", raising=False)
    assert llm_client.resolved_writer_model() == llm_client.DEFAULT_WRITER_MODEL
    monkeypatch.setenv("CODEGEN_LLM_MODEL", "some-other-model")
    assert llm_client.resolved_writer_model() == "some-other-model", \
        "resolution must happen per call, not be frozen at import"


def test_an_empty_env_var_falls_back_rather_than_selecting_an_empty_model(
        monkeypatch):
    monkeypatch.setenv("CODEGEN_LLM_MODEL", "")
    assert llm_client.resolved_writer_model() == llm_client.DEFAULT_WRITER_MODEL


# --------------------------------------------------------------------------- #
#  The banner                                                                 #
# --------------------------------------------------------------------------- #
def test_the_banner_names_writer_and_reasoner_separately(capsys, monkeypatch):
    monkeypatch.delenv("CODEGEN_LLM_MODEL", raising=False)
    driver._print_model_banner()
    out = capsys.readouterr().out
    assert "writer=" in out and "reasoner=" in out
    assert llm_client.DEFAULT_WRITER_MODEL in out
    from llm_calls import client as reasoner_client
    assert reasoner_client.DEFAULT_MODEL in out
    assert reasoner_client.DEFAULT_CHEAP_MODEL in out


def test_the_banner_says_whether_env_or_the_code_default_won(capsys,
                                                            monkeypatch):
    """The load-bearing half. `.env` sets CODEGEN_LLM_MODEL in this repo, so a
    reader checking whether T1.7 took effect cannot tell from the model name
    alone — the banner has to attribute it."""
    monkeypatch.delenv("CODEGEN_LLM_MODEL", raising=False)
    driver._print_model_banner()
    assert "code default" in capsys.readouterr().out

    monkeypatch.setenv("CODEGEN_LLM_MODEL", "env-chosen-model")
    driver._print_model_banner()
    out = capsys.readouterr().out
    assert "env CODEGEN_LLM_MODEL" in out
    assert "env-chosen-model" in out


def test_the_banner_never_raises(monkeypatch, capsys):
    """A banner must not be the reason a 6-hour unattended run fails to start."""
    import builtins
    real_import = builtins.__import__

    def _boom(name, *a, **k):
        if name in ("codegen.llm_client", "llm_calls"):
            raise ImportError("simulated")
        return real_import(name, *a, **k)

    monkeypatch.setattr(builtins, "__import__", _boom)
    driver._print_model_banner()          # must not raise
    out = capsys.readouterr().out
    assert "unknown" in out


def test_a_verbose_run_prints_the_banner(capsys, monkeypatch, tmp_path):
    """It has to actually reach run.log, not just be callable."""
    import random

    from orchestrator.mocks import codegen as mock_codegen
    from orchestrator.mocks import harness as mock_harness
    from orchestrator.mocks import llm as mock_llm

    monkeypatch.setattr(driver, "harness", mock_harness)
    monkeypatch.setattr(driver, "llm", mock_llm)
    monkeypatch.setattr(driver, "codegen", mock_codegen)
    monkeypatch.setattr(driver, "NODES_LOG_PATH", str(tmp_path / "nodes.jsonl"))
    random.seed(0)
    driver.run(max_iters=1, verbose=True,
               progress_path=str(tmp_path / "progress.json"),
               memory_path=str(tmp_path / "memory.json"),
               champion_dir=str(tmp_path / "champions"),
               root_baseline_path=str(tmp_path / "rb.json"),
               confirm_baseline_path=str(tmp_path / "cb.json"))
    out = capsys.readouterr().out
    assert "[models] writer=" in out
    assert "reasoner=" in out
