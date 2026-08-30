"""driver.print_final_summary — the end-of-run block.

The load-bearing assertion here is the baseline flag. global_best only advances
on a promotion confirmed against the sealed valid_confirm split, so a run that
found nothing reports a "best" numerically identical to the baseline — and read
quickly, 0.5946 looks like a result rather than the absence of one. Every run so
far has been that case, which is exactly why it needs a test.
"""
from orchestrator import driver
from orchestrator.counters import Counters

BASE = 0.5946398178736368


def _result(**over):
    r = {"global_best": BASE, "global_best_node_id": "root0000",
         "baseline_primary": BASE, "best_valid_search": None,
         "best_valid_search_node_id": None, "iters_completed": 4,
         "history": [BASE], "counters": Counters()}
    r.update(over)
    return r


def test_flags_that_the_best_is_still_the_baseline(capsys):
    driver.print_final_summary(_result())
    out = capsys.readouterr().out
    assert "STILL THE BASELINE" in out
    assert "nothing was promoted" in out
    # The baseline is reported on its own line regardless.
    assert f"baseline primary: {BASE:.6f}" in out


def test_does_not_flag_a_genuine_improvement(capsys):
    driver.print_final_summary(_result(
        global_best=0.6010, global_best_node_id="abc12345",
        history=[BASE, 0.598, 0.6010]))
    out = capsys.readouterr().out
    assert "STILL THE BASELINE" not in out
    assert "+0.006360 vs baseline" in out
    assert "node abc12345" in out
    assert "promotions: 2" in out


def test_flag_survives_float_noise_at_the_last_bit(capsys):
    """global_best is assigned from a mean over seeds, so an unpromoted run's
    best can differ from the baseline in the last bit without meaning anything.
    The flag compares at NOOP_EPSILON, not with ==."""
    driver.print_final_summary(_result(global_best=BASE + 1e-15))
    assert "STILL THE BASELINE" in capsys.readouterr().out


def test_reports_an_unconfirmed_valid_search_win(capsys):
    """The gap between best_valid_search and global_best IS the finding: a
    candidate beat the baseline on valid_search and failed the sealed split."""
    driver.print_final_summary(_result(
        best_valid_search=0.5966485738754272,
        best_valid_search_node_id="2adbbb82"))
    out = capsys.readouterr().out
    assert "UNCONFIRMED" in out
    assert "2adbbb82" in out
    assert "did not survive promotion" in out


def test_stays_quiet_when_there_is_no_unconfirmed_win(capsys):
    """No spurious line when the best seen IS the promoted best."""
    driver.print_final_summary(_result(
        global_best=0.6010, best_valid_search=0.6010,
        best_valid_search_node_id="abc12345", history=[BASE, 0.6010]))
    assert "UNCONFIRMED" not in capsys.readouterr().out


def test_handles_an_unmeasured_baseline_without_a_stray_delta(capsys):
    """_measure_root can fail; then there is no baseline to subtract."""
    driver.print_final_summary(_result(baseline_primary=None))
    out = capsys.readouterr().out
    assert "baseline UNMEASURED" in out
    assert "vs baseline" not in out
    assert "(, node" not in out, "empty delta left a stray comma"


def test_counters_are_printed_field_per_line(capsys):
    c = Counters()
    c.bump("proposals", 11)
    c.bump_scorer("valid_search", 28)
    driver.print_final_summary(_result(counters=c))
    out = capsys.readouterr().out
    assert "counters:" in out
    assert "proposals          11" in out
    assert "valid_search=28" in out


def test_run_returns_the_fields_the_summary_needs(monkeypatch, tmp_path, capsys):
    """print_final_summary reads keys off run()'s return value, so a rename in
    one and not the other would silently print '?' and nothing would fail."""
    from orchestrator.mocks import codegen as mock_codegen
    from orchestrator.mocks import harness as mock_harness
    from orchestrator.mocks import llm as mock_llm

    monkeypatch.setattr(driver, "codegen", mock_codegen)
    monkeypatch.setattr(driver, "harness", mock_harness)
    monkeypatch.setattr(driver, "llm", mock_llm)
    monkeypatch.setattr(driver, "NODES_LOG_PATH", str(tmp_path / "nodes.jsonl"))

    result = driver.run(
        max_iters=2, verbose=False,
        progress_path=str(tmp_path / "progress.json"),
        root_baseline_path=str(tmp_path / "root.json"),
        confirm_baseline_path=str(tmp_path / "confirm.json"),
        memory_path=str(tmp_path / "memory.json"))

    for key in ("global_best", "global_best_node_id", "baseline_primary",
                "best_valid_search", "best_valid_search_node_id",
                "iters_completed", "history", "counters"):
        assert key in result, key
    assert result["iters_completed"] == 2

    driver.print_final_summary(result)
    out = capsys.readouterr().out
    assert "=== final ===" in out
    assert "?" not in out, "a missing key fell through to the '?' placeholder"


def test_iteration_header_reports_current_best_and_baseline(monkeypatch, tmp_path,
                                                            capsys):
    """The per-iteration header format: open_nodes, current_best, baseline,
    elapsed — with one blank line separating each iteration block."""
    from orchestrator.mocks import codegen as mock_codegen
    from orchestrator.mocks import harness as mock_harness
    from orchestrator.mocks import llm as mock_llm

    monkeypatch.setattr(driver, "codegen", mock_codegen)
    monkeypatch.setattr(driver, "harness", mock_harness)
    monkeypatch.setattr(driver, "llm", mock_llm)
    monkeypatch.setattr(driver, "NODES_LOG_PATH", str(tmp_path / "nodes.jsonl"))

    driver.run(max_iters=2, verbose=True,
               progress_path=str(tmp_path / "progress.json"),
               root_baseline_path=str(tmp_path / "root.json"),
               confirm_baseline_path=str(tmp_path / "confirm.json"),
               memory_path=str(tmp_path / "memory.json"))
    lines = capsys.readouterr().out.splitlines()

    headers = [l for l in lines if l.startswith("[iter ") and "open_nodes=" in l]
    assert len(headers) == 2
    for h in headers:
        assert "open_nodes=" in h and "current_best=" in h
        assert "baseline=" in h and "elapsed=" in h
        # global_best was the old field name and is deliberately gone: it never
        # moves without a promotion, so it printed a constant equal to baseline.
        assert "global_best=" not in h
    # Each header is preceded by a blank line, so iterations read as blocks.
    for h in headers:
        assert lines[lines.index(h) - 1] == ""
