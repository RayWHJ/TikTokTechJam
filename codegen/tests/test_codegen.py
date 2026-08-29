"""
Offline, no-API-key test suite for codegen/. Run: `pytest codegen/tests -q`.

Covers all six contract functions plus the safety gate rules, using the
deterministic fake LLM backend (forced in conftest.py).
"""
import os, sys, textwrap, csv
import pytest

REPO_ROOT = os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
if REPO_ROOT not in sys.path:
    sys.path.insert(0, REPO_ROOT)

import codegen
from codegen import fixtures


# --------------------------------------------------------------------------- #
#  writer.py                                                                   #
# --------------------------------------------------------------------------- #
def test_write_fix_model_route_targets_baseline():
    diff = codegen.write_fix(fixtures.FAKE_HYPOTHESIS, "loss", root=REPO_ROOT)
    assert diff.strip()
    assert "baseline.py" in diff            # routed to model/loss file
    # a fresh gate over the generated diff should pass (writer is gate-aware)
    assert codegen.pre_execution_gate(diff)["pass"] is True


def test_write_fix_feature_route_targets_data():
    diff = codegen.write_fix(fixtures.FAKE_HYPOTHESIS_FEATURE, "user_history_feature",
                             root=REPO_ROOT)
    assert diff.strip()
    assert "data.py" in diff                # routed to feature-encoding file
    assert codegen.pre_execution_gate(diff)["pass"] is True


# --------------------------------------------------------------------------- #
#  gate.py                                                                     #
# --------------------------------------------------------------------------- #
def test_gate_blocks_test_split_access():
    diff = textwrap.dedent("""\
        --- a/baseline.py
        +++ b/baseline.py
        @@
        +    Xte, yte, ute = get_split("test")
        +    leak = evaluate(ute, yte, m.predict(Xte))
        """)
    res = codegen.pre_execution_gate(diff)
    assert res["pass"] is False
    assert any("test" in r.lower() for r in res["reasons"])


def test_gate_blocks_non_causal_without_marker():
    diff = textwrap.dedent("""\
        --- a/data.py
        +++ b/data.py
        @@
        +    feats.append(row['like_cnt'])
        +    feats.append(row['play_duration'])
        """)
    res = codegen.pre_execution_gate(diff)
    assert res["pass"] is False
    assert any("like_cnt" in r for r in res["reasons"])
    assert any("play_duration" in r for r in res["reasons"])


def test_gate_allows_non_causal_with_point_in_time_marker():
    diff = textwrap.dedent("""\
        --- a/data.py
        +++ b/data.py
        @@
        +    # this value is snapshotted at request time
        +    feats.append(get_value(row, 'like_cnt', point_in_time=True))
        """)
    res = codegen.pre_execution_gate(diff)
    assert res["pass"] is True, res["reasons"]


def test_gate_does_not_confuse_userside_bucket_with_noncausal():
    # follow_user_num_range is a coarse user bucket, NOT the follow_user_num stat.
    diff = "+    fields.append(row['follow_user_num_range'])\n"
    res = codegen.pre_execution_gate(diff)
    assert res["pass"] is True, res["reasons"]


def test_gate_blocks_external_pretrained():
    diff = "+from transformers import AutoModel\n+m = AutoModel.from_pretrained('bert-base')\n"
    res = codegen.pre_execution_gate(diff)
    assert res["pass"] is False
    assert any("external" in r.lower() for r in res["reasons"])


def test_gate_blocks_download():
    diff = "+    urllib.request.urlretrieve('https://example.com/w.pt', 'w.pt')\n"
    res = codegen.pre_execution_gate(diff)
    assert res["pass"] is False


def test_gate_blocks_auxiliary_as_input_feature():
    diff = textwrap.dedent("""\
        +    FIELDS.append('is_like')
        +    X = np.column_stack([X, row['is_like']])
        """)
    res = codegen.pre_execution_gate(diff)
    assert res["pass"] is False
    assert any("is_like" in r for r in res["reasons"])


def test_gate_allows_auxiliary_as_loss_target():
    diff = textwrap.dedent("""\
        +    aux_target = batch['is_like']
        +    aux_loss = bce(aux_head(z), aux_target)
        +    loss = main_loss + 0.2 * aux_loss
        """)
    res = codegen.pre_execution_gate(diff)
    assert res["pass"] is True, res["reasons"]


def test_gate_empty_diff_blocks():
    assert codegen.pre_execution_gate("")["pass"] is False
    assert codegen.pre_execution_gate("   \n")["pass"] is False


# --------------------------------------------------------------------------- #
#  sandbox.py                                                                  #
# --------------------------------------------------------------------------- #
def _write(path, body):
    with open(path, "w") as fh:
        fh.write(textwrap.dedent(body))
    return str(path)


def test_execute_ok(tmp_path):
    cand = _write(tmp_path / "cand.py", """\
        import json
        print("training...")
        print("##CODEGEN_METRICS## " + json.dumps({"GAUC":0.66,"nDCG@5":0.53,"primary":0.5955}))
        """)
    r = codegen.execute(cand, seed=0, split="valid_search", wallclock_cap_seconds=30,
                        root=REPO_ROOT)
    assert r["status"] == "ok", r
    assert abs(r["metrics"]["primary"] - 0.5955) < 1e-9


def test_execute_parses_baseline_style_output(tmp_path):
    cand = _write(tmp_path / "cand.py", """\
        print("  valid  GAUC 0.6674 | nDCG@5 0.5357 | primary 0.6016")
        """)
    r = codegen.execute(cand, seed=1, split="valid_search", wallclock_cap_seconds=30,
                        root=REPO_ROOT)
    assert r["status"] == "ok"
    assert r["metrics"]["primary"] == 0.6016


def test_execute_timeout(tmp_path):
    cand = _write(tmp_path / "cand.py", """\
        import time
        time.sleep(30)
        """)
    r = codegen.execute(cand, seed=0, split="valid_search", wallclock_cap_seconds=2,
                        root=REPO_ROOT)
    assert r["status"] == "timeout", r


def test_execute_diverged_on_nan(tmp_path):
    cand = _write(tmp_path / "cand.py", """\
        print("epoch 1 loss nan")
        print("primary 0.7")
        """)
    r = codegen.execute(cand, seed=0, split="valid_search", wallclock_cap_seconds=30,
                        root=REPO_ROOT)
    assert r["status"] == "diverged", r


def test_execute_error_on_crash(tmp_path):
    cand = _write(tmp_path / "cand.py", """\
        raise RuntimeError("boom")
        """)
    r = codegen.execute(cand, seed=0, split="valid_search", wallclock_cap_seconds=30,
                        root=REPO_ROOT)
    assert r["status"] == "error", r
    assert "boom" in r["logs"]


def test_execute_test_named_file_is_absent(tmp_path):
    # A candidate that tries to read a relative test-named file must fail with
    # FileNotFoundError, because no test-named file exists in the sandbox cwd.
    cand = _write(tmp_path / "cand.py", """\
        open("leaked_test_labels.csv").read()
        """)
    r = codegen.execute(cand, seed=0, split="valid_search", wallclock_cap_seconds=30,
                        root=REPO_ROOT)
    assert r["status"] == "error"
    assert "FileNotFoundError" in r["logs"]


def test_execute_renames_test_named_candidate(tmp_path):
    cand = _write(tmp_path / "my_test_candidate.py", """\
        print("primary 0.60")
        """)
    r = codegen.execute(cand, seed=0, split="valid_search", wallclock_cap_seconds=30,
                        root=REPO_ROOT)
    assert r["status"] == "ok", r


# --------------------------------------------------------------------------- #
#  debug.py                                                                    #
# --------------------------------------------------------------------------- #
def test_debug_and_retry_returns_contract_keys(tmp_path):
    cand = _write(tmp_path / "baseline.py", "x = 1\n")
    r = codegen.debug_and_retry(cand, "Traceback ... IndexError")
    assert set(("code_diff", "is_semantic_change")).issubset(r)
    assert isinstance(r["code_diff"], str) and r["code_diff"].strip()
    assert isinstance(r["is_semantic_change"], bool)


def test_is_semantic_change_classifier():
    semantic = "+    loss = bpr(zp, zn)   # switch objective\n"
    crashfix = "+        inter = 0.5 * (S**2).sum((1,2))  # fix axis\n"
    assert codegen.is_semantic_change(semantic) is True
    assert codegen.is_semantic_change(crashfix) is False


def test_debug_sanity_flags_implausible_score(tmp_path):
    cand = _write(tmp_path / "baseline.py", "x = 1\n")
    r = codegen.debug_and_retry(
        cand, "n/a", observed_score=0.95,           # above the 0.8645 oracle ceiling
        history=[0.5946, 0.599], hypothesis=fixtures.FAKE_HYPOTHESIS)
    assert "sanity" in r
    assert r["leak_suspected"] is True


def test_debug_sanity_not_triggered_for_plausible_score(tmp_path):
    cand = _write(tmp_path / "baseline.py", "x = 1\n")
    r = codegen.debug_and_retry(cand, "n/a", observed_score=0.60,
                                history=[0.5946, 0.599])
    assert "sanity" not in r


# --------------------------------------------------------------------------- #
#  submission.py                                                               #
# --------------------------------------------------------------------------- #
def _tiny_dataset(d):
    os.makedirs(d, exist_ok=True)
    with open(os.path.join(d, "video_features_basic_pure.csv"), "w", newline="") as fh:
        w = csv.writer(fh); w.writerow(["video_id", "author_id"])
        for v in range(5):
            w.writerow([v, 100 + v])
    # first log file (train window) — needed by data.load, can be minimal
    with open(os.path.join(d, "log_standard_4_08_to_4_21_pure.csv"), "w", newline="") as fh:
        w = csv.writer(fh)
        w.writerow(["date", "user_id", "video_id", "tab", "duration_ms", "long_view"])
        w.writerow([20220410, 0, 0, "1", 5000, 1])
    # second log file (holds the valid-window rows we validate against)
    with open(os.path.join(d, "log_standard_4_22_to_5_08_pure.csv"), "w", newline="") as fh:
        w = csv.writer(fh)
        w.writerow(["date", "user_id", "video_id", "tab", "duration_ms", "long_view"])
        rows = [(20220423, 0, 1), (20220423, 0, 2), (20220424, 1, 3), (20220424, 1, 4)]
        for dt, u, v in rows:
            w.writerow([dt, u, v, "1", 6000, 0])
    return [(0, 1), (0, 2), (1, 3), (1, 4)]   # (user_id, video_id) of the valid rows


def test_check_submission_aligned_true(tmp_path):
    d = str(tmp_path / "data")
    valid_pairs = _tiny_dataset(d)
    sub = tmp_path / "sub.csv"
    with open(sub, "w", newline="") as fh:
        w = csv.writer(fh); w.writerow(["row_id", "user_id", "video_id", "score"])
        for i, (u, v) in enumerate(valid_pairs):
            w.writerow([i, u, v, -1.5 + i])
    assert codegen.check_submission(str(sub), "valid", data_dir=d, root=REPO_ROOT) is True


def test_check_submission_misaligned_false(tmp_path):
    d = str(tmp_path / "data")
    valid_pairs = _tiny_dataset(d)
    sub = tmp_path / "sub.csv"
    with open(sub, "w", newline="") as fh:
        w = csv.writer(fh); w.writerow(["row_id", "user_id", "video_id", "score"])
        for i, (u, v) in enumerate(valid_pairs):
            w.writerow([i, u, 999, 0.0])          # wrong video_id -> misaligned
    assert codegen.check_submission(str(sub), "valid", data_dir=d, root=REPO_ROOT) is False


# --------------------------------------------------------------------------- #
#  report.py                                                                   #
# --------------------------------------------------------------------------- #
def test_synthesize_report_markdown_contains_facts():
    md = codegen.synthesize_report(fixtures.FAKE_RUN_LOG)
    assert isinstance(md, str) and md.startswith("#")
    assert "0.5946" in md          # baseline mentioned
    assert "0.6013" in md          # promoted best mentioned
