"""Tests for the auxiliary-target plumbing (Tier 4C).

Direction 3 (multi-task auxiliary targets) was structurally unreachable, for two
independent reasons: data.load() never read is_click / is_like / is_follow /
is_comment / is_forward / play_time_ms, and the one-file rule forbade a candidate
from adding them in data.py and consuming them in baseline.py at once. So no
candidate could reach it however it was written, which is why 0 of 11 in the
recorded run tried.

The load() change is APPEND-ONLY and that is the risk this module exists to
cover: data.py, baseline.py and harness/_data.py all index row tuples
positionally, and evaluate.py's callers depend on x[6] being the label.
"""
import csv
import os

import numpy as np
import pytest

import data
from codegen import gate
from codegen.constants import AUXILIARY_SIGNALS


# --------------------------------------------------------------------------- #
#  Layout: indices 0..6 must not move                                          #
# --------------------------------------------------------------------------- #
def test_aux_signals_order_is_pinned():
    """Column order is a contract: aux_targets returns columns in this order and
    a candidate indexes them by position."""
    assert data.AUX_SIGNALS == ['is_click', 'is_like', 'is_follow',
                               'is_comment', 'is_forward', 'play_time_ms']
    assert data.AUX_OFFSET == 7


def test_gate_constant_covers_every_loaded_aux_signal():
    """gate.py rule 4 blocks a same-row aux value used as an input. If
    AUX_SIGNALS gains a name that AUXILIARY_SIGNALS lacks, that signal becomes
    silently leakable."""
    missing = set(data.AUX_SIGNALS) - set(AUXILIARY_SIGNALS)
    assert not missing, f"not covered by the gate: {sorted(missing)}"


def _tiny_dataset(tmp_path, *, with_aux=True, extra_rows=()):
    """Two CSVs shaped like KuaiRand-Pure's, small enough to load instantly."""
    d = tmp_path / "data"
    d.mkdir(exist_ok=True)
    with open(d / "video_features_basic_pure.csv", "w", newline="") as fh:
        w = csv.writer(fh)
        w.writerow(["video_id", "author_id"])
        w.writerow(["v1", "a1"])
        w.writerow(["v2", "a2"])

    cols = ["date", "user_id", "video_id", "tab", "duration_ms", "long_view"]
    if with_aux:
        cols += list(data.AUX_SIGNALS)
    rows = [
        # date, user, video, tab, duration, long_view, then aux
        ["20220408", "u1", "v1", "1", "1000", "1", "1", "1", "0", "0", "0", "5000"],
        ["20220409", "u1", "v2", "1", "2000", "0", "1", "0", "0", "0", "0", "120"],
        ["20220422", "u2", "v1", "0", "3000", "1", "0", "0", "1", "0", "0", "900"],
        ["20220429", "u2", "v2", "0", "4000", "0", "0", "0", "0", "0", "0", "0"],
    ]
    for name, rng in (("log_standard_4_08_to_4_21_pure.csv", (20220408, 20220421)),
                      ("log_standard_4_22_to_5_08_pure.csv", (20220422, 20220508))):
        with open(d / name, "w", newline="") as fh:
            w = csv.writer(fh)
            w.writerow(cols)
            for r in rows:
                if rng[0] <= int(r[0]) <= rng[1]:
                    w.writerow(r[:len(cols)])
    return str(d)


def test_label_is_still_at_index_six(tmp_path):
    """The single most load-bearing assertion here. baseline.py::run_pop uses
    x[6], data.py::encode uses x[6], and evaluate.py's callers depend on it."""
    splits = data.load(_tiny_dataset(tmp_path))
    for rws in splits.values():
        for x in rws:
            assert x[6] in (0, 1)
    train = splits['train']
    assert [x[6] for x in train] == [1, 0], "label order changed"


def test_indices_zero_to_six_keep_their_meaning(tmp_path):
    splits = data.load(_tiny_dataset(tmp_path))
    x = splits['train'][0]
    assert x[0] == 20220408          # date
    assert x[1] == 'u1'              # user_id
    assert x[2] == 'v1'              # video_id
    assert x[3] == 'a1'              # author_id, joined from video features
    assert x[4] == '1'               # tab
    assert x[5] == 1000.0            # duration_ms
    assert x[6] == 1                 # long_view


def test_aux_values_land_at_index_seven_onward(tmp_path):
    splits = data.load(_tiny_dataset(tmp_path))
    x = splits['train'][0]
    assert len(x) == data.AUX_OFFSET + len(data.AUX_SIGNALS)
    aux = dict(zip(data.AUX_SIGNALS, x[data.AUX_OFFSET:]))
    assert aux['is_click'] == 1
    assert aux['is_like'] == 1
    assert aux['is_follow'] == 0
    assert aux['play_time_ms'] == 5000.0


def test_a_missing_aux_column_fills_zero_instead_of_raising(tmp_path):
    """The sandbox fixtures and several tests use cut-down CSVs. A KeyError
    here would break loading entirely for a signal the candidate may not use."""
    splits = data.load(_tiny_dataset(tmp_path, with_aux=False))
    x = splits['train'][0]
    assert len(x) == data.AUX_OFFSET + len(data.AUX_SIGNALS)
    assert list(x[data.AUX_OFFSET:]) == [0, 0, 0, 0, 0, 0.0]
    assert x[6] == 1, "the label must still be right when aux columns are absent"


# --------------------------------------------------------------------------- #
#  aux_targets: shape and alignment                                            #
# --------------------------------------------------------------------------- #
def test_aux_targets_aligns_row_for_row_with_encode_for_every_split(tmp_path):
    """The alignment claim is the whole contract — misaligned targets would
    train an auxiliary head against another row's outcome and look like a
    plausible-but-wrong result rather than a crash."""
    splits = data.load(_tiny_dataset(tmp_path))
    enc, _ = data.encode(splits)
    aux = data.aux_targets(splits)

    assert set(aux) == set(splits)
    for name in splits:
        X = enc[name][0]
        A = aux[name]
        assert A.shape == (X.shape[0], len(data.AUX_SIGNALS)), name
        assert A.dtype == np.float32, name
        # Row n of A must be row n of splits[name], the same list encode() walked.
        for n, x in enumerate(splits[name]):
            assert list(A[n]) == pytest.approx(
                [float(v) for v in x[data.AUX_OFFSET:]]), (name, n)


def test_aux_targets_tolerates_hand_built_seven_tuples():
    """Synthetic 7-tuples are used across the suite; they must yield zeros
    rather than a broadcast error."""
    rows = [(20220408, 'u1', 'v1', 'a1', '1', 1000.0, 1)]
    A = data.aux_targets({'train': rows})['train']
    assert A.shape == (1, len(data.AUX_SIGNALS))
    assert not A.any()


def test_aux_targets_handles_an_empty_split():
    A = data.aux_targets({'train': []})['train']
    assert A.shape == (0, len(data.AUX_SIGNALS))


# --------------------------------------------------------------------------- #
#  The hard rule: TARGET, never INPUT                                          #
# --------------------------------------------------------------------------- #
def test_gate_blocks_aux_targets_built_into_a_feature_matrix():
    """The leak the seam makes possible. None of the individual signal names
    appear on this line, so rule 4 alone would pass it."""
    diff = "+    X = np.column_stack([X, aux_targets(splits)['train']])\n"
    res = gate.pre_execution_gate(diff)
    assert res["pass"] is False
    assert any("FEATURE MATRIX" in r for r in res["reasons"])


@pytest.mark.parametrize("line", [
    "    X = np.concatenate([X, aux_targets(splits)['train']], axis=1)",
    "    Xtr = np.hstack([Xtr, A])  # A = aux_targets(splits)['train']",
    "    features = aux_targets(splits)['train']",
    "    FIELDS.append(AUX_SIGNALS[0])",
])
def test_gate_blocks_other_shapes_of_the_same_leak(line):
    res = gate.pre_execution_gate("+" + line + "\n")
    assert res["pass"] is False, line


def test_gate_allows_aux_targets_as_a_loss_target():
    """The intended use, and the reason the seam check is narrow rather than
    routed through _is_input_use: this line mentions X, and "aux_targets"
    contains the substring "target", so a broader rule blocks exactly the usage
    the plumbing exists to enable."""
    diff = (
        "+    A = aux_targets(splits)['train_fit']\n"
        "+    aux_logits = m.aux_head(X)\n"
        "+    aux_loss = bce(aux_logits, A[idx])\n"
        "+    loss = main_loss + 0.2 * aux_loss\n"
    )
    res = gate.pre_execution_gate(diff)
    assert res["pass"] is True, res["reasons"]


def test_gate_allows_the_single_line_multitask_loss_form():
    """The most natural way to write it, which mentions X and aux_targets on one
    line. A rule keyed on _is_input_use would reject this."""
    diff = "+    aux_loss = bce(m.aux_head(X), aux_targets(splits)['train'][idx])\n"
    res = gate.pre_execution_gate(diff)
    assert res["pass"] is True, res["reasons"]


def test_gate_still_blocks_a_bare_signal_name_as_an_input():
    """Rule 4's original behaviour must survive the addition."""
    res = gate.pre_execution_gate(
        "+    X = np.column_stack([X, row['is_like']])\n")
    assert res["pass"] is False
    assert any("is_like" in r for r in res["reasons"])


# --------------------------------------------------------------------------- #
#  Reachability: the prompts must say the direction is now a one-file edit      #
# --------------------------------------------------------------------------- #
def test_prompts_state_that_multitask_is_a_baseline_only_change():
    from llm_calls import personas
    from codegen import prompts
    assert "aux_targets(splits)" in personas._DATASET_CONTEXT
    assert "baseline.py-ONLY" in personas._DATASET_CONTEXT
    assert "aux_targets(splits)" in prompts._SAFETY_RULES
    # And the relaxation must stay narrow, not become a two-file licence.
    assert "not permission for a two-file" in personas._DATASET_CONTEXT


def test_prompts_still_state_the_target_not_input_rule():
    from llm_calls import personas
    from codegen import prompts
    assert "TARGET, never an INPUT" in personas._DATASET_CONTEXT
    assert "LOSS TARGETS" in prompts._SAFETY_RULES


# --------------------------------------------------------------------------- #
#  Against the real dataset                                                    #
# --------------------------------------------------------------------------- #
@pytest.mark.slow
def test_real_load_carries_aux_and_keeps_the_label_at_six():
    from harness._data import DEFAULT_DATA_DIR
    if not os.path.isdir(DEFAULT_DATA_DIR):
        pytest.skip("KuaiRand-Pure CSVs absent")
    splits = data.load(DEFAULT_DATA_DIR)
    x = splits['train'][0]
    assert len(x) == data.AUX_OFFSET + len(data.AUX_SIGNALS)
    assert x[6] in (0, 1)
    # play_time_ms is continuous and should not be all-zero on real data.
    idx = data.AUX_OFFSET + data.AUX_SIGNALS.index('play_time_ms')
    assert any(r[idx] > 0 for r in splits['train'][:1000])
    # is_click likewise.
    ci = data.AUX_OFFSET + data.AUX_SIGNALS.index('is_click')
    assert any(r[ci] == 1 for r in splits['train'][:1000])
