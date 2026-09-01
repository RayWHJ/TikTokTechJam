"""baseline.py duplicates the valid_search / valid_confirm date ranges because
codegen/sandbox.py deliberately does not expose the harness package to
candidates (that is what keeps harness.get_split('test') out of their reach).
Duplication is only safe if it cannot drift, so assert the two agree.
"""
import baseline
from data import SPLITS
from harness._sizes import SPLIT_RANGES


def test_valid_subsplit_ranges_match_harness():
    for name, rng in baseline.VALID_SUBSPLITS.items():
        assert tuple(rng) == tuple(SPLIT_RANGES[name]), name


def test_valid_subsplits_partition_official_valid():
    lo, hi = SPLITS['valid']
    search_lo, search_hi = baseline.VALID_SUBSPLITS['valid_search']
    confirm_lo, confirm_hi = baseline.VALID_SUBSPLITS['valid_confirm']
    assert search_lo == lo
    assert confirm_hi == hi
    assert search_hi < confirm_lo          # disjoint, in order
    assert baseline.VALID_SUBSPLITS.keys() == {'valid_search', 'valid_confirm'}


def test_cut_valid_subsplits_partitions_rows_without_loss():
    # date is field 0; only that field matters to the cut.
    valid = [(d, f"u{d}", "v", "a", "1", 1000.0, d % 2) for d in range(20220422, 20220429)]
    out = baseline.cut_valid_subsplits({"train": [], "valid": valid, "test": []})
    search, confirm = out["valid_search"], out["valid_confirm"]
    assert len(search) + len(confirm) == len(valid)
    assert not ({id(x) for x in search} & {id(x) for x in confirm})
    assert out["valid"] == valid           # original untouched


def test_per_user_primary_keeps_only_discriminative_users():
    # u_mixed has both labels -> rankable; the other two do not.
    users = ["u_mixed", "u_mixed", "u_allpos", "u_allpos", "u_allneg", "u_allneg"]
    labels = [1, 0, 1, 1, 0, 0]
    scores = [0.9, 0.1, 0.8, 0.7, 0.2, 0.1]
    out = baseline.per_user_primary(users, labels, scores)
    assert set(out) == {"u_mixed"}
    assert 0.0 <= out["u_mixed"] <= 1.0
