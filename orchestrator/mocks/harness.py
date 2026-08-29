"""Mock of harness/. Signatures match Person A's frozen contract exactly."""
import random

_TEST_CALLED = False

def validated_evaluate(user_ids, labels, scores, split_name):
    assert len(user_ids) == len(labels) == len(scores)
    base = 0.5946 + random.gauss(0, 0.005)  # jitter around FM baseline
    return {"GAUC": base + 0.03, "nDCG@5": base - 0.03,
            "primary": base, "users": len(set(user_ids)), "rows": len(labels)}

def get_split(name):
    assert name in {"train", "valid_search", "valid_confirm", "test"}
    global _TEST_CALLED
    if name == "test":
        if _TEST_CALLED:
            raise RuntimeError("test split is one-shot; already consumed")
        _TEST_CALLED = True
    # tiny fake arrays — real harness returns numpy; a list works for the stub
    X = [[0, 1, 2, 3, 4]] * 100
    y = [i % 2 for i in range(100)]
    users = [f"u{i % 10}" for i in range(100)]
    return X, y, users

_NON_CAUSAL = {"show_cnt", "play_cnt", "like_cnt", "cancel_like_cnt"}  # abbreviated for mock

def check_provenance(column_names, point_in_time=False):
    if point_in_time:
        return
    bad = [c for c in column_names if c in _NON_CAUSAL]
    if bad:
        raise ValueError(f"non-causal columns without point_in_time: {bad}")