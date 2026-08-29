"""Mock of codegen/. Simulates gates, sandboxed runs, per-user results."""
import random

def write_fix(hypothesis, target_component):
    return (f"# fake diff for {target_component}\n"
            f"# mechanism: {hypothesis['mechanism']}\n"
            f"# sketch: {hypothesis.get('implementation_sketch', '')}\n")

def pre_execution_gate(code_diff):
    if "cancel_like_cnt" in code_diff and "point_in_time=True" not in code_diff:
        return {"pass": False, "reasons": ["non-causal column without PIT marker"]}
    if "test" in code_diff.lower() and "read" in code_diff.lower():
        return {"pass": False, "reasons": ["suspected test-split file read"]}
    return {"pass": True, "reasons": []}

def execute(code_path, seed, split, wallclock_cap_seconds):
    if random.random() < 0.05:
        return {"status": "error", "metrics": {}, "logs": "fake traceback"}
    rng = random.Random(seed)  # stable per seed so bootstrap sees signal
    users = [f"u{i}" for i in range(10)]
    per_user = {u: 0.5946 + rng.gauss(0.005, 0.02) for u in users}
    primary = sum(per_user.values()) / len(per_user)
    return {"status": "ok",
            "metrics": {"primary": primary,
                        "GAUC": primary + 0.03,
                        "nDCG@5": primary - 0.03,
                        "per_user": per_user},
            "logs": ""}

def debug_and_retry(code_path, error_context):
    # semantic change only when we actually rewrite the mechanism
    return {"code_diff": "# fake repair\n", "is_semantic_change": False}

def check_submission(path, split):
    return True