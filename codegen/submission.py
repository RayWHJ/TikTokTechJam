"""
codegen.check_submission — hard format/alignment gate for a submission file.

Wraps submit.py's `--check` logic EXACTLY (submit.read_submission), which
validates header, row count, row_id continuity, and (user_id, video_id)
alignment against the split — and never reads the label column. Returns a plain
bool so the orchestrator can use it as a gate.
"""
from __future__ import annotations
import os, sys


def _ensure_root_on_path(root: str):
    root = os.path.abspath(root)
    if root not in sys.path:
        sys.path.insert(0, root)


def check_submission(path: str, split: str, *,
                     data_dir: str | None = None, root: str = ".") -> bool:
    """Return True iff `path` passes submit.py's --check for `split`.

    Never reads labels (mirrors submit.py --check). Any format/alignment error,
    a missing file, or a missing dataset returns False (with the reason printed);
    it never raises, so it is safe to call as a gate.
    """
    _ensure_root_on_path(root)
    data_dir = data_dir or os.environ.get(
        "CODEGEN_DATA_DIR", os.path.join(root, "KuaiRand-Pure", "data"))
    try:
        from data import load
        from submit import read_submission
    except Exception as e:
        print(f"[check_submission] cannot import starter kit: {e}")
        return False
    if split not in ("valid", "test"):
        # submit.py only knows valid/test; the harness's finer tiers all live
        # inside the valid date-window, so map them onto 'valid' for alignment.
        split = "valid"
    try:
        rows = load(data_dir)[split]
    except Exception as e:
        print(f"[check_submission] cannot load split {split!r} from {data_dir}: {e}")
        return False
    try:
        read_submission(path, rows)          # the exact --check logic; labels untouched
        return True
    except Exception as e:
        print(f"[check_submission] FAILED: {e}")
        return False
