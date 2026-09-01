"""
codegen.smoke_check — a sub-second dry run of the candidate contract, on 200
synthetic rows, before anything expensive happens.

WHY THIS EXISTS. Every execution failure in the recorded run was detectable on
200 rows: a shape mismatch, an `IndexError` in `data.py::encode`, and a mechanism
whose per-row cost was obvious immediately. Each of those cost a 240 s triage run
plus up to two further 240 s repair runs, so one bad edit could burn 12 minutes
to learn what a 1-second check would have said. Three of five candidates never
produced a paired delta at all.

Moving the repair loop onto the cheap signal is what lets the driver raise the
repair budget without raising the bill: only code already proven to RUN reaches
the expensive path.

WHAT IT CHECKS. The real contract, not a proxy. The subprocess calls the
candidate's own `run_for_orchestrator` — the exact function
`codegen.sandbox.execute` ends up in — with `data.load` swapped for a synthetic
fixture, then parses the `##CODEGEN_METRICS##` line back. So it exercises
encode(), the vocabulary and offset arithmetic, the training loop, the candidate's
loss, evaluate(), per_user_primary() and the JSON payload, all on 200 rows.

TWO DELIBERATE DESIGN CHOICES.

  * The fixture is built SYNTHETICALLY rather than by truncating the real split.
    So the smoke stage works when `KuaiRand-Pure/data/` is absent, and it costs
    no dataset I/O.
  * The fixture is built HERE, by this module, not by anything in the candidate's
    tree. A candidate's whole-file rewrite therefore cannot disable the check by
    dropping a hook — the only way past it is to actually run.

It runs in a subprocess because the orchestrator has already imported its OWN
`data` and `baseline`; importing a candidate's copies in-process would either
silently no-op against sys.modules or corrupt the driver's modules for the rest
of the run.
"""
from __future__ import annotations

import os
import signal
import subprocess
import sys
import time

#: Rows in the synthetic fixture. Small enough to be sub-second, large enough
#: that np.quantile's 9 duration cut points and the per-user GAUC filter
#: (0 < positives < impressions) both have something to work with.
FIXTURE_ROWS = 200
FIXTURE_USERS = 20
ROWS_PER_USER = FIXTURE_ROWS // FIXTURE_USERS

#: Hard ceiling. Generous relative to the ~0.5 s this actually takes, because the
#: cost of being wrong is asymmetric: a false timeout discards a good candidate,
#: while a slow-but-correct candidate is caught by the real wall-clock cap later.
#: A mechanism that cannot do 200 rows in 20 s cannot do 1.4M in 240 s either.
SMOKE_TIMEOUT_S = 20

_METRICS_MARK = "##CODEGEN_METRICS##"

#: How much of the child's output to carry back on failure. Enough for a Python
#: traceback's final frames plus the exception line, which is the part that says
#: what to fix.
_ERROR_CHARS = 2000


# --------------------------------------------------------------------------- #
#  The probe, run inside the candidate's tree                                  #
# --------------------------------------------------------------------------- #
# Passed with `python -c`, so sys.path[0] is "" (the cwd) and `import data` /
# `import baseline` resolve to the CANDIDATE's copies rather than the repo's.
# A temp .py file would put ITS OWN directory on sys.path instead and silently
# smoke-test the orchestrator's own modules.
_PROBE = r'''
import json, os, sys, types

FIXTURE_USERS = {users}
ROWS_PER_USER = {per_user}

def _fail(stage, exc):
    import traceback
    print("SMOKE_STAGE " + stage, file=sys.stderr)
    traceback.print_exc()
    raise SystemExit(3)

try:
    import numpy as np
    import data
    import baseline
except Exception as e:
    _fail("import", e)

# --- the fixture ------------------------------------------------------------
# Dates are chosen so every split AND every sub-split the orchestrator path
# needs is non-empty: train_fit / train_es (data.TRAIN_SUBSPLITS) and
# valid_search / valid_confirm (baseline.VALID_SUBSPLITS).
#
# Labels are laid out so each user has BOTH classes inside train_es (the
# early-stopping split) and inside valid_search (the reported split). Without
# that, GAUC's 0 < positives < impressions filter excludes every user and the
# metric is undefined on a fixture rather than on the candidate.
_DAYS = [20220408, 20220412, 20220416, 20220419,   # train_fit
         20220420, 20220421,                        # train_es
         20220422, 20220425,                        # valid_search
         20220427,                                  # valid_confirm
         20220430]                                  # test
_LABELS = [1, 0, 1, 0,   1, 0,   1, 0,   1,   0]

try:
    n_aux = len(getattr(data, "AUX_SIGNALS", []))
    rows = []
    for u in range(FIXTURE_USERS):
        for j in range(ROWS_PER_USER):
            day = _DAYS[j % len(_DAYS)]
            label = _LABELS[j % len(_LABELS)]
            # Vocabulary overlap across users is what makes the FM's embeddings
            # do anything at all on a fixture this size.
            vid = "v%d" % ((u * 3 + j) % 31)
            aut = "a%d" % ((u + j) % 7)
            tab = str((u + j) % 3)
            dur = float(1000 + ((u * 37 + j * 11) % 89) * 250)
            aux = tuple([label if k < 2 else 0 for k in range(n_aux)])
            rows.append((day, "u%d" % u, vid, aut, tab, dur, label) + aux)
except Exception as e:
    _fail("fixture", e)

# Cut with the CANDIDATE's own SPLITS, mirroring the tail of data.load, so a
# candidate that changed the ranges is still tested against its own definition.
try:
    splits = {{}}
    for name, (lo, hi) in data.SPLITS.items():
        splits[name] = [x for x in rows if lo <= x[0] <= hi]
    for name, rws in splits.items():
        if not rws:
            raise AssertionError(
                "fixture produced no rows for split %r over range %r"
                % (name, data.SPLITS[name]))
except Exception as e:
    _fail("split", e)

# --- stage 1: encode() ------------------------------------------------------
# Checked separately from the training run so the error names the encoder rather
# than whatever downstream shape mismatch it caused. This is the stage that
# catches the IndexError class: two candidates in the recorded run extended
# raw(x) without extending FIELDS and died identically inside
# `for i, v in enumerate(raw(x))`.
try:
    enc, dim = data.encode(splits)
    fields = (data.active_fields() if hasattr(data, "active_fields")
              else list(data.FIELDS))
    for name, rws in splits.items():
        X, y, users = enc[name]
        if X.shape != (len(rws), len(fields)):
            raise AssertionError(
                "encode()[%r] returned X of shape %s; expected (%d, %d) = "
                "(rows, len(FIELDS)).\n  FIELDS (%d): %s"
                % (name, X.shape, len(rws), len(fields), len(fields), fields))
        if X.size and (int(X.max()) >= dim or int(X.min()) < 0):
            raise AssertionError(
                "encode()[%r] produced feature ids outside [0, dim=%d): "
                "min %d max %d — the FM indexes its embedding table with these, "
                "so this is an IndexError at train time."
                % (name, dim, int(X.min()), int(X.max())))
        if len(users) != len(rws):
            raise AssertionError(
                "encode()[%r] returned %d user ids for %d rows"
                % (name, len(users), len(rws)))
except Exception as e:
    _fail("encode", e)

# --- stage 2: the real orchestrator entry point, one epoch ------------------
# baseline.run_for_orchestrator is exactly what codegen.sandbox.execute reaches.
# Calling it (rather than poking FM.step directly) means a candidate's custom
# loss, its early-stopping path, evaluate() and per_user_primary all run.
try:
    baseline.load = lambda data_dir=None: {{
        k: list(v) for k, v in splits.items()}}
    args = types.SimpleNamespace(model=None, data_dir=None, seed=0,
                                 k=4, lr=0.01, epochs=1)
    baseline.run_for_orchestrator(args, "valid_search")
except Exception as e:
    _fail("train", e)
print("SMOKE_OK")
'''


def _build_probe() -> str:
    return _PROBE.format(users=FIXTURE_USERS, per_user=ROWS_PER_USER)


# --------------------------------------------------------------------------- #
#  Public entry point                                                          #
# --------------------------------------------------------------------------- #
def smoke_check(root: str, *, timeout_s: int = SMOKE_TIMEOUT_S,
                python: str | None = None) -> dict:
    """Run the 200-row probe against the candidate tree at `root`.

    Returns {"ok": bool, "error": str, "seconds": float, "stage": str}.

    `error` is empty on success. On failure it names the STAGE that failed
    (import / fixture / split / encode / train) followed by the child's
    traceback tail, because "the encoder is wrong" and "the loss is wrong" call
    for different repairs and a bare traceback does not distinguish them.

    Never raises: a smoke stage that can itself crash the driver is worse than
    no smoke stage. An unexpected failure of the probe machinery reports ok=True
    with a note, so the candidate proceeds to the real run exactly as it did
    before this stage existed rather than being discarded on the checker's bug.
    """
    t0 = time.time()
    if not root or not os.path.isdir(root):
        return {"ok": True, "error": "", "seconds": 0.0,
                "stage": "skipped: no candidate directory"}

    env = dict(os.environ)
    env["PYTHONHASHSEED"] = "0"
    # The probe supplies its own fixture, so nothing here should touch the real
    # dataset. Clearing these makes an accidental data.load() fail loudly in the
    # child instead of quietly costing 20 s of CSV parsing.
    env.pop("CODEGEN_DATA_DIR", None)
    env.pop("CODEGEN_SPLIT", None)
    env.pop("HARNESS_DATA_DIR", None)

    cmd = [python or sys.executable, "-c", _build_probe()]
    try:
        proc = subprocess.Popen(
            cmd, cwd=root, env=env, text=True,
            stdin=subprocess.DEVNULL,
            stdout=subprocess.PIPE, stderr=subprocess.PIPE,
            start_new_session=True)     # own group, so a hang can be killed whole
    except Exception as e:              # noqa: BLE001 — see docstring
        return {"ok": True, "error": "", "seconds": time.time() - t0,
                "stage": f"skipped: could not launch probe ({e})"}

    try:
        out, err = proc.communicate(timeout=timeout_s)
    except subprocess.TimeoutExpired:
        try:
            os.killpg(os.getpgid(proc.pid), signal.SIGKILL)
        except (OSError, ProcessLookupError):
            proc.kill()
        out, err = proc.communicate()
        return {"ok": False, "seconds": time.time() - t0, "stage": "timeout",
                "error": (f"smoke stage timed out after {timeout_s}s on "
                          f"{FIXTURE_ROWS} synthetic rows. A mechanism that "
                          f"cannot finish 200 rows in {timeout_s}s cannot "
                          f"finish 1.4M rows inside the triage cap — it needs a "
                          f"vectorised implementation, not a longer cap.\n"
                          + (err or out or "")[-_ERROR_CHARS:])}

    seconds = time.time() - t0
    if proc.returncode == 0 and "SMOKE_OK" in (out or ""):
        if _METRICS_MARK not in (out or ""):
            return {"ok": False, "seconds": seconds, "stage": "metrics",
                    "error": ("the candidate ran but never printed a "
                              f"`{_METRICS_MARK} {{json}}` line. codegen.sandbox "
                              "parses that line to read the score, so without "
                              "it the run is unscoreable.\n"
                              + (out or "")[-_ERROR_CHARS:])}
        return {"ok": True, "error": "", "seconds": seconds, "stage": "ok"}

    stage = "unknown"
    for line in (err or "").splitlines():
        if line.startswith("SMOKE_STAGE "):
            stage = line.split(" ", 1)[1].strip()
            break
    tail = (err or "")[-_ERROR_CHARS:] or (out or "")[-_ERROR_CHARS:]
    return {"ok": False, "seconds": seconds, "stage": stage,
            "error": (f"smoke stage failed at {stage!r} on {FIXTURE_ROWS} "
                      f"synthetic rows (exit {proc.returncode}):\n{tail}")}
