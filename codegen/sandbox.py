"""
codegen.execute — run a candidate in an isolated subprocess.

Defence in depth against the #1 failure mode (test-label leakage):
  * the candidate runs in a fresh temp working directory that contains COPIES of
    the unchanged root modules EXCEPT any file whose name contains "test" — those
    are physically absent, so a relative open() of a test-named file fails with
    FileNotFoundError instead of silently leaking;
  * the split the candidate is told to score is passed as CODEGEN_SPLIT (never as
    a bare "test" file path);
  * a hard wall-clock cap kills the whole process group;
  * stdout/stderr are scanned for NaN/inf/divergence.

Candidate CLI/env contract (writer-generated candidates honour this):
  invoked as:  python <candidate> --data_dir <dir> --seed <seed>
  env:         CODEGEN_SPLIT, CODEGEN_SEED, CODEGEN_DATA_DIR, PYTHONHASHSEED
  output:      a line `##CODEGEN_METRICS## {json}` (preferred), or baseline-style
               "... primary 0.5946 ..." lines, which are parsed as a fallback.

Returns {"status": "ok"|"error"|"timeout"|"diverged", "metrics": dict, "logs": str}.
"""
from __future__ import annotations
import os, sys, re, json, shutil, tempfile, subprocess

_ROOT_FILES = ("data.py", "evaluate.py", "baseline.py", "submit.py")
_METRICS_MARK = "##CODEGEN_METRICS##"
_DIVERGENCE_RE = re.compile(r"\b(nan|inf|-inf|\+inf)\b|overflow|loss diverged|not finite",
                            re.IGNORECASE)
_LOG_TAIL = 8000


# --------------------------------------------------------------------------- #
#  Label-permutation control (T3.4)                                            #
# --------------------------------------------------------------------------- #
# Appended to the WORKDIR's throwaway copy of evaluate.py when a control run is
# requested. The repo's evaluate.py is never modified — it is a frozen deliverable
# whose own header says the metric definition must not change — and the candidate
# cannot disable this either, because it is appended AFTER the candidate's diff
# has been staged and copied in.
#
# The shuffle is WITHIN EACH USER, which is what makes it the right null for this
# metric: it preserves every user's positive count, so GAUC's
# `0 < npos < len(labs)` filter selects exactly the same users and nDCG's ideal
# DCG is unchanged, while destroying the association between labels and scores. A
# model carrying no label information therefore lands at chance; a model reading
# the label keeps its score, and that gap is the entire signal.
_PERMUTE_SHIM = '\n'.join([
    "# --- appended by codegen.sandbox: label-permutation control ---",
    "import os as _pc_os",
    "",
    'if _pc_os.environ.get("CODEGEN_PERMUTE_LABELS"):',
    "    import collections as _pc_collections",
    "    import random as _pc_random",
    "",
    "    _pc_real_evaluate = evaluate",
    "",
    "    def evaluate(user_ids, labels, scores, k=5):",
    '        """evaluate() with each user\'s labels shuffled among its own rows."""',
    "        rng = _pc_random.Random(",
    '            int(_pc_os.environ.get("CODEGEN_PERMUTE_SEED", "0")))',
    "        rows = _pc_collections.defaultdict(list)",
    "        for i, u in enumerate(user_ids):",
    "            rows[u].append(i)",
    "        labels = list(labels)",
    "        for _u, idxs in sorted(rows.items(), key=lambda kv: str(kv[0])):",
    "            vals = [labels[i] for i in idxs]",
    "            rng.shuffle(vals)",
    "            for i, v in zip(idxs, vals):",
    "                labels[i] = v",
    "        return _pc_real_evaluate(user_ids, labels, scores, k=k)",
    "",
])


def _prepare_workdir(code_path: str, root: str,
                     permute_labels: bool = False) -> tuple[str, str]:
    """Create an isolated workdir with the root modules (test-named files omitted)
    and the candidate. Returns (workdir, candidate_basename)."""
    workdir = tempfile.mkdtemp(prefix="codegen_exec_")
    for name in _ROOT_FILES:
        if "test" in name.lower():
            continue  # never expose a test-named file
        src = os.path.join(root, name)
        if os.path.exists(src):
            shutil.copy2(src, os.path.join(workdir, name))
    base = os.path.basename(code_path)
    if "test" in base.lower():
        base = "candidate_" + base.replace("test", "t_st")
    shutil.copy2(code_path, os.path.join(workdir, base))
    # Belt-and-braces: remove anything test-named that slipped in.
    for f in os.listdir(workdir):
        if "test" in f.lower():
            os.remove(os.path.join(workdir, f))
    if permute_labels:
        ev = os.path.join(workdir, "evaluate.py")
        if os.path.exists(ev):
            with open(ev, "a", encoding="utf-8") as fh:
                fh.write("\n\n" + _PERMUTE_SHIM)
    return workdir, base


def _preexec_limits(cpu_seconds: int):
    """Best-effort resource caps (POSIX). Returns a callable or None."""
    try:
        import resource
    except Exception:
        return None

    def _apply():
        os.setsid()  # own process group so we can kill children too
        try:
            resource.setrlimit(resource.RLIMIT_CPU, (cpu_seconds, cpu_seconds + 2))
        except Exception:
            pass
        try:
            gb = 6 * 1024 ** 3
            resource.setrlimit(resource.RLIMIT_AS, (gb, gb))
        except Exception:
            pass
    return _apply


def _parse_metrics(text: str) -> dict:
    """Prefer an explicit JSON marker line; else parse baseline-style metric words
    (last occurrence wins)."""
    metrics: dict = {}
    for line in reversed(text.splitlines()):
        if _METRICS_MARK in line:
            try:
                metrics = json.loads(line.split(_METRICS_MARK, 1)[1].strip())
                return metrics if isinstance(metrics, dict) else {"value": metrics}
            except Exception:
                continue
    for key, pat in (("primary", r"primary\s+([-+0-9.eE]+)"),
                     ("GAUC", r"GAUC\s+([-+0-9.eE]+)"),
                     ("nDCG@5", r"nDCG@5\s+([-+0-9.eE]+)")):
        m = re.findall(pat, text)
        if m:
            try:
                metrics[key] = float(m[-1])
            except ValueError:
                pass
    return metrics


def _has_divergence(text: str, metrics: dict) -> bool:
    if _DIVERGENCE_RE.search(text):
        return True
    for v in metrics.values():
        try:
            f = float(v)
            if f != f or f in (float("inf"), float("-inf")):
                return True
        except (TypeError, ValueError):
            continue
    return False


def execute(code_path: str, seed: int, split: str, wallclock_cap_seconds: int,
            *, data_dir: str | None = None, root: str = ".",
            permute_labels: bool = False, permute_seed: int = 0) -> dict:
    """Run `code_path` in isolation and report status + parsed metrics + logs.

    Parameters
    ----------
    code_path : str            path to a runnable candidate .py
    seed : int                 seed passed to the candidate (and PYTHONHASHSEED)
    split : str                which split the candidate should score (via CODEGEN_SPLIT)
    wallclock_cap_seconds : int hard timeout; the process group is killed on expiry
    data_dir : str, optional   dataset dir (default env CODEGEN_DATA_DIR or
                               <root>/KuaiRand-Pure/data)
    root : str, optional       repo root holding the unchanged modules
    permute_labels : bool      run the LABEL-PERMUTATION CONTROL: shuffle
                               each user's labels among that user's own rows
                               before scoring. A clean candidate collapses to
                               chance; one reading the label keeps its score.

    Returns
    -------
    {"status": "ok"|"error"|"timeout"|"diverged", "metrics": dict, "logs": str}
    """
    if not os.path.exists(code_path):
        return {"status": "error", "metrics": {},
                "logs": f"candidate not found: {code_path}"}
    data_dir = data_dir or os.environ.get(
        "CODEGEN_DATA_DIR", os.path.join(root, "KuaiRand-Pure", "data"))
    workdir, base = _prepare_workdir(code_path, root,
                                     permute_labels=permute_labels)

    env = {
        "PATH": os.environ.get("PATH", ""),
        "PYTHONHASHSEED": str(seed),
        "CODEGEN_SEED": str(seed),
        "CODEGEN_SPLIT": str(split),
        "CODEGEN_DATA_DIR": os.path.abspath(data_dir),
        "OMP_NUM_THREADS": "1", "OPENBLAS_NUM_THREADS": "1", "MKL_NUM_THREADS": "1",
    }
    if permute_labels:
        # The CONTROL run. A candidate whose score survives this is
        # reading the label; see _PERMUTE_SHIM.
        env["CODEGEN_PERMUTE_LABELS"] = "1"
        env["CODEGEN_PERMUTE_SEED"] = str(permute_seed)
    if "SYSTEMROOT" in os.environ:            # windows needs it
        env["SYSTEMROOT"] = os.environ["SYSTEMROOT"]
    # Forwarded, not defaulted: baseline.py owns the default
    # (ORCHESTRATOR_DEFAULT_MODEL). env is built from scratch above, so without
    # this an FM-vs-GBDT A/B set in the parent shell would be silently dropped.
    if "CODEGEN_MODEL" in os.environ:
        env["CODEGEN_MODEL"] = os.environ["CODEGEN_MODEL"]

    cmd = [sys.executable, base, "--data_dir", os.path.abspath(data_dir), "--seed", str(seed)]
    try:
        proc = subprocess.run(
            cmd, cwd=workdir, env=env, capture_output=True, text=True,
            timeout=wallclock_cap_seconds,
            preexec_fn=_preexec_limits(wallclock_cap_seconds) if os.name == "posix" else None,
        )
        logs = (proc.stdout or "") + ("\n[stderr]\n" + proc.stderr if proc.stderr else "")
        metrics = _parse_metrics(proc.stdout + "\n" + (proc.stderr or ""))
        if _has_divergence(logs, metrics):
            status = "diverged"
        elif proc.returncode != 0:
            status = "error"
        elif metrics:
            status = "ok"
        else:
            status = "error"
            logs += "\n[codegen] process exited 0 but no metrics were parsed."
        return {"status": status, "metrics": metrics, "logs": logs[-_LOG_TAIL:]}
    except subprocess.TimeoutExpired as e:
        # subprocess.run has already sent SIGKILL to the child on timeout; the
        # setsid preexec put it in its own group so no stray children linger for
        # the single-process numpy candidates we run.
        out = (e.stdout or "") if isinstance(e.stdout, str) else ""
        err = (e.stderr or "") if isinstance(e.stderr, str) else ""
        return {"status": "timeout", "metrics": {},
                "logs": (out + err)[-_LOG_TAIL:] +
                        f"\n[codegen] wall-clock cap of {wallclock_cap_seconds}s exceeded."}
    finally:
        shutil.rmtree(workdir, ignore_errors=True)
