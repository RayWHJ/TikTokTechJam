"""Append-only audit log for accesses to the sealed splits.

Why this is separate from the in-process one-shot flag in _split.py: that flag is
module state, so it resets in every subprocess — and codegen.execute() runs each
candidate as a subprocess by design. A subprocess therefore *can* pull the test
split even though the parent already did.

We can't block that from inside a library, but we can make it undeniable. Every
get_split("test") and get_split("valid_confirm") call appends one JSON line here,
with the timestamp, pid, and calling stack. At the end of the run this file is the
evidence that test was touched exactly once — which is a deliverable in its own
right, not just a safety net.
"""
import json
import os
import time
import traceback

DEFAULT_LOG_PATH = os.environ.get(
    'HARNESS_AUDIT_LOG', os.path.join('.harness_cache', 'sealed_access.jsonl')
)

# Splits whose access we record. train and valid_search are unrestricted.
AUDITED_SPLITS = ('valid_confirm', 'test')


def record_access(split_name, log_path=None):
    """Append one JSON line describing this access. Never raises.

    A failure to write must not break a run — the caller has already been
    granted the data by the time we get here, and losing an audit line is
    strictly better than crashing the harness. Failures are surfaced through the
    'harness.audit' logger instead.

    Args:
        split_name: the split being accessed.
        log_path: override for the audit log location; defaults to
            $HARNESS_AUDIT_LOG or .harness_cache/sealed_access.jsonl.

    Returns:
        The path written to, or None if the write failed.
    """
    path = log_path or DEFAULT_LOG_PATH
    entry = {
        'split': split_name,
        'timestamp': time.strftime('%Y-%m-%dT%H:%M:%S'),
        'epoch': time.time(),
        'pid': os.getpid(),
        # [:-2] drops record_access and its caller in _split, leaving the stack
        # that actually asked for the split.
        'stack': [line.rstrip() for line in traceback.format_stack()[:-2]],
    }
    try:
        parent = os.path.dirname(path)
        if parent:
            os.makedirs(parent, exist_ok=True)
        with open(path, 'a', encoding='utf-8') as fh:
            fh.write(json.dumps(entry) + '\n')
        return path
    except OSError:
        import logging
        logging.getLogger('harness.audit').warning(
            'could not write sealed-split audit entry to %s', path
        )
        return None


def read_accesses(log_path=None):
    """Return the recorded access entries, oldest first; [] if no log exists.

    Use this to answer "how many times was test pulled across the whole run,
    including inside subprocesses?" before designating a final submission.
    """
    path = log_path or DEFAULT_LOG_PATH
    entries = []
    try:
        with open(path, encoding='utf-8') as fh:
            for line in fh:
                line = line.strip()
                if line:
                    try:
                        entries.append(json.loads(line))
                    except ValueError:
                        continue  # a torn line from a concurrent write
    except OSError:
        return []
    return entries


def count_accesses(split_name, log_path=None):
    """How many times split_name has been accessed, across all processes."""
    return sum(1 for e in read_accesses(log_path) if e.get('split') == split_name)
