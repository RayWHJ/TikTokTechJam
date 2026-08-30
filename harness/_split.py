"""get_split() — the one function in this package with a stateful gate: 'test'
may be pulled exactly once per process, to stop anyone from tuning against it.

The one-shot flag is process-local by necessity (module state), so it does not
survive into the subprocesses codegen.execute() spawns. _audit.py closes that
observability gap: every sealed-split access is appended to an on-disk log, so a
second access from anywhere is at least provable after the fact.
"""
import logging
import time
import traceback

from ._audit import AUDITED_SPLITS, count_accesses, record_access
from ._data import get_encoded
from ._sizes import SPLIT_NAMES

logger = logging.getLogger('harness.test_access')

_test_called = False


def get_split(name: str):
    """Return (X, y, user_ids) for one of the frozen splits.

    Args:
        name: one of "train", "valid_search", "valid_confirm", "test".
            valid_search  = dates 20220422-20220426 (5 days), touched every iteration.
            valid_confirm = dates 20220427-20220428 (2 days), SEALED — touch only
                when promoting a new global-best candidate. Not blocked here, per
                the contract, but every access is recorded to the audit log.
            test          = ONE-SHOT. Raises RuntimeError on any 2nd call in this
                process.

    Returns:
        (X, y, user_ids): X is an int32 (N, len(data.FIELDS)) array of offset
        feature ids, y is a float32 label array, user_ids is a list of user_id
        strings.

    Raises:
        ValueError: if name is not one of the 4 recognized split names.
        RuntimeError: if name == "test" and this is not the first call.
    """
    if name not in SPLIT_NAMES:
        raise ValueError(f"unknown split {name!r}, expected one of {SPLIT_NAMES}")

    global _test_called
    if name == 'test':
        if _test_called:
            raise RuntimeError(
                "harness.get_split('test') was already called once in this "
                "process — test is one-shot, this is the second call."
            )
        _test_called = True
        logger.warning(
            "harness.get_split('test') called at %s (audit total across all "
            "processes, including this call: %d)\n%s",
            time.strftime('%Y-%m-%d %H:%M:%S'),
            count_accesses('test') + 1,
            ''.join(traceback.format_stack()[:-1]),
        )

    if name in AUDITED_SPLITS:
        record_access(name)

    return get_encoded()['enc'][name]
