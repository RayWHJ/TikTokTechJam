"""harness — wraps data.py / evaluate.py / baseline.py (unmodified) behind the
frozen interface contract: validated_evaluate, get_split, check_provenance.

Beyond the three frozen functions this also exposes two read-only audit helpers,
count_sealed_accesses / read_sealed_accesses, for answering "how many times was
test actually pulled across this whole run, subprocesses included?" before
designating a final submission.
"""
from ._audit import count_accesses as count_sealed_accesses
from ._audit import read_accesses as read_sealed_accesses
from ._evaluate import validated_evaluate
from ._split import get_split
from ._provenance import check_provenance, NON_CAUSAL_COLUMNS

__all__ = [
    'validated_evaluate', 'get_split', 'check_provenance', 'NON_CAUSAL_COLUMNS',
    'count_sealed_accesses', 'read_sealed_accesses',
]
