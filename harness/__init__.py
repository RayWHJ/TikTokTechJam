"""harness — wraps data.py / evaluate.py / baseline.py (unmodified) behind the
frozen interface contract: validated_evaluate, get_split, check_provenance.
"""
from ._evaluate import validated_evaluate
from ._split import get_split
from ._provenance import check_provenance, NON_CAUSAL_COLUMNS

__all__ = ['validated_evaluate', 'get_split', 'check_provenance', 'NON_CAUSAL_COLUMNS']
