"""
codegen/ — Person D's code-generation & execution layer for the TikTok TechJam
Track 2 autonomous ML research agent.

Public contract (frozen interface v1):
    codegen.write_fix(hypothesis, target_component) -> str            # diff text
    codegen.pre_execution_gate(code_diff) -> {"pass": bool, "reasons": [...]}
    codegen.execute(code_path, seed, split, wallclock_cap_seconds) -> {...}
    codegen.debug_and_retry(code_path, error_context) -> {"code_diff", "is_semantic_change"}
    codegen.check_submission(path, split) -> bool
    codegen.synthesize_report(run_log) -> str                         # markdown

All model-calling functions accept an optional `client=` (see llm_client) and run
offline via a deterministic fake backend unless a real backend is configured, so
the package is fully testable with no API key and no teammate module.
"""
from .writer import write_fix
from .gate import pre_execution_gate
from .sandbox import execute
from .debug import debug_and_retry, is_semantic_change
from .submission import check_submission
from .report import synthesize_report
from .llm_client import LLMClient, FakeBackend, AnthropicBackend, get_default_client
from .constants import (NON_CAUSAL_COLUMNS, AUXILIARY_SIGNALS,
                        ORACLE_PRIMARY_CEILING, FM_BASELINE_TEST_PRIMARY)

__all__ = [
    "write_fix", "pre_execution_gate", "execute", "debug_and_retry",
    "is_semantic_change", "check_submission", "synthesize_report",
    "LLMClient", "FakeBackend", "AnthropicBackend", "get_default_client",
    "NON_CAUSAL_COLUMNS", "AUXILIARY_SIGNALS",
    "ORACLE_PRIMARY_CEILING", "FM_BASELINE_TEST_PRIMARY",
]
