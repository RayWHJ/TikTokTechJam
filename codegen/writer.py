"""
codegen.write_fix — generate a code diff implementing a hypothesis.

Routing (per the frozen contract): if target_component is about
features / history / auxiliary signals, frame the task as "extend data.py's
feature encoding"; otherwise frame it as "modify baseline.py's model / loss /
training loop." The relevant existing file's content is passed to the model as
context, together with hypothesis['mechanism'] and hypothesis['implementation_sketch'].
"""
from __future__ import annotations
import os, re

from .constants import FEATURE_COMPONENTS
from . import prompts
from .llm_client import LLMClient, get_default_client, KIND_DIFF


def _is_feature_component(target_component: str) -> bool:
    """True -> edit data.py (features); False -> edit baseline.py (model/loss)."""
    tc = (target_component or "").strip().lower()
    if tc in FEATURE_COMPONENTS:
        return True
    # substring signals so callers can pass e.g. "add_user_history_feature"
    return any(tok in tc for tok in
               ("feature", "history", "sequence", "auxiliary", "aux", "encoding",
                "field", "embedding_input"))


def _read_root_file(name: str, root: str) -> str:
    path = os.path.join(root, name)
    with open(path, "r", encoding="utf-8") as fh:
        return fh.read()


def _extract_diff(text: str) -> str:
    """Return the unified diff. Prefer a ```diff fenced block; else the first
    ``` block; else the raw text (already a diff)."""
    m = re.search(r"```diff\s*\n(.*?)```", text, re.DOTALL)
    if m:
        return m.group(1).rstrip("\n") + "\n"
    m = re.search(r"```(?:patch|python)?\s*\n(.*?)```", text, re.DOTALL)
    if m:
        return m.group(1).rstrip("\n") + "\n"
    return text.strip() + "\n"


def write_fix(hypothesis: dict, target_component: str, *,
              client: LLMClient | None = None, root: str = ".") -> str:
    """Generate a code diff/patch (as text) implementing `hypothesis`.

    Parameters
    ----------
    hypothesis : dict
        Must contain 'mechanism' and 'implementation_sketch'
        (as produced by llm.generate_hypothesis). 'success_criterion_paired'
        is used as context only.
    target_component : str
        Routes the prompt. Feature/history/auxiliary -> data.py; anything else
        (model/loss/training/schedule) -> baseline.py.
    client : LLMClient, optional
        Model client. Defaults to the process client (offline fake backend unless
        a real backend is configured). Inject the real one from the orchestrator.
    root : str
        Repo root containing baseline.py / data.py (default: current dir).

    Returns
    -------
    str
        A unified-diff patch as text (the ```diff fence stripped).
    """
    client = client or get_default_client()
    if _is_feature_component(target_component):
        file_name, system = "data.py", prompts.WRITER_SYSTEM_DATA
    else:
        file_name, system = "baseline.py", prompts.WRITER_SYSTEM_MODEL

    content = _read_root_file(file_name, root)
    user = prompts.build_writer_user(file_name, content, hypothesis, target_component)
    raw = client.complete(system, user, kind=KIND_DIFF, max_tokens=4000, temperature=0.0)
    return _extract_diff(raw)
