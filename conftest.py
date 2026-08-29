"""Ensure the repo root is importable (so `import codegen` and the starter-kit
modules `data`, `submit`, ... resolve) and force the offline fake LLM backend for
all tests, regardless of any ambient ANTHROPIC_API_KEY."""
import os, sys

os.environ["CODEGEN_LLM_BACKEND"] = "fake"
ROOT = os.path.dirname(os.path.abspath(__file__))
if ROOT not in sys.path:
    sys.path.insert(0, ROOT)
