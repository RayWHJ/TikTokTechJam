"""Repo-wide pytest configuration.

Ensure the repo root is importable (so `import harness`, `import codegen`, and the
starter-kit modules `data`, `submit`, ... resolve) regardless of where pytest is
invoked from.

pytest puts each test file's own directory on sys.path (rootdir-relative imports
are not automatic without an installed package), so `import harness` / `import
data` would otherwise only resolve when cwd happens to be the repo root. This
also pins the default data directory to an absolute path so a candidate run from
a scratch working directory still finds the dataset.

Force the offline fake LLM backend for all tests, regardless of any ambient
OPENAI_API_KEY.
"""
import os
import sys

ROOT = os.path.dirname(os.path.abspath(__file__))
if ROOT not in sys.path:
    sys.path.insert(0, ROOT)

# A: pin dataset location for harness tests
os.environ.setdefault('HARNESS_DATA_DIR', os.path.join(ROOT, 'KuaiRand-Pure', 'data'))

# D: force the offline fake LLM backend so codegen tests never hit a real API
os.environ['CODEGEN_LLM_BACKEND'] = 'fake'

# Markers and the default "not slow" filter live in pytest.ini.