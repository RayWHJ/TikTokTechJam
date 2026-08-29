"""Make the repo root importable regardless of where pytest is invoked from.

pytest puts each test file's own directory on sys.path (rootdir-relative imports
are not automatic without an installed package), so `import harness` / `import
data` would otherwise only resolve when cwd happens to be the repo root. This
also pins the default data directory to an absolute path so a candidate run from
a scratch working directory still finds the dataset.
"""
import os
import sys

ROOT = os.path.dirname(os.path.abspath(__file__))
if ROOT not in sys.path:
    sys.path.insert(0, ROOT)

os.environ.setdefault('HARNESS_DATA_DIR', os.path.join(ROOT, 'KuaiRand-Pure', 'data'))

# Markers and the default "not slow" filter live in pytest.ini.
