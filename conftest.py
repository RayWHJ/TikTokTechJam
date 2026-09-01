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

import pytest

ROOT = os.path.dirname(os.path.abspath(__file__))
if ROOT not in sys.path:
    sys.path.insert(0, ROOT)

# A: pin dataset location for harness tests
os.environ.setdefault('HARNESS_DATA_DIR', os.path.join(ROOT, 'KuaiRand-Pure', 'data'))

# D: force the offline fake LLM backend so codegen tests never hit a real API
os.environ['CODEGEN_LLM_BACKEND'] = 'fake'

# Markers and the default "not slow" filter live in pytest.ini.


@pytest.fixture(autouse=True)
def _isolate_champion_archive(tmp_path, monkeypatch):
    """Point driver.CHAMPION_DIR at a per-test tmp dir for every test.

    Autouse and repo-wide rather than per-test, because the thing being
    protected is a deliverable. driver.run() archives every new best candidate
    on sight, so the eleven test modules that drive run() against the mocks all
    wrote into the live orchestrator/_state/champions/ — 27 mock champions
    accumulated there, each scored ~0.60 on the mocks' 10-user synthetic split.
    That directory is where a submission is generated from, so a genuine winner
    would have been indistinguishable from test residue.

    Redirecting the module global rather than passing champion_dir= at each of
    the 14 call sites means a test added later is covered without remembering
    to opt in.
    """
    import orchestrator.driver as _driver
    monkeypatch.setattr(_driver, "CHAMPION_DIR",
                        str(tmp_path / "champions"), raising=False)