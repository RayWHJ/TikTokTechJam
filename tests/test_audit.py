"""Tests for the sealed-split audit log.

These never touch the real dataset — they exercise _audit directly, plus one
subprocess test proving that a get_split('test') pull from a child process still
lands in the shared log even though the child's one-shot flag started fresh.
"""
import json
import os
import subprocess
import sys

from harness import _audit


def test_record_and_read_roundtrip(tmp_path):
    log = str(tmp_path / 'sealed.jsonl')
    assert _audit.read_accesses(log) == []

    assert _audit.record_access('test', log) == log
    entries = _audit.read_accesses(log)
    assert len(entries) == 1
    assert entries[0]['split'] == 'test'
    assert entries[0]['pid'] == os.getpid()
    assert entries[0]['stack'], 'the calling stack must be captured'


def test_appends_rather_than_overwrites(tmp_path):
    log = str(tmp_path / 'sealed.jsonl')
    _audit.record_access('valid_confirm', log)
    _audit.record_access('valid_confirm', log)
    _audit.record_access('test', log)

    assert _audit.count_accesses('valid_confirm', log) == 2
    assert _audit.count_accesses('test', log) == 1
    assert _audit.count_accesses('train', log) == 0


def test_read_skips_torn_lines(tmp_path):
    log = tmp_path / 'sealed.jsonl'
    log.write_text(
        json.dumps({'split': 'test'}) + '\n'
        + '{"split": "test", trunca\n'          # torn write
        + '\n'                                   # blank line
        + json.dumps({'split': 'valid_confirm'}) + '\n',
        encoding='utf-8',
    )
    entries = _audit.read_accesses(str(log))
    assert [e['split'] for e in entries] == ['test', 'valid_confirm']


def test_missing_log_reads_as_empty(tmp_path):
    assert _audit.read_accesses(str(tmp_path / 'nope.jsonl')) == []
    assert _audit.count_accesses('test', str(tmp_path / 'nope.jsonl')) == 0


def test_write_failure_is_non_fatal(tmp_path):
    # A directory where the log file should be makes open(..., 'a') fail. Losing
    # an audit line must never take down a run that already has the data.
    bad = tmp_path / 'sealed.jsonl'
    bad.mkdir()
    assert _audit.record_access('test', str(bad)) is None


def test_creates_parent_directory(tmp_path):
    log = str(tmp_path / 'deep' / 'nested' / 'sealed.jsonl')
    assert _audit.record_access('test', log) == log
    assert _audit.count_accesses('test', log) == 1


def test_subprocess_test_access_lands_in_shared_log(tmp_path):
    """The one-shot flag resets per process; the audit log must not.

    This is the exact hole codegen.execute() opens by running candidates as
    subprocesses. The child's get_split('test') succeeds — nothing can stop it —
    but the parent can afterwards prove it happened.
    """
    log = str(tmp_path / 'sealed.jsonl')
    env = dict(os.environ, HARNESS_AUDIT_LOG=log)

    # Stub out the encode step: this test is about the gate and the log, not
    # about feature arrays, and a real load+encode would cost ~34s per child.
    code = (
        "import harness, harness._split as s\n"
        "s.get_encoded = lambda data_dir=None: {'enc': {'test': ('X', 'y', 'u')}}\n"
        "assert harness.get_split('test') == ('X', 'y', 'u')\n"
    )
    for _ in range(2):  # two independent processes, each with a fresh flag
        r = subprocess.run(
            [sys.executable, '-c', code], cwd='.', env=env, timeout=300,
            capture_output=True, text=True,
        )
        assert r.returncode == 0, r.stderr

    assert _audit.count_accesses('test', log) == 2, (
        'both subprocess pulls of the sealed test split must be recorded'
    )


def test_valid_confirm_is_audited_but_not_blocked():
    # The contract says valid_confirm is sealed by convention, not by a raise.
    assert 'valid_confirm' in _audit.AUDITED_SPLITS
    assert 'test' in _audit.AUDITED_SPLITS
    assert 'train' not in _audit.AUDITED_SPLITS
    assert 'valid_search' not in _audit.AUDITED_SPLITS
