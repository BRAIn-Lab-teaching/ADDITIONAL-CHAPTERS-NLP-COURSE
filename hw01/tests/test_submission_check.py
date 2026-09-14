import json
import sys
from pathlib import Path

import check_submission as submission
import pytest

from check_submission import file_sha256, last_nonempty_line, report_marker_count


def configure_workspace(
    monkeypatch: pytest.MonkeyPatch,
    tmp_path: Path,
    report: str = 'complete report\n',
) -> Path:
    root = tmp_path / 'hw01'
    artifacts = root / 'artifacts'
    course_lm = root / 'course_lm'
    artifacts.mkdir(parents=True)
    course_lm.mkdir()
    (course_lm / '__init__.py').write_text('', encoding='utf-8')
    (course_lm / 'answer.py').write_text('ANSWER = 42\n', encoding='utf-8')
    (root / 'diagnostics.py').write_text('# diagnostics\n', encoding='utf-8')
    (root / 'check_submission.py').write_text('# checker\n', encoding='utf-8')
    (root / 'report.md').write_text(report, encoding='utf-8')
    monkeypatch.setattr(submission, 'ROOT', root)
    monkeypatch.setattr(submission, 'ARTIFACTS', artifacts)
    monkeypatch.setattr(submission, 'MANIFEST', artifacts / 'submission_check.json')
    return root


def successful_logged_run(command: list[str], log_path: Path) -> dict[str, object]:
    log_path.write_text('ok\n', encoding='utf-8')
    if log_path.name == 'diagnostics.txt':
        output_dir = log_path.parent
        (output_dir / 'metrics.json').write_text('{}\n', encoding='utf-8')
        (output_dir / 'yarn_frequencies.png').write_bytes(b'yarn')
    return {'command': command, 'returncode': 0, 'summary': 'ok'}


def test_report_marker_count_finds_only_exact_unfinished_fields(tmp_path: Path) -> None:
    report = tmp_path / 'report.md'
    report.write_text(
        'done\n<!-- FILL -->\nexample: `<!-- FILL -->`\n'
        'not <!-- FILLING -->\n<!-- FILL -->\n',
        encoding='utf-8',
    )
    assert report_marker_count(report) == 2


def test_file_sha256_matches_a_known_digest(tmp_path: Path) -> None:
    path = tmp_path / 'answer.txt'
    path.write_bytes(b'course-lm\n')
    assert (
        file_sha256(path)
        == 'fa5727180d3d65b98d1807e287fb830e4f6df8c57638ed6804f5f0a51a356baa'
    )


def test_last_nonempty_line_ignores_trailing_whitespace() -> None:
    assert last_nonempty_line('first\n\n  74 passed in 1.2s  \n') == '74 passed in 1.2s'
    assert last_nonempty_line('\n\t\n') == ''


def test_run_logged_uses_stdout_for_summary_when_stderr_has_warning(
    tmp_path: Path,
) -> None:
    result = submission.run_logged(
        [
            sys.executable,
            '-c',
            "import sys; print('completed'); print('warning', file=sys.stderr)",
        ],
        tmp_path / 'command.txt',
    )

    assert result['returncode'] == 0
    assert result['summary'] == 'completed'
    assert (tmp_path / 'command.txt').read_text(encoding='utf-8') == (
        'completed\nwarning\n'
    )


def test_main_removes_stale_manifest_before_rejecting_report(
    monkeypatch: pytest.MonkeyPatch,
    tmp_path: Path,
) -> None:
    configure_workspace(monkeypatch, tmp_path, report='unfinished <!-- FILL -->\n')
    submission.MANIFEST.write_text('stale\n', encoding='utf-8')

    assert submission.main() == 1
    assert not submission.MANIFEST.exists()


def test_main_stops_after_pytest_failure_without_manifest(
    monkeypatch: pytest.MonkeyPatch,
    tmp_path: Path,
) -> None:
    configure_workspace(monkeypatch, tmp_path)
    calls: list[str] = []

    def failing_pytest(command: list[str], log_path: Path) -> dict[str, object]:
        calls.append(log_path.name)
        log_path.write_text('failed\n', encoding='utf-8')
        return {'command': command, 'returncode': 7, 'summary': 'failed'}

    monkeypatch.setattr(submission, 'run_logged', failing_pytest)

    assert submission.main() == 7
    assert calls == ['pytest.txt']
    assert not submission.MANIFEST.exists()


def test_main_stops_after_diagnostics_failure_without_manifest(
    monkeypatch: pytest.MonkeyPatch,
    tmp_path: Path,
) -> None:
    configure_workspace(monkeypatch, tmp_path)
    calls: list[str] = []

    def failing_diagnostics(command: list[str], log_path: Path) -> dict[str, object]:
        calls.append(log_path.name)
        log_path.write_text('ok\n' if len(calls) == 1 else 'failed\n', encoding='utf-8')
        return {
            'command': command,
            'returncode': 0 if len(calls) == 1 else 9,
            'summary': 'ok' if len(calls) == 1 else 'failed',
        }

    monkeypatch.setattr(submission, 'run_logged', failing_diagnostics)

    assert submission.main() == 9
    assert calls == ['pytest.txt', 'diagnostics.txt']
    assert not submission.MANIFEST.exists()


def test_main_writes_manifest_only_after_all_checks_succeed(
    monkeypatch: pytest.MonkeyPatch,
    tmp_path: Path,
) -> None:
    root = configure_workspace(monkeypatch, tmp_path)
    submission.MANIFEST.write_text('stale\n', encoding='utf-8')
    monkeypatch.setattr(submission, 'run_logged', successful_logged_run)

    assert submission.main() == 0

    manifest = json.loads(submission.MANIFEST.read_text(encoding='utf-8'))
    assert manifest['schema_version'] == 1
    assert manifest['pytest']['returncode'] == 0
    assert manifest['diagnostics']['returncode'] == 0
    assert 'environment' in manifest
    assert set(manifest['files']) == {
        'course_lm/__init__.py',
        'course_lm/answer.py',
        'diagnostics.py',
        'check_submission.py',
        'report.md',
        'artifacts/pytest.txt',
        'artifacts/diagnostics.txt',
        'artifacts/metrics.json',
        'artifacts/yarn_frequencies.png',
    }
    for relative_path, digest in manifest['files'].items():
        assert digest == file_sha256(root / relative_path)
