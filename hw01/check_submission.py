"""Run the final HW01 checks and record the exact checked files."""

from __future__ import annotations

import hashlib
import json
import os
import subprocess
import sys
from datetime import datetime, timezone
from importlib.metadata import PackageNotFoundError, version
from pathlib import Path


ROOT = Path(__file__).resolve().parent
ARTIFACTS = ROOT / 'artifacts'
MANIFEST = ARTIFACTS / 'submission_check.json'
REPORT_MARKER = '<!-- FILL -->'


def file_sha256(path: Path) -> str:
    """Return the hexadecimal SHA256 digest of one file."""
    digest = hashlib.sha256()
    with path.open('rb') as source:
        for block in iter(lambda: source.read(1024 * 1024), b''):
            digest.update(block)
    return digest.hexdigest()


def last_nonempty_line(text: str) -> str:
    """Return the final nonempty output line, or an empty string."""
    return next(
        (line.strip() for line in reversed(text.splitlines()) if line.strip()), ''
    )


def report_marker_count(path: Path) -> int:
    """Count unfinished fields in the report template."""
    inline_example = f'`{REPORT_MARKER}`'
    return sum(
        line.replace(inline_example, '').count(REPORT_MARKER)
        for line in path.read_text(encoding='utf-8').splitlines()
    )


def package_version(distribution: str) -> str:
    """Return an installed package version without importing the package."""
    try:
        return version(distribution)
    except PackageNotFoundError:
        return 'not installed'


def run_logged(command: list[str], log_path: Path) -> dict[str, object]:
    """Run a command in the homework root and preserve its complete output."""
    environment = os.environ.copy()
    environment.setdefault('MPLBACKEND', 'Agg')
    completed = subprocess.run(
        command,
        cwd=ROOT,
        env=environment,
        capture_output=True,
        text=True,
        check=False,
    )
    output = completed.stdout + completed.stderr
    log_path.write_text(output, encoding='utf-8')
    print(output, end='')
    return {
        'command': command,
        'returncode': completed.returncode,
        'summary': last_nonempty_line(completed.stdout or completed.stderr),
    }


def checked_files() -> list[Path]:
    """Return source, report and generated artifacts covered by the manifest."""
    files = sorted((ROOT / 'course_lm').glob('*.py'))
    files.extend(
        [ROOT / 'diagnostics.py', ROOT / 'check_submission.py', ROOT / 'report.md']
    )
    files.extend(
        ARTIFACTS / name
        for name in (
            'pytest.txt',
            'diagnostics.txt',
            'metrics.json',
            'yarn_frequencies.png',
        )
    )
    return files


def main() -> int:
    """Run all checks; create a manifest only after complete success."""
    ARTIFACTS.mkdir(parents=True, exist_ok=True)
    MANIFEST.unlink(missing_ok=True)

    missing_fields = report_marker_count(ROOT / 'report.md')
    if missing_fields:
        print(f'report.md contains {missing_fields} unfinished <!-- FILL --> markers')
        return 1

    pytest_result = run_logged(
        [sys.executable, '-m', 'pytest', '-q'], ARTIFACTS / 'pytest.txt'
    )
    if pytest_result['returncode'] != 0:
        return int(pytest_result['returncode'])

    diagnostics_result = run_logged(
        [
            sys.executable,
            str(ROOT / 'diagnostics.py'),
            '--output-dir',
            str(ARTIFACTS),
        ],
        ARTIFACTS / 'diagnostics.txt',
    )
    if diagnostics_result['returncode'] != 0:
        return int(diagnostics_result['returncode'])

    files = checked_files()
    missing = [
        path.relative_to(ROOT).as_posix() for path in files if not path.is_file()
    ]
    if missing:
        print('missing required files: ' + ', '.join(missing))
        return 1

    manifest = {
        'schema_version': 1,
        'checked_at_utc': datetime.now(timezone.utc).isoformat(),
        'environment': {
            'python': sys.version.split()[0],
            'torch': package_version('torch'),
        },
        'pytest': pytest_result,
        'diagnostics': diagnostics_result,
        'files': {
            path.relative_to(ROOT).as_posix(): file_sha256(path) for path in files
        },
    }
    MANIFEST.write_text(
        json.dumps(manifest, ensure_ascii=False, indent=2) + '\n', encoding='utf-8'
    )
    print(f'written {MANIFEST.relative_to(ROOT)}')
    return 0


if __name__ == '__main__':
    raise SystemExit(main())
