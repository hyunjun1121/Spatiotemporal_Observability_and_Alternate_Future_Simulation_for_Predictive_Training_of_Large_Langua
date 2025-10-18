from __future__ import annotations

import os
import shutil
import sys
import subprocess
from pathlib import Path

import pytest


@pytest.mark.skipif(shutil.which('bash') is None, reason='bash not available')
def test_run_replicates_dry_run(tmp_path: Path):
    repo_root = Path(__file__).resolve().parent.parent
    runs_root = repo_root / 'tmp_pytest_runs' / tmp_path.name
    paper_root = repo_root / 'tmp_pytest_paper' / tmp_path.name
    runs_root_rel = runs_root.relative_to(repo_root).as_posix()
    paper_root_rel = paper_root.relative_to(repo_root).as_posix()

    cmd = [
        'bash',
        'scripts/run_replicates.sh',
        '--dry-run',
        '--real_data', 'wikitext',
        '--baseline', 'fixedlr',
        '--method', 'C',
        '--config', 'assets/experiments/wikitext_rc1.yaml',
        '--runs_root', runs_root_rel,
        '--paper_dir', paper_root_rel,
    ]

    env = os.environ.copy()
    env['PYTHON'] = Path(sys.executable).as_posix()
    subprocess.run(cmd, check=True, cwd=repo_root, env=env)

    try:
        for seed in (1337, 2337, 3337):
            marker = runs_root / 'wikitext_rc1' / 'fixedlr' / 'C' / f'seed{seed}' / '.dry_run'
            assert marker.exists()
        assert not (paper_root / 'tables' / 'results.tsv').exists()
        assert not (repo_root / 'lock.json').exists()
    finally:
        shutil.rmtree(runs_root.parent, ignore_errors=True)
        shutil.rmtree(paper_root.parent, ignore_errors=True)

