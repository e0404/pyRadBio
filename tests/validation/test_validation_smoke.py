"""Every validation script runs in quick mode and writes a PNG (``@slow``).

Discovery-based: a new ``validation/<name>.py`` is picked up automatically, so
no batch edits this file. The runner (``run_validation_script`` in
``tests/examples/_harness.py``, self-tested in ``tests/examples/test_harness.py``)
sets ``PYRADBIO_QUICK=1``, which selects each script's cheap settings; a script
that stops with ``MissingDataExit`` / ``MissingExtraExit``
(``validation/_common.py``) is skipped, not failed, so a machine without the
external data or the optional extra stays green.

Run it with ``pytest -m slow tests/validation`` (CI: the ``workflow_dispatch``
job ``validation-quick``).
"""

from __future__ import annotations

from pathlib import Path

import pytest

from tests.examples._harness import assert_wrote_png, run_validation_script

pytestmark = pytest.mark.slow

VALIDATION_DIR = Path(__file__).resolve().parents[2] / "validation"
SCRIPTS = sorted(VALIDATION_DIR.glob("[!_]*.py"))

_PARAMS = [pytest.param(p, id=p.stem) for p in SCRIPTS] or [
    pytest.param(None, id="none", marks=pytest.mark.skip(reason="no validation scripts yet"))
]


@pytest.mark.parametrize("path", _PARAMS)
def test_validation_script_runs_in_quick_mode(
    path: Path, monkeypatch: pytest.MonkeyPatch, tmp_path: Path
) -> None:
    run = run_validation_script(path, monkeypatch, tmp_path, VALIDATION_DIR)

    assert_wrote_png(run, path)
