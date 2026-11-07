"""``validation/pide_pfuhl2022.py`` on the synthetic PIDE fixture (quick mode; not slow).

The script does not fall back to a ``tests/`` fixture, so the test hands it one:
``PYRADBIO_PIDE_DIR`` points at a ``tmp_path`` copy of the synthetic fixture. It
keeps the CI coverage (the generic validation smoke test is ``@slow``), runs
whether or not this machine has the real PIDE, and never reads it.
"""

from __future__ import annotations

import shutil
from pathlib import Path

import pytest

from tests.examples._harness import assert_wrote_png, run_script

ROOT = Path(__file__).resolve().parents[2]
VALIDATION_DIR = ROOT / "validation"
SCRIPT = VALIDATION_DIR / "pide_pfuhl2022.py"
FIXTURE = ROOT / "tests" / "benchmark" / "fixtures" / "synthetic_pide.xlsx"


def test_pide_validation_runs_on_the_synthetic_fixture(
    monkeypatch: pytest.MonkeyPatch, tmp_path: Path, capsys: pytest.CaptureFixture[str]
) -> None:
    pytest.importorskip("matplotlib")
    pytest.importorskip("openpyxl")  # the fixture is an .xlsx
    pide_dir = tmp_path / "pide"
    pide_dir.mkdir()
    shutil.copy2(FIXTURE, pide_dir / FIXTURE.name)
    monkeypatch.setenv("PYRADBIO_PIDE_DIR", str(pide_dir))
    monkeypatch.setenv("PYRADBIO_QUICK", "1")
    monkeypatch.syspath_prepend(str(VALIDATION_DIR))

    # Not run_validation_script: a missing-data exit must fail here, not skip.
    run = run_script(SCRIPT, monkeypatch, tmp_path / "figures")

    assert_wrote_png(run, SCRIPT)
    assert [p.name for p in run.pngs] == ["pide_model_validation_lem4.png"]
    out = capsys.readouterr().out
    assert "QUICK MODE" in out
    assert f"from {pide_dir / FIXTURE.name}" in out
    assert "Selected per ion:" in out
    assert "Expected (Pfuhl 2022, Table 2, RBE_alpha, LEM IV): H 0.93, He 1.14, C 0.82" in out


def test_pide_validation_stops_cleanly_without_data(
    monkeypatch: pytest.MonkeyPatch, tmp_path: Path
) -> None:
    """An empty PIDE folder is a missing-data exit (a skip in the smoke harness)."""
    pytest.importorskip("matplotlib")
    monkeypatch.setenv("PYRADBIO_PIDE_DIR", str(tmp_path))
    monkeypatch.setenv("PYRADBIO_QUICK", "1")
    monkeypatch.syspath_prepend(str(VALIDATION_DIR))

    with pytest.raises(SystemExit) as excinfo:
        run_script(SCRIPT, monkeypatch, tmp_path / "figures")
    assert type(excinfo.value).__name__ == "MissingDataExit"
    assert "PYRADBIO_PIDE_DIR" in str(excinfo.value)
