"""LEM II reconstruction (validation/elsasser2007_lem2.py) full-mode smoke test."""

from __future__ import annotations

from pathlib import Path

import pytest

from tests.examples._harness import run_script

VALIDATION_DIR = Path(__file__).resolve().parents[2] / "validation"


@pytest.mark.slow
def test_elsasser2007_lem2_writes_paper_panels(
    monkeypatch: pytest.MonkeyPatch, tmp_path: Path
) -> None:
    """The LEM II reconstruction runs and writes its Elsasser & Scholz 2007 panels.

    Marked ``slow``: in full mode (``PYRADBIO_QUICK`` unset) the six Fig. 6
    modification curves and the five Fig. 7 stagger variants each build a cluster
    table and sweep a carbon energy grid through the radial-dose integral, so the
    script runs in minutes. The quick-mode run is the generic validation smoke test.
    """
    monkeypatch.delenv("PYRADBIO_QUICK", raising=False)
    monkeypatch.syspath_prepend(str(VALIDATION_DIR))
    run_script(VALIDATION_DIR / "elsasser2007_lem2.py", monkeypatch, tmp_path)

    assert (tmp_path / "lem2_cluster_effect.png").is_file()
