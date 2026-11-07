"""Self-tests for the smoke-harness logic in ``_harness.py`` (not slow).

The example, validation and scripts harnesses are discovery-based, and the
validation and scripts parametrizations are empty until the first migration
batch. These tests run the shared helpers on throwaway scripts written to
``tmp_path`` so the harness logic itself is exercised today.
"""

from __future__ import annotations

import textwrap
from pathlib import Path

import pytest

from tests.examples._harness import (
    assert_wrote_png,
    duration_guard_applies,
    run_script,
    run_validation_script,
)

VALIDATION_DIR = Path(__file__).resolve().parents[2] / "validation"


def _script(tmp_path: Path, body: str, name: str = "fake_script.py") -> Path:
    path = tmp_path / name
    path.write_text(textwrap.dedent(body), encoding="utf-8")
    return path


_FIGURE = """\
import os
import matplotlib.pyplot as plt

fig, ax = plt.subplots()
ax.plot([0, 1], [0, 1])
out = os.environ["PYRADBIO_FIGURES_DIR"]
"""


def test_a_png_target_is_recorded_and_exists(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    path = _script(tmp_path, _FIGURE + "fig.savefig(os.path.join(out, 'a.png'))\n")
    figures = tmp_path / "figures"
    figures.mkdir()

    run = run_script(path, monkeypatch, figures)

    assert run.pngs == (figures / "a.png",)
    assert_wrote_png(run, path)


def test_an_extensionless_target_is_recorded_under_the_default_format(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    # matplotlib appends rcParams["savefig.format"] ("png") to a suffix-less name.
    path = _script(tmp_path, _FIGURE + "fig.savefig(os.path.join(out, 'plain'))\n")
    figures = tmp_path / "figures"
    figures.mkdir()

    run = run_script(path, monkeypatch, figures)

    assert run.pngs == (figures / "plain.png",)
    assert_wrote_png(run, path)


def test_a_buffer_target_is_not_recorded_and_does_not_crash(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    path = _script(tmp_path, _FIGURE + "import io\nfig.savefig(io.BytesIO(), format='png')\n")

    run = run_script(path, monkeypatch, tmp_path)

    assert run.pngs == ()
    with pytest.raises(AssertionError, match="saved no PNG"):
        assert_wrote_png(run, path)


def test_a_non_png_figure_does_not_count(tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> None:
    path = _script(tmp_path, _FIGURE + "fig.savefig(os.path.join(out, 'a.pdf'))\n")

    run = run_script(path, monkeypatch, tmp_path)

    with pytest.raises(AssertionError, match="saved no PNG"):
        assert_wrote_png(run, path)


def test_run_script_closes_every_figure(tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> None:
    path = _script(tmp_path, _FIGURE + "fig.savefig(os.path.join(out, 'a.png'))\n")
    run_script(path, monkeypatch, tmp_path)

    import matplotlib.pyplot as plt

    assert plt.get_fignums() == []


def test_the_duration_guard_is_off_in_ci(monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.setenv("CI", "true")
    assert duration_guard_applies() is False


# --- validation runner ----------------------------------------------------------


def test_a_missing_extra_is_a_skip(tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> None:
    path = _script(
        tmp_path,
        "from _common import require_extra\nrequire_extra('pyradbio_no_such_module_xyz', 'mkm')\n",
    )

    with pytest.raises(pytest.skip.Exception, match=r"\[mkm\] extra"):
        run_validation_script(path, monkeypatch, tmp_path, VALIDATION_DIR)


def test_missing_data_is_a_skip(tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> None:
    path = _script(
        tmp_path,
        """\
        from _common import require_data
        from pyradbio.datasets import TRIP98, MissingDatasetError

        def loader():
            raise MissingDatasetError(TRIP98, None)

        require_data(loader, dataset=TRIP98)
        """,
    )

    with pytest.raises(pytest.skip.Exception, match="TRiP98 reference data not available"):
        run_validation_script(path, monkeypatch, tmp_path, VALIDATION_DIR)


def test_a_plain_system_exit_is_a_failure(tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> None:
    path = _script(tmp_path, "raise SystemExit('broken')\n")

    with pytest.raises(SystemExit, match="broken"):
        run_validation_script(path, monkeypatch, tmp_path, VALIDATION_DIR)


def test_the_validation_runner_sets_quick_mode(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    monkeypatch.delenv("PYRADBIO_QUICK", raising=False)
    path = _script(
        tmp_path,
        _FIGURE
        + "from _common import pick\n"
        + "fig.savefig(os.path.join(out, pick('full.png', 'quick.png')))\n",
    )

    run = run_validation_script(path, monkeypatch, tmp_path, VALIDATION_DIR)

    assert run.pngs == (tmp_path / "quick.png",)
