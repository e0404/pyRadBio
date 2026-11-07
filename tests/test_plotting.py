"""``pyradbio.plotting``: figure-folder resolution and the save-and-report helper."""

from __future__ import annotations

import subprocess
import sys
from pathlib import Path
from typing import Any

import pytest

from pyradbio.plotting import FIGURES_DIR_ENV, figures_dir, save_figure


class _RecordingFigure:
    """Duck-typed stand-in for a matplotlib Figure: records ``savefig`` calls."""

    def __init__(self) -> None:
        self.calls: list[tuple[Any, dict[str, Any]]] = []

    def savefig(self, fname: Any, **kwargs: Any) -> None:
        self.calls.append((fname, kwargs))
        Path(fname).write_bytes(b"png")


def test_figures_dir_prefers_the_environment_variable(
    monkeypatch: pytest.MonkeyPatch, tmp_path: Path
) -> None:
    target = tmp_path / "from_env" / "nested"
    monkeypatch.setenv(FIGURES_DIR_ENV, str(target))

    resolved = figures_dir(str(tmp_path / "script.py"))

    assert resolved == target
    assert resolved.is_dir()


def test_figures_dir_falls_back_to_the_script_folder(
    monkeypatch: pytest.MonkeyPatch, tmp_path: Path
) -> None:
    monkeypatch.delenv(FIGURES_DIR_ENV, raising=False)
    script = tmp_path / "examples" / "demo.py"
    script.parent.mkdir()

    resolved = figures_dir(str(script))

    assert resolved == script.resolve().parent / "figures"
    assert resolved.is_dir()


def test_figures_dir_falls_back_to_the_working_directory(
    monkeypatch: pytest.MonkeyPatch, tmp_path: Path
) -> None:
    monkeypatch.delenv(FIGURES_DIR_ENV, raising=False)
    monkeypatch.chdir(tmp_path)

    resolved = figures_dir()

    assert resolved == Path.cwd() / "figures"
    assert resolved.is_dir()


def test_an_empty_environment_variable_is_ignored(
    monkeypatch: pytest.MonkeyPatch, tmp_path: Path
) -> None:
    monkeypatch.setenv(FIGURES_DIR_ENV, "")
    script = tmp_path / "demo.py"

    assert figures_dir(str(script)) == tmp_path.resolve() / "figures"


def test_save_figure_writes_reports_and_returns_the_path(
    monkeypatch: pytest.MonkeyPatch, tmp_path: Path, capsys: pytest.CaptureFixture[str]
) -> None:
    monkeypatch.delenv(FIGURES_DIR_ENV, raising=False)
    fig = _RecordingFigure()

    path = save_figure(fig, "demo.png", directory=tmp_path, dpi=130, bbox_inches="tight")

    assert path == tmp_path / "demo.png"
    assert path.is_file()
    assert fig.calls == [(path, {"dpi": 130, "bbox_inches": "tight"})]
    assert capsys.readouterr().out == f"wrote {path}\n"


def test_save_figure_defaults_to_150_dpi_and_figures_dir(
    monkeypatch: pytest.MonkeyPatch, tmp_path: Path
) -> None:
    monkeypatch.setenv(FIGURES_DIR_ENV, str(tmp_path / "figs"))
    fig = _RecordingFigure()

    path = save_figure(fig, "demo.png")

    assert path == tmp_path / "figs" / "demo.png"
    assert fig.calls == [(path, {"dpi": 150})]


def test_importing_plotting_does_not_import_matplotlib() -> None:
    """The module is public and matplotlib-free, so it needs no ``plot`` extra."""
    code = (
        "import sys\n"
        "import pyradbio.plotting\n"
        "leaked = sorted(m for m in sys.modules if m.split('.')[0] == 'matplotlib')\n"
        "print(leaked)\n"
        "raise SystemExit(1 if leaked else 0)\n"
    )
    result = subprocess.run(
        [sys.executable, "-c", code], capture_output=True, text=True, check=False
    )
    assert result.returncode == 0, result.stdout + result.stderr
