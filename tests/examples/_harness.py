"""Run a figure-writing script under ``runpy`` and record the PNGs it saves.

Shared by the ``examples/`` and ``validation/`` smoke harnesses.
"""

from __future__ import annotations

import os
import runpy
import sys
import time
from dataclasses import dataclass
from pathlib import Path
from typing import Any

import pytest

from pyradbio.plotting import FIGURES_DIR_ENV

# Wall-time ceiling for one example (120 s).
EXAMPLE_BUDGET_S = 120.0


@dataclass(frozen=True)
class ScriptRun:
    """What one script run produced."""

    pngs: tuple[Path, ...]
    elapsed_s: float


def coverage_active() -> bool:
    """``True`` when a coverage tracer is running in this process."""
    coverage = sys.modules.get("coverage")
    if coverage is None:
        return False
    current = getattr(getattr(coverage, "Coverage", None), "current", None)
    return current is not None and current() is not None


def duration_guard_applies() -> bool:
    """The runtime budget is enforced only locally, without coverage."""
    return not coverage_active() and not os.environ.get("CI")


def _png_target(fname: Any, kwargs: dict[str, Any]) -> Path | None:
    """The PNG file a ``Figure.savefig(fname, **kwargs)`` call wrote, if any.

    Mirrors matplotlib's own format resolution (``FigureCanvasBase.print_figure``):
    an explicit ``format`` wins, else the file suffix, else
    ``rcParams["savefig.format"]``, in which case matplotlib appends that
    extension to the name. A file-like target (``io.BytesIO``) is never a file
    on disk, so it is not recorded.
    """
    if not isinstance(fname, (str, os.PathLike)):
        return None
    name = os.fspath(fname)
    if not isinstance(name, str):  # a bytes path
        return None
    fmt = kwargs.get("format")
    if fmt is None:
        fmt = os.path.splitext(name)[1][1:]
        if not fmt:
            import matplotlib

            fmt = matplotlib.rcParams["savefig.format"]
            name = f"{name.rstrip('.')}.{fmt}"
    return Path(name) if str(fmt).lower() == "png" else None


def run_script(path: Path, monkeypatch: pytest.MonkeyPatch, figures: Path) -> ScriptRun:
    """Execute ``path`` as ``__main__`` with the Agg backend; record saved PNGs.

    ``PYRADBIO_FIGURES_DIR`` points at ``figures`` for scripts that use
    ``pyradbio.plotting``; scripts that still compute their own folder are
    covered by recording every ``Figure.savefig`` call. All figures are closed
    and the rcParams restored afterwards.
    """
    matplotlib = pytest.importorskip("matplotlib")
    monkeypatch.setenv("MPLBACKEND", "Agg")
    monkeypatch.setenv(FIGURES_DIR_ENV, str(figures))
    monkeypatch.setattr(sys, "argv", [str(path)])
    matplotlib.use("Agg", force=True)
    import matplotlib.figure
    import matplotlib.pyplot as plt

    written: list[Path] = []
    original = matplotlib.figure.Figure.savefig

    def recording_savefig(self: Any, fname: Any, *args: Any, **kwargs: Any) -> Any:
        result = original(self, fname, *args, **kwargs)
        target = _png_target(fname, kwargs)
        if target is not None:
            written.append(target)
        return result

    monkeypatch.setattr(matplotlib.figure.Figure, "savefig", recording_savefig)
    start = time.perf_counter()
    try:
        with matplotlib.rc_context():
            runpy.run_path(str(path), run_name="__main__")
    finally:
        elapsed = time.perf_counter() - start
        plt.close("all")
    return ScriptRun(tuple(written), elapsed)


# The ``validation/_common.py`` exits that mean "not runnable here", by class
# name (robust to ``_common`` being imported under another module name).
_SKIP_EXITS = frozenset({"MissingDataExit", "MissingExtraExit"})


def run_validation_script(
    path: Path, monkeypatch: pytest.MonkeyPatch, figures: Path, validation_dir: Path
) -> ScriptRun:
    """`run_script` in quick mode with ``validation_dir`` importable.

    A ``MissingDataExit`` / ``MissingExtraExit`` becomes ``pytest.skip`` (this
    machine lacks the external data or the optional extra); any other
    ``SystemExit`` propagates and fails the test.
    """
    monkeypatch.syspath_prepend(str(validation_dir))
    monkeypatch.setenv("PYRADBIO_QUICK", "1")
    try:
        return run_script(path, monkeypatch, figures)
    except SystemExit as exc:
        if type(exc).__name__ in _SKIP_EXITS:
            pytest.skip(str(exc))
        raise


def assert_wrote_png(run: ScriptRun, path: Path) -> None:
    """At least one PNG was saved, and every recorded PNG exists on disk."""
    assert run.pngs, f"{path.name} saved no PNG figure"
    missing = [str(p) for p in run.pngs if not p.is_file()]
    assert not missing, f"{path.name} recorded PNGs that do not exist: {missing}"
