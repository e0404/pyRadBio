"""Where example and validation scripts put their figures, and how they save them.

Two small, **matplotlib-free** helpers shared by ``examples/`` and
``validation/``: this module imports nothing from matplotlib, so it needs no
``plot`` extra and no library module depends on a plotting backend. A figure is
anything with a matplotlib-style ``savefig(fname, **kwargs)`` method (duck
typing).

They replace the block every script carried (e.g. ``examples/survival_and_rbe.py:34-39``,
``examples/dose_protraction_and_bed.py:42-46`` and ``:211-213``):

```text
try:
    HERE = Path(__file__).resolve().parent
except NameError:  # inside a notebook, where __file__ is undefined
    HERE = Path.cwd()
FIGURES = HERE / "figures"
FIGURES.mkdir(exist_ok=True)
...
fig.savefig(out, dpi=130)
print(f"wrote {out}")
```

with ``figures_dir(globals().get("__file__"))`` and ``save_figure(fig, name, dpi=130)``.
"""

from __future__ import annotations

import os
from pathlib import Path
from typing import Any, Protocol

__all__ = ["FIGURES_DIR_ENV", "SavesFigure", "figures_dir", "save_figure"]

FIGURES_DIR_ENV = "PYRADBIO_FIGURES_DIR"
"""Environment variable that, when set (non-empty), overrides every script's
figure folder -- the smoke tests point it at a temporary directory.
"""


class SavesFigure(Protocol):
    """Anything with a matplotlib-``Figure``-style ``savefig`` method."""

    def savefig(self, fname: Any, **kwargs: Any) -> Any:  # pragma: no cover - protocol
        """Write the figure to ``fname``."""
        ...


def figures_dir(script_file: str | None = None) -> Path:
    """Folder a script writes its figures to, created if missing.

    Resolution order:

    1. ``$PYRADBIO_FIGURES_DIR`` ([`FIGURES_DIR_ENV`][]) when set and non-empty;
    2. ``<folder of script_file>/figures`` when ``script_file`` is given;
    3. ``<current working directory>/figures`` otherwise (a notebook kernel, where
       ``__file__`` is undefined -- callers pass ``globals().get("__file__")``).

    Parameters
    ----------
    script_file:
        The calling script's ``__file__``, or ``None``.

    Returns
    -------
    Path
        The resolved folder; ``mkdir(parents=True, exist_ok=True)`` has been run.
    """
    override = os.environ.get(FIGURES_DIR_ENV)
    if override:
        directory = Path(override).expanduser()
    elif script_file is not None:
        directory = Path(script_file).resolve().parent / "figures"
    else:
        directory = Path.cwd() / "figures"
    directory.mkdir(parents=True, exist_ok=True)
    return directory


def save_figure(
    fig: SavesFigure,
    name: str,
    *,
    directory: Path | None = None,
    dpi: int = 150,
    **savefig_kw: Any,
) -> Path:
    """Save ``fig`` as ``<directory>/<name>``, print ``wrote <path>``, return the path.

    Parameters
    ----------
    fig:
        A figure with a ``savefig`` method (a ``matplotlib.figure.Figure``).
    name:
        File name including the extension (``"survival_and_rbe.png"``).
    directory:
        Target folder; ``None`` uses [`figures_dir`][] with no script file
        (environment variable, else ``cwd/figures``). Scripts pass
        ``figures_dir(globals().get("__file__"))`` to keep their own folder.
    dpi:
        Forwarded to ``savefig``. Call sites pass the dpi they used before.
    **savefig_kw:
        Further ``savefig`` keywords (e.g. ``bbox_inches="tight"``).

    Returns
    -------
    Path
        The written file.
    """
    folder = figures_dir() if directory is None else directory
    path = folder / name
    fig.savefig(path, dpi=dpi, **savefig_kw)
    print(f"wrote {path}")
    return path
