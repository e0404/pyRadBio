"""Script plumbing shared by the ``validation/`` scripts (never shipped in the wheel).

Script *policy* lives here rather than in ``pyradbio``: quick mode, the
missing-data / missing-extra exits and two plot idioms. A validation script
imports it as a sibling module (``from _common import pick, require_data``):
``python validation/x.py`` puts ``validation/`` on ``sys.path``, a notebook kernel
started in ``validation/`` has it as its working directory, and the smoke harness
(``tests/validation/test_validation_smoke.py``) prepends it explicitly.

Quick mode
    ``PYRADBIO_QUICK=1`` asks a script for a cheap run (smaller sample counts,
    fewer grid points) that still writes the same figure names. Full mode is the
    default and the only mode whose numbers are compared or quoted. Every cost
    knob is written ``pick(full, quick)``.

Exits
    :class:`MissingDataExit` and :class:`MissingExtraExit` subclass
    :class:`SystemExit`, so a script run from the command line stops with the
    message (exit status 1) and no traceback, while the smoke harness can tell
    "this machine lacks the data / extra" (a skip) from a real failure.

    The messages:

    - data: ``"<label> not available ({exc}).\\nSet <ENV_VAR> (see pyradbio.datasets
      / .env.example) to run this script."`` (e.g. ``"TRiP98 reference data not
      available ({exc}).\\nSet PYRADBIO_TRIP98_DIR (see pyradbio.datasets /
      .env.example) to run this script."``);
    - extras: ``"pyMKM not installed ([mkm] extra): pip install -e '.[mkm]'"``.

    Dataset labels are in ``_DATA_LABELS``.
"""

from __future__ import annotations

import importlib
import os
from collections.abc import Callable, Mapping, Sequence
from types import ModuleType
from typing import Any, TypeVar

from pyradbio.datasets import Dataset, MissingDatasetError

__all__ = [
    "QUICK_ENV",
    "MissingDataExit",
    "MissingExtraExit",
    "pick",
    "plot_reference",
    "proxy_legend",
    "quick_mode",
    "require_data",
    "require_extra",
]

T = TypeVar("T")

#: Environment variable that switches a validation script to its cheap settings.
QUICK_ENV = "PYRADBIO_QUICK"

_FALSY = frozenset({"", "0", "false", "no", "off"})

#: How the exit message names each dataset (``Dataset.key`` -> label). A dataset
#: not listed here is called ``"<key> reference data"``.
_DATA_LABELS: dict[str, str] = {
    "trip98": "TRiP98 reference data",
    "survival": "Survival reference data",
    "survival_src": "Survival source checkout",
    "pide": "PIDE data",
    "radphysbio": "RadPhysBio data",
    "lineal_energy_decunha": "DeCunha 2025 lineal-energy library",
}

#: How the exit message names an optional module (import name -> display name).
_MODULE_LABELS: dict[str, str] = {"pymkm": "pyMKM", "dotenv": "python-dotenv"}


# --- quick mode -----------------------------------------------------------------


def quick_mode() -> bool:
    """``True`` when ``$PYRADBIO_QUICK`` is set to a truthy value.

    Unset, empty, ``0``, ``false``, ``no`` and ``off`` (any case, surrounding
    whitespace ignored) mean full mode; anything else means quick mode.
    """
    return os.environ.get(QUICK_ENV, "").strip().lower() not in _FALSY


def pick(full: T, quick: T) -> T:
    """``quick`` in quick mode, otherwise ``full`` (read at call time)."""
    return quick if quick_mode() else full


# --- data and extra gates -------------------------------------------------------


class MissingDataExit(SystemExit):
    """A script's external dataset is not configured; the run stops cleanly."""


class MissingExtraExit(SystemExit):
    """A script's optional dependency (an install extra) is missing."""


def require_data(loader: Callable[..., T], *args: Any, dataset: Dataset, **kw: Any) -> T:
    """Call ``loader(*args, **kw)``, turning missing external data into a clean exit.

    Parameters
    ----------
    loader:
        A dataset loader, e.g. ``pyradbio.benchmark.reference_tables.load_trip_lem1``.
    *args, **kw:
        Forwarded to ``loader``.
    dataset:
        The :class:`~pyradbio.datasets.Dataset` the loader reads; it names the
        environment variable in the message.

    Raises
    ------
    MissingDataExit
        When the loader raises :class:`~pyradbio.datasets.MissingDatasetError`
        or :class:`FileNotFoundError` (the directory is configured but the file
        is missing). Other errors propagate unchanged.
    """
    try:
        return loader(*args, **kw)
    except (MissingDatasetError, FileNotFoundError) as exc:
        label = _DATA_LABELS.get(dataset.key, f"{dataset.key} reference data")
        raise MissingDataExit(
            f"{label} not available ({exc}).\n"
            f"Set {dataset.env_var} (see pyradbio.datasets / .env.example) to run this script."
        ) from exc


def require_extra(module: str, extra: str) -> ModuleType:
    """Import ``module`` or exit naming the install extra that provides it.

    Parameters
    ----------
    module:
        Import name, e.g. ``"pymkm"`` or ``"numba"``.
    extra:
        The ``pyproject.toml`` optional-dependency group, e.g. ``"mkm"``.

    Returns
    -------
    ModuleType
        The imported module.

    Raises
    ------
    MissingExtraExit
        When the import fails.
    """
    try:
        return importlib.import_module(module)
    except ImportError as exc:
        name = _MODULE_LABELS.get(module, module)
        raise MissingExtraExit(
            f"{name} not installed ([{extra}] extra): pip install -e '.[{extra}]'"
        ) from exc


# --- plot idioms ----------------------------------------------------------------


def proxy_legend(ax: Any, entries: Sequence[Mapping[str, Any]], **legend_kw: Any) -> Any:
    """Add a legend built from proxy handles, one per entry, and return it.

    Each entry holds ``matplotlib.lines.Line2D`` keywords (``label``, ``color``,
    ``ls``, ``marker``, ...). The proxy is ``Line2D([0], [0], **entry)``, the form
    the validation scripts use.

    Every entry must give an explicit ``color`` (or its ``Line2D`` alias ``c``):
    a ``Line2D`` proxy does not take a colour from the axes' property cycle, so a
    colour-less proxy would *not* match an ``ax.plot([], [], ...)`` proxy it
    replaced -- those consume the cycle and shift the colours of later lines.
    Colour-less proxies stay inline in the script, e.g. the open-marker
    ``Line2D(..., ls="None", mfc="none", mec="0.3")``, which sets only face/edge
    colours.

    Raises
    ------
    ValueError
        When an entry has neither ``color`` nor ``c``.
    """
    from matplotlib.lines import Line2D

    for i, entry in enumerate(entries):
        if "color" not in entry and "c" not in entry:
            raise ValueError(
                f"proxy_legend entry {i} ({entry.get('label')!r}) needs an explicit 'color'; "
                "keep colour-less proxies inline as ax.plot([], [], ...)"
            )
    handles = [Line2D([0], [0], **entry) for entry in entries]
    return ax.legend(handles=handles, **legend_kw)


def plot_reference(ax: Any, x: Any, y: Any, *, y_err: Any | None = None, **style: Any) -> Any:
    """Draw reference data: ``ax.plot`` without errors, ``ax.errorbar`` with them.

    ``style`` is passed through unchanged, so each call site keeps its own
    markers and colours. Returns what the underlying matplotlib call returns
    (a list of lines, or an ``ErrorbarContainer``).
    """
    if y_err is None:
        return ax.plot(x, y, **style)
    return ax.errorbar(x, y, yerr=y_err, **style)
