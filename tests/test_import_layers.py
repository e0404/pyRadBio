"""Layering: the low layers (core, track, micro) never import models or benchmark.

Checked on the source with ``ast`` so function-level (lazy) imports count too --
the three cycles this guards against were all lazy or one-off imports
(``track/srim -> benchmark``, ``track/stopping_power -> models.mkm``,
``micro/sato_coefficients -> benchmark.hartzell2025``). pyMKM may be imported
only by the MKM adapter, and matplotlib and ``pyradbio.plotting`` (itself
matplotlib-free) only by the benchmark report, so
plotting stays in the scripts.
"""

from __future__ import annotations

import ast
from pathlib import Path

import pytest

_PACKAGE = Path(__file__).resolve().parents[1] / "pyradbio"
_LOW_LAYERS = ("core", "track", "micro")
_FORBIDDEN = ("pyradbio.models", "pyradbio.benchmark", "pyradbio.tables", "pyradbio.cli")
#: The one library module that may draw figures (lazily, behind the ``plot`` extra).
#: It is exempt from both bans; whether it actually imports matplotlib is not pinned.
_MATPLOTLIB_IMPORTERS = ("benchmark/report.py",)


def _imports(path: Path) -> list[str]:
    tree = ast.parse(path.read_text(encoding="utf-8"), filename=str(path))
    names: list[str] = []
    for node in ast.walk(tree):
        if isinstance(node, ast.Import):
            names.extend(alias.name for alias in node.names)
        elif isinstance(node, ast.ImportFrom) and node.module and node.level == 0:
            names.append(node.module)
    return names


@pytest.mark.parametrize("layer", _LOW_LAYERS)
def test_low_layer_does_not_import_upper_layers(layer: str) -> None:
    offenders = [
        f"{path.relative_to(_PACKAGE)} -> {name}"
        for path in sorted((_PACKAGE / layer).rglob("*.py"))
        for name in _imports(path)
        if name.startswith(_FORBIDDEN)
    ]
    assert offenders == []


def test_only_the_mkm_adapter_imports_pymkm() -> None:
    importers = sorted(
        str(path.relative_to(_PACKAGE).as_posix())
        for path in _PACKAGE.rglob("*.py")
        if any(name == "pymkm" or name.startswith("pymkm.") for name in _imports(path))
    )
    assert importers == ["models/mkm/pymkm_backend.py"]


def _absolute_imports(source: str, module: str) -> list[str]:
    """Imported module names, relative imports resolved against dotted ``module``.

    ``from a import b`` yields both ``a`` and ``a.b`` (``b`` may be a submodule).
    """
    package = module.split(".")[:-1]
    names: list[str] = []
    for node in ast.walk(ast.parse(source)):
        if isinstance(node, ast.Import):
            names.extend(alias.name for alias in node.names)
        elif isinstance(node, ast.ImportFrom):
            if node.level == 0:
                base = node.module or ""
            else:
                parent = package[: len(package) - (node.level - 1)]
                base = ".".join([*parent, *([node.module] if node.module else [])])
            names.append(base)
            names.extend(f"{base}.{alias.name}" for alias in node.names)
    return names


def _module_imports(path: Path) -> list[str]:
    relative = path.relative_to(_PACKAGE).with_suffix("")
    module = ".".join(("pyradbio", *relative.parts))
    return _absolute_imports(path.read_text(encoding="utf-8"), module)


def _is_plotting_import(name: str) -> bool:
    return any(
        name == root or name.startswith(f"{root}.") for root in ("matplotlib", "pyradbio.plotting")
    )


def test_only_the_benchmark_report_imports_matplotlib_or_plotting() -> None:
    offenders = sorted(
        f"{path.relative_to(_PACKAGE).as_posix()} -> {name}"
        for path in _PACKAGE.rglob("*.py")
        if path.relative_to(_PACKAGE).as_posix() not in _MATPLOTLIB_IMPORTERS
        for name in _module_imports(path)
        if _is_plotting_import(name)
    )
    assert offenders == []


def test_the_plotting_checker_resolves_relative_imports() -> None:
    source = (
        "from .. import plotting\n"
        "from ..plotting import save_figure\n"
        "from . import report\n"
        "import matplotlib.pyplot\n"
    )
    names = _absolute_imports(source, "pyradbio.benchmark.probe")
    assert [n for n in names if _is_plotting_import(n)] == [
        "pyradbio.plotting",
        "pyradbio.plotting",
        "pyradbio.plotting.save_figure",
        "matplotlib.pyplot",
    ]
    assert "pyradbio.benchmark.report" in names
