"""Examples import only the standard library, numpy, scipy, matplotlib and pyradbio.

CI installs ``[all]``, so a run cannot tell whether an example silently needs
pandas, numba or pyMKM; only a static check can. Function-level imports count
too. The pyMKM adapter (``pyradbio.models.mkm.pymkm_backend``) is never allowed:
its cross-checks belong in ``validation/``.
"""

from __future__ import annotations

import ast
import sys
from pathlib import Path

import pytest

EXAMPLES_DIR = Path(__file__).resolve().parents[2] / "examples"
EXAMPLES = sorted(EXAMPLES_DIR.glob("[!_]*.py"))

_ALLOWED_THIRD_PARTY = frozenset({"numpy", "scipy", "matplotlib", "pyradbio"})
_FORBIDDEN_MODULE = "pyradbio.models.mkm.pymkm_backend"


def _imported_modules(path: Path) -> list[str]:
    """Absolute module names an example imports (``from a import b`` gives ``a.b`` too)."""
    tree = ast.parse(path.read_text(encoding="utf-8"), filename=str(path))
    names: list[str] = []
    for node in ast.walk(tree):
        if isinstance(node, ast.Import):
            names.extend(alias.name for alias in node.names)
        elif isinstance(node, ast.ImportFrom) and node.level == 0 and node.module:
            names.append(node.module)
            names.extend(f"{node.module}.{alias.name}" for alias in node.names)
    return names


def _violations(path: Path) -> list[str]:
    bad: list[str] = []
    for name in _imported_modules(path):
        root = name.split(".")[0]
        allowed = root in sys.stdlib_module_names or root in _ALLOWED_THIRD_PARTY
        forbidden = name == _FORBIDDEN_MODULE or name.startswith(f"{_FORBIDDEN_MODULE}.")
        if not allowed or forbidden:
            bad.append(name)
    return sorted(set(bad))


@pytest.mark.parametrize("path", [pytest.param(p, id=p.stem) for p in EXAMPLES])
def test_example_imports_are_on_the_allowlist(path: Path) -> None:
    assert _violations(path) == []


def test_the_allowlist_rejects_pymkm_and_optional_extras(tmp_path: Path) -> None:
    """The checker itself: stdlib/numpy pass; pyMKM, its adapter and extras fail."""
    script = tmp_path / "demo.py"
    script.write_text(
        "import json\n"
        "import numpy as np\n"
        "from pyradbio.core.lq import survival\n"
        "def f():\n"
        "    import pandas\n"
        "    from pyradbio.models.mkm import pymkm_backend\n"
        "    from pyradbio.models.mkm.pymkm_backend import PyMKMBackend\n"
        "    import pymkm.physics\n",
        encoding="utf-8",
    )
    assert _violations(script) == [
        "pandas",
        "pymkm.physics",
        "pyradbio.models.mkm.pymkm_backend",
        "pyradbio.models.mkm.pymkm_backend.PyMKMBackend",
    ]
