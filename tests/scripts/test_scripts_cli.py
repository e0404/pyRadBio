"""Offline generators in ``scripts/`` follow the CLI contract (not slow).

Discovery-based over the top-level ``scripts/*.py`` (``scripts/stopping/`` has
its own tests). Each generator:

- does no dataset I/O when imported with every dataset environment variable
  unset (``pyradbio.datasets.dataset_dir`` / ``dataset_file`` /
  ``load_project_dotenv`` are patched to raise);
- exposes ``main(argv=None)``, and ``main(["--dry-run", "--out", <dir>])``
  returns 0 and prints its plan without touching a dataset either.
"""

from __future__ import annotations

import importlib.util
import sys
from collections.abc import Callable, Iterator
from pathlib import Path
from types import ModuleType
from typing import Any

import pytest

import pyradbio.datasets as datasets

SCRIPTS_DIR = Path(__file__).resolve().parents[2] / "scripts"
GENERATORS = sorted(SCRIPTS_DIR.glob("[!_]*.py"))

_PARAMS = [pytest.param(p, id=p.stem) for p in GENERATORS] or [
    pytest.param(None, id="none", marks=pytest.mark.skip(reason="no scripts/ generators yet"))
]

_GUARDED = ("dataset_dir", "dataset_file", "load_project_dotenv")


class _DatasetAccessed(BaseException):
    """Raised by the patched dataset lookups; a BaseException so no script swallows it."""


@pytest.fixture
def no_datasets(monkeypatch: pytest.MonkeyPatch) -> Iterator[list[str]]:
    """Unset every dataset variable and make every dataset lookup raise."""
    for dataset in datasets.DATASETS.values():
        monkeypatch.delenv(dataset.env_var, raising=False)
    calls: list[str] = []

    def guard(name: str) -> Callable[..., Any]:
        def accessed(*args: Any, **kwargs: Any) -> Any:
            calls.append(f"{name}{args!r}")
            raise _DatasetAccessed(f"{name} called with {args!r} {kwargs!r}")

        return accessed

    originals = {name: getattr(datasets, name) for name in _GUARDED}
    # Patch the defining module and every already-imported pyradbio module that
    # bound the function by name (``from pyradbio.datasets import dataset_dir``).
    for module in [m for name, m in sys.modules.items() if name.startswith("pyradbio")]:
        for name, original in originals.items():
            if getattr(module, name, None) is original:
                monkeypatch.setattr(module, name, guard(name))
    yield calls


def _load(path: Path, monkeypatch: pytest.MonkeyPatch) -> ModuleType:
    monkeypatch.syspath_prepend(str(SCRIPTS_DIR))
    monkeypatch.setattr(sys, "argv", [str(path)])
    name = f"_scripts_cli_under_test_{path.stem}"
    spec = importlib.util.spec_from_file_location(name, path)
    assert spec is not None and spec.loader is not None
    module = importlib.util.module_from_spec(spec)
    monkeypatch.setitem(sys.modules, name, module)
    try:
        spec.loader.exec_module(module)
    except _DatasetAccessed as exc:
        pytest.fail(f"importing {path.name} touched a dataset: {exc}")
    return module


def _dry_run(module: ModuleType, path: Path, out: Path) -> Any:
    """``module.main(["--dry-run", "--out", out])``; dataset access fails the test."""
    try:
        return module.main(["--dry-run", "--out", str(out)])
    except _DatasetAccessed as exc:
        pytest.fail(f"{path.name} --dry-run touched a dataset: {exc}")


@pytest.mark.parametrize("path", _PARAMS)
def test_import_performs_no_dataset_io(
    path: Path, monkeypatch: pytest.MonkeyPatch, no_datasets: list[str]
) -> None:
    module = _load(path, monkeypatch)

    assert no_datasets == []
    assert callable(getattr(module, "main", None)), f"{path.name} has no main(argv=None)"


@pytest.mark.parametrize("path", _PARAMS)
def test_dry_run_returns_zero_and_prints_a_plan(
    path: Path,
    monkeypatch: pytest.MonkeyPatch,
    tmp_path: Path,
    capsys: pytest.CaptureFixture[str],
    no_datasets: list[str],
) -> None:
    module = _load(path, monkeypatch)
    capsys.readouterr()

    status = _dry_run(module, path, tmp_path)

    assert status == 0
    assert no_datasets == []
    assert capsys.readouterr().out.strip(), f"{path.name} --dry-run printed no plan"


# --- self-tests of the contract checks on throwaway scripts ---------------------
#
# The parametrizations above are empty while there is no generator script, so
# these keep the guard and the dry-run check honest.

_WELL_BEHAVED = """import argparse

import pyradbio.datasets as datasets


def main(argv=None):
    parser = argparse.ArgumentParser()
    parser.add_argument("--dry-run", action="store_true")
    parser.add_argument("--out")
    args = parser.parse_args(argv)
    if args.dry_run:
        print(f"would write reference.csv to {args.out}")
        return 0
    datasets.dataset_dir(datasets.TRIP98)
    return 0
"""


def _fake(tmp_path: Path, body: str) -> Path:
    path = tmp_path / "fake_generator.py"
    path.write_text(body, encoding="utf-8")
    return path


def test_the_guard_catches_dataset_io_at_import(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch, no_datasets: list[str]
) -> None:
    path = _fake(
        tmp_path,
        """\
import pyradbio.datasets as d

ROOT = d.dataset_dir(d.TRIP98)


def main(argv=None):
    return 0
""",
    )

    with pytest.raises(pytest.fail.Exception, match=r"importing fake_generator.py touched"):
        _load(path, monkeypatch)
    assert no_datasets and no_datasets[0].startswith("dataset_dir")


def test_the_guard_catches_a_name_bound_before_the_patch(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch, no_datasets: list[str]
) -> None:
    path = _fake(
        tmp_path,
        """\
from pyradbio.datasets import TRIP98, dataset_file


def main(argv=None):
    dataset_file(TRIP98, "normal02.rbe")
    return 0
""",
    )
    module = _load(path, monkeypatch)

    with pytest.raises(pytest.fail.Exception, match="--dry-run touched a dataset"):
        _dry_run(module, path, tmp_path)


def test_a_well_behaved_generator_passes_both_checks(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
    capsys: pytest.CaptureFixture[str],
    no_datasets: list[str],
) -> None:
    path = _fake(tmp_path, _WELL_BEHAVED)
    module = _load(path, monkeypatch)
    capsys.readouterr()

    assert _dry_run(module, path, tmp_path) == 0
    assert no_datasets == []
    assert "would write reference.csv" in capsys.readouterr().out
