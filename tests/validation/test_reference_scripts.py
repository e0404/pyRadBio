"""Tests of the external-reference scripts (TRiP98 tables, the Survival code).

``validation/survival_code_mmkm.py`` is, like every validation script, smoke-tested by
``test_validation_smoke.py``; the TRiP98 name check below is derived from the RBE
file the script actually loaded. The two Survival generators in ``scripts/`` are
pinned here against the grids and command line of the original paired-MC script
and the committed cache.
"""

from __future__ import annotations

import importlib.util
import json
import os
import runpy
import subprocess
import sys
from pathlib import Path
from types import ModuleType
from typing import Any

import pytest

ROOT = Path(__file__).resolve().parents[2]
VALIDATION_DIR = ROOT / "validation"
SCRIPTS_DIR = ROOT / "scripts"


def _load_script(name: str, monkeypatch: pytest.MonkeyPatch) -> ModuleType:
    path = SCRIPTS_DIR / f"{name}.py"
    spec = importlib.util.spec_from_file_location(f"_reference_script_{name}", path)
    assert spec is not None and spec.loader is not None
    module = importlib.util.module_from_spec(spec)
    monkeypatch.setitem(sys.modules, spec.name, module)
    spec.loader.exec_module(module)
    return module


# ---- validation/trip98_lem4_rbe.py ------------------------------------------
def _trip98_lem4_available() -> bool:
    from pyradbio.benchmark import reference_tables as rt

    try:
        # The same files the script loads, env overrides included.
        rt.load_trip_lem4(os.environ.get("TRIP98_LEM4_RBE_FILE", "_spinal-cord_lemIV.rbe"))
        rt.load_trip_dedx(os.environ.get("TRIP98_DEDX_FILE", "dEdxFLUKAxTRiP.dedx"))
    except (FileNotFoundError, OSError):
        return False
    return True


@pytest.mark.slow
@pytest.mark.skipif(not _trip98_lem4_available(), reason="TRiP98 LEM IV tables not present")
def test_trip98_lem4_figures_are_named_after_the_loaded_rbe_file(
    monkeypatch: pytest.MonkeyPatch, tmp_path: Path
) -> None:
    """Quick mode writes ``lem4_trip98_{rbe,diagnostics}_<stem>.png`` for its RBE file."""
    matplotlib = pytest.importorskip("matplotlib")
    monkeypatch.setenv("MPLBACKEND", "Agg")
    monkeypatch.setenv("PYRADBIO_FIGURES_DIR", str(tmp_path))
    monkeypatch.setenv("PYRADBIO_QUICK", "1")
    monkeypatch.syspath_prepend(str(VALIDATION_DIR))
    matplotlib.use("Agg", force=True)
    import matplotlib.pyplot as plt

    try:
        namespace = runpy.run_path(str(VALIDATION_DIR / "trip98_lem4_rbe.py"), run_name="__main__")
    finally:
        plt.close("all")

    stem = Path(namespace["RBE_FILE"]).stem.lstrip("_")
    written = {p.name for p in tmp_path.glob("*.png")}
    assert written == {f"lem4_trip98_rbe_{stem}.png", f"lem4_trip98_diagnostics_{stem}.png"}
    assert not list(tmp_path.glob("*.npz")), "quick mode must not write the route cache"


# ---- scripts/survival_paired_mc.py ------------------------------------------
# The grid constants of the original paired-MC script.
_OLD_D_T = {"LEMI": 30.0, "LEMII": 6.0, "LEMIII": 19.0}
_OLD_IONS = {"H": 1, "He": 4, "C": 12, "O": 16}
_OLD_Z_OF = {"H": 1, "He": 2, "C": 6, "O": 8}
_OLD_ENERGIES = (3.0, 10.0, 30.0, 100.0, 300.0)


def _old_todo(cp: dict[str, Any]) -> list[tuple[str, str, float]]:
    """The ``todo`` list of the old ``main()``, verbatim."""
    return sorted(
        (
            (m, ion, e)
            for m in _OLD_D_T
            for ion in _OLD_IONS
            for e in _OLD_ENERGIES
            if f"{m}|{ion}|{e:g}" not in cp
        ),
        key=lambda t: (t[2], {"O": 0, "C": 1, "He": 2, "H": 3}[t[1]], t[0]),
    )


def test_survival_paired_mc_grid_is_the_old_one(monkeypatch: pytest.MonkeyPatch) -> None:
    mod = _load_script("survival_paired_mc", monkeypatch)
    assert (mod.ALPHA_X, mod.BETA_X, mod.R_N) == (0.313, 0.0615, 5.0)
    assert mod.D_T == _OLD_D_T and mod.IONS == _OLD_IONS and mod.Z_OF == _OLD_Z_OF
    assert mod.ENERGIES == _OLD_ENERGIES
    assert mod.plan_points({}) == _old_todo({})
    partial = {"LEMI|O|3": {}, "LEMII|H|300": {}, "LEMIII|C|10|failed": "x"}
    assert mod.plan_points(partial) == _old_todo(partial)


def test_survival_paired_mc_dry_run_prints_the_old_grid(
    monkeypatch: pytest.MonkeyPatch, capsys: pytest.CaptureFixture[str], tmp_path: Path
) -> None:
    mod = _load_script("survival_paired_mc", monkeypatch)
    assert mod.main(["--dry-run", "--out", str(tmp_path)]) == 0
    lines = capsys.readouterr().out.splitlines()
    planned = [line.strip() for line in lines if line.startswith("  ")]
    assert planned == [f"{m}|{ion}|{e:g}" for m, ion, e in _old_todo({})]
    assert not list(tmp_path.iterdir())


def test_survival_paired_mc_runs_survival_with_the_old_command_line(
    monkeypatch: pytest.MonkeyPatch, tmp_path: Path
) -> None:
    """``run_point`` passes every table-1 parameter explicitly (Survival's defaults differ)."""
    mod = _load_script("survival_paired_mc", monkeypatch)
    paths = mod.Paths(
        exe=tmp_path / "survival.exe", data=tmp_path / "data", work=tmp_path, out_dir=tmp_path
    )
    calls: list[dict[str, Any]] = []

    def fake_run(cmd: list[str], **kw: Any) -> subprocess.CompletedProcess[str]:
        calls.append({"cmd": cmd, **kw})
        (tmp_path / "NewProject_LQparameters_LEMII.csv").write_text(
            "alpha,beta,alpha_sigma,beta_sigma,meanLET\n0.9,0.05,0.004,0.001,0.0186\n"
        )
        return subprocess.CompletedProcess(cmd, 0, "", "")

    monkeypatch.setattr(mod.subprocess, "run", fake_run)
    row, _ = mod.run_point(paths, "LEMII", "He", 10.0, 20, timeout=60.0)

    assert row["alpha"] == "0.9"
    (call,) = calls
    assert call["cmd"] == [
        str(tmp_path / "survival.exe"),
        "-output", "LQ_pars", "-model", "LEMII", "-calculusType", "MonteCarlo",
        "-precision", "20", "-parallelismType", "0", "-ion", "He", "-energies", "40",
        "-cellType", "HSG", "-LEM_alpha0", "0.313", "-LEM_beta0", "0.0615",
        "-LEM_rNucleus", "5", "-LEM_Dt", "6",
    ]  # fmt: skip
    assert call["cwd"] == tmp_path and call["timeout"] == 60.0
    assert call["env"]["DATA"] == str(tmp_path / "data")


def test_survival_paired_mc_export_is_the_load_survival_table_form(
    monkeypatch: pytest.MonkeyPatch, tmp_path: Path
) -> None:
    from pyradbio.benchmark.reference_tables import parse_survival_lq

    mod = _load_script("survival_paired_mc", monkeypatch)
    paths = mod.Paths(exe=tmp_path, data=tmp_path, work=tmp_path, out_dir=tmp_path)
    row = {"alpha": 1.2, "beta": 0.01, "alpha_sigma": 0.006, "beta_sigma": 0.002, "cells": 40}
    mod.export(paths, "LEMI", {"LEMI|C|3": row, "LEMI|C|30": row, "LEMI|O|3|failed": "x"})
    out = tmp_path / "LEMI_MonteCarloPaired_HSG_Manganaro2018.json"
    table = parse_survival_lq(out.read_text())
    (carbon,) = table.curves
    assert (carbon.Z, carbon.A, carbon.energies_MeV_u) == (6, 12, (3.0, 30.0))
    assert carbon.alpha_sigma == (0.006, 0.006)
    assert json.loads(out.read_text())["meta"]["modelParameters"]["Dt"] == 30.0


# ---- scripts/lem_mc_vs_survival.py ------------------------------------------
def test_lem_mc_vs_survival_plans_exactly_the_committed_points(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    mod = _load_script("lem_mc_vs_survival", monkeypatch)
    data = json.loads((VALIDATION_DIR / "data" / mod.CACHE_NAME).read_text(encoding="utf-8"))
    data.pop("_provenance")
    committed = sorted((e["model"], e["Z"], e["A"], e["E"]) for e in data.values())
    assert sorted(mod.plan_points()) == committed
    assert mod.plan_points(only=mod.parse_only("LEMI:C:20,LEMIII:H:300")) == [
        ("LEMI", 6, 12, 20.0),
        ("LEMIII", 1, 1, 300.0),
    ]
    assert mod.default_out_dir() == VALIDATION_DIR / "data"
