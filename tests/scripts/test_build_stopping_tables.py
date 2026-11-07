"""The shipped stopping-power tables are exactly what build_tables.py makes from the raw data.

Skipped when the raw sources have not been fetched (they are never committed).
"""

from __future__ import annotations

import importlib.util
import sys
from pathlib import Path
from types import ModuleType

import pytest

_ROOT = Path(__file__).resolve().parents[2]
_SCRIPT = _ROOT / "scripts" / "stopping" / "build_tables.py"


def _load() -> ModuleType:
    spec = importlib.util.spec_from_file_location("build_stopping_tables", _SCRIPT)
    assert spec is not None and spec.loader is not None
    module = importlib.util.module_from_spec(spec)
    sys.modules[spec.name] = module
    spec.loader.exec_module(module)
    return module


bt = _load()


def test_shipped_tables_match_a_fresh_rebuild() -> None:
    raw = bt.RAW
    needed = [
        raw / "pstar" / "pstar_water_liquid.csv",
        raw / "astar" / "astar_water_liquid.csv",
        raw / "nist_isotopes" / "nist_isotopic_compositions.csv",
        raw / "icru73" / "icru73_2005_water_liquid.csv",
        raw / "icru73" / "icru73_2009_water_liquid.csv",
        *(raw / "icru90" / f"icru90_{p}_water_liquid.csv" for p in ("proton", "alpha", "carbon")),
        raw / "mstar" / "MSTAR312" / "Sources" / "msdbs1.d",
        raw / "estar" / "estar_water_liquid.csv",
        raw / "icru90" / "icru90_electron_water_liquid.csv",
        raw / "pdg" / "pdg_water_liquid_sternheimer.csv",
        raw / "arb1973" / "arb1973_table1.csv",
        raw / "icru49" / "icru49_water_empirical.csv",
        raw / "icru49" / "icru49_table41_reduced_nuclear.csv",
        raw / "ahlen1980" / "ahlen1980_table12.csv",
        raw / "brandt1982" / "brandt1982_table2.csv",
    ]
    if not all(p.is_file() for p in needed):
        pytest.skip("raw stopping-power sources not fetched (scripts/stopping/fetch_sources.py)")
    assert bt.check(raw, bt.OUT, computed=False) == []


@pytest.mark.slow
def test_computed_shell_table_matches_a_fresh_computation(tmp_path: Path) -> None:
    """The shipped C(eta) table is what pyradbio.track.stopping.shell computes (~40 s)."""
    built = bt.build_shell(tmp_path)
    assert built.read_bytes() == (bt.OUT / built.name).read_bytes()


def test_shipped_tables_carry_their_provenance() -> None:
    shipped = (
        "pstar_water_liquid.csv",
        "astar_water_liquid.csv",
        "icru73_2005_water_liquid.csv",
        "icru73_2009_water_liquid.csv",
        "icru90_water_liquid.csv",
        "mstar_c_water_liquid.csv",
        "mstar_d_water_liquid.csv",
        "density_effect_water_liquid.csv",
        "sternheimer_water_liquid.csv",
        "nist_isotope_masses.csv",
        "barkas_arb1973.csv",
        "icru49_empirical_water.csv",
        "icru49_reduced_nuclear.csv",
        "ahlen1980_cos_chi.csv",
        "brandt1982_ionization.csv",
    )
    for name in shipped:
        text = (bt.OUT / name).read_text(encoding="utf-8")
        assert "PROVENANCE.md" in text
        assert "sha256" in text
    star = (bt.OUT / "pstar_water_liquid.csv").read_text(encoding="utf-8")
    assert "15 U.S.C. 290e" in star
    assert "NOT the NIST table" in star
    icru73 = (bt.OUT / "icru73_2009_water_liquid.csv").read_text(encoding="utf-8")
    assert "libdedx" in icru73
    assert "NOT the ICRU table" in icru73
    barkas = (bt.OUT / "barkas_arb1973.csv").read_text(encoding="utf-8")
    assert "NOT the ARB table" in barkas
    shell = (bt.OUT / "shell_corrections_water.csv").read_text(encoding="utf-8")
    assert "no raw input" in shell.replace("\n# ", " ")
