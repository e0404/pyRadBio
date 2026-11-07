"""ICRU(edition=...): ICRU 49 + 73 (2005), + 73 errata (2009), ICRU 90 for p/He/C (2016).

Source-point tests read the raw values from the git-ignored data/stopping/ folder at
run time (skipped when absent) and evaluate the SHIPPED tables at the SOURCE points.
"""

from __future__ import annotations

import csv
from collections.abc import Callable
from pathlib import Path

import numpy as np
import numpy.typing as npt
import pytest

from pyradbio.core.radiation import Ion
from pyradbio.track.stopping.base import OutOfRangeError
from pyradbio.track.stopping.grid import LogEnergyGrid, leave_one_out_relative_error
from pyradbio.track.stopping.icru import ICRU, ICRU73_IONS, ICRU90_IONS
from pyradbio.track.stopping.masses import most_abundant_mass_number
from pyradbio.track.stopping.nuclear import WaterNuclearStopping
from pyradbio.track.stopping.xstar import XSTAR

BANDS = ((0.0, 0.01), (0.01, 0.1), (0.1, 1.0), (1.0, 10.0), (10.0, 100.0), (100.0, 1e9))


def _rows(path: Path) -> list[dict[str, str]]:
    return list(csv.DictReader(line for line in path.open() if not line.startswith("#")))


def _band_report(e: npt.NDArray[np.float64], rel: npt.NDArray[np.float64]) -> list[str]:
    out = []
    for lo, hi in BANDS:
        m = (e >= lo) & (e < hi)
        if m.any():
            out.append(
                f"[{lo:g},{hi:g}) n={m.sum():4d} max={np.abs(rel[m]).max():.2e} "
                f"rms={np.sqrt(np.mean(rel[m] ** 2)):.2e}"
            )
    return out


# --- source-point reproduction ---------------------------------------------------------


@pytest.mark.parametrize(("edition", "bound"), [("2005", 1.0e-5), ("2009", 2.0e-5)])
def test_icru73_tables_reproduce_the_source_at_the_source_points(
    raw_source_file: Callable[[str, str], Path], edition: str, bound: float
) -> None:
    """Every Z 3-18 at every source energy (16 x 53 points).

    MEASURED (50 points/decade): max 4.6e-6 (2005), 8.2e-6 (2009, in
    10-100 MeV/u); far below the 4-5 printed digits. Bounds ~2x measured.
    """
    rows = _rows(raw_source_file("icru73", f"icru73_{edition}_water_liquid.csv"))
    icru = ICRU(edition=edition)  # type: ignore[arg-type]
    e_all, rel_all = [], []
    for z in ICRU73_IONS:
        if edition == "2016" and z in ICRU90_IONS:
            continue
        sel = [r for r in rows if int(r["Z"]) == z]
        e = np.array([float(r["energy_MeV_u"]) for r in sel])
        s = np.array([float(r["S_el_MeV_cm2_g"]) for r in sel])
        rel = icru.mass_stopping_power(z, most_abundant_mass_number(z), e) / s - 1.0
        e_all.append(e)
        rel_all.append(rel)
    e_cat, rel_cat = np.concatenate(e_all), np.concatenate(rel_all)
    print(f"\nICRU 73 ({edition}), all Z 3-18:", *_band_report(e_cat, rel_cat), sep="\n  ")
    assert np.abs(rel_cat).max() < bound


@pytest.mark.parametrize(
    ("particle", "z", "a", "column", "bound"),
    [
        ("proton", 1, 1, "S_el_MeV_cm2_g", 1.5e-6),
        ("proton", 1, 1, "S_nuc_MeV_cm2_g", 1.5e-7),
        ("alpha", 2, 4, "S_el_MeV_cm2_g", 1.0e-6),
        ("alpha", 2, 4, "S_nuc_MeV_cm2_g", 3.0e-7),
        ("carbon", 6, 12, "S_el_MeV_cm2_g", 1.7e-6),
        ("carbon", 6, 12, "S_nuc_MeV_cm2_g", 2.1e-7),
    ],
)
def test_icru90_tables_reproduce_the_source_at_the_source_points(
    raw_source_file: Callable[[str, str], Path],
    particle: str,
    z: int,
    a: int,
    column: str,
    bound: float,
) -> None:
    """ICRU 90 p / alpha / C (T is total MeV in the source; T/A here).

    MEASURED, max over all points: electronic p 7.1e-7, alpha 4.4e-7,
    C 8.3e-7; nuclear p 7.0e-8, alpha 1.4e-7, C 1.0e-7. Bounds ~2x measured.
    """
    rows = _rows(raw_source_file("icru90", f"icru90_{particle}_water_liquid.csv"))
    e = np.array([float(r["T_MeV"]) for r in rows]) / a
    s = np.array([float(r[column]) for r in rows])
    kind = "nuclear" if "nuc" in column else "electronic"
    table = ICRU(edition="2016")._table(ICRU(edition="2016")._reference(z, a), kind)
    rel = table(e) / s - 1.0
    print(f"\nICRU 90 {particle} {kind}:", *_band_report(e, rel), sep="\n  ")
    assert np.abs(rel).max() < bound


def test_interpolation_error_near_the_stopping_maximum_is_quantified(
    raw_source_file: Callable[[str, str], Path],
) -> None:
    """Leave-one-out on the RAW ICRU grids within a factor 3 of the peak.

    These grids are much coarser than PSTAR's (ICRU 90: ~8 points per decade), so
    this is where the interpolant matters. MEASURED, cubic vs PCHIP:
    ICRU 90 p 3.1e-3 vs 1.5e-2, alpha 4.8e-3 vs 1.5e-2, C 3.0e-3 vs 2.3e-2; ICRU 73
    (both editions, Z 3/6/8/18) cubic 9e-5..1.9e-3 vs PCHIP 3.1e-3..2.0e-2. The cubic
    default is 3-200x better at the maximum. Bounds ~2x measured worst case.
    """
    cases = []
    for particle, a in (("proton", 1), ("alpha", 4), ("carbon", 12)):
        rows = _rows(raw_source_file("icru90", f"icru90_{particle}_water_liquid.csv"))
        e = np.array([float(r["T_MeV"]) for r in rows]) / a
        cases.append((e, np.array([float(r["S_el_MeV_cm2_g"]) for r in rows]), 1.0e-2, 4.6e-2))
    for edition in ("2005", "2009"):
        rows = _rows(raw_source_file("icru73", f"icru73_{edition}_water_liquid.csv"))
        for z in (3, 6, 8, 18):
            sel = [r for r in rows if int(r["Z"]) == z]
            e = np.array([float(r["energy_MeV_u"]) for r in sel])
            s = np.array([float(r["S_el_MeV_cm2_g"]) for r in sel])
            cases.append((e, s, 3.7e-3, 4.1e-2))
    for e, s, cubic_bound, pchip_bound in cases:
        peak = e[np.argmax(s)]
        near = (e[1:-1] > peak / 3.0) & (e[1:-1] < peak * 3.0)
        cubic = np.abs(leave_one_out_relative_error(e, s, "cubic")[near]).max()
        pchip = np.abs(leave_one_out_relative_error(e, s, "pchip")[near]).max()
        assert cubic < cubic_bound
        assert pchip < pchip_bound
        assert cubic < pchip


# --- editions --------------------------------------------------------------------------


def _all_ions(icru: ICRU, e: npt.NDArray[np.float64]) -> dict[int, npt.NDArray[np.float64]]:
    return {z: icru.mass_stopping_power(z, most_abundant_mass_number(z), e) for z in range(1, 19)}


_E = np.geomspace(0.03, 200.0, 17)  # inside every source's range


def test_errata_change_exactly_the_icru73_ions() -> None:
    old, new = _all_ions(ICRU(edition="2005"), _E), _all_ions(ICRU(edition="2009"), _E)
    changed = {z for z in old if not np.array_equal(old[z], new[z])}
    assert changed == set(ICRU73_IONS)


def test_2016_changes_exactly_the_icru90_ions() -> None:
    errata, icru90 = _all_ions(ICRU(edition="2009"), _E), _all_ions(ICRU(edition="2016"), _E)
    changed = {z for z in errata if not np.array_equal(errata[z], icru90[z])}
    assert changed == set(ICRU90_IONS) == {1, 2, 6}


def test_icru49_ions_are_the_xstar_tables() -> None:
    xstar, icru = XSTAR(), ICRU(edition="2005")
    for z, a in ((1, 1), (2, 4)):
        np.testing.assert_array_equal(
            icru.mass_stopping_power(z, a, _E), xstar.mass_stopping_power(z, a, _E)
        )


def test_unknown_edition_raises() -> None:
    with pytest.raises(ValueError, match="edition"):
        ICRU(edition="2014")  # type: ignore[arg-type]


# --- ranges ----------------------------------------------------------------------------


def test_icru73_starts_at_25_kev_per_u_and_raises_below() -> None:
    icru = ICRU(edition="2009")
    assert icru.energy_range(8, 16) == (0.025, 1000.0)
    icru.mass_stopping_power(8, 16, 0.025)
    with pytest.raises(OutOfRangeError):
        icru.mass_stopping_power(8, 16, 0.0249)


def test_ranges_per_edition() -> None:
    assert ICRU(edition="2005").energy_range(1, 1) == (0.001, 10000.0)
    assert ICRU(edition="2016").energy_range(1, 1) == (0.001, 10000.0)
    assert ICRU(edition="2016").energy_range(2, 4) == (0.00025, 250.0)
    lo, hi = ICRU(edition="2016").energy_range(6, 12)
    assert (lo, hi) == pytest.approx((0.001 / 12, 10000.0 / 12), rel=1e-15)
    assert ICRU(edition="2009").energy_range(6, 12) == (0.025, 1000.0)


# --- metadata --------------------------------------------------------------------------


def test_metadata_is_per_ion() -> None:
    icru = ICRU(edition="2016")
    p, he, c, o = (icru.metadata(z, a) for z, a in ((1, 1), (2, 4), (6, 12), (8, 16)))
    assert (p.source, he.source, c.source) == ("ICRU 90", "ICRU 90", "ICRU 90")
    assert p.i_value_eV == he.i_value_eV == c.i_value_eV == 78.0
    assert o.source == "ICRU 73 errata (2009)"
    # Errata (Sigmund, Schinner & Paul 2009) sec. 2.2: "Liquid water, I = 78.0 eV".
    assert o.i_value_eV == 78.0
    assert "errata" in str(o.options["i_value_note"]).lower()
    assert "printed" in str(o.options["surrogate"])  # verified against the report
    assert all(m.options["edition"] == "2016" for m in (p, he, c, o))


def test_metadata_of_the_older_editions() -> None:
    old = ICRU(edition="2005")
    assert old.metadata(1, 1).source == "PSTAR"
    assert old.metadata(1, 1).i_value_eV == 75.0
    assert old.metadata(6, 12).source == "ICRU 73 (2005)"
    # ICRU 73 prints no water I-value; the errata (sec. 2.2) state that its oscillator
    # spectrum for liquid water "gives an I-value of 67.2 eV".
    assert old.metadata(6, 12).i_value_eV == 67.2
    assert "implied" in str(old.metadata(6, 12).options["i_value_note"])


def test_other_isotopes_are_velocity_scaled() -> None:
    icru = ICRU(edition="2016")
    assert icru.metadata(6, 13).options["velocity_scaled_from"] == "C-12"
    assert icru.metadata(8, 18).options["velocity_scaled_from"] == "O-16"
    assert "velocity_scaled_from" not in icru.metadata(8, 16).options


def test_let_adapter_and_user_grid() -> None:
    grid = LogEnergyGrid.from_points_per_decade(0.05, 500.0, 25)
    icru = ICRU(edition="2016", grid=grid)
    np.testing.assert_allclose(
        icru.mass_stopping_power(6, 12, grid.energies),
        ICRU(edition="2016").mass_stopping_power(6, 12, grid.energies),
        rtol=1e-12,
    )
    s = icru.mass_stopping_power(8, 16, 100.0)[0]
    assert icru.let_keV_um(Ion(8, 16, 100.0)) == pytest.approx(0.1 * s)


# --- total stopping --------------------------------------------------------------------


@pytest.mark.parametrize(("edition", "z", "a"), [("2016", 1, 1), ("2016", 6, 12), ("2005", 2, 4)])
def test_total_uses_the_tabulated_nuclear_column(edition: str, z: int, a: int) -> None:
    e = np.geomspace(0.03, 200.0, 9)
    el = ICRU(edition=edition).mass_stopping_power(z, a, e)  # type: ignore[arg-type]
    tot = ICRU(edition=edition, quantity="total").mass_stopping_power(z, a, e)  # type: ignore[arg-type]
    assert np.all(tot > el)


def test_total_for_icru73_ions_adds_icru49s_nuclear_model() -> None:
    """ICRU 73 is electronic only; total = ICRU 73 + ICRU 49's nuclear method (as ICRU 90
    used for C). Nuclear is 4.7% of the total at 0.03 MeV/u and 0.1% at 1 MeV/u."""
    e = np.array([0.03, 0.1, 1.0])
    total = ICRU(edition="2016", quantity="total")
    el = ICRU(edition="2016").mass_stopping_power(8, 16, e)
    nuc = WaterNuclearStopping(proton_potential="moliere").mass_stopping_power(8, 16, e)
    np.testing.assert_allclose(total.mass_stopping_power(8, 16, e), el + nuc, rtol=1e-12)
    assert "Table 4.1" in str(total.metadata(8, 16).options["nuclear"])
    assert total.metadata(1, 1).options["nuclear"] == "tabulated (p)"
    assert ICRU(edition="2016").metadata(8, 16).options["nuclear"] is None


def test_total_for_other_isotopes_uses_the_model_not_the_table() -> None:
    """Deuterons: velocity-scaled electronic stopping, nuclear from the model (mass matters)."""
    e = np.array([0.01, 0.1])
    total = ICRU(edition="2016", quantity="total").mass_stopping_power(1, 2, e)
    el = ICRU(edition="2016").mass_stopping_power(1, 2, e)
    np.testing.assert_allclose(
        total,
        el + WaterNuclearStopping(proton_potential="moliere").mass_stopping_power(1, 2, e),
        rtol=1e-12,
    )
