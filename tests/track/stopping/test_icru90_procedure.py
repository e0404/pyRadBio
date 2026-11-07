"""ICRU90Procedure: MSTAR (scaled to ICRU 90 alphas) joined to Bethe, as ICRU 90 does for C.

ICRU Report 90 (2014), Appendix A.3 (printed p. 81): "The low-energy electronic stopping
powers from MSTAR have been connected to the high-energy results of the corrected BEST
code (I = 78 eV) by interpolating the quantity beta^2 S_el(T)/rho with cubic splines ...
using the scaled empirical data for T <= 30 MeV and the modified Bethe results for
T >= 60 MeV" (C ions, total kinetic energy); footnote 33: MSTAR "adjusted to reflect the
electronic stopping powers for alpha particles in liquid water ... from Tables A.11 and
A.12".
"""

from __future__ import annotations

import csv
from collections.abc import Callable
from pathlib import Path

import numpy as np
import numpy.typing as npt
import pytest

from pyradbio.core.radiation import Ion
from pyradbio.track.stopping.base import OutOfRangeError, UnsupportedIonError
from pyradbio.track.stopping.bethe import Bethe
from pyradbio.track.stopping.grid import LogEnergyGrid
from pyradbio.track.stopping.icru import ICRU
from pyradbio.track.stopping.icru90_procedure import DEFAULT_THRESHOLDS_MEV_U, ICRU90Procedure
from pyradbio.track.stopping.masses import nuclear_mass_MeV
from pyradbio.track.stopping.mstar.paul import mstar1
from pyradbio.track.stopping.nuclear import WaterNuclearStopping

BANDS = ((0.0, 0.01), (0.01, 0.1), (0.1, 1.0), (1.0, 10.0), (10.0, 100.0), (100.0, 1e9))


def _beta2(z: int, a: int, e: npt.NDArray[np.float64]) -> npt.NDArray[np.float64]:
    gamma = 1.0 + e * a / nuclear_mass_MeV(z, a)
    out: npt.NDArray[np.float64] = 1.0 - 1.0 / gamma**2
    return out


# --- the required comparison --------------------------------------------------------------


def test_reproduces_the_icru90_carbon_table(raw_source_file: Callable[[str, str], Path]) -> None:
    """The key test: ICRU90Procedure vs ICRU 90 table A.15, at the
    table's own energies inside our range. Deviation reported per band; the bounds pin
    the measured maxima (~2x), they are not an assumed accuracy.

    MEASURED, max |S/S_ICRU90 - 1| per band [<0.01, 0.01-0.1, 0.1-1, 1-10,
    10-100, >100 MeV/u]: 0.54%, 0.027%, 0.067%, 0.070%, 0.037%, 0.030% (default Bloch +
    Ahlen; with Lindhard-Sorensen the last three are 0.055/0.048/0.028%). Below T1 the
    MSTAR-on-ICRU-90-alphas part agrees to +-0.03% (confirming our reading of footnote
    33) except 4-5 keV total, at the alpha table's lower edge (-0.54%, +0.16%). Without
    the shell and Barkas corrections in the Bethe part (``shell_barkas=False``) the blend
    region is off by -0.34% (next test).
    """
    rows = list(
        csv.DictReader(
            line
            for line in raw_source_file("icru90", "icru90_carbon_water_liquid.csv").open()
            if not line.startswith("#")
        )
    )
    e = np.array([float(r["T_MeV"]) for r in rows]) / 12.0
    s = np.array([float(r["S_el_MeV_cm2_g"]) for r in rows])
    proc = ICRU90Procedure()
    lo, hi = proc.energy_range(6, 12)
    keep = (e >= lo) & (e <= hi)
    rel = proc.mass_stopping_power(6, 12, e[keep]) / s[keep] - 1.0
    lines = []
    for b_lo, b_hi in BANDS:
        m = (e[keep] >= b_lo) & (e[keep] < b_hi)
        if m.any():
            lines.append(
                f"[{b_lo:g},{b_hi:g}) MeV/u n={m.sum():2d} max={np.abs(rel[m]).max():.2e} "
                f"rms={np.sqrt(np.mean(rel[m] ** 2)):.2e}"
            )
    print("\nICRU90Procedure vs ICRU 90 carbon (A.15):", *lines, sep="\n  ")
    bounds = {0.0: 0.011, 0.01: 6e-4, 0.1: 1.4e-3, 1.0: 1.1e-3, 10.0: 1e-3, 100.0: 6e-4}
    for (b_lo, b_hi), bound in zip(BANDS, bounds.values(), strict=True):
        m = (e[keep] >= b_lo) & (e[keep] < b_hi)
        assert np.abs(rel[m]).max() < bound, (b_lo, b_hi)


# --- structure ----------------------------------------------------------------------------


def test_low_energies_are_mstar_on_icru90_alphas() -> None:
    """Below T1: Paul's MSTAR ratio applied to ICRU 90's alpha table (footnote 33)."""
    proc = ICRU90Procedure(evaluation="direct")
    e = np.array([0.01, 0.1, 1.0, 2.4])
    alpha = ICRU(edition="2016").mass_stopping_power(2, 4, e)
    expected = [mstar1(a, "b", 6, x)[0] for a, x in zip(alpha, e, strict=True)]
    np.testing.assert_allclose(proc.mass_stopping_power(6, 12, e), expected, rtol=1e-6)


def test_without_shell_and_barkas_the_blend_region_is_off(
    raw_source_file: Callable[[str, str], Path],
) -> None:
    """MEASURED (shell_barkas=False): -0.34% at 2.5-5 MeV/u, -0.27% at 5-10."""
    rows = list(
        csv.DictReader(
            line
            for line in raw_source_file("icru90", "icru90_carbon_water_liquid.csv").open()
            if not line.startswith("#")
        )
    )
    e = np.array([float(r["T_MeV"]) for r in rows]) / 12.0
    s = np.array([float(r["S_el_MeV_cm2_g"]) for r in rows])
    keep = (e >= 2.5) & (e < 10.0)
    proc = ICRU90Procedure(shell_barkas=False)
    rel = proc.mass_stopping_power(6, 12, e[keep]) / s[keep] - 1.0
    assert 2e-3 < np.abs(rel).max() < 5e-3


def test_high_energies_are_bethe_with_bloch_ahlen_shell_and_barkas() -> None:
    """The default is ICRU 90's own recipe: BEST + Ahlen's Mott term."""
    proc = ICRU90Procedure(evaluation="direct")
    e = np.array([7.5, 50.0, 500.0])  # oxygen's T2 is 5 x (8/6)^(4/3) = 7.3 MeV/u
    bethe = Bethe(
        effective_charge=None,
        i_value_eV=78.0,
        density_effect="tabulated",
        bloch=True,
        ahlen=True,
        shell=True,
        barkas=True,
    )
    np.testing.assert_allclose(
        proc.mass_stopping_power(8, 16, e), bethe.mass_stopping_power(8, 16, e), rtol=1e-6
    )


def test_the_blend_is_a_cubic_spline_of_beta2_s_that_meets_both_sides() -> None:
    """Continuous at T1 and T2 in value; the gap is filled smoothly (no kinks)."""
    proc = ICRU90Procedure(evaluation="direct")
    t1, t2 = DEFAULT_THRESHOLDS_MEV_U
    e = np.geomspace(t1 * 0.999, t2 * 1.001, 400)
    f = _beta2(6, 12, e) * proc.mass_stopping_power(6, 12, e)
    slopes = np.diff(np.log(f)) / np.diff(np.log(e))
    assert np.max(np.abs(np.diff(slopes))) < 1e-2  # smooth: no jump in d ln f / d ln E


def test_thresholds_are_options() -> None:
    assert DEFAULT_THRESHOLDS_MEV_U == pytest.approx((30.0 / 12, 60.0 / 12))
    proc = ICRU90Procedure(thresholds_MeV_u=(2.0, 6.0))
    info = proc.metadata(6, 12)
    assert info.options["thresholds_MeV_u"] == [2.0, 6.0]
    with pytest.raises(ValueError, match="thresholds"):
        ICRU90Procedure(thresholds_MeV_u=(6.0, 2.0))


def test_interpolated_default_matches_direct_evaluation() -> None:
    """Interpolation between our grid nodes by default, direct evaluation on request."""
    e = np.geomspace(0.01, 500.0, 37)
    fast = ICRU90Procedure().mass_stopping_power(6, 12, e)
    exact = ICRU90Procedure(evaluation="direct").mass_stopping_power(6, 12, e)
    assert np.abs(fast / exact - 1.0).max() < 1e-4


def test_a_user_grid_is_used_for_the_nodes() -> None:
    grid = LogEnergyGrid.from_points_per_decade(0.01, 300.0, 30)
    proc = ICRU90Procedure(grid=grid)
    assert proc.energy_range(6, 12) == (0.01, 300.0)
    np.testing.assert_allclose(
        proc.mass_stopping_power(6, 12, grid.energies),
        ICRU90Procedure(evaluation="direct").mass_stopping_power(6, 12, grid.energies),
        rtol=1e-12,
    )


# --- metadata, range, ions ----------------------------------------------------------------


def test_metadata_records_the_procedure() -> None:
    info = ICRU90Procedure().metadata(6, 12)
    assert info.source == "ICRU 90 procedure (pyRadBio)"
    assert info.i_value_eV == 78.0
    assert info.options["low_energy"].startswith("MSTAR")
    assert info.options["high_energy"].startswith("Bethe")
    assert "shell + Barkas" in str(info.options["high_energy"])
    assert info.options["evaluation"] == "interpolate"


def test_range_and_ions() -> None:
    proc = ICRU90Procedure()
    lo, _ = proc.energy_range(6, 12)
    assert lo == pytest.approx(0.00025)
    with pytest.raises(OutOfRangeError):
        proc.mass_stopping_power(6, 12, lo * 0.9)
    with pytest.raises(UnsupportedIonError):
        proc.mass_stopping_power(19, 39, 10.0)
    assert proc.let_keV_um(Ion(6, 12, 100.0)) > 0


# --- total stopping ---------------------------------------------------------------------

TOTAL_CASES = (
    # table, I, Z, A, max |S_tot/table - 1| (~2x measured)
    ("pstar", "pstar_water_liquid.csv", 75.0, 1, 1, 3e-3),
    ("icru90", "icru90_proton_water_liquid.csv", 78.0, 1, 1, 3e-3),
    ("astar", "astar_water_liquid.csv", 75.0, 2, 4, 6e-3),
    ("icru90", "icru90_alpha_water_liquid.csv", 78.0, 2, 4, 7e-3),
    ("icru90", "icru90_carbon_water_liquid.csv", 78.0, 6, 12, 2e-3),
)


@pytest.mark.parametrize(("source", "name", "i_ev", "z", "a", "bound"), TOTAL_CASES)
def test_total_stopping_reproduces_the_tables(
    raw_source_file: Callable[[str, str], Path],
    source: str,
    name: str,
    i_ev: float,
    z: int,
    a: int,
    bound: float,
) -> None:
    """Electronic + ICRU 49's nuclear method vs S_tot, 1 keV to 1000 MeV/u.

    MEASURED, max |dev|: PSTAR 0.14%, ICRU 90 p 0.14%, ASTAR 0.25%, ICRU 90
    alpha 0.33% (both in the p/alpha T1-T2 gap, as for the electronic part), ICRU 90 C
    0.10%.
    """
    rows = list(
        csv.DictReader(
            line for line in raw_source_file(source, name).open() if not line.startswith("#")
        )
    )
    e = np.array([float(r["T_MeV"]) for r in rows]) / a
    s = np.array([float(r["S_tot_MeV_cm2_g"]) for r in rows])
    proc = ICRU90Procedure(i_value_eV=i_ev, quantity="total")
    lo, hi = proc.energy_range(z, a)
    keep = (e >= lo) & (e <= min(hi, 1000.0))
    rel = proc.mass_stopping_power(z, a, e[keep]) / s[keep] - 1.0
    assert np.abs(rel).max() < bound


def test_total_adds_icru49s_nuclear_method() -> None:
    e = np.array([0.003, 0.3, 30.0])
    for z, a in ((1, 1), (6, 12)):
        total = ICRU90Procedure(quantity="total").mass_stopping_power(z, a, e)
        el = ICRU90Procedure().mass_stopping_power(z, a, e)
        nuc = WaterNuclearStopping(proton_potential="moliere").mass_stopping_power(z, a, e)
        np.testing.assert_allclose(total, el + nuc, rtol=1e-12)
    assert "protons: moliere" in str(
        ICRU90Procedure(quantity="total").metadata(1, 1).options["nuclear"]
    )


# --- protons and alphas: ICRU 49's procedure (sec. 3.5, Table 3.7) ------------------------

P_ALPHA_CASES = (
    # table, I, Z, A, max |dev| below T1 / in the gap / above T2 (~2x measured)
    ("pstar", "pstar_water_liquid.csv", 75.0, 1, 1, (5e-4, 3e-3, 3e-3)),
    ("icru90", "icru90_proton_water_liquid.csv", 78.0, 1, 1, (5e-4, 3e-3, 3e-3)),
    ("astar", "astar_water_liquid.csv", 75.0, 2, 4, (1e-3, 6e-3, 3e-3)),
    ("icru90", "icru90_alpha_water_liquid.csv", 78.0, 2, 4, (1e-3, 7e-3, 3e-3)),
)


@pytest.mark.parametrize(("source", "name", "i_ev", "z", "a", "bounds"), P_ALPHA_CASES)
def test_p_alpha_reproduce_icru49_at_75_ev_and_icru90_at_78_ev(
    raw_source_file: Callable[[str, str], Path],
    source: str,
    name: str,
    i_ev: float,
    z: int,
    a: int,
    bounds: tuple[float, float, float],
) -> None:
    """One procedure, two I-values, two independent tables, 1 keV to 1000 MeV/u.

    MEASURED, max |S/table - 1| below T1 / gap / above T2: PSTAR 0.020 / 0.13 /
    0.13%; ICRU 90 p 0.020 / 0.13 / 0.15%; ASTAR 0.049 / 0.26 / 0.14%; ICRU 90 alpha
    0.049 / 0.33 / 0.14%. Below T1 that is the tables' 4-digit printing; in the gap it is
    our spline knots vs ICRU 49's unstated ones (sec. 3.5: "fitted by a cubic spline").
    """
    rows = list(
        csv.DictReader(
            line for line in raw_source_file(source, name).open() if not line.startswith("#")
        )
    )
    e = np.array([float(r["T_MeV"]) for r in rows]) / a
    s = np.array([float(r["S_el_MeV_cm2_g"]) for r in rows])
    proc = ICRU90Procedure(i_value_eV=i_ev)
    t1, t2 = proc.metadata(z, a).options["thresholds_MeV_u"]  # type: ignore[misc]
    keep = e <= 1000.0
    rel = proc.mass_stopping_power(z, a, e[keep]) / s[keep] - 1.0
    ek = e[keep]
    for mask, bound in zip((ek < t1, (ek >= t1) & (ek < t2), ek >= t2), bounds, strict=True):
        assert mask.any()
        assert np.abs(rel[mask]).max() < bound


def test_p_alpha_low_energies_are_the_icru49_formula(
    raw_source_file: Callable[[str, str], Path],
) -> None:
    """The shipped table is ICRU 49 eqs. 3.1-3.3 at its nodes; the procedure returns it.

    MEASURED: <= 1e-6 except near the proton formula's switch at T_s = 10 keV, where the
    printed A1 leaves a 5.8e-5 step that the (smooth) table interpolates across: 4.2e-5.
    """
    from pyradbio.track.stopping import icru49_empirical as emp

    path = raw_source_file("icru49", "icru49_water_empirical.csv")
    with path.open(encoding="utf-8") as fh:
        rows = list(csv.DictReader(line for line in fh if not line.startswith("#")))
    coeffs: dict[str, dict[str, float]] = {}
    for r in rows:
        coeffs.setdefault(r["particle"], {})[r["quantity"]] = float(r["value"])
    proc = ICRU90Procedure(evaluation="direct")
    for particle, z, a, formula in (
        ("proton", 1, 1, emp.proton_cross_section),
        ("alpha", 2, 4, emp.alpha_cross_section),
    ):
        c = coeffs[particle]
        e = np.geomspace(0.0011 / a, 0.99 * c["T1_MeV"] / a, 17)
        expected = emp.mass_stopping_from_cross_section(formula(e * a, c), c["molecular_weight"])
        got = proc.mass_stopping_power(z, a, e)
        joint = (e * a > 0.009) & (e * a < 0.0115) if z == 1 else np.zeros_like(e, dtype=bool)
        np.testing.assert_allclose(got[~joint], expected[~joint], rtol=1e-6)
        np.testing.assert_allclose(got[joint], expected[joint], rtol=1e-4)


def test_p_alpha_thresholds_are_icru49_table_3_7_and_bethe_has_bloch() -> None:
    proc = ICRU90Procedure()
    assert proc.metadata(1, 1).options["thresholds_MeV_u"] == pytest.approx([0.15, 0.5])
    assert proc.metadata(2, 4).options["thresholds_MeV_u"] == pytest.approx([0.25, 0.5])
    info = proc.metadata(1, 1)
    assert str(info.options["low_energy"]).startswith("ICRU 49 empirical")
    assert "bloch, shell + Barkas" in str(info.options["high_energy"])
    assert "ICRU Report 49" in info.reference
    e = np.array([0.6, 20.0, 300.0])
    bethe = Bethe(i_value_eV=78.0, bloch=True, shell=True, barkas=True, min_beta_gamma=0.02)
    np.testing.assert_allclose(
        ICRU90Procedure(evaluation="direct").mass_stopping_power(1, 1, e),
        bethe.mass_stopping_power(1, 1, e),
        rtol=1e-12,
    )
    assert ICRU90Procedure(thresholds_MeV_u=(2.0, 6.0)).metadata(1, 1).options[
        "thresholds_MeV_u"
    ] == pytest.approx([0.15, 0.5])  # thresholds_MeV_u is for Z 3-18 only


def test_invalid_options_raise() -> None:
    with pytest.raises(ValueError, match="evaluation"):
        ICRU90Procedure(evaluation="exact")  # type: ignore[arg-type]
    with pytest.raises(ValueError, match="i_value_eV"):
        ICRU90Procedure(i_value_eV=0.0)


def test_more_invalid_options_raise() -> None:
    with pytest.raises(ValueError, match="higher_order"):
        ICRU90Procedure(higher_order="ahlen")  # type: ignore[arg-type]
    with pytest.raises(ValueError, match="quantity"):
        ICRU90Procedure(quantity="nuclear")  # type: ignore[arg-type]
    with pytest.raises(ValueError, match="blend"):
        ICRU90Procedure(blend_nodes_per_decade=1)
    with pytest.raises(ValueError, match="Bethe range"):
        ICRU90Procedure(thresholds_MeV_u=(0.5, 1.0)).energy_range(6, 12)
    with pytest.raises(ValueError, match="alpha table"):
        ICRU90Procedure(thresholds_MeV_u=(300.0, 400.0)).energy_range(6, 12)
    with pytest.raises(ValueError, match="exceeds"):
        ICRU90Procedure(grid=LogEnergyGrid(1e-5, 10.0, 20)).energy_range(6, 12)


def test_bloch_ahlen_is_icru90s_own_high_energy_choice() -> None:
    proc = ICRU90Procedure(higher_order="bloch_ahlen", evaluation="direct")
    e = np.array([50.0, 500.0])
    bethe = Bethe(
        effective_charge=None, i_value_eV=78.0, bloch=True, ahlen=True, shell=True, barkas=True
    )
    np.testing.assert_allclose(
        proc.mass_stopping_power(6, 12, e), bethe.mass_stopping_power(6, 12, e), rtol=1e-12
    )
    assert "bloch_ahlen" in str(proc.metadata(6, 12).options["high_energy"])


def test_bloch_mott_variant_and_other_isotopes() -> None:
    proc = ICRU90Procedure(higher_order="bloch_mott")
    assert "bloch_mott" in str(proc.metadata(6, 12).options["high_energy"])
    assert proc.metadata(6, 13).options["velocity_scaled_from"] == "C-12"
    assert proc.mass_stopping_power(6, 13, 100.0)[0] > 0


# --- per-ion thresholds (stripping scaling) ----------------------------------------------


def test_thresholds_scale_with_the_stripping_criterion() -> None:
    """T_i(Z) = T_i(C) max(1, (Z/6)^(4/3)): carbon and lighter unchanged."""
    proc = ICRU90Procedure()
    for z, a in ((3, 7), (6, 12)):
        assert proc.metadata(z, a).options["thresholds_MeV_u"] == pytest.approx([2.5, 5.0])
    f = (18 / 6) ** (4 / 3)
    assert proc.metadata(18, 40).options["thresholds_MeV_u"] == pytest.approx([2.5 * f, 5.0 * f])
    flat = ICRU90Procedure(threshold_scaling="none")
    assert flat.metadata(18, 40).options["thresholds_MeV_u"] == pytest.approx([2.5, 5.0])
    with pytest.raises(ValueError, match="threshold_scaling"):
        ICRU90Procedure(threshold_scaling="bohr")  # type: ignore[arg-type]


MSTAR_BOUNDS = {3: 0.025, 6: 0.03, 8: 0.03, 10: 0.03, 14: 0.025, 18: 0.025}


@pytest.mark.parametrize("z", sorted(MSTAR_BOUNDS))
def test_every_ion_stays_close_to_mstar_across_all_energies(z: int) -> None:
    """No partially stripped ion is pushed through the bare-charge Bethe part.

    MEASURED, worst |S/S_MSTAR - 1| over 0.5-300 MeV/u: Li 1.5%, C 2.0%, O 2.0%,
    Ne 2.0%, Si 1.5%, Ar 1.6% (bounds ~1.5x). With carbon's thresholds per nucleon
    (threshold_scaling="none") Si and Ar were +16% / +21% at 2.5-10 MeV/u.
    """
    from pyradbio.track.stopping import XSTAR
    from pyradbio.track.stopping.masses import most_abundant_mass_number

    a = most_abundant_mass_number(z)
    e = np.geomspace(0.5, 300.0, 120)
    rel = ICRU90Procedure().mass_stopping_power(z, a, e) / XSTAR().mass_stopping_power(z, a, e) - 1
    assert np.abs(rel).max() < MSTAR_BOUNDS[z]
    if z >= 14:
        flat = ICRU90Procedure(threshold_scaling="none").mass_stopping_power(z, a, e)
        assert (flat / XSTAR().mass_stopping_power(z, a, e) - 1).max() > 0.1
