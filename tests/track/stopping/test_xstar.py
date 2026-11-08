"""XSTAR: PSTAR (p), ASTAR (He) and heavy ions, on our shipped regridded tables.

The source-point tests read the raw NIST values from the git-ignored data/stopping/
folder at run time (skipped when absent) and evaluate the SHIPPED tables at the
SOURCE energies, never at our grid's nodes or midpoints.
"""

from __future__ import annotations

import csv
from collections.abc import Callable
from pathlib import Path

import numpy as np
import numpy.typing as npt
import pytest

from pyradbio.core.radiation import Ion
from pyradbio.track.stopping.base import (
    OutOfRangeError,
    UnsupportedIonError,
    beta_from_energy_per_nucleon,
)
from pyradbio.track.stopping.effective_charge import BarkasEffectiveCharge
from pyradbio.track.stopping.grid import LogEnergyGrid, leave_one_out_relative_error
from pyradbio.track.stopping.masses import nuclear_mass_MeV
from pyradbio.track.stopping.nuclear import WaterNuclearStopping
from pyradbio.track.stopping.xstar import XSTAR, XSTAR_I_VALUE_EV

# Energy bands [MeV/u] in which source-point deviations are reported.
BANDS = ((0.0, 0.01), (0.01, 0.1), (0.1, 1.0), (1.0, 10.0), (10.0, 100.0), (100.0, 1e9))


def _raw(path: Path) -> dict[str, npt.NDArray[np.float64]]:
    rows = list(csv.DictReader(line for line in path.open() if not line.startswith("#")))
    return {k: np.array([float(r[k]) for r in rows]) for k in rows[0]}


def _band_report(e: npt.NDArray[np.float64], rel: npt.NDArray[np.float64]) -> list[str]:
    out = []
    for lo, hi in BANDS:
        m = (e >= lo) & (e < hi)
        if m.any():
            out.append(
                f"[{lo:g},{hi:g}) n={m.sum():3d} max={np.abs(rel[m]).max():.2e} "
                f"rms={np.sqrt(np.mean(rel[m] ** 2)):.2e}"
            )
    return out


@pytest.mark.parametrize(
    ("source", "z", "a", "column", "bound"),
    [
        ("pstar", 1, 1, "S_el_MeV_cm2_g", 6.0e-5),
        ("pstar", 1, 1, "S_nuc_MeV_cm2_g", 5.0e-5),
        ("astar", 2, 4, "S_el_MeV_cm2_g", 9.0e-5),
        ("astar", 2, 4, "S_nuc_MeV_cm2_g", 5.0e-5),
    ],
)
def test_shipped_tables_reproduce_the_source_at_the_source_points(
    raw_source_file: Callable[[str, str], Path],
    source: str,
    z: int,
    a: int,
    column: str,
    bound: float,
) -> None:
    """Max relative deviation at every raw NIST point; per-band values in the step report.

    MEASURED (50 points/decade, not-a-knot cubic), max over all points:
    PSTAR electronic 2.8e-5 (worst band 0.1-1 MeV/u), nuclear 2.1e-5; ASTAR
    electronic 4.3e-5 (above 100 MeV/u), nuclear 2.2e-5. All below the sources' own
    printed precision (4 significant digits, up to 5e-4). Bounds: ~2x measured.
    """
    raw = _raw(raw_source_file(source, f"{source}_water_liquid.csv"))
    e = raw["T_MeV"] / a
    xstar = XSTAR(quantity="electronic")
    shipped = xstar._reference_table(z, "nuclear" if "nuc" in column else "electronic")
    rel = shipped(e) / raw[column] - 1.0
    print(f"\n{source} {column}:", *_band_report(e, rel), sep="\n  ")
    assert np.abs(rel).max() < bound


def test_total_is_electronic_plus_nuclear_for_the_reference_particles() -> None:
    e = np.geomspace(0.002, 200.0, 25)
    for z, a in ((1, 1), (2, 4)):
        total = XSTAR(quantity="total").mass_stopping_power(z, a, e)
        el = XSTAR(quantity="electronic").mass_stopping_power(z, a, e)
        nuc = XSTAR()._reference_table(z, "nuclear")(e)
        np.testing.assert_allclose(total, el + nuc, rtol=1e-14)


def test_interpolation_error_near_the_stopping_maximum_is_quantified(
    raw_source_file: Callable[[str, str], Path],
) -> None:
    """Leave-one-out on the RAW grid around the Bragg peak (no truth between points).

    It doubles the local spacing (and includes the 4-digit print rounding), so it is
    a conservative bound on interpolating the raw grid itself. MEASURED
    within a factor 3 of the peak: PSTAR (peak 0.08 MeV/u) cubic 1.9e-3, PCHIP 4.5e-3;
    ASTAR (peak 0.175 MeV/u) cubic 5.4e-4, PCHIP 3.2e-3. Bounds: ~2x measured.
    """
    for source, a, cubic_bound, pchip_bound in (
        ("pstar", 1, 4.0e-3, 9.0e-3),
        ("astar", 4, 1.1e-3, 6.5e-3),
    ):
        raw = _raw(raw_source_file(source, f"{source}_water_liquid.csv"))
        e, s = raw["T_MeV"] / a, raw["S_el_MeV_cm2_g"]
        peak = e[np.argmax(s)]
        near = (e[1:-1] > peak / 3.0) & (e[1:-1] < peak * 3.0)
        cubic = np.abs(leave_one_out_relative_error(e, s, "cubic")[near]).max()
        pchip = np.abs(leave_one_out_relative_error(e, s, "pchip")[near]).max()
        print(f"\n{source}: peak {peak:g} MeV/u, LOO max cubic {cubic:.2e} pchip {pchip:.2e}")
        assert cubic < cubic_bound
        assert pchip < pchip_bound


# --- range, metadata, interface --------------------------------------------------------


def test_ranges_follow_the_sources() -> None:
    xstar = XSTAR()
    assert xstar.energy_range(1, 1) == (0.001, 10000.0)
    assert xstar.energy_range(2, 4) == (0.00025, 250.0)  # ASTAR ends at 1000 MeV total


@pytest.mark.parametrize(("z", "a", "e"), [(2, 4, 250.01), (2, 4, 0.0002), (1, 1, 10001.0)])
def test_out_of_range_raises(z: int, a: int, e: float) -> None:
    with pytest.raises(OutOfRangeError):
        XSTAR().mass_stopping_power(z, a, e)


def test_metadata_records_source_and_the_fixed_i_value() -> None:
    xstar = XSTAR()
    p, he = xstar.metadata(1, 1), xstar.metadata(2, 4)
    assert (p.source, he.source) == ("PSTAR", "ASTAR")
    assert p.i_value_eV == he.i_value_eV == XSTAR_I_VALUE_EV == 75.0
    assert p.edition == "ICRU 49"
    assert p.options["quantity"] == "electronic"
    assert p.options["interpolation"] == "cubic"
    assert "10.18434/T4NC7P" in p.reference


def test_i_value_is_not_an_option() -> None:
    with pytest.raises(TypeError):
        XSTAR(i_value_eV=78.0)  # type: ignore[call-arg]


def test_let_adapter() -> None:
    xstar = XSTAR()
    s = xstar.mass_stopping_power(1, 1, 100.0)[0]
    assert xstar.let_keV_um(Ion(1, 1, 100.0)) == pytest.approx(s * 0.1)


def test_invalid_options_raise() -> None:
    with pytest.raises(ValueError, match="quantity"):
        XSTAR(quantity="nuclear")  # type: ignore[arg-type]
    with pytest.raises(ValueError, match="heavy_ion_source"):
        XSTAR(heavy_ion_source="srim")  # type: ignore[arg-type]
    with pytest.raises(ValueError, match="mstar_mode"):
        XSTAR(mstar_mode="z")
    with pytest.raises(ValueError, match="zeff_scaling"):
        XSTAR(zeff_scaling="cube")  # type: ignore[arg-type]


# --- isotopes: velocity scaling with true masses ---------------------------------------


def test_reference_particles_are_looked_up_unscaled() -> None:
    xstar = XSTAR()
    e = np.array([0.001, 1.0, 10000.0])
    np.testing.assert_array_equal(
        xstar.mass_stopping_power(1, 1, e), xstar._reference_table(1, "electronic")(e)
    )


def test_a_deuteron_is_a_proton_at_the_same_velocity() -> None:
    xstar = XSTAR()
    e_d = 10.0
    beta = beta_from_energy_per_nucleon(e_d, 2, nuclear_mass_MeV(1, 2))
    m_p = nuclear_mass_MeV(1, 1)
    e_p = m_p * (1.0 / np.sqrt(1.0 - beta**2) - 1.0)
    assert xstar.mass_stopping_power(1, 2, e_d)[0] == pytest.approx(
        xstar.mass_stopping_power(1, 1, e_p)[0], rel=1e-12
    )
    assert xstar.metadata(1, 2).options["velocity_scaled_from"] == "p"


def test_helium_3_is_scaled_from_helium_4() -> None:
    xstar = XSTAR()
    assert xstar.metadata(2, 3).options["velocity_scaled_from"] == "He-4"
    lo, hi = xstar.energy_range(2, 3)
    assert hi > 250.0  # a He-3 nucleon is heavier than a He-4 one: same velocity, higher E/A
    assert xstar.mass_stopping_power(2, 3, [lo, hi]).shape == (2,)


# --- heavy ions ------------------------------------------------------------------------


def test_zeff_scaled_heavy_ions_are_pstar_at_equal_velocity() -> None:
    xstar = XSTAR(heavy_ion_source="pstar_zeff_scaled", zeff_scaling="ratio")
    barkas = BarkasEffectiveCharge()
    e = np.array([0.5, 5.0, 50.0])
    beta = np.array([beta_from_energy_per_nucleon(x, 12, nuclear_mass_MeV(6, 12)) for x in e])
    m_p = nuclear_mass_MeV(1, 1)
    e_p = m_p * (1.0 / np.sqrt(1.0 - beta**2) - 1.0)
    s_p = xstar.mass_stopping_power(1, 1, e_p)
    ratio = (barkas.z_eff(6, beta) / barkas.z_eff(1, beta)) ** 2
    np.testing.assert_allclose(xstar.mass_stopping_power(6, 12, e), ratio * s_p, rtol=1e-12)
    absolute = XSTAR(heavy_ion_source="pstar_zeff_scaled", zeff_scaling="absolute")
    np.testing.assert_allclose(
        absolute.mass_stopping_power(6, 12, e), barkas.z_eff(6, beta) ** 2 * s_p, rtol=1e-12
    )
    info = xstar.metadata(6, 12)
    assert info.source.startswith("PSTAR")
    assert info.options["zeff_model"] == "barkas"
    assert info.options["zeff_scaling"] == "ratio"


def test_helium_uses_astar_even_when_heavy_ions_are_zeff_scaled() -> None:
    zeff = XSTAR(heavy_ion_source="pstar_zeff_scaled")
    assert zeff.metadata(2, 4).source == "ASTAR"
    np.testing.assert_array_equal(
        zeff.mass_stopping_power(2, 4, 10.0), XSTAR().mass_stopping_power(2, 4, 10.0)
    )


def test_zeff_scaling_stops_at_argon() -> None:
    xstar = XSTAR(heavy_ion_source="pstar_zeff_scaled")
    assert xstar.supported_Z() == tuple(range(1, 19))
    with pytest.raises(UnsupportedIonError):
        xstar.mass_stopping_power(19, 39, 10.0)


def test_a_custom_effective_charge_model_is_used() -> None:
    class Bare:
        name = "bare"

        def z_eff(self, Z: int, beta: npt.NDArray[np.float64]) -> npt.NDArray[np.float64]:
            return np.full_like(np.asarray(beta, dtype=float), float(Z))

    xstar = XSTAR(heavy_ion_source="pstar_zeff_scaled", zeff_model=Bare())
    # With a bare charge, carbon is exactly 36x a proton at the SAME velocity
    # (equal E/A is only ~equal velocity: m_C/12 < m_p).
    gamma_minus_1 = 100.0 * 12 / nuclear_mass_MeV(6, 12)
    e_p = gamma_minus_1 * nuclear_mass_MeV(1, 1)
    p = xstar.mass_stopping_power(1, 1, e_p)[0]
    c = xstar.mass_stopping_power(6, 12, 100.0)[0]
    assert c / p == pytest.approx(36.0, rel=1e-13)
    assert xstar.metadata(6, 12).options["zeff_model"] == "bare"


# --- user grid -------------------------------------------------------------------------


def test_a_user_grid_holds_the_tables_on_its_nodes() -> None:
    grid = LogEnergyGrid.from_points_per_decade(0.01, 100.0, 20)
    on_grid = XSTAR(grid=grid)
    shipped = XSTAR()
    np.testing.assert_allclose(
        on_grid.mass_stopping_power(1, 1, grid.energies),
        shipped.mass_stopping_power(1, 1, grid.energies),
        rtol=1e-12,
    )
    assert on_grid.energy_range(1, 1) == (0.01, 100.0)
    assert on_grid.metadata(1, 1).options["grid"] == grid.to_dict()
    with pytest.raises(OutOfRangeError):
        on_grid.mass_stopping_power(1, 1, 200.0)


def test_a_user_grid_outside_a_source_raises_at_lookup() -> None:
    xstar = XSTAR(grid=LogEnergyGrid(0.01, 1000.0, 50))  # beyond ASTAR's 250 MeV/u
    with pytest.raises(OutOfRangeError):
        xstar.mass_stopping_power(2, 4, 10.0)


# --- not yet available: wired in later steps -------------------------------------------


def test_total_for_zeff_scaled_ions_adds_the_nuclear_model() -> None:
    e = np.array([0.01, 1.0])
    kw = {"heavy_ion_source": "pstar_zeff_scaled"}
    total = XSTAR(quantity="total", **kw).mass_stopping_power(6, 12, e)  # type: ignore[arg-type]
    el = XSTAR(**kw).mass_stopping_power(6, 12, e)  # type: ignore[arg-type]
    np.testing.assert_allclose(
        total,
        el + WaterNuclearStopping(proton_potential="moliere").mass_stopping_power(6, 12, e),
        rtol=1e-12,
    )


def test_total_for_other_isotopes_adds_the_nuclear_model() -> None:
    """The tabulated nuclear column belongs to p and He-4 only; a deuteron gets the model."""
    e = np.array([0.01, 1.0])
    total = XSTAR(quantity="total")
    el = XSTAR().mass_stopping_power(1, 2, e)
    nuc = WaterNuclearStopping(proton_potential="moliere").mass_stopping_power(1, 2, e)
    np.testing.assert_allclose(total.mass_stopping_power(1, 2, e), el + nuc, rtol=1e-12)
    assert "Table 4.1" in str(total.metadata(1, 2).options["nuclear"])
    assert total.metadata(2, 4).options["nuclear"] == "tabulated (He-4)"


def test_a_custom_nuclear_model_is_used() -> None:
    zbl = WaterNuclearStopping()
    e = np.array([0.01])
    got = XSTAR(quantity="total", nuclear=zbl).mass_stopping_power(1, 2, e)
    np.testing.assert_allclose(
        got, XSTAR().mass_stopping_power(1, 2, e) + zbl.mass_stopping_power(1, 2, e), rtol=1e-12
    )
