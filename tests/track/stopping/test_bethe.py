"""The Bethe stopping-power formula with switchable corrections (track/stopping/bethe.py).

Formula: PDG 2024 (Navas et al., Phys. Rev. D 110, 030001), eqs. (34.4), (34.5), and
the density effect, eq. (34.7) or NIST's tabulated delta(beta*gamma).
"""

from __future__ import annotations

import csv
import math
from collections.abc import Callable
from pathlib import Path

import numpy as np
import numpy.typing as npt
import pytest

from pyradbio.core.constants import BETHE_K_MEV_CM2_MOL, ELECTRON_MASS_MEV, WATER_Z_OVER_A
from pyradbio.track.stopping.barkas import WaterBarkasCorrection
from pyradbio.track.stopping.base import OutOfRangeError
from pyradbio.track.stopping.bethe import (
    Bethe,
    NoDensityEffect,
    SternheimerDensityEffect,
    TabulatedDensityEffect,
)
from pyradbio.track.stopping.higher_order import ahlen_mott_correction
from pyradbio.track.stopping.masses import nuclear_mass_MeV
from pyradbio.track.stopping.nuclear import WaterNuclearStopping
from pyradbio.track.stopping.shell import WaterShellCorrection

BANDS = ((5.0, 10.0), (10.0, 30.0), (30.0, 100.0), (100.0, 300.0), (300.0, 1e9))


def _plain_bethe(z: int, a: int, e_u: float, i_ev: float) -> float:
    """Eq. (34.5) with delta = 0, written out independently of the implementation."""
    m = nuclear_mass_MeV(z, a)
    gamma = 1.0 + e_u * a / m
    b2 = 1.0 - 1.0 / gamma**2
    bg2 = b2 * gamma**2
    w_max = (
        2
        * ELECTRON_MASS_MEV
        * bg2
        / (1 + 2 * gamma * ELECTRON_MASS_MEV / m + (ELECTRON_MASS_MEV / m) ** 2)
    )
    log_arg = 2 * ELECTRON_MASS_MEV * bg2 * w_max / (i_ev * 1e-6) ** 2
    return BETHE_K_MEV_CM2_MOL * WATER_Z_OVER_A * z * z / b2 * (0.5 * math.log(log_arg) - b2)


def _rows(path: Path) -> list[dict[str, str]]:
    return list(csv.DictReader(line for line in path.open() if not line.startswith("#")))


# --- the uncorrected formula -----------------------------------------------------------


@pytest.mark.parametrize(("z", "a"), [(1, 1), (2, 4), (6, 12), (18, 40)])
def test_uncorrected_limit_is_plain_bethe(z: int, a: int) -> None:
    bethe = Bethe(effective_charge=None, i_value_eV=75.0, density_effect="none")
    e = np.array([10.0, 100.0, 1000.0])
    np.testing.assert_allclose(
        bethe.mass_stopping_power(z, a, e), [_plain_bethe(z, a, x, 75.0) for x in e], rtol=1e-13
    )


def test_i_value_enters_only_the_logarithm() -> None:
    e = np.array([50.0])
    s75 = Bethe(i_value_eV=75.0, density_effect="none").mass_stopping_power(1, 1, e)[0]
    s78 = Bethe(i_value_eV=78.0, density_effect="none").mass_stopping_power(1, 1, e)[0]
    m = nuclear_mass_MeV(1, 1)
    gamma = 1 + 50.0 / m
    b2 = 1 - 1 / gamma**2
    expected = BETHE_K_MEV_CM2_MOL * WATER_Z_OVER_A / b2 * math.log(78.0 / 75.0)
    assert s75 - s78 == pytest.approx(expected, rel=1e-10)


def test_range_follows_beta_gamma_validity() -> None:
    """PDG: eq. (34.5) holds for 0.1 <~ beta*gamma <~ 1000 ("a few percent")."""
    bethe = Bethe(i_value_eV=78.0, density_effect="none")
    lo, hi = bethe.energy_range(1, 1)
    m = nuclear_mass_MeV(1, 1)
    assert lo == pytest.approx(m * (math.sqrt(1 + 0.1**2) - 1), rel=1e-12)
    assert hi == pytest.approx(m * (math.sqrt(1 + 1000.0**2) - 1), rel=1e-12)
    with pytest.raises(OutOfRangeError):
        bethe.mass_stopping_power(1, 1, lo * 0.99)


def test_range_bounds_are_options() -> None:
    bethe = Bethe(density_effect="none", min_beta_gamma=0.2, max_beta_gamma=10.0)
    m = nuclear_mass_MeV(6, 12) / 12
    lo, hi = bethe.energy_range(6, 12)
    assert lo == pytest.approx(m * (math.sqrt(1.04) - 1), rel=1e-12)
    assert hi == pytest.approx(m * (math.sqrt(101.0) - 1), rel=1e-12)
    with pytest.raises(ValueError, match="beta_gamma"):
        Bethe(min_beta_gamma=1.0, max_beta_gamma=0.5)


# --- the density effect ----------------------------------------------------------------


def test_density_effect_switch_changes_only_high_energies() -> None:
    """delta = 0 below beta*gamma = 10^x0 in an insulator; it lowers S only above."""
    with_d = Bethe(i_value_eV=78.0, density_effect="tabulated")
    without = Bethe(i_value_eV=78.0, density_effect="none")
    low = np.array([5.0, 20.0, 100.0])  # beta*gamma 0.10-0.47
    np.testing.assert_array_equal(
        with_d.mass_stopping_power(1, 1, low), without.mass_stopping_power(1, 1, low)
    )
    high = np.array([2000.0, 20000.0])
    assert np.all(with_d.mass_stopping_power(1, 1, high) < without.mass_stopping_power(1, 1, high))


def test_tabulated_delta_is_chosen_by_i_value() -> None:
    assert TabulatedDensityEffect.for_i_value(75.0).source == "ESTAR"
    assert TabulatedDensityEffect.for_i_value(78.0).source == "ICRU 90"
    with pytest.raises(ValueError, match="I = 80"):
        TabulatedDensityEffect.for_i_value(80.0)
    with pytest.raises(ValueError, match="tabulated"):
        Bethe(i_value_eV=80.0, density_effect="tabulated")
    assert Bethe(i_value_eV=80.0, density_effect="none").metadata(1, 1).i_value_eV == 80.0


def test_sternheimer_is_the_pdg_parameterisation() -> None:
    """Eq. (34.7) for an insulator, with the coefficients of the shipped table."""
    st = SternheimerDensityEffect.water()
    c = st.coefficients
    for x in (-0.5, c["x0"] + 0.1, 1.5, c["x1"] + 0.5):
        bg = np.array([10.0**x])
        if x < c["x0"]:
            expected = 0.0
        elif x < c["x1"]:
            expected = 2 * math.log(10) * x - c["Cbar"] + c["a"] * (c["x1"] - x) ** c["k"]
        else:
            expected = 2 * math.log(10) * x - c["Cbar"]
        assert st.delta(bg)[0] == pytest.approx(expected, abs=1e-12)
    assert st.i_value_eV == 75.0
    with pytest.raises(ValueError, match="75"):
        Bethe(i_value_eV=78.0, density_effect="sternheimer")


def test_sternheimer_coefficients_belong_to_75_ev() -> None:
    """Cbar = 2 ln(I / hbar*omega_p) + 1 identifies the I the coefficients were made for."""
    c = SternheimerDensityEffect.water().coefficients
    assert 2 * math.log(75.0 / c["plasma_energy_eV"]) + 1 == pytest.approx(c["Cbar"], abs=5e-5)


def test_tabulated_and_sternheimer_delta_agree_at_75_ev() -> None:
    """Two independent routes to delta for the same water (I = 75 eV).

    MEASURED over beta*gamma 0.5-1000: max |difference| 0.099, at the onset
    (beta*gamma ~ 2-4: 0.087 vs 0.041, 0.675 vs 0.592), within 0.09 from beta*gamma ~ 8
    and 0.003 at 1000 -- the parameterisation is a fit, the table the full calculation,
    hence "tabulated" as default. Bound ~2x measured (delta enters as delta/2).
    """
    bg = np.geomspace(0.5, 1000.0, 200)
    diff = np.abs(
        TabulatedDensityEffect.for_i_value(75.0).delta(bg)
        - SternheimerDensityEffect.water().delta(bg)
    )
    print(f"\n|delta_ESTAR - delta_Sternheimer| max {diff.max():.3e}")
    assert diff.max() < 0.2


def test_no_density_effect_is_zero() -> None:
    assert np.all(NoDensityEffect().delta(np.array([0.1, 10.0, 1e4])) == 0.0)


def test_tabulated_delta_reproduces_its_sources(
    raw_source_file: Callable[[str, str], Path],
) -> None:
    """Shipped (regridded) delta at the SOURCE energies, absolute deviation in delta."""
    for source, name, i_ev in (
        ("estar", "estar_water_liquid.csv", 75.0),
        ("icru90", "icru90_electron_water_liquid.csv", 78.0),
    ):
        rows = _rows(raw_source_file(source, name))
        t = np.array([float(r["T_MeV"]) for r in rows])
        delta = np.array([float(r["delta"]) for r in rows])
        bg = np.sqrt(t * (t + 2 * ELECTRON_MASS_MEV)) / ELECTRON_MASS_MEV
        got = TabulatedDensityEffect.for_i_value(i_ev).delta(bg)
        err = np.abs(got - delta).max()
        print(f"\n{source}: max |delta - source| = {err:.2e}")
        assert err < 2e-3


# --- against the tables ----------------------------------------------------------------


def _bands(e: npt.NDArray[np.float64], rel: npt.NDArray[np.float64]) -> list[str]:
    out = []
    for lo, hi in BANDS:
        m = (e >= lo) & (e < hi)
        if m.any():
            out.append(f"[{lo:g},{hi:g}) MeV/u max={np.abs(rel[m]).max():.2e}")
    return out


def test_high_energy_agreement_with_pstar_and_icru90_is_quantified(
    raw_source_file: Callable[[str, str], Path],
) -> None:
    """Bethe WITHOUT shell/Barkas/Bloch vs the evaluated tables (with them: further below).

    MEASURED, max |Bethe/table - 1| per band [5-10, 10-30, 30-100, 100-300,
    >300 MeV/u]: PSTAR 1.3%, 0.67%, 0.20%, 0.06%, 0.035%; ICRU 90 p 1.4%, 0.68%, 0.21%,
    0.05%, 0.03%; ICRU 90 alpha 1.4%, 0.73%, 0.17%, 0.07%; ICRU 90 carbon 3.5%, 1.2%,
    0.22%, 0.48%, 0.62%. The low-energy excess is the omitted shell/Barkas/Bloch terms;
    carbon's high-energy deficit (Bethe LOW, growing with energy) is consistent with a
    missing Mott-type z-dependent term (first-order pi alpha z beta / 2 / L ~ 0.6% at
    833 MeV/u) -- an inference until Mott is implemented. Bounds ~2x measured.
    """
    cases = (
        ("pstar", "pstar_water_liquid.csv", 75.0, 1, 1, 0.015, 0.0014),
        ("icru90", "icru90_proton_water_liquid.csv", 78.0, 1, 1, 0.015, 0.0014),
        ("icru90", "icru90_alpha_water_liquid.csv", 78.0, 2, 4, 0.015, 0.0014),
        ("icru90", "icru90_carbon_water_liquid.csv", 78.0, 6, 12, 0.025, 0.013),
    )
    for source, name, i_ev, z, a, bound_10, bound_100 in cases:
        rows = _rows(raw_source_file(source, name))
        e = np.array([float(r["T_MeV"]) for r in rows]) / a
        s = np.array([float(r["S_el_MeV_cm2_g"]) for r in rows])
        bethe = Bethe(effective_charge=None, i_value_eV=i_ev, density_effect="tabulated")
        lo, hi = bethe.energy_range(z, a)
        keep = (e >= max(lo, 5.0)) & (e <= hi)
        rel = bethe.mass_stopping_power(z, a, e[keep]) / s[keep] - 1.0
        print(f"\n{name} (I={i_ev}):", *_bands(e[keep], rel), sep="\n  ")
        above_10 = e[keep] >= 10.0
        assert np.abs(rel[above_10]).max() < bound_10
        above_100 = e[keep] >= 100.0
        assert np.abs(rel[above_100]).max() < bound_100


# --- metadata, interface, unavailable corrections --------------------------------------


def test_metadata_records_formula_i_value_and_switches() -> None:
    info = Bethe(i_value_eV=78.0).metadata(6, 12)
    assert info.i_value_eV == 78.0
    assert "34.5" in info.reference
    assert info.options["density_effect"] == "tabulated (ICRU 90)"
    assert info.options["shell"] is False
    assert info.options["projectile_charge"] == "effective: hubert1990 (Z2 = 8)"
    assert Bethe().metadata(2, 4).options["projectile_charge"] == "bare (z = Z)"
    bare = Bethe(effective_charge=None).metadata(6, 12)
    assert bare.options["projectile_charge"] == "bare (z = Z)"


def test_invalid_options_raise() -> None:
    with pytest.raises(ValueError, match="density_effect"):
        Bethe(density_effect="exact")  # type: ignore[arg-type]
    with pytest.raises(ValueError, match="i_value_eV"):
        Bethe(i_value_eV=-1.0)
    with pytest.raises(ValueError, match="quantity"):
        Bethe(quantity="nuclear")  # type: ignore[arg-type]


def test_total_adds_zbl_nuclear_stopping_for_every_ion() -> None:
    e = np.array([10.0, 100.0])
    zbl = WaterNuclearStopping()
    for z, a in ((1, 1), (6, 12)):
        total = Bethe(quantity="total").mass_stopping_power(z, a, e)
        el = Bethe().mass_stopping_power(z, a, e)
        np.testing.assert_allclose(total, el + zbl.mass_stopping_power(z, a, e), rtol=1e-12)
    info = Bethe(quantity="total").metadata(1, 1)
    assert info.options["quantity"] == "total"
    assert "protons: zbl" in str(info.options["nuclear"])


def test_a_custom_density_effect_model_is_accepted() -> None:
    class Constant:
        name = "constant 0.5"
        i_value_eV = None
        beta_gamma_range = (0.01, 1e5)

        def delta(self, beta_gamma: npt.NDArray[np.float64]) -> npt.NDArray[np.float64]:
            return np.full_like(beta_gamma, 0.5)

    e = np.array([100.0])
    shifted = Bethe(i_value_eV=78.0, density_effect=Constant()).mass_stopping_power(1, 1, e)
    plain = Bethe(i_value_eV=78.0, density_effect="none").mass_stopping_power(1, 1, e)
    m = nuclear_mass_MeV(1, 1)
    gamma = 1 + 100.0 / m
    b2 = 1 - 1 / gamma**2
    assert plain[0] - shifted[0] == pytest.approx(
        BETHE_K_MEV_CM2_MOL * WATER_Z_OVER_A / b2 * 0.25, rel=1e-10
    )


# --- shell and Barkas (track/stopping/shell.py, barkas.py) ------------------------------


def test_shell_and_barkas_shift_the_bracket_by_their_terms() -> None:
    """S = K (Z/A) z^2 / beta^2 [L0 - C/Z + z L1]: each switch adds exactly its term."""
    ref = Bethe(effective_charge=None, i_value_eV=78.0)
    shell, barkas = WaterShellCorrection(), WaterBarkasCorrection()
    for z, a, e in ((1, 1, 10.0), (6, 12, 20.0)):
        m = nuclear_mass_MeV(z, a)
        b2 = 1 - 1 / (1 + e * a / m) ** 2
        got = _bracket_shift(
            Bethe(effective_charge=None, i_value_eV=78.0, shell=True), ref, z, a, e
        )
        assert got == pytest.approx(-shell.c_over_z(b2)[0], rel=1e-9)
        got = _bracket_shift(
            Bethe(effective_charge=None, i_value_eV=78.0, barkas=True), ref, z, a, e
        )
        assert got == pytest.approx(z * barkas.l1(b2)[0], rel=1e-9)


def test_shell_and_barkas_are_recorded_and_bound_the_range() -> None:
    bethe = Bethe(shell=True, barkas=WaterBarkasCorrection(gamma=1.29), min_beta_gamma=0.01)
    info = bethe.metadata(1, 1)
    assert "Walske" in str(info.options["shell"])
    assert "gamma=1.29" in str(info.options["barkas"])
    b2_lo = WaterBarkasCorrection().beta2_range()[0]
    assert bethe.beta_gamma_range[0] == pytest.approx(math.sqrt(b2_lo / (1 - b2_lo)))
    with pytest.raises(OutOfRangeError):
        bethe.mass_stopping_power(1, 1, 0.1)


def test_icru49_corrections_reproduce_the_evaluated_tables(
    raw_source_file: Callable[[str, str], Path],
) -> None:
    """Bethe + shell + Barkas + Bloch vs PSTAR/ASTAR (ICRU 49, 75 eV) and ICRU 90 (78 eV)
    from 0.5 MeV/u (ICRU 49's own Bethe cut-off T2 for p and alpha) to 1000 MeV/u.

    MEASURED, max |S/table - 1| per band [0.5-1, 1-2, 2-5, 5-10, 10-30, 30-100,
    100-1000 MeV/u]: PSTAR 0.13, 0.09, 0.03, 0.03, 0.02, 0.04, 0.02%; ICRU 90 p 0.15, 0.11,
    0.03, 0.03, 0.01, 0.04, 0.02%; ASTAR 0.14, 0.06, 0.02, 0.04, 0.04, 0.02, 0.03%; ICRU 90
    alpha 0.14, 0.11, 0.02, 0.01, 0.02, 0.00, 0.03%. The two oxygen thetas were fitted to
    PSTAR (shell.py); ICRU 90 (other I) and both alpha tables (z-dependence) were not
    used. Without shell and Barkas: up to +2.6% (p) and -3.0% (alpha) here.
    """
    cases = (
        ("pstar", "pstar_water_liquid.csv", 75.0, 1, 1),
        ("icru90", "icru90_proton_water_liquid.csv", 78.0, 1, 1),
        ("astar", "astar_water_liquid.csv", 75.0, 2, 4),
        ("icru90", "icru90_alpha_water_liquid.csv", 78.0, 2, 4),
    )
    for source, name, i_ev, z, a in cases:
        rows = _rows(raw_source_file(source, name))
        e = np.array([float(r["T_MeV"]) for r in rows]) / a
        s = np.array([float(r["S_el_MeV_cm2_g"]) for r in rows])
        bethe = Bethe(i_value_eV=i_ev, shell=True, barkas=True, bloch=True, min_beta_gamma=0.02)
        keep = (e >= 0.5) & (e <= 1000.0)
        rel = bethe.mass_stopping_power(z, a, e[keep]) / s[keep] - 1.0
        print(f"\n{name} (I={i_ev}, shell+Barkas+Bloch):", *_bands(e[keep], rel), sep="\n  ")
        assert np.abs(rel).max() < 3e-3
        assert np.abs(rel[e[keep] >= 2.0]).max() < 1e-3


# --- Bloch, Mott, Lindhard-Sorensen (Phys. Rev. A 53, 2443 (1996)) -------------------


def _bracket_shift(bethe: Bethe, ref: Bethe, z: int, a: int, e: float) -> float:
    """Change of the Bethe bracket L: (S - S_ref) / (K (Z/A) z^2 / beta^2)."""
    m = nuclear_mass_MeV(z, a)
    gamma = 1 + e * a / m
    b2 = 1 - 1 / gamma**2
    k = BETHE_K_MEV_CM2_MOL * WATER_Z_OVER_A * z * z / b2
    return float((bethe.mass_stopping_power(z, a, e)[0] - ref.mass_stopping_power(z, a, e)[0]) / k)


def test_each_higher_order_switch_adds_exactly_its_term() -> None:
    from pyradbio.track.stopping.higher_order import (
        bloch_correction,
        lindhard_sorensen_correction,
        mott_correction,
    )

    plain = Bethe(effective_charge=None, density_effect="none")
    z, a, e = 6, 12, 200.0
    gamma = 1 + e * a / nuclear_mass_MeV(z, a)
    beta = np.array([math.sqrt(1 - 1 / gamma**2)])
    bloch = Bethe(effective_charge=None, density_effect="none", bloch=True)
    mott = Bethe(effective_charge=None, density_effect="none", mott=True)
    assert _bracket_shift(bloch, plain, z, a, e) == pytest.approx(
        bloch_correction(z, beta)[0], rel=1e-9
    )
    assert _bracket_shift(mott, plain, z, a, e) == pytest.approx(
        mott_correction(z, beta)[0], rel=1e-9
    )
    ls = Bethe(effective_charge=None, density_effect="none", lindhard_sorensen=True)
    assert _bracket_shift(ls, plain, z, a, e) == pytest.approx(
        lindhard_sorensen_correction(z, np.array([gamma]))[0], rel=1e-9
    )


def test_bloch_lowers_and_mott_raises_the_stopping() -> None:
    plain = Bethe(effective_charge=None, density_effect="none")
    assert (
        _bracket_shift(
            Bethe(effective_charge=None, density_effect="none", bloch=True), plain, 6, 12, 20.0
        )
        < 0
    )
    assert (
        _bracket_shift(
            Bethe(effective_charge=None, density_effect="none", mott=True), plain, 6, 12, 20.0
        )
        > 0
    )


def test_lindhard_sorensen_replaces_bloch_and_mott() -> None:
    with pytest.raises(ValueError, match="replaces"):
        Bethe(lindhard_sorensen=True, bloch=True)
    with pytest.raises(ValueError, match="replaces"):
        Bethe(lindhard_sorensen=True, mott=True)


def test_lindhard_sorensen_caps_the_energy_below_finite_nuclear_size() -> None:
    """LS p. 2451: finite size matters once gamma A^(1/3)/160 ~ 1; we stop at 0.1 (ASSUMPTION)."""
    _, hi = Bethe(lindhard_sorensen=True).energy_range(6, 12)
    m_u = nuclear_mass_MeV(6, 12) / 12
    gamma_max = 0.1 * 160 / 12 ** (1 / 3)
    assert hi == pytest.approx(m_u * (gamma_max - 1), rel=1e-12)
    assert Bethe().energy_range(6, 12)[1] > hi


def test_higher_order_switches_are_recorded() -> None:
    info = Bethe(lindhard_sorensen=True).metadata(8, 16)
    assert info.options["lindhard_sorensen"] is True
    assert info.options["bloch"] is False


# --- Sternheimer & Peierls 1971 (Phys. Rev. B 3, 3681) -----------------------------------


def test_sternheimer_peierls_recipe_for_liquids() -> None:
    """Eqs. (3), (5), (8), (9), (33)-(36): any I, water's plasma energy from eq. (5)."""
    from pyradbio.track.stopping.bethe import SternheimerPeierlsDensityEffect

    sp = SternheimerPeierlsDensityEffect(i_value_eV=78.0)
    hv_p = 28.8 * math.sqrt(1.0 * WATER_Z_OVER_A)
    cbar = 2 * math.log(78.0 / hv_p) + 1
    assert sp.parameters["Cbar"] == pytest.approx(cbar, rel=1e-12)
    assert (sp.parameters["x0"], sp.parameters["x1"], sp.parameters["m"]) == (0.2, 2.0, 3.0)
    high = SternheimerPeierlsDensityEffect(i_value_eV=90.0).parameters  # Cbar >= 3.681
    assert high["x0"] == pytest.approx(0.326 * high["Cbar"] - 1.0, rel=1e-12)
    heavy = SternheimerPeierlsDensityEffect(i_value_eV=500.0).parameters  # I >= 100 eV
    assert heavy["x1"] == 3.0
    assert heavy["x0"] == pytest.approx(0.326 * heavy["Cbar"] - 1.5, rel=1e-12)
    # delta(X0) = 0 (eq. 6) and continuity at X1
    for p in (sp.parameters, high, heavy):
        model = SternheimerPeierlsDensityEffect(i_value_eV=p["I_eV"])
        x0, x1 = p["x0"], p["x1"]
        assert model.delta(np.array([10**x0]))[0] == pytest.approx(0.0, abs=1e-12)
        below, above = model.delta(np.array([10 ** (x1 - 1e-9), 10 ** (x1 + 1e-9)]))
        assert below == pytest.approx(above, abs=1e-6)


def test_sternheimer_peierls_against_the_tabulated_delta() -> None:
    """MEASURED (beta*gamma 0.5-1000): max |difference| 0.33 (I=75) and 0.34
    (I=78), against 0.099 for PDG's fitted coefficients: the general recipe trades
    accuracy for generality, as its authors say (dE/dx errors < 2%, typically 1%;
    0.33/2 against a bracket of ~9 is ~1.8%). Bound ~2x measured."""
    from pyradbio.track.stopping.bethe import SternheimerPeierlsDensityEffect

    bg = np.geomspace(0.5, 1000.0, 200)
    for i_ev in (75.0, 78.0):
        diff = np.abs(
            SternheimerPeierlsDensityEffect(i_value_eV=i_ev).delta(bg)
            - TabulatedDensityEffect.for_i_value(i_ev).delta(bg)
        )
        print(f"\nI={i_ev}: |delta_SP - delta_table| max {diff.max():.3e}")
        assert diff.max() < 0.7


def test_sternheimer_peierls_works_at_any_i_value() -> None:
    bethe = Bethe(i_value_eV=80.0, density_effect="sternheimer_peierls")
    assert bethe.metadata(1, 1).options["density_effect"].startswith("sternheimer-peierls")
    assert bethe.mass_stopping_power(1, 1, 5000.0)[0] > 0


def test_ahlen_adds_exactly_half_g_and_excludes_mott_and_ls() -> None:
    ref = Bethe(effective_charge=None, i_value_eV=78.0, bloch=True)
    for z, a, e in ((6, 12, 100.0), (18, 40, 50.0)):
        m = nuclear_mass_MeV(z, a)
        beta = math.sqrt(1 - 1 / (1 + e * a / m) ** 2)
        got = _bracket_shift(
            Bethe(effective_charge=None, i_value_eV=78.0, bloch=True, ahlen=True), ref, z, a, e
        )
        assert got == pytest.approx(float(ahlen_mott_correction(z, np.array([beta]))[0]), rel=1e-9)
    with pytest.raises(ValueError, match="mott"):
        Bethe(ahlen=True, mott=True)
    with pytest.raises(ValueError, match="lindhard_sorensen"):
        Bethe(ahlen=True, lindhard_sorensen=True)
    assert Bethe(bloch=True, ahlen=True).metadata(6, 12).options["ahlen"] is True


def test_lindhard_sorensen_closes_the_carbon_gap_to_icru90(
    raw_source_file: Callable[[str, str], Path],
) -> None:
    """Above 30 MeV/u (no shell/Barkas needed), Bethe vs ICRU 90 carbon (I = 78 eV).

    MEASURED, max |S/S_ICRU90 - 1| over 30-833 MeV/u: plain 0.62%,
    Bloch + linear Mott 0.061%, Lindhard-Sorensen 0.059%, Bloch + Ahlen 0.058% -- ICRU 90's
    own choice (BEST + Ahlen's Mott term) is indistinguishable from LS here. ICRU 90's
    carbon therefore contains a Bloch/Mott-type term, while its protons and alphas do not
    include Mott (adding it worsens them, +0.1% at 300-800 MeV). Bound ~2x measured.
    """
    rows = _rows(raw_source_file("icru90", "icru90_carbon_water_liquid.csv"))
    e = np.array([float(r["T_MeV"]) for r in rows]) / 12
    s = np.array([float(r["S_el_MeV_cm2_g"]) for r in rows])
    for kw, bound in (
        ({"lindhard_sorensen": True}, 0.0012),
        ({"bloch": True, "mott": True}, 0.0012),
        ({"bloch": True, "ahlen": True}, 0.0012),
    ):
        bethe = Bethe(effective_charge=None, i_value_eV=78.0, **kw)  # type: ignore[arg-type]
        _, hi = bethe.energy_range(6, 12)
        keep = (e >= 30.0) & (e <= hi)
        rel = bethe.mass_stopping_power(6, 12, e[keep]) / s[keep] - 1.0
        assert np.abs(rel).max() < bound
