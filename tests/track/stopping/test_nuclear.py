"""Nuclear stopping in water, ICRU 49 sec. 4.3 + ICRU 90 footnote 35 (track/stopping/nuclear.py)."""

from __future__ import annotations

import csv
import math
from collections.abc import Callable
from pathlib import Path

import numpy as np
import pytest

from pyradbio.track.stopping.base import OutOfRangeError
from pyradbio.track.stopping.masses import atomic_mass_u, mean_atomic_mass_u
from pyradbio.track.stopping.nuclear import (
    ICRU49_PROTON_POTENTIAL,
    MOLIERE_SCREENING,
    S_NUC_COEFFICIENT,
    T_HAT_PER_KEV,
    WaterNuclearStopping,
    moliere_born_nuclear_stopping,
    reduced_nuclear_stopping,
    screening_a,
)


def _rows(path: Path) -> list[dict[str, str]]:
    with path.open(encoding="utf-8") as fh:
        return list(csv.DictReader(line for line in fh if not line.startswith("#")))


def test_shipped_table_reproduces_icru49_table_4_1(
    raw_source_file: Callable[[str, str], Path],
) -> None:
    """Regridded copy vs our transcription at the printed points. MEASURED: <= 1.9e-7."""
    rows = _rows(raw_source_file("icru49", "icru49_table41_reduced_nuclear.csv"))
    t = np.array([float(r["T_hat"]) for r in rows])
    for potential, column in (("zbl", "S_zbl"), ("moliere", "S_moliere")):
        expected = np.array([float(r[column]) for r in rows])
        np.testing.assert_allclose(reduced_nuclear_stopping(t, potential), expected, rtol=1e-6)


def test_zbl_column_agrees_with_icru49_fit_as_icru49_states() -> None:
    """ICRU 49 eq. (4.16) "accurate to within 1 percent for T_hat < 3" -- also checks that
    the two columns are not swapped (the Moliere column misses the fit by up to 30%).
    MEASURED: 0.87% (ZBL) below 3."""
    t = np.geomspace(1e-5, 2.9, 60)
    fit = 0.5 * np.log(1 + 1.1383 * t) / (t + 0.01321 * t**0.21226 + 0.1959 * t**0.5)
    assert np.abs(reduced_nuclear_stopping(t, "zbl") / fit - 1).max() < 0.01
    assert np.abs(reduced_nuclear_stopping(t, "moliere") / fit - 1).max() > 0.2


def test_eqs_4_14_and_4_15_with_bragg_additivity() -> None:
    """The model is S_nuc = sum_j w_j 5105.3 z Z_j / [a (1 + M_j/M) A_j] S_hat(T_hat_j)."""
    z, a_ion, e = 6, 12, 0.05
    m = atomic_mass_u(z, a_ion)
    t_kev = e * a_ion * 1e3
    m_h, m_o = mean_atomic_mass_u(1), mean_atomic_mass_u(8)
    expected = 0.0
    for zt, mt, w in ((1, m_h, 2 * m_h), (8, m_o, m_o)):
        a = zt**0.23 + z**0.23
        t_hat = T_HAT_PER_KEV * t_kev / (a * (1 + m / mt) * z * zt)
        s = S_NUC_COEFFICIENT * z * zt / (a * (1 + mt / m) * mt)
        expected += w / (2 * m_h + m_o) * s * reduced_nuclear_stopping(t_hat, "zbl")[0]
    got = WaterNuclearStopping().mass_stopping_power(z, a_ion, e)[0]
    assert got == pytest.approx(expected, rel=1e-12)


def test_water_molecular_weight_is_icru49s() -> None:
    """Mean atomic masses from the NIST isotope table give ICRU 49's 18.0152 (Table 3.1)."""
    assert 2 * mean_atomic_mass_u(1) + mean_atomic_mass_u(8) == pytest.approx(18.0152, abs=2e-4)


@pytest.mark.parametrize(
    ("source", "name", "z", "a", "potential", "max_t_mev", "bound"),
    [  # MEASURED: 0.070%, 0.070%, 0.060%, 0.100% (bounds ~2x)
        ("astar", "astar_water_liquid.csv", 2, 4, "zbl", 1e9, 1.5e-3),
        ("icru90", "icru90_alpha_water_liquid.csv", 2, 4, "zbl", 1e9, 1.5e-3),
        ("icru90", "icru90_carbon_water_liquid.csv", 6, 12, "zbl", 1e9, 1.5e-3),
        ("pstar", "pstar_water_liquid.csv", 1, 1, "moliere", 0.1, 2e-3),
    ],
)
def test_reproduces_the_tabulated_nuclear_columns(
    raw_source_file: Callable[[str, str], Path],
    source: str,
    name: str,
    z: int,
    a: int,
    potential: str,
    max_t_mev: float,
    bound: float,
) -> None:
    """ZBL + footnote 35 for He and C everywhere; Moliere protons below Z/10 MeV only
    (above, ICRU 49 used Moliere's quantum cross section, not implemented)."""
    rows = _rows(raw_source_file(source, name))
    t = np.array([float(r["T_MeV"]) for r in rows])
    s = np.array([float(r["S_nuc_MeV_cm2_g"]) for r in rows])
    keep = (s > 0) & (t < max_t_mev)
    model = WaterNuclearStopping(proton_potential=potential)  # type: ignore[arg-type]
    rel = model.mass_stopping_power(z, a, t[keep] / a) / s[keep] - 1.0
    assert np.abs(rel).max() < bound


PSTAR_NUCLEAR_BANDS = (
    # T band [MeV], bound on |S/S_PSTAR - 1| (~1.5-2x the measurement)
    ((0.1, 0.8), 0.045),  # measured -2.7%
    ((0.8, 3.0), 0.075),  # +5.0%: oxygen alpha_S 1.4-0.7, Born not valid there
    ((3.0, 10.0), 0.025),  # -1.4%
    ((10.0, 1e5), 0.04),  # -2.4%, flat up to 10 GeV
)


def test_moliere_protons_follow_icru49_classical_then_quantum(
    raw_source_file: Callable[[str, str], Path],
) -> None:
    """ICRU 49 sec. 4.3: classical below Z_t/10 MeV, Moliere's quantum cross section above
    (here in Born approximation). MEASURED per band (see the bounds); the
    classical-only branch was +5% at 1-100 MeV and -19% at 10 GeV. Nuclear stopping is
    < 0.1% of the total in this range, so the worst band is <= 4e-5 of S_tot."""
    rows = _rows(raw_source_file("pstar", "pstar_water_liquid.csv"))
    t = np.array([float(r["T_MeV"]) for r in rows])
    s_nuc = np.array([float(r["S_nuc_MeV_cm2_g"]) for r in rows])
    s_tot = np.array([float(r["S_tot_MeV_cm2_g"]) for r in rows])
    keep = (t >= 0.1) & (s_nuc > 0)
    t, s_nuc, s_tot = t[keep], s_nuc[keep], s_tot[keep]
    got = WaterNuclearStopping(proton_potential="moliere").mass_stopping_power(1, 1, t)
    rel = got / s_nuc - 1
    for (lo, hi), bound in PSTAR_NUCLEAR_BANDS:
        band = (t >= lo) & (t < hi)
        assert band.any()
        assert np.abs(rel[band]).max() < bound, (lo, hi)
    assert np.abs((got - s_nuc) / s_tot).max() < 1e-4
    classical = WaterNuclearStopping(proton_potential="moliere", moliere_quantum=False)
    high = t > 5e3
    assert np.abs(classical.mass_stopping_power(1, 1, t[high]) / s_nuc[high] - 1).min() > 0.1


def test_moliere_born_is_the_analytic_screened_rutherford_integral() -> None:
    """Independent check of one target atom: S = (pi hbar^2/M_t) int q^3 |F|^2 dq with the
    Born amplitude of Moliere's three-exponential potential, integrated numerically."""
    from scipy import integrate

    from pyradbio.core.constants import (
        ATOMIC_MASS_UNIT_MEV,
        AVOGADRO_PER_MOL,
        BOHR_RADIUS_CM,
        ELECTRON_MASS_MEV,
        HARTREE_MEV,
    )

    t_mev, zt = 20.0, 8
    mt = mean_atomic_mass_u(zt)
    me_u = ELECTRON_MASS_MEV / ATOMIC_MASS_UNIT_MEV
    m = atomic_mass_u(1, 1) - me_u
    mu = m * mt / (m + mt) / me_u
    beta_nr = math.sqrt(2 * t_mev / (m * ATOMIC_MASS_UNIT_MEV))
    gamma = 1 + t_mev / (m * ATOMIC_MASS_UNIT_MEV)
    q_max = 2 * mu * gamma * math.sqrt(1 - 1 / gamma**2) * 137.035999084
    alpha_s = zt / (137.035999084 * beta_nr)
    r_s = 0.88534 * zt ** (-1 / 3)

    def integrand(lq: float) -> float:
        q = math.exp(lq)
        f = 2 * alpha_s * sum(a / (q * q + (b / r_s) ** 2) for a, b in MOLIERE_SCREENING)
        return q**4 * f * f

    val = integrate.quad(integrand, math.log(1e-6), math.log(q_max), limit=400)[0]
    expected = math.pi / (mt / me_u) * val * HARTREE_MEV * BOHR_RADIUS_CM**2 * AVOGADRO_PER_MOL / mt
    got = moliere_born_nuclear_stopping(1, 1, zt, mt, [t_mev])[0]
    assert got == pytest.approx(expected, rel=1e-8)


def test_proton_potential_option_affects_protons_only() -> None:
    e = np.array([0.001, 0.01])
    zbl, mol = WaterNuclearStopping(), WaterNuclearStopping(proton_potential="moliere")
    assert ICRU49_PROTON_POTENTIAL == "moliere"
    assert not np.allclose(zbl.mass_stopping_power(1, 1, e), mol.mass_stopping_power(1, 1, e))
    np.testing.assert_array_equal(
        zbl.mass_stopping_power(2, 4, e), mol.mass_stopping_power(2, 4, e)
    )
    assert screening_a(1, 8, "moliere") == pytest.approx(8 ** (1 / 3))
    assert screening_a(6, 8, "zbl") == pytest.approx(6**0.23 + 8**0.23)
    assert "protons: moliere" in mol.describe()


def test_heavier_isotopes_of_the_same_velocity_stop_more() -> None:
    """Nuclear stopping depends on the projectile mass, not only on its velocity."""
    nuc = WaterNuclearStopping()
    assert nuc.mass_stopping_power(1, 2, 0.01)[0] > nuc.mass_stopping_power(1, 1, 0.01)[0]


def test_range_ends_and_high_energy_continuation() -> None:
    with pytest.raises(OutOfRangeError):
        reduced_nuclear_stopping([1e-6])
    hi = 1e8
    s_hi = reduced_nuclear_stopping([hi])[0]
    assert reduced_nuclear_stopping([10 * hi])[0] == pytest.approx(s_hi / 10)
    # the table's own last decade: S_hat * T_hat is constant to 0.12% (4-digit printing)
    t = np.array([1e7, 1.5e7, 2e7, 3e7, 4e7, 5e7, 6e7, 8e7, 1e8])
    st = reduced_nuclear_stopping(t) * t
    assert np.ptp(st) / st.mean() < 1.5e-3
    assert math.isfinite(WaterNuclearStopping().mass_stopping_power(1, 1, 5e5)[0])


def test_invalid_options_raise() -> None:
    with pytest.raises(ValueError, match="proton_potential"):
        WaterNuclearStopping(proton_potential="tf")  # type: ignore[arg-type]
    with pytest.raises(ValueError, match="potential"):
        reduced_nuclear_stopping([1.0], "tf")  # type: ignore[arg-type]
