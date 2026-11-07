"""Hubert, Bimbot & Gauvin (1990) alpha scaling for heavy ions (track/stopping/hubert.py)."""

from __future__ import annotations

import csv
from collections.abc import Callable
from pathlib import Path

import numpy as np
import pytest

from pyradbio.core.radiation import Ion
from pyradbio.track.stopping import ICRU, XSTAR, Bethe, ICRU90Procedure
from pyradbio.track.stopping.base import OutOfRangeError, UnsupportedIonError
from pyradbio.track.stopping.effective_charge import HubertEffectiveCharge
from pyradbio.track.stopping.hubert import HUBERT_MIN_ENERGY_MEV_U, HubertAlphaScaling
from pyradbio.track.stopping.masses import most_abundant_mass_number, nuclear_mass_MeV
from pyradbio.track.stopping_power import SimpleBetheStoppingPower, default_stopping_power


def test_is_eq_1_with_the_alpha_at_equal_velocity() -> None:
    alpha = ICRU90Procedure()
    model = HubertAlphaScaling(alpha=alpha)
    z, a, e = 26, 56, np.array([5.0, 50.0])
    gamma = 1 + e * a / nuclear_mass_MeV(z, a)
    beta = np.sqrt(1 - 1 / gamma**2)
    e_alpha = (gamma - 1) * nuclear_mass_MeV(2, 4) / 4
    expected = (HubertEffectiveCharge().z_eff(z, beta) / 2) ** 2 * alpha.mass_stopping_power(
        2, 4, e_alpha
    )
    np.testing.assert_allclose(model.mass_stopping_power(z, a, e), expected, rtol=1e-12)


#: Worst |S/S_ref - 1| per band [2.5-10, 10-50, 50-300 MeV/u], ~1.5x the measured
#: value (vs MSTAR; vs ICRU 73 errata all within 4.7%).
HUBERT_VS_MSTAR = {
    3: (0.04, 0.02, 0.016),
    6: (0.035, 0.03, 0.018),
    8: (0.05, 0.03, 0.02),
    10: (0.03, 0.03, 0.02),
    14: (0.11, 0.022, 0.022),
    18: (0.10, 0.023, 0.023),
}


@pytest.mark.parametrize("z", sorted(HUBERT_VS_MSTAR))
def test_light_and_medium_ions_against_mstar_and_icru73(z: int) -> None:
    """The validation available in water (no reference above Ar is held)."""
    a = most_abundant_mass_number(z)
    model, mstar, icru73 = HubertAlphaScaling(), XSTAR(), ICRU(edition="2009")
    for (lo, hi), bound in zip(((2.5, 10), (10, 50), (50, 300)), HUBERT_VS_MSTAR[z], strict=True):
        e = np.geomspace(lo, hi, 15)
        s = model.mass_stopping_power(z, a, e)
        assert np.abs(s / mstar.mass_stopping_power(z, a, e) - 1).max() < bound
        assert np.abs(s / icru73.mass_stopping_power(z, a, e) - 1).max() < 0.07


def test_icru90_carbon(raw_source_file: Callable[[str, str], Path]) -> None:
    """MEASURED: 1.2% over 2.5-300 MeV/u."""
    with raw_source_file("icru90", "icru90_carbon_water_liquid.csv").open() as fh:
        rows = list(csv.DictReader(line for line in fh if not line.startswith("#")))
    e = np.array([float(r["T_MeV"]) for r in rows]) / 12
    s = np.array([float(r["S_el_MeV_cm2_g"]) for r in rows])
    keep = (e >= 2.5) & (e <= 300)
    rel = HubertAlphaScaling().mass_stopping_power(6, 12, e[keep]) / s[keep] - 1
    assert np.abs(rel).max() < 0.018


def test_range_ions_and_metadata() -> None:
    model = HubertAlphaScaling()
    lo, hi = model.energy_range(92, 238)
    assert lo == HUBERT_MIN_ENERGY_MEV_U and hi > 1e4
    with pytest.raises(OutOfRangeError):
        model.mass_stopping_power(92, 238, 1.0)
    with pytest.raises(UnsupportedIonError):
        model.mass_stopping_power(2, 4, 10.0)
    info = model.metadata(54, 132)
    assert info.source.startswith("Hubert") and "hubert1990" in str(
        info.options["effective_charge"]
    )
    with pytest.raises(ValueError, match="min_energy"):
        HubertAlphaScaling(min_energy_MeV_u=0.0)


def test_range_is_bounded_by_the_alpha_source_at_equal_velocity() -> None:
    """An alpha source starting above 2.5 MeV/u moves the range start; no silent clamp.

    Regression: with ``alpha=Bethe()`` (from 4.65 MeV/u) Fe-56 at 2.5 MeV/u must raise
    rather than receive the alpha value at 4.65 MeV/u.
    """
    alpha = Bethe()
    alpha_lo, alpha_hi = alpha.energy_range(2, 4)
    assert alpha_lo > HUBERT_MIN_ENERGY_MEV_U
    model = HubertAlphaScaling(alpha=alpha)
    lo, hi = model.energy_range(26, 56)
    gamma = 1 + lo * 56 / nuclear_mass_MeV(26, 56)
    assert (gamma - 1) * nuclear_mass_MeV(2, 4) / 4 == pytest.approx(alpha_lo, rel=1e-12)
    gamma = 1 + hi * 56 / nuclear_mass_MeV(26, 56)
    assert (gamma - 1) * nuclear_mass_MeV(2, 4) / 4 == pytest.approx(alpha_hi, rel=1e-12)
    with pytest.raises(OutOfRangeError):
        model.mass_stopping_power(26, 56, HUBERT_MIN_ENERGY_MEV_U)
    model.mass_stopping_power(26, 56, [lo, hi])  # both ends evaluate (rounding only)
    assert HubertAlphaScaling().energy_range(26, 56)[0] == HUBERT_MIN_ENERGY_MEV_U


# --- the default stopping power's three branches ------------------------------------------


def test_default_routes_heavy_ions_to_hubert_from_2_5_mev_u() -> None:
    sp = default_stopping_power()
    assert sp.source(Ion(6, 12, 100.0)) == "icru90_procedure"
    assert sp.source(Ion(26, 56, 100.0)) == "hubert1990"
    assert sp.source(Ion(92, 238, 2.5)) == "hubert1990"
    assert sp.source(Ion(92, 238, 2.0)) == "simple_bethe_anchored"  # below Hubert's range
    ion = Ion(54, 132, 20.0)
    assert sp.let_keV_um(ion) == pytest.approx(HubertAlphaScaling().let_keV_um(ion), rel=1e-12)


def _anchored_bethe(z: int, a: int, e: float) -> float:
    """SimpleBethe scaled to Hubert at the 2.5 MeV/u seam."""
    bethe = SimpleBetheStoppingPower()
    anchor = HubertAlphaScaling().let_keV_um(Ion(z, a, 2.5)) / bethe.let_keV_um(Ion(z, a, 2.5))
    return anchor * bethe.let_keV_um(Ion(z, a, e))


def test_below_the_seam_is_simple_bethe_anchored_to_hubert() -> None:
    sp = default_stopping_power()
    assert sp.let_keV_um(Ion(26, 56, 1.0)) == pytest.approx(_anchored_bethe(26, 56, 1.0), rel=1e-12)


def test_default_switches_branch_within_one_call() -> None:
    sp = default_stopping_power()
    lets = sp.lets_keV_um(26, 56, [1.0, 10.0])
    assert lets[0] == pytest.approx(_anchored_bethe(26, 56, 1.0), rel=1e-12)
    assert lets[1] == pytest.approx(HubertAlphaScaling().let_keV_um(Ion(26, 56, 10.0)), rel=1e-12)


#: Bound on |d ln S/d ln E (above) - d ln S/d ln E (below)| at the seam, ~1.5x the
#: measurement (K 0.012, Fe 0.006, Kr 0.009, Xe 0.005, Pb 0.052, U 0.085).
SEAM_SLOPE_KINK = {19: 0.018, 26: 0.009, 36: 0.013, 54: 0.008, 82: 0.078, 92: 0.128}


@pytest.mark.parametrize("z", sorted(SEAM_SLOPE_KINK))
def test_default_is_continuous_at_the_2_5_mev_u_seam(z: int) -> None:
    """A hard switch would jump +5.8% (K) to +21.9% (Xe); the default must not.

    Below the seam the default is SimpleBethe x S_Hubert(2.5)/S_SimpleBethe(2.5), so the
    value is continuous and only the slope kinks (bounded here).
    """
    a = most_abundant_mass_number(z)
    sp = default_stopping_power()
    h = 1e-4
    e = 2.5 * np.array([1 - 2 * h, 1 - h, 1.0, 1 + h, 1 + 2 * h])
    s = np.log(sp.lets_keV_um(z, a, e))
    below = (s[1] - s[0]) / np.log((1 - h) / (1 - 2 * h))
    above = (s[4] - s[3]) / np.log((1 + 2 * h) / (1 + h))
    from_below = s[1] - below * np.log(1 - h)  # ln S extrapolated to 2.5 MeV/u
    from_above = s[3] - above * np.log(1 + h)
    assert abs(from_below - s[2]) < 1e-6 and abs(from_above - s[2]) < 1e-6  # no step
    assert abs(above - below) < SEAM_SLOPE_KINK[z]


def test_masses_ship_up_to_uranium() -> None:
    assert most_abundant_mass_number(26) == 56
    assert most_abundant_mass_number(92) == 238
    assert nuclear_mass_MeV(92, 238) == pytest.approx(221695.89, rel=1e-6)
