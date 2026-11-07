"""Effective-charge models used to scale stopping powers between ions."""

from __future__ import annotations

import csv
import math
from collections.abc import Callable
from pathlib import Path

import numpy as np
import pytest

from pyradbio.track import stopping_power
from pyradbio.track.stopping.effective_charge import (
    BarkasEffectiveCharge,
    EffectiveChargeModel,
    barkas_effective_charge,
)


def test_barkas_is_the_published_form() -> None:
    """Barkas 1963: z_eff = Z (1 - exp(-125 beta Z^(-2/3)))."""
    for z, beta in ((1, 0.01), (6, 0.05), (18, 0.3)):
        expected = z * (1.0 - math.exp(-125.0 * beta * z ** (-2.0 / 3.0)))
        assert barkas_effective_charge(z, beta) == expected


def test_the_old_name_is_the_same_function() -> None:
    """Katz, MCDS and the lineal-energy engines import it from stopping_power."""
    assert stopping_power.effective_charge is barkas_effective_charge


def test_barkas_model_vectorises_and_agrees_with_the_scalar_form() -> None:
    model = BarkasEffectiveCharge()
    beta = np.array([0.001, 0.01, 0.1, 0.5])
    np.testing.assert_allclose(
        model.z_eff(6, beta), [barkas_effective_charge(6, b) for b in beta], rtol=1e-15
    )
    assert model.name == "barkas"


def test_barkas_tends_to_the_bare_charge_at_high_velocity() -> None:
    # exp(-125 * 0.9 * 18^(-2/3)) = 7.6e-8: argon at beta = 0.9 is bare to 1e-7.
    assert BarkasEffectiveCharge().z_eff(18, np.array([0.9]))[0] == pytest.approx(18.0, rel=1e-7)


def test_a_custom_model_satisfies_the_protocol() -> None:
    class Bare:
        name = "bare"

        def z_eff(self, Z: int, beta: np.ndarray) -> np.ndarray:
            return np.full_like(np.asarray(beta, dtype=float), float(Z))

    model: EffectiveChargeModel = Bare()
    assert model.z_eff(3, np.array([0.1]))[0] == 3.0


# --- stopping-power effective charges ----------------------------------------------------


def test_hubert_is_eqs_2_to_9_with_oxygen() -> None:
    """Hubert, Bimbot & Gauvin 1990 eqs. (2)-(9), independently evaluated at one point."""
    from pyradbio.core.constants import ATOMIC_MASS_UNIT_MEV
    from pyradbio.track.stopping.effective_charge import HubertEffectiveCharge

    z, beta, z2 = 14, 0.2, 8.0
    e = (1 / math.sqrt(1 - beta**2) - 1) * ATOMIC_MASS_UNIT_MEV
    d = 1.164 + 0.2319 * math.exp(-0.004302 * z2)
    x1 = d + 1.658 * math.exp(-0.05170 * z)
    x2 = 8.144 + 0.09876 * math.log(z2)
    x3 = 0.3140 + 0.01072 * math.log(z2)
    x4 = 0.5218 + 0.02521 * math.log(z2)
    gamma = 1 - x1 * math.exp(-x2 * e**x3 * z ** (-x4))
    model = HubertEffectiveCharge()
    assert model.reference_Z == 2
    assert model.z_eff(z, np.array([beta]))[0] == pytest.approx(gamma * z, rel=1e-12)


def test_shipped_brandt_kitagawa_table_reproduces_table_ii(
    raw_source_file: Callable[[str, str], Path],
) -> None:
    from pyradbio.track.stopping.effective_charge import brandt_kitagawa_ionization

    with raw_source_file("brandt1982", "brandt1982_table2.csv").open(encoding="utf-8") as fh:
        rows = list(csv.DictReader(line for line in fh if not line.startswith("#")))
    y = np.array([float(r["y_r"]) for r in rows])
    q = np.array([float(r["q"]) for r in rows])
    # MEASURED: <= 2.4e-5 absolute (1.3e-4 relative) -- the runtime re-interpolates our
    # 50-per-decade copy; Table II itself is printed to +-5e-4.
    np.testing.assert_allclose(brandt_kitagawa_ionization(y), q, rtol=0, atol=5e-5)


def test_brandt_kitagawa_limits() -> None:
    """Eq. (23): zeta -> 1 for a fully stripped ion; ionization above Table II -> 1."""
    from pyradbio.track.stopping.effective_charge import (
        BrandtKitagawaEffectiveCharge,
        brandt_kitagawa_ionization,
    )

    q = brandt_kitagawa_ionization([5.932, 20.0, 200.0])
    assert q[0] == pytest.approx(0.981) and q[0] < q[1] < q[2] < 1.0
    with pytest.raises(ValueError, match="Table II"):
        brandt_kitagawa_ionization([0.01])
    zeta = BrandtKitagawaEffectiveCharge().z_eff(6, np.array([0.1, 0.5, 0.9])) / 6
    assert np.all(np.diff(zeta) > 0) and 0.9 < zeta[0] < zeta[-1] <= 1.0


def test_fitted_barkas_is_barkas_with_the_fitted_coefficient() -> None:
    from pyradbio.track.stopping.effective_charge import FITTED_BARKAS_COEFFICIENT

    b = np.array([0.1, 0.2])
    fitted = BarkasEffectiveCharge(coefficient=FITTED_BARKAS_COEFFICIENT)
    np.testing.assert_allclose(
        fitted.z_eff(10, b), 10 * (1 - np.exp(-FITTED_BARKAS_COEFFICIENT * b * 10 ** (-2 / 3)))
    )
    assert "162" in fitted.name and BarkasEffectiveCharge().name == "barkas"


EFFECTIVE_CHARGE_BOUNDS = {
    # option: (worst |dev| vs MSTAR over Z 3-18, vs ICRU 73 errata, ICRU 90 carbon), ~1.5x
    # the measurement with Bethe(shell, Barkas, Bloch, Ahlen) at 4.7-300 MeV/u
    None: (0.25, 0.20, 0.0010),  # 22.8 / 16.2 / 0.04%
    "hubert": (0.075, 0.07, 0.016),  # 5.1 / 4.6 / 1.08%
    "fitted": (0.05, 0.08, 0.019),  # 3.3 / 5.5 / 1.28% (MSTAR is its fit target)
    "brandt_kitagawa": (0.18, 0.25, 0.11),  # 12.1 / 16.9 / 7.6%
    "barkas": (0.20, 0.27, 0.06),  # 13.7 / 18.3 / 4.0%
}


@pytest.mark.parametrize("option", list(EFFECTIVE_CHARGE_BOUNDS), ids=str)
def test_bethe_effective_charge_options_measured_against_mstar_icru73_icru90(
    option: str | None, raw_source_file: Callable[[str, str], Path]
) -> None:
    from pyradbio.track.stopping import ICRU, XSTAR, Bethe
    from pyradbio.track.stopping.masses import most_abundant_mass_number

    bethe = Bethe(
        i_value_eV=78.0,
        shell=True,
        barkas=True,
        bloch=True,
        ahlen=True,
        effective_charge=option,  # type: ignore[arg-type]
    )
    mstar, icru73 = XSTAR(), ICRU(edition="2009")
    worst_ms = worst_73 = 0.0
    for z in (3, 6, 8, 10, 14, 18):
        a = most_abundant_mass_number(z)
        lo, _ = bethe.energy_range(z, a)
        e = np.geomspace(max(lo, 4.7), 300.0, 40)
        s = bethe.mass_stopping_power(z, a, e)
        worst_ms = max(worst_ms, float(np.abs(s / mstar.mass_stopping_power(z, a, e) - 1).max()))
        worst_73 = max(worst_73, float(np.abs(s / icru73.mass_stopping_power(z, a, e) - 1).max()))
    with raw_source_file("icru90", "icru90_carbon_water_liquid.csv").open() as fh:
        rows = list(csv.DictReader(line for line in fh if not line.startswith("#")))
    ec = np.array([float(r["T_MeV"]) for r in rows]) / 12
    sc = np.array([float(r["S_el_MeV_cm2_g"]) for r in rows])
    lo, hi = bethe.energy_range(6, 12)
    keep = (ec >= lo) & (ec <= hi)
    worst_c = float(np.abs(bethe.mass_stopping_power(6, 12, ec[keep]) / sc[keep] - 1).max())
    b_ms, b_73, b_c = EFFECTIVE_CHARGE_BOUNDS[option]
    assert worst_ms < b_ms and worst_73 < b_73 and worst_c < b_c


def test_bethe_default_is_hubert_and_leaves_light_ions_bare() -> None:
    from pyradbio.track.stopping import Bethe

    e = np.array([10.0, 100.0])
    for z, a in ((1, 1), (2, 4)):
        np.testing.assert_array_equal(
            Bethe().mass_stopping_power(z, a, e),
            Bethe(effective_charge=None).mass_stopping_power(z, a, e),
        )
    assert (
        Bethe().mass_stopping_power(18, 40, [10.0])[0]
        < Bethe(effective_charge=None).mass_stopping_power(18, 40, [10.0])[0]
    )
    with pytest.raises(ValueError, match="effective_charge"):
        Bethe(effective_charge="ziegler")  # type: ignore[arg-type]
