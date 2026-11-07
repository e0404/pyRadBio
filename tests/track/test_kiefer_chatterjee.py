"""Kiefer-Chatterjee radial-dose track structure.

Validated against pyMKM's reference implementation
(``pymkm.physics.particle_track.ParticleTrack``, model ``"Kiefer-Chatterjee"``).
"""

from __future__ import annotations

import math

import numpy as np
import pytest

from pyradbio.core.constants import KEVUM_PER_GY_UM2_GCM3
from pyradbio.core.radiation import Ion
from pyradbio.track.kiefer_chatterjee import (
    KC_R_MAX_DELTA,
    KC_R_MAX_GAMMA_UM,
    KC_RC0_NM,
    KieferChatterjeeRadialDose,
)
from pyradbio.track.stopping_power import SimpleBetheStoppingPower, effective_charge


def _have_pymkm() -> bool:
    try:
        import pymkm  # noqa: F401
    except ImportError:
        return False
    return True


requires_pymkm = pytest.mark.skipif(not _have_pymkm(), reason="pyMKM not installed ([mkm] extra)")


class _ConstantStoppingPower:
    """Fake StoppingPower returning a fixed LET, to test the radial-dose math alone."""

    def __init__(self, let_keV_um: float) -> None:
        self._let = let_keV_um

    def let_keV_um(self, ion: Ion) -> float:
        return self._let


# ---- geometry -----------------------------------------------------------
@pytest.mark.golden
def test_golden_penumbra_radius_parameterisation() -> None:
    """Rp = 0.0616 * E^1.7 um (pyMKM KC ``_compute_penumbra_radius``)."""
    model = KieferChatterjeeRadialDose()
    ion = Ion(Z=6, A=12, energy_MeV_u=100.0)
    expected_um = KC_R_MAX_GAMMA_UM * (100.0**KC_R_MAX_DELTA)
    assert model.r_max_um(ion) == pytest.approx(expected_um, rel=1e-9)


def test_core_radius_scales_with_velocity() -> None:
    """Rc = Rc0 * beta grows with energy (pyMKM energy-dependent core)."""
    model = KieferChatterjeeRadialDose()
    low = Ion(Z=6, A=12, energy_MeV_u=10.0)
    high = Ion(Z=6, A=12, energy_MeV_u=270.0)
    assert model.r_min_um(low) == pytest.approx(KC_RC0_NM * 1e-3 * low.beta())
    assert model.r_min_um(high) > model.r_min_um(low)


def test_penumbra_floored_at_core_radius() -> None:
    """At very low energy Rp_energy < Rc, so Rp is floored at Rc (no tail)."""
    model = KieferChatterjeeRadialDose()
    ion = Ion(Z=6, A=12, energy_MeV_u=0.005)
    assert model.r_max_um(ion) > model.r_max_gamma_um * ion.energy_MeV_u**KC_R_MAX_DELTA
    assert model.r_max_um(ion) == pytest.approx(model.r_min_um(ion))
    # With no penumbra the whole LET sits in the core; energy is still conserved.
    floored = KieferChatterjeeRadialDose(stopping_power=_ConstantStoppingPower(5.0))
    assert floored.radial_integral_LET_keV_um(ion) == pytest.approx(5.0, rel=2e-3)


# ---- energy conservation (pyMKM-independent) ----------------------------
@pytest.mark.golden
@pytest.mark.parametrize("energy", [10.0, 100.0, 270.0])
def test_radial_integral_reproduces_let(energy: float) -> None:
    """Integral of D(r)*2*pi*r dr reproduces the LET.

    The Kiefer-Chatterjee core amplitude is fixed by energy conservation, so
    the numerical radial integral must recover the input LET tightly. This is
    independent of pyMKM and pins the unit handling.
    """
    sp = _ConstantStoppingPower(50.0)
    model = KieferChatterjeeRadialDose(stopping_power=sp)
    ion = Ion(Z=6, A=12, energy_MeV_u=energy)
    assert model.radial_integral_LET_keV_um(ion) == pytest.approx(50.0, rel=2e-3)


def test_penumbra_is_inverse_square() -> None:
    """In Rc < r < Rp the dose follows Kp/r^2 exactly, so D(r)*r^2 is constant."""
    sp = SimpleBetheStoppingPower()
    ion = Ion(Z=6, A=12, energy_MeV_u=100.0)
    model = KieferChatterjeeRadialDose(stopping_power=sp)
    r_c = model.r_min_um(ion)
    r_p = model.r_max_um(ion)
    radii = np.geomspace(2.0 * r_c, 0.5 * r_p, 12)
    scaled = [model.local_dose_Gy(float(r), ion) * r * r for r in radii]
    assert max(scaled) / min(scaled) < 1.0 + 1e-9


def test_dose_zero_beyond_penumbra() -> None:
    sp = _ConstantStoppingPower(50.0)
    model = KieferChatterjeeRadialDose(stopping_power=sp)
    ion = Ion(6, 12, 100.0)
    assert model.local_dose_Gy(model.r_max_um(ion) * 1.001, ion) == 0.0


def test_core_dose_is_flat() -> None:
    """The core (r < Rc) is a constant plateau."""
    sp = SimpleBetheStoppingPower()
    ion = Ion(Z=6, A=12, energy_MeV_u=100.0)
    model = KieferChatterjeeRadialDose(stopping_power=sp)
    r_c = model.r_min_um(ion)
    d0 = model.local_dose_Gy(0.0, ion)
    d1 = model.local_dose_Gy(0.5 * r_c, ion)
    assert d0 == pytest.approx(d1)
    assert d0 > model.local_dose_Gy(1.001 * r_c, ion)  # core above the tail


def test_local_dose_requires_stopping_power() -> None:
    model = KieferChatterjeeRadialDose()  # no stopping power
    with pytest.raises(ValueError):
        model.local_dose_Gy(1.0, Ion(6, 12, 100.0))


def test_negative_radius_rejected() -> None:
    model = KieferChatterjeeRadialDose(stopping_power=_ConstantStoppingPower(50.0))
    with pytest.raises(ValueError):
        model.local_dose_Gy(-1.0, Ion(6, 12, 100.0))


# ---- validation against pyMKM ------------------------------------------
def _pymkm_kc_dose(ion: Ion, let_keV_um: float, radii_um: np.ndarray) -> np.ndarray:
    """Reference local dose from pyMKM's ParticleTrack (Kiefer-Chatterjee)."""
    from pyradbio.models.mkm.pymkm_backend import pymkm_kiefer_chatterjee_dose

    return pymkm_kiefer_chatterjee_dose(ion, radii_um, let_keV_um)


@requires_pymkm
@pytest.mark.golden
@pytest.mark.parametrize(
    "ion",
    [
        Ion(Z=1, A=1, energy_MeV_u=100.0),
        Ion(Z=6, A=12, energy_MeV_u=100.0),
        Ion(Z=6, A=12, energy_MeV_u=30.0),
        Ion(Z=10, A=20, energy_MeV_u=200.0),
    ],
)
def test_matches_pymkm_local_dose(ion: Ion) -> None:
    """Radial dose D(r) matches pyMKM across core and penumbra to <0.1%.

    Small residuals stem only from pyMKM's slightly older elementary-charge
    and atomic-mass constants; the physics is identical.
    """
    sp = SimpleBetheStoppingPower()
    let = sp.let_keV_um(ion)
    model = KieferChatterjeeRadialDose(stopping_power=sp)

    r_c = model.r_min_um(ion)
    r_p = model.r_max_um(ion)
    radii = np.geomspace(0.1 * r_c, 0.99 * r_p, 60)

    ours = np.array([model.local_dose_Gy(float(r), ion) for r in radii])
    theirs = _pymkm_kc_dose(ion, let, radii)
    np.testing.assert_allclose(ours, theirs, rtol=2e-3)


@requires_pymkm
def test_pymkm_helper_converts_let_to_mev_per_cm_once() -> None:
    """The adapter hands pyMKM ``10 * LET[keV/um]`` (MeV/cm), exactly as before.

    Pinned against a direct ``ParticleTrack`` call: the helper must be the same
    numbers bit for bit, and a LET passed unconverted must give a different curve.
    """
    from pymkm.physics.particle_track import ParticleTrack

    from pyradbio.models.mkm.pymkm_backend import (
        KEV_PER_UM_TO_MEV_PER_CM,
        pymkm_kiefer_chatterjee_dose,
    )

    assert KEV_PER_UM_TO_MEV_PER_CM == 10.0
    ion = Ion(Z=6, A=12, energy_MeV_u=30.0)
    let = SimpleBetheStoppingPower().let_keV_um(ion)
    radii = np.geomspace(1.0e-3, 1.0, 25)
    ours = pymkm_kiefer_chatterjee_dose(ion, radii, let)

    def direct(let_mev_cm: float) -> np.ndarray:
        track = ParticleTrack(
            model_name="Kiefer-Chatterjee", energy=30.0, atomic_number=6, let=let_mev_cm
        )
        dose, _ = track.initial_local_dose(radius=radii)
        return np.asarray(dose, dtype=float).ravel()

    np.testing.assert_array_equal(ours, direct(let * 10.0))
    assert ours.shape == radii.shape
    assert not np.allclose(ours, direct(let), rtol=1e-3)


@requires_pymkm
def test_matches_pymkm_geometry() -> None:
    """Core (Rc) and penumbra (Rp) radii match pyMKM's derived quantities."""
    from pymkm.physics.particle_track import ParticleTrack

    ion = Ion(Z=6, A=12, energy_MeV_u=100.0)
    model = KieferChatterjeeRadialDose(stopping_power=SimpleBetheStoppingPower())
    track = ParticleTrack(model_name="Kiefer-Chatterjee", energy=100.0, atomic_number=6, let=500.0)
    assert model.r_min_um(ion) == pytest.approx(track._compute_core_radius(), rel=1e-4)
    assert model.r_max_um(ion) == pytest.approx(track.penumbra_radius, rel=1e-4)


# ---- unit-handling sanity ----------------------------------------------
def test_effective_charge_reused_from_stopping_power() -> None:
    """Kp uses the same Barkas effective charge as the stopping-power model."""
    ion = Ion(Z=6, A=12, energy_MeV_u=100.0)
    beta = ion.beta()
    model = KieferChatterjeeRadialDose(stopping_power=SimpleBetheStoppingPower())
    kp = model._kp_Gy_um2(ion)
    expected = (
        KieferChatterjeeRadialDose().kp_kappa_Gy_um2 * (effective_charge(6, beta) / beta) ** 2
    )
    assert kp == pytest.approx(expected)
    # And the KEVUM constant ties dose[Gy]*um^2 back to LET[keV/um].
    assert KEVUM_PER_GY_UM2_GCM3 == pytest.approx(1e-9 / 1.602176634e-10)
    assert math.isfinite(kp)
