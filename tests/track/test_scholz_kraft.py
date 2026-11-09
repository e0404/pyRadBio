"""Radial-dose track structure."""

from __future__ import annotations

import numpy as np
import pytest

from pyradbio.core.radiation import Ion
from pyradbio.track.scholz_kraft import LEM3_RC_NM, ScholzKraftRadialDose
from pyradbio.track.stopping_power import SimpleBetheStoppingPower


class _ConstantStoppingPower:
    """Fake StoppingPower returning a fixed LET, to test radial-dose math alone."""

    def __init__(self, let_keV_um: float) -> None:
        self._let = let_keV_um

    def let_keV_um(self, ion: Ion) -> float:
        return self._let


@pytest.mark.golden
def test_golden_r_max_parameterisation() -> None:
    """r_max = 0.062 * E^1.7 um (Elsasser & Scholz 2007, after eq. 6)."""
    model = ScholzKraftRadialDose()
    ion = Ion(Z=6, A=12, energy_MeV_u=100.0)
    expected_um = 0.062 * (100.0**1.7)
    assert model.r_max_um(ion) == pytest.approx(expected_um, rel=1e-6)


@pytest.mark.golden
def test_golden_lem1_r_max_uses_original_parameterisation() -> None:
    """LEM I uses r_max = 0.05 * E^1.7 um.

    Reference: Scholz & Kraft 1996 / Scholz et al. 1997
    (doi:10.1007/s004110050055); LEM II later uses 0.062.
    """
    model = ScholzKraftRadialDose.lem1()
    ion = Ion(Z=6, A=12, energy_MeV_u=100.0)
    expected_um = 0.05 * (100.0**1.7)
    assert model.r_max_um(ion) == pytest.approx(expected_um, rel=1e-6)


@pytest.mark.parametrize("energy", [10.0, 100.0, 270.0])
def test_radial_integral_reproduces_let_lem1(energy: float) -> None:
    """Integral of D(r)*2*pi*r dr over [0, r_max] reproduces LET (normalisation).

    LEM I (no radical diffusion) -> the normalisation is analytic, so the
    numerical energy integral must recover the input LET tightly.
    """
    sp = _ConstantStoppingPower(50.0)
    model = ScholzKraftRadialDose.lem1(sp)
    ion = Ion(Z=6, A=12, energy_MeV_u=energy)
    assert model.radial_integral_LET_keV_um(ion) == pytest.approx(50.0, rel=1e-3)


@pytest.mark.parametrize("model_factory", [ScholzKraftRadialDose.lem2, ScholzKraftRadialDose.lem3])
@pytest.mark.parametrize("energy", [10.0, 100.0, 400.0])
def test_radial_integral_reproduces_let_with_diffusion(model_factory, energy: float) -> None:
    """Radial-diffusion convolution conserves the radial energy integral.

    A Gaussian smear moves energy around; it does not create or destroy any, so
    this must hold at *every* energy and it is the sharpest available probe of
    the convolution quadrature. It has to sweep the energy: ``r_max ~ E^1.7``
    sets how far apart the log and peak node spacings are inside the
    convolution, so the quadrature degrades with energy and a single-energy
    check is blind to it (at 100 MeV/u with ``rel=5e-3`` the test would pass while the
    400 MeV/u profile lost 2.9% of the LET).
    """
    sp = SimpleBetheStoppingPower()
    model = model_factory(sp)
    ion = Ion(Z=6, A=12, energy_MeV_u=energy)
    let = sp.let_keV_um(ion)
    got = model.radial_integral_LET_keV_um(ion, nodes=1024)
    assert got == pytest.approx(let, rel=1e-3)


@pytest.mark.parametrize("model_factory", [ScholzKraftRadialDose.lem2, ScholzKraftRadialDose.lem3])
@pytest.mark.parametrize("energy", [0.1, 0.2, 0.5, 1.0, 3.0])
def test_radial_integral_conserves_let_when_the_track_is_only_nm_wide(
    model_factory, energy: float
) -> None:
    """Energy conservation still holds where ``r_max`` is comparable to ``sigma``.

    The sweep above starts at 10 MeV/u, where ``r_max`` is 6000x ``sigma`` and the
    smeared tail past ``r_max`` is 2.6e-5 of the track. Below ~0.5 MeV/u the two
    are the *same size* -- carbon at 0.1 MeV/u has ``r_max = 1.24 nm`` against a
    4 nm ``sigma`` -- so most of the convolved dose lands outside the delta-ray
    range. Integrating to ``r_max`` there recovered **4.6%** of the LET; the
    domain has to run to ``r_support_um``. This is the regime TRiP98's tables
    reach at a distal field edge, so it is not an academic corner.
    """
    sp = SimpleBetheStoppingPower()
    model = model_factory(sp)
    ion = Ion(Z=6, A=12, energy_MeV_u=energy)
    let = sp.let_keV_um(ion)
    got = model.radial_integral_LET_keV_um(ion, nodes=1024)
    assert got == pytest.approx(let, rel=1e-3)


@pytest.mark.parametrize("model_factory", [ScholzKraftRadialDose.lem2, ScholzKraftRadialDose.lem3])
def test_support_exceeds_r_max_only_with_radical_diffusion(model_factory) -> None:
    """``r_support_um`` is ``r_max`` plus the diffusion tail, and ``r_max`` without it."""
    sp = SimpleBetheStoppingPower()
    ion = Ion(Z=6, A=12, energy_MeV_u=1.0)
    smeared = model_factory(sp)
    assert smeared.r_support_um(ion) > smeared.r_max_um(ion)
    # the whole excess is the Gaussian tail, in absolute (not relative) units
    assert smeared.r_support_um(ion) - smeared.r_max_um(ion) == pytest.approx(
        8.0 * smeared.sigma_nm * 1e-3
    )
    sharp = ScholzKraftRadialDose.lem1(sp)
    assert sharp.r_support_um(ion) == sharp.r_max_um(ion)


def test_lem3_velocity_dependent_core() -> None:
    """LEM III core radius scales with beta: r_min = beta * r_c (Elsasser 2008 eq.6)."""
    model = ScholzKraftRadialDose.lem3()
    low = Ion(Z=6, A=12, energy_MeV_u=10.0)
    high = Ion(Z=6, A=12, energy_MeV_u=270.0)
    assert model.r_min_um(low) == pytest.approx(low.beta() * LEM3_RC_NM * 1e-3)
    assert model.r_min_um(high) > model.r_min_um(low)  # grows with velocity


def test_lem1_fixed_core_independent_of_energy() -> None:
    model = ScholzKraftRadialDose.lem1()
    assert model.r_min_um(Ion(6, 12, 10.0)) == pytest.approx(0.010)
    assert model.r_min_um(Ion(6, 12, 270.0)) == pytest.approx(0.010)


def test_lem2_profile_qualitative() -> None:
    """LEM II radial diffusion (Elsasser 2007 Fig. 4): smeared/reduced core,
    1/r^2 tail preserved, reduced dose at the outer edge."""
    sp = SimpleBetheStoppingPower()
    ion = Ion(Z=6, A=12, energy_MeV_u=100.0)
    base = ScholzKraftRadialDose(r_min_nm=0.3, radical_diffusion=False, stopping_power=sp)
    conv = ScholzKraftRadialDose.lem2(sp)
    r_max = base.r_max_um(ion)

    sigma = conv.sigma_nm * 1.0e-3

    # Core peak (r < r_min) is reduced by the smearing.
    assert conv.local_dose_Gy(0.0005, ion) < base.local_dose_Gy(0.0005, ion)

    # The outer edge is reduced, but ONLY within a few sigma of the cutoff, and
    # the reduction is not free-form: at exactly r_max half the Gaussian falls
    # beyond the delta-electron range, so the dose is halved. (A check at
    # 0.99 r_max would be 389 sigma inside the edge, where a 4 nm smear cannot reach.)
    assert conv.local_dose_Gy(r_max, ion) / base.local_dose_Gy(r_max, ion) == pytest.approx(
        0.5, rel=1.0e-3
    )
    assert conv.local_dose_Gy(r_max - 2.0 * sigma, ion) < base.local_dose_Gy(
        r_max - 2.0 * sigma, ion
    )

    # ... and beyond a few sigma the convolution is a no-op: the kernel is far
    # narrower than the scale on which 1/r^2 varies, leaving only the O((sigma/r)^2)
    # convexity term. This is what pins the quadrature -- Simpson on the merged
    # log/linear node grid drifts percent-level here, and worse (sign-flipping)
    # further out.
    for factor in (0.5, 0.9, 0.99):
        r = factor * r_max
        assert conv.local_dose_Gy(r, ion) == pytest.approx(base.local_dose_Gy(r, ion), rel=1.0e-4)

    # The 1/r^2 tail is preserved: D(r)*r^2 ~ constant and monotone decreasing.
    tail = np.linspace(1.0, 0.9 * r_max, 25)
    vals = [conv.local_dose_Gy(float(r), ion) for r in tail]
    assert all(vals[i] > vals[i + 1] for i in range(len(vals) - 1))
    scaled = [conv.local_dose_Gy(r, ion) * r * r for r in (1.0, 10.0, 50.0)]
    assert max(scaled) / min(scaled) < 1.01  # constant to ~1%


def test_dose_zero_beyond_r_max() -> None:
    sp = _ConstantStoppingPower(50.0)
    model = ScholzKraftRadialDose.lem1(sp)
    ion = Ion(6, 12, 100.0)
    assert model.local_dose_Gy(model.r_max_um(ion) * 1.001, ion) == 0.0


@pytest.mark.parametrize("density", [1.0, 1.1])
def test_local_dose_defaults_to_the_library_stopping_power(density: float) -> None:
    """No stopping power means ``default_stopping_power()`` at the model's density."""
    from dataclasses import replace

    from pyradbio.track.stopping_power import default_stopping_power

    ion = Ion(6, 12, 100.0)
    implicit = replace(ScholzKraftRadialDose.lem1(), density_g_cm3=density)
    explicit = replace(implicit, stopping_power=default_stopping_power(density))
    for r in (1.0e-3, 0.1, 1.0):
        assert implicit.local_dose_Gy(r, ion) == explicit.local_dose_Gy(r, ion)
    assert implicit.radial_integral_LET_keV_um(ion) == pytest.approx(
        default_stopping_power(density).let_keV_um(ion), rel=1e-3
    )


def test_negative_radius_rejected() -> None:
    model = ScholzKraftRadialDose.lem1(_ConstantStoppingPower(50.0))
    with pytest.raises(ValueError):
        model.local_dose_Gy(-1.0, Ion(6, 12, 100.0))


def test_energy_too_low_raises() -> None:
    """When r_max <= r_min the track model is undefined and must error."""
    sp = _ConstantStoppingPower(50.0)
    model = ScholzKraftRadialDose(r_min_nm=10.0, stopping_power=sp)
    # r_max = 0.062*E^1.7; at very low E this drops below r_min = 0.01 um.
    with pytest.raises(ValueError):
        model.local_dose_Gy(0.001, Ion(6, 12, 0.1))


@pytest.mark.parametrize(
    "radial", [ScholzKraftRadialDose.lem1(), ScholzKraftRadialDose.lem2()], ids=["lem1", "lem2"]
)
def test_local_doses_is_the_scalar_loop_bit_for_bit(radial: ScholzKraftRadialDose) -> None:
    """``local_doses_Gy`` keeps the input shape and equals the scalar call per radius."""
    radial = ScholzKraftRadialDose(
        r_min_nm=radial.r_min_nm,
        radical_diffusion=radial.radical_diffusion,
        stopping_power=SimpleBetheStoppingPower(),
        r_max_gamma_um=radial.r_max_gamma_um,
    )
    ion = Ion(Z=8, A=16, energy_MeV_u=2.57)
    r_cm = np.logspace(-8.1, -3.0, 24).reshape(4, 6)
    got = radial.local_doses_Gy(r_cm * 1e4, ion)
    assert got.shape == (4, 6)
    want = np.array([radial.local_dose_Gy(float(r * 1e4), ion) for r in r_cm.ravel()])
    assert np.array_equal(got.ravel(), want)
    assert radial.local_doses_Gy([], ion).shape == (0,)
    with pytest.raises(ValueError):
        radial.local_doses_Gy([0.1, -1.0], ion)
