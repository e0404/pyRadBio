"""Analytic impact-parameter integration over a spherical site."""

from __future__ import annotations

import math

import numpy as np
import pytest

from pyradbio.core.radiation import Ion
from pyradbio.micro import SphericalSite, specific_energy_impact_profile
from pyradbio.track.kiefer_chatterjee import KieferChatterjeeRadialDose
from pyradbio.track.scholz_kraft import ScholzKraftRadialDose
from pyradbio.track.stopping_power import SimpleBetheStoppingPower


def test_volume_integral_constant_field_recovers_sphere_volume() -> None:
    """integral_V 1 dV at b=0 equals the sphere volume (geometry primitive check)."""
    site = SphericalSite(diameter_um=0.64)
    radius = site.radius_um()
    s = np.logspace(-4.0, math.log10(radius), 400)
    ones = np.ones_like(s)
    integral = site.volume_integral_over_site(ones, s, np.array([radius * 1e-6]))[0]
    assert integral == pytest.approx((4.0 / 3.0) * math.pi * radius**3, rel=2e-3)


def _profile(radius_um: float = 0.32, energy: float = 11.0, geometry: str = "cylinder"):  # type: ignore[no-untyped-def]
    rd = KieferChatterjeeRadialDose(SimpleBetheStoppingPower())
    return specific_energy_impact_profile(
        rd, Ion(6, 12, energy), region_radius_um=radius_um, geometry=geometry
    )


def test_frequency_weight_is_normalized() -> None:
    """<1>_F = integral 2b db / b_max^2 = 1."""
    assert _profile().average(lambda z: np.ones_like(z)) == pytest.approx(1.0, rel=1e-6)


def test_dose_mean_at_least_frequency_mean() -> None:
    """Dose-weighting emphasises high-z events, so z_D >= z_F."""
    p = _profile()
    assert p.dose_mean_Gy() >= p.frequency_mean_Gy() > 0.0


def test_specific_energy_positive_and_decreasing_overall() -> None:
    p = _profile()
    assert np.all(p.specific_energy_Gy >= 0.0)
    assert p.specific_energy_Gy[0] > 0.0  # the crosser core deposits a positive dose
    assert p.specific_energy_Gy[0] > p.specific_energy_Gy[-1]  # core deposits more than penumbra


def test_saturation_approaches_dose_mean_without_saturation() -> None:
    """As z0 -> infinity the saturated dose-mean tends to the plain dose-mean."""
    p = _profile()
    assert p.saturation_corrected_dose_mean_Gy(1.0e9) == pytest.approx(p.dose_mean_Gy(), rel=1e-3)


def test_saturation_reduces_dose_mean() -> None:
    p = _profile()
    assert 0.0 < p.saturation_corrected_dose_mean_Gy(3.0) < p.dose_mean_Gy()


def test_works_across_radial_dose_models() -> None:
    """The integrator is track-model-agnostic (Kiefer-Chatterjee and Scholz-Kraft)."""
    ion = Ion(6, 12, 11.0)
    for rd in (
        KieferChatterjeeRadialDose(SimpleBetheStoppingPower()),
        ScholzKraftRadialDose.lem1(SimpleBetheStoppingPower()),
    ):
        p = specific_energy_impact_profile(rd, ion, region_radius_um=0.32)
        assert p.dose_mean_Gy() > 0.0
        assert p.let_keV_um > 0.0


# --- cylindrical / disk geometry (LEM nucleus) ----------------------------
def test_disk_arc_weight_matches_lem_angular_fraction() -> None:
    """The disk arc-angle weight matches LEM's own ``_angular_fraction_inside``."""
    from pyradbio.micro import disk_arc_weight
    from pyradbio.models.lem.base import _angular_fraction_inside

    s = np.linspace(0.05, 8.0, 50)
    radius = 5.0
    for b in (0.0, 1.0, 4.9, 5.0, 7.0):
        ours = disk_arc_weight(s, np.array([b]), radius)[0]
        lem = _angular_fraction_inside(s, b, radius)
        assert np.allclose(ours, lem, atol=1e-12)


def test_disk_area_integral_constant_field_recovers_disk_area() -> None:
    """integral_disk 1 dA at b=0 equals pi R^2 (geometry primitive check)."""
    from pyradbio.micro import disk_area_integral_over_site

    radius = 5.0
    s = np.linspace(1e-4, radius, 600)
    ones = np.ones_like(s)
    area = disk_area_integral_over_site(ones, s, np.array([radius * 1e-6]), radius_um=radius)[0]
    assert area == pytest.approx(math.pi * radius**2, rel=2e-3)


def test_nuclear_functional_profile_positive_and_reduces() -> None:
    """A disk functional profile (LEM-style log-kill) yields sane nu(b) and sigma.

    (The full cross-check against LEM's own machinery lives in
    tests/models/test_lem_impact_parameter_crosscheck.py.)
    """
    from pyradbio.core.photon import PhotonParameters
    from pyradbio.micro import nuclear_functional_impact_profile

    photon = PhotonParameters(alpha_x=0.18, beta_x=0.028, D_t=30.0)
    rd = KieferChatterjeeRadialDose(SimpleBetheStoppingPower())
    profile = nuclear_functional_impact_profile(
        rd, Ion(6, 12, 11.0), radius_um=5.0, transform=np.vectorize(photon.log_kill)
    )
    assert profile.central_value() > 0.0
    # sigma (with overkill) <= lethal-events area (unsaturated), both positive.
    assert 0.0 < profile.inactivation_cross_section_um2() <= profile.lethal_events_um2()
