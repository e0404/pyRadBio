"""MCF MKM and mMKM: LQ parameters / RBE from a lineal-energy spectrum.

Physics acceptance for :mod:`pyradbio.models.mkm.spectrum_models`:

* specific-energy helper scales as ``1/r^2`` and matches the reference value;
* photon limit (a spectrum at ``y_D ~ y_D,ref``) recovers ``(alpha_ref, beta_ref)``;
* the mMKM ``y*_D`` agrees with :meth:`Spectrum.y_star`;
* MCF ``beta`` drops below ``beta_ref`` at high LET while mMKM ``beta`` stays put;
* ``RBE_10`` for carbon rises with LET to an overkill peak then drops
  (using the bundled Hartzell 2025 digitized AMF spectra).
"""

from __future__ import annotations

import importlib.util
import math

import numpy as np
import pytest
from scipy.integrate import simpson

from pyradbio.benchmark.hartzell2025 import load_amf_curves
from pyradbio.core.constants import KEVUM_PER_GY_UM2_GCM3
from pyradbio.core.radiation import Ion
from pyradbio.micro.lineal_energy import AnalyticSpectrumEngine, Spectrum, SphericalSite
from pyradbio.models.mkm.spectrum_models import (
    MCFMKMModel,
    ModifiedMKM,
    StochasticMKM,
    specific_energy_Gy,
)
from pyradbio.track.stopping_power import SimpleBetheStoppingPower

requires_pymkm = pytest.mark.skipif(
    importlib.util.find_spec("pymkm") is None,
    reason="pyMKM not installed ([mkm] extra)",
)

ALPHA_REF = 0.217
BETA_REF = 0.0615
YD_REF = 2.5


def _lognormal_spectrum(y_D_target_keV_um: float, sigma: float = 0.35, n: int = 6000) -> Spectrum:
    """A narrow log-normal ``f1(y)`` whose dose-mean lineal energy is the target.

    For a log-normal with median ``m`` and shape ``sigma``,
    ``y_D = m * exp(1.5 sigma^2)``; invert for ``m`` so the spectrum has a known
    ``y_D`` (used to place the photon-limit and high-LET test spectra).
    """
    median = y_D_target_keV_um / math.exp(1.5 * sigma * sigma)
    mu = math.log(median)
    y = np.logspace(math.log10(median) - 6.0 * sigma, math.log10(median) + 6.0 * sigma, n)
    f = np.exp(-((np.log(y) - mu) ** 2) / (2.0 * sigma * sigma)) / (
        y * math.sqrt(2.0 * math.pi) * sigma
    )
    return Spectrum(y_keV_um=y, f=f / float(np.trapezoid(f, y)))


# --------------------------------------------------------------------------
# specific energy helper
# --------------------------------------------------------------------------
def test_specific_energy_scales_as_inverse_radius_squared() -> None:
    z1 = float(specific_energy_Gy(10.0, 0.28))
    z2 = float(specific_energy_Gy(10.0, 0.56))
    assert z2 == pytest.approx(z1 / 4.0, rel=1e-12)


def test_specific_energy_reference_value() -> None:
    # z_d(y_D,ref) with r_d = 0.28 um: y / (KEVUM * rho * pi * r^2).
    z = float(specific_energy_Gy(YD_REF, 0.28))
    expected = YD_REF / (KEVUM_PER_GY_UM2_GCM3 * math.pi * 0.28**2)
    assert z == pytest.approx(expected, rel=1e-12)


def test_specific_energy_rejects_nonpositive_radius() -> None:
    with pytest.raises(ValueError):
        specific_energy_Gy(1.0, 0.0)


# --------------------------------------------------------------------------
# MCF MKM
# --------------------------------------------------------------------------
def test_mcf_alpha0_matches_eq14() -> None:
    model = MCFMKMModel()
    z_d_ref = float(specific_energy_Gy(YD_REF, model.domain_radius_um))
    assert model.beta0 == pytest.approx(BETA_REF)
    assert model.alpha0 == pytest.approx(ALPHA_REF - BETA_REF * z_d_ref, rel=1e-12)


def test_mcf_saturation_correction_goes_to_one_at_low_y() -> None:
    model = MCFMKMModel()
    c = model.saturation_correction(np.array([1e-3, 1.0, 1e4]))
    assert c[0] == pytest.approx(1.0, abs=1e-6)  # no saturation at low y
    assert c[2] < 0.5  # heavy overkill at very high y
    assert np.all(np.diff(c) < 0.0)  # monotonically decreasing


def test_mcf_photon_limit_recovers_reference() -> None:
    model = MCFMKMModel()
    lq = model.alpha_beta_from_spectrum(_lognormal_spectrum(YD_REF))
    assert lq.alpha == pytest.approx(ALPHA_REF, rel=0.01)
    assert lq.beta == pytest.approx(BETA_REF, rel=0.01)


def test_mcf_beta_drops_below_reference_at_high_let() -> None:
    model = MCFMKMModel()
    lq = model.alpha_beta_from_spectrum(_lognormal_spectrum(150.0))
    assert lq.beta < BETA_REF  # saturation suppresses beta (MCF-specific behaviour)


def test_mcf_alpha_rises_then_overkill_drops_with_let() -> None:
    model = MCFMKMModel()
    alphas = [
        model.alpha_beta_from_spectrum(_lognormal_spectrum(yd)).alpha for yd in (2.5, 60.0, 600.0)
    ]
    assert alphas[1] > alphas[0]  # rising limb
    assert alphas[2] < alphas[1]  # overkill drop at very high LET


# --------------------------------------------------------------------------
# mMKM
# --------------------------------------------------------------------------
def test_mmkm_y_star_matches_spectrum_y_star() -> None:
    model = ModifiedMKM()
    spec = _lognormal_spectrum(50.0)
    direct = model.y_star_D(spec.y_keV_um, spec.d)
    assert direct == pytest.approx(spec.y_star(model.y0_keV_um), rel=1e-6)


def test_mmkm_photon_limit_recovers_reference() -> None:
    model = ModifiedMKM()
    lq = model.alpha_beta_from_spectrum(_lognormal_spectrum(YD_REF))
    assert lq.alpha == pytest.approx(ALPHA_REF, rel=0.02)
    assert lq.beta == pytest.approx(BETA_REF)


def test_mmkm_beta_is_radiation_independent() -> None:
    model = ModifiedMKM()
    for yd in (2.5, 50.0, 500.0):
        assert model.alpha_beta_from_spectrum(_lognormal_spectrum(yd)).beta == BETA_REF


def test_mmkm_z_star_1d_equals_kase_dose_average() -> None:
    """Our ``z*_1D = y*_D / (KEVUM rho pi r_d^2)`` is the Kase/Inaniwa quantity.

    Cross-check the ``y0`` (eq 12) / unit algebra by computing ``z*_1D`` the
    *other* way: dose-average the saturation-corrected specific energy
    ``z_sat = (z0^2/z)(1 - exp(-(z/z0)^2))`` directly over ``d(y)`` (Inaniwa eqs
    3-4). The two must agree to machine precision.
    """
    model = ModifiedMKM()
    spec = _lognormal_spectrum(80.0)
    y, d = spec.y_keV_um, spec.d
    scale = KEVUM_PER_GY_UM2_GCM3 * math.pi * model.domain_radius_um**2  # y -> z_d [Gy]
    z = y / scale
    z0 = model.y0_keV_um / scale
    d_norm = d / float(simpson(d, x=y))
    z_sat = (z0 * z0 / z) * (1.0 - np.exp(-((z / z0) ** 2)))
    z_star_direct = float(simpson(z_sat * d_norm, x=y))  # dose-average of z_sat
    assert model.z_star_1D_Gy(y, d) == pytest.approx(z_star_direct, rel=1e-9)


@requires_pymkm
def test_z_star_1d_matches_pymkm_at_high_let() -> None:
    """Independent magnitude cross-check of ``z*_1D`` against pyMKM.

    pyMKM computes ``z_bar_star_domain`` from its own Kiefer-Chatterjee track and
    stopping power; we compute the same quantity from our restricted analytic
    single-event spectrum. The two track-structure / restricted-LET pipelines
    coincide only in the **crosser-dominated high-LET regime** (carbon at a few
    MeV/u) -- at low LET they diverge (different delta-ray escape treatments), so
    this asserts magnitude + units where the input spectra agree, not a tight
    golden across all energies. Both also show the saturation turnover.
    """
    from pyradbio.models.mkm.pymkm_backend import PyMKMBackend

    r_d, r_n, beta0 = 0.32, 3.9, 0.0615  # Inaniwa HSG geometry (pyMKM default)
    sp = SimpleBetheStoppingPower()
    engine = AnalyticSpectrumEngine(stopping_power=sp, site=SphericalSite(2 * r_d), restricted=True)
    mmkm = ModifiedMKM(domain_radius_um=r_d, nucleus_radius_um=r_n, beta_ref=beta0)
    backend = PyMKMBackend(domain_radius_um=r_d, nucleus_radius_um=r_n, beta0=beta0)

    for energy in (5.0, 12.0):
        ion = Ion(6, 12, energy)
        spec = engine.single_event_spectrum(ion)
        z_our = mmkm.z_star_1D_Gy(spec.y_keV_um, spec.d)
        z_pymkm = backend.z_star_1D_Gy(ion)
        assert z_our == pytest.approx(z_pymkm, rel=0.15)


# --------------------------------------------------------------------------
# RBE trend vs the digitized Hartzell 2025 carbon AMF spectra
# --------------------------------------------------------------------------
def _carbon_amf_dose_spectrum(energy_MeV_u: float) -> tuple[np.ndarray, np.ndarray]:
    (curve,) = [
        c
        for c in load_amf_curves()
        if c.figure == "fig4"
        and c.ion == "carbon"
        and c.method == "AMF"
        and c.energy_MeV_u == energy_MeV_u
    ]
    return curve.dose_spectrum()  # d(y) = (y d(y)) / y


@pytest.mark.parametrize("model_cls", [MCFMKMModel, ModifiedMKM])
def test_rbe10_carbon_overkill_peak(model_cls: type) -> None:
    model = model_cls()
    rbe = {
        e: model.rbe_at_survival_from_dose_spectrum(*_carbon_amf_dose_spectrum(e))
        for e in (1.0, 10.0, 100.0)
    }
    assert all(v > 1.0 for v in rbe.values())  # carbon is more effective than photons
    assert rbe[10.0] > rbe[100.0]  # rising limb (LET increases 100 -> 10 MeV/u)
    assert rbe[1.0] < rbe[10.0]  # overkill drop at the highest LET (1 MeV/u)


@pytest.mark.parametrize("model_cls", [MCFMKMModel, ModifiedMKM])
def test_alpha_beta_uses_default_amorphous_engine(model_cls: type) -> None:
    """With no engine injected, ``alpha_beta(ion)`` uses the default amorphous engine.

    Reproduces the carbon RBE_alpha-vs-energy trend (rise then overkill drop) with
    no pyMKM / external microdosimetry table -- the MKM family is self-contained.
    """
    model = model_cls(
        alpha_ref=0.19,
        beta_ref=0.019,
        nucleus_radius_um=4.0,
        domain_radius_um=0.26,
        yD_ref_keV_um=4.3,
    )
    alpha = {e: model.alpha_beta(Ion(6, 12, e)).alpha for e in (10.0, 100.0)}
    assert all(a > 0.0 for a in alpha.values())
    assert alpha[10.0] > alpha[100.0]  # higher LET (10 MeV/u) -> larger alpha


def test_delta_tail_reduces_v79_carbon_alpha_overprediction() -> None:
    """The analytic-engine delta-ray tail lowers the (hot) MCF-MKM V79-carbon alpha.

    The crosser-only Xapsos engine over-predicts RBE_alpha for V79 carbon; folding
    in the delta-ray toucher tail (which lowers y_D toward the track-structure /
    Sato-AMF values) lowers the predicted alpha across the spectrum, most in the
    mid-energy regime where the crosser runs hottest -- moving the prediction
    toward the reference (Parisi 2022; V79 200-kVp reference, r_d = 0.26 um).
    """
    r_d = 0.26
    sp = SimpleBetheStoppingPower()
    site = SphericalSite(2 * r_d)
    crosser = AnalyticSpectrumEngine(sp, site, restricted=True)
    tailed = AnalyticSpectrumEngine(sp, site, restricted=True, include_delta_tail=True)

    def model(engine: AnalyticSpectrumEngine) -> MCFMKMModel:
        return MCFMKMModel(
            alpha_ref=0.19,
            beta_ref=0.019,
            nucleus_radius_um=4.0,
            domain_radius_um=r_d,
            yD_ref_keV_um=4.3,
            engine=engine,
        )

    mc, mt = model(crosser), model(tailed)
    for energy in (10.0, 30.0, 100.0):
        ion = Ion(6, 12, energy)
        assert mt.alpha_beta(ion).alpha < mc.alpha_beta(ion).alpha
    # Largest reduction in the mid-energy (hottest crosser) regime: >= ~15% at 10 MeV/u.
    a_cross = mc.alpha_beta(Ion(6, 12, 10.0)).alpha
    a_tail = mt.alpha_beta(Ion(6, 12, 10.0)).alpha
    assert a_tail < 0.85 * a_cross


# --------------------------------------------------------------------------
# site geometry: the ion -> specific-energy route
# --------------------------------------------------------------------------
def test_cylinder_is_the_default_site_geometry() -> None:
    """The MKM convention (Inaniwa 2010/2018 sec 3.2): cylinders, not spheres."""
    assert ModifiedMKM().site_geometry == "cylinder"
    assert StochasticMKM().site_geometry == "cylinder"


def test_site_geometry_is_validated() -> None:
    with pytest.raises(ValueError, match="site_geometry"):
        ModifiedMKM(site_geometry="torus").specific_energies(Ion(6, 12, 100.0))


def test_cylinder_route_matches_the_native_mkm_backend() -> None:
    """``specific_energies`` and ``NativeMKMBackend`` must share one z* definition.

    Both integrate the same Kiefer-Chatterjee radial dose over the same domain disk
    with the same saturation parameter, so they agree exactly. Pinning it stops the
    two z* routes in the package from drifting apart.
    """
    from pyradbio.models.mkm.native_backend import NativeMKMBackend

    r_d, r_n, beta0 = 0.32, 3.9, 0.0615
    model = ModifiedMKM(domain_radius_um=r_d, nucleus_radius_um=r_n, beta_ref=beta0)
    backend = NativeMKMBackend(domain_radius_um=r_d, nucleus_radius_um=r_n, beta0=beta0)
    for energy in (5.0, 12.0, 100.0):
        ion = Ion(6, 12, energy)
        assert model.specific_energies(ion).z_star_1D_Gy == pytest.approx(
            backend.z_star_1D_Gy(ion), rel=1e-12
        )


def test_spherical_site_geometry_overshoots_z_star_at_high_energy() -> None:
    """Why the default is a cylinder: the sphere inflates ``z*_1D`` by ~25-30%.

    A spherical site volume-averages the ``1/r^2`` radial dose over chords that
    sample the track core differently from the disk the MKM actually uses. The
    error grows with energy (a wider penumbra makes the geometry matter more) and
    is the reason ``mkm-domain-disk-geometry`` was flagged in the first place.
    """
    r_d, r_n, beta0 = 0.32, 3.9, 0.0615
    kw = {"domain_radius_um": r_d, "nucleus_radius_um": r_n, "beta_ref": beta0}
    cylinder = ModifiedMKM(**kw)  # type: ignore[arg-type]
    sphere = ModifiedMKM(site_geometry="sphere", **kw)  # type: ignore[arg-type]

    def ratio(energy: float) -> float:
        ion = Ion(6, 12, energy)
        return (
            sphere.specific_energies(ion).z_star_1D_Gy
            / cylinder.specific_energies(ion).z_star_1D_Gy
        )

    assert ratio(12.0) < 1.0  # slightly low where the core dominates
    assert ratio(100.0) > 1.25  # ... and badly high once the penumbra does
    assert ratio(400.0) > ratio(100.0)  # error grows with energy


@requires_pymkm
def test_cylinder_route_tracks_pymkm_across_energies() -> None:
    """The aligned route follows pyMKM's ``z_bar_star_domain`` within a few percent.

    The residual is our Kiefer-Chatterjee/stopping-power pipeline vs pyMKM's own,
    not the geometry -- which is the point of aligning it. The spherical route
    misses by up to ~32% at 400 MeV/u for comparison.
    """
    from pyradbio.models.mkm.pymkm_backend import PyMKMBackend

    r_d, r_n, beta0 = 0.32, 3.9, 0.0615
    model = ModifiedMKM(domain_radius_um=r_d, nucleus_radius_um=r_n, beta_ref=beta0)
    backend = PyMKMBackend(domain_radius_um=r_d, nucleus_radius_um=r_n, beta0=beta0)
    for energy in (5.0, 12.0, 100.0, 400.0):
        ion = Ion(6, 12, energy)
        assert model.specific_energies(ion).z_star_1D_Gy == pytest.approx(
            backend.z_star_1D_Gy(ion), rel=0.05
        )


def test_injected_engine_overrides_the_geometry_route() -> None:
    """An injected ``SpectrumEngine`` already fixes its own site geometry, so it wins.

    Silently re-scoring a caller's engine in a different geometry would be hidden
    physics; ``specific_energies`` defers to it instead.
    """
    r_d = 0.32
    engine = AnalyticSpectrumEngine(
        SimpleBetheStoppingPower(), SphericalSite(2 * r_d), restricted=True
    )
    ion = Ion(6, 12, 12.0)
    injected = ModifiedMKM(domain_radius_um=r_d, nucleus_radius_um=3.9, engine=engine)
    analytic = ModifiedMKM(domain_radius_um=r_d, nucleus_radius_um=3.9)

    spectrum = engine.single_event_spectrum(ion)
    assert injected.specific_energies(ion).z_star_1D_Gy == pytest.approx(
        injected.z_star_1D_Gy(spectrum.y_keV_um, spectrum.d), rel=1e-12
    )
    # ... and that is genuinely a different number from the impact-parameter route
    assert injected.specific_energies(ion).z_star_1D_Gy != pytest.approx(
        analytic.specific_energies(ion).z_star_1D_Gy, rel=1e-3
    )


def test_z0_override_feeds_the_geometry_route() -> None:
    """An explicit ``z0_Gy`` must reach the impact-parameter saturation, not just d(y)."""
    ion = Ion(6, 12, 12.0)
    default = ModifiedMKM(nucleus_radius_um=3.9)
    lowered = ModifiedMKM(nucleus_radius_um=3.9, z0_Gy=0.5 * default.z0_effective_Gy)
    assert lowered.specific_energies(ion).z_star_1D_Gy < default.specific_energies(ion).z_star_1D_Gy
