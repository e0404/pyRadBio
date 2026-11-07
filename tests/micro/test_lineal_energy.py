"""Analytic single-event lineal-energy spectra (microdosimetry)."""

from __future__ import annotations

import math

import numpy as np
import pytest

from pyradbio.core.radiation import Ion
from pyradbio.micro.lineal_energy import (
    AmorphousTrackEngine,
    AnalyticSpectrumEngine,
    CylindricalSite,
    DeltaRayMonteCarlo,
    LinealEnergyCalculator,
    SphericalSite,
    max_delta_ray_energy_keV,
)
from pyradbio.track.kiefer_chatterjee import KieferChatterjeeRadialDose
from pyradbio.track.scholz_kraft import (
    LEM1_R_MAX_GAMMA_UM,
    R_MAX_DELTA,
    ScholzKraftRadialDose,
)
from pyradbio.track.stopping_power import SimpleBetheStoppingPower, effective_charge


class _ConstantStoppingPower:
    """Fake StoppingPower returning a fixed LET, to test the spectrum math alone."""

    def __init__(self, let_keV_um: float) -> None:
        self._let = let_keV_um

    def let_keV_um(self, ion: Ion) -> float:
        return self._let


# --- geometry ---------------------------------------------------------------


def test_sphere_chord_moments() -> None:
    """Sphere chord pdf f(l)=2l/d^2: normalised, mean 2d/3, <l^2>=d^2/2 (Kellerer)."""
    site = SphericalSite(diameter_um=1.0)
    ell = np.linspace(0.0, 1.0, 20001)
    pdf = site.chord_pdf(ell)
    assert np.trapezoid(pdf, ell) == pytest.approx(1.0, rel=1e-4)
    assert np.trapezoid(pdf * ell, ell) == pytest.approx(2.0 / 3.0, rel=1e-4)
    assert site.mean_chord_um() == pytest.approx(2.0 / 3.0)
    assert np.trapezoid(pdf * ell**2, ell) == pytest.approx(0.5, rel=1e-4)


def test_sphere_rejects_nonpositive_diameter() -> None:
    with pytest.raises(ValueError):
        SphericalSite(diameter_um=0.0)


@pytest.mark.parametrize(
    ("diameter", "height"),
    [(1.0, 1.0), (1.0, 5.0), (5.0, 1.0), (0.5, 2.0)],
)
def test_cylinder_mean_chord_matches_cauchy(diameter: float, height: float) -> None:
    """Sampled mu-randomness mean chord reproduces Cauchy's l_bar = 4V/S.

    For a right cylinder 4V/S = 2 d h / (d + 2 h); the sampled secants must
    reproduce it (a model-independent geometric identity).
    """
    site = CylindricalSite(diameter, height, n_samples=1_000_000, seed=0)
    analytic = 2.0 * diameter * height / (diameter + 2.0 * height)
    assert site.mean_chord_um() == pytest.approx(analytic)
    # Sampled chords: their mean must match too (validates the sampler itself).
    ell = np.linspace(0.0, site.max_chord_um(), 6000)
    pdf = site.chord_pdf(ell)
    sampled_mean = np.trapezoid(pdf * ell, ell)
    assert sampled_mean == pytest.approx(analytic, rel=1e-2)


def test_cylinder_max_chord_is_diagonal() -> None:
    site = CylindricalSite(2.0, 3.0)
    assert site.max_chord_um() == pytest.approx(math.hypot(2.0, 3.0))


def test_cylinder_chord_pdf_normalised_and_bounded() -> None:
    site = CylindricalSite(1.0, 2.0, seed=1)
    ell = np.linspace(0.0, site.max_chord_um(), 8000)
    assert np.trapezoid(site.chord_pdf(ell), ell) == pytest.approx(1.0, rel=2e-2)
    # No density beyond the diagonal.
    assert site.chord_pdf(np.array([site.max_chord_um() * 1.001]))[0] == 0.0


def test_cylinder_is_deterministic_for_a_seed() -> None:
    a = CylindricalSite(1.0, 1.0, n_samples=200_000, seed=7)
    b = CylindricalSite(1.0, 1.0, n_samples=200_000, seed=7)
    grid = np.linspace(0.0, a.max_chord_um(), 100)
    assert np.array_equal(a.chord_pdf(grid), b.chord_pdf(grid))


def test_cylinder_rejects_bad_geometry() -> None:
    with pytest.raises(ValueError):
        CylindricalSite(0.0, 1.0)
    with pytest.raises(ValueError):
        CylindricalSite(1.0, -1.0)


def test_cylinder_spectrum_preserves_yF() -> None:
    """The analytic engine works with a cylinder site: y_F tracks LET."""
    let = 40.0
    site = CylindricalSite(1.0, 1.0, seed=0)
    engine = AnalyticSpectrumEngine(_ConstantStoppingPower(let), site)
    spec = engine.single_event_spectrum(Ion(6, 12, 100.0))
    assert spec.y_F == pytest.approx(let, rel=2e-2)
    assert spec.y_D > spec.y_F


# --- geometric limit --------------------------------------------------------


def test_geometric_limit_moments() -> None:
    """No straggling on a sphere: y_F = LET and y_D = (9/8) LET exactly.

    In the track-segment limit eps = LET*l, so y = LET*l/l_bar collapses onto
    the chord distribution: y_F = LET*<l>/l_bar = LET and
    y_D = LET*<l^2>/l_bar^2 = (9/8) LET (Kellerer 1985).
    """
    let = 50.0
    engine = AnalyticSpectrumEngine(
        stopping_power=_ConstantStoppingPower(let),
        site=SphericalSite(diameter_um=1.0),
        straggling="none",
    )
    spec = engine.single_event_spectrum(Ion(6, 12, 100.0))
    assert spec.norm == pytest.approx(1.0, rel=1e-3)
    assert spec.y_F == pytest.approx(let, rel=2e-3)
    assert spec.y_D == pytest.approx(1.125 * let, rel=3e-3)


# --- straggling -------------------------------------------------------------


def test_lognormal_straggling_preserves_yF_and_matches_analytic_yD() -> None:
    """Log-normal straggling is mean-preserving (y_F = LET) and lifts y_D exactly.

    With Bohr variance Omega^2 = c*l (c per unit length), the second moment gives
    y_D = (9/8) LET + c/(LET*l_bar); the log-normal reproduces it because it is
    moment-matched to the Bohr mean and variance and stays positive-definite.
    """
    let = 30.0
    ion = Ion(6, 12, 100.0)
    site = SphericalSite(diameter_um=1.0)
    engine = AnalyticSpectrumEngine(
        stopping_power=_ConstantStoppingPower(let),
        site=site,
        straggling="lognormal",
        chord_nodes=600,
        y_nodes=800,
    )
    spec = engine.single_event_spectrum(ion)

    l_bar = site.mean_chord_um()
    # per-unit-length variance coefficient c [keV^2/um]
    c = float(engine.straggling_variance_keV2(ion, np.array([1.0]))[0])
    expected_yD = 1.125 * let + c / (let * l_bar)

    assert spec.norm == pytest.approx(1.0, rel=1e-3)
    assert spec.y_F == pytest.approx(let, rel=5e-3)
    assert spec.y_D == pytest.approx(expected_yD, rel=1e-2)
    # Straggling broadens the spectrum: y_D above the geometric-limit value.
    assert spec.y_D > 1.125 * let


def test_gaussian_straggling_broadens_but_is_documented_biased() -> None:
    """Gaussian straggling still broadens y_D, but leaks to negative energies.

    It is a large-site approximation only; here we assert it broadens the
    spectrum (y_D above the geometric limit) without pinning y_F, which is
    knowingly biased high for a sphere's short chords (see module docstring).
    """
    let = 30.0
    ion = Ion(6, 12, 100.0)
    engine = AnalyticSpectrumEngine(
        _ConstantStoppingPower(let), SphericalSite(1.0), straggling="gaussian"
    )
    spec = engine.single_event_spectrum(ion)
    assert spec.y_D > 1.125 * let
    # Known small-site artifact: y_F is biased *high* (negative-energy leakage),
    # by an amount that grows as straggling/mean grows. Assert the direction and
    # a loose bound rather than pinning it (log-normal is the unbiased default).
    assert let < spec.y_F < 1.5 * let


def test_straggling_grows_with_charge() -> None:
    """Heavier ions at equal LET straggle more (Omega^2 ~ z_eff^2), so y_D rises."""
    let = 40.0
    site = SphericalSite(diameter_um=1.0)
    proton = Ion(1, 1, 100.0)
    carbon = Ion(6, 12, 100.0)
    # sanity: carbon has the larger effective charge at this velocity
    assert effective_charge(6, carbon.beta()) > effective_charge(1, proton.beta())

    def y_D(ion: Ion) -> float:
        engine = AnalyticSpectrumEngine(_ConstantStoppingPower(let), site)
        return engine.single_event_spectrum(ion).y_D

    assert y_D(carbon) > y_D(proton)


def test_larger_site_narrows_relative_straggling() -> None:
    """Bigger sites average more collisions: y_D approaches the geometric limit."""
    let = 30.0
    ion = Ion(6, 12, 100.0)
    small = AnalyticSpectrumEngine(_ConstantStoppingPower(let), SphericalSite(0.1))
    large = AnalyticSpectrumEngine(_ConstantStoppingPower(let), SphericalSite(5.0))
    small_excess = small.single_event_spectrum(ion).y_D - 1.125 * let
    large_excess = large.single_event_spectrum(ion).y_D - 1.125 * let
    assert large_excess < small_excess
    assert large_excess >= 0.0


def test_restricted_energy_loss_lowers_yF_and_caps_yD() -> None:
    """Restricted energy loss pulls y_F below LET and tames the runaway y_D.

    A 1 um chord traversed by a fast (100 MeV) proton has a huge Bohr variance
    relative to its tiny mean deposit, so the unrestricted y_D blows up, and its
    energetic delta-rays (T_max ~ 230 keV) escape the site so the mean deposit is
    below the unrestricted LET too. Enabling ``restricted`` drops both: the mean
    to the restricted LET L_Delta (< LET) and the variance to c*(T_cut/T_max).
    """
    let = 0.72
    ion = Ion(1, 1, 100.0)
    site = SphericalSite(diameter_um=1.0)
    unrestricted = AnalyticSpectrumEngine(_ConstantStoppingPower(let), site)
    restricted = AnalyticSpectrumEngine(_ConstantStoppingPower(let), site, restricted=True)

    # The cut ratio T_cut/T_max is the closed-form delta-ray range fraction,
    # strongly active for this small site (r_max ~ 125 um >> 1 um).
    r_max = LEM1_R_MAX_GAMMA_UM * ion.energy_MeV_u**R_MAX_DELTA
    frac = (site.max_chord_um() / r_max) ** (1.0 / R_MAX_DELTA)
    assert restricted._restricted_fraction(ion) == pytest.approx(frac)
    assert 0.0 < frac < 0.1
    # Variance scaled by the same fraction.
    c_full = float(unrestricted.straggling_variance_keV2(ion, np.array([1.0]))[0])
    c_restr = float(restricted.straggling_variance_keV2(ion, np.array([1.0]))[0])
    assert c_restr == pytest.approx(frac * c_full)

    # Restricted LET sits strictly below the unrestricted LET but stays positive.
    l_delta = restricted.restricted_let_keV_um(ion, let)
    assert 0.0 < l_delta < let

    su = unrestricted.single_event_spectrum(ion)
    sr = restricted.single_event_spectrum(ion)
    assert sr.y_F == pytest.approx(l_delta, rel=2e-2)  # y_F follows restricted LET
    assert sr.y_F < su.y_F  # ... which is below the unrestricted y_F = LET
    assert sr.y_D < 0.25 * su.y_D  # runaway tamed
    assert sr.y_D > 1.125 * l_delta  # but still broadened above the geometric limit
    assert sr.meta["let_unrestricted_keV_um"] == pytest.approx(let)


def test_restricted_energy_loss_inactive_for_large_site() -> None:
    """When the site exceeds the delta-ray range nothing is restricted.

    A slow (1 MeV) proton has a sub-um delta-ray range, so a 5 um site sees the
    whole track: the cut is inactive (fraction 1), the restricted LET equals the
    unrestricted LET, and the restricted and unrestricted spectra coincide --
    preserving the analytic large-site limit.
    """
    let = 30.0
    ion = Ion(1, 1, 1.0)
    site = SphericalSite(diameter_um=5.0)
    restricted = AnalyticSpectrumEngine(_ConstantStoppingPower(let), site, restricted=True)
    assert restricted._restricted_fraction(ion) == 1.0
    assert restricted.restricted_let_keV_um(ion, let) == pytest.approx(let)
    unrestricted = AnalyticSpectrumEngine(_ConstantStoppingPower(let), site)
    assert restricted.single_event_spectrum(ion).y_D == pytest.approx(
        unrestricted.single_event_spectrum(ion).y_D
    )


# --- delta-ray toucher tail (analytic engine) -------------------------------


def test_delta_tail_off_by_default_and_unchanged() -> None:
    """``include_delta_tail`` defaults off; the spectrum is the crosser-only one."""
    let = 40.0
    site = SphericalSite(0.6)
    ion = Ion(6, 12, 100.0)
    default = AnalyticSpectrumEngine(_ConstantStoppingPower(let), site)
    assert default.include_delta_tail is False
    spec = default.single_event_spectrum(ion)
    assert spec.meta["delta_tail"] is False
    explicit_off = AnalyticSpectrumEngine(
        _ConstantStoppingPower(let), site, include_delta_tail=False
    ).single_event_spectrum(ion)
    assert np.array_equal(spec.f, explicit_off.f)


def test_delta_tail_requires_spherical_site() -> None:
    """The toucher integral needs the spherical impact-parameter geometry."""
    engine = AnalyticSpectrumEngine(
        _ConstantStoppingPower(40.0), CylindricalSite(1.0, 1.0, seed=0), include_delta_tail=True
    )
    with pytest.raises(ValueError):
        engine.single_event_spectrum(Ion(6, 12, 100.0))


def test_delta_tail_inactive_when_range_within_site() -> None:
    """No touchers when the delta-ray range fits inside the site (r_max <= R_s).

    A 1 MeV/u carbon has ``r_max = 0.05 um`` << the 0.3 um site radius, so there
    is no toucher region and the spectrum is the crosser-only one unchanged.
    """
    sp = SimpleBetheStoppingPower()
    site = SphericalSite(0.6)
    ion = Ion(6, 12, 1.0)
    crosser = AnalyticSpectrumEngine(sp, site).single_event_spectrum(ion)
    tailed = AnalyticSpectrumEngine(sp, site, include_delta_tail=True).single_event_spectrum(ion)
    assert tailed.meta["toucher_event_fraction"] == 0.0
    assert np.array_equal(crosser.f, tailed.f)


def test_delta_tail_adds_low_y_events_and_lowers_moments() -> None:
    """The toucher tail adds many low-y events, collapsing y_F and lowering y_D.

    Touchers (track core outside the site, delta-rays reaching in) are frequent
    low-y events, so the frequency-mean y_F falls far below LET, and the
    dose-mean y_D drops from the (hot) crosser-only value.
    """
    sp = SimpleBetheStoppingPower()
    site = SphericalSite(0.6)  # Hartzell 0.3 um radius
    ion = Ion(6, 12, 10.0)
    crosser = AnalyticSpectrumEngine(sp, site).single_event_spectrum(ion)
    tailed = AnalyticSpectrumEngine(sp, site, include_delta_tail=True).single_event_spectrum(ion)

    assert tailed.meta["delta_tail"] is True
    assert 0.0 < tailed.meta["toucher_event_fraction"] < 1.0
    assert tailed.norm == pytest.approx(1.0, rel=1e-2)
    # Frequency mean collapses: touchers dominate the event count at low y.
    assert tailed.y_F < 0.2 * crosser.y_F
    # Dose mean drops toward the reference (the crosser-only value runs hot).
    assert tailed.y_D < crosser.y_D


def test_delta_tail_moves_yD_toward_hartzell_amf() -> None:
    """Enabling the tail brings y_D closer to the Hartzell 2025 AMF y_D.

    For the heavy ions at 10 MeV/u -- where the crosser-only analytic engine runs
    hot -- the toucher tail moves the dose-mean lineal energy toward the bundled
    digitized AMF reference (Hartzell et al. 2025, doi:10.1088/1361-6560/adcfec,
    Fig. 4, 0.3 um radius).
    """
    from pyradbio.benchmark.hartzell2025 import get_amf_curve

    sp = SimpleBetheStoppingPower()
    site = SphericalSite(0.6)
    for name, Z, A in [("carbon", 6, 12), ("oxygen", 8, 16), ("neon", 10, 20)]:
        ion = Ion(Z, A, 10.0)
        curve = get_amf_curve(figure="fig4", ion=name, method="AMF", energy_MeV_u=10.0)
        y, d = curve.dose_spectrum(normalized=True)
        hartzell_yd = float(np.trapezoid(y * d, y))
        crosser = AnalyticSpectrumEngine(sp, site).single_event_spectrum(ion).y_D
        tailed = (
            AnalyticSpectrumEngine(sp, site, include_delta_tail=True).single_event_spectrum(ion).y_D
        )
        assert abs(tailed - hartzell_yd) < abs(crosser - hartzell_yd)


# --- result object ----------------------------------------------------------


def test_dose_weighted_spectrum_and_y_star() -> None:
    let = 50.0
    engine = AnalyticSpectrumEngine(_ConstantStoppingPower(let), SphericalSite(1.0))
    spec = engine.single_event_spectrum(Ion(6, 12, 100.0))

    # d1(y) = y f / y_F is normalised to 1 by construction.
    assert np.trapezoid(spec.d, spec.y_keV_um) == pytest.approx(1.0, rel=1e-2)
    # y* saturates: below y_D and positive. As y0 -> inf the saturation vanishes
    # and y* -> y_D; as y0 -> 0 it -> 0 (Kase 2006).
    y_star = spec.y_star(y0_keV_um=150.0)
    assert 0.0 < y_star < spec.y_D
    assert spec.y_star(y0_keV_um=1.0e6) == pytest.approx(spec.y_D, rel=1e-2)
    assert spec.y_star(y0_keV_um=1.0e-3) < y_star


def test_calculator_facade_matches_engine() -> None:
    let = 25.0
    site = SphericalSite(0.5)
    calc = LinealEnergyCalculator(_ConstantStoppingPower(let), site, straggling="lognormal")
    engine = AnalyticSpectrumEngine(_ConstantStoppingPower(let), site, straggling="lognormal")
    ion = Ion(2, 4, 60.0)
    assert calc.spectrum(ion).y_D == pytest.approx(engine.single_event_spectrum(ion).y_D)


def test_custom_y_grid_is_respected() -> None:
    engine = AnalyticSpectrumEngine(_ConstantStoppingPower(30.0), SphericalSite(1.0))
    grid = np.logspace(-2, math.log10(400.0), 800)
    spec = engine.single_event_spectrum(Ion(6, 12, 100.0), y_grid_keV_um=grid)
    assert np.array_equal(spec.y_keV_um, grid)
    assert spec.y_F == pytest.approx(30.0, rel=2e-2)


# --- amorphous track engine -------------------------------------------------


def test_amorphous_epsilon_at_centre_equals_LET_times_diameter() -> None:
    """eps(b=0) = LET * d (a track through the centre deposits LET over a chord d).

    In the geometric limit (r_max << R_s) this is exact; here the penumbra is
    tiny (proton at 1 MeV/u, r_max ~ 0.05 um) in a 20 um sphere.
    """
    sp = SimpleBetheStoppingPower()
    rdd = ScholzKraftRadialDose.lem1(sp)
    site = SphericalSite(20.0)
    ion = Ion(1, 1, 1.0)
    s = np.logspace(math.log10(rdd.r_min_um(ion) * 1e-3), math.log10(rdd.r_max_um(ion)), 200)
    dose = np.array([rdd.local_dose_Gy(float(x), ion) for x in s])
    eps0 = float(site.mean_energy_deposited_keV(dose, s, np.array([1e-6]), angular_nodes=128)[0])
    let = sp.let_keV_um(ion)
    assert eps0 == pytest.approx(let * site.diameter_um, rel=5e-3)


def test_amorphous_epsilon_monotonic_decreasing() -> None:
    """eps(b) decreases monotonically with impact parameter (energy falls off)."""
    sp = SimpleBetheStoppingPower()
    rdd = ScholzKraftRadialDose.lem1(sp)
    site = SphericalSite(2.0)
    ion = Ion(1, 1, 5.0)
    r_max = rdd.r_max_um(ion)
    s = np.logspace(math.log10(rdd.r_min_um(ion) * 1e-3), math.log10(r_max), 200)
    dose = np.array([rdd.local_dose_Gy(float(x), ion) for x in s])
    b = np.linspace(1e-4, site.radius_um() + r_max, 400)
    eps = site.mean_energy_deposited_keV(dose, s, b)
    assert np.all(np.diff(eps) <= 1e-9)


def test_amorphous_geometric_limit_matches_analytic_none() -> None:
    """Large site (R_s >> r_max): amorphous -> geometric chord spectrum.

    y_F -> LET and y_D -> (9/8) LET, agreeing with the analytic engine's
    ``straggling='none'`` mode (the cross-check from the lineal-energy design).
    """
    sp = SimpleBetheStoppingPower()
    rdd = ScholzKraftRadialDose.lem1(sp)
    site = SphericalSite(50.0)  # r_max/R_s ~ 0.002 at 1 MeV/u
    ion = Ion(1, 1, 1.0)
    amorph = AmorphousTrackEngine(rdd, site, stopping_power=sp).single_event_spectrum(ion)
    let = amorph.meta["let_keV_um"]
    assert amorph.norm == pytest.approx(1.0, rel=1e-2)
    assert amorph.y_F == pytest.approx(let, rel=2e-2)
    assert amorph.y_D == pytest.approx(1.125 * let, rel=2e-2)


def test_amorphous_no_degeneracy_at_mkm_domain_size() -> None:
    """Regression: y_D stays physical at MKM domain sizes (r_d ~ 0.26 um), never ~0.

    Guards the two bugs where the amorphous engine collapsed to y_D ~ 0 once
    r_max >> R_s (high-energy ions in a sub-um site): a coarse *linear* impact grid
    that skipped the crosser core, and the singular ``f = pdf/|dy/db|`` spectrum
    construction. The deterministic amorphous mean sits below the (9/8) LET anchor
    (no straggling) but must be a substantial fraction of it, not the collapsed 0.
    """
    sp = SimpleBetheStoppingPower()
    kc = KieferChatterjeeRadialDose(stopping_power=sp)
    eng = AmorphousTrackEngine(kc, SphericalSite(0.52), stopping_power=sp)  # r_d = 0.26 um
    for energy in (10.0, 100.0, 300.0):
        spec = eng.single_event_spectrum(Ion(6, 12, energy))
        let = spec.meta["let_keV_um"]
        assert 0.4 * 1.125 * let < spec.y_D < 1.2 * 1.125 * let


def test_amorphous_delta_mc_lifts_light_ion_yD() -> None:
    """The DeltaRayMonteCarlo straggling layer raises the light-ion high-energy y_D.

    The deterministic amorphous *mean* under-estimates y_D for low-LET ions (no
    event-to-event straggling); the stochastic layer lifts it toward the bundled
    Hartzell 2025 AMF value (He at 100 MeV/u: deterministic ~0.5, stochastic ~0.8
    of the AMF). Also exercises the stochastic path on the fixed impact grid.
    """
    from pyradbio.benchmark.hartzell2025 import get_amf_curve

    sp = SimpleBetheStoppingPower()
    kc = KieferChatterjeeRadialDose(stopping_power=sp)
    site = SphericalSite(0.6)  # 0.3 um radius
    ion = Ion(2, 4, 100.0)
    det = AmorphousTrackEngine(kc, site, stopping_power=sp).single_event_spectrum(ion)
    mc = AmorphousTrackEngine(
        kc, site, stopping_power=sp, stochastic=DeltaRayMonteCarlo(n_samples=100_000, seed=0)
    ).single_event_spectrum(ion)

    curve = get_amf_curve(figure="fig4", ion="helium", method="AMF", energy_MeV_u=100.0)
    y, d = curve.dose_spectrum(normalized=True)
    hartzell_yd = float(np.trapezoid(y * d, y))

    assert mc.y_D > det.y_D  # straggling lifts y_D above the deterministic mean
    assert det.y_D / hartzell_yd < mc.y_D / hartzell_yd < 1.0  # ... toward the AMF


def test_amorphous_yD_tracks_hartzell_amf_for_carbon() -> None:
    """The fixed amorphous engine reproduces the Hartzell 2025 AMF y_D for carbon.

    Cross-check against the bundled digitized analytical-microdosimetric-function
    (AMF) curves (Hartzell et al. 2025, doi:10.1088/1361-6560/adcfec, Fig. 4,
    0.3 um radius) -- the amorphous track route matches the AMF dose-mean lineal
    energy to ~15% in the high-LET (carbon) regime that drives the MKM.
    """
    from pyradbio.benchmark.hartzell2025 import get_amf_curve

    sp = SimpleBetheStoppingPower()
    kc = KieferChatterjeeRadialDose(stopping_power=sp)
    eng = AmorphousTrackEngine(kc, SphericalSite(0.6), stopping_power=sp)  # 0.3 um radius
    for energy in (1.0, 10.0, 100.0):
        curve = get_amf_curve(figure="fig4", ion="carbon", method="AMF", energy_MeV_u=energy)
        y, d = curve.dose_spectrum(normalized=True)
        hartzell_yd = float(np.trapezoid(y * d, y))
        assert eng.single_event_spectrum(Ion(6, 12, energy)).y_D == pytest.approx(
            hartzell_yd, rel=0.15
        )


def test_amorphous_restricted_let_small_site() -> None:
    """Site much smaller than r_max: energy escapes, so y_F falls well below LET.

    A high-energy proton has a large penumbra; only the dose deposited inside the
    small site counts (restricted-LET behaviour), unlike the analytic engine's
    unrestricted-LET crossers.
    """
    sp = SimpleBetheStoppingPower()
    rdd = ScholzKraftRadialDose.lem1(sp)
    site = SphericalSite(0.5)
    ion = Ion(1, 1, 100.0)  # r_max >> R_s
    amorph = AmorphousTrackEngine(rdd, site, stopping_power=sp).single_event_spectrum(ion)
    assert amorph.y_F < amorph.meta["let_keV_um"]


@pytest.mark.parametrize(
    "factory", [ScholzKraftRadialDose.lem1, ScholzKraftRadialDose.lem2, ScholzKraftRadialDose.lem3]
)
def test_amorphous_flexible_over_radial_dose_models(factory) -> None:
    """The engine accepts any RadialDoseModel (Scholz-Kraft LEM I/II/III)."""
    sp = SimpleBetheStoppingPower()
    rdd = factory(sp)
    engine = AmorphousTrackEngine(
        rdd, SphericalSite(2.0), stopping_power=sp, radial_nodes=48, impact_nodes=120
    )
    spec = engine.single_event_spectrum(Ion(6, 12, 100.0))
    assert spec.norm == pytest.approx(1.0, rel=1e-2)
    assert spec.y_F > 0.0
    assert spec.y_D > spec.y_F
    assert spec.meta["radial_dose"] == "ScholzKraftRadialDose"


# --- stochastic delta-ray layer ---------------------------------------------


def test_max_delta_ray_energy_increases_with_energy() -> None:
    """T_max grows with ion velocity (kinematic head-on limit)."""
    slow = max_delta_ray_energy_keV(Ion(1, 1, 10.0))
    fast = max_delta_ray_energy_keV(Ion(1, 1, 100.0))
    assert 0.0 < slow < fast
    # ~2 m_e c^2 beta^2 gamma^2: a 100 MeV proton reaches a few hundred keV.
    assert 100.0 < fast < 400.0


def test_collision_moments_of_inverse_square_spectrum() -> None:
    """<w> and <w^2> of p(T) ~ 1/T^2 match the closed forms on [T_min, T_max]."""
    mc = DeltaRayMonteCarlo(t_min_keV=0.078)
    t_max = 20.0
    m1, m2 = mc._collision_moments(t_max)
    a, b = 0.078, t_max
    norm = 1.0 / a - 1.0 / b
    assert m1 == pytest.approx(math.log(b / a) / norm)
    assert m2 == pytest.approx((b - a) / norm)


def test_delta_mc_cures_toucher_tail_raises_yF() -> None:
    """In a small site the stochastic layer lifts y_F well above the deterministic
    amorphous value by dropping the (non-physical) zero-deposit touchers."""
    sp = SimpleBetheStoppingPower()
    rdd = ScholzKraftRadialDose.lem1(sp)
    site = SphericalSite(1.0)
    ion = Ion(1, 1, 10.0)
    det = AmorphousTrackEngine(rdd, site, stopping_power=sp).single_event_spectrum(ion)
    mc = DeltaRayMonteCarlo(n_samples=200_000, seed=0)
    sto = AmorphousTrackEngine(rdd, site, stopping_power=sp, stochastic=mc).single_event_spectrum(
        ion
    )
    assert sto.y_F > 3.0 * det.y_F
    assert 0.0 < sto.meta["event_fraction"] < 1.0
    assert sto.meta["engine"] == "amorphous+delta_mc"


def test_delta_mc_preserves_geometric_limit() -> None:
    """Large site: the compound-Poisson mean is preserved, so y_F stays ~ LET."""
    sp = SimpleBetheStoppingPower()
    rdd = ScholzKraftRadialDose.lem1(sp)
    site = SphericalSite(20.0)
    ion = Ion(1, 1, 1.0)
    mc = DeltaRayMonteCarlo(n_samples=200_000, seed=1)
    sto = AmorphousTrackEngine(rdd, site, stopping_power=sp, stochastic=mc).single_event_spectrum(
        ion
    )
    let = sto.meta["let_keV_um"]
    assert sto.y_F == pytest.approx(let, rel=5e-2)
    assert sto.y_D == pytest.approx(1.125 * let, rel=5e-2)


def test_delta_mc_low_y_floor_at_one_quantum() -> None:
    """The smallest deposit is a single ~T_min collision, giving a low-y floor.

    Below ~T_min / l_bar there is essentially no probability (the deterministic
    engine's continuous sub-quantum tail is gone).
    """
    sp = SimpleBetheStoppingPower()
    rdd = ScholzKraftRadialDose.lem1(sp)
    site = SphericalSite(1.0)
    mc = DeltaRayMonteCarlo(
        n_samples=200_000, seed=0, t_min_keV=0.078, soft_core=False, partial_deposition=False
    )
    sto = AmorphousTrackEngine(rdd, site, stopping_power=sp, stochastic=mc).single_event_spectrum(
        Ion(1, 1, 10.0)
    )
    floor = 0.078 / site.mean_chord_um()
    below = sto.f[sto.y_keV_um < 0.3 * floor]
    assert float(np.sum(below)) == 0.0


def test_delta_mc_restricts_collision_spectrum_in_small_sites() -> None:
    """Delta-rays that escape a small site cannot deposit their full energy: the
    effective T_max is capped below the kinematic value, taming the high-y tail."""
    sp = SimpleBetheStoppingPower()
    rdd = ScholzKraftRadialDose.lem1(sp)
    ion = Ion(1, 1, 100.0)
    t_max_kin = max_delta_ray_energy_keV(ion)
    mc = DeltaRayMonteCarlo(n_samples=200_000, seed=0, soft_core=False, partial_deposition=False)
    small = AmorphousTrackEngine(
        rdd, SphericalSite(1.0), stopping_power=sp, stochastic=mc
    ).single_event_spectrum(ion)
    # r_max (~125 um) >> 1 um site -> the collision spectrum is restricted.
    assert small.meta["t_max_keV"] < 0.5 * t_max_kin
    # The tail is bounded: essentially no dose above a few * T_max_eff / l_bar.
    y_cut = 5.0 * small.meta["t_max_keV"] / small.meta["mean_chord_um"]
    assert float(np.sum(small.d[small.y_keV_um > y_cut])) < 1e-3 * float(np.sum(small.d))


def test_delta_mc_no_restriction_in_large_site() -> None:
    """When the site contains the delta-ray range, the full kinematic T_max is used."""
    sp = SimpleBetheStoppingPower()
    rdd = ScholzKraftRadialDose.lem1(sp)
    ion = Ion(1, 1, 1.0)  # r_max ~ 0.05 um << 50 um site
    mc = DeltaRayMonteCarlo(n_samples=100_000, seed=0, soft_core=False, partial_deposition=False)
    spec = AmorphousTrackEngine(
        rdd, SphericalSite(50.0), stopping_power=sp, stochastic=mc
    ).single_event_spectrum(ion)
    assert spec.meta["t_max_keV"] == pytest.approx(max_delta_ray_energy_keV(ion))


def test_delta_mc_is_deterministic_for_a_seed() -> None:
    sp = SimpleBetheStoppingPower()
    rdd = ScholzKraftRadialDose.lem1(sp)
    site = SphericalSite(1.0)
    ion = Ion(1, 1, 8.0)
    grid = np.logspace(-1, 2, 150)
    a = AmorphousTrackEngine(
        rdd, site, stopping_power=sp, stochastic=DeltaRayMonteCarlo(n_samples=50_000, seed=5)
    ).single_event_spectrum(ion, y_grid_keV_um=grid)
    b = AmorphousTrackEngine(
        rdd, site, stopping_power=sp, stochastic=DeltaRayMonteCarlo(n_samples=50_000, seed=5)
    ).single_event_spectrum(ion, y_grid_keV_um=grid)
    assert np.array_equal(a.f, b.f)


def test_delta_ray_csda_range_matches_estar() -> None:
    """The CSDA range fit reproduces NIST ESTAR water values (~<5%); R_dep = CSDA/detour."""
    from pyradbio.micro.lineal_energy import DELTA_DETOUR_FACTOR

    mc = DeltaRayMonteCarlo()
    for t_keV, r_estar_um in [(1.0, 0.0457), (10.0, 2.515), (100.0, 143.1)]:
        r_dep = float(mc._deposition_range_um(np.array([t_keV]))[0])
        assert r_dep * DELTA_DETOUR_FACTOR == pytest.approx(r_estar_um, rel=0.05)


def test_delta_ray_range_matches_cole_and_estar() -> None:
    """The electron range fit agrees with two independent published relations.

    Cross-checks the ``R = A T^EXP`` fit (used for delta-ray containment) against
    the measured practical range of Cole 1969 (Radiat. Res. 38, 7-33, p.30,
    ``R = 0.0431(T+0.367)^1.77 - 0.007`` um) and the NIST ESTAR CSDA range, over
    1-100 keV. All three agree to ~7%, confirming the range is not the source of
    the high-energy y_D undershoot (see
    docs/references/Microdosimetry.md).
    """
    from pyradbio.micro.lineal_energy import DELTA_RANGE_A_UM, DELTA_RANGE_EXP

    def cole_range_um(t_keV: float) -> float:
        return 0.0431 * (t_keV + 0.367) ** 1.77 - 0.007

    # NIST ESTAR CSDA range (g/cm^2 = um at unit density) for liquid water; the
    # table starts at 10 keV. Cole's +0.367 offset makes it diverge below a few
    # keV, so the cross-check is over 10-100 keV where all three coincide.
    estar_csda_um = {10.0: 2.515, 100.0: 143.1}
    for t_keV in (10.0, 100.0):
        fit = DELTA_RANGE_A_UM * t_keV**DELTA_RANGE_EXP
        assert fit == pytest.approx(cole_range_um(t_keV), rel=0.07)
        assert fit == pytest.approx(estar_csda_um[t_keV], rel=0.07)


def test_deposit_fraction_stops_short_and_caps_long_delta_rays() -> None:
    """phi(T) = 1 for a delta contained in the in-site path, < 1 for a long-range one.

    The range is the scattering-aware energy-deposition range R_dep = R_csda / detour.
    """
    mc = DeltaRayMonteCarlo(partial_deposition=True)
    s_eff = 0.667  # um
    # A low-energy delta whose R_dep <= s_eff is contained (phi = 1).
    t_short = 0.3  # keV -> R_dep ~ few nm << s_eff
    assert float(mc._deposition_range_um(np.array([t_short]))[0]) < s_eff
    assert float(mc._deposit_fraction(np.array([t_short]), s_eff)[0]) == pytest.approx(1.0)
    # A high-energy delta (R_dep >> s_eff) deposits only ~(1/EXP) * s_eff / R_dep.
    t_long = 200.0  # keV
    r_long = float(mc._deposition_range_um(np.array([t_long]))[0])
    from pyradbio.micro.lineal_energy import DELTA_RANGE_EXP

    phi_long = float(mc._deposit_fraction(np.array([t_long]), s_eff)[0])
    assert phi_long == pytest.approx((1.0 / DELTA_RANGE_EXP) * s_eff / r_long, rel=5e-2)
    assert 0.0 < phi_long < 0.1


def test_partial_deposition_softens_tail() -> None:
    """Partial deposition lowers y_D by softening the high-y tail.

    A high-energy proton in a small site has energetic delta-rays whose range
    exceeds the site; depositing only their in-site fraction (rather than the full
    T of the restricted-cut model) removes high-y events, so y_D drops. The per-b
    mean deposit is preserved by the <w> renormalisation (dose is conserved); the
    frequency mean y_F shifts modestly as finer quanta admit more low-y events, so
    it is only loosely comparable.
    """
    sp = SimpleBetheStoppingPower()
    rdd = ScholzKraftRadialDose.lem1(sp)
    site = SphericalSite(1.0)
    ion = Ion(1, 1, 100.0)  # r_max >> R_s
    full = AmorphousTrackEngine(
        rdd,
        site,
        stopping_power=sp,
        stochastic=DeltaRayMonteCarlo(
            n_samples=200_000, seed=0, soft_core=False, partial_deposition=False
        ),
    ).single_event_spectrum(ion)
    partial = AmorphousTrackEngine(
        rdd,
        site,
        stopping_power=sp,
        stochastic=DeltaRayMonteCarlo(
            n_samples=200_000, seed=0, soft_core=False, partial_deposition=True
        ),
    ).single_event_spectrum(ion)

    assert partial.meta["partial_deposition"] is True
    assert partial.y_F == pytest.approx(full.y_F, rel=0.3)  # loosely comparable
    assert partial.y_D < full.y_D  # high-y tail softened


def test_soft_core_split_radius_options() -> None:
    """The soft/hard boundary comes from the core radius when asked, else the fixed
    cut, kept consistent through the delta-ray range law R(T) = r_max (T/T_max)^delta.
    """
    sp = SimpleBetheStoppingPower()
    rdd = ScholzKraftRadialDose.lem1(sp)
    site = SphericalSite(1.0)
    ion = Ion(1, 1, 50.0)
    r_max = rdd.r_max_um(ion)
    t_max = max_delta_ray_energy_keV(ion)

    const = AmorphousTrackEngine(
        rdd,
        site,
        stopping_power=sp,
        stochastic=DeltaRayMonteCarlo(
            soft_core=True, soft_cut_keV=1.5, soft_cut_from_core_radius=False
        ),
    )
    r0, t0 = const._soft_split_radius(ion, r_max)
    assert t0 == pytest.approx(1.5)
    assert r0 == pytest.approx(r_max * (1.5 / t_max) ** R_MAX_DELTA)

    core = AmorphousTrackEngine(
        rdd,
        site,
        stopping_power=sp,
        stochastic=DeltaRayMonteCarlo(soft_core=True, soft_cut_from_core_radius=True),
    )
    r0c, t0c = core._soft_split_radius(ion, r_max)
    assert r0c == pytest.approx(rdd.r_min_um(ion))
    assert t0c == pytest.approx(t_max * (r0c / r_max) ** (1.0 / R_MAX_DELTA))


def test_soft_core_sharpens_peak_and_lowers_yD() -> None:
    """The soft Gaussian core concentrates the crosser peak (removing the hard 1/T^2
    tail from it), lowering y_D versus the single-component model.

    Uses the Kiefer-Chatterjee dose, whose explicit dense core carries a real
    fraction of the near-track energy (unlike LEM I's diffuse 1/r^2 profile, which
    puts little energy within the soft-cut radius, so the split barely helps there).
    """
    sp = SimpleBetheStoppingPower()
    rdd = KieferChatterjeeRadialDose(sp)
    site = SphericalSite(1.0)
    ion = Ion(1, 1, 100.0)
    plain = AmorphousTrackEngine(
        rdd,
        site,
        stopping_power=sp,
        stochastic=DeltaRayMonteCarlo(
            n_samples=200_000, seed=0, partial_deposition=True, soft_core=False
        ),
    ).single_event_spectrum(ion)
    soft = AmorphousTrackEngine(
        rdd,
        site,
        stopping_power=sp,
        stochastic=DeltaRayMonteCarlo(
            n_samples=200_000,
            seed=0,
            partial_deposition=True,
            soft_core=True,
            soft_cut_from_core_radius=True,
        ),
    ).single_event_spectrum(ion)

    assert soft.meta["soft_core"] is True
    assert soft.meta["soft_cut_keV"] > 0.0
    assert soft.norm == pytest.approx(1.0, rel=1e-2)
    assert soft.y_D < plain.y_D  # sharper peak -> lower dose-mean


def test_defaults_enable_soft_core_and_partial_deposition() -> None:
    """The DeltaRayMonteCarlo defaults are the faithful Vavilov configuration."""
    mc = DeltaRayMonteCarlo()
    assert mc.partial_deposition is True
    assert mc.soft_core is True
    assert mc.soft_cut_from_core_radius is True
    assert mc.soft_variance == "collision"


def test_bohr_soft_variance_is_narrower_and_sharpens_more() -> None:
    """The Bohr soft variance (rigorous, includes sub-I collisions) is narrower than
    the 1/T^2 moments, so it sharpens the peak further (lower y_D)."""
    sp = SimpleBetheStoppingPower()
    rdd = KieferChatterjeeRadialDose(sp)
    site = SphericalSite(1.0)
    ion = Ion(1, 1, 100.0)
    let = sp.let_keV_um(ion)

    engine = AmorphousTrackEngine(
        rdd, site, stopping_power=sp, stochastic=DeltaRayMonteCarlo(soft_cut_from_core_radius=True)
    )
    _r0, t_0 = engine._soft_split_radius(ion, rdd.r_max_um(ion))
    # kappa_soft(bohr) < kappa_soft(collision) = <T^2>/<T> of the soft 1/T^2 spectrum.
    kappa_bohr = engine._soft_variance_coeff_keV(ion, let, t_0)
    m1_soft, m2_soft = DeltaRayMonteCarlo._moments_interval(DeltaRayMonteCarlo().t_min_keV, t_0)
    assert 0.0 < kappa_bohr < m2_soft / m1_soft

    def y_D(variance: str) -> float:
        mc = DeltaRayMonteCarlo(
            n_samples=200_000, seed=0, soft_cut_from_core_radius=True, soft_variance=variance
        )
        return (
            AmorphousTrackEngine(rdd, site, stopping_power=sp, stochastic=mc)
            .single_event_spectrum(ion)
            .y_D
        )

    assert y_D("bohr") < y_D("collision")


def test_hartzell_dose_spectrum_divides_out_the_leading_y() -> None:
    """``Hartzell2025Curve.dose_spectrum`` is the inline ``yd / y`` it replaced, bit for bit.

    Adopted by ``micro/sato_coefficients.py`` (the Fig.-4 coefficient fits) and the
    ``y_D`` references above; ``normalized=True`` adds the ``np.trapezoid`` unit-area
    normalisation those call sites apply.
    """
    from pyradbio.micro.hartzell2025 import load_amf_curves

    for curve in load_amf_curves():
        y_inline = np.asarray(curve.y, dtype=np.float64)
        d_inline = np.asarray(curve.yd, dtype=np.float64) / y_inline
        y, d = curve.dose_spectrum()
        assert np.array_equal(y, y_inline) and np.array_equal(d, d_inline)
        y_n, d_n = curve.dose_spectrum(normalized=True)
        assert np.array_equal(y_n, y_inline)
        assert np.array_equal(d_n, d_inline / np.trapezoid(d_inline, y_inline))
        assert float(np.trapezoid(d_n, y_n)) == pytest.approx(1.0, rel=1e-12)
