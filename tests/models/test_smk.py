"""SMK -- stochastic microdosimetric kinetic model (Inaniwa & Kanematsu 2018).

Physics acceptance for :class:`~pyradbio.models.mkm.spectrum_models.StochasticMKM`
(doi:10.1088/1361-6560/aabede; equations restated in
``docs/references/SMK_Inaniwa2018.md``):

* ``alpha_SMK`` is *identical* to the mMKM ``alpha`` (both are ``alpha0 + beta0 z*_1D``);
* ``beta_SMK = (z*_1D / z_1D) beta0`` -- quality-dependent, unlike mMKM's flat beta;
* photon limit recovers the LQ reference in both the LQ core and ``S(D)``;
* the non-LQ factor ``(1 + gamma D)`` reproduces a direct numerical average of
  ``exp(-f(z_n))`` over a compound-Poisson nucleus specific energy -- the
  independent check that eq. (3) is the right second-order expansion;
* SMK survival exceeds the LQ prediction at high dose (the over-prediction fix);
* the whole pipeline agrees with pyMKM's ``SFTable.compute(model="stochastic")``.
"""

from __future__ import annotations

import importlib.util
import math
from itertools import pairwise

import numpy as np
import pytest

from pyradbio.benchmark.hartzell2025 import load_amf_curves
from pyradbio.core.lq import LQParameters, dose_for_survival
from pyradbio.core.lq import survival as lq_survival
from pyradbio.core.radiation import Ion
from pyradbio.micro.lineal_energy import Spectrum
from pyradbio.models.mkm.spectrum_models import (
    ModifiedMKM,
    StochasticMKM,
    saturation_y0_keV_um,
)

requires_pymkm = pytest.mark.skipif(
    importlib.util.find_spec("pymkm") is None,
    reason="pyMKM not installed ([mkm] extra)",
)

ALPHA_REF = 0.217
BETA_REF = 0.0615
YD_REF = 2.5


def _lognormal_spectrum(y_D_target_keV_um: float, sigma: float = 0.35, n: int = 6000) -> Spectrum:
    """A narrow log-normal ``f1(y)`` whose dose-mean lineal energy is the target.

    Same helper as ``test_spectrum_models.py``: for a log-normal with median ``m``
    and shape ``sigma``, ``y_D = m exp(1.5 sigma^2)``; invert for ``m``.
    """
    median = y_D_target_keV_um / math.exp(1.5 * sigma * sigma)
    mu = math.log(median)
    y = np.logspace(math.log10(median) - 6.0 * sigma, math.log10(median) + 6.0 * sigma, n)
    f = np.exp(-((np.log(y) - mu) ** 2) / (2.0 * sigma * sigma)) / (
        y * math.sqrt(2.0 * math.pi) * sigma
    )
    return Spectrum(y_keV_um=y, f=f / float(np.trapezoid(f, y)))


def _spectra(y_D_domain: float, y_D_nucleus: float) -> tuple[Spectrum, Spectrum]:
    """A (domain, nucleus) spectrum pair with independently chosen dose-means."""
    return _lognormal_spectrum(y_D_domain), _lognormal_spectrum(y_D_nucleus)


# --------------------------------------------------------------------------
# eq. (1): alpha_SMK is the MKM/mMKM alpha
# --------------------------------------------------------------------------
def test_smk_alpha_equals_mmkm_alpha() -> None:
    """``alpha_SMK = alpha0 + beta0 z*_1D`` is exactly the mMKM ``alpha``.

    Both are anchored on the same reference calibration (``alpha0 = alpha_ref -
    beta0 z_d(y_D,ref)``), so for a shared geometry and spectrum they must agree
    to machine precision. This is the ``_SpectrumMKModel`` shared-calibration
    contract, not a coincidence.
    """
    smk, mmkm = StochasticMKM(nucleus_radius_um=4.3), ModifiedMKM(nucleus_radius_um=4.3)
    spec = _lognormal_spectrum(60.0)
    assert smk.alpha_beta_from_spectrum(spec).alpha == pytest.approx(
        mmkm.alpha_beta_from_spectrum(spec).alpha, rel=1e-12
    )


# --------------------------------------------------------------------------
# eq. (2): quality-dependent beta
# --------------------------------------------------------------------------
def test_smk_beta_is_saturation_ratio_times_beta0() -> None:
    """``beta_SMK = (z*_1D / z_1D) beta0`` -- transcribed straight from eq. (2)."""
    smk = StochasticMKM(nucleus_radius_um=4.3)
    spec = _lognormal_spectrum(60.0)
    y, d = spec.y_keV_um, spec.d
    ratio = smk.z_star_1D_Gy(y, d) / smk.z_1D_Gy(y, d)
    assert smk.alpha_beta_from_spectrum(spec).beta == pytest.approx(ratio * BETA_REF, rel=1e-12)


def test_smk_beta_falls_below_reference_at_high_let() -> None:
    """Saturation makes ``z*_1D < z_1D``, so ``beta_SMK < beta0`` -- unlike mMKM."""
    smk, mmkm = StochasticMKM(nucleus_radius_um=4.3), ModifiedMKM(nucleus_radius_um=4.3)
    spec = _lognormal_spectrum(200.0)
    beta_smk = smk.alpha_beta_from_spectrum(spec).beta
    assert beta_smk < BETA_REF
    assert mmkm.alpha_beta_from_spectrum(spec).beta == pytest.approx(BETA_REF)


def test_smk_beta_decreases_monotonically_with_let() -> None:
    betas = [
        StochasticMKM(nucleus_radius_um=4.3).alpha_beta_from_spectrum(_lognormal_spectrum(yd)).beta
        for yd in (5.0, 30.0, 100.0, 300.0)
    ]
    assert all(b1 > b2 for b1, b2 in pairwise(betas))


# --------------------------------------------------------------------------
# photon limit
# --------------------------------------------------------------------------
def test_smk_photon_limit_recovers_reference_lq() -> None:
    """A reference-like spectrum returns ``(alpha_ref, beta_ref)``."""
    smk = StochasticMKM(nucleus_radius_um=4.3)
    lq = smk.alpha_beta_from_spectrum(_lognormal_spectrum(YD_REF))
    assert lq.alpha == pytest.approx(ALPHA_REF, rel=2e-2)
    assert lq.beta == pytest.approx(BETA_REF, rel=2e-2)


def test_smk_photon_limit_survival_is_lq() -> None:
    """With a reference-like domain spectrum and a tiny ``z_n``, ``S(D) -> LQ``."""
    smk = StochasticMKM(nucleus_radius_um=4.3)
    domain, nucleus = _spectra(YD_REF, 1.0e-3)
    doses = np.array([0.5, 2.0, 6.0])
    s_smk = smk.survival_from_spectra(doses, domain, nucleus)
    s_lq = lq_survival_array(doses, ALPHA_REF, BETA_REF)
    assert s_smk == pytest.approx(s_lq, rel=3e-2)


def lq_survival_array(doses: np.ndarray, alpha: float, beta: float) -> np.ndarray:
    return np.array([lq_survival(float(d), alpha, beta) for d in doses])


# --------------------------------------------------------------------------
# eq. (3)-(4): the non-LQ correction
# --------------------------------------------------------------------------
def test_smk_gamma_vanishes_as_nucleus_specific_energy_goes_to_zero() -> None:
    smk = StochasticMKM(nucleus_radius_um=4.3)
    lq = smk.alpha_beta_from_spectrum(_lognormal_spectrum(60.0))
    assert smk.gamma_Gy_inv(2.0, 0.0, lq) == 0.0


def test_smk_survival_exceeds_lq_at_high_dose() -> None:
    """The ``(1 + gamma D)`` factor lifts survival above LQ -- MKM's over-prediction fix."""
    smk = StochasticMKM(nucleus_radius_um=4.3)
    domain, nucleus = _spectra(60.0, 30.0)
    lq = smk.alpha_beta_from_spectrum(domain)
    doses = np.array([1.0, 4.0, 8.0])
    s_smk = smk.survival_from_spectra(doses, domain, nucleus)
    s_lq = lq_survival_array(doses, lq.alpha, lq.beta)
    assert np.all(s_smk > s_lq)
    # ... and the gap widens with dose
    assert np.all(np.diff(s_smk / s_lq) > 0.0)


def test_smk_gamma_matches_compound_poisson_average() -> None:
    """Independent check of eq. (3) against a direct numerical average.

    ``S = E[exp(-f(z_n))]`` with ``z_n`` a compound-Poisson sum of single events.
    Build that distribution explicitly for a *mono-energetic* event (every event
    deposits the same ``z_1``, so ``z_n = k z_1`` with ``k ~ Poisson(D / z_1)``,
    giving mean ``D`` and variance ``z_1 D = z_n_bar D``), average
    ``exp(-f(k z_1))`` over ``k``, and compare with ``exp(-f(D)) (1 + gamma D)``.
    The expansion is second order, so agreement is asserted in the regime it
    claims -- small ``z_n_bar * D`` -- and the residual must shrink quadratically
    as ``z_1`` falls.
    """
    smk = StochasticMKM(nucleus_radius_um=4.3)
    alpha, beta, dose = 0.5, 0.05, 2.0
    lq = type("LQ", (), {"alpha": alpha, "beta": beta})()

    def exact(z_1: float) -> float:
        lam = dose / z_1
        k = np.arange(0, int(lam + 40.0 * math.sqrt(lam) + 60))
        from scipy.stats import poisson

        w = poisson.pmf(k, lam)
        z = k * z_1
        return float(np.sum(w * np.exp(-(alpha * z + beta * z * z))))

    residuals = []
    for z_1 in (0.04, 0.02, 0.01):
        approx = math.exp(-(alpha * dose + beta * dose**2)) * (
            1.0 + smk.gamma_Gy_inv(dose, z_1, lq) * dose
        )
        residuals.append(abs(approx - exact(z_1)) / exact(z_1))

    assert residuals[-1] < 2.0e-3  # accurate where the expansion is claimed to hold
    # halving z_1 must shrink the residual by ~4x (second-order truncation)
    assert residuals[0] / residuals[1] > 2.5
    assert residuals[1] / residuals[2] > 2.5


def test_smk_survival_is_clipped_to_a_valid_range() -> None:
    """The truncated expansion must never return ``S > 1`` or ``S < 0``."""
    smk = StochasticMKM(nucleus_radius_um=4.3)
    domain, nucleus = _spectra(300.0, 200.0)
    s = smk.survival_from_spectra(np.linspace(0.0, 40.0, 200), domain, nucleus)
    assert np.all((s >= 0.0) & (s <= 1.0))
    assert s[0] == pytest.approx(1.0)


# --------------------------------------------------------------------------
# RBE
# --------------------------------------------------------------------------
def test_smk_rbe_needs_the_nucleus_spectrum() -> None:
    """SMK is not LQ, so the inherited single-spectrum RBE must refuse to guess."""
    smk = StochasticMKM(nucleus_radius_um=4.3)
    with pytest.raises(ValueError, match="nucleus"):
        smk.rbe_at_survival_from_spectrum(_lognormal_spectrum(60.0))


def test_smk_rbe_is_lower_than_mmkm_at_high_dose_per_fraction() -> None:
    """At ``S = 0.1`` the stochastic correction reduces effectiveness vs mMKM.

    Both share ``alpha``; SMK's smaller ``beta`` and its ``(1 + gamma D)`` factor
    both push survival up at the multi-Gy doses where ``S = 0.1`` sits for a
    high-LET field, so its iso-survival RBE must come out below mMKM's.
    """
    domain, nucleus = _spectra(60.0, 30.0)
    smk = StochasticMKM(nucleus_radius_um=4.3)
    mmkm = ModifiedMKM(nucleus_radius_um=4.3)
    rbe_smk = smk.rbe_at_survival_from_spectrum(domain, nucleus_spectrum=nucleus)
    rbe_mmkm = mmkm.rbe_at_survival_from_spectrum(domain)
    assert 1.0 < rbe_smk < rbe_mmkm


def _carbon_amf_dose_spectrum(energy_MeV_u: float) -> tuple[np.ndarray, np.ndarray]:
    (curve,) = [
        c
        for c in load_amf_curves()
        if c.figure == "fig4"
        and c.ion == "carbon"
        and c.method == "AMF"
        and c.energy_MeV_u == energy_MeV_u
    ]
    y = np.asarray(curve.y, dtype=float)
    d = np.asarray(curve.yd, dtype=float) / y
    return y, d


# --------------------------------------------------------------------------
# published parameters (Inaniwa & Kanematsu 2018 table 1)
# --------------------------------------------------------------------------
@pytest.mark.golden
def test_hsg_parameters_match_inaniwa2018_table1() -> None:
    """HSG modified-SMK parameters, doi:10.1088/1361-6560/aabede table 1.

    ``alpha0 = 0.174 Gy^-1``, ``beta0 = 0.0568 Gy^-2``, ``r_d = 0.28 um``,
    ``R_n = 8.1 um``, ``z0 = 66.0 Gy``.
    """
    m = StochasticMKM.hsg_inaniwa2018()
    assert (m.alpha0, m.beta0) == (0.174, 0.0568)
    assert (m.domain_radius_um, m.nucleus_radius_um, m.z0_Gy) == (0.28, 8.1, 66.0)


@pytest.mark.golden
def test_smk_fits_z0_free_of_the_kase_geometric_relation() -> None:
    """SMK's ``z0`` is a *fitted* parameter, not the mMKM ``z0(r_d, R_n, beta0)``.

    doi:10.1088/1361-6560/aabede table 1 gives ``z0 = 66.0 Gy`` for HSG, whereas
    the Kase relation the mMKM uses would give ~121 Gy for the same geometry --
    an ~1.8x discrepancy. Pinning this stops a future refactor from silently
    re-coupling the two (the paper uses ``R_n`` *only* to derive ``z_n,D``).
    """
    m = StochasticMKM.hsg_inaniwa2018()
    fitted = m.y0_keV_um / m._z_domain_scale_keV_um_per_Gy
    assert fitted == pytest.approx(66.0, rel=1e-12)

    geometric_y0 = saturation_y0_keV_um(
        domain_radius_um=0.28, nucleus_radius_um=8.1, beta_Gy2=0.0568
    )
    geometric = geometric_y0 / m._z_domain_scale_keV_um_per_Gy
    assert geometric == pytest.approx(121.3, rel=1e-2)
    assert geometric / fitted == pytest.approx(1.84, rel=0.02)


@pytest.mark.golden
def test_nucleus_specific_energy_is_three_orders_below_the_domain() -> None:
    """ "``z_n,D`` about three orders of magnitude lower than ``z_d,D``" (sec 4.2.1).

    doi:10.1088/1361-6560/aabede: the deposition per event scales with the site
    radius while the mass scales with its cube, so ``z ~ 1/r^2`` and the ratio is
    the geometric ``(R_n / r_d)^2 = (8.1 / 0.28)^2 = 837`` -- independent of ion
    and energy. An ``R_n = 4.3 um`` would give only ~190 (two orders) and badly
    overstate ``z_n``.
    """
    m = StochasticMKM.hsg_inaniwa2018()
    for e in (1.0, 10.0, 100.0):
        y, d = _carbon_amf_dose_spectrum(e)
        ratio = m.z_1D_Gy(y, d) / m.z_nucleus_Gy(y, d)
        assert ratio == pytest.approx((8.1 / 0.28) ** 2, rel=1e-9)
        assert 500.0 < ratio < 1500.0  # "about three orders of magnitude"


@pytest.mark.golden
def test_smk_stays_within_its_expansion_range_over_the_papers_dose_range() -> None:
    """The paper applies eq. (24) to therapeutic doses; the guard must agree.

    doi:10.1088/1361-6560/aabede states the SMK formalism is accurate for
    ``D <~ 10 Gy`` for a therapeutic carbon beam (sec 2) and plans up to 15 Gy
    (RBE) (sec 4.2.3). With the paper's *own* HSG parameters the ``gamma D = 1``
    ceiling sits at 6.3 / 3.8 / 17.8 Gy for carbon at 1 / 10 / 100 MeV/u -- i.e.
    the same order as the doses the paper works at, not far below them.
    """
    m = StochasticMKM.hsg_inaniwa2018()
    ceilings = {}
    for e in (1.0, 10.0, 100.0):
        y, d = _carbon_amf_dose_spectrum(e)
        spectrum = Spectrum(y_keV_um=y, f=d / y)
        ceilings[e] = m.survival_valid_to_Gy_from_spectra(spectrum, spectrum)
    assert ceilings[1.0] == pytest.approx(6.31, rel=0.02)
    assert ceilings[10.0] == pytest.approx(3.82, rel=0.02)
    assert ceilings[100.0] == pytest.approx(17.81, rel=0.02)


def test_smk_rbe_from_dose_spectrum_needs_the_nucleus_pair() -> None:
    smk = StochasticMKM(nucleus_radius_um=4.3)
    spec = _lognormal_spectrum(60.0)
    with pytest.raises(ValueError, match="nucleus"):
        smk.rbe_at_survival_from_dose_spectrum(spec.y_keV_um, spec.d)


def test_smk_rbe_rejects_an_out_of_range_survival_fraction() -> None:
    smk = StochasticMKM(nucleus_radius_um=4.3)
    domain, nucleus = _spectra(60.0, 30.0)
    with pytest.raises(ValueError, match="survival fraction"):
        smk.rbe_at_survival_from_spectrum(domain, 0.0, nucleus_spectrum=nucleus)


def test_smk_rbe_brackets_a_dose_beyond_the_reference_dose() -> None:
    """A sub-reference-quality field has ``RBE < 1``, so the bracket must expand.

    ``D_ion(S) > D_ref(S)`` there, which exercises the bracket-doubling loop in the
    root solve rather than the usual ``D_ion < D_ref`` case.
    """
    smk = StochasticMKM(nucleus_radius_um=4.3)
    domain, nucleus = _spectra(0.5, 0.5)  # y_D well below the 2.5 keV/um reference
    assert smk.rbe_at_survival_from_spectrum(domain, nucleus_spectrum=nucleus) < 1.0


def test_smk_degenerates_gracefully_without_energy_deposition() -> None:
    """``z_1D -> 0`` leaves no quality dependence to apply."""
    smk = StochasticMKM(nucleus_radius_um=4.3)
    lq = smk.alpha_beta_from_specific_energies(z_star_1D_Gy=0.0, z_1D_Gy=0.0)
    assert lq.alpha == pytest.approx(smk.alpha0)
    assert lq.beta == pytest.approx(BETA_REF)


def test_smk_validity_limit_is_unreachable_for_a_negligible_nucleus_dose() -> None:
    """As ``z_n -> 0`` the validity ceiling runs away to absurd doses.

    ``gamma D`` still reaches the limit *eventually* (the ``beta D^2`` term keeps
    growing), so the honest answer is a finite but astronomically large dose rather
    than ``inf`` -- the point is that nothing clinical is anywhere near it.
    """
    smk = StochasticMKM(nucleus_radius_um=4.3)
    flat = _lognormal_spectrum(1.0e-12)
    assert smk.survival_valid_to_Gy_from_spectra(flat, flat) > 1.0e6


def test_smk_rbe_for_ion_uses_both_default_engines() -> None:
    smk = StochasticMKM(nucleus_radius_um=4.0, domain_radius_um=0.26, yD_ref_keV_um=4.3)
    assert smk.rbe_at_survival(Ion(6, 12, 100.0), 0.1) > 1.0


def test_smk_carbon_rbe_rises_with_let_and_sits_below_mmkm() -> None:
    """Carbon RBE rises with LET, and SMK is less effective than mMKM.

    Evaluated at ``S = 0.2`` and 10/100 MeV/u -- a point the model itself certifies
    as inside the eq.-(4) expansion's validity range (asserted below, not assumed).
    At the conventional ``S = 0.1`` the 10 MeV/u iso-survival dose already exceeds
    the ceiling, so an RBE_10 trend there would be comparing extrapolations.
    """
    survival_fraction = 0.2
    smk = StochasticMKM(nucleus_radius_um=4.3)
    mmkm = ModifiedMKM(nucleus_radius_um=4.3)
    d_ref = dose_for_survival(survival_fraction, ALPHA_REF, BETA_REF)

    rbe_smk, rbe_mmkm = {}, {}
    for e in (10.0, 100.0):
        y, d = _carbon_amf_dose_spectrum(e)
        spectrum = Spectrum(y_keV_um=y, f=d / y)
        rbe_smk[e] = smk.rbe_at_survival_from_dose_spectrum(
            y, d, survival_fraction, nucleus_dose_spectrum=(y, d)
        )
        rbe_mmkm[e] = mmkm.rbe_at_survival_from_dose_spectrum(y, d, survival_fraction)
        # the dose this RBE was read at must sit inside the expansion's range
        assert d_ref / rbe_smk[e] <= smk.survival_valid_to_Gy_from_spectra(spectrum, spectrum)

    assert all(v > 1.0 for v in rbe_smk.values())
    assert rbe_smk[10.0] > rbe_smk[100.0]  # rising limb (LET increases 100 -> 10 MeV/u)
    # smaller beta (eq. 2) + the (1 + gamma D) lift (eq. 4) both reduce effectiveness
    assert all(rbe_smk[e] < rbe_mmkm[e] for e in rbe_smk)


def test_smk_deep_overkill_falls_outside_the_expansion_validity_range() -> None:
    """1 MeV/u carbon at ``S = 0.1`` is beyond where eq. (4) is a controlled expansion.

    ``z_n`` is large enough there that ``gamma D`` exceeds unity (a >100%
    correction) well before the iso-survival dose is reached, so the truncated
    second-order expansion is extrapolating. The model must *report* that via
    :meth:`survival_valid_to_Gy` rather than return a confident number -- this
    pins the guard, and documents why the trend test above stops at 10 MeV/u.
    """
    smk = StochasticMKM(nucleus_radius_um=4.3)
    y, d = _carbon_amf_dose_spectrum(1.0)
    spectrum = Spectrum(y_keV_um=y, f=d / y)
    valid_to = smk.survival_valid_to_Gy_from_spectra(spectrum, spectrum)
    d_ref_at_10pct = dose_for_survival(0.1, ALPHA_REF, BETA_REF)
    assert valid_to < d_ref_at_10pct

    # ... while a low-LET (100 MeV/u) field stays valid well past the same dose
    y, d = _carbon_amf_dose_spectrum(100.0)
    spectrum = Spectrum(y_keV_um=y, f=d / y)
    assert smk.survival_valid_to_Gy_from_spectra(spectrum, spectrum) > d_ref_at_10pct


# --------------------------------------------------------------------------
# end-to-end + pyMKM oracle
# --------------------------------------------------------------------------
def test_smk_alpha_beta_uses_default_engines_at_both_site_sizes() -> None:
    """``alpha_beta(ion)`` / ``survival(ion, D)`` work with no engine injected.

    The nucleus spectrum is scored by a *second* default engine sized to ``R_n``;
    this is the one structural difference from mMKM/MCF-MKM.
    """
    smk = StochasticMKM(nucleus_radius_um=4.0, domain_radius_um=0.26, yD_ref_keV_um=4.3)
    alpha = {e: smk.alpha_beta(Ion(6, 12, e)).alpha for e in (10.0, 100.0)}
    assert all(a > 0.0 for a in alpha.values())
    assert alpha[10.0] > alpha[100.0]

    s = smk.survival(Ion(6, 12, 10.0), np.array([0.0, 2.0, 4.0]))
    assert s[0] == pytest.approx(1.0)
    assert np.all(np.diff(s) < 0.0)


@requires_pymkm
def test_smk_survival_matches_pymkm_stochastic_model() -> None:
    """Cross-check the SMK algebra against pyMKM's ``model="stochastic"``.

    pyMKM is the oracle for the MKM family (``AGENTS.md``).
    Feeding *its* microdosimetric quantities (``z*_d``, ``z_d``, ``z_n``) into our
    equations must reproduce its survival curve to machine precision -- this pins
    eqs. (1)-(4) independently of which track-structure engine produced the
    spectra (that comparison is a separate, looser test).
    """
    from pymkm.mktable.compute import _compute_for_energy_let_pair

    from pyradbio.track.stopping_power import SimpleBetheStoppingPower

    r_d, r_n, alpha0, beta0 = 0.32, 3.9, 0.172, 0.0615  # Inaniwa HSG geometry
    z0 = 15.0  # Gy; any positive saturation parameter exercises the same algebra
    ion = Ion(6, 12, 12.0)
    # pyMKM's ParticleTrack takes LET in MeV/cm = 10 * keV/um.
    let_MeV_cm = 10.0 * SimpleBetheStoppingPower().let_keV_um(ion)
    result = _compute_for_energy_let_pair(
        {
            "model_name": "Kiefer-Chatterjee",
            "core_radius_type": "energy-dependent",
            "base_points_b": 150,
            "base_points_r": 150,
            "domain_radius": r_d,
            "nucleus_radius": r_n,
            "z0": z0,
            "use_stochastic_model": True,
            "integration_method": "trapz",
        },
        energy=ion.energy_MeV_u,
        let=let_MeV_cm,
        atomic_number=6,
    )
    z_star_d = float(result["z_bar_star_domain"])
    z_d = float(result["z_bar_domain"])
    z_n = float(result["z_bar_nucleus"])

    doses = np.linspace(0.0, 10.0, 25)

    # pyMKM's own expressions (sftable/compute.py, model="stochastic")
    a_expected = alpha0 + beta0 * z_star_d
    b_expected = (z_star_d / z_d) * beta0
    g_expected = z_n * (0.5 * (a_expected + 2 * b_expected * doses) ** 2 - b_expected)
    s_expected = np.exp(-(a_expected * doses + b_expected * doses**2)) * (1 + g_expected * doses)

    smk = StochasticMKM(nucleus_radius_um=r_n, domain_radius_um=r_d, beta_ref=beta0)
    lq = smk.alpha_beta_from_specific_energies(z_star_1D_Gy=z_star_d, z_1D_Gy=z_d, alpha0=alpha0)
    assert lq.alpha == pytest.approx(a_expected, rel=1e-12)
    assert lq.beta == pytest.approx(b_expected, rel=1e-12)

    s_ours = smk.survival_from_specific_energies(
        doses, z_star_1D_Gy=z_star_d, z_1D_Gy=z_d, z_nucleus_Gy=z_n, alpha0=alpha0
    )
    assert s_ours == pytest.approx(np.clip(s_expected, 0.0, 1.0), rel=1e-12)


# --------------------------------------------------------------------------
# Sato & Furusawa 2012 eq. (33): the domain_treatment axis
#
# Sato's SMK is the exact parent of Inaniwa's. Its eq. (33) is
#     ln(S_n(z_n)) = -alpha0 z_n (z*_d,F / z_d,F)
#                    - beta0 (z_d,D z_n + z_n^2) (z*_d,D / z_d,D)
# whose beta is IDENTICAL to Inaniwa eq. (2), while its alpha carries an extra
# saturation ratio built on the *frequency* mean (Sato eq. 30) that Inaniwa
# takes as 1. Read visually from the PDF, doi:10.1667/RR2842.1.
# --------------------------------------------------------------------------
def _frequency_saturation_ratio(smk: StochasticMKM, spec: Spectrum) -> float:
    """``z*_d,F / z_d,F`` by direct quadrature on ``f(y)`` -- the reference.

    Independent of the implementation: it weights by the *frequency* density
    ``f(y)`` rather than going through the dose-weighted ``d(y)``, and the
    domain z-scale cancels in the ratio.
    """
    y, f = spec.y_keV_um, spec.f
    y0 = smk.y0_keV_um
    y_sat = y0 * np.sqrt(-np.expm1(-((y / y0) ** 2)))
    return float(np.trapezoid(y_sat * f, y)) / float(np.trapezoid(y * f, y))


def test_default_domain_treatment_is_inaniwa_and_unchanged() -> None:
    """The shipped default must not move when the axis is added."""
    smk = StochasticMKM(nucleus_radius_um=4.3)
    assert smk.domain_treatment == "inaniwa_moments"
    spec = _lognormal_spectrum(60.0)
    lq = smk.alpha_beta_from_spectrum(spec)
    # alpha0 + beta0 z*_1D, with no frequency-mean factor
    expected_alpha = smk.alpha0 + smk.beta0 * smk.z_star_1D_Gy(spec.y_keV_um, spec.d)
    assert lq.alpha == pytest.approx(expected_alpha, rel=1e-12)


def test_frequency_saturation_ratio_matches_direct_quadrature() -> None:
    """``z*_d,F / z_d,F`` (Sato eq. 30) against a frequency-weighted reference."""
    smk = StochasticMKM(nucleus_radius_um=4.3)
    for y_D in (10.0, 60.0, 200.0):
        spec = _lognormal_spectrum(y_D)
        y, d = spec.y_keV_um, spec.d
        got = smk.z_star_F_Gy(y, d) / smk.z_F_Gy(y, d)
        assert got == pytest.approx(_frequency_saturation_ratio(smk, spec), rel=2e-3)


def test_frequency_saturation_ratio_is_unity_without_saturation() -> None:
    """As ``z0 -> infinity`` the correction switches off and the ratio -> 1."""
    smk = StochasticMKM(nucleus_radius_um=4.3, z0_Gy=1e9)
    spec = _lognormal_spectrum(60.0)
    y, d = spec.y_keV_um, spec.d
    assert smk.z_star_F_Gy(y, d) / smk.z_F_Gy(y, d) == pytest.approx(1.0, rel=1e-6)


def test_frequency_saturation_ratio_falls_with_let() -> None:
    smk = StochasticMKM(nucleus_radius_um=4.3)
    ratios = []
    for y_D in (5.0, 20.0, 80.0, 300.0):
        spec = _lognormal_spectrum(y_D)
        ratios.append(smk.z_star_F_Gy(spec.y_keV_um, spec.d) / smk.z_F_Gy(spec.y_keV_um, spec.d))
    assert all(a > b for a, b in pairwise(ratios))
    assert all(0.0 < r <= 1.0 for r in ratios)


def test_frequency_ratio_saturates_less_than_the_dose_ratio() -> None:
    """The frequency mean weights low ``z``, where saturation barely bites.

    So ``z*_F/z_F`` must sit *above* ``z*_D/z_D`` at the same quality -- which is
    why dropping it (as Inaniwa does) is a smaller error than dropping the dose
    ratio would be.
    """
    smk = StochasticMKM(nucleus_radius_um=4.3)
    spec = _lognormal_spectrum(200.0)
    y, d = spec.y_keV_um, spec.d
    freq = smk.z_star_F_Gy(y, d) / smk.z_F_Gy(y, d)
    dose = smk.z_star_1D_Gy(y, d) / smk.z_1D_Gy(y, d)
    assert freq > dose


def test_sato_moments_lowers_alpha_at_high_let() -> None:
    """Sato's extra factor is < 1 at high LET, so his alpha is the lower one."""
    inaniwa = StochasticMKM(nucleus_radius_um=4.3)
    sato = StochasticMKM(nucleus_radius_um=4.3, domain_treatment="sato_moments")
    spec = _lognormal_spectrum(200.0)
    assert sato.alpha_beta_from_spectrum(spec).alpha < inaniwa.alpha_beta_from_spectrum(spec).alpha


def test_sato_moments_converges_to_inaniwa_at_low_let() -> None:
    """The two readings agree where saturation is negligible."""
    inaniwa = StochasticMKM(nucleus_radius_um=4.3)
    sato = StochasticMKM(nucleus_radius_um=4.3, domain_treatment="sato_moments")
    spec = _lognormal_spectrum(1.0)
    assert sato.alpha_beta_from_spectrum(spec).alpha == pytest.approx(
        inaniwa.alpha_beta_from_spectrum(spec).alpha, rel=2e-3
    )


def test_sato_moments_leaves_beta_identical() -> None:
    """Eq. (33)'s beta IS Inaniwa eq. (2) -- the papers differ only in alpha."""
    inaniwa = StochasticMKM(nucleus_radius_um=4.3)
    sato = StochasticMKM(nucleus_radius_um=4.3, domain_treatment="sato_moments")
    for y_D in (10.0, 200.0):
        spec = _lognormal_spectrum(y_D)
        assert sato.alpha_beta_from_spectrum(spec).beta == pytest.approx(
            inaniwa.alpha_beta_from_spectrum(spec).beta, rel=1e-12
        )


def test_sato_moments_alpha_is_eq33_transcribed() -> None:
    """alpha = alpha0 (z*_d,F/z_d,F) + beta0 z*_d,D, straight from eq. (33)."""
    sato = StochasticMKM(nucleus_radius_um=4.3, domain_treatment="sato_moments")
    spec = _lognormal_spectrum(120.0)
    y, d = spec.y_keV_um, spec.d
    expected = sato.alpha0 * (sato.z_star_F_Gy(y, d) / sato.z_F_Gy(y, d)) + sato.beta0 * (
        sato.z_star_1D_Gy(y, d)
    )
    assert sato.alpha_beta_from_spectrum(spec).alpha == pytest.approx(expected, rel=1e-12)


def test_unknown_domain_treatment_is_rejected() -> None:
    with pytest.raises(ValueError):
        StochasticMKM(nucleus_radius_um=4.3, domain_treatment="nonsense")


# --------------------------------------------------------------------------
# The axis must reach the ION path too -- or refuse to run there.
#
# Regression for a real defect: domain_treatment was wired only into
# alpha_beta_from_dose_spectrum, so alpha_beta_for_ion and survival silently
# ignored it. Every test above goes through the spectrum path, which is exactly
# why none of them caught it.
# --------------------------------------------------------------------------
def _amorphous_engine(radius_um: float) -> object:
    from pyradbio.micro.lineal_energy import AmorphousTrackEngine, SphericalSite
    from pyradbio.track.kiefer_chatterjee import KieferChatterjeeRadialDose
    from pyradbio.track.stopping_power import SimpleBetheStoppingPower

    sp = SimpleBetheStoppingPower()
    return AmorphousTrackEngine(
        radial_dose=KieferChatterjeeRadialDose(stopping_power=sp),
        site=SphericalSite(diameter_um=2.0 * radius_um),
        stopping_power=sp,
    )


def test_sato_moments_reaches_the_ion_path_with_an_injected_engine() -> None:
    """With a real d(y) the ion path must honour the axis, not ignore it."""
    ion = Ion(Z=6, A=12, energy_MeV_u=10.0)
    kwargs = {
        "nucleus_radius_um": 4.3,
        "engine": _amorphous_engine(0.28),
        "nucleus_engine": _amorphous_engine(4.3),
    }
    inaniwa = StochasticMKM(**kwargs)
    sato = StochasticMKM(**kwargs, domain_treatment="sato_moments")
    a_i = inaniwa.alpha_beta(ion).alpha
    a_s = sato.alpha_beta(ion).alpha
    assert a_s < a_i, "sato_moments must lower alpha at high LET on the ion path too"
    assert a_s / a_i < 0.999


def test_sato_moments_survival_reaches_the_ion_path() -> None:
    ion = Ion(Z=6, A=12, energy_MeV_u=10.0)
    kwargs = {
        "nucleus_radius_um": 4.3,
        "engine": _amorphous_engine(0.28),
        "nucleus_engine": _amorphous_engine(4.3),
    }
    inaniwa = StochasticMKM(**kwargs)
    sato = StochasticMKM(**kwargs, domain_treatment="sato_moments")
    # lower alpha -> higher survival at the same dose
    assert float(sato.survival(ion, 2.0)) > float(inaniwa.survival(ion, 2.0))


def test_sato_moments_refuses_the_impact_parameter_route() -> None:
    """No engine -> no d(y) -> no event definition -> the frequency mean is undefined.

    The impact-parameter route has no minimum-energy-deposition threshold, so its
    frequency mean depends on the conventional ``b_max`` cutoff: measured over a
    20x cut of b_max on 12C at 10 MeV/u, ``z*_F/z_F`` drifts 8% and never
    plateaus, while the dose ratio ``z*_D/z_D`` is flat to four digits. Returning
    a cutoff-dependent alpha would be exactly the silent physics AGENTS.md
    forbids, so this raises instead.
    """
    sato = StochasticMKM(nucleus_radius_um=4.3, domain_treatment="sato_moments")
    with pytest.raises(ValueError, match="impact-parameter"):
        sato.alpha_beta(Ion(Z=6, A=12, energy_MeV_u=10.0))


def test_inaniwa_default_still_runs_on_the_impact_parameter_route() -> None:
    """The refusal is specific to sato_moments; the shipped default is untouched."""
    smk = StochasticMKM(nucleus_radius_um=4.3)
    assert smk.alpha_beta(Ion(Z=6, A=12, energy_MeV_u=10.0)).alpha > 0.0


# --------------------------------------------------------------------------
# nucleus_treatment: the Taylor expansion vs the numerical integral
#
# Sato & Furusawa 2012 eq. (7):  S(D) = integral S_n(z_n) f_n(z_n; D) dz_n.
# Inaniwa & Kanematsu 2018 replaces that integral with a second-order Taylor
# expansion about z_n = D, which is the shipped (1 + gamma D) factor. Both are
# the same model; the numerical route is the parent.
#
# Per the lesson of the domain_treatment defect, the ION entry point is tested
# here FIRST, not last.
# --------------------------------------------------------------------------
def test_default_nucleus_treatment_is_taylor_and_unchanged() -> None:
    smk = StochasticMKM.hsg_inaniwa2018()
    assert smk.nucleus_treatment == "taylor"
    ion = Ion(Z=6, A=12, energy_MeV_u=10.0)
    doses = np.array([0.5, 2.0, 5.0])
    z = smk.specific_energies(ion)
    expected = smk.survival_from_specific_energies(
        doses,
        z_star_1D_Gy=z.z_star_1D_Gy,
        z_1D_Gy=z.z_1D_Gy,
        z_nucleus_Gy=z.z_nucleus_Gy,
    )
    assert np.allclose(smk.survival(ion, doses), expected, rtol=1e-12)


def test_numerical_nucleus_treatment_reaches_the_ion_path() -> None:
    """The axis must change `survival(ion, ...)`, not only a seam method."""
    ion = Ion(Z=6, A=12, energy_MeV_u=10.0)
    taylor = StochasticMKM.hsg_inaniwa2018()
    numerical = StochasticMKM.hsg_inaniwa2018(nucleus_treatment="numerical")
    doses = np.array([1.0, 4.0, 8.0])
    assert not np.allclose(numerical.survival(ion, doses), taylor.survival(ion, doses), rtol=1e-6)


def test_numerical_and_taylor_agree_where_the_expansion_is_valid() -> None:
    """At low LET and low dose the correction is tiny and both must coincide."""
    ion = Ion(Z=2, A=4, energy_MeV_u=100.0)  # low LET helium
    taylor = StochasticMKM.hsg_inaniwa2018()
    numerical = StochasticMKM.hsg_inaniwa2018(nucleus_treatment="numerical")
    doses = np.array([0.25, 0.5, 1.0])
    assert np.allclose(numerical.survival(ion, doses), taylor.survival(ion, doses), rtol=0.02)


def test_numerical_survival_is_always_a_valid_survival() -> None:
    """A weighted average of exp(-...) is in (0, 1] by construction.

    The Taylor route's ``(1 + gamma D)`` grows without bound and has to be
    clipped -- ``expansion_validity_limit`` exists to say where that starts. The
    integral needs no such ceiling, which is the practical reason to keep the
    parent around.
    """
    ion = Ion(Z=10, A=20, energy_MeV_u=6.0)  # deep overkill, where Taylor breaks
    numerical = StochasticMKM.hsg_inaniwa2018(nucleus_treatment="numerical")
    s = numerical.survival(ion, np.array([0.5, 5.0, 20.0, 60.0]))
    assert np.all(s > 0.0) and np.all(s <= 1.0)
    assert np.all(np.diff(s) < 0.0), "survival must fall monotonically with dose"


def test_numerical_exceeds_the_lq_prediction_at_high_let() -> None:
    """The whole point: nucleus stochastics raise survival above plain LQ."""
    ion = Ion(Z=6, A=12, energy_MeV_u=10.0)
    numerical = StochasticMKM.hsg_inaniwa2018(nucleus_treatment="numerical")
    lq = numerical.alpha_beta(ion)
    dose = 6.0
    assert float(numerical.survival(ion, dose)) > lq_survival(dose, lq.alpha, lq.beta)


def test_numerical_matches_a_monte_carlo_of_the_same_average() -> None:
    """Independent check on eq. (7): sample z_n, do not transform it."""
    from pyradbio.micro.multi_event import single_event_density

    smk = StochasticMKM.hsg_inaniwa2018(nucleus_treatment="numerical")
    lq = LQParameters(0.5, 0.06)  # the paper's own fig. A1 coefficients
    z1, p1 = single_event_density(let_keV_um=200.0, radius_um=6.2)
    dose = 4.0

    exact = float(smk.survival_from_nucleus_density(dose, lq=lq, z_Gy=z1, single_event_mass=p1))

    z_F = float(np.sum(z1 * p1))
    rng = np.random.default_rng(11)
    n = 300_000
    counts = rng.poisson(dose / z_F, size=n)
    draws = rng.choice(z1, size=(n, int(counts.max()) + 1), p=p1)
    mask = np.arange(draws.shape[1])[None, :] < counts[:, None]
    z_n = (draws * mask).sum(axis=1)
    sampled = float(np.mean(np.exp(-(lq.alpha * z_n + lq.beta * z_n**2))))
    assert exact == pytest.approx(sampled, rel=0.01)


def test_numerical_photon_limit_recovers_the_lq_reference() -> None:
    """As the single-event z_n -> 0 the integral collapses onto plain LQ."""
    smk = StochasticMKM.hsg_inaniwa2018(nucleus_treatment="numerical")
    lq = LQParameters(0.3, 0.05)
    # a very low LET in a large nucleus: many tiny events
    z1, p1 = _tiny_event_density()
    dose = 3.0
    got = float(smk.survival_from_nucleus_density(dose, lq=lq, z_Gy=z1, single_event_mass=p1))
    assert got == pytest.approx(lq_survival(dose, lq.alpha, lq.beta), rel=5e-3)


def _tiny_event_density() -> tuple[np.ndarray, np.ndarray]:
    from pyradbio.micro.multi_event import single_event_density

    return single_event_density(let_keV_um=0.3, radius_um=8.1)


def test_unknown_nucleus_treatment_is_rejected() -> None:
    with pytest.raises(ValueError):
        StochasticMKM(nucleus_radius_um=4.3, nucleus_treatment="nonsense")


# --------------------------------------------------------------------------
# ...and the axes must reach the RBE / validity entry points too.
#
# Same class of regression as the domain_treatment axis above: both axes must be
# wired into ``rbe_at_survival`` as well as ``survival``. Otherwise the LQ pair is
# re-derived without the eq.-(33) factor and the Taylor form is always inverted,
# which gives bit-identical RBE across both axes at every energy. RBE is the
# quantity a planning system actually consumes, so a silent no-op there is the
# worst place for one.
# --------------------------------------------------------------------------
def test_numerical_nucleus_treatment_reaches_the_rbe(monkeypatch: pytest.MonkeyPatch) -> None:
    """The exact integral lies above the expansion, so it needs *more* dose."""
    ion = Ion(Z=6, A=12, energy_MeV_u=10.0)
    taylor = StochasticMKM.hsg_inaniwa2018()
    numerical = StochasticMKM.hsg_inaniwa2018(nucleus_treatment="numerical")
    r_t = taylor.rbe_at_survival(ion, 0.1)
    r_n = numerical.rbe_at_survival(ion, 0.1)
    assert r_n != r_t, "nucleus_treatment must reach rbe_at_survival"
    # S_numerical > S_taylor at fixed dose => larger iso-survival dose => lower RBE
    assert r_n < r_t


def test_sato_moments_reaches_the_rbe() -> None:
    """Lower alpha at high LET must show up as a lower RBE, not be discarded."""
    ion = Ion(Z=6, A=12, energy_MeV_u=10.0)
    kwargs = {
        "nucleus_radius_um": 4.3,
        "engine": _amorphous_engine(0.28),
        "nucleus_engine": _amorphous_engine(4.3),
    }
    inaniwa = StochasticMKM(**kwargs)
    sato = StochasticMKM(**kwargs, domain_treatment="sato_moments")
    assert sato.rbe_at_survival(ion, 0.1) < inaniwa.rbe_at_survival(ion, 0.1)


def test_sato_moments_reaches_the_validity_ceiling() -> None:
    """``survival_valid_to_Gy`` re-derives the pair -- with the same factor."""
    ion = Ion(Z=6, A=12, energy_MeV_u=10.0)
    kwargs = {
        "nucleus_radius_um": 4.3,
        "engine": _amorphous_engine(0.28),
        "nucleus_engine": _amorphous_engine(4.3),
    }
    inaniwa = StochasticMKM(**kwargs)
    sato = StochasticMKM(**kwargs, domain_treatment="sato_moments")
    assert sato.survival_valid_to_Gy(ion) != inaniwa.survival_valid_to_Gy(ion)


def test_the_numerical_route_has_no_expansion_ceiling() -> None:
    """It is a weighted average of survivals, so there is nothing to outgrow."""
    ion = Ion(Z=6, A=12, energy_MeV_u=10.0)
    numerical = StochasticMKM.hsg_inaniwa2018(nucleus_treatment="numerical")
    assert numerical.survival_valid_to_Gy(ion) == math.inf


def test_the_spectrum_route_refuses_the_numerical_treatment() -> None:
    """A nucleus ``d(y)`` is not the single-event ``z`` density the integral needs.

    Silently falling back to the Taylor form is the failure mode this whole
    section exists to prevent, so the spectrum-driven entry points raise.
    """
    smk = StochasticMKM.hsg_inaniwa2018(nucleus_treatment="numerical")
    ion = Ion(Z=6, A=12, energy_MeV_u=10.0)
    domain = _amorphous_engine(0.28).single_event_spectrum(ion)  # type: ignore[attr-defined]
    nucleus = _amorphous_engine(8.1).single_event_spectrum(ion)  # type: ignore[attr-defined]
    with pytest.raises(ValueError, match="nucleus_treatment"):
        smk.survival_from_spectra(2.0, domain, nucleus)
    with pytest.raises(ValueError, match="nucleus_treatment"):
        smk.rbe_at_survival_from_spectrum(domain, nucleus_spectrum=nucleus)


def test_osmk_rejects_the_smk_treatment_axes() -> None:
    """OSMK has its own published ``alpha`` form; neither axis is derived for it.

    Inheriting the fields silently would make them no-ops there -- the exact
    defect this section documents -- so they are refused instead.
    """
    from pyradbio.models.mkm.osmk import OSMK2021

    with pytest.raises(ValueError, match="OSMK"):
        OSMK2021.hsg_inaniwa2021(domain_treatment="sato_moments")
    with pytest.raises(ValueError, match="OSMK"):
        OSMK2021.hsg_inaniwa2021(nucleus_treatment="numerical")
