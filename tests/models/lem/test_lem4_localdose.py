"""LEM IV full-simulation ``beta_I`` via the local-dose convolution.

The Friedrich et al. 2012 (doi:10.3109/09553002.2011.611213) Fig. 4 full
simulation is reconstructed by making the lethal-event density a function of
each subvolume's *accumulated local dose*, per the Elsasser & Scholz 2007
(doi:10.1667/RR0467.1) local-effect principle. The local dose at a random
nucleus point is a compound-Poisson (shot-noise) sum over the infinite track
field, so ``f_D(d)`` follows from the exact characteristic function and
``N_ion(D) = N_sv <rho(d)>_{f_D}``.

These tests pin the numerics of `pyradbio.models.lem._localdose` bottom-up
(dose-area conservation, compound-Poisson moments, the photon limit) before the
assembled ``beta_I(LET)`` is compared against the paper's digitized curve in
``test_lem4_intertrack.py``.
"""

from __future__ import annotations

import math
import warnings

import numpy as np
import pytest
from scipy.optimize import brentq

from pyradbio.benchmark import friedrich2012 as f12
from pyradbio.core.cell import CellTarget
from pyradbio.core.lq import LQParameters, rbe
from pyradbio.core.photon import PhotonParameters
from pyradbio.core.radiation import Ion
from pyradbio.models.lem import _localdose, _localdose_kernel
from pyradbio.models.lem.lem4 import (
    _CLUSTER_INDEX_MAX,
    _DISCRETE_DOSE_FRACTIONS,
    _MC_IMPACT_CUT_FACTOR,
    _MC_IMPACT_CUT_MIN_DOMAIN_FRACTION,
    LEM4,
    SUBVOLUME_SIDE_NM,
    _cluster_index_bounds,
    _eta,
)
from pyradbio.track.stopping_power import SimpleBetheStoppingPower


# ---- trapezoid weights -------------------------------------------------
def test_trapezoid_weights_reproduce_trapezoid_rule() -> None:
    """``sum(w * f)`` equals ``numpy.trapezoid(f, x)`` on a non-uniform grid."""
    x = np.logspace(-3.0, 1.0, 257)
    f = 1.0 / (1.0 + x * x)
    w = _localdose.trapezoid_weights(x)
    assert float(np.sum(w * f)) == pytest.approx(float(np.trapezoid(f, x)), rel=1e-12)


# ---- g_B: box-averaged single-track dose -------------------------------
def test_box_average_of_uniform_field_is_the_same_constant() -> None:
    """Box-averaging a constant dose field returns that constant at every radius.

    Pins the exact circle-in-square angular measure: the angular weights must
    integrate to the square's area at every radius, or the profile carries a
    radius-dependent gain.
    """
    r = np.logspace(-4.0, 1.0, 4000)
    rho = np.logspace(-3.0, 0.0, 40)
    g = _localdose.box_averaged_profile(r, np.full_like(r, 3.0), rho, ell=0.54)
    assert np.allclose(g, 3.0, rtol=2e-3)


def test_box_average_conserves_the_dose_area_integral() -> None:
    """``int g_B 2 pi rho drho == int D 2 pi r dr`` for a 1/r^2 track profile.

    Box-averaging is a convolution with a unit-mass kernel, so it preserves the
    total dose-area exactly in the continuum. Any residual here is pure
    quadrature error -- the check that the sub-sampling actually resolves the
    singular 1/r^2 core (a crude near-core box quadrature over-counts it by
    ~35%, which biases every downstream ``beta``).
    """
    r_min, r_max = 1.0e-4, 50.0
    ell = 0.54

    def dose_at(r: np.ndarray) -> np.ndarray:
        r = np.asarray(r, dtype=np.float64)
        inner = np.where(r < r_min, r_min, r)
        return np.where(r > r_max, 0.0, 1.0 / (inner * inner))

    r = np.logspace(math.log10(r_min * 1.0e-3), math.log10(r_max * 1.02), 6000)
    i_track = float(np.sum(_localdose.trapezoid_weights(r) * dose_at(r) * 2.0 * math.pi * r))
    # The box average spreads dose out to r_max plus the square's circumradius.
    rho = np.logspace(math.log10(r_min * 1.0e-3), math.log10(r_max + ell), 1500)
    g = _localdose.box_averaged_profile(r, dose_at(r), rho, ell=ell)
    w = _localdose.trapezoid_weights(rho)
    area = float(np.sum(w * g * 2.0 * math.pi * rho))
    assert area == pytest.approx(i_track, rel=5e-3), (
        f"dose-area not conserved: {area / i_track:.4f}"
    )


def test_box_average_is_bounded_by_the_pointwise_core_dose() -> None:
    """The 540 nm average tames the singular core: ``g_B(0) << D(r_min)``."""
    r_min, r_max = 1.0e-4, 50.0

    def dose_at(r: np.ndarray) -> np.ndarray:
        r = np.asarray(r, dtype=np.float64)
        inner = np.where(r < r_min, r_min, r)
        return np.where(r > r_max, 0.0, 1.0 / (inner * inner))

    r = np.logspace(math.log10(r_min * 1.0e-3), math.log10(r_max), 3000)
    rho = np.array([0.0, 1.0e-3, 1.0])
    g = _localdose.box_averaged_profile(r, dose_at(r), rho, ell=0.54)
    assert g[0] < 1.0e-3 * dose_at(np.array([r_min]))[0]
    assert g[0] > g[-1]  # still peaked at the core


def test_box_average_is_finite_when_a_node_coincides_with_the_box_centre() -> None:
    """No NaN when an integration node lands exactly on ``rho``.

    The kernel's law-of-cosines term is the cancelling form of
    ``(r - rho)^2``; at ``r ~ rho`` and ``delta ~ 0`` it can round negative and
    NaN-poison the whole profile. That fired for real -- carbon at 40 keV/um on
    the pyMKM stopping power -- while neighbouring LETs were fine, so it is not
    reliably caught by a coarse LET sweep.
    """
    r = np.logspace(-4.0, 1.0, 2000)
    dose = 1.0 / np.maximum(r, 1.0e-3) ** 2
    # Include radii that coincide with integration nodes by construction.
    rho = np.concatenate([np.array([r[900], r[1500]]), np.logspace(-3.0, 0.5, 20)])
    g = _localdose.box_averaged_profile(r, dose, rho, ell=0.54)
    assert np.all(np.isfinite(g)), g
    assert np.all(g > 0.0)


def test_box_average_matches_a_direct_two_dimensional_quadrature() -> None:
    """The polar direction-averaged form agrees with brute-force 2-D sub-sampling.

    Cross-checks the closed-form kernel against the obvious (but core-blind)
    implementation: sub-sample the square on a fine Cartesian grid and average
    over the direction of the box centre, which is what a track landing at a
    uniformly random position relative to the fixed subvolume lattice produces.
    """
    r_min, r_max = 1.0e-3, 20.0

    def dose_at(x: np.ndarray) -> np.ndarray:
        x = np.asarray(x, dtype=np.float64)
        inner = np.where(x < r_min, r_min, x)
        return np.where(x > r_max, 0.0, 1.0 / (inner * inner))

    ell = 0.54
    r = np.logspace(-6.0, math.log10(r_max), 4000)
    rho = np.array([0.15, 0.3, 0.6, 1.5])
    g = _localdose.box_averaged_profile(r, dose_at(r), rho, ell=ell)

    nq = 301
    q = (np.arange(nq) - 0.5 * (nq - 1)) / nq * ell
    du, dv = np.meshgrid(q, q)
    brute = []
    for c in rho:
        angles = np.linspace(0.0, 2.0 * math.pi, 64, endpoint=False)
        brute.append(
            np.mean(
                [
                    dose_at(np.hypot(c * math.cos(p) + du, c * math.sin(p) + dv)).mean()
                    for p in angles
                ]
            )
        )
    assert np.allclose(g, np.array(brute), rtol=1e-2)


def test_box_average_engine_validation() -> None:
    """``engine`` must be one of the three known backends."""
    r = np.logspace(-4.0, 1.0, 100)
    with pytest.raises(ValueError, match="engine"):
        _localdose.box_averaged_profile(
            r,
            np.ones_like(r),
            np.array([0.1]),
            ell=0.54,
            engine="fast",  # type: ignore[arg-type]
        )


def test_box_average_numba_matches_numpy_to_machine_precision() -> None:
    """The numba kernel runs the identical quadrature, so it must agree to rounding.

    Unlike the discrete DSB-pattern kernel (which substitutes a polynomial ``i0e``
    and trapezoid for scipy's ``i0e``/Simpson and so agrees only to ~0.5%), the box
    average is the *same* nested trapezoid on the *same* nodes compiled with
    ``prange``. A machine-precision match is the contract that lets ``engine="auto"``
    default to numba with no separate tolerance and no reproducibility caveat.
    """
    if not _localdose_kernel.HAVE_NUMBA:
        pytest.skip("numba (the 'fast' extra) is not installed")
    r_min, r_max, ell = 1.0e-4, 30.0, 0.54
    r = np.logspace(math.log10(r_min * 1.0e-3), math.log10(r_max), 1500)
    dose = np.where(r > r_max, 0.0, 1.0 / np.maximum(r, r_min) ** 2)
    rho = np.logspace(math.log10(r_min * 1.0e-3), math.log10(r_max + ell), 400)
    ref = _localdose.box_averaged_profile(r, dose, rho, ell=ell, engine="numpy")
    fast = _localdose.box_averaged_profile(r, dose, rho, ell=ell, engine="numba")
    assert np.allclose(fast, ref, rtol=1e-10, atol=0.0), np.abs(fast - ref).max()


# ---- h1: dose-area histogram -------------------------------------------
def test_dose_area_histogram_preserves_weight_and_dose_area() -> None:
    """Binning jumps onto a log-dose grid preserves both the total intensity
    ``sum w`` and the mean dose ``sum w x`` (the compound-Poisson mean)."""
    rng = np.random.default_rng(0)
    x = np.sort(rng.uniform(1.0e-5, 1.0e2, 4000))
    w = rng.uniform(0.0, 1.0, 4000)
    xb, wb = _localdose.dose_area_histogram(x, w, bins_per_decade=40)
    assert float(np.sum(wb)) == pytest.approx(float(np.sum(w)), rel=1e-12)
    assert float(np.sum(wb * xb)) == pytest.approx(float(np.sum(w * x)), rel=1e-12)
    assert xb.size < x.size


def test_dose_area_histogram_drops_zero_weight_bins() -> None:
    """Empty log-dose bins are dropped, and jump sizes stay strictly positive."""
    x = np.array([1.0e-4, 1.0e-4, 5.0e1])
    w = np.array([1.0, 2.0, 0.5])
    xb, wb = _localdose.dose_area_histogram(x, w, bins_per_decade=10)
    assert xb.size == 2
    assert np.all(xb > 0.0)
    assert np.all(wb > 0.0)


# ---- compound-Poisson density ------------------------------------------
def test_compound_poisson_single_jump_matches_poisson() -> None:
    """One jump size ``x0`` with intensity ``lam`` gives a Poisson comb at ``k*x0``."""
    x0, lam = 1.0, 3.0
    solver = _localdose.CompoundPoissonSolver(
        np.array([x0]), np.array([lam]), d_hi=30.0, n_grid=1 << 14
    )
    f = solver.density(1.0)
    dd = solver.dd
    for k in range(8):
        idx = round(k * x0 / dd)
        # The resolving kernel spreads each atom over a few grid cells.
        mass = float(f[max(idx - 8, 0) : idx + 9].sum())
        expected = math.exp(-lam) * lam**k / math.factorial(k)
        assert mass == pytest.approx(expected, abs=2.0e-3), f"k={k}"


def test_compound_poisson_mean_matches_the_analytic_mean() -> None:
    """``<Z> = sum_j w_j x_j`` -- the guard against periodic wrap-around.

    A Gaussian/Gamma small-jump approximation wraps its (spurious) negative tail
    around the periodic FFT grid and inflates the mean by orders of magnitude;
    the exact characteristic function summed directly over the jump nodes does
    not. Mixing a huge, rare jump with many tiny ones is exactly the regime that
    breaks the approximations.
    """
    x = np.concatenate([np.logspace(-4.0, 0.0, 400), np.array([250.0])])
    w = np.concatenate([np.full(400, 0.05), np.array([1.0e-3])])
    solver = _localdose.CompoundPoissonSolver(x, w, d_hi=1500.0, n_grid=1 << 16)
    f = solver.density(1.0)
    d = solver.dose_grid
    mean = float(np.sum(f * d))
    assert mean == pytest.approx(float(np.sum(w * x)), rel=2.0e-3)
    assert float(f.sum()) == pytest.approx(1.0, rel=1e-9)


def test_compound_poisson_scales_intensity_without_rebuilding() -> None:
    """``density(scale)`` rescales the whole jump intensity (dose enters only here).

    ``phi = D / I_track``, so every dose point reuses the one expensive
    characteristic-function pass.
    """
    x = np.logspace(-3.0, 1.0, 200)
    w = np.full(200, 0.02)
    solver = _localdose.CompoundPoissonSolver(x, w, d_hi=400.0, n_grid=1 << 15)
    for scale in (0.5, 2.0, 7.0):
        f = solver.density(scale)
        mean = float(np.sum(f * solver.dose_grid))
        assert mean == pytest.approx(scale * float(np.sum(w * x)), rel=3.0e-3)


def test_compound_poisson_approaches_a_delta_in_the_many_track_limit() -> None:
    """Many small jumps -> ``f_D`` concentrates at the macroscopic dose (photon limit).

    This is the low-LET limit that makes ``beta_I -> beta_x``: the local dose
    everywhere equals the macroscopic dose, so the response is the photon
    response.
    """
    x = np.full(500, 2.0e-3)
    w = np.full(500, 1.0)  # mean = 1 Gy, relative std ~ 1/sqrt(500*... ) small
    solver = _localdose.CompoundPoissonSolver(x, w, d_hi=20.0, n_grid=1 << 16)
    f = solver.density(1.0)
    d = solver.dose_grid
    mean = float(np.sum(f * d))
    var = float(np.sum(f * (d - mean) ** 2))
    assert mean == pytest.approx(1.0, rel=1e-3)
    assert math.sqrt(var) / mean < 0.05


def test_wrapped_mass_flags_a_too_short_dose_grid() -> None:
    """``wrapped_mass`` is negligible for a generous grid and large for a short one.

    It is the folding assumption's own alarm: `density` moves the upper
    half of the grid to ``d = 0``, which is only right while ``d_hi`` bounds the
    support. Squeezing the grid below the distribution's mean must show up.
    """
    x = np.logspace(-3.0, 0.0, 200)
    w = np.full(200, 0.5)
    generous = _localdose.CompoundPoissonSolver(x, w, d_hi=40.0, n_grid=1 << 15)
    assert generous.wrapped_mass(1.0) < 1.0e-6
    squeezed = _localdose.CompoundPoissonSolver(x, w, d_hi=1.2, n_grid=1 << 15)
    assert squeezed.wrapped_mass(20.0) > 1.0e-2


def test_compound_poisson_rejects_jumps_beyond_the_grid() -> None:
    """A jump larger than ``d_hi`` would alias around the periodic grid."""
    with pytest.raises(ValueError, match="d_hi"):
        _localdose.CompoundPoissonSolver(np.array([5.0]), np.array([1.0]), d_hi=4.0, n_grid=1 << 10)


# ---- low-dose LQ fit ---------------------------------------------------
def test_lq_coefficients_recover_a_planted_quadratic() -> None:
    """``N(D) = a D + b D^2`` is recovered exactly from ``N/D = a + b D``."""
    doses = np.linspace(0.5, 3.0, 7)
    effect = 0.3 * doses + 0.04 * doses * doses
    alpha, beta = _localdose.lq_coefficients(doses, effect)
    assert alpha == pytest.approx(0.3, rel=1e-10)
    assert beta == pytest.approx(0.04, rel=1e-10)


def test_lq_coefficients_do_not_clamp_a_negative_beta() -> None:
    """A concave response returns a negative ``beta`` rather than 0.

    Clamping biases any subsequent seed/LET averaging upward, and the negative
    value is diagnostic: it says the fit window has run past the single-track
    overkill saturation, where the LQ form no longer describes the response.
    """
    doses = np.linspace(0.5, 3.0, 7)
    _alpha, beta = _localdose.lq_coefficients(doses, 0.3 * doses - 0.02 * doses * doses)
    assert beta == pytest.approx(-0.02, rel=1e-10)


# ---- assembled model: LEM4 local-dose beta_I ---------------------------
FIG4 = PhotonParameters(alpha_x=0.25, beta_x=0.05, D_t=9.5)
FIG4_CELL = CellTarget(nucleus_radius_um=5.0)
_SP = SimpleBetheStoppingPower()


def _carbon_for_let(let_keV_um: float) -> Ion:
    """Carbon-12 whose water LET equals ``let_keV_um``."""
    energy = float(brentq(lambda e: _SP.let_keV_um(Ion(6, 12, e)) - let_keV_um, 1.6, 600.0))
    return Ion(6, 12, energy)


def test_every_cluster_index_path_honours_the_overkill_cap(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """All three local-dose/discrete paths must clip ``C_I`` at ``_CLUSTER_INDEX_MAX``.

    That constant is the *derived* float64 invertibility guard (see its comment),
    not the TRiP98-tuned 0.95, so this test drives it explicitly to two values
    rather than comparing against the default. Every path must read the shared
    bound; clipping to the ``C_g -> g`` lookup grid's own extent instead is *not*
    equivalent:
    ``C_g`` saturates to exactly 1.0 in float64 near 3175 Gy (``P_iso = m e^-m``
    underflows), so the grid top is 1.0 and a saturated ``C_I`` lands at a ``D_eq``
    ceiling >= 3175 Gy against the
    analytic path's 453 Gy, giving the local-dose family a deep-overkill fitted
    ``alpha`` 2.4x the analytic one -- same model, two regularisations.

    Raising the constant must move every path. A path that clips to ``c_g[-1]``
    ignores it and fails here.
    """
    lem = LEM4(photon=FIG4, cell=FIG4_CELL, stopping_power=_SP)
    c_g, _g = lem._intertrack_pe_lookup()

    # The trap itself: the grid's top entry is exactly 1.0, so it is no bound at all.
    assert c_g[-1] == 1.0
    assert _cluster_index_bounds(c_g) == (float(c_g[0]), _CLUSTER_INDEX_MAX)
    assert float(c_g[0]) < _CLUSTER_INDEX_MAX < c_g[-1]

    # Each path forms C_I at its own scale, so each needs its own saturating regime:
    # the convolution averages over the *whole nucleus* and only crosses 0.95 above
    # ~300 Gy; the discrete engine's per-traversal-number C_I crosses it above
    # LET ~700; the MC scorer sees one realisation's subvolume doses directly.
    ell = SUBVOLUME_SIDE_NM * 1.0e-3
    conv_ion, conv_doses = Ion(6, 12, 2.0), np.array([1000.0])
    disc_ion, disc_doses = Ion(6, 12, 0.4), np.array([2.0])
    saturating_local = np.full(4, 500.0)

    def sample() -> dict[str, float]:
        score = lem._local_dose_pattern_score(
            "cluster_index", n_sv=4, r_eff=lem.cell.effective_nucleus_radius_um(), ell=ell
        )
        return {
            "convolution": float(lem.local_dose_lethal_events(conv_ion, conv_doses)[-1]),
            "discrete": float(lem.discrete_dsb_lethal_events(disc_ion, disc_doses)[-1]),
            "mc scorer": float(score(saturating_local)),
        }

    monkeypatch.setattr("pyradbio.models.lem.lem4._CLUSTER_INDEX_MAX", 0.95)
    base = sample()
    monkeypatch.setattr("pyradbio.models.lem.lem4._CLUSTER_INDEX_MAX", 0.999)
    raised = sample()

    # A site clipping at ``c_g[-1]`` is unmoved by the constant -- 1.0 bounds both
    # 0.95 and 0.999 -- so strict inequality is the discriminator.
    stuck = {k: (base[k], raised[k]) for k in base if raised[k] <= base[k] * (1.0 + 1e-6)}
    assert not stuck, f"paths ignoring _CLUSTER_INDEX_MAX: {stuck}"


def test_lem4_local_dose_kernel_validation() -> None:
    """``local_dose_kernel`` must be known; the default is the cluster index."""
    with pytest.raises(ValueError, match="local_dose_kernel"):
        LEM4(photon=FIG4, cell=FIG4_CELL, local_dose_kernel="deq")  # type: ignore[arg-type]
    assert LEM4(photon=FIG4, cell=FIG4_CELL).local_dose_kernel == "cluster_index"


def test_lem4_beta_model_local_dose_routes_and_leaves_alpha_alone(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """``beta_model="local_dose"`` supplies only ``beta``; ``alpha_I`` is untouched.

    Friedrich et al. 2012 report ``alpha_I`` as identical between the full
    simulation and the approximation, so the beta model must never move it.
    """
    ion = _carbon_for_let(120.0)
    monkeypatch.setattr(LEM4, "local_dose_beta", lambda self, i: 0.0321)
    lem_ld = LEM4(photon=FIG4, cell=FIG4_CELL, beta_model="local_dose")
    lem_sm = LEM4(photon=FIG4, cell=FIG4_CELL, beta_model="slope_matching")
    lq = lem_ld.alpha_beta(ion)
    assert lq.beta == pytest.approx(0.0321)
    assert lq.alpha == pytest.approx(lem_sm.alpha_beta(ion).alpha)


def test_local_dose_mean_local_dose_equals_the_macroscopic_dose() -> None:
    """``<d> = D``: the jump histogram's dose-area equals ``I_track``.

    The compound-Poisson mean is ``phi sum(w x) = (D / I_track) I_track``, so this
    is the end-to-end statement that box-averaging conserved the dose-area. If it
    drifts, every ``beta`` is scaled by the same factor.
    """
    lem = LEM4(photon=FIG4, cell=FIG4_CELL)
    sizes, weights, i_track = lem._local_dose_jumps(_carbon_for_let(55.0))
    assert float(np.sum(weights * sizes)) == pytest.approx(i_track, rel=1e-12)
    # ... and I_track from the averaged profile matches the radial dose-area.
    r, dose, _r_min, _r_max = lem._local_dose_profile(_carbon_for_let(55.0))
    radial = float(np.sum(_localdose.trapezoid_weights(r) * dose * 2.0 * math.pi * r))
    assert i_track == pytest.approx(radial, rel=2e-3)


@pytest.mark.slow
def test_local_dose_recovers_the_photon_response_at_low_let() -> None:
    """At low LET the local dose is uniform, so ``N_ion(D) -> alpha_x D + beta_x D^2``.

    The photon limit is *derived* here rather than calibrated: dense tracks make
    the compound-Poisson jumps many and small, so ``f_D -> delta(d - D)``.
    """
    lem = LEM4(photon=FIG4, cell=FIG4_CELL)
    doses = np.array([0.5, 1.0, 2.0, 3.0])
    n_ion = lem.local_dose_lethal_events(_carbon_for_let(10.0), doses)
    expected = FIG4.alpha_x * doses + FIG4.beta_x * doses * doses
    assert np.allclose(n_ion, expected, rtol=0.12)
    assert lem.local_dose_beta(_carbon_for_let(10.0)) == pytest.approx(FIG4.beta_x, rel=0.05)


@pytest.mark.slow
def test_local_dose_beta_rises_above_beta_x_then_collapses() -> None:
    """Both Fig. 4 features: a rise above ``beta_x``, then a collapse toward 0.

    The paper's full-simulation ``beta_I`` starts at ``beta_x``, rises a few
    percent above it at intermediate LET (intertrack clustering adding curvature
    beyond the photon response), and then falls to ~0 as the ``D_t`` saturation
    takes over. No previous engine here produced both: slope-matching only falls
    (too early), the discrete-DSB intertrack Monte-Carlo only stays at ``beta_x``,
    and the per-subvolume ``"photon_effect"`` kernel cannot exceed ``beta_x`` at
    all (see the theorem test below).
    """
    lem = LEM4(photon=FIG4, cell=FIG4_CELL)
    b10, b25, b70, b130 = (
        lem.local_dose_beta(_carbon_for_let(let)) for let in (10.0, 25.0, 70.0, 130.0)
    )
    assert b10 == pytest.approx(FIG4.beta_x, rel=0.05), b10
    assert b25 > FIG4.beta_x, f"no rise above beta_x at LET 25: {b25}"
    assert b70 < 0.5 * b25, (b25, b70)
    assert b130 < 0.02 * FIG4.beta_x, f"did not collapse toward 0: {b130}"


@pytest.mark.slow
def test_local_dose_overkill_keeps_beta_x_and_rolls_off_faster() -> None:
    """The single-track overkill (B2) recovers ``beta_x`` at low LET and improves
    the mid-LET roll-off.

    The plain convolution scores the mean effect ``exp(-<E>)``; ``local_dose_overkill``
    keeps the Poisson track-number fluctuation, applying ``<exp(-E)>`` on the effect
    curve. It must (a) still recover ``beta_x`` at the photon limit and (b) sit
    *below* the mean-effect curve at mid LET (moving toward the paper's faster
    roll-off). Deep overkill is covered by the clamp test below.
    """
    mean = LEM4(photon=FIG4, cell=FIG4_CELL)
    ovk = LEM4(photon=FIG4, cell=FIG4_CELL, local_dose_overkill=True)
    assert ovk.local_dose_beta(_carbon_for_let(10.0)) == pytest.approx(FIG4.beta_x, rel=0.05)
    # mid-LET: overkill rolls off faster (below the mean-effect value)
    assert ovk.local_dose_beta(_carbon_for_let(55.0)) < mean.local_dose_beta(_carbon_for_let(55.0))


@pytest.mark.parametrize("overkill", [False, True])
def test_local_dose_beta_clamps_deep_overkill_to_zero_and_says_so(overkill: bool) -> None:
    """Both paths clamp a negative deep-overkill fit to 0, and warn.

    Past ~150 keV/um the simulated dose response is saturated, so the low-dose LQ
    fit has no curvature left and least squares returns a marginally negative
    ``beta``. That is a fit artifact, not a physical negative curvature.

    The clamp is not specific to ``local_dose_overkill``: the plain mean-effect path
    goes negative too, and an unclamped value would escape into
    ``LQParameters``, which rejects ``beta < 0`` -- so ``LEM4(beta_model=
    "local_dose").alpha_beta()`` would raise for carbon above ~150 keV/um on the
    paper's own V79 cell. A floor is not a value, hence the warning.
    """
    lem = LEM4(photon=FIG4, cell=FIG4_CELL, local_dose_overkill=overkill)
    with pytest.warns(UserWarning, match="clamped to 0"):
        assert lem.local_dose_beta(_carbon_for_let(200.0)) == 0.0


def test_local_dose_alpha_beta_survives_deep_overkill() -> None:
    """The regression itself: the public entry point must not raise past ~150 keV/um."""
    lem = LEM4(photon=FIG4, cell=FIG4_CELL, beta_model="local_dose")
    with pytest.warns(UserWarning, match="clamped to 0"):
        lq = lem.alpha_beta(_carbon_for_let(200.0))
    assert lq.beta == 0.0
    assert lq.alpha > 0.0


def test_local_dose_overkill_routes_through_alpha_beta(monkeypatch: pytest.MonkeyPatch) -> None:
    """``local_dose_overkill`` supplies the clamped ``beta``; ``alpha_I`` stays the
    single-particle ``initial_slope`` (Friedrich 2012 fixes ``alpha_I``)."""
    ion = _carbon_for_let(150.0)
    monkeypatch.setattr(
        LEM4, "local_dose_beta", lambda self, i: 0.0 if self.local_dose_overkill else 0.9
    )
    lem = LEM4(photon=FIG4, cell=FIG4_CELL, beta_model="local_dose", local_dose_overkill=True)
    ref = LEM4(photon=FIG4, cell=FIG4_CELL, beta_model="slope_matching")
    lq = lem.alpha_beta(ion)
    assert lq.beta == 0.0
    assert lq.alpha == pytest.approx(ref.alpha_beta(ion).alpha)


@pytest.mark.slow
def test_local_dose_photon_effect_kernel_cannot_exceed_beta_x() -> None:
    """A response that is a function of the *physical* local dose has ``beta <= beta_x``.

    Below ``D_t`` the response is ``alpha d + beta d^2``, so ``<rho> = alpha D +
    beta (D^2 + Var)``; tracks are an independent Poisson field, so ``Var`` is
    strictly linear in ``D``. The variance therefore lands entirely in ``alpha``
    and the ``D^2`` coefficient is pinned at ``beta_x`` -- exactly ``beta_x`` when
    nothing saturates, less once ``D_t`` bites. That is why reproducing the
    paper's rise needs the ``"cluster_index"`` kernel, whose DSB channel sees the
    ``eta``-enhanced dose rather than the physical one.
    """
    unreachable_dt = PhotonParameters(alpha_x=0.25, beta_x=0.05, D_t=1.0e6)
    lem = LEM4(photon=unreachable_dt, cell=FIG4_CELL, local_dose_kernel="photon_effect")
    for let in (10.0, 55.0, 100.0):
        beta = lem.local_dose_beta(_carbon_for_let(let))
        assert beta == pytest.approx(FIG4.beta_x, rel=1e-3), (let, beta)
    saturating = LEM4(photon=FIG4, cell=FIG4_CELL, local_dose_kernel="photon_effect")
    assert saturating.local_dose_beta(_carbon_for_let(25.0)) < FIG4.beta_x


@pytest.mark.slow
def test_local_dose_globle_kernel_stays_at_beta_x_at_every_let() -> None:
    """The GLOBLE kernel *cannot* reproduce the Fig. 4 fall -- pinned deliberately.

    GLOBLE has no threshold dose ``D_t``, and the subvolume-averaged single-track
    dose peaks at only ~35 Gy even at 100 keV/um, so its Poisson mean stays
    ``m <= 0.33`` and ``rho_B ~ eps_i m + eps_c m^2 / 2`` throughout. The ``D^2``
    coefficient is then algebraically ``beta_x``, independent of LET.

    Sampling the *pointwise* ``1/r^2`` dose at subvolume centres over-reads the
    core by orders of magnitude and makes this kernel appear to collapse at high
    LET; with the dose-area-conserving subvolume average the flatness is
    unambiguous. This test exists so that artifact cannot quietly return.
    """
    lem = LEM4(photon=FIG4, cell=FIG4_CELL, local_dose_kernel="globle")
    betas = [lem.local_dose_beta(_carbon_for_let(let)) for let in (10.0, 55.0, 100.0)]
    assert all(b == pytest.approx(FIG4.beta_x, rel=0.35) for b in betas), betas
    # The photon-effect kernel, on the same tracks, does fall.
    photon_kernel = LEM4(photon=FIG4, cell=FIG4_CELL, local_dose_kernel="photon_effect")
    assert photon_kernel.local_dose_beta(_carbon_for_let(100.0)) < 0.6 * betas[-1]


@pytest.mark.slow
def test_local_dose_monte_carlo_reproduces_the_convolution() -> None:
    """Explicit-track MC agrees with the analytic convolution of the same physics.

    The MC places tracks one at a time and accumulates subvolume dose; the
    convolution integrates the identical Poisson statistics analytically. They
    must agree on the mean effect (the MC is additionally truncated at
    ``R_eff + r_max`` and carries realisation noise, hence the loose tolerance).
    """
    lem = LEM4(photon=FIG4, cell=FIG4_CELL, local_dose_kernel="photon_effect")
    ion = _carbon_for_let(70.0)
    mc = lem.local_dose_mc_beta(ion, n_realizations=600, seed=3)
    assert mc == pytest.approx(lem.local_dose_beta(ion), rel=0.25), mc


@pytest.mark.slow
def test_overkill_correction_is_small_on_the_photon_effect_kernel_only() -> None:
    """``<exp(-E)>`` vs ``exp(-<E>)`` is a modest effect **on this kernel**.

    The measurement does not generalise: it constructs
    ``local_dose_kernel="photon_effect"``, so it says nothing about the production
    ``"cluster_index"`` kernel, where the reduction is 6/12/20/29/47% at LET
    25/40/55/70/100; see docs/references/LEM_IV_Friedrich2012.md.

    The assertion is a measurement of this kernel only, and
    the structural reason applies to it: for a Poisson track process every
    cumulant of the cell effect scales linearly in dose, so on a response that is
    a pointwise function of the local dose the correction is almost entirely
    linear and barely touches the quadratic coefficient.
    """
    lem = LEM4(photon=FIG4, cell=FIG4_CELL, local_dose_kernel="photon_effect")
    ion = _carbon_for_let(85.0)
    plain = lem.local_dose_mc_beta(ion, n_realizations=600, seed=5)
    overkill = lem.local_dose_mc_beta(ion, n_realizations=600, seed=5, overkill=True)
    assert 0.6 * plain < overkill < plain, (plain, overkill)


@pytest.mark.slow
def test_b2_overcorrects_against_the_exact_stochastic_survival() -> None:
    """The analytic B2 correction drops ``beta`` by far more than the exact one does.

    B2 (`LEM4.local_dose_overkill`) approximates ``<exp(-E)>`` by evaluating
    the *mean-effect curve* at ``n D_1``; the explicit-track MC computes it from
    real ``n``-track configurations. Both drops are measured here as
    ``beta(mean effect) - beta(overkill)`` so any common bias of the truncated MC
    domain cancels -- which is what makes this comparison survive the MC's own
    1-2 sigma offsets from the convolution.

    Measured with 12 seeds x 400 realisations, cluster_index kernel:

    | LET | B2 drop | exact drop | ratio |
    |-----|---------|------------|-------|
    | 55  | 0.00769 | 0.00411 +- 0.00048 | 1.87 |
    | 70  | 0.00522 | 0.00105 +- 0.00072 | 4.95 |
    | 100 | 0.00235 | -0.00260 +- 0.00162 | -0.90 (opposite sign) |

    **So B2's best-in-class RMS against Fig. 4 is compensation, not correctness.**
    Corroborated independently: applying the exact drop gives ``beta`` 0.0346 /
    0.0172 at LET 55/70, within 2-5% of the discrete engine's stochastic path
    (0.0364 / 0.0176) -- two structurally different engines agreeing under the
    exact convention, with B2 (0.0311 / 0.0130) the outlier.

    Uses 6 seeds here to stay affordable; the assertion is the sign of the
    inequality, which sits ~5 sigma clear at that count.
    """
    lem = LEM4(photon=FIG4, cell=FIG4_CELL)  # cluster_index kernel
    b2 = LEM4(photon=FIG4, cell=FIG4_CELL, local_dose_overkill=True)
    ion = _carbon_for_let(55.0)
    drop_b2 = lem.local_dose_beta(ion) - b2.local_dose_beta(ion)

    pairs = np.array(
        [lem.local_dose_mc_beta_pair(ion, n_realizations=400, seed=s) for s in range(6)]
    )
    drops = pairs[:, 0] - pairs[:, 1]
    exact = float(drops.mean())
    assert exact > 0.0, drops  # the exact correction does lower beta at this LET
    assert exact < drop_b2, (exact, drop_b2)  # ...but by much less than B2 claims


@pytest.mark.slow
def test_cluster_index_monte_carlo_reproduces_the_convolutions_effect_curve() -> None:
    """The explicit-track MC runs the **production** kernel, and agrees with it.

    The cluster index has no per-subvolume decomposition -- it forms
    ``C_I = N_cDSB/N_dmg`` over the whole nucleus first -- so ``local_dose_mc_beta``
    scores the realisation's whole pattern for ``local_dose_kernel="cluster_index"``.
    This pins that the two routes compute the same model.

    Asserted on the **effect curve**, not on ``beta``: ``beta`` is a small
    curvature on a dominant linear term and at LET 100 three seeds of 400
    realisations give 0.0049 / 0.0071 / 0.0371 -- a noise trap. ``E(D)`` is far better conditioned,
    and it is the
    quantity the scorer is actually responsible for.
    """
    lem = LEM4(photon=FIG4, cell=FIG4_CELL)  # cluster_index is the default kernel
    ion = _carbon_for_let(100.0)
    doses = lem.fit_doses_Gy()
    exact = lem.local_dose_lethal_events(ion, doses)

    r, dose, _r_min, r_max = lem._local_dose_profile(ion)
    ell = SUBVOLUME_SIDE_NM * 1.0e-3
    rho = np.logspace(math.log10(float(r[0])), math.log10(r_max + ell), 1200)
    g = _localdose.box_averaged_profile(r, _eta(dose) * dose, rho, ell=ell, engine=lem.engine)
    log_rho, log_g, top = np.log(rho), np.log(np.maximum(g, 1e-300)), float(rho[-1])

    def jump_at(dist: np.ndarray) -> np.ndarray:
        out = np.exp(np.interp(np.log(np.clip(dist, rho[0], top)), log_rho, log_g))
        out = np.where(dist < rho[0], g[0], out)
        return np.asarray(np.where(dist > top, 0.0, out), dtype=np.float64)

    r_eff = lem.cell.effective_nucleus_radius_um()
    n_sv = _localdose.subvolume_centres(r_eff, ell)[0].size
    r_sim = r_eff + r_max
    effect, _survival = _localdose.monte_carlo_effect(
        jump_at,
        lem._local_dose_pattern_score("cluster_index", n_sv=n_sv, r_eff=r_eff, ell=ell),
        doses=doses,
        tracks_per_Gy=(r_sim / r_eff) ** 2 / lem.mean_traversal_dose_Gy(ion),
        r_eff=r_eff,
        r_sim=r_sim,
        ell=ell,
        n_real=400,
        rng=np.random.default_rng(0),
    )
    assert np.allclose(effect, exact, rtol=0.05), (effect, exact)


@pytest.mark.slow
@pytest.mark.golden
def test_golden_lem4_local_dose_reproduces_friedrich2012_fig4_full_simulation() -> None:
    """The local-dose convolution reproduces Friedrich et al. 2012's own digitized
    Fig. 4 *full-simulation* ``beta_I(LET)`` through the low- and mid-LET range.

    Reference: Friedrich, Scholz, Elsasser, Durante & Scholz 2012, Int. J. Radiat.
    Biol. 88(1-2):103-107, doi:10.3109/09553002.2011.611213, Fig. 4 (p. 107),
    hypothetical cell (alpha_g=0.25, beta_g=0.05, D_t=9.5 Gy, R_n=5 um) --
    digitized in `pyradbio.benchmark.friedrich2012`.

    The default ``"cluster_index"`` kernel reproduces **both** features of the
    curve with no free parameter -- the small rise above ``beta_x`` at
    intermediate LET and the collapse toward 0 beyond it -- and its fitted
    ``alpha`` tracks the paper's own full-simulation ``alpha_I`` over the same
    range, which is the independent cross-check that the effect functional is
    normalised correctly (a ``beta`` match with a wrong ``alpha`` would not mean
    much).

    The residual is that **the intertrack enhancement never rolls off**, which is a
    structural limit of this formulation, not a tuning gap -- hence the RMS
    tolerance below is loose enough to admit the LET 30-50 band, with the shape
    asserted separately. The enhancement is the dose-slope of the cluster index, and
    ``<h> = D`` by dose conservation pins that slope at the photon rate ``kappa/2``
    for every LET, so no choice of track population or clustering domain can change
    it. Reproducing the roll-off needs a discrete damage pattern. Documented in
    docs/references/LEM_IV_Friedrich2012.md; not tolerance-fitted away.
    """
    lem = LEM4(photon=FIG4, cell=FIG4_CELL)
    beta_curve = f12.get_curve("fig4_hypothetical_vs_let", series="full_simulation", variant="beta")
    alpha_curve = f12.get_curve(
        "fig4_hypothetical_vs_let", series="full_simulation", variant="alpha"
    )
    lets = (10.0, 13.0, 16.0, 20.0, 25.0, 30.0, 55.0, 70.0, 85.0, 100.0, 130.0)
    doses = lem.fit_doses_Gy()

    betas, alphas = [], []
    for let in lets:
        events = lem.local_dose_lethal_events(_carbon_for_let(let), doses)
        a, b = _localdose.lq_coefficients(doses, events)
        alphas.append(a)
        betas.append(b)

    rms_beta = float(
        np.sqrt(
            np.mean(
                [
                    (b - np.interp(L, beta_curve.x, beta_curve.y)) ** 2
                    for L, b in zip(lets, betas, strict=True)
                ]
            )
        )
    )
    rms_alpha = float(
        np.sqrt(
            np.mean(
                [
                    (a - np.interp(L, alpha_curve.x, alpha_curve.y)) ** 2
                    for L, a in zip(lets, alphas, strict=True)
                ]
            )
        )
    )
    assert rms_beta < 0.010, f"beta_I RMS vs Fig. 4 full simulation = {rms_beta:.4f}"
    # alpha spans ~0.27-1.56 over this range, so this is ~8% RMS. The per-subvolume
    # photon-effect kernel scores 0.34 here -- alpha is what discriminates them.
    assert rms_alpha < 0.15, f"alpha_I RMS vs Fig. 4 full simulation = {rms_alpha:.4f}"

    # Shape: photon limit, a genuine rise above beta_x, then collapse to ~0.
    assert betas[0] == pytest.approx(FIG4.beta_x, rel=0.05), betas[0]
    assert max(betas) > FIG4.beta_x, betas
    assert betas[-1] < 0.02 * FIG4.beta_x, betas[-1]
    assert alphas[0] == pytest.approx(FIG4.alpha_x, rel=0.2), alphas[0]


# ---- Fig. 3 as a SECOND, independent constraint on beta_I ---------------
V79_FIG3 = PhotonParameters(alpha_x=0.184, beta_x=0.02, D_t=17.0)
V79_FIG3_CELL = CellTarget(nucleus_radius_um=4.7)


@pytest.mark.golden
@pytest.mark.slow
def test_golden_fig3_rbe10_matches_the_papers_own_single_particle_approximation() -> None:
    """Fig. 3's model line is the paper's *single-particle approximation*, and
    ``slope_matching`` reproduces it while ``local_dose`` (correctly) does not.

    Friedrich 2012 states it outright -- Results, p. 106: "This comparison is
    based on the single particle approximation as described above"; Discussion:
    the Fig. 3 RBE_10 underestimation "is probably due to the underestimation of
    the b-term in the single particle approximation, on which the predictions
    were based". So the published Fig. 3 line carries the Kraemer-Scholz
    ``beta``, and reproducing the published computation requires
    ``slope_matching`` -- its 1.7% relRMS here measures ``alpha_I`` and geometry
    fidelity (the ``beta`` formula is shared, i.e. agreement on ``beta`` is by
    construction). Inverting the digitized line for its implied ``beta``
    confirms this: it tracks ``beta_sm`` (LET 45: 0.0105 +- 0.0020
    vs sm 0.0139, ld 0.0234) in the well-conditioned band.

    ``local_dose`` sits +7-12% above the line over LET 27-72 -- which is the
    direction and band the paper *says* its full simulation deviates from the
    approximation, so this excess is expected of any full-simulation ``beta``
    and is NOT evidence for (or against) the Fig. 4 ``beta`` residual. It is not an
    "independent confirmation" of that residual -- Fig. 3 has no information about
    the full simulation's ``beta``, and the Fig. 4 excess rests on Fig. 4 alone.
    The assertions below are measurements against the published line.

    Everything asserted here is against the paper's own **model line** -- the
    target is the published computation, not the plotted measurements.
    """
    curve = f12.get_curve("fig3_v79_rbe_vs_let", series="calc", variant="rbe_10")
    px = np.asarray(curve.x, dtype=float)
    order = np.argsort(px)
    px, py = px[order], np.asarray(curve.y, dtype=float)[order]
    # The energy sweep below does not reach the panel's extreme overkill tail;
    # compare only where the model curve actually covers the paper's x, never by
    # extrapolating (``np.interp`` would silently clamp and manufacture a residual).
    energies = np.geomspace(3.0, 400.0, 60)
    biases: dict[str, np.ndarray] = {}
    covered = np.ones_like(px, dtype=bool)
    for beta_model in ("slope_matching", "local_dose"):
        lem = LEM4(photon=V79_FIG3, cell=V79_FIG3_CELL, beta_model=beta_model)  # type: ignore[arg-type]
        pts = []
        for energy in energies:
            ion = Ion(6, 12, float(energy))
            with warnings.catch_warnings():  # deep overkill clamps beta to 0 and says so
                warnings.simplefilter("ignore", UserWarning)
                lq = lem.alpha_beta(ion)
            pts.append(
                (
                    lem.stopping_power.let_keV_um(ion),
                    rbe(0.10, lq.alpha, lq.beta, reference=LQParameters(0.184, 0.02)),
                )
            )
        pts.sort()
        let = np.array([p[0] for p in pts])
        val = np.array([p[1] for p in pts])
        mine = np.interp(np.log(px), np.log(let), val, left=np.nan, right=np.nan)
        covered &= np.isfinite(mine)
        biases[beta_model] = mine / py - 1.0

    assert covered.sum() >= 25, covered.sum()
    px, py = px[covered], py[covered]
    biases = {k: v[covered] for k, v in biases.items()}

    rms = {k: float(np.sqrt(np.mean(v**2))) for k, v in biases.items()}
    assert rms["slope_matching"] < 0.025, rms
    assert rms["local_dose"] > 0.045, rms
    assert rms["local_dose"] > 2.5 * rms["slope_matching"], rms

    # The excess is localised to the band Fig. 4 flags, not spread over the panel.
    band = (px >= 27.0) & (px <= 72.0)
    assert band.sum() >= 5
    assert float(np.mean(biases["local_dose"][band])) > 0.07
    assert float(np.mean(np.abs(biases["slope_matching"][band]))) < 0.04
    # ... and it is gone again past the RBE maximum, where beta no longer matters.
    high = px >= 110.0
    assert float(np.mean(np.abs(biases["local_dose"][high]))) < 0.03


# ---- what the beta_I excess IS, and is not -----------------------------
def _fig4_curve(variant: str, series: str = "full_simulation") -> tuple[np.ndarray, np.ndarray]:
    c = f12.get_curve("fig4_hypothetical_vs_let", series=series, variant=variant)
    return np.asarray(c.x, dtype=float), np.asarray(c.y, dtype=float)


@pytest.mark.golden
@pytest.mark.slow
def test_the_beta_excess_is_not_an_alpha_conditioning_artifact() -> None:
    """Pinning ``alpha`` to the paper's own trace makes ``beta`` worse, not better.

    ``beta`` lives on ``s_max - alpha``, so a small ``alpha`` error would produce a
    relative ``beta`` error ``(dAlpha/s_max)/(1 - alpha/s_max)`` that diverges as
    ``alpha -> s_max``. That is threshold-then-monotone-growth -- exactly the shape
    of the excess measured under an absolute 0.4-3.0 Gy window -- which suggests "the
    excess is really an ``alpha`` defect, amplified". It is wrong, and this test
    pins the refutation so the explanation cannot come back.

    On the ``0.05-0.85 D_t`` fit window the excess has largely gone:
    ``b_free/b_pap`` at LET 55/60 is 1.02/1.07 (1.44/1.45 on the absolute window),
    so there is no LET 55-60 counterexample ("alpha right, beta >40% high").

    The refutation is the whole test. Pinning ``alpha`` drives ``beta`` from a 0.82-2.41 spread
    against the paper,
    where leaving it free holds 1.00-1.31 -- so correcting ``alpha`` does not
    correct ``beta``, it wrecks it. LET 40 is the sharpest single point: ``alpha``
    is 8.0% low while ``beta`` is only 3.8% high, and forcing ``alpha`` onto the
    paper's value throws ``beta`` 18% *low*. An amplified-``alpha`` story cannot
    produce that sign.

    Caveat that must travel with any "beta agrees" reading: the window is a
    FREE parameter (neither Friedrich 2012 nor TRiP98 states one) and it was chosen
    by scoring against this very curve -- see LEM_IV_OPEN_CHOICES.md entry 1.2. The
    improved agreement is therefore partly by construction, not independent
    validation. See docs/references/LEM_IV_Friedrich2012.md.
    """
    ax, ay = _fig4_curve("alpha")
    bx, by = _fig4_curve("beta")
    lem = LEM4(photon=FIG4, cell=FIG4_CELL)
    doses = lem.fit_doses_Gy()

    free_ratio, pinned_ratio = [], []
    for let in (20.0, 30.0, 40.0, 50.0, 55.0, 60.0, 70.0):
        ion = _carbon_for_let(let)
        effect = lem.local_dose_lethal_events(ion, doses)
        a_fit, b_free = _localdose.lq_coefficients(doses, effect)
        a_pap = float(np.interp(let, ax, ay, left=np.nan, right=np.nan))
        b_pap = float(np.interp(let, bx, by, left=np.nan, right=np.nan))
        assert math.isfinite(a_pap) and math.isfinite(b_pap), let
        # One-parameter least squares of (E/D - alpha_paper) = beta * D.
        y = effect / doses
        b_pin = float(np.sum(doses * (y - a_pap)) / np.sum(doses * doses))
        free_ratio.append(b_free / b_pap)
        pinned_ratio.append(b_pin / b_pap)

        if let == 40.0:
            # THE COUNTEREXAMPLE, in the sign of the response rather than its size.
            # alpha is 8% LOW and beta only 3.8% high; if beta's residual were an
            # amplified alpha error, restoring alpha would move beta toward the
            # paper. It does the opposite, and overshoots past it: 1.04 -> 0.82.
            assert a_fit / a_pap < 0.95, (let, a_fit, a_pap)
            assert 0.95 < b_free / b_pap < 1.10, (let, b_free, b_pap)
            assert b_pin / b_pap < 0.90, (let, b_pin, b_pap)

    def rms_log(ratios: list[float]) -> float:
        return float(np.exp(np.sqrt(np.mean(np.log(np.abs(ratios)) ** 2))))

    # Pinning alpha does not rescue beta -- it degrades it, by the alpha-beta
    # anticorrelation of the fit window absorbing the residual. Measured 1.44
    # pinned against 1.11 free over this LET set (absolute window: 1.93 vs 1.39).
    assert rms_log(pinned_ratio) > 1.2 * rms_log(free_ratio), (
        rms_log(free_ratio),
        rms_log(pinned_ratio),
    )


@pytest.mark.slow
def test_the_beta_excess_magnitude_depends_on_the_fit_window() -> None:
    """The mid-LET ``beta_I`` excess over Fig. 4 is a fit-window quantity; the
    late roll-off is not.

    Refitting the *same* simulated dose response over three fixed windows:
    below LET ~35 the fitted ``beta`` is window-independent (the response is
    LQ there), but inside the excess band it is not -- at LET 50 the
    ratio-to-paper spans ~1.03 to ~1.78. The paper's own fit protocol is
    unstated, so the *magnitude* of the excess must always be quoted with its
    window; what is window-robust is the late roll-off (``beta`` still above the
    paper's at LET 50 under every window). The fitted ``alpha`` moves <~3% across
    windows, so this is genuine non-LQ curvature projection, not an alpha tradeoff.

    The ratios are 1.78 for the 0.25-1.5 Gy window and 1.10 for 1-6 Gy; the
    *default* window is 0.05-0.85 ``D_t``, i.e. 0.475-8.075 Gy, whose ceiling is the
    **highest** of the three, with ratio 1.03. The claim is not about the three
    window names but the mechanism ``ratio falls as the window ceiling rises``,
    asserted directly, which is default-proof: ordering by ceiling gives
    1.5 Gy -> 1.78, 6 Gy -> 1.10, 8.075 Gy -> 1.03.
    """
    windows = {
        "low": np.array([0.25, 0.5, 0.75, 1.0, 1.25, 1.5]),
        "default": LEM4(photon=FIG4, cell=FIG4_CELL).fit_doses_Gy(),
        "high": np.array([1.0, 2.0, 3.0, 4.0, 5.0, 6.0]),
    }
    all_doses = np.array(sorted({float(d) for w in windows.values() for d in w}))
    lem = LEM4(photon=FIG4, cell=FIG4_CELL)
    bx, by = _fig4_curve("beta")

    def fits(let: float) -> dict[str, float]:
        events = lem.local_dose_lethal_events(_carbon_for_let(let), all_doses)
        out = {}
        for name, w in windows.items():
            idx = np.isin(all_doses, w)
            out[name] = _localdose.lq_coefficients(all_doses[idx], events[idx])[1]
        return out

    # LQ region: all three windows agree to ~1%.
    low_let = fits(25.0)
    assert max(low_let.values()) / min(low_let.values()) < 1.02, low_let

    # Excess band: the ratio-to-paper is window-dependent by tens of percent, and
    # falls monotonically in the window's CEILING -- the mechanism, rather than an
    # ordering of the three window names that a change of default can invert.
    band = fits(50.0)
    paper = float(np.interp(50.0, bx, by))
    ratios = {k: v / paper for k, v in band.items()}
    by_ceiling = sorted(windows, key=lambda k: float(windows[k].max()))
    ordered = [ratios[k] for k in by_ceiling]
    assert ordered == sorted(ordered, reverse=True), (by_ceiling, ratios)
    assert max(ratios.values()) - min(ratios.values()) > 0.3, ratios
    # ...while the late roll-off (still above the paper here) survives every window.
    assert min(ratios.values()) > 1.0, ratios


@pytest.mark.golden
@pytest.mark.slow
def test_the_beta_rolloff_tracks_the_paper_and_costs_under_1_percent_of_rbe10() -> None:
    """Pin *what* the ``beta_I`` residual is, and *what it costs*.

    **The mid-LET ``beta`` excess is largely an artifact of an absolute
    0.4-3.0 Gy fit window.** Under the ``0.05-0.85 D_t`` window every number
    inverts:

    ======================  =================  ==================
    quantity                absolute window    D_t-rel. window
    ======================  =================  ==================
    our ``beta_I`` peak     LET 45 @ 0.0585    LET 27.5 @ 0.0524
    paper's peak            LET 24 @ 0.0522    LET 24 @ 0.0522
    ratio of maxima         1.12               **1.004**
    peak separation         ~21 keV/um         **3.5 keV/um**
    worst ``RBE_10`` cost   +5.4% at LET 45    **+0.61% at LET 40**
    ======================  =================  ==================

    So "the roll-off starts ~21 keV/um late from a 12% higher maximum" is not a
    statement about the model. What remains is a **high-LET tail**, not a
    mid-LET excess: ``beta`` tracks the paper within ~4% out to LET 50 and then
    rises, reaching 1.18x at LET 65.

    Two things this does NOT say. First, the window is a FREE parameter chosen by
    scoring against this very curve (LEM_IV_OPEN_CHOICES.md entry 1.2), so the
    agreement is partly by construction. Second, it does not touch the
    enhancement-roll-off no-go, which was established on the *clustering*
    intermediate via the discrete-DSB engine, not on fitted ``beta``.

    What it costs: holding ``alpha`` fixed and swapping only ``beta``, the whole
    residual moves ``RBE_10`` by at most ~0.6%. That bound is why no runtime
    warning is raised over the mid-LET band, and why the existing deep-overkill
    warning (~150 keV/um) must not be read as marking it.
    """
    bx, by = _fig4_curve("beta")
    lem = LEM4(photon=FIG4, cell=FIG4_CELL)
    doses = lem.fit_doses_Gy()

    lets = np.arange(15.0, 66.0, 2.5)
    fits = [
        _localdose.lq_coefficients(doses, lem.local_dose_lethal_events(_carbon_for_let(L), doses))
        for L in lets
    ]
    betas = np.array([b for _a, b in fits])

    our_peak_let = float(lets[int(np.argmax(betas))])
    paper_peak_let = float(bx[int(np.argmax(by))])
    assert 20.0 <= paper_peak_let <= 28.0, paper_peak_let
    # The peak lands on the paper's, within the 2.5 keV/um grid step plus one.
    assert abs(our_peak_let - paper_peak_let) < 5.0, (our_peak_let, paper_peak_let)
    assert 0.98 < betas.max() / by.max() < 1.03, betas.max() / by.max()

    # The residual is a high-LET tail, not a mid-LET excess: flat to ~4% through
    # LET 50, rising after. Pinned in both halves so a regression to a
    # mid-LET excess would fail the first assertion.
    ratios = np.array([b / float(np.interp(L, bx, by)) for L, b in zip(lets, betas, strict=True)])
    mid = ratios[lets <= 50.0]
    assert mid.max() < 1.06, (lets[lets <= 50.0], mid)
    assert ratios[-1] > 1.10, ratios[-1]

    # RBE_10 cost of the beta residual alone (same alpha in both arms). Measured
    # 0.61% at LET 40 (5.4% at LET 45 under the absolute window).
    ref = LQParameters(FIG4.alpha_x, FIG4.beta_x)
    worst = 0.0
    for let, (a_fit, b_our) in zip(lets, fits, strict=True):
        b_pap = float(np.interp(let, bx, by, left=np.nan, right=np.nan))
        assert math.isfinite(b_pap), let
        ours = rbe(0.10, a_fit, max(b_our, 0.0), reference=ref)
        theirs = rbe(0.10, a_fit, max(b_pap, 0.0), reference=ref)
        worst = max(worst, abs(ours / theirs - 1.0))
    # Two-sided: the upper bound is the validity claim, the lower one guards
    # against the residual being silently zeroed by some future change.
    assert 0.002 < worst < 0.010, worst


def test_both_cluster_index_kernels_read_the_eta_enhanced_dose() -> None:
    """``discrete_cluster_index`` must be dosed like ``cluster_index``, not like the
    per-subvolume kernels.

    The two cluster-index kernels differ only in how a realisation's ratio is
    formed; the loops they score are dosed identically, with the *eta-enhanced*
    dose. The MC picks that field by kernel name, and matching on the single
    string ``"cluster_index"`` silently handed the discrete variant the plain
    physical dose -- worth a factor ~4 in effect, which read convincingly as a
    physics result ("the discrete convention under-clusters") rather than a typo.

    Guarded by the concentration limit rather than a tolerance pulled from air:
    a whole-nucleus realisation damages hundreds of loops, so ``E[A/B]`` and
    ``E[A]/E[B]`` must agree closely. A factor of 4 is not a convention
    difference, it is a different dose field.
    """
    ion = _carbon_for_let(100.0)
    curves = {}
    for kernel in ("cluster_index", "discrete_cluster_index"):
        lem = LEM4(photon=FIG4, cell=FIG4_CELL, stopping_power=_SP, local_dose_kernel=kernel)
        effect, _surv, _doses = lem._local_dose_mc_curves(
            ion, doses_Gy=[1.0, 2.0], n_realizations=200, seed=0
        )
        curves[kernel] = np.asarray(effect)

    ratio = curves["discrete_cluster_index"] / curves["cluster_index"]
    assert np.all(np.isfinite(ratio)), ratio
    assert np.all((ratio > 0.7) & (ratio < 1.4)), (
        f"the two cluster-index kernels disagree by {ratio}; a factor this large means "
        "they are not reading the same (eta-enhanced) dose field"
    )


def test_discrete_cluster_index_is_monte_carlo_only() -> None:
    """The deterministic convolution must refuse the average-of-ratios kernel.

    Convolving a mean dose field *is* the mean-field construction that kernel
    exists to avoid, so silently returning the ``cluster_index`` answer there
    would be a wrong number under a label that promises otherwise.
    """
    lem = LEM4(
        photon=FIG4, cell=FIG4_CELL, stopping_power=_SP, local_dose_kernel="discrete_cluster_index"
    )
    with pytest.raises(NotImplementedError, match="Monte-Carlo only"):
        lem.local_dose_lethal_events(_carbon_for_let(100.0), np.array([1.0, 2.0]))


def test_impact_cut_defaults_to_the_exact_engine() -> None:
    """The split is opt-in: unset, the MC is bit-identical to the pre-split engine.

    A cost optimisation that silently moves numbers is worse than no optimisation, so
    ``local_dose_mc_impact_cut=None`` must not merely *agree* with the exact engine --
    it must take the same code path and return the same floats.
    """
    lem = LEM4(photon=FIG4, cell=FIG4_CELL, stopping_power=_SP)
    ion = _carbon_for_let(40.0)
    assert lem.resolve_mc_impact_cut(ion) is None
    a, b = lem.local_dose_mc_alpha_beta(ion, n_realizations=40, seed=0)
    a2, b2 = lem.local_dose_mc_alpha_beta(ion, n_realizations=40, seed=0)
    assert (a, b) == (a2, b2)


def test_impact_cut_engages_only_where_it_saves_work() -> None:
    """``"auto"`` must not split when the domain is already small.

    Splitting a domain that is barely larger than the cut costs the approximation and
    buys nothing, so the rule returns ``None`` there and the exact engine runs.
    """
    lem = LEM4(photon=FIG4, cell=FIG4_CELL, stopping_power=_SP, local_dose_mc_impact_cut="auto")
    r_eff = FIG4_CELL.effective_nucleus_radius_um()
    assert lem.resolve_mc_impact_cut(_carbon_for_let(200.0)) is None  # tiny track
    far = lem.resolve_mc_impact_cut(Ion(6, 12, 143.7))  # LET ~20, r_sim ~ 293 um
    assert far is not None
    # whichever of the two rules binds, never below either
    assert far >= _MC_IMPACT_CUT_FACTOR * r_eff
    assert far >= _MC_IMPACT_CUT_MIN_DOMAIN_FRACTION * (
        r_eff + lem._local_dose_profile(Ion(6, 12, 143.7))[3]
    )


def test_impact_cut_rejects_a_cut_outside_the_domain() -> None:
    """A cut at or beyond ``r_eff + r_max`` is a silent no-op; refuse it instead."""
    lem = LEM4(photon=FIG4, cell=FIG4_CELL, stopping_power=_SP, local_dose_mc_impact_cut=1.0e6)
    with pytest.raises(ValueError, match="local_dose_mc_impact_cut"):
        lem.resolve_mc_impact_cut(_carbon_for_let(40.0))


@pytest.mark.slow
def test_impact_cut_reproduces_the_full_simulation() -> None:
    """The track/background split must not move ``alpha`` or ``beta``.

    Validated at LET 40, where the exact engine is still affordable, with the cut
    pushed to 3 and 8 nucleus radii -- far harder than the 20 that ``"auto"`` uses,
    so this measures the approximation rather than confirming a safe default.

    What the split drops is the far population's *fluctuation*, so the failure mode
    is a biased ``beta`` (a curvature that is not there), not a biased ``alpha``.
    Measured over 6 seeds x 400 realisations, every cut from 3 to 12 r_eff agreed
    within 1.8 sigma with no trend in cut size, at up to 57x the speed.
    """
    ion = _carbon_for_let(40.0)
    base = LEM4(photon=FIG4, cell=FIG4_CELL, stopping_power=_SP)
    doses = base.fit_doses_Gy()
    seeds = range(4)

    def run(cut: float | None) -> tuple[float, float, float, float]:
        lem = LEM4(photon=FIG4, cell=FIG4_CELL, stopping_power=_SP, local_dose_mc_impact_cut=cut)
        p = np.array(
            [
                lem.local_dose_mc_alpha_beta(ion, doses_Gy=doses, n_realizations=300, seed=s)
                for s in seeds
            ]
        )
        n = math.sqrt(p.shape[0])
        return (
            float(p[:, 0].mean()),
            float(p[:, 0].std(ddof=1) / n),
            float(p[:, 1].mean()),
            float(p[:, 1].std(ddof=1) / n),
        )

    a0, ae0, b0, be0 = run(None)
    for k in (3.0, 8.0):
        a, ae, b, be = run(k * FIG4_CELL.effective_nucleus_radius_um())
        assert abs(a - a0) < 4.0 * math.hypot(ae0, ae), (
            f"cut at {k} r_eff moved alpha: {a0:.4f}+-{ae0:.4f} -> {a:.4f}+-{ae:.4f}"
        )
        assert abs(b - b0) < 4.0 * math.hypot(be0, be), (
            f"cut at {k} r_eff moved beta: {b0:.5f}+-{be0:.5f} -> {b:.5f}+-{be:.5f}"
        )


def test_impact_cut_is_floored_by_a_domain_fraction_not_only_by_r_eff() -> None:
    """``8 * r_eff`` alone biases ``beta`` low at large domains -- measured, not argued.

    What controls the mean-field replacement is the cut as a FRACTION OF THE DOMAIN.
    Two validations of the ``8 * r_eff`` rule missed this because both sat at a
    comfortable fraction by accident (0.136 of ``r_sim`` at LET 20, 0.62 at LET 40).
    At LET 15 the domain is 609 um, ``8 * r_eff`` is 0.066 of it, and ``beta`` reads
    0.04952 / 0.04987 / 0.04991 for cuts of 40 / 80 / 120 um -- biased low by 0.0004,
    which is the size of the intertrack enhancement this engine exists to measure. It
    depressed the low-LET baseline and manufactured an apparent ``beta`` peak.

    So the rule carries a floor, and this pins that the floor actually binds where it
    has to: at large domains the cut must scale with ``r_sim``, not with the nucleus.
    """
    lem = LEM4(photon=FIG4, cell=FIG4_CELL, stopping_power=_SP, local_dose_mc_impact_cut="auto")
    r_eff = FIG4_CELL.effective_nucleus_radius_um()

    big = Ion(6, 12, 222.0)  # LET ~15, r_sim ~ 609 um
    r_sim = r_eff + lem._local_dose_profile(big)[3]
    cut = lem.resolve_mc_impact_cut(big)
    assert cut is not None
    assert cut > _MC_IMPACT_CUT_FACTOR * r_eff, "the domain floor is not binding at LET 15"
    assert cut / r_sim == pytest.approx(_MC_IMPACT_CUT_MIN_DOMAIN_FRACTION, rel=1e-9)

    # ...and it must NOT bind where r_eff already gives a generous fraction, or every
    # mid-LET point pays for a problem it does not have.
    small = Ion(6, 12, 57.0)  # LET ~40, r_sim ~ 65 um
    cut_small = lem.resolve_mc_impact_cut(small)
    assert cut_small == pytest.approx(_MC_IMPACT_CUT_FACTOR * r_eff)


# ---- the survival-bounded fit window (Fig. 4's "S>0.01" legend) --------
def test_fit_survival_bound_is_off_by_default_and_validated() -> None:
    """The survival-bounded window is opt-in, and the bound is a survival level."""
    lem = LEM4(photon=FIG4, cell=FIG4_CELL)
    assert lem.fit_survival_bound is None
    # ...and with it unset the ``ion`` argument changes nothing.
    assert np.array_equal(lem.fit_doses_Gy(), lem.fit_doses_Gy(ion=_carbon_for_let(50.0)))
    with pytest.raises(ValueError, match="survival level in"):
        LEM4(photon=FIG4, cell=FIG4_CELL, fit_survival_bound=1.0)
    with pytest.raises(ValueError, match="survival level in"):
        LEM4(photon=FIG4, cell=FIG4_CELL, fit_survival_bound=0.0)
    # A bounded model with no ion falls back to the fixed grid rather than raising:
    # callers that legitimately have no ion (a photon-side grid) keep working.
    bounded = LEM4(photon=FIG4, cell=FIG4_CELL, fit_survival_bound=0.01)
    assert np.array_equal(bounded.fit_doses_Gy(), lem.fit_doses_Gy())


def test_survival_bounded_ceiling_reads_the_simulated_response_not_the_fit() -> None:
    """``S = e^{-E}`` is thresholded on the simulated response itself.

    Non-circularity is the whole point: a ceiling derived from the fitted LQ curve
    would depend on the fit it bounds. Verified by evaluating the *simulated*
    effect at the returned dose and checking it is exactly ``-ln 0.01``.
    """
    lem = LEM4(photon=FIG4, cell=FIG4_CELL, fit_survival_bound=0.01)
    ion = _carbon_for_let(50.0)
    ceiling = lem.survival_bounded_ceiling_Gy(ion)
    effect = float(lem.local_dose_lethal_events(ion, np.array([ceiling]))[0])
    assert effect == pytest.approx(-math.log(0.01), rel=2e-3)
    # It is well inside the fixed window's ceiling here, and the fit is the one the
    # narrower window implies -- i.e. the option is actually wired to the routes.
    assert ceiling < 0.85 * 9.5
    doses = lem.fit_doses_Gy(ion=ion)
    assert doses[-1] == pytest.approx(ceiling)
    expected = _localdose.lq_coefficients(doses, lem.local_dose_lethal_events(ion, doses))[1]
    assert lem.local_dose_beta(ion) == pytest.approx(expected)


def test_survival_bounded_window_only_moves_the_ceiling() -> None:
    """The node *shape* is preserved: a ceiling at ``0.85 D_t`` gives the fixed grid.

    This is what makes a fixed-vs-bounded comparison a comparison of ceilings
    rather than of node distributions -- the fractions are re-mapped onto
    ``[lo, ceiling]`` at their own log positions, and the lower bound never moves.
    """
    lem = LEM4(photon=FIG4, cell=FIG4_CELL, fit_survival_bound=0.01)
    fixed = LEM4(photon=FIG4, cell=FIG4_CELL).fit_doses_Gy()
    ion = _carbon_for_let(50.0)
    with pytest.MonkeyPatch.context() as mp:
        mp.setattr(LEM4, "survival_bounded_ceiling_Gy", lambda self, i, s=None: float(fixed[-1]))
        assert lem.fit_doses_Gy(ion=ion) == pytest.approx(fixed, rel=1e-12)
    assert lem.fit_doses_Gy(ion=ion)[0] == pytest.approx(fixed[0])


@pytest.mark.golden
@pytest.mark.slow
def test_the_paper_s_beta_needs_a_fixed_dose_ceiling_not_a_survival_bounded_one() -> None:
    """Friedrich 2012 Fig. 4 labels its full-simulation series ``"Full Simulation,
    S>0.01"``. Read as a fit protocol that makes the LQ window's ceiling the dose
    at which simulated survival reaches 1%, and so LET-dependent. **Our response
    refitted over exactly that window does not reproduce the paper's**
    ``beta_I`` (doi:10.3109/09553002.2011.611213).

    Measured by refitting the *same* simulated response (the
    deterministic ``mean_effect`` convolution, common points) over each window:

    ==== ============== ============= ==============
    LET  ceiling S>0.01 beta / paper  fixed / paper
    ==== ============== ============= ==============
    30   6.0 Gy         1.046         1.014
    40   5.3 Gy         1.194         1.038
    50   4.7 Gy         1.312         1.033
    55   4.4 Gy         1.288         1.021
    ==== ============== ============= ==============

    Two things this test pins, in the order they matter.

    1. **The direction.** A lower ceiling reads a higher ``beta``; ``S>0.01`` sits
       below ``0.85 D_t`` at every LET here, so it must read high. This is the
       diagnostic, not a failure of the hypothesis: if the paper fitted over
       ``S>0.01`` and our ``S>0.01`` fit disagrees with it, then our simulated
       survival curve differs from theirs and the fixed window's agreement is
       compensation.
    2. **The shape.** The ceiling that *would* reproduce the paper's ``beta`` is a
       near-constant **dose** (8.15-8.77 Gy over LET 17.5-55, a 7% spread), not a
       constant **survival**: the survival at that required ceiling spans 1.7e-3
       to 7.8e-5, a factor 21. So the paper's own curve is not consistent with any
       single survival bound -- freely tuning the threshold, the best member of the
       survival-bounded family (``S>1e-4``) still scores worse over LET 13-70 than
       every fixed ceiling between 0.85 and 1.10 ``D_t``.

    Robust to the escape routes: the verdict is unchanged under the exact
    ``<e^{-E}>`` survival convention (entry 2.1; LET 45/50/55 give 1.12/1.11/1.28
    bounded against 0.95/0.93/1.00 fixed on a paired explicit-track MC), and under
    the lower bound and node spacing (LET 50 gives 1.22-1.35 for four lower-bound
    and linear/log spacing choices).

    See ``docs/references/LEM_IV_OPEN_CHOICES.md`` entry 1.2.
    """
    bx, by = _fig4_curve("beta")
    fixed = LEM4(photon=FIG4, cell=FIG4_CELL)
    bounded = LEM4(photon=FIG4, cell=FIG4_CELL, fit_survival_bound=0.01)

    ratios: dict[float, tuple[float, float, float]] = {}
    for let in (30.0, 40.0, 50.0, 55.0):
        ion = _carbon_for_let(let)
        paper = float(np.interp(let, bx, by))
        d_fix = fixed.fit_doses_Gy()
        d_bnd = bounded.fit_doses_Gy(ion=ion)
        # Common points: one response, two windows.
        nodes = np.unique(np.concatenate([d_fix, d_bnd]))
        effect = fixed.local_dose_lethal_events(ion, nodes)
        b_fix = _localdose.lq_coefficients(d_fix, effect[np.isin(nodes, d_fix)])[1]
        b_bnd = _localdose.lq_coefficients(d_bnd, effect[np.isin(nodes, d_bnd)])[1]
        ratios[let] = (b_fix / paper, b_bnd / paper, float(d_bnd[-1]))

    for let, (r_fix, r_bnd, ceiling) in ratios.items():
        assert ceiling < 0.85 * 9.5, (let, ceiling)
        assert r_bnd > r_fix, (let, r_fix, r_bnd)
    # The bounded window is 19-31% high where the paper's beta is largest and
    # window-sensitive, while the fixed one is within ~4%.
    assert max(r for _f, r, _c in ratios.values()) > 1.15, ratios
    assert max(f for f, _r, _c in ratios.values()) < 1.06, ratios

    # ...and the required ceiling is a constant DOSE, not a constant SURVIVAL.
    required: list[tuple[float, float]] = []
    for let in (20.0, 30.0, 40.0, 50.0, 55.0):
        ion = _carbon_for_let(let)
        paper = float(np.interp(let, bx, by))
        lo = float(fixed.fit_doses_Gy()[0])

        def beta_at(hi: float, _ion: Ion = ion, _lo: float = lo) -> float:
            grid = np.geomspace(_lo, hi, 12)
            return _localdose.lq_coefficients(grid, fixed.local_dose_lethal_events(_ion, grid))[1]

        hi = brentq(lambda h, _p=paper: beta_at(h) - _p, 3.0, 30.0, xtol=1e-3)
        survival = math.exp(-float(fixed.local_dose_lethal_events(ion, np.array([hi]))[0]))
        required.append((hi, survival))
    doses = [d for d, _s in required]
    survivals = [s for _d, s in required]
    assert max(doses) / min(doses) < 1.25, required
    assert max(survivals) / min(survivals) > 5.0, required


# ---- public (alpha_I, beta_I) entry points ------------------------------
#
# The public methods must give the same bits as an inline loop over
# ``_localdose.lq_coefficients`` and the private discrete-DSB helpers.


def _hypothetical(**kw: object) -> LEM4:
    inputs = f12.FIGURE_INPUTS["fig4_hypothetical"]
    return LEM4(photon=inputs.photon(), cell=inputs.cell_target(), **kw)  # type: ignore[arg-type]


def test_local_dose_alpha_beta_is_the_reach_in_fit_bit_for_bit() -> None:
    """``local_dose_alpha_beta`` = ``lq_coefficients(fit_doses_Gy(), lethal_events)``,
    unclamped, and ``local_dose_beta`` is its clamped ``beta`` (the overkill model)."""
    lem = _hypothetical(local_dose_overkill=True)
    doses = lem.fit_doses_Gy()
    ion = _carbon_for_let(100.0)
    inline = _localdose.lq_coefficients(doses, lem.local_dose_lethal_events(ion, doses))
    assert lem.local_dose_alpha_beta(ion) == inline
    assert lem.local_dose_beta(ion, doses_Gy=list(doses)) == max(inline[1], 0.0)


def test_local_dose_alpha_beta_does_not_clamp_or_warn_in_deep_overkill() -> None:
    lem = _hypothetical(local_dose_overkill=True)
    with warnings.catch_warnings():
        warnings.simplefilter("error")
        _alpha, beta = lem.local_dose_alpha_beta(_carbon_for_let(200.0))
    assert beta < 0.0


def _script_discrete_overkill(lem: LEM4, ion: Ion) -> tuple[float, float]:
    """An inline loop over the private helpers: cluster index clipped to ``[C_g[0], C_g[-1]]``."""
    doses = lem.fit_doses_Gy()
    c_g, g_per_loop = lem._intertrack_pe_lookup()
    setup = lem._discrete_setup(ion)
    effect = np.empty(doses.size)
    for j, dose in enumerate(doses):
        weights, n_dmg, n_cdsb = lem._discrete_counts(
            setup, float(dose), setup.retained_dose_fraction
        )
        cluster = np.clip(
            np.divide(n_cdsb, n_dmg, out=np.zeros_like(n_dmg), where=n_dmg > 0.0),
            c_g[0],
            c_g[-1],
        )
        e_n = n_dmg * np.interp(cluster, c_g, g_per_loop)
        survival = float(np.sum(weights * np.exp(-np.minimum(e_n, 700.0))))
        effect[j] = -np.log(max(survival, 1.0e-300))
    return _localdose.lq_coefficients(doses, effect)


@pytest.mark.parametrize("let", [10.0, 150.0])
def test_discrete_dsb_alpha_beta_matches_the_scripts_clip_bit_for_bit(let: float) -> None:
    """The C_I-clip gate: the library clips to ``1 - 1e-15``, an inline loop to
    ``C_g[-1]`` (= 1.0). Across the LET 10-150 grid the per-traversal cluster
    index stays below 0.43, so neither ceiling binds and the values are identical (the
    full 50-point grid was checked; the ends are pinned here)."""
    lem = _hypothetical(beta_model="discrete_dsb")
    ion = _carbon_for_let(let)
    public = lem.discrete_dsb_alpha_beta(ion, convention="per_realisation", survival="stochastic")
    assert public == _script_discrete_overkill(lem, ion)
    # discrete_dsb_beta (which delegates) equals the inline fit of this model's own
    # convention/survival on the discrete window
    doses = lem.fit_doses_Gy(_DISCRETE_DOSE_FRACTIONS, ion=ion)
    inline = _localdose.lq_coefficients(doses, lem.discrete_dsb_lethal_events(ion, doses))[1]
    assert lem.discrete_dsb_beta(ion) == inline
