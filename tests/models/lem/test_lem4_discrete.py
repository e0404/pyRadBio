"""Tests for the LEM IV discrete-DSB engine (``_discrete`` + ``LEM4`` wiring).

The engine is the Friedrich et al. 2012 (doi:10.3109/09553002.2011.611213) Fig. 4
full simulation built as an *explicit discrete damage pattern*: Poisson
nucleus-traversing tracks placing discrete DSBs, plus the escaped dose as a
uniform Poisson DSB background. It is meant to recover the roll-off of the
intertrack enhancement that the shipped continuum ``"local_dose"`` engine cannot
produce.

**It does not recover it, and the reason is a theorem** -- see
:func:`test_discrete_mean_counts_equals_the_continuum_density` and
``docs/references/LEM_IV_Friedrich2012.md``. These tests pin that result so the
line of attack cannot be quietly reopened.
"""

from __future__ import annotations

import math

import numpy as np
import pytest
from scipy.optimize import brentq
from scipy.stats import poisson

from pyradbio.benchmark import friedrich2012 as f12
from pyradbio.core.cell import CellTarget
from pyradbio.core.photon import PhotonParameters
from pyradbio.core.radiation import Ion
from pyradbio.models.lem import _discrete, _localdose
from pyradbio.models.lem.lem4 import (
    _DISCRETE_DOSE_FRACTIONS,
    DSB_PER_GY_PER_LOOP,
    LEM4,
)
from pyradbio.track.stopping_power import SimpleBetheStoppingPower

FIG4 = PhotonParameters(alpha_x=0.25, beta_x=0.05, D_t=9.5)
FIG4_CELL = CellTarget(nucleus_radius_um=5.0)
_SP = SimpleBetheStoppingPower()


def _carbon_for_let(let_keV_um: float) -> Ion:
    """Carbon-12 whose water LET equals ``let_keV_um``."""
    energy = float(brentq(lambda e: _SP.let_keV_um(Ion(6, 12, e)) - let_keV_um, 1.6, 600.0))
    return Ion(6, 12, energy)


# ---- geometry: the two-disk overlap and the retained dose fraction ------
def test_disk_overlap_fraction_hits_its_three_known_values() -> None:
    """``O(0) = 1``, ``O(2R) = 0``, and ``O(R) = (2/pi)(pi/3 - sqrt(3)/4)``."""
    r = np.array([0.0, 5.0, 10.0, 12.0])
    got = _discrete.disk_overlap_fraction(r, 5.0)
    expected_at_r = (2.0 / math.pi) * (math.acos(0.5) - 0.5 * math.sqrt(0.75))
    assert got[0] == pytest.approx(1.0)
    assert got[1] == pytest.approx(expected_at_r)
    assert got[2] == pytest.approx(0.0)
    assert got[3] == pytest.approx(0.0)  # beyond 2R the disks are disjoint


def test_retained_dose_fraction_brackets_a_point_track_and_a_far_track() -> None:
    """All dose at ``r -> 0`` is retained; all dose beyond ``2 R_eff`` escapes.

    ``f_ret`` is the dose-conserving track/background split -- the fraction of a
    uniformly-placed track's dose that lands in the nucleus. It is derived, not
    fitted, so these limits must hold exactly.
    """
    r = np.logspace(-4.0, 2.0, 400)
    near = np.where(r < 1.0e-3, 1.0, 0.0)
    far = np.where(r > 20.0, 1.0, 0.0)
    assert _discrete.retained_dose_fraction(r, near, r_eff=5.0) == pytest.approx(1.0, abs=2e-3)
    assert _discrete.retained_dose_fraction(r, far, r_eff=5.0) == pytest.approx(0.0, abs=1e-9)


def test_retained_dose_fraction_falls_as_the_track_grows() -> None:
    """Monotone in the track's radial extent -- the LET trend the split carries."""
    r = np.logspace(-4.0, 2.0, 600)
    fractions = [
        _discrete.retained_dose_fraction(r, np.where(r < top, 1.0 / (r * r), 0.0), r_eff=5.0)
        for top in (0.5, 2.0, 10.0, 50.0)
    ]
    assert fractions == sorted(fractions, reverse=True), fractions
    assert 0.0 < fractions[-1] < fractions[0] < 1.0


# ---- the per-loop single-track factors a0, a1 --------------------------
def test_track_factors_reduce_to_the_uniform_field_values() -> None:
    """A spatially constant ``mu`` gives ``a0 = e^-mu`` and ``a1 = mu e^-mu``.

    ``a0(x) = <e^{-mu}>_b`` and ``a1(x) = <mu e^{-mu}>_b`` over track positions
    ``b`` uniform on the nucleus disk; for constant ``mu`` the average is trivial,
    at every loop position including the rim.
    """
    rho = np.logspace(-3.0, 1.5, 200)
    mu = np.full_like(rho, 0.35)
    a0, a1 = _discrete.track_factors(rho, mu, r_eff=5.0, ell=0.54, step_um=0.05)
    assert np.allclose(a0, math.exp(-0.35), rtol=2e-3)
    assert np.allclose(a1, 0.35 * math.exp(-0.35), rtol=2e-3)


def test_track_factors_match_a_direct_quadrature() -> None:
    """The FFT disk-convolution equals a brute-force 2-D average over track positions.

    The 4 ``R_eff`` box is exactly wide enough that the circular convolution is
    alias-free at every sampled loop centre (``|x - b| <= 2 R_eff`` for ``x``, ``b``
    both in the disk), which is the property this test guards.
    """
    r_eff, ell = 4.0, 0.54
    rho = np.logspace(-3.0, 1.2, 300)
    mu = 0.8 * np.exp(-rho)
    a0, _a1 = _discrete.track_factors(rho, mu, r_eff=r_eff, ell=ell, step_um=0.04)
    lx, ly = _localdose.subvolume_centres(r_eff, ell)

    step = 0.02
    ax = np.arange(-r_eff, r_eff, step) + 0.5 * step
    bx, by = np.meshgrid(ax, ax, indexing="ij")
    inside = (bx * bx + by * by) <= r_eff * r_eff
    bx, by = bx[inside], by[inside]
    for probe in (0, len(lx) // 3, len(lx) - 1):
        dist = np.sqrt((lx[probe] - bx) ** 2 + (ly[probe] - by) ** 2)
        mu_b = np.interp(dist, rho, mu, left=float(mu[0]), right=0.0)
        assert a0[probe] == pytest.approx(float(np.mean(np.exp(-mu_b))), rel=3e-3)


# ---- conditional loop counts -------------------------------------------
def test_conditional_counts_with_no_tracks_are_the_uniform_photon_pattern() -> None:
    """``n = 0`` leaves only the background: Poisson``(m_bg)`` DSBs per loop.

    So ``N_dmg = N (1 - e^-m)`` and ``N_cDSB = N (1 - e^-m (1 + m))`` -- the photon
    limit of the discrete pattern, recovered rather than calibrated.
    """
    a0 = np.full(40, 0.9)
    a1 = np.full(40, 0.08)
    m_bg = 0.03
    n_dmg, n_cdsb = _discrete.conditional_loop_counts(a0, a1, n_tracks=0, m_bg=m_bg, n_z=3.0)
    n_loops = 40 * 3.0
    assert n_dmg == pytest.approx(n_loops * (1.0 - math.exp(-m_bg)))
    assert n_cdsb == pytest.approx(n_loops * (1.0 - math.exp(-m_bg) * (1.0 + m_bg)))


def test_conditional_counts_grow_with_track_number() -> None:
    """More tracks damage more loops and cluster a larger share of them."""
    a0 = np.full(60, 0.85)
    a1 = np.full(60, 0.12)
    counts = [
        _discrete.conditional_loop_counts(a0, a1, n_tracks=n, m_bg=0.01, n_z=2.0)
        for n in (0, 1, 4, 12)
    ]
    assert [c[0] for c in counts] == sorted(c[0] for c in counts)
    ratios = [c[1] / c[0] for c in counts]
    assert ratios == sorted(ratios), ratios


# ---- THE theorem: discreteness is invisible to the cluster index -------
@pytest.mark.slow
def test_discrete_mean_counts_equals_the_continuum_density() -> None:
    """Mean-counts discrete scoring **is** the continuum ``cluster_index`` kernel.

    A track's Poisson DSB total split over loops by its radial profile gives
    *independent Poisson* counts per loop, so a loop's occupancy ``N`` is compound
    Poisson with PGF ``G(z) = E[e^{m(z-1)}]``, ``m = kappa d`` the eta-enhanced
    local dose. Hence

        P(N = 0) = G(0)  = E[e^-m]        -> the same P_dmg
        P(N = 1) = G'(0) = E[m e^-m]      -> the same P_cDSB

    which are exactly the expectations ``_local_dose_cluster_index_events``
    evaluates. **Discreteness therefore cannot change the cluster index**, and the
    hoped-for distinction between "two DSBs in a loop from the same track" and
    "from different tracks" does not exist: ``C_I`` is a function of loop occupancy
    only and never sees track provenance.

    This is why the discrete engine does not recover the Fig. 4 enhancement
    roll-off. Pinned here so the approach is not re-attempted.
    """
    lem = LEM4(photon=FIG4, cell=FIG4_CELL, beta_model="discrete_dsb")
    doses = lem.fit_doses_Gy(_DISCRETE_DOSE_FRACTIONS)
    for let in (30.0, 70.0, 100.0):
        ion = _carbon_for_let(let)
        discrete = lem.discrete_dsb_lethal_events(ion, doses, convention="mean_counts")
        continuum = LEM4(photon=FIG4, cell=FIG4_CELL).local_dose_lethal_events(ion, doses)
        assert np.allclose(discrete, continuum, rtol=0.04), (let, discrete, continuum)
        b_d = _localdose.lq_coefficients(doses, discrete)[1]
        b_c = _localdose.lq_coefficients(doses, continuum)[1]
        assert b_d == pytest.approx(b_c, abs=0.0025), (let, b_d, b_c)


def test_poisson_track_weights_cover_the_distribution() -> None:
    """The truncated track-number grid keeps essentially all the Poisson mass."""
    for lam in (0.4, 5.0, 60.0, 300.0):
        ns, w = _discrete.poisson_track_weights(lam)
        assert w.sum() == pytest.approx(1.0, abs=1e-9)
        assert float(np.sum(ns * w)) == pytest.approx(lam, rel=1e-6)
        assert poisson.sf(int(ns[-1]), lam) < 1e-9


# ---- LEM4 wiring --------------------------------------------------------
def test_discrete_beta_model_validation_and_routing(monkeypatch: pytest.MonkeyPatch) -> None:
    """``beta_model="discrete_dsb"`` supplies only ``beta``; ``alpha_I`` is untouched."""
    with pytest.raises(ValueError, match="discrete_convention"):
        LEM4(photon=FIG4, cell=FIG4_CELL, discrete_convention="both")  # type: ignore[arg-type]
    ion = _carbon_for_let(120.0)
    monkeypatch.setattr(LEM4, "discrete_dsb_beta", lambda self, i: 0.0177)
    lem = LEM4(photon=FIG4, cell=FIG4_CELL, beta_model="discrete_dsb")
    reference = LEM4(photon=FIG4, cell=FIG4_CELL, beta_model="slope_matching")
    lq = lem.alpha_beta(ion)
    assert lq.beta == pytest.approx(0.0177)
    assert lq.alpha == pytest.approx(reference.alpha_beta(ion).alpha)


def test_discrete_survival_validation() -> None:
    """``discrete_survival`` is checked, and ``"stochastic"`` needs per-traversal effects."""
    with pytest.raises(ValueError, match="discrete_survival"):
        LEM4(photon=FIG4, cell=FIG4_CELL, discrete_survival="both")  # type: ignore[arg-type]
    with pytest.raises(ValueError, match="per_realisation"):
        LEM4(
            photon=FIG4,
            cell=FIG4_CELL,
            discrete_survival="stochastic",
            discrete_convention="mean_counts",
        )


@pytest.mark.slow
def test_stochastic_survival_lowers_the_effect_and_converges_at_low_let() -> None:
    """``-ln <exp(-E)> <= <E>`` pointwise, with equality approached as tracks get dense.

    Jensen gives the inequality at every dose, so the stochastic effect curve can
    never sit above the mean-effect one -- the sharpest available check that the
    convention is wired up right. And the gap must *close* toward low LET, where the
    traversal count ``lambda = D/D_1`` is large and its relative fluctuation small:
    at LET 10 the two ``beta`` agree to ~1% (0.0491 vs 0.0488, both on ``beta_x``),
    while at LET 130 the effect is a third lower.

    Note the ordering is asserted on the *effect*, not on ``beta``: ``beta`` is a
    curvature, and past ~100 keV/um the stochastic ``beta`` turns back *up* (0.0079
    at LET 100, 0.0063 at 130) as deep-overkill convexity re-enters -- which is real,
    and is exactly what the ``local_dose`` B2 correction drops.
    """
    mean = LEM4(photon=FIG4, cell=FIG4_CELL, beta_model="discrete_dsb")
    stoch = LEM4(
        photon=FIG4, cell=FIG4_CELL, beta_model="discrete_dsb", discrete_survival="stochastic"
    )
    doses = stoch.fit_doses_Gy(_DISCRETE_DOSE_FRACTIONS)
    for let in (10.0, 55.0, 130.0):
        ion = _carbon_for_let(let)
        e_mean = mean.discrete_dsb_lethal_events(ion, doses)
        e_stoch = stoch.discrete_dsb_lethal_events(ion, doses)
        assert np.all(e_stoch <= e_mean + 1e-12), (let, e_mean, e_stoch)
    low = _carbon_for_let(10.0)
    assert stoch.discrete_dsb_beta(low) == pytest.approx(mean.discrete_dsb_beta(low), rel=0.02)
    assert stoch.discrete_dsb_beta(low) == pytest.approx(FIG4.beta_x, rel=0.06)


@pytest.mark.slow
def test_stochastic_survival_does_not_rescue_the_discrete_engine() -> None:
    """Scholz 1997's own survival convention is a tradeoff, and the sign is not robust.

    **Both halves of the finding invert with the LQ fit window** (absolute
    ``0.4-3.0 Gy`` against ``0.05-0.85 D_t``).

    Over LET 10-130 against the digitized Fig. 4 full simulation, ``mean_effect``
    -> ``stochastic``:

    =========  ==========================  ==========================
    quantity   absolute (0.4-3.0 Gy)       D_t-rel. (0.05-0.85 Dt)
    =========  ==========================  ==========================
    RMS beta   0.00747 -> 0.00581 (better) 0.00210 -> 0.00217 (worse)
    RMS alpha  0.0794 -> 0.0928 (worse)    0.1130 -> 0.0613 (better)
    =========  ==========================  ==========================

    Note both ``beta`` RMS fall ~3x on the window change alone. The 3% gap between
    the conventions is **not** claimed as significant -- it is asserted only as
    "neither convention wins on beta", which is the window-robust statement. The
    ``alpha`` gap (46%) is large enough to assert a direction.

    The load-bearing conclusion is what the name says: the stochastic convention
    does not rescue the engine. Which half of the tradeoff improves depends on the
    fit window, so neither engine nor convention can be chosen on RMS.

    The intertrack enhancement peaks at or below LET 30 for both conventions (on the
    ``D_t``-relative window), i.e. at the paper's own LET 24-25 within this coarse
    grid's resolution; on the absolute window it peaks at LET 40-45. Below LET 30
    the curve is nearly flat (stochastic: 0.0487 / 0.0489 / 0.0485 at LET 10/20/30),
    so the argmax there is close to degenerate and only the ``<= 30`` bound is
    asserted.

    Not measured under the ``D_t``-relative window: the comparison against
    ``local_dose`` + ``local_dose_overkill`` (0.00339 / 0.0491 under the absolute
    one). Do not quote those numbers against the ones above. See
    docs/references/LEM_IV_Friedrich2012.md.
    """
    beta_curve = f12.get_curve("fig4_hypothetical_vs_let", series="full_simulation", variant="beta")
    alpha_curve = f12.get_curve(
        "fig4_hypothetical_vs_let", series="full_simulation", variant="alpha"
    )
    lets = (10.0, 20.0, 30.0, 45.0, 70.0, 100.0, 130.0)
    doses = LEM4(photon=FIG4, cell=FIG4_CELL).fit_doses_Gy(_DISCRETE_DOSE_FRACTIONS)
    out = {}
    for tag, survival in (("mean", "mean_effect"), ("stoch", "stochastic")):
        lem = LEM4(
            photon=FIG4,
            cell=FIG4_CELL,
            beta_model="discrete_dsb",
            discrete_survival=survival,  # type: ignore[arg-type]
        )
        fits = [
            _localdose.lq_coefficients(
                doses, lem.discrete_dsb_lethal_events(_carbon_for_let(let), doses)
            )
            for let in lets
        ]
        alphas = np.array([a for a, _b in fits])
        betas = np.array([b for _a, b in fits])
        rms_b = float(np.sqrt(np.mean((betas - np.interp(lets, beta_curve.x, beta_curve.y)) ** 2)))
        rms_a = float(
            np.sqrt(np.mean((alphas - np.interp(lets, alpha_curve.x, alpha_curve.y)) ** 2))
        )
        out[tag] = (rms_b, betas, rms_a)

    # Neither convention wins on beta -- they are within a few percent either way,
    # so "stochastic improves beta" is not assertable.
    ratio = out["stoch"][0] / out["mean"][0]
    assert 0.9 < ratio < 1.1, out
    # ...but stochastic is decisively better on alpha.
    assert out["stoch"][2] < 0.75 * out["mean"][2], (out["stoch"][2], out["mean"][2])
    # The enhancement peak sits at or below LET 30 under both conventions --
    # the paper's LET 24-25, within this grid.
    for tag in ("mean", "stoch"):
        assert lets[int(np.argmax(out[tag][1]))] <= 30.0, (tag, out[tag][1])
        # ...and the roll-off is unambiguously under way by LET 45 either way, which
        # is the part that does not depend on a near-degenerate argmax.
        assert out[tag][1][3] < 0.9 * out[tag][1][2], (tag, out[tag][1])


@pytest.mark.slow
def test_discrete_engine_recovers_the_photon_limit() -> None:
    """``beta(LET 10) -> beta_x``: at low LET half the dose is uniform background
    and the DSBs tile the nucleus at random, so the photon response is recovered.

    It lands at 0.0472, i.e. **5.5% low** -- marginally outside the 5% the engine
    was specified to hit, and recorded here rather than tuned away. The deficit is
    physical, not numerical: it is converged to 0.3% over a 6x range of the
    transverse grid step, and it is a property of this engine's *track population*
    (tracks restricted to the nucleus disk plus a uniform background) rather than
    of the continuum engine's infinite track plane. Against that continuum engine
    the same quantity is 3-6% low at LET 10-30 and only ~1% low at LET 55-100,
    tracking ``r_max / R_eff``.

    No single functional form was found to fit the deficit -- it is *not*
    proportional to the background fraction ``1 - f_ret`` (the ratio scatters over
    2.7-12.8%), so it is left as a measured residual rather than an explained one.
    """
    lem = LEM4(photon=FIG4, cell=FIG4_CELL, beta_model="discrete_dsb")
    assert lem.discrete_dsb_beta(_carbon_for_let(10.0)) == pytest.approx(FIG4.beta_x, rel=0.06)


@pytest.mark.slow
def test_discrete_engine_uses_no_domain_extension() -> None:
    """The escaped dose is a **uniform** background, never resolved near-miss tracks.

    The ``"intertrack_mc"`` engine extends the domain to ``R_sim = R_eff +
    r_max`` and draws ``(D/D_1)(R_sim/R_eff)^2`` tracks -- a 3400x multiplier at
    LET 20 whose near-uniform penumbra clusters like a photon field and pins
    ``beta ~ beta_x`` at every LET. This engine draws exactly ``D/D_1`` tracks,
    all centred **inside** the nucleus.

    Turning the background off instead (all dose in resolved tracks) is the
    opposite failure: it destroys the low-LET photon limit. Both are asserted so
    neither pathology can return unnoticed.
    """
    lem = LEM4(photon=FIG4, cell=FIG4_CELL, beta_model="discrete_dsb")
    ion = _carbon_for_let(20.0)
    setup = lem._discrete_setup(ion)
    r_eff = FIG4_CELL.effective_nucleus_radius_um()
    assert setup.mean_tracks_per_Gy == pytest.approx(1.0 / lem.mean_traversal_dose_Gy(ion))
    assert setup.track_disk_radius_um == pytest.approx(r_eff)
    assert 0.0 < setup.retained_dose_fraction < 1.0

    doses = lem.fit_doses_Gy(_DISCRETE_DOSE_FRACTIONS)
    no_background = _localdose.lq_coefficients(
        doses, lem._discrete_events(setup, doses, convention="per_realisation", retained=1.0)
    )[1]
    assert no_background < 0.6 * FIG4.beta_x, no_background


@pytest.mark.slow
def test_discrete_cluster_index_slope_stays_at_the_photon_rate() -> None:
    """``dC_I/dD = kappa/2`` at every LET -- the no-go, measured independently.

    The intertrack enhancement *is* the dose-slope of the cluster index. Dose
    conservation (``<m> = kappa D``) pins that slope at the photon rate whatever
    the track population, so the enhancement cannot roll off with LET. The shipped
    continuum engine measured this; the discrete engine reproduces it by a
    completely different computational route (FFT disk-convolution, conditioning
    on track number), which is what makes it a real cross-check.
    """
    lem = LEM4(photon=FIG4, cell=FIG4_CELL, beta_model="discrete_dsb")
    doses = lem.fit_doses_Gy(_DISCRETE_DOSE_FRACTIONS)
    for let in (20.0, 55.0, 100.0):
        setup = lem._discrete_setup(_carbon_for_let(let))
        cluster = lem._discrete_cluster_index(setup, doses)
        slope = float(np.polyfit(doses, cluster, 1)[0])
        assert slope == pytest.approx(0.5 * DSB_PER_GY_PER_LOOP, rel=0.10), (let, slope)


@pytest.mark.slow
@pytest.mark.golden
def test_golden_lem4_discrete_dsb_against_friedrich2012_fig4() -> None:
    """The discrete engine against the digitized Fig. 4 full simulation.

    Reference: Friedrich, Scholz, Elsasser, Durante & Scholz 2012, Int. J. Radiat.
    Biol. 88(1-2):103-107, doi:10.3109/09553002.2011.611213, Fig. 4 (p. 107),
    hypothetical cell (alpha_g = 0.25, beta_g = 0.05, D_t = 9.5 Gy, R_n = 5 um) --
    digitized in :mod:`pyradbio.benchmark.friedrich2012`.

    This engine was built to beat the shipped ``"local_dose"`` convolution on the
    high-LET roll-off. **It does not**, for the theorem reason pinned above, and it
    is very slightly worse on both metrics (RMS beta 0.0069 vs 0.0063, RMS alpha
    0.12 vs 0.08 on this LET grid). The tolerances below are therefore set to the
    engine's honest performance, and it ships as a documented cross-check of the
    production engine rather than as a replacement -- see
    docs/references/LEM_IV_Friedrich2012.md.
    """
    lem = LEM4(photon=FIG4, cell=FIG4_CELL, beta_model="discrete_dsb")
    beta_curve = f12.get_curve("fig4_hypothetical_vs_let", series="full_simulation", variant="beta")
    alpha_curve = f12.get_curve(
        "fig4_hypothetical_vs_let", series="full_simulation", variant="alpha"
    )
    lets = (10.0, 20.0, 25.0, 30.0, 40.0, 55.0, 70.0, 85.0, 100.0, 130.0)
    doses = lem.fit_doses_Gy(_DISCRETE_DOSE_FRACTIONS)

    alphas, betas = [], []
    for let in lets:
        a, b = _localdose.lq_coefficients(
            doses, lem.discrete_dsb_lethal_events(_carbon_for_let(let), doses)
        )
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
    assert rms_beta < 0.008, rms_beta
    assert rms_alpha < 0.13, rms_alpha
    # Both Fig. 4 features are still reproduced: a rise above beta_x, then a collapse.
    assert betas[0] == pytest.approx(FIG4.beta_x, rel=0.06), betas[0]
    assert max(betas) > FIG4.beta_x
    assert betas[-1] < 0.1 * FIG4.beta_x, betas[-1]
