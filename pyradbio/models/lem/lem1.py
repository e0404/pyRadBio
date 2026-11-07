"""LEM I: original LEM, r_min ~ 10 nm, no clustering.

Scholz & Kraft 1996 (doi:10.1016/0273-1177(95)00784-C); approximation in
Scholz, Kellerer, Kraft-Weyrather & Kraft 1997 (doi:10.1007/s004110050055,
Radiat. Environ. Biophys. 36:59-66).
"""

from __future__ import annotations

import math
from collections.abc import Iterable
from dataclasses import dataclass
from typing import Literal, get_args

import numpy as np
import numpy.typing as npt
from scipy.special import gammaln, logsumexp

from pyradbio.core.lq import LQParameters
from pyradbio.core.radiation import Ion
from pyradbio.models.base import check_choice

# Re-exported: the Monte Carlo moved to LEMBase (it serves LEM I, II and III
# alike), but these names were part of this module's surface and callers should
# not have to care which file they now live in.
from pyradbio.models.lem.base import (
    _TWO_TRACK_ANGULAR_NODES,
    _TWO_TRACK_IMPACT_NODES,
    _TWO_TRACK_RADIAL_NODES,
    _TWO_TRACK_SEPARATION_NODES,
    SHARED_BETA_MODELS,
    LEMBase,
)
from pyradbio.models.lem.engine import Engine
from pyradbio.track.base import RadialDoseModel
from pyradbio.track.scholz_kraft import ScholzKraftRadialDose

LEM1BetaModel = Literal[
    "slope_matching",
    "rapid_lem",
    "scholz1997_stochastic",
    "scholz1997_two_track",
    "scholz1997_montecarlo",
]

# Execution engine for the stochastic-LEM Monte Carlo. ``numpy`` is the
# deterministic (seed-reproducible) reference loop; ``numba`` is the compiled
# ``prange`` kernel (fastest single machine, per-thread RNG so not bit-exact);
# ``auto`` picks ``numba`` when available, else ``numpy``.
#
# There is no sampler choice any more. Field points are drawn by Kellerer
# weighted sampling (Kellerer 1985 sec. VI.A.2), the method Scholz et al. 1997
# cite for their own simulations. The uniform hybrid estimator this class shipped
# with -- choose a track uniformly, draw log-radially over the whole
# ``[r_min, r_max]`` shell -- was removed: it is the scheme Kellerer calls "too
# wasteful to be practicable", it cost ``O(n_bar^2)``, and above ``n_bar ~ 1400``
# it thinned to under one sample per track and returned an unphysical
# ``beta > 2 beta_x`` without any error bar noticing. The ``multiprocess`` engine
# went with it: it existed to parallelise that NumPy loop, and the weighted path
# is compiled instead.

# Quadrature resolution for the two-track overlap beta (Scholz 1997 "exact"
# beta). Defaults reproduce the photon beta_x low-LET limit to <2% and the paper
# Fig. 3 exact curve to ~10-15%; tests override them lower for speed.

# Defaults for the full stochastic-LEM Monte Carlo (Scholz & Kraft 1994/1996).
# Weighted sampling needs no scaling with n_bar: its precision is flat in the
# track count (relative SD 0.48/0.55/0.81% over a 36x range), so a fixed budget
# is right where the uniform sampler needed one proportional to n_bar.

# The traversal-conditioned estimator below still uses the hybrid uniform/near-
# track mixture: it scores every traversal-count prefix from one sweep, so its
# sample points cannot be weighted by a dose field that depends on which prefix
# is being scored. Its n_bar stays far below the thinning regime (17-569 across
# the 1996 grid), which is why it needed no rework when the per-dose path did.

# Traversal-conditioned (Rao-Blackwellised) MC. ``g_k`` is estimated once per
# realization for *every* traversal-count prefix, so one sweep yields the whole
# survival curve; batches give the standard error without a second stream.
# Parallel blocks for the compiled prefix kernel: each gets its own accumulator
# row, so the reduction is race-free and independent of thread scheduling.
# (field points x tracks) block size. Caps memory on the light-ion entrance
# points, where the traversal grid runs to ~14,000: too small a block leaves only
# a handful of field points per iteration and the sweep becomes Python-call
# bound rather than throughput bound.
# Nodes used to *test* whether a track model has a closed-form radial dose, and
# the relative spread allowed before the answer is "no". Tight on purpose: this
# decides whether a faster code path may be used, so it must not accept a profile
# that merely looks close.


# Impact-parameter nodes for the weighted sampler's tables. The grid must stay
# UNIFORM -- the compiled kernel indexes it arithmetically -- so its resolution
# near b = 0 is set by b_max = r_eff + r_max, and r_max grows as E^1.7. A fixed
# node count therefore silently coarsens with energy: 512 nodes give 194 points
# across the nucleus at carbon 20 MeV/u but 1.9 at carbon 400, where d_bar(b) is
# largest and least resolved. That is not a rounding error -- it broke the table's
# normalisation outright, `integral d_bar dPhi` falling to 0.85 D and dragging the
# MC alpha 20% below the exact cross-section.
#
# So the count is chosen from the *near-field spacing* instead, and 512 is only a
# floor: `_KELLERER_B_PER_R_EFF` points across the nucleus, whatever b_max is.
#
# 40 rather than 20, because `d_bar(b)` has a kink at b = r_eff (where the track
# axis crosses the nuclear edge) and the trapezoid straddles it differently
# depending on whether a node lands near it. Near the floor that shows up as
# *oscillation*, not convergence -- carbon 100 MeV/u gives 0.9994 at 512 nodes but
# 1.0040 at 524, a 0.46% swing for a 2% change in node count, which then drove a
# 1.5% shift in the MC alpha. Measured normalisation (`integral d_bar dPhi / D`),
# converged value 1.0014 in every column:
#
#     points/r_eff:      10        20        40        80
#     carbon 100     0.9994    1.0040    1.0016    1.0014
#     carbon 200     1.0059    1.0022    1.0015    1.0014
#     carbon 400     1.0015    1.0015    1.0014       cap
#     proton 200     1.0059    1.0022    1.0015    1.0014
#
# `_KELLERER_B_MAX_NODES` caps the 2-D CDF tables, which cost
# `n_b * _KELLERER_U_NODES * 8` bytes each -- carbon 400 needs ~10 600 nodes for
# ~350 MB across the tables, and the cap refuses rather than quietly thrashing
# beyond that.
# Cap on the (sample points x tracks) block; n reaches 1e5-1e6 at high energy.


@dataclass(frozen=True, slots=True)
class SpecificEnergySurvival:
    r"""Survival conditioned on nuclear **specific energy** ``z``, not on dose.

    The observable behind Scholz et al. 1997 Fig. 4's left-hand panels. ``z`` is
    a per-cell stochastic quantity -- the energy the nucleus actually received,
    divided by its mass -- so ``S(z)`` is not the dose-domain ``S(D)`` and is not
    obtainable from :class:`TraversalConditionedSurvival` either: at a *fixed*
    traversal count ``z`` still varies with the impact parameters.

    ``z_Gy`` are the bin-mean specific energies and ``log_kill`` the
    corresponding ``-ln <S>``, with ``log_kill_err`` its standard error.
    ``counts`` is how many realizations landed in each bin.

    The error matters more here than in most binned estimates. ``-ln <S>`` is a
    log of a mean, and deep in the curve ``S`` spans orders of magnitude within a
    single ``z`` bin, so the mean is carried by its few largest members and the
    estimator is heavy-tailed. Read ``log_kill_err`` before believing structure in
    the high-``z`` end.
    """

    z_Gy: npt.NDArray[np.float64]
    log_kill: npt.NDArray[np.float64]
    log_kill_err: npt.NDArray[np.float64]
    counts: npt.NDArray[np.int64]

    def slope_Gy_inv(self) -> npt.NDArray[np.float64]:
        """``d[-ln S]/dz`` by central differences on the binned curve.

        The quantity Fig. 4 plots. The fast path's answer is the straight line
        ``alpha_z + 2 beta_z z`` (:meth:`LEM1.scholz1997_specific_energy_slope`);
        this is what the spatially explicit model actually does.

        Differencing amplifies noise, so prefer :meth:`straightness` for the
        question "is this a straight line" -- it asks the equivalent question of
        ``log_kill`` directly and never differentiates.
        """
        return np.asarray(np.gradient(self.log_kill, self.z_Gy), dtype=np.float64)

    def straightness(self) -> tuple[float, float]:
        r"""Is the slope a straight line? Returns ``(cubic / sigma, chi2 per dof)``.

        A straight ``d[-ln S]/dz`` means a *quadratic* ``-ln S(z)``, so this fits
        ``-ln S`` with a cubic weighted by ``log_kill_err`` and reports how many
        standard errors the cubic coefficient sits from zero, plus the
        goodness-of-fit of the quadratic that straightness would imply.

        Testing it this way avoids differentiating a binned Monte-Carlo estimate,
        which would fold neighbouring bins' noise into every point and make a
        genuinely straight line look ragged.
        """
        n = self.z_Gy.size
        if n < 5:
            raise ValueError(f"need >= 5 bins to fit a cubic, have {n}")
        weights = 1.0 / np.maximum(self.log_kill_err, 1.0e-12)
        design = np.vander(self.z_Gy, 4)
        wd = design * weights[:, None]
        coeffs, *_ = np.linalg.lstsq(wd, self.log_kill * weights, rcond=None)
        covariance = np.linalg.inv(wd.T @ wd)
        cubic_sigma = float(coeffs[0] / math.sqrt(covariance[0, 0]))

        quad = np.vander(self.z_Gy, 3) * weights[:, None]
        q_coeffs, *_ = np.linalg.lstsq(quad, self.log_kill * weights, rcond=None)
        residual = (np.vander(self.z_Gy, 3) @ q_coeffs - self.log_kill) / np.maximum(
            self.log_kill_err, 1.0e-12
        )
        return cubic_sigma, float(np.sum(residual**2) / max(n - 3, 1))


@dataclass(frozen=True, slots=True)
class LEM1(LEMBase):
    """Original LEM: ``r_min ~ 10 nm`` core, no clustering, no radical diffusion.

    Uses the shouldered->exponential photon reference directly (no high-dose
    cluster enhancement, which arrives in LEM II) and the effective nuclear
    cross-section from :class:`~pyradbio.core.cell.CellTarget`.
    """

    beta_model: LEM1BetaModel = "slope_matching"

    def __post_init__(self) -> None:
        check_choice("beta_model", self.beta_model, get_args(LEM1BetaModel))
        LEMBase.__post_init__(self)

    def _radial_dose_model(self) -> RadialDoseModel:
        return ScholzKraftRadialDose.lem1(stopping_power=self.stopping_power)

    def _beta_for_ion(self, ion: Ion, alpha_ion: float) -> float:
        if self.beta_model in SHARED_BETA_MODELS:
            return self._beta_from_shared_model(self.beta_model, ion, alpha_ion)
        if self.beta_model == "scholz1997_montecarlo":
            return self.scholz1997_montecarlo_beta(ion)
        return self._scholz1997_stochastic_beta(ion)

    def traversal_energy_fraction(self, ion: Ion) -> float:
        r"""Share of one traversal's energy that lands inside the nucleus.

        ``f = integral_0^r_top D(r) 2 pi r dr / integral_0^r_max D(r) 2 pi r dr``
        with ``r_top = min(R_eff, r_max)``. Both integrals run over the *same*
        profile, so ``f`` is exactly 1 whenever the track fits inside the nucleus
        (``r_max <= R_eff``, i.e. ``E <= ~15 MeV/u`` for LEM I's
        ``r_max = 0.05 E^1.7``) rather than 1 plus a quadrature error.

        This is the bookkeeping gap between :meth:`single_traversal_log_kill`,
        which integrates the local dose over the *nucleus*, and
        :meth:`mean_traversal_dose_Gy`, which converts the *whole* LET into a mean
        nuclear dose. Measured for LEM I: 0.93 at 20 MeV/u, 0.68 at 100, 0.55 at
        400 -- so at 400 MeV/u nearly half of the traversal's energy is deposited
        outside the nucleus it is being charged to.
        """
        r, dose, _r_lo, r_max = self._two_track_dose_profile(ion, _TWO_TRACK_RADIAL_NODES)
        r_top = min(self.cell.effective_nucleus_radius_um(), r_max)
        if r_top >= r_max:
            return 1.0
        weight = dose * r
        total = float(np.trapezoid(weight, r))
        if total <= 0.0:
            return 1.0
        inside_r = np.concatenate([r[r < r_top], [r_top]])
        inside_w = np.concatenate([weight[r < r_top], [float(np.interp(r_top, r, weight))]])
        return float(np.clip(np.trapezoid(inside_w, inside_r) / total, 0.0, 1.0))

    def scholz1997_central_initial_slope(self, ion: Ion) -> float:
        r"""Initial slope of the Scholz 1997 Fig. 1 *approximation* branch.

        The central-traversal shortcut ``(1 - S_1)/D_1`` (Scholz 1997 eqs. (5)-(6),
        the same algebra as Krämer & Scholz 2006 eq. 9) **plus the dose delivered
        by tracks that miss the nucleus**::

            alpha = (1 - S_1)/D_1 + alpha_x * (1 - f)

        with ``f`` from :meth:`traversal_energy_fraction`.

        **Why the second term is not a fudge.** The first term's bookkeeping is
        self-consistent -- ``n = Phi A_eff`` traversals, each killing ``N_1``, over
        a macroscopic ``D_1 = LET/(rho A_eff)`` -- but it scores *only* tracks that
        hit the nucleus, while ``D_1`` charges the nucleus for the whole LET. Once
        ``r_max > R_eff`` those stop being the same traversal: the halo of
        non-traversing tracks carries the remaining ``(1 - f)`` of the dose and is
        simply dropped. In the low-local-dose limit that halo kills at the photon
        rate, so restoring it adds exactly ``alpha_x (1 - f)``. Without it the
        approximation converges to ``alpha_x f`` and so falls *below* the photon
        slope at high energy -- 0.113 against ``alpha_x = 0.18`` for protons at
        400 MeV/u, which no dilute track field can do.

        ``f = 1`` for ``r_max <= R_eff``, so this is identical to the bare
        central-traversal form everywhere below ~15 MeV/u, including the whole
        high-LET overkill regime where the ``1 - exp`` saturation legitimately
        drives ``alpha`` under ``alpha_x`` (thindown).

        **Validated against the paper's own approximation curve**, which the bare
        form does not reproduce: over Fig. 1's digitized approximation branch the
        ratio to the paper goes from 0.66-0.75 (bare, protons above 50 MeV/u) to
        0.95-1.00, and from 0.80-0.90 to 0.98-1.01 for carbon. The residual few
        percent is the expected sign -- ``alpha_x (1 - f)`` is a lower bound, since
        tracks passing just outside the nucleus deposit enough local dose for the
        ``beta_x`` term to matter.

        Note this is therefore **not** ``lethal_central_saturating_initial_slope``
        (KS2006 eq. 9) above ~15 MeV/u; that method is the rapidLEM formula and
        keeps its documented undershoot. Reference: Scholz et al. 1997,
        doi:10.1007/s004110050055.
        """
        bare = self.lethal_central_saturating_initial_slope(ion)
        return bare + self.photon.alpha_x * (1.0 - self.traversal_energy_fraction(ion))

    def scholz1997_specific_energy_lq(self, ion: Ion) -> LQParameters:
        """LQ parameters in nuclear specific energy ``z`` for the fast path.

        Scholz et al. 1997 define ``-ln S(z) = alpha_z z + beta_z z^2`` (eq. 9),
        estimate ``alpha_z = N_1 / Delta z`` from one central traversal (eq. 10),
        and approximate ``beta_z = (s_max - alpha_z) / (2 D_t)`` (eq. 11).
        The returned units are Gy^-1 and Gy^-2 with respect to ``z`` [Gy].
        """
        delta_z = self.mean_traversal_dose_Gy(ion)
        n_1 = self.single_traversal_log_kill(ion)
        return self._scholz1997_specific_energy_lq_from(delta_z=delta_z, n_1=n_1)

    def scholz1997_specific_energy_slope(self, ion: Ion, z_Gy: float) -> float:
        """Slope ``d[-ln S(z)]/dz = alpha_z + 2 beta_z z`` for Fig. 4 overlays."""
        if z_Gy < 0.0:
            raise ValueError(f"z_Gy must be >= 0, got {z_Gy!r}")
        z_lq = self.scholz1997_specific_energy_lq(ion)
        return z_lq.alpha + 2.0 * z_lq.beta * z_Gy

    def scholz1997_approximate_log_kill(self, ion: Ion, dose_Gy: float) -> float:
        """Fast-path dose log-kill from Poisson-distributed traversals.

        For a monoenergetic ion, the number of traversals is Poisson distributed
        with mean ``D / Delta z``. The fast simulation described after Scholz
        1997 eq. (11) increments ``-ln S`` once per traversal using the current
        z-domain slope. For ``n`` traversals this gives

        ``N_n = n*N_1 + beta_z*Delta z^2*n*(n-1)``

        (the first traversal contributes exactly ``N_1``), and this computes

        ``S(D) = sum_n Poisson(n; D/Delta z) * exp(-N_n)``

        and returns ``-ln S(D)``. The summation is truncated at a wide window
        around the Poisson mean and evaluated in log space so high-energy ions
        with very small ``Delta z`` remain stable.
        """
        return self.scholz1997_approximate_log_kills(ion, (dose_Gy,))[0]

    def scholz1997_approximate_log_kills(
        self, ion: Ion, doses_Gy: Iterable[float]
    ) -> tuple[float, ...]:
        """Bulk version of :meth:`scholz1997_approximate_log_kill` for one ion.

        The central-traversal quantities for ``ion`` are evaluated once and then
        reused for every dose. This keeps example/benchmark overlays cheap while
        preserving the scalar method's numerical path.

        Carries the same missed-track halo as
        :meth:`scholz1997_central_initial_slope` -- the Poisson sum runs over
        *nuclear traversals*, so like the initial slope it drops the tracks that
        pass outside while ``Delta z`` still charges the nucleus for the whole
        LET. Adding ``alpha_x (1 - f) D`` keeps this path's low-dose slope equal
        to the initial slope it is supposed to start from at every energy, rather
        than only below ~15 MeV/u where ``f = 1``.
        """
        delta_z = self.mean_traversal_dose_Gy(ion)
        n_1 = self.single_traversal_log_kill(ion)
        beta_z = self._scholz1997_specific_energy_lq_from(delta_z=delta_z, n_1=n_1).beta
        halo = self.photon.alpha_x * (1.0 - self.traversal_energy_fraction(ion))
        return tuple(
            self._scholz1997_approximate_log_kill_from_parameters(
                float(dose_Gy),
                delta_z=delta_z,
                n_1=n_1,
                beta_z=beta_z,
            )
            + halo * float(dose_Gy)
            for dose_Gy in doses_Gy
        )

    @staticmethod
    def _scholz1997_approximate_log_kill_from_parameters(
        dose_Gy: float, *, delta_z: float, n_1: float, beta_z: float
    ) -> float:
        if dose_Gy < 0.0:
            raise ValueError(f"dose_Gy must be >= 0, got {dose_Gy!r}")
        if dose_Gy == 0.0:
            return 0.0

        mean = dose_Gy / delta_z

        width = max(12.0 * math.sqrt(mean), 20.0)
        n_min = max(0, math.floor(mean - width))
        n_max = math.ceil(mean + width)
        n = np.arange(n_min, n_max + 1, dtype=np.float64)
        log_kill_n = n * n_1 + beta_z * delta_z * delta_z * n * (n - 1.0)

        log_p = -mean + n * math.log(mean) - gammaln(n + 1.0)
        log_survival_terms = log_p - log_kill_n
        return max(float(-logsumexp(log_survival_terms)), 0.0)

    def scholz1997_approximate_survival(self, ion: Ion, dose_Gy: float) -> float:
        """Surviving fraction for the Scholz 1997 fast approximation."""
        return math.exp(-self.scholz1997_approximate_log_kill(ion, dose_Gy))

    def _scholz1997_stochastic_beta(self, ion: Ion) -> float:
        r"""Dose-domain beta from the Scholz 1997 specific-energy fast simulation.

        Scholz et al. 1997 define a quadratic survival relation in nuclear
        specific energy ``z`` (eq. 9), estimate ``alpha_z`` from one central
        traversal (eq. 10), and approximate ``beta_z = (s_max - alpha_z)/(2 D_t)``
        (eq. 11). For a monoenergetic beam, traversals are Poisson distributed
        with mean ``D / Delta z``; expanding the resulting arithmetic-mean
        population survival to second order in dose gives the returned dose-domain
        ``beta = beta_z * exp(-2 N_1)`` (``N_1`` the single-central-traversal
        log-kill), keeping the public value in the usual ``-ln S(D) = alpha D +
        beta D^2`` convention.

        This is a *fast, crude* approximation. Because it collapses the track to a
        nuclear-average specific energy, it does not resolve the outer-track
        overlap that generates the shoulder (Scholz & Kraft 1994) and it mis-states
        the low-energy carbon curvature. The full stochastic reference
        :meth:`scholz1997_montecarlo_beta` confirms that the deterministic
        :meth:`scholz1997_two_track_beta` matches the true dose-domain curvature
        (both give a small low-energy carbon beta, an order of magnitude below the
        z-domain curvature ``beta_z``), whereas this z-domain estimate should be
        treated as indicative only. Prefer ``beta_model="scholz1997_two_track"``
        for quantitative work.
        """
        delta_z = self.mean_traversal_dose_Gy(ion)
        n_1 = self.single_traversal_log_kill(ion)
        z_lq = self._scholz1997_specific_energy_lq_from(delta_z=delta_z, n_1=n_1)

        q_1 = math.exp(-n_1)
        q_2 = math.exp(-(2.0 * n_1 + 2.0 * z_lq.beta * delta_z * delta_z))
        survival_d1 = (q_1 - 1.0) / delta_z
        survival_d2 = (0.5 - q_1 + 0.5 * q_2) / (delta_z * delta_z)
        beta = -survival_d2 + 0.5 * survival_d1 * survival_d1
        return max(beta, 0.0)

    # ---- two-track overlap beta (Scholz 1997 "exact" beta) --------------

    def scholz1997_explicit_lq(
        self,
        ion: Ion,
        *,
        radial_nodes: int = _TWO_TRACK_RADIAL_NODES,
        separation_nodes: int = _TWO_TRACK_SEPARATION_NODES,
        angular_nodes: int = _TWO_TRACK_ANGULAR_NODES,
        impact_nodes: int = _TWO_TRACK_IMPACT_NODES,
    ) -> LQParameters:
        """Explicit-model dose LQ coefficients for Scholz 1997 Figs. 2, 3, and 5.

        The paper's explicit stochastic calculation is linear-quadratic over its
        displayed dose range (eq. 8). Its first two dose coefficients are the
        full impact-parameter inactivation-cross-section slope and the two-track
        overlap curvature. Bundling them here keeps that explicit path separate
        from the central-traversal/specific-energy approximation used in the
        paper's right-hand and approximation panels.

        The node parameters are forwarded to :meth:`scholz1997_two_track_beta`;
        defaults are the quantitative reference resolution, while examples may
        request a documented reduced quadrature for plotting.

        Reference: Scholz et al. 1997 eqs. (2), (6), and (8),
        doi:10.1007/s004110050055.
        """
        return LQParameters(
            alpha=self.initial_slope(ion),
            beta=self.scholz1997_two_track_beta(
                ion,
                radial_nodes=radial_nodes,
                separation_nodes=separation_nodes,
                angular_nodes=angular_nodes,
                impact_nodes=impact_nodes,
            ),
        )

    # ---- full stochastic-LEM Monte Carlo (Scholz & Kraft 1994/1996) -----

    def scholz1997_specific_energy_survival(
        self,
        ion: Ion,
        *,
        z_max_Gy: float,
        n_bins: int = 12,
        n_poisson_means: int = 120,
        realizations_per_mean: int = 2500,
        n_field_points: int = 400,
        b_cut_um: float | None = None,
        min_bin_count: int = 60,
        seed: int = 0,
        engine: Engine = "auto",
    ) -> SpecificEnergySurvival:
        r"""Exact ``S(z)`` from the spatially explicit model (Scholz 1997 Fig. 4).

        Reconstructs the paper's *exact* z-domain path, against which
        :meth:`scholz1997_specific_energy_slope` is the fast approximation. For
        each realization both quantities come out of the same track
        configuration: the nuclear specific energy ``z = sum_i d_bar(b_i)``
        straight off :meth:`nuclear_dose_table`, and the survival
        ``exp(-N_leth)`` from :meth:`kellerer_field_integral`. Binning by ``z``
        then gives ``S(z) = E[exp(-N_leth) | z]``.

        Note what this is *not*. It is not ``S(D)``, which averages over the
        specific energies a given dose produces, and it is not recoverable from
        :meth:`scholz1997_traversal_conditioned_survival`: that conditions on the
        traversal count, and ``z`` at fixed count still varies with the impact
        parameters. Conditioning on ``z`` is the only way to reach this panel,
        which is why it needs its own estimator.

        Realizations are pooled across ``n_poisson_means`` fluences. That is
        exact, not an approximation: ``S(z)`` is conditioned on ``z``, so it does
        not depend on the fluence that produced a given ``z``, and pooling merely
        fills the whole range -- a single Poisson mean concentrates ``z`` near
        ``lambda * Delta z`` with relative width ``1/sqrt(lambda)`` and would
        leave most of the axis empty.

        **Use many means rather than many realizations per mean.** Each fluence
        deposits a narrow bump in ``z``, so with few means the pooled coverage is
        lumpy: bins fill unevenly, their mean ``z`` drifts off their centres, and
        the within-bin ``z`` spread goes lopsided, which distorts the *shape* the
        estimator exists to measure. Measured at proton 8 MeV/u with the history
        budget held fixed at 48 000, going from 12 means to 120 dropped the
        chi-square per degree of freedom of a quadratic fit from 64 to 7 at no
        extra cost. Statistical error is not the binding constraint here -- it is
        already 0.06-0.9% per bin -- coverage is.

        **This is not quite the paper's own prescription, and the difference was
        measured rather than assumed.** Scholz et al. write that the exact panels
        use Eq. (2) "but with the increments of specific energy approximated in
        terms of LET" -- each traversal of the nucleus contributes a fixed
        LET-derived ``Delta z``, so their ``z`` is a deterministic multiple of the
        nuclear traversal count and carries nothing from tracks that miss. The
        ``z`` here is the real one: ``sum_i d_bar(b_i)`` over every track,
        integrated from the radial profile, halo included.

        Running both scorings on the *same* realizations, they agree in the mean
        to 0.1% -- which is a check on the dose table, since the two routes to the
        nuclear specific energy are independent -- and the resulting slope curves
        differ by only a few percent (carbon 8 MeV/u: 1.137 -> 1.463 against
        1.141 -> 1.509 over 0-30 Gy). So the LET approximation is not what shapes
        those panels, and it is not worth a second code path.

        In particular it is *not* the reason the paper's panels are straight lines:
        this estimator returns near-straight lines too, R^2 = 0.95-0.99 against a
        linear fit wherever ``beta_z > 0``. Straightness is a property of the
        explicit model in this representation, not of either approximation.

        Reference: Scholz et al. 1997 Fig. 4, doi:10.1007/s004110050082.
        """
        if not math.isfinite(z_max_Gy) or z_max_Gy <= 0.0:
            raise ValueError(f"z_max_Gy must be finite and > 0, got {z_max_Gy!r}")
        if n_bins < 2 or n_poisson_means < 1 or realizations_per_mean < 1:
            raise ValueError("n_bins >= 2, n_poisson_means >= 1, realizations_per_mean >= 1")

        r, dose, r_lo, r_max = self._two_track_dose_profile(ion, _TWO_TRACK_RADIAL_NODES)
        analytic = self._analytic_profile(ion)
        table = self.nuclear_dose_table(ion)
        r_eff = self.cell.effective_nucleus_radius_um()
        a_eff = math.pi * r_eff * r_eff
        b_max = r_eff + r_max
        d_1 = self.mean_traversal_dose_Gy(ion)
        rng = np.random.default_rng(seed)
        # Traversals in the b_max disc that produce a mean specific energy z.
        # This is the same relation the dose-domain MC uses for n_bar -- z
        # averages to the macroscopic dose -- and NOT `z / d_1`: `d_1` is the
        # mean nuclear dose per track *through the nucleus*, while tracks out to
        # b_max also deposit. The two differ by pi b_max^2 / A_eff, which is ~1
        # only where r_max << r_eff (carbon at 1 MeV/u), so using z / d_1 puts
        # every realization in the lowest bin as soon as the track gets wide.
        tracks_per_gray = math.pi * b_max * b_max / (a_eff * d_1)

        # Span a little past z_max so the top bin is not fed only by its own
        # lower edge, which would bias the last slope point low.
        z_targets = np.linspace(z_max_Gy / n_bins, 1.15 * z_max_Gy, n_poisson_means)
        zs, ss = self._z_samples(
            np.asarray(z_targets * tracks_per_gray, dtype=np.float64),
            b_max,
            table,
            r,
            dose,
            r_lo,
            r_max,
            n_field_points,
            realizations_per_mean,
            rng,
            analytic,
            b_cut_um,
            engine,
        )

        edges = np.linspace(0.0, z_max_Gy, n_bins + 1)
        which = np.digitize(zs, edges) - 1

        z_mean: list[float] = []
        log_kill: list[float] = []
        log_kill_err: list[float] = []
        z_var: list[float] = []
        counts: list[int] = []
        for b in range(n_bins):
            inside = which == b
            hits = int(inside.sum())
            if hits < min_bin_count:
                continue
            mean_survival = float(ss[inside].mean())
            if mean_survival <= 0.0:
                # Every realization in this bin was killed outright; -ln S is not
                # resolvable from this sample, so drop rather than clip.
                continue
            z_mean.append(float(zs[inside].mean()))
            log_kill.append(-math.log(mean_survival))
            # delta method on -ln(mean S): sd(S) / (mean(S) sqrt(n)).
            log_kill_err.append(float(ss[inside].std(ddof=1) / (mean_survival * math.sqrt(hits))))
            z_var.append(float(zs[inside].var()))
            counts.append(hits)
        if len(z_mean) < 2:
            raise ValueError(
                f"only {len(z_mean)} z bins reached min_bin_count={min_bin_count}; "
                "raise realizations_per_mean or lower n_bins"
            )
        z_arr = np.asarray(z_mean, dtype=np.float64)
        lk_arr = np.asarray(log_kill, dtype=np.float64)
        var_arr = np.asarray(z_var, dtype=np.float64)

        # De-bias the bin average. A bin reports `-ln <S>`, but the curve wants
        # `-ln S` at the bin's mean z, and those differ because S is convex in z:
        # writing u = -ln S and expanding about z_mean,
        #     -ln <S> ~ u(z_mean) - (u'^2 - u'') var(z)/2
        # so the bin average sits LOW by that amount. It is not a small effect at
        # useful bin widths -- u' ~ 1.4 Gy^-1 and a 2.5 Gy bin give var ~ 0.52 and
        # a bias near 0.5 in -ln S, against statistical errors of ~0.03. Left in,
        # it masquerades as curvature, which is exactly what this estimator is
        # asked to measure. Same Jensen correction in spirit as
        # `_mc_survival_from_sums`, one order up.
        if lk_arr.size >= 3:
            slope = np.gradient(lk_arr, z_arr)
            curvature = np.gradient(slope, z_arr)
            lk_arr = lk_arr + 0.5 * (slope**2 - curvature) * var_arr

        return SpecificEnergySurvival(
            z_Gy=z_arr,
            log_kill=lk_arr,
            log_kill_err=np.asarray(log_kill_err, dtype=np.float64),
            counts=np.asarray(counts, dtype=np.int64),
        )
