"""Shared LEM machinery.

The Local Effect Model maps the photon dose-response onto the inhomogeneous
*local* dose an ion track deposits inside the cell nucleus. Survival follows
from integrating the photon lethal-event density over the nuclear cross-section
(Scholz & Kraft 1996 eqs. 4-7; general formulation Elsasser & Scholz 2007).

This base owns the parts every LEM version shares:

* the local-dose accumulation delegated to a ``track`` radial-dose model;
* the single-traversal nuclear integral ``N_1`` of ``-ln S_x(D(r))`` over the
  (effective) nuclear disk;
* the initial slope ``alpha_I`` from the low-dose limit of the amorphous track
  model (Scholz & Kraft 1996 eqs. 4-7): the inactivation cross-section
  ``sigma = integral (1 - exp(-nu(b))) 2 pi b db`` over the track impact
  parameter ``b``, where ``nu(b)`` is the nuclear-averaged log-kill of a track
  offset by ``b``. This includes the *indirect* traversals (ion misses the
  nucleus but its 1/r^2 tail still irradiates it; Scholz & Kraft 1996, Fig. 3
  discussion) that the central-traversal shortcut drops, so ``alpha_I -> alpha_x``
  in the low-LET limit and saturates at ``1/D_1`` (single-traversal overkill)
  at high LET. ``D_1 = 0.16 * LET / (rho * A_eff)`` is the mean dose of one
  traversal (Elsasser & Scholz 2007 eq. 8);
* the ``beta_I`` slope-matching approximation (Kraemer & Scholz 2006); and
* mixed-field LQ mixing (dose-average ``alpha``, dose-average ``sqrt(beta)``).

Versions differ only in the radial-dose model they supply via
`LEMBase._radial_dose_model` (and, from LEM II on, the photon high-dose
extrapolation), so they subclass this with a one-line override.
"""

from __future__ import annotations

import math
import warnings
from abc import abstractmethod
from collections.abc import Callable, Iterable
from dataclasses import dataclass, field
from typing import ClassVar, Literal, get_args

import numpy as np
import numpy.typing as npt
from scipy.integrate import simpson, trapezoid
from scipy.optimize import brentq
from scipy.special import gammaln

from pyradbio.core.cell import CellTarget
from pyradbio.core.constants import LEM_D1_PREFACTOR, WATER_DENSITY_G_CM3
from pyradbio.core.lq import LQParameters, dose_for_survival
from pyradbio.core.photon import PhotonParameters
from pyradbio.core.radiation import Ion
from pyradbio.models.base import BiophysicalModel, ModelCapabilities
from pyradbio.models.lem import _mc_kernel
from pyradbio.models.lem.engine import ENGINES, Engine
from pyradbio.track.base import RadialDoseModel
from pyradbio.track.stopping_power import StoppingPower, default_stopping_power

# Log-spaced radial nodes for the nuclear-disk integral. 512 reproduces an
# adaptive-quadrature reference to <0.01% across the ion-therapy LET range
# (validated in the slow nuclear-integral test).
_NUCLEAR_NODES: int = 512

# Radial nodes for the precomputed log-kill profile -ln S_x(D(s)) (built once per
# ion and reused for every impact parameter), the per-impact shell refinement,
# and the impact-parameter (b) grid of the inactivation-cross-section integral.
_PROFILE_NODES: int = 512
_SHELL_NODES: int = 257
_IMPACT_NEAR_NODES: int = 200
_IMPACT_FAR_NODES: int = 300


def _angular_fraction_inside(
    s: npt.NDArray[np.float64], b: float, r_eff: float
) -> npt.NDArray[np.float64]:
    """Angular measure (0..2*pi) of the circle of radius ``s`` about a track at
    distance ``b`` from the nucleus centre that lies inside the nuclear disk of
    radius ``r_eff``.

    By the law of cosines a point on that circle is inside the disk when the
    chord half-angle ``psi`` satisfies ``cos psi = (s^2 + b^2 - r_eff^2)/(2 s b)``;
    clipping the cosine to ``[-1, 1]`` folds in the fully-inside (2*pi) and
    no-overlap (0) cases. For ``b = 0`` the circle is concentric: fully inside
    while ``s <= r_eff``.

    Both degenerate radii need their own branch, because ``2 s b`` vanishes. The
    ``s = 0`` case is reachable: `LEMBase._impact_log_kill` starts its shell
    at ``b - r_eff``, which is exactly ``0`` for a track grazing the nuclear edge,
    and whether the impact grid lands on ``b = r_eff`` is an accident of the node
    count. The circle is then the single point at distance ``b``, so it is inside
    while ``b <= r_eff``. It carries no weight either way (the integrand is
    ``logkill * phi * s``), but returning ``nan`` here poisons the quadrature.
    """
    if b == 0.0:
        return np.where(s <= r_eff, 2.0 * math.pi, 0.0)
    with np.errstate(divide="ignore", invalid="ignore"):
        cos_psi = np.clip((s * s + b * b - r_eff * r_eff) / (2.0 * s * b), -1.0, 1.0)
    degenerate = 2.0 * math.pi if b <= r_eff else 0.0
    return np.where(s > 0.0, 2.0 * np.arccos(cos_psi), degenerate)


_TWO_TRACK_RADIAL_NODES: int = 600  # log-radius nodes for D(r) and the J_full integral
_TWO_TRACK_SEPARATION_NODES: int = 90  # separation nodes for Delta_plane(s)
_TWO_TRACK_ANGULAR_NODES: int = 400  # azimuthal nodes in the pair-overlap integral
_TWO_TRACK_IMPACT_NODES: int = 120  # impact-parameter nodes for the nu_1 suppression
_ANALYTIC_PROBE_NODES: int = 4096
# Empty eta table -> the kernel's LEM I branch. Module-level so every caller
# shares one array rather than allocating per Monte-Carlo point.
_NO_ETA: npt.NDArray[np.float64] = np.empty(0, dtype=np.float64)
# How closely `_kernel_eta_table` must reproduce `_photon_log_kill_array`.
# This guards against the kernel simulating a *different model* -- which would
# differ by tens of percent, not by rounding -- so it must be loose enough to
# admit a legitimately sampled eta and tight enough that nothing structural
# slips through. Tabulated cluster models land at 3e-16 and a densely sampled
# closed form at ~5e-6, both far inside this.
_MC_RESPONSE_RTOL: float = 1e-3
_ANALYTIC_PROBE_RTOL: float = 1.0e-12
# Radical-diffusion closed form (see AnalyticRadialProfile). The tolerance is set
# by the *sampled* profile, not by the expansion: the trapezoid convolution it is
# measured against carries its own ~1e-5 relative noise, so a tighter bound would
# be fitting that noise. Measured, the two-term expansion sits inside 1e-5 from
# ~50 nm outward at sigma = 4 nm, which is where this puts the switch.
_CONV_PROFILE_RTOL: float = 1.0e-4
_CONV_K_LO_FRAC: float = 0.3  # take k from [0.3, 0.9] r_max: correction dead, ...
_CONV_K_HI_FRAC: float = 0.9  # ... cut-off edge excluded
_CONV_EDGE_SIGMA: float = 6.0  # skip the r_max roll-off, ~6 sigma wide in absolute terms
_CONV_FIT_REL_LO: float = 3.0e-4  # fit 2 sigma^2 where the correction is visible
_CONV_FIT_REL_HI: float = 0.1  # ... and the asymptotic series still converges
_CONV_FIT_MIN_NODES: int = 8
_CONV_SWITCH_MARGIN: float = 1.5  # pad the measured switch radius


def _is_flat(values: npt.NDArray[np.float64]) -> bool:
    """True if every entry matches the mean to ``_ANALYTIC_PROBE_RTOL``."""
    mean = float(values.mean())
    if mean == 0.0:
        return bool(np.all(values == 0.0))
    return bool(np.max(np.abs(values / mean - 1.0)) <= _ANALYTIC_PROBE_RTOL)


@dataclass(frozen=True, slots=True)
class AnalyticRadialProfile:
    r"""A radial dose model of the form constant core / ``1/r^2`` / cut-off.

    ```text
    D(r) = core_dose_Gy          r < r_min
         = k_Gy_um2 / r^2        r_min <= r < r_max
         = 0                     r >= r_max
    ```

    Both amorphous track models in the package land here, with *different* cores:
    Scholz & Kraft continue the ``1/r^2`` law into the core
    (``core = k / r_min^2``), while Kiefer-Chatterjee sets an independent core
    level, so the profile is discontinuous at ``r_min``. Keeping the core as a
    free parameter covers both -- and the constructor is never called by name,
    only by `LEMBase._analytic_profile` measuring an actual sampled profile.

    **Radical-diffusion (LEM II/III) profiles are covered too, outside a radius.**
    Convolving ``k/r^2`` with a 2-D Gaussian has no elementary closed form -- it
    is an inverse moment of a noncentral chi-square with 2 dof and
    ``lambda = r^2/sigma^2`` (Marcum Q plus Poisson-weighted incomplete gammas),
    and that series needs ``~r^2/(2 sigma^2)`` terms, so it overflows by
    ``r ~ 150 nm``. The *small-sigma* expansion is elementary instead: in 2-D
    ``grad^2 r^-m = m^2 r^-m-2``, hence ``grad^2n r^-2 = 4^n (n!)^2 r^-2-2n`` and

    ```text
    D_conv(r) = (k/r^2) * sum_n n! x^n,     x = 2 sigma^2 / r^2
    ```

    an asymptotic series with optimal truncation at ``n ~ 1/x``; two terms are
    kept. It is valid on ``r_switch_um <= r <= r_edge_um`` only -- outside that
    band the caller must use the sampled profile. ``two_sigma_sq_um2 = 0`` means
    no convolution, where the correction factor is exactly ``1.0``,
    ``r_switch_um`` is ``0`` and ``r_edge_um`` is ``r_max_um``.

    ``r_edge_um`` exists because the convolution also smooths the ``r_max``
    cut-off, over ~6 ``sigma`` of *absolute* width -- 2.6e-5 of ``r_max`` at
    carbon 100 MeV/u but 1.3e-3 at 10 MeV/u, so it cannot be excluded as a fixed
    fraction. Inside that band the closed form runs ~2x high (it sees a hard cut
    where the true profile is half rolled off), and deferring to the table there
    keeps the two paths in agreement by construction rather than by argument.
    """

    core_dose_Gy: float
    k_Gy_um2: float
    r_min_um: float
    r_max_um: float
    two_sigma_sq_um2: float = 0.0
    r_switch_um: float = 0.0
    r_edge_um: float = math.inf

    @property
    def kernel_r_edge_um(self) -> float:
        """``r_edge_um`` clamped to ``r_max_um``, which is what the kernel wants.

        An unconvolved profile leaves ``r_edge_um`` infinite, and this makes the
        kernel's outer test ``d > r_edge`` unreachable below the ``r > r_max``
        cut it already applies -- so LEM I keeps the arithmetic it always had.
        """
        return min(self.r_edge_um, self.r_max_um)

    def dose_Gy(self, r_um: npt.NDArray[np.float64] | float) -> npt.NDArray[np.float64]:
        """``D(r)``, vectorised. Valid only for ``r >= r_switch_um``."""
        r = np.asarray(r_um, dtype=np.float64)
        inner = np.maximum(r, self.r_min_um)
        tail = self.k_Gy_um2 / (inner * inner)
        if self.two_sigma_sq_um2 > 0.0:
            x = self.two_sigma_sq_um2 / (inner * inner)
            tail = tail * (1.0 + x * (1.0 + 2.0 * x))
        value = np.where(r < self.r_min_um, self.core_dose_Gy, tail)
        # ``>`` not ``>=``, matching the interpolated path's cut-off convention so
        # the two agree on the (measure-zero) boundary as well as off it.
        return np.asarray(np.where(r > self.r_max_um, 0.0, value), dtype=np.float64)

    def valid(self, r_um: npt.NDArray[np.float64] | float) -> npt.NDArray[np.bool_]:
        """Where [`dose_Gy`][] may be trusted: ``r_switch <= r <= r_edge``."""
        r = np.asarray(r_um, dtype=np.float64)
        return np.asarray((r >= self.r_switch_um) & (r <= self.kernel_r_edge_um))


# Which formula maps a track to the ion initial slope ``alpha_I``. All three
# integrate the single-track nuclear log-kill ``nu(b)``, and differ on two axes:
# whether the intra-track *overkill* saturation ``1-exp(-nu)`` is kept, and
# central-traversal vs full impact-parameter geometry.
#   "cross_section"  -- the EXACT low-dose LEM slope (Scholz & Kraft 1996
#                   eqs. 4-7): population survival ``-ln<S> = Phi * sigma`` with
#                   the inactivation cross-section ``sigma = integral (1-exp(-nu(b)))
#                   2 pi b db`` (Campbell's theorem; the ``1-exp(-nu)`` is the
#                   Jensen/overkill correction). Full impact integral, KEEPS
#                   overkill, so it *thins down* at very high LET. This is what the
#                   explicit Scholz 1997 model and the stochastic MC give.
#   "lethal_central" -- TRiP98 / rapidLEM (Kraemer & Scholz 2006): the single
#                   central traversal ``alpha_I = min(N_1/D_1, s_max)``. A fast
#                   treatment-planning approximation that DROPS overkill (caps at
#                   the photon ``s_max`` instead of thinning down).
#   "lethal_full"    -- the full impact-parameter integral of the *unsaturated*
#                   lethal events, ``alpha_I = (1/(A_eff D_1)) integral nu(b) 2 pi
#                   b db`` (``E[N]/D``, no ``1-exp(-nu)``). Same overkill-free
#                   simplification as "lethal_central" but with the indirect
#                   traversals it drops, so it reproduces the whole TRiP98 RBE
#                   curve across tissues. NOT the exact slope (it omits the
#                   overkill term "cross_section" keeps); it matches TRiP because
#                   rapidLEM makes the same omission.
# The two ``lethal_*`` models are the overkill-free family (central vs full);
# ``cross_section`` is the exact one. See docs/references/LEM_I_ScholzKraft.md.
InitialSlopeModel = Literal[
    "cross_section",
    "lethal_central",
    "lethal_central_saturating",
    "lethal_full",
    # LEM IV only: alpha_I read off a *simulated* survival curve rather than taken
    # as its D -> 0 limit. See ``LEM4.full_simulation_initial_slope``.
    "full_simulation",
]

# The ``beta_I`` models every *local-dose* LEM version shares, i.e. every version
# whose lethal-event density is a pointwise function of the local dose. All three
# live on `LEMBase` so LEM I/II/III get them from one implementation:
#   "slope_matching"        -- beta = (s_max - alpha_I)/(2 D_t), applied to the
#                          *saturated* alpha_I (Kraemer & Scholz 2006). Cheap, and
#                          the default -- but because it is
#                          anti-correlated with alpha_I, and alpha_I is NOT monotone
#                          in LET (it thins down under overkill), it revives a
#                          spurious beta at very high LET where the exact answer is 0.
#   "rapid_lem"             -- KS2006 eq. 10: slope-match the *unsaturated* alpha_z,
#                          then scale by f^2, f = (1-e^-N_1)/N_1. Reproduces the
#                          rapidLEM/TRiP and "Survival"-code beta.
#   "scholz1997_two_track"  -- the exact O(phi^2) pair-overlap coefficient. The only
#                          LEM-native one of the three; no slope-matching anywhere.
# LEM IV is deliberately NOT in this family: its effect is a function of a *global*
# DSB-pattern statistic, not of the local dose pointwise, so neither the pair
# expansion nor the central-traversal N_1 these assume is defined for it. See
# `LEM4._beta_for_ion`.
SharedBetaModel = Literal["slope_matching", "rapid_lem", "scholz1997_two_track"]
SHARED_BETA_MODELS: tuple[SharedBetaModel, ...] = get_args(SharedBetaModel)


def _nuclear_arc_half_angle(
    b: npt.NDArray[np.float64], r: npt.NDArray[np.float64], r_eff: float
) -> npt.NDArray[np.float64]:
    r"""Half-angle of the arc at radius ``r`` from a track at ``b`` that lies inside.

    A point at distance ``r`` from the track centre and azimuth ``phi`` measured
    from the track-to-nucleus-centre direction sits at ``b^2 + r^2 - 2 b r cos
    phi`` from the nucleus centre, so it is inside iff ``cos phi >= (b^2 + r^2 -
    R^2) / (2 b r)``. Returns ``pi`` when the whole circle is inside and ``0``
    when none of it is.
    """
    with np.errstate(divide="ignore", invalid="ignore"):
        cos_phi = (b * b + r * r - r_eff * r_eff) / (2.0 * b * r)
    # b -> 0: the track sits at the nucleus centre, so the circle is wholly in
    # or wholly out depending on r alone.
    cos_phi = np.where(b * r <= 0.0, np.where(r <= r_eff, -1.0, 1.0), cos_phi)
    return np.asarray(np.arccos(np.clip(cos_phi, -1.0, 1.0)), dtype=np.float64)


def _delete_one_jackknife(
    batches: npt.NDArray[np.float64],
    observable: Callable[[npt.NDArray[np.float64]], float],
) -> tuple[float, float]:
    """``(value, standard_error)`` for a functional of the pooled batch mean.

    Delete-one jackknife over the leading axis of ``batches`` (one row per
    independent batch). The central value is the observable of the pooled mean;
    each replicate re-evaluates it on all but one batch, so *nonlinear*
    observables -- root-finds, ratios, least-squares fits -- stay
    well-conditioned. A two-sample range would instead be dominated by its own
    sampling noise, which is why one is never used here (see
    docs/references/LEM_I_ScholzKraft.md).

    Replicates raising ``ValueError``/``RuntimeError`` are dropped, since
    deleting a batch can legitimately put a root-find out of range; below two
    survivors the error is ``nan`` rather than a fabricated number.

    Shared by [`TraversalConditionedSurvival.jackknife`][] (LEM I-III
    traversal-conditioned curves) and the LEM IV intertrack Monte-Carlo
    ([`IntertrackSurvival`][pyradbio.models.lem.lem4.IntertrackSurvival]).
    """
    n_batches = int(batches.shape[0])
    if n_batches < 2:
        raise ValueError(f"need >= 2 batches for a jackknife, got {n_batches}")
    value = observable(np.asarray(batches.mean(axis=0), dtype=np.float64))
    total = batches.sum(axis=0)
    replicates = []
    for index in range(n_batches):
        reduced = (total - batches[index]) / (n_batches - 1)
        try:
            replicates.append(observable(reduced))
        except (ValueError, RuntimeError):
            continue
    if len(replicates) < 2:
        return value, math.nan
    sample = np.asarray(replicates, dtype=np.float64)
    scale = (len(sample) - 1) / len(sample)
    return value, float(math.sqrt(scale * float(np.sum((sample - sample.mean()) ** 2))))


@dataclass(frozen=True, slots=True)
class TraversalConditionedSurvival:
    r"""Full stochastic-LEM survival curve, conditioned on the traversal count.

    The quantity ``g_k = E[exp(-N_leth) | exactly k tracks in the b_max disc]``
    does **not** depend on dose: dose enters only through the Poisson mean
    ``lambda(D) = phi(D) pi b_max^2``. One ``g`` vector therefore determines the
    entire survival curve,

        ``S(D) = sum_k Poisson(k; lambda(D)) g_k``,

    which is monotone by construction and analytically differentiable -- so the
    Scholz & Kraft 1996 final slope needs no numerical differentiation of noisy
    samples, and Fig. 4's iso-survival doses are exact root-finds.

    Conditioning is also the variance reduction that matters here. At 1 %
    survival with ``lambda ~ 5`` the zero-traversal term alone carries ~60 % of
    ``S`` but occurs in well under 1 % of unconditioned histories; here it is the
    exactly-known ``g_0 = 1``.

    ``batch_g`` holds one ``g`` estimate per independent batch. Central values
    use the pooled mean; uncertainties come from a delete-one jackknife over
    batches, which handles nonlinear observables (root-finds, ratios) without a
    second stream and without assuming the batches are Gaussian.
    """

    batch_g: npt.NDArray[np.float64]  # (n_batches, k_max + 1)
    b_max_um: float
    mean_traversal_dose_Gy: float
    effective_area_um2: float
    realizations_per_batch: int
    max_supported_dose_Gy: float  # dose the traversal grid was sized for

    def __post_init__(self) -> None:
        if self.batch_g.ndim != 2 or self.batch_g.shape[0] < 2:
            raise ValueError("batch_g must be a (n_batches >= 2, k_max + 1) array")
        if self.realizations_per_batch <= 0:
            raise ValueError("realizations_per_batch must be > 0")

    @property
    def n_batches(self) -> int:
        return int(self.batch_g.shape[0])

    @property
    def k_max(self) -> int:
        return int(self.batch_g.shape[1] - 1)

    @property
    def n_realizations(self) -> int:
        return self.n_batches * self.realizations_per_batch

    @property
    def g(self) -> npt.NDArray[np.float64]:
        """Pooled ``g_k`` over all batches."""
        return np.asarray(self.batch_g.mean(axis=0), dtype=np.float64)

    def pooled_with(self, other: TraversalConditionedSurvival) -> TraversalConditionedSurvival:
        """Concatenate the batches of two runs of the same curve.

        Lets an adaptive allocator *add* histories instead of discarding the
        pilot and restarting, which is what a fixed-seed restart amounts to.
        """
        if other.k_max != self.k_max:
            raise ValueError(f"k_max mismatch: {self.k_max} vs {other.k_max}")
        if other.realizations_per_batch != self.realizations_per_batch:
            raise ValueError("realizations_per_batch must match to pool batches")
        for attr in ("b_max_um", "mean_traversal_dose_Gy", "effective_area_um2"):
            if not math.isclose(getattr(self, attr), getattr(other, attr), rel_tol=1.0e-12):
                raise ValueError(f"{attr} mismatch; curves describe different ions")
        return TraversalConditionedSurvival(
            batch_g=np.concatenate([self.batch_g, other.batch_g], axis=0),
            b_max_um=self.b_max_um,
            mean_traversal_dose_Gy=self.mean_traversal_dose_Gy,
            effective_area_um2=self.effective_area_um2,
            realizations_per_batch=self.realizations_per_batch,
            max_supported_dose_Gy=min(self.max_supported_dose_Gy, other.max_supported_dose_Gy),
        )

    # ---- dose <-> Poisson mean ------------------------------------------
    @property
    def _dose_per_traversal_mean(self) -> float:
        """``dlambda/dD``: fluence per Gy times the b_max disc area."""
        return math.pi * self.b_max_um**2 / (self.effective_area_um2 * self.mean_traversal_dose_Gy)

    def traversal_mean(self, dose_Gy: float) -> float:
        """Mean number of tracks in the ``b_max`` disc at ``dose_Gy``."""
        return dose_Gy * self._dose_per_traversal_mean

    def _poisson_weights(self, lam: float) -> npt.NDArray[np.float64]:
        k = np.arange(self.k_max + 1, dtype=np.float64)
        if lam <= 0.0:
            weights = np.zeros_like(k)
            weights[0] = 1.0
            return weights
        # Built in log space: lambda reaches ~10^4 traversals at the light-ion
        # entrance points, where k! overflows long before the pmf underflows.
        return np.asarray(np.exp(-lam + k * math.log(lam) - gammaln(k + 1.0)), dtype=np.float64)

    # ---- observables -----------------------------------------------------
    def survival(self, dose_Gy: float, *, g: npt.NDArray[np.float64] | None = None) -> float:
        """``S(D)`` from the Poisson mixture over traversal counts."""
        weights = self._poisson_weights(self.traversal_mean(dose_Gy))
        return float(weights @ (self.g if g is None else g))

    def log_kill_slope_Gy_inv(
        self, dose_Gy: float, *, g: npt.NDArray[np.float64] | None = None
    ) -> float:
        r"""``d[-ln S]/dD`` evaluated analytically at ``dose_Gy``.

        With ``w_k = Poisson(k; lambda)``, ``dw_k/dlambda = w_{k-1} - w_k``, so
        the derivative is an exact linear functional of the same ``g`` vector --
        no local fit, no window width, no differentiation of sample noise.
        """
        values = self.g if g is None else g
        weights = self._poisson_weights(self.traversal_mean(dose_Gy))
        d_weights = np.concatenate([[0.0], weights[:-1]]) - weights
        survival = float(weights @ values)
        if survival <= 0.0:
            raise ValueError(f"survival is not positive at D={dose_Gy:g} Gy")
        return -float(d_weights @ values) * self._dose_per_traversal_mean / survival

    def dose_for_survival(
        self,
        survival_level: float,
        *,
        g: npt.NDArray[np.float64] | None = None,
        max_dose_Gy: float | None = None,
    ) -> float:
        """Dose at which the curve crosses ``survival_level`` (exact root-find)."""
        if not 0.0 < survival_level < 1.0:
            raise ValueError(f"survival_level must be in (0, 1), got {survival_level!r}")
        values = self.g if g is None else g
        upper = max_dose_Gy if max_dose_Gy is not None else self.max_supported_dose_Gy
        if self.survival(upper, g=values) > survival_level:
            raise ValueError(
                f"curve does not reach S={survival_level:g} within its supported range "
                f"(S={self.survival(upper, g=values):g} at D={upper:g} Gy); "
                "rebuild it with a larger max_dose_Gy"
            )
        return float(
            brentq(
                lambda dose: math.log(self.survival(dose, g=values) / survival_level),
                1.0e-9,
                upper,
                xtol=1.0e-10,
                rtol=1.0e-12,
            )
        )

    def poisson_tail_mass(self, dose_Gy: float) -> float:
        """Probability mass beyond ``k_max`` -- the truncation error bound on ``S``.

        ``g_k`` is non-increasing in ``k``, so the neglected terms are bounded by
        this mass times ``g_kmax``; a value comparable to ``S`` itself means the
        grid is too short for that dose.
        """
        return float(max(1.0 - self._poisson_weights(self.traversal_mean(dose_Gy)).sum(), 0.0))

    def jackknife(
        self, observable: Callable[[npt.NDArray[np.float64]], float]
    ) -> tuple[float, float]:
        """Return ``(value, standard_error)`` for any functional of ``g``.

        Delete-one jackknife over batches (`_delete_one_jackknife`). Each
        replicate still uses all but one batch, so nonlinear observables stay
        well-conditioned where a two-sample range would be dominated by its own
        sampling noise.
        """
        return _delete_one_jackknife(self.batch_g, observable)


_TC_PREFIX_BLOCKS: int = 64
_TC_FIELD_POINT_CHUNK: int = 1_000_000

# ---- Monte-Carlo constants and table ------------------------------------
# Moved here from LEM1: the local-dose Monte Carlo is the reference path for
# every version whose lethal-event density is a pointwise function of the
# local dose, which is LEM I, II and III. LEM IV is not (see LEM4).
_MC_REALIZATIONS: int = 20000  # irradiated cells simulated per dose
_MC_KELLERER_FIELD_POINTS: int = 4000
_MC_FIT_DOSES_GY: tuple[float, ...] = (0.5, 1.0, 1.5, 2.0, 2.5, 3.0)
_MC_FIELD_POINTS: int = 2000  # importance-sampled field points per cell
_MC_TRACK_FRACTION: float = 0.7  # fraction of field points drawn near tracks (rest uniform)
_MC_MAX_MEAN_TRAVERSALS: float = 800.0  # guard: refuse energies where n_bar blows up
_TC_REALIZATIONS_PER_BATCH: int = 25
_TC_BATCHES: int = 16
_TC_POISSON_TAIL_SIGMA: float = 8.0  # k_max = lambda + this many sqrt(lambda), + pad
_TC_POISSON_TAIL_PAD: int = 12
_KELLERER_B_NODES: int = 512
_KELLERER_B_PER_R_EFF: int = 40
_KELLERER_B_MAX_NODES: int = 20_000
_KELLERER_R_NODES: int = 1024
_KELLERER_U_NODES: int = 2048
_KELLERER_CHUNK_ELEMENTS: int = 4_000_000

MC_KELLERER_FIELD_POINTS: int = _MC_KELLERER_FIELD_POINTS
"""Default number of Kellerer weighted-sampling field points per cell, i.e. the
``n_field_points`` default of [`LEMBase.scholz1997_montecarlo_survival`][].
The public name of ``_MC_KELLERER_FIELD_POINTS`` -- the *same* object, so a
script that records it records exactly what the library ran.
"""


def _auto_engine() -> Literal["numba", "numpy"]:
    """What ``engine="auto"`` means: ``"numba"`` when numba is importable, else ``"numpy"``.

    The one definition of the rule, shared by [`resolve_mc_engine`][] and the
    private Monte-Carlo helpers of [`LEMBase`][]. Those helpers resolve only
    ``"auto"`` here and keep their own handling of an explicit ``"numba"``
    without numba (they fall through to the un-jitted Python path rather than
    raise); their public callers validate first.
    """
    return "numba" if _mc_kernel.HAVE_NUMBA else "numpy"


def resolve_mc_engine(engine: Engine) -> Literal["numba", "numpy"]:
    """Resolve an [`Engine`][] for the local-dose Monte Carlo.

    ``"auto"`` becomes ``"numba"`` when the compiled kernel
    (`pyradbio.models.lem._mc_kernel`, the ``fast`` extra) is importable and
    ``"numpy"`` otherwise; ``"numba"`` without numba raises. This is the
    resolution [`LEMBase.scholz1997_montecarlo_survival`][] applies,
    exposed so the Monte-Carlo batch scripts need not reach into
    ``_mc_kernel.HAVE_NUMBA`` themselves.

    Raises:
        ValueError: ``engine`` is not one of [`ENGINES`][],
            or is ``"numba"`` while numba is not importable.
    """
    if engine not in ENGINES:
        raise ValueError(f"engine must be one of {ENGINES!r}, got {engine!r}")
    if engine == "auto":
        return _auto_engine()
    if engine == "numba":
        if not _mc_kernel.HAVE_NUMBA:
            raise ValueError(
                "engine='numba' requires the 'fast' extra (pip install pyradbio[fast]); "
                "use engine='numpy' instead."
            )
        return "numba"
    return "numpy"


@dataclass(frozen=True, slots=True)
class NuclearDoseTable:
    r"""Per-impact-parameter nuclear dose, plus its conditional radial sampler.

    Everything needed for Kellerer's *weighted sampling* (1985, section VI.A.2,
    the method Scholz et al. 1997 cite as ref. [6] but do not restate):

    * ``mean_dose_Gy[j]`` is ``d_bar(b_j)``, the nucleus-average dose deposited by
      one track at impact parameter ``b_j``. Selecting a track with probability
      proportional to this is Kellerer's "selection probability being
      proportional to the corresponding energy transfer".
    * ``radial_cdf[j]`` inverts to a radius drawn with density proportional to
      ``D(r)`` *restricted to the nucleus*, so every sample lands inside it.

    The two together give ``p(x) = d(x) / (A * sum_i d_bar(b_i))`` exactly -- the
    sampling density is the local dose, which is the whole point.
    """

    b_grid: npt.NDArray[np.float64]  # (nb,) impact parameters [um]
    r_grid: npt.NDArray[np.float64]  # (nr,) radii from the track [um]
    mean_dose_Gy: npt.NDArray[np.float64]  # (nb,) nucleus-average dose [Gy]
    radial_cdf: npt.NDArray[np.float64]  # (nb, nr) normalised, 0 -> 1
    half_angle: npt.NDArray[np.float64]  # (nb, nr) arc half-width inside the nucleus
    # Pre-inverted sampler, on a UNIFORM grid in the CDF variable u. Drawing a
    # radius is then a gather plus a lerp, with no per-impact-parameter Python
    # loop and no search -- which is what makes the sampler, rather than the
    # compiled dose kernel, stop dominating the run.
    log_r_of_u: npt.NDArray[np.float64]  # (nb, nu) log radius at u
    half_of_u: npt.NDArray[np.float64]  # (nb, nu) arc half-width at that radius


@dataclass(frozen=True, slots=True)
class LEMBase(BiophysicalModel):
    capabilities: ClassVar[ModelCapabilities] = ModelCapabilities()
    initial_slope_models: ClassVar[tuple[InitialSlopeModel, ...]] = (
        "cross_section",
        "lethal_central",
        "lethal_central_saturating",
        "lethal_full",
    )
    """The ``initial_slope_model`` values this version implements; checked at
    construction. ``"full_simulation"`` needs LEM IV's simulated survival curve.
    """

    photon: PhotonParameters
    cell: CellTarget
    stopping_power: StoppingPower = field(default_factory=default_stopping_power)
    radial_dose: RadialDoseModel | None = None
    """Optional injection; None selects the version default at the current stopping power."""
    initial_slope_model: InitialSlopeModel = "cross_section"
    resolved_radial_dose: RadialDoseModel = field(init=False, repr=False, compare=False)
    """The active track model. Rebuilt on replace; never copied as a constructor input."""

    def __post_init__(self) -> None:
        if self.photon.D_t is None:
            raise ValueError(
                "LEM requires a photon transition dose D_t (shouldered->exponential "
                "reference survival); set PhotonParameters(..., D_t=...)"
            )
        if self.initial_slope_model not in self.initial_slope_models:
            raise ValueError(
                f"initial_slope_model must be one of {self.initial_slope_models!r} for "
                f"{type(self).__name__}, got {self.initial_slope_model!r}"
                + (
                    " ('full_simulation' fits a simulated survival curve, which only LEM IV builds)"
                    if self.initial_slope_model == "full_simulation"
                    else ""
                )
            )
        radial = self._radial_dose_model() if self.radial_dose is None else self.radial_dose
        object.__setattr__(self, "resolved_radial_dose", radial)

    @abstractmethod
    def _radial_dose_model(self) -> RadialDoseModel:
        """Version-specific radial-dose model (LEM I/II/III differ here)."""

    def _radial(self) -> RadialDoseModel:
        return self.resolved_radial_dose

    def _kernel_eta_table(self) -> tuple[npt.NDArray[np.float64], npt.NDArray[np.float64]]:
        """``(log10 dose, eta)`` describing this model's supra-threshold response.

        The compiled Monte-Carlo kernel cannot call back into Python, so the one
        thing that differs between LEM versions above ``D_t`` -- the cluster
        enhancement ``eta(D)`` of eq. 10 -- is handed to it as a table. Returning
        an **empty** pair means ``eta == 1``, i.e. the LEM I form
        ``lk_t + s_max (D - D_t)``, and the kernel then runs the exact arithmetic
        it ran before this hook existed.

        Overriding this is what makes the Monte Carlo correct for a new version;
        forgetting to override it yields a *LEM I* answer under that version's
        name, so `_check_mc_response` cross-checks the table against
        `_photon_log_kill_array` before any run.
        """
        return _NO_ETA, _NO_ETA

    def _check_mc_response(self) -> None:
        """Fail loudly if ``_kernel_eta_table`` does not reproduce this model's response.

        The kernel evaluates the photon response itself, so a version whose
        response is not captured by ``(LQ below D_t, s_max (eta D - D_t) above)``
        would be silently simulated as LEM I. Rather than trust that every
        subclass remembers, verify it: compare the kernel's reconstruction
        against `_photon_log_kill_array` on a dose grid spanning the range
        a track can reach.
        """
        d_t = self.photon.D_t
        assert d_t is not None  # enforced in __post_init__
        eta_log_d, eta_val = self._kernel_eta_table()
        # Strictly *above* D_t: at D == D_t the kernel takes its `d <= d_t` LQ
        # branch, so comparing the supra-threshold formula there would flag a
        # difference that neither path actually computes.
        dose = np.logspace(math.log10(max(d_t, 1e-3)) + 1e-9, 9.0, 256, dtype=np.float64)
        s_max = self.photon.max_slope()
        lk_t = float(self._photon_log_kill_array(np.array([d_t]))[0])
        if eta_log_d.size == 0:
            eta = np.ones_like(dose)
        else:
            eta = np.interp(np.log10(dose), eta_log_d, eta_val)
        kernel = lk_t + s_max * (eta * dose - d_t)
        exact = self._photon_log_kill_array(dose)
        worst = float(np.max(np.abs(kernel - exact) / np.maximum(np.abs(exact), 1e-12)))
        if worst > _MC_RESPONSE_RTOL:
            raise NotImplementedError(
                f"{type(self).__name__}'s photon response is not reproduced by its "
                f"_kernel_eta_table (worst relative error {worst:.3g} > {_MC_RESPONSE_RTOL:g}), "
                "so the compiled Monte-Carlo kernel would simulate a different model. "
                "Override _kernel_eta_table, or run with engine='numpy', which "
                "evaluates _photon_log_kill_array directly."
            )

    # ---- single-traversal quantities -----------------------------------
    def mean_traversal_dose_Gy(self, ion: Ion) -> float:
        """Mean dose ``D_1`` imparted to the nucleus by one ion traversal [Gy].

        ``D_1 = 0.16 * LET / (rho * A_eff)`` (Elsasser & Scholz 2007 eq. 8): the
        specific energy of a single track averaged over the effective nuclear
        cross-section ``A_eff = pi * R_eff^2``. The prefactor ``0.16`` keV/um per
        Gy*um^2*(g/cm^3) is ``1 / KEVUM_PER_GY_UM2_GCM3`` (energy conservation,
        see ``core/constants.py``); ``rho`` sits in the denominator because dose
        is energy per unit *mass*.
        """
        let = self.stopping_power.let_keV_um(ion)
        r_eff = self.cell.effective_nucleus_radius_um()
        area = math.pi * r_eff * r_eff
        return LEM_D1_PREFACTOR * let / (WATER_DENSITY_G_CM3 * area)

    def mean_traversals(self, ion: Ion, dose_Gy: float) -> float:
        """Mean number of tracks ``n_bar`` that can reach the nucleus at ``dose_Gy``.

        Counts the tracks whose impact parameter lies inside
        ``b_max = R_eff + r_max``: the fluence ``phi = D / (A_eff D_1)`` (Scholz &
        Kraft 1996; ``D_1`` from [`mean_traversal_dose_Gy`][], Elsasser &
        Scholz 2007 eq. 8, doi:10.1667/RR0467.1) times the disc area ``pi b_max^2``.
        It is the Poisson mean of the track count per cell in
        [`scholz1997_montecarlo_survival`][] and so the driver of its cost.

        The expression order is kept fixed (the Monte Carlo evaluates its own
        ``n_bar`` inline, in a different order) so cost estimates and stored rows
        stay bit-identical.
        """
        r_eff = self.cell.effective_nucleus_radius_um()
        b_max = r_eff + self.resolved_radial_dose.r_max_um(ion)
        fluence = dose_Gy / (math.pi * r_eff**2 * self.mean_traversal_dose_Gy(ion))
        return fluence * math.pi * b_max**2

    def near_traversals(self, ion: Ion, n_bar: float, b_cut_um: float | None) -> float:
        """Upper-bound cost estimate of the tracks resolved after the near/far split.

        With a cut set, [`scholz1997_montecarlo_survival`][] replaces the
        distant tracks by one compound-Poisson background draw, so they cost
        nothing per track. Since ``n_bar`` ([`mean_traversals`][]) counts tracks
        in a disc of radius ``b_max = R_eff + r_max``, the near count is an area
        ratio, taken here out to ``min(R_eff + b_cut_um, b_max)``.

        This is a runtime *estimate*, not the sampler's Poisson mean: the Monte
        Carlo measures its cut from the nucleus centre and generates
        ``n_bar (min(b_cut_um, b_max) / b_max)^2`` tracks (see
        ``_mc_accumulate_kellerer``), so for the same ``b_cut_um`` this proxy is
        larger by up to ``((R_eff + b_cut_um) / b_cut_um)^2`` (2.25x at
        ``b_cut_um = 10`` um, ``R_eff = 5`` um).

        Whether to cut at all is the caller's policy; a caller that wants no cut
        below some ``n_bar`` threshold passes ``None`` there.

        Args:
            ion: The ion whose ``r_max`` sets ``b_max``.
            n_bar: Mean track count in the ``b_max`` disc ([`mean_traversals`][]).
            b_cut_um: Near/far cut in um, measured from the nucleus **edge**
                (``R_eff + b_cut_um``) -- unlike the ``b_cut_um`` of
                [`scholz1997_montecarlo_survival`][], which is measured from
                the nucleus centre. ``None`` means no split and returns ``n_bar``
                unchanged.

        Raises:
            ValueError: ``b_cut_um`` is not ``None`` and not a positive number.
        """
        if b_cut_um is None:
            return n_bar
        if not b_cut_um > 0.0:
            raise ValueError(f"b_cut_um must be positive or None, got {b_cut_um!r}")
        r_eff = self.cell.effective_nucleus_radius_um()
        b_max = r_eff + self.resolved_radial_dose.r_max_um(ion)
        gen_radius = min(r_eff + b_cut_um, b_max)
        return n_bar * (gen_radius / b_max) ** 2

    def single_traversal_log_kill(self, ion: Ion) -> float:
        """Mean lethal events ``N_1`` from one central nuclear traversal.

        Integrates the photon log-kill of the local dose over the effective
        nuclear disk (Scholz & Kraft 1996 eqs. 5-6),

            N_1 = (2 / R_eff^2) * integral_0^{r_top} [-ln S_x(D(r))] r dr,

        with ``r_top = min(R_eff, r_max)`` (the track contributes nothing beyond
        ``r_max``; the nucleus has no chromatin beyond ``R_eff``). The 2-D
        cross-section integral reduces to this 1-D radial form by azimuthal
        symmetry of a central track (validated against an explicit 2-D
        quadrature in the tests).
        """
        radial = self._radial()
        r_eff = self.cell.effective_nucleus_radius_um()
        r_min = radial.r_min_um(ion)
        r_max = radial.r_max_um(ion)
        r_top = min(r_eff, r_max)
        # Log grid resolves both the saturated core (r < r_min, constant dose)
        # and the 1/r^2 tail. The sub-r_min floor contributes negligibly
        # (integrand ~ g*r -> 0) but keeps the core sampled.
        r = np.logspace(math.log10(r_min * 1.0e-3), math.log10(r_top), _NUCLEAR_NODES)
        log_kill = np.array(
            [self._photon_log_kill(radial.local_dose_Gy(float(ri), ion)) for ri in r]
        )
        integral = float(simpson(log_kill * r, x=r))
        return 2.0 / (r_eff * r_eff) * integral

    def _photon_log_kill(self, dose_Gy: float) -> float:
        """Local photon log-kill ``-ln S_x(D)`` at a deposited local dose [Gy].

        The hook every LEM version applies the photon dose-response through. LEM
        I uses the plain shouldered->exponential reference; LEM II onward override
        this to add the high-dose cluster enhancement ``eta(D)`` to the tail.
        """
        return self.photon.log_kill(dose_Gy)

    # ---- saturation geometry: where the cluster expansion is complete ----
    #
    # The photon response splits exactly as
    #
    #     h(d) = [alpha_x d + beta_x d^2]  -  beta_x (d - D_t)^2 1[d > D_t]
    #
    # The first piece is a degree-2 polynomial, and a quadratic has **no**
    # inclusion-exclusion excess beyond second order: ``psi_n = 0`` for every
    # ``n >= 3``. So all higher-order track interplay in LEM comes from the
    # second piece, and it exists only because the indicator ``1[d > D_t]``
    # depends on *which* tracks are present. Where the nucleus stays in the
    # quadratic branch, the nuclear log-kill ``nu`` is exactly a pair sum -- the
    # two-track integral is missing no 3-body geometry, and its only remaining
    # approximation is in the fluence average, not in the track configuration.
    #
    # These two methods locate a case on that map. See
    # ``docs/references/LEM_I_ScholzKraft.md``.

    def saturation_radius_um(self, ion: Ion) -> float:
        """Radius at which a *single* track's local dose falls to ``D_t`` [um].

        ``0.0`` when the track core never reaches ``D_t`` (the whole track sits
        in the quadratic branch); ``r_max`` when even the outermost radius is
        still above it. Requires a transition dose, as all of LEM does.
        """
        if self.photon.D_t is None:
            raise ValueError("saturation_radius_um requires a transition dose D_t")
        radial = self._radial()
        r_lo = radial.r_min_um(ion) * 1.0e-3
        r_max = radial.r_max_um(ion)
        if radial.local_dose_Gy(r_lo, ion) <= self.photon.D_t:
            return 0.0
        if radial.local_dose_Gy(r_max, ion) >= self.photon.D_t:
            return float(r_max)
        return float(
            brentq(
                lambda r: radial.local_dose_Gy(float(r), ion) - self.photon.D_t,
                r_lo,
                r_max,
                xtol=1.0e-12,
            )
        )

    def saturated_area_fraction(self, ion: Ion, dose_Gy: float) -> float:
        """Expected fraction of the nucleus driven past ``D_t`` by single tracks.

        Track centres are Poisson at fluence ``Phi = D / (A_eff D_1)``, and each
        carries a saturated disc of radius [`saturation_radius_um`][], so a
        point escapes saturation with probability ``exp(-Phi pi r_sat^2)``.

        This counts **only** single-track saturation, so it is a *lower* bound:
        several sub-threshold tracks can jointly cross ``D_t`` where none does
        alone. The cheap companion upper bound is Markov's -- the mean local dose
        over the nucleus is the macroscopic dose, so the saturated fraction is at
        most ``D / D_t``. The two together bracket the case: at 3 Gy on the
        Scholz 1997 CHO inputs (``D_t = 30`` Gy) the bracket is 0.5%-10%, i.e.
        the nucleus is overwhelmingly in the quadratic branch by area. That does not
        make the two-track pair integral complete: its integrand concentrates on the
        small saturated area (see ``docs/references/LEM_I_ScholzKraft.md``).
        """
        if dose_Gy < 0.0:
            raise ValueError("dose_Gy must be >= 0")
        r_sat = self.saturation_radius_um(ion)
        if r_sat <= 0.0 or dose_Gy == 0.0:
            return 0.0
        r_eff = self.cell.effective_nucleus_radius_um()
        fluence = dose_Gy / (math.pi * r_eff * r_eff * self.mean_traversal_dose_Gy(ion))
        return float(-math.expm1(-fluence * math.pi * r_sat * r_sat))

    def _max_local_slope(self) -> float:
        """Maximum local log-kill slope ``max_D d(-ln S_x)/dD`` [Gy^-1].

        The un-enhanced photon ``s_max = alpha_x + 2 beta_x D_t`` for LEM I; the
        cluster-extended LEM II/III override this with the enhanced ``s_max *
        eta_max`` (the local tail slope is ``s_max * eta(D)``, Elsasser 2007
        eq. 10). It is the saturation cap of the rapidLEM (``lethal_central``)
        initial slope -- see [`lethal_central_initial_slope`][].
        """
        return self.photon.max_slope()

    def single_traversal_survival(self, ion: Ion) -> float:
        """``S_1 = exp(-N_1)`` for one central nuclear traversal."""
        return math.exp(-self.single_traversal_log_kill(ion))

    # ---- log-kill profile and impact-parameter geometry ----------------
    def _radial_logkill_profile(
        self, ion: Ion
    ) -> tuple[npt.NDArray[np.float64], npt.NDArray[np.float64]]:
        """``(-ln S_x(D(s)))`` on a log-spaced radius grid ``s`` in ``[s0, r_max]``.

        Built once per ion and reused for every impact parameter. The grid floor
        ``s0 = r_min*1e-3`` sits inside the saturated core (constant dose), and
        ``D(s) = 0`` for ``s > r_max``. For LEM II/III this evaluates the
        (expensive) radical-diffusion convolution once -- the same cost the
        old single-central-traversal slope already paid.
        """
        radial = self._radial()
        r_min = radial.r_min_um(ion)
        r_max = radial.r_max_um(ion)
        s = np.logspace(math.log10(r_min * 1.0e-3), math.log10(r_max), _PROFILE_NODES).astype(
            np.float64
        )
        # The LEM II/III radical-diffusion convolution can dip to ~ -1e-9 Gy in the
        # far tail (Bessel-I0 quadrature round-off); clamp at 0 before the log-kill.
        logkill = np.array(
            [self._photon_log_kill(max(radial.local_dose_Gy(float(si), ion), 0.0)) for si in s],
            dtype=np.float64,
        )
        return s, logkill

    @staticmethod
    def _interp_logkill(
        s: npt.NDArray[np.float64],
        s_grid: npt.NDArray[np.float64],
        logkill_grid: npt.NDArray[np.float64],
    ) -> npt.NDArray[np.float64]:
        """Log-log interpolate the profile; constant core below ``s_grid[0]`` and
        zero beyond ``r_max = s_grid[-1]`` (no dose past the delta-electron range).
        """
        clamped = np.clip(s, s_grid[0], s_grid[-1])
        log_grid = np.log(np.maximum(logkill_grid, np.finfo(np.float64).tiny))
        interp = np.exp(np.interp(np.log(clamped), np.log(s_grid), log_grid))
        out = np.where(s <= s_grid[-1], interp, 0.0)
        return np.where(s < s_grid[0], logkill_grid[0], out)

    def _impact_log_kill(
        self,
        b: float,
        profile: tuple[npt.NDArray[np.float64], npt.NDArray[np.float64]],
        r_eff: float,
        r_max: float,
    ) -> float:
        """Nuclear-averaged log-kill ``nu(b)`` for a track at impact parameter ``b``.

        The 2-D integral of ``-ln S_x(D)`` over the nuclear disk (offset by ``b``
        from the track) reduces by the chord-angle weighting to a 1-D radial
        quadrature:

        ```text
        nu(b) = (1/(pi R^2)) integral logkill(s) * phi_in(s,b) * s ds,
        ```

        with ``phi_in`` from `_angular_fraction_inside`. The integration
        spans only the shell ``s in [max(s0, b-R), min(r_max, b+R)]`` the nucleus
        actually overlaps, refined linearly so distant (large-``b``) shells stay
        resolved; for ``b < R`` it reaches the core to capture the high central
        dose.
        """
        s_grid, logkill_grid = profile
        s_hi = min(r_max, b + r_eff)
        s_lo = (b - r_eff) if b >= r_eff else s_grid[0]
        if s_hi <= s_lo:
            return 0.0
        in_range = s_grid[(s_grid >= s_lo) & (s_grid <= s_hi)]
        shell = np.linspace(s_lo, s_hi, _SHELL_NODES)
        s = np.unique(np.concatenate([in_range, shell]))
        logkill = self._interp_logkill(s, s_grid, logkill_grid)
        phi = _angular_fraction_inside(s, b, r_eff)
        # trapezoid (not simpson): the merged profile+shell grid can place two
        # nodes ~1e-14 apart, and non-uniform Simpson divides by that ~0 interval.
        integral = float(trapezoid(logkill * phi * s, x=s))
        return integral / (math.pi * r_eff * r_eff)

    def _impact_profile(self, ion: Ion) -> tuple[npt.NDArray[np.float64], npt.NDArray[np.float64]]:
        """``(b, nu(b))`` on the impact-parameter grid out to ``b = R + r_max``.

        The single-track nuclear log-kill ``nu(b)`` (`_impact_log_kill`)
        sampled linearly through the nucleus and logarithmically over the indirect
        (penumbra) tail. Shared by the inactivation cross-section and the full-LEM
        initial slope, which differ only in how they weight ``nu(b)``.
        """
        radial = self._radial()
        r_eff = self.cell.effective_nucleus_radius_um()
        r_max = radial.r_max_um(ion)
        profile = self._radial_logkill_profile(ion)

        b_max = r_eff + r_max
        b_near = np.linspace(0.0, 1.5 * r_eff, _IMPACT_NEAR_NODES)
        b_far = np.logspace(math.log10(1.5 * r_eff), math.log10(b_max), _IMPACT_FAR_NODES)
        b = np.unique(np.concatenate([b_near, b_far]))
        nu = np.array([self._impact_log_kill(float(bi), profile, r_eff, r_max) for bi in b])
        return b, nu

    def inactivation_cross_section_um2(self, ion: Ion) -> float:
        """Lethal-event inactivation cross-section ``sigma`` [um^2] (low-dose limit).

        For Poisson-distributed traversals the survival is
        ``S(D) = exp(-Phi * integral (1 - exp(-nu(b))) 2 pi b db)`` (Campbell's
        theorem on the non-overlapping single-track kills ``nu(b)``), so

            sigma = integral_0^{R+r_max} (1 - exp(-nu(b))) 2 pi b db.

        This is the exact zero-dose/single-track quantity ``sigma_0``. It must not
        be identified with Scholz & Kraft 1996 Figs. 2-3, whose ``sigma_inact`` is
        evaluated from the final survival-curve slope at 1% survival. Use
        [`approximate_inactivation_cross_section_um2`][] for an LQ approximation
        to that finite-survival observable, or a full stochastic survival curve
        for the paper calculation. The impact parameter is sampled linearly
        through the nucleus and logarithmically over the indirect tail out to
        ``b = R + r_max``.
        """
        b, nu = self._impact_profile(ion)
        integrand = (1.0 - np.exp(-nu)) * 2.0 * math.pi * b
        return float(trapezoid(integrand, x=b))

    def approximate_inactivation_cross_section_um2(
        self,
        ion: Ion,
        *,
        survival_level: float,
    ) -> float:
        r"""Approximate finite-survival inactivation cross-section [um^2].

        The selected model's ion LQ parameters define the dose ``D_S`` at the
        requested survival and the local curve slope

        ``d[-ln S]/dD | D_S = alpha_I + 2 beta_I D_S``.

        Converting dose slope to fluence slope gives

        ``sigma_approx(S) = 0.16 LET/rho * (alpha_I + 2 beta_I D_S)``.

        This supplies a fast, deterministic counterpart to the full stochastic
        final-slope calculation in Scholz & Kraft 1996 Figs. 2-3
        (doi:10.1016/0273-1177(95)00784-C). It is explicitly an approximation:
        the result depends on ``initial_slope_model`` and the version-specific
        beta model. At ``survival_level=1`` it reduces to ``sigma_0`` when the
        default ``initial_slope_model="cross_section"`` is selected.
        """
        lq = self.alpha_beta(ion)
        dose_at_level = dose_for_survival(survival_level, lq.alpha, lq.beta)
        slope = lq.alpha + 2.0 * lq.beta * dose_at_level
        let_keV_um = self.stopping_power.let_keV_um(ion)
        return LEM_D1_PREFACTOR * let_keV_um * slope / WATER_DENSITY_G_CM3

    def lethal_full_initial_slope(self, ion: Ion) -> float:
        r"""Overkill-free lethal-events initial slope (``initial_slope_model='lethal_full'``).

        The impact-parameter integral of the *unsaturated* mean lethal events --
        the mean lethal-event yield per dose ``E[N]/D`` -- with ``Phi = D/(A_eff D_1)``,

            alpha_I = (1 / (A_eff D_1)) * integral_0^{R+r_max} nu(b) 2 pi b db.

        This is **not** the exact LEM slope. The exact population-survival slope
        ([`inactivation_cross_section_um2`][], the ``"cross_section"`` model) keeps the
        single-track overkill term: ``-ln<S> = Phi * integral (1-exp(-nu(b)))
        2 pi b db`` (Campbell/Jensen). Integrating ``nu(b)`` directly drops that
        ``1-exp(-nu)`` factor, so at high LET this does **not** thin down --
        ``nu(b)`` is bounded by ``s_max D_1`` (the photon response caps the local-
        dose log-kill slope at ``s_max``), giving ``alpha_I -> s_max`` instead of
        the ``"cross_section"`` overkill ``-> 1/D_1``.

        It matches the ``"cross_section"`` slope at low LET (``nu << 1``, overkill
        negligible) and reproduces the **TRiP98 / rapidLEM** RBE tables across the
        whole energy range and every tissue -- because rapidLEM makes the same
        overkill-free simplification (capping at ``s_max``), but with the indirect
        traversals that ``"lethal_central"``'s central-only form drops. Use it to
        reproduce TRiP; use ``"cross_section"`` for the exact explicit-LEM / MC slope.
        Not defined for LEM IV's DSB-pattern override.
        """
        b, nu = self._impact_profile(ion)
        lethal_full_um2 = float(trapezoid(nu * 2.0 * math.pi * b, x=b))
        r_eff = self.cell.effective_nucleus_radius_um()
        area = math.pi * r_eff * r_eff
        return lethal_full_um2 / area / self.mean_traversal_dose_Gy(ion)

    def initial_slope(self, ion: Ion) -> float:
        """Ion initial slope ``alpha_I`` from the inactivation cross-section.

        At low dose ``-ln S(D) = (sigma / (A_eff * D_1)) * D`` (one-track limit of
        [`inactivation_cross_section_um2`][]), so ``alpha_I = sigma/(A_eff*D_1)``
        with ``D_1`` the mean single-traversal dose. This reproduces the photon
        limit ``alpha_I -> alpha_x`` at low LET (indirect traversals included) and
        the overkill cap ``alpha_I -> 1/D_1`` (``sigma -> A_eff``) at high LET.

        With ``initial_slope_model="lethal_central"`` the TRiP98 / rapidLEM formula is
        used instead (see [`lethal_central_initial_slope`][]); with ``"lethal_full"``
        the LEM low-dose lethal-events slope (see [`lethal_full_initial_slope`][]).
        """
        if self.initial_slope_model == "lethal_central":
            return self.lethal_central_initial_slope(ion)
        if self.initial_slope_model == "lethal_central_saturating":
            return self.lethal_central_saturating_initial_slope(ion)
        if self.initial_slope_model == "lethal_full":
            return self.lethal_full_initial_slope(ion)
        r_eff = self.cell.effective_nucleus_radius_um()
        area = math.pi * r_eff * r_eff
        sigma = self.inactivation_cross_section_um2(ion)
        return sigma / area / self.mean_traversal_dose_Gy(ion)

    def lethal_central_initial_slope(self, ion: Ion) -> float:
        """TRiP-plateau initial slope ``alpha_I = min(N_1/D_1, s_max_local)``.

        A central-traversal shortcut: the mean lethal events of one central
        traversal ``N_1`` ([`single_traversal_log_kill`][]) over its mean nuclear
        dose ``D_1`` ([`mean_traversal_dose_Gy`][]), capped at the maximum *local*
        log-kill slope ``s_max_local = _max_local_slope()``. Since
        ``N_1/D_1 ~ alpha_z`` (the intrinsic ion slope), the cap makes this
        **saturate** at ``s_max`` toward high LET (TRiP's ``RBE_alpha`` plateau).

        NOTE: this is *not* the Krämer & Scholz 2006 rapidLEM initial slope, which
        is ``alpha_D = (1 - exp(-N_1))/D_1`` (their eq 9) -- a smooth ``1 - exp``
        saturation with **no** ``s_max`` cap that thins down as ``1/LET`` at high
        LET. The faithful KS2006 behaviour is reproduced by the default
        ``"cross_section"`` mode (the impact-parameter integral); ``lethal_central``
        is a coarser plateau variant. See
        docs/references/rapidLEM_KraemerScholz2006.md. Not defined for LEM IV's
        DSB-pattern override.

        The cap uses the **enhanced** maximum slope for the cluster-extended LEM
        II/III: the local tail slope there is ``s_max * eta(D)`` (Elsasser 2007
        eq. 10), so ``N_1 <= s_max * eta_max * D_1`` and the correct cap is
        ``s_max * eta_max`` (`_max_local_slope`), not the un-enhanced
        ``s_max``. Capping at the un-enhanced ``s_max`` would silently discard the
        cluster enhancement that the ``"cross_section"`` default and
        ``"lethal_full"`` both keep; the enhanced cap makes all three modes
        consistent. For LEM I (``eta = 1``) it reduces to the plain ``s_max`` and
        is never active (``N_1/D_1 < s_max`` there).
        """
        d_1 = self.mean_traversal_dose_Gy(ion)
        return min(self.single_traversal_log_kill(ion) / d_1, self._max_local_slope())

    def lethal_central_saturating_initial_slope(self, ion: Ion) -> float:
        """rapidLEM initial slope ``alpha_D = (1 - exp(-N_1))/D_1`` (KS2006 eq 9).

        The Krämer & Scholz 2006 low-dose approximation eq (9), identical to
        Scholz 1997 eqs 5-6: the single-traversal inactivation probability
        ``1 - S_1 = 1 - exp(-N_1)`` per unit mean nuclear dose ``D_1``. Unlike
        [`lethal_central_initial_slope`][] (which plateaus at ``s_max``) this is a
        smooth ``1 - exp`` saturation with **no** cap that **thins down** as
        ``1/D_1`` (~ ``1/LET``) at high LET (overkill). It uses only the *central*
        traversal, so -- like the Scholz-1997 fast path -- it **under-shoots** the
        photon ``alpha_x`` at low LET (it drops toward the bare central slope,
        missing the indirect large-``b`` traversals); the default ``"cross_section"``
        impact-parameter integral is the more faithful KS2006 alpha that recovers
        ``alpha_x``. Pairs with LEM I's ``beta_model="rapid_lem"``
        (``beta_D = f^2 beta_z``). See docs/references/rapidLEM_KraemerScholz2006.md.
        """
        return (1.0 - self.single_traversal_survival(ion)) / self.mean_traversal_dose_Gy(ion)

    def _beta(self, alpha_ion: float) -> float:
        """``beta_I`` via slope matching (the LEM shared-``s_max`` construction).

        The maximum (high-dose) slope is radiation-quality independent in the
        LEM -- at high dose the local dose everywhere exceeds ``D_t`` -- so the
        ion curve shares the photon ``s_max = alpha_x + 2 beta_x D_t`` and turns
        exponential at the same ``D_t``: ``s_max = alpha_I + 2 beta_I D_t``.
        Hence ``beta_I = (s_max - alpha_I)/(2 D_t)``, clamped at 0 (a track dense
        enough that ``alpha_I > s_max`` gives a purely exponential curve).
        Recovers ``beta_x`` at low LET where ``alpha_I -> alpha_x``.

        NOTE: this is *not* the Krämer & Scholz 2006 rapidLEM beta ansatz, which is
        ``beta_D = f^2 * beta_z`` with ``f = alpha_D/alpha_z`` (their eq 10). The
        two differ; slope-matching over-steepens carbon beta ~2x vs the exact
        two-track (see ``lem1.scholz1997_two_track_beta`` and
        docs/references/rapidLEM_KraemerScholz2006.md).
        """
        assert self.photon.D_t is not None  # enforced in __post_init__
        beta = (self.photon.max_slope() - alpha_ion) / (2.0 * self.photon.D_t)
        return max(beta, 0.0)

    def rapid_lem_beta(self, ion: Ion) -> float:
        """rapidLEM quadratic term ``beta_D = f^2 * beta_z`` (Krämer & Scholz 2006 eq 10).

        The scaling factor ``f = alpha_D/alpha_z = (1 - exp(-N_1))/N_1`` (with
        ``alpha_D`` the saturating initial slope, KS2006 eq 9, and ``alpha_z =
        N_1/D_1`` the central-traversal z-domain slope) squared, times the z-domain
        ``beta_z = (s_max - alpha_z)/(2 D_t)`` (Scholz 1997 eq 11,
        [`scholz1997_specific_energy_lq`][pyradbio.models.lem.lem1.LEM1.scholz1997_specific_energy_lq]).
        It is the ``f^2`` suppression of ``beta_z`` (``f -> 1`` low LET, ``f -> 0`` high LET). Like
        its central ``alpha_z`` it uses only the central traversal, so it does not reproduce the
        photon ``beta_x`` limit as cleanly as the impact-parameter path at low LET. Pairs with
        ``initial_slope_model="lethal_central_saturating"`` for the central-traversal KS2006
        rapidLEM; see docs/references/rapidLEM_KraemerScholz2006.md.
        """
        n_1 = self.single_traversal_log_kill(ion)
        delta_z = self.mean_traversal_dose_Gy(ion)
        beta_z = self._scholz1997_specific_energy_lq_from(delta_z=delta_z, n_1=n_1).beta
        f = (1.0 - math.exp(-n_1)) / n_1 if n_1 > 0.0 else 1.0
        return f * f * beta_z

    def _scholz1997_specific_energy_lq_from(self, *, delta_z: float, n_1: float) -> LQParameters:
        """Build the z-domain LQ from precomputed central-traversal quantities.

        Callers that have already evaluated the (uncached, integral-based)
        ``delta_z`` and ``n_1`` reuse them here instead of recomputing, and the
        ``beta_z`` slope-matching formula lives in this one place.
        """
        assert self.photon.D_t is not None  # enforced in __post_init__
        alpha_z = n_1 / delta_z
        beta_z = max((self.photon.max_slope() - alpha_z) / (2.0 * self.photon.D_t), 0.0)
        return LQParameters(alpha=alpha_z, beta=beta_z)

    def _photon_log_kill_array(self, dose_Gy: npt.NDArray[np.float64]) -> npt.NDArray[np.float64]:
        """Vectorised photon ``-ln S_x(D)`` (LQ below ``D_t``, linear above).

        The array twin of `_photon_log_kill`, and it **must stay in step
        with it**: the pair-overlap integrals ([`scholz1997_two_track_beta`][])
        evaluate the response on ~1e5-element grids and go through this one.
        This base implementation is the plain shouldered->exponential reference;
        LEM II/III override *both* to add the cluster enhancement. A version that
        overrode only the scalar would silently give the un-enhanced answer to
        every array-based beta model.
        """
        assert self.photon.D_t is not None  # enforced in __post_init__
        a, b, d_t = self.photon.alpha_x, self.photon.beta_x, self.photon.D_t
        s_max = self.photon.max_slope()
        lk_t = a * d_t + b * d_t * d_t
        lq = a * dose_Gy + b * dose_Gy * dose_Gy
        linear = lk_t + s_max * (dose_Gy - d_t)
        return np.asarray(np.where(dose_Gy <= d_t, lq, linear), dtype=np.float64)

    def _two_track_dose_profile(
        self, ion: Ion, nodes: int
    ) -> tuple[npt.NDArray[np.float64], npt.NDArray[np.float64], float, float]:
        """Radial ``D(r)`` on a log grid ``[r_min*1e-3, r_max)`` for the overlap integrals.

        Returns ``(r, dose, r_lo, r_max)``. The top node sits just inside ``r_max``
        so ``D(r)`` stays positive there (``local_dose_Gy`` is exactly 0 at/beyond
        ``r_max``); the ``r > r_max`` cut-off is applied by `_two_track_dose_at`.
        """
        radial = self._radial()
        r_min = radial.r_min_um(ion)
        r_max = radial.r_max_um(ion)  # raises if r_max <= r_min (energy too low)
        r_lo = r_min * 1.0e-3
        r = np.logspace(
            math.log10(r_lo), math.log10(r_max * (1.0 - 1.0e-9)), nodes, dtype=np.float64
        )
        dose = np.array([radial.local_dose_Gy(float(ri), ion) for ri in r], dtype=np.float64)
        return r, dose, r_lo, r_max

    def _analytic_profile(self, ion: Ion) -> AnalyticRadialProfile | None:
        """Closed form for ``D(r)`` if this track model has one, else ``None``.

        Decided by **measuring the sampled profile**, never by recognising a track
        model, so a model whose ``D(r)`` is anything else falls back automatically
        -- LEM II/III radical diffusion, for instance, fails both checks. A model
        added later that does happen to be core/``1/r^2`` picks the fast path up
        without being told about it.
        """
        radial = self._radial()
        try:
            r_min = radial.r_min_um(ion)
            r_max = radial.r_max_um(ion)
        except ValueError:
            return None
        r, dose, _, _ = self._two_track_dose_profile(ion, _ANALYTIC_PROBE_NODES)
        if np.any(dose <= 0.0):
            return None
        core, tail = r < r_min, r >= r_min
        if not core.any() or not tail.any():
            return None
        core_values = dose[core]
        tail_values = dose[tail] * r[tail] ** 2
        if _is_flat(core_values) and _is_flat(tail_values):
            return AnalyticRadialProfile(
                core_dose_Gy=float(core_values.mean()),
                k_Gy_um2=float(tail_values.mean()),
                r_min_um=float(r_min),
                r_max_um=float(r_max),
            )
        return self._convolved_analytic_profile(r, dose, float(r_min), float(r_max))

    @staticmethod
    def _convolved_analytic_profile(
        r: npt.NDArray[np.float64],
        dose: npt.NDArray[np.float64],
        r_min: float,
        r_max: float,
    ) -> AnalyticRadialProfile | None:
        r"""The radical-diffusion closed form, or ``None`` if the profile is not one.

        ``sigma`` is *measured*, never read off a track model, so this keeps
        `_analytic_profile`'s contract: a model whose ``D(r)`` happens to be
        a Gaussian-convolved ``1/r^2`` picks the fast path up without being told,
        and one that is anything else falls back.

        From ``D r^2 / k = 1 + 2 sigma^2/r^2 + O(r^-4)``, the quantity
        ``(D r^2/k - 1) * r^2`` is a *constant* estimate of ``2 sigma^2`` wherever
        the expansion holds. Requiring it to be constant across the outer decade
        is therefore both the fit and its own falsification: a profile that is not
        of this form gives no plateau and is rejected.
        """
        # ``k`` from the outer part, where the correction has died away to well
        # under the profile's own noise and cannot bias it -- but *not* from the
        # last node, which sits in the r_max cut-off. There the convolution rolls
        # the profile off over ~sigma, which on a log grid spanning nine decades
        # is entirely sub-grid: only the final node is affected, and it comes out
        # a factor ~2 low.
        outer = (r >= r_max * _CONV_K_LO_FRAC) & (r <= r_max * _CONV_K_HI_FRAC)
        if outer.sum() < _CONV_FIT_MIN_NODES:
            return None
        k = float(np.median(dose[outer] * r[outer] ** 2))
        if k <= 0.0:
            return None

        # ``(D r^2/k - 1) r^2`` estimates 2 sigma^2 wherever the expansion holds.
        # Select the nodes where it can actually be seen: the relative correction
        # must clear the ~1e-5 sampling noise, and stay small enough that the
        # asymptotic series is still converging. That window locates itself, so
        # no prior knowledge of sigma is needed.
        rel = dose * r**2 / k - 1.0
        fit = (rel > _CONV_FIT_REL_LO) & (rel < _CONV_FIT_REL_HI) & (r >= r_min)
        if fit.sum() < _CONV_FIT_MIN_NODES:
            return None
        two_sigma_sq = float(np.median(rel[fit] * r[fit] ** 2))
        if not (0.0 < two_sigma_sq < r_max * r_max):
            return None

        # Falsification, and the switch radius in one pass: where the expansion
        # holds the residual must sit at the profile's own numerical floor. If it
        # does not hold even in the far field, this is not a convolved 1/r^2 and
        # the whole closed form is refused.
        # Exclude the cut-off roll-off, which is ~6 sigma of *absolute* width --
        # 2.6e-5 of r_max at carbon 100 MeV/u but 1.3e-3 at 10, so a fixed
        # fraction rejects the low-energy profiles and passes the high-energy
        # ones. The Monte-Carlo profile does not resolve that roll-off either
        # (its node spacing at r_max is 2.7% of r_max against a 24 nm roll-off),
        # so both paths treat the cut-off as hard and agree there; it is only
        # this 4096-node probe that sees it at all.
        usable = (r >= r_min) & (r <= r_max - _CONV_EDGE_SIGMA * math.sqrt(two_sigma_sq / 2.0))
        rr = r[usable]
        x = two_sigma_sq / rr**2
        resid = np.abs(dose[usable] / ((k / rr**2) * (1.0 + x * (1.0 + 2.0 * x))) - 1.0)
        far = rr >= r_max * _CONV_K_LO_FRAC
        if not far.any() or float(np.max(resid[far])) > _CONV_PROFILE_RTOL:
            return None
        bad = np.nonzero(resid > _CONV_PROFILE_RTOL)[0]
        r_switch = max(float(rr[bad[-1]]) * _CONV_SWITCH_MARGIN if bad.size else r_min, r_min)
        if r_switch >= r_max:
            return None
        return AnalyticRadialProfile(
            core_dose_Gy=float(dose[0]),  # the convolved core, flat for r << sigma
            k_Gy_um2=k,
            r_min_um=r_min,
            r_max_um=r_max,
            two_sigma_sq_um2=two_sigma_sq,
            r_switch_um=r_switch,
            r_edge_um=r_max - _CONV_EDGE_SIGMA * math.sqrt(two_sigma_sq / 2.0),
        )

    @staticmethod
    def _two_track_dose_at(
        dist: npt.NDArray[np.float64],
        r: npt.NDArray[np.float64],
        dose: npt.NDArray[np.float64],
        r_lo: float,
        r_max: float,
        analytic: AnalyticRadialProfile | None = None,
    ) -> npt.NDArray[np.float64]:
        """Log-log interpolate ``D(dist)`` from the profile; 0 beyond ``r_max``.

        With ``analytic`` supplied the closed form is used **where it is valid**,
        which for an unconvolved profile is everywhere: there it is both faster
        and marginally more accurate, since log-log interpolation rounds the
        corner at ``r_min`` and the two differ only inside the single grid
        interval straddling it.

        A radical-diffusion profile is valid only on
        ``[r_switch_um, r_edge_um]`` and the interpolation is kept outside it.
        Skipping that check is not a small error: the correction factor
        ``1 + x + 2x^2`` has ``x = 2 sigma^2/r^2``, so at ``r = 1 nm`` with
        ``sigma = 4 nm`` it reaches 2081, and the resulting dose drove nuclear
        survival to a hard zero. This method feeds ``nuclear_dose_table``, so a
        wrong value here corrupts ``d_bar(b)`` and the sampling CDF, not just a
        dose read.
        """
        if analytic is not None and analytic.two_sigma_sq_um2 == 0.0:
            return analytic.dose_Gy(dist)  # unconvolved: valid at every radius
        clamped = np.clip(dist, r_lo, r_max)
        interp = np.exp(np.interp(np.log(clamped), np.log(r), np.log(dose)))
        value = np.asarray(np.where(dist > r_max, 0.0, interp), dtype=np.float64)
        if analytic is None:
            return value
        return np.asarray(np.where(analytic.valid(dist), analytic.dose_Gy(dist), value))

    def scholz1997_two_track_beta(
        self,
        ion: Ion,
        *,
        radial_nodes: int = _TWO_TRACK_RADIAL_NODES,
        separation_nodes: int = _TWO_TRACK_SEPARATION_NODES,
        angular_nodes: int = _TWO_TRACK_ANGULAR_NODES,
        impact_nodes: int = _TWO_TRACK_IMPACT_NODES,
    ) -> float:
        r"""Dose-domain ``beta`` from the two-track dose overlap (Scholz 1997 exact path).

        The stochastic LEM survival is ``S(D) = E[exp(-nu)]`` with
        ``nu = (1/A) integral_A -ln S_x(d(x)) dx`` and local dose
        ``d(x) = sum_i D(|x - c_i|)`` (Scholz et al. 1997 eqs. 2-5). Expanding the
        Poisson average in fluence ``phi = D/(A_eff D_1)`` gives the cross-section
        ``alpha`` at first order (a purely exponential, ``beta = 0`` survival if
        single-track kills were independent) and, at second order, a ``beta`` from
        the *non-additive overlap* of pairs of tracks -- overlapping tracks add
        their local doses before the nonlinear ``-ln S_x`` is applied.

        Reordering the pair integral field-point-first splits ``beta`` into a
        peak-free main term plus two localised corrections,
        ``beta = beta_A - beta_R - beta_S``:

        * ``beta_A = J_full / (2 (A_eff D_1)^2)`` with the position-independent
          pair integral ``J_full = integral integral [h(D(r_1)+D(r_2)) - h(D(r_1))
          - h(D(r_2))] (2 pi r_1 dr_1)(2 pi r_2 dr_2)`` (``h = -ln S_x``). Its LQ
          part equals ``2 beta_x (A_eff D_1)^2`` so ``beta_A -> beta_x`` at low LET,
          and it carries the overlap-saturation reduction automatically.
        * ``beta_R`` removes the ``1 - e^{-Delta}`` (pair-overlap saturation) and
          ``beta_S`` the ``e^{-nu_1}`` (already-dead central-track) nonlinearities,
          both evaluated from the plane pair-overlap ``Delta(s)`` (a function of
          track separation ``s``, computed peak-free by a perpendicular-bisector
          split so each track's ``1/r^2`` core is resolved in its own half) and the
          single-track nuclear log-kill ``nu_1(b)``. Both corrections are localised
          to close, central pairs, where the disk-restricted and plane overlaps
          agree; they vanish at low LET (recovering ``beta_x``) and grow at high LET
          (the paper's Fig. 3 fall-off).

        Deterministic and peak-free (no Monte-Carlo noise); reproduces the photon
        ``beta_x`` limit to <2% and the Scholz 1997 Fig. 3 exact curve to ~10-15%.
        Heavier than the other beta models -- a benchmark/reference path, not for
        inner loops.

        **This is the exact O(phi^2) cluster coefficient, not a fitted beta**, and
        the difference matters when comparing against the Monte Carlo. A beta
        obtained by least-squares fitting ``-ln S`` over a dose window projects
        part of the cubic and higher cluster terms onto ``D^2``, so it is a
        window-dependent quantity that need not equal this one. Measured at proton
        7 MeV/u against ``scholz1997_montecarlo_survival``, the ratio runs 0.995 /
        0.986 / 0.971 for windows 0.25-1.5 / 0.5-3.0 / 1.0-6.0 Gy -- converging on
        this value as the window approaches the origin, which is where the two
        definitions must agree. A percent-level gap at a clinical window is that
        projection, not an error in either path; the quadrature here is converged
        to <0.08% under a 4x refinement of every axis.
        """
        r_eff = self.cell.effective_nucleus_radius_um()
        a_eff = math.pi * r_eff * r_eff
        d_1 = self.mean_traversal_dose_Gy(ion)
        norm = 2.0 * (a_eff * d_1) ** 2

        r, dose, r_lo, r_max = self._two_track_dose_profile(ion, radial_nodes)
        h = self._photon_log_kill_array

        # --- beta_A: position-independent pair integral J_full (2-D radial) ------
        d1 = dose[:, None]
        d2 = dose[None, :]
        psi_full = h(d1 + d2) - h(d1) - h(d2)
        ring = 2.0 * math.pi * r
        j_full = float(trapezoid(trapezoid(psi_full * ring[None, :], r, axis=1) * ring, r))
        beta_a = j_full / norm

        # --- Delta_plane(s): plane pair-overlap log-kill vs separation ----------
        # Bisector split: integrate only the half closer to track 1 (r < R) in
        # track-1-centred log-polar coords -- track 2's core lies in the other
        # half, so each 1/r^2 peak is resolved by the log-radius grid, then double.
        dose_r = self._two_track_dose_at(r, r, dose, r_lo, r_max)
        w_r = np.gradient(r)
        psi = np.linspace(0.0, 2.0 * math.pi, angular_nodes, endpoint=False)
        d_psi = 2.0 * math.pi / angular_nodes
        cos_psi = np.cos(psi)
        sep = np.logspace(math.log10(r_lo), math.log10(2.0 * r_max), separation_nodes)
        delta_plane = np.empty(separation_nodes)
        for i, s in enumerate(sep):
            r_other = np.sqrt(r[:, None] ** 2 + s * s - 2.0 * r[:, None] * s * cos_psi[None, :])
            dose_other = self._two_track_dose_at(r_other, r, dose, r_lo, r_max)
            excess = h(dose_r[:, None] + dose_other) - h(dose_r[:, None]) - h(dose_other)
            in_half = r[:, None] < r_other
            radial_contrib = np.sum(np.where(in_half, excess, 0.0) * d_psi, axis=1) * (r * w_r)
            delta_plane[i] = 2.0 * float(np.sum(radial_contrib)) / a_eff

        # --- nu_1(b): single-track nuclear log-kill vs impact parameter ---------
        b = np.linspace(0.0, r_eff, impact_nodes)
        nu1 = np.array(
            [
                float(np.sum(h(dose_r) * r * w_r * _angular_fraction_inside(r, float(bi), r_eff)))
                / a_eff
                for bi in b
            ]
        )

        # --- corrections (factorised over pair centre-of-mass and separation) ---
        r_of_delta = delta_plane - 1.0 + np.exp(-delta_plane)  # R(Delta) >= 0
        i_r = 2.0 * math.pi * float(trapezoid(r_of_delta * sep, sep))
        i_s = 2.0 * math.pi * float(trapezoid((1.0 - np.exp(-delta_plane)) * sep, sep))
        beta_r = a_eff * i_r / norm
        suppression = 2.0 * math.pi * float(trapezoid((1.0 - np.exp(-2.0 * nu1)) * b, b))
        beta_s = i_s * suppression / norm

        raw = beta_a - beta_r - beta_s
        if raw < 0.0:
            # Not a rounding artifact: the second-order cluster term genuinely
            # goes negative once the track is deeply saturated, because above
            # D_t the n-track terms alternate in sign with roughly constant
            # magnitude (see docs/references/LEM_I_ScholzKraft.md), so
            # truncating at O(phi^2) is not an improvement scheme there. The
            # clamp keeps a usable, physically-signed number, but it is a
            # placeholder for a resummation or a saturated/unsaturated split --
            # so say so rather than returning 0.0 as though it were computed.
            # Measured to fire for uranium at 12-20 MeV/u and argon at 8-12
            # (argon is positive again at 16-20, i.e. the zero is a sign change,
            # not a limit).
            warnings.warn(
                f"two-track beta expansion returned {raw:.3e} < 0 for Z={ion.Z} at "
                f"{ion.energy_MeV_u:g} MeV/u and was clamped to 0. The O(phi^2) "
                "cluster truncation is not valid for a deeply saturated track, so "
                "this beta is a floor, not a value; treat the resulting curvature "
                "(and any sigma taken at a low survival level) as unreliable here.",
                UserWarning,
                stacklevel=2,
            )
        return max(raw, 0.0)

    def _beta_from_shared_model(self, model: str, ion: Ion, alpha_ion: float) -> float:
        """Dispatch the [`SHARED_BETA_MODELS`][], in one place for every version.

        Subclasses with a superset of models (LEM I) call this for the shared
        three and handle their own extras; subclasses whose effect model makes
        these undefined (LEM IV) must not call it at all.
        """
        if model == "slope_matching":
            return self._beta(alpha_ion)
        if model == "rapid_lem":
            return self.rapid_lem_beta(ion)
        if model == "scholz1997_two_track":
            return self.scholz1997_two_track_beta(ion)
        raise ValueError(
            f"beta_model={model!r} is not one of the shared models {SHARED_BETA_MODELS}"
        )

    def _beta_for_ion(self, ion: Ion, alpha_ion: float) -> float:
        """Version hook for ion beta models.

        The shared default is the later slope-matching approximation. LEM I also
        exposes the Scholz 1997 specific-energy stochastic approximations.
        """
        return self._beta(alpha_ion)

    # ---- BiophysicalModel hooks -----------------------------------------
    def _alpha_beta_ion(self, ion: Ion) -> LQParameters:
        """Single ion: ``alpha = initial_slope``, ``beta`` from the version's beta model.

        Mixed fields use the inherited [`BiophysicalModel.alpha_beta`][]
        template: dose-average ``alpha`` and ``sqrt(beta)`` (Zaider-Rossi /
        Kanai; AGENTS.md numerics), with binned components integrated over a
        log sub-grid of their energy range.
        """
        alpha = self.initial_slope(ion)
        return LQParameters(alpha=alpha, beta=self._beta_for_ion(ion, alpha))

    def _dose_weight_stopping_power(self, ion: Ion) -> float:
        """Fluence -> dose weights use this model's own [`stopping_power`][]."""
        return self.stopping_power.let_keV_um(ion)

    def reference(self) -> PhotonParameters:
        """The photon reference [`photon`][], shouldered above ``D_t`` (Scholz 1997 eq. 1).

        RBE from the inherited
        [`rbe_at_survival`][pyradbio.models.base.BiophysicalModel.rbe_at_survival] /
        [`rbe_at_dose`][pyradbio.models.base.BiophysicalModel.rbe_at_dose] therefore inverts the
        same shouldered-exponential photon curve the LEM is built on.
        """
        return self.photon

    def nuclear_dose_table(self, ion: Ion) -> NuclearDoseTable:
        r"""Build the [`NuclearDoseTable`][] for Kellerer weighted sampling.

        ``d_bar(b) = (1/A) int_A D(|x - c|) dx`` is evaluated by integrating the
        radial profile against the arc of each radius that falls inside the
        nucleus -- the same quadrature that yields the conditional radial CDF, so
        both come out of one pass.
        """
        r, dose, r_lo, r_max = self._two_track_dose_profile(ion, _TWO_TRACK_RADIAL_NODES)
        analytic = self._analytic_profile(ion)
        r_eff = self.cell.effective_nucleus_radius_um()

        b_max = r_eff + r_max
        wanted = math.ceil(b_max / (r_eff / _KELLERER_B_PER_R_EFF)) + 1
        n_b = min(max(_KELLERER_B_NODES, wanted), _KELLERER_B_MAX_NODES)
        if wanted > _KELLERER_B_MAX_NODES:
            raise ValueError(
                f"weighted sampling needs {wanted} impact-parameter nodes at "
                f"E={ion.energy_MeV_u} MeV/u (r_max={r_max:.0f} um) to keep "
                f"{_KELLERER_B_PER_R_EFF} across the nucleus, above the "
                f"{_KELLERER_B_MAX_NODES} cap. Raise _KELLERER_B_MAX_NODES if the "
                "memory is available; do not lower the resolution silently."
            )
        b_grid = np.linspace(0.0, b_max, n_b, dtype=np.float64)
        r_grid = np.geomspace(r_lo, r_max, _KELLERER_R_NODES, dtype=np.float64)
        bb, rr = b_grid[:, None], r_grid[None, :]

        half = _nuclear_arc_half_angle(bb, rr, r_eff)
        # d(dose integral)/dr inside the nucleus = D(r) * r * 2 phi_max(r)
        weight = self._two_track_dose_at(rr, r, dose, r_lo, r_max, analytic) * rr * 2.0 * half
        steps = 0.5 * (weight[:, 1:] + weight[:, :-1]) * np.diff(r_grid)[None, :]
        cum = np.concatenate([np.zeros((b_grid.size, 1)), np.cumsum(steps, axis=1)], axis=1)

        total = cum[:, -1:]
        safe = np.where(total > 0.0, total, 1.0)
        cdf = np.asarray(cum / safe, dtype=np.float64)

        # Pre-invert the CDF onto a uniform u grid, once. Interpolating in
        # *log* r makes this exact wherever D(r) follows the 1/r^2 law: there
        # the density goes as 1/r, so the CDF is linear in log r and the
        # inverse is exactly linear in u.
        u_grid = np.linspace(0.0, 1.0, _KELLERER_U_NODES, dtype=np.float64)
        log_r = np.log(r_grid)
        log_r_of_u = np.empty((b_grid.size, _KELLERER_U_NODES), dtype=np.float64)
        half_of_u = np.empty_like(log_r_of_u)
        for j in range(b_grid.size):
            log_r_of_u[j] = np.interp(u_grid, cdf[j], log_r)
            half_of_u[j] = np.interp(np.exp(log_r_of_u[j]), r_grid, half[j])

        return NuclearDoseTable(
            b_grid=b_grid,
            r_grid=r_grid,
            mean_dose_Gy=np.asarray(total[:, 0] / (math.pi * r_eff * r_eff), dtype=np.float64),
            radial_cdf=cdf,
            half_angle=np.asarray(half, dtype=np.float64),
            log_r_of_u=log_r_of_u,
            half_of_u=half_of_u,
        )

    def kellerer_field_integral(
        self,
        centers: npt.NDArray[np.float64],
        rng: np.random.Generator,
        table: NuclearDoseTable,
        r: npt.NDArray[np.float64],
        dose: npt.NDArray[np.float64],
        r_lo: float,
        r_max: float,
        n_field_points: int,
        analytic: AnalyticRadialProfile | None = None,
        b_cut_um: float | None = None,
        engine: Engine = "auto",
        dose_bg_Gy: float | None = None,
    ) -> float:
        r"""``N_leth`` for one track configuration by Kellerer *weighted* sampling.

        Kellerer 1985 section VI.A.2 (the method Scholz et al. 1997 cite as ref.
        [6] without restating): draw sample points with spatial density
        proportional to the local dose. Done as he prescribes -- select a track
        with probability proportional to its energy contribution ``d_bar(b_i)``,
        *then* a point around it -- the mixture density is exactly

        ``p(x) = d(x) / (A * D_tot)`` with ``D_tot = sum_i d_bar(b_i)``

        so the estimator collapses to ``N_leth = D_tot * <h(d)/d>``. ``D_tot`` is
        exact (no sampling), and ``h(d)/d`` is bounded between ``alpha_x`` and
        ``s_max``, so the variance does **not** grow with the track count.

        Two deviations from Kellerer made the *removed* uniform field integral
        cost ``O(n_bar^2)``: it selected a track *uniformly* rather than by its
        dose, and drew over the whole ``[r_min, r_max]`` shell rather than inside
        the nucleus -- so at carbon 100 only 0.12% of its track-targeted points
        landed inside and the rest were discarded, collapsing it to the uniform
        sampling Kellerer rejects as impracticable. That is why it is gone; see
        "Kellerer weighted sampling" in ``docs/references/LEM_I_ScholzKraft.md``.

        ``b_cut_um`` additionally treats tracks beyond that impact parameter as a
        *uniform background* ``D_bg`` rather than resolving them, removing the
        residual ``O(n_field_points * n)`` of evaluating ``d(x)``. This costs
        nothing in structure: the background enters as one extra source of weight
        ``D_bg`` sampled uniformly over the nucleus, so ``A p(x)`` is still
        ``d(x) / D_tot`` and the estimator is unchanged. The only approximation is
        taking the far field as flat, justified because its Poisson fluctuation is
        0.25-0.34% at the high-energy points where it matters (but 4.5% at carbon
        40, which is cheap anyway) -- so pick ``b_cut_um`` from the measured
        fluctuation, not a fixed radius. ``None`` resolves every track exactly.
        """
        n = centers.shape[0]
        if n == 0 or n_field_points <= 0:
            return 0.0

        b_i = np.hypot(centers[:, 0], centers[:, 1])
        d_bar = np.interp(b_i, table.b_grid, table.mean_dose_Gy)
        d_tot = float(d_bar.sum())
        if d_tot <= 0.0:
            return 0.0

        if dose_bg_Gy is not None:
            # The caller generated only the near tracks and supplied the far
            # field's dose directly, so there is nothing to filter.
            dose_bg = dose_bg_Gy
            centers_near, b_near, n_near = centers, b_i, n
            d_tot += dose_bg
        else:
            near = np.ones(n, dtype=bool) if b_cut_um is None else b_i < b_cut_um
            dose_bg = float(d_bar[~near].sum())
            centers_near = centers[near]
            b_near = b_i[near]
            d_bar = d_bar[near]
            n_near = centers_near.shape[0]

        # 1. select a source with probability proportional to its nuclear dose.
        #    The far field is one further source, of weight `dose_bg`, drawn
        #    uniformly over the nucleus; it sits at index `n_near`.
        # Inverse-CDF selection rather than rng.choice(p=...): same distribution,
        # without rebuilding and revalidating the weight vector on every call.
        cw = np.cumsum(np.concatenate([d_bar, [dose_bg]]))
        pick = np.searchsorted(cw, rng.random(n_field_points) * cw[-1])
        np.clip(pick, 0, n_near, out=pick)
        is_bg = pick == n_near
        n_bg = int(is_bg.sum())
        pts = np.empty((n_field_points, 2), dtype=np.float64)

        if n_bg:
            r_eff = self.cell.effective_nucleus_radius_um()
            rad_u = r_eff * np.sqrt(rng.random(n_bg))
            ang_u = 2.0 * math.pi * rng.random(n_bg)
            pts[is_bg] = np.stack([rad_u * np.cos(ang_u), rad_u * np.sin(ang_u)], axis=1)

        src = pick[~is_bg]
        if src.size:
            # 2. a radius around each, with density proportional to D(r) *inside* A
            b_src = b_near[src]
            slot = np.clip(np.searchsorted(table.b_grid, b_src) - 1, 0, table.b_grid.size - 1)
            # Gather + lerp on the pre-inverted table. The obvious version loops
            # over the distinct impact-parameter slots in Python, which was 78-96%
            # of this method's run time -- more than the compiled dose kernel.
            pos = rng.random(src.size) * (_KELLERER_U_NODES - 1)
            i0 = pos.astype(np.int64)
            np.clip(i0, 0, _KELLERER_U_NODES - 2, out=i0)
            frac = pos - i0
            lo_w, hi_w = 1.0 - frac, frac
            rad = np.exp(table.log_r_of_u[slot, i0] * lo_w + table.log_r_of_u[slot, i0 + 1] * hi_w)
            half = table.half_of_u[slot, i0] * lo_w + table.half_of_u[slot, i0 + 1] * hi_w
            # 3. an azimuth within the arc of that radius which lies inside A,
            #    measured from the track -> nucleus-centre direction. Every sample
            #    is therefore inside the nucleus by construction -- there is no
            #    accept/reject step and nothing is discarded, which is the point.
            phi = (2.0 * rng.random(src.size) - 1.0) * half
            # Unit vector from the track toward the nucleus centre. At b = 0 that
            # direction is undefined -- and dividing (0, 0) by the guard value 1.0
            # yields a *zero-length* vector, not a unit one, which collapses every
            # sample onto the track axis where the dose is the core maximum. The
            # circle is symmetric there, so any fixed direction is correct; pick
            # +x. (Zero measure in the production MC, whose centres are drawn from
            # a continuous distribution, but exact for a deliberately centred
            # track: it drove the single-traversal field integral 31% above N_1.)
            on_axis = b_src <= 0.0
            base = np.where(on_axis, 1.0, b_src)
            ux = np.where(on_axis, 1.0, -centers_near[src, 0] / base)
            uy = np.where(on_axis, 0.0, -centers_near[src, 1] / base)
            cos_p, sin_p = np.cos(phi), np.sin(phi)
            pts[~is_bg] = centers_near[src] + rad[:, None] * np.stack(
                [ux * cos_p - uy * sin_p, ux * sin_p + uy * cos_p], axis=1
            )

        if engine == "auto":
            engine = _auto_engine()
        if engine == "numba":
            a_x, b_x, d_t_ = self.photon.alpha_x, self.photon.beta_x, self.photon.D_t
            assert d_t_ is not None  # enforced in __post_init__
            total = _mc_kernel.kellerer_ratio_sum(
                np.ascontiguousarray(pts[:, 0]),
                np.ascontiguousarray(pts[:, 1]),
                np.ascontiguousarray(centers_near[:, 0]),
                np.ascontiguousarray(centers_near[:, 1]),
                dose_bg,
                np.log(r),
                np.log(dose),
                r_lo,
                r_max,
                analytic is not None,
                0.0 if analytic is None else analytic.core_dose_Gy,
                0.0 if analytic is None else analytic.k_Gy_um2,
                0.0 if analytic is None else analytic.r_min_um,
                0.0 if analytic is None else analytic.r_switch_um,
                0.0 if analytic is None else analytic.kernel_r_edge_um,
                0.0 if analytic is None else analytic.two_sigma_sq_um2,
                a_x,
                b_x,
                d_t_,
                self.photon.max_slope(),
                a_x * d_t_ + b_x * d_t_ * d_t_,
                *self._kernel_eta_table(),
            )
            return d_tot * float(total) / n_field_points

        # Chunked over sample points: the (points, tracks) block is the only
        # large allocation here, and n can reach 1e5-1e6 at high energy.
        step = max(1, _KELLERER_CHUNK_ELEMENTS // max(n_near, 1))
        total = 0.0
        for lo in range(0, n_field_points, step):
            blk = pts[lo : lo + step]
            if n_near:
                dx = blk[:, None, 0] - centers_near[None, :, 0]
                dy = blk[:, None, 1] - centers_near[None, :, 1]
                dist = np.sqrt(dx * dx + dy * dy)
                d_local = self._two_track_dose_at(dist, r, dose, r_lo, r_max, analytic).sum(axis=1)
            else:
                d_local = np.zeros(blk.shape[0], dtype=np.float64)
            d_local = d_local + dose_bg
            good = d_local > 0.0
            if good.any():
                total += float((self._photon_log_kill_array(d_local[good]) / d_local[good]).sum())
        return d_tot * total / n_field_points

    def scholz1997_traversal_conditioned_survival(
        self,
        ion: Ion,
        *,
        max_dose_Gy: float,
        n_batches: int = _TC_BATCHES,
        realizations_per_batch: int = _TC_REALIZATIONS_PER_BATCH,
        n_field_points: int = _MC_FIELD_POINTS,
        track_fraction: float = _MC_TRACK_FRACTION,
        seed: int = 0,
        max_mean_traversals: float = _MC_MAX_MEAN_TRAVERSALS,
    ) -> TraversalConditionedSurvival:
        r"""Build the traversal-conditioned survival curve out to ``max_dose_Gy``.

        Runs the full stochastic LEM once and returns
        [`TraversalConditionedSurvival`][], from which ``S(D)``, the exact
        ``d[-ln S]/dD`` and any iso-survival dose follow for *every* dose without
        further simulation. Prefer this over
        [`scholz1997_montecarlo_survival`][] whenever a whole curve -- and
        especially a slope or a crossing dose -- is wanted: that method resamples
        independently per dose, so its curve is neither monotone nor
        differentiable, and its deep-survival points are carried by a handful of
        rare zero-traversal histories.

        ``max_dose_Gy`` fixes the traversal grid ``k = 0..k_max``; the grid is
        padded well past ``lambda(max_dose_Gy)`` so the Poisson tail is
        negligible (check with
        [`TraversalConditionedSurvival.poisson_tail_mass`][]).

        Reference: Scholz & Kraft 1994/1996 explicit calculation.
        """
        if not math.isfinite(max_dose_Gy) or max_dose_Gy <= 0.0:
            raise ValueError("max_dose_Gy must be finite and > 0")
        if n_batches < 2:
            raise ValueError("n_batches must be >= 2 for a jackknife standard error")

        r, dose, r_lo, r_max = self._two_track_dose_profile(ion, _TWO_TRACK_RADIAL_NODES)
        analytic = self._analytic_profile(ion)
        r_eff = self.cell.effective_nucleus_radius_um()
        a_eff = math.pi * r_eff * r_eff
        d_1 = self.mean_traversal_dose_Gy(ion)
        b_max = r_eff + r_max

        lam_max = max_dose_Gy / (a_eff * d_1) * math.pi * b_max * b_max
        if lam_max > max_mean_traversals:
            raise ValueError(
                f"traversal-conditioned MC intractable at E={ion.energy_MeV_u} MeV/u, "
                f"D={max_dose_Gy} Gy: mean traversals {lam_max:.0f} exceeds "
                f"max_mean_traversals={max_mean_traversals:g} (r_max={r_max:.1f} um)."
            )
        k_max = (
            math.ceil(lam_max + _TC_POISSON_TAIL_SIGMA * math.sqrt(lam_max)) + _TC_POISSON_TAIL_PAD
        )

        profile = self._radial_logkill_profile(ion)
        b_grid = np.linspace(0.0, b_max, 512, dtype=np.float64)
        nu_grid = np.array(
            [self._impact_log_kill(float(b), profile, r_eff, r_max) for b in b_grid],
            dtype=np.float64,
        )

        rng = np.random.default_rng(seed)
        batch_g = np.empty((n_batches, k_max + 1), dtype=np.float64)
        for batch in range(n_batches):
            accumulator = np.zeros(k_max + 1)
            for _ in range(realizations_per_batch):
                rad = b_max * np.sqrt(rng.random(k_max))
                ang = 2.0 * math.pi * rng.random(k_max)
                centers = np.stack([rad * np.cos(ang), rad * np.sin(ang)], axis=1)
                accumulator += np.exp(
                    -self._montecarlo_prefix_field_integrals(
                        centers,
                        rng,
                        r,
                        dose,
                        r_lo,
                        r_max,
                        n_field_points,
                        track_fraction,
                        b_grid,
                        nu_grid,
                        analytic,
                    )
                )
            batch_g[batch] = accumulator / realizations_per_batch

        return TraversalConditionedSurvival(
            batch_g=batch_g,
            b_max_um=b_max,
            mean_traversal_dose_Gy=d_1,
            effective_area_um2=a_eff,
            realizations_per_batch=realizations_per_batch,
            max_supported_dose_Gy=max_dose_Gy,
        )

    def _mc_accumulate_kellerer(
        self,
        n_realizations: int,
        n_bar: float,
        b_max: float,
        table: NuclearDoseTable | None,
        r: npt.NDArray[np.float64],
        dose: npt.NDArray[np.float64],
        r_lo: float,
        r_max: float,
        n_field_points: int,
        rng: np.random.Generator,
        analytic: AnalyticRadialProfile | None,
        b_cut_um: float | None,
        engine: Engine = "auto",
    ) -> tuple[float, float]:
        """Accumulate ``sum S`` and ``sum S^2`` using Kellerer weighted sampling.

        Unlike the uniform path this needs no ``nu(b)`` control variate: with
        ``p(x) proportional to d(x)`` the whole integral is estimated at bounded
        variance, so there is no additive part to subtract off.
        """
        assert table is not None
        s_sum = 0.0
        s_sq_sum = 0.0

        # With a b_cut, generate ONLY the near tracks: touching all n_bar of them
        # each realization is O(n_bar) host work (the d_bar interpolation alone is
        # 10.9 ms at proton 70, against 7.1 ms for the whole compiled kernel), and
        # the far ones are not resolved anyway. Their summed dose is a compound
        # Poisson with closed-form moments, drawn directly.
        gen_radius = b_max
        bg_mean = bg_sd = 0.0
        if b_cut_um is not None and b_cut_um < b_max:
            gen_radius = b_cut_um
            fluence = n_bar / (math.pi * b_max * b_max)
            far = table.b_grid >= b_cut_um
            bb, dd = table.b_grid[far], table.mean_dose_Gy[far]
            ring = 2.0 * math.pi * bb
            bg_mean = fluence * float(np.trapezoid(dd * ring, bb))
            bg_sd = math.sqrt(max(fluence * float(np.trapezoid(dd * dd * ring, bb)), 0.0))
        lam = n_bar * (gen_radius / b_max) ** 2

        if engine == "auto":
            engine = _auto_engine()
        b_lo = float(table.b_grid[0])
        b_step = float(table.b_grid[-1] - b_lo) / (table.b_grid.size - 1)
        # The kernel indexes the impact-parameter grid arithmetically, so it is
        # only valid on the uniform grid `nuclear_dose_table` builds. A
        # hand-supplied table with any other spacing takes the Python path.
        uniform_b = bool(
            np.allclose(
                table.b_grid, b_lo + b_step * np.arange(table.b_grid.size), rtol=0, atol=1e-9
            )
        )
        if engine == "numba" and uniform_b:
            # One compiled call for the whole point. The loop below is correct but
            # spends ~1.9 ms per realization in Python/NumPy overhead -- Poisson
            # draw, centre placement, the d_bar interpolation, the selection
            # search, the table gather -- which is the entire cost wherever n_bar
            # is small, and still 32% of it at proton 70.
            a_x, b_x, d_t_ = self.photon.alpha_x, self.photon.beta_x, self.photon.D_t
            assert d_t_ is not None  # enforced in __post_init__
            return _mc_kernel.kellerer_accumulate(
                n_realizations,
                lam,
                gen_radius,
                bg_mean,
                bg_sd,
                self.cell.effective_nucleus_radius_um(),
                b_lo,
                b_step,
                table.mean_dose_Gy,
                table.log_r_of_u,
                table.half_of_u,
                n_field_points,
                np.log(r),
                np.log(dose),
                r_lo,
                r_max,
                analytic is not None,
                0.0 if analytic is None else analytic.core_dose_Gy,
                0.0 if analytic is None else analytic.k_Gy_um2,
                0.0 if analytic is None else analytic.r_min_um,
                0.0 if analytic is None else analytic.r_switch_um,
                0.0 if analytic is None else analytic.kernel_r_edge_um,
                0.0 if analytic is None else analytic.two_sigma_sq_um2,
                a_x,
                b_x,
                d_t_,
                self.photon.max_slope(),
                a_x * d_t_ + b_x * d_t_ * d_t_,
                *self._kernel_eta_table(),
            )

        for _ in range(n_realizations):
            n = int(rng.poisson(lam))
            dose_bg = max(bg_mean + bg_sd * rng.standard_normal(), 0.0) if bg_sd > 0.0 else 0.0
            if n == 0:
                if dose_bg <= 0.0:
                    s_sum += 1.0
                    s_sq_sum += 1.0
                    continue
                n_leth = self._photon_log_kill_array(np.array([dose_bg]))[0]
                s = math.exp(-float(n_leth))
                s_sum += s
                s_sq_sum += s * s
                continue
            rad = gen_radius * np.sqrt(rng.random(n))
            ang = 2.0 * math.pi * rng.random(n)
            centers = np.stack([rad * np.cos(ang), rad * np.sin(ang)], axis=1)
            n_leth = self.kellerer_field_integral(
                centers,
                rng,
                table,
                r,
                dose,
                r_lo,
                r_max,
                n_field_points,
                analytic,
                None,
                engine,
                dose_bg,
            )
            s = math.exp(-n_leth)
            s_sum += s
            s_sq_sum += s * s
        return s_sum, s_sq_sum

    @staticmethod
    def _mc_survival_from_sums(s_sum: float, s_sq_sum: float, n_realizations: int) -> float:
        """Population survival from the moment sums, delta-method (Jensen) de-biased.

        ``-ln(mean_s)`` over-estimates ``-ln S(D)`` by ~``Var(mean_s)/(2 mean_s^2)``;
        this bias grows with dose and would inflate the fitted beta, so ``exp(+bias)``
        on ``mean_s`` removes it to leading order (identical for every engine).
        """
        mean_s = s_sum / n_realizations
        var_s = max(s_sq_sum / n_realizations - mean_s * mean_s, 0.0)
        bias = var_s / (2.0 * n_realizations * mean_s * mean_s) if mean_s > 0.0 else 0.0
        return min(mean_s * math.exp(bias), 1.0)

    def scholz1997_montecarlo_survival(
        self,
        ion: Ion,
        doses_Gy: Iterable[float],
        *,
        n_realizations: int = _MC_REALIZATIONS,
        n_field_points: int = _MC_KELLERER_FIELD_POINTS,
        seed: int = 0,
        engine: Engine = "auto",
        b_cut_um: float | None = None,
    ) -> npt.NDArray[np.float64]:
        r"""Survival ``S(D)`` from the full stochastic LEM Monte Carlo.

        Reproduces the explicit calculation of Scholz & Kraft (1994/1996) and the
        Scholz 1997 "exact" path: for each dose, Poisson-distributed track
        traversals are simulated (mean ``n_bar = phi * pi * b_max^2`` with fluence
        ``phi = D / (A_eff Delta_z)`` and ``b_max = R_eff + r_max``), the per-cell
        lethal events ``N_leth`` give ``S_cell = exp(-N_leth)``, and ``S(D)`` is
        their population average.

        The field integral uses **Kellerer weighted sampling**
        ([`kellerer_field_integral`][]), which is the method Scholz et al. 1997
        cite for their own simulations -- sample points are drawn with spatial
        density proportional to the local dose. There is no choice of
        sampler: a uniform one would be the very scheme
        Kellerer 1985 calls "too wasteful to be practicable": it costs
        ``O(n_bar^2)`` and silently loses accuracy above ``n_bar ~ 1400`` where
        it thinned to under one sample per track. See ``kellerer_field_integral``
        for the estimator and ``b_cut_um`` for the near/far split.

        ``engine`` selects the execution backend (statistically equivalent):

        * ``"numba"`` -- the compiled kernel (`_mc_kernel`), which runs the
          whole realization; uses numba's per-thread RNG, so it is *not*
          bit-reproducible. Requires the ``fast`` extra.
        * ``"numpy"`` -- the vectorised reference loop; deterministic for a fixed
          ``seed``. Pin this for assertions that need an exact number.
        * ``"auto"`` -- ``"numba"`` if numba is importable, else ``"numpy"``.

        The choice is resolved by [`resolve_mc_engine`][].

        This is a *reference* path, slower than
        ``beta_model="scholz1997_two_track"``, but it carries no ``n_bar`` ceiling:
        precision is flat in the track count, measured to ``n_bar = 336 293``.

        Note the returned curve is resampled independently per dose, so it is
        neither monotone nor differentiable; for a whole curve, a slope, or an
        iso-survival dose prefer
        [`scholz1997_traversal_conditioned_survival`][].

        Raises:
            ValueError: ``engine`` is not one of
                [`ENGINES`][], or is ``"numba"``
                while numba is not importable.
        """
        engine = resolve_mc_engine(engine)
        if engine == "numba":
            # The kernel evaluates the photon response itself, so a version whose
            # response it cannot express would be silently simulated as LEM I.
            self._check_mc_response()

        r, dose, r_lo, r_max = self._two_track_dose_profile(ion, _TWO_TRACK_RADIAL_NODES)
        analytic = self._analytic_profile(ion)
        r_eff = self.cell.effective_nucleus_radius_um()
        a_eff = math.pi * r_eff * r_eff
        d_1 = self.mean_traversal_dose_Gy(ion)
        b_max = r_eff + r_max
        rng = np.random.default_rng(seed)
        table = self.nuclear_dose_table(ion)

        survivals: list[float] = []
        for dose_gy in doses_Gy:
            if dose_gy <= 0.0:
                survivals.append(1.0)
                continue
            n_bar = dose_gy / (a_eff * d_1) * math.pi * b_max * b_max
            s_sum, s_sq_sum = self._mc_accumulate_kellerer(
                n_realizations,
                n_bar,
                b_max,
                table,
                r,
                dose,
                r_lo,
                r_max,
                n_field_points,
                rng,
                analytic,
                b_cut_um,
                engine,
            )
            survivals.append(self._mc_survival_from_sums(s_sum, s_sq_sum, n_realizations))
        return np.array(survivals, dtype=np.float64)

    def _z_samples(
        self,
        lambdas: npt.NDArray[np.float64],
        b_max: float,
        table: NuclearDoseTable,
        r: npt.NDArray[np.float64],
        dose: npt.NDArray[np.float64],
        r_lo: float,
        r_max: float,
        n_field_points: int,
        realizations_per_mean: int,
        rng: np.random.Generator,
        analytic: AnalyticRadialProfile | None,
        b_cut_um: float | None,
        engine: Engine,
    ) -> tuple[npt.NDArray[np.float64], npt.NDArray[np.float64]]:
        """Pooled ``(z, S)`` samples over a set of Poisson means.

        Shares the near/far split of `_mc_accumulate_kellerer`: only tracks
        inside ``b_cut_um`` are generated, and the far field enters as one
        Gaussian draw with closed-form compound-Poisson moments. For ``z`` that
        split is what makes high energies reachable at all -- proton 200 MeV/u
        needs 2.3e8 traversals to reach 30 Gy, which cannot be enumerated, but
        only a few thousand of them are ever resolved.
        """
        r_eff = self.cell.effective_nucleus_radius_um()
        a_x, b_x, d_t_ = self.photon.alpha_x, self.photon.beta_x, self.photon.D_t
        assert d_t_ is not None  # enforced in __post_init__
        if engine == "auto":
            engine = _auto_engine()

        b_lo = float(table.b_grid[0])
        b_step = float(table.b_grid[-1] - b_lo) / (table.b_grid.size - 1)
        uniform_b = bool(
            np.allclose(
                table.b_grid, b_lo + b_step * np.arange(table.b_grid.size), rtol=0, atol=1e-9
            )
        )

        z_parts: list[npt.NDArray[np.float64]] = []
        s_parts: list[npt.NDArray[np.float64]] = []
        for n_bar in lambdas:
            gen_radius = b_max
            bg_mean = bg_sd = 0.0
            if b_cut_um is not None and b_cut_um < b_max:
                gen_radius = b_cut_um
                fluence = float(n_bar) / (math.pi * b_max * b_max)
                far = table.b_grid >= b_cut_um
                bb, dd = table.b_grid[far], table.mean_dose_Gy[far]
                ring = 2.0 * math.pi * bb
                bg_mean = fluence * float(np.trapezoid(dd * ring, bb))
                bg_sd = math.sqrt(max(fluence * float(np.trapezoid(dd * dd * ring, bb)), 0.0))
            lam = float(n_bar) * (gen_radius / b_max) ** 2

            if engine == "numba" and uniform_b:
                z_k, s_k = _mc_kernel.kellerer_z_samples(
                    realizations_per_mean,
                    lam,
                    gen_radius,
                    bg_mean,
                    bg_sd,
                    r_eff,
                    b_lo,
                    b_step,
                    table.mean_dose_Gy,
                    table.log_r_of_u,
                    table.half_of_u,
                    n_field_points,
                    np.log(r),
                    np.log(dose),
                    r_lo,
                    r_max,
                    analytic is not None,
                    0.0 if analytic is None else analytic.core_dose_Gy,
                    0.0 if analytic is None else analytic.k_Gy_um2,
                    0.0 if analytic is None else analytic.r_min_um,
                    0.0 if analytic is None else analytic.r_switch_um,
                    0.0 if analytic is None else analytic.kernel_r_edge_um,
                    0.0 if analytic is None else analytic.two_sigma_sq_um2,
                    a_x,
                    b_x,
                    d_t_,
                    self.photon.max_slope(),
                    a_x * d_t_ + b_x * d_t_ * d_t_,
                    *self._kernel_eta_table(),
                )
                z_parts.append(np.asarray(z_k, dtype=np.float64))
                s_parts.append(np.asarray(s_k, dtype=np.float64))
                continue

            zz = np.empty(realizations_per_mean)
            ssv = np.empty(realizations_per_mean)
            for k in range(realizations_per_mean):
                n = int(rng.poisson(lam))
                dose_bg = max(bg_mean + bg_sd * rng.standard_normal(), 0.0) if bg_sd > 0.0 else 0.0
                if n == 0:
                    zz[k] = dose_bg
                    ssv[k] = (
                        1.0
                        if dose_bg <= 0.0
                        else math.exp(-self._photon_log_kill_array(np.array([dose_bg]))[0])
                    )
                    continue
                rad = gen_radius * np.sqrt(rng.random(n))
                ang = 2.0 * math.pi * rng.random(n)
                centers = np.stack([rad * np.cos(ang), rad * np.sin(ang)], axis=1)
                zz[k] = float(np.interp(rad, table.b_grid, table.mean_dose_Gy).sum()) + dose_bg
                ssv[k] = math.exp(
                    -self.kellerer_field_integral(
                        centers,
                        rng,
                        table,
                        r,
                        dose,
                        r_lo,
                        r_max,
                        n_field_points,
                        analytic,
                        None,
                        engine,
                        dose_bg,
                    )
                )
            z_parts.append(zz)
            s_parts.append(ssv)
        return np.concatenate(z_parts), np.concatenate(s_parts)

    def scholz1997_montecarlo_beta(
        self,
        ion: Ion,
        *,
        doses_Gy: Iterable[float] = _MC_FIT_DOSES_GY,
        n_realizations: int = _MC_REALIZATIONS,
        n_field_points: int = _MC_KELLERER_FIELD_POINTS,
        seed: int = 0,
        engine: Engine = "numpy",
        b_cut_um: float | None = None,
    ) -> float:
        """Dose-domain ``beta`` from the stochastic LEM MC (Douglas-Fowler fit).

        Runs [`scholz1997_montecarlo_survival`][] on ``doses_Gy`` and fits
        ``-ln S(D)/D = alpha + beta D`` (Scholz 1997 eq. 8), returning ``beta``.
        Because ``beta`` is a small curvature on a dominant linear term the MC
        value carries statistical noise (~1e-3) and mild fit-range dependence;
        raise ``n_realizations`` or average over seeds for a tighter estimate.
        ``engine`` selects the backend (see [`scholz1997_montecarlo_survival`][]).

        The fit-range dependence is not merely "mild" when comparing against
        [`scholz1997_two_track_beta`][]: that method returns the exact
        ``O(phi^2)`` cluster coefficient, while this one returns a fitted
        effective coefficient over ``doses_Gy``. See its docstring -- the ratio
        runs 0.995 / 0.986 / 0.971 for windows 0.25-1.5 / 0.5-3.0 / 1.0-6.0 Gy at
        proton 7 MeV/u. Quote the window with the number.
        """
        doses = np.asarray(list(doses_Gy), dtype=np.float64)
        log_kills = -np.log(
            self.scholz1997_montecarlo_survival(
                ion,
                doses,
                n_realizations=n_realizations,
                n_field_points=n_field_points,
                seed=seed,
                engine=engine,
                b_cut_um=b_cut_um,
            )
        )
        design = np.vstack([np.ones_like(doses), doses]).T
        beta = float(np.linalg.lstsq(design, log_kills / doses, rcond=None)[0][1])
        return max(beta, 0.0)

    def _montecarlo_prefix_field_integrals(
        self,
        centers: npt.NDArray[np.float64],
        rng: np.random.Generator,
        r: npt.NDArray[np.float64],
        dose: npt.NDArray[np.float64],
        r_lo: float,
        r_max: float,
        n_field_points: int,
        track_fraction: float,
        b_grid: npt.NDArray[np.float64],
        nu_grid: npt.NDArray[np.float64],
        analytic: AnalyticRadialProfile | None = None,
        engine: Engine = "auto",
    ) -> npt.NDArray[np.float64]:
        r"""``N_leth`` for *every* traversal-count prefix of ``centers``.

        Uses the hybrid uniform/near-track mixture rather than Kellerer weighted
        sampling, and has to: the sample points are shared across every prefix,
        so they cannot be drawn proportional to a dose field that depends on which
        prefix is being scored. Its ``n_bar`` stays far below the thinning regime
        (17-569 across the 1996 grid), so the accuracy problem that removed the
        per-dose uniform sampler does not reach here.

        Evaluated for all ``k = 0..n`` at once. The additive single-track part is a running
        sum of ``nu(b_i)``; the overlap excess for prefix ``k`` follows from the
        cumulative per-track dose, so ``np.cumsum`` over the ``(points, tracks)``
        block yields every prefix at the asymptotic cost of the ``k = n`` case
        alone. That is what makes one sweep enough for the whole survival curve.

        Field points are drawn once from the mixture density of the *full* track
        set. That density is what the estimator divides out, so every prefix
        stays unbiased; short prefixes merely spend some samples away from their
        own tracks, where their excess is zero anyway.
        """
        r_eff = self.cell.effective_nucleus_radius_um()
        a_eff = math.pi * r_eff * r_eff
        n = centers.shape[0]
        ln_ratio = math.log(r_max / r_lo)

        c_norm = np.hypot(centers[:, 0], centers[:, 1])
        linear = np.concatenate([[0.0], np.cumsum(np.interp(c_norm, b_grid, nu_grid))])
        if n == 0:
            return np.asarray(linear, dtype=np.float64)

        n_uniform = round((1.0 - track_fraction) * n_field_points)
        n_track = n_field_points - n_uniform
        chunks: list[npt.NDArray[np.float64]] = []
        if n_uniform > 0:
            rad = r_eff * np.sqrt(rng.random(n_uniform))
            ang = 2.0 * math.pi * rng.random(n_uniform)
            chunks.append(np.stack([rad * np.cos(ang), rad * np.sin(ang)], axis=1))
        if n_track > 0:
            # Pick a uniformly random track per point rather than a fixed number
            # of points per track. The mixture density is identical -- it is the
            # ``track_fraction / n`` sum divided out below -- but it survives
            # ``n > n_track``, where per-track stratification would drop the
            # importance-sampled component entirely. The traversal grid is padded
            # past ``lambda``, so that regime is reached sooner here than in the
            # per-dose path.
            idx = rng.integers(0, n, n_track)
            rad = r_lo * np.exp(ln_ratio * rng.random(n_track))
            ang = 2.0 * math.pi * rng.random(n_track)
            base = centers[idx]
            chunks.append(base + np.stack([rad * np.cos(ang), rad * np.sin(ang)], axis=1))
        pts = np.concatenate(chunks, axis=0)

        if engine == "auto":
            engine = _auto_engine()
        if engine == "numba":
            a_x, b_x, d_t_ = self.photon.alpha_x, self.photon.beta_x, self.photon.D_t
            assert d_t_ is not None  # enforced in __post_init__
            value_sum, value_sq_sum = _mc_kernel.prefix_field_integrals(
                np.ascontiguousarray(pts[:, 0]),
                np.ascontiguousarray(pts[:, 1]),
                np.ascontiguousarray(centers[:, 0]),
                np.ascontiguousarray(centers[:, 1]),
                np.log(r),
                np.log(dose),
                r_lo,
                r_max,
                analytic is not None,
                0.0 if analytic is None else analytic.core_dose_Gy,
                0.0 if analytic is None else analytic.k_Gy_um2,
                0.0 if analytic is None else analytic.r_min_um,
                0.0 if analytic is None else analytic.r_switch_um,
                0.0 if analytic is None else analytic.kernel_r_edge_um,
                0.0 if analytic is None else analytic.two_sigma_sq_um2,
                r_eff,
                a_eff,
                track_fraction,
                ln_ratio,
                a_x,
                b_x,
                d_t_,
                self.photon.max_slope(),
                a_x * d_t_ + b_x * d_t_ * d_t_,
                *self._kernel_eta_table(),
                _TC_PREFIX_BLOCKS,
            )
            total = pts.shape[0]
            mean = value_sum / total
            variance = (value_sq_sum / total - mean * mean) / total
            return np.asarray(
                linear + np.concatenate([[0.0], (mean + 0.5 * variance / a_eff) / a_eff]),
                dtype=np.float64,
            )

        # Stream the (points x tracks) block so memory stays bounded when a
        # high-energy heavy ion needs thousands of traversals.
        block_rows = max(1, _TC_FIELD_POINT_CHUNK // n)
        value_sum = np.zeros(n)
        value_sq_sum = np.zeros(n)
        for start in range(0, pts.shape[0], block_rows):
            block = pts[start : start + block_rows]
            inside = (block[:, 0] ** 2 + block[:, 1] ** 2) <= r_eff * r_eff
            diff = block[:, None, :] - centers[None, :, :]
            dist = np.sqrt(diff[:, :, 0] ** 2 + diff[:, :, 1] ** 2)
            dose_i = self._two_track_dose_at(dist, r, dose, r_lo, r_max, analytic)
            # Column k-1 is the prefix of the first k tracks.
            excess = self._photon_log_kill_array(np.cumsum(dose_i, axis=1)) - np.cumsum(
                self._photon_log_kill_array(dose_i), axis=1
            )
            p = np.where(inside, (1.0 - track_fraction) / a_eff, 0.0)
            in_shell = (dist >= r_lo) & (dist <= r_max)
            p = p + np.where(
                in_shell,
                track_fraction / n / (2.0 * math.pi * ln_ratio * np.maximum(dist, r_lo) ** 2),
                0.0,
            ).sum(axis=1)
            good = p > 0.0
            vals = np.where((inside & good)[:, None], excess / np.where(good, p, 1.0)[:, None], 0.0)
            value_sum += vals.sum(axis=0)
            value_sq_sum += (vals * vals).sum(axis=0)

        total = pts.shape[0]
        mean = value_sum / total
        variance = (value_sq_sum / total - mean * mean) / total
        return np.asarray(
            linear + np.concatenate([[0.0], (mean + 0.5 * variance / a_eff) / a_eff]),
            dtype=np.float64,
        )
