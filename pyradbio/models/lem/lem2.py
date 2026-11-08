"""LEM II: cluster extension + refined radial dose.

Elsasser & Scholz 2007 (doi:10.1667/RR0467.1). Two additions over LEM I:

1. **Cluster extension.** At high local dose, single-strand breaks (SSBs) become
   dense enough that two SSBs within a short stagger ``h`` (<= 25 bp) on the DNA
   form an *additional* double-strand break (DSB). This scales the photon
   survival's exponential tail above ``D_t`` by a damage enhancement ``eta(D)``
   (eqs. 9-10): ``S(D) = S_t * exp(-s_max * (eta(D)*D - D_t))`` for ``D > D_t``.
   [`ClusterModel`][] computes ``eta(D)`` from a Monte-Carlo base-distribution
   of SSBs (SSB=1250/Gy/cell, DSB=30/Gy/cell).

   Note ``eta`` **multiplies the dose**; it is *not* integrated over it. Reading
   eq. 10 instead as an enhanced local slope (``d(-ln S_x)/dD = s_max * eta(D)``,
   i.e. ``s_max * integral eta dD'``) is a different -- and wrong -- function: it
   misses the paper's own Fig. 3 by 22.7% RMS where the literal form lands within
   2.7%. The decisive check is parameter-free: at ``D ~ 1e6`` Gy the digitized
   Fig. 3a solid/dashed ratio is 13.05, which is just ``eta(1e6) = 12.7`` -- the
   clustered curve is the straight tail *times* ``eta(D)``, which no integral can
   reproduce. See ``pyradbio/benchmark/elsasser2007.py`` and the Fig. 3 panel of
   ``validation/elsasser2007_lem2.py``.
2. **Refined radial dose.** ``r_min`` drops from ~10 nm to 0.3 nm and the profile
   is convolved with a sigma=4 nm Gaussian (radical diffusion). Both live in
   [`ScholzKraftRadialDose`][] (``.lem2``).

The refit reduces ``D_t`` per cell line (V79 30 -> 7.5 Gy); both the cluster
extension and the refined radial dose are needed -- combining them without
reducing ``D_t`` overshoots the RBE (Elsasser & Scholz 2007 Fig. 6 thin dashed).
"""

from __future__ import annotations

import math
from dataclasses import dataclass, field
from functools import cache
from typing import Protocol, runtime_checkable

import numpy as np
import numpy.typing as npt

from pyradbio.core.photon import ClusterParameters
from pyradbio.core.radiation import Ion
from pyradbio.models.base import check_choice
from pyradbio.models.lem.base import SHARED_BETA_MODELS, LEMBase, SharedBetaModel
from pyradbio.track.base import RadialDoseModel
from pyradbio.track.scholz_kraft import ScholzKraftRadialDose

# Damage-enhancement table: log-spaced local doses [Gy] spanning the onset
# (~10^3 Gy) through the maximum (~5x10^5 Gy) and back toward eta->1 at very high
# dose, where direct DSBs again dominate (Elsasser & Scholz 2007 Fig. 2).
_ETA_DOSE_LO_GY: float = 1.0
_ETA_DOSE_HI_GY: float = 1.0e9
# Sampling grid for cluster models whose eta is a closed form rather than a
# table, so the compiled kernel can interpolate it. Linear-in-log10 error falls
# as the square of the node spacing: over 1-1e9 Gy, 120 nodes/decade leave
# 1.2e-4 relative error in eta and ~1e-3 in the response, which
# `_check_mc_response` rejects; 1920/decade leave 4.6e-7 and ~5e-6, i.e.
# negligible against a Monte Carlo whose own error is ~1%. 276 kB, built once.
_KERNEL_ETA_LO_GY: float = 1.0
_KERNEL_ETA_HI_GY: float = 1.0e9
_KERNEL_ETA_NODES: int = 17281
_ETA_GRID_NODES: int = 180

# Monte-Carlo base distribution: number of stagger-width (``h`` bp) bins of the
# representative genome segment, and trials averaged for a smooth curve. A
# clustered DSB is any bin holding >= 2 SSBs (further breaks in it are discarded,
# Elsasser 2007 p.321), so the count saturates at one per bin and n_bins also
# caps the table at its physical maximum density (genome_size / h).
_MC_BINS: int = 100000
_MC_TRIALS: int = 4
# Above this mean SSB count per bin essentially every bin already holds >= 2
# SSBs; use the closed-form Poisson P(>=2) instead of sampling huge point sets
# (Monte-Carlo and the analytic limit agree to <0.1% there).
_MC_SATURATION_MU: float = 20.0


@runtime_checkable
class ClusterResponse(Protocol):
    """The damage-enhancement ``eta(D)`` interface LEM II/III consume.

    Elsasser & Scholz 2007 eq. 10 scales the photon tail's dose by ``eta(D)``;
    everything [`LEM2`][] needs from a cluster model is listed here, so any
    object implementing it can be injected as ``LEM2(cluster=...)``. Both
    bundled models -- the Monte-Carlo [`ClusterModel`][] and the closed-form
    [`Friedrich2015ClusterModel`][] -- satisfy it.
    """

    def eta(self, dose_Gy: float) -> float:
        """Damage enhancement ``eta(D)`` at one local dose [Gy] (>= 1)."""
        ...

    def eta_array(self, dose_Gy: npt.NDArray[np.float64]) -> npt.NDArray[np.float64]:
        """Vectorised [`eta`][]; the pair-overlap integrals call it on ~1e5 doses."""
        ...

    def kernel_eta_table(self) -> tuple[npt.NDArray[np.float64], npt.NDArray[np.float64]]:
        """``(log10 dose, eta)`` nodes for the compiled Monte-Carlo kernel."""
        ...

    @property
    def eta_max(self) -> float:
        """Peak enhancement ``max_D eta(D)``; caps the rapidLEM initial slope."""
        ...


@dataclass(frozen=True, slots=True)
class ClusterModel:
    """Damage enhancement ``eta(D)`` from clustered SSBs (Elsasser 2007 eqs. 9-10).

    The genome carries ``ssb_per_gy_cell`` SSBs and ``dsb_per_gy_cell`` direct
    DSBs per Gy. Following Elsasser & Scholz 2007 (p.321), the SSBs are
    distributed at random over the ``genome_size_bp`` bases ("by randomly choosing
    the damaged base", so only the genomic size is needed); whenever two SSBs fall
    within the stagger window ``h = stagger_bp`` they are scored as one additional
    DSB, and any further break inside that already-clustered window is discarded.
    Hence

        eta(D) = [N(DSB) + N(SSB + SSB -> DSB)] / N(DSB)                    (eq. 9)

    with ``N(DSB) = dsb_per_gy_cell * D``. ``N(SSB+SSB)`` is estimated by a
    seedable Monte-Carlo: SSBs are dropped into ``h``-wide bins of a
    representative segment and a clustered DSB is registered for every bin holding
    >= 2 SSBs. The "discard further breaks" rule caps each bin at one DSB, so the
    mean per bin is the Poisson ``P(>=2) = 1 - e^{-mu}(1 + mu)`` with
    ``mu = ssb_per_gy_cell * D * h / genome_size_bp`` SSBs per bin; scaled to the
    genome this is ``(genome_size_bp / h) * P(>=2)`` clustered DSBs.

    ``eta`` starts at 1 (no SSB coincidences), rises through an onset near
    ``10^3`` Gy, peaks where ``mu ~ 1.8`` (the saturation of ``N(SSB+SSB)``, around
    ``5x10^5`` Gy, Fig. 2), then falls back toward 1 as the linearly-growing direct
    DSBs again dominate. The curve is tabulated once at construction and
    interpolated.
    """

    params: ClusterParameters = field(default_factory=ClusterParameters)
    rng_seed: int = 0
    n_bins: int = _MC_BINS
    n_trials: int = _MC_TRIALS
    _log_dose: npt.NDArray[np.float64] = field(init=False, repr=False, compare=False)
    _eta: npt.NDArray[np.float64] = field(init=False, repr=False, compare=False)

    def __post_init__(self) -> None:
        doses: npt.NDArray[np.float64] = np.logspace(
            math.log10(_ETA_DOSE_LO_GY),
            math.log10(_ETA_DOSE_HI_GY),
            _ETA_GRID_NODES,
            dtype=np.float64,
        )
        rng = np.random.default_rng(self.rng_seed)
        eta = np.array([self._eta_montecarlo(float(d), rng) for d in doses], dtype=np.float64)
        # Frozen: the table is built once here and is read-only afterwards.
        object.__setattr__(self, "_log_dose", np.log10(doses))
        object.__setattr__(self, "_eta", eta)

    # ---- Monte-Carlo damage enhancement --------------------------------
    def _clustered_fraction(self, mu: float, rng: np.random.Generator) -> float:
        """Mean fraction of ``h``-bins holding >= 2 SSBs at mean occupancy ``mu``.

        ``mu`` is the expected number of SSBs per ``h``-wide bin. For dense tracks
        (``mu`` large) essentially every bin already clusters, so the closed-form
        Poisson ``P(>=2) = 1 - e^{-mu}(1 + mu)`` is used directly; otherwise the
        bins are sampled and bins with two or more SSBs are counted (the "discard
        further breaks" rule caps each at one clustered DSB).
        """
        if mu >= _MC_SATURATION_MU:
            return 1.0 - math.exp(-mu) * (1.0 + mu)
        total = 0.0
        for _ in range(self.n_trials):
            n = int(rng.poisson(mu * self.n_bins))
            counts = np.bincount(rng.integers(0, self.n_bins, size=n), minlength=self.n_bins)
            total += float(np.count_nonzero(counts >= 2)) / self.n_bins
        return total / self.n_trials

    def _eta_montecarlo(self, dose_Gy: float, rng: np.random.Generator) -> float:
        p = self.params
        n_dsb_direct = p.dsb_per_gy_cell * dose_Gy
        if n_dsb_direct <= 0.0:
            return 1.0
        mu = (p.ssb_per_gy_cell * dose_Gy / p.genome_size_bp) * p.stagger_bp
        clustered_frac = self._clustered_fraction(mu, rng)
        n_dsb_clustered = clustered_frac * (p.genome_size_bp / p.stagger_bp)
        return 1.0 + n_dsb_clustered / n_dsb_direct

    # ---- public interface ----------------------------------------------
    def eta(self, dose_Gy: float) -> float:
        """Damage enhancement ``eta(D)`` (1 below the tabulated onset)."""
        if dose_Gy <= _ETA_DOSE_LO_GY:
            return 1.0
        return float(np.interp(math.log10(dose_Gy), self._log_dose, self._eta))

    def eta_array(self, dose_Gy: npt.NDArray[np.float64]) -> npt.NDArray[np.float64]:
        """Vectorised [`eta`][], for the array photon response.

        The pair-overlap integrals evaluate the log-kill on ~1e5-element dose
        grids, so a scalar ``eta`` in that loop dominates the runtime. Same
        interpolation, same below-onset floor of 1.
        """
        dose = np.asarray(dose_Gy, dtype=np.float64)
        with np.errstate(divide="ignore"):
            interpolated = np.interp(np.log10(dose), self._log_dose, self._eta)
        return np.asarray(np.where(dose <= _ETA_DOSE_LO_GY, 1.0, interpolated), dtype=np.float64)

    def kernel_eta_table(self) -> tuple[npt.NDArray[np.float64], npt.NDArray[np.float64]]:
        """``(log10 dose, eta)`` for the compiled Monte-Carlo kernel.

        The kernel interpolates linearly in ``log10 D`` and clamps flat outside,
        which is exactly what [`eta`][] does -- so handing it this table costs
        no accuracy at all. The first node is ``(1 Gy, 1.0)``, so the clamp below
        it *is* the below-onset floor.
        """
        return self._log_dose, self._eta

    @property
    def eta_max(self) -> float:
        """Peak damage enhancement ``max_D eta(D)`` (>= 1).

        ``s_max * eta_max`` bounds the photon log-kill *per unit dose* and is the
        saturation cap for the rapidLEM (``lethal_central``) initial slope: under
        eq. 10 (see [`LEM2.photon_log_kill`][]),
        ``N(D)/D = s_max*eta(D) - beta*D_t^2/D <= s_max*eta_max`` for ``D > D_t``,
        and below ``D_t`` the LQ gives ``N(D)/D = alpha + beta*D < s_max``. So
        ``N(D) <= s_max*eta_max*D`` pointwise, hence ``N_1 <= s_max*eta_max*D_1``.
        """
        return float(self._eta.max())


@dataclass(frozen=True, slots=True)
class Friedrich2015ClusterModel:
    """Closed-form ``eta(D)`` from Friedrich, Durante & Scholz 2015.

    doi:10.1093/rpd/ncv147, eqs. (2)-(8). Same group, same endpoint and the same
    inputs as [`ClusterModel`][] (``alpha_SSB = 1250``, ``alpha_DSB = 30``,
    ``t = 25`` bp), but a *different and more detailed* coincidence rule, and
    analytic rather than Monte-Carlo -- so it is an independent check on our
    reading of Elsasser & Scholz 2007 eq. (9), not a reimplementation of it.

    With SSB/DSB line densities ``rho = alpha D / L_gen``, an SSB is converted
    into an inter-track DSB with probability ``P = P_noDSB * P_cand * P_conv``:

    ```text
    P_noDSB = exp(-rho_DSB t)              (4)  no intra-track DSB within t/2
    P_cand  = (1 - exp(-rho_SSB t)) / 2    (5)  next SSB within t, opposite strand
    P_conv  = 1 / (1 + P_cand)             (6)  not already spent on another DSB
    eta     = 1 + P alpha_SSB / alpha_DSB  (8)
    ```

    The three factors are what our binned rule leaves out: it neither excludes
    SSBs shadowed by a real DSB nor stops one SSB from serving two DSBs, and it
    has no explicit opposite-strand factor.

    **Why it is worth carrying.** It settles whether our ``eta`` is "too low"
    against Elsasser 2007 Fig. 2. Over the doses a track actually produces
    (<= 1e5 Gy in a carbon track centre, per this paper's own Discussion) the two
    agree to 3.0% RMS and both sit 0.5-4.7% *under* the digitized Fig. 2. They
    part company only above ~1e6 Gy -- +37% vs -42% against Fig. 2, 140% apart
    from each other -- which is exactly the regime the paper says is not pinned
    down ("it is not clear how to exactly implement the conversion of candidates
    ... the high dose part around the maximum of eta and beyond is affected,
    while the course of the curve at lower doses remains rather unchanged").
    So the Fig. 2 peak is the outlier, not our curve; see
    ``docs/references/LEM_II_Elsasser2007.md``.

    Note ``genome_size_bp``: this paper uses ``L_gen = 5.4e9`` bp where
    [`ClusterParameters`][] defaults to ``6.0e9``. That shifts the dose axis
    but not the peak height, which is ``1 + (alpha_SSB/alpha_DSB) max_D P`` with
    ``L_gen`` cancelling out of ``P``'s two exponents jointly.
    """

    params: ClusterParameters = field(default_factory=ClusterParameters)

    def eta(self, dose_Gy: float) -> float:
        """Damage enhancement ``eta(D)`` (eqs. 2-8); exactly 1 at zero dose."""
        if dose_Gy <= 0.0:
            return 1.0
        p = self.params
        rho_ssb = p.ssb_per_gy_cell * dose_Gy / p.genome_size_bp
        rho_dsb = p.dsb_per_gy_cell * dose_Gy / p.genome_size_bp
        p_no_dsb = math.exp(-rho_dsb * p.stagger_bp)
        p_cand = 0.5 * (1.0 - math.exp(-rho_ssb * p.stagger_bp))
        p_conv = 1.0 / (1.0 + p_cand)
        return 1.0 + p_no_dsb * p_cand * p_conv * p.ssb_per_gy_cell / p.dsb_per_gy_cell

    def eta_array(self, dose_Gy: npt.NDArray[np.float64]) -> npt.NDArray[np.float64]:
        """Vectorised [`eta`][] (the closed form is elementwise already)."""
        dose = np.asarray(dose_Gy, dtype=np.float64)
        p = self.params
        rho_ssb = p.ssb_per_gy_cell * dose / p.genome_size_bp
        rho_dsb = p.dsb_per_gy_cell * dose / p.genome_size_bp
        p_no_dsb = np.exp(-rho_dsb * p.stagger_bp)
        p_cand = 0.5 * (1.0 - np.exp(-rho_ssb * p.stagger_bp))
        p_conv = 1.0 / (1.0 + p_cand)
        eta = 1.0 + p_no_dsb * p_cand * p_conv * p.ssb_per_gy_cell / p.dsb_per_gy_cell
        return np.asarray(np.where(dose <= 0.0, 1.0, eta), dtype=np.float64)

    def kernel_eta_table(self) -> tuple[npt.NDArray[np.float64], npt.NDArray[np.float64]]:
        """``(log10 dose, eta)`` for the compiled Monte-Carlo kernel.

        Unlike [`ClusterModel`][] this eta is a closed form, so the table is a
        *sampling* of it and the kernel's linear-in-``log10 D`` interpolation is
        an approximation. eta is smooth and the grid is dense (see
        ``_KERNEL_ETA_NODES``); ``tests/models/lem/test_mc_eta_kernel.py`` pins
        the resulting error on the sampled response.
        """
        doses = np.logspace(
            math.log10(_KERNEL_ETA_LO_GY),
            math.log10(_KERNEL_ETA_HI_GY),
            _KERNEL_ETA_NODES,
            dtype=np.float64,
        )
        return np.log10(doses), self.eta_array(doses)

    def beta_dsb_per_gy2(self) -> float:
        """Quadratic DSB-induction coefficient ``alpha_SSB^2 t / (2 L_gen)`` (eq. 10).

        The small-dose expansion of [`eta`][]: two independent SSBs are needed,
        hence ``alpha_SSB^2``; ``t / L_gen`` is the chance the partner lands close
        enough; the ``1/2`` is because two SSBs make one DSB. ``alpha_DSB`` over
        this is the local dose where inter- and intra-track DSBs balance
        (``eta = 2``), which the paper quotes as ~8300 Gy for its parameter set.
        """
        p = self.params
        return p.ssb_per_gy_cell**2 * p.stagger_bp / (2.0 * p.genome_size_bp)

    @property
    def eta_max(self) -> float:
        """Peak damage enhancement, found on a log-dose scan (see [`ClusterModel`][])."""
        doses = np.logspace(0.0, 9.0, 20_000)
        return float(max(self.eta(float(d)) for d in doses))


@cache
def _default_cluster_model(
    params: ClusterParameters, rng_seed: int, n_bins: int, n_trials: int
) -> ClusterModel:
    """Memoised default [`ClusterModel`][].

    The eta table is a seeded Monte-Carlo that depends only on its
    ``ClusterParameters`` (and RNG/grid settings), never on the photon or cell,
    yet it costs ~0.3 s to build. Every LEM II/III instance that does not supply
    its own cluster model shares one memoised table (keyed on the hashable frozen
    [`ClusterParameters`][]), so building a per-cell-line table set does not
    rebuild the identical curve tens of times. The returned model is read-only
    after construction, so sharing one instance is safe.
    """
    return ClusterModel(params=params, rng_seed=rng_seed, n_bins=n_bins, n_trials=n_trials)


@dataclass(frozen=True, slots=True)
class LEM2(LEMBase):
    """LEM II: refined radial dose (``r_min=0.3 nm`` + radical diffusion) plus the
    high-dose cluster enhancement ``eta(D)`` on the photon tail.

    Set the *refit* (reduced) ``D_t`` on the photon parameters (V79 7.5 Gy, not
    the LEM I 30 Gy). ``cluster`` is any [`ClusterResponse`][]; when omitted,
    a [`ClusterModel`][] is built from ``photon.cluster`` (or default
    [`ClusterParameters`][]).
    """

    cluster: ClusterResponse | None = None
    """Optional injection; None derives the response from photon.cluster."""
    beta_model: SharedBetaModel = "slope_matching"
    resolved_cluster: ClusterResponse = field(init=False, repr=False, compare=False)
    """Active eq.-10 response, recomputed from constructor inputs on replace."""

    def _kernel_eta_table(self) -> tuple[npt.NDArray[np.float64], npt.NDArray[np.float64]]:
        """eq. 10's ``eta(D)``, so the compiled MC kernel simulates LEM II/III.

        Without this the kernel would fall back to the LEM I response and return
        an un-enhanced answer under a LEM II/III label; ``_check_mc_response``
        turns that into an error rather than a wrong number.
        """
        return self._cluster().kernel_eta_table()

    def __post_init__(self) -> None:
        LEMBase.__post_init__(self)
        check_choice("beta_model", self.beta_model, SHARED_BETA_MODELS)
        cluster = self.cluster
        if cluster is None:
            params = self.photon.cluster if self.photon.cluster is not None else ClusterParameters()
            cluster = _default_cluster_model(params, 0, _MC_BINS, _MC_TRIALS)
        object.__setattr__(self, "resolved_cluster", cluster)

    def _beta_for_ion(self, ion: Ion, alpha_ion: float) -> float:
        """``beta_I`` per [`beta_model`][]; all three come from [`LEMBase`][].

        LEM II/III inherit the whole family unchanged because their lethal-event
        density is still a *pointwise* function of the local dose -- eq. 10 scales
        the dose by ``eta(D)``, and ``eta`` depends on nothing but ``D``. The only
        thing that had to be version-aware is the photon response itself, and both
        its scalar and array forms are overridden together above.

        ``"rapid_lem"`` uses the un-enhanced ``s_max`` for ``beta_z``, not the
        ``s_max * eta_max`` of `_max_local_slope`, and that is deliberate:
        ``beta_z`` is a ``D_t``-scale quantity and ``eta(D_t) ~ 1`` (the cluster
        onset is ~1e3 Gy, ``D_t`` is tens of Gy). The enhanced cap belongs to
        ``alpha``, which samples core doses ~1e5 Gy.
        """
        return self._beta_from_shared_model(self.beta_model, ion, alpha_ion)

    def _cluster(self) -> ClusterResponse:
        return self.resolved_cluster

    def _radial_dose_model(self) -> RadialDoseModel:
        return ScholzKraftRadialDose.lem2(stopping_power=self.stopping_power)

    def _max_local_slope(self) -> float:
        """Enhanced maximum local slope ``s_max * eta_max`` (cluster-extended tail).

        The rapidLEM (``lethal_central``) saturation cap for LEM II/III: the local
        photon tail slope is ``s_max * eta(D)`` (Elsasser 2007 eq. 10), so the
        steepest local slope -- and the correct cap -- is ``s_max * eta_max``. This
        keeps the cluster enhancement the un-enhanced ``s_max`` cap would drop (see
        [`lethal_central_initial_slope`][pyradbio.models.lem.base.LEMBase.lethal_central_initial_slope]).
        """
        return self.photon.max_slope() * self._cluster().eta_max

    def photon_log_kill(self, dose_Gy: float) -> float:
        """Cluster-enhanced photon log-kill ``-ln S_x(D)`` (Elsasser & Scholz 2007 eq. 10).

        The public face of the photon response this model integrates over the track:
        the shouldered LQ below ``D_t`` and ``-ln S_x(D_t) + s_max * (eta(D)*D - D_t)``
        above it (doi:10.1667/RR0467.1, eq. 10; ``s_max = alpha_x + 2*beta_x*D_t``).
        This is exactly the curve of the paper's Fig. 3, so a reconstruction can draw
        it without reaching into the model. It returns what the private hook
        `_photon_log_kill` returns -- that hook stays, because
        [`LEMBase`][] calls it polymorphically for every
        LEM version (and `_photon_log_kill_array` must stay in step with it).
        """
        return self._photon_log_kill(dose_Gy)

    def _photon_log_kill(self, dose_Gy: float) -> float:
        """Photon log-kill with the high-dose cluster enhancement (Elsasser 2007 eq. 10).

        Below ``D_t`` the shouldered LQ response is unchanged. Above it the
        *dose* in the exponential tail is scaled by ``eta(D)``:

            -ln S_x(D) = -ln S_x(D_t) + s_max * (eta(D)*D - D_t)

        Continuity at ``D_t`` holds because ``eta(D_t) = 1`` to within 1e-4 for any
        realistic ``D_t`` (tens of Gy), far below the ~1e3 Gy cluster onset.

        ``eta`` multiplies the dose here; it is **not** integrated over it. See the
        module docstring -- the integral reading misses the paper's Fig. 3 by
        22.7% RMS where this lands within 2.7%.
        """
        d_t = self.photon.D_t
        assert d_t is not None  # enforced in LEMBase.__post_init__
        if dose_Gy <= d_t:
            return self.photon.log_kill(dose_Gy)
        lk_t = self.photon.log_kill(d_t)
        enhanced = self._cluster().eta(dose_Gy) * dose_Gy
        return lk_t + self.photon.max_slope() * (enhanced - d_t)

    def _photon_log_kill_array(self, dose_Gy: npt.NDArray[np.float64]) -> npt.NDArray[np.float64]:
        """Vectorised `_photon_log_kill`, cluster enhancement included.

        Kept in step with the scalar override by construction. Overriding only
        the scalar would leave the array-based beta models
        (``beta_model="scholz1997_two_track"``) silently computing an
        *un-enhanced* LEM I response under a LEM II/III label -- so this exists
        for correctness, not speed.

        Uses the cluster's vectorised ``eta_array`` (part of
        [`ClusterResponse`][]): the pair integrals evaluate it on ~1e5 doses,
        so a scalar loop is never an option.
        """
        d_t = self.photon.D_t
        assert d_t is not None  # enforced in LEMBase.__post_init__
        dose = np.asarray(dose_Gy, dtype=np.float64)
        # Eq. 10 differs from the un-enhanced tail by exactly s_max*(eta-1)*D:
        #   plain    = lk_t + s_max*(    D - D_t)
        #   enhanced = lk_t + s_max*(eta*D - D_t)
        # so build on the base response rather than restating the LQ branch, and
        # the two cannot drift apart below D_t.
        excess = self.photon.max_slope() * (self._cluster().eta_array(dose) - 1.0) * dose
        return np.asarray(
            LEMBase._photon_log_kill_array(self, dose) + np.where(dose > d_t, excess, 0.0),
            dtype=np.float64,
        )
