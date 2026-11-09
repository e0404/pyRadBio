"""LEM IV: DSB-pattern / cluster-index formulation.

Friedrich et al. 2012 (doi:10.3109/09553002.2011.611213) and the detailed
formalism of Pfuhl, Friedrich & Scholz 2020 (Radiat. Res. 193:130, eqs. 1-7),
with the DSB-yield enhancement of Friedrich, Durante & Scholz 2015 (Radiat. Prot.
Dosimetry 166:61). LEM IV replaces the local *dose* mapping of LEM I-III with a
mapping of the spatial *DSB damage pattern*.

DNA is organised into ~2 Mbp "giant loops"; the nucleus is tiled into cubic
subvolumes of side ``l_DSB = 540 nm``, one loop each (6e9 bp / 500 um^3 -> ~3000
loops). A loop with exactly one DSB carries an *isolated* DSB (iDSB), one with two or
more a *clustered* DSB (cDSB). A damage pattern is characterised by the

    cluster index   C = N_cDSB / (N_iDSB + N_cDSB)                (Pfuhl eq. 2)

**Discrete subvolume scoring.** The DSB count of a subvolume is Poisson with mean
``m``, the DSB yield integrated over the *whole* 540 nm subvolume (Pfuhl 2020), not
the pointwise local dose. With 30 DSB/Gy/cell (Stenerlow 2003 / Prise 2001) and the
SSB->DSB enhancement ``eta(d)`` (Friedrich 2015),

    m = (DSB_PER_GY_PER_LOOP / l_DSB^2) * integral_square eta(d(x,y)) d dx dy,

    P(iDSB) = m e^-m ,  P(cDSB) = 1 - e^-m (1 + m) ,  P(dmg) = 1 - e^-m .

(Pointwise ``m(r)`` would over-saturate the 1/r^2 core and lose both the clustering
and the ``eta`` enhancement.)

**Single-particle effect (Pfuhl eqs. 3-7).** The photon-equivalent dose ``D_eq`` is
defined by ``C_g(D_eq) = C_I`` (``C_g`` is monotone in dose), and

    kappa = (N_iDSB,Ion + N_cDSB,Ion) / (N_iDSB,g(D_eq) + N_cDSB,g(D_eq)) ,
    N_1   = kappa * E_eq ,   E_eq = -ln S_g(D_eq) ,

with ``E_eq`` the shouldered LQ photon response (linear above ``D_t``, enhanced by
``eta`` at very high ``D_eq``; Pfuhl eq. 4). ``alpha_ion`` follows "the method of the
original LEM" (Scholz 1997), the Poisson single-track average

    alpha_I = (1 - e^-N_1) / D_1        (overkill saturation at 1/D_1),

with ``D_1`` the mean single-traversal nuclear dose (Elsasser & Scholz 2007 eq. 8),
plus an escaped-energy term ([`LEM4.initial_slope`][]) that restores
``alpha_I -> alpha_x`` at low LET.

**beta_I.** The default Kraemer & Scholz 2006 slope matching reproduces Friedrich 2012
Fig. 3, which "is based on the single particle approximation" (p. 105); the
full-simulation ``beta_model`` options target its Fig. 4 (see [`LEM4BetaModel`][]).

**Track structure.** Amorphous 1/r^2 profile with velocity-dependent core
``r_min = beta * r_c``, ``r_c = 6.5 nm`` (Pfuhl 2020 eq. 1; LEM III uses 40 nm),
folded with the sigma = 4 nm radical-diffusion Gaussian.

**Limitation: deep overkill.** When ``r_max`` falls below the 540 nm subvolume the
track collapses into one loop column (``C_I -> 1``) and the TRiP98 tables are not
reproduced; see "Low energy / deep overkill against the TRiP98 tables" in
``docs/references/LEM_IV_Friedrich2012.md``. Choices without a paper basis are
registered in ``docs/references/LEM_IV_OPEN_CHOICES.md``.
"""

from __future__ import annotations

import math
import warnings
from collections.abc import Callable, Iterable
from dataclasses import dataclass
from typing import ClassVar, Literal, get_args

import numpy as np
import numpy.typing as npt
from scipy.integrate import cumulative_trapezoid, trapezoid
from scipy.interpolate import PchipInterpolator
from scipy.optimize import brentq

from pyradbio.core.radiation import Ion
from pyradbio.models.base import check_choice
from pyradbio.models.lem import _discrete, _intertrack_kernel, _lem4_kernel, _localdose
from pyradbio.models.lem.base import (
    InitialSlopeModel,
    LEMBase,
    TraversalConditionedSurvival,
    _delete_one_jackknife,
)
from pyradbio.models.lem.engine import ENGINES, Engine
from pyradbio.track.base import RadialDoseModel
from pyradbio.track.scholz_kraft import ScholzKraftRadialDose

# Execution engine for the discrete DSB-pattern scoring. ``numpy`` is the
# vectorised reference; ``numba`` is the compiled parallel kernel (`._lem4_kernel`,
# ~30x faster, requires the ``fast`` extra); ``auto`` uses numba when importable and
# the radial dose is the built-in Scholz-Kraft radical-diffusion profile, else numpy.
LEM4Engine = Engine  # one definition: pyradbio.models.lem.engine

# How a single traversal's DSB pattern maps to lethal events.
#   "photon_equivalent" (default) -- Elsasser 2010 / Pfuhl 2020 eqs. 3-6: match the ion
#       cluster index to a photon dose ``D_eq`` and rescale the photon effect.
#   "globle" -- Friedrich, Durante & Scholz 2012 (Radiat. Res. 178(5):385-394,
#       doi:10.1667/RR2964.1) per-loop effect ``E = n_iDSB*eps_i + n_cDSB*eps_c``
#       (eqs. 2, 8, 9): bounded, with no ``D_eq`` inversion and no extrapolation of the
#       photon curve to ~100+ Gy.
LEM4EffectModel = Literal["photon_equivalent", "globle"]

# How the ion quadratic term ``beta_I`` is obtained.
#   "slope_matching" (default) -- Kraemer-Scholz 2006 slope matching
#       ``beta_I = (s_max - alpha_I) / (2 D_t)`` (LEMBase._beta); Friedrich 2012 Fig. 3.
#   "intertrack_mc" -- Friedrich 2012 Fig. 4 full simulation as an explicit-track
#       Monte-Carlo placing discrete DSBs into a shared 540 nm grid
#       (`LEM4.intertrack_mc_beta`). Slow diagnostic; ``alpha_I`` is unchanged.
#   "local_dose" -- the Fig. 4 full simulation as a deterministic compound-Poisson
#       local-dose convolution (Elsasser & Scholz 2007; `LEM4.local_dose_beta`).
#   "discrete_dsb" -- the same simulation as an explicit discrete damage pattern
#       (`LEM4.discrete_dsb_beta`); a cross-check of ``"local_dose"`` by an
#       independent route, provably unable to add the high-LET roll-off it misses.
LEM4BetaModel = Literal[
    "slope_matching", "rapid_lem", "intertrack_mc", "local_dose", "discrete_dsb"
]

# How the discrete-DSB engine forms the cluster index from its damage pattern.
#   "per_realisation" (default) -- ``C_I`` within each realisation (track number), then
#       average the effect; the only part of the engine not algebraically identical to
#       the continuum convolution.
#   "mean_counts" -- ``C_I`` from the realisation-averaged loop counts; provably identical
#       to ``local_dose_kernel="cluster_index"`` (`LEM4.discrete_dsb_lethal_events`).
LEM4DiscreteConvention = Literal["per_realisation", "mean_counts"]

# Cell-survival convention of the discrete-DSB engine.
#   "mean_effect" (default) -- ``S = exp(-<E>)`` over the traversal count.
#   "stochastic" -- ``S = <exp(-E)> = sum_n Pois(n; D/D_1) exp(-E_n)``, the Scholz 1997
#       convention, evaluated exactly from the per-traversal loop counts. Requires
#       ``discrete_convention="per_realisation"``. A diagnostic; it does not make this
#       engine preferable (`LEM4.discrete_dsb_lethal_events`).
LEM4DiscreteSurvival = Literal["mean_effect", "stochastic"]

# How the single-particle initial slope averages over track configurations
# (LEM_IV_OPEN_CHOICES.md entry 2.1).
#   "mean_survival" (default) -- S = <e^-E>: alpha_I = (1 - e^-N_1)/D_1 (Pfuhl 2021
#       eq. 2.32, the standard LEM formula), bounded by 1/D_1.
#   "mean_effect" -- S = e^-<E>: the unsaturated alpha_I = N_1/D_1 (eq. 2.31). TRiP98's
#       LEM IV tables behave like this, but it has no overkill ceiling, is not the formula
#       the papers state, and is worse for protons and helium at low energy. The two are
#       identical above ~10 MeV/u and differ by at most ~12% (N_1 stays below ~0.37).
LEM4SurvivalConvention = Literal["mean_survival", "mean_effect"]

# Impact parameters for the deterministic discrete cluster index: holds ``1 - C_I`` to
# ~1% (see ``LEM4.analytic_discrete_cluster_index`` for why that is the criterion).
_DISCRETE_N_IMPACT: int = 384

# Which estimator supplies the single-particle cluster index to the RBE path
# (LEM_IV_OPEN_CHOICES.md entry 2.3). None of them is right everywhere.
#   "mean_field" (default) -- `LEM4.cluster_index`, track at a subvolume centre.
#       Over-clusters (near 1 over 0.4-2.5 MeV/u); right above ~3.5 MeV/u.
#   "discrete" -- `LEM4.analytic_discrete_cluster_index`, deterministic; matches
#       the sampled MC from ~2.5 MeV/u up, runs 3-6x above it below ~1 MeV/u.
#   "discrete_mc" -- `LEM4.discrete_cluster_index`, sampled (seeded); the estimator
#       below ~2 MeV/u and the one reproducing the Pfuhl 2020 Table 3 anchors.
#   "trip98_composite" -- "discrete" while the whole profile fits one 540 nm loop
#       (`LEM4.pattern_is_single_loop`, 2.206 MeV/u for every ion), "mean_field"
#       above: a geometric threshold with no free parameter, for reproducing the TRiP98
#       RBE_alpha TABLES only (not a general improvement). "discrete" is the low arm as it
#       is deterministic and ~10x cheaper. The seam is discontinuous (up to 4x for carbon),
#       the model's own inconsistency, which moving it cannot fix.
LEM4ClusterIndexModel = Literal["mean_field", "discrete", "discrete_mc", "trip98_composite"]

# Realisations for ``cluster_index_model="discrete_mc"``, chosen on the seed-to-seed
# spread of ``1 - C_I`` (never of ``C_I``), which is what ``D_eq = C_g^-1(C_I)`` reads.
# The spread scales as ``1/sqrt(n)`` and is worst deep in overkill; 16000 keeps it below
# ~10% even for 0.126 MeV/u carbon, at a few seconds per point.
_DISCRETE_MC_REALISATIONS: int = 16000
_DISCRETE_MC_SEED: int = 0

# Numerical guard for the ``C_g^-1`` inversion on the *discrete* path. Not the 0.95
# mean-field cap: the discrete index is self-bounded (below 0.99) and sits above 0.95
# through the overkill band, so clipping it there would collapse it onto the mean field.
_DISCRETE_CLUSTER_INDEX_MAX: float = 0.999

# How the local-dose convolution maps its dose density to lethal events
# (``beta_model="local_dose"``; see `LEM4.local_dose_beta`).
#   "cluster_index" (default) -- LEM IV's own mapping (Pfuhl eqs. 3-6) on the local-dose
#       density: nucleus-wide damaged and clustered loop counts, ``C_I = N_cDSB / N_dmg``,
#       ``C_g(D_eq) = C_I``, ``E = N_dmg * (-ln S_g(D_eq)) / N_dmg,g(D_eq)``, on the
#       eta-enhanced dose (`LEM4._local_dose_jumps`). The only kernel reproducing
#       both Fig. 4 features (``beta_I`` rising above ``beta_x``, then collapsing); any
#       response that is a function of the *physical* local dose has ``beta_I <= beta_x``.
#   "photon_effect" -- the Elsasser & Scholz 2007 local effect ``rho_A(d) = -ln S_x(d)``
#       (`LEM4._photon_effect`); its threshold ``D_t`` makes ``beta_I`` fall with LET.
#   "globle" -- the per-loop GLOBLE response ``P_iDSB(m) eps_i + P_cDSB(m) eps_c``. A
#       cross-check: without ``D_t`` and with ``m <= 0.33`` it stays quadratic, so
#       ``beta_I = beta_x`` for every ion.
#   "discrete_cluster_index" -- as "cluster_index" but with the average-of-ratios
#       ``E[N_cDSB/N_dmg]`` (`._discrete.expected_damaged_ratio`); Monte-Carlo only.
#       A cross-check; it does not fix the high-Z ``initial_slope_model="full_simulation"``
#       route (docs/references/LEM_IV_Friedrich2012.md).
LEM4LocalDoseKernel = Literal["cluster_index", "discrete_cluster_index", "photon_effect", "globle"]

# Execution backend for the intertrack Monte-Carlo (not the model's ``engine``):
# "numpy" (default) is the seed-reproducible reference; "numba" (`._intertrack_kernel`,
# ``fast`` extra) is much faster but per-thread RNG makes it *not* bit-reproducible for a
# fixed ``seed``; "auto" picks numba when importable.
LEM4MCEngine = Engine  # the same three engines

# GLOBLE giant-loop count N_L (Friedrich, Durante & Scholz 2012, Radiat. Res.
# 178(5):385-394, doi:10.1667/RR2964.1). 2000 is the paper's best fit to its >150-curve
# photon ensemble (Fig. 5B); 3000 = 6e9 bp / 2 Mbp is only the value of its illustrative
# Figs. 2-4. The paper never applies GLOBLE to ions, so N_L is a free, configurable
# parameter (``LEM4.globle_n_loops``; LEM_IV_OPEN_CHOICES.md entry 1.4).
GLOBLE_N_LOOPS: float = 2000.0

# Chromatin / DSB parameterisation (Elsasser 2010 Fig. 2 caption; Pfuhl 2020). The
# nucleus is tiled into cubic subvolumes of side ``l_DSB``; ``l_DSB`` and the
# reference nuclear volume ``V`` are *fitted, generic* constants held fixed for all
# cells and ions.
SUBVOLUME_SIDE_NM: float = 540.0  # l_DSB (Elsasser 2010, fitted generic value)
NUCLEUS_VOLUME_UM3: float = 500.0  # reference nuclear volume V (Elsasser 2010)
DSB_PER_GY_CELL: float = 30.0  # DSB yield (Stenerlow 2003 / Prise 2001, via Elsasser 2010)

# Mean DSB in a subvolume at uniform local dose ``d`` is ``30 DSB/Gy/cell * (dV/V)
# * d``, so the Poisson mean per subvolume per Gy is ``DSB_PER_GY_PER_LOOP =
# 30 * l_DSB^3 / V`` ~ 0.00945 /Gy (~2 Mbp per subvolume over a ~6000 Mbp genome ->
# ~3200 subvolumes; Yokota 1995, Solovjeva 1998).
DSB_PER_GY_PER_LOOP: float = (
    DSB_PER_GY_CELL * (SUBVOLUME_SIDE_NM * 1.0e-3) ** 3 / NUCLEUS_VOLUME_UM3
)

# Geometric giant-loop count of the 540 nm grid, ``V / l_DSB^3`` ~ 3175. The intertrack
# MC bins DSBs into this grid, so its GLOBLE ``eps_c`` is calibrated to this count (the
# photon limit recovers ``beta_x * N_L(eps_c) / N_grid``), not to the empirically fitted
# `LEM4.globle_n_loops`.
MC_GRID_N_LOOPS: float = DSB_PER_GY_CELL / DSB_PER_GY_PER_LOOP

# DSB-yield enhancement eta(D) parameters (Friedrich, Durante & Scholz 2015). At
# high local dose two SSB in close proximity (<= 25 bp) on opposite strands form an
# additional (inter-track) DSB; eta is the ratio of enhanced to unenhanced yield.
SSB_PER_GY_CELL: float = 1250.0  # SSB yield (Friedrich 2015)
SSB_STAGGER_BP: float = 25.0  # max SSB separation forming a DSB [base pairs]
GENOME_SIZE_BP: float = 5.4e9  # genomic length used in Friedrich 2015

# Velocity-dependent inner track radius ``r_min = beta * r_c`` (Pfuhl 2020 eq. 1);
# LEM III uses ``r_c = 40 nm``. Kept deliberately although Friedrich 2012 (p. 104) cites
# the Elsasser 2008b profile: the Fig. 4 ``alpha_I`` reconstruction needs this sharper
# core. See "The track core" in docs/references/LEM_IV_Friedrich2012.md.
LEM4_RC_NM: float = 6.5

# Fine 2-D scoring grid over the central ``(2*_CORE_HALF_SUBVOX+1)^2`` subvolumes at
# ``_FINE_STEP_NM`` (the penumbra beyond is integrated analytically); 4 / 3 nm matches
# a finer-grid reference to <1% across the ion-therapy LET range.
_CORE_HALF_SUBVOX: int = 4
_FINE_STEP_NM: float = 3.0
# Radial nodes for the precomputed local-dose profile and the penumbra integral.
_RADIAL_NODES: int = 1500

# Photon-equivalent-dose bracket [Gy]: C_g(D) is monotone from 0 (D->0) to 1
# (D->inf), so any ion cluster index in (0, 1) is bracketed here.
_DEQ_LO_GY: float = 1.0e-6
_DEQ_HI_GY: float = 1.0e8

# Largest m for which ``m e^-m`` is still a normal float64 (~745 by exp underflow).
_LOG_MAX_FLOAT: float = 700.0

# Numerical invertibility guard for D_eq = C_g^-1(C_I). ``C_g`` reaches exactly 1.0 at
# 3173 Gy in float64 (``P_iso = m e^-m`` underflows); this bound keeps the ``C_g -> g``
# lookup off its saturated top entry. Not a modelling choice: it binds nowhere a
# resolvable cluster index occurs. EVERY path that maps a cluster index to an effect
# must apply it -- use ``_cluster_index_bounds``.
_CLUSTER_INDEX_MAX: float = 1.0 - 1.0e-15

# The legacy cap, opt-in via ``LEM4(legacy_trip98_cluster_index_cap=...)``. Not derived:
# a free parameter tuned on the TRiP98 carbon curves (the inversion is not
# ill-conditioned near C_I -> 1). The published-figure goldens still need it to mask the
# mean-field lattice-registration bump. See LEM_IV_OPEN_CHOICES.md entry 1.1.
_CLUSTER_INDEX_TRIP98_LEGACY: float = 0.95

# Intertrack-cDSB Monte-Carlo defaults (``beta_model="intertrack_mc"``). ``beta`` is a
# small curvature on a dominant linear term, so it needs many realisations and an
# adaptive per-LET dose grid that stays in the LQ region.
_INTERTRACK_REALIZATIONS: int = 600  # shared-grid dose-response realisations per beta
# The same 600 realisations in 12 batches for the delete-one jackknife
# (`LEM4.intertrack_mc_beta_jackknife`), as LEM I's ``base._TC_BATCHES``.
_INTERTRACK_BATCHES: int = 12
_INTERTRACK_REALIZATIONS_PER_BATCH: int = 50
# Ceiling and per-round growth cap for `LEM4.intertrack_mc_beta_to_precision`;
# the 4x cap keeps an unluckily noisy pilot from allocating the whole ceiling at once.
_INTERTRACK_MAX_REALIZATIONS: int = 120_000
_INTERTRACK_MAX_ALLOCATION_GROWTH: float = 4.0
_INTERTRACK_N_DOSES: int = 9  # doses in the adaptive LQ-region grid
# Adaptive dosing (`LEM4._adaptive_doses`): ``D_max = min(target/alpha_sp, frac*D_t)``.
_INTERTRACK_EFFECT_TARGET: float = 2.0
_INTERTRACK_DMAX_DT_FRACTION: float = 0.9

# Local-dose convolution settings (``beta_model="local_dose"``).
# Radial nodes for the subvolume-averaged single-track dose ``g(rho)``, out to
# ``r_max + l_DSB``; 500 log-spaced nodes conserve the dose-area to <0.1%.
_LOCALDOSE_RHO_NODES: int = 500
# Log-dose bins for compressing the jump sizes before the characteristic function is
# summed; preserves total intensity and dose-area exactly.
_LOCALDOSE_BINS_PER_DECADE: int = 48
# FFT grid for the local-dose density. The reported grid is the lower half.
_LOCALDOSE_N_GRID: int = 1 << 16
# Dose window for the LQ fit, as FRACTIONS OF ``D_t`` -- see LEM4.fit_doses_Gy and
# LEM_IV_OPEN_CHOICES.md entry 1.2.
# UNVERIFIED: no paper (nor TRiP98) states a fit window; this free parameter was chosen
# against Friedrich 2012 Fig. 4's ``beta_I``, so agreement there is partly by
# construction. Relative to ``D_t`` (where the photon curve goes linear) and FIXED: a
# LET-dependent window manufactures LET dependence in ``beta_I``. High-LET survival can
# reach ~1e-5 inside it; that is the model's response, not clipped.
_LOCALDOSE_DOSE_FRACTIONS: tuple[float, ...] = (
    0.05,
    0.075,
    0.10,
    0.15,
    0.20,
    0.25,
    0.32,
    0.40,
    0.50,
    0.63,
    0.75,
    0.85,
)
# Realisations for the explicit-track Monte-Carlo cross-check.
_LOCALDOSE_MC_REALIZATIONS: int = 400

# Impact-parameter cut for ``local_dose_mc_impact_cut="auto"``, in nucleus radii. Far
# tracks (most of the low-LET cost) are replaced by their expectation
# (`_localdose.far_track_background_per_Gy`), dropping only their fluctuation. A
# convergence choice, valid only with the domain-fraction floor below
# (LEM_IV_OPEN_CHOICES.md entry 1.3).
_MC_IMPACT_CUT_FACTOR: float = 8.0

# Minimum cut as a fraction of ``r_sim``: ``8 * r_eff`` alone biases ``beta`` low once
# it falls below ~0.13 of the domain (below LET ~18); 0.15 adds margin.
_MC_IMPACT_CUT_MIN_DOMAIN_FRACTION: float = 0.15

# Dose window for the discrete-DSB engine's LQ fit. Same protocol and same window as
# the local-dose convolution, so the two engines are directly comparable.
_DISCRETE_DOSE_FRACTIONS: tuple[float, ...] = _LOCALDOSE_DOSE_FRACTIONS

# Probe grid for LEM4.survival_bounded_ceiling_Gy, as fractions of ``D_t``: one
# vectorised local-dose call, then monotone interpolation in (log D, log E) (<3e-5
# relative error on E).
_SURVIVAL_PROBE_FRACTIONS: tuple[float, ...] = tuple(
    float(x) for x in np.logspace(math.log10(0.05), math.log10(3.0), 24)
)

# Seeds averaged by ``LEM4.full_simulation_initial_slope``: four fixed seeds bring the
# few-% single-fit scatter under ~1% and keep ``alpha_beta`` deterministic. Fixed, not
# configurable: changing them changes the number.
_FULLSIM_SEEDS: tuple[int, ...] = (0, 1, 2, 3)
# ``(R_eff + r_max)/R_eff`` past which the full simulation warns about its own cost
# (between 20 and 50 MeV/u carbon, where a run stops being interactive).
_FULLSIM_DOMAIN_WARN: float = 5.0

# Guards for the ``<exp(-E)>`` survival conventions: clip the exponent and floor the
# survival so deep overkill cannot produce ``-log(0)``.
_MAX_EXPONENT: float = 700.0
_MIN_SURVIVAL: float = 1.0e-300


def _eta(dose_Gy: npt.NDArray[np.float64]) -> npt.NDArray[np.float64]:
    """SSB->DSB DSB-yield enhancement ``eta(D)`` (Friedrich 2015 eqs. 7-8).

    ``eta = 1 + P * (a_SSB / a_DSB)`` with the SSB->DSB conversion probability
    ``P = (1 - e^{-rho_SSB t}) / (3 - e^{-rho_SSB t}) * e^{-rho_DSB t}`` and lesion
    densities ``rho = a * D / L_gen``. ``eta = 1`` below ~1000 Gy, peaks ~13 near
    5e5 Gy, and returns toward 1 at very high dose (direct DSB again dominate).
    """
    d = np.asarray(dose_Gy, dtype=np.float64)
    rho_ssb_t = SSB_PER_GY_CELL * d / GENOME_SIZE_BP * SSB_STAGGER_BP
    rho_dsb_t = DSB_PER_GY_CELL * d / GENOME_SIZE_BP * SSB_STAGGER_BP
    e_ssb = np.exp(-rho_ssb_t)
    p_conv = (1.0 - e_ssb) / (3.0 - e_ssb) * np.exp(-rho_dsb_t)
    return np.asarray(1.0 + p_conv * (SSB_PER_GY_CELL / DSB_PER_GY_CELL), dtype=np.float64)


# Plastic ratio, the real root of ``x^3 = x + 1``: generator of the R2 low-discrepancy
# sequence (Roberts 2018), every prefix of which is well spread.
_PLASTIC_RATIO: float = 1.324717957244746


def _r2_points(n: int) -> npt.NDArray[np.float64]:
    """``n`` points of the R2 low-discrepancy sequence on the unit square.

    Deterministic (no seed), used to marginalise the giant-loop lattice's phase in
    [`LEM4.analytic_discrete_cluster_index`][].
    """
    i = np.arange(1, n + 1, dtype=np.float64)
    a1, a2 = 1.0 / _PLASTIC_RATIO, 1.0 / (_PLASTIC_RATIO * _PLASTIC_RATIO)
    return np.stack([(0.5 + a1 * i) % 1.0, (0.5 + a2 * i) % 1.0], axis=1)


def _bin_dsb_3d(
    x: npt.NDArray[np.float64],
    y: npt.NDArray[np.float64],
    z: npt.NDArray[np.float64],
    *,
    ell: float,
    r_eff: float,
) -> tuple[int, int]:
    """Bin discrete DSB positions into ``ell``-sided cubes -> ``(N_damaged, N_clustered)``.

    A cube is a nuclear loop iff its transverse *centre* lies within ``r_eff`` (the
    centre-inside boundary rule); ``N_clustered`` counts valid loops holding >= 2 DSBs.
    Used by [`LEM4.discrete_cluster_index`][].
    """
    codes = np.stack([np.floor(x / ell), np.floor(y / ell), np.floor(z / ell)], axis=1).astype(
        np.int64
    )
    uniq, counts = np.unique(codes, axis=0, return_counts=True)
    cx = (uniq[:, 0].astype(np.float64) + 0.5) * ell
    cy = (uniq[:, 1].astype(np.float64) + 0.5) * ell
    inside = cx * cx + cy * cy <= r_eff * r_eff
    n_damaged = int(np.count_nonzero(inside))
    n_clustered = int(np.count_nonzero(inside & (counts >= 2)))
    return n_damaged, n_clustered


@dataclass(frozen=True, slots=True)
class DiscreteSetup:
    """Per-ion inputs of the discrete-DSB engine (`LEM4._discrete_setup`).

    ``a0`` / ``a1`` are the per-loop single-track factors ``<e^-mu>`` / ``<mu e^-mu>``
    over track positions uniform on the nucleus disk (`_discrete.track_factors`),
    ``n_z`` the number of loop layers along the beam, and ``n_loops`` the total giant-loop
    count of the lattice (``~ V / l_DSB^3``).

    ``mean_tracks_per_Gy`` is exactly ``1 / D_1`` and ``track_disk_radius_um`` exactly
    ``R_eff`` (no domain extension); ``retained_dose_fraction`` is the dose-conserving
    track/background split, derived rather than a parameter.
    """

    a0: npt.NDArray[np.float64]
    a1: npt.NDArray[np.float64]
    n_z: float
    n_loops: float
    retained_dose_fraction: float
    mean_tracks_per_Gy: float
    track_disk_radius_um: float


@dataclass(frozen=True, slots=True)
class IntertrackSurvival:
    """Batched intertrack-MC dose response, with a delete-one jackknife error.

    ``batch_effect`` holds one row of mean *effects* ``<E(D)>`` per independent batch.
    The effect, not the survival, is the poolable quantity (survival is ``exp(-<E>)``).
    Mirrors [`TraversalConditionedSurvival`][] (LEM I-III).
    ``beta`` is a nonlinear functional (a least-squares fit) of the pooled curve, so its
    error comes from a delete-one jackknife over batches
    (`_delete_one_jackknife`).
    """

    batch_effect: npt.NDArray[np.float64]  # (n_batches, n_doses)
    doses_Gy: npt.NDArray[np.float64]
    realizations_per_batch: int

    def __post_init__(self) -> None:
        if self.batch_effect.ndim != 2 or self.batch_effect.shape[0] < 2:
            raise ValueError("batch_effect must be a (n_batches >= 2, n_doses) array")
        if self.batch_effect.shape[1] != self.doses_Gy.size:
            raise ValueError("batch_effect columns must match doses_Gy")
        if self.realizations_per_batch <= 0:
            raise ValueError("realizations_per_batch must be > 0")

    @property
    def n_batches(self) -> int:
        return int(self.batch_effect.shape[0])

    @property
    def n_realizations(self) -> int:
        return self.n_batches * self.realizations_per_batch

    @property
    def effect(self) -> npt.NDArray[np.float64]:
        """Pooled mean effect ``<E(D)>`` over all batches."""
        return np.asarray(self.batch_effect.mean(axis=0), dtype=np.float64)

    @property
    def survival(self) -> npt.NDArray[np.float64]:
        """Pooled survival ``S(D) = exp(-<E>)``."""
        return np.asarray(np.exp(-self.effect), dtype=np.float64)

    def pooled_with(self, other: IntertrackSurvival) -> IntertrackSurvival:
        """Concatenate the batches of two runs of the same ion and dose grid.

        Lets a precision allocator add realisations instead of restarting.
        """
        if other.realizations_per_batch != self.realizations_per_batch:
            raise ValueError("realizations_per_batch must match to pool batches")
        if not np.allclose(other.doses_Gy, self.doses_Gy, rtol=1.0e-12, atol=0.0):
            raise ValueError("dose grids differ; the curves are not the same measurement")
        return IntertrackSurvival(
            batch_effect=np.concatenate([self.batch_effect, other.batch_effect], axis=0),
            doses_Gy=self.doses_Gy,
            realizations_per_batch=self.realizations_per_batch,
        )

    def jackknife(
        self, observable: Callable[[npt.NDArray[np.float64]], float]
    ) -> tuple[float, float]:
        """``(value, standard_error)`` for any functional of the pooled *effect*."""
        return _delete_one_jackknife(self.batch_effect, observable)

    def jackknife_beta(self) -> tuple[float, float]:
        """``(beta, standard_error)`` of the LQ fit over this curve's dose grid.

        **Unclamped**: in deep overkill a negative ``beta`` is a measurement, and
        flooring it would bias the mean and shrink the error. The scalar
        [`LEM4.intertrack_mc_beta`][] floors its return value (and warns).
        """
        return self.jackknife(
            lambda effect: _intertrack_kernel.fit_alpha_beta(
                self.doses_Gy, np.exp(-np.asarray(effect, dtype=np.float64))
            )[1]
        )


def _p_cdsb(m: npt.NDArray[np.float64]) -> npt.NDArray[np.float64]:
    """Poisson probability of >= 2 DSB in a loop, ``1 - e^-m (1 + m)``."""
    return 1.0 - np.exp(-m) * (1.0 + m)


def _p_damaged(m: npt.NDArray[np.float64]) -> npt.NDArray[np.float64]:
    """Poisson probability of >= 1 DSB in a loop, ``1 - e^-m``."""
    return 1.0 - np.exp(-m)


def _cluster_index_from_dsb_mean(m: float) -> float:
    """Cluster index of a *single* loop with Poisson mean ``m``.

    ``C(m) = P(>=2) / P(>=1)``, written as ``1 - m e^-m / (1 - e^-m)`` so the
    unclustered fraction is formed directly instead of by cancelling two numbers
    that both approach 1. Both the photon pattern (uniform dose, every loop the
    same ``m``) and the single-loop ion pattern are exactly this function, which
    is why matching ``C`` and matching ``m`` are the same condition there.
    """
    if m <= 0.0:
        return 0.0
    if m > _LOG_MAX_FLOAT:  # m e^-m has underflowed; the loop is fully clustered
        return 1.0
    return 1.0 - m * math.exp(-m) / (-math.expm1(-m))


def _outside_square_fraction(
    r: npt.NDArray[np.float64], half_width: float
) -> npt.NDArray[np.float64]:
    """Fraction of the circle of radius ``r`` lying OUTSIDE an axis-aligned square
    of half-width ``half_width`` centred on it.

    0 for ``r <= s`` (circle inscribed), 1 for ``r >= s sqrt(2)`` (square inscribed),
    and ``(4/pi) arccos(s/r)`` between: the circle leaves the square through four
    congruent arcs, each spanning ``2 arccos(s/r)``.

    Lets the penumbra dose-area be integrated directly rather than as ``(total radial
    integral) - (fine-grid sum)``: that difference of two quadratures does not vanish
    when the whole profile fits in one subvolume (sub-MeV), where it would be the whole
    ``1 - C_I``.
    """
    s = half_width
    out = np.ones_like(r)
    out[r <= s] = 0.0
    mid = (r > s) & (r < s * math.sqrt(2.0))
    out[mid] = (4.0 / math.pi) * np.arccos(np.clip(s / r[mid], -1.0, 1.0))
    return out


def _cluster_index_bounds(c_g: npt.NDArray[np.float64]) -> tuple[float, float]:
    """``(lo, hi)`` clip bounds for a cluster index fed to a ``C_g -> g`` lookup.

    ``lo`` is the lookup grid's floor, ``hi`` is `_CLUSTER_INDEX_MAX`, the same
    regularisation `LEM4._damage_and_deq` and the intertrack kernels (``ci_max``)
    apply. Do not clip to ``c_g[-1]`` instead: ``C_g`` saturates to exactly 1.0 near
    3175 Gy, so the top grid entries tie and a saturated ``C_I`` would map to a far
    higher ``D_eq`` than on the analytic path. Route every cluster index through here.
    """
    return float(c_g[0]), _CLUSTER_INDEX_MAX


@dataclass(frozen=True, slots=True)
class LEM4(LEMBase):
    """LEM IV: DSB-pattern / cluster-index model with discrete 540 nm subvolume
    scoring on the ``r_c = 6.5 nm`` velocity-core radial dose.

    Supply photon parameters and geometry on ``photon`` / ``cell`` as for the other
    LEM versions (Friedrich 2012 Fig. 3 V79: ``alpha_x=0.184``, ``beta_x=0.02``,
    ``D_t=17`` Gy, ``R_n=4.7`` um). ``photon.cluster`` is not used (the SSB->DSB
    enhancement is the built-in Friedrich 2015 ``eta``); passing one warns.

    Options (details at the ``LEM4*`` type aliases and the methods named):

    - ``engine``: DSB-pattern scoring backend; ``"auto"`` uses numba when installed and
      the radial dose is the built-in Scholz-Kraft profile (NumPy and numba agree to
      <~0.5%).
    - ``effect_model``: ``"photon_equivalent"`` (default, ``D_eq`` mapping) or
      ``"globle"`` (bounded per-loop effect); ``globle_n_loops`` is GLOBLE's ``N_L``.
    - ``beta_model``: ``"slope_matching"`` (default, Friedrich 2012 Fig. 3) or a Fig. 4
      full simulation ([`local_dose_beta`][], [`discrete_dsb_beta`][],
      [`intertrack_mc_beta`][]). ``alpha_I`` is the single-particle slope in every case.
    - ``fit_survival_bound`` (default ``None``): survival-bounded LQ fit ceiling instead
      of the fixed ``0.05-0.85 D_t`` ([`fit_doses_Gy`][]); off by default because it
      reproduces Fig. 4 worse and is unbounded on radioresistant cells.
    - ``mc_effect_model``: effect model of the intertrack MC, independent of
      ``effect_model`` ([`intertrack_mc_survival`][]).

    Model choices without a paper basis: docs/references/LEM_IV_OPEN_CHOICES.md.
    """

    initial_slope_models: ClassVar[tuple[InitialSlopeModel, ...]] = (
        "cross_section",
        "full_simulation",
    )
    """LEM IV's alpha_I routes: the D -> 0 cross section, or a fit to the simulated
    survival curve (TRiP98's protocol). The ``lethal_*`` models are not defined
    for LEM IV's DSB-pattern override.
    """

    engine: LEM4Engine = "auto"
    effect_model: LEM4EffectModel = "photon_equivalent"
    beta_model: LEM4BetaModel = "slope_matching"
    globle_n_loops: float = GLOBLE_N_LOOPS
    mc_effect_model: LEM4EffectModel = "globle"
    local_dose_kernel: LEM4LocalDoseKernel = "cluster_index"
    discrete_convention: LEM4DiscreteConvention = "per_realisation"
    discrete_survival: LEM4DiscreteSurvival = "mean_effect"
    survival_convention: LEM4SurvivalConvention = "mean_survival"
    cluster_index_model: LEM4ClusterIndexModel = "mean_field"
    cluster_index_mc_realisations: int = _DISCRETE_MC_REALISATIONS
    cluster_index_mc_seed: int = _DISCRETE_MC_SEED
    legacy_trip98_cluster_index_cap: float | None = None
    local_dose_overkill: bool = False
    local_dose_mc_impact_cut: float | Literal["auto"] | None = None
    fit_survival_bound: float | None = None

    def __post_init__(self) -> None:
        LEMBase.__post_init__(self)
        check_choice("engine", self.engine, ENGINES)
        # "scholz1997_two_track" is deliberately absent from LEM4BetaModel: see
        # `scholz1997_two_track_beta` for why it cannot be defined here.
        beta_models = get_args(LEM4BetaModel)
        if self.beta_model not in beta_models:
            raise ValueError(
                f"beta_model must be one of {beta_models!r}, got {self.beta_model!r}. "
                "('scholz1997_two_track' exists for LEM I-III but is undefined for "
                "LEM IV -- its effect is not a pointwise function of the local dose.)"
            )
        check_choice(
            "discrete_convention", self.discrete_convention, get_args(LEM4DiscreteConvention)
        )
        check_choice("discrete_survival", self.discrete_survival, get_args(LEM4DiscreteSurvival))
        check_choice(
            "survival_convention", self.survival_convention, get_args(LEM4SurvivalConvention)
        )
        check_choice(
            "cluster_index_model", self.cluster_index_model, get_args(LEM4ClusterIndexModel)
        )
        if self.cluster_index_mc_realisations < 1:
            raise ValueError(
                "cluster_index_mc_realisations must be >= 1, got "
                f"{self.cluster_index_mc_realisations!r}"
            )
        if self.legacy_trip98_cluster_index_cap is not None and not (
            0.0 < self.legacy_trip98_cluster_index_cap < 1.0
        ):
            raise ValueError(
                "legacy_trip98_cluster_index_cap must be in (0, 1) or None, got "
                f"{self.legacy_trip98_cluster_index_cap!r}"
            )
        if self.discrete_survival == "stochastic" and self.discrete_convention == "mean_counts":
            raise ValueError(
                "discrete_survival='stochastic' needs discrete_convention="
                "'per_realisation': 'mean_counts' forms one cluster index from the "
                "traversal-averaged loop counts, so there is no per-traversal effect "
                "left to average exp(-E) over."
            )
        check_choice("local_dose_kernel", self.local_dose_kernel, get_args(LEM4LocalDoseKernel))
        check_choice("effect_model", self.effect_model, get_args(LEM4EffectModel))
        check_choice("mc_effect_model", self.mc_effect_model, get_args(LEM4EffectModel))
        if self.globle_n_loops <= 0.0:
            raise ValueError(f"globle_n_loops must be > 0, got {self.globle_n_loops!r}")
        if self.fit_survival_bound is not None and not 0.0 < self.fit_survival_bound < 1.0:
            raise ValueError(
                "fit_survival_bound must be a survival level in (0, 1) or None, got "
                f"{self.fit_survival_bound!r}"
            )
        if self.engine == "numba" and not _lem4_kernel.HAVE_NUMBA:
            raise ValueError(
                "engine='numba' requires the 'fast' extra (pip install pyradbio[fast]); "
                "use engine='numpy' or 'auto'."
            )
        if self.photon.cluster is not None:
            warnings.warn(
                "LEM IV does not use photon.cluster: the SSB->DSB enhancement is "
                "modelled through the built-in DSB-yield factor eta (Friedrich 2015), "
                "so a ClusterParameters on the photon object is ignored.",
                UserWarning,
                stacklevel=2,
            )

    def _radial_dose_model(self) -> RadialDoseModel:
        """Velocity-core (``r_c = 6.5 nm``) radical-diffusion profile (Pfuhl 2020 eq. 1)."""
        return ScholzKraftRadialDose(
            radical_diffusion=True,
            velocity_dependent_core=True,
            r_c_nm=LEM4_RC_NM,
            stopping_power=self.stopping_power,
        )

    # ---- discrete DSB pattern of a single central traversal ------------
    def _local_dose_profile(
        self, ion: Ion
    ) -> tuple[npt.NDArray[np.float64], npt.NDArray[np.float64], float, float]:
        """Log-spaced ``(r, D(r))`` and ``(r_min, r_top)`` for the (expensive)
        radical-diffusion radial dose, evaluated once and interpolated below.

        The grid runs to the profile's *support*
        ([`r_support_um`][pyradbio.track.scholz_kraft.ScholzKraftRadialDose.r_support_um]),
        ``8 sigma`` past ``r_max``; callers treat the top radius as "where the track ends",
        and stopping at ``r_max`` would drop most of a 0.1 MeV/u carbon track's energy.
        """
        radial = self._radial()
        r_min = radial.r_min_um(ion)
        r_max = radial.r_support_um(ion)
        r = np.logspace(
            math.log10(r_min * 1.0e-3), math.log10(r_max), _RADIAL_NODES, dtype=np.float64
        )
        dose = np.array(
            [max(radial.local_dose_Gy(float(ri), ion), 0.0) for ri in r], dtype=np.float64
        )
        return r, dose, r_min, r_max

    def _use_numba(self) -> bool:
        """Whether to use the compiled kernel: available + a Scholz-Kraft profile."""
        if self.engine == "numpy":
            return False
        radial = self._radial()
        supported = isinstance(radial, ScholzKraftRadialDose) and radial.radical_diffusion
        if self.engine == "numba":
            if not supported:
                raise ValueError(
                    "engine='numba' needs the built-in Scholz-Kraft radical-diffusion "
                    "radial dose; a custom radial_dose is only supported by engine='numpy'."
                )
            return True
        return _lem4_kernel.HAVE_NUMBA and supported

    def _damage_pattern(self, ion: Ion) -> tuple[float, float, float]:
        """Discrete DSB-pattern loop counts and in-nucleus dose-area
        ``(N_dmg, N_cDSB, in_nucleus_dose_area)`` of one central traversal, as areas
        [um^2] (the loop density ``1/l_DSB^2`` and cylinder height cancel against
        the photon normalisation in every ratio).

        The core is scored on a fine 2-D grid binning the eta-enhanced DSB yield into
        540 nm subvolumes; the penumbra carries isolated DSB only (``P_dmg ~ m``) and is
        added analytically out to ``r_eff``. ``in_nucleus_dose_area = int_0^{r_top} D 2 pi
        r dr`` feeds the escaped-energy slope term. Uses numba or NumPy per
        `_use_numba`.
        """
        if self._use_numba():
            return self._damage_pattern_numba(ion)
        return self._damage_pattern_numpy(ion)

    def _fine_grid_geometry(self) -> tuple[npt.NDArray[np.float64], float, float, int, float]:
        """Shared fine-grid geometry ``(xs, span, ell, nsub, cell_area)`` for the
        central subvolume block (identical for the NumPy and numba paths)."""
        ell = SUBVOLUME_SIDE_NM * 1.0e-3
        half = _CORE_HALF_SUBVOX
        span = (half + 0.5) * ell
        dx = _FINE_STEP_NM * 1.0e-3
        n = max(2, round(2.0 * span / dx))
        xs = (np.arange(n) - 0.5 * n + 0.5) * (2.0 * span / n)
        cell_area = (2.0 * span / n) ** 2
        return xs.astype(np.float64), span, ell, 2 * half + 1, cell_area

    def _damage_pattern_numba(self, ion: Ion) -> tuple[float, float, float]:
        """Compiled-kernel path (`_lem4_kernel.damage_pattern`)."""
        radial = self._radial()
        assert isinstance(radial, ScholzKraftRadialDose)
        r_min = radial.r_min_um(ion)
        r_max = radial.r_max_um(ion)
        sigma = radial.sigma_nm * 1.0e-3
        lam_let = radial._lambda(ion) * self.stopping_power.let_keV_um(ion)
        r_floor = min(r_min, sigma) * 1.0e-3
        log_nodes = np.logspace(math.log10(r_floor), math.log10(r_max), 512, dtype=np.float64)
        # ``r_max`` stays the convolution's *source* domain (delta rays really do
        # stop there, and ``log_nodes`` above spans it); the profile's *support*
        # is 8 sigma wider, and that is what the scoring grid must cover.
        r_support = radial.r_support_um(ion)
        r_grid = np.logspace(
            math.log10(r_min * 1.0e-3), math.log10(r_support), _RADIAL_NODES, dtype=np.float64
        )
        r_eff = self.cell.effective_nucleus_radius_um()
        r_top = min(r_support, r_eff)
        xs, span, ell, nsub, cell_area = self._fine_grid_geometry()
        return _lem4_kernel.damage_pattern(
            r_grid,
            r_min,
            r_max,
            sigma,
            lam_let,
            r_floor,
            log_nodes,
            r_eff,
            r_top,
            xs,
            span,
            ell,
            nsub,
            cell_area,
            DSB_PER_GY_PER_LOOP / (ell * ell),
            DSB_PER_GY_PER_LOOP,
            SSB_PER_GY_CELL,
            DSB_PER_GY_CELL,
            SSB_STAGGER_BP,
            GENOME_SIZE_BP,
        )

    def _damage_pattern_numpy(self, ion: Ion) -> tuple[float, float, float]:
        """Vectorised NumPy path (reference); see `_damage_pattern`."""
        r_grid, dose_grid, _r_min, r_max = self._local_dose_profile(ion)
        log_r = np.log(r_grid)
        log_d = np.log(np.maximum(dose_grid, 1.0e-300))
        r_eff = self.cell.effective_nucleus_radius_um()
        r_top = min(r_max, r_eff)

        def dose_at(r: npt.NDArray[np.float64]) -> npt.NDArray[np.float64]:
            interp = np.exp(np.interp(np.log(np.clip(r, r_grid[0], r_grid[-1])), log_r, log_d))
            interp = np.where(r < r_grid[0], dose_grid[0], interp)
            return np.where(r > r_top, 0.0, interp)

        # Fine 2-D grid over the central (2h+1)^2 subvolumes, track at a subvolume
        # centre (single-central-traversal approximation).
        xs, span, ell, nsub, cell_area = self._fine_grid_geometry()
        gx, gy = np.meshgrid(xs, xs)
        rr = np.sqrt(gx * gx + gy * gy)
        dloc = dose_at(rr)
        # eta-enhanced dose*area per fine cell (the plain dose-area is not
        # summed here -- the penumbra is integrated radially below instead)
        eta_dose = _eta(dloc) * dloc * cell_area
        # bin into subvolume indices
        ix = np.clip(np.floor((gx + span) / ell).astype(np.int64), 0, nsub - 1)
        iy = np.clip(np.floor((gy + span) / ell).astype(np.int64), 0, nsub - 1)
        flat = (ix * nsub + iy).ravel()
        sub_eta_dose = np.bincount(flat, weights=eta_dose.ravel(), minlength=nsub * nsub)
        m = np.asarray(
            (DSB_PER_GY_PER_LOOP / (ell * ell)) * sub_eta_dose, dtype=np.float64
        )  # Poisson mean per subvolume
        n_dmg = ell * ell * float(_p_damaged(m).sum())
        n_cdsb = ell * ell * float(_p_cdsb(m).sum())

        # Penumbra isolated DSB (outside the fine square): P_dmg ~ m ~ kappa_dsb * d
        # (eta ~ 1 there), integrated DIRECTLY over the square's complement with
        # `_outside_square_fraction`, not as (total - fine sum).
        radial_area = dose_at(r_grid) * 2.0 * math.pi * r_grid
        total_dose_area = float(trapezoid(radial_area, x=r_grid))
        n_dmg += DSB_PER_GY_PER_LOOP * float(
            trapezoid(radial_area * _outside_square_fraction(r_grid, span), x=r_grid)
        )
        # total_dose_area is the in-nucleus radial dose-area (dose_at zeroes beyond
        # r_top = min(r_max, r_eff)); used for the escaped-energy slope correction.
        return n_dmg, n_cdsb, total_dose_area

    def cluster_index(self, ion: Ion) -> float:
        """Ion cluster index ``C_I = N_cDSB / (N_iDSB + N_cDSB)`` for one traversal.

        Rises from ~0 at low LET (sparse, isolated DSB) toward 1 at very high LET
        (the dense track core clusters every loop it crosses); Friedrich 2012
        Fig. 2b.

        !!! warning
            A **mean-field** ratio of areas, so by Jensen it runs **high** and misses the
            Pfuhl 2019/2020 Table 3 anchors (12C 2.5 MeV/u ~0.99, 20Ne 31 MeV/u ~0.31 vs
            0.80 / 0.19); the goldens validate the RBE built on it, not this value. For
            the discrete value see [`discrete_cluster_index`][].
        """
        n_dmg, n_cdsb, _ = self._damage_pattern(ion)
        return n_cdsb / n_dmg

    def discrete_cluster_index(
        self, ion: Ion, *, n_realisations: int = 4000, seed: int = 0
    ) -> float:
        """Discrete single-traversal cluster index ``<C_I>`` (Pfuhl Table 3 anchor).

        Samples an explicit **3-D** discrete DSB pattern and averages the
        per-realisation ratio (average-of-ratios), reproducing the Pfuhl 2019/2020
        Table 3 anchors (12C 2.5 MeV/u ~0.80, 20Ne 31 MeV/u ~0.19) that the mean-field
        [`cluster_index`][] cannot.

        One ion crosses at an impact parameter uniform on the disk, depositing
        ``Poisson(a_DSB D_1 <eta>)`` DSBs with radii from the *plain* dose density
        ``D(r) 2 pi r`` (**Variant B**: ``eta`` scales the count, not the placement) and
        depth uniform over ``V / A_eff``, binned into 540 nm cubes (`_bin_dsb_3d`).

        Used by ``cluster_index_model="discrete_mc"``; prefer it to
        [`analytic_discrete_cluster_index`][] **below ~2 MeV/u**, where ``1 - C_I`` lives
        in a thin shell of track positions a regular grid cannot resolve. Raises
        ``ValueError`` if ``n_realisations < 1``.

        !!! warning
            Judge convergence on ``1 - C_I``, never on ``C_I``: its relative noise
            (``~1/sqrt(n)``) is worst where ``C_I`` is highest.
        """
        if n_realisations < 1:
            raise ValueError(f"n_realisations must be >= 1, got {n_realisations!r}")
        r, dose, _r_min, _r_max = self._local_dose_profile(ion)
        density = dose * 2.0 * math.pi * r  # plain dose-area density D(r) 2 pi r
        dose_area = float(trapezoid(density, r))
        if dose_area <= 0.0:
            return 0.0
        cdf = cumulative_trapezoid(density, r, initial=0.0)
        cdf /= cdf[-1]
        eta_weighted = float(trapezoid(_eta(dose) * density, r))
        mean_dsb = DSB_PER_GY_CELL * self.mean_traversal_dose_Gy(ion) * (eta_weighted / dose_area)

        r_eff = self.cell.effective_nucleus_radius_um()
        ell = SUBVOLUME_SIDE_NM * 1.0e-3
        height = NUCLEUS_VOLUME_UM3 / (math.pi * r_eff * r_eff)
        clip2 = (r_eff + ell) ** 2
        rng = np.random.default_rng(seed)
        ci_acc = 0.0
        n_valid = 0
        for _ in range(n_realisations):
            n = int(rng.poisson(mean_dsb))
            if n == 0:
                continue
            radii = np.interp(rng.random(n), cdf, r)
            phi = 2.0 * math.pi * rng.random(n)
            b = r_eff * math.sqrt(rng.random())  # impact parameter, uniform on the disk
            b_theta = 2.0 * math.pi * rng.random()
            x = b * math.cos(b_theta) + radii * np.cos(phi)
            y = b * math.sin(b_theta) + radii * np.sin(phi)
            near = x * x + y * y <= clip2
            x, y = x[near], y[near]
            if x.size == 0:
                continue
            z = height * rng.random(x.size)
            damaged, clustered = _bin_dsb_3d(x, y, z, ell=ell, r_eff=r_eff)
            if damaged > 0:
                ci_acc += clustered / damaged
                n_valid += 1
        return ci_acc / n_valid if n_valid > 0 else 0.0

    def analytic_discrete_cluster_index(
        self, ion: Ion, *, n_impact: int = _DISCRETE_N_IMPACT, n_nodes: int = 256
    ) -> float:
        """Deterministic discrete single-traversal cluster index ``<C_I>``.

        The *average-of-ratios* cluster index by quadrature, without seed noise. Given
        the track position the loop occupancies are independent Poisson, and
        `_discrete.expected_damaged_ratio` closes the ratio expectation into one
        1-D integral; this method supplies the geometry (per-loop means from the
        box-averaged eta-enhanced dose, ``n_z`` layers per column) and marginalises the
        track-to-lattice registration over ``n_impact`` track positions (``n_nodes``
        quadrature nodes). Used by ``cluster_index_model="discrete"``.

        Agrees with [`discrete_cluster_index`][] (under this method's eta convention)
        to 1-7% in ``1 - C_I`` above ~2.5 MeV/u; below ~1 MeV/u it runs 1.4-6x high, as
        `_localdose.box_averaged_profile` smears the track ~2.5x too wide
        (LEM_IV_OPEN_CHOICES.md entry 3.2). It places ``eta`` in the DSB *placement*
        (Variant A), the sampled estimator in the *count* (Variant B; entry 2.4), and does
        **not** reproduce the Pfuhl 2019/2020 Table 3 anchors.

        !!! warning
            Judge convergence on ``1 - C_I``, never on ``C_I``: at ``C_I`` ~ 0.98 a "1%"
            agreement in ``C_I`` is a ~60% error in the ``1 - C_I`` that ``D_eq`` reads.
            Convergence in ``n_impact`` is oscillatory (two node counts are not a test);
            `_DISCRETE_N_IMPACT` holds ``1 - C_I`` to ~1% from 2.5 MeV/u up, only
            ~12% below ~1 MeV/u.
        """
        if n_impact < 1 or n_nodes < 8:
            raise ValueError(f"need n_impact >= 1 and n_nodes >= 8, got {n_impact}, {n_nodes}")
        r, dose, _r_min, r_max = self._local_dose_profile(ion)
        ell = SUBVOLUME_SIDE_NM * 1.0e-3
        r_eff = self.cell.effective_nucleus_radius_um()
        n_z = max(1, round((NUCLEUS_VOLUME_UM3 / (math.pi * r_eff * r_eff)) / ell))

        rho = np.logspace(
            math.log10(float(r[0])), math.log10(r_max + ell), _LOCALDOSE_RHO_NODES, dtype=np.float64
        )
        g_eta = _localdose.box_averaged_profile(
            r, _eta(dose) * dose, rho, ell=ell, engine=self.engine
        )
        log_rho, log_g = np.log(rho), np.log(np.maximum(g_eta, 1.0e-300))

        # Impact parameters uniform on the disk (midpoint rule in b^2), track on the +x
        # axis, each paired with its own R2 lattice phase: one quasi-random sweep over the
        # track-to-lattice registration rather than a product with the phase.
        b_grid = r_eff * np.sqrt((np.arange(n_impact) + 0.5) / n_impact)
        offsets = _r2_points(n_impact) * ell

        total, weight = 0.0, 0.0
        for i, b in enumerate(b_grid):
            cx, cy = _localdose.subvolume_centres(
                r_eff, ell, dx=float(offsets[i, 0]), dy=float(offsets[i, 1])
            )
            dist = np.hypot(cx - b, cy)
            clipped = np.clip(dist, rho[0], rho[-1])
            mu_col = np.exp(np.interp(np.log(clipped), log_rho, log_g))
            mu_col = np.where(dist < rho[0], g_eta[0], mu_col)
            mu_col = np.where(dist > rho[-1], 0.0, mu_col)
            # Every one of the n_z cubes in a column sees the same local dose (the
            # track is uniform along the beam): do NOT divide by n_z.
            mu = np.repeat(DSB_PER_GY_PER_LOOP * mu_col, n_z)
            # Weight by P(this track damages any loop), the sampled MC's own acceptance
            # rule: unweighted, rim tracks that retain no dose would count as perfectly
            # unclustered and inflate ``1 - C_I``.
            w = -math.expm1(-float(mu.sum()))
            if w <= 0.0:
                continue
            total += w * _discrete.expected_damaged_ratio(mu, n_nodes=n_nodes)
            weight += w
        return total / weight if weight > 0.0 else 0.0

    def survival_bounded_ceiling_Gy(self, ion: Ion, survival: float | None = None) -> float:
        """Dose [Gy] at which the *simulated* response reaches ``S = survival``.

        Friedrich 2012 Fig. 4's ``"Full Simulation, S>0.01"`` legend read as a fit
        protocol. The threshold is applied to the simulated effect of the deterministic
        convolution ([`local_dose_lethal_events`][], ``S = e^{-E}``), never to a fitted
        curve, and that one response serves every fitted route.

        ``survival`` defaults to [`fit_survival_bound`][]; ``ValueError`` if neither is
        given or it lies outside ``(0, 1)``. If the threshold is not crossed on the probe
        grid, the grid's top (or bottom) dose is returned rather than raising.
        """
        s_target = self.fit_survival_bound if survival is None else survival
        if s_target is None:
            raise ValueError(
                "survival_bounded_ceiling_Gy needs a survival level: pass one, or set "
                "fit_survival_bound on the model."
            )
        if not 0.0 < s_target < 1.0:
            raise ValueError(f"survival must be in (0, 1), got {s_target!r}")
        d_t = self.photon.D_t
        assert d_t is not None  # enforced in LEMBase.__post_init__
        probe = np.asarray(_SURVIVAL_PROBE_FRACTIONS, dtype=np.float64) * d_t
        effect = self.local_dose_lethal_events(ion, probe)
        target = math.log(-math.log(s_target))
        ln_e = PchipInterpolator(np.log(probe), np.log(np.maximum(effect, 1.0e-300)))
        if float(ln_e(math.log(probe[-1]))) <= target:
            return float(probe[-1])
        if float(ln_e(math.log(probe[0]))) >= target:
            return float(probe[0])
        return float(
            np.exp(
                brentq(lambda x: float(ln_e(x)) - target, math.log(probe[0]), math.log(probe[-1]))
            )
        )

    def fit_doses_Gy(
        self,
        fractions: Iterable[float] | None = None,
        *,
        ion: Ion | None = None,
    ) -> npt.NDArray[np.float64]:
        """Dose grid [Gy] the LQ fits use, as fractions of this cell's ``D_t``.

        Every fitted route ([`local_dose_beta`][], [`local_dose_mc_alpha_beta`][],
        [`discrete_dsb_beta`][], [`full_simulation_initial_slope`][]) reads its window
        here. ``fractions`` defaults to `_LOCALDOSE_DOSE_FRACTIONS`, a pyRadBio
        choice (no paper documents one); quote it with any fitted number.

        With [`fit_survival_bound`][] set and an ``ion`` given, the ceiling becomes
        [`survival_bounded_ceiling_Gy`][] and the fractions are re-mapped onto
        ``[lo, ceiling]`` at their own log positions (lower bound unchanged); without
        ``ion`` the fixed grid is returned. Raises ``ValueError`` if a bounded window gets
        fewer than 2 positive, increasing fractions.
        """
        fr = np.asarray(
            _LOCALDOSE_DOSE_FRACTIONS if fractions is None else tuple(fractions),
            dtype=np.float64,
        )
        d_t = self.photon.D_t
        assert d_t is not None  # enforced in LEMBase.__post_init__
        doses = fr * d_t
        if self.fit_survival_bound is None or ion is None:
            return doses
        if doses.size < 2 or doses[0] <= 0.0 or doses[-1] <= doses[0]:
            raise ValueError(
                "a survival-bounded window needs >= 2 strictly positive, increasing "
                f"fractions, got {fr!r}"
            )
        # Re-map the node shape onto [lo, ceiling] at the nodes' own log positions,
        # so ceiling == doses[-1] reproduces the fixed grid exactly.
        lo, hi = float(doses[0]), self.survival_bounded_ceiling_Gy(ion)
        u = (np.log(doses) - math.log(lo)) / (math.log(float(doses[-1])) - math.log(lo))
        return lo * (hi / lo) ** u

    @staticmethod
    def photon_cluster_index(dose_Gy: float) -> float:
        """Photon cluster index ``C_g(D)`` at a uniform macroscopic dose [Gy].

        Every loop shares the same Poisson mean ``m = kappa_dsb * eta(D) * D`` (the
        eta enhancement applies to the photon DSB yield too), so ``C_g(D) =
        P(cDSB)/P(dmg)``. It grows monotonically from 0 (``D->0``) to 1 (``D->inf``).
        """
        if dose_Gy <= 0.0:
            return 0.0
        d = np.array([dose_Gy])
        m = DSB_PER_GY_PER_LOOP * _eta(d) * d
        return float(_p_cdsb(m)[0] / _p_damaged(m)[0])

    def photon_equivalent_dose(self, cluster_index: float) -> float:
        """Photon-equivalent dose ``D_eq`` solving ``C_g(D_eq) = cluster_index``.

        The macroscopic photon dose whose damage pattern has the same cluster index
        as the ion traversal (Pfuhl eq. 3). Monotone ``C_g`` guarantees a unique
        root; ``cluster_index`` must lie in ``(0, 1)``.
        """
        if not 0.0 < cluster_index < 1.0:
            raise ValueError(f"cluster_index must be in (0, 1), got {cluster_index!r}")
        return float(
            brentq(lambda d: self.photon_cluster_index(d) - cluster_index, _DEQ_LO_GY, _DEQ_HI_GY)
        )

    def _photon_effect(self, dose_Gy: float) -> float:
        """Photon effect ``-ln S_g(D)``: shouldered LQ below ``D_t``, an
        eta-enhanced linear tail above it (Pfuhl eq. 4)."""
        a, b = self.photon.alpha_x, self.photon.beta_x
        d_t = self.photon.D_t
        assert d_t is not None  # enforced in LEMBase.__post_init__
        if dose_Gy < d_t:
            return a * dose_Gy + b * dose_Gy * dose_Gy
        s_max = a + 2.0 * b * d_t
        eta = float(_eta(np.array([dose_Gy]))[0])
        return s_max * (eta * dose_Gy - d_t) + (a * d_t + b * d_t * d_t)

    def _damage_and_deq(self, ion: Ion) -> tuple[float, float, float, float]:
        """Shared ``(N_dmg, N_cDSB, D_eq, in_nucleus_dose_area)`` for the
        scaling/log-kill/initial-slope methods.

        ``D_eq`` comes from [`mean_field_deq`][], which matches the loop's
        Poisson mean directly in the single-loop regime and inverts ``C_g`` above
        it.
        """
        pattern = self._damage_pattern(ion)
        n_dmg, n_cdsb, in_nuc = pattern
        return n_dmg, n_cdsb, self.mean_field_deq(ion, pattern), in_nuc

    def _photon_damaged_loops(self, d_eq: float) -> float:
        """Photon damaged-loop area ``A_eff * P_dmg(m_eq)`` at ``D_eq`` [um^2]."""
        r_eff = self.cell.effective_nucleus_radius_um()
        m_eq = DSB_PER_GY_PER_LOOP * float(_eta(np.array([d_eq]))[0]) * d_eq
        return math.pi * r_eff * r_eff * float(_p_damaged(np.array([m_eq]))[0])

    def scaling_factor(self, ion: Ion) -> float:
        """Damage-count scaling ``kappa`` from ion traversal to photon at ``D_eq``.

        ``kappa = N_total,Ion / N_total,g(D_eq)`` with ``N_total = N_iDSB + N_cDSB``
        (Pfuhl eq. 6): the single traversal produces the same *kind* of pattern as
        photons at ``D_eq`` but only over the "cut-out" it irradiates, so its effect
        is that fraction of the photon effect.
        """
        n_dmg, _n_cdsb, d_eq, _in = self._damage_and_deq(ion)
        return n_dmg / self._photon_damaged_loops(d_eq)

    def single_particle_log_kill(self, ion: Ion) -> float:
        """Lethal events ``N_1 = kappa * (-ln S_g(D_eq))`` from one traversal (Pfuhl
        eqs. 4, 6). The photon effect uses the eta-enhanced shouldered->linear
        response `_photon_effect`.
        """
        n_dmg, _n_cdsb, d_eq, _in = self._damage_and_deq(ion)
        kappa = n_dmg / self._photon_damaged_loops(d_eq)
        return kappa * self._photon_effect(d_eq)

    # ---- GLOBLE per-loop effect (Friedrich, Durante & Scholz 2012) ------
    def _globle_lethalities(self, n_loops: float | None = None) -> tuple[float, float]:
        """Per-loop lethalities ``(eps_iDSB, eps_cDSB)`` of the GLOBLE effect model.

        Friedrich, Durante & Scholz 2012 (Radiat. Res. 178(5):385-394, doi:10.1667/RR2964.1)
        eqs. 8-9: ``eps_i = alpha_x / a_DSB`` (the linear / alpha channel) and
        ``eps_c = 2 (N_L beta_x + a_DSB alpha_x) / a_DSB^2`` (the quadratic / beta
        channel), with the DSB yield ``a_DSB = 30 /Gy`` ([`DSB_PER_GY_CELL`][]) and the
        giant-loop count ``N_L``. Both are < 1 and are the mean lethal events of an
        isolated / clustered damaged loop; folding the Poisson loop counts with them
        reproduces the photon LQ ``alpha_x D + beta_x D^2`` at low dose *when DSBs are
        distributed over exactly ``N_L`` loops*.

        ``n_loops`` defaults to [`globle_n_loops`][]; the intertrack MC passes
        [`MC_GRID_N_LOOPS`][] so that ``eps_c`` matches the grid it bins into.
        """
        n_l = self.globle_n_loops if n_loops is None else n_loops
        a_g, b_g = self.photon.alpha_x, self.photon.beta_x
        a_dsb = DSB_PER_GY_CELL
        eps_i = a_g / a_dsb
        eps_c = 2.0 * (n_l * b_g + a_dsb * a_g) / (a_dsb * a_dsb)
        return eps_i, eps_c

    def _globle_log_kill(self, n_dmg: float, n_cdsb: float) -> float:
        """GLOBLE single-traversal lethal events ``N_1 = n_iDSB eps_i + n_cDSB eps_c``.

        ``n_dmg`` / ``n_cdsb`` are the damaged / clustered loop *areas* [um^2] of one
        transverse layer from `_damage_pattern`. The full-traversal loop *counts*
        follow by multiplying by the geometric giant-loop count ``N_geom = V / l_DSB^3 =
        DSB_PER_GY_CELL / DSB_PER_GY_PER_LOOP`` per effective nuclear area (the track spans
        the whole nuclear depth ``V / A_eff``), i.e. ``n_loop = area * N_geom / A_eff``.
        The effect is bounded by construction -- no ``D_eq`` inversion and no
        ``_CLUSTER_INDEX_MAX`` cap (Pfuhl-model artefacts the GLOBLE approach avoids).
        """
        r_eff = self.cell.effective_nucleus_radius_um()
        a_eff = math.pi * r_eff * r_eff
        n_geom = DSB_PER_GY_CELL / DSB_PER_GY_PER_LOOP  # V / l_DSB^3 giant-loop count
        scale = n_geom / a_eff
        n_i = (n_dmg - n_cdsb) * scale
        n_c = n_cdsb * scale
        eps_i, eps_c = self._globle_lethalities()
        return n_i * eps_i + n_c * eps_c

    def _single_traversal_effect(self, ion: Ion) -> tuple[float, float]:
        """``(N_1, in_nucleus_dose_area)`` per [`effect_model`][] (one scoring pass).

        ``"photon_equivalent"`` (default) is the capped ``kappa * E_eq(D_eq)`` mapping;
        ``"globle"`` is the bounded per-loop `_globle_log_kill`. Both share the
        escaped-energy indirect term added in [`initial_slope`][].

        [`cluster_index_model`][] selects the ``C_I`` estimator; ``n_dmg`` still comes
        from the damage pattern, because ``kappa`` needs the damaged *area*. The discrete
        estimators are bounded by `_DISCRETE_CLUSTER_INDEX_MAX`, not the mean-field
        guard. ``"discrete_mc"`` makes this method **stochastic and slow** (seeded, so
        reproducible per model, but uncached and carrying realisation noise).
        """
        pattern = self._damage_pattern(ion)
        n_dmg, n_cdsb, in_nuc = pattern
        if self.effect_model == "globle":
            return self._globle_log_kill(n_dmg, n_cdsb), in_nuc
        model = self.cluster_index_model
        if model == "trip98_composite":
            # Geometric criterion, no free parameter; see LEM4ClusterIndexModel.
            model = "discrete" if self.pattern_is_single_loop(ion) else "mean_field"
        if model == "discrete":
            d_eq = self.photon_equivalent_dose(
                min(self.analytic_discrete_cluster_index(ion), _DISCRETE_CLUSTER_INDEX_MAX)
            )
        elif model == "discrete_mc":
            d_eq = self.photon_equivalent_dose(
                min(
                    self.discrete_cluster_index(
                        ion,
                        n_realisations=self.cluster_index_mc_realisations,
                        seed=self.cluster_index_mc_seed,
                    ),
                    _DISCRETE_CLUSTER_INDEX_MAX,
                )
            )
        else:
            # The mean field goes through mean_field_deq (matches the loop's Poisson
            # mean below 2.206 MeV/u); the discrete estimators define only C_I, not m.
            d_eq = self.mean_field_deq(ion, pattern)
        kappa = n_dmg / self._photon_damaged_loops(d_eq)
        return kappa * self._photon_effect(d_eq), in_nuc

    def pattern_is_single_loop(self, ion: Ion) -> bool:
        """Does the whole radial profile fit inside one 540 nm giant loop?

        True when the profile support ``r_max + 8 sigma`` is at most half a subvolume
        side (the track is centred on a subvolume): below **2.206 MeV/u for every ion and
        nucleus size**, since ``r_max = 0.062 E^1.7`` depends on energy per nucleon only.
        Purely geometric; it does **not** mean the pattern is degenerate (at 1 MeV/u
        ``C_I`` is 0.12 for protons, ~1 for carbon). It guarantees that one Poisson mean
        ``m`` describes the pattern, so ``D_eq`` can be matched on ``m``
        ([`photon_equivalent_dose_from_dsb_mean`][]).
        """
        return self._radial().r_support_um(ion) <= 0.5 * SUBVOLUME_SIDE_NM * 1.0e-3

    def single_loop_dsb_mean(self, ion: Ion) -> float:
        """Poisson mean ``m`` of the one loop the track occupies (single-loop regime).

        With the whole profile inside one subvolume the fine 2-D grid of
        `_damage_pattern` reduces to the radial integral

            ``m = (kappa_dsb / l_DSB^2) int eta(D) D 2 pi r dr``

        computed here in 1-D: cheaper, engine-independent, and free of the float64
        saturation of ``1 - e^-m`` (above m ~ 37) that makes the 2-D route
        non-invertible.
        """
        r_grid, dose_grid, _r_min, r_support = self._local_dose_profile(ion)
        r_top = min(r_support, self.cell.effective_nucleus_radius_um())
        dose = np.where(r_grid > r_top, 0.0, dose_grid)
        ell = SUBVOLUME_SIDE_NM * 1.0e-3
        return (DSB_PER_GY_PER_LOOP / (ell * ell)) * float(
            trapezoid(_eta(dose) * dose * 2.0 * math.pi * r_grid, x=r_grid)
        )

    def photon_equivalent_dose_from_dsb_mean(self, dsb_mean: float) -> float:
        """``D_eq`` matching a per-loop Poisson mean ``m``, without touching ``C_I``.

        A uniform photon dose gives every loop the same mean ``m_g = kappa_dsb eta(D) D``
        and ``C_g`` is strictly monotone in ``m_g``, so for a single-loop ion pattern
        ``C_g(D_eq) = C_I`` and ``m_g(D_eq) = m`` are the same equation (Pfuhl eq. 3). The
        ``m`` form avoids the cancellation in ``1 - C_I = m e^-m / (1 - e^-m)``, which
        underflows near m ~ 37. Raises ``ValueError`` if ``dsb_mean <= 0``.
        """
        if dsb_mean <= 0.0:
            raise ValueError(f"dsb_mean must be positive, got {dsb_mean!r}")

        def residual(dose_Gy: float) -> float:
            d = np.array([dose_Gy])
            return float(DSB_PER_GY_PER_LOOP * _eta(d)[0] * dose_Gy) - dsb_mean

        return float(brentq(residual, _DEQ_LO_GY, _DEQ_HI_GY))

    def mean_field_deq(self, ion: Ion, pattern: tuple[float, float, float] | None = None) -> float:
        """``D_eq`` for the mean-field pattern, by whichever route is well posed.

        Single-loop regime (below 2.206 MeV/u): match the loop's Poisson mean
        directly. Otherwise: the usual ``C_g^-1(C_I)`` root-find, since a
        multi-loop pattern has no single ``m`` to match.

        Setting [`legacy_trip98_cluster_index_cap`][] forces the ``C_I`` route
        everywhere and clips the index at that cap. ``pattern`` passes an
        already-computed `_damage_pattern` result so it is not scored twice.
        """
        cap = self.legacy_trip98_cluster_index_cap
        if cap is None and self.pattern_is_single_loop(ion):
            return self.photon_equivalent_dose_from_dsb_mean(self.single_loop_dsb_mean(ion))
        n_dmg, n_cdsb, _ = pattern if pattern is not None else self._damage_pattern(ion)
        return self.photon_equivalent_dose(
            self._mean_field_cluster_index_for_inversion(ion, n_cdsb / n_dmg)
        )

    def mean_field_pattern_is_degenerate(self, ion: Ion) -> bool:
        """Has the mean-field pattern lost the information ``D_eq`` needs?

        True only when the ``C_I`` route is non-invertible (``1 - C_I`` below the float64
        bound of `_CLUSTER_INDEX_MAX`). That is set by track density, not by the
        single-loop geometry of [`pattern_is_single_loop`][] (protons in that regime
        invert cleanly). In the single-loop regime [`mean_field_deq`][] matches ``m``
        instead, so the predicate matters for the multi-loop branch and for callers
        checking the ``C_I`` route itself.
        """
        if self.pattern_is_single_loop(ion):
            m = self.single_loop_dsb_mean(ion)
            return bool(
                m <= 0.0 or 1.0 - _cluster_index_from_dsb_mean(m) <= 1.0 - _CLUSTER_INDEX_MAX
            )
        n_dmg, n_cdsb, _ = self._damage_pattern(ion)
        return bool(n_dmg <= 0.0 or n_cdsb / n_dmg >= _CLUSTER_INDEX_MAX)

    def _mean_field_cluster_index_for_inversion(self, ion: Ion, c_i: float) -> float:
        """``C_I`` for the ``D_eq`` inversion, or raise if the mean field has degenerated.

        Raises ``ValueError`` rather than clipping: any clip value (such as 0.95) is a
        free parameter that would fabricate the magnitude. Set
        [`legacy_trip98_cluster_index_cap`][] to restore the capped behaviour.
        """
        cap = self.legacy_trip98_cluster_index_cap
        if cap is not None:
            return min(c_i, cap)
        if c_i < _CLUSTER_INDEX_MAX and not self.mean_field_pattern_is_degenerate(ion):
            return c_i
        raise ValueError(
            f"the mean-field damage pattern is fully clustered at {ion.Z}-{ion.A} "
            f"{ion.energy_MeV_u:g} MeV/u (1 - C_I is below the float64 bound "
            f"{1.0 - _CLUSTER_INDEX_MAX:g}), so D_eq = C_g^-1(C_I) cannot be "
            "inverted. Reachable only on the multi-loop branch: below 2.206 MeV/u "
            "the pattern is a single loop and LEM4.mean_field_deq matches its "
            "Poisson mean instead, which does not saturate. Use "
            "cluster_index_model='discrete' or 'discrete_mc', or set "
            "legacy_trip98_cluster_index_cap=0.95 to clip the index at that cap."
        )

    def single_traversal_log_kill(self, ion: Ion) -> float:
        """``N_1`` from LEM IV's **own** effect model, not the base local-dose integral.

        Overrides [`LEMBase`][]'s ``-ln S_x(D(r))`` disk integral (the LEM I-III
        lethal-event density) with LEM IV's DSB-pattern effect
        (`_single_traversal_effect`), so the inherited ``beta_model="rapid_lem"``
        is correct here instead of silently returning a LEM I number. In-nucleus dose
        only; the escaped-energy term is added by [`initial_slope`][].
        """
        return self._single_traversal_effect(ion)[0]

    def scholz1997_montecarlo_survival(
        self, *args: object, **kwargs: object
    ) -> npt.NDArray[np.float64]:
        """Not defined for LEM IV -- always raises.

        LEMBase's local-dose Monte Carlo scores ``exp(-(1/A) int h(d(x)) dx)`` with
        ``h`` pointwise in the local dose; LEM IV's effect is a nonlinear function of a
        global DSB-pattern statistic. ``beta_model="intertrack_mc"`` is LEM IV's own
        stochastic path.
        """
        raise NotImplementedError(
            "LEM IV has no local-dose Monte Carlo: its lethal-event density is not a "
            "pointwise function of the local dose, so LEMBase's realization score does "
            "not describe it. Use beta_model='intertrack_mc' for LEM IV's own "
            "stochastic averaging."
        )

    def scholz1997_montecarlo_beta(self, *args: object, **kwargs: object) -> float:
        """Not defined for LEM IV -- always raises (see above)."""
        raise NotImplementedError(
            "LEM IV has no local-dose Monte Carlo; use beta_model='intertrack_mc'."
        )

    def scholz1997_traversal_conditioned_survival(
        self, *args: object, **kwargs: object
    ) -> TraversalConditionedSurvival:
        """Not defined for LEM IV -- always raises (see above)."""
        raise NotImplementedError(
            "LEM IV has no local-dose Monte Carlo; use beta_model='intertrack_mc'."
        )

    def scholz1997_two_track_beta(self, ion: Ion, **kwargs: object) -> float:
        """Not defined for LEM IV -- always raises.

        The pair expansion needs the lethal-event functional to be an integral of a
        *pointwise* function of the local dose, ``nu = (1/A) int h(d(x)) dx``. LEM IV's
        effect is a nonlinear function of a **global** DSB-pattern statistic (``C_I``,
        then ``D_eq``), so two overlapping tracks do not decompose. This is a property of
        the model. Use ``beta_model="intertrack_mc"``.
        """
        del ion, kwargs
        raise NotImplementedError(
            "scholz1997_two_track_beta is not defined for LEM IV: its effect is a "
            "function of a global DSB-pattern statistic, not a pointwise function "
            "of the local dose, so the O(phi^2) pair expansion has nothing to "
            'expand. Use beta_model="intertrack_mc" instead.'
        )

    # ---- initial slope (overrides the cross-section formulation) --------
    def initial_slope(self, ion: Ion) -> float:
        r"""Ion initial slope ``alpha_I`` (single-particle, with escaped-energy term).

        A *direct* central-traversal term plus an *indirect* correction:

            alpha_I = (1 - e^-N_1) / D_1  +  alpha_x * (1 - D_1^in / D_1) .

        ``N_1`` comes from [`effect_model`][]; the direct term is the Poisson
        single-track average (Scholz 1997) on the *in-nucleus* dose. The indirect term
        is the impact-parameter cross-section of tracks with ``b > R_eff`` in its linear
        limit (a weak photon field, ``alpha_x`` per unit escaped dose;
        ``D_1^in = in_nucleus_dose_area / A_eff``): zero when ``r_max <= R_eff``, it
        restores ``alpha_I -> alpha_x`` at low LET. The full impact-parameter integral
        over the discrete pattern is not implemented.

        ``survival_convention="mean_effect"`` replaces ``1 - e^-N_1`` by ``N_1``
        ([`LEM4SurvivalConvention`][]); ``initial_slope_model="full_simulation"`` fits
        the slope instead ([`full_simulation_initial_slope`][]). The ``lethal_central``
        / ``lethal_full`` models are not defined for LEM IV.
        """
        if self.initial_slope_model == "full_simulation":
            return self.full_simulation_initial_slope(ion)
        n_1, in_nuc = self._single_traversal_effect(ion)
        d_1 = self.mean_traversal_dose_Gy(ion)
        r_eff = self.cell.effective_nucleus_radius_um()
        frac_in = in_nuc / (math.pi * r_eff * r_eff * d_1)  # escaped when < 1
        saturated = self.survival_convention == "mean_survival"
        direct = (1.0 - math.exp(-n_1)) / d_1 if saturated else n_1 / d_1
        indirect = self.photon.alpha_x * max(1.0 - frac_in, 0.0)
        return direct + indirect

    def full_simulation_initial_slope(self, ion: Ion) -> float:
        r"""``alpha_I`` *fitted* to a simulated survival curve -- the TRiP98 protocol.

        TRiP98's LEM IV tables come from the "full simulation" (``-iter 50000``; Pfuhl
        2021 thesis ch. 2.3.2: hit classes weighted "according to the requested
        irradiation dose level"), i.e. an LQ fit over a finite dose window rather than
        the ``D -> 0`` limit of [`initial_slope`][]. Under overkill ``-ln S(D)`` is
        concave, so the fitted ``beta`` goes negative and lifts ``alpha`` above the true
        initial slope (1.0x at 10 MeV/u, ~1.4x by 1 MeV/u). See
        docs/references/LEM_IV_Friedrich2012.md.

        Computed as [`local_dose_mc_alpha_beta`][] averaged over `_FULLSIM_SEEDS`
        (deterministic); [`survival_convention`][] selects whether the fit runs on
        ``-ln <e^-E>`` or on ``<E>``. Use it in the overkill band only: cost grows as
        ``(R_eff + r_max)^2`` (about a minute at 50 MeV/u carbon; warns past
        `_FULLSIM_DOMAIN_WARN`) while above it the two slopes agree to a few %. The
        window ([`fit_doses_Gy`][]) moves the slope by up to ~10%; quote it.
        """
        r_eff = self.cell.effective_nucleus_radius_um()
        ratio = (r_eff + self._radial().r_support_um(ion)) / r_eff
        if ratio > _FULLSIM_DOMAIN_WARN:
            warnings.warn(
                f"full_simulation at E={ion.energy_MeV_u:g} MeV/u samples a domain "
                f"{ratio:.0f}x the nuclear radius (r_max grows as E^1.7), so this call "
                "costs minutes and scales as the square of that. The fitted and "
                "analytic slopes agree to a few % this far from overkill -- use "
                "initial_slope_model='cross_section' above ~20 MeV/u.",
                stacklevel=2,
            )
        overkill = self.survival_convention == "mean_survival"
        alphas = [
            self.local_dose_mc_alpha_beta(ion, seed=seed, overkill=overkill)[0]
            for seed in _FULLSIM_SEEDS
        ]
        return float(np.mean(alphas))

    # ---- ion beta (slope-matching default / intertrack Monte-Carlo) -----
    def _beta_for_ion(self, ion: Ion, alpha_ion: float) -> float:
        """``beta_I`` per [`beta_model`][] (default Kraemer-Scholz slope matching).

        ``"local_dose"`` runs the Friedrich 2012 Fig. 4 full simulation as a
        local-dose convolution ([`local_dose_beta`][]; deterministic, ~1 s per
        ion); ``"intertrack_mc"`` runs the explicit-track discrete-DSB Monte-Carlo
        ([`intertrack_mc_beta`][]), which is slow and does not reproduce Fig. 4.
        """
        if self.beta_model == "local_dose":
            return self.local_dose_beta(ion)
        if self.beta_model == "discrete_dsb":
            return self.discrete_dsb_beta(ion)
        if self.beta_model == "intertrack_mc":
            return self.intertrack_mc_beta(ion)
        if self.beta_model == "rapid_lem":
            return self.rapid_lem_beta(ion)
        return self._beta(alpha_ion)

    def _globle_single_particle_slope(self, ion: Ion) -> float:
        """Single-particle ``alpha_I`` under the GLOBLE effect (adaptive-dose helper).

        The same direct + escaped-energy assembly as [`initial_slope`][], but with
        the bounded GLOBLE per-loop ``N_1`` regardless of [`effect_model`][] -- the
        intertrack MC uses the GLOBLE effect, and Friedrich 2012 fixes ``alpha_I`` as
        model-invariant, so this alpha both sets the adaptive dose scale and is the
        cross-check the fitted MC ``alpha`` must reproduce.
        """
        n_dmg, n_cdsb, in_nuc = self._damage_pattern(ion)
        n_1 = self._globle_log_kill(n_dmg, n_cdsb)
        d_1 = self.mean_traversal_dose_Gy(ion)
        r_eff = self.cell.effective_nucleus_radius_um()
        frac_in = in_nuc / (math.pi * r_eff * r_eff * d_1)
        direct = (1.0 - math.exp(-n_1)) / d_1
        indirect = self.photon.alpha_x * max(1.0 - frac_in, 0.0)
        return direct + indirect

    def _intertrack_setup(
        self, ion: Ion
    ) -> tuple[float, npt.NDArray[np.float64], npt.NDArray[np.float64], float, float, float, float]:
        """Precompute the ion's shared-grid MC inputs (one radial pass).

        Returns ``(m_total, cdf, cdf_r, d_1, r_eff, r_sim, height)`` with ``m_total``
        the mean DSB per track (``= 30 D_1`` for ``eta -> 1``), ``(cdf, cdf_r)`` the
        inverse-CDF nodes of the transverse DSB pdf ``p(r) ~ eta(D) D r``, and
        ``r_sim = R_eff + r_max`` the extended-domain radius.
        """
        r, dose, _r_min, r_max = self._local_dose_profile(ion)
        weight = _eta(dose) * dose * r
        cdf, cdf_r, integral = _intertrack_kernel.build_radial_cdf(r, weight)
        r_eff = self.cell.effective_nucleus_radius_um()
        a_eff = math.pi * r_eff * r_eff
        height = NUCLEUS_VOLUME_UM3 / a_eff
        ell = SUBVOLUME_SIDE_NM * 1.0e-3
        kappa_over_l2 = DSB_PER_GY_PER_LOOP / (ell * ell)
        n_z = height / ell
        m_total = kappa_over_l2 * n_z * 2.0 * math.pi * integral
        d_1 = self.mean_traversal_dose_Gy(ion)
        return m_total, cdf, cdf_r, d_1, r_eff, r_eff + r_max, height

    def _intertrack_pe_lookup(self) -> tuple[npt.NDArray[np.float64], npt.NDArray[np.float64]]:
        """``(C_g, g)`` lookup for the photon-equivalent MC scoring.

        A pattern with cluster index ``C_I`` and ``N_dmg`` damaged loops has effect
        ``E = N_dmg * g(C_I)`` with ``C_g(D_eq) = C_I`` (Pfuhl eqs. 3/6) and:

        ```text
        g(C_I) = _photon_effect(D_eq) / (MC_GRID_N_LOOPS * P_dmg(m(D_eq)))
        ```

        i.e. the photon effect per damaged photon loop, tabulated once against the
        monotone ``C_g`` so callers need an ``interp`` instead of a per-sample ``brentq``.
        """
        d = np.logspace(-3.0, 5.0, 1400, dtype=np.float64)
        eta = _eta(d)
        m = DSB_PER_GY_PER_LOOP * eta * d
        c_g = _p_cdsb(m) / _p_damaged(m)  # photon cluster index C_g(D), monotone in D
        n_dmg_g = MC_GRID_N_LOOPS * _p_damaged(m)  # photon damaged-loop *count* at D
        a, b = self.photon.alpha_x, self.photon.beta_x
        d_t = self.photon.D_t
        assert d_t is not None  # enforced in LEMBase.__post_init__
        s_max = a + 2.0 * b * d_t
        # -ln S_gamma(D): shouldered LQ below D_t, eta-enhanced linear tail above (eq. 4).
        eff = np.where(
            d < d_t,
            a * d + b * d * d,
            s_max * (eta * d - d_t) + (a * d_t + b * d_t * d_t),
        )
        return c_g.astype(np.float64), (eff / n_dmg_g).astype(np.float64)

    def _pe_lookup_by_unclustered(self) -> tuple[npt.NDArray[np.float64], npt.NDArray[np.float64]]:
        """``(u, g)`` -- the same photon-equivalent lookup, keyed on ``1 - C_g``.

        ``C_g`` saturates to exactly 1.0 over the top nodes of
        `_intertrack_pe_lookup` (a tie spanning 193x in ``g``, undefined for
        ``np.interp``); the unclustered fraction ``u = m e^-m / (1 - e^-m)``, formed
        directly, stays strictly monotone. Returned ascending in ``u``. Callers must form
        ``u`` **without** subtracting ``C_I`` from 1, or the fix is undone.
        """
        _c_g, g_per_loop = self._intertrack_pe_lookup()
        # Keyed on m directly, not on 1 - c_g, which cancels near the top.
        d = np.logspace(-3.0, 5.0, g_per_loop.size, dtype=np.float64)
        m = DSB_PER_GY_PER_LOOP * _eta(d) * d
        with np.errstate(under="ignore", over="ignore"):
            u = np.where(m < _LOG_MAX_FLOAT, m * np.exp(-m) / -np.expm1(-m), 0.0)
        keep = u > 0.0
        return u[keep][::-1].copy(), g_per_loop[keep][::-1].copy()

    def _adaptive_doses(self, ion: Ion, n_doses: int) -> npt.NDArray[np.float64]:
        """Per-LET LQ-region dose grid keeping the effect in ~[0.2, 2].

        ``D_max = min(target/alpha_sp, frac*D_t)`` from the single-particle GLOBLE
        ``alpha`` (`_globle_single_particle_slope`): a fixed grid fails across
        LET because ``alpha*D >> 1`` at high LET breaks the LQ fit.
        """
        alpha_sp = self._globle_single_particle_slope(ion)
        d_t = self.photon.D_t
        assert d_t is not None  # enforced in LEMBase.__post_init__
        d_max = min(
            _INTERTRACK_EFFECT_TARGET / max(alpha_sp, 1.0e-6),
            _INTERTRACK_DMAX_DT_FRACTION * d_t,
        )
        return np.linspace(0.15 * d_max, d_max, n_doses, dtype=np.float64)

    def intertrack_mc_survival(
        self,
        ion: Ion,
        doses_Gy: Iterable[float],
        *,
        n_realizations: int = _INTERTRACK_REALIZATIONS,
        seed: int = 0,
        engine: LEM4MCEngine = "numpy",
    ) -> npt.NDArray[np.float64]:
        """Intertrack-cDSB full-simulation survival ``S(D)`` (Friedrich 2012 Fig. 4).

        Many Poisson-distributed tracks over the extended domain
        ``R_sim = R_eff + r_max`` place discrete DSBs into a **shared** 540 nm grid
        (so DSBs from different tracks cluster; intertrack cDSB), scored with common
        random numbers. ``doses_Gy`` are sorted ascending. ``engine`` selects the
        backend: ``"numpy"`` (default, deterministic for a fixed ``seed``), ``"numba"``
        (the compiled parallel kernel; faster but not bit-reproducible), or ``"auto"``.

        Scoring follows [`mc_effect_model`][]: ``"globle"`` (default) with ``eps_c``
        calibrated to [`MC_GRID_N_LOOPS`][] so the photon limit recovers ``beta_x``; or
        ``"photon_equivalent"`` (experimental), whose ``beta`` *rises* through mid-LET
        instead of collapsing -- a diagnostic ruling out the effect model as the cause of
        the Fig. 4 collapse (docs/references/LEM_IV_Friedrich2012.md).
        """
        doses = np.array(sorted(doses_Gy), dtype=np.float64)
        m_total, cdf, cdf_r, d_1, r_eff, r_sim, height = self._intertrack_setup(ion)
        ell = SUBVOLUME_SIDE_NM * 1.0e-3

        if self.mc_effect_model == "photon_equivalent":
            # Cluster-index -> D_eq -> S_gamma(D_eq) scoring. Both backends supported so
            # low-LET points (huge domain extension) stay tractable on the numba path.
            ci_grid, g_grid = self._intertrack_pe_lookup()
            if self._use_intertrack_numba(engine):
                return _intertrack_kernel.ion_survival_photon_equivalent_numba(
                    doses,
                    m_total=m_total,
                    cdf=cdf,
                    cdf_r=cdf_r,
                    ci_grid=ci_grid,
                    g_grid=g_grid,
                    d_1=d_1,
                    r_eff=r_eff,
                    r_sim=r_sim,
                    height=height,
                    ell=ell,
                    ci_max=_CLUSTER_INDEX_MAX,
                    n_real=n_realizations,
                )
            rng = np.random.default_rng(seed)
            return _intertrack_kernel.ion_survival_photon_equivalent_numpy(
                doses,
                m_total=m_total,
                cdf=cdf,
                cdf_r=cdf_r,
                ci_grid=ci_grid,
                g_grid=g_grid,
                d_1=d_1,
                r_eff=r_eff,
                r_sim=r_sim,
                height=height,
                ell=ell,
                ci_max=_CLUSTER_INDEX_MAX,
                n_real=n_realizations,
                rng=rng,
            )

        # eps_c calibrated to the geometric grid count, not globle_n_loops (photon limit).
        eps_i, eps_c = self._globle_lethalities(n_loops=MC_GRID_N_LOOPS)
        if self._use_intertrack_numba(engine):
            return _intertrack_kernel.ion_survival_numba(
                doses,
                m_total=m_total,
                cdf=cdf,
                cdf_r=cdf_r,
                eps_i=eps_i,
                eps_c=eps_c,
                d_1=d_1,
                r_eff=r_eff,
                r_sim=r_sim,
                height=height,
                ell=ell,
                n_real=n_realizations,
            )
        rng = np.random.default_rng(seed)
        return _intertrack_kernel.ion_survival_numpy(
            doses,
            m_total=m_total,
            cdf=cdf,
            cdf_r=cdf_r,
            eps_i=eps_i,
            eps_c=eps_c,
            d_1=d_1,
            r_eff=r_eff,
            r_sim=r_sim,
            height=height,
            ell=ell,
            n_real=n_realizations,
            rng=rng,
        )

    @staticmethod
    def _use_intertrack_numba(engine: LEM4MCEngine) -> bool:
        """Resolve the intertrack MC backend choice to a bool (validates ``"numba"``)."""
        check_choice("engine", engine, ENGINES)
        if engine == "numba" and not _intertrack_kernel.HAVE_NUMBA:
            raise ValueError(
                "engine='numba' requires the 'fast' extra (pip install pyradbio[fast]); "
                "use engine='numpy' or 'auto'."
            )
        if engine == "auto":
            return _intertrack_kernel.HAVE_NUMBA
        return engine == "numba"

    def intertrack_mc_beta(
        self,
        ion: Ion,
        *,
        doses_Gy: Iterable[float] | None = None,
        n_realizations: int = _INTERTRACK_REALIZATIONS,
        n_doses: int = _INTERTRACK_N_DOSES,
        seed: int = 0,
        engine: LEM4MCEngine = "numpy",
    ) -> float:
        """Full-simulation ion ``beta_I`` (Friedrich 2012 Fig. 4), the reference engine.

        Runs [`intertrack_mc_survival`][] on an adaptive per-LET dose grid (unless
        ``doses_Gy`` is given) and fits ``-ln S(D)/D = alpha + beta D``, returning
        ``beta``; the fitted ``alpha`` cross-checks the single-particle GLOBLE ``alpha``.
        ``engine`` selects the backend (see [`intertrack_mc_survival`][]).

        The default grid is LET-dependent (`_adaptive_doses`) and a narrower window
        reads a higher ``beta``; over one common window this engine's ``beta`` is flat at
        ``beta_x``, so pass explicit ``doses_Gy`` to compare LET points. A single call is
        one noisy draw: use [`intertrack_mc_beta_jackknife`][] for an error bar. A
        negative fit is clamped to 0 with a warning (the jackknife path does not clamp).
        """
        doses = (
            self._adaptive_doses(ion, n_doses)
            if doses_Gy is None
            else np.array(sorted(doses_Gy), dtype=np.float64)
        )
        survival = self.intertrack_mc_survival(
            ion, doses, n_realizations=n_realizations, seed=seed, engine=engine
        )
        _alpha, beta = _intertrack_kernel.fit_alpha_beta(doses, survival)
        if beta < 0.0:
            warnings.warn(
                f"intertrack-MC beta fit returned {beta:.3e} < 0 for Z={ion.Z} at "
                f"{ion.energy_MeV_u:g} MeV/u and was clamped to 0. The simulated dose "
                "response is saturated here, so the low-dose LQ fit has no curvature "
                "left to measure; treat this beta as a floor, not a value, and use "
                "intertrack_mc_beta_jackknife for the unclamped value and its error.",
                UserWarning,
                stacklevel=2,
            )
        return max(beta, 0.0)

    def intertrack_mc_batches(
        self,
        ion: Ion,
        *,
        doses_Gy: Iterable[float] | None = None,
        n_batches: int = _INTERTRACK_BATCHES,
        realizations_per_batch: int = _INTERTRACK_REALIZATIONS_PER_BATCH,
        n_doses: int = _INTERTRACK_N_DOSES,
        seed: int = 0,
        engine: LEM4MCEngine = "numpy",
    ) -> IntertrackSurvival:
        """Batched intertrack dose response -> [`IntertrackSurvival`][].

        The same simulation as [`intertrack_mc_survival`][], partitioned into
        independent batches so a delete-one jackknife can put an error bar on any
        functional of the curve. On the ``"numpy"`` engine the realisations are
        drawn in one uninterrupted stream, so the pooled curve reproduces the
        unbatched call for the same ``seed``; the ``"numba"`` engine is not
        seed-reproducible either way (per-thread RNG).
        """
        doses = (
            self._adaptive_doses(ion, n_doses)
            if doses_Gy is None
            else np.array(sorted(doses_Gy), dtype=np.float64)
        )
        m_total, cdf, cdf_r, d_1, r_eff, r_sim, height = self._intertrack_setup(ion)
        ell = SUBVOLUME_SIDE_NM * 1.0e-3
        use_numba = self._use_intertrack_numba(engine)

        if self.mc_effect_model == "photon_equivalent":
            ci_grid, g_grid = self._intertrack_pe_lookup()
            if use_numba:
                batch_effect = _intertrack_kernel.ion_survival_photon_equivalent_batches_numba(
                    doses,
                    m_total=m_total,
                    cdf=cdf,
                    cdf_r=cdf_r,
                    ci_grid=ci_grid,
                    g_grid=g_grid,
                    d_1=d_1,
                    r_eff=r_eff,
                    r_sim=r_sim,
                    height=height,
                    ell=ell,
                    ci_max=_CLUSTER_INDEX_MAX,
                    n_batches=n_batches,
                    realizations_per_batch=realizations_per_batch,
                )
            else:
                batch_effect = _intertrack_kernel.ion_survival_photon_equivalent_batches_numpy(
                    doses,
                    m_total=m_total,
                    cdf=cdf,
                    cdf_r=cdf_r,
                    ci_grid=ci_grid,
                    g_grid=g_grid,
                    d_1=d_1,
                    r_eff=r_eff,
                    r_sim=r_sim,
                    height=height,
                    ell=ell,
                    ci_max=_CLUSTER_INDEX_MAX,
                    n_batches=n_batches,
                    realizations_per_batch=realizations_per_batch,
                    rng=np.random.default_rng(seed),
                )
        else:
            eps_i, eps_c = self._globle_lethalities(n_loops=MC_GRID_N_LOOPS)
            if use_numba:
                batch_effect = _intertrack_kernel.ion_survival_batches_numba(
                    doses,
                    m_total=m_total,
                    cdf=cdf,
                    cdf_r=cdf_r,
                    eps_i=eps_i,
                    eps_c=eps_c,
                    d_1=d_1,
                    r_eff=r_eff,
                    r_sim=r_sim,
                    height=height,
                    ell=ell,
                    n_batches=n_batches,
                    realizations_per_batch=realizations_per_batch,
                )
            else:
                batch_effect = _intertrack_kernel.ion_survival_batches_numpy(
                    doses,
                    m_total=m_total,
                    cdf=cdf,
                    cdf_r=cdf_r,
                    eps_i=eps_i,
                    eps_c=eps_c,
                    d_1=d_1,
                    r_eff=r_eff,
                    r_sim=r_sim,
                    height=height,
                    ell=ell,
                    n_batches=n_batches,
                    realizations_per_batch=realizations_per_batch,
                    rng=np.random.default_rng(seed),
                )
        return IntertrackSurvival(
            batch_effect=batch_effect,
            doses_Gy=doses,
            realizations_per_batch=realizations_per_batch,
        )

    def intertrack_mc_beta_jackknife(
        self,
        ion: Ion,
        *,
        doses_Gy: Iterable[float] | None = None,
        n_batches: int = _INTERTRACK_BATCHES,
        realizations_per_batch: int = _INTERTRACK_REALIZATIONS_PER_BATCH,
        n_doses: int = _INTERTRACK_N_DOSES,
        seed: int = 0,
        engine: LEM4MCEngine = "numpy",
    ) -> tuple[float, float]:
        """Full-simulation ``beta_I`` **with its standard error** -> ``(beta, sem)``.

        Prefer this over [`intertrack_mc_beta`][] whenever the number is compared to
        anything: the per-seed spread can exceed the value at high LET. The error is a
        delete-one jackknife over batches (``beta`` is a nonlinear functional of the
        pooled curve); the returned ``beta`` is **not clamped**
        ([`IntertrackSurvival.jackknife_beta`][]).
        """
        return self.intertrack_mc_batches(
            ion,
            doses_Gy=doses_Gy,
            n_batches=n_batches,
            realizations_per_batch=realizations_per_batch,
            n_doses=n_doses,
            seed=seed,
            engine=engine,
        ).jackknife_beta()

    def intertrack_mc_beta_to_precision(
        self,
        ion: Ion,
        *,
        target_sem: float,
        doses_Gy: Iterable[float] | None = None,
        n_batches: int = _INTERTRACK_BATCHES,
        realizations_per_batch: int = _INTERTRACK_REALIZATIONS_PER_BATCH,
        max_realizations: int = _INTERTRACK_MAX_REALIZATIONS,
        n_doses: int = _INTERTRACK_N_DOSES,
        seed: int = 0,
        engine: LEM4MCEngine = "numpy",
    ) -> tuple[float, float, int]:
        """``beta_I`` to a requested precision -> ``(beta, standard_error, n_realizations)``.

        Runs a pilot, then keeps *adding* batches (never restarting -- see
        [`IntertrackSurvival.pooled_with`][]) until the jackknife error meets
        ``target_sem`` or ``max_realizations`` is reached.

        ``target_sem`` is **absolute** (Gy^-2), since ``beta_I`` passes through zero in
        deep overkill; judge it against ``beta_x``. Raises ``ValueError`` unless it is
        finite and positive. If ``max_realizations`` binds first, the result is
        returned with a ``UserWarning``.
        """
        if not math.isfinite(target_sem) or target_sem <= 0.0:
            raise ValueError(f"target_sem must be finite and > 0, got {target_sem!r}")

        curve = self.intertrack_mc_batches(
            ion,
            doses_Gy=doses_Gy,
            n_batches=n_batches,
            realizations_per_batch=realizations_per_batch,
            n_doses=n_doses,
            seed=seed,
            engine=engine,
        )
        round_index = 0
        while True:
            beta, sem = curve.jackknife_beta()
            # A nan error means the jackknife could not form replicates; more
            # realisations will not fix that, so stop rather than spin.
            if not math.isfinite(sem) or sem <= target_sem:
                break
            if curve.n_realizations >= max_realizations:
                warnings.warn(
                    f"intertrack beta for Z={ion.Z} at {ion.energy_MeV_u:g} MeV/u stopped at "
                    f"{curve.n_realizations} realizations (max_realizations) with standard "
                    f"error {sem:.4g} > target_sem {target_sem:.4g}. The returned beta is "
                    "the best available at this budget, not a converged value; raise "
                    "max_realizations or relax target_sem.",
                    UserWarning,
                    stacklevel=2,
                )
                break
            required = (sem / target_sem) ** 2
            target_total = min(
                max_realizations,
                math.ceil(curve.n_realizations * min(required, _INTERTRACK_MAX_ALLOCATION_GROWTH)),
            )
            extra = max(target_total - curve.n_realizations, realizations_per_batch)
            extra_batches = max(2, math.ceil(extra / realizations_per_batch))
            round_index += 1
            curve = curve.pooled_with(
                self.intertrack_mc_batches(
                    ion,
                    doses_Gy=curve.doses_Gy,
                    n_batches=extra_batches,
                    realizations_per_batch=realizations_per_batch,
                    seed=seed + 7919 * round_index,
                    engine=engine,
                )
            )
        return beta, sem, curve.n_realizations

    # ---- local-dose compound-Poisson convolution (Fig. 4 full simulation) ----
    def _local_dose_jumps(
        self, ion: Ion, *, eta_enhanced: bool = False
    ) -> tuple[npt.NDArray[np.float64], npt.NDArray[np.float64], float]:
        """Single-track jump histogram ``(sizes, unit_intensities, I_track)``.

        The jump a track deposits in a subvolume whose centre passes at distance
        ``rho`` is the subvolume-*averaged* dose ``g(rho)``
        (`_localdose.box_averaged_profile`), and the mean number of tracks
        passing at ``[rho, rho+drho]`` is ``phi 2 pi rho drho`` with
        ``phi = D / I_track``. The returned intensities carry the geometric
        ``2 pi rho drho`` factor only, so a whole dose-response curve is obtained
        by scaling them -- the dose enters nowhere else.

        ``I_track = integral g 2 pi rho drho`` is taken from the *averaged* profile, so
        the mean local dose is exactly ``D``. ``eta_enhanced`` averages ``eta(d) d``
        instead of ``d`` (as `_damage_pattern` does; ``eta`` acts on the pointwise
        core dose, so averaging plain dose first destroys it); ``I_track`` stays the
        *plain* dose-area, since ``phi = D / I_track`` follows the physical dose.
        """
        r, dose, _r_min, r_max = self._local_dose_profile(ion)
        ell = SUBVOLUME_SIDE_NM * 1.0e-3
        rho = np.logspace(
            math.log10(float(r[0])), math.log10(r_max + ell), _LOCALDOSE_RHO_NODES, dtype=np.float64
        )
        g = _localdose.box_averaged_profile(r, dose, rho, ell=ell, engine=self.engine)
        area = _localdose.trapezoid_weights(rho) * 2.0 * math.pi * rho
        i_track = float(np.sum(area * g))
        profile = (
            _localdose.box_averaged_profile(r, _eta(dose) * dose, rho, ell=ell, engine=self.engine)
            if eta_enhanced
            else g
        )
        sizes, weights = _localdose.dose_area_histogram(
            profile, area, bins_per_decade=_LOCALDOSE_BINS_PER_DECADE
        )
        return sizes, weights, i_track

    def _local_dose_solver(
        self, ion: Ion, d_max: float, *, eta_enhanced: bool
    ) -> tuple[_localdose.CompoundPoissonSolver, float]:
        """``(solver, I_track)`` for the local-dose density up to dose ``d_max``."""
        sizes, weights, i_track = self._local_dose_jumps(ion, eta_enhanced=eta_enhanced)
        # Bound the support: the largest single jump, plus the mean, plus a wide
        # margin of the fluctuation. ``wrapped_mass`` verifies the bound holds.
        spread = math.sqrt(d_max / i_track * float(np.sum(weights * sizes * sizes)))
        d_hi = 1.5 * float(sizes.max()) + d_max + 8.0 * spread
        solver = _localdose.CompoundPoissonSolver(
            sizes, weights, d_hi=d_hi, n_grid=_LOCALDOSE_N_GRID
        )
        return solver, i_track

    def _local_dose_response(
        self, kernel: LEM4LocalDoseKernel, dose_grid: npt.NDArray[np.float64]
    ) -> tuple[npt.NDArray[np.float64], bool]:
        """``(rho(d), per_loop)`` for the selected local-dose kernel.

        ``per_loop`` says how subvolume responses aggregate into a cell effect:
        GLOBLE's is a per-loop lethality (sum over all loops), while
        ``-ln S_x(d)`` is already the whole-cell effect at uniform dose ``d``
        (average over subvolumes, so a uniform field reproduces the photon curve
        exactly).
        """
        if kernel == "globle":
            m = DSB_PER_GY_PER_LOOP * _eta(dose_grid) * dose_grid
            eps_i, eps_c = self._globle_lethalities(n_loops=MC_GRID_N_LOOPS)
            return m * np.exp(-m) * eps_i + _p_cdsb(m) * eps_c, True
        a, b = self.photon.alpha_x, self.photon.beta_x
        d_t = self.photon.D_t
        assert d_t is not None  # enforced in LEMBase.__post_init__
        s_max = a + 2.0 * b * d_t
        shouldered = a * dose_grid + b * dose_grid * dose_grid
        linear_tail = s_max * (_eta(dose_grid) * dose_grid - d_t) + (a * d_t + b * d_t * d_t)
        return np.where(dose_grid < d_t, shouldered, linear_tail), False

    def local_dose_lethal_events(
        self, ion: Ion, doses_Gy: Iterable[float]
    ) -> npt.NDArray[np.float64]:
        r"""Lethal events ``N_ion(D)`` from the local-dose convolution.

        The local dose at a random nucleus point is a compound-Poisson sum over
        the track field (see `_localdose`), so with ``f_D`` its density and
        ``rho`` the per-subvolume response of [`local_dose_kernel`][],

            N_ion(D) = N_sv <rho(d)>_{f_D}        (per-loop response) ,
            N_ion(D) =        <rho(d)>_{f_D}      (whole-cell response) .

        The infinite track plane includes the far penumbra analytically (no simulated
        domain radius); as tracks become dense ``f_D -> delta(d - D)`` and the photon
        response is recovered exactly. With [`local_dose_overkill`][] the single-track
        Poisson overkill is applied (`_local_dose_overkill_events`); otherwise this
        is the mean effect (``exp(-<E>)`` convention).
        """
        doses = np.asarray(list(doses_Gy), dtype=np.float64)
        if self.local_dose_overkill:
            return self._local_dose_overkill_events(ion, doses)
        return self._local_dose_mean_events(ion, doses)

    def _local_dose_mean_events(
        self, ion: Ion, doses: npt.NDArray[np.float64]
    ) -> npt.NDArray[np.float64]:
        """Mean lethal events ``<E>`` (the ``exp(-<E>)`` convention); see
        [`local_dose_lethal_events`][]."""
        d_max = float(doses.max())
        if self.local_dose_kernel == "discrete_cluster_index":
            raise NotImplementedError(
                "local_dose_kernel='discrete_cluster_index' is Monte-Carlo only. The "
                "deterministic convolution cannot express an average-of-ratios cluster "
                "index: convolving a mean dose field IS the mean-field construction it "
                "exists to avoid. Use initial_slope_model='full_simulation' / "
                "local_dose_mc_alpha_beta, or local_dose_kernel='cluster_index' here."
            )
        if self.local_dose_kernel == "cluster_index":
            return self._local_dose_cluster_index_events(ion, doses, d_max)
        solver, i_track = self._local_dose_solver(ion, d_max, eta_enhanced=False)
        response, per_loop = self._local_dose_response(self.local_dose_kernel, solver.dose_grid)
        # The nucleus holds ``V / l_DSB^3`` subvolumes, the same count the 540 nm
        # tiling gives (`MC_GRID_N_LOOPS`).
        scale = MC_GRID_N_LOOPS if per_loop else 1.0
        return np.array(
            [scale * solver.response_mean(float(d) / i_track, response) for d in doses],
            dtype=np.float64,
        )

    def _local_dose_overkill_events(
        self, ion: Ion, doses: npt.NDArray[np.float64]
    ) -> npt.NDArray[np.float64]:
        r"""Single-track Poisson-overkill lethal events (the "B2" correction).

        The plain convolution scores the mean effect ``<E>`` (``exp(-<E>)``), whereas the
        survival is ``<exp(-E)>``. Keeping the Poisson track-*number* fluctuation
        analytically -- the effect at ``n`` traversals taken as the mean effect at dose
        ``n D_1`` -- gives

            -ln S(D) = -ln sum_n Pois(n; D/D_1) exp(-E_mean(n D_1)) ,

        with ``E_mean`` the mean-effect curve and ``D_1`` the mean single-traversal dose.
        Exact in both limits (``D -> 0`` and dense tracks), but it misses the per-track
        variance: ``beta_I`` can dip negative past ~150 keV/um, and against the exact
        ``<exp(-E)>`` it over-corrects (LEM_IV_OPEN_CHOICES.md entry 2.6).
        """
        d_1 = self.mean_traversal_dose_Gy(ion)
        lam_max = float(doses.max()) / d_1
        n_max = math.ceil(lam_max + 8.0 * math.sqrt(lam_max)) + 12
        # Mean effect at n D_1 (n = 1..n_max); E(0) = 0.
        grid = np.arange(1, n_max + 1, dtype=np.float64) * d_1
        e_grid = np.concatenate([[0.0], self._local_dose_mean_events(ion, grid)])
        ns = np.arange(n_max + 1)
        log_fact = np.array([math.lgamma(k + 1) for k in ns], dtype=np.float64)
        surv_kernel = np.exp(-np.minimum(e_grid, _MAX_EXPONENT))
        out = np.empty(doses.size, dtype=np.float64)
        for i, dose in enumerate(doses):
            lam = float(dose) / d_1
            weights = np.exp(-lam + ns * math.log(lam) - log_fact)
            surv = float(np.sum(weights * surv_kernel))
            out[i] = -math.log(max(surv, _MIN_SURVIVAL))
        return out

    def _local_dose_cluster_index_events(
        self, ion: Ion, doses: npt.NDArray[np.float64], d_max: float
    ) -> npt.NDArray[np.float64]:
        r"""LEM IV's damage-pattern mapping on the local-dose density.

        The nucleus-wide loop counts follow from the density directly,

            N_dmg(D)  = N_sv <P_dmg(m(d))> ,   N_cDSB(D) = N_sv <P_cDSB(m(d))> ,

        with ``m = a_DSB/loop * d`` on the *eta-enhanced* jump field, then (Pfuhl
        eqs. 3-6, via the ``C_g -> g`` lookup `_intertrack_pe_lookup`)

            C_g(D_eq) = C_I ,   E(D) = N_dmg * (-ln S_g(D_eq)) / N_dmg,g(D_eq) .

        ``C_I`` grows with dose, lifting ``beta_I`` above ``beta_x`` at intermediate LET;
        the ``D_t`` saturation of ``S_g(D_eq)`` then collapses it (both Fig. 4 features).
        """
        solver, i_track = self._local_dose_solver(ion, d_max, eta_enhanced=True)
        m = DSB_PER_GY_PER_LOOP * solver.dose_grid
        p_dmg, p_cdsb = _p_damaged(m), _p_cdsb(m)
        n_dmg = np.array(
            [MC_GRID_N_LOOPS * solver.response_mean(float(d) / i_track, p_dmg) for d in doses]
        )
        n_cdsb = np.array(
            [MC_GRID_N_LOOPS * solver.response_mean(float(d) / i_track, p_cdsb) for d in doses]
        )
        c_g, g_per_loop = self._intertrack_pe_lookup()
        cluster = np.clip(n_cdsb / n_dmg, *_cluster_index_bounds(c_g))
        return np.asarray(n_dmg * np.interp(cluster, c_g, g_per_loop), dtype=np.float64)

    def local_dose_alpha_beta(
        self, ion: Ion, *, doses_Gy: Iterable[float] | None = None
    ) -> tuple[float, float]:
        """Full-simulation ``(alpha_I, beta_I)`` from one LQ fit of the local-dose response.

        One low-dose LQ fit ``-ln S / D = alpha + beta D``
        ([`pyradbio.core.lq.fit_lq_linearised`][]) of [`local_dose_lethal_events`][]
        over ``doses_Gy`` (default [`fit_doses_Gy`][]): the full simulation of Scholz et
        al. 1997 (doi:10.1007/s004110050055), as in Friedrich et al. 2012
        (doi:10.3109/09553002.2011.611213) Fig. 4.

        **Neither coefficient is clamped** (unlike [`local_dose_beta`][]); ``alpha`` is
        the window-projected linear coefficient, not the ``D -> 0`` [`initial_slope`][].
        Raises ``ValueError`` when the response is non-finite (zero survival in the
        window) or ``doses_Gy`` is degenerate (fewer than 2 doses, a non-positive dose,
        or a single repeated dose).
        """
        doses = (
            self.fit_doses_Gy(ion=ion)
            if doses_Gy is None
            else np.asarray(list(doses_Gy), np.float64)
        )
        return _localdose.lq_coefficients(doses, self.local_dose_lethal_events(ion, doses))

    def local_dose_beta(self, ion: Ion, *, doses_Gy: Iterable[float] | None = None) -> float:
        """Full-simulation ion ``beta_I`` from the local-dose convolution.

        Reproduces the Friedrich et al. 2012 (doi:10.3109/09553002.2011.611213) Fig. 4
        full-simulation ``beta_I(LET)``: ``beta_x`` in the photon limit, a slight rise at
        intermediate LET, then a collapse toward 0. ``beta_I`` is the LQ coefficient over
        [`fit_doses_Gy`][], a window chosen on this very panel (agreement partly by
        construction; quote it). Deterministic; [`local_dose_mc_beta`][] is the
        explicit-track cross-check.

        Within ~4% of the paper up to LET 50, increasingly high above LET ~65 (open). The
        enhancement cannot roll off here: by dose conservation the cluster index's dose
        slope is the photon rate ``kappa/2`` at every LET (also [`discrete_dsb_beta`][]).
        ``"local_dose"`` is the faithful full simulation for Fig. 4b
        (LEM_IV_OPEN_CHOICES.md entry 2.5; docs/references/LEM_IV_Friedrich2012.md).
        Kernels: [`LEM4LocalDoseKernel`][]; overkill: [`local_dose_overkill`][].

        **Deep overkill returns a clamped 0, with a warning.** Past roughly 150 keV/um
        the response has no curvature left and the least-squares ``beta`` goes
        marginally negative (a fit artifact that [`LQParameters`][]
        would reject); the floor is not a computed value, hence the warning.
        """
        doses = (
            self.fit_doses_Gy(ion=ion)
            if doses_Gy is None
            else np.asarray(list(doses_Gy), np.float64)
        )
        beta = self.local_dose_alpha_beta(ion, doses_Gy=doses)[1]
        if beta < 0.0:
            warnings.warn(
                f"local-dose beta fit returned {beta:.3e} < 0 for Z={ion.Z} at "
                f"{ion.energy_MeV_u:g} MeV/u and was clamped to 0. The simulated "
                "dose response is saturated here, so the low-dose LQ fit has no "
                "curvature left to measure; treat this beta as a floor, not a value.",
                UserWarning,
                stacklevel=2,
            )
        return max(beta, 0.0)

    def local_dose_mc_beta(
        self,
        ion: Ion,
        *,
        doses_Gy: Iterable[float] | None = None,
        n_realizations: int = _LOCALDOSE_MC_REALIZATIONS,
        seed: int = 0,
        overkill: bool = False,
    ) -> float:
        """Explicit-track Monte-Carlo cross-check of [`local_dose_beta`][].

        Places tracks one by one on the extended domain ``R_eff + r_max`` and
        accumulates each subvolume's local dose. Slow and noisy; it validates the
        convolution rather than replacing it.

        ``overkill=False`` (default) fits the mean effect ``<E>`` (what the convolution
        computes); ``overkill=True`` fits the exact ``-ln <exp(-E)>`` that the analytic B2
        correction approximates (judge it on ``"cluster_index"`` only; other kernels
        understate it). Cost grows as ``(R_sim/R_eff)^2``, so without
        [`local_dose_mc_impact_cut`][] it is practical from about LET 40 up. A single
        ``beta`` is noise-dominated; compare conventions with
        [`local_dose_mc_beta_pair`][].
        """
        effect, survival, doses = self._local_dose_mc_curves(
            ion, doses_Gy, n_realizations=n_realizations, seed=seed
        )
        scored = -np.log(np.clip(survival, _MIN_SURVIVAL, 1.0)) if overkill else effect
        return _localdose.lq_coefficients(doses, scored)[1]

    def local_dose_mc_alpha_beta(
        self,
        ion: Ion,
        *,
        doses_Gy: Iterable[float] | None = None,
        n_realizations: int = _LOCALDOSE_MC_REALIZATIONS,
        seed: int = 0,
        overkill: bool = True,
    ) -> tuple[float, float]:
        r"""``(alpha_I, beta_I)`` read off **one** simulated survival curve.

        The Scholz 1997 program Elsasser 2010 invokes for the macroscopic coefficients
        (build ``S(D)``, fit both coefficients) run on LEM IV's damage-pattern effect.
        ``overkill=True`` (default here) scores ``-ln <exp(-E)>`` over the sampled track
        configurations, with no B2-style truncation.

        ``alpha`` is the *window-projected* linear coefficient, not the ``D -> 0`` slope,
        which is analytically [`initial_slope`][]'s direct term ``(1 - e^{-N_1})/D_1``.
        A single MC ``beta`` is dominated by realisation scatter: average over seeds, and
        use [`local_dose_mc_beta_pair`][] for differences between conventions.
        """
        effect, survival, doses = self._local_dose_mc_curves(
            ion, doses_Gy, n_realizations=n_realizations, seed=seed
        )
        scored = -np.log(np.clip(survival, _MIN_SURVIVAL, 1.0)) if overkill else effect
        alpha, beta = _localdose.lq_coefficients(doses, scored)
        return alpha, beta

    def local_dose_mc_beta_pair(
        self,
        ion: Ion,
        *,
        doses_Gy: Iterable[float] | None = None,
        n_realizations: int = _LOCALDOSE_MC_REALIZATIONS,
        seed: int = 0,
    ) -> tuple[float, float]:
        """``(beta_mean_effect, beta_overkill)`` from **one** set of realisations.

        Both conventions are scored on the *same* sampled track configurations, so their
        difference is a common-random-numbers estimator far better conditioned than
        either term (biases of the truncated domain cancel). It is the estimator that
        shows the analytic B2 correction ([`local_dose_overkill`][]) over-corrects
        (LEM_IV_OPEN_CHOICES.md entry 2.6).
        """
        effect, survival, doses = self._local_dose_mc_curves(
            ion, doses_Gy, n_realizations=n_realizations, seed=seed
        )
        beta_mean = _localdose.lq_coefficients(doses, effect)[1]
        beta_overkill = _localdose.lq_coefficients(
            doses, -np.log(np.clip(survival, _MIN_SURVIVAL, 1.0))
        )[1]
        return beta_mean, beta_overkill

    def _local_dose_mc_curves(
        self,
        ion: Ion,
        doses_Gy: Iterable[float] | None,
        *,
        n_realizations: int,
        seed: int,
    ) -> tuple[npt.NDArray[np.float64], npt.NDArray[np.float64], npt.NDArray[np.float64]]:
        """``(mean_effect, mean_survival, doses)`` of one explicit-track MC run."""
        doses = (
            self.fit_doses_Gy(ion=ion)
            if doses_Gy is None
            else np.asarray(list(doses_Gy), np.float64)
        )
        kernel = self.local_dose_kernel
        r, dose, _r_min, r_max = self._local_dose_profile(ion)
        ell = SUBVOLUME_SIDE_NM * 1.0e-3
        rho = np.logspace(
            math.log10(float(r[0])), math.log10(r_max + ell), _LOCALDOSE_RHO_NODES, dtype=np.float64
        )
        # Both cluster-index kernels read the subvolume's *eta-enhanced* dose (what
        # `_damage_pattern` integrates); the per-subvolume kernels read the physical one.
        # The Poisson intensity below always uses the plain dose (as _local_dose_jumps).
        cluster_kernels = ("cluster_index", "discrete_cluster_index")
        field = _eta(dose) * dose if kernel in cluster_kernels else dose
        g = _localdose.box_averaged_profile(r, field, rho, ell=ell, engine=self.engine)
        log_rho, log_g = np.log(rho), np.log(np.maximum(g, 1.0e-300))
        top = float(rho[-1])

        def jump_at(dist: npt.NDArray[np.float64]) -> npt.NDArray[np.float64]:
            out = np.exp(np.interp(np.log(np.clip(dist, rho[0], top)), log_rho, log_g))
            out = np.where(dist < rho[0], g[0], out)
            return np.asarray(np.where(dist > top, 0.0, out), dtype=np.float64)

        r_eff = self.cell.effective_nucleus_radius_um()
        n_sv = _localdose.subvolume_centres(r_eff, ell)[0].size
        score = self._local_dose_pattern_score(kernel, n_sv=n_sv, r_eff=r_eff, ell=ell)

        d_1 = self.mean_traversal_dose_Gy(ion)
        r_sim = r_eff + r_max
        effect, survival = _localdose.monte_carlo_effect(
            jump_at,
            score,
            doses=doses,
            tracks_per_Gy=(r_sim / r_eff) ** 2 / d_1,
            r_eff=r_eff,
            r_sim=r_sim,
            ell=ell,
            n_real=n_realizations,
            rng=np.random.default_rng(seed),
            b_cut=self.resolve_mc_impact_cut(ion),
        )
        return effect, survival, doses

    def resolve_mc_impact_cut(self, ion: Ion) -> float | None:
        """Impact-parameter cut for the explicit-track MC, or ``None`` to sample all.

        ``local_dose_mc_impact_cut`` is either an explicit radius [um], ``None`` (no
        split, the exact engine), or ``"auto"``, which returns
        ``_MC_IMPACT_CUT_FACTOR * r_eff`` once that is smaller than the full domain
        ``r_eff + r_max`` and ``None`` otherwise -- so the split engages only where it
        actually saves work, and low-energy ions keep the exact engine.
        """
        setting = self.local_dose_mc_impact_cut
        if setting is None:
            return None
        r_eff = self.cell.effective_nucleus_radius_um()
        r_sim = r_eff + self._local_dose_profile(ion)[3]
        if setting != "auto":
            cut = float(setting)
            if not 0.0 < cut < r_sim:
                raise ValueError(
                    f"local_dose_mc_impact_cut must lie in (0, r_eff + r_max) = "
                    f"(0, {r_sim:.3g}) um for this ion, got {cut}"
                )
            return cut
        cut = max(
            _MC_IMPACT_CUT_FACTOR * r_eff,
            _MC_IMPACT_CUT_MIN_DOMAIN_FRACTION * r_sim,
        )
        return cut if cut < r_sim else None

    def _local_dose_pattern_score(
        self, kernel: LEM4LocalDoseKernel, *, n_sv: int, r_eff: float, ell: float
    ) -> Callable[[npt.NDArray[np.float64]], float]:
        """Map a realisation's subvolume local doses to one cell effect.

        Each kernel's aggregation (see `_localdose.monte_carlo_effect`):
        ``"globle"`` sums a per-loop response over all loops, ``"photon_effect"``
        averages a whole-cell response, and the cluster-index kernels form ``C_I`` over
        the *whole nucleus* first, then map it to an effect. The cluster-index branches
        pin the loop count to [`MC_GRID_N_LOOPS`][] (the count the ``C_g -> g`` lookup
        divides by), so they reproduce `_local_dose_cluster_index_events` term by
        term; the per-subvolume branches keep the ``V / (A ell)`` layer count.
        """
        if kernel == "cluster_index":
            # Keyed on the UNCLUSTERED fraction, not on C_I, whose saturated lookup top
            # is a tie (see `_pe_lookup_by_unclustered`).
            u_grid, g_grid = self._pe_lookup_by_unclustered()
            # The shared overkill regularisation (_CLUSTER_INDEX_MAX), expressed in u.
            u_lo = max(float(u_grid[0]), 1.0 - _CLUSTER_INDEX_MAX)
            u_hi = float(u_grid[-1])
            per_column = MC_GRID_N_LOOPS / float(n_sv)

            def score_cluster(local: npt.NDArray[np.float64]) -> float:
                m = DSB_PER_GY_PER_LOOP * local
                n_dmg = per_column * float(_p_damaged(m).sum())
                if n_dmg <= 0.0:  # no dose yet in this realisation
                    return 0.0
                # Isolated DSB accumulated DIRECTLY: n_dmg - n_cdsb would reintroduce
                # the cancellation the u-keyed lookup avoids.
                with np.errstate(under="ignore"):
                    n_iso = per_column * float((m * np.exp(-m)).sum())
                u = min(max(n_iso / n_dmg, u_lo), u_hi)
                return n_dmg * float(np.interp(u, u_grid, g_grid))

            return score_cluster

        if kernel == "discrete_cluster_index":
            # As above, but C_I is the average-of-ratios E[N_cDSB/N_dmg] over the
            # realisation's independent Poisson loops; N_dmg is unchanged (kappa needs
            # the damaged area, as in `_single_traversal_effect`).
            u_grid, g_grid = self._pe_lookup_by_unclustered()
            u_lo = max(float(u_grid[0]), 1.0 - _CLUSTER_INDEX_MAX)
            u_hi = float(u_grid[-1])
            per_column = MC_GRID_N_LOOPS / float(n_sv)
            # Every loop in a column sees the same local dose, so a column enters the
            # loop list n_z times (as in analytic_discrete_cluster_index); do NOT divide.
            n_z = max(1, round(per_column))

            def score_discrete(local: npt.NDArray[np.float64]) -> float:
                m = DSB_PER_GY_PER_LOOP * local
                n_dmg = per_column * float(_p_damaged(m).sum())
                if n_dmg <= 0.0:  # no dose yet in this realisation
                    return 0.0
                c_i = _discrete.expected_damaged_ratio(np.repeat(m, n_z))
                u = min(max(1.0 - c_i, u_lo), u_hi)
                return n_dmg * float(np.interp(u, u_grid, g_grid))

            return score_discrete

        layers = (NUCLEUS_VOLUME_UM3 / (math.pi * r_eff * r_eff)) / ell
        per_loop = kernel == "globle"

        def score_subvolume(local: npt.NDArray[np.float64]) -> float:
            scored = self._local_dose_response(kernel, local)[0]
            return layers * float(scored.sum()) if per_loop else float(scored.mean())

        return score_subvolume

    # ---- discrete-DSB damage pattern (Fig. 4 full simulation, discrete reading) ----
    def _discrete_setup(self, ion: Ion) -> DiscreteSetup:
        """Per-ion inputs of the discrete-DSB engine (one radial pass).

        ``mu(rho) = a_DSB/loop * g_eta(rho)`` is the mean DSB one track puts into a
        loop at transverse distance ``rho``, with ``g_eta`` the **box-averaged**
        eta-enhanced dose -- the same quantity `_damage_pattern` integrates over
        a subvolume, and never the pointwise ``1/r^2`` dose.
        """
        r, dose, _r_min, r_max = self._local_dose_profile(ion)
        ell = SUBVOLUME_SIDE_NM * 1.0e-3
        r_eff = self.cell.effective_nucleus_radius_um()
        rho = np.logspace(
            math.log10(float(r[0])), math.log10(r_max + ell), _LOCALDOSE_RHO_NODES, dtype=np.float64
        )
        g_eta = _localdose.box_averaged_profile(
            r, _eta(dose) * dose, rho, ell=ell, engine=self.engine
        )
        a0, a1 = _discrete.track_factors(rho, DSB_PER_GY_PER_LOOP * g_eta, r_eff=r_eff, ell=ell)
        # The centre-inside lattice overshoots ``pi R_eff^2`` by ~3%; pin the total loop
        # count to the geometric ``V / l_DSB^3`` (`MC_GRID_N_LOOPS`), the count the
        # ``C_g -> g`` lookup divides by, so the photon normalisation is consistent.
        n_z = MC_GRID_N_LOOPS / float(a0.size)
        return DiscreteSetup(
            a0=a0,
            a1=a1,
            n_z=n_z,
            n_loops=MC_GRID_N_LOOPS,
            retained_dose_fraction=_discrete.retained_dose_fraction(r, dose, r_eff=r_eff),
            mean_tracks_per_Gy=1.0 / self.mean_traversal_dose_Gy(ion),
            track_disk_radius_um=r_eff,
        )

    def _discrete_counts(
        self, setup: DiscreteSetup, dose_Gy: float, retained: float
    ) -> tuple[npt.NDArray[np.float64], npt.NDArray[np.float64], npt.NDArray[np.float64]]:
        """``(weights, N_dmg(n), N_cDSB(n))`` over the track-number grid at one dose."""
        lam = dose_Gy * setup.mean_tracks_per_Gy
        m_bg = DSB_PER_GY_CELL * dose_Gy * (1.0 - retained) / setup.n_loops
        ns, weights = _discrete.poisson_track_weights(lam)
        counts = [
            _discrete.conditional_loop_counts(
                setup.a0, setup.a1, n_tracks=int(n), m_bg=m_bg, n_z=setup.n_z
            )
            for n in ns
        ]
        n_dmg = np.array([c[0] for c in counts], dtype=np.float64)
        n_cdsb = np.array([c[1] for c in counts], dtype=np.float64)
        return weights, n_dmg, n_cdsb

    def _discrete_events(
        self,
        setup: DiscreteSetup,
        doses: npt.NDArray[np.float64],
        *,
        convention: LEM4DiscreteConvention,
        survival: LEM4DiscreteSurvival = "mean_effect",
        retained: float | None = None,
    ) -> npt.NDArray[np.float64]:
        """Lethal events ``N_ion(D)`` of the discrete pattern under one convention.

        ``survival="stochastic"`` returns ``-ln <exp(-E)>`` instead of ``<E>``; it is
        exact here because ``E_n`` is the true effect at ``n`` traversals rather than a
        mean curve evaluated at ``n D_1`` (see [`LEM4DiscreteSurvival`][]). Only
        defined for ``convention="per_realisation"``.

        ``retained`` overrides the dose-conserving split (diagnostic only -- any other
        value breaks dose conservation; ``retained=1.0`` reproduces the documented
        no-background pathology).
        """
        if survival == "stochastic" and convention == "mean_counts":
            raise ValueError(
                "survival='stochastic' is undefined for convention='mean_counts' "
                "(no per-traversal effect to average exp(-E) over)"
            )
        frac = setup.retained_dose_fraction if retained is None else retained
        c_g, g_per_loop = self._intertrack_pe_lookup()
        ci_lo, ci_hi = _cluster_index_bounds(c_g)
        out = np.empty(doses.size, dtype=np.float64)
        for i, dose in enumerate(doses):
            weights, n_dmg, n_cdsb = self._discrete_counts(setup, float(dose), frac)
            if convention == "mean_counts":
                mean_dmg = float(np.sum(weights * n_dmg))
                mean_cdsb = float(np.sum(weights * n_cdsb))
                cluster = min(max(mean_cdsb / mean_dmg, ci_lo), ci_hi)
                out[i] = mean_dmg * float(np.interp(cluster, c_g, g_per_loop))
                continue
            per_n = np.clip(
                np.divide(n_cdsb, n_dmg, out=np.zeros_like(n_dmg), where=n_dmg > 0.0),
                ci_lo,
                ci_hi,
            )
            effect_n = n_dmg * np.interp(per_n, c_g, g_per_loop)
            if survival == "stochastic":
                # exp(-E_n) underflows to 0 well before the exponent overflows; clip so
                # deep overkill yields S -> 0 rather than a NaN.
                surv = float(np.sum(weights * np.exp(-np.minimum(effect_n, _MAX_EXPONENT))))
                out[i] = -math.log(max(surv, _MIN_SURVIVAL))
            else:
                out[i] = float(np.sum(weights * effect_n))
        return out

    def _discrete_cluster_index(
        self, setup: DiscreteSetup, doses: npt.NDArray[np.float64]
    ) -> npt.NDArray[np.float64]:
        """Realisation-averaged cluster index ``C_I(D)`` of the discrete pattern.

        Its *dose slope* is the intertrack enhancement, and measuring that slope is
        how the no-go is checked (see
        `tests...test_discrete_cluster_index_slope_stays_at_the_photon_rate`).
        """
        out = np.empty(doses.size, dtype=np.float64)
        for i, dose in enumerate(doses):
            weights, n_dmg, n_cdsb = self._discrete_counts(
                setup, float(dose), setup.retained_dose_fraction
            )
            out[i] = float(np.sum(weights * n_cdsb)) / float(np.sum(weights * n_dmg))
        return out

    def discrete_dsb_lethal_events(
        self,
        ion: Ion,
        doses_Gy: Iterable[float],
        *,
        convention: LEM4DiscreteConvention | None = None,
        survival: LEM4DiscreteSurvival | None = None,
    ) -> npt.NDArray[np.float64]:
        r"""Lethal events ``N_ion(D)`` from the explicit discrete DSB damage pattern.

        ``Poisson(D/D_1)`` tracks cross the nucleus (centres uniform on the ``R_eff``
        disk, **no domain extension**) placing discrete DSBs from their radial profile;
        the escaped dose ``D (1 - f_ret)`` arrives as a uniform Poisson DSB background.
        Scored with the cluster-index functional of
        `_local_dose_cluster_index_events`. Deterministic: conditioning on the track
        number ``n`` leaves closed-form loop statistics (`_discrete`).

        ``convention`` overrides [`discrete_convention`][]. **``"mean_counts"`` is
        provably identical to ``local_dose_kernel="cluster_index"``**: averaging over
        ``n ~ Poisson(D/D_1)`` gives ``sum_n Pois(n) a0^n = exp(-(D/D_1)(1 - a0))``, the
        compound-Poisson exponent of the local-dose density. The default
        ``"per_realisation"`` is *not* (``beta`` differs by -4% to +55% over LET 10-100).

        ``survival`` overrides [`discrete_survival`][]: ``"stochastic"`` returns
        Scholz 1997's ``-ln <exp(-E)>`` rather than ``<E>``.
        """
        doses = np.asarray(list(doses_Gy), dtype=np.float64)
        return self._discrete_events(
            self._discrete_setup(ion),
            doses,
            convention=self.discrete_convention if convention is None else convention,
            survival=self.discrete_survival if survival is None else survival,
        )

    def discrete_dsb_alpha_beta(
        self,
        ion: Ion,
        *,
        doses_Gy: Iterable[float] | None = None,
        convention: LEM4DiscreteConvention | None = None,
        survival: LEM4DiscreteSurvival | None = None,
    ) -> tuple[float, float]:
        """Full-simulation ``(alpha_I, beta_I)`` from one LQ fit of the discrete-DSB response.

        As [`local_dose_alpha_beta`][], on [`discrete_dsb_lethal_events`][] (same
        default window, no clamping, same ``ValueError`` cases). ``convention`` /
        ``survival`` default to [`discrete_convention`][] / [`discrete_survival`][];
        ``"per_realisation"`` with ``"stochastic"`` is the exact
        ``<exp(-E)> = sum_n P(n) exp(-E_n)`` reference for the analytic B2 overkill
        ([`local_dose_overkill`][]). Deterministic (no seed).
        """
        doses = (
            self.fit_doses_Gy(_DISCRETE_DOSE_FRACTIONS, ion=ion)
            if doses_Gy is None
            else np.asarray(list(doses_Gy), np.float64)
        )
        events = self.discrete_dsb_lethal_events(
            ion, doses, convention=convention, survival=survival
        )
        return _localdose.lq_coefficients(doses, events)

    def discrete_dsb_beta(self, ion: Ion, *, doses_Gy: Iterable[float] | None = None) -> float:
        """Full-simulation ion ``beta_I`` from the discrete-DSB damage pattern.

        Deterministic, ~2 s per ion, no free parameters. Reproduces both Friedrich et
        al. 2012 (doi:10.3109/09553002.2011.611213) Fig. 4 features (the rise of
        ``beta_I`` above ``beta_x`` at intermediate LET, the collapse toward 0) and the
        photon limit at LET 10 to within ~5%.

        It **cannot** add the high-LET roll-off that [`local_dose_beta`][] lacks:
        ``C_I`` sees loop occupancy counts only, never track provenance, and those are the
        continuum's ([`discrete_dsb_lethal_events`][]). A **cross-check of**
        ``beta_model="local_dose"``, **not a replacement**; it fits Fig. 4 marginally worse
        under either ``discrete_survival`` (LEM_IV_OPEN_CHOICES.md, "Why RMS cannot pick
        the engine"; docs/references/LEM_IV_Friedrich2012.md).
        """
        return self.discrete_dsb_alpha_beta(ion, doses_Gy=doses_Gy)[1]

    def intertrack_mc_photon_survival(
        self,
        doses_Gy: Iterable[float],
        *,
        n_realizations: int = _INTERTRACK_REALIZATIONS,
        seed: int = 0,
    ) -> npt.NDArray[np.float64]:
        """Photon-limit (uniform-field) discrete survival for the GLOBLE effect.

        ``Poisson(30 D)`` DSBs placed uniformly over exactly [`globle_n_loops`][] giant
        loops and scored with the same ``(eps_i, eps_c)``. Its fitted ``(alpha, beta)``
        must recover the photon ``(alpha_x, beta_x)`` and its ``N_lethal`` the analytic
        GLOBLE eqs. (2)-(3); hence it uses [`globle_n_loops`][] (the loop count those
        equations assume), not the spatial 540 nm grid of the ion path.
        """
        eps_i, eps_c = self._globle_lethalities()
        n_loops = round(self.globle_n_loops)
        rng = np.random.default_rng(seed)
        return _intertrack_kernel.uniform_survival_numpy(
            np.asarray(list(doses_Gy), dtype=np.float64),
            eps_i=eps_i,
            eps_c=eps_c,
            a_dsb_per_gy=DSB_PER_GY_CELL,
            n_loops=n_loops,
            n_real=n_realizations,
            rng=rng,
        )
