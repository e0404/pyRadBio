"""Discrete-DSB damage-pattern machinery for the LEM IV full simulation.

Numerical core of the *discrete* reading of the Friedrich et al. 2012 (Int. J.
Radiat. Biol. 88:103-107, doi:10.3109/09553002.2011.611213, Fig. 4) full
simulation. Like :mod:`._localdose` it holds no biophysics --
:mod:`pyradbio.models.lem.lem4` supplies the radial dose profile and the effect
functional, this module supplies the statistics.

**The model.** The dose is split exactly as the traversal decomposition does, with
no free parameter and, crucially, **no domain extension**:

    N_trav ~ Poisson(D / D_1)   tracks crossing the nucleus, centred uniformly on
                                the disk of radius R_eff, each placing discrete
                                DSBs drawn from its own radial profile
    B      = D (1 - f_ret)      the escaped / near-miss dose, delivered as a
                                UNIFORM Poisson DSB background over the loops

``f_ret`` (:func:`retained_dose_fraction`) is the fraction of a uniformly-placed
track's dose that lands in the nucleus, so the split conserves dose exactly. The
alternative of resolving the near-miss penumbra spatially is what breaks the
explicit-track ``"intertrack_mc"`` engine: at LET 20 it needed a 3400x track multiplier whose
near-uniform penumbra clusters like a photon field, pinning ``beta ~ beta_x`` at
every LET. Switching the background off instead is the opposite failure -- it
destroys the low-LET photon limit (``beta`` 0.020 against 0.052 at LET 20).

**Why this is evaluated deterministically.** Sampling this pattern by Monte Carlo
works but is hopeless for reading off a small curvature: single-seed ``beta`` at
LET 55 scattered over 0.013-0.048 across seeds, enough to fake a dramatic
high-LET roll-off that six seeds then erased. Instead the construction conditions
on the track number ``n`` -- the only fluctuation that carries ``1/LET`` -- and
evaluates the loop statistics in closed form. For a loop at transverse position
``x``, one track placed uniformly on the disk contributes ``Poisson(mu(|x - b|))``
DSBs with ``b`` uniform, so

    a0(x) = <e^{-mu}>_b ,      a1(x) = <mu e^{-mu}>_b        (:func:`track_factors`)

are disk-convolutions, computed exactly by FFT. With ``n`` independent tracks plus
a uniform background of mean ``m_bg`` per loop,

    P(0 DSB | n) = a0^n e^{-m_bg}
    P(1 DSB | n) = [n a1 a0^{n-1} + m_bg a0^n] e^{-m_bg}

giving ``N_dmg(n)`` and ``N_cDSB(n)`` exactly (:func:`conditional_loop_counts`).
The result is deterministic, seed-free and fast.

**The theorem this module exists to demonstrate.** Averaging those conditional
counts over ``n ~ Poisson(D/D_1)`` reproduces the continuum ``cluster_index``
kernel *identically*, because ``sum_n Pois(n) a0^n = exp(-(D/D_1)(1 - a0))`` is
exactly the compound-Poisson exponent of the local-dose density. Equivalently: a
track's Poisson DSB total split over loops gives independent Poisson counts per
loop, so loop occupancy ``N`` is compound Poisson with PGF ``G(z) = E[e^{m(z-1)}]``
and ``P(N=0) = E[e^-m]``, ``P(N=1) = E[m e^-m]`` -- the very expectations the
continuum kernel evaluates.

So **discreteness is invisible to the cluster index**, and the intuition that a
discrete pattern distinguishes "two DSBs in a loop from the same track" from "two
from different tracks" is false: ``C_I`` is a function of loop *occupancy counts*
only and never sees track provenance. That is why this engine does not recover the
Fig. 4 enhancement roll-off; see ``docs/references/LEM_IV_Friedrich2012.md``.
"""

from __future__ import annotations

import math

import numpy as np
import numpy.typing as npt

from pyradbio.models.lem._localdose import subvolume_centres, trapezoid_weights

FloatArray = npt.NDArray[np.float64]
IntArray = npt.NDArray[np.int64]

# Transverse grid step [um] for the FFT disk-convolution of :func:`track_factors`.
# The convolution is smooth on the 540 nm loop scale, so ~12 nodes per loop side
# resolves it; halving the step moves ``a0`` by <0.2%.
_CONV_STEP_UM: float = 0.045

# Poisson tail cut for the track-number sum (:func:`poisson_track_weights`): the
# grid runs to ``lam + _POISSON_SIGMA sqrt(lam) + _POISSON_FLOOR``, then the
# weights are renormalised. 9 sigma leaves < 1e-9 of the mass outside.
_POISSON_SIGMA: float = 9.0
_POISSON_FLOOR: int = 12


def disk_overlap_fraction(r: FloatArray, r_eff: float) -> FloatArray:
    """Overlap area of two disks of radius ``r_eff`` at separation ``r``, over ``pi r_eff^2``.

    The classical lens area, ``O(r) = (2/pi)[arccos(u) - u sqrt(1 - u^2)]`` with
    ``u = r / (2 r_eff)``; ``O(0) = 1`` and ``O(r >= 2 r_eff) = 0``.
    """
    u = np.clip(np.asarray(r, dtype=np.float64) / (2.0 * r_eff), 0.0, 1.0)
    return np.asarray(
        (2.0 / math.pi) * (np.arccos(u) - u * np.sqrt(np.maximum(1.0 - u * u, 0.0))),
        dtype=np.float64,
    )


def retained_dose_fraction(r: FloatArray, dose: FloatArray, *, r_eff: float) -> float:
    r"""Fraction ``f_ret`` of a uniformly-placed track's dose that lands in the nucleus.

    Tracks are centred uniformly on the disk of radius ``r_eff`` with areal density
    ``phi``. The dose they deposit inside the nucleus, per unit nuclear area, is
    ``phi int D(r) 2 pi r O(r) dr`` with ``O`` the two-disk overlap fraction
    (:func:`disk_overlap_fraction`), because a shell of radius ``r`` around a track
    at ``b`` lies inside the nucleus for exactly that fraction of ``b``. Since
    ``D_1`` is defined from the *full* track energy, that means the retained mean
    dose is ``D f_ret`` with

        f_ret = int D(r) 2 pi r O(r) dr / int D(r) 2 pi r dr .

    The remainder ``D (1 - f_ret)`` is returned as the uniform background, so the
    split conserves dose exactly. In a uniform field the dose escaping a traversing
    track equals the dose near-miss tracks deposit, which is what makes the uniform
    background the *correct* treatment of the near misses rather than an
    approximation -- and it is not a free parameter.
    """
    r = np.asarray(r, dtype=np.float64)
    dose = np.asarray(dose, dtype=np.float64)
    weight = trapezoid_weights(r) * dose * 2.0 * math.pi * r
    total = float(np.sum(weight))
    if total <= 0.0:
        return 0.0
    return float(np.sum(weight * disk_overlap_fraction(r, r_eff)) / total)


def track_factors(
    rho: FloatArray,
    mu: FloatArray,
    *,
    r_eff: float,
    ell: float,
    step_um: float = _CONV_STEP_UM,
) -> tuple[FloatArray, FloatArray]:
    r"""Per-loop single-track factors ``(a0, a1)`` at every transverse loop centre.

    ``mu(rho)`` is the mean number of DSB one track deposits in a loop whose centre
    lies at transverse distance ``rho`` -- i.e. the *box-averaged* eta-enhanced dose
    times the DSB yield per loop per Gy. It must be the subvolume average, never the
    pointwise ``1/r^2`` dose (see :func:`._localdose.box_averaged_profile`).

    With the track centre ``b`` uniform on the nucleus disk,

        a0(x) = <e^{-mu(|x - b|)}>_b ,     a1(x) = <mu e^{-mu}>_b ,

    both disk-convolutions evaluated by FFT. ``a0`` is formed as
    ``1 + conv(e^{-mu} - 1)`` so the far penumbra (where ``mu -> 0``) contributes
    its small deficit rather than a difference of large numbers.

    The transform box spans ``4 r_eff``, which makes the circular convolution
    **alias-free at every sampled loop centre**: the disk indicator confines ``b``
    to ``|b| <= r_eff`` and the loop centres to ``|x| <= r_eff``, so every offset
    ``x - b`` actually summed satisfies ``|x - b| <= 2 r_eff`` and is represented
    exactly on the periodic grid.

    Returns ``(a0, a1)`` sampled at :func:`._localdose.subvolume_centres`.
    """
    rho = np.asarray(rho, dtype=np.float64)
    mu = np.asarray(mu, dtype=np.float64)
    a_eff = math.pi * r_eff * r_eff

    n = math.ceil(4.0 * r_eff / step_um)
    n += n % 2
    axis = (np.arange(n, dtype=np.float64) - 0.5 * n + 0.5) * step_um
    gx, gy = np.meshgrid(axis, axis, indexing="ij")
    dist = np.sqrt(gx * gx + gy * gy)
    mu_grid = np.interp(dist, rho, mu, left=float(mu[0]), right=0.0)
    disk = (dist <= r_eff).astype(np.float64)
    cell_area = step_um * step_um

    disk_ft = np.fft.rfft2(np.fft.ifftshift(disk))

    def _convolve(field: FloatArray) -> FloatArray:
        spectrum = np.fft.rfft2(np.fft.ifftshift(field)) * disk_ft
        out = np.fft.fftshift(np.fft.irfft2(spectrum, s=field.shape))
        return np.asarray(out * cell_area / a_eff, dtype=np.float64)

    a0_grid = 1.0 + _convolve(np.exp(-mu_grid) - 1.0)
    a1_grid = _convolve(mu_grid * np.exp(-mu_grid))

    lx, ly = subvolume_centres(r_eff, ell)
    ix = np.clip(np.round((lx - axis[0]) / step_um).astype(np.int64), 0, n - 1)
    iy = np.clip(np.round((ly - axis[0]) / step_um).astype(np.int64), 0, n - 1)
    a0 = np.clip(a0_grid[ix, iy], 1.0e-15, 1.0)
    a1 = np.clip(a1_grid[ix, iy], 0.0, 1.0)
    return np.asarray(a0, dtype=np.float64), np.asarray(a1, dtype=np.float64)


def conditional_loop_counts(
    a0: FloatArray,
    a1: FloatArray,
    *,
    n_tracks: int,
    m_bg: float,
    n_z: float,
) -> tuple[float, float]:
    """Damaged / clustered loop counts ``(N_dmg, N_cDSB)`` given exactly ``n_tracks``.

    The ``n`` tracks and the uniform background are independent, so their
    probability generating functions multiply:

        P(0 DSB) = a0^n e^{-m_bg}
        P(1 DSB) = [n a1 a0^{n-1} + m_bg a0^n] e^{-m_bg}

    and ``N_dmg = sum (1 - P0)``, ``N_cDSB = sum (1 - P0 - P1)`` over the transverse
    loop lattice, times the ``n_z`` layers along the beam (a track spans the whole
    nuclear depth, so every layer is statistically identical).
    """
    e_bg = math.exp(-m_bg)
    if n_tracks <= 0:
        p0 = np.full_like(a0, e_bg)
        p1 = m_bg * p0
    else:
        pow_n = a0**n_tracks
        p0 = pow_n * e_bg
        p1 = (n_tracks * a1 * a0 ** (n_tracks - 1) + m_bg * pow_n) * e_bg
    n_dmg = n_z * float(np.sum(1.0 - p0))
    n_cdsb = n_z * float(np.sum(np.maximum(1.0 - p0 - p1, 0.0)))
    return n_dmg, n_cdsb


def poisson_track_weights(lam: float) -> tuple[IntArray, FloatArray]:
    """Track-number grid ``n`` and Poisson weights ``P(n; lam)``, tail-truncated.

    Runs to ``lam + 9 sqrt(lam) + 12`` and renormalises, so the weights sum to 1
    exactly and the mean reproduces ``lam`` to rounding.
    """
    n_hi = math.ceil(lam + _POISSON_SIGMA * math.sqrt(max(lam, 0.0))) + _POISSON_FLOOR
    ns = np.arange(n_hi + 1, dtype=np.int64)
    log_w = -lam + ns * math.log(lam) - np.array([math.lgamma(k + 1) for k in ns])
    w = np.exp(log_w)
    return ns, np.asarray(w / w.sum(), dtype=np.float64)


def expected_damaged_ratio(mu: FloatArray, *, n_nodes: int = 256) -> float:
    r"""``E[N_cDSB / N_dmg | N_dmg >= 1]`` for independent Poisson loop occupancies.

    The *average-of-ratios* cluster index of one damage pattern, evaluated by
    quadrature rather than sampled. ``mu`` is the mean DSB count of every loop
    (one entry per loop, so a column of ``n_z`` layers appears ``n_z`` times).

    Conditioned on the track position the occupancies ``N_j ~ Poisson(mu_j)`` are
    independent -- the same fact that makes the discrete engine algebraically
    identical to the continuum convolution. Writing
    ``A = #{N_j >= 2}``, ``B = #{N_j >= 1}``, ``q_j = e^{-mu_j}`` and
    ``p2_j = 1 - e^{-mu_j}(1 + mu_j)``, the identity ``1/B = int_0^1 t^{B-1} dt``
    (valid for ``B >= 1``) turns the expectation into one integral of the
    occupancy generating function::

        E[A t^B] = t G(t) sum_j p2_j / (q_j + (1 - q_j) t) ,
        G(t)     = prod_k (q_k + (1 - q_k) t) ,

        E[A/B | B >= 1] = int_0^1 G(t) sum_j p2_j/(q_j + (1-q_j)t) dt
                          / (1 - prod_j q_j) .

    Why this and not the mean-field ``sum_j p2_j / sum_j (1 - q_j)``: a single
    traversal damages only a handful of loops, so ``A`` and ``B`` are far from
    their concentration limit and ``E[A/B] != E[A]/E[B]``. Jensen then pushes the
    ratio-of-averages high -- the documented over-clustering of
    :meth:`~pyradbio.models.lem.lem4.LEM4.cluster_index`.

    ``G`` is accumulated in logs, since it is a product over thousands of loops.
    """
    mu = np.asarray(mu, dtype=np.float64)
    mu = mu[mu > 1.0e-12]  # loops that can never be hit contribute q=1, p2=0
    if mu.size == 0:
        return 0.0
    q = np.exp(-mu)
    p2 = 1.0 - q * (1.0 + mu)
    p_any = -np.expm1(float(np.log(q).sum()))  # 1 - prod q, stable when tiny
    if p_any <= 0.0:
        return 0.0
    t = np.linspace(0.0, 1.0, n_nodes)
    lin = q[None, :] + (1.0 - q)[None, :] * t[:, None]
    integrand = np.exp(np.log(lin).sum(axis=1)) * (p2[None, :] / lin).sum(axis=1)
    return float(np.trapezoid(integrand, t)) / float(p_any)


__all__ = [
    "conditional_loop_counts",
    "disk_overlap_fraction",
    "expected_damaged_ratio",
    "poisson_track_weights",
    "retained_dose_fraction",
    "track_factors",
]
