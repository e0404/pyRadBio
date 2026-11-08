"""Intertrack-cDSB Monte-Carlo kernel for the LEM IV ``beta_I`` (Friedrich 2012).

Reference engine for the *full-simulation* ion ``beta_I`` of Friedrich, Scholz,
Elsasser, Durante & Scholz 2012 (Int. J. Radiat. Biol. 88(1-2):103-107,
doi:10.3109/09553002.2011.611213, Fig. 4). Unlike the single-particle
Kraemer-Scholz slope-matching approximation (`LEMBase._beta`), this places
the discrete DSBs of *many* tracks into a **shared** 540 nm giant-loop grid, so
DSBs from *different* tracks can cluster in the same loop (**intertrack cDSB**).
That raises ``beta_I`` above the slope-matching value in the intermediate-LET
band and shifts the ``beta_I(LET)`` drop to higher LET (the paper's Fig. 4).

The per-loop lethal effect is the bounded GLOBLE model (Friedrich, Durante &
Scholz 2012, Radiat. Res. 178:385, doi:10.1667/RR2964.1):

    E = n_iDSB * eps_i + n_cDSB * eps_c ,   S = exp(-E),

with ``n_iDSB`` / ``n_cDSB`` the number of 540 nm loops holding exactly one /
two-or-more DSB. Using the same ``(eps_i, eps_c)`` as ``LEM4._globle_lethalities``
makes the single-particle and full-simulation ``alpha`` agree by construction
(Friedrich 2012: ``alpha_I`` is unchanged; only ``beta_I`` rises).

The NumPy path (`ion_survival_numpy`) is the deterministic reference; the
compiled numba kernel (`ion_survival_numba`, ``prange`` over realisations)
is the fast path, mirroring `_mc_kernel`. Both compute the same estimator
(mean effect ``S = exp(-<E>)``) and agree within Monte-Carlo error; the numba path
uses numba's per-thread RNG so it is *not* bit-reproducible for a fixed seed (the
NumPy path is). `HAVE_NUMBA` guards the compiled kernel exactly as there.
"""

from __future__ import annotations

import math
from collections.abc import Callable
from typing import TYPE_CHECKING, Any, TypeVar

import numpy as np
import numpy.typing as npt

from pyradbio.core.lq import fit_lq_linearised

if TYPE_CHECKING:
    HAVE_NUMBA = True
    _F = TypeVar("_F", bound=Callable[..., Any])

    def njit(*args: Any, **kwargs: Any) -> Callable[[_F], _F]:
        """Type-preserving decorator-factory signature for ``@njit(...)``."""

    def prange(*args: int) -> range: ...

else:
    try:  # pragma: no cover - trivial import guard
        from numba import njit, prange

        HAVE_NUMBA = True
    except ImportError:  # pragma: no cover - exercised only without the fast extra
        HAVE_NUMBA = False

        def njit(*args, **kwargs):
            """No-op stand-in so the module imports without numba (never JIT-called)."""

            def wrap(func):
                return func

            if len(args) == 1 and callable(args[0]) and not kwargs:
                return args[0]
            return wrap

        def prange(*args):
            return range(*args)


def count_loops(
    ix: npt.NDArray[np.int64],
    iy: npt.NDArray[np.int64],
    iz: npt.NDArray[np.int64],
) -> tuple[int, int]:
    """Isolated / clustered loop counts ``(n_iDSB, n_cDSB)`` from DSB cube indices.

    ``ix, iy, iz`` are the 540 nm-cube indices of every (in-nucleus) DSB. Loops
    (cubes) holding exactly one DSB are isolated (iDSB); two-or-more are clustered
    (cDSB). Empty input -> ``(0, 0)``.
    """
    n = ix.shape[0]
    if n == 0:
        return 0, 0
    # Compact the indices to a single non-negative key per occupied cube.
    ox = ix - ix.min()
    oy = iy - iy.min()
    oz = iz - iz.min()
    nx = int(ox.max()) + 1
    ny = int(oy.max()) + 1
    key = ox.astype(np.int64) + nx * (oy.astype(np.int64) + ny * oz.astype(np.int64))
    counts = np.unique(key, return_counts=True)[1]
    n_i = int(np.count_nonzero(counts == 1))
    n_c = int(np.count_nonzero(counts >= 2))
    return n_i, n_c


def build_radial_cdf(
    r: npt.NDArray[np.float64],
    weight: npt.NDArray[np.float64],
) -> tuple[npt.NDArray[np.float64], npt.NDArray[np.float64], float]:
    """Inverse-CDF nodes for the transverse DSB pdf ``p(r) ~ weight(r)``.

    ``weight = eta(D(r)) * D(r) * r`` is the (unnormalised) DSB density per unit
    radius. Returns ``(cdf, r, integral)`` where ``cdf`` is the normalised
    cumulative of ``weight`` on the log-radius grid ``r`` (so ``r = interp(u, cdf,
    r)`` inverts it for ``u ~ U(0, 1)``) and ``integral = trapz(weight, r)`` (the
    per-track DSB normalisation, ``m_total = 2*pi * kappa * N_z * integral``).
    """
    cum = np.concatenate([[0.0], np.cumsum(0.5 * (weight[1:] + weight[:-1]) * np.diff(r))])
    total = float(cum[-1])
    cdf = cum / total if total > 0.0 else np.linspace(0.0, 1.0, r.shape[0])
    return cdf.astype(np.float64), r.astype(np.float64), total


def _bin_indices(
    x: npt.NDArray[np.float64], y: npt.NDArray[np.float64], z: npt.NDArray[np.float64], ell: float
) -> tuple[npt.NDArray[np.int64], npt.NDArray[np.int64], npt.NDArray[np.int64]]:
    """540 nm-cube indices ``floor(coord / ell)`` for DSB coordinates [um]."""
    return (
        np.floor(x / ell).astype(np.int64),
        np.floor(y / ell).astype(np.int64),
        np.floor(z / ell).astype(np.int64),
    )


def uniform_survival_numpy(
    doses: npt.NDArray[np.float64],
    *,
    eps_i: float,
    eps_c: float,
    a_dsb_per_gy: float,
    n_loops: int,
    n_real: int,
    rng: np.random.Generator,
) -> npt.NDArray[np.float64]:
    """Discrete photon (uniform-field) survival ``S(D)`` for the GLOBLE effect.

    The *photon limit* of the intertrack MC: ``Poisson(a_dsb_per_gy * D)`` discrete
    DSBs are scattered uniformly over exactly ``n_loops = V / l^3`` giant loops (each
    equally likely), and the survival is ``S = exp(-<n_i eps_i + n_c eps_c>)`` (mean
    effect over ``n_real`` realisations), with ``n_i`` / ``n_c`` the loops holding
    exactly one / two-or-more DSB.

    Distributing the *integer* DSB count over the exact loop count reproduces the
    Friedrich 2012 Radiat. Res. 178 Table 1 discreteness setup (their MC vs analytic),
    so it matches the analytic GLOBLE eqs. (2)-(3) to MC noise and its fitted
    ``(alpha, beta)`` recovers the photon ``(alpha_x, beta_x)`` -- the calibration
    anchor. (The uniform limit is shape-independent; the 540 nm *spatial* binning of
    the true nuclear disk is exercised instead by the ion path's alpha cross-check.)
    """
    doses = np.asarray(doses, dtype=np.float64)
    e_sum = np.zeros(doses.shape[0], dtype=np.float64)
    for _ in range(n_real):
        for di, dose in enumerate(doses):
            n_dsb = rng.poisson(a_dsb_per_gy * float(dose))
            if n_dsb == 0:
                continue
            idx = rng.integers(0, n_loops, n_dsb)
            counts = np.unique(idx, return_counts=True)[1]
            n_i = int(np.count_nonzero(counts == 1))
            n_c = int(np.count_nonzero(counts >= 2))
            e_sum[di] += n_i * eps_i + n_c * eps_c
    return np.exp(-e_sum / n_real)


def _ion_effect_batches_numpy(
    doses: npt.NDArray[np.float64],
    *,
    m_total: float,
    cdf: npt.NDArray[np.float64],
    cdf_r: npt.NDArray[np.float64],
    d_1: float,
    r_eff: float,
    r_sim: float,
    height: float,
    ell: float,
    n_batches: int,
    realizations_per_batch: int,
    rng: np.random.Generator,
    score: Callable[[int, int], float],
) -> npt.NDArray[np.float64]:
    """Shared track-placement loop -> ``(n_batches, n_doses)`` mean effects.

    The placement is identical for every scoring rule (GLOBLE or
    photon-equivalent), so both public entry points share this body and differ
    only in ``score(n_iDSB, n_cDSB) -> effect``. Realisations are consumed from
    ``rng`` in one uninterrupted sequence, so a run partitioned into batches
    draws exactly the stream an unbatched run of the same total would -- the
    property `ion_survival_numpy` relies on to stay a thin wrapper.

    Returns **mean effects**, not survivals. Survival is ``exp(-<E>)`` over
    realisations, so the effect is the quantity that pools linearly across
    batches; averaging per-batch survivals would apply Jensen at the batch
    boundary. This is what makes a delete-one jackknife over the rows valid
    (`pyradbio.models.lem.base._delete_one_jackknife`).
    """
    doses = np.asarray(doses, dtype=np.float64)
    d_max = float(doses[-1])
    mean_tracks = (d_max / d_1) * (r_sim / r_eff) ** 2
    r_eff2 = r_eff * r_eff
    e_batches = np.zeros((n_batches, doses.shape[0]), dtype=np.float64)

    for batch in range(n_batches):
        row = e_batches[batch]
        for _ in range(realizations_per_batch):
            n_tracks = int(rng.poisson(mean_tracks))
            if n_tracks == 0:
                continue
            # Track centres in the extended disk, and each track's arrival dose.
            c_rad = r_sim * np.sqrt(rng.random(n_tracks))
            c_ang = 2.0 * math.pi * rng.random(n_tracks)
            x0 = c_rad * np.cos(c_ang)
            y0 = c_rad * np.sin(c_ang)
            arrival = d_max * rng.random(n_tracks)

            counts = rng.poisson(m_total, n_tracks)
            n_dsb = int(counts.sum())
            if n_dsb == 0:
                continue
            track_of = np.repeat(np.arange(n_tracks), counts)
            rr = np.interp(rng.random(n_dsb), cdf, cdf_r)
            th = 2.0 * math.pi * rng.random(n_dsb)
            x = x0[track_of] + rr * np.cos(th)
            y = y0[track_of] + rr * np.sin(th)
            z = height * rng.random(n_dsb)
            a = arrival[track_of]

            inside = (x * x + y * y) <= r_eff2
            x, y, z, a = x[inside], y[inside], z[inside], a[inside]
            if x.shape[0] == 0:
                continue
            ix, iy, iz = _bin_indices(x, y, z, ell)

            for di, dose in enumerate(doses):
                sel = a <= float(dose)
                n_i, n_c = count_loops(ix[sel], iy[sel], iz[sel])
                row[di] += score(n_i, n_c)
    return np.asarray(e_batches / realizations_per_batch, dtype=np.float64)


def ion_survival_batches_numpy(
    doses: npt.NDArray[np.float64],
    *,
    m_total: float,
    cdf: npt.NDArray[np.float64],
    cdf_r: npt.NDArray[np.float64],
    eps_i: float,
    eps_c: float,
    d_1: float,
    r_eff: float,
    r_sim: float,
    height: float,
    ell: float,
    n_batches: int,
    realizations_per_batch: int,
    rng: np.random.Generator,
) -> npt.NDArray[np.float64]:
    """Per-batch mean effects for the GLOBLE scoring -> ``(n_batches, n_doses)``.

    The batched form of `ion_survival_numpy`, supplying the independent
    replicates a delete-one jackknife needs. See `_ion_effect_batches_numpy`
    for why the batch quantity is the effect rather than the survival.
    """
    return _ion_effect_batches_numpy(
        doses,
        m_total=m_total,
        cdf=cdf,
        cdf_r=cdf_r,
        d_1=d_1,
        r_eff=r_eff,
        r_sim=r_sim,
        height=height,
        ell=ell,
        n_batches=n_batches,
        realizations_per_batch=realizations_per_batch,
        rng=rng,
        score=lambda n_i, n_c: n_i * eps_i + n_c * eps_c,
    )


def ion_survival_numpy(
    doses: npt.NDArray[np.float64],
    *,
    m_total: float,
    cdf: npt.NDArray[np.float64],
    cdf_r: npt.NDArray[np.float64],
    eps_i: float,
    eps_c: float,
    d_1: float,
    r_eff: float,
    r_sim: float,
    height: float,
    ell: float,
    n_real: int,
    rng: np.random.Generator,
) -> npt.NDArray[np.float64]:
    """Intertrack-cDSB survival ``S(D)`` for one ion over the dose grid ``doses``.

    Per realisation (Friedrich 2012 full simulation, with two variance-reduction
    devices):

    * **Domain extension** -- ``Poisson((D_max/D_1) (R_sim/R_eff)^2)`` track
      centres are placed uniformly in the extended disk ``R_sim = R_eff + r_max``
      (tracks centred outside the nucleus still deposit penumbra DSBs inside), so
      both the photon limit and the indirect-track contribution are captured.
    * **Common random numbers** -- each track gets a random arrival dose
      ``a_t ~ U(0, D_max)``; the survival at dose ``D`` uses only tracks with
      ``a_t <= D``, giving one correlated, monotone ``S(D)`` curve per realisation
      and drastically reducing the curvature (beta) variance.

    Each track deposits ``Poisson(m_total)`` DSBs at transverse radius sampled from
    the inverse CDF (angle uniform, ``z`` uniform in ``[0, height]``); DSBs are
    masked to the nucleus disk and binned into the **shared** 540 nm grid, so DSBs
    from different tracks land in the same loop (intertrack cDSB). ``doses`` must
    be ascending; ``doses[-1]`` is ``D_max``.

    One pooled curve with no error estimate. `ion_survival_batches_numpy`
    returns the same realisations partitioned into batches, which is what the
    jackknife needs.
    """
    batches = ion_survival_batches_numpy(
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
        n_batches=1,
        realizations_per_batch=n_real,
        rng=rng,
    )
    return np.asarray(np.exp(-batches[0]), dtype=np.float64)


def ion_survival_photon_equivalent_numpy(
    doses: npt.NDArray[np.float64],
    *,
    m_total: float,
    cdf: npt.NDArray[np.float64],
    cdf_r: npt.NDArray[np.float64],
    ci_grid: npt.NDArray[np.float64],
    g_grid: npt.NDArray[np.float64],
    d_1: float,
    r_eff: float,
    r_sim: float,
    height: float,
    ell: float,
    ci_max: float,
    n_real: int,
    rng: np.random.Generator,
) -> npt.NDArray[np.float64]:
    """Intertrack survival ``S(D)`` scored via the photon-equivalent (``D_eq``) effect.

    Identical discrete-DSB placement to `ion_survival_numpy` (many domain-extended
    tracks, shared 540 nm grid, common random numbers), but the per-dose damage pattern
    ``(n_iDSB, n_cDSB)`` is scored through the LEM IV cluster-index -> ``D_eq`` ->
    ``S_gamma(D_eq)`` mapping instead of the bounded GLOBLE per-loop effect: the cluster
    index ``C_I = n_cDSB / (n_iDSB + n_cDSB)`` (capped at ``ci_max``) selects the
    pre-tabulated per-damaged-loop photon effect ``g(C_I)`` and the effect is
    ``E = (n_iDSB + n_cDSB) * g(C_I)``. ``ci_grid`` is ascending;
    ``g = interp(C_I, ci_grid, g_grid)``.

    **Experimental / diagnostic.** The motivation was that ``g``'s ``D_t``
    linear tail (``_photon_effect``) might collapse ``beta`` at high LET (the Friedrich
    2012 Fig. 4 behaviour GLOBLE cannot reproduce). It does not: this scoring recovers
    ``beta_x`` at low LET but ``beta`` *rises* through mid-LET (worse than GLOBLE),
    because the intertrack MC accumulates the damage pattern over dose, so ``C_I(D)``
    *grows* with dose (more tracks -> more intertrack clusters), pushing ``D_eq(D)`` up
    and adding curvature. The collapse needs a dose-*independent* ``C_I``, which a
    dose-accumulating intertrack MC structurally cannot give -- i.e. the collapse is not
    an effect-model artifact. Kept as an opt-in path for the example comparison and to
    document the ruled-out hypothesis.

    One pooled curve with no error estimate;
    `ion_survival_photon_equivalent_batches_numpy` is the batched form.
    """
    batches = ion_survival_photon_equivalent_batches_numpy(
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
        ci_max=ci_max,
        n_batches=1,
        realizations_per_batch=n_real,
        rng=rng,
    )
    return np.asarray(np.exp(-batches[0]), dtype=np.float64)


def ion_survival_photon_equivalent_batches_numpy(
    doses: npt.NDArray[np.float64],
    *,
    m_total: float,
    cdf: npt.NDArray[np.float64],
    cdf_r: npt.NDArray[np.float64],
    ci_grid: npt.NDArray[np.float64],
    g_grid: npt.NDArray[np.float64],
    d_1: float,
    r_eff: float,
    r_sim: float,
    height: float,
    ell: float,
    ci_max: float,
    n_batches: int,
    realizations_per_batch: int,
    rng: np.random.Generator,
) -> npt.NDArray[np.float64]:
    """Per-batch mean effects for the photon-equivalent scoring.

    Batched form of `ion_survival_photon_equivalent_numpy`; see
    `_ion_effect_batches_numpy`.
    """

    def score(n_i: int, n_c: int) -> float:
        n_dmg = n_i + n_c
        if n_dmg == 0:
            return 0.0
        c_i = min(n_c / n_dmg, ci_max)
        return n_dmg * float(np.interp(c_i, ci_grid, g_grid))

    return _ion_effect_batches_numpy(
        doses,
        m_total=m_total,
        cdf=cdf,
        cdf_r=cdf_r,
        d_1=d_1,
        r_eff=r_eff,
        r_sim=r_sim,
        height=height,
        ell=ell,
        n_batches=n_batches,
        realizations_per_batch=realizations_per_batch,
        rng=rng,
        score=score,
    )


@njit(cache=True, fastmath=True)
def _interp_cdf(u: float, cdf: npt.NDArray[np.float64], cdf_r: npt.NDArray[np.float64]) -> float:
    """Invert the transverse pdf CDF: ``np.interp(u, cdf, cdf_r)`` (cdf ascending)."""
    n = cdf.shape[0]
    if u <= cdf[0]:
        return float(cdf_r[0])
    if u >= cdf[n - 1]:
        return float(cdf_r[n - 1])
    lo = 0
    hi = n - 1
    while hi - lo > 1:
        mid = (lo + hi) // 2
        if cdf[mid] <= u:
            lo = mid
        else:
            hi = mid
    t = (u - cdf[lo]) / (cdf[hi] - cdf[lo])
    return float(cdf_r[lo] + t * (cdf_r[hi] - cdf_r[lo]))


@njit(cache=True, parallel=True, fastmath=True)
def _ion_accumulate(
    n_real: int,
    doses: npt.NDArray[np.float64],
    mean_tracks: float,
    m_total: float,
    cdf: npt.NDArray[np.float64],
    cdf_r: npt.NDArray[np.float64],
    eps_i: float,
    eps_c: float,
    d_max: float,
    r_eff: float,
    r_sim: float,
    height: float,
    ell: float,
    nbx: int,
    nbz: int,
) -> npt.NDArray[np.float64]:
    """Per-realisation effect ``E(D)`` for the intertrack MC (compiled, parallel).

    Row ``k`` of the returned ``(n_real, n_doses)`` array is one realisation's
    correlated effect curve (common random numbers). Each realisation is
    independent, so the ``prange`` writes disjoint rows. The DSBs are binned into a
    flat ``(2 nbx+1)^2 (nbz+1)`` loop grid and the isolated/clustered counts are
    tracked *incrementally* in arrival-dose order (a DSB entering an empty cube adds
    an iDSB; the second DSB converts it to a cDSB), so each dose threshold reads off
    the running ``(n_i, n_c)`` in one pass -- the compiled analogue of the NumPy
    path's per-dose `count_loops`.
    """
    n_doses = doses.shape[0]
    e_by_real = np.zeros((n_real, n_doses))
    stride1 = 2 * nbx + 1
    stride2 = stride1 * stride1
    ncells = stride2 * (nbz + 1)
    r_eff2 = r_eff * r_eff
    two_pi = 2.0 * math.pi
    for k in prange(n_real):
        n_tracks = np.random.poisson(mean_tracks)
        if n_tracks == 0:
            continue
        cx = np.empty(n_tracks)
        cy = np.empty(n_tracks)
        at = np.empty(n_tracks)
        counts = np.empty(n_tracks, dtype=np.int64)
        n_dsb = 0
        for i in range(n_tracks):
            rad = r_sim * math.sqrt(np.random.random())
            ang = two_pi * np.random.random()
            cx[i] = rad * math.cos(ang)
            cy[i] = rad * math.sin(ang)
            at[i] = d_max * np.random.random()
            c = np.random.poisson(m_total)
            counts[i] = c
            n_dsb += c
        if n_dsb == 0:
            continue
        key = np.empty(n_dsb, dtype=np.int64)
        arr = np.empty(n_dsb)
        p = 0
        for i in range(n_tracks):
            for _ in range(counts[i]):
                r = _interp_cdf(np.random.random(), cdf, cdf_r)
                ang = two_pi * np.random.random()
                x = cx[i] + r * math.cos(ang)
                y = cy[i] + r * math.sin(ang)
                if x * x + y * y > r_eff2:  # mask to the nucleus disk
                    continue
                z = height * np.random.random()
                ix = int(math.floor(x / ell)) + nbx
                iy = int(math.floor(y / ell)) + nbx
                iz = int(math.floor(z / ell))
                if ix < 0:
                    ix = 0
                elif ix >= stride1:
                    ix = stride1 - 1
                if iy < 0:
                    iy = 0
                elif iy >= stride1:
                    iy = stride1 - 1
                if iz < 0:
                    iz = 0
                elif iz > nbz:
                    iz = nbz
                key[p] = ix + stride1 * iy + stride2 * iz
                arr[p] = at[i]
                p += 1
        n_in = p
        if n_in == 0:
            continue
        order = np.argsort(arr[:n_in])
        count = np.zeros(ncells, dtype=np.int32)
        n_i = 0
        n_c = 0
        di = 0
        for idx in range(n_in):
            j = order[idx]
            a = arr[j]
            while di < n_doses and doses[di] < a:
                e_by_real[k, di] = n_i * eps_i + n_c * eps_c
                di += 1
            cell = key[j]
            cur = count[cell]
            if cur == 0:
                n_i += 1
            elif cur == 1:
                n_i -= 1
                n_c += 1
            count[cell] = cur + 1
        while di < n_doses:
            e_by_real[k, di] = n_i * eps_i + n_c * eps_c
            di += 1
    return e_by_real


def ion_survival_numba(
    doses: npt.NDArray[np.float64],
    *,
    m_total: float,
    cdf: npt.NDArray[np.float64],
    cdf_r: npt.NDArray[np.float64],
    eps_i: float,
    eps_c: float,
    d_1: float,
    r_eff: float,
    r_sim: float,
    height: float,
    ell: float,
    n_real: int,
) -> npt.NDArray[np.float64]:
    """Compiled parallel intertrack survival ``S(D)`` (fast path of
    `ion_survival_numpy`; same estimator, numba per-thread RNG so not
    bit-reproducible). ``doses`` must be ascending."""
    batches = ion_survival_batches_numba(
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
        n_batches=1,
        realizations_per_batch=n_real,
    )
    return np.asarray(np.exp(-batches[0]), dtype=np.float64)


def ion_survival_batches_numba(
    doses: npt.NDArray[np.float64],
    *,
    m_total: float,
    cdf: npt.NDArray[np.float64],
    cdf_r: npt.NDArray[np.float64],
    eps_i: float,
    eps_c: float,
    d_1: float,
    r_eff: float,
    r_sim: float,
    height: float,
    ell: float,
    n_batches: int,
    realizations_per_batch: int,
) -> npt.NDArray[np.float64]:
    """Per-batch mean effects from the compiled kernel -> ``(n_batches, n_doses)``.

    ``_ion_accumulate`` already returns one row per realisation, so batching is a
    reshape: realisations are independent and identically distributed, and the
    ``prange`` schedule does not correlate them, so any equal partition into
    batches is a valid set of jackknife replicates. Unlike the NumPy path this is
    not seed-reproducible (numba per-thread RNG), so the *error bar* is
    reproducible in distribution only.
    """
    doses = np.asarray(doses, dtype=np.float64)
    d_max = float(doses[-1])
    mean_tracks = (d_max / d_1) * (r_sim / r_eff) ** 2
    nbx = int(math.floor(r_eff / ell)) + 2
    nbz = int(math.floor(height / ell)) + 2
    e_by_real = _ion_accumulate(
        n_batches * realizations_per_batch,
        doses,
        mean_tracks,
        m_total,
        np.ascontiguousarray(cdf, dtype=np.float64),
        np.ascontiguousarray(cdf_r, dtype=np.float64),
        eps_i,
        eps_c,
        d_max,
        r_eff,
        r_sim,
        height,
        ell,
        nbx,
        nbz,
    )
    batched = e_by_real.reshape(n_batches, realizations_per_batch, doses.shape[0])
    return np.asarray(batched.mean(axis=1), dtype=np.float64)


@njit(cache=True, parallel=True, fastmath=True)
def _ion_accumulate_pe(
    n_real: int,
    doses: npt.NDArray[np.float64],
    mean_tracks: float,
    m_total: float,
    cdf: npt.NDArray[np.float64],
    cdf_r: npt.NDArray[np.float64],
    ci_grid: npt.NDArray[np.float64],
    g_grid: npt.NDArray[np.float64],
    ci_max: float,
    d_max: float,
    r_eff: float,
    r_sim: float,
    height: float,
    ell: float,
    nbx: int,
    nbz: int,
) -> npt.NDArray[np.float64]:
    """Per-realisation effect ``E(D)`` for the photon-equivalent MC (compiled).

    Identical incremental placement/counting to `_ion_accumulate`, but scores
    each dose threshold as ``E = (n_i + n_c) * g(C_I)`` with ``C_I = n_c/(n_i+n_c)``
    (capped at ``ci_max``) and ``g = interp(C_I, ci_grid, g_grid)`` -- the compiled
    analogue of `ion_survival_photon_equivalent_numpy` (``_interp_cdf`` is a
    generic monotone-``xp`` interpolator, reused here for the ``C_I -> g`` lookup).
    """
    n_doses = doses.shape[0]
    e_by_real = np.zeros((n_real, n_doses))
    stride1 = 2 * nbx + 1
    stride2 = stride1 * stride1
    ncells = stride2 * (nbz + 1)
    r_eff2 = r_eff * r_eff
    two_pi = 2.0 * math.pi
    for k in prange(n_real):
        n_tracks = np.random.poisson(mean_tracks)
        if n_tracks == 0:
            continue
        cx = np.empty(n_tracks)
        cy = np.empty(n_tracks)
        at = np.empty(n_tracks)
        counts = np.empty(n_tracks, dtype=np.int64)
        n_dsb = 0
        for i in range(n_tracks):
            rad = r_sim * math.sqrt(np.random.random())
            ang = two_pi * np.random.random()
            cx[i] = rad * math.cos(ang)
            cy[i] = rad * math.sin(ang)
            at[i] = d_max * np.random.random()
            c = np.random.poisson(m_total)
            counts[i] = c
            n_dsb += c
        if n_dsb == 0:
            continue
        key = np.empty(n_dsb, dtype=np.int64)
        arr = np.empty(n_dsb)
        p = 0
        for i in range(n_tracks):
            for _ in range(counts[i]):
                r = _interp_cdf(np.random.random(), cdf, cdf_r)
                ang = two_pi * np.random.random()
                x = cx[i] + r * math.cos(ang)
                y = cy[i] + r * math.sin(ang)
                if x * x + y * y > r_eff2:  # mask to the nucleus disk
                    continue
                z = height * np.random.random()
                ix = int(math.floor(x / ell)) + nbx
                iy = int(math.floor(y / ell)) + nbx
                iz = int(math.floor(z / ell))
                if ix < 0:
                    ix = 0
                elif ix >= stride1:
                    ix = stride1 - 1
                if iy < 0:
                    iy = 0
                elif iy >= stride1:
                    iy = stride1 - 1
                if iz < 0:
                    iz = 0
                elif iz > nbz:
                    iz = nbz
                key[p] = ix + stride1 * iy + stride2 * iz
                arr[p] = at[i]
                p += 1
        n_in = p
        if n_in == 0:
            continue
        order = np.argsort(arr[:n_in])
        count = np.zeros(ncells, dtype=np.int32)
        n_i = 0
        n_c = 0
        di = 0
        for idx in range(n_in):
            j = order[idx]
            a = arr[j]
            while di < n_doses and doses[di] < a:
                n_dmg = n_i + n_c
                if n_dmg > 0:
                    ci = n_c / n_dmg
                    if ci > ci_max:
                        ci = ci_max
                    e_by_real[k, di] = n_dmg * _interp_cdf(ci, ci_grid, g_grid)
                di += 1
            cell = key[j]
            cur = count[cell]
            if cur == 0:
                n_i += 1
            elif cur == 1:
                n_i -= 1
                n_c += 1
            count[cell] = cur + 1
        while di < n_doses:
            n_dmg = n_i + n_c
            if n_dmg > 0:
                ci = n_c / n_dmg
                if ci > ci_max:
                    ci = ci_max
                e_by_real[k, di] = n_dmg * _interp_cdf(ci, ci_grid, g_grid)
            di += 1
    return e_by_real


def ion_survival_photon_equivalent_numba(
    doses: npt.NDArray[np.float64],
    *,
    m_total: float,
    cdf: npt.NDArray[np.float64],
    cdf_r: npt.NDArray[np.float64],
    ci_grid: npt.NDArray[np.float64],
    g_grid: npt.NDArray[np.float64],
    d_1: float,
    r_eff: float,
    r_sim: float,
    height: float,
    ell: float,
    ci_max: float,
    n_real: int,
) -> npt.NDArray[np.float64]:
    """Compiled parallel photon-equivalent survival ``S(D)`` (fast path of
    `ion_survival_photon_equivalent_numpy`; same estimator, numba per-thread RNG
    so not bit-reproducible). ``doses`` must be ascending."""
    batches = ion_survival_photon_equivalent_batches_numba(
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
        ci_max=ci_max,
        n_batches=1,
        realizations_per_batch=n_real,
    )
    return np.asarray(np.exp(-batches[0]), dtype=np.float64)


def ion_survival_photon_equivalent_batches_numba(
    doses: npt.NDArray[np.float64],
    *,
    m_total: float,
    cdf: npt.NDArray[np.float64],
    cdf_r: npt.NDArray[np.float64],
    ci_grid: npt.NDArray[np.float64],
    g_grid: npt.NDArray[np.float64],
    d_1: float,
    r_eff: float,
    r_sim: float,
    height: float,
    ell: float,
    ci_max: float,
    n_batches: int,
    realizations_per_batch: int,
) -> npt.NDArray[np.float64]:
    """Per-batch mean effects for the compiled photon-equivalent kernel.

    Batched form of `ion_survival_photon_equivalent_numba`; see
    `ion_survival_batches_numba` for the reshape argument and the
    reproducibility caveat.
    """
    doses = np.asarray(doses, dtype=np.float64)
    d_max = float(doses[-1])
    mean_tracks = (d_max / d_1) * (r_sim / r_eff) ** 2
    nbx = int(math.floor(r_eff / ell)) + 2
    nbz = int(math.floor(height / ell)) + 2
    e_by_real = _ion_accumulate_pe(
        n_batches * realizations_per_batch,
        doses,
        mean_tracks,
        m_total,
        np.ascontiguousarray(cdf, dtype=np.float64),
        np.ascontiguousarray(cdf_r, dtype=np.float64),
        np.ascontiguousarray(ci_grid, dtype=np.float64),
        np.ascontiguousarray(g_grid, dtype=np.float64),
        ci_max,
        d_max,
        r_eff,
        r_sim,
        height,
        ell,
        nbx,
        nbz,
    )
    batched = e_by_real.reshape(n_batches, realizations_per_batch, doses.shape[0])
    return np.asarray(batched.mean(axis=1), dtype=np.float64)


def fit_alpha_beta(
    doses: npt.NDArray[np.float64], survival: npt.NDArray[np.float64]
) -> tuple[float, float]:
    """Fit ``-ln S(D) / D = alpha + beta D`` (LQ line) -> ``(alpha, beta)``.

    Fitting the per-dose effect *slope* (a line whose intercept is ``alpha`` and
    slope is ``beta``) is a more stable curvature estimator than a raw quadratic
    on ``-ln S`` when ``beta`` is a small curvature on a dominant linear term
    (same estimator as the LEM I MC).

    **Neither coefficient is clamped.** In deep overkill the dose response has no
    quadratic curvature left, so the fitted ``beta`` scatters through zero, and a
    negative draw is a measurement rather than an error. Flooring it would
    bias the mean upward *and* shrink the apparent spread -- measured on the
    Friedrich 2012 cell at 600 realisations, 16 seeds: at LET 100 the floor bound
    3/16 seeds and moved ``beta`` 0.0632 +- 0.0705 -> 0.0708 +- 0.0542, and at
    LET 130 it inflated the mean 2.1x (0.069 -> 0.148) while halving the spread.
    An error bar computed on a clamped statistic is not an error bar, which is
    why the jackknife path
    ([`LEM4.intertrack_mc_beta_jackknife`][pyradbio.models.lem.lem4.LEM4.intertrack_mc_beta_jackknife])
    consumes this raw value. Same convention, and same reason, as the LEM I Monte-Carlo (the Scholz
    1997 reference batch, ``scripts/lem1_mc_reference.py alpha-beta``).

    Callers needing a physically-signed scalar clamp it themselves and say so:
    [`LEM4.intertrack_mc_beta`][pyradbio.models.lem.lem4.LEM4.intertrack_mc_beta] floors it at 0
    with a ``UserWarning``.

    ``S`` is clipped to ``[1e-300, 1]`` before the log, then the fit is
    [`pyradbio.core.lq.fit_lq_linearised`][] -- the same arithmetic in the same
    order as this function's former body (bit-equality on recorded LEM IV inputs
    is pinned in ``tests/core/test_lq_fits.py``). Degenerate doses (fewer than 2,
    non-positive, a single repeated value) or a NaN survival raise ``ValueError``
    where the former body returned NaN or a minimum-norm solution.
    """
    survival = np.asarray(survival, dtype=np.float64)
    return fit_lq_linearised(doses, -np.log(np.clip(survival, 1.0e-300, 1.0)))
