"""Numba-accelerated kernel for the stochastic-LEM Monte Carlo (Scholz 1997 exact path).

This module is an *optional acceleration* of the two stochastic-LEM estimators in
``lem1.py``, reproducing each as compiled scalar loops so the NumPy temporaries
per field integral disappear and the embarrassingly parallel work spreads across
cores:

* `kellerer_accumulate` (with `kellerer_ratio_sum`) -- the per-dose
  path, Kellerer weighted sampling, fused all the way from the Poisson traversal
  draw to ``S = exp(-N_leth)``.
* `prefix_field_integrals` -- the traversal-conditioned path, which scores
  every traversal-count prefix from one sweep and therefore still uses the hybrid
  uniform/near-track mixture (its points are shared across prefixes, so they
  cannot be weighted by a prefix-dependent dose field).

Numba is an optional (``fast`` extra) dependency. If it is not installed the
``@njit`` decorators degrade to no-ops and `HAVE_NUMBA` is ``False``; the
kernel functions are then never called (the pure-NumPy engine in ``lem1.py`` is
used instead), so importing this module never requires numba.

The numba engine uses numba's per-thread parallel RNG and is therefore *not*
bit-reproducible for a fixed seed (unlike the deterministic NumPy engine); its
statistics are unbiased and match the NumPy engine within Monte-Carlo error.
"""

from __future__ import annotations

import math
from collections.abc import Callable
from typing import TYPE_CHECKING, Any, TypeVar

import numpy as np
import numpy.typing as npt

if TYPE_CHECKING:
    # numba ships py.typed, but its jit-only ``prange`` is not iterable in a plain
    # typed context; give mypy clean signatures here and use the real numba (or the
    # no-op fallback) at runtime below.
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

            # support both @njit and @njit(...) forms
            if len(args) == 1 and callable(args[0]) and not kwargs:
                return args[0]
            return wrap

        def prange(*args):
            return range(*args)


_TWO_PI = 2.0 * math.pi


@njit(cache=True, fastmath=True)
def _interp1d(x: float, xs: npt.NDArray[np.float64], ys: npt.NDArray[np.float64]) -> float:
    """Linear interpolation of ``ys`` at ``x`` on ascending ``xs`` (flat outside)."""
    n = xs.shape[0]
    if x <= xs[0]:
        return float(ys[0])
    if x >= xs[n - 1]:
        return float(ys[n - 1])
    lo = 0
    hi = n - 1
    while hi - lo > 1:
        mid = (lo + hi) // 2
        if xs[mid] <= x:
            lo = mid
        else:
            hi = mid
    t = (x - xs[lo]) / (xs[hi] - xs[lo])
    return float(ys[lo] + t * (ys[hi] - ys[lo]))


@njit(cache=True, fastmath=True)
def _dose_at(
    dist: float,
    log_r: npt.NDArray[np.float64],
    log_dose: npt.NDArray[np.float64],
    r_lo: float,
    r_max: float,
) -> float:
    """Log-log interpolate ``D(dist)`` from the profile; 0 beyond ``r_max``.

    Mirrors ``LEM1._two_track_dose_at``: clamp to ``[r_lo, r_max]``, interpolate
    in log-log space, and zero out past ``r_max``.
    """
    if dist > r_max:
        return 0.0
    d = dist if dist > r_lo else r_lo
    return math.exp(_interp1d(math.log(d), log_r, log_dose))


@njit(cache=True, inline="always", fastmath=True)
def _analytic_dose(
    d2: float,
    r_max2: float,
    r_min2: float,
    core_dose: float,
    k_dose: float,
) -> float:
    """Unconvolved core / ``1/r^2`` / cut-off, on the *squared* distance.

    Kept separate from `_profile_dose` rather than folded into it because
    this body contains no call at all, which is what lets numba vectorise the
    track loop. Routing the unconvolved case through the banded helper -- whose
    rare ``_dose_at`` branch is never taken there -- measured **2.4x slower** on
    LEM I carbon 100: the mere presence of the call blocks the SIMD, so the cost
    is paid on every evaluation to serve a branch taken on none of them.
    """
    if d2 > r_max2:
        return 0.0
    if d2 < r_min2:
        return core_dose
    return k_dose / d2


@njit(cache=True, inline="always", fastmath=True)
def _profile_dose(
    d2: float,
    r_max2: float,
    r_min2: float,
    r_switch2: float,
    r_edge2: float,
    core_dose: float,
    k_dose: float,
    two_sigma2: float,
    log_r: npt.NDArray[np.float64],
    log_dose: npt.NDArray[np.float64],
    r_lo: float,
    r_max: float,
) -> float:
    r"""One track's dose contribution at *squared* distance ``d2``.

    The closed form is ``k/r^2`` outside the core, times a radical-diffusion
    correction where the profile was convolved. Convolving the ``1/r^2`` law with
    a 2-D Gaussian has no elementary closed form -- it is an inverse moment of a
    noncentral chi-square with 2 dof and ``lambda = r^2/sigma^2``, i.e. a
    Marcum-Q plus Poisson-weighted incomplete gammas, whose term count grows like
    ``r^2/(2 sigma^2)`` and so is unusable in the far field. But the *small-sigma*
    (heat-kernel) expansion is elementary. In 2-D ``grad^2 r^-m = m^2 r^-m-2``, so
    ``grad^2n r^-2 = 4^n (n!)^2 r^-2-2n`` and

    ```text
    D_conv(r) = (k/r^2) * sum_n n! x^n,     x = 2 sigma^2 / r^2
    ```

    which is asymptotic (the ``n!`` diverges) with optimal truncation at
    ``n ~ 1/x``. Two terms are kept. Measured against the actual convolution that
    is within ~1e-5 from ``r = 50 nm`` outward and ~3e-6 beyond 100 nm, for LEM II
    and LEM III alike over H/C and 10-300 MeV/u -- *below* the ~1e-5 numerical
    floor of the tabulated profile it replaces. Inside ``r_switch`` the expansion
    is not valid and the table is used; ``LEMBase._analytic_profile`` measures
    both ``sigma`` and ``r_switch`` from the sampled profile rather than
    recognising a track model.

    An unconvolved profile passes ``two_sigma2 = r_switch2 = 0``, which makes the
    correction factor exactly ``1.0`` and the table branch unreachable -- so LEM I
    runs the arithmetic it ran before this argument existed.
    """
    if d2 > r_max2:
        return 0.0
    if d2 < r_switch2 or d2 > r_edge2:
        return _dose_at(math.sqrt(d2), log_r, log_dose, r_lo, r_max)
    if d2 < r_min2:
        return core_dose
    x = two_sigma2 / d2
    return (k_dose / d2) * (1.0 + x * (1.0 + 2.0 * x))


@njit(cache=True, fastmath=True)
def _logkill(
    d: float,
    a: float,
    bq: float,
    d_t: float,
    s_max: float,
    lk_t: float,
    eta_log_d: npt.NDArray[np.float64],
    eta_val: npt.NDArray[np.float64],
) -> float:
    """Photon ``-ln S_x(D)``: LQ below ``D_t``, linear above (see ``_photon_log_kill_array``).

    An empty ``eta_log_d`` selects the LEM I form ``lk_t + s_max (D - D_t)`` on
    exactly the arithmetic it used before this argument existed. A non-empty one
    selects LEM II/III eq. 10, ``lk_t + s_max (eta(D) D - D_t)``, with ``eta``
    linearly interpolated in ``log10 D`` and flat outside the table -- which is
    what ``ClusterModel.eta`` does, and its first node is ``(1 Gy, 1.0)``, so the
    flat clamp reproduces the below-onset floor rather than approximating it.
    """
    if d <= d_t:
        return a * d + bq * d * d
    if eta_log_d.shape[0] == 0:
        return lk_t + s_max * (d - d_t)
    return lk_t + s_max * (_interp1d(math.log10(d), eta_log_d, eta_val) * d - d_t)


@njit(cache=True, parallel=True, fastmath=True)
def kellerer_ratio_sum(
    pts_x: npt.NDArray[np.float64],
    pts_y: npt.NDArray[np.float64],
    cx: npt.NDArray[np.float64],
    cy: npt.NDArray[np.float64],
    dose_bg: float,
    log_r: npt.NDArray[np.float64],
    log_dose: npt.NDArray[np.float64],
    r_lo: float,
    r_max: float,
    analytic: bool,
    core_dose: float,
    k_dose: float,
    r_min: float,
    r_switch: float,
    r_edge: float,
    two_sigma2: float,
    a: float,
    bq: float,
    d_t: float,
    s_max: float,
    lk_t: float,
    eta_log_d: npt.NDArray[np.float64],
    eta_val: npt.NDArray[np.float64],
) -> float:
    r"""``sum_i h(d(x_i)) / d(x_i)`` over Kellerer-weighted sample points.

    The whole inner loop of `LEM1.kellerer_field_integral`. Because the
    sampling density is the local dose, the estimator only needs this ratio sum;
    the ``D_tot`` prefactor is exact and stays on the Python side.

    ``dose_bg`` is the uniform far-field background from the ``b_cut_um`` split,
    added to every point. With ``analytic`` the radial lookup needs no sqrt, log,
    binary search or exp -- only a compare and a divide on the *squared*
    distance. See `_profile_dose` for ``r_switch`` / ``two_sigma2``.
    """
    total = 0.0
    n = cx.size
    r_max2 = r_max * r_max
    r_min2 = r_min * r_min
    r_switch2 = r_switch * r_switch
    r_edge2 = r_edge * r_edge
    # Hoisted out of the track loop: see _analytic_dose for why it matters.
    banded = r_switch2 > 0.0 or r_edge2 < r_max2
    for i in prange(pts_x.size):
        px = pts_x[i]
        py = pts_y[i]
        d = dose_bg
        if analytic and not banded:
            for j in range(n):
                ddx = px - cx[j]
                ddy = py - cy[j]
                d += _analytic_dose(ddx * ddx + ddy * ddy, r_max2, r_min2, core_dose, k_dose)
        elif analytic:
            for j in range(n):
                ddx = px - cx[j]
                ddy = py - cy[j]
                d += _profile_dose(
                    ddx * ddx + ddy * ddy,
                    r_max2,
                    r_min2,
                    r_switch2,
                    r_edge2,
                    core_dose,
                    k_dose,
                    two_sigma2,
                    log_r,
                    log_dose,
                    r_lo,
                    r_max,
                )
        else:
            for j in range(n):
                ddx = px - cx[j]
                ddy = py - cy[j]
                d += _dose_at(math.sqrt(ddx * ddx + ddy * ddy), log_r, log_dose, r_lo, r_max)
        if d > 0.0:
            total += _logkill(d, a, bq, d_t, s_max, lk_t, eta_log_d, eta_val) / d
    return total


@njit(cache=True, parallel=True, fastmath=True)
def prefix_field_integrals(
    pts_x: npt.NDArray[np.float64],
    pts_y: npt.NDArray[np.float64],
    cx: npt.NDArray[np.float64],
    cy: npt.NDArray[np.float64],
    log_r: npt.NDArray[np.float64],
    log_dose: npt.NDArray[np.float64],
    r_lo: float,
    r_max: float,
    analytic: bool,
    core_dose: float,
    k_dose: float,
    r_min: float,
    r_switch: float,
    r_edge: float,
    two_sigma2: float,
    r_eff: float,
    a_eff: float,
    track_fraction: float,
    ln_ratio: float,
    a: float,
    bq: float,
    d_t: float,
    s_max: float,
    lk_t: float,
    eta_log_d: npt.NDArray[np.float64],
    eta_val: npt.NDArray[np.float64],
    n_blocks: int,
) -> tuple[npt.NDArray[np.float64], npt.NDArray[np.float64]]:
    r"""Per-prefix overlap-excess sums for the traversal-conditioned estimator.

    The compiled twin of ``LEM1._montecarlo_prefix_field_integrals``' inner
    block. Returns ``(sum vals, sum vals^2)`` over field points, indexed by
    prefix ``k = 1..n``; the caller keeps the exact ``nu(b)`` additive part, the
    ``1/total`` normalisation and the delta-method de-biasing.

    NumPy has to materialise the whole ``(points, tracks)`` block and sweep it
    several times (dose, log-kill, two cumulative sums, the density). Here one
    pass per point carries a running dose and running log-kill, so every prefix
    falls out with no allocation at all.

    Field points are still drawn once from the *full* track set by the caller --
    that is what keeps every prefix unbiased, and it is why the density loop runs
    over all ``n`` tracks even when scoring prefix ``k``.

    Parallel over blocks of points, each with its own accumulator row, so the
    reduction is race-free and the result does not depend on thread scheduling.
    """
    n = cx.size
    npts = pts_x.size
    acc = np.zeros((n_blocks, n))
    acc_sq = np.zeros((n_blocks, n))
    per = (npts + n_blocks - 1) // n_blocks
    r_max2 = r_max * r_max
    r_min2 = r_min * r_min
    r_switch2 = r_switch * r_switch
    r_edge2 = r_edge * r_edge
    # Hoisted out of the track loop: see _analytic_dose for why it matters.
    banded = r_switch2 > 0.0 or r_edge2 < r_max2
    two_pi_ln = 2.0 * math.pi * ln_ratio

    for blk in prange(n_blocks):
        lo = blk * per
        hi = lo + per
        if hi > npts:
            hi = npts
        for i in range(lo, hi):
            px = pts_x[i]
            py = pts_y[i]
            if px * px + py * py > r_eff * r_eff:
                continue  # outside the nucleus: scores zero, but still counted
            # density of the full mixture at this point (all n tracks)
            p = (1.0 - track_fraction) / a_eff
            for j in range(n):
                dx = px - cx[j]
                dy = py - cy[j]
                dist = math.sqrt(dx * dx + dy * dy)
                if r_lo <= dist <= r_max:
                    dd = dist if dist > r_lo else r_lo
                    p += track_fraction / n / (two_pi_ln * dd * dd)
            if p <= 0.0:
                continue
            inv_p = 1.0 / p
            # one sweep: running dose and running log-kill give every prefix
            cum_d = 0.0
            cum_h = 0.0
            for j in range(n):
                dx = px - cx[j]
                dy = py - cy[j]
                if analytic and not banded:
                    dj = _analytic_dose(dx * dx + dy * dy, r_max2, r_min2, core_dose, k_dose)
                elif analytic:
                    dj = _profile_dose(
                        dx * dx + dy * dy,
                        r_max2,
                        r_min2,
                        r_switch2,
                        r_edge2,
                        core_dose,
                        k_dose,
                        two_sigma2,
                        log_r,
                        log_dose,
                        r_lo,
                        r_max,
                    )
                else:
                    dj = _dose_at(math.sqrt(dx * dx + dy * dy), log_r, log_dose, r_lo, r_max)
                cum_d += dj
                cum_h += _logkill(dj, a, bq, d_t, s_max, lk_t, eta_log_d, eta_val)
                v = (_logkill(cum_d, a, bq, d_t, s_max, lk_t, eta_log_d, eta_val) - cum_h) * inv_p
                acc[blk, j] += v
                acc_sq[blk, j] += v * v

    out = np.zeros(n)
    out_sq = np.zeros(n)
    for blk in range(n_blocks):
        for j in range(n):
            out[j] += acc[blk, j]
            out_sq[j] += acc_sq[blk, j]
    return out, out_sq


@njit(cache=True, fastmath=True)
def _bisect_left(a: npt.NDArray[np.float64], x: float, n: int) -> int:
    """First index ``i < n`` with ``a[i] >= x``; ``n`` if there is none.

    Matches ``np.searchsorted(a[:n], x, side='left')`` on an ascending ``a``.
    """
    lo = 0
    hi = n
    while lo < hi:
        mid = (lo + hi) // 2
        if a[mid] < x:
            lo = mid + 1
        else:
            hi = mid
    return lo


@njit(cache=True, fastmath=True)
def _kellerer_z_and_survival(
    lam: float,
    gen_radius: float,
    bg_mean: float,
    bg_sd: float,
    r_eff: float,
    b_lo: float,
    b_step: float,
    mean_dose: npt.NDArray[np.float64],
    log_r_of_u: npt.NDArray[np.float64],
    half_of_u: npt.NDArray[np.float64],
    n_field: int,
    log_r: npt.NDArray[np.float64],
    log_dose: npt.NDArray[np.float64],
    r_lo: float,
    r_max: float,
    analytic: bool,
    core_dose: float,
    k_dose: float,
    r_min: float,
    r_switch: float,
    r_edge: float,
    two_sigma2: float,
    a: float,
    bq: float,
    d_t: float,
    s_max: float,
    lk_t: float,
    eta_log_d: npt.NDArray[np.float64],
    eta_val: npt.NDArray[np.float64],
) -> tuple[float, float]:
    r"""``(z, S)`` for one Kellerer-weighted realization.

    The compiled twin of ``LEM1._mc_accumulate_kellerer``'s loop body fused with
    ``LEM1.kellerer_field_integral``: Poisson traversal count, near-track
    placement, dose-proportional source selection, the pre-inverted radial draw,
    and the ratio sum, with no intermediate array crossing back into Python.

    ``z`` is the nuclear **specific energy** the configuration deposited,
    ``sum_i d_bar(b_i)`` over every track. It costs nothing to return: the
    weighted estimator already forms it as the normalisation ``D_tot`` of its own
    selection CDF. With a ``b_cut_um`` split the far tracks enter through
    ``dose_bg``, whose closed-form compound-Poisson moments are exactly their
    contribution to ``z`` -- so the split makes ``z`` reachable at energies where
    enumerating every track never would be.

    ``mean_dose``, ``log_r_of_u`` and ``half_of_u`` are the ``NuclearDoseTable``
    columns; the impact-parameter grid is passed as ``(b_lo, b_step)`` rather
    than as an array because ``LEM1.nuclear_dose_table`` builds it with
    ``linspace``, so the lookup is index arithmetic instead of a binary search.
    That matters: at proton 70 the near-track loop runs 14 000 times per
    realization, six times per configuration.
    """
    n = np.random.poisson(lam)

    dose_bg = 0.0
    if bg_sd > 0.0:
        dose_bg = bg_mean + bg_sd * np.random.normal()
        if dose_bg < 0.0:
            dose_bg = 0.0

    if n == 0:
        if dose_bg <= 0.0:
            return 0.0, 1.0
        return dose_bg, math.exp(-_logkill(dose_bg, a, bq, d_t, s_max, lk_t, eta_log_d, eta_val))

    n_b = mean_dose.shape[0]
    n_u = log_r_of_u.shape[1]
    b_hi = b_lo + b_step * (n_b - 1)

    # Near-track centres, and the running dose-weight CDF in the same pass.
    cx = np.empty(n)
    cy = np.empty(n)
    cw = np.empty(n + 1)
    run = 0.0
    for i in range(n):
        rad = gen_radius * math.sqrt(np.random.random())
        ang = _TWO_PI * np.random.random()
        cx[i] = rad * math.cos(ang)
        cy[i] = rad * math.sin(ang)
        # |centre| is `rad` by construction, so d_bar needs no hypot.
        if rad <= b_lo:
            run += mean_dose[0]
        elif rad >= b_hi:
            run += mean_dose[n_b - 1]
        else:
            t = (rad - b_lo) / b_step
            j0 = int(t)
            frac = t - j0
            run += mean_dose[j0] + frac * (mean_dose[j0 + 1] - mean_dose[j0])
        cw[i] = run
    if run <= 0.0:
        return run + dose_bg, 1.0
    cw[n] = run + dose_bg
    d_tot = cw[n]

    r_max2 = r_max * r_max
    r_min2 = r_min * r_min
    r_switch2 = r_switch * r_switch
    r_edge2 = r_edge * r_edge
    # Hoisted out of the track loop: see _analytic_dose for why it matters.
    banded = r_switch2 > 0.0 or r_edge2 < r_max2
    ratio_sum = 0.0
    for _fp in range(n_field):
        pick = _bisect_left(cw, np.random.random() * d_tot, n + 1)
        if pick >= n:
            # The far field: one further source of weight `dose_bg`, uniform.
            rad = r_eff * math.sqrt(np.random.random())
            ang = _TWO_PI * np.random.random()
            px = rad * math.cos(ang)
            py = rad * math.sin(ang)
        else:
            b_src = math.hypot(cx[pick], cy[pick])
            slot = int(math.ceil((b_src - b_lo) / b_step)) - 1
            if slot < 0:
                slot = 0
            elif slot > n_b - 1:
                slot = n_b - 1
            pos = np.random.random() * (n_u - 1)
            i0 = int(pos)
            if i0 > n_u - 2:
                i0 = n_u - 2
            frac = pos - i0
            lo_w = 1.0 - frac
            rad = math.exp(log_r_of_u[slot, i0] * lo_w + log_r_of_u[slot, i0 + 1] * frac)
            half = half_of_u[slot, i0] * lo_w + half_of_u[slot, i0 + 1] * frac
            # Azimuth inside the arc of that radius which lies in the nucleus,
            # measured from the track -> centre direction: every point is inside
            # by construction, so nothing is drawn and discarded.
            phi = (2.0 * np.random.random() - 1.0) * half
            # b = 0 leaves the track->centre direction undefined, and dividing
            # (0, 0) by the guard gives a zero-length vector rather than a unit
            # one, collapsing every sample onto the track axis. Any fixed
            # direction is right there; keep this in step with the NumPy path.
            if b_src > 0.0:
                ux = -cx[pick] / b_src
                uy = -cy[pick] / b_src
            else:
                ux = 1.0
                uy = 0.0
            cos_p = math.cos(phi)
            sin_p = math.sin(phi)
            px = cx[pick] + rad * (ux * cos_p - uy * sin_p)
            py = cy[pick] + rad * (ux * sin_p + uy * cos_p)

        d = dose_bg
        if analytic and not banded:
            for j in range(n):
                ddx = px - cx[j]
                ddy = py - cy[j]
                d += _analytic_dose(ddx * ddx + ddy * ddy, r_max2, r_min2, core_dose, k_dose)
        elif analytic:
            for j in range(n):
                ddx = px - cx[j]
                ddy = py - cy[j]
                d += _profile_dose(
                    ddx * ddx + ddy * ddy,
                    r_max2,
                    r_min2,
                    r_switch2,
                    r_edge2,
                    core_dose,
                    k_dose,
                    two_sigma2,
                    log_r,
                    log_dose,
                    r_lo,
                    r_max,
                )
        else:
            for j in range(n):
                ddx = px - cx[j]
                ddy = py - cy[j]
                d += _dose_at(math.sqrt(ddx * ddx + ddy * ddy), log_r, log_dose, r_lo, r_max)
        if d > 0.0:
            ratio_sum += _logkill(d, a, bq, d_t, s_max, lk_t, eta_log_d, eta_val) / d

    return d_tot, math.exp(-d_tot * ratio_sum / n_field)


@njit(cache=True, parallel=True, fastmath=True)
def kellerer_z_samples(
    n_real: int,
    lam: float,
    gen_radius: float,
    bg_mean: float,
    bg_sd: float,
    r_eff: float,
    b_lo: float,
    b_step: float,
    mean_dose: npt.NDArray[np.float64],
    log_r_of_u: npt.NDArray[np.float64],
    half_of_u: npt.NDArray[np.float64],
    n_field: int,
    log_r: npt.NDArray[np.float64],
    log_dose: npt.NDArray[np.float64],
    r_lo: float,
    r_max: float,
    analytic: bool,
    core_dose: float,
    k_dose: float,
    r_min: float,
    r_switch: float,
    r_edge: float,
    two_sigma2: float,
    a: float,
    bq: float,
    d_t: float,
    s_max: float,
    lk_t: float,
    eta_log_d: npt.NDArray[np.float64],
    eta_val: npt.NDArray[np.float64],
) -> tuple[npt.NDArray[np.float64], npt.NDArray[np.float64]]:
    """Per-realization ``(z, S)`` pairs, weighted sampler, in parallel.

    The sample-returning twin of `kellerer_accumulate`, for survival
    conditioned on specific energy rather than on dose. Both come out of the same
    track configuration, which is the whole point -- ``S(z)`` is an average over
    configurations that produced a given ``z``, so the two must not be sampled
    independently.

    Writes into preallocated rows rather than accumulating, since the caller bins
    by ``z`` afterwards; ``prange`` over realizations is race-free because each
    writes its own index.
    """
    z_out = np.empty(n_real)
    s_out = np.empty(n_real)
    for k in prange(n_real):
        z_out[k], s_out[k] = _kellerer_z_and_survival(
            lam,
            gen_radius,
            bg_mean,
            bg_sd,
            r_eff,
            b_lo,
            b_step,
            mean_dose,
            log_r_of_u,
            half_of_u,
            n_field,
            log_r,
            log_dose,
            r_lo,
            r_max,
            analytic,
            core_dose,
            k_dose,
            r_min,
            r_switch,
            r_edge,
            two_sigma2,
            a,
            bq,
            d_t,
            s_max,
            lk_t,
            eta_log_d,
            eta_val,
        )
    return z_out, s_out


@njit(cache=True, parallel=True, fastmath=True)
def kellerer_accumulate(
    n_real: int,
    lam: float,
    gen_radius: float,
    bg_mean: float,
    bg_sd: float,
    r_eff: float,
    b_lo: float,
    b_step: float,
    mean_dose: npt.NDArray[np.float64],
    log_r_of_u: npt.NDArray[np.float64],
    half_of_u: npt.NDArray[np.float64],
    n_field: int,
    log_r: npt.NDArray[np.float64],
    log_dose: npt.NDArray[np.float64],
    r_lo: float,
    r_max: float,
    analytic: bool,
    core_dose: float,
    k_dose: float,
    r_min: float,
    r_switch: float,
    r_edge: float,
    two_sigma2: float,
    a: float,
    bq: float,
    d_t: float,
    s_max: float,
    lk_t: float,
    eta_log_d: npt.NDArray[np.float64],
    eta_val: npt.NDArray[np.float64],
) -> tuple[float, float]:
    """Accumulate ``(sum S, sum S^2)`` over realizations, weighted sampler, in parallel.

    It needs no ``nu(b)`` control variate: with the selection density equal to the
    local dose the estimator is already bounded-variance, so there is no additive
    part to subtract off -- unlike the removed uniform accumulator, which carried
    the exact single-track sum and sampled only the overlap excess.

    Like every ``prange`` kernel here the result is not bit-reproducible across
    thread counts -- numba seeds one generator per thread. Pin ``engine="numpy"``
    for assertions that need an exact number.
    """
    s_sum = 0.0
    s_sq = 0.0
    for _k in prange(n_real):
        _z, s = _kellerer_z_and_survival(
            lam,
            gen_radius,
            bg_mean,
            bg_sd,
            r_eff,
            b_lo,
            b_step,
            mean_dose,
            log_r_of_u,
            half_of_u,
            n_field,
            log_r,
            log_dose,
            r_lo,
            r_max,
            analytic,
            core_dose,
            k_dose,
            r_min,
            r_switch,
            r_edge,
            two_sigma2,
            a,
            bq,
            d_t,
            s_max,
            lk_t,
            eta_log_d,
            eta_val,
        )
        s_sum += s
        s_sq += s * s
    return s_sum, s_sq
