"""Numba-accelerated kernel for the LEM IV discrete DSB-pattern.

This module is an *optional acceleration* of `LEM4._damage_pattern`. It
reproduces, as compiled parallel loops, the two hot spots of the discrete
single-traversal DSB pattern: (1) the radical-diffusion convolution of the
Scholz-Kraft radial dose (Bessel-I0 form) evaluated over the radial profile, and
(2) the fine 2-D subvolume scoring that bins the eta-enhanced DSB yield into
540 nm giant-loop subvolumes. Both are embarrassingly parallel and are fused into
``prange`` kernels so the profile's ~1500 convolution integrals and the ~10^6
grid points spread across cores.

Numba is an optional (``fast`` extra) dependency. If it is not installed the
``@njit`` decorators degrade to no-ops and `HAVE_NUMBA` is ``False``; the
kernel is then never called (the pure-NumPy path in ``lem4.py`` is used instead,
so importing this module never requires numba). The compiled path is *not*
bit-identical to the NumPy path (it uses a polynomial ``i0e`` and trapezoidal
quadrature instead of scipy's ``i0e``/Simpson) but agrees to <~0.5%.
"""

from __future__ import annotations

import math
from collections.abc import Callable
from typing import TYPE_CHECKING, Any, TypeVar

import numpy as np
import numpy.typing as npt

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


_TWO_PI = 2.0 * math.pi
_SQRT2 = math.sqrt(2.0)
_TINY = 1.0e-300


@njit(cache=True, fastmath=True)
def _i0e(x: float) -> float:
    """Exponentially scaled modified Bessel function ``I0(x) e^{-|x|}``.

    Abramowitz & Stegun 9.8.1/9.8.2 polynomial approximation (rel. error <5e-7
    vs scipy ``i0e`` over the ion-track argument range).
    """
    ax = x if x >= 0.0 else -x
    if ax < 3.75:
        t = x / 3.75
        t2 = t * t
        i0 = 1.0 + t2 * (
            3.5156229
            + t2
            * (3.0899424 + t2 * (1.2067492 + t2 * (0.2659732 + t2 * (0.0360768 + t2 * 0.0045813))))
        )
        return i0 * math.exp(-ax)
    t = 3.75 / ax
    p = 0.39894228 + t * (
        0.01328592
        + t
        * (
            0.00225319
            + t
            * (
                -0.00157565
                + t
                * (
                    0.00916281
                    + t * (-0.02057706 + t * (0.02635537 + t * (-0.01647633 + t * 0.00392377)))
                )
            )
        )
    )
    return p / math.sqrt(ax)


@njit(cache=True, fastmath=True)
def _interp_loglog(
    r: float,
    log_r: npt.NDArray[np.float64],
    log_d: npt.NDArray[np.float64],
    r_top: float,
) -> float:
    """Log-log interpolate the radial dose at ``r``; flat below the grid, 0 past ``r_top``."""
    if r > r_top:
        return 0.0
    n = log_r.shape[0]
    lr = math.log(r) if r > 0.0 else log_r[0]
    if lr <= log_r[0]:
        return math.exp(log_d[0])
    if lr >= log_r[n - 1]:
        return math.exp(log_d[n - 1])
    lo = 0
    hi = n - 1
    while hi - lo > 1:
        mid = (lo + hi) // 2
        if log_r[mid] <= lr:
            lo = mid
        else:
            hi = mid
    w = (lr - log_r[lo]) / (log_r[hi] - log_r[lo])
    return math.exp(log_d[lo] + w * (log_d[hi] - log_d[lo]))


@njit(cache=True, fastmath=True)
def _convolved_dose(
    r_out: float,
    r_min: float,
    r_max: float,
    sigma: float,
    lam_let: float,
    r_floor: float,
    log_nodes: npt.NDArray[np.float64],
) -> float:
    """Radical-diffusion-convolved Scholz-Kraft dose at ``r_out`` (Bessel-I0 form).

    Mirrors ``ScholzKraftRadialDose._convolved_dose_Gy``: integrate
    ``base(r') * (r'/sigma^2) * exp(-(r-r')^2/2sigma^2) * I0(r r'/sigma^2)`` over
    a log grid (shared ``log_nodes``) merged with a linear peak grid spanning
    +-6 sigma about ``r_out``. Trapezoidal quadrature on the sorted union.
    """
    s2 = sigma * sigma
    n_log = log_nodes.shape[0]
    # Peak-refinement nodes (linear, +-6 sigma about r_out), 129 as in the NumPy path.
    peak_lo = r_out - 6.0 * sigma
    if peak_lo < r_floor:
        peak_lo = r_floor
    peak_hi = r_out + 6.0 * sigma
    if peak_hi > r_max:
        peak_hi = r_max
    n_peak = 129 if peak_hi > peak_lo else 0
    n_tot = n_log + n_peak
    nodes = np.empty(n_tot)
    for i in range(n_log):
        nodes[i] = log_nodes[i]
    if n_peak > 0:
        step = (peak_hi - peak_lo) / (n_peak - 1)
        for i in range(n_peak):
            nodes[n_log + i] = peak_lo + i * step
    nodes.sort()

    integral = 0.0
    prev_r = nodes[0]
    base_prev = lam_let / (max(prev_r, r_min) ** 2)
    ker_prev = (
        (prev_r / s2) * math.exp(-((r_out - prev_r) ** 2) / (2.0 * s2)) * _i0e(r_out * prev_r / s2)
    )
    f_prev = base_prev * ker_prev
    for i in range(1, n_tot):
        rp = nodes[i]
        if rp == prev_r:
            continue
        base = lam_let / (max(rp, r_min) ** 2)
        ker = (rp / s2) * math.exp(-((r_out - rp) ** 2) / (2.0 * s2)) * _i0e(r_out * rp / s2)
        f = base * ker
        integral += 0.5 * (f + f_prev) * (rp - prev_r)
        prev_r = rp
        f_prev = f
    return integral if integral > 0.0 else 0.0


@njit(cache=True, parallel=True, fastmath=True)
def damage_pattern(
    r_grid: npt.NDArray[np.float64],
    r_min: float,
    r_max: float,
    sigma: float,
    lam_let: float,
    r_floor: float,
    log_nodes: npt.NDArray[np.float64],
    r_eff: float,
    r_top: float,
    xs: npt.NDArray[np.float64],
    span: float,
    ell: float,
    nsub: int,
    cell_area: float,
    kappa_over_l2: float,
    kappa_dsb: float,
    ssb: float,
    dsb: float,
    stagger: float,
    genome: float,
) -> tuple[float, float, float]:
    """Discrete DSB-pattern loop counts ``(N_dmg, N_cDSB, in_nucleus_dose_area)``
    [um^2] of one central traversal. Parallel over the radial profile (convolution)
    and the fine grid.

    Reproduces `LEM4._damage_pattern`: the convolved dose profile on
    ``r_grid``, then the eta-enhanced DSB yield binned into ``nsub^2`` subvolumes
    on the fine grid ``xs x xs`` (track at a subvolume centre), plus the analytic
    isolated-DSB penumbra (total radial dose-area minus the fine-square dose-area).
    """
    # 1) convolved dose profile on r_grid (parallel over radii)
    ng = r_grid.shape[0]
    dose_grid = np.empty(ng)
    for k in prange(ng):
        dose_grid[k] = _convolved_dose(r_grid[k], r_min, r_max, sigma, lam_let, r_floor, log_nodes)
    log_r = np.log(r_grid)
    log_d = np.empty(ng)
    for k in range(ng):
        log_d[k] = math.log(dose_grid[k] if dose_grid[k] > _TINY else _TINY)

    # 2) fine 2-D subvolume scoring (parallel over rows, per-row subvolume rows)
    n = xs.shape[0]
    inv_ratio = ssb / dsb
    sub = np.zeros((n, nsub * nsub))  # per-row subvolume eta-dose accumulators
    # (the plain dose-area is not accumulated here -- the penumbra is
    # integrated radially below, see the comment at `pen`)
    for i in prange(n):
        xi = xs[i]
        ix = int((xi + span) / ell)
        if ix < 0:
            ix = 0
        elif ix >= nsub:
            ix = nsub - 1
        for j in range(n):
            xj = xs[j]
            r = math.sqrt(xi * xi + xj * xj)
            d = _interp_loglog(r, log_r, log_d, r_top)
            if d <= 0.0:
                continue
            rs = ssb * d / genome * stagger
            rd = dsb * d / genome * stagger
            es = math.exp(-rs)
            eta = 1.0 + (1.0 - es) / (3.0 - es) * math.exp(-rd) * inv_ratio
            iy = int((xj + span) / ell)
            if iy < 0:
                iy = 0
            elif iy >= nsub:
                iy = nsub - 1
            sub[i, ix * nsub + iy] += eta * d * cell_area

    # reduce rows -> per-subvolume mean DSB m, then Poisson iDSB/cDSB counts
    n_dmg = 0.0
    n_cdsb = 0.0
    l2 = ell * ell
    for s in range(nsub * nsub):
        acc = 0.0
        for i in range(n):
            acc += sub[i, s]
        m = kappa_over_l2 * acc
        em = math.exp(-m)
        n_dmg += l2 * (1.0 - em)
        n_cdsb += l2 * (1.0 - em * (1.0 + m))
    # Penumbra isolated DSB, integrated DIRECTLY over the complement of the fine
    # square. (A difference `(total radial dose-area) - (fine-grid sum)` would be a
    # difference of two different quadratures of overlapping domains: it does not
    # vanish when the whole profile fits inside the square, which is the entire
    # sub-MeV regime, and there that residual WAS the isolated-DSB area.)
    # `out` is the fraction of the circle of radius r lying outside a square of
    # half-width `span`: 0 inside the inscribed circle, 1 beyond span*sqrt(2), and
    # (4/pi) acos(span/r) between (four congruent arcs of 2 acos(span/r) each).
    span_diag = span * _SQRT2
    total_area = 0.0
    pen = 0.0
    prev_r = r_grid[0]
    f_prev = dose_grid[0] * _TWO_PI * prev_r
    p_prev = (
        0.0
        if prev_r <= span
        else f_prev * (1.0 if prev_r >= span_diag else (4.0 / math.pi) * math.acos(span / prev_r))
    )
    for k in range(1, ng):
        rr = r_grid[k]
        d = dose_grid[k]
        if rr > r_top:
            d = 0.0
        f = d * _TWO_PI * rr
        if rr <= span:
            out = 0.0
        elif rr >= span_diag:
            out = 1.0
        else:
            out = (4.0 / math.pi) * math.acos(span / rr)
        p = f * out
        total_area += 0.5 * (f + f_prev) * (rr - prev_r)
        pen += 0.5 * (p + p_prev) * (rr - prev_r)
        prev_r = rr
        f_prev = f
        p_prev = p
    if pen > 0.0:
        n_dmg += kappa_dsb * pen
    # total_area = in-nucleus radial dose-area (integrated to r_top = min(r_max, r_eff));
    # the caller uses it for the escaped-energy (indirect) initial-slope correction.
    return n_dmg, n_cdsb, total_area
