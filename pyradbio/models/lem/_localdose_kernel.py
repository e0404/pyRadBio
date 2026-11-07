"""Numba-accelerated kernel for the local-dose box average.

Optional acceleration of :func:`pyradbio.models.lem._localdose.box_averaged_profile`,
which is ~94% of a ``LEM4.local_dose_beta`` call: a per-``rho`` loop of independent
2-D numerical integrals (the direction-averaged circle-in-square overlap ``K(r, rho)``
folded against the radial dose). The loop over ``rho`` is embarrassingly parallel and
pure floating point, so it maps onto a ``prange`` kernel for a ~20-25x speedup.

Unlike :mod:`._lem4_kernel`, this kernel is **numerically identical** to the NumPy
reference to machine precision (max relative error ~1e-13): it runs the *same*
trapezoidal quadrature over the *same* log-``r`` / linear-``delta`` nodes, just as a
compiled loop instead of vectorised NumPy. There is no polynomial-``i0e`` / Simpson
substitution to diverge, so the accelerated path needs no separate tolerance and the
dose-area conservation identity holds identically.

Numba is an optional (``fast`` extra) dependency. Without it the ``@njit`` decorators
degrade to no-ops, :data:`HAVE_NUMBA` is ``False``, and the kernel is never called
(``box_averaged_profile`` uses its NumPy path), so importing this module never requires
numba.
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


@njit(cache=True, fastmath=False)
def _centred_arc_scalar(s: float, half: float) -> float:
    """Angular measure of the circle ``|p| = s`` inside the origin-centred square of
    half-side ``half`` (scalar form of ``_localdose._centred_square_arc``)."""
    if s <= half:
        return 2.0 * math.pi
    c = half / s
    if c > 1.0:
        c = 1.0
    elif c < -1.0:
        c = -1.0
    v = 2.0 * math.pi - 8.0 * math.acos(c)
    return v if v > 0.0 else 0.0


@njit(parallel=True, cache=True, fastmath=False)
def box_average(
    log_r: npt.NDArray[np.float64],
    log_d: npt.NDArray[np.float64],
    r_lo: float,
    r_hi: float,
    inner: float,
    rho: npt.NDArray[np.float64],
    ell: float,
    n_nodes: int,
) -> npt.NDArray[np.float64]:
    """Compiled ``box_averaged_profile``: mean subvolume dose ``g_B(rho)``.

    ``(log_r, log_d)`` are the log radial grid and log dose of the single-track
    profile (with ``inner = dose[0]`` the flat core value and ``r_lo``/``r_hi`` the
    profile support). For each ``rho`` this runs the identical nested trapezoid
    quadrature as the NumPy reference: an inner integral over ``delta`` giving the
    kernel ``K(r, rho) = (1/pi) int Theta_0(s) ddelta`` and an outer integral over a
    per-``rho`` log-``r`` band, folded against the log-log-interpolated dose. The
    ``rho`` loop is a ``prange`` (each output independent), so results are identical
    to the serial NumPy path to rounding.
    """
    half = 0.5 * ell
    reach = half * math.sqrt(2.0)
    out = np.empty(rho.shape[0], dtype=np.float64)
    for i in prange(rho.shape[0]):
        centre = rho[i]
        rl = centre - reach
        if rl < r_lo:
            rl = r_lo
        rh = centre + reach
        lr0 = math.log10(rl)
        lr1 = math.log10(rh)
        d_max = math.pi
        if centre > reach:
            a = reach / centre
            d_max = math.asin(a if a < 1.0 else 1.0)
        ddelta = d_max / (n_nodes - 1)
        acc = 0.0
        prev_integrand = 0.0
        prev_node = 0.0
        for k in range(n_nodes):
            node = 10.0 ** (lr0 + (lr1 - lr0) * k / (n_nodes - 1))
            kacc = 0.0
            prev_arc = 0.0
            for j in range(n_nodes):
                delta = ddelta * j
                s2 = node * node + centre * centre - 2.0 * centre * node * math.cos(delta)
                s = math.sqrt(s2) if s2 > 0.0 else 0.0
                arc = _centred_arc_scalar(s, half)
                if j > 0:
                    kacc += 0.5 * (arc + prev_arc) * ddelta
                prev_arc = arc
            kernel = kacc / math.pi
            if node <= r_lo:
                dose = inner
            elif node >= r_hi:
                dose = 0.0
            else:
                dose = math.exp(np.interp(math.log(node), log_r, log_d))
            integrand = dose * kernel * node
            if k > 0:
                acc += 0.5 * (integrand + prev_integrand) * (node - prev_node)
            prev_integrand = integrand
            prev_node = node
        out[i] = acc / (ell * ell)
    return out


__all__ = ["HAVE_NUMBA", "box_average"]
