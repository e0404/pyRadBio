"""Linear-quadratic survival engine.

Conventions: ``-ln S(D) = alpha*D + G*beta*D**2`` with alpha [Gy^-1],
beta [Gy^-2], dose D [Gy] and ``G`` the Lea-Catcheside dose-protraction factor.

``G = 1`` is an acute dose and is the default everywhere, so callers that do not
care about time see the plain LQ model. Anything protracted -- fractionated,
continuous, split-course -- supplies ``G`` from
[`pyradbio.core.protraction`][], which owns the temporal side and computes
``G`` from a [`DoseSchedule`][]. The split is
deliberate: ``G`` depends only on the timing of the irradiation, never on the
cell or the radiation quality, so it is an argument here rather than a field on
[`LQParameters`][].

Two least-squares estimators read ``(alpha, beta)`` back off a dose response:
[`fit_lq_linearised`][] (``-ln S / D = alpha + beta D``, unweighted) and
[`fit_lq_weighted`][] (``-ln S = alpha D + beta D^2``, inverse-sigma weighted,
optional intercept). They are different estimators and give different numbers
on noisy data; pick the one the reference you compare with used.
"""

from __future__ import annotations

import math
from dataclasses import dataclass
from typing import Literal, overload

import numpy as np
import numpy.typing as npt

_G_ROUNDOFF_TOLERANCE: float = 1e-9
"""``G`` is mathematically in ``[0, 1]``, but
[`lea_catcheside_g`][] can return the last bit
above 1.0 through floating-point round-off in its final sum. Accept that
instead of raising on a value that is 1.0 to within a part in 1e9.
"""


def _validate_g(g: float) -> None:
    if not math.isfinite(g) or not 0.0 <= g <= 1.0 + _G_ROUNDOFF_TOLERANCE:
        raise ValueError(f"protraction factor g must be in [0, 1], got {g!r}")


@dataclass(frozen=True, slots=True)
class LQParameters:
    """Linear-quadratic parameters for one (ion, energy) or a mixed field."""

    alpha: float
    beta: float

    def __post_init__(self) -> None:
        if not math.isfinite(self.alpha) or self.alpha < 0.0:
            raise ValueError(f"alpha must be finite and >= 0, got {self.alpha!r}")
        if not math.isfinite(self.beta) or self.beta < 0.0:
            raise ValueError(f"beta must be finite and >= 0, got {self.beta!r}")


def survival(dose_Gy: float, alpha: float, beta: float, *, g: float = 1.0) -> float:
    """Return surviving fraction ``S(D) = exp(-(alpha*D + g*beta*D**2))``."""
    return math.exp(-log_kill(dose_Gy, alpha, beta, g=g))


def log_kill(dose_Gy: float, alpha: float, beta: float, *, g: float = 1.0) -> float:
    """Return ``-ln S(D) = alpha*D + g*beta*D**2``.

    ``g`` is the Lea-Catcheside protraction factor (default ``1.0``, acute);
    see [`pyradbio.core.protraction.lea_catcheside_g`][].
    """
    if dose_Gy < 0.0:
        raise ValueError(f"dose must be >= 0, got {dose_Gy!r}")
    _validate_g(g)
    return alpha * dose_Gy + g * beta * dose_Gy * dose_Gy


def dose_for_survival(
    survival_fraction: float, alpha: float, beta: float, *, g: float = 1.0
) -> float:
    """Invert the LQ model for the dose giving ``survival_fraction``.

    Solves ``g*beta*D**2 + alpha*D - c = 0`` with ``c = -ln S >= 0``. The
    numerically stable root ``D = 2c / (alpha + sqrt(alpha**2 + 4*g*beta*c))``
    is used; it reduces continuously to the ``beta -> 0`` (or ``g -> 0``)
    exponential limit ``D = c / alpha`` (Furusawa/Kanai LQ inversion).
    """
    if not 0.0 < survival_fraction <= 1.0:
        raise ValueError(f"survival fraction must be in (0, 1], got {survival_fraction!r}")
    if alpha < 0.0 or beta < 0.0:
        raise ValueError(f"alpha, beta must be >= 0, got alpha={alpha!r}, beta={beta!r}")
    _validate_g(g)

    c = -math.log(survival_fraction)  # required log-kill, >= 0
    denom = alpha + math.sqrt(alpha * alpha + 4.0 * g * beta * c)
    if denom == 0.0:
        raise ValueError("alpha and beta cannot both be zero")
    return 2.0 * c / denom


def rbe(
    survival_fraction: float,
    alpha: float,
    beta: float,
    *,
    reference: LQParameters,
    g: float = 1.0,
    reference_g: float = 1.0,
) -> float:
    """RBE at a given iso-survival level vs a reference radiation.

    ``RBE = D_reference(S) / D_test(S)`` (Inaniwa 2010, eq. 6 form).

    ``g`` and ``reference_g`` are the protraction factors of the test and
    reference irradiations. They are separate because the two arms need not
    share a schedule -- and because it is the comparison that is most often got
    wrong: RBE quoted for a fractionated ion treatment against an *acute*
    photon reference mixes the fractionation effect into the RBE.
    """
    d_ref = dose_for_survival(survival_fraction, reference.alpha, reference.beta, g=reference_g)
    d_test = dose_for_survival(survival_fraction, alpha, beta, g=g)
    return d_ref / d_test


def bed(dose_Gy: float, *, alpha_beta_ratio_Gy: float, g: float = 1.0) -> float:
    """Biologically effective dose ``BED = D (1 + g D / (alpha/beta))`` [Gy].

    BED is the LQ effect expressed in units of ``alpha``, i.e.
    ``BED = (-ln S)/alpha``: the total dose that would produce the same effect
    if delivered so slowly that the quadratic term vanished. It therefore adds
    across phases of a treatment, which is what makes it the working currency
    for comparing fractionation schedules.

    For a conventional course of ``n`` fractions of size ``d`` with complete
    repair between them, ``g = 1/n`` and ``D = n d``, recovering the familiar
    ``BED = n d (1 + d / (alpha/beta))``.

    Reference: Jones et al. 2001, doi:10.1053/clon.2001.9221.

    Parameters
    ----------
    dose_Gy
        Total physical dose ``>= 0``.
    alpha_beta_ratio_Gy
        The tissue's ``alpha/beta`` [Gy], ``> 0``. Note BED is quoted against a
        specific ratio ("72 Gy10"), and the same schedule has a different BED in
        every tissue.
    g
        Lea-Catcheside protraction factor of the schedule; see
        [`pyradbio.core.protraction.lea_catcheside_g`][].
    """
    if dose_Gy < 0.0:
        raise ValueError(f"dose must be >= 0, got {dose_Gy!r}")
    if not math.isfinite(alpha_beta_ratio_Gy) or alpha_beta_ratio_Gy <= 0.0:
        raise ValueError(f"alpha_beta_ratio_Gy must be finite and > 0, got {alpha_beta_ratio_Gy!r}")
    _validate_g(g)
    return dose_Gy * (1.0 + g * dose_Gy / alpha_beta_ratio_Gy)


def eqd2(
    dose_Gy: float,
    *,
    alpha_beta_ratio_Gy: float,
    g: float = 1.0,
    reference_fraction_Gy: float = 2.0,
) -> float:
    """Dose in 2 Gy fractions that is isoeffective with ``dose_Gy`` [Gy].

    ``EQD2 = BED / (1 + d_ref / (alpha/beta))`` -- the BED converted back into
    the everyday currency of a conventional course delivered in
    ``reference_fraction_Gy`` fractions with complete repair between them. A
    schedule that already *is* 2 Gy fractions returns its own total dose.

    Reference: Withers, Thames & Peters 1983, doi:10.1016/s0167-8140(83)80021-8
    (fraction-size isoeffect); Jones et al. 2001, doi:10.1053/clon.2001.9221.

    ``reference_fraction_Gy`` generalises this to ``EQD_d`` for any reference
    fraction size; the default 2.0 Gy is the clinical convention, not a physical
    constant.
    """
    if not math.isfinite(reference_fraction_Gy) or reference_fraction_Gy <= 0.0:
        raise ValueError(
            f"reference_fraction_Gy must be finite and > 0, got {reference_fraction_Gy!r}"
        )
    effective = bed(dose_Gy, alpha_beta_ratio_Gy=alpha_beta_ratio_Gy, g=g)
    return effective / (1.0 + reference_fraction_Gy / alpha_beta_ratio_Gy)


# ---------------------------------------------------------------------------
# Least-squares LQ fits
# ---------------------------------------------------------------------------

_SIGMA_FLOOR: float = 1e-12
"""Floor on ``sigma`` in [`fit_lq_weighted`][]: a zero spread (every repeat
identical, e.g. no kill at all at the lowest dose) becomes a weight of 1e12
rather than a division by zero. Same floor as ``_lq_fit`` in
``examples/survival_code_lem_comparison.py`` and the cubic fit in
[`pyradbio.models.lem.lem1.SpecificEnergySurvival.straightness`][].
"""


def _as_1d_pair(
    doses: npt.ArrayLike, log_kill: npt.ArrayLike
) -> tuple[npt.NDArray[np.float64], npt.NDArray[np.float64]]:
    d = np.asarray(doses, dtype=np.float64)
    y = np.asarray(log_kill, dtype=np.float64)
    if d.ndim != 1 or y.ndim != 1:
        raise ValueError(f"doses and log_kill must be 1-D, got shapes {d.shape} and {y.shape}")
    if d.shape != y.shape:
        raise ValueError(
            f"doses and log_kill must have the same shape, got {d.shape} and {y.shape}"
        )
    if not np.all(np.isfinite(y)):
        raise ValueError("log_kill/effect must be finite (a survival of 0 gives -ln S = inf)")
    return d, y


def fit_lq_linearised(doses: npt.ArrayLike, log_kill: npt.ArrayLike) -> tuple[float, float]:
    """Unweighted least-squares ``(alpha, beta)`` of the linearised LQ form.

    Fits the straight line ``-ln S(D) / D = alpha + beta D`` -- intercept
    ``alpha``, slope ``beta`` -- to ``log_kill / doses``. This is the estimator
    the LEM uses to read ``alpha``/``beta`` off a simulated dose response
    (Scholz et al. 1997, doi:10.1007/s004110050055, eq. 8; Friedrich et al.
    2012, doi:10.3109/09553002.2011.611213), and it is the more stable
    curvature estimator when ``beta`` is a small correction on a dominant
    linear term. It is **not** the same estimator as an unweighted quadratic
    fit of ``-ln S`` (dividing by ``D`` re-weights the points), so the two give
    different numbers on noisy data.

    **Neither coefficient is clamped.** In single-track overkill the response
    is concave and the fitted ``beta`` scatters through zero; a negative value
    is a measurement, and flooring it would bias any later average upward.
    Callers needing a physically signed ``beta`` clamp it themselves.

    The arithmetic is, in this order, ``y = log_kill / doses``, the design
    ``np.vstack([np.ones_like(doses), doses]).T`` and
    ``np.linalg.lstsq(design, y, rcond=None)``. That is bit-for-bit the
    expression of the copies it stands in for: the original LEM I MC batch
    script's ``fit_alpha_beta`` (called with ``log_kill = -np.log(S)``),
    ``pyradbio/models/lem/_localdose.py`` ``lq_coefficients`` and
    ``pyradbio/models/lem/_intertrack_kernel.py`` ``fit_alpha_beta``; keep it
    that way (``tests/core/test_lq_fits.py`` pins it with ``==``).

    Args:
        doses: 1-D doses [Gy], finite and ``> 0``, with at least 2 distinct values.
        log_kill: ``-ln S(D)`` (or any effect ``E(D)``) at ``doses``, finite.

    Returns:
        ``(alpha [Gy^-1], beta [Gy^-2])`` as Python floats.

    Raises:
        ValueError: on mismatched or non-1-D inputs, fewer than 2 points,
            non-finite ``log_kill``, a non-positive or non-finite dose, or fewer
            than 2 distinct doses (the line is then undetermined).
    """
    # ``log_kill`` (the public name) shadows the module's log_kill() here: use the
    # array ``log_kill_arr`` below, never call log_kill() inside this function.
    d, log_kill_arr = _as_1d_pair(doses, log_kill)
    if d.size < 2:
        raise ValueError(f"need at least 2 doses for an LQ fit, got {d.size}")
    if not np.all(np.isfinite(d) & (d > 0.0)):
        raise ValueError(f"doses must be finite and > 0 for the linearised fit, got {d}")
    if np.unique(d).size < 2:
        raise ValueError(f"need at least 2 distinct doses for an LQ fit, got {d}")
    y = log_kill_arr / d
    design = np.vstack([np.ones_like(d), d]).T
    alpha, beta = np.linalg.lstsq(design, y, rcond=None)[0]
    return float(alpha), float(beta)


@overload
def fit_lq_weighted(
    doses: npt.ArrayLike,
    log_kill: npt.ArrayLike,
    sigma: npt.ArrayLike,
    *,
    intercept: Literal[False] = ...,
) -> tuple[float, float]: ...


@overload
def fit_lq_weighted(
    doses: npt.ArrayLike,
    log_kill: npt.ArrayLike,
    sigma: npt.ArrayLike,
    *,
    intercept: Literal[True],
) -> tuple[float, float, float]: ...


@overload
def fit_lq_weighted(
    doses: npt.ArrayLike,
    log_kill: npt.ArrayLike,
    sigma: npt.ArrayLike,
    *,
    intercept: bool,
) -> tuple[float, float] | tuple[float, float, float]: ...


def fit_lq_weighted(
    doses: npt.ArrayLike,
    log_kill: npt.ArrayLike,
    sigma: npt.ArrayLike,
    *,
    intercept: bool = False,
) -> tuple[float, float] | tuple[float, float, float]:
    """Inverse-sigma weighted least squares of ``-ln S = alpha D + beta D^2``.

    Each row of the design ``(D, D^2)`` and of ``log_kill`` is multiplied by
    ``w = 1 / max(sigma, 1e-12)``, i.e. the residuals are measured in units of
    their own standard error. With ``intercept=False`` (default) the curve is
    forced through the origin; this is the Survival code's own fit
    (``usefulFunctions.cpp:fit_LQ``; Manganaro et al. 2018,
    doi:10.1088/1361-6560/aab697), so Monte-Carlo points fitted with it are
    comparable to Survival's. With ``intercept=True`` a constant column is
    added, ``-ln S = c + alpha D + beta D^2``, for data whose zero-dose
    normalisation is itself uncertain (e.g. a plating-efficiency offset).

    A ``sigma`` of exactly 0 is floored at 1e-12 (an enormous weight) rather
    than rejected: a Monte-Carlo spread over repeats is legitimately 0 where no
    repeat recorded a kill. With ``sigma`` all ones the weights are exactly 1.0,
    so the fit is bit-for-bit the plain unweighted quadratic
    ``np.linalg.lstsq(np.vstack([D, D**2]).T, log_kill)`` (as in
    ``pyradbio/benchmark/matsuya2025.py`` ``Fig2Curve.lq_fit``).

    The arithmetic is, in this order, ``w = 1.0 / np.maximum(sigma, 1e-12)``,
    ``np.column_stack([doses, doses**2]) * w[:, None]`` and
    ``np.linalg.lstsq(design, log_kill * w, rcond=None)`` -- bit-for-bit
    ``examples/survival_code_lem_comparison.py`` ``_lq_fit``,
    pinned with ``==`` in ``tests/core/test_lq_fits.py``. Neither coefficient
    is clamped.

    Args:
        doses: 1-D doses [Gy], finite and ``>= 0`` (a zero dose is a valid
            row here, nothing is divided by ``D``).
        log_kill: ``-ln S(D)`` at ``doses``, finite.
        sigma: standard error of each ``log_kill``, finite and ``>= 0``.
        intercept: also fit a constant term ``c``.

    Returns:
        ``(alpha, beta)`` without intercept, ``(alpha, beta, c)`` with it, as
        Python floats.

    Raises:
        ValueError: on mismatched or non-1-D inputs, non-finite ``log_kill``,
            a negative or non-finite dose or ``sigma``, or too few distinct
            doses to determine the model (2 distinct positive doses without
            intercept, 3 distinct doses with it).
    """
    # ``log_kill`` shadows the module's log_kill() here (see fit_lq_linearised).
    d, log_kill_arr = _as_1d_pair(doses, log_kill)
    s = np.asarray(sigma, dtype=np.float64)
    if s.shape != d.shape:
        raise ValueError(f"sigma must have the same shape as doses, got {s.shape} and {d.shape}")
    if not np.all(np.isfinite(s) & (s >= 0.0)):
        raise ValueError(f"sigma must be finite and >= 0, got {s}")
    if not np.all(np.isfinite(d) & (d >= 0.0)):
        raise ValueError(f"doses must be finite and >= 0, got {d}")
    if intercept:
        if np.unique(d).size < 3:
            raise ValueError(
                f"need at least 3 distinct doses for an LQ fit with intercept, got {d}"
            )
    elif np.unique(d[d > 0.0]).size < 2:
        raise ValueError(f"need at least 2 distinct positive doses for an LQ fit, got {d}")
    w = 1.0 / np.maximum(s, _SIGMA_FLOOR)
    if intercept:
        design = np.column_stack([np.ones_like(d), d, d**2]) * w[:, None]
        coeffs, *_ = np.linalg.lstsq(design, log_kill_arr * w, rcond=None)
        return float(coeffs[1]), float(coeffs[2]), float(coeffs[0])
    design = np.column_stack([d, d**2]) * w[:, None]
    coeffs, *_ = np.linalg.lstsq(design, log_kill_arr * w, rcond=None)
    return float(coeffs[0]), float(coeffs[1])
