"""Benchmark metrics.

Aggregate metrics follow the community / PIDE conventions so pyRadBio results are
directly comparable to the RBE-model surveys built on the same database
(Friedrich et al. 2013, doi:10.1093/jrr/rrs114; the model comparison in
doi:10.1667/RR13511.1). They quantify how well a model's predicted alpha (or RBE at a
fixed survival) reproduces the measured value across a set of ion/energy points:

  - **mean log error** ``mean( log10(pred) - log10(meas) )`` — the *signed*
    logarithmic deviation, i.e. the systematic over/under-prediction bias. Zero
    means unbiased; +0.3 means predictions are on average ~2x too high.
  - **mean absolute log error** ``mean( |log10(pred) - log10(meas)| )`` — the
    *unsigned* logarithmic spread, the headline "how far off" number used to rank
    models (lower is better).
  - **SMAPE** — symmetric mean absolute percentage error,
    ``mean( |pred - meas| / (|pred| + |meas|) )`` in ``[0, 1]``; a bounded,
    scale-free complement to the log metrics that also tolerates zeros.

The log metrics operate in log space because alpha and RBE span orders of magnitude
and multiplicative error is the physically meaningful quantity; they therefore
require strictly positive inputs. Residual stratification (by ion species and by
alpha/β) is done by the caller on the per-point residuals, not here.

Two curve-level summaries, used by the reconstruction scripts, sit beside them:
:func:`shape_residuals` (a model curve vs a digitized paper curve at the paper's
x-values: point residuals plus the scale-vs-shape split) and :func:`ratio_summary`
(the ``ours / reference`` ratio statistics the scripts print). Printing stays in
the scripts.
"""

from __future__ import annotations

import math
from collections.abc import Iterator, Sequence
from dataclasses import dataclass

import numpy as np
import numpy.typing as npt


def _paired(predicted: Sequence[float], measured: Sequence[float]) -> Iterator[tuple[float, float]]:
    """Yield ``(pred, meas)`` pairs, validating length and non-emptiness."""
    if len(predicted) != len(measured):
        raise ValueError(f"predicted/measured length mismatch: {len(predicted)} vs {len(measured)}")
    if not predicted:
        raise ValueError("need at least one (predicted, measured) pair")
    return zip(predicted, measured, strict=True)


def _log10_residuals(predicted: Sequence[float], measured: Sequence[float]) -> list[float]:
    """``log10(pred) - log10(meas)`` per pair; requires strictly positive inputs."""
    residuals: list[float] = []
    for p, m in _paired(predicted, measured):
        if p <= 0.0 or m <= 0.0:
            raise ValueError(
                "log-error metrics require strictly positive values "
                f"(got predicted={p}, measured={m})"
            )
        residuals.append(math.log10(p) - math.log10(m))
    return residuals


def mean_log_error(predicted: Sequence[float], measured: Sequence[float]) -> float:
    """Signed mean logarithmic error ``mean(log10(pred) - log10(meas))``.

    The systematic bias: positive => the model over-predicts on average. Inputs
    must be strictly positive (alpha, RBE). Raises ``ValueError`` on empty input,
    length mismatch, or a non-positive value.
    """
    residuals = _log10_residuals(predicted, measured)
    return math.fsum(residuals) / len(residuals)


def mean_absolute_log_error(predicted: Sequence[float], measured: Sequence[float]) -> float:
    """Unsigned mean logarithmic error ``mean(|log10(pred) - log10(meas)|)``.

    The headline goodness-of-fit number (lower is better); insensitive to the
    sign of the deviation. Same positivity/shape requirements as
    :func:`mean_log_error`.
    """
    residuals = _log10_residuals(predicted, measured)
    return math.fsum(abs(r) for r in residuals) / len(residuals)


def smape(predicted: Sequence[float], measured: Sequence[float]) -> float:
    """Symmetric mean absolute percentage error in ``[0, 1]``.

    ``mean( |pred - meas| / (|pred| + |meas|) )``. Bounded and scale-free: 0 for a
    perfect match, 1 when one value is zero (or they have opposite signs). A pair
    that is exactly ``(0, 0)`` contributes 0 (no error). Raises ``ValueError`` on
    empty input or length mismatch.
    """
    total = 0.0
    n = 0
    for p, m in _paired(predicted, measured):
        n += 1
        denom = abs(p) + abs(m)
        if denom == 0.0:
            continue  # (0, 0): no error
        total += abs(p - m) / denom
    return total / n


# ---- curve-level residual summaries (used by the reconstruction scripts) ----------


@dataclass(frozen=True)
class ShapeResiduals:
    """Point and shape residuals of a model curve against one reference curve.

    ``rms``            ``sqrt(mean((pred - ref)^2))`` -- absolute, NaN if any point is.
    ``rel_rms``        ``sqrt(nanmean(((pred - ref)/ref)^2))`` over points with ``|ref| > 1e-9``.
    ``rel_bias``       ``nanmean((pred - ref)/ref)``, same points -- the signed offset.
    ``scale_k``        least-squares ``k`` minimising ``||k ref - pred||``
                       (1.0 if ``ref . ref == 0``).
    ``shape_fraction`` ``rms(pred - k ref) / rms``: near 0 the mismatch is a pure scale
                       offset (normalisation, stopping power), near 1 the shapes differ;
                       0.0 when ``rms <= 1e-12`` *or ``rms`` is NaN*.
    ``max_rel``        ``nanmax(|pred - ref| / |ref|)``, same points as ``rel_rms``.
    ``n``              number of reference points (NaN predictions included).
    """

    rms: float
    rel_rms: float
    rel_bias: float
    scale_k: float
    shape_fraction: float
    max_rel: float
    n: int


def shape_residuals(reference: npt.ArrayLike, predicted: npt.ArrayLike) -> ShapeResiduals:
    """Point + shape residuals of ``predicted`` against ``reference`` (same x-values).

    The caller evaluates its model at the reference x-values and passes the result
    as ``predicted``; the expressions are those of ``validation/elsasser2007_lem2.py``
    and ``validation/scholz_lem1.py``, in the same order.

    NaN semantics: a NaN prediction (e.g. an
    interpolation outside the model's support) makes the absolute ``rms`` and
    ``scale_k`` NaN (``np.mean``/``np.dot`` propagate), which in turn sets
    ``shape_fraction`` to 0.0 (``nan > 1e-12`` is False); the relative metrics
    (``rel_rms``, ``rel_bias``, ``max_rel``) use the nan-aware reductions and skip
    it, together with every point whose ``|reference| <= 1e-9``. With no usable
    relative point they are NaN and NumPy emits its usual ``RuntimeWarning``.
    """
    py = np.asarray(reference, dtype=float)
    my = np.asarray(predicted, dtype=float)
    if py.shape != my.shape:
        raise ValueError(f"reference/predicted length mismatch: {py.shape} vs {my.shape}")

    diff = my - py
    rms = float(np.sqrt(np.mean(diff**2)))
    denom = np.where(np.abs(py) > 1e-9, py, np.nan)
    rel_bias = float(np.nanmean(diff / denom))
    k = float(np.dot(py, my) / np.dot(py, py)) if np.dot(py, py) > 0 else 1.0
    rms_scaled = float(np.sqrt(np.mean((my - k * py) ** 2)))
    return ShapeResiduals(
        rms=rms,
        rel_rms=float(np.sqrt(np.nanmean((diff / denom) ** 2))),
        rel_bias=rel_bias,
        scale_k=k,
        shape_fraction=rms_scaled / rms if rms > 1e-12 else 0.0,
        max_rel=float(np.nanmax(np.abs(diff / denom))),
        n=len(py),
    )


@dataclass(frozen=True)
class RatioSummary:
    """Statistics of ``r = ours / reference`` over the non-NaN ratios.

    ``n``            number of non-NaN ratios summarised.
    ``median``       ``nanmedian(r)``.
    ``mean_rel``     ``nanmean(r - 1)`` -- the signed bias.
    ``rms_rel``      ``sqrt(nanmean((r - 1)^2))``.
    ``rms_log``      ``sqrt(nanmean(ln(r)^2))``, unmasked as in osmk: a zero ratio makes
                     it ``inf``, a negative one has a NaN log and is skipped here only
                     (``n`` and every other field still count it); NumPy warns.
    ``max_abs_rel``  ``nanmax(|r - 1|)``.
    ``min_ratio``    ``nanmin(r)``.
    ``max_ratio``    ``nanmax(r)``.

    All fields but ``n`` are NaN when ``n == 0``. Fractions, not percent: the
    scripts multiply by 100 when they print.
    """

    n: int
    median: float
    mean_rel: float
    rms_rel: float
    rms_log: float
    max_abs_rel: float
    min_ratio: float
    max_ratio: float


def ratio_summary(
    ours: npt.ArrayLike,
    reference: npt.ArrayLike,
    *,
    mask: npt.ArrayLike | None = None,
) -> RatioSummary:
    """Summarise ``ours / reference`` -- the ratio lines the scripts print.

    ``mask`` (boolean, same shape) selects points *before* the division, so a
    masked-out zero reference never divides; NaN ratios that survive the mask
    (e.g. an interpolation outside the model's support) are skipped by the
    nan-aware reductions. On ratios without NaN those reductions return the same
    bits as the plain ones, so one function serves every call site below -- but the
    sites marked *plain* use ``min``/``max``/``mean``/``median`` and print ``nan``
    on a NaN ratio where this function prints a number: they match only while the
    ratio is NaN-free (true of their inputs today), which an adopter can check
    with ``n == ratio count`` (after any mask). The call sites are in
    ``validation/``:

    - the OSMK per-series row (``nanmedian``, RMS of ``ln r``, ``max |r - 1|``
      over a ratio carrying NaN) and its pooled "overall" line (the same three,
      plain, over the NaN-stripped pool);
    - the SMK pooled z* ratio line (*plain*: median, RMS of ``ln r``,
      ``max |r - 1|``; ``mask`` = resolvable and ``mine > 0``);
    - the LEM IV ``rel_vs_curve`` (*plain*; ``mask=ref != 0``; ``rms_rel`` and
      ``mean_rel``);
    - the LEM I RBE(1%) ratio line (*plain*: ``min_ratio``, ``max_ratio``,
      ``rms_rel``);
    - the LEM II ``fig4_logratio`` median ratio (``mask`` = both positive) and
      the genome-size bias (``mean_rel``, ``mask`` = both finite) -- *plain*, but
      NaN-free by their masks;
    - the IMK z*_1D spread (*plain*: ``min_ratio``, ``max_ratio``).
    """
    o = np.asarray(ours, dtype=float)
    r = np.asarray(reference, dtype=float)
    if o.shape != r.shape:
        raise ValueError(f"ours/reference length mismatch: {o.shape} vs {r.shape}")
    if mask is not None:
        m = np.asarray(mask, dtype=bool)
        if m.shape != o.shape:
            raise ValueError(f"mask shape {m.shape} does not match the data shape {o.shape}")
        o, r = o[m], r[m]

    ratio = o / r
    n = int(np.count_nonzero(~np.isnan(ratio)))
    if n == 0:
        nan = math.nan
        return RatioSummary(
            n=0,
            median=nan,
            mean_rel=nan,
            rms_rel=nan,
            rms_log=nan,
            max_abs_rel=nan,
            min_ratio=nan,
            max_ratio=nan,
        )
    return RatioSummary(
        n=n,
        median=float(np.nanmedian(ratio)),
        mean_rel=float(np.nanmean(ratio - 1.0)),
        rms_rel=float(np.sqrt(np.nanmean((ratio - 1.0) ** 2))),
        rms_log=float(np.sqrt(np.nanmean(np.log(ratio) ** 2))),
        max_abs_rel=float(np.nanmax(np.abs(ratio - 1))),
        min_ratio=float(np.nanmin(ratio)),
        max_ratio=float(np.nanmax(ratio)),
    )
