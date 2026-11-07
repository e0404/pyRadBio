"""Residual summaries: each test reproduces a script's inline computation and
asserts the helper returns the same bits, NaN cases included."""

from __future__ import annotations

import dataclasses
import math
import warnings

import numpy as np
import numpy.typing as npt
import pytest

from pyradbio.benchmark.metrics import (
    RatioSummary,
    ShapeResiduals,
    ratio_summary,
    shape_residuals,
)


def _same(a: float, b: float) -> bool:
    """Bit-equality that also treats NaN == NaN."""
    return (math.isnan(a) and math.isnan(b)) or a == b


def _lem2_compare(py: npt.NDArray[np.float64], my: npt.NDArray[np.float64]) -> dict[str, float]:
    """The original ``lem2_cluster_effect`` script's ``compare``, with ``model_fn(px)``
    already evaluated into ``my``."""
    py = np.asarray(py, dtype=float)
    my = np.asarray(my, dtype=float)

    diff = my - py
    rms = float(np.sqrt(np.mean(diff**2)))
    denom = np.where(np.abs(py) > 1e-9, py, np.nan)
    rel_bias = float(np.nanmean(diff / denom))
    k = float(np.dot(py, my) / np.dot(py, py)) if np.dot(py, py) > 0 else 1.0
    rms_scaled = float(np.sqrt(np.mean((my - k * py) ** 2)))
    return {
        "rms": rms,
        "rel_rms": float(np.sqrt(np.nanmean((diff / denom) ** 2))),
        "rel_bias": rel_bias,
        "scale_k": k,
        "shape_fraction": rms_scaled / rms if rms > 1e-12 else 0.0,
        "max_rel": float(np.nanmax(np.abs(diff / denom))),
        "n": float(len(py)),
    }


def _assert_matches_lem2(stats: ShapeResiduals, ref: dict[str, float]) -> None:
    for name in ("rms", "rel_rms", "rel_bias", "scale_k", "shape_fraction", "max_rel"):
        assert _same(getattr(stats, name), ref[name]), name
    assert stats.n == int(ref["n"])


SHAPE_CASES = {
    "clean": (np.array([1.0, 2.0, 4.0, 8.0]), np.array([1.1, 2.3, 3.9, 8.8])),
    "pure_scale": (np.array([1.0, 2.0, 4.0]), np.array([1.5, 3.0, 6.0])),
    "zero_reference_point": (np.array([0.0, 2.0, 4.0]), np.array([0.3, 2.2, 3.8])),
    # interp_on(...) returns NaN outside the model's LET support (left/right=nan):
    "nan_prediction": (np.array([1.0, 2.0, 4.0, 8.0]), np.array([np.nan, 2.3, 3.9, np.nan])),
    "exact_match": (np.array([1.0, 2.0]), np.array([1.0, 2.0])),
    "all_zero_reference": (np.array([0.0, 0.0]), np.array([0.5, 0.1])),
    # small but above lem2's |ref| > 1e-9 cut: a relative point, not a dropped one
    "tiny_reference": (np.array([1e-6, 2.0, 4.0]), np.array([2e-6, 2.1, 3.9])),
    # nonzero but below the cut: dropped from the relative metrics, kept in the rest
    "below_cut_reference": (np.array([1e-10, 2.0, 4.0]), np.array([0.5, 2.1, 3.9])),
}


@pytest.mark.parametrize("case", sorted(SHAPE_CASES))
def test_shape_residuals_bit_identical_to_lem2_compare(case: str) -> None:
    py, my = SHAPE_CASES[case]
    with warnings.catch_warnings():
        warnings.simplefilter("ignore", RuntimeWarning)  # all-NaN nanmean/nanmax
        ref = _lem2_compare(py, my)
        stats = shape_residuals(py, my)
    _assert_matches_lem2(stats, ref)


def test_shape_residuals_nan_semantics_are_lem2s() -> None:
    """A NaN prediction poisons the absolute ``rms``/``scale_k`` (np.mean, np.dot),
    which zeroes ``shape_fraction`` (``nan > 1e-12`` is False), while the relative
    metrics skip it (nanmean/nanmax)."""
    py, my = SHAPE_CASES["nan_prediction"]
    stats = shape_residuals(py, my)
    assert math.isnan(stats.rms) and math.isnan(stats.scale_k)
    assert stats.shape_fraction == 0.0
    assert math.isfinite(stats.rel_rms) and math.isfinite(stats.rel_bias)
    assert stats.n == 4


def test_shape_residuals_relative_cut_is_lem2s_1e9() -> None:
    """lem2 keeps a relative point only where ``|ref| > 1e-9``: a 1e-10 reference
    is dropped (a zero or 1e-12 cut would keep it and blow up ``rel_rms``), while
    the absolute metrics still see it."""
    py, my = SHAPE_CASES["below_cut_reference"]
    stats = shape_residuals(py, my)
    tail = shape_residuals(py[1:], my[1:])
    assert (stats.rel_rms, stats.rel_bias, stats.max_rel) == (
        tail.rel_rms,
        tail.rel_bias,
        tail.max_rel,
    )
    assert stats.rms == float(np.sqrt(np.mean((my - py) ** 2)))
    assert stats.n == 3


def test_shape_residuals_matches_lem1_reconstruction_compare() -> None:
    """The original ``lem1_reconstruction`` script's ``compare`` is the same
    expression order for the five fields it shares with lem2's."""
    py, my = SHAPE_CASES["zero_reference_point"]
    diff = my - py
    rms = float(np.sqrt(np.mean(diff**2)))
    denom = np.where(np.abs(py) > 1e-9, py, np.nan)
    rel_bias = float(np.nanmean(diff / denom))
    k = float(np.dot(py, my) / np.dot(py, py)) if np.dot(py, py) > 0 else 1.0
    rms_scaled = float(np.sqrt(np.mean((my - k * py) ** 2)))
    shape_fraction = rms_scaled / rms if rms > 1e-12 else 0.0

    stats = shape_residuals(py, my)
    assert (stats.rms, stats.rel_bias, stats.scale_k, stats.shape_fraction, stats.n) == (
        rms,
        rel_bias,
        k,
        shape_fraction,
        len(py),
    )


def test_shape_residuals_length_mismatch_raises() -> None:
    with pytest.raises(ValueError, match="length"):
        shape_residuals([1.0, 2.0], [1.0])


def test_shape_residuals_is_frozen() -> None:
    stats = shape_residuals([1.0, 2.0], [1.0, 2.0])
    with pytest.raises(dataclasses.FrozenInstanceError):
        stats.rms = 0.0  # type: ignore[misc]


# ---- ratio_summary ---------------------------------------------------------------

OURS = np.array([1.05, 0.98, np.nan, 1.20, 0.91, 1.01, 1.33])
REF = np.array([1.00, 1.00, 1.00, 1.00, 1.00, 0.97, 1.10])


def test_ratio_summary_osmk_per_series_row_with_nan() -> None:
    """The original ``osmk_reconstruction`` script's per-series row: nan-aware
    median / RMS-log / max|r-1| over a ratio that carries NaN."""
    r21 = OURS / REF
    summary = ratio_summary(OURS, REF)
    assert summary.median == np.nanmedian(r21)
    assert summary.rms_log == np.sqrt(np.nanmean(np.log(r21) ** 2))
    assert summary.max_abs_rel == np.nanmax(np.abs(r21 - 1))
    assert summary.n == 6
    # The other fields are nan-aware too (a plain reduction would give NaN here).
    assert summary.mean_rel == np.nanmean(r21 - 1.0)
    assert summary.rms_rel == np.sqrt(np.nanmean((r21 - 1.0) ** 2))
    assert (summary.min_ratio, summary.max_ratio) == (np.nanmin(r21), np.nanmax(r21))


def test_ratio_summary_osmk_pooled_overall() -> None:
    """The original ``osmk_reconstruction`` script's ``all_ratios`` + overall line:
    NaN-stripped pool, then plain median / RMS-log / max|r-1|."""
    pooled: list[float] = []
    for ours, ref in ((OURS[:4], REF[:4]), (OURS[4:], REF[4:])):
        r = ours / ref
        pooled += list(r[~np.isnan(r)])
    a = np.array(pooled)

    summary = ratio_summary(a, np.ones_like(a))
    assert summary.median == np.median(a)
    assert summary.rms_log == np.sqrt(np.mean(np.log(a) ** 2))
    assert summary.max_abs_rel == np.max(np.abs(a - 1))


def test_ratio_summary_lem4_rel_vs_curve() -> None:
    """The original ``lem4_reconstruction`` script's ``rel_vs_curve``: drop zero
    references, then RMS and mean of ``r - 1``."""
    model_at = np.array([2.1, 3.0, 0.4, 5.5])
    ref = np.array([2.0, 0.0, 0.5, 5.0])
    rel = model_at[ref != 0] / ref[ref != 0] - 1.0

    with warnings.catch_warnings():
        warnings.simplefilter("error")  # the masked-out zero reference never divides
        summary = ratio_summary(model_at, ref, mask=ref != 0)
    assert summary.rms_rel == np.sqrt(np.mean(rel**2))
    assert summary.mean_rel == np.mean(rel)
    assert float(summary.rms_rel * 100.0) == float(np.sqrt(np.mean(rel**2)) * 100.0)
    assert summary.n == 3


def test_ratio_summary_lem1_rbe_ratio_line() -> None:
    """The original ``lem1_reconstruction`` script's RBE(1%) ratio line: min, max, RMS."""
    ours = np.array([1.2, 1.9, 2.4, 2.2])
    paper = np.array([1.25, 1.8, 2.5, 2.0])
    ratio = ours / paper

    summary = ratio_summary(ours, paper)
    assert (summary.min_ratio, summary.max_ratio) == (ratio.min(), ratio.max())
    assert summary.rms_rel == float(np.sqrt(np.mean((ratio - 1.0) ** 2)))


def test_ratio_summary_lem2_fig4_median_and_genome_bias() -> None:
    """The original ``lem2_cluster_effect`` script's ``fig4_logratio`` median ratio
    (mask = both positive) and the genome-size bias (mask = finite)."""
    ours = np.array([3.0, 0.0, 2e-3, 7.0, 1.0])
    ref_y = np.array([2.5, 1.0, 1e-3, 0.0, 1.2])
    m = (ref_y > 0) & (ours > 0)
    summary = ratio_summary(ours, ref_y, mask=m)
    assert summary.median == float(np.median(ours[m] / ref_y[m]))
    assert summary.n == int(m.sum())

    g_ratio = np.array([1.1, np.nan, 0.9, 1.3])
    g_paper = np.array([1.0, 1.0, np.nan, 1.2])
    ok = np.isfinite(g_ratio) & np.isfinite(g_paper)
    bias = ratio_summary(g_ratio, g_paper, mask=ok).mean_rel
    assert bias == float(np.mean(g_ratio[ok] / g_paper[ok] - 1))


def test_ratio_summary_smk_pooled_z_star_line() -> None:
    """The original ``smk_reconstruction`` script's pooled z* ratio line: median,
    ``sqrt((log r)^2).mean()`` and ``abs(r - 1).max()`` of the masked pool."""
    rng = np.random.default_rng(20261023)
    mine = rng.lognormal(0.0, 0.2, 40)
    paper = rng.lognormal(0.0, 0.2, 40)
    mask = (mine > 0.7) & (paper < 1.5)
    r = np.array(list(mine[mask] / paper[mask]))

    summary = ratio_summary(mine, paper, mask=mask)
    assert summary.n == len(r)
    assert summary.median == np.median(r)
    assert summary.rms_log == np.sqrt((np.log(r) ** 2).mean())
    assert summary.max_abs_rel == np.abs(r - 1).max()


def test_ratio_summary_imk_spread() -> None:
    """The original ``imk_microdosimetry_routes`` script's z*_1D spread: min / max."""
    values = np.array([1.2, 1.5, 1.1])
    reference = np.array([1.0, 1.4, 1.3])
    ratio = values / reference
    summary = ratio_summary(values, reference)
    assert (summary.min_ratio, summary.max_ratio) == (ratio.min(), ratio.max())


def test_ratio_summary_mean_and_rms_rel_bits_on_a_long_irregular_array() -> None:
    """``mean_rel`` is ``mean(r - 1)``, not ``mean(r) - 1`` (lem4 ``rel_vs_curve``,
    lem2 genome bias): the two round differently on most longer arrays."""
    rng = np.random.default_rng(20261023)
    o = rng.lognormal(0.0, 0.3, 50)
    r = rng.lognormal(0.0, 0.3, 50)
    m = r > 0.8
    rel = o[m] / r[m] - 1.0
    assert np.mean(rel) != np.mean(o[m] / r[m]) - 1.0  # the forms are distinguishable

    summary = ratio_summary(o, r, mask=m)
    assert summary.mean_rel == float(np.mean(rel))
    assert summary.rms_rel == float(np.sqrt(np.mean(rel**2)))
    assert summary.n == int(m.sum())


def test_ratio_summary_zero_and_negative_ratios_in_rms_log() -> None:
    """osmk's ``np.log(r)`` unmasked: a zero ratio makes ``rms_log`` inf, a negative
    one is skipped by ``rms_log`` only (its log is NaN) while ``n`` and every other
    field still count it."""
    with pytest.warns(RuntimeWarning):
        zero = ratio_summary([0.0, 1.1], [1.0, 1.0])
    assert zero.rms_log == math.inf and zero.n == 2 and zero.min_ratio == 0.0

    with pytest.warns(RuntimeWarning):
        neg = ratio_summary([-1.0, 2.0], [1.0, 1.0])
    assert neg.n == 2 and neg.min_ratio == -1.0
    assert neg.rms_log == float(np.sqrt(np.mean(np.log([2.0]) ** 2)))
    assert neg.mean_rel == float(np.mean([-2.0, 1.0]))


def test_ratio_summary_empty_is_all_nan_without_warnings() -> None:
    with warnings.catch_warnings():
        warnings.simplefilter("error")
        summary = ratio_summary([np.nan, 1.0], [1.0, 1.0], mask=np.array([True, False]))
    assert summary.n == 0
    for name in (
        "median",
        "mean_rel",
        "rms_rel",
        "rms_log",
        "max_abs_rel",
        "min_ratio",
        "max_ratio",
    ):
        assert math.isnan(getattr(summary, name)), name


def test_ratio_summary_validates_shapes() -> None:
    with pytest.raises(ValueError, match="length"):
        ratio_summary([1.0, 2.0], [1.0])
    with pytest.raises(ValueError, match="mask"):
        ratio_summary([1.0, 2.0], [1.0, 2.0], mask=np.array([True]))


def test_ratio_summary_is_frozen() -> None:
    summary = ratio_summary([1.0], [1.0])
    assert isinstance(summary, RatioSummary)
    with pytest.raises(dataclasses.FrozenInstanceError):
        summary.median = 0.0  # type: ignore[misc]
