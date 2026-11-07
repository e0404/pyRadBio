"""Benchmark metrics: known-value and property checks."""

from __future__ import annotations

import math

import pytest

from pyradbio.benchmark import metrics


class TestMeanLogError:
    def test_perfect_match_is_zero(self) -> None:
        assert metrics.mean_log_error([0.3, 1.2, 5.0], [0.3, 1.2, 5.0]) == pytest.approx(0.0)

    def test_factor_of_ten_over_prediction(self) -> None:
        # log10(10/1) = 1 for every point.
        assert metrics.mean_log_error([10.0, 100.0], [1.0, 10.0]) == pytest.approx(1.0)

    def test_signed_bias_cancels(self) -> None:
        # +1 decade then -1 decade averages to zero (bias), unlike the abs metric.
        assert metrics.mean_log_error([100.0, 1.0], [10.0, 10.0]) == pytest.approx(0.0)

    def test_under_prediction_is_negative(self) -> None:
        assert metrics.mean_log_error([1.0], [10.0]) == pytest.approx(-1.0)


class TestMeanAbsoluteLogError:
    def test_perfect_match_is_zero(self) -> None:
        assert metrics.mean_absolute_log_error([0.5, 2.0], [0.5, 2.0]) == pytest.approx(0.0)

    def test_does_not_cancel(self) -> None:
        # Same inputs that cancel for the signed metric give 1.0 unsigned.
        assert metrics.mean_absolute_log_error([100.0, 1.0], [10.0, 10.0]) == pytest.approx(1.0)

    def test_matches_abs_of_single_point(self) -> None:
        assert metrics.mean_absolute_log_error([1.0], [10.0]) == pytest.approx(1.0)


class TestSmape:
    def test_perfect_match_is_zero(self) -> None:
        assert metrics.smape([1.0, 1.0], [1.0, 1.0]) == pytest.approx(0.0)

    def test_known_value(self) -> None:
        # |2-1| / (2+1) = 1/3.
        assert metrics.smape([2.0], [1.0]) == pytest.approx(1.0 / 3.0)

    def test_zero_vs_positive_is_one(self) -> None:
        assert metrics.smape([0.0], [5.0]) == pytest.approx(1.0)

    def test_both_zero_contributes_no_error(self) -> None:
        assert metrics.smape([0.0, 2.0], [0.0, 2.0]) == pytest.approx(0.0)

    def test_symmetric_in_arguments(self) -> None:
        assert metrics.smape([0.3], [1.7]) == pytest.approx(metrics.smape([1.7], [0.3]))

    def test_bounded_in_unit_interval(self) -> None:
        val = metrics.smape([0.01, 50.0, 3.0], [40.0, 0.02, 3.1])
        assert 0.0 <= val <= 1.0


@pytest.mark.parametrize(
    "fn",
    [metrics.mean_log_error, metrics.mean_absolute_log_error, metrics.smape],
)
class TestValidation:
    def test_empty_raises(self, fn) -> None:  # type: ignore[no-untyped-def]
        with pytest.raises(ValueError):
            fn([], [])

    def test_length_mismatch_raises(self, fn) -> None:  # type: ignore[no-untyped-def]
        with pytest.raises(ValueError):
            fn([1.0, 2.0], [1.0])


@pytest.mark.parametrize("fn", [metrics.mean_log_error, metrics.mean_absolute_log_error])
def test_log_metrics_reject_non_positive(fn) -> None:  # type: ignore[no-untyped-def]
    with pytest.raises(ValueError):
        fn([1.0, -0.5], [1.0, 2.0])
    with pytest.raises(ValueError):
        fn([1.0, 2.0], [0.0, 2.0])


def test_mean_log_error_is_log_of_geometric_mean_ratio() -> None:
    """MLE equals log10 of the geometric-mean ratio (definition cross-check)."""
    pred = [0.4, 1.1, 3.7]
    meas = [0.5, 0.9, 2.2]
    ratios = [p / m for p, m in zip(pred, meas, strict=True)]
    geo_mean_ratio = math.prod(ratios) ** (1.0 / len(ratios))
    assert metrics.mean_log_error(pred, meas) == pytest.approx(math.log10(geo_mean_ratio))
