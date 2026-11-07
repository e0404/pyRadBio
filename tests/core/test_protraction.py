"""The time axis: dose protraction and the Lea-Catcheside G factor.

Written before the implementation (TDD). The independent references used here
are (a) the two published closed forms -- continuous irradiation and equal
acute fractions -- written out explicitly below, and (b) a brute-force
numerical evaluation of the Lea-Catcheside double integral on a dense time
grid, which shares no algebra with the implementation.
"""

from __future__ import annotations

import math

import numpy as np
import pytest
from hypothesis import given, settings
from hypothesis import strategies as st

from pyradbio.core import lq
from pyradbio.core import protraction as pr

LN2 = math.log(2.0)


# --------------------------------------------------------------------------
# Independent references (published closed forms, re-derived here by hand)
# --------------------------------------------------------------------------
def _g_continuous_reference(duration_h: float, half_time_h: float) -> float:
    """``G = 2 (x - 1 + e^-x) / x**2`` with ``x = lambda*T`` (Dale 1985)."""
    x = LN2 / half_time_h * duration_h
    return 2.0 * (x - 1.0 + math.exp(-x)) / (x * x)


def _g_fractions_reference(n: int, interval_h: float, half_time_h: float) -> float:
    """``G = 1/n + (2/n^2) sum_k (n-k) phi^k`` for n equal acute fractions.

    Thames' incomplete-repair sum, evaluated as an explicit loop rather than in
    the closed form the implementation uses.
    """
    phi = math.exp(-LN2 / half_time_h * interval_h)
    total = sum((n - k) * phi**k for k in range(1, n))
    return 1.0 / n + 2.0 / (n * n) * total


def _g_brute_force(
    blocks: list[tuple[float, float, float]],
    half_time_h: float,
    *,
    nodes: int = 200001,
) -> float:
    """G from the definition, by dense-grid quadrature of the double integral.

    ``G = (2/D^2) int dt Ddot(t) int_{-inf}^{t} dt' Ddot(t') exp(-lam (t-t'))``.
    Blocks are ``(dose_Gy, start_h, duration_h)`` with finite duration, so the
    dose rate is a piecewise-constant function sampled on a uniform grid.
    """
    lam = LN2 / half_time_h
    t0 = min(b[1] for b in blocks)
    t1 = max(b[1] + b[2] for b in blocks)
    t = np.linspace(t0, t1, nodes)
    rate = np.zeros_like(t)
    for dose, start, duration in blocks:
        inside = (t >= start) & (t <= start + duration)
        rate[inside] += dose / duration
    step = float(t[1] - t[0])
    # inner(t) = int_{-inf}^{t} Ddot(t') e^{-lam (t - t')} dt', by cumulative
    # trapezoid on the same grid. The exp(+lam t) weight is pulled out and put
    # back so the running integral is a single cumulative sum.
    weighted = rate * np.exp(lam * (t - t0))
    cumulative = np.concatenate(([0.0], np.cumsum((weighted[:-1] + weighted[1:]) / 2.0 * step)))
    inner = np.exp(-lam * (t - t0)) * cumulative
    outer = float(np.trapezoid(rate * inner, t))
    dose_total = sum(b[0] for b in blocks)
    return 2.0 * outer / (dose_total * dose_total)


# --------------------------------------------------------------------------
# Exact limits
# --------------------------------------------------------------------------
@pytest.mark.parametrize("half_time_h", [0.1, 1.0, 24.0, math.inf])
def test_single_acute_dose_has_g_exactly_one(half_time_h: float) -> None:
    """A single instantaneous dose is the LQ reference case: G == 1."""
    schedule = pr.DoseSchedule.acute(2.0)
    assert pr.lea_catcheside_g(schedule, repair_half_time_h=half_time_h) == 1.0


def test_no_repair_limit_is_one() -> None:
    """With an infinite repair half-time no sublethal damage is lost: G -> 1."""
    schedule = pr.DoseSchedule.fractionated(2.0, n_fractions=30, interval_h=24.0)
    assert pr.lea_catcheside_g(schedule, repair_half_time_h=math.inf) == pytest.approx(1.0)


def test_complete_repair_limit_is_one_over_n() -> None:
    """Fractions that repair fully between them add effects: G -> 1/n."""
    schedule = pr.DoseSchedule.fractionated(2.0, n_fractions=5, interval_h=24.0)
    g = pr.lea_catcheside_g(schedule, repair_half_time_h=1e-6)
    assert g == pytest.approx(0.2, rel=1e-12)


def test_continuous_irradiation_matches_the_published_closed_form() -> None:
    for duration_h in (0.5, 2.0, 10.0, 100.0):
        schedule = pr.DoseSchedule.continuous(10.0, duration_h=duration_h)
        got = pr.lea_catcheside_g(schedule, repair_half_time_h=1.5)
        assert got == pytest.approx(_g_continuous_reference(duration_h, 1.5), rel=1e-12)


def test_continuous_irradiation_tends_to_two_over_lambda_t() -> None:
    """Very slow delivery: G -> 2/(lambda T), the classic low-dose-rate limit."""
    duration_h = 5000.0
    schedule = pr.DoseSchedule.continuous(10.0, duration_h=duration_h)
    g = pr.lea_catcheside_g(schedule, repair_half_time_h=1.0)
    assert g == pytest.approx(2.0 / (LN2 * duration_h), rel=1e-3)


def test_fractionated_matches_the_incomplete_repair_sum() -> None:
    for n, interval_h, half_time_h in ((2, 6.0, 1.5), (5, 4.0, 2.0), (30, 24.0, 0.5)):
        schedule = pr.DoseSchedule.fractionated(2.0, n_fractions=n, interval_h=interval_h)
        got = pr.lea_catcheside_g(schedule, repair_half_time_h=half_time_h)
        assert got == pytest.approx(_g_fractions_reference(n, interval_h, half_time_h), rel=1e-12)


# --------------------------------------------------------------------------
# The general schedule against the definition
# --------------------------------------------------------------------------
def test_general_schedule_matches_the_double_integral() -> None:
    """The anchor test: arbitrary blocks vs brute-force quadrature of the definition."""
    blocks = [(2.0, 0.0, 0.25), (1.0, 3.0, 1.5), (3.5, 8.0, 0.5)]
    schedule = pr.DoseSchedule(
        tuple(pr.DoseBlock(dose_Gy=d, start_h=s, duration_h=t) for d, s, t in blocks)
    )
    got = pr.lea_catcheside_g(schedule, repair_half_time_h=1.0)
    assert got == pytest.approx(_g_brute_force(blocks, 1.0), rel=2e-4)


def test_splitting_a_block_in_half_does_not_change_g() -> None:
    """A continuous block is the same irradiation as two contiguous half-blocks."""
    whole = pr.DoseSchedule.continuous(6.0, duration_h=4.0)
    split = pr.DoseSchedule(
        (
            pr.DoseBlock(dose_Gy=3.0, start_h=0.0, duration_h=2.0),
            pr.DoseBlock(dose_Gy=3.0, start_h=2.0, duration_h=2.0),
        )
    )
    half_time_h = 1.3
    assert pr.lea_catcheside_g(split, repair_half_time_h=half_time_h) == pytest.approx(
        pr.lea_catcheside_g(whole, repair_half_time_h=half_time_h), rel=1e-12
    )


def test_g_is_invariant_under_time_translation() -> None:
    a = pr.DoseSchedule.fractionated(2.0, n_fractions=4, interval_h=6.0)
    b = pr.DoseSchedule(
        tuple(
            pr.DoseBlock(
                dose_Gy=blk.dose_Gy, start_h=blk.start_h + 137.0, duration_h=blk.duration_h
            )
            for blk in a.blocks
        )
    )
    assert pr.lea_catcheside_g(b, repair_half_time_h=2.0) == pytest.approx(
        pr.lea_catcheside_g(a, repair_half_time_h=2.0), rel=1e-12
    )


def test_g_depends_on_the_shape_not_the_size_of_the_dose() -> None:
    """Scaling every block by the same factor leaves G unchanged."""
    a = pr.DoseSchedule.fractionated(2.0, n_fractions=3, interval_h=8.0)
    b = pr.DoseSchedule.fractionated(20.0, n_fractions=3, interval_h=8.0)
    assert pr.lea_catcheside_g(b, repair_half_time_h=3.0) == pytest.approx(
        pr.lea_catcheside_g(a, repair_half_time_h=3.0), rel=1e-12
    )


def test_protraction_never_increases_the_quadratic_term() -> None:
    """G in (0, 1]: protraction can only spare, never sensitise."""
    for n in (1, 2, 10):
        for interval_h in (0.5, 24.0):
            schedule = pr.DoseSchedule.fractionated(2.0, n_fractions=n, interval_h=interval_h)
            g = pr.lea_catcheside_g(schedule, repair_half_time_h=1.0)
            assert 0.0 < g <= 1.0


@settings(max_examples=50, deadline=None)
@given(
    n=st.integers(min_value=1, max_value=12),
    interval_h=st.floats(min_value=0.01, max_value=48.0),
    half_time_h=st.floats(min_value=0.01, max_value=48.0),
)
def test_g_in_unit_interval_property(n: int, interval_h: float, half_time_h: float) -> None:
    schedule = pr.DoseSchedule.fractionated(2.0, n_fractions=n, interval_h=interval_h)
    g = pr.lea_catcheside_g(schedule, repair_half_time_h=half_time_h)
    assert 1.0 / n - 1e-12 <= g <= 1.0 + 1e-12


def test_g_is_monotone_in_the_interval() -> None:
    """Longer gaps mean more repair, so G falls monotonically toward 1/n."""
    previous = 1.1
    for interval_h in (0.1, 0.5, 1.0, 4.0, 12.0, 48.0):
        schedule = pr.DoseSchedule.fractionated(2.0, n_fractions=6, interval_h=interval_h)
        g = pr.lea_catcheside_g(schedule, repair_half_time_h=1.0)
        assert g < previous
        previous = g


# --------------------------------------------------------------------------
# Rate <-> half-time (models that state a repair rate, e.g. IMK's (a+c))
# --------------------------------------------------------------------------
@pytest.mark.parametrize("half_time_h", [0.1, 0.5, 1.0, 2.5, 24.0])
def test_rate_and_half_time_round_trip(half_time_h: float) -> None:
    rate = pr.repair_rate_per_h(half_time_h)
    assert pr.repair_half_time_h_from_rate(rate) == pytest.approx(half_time_h, rel=1e-12)
    assert pr.repair_rate_per_h(pr.repair_half_time_h_from_rate(rate)) == pytest.approx(
        rate, rel=1e-12
    )


def test_half_time_from_rate_is_ln2_over_the_rate() -> None:
    assert pr.repair_half_time_h_from_rate(2.19) == pytest.approx(LN2 / 2.19, rel=1e-12)


def test_zero_rate_is_the_no_repair_limit() -> None:
    """Both ends of the no-repair limit agree: rate 0 <-> infinite half-time."""
    assert pr.repair_half_time_h_from_rate(0.0) == math.inf
    assert pr.repair_rate_per_h(math.inf) == 0.0
    schedule = pr.DoseSchedule.continuous(4.0, duration_h=48.0)
    assert pr.lea_catcheside_g(
        schedule, repair_half_time_h=pr.repair_half_time_h_from_rate(0.0)
    ) == pytest.approx(1.0)


@pytest.mark.parametrize("bad", [-1.0, -1e-12, math.nan])
def test_bad_repair_rate_rejected(bad: float) -> None:
    with pytest.raises(ValueError, match="repair_rate_per_h"):
        pr.repair_half_time_h_from_rate(bad)


def test_a_rate_stated_schedule_matches_the_half_time_route() -> None:
    """A model carrying ``(a+c)`` gets the same ``G`` as one carrying ``T_half``."""
    rate = 2.19  # IMK's HSG (a+c), Matsuya et al. 2025 table 1
    schedule = pr.DoseSchedule.continuous(4.0, duration_h=2.0)
    x = rate * 2.0
    expected = 2.0 * (x - 1.0 + math.exp(-x)) / x**2
    assert pr.lea_catcheside_g(
        schedule, repair_half_time_h=pr.repair_half_time_h_from_rate(rate)
    ) == pytest.approx(expected, rel=1e-12)


# --------------------------------------------------------------------------
# Numerical stability
# --------------------------------------------------------------------------
def test_tiny_repair_rate_does_not_lose_precision() -> None:
    """x = lambda*T ~ 1e-9 is where the naive (x - 1 + e^-x)/x^2 cancels away."""
    schedule = pr.DoseSchedule.continuous(2.0, duration_h=1e-3)
    g = pr.lea_catcheside_g(schedule, repair_half_time_h=1e6)
    assert g == pytest.approx(1.0, abs=1e-9)
    assert g <= 1.0


def test_dose_rate_constructor_matches_the_dose_constructor() -> None:
    a = pr.DoseSchedule.continuous(6.0, duration_h=3.0)
    b = pr.DoseSchedule.at_dose_rate(2.0, duration_h=3.0)
    assert b.total_dose_Gy == pytest.approx(a.total_dose_Gy)
    assert pr.lea_catcheside_g(b, repair_half_time_h=1.0) == pytest.approx(
        pr.lea_catcheside_g(a, repair_half_time_h=1.0)
    )


# --------------------------------------------------------------------------
# Validation
# --------------------------------------------------------------------------
def test_total_dose_is_the_sum_of_the_blocks() -> None:
    schedule = pr.DoseSchedule.fractionated(2.5, n_fractions=4, interval_h=24.0)
    assert schedule.total_dose_Gy == pytest.approx(10.0)


@pytest.mark.parametrize(
    ("dose", "start", "duration"),
    [(-1.0, 0.0, 0.0), (1.0, 0.0, -1.0), (math.nan, 0.0, 0.0)],
)
def test_bad_block_rejected(dose: float, start: float, duration: float) -> None:
    with pytest.raises(ValueError):
        pr.DoseBlock(dose_Gy=dose, start_h=start, duration_h=duration)


def test_overlapping_blocks_rejected() -> None:
    with pytest.raises(ValueError):
        pr.DoseSchedule(
            (
                pr.DoseBlock(dose_Gy=1.0, start_h=0.0, duration_h=2.0),
                pr.DoseBlock(dose_Gy=1.0, start_h=1.0, duration_h=2.0),
            )
        )


def test_empty_schedule_rejected() -> None:
    with pytest.raises(ValueError):
        pr.DoseSchedule(())


def test_zero_dose_schedule_rejected() -> None:
    schedule = pr.DoseSchedule((pr.DoseBlock(dose_Gy=0.0, start_h=0.0, duration_h=0.0),))
    with pytest.raises(ValueError):
        pr.lea_catcheside_g(schedule, repair_half_time_h=1.0)


@pytest.mark.parametrize("bad", [0.0, -1.0, math.nan])
def test_bad_repair_half_time_rejected(bad: float) -> None:
    schedule = pr.DoseSchedule.acute(2.0)
    with pytest.raises(ValueError):
        pr.lea_catcheside_g(schedule, repair_half_time_h=bad)


@pytest.mark.parametrize("n", [0, -3])
def test_bad_fraction_number_rejected(n: int) -> None:
    with pytest.raises(ValueError):
        pr.DoseSchedule.fractionated(2.0, n_fractions=n, interval_h=24.0)


# --------------------------------------------------------------------------
# End to end: a schedule -> G -> the LQ layer
# --------------------------------------------------------------------------
def test_schedule_g_feeds_the_lq_layer() -> None:
    """A real course, and the two bounds it must sit between.

    30 x 2 Gy daily with a 1.5 h repair half-time: 24 h is 16 half-times, so
    repair is essentially complete and G lands just above 1/30. Survival must
    then sit between the acute bound (all 60 Gy at once) and the complete-repair
    bound (G = 1/30 exactly), and BED must do the same.
    """
    schedule = pr.DoseSchedule.fractionated(2.0, n_fractions=30, interval_h=24.0)
    g = pr.lea_catcheside_g(schedule, repair_half_time_h=1.5)
    assert 1.0 / 30.0 <= g < 1.0 / 30.0 + 1e-3

    dose = schedule.total_dose_Gy
    alpha, beta = 0.3, 0.03
    acute = lq.survival(dose, alpha, beta, g=1.0)
    complete_repair = lq.survival(dose, alpha, beta, g=1.0 / 30.0)
    assert acute < lq.survival(dose, alpha, beta, g=g) <= complete_repair

    ratio = alpha / beta
    assert (
        lq.bed(dose, alpha_beta_ratio_Gy=ratio, g=1.0 / 30.0)
        <= lq.bed(dose, alpha_beta_ratio_Gy=ratio, g=g)
        < lq.bed(dose, alpha_beta_ratio_Gy=ratio, g=1.0)
    )


def test_a_two_hour_gap_within_a_day_recovers_some_of_the_shoulder() -> None:
    """Two 1 Gy half-fractions 2 h apart spare less than a full day's gap."""
    two_hours = pr.DoseSchedule.fractionated(1.0, n_fractions=2, interval_h=2.0)
    one_day = pr.DoseSchedule.fractionated(1.0, n_fractions=2, interval_h=24.0)
    together = pr.DoseSchedule.acute(2.0)
    kwargs = {"repair_half_time_h": 1.5}
    assert (
        pr.lea_catcheside_g(one_day, **kwargs)
        < pr.lea_catcheside_g(two_hours, **kwargs)
        < pr.lea_catcheside_g(together, **kwargs)
    )


def test_non_finite_start_time_rejected() -> None:
    with pytest.raises(ValueError):
        pr.DoseBlock(dose_Gy=1.0, start_h=math.inf, duration_h=0.0)


def test_schedule_duration_spans_first_start_to_last_end() -> None:
    schedule = pr.DoseSchedule.fractionated(
        2.0, n_fractions=3, interval_h=24.0, fraction_duration_h=0.5
    )
    assert schedule.duration_h == pytest.approx(48.5)


@pytest.mark.parametrize("bad", [-1.0, math.inf])
def test_bad_dose_rate_rejected(bad: float) -> None:
    with pytest.raises(ValueError):
        pr.DoseSchedule.at_dose_rate(bad, duration_h=1.0)


def test_negative_interval_rejected() -> None:
    with pytest.raises(ValueError):
        pr.DoseSchedule.fractionated(2.0, n_fractions=3, interval_h=-1.0)


def test_interval_shorter_than_the_fraction_is_rejected() -> None:
    """interval_h is start-to-start, so a shorter one would overlap the fractions."""
    with pytest.raises(ValueError):
        pr.DoseSchedule.fractionated(2.0, n_fractions=3, interval_h=0.2, fraction_duration_h=0.5)


def test_finite_fraction_duration_lowers_g_slightly() -> None:
    """Delivering each fraction over 20 min rather than instantly spares a little."""
    instant = pr.DoseSchedule.fractionated(2.0, n_fractions=5, interval_h=24.0)
    spread = pr.DoseSchedule.fractionated(
        2.0, n_fractions=5, interval_h=24.0, fraction_duration_h=1.0 / 3.0
    )
    kwargs = {"repair_half_time_h": 1.5}
    assert pr.lea_catcheside_g(spread, **kwargs) < pr.lea_catcheside_g(instant, **kwargs)


# --------------------------------------------------------------------------
# Golden: Dale 1985, the one printed (undigitized) reference for the time axis
# --------------------------------------------------------------------------
#: Dale 1985 table II -- Mitchell's (1960) data for the healing of small skin
#: tumours: isoeffective total dose vs overall treatment time, delivered
#: continuously. Printed in the paper, so no digitization is involved.
_DALE_TABLE_II_TIME_H = (1.67, 10.0, 24.0, 48.5, 70.3, 169.0, 336.0, 600.0)
_DALE_TABLE_II_DOSE_GY = (18.0, 22.5, 28.8, 33.0, 36.0, 45.0, 48.0, 49.5)
#: The dose-rate column the paper prints alongside them, used only as a
#: transcription check (it is just dose/time).
_DALE_TABLE_II_RATE_GY_H = (10.78, 2.25, 1.20, 0.68, 0.51, 0.27, 0.14, 0.08)


def _dale_reciprocal_dose_fit(mu_per_h: float) -> tuple[float, float, float]:
    """Dale's fig. 1 analysis: regress ``1/D`` on ``K*R`` (= ``G*D`` here).

    His eq. (15) is ``1/D = 1/ERD + K*R / (ERD * (alpha/beta))``, so the fit
    gives ``ERD`` from the intercept and ``alpha/beta`` from intercept/slope.
    Returns ``(rms_residual, ERD_Gy, alpha_beta_Gy)``.
    """
    half_time_h = math.log(2.0) / mu_per_h
    g_d = [
        pr.lea_catcheside_g(
            pr.DoseSchedule.continuous(dose, duration_h=time_h),
            repair_half_time_h=half_time_h,
        )
        * dose
        for dose, time_h in zip(_DALE_TABLE_II_DOSE_GY, _DALE_TABLE_II_TIME_H, strict=True)
    ]
    reciprocal = [1.0 / dose for dose in _DALE_TABLE_II_DOSE_GY]
    slope, intercept = np.polyfit(np.array(g_d), np.array(reciprocal), 1)
    residual = np.array(reciprocal) - (slope * np.array(g_d) + intercept)
    return float(np.sqrt(np.mean(residual**2))), 1.0 / intercept, intercept / slope


def test_dale_table_ii_transcription() -> None:
    """The printed dose-rate column must equal dose/time, or we typed it wrong."""
    for dose, time_h, rate in zip(
        _DALE_TABLE_II_DOSE_GY, _DALE_TABLE_II_TIME_H, _DALE_TABLE_II_RATE_GY_H, strict=True
    ):
        assert dose / time_h == pytest.approx(rate, rel=0.04)


@pytest.mark.golden
def test_golden_dale1985_selects_the_papers_repair_constant() -> None:
    """Dale 1985 fig. 1: mu = 0.32/h linearises 1/D vs K*R; 0.69 and 0.17 do not.

    Reference: Dale, R. G. (1985) Br. J. Radiol. 58:515-528,
    doi:10.1259/0007-1285-58-690-515, table II and fig. 1. The paper reports
    that of the three repair constants tried, only the middle one gives "a good
    straight-line fit to the data (line B)", while "the smallest and largest mu
    values examined both impart a distinct non-linear form" (curves A and C).
    """
    rms = {mu: _dale_reciprocal_dose_fit(mu)[0] for mu in (0.69, 0.32, 0.17)}
    assert min(rms, key=lambda mu: rms[mu]) == 0.32
    # and it is not a marginal win -- both neighbours are clearly more curved
    assert rms[0.69] > 2.5 * rms[0.32]
    assert rms[0.17] > 2.0 * rms[0.32]


@pytest.mark.golden
def test_golden_dale1985_erd_and_alpha_beta() -> None:
    """Dale 1985: the fit at mu = 0.32/h gives ERD = 52.6 Gy, alpha/beta = 7.6 Gy.

    Reference: Dale 1985, doi:10.1259/0007-1285-58-690-515, p. 519. ERD
    ("extrapolated response dose") is the quantity now called BED, so this is a
    published-number check on the whole chain: continuous-irradiation ``G`` ->
    ``bed`` -> its inversion.

    Tolerance is 5%: the paper's own value comes from "the best eye-line fit to
    the data" (its words), while this reproduces it by least squares, so exact
    agreement is not available and a tighter bound would be pinning our
    regression rather than the paper.
    """
    _, erd_Gy, alpha_beta_Gy = _dale_reciprocal_dose_fit(0.32)
    assert erd_Gy == pytest.approx(52.6, rel=0.05)
    assert alpha_beta_Gy == pytest.approx(7.6, rel=0.05)


@pytest.mark.golden
def test_golden_dale1985_equivalent_single_acute_dose() -> None:
    """Dale 1985 p. 519: the same effect as a single acute dose of 16.6 Gy.

    Reference: Dale 1985, doi:10.1259/0007-1285-58-690-515 -- "the equivalent
    single acute dose may be calculated more exactly to be 16.6 Gy (using eqns.
    3 and 5)". Inverting ``bed(D, alpha/beta, g=1) = ERD`` must reproduce it.
    """
    _, erd_Gy, alpha_beta_Gy = _dale_reciprocal_dose_fit(0.32)
    acute_Gy = (-1.0 + math.sqrt(1.0 + 4.0 * erd_Gy / alpha_beta_Gy)) / (2.0 / alpha_beta_Gy)
    assert acute_Gy == pytest.approx(16.6, rel=0.05)
    assert lq.bed(acute_Gy, alpha_beta_ratio_Gy=alpha_beta_Gy) == pytest.approx(erd_Gy, rel=1e-9)
