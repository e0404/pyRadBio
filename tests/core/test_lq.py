"""LQ engine."""

from __future__ import annotations

import math

import pytest
from hypothesis import given
from hypothesis import strategies as st

from pyradbio.core import lq


def test_survival_beta_zero_is_exponential() -> None:
    assert lq.survival(2.0, alpha=0.3, beta=0.0) == pytest.approx(math.exp(-0.6))


def test_dose_for_survival_roundtrip() -> None:
    alpha, beta, dose = 0.3, 0.05, 2.5
    s = lq.survival(dose, alpha, beta)
    assert lq.dose_for_survival(s, alpha, beta) == pytest.approx(dose, rel=1e-9)


def test_survival_monotone_decreasing() -> None:
    a, b = 0.3, 0.05
    assert lq.survival(1.0, a, b) > lq.survival(2.0, a, b) > lq.survival(4.0, a, b)


# Ranges kept modest so log-kill stays well below the exp() underflow limit,
# i.e. S stays strictly positive in floating point (mathematically S in (0, 1]).
_doses = st.floats(min_value=0.0, max_value=10.0, allow_nan=False, allow_infinity=False)
_alphas = st.floats(min_value=0.0, max_value=2.0, allow_nan=False, allow_infinity=False)
_betas = st.floats(min_value=0.0, max_value=0.5, allow_nan=False, allow_infinity=False)


@given(dose=_doses, alpha=_alphas, beta=_betas)
def test_survival_in_unit_interval(dose: float, alpha: float, beta: float) -> None:
    s = lq.survival(dose, alpha, beta)
    assert 0.0 < s <= 1.0


@given(
    d1=_doses,
    d2=_doses,
    alpha=st.floats(min_value=1e-3, max_value=2.0),
    beta=_betas,
)
def test_survival_monotone_property(d1: float, d2: float, alpha: float, beta: float) -> None:
    lo, hi = sorted((d1, d2))
    assert lq.survival(lo, alpha, beta) >= lq.survival(hi, alpha, beta)


def test_log_kill_matches_survival() -> None:
    assert lq.log_kill(2.0, 0.3, 0.05) == pytest.approx(-math.log(lq.survival(2.0, 0.3, 0.05)))


def test_negative_dose_rejected() -> None:
    with pytest.raises(ValueError):
        lq.survival(-1.0, 0.3, 0.05)


@pytest.mark.parametrize("bad", [0.0, -0.1, 1.1])
def test_dose_for_survival_rejects_out_of_range(bad: float) -> None:
    with pytest.raises(ValueError):
        lq.dose_for_survival(bad, 0.3, 0.05)


def test_dose_for_survival_both_zero_rejected() -> None:
    with pytest.raises(ValueError):
        lq.dose_for_survival(0.5, 0.0, 0.0)


@pytest.mark.parametrize(("alpha", "beta"), [(-0.1, 0.05), (0.3, -0.05)])
def test_dose_for_survival_rejects_negative_params(alpha: float, beta: float) -> None:
    with pytest.raises(ValueError):
        lq.dose_for_survival(0.5, alpha, beta)


def test_lq_parameters_validation() -> None:
    lq.LQParameters(0.3, 0.05)  # valid
    with pytest.raises(ValueError):
        lq.LQParameters(-0.1, 0.05)
    with pytest.raises(ValueError):
        lq.LQParameters(0.3, -0.05)


def test_rbe_self_reference_is_unity() -> None:
    ref = lq.LQParameters(0.3, 0.05)
    assert lq.rbe(0.1, 0.3, 0.05, reference=ref) == pytest.approx(1.0)


def test_rbe_higher_alpha_more_effective() -> None:
    """An ion with larger alpha than the photon reference has RBE > 1."""
    ref = lq.LQParameters(0.1, 0.05)
    assert lq.rbe(0.1, 0.5, 0.05, reference=ref) > 1.0


@pytest.mark.golden
def test_golden_hsg_photon_d10() -> None:
    """HSG photon LQ (alpha_x=0.313, beta_x=0.0615) -> D10.

    Reference: Elsasser & Scholz 2007, Fig. 11 legend (doi:10.1667/RR0467.1).
    Solve 0.313*D + 0.0615*D^2 = -ln(0.1) = 2.302585. Expected D10 ~ 4.04 Gy.
    """
    d10 = lq.dose_for_survival(0.10, alpha=0.313, beta=0.0615)
    assert d10 == pytest.approx(4.04, abs=0.05)


# --------------------------------------------------------------------------
# The time axis: protraction (G), BED and EQD2
# --------------------------------------------------------------------------
def test_g_scales_only_the_quadratic_term() -> None:
    alpha, beta, dose, g = 0.3, 0.05, 4.0, 0.25
    assert lq.log_kill(dose, alpha, beta, g=g) == pytest.approx(
        alpha * dose + g * beta * dose * dose
    )


def test_g_of_one_is_the_acute_default() -> None:
    assert lq.log_kill(3.0, 0.3, 0.05, g=1.0) == lq.log_kill(3.0, 0.3, 0.05)
    assert lq.survival(3.0, 0.3, 0.05, g=1.0) == lq.survival(3.0, 0.3, 0.05)
    assert lq.dose_for_survival(0.2, 0.3, 0.05, g=1.0) == lq.dose_for_survival(0.2, 0.3, 0.05)


def test_protraction_spares() -> None:
    """Sparing at fixed total dose: more protraction, higher survival."""
    assert lq.survival(6.0, 0.3, 0.05, g=0.2) > lq.survival(6.0, 0.3, 0.05, g=1.0)


def test_dose_for_survival_roundtrip_with_g() -> None:
    alpha, beta, dose, g = 0.3, 0.05, 2.5, 0.4
    s = lq.survival(dose, alpha, beta, g=g)
    assert lq.dose_for_survival(s, alpha, beta, g=g) == pytest.approx(dose, rel=1e-9)


def test_g_of_zero_is_the_pure_exponential_limit() -> None:
    """Infinitely protracted: only the linear term survives."""
    assert lq.survival(5.0, 0.3, 0.05, g=0.0) == pytest.approx(math.exp(-1.5))
    assert lq.dose_for_survival(math.exp(-1.5), 0.3, 0.05, g=0.0) == pytest.approx(5.0)


@pytest.mark.parametrize("bad", [-0.1, 1.5])
def test_bad_g_rejected(bad: float) -> None:
    with pytest.raises(ValueError):
        lq.log_kill(2.0, 0.3, 0.05, g=bad)


def test_bed_is_the_log_kill_in_units_of_alpha() -> None:
    """BED = E/alpha by definition, so it must agree with log_kill/alpha."""
    alpha, beta, dose, g = 0.3, 0.05, 4.0, 0.35
    got = lq.bed(dose, alpha_beta_ratio_Gy=alpha / beta, g=g)
    assert got == pytest.approx(lq.log_kill(dose, alpha, beta, g=g) / alpha)


def test_bed_of_a_conventional_schedule() -> None:
    """60 Gy in 30 x 2 Gy with full inter-fraction repair (G = 1/30).

    BED = D (1 + G D / (alpha/beta)) = 60 (1 + 2/10) = 72 Gy10; the same
    schedule at alpha/beta = 3 gives 60 (1 + 2/3) = 100 Gy3. Arithmetic from
    the definition (Jones et al. 2001, doi:10.1053/clon.2001.9221), not a
    value read off a table.
    """
    assert lq.bed(60.0, alpha_beta_ratio_Gy=10.0, g=1.0 / 30.0) == pytest.approx(72.0)
    assert lq.bed(60.0, alpha_beta_ratio_Gy=3.0, g=1.0 / 30.0) == pytest.approx(100.0)


def test_bed_of_a_single_acute_dose() -> None:
    assert lq.bed(10.0, alpha_beta_ratio_Gy=10.0) == pytest.approx(20.0)


def test_bed_increases_with_dose() -> None:
    kw = {"alpha_beta_ratio_Gy": 3.0, "g": 0.1}
    assert lq.bed(1.0, **kw) < lq.bed(2.0, **kw) < lq.bed(10.0, **kw)


def test_eqd2_of_two_gray_fractions_is_the_total_dose() -> None:
    """2 Gy fractions with full repair are the EQD2 reference: EQD2 == D."""
    for alpha_beta in (2.0, 3.0, 10.0):
        n = 30
        assert lq.eqd2(2.0 * n, alpha_beta_ratio_Gy=alpha_beta, g=1.0 / n) == pytest.approx(2.0 * n)


def test_eqd2_of_a_hypofractionated_schedule() -> None:
    """20 Gy in 5 x 4 Gy at alpha/beta = 10: BED 28 Gy10 -> EQD2 28/1.2."""
    got = lq.eqd2(20.0, alpha_beta_ratio_Gy=10.0, g=1.0 / 5.0)
    assert got == pytest.approx(28.0 / 1.2)


def test_eqd2_hypofractionation_costs_more_in_late_responding_tissue() -> None:
    """The Withers isoeffect statement, on two schedules matched in the tumour.

    40 Gy in 5 x 8 Gy and 60 Gy in 30 x 2 Gy are both EQD2 = 60 Gy at
    alpha/beta = 10 (solve ``D + D^2/50 = 72`` for the 5-fraction course). In
    late-responding tissue at alpha/beta = 3 the same two schedules are
    88 Gy and 60 Gy: matching the tumour effect with fewer, larger fractions
    costs 28 Gy of late-tissue EQD2. That divergence between the two ratios is
    the whole content of the fraction-size isoeffect.

    Reference: Withers, Thames & Peters 1983, doi:10.1016/s0167-8140(83)80021-8.
    All four numbers are arithmetic from the BED definition, not read off a
    published table.
    """
    hypo = {"g": 1.0 / 5.0}  # 5 fractions of 8 Gy
    conventional = {"g": 1.0 / 30.0}  # 30 fractions of 2 Gy
    tumour_hypo = lq.eqd2(40.0, alpha_beta_ratio_Gy=10.0, **hypo)
    tumour_conv = lq.eqd2(60.0, alpha_beta_ratio_Gy=10.0, **conventional)
    assert tumour_hypo == pytest.approx(60.0)
    assert tumour_conv == pytest.approx(60.0)

    late_hypo = lq.eqd2(40.0, alpha_beta_ratio_Gy=3.0, **hypo)
    late_conv = lq.eqd2(60.0, alpha_beta_ratio_Gy=3.0, **conventional)
    assert late_hypo == pytest.approx(88.0)
    assert late_conv == pytest.approx(60.0)
    assert late_hypo > late_conv


def test_eqd2_accepts_another_reference_fraction_size() -> None:
    """EQD_d with d = 2 Gy is EQD2; with d equal to the schedule's own d it is D."""
    got = lq.eqd2(20.0, alpha_beta_ratio_Gy=10.0, g=1.0 / 5.0, reference_fraction_Gy=4.0)
    assert got == pytest.approx(20.0)


@pytest.mark.parametrize("bad", [0.0, -3.0])
def test_bed_rejects_bad_alpha_beta_ratio(bad: float) -> None:
    with pytest.raises(ValueError):
        lq.bed(10.0, alpha_beta_ratio_Gy=bad)


def test_rbe_rises_with_fractionation_when_the_reference_has_the_bigger_shoulder() -> None:
    """Fractionating both arms *raises* the ion RBE -- the reference loses more.

    G multiplies beta on both arms, so it spares whichever radiation carries
    more of its effect in the quadratic term. A photon reference with
    alpha/beta = 2 has a much bigger shoulder to lose than an ion with
    alpha/beta = 10, so the ion's advantage grows: the reduced fractionation
    effect of high-LET beams. Quoting a fractionated ion RBE against an *acute*
    photon reference is the mistake this guards against.
    """
    ref = lq.LQParameters(0.1, 0.05)  # alpha/beta = 2
    acute = lq.rbe(0.1, 0.5, 0.05, reference=ref)  # ion alpha/beta = 10
    fractionated = lq.rbe(0.1, 0.5, 0.05, reference=ref, g=1.0 / 6.0, reference_g=1.0 / 6.0)
    assert fractionated > acute


def test_rbe_falls_with_fractionation_when_the_ion_has_the_bigger_shoulder() -> None:
    """The converse, to pin that the direction is set by the two alpha/beta ratios."""
    ref = lq.LQParameters(0.5, 0.05)  # alpha/beta = 10
    acute = lq.rbe(0.1, 0.6, 0.3, reference=ref)  # test alpha/beta = 2
    fractionated = lq.rbe(0.1, 0.6, 0.3, reference=ref, g=1.0 / 6.0, reference_g=1.0 / 6.0)
    assert fractionated < acute


def test_rbe_protraction_defaults_are_acute_on_both_arms() -> None:
    ref = lq.LQParameters(0.1, 0.05)
    assert lq.rbe(0.1, 0.5, 0.05, reference=ref, g=1.0, reference_g=1.0) == lq.rbe(
        0.1, 0.5, 0.05, reference=ref
    )


def test_bed_rejects_negative_dose() -> None:
    with pytest.raises(ValueError):
        lq.bed(-1.0, alpha_beta_ratio_Gy=10.0)


@pytest.mark.parametrize("bad", [0.0, -2.0])
def test_eqd2_rejects_bad_reference_fraction(bad: float) -> None:
    with pytest.raises(ValueError):
        lq.eqd2(60.0, alpha_beta_ratio_Gy=10.0, reference_fraction_Gy=bad)
