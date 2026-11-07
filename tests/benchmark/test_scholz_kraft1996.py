"""Scholz & Kraft 1996 LEM I benchmark reference data and reconstruction.

Reference: Scholz, M. & Kraft, G. 1996, Adv. Space Res. 18(1/2):5-14,
doi:10.1016/0273-1177(95)00784-C.
"""

from __future__ import annotations

import math

import numpy as np
import pytest

from pyradbio import Ion
from pyradbio.benchmark import scholz_kraft1996 as sk
from pyradbio.core.lq import dose_for_survival
from pyradbio.models.lem.lem1 import LEM1
from pyradbio.track.stopping_power import SimpleBetheStoppingPower


def _lem1_for(inputs: sk.PaperCellInputs) -> LEM1:
    return LEM1(
        photon=inputs.photon(),
        cell=inputs.cell_target(),
        stopping_power=SimpleBetheStoppingPower(),
    )


def test_curve_loader_groups_digitized_points() -> None:
    curves = sk.load_curves()

    assert len(curves) >= 20
    assert all(len(c.x) == len(c.y) >= 2 for c in curves)
    assert all(c.doi == sk.SCHOLZ_KRAFT1996_DOI for c in curves)
    # x-values are stored sorted ascending (grouped-and-sorted loader contract).
    assert all(list(c.x) == sorted(c.x) for c in curves)


def test_compare_points_to_curve_interpolates_and_excludes_unsupported_points() -> None:
    """MC residuals are evaluated only where the digitized paper line exists."""
    curve = sk.ScholzKraft1996Curve(
        figure="test",
        particle="carbon",
        series="calc",
        survival_level=None,
        energy_MeV_u=None,
        x_quantity="let",
        x_unit="keV/um",
        y_quantity="rbe",
        y_unit="1",
        x=(1.0, 10.0, 100.0),
        y=(1.0, 2.0, 3.0),
        source_page=1,
        note="synthetic interpolation fixture",
    )

    comparison = sk.compare_points_to_curve(
        curve,
        x=(0.5, math.sqrt(10.0), 10.0, 200.0),
        model_y=(99.0, 1.8, 2.2, 99.0),
        model_error=(1.0, 0.18, 0.11, 1.0),
        log_x=True,
    )

    assert comparison.points_outside_support == 2
    assert len(comparison.points) == 2
    assert comparison.points[0].paper_y == pytest.approx(1.5)
    assert comparison.points[0].relative_residual == pytest.approx(0.2)
    assert comparison.points[1].relative_residual == pytest.approx(0.1)
    assert comparison.mean_ratio == pytest.approx(1.15)
    assert comparison.rms_relative_error == pytest.approx(math.sqrt((0.2**2 + 0.1**2) / 2))
    assert comparison.max_model_relative_uncertainty == pytest.approx(0.12)


def test_compare_points_to_curve_validates_point_arrays() -> None:
    curve = sk.get_curve("fig3_sigma_vs_energy_v79", particle="carbon", series="calc")

    with pytest.raises(ValueError, match="same length"):
        sk.compare_points_to_curve(curve, x=(1.0, 2.0), model_y=(1.0,))


def test_fig2_model_curves_rise_and_saturate_near_effective_nuclear_size() -> None:
    """The digitized Fig. 2 model sigma(LET) curves (curve1..curve6) rise from the
    photon limit and saturate at the ~50 um^2 effective nuclear-size scale (the
    heaviest ions overshoot toward ~100 um^2)."""
    curves = sk.iter_curves(figure="fig2_sigma_vs_let_v79", series="calc")

    assert len(curves) >= 5
    assert all(c.x_quantity == "let" and c.y_quantity == "sigma_inact" for c in curves)
    # the lowest-LET curve starts well below the nuclear size (the photon limit)
    low_let = min(curves, key=lambda c: c.x[0])
    assert low_let.y[0] < 5.0
    # the curves saturate at / above the 50 um^2 nuclear-size scale
    assert 45.0 <= max(y for c in curves for y in c.y) <= 130.0


def test_fig3_carbon_falls_argon_flat_uranium_rises() -> None:
    """Fig. 3 branches: sigma(E) for C decreases, Ar ~flat, U increases."""
    carbon = sk.get_curve("fig3_sigma_vs_energy_v79", particle="carbon", series="calc")
    argon = sk.get_curve("fig3_sigma_vs_energy_v79", particle="argon", series="calc")
    uranium = sk.get_curve("fig3_sigma_vs_energy_v79", particle="uranium", series="calc")

    assert carbon.y[-1] < carbon.y[0]  # falls
    assert uranium.y[-1] > uranium.y[0]  # rises
    assert abs(argon.y[-1] - argon.y[0]) < 0.2 * argon.y[0]  # roughly flat
    # All three converge near the effective nuclear size at low energy.
    for curve in (carbon, argon, uranium):
        assert curve.y[0] == pytest.approx(51.0, abs=4.0)


def test_final_slope_cross_section_recovers_lq_derivative_at_one_percent() -> None:
    """The 1996 sigma is the final slope at 1% survival, not the initial slope.

    Scholz & Kraft 1996 (doi:10.1016/0273-1177(95)00784-C) state that the
    inactivation cross-section is determined from the final slope of the survival
    curve.  The preceding 1994 paper's Fig. 4 caption identifies the evaluation
    point explicitly as the 1% survival level.
    """
    alpha, beta = 0.2, 0.04
    let_keV_um = 100.0
    target = 0.01
    target_dose = dose_for_survival(target, alpha, beta)
    doses = np.linspace(0.7 * target_dose, 1.3 * target_dose, 9)
    survivals = np.exp(-(alpha * doses + beta * doses**2))

    estimate = sk.estimate_final_slope_cross_section(
        doses,
        survivals,
        let_keV_um=let_keV_um,
        survival_level=target,
    )

    expected_slope = alpha + 2.0 * beta * target_dose
    assert estimate.dose_Gy == pytest.approx(target_dose, rel=1e-10)
    assert estimate.slope_Gy_inv == pytest.approx(expected_slope, rel=1e-10)
    assert estimate.sigma_um2 == pytest.approx(0.16 * let_keV_um * expected_slope, rel=1e-10)


def test_final_slope_cross_section_differs_from_initial_slope_for_a_shoulder() -> None:
    """A shouldered curve has sigma_1% > sigma_0 (Scholz & Kraft 1996)."""
    alpha, beta = 0.2, 0.04
    let_keV_um = 100.0
    doses = np.linspace(2.0, 12.0, 21)
    survivals = np.exp(-(alpha * doses + beta * doses**2))

    estimate = sk.estimate_final_slope_cross_section(doses, survivals, let_keV_um=let_keV_um)
    initial_cross_section = 0.16 * let_keV_um * alpha

    assert estimate.sigma_um2 > initial_cross_section


def test_final_slope_cross_section_does_not_differentiate_high_let_mc_noise() -> None:
    """The final-tail slope must remain stable for a noisy 1%-survival curve.

    These points reproduce one 500-history U, 2 MeV/u diagnostic stream. The
    underlying high-LET curve is nearly exponential: its 1% dose and independent
    single-track cross-section both imply about 52 um2. A five-point quadratic
    derivative amplifies the non-monotonic MC samples to about 100 um2.

    Reference: Scholz & Kraft 1994 Fig. 4 (final slope at 1% survival) and
    Scholz & Kraft 1996 Fig. 3, doi:10.1016/0273-1177(95)00784-C.
    """
    doses = np.array(
        [
            118.792095,
            133.641107,
            148.490119,
            163.339131,
            178.188143,
            193.037155,
            207.886167,
            222.735179,
            237.584190,
        ]
    )
    log_kill = np.array(
        [
            2.53760781,
            2.71925424,
            3.18992352,
            4.06954872,
            3.90593357,
            4.32674569,
            4.95031964,
            4.43225400,
            5.71560810,
        ]
    )
    let_keV_um = 14_829.5447

    estimate = sk.estimate_final_slope_cross_section(
        doses,
        np.exp(-log_kill),
        let_keV_um=let_keV_um,
    )

    expected_from_one_percent_dose = 0.16 * let_keV_um * math.log(100.0) / 207.0
    assert estimate.sigma_um2 == pytest.approx(expected_from_one_percent_dose, rel=0.20)


def test_final_slope_cross_section_requires_target_to_be_bracketed() -> None:
    doses = np.array([0.0, 1.0, 2.0])
    survivals = np.exp(-0.2 * doses)

    with pytest.raises(ValueError, match="bracket"):
        sk.estimate_final_slope_cross_section(doses, survivals, let_keV_um=100.0)


@pytest.mark.parametrize("target", [0.80, 0.20, 0.01])
def test_iso_survival_rbe_recovers_lq_dose_at_paper_levels(target: float) -> None:
    """Fig. 4 RBE uses full-curve iso-survival doses at 80, 20, and 1%.

    Scholz & Kraft 1996, Fig. 4,
    doi:10.1016/0273-1177(95)00784-C.
    """
    alpha_ion, beta_ion = 0.4, 0.025
    alpha_x, beta_x = 0.136, 0.040
    target_dose = dose_for_survival(target, alpha_ion, beta_ion)
    doses = np.linspace(0.7 * target_dose, 1.3 * target_dose, 9)
    survivals = np.exp(-(alpha_ion * doses + beta_ion * doses**2))

    estimate = sk.estimate_iso_survival_rbe(
        doses,
        survivals,
        survival_level=target,
        reference_alpha_Gy_inv=alpha_x,
        reference_beta_Gy_inv2=beta_x,
    )

    expected_rbe = dose_for_survival(target, alpha_x, beta_x) / target_dose
    assert estimate.dose_Gy == pytest.approx(target_dose, rel=1e-10)
    assert estimate.rbe == pytest.approx(expected_rbe, rel=1e-10)


def test_iso_survival_rbe_requires_target_to_be_bracketed() -> None:
    doses = np.array([0.0, 1.0, 2.0])
    survivals = np.exp(-0.2 * doses)

    with pytest.raises(ValueError, match="bracket"):
        sk.estimate_iso_survival_rbe(
            doses,
            survivals,
            survival_level=0.01,
            reference_alpha_Gy_inv=0.136,
            reference_beta_Gy_inv2=0.040,
        )


def test_noisy_crossing_follows_the_monotone_trend_not_the_first_upward_spike() -> None:
    """Both estimators must read a noisy crossing off the projected trend.

    The raw log-kill here bounces (0.65, 0.04, 1.96, 1.91, 0.30, ...), so the
    first sample pair to straddle -ln 0.2 is an isolated upward excursion at
    ~2.5-3.75 Gy. The monotone projection pools those violators and puts the
    crossing at 6.25-7.5 Gy, where the underlying trend actually reaches the
    level. Anchoring on the first raw crossing is exactly how MC noise turned
    into a spurious high-LET cross-section peak.
    """
    doses = np.linspace(0.0, 10.0, 9)
    noisy_log_kill = np.array(
        [
            0.0,
            0.6519420308453809,
            0.0421813329862043,
            1.9596494761563374,
            1.9063097457652658,
            0.29930050678701514,
            2.347559559736236,
            1.613241604821167,
            3.9090521186054907,
        ]
    )

    estimate = sk.estimate_iso_survival_rbe(
        doses,
        np.exp(-noisy_log_kill),
        survival_level=0.20,
        reference_alpha_Gy_inv=0.136,
        reference_beta_Gy_inv2=0.040,
    )

    # Past the isolated spike at doses[3], inside the projected crossing region.
    assert doses[3] < estimate.dose_Gy < doses[6]
    assert estimate.rbe > 0.0

    slope_estimate = sk.estimate_final_slope_cross_section(
        doses,
        np.exp(-noisy_log_kill),
        let_keV_um=100.0,
        survival_level=0.20,
    )
    # Wider default window (9 points) shifts it slightly, but not back to the spike.
    assert doses[3] < slope_estimate.dose_Gy < doses[6]
    assert slope_estimate.sigma_um2 > 0.0


@pytest.mark.slow
def test_lem1_low_dose_cross_section_reaches_effective_size_at_high_let() -> None:
    """The separate low-dose cross-section reaches the effective-size scale."""
    model = _lem1_for(sk.FIGURE_INPUTS["fig2_sigma_vs_let_v79"])
    # Argon at 3 MeV/u is in the model's high-LET overkill regime.
    sigma = model.inactivation_cross_section_um2(Ion(18, 40, 3.0))
    assert sigma == pytest.approx(50.0, abs=8.0)


@pytest.mark.slow
def test_lem1_reproduces_rbe_maximum_shift() -> None:
    """RBE_max shifts to higher LET for heavier ions (paper: p<He<C)."""
    model = _lem1_for(sk.FIGURE_INPUTS["fig4_rbe_vs_let"])
    ref = model.photon
    a_of = {1: 1, 2: 4, 6: 12}

    def rbe_max_let(z: int) -> float:
        d_ref = dose_for_survival(0.80, ref.alpha_x, ref.beta_x)
        best_let, best_rbe = 0.0, 0.0
        for energy in (0.5, 0.7, 1.0, 1.5, 2.0, 3.0, 5.0, 8.0, 12.0, 20.0, 50.0, 120.0):
            ion = Ion(z, a_of[z], energy)
            lq = model.alpha_beta(ion)
            rbe = d_ref / dose_for_survival(0.80, lq.alpha, lq.beta)
            if rbe > best_rbe:
                best_rbe, best_let = rbe, model.stopping_power.let_keV_um(ion)
        return best_let

    let_p = rbe_max_let(1)
    let_he = rbe_max_let(2)
    let_c = rbe_max_let(6)
    assert let_p < let_he < let_c
    # Paper positions ~30 / ~90 / ~300 keV/um (order-of-magnitude window).
    assert 15.0 < let_p < 60.0
    assert 40.0 < let_he < 150.0
    assert 150.0 < let_c < 500.0
