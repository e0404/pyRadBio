"""Digitized Fig. 2 of Matsuya et al. 2025 (doi:10.1002/mp.70040).

What these tests protect is not the loader but the *self-validation* of the
digitization, which is what makes the panel usable as an IMK reference:

* every calculated curve's fitted ``beta`` must land on the paper's printed
  ``beta0`` -- the y-axis calibration and the reading ``beta = F beta0`` at once;
* inverting eq. (1) must return a physically sensible photon ``y*``, and the two
  panels (different cell lines, independently digitized) must agree on it;
* the fig. 2a experimental series must keep pairing with the calculated curve
  they actually match, which is *not* the one they were labelled with.

Fig. 4 -- the micronucleus panel -- is held to a stricter standard, because it is
a *prediction*: eq. (3) forces it to be ``h`` times the already-recovered fig. 2a
curves, so these tests check that it is, and read ``h`` off it rather than
assuming which of the paper's two tables supplied it.

Fig. 7 -- the dose-rate panel -- is where ``core/protraction.py`` meets a
published model line. Its DU145 half discriminates sharply and is asserted on
both ways round (the printed repair rate must fit, and a wrong one must not);
its WI-38 half does not discriminate at all, and there is a test pinning *that*,
so the pair is never mistaken for two validations.
"""

from __future__ import annotations

import itertools
import math

import numpy as np
import pytest
from scipy.optimize import brentq

from pyradbio.benchmark.matsuya2025 import (
    ANCHORS,
    FIG7_DOSE_GY,
    FIG7_NORM_RATE_GY_PER_H,
    PAPER_ROUTE,
    RECOVERED_Z_STAR_1D_GY,
    dose_rate_shape_fit,
    fig7_protocol,
    fig7_schedule,
    get_fig2_curve,
    get_fig4_curve,
    get_fig7_curve,
    load_fig2_curves,
    load_fig4_curves,
    load_fig7_curves,
    microdosimetry_routes,
    paper_rbe_alpha,
    recovered_h,
    recovered_z_star_1D,
    recovered_z_star_1D_from_mn,
    relative_mn,
    series_pairing_residuals,
)
from pyradbio.core.lq import LQParameters, dose_for_survival
from pyradbio.core.protraction import DoseSchedule
from pyradbio.core.radiation import Ion
from pyradbio.micro.lineal_energy import AnalyticSpectrumEngine, SphericalSite
from pyradbio.micro.sato import SatoLinealEnergyEngine
from pyradbio.models.mkm.imk import RECOVERED_PHOTON_Z_STAR_1D_GY, IMKModel
from pyradbio.track.stopping_power import SimpleBetheStoppingPower, energy_for_let_MeV_u

CELLS = {"HSG": IMKModel.hsg_matsuya2025(), "SCC": IMKModel.scc_matsuya2025()}


def test_all_twelve_series_load() -> None:
    curves = load_fig2_curves()
    assert len(curves) == 12
    assert {c.cell_line for c in curves} == {"HSG", "SCC"}
    assert {(c.cell_line, c.let_keV_um) for c in curves if c.series == "cal"} == {
        ("HSG", 0.0),
        ("HSG", 13.0),
        ("HSG", 100.0),
        ("SCC", 0.0),
        ("SCC", 46.0),
        ("SCC", 80.0),
    }


def test_curves_are_monotone_and_in_range() -> None:
    for curve in load_fig2_curves():
        assert all(0.0 < s <= 1.0 for s in curve.survival)
        assert list(curve.dose_Gy) == sorted(curve.dose_Gy)
        if curve.series == "cal":
            assert len(curve.dose_Gy) >= 30
            assert all(b <= a for a, b in zip(curve.survival, curve.survival[1:], strict=False))


def test_experimental_error_bars_bracket_the_centre() -> None:
    for curve in (c for c in load_fig2_curves() if c.series == "exp"):
        lower, upper = curve.error_bars()
        assert all(v >= 0.0 for v in lower)
        assert all(v >= 0.0 for v in upper)


@pytest.mark.golden
def test_calculated_betas_reproduce_the_printed_beta0() -> None:
    """The three lines per panel share the paper's table-1 ``beta0``.

    This is the digitization's own calibration check: ``beta`` is fixed by the
    curvature of the line on a log axis, so hitting the printed value to 1%
    across six independently traced curves says the y-axis was read correctly --
    and confirms ``beta = F beta0`` with ``F = 1`` for these acute curves.
    """
    for cell_line, model in CELLS.items():
        for curve in (
            c for c in load_fig2_curves() if c.cell_line == cell_line and c.series == "cal"
        ):
            _, beta = curve.lq_fit()
            assert beta == pytest.approx(model.beta0, rel=0.01), f"{cell_line} {curve.label}"


@pytest.mark.golden
def test_recovered_z_star_matches_the_recorded_values() -> None:
    """``z*_1D = (alpha - alpha0)/beta0`` for each calculated curve."""
    for cell_line, model in CELLS.items():
        for curve in (
            c for c in load_fig2_curves() if c.cell_line == cell_line and c.series == "cal"
        ):
            _, _, z_star = recovered_z_star_1D(
                curve, alpha0_Gy_inv=model.alpha0, beta0_Gy2=model.beta0
            )
            expected = RECOVERED_Z_STAR_1D_GY[(cell_line, curve.let_keV_um)]
            assert z_star == pytest.approx(expected, rel=0.01)


@pytest.mark.golden
def test_the_two_panels_agree_on_the_photon_y_star() -> None:
    """Independent digitizations, different cell lines, same domain size.

    Both panels are scored at ``r_d = 0.42 um`` with ``y0 = 150 keV/um``, so the
    photon ``y*`` recovered from each must agree -- and land somewhere sensible
    for a kV X-ray beam (a few keV/um), which nothing in the inversion forces.
    """
    y_star = {}
    for cell_line, model in CELLS.items():
        curve = get_fig2_curve(cell_line=cell_line, series="cal", let_keV_um=0.0)
        _, _, z_star = recovered_z_star_1D(curve, alpha0_Gy_inv=model.alpha0, beta0_Gy2=model.beta0)
        y_star[cell_line] = z_star * model._z_domain_scale_keV_um_per_Gy
    assert y_star["HSG"] == pytest.approx(3.62, rel=0.02)
    assert y_star["SCC"] == pytest.approx(3.41, rel=0.02)
    assert y_star["SCC"] / y_star["HSG"] == pytest.approx(1.0, abs=0.07)
    assert all(2.0 < v < 6.0 for v in y_star.values())


@pytest.mark.golden
def test_photon_reference_is_not_the_furusawa_one() -> None:
    """Furusawa et al. 2000's HSG 200 kVp photon fit is a *different* one: 0.313/0.061.

    Pinned because substituting it -- or this package's mMKM HSG defaults, which
    are in the same family -- would change every IMK RBE by ~1.6x in the photon
    arm. The paper's own X-ray line is much shallower.
    """
    alpha, beta = get_fig2_curve(cell_line="HSG", series="cal", let_keV_um=0.0).lq_fit()
    assert alpha == pytest.approx(0.1988, rel=0.02)
    assert beta == pytest.approx(0.0467, rel=0.02)
    assert alpha < 0.75 * 0.313  # Furusawa 2000 alpha_x for the same cell line


def test_fig2a_experimental_series_pair_with_the_curve_they_match() -> None:
    """The stored LET of each fig. 2a carbon point set is the one it fits.

    They were digitized under the opposite labels; see the module docstring.
    The diagonal (as stored) must beat the off-diagonal by a wide margin, which
    is the whole evidence for the swap.
    """
    residuals = series_pairing_residuals("HSG")
    for let, other in ((13.0, 100.0), (100.0, 13.0)):
        assert residuals[(let, let)] < 0.15
        assert residuals[(let, other)] > 0.7
        assert residuals[(let, other)] > 5.0 * residuals[(let, let)]


def test_fig2b_experimental_series_are_not_swapped() -> None:
    """SCC's panel needed no correction -- the diagnostic is symmetric, so run it."""
    residuals = series_pairing_residuals("SCC")
    for let in (0.0, 46.0, 80.0):
        matched = residuals[(let, let)]
        assert matched < 0.1
        assert all(matched < residuals[(let, o)] for o in (0.0, 46.0, 80.0) if o != let)


def test_the_experimental_points_support_their_own_curve() -> None:
    """Each measured series sits on its calculated curve within digitizing scatter."""
    for cell_line in CELLS:
        for let, tolerance in (
            (0.0, 0.10),
            (13.0, 0.12),
            (46.0, 0.10),
            (80.0, 0.10),
            (100.0, 0.12),
        ):
            try:
                get_fig2_curve(cell_line=cell_line, series="exp", let_keV_um=let)
            except KeyError:
                continue
            assert series_pairing_residuals(cell_line)[(let, let)] < tolerance


def test_error_bars_refuse_a_calculated_curve() -> None:
    """A model line has no measurement uncertainty; asking for it is a mistake."""
    curve = get_fig2_curve(cell_line="HSG", series="cal", let_keV_um=13.0)
    with pytest.raises(ValueError, match="error bars"):
        curve.error_bars()


def test_lq_fit_rejects_a_window_with_too_few_points() -> None:
    curve = get_fig2_curve(cell_line="HSG", series="exp", let_keV_um=13.0)
    with pytest.raises(ValueError, match="too few points"):
        curve.lq_fit(min_log_kill=50.0, max_log_kill=60.0)


def test_z_star_recovery_refuses_experimental_series() -> None:
    curve = get_fig2_curve(cell_line="HSG", series="exp", let_keV_um=13.0)
    with pytest.raises(ValueError, match="calculated curve"):
        recovered_z_star_1D(curve, alpha0_Gy_inv=0.150, beta0_Gy2=0.0467)


def test_get_fig2_curve_rejects_an_unknown_key() -> None:
    with pytest.raises(KeyError, match=r"no Fig\. 2 series"):
        get_fig2_curve(cell_line="V79", series="cal", let_keV_um=13.0)


def test_recovered_carbon_z_star_exceeds_the_photon_one() -> None:
    """Sanity: quality ordering must survive the whole recovery chain."""
    for cell_line in CELLS:
        photon = RECOVERED_Z_STAR_1D_GY[(cell_line, 0.0)]
        carbon = [v for (c, let), v in RECOVERED_Z_STAR_1D_GY.items() if c == cell_line and let]
        assert all(v > 2.0 * photon for v in carbon)
        assert math.isclose(max(carbon), sorted(carbon)[-1])


# --- the reconstruction itself ----------------------------------------------
# Our ``z*_1D`` deviation from the paper's recovered value, per anchor, for the
# two impact-parameter site geometries. **Regression pins on a known-imperfect
# reconstruction, not accuracy targets** -- see
# ``docs/references/IMK_Matsuya2025.md``, "What fig. 2 gives back" ("The four
# carbon anchors are not one curve in LET"), for why the mid-LET overshoot is
# not closable by a choice of route (the paper's own four anchors do not lie on
# one curve in LET, because its ``y*`` is beam-specific).
_ANCHOR_DEVIATION = {
    "cylinder": {
        ("HSG", 13.0): -0.299,
        ("SCC", 46.0): +0.153,
        ("SCC", 80.0): +0.312,
        ("HSG", 100.0): +0.067,
    },
    "sphere": {
        ("HSG", 13.0): -0.064,
        ("SCC", 46.0): +0.393,
        ("SCC", 80.0): +0.466,
        ("HSG", 100.0): +0.140,
    },
}


def _carbon_at(let_keV_um: float) -> Ion:
    energy = energy_for_let_MeV_u(
        SimpleBetheStoppingPower(),
        6,
        12,
        let_keV_um,
        branch="bracket",
        energy_bounds_MeV_u=(0.6, 600.0),
        xtol=1e-8,
    )
    return Ion(Z=6, A=12, energy_MeV_u=energy)


@pytest.mark.parametrize("let_keV_um", [13.0, 46.0, 80.0, 100.0])
def test_carbon_inversion_is_the_inline_brentq(let_keV_um: float) -> None:
    """``_carbon_at`` and validation/imk_matsuya2025.py ``carbon_at_let`` adopt
    ``energy_for_let_MeV_u(branch="bracket")``: the inline ``brentq`` energy,
    bit for bit."""
    stopping = SimpleBetheStoppingPower()
    inline = brentq(
        lambda e: stopping.let_keV_um(Ion(Z=6, A=12, energy_MeV_u=e)) - let_keV_um,
        0.6,
        600.0,
        xtol=1e-8,
    )
    assert _carbon_at(let_keV_um).energy_MeV_u == inline


@pytest.mark.golden
@pytest.mark.parametrize("geometry", ["cylinder", "sphere"])
def test_golden_reconstruction_of_the_four_anchors(geometry: str) -> None:
    """Our ``z*_1D`` against the paper's own, on its printed parameters.

    This is the IMK reconstruction score. It lives as a test rather than only in
    the example because the numbers move whenever the microdosimetry engines
    move, and that must not happen silently.
    """
    for (cell_line, let), expected in _ANCHOR_DEVIATION[geometry].items():
        built = (
            IMKModel.hsg_matsuya2025(site_geometry=geometry)
            if cell_line == "HSG"
            else IMKModel.scc_matsuya2025(site_geometry=geometry)
        )
        ours = built.specific_energies(_carbon_at(let)).z_star_1D_Gy
        theirs = RECOVERED_Z_STAR_1D_GY[(cell_line, let)]
        assert ours / theirs - 1.0 == pytest.approx(expected, abs=0.02), f"{cell_line} {let}"


@pytest.mark.golden
def test_golden_rbe10_against_the_papers_own_curves() -> None:
    """``RBE_10`` from our model vs from the paper's fig.-2 lines (its eqs. 4-5).

    Both arms come from the paper here -- the photon reference is its own
    recovered X-ray curve -- so this is model-vs-model, which is the comparison
    that means something. The measured spread is -5.5% at LET 13 to +15.7% at 80.
    """
    expected = {
        ("HSG", 13.0): -0.055,
        ("SCC", 46.0): +0.055,
        ("SCC", 80.0): +0.157,
        ("HSG", 100.0): +0.045,
    }
    for (cell_line, let), want in expected.items():
        model = IMKModel.hsg_matsuya2025() if cell_line == "HSG" else IMKModel.scc_matsuya2025()
        ours = model.rbe_at_survival(_carbon_at(let), 0.1)
        ion_line = get_fig2_curve(cell_line=cell_line, series="cal", let_keV_um=let)
        photon_line = get_fig2_curve(cell_line=cell_line, series="cal", let_keV_um=0.0)
        theirs = dose_for_survival(0.1, *photon_line.lq_fit()) / dose_for_survival(
            0.1, *ion_line.lq_fit()
        )
        assert ours / theirs - 1.0 == pytest.approx(want, abs=0.02), f"{cell_line} {let}"


def test_our_photon_arm_reproduces_the_papers_x_ray_line() -> None:
    """Exact by construction -- but it is the construction worth pinning."""
    for cell_line in CELLS:
        model = IMKModel.hsg_matsuya2025() if cell_line == "HSG" else IMKModel.scc_matsuya2025()
        alpha, beta = get_fig2_curve(cell_line=cell_line, series="cal", let_keV_um=0.0).lq_fit()
        assert model.reference().alpha_x == pytest.approx(alpha, rel=0.005)
        assert model.reference().beta_x == pytest.approx(beta, rel=0.01)


def test_model_and_benchmark_agree_on_the_recovered_photon_reference() -> None:
    """``models`` keeps its own copy so it need not import ``benchmark``."""
    for cell_line, z_star in RECOVERED_PHOTON_Z_STAR_1D_GY.items():
        assert z_star == pytest.approx(RECOVERED_Z_STAR_1D_GY[(cell_line, 0.0)], abs=5e-4)


# --------------------------------------------------------------------------
# Fig. 4 -- micronuclei, the panel the model predicts with no free parameter
# --------------------------------------------------------------------------
FIG4_LETS = (0.0, 13.0, 100.0)
HSG_ALPHA0, HSG_BETA0 = 0.150, 0.0467


def test_all_six_fig4_series_load() -> None:
    curves = load_fig4_curves()
    assert len(curves) == 6
    assert {c.cell_line for c in curves} == {"HSG"}
    assert {(c.series, c.let_keV_um) for c in curves} == {
        (series, let) for series in ("cal", "exp") for let in FIG4_LETS
    }


def test_fig4_curves_are_monotone_and_positive() -> None:
    for curve in load_fig4_curves():
        assert all(v > 0.0 for v in curve.mn_per_bn_cell)
        assert list(curve.dose_Gy) == sorted(curve.dose_Gy)
        if curve.series == "cal":
            assert len(curve.dose_Gy) >= 70
            trimmed = [
                v for d, v in zip(curve.dose_Gy, curve.mn_per_bn_cell, strict=True) if d > 0.05
            ]
            assert all(a <= b for a, b in itertools.pairwise(trimmed))


def test_fig4_error_bars_are_symmetric_about_the_centre() -> None:
    """Linear axes: only the bar top was traced, so the lower end is its mirror."""
    for curve in (c for c in load_fig4_curves() if c.series == "exp"):
        lower, upper = curve.error_bars()
        assert all(v > 0.0 for v in lower)
        # 1e-5, not exact: the CSV stores six significant figures.
        assert lower == pytest.approx(upper, abs=1e-5)


@pytest.mark.golden
def test_fig4_measures_the_table1_h_and_excludes_the_table2_one() -> None:
    """``h`` read off the quadratic term is 0.275 (table 1), not 0.245 (table 2).

    ``beta_m = h beta0`` and eq. (1) fixes ``beta0`` for *every* quality, so this
    reads ``h`` without knowing the curve's ``z*_1D``. All three curves land on
    the table-1 value; the HSG-specific table-2 value is 11% away.
    """
    for let in FIG4_LETS:
        curve = get_fig4_curve(series="cal", let_keV_um=let)
        _alpha_m, _beta_m, h = recovered_h(curve, beta0_Gy2=HSG_BETA0)
        assert h == pytest.approx(0.275, rel=0.012), f"LET {let}"
        assert abs(h / 0.245 - 1.0) > 0.08, f"LET {let}: table-2 h is not excluded"


@pytest.mark.golden
def test_fig4_recovers_the_fig2a_z_star_from_a_different_panel() -> None:
    """A second, independent hand digitization returns the same PHITS inputs.

    Fig. 4 is a different figure with a different y quantity and linear axes, so
    nothing but the model ties it to fig. 2a. Agreement to <2% is the strongest
    available check on `recovered_z_star_1D`.
    """
    for let, tolerance in ((0.0, 0.01), (13.0, 0.01), (100.0, 0.02)):
        curve = get_fig4_curve(series="cal", let_keV_um=let)
        _h, z_star = recovered_z_star_1D_from_mn(
            curve, alpha0_Gy_inv=HSG_ALPHA0, beta0_Gy2=HSG_BETA0
        )
        assert z_star == pytest.approx(RECOVERED_Z_STAR_1D_GY[("HSG", let)], rel=tolerance)


@pytest.mark.golden
def test_the_micronucleus_identity_reproduces_the_panel_pointwise() -> None:
    """``F_MN = h <w>``: our eq. (3) redraws fig. 4 from the fig. 2a survival LQ.

    Zero fitted quantities -- the alphas and betas come from fig. 2a, ``h`` from
    table 1 -- so a small residual here is the identity holding, not a fit.
    """
    model = IMKModel.hsg_matsuya2025()
    assert model.mn_probability == 0.275
    for let in FIG4_LETS:
        alpha, beta = get_fig2_curve(cell_line="HSG", series="cal", let_keV_um=let).lq_fit()
        lq = LQParameters(alpha=alpha, beta=beta)
        curve = get_fig4_curve(series="cal", let_keV_um=let)
        residuals = [
            model.micronucleus_frequency(d, lq) / f - 1.0
            for d, f in zip(curve.dose_Gy, curve.mn_per_bn_cell, strict=True)
            if d > 0.05
        ]
        rms = math.sqrt(sum(r * r for r in residuals) / len(residuals))
        assert rms < 0.01, f"LET {let}: RMS {rms:.4f}"


def test_fig4_carbon100_points_sit_on_their_own_calculated_curve() -> None:
    """The one pairing fig. 4 can decide -- and it is *not* swapped.

    The X-ray and LET-13 lines differ by only ~30% over this dose range, well
    inside the scatter, so they are deliberately not asserted on.
    """
    residuals = {}
    exp = get_fig4_curve(series="exp", let_keV_um=100.0)
    for let in FIG4_LETS:
        alpha_m, beta_m = get_fig4_curve(series="cal", let_keV_um=let).lq_fit()
        predicted = [alpha_m * d + beta_m * d**2 for d in exp.dose_Gy]
        residuals[let] = math.sqrt(
            sum((f / p - 1.0) ** 2 for f, p in zip(exp.mn_per_bn_cell, predicted, strict=True))
            / len(predicted)
        )
    assert residuals[100.0] < 0.15
    assert all(residuals[let] > 10.0 * residuals[100.0] for let in (0.0, 13.0))


def test_fig4_error_bars_refuse_a_calculated_curve() -> None:
    with pytest.raises(ValueError, match="error bars"):
        get_fig4_curve(series="cal", let_keV_um=13.0).error_bars()


def test_fig4_h_recovery_refuses_experimental_series() -> None:
    with pytest.raises(ValueError, match="calculated curve"):
        recovered_h(get_fig4_curve(series="exp", let_keV_um=13.0), beta0_Gy2=HSG_BETA0)


def test_fig4_lq_fit_rejects_too_high_a_dose_floor() -> None:
    with pytest.raises(ValueError, match="too few points"):
        get_fig4_curve(series="exp", let_keV_um=13.0).lq_fit(min_dose_Gy=50.0)


def test_get_fig4_curve_rejects_an_unknown_key() -> None:
    with pytest.raises(KeyError, match=r"no Fig\. 4 series"):
        get_fig4_curve(series="cal", let_keV_um=46.0)


# --------------------------------------------------------------------------
# Fig. 7 -- dose rate, i.e. the Lea-Catcheside factor against a published line
# --------------------------------------------------------------------------
FIG7_NORM_RATE = FIG7_NORM_RATE_GY_PER_H
FIG7_CELLS = {
    "DU145": IMKModel.du145_matsuya2025(),
    "WI-38": IMKModel.wi38_matsuya2025(),
}


def _time_factor(model: IMKModel, rate_Gy_per_h: float) -> float:
    """``F`` for the paper's continuous delivery of 4 Gy at ``rate_Gy_per_h``."""
    return model.time_factor(fig7_schedule(rate_Gy_per_h))


def _relative_mn(model: IMKModel, rate_Gy_per_h: float, z_photon_Gy: float) -> float:
    """Eq. (1) + eq. (3), normalised at 60 Gy/h as sec. 3.1.3 specifies.

    ``h`` cancels in the ratio, exactly as it does in eq. (6).
    """
    return relative_mn(model, rate_Gy_per_h, z_photon_Gy=z_photon_Gy)


def test_all_eight_fig7_series_load() -> None:
    curves = load_fig7_curves()
    assert len(curves) == 8
    assert {(c.cell_line, c.series) for c in curves} == {
        (cell, series) for cell in FIG7_CELLS for series in ("cal", "cal_ci_lo", "cal_ci_hi", "exp")
    }
    for curve in curves:
        assert list(curve.dose_rate_Gy_per_h) == sorted(curve.dose_rate_Gy_per_h)
        assert all(v > 0.0 for v in curve.relative_mn)


def test_fig7_confidence_band_brackets_the_mean_line() -> None:
    for cell_line in FIG7_CELLS:
        mean = get_fig7_curve(cell_line=cell_line, series="cal")
        for series, sign in (("cal_ci_hi", 1.0), ("cal_ci_lo", -1.0)):
            bound = get_fig7_curve(cell_line=cell_line, series=series)
            interpolated = [float(v) for v in np.interp(bound.rate, mean.rate, mean.mn)]
            assert all(
                sign * (b - m) > 0.0 for b, m in zip(bound.relative_mn, interpolated, strict=True)
            ), f"{cell_line} {series}"


def test_fig7_is_normalised_at_60_gy_per_hour() -> None:
    """Sec. 3.1.3 fixes the 60 Gy/h arm at 1, so the digitized point must be 1.

    A y-calibration check that costs nothing: the paper did not measure this
    value, it defined it.
    """
    for cell_line in FIG7_CELLS:
        exp = get_fig7_curve(cell_line=cell_line, series="exp")
        highest = max(zip(exp.dose_rate_Gy_per_h, exp.relative_mn, strict=True))
        assert highest[0] == pytest.approx(FIG7_NORM_RATE, rel=0.002), cell_line
        assert highest[1] == pytest.approx(1.0, abs=0.005), cell_line


@pytest.mark.golden
def test_du145_dose_rate_shape_is_our_lea_catcheside_factor() -> None:
    """The paper's own line is ``A + B F(rate)`` with our ``F`` at its printed a+c.

    Both directions matter: the printed 2.10/h must fit a 58-point curve to
    0.2% of its span, *and* halving or doubling it must not. Without the second
    half this would only be saying that a two-parameter affine fit can follow a
    monotone curve.
    """
    model = FIG7_CELLS["DU145"]
    curve = get_fig7_curve(cell_line="DU145", series="cal")
    span = max(curve.relative_mn) - min(curve.relative_mn)
    residuals = {}
    for factor in (0.5, 1.0, 2.0):
        wrong = IMKModel.du145_matsuya2025(repair_rate_per_h=factor * model.repair_rate_per_h)
        *_, residuals[factor] = dose_rate_shape_fit(curve, lambda r, m=wrong: _time_factor(m, r))
    assert residuals[1.0] < 0.003, residuals
    assert residuals[1.0] / span < 0.005
    assert min(residuals[0.5], residuals[2.0]) > 10.0 * residuals[1.0], residuals


def test_wi38_dose_rate_shape_cannot_discriminate_the_repair_rate() -> None:
    """And is therefore not a second validation. Pinning the *absence* of one.

    WI-38's whole effect is 0.03 in y; the same fit that separates DU145's
    repair rate by 25x separates nothing here, so reading its agreement as
    confirmation would be a mistake the pair invites.
    """
    model = FIG7_CELLS["WI-38"]
    curve = get_fig7_curve(cell_line="WI-38", series="cal")
    assert max(curve.relative_mn) - min(curve.relative_mn) < 0.05
    residuals = {}
    for factor in (0.5, 1.0, 2.0):
        wrong = IMKModel.wi38_matsuya2025(repair_rate_per_h=factor * model.repair_rate_per_h)
        *_, residuals[factor] = dose_rate_shape_fit(curve, lambda r, m=wrong: _time_factor(m, r))
    assert max(residuals.values()) < 1.5 * min(residuals.values()), residuals


@pytest.mark.golden
def test_our_dose_rate_prediction_sits_inside_the_papers_own_95_percent_ci() -> None:
    """Printed parameters, no fitting -- scored against the paper's band.

    ``h`` cancels in the ratio and the photon ``z*_1D`` barely enters, so this
    is essentially eq. (1)'s time factor and the printed ``alpha0``/``beta0``
    against a published prediction.
    """
    for cell_line, model in FIG7_CELLS.items():
        mean = get_fig7_curve(cell_line=cell_line, series="cal")
        for series, want_below in (("cal_ci_hi", True), ("cal_ci_lo", False)):
            bound = get_fig7_curve(cell_line=cell_line, series=series)
            for rate, edge in zip(bound.dose_rate_Gy_per_h, bound.relative_mn, strict=True):
                ours = _relative_mn(model, rate, z_photon_Gy=1.0)
                inside = ours <= edge if want_below else ours >= edge
                assert inside, f"{cell_line} {series} at {rate:.2f} Gy/h: {ours:.4f} vs {edge:.4f}"
        worst = max(
            abs(_relative_mn(model, rate, z_photon_Gy=1.0) - value)
            for rate, value in zip(mean.dose_rate_Gy_per_h, mean.relative_mn, strict=True)
        )
        assert worst < 0.02, f"{cell_line}: worst deviation from the mean line {worst:.4f}"


def test_fig7_photon_reference_is_not_recoverable_from_this_panel() -> None:
    """Why DU145/WI-38 still have no photon reference, stated as a test.

    Unlike fig. 2, the relative curve is nearly blind to ``alpha``: moving the
    photon ``z*_1D`` over the whole plausible range moves DU145's line by less
    than a third of its span, and WI-38's by essentially nothing.
    """
    for cell_line, tolerance in (("DU145", 0.15), ("WI-38", 0.01)):
        model = FIG7_CELLS[cell_line]
        curve = get_fig7_curve(cell_line=cell_line, series="cal")
        lowest = min(curve.dose_rate_Gy_per_h)
        spread = abs(
            _relative_mn(model, lowest, z_photon_Gy=0.25)
            - _relative_mn(model, lowest, z_photon_Gy=1.3)
        )
        assert spread < tolerance, f"{cell_line}: {spread:.4f}"
        assert model.reference_z_star_1D_Gy is None


def test_fig7_error_bars_refuse_a_calculated_curve() -> None:
    with pytest.raises(ValueError, match="error bars"):
        get_fig7_curve(cell_line="DU145", series="cal").error_bars()


def test_fig7_shape_fit_refuses_a_confidence_bound() -> None:
    with pytest.raises(ValueError, match="calculated mean line"):
        dose_rate_shape_fit(
            get_fig7_curve(cell_line="DU145", series="cal_ci_hi"),
            lambda r: _time_factor(FIG7_CELLS["DU145"], r),
        )


def test_get_fig7_curve_rejects_an_unknown_key() -> None:
    with pytest.raises(KeyError, match=r"no Fig\. 7 series"):
        get_fig7_curve(cell_line="HSG", series="cal")


# --------------------------------------------------------------------------
# Reconstruction helpers shared with validation/imk_matsuya2025.py
# --------------------------------------------------------------------------
def _inline_relative_mn(model: IMKModel, rate_Gy_per_h: float, z_photon_Gy: float) -> float:
    """The pre-library form of `relative_mn`."""
    lq_hi = model.alpha_beta_from_z_star_1D(
        z_photon_Gy,
        schedule=DoseSchedule.continuous(4.0, duration_h=4.0 / 60.0),
    )
    lq = model.alpha_beta_from_z_star_1D(
        z_photon_Gy,
        schedule=DoseSchedule.continuous(4.0, duration_h=4.0 / rate_Gy_per_h),
    )
    return model.micronucleus_frequency(4.0, lq) / model.micronucleus_frequency(4.0, lq_hi)


@pytest.mark.parametrize("rate", [1.0, 3.7, 60.0, 190.0])
@pytest.mark.parametrize("z_photon", [0.25, 1.0, 1.3])
def test_relative_mn_is_the_inline_formula(rate: float, z_photon: float) -> None:
    for model in FIG7_CELLS.values():
        assert relative_mn(model, rate, z_photon_Gy=z_photon) == _inline_relative_mn(
            model, rate, z_photon
        )
    assert relative_mn(FIG7_CELLS["DU145"], FIG7_NORM_RATE_GY_PER_H) == 1.0


def test_fig7_schedule_is_continuous_4_gy() -> None:
    assert FIG7_DOSE_GY == 4.0
    assert FIG7_NORM_RATE_GY_PER_H == 60.0
    assert fig7_schedule(30.0) == DoseSchedule.continuous(4.0, duration_h=4.0 / 30.0)


def test_fig7_protocol_is_the_sec_3_1_4_regimen() -> None:
    """The schedules of ``tests/models/test_imk.py``'s dose-rate protocol test."""
    assert fig7_protocol(1.0) == DoseSchedule.continuous(4.0, duration_h=4.0 / 60.0)
    assert fig7_protocol(2.0) == DoseSchedule.continuous(4.0, duration_h=4.0 / 60.0)
    for per_fraction, n in ((0.5, 8), (0.1, 40), (0.05, 80)):
        assert fig7_protocol(per_fraction) == DoseSchedule.fractionated(
            per_fraction,
            n_fractions=n,
            interval_h=1.0 / 60.0,
            fraction_duration_h=per_fraction / 60.0,
        )


@pytest.mark.parametrize("rate", [0.3, 0.7, 0.0, -0.5])
def test_fig7_protocol_rejects_rates_that_do_not_deliver_4_gy(rate: float) -> None:
    """0.3 Gy/fraction would round to 13 fractions = 3.9 Gy; <= 0 has no schedule."""
    with pytest.raises(ValueError, match=r"fig\. 7 protocol rate|whole fractions"):
        fig7_protocol(rate)


@pytest.mark.parametrize("rate", [0.0, -30.0])
def test_fig7_schedule_rejects_non_positive_rates(rate: float) -> None:
    with pytest.raises(ValueError, match="must be > 0 Gy/h"):
        fig7_schedule(rate)


def test_anchors_are_the_recovered_carbon_keys_in_let_order() -> None:
    carbon = [key for key in RECOVERED_Z_STAR_1D_GY if key[1] > 0.0]
    assert sorted(carbon, key=lambda key: key[1]) == list(ANCHORS)
    assert [let for _, let in ANCHORS] == [13.0, 46.0, 80.0, 100.0]


def test_paper_rbe_alpha_is_the_ratio_of_the_fig2_alphas() -> None:
    for cell_line, let in ANCHORS:
        cal = get_fig2_curve(cell_line=cell_line, series="cal", let_keV_um=let)
        photon = get_fig2_curve(cell_line=cell_line, series="cal", let_keV_um=0.0)
        assert paper_rbe_alpha(cell_line, let) == cal.lq_fit()[0] / photon.lq_fit()[0]
    assert paper_rbe_alpha("HSG", 100.0) > paper_rbe_alpha("HSG", 13.0) > 1.0


# Fig. 2's fitted ``alpha`` [1/Gy] as printed, to 4 decimals, in
# docs/references/IMK_Matsuya2025.md ("What fig. 2 gives back", the table).
_FIG2_ALPHA_DOC = {
    ("HSG", 0.0): 0.1988,
    ("HSG", 13.0): 0.2865,
    ("HSG", 100.0): 0.9058,
    ("SCC", 0.0): 0.1119,
    ("SCC", 46.0): 0.2913,
    ("SCC", 80.0): 0.4051,
}


@pytest.mark.parametrize(("cell_line", "let_keV_um"), list(ANCHORS))
def test_paper_rbe_alpha_matches_the_documented_fig2_alphas(
    cell_line: str, let_keV_um: float
) -> None:
    """Independent of the function body: the ratio of the doc's printed alphas.

    The tolerance is the doc table's 4-decimal rounding (half a unit in the last
    place of each alpha, propagated through the ratio), not a fit tolerance.
    """
    for let in (0.0, let_keV_um):
        fitted = get_fig2_curve(cell_line=cell_line, series="cal", let_keV_um=let).lq_fit()[0]
        assert fitted == pytest.approx(_FIG2_ALPHA_DOC[(cell_line, let)], abs=5e-5)
    ion, photon = _FIG2_ALPHA_DOC[(cell_line, let_keV_um)], _FIG2_ALPHA_DOC[(cell_line, 0.0)]
    rounding = 5e-5 / ion + 5e-5 / photon
    assert paper_rbe_alpha(cell_line, let_keV_um) == pytest.approx(ion / photon, rel=rounding)


@pytest.mark.parametrize("cell_line", ["HSG", "SCC"])
def test_microdosimetry_routes_build_the_four_routes(cell_line: str) -> None:
    sp = SimpleBetheStoppingPower()
    routes = microdosimetry_routes(cell_line, stopping_power=sp)
    assert list(routes) == ["impact/cyl", "impact/sph", "sato-AMF", "analytic"]
    assert PAPER_ROUTE in routes
    cyl, sph, sato, analytic = routes.values()
    assert cyl == CELLS[cell_line]
    assert (cyl.engine, sph.engine, sph.site_geometry) == (None, None, "sphere")
    assert isinstance(sato.engine, SatoLinealEnergyEngine)
    assert sato.engine.site == SphericalSite(diameter_um=2.0 * 0.42)
    assert sato.engine.stopping_power is sp
    assert isinstance(analytic.engine, AnalyticSpectrumEngine)
    for model in routes.values():
        # every route scores eq. (6) on the fig.-2 photon reference as built
        assert model.reference_z_star_1D_Gy == RECOVERED_Z_STAR_1D_GY[(cell_line, 0.0)]
        assert model.domain_radius_um == 0.42


def test_microdosimetry_routes_accept_the_other_cell_lines_and_reject_unknown() -> None:
    sp = SimpleBetheStoppingPower()
    routes = microdosimetry_routes("WI-38", stopping_power=sp)
    assert routes["impact/cyl"] == IMKModel.wi38_matsuya2025()
    # the site follows the preset's table-1 radius (0.50 um), not HSG's 0.42
    assert routes["sato-AMF"].engine.site == SphericalSite(diameter_um=2.0 * 0.50)
    du145 = microdosimetry_routes("du145", stopping_power=sp)
    assert du145["impact/cyl"] == IMKModel.du145_matsuya2025()
    with pytest.raises(KeyError, match="table-1 cell line"):
        microdosimetry_routes("V79", stopping_power=sp)


def test_microdosimetry_routes_require_an_explicit_stopping_power() -> None:
    """No hidden Bethe default: the caller picks the d(y) engines' stopping power."""
    with pytest.raises(TypeError, match="stopping_power"):
        microdosimetry_routes("HSG")  # type: ignore[call-arg]
