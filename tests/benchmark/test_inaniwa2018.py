"""Reconstruction of Inaniwa & Kanematsu 2018 Fig. 4 (SMK microdosimetry).

doi:10.1088/1361-6560/aabede. The figure gives the three dose-averaged specific
energies per event -- ``z_d,D`` (eq. 11), ``z*_d,D`` (eq. 12) and ``z_n,D``
(eq. 7) -- for ``Z_p = 1-10`` against kinetic energy, computed with the table-1
HSG parameters (``r_d = 0.28 um``, ``R_n = 8.1 um``, ``z0 = 66.0 Gy``).

This is the *intermediate*-quantity benchmark: it tests our radial dose + MKM
site geometry with the effect model taken out of the loop entirely, so a later
survival-curve comparison has a diagnosed microdosimetry underneath it rather
than an unattributable residual.

Two families of test here:

* **digitization integrity** -- checks on the extracted data alone, independent
  of any pyRadBio model, so a bad trace cannot masquerade as a model finding;
* **reconstruction** -- ``StochasticMKM.hsg_inaniwa2018().specific_energies``
  against the digitized curves, over each panel's resolvable range.
"""

from __future__ import annotations

import numpy as np
import pytest

from pyradbio.benchmark.inaniwa2018 import (
    CELL_MODELS,
    FIG4_PARAMETERS,
    PANELS,
    Fig4Curve,
    SurvivalCurve,
    delta2_per_point,
    dose_at_survival,
    fig4_comparison,
    fig4_model_values,
    fig4_ratios,
    get_fig4_curve,
    get_survival_curve,
    load_fig4_curves,
    load_survival_curves,
    reconstruct_panel,
    resolvable,
    survival_panels,
)
from pyradbio.core.elements import default_mass_number
from pyradbio.core.radiation import Ion
from pyradbio.models.mkm.spectrum_models import StochasticMKM
from pyradbio.track.stopping_power import SimpleBetheStoppingPower

QUANTITIES = ("z_domain", "z_star_domain", "z_nucleus")
GRID = np.logspace(np.log10(0.3), np.log10(500.0), 24)


def _ratios(model: StochasticMKM, quantity: str, Z_p: int) -> np.ndarray:
    return fig4_ratios(model, quantity, Z_p, GRID)


# --------------------------------------------------------------------------
# digitization integrity (no model involved)
# --------------------------------------------------------------------------
def test_all_thirty_fig4_series_are_present() -> None:
    """3 quantities x Z_p 1-10. The Z_p=8 panel-(c) curve is stroked at a
    different line width from every other curve in the figure, so a width-based
    extraction drops exactly one series -- this is the guard for that.
    """
    curves = load_fig4_curves()
    assert len(curves) == 30
    for quantity in QUANTITIES:
        assert sorted(c.Z_p for c in curves if c.quantity == quantity) == list(range(1, 11))


@pytest.mark.golden
def test_digitized_domain_to_nucleus_ratio_is_geometric_for_contained_tracks() -> None:
    """``z_d,D / z_n,D -> (R_n/r_d)^2 = 837`` at low energy, and falls above it.

    doi:10.1088/1361-6560/aabede sec 4.2.1 states ``z_n,D`` runs "about three
    orders of magnitude" below ``z_d,D``, reasoning from ``z ~ 1/r^2``. That holds
    only while the track is **contained in both sites**: once ``r_max`` exceeds
    the 0.28 um domain, the domain keeps only the core while the 8.1 um nucleus
    still captures much of the penumbra, so the ratio must *fall* with energy. It
    does -- 834 at 0.3 MeV/u down to ~600 by 20 MeV/u, identically for every ion.

    Panels (a) and (c) were calibrated independently and the colour->species
    assignment is a legend convention, so recovering 837 in the contained limit
    checks both at once; the fall-off is checked separately against our own model
    in `test_model_reproduces_the_domain_to_nucleus_ratio_falloff`.
    """
    expected = (FIG4_PARAMETERS["nucleus_radius_um"] / FIG4_PARAMETERS["domain_radius_um"]) ** 2
    assert expected == pytest.approx(837.0, rel=1e-3)
    for Z_p in (6, 8, 10):  # heaviest ions: both panels well clear of their axes
        a = get_fig4_curve("z_domain", Z_p)
        c = get_fig4_curve("z_nucleus", Z_p)
        contained = a.interp(0.3) / c.interp(0.3)
        assert contained == pytest.approx(expected, rel=0.02)
        assert a.interp(20.0) / c.interp(20.0) < 0.8 * contained  # delta rays escape


@pytest.mark.golden
def test_model_reproduces_the_domain_to_nucleus_ratio_falloff() -> None:
    """Our two site sizes lose track energy the way the paper's do.

    doi:10.1088/1361-6560/aabede Fig. 4(a)/(c). A pure two-site geometry check
    that cancels the absolute track normalisation entirely: only the *relative*
    delta-ray escape from a 0.28 um vs an 8.1 um site survives in the ratio.
    """
    model = StochasticMKM.hsg_inaniwa2018()
    curve_a, curve_c = get_fig4_curve("z_domain", 10), get_fig4_curve("z_nucleus", 10)
    for T, tol in ((0.3, 0.03), (3.0, 0.05), (20.0, 0.06)):
        se = model.specific_energies(Ion(10, default_mass_number(10), T))
        ours = se.z_1D_Gy / se.z_nucleus_Gy
        assert ours == pytest.approx(curve_a.interp(T) / curve_c.interp(T), rel=tol)


@pytest.mark.golden
def test_digitized_z_star_peaks_match_the_papers_stated_inflection_points() -> None:
    """ "...around T = 40 MeV/u in 20Ne, ... 12 MeV/u and 1 MeV/u in 12C and 4He".

    doi:10.1088/1361-6560/aabede sec 4.2.1, quoted for ``z*_d,D``. Checking the
    *digitized* peaks against the paper's own prose validates the x-axis
    calibration independently of the y-axis.
    """
    for Z_p, stated in ((2, 1.0), (6, 12.0), (10, 40.0)):
        T_peak, _ = get_fig4_curve("z_star_domain", Z_p).peak
        assert T_peak == pytest.approx(stated, rel=0.25)


def test_panel_resolution_floor_masks_light_ions_on_the_linear_panels() -> None:
    """Panels (a)/(c) are scaled for the heaviest ions, so light curves ride the axis."""
    protons = get_fig4_curve("z_domain", 1)
    assert not resolvable(protons.interp(100.0), "a")  # <1 Gy against a 1200 Gy scale
    assert resolvable(get_fig4_curve("z_domain", 10).interp(1.0), "a")


# --------------------------------------------------------------------------
# reconstruction
# --------------------------------------------------------------------------
@pytest.mark.golden
@pytest.mark.parametrize("Z_p", [2, 6, 10])
def test_z_star_domain_reconstructs_the_paper(Z_p: int) -> None:
    """``z*_d,D`` -- the quantity that drives ``alpha_SMK`` -- to within ~10%.

    doi:10.1088/1361-6560/aabede Fig. 4(b), with the table-1 HSG parameters. This
    is the best-resolved panel and the one that matters: it is *unbiased* across
    the full ion range (median ratio 1.00), so the residual is scatter from our
    Kiefer-Chatterjee track and stopping power against the paper's, not an offset.
    """
    r = _ratios(StochasticMKM.hsg_inaniwa2018(), "z_star_domain", Z_p)
    assert len(r) >= 15
    assert np.median(r) == pytest.approx(1.0, abs=0.05)
    assert np.abs(r - 1.0).max() < 0.12


@pytest.mark.golden
def test_z_star_domain_is_unbiased_across_all_ten_species() -> None:
    """Pooled over ``Z_p = 1-10``: no systematic offset in ``z*_d,D``."""
    model = StochasticMKM.hsg_inaniwa2018()
    pooled = np.concatenate([_ratios(model, "z_star_domain", z) for z in range(1, 11)])
    assert len(pooled) > 200
    assert np.median(pooled) == pytest.approx(1.0, abs=0.03)
    assert np.sqrt((np.log(pooled) ** 2).mean()) < 0.07  # RMS log-ratio


@pytest.mark.golden
@pytest.mark.parametrize("quantity", ["z_domain", "z_nucleus"])
def test_uncorrected_dose_means_run_low_by_a_few_percent(quantity: str) -> None:
    """``z_d,D`` and ``z_n,D`` sit ~5-7% below the paper -- and *why* they do.

    doi:10.1088/1361-6560/aabede Fig. 4(a)/(c). Unlike ``z*_d,D`` these carry a
    consistent negative offset. They are dominated by the track *core*, where the
    single-event specific energy is largest, so they are maximally sensitive to
    the residual difference between our Kiefer-Chatterjee radial dose / stopping power
    and the paper's. The saturation correction flattens exactly that region away
    (``z > z0`` saturates), which is why ``z*_d,D`` comes out unbiased. Pinned as
    a *characterisation*: if this offset moves, the track model moved.
    """
    model = StochasticMKM.hsg_inaniwa2018()
    pooled = np.concatenate([_ratios(model, quantity, z) for z in range(2, 11)])
    assert np.median(pooled) == pytest.approx(0.94, abs=0.04)
    assert np.abs(pooled - 1.0).max() < 0.16


@pytest.mark.golden
@pytest.mark.parametrize("Z_p, stated", [(2, 1.0), (6, 12.0), (10, 40.0)])
def test_model_reproduces_the_z_star_peak_positions(Z_p: int, stated: float) -> None:
    """Our ``z*_d,D`` peaks where the paper says it does (sec 4.2.1).

    doi:10.1088/1361-6560/aabede: "around T = 40 MeV/u in 20Ne, while the points
    appeared around 12 MeV/u and 1 MeV/u in 12C and 4He". The peak position is a
    pure track-structure/geometry statement -- no biology, no calibration.
    """
    model = StochasticMKM.hsg_inaniwa2018()
    T = np.logspace(-0.4, 2.3, 70)
    z = fig4_model_values(model, Z_p, T)["z_star_domain"]
    assert float(T[int(np.argmax(z))]) == pytest.approx(stated, rel=0.25)


def test_curve_helpers() -> None:
    curve: Fig4Curve = get_fig4_curve("z_star_domain", 6)
    assert curve.panel == "b"
    assert curve.full_scale_Gy == pytest.approx(PANELS["b"][1])
    assert curve.interp(curve.peak[0]) == pytest.approx(curve.peak[1], rel=1e-6)
    with pytest.raises(KeyError, match=r"no Fig\. 4 curve"):
        get_fig4_curve("nonsense", 1)


# --------------------------------------------------------------------------
# Figs. 2-3: survival curves
# --------------------------------------------------------------------------
DOSE = np.linspace(0.01, 12.0, 800)
# One LET source for both the LET -> energy inversion and the model's own track, as
# this reconstruction was validated with (the library default has moved on since).
_SP = SimpleBetheStoppingPower()


def _model_d10(cell_line: str, curve: SurvivalCurve, which: str) -> float:
    """``D_10`` of our SMK or MK curve for one panel, from its quoted LET alone.

    doi:10.1088/1361-6560/aabede sec 3.2 (monoenergetic beams at the reported
    LETs); `reconstruct_panel` inverts the LET on the high-energy root.
    """
    smk_factory, mk_factory = CELL_MODELS[cell_line]
    if which == "smk":
        s = reconstruct_panel(
            curve, DOSE, stopping_power=_SP, smk=smk_factory(stopping_power=_SP)
        ).smk
    else:
        s = reconstruct_panel(curve, DOSE, stopping_power=_SP, mk=mk_factory(stopping_power=_SP)).mk
    assert s is not None
    return dose_at_survival(DOSE, s)


def test_all_survival_series_are_present() -> None:
    """2 cell lines x 9 panels x {exp, smk, mk}."""
    curves = load_survival_curves()
    assert len(curves) == 54
    for cell in ("HSG", "V79"):
        for series in ("exp", "smk", "mk"):
            got = {c.panel for c in curves if c.cell_line == cell and c.series == series}
            assert got == set("abcdefghi"), (cell, series, sorted(got))


@pytest.mark.golden
@pytest.mark.parametrize("cell_line", ["HSG", "V79"])
def test_survival_curves_are_monotone_and_start_at_unity(cell_line: str) -> None:
    """Digitization integrity: a survival curve falls, and starts at S=1.

    Guards the extraction rather than the model -- Fig. 2 draws its legend
    *inside* panel (a) in the same colours as the data, so a bad trace shows up
    here as a non-monotone curve.
    """
    for curve in load_survival_curves():
        if curve.cell_line != cell_line or curve.series == "exp":
            continue
        s = curve.s
        assert np.all(np.diff(s) < 1e-9), f"{curve.cell_line} {curve.panel} {curve.series}"
        assert s.max() == pytest.approx(1.0, abs=0.12)


@pytest.mark.golden
@pytest.mark.parametrize("cell_line", ["HSG", "V79"])
def test_smk_reproduces_the_papers_own_survival_curves(cell_line: str) -> None:
    """End-to-end SMK against the paper's published curves, via ``D_10``.

    doi:10.1088/1361-6560/aabede figs. 2-3, solid curves. Everything is in the
    loop here: our stopping power inverts the quoted LET to an energy, our
    Kiefer-Chatterjee track and disk geometry give the three specific energies,
    and eqs. (15), (16), (24) turn those into a survival curve -- with the
    paper's own table-1 parameters and nothing fitted.
    """
    ratios = []
    for curve in survival_panels(cell_line):
        ours = _model_d10(cell_line, curve, "smk")
        ratios.append(ours / curve.dose_at_survival(0.1))
    r = np.array(ratios)
    assert len(r) == 9
    assert np.abs(r - 1.0).max() < 0.05
    assert np.median(r) == pytest.approx(1.0, abs=0.02)


@pytest.mark.golden
@pytest.mark.parametrize("cell_line", ["HSG", "V79"])
def test_mmkm_reproduces_the_papers_own_mk_curves(cell_line: str) -> None:
    """The same for the paper's MK curves, driving our ``ModifiedMKM``.

    doi:10.1088/1361-6560/aabede figs. 2-3, dashed curves, with the MK-model
    column of table 1. Looser than the SMK arm: the MK column's ``R_n`` is an
    effective radius entangled with the saturation rather than a measured nucleus,
    so it sits where our track model and the paper's differ most.
    """
    ratios = [
        _model_d10(cell_line, c, "mk")
        / get_survival_curve(cell_line, c.panel, "mk").dose_at_survival(0.1)
        for c in survival_panels(cell_line)
    ]
    assert np.abs(np.array(ratios) - 1.0).max() < 0.09


@pytest.mark.golden
@pytest.mark.parametrize("cell_line", ["HSG", "V79"])
def test_smk_fits_the_measurements_better_than_mk_overall(cell_line: str) -> None:
    """The paper's central claim, on the digitized measurements.

    doi:10.1088/1361-6560/aabede table 1 reports ``Delta^2/n`` (eq. 25) of
    0.0229 (SMK) vs 0.0452 (MK) for HSG and 0.0430 vs 0.0575 for V79 -- SMK fits
    better for both cell lines. Recomputing that metric on the points the figures
    actually draw reproduces the ordering with room to spare.

    Our absolute values are *lower* than table 1's and that is expected, not a
    discrepancy: the paper fitted ~300 points per cell line across ~9 LETs per
    ion, while figs. 2-3 draw only three LETs per ion (~50 points). This is
    scoring the same metric on a subset, so only the ordering is comparable.
    """
    smk = delta2_per_point(cell_line, "smk", "abcdefghi")
    mk = delta2_per_point(cell_line, "mk", "abcdefghi")
    assert smk < mk, (cell_line, smk, mk)


@pytest.mark.golden
@pytest.mark.parametrize("cell_line, panel", [("HSG", "i"), ("V79", "i"), ("V79", "f")])
def test_mk_fails_at_the_highest_let_where_smk_does_not(cell_line: str, panel: str) -> None:
    """Where the two models actually diverge, MK misses badly and SMK does not.

    doi:10.1088/1361-6560/aabede sec 5: the MK model "underestimated the
    cell-survival fractions for high-LET and high-dose irradiations". The
    advantage is *not* uniform -- at low LET the two models coincide and MK is
    sometimes marginally better, which is why the pooled test above is the right
    place for the overall claim. It is the top-LET panels that carry it: 654
    keV/um Ne (HSG), 528 keV/um Ne and 432 keV/um C (V79).
    """
    smk = delta2_per_point(cell_line, "smk", panel)
    mk = delta2_per_point(cell_line, "mk", panel)
    assert mk > 5.0 * smk, (cell_line, panel, smk, mk)


# --------------------------------------------------------------------------
# the reconstruction helpers reproduce the original hand-written chains bit for bit
# --------------------------------------------------------------------------
def test_fig4_model_values_matches_the_hand_written_chain() -> None:
    model = StochasticMKM.hsg_inaniwa2018()
    T = np.array([0.5, 12.0, 300.0])
    got = fig4_model_values(model, 6, T)
    se = [model.specific_energies(Ion(6, default_mass_number(6), float(t))) for t in T]
    assert got["z_domain"].tolist() == [s.z_1D_Gy for s in se]
    assert got["z_star_domain"].tolist() == [s.z_star_1D_Gy for s in se]
    assert got["z_nucleus"].tolist() == [s.z_nucleus_Gy for s in se]


def test_fig4_comparison_feeds_ratio_summary_the_fig4_ratios() -> None:
    """``ratio_summary(ours, paper, mask=mask)`` summarises exactly ``fig4_ratios``."""
    from pyradbio.benchmark.metrics import ratio_summary

    model = StochasticMKM.hsg_inaniwa2018()
    T = np.logspace(np.log10(0.3), np.log10(500.0), 8)
    ours, paper, mask = fig4_comparison(model, "z_star_domain", 6, T)
    assert ours.shape == paper.shape == mask.shape == T.shape
    assert ours.tolist() == fig4_model_values(model, 6, T)["z_star_domain"].tolist()
    r = fig4_ratios(model, "z_star_domain", 6, T)
    assert r.tolist() == (ours[mask] / paper[mask]).tolist()
    s = ratio_summary(ours, paper, mask=mask)
    assert s.n == len(r) == np.count_nonzero(mask) > 0
    assert s.median == np.median(r)
    assert s.rms_log == np.sqrt((np.log(r) ** 2).mean())
    assert s.max_abs_rel == np.abs(r - 1).max()


def test_dose_at_survival_is_the_log_interpolation_rule() -> None:
    curve = get_survival_curve("HSG", "e", "smk")
    old = float(np.interp(1.0, -np.log10(np.clip(curve.s, 1e-12, 1.0)), curve.dose))
    assert dose_at_survival(curve.dose, curve.s) == old
    assert curve.dose_at_survival() == old
    assert curve.dose_at_survival(0.01) == dose_at_survival(curve.dose, curve.s, 0.01)


def test_reconstruct_panel_matches_the_hand_written_chain() -> None:
    from pyradbio.track.stopping_power import energy_for_let_MeV_u

    curve = get_survival_curve("V79", "f", "smk")
    smk_model, mk_model = (factory() for factory in CELL_MODELS["V79"])
    got = reconstruct_panel(curve, DOSE, stopping_power=_SP, smk=smk_model, mk=mk_model)

    ion = Ion(curve.Z, curve.A, energy_for_let_MeV_u(_SP, curve.Z, curve.A, curve.let_keV_um))
    z = smk_model.specific_energies(ion)
    s_smk = smk_model.survival_from_specific_energies(
        DOSE, z_star_1D_Gy=z.z_star_1D_Gy, z_1D_Gy=z.z_1D_Gy, z_nucleus_Gy=z.z_nucleus_Gy
    )
    lq = mk_model.alpha_beta_from_z_star_1D(mk_model.specific_energies(ion).z_star_1D_Gy)
    s_mk = np.exp(-(lq.alpha * DOSE + lq.beta * DOSE**2))

    assert got.ion == ion
    assert got.smk is not None and got.mk is not None
    assert np.array_equal(got.smk, s_smk)
    assert np.array_equal(got.mk, s_mk)
    alone = reconstruct_panel(curve, DOSE, stopping_power=_SP, mk=mk_model)
    assert alone.smk is None
    assert alone.mk is not None and np.array_equal(alone.mk, s_mk)


def test_delta2_per_point_pools_the_points_of_all_panels() -> None:
    def per_panel(panel: str) -> np.ndarray:
        exp = get_survival_curve("HSG", panel, "exp")
        m = get_survival_curve("HSG", panel, "mk")
        return (np.log10(exp.s) - np.interp(exp.dose, m.dose, np.log10(m.s))) ** 2

    assert delta2_per_point("HSG", "mk", "i") == per_panel("i").mean()
    pooled = list(per_panel("a")) + list(per_panel("b"))
    assert delta2_per_point("HSG", "mk", "ab") == np.mean(pooled)


@pytest.mark.slow
def test_z_d_bias_is_the_pooled_panel_a_offset_with_simple_bethe() -> None:
    """``Z_D_BIAS`` is ``1 / median(ours / paper)`` of Fig. 4(a), rounded.

    doi:10.1088/1361-6560/aabede Fig. 4(a), pooled over ``Z_p = 1-10`` with
    ``SimpleBetheStoppingPower`` on the model -- the measurement
    docs/references/SMK_Inaniwa2018.md records (median 0.933, 134 points).
    """
    from pyradbio.benchmark.inaniwa2018 import Z_D_BIAS

    model = StochasticMKM.hsg_inaniwa2018(stopping_power=_SP)
    pooled = np.concatenate([_ratios(model, "z_domain", z) for z in range(1, 11)])
    assert len(pooled) == 134
    assert 1.0 / np.median(pooled) == pytest.approx(Z_D_BIAS, abs=0.005)
