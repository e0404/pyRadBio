"""Elsasser & Scholz 2007 LEM II benchmark reference data.

Reference: Elsasser, T. & Scholz, M. 2007, Radiat. Res. 167(3):319-329,
doi:10.1667/RR0467.1.
"""

from __future__ import annotations

import itertools

import numpy as np
import pytest

from pyradbio import Ion
from pyradbio.benchmark import elsasser2007 as es
from pyradbio.benchmark.sweeps import sweep_energies
from pyradbio.core.cell import CellTarget
from pyradbio.core.photon import PhotonParameters
from pyradbio.models.lem.lem1 import LEM1
from pyradbio.models.lem.lem2 import LEM2, ClusterModel, ClusterResponse
from pyradbio.track.scholz_kraft import LEM2_RMIN_NM, R_MAX_GAMMA_UM, ScholzKraftRadialDose
from pyradbio.track.stopping_power import SimpleBetheStoppingPower


def test_curve_loader_groups_digitized_points() -> None:
    curves = es.load_curves()

    assert len(curves) >= 4
    assert all(len(c.x) == len(c.y) >= 2 for c in curves)
    assert all(c.doi == es.ELSASSER2007_DOI for c in curves)
    # x-values are stored sorted ascending (grouped-and-sorted loader contract).
    assert all(list(c.x) == sorted(c.x) for c in curves)


def test_get_curve_rejects_ambiguous_and_empty_filters() -> None:
    with pytest.raises(LookupError):
        es.get_curve("fig9_rbe10_vs_let")  # two calc lines + one data set
    with pytest.raises(LookupError):
        es.get_curve("no_such_figure")


def test_figure_inputs_carry_the_loadbearing_dt_refit() -> None:
    """The LEM II refit *reduces* D_t for every cell line (Fig. 6 thin dashed:
    both modifications without the reduction overshoot the RBE)."""
    for key in ("fig6_rbealpha_vs_let", "fig8_cho", "fig11_hsg"):
        inputs = es.FIGURE_INPUTS[key]
        assert inputs.D_t_new < inputs.D_t_old
    v79 = es.FIGURE_INPUTS["fig6_rbealpha_vs_let"]
    assert (v79.alpha_x, v79.beta_x) == (0.093, 0.026)
    assert (v79.D_t_old, v79.D_t_new) == (30.0, 7.5)
    cho = es.FIGURE_INPUTS["fig8_cho"]
    assert (cho.D_t_old, cho.D_t_new) == (40.0, 9.5)
    hsg = es.FIGURE_INPUTS["fig11_hsg"]
    assert (hsg.D_t_old, hsg.D_t_new) == (30.0, 6.0)


def test_fig8_carries_both_cell_lines_with_error_bars() -> None:
    """Fig. 8 draws V79 *and* CHO, each with two model lines plus Weyrather data.

    The experimental symbols carry error bars, so ``y_err`` is populated (the
    other figures' data sets have none and leave it ``None``).
    """
    for line in ("V79", "CHO"):
        for variant in ("original_lem", "modified_lem"):
            curve = es.get_curve(
                "fig8_rbealpha_vs_let", cell_line=line, series="calc", variant=variant
            )
            assert len(curve.x) > 30
            assert curve.y_err is None  # model lines have no uncertainty

        data = es.get_curve("fig8_rbealpha_vs_let", cell_line=line, series="data")
        assert data.y_err is not None
        assert len(data.y_err) == len(data.y)
        # Every bar is a real, positive uncertainty below its own mean.
        assert all(0.0 < e < m for e, m in zip(data.y_err, data.y, strict=True))


def test_fig8_v79_lines_agree_with_fig6() -> None:
    """Fig. 8's V79 lines ARE Fig. 6's ``lem_old``/``lem_new``, traced separately.

    The panels share V79's inputs, so the two independent hand digitizations must
    agree. This is the cross-figure consistency probe (the LEM I analogue caught a
    real ~15% paper inconsistency); it also guards the digitizations themselves.
    """
    for fig6_variant, fig8_variant in (("lem_old", "original_lem"), ("lem_new", "modified_lem")):
        c6 = es.get_curve("fig6_rbealpha_vs_let", series="calc", variant=fig6_variant)
        c8 = es.get_curve(
            "fig8_rbealpha_vs_let", cell_line="V79", series="calc", variant=fig8_variant
        )
        common = np.array([x for x in c6.x if min(c8.x) <= x <= max(c8.x)])
        on6 = np.interp(common, c6.x, c6.y)
        on8 = np.interp(common, c8.x, c8.y)
        rel_rms = float(np.sqrt(np.mean(((on8 - on6) / on6) ** 2)))
        assert rel_rms < 0.03, f"{fig6_variant}: fig6 vs fig8 disagree by {rel_rms:.1%}"


def test_fig10_and_fig12_are_survival_curves_in_dose_and_S() -> None:
    """Figs. 10 & 12 store survival curves: x = dose (Gy), y = surviving fraction.

    Every model curve starts near S = 1 at low dose and falls monotonically.
    """
    for figure, energies in (
        ("fig10_survival_v79", (4.2, 11.0, 76.9, 266.4)),
        ("fig12_survival_xrs", (4.2, 11.0, 76.9)),
    ):
        for energy in energies:
            for variant in (
                ("modified_lem", "original_lem") if "fig10" in figure else ("modified_lem",)
            ):
                c = es.get_curve(figure, series="calc", variant=variant, energy_MeV_u=energy)
                assert (c.x_quantity, c.x_unit) == ("dose", "Gy")
                assert (c.y_quantity, c.y_unit) == ("survival", "1")
                assert max(c.y) <= 1.001 and min(c.y) < 0.1  # starts at 1, spans down
            data = es.get_curve(figure, series="data", energy_MeV_u=energy)
            assert len(data.x) >= 5


def test_fig12_xray_reference_is_pure_exponential_alpha_1p7() -> None:
    """XRS's x-ray curve is single-parameter: ln S = -alpha_x * D with alpha_x = 1.7.

    This is what makes Fig. 12 an isolated test of the cluster eta -- a shoulderless
    photon response leaves eta as the only source of ion RBE.
    """
    xray = es.get_curve("fig12_survival_xrs", series="calc", variant="xray")
    x, y = np.array(xray.x), np.array(xray.y)
    m = (x > 0.05) & (y > 1e-3)
    slope = float(np.polyfit(x[m], np.log(y[m]), 1)[0])
    assert slope == pytest.approx(-1.7, abs=0.05)
    xrs = es.FIGURE_INPUTS["fig12_xrs"]
    assert (xrs.alpha_x, xrs.beta_x) == (1.7, 0.0)
    assert xrs.nucleus_radius_um == 5.5  # a fourth cell line, distinct radius


@pytest.mark.golden
def test_fig12_xrs_rbe_is_the_cluster_extension_only() -> None:
    """XRS ion RBE comes entirely from the cluster eta (Elsasser & Scholz 2007 Fig. 12).

    XRS's photon curve is a pure exponential (beta_x = 0), so it has no shoulder for
    an amorphous-track LEM to exploit: without eta the model only ever *loses* ground
    to overkill saturation (LEM I gives alpha ~ 1.4 < the photon 1.7 here). LEM II's
    cluster eta is the whole reason alpha climbs back above the photon value, and it
    reproduces the digitized 11 MeV/u model line at alpha ~ 2.6 (RBE ~ 1.5).
    """
    xrs = es.FIGURE_INPUTS["fig12_xrs"]
    sp = SimpleBetheStoppingPower()
    cell = CellTarget(nucleus_radius_um=xrs.nucleus_radius_um, effective_size_factor=1.0)
    photon = PhotonParameters(alpha_x=xrs.alpha_x, beta_x=xrs.beta_x, D_t=xrs.D_t_new)
    ion = Ion(Z=6, A=12, energy_MeV_u=11.0)

    lem1 = LEM1(photon=photon, cell=cell, stopping_power=sp)
    lem2 = LEM2(photon=photon, cell=cell, stopping_power=sp, cluster=ClusterModel())

    # No shoulder + no eta -> no enhancement, only overkill: LEM I sits below alpha_x.
    assert lem1.alpha_beta(ion).alpha < xrs.alpha_x
    # The cluster eta lifts it to the paper's ~2.6, well above the un-enhanced LEM I.
    assert lem2.alpha_beta(ion).alpha == pytest.approx(2.6, rel=0.10)
    assert lem2.alpha_beta(ion).alpha > 1.6 * lem1.alpha_beta(ion).alpha


def test_fig4_is_absolute_radial_dose_in_cm_and_gy() -> None:
    """Fig. 4 compares D(r) to measurements in absolute units: r in cm, dose in Gy.

    The unit reading is forced by the curves themselves, not by the axis labels
    alone: LEM I's 10 nm core = 1e-6 cm, and the "original" lines plateau below
    exactly that, then fall.
    """
    for particle, energy in (("oxygen", 2.57), ("neon", 377.0)):
        for variant in ("original", "modified"):
            c = es.get_curve(
                "fig4_radial_dose_vs_exp", particle=particle, series="calc", variant=variant
            )
            assert (c.x_quantity, c.x_unit) == ("radius", "cm")
            assert (c.y_quantity, c.y_unit) == ("dose", "Gy")
            assert c.energy_MeV_u == pytest.approx(energy)
            assert c.cell_line == ""  # a track-structure panel: no cell
        data = es.get_curve("fig4_radial_dose_vs_exp", particle=particle, series="data")
        assert len(data.x) >= 10

    # The original (LEM I, r_min = 10 nm = 1e-6 cm) curves are flat inside the core
    # and falling outside it -- the check that pins the cm reading.
    orig = es.get_curve(
        "fig4_radial_dose_vs_exp", particle="oxygen", series="calc", variant="original"
    )
    x, y = np.array(orig.x), np.array(orig.y)
    core, tail = y[x < 5e-7], y[x > 2e-6]
    assert core.max() / core.min() < 1.1, "core is not flat -> r is not in cm"
    assert tail.max() / tail.min() > 100.0, "no 1/r^2 tail -> r is not in cm"


@pytest.mark.golden
def test_fig4_modified_line_is_the_unconvolved_profile() -> None:
    """Fig. 4's "modified" line is the r_min = 0.3 nm profile WITHOUT the sigma =
    4 nm radical-diffusion Gaussian that Fig. 5's ``D~(r)`` carries.

    Reference: Elsasser & Scholz 2007 Fig. 4a (2.57 MeV/u 16O). The raster shows a
    flat plateau to a *sharp* knee at ~3e-8 cm and a straight 1/r^2 line after it;
    a Gaussian would round the knee and drop the plateau ~50x. So the panel must be
    compared against ``radical_diffusion=False`` -- using ``lem2()`` here looks like
    a 50x core deficit that does not exist.
    """
    sp = SimpleBetheStoppingPower()
    ion = Ion(Z=8, A=16, energy_MeV_u=2.57)
    ref = es.get_curve(
        "fig4_radial_dose_vs_exp", particle="oxygen", series="calc", variant="modified"
    )
    x, y = np.array(ref.x), np.array(ref.y)

    unconvolved = ScholzKraftRadialDose(
        r_min_nm=LEM2_RMIN_NM, radical_diffusion=False, stopping_power=sp
    )
    convolved = ScholzKraftRadialDose.lem2(sp)

    def rms_log10(model: ScholzKraftRadialDose) -> float:
        ours = np.array([model.local_dose_Gy(float(r * 1e4), ion) for r in x])
        m = (ours > 0) & (y > 0)
        return float(np.sqrt(np.mean((np.log10(ours[m]) - np.log10(y[m])) ** 2)))

    # The un-convolved profile IS the panel (~3% in dose); the convolved one is not.
    assert rms_log10(unconvolved) < 0.05
    assert rms_log10(convolved) > 0.5


def test_fig11_carries_two_ion_species_for_hsg() -> None:
    """Fig. 11 is the only panel with a second ion (helium) and a third cell line.

    Its y-axis is the initial slope ``alpha`` in Gy^-1, NOT ``RBE_alpha``, and the
    panel draws no error bars.
    """
    for particle in ("helium", "carbon"):
        for variant in ("original_lem", "modified_lem"):
            curve = es.get_curve(
                "fig11_alpha_vs_let", particle=particle, series="calc", variant=variant
            )
            assert curve.cell_line == "HSG"
            assert (curve.y_quantity, curve.y_unit) == ("alpha", "1/Gy")
            assert curve.y_err is None
        data = es.get_curve("fig11_alpha_vs_let", particle=particle, series="data")
        assert data.y_err is None  # Fig. 11 draws no error bars (checked on the raster)
        assert len(data.x) >= 10

    # The particle filter must actually discriminate, or He/C would silently merge.
    assert len(es.iter_curves(figure="fig11_alpha_vs_let")) == 6
    assert len(es.iter_curves(figure="fig11_alpha_vs_let", particle="helium")) == 3
    with pytest.raises(LookupError):
        es.get_curve("fig11_alpha_vs_let", series="calc", variant="modified_lem")  # He + C


def test_fig11_let_axis_spans_the_panel() -> None:
    """Fig. 11's LET axis runs ~15..1000 keV/um, not ~2..1000.

    Its ticks read 2,3..9, 10^2, 2,3..9, 10^3, so the leftmost labelled tick is
    2e1 = 20 -- easy to anchor at 2 instead, which log-compresses every point
    toward the left. A first hand-export did exactly that (1.78..333) and our own
    LEM II caught it: 29-58% relRMS against the compressed axis vs 7-14% against
    the corrected one, with the carbon peak at 70 keV/um where the model puts it
    near 196. Cheap guard against a re-import of the compressed data.
    """
    for curve in es.iter_curves(figure="fig11_alpha_vs_let"):
        assert min(curve.x) >= 15.0, (
            f"{curve.particle} {curve.series}: LET starts at {min(curve.x)}"
        )
        assert max(curve.x) <= 1000.0
    # Carbon reaches the panel's high-LET end; helium stops near 70 keV/um.
    assert max(es.get_curve("fig11_alpha_vs_let", particle="carbon", series="data").x) > 400.0
    assert max(es.get_curve("fig11_alpha_vs_let", particle="helium", series="data").x) < 100.0


def test_fig11_y_axis_is_the_recalibrated_one() -> None:
    """Fig. 11's lowest y tick is alpha = 0.25, not 0 -- guard the axis calibration.

    The panel's ticks are 0.25 apart, labelled every 0.5 from 0.5 to 3.0, and its
    lowest tick is an unlabelled 0.25. A first export anchored that tick as
    ``alpha = 0`` with the top (3.0) correct, so every value came out as
    ``1.0909*true - 0.2727``; the CSVs carry the values on the correct axis.
    Uncorrected, our LEM II reads +10.7%/+6.7% high here; corrected, -2.5%/-3.4%.

    Two independent tells, so a re-import of the raw export cannot pass both: the
    lowest carbon point must clear 0.25 by a good margin, and the low-LET ends of
    the two carbon model lines must not fall below the axis floor.
    """
    data = es.get_curve("fig11_alpha_vs_let", particle="carbon", series="data")
    assert min(data.y) == pytest.approx(0.431, abs=0.01), (
        "carbon's lowest alpha should be Furusawa's 0.429, not the uncorrected 0.197"
    )
    assert min(es.get_curve("fig11_alpha_vs_let", particle="helium", series="data").y) == (
        pytest.approx(0.540, abs=0.01)
    )
    for curve in es.iter_curves(figure="fig11_alpha_vs_let"):
        assert min(curve.y) > 0.25, f"{curve.particle} {curve.variant}: below the axis floor"


def test_figure_inputs_use_the_quoted_radius_directly() -> None:
    """The paper's nucleus radius is used as-is, NOT shrunk by ``CellTarget``'s
    0.8 ``effective_size_factor`` default.

    Letting that default through shrinks the nuclear area by 0.64, inflating
    ``D_1 = LET/(rho*A)`` by 1.56x, so overkill sets in too early: every Fig. 6
    variant then peaks ~1.25x low in LET and 10-20% low in height, which is
    exactly what once looked like a LEM II cluster deficit. Pinned because the
    default is silent -- nothing else fails when it leaks through.
    """
    for inputs in es.FIGURE_INPUTS.values():
        assert inputs.effective_size_factor == pytest.approx(1.0)


@pytest.mark.golden
def test_fig6_lem_old_peak_needs_the_unshrunk_radius() -> None:
    """LEM I on Fig. 6 reproduces the paper's RBE_alpha peak only at the quoted
    radius (Elsasser & Scholz 2007 Fig. 6, thick dashed: peak ~7.8 at ~228 keV/um).

    Guards the ``effective_size_factor`` regression end-to-end rather than by
    inspection: at the 0.8 default this peaks at 6.9 near 153 keV/um.
    """
    v79 = es.FIGURE_INPUTS["fig6_rbealpha_vs_let"]
    sp = SimpleBetheStoppingPower()
    model = LEM1(
        photon=PhotonParameters(alpha_x=v79.alpha_x, beta_x=v79.beta_x, D_t=v79.D_t_old),
        cell=CellTarget(
            nucleus_radius_um=v79.nucleus_radius_um,
            effective_size_factor=v79.effective_size_factor,
        ),
        stopping_power=sp,
    )
    sweep = sweep_energies(
        model, 6, 12, np.geomspace(2.0, 400.0, 60), stopping_power=sp, on_error="raise"
    )
    lets, rbes = sweep.let_keV_um, sweep.rbe_alpha(v79.alpha_x)
    peak = int(np.argmax(rbes))

    assert rbes[peak] == pytest.approx(7.8, rel=0.06)
    assert lets[peak] == pytest.approx(228.0, rel=0.10)


@pytest.mark.golden
def test_fig8_cho_modified_lem_matches_the_paper() -> None:
    """LEM II reproduces Fig. 8's CHO 'modified LEM' line (peak ~5.46 @ ~274 keV/um).

    CHO is the sharpest geometry check in the paper: Fig. 8 draws it beside V79
    from the same code, and the two differ mainly by r_nucleus (4.7 vs 3.8 um). If
    ``effective_size_factor`` were reapplied to the paper's quoted radius, this
    line would demand r ~ 5.9 um to fit; at the quoted 4.7 it lands within a few %.
    """
    cho = es.FIGURE_INPUTS["fig8_cho"]
    sp = SimpleBetheStoppingPower()
    model = LEM2(
        photon=PhotonParameters(alpha_x=cho.alpha_x, beta_x=cho.beta_x, D_t=cho.D_t_new),
        cell=CellTarget(
            nucleus_radius_um=cho.nucleus_radius_um,
            effective_size_factor=cho.effective_size_factor,
        ),
        stopping_power=sp,
        cluster=ClusterModel(),
    )
    ref = es.get_curve(
        "fig8_rbealpha_vs_let", cell_line="CHO", series="calc", variant="modified_lem"
    )

    sweep = sweep_energies(
        model,
        6,
        12,
        np.geomspace(1.5, 350.0, 48),
        stopping_power=sp,
        on_error="raise",
        sort_by_let=True,
    )
    lets, rbes = sweep.let_keV_um, sweep.rbe_alpha(cho.alpha_x)

    assert rbes.max() == pytest.approx(max(ref.y), rel=0.10)
    # Residual against the whole digitized line, not just its peak.
    px = np.array([x for x in ref.x if lets.min() <= x <= lets.max()])
    py = np.interp(px, ref.x, ref.y)
    fit = np.interp(np.log(px), np.log(lets), rbes)
    assert float(np.sqrt(np.mean(((fit - py) / py) ** 2))) < 0.10


def test_fig2_eta_onset_and_peak_match_the_paper() -> None:
    """eta ~ 1 below the ~1e3 Gy onset and peaks near 5e5 Gy (caption)."""
    eta = es.get_curve("fig2_eta_vs_dose")

    x, y = np.array(eta.x), np.array(eta.y)
    assert eta.x_unit == "Gy" and eta.y_quantity == "eta"
    assert y[0] == pytest.approx(1.0, abs=0.05)  # no clustering at 1e2 Gy
    assert y[np.argmin(np.abs(x - 1.0e3))] == pytest.approx(1.0, abs=0.2)
    peak_dose = x[np.argmax(y)]
    assert 3.0e5 <= peak_dose <= 7.0e5  # caption: "about 5 x 10^5 Gy"
    assert y.max() == pytest.approx(14.79, abs=0.15)


def test_fig9_modified_lem_exceeds_original_at_the_rbe_maximum() -> None:
    """The key LEM II feature: the cluster extension raises the RBE maximum
    while leaving the low-LET end at/below the original model (Fig. 9)."""
    mod = es.get_curve("fig9_rbe10_vs_let", series="calc", variant="modified_lem")
    orig = es.get_curve("fig9_rbe10_vs_let", series="calc", variant="original_lem")

    assert max(mod.y) > max(orig.y)  # 3.00 vs 2.44
    assert mod.y[0] < orig.y[0]  # the curves cross near 75 keV/um
    # Both peak in the Bragg-peak LET region.
    assert 150.0 <= mod.x[int(np.argmax(mod.y))] <= 300.0
    assert 150.0 <= orig.x[int(np.argmax(orig.y))] <= 300.0


def test_fig9_experimental_squares_bracket_the_model_curves() -> None:
    data = es.get_curve("fig9_rbe10_vs_let", series="data")

    assert len(data.x) == 7
    assert data.cell_line == "V79"
    assert min(data.x) == pytest.approx(13.8, abs=1.0)
    assert max(data.x) == pytest.approx(480.0, abs=20.0)
    assert 1.0 < min(data.y) and max(data.y) < 4.0


def test_fig3_original_tail_pins_cho_and_dt40_not_the_captions_dt30() -> None:
    """Fig. 3's dashed tail is the un-enhanced LEM I line, so it fixes D_t exactly.

    It is straight in linear space (N = s_max*D + c), and ``s_max = alpha + 2*beta*D_t``.
    The digitized slope gives D_t ~ 40 Gy -- CHO's old value -- while the caption's
    30 Gy would need s_max ~ 1.48. This is the evidence that the caption is wrong.
    """
    original = es.get_curve("fig3a_dose_effect", variant="original")
    x, y = np.array(original.x), np.array(original.y)

    deep = x > 1.0e4  # well past D_t, where the tail is asymptotically straight
    s_max, _ = np.polyfit(x[deep], y[deep], 1)
    assert s_max == pytest.approx(1.87, rel=0.05)

    cho = es.FIGURE_INPUTS["fig8_cho"]
    d_t = (s_max - cho.alpha_x) / (2.0 * cho.beta_x)
    assert d_t == pytest.approx(es.FIG3_D_T_OLD_GY, rel=0.06)  # ~40 Gy
    assert d_t > 35.0  # decisively not the caption's 30 Gy
    assert es.FIG3_CELL_LINE == "CHO"


def test_fig3_lq_shoulder_identifies_cho_not_v79() -> None:
    """The sub-D_t branch is the bare LQ, so it reads off alpha and beta directly."""
    shoulder = es.get_curve("fig3a_dose_effect", variant="lq_shoulder")
    x, y = np.array(shoulder.x), np.array(shoulder.y)

    (alpha, beta), *_ = np.linalg.lstsq(np.vstack([x, x**2]).T, y, rcond=None)
    cho = es.FIGURE_INPUTS["fig8_cho"]
    v79 = es.FIGURE_INPUTS["fig6_rbealpha_vs_let"]
    assert alpha == pytest.approx(cho.alpha_x, rel=0.10)
    assert beta == pytest.approx(cho.beta_x, rel=0.10)
    # V79 is not a near miss -- it is out by more than 100% in alpha.
    assert abs(alpha - v79.alpha_x) / v79.alpha_x > 1.0


def test_lem2_reproduces_digitized_fig3_cluster_tail() -> None:
    """Eq. 10 must reproduce Fig. 3's clustered tail when fed the paper's own eta.

    This pins the *functional form*: feeding the paper's digitized Fig. 2 eta into
    ``photon_log_kill`` removes our clustering rule from the comparison, so a
    failure here means the formula is wrong. Reading eq. 10 as an
    integral of an enhanced local slope, ``s_max*integral eta dD'``, misses this
    by ~23% RMS; the literal product form lands within a few percent.
    """
    cho = es.FIGURE_INPUTS["fig8_cho"]
    model = LEM2(
        photon=PhotonParameters(alpha_x=cho.alpha_x, beta_x=cho.beta_x, D_t=es.FIG3_D_T_OLD_GY),
        cell=CellTarget(nucleus_radius_um=cho.nucleus_radius_um),
        cluster=es.digitized_eta_cluster(),
    )

    curve = es.get_curve("fig3a_dose_effect", variant="cluster")
    x, y = np.array(curve.x), np.array(curve.y)
    predicted = np.array([model.photon_log_kill(float(d)) for d in x])
    rel = (predicted - y) / y
    assert np.sqrt(np.mean(rel**2)) < 0.08
    assert abs(np.mean(rel)) < 0.05


@pytest.mark.xfail(
    reason=(
        "our fixed-h-bin clustering rule is not the paper's: eta_peak 13.40 vs 14.79. "
        "The peak height is genome-size independent (eta-1 = 41.667*q(mu)/mu), so this "
        "is a rule mismatch, not a parameter fit. Two independent digitizations (hand "
        "and raster auto-trace) agree to 0.9% RMS, so it is not a digitization artifact. "
        "Nor is it ours: Friedrich, Durante & Scholz 2015 (doi:10.1093/rpd/ncv147) "
        "recompute eta from the same alpha_SSB/alpha_DSB/t with a stricter closed-form "
        "rule and peak at 13.51 -- on top of ours, 8.6% below Fig. 2 -- and the two "
        "agree to 3.0% RMS over the <=1e5 Gy a track can actually reach. Fig. 2 is the "
        "outlier. See the module docstring."
    ),
    strict=True,
)
def test_cluster_model_reproduces_digitized_fig2() -> None:
    eta = es.get_curve("fig2_eta_vs_dose")
    model = ClusterModel()

    predicted = np.array([model.eta(float(d)) for d in eta.x])
    assert predicted == pytest.approx(np.array(eta.y), rel=0.05)


# --- Fig. 7: stagger-size dependence -------------------------------------------


def test_figure7_stagger_curves_are_loaded() -> None:
    """All five stagger variants plus both LEM references."""
    variants = {c.variant for c in es.load_curves() if c.figure == "fig7_stagger_vs_let"}
    assert variants == {
        "cluster_h10",
        "cluster_h20",
        "cluster_h35",
        "cluster_h10_dt11",
        "lem_new",
        "lem_old",
    }


def test_figure7_repeats_figure6_reference_lines() -> None:
    """Fig. 7 redraws Fig. 6's two LEM lines, and they were traced separately.

    A cross-digitization check that costs nothing: the same two curves appear in
    both panels, so tracing them twice bounds the digitization error of *both*
    figures without any model in the loop.
    """
    for variant in ("lem_new", "lem_old"):
        a = es.get_curve("fig7_stagger_vs_let", variant=variant)
        b = es.get_curve("fig6_rbealpha_vs_let", variant=variant)
        ax, ay = np.array(a.x), np.array(a.y)
        bx, by = np.array(b.x), np.array(b.y)
        grid = np.geomspace(max(ax.min(), bx.min()), min(ax.max(), bx.max()), 60)
        ratio = np.interp(np.log(grid), np.log(ax), ay) / np.interp(np.log(grid), np.log(bx), by)
        assert np.sqrt(np.mean((ratio - 1.0) ** 2)) < 0.01, variant


def test_figure7_max_over_min_rbe_rises_with_stagger_size() -> None:
    """The paper's own claim for Fig. 7 (p. 325): a larger stagger window widens
    the RBE_alpha range, since more SSB pairs cluster at high local dose."""
    ratios = [
        max(es.get_curve("fig7_stagger_vs_let", variant=v).y)
        / min(es.get_curve("fig7_stagger_vs_let", variant=v).y)
        for v in ("cluster_h10", "cluster_h20", "lem_new", "cluster_h35")  # h = 10, 20, 25, 35
    ]
    assert ratios == sorted(ratios)
    # ... and every cluster variant beats the un-clustered reference.
    lem_old = es.get_curve("fig7_stagger_vs_let", variant="lem_old")
    assert min(ratios) > max(lem_old.y) / min(lem_old.y)


# --- Which method produced the paper's RBE_alpha curves? ------------------------


@pytest.mark.golden
def test_golden_paper_curves_follow_the_full_average_not_its_stated_equation_8() -> None:
    """Elsasser 2007's figures do not behave like Elsasser 2007's eq. (8).

    The paper states one procedure for every RBE_alpha curve in Figs. 6 onward
    (p. 323): "we exploit the low-dose approximation and simulate only a single
    ion traversal at the center of the cell nucleus [...] Using Eq. (8), we obtain
    the desired value alpha_I", i.e. ``alpha_I = (1 - S_1)/D_1``.

    That shortcut drops every track that misses the nucleus while still depositing
    dose inside it, so its error is not a constant -- it is governed by
    ``r_max/R_eff``, which sweeps three decades across a single panel (84 at
    14 keV/um down to 0.1 at 500). That gives a falsifiable prediction. If the
    paper's lines were made with eq. (8), our eq.-(8) residual would be flat in
    ``r_max/R_eff``. It is not: it grows monotonically from -0.1% to -10.2% as the
    track outgrows the nucleus, while the full impact-parameter average stays flat
    to ~2% over the same range.

    Measured on Fig. 6's ``lem_old`` -- the simplest line in the paper: no
    clusters, no radical-diffusion convolution, LQ photon input, every parameter
    stated in the caption. So this is not a statement about the cluster extension.
    """
    v79 = es.FIGURE_INPUTS["fig6_rbealpha_vs_let"]
    sp = SimpleBetheStoppingPower()
    cell = CellTarget(
        nucleus_radius_um=v79.nucleus_radius_um,
        effective_size_factor=v79.effective_size_factor,
    )
    radial = ScholzKraftRadialDose(r_min_nm=10.0, stopping_power=sp, r_max_gamma_um=R_MAX_GAMMA_UM)
    bands = es.paper_method_bands(
        PhotonParameters(alpha_x=v79.alpha_x, beta_x=v79.beta_x, D_t=v79.D_t_old),
        cell,
        radial,
        stopping_power=sp,
        energies_MeV_u=np.geomspace(1.5, 350.0, 60),
        reference=es.get_curve("fig6_rbealpha_vs_let", variant="lem_old"),
    )
    # paper_method_bands skips a band with < 2 reference points: pin the count so
    # eq8[0] / eq8[-1] below are always the lowest / highest of the 5 bands.
    assert len(bands) == len(es.METHOD_BANDS_KEV_UM)

    eq8 = [b.bias_eq8 for b in bands]  # ordered low LET (big r_max) -> high LET
    exact = [b.bias_exact for b in bands]

    # eq. (8) is worst where the track most outgrows the nucleus, and improves
    # monotonically as r_max/R_eff falls.
    assert eq8[0] < -0.07, f"expected a large low-LET deficit, got {eq8[0]:.1%}"
    assert all(a < b for a, b in itertools.pairwise(eq8)), eq8
    assert abs(eq8[-1]) < 0.02, f"expected eq. (8) to be fine at high LET, got {eq8[-1]:.1%}"

    # The full average carries no such trend: flat, and far better where they differ.
    assert max(abs(b) for b in exact) < 0.02, exact
    assert abs(exact[0]) < 0.25 * abs(eq8[0])


def test_fig9_model_lines_fall_monotonically_past_the_rbe_maximum() -> None:
    """Both Fig. 9 model lines are single-peaked: no turn-up in the overkill tail.

    Past the RBE maximum a LEM curve can only thin down -- each further increment
    of LET is spent on already-dead subvolumes -- so a rising tail is a tracing
    artifact, not physics. A raster auto-trace of the `original_lem` line rose +1.2%
    at 428 keV/um before falling again, while the bundled hand trace is smooth (the
    two traces otherwise agree to 1.3% / 1.9% RMS, which is what bounds this
    panel's digitization error).
    """
    for variant in ("modified_lem", "original_lem"):
        curve = es.get_curve("fig9_rbe10_vs_let", series="calc", variant=variant)
        y = np.array(curve.y)
        tail = y[int(np.argmax(y)) :]
        assert np.all(np.diff(tail) <= 0.0), (
            f"{variant} rises after its peak at "
            f"{np.array(curve.x)[int(np.argmax(y)) + int(np.argmax(np.diff(tail)))]:.0f} keV/um"
        )
        # ... and the peak is interior, not an endpoint (which would mean the
        # trace stopped before the panel did).
        assert 0 < int(np.argmax(y)) < len(y) - 1


# --- reconstruction machinery (validation/elsasser2007_lem2.py) -----------------


def _inline_paper_eta(eta_x: np.ndarray, eta_y: np.ndarray, dose_Gy: float) -> float:
    """The digitized-eta rule as the reconstruction script writes it inline."""
    if dose_Gy <= eta_x[0]:
        return 1.0
    return float(np.interp(np.log10(dose_Gy), np.log10(eta_x), eta_y))


def test_digitized_eta_cluster_is_the_inline_rule_bit_for_bit() -> None:
    """``digitized_eta_cluster`` reproduces the script's inline ``PaperEtaCluster``.

    Same floor (1 at and below the first digitized dose), same log-dose
    interpolation, same peak; ``eta_array`` is the scalar rule element-wise, and
    the object is a ``ClusterResponse``, so it plugs into ``LEM2(cluster=...)``.
    """
    fig2 = es.get_curve("fig2_eta_vs_dose")
    eta_x, eta_y = np.array(fig2.x), np.array(fig2.y)
    cluster = es.digitized_eta_cluster()
    doses = np.concatenate([np.logspace(0.0, 8.0, 801), eta_x, eta_x * (1.0 + 1e-12)])

    scalar = np.array([cluster.eta(float(d)) for d in doses])
    inline = np.array([_inline_paper_eta(eta_x, eta_y, float(d)) for d in doses])
    assert np.array_equal(scalar, inline)
    assert np.array_equal(cluster.eta_array(doses), scalar)
    assert cluster.eta_max == float(eta_y.max())
    assert isinstance(cluster, ClusterResponse)
    log_dose, eta_nodes = cluster.kernel_eta_table()
    assert np.array_equal(log_dose, np.log10(eta_x))
    assert np.array_equal(eta_nodes, eta_y)


def test_digitized_eta_cluster_rejects_a_malformed_table() -> None:
    with pytest.raises(ValueError, match="equal-length"):
        es.DigitizedEtaCluster(dose_Gy=np.array([1.0, 2.0]), eta_values=np.array([1.0]))
    with pytest.raises(ValueError, match="strictly increasing"):
        es.DigitizedEtaCluster(dose_Gy=np.array([2.0, 1.0]), eta_values=np.array([1.0, 2.0]))


def test_paper_method_bands_is_the_inline_diagnostic_bit_for_bit() -> None:
    """``paper_method_bands`` reproduces the reconstruction's inline METHOD_BANDS loop.

    The script swept carbon on its energy grid, interpolated ``RBE_alpha`` in log
    LET (NaN outside), and took the band means and ``r_max/R_eff`` at the band's
    geometric-mean LET. A band with fewer than two reference points is skipped.
    """
    v79 = es.FIGURE_INPUTS["fig6_rbealpha_vs_let"]
    sp = SimpleBetheStoppingPower()
    cell = v79.cell_target()
    photon = v79.photon("old")
    radial = ScholzKraftRadialDose(r_min_nm=10.0, stopping_power=sp, r_max_gamma_um=R_MAX_GAMMA_UM)
    ref = es.get_curve("fig6_rbealpha_vs_let", variant="lem_old")
    energies = np.geomspace(1.5, 350.0, 12)
    bands = (*es.METHOD_BANDS_KEV_UM, (600.0, 700.0))  # the last one is empty

    got = es.paper_method_bands(
        photon, cell, radial, stopping_power=sp, energies_MeV_u=energies, bands=bands
    )

    rx, ry = np.array(ref.x), np.array(ref.y)
    curves = {}
    for mode in ("lethal_central_saturating", "cross_section"):
        model = LEM1(
            photon=photon,
            cell=cell,
            stopping_power=sp,
            radial_dose=radial,
            initial_slope_model=mode,
        )
        pts = []
        for energy in energies:
            ion = Ion(Z=6, A=12, energy_MeV_u=float(energy))
            pts.append((sp.let_keV_um(ion), model.alpha_beta(ion).alpha / photon.alpha_x))
        pts.sort()
        curves[mode] = (np.array([p[0] for p in pts]), np.array([p[1] for p in pts]))
    let_grid = np.array([sp.let_keV_um(Ion(Z=6, A=12, energy_MeV_u=float(e))) for e in energies])
    order = np.argsort(let_grid)
    want = []
    for lo, hi in bands:
        m = (rx >= lo) & (rx < hi)
        if m.sum() < 2:
            continue
        mid = float(np.exp(np.mean(np.log(rx[m]))))
        e_mid = float(np.interp(mid, let_grid[order], energies[order]))
        ratio = radial.r_max_um(Ion(Z=6, A=12, energy_MeV_u=e_mid)) / (
            cell.effective_nucleus_radius_um()
        )
        bias = [
            float(
                np.mean(
                    np.interp(np.log(rx[m]), np.log(let), value, left=np.nan, right=np.nan) / ry[m]
                    - 1
                )
            )
            for let, value in curves.values()
        ]
        want.append(es.MethodBand(lo, hi, int(m.sum()), ratio, bias[0], bias[1]))

    assert len(got) == len(es.METHOD_BANDS_KEV_UM)
    assert got == tuple(want)
