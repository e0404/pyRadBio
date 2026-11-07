"""Friedrich et al. 2012 LEM IV proceedings-note benchmark reference data (Fig. 3).

Reference: Friedrich, T. et al. 2012, Int. J. Radiat. Biol. 88(1-2):103-107,
doi:10.3109/09553002.2011.611213.
"""

from __future__ import annotations

from itertools import pairwise
from pathlib import Path

import pytest

from pyradbio.benchmark import friedrich2012 as f12


def test_curve_loader_groups_digitized_points() -> None:
    curves = f12.load_curves()
    # 4 Fig. 3 series (model/data x rbe_alpha/rbe_10) + 4 Fig. 4 series
    # (approx/full_simulation x alpha/beta).
    assert len(curves) == 8
    assert all(len(c.x) == len(c.y) >= 2 for c in curves)
    assert all(c.doi == f12.FRIEDRICH2012_DOI for c in curves)
    # x-values are stored sorted ascending (grouped-and-sorted loader contract).
    assert all(list(c.x) == sorted(c.x) for c in curves)


def test_get_curve_rejects_ambiguous_and_empty_filters() -> None:
    with pytest.raises(LookupError):
        f12.get_curve("fig3_v79_rbe_vs_let")  # 4 series: calc/data x rbe_alpha/rbe_10
    with pytest.raises(LookupError):
        f12.get_curve("fig3_v79_rbe_vs_let", series="calc")  # 2 series: rbe_alpha, rbe_10
    with pytest.raises(LookupError):
        f12.get_curve("fig4_hypothetical_vs_let")  # 4 series: approx/full_sim x alpha/beta
    with pytest.raises(LookupError):
        f12.get_curve("fig4_hypothetical_vs_let", series="approx")  # 2 variants: alpha, beta
    with pytest.raises(LookupError):
        f12.get_curve("no_such_figure")


def test_figure_inputs_match_the_captions() -> None:
    """Fig. 3 caption (p. 106): alpha_g=0.184, beta_g=0.02, D_t=17, r_N=4.7 --
    identical to the ``V79``/``V79_CELL`` fixture in ``test_lem4.py``. Fig. 4
    caption (p. 107): the "hypothetical" cell, alpha_g=0.25, beta_g=0.05,
    D_t=9.5, r_N=5 -- identical to ``test_lem4_intertrack.py``'s
    ``FIG4``/``FIG4_CELL`` fixture."""
    v79 = f12.FIGURE_INPUTS["fig3_v79"]
    assert (v79.alpha_x, v79.beta_x) == (0.184, 0.02)
    assert v79.D_t == 17.0
    assert v79.nucleus_radius_um == 4.7
    assert v79.effective_size_factor == 1.0

    hypothetical = f12.FIGURE_INPUTS["fig4_hypothetical"]
    assert (hypothetical.alpha_x, hypothetical.beta_x) == (0.25, 0.05)
    assert hypothetical.D_t == 9.5
    assert hypothetical.nucleus_radius_um == 5.0
    assert hypothetical.effective_size_factor == 1.0


def test_model_rbe_alpha_exceeds_rbe_10_at_every_digitized_let() -> None:
    """RBE_alpha (initial slope) is always >= RBE_10 (dose-averaged) for the same
    LQ curve: the model lines must respect that ordering everywhere they overlap.

    The two curves were traced independently and do not share a LET grid, so
    RBE_10 is linearly interpolated onto RBE_alpha's (denser) grid.
    """
    import numpy as np

    alpha = f12.get_curve("fig3_v79_rbe_vs_let", series="calc", variant="rbe_alpha")
    rbe10 = f12.get_curve("fig3_v79_rbe_vs_let", series="calc", variant="rbe_10")
    lo, hi = max(alpha.x[0], rbe10.x[0]), min(alpha.x[-1], rbe10.x[-1])
    lets = [let for let in alpha.x if lo <= let <= hi]
    assert len(lets) > 5
    rbe10_at = np.interp(lets, rbe10.x, rbe10.y)
    alpha_at = dict(zip(alpha.x, alpha.y, strict=True))
    for let, r10 in zip(lets, rbe10_at, strict=True):
        assert alpha_at[let] >= r10, f"RBE_alpha < RBE_10 at LET={let}"


def test_model_curve_has_single_interior_maximum() -> None:
    """The paper's own RBE_alpha(LET) model line rises to a single interior
    maximum near ~185 keV/um then declines (overkill) -- the shape the golden
    ``test_golden_lem4_v79_carbon_rbe_alpha_shape`` shape-only test targets,
    checked point-by-point here."""
    model = f12.get_curve("fig3_v79_rbe_vs_let", series="calc", variant="rbe_alpha")
    i_peak = int(model.y.index(max(model.y)))
    assert 0 < i_peak < len(model.y) - 1
    assert list(model.y[: i_peak + 1]) == sorted(model.y[: i_peak + 1])
    assert list(model.y[i_peak:]) == sorted(model.y[i_peak:], reverse=True)
    assert 150.0 <= model.x[i_peak] <= 220.0


def test_experimental_data_has_no_error_bar_duplication() -> None:
    """Unlike Elsasser et al. 2010's Fig. 4 data, this figure's experimental
    points are genuine independent measurements even where LET values sit
    close together -- e.g. ~40.0 and ~40.6 keV/um differ by ~1.8 RBE units,
    real scatter rather than an error-bar top/bottom pair (see the module
    docstring)."""
    data = f12.get_curve("fig3_v79_rbe_vs_let", series="data", variant="rbe_alpha")
    pts = list(zip(data.x, data.y, strict=True))
    near_dupes = [
        (x1, y1, x2, y2)
        for (x1, y1), (x2, y2) in pairwise(pts)
        if abs(x1 - x2) <= 0.03 * max(1.0, x1)
    ]
    assert near_dupes, "expected at least one close-LET pair to exercise the check"
    for x1, y1, x2, y2 in near_dupes:
        assert abs(y1 - y2) > 0.3, f"LET~{x1:.1f}/{x2:.1f} look like an error-bar pair, not scatter"


def test_fig4_covers_alpha_and_beta_for_both_series() -> None:
    """Fig. 4 has all four series digitized: approx/full_simulation x
    alpha/beta. The alpha traces are calibrated to the alpha_I axis
    (see the module docstring) and sit in the physically expected ~0.25-1.6 Gy^-1
    range, not ~0.01-0.06 (a factor 25.0 lower)."""
    curves = f12.iter_curves(figure="fig4_hypothetical_vs_let")
    assert {c.variant for c in curves} == {"alpha", "beta"}
    assert {c.series for c in curves} == {"approx", "full_simulation"}
    for c in curves:
        if c.variant == "alpha":
            assert 0.2 < min(c.y) and max(c.y) < 2.0, f"{c.series} alpha_I out of range: {c.y}"


def test_fig4_full_simulation_beta_exceeds_approximation_at_intermediate_let() -> None:
    """Fig. 4's central finding: the full intertrack Monte-Carlo ``beta_I``
    stays well above the Kraemer-Scholz slope-matching approximation in the
    intermediate-LET range -- "the drop of the beta_I(LET)-curve is
    significantly shifted to higher LET" (paper text, p. 106)."""
    import numpy as np

    approx = f12.get_curve("fig4_hypothetical_vs_let", series="approx", variant="beta")
    full_sim = f12.get_curve("fig4_hypothetical_vs_let", series="full_simulation", variant="beta")
    lo, hi = max(approx.x[0], full_sim.x[0]), min(approx.x[-1], full_sim.x[-1])
    lets = [let for let in approx.x if lo <= let <= hi]
    assert len(lets) > 10
    full_sim_at = np.interp(lets, full_sim.x, full_sim.y)
    approx_at = dict(zip(approx.x, approx.y, strict=True))
    # At intermediate LET (60-100 keV/um) the approximation has already
    # collapsed toward zero while the full simulation has not.
    intermediate = [
        (let, a, fs)
        for let, a, fs in zip(lets, [approx_at[let] for let in lets], full_sim_at, strict=True)
        if 60.0 <= let <= 100.0
    ]
    assert len(intermediate) > 3
    for let, a, fs in intermediate:
        assert fs > a, f"full-sim beta_I {fs:.4f} !> approximation {a:.4f} at LET={let:.0f}"
        assert a < 0.02, f"approximation beta_I={a:.4f} has not collapsed by LET={let:.0f}"


def test_fig4_alpha_almost_identical_between_approx_and_full_simulation() -> None:
    """Friedrich 2012 (p. 106): "The alpha_I-values are almost identical"
    between the single-particle approximation and the full simulation --
    only beta_I differs. Checked at the digitized LET points they share."""
    import numpy as np

    approx = f12.get_curve("fig4_hypothetical_vs_let", series="approx", variant="alpha")
    full_sim = f12.get_curve("fig4_hypothetical_vs_let", series="full_simulation", variant="alpha")
    lo, hi = max(approx.x[0], full_sim.x[0]), min(approx.x[-1], full_sim.x[-1])
    lets = [let for let in approx.x if lo <= let <= hi]
    assert len(lets) > 10
    full_sim_at = np.interp(lets, full_sim.x, full_sim.y)
    approx_at = np.interp(lets, approx.x, approx.y)
    rel_diff = np.abs(full_sim_at - approx_at) / approx_at
    assert np.mean(rel_diff) < 0.15, f"mean relative alpha_I difference = {np.mean(rel_diff):.3f}"


# ---- Fig. 4 explicit-track MC: model, LET inversion and the CSV reader ------------


def test_fig4_mc_model_is_the_examples_lem_f4_mc() -> None:
    """``fig4_mc_model`` builds the ``lem_f4_mc`` model of the original script."""
    from pyradbio.core.cell import CellTarget
    from pyradbio.core.photon import PhotonParameters
    from pyradbio.models.lem.lem4 import LEM4
    from pyradbio.track.stopping_power import default_stopping_power

    sp = default_stopping_power()
    inline = LEM4(
        photon=PhotonParameters(alpha_x=0.25, beta_x=0.05, D_t=9.5),
        cell=CellTarget(5.0, effective_size_factor=1.0),
        stopping_power=sp,
        engine="numpy",
        local_dose_mc_impact_cut="auto",
    )
    assert f12.fig4_mc_model(engine="numpy", stopping_power=sp) == inline
    assert f12.fig4_mc_model().local_dose_mc_impact_cut == "auto"


def test_fig4_mc_ion_is_the_examples_inline_inversion() -> None:
    from scipy.optimize import brentq

    from pyradbio.core.radiation import Ion
    from pyradbio.track.stopping_power import default_stopping_power

    sp = default_stopping_power()
    for let in (15.0, 100.0):
        e = brentq(lambda x, _let=let: sp.let_keV_um(Ion(6, 12, x)) - _let, 0.4, 600.0)
        assert f12.fig4_mc_ion(let, sp) == Ion(6, 12, float(e))
    with pytest.raises(ValueError, match="not bracketed"):
        f12.fig4_mc_ion(5000.0, sp)


_MC_CSV = """\
# generator: scripts/lem4_friedrich2012_mc.py
# n_seeds: 2
# fingerprint: [{"alpha": 1.0, "beta": 0.5}]
let_keV_um,seed,energy_MeV_u,n_realizations,alpha,beta,wall_s
40.0,0,57.0,50,0.6,0.04,1.0
40.0,1,57.0,50,0.7,0.03,1.0
100.0,1,18.1,50,1.3,0.017,1.0
100.0,0,18.1,50,1.2,0.015,1.0
55.0,0,38.1,50,0.8,0.03,1.0
"""


def test_load_fig4_mc_reduces_complete_lets_to_seed_statistics(tmp_path: Path) -> None:
    path = tmp_path / f12.FIG4_MC_CSV_NAME
    path.write_text(_MC_CSV, encoding="utf-8")
    mc = f12.load_fig4_mc(path)

    assert tuple(mc.let_keV_um) == (40.0, 100.0)
    assert mc.incomplete_lets == (55.0,)  # one seed of the header's two
    assert mc.seeds == ((0, 1), (0, 1))  # rows are re-ordered by seed
    assert mc.n_realizations == (50, 50)
    assert mc.header["fingerprint"] == [{"alpha": 1.0, "beta": 0.5}]
    expected = f12.fig4_mc_seed_statistics([[1.2, 0.015], [1.3, 0.017]])
    assert (mc.alpha[1], mc.alpha_err[1], mc.beta[1], mc.beta_err[1]) == expected
    # n_seeds overrides the header
    none_complete = f12.load_fig4_mc(path, n_seeds=3)
    assert none_complete.incomplete_lets == (40.0, 55.0, 100.0)
    assert none_complete.let_keV_um.shape == none_complete.beta_err.shape == (0,)


def test_load_fig4_mc_rejects_a_missing_file_and_mixed_sample_sizes(tmp_path: Path) -> None:
    with pytest.raises(FileNotFoundError):
        f12.load_fig4_mc(tmp_path / "absent.csv")
    path = tmp_path / f12.FIG4_MC_CSV_NAME
    path.write_text(_MC_CSV.replace("40.0,1,57.0,50", "40.0,1,57.0,60"), encoding="utf-8")
    with pytest.raises(ValueError, match="mixes n_realizations"):
        f12.load_fig4_mc(path)
