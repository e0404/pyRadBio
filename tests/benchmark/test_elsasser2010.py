"""Elsasser et al. 2010 LEM IV benchmark reference data (Fig. 4a/4b).

Reference: Elsasser, T. et al. 2010, Int. J. Radiat. Oncol. Biol. Phys.
78(4):1177-1183, doi:10.1016/j.ijrobp.2010.05.014.
"""

from __future__ import annotations

import pytest

from pyradbio.benchmark import elsasser2010 as es


def test_curve_loader_groups_digitized_points() -> None:
    curves = es.load_curves()
    assert len(curves) == 12
    assert all(len(c.x) == len(c.y) >= 2 for c in curves)
    assert all(c.doi == es.ELSASSER2010_DOI for c in curves)
    # x-values are stored sorted ascending (grouped-and-sorted loader contract).
    assert all(list(c.x) == sorted(c.x) for c in curves)


def test_error_bar_series_expose_y_err() -> None:
    """Fig. 4a's proton points are the only ones the paper draws bars on, so they
    are the only series carrying ``y_err``; model lines never do.

    Fig. 4b's helium series must not carry ``y_err``: pairs of *distinct* Furusawa
    measurements at neighbouring LETs (18.5 and 18.6, 23.6 and 24.0, ...) are not
    error bars and must not be consolidated into one midpoint each. Fig. 4b draws
    no error bars at all.
    """
    proton_data = es.get_curve("fig4a_v79_rbealpha_vs_let", particle="proton", series="data")
    assert proton_data.y_err is not None
    assert len(proton_data.y_err) == len(proton_data.x) == 11
    assert all(e > 0.0 for e in proton_data.y_err)

    helium_alpha = es.get_curve(
        "fig4b_hsg_rbe_vs_let", particle="helium", series="data", variant="rbe_alpha"
    )
    assert helium_alpha.y_err is None
    assert len(helium_alpha.x) == 13  # every marker the panel draws

    proton_model = es.get_curve("fig4a_v79_rbealpha_vs_let", particle="proton", series="calc")
    assert proton_model.y_err is None


def test_fig4a_proton_model_and_helium_data_corrected_digitization() -> None:
    """Fig. 4a's proton model line and helium experimental data (see the module
    docstring): the proton model line is dense and covers the paper's full LET
    range, and the helium data are a sparse, non-duplicated series."""
    proton_model = es.get_curve("fig4a_v79_rbealpha_vs_let", particle="proton", series="calc")
    assert len(proton_model.x) == 142  # every vertex of the drawn polyline
    assert proton_model.y_err is None
    assert max(proton_model.x) > 35.0  # extends well past LET 25

    helium_data = es.get_curve("fig4a_v79_rbealpha_vs_let", particle="helium", series="data")
    assert len(helium_data.x) == 5
    assert helium_data.y_err is None  # no duplicate error-bar-endpoint pairs to consolidate
    assert 15.0 < min(helium_data.x) and max(helium_data.x) < 90.0


def test_get_curve_rejects_ambiguous_and_empty_filters() -> None:
    with pytest.raises(LookupError):
        es.get_curve("fig4b_hsg_rbe_vs_let")  # 8 series: 2 particles x calc/data x alpha/10
    with pytest.raises(LookupError):
        es.get_curve("fig4a_v79_rbealpha_vs_let", particle="proton")  # 2 series: calc + data
    with pytest.raises(LookupError):
        es.get_curve("no_such_figure")


def test_figure_inputs_match_the_captions() -> None:
    """Fig. 4a: alpha_g=0.129, beta_g=0.049, D_t=3.0, r_N=5 (p. 1181 caption).
    Fig. 4b: alpha_g=0.313, beta_g=0.0615, D_t=7.5, r_N=5."""
    v79 = es.FIGURE_INPUTS["fig4a_v79"]
    assert (v79.alpha_x, v79.beta_x) == (0.129, 0.049)
    assert v79.D_t == 3.0
    assert v79.nucleus_radius_um == 5.0
    assert v79.effective_size_factor == 1.0

    hsg = es.FIGURE_INPUTS["fig4b_hsg"]
    assert (hsg.alpha_x, hsg.beta_x) == (0.313, 0.0615)
    assert hsg.D_t == 7.5
    assert hsg.nucleus_radius_um == 5.0
    assert hsg.effective_size_factor == 1.0


def test_fig4a_helium_model_peak_exceeds_proton() -> None:
    """Fig. 4a: helium's RBE_alpha model line reaches a higher peak than
    proton's over its (higher-LET) digitized range.

    Note this is *not* an ordering-at-matched-LET claim: in the LET band
    where the two model lines overlap (~19-40 keV/um, since the proton line extends
    that far), the *proton* line actually sits above the helium line there (proton ~4.3-6.5 vs
    helium ~3.1-4.6) -- helium only overtakes proton once its own line
    continues rising past LET~40 to its ~10.7 peak, well outside the overlap
    band. Helium does not always lead at matched LET (as in the Fig. 4b He/C
    relationship); trust the digitized data over that analogy.
    """
    proton = es.get_curve("fig4a_v79_rbealpha_vs_let", particle="proton", series="calc")
    helium = es.get_curve("fig4a_v79_rbealpha_vs_let", particle="helium", series="calc")
    assert max(helium.y) > max(proton.y)


@pytest.mark.parametrize("particle", ["carbon", "helium"])
def test_fig4b_model_curves_rise_from_near_one(particle: str) -> None:
    """Both digitized model lines (per ion) start near the photon limit (RBE ~
    1) at low LET and stay above it -- the entrance-channel shape every LEM
    version reproduces."""
    for variant in ("model_rbe_alpha", "model_rbe_10"):
        curve = es.get_curve(
            "fig4b_hsg_rbe_vs_let", particle=particle, series="calc", variant=variant
        )
        assert 0.8 < curve.y[0] < 1.4, f"{particle} {variant} low-LET RBE={curve.y[0]:.2f}"
        assert max(curve.y) > curve.y[0]


@pytest.mark.parametrize("particle", ["carbon", "helium"])
def test_fig4b_model_rbe_alpha_exceeds_rbe_10_at_every_digitized_let(particle: str) -> None:
    """RBE_alpha (initial slope) is always >= RBE_10 (dose-averaged) for the same
    LQ curve: the model lines must respect that ordering everywhere they overlap.

    The two curves were traced independently and do not share a LET grid, so
    RBE_10 is linearly interpolated onto RBE_alpha's (denser) grid.
    """
    import numpy as np

    alpha = es.get_curve(
        "fig4b_hsg_rbe_vs_let", particle=particle, series="calc", variant="model_rbe_alpha"
    )
    rbe10 = es.get_curve(
        "fig4b_hsg_rbe_vs_let", particle=particle, series="calc", variant="model_rbe_10"
    )
    lo, hi = max(alpha.x[0], rbe10.x[0]), min(alpha.x[-1], rbe10.x[-1])
    lets = [let for let in alpha.x if lo <= let <= hi]
    assert len(lets) > 5
    rbe10_at = np.interp(lets, rbe10.x, rbe10.y)
    alpha_at = dict(zip(alpha.x, alpha.y, strict=True))
    for let, r10 in zip(lets, rbe10_at, strict=True):
        assert alpha_at[let] >= r10, f"{particle} RBE_alpha < RBE_10 at LET={let}"


def test_fig4b_helium_model_exceeds_carbon_rbe_alpha_at_high_let() -> None:
    """Fig. 4b: the two ions' RBE_alpha model lines are close at low LET and
    diverge above ~60 keV/um, helium continuing to rise while carbon's turns
    over (see the module docstring's He/C peak-position argument)."""
    carbon = es.get_curve(
        "fig4b_hsg_rbe_vs_let", particle="carbon", series="calc", variant="model_rbe_alpha"
    )
    helium = es.get_curve(
        "fig4b_hsg_rbe_vs_let", particle="helium", series="calc", variant="model_rbe_alpha"
    )
    assert max(helium.y) > max(carbon.y)
    # carbon's peak-then-decline shape: not monotone over its full digitized range.
    assert list(carbon.y) != sorted(carbon.y)
