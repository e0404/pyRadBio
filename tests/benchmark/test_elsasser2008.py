"""Elsasser, Kraemer & Scholz 2008 (LEM III) benchmark reference tables.

Reference: Elsasser, Kraemer & Scholz 2008, Int. J. Radiat. Oncol. Biol. Phys.
71(3):866-872, doi:10.1016/j.ijrobp.2008.02.037, Tables 1-2 (Suzuki 2000 panel).
"""

from __future__ import annotations

import math

import numpy as np
import pytest
from scipy.optimize import brentq

from pyradbio.benchmark import elsasser2008 as es
from pyradbio.core.cell import CellTarget
from pyradbio.core.photon import ClusterParameters, PhotonParameters
from pyradbio.core.radiation import Ion
from pyradbio.models.lem.lem1 import LEM1
from pyradbio.models.lem.lem2 import LEM2, ClusterModel
from pyradbio.models.lem.lem3 import LEM3
from pyradbio.track.stopping_power import SimpleBetheStoppingPower


def test_panel_covers_the_fifteen_suzuki_lines() -> None:
    assert len(es.SUZUKI_PANEL) == 15
    names = {c.name for c in es.SUZUKI_PANEL}
    assert {"KS-1", "A549", "KNS-89"} <= names
    # Every per-version tuple is length 3 (LEM I / II / III).
    for c in es.SUZUKI_PANEL:
        assert len(c.D_t) == len(c.dRBE_13) == len(c.dRBE_77) == 3
        assert len(c.rbe_alpha_exp) == 2


def test_suzuki_areas_transcribed_and_give_sane_radii() -> None:
    """Nuclear areas (Suzuki 2000 Table 3) map to plausible nuclear radii.

    Suzuki's measured areas span ~140-280 um^2, so the equivalent-circle geometric
    radius sqrt(area/pi) is ~6.7-9.5 um and the effective (x0.8) radius ~5.4-7.6 um.
    A couple of spot values guard the transcription.
    """
    by_name = {c.name: c for c in es.SUZUKI_PANEL}
    assert by_name["KS-1"].nucleus_area_um2 == 221.4
    assert by_name["T98G"].nucleus_area_um2 == 282.5  # largest nucleus in the panel
    assert by_name["HFL-III"].nucleus_area_um2 == 142.9  # smallest
    for c in es.SUZUKI_PANEL:
        assert 6.0 < c.geometric_radius_um < 10.0
        assert c.geometric_radius_um == pytest.approx((c.nucleus_area_um2 / math.pi) ** 0.5)


def test_photon_lq_is_suzuki_table_2_and_rounds_to_elsassers_print() -> None:
    """``alpha_x``/``beta_x`` are Suzuki 2000 Table 2; Elsasser printed them rounded.

    Elsasser's Table 1 shows 2 (alpha) and 3 (beta) decimals; the panel carries the
    measured values, and each printed number must be their rounding -- except the
    one documented misprint, SK-MG-1 beta 0.054 (Suzuki: 0.05292).
    """
    by_name = {c.name: c for c in es.SUZUKI_PANEL}
    assert (by_name["KNS-60"].alpha_x, by_name["KNS-60"].beta_x) == (0.03156, 0.04816)
    assert (by_name["A549"].alpha_x, by_name["A549"].beta_x) == (0.07456, 0.04548)
    assert es.ELSASSER2008_PRINTED_PHOTON_LQ.keys() == by_name.keys()
    for name, (alpha_printed, beta_printed) in es.ELSASSER2008_PRINTED_PHOTON_LQ.items():
        c = by_name[name]
        assert abs(c.alpha_x - alpha_printed) <= 0.005 + 1e-12, name
        rounds = abs(c.beta_x - beta_printed) <= 0.0005 + 1e-12
        assert rounds != (name in es.ELSASSER2008_BETA_X_MISPRINTS), name


@pytest.mark.parametrize("version", range(3))
def test_per_cell_deviations_average_to_the_published_row(version: int) -> None:
    """The transcribed per-cell dRBE average back to Tables 1-2's "Average" row.

    This checks the load-bearing transcription details -- the signs of the two
    channels and the interleaved D_t +-10% error columns -- since a dropped minus
    sign or a shifted column would move the mean off the published value.
    """
    m13, m77 = es.paper_mean_delta(version)
    p13, p77 = es.PAPER_AVERAGE_DELTA[version]
    assert m13 == pytest.approx(p13, abs=0.005)
    assert m77 == pytest.approx(p77, abs=0.005)


def test_entrance_over_prediction_shrinks_across_versions() -> None:
    """The paper's headline result: mean dRBE(13) falls monotonically I->II->III."""
    means_13 = [es.paper_mean_delta(v)[0] for v in range(3)]
    assert means_13[0] > means_13[1] > means_13[2] > 0.0


def test_dt_refit_matches_fig1_caption_for_in_vivo() -> None:
    rsc = es.RAT_SPINAL_CORD
    assert (rsc.alpha_x, rsc.beta_x, rsc.nucleus_radius_um) == (0.1, 0.05, 5.0)
    assert rsc.D_t == (30.0, 9.5, 60.0)


def test_rbe_alpha_paper_inverts_the_deviation() -> None:
    """``rbe_alpha_paper`` reconstructs the paper's predicted RBE from exp*(1+dRBE)."""
    ks1 = next(c for c in es.SUZUKI_PANEL if c.name == "KS-1")
    # LEM III: dRBE(13) = -0.02 on an experimental 1.20 -> predicted 1.176.
    pred13, _pred77 = ks1.rbe_alpha_paper(2)
    assert pred13 == pytest.approx(1.20 * (1.0 - 0.02))


# --- reconstruction machinery (validation/elsasser2008_lem3.py) -----------------

_SP = SimpleBetheStoppingPower()


def test_cell_target_is_the_geometric_radius_times_point_eight() -> None:
    for c in es.SUZUKI_PANEL:
        target = c.cell_target()
        assert target == CellTarget(
            nucleus_radius_um=c.geometric_radius_um,
            effective_size_factor=es.EFFECTIVE_SIZE_FACTOR,
        )
        assert target.effective_nucleus_radius_um() == pytest.approx(0.8 * c.geometric_radius_um)


def test_channel_ions_are_the_inline_brentq_roots_bit_for_bit() -> None:
    """The 13 / 77 keV/um carbon energies equal the script's ``brentq(..., 1.5, 500)``."""
    ion_13, ion_77 = es.channel_ions(_SP)
    for ion, let in ((ion_13, 13.0), (ion_77, 77.0)):
        want = float(brentq(lambda e, let=let: _SP.let_keV_um(Ion(6, 12, e)) - let, 1.5, 500.0))
        assert (ion.Z, ion.A, ion.energy_MeV_u) == (6, 12, want)
    assert 270.0 < ion_13.energy_MeV_u < 300.0
    assert 20.0 < ion_77.energy_MeV_u < 30.0


def test_build_panel_model_picks_the_version_and_passes_the_cluster() -> None:
    cluster = ClusterModel()
    cell = es.SUZUKI_PANEL[0].cell_target()
    lem1 = es.build_panel_model("LEM I", 0.1, 0.05, 30.0, cell, stopping_power=_SP)
    lem2 = es.build_panel_model("LEM II", 0.1, 0.05, 9.5, cell, stopping_power=_SP, cluster=cluster)
    lem3 = es.build_panel_model(
        "LEM III", 0.1, 0.05, 60.0, cell, stopping_power=_SP, cluster=cluster
    )
    assert type(lem1) is LEM1 and lem1.stopping_power is _SP
    assert type(lem2) is LEM2 and lem2.resolved_cluster is cluster
    assert type(lem3) is LEM3 and lem3.resolved_cluster is cluster
    assert lem3.photon == PhotonParameters(
        alpha_x=0.1, beta_x=0.05, D_t=60.0, cluster=ClusterParameters()
    )
    with pytest.raises(ValueError, match="LEM IV"):
        es.build_panel_model("LEM IV", 0.1, 0.05, 30.0, cell, stopping_power=_SP)


@pytest.mark.parametrize("name", es.MODEL_NAMES)
def test_predict_panel_and_summaries_are_the_inline_loops_bit_for_bit(name: str) -> None:
    """``predict_panel``/``mean_delta``/``per_cell_rms`` reproduce the script's loops.

    The original script built each line's model with the Suzuki-area target and a
    shared cluster model, stored ``(pred13, pred77, exp13, exp77)`` rows, and
    reduced them with ``np.mean`` / an RMS over the paper's tabulated deviations.
    Three lines keep the test fast; KS-1 (index 0) exercises ``exclude``.
    """
    panel = es.SUZUKI_PANEL[:3]
    ions = es.channel_ions(_SP)
    cluster = ClusterModel()
    rows = es.predict_panel(name, ions, stopping_power=_SP, cluster=cluster, panel=panel)

    v = es.MODEL_NAMES.index(name)
    want = []
    for c in panel:
        photon = PhotonParameters(
            alpha_x=c.alpha_x, beta_x=c.beta_x, D_t=c.D_t[v], cluster=ClusterParameters()
        )
        target = CellTarget(
            nucleus_radius_um=c.geometric_radius_um, effective_size_factor=es.EFFECTIVE_SIZE_FACTOR
        )
        if name == "LEM I":
            model: LEM1 | LEM2 = LEM1(photon=photon, cell=target, stopping_power=_SP)
        else:
            cls = LEM2 if name == "LEM II" else LEM3
            model = cls(photon=photon, cell=target, stopping_power=_SP, cluster=cluster)
        want.append(
            (
                model.alpha_beta(ions[0]).alpha / c.alpha_x,
                model.alpha_beta(ions[1]).alpha / c.alpha_x,
                *c.rbe_alpha_exp,
            )
        )
    assert rows == tuple(want)

    arr = np.array(want)
    assert es.mean_delta(rows) == (
        float(np.mean(arr[:, 0] / arr[:, 2] - 1.0)),
        float(np.mean(arr[:, 1] / arr[:, 3] - 1.0)),
    )
    for channel, paper in ((0, [c.dRBE_13[v] for c in panel]), (1, [c.dRBE_77[v] for c in panel])):
        for exclude in (frozenset(), es.OUTLIER_CELLS_77):
            residuals = [
                (want[i][channel] / want[i][channel + 2] - 1.0) - paper[i]
                for i, c in enumerate(panel)
                if c.name not in exclude
            ]
            got = es.per_cell_rms(rows, name, channel, exclude=exclude, panel=panel)
            assert got == float(np.sqrt(np.mean(np.square(residuals))))


def test_per_cell_rms_rejects_a_bad_channel_or_mismatched_rows() -> None:
    rows = ((1.0, 2.0, 1.0, 2.0),)
    with pytest.raises(ValueError, match="channel"):
        es.per_cell_rms(rows, "LEM I", 2, panel=es.SUZUKI_PANEL[:1])
    with pytest.raises(ValueError, match="differ in length"):
        es.per_cell_rms(rows, "LEM I", 0)
