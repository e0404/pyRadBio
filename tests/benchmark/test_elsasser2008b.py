"""Elsasser et al. 2008b (NJP 10 075005) Table 1 -- printed RBE_alpha goldens.

Every other LEM reconstruction here is scored against *digitized* figures. Table 1
is printed, so this module is the corpus's cleanest model-line cross-check: three
track-structure variants (mapping onto LEM I, II and III), two carbon LETs, one
cell line, six numbers with the paper's own uncertainties.

It also settles the initial-slope method question, which the digitized LEM II
panels could only argue.
"""

from __future__ import annotations

import pytest
from scipy.optimize import brentq

from pyradbio.benchmark import elsasser2008b as njp
from pyradbio.core.cell import CellTarget
from pyradbio.core.photon import PhotonParameters
from pyradbio.core.radiation import Ion
from pyradbio.models.lem.base import InitialSlopeModel
from pyradbio.models.lem.lem1 import LEM1
from pyradbio.models.lem.lem2 import LEM2
from pyradbio.models.lem.lem3 import LEM3
from pyradbio.track.scholz_kraft import ScholzKraftRadialDose
from pyradbio.track.stopping_power import SimpleBetheStoppingPower

_SP = SimpleBetheStoppingPower()
_CELL = CellTarget(nucleus_radius_um=njp.HSG_NUCLEUS_RADIUS_UM)


def _carbon_for_let(let_keV_um: float) -> Ion:
    energy = float(brentq(lambda e: _SP.let_keV_um(Ion(6, 12, e)) - let_keV_um, 1.6, 600.0))
    return Ion(6, 12, energy)


def _model(label: str, slope: InitialSlopeModel) -> LEM1 | LEM2 | LEM3:
    """Build the LEM version Table 1's row corresponds to, with its fitted ``D_t``."""
    track = njp.get_track_model(label)
    photon = PhotonParameters(alpha_x=njp.HSG_ALPHA_X, beta_x=njp.HSG_BETA_X, D_t=track.d_t_Gy)
    common = {
        "photon": photon,
        "cell": _CELL,
        "stopping_power": _SP,
        "initial_slope_model": slope,
    }
    if label == "CR10":
        # LEM I: the large core stands in for radical diffusion, and there is no
        # cluster eta -- but r_max uses this paper's gamma = 0.062, not LEM I's 0.05.
        return LEM1(
            radial_dose=ScholzKraftRadialDose(
                r_min_nm=10.0,
                radical_diffusion=False,
                r_max_gamma_um=njp.R_MAX_GAMMA_UM,
                stopping_power=_SP,
            ),
            **common,  # type: ignore[arg-type]
        )
    if label == "CR03":
        return LEM2(radial_dose=ScholzKraftRadialDose.lem2(_SP), **common)  # type: ignore[arg-type]
    if label == "EDR40":
        return LEM3(**common)  # type: ignore[arg-type]
    raise AssertionError(label)


def _rbe_alpha(label: str, let_keV_um: float, slope: InitialSlopeModel) -> float:
    lem = _model(label, slope)
    return lem.alpha_beta(_carbon_for_let(let_keV_um)).alpha / njp.HSG_ALPHA_X


@pytest.mark.golden
@pytest.mark.parametrize("label", njp.IMPLEMENTED_LABELS)
@pytest.mark.parametrize("let_keV_um", njp.TABLE1_LET_KEV_UM)
def test_golden_table1_rbe_alpha(label: str, let_keV_um: float) -> None:
    """Reproduce the printed Table 1 ``RBE_alpha`` for every implemented core.

    Three cores spanning a 100x range of central dose, each with the ``D_t`` the
    paper fitted for it, all landing within a few per cent -- which tests the core
    treatment, the cluster enhancement, the ``D_t`` degeneracy and the
    initial-slope method simultaneously. Measured deviations are -3.9% to +1.3%.
    """
    ours = _rbe_alpha(label, let_keV_um, "cross_section")
    paper, _uncertainty = njp.get_track_model(label).rbe_at(let_keV_um)
    assert ours == pytest.approx(paper, rel=0.06), (
        f"{label} at {let_keV_um} keV/um: ours {ours:.3f} vs paper {paper:.3f}"
    )


@pytest.mark.golden
def test_golden_table1_selects_the_cross_section_initial_slope() -> None:
    """Table 1 discriminates the initial-slope methods, and picks the paper's own.

    Sec. 4 says the calculations use "the approximation described in [10]" =
    Scholz et al. 1997, i.e. the inactivation *cross-section* integral
    ``sigma = int (1 - exp(-nu(b))) 2 pi b db``. This checks that the printed
    numbers actually behave that way, which the digitized LEM II panels could only
    argue indirectly.

    At 13 keV/um the carbon track is ~282 MeV/u, so ``r_max ~ 900 um`` against a
    5 um nucleus -- ``r_max/R_eff ~ 180``. A ``b = 0`` central-traversal shortcut
    (Elsasser & Scholz 2007 eq. 8) discards every one of those near-miss
    traversals, and the printed value catches it: the shortcut lands 18-30% low on
    all three cores, while the full integral lands within a few per cent. The
    overkill-free ``lethal_full`` recovers the low-LET point but overshoots at
    77 keV/um, so only ``cross_section`` satisfies both columns at once.
    """
    for label in njp.IMPLEMENTED_LABELS:
        paper_low, _ = njp.get_track_model(label).rbe_at(13.0)
        exact = _rbe_alpha(label, 13.0, "cross_section")
        central = _rbe_alpha(label, 13.0, "lethal_central_saturating")

        assert exact == pytest.approx(paper_low, rel=0.06), label
        # the b=0 shortcut is low by far more than the paper's own uncertainty
        assert central < 0.85 * paper_low, (
            f"{label}: central traversal {central:.3f} vs paper {paper_low:.3f}"
        )
        assert central < exact

    # ... and the overkill-free variant fails the *other* column, so the choice is
    # not simply "anything with the full impact-parameter integral".
    paper_high, _ = njp.get_track_model("CR10").rbe_at(77.0)
    assert _rbe_alpha("CR10", 77.0, "lethal_full") > 1.10 * paper_high


def test_track_model_table_is_self_consistent() -> None:
    """Guard the transcription: labels unique, LET-indexed lookup, known versions."""
    labels = [m.label for m in njp.TRACK_MODELS]
    assert len(labels) == len(set(labels))
    assert set(njp.IMPLEMENTED_LABELS) <= set(labels)
    for model in njp.TRACK_MODELS:
        assert model.d_t_Gy > 0.0
        assert len(model.rbe_alpha) == len(njp.TABLE1_LET_KEV_UM)
        # RBE rises with LET over these two qualities for every variant.
        assert model.rbe_alpha[1] > model.rbe_alpha[0]
        assert model.rbe_at(13.0)[0] == model.rbe_alpha[0]
        assert model.rbe_at(77.0)[0] == model.rbe_alpha[1]
    with pytest.raises(KeyError):
        njp.get_track_model("EDR11")  # surveyed by the paper, absent from Table 1
    with pytest.raises(KeyError):
        njp.TRACK_MODELS[0].rbe_at(50.0)


def test_the_two_2008_papers_are_not_confused() -> None:
    """``elsasser2008`` is 2008a (LEM III, IJROBP); this module is 2008b (NJP).

    Friedrich 2012 cites them separately and they are easy to mix up -- the LEM IV
    reconstruction did exactly that once. They do share the 13/77 keV/um beam
    qualities, which is part of why.
    """
    from pyradbio.benchmark import elsasser2008 as lem3_paper

    assert njp.ELSASSER2008B_DOI == "10.1088/1367-2630/10/7/075005"
    assert lem3_paper.__doc__ is not None
    assert "ijrobp" in lem3_paper.__doc__.lower()
