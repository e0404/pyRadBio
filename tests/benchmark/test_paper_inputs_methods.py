"""``PaperCellInputs.photon()`` / ``.cell_target()`` equal the hand constructions.

Every benchmark ``PaperCellInputs`` class has explicit ``photon()`` and
``cell_target()`` methods so that the examples and tests do not have to spell
``PhotonParameters(alpha_x=inp.alpha_x, ...)`` and
``CellTarget(nucleus_radius_um=..., effective_size_factor=...)`` by hand. Each test
below writes out the hand construction and asserts the method returns an *equal*
object. Both classes are frozen dataclasses, so ``==`` compares every field and
floats compare by exact value, which makes the derived Scholz & Kraft 1996 radius
bit-identical to the hand ``math.sqrt(A / math.pi)``.
"""

from __future__ import annotations

import math

import pytest

from pyradbio.benchmark import elsasser2007 as es07
from pyradbio.benchmark import elsasser2010 as es10
from pyradbio.benchmark import friedrich2012 as f12
from pyradbio.benchmark import scholz1997
from pyradbio.benchmark import scholz_kraft1996 as sk
from pyradbio.core.cell import CellTarget
from pyradbio.core.photon import PhotonParameters

# --------------------------------------------------------------------------- #
# Scholz & Kraft 1996: A_nucl_um2, no radius -> radius = sqrt(A / pi)
# --------------------------------------------------------------------------- #

_SK_INPUTS = {**sk.FIGURE_INPUTS, "FIG1A_BELLI_INPUTS": sk.FIG1A_BELLI_INPUTS}


@pytest.mark.parametrize("key", sorted(_SK_INPUTS))
def test_scholz_kraft1996_photon_matches_hand_construction(key: str) -> None:
    inputs = _SK_INPUTS[key]
    keyword = PhotonParameters(alpha_x=inputs.alpha_x, beta_x=inputs.beta_x, D_t=inputs.D_t)
    # positional form
    positional = PhotonParameters(inputs.alpha_x, inputs.beta_x, inputs.D_t)

    photon = inputs.photon()

    assert type(photon) is PhotonParameters
    assert photon == keyword
    assert photon == positional
    assert photon.cluster is None


@pytest.mark.parametrize("key", sorted(_SK_INPUTS))
def test_scholz_kraft1996_cell_target_matches_hand_construction(key: str) -> None:
    inputs = _SK_INPUTS[key]
    explicit = CellTarget(
        nucleus_radius_um=math.sqrt(inputs.A_nucl_um2 / math.pi),
        effective_size_factor=1.0,
    )
    # relies on the 1.0 library default
    defaulted = CellTarget(math.sqrt(inputs.A_nucl_um2 / math.pi))

    cell = inputs.cell_target()

    assert type(cell) is CellTarget
    assert cell == explicit
    assert cell == defaulted
    assert cell.effective_size_factor == 1.0
    assert cell.domain_radius_um is None
    assert cell.height_um is None


def test_scholz_kraft1996_matches_literal_lem1_fixtures() -> None:
    """The LEM I test fixtures spell the SK1996 V79 record out as literals."""
    v79 = sk.FIGURE_INPUTS["fig3_sigma_vs_energy_v79"]
    assert v79.photon() == PhotonParameters(alpha_x=0.136, beta_x=0.040, D_t=12.0)
    assert v79.cell_target() == CellTarget(
        nucleus_radius_um=math.sqrt(50.0 / math.pi), effective_size_factor=1.0
    )
    assert v79.cell_target() == CellTarget(nucleus_radius_um=math.sqrt(50.0 / math.pi))


# --------------------------------------------------------------------------- #
# Scholz et al. 1997: D_t + radius (single CHO record)
# --------------------------------------------------------------------------- #


def test_scholz1997_photon_matches_hand_construction() -> None:
    inputs = scholz1997.CHO_INPUTS
    hand = PhotonParameters(alpha_x=inputs.alpha_x, beta_x=inputs.beta_x, D_t=inputs.D_t)

    photon = inputs.photon()

    assert type(photon) is PhotonParameters
    assert photon == hand
    assert photon.cluster is None
    # the literal fixture of the LEM I tests
    assert photon == PhotonParameters(alpha_x=0.18, beta_x=0.028, D_t=30.0)


def test_scholz1997_cell_target_matches_hand_construction() -> None:
    inputs = scholz1997.CHO_INPUTS
    hand = CellTarget(
        nucleus_radius_um=inputs.nucleus_radius_um,
        effective_size_factor=inputs.effective_size_factor,
    )

    cell = inputs.cell_target()

    assert type(cell) is CellTarget
    assert cell == hand
    # the literal fixture of the LEM I tests
    assert cell == CellTarget(nucleus_radius_um=5.0, effective_size_factor=1.0)


def test_scholz1997_cell_target_passes_effective_size_factor_on() -> None:
    """A non-default factor reaches the CellTarget (it is passed explicitly)."""
    inputs = scholz1997.PaperCellInputs(0.18, 0.028, 30.0, 5.0, effective_size_factor=0.8)

    assert inputs.cell_target().effective_size_factor == 0.8


# --------------------------------------------------------------------------- #
# Elsasser & Scholz 2007: D_t_old / D_t_new -> photon(which)
# --------------------------------------------------------------------------- #


@pytest.mark.parametrize("key", sorted(es07.FIGURE_INPUTS))
def test_elsasser2007_photon_old_matches_hand_construction(key: str) -> None:
    inp = es07.FIGURE_INPUTS[key]
    hand = PhotonParameters(alpha_x=inp.alpha_x, beta_x=inp.beta_x, D_t=inp.D_t_old)

    photon = inp.photon("old")

    assert type(photon) is PhotonParameters
    assert photon == hand
    assert photon.cluster is None


@pytest.mark.parametrize("key", sorted(es07.FIGURE_INPUTS))
def test_elsasser2007_photon_new_matches_hand_construction(key: str) -> None:
    inp = es07.FIGURE_INPUTS[key]
    hand = PhotonParameters(alpha_x=inp.alpha_x, beta_x=inp.beta_x, D_t=inp.D_t_new)

    photon = inp.photon("new")

    assert type(photon) is PhotonParameters
    assert photon == hand
    assert photon.cluster is None


def test_elsasser2007_photon_literal_fixture() -> None:
    # the literal fixture of the LEM I tests ("Fig. 6 'LEM old'")
    v79 = es07.FIGURE_INPUTS["fig6_rbealpha_vs_let"]

    assert v79.photon("old") == PhotonParameters(alpha_x=0.093, beta_x=0.026, D_t=30.0)


def test_elsasser2007_photon_rejects_unknown_variant() -> None:
    v79 = es07.FIGURE_INPUTS["fig6_rbealpha_vs_let"]

    with pytest.raises(ValueError, match="'old' or 'new'"):
        v79.photon("lem2")  # type: ignore[arg-type]


@pytest.mark.parametrize("key", sorted(es07.FIGURE_INPUTS))
def test_elsasser2007_cell_target_matches_hand_construction(key: str) -> None:
    inp = es07.FIGURE_INPUTS[key]
    hand = CellTarget(
        nucleus_radius_um=inp.nucleus_radius_um,
        effective_size_factor=inp.effective_size_factor,
    )

    cell = inp.cell_target()

    assert type(cell) is CellTarget
    assert cell == hand
    # the literal 1.0 and the library default build the same object for every entry.
    assert cell == CellTarget(nucleus_radius_um=inp.nucleus_radius_um, effective_size_factor=1.0)
    assert cell == CellTarget(nucleus_radius_um=inp.nucleus_radius_um)


def test_elsasser2007_cell_target_literal_fixture() -> None:
    # the literal fixture of the LEM I tests (V79, r = 3.8 um)
    assert es07.FIGURE_INPUTS["fig6_rbealpha_vs_let"].cell_target() == CellTarget(
        nucleus_radius_um=3.8
    )


def test_elsasser2007_cell_target_passes_effective_size_factor_on() -> None:
    inp = es07.PaperCellInputs(0.093, 0.026, 30.0, 7.5, 3.8, (32.0, 153.0), 0.8)

    assert inp.cell_target().effective_size_factor == 0.8


# --------------------------------------------------------------------------- #
# Elsasser et al. 2010 / Friedrich et al. 2012: D_t + radius
# --------------------------------------------------------------------------- #


@pytest.mark.parametrize("key", sorted(es10.FIGURE_INPUTS))
def test_elsasser2010_photon_matches_hand_construction(key: str) -> None:
    inp = es10.FIGURE_INPUTS[key]
    hand = PhotonParameters(alpha_x=inp.alpha_x, beta_x=inp.beta_x, D_t=inp.D_t)

    photon = inp.photon()

    assert type(photon) is PhotonParameters
    assert photon == hand
    assert photon.cluster is None


@pytest.mark.parametrize("key", sorted(es10.FIGURE_INPUTS))
def test_elsasser2010_cell_target_matches_hand_construction(key: str) -> None:
    inp = es10.FIGURE_INPUTS[key]
    hand = CellTarget(inp.nucleus_radius_um, effective_size_factor=inp.effective_size_factor)

    cell = inp.cell_target()

    assert type(cell) is CellTarget
    assert cell == hand


def test_elsasser2010_cell_target_passes_effective_size_factor_on() -> None:
    assert (
        es10.PaperCellInputs(0.129, 0.049, 3.0, 5.0, 0.8).cell_target().effective_size_factor == 0.8
    )


@pytest.mark.parametrize("key", sorted(f12.FIGURE_INPUTS))
def test_friedrich2012_photon_matches_hand_construction(key: str) -> None:
    inp = f12.FIGURE_INPUTS[key]
    hand = PhotonParameters(alpha_x=inp.alpha_x, beta_x=inp.beta_x, D_t=inp.D_t)

    photon = inp.photon()

    assert type(photon) is PhotonParameters
    assert photon == hand
    assert photon.cluster is None


@pytest.mark.parametrize("key", sorted(f12.FIGURE_INPUTS))
def test_friedrich2012_cell_target_matches_hand_construction(key: str) -> None:
    inp = f12.FIGURE_INPUTS[key]
    hand = CellTarget(inp.nucleus_radius_um, effective_size_factor=inp.effective_size_factor)

    cell = inp.cell_target()

    assert type(cell) is CellTarget
    assert cell == hand


def test_friedrich2012_matches_literal_lem4_fixtures() -> None:
    """The LEM IV test fixtures spell these same two figures out as literals."""
    fig3 = f12.FIGURE_INPUTS["fig3_v79"]
    fig4 = f12.FIGURE_INPUTS["fig4_hypothetical"]
    assert fig3.photon() == PhotonParameters(alpha_x=0.184, beta_x=0.02, D_t=17.0)
    assert fig3.cell_target() == CellTarget(nucleus_radius_um=4.7)
    assert fig4.photon() == PhotonParameters(alpha_x=0.25, beta_x=0.05, D_t=9.5)
    assert fig4.cell_target() == CellTarget(nucleus_radius_um=5.0)


def test_friedrich2012_cell_target_passes_effective_size_factor_on() -> None:
    assert (
        f12.PaperCellInputs(0.184, 0.02, 17.0, 4.7, 0.8).cell_target().effective_size_factor == 0.8
    )
