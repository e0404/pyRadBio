"""Cross-check LEM's bespoke impact-parameter integrals against the general
``micro.impact_parameter`` primitive.

LEM's benchmark routes (``inactivation_cross_section``, ``lethal_full`` and the
central single-traversal ``N_1``) integrate the local log-kill ``-ln S_x(D(r))``
over the nuclear disk. `pyradbio.micro.nuclear_functional_impact_profile`
does the same integral independently (different grid, Simpson vs LEM's trapezoid,
no shell refinement). These tests pin that the two agree to <=1% across the
benchmark range, so a future change to either implementation that drifts is caught.
"""

from __future__ import annotations

import math

import numpy as np
import pytest

from pyradbio.core.cell import CellTarget
from pyradbio.core.photon import ClusterParameters, PhotonParameters
from pyradbio.core.radiation import Ion
from pyradbio.micro import NuclearImpactProfile, nuclear_functional_impact_profile
from pyradbio.models.lem.base import LEMBase
from pyradbio.models.lem.lem1 import LEM1
from pyradbio.models.lem.lem2 import LEM2

# V79 reference (Scholz & Kraft / Elsasser & Scholz 2007), as the LEM examples use.
_V79 = dict(alpha_x=0.18, beta_x=0.028, D_t=30.0)
_ENERGIES = (1.0, 3.0, 11.0, 30.0, 100.0, 400.0)


def _general_profile(lem: LEMBase, ion: Ion) -> NuclearImpactProfile:
    """The independent disk impact-parameter profile of LEM's own nu(b)."""
    return nuclear_functional_impact_profile(
        lem._radial(),
        ion,
        radius_um=lem.cell.effective_nucleus_radius_um(),
        transform=np.vectorize(lem._photon_log_kill),
        radial_nodes=600,
        core_nodes=600,
        penumbra_nodes=600,
    )


def _lem1() -> LEM1:
    return LEM1(photon=PhotonParameters(**_V79), cell=CellTarget(5.0, effective_size_factor=1.0))


@pytest.mark.parametrize("energy", _ENERGIES)
def test_cross_section_matches_general_primitive(energy: float) -> None:
    """``sigma = integral (1-exp(-nu)) 2 pi b db`` (the "cross_section" route)."""
    lem = _lem1()
    ion = Ion(6, 12, energy)
    assert _general_profile(lem, ion).inactivation_cross_section_um2() == pytest.approx(
        lem.inactivation_cross_section_um2(ion), rel=0.01
    )


@pytest.mark.parametrize("energy", _ENERGIES)
def test_lethal_full_area_matches_general_primitive(energy: float) -> None:
    """``integral nu(b) 2 pi b db`` (the unsaturated "lethal_full" numerator)."""
    lem = _lem1()
    ion = Ion(6, 12, energy)
    r_eff = lem.cell.effective_nucleus_radius_um()
    lem_area = (
        lem.lethal_full_initial_slope(ion) * math.pi * r_eff**2 * lem.mean_traversal_dose_Gy(ion)
    )
    assert _general_profile(lem, ion).lethal_events_um2() == pytest.approx(lem_area, rel=0.01)


@pytest.mark.parametrize("energy", _ENERGIES)
def test_central_n1_matches_general_primitive(energy: float) -> None:
    """The central single-traversal ``N_1`` equals ``nu(b -> 0)`` ("lethal_central")."""
    lem = _lem1()
    ion = Ion(6, 12, energy)
    assert _general_profile(lem, ion).central_value() == pytest.approx(
        lem.single_traversal_log_kill(ion), rel=0.01
    )


@pytest.mark.parametrize("energy", [3.0, 11.0, 30.0, 100.0])
def test_cross_section_matches_with_cluster_extension(energy: float) -> None:
    """The cross-check still holds with LEM II's radical-diffusion D(r) + eta(D)."""
    lem = LEM2(
        photon=PhotonParameters(alpha_x=0.093, beta_x=0.026, D_t=7.5, cluster=ClusterParameters()),
        cell=CellTarget(3.8, effective_size_factor=1.0),
    )
    ion = Ion(6, 12, energy)
    assert _general_profile(lem, ion).inactivation_cross_section_um2() == pytest.approx(
        lem.inactivation_cross_section_um2(ion), rel=0.01
    )
