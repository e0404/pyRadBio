"""Shared LEM machinery in :mod:`pyradbio.models.lem.base`.

Everything here is exercised by all four LEM versions, so a defect is a defect
four times over. The version-specific behaviour lives in ``test_lem1.py`` and
friends.
"""

from __future__ import annotations

import math

import numpy as np
import numpy.typing as npt
import pytest

from pyradbio.core.cell import CellTarget
from pyradbio.core.photon import ClusterParameters, PhotonParameters
from pyradbio.core.radiation import Ion
from pyradbio.models.lem import base as lem_base
from pyradbio.models.lem.base import _angular_fraction_inside
from pyradbio.models.lem.lem1 import LEM1
from pyradbio.models.lem.lem2 import LEM2
from pyradbio.models.lem.lem3 import LEM3
from pyradbio.track.stopping_power import SimpleBetheStoppingPower

_R_EFF = 5.0


def _model(cls: type) -> LEM1 | LEM2 | LEM3:
    return cls(
        photon=PhotonParameters(alpha_x=0.129, beta_x=0.049, D_t=3.0, cluster=ClusterParameters()),
        cell=CellTarget(nucleus_radius_um=_R_EFF),
        stopping_power=SimpleBetheStoppingPower(),
    )


def test_angular_fraction_is_finite_on_the_degenerate_shell() -> None:
    """``s = 0`` must not divide by zero.

    The chord half-angle carries a ``1/(2 s b)``, and ``s = 0`` is reachable:
    ``_impact_log_kill`` starts its shell at ``b - r_eff``, which is exactly zero
    for a track grazing the nuclear edge. The circle of radius zero is the single
    point at distance ``b``, so it is inside the nucleus exactly when
    ``b <= r_eff``.
    """
    s = np.array([0.0])
    assert _angular_fraction_inside(s, _R_EFF, _R_EFF)[0] == pytest.approx(2.0 * math.pi)
    assert _angular_fraction_inside(s, 2.0 * _R_EFF, _R_EFF)[0] == 0.0

    # and the b = 0 concentric branch stays what it was
    inside = _angular_fraction_inside(np.array([0.5 * _R_EFF]), 0.0, _R_EFF)
    assert inside[0] == pytest.approx(2.0 * math.pi)


@pytest.mark.parametrize("cls", [LEM1, LEM2, LEM3])
def test_alpha_survives_an_impact_grid_that_lands_on_the_nuclear_edge(
    cls: type, monkeypatch: pytest.MonkeyPatch
) -> None:
    """A node at exactly ``b = r_eff`` must not poison the quadrature.

    ``_IMPACT_NEAR_NODES`` spans ``[0, 1.5 r_eff]``, so whether any node lands on
    ``b = r_eff`` is pure arithmetic: the shipped 200 misses it (199/1.5 is not an
    integer), 400 hits it dead on. The shipped value is therefore *lucky*, not
    safe, and this pins the unlucky one.
    """
    ion = Ion(Z=6, A=12, energy_MeV_u=100.0)
    reference = _model(cls).alpha_beta(ion).alpha

    monkeypatch.setattr(lem_base, "_IMPACT_NEAR_NODES", 400)
    assert np.any(np.linspace(0.0, 1.5 * _R_EFF, 400) == _R_EFF), "grid no longer hits the edge"

    alpha = _model(cls).alpha_beta(ion).alpha
    assert math.isfinite(alpha)
    # A finer grid, so it may legitimately differ -- but only at the sub-percent
    # level, and must be finite (no NaN).
    assert alpha == pytest.approx(reference, rel=0.01)


def test_delete_one_jackknife_reproduces_the_standard_error_for_a_linear_observable() -> None:
    """For a *linear* functional the jackknife must equal the plain standard error.

    A delete-one jackknife of ``f(g) = c.g`` collapses algebraically to
    ``sqrt(sum (c.b_i - c.b_bar)^2 / (n (n-1)))``, i.e. the standard error of the
    per-batch values, so this pins the estimator against a closed form rather
    than against itself. Nonlinear observables are where the jackknife earns its
    keep, but they have no closed form to check.
    """
    rng = np.random.default_rng(20260813)
    batches = rng.normal(size=(11, 4))
    weights = np.array([0.25, -1.5, 2.0, 0.75])

    value, error = lem_base._delete_one_jackknife(batches, lambda g: float(weights @ g))

    per_batch = batches @ weights
    assert value == pytest.approx(float(per_batch.mean()))
    assert error == pytest.approx(float(per_batch.std(ddof=1) / math.sqrt(per_batch.size)))


def test_delete_one_jackknife_reports_nan_rather_than_inventing_an_error() -> None:
    """Too few surviving replicates must give ``nan``, not a fabricated number.

    A deleted batch can push a nonlinear observable out of range (a root-find
    with no bracket); those replicates are dropped, and below two survivors
    there is nothing to estimate a spread from.
    """

    def only_works_on_the_full_sample(g: npt.NDArray[np.float64]) -> float:
        if g[0] < 100.0:
            raise ValueError("out of range")
        return float(g[0])

    batches = np.full((5, 1), 200.0)
    batches[0, 0] = -300.0  # pooled mean 100; every delete-one replicate but one is 75

    value, error = lem_base._delete_one_jackknife(batches, only_works_on_the_full_sample)

    assert value == pytest.approx(100.0)
    assert math.isnan(error)
