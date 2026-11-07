"""The local-dose Monte Carlo on ``LEMBase``, and its compiled kernel's eta table.

The kernel evaluates the photon response *itself* -- it cannot call back into
Python -- so the one thing that differs between LEM versions above ``D_t``, the
cluster enhancement ``eta(D)`` of eq. 10, is handed to it as a table. If that
table is wrong or missing, the kernel silently simulates **LEM I** under another
version's name. These tests pin every part of that contract.
"""

from __future__ import annotations

import numpy as np
import pytest

from pyradbio import CellTarget, Ion, PhotonParameters
from pyradbio.models.lem import _mc_kernel
from pyradbio.models.lem.lem1 import LEM1
from pyradbio.models.lem.lem2 import LEM2, ClusterModel, Friedrich2015ClusterModel
from pyradbio.models.lem.lem3 import LEM3
from pyradbio.models.lem.lem4 import LEM4

ALPHA_X, BETA_X = 0.313, 0.0615
D_T = {"LEM1": 30.0, "LEM2": 6.0, "LEM3": 19.0}


def _photon(d_t: float) -> PhotonParameters:
    return PhotonParameters(alpha_x=ALPHA_X, beta_x=BETA_X, D_t=d_t)


def _model(cls, d_t: float, **kwargs):
    return cls(photon=_photon(d_t), cell=CellTarget(5.0), **kwargs)


def _kernel_log_kill(model, dose: np.ndarray) -> np.ndarray:
    """``-ln S_x(D)`` as the compiled kernel computes it, for the whole grid."""
    d_t = model.photon.D_t
    lk_t = ALPHA_X * d_t + BETA_X * d_t * d_t
    eta_log_d, eta_val = model._kernel_eta_table()
    return np.array(
        [
            _mc_kernel._logkill(
                float(d), ALPHA_X, BETA_X, d_t, model.photon.max_slope(), lk_t, eta_log_d, eta_val
            )
            for d in dose
        ]
    )


@pytest.mark.parametrize(
    ("cls", "name", "kwargs"),
    [
        (LEM1, "LEM1", {}),
        (LEM2, "LEM2", {}),
        (LEM3, "LEM3", {}),
        (LEM2, "LEM2", {"cluster": Friedrich2015ClusterModel()}),
    ],
)
def test_kernel_response_matches_the_model(cls, name, kwargs) -> None:
    """The kernel's ``-ln S_x`` is the model's own, across nine decades of dose."""
    model = _model(cls, D_T[name], **kwargs)
    dose = np.logspace(-1.0, 8.0, 400)
    got = _kernel_log_kill(model, dose)
    want = model._photon_log_kill_array(dose)
    assert np.max(np.abs(got - want) / np.maximum(np.abs(want), 1e-12)) < 1e-5


def test_tabulated_cluster_is_exact_not_merely_close() -> None:
    """A tabulated eta needs no resampling, so LEM II/III should be at round-off.

    `ClusterModel` already interpolates linearly in log10 D, which is exactly what
    the kernel does, so handing over its own nodes costs nothing. If this ever
    degrades to ~1e-6 someone has inserted a resampling step.
    """
    dose = np.logspace(-1.0, 8.0, 400)
    for cls, name in ((LEM2, "LEM2"), (LEM3, "LEM3")):
        model = _model(cls, D_T[name])
        want = model._photon_log_kill_array(dose)
        got = _kernel_log_kill(model, dose)
        assert np.max(np.abs(got - want) / np.maximum(np.abs(want), 1e-12)) < 1e-14


def test_eta_actually_reaches_the_kernel() -> None:
    """LEM II's kernel response must exceed LEM I's at the same ``D_t``.

    The failure this guards against is silent: without the table the kernel still
    returns a perfectly plausible number -- LEM I's.
    """
    lem1 = _model(LEM1, 6.0)
    lem2 = _model(LEM2, 6.0)
    dose = np.array([1.0e3, 1.0e5, 5.0e5])
    r1, r2 = _kernel_log_kill(lem1, dose), _kernel_log_kill(lem2, dose)
    assert np.all(r2 > r1)
    # at the eta peak the enhancement is an order of magnitude, not a nudge
    assert r2[-1] / r1[-1] > 10.0


def test_lem1_table_is_empty_so_its_arithmetic_is_untouched() -> None:
    """LEM I must take the kernel's original branch, not an eta branch with eta=1."""
    eta_log_d, eta_val = _model(LEM1, 30.0)._kernel_eta_table()
    assert eta_log_d.size == 0
    assert eta_val.size == 0


def test_check_mc_response_rejects_a_mismatched_table() -> None:
    """A model whose response the kernel cannot express must refuse to run on numba."""

    class WrongEta(LEM2):
        def _kernel_eta_table(self):
            log_d, eta = super()._kernel_eta_table()
            return log_d, eta * 2.0  # not this model's eta

    model = WrongEta(photon=_photon(6.0), cell=CellTarget(5.0))
    with pytest.raises(NotImplementedError, match="not reproduced by its _kernel_eta_table"):
        model._check_mc_response()


def test_cluster_without_kernel_table_is_a_pointed_error() -> None:
    class ScalarOnlyCluster(ClusterModel):
        def kernel_eta_table(self):  # type: ignore[override]
            raise AttributeError("no table")

    model = LEM2(photon=_photon(6.0), cell=CellTarget(5.0), cluster=ScalarOnlyCluster())
    with pytest.raises(AttributeError):
        model._kernel_eta_table()


@pytest.mark.parametrize(
    "method",
    [
        "scholz1997_montecarlo_survival",
        "scholz1997_montecarlo_beta",
        "scholz1997_traversal_conditioned_survival",
    ],
)
def test_lem4_refuses_the_local_dose_monte_carlo(method: str) -> None:
    """LEM IV's effect is not pointwise in the local dose, so the MC is not its model."""
    model = LEM4(photon=_photon(30.0), cell=CellTarget(5.0))
    with pytest.raises(NotImplementedError, match="no local-dose Monte Carlo"):
        getattr(model, method)(Ion(6, 12, 10.0), [1.0])


def test_monte_carlo_is_available_on_lem2_and_lem3() -> None:
    """The point of the move: LEM II/III can run the reference path at all."""
    for cls, name in ((LEM2, "LEM2"), (LEM3, "LEM3")):
        model = cls(photon=_photon(D_T[name]), cell=CellTarget(5.0))
        s = model.scholz1997_montecarlo_survival(
            Ion(6, 12, 10.0), [1.0, 3.0], n_realizations=200, seed=3, engine="numpy"
        )
        assert s.shape == (2,)
        assert np.all((s > 0.0) & (s <= 1.0))
        assert s[1] < s[0]
