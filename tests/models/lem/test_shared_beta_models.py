"""The beta models shared across LEM I-III, and the LEM IV boundaries.

``slope_matching`` / ``rapid_lem`` / ``scholz1997_two_track`` live on
`pyradbio.models.lem.base.LEMBase` because all three only need the
lethal-event density to be a *pointwise* function of the local dose -- true for
LEM I-III, false for LEM IV. These tests pin both halves of that: the shared
family works for LEM II/III, and the LEM IV paths that would be silently wrong
raise instead.
"""

from __future__ import annotations

import numpy as np
import pytest

from pyradbio.core.cell import CellTarget
from pyradbio.core.photon import PhotonParameters
from pyradbio.core.radiation import Ion
from pyradbio.models.lem.base import SHARED_BETA_MODELS, LEMBase
from pyradbio.models.lem.lem1 import LEM1
from pyradbio.models.lem.lem2 import LEM2, ClusterModel, ClusterResponse, Friedrich2015ClusterModel
from pyradbio.models.lem.lem3 import LEM3
from pyradbio.models.lem.lem4 import LEM4

ION = Ion(6, 12, 100.0)
D_T = {LEM1: 30.0, LEM2: 6.0, LEM3: 19.0, LEM4: 6.0}


def _model(cls, **kwargs):
    photon = PhotonParameters(alpha_x=0.313, beta_x=0.0615, D_t=D_T[cls])
    return cls(photon=photon, cell=CellTarget(5.0), **kwargs)


# ---- the shared family reaches LEM II and LEM III -----------------------
@pytest.mark.parametrize("cls", [LEM2, LEM3])
@pytest.mark.parametrize("beta_model", SHARED_BETA_MODELS)
def test_lem2_lem3_support_every_shared_beta_model(cls, beta_model) -> None:
    lq = _model(cls, beta_model=beta_model).alpha_beta(ION)
    assert lq.alpha > 0.0
    assert 0.0 <= lq.beta < 1.0


@pytest.mark.parametrize("cls", [LEM2, LEM3])
def test_the_three_shared_models_actually_disagree(cls) -> None:
    """Otherwise the parametrisation above would pass on a stub."""
    betas = {bm: _model(cls, beta_model=bm).alpha_beta(ION).beta for bm in SHARED_BETA_MODELS}
    assert len(set(betas.values())) == 3, betas
    # alpha is untouched by the beta model -- they differ only in the quadratic term.
    alphas = {_model(cls, beta_model=bm).alpha_beta(ION).alpha for bm in SHARED_BETA_MODELS}
    assert len(alphas) == 1


@pytest.mark.parametrize("cls", [LEM2, LEM3])
def test_invalid_beta_model_is_rejected_at_construction(cls) -> None:
    with pytest.raises(ValueError, match="beta_model must be one of"):
        _model(cls, beta_model="scholz1997_montecarlo")  # LEM I only


def test_lem1_still_dispatches_the_shared_models_through_the_base() -> None:
    """LEM I keeps its superset; the shared three must go through one implementation."""
    lem = _model(LEM1, beta_model="rapid_lem")
    assert lem.alpha_beta(ION).beta == pytest.approx(lem.rapid_lem_beta(ION))
    assert LEM1.rapid_lem_beta is LEMBase.rapid_lem_beta


# ---- the array photon response must track the scalar one ----------------
@pytest.mark.parametrize("cls", [LEM1, LEM2, LEM3])
def test_array_photon_response_matches_the_scalar_one(cls) -> None:
    """The invariant that makes ``two_track`` safe to inherit.

    The pair integrals go through ``_photon_log_kill_array``. A version that
    overrode only the scalar ``_photon_log_kill`` -- as LEM II/III once did --
    would silently hand the *un-enhanced* response to every array-based beta
    model, so the two must agree everywhere, including across ``D_t`` and deep
    into the cluster-enhanced tail.
    """
    lem = _model(cls)
    doses = np.array([0.1, 1.0, 5.0, 6.0, 19.0, 30.0, 100.0, 1.0e3, 1.0e5, 1.0e6])
    array = lem._photon_log_kill_array(doses)
    scalar = np.array([lem._photon_log_kill(float(d)) for d in doses])
    assert array == pytest.approx(scalar, rel=1e-12)


@pytest.mark.parametrize("cls", [LEM2, LEM3])
def test_the_cluster_enhancement_actually_reaches_the_array_response(cls) -> None:
    """Guards against the override existing but being a copy of the base one."""
    lem = _model(cls)
    dose = np.array([1.0e5])  # well above the ~1e3 Gy cluster onset
    enhanced = float(lem._photon_log_kill_array(dose)[0])
    plain = float(LEMBase._photon_log_kill_array(lem, dose)[0])
    assert enhanced > 5.0 * plain, (enhanced, plain)


@pytest.mark.parametrize("cluster", [ClusterModel(), Friedrich2015ClusterModel()])
def test_bundled_cluster_models_satisfy_the_cluster_response_protocol(cluster) -> None:
    """``LEM2.cluster`` is typed by `ClusterResponse`; both bundled models conform."""
    assert isinstance(cluster, ClusterResponse)
    lem = _model(LEM2, cluster=cluster)
    assert lem._photon_log_kill_array(np.array([1.0e5]))[0] > 0.0


@pytest.mark.parametrize("cluster", [ClusterModel(), Friedrich2015ClusterModel()])
def test_bundled_cluster_models_vectorise_eta_consistently(cluster) -> None:
    doses = np.array([1.0, 1.0e2, 1.0e3, 1.0e5, 1.0e6, 5.0e6])
    assert cluster.eta_array(doses) == pytest.approx(
        [cluster.eta(float(d)) for d in doses], rel=1e-12
    )


# ---- LEM IV: the boundaries ---------------------------------------------
def test_lem4_two_track_beta_raises_with_the_reason() -> None:
    lem = _model(LEM4)
    with pytest.raises(NotImplementedError, match="global DSB-pattern statistic"):
        lem.scholz1997_two_track_beta(ION)


def test_lem4_rejects_two_track_at_construction() -> None:
    with pytest.raises(ValueError, match="scholz1997_two_track"):
        _model(LEM4, beta_model="scholz1997_two_track")


def test_lem4_rapid_lem_uses_lem4s_own_single_traversal_effect() -> None:
    """The override is load-bearing, not cosmetic.

    Inheriting ``LEMBase.single_traversal_log_kill`` would integrate the LEM I-III
    local-dose response and return a plausible non-LEM-IV number, so pin that the
    two genuinely differ and that ``rapid_lem`` follows LEM IV's own one.
    """
    lem = _model(LEM4)
    own = lem.single_traversal_log_kill(ION)
    base = LEMBase.single_traversal_log_kill(lem, ION)
    assert own != pytest.approx(base, rel=1e-3)
    assert own == pytest.approx(lem._single_traversal_effect(ION)[0])

    lq = _model(LEM4, beta_model="rapid_lem").alpha_beta(ION)
    assert 0.0 <= lq.beta < 1.0
    assert lq.beta != pytest.approx(_model(LEM4).alpha_beta(ION).beta)
