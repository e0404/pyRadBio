"""Photon reference survival (shouldered -> exponential)."""

from __future__ import annotations

import math

import pytest

from pyradbio.core.lq import dose_for_survival
from pyradbio.core.photon import ClusterParameters, PhotonParameters


def test_pure_lq_when_no_transition_dose() -> None:
    p = PhotonParameters(alpha_x=0.3, beta_x=0.05)
    assert p.reference_survival(2.0) == pytest.approx(math.exp(-(0.3 * 2.0 + 0.05 * 4.0)))


def test_lq_below_transition_dose() -> None:
    p = PhotonParameters(alpha_x=0.3, beta_x=0.05, D_t=5.0)
    # below D_t the curve is plain LQ
    assert p.log_kill(2.0) == pytest.approx(0.3 * 2.0 + 0.05 * 4.0)


def test_exponential_tail_above_transition() -> None:
    alpha, beta, d_t = 0.3, 0.05, 5.0
    p = PhotonParameters(alpha_x=alpha, beta_x=beta, D_t=d_t)
    s_max = alpha + 2.0 * beta * d_t
    assert p.max_slope() == pytest.approx(s_max)
    # above D_t the log-kill grows linearly with slope s_max
    lk_t = alpha * d_t + beta * d_t * d_t
    assert p.log_kill(7.0) == pytest.approx(lk_t + s_max * 2.0)


def test_survival_continuous_at_transition() -> None:
    p = PhotonParameters(alpha_x=0.3, beta_x=0.05, D_t=5.0)
    just_below = p.reference_survival(5.0 - 1e-6)
    just_above = p.reference_survival(5.0 + 1e-6)
    assert just_below == pytest.approx(just_above, rel=1e-4)


def test_max_slope_requires_transition() -> None:
    with pytest.raises(ValueError):
        PhotonParameters(alpha_x=0.3, beta_x=0.05).max_slope()


def test_validation() -> None:
    with pytest.raises(ValueError):
        PhotonParameters(alpha_x=-0.1, beta_x=0.05)
    with pytest.raises(ValueError):
        PhotonParameters(alpha_x=0.3, beta_x=0.05, D_t=-1.0)
    with pytest.raises(ValueError):
        PhotonParameters(alpha_x=0.3, beta_x=0.05).log_kill(-1.0)


def test_cluster_defaults() -> None:
    c = ClusterParameters()
    assert c.ssb_per_gy_cell == 1250.0
    assert c.dsb_per_gy_cell == 30.0


def test_dose_for_survival_round_trips_below_and_above_the_transition() -> None:
    """D(S) must invert the shouldered->exponential curve on both branches."""
    photon = PhotonParameters(alpha_x=0.136, beta_x=0.040, D_t=12.0)

    for level in (0.9, 0.5, 0.01, 1e-4, 1e-8):
        dose = photon.dose_for_survival(level)
        assert photon.reference_survival(dose) == pytest.approx(level, rel=1e-12)


def test_dose_for_survival_exceeds_the_plain_lq_root_past_the_transition() -> None:
    """Above D_t the LQ root is too small; using it would inflate any RBE.

    For the Scholz & Kraft 1996 V79 inputs the 1% photon dose is 9.2 Gy against
    D_t = 12 Gy, so the two branches agree there -- but only for those numbers.
    """
    photon = PhotonParameters(alpha_x=0.136, beta_x=0.040, D_t=12.0)

    assert photon.dose_for_survival(0.01) == pytest.approx(
        dose_for_survival(0.01, photon.alpha_x, photon.beta_x), rel=1e-12
    )
    deep = 1.0e-6
    assert photon.dose_for_survival(deep) > dose_for_survival(deep, photon.alpha_x, photon.beta_x)


def test_dose_for_survival_without_transition_is_the_lq_inversion() -> None:
    photon = PhotonParameters(alpha_x=0.2, beta_x=0.03)

    assert photon.dose_for_survival(0.05) == pytest.approx(
        dose_for_survival(0.05, 0.2, 0.03), rel=1e-12
    )
