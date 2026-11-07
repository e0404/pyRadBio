"""The oxygen axis -- Alper & Howard-Flanders relative radioresistance.

Physics acceptance for :mod:`pyradbio.core.oxygen`, the model-agnostic half of
the OSMK work (Inaniwa & Kanematsu 2021 eq. 13 / 2023 eq. 1; restated in
``docs/references/OSMK_Inaniwa2021_2023.md``):

* the two limits are exact -- ``R(0) = Rmax`` and ``R(inf) = 1``;
* ``R`` decreases monotonically with ``pO2`` and never drops below 1;
* the ``pO2 = K`` point is the *harmonic* midpoint ``2 Rmax/(1 + Rmax)``, which
  is what eq. (13) says and **not** the ``(1 + Rmax)/2`` its prose claims.
"""

from __future__ import annotations

import math

import numpy as np
import pytest

from pyradbio.core.oxygen import (
    AEROBIC_PO2_MMHG,
    ANOXIC_PO2_MMHG,
    K_HALF_EFFECT_MMHG,
    OxygenCondition,
    relative_radioresistance,
)

RMAX_HSG_2023 = 4.46  # OSMK 2023 table 1


def test_anoxic_limit_is_rmax_exactly() -> None:
    """``R(0) = K / (K/Rmax) = Rmax`` -- no dependence on ``K``."""
    for k in (1.0, 3.0, 10.0):
        assert relative_radioresistance(0.0, k_mmHg=k, r_max=RMAX_HSG_2023) == pytest.approx(
            RMAX_HSG_2023, rel=1e-12
        )


def test_aerobic_limit_approaches_unity() -> None:
    """``R -> 1`` as ``pO2 -> inf``: the oxygen correction switches itself off."""
    r_inf = relative_radioresistance(math.inf, k_mmHg=3.0, r_max=RMAX_HSG_2023)
    assert r_inf == pytest.approx(1.0, rel=1e-12)
    assert relative_radioresistance(1.0e6, k_mmHg=3.0, r_max=RMAX_HSG_2023) == pytest.approx(
        1.0, rel=1e-5
    )


def test_experimental_aerobic_condition_is_not_unity() -> None:
    """160 mmHg is *not* ``R = 1`` -- gotcha 2 of the reference summary.

    OSMK fitted at "aerobic" carries a ~1.4% residual radioresistance, so an
    OSMK model evaluated at 160 mmHg is not its own ``pO2 -> inf`` limit.
    """
    r = relative_radioresistance(AEROBIC_PO2_MMHG, k_mmHg=3.0, r_max=RMAX_HSG_2023)
    assert r == pytest.approx(1.0145, abs=5e-4)
    assert r > 1.0


def test_half_effect_point_is_the_harmonic_midpoint() -> None:
    """At ``pO2 = K``, eq. (13) gives ``2 Rmax/(1 + Rmax)``, not ``(1 + Rmax)/2``.

    Both papers' prose claims the arithmetic midpoint; their own equation says
    otherwise. Pin the algebra.
    """
    k = 3.0
    r = relative_radioresistance(k, k_mmHg=k, r_max=RMAX_HSG_2023)
    assert r == pytest.approx(2.0 * RMAX_HSG_2023 / (1.0 + RMAX_HSG_2023), rel=1e-12)
    assert r != pytest.approx((1.0 + RMAX_HSG_2023) / 2.0, rel=1e-3)


def test_monotone_decreasing_and_bounded() -> None:
    po2 = np.geomspace(1e-3, 1e4, 200)
    r = np.array([relative_radioresistance(p, k_mmHg=3.0, r_max=RMAX_HSG_2023) for p in po2])
    assert np.all(np.diff(r) < 0.0)
    assert np.all(r >= 1.0) and np.all(r <= RMAX_HSG_2023)


def test_vectorises_over_po2() -> None:
    po2 = np.array([0.0, 3.0, 160.0])
    r = relative_radioresistance(po2, k_mmHg=3.0, r_max=RMAX_HSG_2023)
    assert r.shape == po2.shape
    assert r[0] == pytest.approx(RMAX_HSG_2023, rel=1e-12)


def test_rmax_of_one_means_no_oxygen_effect_at_any_pressure() -> None:
    for p in (0.0, 1.0, 160.0):
        assert relative_radioresistance(p, k_mmHg=3.0, r_max=1.0) == pytest.approx(1.0, rel=1e-12)


@pytest.mark.parametrize("bad", [-1.0, -0.001])
def test_negative_po2_rejected(bad: float) -> None:
    with pytest.raises(ValueError, match="pO2"):
        relative_radioresistance(bad, k_mmHg=3.0, r_max=RMAX_HSG_2023)


def test_rmax_below_one_rejected() -> None:
    """``Rmax < 1`` would mean hypoxia *sensitises* -- not a thing this form models."""
    with pytest.raises(ValueError, match="r_max"):
        relative_radioresistance(10.0, k_mmHg=3.0, r_max=0.5)


def test_nonpositive_k_rejected() -> None:
    with pytest.raises(ValueError, match="k_mmHg"):
        relative_radioresistance(10.0, k_mmHg=0.0, r_max=2.0)


class TestOxygenCondition:
    def test_named_conditions(self) -> None:
        assert OxygenCondition.anoxic().pO2_mmHg == ANOXIC_PO2_MMHG == 0.0
        assert OxygenCondition.aerobic().pO2_mmHg == AEROBIC_PO2_MMHG == 160.0

    def test_default_k_is_the_papers_assumption(self) -> None:
        """``K = 3 mmHg`` for every cell line and every LET (OSMK 2021 sec 3.2)."""
        assert K_HALF_EFFECT_MMHG == 3.0
        assert OxygenCondition(5.0).k_mmHg == 3.0

    def test_radioresistance_delegates(self) -> None:
        cond = OxygenCondition(pO2_mmHg=0.0)
        assert cond.relative_radioresistance(RMAX_HSG_2023) == pytest.approx(RMAX_HSG_2023)

    def test_negative_pressure_rejected(self) -> None:
        with pytest.raises(ValueError, match="pO2"):
            OxygenCondition(-1.0)
