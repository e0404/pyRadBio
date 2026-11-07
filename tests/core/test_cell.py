"""Cell / target geometry."""

from __future__ import annotations

import pytest

from pyradbio.core.cell import CellTarget


def test_effective_nucleus_radius() -> None:
    cell = CellTarget(nucleus_radius_um=5.0, effective_size_factor=0.8)
    assert cell.effective_nucleus_radius_um() == pytest.approx(4.0)


def test_effective_size_factor_defaults_to_one() -> None:
    """A radius passed in is used as-is unless the caller opts into shrinking it.

    Published LEM inputs quote the *effective* nuclear size already (Scholz &
    Kraft 1996's 50 um^2; Elsasser & Scholz 2007's 3.8 / 4.7 um), so converting
    again shrinks the area by 0.64 and inflates ``D_1 = LET/(rho*A)`` by 1.56x.
    This default was 0.8 and that silently displaced every LEM ``RBE_alpha`` peak
    by ~1.25x in LET -- for a long time mistaken for a LEM II cluster deficit.
    The 0.8 conversion belongs only where the radius is genuinely geometric
    (Suzuki 2000 areas; see docs/references/LEM_III_Elsasser2008.md).
    """
    cell = CellTarget(nucleus_radius_um=5.0)
    assert cell.effective_size_factor == pytest.approx(1.0)
    assert cell.effective_nucleus_radius_um() == pytest.approx(5.0)


def test_cylinder_height_default_is_diameter() -> None:
    cell = CellTarget(nucleus_radius_um=5.0)
    assert cell.cylinder_height_um() == pytest.approx(10.0)


def test_cylinder_height_explicit() -> None:
    cell = CellTarget(nucleus_radius_um=5.0, height_um=3.0)
    assert cell.cylinder_height_um() == pytest.approx(3.0)


@pytest.mark.parametrize(
    "kwargs",
    [
        {"nucleus_radius_um": 0.0},
        {"nucleus_radius_um": 5.0, "domain_radius_um": 0.0},
        {"nucleus_radius_um": 5.0, "effective_size_factor": 0.0},
        {"nucleus_radius_um": 5.0, "effective_size_factor": 1.5},
        {"nucleus_radius_um": 5.0, "height_um": -1.0},
    ],
)
def test_validation(kwargs: dict[str, float]) -> None:
    with pytest.raises(ValueError):
        CellTarget(**kwargs)
