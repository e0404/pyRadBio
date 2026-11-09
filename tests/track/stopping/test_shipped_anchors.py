"""CI anchors: the central accuracy checks against our committed, regridded tables.

The primary checks compare against the raw sources at their printed points and skip
when ``data/stopping/`` has not been fetched (it is git-ignored), i.e. in clean CI.
These twins run everywhere: they compare against the shipped
tables -- ``ICRU(edition="2016")`` (ICRU 90 p, alpha, C) and ``XSTAR()`` (PSTAR /
ASTAR, ICRU 49) -- on a dense grid, 40 points per decade. The shipped tables reproduce
the raw ones at every printed point to <= 8.3e-7 (``test_icru.py``, run where the raw
files exist), so the chain to the printed reports holds; between printed points they
add the tables' own interpolation uncertainty (~1e-3 near the stopping maximum, see
``test_interpolation_error_near_the_stopping_maximum_is_quantified``), which is why
these bounds are looser than the raw-point ones. All MEASURED; bounds ~1.5-2x.
"""

from __future__ import annotations

import numpy as np
import numpy.typing as npt
import pytest

from pyradbio.track.stopping import ICRU, XSTAR, Bethe, ICRU90Procedure
from pyradbio.track.stopping.base import MassStoppingPowerModel
from pyradbio.track.stopping.hubert import HubertAlphaScaling

POINTS_PER_DECADE = 40


def _grid(lo: float, hi: float) -> npt.NDArray[np.float64]:
    return np.geomspace(lo, hi, int(np.log10(hi / lo) * POINTS_PER_DECADE) + 1)


def _relative(
    model: MassStoppingPowerModel,
    table: MassStoppingPowerModel,
    z: int,
    a: int,
    e: npt.NDArray[np.float64],
) -> npt.NDArray[np.float64]:
    out: npt.NDArray[np.float64] = (
        model.mass_stopping_power(z, a, e) / table.mass_stopping_power(z, a, e) - 1.0
    )
    return out


# Twin of test_icru90_procedure.py::test_reproduces_the_icru90_carbon_table. Measured max
# per band [<0.01, 0.01-0.1, 0.1-1, 1-10, 10-100, >100 MeV/u]: 5.6e-3, 3.8e-4, 1.1e-3,
# 9.7e-4, 3.8e-4, 3.0e-4 (the first band is the alpha table's lower edge, as in the raw test).
PROCEDURE_CARBON_BANDS = (
    ((0.0, 0.01), 0.011),
    ((0.01, 0.1), 7.5e-4),
    ((0.1, 1.0), 2e-3),
    ((1.0, 10.0), 2e-3),
    ((10.0, 100.0), 7.5e-4),
    ((100.0, np.inf), 6e-4),
)


def test_procedure_reproduces_the_shipped_icru90_carbon_table() -> None:
    proc, table = ICRU90Procedure(), ICRU(edition="2016")
    lo = max(proc.energy_range(6, 12)[0], table.energy_range(6, 12)[0])
    e = _grid(lo, table.energy_range(6, 12)[1])
    rel = _relative(proc, table, 6, 12, e)
    for (b_lo, b_hi), bound in PROCEDURE_CARBON_BANDS:
        band = (e >= b_lo) & (e < b_hi)
        assert band.any()
        assert np.abs(rel[band]).max() < bound, (b_lo, b_hi)


# Twin of test_icru90_procedure.py::test_p_alpha_reproduce_icru49_at_75_ev_and_icru90_at_78_ev.
# Measured max below T1 / gap / above T2 (to 1000 MeV/u, ASTAR to its 250): PSTAR 1.3e-3 /
# 1.3e-3 / 1.3e-3; ICRU 90 p 2.5e-3 / 1.3e-3 / 1.5e-3; ASTAR 7.6e-4 / 2.6e-3 / 1.5e-3;
# ICRU 90 alpha 5.8e-4 / 3.3e-3 / 1.2e-3.
P_ALPHA_CASES = (
    ("pstar", 75.0, 1, 1, (2.5e-3, 2.5e-3, 2.5e-3)),
    ("icru90", 78.0, 1, 1, (4.5e-3, 2.5e-3, 3e-3)),
    ("astar", 75.0, 2, 4, (1.5e-3, 5e-3, 3e-3)),
    ("icru90", 78.0, 2, 4, (1.2e-3, 6e-3, 2.5e-3)),
)


@pytest.mark.parametrize(("table_name", "i_ev", "z", "a", "bounds"), P_ALPHA_CASES)
def test_procedure_p_alpha_reproduce_the_shipped_tables(
    table_name: str, i_ev: float, z: int, a: int, bounds: tuple[float, float, float]
) -> None:
    table = ICRU(edition="2016") if table_name == "icru90" else XSTAR()
    proc = ICRU90Procedure(i_value_eV=i_ev)
    t1, t2 = proc.metadata(z, a).options["thresholds_MeV_u"]  # type: ignore[misc]
    lo = max(proc.energy_range(z, a)[0], table.energy_range(z, a)[0])
    e = _grid(lo, min(1000.0, table.energy_range(z, a)[1]))
    rel = _relative(proc, table, z, a, e)
    for mask, bound in zip((e < t1, (e >= t1) & (e < t2), e >= t2), bounds, strict=True):
        assert mask.any()
        assert np.abs(rel[mask]).max() < bound


@pytest.mark.parametrize(
    "corrections",
    [{"lindhard_sorensen": True}, {"bloch": True, "mott": True}, {"bloch": True, "ahlen": True}],
)
def test_bethe_high_energy_carbon_against_the_shipped_icru90_table(
    corrections: dict[str, bool],
) -> None:
    """Twin of test_bethe.py::test_lindhard_sorensen_closes_the_carbon_gap_to_icru90.

    Measured 30-833 MeV/u: LS 6.0e-4, Bloch + Mott 6.1e-4, Bloch + Ahlen 5.9e-4 (raw: same
    to 1e-5); the raw test's bound.
    """
    bethe = Bethe(effective_charge=None, i_value_eV=78.0, **corrections)  # type: ignore[arg-type]
    table = ICRU(edition="2016")
    e = _grid(30.0, min(table.energy_range(6, 12)[1], bethe.energy_range(6, 12)[1]))
    assert np.abs(_relative(bethe, table, 6, 12, e)).max() < 1.2e-3


def test_hubert_carbon_against_the_shipped_icru90_table() -> None:
    """Twin of test_hubert.py::test_icru90_carbon. Measured 2.5-300 MeV/u: 1.24e-2 (raw
    1.2%); the raw test's bound."""
    e = _grid(2.5, 300.0)
    assert np.abs(_relative(HubertAlphaScaling(), ICRU(edition="2016"), 6, 12, e)).max() < 0.018
