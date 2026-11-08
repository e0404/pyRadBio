"""Energy grids, log-log regridding and the shipped-table CSV format (track/stopping/grid.py).

All data here is synthetic. The same checks run against the real PSTAR/ASTAR/MSTAR
values in the per-source test modules, which read the raw files at run time.
"""

from __future__ import annotations

from pathlib import Path

import numpy as np
import numpy.typing as npt
import pytest

from pyradbio.track.stopping.base import IonSourceInfo, OutOfRangeError
from pyradbio.track.stopping.grid import (
    IonTable,
    LogEnergyGrid,
    RegriddedTable,
    leave_one_out_relative_error,
    parse_table_csv,
    read_table_csv,
    write_table_csv,
)


def _stopping_like(energy: npt.ArrayLike) -> npt.NDArray[np.float64]:
    """Smooth, Bragg-peak-shaped synthetic curve: rises as E^0.45, falls as E^-0.8.

    Its maximum sits at E0 * (p/q)^(1/(p+q)) = 0.0504 MeV/u, i.e. where a proton's
    electronic stopping peaks; the shape, not the numbers, is what matters here.
    """
    e = np.asarray(energy, dtype=np.float64)
    p, q, e0 = 0.45, 0.8, 0.08
    return 800.0 * (e / e0) ** p / (1.0 + (e / e0) ** (p + q))


_PEAK_MEV_U = 0.08 * (0.45 / 0.8) ** (1.0 / 1.25)

# An irregular source grid, 14 points per decade with mantissas like a printed table.
_MANTISSAS = (1.0, 1.25, 1.5, 1.75, 2.0, 2.5, 3.0, 3.5, 4.0, 5.0, 6.0, 7.0, 8.0, 9.0)
_SOURCE_E = np.array([m * 10.0**k for k in range(-3, 3) for m in _MANTISSAS] + [1000.0])
_SOURCE_S = _stopping_like(_SOURCE_E)


# --- LogEnergyGrid ----------------------------------------------------------------------


def test_log_grid_has_exact_endpoints_and_uniform_log_spacing() -> None:
    grid = LogEnergyGrid(0.001, 1000.0, 61)
    e = grid.energies
    assert e[0] == 0.001
    assert e[-1] == 1000.0
    assert e.size == 61
    np.testing.assert_allclose(np.diff(np.log10(e)), 0.1, rtol=1e-12)


def test_log_grid_from_points_per_decade_covers_the_range() -> None:
    grid = LogEnergyGrid.from_points_per_decade(0.025, 1000.0, 20)
    decades = np.log10(1000.0 / 0.025)
    assert grid.n_points == int(np.ceil(decades * 20)) + 1
    assert np.max(np.diff(np.log10(grid.energies))) <= 1.0 / 20 + 1e-12


def test_log_grid_round_trips_through_its_description() -> None:
    grid = LogEnergyGrid(0.1, 100.0, 31)
    assert LogEnergyGrid.from_dict(grid.to_dict()) == grid


@pytest.mark.parametrize(("lo", "hi", "n"), [(0.0, 1.0, 10), (1.0, 1.0, 10), (1.0, 2.0, 1)])
def test_log_grid_rejects_degenerate_input(lo: float, hi: float, n: int) -> None:
    with pytest.raises(ValueError):
        LogEnergyGrid(lo, hi, n)


# --- RegriddedTable ---------------------------------------------------------------------


@pytest.mark.parametrize("method", ["cubic", "pchip"])
def test_table_reproduces_its_own_nodes(method: str) -> None:
    table = RegriddedTable(_SOURCE_E, _SOURCE_S, method=method)  # type: ignore[arg-type]
    np.testing.assert_allclose(table(_SOURCE_E), _SOURCE_S, rtol=1e-12)


def test_table_raises_outside_its_nodes() -> None:
    table = RegriddedTable(_SOURCE_E, _SOURCE_S)
    with pytest.raises(OutOfRangeError) as info:
        table([1.0, 1000.1])
    assert info.value.valid_range_MeV_u == (0.001, 1000.0)
    with pytest.raises(OutOfRangeError):
        table(0.000999)


@pytest.mark.parametrize(
    ("energy", "values"),
    [
        ([1.0, 2.0], [1.0]),
        ([1.0], [1.0]),
        ([2.0, 1.0], [1.0, 1.0]),
        ([1.0, 1.0], [1.0, 1.0]),
        ([1.0, 2.0], [1.0, 0.0]),
        ([0.0, 2.0], [1.0, 1.0]),
    ],
)
def test_table_rejects_malformed_nodes(energy: list[float], values: list[float]) -> None:
    with pytest.raises(ValueError):
        RegriddedTable(energy, values)


def test_table_rejects_an_unknown_method() -> None:
    with pytest.raises(ValueError, match="method"):
        RegriddedTable(_SOURCE_E, _SOURCE_S, method="linear")  # type: ignore[arg-type]


def test_regridding_reproduces_the_source_values_at_the_source_points() -> None:
    """Source -> our grid -> back to the SOURCE points (never our grid's midpoints).

    Both legs interpolate, so this measures the round-trip error a shipped table
    adds on top of the source. Measured on this curve: 2.4e-6 at 10, 6.7e-7 at 20,
    2.5e-8 at 50 and 5.6e-9 at 100 points per decade; pinned at 50 with a 2x margin.
    """
    grid = LogEnergyGrid.from_points_per_decade(0.001, 1000.0, 50)
    regridded = RegriddedTable.from_source(_SOURCE_E, _SOURCE_S, grid)
    np.testing.assert_array_equal(regridded.energy_MeV_u, grid.energies)
    rel = np.abs(regridded(_SOURCE_E) / _SOURCE_S - 1.0)
    assert rel.max() < 5e-8


def test_regridding_needs_the_grid_inside_the_source_range() -> None:
    grid = LogEnergyGrid(0.0005, 10.0, 20)
    with pytest.raises(OutOfRangeError):
        RegriddedTable.from_source(_SOURCE_E, _SOURCE_S, grid)


# --- interpolation error near the stopping maximum --------------------------------------


def _near_peak_error(method: str) -> float:
    table = RegriddedTable(_SOURCE_E, _SOURCE_S, method=method)  # type: ignore[arg-type]
    probe = np.geomspace(_PEAK_MEV_U / 3.0, _PEAK_MEV_U * 3.0, 2001)
    return float(np.max(np.abs(table(probe) / _stopping_like(probe) - 1.0)))


def test_interpolation_error_near_the_maximum_is_quantified() -> None:
    """Measured, not assumed: max relative error within a factor 3 of the peak.

    On this 14-points-per-decade grid the not-a-knot cubic reaches 2.6e-6 and
    PCHIP 5.5e-4: PCHIP's monotonicity limiter flattens the curve at the
    turning point, so it is ~200x worse exactly where stopping power peaks, which
    is why the cubic is the default. The bounds pin those measurements with
    a 2x margin. The synthetic curve is smoother than real data; the per-source
    tests measure the real tables.
    """
    cubic, pchip = _near_peak_error("cubic"), _near_peak_error("pchip")
    assert cubic < 5.3e-6
    assert pchip < 1.1e-3
    assert cubic < pchip


def test_leave_one_out_error_bounds_the_true_error_near_the_maximum() -> None:
    """With no truth between source points, leave-one-out is the estimate we can make.

    Dropping a node doubles the local spacing, so for a smooth curve it over-
    estimates the true interpolation error: it is a conservative bound, which is
    what the per-source tests report for the real tables.
    """
    loo = leave_one_out_relative_error(_SOURCE_E, _SOURCE_S, method="cubic")
    assert loo.shape == (_SOURCE_E.size - 2,)
    interior = _SOURCE_E[1:-1]
    near = (interior > _PEAK_MEV_U / 3.0) & (interior < _PEAK_MEV_U * 3.0)
    assert np.max(np.abs(loo[near])) > _near_peak_error("cubic")


# --- CSV format with a metadata header --------------------------------------------------


def _info(z: int, a: int, source: str) -> IonSourceInfo:
    return IonSourceInfo(
        Z=z,
        A=a,
        source=source,
        reference="doi:10.0000/synthetic",
        i_value_eV=75.0,
        energy_range_MeV_u=(0.001, 1000.0),
        edition="test",
        options={"quantity": "electronic", "mstar_mode": "b"},
    )


def _tables() -> list[IonTable]:
    e = _SOURCE_E
    return [
        IonTable(
            _info(1, 1, "synthetic-p"),
            e,
            {"electronic_MeV_cm2_g": _SOURCE_S, "nuclear_MeV_cm2_g": _SOURCE_S * 1e-3},
        ),
        IonTable(
            _info(6, 12, "synthetic-C"),
            e,
            {"electronic_MeV_cm2_g": 36 * _SOURCE_S, "nuclear_MeV_cm2_g": _SOURCE_S * 0.1},
        ),
    ]


def test_csv_round_trips_values_and_metadata_exactly(tmp_path: Path) -> None:
    path = tmp_path / "table.csv"
    metadata = {"generator": "test", "grid": LogEnergyGrid(0.001, 1000.0, 85).to_dict()}
    notes = ["Synthetic table.", "Second line of provenance prose."]
    write_table_csv(path, _tables(), metadata=metadata, notes=notes)

    loaded = read_table_csv(path)
    assert loaded.metadata == metadata
    assert loaded.notes == tuple(notes)
    assert [t.info for t in loaded.tables] == [t.info for t in _tables()]
    for got, want in zip(loaded.tables, _tables(), strict=True):
        np.testing.assert_array_equal(got.energy_MeV_u, want.energy_MeV_u)
        assert got.columns.keys() == want.columns.keys()
        for name in want.columns:
            np.testing.assert_array_equal(got.columns[name], want.columns[name])


def test_csv_header_is_human_readable(tmp_path: Path) -> None:
    path = tmp_path / "table.csv"
    write_table_csv(path, _tables(), metadata={"generator": "test"}, notes=["Prose."])
    lines = path.read_text(encoding="utf-8").splitlines()
    assert lines[0] == "# Prose."
    header = next(line for line in lines if not line.startswith("#"))
    assert header == "Z,A,energy_MeV_u,electronic_MeV_cm2_g,nuclear_MeV_cm2_g"
    assert any(line.startswith("#@ ion.6.12 = ") for line in lines)


def test_csv_lookup_by_ion(tmp_path: Path) -> None:
    path = tmp_path / "table.csv"
    write_table_csv(path, _tables(), metadata={})
    loaded = read_table_csv(path)
    assert loaded.table(6, 12).info.source == "synthetic-C"
    with pytest.raises(KeyError):
        loaded.table(2, 4)


def test_csv_rejects_tables_with_different_columns(tmp_path: Path) -> None:
    tables = _tables()
    odd = IonTable(tables[1].info, _SOURCE_E, {"electronic_MeV_cm2_g": _SOURCE_S})
    with pytest.raises(ValueError, match="columns"):
        write_table_csv(tmp_path / "t.csv", [tables[0], odd], metadata={})


def test_ion_table_rejects_mismatched_lengths() -> None:
    with pytest.raises(ValueError, match="length"):
        IonTable(_info(1, 1, "x"), _SOURCE_E, {"electronic_MeV_cm2_g": _SOURCE_S[:-1]})


def test_csv_body_without_ion_metadata_is_rejected() -> None:
    text = (
        '#@ format = "pyradbio-stopping-table/1"\n'
        "Z,A,energy_MeV_u,electronic_MeV_cm2_g\n"
        "1,1,1.0,10.0\n"
    )
    with pytest.raises(ValueError, match=r"ion\.1\.1"):
        parse_table_csv(text)


def test_csv_with_an_unknown_format_is_rejected() -> None:
    text = '#@ format = "something-else/9"\nZ,A,energy_MeV_u,x\n'
    with pytest.raises(ValueError, match="format"):
        parse_table_csv(text)


def test_points_per_decade_must_be_positive() -> None:
    with pytest.raises(ValueError, match="points_per_decade"):
        LogEnergyGrid.from_points_per_decade(0.1, 10.0, 0)


def test_table_rejects_non_finite_nodes() -> None:
    with pytest.raises(ValueError, match="finite"):
        RegriddedTable([1.0, 2.0], [1.0, float("inf")])


def test_leave_one_out_needs_three_nodes() -> None:
    with pytest.raises(ValueError, match="three"):
        leave_one_out_relative_error([1.0, 2.0], [1.0, 2.0])


def test_ion_table_rejects_a_reserved_column_name() -> None:
    with pytest.raises(ValueError, match="reserved"):
        IonTable(_info(1, 1, "x"), _SOURCE_E, {"Z": _SOURCE_S})


@pytest.mark.parametrize("metadata", [{"format": "x"}, {"ion.1.1": {}}])
def test_csv_rejects_reserved_metadata_keys(tmp_path: Path, metadata: dict[str, object]) -> None:
    with pytest.raises(ValueError, match="reserved"):
        write_table_csv(tmp_path / "t.csv", _tables(), metadata=metadata)


def test_csv_rejects_an_empty_table_list(tmp_path: Path) -> None:
    with pytest.raises(ValueError, match="nothing"):
        write_table_csv(tmp_path / "t.csv", [], metadata={})


@pytest.mark.parametrize(
    "text",
    [
        '#@ format = "pyradbio-stopping-table/1"\n#@ no-separator\nZ,A,energy_MeV_u,x\n',
        '#@ format = "pyradbio-stopping-table/1"\nenergy,Z,A,x\n',
    ],
)
def test_csv_rejects_malformed_headers(text: str) -> None:
    with pytest.raises(ValueError):
        parse_table_csv(text)
