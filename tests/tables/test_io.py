"""Table serialization round-trips."""

from __future__ import annotations

import pytest

from pyradbio.tables import io
from pyradbio.tables.alpha_beta import AlphaBetaRow, AlphaBetaTable


def _sample_table() -> AlphaBetaTable:
    return AlphaBetaTable(
        rows=(
            AlphaBetaRow(Z=1, A=1, energy_MeV_u=10.0, alpha=0.31234567890123, beta=0.0615),
            AlphaBetaRow(Z=6, A=12, energy_MeV_u=200.0, alpha=0.664, beta=0.0615),
            AlphaBetaRow(Z=6, A=12, energy_MeV_u=13.7, alpha=1.2345e-3, beta=0.05),
        ),
        metadata={"model": "mkm", "cell": "HSG", "alpha_x": 0.313, "beta_x": 0.0615},
    )


def test_csv_round_trip_is_identity(tmp_path) -> None:  # type: ignore[no-untyped-def]
    table = _sample_table()
    path = tmp_path / "ab.csv"
    io.write_csv(table, path)
    assert io.read_csv(path) == table


def test_csv_round_trip_without_metadata(tmp_path) -> None:  # type: ignore[no-untyped-def]
    table = AlphaBetaTable(
        rows=(AlphaBetaRow(Z=2, A=4, energy_MeV_u=1.0, alpha=0.5, beta=0.05),),
    )
    path = tmp_path / "ab_nometa.csv"
    io.write_csv(table, path)
    round_tripped = io.read_csv(path)
    assert round_tripped == table
    assert round_tripped.metadata == {}


def test_csv_preserves_float_precision(tmp_path) -> None:  # type: ignore[no-untyped-def]
    table = _sample_table()
    path = tmp_path / "ab.csv"
    io.write_csv(table, path)
    read = io.read_csv(path)
    assert read.rows[0].alpha == table.rows[0].alpha


def test_parquet_round_trip_is_identity(tmp_path) -> None:  # type: ignore[no-untyped-def]
    pytest.importorskip("pyarrow")
    table = _sample_table()
    path = tmp_path / "ab.parquet"
    io.write_parquet(table, path)
    assert io.read_parquet(path) == table
