"""build_zstar_table over the MKM backend seam."""

from __future__ import annotations

from collections.abc import Sequence

from pyradbio.core.radiation import Ion
from pyradbio.models.mkm.base import BackendZStarRow, BackendZStarTable
from pyradbio.tables import io
from pyradbio.tables.zstar import ZStarTable, build_zstar_table
from tests.conftest import requires_pymkm


class _FakeBackend:
    """MKMBackend returning a known z*_1D grid (no pyMKM)."""

    def z_star_1D_Gy(self, ion: Ion) -> float:  # pragma: no cover - unused here
        return 3.43

    def z_star_1D_table_Gy(
        self, ions: Sequence[int], energies_MeV_u: Sequence[float]
    ) -> BackendZStarTable:
        return BackendZStarTable(
            [
                BackendZStarRow(ion_Z=1, energy_MeV_u=10.0, let_keV_um=7.0, z_star_1D_Gy=1.2),
                BackendZStarRow(ion_Z=6, energy_MeV_u=100.0, let_keV_um=20.0, z_star_1D_Gy=2.8),
            ]
        )


def test_build_zstar_table_maps_backend_rows_and_header() -> None:
    table = build_zstar_table(
        [1, 6],
        [10.0, 100.0],
        domain_radius_um=0.32,
        nucleus_radius_um=3.9,
        alpha0_Gy=0.172,
        beta0_Gy2=0.0615,
        cell_type="HSG",
        backend=_FakeBackend(),
    )
    assert isinstance(table, ZStarTable)
    assert table.cell_type == "HSG"
    assert table.alpha0_Gy == 0.172
    assert table.domain_radius_um == 0.32
    assert [(r.Z, r.energy_MeV_u, r.z_star_1D_Gy) for r in table.rows] == [
        (1, 10.0, 1.2),
        (6, 100.0, 2.8),
    ]


def test_build_zstar_table_round_trips_through_pymkm_txt(tmp_path) -> None:  # type: ignore[no-untyped-def]
    table = build_zstar_table(
        [1, 6],
        [10.0, 100.0],
        domain_radius_um=0.32,
        nucleus_radius_um=3.9,
        alpha0_Gy=0.172,
        beta0_Gy2=0.0615,
        cell_type="HSG",
        backend=_FakeBackend(),
    )
    path = tmp_path / "zstar.txt"
    io.write_pymkm_txt(table, path)
    assert io.read_pymkm_txt(path) == table


def test_build_zstar_table_default_backend_zstar_decreases_with_energy() -> None:
    """Default (native) carbon z*_1D falls with energy (Inaniwa 2010 Fig. 5 shape)."""
    table = build_zstar_table(
        [6],
        [11.0, 100.0, 400.0],
        domain_radius_um=0.32,
        nucleus_radius_um=3.9,
        alpha0_Gy=0.172,
        beta0_Gy2=0.0615,
        cell_type="HSG",
    )
    z_by_energy = {r.energy_MeV_u: r.z_star_1D_Gy for r in table.rows}
    assert z_by_energy[11.0] > z_by_energy[100.0] > z_by_energy[400.0]


@requires_pymkm
def test_build_zstar_table_pymkm_backend_zstar_decreases_with_energy() -> None:
    """The same table through an injected pyMKM backend (cross-check)."""
    from pyradbio.models.mkm.pymkm_backend import PyMKMBackend

    table = build_zstar_table(
        [6],
        [11.0, 100.0, 400.0],
        domain_radius_um=0.32,
        nucleus_radius_um=3.9,
        alpha0_Gy=0.172,
        beta0_Gy2=0.0615,
        cell_type="HSG",
        backend=PyMKMBackend(domain_radius_um=0.32, nucleus_radius_um=3.9, beta0=0.0615),
    )
    z_by_energy = {r.energy_MeV_u: r.z_star_1D_Gy for r in table.rows}
    assert z_by_energy[11.0] > z_by_energy[100.0] > z_by_energy[400.0]
