"""Round-trip and format fidelity for pyMKM classic .txt z* tables.

The exact format is taken from the installed pyMKM ``MKTable.write_txt`` (classic
model): two-space ``CellType`` line, ``.4f`` header parameters, ``.5e`` data, and
a blank line after each fragment block. See docs/references/pyMKM_API.md.
"""

from __future__ import annotations

from pyradbio.tables import io
from pyradbio.tables.zstar import ZStarRow, ZStarTable


def _sample() -> ZStarTable:
    return ZStarTable(
        rows=(
            ZStarRow(Z=1, energy_MeV_u=1.0, z_star_1D_Gy=2.3),
            ZStarRow(Z=1, energy_MeV_u=10.0, z_star_1D_Gy=1.5),
            ZStarRow(Z=6, energy_MeV_u=200.0, z_star_1D_Gy=3.43),
        ),
        cell_type="HSG",
        alpha0_Gy=0.172,
        beta_Gy2=0.0615,
        domain_radius_um=0.32,
        nucleus_radius_um=3.9,
    )


def test_written_file_matches_pymkm_classic_format(tmp_path) -> None:  # type: ignore[no-untyped-def]
    path = tmp_path / "mktable.txt"
    io.write_pymkm_txt(_sample(), path)
    expected = (
        "CellType  HSG\n"
        "\n"
        "Parameter Alpha_0 0.1720\n"
        "Parameter Beta 0.0615\n"
        "Parameter DomainRadius 0.3200\n"
        "Parameter NucleusRadius 3.9000\n"
        "\n"
        "Fragment H\n"
        "1.00000e+00 2.30000e+00\n"
        "1.00000e+01 1.50000e+00\n"
        "\n"
        "Fragment C\n"
        "2.00000e+02 3.43000e+00\n"
        "\n"
    )
    assert path.read_text(encoding="utf-8") == expected


def test_pymkm_txt_round_trip(tmp_path) -> None:  # type: ignore[no-untyped-def]
    table = _sample()
    path = tmp_path / "mktable.txt"
    io.write_pymkm_txt(table, path)
    assert io.read_pymkm_txt(path) == table


def test_read_groups_fragments_by_element(tmp_path) -> None:  # type: ignore[no-untyped-def]
    path = tmp_path / "mktable.txt"
    io.write_pymkm_txt(_sample(), path)
    read = io.read_pymkm_txt(path)
    assert {r.Z for r in read.rows} == {1, 6}
    assert read.cell_type == "HSG"
    assert read.domain_radius_um == 0.32
