"""Parsers and safety checks of scripts/stopping/fetch_sources.py, on synthetic snippets.

No real source value appears here: the snippets mimic each source's *layout* with
made-up numbers. The real files are checked by tests/track/stopping/test_raw_sources.py,
which reads them from the git-ignored data/stopping/ folder at run time.
"""

from __future__ import annotations

import importlib.util
import sys
import zipfile
from decimal import Decimal
from pathlib import Path
from types import ModuleType

import pytest

_SCRIPT = Path(__file__).resolve().parents[2] / "scripts" / "stopping" / "fetch_sources.py"


def _load() -> ModuleType:
    spec = importlib.util.spec_from_file_location("fetch_stopping_sources", _SCRIPT)
    assert spec is not None and spec.loader is not None
    module = importlib.util.module_from_spec(spec)
    sys.modules[spec.name] = module
    spec.loader.exec_module(module)
    return module


fs = _load()


def _star_html(title: str, material: str, rows: list[str]) -> str:
    body = "<br>".join(rows)
    return (
        f"<html><body><pre> {title} STOPPING POWER AND RANGE<br> <br><br> {material}   <br>"
        f" <br>   T = kinetic energy, MeV<br>  T STOP(e) STOP(n) STOP(t)<br> <br>{body}<br>"
        "</pre></body></html>"
    )


_GOOD_ROWS = [
    "  1.000E-03  1.000E+02  2.000E+01  1.200E+02  1.000E-06  5.000E-07  0.5000",
    "  2.000E-03  1.500E+02  1.000E+01  1.600E+02  2.000E-06  1.000E-06  0.6000",
]


def test_star_parser_keeps_the_printed_strings() -> None:
    table = fs.parse_star_html(_star_html("PROTON", "WATER, LIQUID", _GOOD_ROWS), "proton")
    assert table.rows[0][:4] == ("1.000E-03", "1.000E+02", "2.000E+01", "1.200E+02")
    assert table.column("T_MeV") == ["1.000E-03", "2.000E-03"]


def test_star_parser_knows_the_alpha_title() -> None:
    html = _star_html("ALPHA PARTICLE", "WATER, LIQUID", _GOOD_ROWS)
    assert len(fs.parse_star_html(html, "alpha").rows) == 2


@pytest.mark.parametrize(
    ("title", "material", "particle", "match"),
    [
        ("ALPHA PARTICLE", "WATER, LIQUID", "proton", "not a proton"),
        ("PROTON", "WATER VAPOR", "proton", "liquid water"),
    ],
)
def test_star_parser_rejects_the_wrong_table(
    title: str, material: str, particle: str, match: str
) -> None:
    with pytest.raises(fs.ExtractionError, match=match):
        fs.parse_star_html(_star_html(title, material, _GOOD_ROWS), particle)


def test_star_parser_rejects_a_broken_sum_rule() -> None:
    rows = [_GOOD_ROWS[0].replace("1.200E+02", "1.300E+02")]
    with pytest.raises(fs.ExtractionError, match="total"):
        fs.parse_star_html(_star_html("PROTON", "WATER, LIQUID", rows), "proton")


def test_star_parser_rejects_non_increasing_energies() -> None:
    with pytest.raises(fs.ExtractionError, match="increasing"):
        fs.parse_star_html(_star_html("PROTON", "WATER, LIQUID", _GOOD_ROWS[::-1]), "proton")


def test_star_parser_rejects_an_empty_table() -> None:
    with pytest.raises(fs.ExtractionError, match="no data"):
        fs.parse_star_html(_star_html("PROTON", "WATER, LIQUID", []), "proton")


def test_half_unit_in_last_place() -> None:
    assert fs.half_unit_in_last_place("1.337E+02") == Decimal("0.05")
    assert fs.half_unit_in_last_place("0.4555") == Decimal("0.00005")
    assert fs.half_unit_in_last_place("10000.0000") == Decimal("0.00005")


def test_sum_rule_allows_exactly_the_printed_rounding() -> None:
    columns = ("T", "e", "n", "t")
    ok = fs.Extracted(columns, (("1", "1.000", "0.0004", "1.001"),))
    assert fs.check_sum_rule(ok, "e", "n", "t") == []
    bad = fs.Extracted(columns, (("1", "1.000", "0.0004", "1.002"),))
    assert len(fs.check_sum_rule(bad, "e", "n", "t")) == 1


_ISOTOPES = """<pre>
Atomic Number = 1
Atomic Symbol = H
Mass Number = 1
Relative Atomic Mass = 1.0001(2)
Isotopic Composition = 0.9
Standard Atomic Weight = [1.0,1.1]
Notes = m

Atomic Number = 9
Atomic Symbol = X
Mass Number = 20
Relative Atomic Mass = 19.5#
Isotopic Composition =
Standard Atomic Weight =
Notes =
</pre>"""


def test_isotope_parser_splits_mass_and_uncertainty() -> None:
    table = fs.parse_nist_isotopes(_ISOTOPES)
    assert table.rows == (
        ("1", "H", "1", "1.0001", "2", "0.9"),
        ("9", "X", "20", "19.5", "", ""),
    )


def test_isotope_parser_rejects_garbage() -> None:
    with pytest.raises(fs.ExtractionError):
        fs.parse_nist_isotopes("<html>nothing here</html>")
    with pytest.raises(fs.ExtractionError, match="mass"):
        fs.parse_nist_isotopes(
            "Atomic Number = 1\nAtomic Symbol = H\nMass Number = 1\nRelative Atomic Mass = abc\n"
        )


def _zip(path: Path, names: list[str]) -> Path:
    with zipfile.ZipFile(path, "w") as zf:
        for name in names:
            zf.writestr(name, b"x")
    return path


def test_mstar_extraction_skips_the_executable(tmp_path: Path) -> None:
    archive = _zip(
        tmp_path / "m.zip",
        [
            "MSTAR312/Sources/msdbs1.d",
            "MSTAR312/Sources/mstar1.f",
            "MSTAR312/DOS exe/MSTAR.EXE",
            "MSTAR312/Doc/MSTARInstr.htm",
        ],
    )
    written = fs.extract_mstar(archive, tmp_path / "out")
    names = sorted(p.relative_to(tmp_path / "out").as_posix() for p in written)
    assert names == [
        "MSTAR312/Doc/MSTARInstr.htm",
        "MSTAR312/Sources/msdbs1.d",
        "MSTAR312/Sources/mstar1.f",
    ]


def test_mstar_extraction_refuses_path_traversal(tmp_path: Path) -> None:
    archive = _zip(tmp_path / "m.zip", ["MSTAR312/Sources/../../evil.f"])
    with pytest.raises(fs.ExtractionError, match="unsafe"):
        fs.extract_mstar(archive, tmp_path / "out")


def test_mstar_extraction_needs_the_database(tmp_path: Path) -> None:
    archive = _zip(tmp_path / "m.zip", ["MSTAR312/Sources/mstar1.f"])
    with pytest.raises(fs.ExtractionError, match="msdbs1"):
        fs.extract_mstar(archive, tmp_path / "out")


def test_verify_compares_provenance_checksums(tmp_path: Path) -> None:
    folder = tmp_path / "src"
    folder.mkdir()
    data = folder / "raw.txt"
    data.write_bytes(b"hello")
    good = fs.sha256(data)
    (folder / "PROVENANCE.md").write_text(
        f"| file | sha256 | bytes |\n|---|---|---|\n| `raw.txt` | `{good}` | 5 |\n"
        f"| `gone.txt` | `{'0' * 64}` | 1 |\n",
        encoding="utf-8",
    )
    assert fs.provenance_checksums(folder / "PROVENANCE.md") == {
        "raw.txt": good,
        "gone.txt": "0" * 64,
    }
    assert fs.verify(tmp_path) == [f"missing: {folder / 'gone.txt'}"]
    data.write_bytes(b"changed")
    assert f"checksum differs: {data}" in fs.verify(tmp_path)


# --- ICRU 73 (libdedx raw layout -> our CSV format) ------------------------------------

_GRID = "3\n0.025\n0.1\n1\n"
_BLOCKS = (
    "#1:3:3\n1.0E+03\n2.0E+03\n1.5E+03\n"
    "#276:03:3\n1.1E+03\n2.1E+03\n1.6E+03\n\n"
    "#276:4:3\n4E+3\n5E+3\n3E+3\n"
)


def test_libdedx_grid_and_blocks_parse_as_printed() -> None:
    assert fs.parse_libdedx_grid(_GRID) == ["0.025", "0.1", "1"]
    blocks = fs.parse_libdedx_blocks(_BLOCKS)
    assert set(blocks) == {(1, 3), (276, 3), (276, 4)}  # "03" and "3" are the same ion
    assert blocks[(276, 3)] == ["1.1E+03", "2.1E+03", "1.6E+03"]


@pytest.mark.parametrize(
    ("text", "match"),
    [
        ("#276:3:3\n1\n2\n", "3 values"),
        ("1.0\n#276:3:1\n1\n", "before the first"),
        ("#276:3\n1\n", "header"),
        ("#276:3:1\n1\n#276:03:1\n2\n", "twice"),
    ],
)
def test_libdedx_blocks_reject_malformed_files(text: str, match: str) -> None:
    with pytest.raises(fs.ExtractionError, match=match):
        fs.parse_libdedx_blocks(text)


def test_libdedx_grid_rejects_a_wrong_count() -> None:
    with pytest.raises(fs.ExtractionError, match="count"):
        fs.parse_libdedx_grid("4\n0.025\n0.1\n1\n")


def test_icru73_water_table_takes_liquid_water_only() -> None:
    grid = fs.parse_libdedx_grid(_GRID)
    table = fs.icru73_water_table(fs.parse_libdedx_blocks(_BLOCKS), grid, ions=(3, 4))
    assert table.columns == ("Z", "energy_MeV_u", "S_el_MeV_cm2_g")
    assert table.rows[:2] == (("3", "0.025", "1.1E+03"), ("3", "0.1", "2.1E+03"))
    assert len(table.rows) == 6


def test_icru73_water_table_requires_every_ion() -> None:
    grid = fs.parse_libdedx_grid(_GRID)
    with pytest.raises(fs.ExtractionError, match="Z=5"):
        fs.icru73_water_table(fs.parse_libdedx_blocks(_BLOCKS), grid, ions=(3, 4, 5))


def test_icru73_water_table_rejects_non_positive_values() -> None:
    blocks = fs.parse_libdedx_blocks("#276:3:3\n1E3\n0.000E+03\n1E3\n")
    with pytest.raises(fs.ExtractionError, match="positive"):
        fs.icru73_water_table(blocks, fs.parse_libdedx_grid(_GRID), ions=(3,))


def test_icru73_csv_round_trips_with_its_metadata(tmp_path: Path) -> None:
    table = fs.icru73_water_table(
        fs.parse_libdedx_blocks(_BLOCKS), fs.parse_libdedx_grid(_GRID), ions=(3, 4)
    )
    path = tmp_path / "icru73_2005_water_liquid.csv"
    fs.write_icru73_csv(path, table, edition="2005", source=["synthetic"])
    meta, loaded = fs.read_icru73_csv(path)
    assert loaded == table
    assert meta["edition"] == "2005"
    assert meta["material"] == "liquid water"
    assert meta["quantity"] == "electronic"


def test_icru73_csv_reader_requires_the_metadata(tmp_path: Path) -> None:
    path = tmp_path / "t.csv"
    path.write_text("# edition: 2005\nZ,energy_MeV_u,S_el_MeV_cm2_g\n3,0.025,1E3\n")
    with pytest.raises(fs.ExtractionError, match="material"):
        fs.read_icru73_csv(path)


# --- density-effect sources: ESTAR, PDG ------------------------------------------------

_ESTAR = (
    "<pre>                             Electrons in  WATER, LIQUID      <br>"
    "                              Density, g/cm3     I, eV<br>"
    "                                1.0000E+00        75.0<br>"
    "      KINETIC         STOPPING POWER             CSDA   RADIATION   DENSITY<br>"
    "       MeV    MeV cm2/g  MeV cm2/g  MeV cm2/g    g/cm2<br>"
    "    1.000E-02  2.000E+01  4.000E-03  2.000E+01  2.000E-04  9.000E-05  0.000E+00<br>"
    "    1.000E+03  2.000E+00  2.000E+01  2.200E+01  1.000E+02  7.000E-01  1.000E+01<br>"
    "</pre>"
)


def test_estar_parser_keeps_delta_and_the_i_value() -> None:
    table, meta = fs.parse_estar_html(_ESTAR)
    assert meta == {"i_value_eV": "75.0", "density_g_cm3": "1.0000E+00"}
    assert table.column("delta") == ["0.000E+00", "1.000E+01"]
    assert table.columns[0] == "T_MeV"


def test_estar_parser_checks_collision_plus_radiative() -> None:
    with pytest.raises(fs.ExtractionError, match="total"):
        fs.parse_estar_html(_ESTAR.replace("2.200E+01", "2.300E+01"))
    with pytest.raises(fs.ExtractionError, match="liquid water"):
        fs.parse_estar_html(_ESTAR.replace("WATER, LIQUID", "WATER VAPOR"))


_PDG_MU = (
    " Incident particle is a   Muon with M =   105.65839 MeV\n"
    " Index =  276:  water (liquid) (H\sub{2}O)\n"
    " Sternheimer coef:  a     k=m_s   x_0    x_1    I[eV]   Cbar  delta0 \n"
    "                 0.1000  3.0000  0.2000  3.0000   80.0  3.6000 0.00\n"
)
_PDG_HTML = "<tr><td>Plasma energy &#8463; &#969; p</td><td>\n 21.00 \n</td><td>eV</td></tr>"


def test_pdg_parser_reads_the_sternheimer_coefficients() -> None:
    coefficients = fs.parse_pdg_sternheimer(_PDG_MU)
    assert coefficients == {
        "a": "0.1000",
        "k": "3.0000",
        "x0": "0.2000",
        "x1": "3.0000",
        "I_eV": "80.0",
        "Cbar": "3.6000",
        "delta0": "0.00",
    }
    assert fs.parse_pdg_plasma_energy(_PDG_HTML) == "21.00"


def test_pdg_parsers_reject_other_files() -> None:
    with pytest.raises(fs.ExtractionError):
        fs.parse_pdg_sternheimer(_PDG_MU.replace("276", "277").replace("water (liquid)", "steam"))
    with pytest.raises(fs.ExtractionError):
        fs.parse_pdg_plasma_energy("<html>no plasma here</html>")
