"""Fetch the raw water stopping-power sources into ``data/stopping/<source>/``.

Raw downloads are never committed (``data/stopping/`` is git-ignored except for each
source's ``PROVENANCE.md``). For every source this script

1. downloads the raw file exactly as served (or, for MSTAR, checks a file the user
   downloaded by hand, see below),
2. extracts the tabulated values into a plain CSV next to it, keeping every number
   as the *printed string* so no digit is reformatted,
3. checks the extraction (row counts, monotone energies, and electronic + nuclear =
   total to the printed precision on every row), and
4. prints the SHA-256 checksums that go into ``PROVENANCE.md``.

``--verify`` recomputes the checksums of the files on disk and compares them with
the tables in the ``PROVENANCE.md`` files, without touching the network. The NIST
CGI pages embed a per-request token, so their raw HTML differs on every fetch; the
PROVENANCE tables therefore pin the *extracted* CSVs (stable while NIST's values are
unchanged) and record the raw HTML checksum only as retrieved.

Sources (see each ``PROVENANCE.md`` for citations and notices):

* ``pstar`` / ``astar`` -- NIST SRD 124 text output for liquid water (ICRU 49).
* ``icru90`` -- NIST's ICRU Report 90 addendum (PDF); tables A.9, A.12, A.15.
* ``mstar`` -- MSTAR 3.12 (Paul & Schinner) from the IAEA. The IAEA site rejects
  non-browser clients, so download ``MSTAR312.zip`` by hand from
  https://www-nds.iaea.org/stopping/mstar into ``data/stopping/mstar/``; this script
  verifies its checksum and extracts the sources (never the bundled DOS executable).
* ``nist_isotopes`` -- NIST "Atomic Weights and Isotopic Compositions" (all isotopes).
* ``icru73`` -- ICRU 73 liquid-water values from libdedx's raw regeneration inputs
  (GPL-3.0 repository, pinned commit), a surrogate for the report tables, rewritten in
  our own format (``data/stopping/icru73/README.md``), which report-transcribed files
  use too.
* ``arb1973`` -- Ashley, Ritchie & Brandt, Phys. Rev. A 8, 2402 (1973), Table I: a hand
  transcription (nothing to download); this script checks its structure.
* ``ahlen1980`` -- Ahlen, Rev. Mod. Phys. 52, 121 (1980), Table XII (cos chi): a hand
  transcription (nothing to download); structure checked.
* ``brandt1982`` -- Brandt & Kitagawa, Phys. Rev. B 25, 5631 (1982), Table II (ion
  ionization q(y_r)): a hand transcription (nothing to download); structure checked.
* ``icru49`` -- ICRU Report 49's empirical low-energy coefficients and cut-offs for
  liquid water: a hand transcription (nothing to download); structure checked.

Downloaded files are treated as untrusted data: they are parsed, never executed.
The PDF is read with ``pypdf`` (BSD licence, ``dev`` extra), which is used only here.

Usage::

    python scripts/stopping/fetch_sources.py [--dest data/stopping] [--only pstar ...]
    python scripts/stopping/fetch_sources.py --verify
"""

from __future__ import annotations

import argparse
import csv
import datetime as dt
import hashlib
import json
import re
import sys
import urllib.parse
import urllib.request
import zipfile
from collections.abc import Callable, Sequence
from dataclasses import dataclass, field
from decimal import Decimal
from itertools import pairwise
from pathlib import Path, PurePosixPath

USER_AGENT = "pyradbio-fetch/0.1 (+https://github.com/e0404/pyRadBio)"
REPO_ROOT = Path(__file__).resolve().parents[2]
DEFAULT_DEST = REPO_ROOT / "data" / "stopping"

STAR_URL = "https://physics.nist.gov/cgi-bin/Star/ap_table-t.pl"
STAR_WATER_LIQUID = "276"  # NIST material number of "Water, Liquid" in PSTAR/ASTAR
ICRU90_URL = "https://www.nist.gov/document/update-estar-pstar-and-astar-databases"
MSTAR_URL = "https://www-nds.iaea.org/stopping-media/MSTAR312.zip"
ISOTOPES_URL = (
    "https://physics.nist.gov/cgi-bin/Compositions/stand_alone.pl?ele=&ascii=ascii2&isotype=all"
)

# The ICRU 90 water tables we use: table number -> particle as printed in the title.
ICRU90_WATER_TABLES = {
    "A.3": "Electrons",  # for its density-effect column delta (I = 78 eV)
    "A.9": "Protons",
    "A.12": "Alpha particles",
    "A.15": "Carbon ions",
}
ESTAR_URL = "https://physics.nist.gov/cgi-bin/Star/e_table-t.pl"
PDG_MUE_URL = "https://pdg.lbl.gov/2024/AtomicNuclearProperties/MUE/muE_water_liquid.txt"
PDG_WATER_URL = "https://pdg.lbl.gov/2024/AtomicNuclearProperties/HTML/water_liquid.html"

# libdedx (GPL-3.0) keeps ICRU 73 report values as raw regeneration inputs; used as a
# surrogate for the report itself, pinned to a commit.
LIBDEDX_COMMIT = "5f345aa031d52ac1eed8a71b1e51781c3089bd53"
LIBDEDX_RAW = f"https://raw.githubusercontent.com/APTG/libdedx/{LIBDEDX_COMMIT}/data/raw/"
# Our edition -> libdedx files. ICRU73.dat is what libdedx serves as "ICRU73_OLD";
# ICRU73_NEW.dat is what it serves for water and air as "ICRU73". Verified
# against the printed reports: the water blocks equal ICRU 73 pp. 232-233 and the 2009
# errata's Appendix B respectively, at all 848 points (data/stopping/icru73/PROVENANCE.md).
ICRU73_FILES = {
    "2005": ("ICRU73.dat", "icru73Eng.dat"),
    "2009": ("ICRU73_NEW.dat", "icru73_newEng.dat"),
}
# ICRU's (and NIST's) identification number of liquid water.
ICRU_WATER_LIQUID = 276
# ICRU 73 covers ions 3Li..18Ar; H and He come from ICRU 49 (PSTAR/ASTAR).
ICRU73_IONS = tuple(range(3, 19))

# Files MSTAR needs from the archive; the DOS executable is deliberately excluded.
MSTAR_KEEP_PREFIXES = ("MSTAR312/Sources/", "MSTAR312/Doc/MSTARInstr.htm")


class ExtractionError(ValueError):
    """A raw file did not have the structure or internal consistency we expect."""


# --- extracted tables ------------------------------------------------------------------


@dataclass(frozen=True)
class Extracted:
    """A table extracted from a raw file, every value kept as its printed string."""

    columns: tuple[str, ...]
    rows: tuple[tuple[str, ...], ...]
    notes: tuple[str, ...] = ()

    def column(self, name: str) -> list[str]:
        j = self.columns.index(name)
        return [row[j] for row in self.rows]


def half_unit_in_last_place(printed: str) -> Decimal:
    """Half a unit in the last printed digit, e.g. ``1.337E+02`` -> ``0.05``."""
    value = Decimal(printed)
    exponent = value.as_tuple().exponent
    assert isinstance(exponent, int)
    return Decimal(5).scaleb(exponent - 1)


def check_sum_rule(table: Extracted, el: str, nuc: str, tot: str) -> list[str]:
    """Rows where electronic + nuclear differs from total by more than printed rounding."""
    bad = []
    for row in table.rows:
        e, n, t = (row[table.columns.index(c)] for c in (el, nuc, tot))
        slack = half_unit_in_last_place(e) + half_unit_in_last_place(n)
        slack += half_unit_in_last_place(t)
        if abs(Decimal(e) + Decimal(n) - Decimal(t)) > slack:
            bad.append(f"{row[0]}: {e} + {n} != {t}")
    return bad


def check_increasing(values: Sequence[str], what: str) -> None:
    numbers = [Decimal(v) for v in values]
    if any(b <= a for a, b in pairwise(numbers)):
        raise ExtractionError(f"{what}: energies are not strictly increasing")


# --- NIST PSTAR / ASTAR text output ----------------------------------------------------

_STAR_COLUMNS = (
    "T_MeV",
    "S_el_MeV_cm2_g",
    "S_nuc_MeV_cm2_g",
    "S_tot_MeV_cm2_g",
    "csda_range_g_cm2",
    "projected_range_g_cm2",
    "detour_factor",
)
_FLOAT = r"[-+]?\d+(?:\.\d*)?(?:[Ee][-+]?\d+)?"


def parse_star_html(text: str, particle: str) -> Extracted:
    """Parse the ``ap_table-t.pl`` text output (one ``<br>``-separated row per energy)."""
    lines = [re.sub(r"<[^>]*>", "", line).strip() for line in re.split(r"<br>|\n", text)]
    title = {"proton": "PROTON", "alpha": "ALPHA PARTICLE"}[particle] + " STOPPING POWER"
    if not any(line.startswith(title) for line in lines):
        raise ExtractionError(f"not a {particle} stopping-power table")
    if "WATER, LIQUID" not in lines:
        raise ExtractionError("table is not for liquid water")
    rows = tuple(
        tuple(fields)
        for fields in (line.split() for line in lines)
        if len(fields) == len(_STAR_COLUMNS) and all(re.fullmatch(_FLOAT, f) for f in fields)
    )
    if not rows:
        raise ExtractionError("no data rows found")
    table = Extracted(_STAR_COLUMNS, rows)
    check_increasing(table.column("T_MeV"), particle)
    bad = check_sum_rule(table, "S_el_MeV_cm2_g", "S_nuc_MeV_cm2_g", "S_tot_MeV_cm2_g")
    if bad:
        raise ExtractionError(f"{particle}: electronic + nuclear != total in {bad}")
    return table


# --- ICRU 90 addendum PDF --------------------------------------------------------------

_ICRU90_TITLE = re.compile(
    r"Table (A\.\d+)\. (.+?) in liquid water, I = (\d+(?:\.\d+)?) eV, .*= (\d\.\d+) g cm"
)


def _pdf_lines(page: object) -> list[list[str]]:
    """Words of one PDF page grouped into lines by their baseline, left to right."""
    words: dict[int, list[tuple[float, str]]] = {}

    def visit(text: str, cm: Sequence[float], tm: Sequence[float], *_: object) -> None:
        if text.strip():
            x = tm[4] * cm[0] + cm[4]
            y = tm[5] * cm[3] + cm[5]
            words.setdefault(round(y), []).append((x, text.strip()))

    page.extract_text(visitor_text=visit)  # type: ignore[attr-defined]
    return [" ".join(t for _, t in sorted(words[y])).split() for y in sorted(words, reverse=True)]


def parse_icru90_pdf(path: Path) -> dict[str, tuple[Extracted, dict[str, str]]]:
    """Extract the liquid-water tables A.9 (p), A.12 (alpha), A.15 (C) by word position.

    Plain text extraction (e.g. ``pdftotext -layout``) misaligns the energy column of
    these tables by a row; grouping words by baseline does not.
    """
    from pypdf import PdfReader  # dev-only dependency, see module docstring

    found: dict[str, tuple[Extracted, dict[str, str]]] = {}
    for page in PdfReader(path).pages:
        # The title's words do not come out in reading order by position, so find it
        # in the plain page text; positions are used only for the numeric rows.
        title = _ICRU90_TITLE.search(" ".join(page.extract_text().split()))
        lines = _pdf_lines(page)
        if title is None or title.group(1) not in ICRU90_WATER_TABLES:
            continue
        number, particle, i_value, density = title.groups()
        if particle != ICRU90_WATER_TABLES[number]:
            raise ExtractionError(f"table {number} is for {particle!r}")
        electrons = number == "A.3"
        rows = tuple(
            tuple(line)
            for line in lines
            if len(line) in ((7,) if electrons else (5, 6))
            and all(re.fullmatch(_FLOAT, f) for f in line)
            and "E" in line[1]
        )
        if electrons:
            columns: tuple[str, ...] = _ESTAR_COLUMNS
            parts = ("S_col_MeV_cm2_g", "S_rad_MeV_cm2_g", "S_tot_MeV_cm2_g")
        else:
            columns = (
                "T_MeV",
                "S_el_MeV_cm2_g",
                "S_nuc_MeV_cm2_g",
                "S_tot_MeV_cm2_g",
                "csda_range_g_cm2",
                "detour_factor",
            )[: len(rows[0])]
            parts = ("S_el_MeV_cm2_g", "S_nuc_MeV_cm2_g", "S_tot_MeV_cm2_g")
        if any(len(r) != len(columns) for r in rows):
            raise ExtractionError(f"table {number}: rows of unequal length")
        table = Extracted(columns, rows)
        check_increasing(table.column("T_MeV"), f"table {number}")
        bad = check_sum_rule(table, *parts)
        if bad:
            raise ExtractionError(f"table {number}: components != total in {bad}")
        meta = {"particle": particle, "i_value_eV": i_value, "density_g_cm3": density}
        found[number] = (table, meta)
    missing = set(ICRU90_WATER_TABLES) - set(found)
    if missing:
        raise ExtractionError(f"ICRU 90 tables not found: {sorted(missing)}")
    return found


# --- NIST ESTAR (density-effect delta at I = 75 eV) ------------------------------------

_ESTAR_COLUMNS = (
    "T_MeV",
    "S_col_MeV_cm2_g",
    "S_rad_MeV_cm2_g",
    "S_tot_MeV_cm2_g",
    "csda_range_g_cm2",
    "radiation_yield",
    "delta",
)


def parse_estar_html(text: str) -> tuple[Extracted, dict[str, str]]:
    """ESTAR text output: electron table with the density-effect column ``delta``."""
    lines = [re.sub(r"<[^>]*>", "", line).strip() for line in re.split(r"<br>|\n", text)]
    if not any(re.fullmatch(r"Electrons in\s+WATER, LIQUID", line) for line in lines):
        raise ExtractionError("not an ESTAR table for liquid water")
    header = next(i for i, line in enumerate(lines) if line.startswith("Density, g/cm3"))
    density, i_value = lines[header + 1].split()
    rows = tuple(
        tuple(fields)
        for fields in (line.split() for line in lines)
        if len(fields) == len(_ESTAR_COLUMNS) and all(re.fullmatch(_FLOAT, f) for f in fields)
    )
    table = Extracted(_ESTAR_COLUMNS, rows)
    check_increasing(table.column("T_MeV"), "ESTAR")
    bad = check_sum_rule(table, "S_col_MeV_cm2_g", "S_rad_MeV_cm2_g", "S_tot_MeV_cm2_g")
    if bad:
        raise ExtractionError(f"ESTAR: collision + radiative != total in {bad}")
    return table, {"i_value_eV": i_value, "density_g_cm3": density}


# --- PDG: Sternheimer density-effect coefficients for liquid water ----------------------

_STERNHEIMER_KEYS = ("a", "k", "x0", "x1", "I_eV", "Cbar", "delta0")


def parse_pdg_sternheimer(text: str) -> dict[str, str]:
    """The ``Sternheimer coef:`` line of a PDG muon table, for liquid water (index 276)."""
    lines = text.splitlines()
    if not any("Index =  276:" in line and "water (liquid)" in line for line in lines):
        raise ExtractionError("not the PDG table for liquid water")
    for i, line in enumerate(lines):
        if line.strip().startswith("Sternheimer coef:"):
            values = lines[i + 1].split()
            if len(values) != len(_STERNHEIMER_KEYS):
                break
            return dict(zip(_STERNHEIMER_KEYS, values, strict=True))
    raise ExtractionError("no Sternheimer coefficient line")


def parse_pdg_plasma_energy(html: str) -> str:
    """The plasma energy [eV] printed on PDG's material page."""
    text = re.sub(r"&#?\w+;", " ", re.sub(r"<[^>]*>", " ", html))  # entities hold digits
    m = re.search(r"Plasma energy[^0-9]*?(\d+\.\d+)\s*eV", " ".join(text.split()))
    if m is None:
        raise ExtractionError("no plasma energy on the page")
    return m.group(1)


# --- NIST atomic weights and isotopic compositions -------------------------------------

_ISOTOPE_COLUMNS = (
    "Z",
    "symbol",
    "A",
    "relative_atomic_mass",
    "relative_atomic_mass_uncertainty",
    "isotopic_composition",
)


def parse_nist_isotopes(text: str) -> Extracted:
    """Parse the ``ascii2`` output: blocks of ``key = value`` lines, one per isotope."""
    plain = re.sub(r"<[^>]*>", "", text).replace("&nbsp;", " ")
    rows = []
    record: dict[str, str] = {}
    for line in [*plain.splitlines(), ""]:
        key, sep, value = line.partition("=")
        if sep:
            record[key.strip()] = value.strip()
            continue
        if "Relative Atomic Mass" in record:
            mass = re.fullmatch(r"([\d.]+)(?:\((\d+)#?\))?#?", record["Relative Atomic Mass"])
            if mass is None:
                raise ExtractionError(f"unreadable mass {record['Relative Atomic Mass']!r}")
            rows.append(
                (
                    record["Atomic Number"],
                    record["Atomic Symbol"],
                    record["Mass Number"],
                    mass.group(1),
                    mass.group(2) or "",
                    record.get("Isotopic Composition", ""),
                )
            )
        record = {}
    if not rows:
        raise ExtractionError("no isotope records found")
    return Extracted(_ISOTOPE_COLUMNS, tuple(rows))


# --- MSTAR archive ---------------------------------------------------------------------


def extract_mstar(archive: Path, dest: Path) -> list[Path]:
    """Extract MSTAR's sources and documentation; skip the executable; refuse odd paths."""
    written = []
    with zipfile.ZipFile(archive) as zf:
        for info in zf.infolist():
            name = PurePosixPath(info.filename)
            if info.is_dir() or not info.filename.startswith(MSTAR_KEEP_PREFIXES):
                continue
            if name.is_absolute() or ".." in name.parts:
                raise ExtractionError(f"unsafe path in archive: {info.filename!r}")
            target = dest / Path(*name.parts)
            target.parent.mkdir(parents=True, exist_ok=True)
            target.write_bytes(zf.read(info))
            written.append(target)
    if not any(p.name == "msdbs1.d" for p in written):
        raise ExtractionError("MSTAR database msdbs1.d not found in the archive")
    return written


# --- ICRU 73 via libdedx's raw files, and our own ICRU 73 CSV format -------------------


def parse_libdedx_grid(text: str) -> list[str]:
    """``*Eng.dat``: a count, then that many energies [MeV/u], one per line."""
    lines = [line.strip() for line in text.splitlines() if line.strip()]
    if not lines or int(lines[0]) != len(lines) - 1:
        raise ExtractionError("energy grid count does not match the number of energies")
    check_increasing(lines[1:], "energy grid")
    return lines[1:]


_LIBDEDX_HEADER = re.compile(r"#(\d+):(\d+):(\d+)")


def parse_libdedx_blocks(text: str) -> dict[tuple[int, int], list[str]]:
    """``#target:Z:n`` header lines, each followed by ``n`` values (printed strings)."""
    blocks: dict[tuple[int, int], list[str]] = {}
    sizes: dict[tuple[int, int], int] = {}
    key: tuple[int, int] | None = None
    for line in (raw.strip() for raw in text.splitlines()):
        if not line:
            continue
        if line.startswith("#"):
            m = _LIBDEDX_HEADER.fullmatch(line)
            if m is None:
                raise ExtractionError(f"malformed block header {line!r}")
            key = (int(m.group(1)), int(m.group(2)))
            if key in blocks:
                raise ExtractionError(f"block {key} appears twice")
            blocks[key], sizes[key] = [], int(m.group(3))
        elif key is None:
            raise ExtractionError("value before the first block header")
        else:
            blocks[key].append(line)
    for key, values in blocks.items():
        if len(values) != sizes[key]:
            raise ExtractionError(
                f"block {key}: header says {sizes[key]} values, found {len(values)}"
            )
    return blocks


def icru73_water_table(
    blocks: dict[tuple[int, int], list[str]],
    grid: Sequence[str],
    ions: Sequence[int] = ICRU73_IONS,
) -> Extracted:
    """Liquid water (target 276) for every ion in ``ions``, in our long CSV layout."""
    rows: list[tuple[str, ...]] = []
    for z in ions:
        values = blocks.get((ICRU_WATER_LIQUID, z))
        if values is None:
            raise ExtractionError(f"no liquid-water block for Z={z}")
        if len(values) != len(grid):
            raise ExtractionError(f"Z={z}: {len(values)} values for {len(grid)} energies")
        if any(Decimal(v) <= 0 for v in values):
            raise ExtractionError(f"Z={z}: stopping powers must be positive")
        rows += [(str(z), e, v) for e, v in zip(grid, values, strict=True)]
    return Extracted(("Z", "energy_MeV_u", "S_el_MeV_cm2_g"), tuple(rows))


# Metadata lines every ICRU 73 CSV must carry (format: data/stopping/icru73/README.md).
ICRU73_REQUIRED = frozenset({"edition", "material", "quantity", "energy", "source"})


def write_icru73_csv(path: Path, table: Extracted, edition: str, source: Sequence[str]) -> None:
    """Write ``table`` in our ICRU 73 format (data/stopping/icru73/README.md)."""
    header = [
        f"# edition: {edition}",
        "# material: liquid water",
        "# quantity: electronic",
        "# energy: kinetic energy per nucleon, MeV/u",
        *(f"# source: {line}" for line in source),
    ]
    with path.open("w", encoding="utf-8", newline="\n") as fh:
        fh.write("\n".join(header) + "\n")
        writer = csv.writer(fh, lineterminator="\n")
        writer.writerow(table.columns)
        writer.writerows(table.rows)


def read_icru73_csv(path: Path) -> tuple[dict[str, str], Extracted]:
    """Read our ICRU 73 format; ``source`` lines are joined, other keys appear once."""
    meta: dict[str, str] = {}
    body = []
    for line in path.read_text(encoding="utf-8").splitlines():
        if line.startswith("#"):
            key, sep, value = line[1:].partition(":")
            if sep:
                key, value = key.strip(), value.strip()
                joined = key == "source" and key in meta
                meta[key] = f"{meta[key]} {value}" if joined else value
        elif line.strip():
            body.append(line)
    missing = ICRU73_REQUIRED - set(meta)
    if missing:
        raise ExtractionError(f"{path.name}: missing metadata {sorted(missing)}")
    if meta["material"] != "liquid water":
        raise ExtractionError(f"{path.name}: only liquid water is supported")
    rows = list(csv.reader(body))
    columns = tuple(rows[0])
    if columns[:3] != ("Z", "energy_MeV_u", "S_el_MeV_cm2_g"):
        raise ExtractionError(f"{path.name}: unexpected columns {columns}")
    return meta, Extracted(columns, tuple(tuple(r) for r in rows[1:]))


# --- I/O -------------------------------------------------------------------------------


def sha256(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()


def write_extracted(path: Path, table: Extracted, source: Path, extra: Sequence[str]) -> None:
    # No raw-file checksum here: NIST's CGI pages embed a per-request token, so the raw
    # HTML changes on every fetch while the extracted values (and this file) do not.
    lines = [
        f"# Extracted from {source.name} by scripts/stopping/fetch_sources.py.",
        "# Values are the printed strings, unchanged.",
        "# Raw source data: NOT for redistribution; see PROVENANCE.md in this folder.",
        *(f"# {e}" for e in extra),
    ]
    with path.open("w", encoding="utf-8", newline="\n") as fh:
        fh.write("\n".join(lines) + "\n")
        writer = csv.writer(fh, lineterminator="\n")
        writer.writerow(table.columns)
        writer.writerows(table.rows)


def download(url: str, data: dict[str, str] | None = None) -> bytes:
    body = urllib.parse.urlencode(data).encode() if data is not None else None
    request = urllib.request.Request(url, data=body, headers={"User-Agent": USER_AGENT})
    with urllib.request.urlopen(request, timeout=120) as response:
        payload: bytes = response.read()
    return payload


@dataclass
class Fetched:
    source: str
    files: list[Path] = field(default_factory=list)
    record: dict[str, object] = field(default_factory=dict)


def _now() -> str:
    return dt.datetime.now(dt.timezone.utc).strftime("%Y-%m-%dT%H:%M:%SZ")


def fetch_star(dest: Path, program: str) -> Fetched:
    folder = dest / program.lower()
    folder.mkdir(parents=True, exist_ok=True)
    form = {"prog": program, "matno": STAR_WATER_LIQUID, "ShowDefault": "on"}
    raw = folder / f"{program.lower()}_water_liquid.html"
    raw.write_bytes(download(STAR_URL, form))
    particle = {"PSTAR": "proton", "ASTAR": "alpha"}[program]
    table = parse_star_html(raw.read_text(encoding="latin-1"), particle)
    out = folder / f"{program.lower()}_water_liquid.csv"
    write_extracted(out, table, raw, [f"{len(table.rows)} rows; T is the total kinetic energy."])
    return Fetched(
        program.lower(),
        [raw, out],
        {"url": STAR_URL, "post": form, "retrieved_utc": _now(), "rows": len(table.rows)},
    )


def fetch_icru90(dest: Path) -> Fetched:
    folder = dest / "icru90"
    folder.mkdir(parents=True, exist_ok=True)
    raw = folder / "nist_icru90_update.pdf"
    raw.write_bytes(download(ICRU90_URL))
    files = [raw]
    rows = {}
    for number, (table, meta) in parse_icru90_pdf(raw).items():
        name = {"A.3": "electron", "A.9": "proton", "A.12": "alpha", "A.15": "carbon"}[number]
        out = folder / f"icru90_{name}_water_liquid.csv"
        extra = [
            f"Table {number}: {meta['particle']} in liquid water, I = {meta['i_value_eV']} eV,",
            f"density {meta['density_g_cm3']} g/cm3; {len(table.rows)} rows; T as printed (MeV).",
        ]
        write_extracted(out, table, raw, extra)
        files.append(out)
        rows[number] = len(table.rows)
    return Fetched("icru90", files, {"url": ICRU90_URL, "retrieved_utc": _now(), "rows": rows})


def fetch_estar(dest: Path) -> Fetched:
    folder = dest / "estar"
    folder.mkdir(parents=True, exist_ok=True)
    form = {"matno": STAR_WATER_LIQUID, "ShowDefault": "on"}
    raw = folder / "estar_water_liquid.html"
    raw.write_bytes(download(ESTAR_URL, form))
    table, meta = parse_estar_html(raw.read_text(encoding="latin-1"))
    out = folder / "estar_water_liquid.csv"
    extra = [
        f"ESTAR, electrons in liquid water, I = {meta['i_value_eV']} eV, density "
        f"{meta['density_g_cm3']} g/cm3; {len(table.rows)} rows; T is the electron energy."
    ]
    write_extracted(out, table, raw, extra)
    record = {"url": ESTAR_URL, "post": form, "retrieved_utc": _now(), "rows": len(table.rows)}
    return Fetched("estar", [raw, out], record)


def fetch_pdg(dest: Path) -> Fetched:
    folder = dest / "pdg"
    folder.mkdir(parents=True, exist_ok=True)
    mu = folder / "muE_water_liquid.txt"
    page = folder / "water_liquid.html"
    mu.write_bytes(download(PDG_MUE_URL))
    page.write_bytes(download(PDG_WATER_URL))
    coefficients = parse_pdg_sternheimer(mu.read_text(encoding="latin-1"))
    plasma = parse_pdg_plasma_energy(page.read_text(encoding="latin-1"))
    table = Extracted(
        (*_STERNHEIMER_KEYS, "plasma_energy_eV"),
        ((*(coefficients[k] for k in _STERNHEIMER_KEYS), plasma),),
    )
    out = folder / "pdg_water_liquid_sternheimer.csv"
    write_extracted(out, table, mu, [f"plasma energy from {page.name}"])
    record = {"urls": [PDG_MUE_URL, PDG_WATER_URL], "retrieved_utc": _now()}
    return Fetched("pdg", [mu, page, out], record)


def fetch_mstar(dest: Path) -> Fetched:
    folder = dest / "mstar"
    archive = folder / "MSTAR312.zip"
    if not archive.is_file():
        raise SystemExit(
            f"MSTAR: download {MSTAR_URL} by hand (from https://www-nds.iaea.org/stopping/mstar,"
            f" which rejects scripted clients) and save it as {archive}, then re-run."
        )
    files = [archive, *extract_mstar(archive, folder)]
    return Fetched("mstar", files, {"url": MSTAR_URL, "checked_utc": _now()})


def fetch_isotopes(dest: Path) -> Fetched:
    folder = dest / "nist_isotopes"
    folder.mkdir(parents=True, exist_ok=True)
    raw = folder / "nist_isotopic_compositions.html"
    raw.write_bytes(download(ISOTOPES_URL))
    table = parse_nist_isotopes(raw.read_text(encoding="utf-8", errors="replace"))
    out = folder / "nist_isotopic_compositions.csv"
    write_extracted(out, table, raw, [f"{len(table.rows)} isotopes."])
    return Fetched(
        "nist_isotopes",
        [raw, out],
        {"url": ISOTOPES_URL, "retrieved_utc": _now(), "rows": len(table.rows)},
    )


def fetch_icru73(dest: Path) -> Fetched:
    folder = dest / "icru73"
    raw_dir = folder / "libdedx"
    raw_dir.mkdir(parents=True, exist_ok=True)
    files = []
    for edition, (data_name, grid_name) in ICRU73_FILES.items():
        for name in (data_name, grid_name):
            (raw_dir / name).write_bytes(download(LIBDEDX_RAW + name))
            files.append(raw_dir / name)
        grid = parse_libdedx_grid((raw_dir / grid_name).read_text(encoding="utf-8"))
        blocks = parse_libdedx_blocks((raw_dir / data_name).read_text(encoding="utf-8"))
        table = icru73_water_table(blocks, grid)
        out = folder / f"icru73_{edition}_water_liquid.csv"
        source = [
            f"libdedx data/raw/{data_name} + {grid_name} (commit {LIBDEDX_COMMIT}),",
            "surrogate for the ICRU Report 73 tables; liquid water (ID 276), Z 3-18.",
        ]
        write_icru73_csv(out, table, edition, source)
        files.append(out)
    record = {"url": LIBDEDX_RAW, "commit": LIBDEDX_COMMIT, "retrieved_utc": _now()}
    return Fetched("icru73", files, record)


ARB1973_COLUMNS = ("x", "kappa_b1.6", "kappa_b1.8", "kappa_b2.0")


def read_arb1973_csv(path: Path) -> Extracted:
    """Our hand transcription of Ashley, Ritchie & Brandt (1973), Table I (checked)."""
    lines = [line for line in path.read_text(encoding="utf-8").splitlines() if line]
    rows = list(csv.reader(line for line in lines if not line.startswith("#")))
    if tuple(rows[0]) != ARB1973_COLUMNS:
        raise ValueError(f"{path.name}: expected columns {ARB1973_COLUMNS}, got {rows[0]}")
    table = Extracted(ARB1973_COLUMNS, tuple(tuple(r) for r in rows[1:]))
    if len(table.rows) != 41:
        raise ValueError(f"{path.name}: Table I has 41 rows, found {len(table.rows)}")
    check_increasing(table.column("x"), "x")
    return table


def fetch_arb1973(dest: Path) -> Fetched:
    path = dest / "arb1973" / "arb1973_table1.csv"
    if not path.is_file():
        raise SystemExit(
            f"ARB 1973: {path} is a hand transcription of Phys. Rev. A 8, 2402 (1973), "
            "Table I; see data/stopping/arb1973/PROVENANCE.md (it cannot be downloaded)."
        )
    read_arb1973_csv(path)
    return Fetched("arb1973", [path], {"doi": "10.1103/PhysRevA.8.2402", "checked_utc": _now()})


ICRU49_QUANTITIES = {
    "proton": (
        "A1",
        "A2",
        "A3",
        "A4",
        "A5",
        "molecular_weight",
        "proton_mass_u",
        "T1_MeV",
        "T2_MeV",
    ),
    "alpha": ("a1", "a2", "a3", "a4", "a5", "molecular_weight", "T1_MeV", "T2_MeV"),
}


def read_icru49_csv(path: Path) -> dict[str, dict[str, str]]:
    """Our transcription of ICRU 49's liquid-water coefficients: particle -> {name: value}."""
    lines = path.read_text(encoding="utf-8").splitlines()
    rows = list(csv.DictReader(line for line in lines if line and not line.startswith("#")))
    out: dict[str, dict[str, str]] = {}
    for row in rows:
        out.setdefault(row["particle"], {})[row["quantity"]] = row["value"]
    for particle, names in ICRU49_QUANTITIES.items():
        if set(out.get(particle, {})) != set(names):
            raise ExtractionError(f"{path.name}: {particle} needs exactly {names}")
    return out


ICRU49_TABLE41_COLUMNS = ("T_hat", "S_moliere", "S_zbl")


def read_icru49_table41(path: Path) -> Extracted:
    """Our transcription of ICRU 49 Table 4.1 (104 rows, increasing T-hat)."""
    lines = [line for line in path.read_text(encoding="utf-8").splitlines() if line]
    rows = list(csv.reader(line for line in lines if not line.startswith("#")))
    if tuple(rows[0]) != ICRU49_TABLE41_COLUMNS:
        raise ExtractionError(f"{path.name}: expected columns {ICRU49_TABLE41_COLUMNS}")
    table = Extracted(ICRU49_TABLE41_COLUMNS, tuple(tuple(r) for r in rows[1:]))
    if len(table.rows) != 104:
        raise ExtractionError(f"{path.name}: Table 4.1 has 104 rows, found {len(table.rows)}")
    check_increasing(table.column("T_hat"), "T_hat")
    return table


def fetch_icru49(dest: Path) -> Fetched:
    folder = dest / "icru49"
    paths = [folder / "icru49_water_empirical.csv", folder / "icru49_table41_reduced_nuclear.csv"]
    for path in paths:
        if not path.is_file():
            raise SystemExit(
                f"ICRU 49: {path} is a hand transcription from the report; see "
                "data/stopping/icru49/PROVENANCE.md (it cannot be downloaded)."
            )
    read_icru49_csv(paths[0])
    read_icru49_table41(paths[1])
    return Fetched("icru49", paths, {"isbn": "978-0-913394-47-2", "checked_utc": _now()})


AHLEN1980_COLUMNS = ("abs_z1_alpha_over_beta", "cos_chi")


def read_ahlen1980_csv(path: Path) -> Extracted:
    """Our transcription of Ahlen (1980) Table XII (14 rows, increasing |Z alpha/beta|)."""
    lines = [line for line in path.read_text(encoding="utf-8").splitlines() if line]
    rows = list(csv.reader(line for line in lines if not line.startswith("#")))
    if tuple(rows[0]) != AHLEN1980_COLUMNS:
        raise ExtractionError(f"{path.name}: expected columns {AHLEN1980_COLUMNS}")
    table = Extracted(AHLEN1980_COLUMNS, tuple(tuple(r) for r in rows[1:]))
    if len(table.rows) != 14:
        raise ExtractionError(f"{path.name}: Table XII has 14 rows, found {len(table.rows)}")
    check_increasing(table.column("abs_z1_alpha_over_beta"), "|Z alpha/beta|")
    return table


def fetch_ahlen1980(dest: Path) -> Fetched:
    path = dest / "ahlen1980" / "ahlen1980_table12.csv"
    if not path.is_file():
        raise SystemExit(
            f"Ahlen 1980: {path} is a hand transcription of Rev. Mod. Phys. 52, 121, Table XII; "
            "see data/stopping/ahlen1980/PROVENANCE.md (it cannot be downloaded)."
        )
    read_ahlen1980_csv(path)
    return Fetched("ahlen1980", [path], {"doi": "10.1103/RevModPhys.52.121", "checked_utc": _now()})


BRANDT1982_COLUMNS = ("y_r", "q")


def read_brandt1982_csv(path: Path) -> Extracted:
    """Our transcription of Brandt & Kitagawa (1982) Table II (24 rows, increasing y_r)."""
    lines = [line for line in path.read_text(encoding="utf-8").splitlines() if line]
    rows = list(csv.reader(line for line in lines if not line.startswith("#")))
    if tuple(rows[0]) != BRANDT1982_COLUMNS:
        raise ExtractionError(f"{path.name}: expected columns {BRANDT1982_COLUMNS}")
    table = Extracted(BRANDT1982_COLUMNS, tuple(tuple(r) for r in rows[1:]))
    if len(table.rows) != 24:
        raise ExtractionError(f"{path.name}: Table II has 24 rows, found {len(table.rows)}")
    check_increasing(table.column("y_r"), "y_r")
    check_increasing(table.column("q"), "q")
    return table


def fetch_brandt1982(dest: Path) -> Fetched:
    path = dest / "brandt1982" / "brandt1982_table2.csv"
    if not path.is_file():
        raise SystemExit(
            f"Brandt & Kitagawa 1982: {path} is a hand transcription of Phys. Rev. B 25, 5631, "
            "Table II; see data/stopping/brandt1982/PROVENANCE.md (it cannot be downloaded)."
        )
    read_brandt1982_csv(path)
    return Fetched("brandt1982", [path], {"doi": "10.1103/PhysRevB.25.5631", "checked_utc": _now()})


FETCHERS: dict[str, Callable[[Path], Fetched]] = {
    "ahlen1980": fetch_ahlen1980,
    "brandt1982": fetch_brandt1982,
    "arb1973": fetch_arb1973,
    "icru49": fetch_icru49,
    "icru73": fetch_icru73,
    "estar": fetch_estar,
    "pdg": fetch_pdg,
    "pstar": lambda d: fetch_star(d, "PSTAR"),
    "astar": lambda d: fetch_star(d, "ASTAR"),
    "icru90": fetch_icru90,
    "mstar": fetch_mstar,
    "nist_isotopes": fetch_isotopes,
}


# --- PROVENANCE checksum verification --------------------------------------------------

_CHECKSUM_ROW = re.compile(r"^\|\s*`([^`]+)`\s*\|\s*`([0-9a-f]{64})`\s*\|")


def provenance_checksums(provenance: Path) -> dict[str, str]:
    """``file -> sha256`` from the ``| `file` | `sha256` | ...`` rows of a PROVENANCE.md."""
    out = {}
    for line in provenance.read_text(encoding="utf-8").splitlines():
        m = _CHECKSUM_ROW.match(line)
        if m:
            out[m.group(1)] = m.group(2)
    return out


def verify(dest: Path) -> list[str]:
    """Mismatches between files on disk and their PROVENANCE.md checksums."""
    problems = []
    for provenance in sorted(dest.glob("*/PROVENANCE.md")):
        for name, expected in provenance_checksums(provenance).items():
            path = provenance.parent / name
            if not path.is_file():
                problems.append(f"missing: {path}")
            elif sha256(path) != expected:
                problems.append(f"checksum differs: {path}")
    return problems


def main(argv: Sequence[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    parser.add_argument("--dest", type=Path, default=DEFAULT_DEST)
    parser.add_argument("--only", nargs="+", choices=sorted(FETCHERS), default=sorted(FETCHERS))
    parser.add_argument("--verify", action="store_true", help="check checksums only")
    args = parser.parse_args(argv)

    if args.verify:
        problems = verify(args.dest)
        print("\n".join(problems) or "all PROVENANCE checksums match")
        return 1 if problems else 0

    for name in args.only:
        fetched = FETCHERS[name](args.dest)
        folder = args.dest / fetched.source
        (folder / "FETCHED.json").write_text(
            json.dumps(fetched.record, indent=2, sort_keys=True) + "\n", encoding="utf-8"
        )
        print(f"\n## {name}  ({json.dumps(fetched.record, sort_keys=True)})")
        print("| file | sha256 | bytes |\n|---|---|---|")
        for path in fetched.files:
            rel = path.relative_to(folder).as_posix()
            print(f"| `{rel}` | `{sha256(path)}` | {path.stat().st_size} |")
    return 0


if __name__ == "__main__":
    sys.exit(main())
