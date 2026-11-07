"""Regenerate the shipped stopping-power tables from the raw sources.

Reads the extracted raw files in ``data/stopping/<source>/`` (see
``scripts/stopping/fetch_sources.py``) and writes OUR regridded tables into
``pyradbio/track/stopping/data/``:

* ``pstar_water_liquid.csv`` / ``astar_water_liquid.csv`` -- electronic and nuclear
  columns interpolated (not-a-knot cubic in log-log) from NIST's grid onto a
  log-uniform grid of 50 points per decade over the source's exact range;
* ``icru73_2005_water_liquid.csv`` / ``icru73_2009_water_liquid.csv`` -- ICRU 73 and
  its errata for Z 3-18 (electronic only), from the libdedx surrogate files;
* ``icru90_water_liquid.csv`` -- ICRU 90 protons, He-4 and C-12 (electronic, nuclear);
* ``mstar_c_water_liquid.csv`` / ``mstar_d_water_liquid.csv`` -- our MSTAR 3.12
  port (modes 'c' and 'd', which 'a' and 'b' resolve to for liquid water) evaluated
  DIRECTLY at the nodes of a 50-points-per-decade grid, Z 3-18 (needs the raw
  ``msdbs1.d``; the database itself is never shipped);
* ``density_effect_water_liquid.csv`` -- delta(beta*gamma) of liquid water from ESTAR
  (I = 75 eV) and ICRU 90 table A.3 (I = 78 eV), interpolated (PCHIP in log10 of
  beta*gamma) onto 50 points per decade;
* ``sternheimer_water_liquid.csv`` -- PDG's Sternheimer coefficients for water (CC BY
  4.0), which belong to I = 75 eV;
* ``nist_isotope_masses.csv`` -- isotope masses for Z 1-92 (reprint confirmed);
* ``barkas_arb1973.csv`` -- the Barkas function F(b/x^1/2) = kappa x^1/2 ln(x/0.18) from
  our transcription of Ashley, Ritchie & Brandt (1973) Table I (x >= 1 only),
  interpolated (cubic in ln x) onto 50 points per decade;
* ``icru49_empirical_water.csv`` -- ICRU 49's empirical (Varelas-Biersack) proton and
  alpha stopping of liquid water, our transcription of its coefficients evaluated
  DIRECTLY at the nodes of a 50-points-per-decade grid up to the cut-off T1, with T1 and
  T2 of ICRU 49 Table 3.7 in each ion's metadata;
* ``ahlen1980_cos_chi.csv`` -- cos(chi) of Ahlen (1980) Table XII for his Mott correction,
  our transcription interpolated (cubic) onto steps of 0.01 in |Z alpha/beta|;
* ``brandt1982_ionization.csv`` -- Brandt & Kitagawa (1982) Table II, ion ionization
  q(y_r) for their effective charge, our transcription interpolated (PCHIP of ln(1-q));
* ``shell_corrections_water.csv`` -- COMPUTED, no raw input: Walske's K/L-shell and the
  hydrogen-atom shell corrections C(eta) at the default thetas of
  :mod:`pyradbio.track.stopping.shell` (~40 s; ``--no-computed`` skips it).

The output is deterministic (no timestamps; the raw inputs are identified by the
SHA-256 of their extracted CSVs), so ``--check`` can rebuild into a temporary folder
and compare byte for byte with what is shipped.

Usage::

    python scripts/stopping/build_tables.py [--raw data/stopping] [--out <dir>]
    python scripts/stopping/build_tables.py --check
"""

from __future__ import annotations

import argparse
import csv
import hashlib
import re
import sys
import tempfile
from collections.abc import Sequence
from pathlib import Path

import numpy as np

from pyradbio.track.stopping.base import IonSourceInfo, OptionValue
from pyradbio.track.stopping.grid import (
    IonTable,
    LogEnergyGrid,
    RegriddedTable,
    write_table_csv,
)
from pyradbio.track.stopping.masses import most_abundant_mass_number
from pyradbio.track.stopping.mstar import MSTAR, MSTAR_I_VALUE_EV, MSTAR_REFERENCE

sys.path.insert(0, str(Path(__file__).resolve().parent))
from fetch_sources import (
    read_ahlen1980_csv,
    read_arb1973_csv,
    read_brandt1982_csv,
    read_icru49_csv,
    read_icru49_table41,
    read_icru73_csv,
)

REPO_ROOT = Path(__file__).resolve().parents[2]
RAW = REPO_ROOT / "data" / "stopping"
OUT = REPO_ROOT / "pyradbio" / "track" / "stopping" / "data"

#: Shipped-table resolution.
POINTS_PER_DECADE = 50
METHOD = "cubic"

NIST_STAR_REFERENCE = (
    "M.J. Berger, J.S. Coursey, M.A. Zucker, J. Chang, ESTAR, PSTAR, and ASTAR, NIST "
    "Standard Reference Database 124, NISTIR 4999, doi:10.18434/T4NC7P; ICRU Report 49 (1993)"
)
NIST_SRD_NOTICE = (
    "Source data: NIST Standard Reference Data, copyright secured by the U.S. Secretary "
    "of Commerce under 15 U.S.C. 290e. This file is NOT the NIST table: it holds values "
    "interpolated by pyRadBio onto its own grid."
)

#: XSTAR's fixed mean excitation energy of water (ICRU 49).
I_VALUE_EV = 75.0

ICRU73_REFERENCE = (
    "ICRU Report 73 (2005), Stopping of Ions Heavier than Helium, J. ICRU 5(1), "
    "doi:10.1093/jicru/ndi001"
)
ICRU73_ERRATA_REFERENCE = (
    "P. Sigmund, A. Schinner and H. Paul, Errata and Addenda for ICRU Report 73, "
    "Stopping of Ions Heavier than Helium (May 14, 2009; standalone document, no "
    "journal or DOI printed), Appendix B, replacing ICRU 73 pp. 232-233"
)
#: Water I-values of the two editions, from the errata, sec. 2.2: "The oscillator-strength
#: spectrum used in Report 73 for liquid water gives an I-value of 67.2 eV"; the revised
#: tables use "Liquid water, I = 78.0 eV" (errata Table 1). ICRU 73 itself prints none.
ICRU73_I_VALUE_EV = {"2005": 67.2, "2009": 78.0}
ICRU73_I_VALUE_NOTE = {
    "2005": (
        "67.2 eV is implied by ICRU 73's oscillator-strength spectrum for liquid water "
        "(stated in the 2009 errata, sec. 2.2); ICRU 73 itself prints no water I-value"
    ),
    "2009": "78.0 eV: errata (Sigmund, Schinner & Paul 2009), sec. 2.2 and Table 1",
}
ICRU90_REFERENCE = (
    "ICRU Report 90 (2014, released 2016), Key Data for Ionizing-Radiation Dosimetry, "
    "J. ICRU 14; tables distributed by NIST as an addendum to SRD 124, doi:10.18434/T4NC7P"
)
#: ICRU 90's mean excitation energy of liquid water, printed in the titles of tables
#: A.9, A.12 and A.15 of the NIST addendum ("I = 78 eV").
ICRU90_I_VALUE_EV = 78.0


def _sha256(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()


def _read_extracted(path: Path) -> dict[str, np.ndarray]:
    rows = list(csv.DictReader(line for line in path.open() if not line.startswith("#")))
    return {key: np.array([float(r[key]) for r in rows]) for key in rows[0]}


def build_star(raw: Path, out: Path, program: str) -> Path:
    """Regrid one NIST STAR table (p: A=1, He-4: A=4; T/A is our axis)."""
    source = raw / program / f"{program}_water_liquid.csv"
    data = _read_extracted(source)
    z, a, particle = {"pstar": (1, 1, "protons"), "astar": (2, 4, "He-4 ions")}[program]
    energy = data["T_MeV"] / a
    grid = LogEnergyGrid.from_points_per_decade(
        float(energy[0]), float(energy[-1]), POINTS_PER_DECADE
    )
    columns = {
        "electronic_MeV_cm2_g": RegriddedTable.from_source(
            energy, data["S_el_MeV_cm2_g"], grid, METHOD
        ).values,
        "nuclear_MeV_cm2_g": RegriddedTable.from_source(
            energy, data["S_nuc_MeV_cm2_g"], grid, METHOD
        ).values,
    }
    info = IonSourceInfo(
        Z=z,
        A=a,
        source=program.upper(),
        reference=NIST_STAR_REFERENCE,
        i_value_eV=I_VALUE_EV,
        energy_range_MeV_u=(grid.e_min_MeV_u, grid.e_max_MeV_u),
        edition="ICRU 49",
        options={"quantities": "electronic, nuclear"},
    )
    target = out / f"{program}_water_liquid.csv"
    write_table_csv(
        target,
        [IonTable(info, grid.energies, columns)],
        metadata={
            "generator": "scripts/stopping/build_tables.py",
            "grid": grid.to_dict(),
            "interpolation": f"{METHOD} (not-a-knot) in log E -> log S",
            "raw_input": {"file": source.name, "sha256": _sha256(source)},
            "energy_axis": "kinetic energy per nucleon T/A [MeV/u]",
        },
        notes=[
            f"Mass stopping power of liquid water for {particle}, regridded by pyRadBio",
            f"from NIST {program.upper()} (ICRU 49, I = 75 eV) onto {POINTS_PER_DECADE}",
            "log-uniform points per decade. Provenance: data/stopping/"
            f"{program}/PROVENANCE.md in the pyRadBio repository.",
            NIST_SRD_NOTICE,
        ],
    )
    return target


def _regridded_ion(
    info_kwargs: dict[str, object],
    energy: np.ndarray,
    columns_in: dict[str, np.ndarray],
) -> IonTable:
    grid = LogEnergyGrid.from_points_per_decade(
        float(energy[0]), float(energy[-1]), POINTS_PER_DECADE
    )
    columns = {
        name: RegriddedTable.from_source(energy, values, grid, METHOD).values
        for name, values in columns_in.items()
    }
    info = IonSourceInfo(
        energy_range_MeV_u=(grid.e_min_MeV_u, grid.e_max_MeV_u),
        **info_kwargs,  # type: ignore[arg-type]
    )
    return IonTable(info, grid.energies, columns)


def build_icru73(raw: Path, out: Path, edition: str) -> Path:
    """ICRU 73 (``2005``) or its errata (``2009``), Z 3-18, from our ICRU 73 CSV format."""
    source = raw / "icru73" / f"icru73_{edition}_water_liquid.csv"
    meta, table = read_icru73_csv(source)
    if meta["edition"] != edition:
        raise ValueError(f"{source.name} says edition {meta['edition']!r}")
    zs = sorted({int(row[0]) for row in table.rows})
    tables = []
    for z in zs:
        rows = [row for row in table.rows if int(row[0]) == z]
        energy = np.array([float(row[1]) for row in rows])
        a = most_abundant_mass_number(z)
        options: dict[str, OptionValue] = {
            "quantities": "electronic",
            "surrogate": (
                f"{meta['source']} Verified to equal the printed table "
                + ("(ICRU 73 pp. 232-233)" if edition == "2005" else "(errata App. B)")
                + " at all 848 points."
            ),
            "i_value_note": ICRU73_I_VALUE_NOTE[edition],
            "isotope_note": (
                "ICRU 73 tabulates E/A1 per element and makes 'no distinction ... between "
                "energy per nucleon and energy per atomic mass unit' (its footnote 6); "
                "stored as the most abundant isotope, other isotopes are velocity-scaled"
            ),
        }
        tables.append(
            _regridded_ion(
                {
                    "Z": z,
                    "A": a,
                    "source": "ICRU 73 (2005)" if edition == "2005" else "ICRU 73 errata (2009)",
                    "reference": ICRU73_REFERENCE if edition == "2005" else ICRU73_ERRATA_REFERENCE,
                    "i_value_eV": ICRU73_I_VALUE_EV[edition],
                    "edition": edition,
                    "options": options,
                },
                energy,
                {"electronic_MeV_cm2_g": np.array([float(row[2]) for row in rows])},
            )
        )
    target = out / f"icru73_{edition}_water_liquid.csv"
    label = "ICRU Report 73" + (" with its 2009 errata" if edition == "2009" else "")
    write_table_csv(
        target,
        tables,
        metadata={
            "generator": "scripts/stopping/build_tables.py",
            "points_per_decade": POINTS_PER_DECADE,
            "interpolation": f"{METHOD} (not-a-knot) in log E -> log S",
            "raw_input": {"file": source.name, "sha256": _sha256(source)},
            "energy_axis": "kinetic energy per nucleon T/A [MeV/u]",
        },
        notes=[
            f"Electronic mass stopping power of liquid water for Z 3-18 after {label},",
            f"regridded by pyRadBio onto {POINTS_PER_DECADE} log-uniform points per decade.",
            "Input: the ICRU 73 tables as distributed in libdedx's raw data (GPL-3.0",
            "repository, pinned commit), verified equal to the printed report tables;",
            "provenance: data/stopping/icru73/PROVENANCE.md in the pyRadBio repository.",
            "This file is NOT the ICRU table: it holds interpolated values.",
        ],
    )
    return target


def build_icru90(raw: Path, out: Path) -> Path:
    """ICRU 90 protons, He-4 and C-12 (NIST addendum tables A.9, A.12, A.15; T/A axis)."""
    tables, inputs = [], {}
    for particle, z, a in (("proton", 1, 1), ("alpha", 2, 4), ("carbon", 6, 12)):
        source = raw / "icru90" / f"icru90_{particle}_water_liquid.csv"
        data = _read_extracted(source)
        inputs[particle] = {"file": source.name, "sha256": _sha256(source)}
        tables.append(
            _regridded_ion(
                {
                    "Z": z,
                    "A": a,
                    "source": "ICRU 90",
                    "reference": ICRU90_REFERENCE,
                    "i_value_eV": ICRU90_I_VALUE_EV,
                    "edition": "ICRU 90",
                    "options": {"quantities": "electronic, nuclear"},
                },
                data["T_MeV"] / a,
                {
                    "electronic_MeV_cm2_g": data["S_el_MeV_cm2_g"],
                    "nuclear_MeV_cm2_g": data["S_nuc_MeV_cm2_g"],
                },
            )
        )
    target = out / "icru90_water_liquid.csv"
    write_table_csv(
        target,
        tables,
        metadata={
            "generator": "scripts/stopping/build_tables.py",
            "points_per_decade": POINTS_PER_DECADE,
            "interpolation": f"{METHOD} (not-a-knot) in log E -> log S",
            "raw_input": inputs,
            "energy_axis": "kinetic energy per nucleon T/A [MeV/u]",
        },
        notes=[
            "Mass stopping power of liquid water for protons, He-4 and C-12 after ICRU",
            "Report 90 (I = 78 eV), from NIST's ICRU 90 addendum to SRD 124 (tables A.9,",
            f"A.12, A.15), regridded by pyRadBio onto {POINTS_PER_DECADE} log-uniform points",
            "per decade. Provenance: data/stopping/icru90/PROVENANCE.md in the pyRadBio",
            "repository.",
            NIST_SRD_NOTICE,
        ],
    )
    return target


def build_mstar(raw: Path, out: Path, mode: str) -> Path:
    """Our MSTAR port, mode ``c`` or ``d``, evaluated exactly at the grid nodes."""
    database = raw / "mstar" / "MSTAR312" / "Sources" / "msdbs1.d"
    model = MSTAR(database, mode=mode)
    tables = []
    for z in model.supported_Z():
        a = most_abundant_mass_number(z)
        lo, hi = model.energy_range(z, a)
        grid = LogEnergyGrid.from_points_per_decade(lo, hi, POINTS_PER_DECADE)
        info = IonSourceInfo(
            Z=z,
            A=a,
            source=f"MSTAR 3.12 (mode {mode})",
            reference=MSTAR_REFERENCE,
            i_value_eV=MSTAR_I_VALUE_EV,
            energy_range_MeV_u=(grid.e_min_MeV_u, grid.e_max_MeV_u),
            edition="3.12",
            options={"quantities": "electronic", "mstar_mode_used": mode},
        )
        values = model.mass_stopping_power(z, a, grid.energies)
        tables.append(IonTable(info, grid.energies, {"electronic_MeV_cm2_g": values}))
    target = out / f"mstar_{mode}_water_liquid.csv"
    write_table_csv(
        target,
        tables,
        metadata={
            "generator": "scripts/stopping/build_tables.py (pyradbio.track.stopping.mstar)",
            "points_per_decade": POINTS_PER_DECADE,
            "nodes": "exact values of the MSTAR port at each node (no interpolation)",
            "raw_input": {"file": database.name, "sha256": _sha256(database)},
            "energy_axis": "kinetic energy per nucleon T/A [MeV/u]",
        },
        notes=[
            "Electronic mass stopping power of liquid water for Z 3-18 from MSTAR 3.12,",
            f"mode '{mode}' (H. Paul and A. Schinner), computed by pyRadBio's Python port",
            "of the authors' FORTRAN at the nodes of a log-uniform grid of",
            f"{POINTS_PER_DECADE} points per decade. MSTAR scales M.J. Berger's ASTAR",
            "alpha stopping powers (NIST SRD 124, ICRU 49) with Paul's fitted ratios;",
            "provenance: data/stopping/mstar/PROVENANCE.md in the pyRadBio repository.",
        ],
    )
    return target


def _density_from_comment(path: Path) -> str:
    """The density [g/cm3] printed in an extracted file's header comment."""
    for line in path.read_text(encoding="utf-8").splitlines():
        m = re.search(r"density ([0-9.E+-]+) g/cm3", line)
        if m:
            return m.group(1)
    raise ValueError(f"no density in the header of {path.name}")


def build_density_effect(raw: Path, out: Path) -> Path:
    """delta(beta*gamma) of liquid water from the ESTAR and ICRU 90 electron tables."""
    from scipy.interpolate import PchipInterpolator

    from pyradbio.core.constants import ELECTRON_MASS_MEV

    sources = (
        ("ESTAR", raw / "estar" / "estar_water_liquid.csv", "75.0"),
        ("ICRU 90", raw / "icru90" / "icru90_electron_water_liquid.csv", "78.0"),
    )
    target = out / "density_effect_water_liquid.csv"
    inputs = []
    rows: list[list[str]] = []
    for name, path, i_value in sources:
        data = _read_extracted(path)
        t_mev = data["T_MeV"]
        beta_gamma = np.sqrt(t_mev * (t_mev + 2.0 * ELECTRON_MASS_MEV)) / ELECTRON_MASS_MEV
        x = np.log10(beta_gamma)
        interpolant = PchipInterpolator(x, data["delta"])
        n = int(np.ceil((x[-1] - x[0]) * POINTS_PER_DECADE - 1e-9)) + 1
        grid = np.linspace(x[0], x[-1], n)
        density = _density_from_comment(path)
        for xi, di in zip(grid, interpolant(grid), strict=True):
            rows.append([name, i_value, density, repr(float(10.0**xi)), repr(float(di))])
        inputs.append(f"{path.name} sha256 {_sha256(path)}")
    with target.open("w", encoding="utf-8", newline="\n") as fh:
        fh.write(
            "# Density-effect correction delta(beta*gamma) of liquid water, interpolated by\n"
            "# pyRadBio (PCHIP in log10(beta*gamma)) onto "
            f"{POINTS_PER_DECADE} points per decade from:\n"
            "#   ESTAR (NIST SRD 124), electrons in liquid water, I = 75 eV;\n"
            "#   NIST's ICRU Report 90 addendum, table A.3, electrons in water, I = 78 eV.\n"
            "# delta depends only on the medium and beta*gamma. Provenance: data/stopping/\n"
            "# {estar,icru90}/PROVENANCE.md. NIST SRD notice: 15 U.S.C. 290e. NOT the NIST\n"
            "# tables: interpolated values. Written by scripts/stopping/build_tables.py.\n"
            + "".join(f"# raw input: {s}\n" for s in inputs)
        )
        writer = csv.writer(fh, lineterminator="\n")
        writer.writerow(["source", "i_value_eV", "density_g_cm3", "beta_gamma", "delta"])
        writer.writerows(rows)
    return target


def build_sternheimer(raw: Path, out: Path) -> Path:
    """PDG's Sternheimer coefficients for liquid water (I = 75 eV, see PROVENANCE)."""
    source = raw / "pdg" / "pdg_water_liquid_sternheimer.csv"
    rows = list(csv.DictReader(line for line in source.open() if not line.startswith("#")))
    (row,) = rows
    target = out / "sternheimer_water_liquid.csv"
    with target.open("w", encoding="utf-8", newline="\n") as fh:
        fh.write(
            "# Sternheimer density-effect coefficients of liquid water, as printed by the\n"
            "# Particle Data Group (S. Navas et al., Phys. Rev. D 110, 030001 (2024), CC BY\n"
            "# 4.0; Atomic and Nuclear Properties, muE_water_liquid.txt), for eq. (34.7).\n"
            "# PDG lists them next to I = 79.7 eV, but Cbar = 2 ln(I/plasma energy) + 1\n"
            "# identifies I = 75 eV (data/stopping/pdg/PROVENANCE.md); they are used with\n"
            "# I = 75 eV only. Written by scripts/stopping/build_tables.py.\n"
            f"# raw input: {source.name} sha256 {_sha256(source)}\n"
        )
        writer = csv.writer(fh, lineterminator="\n")
        writer.writerow(["a", "k", "x0", "x1", "Cbar", "delta0", "plasma_energy_eV", "i_value_eV"])
        writer.writerow(
            [row[k] for k in ("a", "k", "x0", "x1", "Cbar", "delta0", "plasma_energy_eV")]
            + ["75.0"]
        )
    return target


#: Isotope masses ship up to uranium (heavy-ion branch of the default stopping power).
MAX_SHIPPED_Z = 92


def build_masses(raw: Path, out: Path) -> Path:
    source = raw / "nist_isotopes" / "nist_isotopic_compositions.csv"
    rows = list(csv.DictReader(line for line in source.open() if not line.startswith("#")))
    target = out / "nist_isotope_masses.csv"
    with target.open("w", encoding="utf-8", newline="\n") as fh:
        fh.write(
            "# Relative atomic masses [u] of the isotopes of Z 1-92, as printed by NIST,\n"
            "# Atomic Weights and Isotopic Compositions with Relative Atomic Masses, v4.1\n"
            "# (J.S. Coursey, D.J. Schwab, J.J. Tsai, R.A. Dragoset; masses from AME2012).\n"
            "# Reprinted with the maintainer's confirmation; provenance:\n"
            "# data/stopping/nist_isotopes/PROVENANCE.md. Written by build_tables.py.\n"
            f"# raw input sha256: {_sha256(source)}\n"
        )
        writer = csv.writer(fh, lineterminator="\n")
        writer.writerow(["Z", "symbol", "A", "relative_atomic_mass_u", "isotopic_composition"])
        for row in rows:
            if int(row["Z"]) <= MAX_SHIPPED_Z:
                writer.writerow(
                    [
                        row["Z"],
                        row["symbol"],
                        row["A"],
                        row["relative_atomic_mass"],
                        row["isotopic_composition"],
                    ]
                )
    return target


def build_barkas(raw: Path, out: Path) -> Path:
    """F(b/x^1/2) of ARB 1972 from the ARB 1973 kappa table, x >= 1, regridded."""
    from scipy.interpolate import CubicSpline

    from pyradbio.track.stopping.barkas import B_VALUES, f_from_kappa

    source = raw / "arb1973" / "arb1973_table1.csv"
    table = read_arb1973_csv(source)
    x = np.array([float(v) for v in table.column("x")])
    keep = x >= 1.0
    n = round(float(np.log10(x[keep][-1] / x[keep][0])) * POINTS_PER_DECADE) + 1
    grid = np.geomspace(x[keep][0], x[keep][-1], n)
    columns = {}
    for b in B_VALUES:
        kappa = np.array([float(v) for v in table.column(f"kappa_b{b:.1f}")])
        f = f_from_kappa(kappa[keep], x[keep])
        columns[f"F_b{b:.1f}"] = CubicSpline(np.log(x[keep]), f)(np.log(grid))
    target = out / "barkas_arb1973.csv"
    with target.open("w", encoding="utf-8", newline="\n") as fh:
        fh.write(
            "# Barkas function F(b/x^1/2) of J.C. Ashley, R.H. Ritchie and W. Brandt, Phys. Rev.\n"
            "# B 5, 2393 (1972), recovered as kappa x^1/2 ln(x/0.18) from their Phys. Rev. A 8,\n"
            "# 2402 (1973) Table I (eqs. 2, 3; x > 1), for b = 1.6, 1.8, 2.0; interpolated by\n"
            f"# pyRadBio (cubic in ln x) onto {POINTS_PER_DECADE} points per decade, x = 1-100.\n"
            "# NOT the ARB table: derived, interpolated values. Provenance:\n"
            "# data/stopping/arb1973/PROVENANCE.md. Written by scripts/stopping/build_tables.py.\n"
            f"# raw input: {source.name} sha256 {_sha256(source)}\n"
        )
        writer = csv.writer(fh, lineterminator="\n")
        writer.writerow(["x", *columns])
        for i, xi in enumerate(grid):
            writer.writerow([repr(float(xi)), *(repr(float(c[i])) for c in columns.values())])
    return target


#: ICRU 49's lowest tabulated energy, 1 keV (total), for protons and alphas.
ICRU49_LOWEST_MEV = 0.001


def build_icru49_empirical(raw: Path, out: Path) -> Path:
    """ICRU 49 eqs. 3.1-3.3 for water, evaluated at grid nodes from 1 keV to T1."""
    from pyradbio.track.stopping import icru49_empirical as emp

    source = raw / "icru49" / "icru49_water_empirical.csv"
    coefficients = read_icru49_csv(source)
    tables = []
    for particle, (z, a, formula) in {
        "proton": (1, 1, emp.proton_cross_section),
        "alpha": (2, 4, emp.alpha_cross_section),
    }.items():
        c = {k: float(v) for k, v in coefficients[particle].items()}
        grid = LogEnergyGrid.from_points_per_decade(
            ICRU49_LOWEST_MEV / a, c["T1_MeV"] / a, POINTS_PER_DECADE
        )
        eps = formula(grid.energies * a, c)
        values = emp.mass_stopping_from_cross_section(eps, c["molecular_weight"])
        info = IonSourceInfo(
            Z=z,
            A=a,
            source="ICRU 49 empirical",
            reference=emp.ICRU49_EMPIRICAL_REFERENCE,
            i_value_eV=None,
            energy_range_MeV_u=(grid.e_min_MeV_u, grid.e_max_MeV_u),
            edition="ICRU 49",
            options={
                "formula": "eqs. (3.1), (3.2)" if z == 1 else "eq. (3.3)",
                "T1_MeV_u": c["T1_MeV"] / a,
                "T2_MeV_u": c["T2_MeV"] / a,
            },
        )
        tables.append(IonTable(info, grid.energies, {"electronic_MeV_cm2_g": values}))
    target = out / f"{emp.SHIPPED_STEM}.csv"
    write_table_csv(
        target,
        tables,
        metadata={
            "generator": "scripts/stopping/build_tables.py",
            "evaluation": "formula evaluated at each node (no regridding)",
            "raw_input": {"file": source.name, "sha256": _sha256(source)},
            "energy_axis": "kinetic energy per nucleon T/A [MeV/u]",
        },
        notes=[
            "ICRU Report 49's empirical (Varelas-Biersack) electronic stopping of liquid",
            "water for protons and alphas, evaluated by pyRadBio at its own grid nodes",
            "from 1 keV to the cut-off T1. NOT the ICRU table: computed values.",
            "Provenance: data/stopping/icru49/PROVENANCE.md in the pyRadBio repository.",
        ],
    )
    return target


def build_icru49_reduced_nuclear(raw: Path, out: Path) -> Path:
    """ICRU 49 Table 4.1, S_hat(T_hat) for both potentials, regridded (cubic log-log)."""
    from pyradbio.track.stopping.nuclear import SHIPPED_STEM

    source = raw / "icru49" / "icru49_table41_reduced_nuclear.csv"
    table = read_icru49_table41(source)
    t_hat = np.array([float(v) for v in table.column("T_hat")])
    grid = LogEnergyGrid.from_points_per_decade(
        float(t_hat[0]), float(t_hat[-1]), POINTS_PER_DECADE
    )
    columns = {
        name: RegriddedTable.from_source(
            t_hat, np.array([float(v) for v in table.column(name)]), grid, METHOD
        ).values
        for name in ("S_moliere", "S_zbl")
    }
    target = out / f"{SHIPPED_STEM}.csv"
    with target.open("w", encoding="utf-8", newline="\n") as fh:
        fh.write(
            "# Scaled nuclear stopping power S_hat(T_hat) of ICRU Report 49 (1993), Table 4.1,\n"
            "# for the Moliere (Thomas-Fermi) and the ZBL universal potential, interpolated by\n"
            f"# pyRadBio (cubic in log-log) onto {POINTS_PER_DECADE} points per decade of T_hat.\n"
            "# NOT the ICRU table: interpolated values. Provenance:\n"
            "# data/stopping/icru49/PROVENANCE.md. Written by scripts/stopping/build_tables.py.\n"
            f"# raw input: {source.name} sha256 {_sha256(source)}\n"
        )
        writer = csv.writer(fh, lineterminator="\n")
        writer.writerow(["T_hat", *columns])
        for i, t in enumerate(grid.energies):
            writer.writerow([repr(float(t)), *(repr(float(c[i])) for c in columns.values())])
    return target


def build_ahlen_cos_chi(raw: Path, out: Path) -> Path:
    """Ahlen (1980) Table XII, cos(chi) vs |Z alpha/beta|, interpolated (cubic) onto 0.01 steps."""
    from scipy.interpolate import CubicSpline

    from pyradbio.track.stopping.higher_order import AHLEN_COS_CHI_TABLE

    source = raw / "ahlen1980" / "ahlen1980_table12.csv"
    table = read_ahlen1980_csv(source)
    y = np.array([float(v) for v in table.column("abs_z1_alpha_over_beta")])
    c = np.array([float(v) for v in table.column("cos_chi")])
    grid = np.linspace(y[0], y[-1], round((y[-1] - y[0]) / 0.01) + 1)
    values = CubicSpline(y, c)(grid)
    target = out / AHLEN_COS_CHI_TABLE
    with target.open("w", encoding="utf-8", newline="\n") as fh:
        fh.write(
            "# cos(chi) of S.P. Ahlen, Rev. Mod. Phys. 52, 121 (1980), Table XII, for the Mott\n"
            "# correction G(Z1, beta) of his eq. (4.23); interpolated by pyRadBio (cubic spline\n"
            "# through the printed values) onto steps of 0.01 in |Z1 alpha/beta| = 0-2.\n"
            "# NOT the published table: interpolated values. Provenance: data/stopping/\n"
            "# ahlen1980/PROVENANCE.md. Written by scripts/stopping/build_tables.py.\n"
            f"# raw input: {source.name} sha256 {_sha256(source)}\n"
        )
        writer = csv.writer(fh, lineterminator="\n")
        writer.writerow(["abs_z1_alpha_over_beta", "cos_chi"])
        for gi, vi in zip(grid, values, strict=True):
            writer.writerow([f"{float(gi):.2f}", repr(float(vi))])
    return target


def build_brandt_ionization(raw: Path, out: Path) -> Path:
    """Brandt & Kitagawa (1982) Table II, q(y_r): PCHIP of ln(1-q) vs ln y_r, 50 per decade."""
    from scipy.interpolate import PchipInterpolator

    from pyradbio.track.stopping.effective_charge import BRANDT_KITAGAWA_TABLE

    source = raw / "brandt1982" / "brandt1982_table2.csv"
    table = read_brandt1982_csv(source)
    y = np.array([float(v) for v in table.column("y_r")])
    q = np.array([float(v) for v in table.column("q")])
    grid = LogEnergyGrid.from_points_per_decade(float(y[0]), float(y[-1]), POINTS_PER_DECADE)
    values = 1.0 - np.exp(PchipInterpolator(np.log(y), np.log(1.0 - q))(np.log(grid.energies)))
    target = out / BRANDT_KITAGAWA_TABLE
    with target.open("w", encoding="utf-8", newline="\n") as fh:
        fh.write(
            "# Mean ionization q of ions vs reduced relative velocity y_r = v_r/(Z1^(2/3) v0) of\n"
            "# W. Brandt and M. Kitagawa, Phys. Rev. B 25, 5631 (1982), Table II (b = 1.33);\n"
            f"# interpolated by pyRadBio (PCHIP of ln(1-q) vs ln y_r) onto {POINTS_PER_DECADE}\n"
            "# points per decade. NOT the published table: interpolated values. Provenance:\n"
            "# data/stopping/brandt1982/PROVENANCE.md. Written by\n"
            "# scripts/stopping/build_tables.py.\n"
            f"# raw input: {source.name} sha256 {_sha256(source)}\n"
        )
        writer = csv.writer(fh, lineterminator="\n")
        writer.writerow(["y_r", "q"])
        for yi, qi in zip(grid.energies, values, strict=True):
            writer.writerow([repr(float(yi)), repr(float(qi))])
    return target


def build_shell(out: Path) -> Path:
    """Walske / hydrogen shell corrections C(eta) at the default thetas (computed)."""
    from pyradbio.track.stopping import shell

    theta_k, theta_l = shell.DEFAULT_THETA_K, shell.DEFAULT_THETA_L
    table = shell.shell_table(theta_k, theta_l)
    target = out / shell.SHIPPED_TABLE
    with target.open("w", encoding="utf-8", newline="\n") as fh:
        fh.write(
            "# Shell corrections C(eta) computed by pyRadBio (pyradbio.track.stopping.shell) from\n"
            "# M.C. Walske, Phys. Rev. 88, 1283 (1952) [K shell, 2 electrons] and Phys. Rev. 101,\n"
            "# 940 (1956) [L shell, 8 electrons], and for the hydrogen atom [1 electron]; no\n"
            "# raw input. eta = m v^2 / (2 Z_eff^2 Ry). 12 significant digits.\n"
            "# Written by scripts/stopping/build_tables.py.\n"
            f"#@ theta_K = {theta_k!r}\n"
            f"#@ theta_L = {theta_l!r}\n"
        )
        writer = csv.writer(fh, lineterminator="\n")
        writer.writerow(list(table))
        for i in range(len(table["eta"])):
            writer.writerow([f"{float(v[i]):.11e}" for v in table.values()])
    return target


def build_all(raw: Path, out: Path, *, computed: bool = True) -> list[Path]:
    out.mkdir(parents=True, exist_ok=True)
    # Masses first: the ICRU 73 builder stores each element as its most abundant
    # isotope, read through pyradbio.track.stopping.masses from the shipped file.
    return [
        build_masses(raw, out),
        build_star(raw, out, "pstar"),
        build_star(raw, out, "astar"),
        build_icru73(raw, out, "2005"),
        build_icru73(raw, out, "2009"),
        build_icru90(raw, out),
        build_mstar(raw, out, "c"),
        build_mstar(raw, out, "d"),
        build_density_effect(raw, out),
        build_sternheimer(raw, out),
        build_barkas(raw, out),
        build_icru49_empirical(raw, out),
        build_icru49_reduced_nuclear(raw, out),
        build_ahlen_cos_chi(raw, out),
        build_brandt_ionization(raw, out),
        *([build_shell(out)] if computed else []),
    ]


def check(raw: Path, shipped: Path, *, computed: bool = True) -> list[str]:
    """Shipped files that differ from a fresh rebuild (empty when up to date)."""
    with tempfile.TemporaryDirectory() as tmp:
        return [
            path.name
            for path in build_all(raw, Path(tmp), computed=computed)
            if not (shipped / path.name).is_file()
            or (shipped / path.name).read_bytes() != path.read_bytes()
        ]


def main(argv: Sequence[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    parser.add_argument("--raw", type=Path, default=RAW)
    parser.add_argument("--out", type=Path, default=OUT)
    parser.add_argument("--check", action="store_true", help="compare a rebuild with --out")
    parser.add_argument("--no-computed", action="store_true", help="skip the computed tables")
    args = parser.parse_args(argv)
    if args.check:
        stale = check(args.raw, args.out, computed=not args.no_computed)
        print("\n".join(f"stale: {name}" for name in stale) or "shipped tables are up to date")
        return 1 if stale else 0
    for path in build_all(args.raw, args.out, computed=not args.no_computed):
        print(f"wrote {path} ({path.stat().st_size} bytes)")
    return 0


if __name__ == "__main__":
    sys.exit(main())
