"""Generate the synthetic PIDE-schema fixture used by the PIDE loader tests.

This is NOT real PIDE data (which is non-redistributable). It is a tiny, made-up
table whose columns match the canonical schema documented in
``docs/references/pide_schema.md`` -- enough to exercise the loader end to end in
CI. Regenerate with::

    python tests/benchmark/fixtures/make_synthetic_pide.py

The numbers are physically plausible but invented; do not treat them as references.
"""

from __future__ import annotations

from pathlib import Path

from openpyxl import Workbook

HEADERS = [
    "Cell",
    "Ion",
    "Charge",
    "Energy",
    "LET",
    "alpha_I",
    "beta_I",
    "alpha_X",
    "beta_X",
    "PhotonRadiation",
]

# (cell, ion, Z, E[MeV/u], LET[keV/um], a_ion, b_ion, a_x, b_x, photon)
ROWS: list[tuple[object, ...]] = [
    ("V79", "C-12", 6, 100.0, 30.0, 0.45, 0.045, 0.13, 0.05, "200kVp"),
    ("V79", "C-12", 6, 11.0, 153.0, 0.90, 0.030, 0.13, 0.05, "200kVp"),
    ("HSG", "He-4", 2, 20.0, 20.0, 0.60, 0.060, 0.31, 0.062, "60Co"),
    ("CHO", "H-1", 1, 5.0, 8.0, 0.35, 0.055, 0.20, 0.040, "6MV"),
    # Energy left blank on purpose (LET-only entry): loader keeps it, energy=None.
    # PhotonRadiation "N/A" must normalise to "" (PIDE's missing marker).
    ("V79", "12C", 6, None, 75.0, 0.70, 0.038, 0.13, 0.05, "N/A"),
    # Incomplete row (missing beta_X): loader must skip it.
    ("Bad", "C-12", 6, 50.0, 40.0, 0.5, 0.04, 0.13, None, "250kVp"),
    # Unknown ion + no charge: loader must skip it.
    ("Weird", "Zz-99", None, 10.0, 12.0, 0.4, 0.03, 0.2, 0.05, "137Cs"),
]


def build_workbook() -> Workbook:
    wb = Workbook()
    ws = wb.active
    ws.title = "PIDE"
    ws.append(HEADERS)
    for row in ROWS:
        ws.append(row)
    return wb


def main() -> None:
    out = Path(__file__).with_name("synthetic_pide.xlsx")
    build_workbook().save(out)
    print(f"wrote {out}")


if __name__ == "__main__":
    main()
