"""Isotope masses for ion kinematics, shipped from NIST's isotope table.

Relative atomic masses (in u) for every isotope of Z = 1-92 listed by J.S. Coursey,
D.J. Schwab, J.J. Tsai and R.A. Dragoset, *Atomic Weights and Isotopic Compositions
with Relative Atomic Masses*, NIST PML, version 4.1, whose isotope masses are those of
the AME2012 atomic mass evaluation (provenance: ``data/stopping/nist_isotopes/``;
reprinting confirmed by the maintainer). The shipped file
``data/nist_isotope_masses.csv`` is written by ``scripts/stopping/build_tables.py``.

The nuclear (bare-ion) rest energy is the atomic mass minus ``Z`` electron masses;
electron binding energies (at most ~14 keV in total for argon) are neglected.
"""

from __future__ import annotations

import csv
import re
from functools import cache, lru_cache
from importlib import resources

from pyradbio.core.constants import ATOMIC_MASS_UNIT_MEV, ELECTRON_MASS_MEV

MASS_SOURCE = (
    "NIST Atomic Weights and Isotopic Compositions v4.1 (Coursey et al.); "
    "isotope masses from AME2012"
)
_DATA_FILE = "nist_isotope_masses.csv"


@lru_cache(maxsize=1)
def _symbols() -> dict[int, str]:
    text = (
        resources.files(__package__)
        .joinpath("data")
        .joinpath(_DATA_FILE)
        .read_text(encoding="utf-8")
    )
    rows = csv.DictReader(line for line in text.splitlines() if not line.startswith("#"))
    out: dict[int, str] = {}
    for row in rows:
        out.setdefault(int(row["Z"]), row["symbol"])  # first row: H, not D/T
    return out


@lru_cache(maxsize=1)
def _table() -> dict[tuple[int, int], tuple[float, float | None]]:
    text = (
        resources.files(__package__)
        .joinpath("data")
        .joinpath(_DATA_FILE)
        .read_text(encoding="utf-8")
    )
    rows = csv.DictReader(line for line in text.splitlines() if not line.startswith("#"))
    out: dict[tuple[int, int], tuple[float, float | None]] = {}
    for row in rows:
        # Compositions are printed with their uncertainty, e.g. "0.999885(70)".
        composition = re.match(r"[0-9.]+", row["isotopic_composition"])
        out[(int(row["Z"]), int(row["A"]))] = (
            float(row["relative_atomic_mass_u"]),
            float(composition.group()) if composition else None,
        )
    return out


def atomic_mass_u(Z: int, A: int) -> float:
    """Relative atomic mass of the neutral atom ``(Z, A)`` [u]."""
    try:
        return _table()[(Z, A)][0]
    except KeyError:
        raise KeyError(f"no isotope mass for Z={Z} A={A} (shipped: Z 1-92)") from None


def nuclear_mass_MeV(Z: int, A: int) -> float:
    """Rest energy of the bare nucleus ``(Z, A)`` [MeV]: atomic mass minus Z electrons."""
    return atomic_mass_u(Z, A) * ATOMIC_MASS_UNIT_MEV - Z * ELECTRON_MASS_MEV


def element_symbol(Z: int) -> str:
    """Chemical symbol of ``Z`` as listed by NIST (Z 1-92)."""
    try:
        return _symbols()[Z]
    except KeyError:
        raise KeyError(f"no element symbol for Z={Z} (shipped: Z 1-92)") from None


@cache  # a pure function of Z, hit on every LET lookup
def most_abundant_mass_number(Z: int) -> int:
    """Mass number of the most abundant stable isotope of ``Z`` (NIST compositions)."""
    candidates = [(c, a) for (z, a), (_, c) in _table().items() if z == Z and c is not None]
    if not candidates:
        raise KeyError(f"no isotopic composition for Z={Z}")
    return max(candidates)[1]


@cache  # a pure function of Z, hit on every LET lookup
def mean_atomic_mass_u(Z: int) -> float:
    """Mean atomic mass of element ``Z`` in its natural isotopic composition [u].

    The composition-weighted mean of NIST's isotope masses (normalised by the sum of the
    printed compositions); it equals the conventional atomic weight to the precision of
    the compositions (H: 1.00794, O: 15.9994).
    """
    rows = [(m, c) for (z, _), (m, c) in _table().items() if z == Z and c is not None]
    if not rows:
        raise KeyError(f"no isotopic composition for Z={Z}")
    return sum(m * c for m, c in rows) / sum(c for _, c in rows)


__all__ = [
    "MASS_SOURCE",
    "atomic_mass_u",
    "element_symbol",
    "mean_atomic_mass_u",
    "most_abundant_mass_number",
    "nuclear_mass_MeV",
]
