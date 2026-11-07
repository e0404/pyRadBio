"""Isotope masses shipped from the NIST Atomic Weights and Isotopic Compositions table."""

from __future__ import annotations

import csv
from collections.abc import Callable
from pathlib import Path

import pytest

from pyradbio.core.constants import ATOMIC_MASS_UNIT_MEV, ELECTRON_MASS_MEV
from pyradbio.track.stopping.masses import (
    MASS_SOURCE,
    atomic_mass_u,
    element_symbol,
    most_abundant_mass_number,
    nuclear_mass_MeV,
)


def test_nuclear_mass_is_atomic_mass_minus_the_electrons() -> None:
    for z, a in ((1, 1), (2, 4), (6, 12), (18, 40)):
        expected = atomic_mass_u(z, a) * ATOMIC_MASS_UNIT_MEV - z * ELECTRON_MASS_MEV
        assert nuclear_mass_MeV(z, a) == expected


def test_carbon_12_defines_the_mass_unit() -> None:
    assert atomic_mass_u(6, 12) == 12.0


def test_unknown_isotopes_raise() -> None:
    with pytest.raises(KeyError, match="Z=6 A=40"):
        atomic_mass_u(6, 40)
    with pytest.raises(KeyError, match="Z=93"):
        atomic_mass_u(93, 237)


def test_most_abundant_isotope_matches_the_package_defaults_for_light_ions() -> None:
    from pyradbio.core.elements import MOST_ABUNDANT_A

    for z, a in MOST_ABUNDANT_A.items():
        assert most_abundant_mass_number(z) == a
    assert {z: most_abundant_mass_number(z) for z in (11, 12, 13, 14, 15, 16, 17, 18)}


def test_mass_source_is_recorded() -> None:
    assert "AME2012" in MASS_SOURCE
    assert "NIST" in MASS_SOURCE


def test_shipped_masses_reproduce_the_raw_nist_table(
    raw_source_file: Callable[[str, str], Path],
) -> None:
    """Every shipped (Z, A) mass equals the raw NIST value as printed (Z 1-18)."""
    raw = raw_source_file("nist_isotopes", "nist_isotopic_compositions.csv")
    rows = list(csv.DictReader(line for line in raw.open() if not line.startswith("#")))
    checked = 0
    for row in rows:
        z, a = int(row["Z"]), int(row["A"])
        if z > 18:  # D and T are listed as Z=1, A=2/3 and are kept
            continue
        assert atomic_mass_u(z, a) == float(row["relative_atomic_mass"]), (z, a)
        checked += 1
    assert checked > 200


def test_most_abundant_needs_a_composition() -> None:
    with pytest.raises(KeyError, match="Z=93"):
        most_abundant_mass_number(93)


def test_element_symbols() -> None:
    assert [element_symbol(z) for z in (1, 6, 18)] == ["H", "C", "Ar"]
    with pytest.raises(KeyError, match="Z=93"):
        element_symbol(93)


def test_per_element_lookups_are_memoised() -> None:
    """The default stopping power asks for these on every LET lookup (ICRU90Procedure
    ._ref); each scans the whole isotope table, so they are cached per Z."""
    from pyradbio.track.stopping.masses import mean_atomic_mass_u

    for lookup in (most_abundant_mass_number, mean_atomic_mass_u):
        first = lookup(6)
        hits = lookup.cache_info().hits  # type: ignore[attr-defined]
        assert lookup(6) == first
        assert lookup.cache_info().hits == hits + 1  # type: ignore[attr-defined]
