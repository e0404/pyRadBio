"""Element symbol <-> atomic number for the ions used across pyRadBio.

Covers hydrogen through neon, the light-ion range of the LEM/MKM references and
the pyMKM stopping-power tables. Extend as heavier species are needed.
"""

from __future__ import annotations

from collections.abc import Mapping

SYMBOL_TO_Z: Mapping[str, int] = {
    "H": 1,
    "He": 2,
    "Li": 3,
    "Be": 4,
    "B": 5,
    "C": 6,
    "N": 7,
    "O": 8,
    "F": 9,
    "Ne": 10,
}
"""Element symbol -> atomic number Z (H..Ne)."""

Z_TO_SYMBOL: Mapping[int, str] = {z: sym for sym, z in SYMBOL_TO_Z.items()}
"""Atomic number Z -> element symbol (inverse of [`SYMBOL_TO_Z`][])."""

MOST_ABUNDANT_A: Mapping[int, int] = {
    1: 1,
    2: 4,
    3: 7,
    4: 9,
    5: 11,
    6: 12,
    7: 14,
    8: 16,
    9: 19,
    10: 20,
}
"""Most-abundant / therapy-relevant mass number A per Z (for Z-only inputs)."""


def default_mass_number(z: int) -> int:
    """Default mass number A for atomic number ``z`` (e.g. ``6 -> 12``)."""
    try:
        return MOST_ABUNDANT_A[z]
    except KeyError:
        raise ValueError(f"No default mass number for Z={z}") from None


def symbol_to_z(symbol: str) -> int:
    """Atomic number for an element ``symbol`` (e.g. ``"C" -> 6``)."""
    try:
        return SYMBOL_TO_Z[symbol]
    except KeyError:
        raise ValueError(f"Unknown element symbol {symbol!r}") from None


def z_to_symbol(z: int) -> str:
    """Element symbol for an atomic number ``z`` (e.g. ``6 -> "C"``)."""
    try:
        return Z_TO_SYMBOL[z]
    except KeyError:
        raise ValueError(f"No element symbol for Z={z}") from None
