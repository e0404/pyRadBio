"""Northcliffe & Schilling 1970 water stopping-power table (paper-era source).

Scholz & Kraft normalise the LEM radial dose profile so that its integral
reproduces "the LET as given by the tables of Hubert et al. and Northcliffe and
Schilling" (Scholz & Kraft 1996, eq. 9 discussion; their refs. 17/18). Of those
two, **only Northcliffe & Schilling tabulates water**: Hubert/Bimbot/Gauvin
covers 36 solid *elements* and starts at Z >= 2, E >= 2.5 MeV/u, so it cannot
supply a water LET for a proton, or for anything, at all. Reconstructions of the
1996 figures that want the paper's own LET therefore use this table.

Scope and honest limits:

* **All 103 ions** the source tabulates (``1 <= Z <= 103``) on its fixed
  38-point grid, ``0.0125 <= E <= 12 MeV/u``.
* 60 columns were read off the page scans and cross-checked against an
  independent OCR pass of the PDF text layer; the other 43 come from that OCR
  pass and passed both physical screens (see
  ``docs/references/NorthcliffeSchilling1970.md``).
  :attr:`WaterStoppingPowerColumn.source` records which.
* The grid **stops at 12 MeV/u**, while Scholz & Kraft 1996 Fig. 4 runs to
  20 MeV/u. :class:`~pyradbio.track.stopping_power.StoppingPowerTable` raises
  rather than extrapolating, so above 12 MeV/u use
  :class:`~pyradbio.track.stopping_power.SimpleBetheStoppingPower` (the library
  default) and say so; do not silently extend this table.
* It is a pre-ICRU-49 compilation. Its proton column matches NIST PSTAR for
  liquid water to 1.2% for ``E >= 5 MeV/u``, then runs **high** below that:
  +6% at 2 MeV/u, +18% at 1 MeV/u, +29% at 0.5 MeV/u. The Bragg peak sits at
  0.25 MeV/u, where modern data put it near 0.08 MeV/u, and below ~0.15 MeV/u
  the column falls *under* PSTAR, to about half at 0.05 MeV/u. He and C show
  the same excess (+29% and +25% over pyMKM's MSTAR 3.12 table at 0.5 MeV/u).
  **This is the printed table, not a transcription error.** The proton, He and
  C water columns were re-read against the page images (pp. 254, 256, 264) and
  match cell for cell. NS1970 builds water by Bragg additivity from its
  *gaseous* H and O curves (their footnote 31, p. 250), and the printed
  proton-in-water column is that sum of their printed H and O columns to within
  rounding. The authors also say they chose "to ignore the poor fits to proton
  stopping power at low energy since the main objective of this tabulation was
  to cover heavier ions" (Sec. VIII, p. 249). Use it to reproduce the 1996
  papers, not as a modern reference.

Reference: Northcliffe, L.C. & Schilling, R.F. 1970, Nuclear Data Tables A7,
233-463, doi:10.1016/S0092-640X(70)80016-X.
"""

from __future__ import annotations

import csv
from dataclasses import dataclass
from functools import lru_cache
from importlib import resources

from pyradbio.core.constants import MEV_PER_CM_TO_KEV_PER_UM, WATER_DENSITY_G_CM3
from pyradbio.core.radiation import Ion
from pyradbio.track.stopping_power import (
    SimpleBetheStoppingPower,
    TabularStoppingPower,
)

NORTHCLIFFE_SCHILLING1970_DOI = "10.1016/S0092-640X(70)80016-X"
_DATA_FILE = "northcliffe_schilling1970_water.csv"

#: Energy range the source actually tabulates [MeV/u]. Outside this the loader's
#: table raises; see the module docstring for what to fall back to.
ENERGY_RANGE_MEV_U = (0.0125, 12.0)


@dataclass(frozen=True, slots=True)
class WaterStoppingPowerColumn:
    """One ion's water column exactly as printed, in mass-stopping-power units."""

    Z: int
    symbol: str
    energy_MeV_u: tuple[float, ...]
    mass_stopping_power_MeV_cm2_g: tuple[float, ...]
    source: str  # "scan" (read off the page) or "ocr" (text layer, screened)

    def let_keV_um(self, density_g_cm3: float = WATER_DENSITY_G_CM3) -> tuple[float, ...]:
        """Convert the tabulated mass stopping power to LET."""
        scale = density_g_cm3 * MEV_PER_CM_TO_KEV_PER_UM
        return tuple(value * scale for value in self.mass_stopping_power_MeV_cm2_g)


@lru_cache(maxsize=1)
def load_water_columns() -> dict[int, WaterStoppingPowerColumn]:
    """Return the digitized water columns keyed by atomic number."""
    data_dir = resources.files(__package__).joinpath("data")
    text = data_dir.joinpath(_DATA_FILE).read_text(encoding="utf-8")
    rows = [line for line in text.splitlines() if line and not line.startswith("#")]
    grouped: dict[int, tuple[str, str, list[float], list[float]]] = {}
    for row in csv.DictReader(rows):
        z = int(row["Z"])
        _symbol, _source, energies, values = grouped.setdefault(
            z, (row["symbol"], row["source"], [], [])
        )
        energies.append(float(row["energy_MeV_u"]))
        values.append(float(row["mass_stopping_power_MeV_cm2_g"]))
    return {
        z: WaterStoppingPowerColumn(
            Z=z,
            symbol=symbol,
            energy_MeV_u=tuple(energies),
            mass_stopping_power_MeV_cm2_g=tuple(values),
            source=source,
        )
        for z, (symbol, source, energies, values) in sorted(grouped.items())
    }


def northcliffe_schilling1970_water(
    density_g_cm3: float = WATER_DENSITY_G_CM3,
) -> TabularStoppingPower:
    """Return the 1970 water table as a :class:`StoppingPower` implementation.

    Interpolates log-log in energy, and raises outside ``ENERGY_RANGE_MEV_U``
    rather than extrapolating a 12 MeV/u table into the tens of MeV/u.
    """
    columns = load_water_columns()
    return TabularStoppingPower.from_arrays(
        {
            z: (column.energy_MeV_u, column.let_keV_um(density_g_cm3))
            for z, column in columns.items()
        }
    )


class NorthcliffeSchillingWithBetheTail:
    """The 1970 water table inside its grid, analytic Bethe above 12 MeV/u.

    A hybrid is unavoidable -- the source stops at 12 MeV/u and the Scholz &
    Kraft figures run well past it -- so the point is to make the seam visible
    rather than to hide it. :attr:`fallback_energies` records every energy that
    had to leave the table and :attr:`table_hits` counts those that did not, so a
    caller can report the tally instead of guessing.

    This lives in the library because *every* consumer must agree on it. LET
    enters both the x axis and the cross-section conversion, so a reconstruction
    whose plotted curve and whose Monte-Carlo checkpoint were built from
    different LETs is not merely imprecise, it is incoherent -- and the two
    differ by 13% in alpha at carbon 1 MeV/u, which is visible on the figure.
    :attr:`name` is what callers stamp into a checkpoint to keep that honest.
    """

    name = "ns1970+bethe"

    def __init__(self, density_g_cm3: float = WATER_DENSITY_G_CM3) -> None:
        self._table = northcliffe_schilling1970_water(density_g_cm3)
        self._bethe = SimpleBetheStoppingPower()
        self.table_hits = 0
        self.fallback_energies: set[float] = set()

    def let_keV_um(self, ion: Ion) -> float:
        try:
            value = self._table.let_keV_um(ion)
        except (KeyError, ValueError):
            self.fallback_energies.add(round(ion.energy_MeV_u, 4))
            return self._bethe.let_keV_um(ion)
        self.table_hits += 1
        return value


class BetheOnly(SimpleBetheStoppingPower):
    """Analytic Bethe everywhere, named so a checkpoint can record the choice."""

    name = "bethe"


def build_stopping_power(choice: str) -> NorthcliffeSchillingWithBetheTail | BetheOnly:
    """Map a ``ns1970|bethe`` CLI choice onto a named stopping power."""
    if choice == "ns1970":
        return NorthcliffeSchillingWithBetheTail()
    if choice == "bethe":
        return BetheOnly()
    raise ValueError(f"unknown stopping power {choice!r}; use ns1970|bethe")


__all__ = [
    "ENERGY_RANGE_MEV_U",
    "NORTHCLIFFE_SCHILLING1970_DOI",
    "BetheOnly",
    "NorthcliffeSchillingWithBetheTail",
    "WaterStoppingPowerColumn",
    "build_stopping_power",
    "load_water_columns",
    "northcliffe_schilling1970_water",
]
