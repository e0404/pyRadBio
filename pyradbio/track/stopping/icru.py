"""ICRU water stopping powers by edition: ICRU 49, 73 (+ errata) and 90.

``ICRU(edition=...)`` composes the ICRU reports per ion:

========  =========================  ==============================  ============
edition   p, He                      Li-Ar (Z 3-18)                  C
========  =========================  ==============================  ============
"2005"    ICRU 49 (PSTAR / ASTAR)    ICRU 73 as published (2005)     ICRU 73
"2009"    ICRU 49 (PSTAR / ASTAR)    ICRU 73 with its errata (2009)  ICRU 73 err.
"2016"    ICRU 90                    ICRU 73 with its errata (2009)  ICRU 90
========  =========================  ==============================  ============

There is no I-value input: each source's own value is exposed **per ion** in
[`ICRU.metadata`][]: 75 eV for ICRU 49 (PSTAR/ASTAR),
78 eV for ICRU 90 and for the ICRU 73 errata (Sigmund, Schinner & Paul 2009, sec. 2.2
and Table 1), and 67.2 eV for ICRU 73 as published -- implied by its
oscillator-strength spectrum for liquid water, as the errata state; ICRU 73 itself
prints no water I-value.

Tables are our regridded copies in ``data/`` (50 log-uniform points per decade,
``scripts/stopping/build_tables.py``):

* ICRU 49 = the NIST PSTAR/ASTAR tables shared with
  [`XSTAR`][] (NIST's interactive databases
  serve ICRU 49; for liquid water they equal libdedx's transcription of the printed
  report at every shared proton energy and at all but two alpha energies below
  1 keV/u, which differ by < 0.1%);
* ICRU 73 / errata = libdedx's raw ICRU 73 files,
  verified to equal the printed tables -- ICRU 73 pp. 232-233 and the
  errata's Appendix B -- at all 848 points (``data/stopping/icru73/PROVENANCE.md``).
  Both are electronic stopping only (ICRU 73 sec. 6.1). They run from 0.025 to
  1000 MeV/u and raise below;
* ICRU 90 = NIST's ICRU 90 addendum (tables A.9, A.12, A.15).

ICRU 73 tabulates each element against E/A1, making "no distinction ... between
energy per nucleon and energy per atomic mass unit" (its footnote 6: at most 0.25%
from Li up); we store it as the element's most abundant isotope and velocity-scale
other isotopes with true masses,
like every other ion here. ``quantity="total"`` adds the tabulated nuclear column
for p, He-4 (ICRU 49, 90) and C-12 (ICRU 90). ICRU 73 has none (electronic only), so its
ions -- and every other isotope -- get ``nuclear``
([`WaterNuclearStopping`][], default: ICRU 49's
method, as ICRU 90 used for C).
"""

from __future__ import annotations

from typing import ClassVar, Literal, get_args

from pyradbio.core.constants import WATER_DENSITY_G_CM3
from pyradbio.track.stopping._tabulated import Quantity, ShippedTablesModel, TableRef
from pyradbio.track.stopping.base import IonSourceInfo
from pyradbio.track.stopping.grid import Interpolation, LogEnergyGrid
from pyradbio.track.stopping.masses import element_symbol, most_abundant_mass_number
from pyradbio.track.stopping.nuclear import WaterNuclearStopping
from pyradbio.track.stopping.xstar import ASTAR_REF, PSTAR_REF

Edition = Literal["2005", "2009", "2016"]

ICRU73_IONS = tuple(range(3, 19))
"""Ions covered by ICRU 73 (and its errata): Li to Ar."""
ICRU90_IONS = (1, 2, 6)
"""Ions ICRU 90 tabulates for water: p, He-4, C-12."""

_ICRU90_REFS = {
    1: TableRef("icru90_water_liquid", 1, 1, "p"),
    2: TableRef("icru90_water_liquid", 2, 4, "He-4"),
    6: TableRef("icru90_water_liquid", 6, 12, "C-12"),
}


def _icru73_ref(Z: int, errata: bool) -> TableRef:
    a = most_abundant_mass_number(Z)
    stem = "icru73_2009_water_liquid" if errata else "icru73_2005_water_liquid"
    return TableRef(stem, Z, a, f"{element_symbol(Z)}-{a}")


class ICRU(ShippedTablesModel):
    """ICRU 49 / 73 / 90 water stopping power for ``edition`` "2005", "2009" or "2016"."""

    name: ClassVar[str] = "icru"

    def __init__(
        self,
        *,
        edition: Edition = "2016",
        quantity: Quantity = "electronic",
        grid: LogEnergyGrid | None = None,
        interpolation: Interpolation = "cubic",
        density_g_cm3: float = WATER_DENSITY_G_CM3,
        nuclear: WaterNuclearStopping | None = None,
    ) -> None:
        if edition not in get_args(Edition):
            raise ValueError(f"edition must be one of {get_args(Edition)}, got {edition!r}")
        super().__init__(
            quantity=quantity,
            grid=grid,
            interpolation=interpolation,
            density_g_cm3=density_g_cm3,
            nuclear=nuclear,
        )
        self.edition = edition

    def supported_Z(self) -> tuple[int, ...]:
        return tuple(range(1, 19))

    def _reference(self, Z: int, A: int) -> TableRef:
        if self.edition == "2016" and Z in _ICRU90_REFS:
            return _ICRU90_REFS[Z]
        if Z == 1:
            return PSTAR_REF
        if Z == 2:
            return ASTAR_REF
        return _icru73_ref(Z, errata=self.edition != "2005")

    def metadata(self, Z: int, A: int) -> IonSourceInfo:
        lo, hi = self.energy_range(Z, A)
        ref = self._reference(Z, A)
        shipped = self._shipped_info(ref)
        options = {**shipped.options, **self._common_options(Z, A, ref), "edition": self.edition}
        return IonSourceInfo(
            Z=Z,
            A=A,
            source=shipped.source,
            reference=shipped.reference,
            i_value_eV=shipped.i_value_eV,
            energy_range_MeV_u=(lo, hi),
            edition=shipped.edition,
            options=options,
        )


__all__ = ["ICRU", "ICRU73_IONS", "ICRU90_IONS", "Edition"]
