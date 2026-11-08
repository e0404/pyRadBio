"""Shared machinery of the classes backed by our shipped regridded tables (XSTAR, ICRU).

* each ion is served by a *reference* table -- one (Z, A) stored in a shipped file;
  an ion with another mass is looked up at the same velocity (equal Lorentz factor,
  true nuclear masses, [`masses`][]);
* an optional user [`LogEnergyGrid`][] re-holds every
  table on its own nodes;
* ``quantity="total"`` adds a source's own nuclear column for the exact isotope it was
  tabulated for; nuclear stopping depends on the projectile mass, not just its velocity,
  so every other ion gets ``nuclear``
  ([`WaterNuclearStopping`][]; default: ICRU 49's
  own method, Moliere protons and ZBL otherwise -- what ICRU 49/90 did where they
  tabulate nuclear stopping).
"""

from __future__ import annotations

from abc import abstractmethod
from dataclasses import dataclass
from functools import cache
from importlib import resources
from typing import Literal, get_args

import numpy as np
import numpy.typing as npt

from pyradbio.core.constants import WATER_DENSITY_G_CM3
from pyradbio.track.stopping.base import IonSourceInfo, MassStoppingPowerModel, OptionValue
from pyradbio.track.stopping.grid import (
    Interpolation,
    LogEnergyGrid,
    RegriddedTable,
    TableFile,
    read_table_csv,
)
from pyradbio.track.stopping.masses import nuclear_mass_MeV
from pyradbio.track.stopping.nuclear import ICRU49_PROTON_POTENTIAL, WaterNuclearStopping

Quantity = Literal["electronic", "total"]
Kind = Literal["electronic", "nuclear"]
_COLUMNS: dict[str, str] = {"electronic": "electronic_MeV_cm2_g", "nuclear": "nuclear_MeV_cm2_g"}


@cache
def shipped_table_file(stem: str) -> TableFile:
    """A shipped table ``data/<stem>.csv`` (cached)."""
    return read_table_csv(resources.files(__package__).joinpath("data").joinpath(f"{stem}.csv"))


@dataclass(frozen=True)
class TableRef:
    """The shipped table that serves an ion: file stem, and the (Z, A) stored in it."""

    stem: str
    Z: int
    A: int
    label: str


class ShippedTablesModel(MassStoppingPowerModel):
    """Base of XSTAR and ICRU; subclasses choose the reference table per ion."""

    def __init__(
        self,
        *,
        quantity: Quantity,
        grid: LogEnergyGrid | None,
        interpolation: Interpolation,
        density_g_cm3: float = WATER_DENSITY_G_CM3,
        nuclear: WaterNuclearStopping | None = None,
    ) -> None:
        if quantity not in get_args(Quantity):
            raise ValueError(f"quantity must be one of {get_args(Quantity)}, got {quantity!r}")
        self.quantity = quantity
        self.nuclear = nuclear or WaterNuclearStopping(proton_potential=ICRU49_PROTON_POTENTIAL)
        self.grid = grid
        self.interpolation = interpolation
        self.density_g_cm3 = density_g_cm3
        self._tables: dict[tuple[TableRef, str], RegriddedTable] = {}

    @abstractmethod
    def _reference(self, Z: int, A: int) -> TableRef:
        """The reference table for the ion (may raise for unavailable sources)."""

    # --- tables ------------------------------------------------------------------------

    def _table(self, ref: TableRef, kind: Kind) -> RegriddedTable:
        """``ref``'s column ``kind``, on the user grid if one is set."""
        key = (ref, kind)
        if key not in self._tables:
            ion = shipped_table_file(ref.stem).table(ref.Z, ref.A)
            table = RegriddedTable(
                ion.energy_MeV_u, ion.columns[_COLUMNS[kind]], self.interpolation
            )
            if self.grid is not None:
                table = RegriddedTable.from_source(
                    table.energy_MeV_u, table.values, self.grid, self.interpolation
                )
            self._tables[key] = table
        return self._tables[key]

    def _has_nuclear(self, ref: TableRef) -> bool:
        return _COLUMNS["nuclear"] in shipped_table_file(ref.stem).table(ref.Z, ref.A).columns

    def _shipped_info(self, ref: TableRef) -> IonSourceInfo:
        return shipped_table_file(ref.stem).table(ref.Z, ref.A).info

    # --- equal-velocity mapping --------------------------------------------------------

    @staticmethod
    def _to_reference(
        Z: int, A: int, energy: npt.NDArray[np.float64], ref: TableRef
    ) -> npt.NDArray[np.float64]:
        """Energy per nucleon of the reference ion at the same velocity (equal gamma)."""
        if (Z, A) == (ref.Z, ref.A):
            return energy
        gamma_minus_1 = energy * A / nuclear_mass_MeV(Z, A)
        out: npt.NDArray[np.float64] = gamma_minus_1 * nuclear_mass_MeV(ref.Z, ref.A) / ref.A
        return out

    @staticmethod
    def _from_reference(Z: int, A: int, energy_ref: float, ref: TableRef) -> float:
        if (Z, A) == (ref.Z, ref.A):
            return energy_ref
        gamma_minus_1 = energy_ref * ref.A / nuclear_mass_MeV(ref.Z, ref.A)
        return gamma_minus_1 * nuclear_mass_MeV(Z, A) / A

    def _lookup(
        self, Z: int, A: int, energy: npt.NDArray[np.float64], ref: TableRef, kind: Kind
    ) -> npt.NDArray[np.float64]:
        table = self._table(ref, kind)
        # The range check ran on the ion's own energies; mapping them to the reference
        # can move an endpoint by an ulp, which is clipped (not extrapolated).
        return table(np.clip(self._to_reference(Z, A, energy, ref), *table.energy_range))

    # --- MassStoppingPowerModel ----------------------------------------------------------

    def energy_range(self, Z: int, A: int) -> tuple[float, float]:
        self._require_supported(Z)
        ref = self._reference(Z, A)
        lo, hi = self._table(ref, "electronic").energy_range
        return self._from_reference(Z, A, lo, ref), self._from_reference(Z, A, hi, ref)

    def _electronic(
        self, Z: int, A: int, energy: npt.NDArray[np.float64], ref: TableRef
    ) -> npt.NDArray[np.float64]:
        """Electronic stopping; subclasses may rescale the reference (e.g. by z_eff)."""
        return self._lookup(Z, A, energy, ref, "electronic")

    def _evaluate(
        self, Z: int, A: int, energy_MeV_u: npt.NDArray[np.float64]
    ) -> npt.NDArray[np.float64]:
        ref = self._reference(Z, A)
        stopping = self._electronic(Z, A, energy_MeV_u, ref)
        if self.quantity == "total":
            stopping = stopping + self._nuclear(Z, A, energy_MeV_u, ref)
        return stopping

    def _tabulated_nuclear(self, Z: int, A: int, ref: TableRef) -> bool:
        return (Z, A) == (ref.Z, ref.A) and self._has_nuclear(ref)

    def _nuclear(
        self, Z: int, A: int, energy: npt.NDArray[np.float64], ref: TableRef
    ) -> npt.NDArray[np.float64]:
        if self._tabulated_nuclear(Z, A, ref):
            return self._lookup(Z, A, energy, ref, "nuclear")
        return self.nuclear.mass_stopping_power(Z, A, energy)

    def _common_options(self, Z: int, A: int, ref: TableRef) -> dict[str, OptionValue]:
        options: dict[str, OptionValue] = {
            "quantity": self.quantity,
            "interpolation": self.interpolation,
            "nuclear": (
                None
                if self.quantity == "electronic"
                else f"tabulated ({ref.label})"
                if self._tabulated_nuclear(Z, A, ref)
                else self.nuclear.describe()
            ),
            "grid": None if self.grid is None else self.grid.to_dict(),
        }
        if (Z, A) != (ref.Z, ref.A):
            options["velocity_scaled_from"] = ref.label
        return options


__all__ = ["Kind", "Quantity", "ShippedTablesModel", "TableRef", "shipped_table_file"]
