"""Python port of MSTAR 3.12 by Helmut Paul and Andreas Schinner, for liquid water.

MSTAR (program system MORESTAR, V 3.12, 24-05-2004; written by Andreas Schinner and
Helmut Paul; IAEA Nuclear Data Section, https://www-nds.iaea.org/stopping/mstar) is
released by its authors as freeware: "you can freely use, copy, modify, and/or
distribute the source code. However, please do not remove the authors' names from
the source code comments." This port keeps them in every module.

This file is under the MSTAR notice in ``THIRD_PARTY_NOTICES.md`` (AGENTS.md rule 3),
not pyRadBio's MIT licence.

References: H. Paul and A. Schinner, Nucl. Instrum. Methods Phys. Res. B 179 (2001)
299; B 195 (2002) 166; Atomic Data Nucl. Data Tables 85 (2003) 377.

* [`spline`][] -- MSSPL1/MSSPL2 (natural cubic spline, linear axes);
* [`database`][] -- MSRDB1 (reader for the alpha database ``msdbs1.d``);
* [`paul`][] -- MSPAUL and MSTAR1 (Paul's ratio, mode handling);
* [`AlphaSpline`][] -- MSSEAL (alpha stopping power of one target);
* [`MSTAR`][] -- the port as a [`MassStoppingPowerModel`][],
  evaluated directly from a local ``msdbs1.d`` (the database is raw source data and
  is not shipped; see ``data/stopping/mstar/PROVENANCE.md``).

[`XSTAR`][] serves MSTAR heavy ions from tables
this port generated (``scripts/stopping/build_tables.py``), and evaluates the port
directly only on request.

Conventions: MSTAR's energy is kinetic energy per nucleon and it evaluates the alpha
at exactly 4x that energy; it ignores the ion mass. We take its value as that of the
element's most abundant isotope and velocity-scale other isotopes with true masses,
as everywhere in [`pyradbio.track.stopping`][].
"""

from __future__ import annotations

from pathlib import Path
from typing import ClassVar, Literal

import numpy as np
import numpy.typing as npt

from pyradbio.core.constants import WATER_DENSITY_G_CM3
from pyradbio.track.stopping.base import (
    IonSourceInfo,
    MassStoppingPowerModel,
    OutOfRangeError,
)
from pyradbio.track.stopping.masses import most_abundant_mass_number, nuclear_mass_MeV
from pyradbio.track.stopping.mstar.database import AlphaTable, MSTARDatabase, MSTARDatabaseError
from pyradbio.track.stopping.mstar.paul import (
    MODE_H_IONS,
    MODES,
    WATER_LIQUID_ID,
    MSTARModeError,
    mspaul,
    mstar1,
    resolve_mode,
)
from pyradbio.track.stopping.mstar.spline import msspl1, msspl2
from pyradbio.track.stopping.nuclear import WaterNuclearStopping

MSTAR_I_VALUE_EV = 75.0
"""MSTAR's water I-value is that of its alpha data, ICRU 49 / ASTAR [eV]."""
MSTAR_REFERENCE = (
    "H. Paul and A. Schinner, MSTAR 3.12 (IAEA NDS); Nucl. Instrum. Methods B 179 (2001) "
    "299, B 195 (2002) 166; At. Data Nucl. Data Tables 85 (2003) 377"
)
_PER_MG_TO_PER_G = 1000.0
"""mg -> g: MSTAR returns MeV cm^2/mg."""


class AlphaSpline:
    """MSSEAL: one target's alpha stopping power [MeV cm^2/mg] at total energy [MeV]."""

    def __init__(self, table: AlphaTable) -> None:
        self.table = table
        self._x = list(table.energy_MeV)
        self._y = list(table.stopping_MeV_cm2_mg)
        self._ypp = msspl1(self._x, self._y)

    @property
    def energy_range_MeV(self) -> tuple[float, float]:
        return self._x[0], self._x[-1]

    def __call__(self, energy_MeV: float) -> float:
        lo, hi = self.energy_range_MeV
        if energy_MeV < lo or energy_MeV > hi:  # MSSEAL's RC=30
            raise OutOfRangeError((energy_MeV, energy_MeV), (lo, hi), 2, 4, "MSTAR alpha")
        return msspl2(self._x, self._y, self._ypp, energy_MeV)


def mstar_stopping(
    alpha: AlphaSpline, mode: str, z1: int, energy_MeV_u: float
) -> tuple[float, str]:
    """The port of MSTAR1: ``(stopping power [MeV cm^2/mg], mode actually used)``."""
    return mstar1(alpha(4.0 * energy_MeV_u), mode, z1, energy_MeV_u)


class MSTAR(MassStoppingPowerModel):
    """MSTAR 3.12 for liquid water, evaluated directly from ``msdbs1.d`` (Z 3-18).

    MSTAR itself is electronic stopping only; ``quantity="total"`` adds ``nuclear``
    ([`WaterNuclearStopping`][], default ZBL).
    ``mode`` is MSTAR's mode of operation ('b' default, the authors' recommendation).
    Raises [`MSTARModeError`][] where the original refuses a mode (mode 'h' for Mg,
    Al, Si, P).
    """

    name: ClassVar[str] = "mstar"

    def __init__(
        self,
        database: str | Path | MSTARDatabase,
        *,
        mode: str = "b",
        quantity: Literal["electronic", "total"] = "electronic",
        nuclear: WaterNuclearStopping | None = None,
        density_g_cm3: float = WATER_DENSITY_G_CM3,
    ) -> None:
        resolve_mode(mode)  # validates
        if quantity not in ("electronic", "total"):
            raise ValueError(f"quantity must be 'electronic' or 'total', got {quantity!r}")
        self.quantity = quantity
        self.nuclear = nuclear or WaterNuclearStopping()
        self.mode = mode
        self.density_g_cm3 = density_g_cm3
        db = database if isinstance(database, MSTARDatabase) else MSTARDatabase.from_file(database)
        self.alpha = AlphaSpline(db.target(WATER_LIQUID_ID))

    def supported_Z(self) -> tuple[int, ...]:
        return tuple(range(3, 19))

    def _scale(self, Z: int, A: int) -> tuple[int, float]:
        """(reference A, ion-to-reference energy factor at equal velocity)."""
        a_ref = most_abundant_mass_number(Z)
        if A == a_ref:
            return a_ref, 1.0
        return a_ref, (nuclear_mass_MeV(Z, a_ref) / a_ref) / (nuclear_mass_MeV(Z, A) / A)

    def energy_range(self, Z: int, A: int) -> tuple[float, float]:
        self._require_supported(Z)
        _, factor = self._scale(Z, A)
        lo, hi = self.alpha.energy_range_MeV
        return lo / 4.0 / factor, hi / 4.0 / factor

    def mode_used(self, Z: int, energy_MeV_u: float) -> str:
        """The mode MSTAR1 actually applies to ion ``Z`` (after 'a'/'b' resolution)."""
        return mstar1(1.0, self.mode, Z, energy_MeV_u)[1]

    def _evaluate(
        self, Z: int, A: int, energy_MeV_u: npt.NDArray[np.float64]
    ) -> npt.NDArray[np.float64]:
        _, factor = self._scale(Z, A)
        lo, hi = self.alpha.energy_range_MeV
        out = np.empty_like(energy_MeV_u)
        for i, e in enumerate(energy_MeV_u):
            e_ref = min(max(float(e) * factor, lo / 4.0), hi / 4.0)  # ulp clip only
            out[i] = mstar_stopping(self.alpha, self.mode, Z, e_ref)[0] * _PER_MG_TO_PER_G
        if self.quantity == "total":
            out = out + self.nuclear.mass_stopping_power(Z, A, energy_MeV_u)
        return out

    def metadata(self, Z: int, A: int) -> IonSourceInfo:
        lo, hi = self.energy_range(Z, A)
        a_ref, _ = self._scale(Z, A)
        options: dict[str, str | float | int | bool | None] = {
            "quantity": self.quantity,
            "nuclear": None if self.quantity == "electronic" else self.nuclear.describe(),
            "mstar_mode": self.mode,
            "evaluation": "direct",
        }
        if A != a_ref:
            options["velocity_scaled_from"] = f"A={a_ref}"
        return IonSourceInfo(
            Z=Z,
            A=A,
            source="MSTAR 3.12",
            reference=MSTAR_REFERENCE,
            i_value_eV=MSTAR_I_VALUE_EV,
            energy_range_MeV_u=(lo, hi),
            edition="3.12",
            options=options,
        )


__all__ = [
    "MODES",
    "MODE_H_IONS",
    "MSTAR",
    "MSTAR_I_VALUE_EV",
    "MSTAR_REFERENCE",
    "AlphaSpline",
    "MSTARDatabase",
    "MSTARDatabaseError",
    "MSTARModeError",
    "mspaul",
    "mstar_stopping",
    "resolve_mode",
]
