"""XSTAR: NIST PSTAR (protons), ASTAR (helium) and MSTAR (Li-Ar) for liquid water.

Protons come from PSTAR and helium from ASTAR (NIST SRD 124, i.e. the ICRU 49
tables, I = 75 eV); ions with Z 3-18 come from MSTAR 3.12 (Paul & Schinner, built on
ASTAR) or, as an alternative, from PSTAR scaled by an effective charge. The mean
excitation energy is fixed at **75 eV** -- it is a property of these tables, not an
option.

Tables are our regridded copies in ``data/`` (50 log-uniform points per decade,
written by ``scripts/stopping/build_tables.py``), interpolated with a not-a-knot
cubic in log-log space; a user [`LogEnergyGrid`][]
re-holds them on its own nodes. Nothing is extrapolated: outside a source's range
[`OutOfRangeError`][] is raised (ASTAR ends at
1000 MeV total = 250 MeV/u).

Behaviour:

* **Isotopes** are velocity-scaled with true nuclear masses
  ([`masses`][]): an ion is looked up in its reference
  particle's table (p for Z = 1 and Z >= 3, He-4 for Z = 2) at the same Lorentz
  factor. For p and He-4 themselves the lookup is the identity.
* ``heavy_ion_source="pstar_zeff_scaled"`` covers Z 3-18 only; helium always
  uses ASTAR. It multiplies PSTAR at equal velocity by ``(z_eff/z_eff,p)^2``
  (``zeff_scaling="ratio"``, the default: PSTAR's protons themselves carry a
  reduced charge at low velocity) or by ``z_eff^2`` (``"absolute"``), with ``z_eff``
  from a pluggable [`EffectiveChargeModel`][]
  (Barkas by default). Measured against ICRU 73 with errata (water,
  Z 3-18, equal velocity): from 10 MeV/u up both forms agree to within 0.1-5%; at
  0.3-1 MeV/u ``ratio`` is within 3.5-7% and ``absolute`` 3-13.5% low; at
  0.05-0.1 MeV/u ``absolute`` is 14-52% low while ``ratio`` stays within 15% for
  Z <= 8 but overshoots heavier ions (up to +63% for Ar). ``ratio`` is the default;
  neither is accurate below ~0.3 MeV/u, hence MSTAR.
* ``quantity="total"`` adds PSTAR's / ASTAR's own nuclear column for p and He-4, and
  ``nuclear`` ([`WaterNuclearStopping`][], default
  ICRU 49's method) for every other ion.
* ``heavy_ion_source="mstar"`` (the default) serves Z 3-18 from our Python port of
  MSTAR 3.12 ([`mstar`][], which reproduces the authors'
  compiled FORTRAN to <= 2e-15). By default (``mstar_evaluation="interpolate"``)
  it reads the shipped tables -- the port's exact values at 50 nodes per decade --
  and interpolates between them; ``mstar_evaluation="direct"`` evaluates the port at
  every requested energy from a local MSTAR database ``msdbs1.d`` (``mstar_database``
  or the ``PYRADBIO_MSTAR_DATABASE`` environment variable; the database is raw source
  data and is not shipped). Tables are shipped for MSTAR modes 'c' and 'd', i.e. for
  'a', 'b' (default), 'c', 'd' in liquid water; the gas modes 'g', 'h' need
  ``mstar_evaluation="direct"``. MSTAR has no nuclear stopping;
  ``quantity="total"`` adds ``nuclear``.
"""

from __future__ import annotations

import os
from pathlib import Path
from typing import ClassVar, Literal, get_args

import numpy as np
import numpy.typing as npt

from pyradbio.core.constants import WATER_DENSITY_G_CM3
from pyradbio.track.stopping._tabulated import (
    Kind,
    Quantity,
    ShippedTablesModel,
    TableRef,
)
from pyradbio.track.stopping.base import IonSourceInfo, OptionValue
from pyradbio.track.stopping.effective_charge import BarkasEffectiveCharge, EffectiveChargeModel
from pyradbio.track.stopping.grid import Interpolation, LogEnergyGrid, RegriddedTable
from pyradbio.track.stopping.masses import (
    element_symbol,
    most_abundant_mass_number,
    nuclear_mass_MeV,
)
from pyradbio.track.stopping.mstar import MSTAR, resolve_mode
from pyradbio.track.stopping.nuclear import WaterNuclearStopping

HeavyIonSource = Literal["mstar", "pstar_zeff_scaled"]
ZeffScaling = Literal["ratio", "absolute"]
MstarEvaluation = Literal["interpolate", "direct"]

MSTAR_DATABASE_ENV = "PYRADBIO_MSTAR_DATABASE"
"""Environment variable naming a local MSTAR database for ``mstar_evaluation="direct"``."""
_SHIPPED_MSTAR_MODES = ("c", "d")
"""MSTAR modes (as resolved for liquid water) with shipped tables."""

MSTAR_MODES = ("a", "b", "c", "d", "g", "h")
"""MSTAR's modes of operation (see data/stopping/mstar/PROVENANCE.md); 'b' is the
authors' recommendation: ion-specific fit where available, otherwise all ions.
"""

XSTAR_I_VALUE_EV = 75.0
"""Mean excitation energy of water in PSTAR/ASTAR/MSTAR (ICRU 49) [eV]; fixed."""

PSTAR_REF = TableRef("pstar_water_liquid", 1, 1, "p")
"""Reference tables: PSTAR protons and ASTAR He-4. Z >= 3 scale from the proton."""
ASTAR_REF = TableRef("astar_water_liquid", 2, 4, "He-4")


class XSTAR(ShippedTablesModel):
    """PSTAR / ASTAR / MSTAR water stopping power (ICRU 49 basis, I = 75 eV fixed)."""

    name: ClassVar[str] = "xstar"

    def __init__(
        self,
        *,
        heavy_ion_source: HeavyIonSource = "mstar",
        zeff_model: EffectiveChargeModel | None = None,
        zeff_scaling: ZeffScaling = "ratio",
        mstar_mode: str = "b",
        mstar_evaluation: MstarEvaluation = "interpolate",
        mstar_database: str | Path | None = None,
        quantity: Quantity = "electronic",
        grid: LogEnergyGrid | None = None,
        interpolation: Interpolation = "cubic",
        density_g_cm3: float = WATER_DENSITY_G_CM3,
        nuclear: WaterNuclearStopping | None = None,
    ) -> None:
        super().__init__(
            quantity=quantity,
            grid=grid,
            interpolation=interpolation,
            density_g_cm3=density_g_cm3,
            nuclear=nuclear,
        )
        if heavy_ion_source not in get_args(HeavyIonSource):
            raise ValueError(f"heavy_ion_source must be one of {get_args(HeavyIonSource)}")
        if zeff_scaling not in get_args(ZeffScaling):
            raise ValueError(f"zeff_scaling must be one of {get_args(ZeffScaling)}")
        if mstar_mode not in MSTAR_MODES:
            raise ValueError(f"mstar_mode must be one of {MSTAR_MODES}, got {mstar_mode!r}")
        if mstar_evaluation not in get_args(MstarEvaluation):
            raise ValueError(f"mstar_evaluation must be one of {get_args(MstarEvaluation)}")
        self.heavy_ion_source = heavy_ion_source
        self.zeff_model: EffectiveChargeModel = zeff_model or BarkasEffectiveCharge()
        self.zeff_scaling = zeff_scaling
        self.mstar_mode = mstar_mode
        self.mstar_evaluation = mstar_evaluation
        self.mstar_mode_used = resolve_mode(mstar_mode)[0]
        """The MSTAR mode liquid water actually uses ('a' -> 'c', 'b' -> 'd')."""
        self._mstar: MSTAR | None = None
        if heavy_ion_source == "mstar":
            if mstar_evaluation == "direct":
                path = mstar_database or os.environ.get(MSTAR_DATABASE_ENV)
                if not path:
                    raise FileNotFoundError(
                        "mstar_evaluation='direct' needs MSTAR's database msdbs1.d: pass "
                        f"mstar_database=... or set {MSTAR_DATABASE_ENV} (it is not shipped; "
                        "see data/stopping/mstar/PROVENANCE.md in the pyRadBio repository)"
                    )
                self._mstar = MSTAR(path, mode=mstar_mode, density_g_cm3=density_g_cm3)
            elif self.mstar_mode_used not in _SHIPPED_MSTAR_MODES:
                raise ValueError(
                    f"MSTAR mode {mstar_mode!r} (gas fit {self.mstar_mode_used!r}) has no "
                    "shipped table; use mstar_evaluation='direct' with an MSTAR database"
                )

    def supported_Z(self) -> tuple[int, ...]:
        return tuple(range(1, 19))

    def _reference(self, Z: int, A: int) -> TableRef:
        if Z == 2:
            return ASTAR_REF
        if Z >= 3 and self.heavy_ion_source == "mstar":
            a = most_abundant_mass_number(Z)
            stem = f"mstar_{self.mstar_mode_used}_water_liquid"
            return TableRef(stem, Z, a, f"{element_symbol(Z)}-{a}")
        return PSTAR_REF

    def _direct(self, Z: int) -> MSTAR | None:
        return self._mstar if Z >= 3 else None

    def energy_range(self, Z: int, A: int) -> tuple[float, float]:
        direct = self._direct(Z)
        if direct is not None:
            self._require_supported(Z)
            return direct.energy_range(Z, A)
        return super().energy_range(Z, A)

    def _evaluate(
        self, Z: int, A: int, energy_MeV_u: npt.NDArray[np.float64]
    ) -> npt.NDArray[np.float64]:
        direct = self._direct(Z)
        if direct is None:
            return super()._evaluate(Z, A, energy_MeV_u)
        stopping = direct._evaluate(Z, A, energy_MeV_u)
        if self.quantity == "total":
            stopping = stopping + self.nuclear.mass_stopping_power(Z, A, energy_MeV_u)
        return stopping

    def _reference_table(self, z_ref: int, kind: Kind) -> RegriddedTable:
        """The PSTAR (``z_ref=1``) or ASTAR (``z_ref=2``) table, on the user grid if set."""
        return self._table(ASTAR_REF if z_ref == 2 else PSTAR_REF, kind)

    def _electronic(
        self, Z: int, A: int, energy: npt.NDArray[np.float64], ref: TableRef
    ) -> npt.NDArray[np.float64]:
        stopping = self._lookup(Z, A, energy, ref, "electronic")
        if Z >= 3 and self.heavy_ion_source == "pstar_zeff_scaled":
            beta = np.sqrt(1.0 - 1.0 / (1.0 + energy * A / nuclear_mass_MeV(Z, A)) ** 2)
            z_eff = self.zeff_model.z_eff(Z, beta)
            if self.zeff_scaling == "ratio":
                z_eff = z_eff / self.zeff_model.z_eff(1, beta)
            stopping = stopping * z_eff**2
        return stopping

    def metadata(self, Z: int, A: int) -> IonSourceInfo:
        direct = self._direct(Z)
        if direct is not None:
            return self._direct_metadata(direct, Z, A)
        lo, hi = self.energy_range(Z, A)
        ref = self._reference(Z, A)
        shipped = self._shipped_info(ref)
        options = self._common_options(Z, A, ref)
        options["heavy_ion_source"] = self.heavy_ion_source
        options["mstar_mode"] = self.mstar_mode
        source = shipped.source
        if Z >= 3 and self.heavy_ion_source == "mstar":
            source = f"MSTAR 3.12 (mode {self.mstar_mode_used})"
            options["mstar_mode_used"] = self.mstar_mode_used
            options["mstar_evaluation"] = self.mstar_evaluation
        elif Z >= 3:
            form = "(z_eff/z_eff,p)^2" if self.zeff_scaling == "ratio" else "z_eff^2"
            source = f"PSTAR x {form}"
            options["zeff_model"] = self.zeff_model.name
            options["zeff_scaling"] = self.zeff_scaling
        return IonSourceInfo(
            Z=Z,
            A=A,
            source=source,
            reference=shipped.reference,
            i_value_eV=XSTAR_I_VALUE_EV,
            energy_range_MeV_u=(lo, hi),
            edition=shipped.edition,
            options=options,
        )

    def _direct_metadata(self, direct: MSTAR, Z: int, A: int) -> IonSourceInfo:
        """Provenance of the port itself: no shipped table is involved (any MSTAR mode)."""
        self._require_supported(Z)
        port = direct.metadata(Z, A)
        options: dict[str, OptionValue] = {
            "quantity": self.quantity,
            "nuclear": None if self.quantity == "electronic" else self.nuclear.describe(),
            "grid": None,  # the port is evaluated at every requested energy
            "heavy_ion_source": self.heavy_ion_source,
            "mstar_mode": self.mstar_mode,
            "mstar_mode_used": self.mstar_mode_used,
            "mstar_evaluation": self.mstar_evaluation,
        }
        if "velocity_scaled_from" in port.options:
            options["velocity_scaled_from"] = port.options["velocity_scaled_from"]
        return IonSourceInfo(
            Z=Z,
            A=A,
            source=f"MSTAR 3.12 (mode {self.mstar_mode_used})",
            reference=port.reference,
            i_value_eV=port.i_value_eV,
            energy_range_MeV_u=self.energy_range(Z, A),
            edition=port.edition,
            options=options,
        )


__all__ = [
    "ASTAR_REF",
    "MSTAR_DATABASE_ENV",
    "MSTAR_MODES",
    "PSTAR_REF",
    "XSTAR",
    "XSTAR_I_VALUE_EV",
    "HeavyIonSource",
    "MstarEvaluation",
    "ZeffScaling",
]
