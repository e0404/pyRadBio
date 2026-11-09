"""ICRU90Procedure: water stopping power computed the way ICRU Reports 49 and 90 build it.

ICRU Report 90 (2014), Appendix A.3 (printed pp. 80-81), for carbon ions in liquid water:

* below ``T1``: MSTAR 3.12 (Paul & Schinner), "adjusted to reflect the electronic stopping
  powers for alpha particles in liquid water ... from Tables A.11 and A.12, produced with
  the new I values" (footnote 33);
* above ``T2``: Bethe ("the corrected BEST code (I = 78 eV)") with the relativistic Mott
  correction of Ahlen (1980);
* in between: "interpolating the quantity beta^2 S_el(T)/rho with cubic splines ... using
  the scaled empirical data for T <= 30 MeV and the modified Bethe results for
  T >= 60 MeV" (total kinetic energy of C-12, i.e. 2.5 and 5 MeV/u).

For protons and alphas ICRU 90 keeps ICRU 49's procedure (ICRU 49 sec. 3.5, Table 3.7):
empirical Varelas-Biersack fits below ``T1``, Bethe (BEST) above ``T2``, a cubic spline of
``beta^2 S`` vs ``log T`` between, only now with I = 78 eV. Measured: ICRU 90's
p and alpha tables equal ICRU 49's (PSTAR/ASTAR) below ``T1`` (p: identical; alpha:
<= 0.08%) and differ above.

Our implementation, Z 3-18:

* low energies: Paul's MSTAR ratio ([`mstar1`][],
  ``mstar_mode``) applied to **ICRU 90's alpha table** at the same energy per nucleon.
  ASSUMPTION: this is how we read footnote 33's "adjusted to reflect"; ICRU 90 does not
  spell the adjustment out. The alpha table is at I = 78 eV whatever ``i_value_eV`` is.
* high energies: [`Bethe`][] at ``i_value_eV`` with
  ``density_effect`` (default: ICRU 90's own delta, table A.3) and, by default, ICRU 90's
  own higher-order terms: Bloch + Ahlen (1980)'s Mott correction (``higher_order=
  "bloch_ahlen"``); ``"lindhard_sorensen"`` reproduces ICRU
  90 carbon equally well, 0.06% above 30 MeV/u), and --
  like BEST -- shell and Barkas corrections (``shell_barkas``,
  default on: [`shell`][], [`barkas`][]).
  Measured against ICRU 90 carbon: with them, <= 0.070% at every tabulated
  energy above 1 MeV/u (blend included); without them, -0.34% in the blend region.
* blend: an interpolating not-a-knot cubic spline of ``beta^2 S`` in ``ln(E)``, through
  nodes over ``blend_span_decades`` below ``T1`` and above ``T2`` at
  ``blend_nodes_per_decade``. ASSUMPTION: ICRU 49/90 do not state their knots.
* thresholds: carbon's 30 / 60 MeV total, i.e. 2.5 / 5 MeV/u (``thresholds_MeV_u``),
  scaled per ion by ``max(1, (Z/6)^(4/3))`` (``threshold_scaling="stripping"``, default).
  ASSUMPTION: ICRU 90 states them for carbon only; Bohr's stripping criterion
  ``v >~ Z^(2/3) v_0`` makes ``T/A ~ Z^(4/3)`` the energy at which a heavier ion is as
  stripped as carbon at the same threshold, where the bare-charge Bethe part is valid.
  Measured vs MSTAR (mode b), worst band over 0.5-300 MeV/u: Li -1.5%, C -2.0%,
  O -2.0%, Ne -2.0%, Si -1.5%, Ar -1.6% (Ar's T2 = 21.6 MeV/u). Applying carbon's values
  per nucleon instead (``threshold_scaling="none"``) put Si / Ar +16% / +21% above MSTAR
  (+8% / +15% above ICRU 73) at 2.5-10 MeV/u, where they are far from bare. Below ~10
  MeV/u the procedure sits 5-8% under the ICRU 73 errata: MSTAR vs ICRU 73, the same
  choice ICRU 90 made for carbon (its own table is -5.6% from ICRU 73 there).

Protons and alphas (Z 1, 2):

* low energies: ICRU 49's empirical formulae for liquid water
  ([`icru49_empirical`][]; shipped table, evaluated from our
  transcription of its coefficients);
* high energies: Bethe with Bloch, shell and Barkas (``shell_barkas``) -- ICRU 49 eq. 2.1;
  no Mott/Lindhard-Sorensen term (``higher_order`` applies to Z 3-18 only);
* thresholds: ICRU 49 Table 3.7 for water, fixed -- p 0.15 / 0.5 MeV, alpha 1 / 2 MeV
  total (0.25 / 0.5 MeV/u); ``thresholds_MeV_u`` applies to Z 3-18 only;
* blend: as for Z 3-18. ICRU 49 says the spline was "fitted" and does not give its knots;
* ``i_value_eV``: 75 eV gives ICRU 49 (PSTAR/ASTAR), 78 eV ICRU 90. Measured,
  max deviation from the respective table below ``T1`` / in the gap / above ``T2``:
  PSTAR 0.020 / 0.13 / 0.13%, ICRU 90 p 0.020 / 0.13 / 0.15%, ASTAR 0.049 / 0.26 /
  0.14%, ICRU 90 alpha 0.049 / 0.33 / 0.14%.

Both:
* evaluation: by default the procedure is evaluated at the nodes of ``grid`` (default
  50 log-uniform points per decade over the full range) and interpolated (not-a-knot cubic
  in log-log); ``evaluation="direct"`` evaluates it at every requested energy.
* isotopes: computed for the element's most abundant isotope and velocity-scaled for the
  others, as everywhere in [`pyradbio.track.stopping`][].
* ``quantity="total"`` adds ``nuclear``
  ([`WaterNuclearStopping`][]; default: ICRU 49's
  method -- Moliere protons, ZBL otherwise -- as ICRU 49 and 90 tabulate it).
"""

from __future__ import annotations

import math
from typing import ClassVar, Literal, get_args

import numpy as np
import numpy.typing as npt
from scipy.interpolate import CubicSpline

from pyradbio.core.constants import WATER_DENSITY_G_CM3
from pyradbio.track.stopping._tabulated import ShippedTablesModel, TableRef, shipped_table_file
from pyradbio.track.stopping.base import IonSourceInfo, MassStoppingPowerModel, OptionValue
from pyradbio.track.stopping.bethe import Bethe, DensityEffectName
from pyradbio.track.stopping.grid import IonTable, LogEnergyGrid, RegriddedTable
from pyradbio.track.stopping.icru import ICRU
from pyradbio.track.stopping.icru49_empirical import ICRU49_EMPIRICAL_REFERENCE, SHIPPED_STEM
from pyradbio.track.stopping.masses import (
    element_symbol,
    most_abundant_mass_number,
    nuclear_mass_MeV,
)
from pyradbio.track.stopping.mstar.paul import mstar1, resolve_mode
from pyradbio.track.stopping.nuclear import ICRU49_PROTON_POTENTIAL, WaterNuclearStopping

Evaluation = Literal["interpolate", "direct"]
HigherOrder = Literal["lindhard_sorensen", "bloch_ahlen", "bloch_mott", "none"]

DEFAULT_THRESHOLDS_MEV_U = (30.0 / 12.0, 60.0 / 12.0)
"""ICRU 90, App. A.3: MSTAR for T <= 30 MeV, Bethe for T >= 60 MeV (C-12, total energy)."""
ThresholdScaling = Literal["stripping", "none"]
_THRESHOLD_REFERENCE_Z = 6
"""The ion the thresholds are stated for (ICRU 90: carbon) and the stripping exponent."""
_STRIPPING_EXPONENT = 4.0 / 3.0
_LIGHT_MIN_BETA_GAMMA = 0.02
"""Lowest beta*gamma of the p/alpha Bethe (its Barkas term raises below ~0.0206)."""
ICRU90_PROCEDURE_REFERENCE = (
    "ICRU Report 90 (2014), Appendix A.3 (C ions): MSTAR below 30 MeV, Bethe above 60 MeV, "
    "cubic spline of beta^2 S between; reimplemented by pyRadBio"
)


class ICRU90Procedure(MassStoppingPowerModel):
    """ICRU 49/90's procedure for liquid water, Z 1-18 (see module docstring)."""

    name: ClassVar[str] = "icru90_procedure"

    def __init__(
        self,
        *,
        i_value_eV: float = 78.0,
        thresholds_MeV_u: tuple[float, float] = DEFAULT_THRESHOLDS_MEV_U,
        threshold_scaling: ThresholdScaling = "stripping",
        mstar_mode: str = "b",
        density_effect: DensityEffectName = "tabulated",
        higher_order: HigherOrder = "bloch_ahlen",
        shell_barkas: bool = True,
        evaluation: Evaluation = "interpolate",
        grid: LogEnergyGrid | None = None,
        blend_nodes_per_decade: int = 10,
        blend_span_decades: float = 0.5,
        quantity: Literal["electronic", "total"] = "electronic",
        nuclear: WaterNuclearStopping | None = None,
        density_g_cm3: float = WATER_DENSITY_G_CM3,
    ) -> None:
        t1, t2 = thresholds_MeV_u
        if not 0.0 < t1 < t2:
            raise ValueError(f"thresholds_MeV_u must satisfy 0 < T1 < T2, got {thresholds_MeV_u}")
        if evaluation not in get_args(Evaluation):
            raise ValueError(f"evaluation must be one of {get_args(Evaluation)}")
        if threshold_scaling not in get_args(ThresholdScaling):
            raise ValueError(f"threshold_scaling must be one of {get_args(ThresholdScaling)}")
        self.threshold_scaling = threshold_scaling
        if higher_order not in get_args(HigherOrder):
            raise ValueError(f"higher_order must be one of {get_args(HigherOrder)}")
        if quantity not in ("electronic", "total"):
            raise ValueError(f"quantity must be 'electronic' or 'total', got {quantity!r}")
        self.quantity = quantity
        self.nuclear = nuclear or WaterNuclearStopping(proton_potential=ICRU49_PROTON_POTENTIAL)
        if blend_nodes_per_decade < 2 or blend_span_decades <= 0.0:
            raise ValueError("need blend_nodes_per_decade >= 2 and blend_span_decades > 0")
        self.thresholds_MeV_u = (float(t1), float(t2))
        self.mstar_mode = mstar_mode
        self.mstar_mode_used = resolve_mode(mstar_mode)[0]
        self.higher_order = higher_order
        self.shell_barkas = shell_barkas
        self.evaluation = evaluation
        self.grid = grid
        self.blend_nodes_per_decade = blend_nodes_per_decade
        self.blend_span_decades = blend_span_decades
        self.density_g_cm3 = density_g_cm3
        self.bethe = Bethe(
            i_value_eV=i_value_eV,
            density_effect=density_effect,
            bloch=higher_order in ("bloch_mott", "bloch_ahlen"),
            mott=higher_order == "bloch_mott",
            ahlen=higher_order == "bloch_ahlen",
            lindhard_sorensen=higher_order == "lindhard_sorensen",
            shell=shell_barkas,
            barkas=shell_barkas,
            effective_charge=None,  # bare above the scaled thresholds, as BEST
            density_g_cm3=density_g_cm3,
        )
        self.bethe_light = Bethe(
            i_value_eV=i_value_eV,
            density_effect=density_effect,
            bloch=True,
            shell=shell_barkas,
            barkas=shell_barkas,
            effective_charge=None,  # bare above the scaled thresholds, as BEST
            min_beta_gamma=_LIGHT_MIN_BETA_GAMMA,
            density_g_cm3=density_g_cm3,
        )
        self.i_value_eV = self.bethe.i_value_eV
        self._alpha = ICRU(edition="2016")
        self._empirical: dict[int, RegriddedTable] = {}
        self._tables: dict[int, RegriddedTable] = {}
        self._splines: dict[int, CubicSpline] = {}

    # --- the procedure for the reference isotope -----------------------------------------

    def supported_Z(self) -> tuple[int, ...]:
        return tuple(range(1, 19))

    def _ref(self, Z: int) -> TableRef:
        a = most_abundant_mass_number(Z)
        return TableRef("", Z, a, f"{element_symbol(Z)}-{a}")

    def _bethe_for(self, Z: int) -> Bethe:
        return self.bethe_light if Z <= 2 else self.bethe

    def _empirical_ion(self, Z: int) -> IonTable:
        return shipped_table_file(SHIPPED_STEM).table(Z, self._ref(Z).A)

    def _thresholds(self, Z: int) -> tuple[float, float]:
        if Z <= 2:
            options = self._empirical_ion(Z).info.options
            return float(options["T1_MeV_u"]), float(options["T2_MeV_u"])  # type: ignore[arg-type]
        t1, t2 = self.thresholds_MeV_u
        if self.threshold_scaling == "none":
            return t1, t2
        f = max(1.0, (Z / _THRESHOLD_REFERENCE_Z) ** _STRIPPING_EXPONENT)
        return t1 * f, t2 * f

    def _raw_range(self, Z: int) -> tuple[float, float]:
        """Range for the reference isotope (low: empirical/alpha table; high: Bethe)."""
        a = self._ref(Z).A
        if Z <= 2:
            low_lo, low_hi = self._empirical_ion(Z).info.energy_range_MeV_u
        else:
            low_lo, low_hi = self._alpha.energy_range(2, 4)
        b_lo, b_hi = self._bethe_for(Z).energy_range(Z, a)
        t1, t2 = self._thresholds(Z)
        if t2 < b_lo:
            raise ValueError(f"T2 = {t2} MeV/u lies below the Bethe range ({b_lo:.3g} MeV/u)")
        if t1 > low_hi * (1.0 + 1e-12):
            raise ValueError(f"T1 = {t1} MeV/u lies above the alpha table")
        return low_lo, b_hi

    def _low(self, Z: int, energy: npt.NDArray[np.float64]) -> npt.NDArray[np.float64]:
        if Z <= 2:
            if Z not in self._empirical:
                ion = self._empirical_ion(Z)
                self._empirical[Z] = RegriddedTable(
                    ion.energy_MeV_u, ion.columns["electronic_MeV_cm2_g"]
                )
            table = self._empirical[Z]
            return table(np.clip(energy, *table.energy_range))
        alpha = self._alpha.mass_stopping_power(2, 4, energy)
        return np.array(
            [
                mstar1(float(s), self.mstar_mode, Z, float(e))[0]
                for s, e in zip(alpha, energy, strict=True)
            ]
        )

    def _high(self, Z: int, energy: npt.NDArray[np.float64]) -> npt.NDArray[np.float64]:
        return self._bethe_for(Z).mass_stopping_power(Z, self._ref(Z).A, energy)

    def _beta2(self, Z: int, energy: npt.NDArray[np.float64]) -> npt.NDArray[np.float64]:
        a = self._ref(Z).A
        gamma = 1.0 + energy * a / nuclear_mass_MeV(Z, a)
        out: npt.NDArray[np.float64] = 1.0 - 1.0 / (gamma * gamma)
        return out

    def _blend(self, Z: int) -> CubicSpline:
        if Z not in self._splines:
            t1, t2 = self._thresholds(Z)
            n = max(2, math.ceil(self.blend_span_decades * self.blend_nodes_per_decade) + 1)
            span = 10.0**self.blend_span_decades
            e_low = np.geomspace(t1 / span, t1, n)
            e_high = np.geomspace(t2, t2 * span, n)
            e = np.concatenate([e_low, e_high])
            y = self._beta2(Z, e) * np.concatenate([self._low(Z, e_low), self._high(Z, e_high)])
            self._splines[Z] = CubicSpline(np.log(e), y, bc_type="not-a-knot")
        return self._splines[Z]

    def _direct(self, Z: int, energy: npt.NDArray[np.float64]) -> npt.NDArray[np.float64]:
        t1, t2 = self._thresholds(Z)
        out = np.empty_like(energy)
        low, high = energy <= t1, energy >= t2
        mid = ~(low | high)
        if low.any():
            out[low] = self._low(Z, energy[low])
        if high.any():
            out[high] = self._high(Z, energy[high])
        if mid.any():
            out[mid] = self._blend(Z)(np.log(energy[mid])) / self._beta2(Z, energy[mid])
        return out

    def _table(self, Z: int) -> RegriddedTable:
        if Z not in self._tables:
            lo, hi = self._raw_range(Z)
            grid = self.grid or LogEnergyGrid.from_points_per_decade(lo, hi, 50)
            nodes = grid.energies
            if nodes[0] < lo or nodes[-1] > hi:
                raise ValueError(
                    f"grid [{nodes[0]:g}, {nodes[-1]:g}] exceeds the range [{lo:g}, {hi:g}]"
                )
            self._tables[Z] = RegriddedTable(nodes, self._direct(Z, nodes))
        return self._tables[Z]

    # --- MassStoppingPowerModel ----------------------------------------------------------

    def energy_range(self, Z: int, A: int) -> tuple[float, float]:
        self._require_supported(Z)
        ref = self._ref(Z)
        lo, hi = (
            self._table(Z).energy_range if self.evaluation == "interpolate" else self._raw_range(Z)
        )
        return (
            ShippedTablesModel._from_reference(Z, A, lo, ref),
            ShippedTablesModel._from_reference(Z, A, hi, ref),
        )

    def _evaluate(
        self, Z: int, A: int, energy_MeV_u: npt.NDArray[np.float64]
    ) -> npt.NDArray[np.float64]:
        ref = self._ref(Z)
        if self.evaluation == "direct":
            lo, hi = self._raw_range(Z)
        else:
            lo, hi = self._table(Z).energy_range
        e_ref = np.clip(ShippedTablesModel._to_reference(Z, A, energy_MeV_u, ref), lo, hi)
        stopping = self._direct(Z, e_ref) if self.evaluation == "direct" else self._table(Z)(e_ref)
        if self.quantity == "total":
            stopping = stopping + self.nuclear.mass_stopping_power(Z, A, energy_MeV_u)
        return stopping

    def metadata(self, Z: int, A: int) -> IonSourceInfo:
        lo, hi = self.energy_range(Z, A)
        ref = self._ref(Z)
        light = Z <= 2
        sb = ", shell + Barkas)" if self.shell_barkas else ")"
        options: dict[str, OptionValue] = {
            "quantity": self.quantity,
            "nuclear": None if self.quantity == "electronic" else self.nuclear.describe(),
            "thresholds_MeV_u": list(self._thresholds(Z)),  # type: ignore[dict-item]
            "low_energy": (
                "ICRU 49 empirical (Varelas-Biersack), liquid water"
                if light
                else f"MSTAR 3.12 ratio (mode {self.mstar_mode_used}) x ICRU 90 alpha table"
            ),
            "high_energy": (
                f"Bethe (I = {self.i_value_eV:g} eV, {self.bethe.density_effect.name}, "
                + ("bloch" if light else self.higher_order)
                + sb
            ),
            "blend": (
                f"not-a-knot cubic spline of beta^2 S in ln E, {self.blend_nodes_per_decade} "
                f"nodes/decade over {self.blend_span_decades:g} decades each side"
            ),
            "evaluation": self.evaluation,
            "grid": None if self.grid is None else self.grid.to_dict(),
            "mstar_mode": self.mstar_mode,
            "threshold_scaling": self.threshold_scaling,
        }
        if A != ref.A:
            options["velocity_scaled_from"] = ref.label
        return IonSourceInfo(
            Z=Z,
            A=A,
            source="ICRU 90 procedure (pyRadBio)",
            reference=(
                f"{ICRU49_EMPIRICAL_REFERENCE}; ICRU 49 sec. 3.5; ICRU 90 (2014)"
                if light
                else ICRU90_PROCEDURE_REFERENCE
            ),
            i_value_eV=self.i_value_eV,
            energy_range_MeV_u=(lo, hi),
            edition="ICRU 90",
            options=options,
        )


__all__ = [
    "DEFAULT_THRESHOLDS_MEV_U",
    "ICRU90_PROCEDURE_REFERENCE",
    "Evaluation",
    "HigherOrder",
    "ICRU90Procedure",
    "ThresholdScaling",
]
