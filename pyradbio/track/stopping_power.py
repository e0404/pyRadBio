"""Stopping-power interface and water stopping-power sources.

A ``StoppingPower`` provides the linear (unrestricted, electronic) stopping power
-- the LET -- in water for an ion at a given energy, in keV/um.

* [`default_stopping_power`][] is what every pyRadBio model uses unless told
  otherwise: [`DefaultStoppingPower`][] -- ICRU 90's procedure for Z 1-18
  ([`pyradbio.track.stopping`][]), Hubert alpha scaling for Z 19-92 from
  2.5 MeV/u, and the dependency-free analytic [`SimpleBetheStoppingPower`][]
  (good to ~2-5% above ~2 MeV/u, up to ~30% low below ~1 MeV/u) everywhere else,
  anchored to the adjacent branch at its range edge. It is the one seam every
  model default goes through; other sources are explicit opt-ins.
* [`TabularStoppingPower`][] / [`MassStoppingPowerSource`][] /
  [`LibdedxStoppingPower`][] plug in any other table or library (libdEdx is
  GPL and a runtime opt-in, never imported by default).
* [`pyradbio.track.northcliffe_schilling`][] (paper-era NS1970) and
  [`pyradbio.track.srim`][] (user-supplied SRIM tables) serve reconstructions
  that must use a specific historical source.

pyMKM's stopping-power tables are reachable only through the optional pyMKM
adapter ([`pyradbio.models.mkm.pymkm_backend`][]), as a validation reference;
this module never imports pyMKM.
"""

from __future__ import annotations

import math
from collections.abc import Callable, Mapping, Sequence
from dataclasses import dataclass
from functools import cache
from types import MappingProxyType
from typing import Any, Literal, Protocol

import numpy as np
import numpy.typing as npt
from scipy.optimize import brentq

from pyradbio.core.constants import (
    ATOMIC_MASS_UNIT_MEV,
    BETHE_K_MEV_CM2_MOL,
    ELECTRON_MASS_MEV,
    MEV_PER_CM_TO_KEV_PER_UM,
    WATER_DENSITY_G_CM3,
    WATER_MEAN_EXCITATION_EV,
    WATER_Z_OVER_A,
)
from pyradbio.core.radiation import Ion
from pyradbio.track.stopping.effective_charge import barkas_effective_charge

effective_charge = barkas_effective_charge
"""Barkas effective charge; the implementation lives in
[`pyradbio.track.stopping.effective_charge`][] (Katz, MCDS and the lineal-energy
engines import it under this name).
"""


def let_from_mass_stopping_power(
    mass_stopping_power_MeV_cm2_g: float, density_g_cm3: float = WATER_DENSITY_G_CM3
) -> float:
    """LET [keV/um] from a mass stopping power [MeV cm^2/g] and a density [g/cm^3]."""
    return mass_stopping_power_MeV_cm2_g * density_g_cm3 * MEV_PER_CM_TO_KEV_PER_UM


class StoppingPower(Protocol):
    def let_keV_um(self, ion: Ion) -> float: ...


class VectorizedStoppingPower(StoppingPower, Protocol):
    def lets_keV_um(
        self, Z: int, A: int, energies_MeV_u: Sequence[float]
    ) -> npt.NDArray[np.float64]: ...


class SimpleBetheStoppingPower:
    """Dependency-free water stopping power via the Bethe formula.

    Mass stopping power [MeV cm^2/g]:

    ```text
    -dE/dx = K * (Z/A)_water * z_eff^2 / beta^2 * [ln(2 m_e c^2 beta^2 gamma^2 / I)
             - beta^2]
    ```

    using the heavy-ion limit of the maximum energy transfer
    (``W_max ~ 2 m_e c^2 beta^2 gamma^2``) so the usual ``0.5*ln(...W_max...)``
    term collapses to a single logarithm. The density-effect correction is
    neglected (sub-percent below ~1 GeV/u). LET = mass stopping power * density.

    Accuracy: validated against the Geant4 (pyMKM) tables to <=4% (~1.4% RMS)
    for Z<=10 over 1-200 MeV/u (see
    ``test_bethe_matches_reference_tables_within_a_few_percent``), but up to ~30%
    *low* between 0.1 and 1 MeV/u, and invalid (raises) at the lowest energies.
    This is close to the ceiling of the effective-charge Bethe form: the ICRU
    shell correction diverges below its validity range (beta*gamma < ~0.13, i.e.
    the low velocities of interest) and the Bloch close-collision term
    *regresses* the table agreement here because [`effective_charge`][]
    already absorbs those effects empirically -- so neither is applied.

    Heavy ions (Z>10) have no bundled reference table, and the low-velocity
    heavy-ion regime (where the projectile is slower than target K-shell
    electrons and the effective charge dominates) is the model's main
    uncertainty. For production heavy-ion accuracy prefer a table- or
    libdedx-backed source.

    It is the library default's fallback for Z > 18 ([`DefaultStoppingPower`][])
    and a quick, any-Z reference; for stopping-power sensitivity studies prefer
    [`pyradbio.track.stopping.bethe.Bethe`][], whose corrections switch one by one.
    """

    def __init__(self, density_g_cm3: float = WATER_DENSITY_G_CM3) -> None:
        self.density_g_cm3 = density_g_cm3

    def settings(self) -> dict[str, object]:
        """The fixed formula's settings, JSON-safe (table provenance)."""
        return {
            "name": "simple_bethe",
            "density_g_cm3": self.density_g_cm3,
            "i_value_eV": WATER_MEAN_EXCITATION_EV,
            "effective_charge": "barkas",
            "corrections": None,
        }

    def mass_stopping_power_MeV_cm2_g(self, ion: Ion) -> float:
        gamma = 1.0 + ion.energy_MeV_u / ATOMIC_MASS_UNIT_MEV
        beta2 = 1.0 - 1.0 / (gamma * gamma)
        beta = math.sqrt(beta2)
        z_eff = effective_charge(ion.Z, beta)
        i_mev = WATER_MEAN_EXCITATION_EV * 1.0e-6
        arg = 2.0 * ELECTRON_MASS_MEV * beta2 * gamma * gamma / i_mev
        bracket = math.log(arg) - beta2
        if bracket <= 0.0:
            raise ValueError(f"Bethe formula invalid (too low energy) for {ion!r}")
        return BETHE_K_MEV_CM2_MOL * WATER_Z_OVER_A * z_eff * z_eff / beta2 * bracket

    def let_keV_um(self, ion: Ion) -> float:
        sp = self.mass_stopping_power_MeV_cm2_g(ion)
        return let_from_mass_stopping_power(sp, self.density_g_cm3)

    def lets_keV_um(
        self, Z: int, A: int, energies_MeV_u: Sequence[float]
    ) -> npt.NDArray[np.float64]:
        return np.array(
            [self.let_keV_um(Ion(Z, A, float(energy))) for energy in energies_MeV_u],
            dtype=np.float64,
        )


@dataclass(frozen=True, slots=True)
class MassStoppingPowerSource:
    """Adapt any external mass-stopping-power callable to [`StoppingPower`][].

    ``mass_sp_MeV_cm2_g(Z, A, energy_MeV_u)`` must return the water mass stopping
    power in ``MeV cm^2/g`` (the conventional output of stopping-power libraries);
    this converts it to LET in ``keV/um``. It is the dependency-free seam for
    plugging in an arbitrary external source without a hard dependency, e.g. a
    custom fit or an in-house table:

    ```text
    sp = MassStoppingPowerSource(lambda z, a, e: my_table_MeV_cm2_g(z, e))
    ```

    For libdEdx specifically, prefer [`LibdedxStoppingPower`][], which wraps
    its Python binding directly.
    """

    mass_sp_MeV_cm2_g: Callable[[int, int, float], float]
    density_g_cm3: float = WATER_DENSITY_G_CM3

    def let_keV_um(self, ion: Ion) -> float:
        mass_sp = self.mass_sp_MeV_cm2_g(ion.Z, ion.A, ion.energy_MeV_u)
        return let_from_mass_stopping_power(mass_sp, self.density_g_cm3)


@dataclass(frozen=True, slots=True)
class StoppingPowerTable:
    """One ion's tabulated LET values, interpolated in log(energy)-log(LET).

    ``interpolation="linear"`` switches to linear-in-energy, linear-in-LET
    interpolation. That is *worse* physics on a coarse grid -- ``LET(E)`` is
    close to a power law, so log-log is the better interpolant and stays the
    default -- but it is what some reference implementations do, and matching
    an implementation's interpolant is the only way to remove stopping power as
    a free variable when benchmarking against it. See
    [`pyradbio.track.srim`][].
    """

    energy_MeV_u: npt.NDArray[np.float64]
    let_keV_um: npt.NDArray[np.float64]
    interpolation: Literal["log-log", "linear"] = "log-log"

    @classmethod
    def from_arrays(
        cls,
        energy_MeV_u: Sequence[float],
        let_keV_um: Sequence[float],
        interpolation: Literal["log-log", "linear"] = "log-log",
    ) -> StoppingPowerTable:
        energy = np.asarray(energy_MeV_u, dtype=np.float64)
        let = np.asarray(let_keV_um, dtype=np.float64)
        if energy.ndim != 1 or let.ndim != 1 or energy.size != let.size:
            raise ValueError("energy and LET arrays must be one-dimensional and equally sized")
        if energy.size < 2:
            raise ValueError("a stopping-power table needs at least two energy points")
        if np.any(energy <= 0.0) or np.any(let <= 0.0):
            raise ValueError("tabulated energy and LET values must be positive")

        order = np.argsort(energy)
        energy = energy[order]
        let = let[order]
        if np.any(np.diff(energy) <= 0.0):
            raise ValueError("tabulated energies must be unique")
        if interpolation not in ("log-log", "linear"):
            raise ValueError(f"interpolation must be 'log-log' or 'linear', got {interpolation!r}")
        return cls(energy_MeV_u=energy, let_keV_um=let, interpolation=interpolation)

    def interpolate(self, energies_MeV_u: Sequence[float]) -> npt.NDArray[np.float64]:
        energies = np.asarray(energies_MeV_u, dtype=np.float64)
        if np.any(energies < self.energy_MeV_u[0]) or np.any(energies > self.energy_MeV_u[-1]):
            raise ValueError(
                "energy outside tabulated range "
                f"[{self.energy_MeV_u[0]:.6g}, {self.energy_MeV_u[-1]:.6g}] MeV/u"
            )
        if self.interpolation == "linear":
            linear: npt.NDArray[np.float64] = np.interp(
                energies, self.energy_MeV_u, self.let_keV_um
            )
            return linear
        log_let = np.interp(
            np.log(energies),
            np.log(self.energy_MeV_u),
            np.log(self.let_keV_um),
        )
        let: npt.NDArray[np.float64] = np.exp(log_let)
        return let


class TabularStoppingPower:
    """In-memory tabulated stopping power with scalar and vectorized interpolation."""

    def __init__(self, tables: Mapping[int, StoppingPowerTable]) -> None:
        if not tables:
            raise ValueError("at least one stopping-power table is required")
        self._tables = dict(tables)

    @classmethod
    def from_arrays(
        cls,
        tables: Mapping[
            int,
            tuple[Sequence[float], Sequence[float]] | Mapping[str, Sequence[float]],
        ],
        interpolation: Literal["log-log", "linear"] = "log-log",
    ) -> TabularStoppingPower:
        parsed: dict[int, StoppingPowerTable] = {}
        for z, data in tables.items():
            if isinstance(data, Mapping):
                energy = data["energy"]
                let = data["let"]
            else:
                energy, let = data
            parsed[int(z)] = StoppingPowerTable.from_arrays(energy, let, interpolation)
        return cls(parsed)

    def _table_for_z(self, z: int) -> StoppingPowerTable:
        try:
            return self._tables[z]
        except KeyError as exc:
            raise KeyError(f"no stopping-power table for Z={z}") from exc

    def let_keV_um(self, ion: Ion) -> float:
        return float(self.lets_keV_um(ion.Z, ion.A, [ion.energy_MeV_u])[0])

    def lets_keV_um(
        self, Z: int, A: int, energies_MeV_u: Sequence[float]
    ) -> npt.NDArray[np.float64]:
        del A  # tables are keyed by ion species Z (stopping power at fixed E/A).
        return self._table_for_z(Z).interpolate(energies_MeV_u)


class LibdedxStoppingPower:
    """Optional runtime source backed by libdEdx (ICRU49/73, MSTAR, PSTAR...).

    libdEdx (https://github.com/APTG/libdedx) is a reference-grade stopping-power
    library whose ``DEFAULT`` program auto-selects the best available table per
    ion -- MSTAR / ICRU73 experimental tables for ``Z <= 18`` -- and falls back
    to its own extended Bethe (``BETHE_EXT00``) for heavier ions, reaching
    ``Z = 112``. That fills exactly the gap where the bundled analytic
    [`SimpleBetheStoppingPower`][] is least certain (heavy ions, low velocity)
    and where the pyMKM tables stop (``Z <= 10``).

    This is a *runtime* opt-in: ``libdedx`` is **not** a pyRadBio dependency (it is
    GPL-licensed), so importing [`pyradbio.track`][] never imports it. Construct
    this class only when the user has ``libdedx`` importable; otherwise it raises
    a helpful `ImportError`.

    ``program`` and ``material`` accept either a libdEdx enum name (e.g.
    ``"DEFAULT"``, ``"ICRU73"``, ``"WATER_LIQUID"``) or its integer id. libdEdx
    keys ions by atomic number ``Z`` and returns mass stopping power
    [MeV cm^2/g], converted here to LET [keV/um]. Energies outside the library's
    tabulated range raise ``RuntimeError`` from libdEdx.
    """

    def __init__(
        self,
        program: str | int = "DEFAULT",
        material: str | int = "WATER_LIQUID",
        density_g_cm3: float = WATER_DENSITY_G_CM3,
    ) -> None:
        self._core = self._load_libdedx()
        self.program = self._resolve_constant(program)
        self.material = self._resolve_constant(material)
        self.density_g_cm3 = density_g_cm3

    @staticmethod
    def _load_libdedx() -> Any:
        try:
            from libdedx import _core  # type: ignore[import-not-found]
        except ImportError as exc:  # pragma: no cover - requires the optional package
            raise ImportError(
                "LibdedxStoppingPower requires the optional 'libdedx' package "
                "(GPL; not a pyRadBio dependency). Install libdEdx into this "
                "environment to use it: https://github.com/APTG/libdedx"
            ) from exc
        return _core

    def _resolve_constant(self, value: str | int) -> int:
        return value if isinstance(value, int) else int(getattr(self._core, value))

    def let_keV_um(self, ion: Ion) -> float:
        mass_sp = self._core.simple_stp_for_program(
            self.program, ion.Z, self.material, ion.energy_MeV_u
        )
        return let_from_mass_stopping_power(float(mass_sp), self.density_g_cm3)

    def lets_keV_um(
        self, Z: int, A: int, energies_MeV_u: Sequence[float]
    ) -> npt.NDArray[np.float64]:
        del A  # libdEdx keys ions by atomic number Z only.
        energies = np.ascontiguousarray(energies_MeV_u, dtype=np.float64)
        mass_sp = self._core.stp_table(self.program, Z, self.material, energies)
        return np.asarray(mass_sp, dtype=np.float64) * self.density_g_cm3 * MEV_PER_CM_TO_KEV_PER_UM


@cache
def _shared_procedure(density_g_cm3: float) -> Any:
    """One ICRU90Procedure per density, shared by every model (its per-ion tables are
    built lazily, ~0.3 s per ion on first use)."""
    from pyradbio.track.stopping.icru90_procedure import ICRU90Procedure

    return ICRU90Procedure(density_g_cm3=density_g_cm3)


@cache
def _shared_heavy(density_g_cm3: float) -> Any:
    """Hubert's alpha scaling on the shared procedure's alpha (ICRU 90)."""
    from pyradbio.track.stopping.hubert import HubertAlphaScaling

    return HubertAlphaScaling(alpha=_shared_procedure(density_g_cm3), density_g_cm3=density_g_cm3)


DefaultSource = Literal["icru90_procedure", "hubert1990", "simple_bethe_anchored", "simple_bethe"]
"""``source()`` labels of [`DefaultStoppingPower`][]'s three branches."""


class DefaultStoppingPower:
    """The library default: ICRU 90's procedure for Z 1-18, Hubert scaling above, Bethe last.

    * ``1 <= Z <= 18`` inside its energy range:
      [`ICRU90Procedure`][pyradbio.track.stopping.icru90_procedure.ICRU90Procedure] with its
      defaults (electronic stopping, I = 78 eV; reproduces ICRU 90's p, alpha and C tables to <=
      0.33% from 1 keV/u up; Z 3-18 within 2% of MSTAR over 0.5-300 MeV/u);
    * ``19 <= Z <= 92`` from 2.5 MeV/u:
      [`HubertAlphaScaling`][pyradbio.track.stopping.hubert.HubertAlphaScaling] -- Hubert, Bimbot &
      Gauvin (1990)'s ``(gamma Z/2)^2 S_alpha`` with ICRU 90's alpha; fitted by them to O-U data,
      ~5% in solids; no water reference above Ar is held;
    * outside that branch's energy range (``Z > 18`` below 2.5 MeV/u; above ~1e6 MeV/u):
      [`SimpleBetheStoppingPower`][] **anchored** to the branch at its nearest edge,
      ``S = S_Bethe(E) S_branch(E_edge) / S_Bethe(E_edge)`` (a hard
      switch would jump +5.8% for K to +21.9% for Xe at 2.5 MeV/u). The value is continuous;
      the slope kinks by <= 0.012 in ``d ln S/d ln E`` for Z <= 54, 0.05-0.09 for Pb/U.
      This is the "ratio x reference" construction of MSTAR and ICRU 90, with the shape
      from Bethe; measured for Z 6-18 against MSTAR (anchored at 2.5 MeV/u) it is the
      best of the candidate shapes (<= 4% at 1-2.5 MeV/u, <= 10% at 0.5-1, <= 19% at
      0.1-0.5), against Hubert's own formula extrapolated (collapses) and Brandt-Kitagawa
      x protons (+5 to +60%). UNVALIDATED for Z > 18 (no water reference is held);
    * no branch at all (``Z > 92``, isotopes without a shipped mass): plain
      [`SimpleBetheStoppingPower`][], UNVALIDATED (it differs from ICRU 90/73 by up to
      -29% at 0.1-1 MeV/u for Z <= 18).

    The switches between branches are seams (``docs/references/StoppingPower.md``);
    [`source`][] names the branch serving an ion.
    """

    def __init__(self, density_g_cm3: float = WATER_DENSITY_G_CM3) -> None:
        self.density_g_cm3 = density_g_cm3
        self._fallback = SimpleBetheStoppingPower(density_g_cm3=density_g_cm3)
        self._anchors: dict[tuple[int, int, float], float] = {}
        """``(Z, A, edge energy) -> S_branch / S_Bethe`` there (the anchor factor)."""

    @property
    def procedure(self) -> Any:
        return _shared_procedure(self.density_g_cm3)

    @property
    def heavy(self) -> Any:
        return _shared_heavy(self.density_g_cm3)

    def _branch(self, Z: int, A: int) -> tuple[DefaultSource, Any, tuple[float, float]] | None:
        """``(label, model, energy range)`` of the table-quality branch, or None."""
        try:
            if 1 <= Z <= 18:
                return "icru90_procedure", self.procedure, self.procedure.energy_range(Z, A)
            if Z <= self.heavy.supported_Z()[-1]:
                return "hubert1990", self.heavy, self.heavy.energy_range(Z, A)
        except KeyError:  # no shipped mass for this isotope
            return None
        return None

    def source(self, ion: Ion) -> DefaultSource:
        """Which branch serves ``ion``: ``"icru90_procedure"``, ``"hubert1990"``,
        ``"simple_bethe_anchored"`` (outside that branch's range) or ``"simple_bethe"``
        (no branch for this ion)."""
        branch = self._branch(ion.Z, ion.A)
        if branch is None:
            return "simple_bethe"
        if branch[2][0] <= ion.energy_MeV_u <= branch[2][1]:
            return branch[0]
        return "simple_bethe_anchored"

    def settings(self) -> dict[str, object]:
        """Each branch's resolved settings and the fallback rule, JSON-safe."""
        return {
            "name": "default",
            "density_g_cm3": self.density_g_cm3,
            "branches": {
                "icru90_procedure": {"Z": [1, 18], **self.procedure.settings()},
                "hubert1990": {"Z": [19, self.heavy.supported_Z()[-1]], **self.heavy.settings()},
            },
            "fallback": {
                **self._fallback.settings(),
                "anchoring": "outside a branch's range: anchored to it at the nearest range "
                "edge, S_Bethe(E) S_branch(E_edge)/S_Bethe(E_edge); no branch: unscaled",
            },
        }

    def _anchor(self, Z: int, A: int, model: Any, edge: float) -> float:
        """``S_branch / S_Bethe`` at the branch's range edge ``edge`` [MeV/u]."""
        key = (Z, A, edge)
        if key not in self._anchors:
            branch = float(model.lets_keV_um(Z, A, [edge])[0])
            self._anchors[key] = branch / float(self._fallback.lets_keV_um(Z, A, [edge])[0])
        return self._anchors[key]

    def let_keV_um(self, ion: Ion) -> float:
        return float(self.lets_keV_um(ion.Z, ion.A, [ion.energy_MeV_u])[0])

    def lets_keV_um(
        self, Z: int, A: int, energies_MeV_u: Sequence[float]
    ) -> npt.NDArray[np.float64]:
        e = np.asarray(energies_MeV_u, dtype=np.float64)
        branch = self._branch(Z, A)
        if branch is None:
            inside = np.zeros(e.shape, dtype=bool)
        else:
            inside = (e >= branch[2][0]) & (e <= branch[2][1])
        out = np.empty_like(e)
        if inside.any():
            assert branch is not None
            out[inside] = branch[1].lets_keV_um(Z, A, e[inside])
        if (~inside).any():
            out[~inside] = self._fallback.lets_keV_um(Z, A, e[~inside].tolist())
            if branch is not None:
                _, model, (lo, hi) = branch
                for mask, edge in ((e < lo, lo), (e > hi, hi)):
                    if mask.any():
                        out[mask] *= self._anchor(Z, A, model, edge)
        return out


def default_stopping_power(density_g_cm3: float = WATER_DENSITY_G_CM3) -> StoppingPower:
    """The stopping power every pyRadBio model uses unless one is passed explicitly.

    Returns a [`DefaultStoppingPower`][] at ``density_g_cm3``: ICRU 90's procedure
    ([`ICRU90Procedure`][pyradbio.track.stopping.icru90_procedure.ICRU90Procedure]) for Z 1-18,
    Hubert's alpha scaling
    ([`HubertAlphaScaling`][pyradbio.track.stopping.hubert.HubertAlphaScaling]) for Z 19-92 from 2.5
    MeV/u, and the analytic [`SimpleBetheStoppingPower`][] elsewhere, anchored to the adjacent
    branch at its range edge. Other sources -- the [`pyradbio.track.stopping`][] classes,
    [`northcliffe_schilling`][] (NS1970), [`srim`][] (user SRIM tables), [`LibdedxStoppingPower`][]
    and pyMKM's tables via [`pyradbio.models.mkm.pymkm_backend`][] -- are explicit opt-ins passed to
    a model's ``stopping_power`` field.
    """
    return DefaultStoppingPower(density_g_cm3=density_g_cm3)


STOPPING_POWER_DESCRIPTIONS: Mapping[str, str] = MappingProxyType(
    {
        # settings()["name"] of this module's sources
        "default": "ICRU 90 procedure (Z 1-18) + Hubert 1990 + Bethe (library default)",
        "simple_bethe": "analytic Bethe",
        # ``name`` of pyradbio.track.northcliffe_schilling (BetheOnly reports simple_bethe
        # through its inherited settings(), so its "bethe" CLI name is never looked up)
        "ns1970+bethe": "Northcliffe & Schilling 1970 table + analytic Bethe above 12 MeV/u",
        # ``name`` of the pyradbio.track.stopping MassStoppingPowerModel subclasses
        "bethe": "Bethe formula with switchable corrections",
        "icru90_procedure": "ICRU 90 procedure",
        "hubert1990": "Hubert, Bimbot & Gauvin 1990 alpha scaling",
        "icru": "ICRU 49/73/90 tables",
        "mstar": "MSTAR 3.12 (Paul & Schinner)",
        "xstar": "NIST PSTAR/ASTAR + MSTAR 3.12 (XSTAR)",
        # The TRiP98 FLUKA dE/dx adapter of examples/lem{1,4}_trip98_rbe_reproduction.py,
        # under its registered name and, while that adapter has no ``name``, its class
        # name.
        "trip98-fluka-dedx": "TRiP98 FLUKA dE/dx table",
        "TripDedxStoppingPower": "TRiP98 FLUKA dE/dx table",
        # class name of pyradbio.models.mkm.pymkm_backend's unnamed source (never
        # imported here: track/ does not import models/)
        "PyMKMStoppingPower": "pyMKM bundled table (validation reference)",
    }
)
"""Human-readable text for every name a stopping-power source reports, read by
[`describe_stopping_power`][] (read-only). Keys are ``settings()["name"]``, a
``name`` attribute, or -- for sources that carry neither -- the class name.
"""

_DESCRIPTION_OPTIONS: Mapping[str, tuple[str, str]] = MappingProxyType(
    {
        "icru": ("edition", "{text} (edition {value})"),
        "PyMKMStoppingPower": ("source", "{text}: {value}"),
    }
)
"""Keys of [`STOPPING_POWER_DESCRIPTIONS`][] whose sources come in variants, mapped
to the option that tells them apart and how it extends the text: the ICRU edition
(``settings()["edition"]``) and the pyMKM table (``PyMKMStoppingPower.source``).
"""


def describe_stopping_power(sp: object) -> str:
    """A human-readable name for the stopping-power source ``sp`` (for prints and titles).

    The text comes from [`STOPPING_POWER_DESCRIPTIONS`][], looked up by, in order,
    ``sp.settings()["name"]``, the ``sp.name`` attribute and ``type(sp).__name__``; the
    first key found wins. ``settings()`` goes first because it names what a source
    computes: [`BetheOnly`][pyradbio.track.northcliffe_schilling.BetheOnly] is
    [`SimpleBetheStoppingPower`][] under the CLI name ``"bethe"``, which is also the
    name of the corrected [`pyradbio.track.stopping.bethe.Bethe`][]. A ``settings()``
    that cannot be called bare (``TypeError``) or refuses (``ValueError``) is skipped.
    Sources that come in variants name theirs: the ICRU edition and the pyMKM table
    (e.g. ``"ICRU 49/73/90 tables (edition 2016)"``); other options (I-value, XSTAR's
    heavy-ion source, ...) are not in the text -- ``settings()`` holds them. A source
    none of whose keys is mapped is described by ``type(sp).__name__``. The TRiP98 and
    pyMKM sources carry no name, so they are keyed by class name.

    Use this instead of hard-coding a label such as ``"analytic Bethe (library default)"``:
    the default source is [`DefaultStoppingPower`][], not a single formula.
    """
    settings: dict[str, object] = {}
    keys: list[str] = []
    method = getattr(sp, "settings", None)
    if callable(method):
        try:
            resolved = method()
        except (TypeError, ValueError):
            resolved = None
        if isinstance(resolved, Mapping):
            settings = dict(resolved)
            if isinstance(settings.get("name"), str):
                keys.append(str(settings["name"]))
    name = getattr(sp, "name", None)
    if isinstance(name, str):
        keys.append(name)
    keys.append(type(sp).__name__)
    for key in keys:
        if key not in STOPPING_POWER_DESCRIPTIONS:
            continue
        text = STOPPING_POWER_DESCRIPTIONS[key]
        if key in _DESCRIPTION_OPTIONS:
            option, template = _DESCRIPTION_OPTIONS[key]
            value = settings.get(option, getattr(sp, option, None))
            if isinstance(value, str):
                return template.format(text=text, value=value)
        return text
    return type(sp).__name__


def energy_for_let_MeV_u(
    stopping_power: StoppingPower,
    Z: int,
    A: int,
    let_keV_um: float,
    *,
    branch: str = "high",
    energy_bounds_MeV_u: tuple[float, float] = (0.1, 1000.0),
    xtol: float | None = None,
    rtol: float | None = None,
) -> float:
    """Invert ``LET(E)`` for the kinetic energy giving ``let_keV_um`` [MeV/u].

    ``LET(E)`` is not monotone -- it rises to a peak (the Bragg peak, ~0.3-0.5
    MeV/u for light ions) and falls again -- so every LET below that peak has two
    solutions. ``branch="high"`` (the default) returns the one *above* the peak,
    which is the physically relevant root for a degraded therapeutic beam;
    ``branch="low"`` returns the sub-peak root. Both scan ``energy_bounds_MeV_u``
    on 400 log-spaced points for the peak first and solve to ``xtol=1e-10``,
    ``rtol=1e-12``.

    ``branch="bracket"`` is the strict form: no peak scan, just
    ``scipy.optimize.brentq`` on ``energy_bounds_MeV_u`` itself, with ``xtol`` /
    ``rtol`` forwarded only when given (scipy's defaults otherwise). It returns
    exactly what an inline ``brentq(lambda e: sp.let_keV_um(Ion(Z, A, e)) - let,
    lo, hi[, xtol=...])`` call returns, with the same LET evaluations. The bracket must
    hold a single sign change, i.e. sit on one side of the peak; that is the caller's to
    choose.

    Needed to reconstruct experiments quoted against LET rather than energy --
    e.g. Furusawa et al. 2000, whose survival data Inaniwa & Kanematsu 2018
    figs. 2-3 plot per LET, having assumed monoenergetic beams at those LETs
    (doi:10.1088/1361-6560/aabede sec 3.2).

    :raises ValueError: if ``let_keV_um`` exceeds this ion's peak LET (``"high"`` /
        ``"low"``), if ``LET - let_keV_um`` has no sign change over
        ``energy_bounds_MeV_u`` (``"bracket"``), or if ``xtol`` / ``rtol`` are given
        with a branch other than ``"bracket"``.
    """
    lo, hi = energy_bounds_MeV_u
    if not 0.0 < lo < hi:
        raise ValueError(f"energy_bounds_MeV_u must be 0 < lo < hi, got {energy_bounds_MeV_u!r}")
    if let_keV_um <= 0.0:
        raise ValueError(f"let_keV_um must be > 0, got {let_keV_um!r}")
    if branch not in ("high", "low", "bracket"):
        raise ValueError(f"branch must be 'high', 'low' or 'bracket', got {branch!r}")
    if branch == "bracket":
        return _energy_for_let_in_bracket(stopping_power, Z, A, let_keV_um, lo, hi, xtol, rtol)
    if xtol is not None or rtol is not None:
        raise ValueError(f"xtol/rtol apply only to branch='bracket', not {branch!r}")

    grid = np.logspace(math.log10(lo), math.log10(hi), 400)
    lets = np.array([stopping_power.let_keV_um(Ion(Z, A, float(e))) for e in grid])
    peak = int(np.argmax(lets))
    if let_keV_um > lets[peak]:
        raise ValueError(
            f"LET {let_keV_um} keV/um exceeds the peak LET {lets[peak]:.1f} for Z={Z} A={A}"
        )

    def excess(energy: float) -> float:
        return stopping_power.let_keV_um(Ion(Z, A, energy)) - let_keV_um

    a, b = (grid[peak], grid[-1]) if branch == "high" else (grid[0], grid[peak])
    if excess(a) * excess(b) > 0.0:
        raise ValueError(f"LET {let_keV_um} keV/um is not bracketed on the {branch} branch")
    return float(brentq(excess, a, b, xtol=1e-10, rtol=1e-12))


def _energy_for_let_in_bracket(
    stopping_power: StoppingPower,
    Z: int,
    A: int,
    let_keV_um: float,
    lo: float,
    hi: float,
    xtol: float | None,
    rtol: float | None,
) -> float:
    """``energy_for_let_MeV_u(branch="bracket")``: brentq on ``[lo, hi]`` as given.

    The endpoint values are computed once, checked for a sign change, and handed back
    to brentq when it asks for them (its first two calls), so the source sees the same
    LET evaluations as an inline brentq call -- the NS1970 hybrid tallies every lookup.
    """

    def excess(energy: float) -> float:
        return stopping_power.let_keV_um(Ion(Z, A, energy)) - let_keV_um

    f_lo, f_hi = excess(lo), excess(hi)
    if f_lo * f_hi > 0.0:  # brentq's own test (scipy/optimize/Zeros/brentq.c)
        raise ValueError(
            f"LET {let_keV_um} keV/um is not bracketed by energy_bounds_MeV_u=({lo!r}, {hi!r}) "
            f"for Z={Z} A={A}: LET there is {f_lo + let_keV_um:.6g} and "
            f"{f_hi + let_keV_um:.6g} keV/um (no sign change)"
        )

    def excess_reusing_ends(energy: float) -> float:
        if energy == lo:
            return f_lo
        if energy == hi:
            return f_hi
        return excess(energy)

    tolerances: dict[str, float] = {}
    if xtol is not None:
        tolerances["xtol"] = xtol
    if rtol is not None:
        tolerances["rtol"] = rtol
    return float(brentq(excess_reusing_ends, lo, hi, **tolerances))
