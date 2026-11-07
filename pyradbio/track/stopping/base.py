"""Common interface of the water stopping-power classes.

Every class in :mod:`pyradbio.track.stopping` returns the **mass** stopping power
of liquid water in ``MeV cm^2/g`` as a function of the projectile's kinetic
energy per nucleon in ``MeV/u``, for a given ion ``Z`` and mass number ``A``:

* :meth:`MassStoppingPowerModel.mass_stopping_power` -- the values, always an array;
* :meth:`MassStoppingPowerModel.energy_range` -- where the source is valid;
* :meth:`MassStoppingPowerModel.metadata` -- an :class:`IonSourceInfo` naming the
  source, edition, I-value, range and options *per ion* (one table may mix sources);
* :meth:`MassStoppingPowerModel.let_keV_um` / ``lets_keV_um`` -- the adapter onto
  the package-wide :class:`~pyradbio.track.stopping_power.StoppingPower` protocol, so
  every class can be passed to any model's ``stopping_power`` field.

Requests outside :meth:`~MassStoppingPowerModel.energy_range` raise
:class:`OutOfRangeError`; nothing is extrapolated silently.

Conventions
-----------
* **Energy axis.** ``MeV/u`` is the kinetic energy *per nucleon*, ``T/A``,
  the convention of :class:`~pyradbio.core.radiation.Ion`. A proton's ``MeV/u`` is
  therefore plain ``MeV``, and converting a NIST total kinetic energy is an exact
  division by ``A`` (:func:`energy_per_nucleon_MeV_u`). Users may read "MeV/u" as
  energy per atomic mass unit instead; the two differ by ``m/(A u)`` (0.7% for
  protons), which is why the convention is fixed here and stated on every table.
  Wherever velocity enters the physics (Bethe, velocity scaling between ions) it
  is computed from the true ion mass with :func:`beta_from_energy_per_nucleon`,
  never from ``T/A`` alone.
* **Isotope masses.** Atomic masses come from the NIST "Atomic Weights and
  Isotopic Compositions" database (fetched and cited, see
  ``data/stopping/nist_isotopes/PROVENANCE.md``; the maintainer confirmed it may be
  reprinted); nuclear mass = atomic mass minus ``Z`` electron masses (electron
  binding is negligible here).
* **Interpolation.** Tabulated sources are interpolated with a not-a-knot
  cubic spline in log-log space; PCHIP is available for comparison
  (:mod:`pyradbio.track.stopping.grid`).
* **Computed sources.** Sources with a per-energy formalism we implement
  (MSTAR, the ICRU 90 procedure) evaluate it on the user's grid and interpolate
  between nodes by default; full per-energy evaluation is an opt-in option. The
  plain analytic Bethe is always evaluated directly.
* **Material.** Liquid water only (:data:`LIQUID_WATER`), kept as an explicit
  field so other targets can be added later without changing the interface.
"""

from __future__ import annotations

import math
from abc import ABC, abstractmethod
from collections.abc import Mapping, Sequence
from dataclasses import dataclass, field, fields, is_dataclass
from types import MappingProxyType
from typing import Any, ClassVar

import numpy as np
import numpy.typing as npt

from pyradbio.core.constants import MEV_PER_CM_TO_KEV_PER_UM, WATER_DENSITY_G_CM3
from pyradbio.core.radiation import Ion

#: The one target material supported so far.
LIQUID_WATER = "liquid water"

#: Option values a table records in its metadata (JSON-serialisable scalars or mappings).
OptionValue = str | float | int | bool | None | Mapping[str, Any]


class OutOfRangeError(ValueError):
    """An energy outside the range a source is valid for; nothing is extrapolated.

    A :class:`ValueError` subclass, so existing callers that catch ``ValueError``
    (e.g. the NS1970 + Bethe-tail hybrid) keep working.
    """

    def __init__(
        self,
        requested_MeV_u: tuple[float, float],
        valid_range_MeV_u: tuple[float, float],
        Z: int | None = None,
        A: int | None = None,
        source: str | None = None,
    ) -> None:
        self.requested_MeV_u = requested_MeV_u
        self.valid_range_MeV_u = valid_range_MeV_u
        self.Z = Z
        self.A = A
        self.source = source
        ion = "" if Z is None else f" for Z={Z}" + ("" if A is None else f" A={A}")
        src = "" if source is None else f" ({source})"
        lo, hi = valid_range_MeV_u
        rlo, rhi = requested_MeV_u
        super().__init__(
            f"energy [{rlo:.6g}, {rhi:.6g}] MeV/u outside the valid range "
            f"[{lo:.6g}, {hi:.6g}] MeV/u{ion}{src}; stopping powers are not extrapolated"
        )


class UnsupportedIonError(ValueError):
    """An ion (``Z``) a source does not cover."""


@dataclass(frozen=True)
class IonSourceInfo:
    """Where one ion's stopping power comes from, and under which options.

    ``i_value_eV`` is the mean excitation energy of water the source used (``None``
    only if the source states none); ``options`` records every option that changes
    the numbers (quantity, ``mstar_mode``, edition-specific switches, ...).
    """

    Z: int
    A: int
    source: str
    reference: str
    i_value_eV: float | None
    energy_range_MeV_u: tuple[float, float]
    material: str = LIQUID_WATER
    edition: str | None = None
    options: Mapping[str, OptionValue] = field(default_factory=dict)

    def __post_init__(self) -> None:
        lo, hi = self.energy_range_MeV_u
        if not 0.0 < lo < hi:
            raise ValueError(f"energy_range_MeV_u must satisfy 0 < lo < hi, got {(lo, hi)!r}")
        object.__setattr__(self, "energy_range_MeV_u", (float(lo), float(hi)))
        object.__setattr__(self, "options", MappingProxyType(dict(self.options)))

    def to_dict(self) -> dict[str, Any]:
        """A JSON-serialisable description (the inverse of :meth:`from_dict`)."""
        return {
            "Z": self.Z,
            "A": self.A,
            "source": self.source,
            "reference": self.reference,
            "i_value_eV": self.i_value_eV,
            "energy_range_MeV_u": list(self.energy_range_MeV_u),
            "material": self.material,
            "edition": self.edition,
            "options": dict(self.options),
        }

    @classmethod
    def from_dict(cls, data: Mapping[str, Any]) -> IonSourceInfo:
        lo, hi = data["energy_range_MeV_u"]
        i_value = data["i_value_eV"]
        return cls(
            Z=int(data["Z"]),
            A=int(data["A"]),
            source=str(data["source"]),
            reference=str(data["reference"]),
            i_value_eV=None if i_value is None else float(i_value),
            energy_range_MeV_u=(float(lo), float(hi)),
            material=str(data.get("material", LIQUID_WATER)),
            edition=data.get("edition"),
            options=dict(data.get("options", {})),
        )


class MassStoppingPowerModel(ABC):
    """Base class of every water stopping-power source in this package.

    Subclasses implement :meth:`supported_Z`, :meth:`energy_range`,
    :meth:`metadata` and :meth:`_evaluate`; the base class owns input validation,
    the range check (so no subclass can extrapolate by accident) and the LET
    adapter.
    """

    #: Short name stamped into checkpoints and CLI choices.
    name: ClassVar[str]
    #: Target density used only to convert mass stopping power to LET [g/cm^3].
    density_g_cm3: float = WATER_DENSITY_G_CM3

    def settings(self) -> dict[str, object]:
        """The resolved options this source computes with, JSON-safe (table provenance).

        Every public instance attribute, i.e. each constructor option after defaults
        are resolved (the ICRU edition, I-value, corrections, effective charge...);
        private caches are left out. A nested source reports its own settings.
        """
        out: dict[str, object] = {"name": self.name}
        for key, value in vars(self).items():
            if not key.startswith("_"):
                out[key] = setting_value(value)
        return out

    @abstractmethod
    def supported_Z(self) -> tuple[int, ...]:
        """The atomic numbers this source covers."""

    @abstractmethod
    def energy_range(self, Z: int, A: int) -> tuple[float, float]:
        """``(lo, hi)`` in MeV/u where this source is valid for the ion."""

    @abstractmethod
    def metadata(self, Z: int, A: int) -> IonSourceInfo:
        """Source, edition, I-value, range and options for the ion."""

    @abstractmethod
    def _evaluate(
        self, Z: int, A: int, energy_MeV_u: npt.NDArray[np.float64]
    ) -> npt.NDArray[np.float64]:
        """Mass stopping power [MeV cm^2/g] at validated, in-range energies."""

    def _require_supported(self, Z: int) -> None:
        if Z not in self.supported_Z():
            raise UnsupportedIonError(
                f"{type(self).__name__} has no data for Z={Z}; it covers Z in "
                f"{list(self.supported_Z())}"
            )

    def mass_stopping_power(
        self, Z: int, A: int, energy_MeV_u: float | Sequence[float] | npt.NDArray[np.float64]
    ) -> npt.NDArray[np.float64]:
        """Mass stopping power of water [MeV cm^2/g] at ``energy_MeV_u`` (T/A) [MeV/u].

        :raises UnsupportedIonError: if the source does not cover ``Z``.
        :raises OutOfRangeError: if any energy lies outside :meth:`energy_range`.
        """
        energies = np.atleast_1d(np.asarray(energy_MeV_u, dtype=np.float64))
        if energies.ndim != 1:
            raise ValueError("energy_MeV_u must be a scalar or a one-dimensional sequence")
        if not np.all(np.isfinite(energies) & (energies > 0.0)):
            raise ValueError("energies must be finite and positive [MeV/u]")
        self._require_supported(Z)
        lo, hi = self.energy_range(Z, A)
        if energies.min() < lo or energies.max() > hi:
            raise OutOfRangeError(
                (float(energies.min()), float(energies.max())), (lo, hi), Z, A, self.name
            )
        return np.asarray(self._evaluate(Z, A, energies), dtype=np.float64)

    def lets_keV_um(
        self, Z: int, A: int, energies_MeV_u: Sequence[float]
    ) -> npt.NDArray[np.float64]:
        """LET [keV/um] at :attr:`density_g_cm3` (``VectorizedStoppingPower``)."""
        mass_sp = self.mass_stopping_power(Z, A, np.asarray(energies_MeV_u, dtype=np.float64))
        return mass_sp * self.density_g_cm3 * MEV_PER_CM_TO_KEV_PER_UM

    def let_keV_um(self, ion: Ion) -> float:
        """LET [keV/um] (the package-wide ``StoppingPower`` protocol)."""
        return float(self.lets_keV_um(ion.Z, ion.A, [ion.energy_MeV_u])[0])


def setting_value(value: object) -> object:
    """JSON-safe form of one option for :meth:`MassStoppingPowerModel.settings`."""
    if value is None or isinstance(value, bool | int | float | str):
        return value
    if isinstance(value, Mapping):
        return {str(k): setting_value(v) for k, v in value.items()}
    if isinstance(value, list | tuple):
        return [setting_value(v) for v in value]
    for method in ("settings", "to_dict", "describe"):  # a nested source, grid, correction
        found = getattr(value, method, None)
        if callable(found):
            return setting_value(found())
    name = getattr(value, "name", None)
    if is_dataclass(value) and not isinstance(value, type):
        # Effective-charge and density-effect models: their name AND every scalar
        # parameter (e.g. Brandt-Kitagawa's band gap); array fields (tabulated values)
        # stay out, their source names them.
        out: dict[str, object] = {} if not isinstance(name, str) else {"name": name}
        for f in fields(value):
            field_value = getattr(value, f.name)
            if not f.name.startswith("_") and (
                field_value is None or isinstance(field_value, bool | int | float | str)
            ):
                out[f.name] = field_value
        return out
    return name if isinstance(name, str) else type(value).__name__


def energy_per_nucleon_MeV_u(total_kinetic_energy_MeV: float, A: int) -> float:
    """Kinetic energy per nucleon ``T/A`` [MeV/u] from the total kinetic energy ``T`` [MeV].

    NIST PSTAR and ASTAR tabulate ``T`` of the whole proton / He-4 ion.
    """
    if A < 1:
        raise ValueError(f"mass number A must be >= 1, got {A}")
    return total_kinetic_energy_MeV / A


def total_kinetic_energy_MeV(energy_MeV_u: float, A: int) -> float:
    """Total kinetic energy ``T = A * (T/A)`` [MeV]; inverse of :func:`energy_per_nucleon_MeV_u`."""
    if A < 1:
        raise ValueError(f"mass number A must be >= 1, got {A}")
    return energy_MeV_u * A


def beta_from_energy_per_nucleon(energy_MeV_u: float, A: int, ion_mass_MeV: float) -> float:
    """Speed ``v/c`` of an ion with kinetic energy per nucleon ``energy_MeV_u`` (T/A).

    ``gamma = 1 + T / (M c^2)`` with ``T = A * energy_MeV_u`` and ``M c^2`` the ion's
    rest energy ``ion_mass_MeV`` (nuclear mass for a bare ion). With
    ``ion_mass_MeV = A * u c^2`` this reduces to
    :func:`pyradbio.core.radiation.relativistic_beta`, which assumes exactly that.
    """
    if energy_MeV_u <= 0.0 or not math.isfinite(energy_MeV_u):
        raise ValueError(f"energy_MeV_u must be finite and > 0, got {energy_MeV_u!r}")
    if ion_mass_MeV <= 0.0:
        raise ValueError(f"ion_mass_MeV must be > 0, got {ion_mass_MeV!r}")
    gamma = 1.0 + total_kinetic_energy_MeV(energy_MeV_u, A) / ion_mass_MeV
    return math.sqrt(1.0 - 1.0 / (gamma * gamma))


__all__ = [
    "LIQUID_WATER",
    "IonSourceInfo",
    "MassStoppingPowerModel",
    "OptionValue",
    "OutOfRangeError",
    "UnsupportedIonError",
    "beta_from_energy_per_nucleon",
    "energy_per_nucleon_MeV_u",
    "setting_value",
    "total_kinetic_energy_MeV",
]
