"""Radiation fields: single ions and mixed spectra."""

from __future__ import annotations

import math
from collections.abc import Callable, Sequence
from dataclasses import dataclass

from pyradbio.core.constants import (
    ATOMIC_MASS_UNIT_MEV,
    FLUENCE_LET_TO_DOSE_GY,
    WATER_DENSITY_G_CM3,
)


def relativistic_beta(energy_MeV_u: float) -> float:
    """Speed ``v/c`` of a nucleon with kinetic energy ``energy_MeV_u`` [MeV/u].

    From the relativistic energy ``gamma = 1 + T/(u c^2)`` with ``u c^2`` the
    atomic-mass-unit energy (used for r_min(beta) in LEM III and the Bethe
    stopping power).
    """
    if energy_MeV_u <= 0.0:
        raise ValueError("energy_MeV_u must be > 0")
    gamma = 1.0 + energy_MeV_u / ATOMIC_MASS_UNIT_MEV
    return math.sqrt(1.0 - 1.0 / (gamma * gamma))


def fluence_to_dose(
    fluence_cm2: float, let_keV_um: float, density_g_cm3: float = WATER_DENSITY_G_CM3
) -> float:
    """Absorbed dose [Gy] from a particle fluence [cm^-2] and its LET [keV/um]."""
    return FLUENCE_LET_TO_DOSE_GY * fluence_cm2 * let_keV_um / density_g_cm3


def dose_to_fluence(
    dose_Gy: float, let_keV_um: float, density_g_cm3: float = WATER_DENSITY_G_CM3
) -> float:
    """Particle fluence [cm^-2] needed to deposit ``dose_Gy`` at a given LET."""
    if let_keV_um <= 0.0:
        raise ValueError("LET must be > 0 to convert dose to fluence")
    return dose_Gy * density_g_cm3 / (FLUENCE_LET_TO_DOSE_GY * let_keV_um)


@dataclass(frozen=True, slots=True)
class Ion:
    """A mono-energetic ion. Energy is kinetic energy per nucleon [MeV/u]."""

    Z: int
    A: int
    energy_MeV_u: float

    def __post_init__(self) -> None:
        if self.Z < 1 or self.A < 1:
            raise ValueError("Z and A must be >= 1")
        if self.energy_MeV_u <= 0.0:
            raise ValueError("energy_MeV_u must be > 0")

    def beta(self) -> float:
        """Relativistic speed ``v/c`` from the kinetic energy per nucleon."""
        return relativistic_beta(self.energy_MeV_u)


@dataclass(frozen=True, slots=True)
class SpectrumComponent:
    """One component of a mixed field.

    Exactly one of ``fluence`` (particles/cm^2) or ``dose_Gy`` is given; the
    other is derived via a stopping-power lookup (see RadiationField).
    """

    ion: Ion
    fluence: float | None = None
    dose_Gy: float | None = None
    energy_lo_MeV_u: float | None = None
    energy_hi_MeV_u: float | None = None

    def __post_init__(self) -> None:
        if (self.fluence is None) == (self.dose_Gy is None):
            raise ValueError("provide exactly one of fluence or dose_Gy")
        if self.fluence is not None and self.fluence < 0.0:
            raise ValueError("fluence must be >= 0")
        if self.dose_Gy is not None and self.dose_Gy < 0.0:
            raise ValueError("dose_Gy must be >= 0")
        has_lo = self.energy_lo_MeV_u is not None
        has_hi = self.energy_hi_MeV_u is not None
        if has_lo != has_hi:
            raise ValueError("provide both energy_lo_MeV_u and energy_hi_MeV_u for a bin")
        if has_lo and has_hi:
            assert self.energy_lo_MeV_u is not None
            assert self.energy_hi_MeV_u is not None
            if self.energy_lo_MeV_u <= 0.0 or self.energy_hi_MeV_u <= 0.0:
                raise ValueError("energy bin bounds must be > 0")
            if self.energy_hi_MeV_u <= self.energy_lo_MeV_u:
                raise ValueError("energy_hi_MeV_u must exceed energy_lo_MeV_u")
            if not self.energy_lo_MeV_u <= self.ion.energy_MeV_u <= self.energy_hi_MeV_u:
                raise ValueError("ion.energy_MeV_u must lie inside the energy bin")

    @property
    def is_energy_bin(self) -> bool:
        """Whether this component represents a finite energy bin."""
        return self.energy_lo_MeV_u is not None

    def energy_bounds_MeV_u(self) -> tuple[float, float] | None:
        """Finite bin support ``(E_lo, E_hi)`` [MeV/u], or ``None`` for mono."""
        if self.energy_lo_MeV_u is None:
            return None
        assert self.energy_hi_MeV_u is not None
        return (self.energy_lo_MeV_u, self.energy_hi_MeV_u)

    def absorbed_dose_Gy(self, stopping_power: Callable[[Ion], float]) -> float:
        """Resolve this component's absorbed dose [Gy].

        Uses ``dose_Gy`` directly when given, else converts the fluence via the
        ion's LET from ``stopping_power`` (keV/um). For an energy bin the fluence is
        spread uniformly over the bin and the dose integrated on the same sub-grid the
        models mix over ([`dose_quadrature`][pyradbio.core.mixing.dose_quadrature]),
        never converted at the representative energy.
        """
        if self.dose_Gy is not None:
            return self.dose_Gy
        assert self.fluence is not None  # guaranteed by __post_init__
        if self.is_energy_bin:
            # Imported here: core.mixing builds on this module.
            from pyradbio.core.mixing import dose_quadrature

            nodes = dose_quadrature(RadiationField([self]), stopping_power)
            return math.fsum(dose for _ion, dose in nodes)
        return fluence_to_dose(self.fluence, stopping_power(self.ion))


@dataclass(frozen=True, slots=True)
class RadiationField:
    """A single ion or a list of spectrum components at a point."""

    components: Sequence[SpectrumComponent]

    def __post_init__(self) -> None:
        if not self.components:
            raise ValueError("a RadiationField needs at least one component")

    @classmethod
    def mono(cls, ion: Ion, dose_Gy: float) -> RadiationField:
        """A single mono-energetic ion delivering ``dose_Gy``."""
        return cls([SpectrumComponent(ion, dose_Gy=dose_Gy)])

    @property
    def is_mixed(self) -> bool:
        return len(self.components) > 1

    def dose_weights(self, stopping_power: Callable[[Ion], float]) -> list[float]:
        """Per-component absorbed doses [Gy] (see [`SpectrumComponent.absorbed_dose_Gy`][])."""
        return [c.absorbed_dose_Gy(stopping_power) for c in self.components]

    def total_dose_Gy(self, stopping_power: Callable[[Ion], float]) -> float:
        """Total absorbed dose summed over all components."""
        return sum(self.dose_weights(stopping_power))
