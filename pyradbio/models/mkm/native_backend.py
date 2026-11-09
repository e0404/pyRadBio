"""Native, pyMKM-free MKM engine (Kiefer-Chatterjee track structure).

The default [`MKMBackend`][] of
[`MKMModel`][]; it needs no optional extra.
[`PyMKMBackend`][pyradbio.models.mkm.pymkm_backend.PyMKMBackend] is its pyMKM-backed
cross-check. It reuses pyRadBio's own pieces:

* the **analytic impact-parameter integral** of an amorphous
  [`RadialDoseModel`][] over the **MKM domain disk**
  ([`specific_energy_impact_profile`][pyradbio.micro.impact_parameter.specific_energy_impact_profile],
  ``geometry="cylinder"`` -- pyMKM's own cylinder-perpendicular-to-track
  geometry) -- by default the **Kiefer-Chatterjee** track, the same track
  structure pyMKM uses. This is deterministic (no delta-ray Monte Carlo, no
  straggling model) and robust across the whole energy range;
* the Kase 2006 / Inaniwa 2010 saturation
  ([`saturation_y0_keV_um`][] + the profile's
  saturation-corrected dose-mean) -- the same ``y -> z*`` math as
  [`ModifiedMKM`][].

Accuracy: with the (correct) domain-disk geometry ``z*_1D`` tracks pyMKM within
~1-6% across ions (H-O) and energies (1-400 MeV/u). The saturation parameter
``z0 = (R_n/r_d)^2 / sqrt(beta0 (1 + (R_n/r_d)^2))`` and the square-root dose-average
match pyMKM's ``SpecificEnergy`` exactly, so the only residual is the amorphous
Kiefer-Chatterjee track vs pyMKM's own. (A spherical domain instead overshoots
``z*_1D`` by +25-40% at high energy -- the geometry, not the track model, was the
discrepancy.) The exact Inaniwa golden values remain a ``@requires_pymkm`` check on
[`PyMKMBackend`][pyradbio.models.mkm.pymkm_backend.PyMKMBackend].

Pass ``engine=<SpectrumEngine>`` to instead compute ``z*_1D`` from a single-event
lineal-energy spectrum (analytic straggling or the amorphous + delta-ray Monte
Carlo engines) -- less robust at low LET, kept for comparison.
"""

from __future__ import annotations

from collections.abc import Sequence
from dataclasses import dataclass, field

from pyradbio.core.radiation import Ion
from pyradbio.micro import SpectrumEngine, specific_energy_impact_profile
from pyradbio.models.mkm.base import BackendZStarRow, BackendZStarTable
from pyradbio.models.mkm.spectrum_models import (
    saturation_y0_keV_um,
    specific_energy_Gy,
    z_star_1D_from_dose_spectrum,
)
from pyradbio.track.base import RadialDoseModel
from pyradbio.track.kiefer_chatterjee import KieferChatterjeeRadialDose
from pyradbio.track.stopping_power import StoppingPower, default_stopping_power


@dataclass(frozen=True, slots=True)
class NativeMKMBackend:
    """MKM engine computing ``z*_1D`` from Kiefer-Chatterjee track structure.

    Satisfies [`MKMBackend`][]. Geometry and the
    saturation parameter ``beta0`` are fixed at construction (as in
    [`PyMKMBackend`][pyradbio.models.mkm.pymkm_backend.PyMKMBackend]); the methods depend
    only on the ion. Defaults are Inaniwa 2010 HSG geometry with a
    Kiefer-Chatterjee radial dose evaluated by analytic impact-parameter integration.
    """

    domain_radius_um: float = 0.32
    nucleus_radius_um: float = 3.9
    beta0: float = 0.0615  # Gy^-2, enters the saturation correction (Inaniwa 2010 eq. 5)
    stopping_power: StoppingPower = field(default_factory=default_stopping_power)
    radial_dose: RadialDoseModel | None = None
    engine: SpectrumEngine | None = None

    def _radial_dose(self) -> RadialDoseModel:
        if self.radial_dose is not None:
            return self.radial_dose
        return KieferChatterjeeRadialDose(self.stopping_power)

    def _z0_Gy(self) -> float:
        y0 = saturation_y0_keV_um(
            domain_radius_um=self.domain_radius_um,
            nucleus_radius_um=self.nucleus_radius_um,
            beta_Gy2=self.beta0,
        )
        return float(specific_energy_Gy(y0, self.domain_radius_um))

    def z_star_1D_Gy(self, ion: Ion) -> float:
        if self.engine is not None:
            spectrum = self.engine.single_event_spectrum(ion)
            return z_star_1D_from_dose_spectrum(
                spectrum.y_keV_um,
                spectrum.d,
                domain_radius_um=self.domain_radius_um,
                nucleus_radius_um=self.nucleus_radius_um,
                beta_Gy2=self.beta0,
            )
        # High node counts: at high energy r_max spans many decades, so the domain
        # core and the far penumbra both need resolving for z*_1D to converge.
        profile = specific_energy_impact_profile(
            self._radial_dose(),
            ion,
            region_radius_um=self.domain_radius_um,
            geometry="cylinder",
            radial_nodes=700,
            core_nodes=600,
            penumbra_nodes=600,
        )
        return profile.saturation_corrected_dose_mean_Gy(self._z0_Gy())

    def let_keV_um(self, ion: Ion) -> float:
        """Unrestricted LET [keV/um] from this engine's ``stopping_power``.

        Part of the [`MKMBackend`][] protocol: the model uses it for the
        fluence -> dose weights of mixed fields.
        """
        return self.stopping_power.let_keV_um(ion)

    def z_star_1D_table_Gy(
        self, ions: Sequence[int], energies_MeV_u: Sequence[float]
    ) -> BackendZStarTable:
        """``z*_1D`` over an (ion Z, energy) grid as dependency-free rows."""
        rows: list[BackendZStarRow] = []
        for z in dict.fromkeys(int(i) for i in ions):
            for energy in energies_MeV_u:
                ion = Ion(Z=z, A=_mass_number(z), energy_MeV_u=float(energy))
                rows.append(
                    BackendZStarRow(
                        ion_Z=z,
                        energy_MeV_u=float(energy),
                        let_keV_um=self.stopping_power.let_keV_um(ion),
                        z_star_1D_Gy=self.z_star_1D_Gy(ion),
                    )
                )
        return BackendZStarTable(rows)


def _mass_number(z: int) -> int:
    """Default mass number for a Z-only table request (most-abundant isotope)."""
    from pyradbio.core.elements import default_mass_number

    return default_mass_number(z)
