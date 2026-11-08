"""Microdosimetry and analytic sub-cellular damage yields.

Analytic single-event lineal-energy spectra ``f1(y)`` / ``d1(y)`` for ions in
water, built on the shared ``Ion`` and ``StoppingPower`` seams. See
[`pyradbio.micro.lineal_energy`][] for the physics and assumptions.

[`pyradbio.micro.mcds`][] sits alongside them: the closed-form MCDS DNA-damage
yields (Semenenko & Stewart 2006), which are a function of the same track
quality but score *lesions* rather than energy deposition. It is the DSB input
to the RMF model and is deliberately kept here rather than under ``models/``,
because nothing about it is specific to RMF.
"""

from __future__ import annotations

from pyradbio.micro.impact_parameter import (
    NuclearImpactProfile,
    SpecificEnergyProfile,
    disk_arc_weight,
    disk_area_integral_over_site,
    impact_parameter_grid,
    nuclear_functional_impact_profile,
    specific_energy_impact_profile,
)
from pyradbio.micro.library import (
    DeCunhaLinealEnergyLibrary,
    load_decunha_lineal_energy_csv,
)
from pyradbio.micro.lineal_energy import (
    AmorphousTrackEngine,
    AnalyticSpectrumEngine,
    CylindricalSite,
    DeltaRayMonteCarlo,
    LinealEnergyCalculator,
    SiteGeometry,
    Spectrum,
    SpectrumEngine,
    SphericalSite,
    TabulatedLinealEnergySpectrum,
    max_delta_ray_energy_keV,
)
from pyradbio.micro.mcds import (
    GBP_PER_CELL,
    MCDS_DOMAIN_MAX,
    MCDS_DOMAIN_MIN,
    MCDSYields,
    effective_charge_barkas,
    quality_parameter,
)
from pyradbio.micro.sato import SatoAmfCoefficients, SatoLinealEnergyEngine

__all__ = [
    "GBP_PER_CELL",
    "MCDS_DOMAIN_MAX",
    "MCDS_DOMAIN_MIN",
    "AmorphousTrackEngine",
    "AnalyticSpectrumEngine",
    "CylindricalSite",
    "DeCunhaLinealEnergyLibrary",
    "DeltaRayMonteCarlo",
    "LinealEnergyCalculator",
    "MCDSYields",
    "NuclearImpactProfile",
    "SatoAmfCoefficients",
    "SatoLinealEnergyEngine",
    "SiteGeometry",
    "SpecificEnergyProfile",
    "Spectrum",
    "SpectrumEngine",
    "SphericalSite",
    "TabulatedLinealEnergySpectrum",
    "disk_arc_weight",
    "disk_area_integral_over_site",
    "effective_charge_barkas",
    "impact_parameter_grid",
    "load_decunha_lineal_energy_csv",
    "max_delta_ray_energy_keV",
    "nuclear_functional_impact_profile",
    "quality_parameter",
    "specific_energy_impact_profile",
]
