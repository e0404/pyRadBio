"""Model-agnostic core: LQ engine, radiation fields, photon/cell, mixing."""

from __future__ import annotations

from pyradbio.core.cell import CellTarget
from pyradbio.core.chromatin import (
    chromatin_domain_radius_um,
    domain_radius_from_genome,
    nuclear_dna_content_mbp,
    ploidy_from_chromosome_count,
)
from pyradbio.core.lq import (
    LQParameters,
    bed,
    dose_for_survival,
    eqd2,
    log_kill,
    rbe,
    survival,
)
from pyradbio.core.mixing import (
    dose_average,
    dose_average_sqrt,
    dose_quadrature,
    spectral_dose_average,
    spectral_integral,
)
from pyradbio.core.oxygen import (
    AEROBIC_PO2_MMHG,
    ANOXIC_PO2_MMHG,
    K_HALF_EFFECT_MMHG,
    NORMOXIC_TISSUE_PO2_MMHG,
    OxygenCondition,
    relative_radioresistance,
)
from pyradbio.core.photon import ClusterParameters, PhotonParameters
from pyradbio.core.protraction import (
    DoseBlock,
    DoseSchedule,
    lea_catcheside_g,
    repair_rate_per_h,
)
from pyradbio.core.radiation import (
    Ion,
    RadiationField,
    SpectrumComponent,
    dose_to_fluence,
    fluence_to_dose,
    relativistic_beta,
)

__all__ = [
    "AEROBIC_PO2_MMHG",
    "ANOXIC_PO2_MMHG",
    "K_HALF_EFFECT_MMHG",
    "NORMOXIC_TISSUE_PO2_MMHG",
    "CellTarget",
    "ClusterParameters",
    "DoseBlock",
    "DoseSchedule",
    "Ion",
    "LQParameters",
    "OxygenCondition",
    "PhotonParameters",
    "RadiationField",
    "SpectrumComponent",
    "bed",
    "chromatin_domain_radius_um",
    "domain_radius_from_genome",
    "dose_average",
    "dose_average_sqrt",
    "dose_for_survival",
    "dose_quadrature",
    "dose_to_fluence",
    "eqd2",
    "fluence_to_dose",
    "lea_catcheside_g",
    "log_kill",
    "nuclear_dna_content_mbp",
    "ploidy_from_chromosome_count",
    "rbe",
    "relative_radioresistance",
    "relativistic_beta",
    "repair_rate_per_h",
    "spectral_dose_average",
    "spectral_integral",
    "survival",
]
