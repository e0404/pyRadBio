"""pyRadBio: radiobiological modelling for ion-beam therapy.

Centred on the linear-quadratic model ``-ln S(D) = alpha*D + beta*D**2``.

"""

from __future__ import annotations

from importlib.metadata import PackageNotFoundError, version

try:
    #: The installed distribution's version: ``pyproject.toml`` is the single source.
    __version__ = version("pyradbio")
except PackageNotFoundError:  # pragma: no cover - running from an uninstalled tree
    __version__ = "0+unknown"

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
from pyradbio.core.photon import PhotonParameters
from pyradbio.core.protraction import DoseBlock, DoseSchedule, lea_catcheside_g
from pyradbio.core.radiation import Ion, RadiationField, SpectrumComponent

__all__ = [
    "CellTarget",
    "DoseBlock",
    "DoseSchedule",
    "Ion",
    "LQParameters",
    "PhotonParameters",
    "RadiationField",
    "SpectrumComponent",
    "bed",
    "chromatin_domain_radius_um",
    "domain_radius_from_genome",
    "dose_for_survival",
    "eqd2",
    "lea_catcheside_g",
    "log_kill",
    "nuclear_dna_content_mbp",
    "ploidy_from_chromosome_count",
    "rbe",
    "survival",
]

# Models are imported lazily by name to keep optional deps (pymkm) out of the
# import path until requested. See pyradbio.models.
