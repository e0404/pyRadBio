"""Repair-Misrepair-Fixation (RMF) model, with MCDS DSB yields.

Carlson et al. 2008 (doi:10.1667/RR1046.1) and Frese et al. 2012
(doi:10.1016/j.ijrobp.2011.06.1983). The one model family in this package whose
*repair kinetics* are an independent lineage from LEM/MKM: cell killing is
linked to DSB induction and pairwise (binary) misrepair rather than to a local
dose or a specific energy.

Its DSB yields come from :mod:`pyradbio.micro.mcds`, in closed form -- no Monte
Carlo. Note the track-quality parameterisation underneath is Katz's
``Z_eff^2/beta^2``, so the independence is in the biology, not the track physics.
"""

from pyradbio.models.rmf.model import (
    CO60_REFERENCE,
    RMF_LET_CEILING_KEV_UM,
    X_RAY_200KVP_REFERENCE,
    Z_F_PREFACTOR,
    RMFAlphaTerms,
    RMFModel,
    RMFReferenceRadiation,
)

__all__ = [
    "CO60_REFERENCE",
    "RMF_LET_CEILING_KEV_UM",
    "X_RAY_200KVP_REFERENCE",
    "Z_F_PREFACTOR",
    "RMFAlphaTerms",
    "RMFModel",
    "RMFReferenceRadiation",
]
