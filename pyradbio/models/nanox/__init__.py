"""NanOx -- NANodosimetry and OXidative stress (Cunha et al. 2017; Beuve group, Lyon).

The modern French framework, built to repair two theoretical inconsistencies its
authors identify in the LEM and the MKM: the treatment of local versus non-local
events, and the incomplete accounting of the stochastic nature of energy
deposition. Cell survival factorises into **local lethal events** in nanometric
targets -- with no Poisson assumption, hence no "non-Poissonian" corrections --
and **global events** representing oxidative-stress accumulation, which supply
the whole shoulder.

This package implements the assembly and calibration layer: the effective local
lethal function, the effective number of local lethal events, the core/penumbra
decomposition, the Poisson impact statistics, the global-event branch, the
photon calibration ``beta_G = beta_r/eta^2``, and the track-segment closed
forms. The nanodosimetric spectrum those equations integrate is **not** shipped:
it comes from the group's non-public LQD Monte Carlo and is figure-only in the
published corpus, so :class:`NanOxModel.alpha_beta` raises
:class:`NanOxNanodosimetryRequired` unless a
:class:`NanOxNanodosimetry` backend is supplied. See
``docs/references/NanOx.md``.
"""

from pyradbio.models.nanox.base import NanOxNanodosimetry, NanOxNanodosimetryRequired
from pyradbio.models.nanox.ellf import (
    DEFAULT_LOCAL_TARGET,
    EffectiveLocalLethalFunction,
    LocalTarget,
    LocalTargetSpectrum,
)
from pyradbio.models.nanox.model import (
    CHO_K1,
    CORE_EDGE_UM,
    ETA_RESTRICTED_FRACTION,
    GY_PER_KEV_UM3,
    HSG,
    T_RCE_S,
    V79,
    NanOxCellLine,
    NanOxModel,
)

__all__ = [
    "CHO_K1",
    "CORE_EDGE_UM",
    "DEFAULT_LOCAL_TARGET",
    "ETA_RESTRICTED_FRACTION",
    "GY_PER_KEV_UM3",
    "HSG",
    "T_RCE_S",
    "V79",
    "EffectiveLocalLethalFunction",
    "LocalTarget",
    "LocalTargetSpectrum",
    "NanOxCellLine",
    "NanOxModel",
    "NanOxNanodosimetry",
    "NanOxNanodosimetryRequired",
]
