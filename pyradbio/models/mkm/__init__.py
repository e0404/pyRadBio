"""MKM model family.

Modular by design: the engine-agnostic [`MKMModel`][] applies the MKM
biophysics, while the ``z*_1D`` computation lives behind the
[`MKMBackend`][] protocol. The default backend is the pyMKM-free
[`NativeMKMBackend`][] (Kiefer-Chatterjee track structure + the Kase
saturation, LET from the library-default analytic Bethe stopping power); any
other engine is a drop-in sibling:

```python
from pyradbio.models.mkm import MKMModel, NativeMKMBackend
model = MKMModel(domain_radius_um=0.32, nucleus_radius_um=3.9)  # NativeMKMBackend
model = MKMModel(backend=NativeMKMBackend(domain_radius_um=0.32))  # explicit
model = MKMModel(backend=MyBackend())                            # swap engine
```

``PyMKMBackend`` -- pyMKM as an independent cross-check, needing the optional
``[mkm]`` extra -- is imported lazily and therefore not re-exported here; import
it explicitly when validating against pyMKM:

```python
from pyradbio.models.mkm.pymkm_backend import PyMKMBackend
model = MKMModel(backend=PyMKMBackend(domain_radius_um=0.32, nucleus_radius_um=3.9))
```
"""

from __future__ import annotations

from pyradbio.models.mkm.base import (
    BackendZStarRow,
    BackendZStarTable,
    MKMAlphaBetaTable,
    MKMAlphaBetaTableRow,
    MKMBackend,
)
from pyradbio.models.mkm.imk import IMKModel, matsuya2025_h_values
from pyradbio.models.mkm.model import MKMModel
from pyradbio.models.mkm.native_backend import NativeMKMBackend
from pyradbio.models.mkm.osmk import OSMK2021, OSMK2023
from pyradbio.models.mkm.spectrum_models import (
    MCFMKMModel,
    ModifiedMKM,
    SiteSpecificEnergies,
    StochasticMKM,
    saturation_y0_keV_um,
    specific_energy_Gy,
    z_star_1D_from_dose_spectrum,
)

__all__ = [
    "OSMK2021",
    "OSMK2023",
    "BackendZStarRow",
    "BackendZStarTable",
    "IMKModel",
    "MCFMKMModel",
    "MKMAlphaBetaTable",
    "MKMAlphaBetaTableRow",
    "MKMBackend",
    "MKMModel",
    "ModifiedMKM",
    "NativeMKMBackend",
    "SiteSpecificEnergies",
    "StochasticMKM",
    "matsuya2025_h_values",
    "saturation_y0_keV_um",
    "specific_energy_Gy",
    "z_star_1D_from_dose_spectrum",
]
