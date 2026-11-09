"""ICRU 49's empirical low-energy stopping of protons and alphas in water (Varelas-Biersack).

ICRU Report 49 (1993), sec. 3.3/3.4. Stopping cross section ``eps`` in
``1e-15 eV cm^2`` per molecule:

* protons, ``T_s = T[keV] / (M_p/u)`` (eqs. 3.1, 3.2):

```text
eps = eps_low eps_high / (eps_low + eps_high)                    T_s >= 10 keV
eps_low = A2 T_s^0.45,  eps_high = (A3/T_s) ln(1 + A4/T_s + A5 T_s)
eps = A1 T_s^0.5                                                  1-10 keV
```

* alphas, ``T`` = alpha kinetic energy in MeV, 1 keV to 10 MeV (eq. 3.3):

```text
eps_low = a1 (1000 T)^a2,  eps_high = (a3/T) ln(1 + a4/T + a5 T)
```

ICRU 49 used these, with water's own refitted coefficients (ID 276, Tables 3.1 and 3.5),
below its cut-off ``T1`` and joined them to Bethe above ``T2`` (Table 3.7) by a cubic
spline of ``beta^2 S`` vs ``log T`` (sec. 3.5); ICRU 90 kept them unchanged (its p/alpha
tables equal ICRU 49's below ``T1``, measured).

This module holds only the formulae; the coefficients are raw data
(``data/stopping/icru49/PROVENANCE.md``). ``scripts/stopping/build_tables.py`` evaluates
them onto our grid and ships ``data/icru49_empirical_water.csv``, which
[`ICRU90Procedure`][] reads.
"""

from __future__ import annotations

from collections.abc import Mapping

import numpy as np
import numpy.typing as npt

from pyradbio.core.constants import AVOGADRO_PER_MOL

FloatArray = npt.NDArray[np.float64]

ICRU49_EMPIRICAL_REFERENCE = (
    "ICRU Report 49 (1993), Stopping Powers and Ranges for Protons and Alpha Particles, "
    "eqs. (3.1)-(3.3), Tables 3.1, 3.5 and 3.7 (liquid water, ID 276)"
)
SHIPPED_STEM = "icru49_empirical_water"


def mass_stopping_from_cross_section(eps: npt.ArrayLike, molecular_weight: float) -> FloatArray:
    """``eps`` [1e-15 eV cm^2 / molecule] -> mass stopping power [MeV cm^2/g]."""
    out: FloatArray = (
        np.asarray(eps, dtype=np.float64) * 1e-21 * AVOGADRO_PER_MOL / molecular_weight
    )
    return out


def proton_cross_section(T_MeV: npt.ArrayLike, c: Mapping[str, float]) -> FloatArray:
    """ICRU 49 eqs. (3.1), (3.2); ``c`` holds A1-A5 and ``proton_mass_u``."""
    ts = np.asarray(T_MeV, dtype=np.float64) * 1e3 / c["proton_mass_u"]
    low = c["A2"] * ts**0.45
    high = (c["A3"] / ts) * np.log(1.0 + c["A4"] / ts + c["A5"] * ts)
    out: FloatArray = np.where(ts >= 10.0, low * high / (low + high), c["A1"] * np.sqrt(ts))
    return out


def alpha_cross_section(T_MeV: npt.ArrayLike, c: Mapping[str, float]) -> FloatArray:
    """ICRU 49 eq. (3.3); ``T`` is the alpha kinetic energy [MeV]; ``c`` holds a1-a5."""
    t = np.asarray(T_MeV, dtype=np.float64)
    low = c["a1"] * (1000.0 * t) ** c["a2"]
    high = (c["a3"] / t) * np.log(1.0 + c["a4"] / t + c["a5"] * t)
    out: FloatArray = low * high / (low + high)
    return out


__all__ = [
    "ICRU49_EMPIRICAL_REFERENCE",
    "SHIPPED_STEM",
    "alpha_cross_section",
    "mass_stopping_from_cross_section",
    "proton_cross_section",
]
