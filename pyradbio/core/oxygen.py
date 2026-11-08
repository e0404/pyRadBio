"""The oxygen axis: partial pressure and relative radioresistance.

Model-agnostic. Hypoxic cells are radioresistant because a fraction of the
initial DNA lesions is chemically repaired (~1 us) instead of being fixed by
molecular oxygen. Every OER model in the literature parameterises that with the
same **Alper & Howard-Flanders** hyperbola (doi:10.1038/178978a0), so it lives
here rather than inside any one model.

``pO2`` is an *environment* variable -- not cell geometry
([`CellTarget`][]) and not radiation quality
([`RadiationField`][]). It is therefore passed to
models as an explicit argument, the way dose is, and [`OxygenCondition`][] is
offered only as a convenience bundle for ``pO2`` with its ``K``.

Form
----

```text
R(pO2) = (pO2 + K) / (pO2 + K / Rmax)
```

``R >= 1`` is the factor by which the saturation-corrected domain specific energy
``z*_d`` must be *increased* at ``pO2`` to reach the effect it would have at the
aerobic reference (Inaniwa & Kanematsu 2021 eq. 13 / 2023 eq. 1). ``R(0) = Rmax``
and ``R -> 1`` as ``pO2 -> inf``, both exactly.

Note this is the *reciprocal* arrangement of the OER-form usually written
``(m pO2 + K)/(pO2 + K)``. The consequence is a documented prose/equation
mismatch in both OSMK papers: they describe ``K`` as the pressure at which
``R = (1 + Rmax)/2``, but their own equation puts ``R(K) = 2 Rmax/(1 + Rmax)``,
the *harmonic* midpoint. See ``docs/references/OSMK_Inaniwa2021_2023.md``.
"""

from __future__ import annotations

import math
from dataclasses import dataclass

import numpy as np
import numpy.typing as npt

ANOXIC_PO2_MMHG: float = 0.0
"""Anoxic condition [mmHg]."""

AEROBIC_PO2_MMHG: float = 160.0
"""The experimental "aerobic" condition [mmHg] -- 21% oxygen at 1 atm, the
reference oxygenation of the Furusawa 2000 and Tinganelli 2015 data OSMK is
fitted to. Note this is *not* the ``R = 1`` limit; see [`relative_radioresistance`][].
"""

NORMOXIC_TISSUE_PO2_MMHG: float = 50.0
"""Physiological tumour normoxia used in the OSMK 2021 treatment-planning study
[mmHg] (its pancreatic case assigns 50 mmHg outside the GTV, falling to
2.5 mmHg at the GTV centre; Koong et al 2000, McKeown 2014).
"""

K_HALF_EFFECT_MMHG: float = 3.0
"""Half-effect pressure ``K`` [mmHg]. Measured for cells exposed to **photons**
(Wouters & Brown 1997; Hall & Giaccia 2006) and assumed by both OSMK papers to
be independent of cell line *and* radiation quality -- "as a first
approximation", since no high-LET measurement exists (2021 sec 3.2).
"""


def relative_radioresistance(
    pO2_mmHg: npt.NDArray[np.float64] | float,
    *,
    k_mmHg: float = K_HALF_EFFECT_MMHG,
    r_max: float,
) -> npt.NDArray[np.float64]:
    """Relative radioresistance ``R(pO2) = (pO2 + K)/(pO2 + K/Rmax)`` [dimensionless].

    Parameters
    ----------
    pO2_mmHg
        Oxygen partial pressure, ``>= 0``. ``inf`` is allowed and returns 1.
    k_mmHg
        Half-effect pressure ``K``. See [`K_HALF_EFFECT_MMHG`][].
    r_max
        Maximum radioresistance, reached at ``pO2 = 0``. Must be ``>= 1``.
        In OSMK 2021 this is itself a function of radiation quality
        (``Rmax(z_d,D)``, eq. 14); in OSMK 2023 it is a cell constant.

    Returns
    -------
    ``R`` with the shape of ``pO2_mmHg`` (a 0-d array for a scalar input, which
    compares and does arithmetic as a float).
    """
    if not r_max >= 1.0:
        raise ValueError(f"r_max must be >= 1 (hypoxia cannot sensitise), got {r_max!r}")
    if not k_mmHg > 0.0:
        raise ValueError(f"k_mmHg must be > 0, got {k_mmHg!r}")
    p = np.asarray(pO2_mmHg, dtype=np.float64)
    if np.any(p < 0.0) or np.any(np.isnan(p)):
        raise ValueError("pO2_mmHg must be >= 0")
    if np.any(np.isinf(p)):  # the R -> 1 limit, which the ratio would give as nan
        r_finite = np.where(np.isinf(p), 1.0, 0.0)
        finite_p = np.where(np.isinf(p), 0.0, p)
        r_calc = (finite_p + k_mmHg) / (finite_p + k_mmHg / r_max)
        out: npt.NDArray[np.float64] = np.where(np.isinf(p), r_finite, r_calc)
        return out
    result: npt.NDArray[np.float64] = (p + k_mmHg) / (p + k_mmHg / r_max)
    return result


@dataclass(frozen=True, slots=True)
class OxygenCondition:
    """An oxygenation state: ``pO2`` plus the half-effect pressure ``K``.

    A convenience bundle. Models take ``pO2_mmHg`` as a plain argument; use this
    when carrying an oxygenation around (e.g. a per-region assignment) or when
    overriding ``K`` from its assumed 3 mmHg.
    """

    pO2_mmHg: float
    k_mmHg: float = K_HALF_EFFECT_MMHG

    def __post_init__(self) -> None:
        if self.pO2_mmHg < 0.0 or math.isnan(self.pO2_mmHg):
            raise ValueError(f"pO2_mmHg must be >= 0, got {self.pO2_mmHg!r}")
        if not self.k_mmHg > 0.0:
            raise ValueError(f"k_mmHg must be > 0, got {self.k_mmHg!r}")

    @classmethod
    def anoxic(cls) -> OxygenCondition:
        """``pO2 = 0`` -- the maximum-radioresistance limit ``R = Rmax``."""
        return cls(ANOXIC_PO2_MMHG)

    @classmethod
    def aerobic(cls) -> OxygenCondition:
        """``pO2 = 160 mmHg`` -- the *experimental* aerobic reference, where ``R`` is
        ~1.4% above unity rather than exactly 1."""
        return cls(AEROBIC_PO2_MMHG)

    def relative_radioresistance(self, r_max: float) -> float:
        """[`relative_radioresistance`][] at this condition."""
        return float(relative_radioresistance(self.pO2_mmHg, k_mmHg=self.k_mmHg, r_max=r_max))
