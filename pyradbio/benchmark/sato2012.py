"""Sato & Furusawa 2012 fig. A1 -- the three survival conventions.

Source paper:

Sato, T. & Furusawa, Y. 2012, "Cell Survival Fraction Estimation Based on the
Probability Densities of Domain and Cell Nucleus Specific Energies Using Improved
Microdosimetric Kinetic Models", Radiat. Res. **178** 341-356,
doi:10.1667/RR2842.1.

Why this figure and not the others
----------------------------------
Figs. 6-8 of the same paper compare model and measurement for HSG and V79 cells,
but their radiation fields come from **PHITS simulations of a specific
experimental setup** (its fig. 1: a lead scatterer, air, and a PMMA degrader
reproducing Furusawa 2000's geometry) including secondaries from nuclear
interactions. Those spectra cannot be reproduced here, so any comparison against
them carries an unquantifiable offset.

Fig. A1 has none of that. It is a **fully specified numerical experiment**:
``alpha = 0.5 Gy^-1``, ``beta = 0.06 Gy^-2``, nucleus radius ``6.2 um``, and
mono-energetic particles of LET 10 keV/um (panel A) and 200 keV/um (panel B).
No cell line, no fitting, no transport. It is therefore the one place the whole
``f_n(z_n; D)`` chain can be checked against the paper with nothing else in the
way.

What the three curves are
-------------------------
All three take the *same* LQ coefficients and differ only in how the nucleus
specific energy is treated (the paper's appendix):

======  ==========================================  ==========================
curve   expression                                  status here
======  ==========================================  ==========================
A-2     ``integral exp(-a z - b z^2) f_n(z;D) dz``  the model under test
A-3     ``exp(integral (-a z - b z^2) f_n dz)``     closed form (see below)
A-4     ``exp(-a D - b D^2)``                       closed form, plain LQ
======  ==========================================  ==========================

**A-3 and A-4 are not digitized, because they do not need to be.** Moving the
logarithm outside the integral makes A-3 depend only on the first two moments,
and for a compound Poisson those are ``E[z] = D`` and ``E[z^2] = D z_D + D^2``,
so:

```text
A-3 = exp(-(alpha + beta z_D) D - beta D^2)
A-4 = exp(-alpha D - beta D^2)
```

Only A-2 carries information this package does not already have, so only A-2 is
extracted. [`survival_A3`][] and [`survival_A4`][] evaluate the closed forms.

How A-2 was obtained
--------------------
**Extracted from the publisher's PDF as vector geometry, not traced.** The page
carries no raster image at all; the curve is a stroked polyline, and the three
curves separate cleanly by stroke width and colour (A-2 0.59 pt dark, A-3 1.77 pt
grey, A-4 1.18 pt dark). The A-2 path also contains the plot frame and the legend
rules, so the curve is taken as the **longest contiguous run** of segments -- the
frame and legend pieces are disjoint from it.

Calibration is measured, not assumed (the trap this package has hit repeatedly is
taking a log panel's frame edge for a round number). The axis lines are single
segments: ``x`` 81.59 -> 186.07 pt for panel A and 196.55 -> 301.08 for panel B,
``y`` 82.65 -> 216.93 for both, spanning 0-10 Gy and 10^0-10^-5. Two independent
checks that it is right:

1. **The recovered doses land on the paper's own plotting grid** -- 0, 0.1, 0.15,
   0.2, 0.3, 0.5, 0.7, 1, 1.5, 2, 3, 5, 7, 10 Gy -- to within 0.3%, without the
   extractor being told what that grid was.
2. **In panel A the extracted A-2 matches the closed-form A-4 to 0.1-0.4% below
   2 Gy**, which is exactly where the paper says the three curves coincide.

One thing that does *not* work, recorded so nobody repeats it: the A-3 and A-4
curves are dashed and dotted, and a dash break falls at an arbitrary point along
a chord of the underlying polyline rather than at a vertex. On a convex log axis
a chord lies below the curve, so dot endpoints are biased low -- measured at a
median of 4% and up to 13% against the exactly-known A-4. **Never digitize a
dashed curve from vector segments if a solid one carries the same information.**
"""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np
import numpy.typing as npt

ALPHA_GY_INV: float = 0.5
"""The paper's fig. A1 coefficients (its appendix): fixed, not fitted."""
BETA_GY_INV2: float = 0.06

NUCLEUS_RADIUS_UM: float = 6.2
"""Nucleus radius used for fig. A1 [um]."""


@dataclass(frozen=True, slots=True)
class FigureA1Panel:
    """One panel of fig. A1: the Eq. (A-2) curve at a single LET."""

    label: str
    let_keV_um: float
    dose_Gy: npt.NDArray[np.float64]
    survival_A2: npt.NDArray[np.float64]


_PANEL_A_DOSE = (
    0.1007, 0.1511, 0.2016, 0.3023, 0.4987, 0.7002, 1.0025,
    1.4963, 1.9950, 2.9976, 5.0027, 7.0027, 10.0003,
)  # fmt: skip
"""Panel A -- "small variance", mono-energetic 10 keV/um."""
_PANEL_A_SURVIVAL = (
    9.470140e-01, 9.258985e-01, 9.011466e-01, 8.575052e-01, 7.694992e-01,
    6.843378e-01, 5.712834e-01, 4.146326e-01, 2.916164e-01, 1.317838e-01,
    1.901273e-02, 1.738780e-03, 2.049396e-05,
)  # fmt: skip

_PANEL_B_DOSE = (
    0.1005, 0.1510, 0.2012, 0.3020, 0.4984, 0.6998, 1.0019,
    1.5004, 1.9989, 3.0011, 5.0052, 6.9993, 10.0005,
)  # fmt: skip
"""Panel B -- "large variance", mono-energetic 200 keV/um. This is the panel with
the content: A-2 runs 20x above plain LQ by 10 Gy.
"""
_PANEL_B_SURVIVAL = (
    9.556564e-01, 9.258985e-01, 9.052539e-01, 8.614091e-01, 7.799956e-01,
    7.030721e-01, 5.976372e-01, 4.477031e-01, 3.279353e-01, 1.673833e-01,
    3.641267e-02, 6.436459e-03, 3.394064e-04,
)  # fmt: skip


def figure_a1_panel(label: str) -> FigureA1Panel:
    """Return panel ``"A"`` (10 keV/um) or ``"B"`` (200 keV/um) of fig. A1."""
    if label == "A":
        dose, surv, let = _PANEL_A_DOSE, _PANEL_A_SURVIVAL, 10.0
    elif label == "B":
        dose, surv, let = _PANEL_B_DOSE, _PANEL_B_SURVIVAL, 200.0
    else:
        raise ValueError(f"fig. A1 has panels 'A' and 'B', got {label!r}")
    return FigureA1Panel(
        label=label,
        let_keV_um=let,
        dose_Gy=np.asarray(dose, dtype=np.float64),
        survival_A2=np.asarray(surv, dtype=np.float64),
    )


def survival_A4(
    dose_Gy: npt.NDArray[np.float64] | float,
    *,
    alpha: float = ALPHA_GY_INV,
    beta: float = BETA_GY_INV2,
) -> npt.NDArray[np.float64]:
    """Eq. (A-4): plain LQ, ``exp(-alpha D - beta D^2)``."""
    d = np.asarray(dose_Gy, dtype=np.float64)
    out: npt.NDArray[np.float64] = np.exp(-(alpha * d + beta * d * d))
    return out


def survival_A3(
    dose_Gy: npt.NDArray[np.float64] | float,
    *,
    z_dose_mean_Gy: float,
    alpha: float = ALPHA_GY_INV,
    beta: float = BETA_GY_INV2,
) -> npt.NDArray[np.float64]:
    """Eq. (A-3): the logarithm moved outside the integral.

    ``exp(-(alpha + beta z_D) D - beta D^2)``, since the compound-Poisson moments
    are ``E[z] = D`` and ``E[z^2] = D z_D + D^2``. Needs only the dose-mean
    single-event specific energy in the nucleus, ``z_D``.
    """
    d = np.asarray(dose_Gy, dtype=np.float64)
    out: npt.NDArray[np.float64] = np.exp(-((alpha + beta * z_dose_mean_Gy) * d + beta * d * d))
    return out
