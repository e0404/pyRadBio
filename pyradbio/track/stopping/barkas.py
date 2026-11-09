"""Barkas (z^3) correction of Ashley, Ritchie & Brandt, as used by ICRU 49, for water.

ICRU 49 (sec. 2.3, eqs. 2.8, 2.9) adds ``z L1`` to the stopping number per electron with,
per target element of atomic number ``Z``:

```text
L1 = gamma F_ARB(b / x^1/2) / (Z^1/2 x^3/2),     x = (beta / alpha)^2 / Z
```

``F`` is the function of J.C. Ashley, R.H. Ritchie and W. Brandt, Phys. Rev. B 5, 2393
(1972), eq. (14) (Lenz-Jensen atom). We recover it from their Phys. Rev. A 8, 2402 (1973)
Table I, which prints ``kappa(b, x) = F(b/x^1/2) / (x^1/2 L(x))`` with ``L(x) = ln(x/q)``,
``q = 0.18`` for ``x > 1`` (their eqs. 2, 3): ``F = kappa x^1/2 ln(x/0.18)``. Below
``x = 1`` their ``L`` carries an inner-shell term known only as a figure, so the
correction raises there ([`OutOfRangeError`][]); for
water that is ``x_O < 1``, i.e. below ~0.2 MeV/u. Above the table (``x > 100``) ``F`` is
continued linearly in ``ln x`` (ASSUMPTION; ARB 1972, p. 2394: for small ``xi`` the
underlying ``I(xi) ~ -(3/2) pi ln xi + A``), fitted to the shipped nodes with
``30 <= x <= 100``.

``F`` ships as our regridded table ``data/barkas_arb1973.csv`` (cubic in ``ln x`` through
the printed values, 50 points per decade); the transcription itself is raw data
(``data/stopping/arb1973/PROVENANCE.md``).

**gamma.** ICRU 49 prints ``gamma = 1.29`` in eq. (2.8) ("approximately equal to
sqrt 2", from the free-electron gas). With it, our water ``zB1`` exceeds BEST's -- the
code behind ICRU 49 and 90 -- as plotted in ICRU 90 Fig. 4.1 (protons in liquid water,
digitised) by a near-constant factor 1.22-1.30 from 0.5 to 5 MeV; with
``gamma = 1``, i.e. ARB 1973's own eq. (1), it agrees within 5%. ICRU 49 refers to a
tabulation of ``F_ARB`` (Ashley et al. 1974) that we do not hold and that may absorb the
factor. Default ``gamma = 1.0`` (measured); ``gamma`` is an option.

**Water**: ``L1 = (2/10) L1(H) + (8/10) L1(O)`` (electron-weighted, Bragg additivity as in
ICRU 49 sec. 2.4, eq. 2.20), ``b = 1.8`` for "H in compounds" and for Z 3-10 (ICRU 49
Table 2.3).
"""

from __future__ import annotations

import csv
import math
from dataclasses import dataclass
from functools import cache
from importlib import resources

import numpy as np
import numpy.typing as npt
from scipy.interpolate import CubicSpline

from pyradbio.track.stopping.base import OutOfRangeError

FloatArray = npt.NDArray[np.float64]

ARB_1972 = "J.C. Ashley, R.H. Ritchie and W. Brandt, Phys. Rev. B 5, 2393 (1972)"
ARB_1973 = "J.C. Ashley, R.H. Ritchie and W. Brandt, Phys. Rev. A 8, 2402 (1973), Table I"
ARB_Q = 0.18
"""ARB 1973 eq. (3): L(x) = ln(x/q) for x > 1."""
B_VALUES = (1.6, 1.8, 2.0)
"""Values of the scaled minimum impact parameter tabulated by ARB 1973."""
ICRU49_B_WATER = 1.8
"""ICRU 49 Table 2.3: "1.8 for H in compounds", 1.8 for Z 3-10."""
ICRU49_GAMMA = 1.29
SHIPPED_TABLE = "barkas_arb1973.csv"
_EXTRAPOLATE_FROM_X = 30.0
_FINE_STRUCTURE = 1.0 / 137.035999084


def f_from_kappa(kappa: npt.ArrayLike, x: npt.ArrayLike) -> FloatArray:
    """``F(b/x^1/2) = kappa(b, x) x^1/2 ln(x/q)``, valid for ``x > 1`` (ARB 1973)."""
    xa = np.asarray(x, dtype=np.float64)
    out: FloatArray = np.asarray(kappa, dtype=np.float64) * np.sqrt(xa) * np.log(xa / ARB_Q)
    return out


@cache
def _table() -> dict[str, FloatArray]:
    text = resources.files(__package__).joinpath("data").joinpath(SHIPPED_TABLE)
    lines = text.read_text(encoding="utf-8").splitlines()
    rows = list(csv.DictReader(line for line in lines if not line.startswith("#")))
    return {k: np.array([float(r[k]) for r in rows]) for k in rows[0]}


@cache
def _interpolant(b: float) -> tuple[CubicSpline, float, float, tuple[float, float]]:
    if b not in B_VALUES:
        raise ValueError(f"b must be one of {B_VALUES} (ARB 1973 Table I), got {b!r}")
    t = _table()
    lx, f = np.log(t["x"]), t[f"F_b{b:.1f}"]
    tail = t["x"] >= _EXTRAPOLATE_FROM_X
    slope, intercept = np.polyfit(lx[tail], f[tail], 1)
    return CubicSpline(lx, f), float(t["x"][0]), float(t["x"][-1]), (slope, intercept)


def arb_function(b: float, x: npt.ArrayLike) -> FloatArray:
    """``F(b/x^1/2)`` for ``x >= 1``; linear in ``ln x`` beyond the table (``x > 100``)."""
    spline, lo, hi, (slope, intercept) = _interpolant(float(b))
    xa = np.atleast_1d(np.asarray(x, dtype=np.float64))
    if xa.size and xa.min() < lo:
        raise OutOfRangeError(
            (float(xa.min()), float(xa.max())), (lo, math.inf), source="ARB F (x)"
        )
    lx = np.log(xa)
    out: FloatArray = np.where(
        xa <= hi, spline(np.minimum(lx, math.log(hi))), slope * lx + intercept
    )
    return out


def barkas_l1(
    Z: int, x: npt.ArrayLike, *, b: float = ICRU49_B_WATER, gamma: float = 1.0
) -> FloatArray:
    """ICRU 49 eq. (2.8) for one target element: ``gamma F(b/x^1/2) / (Z^1/2 x^3/2)``."""
    xa = np.asarray(x, dtype=np.float64)
    out: FloatArray = gamma * arb_function(b, xa) / (math.sqrt(Z) * xa**1.5)
    return out


@dataclass(frozen=True)
class WaterBarkasCorrection:
    """``L1`` of liquid water per electron (multiply by the projectile charge ``z``)."""

    b_H: float = ICRU49_B_WATER
    b_O: float = ICRU49_B_WATER
    gamma: float = 1.0

    def __post_init__(self) -> None:
        for b in (self.b_H, self.b_O):
            if b not in B_VALUES:
                raise ValueError(f"b must be one of {B_VALUES}, got {b!r}")
        if not self.gamma > 0.0:
            raise ValueError("gamma must be > 0")

    def beta2_range(self) -> tuple[float, float]:
        """From ``x_O = 1`` (the end of the recoverable F) up."""
        return 8.0 * _FINE_STRUCTURE**2, 1.0

    def l1(self, beta2: npt.ArrayLike) -> FloatArray:
        y = np.atleast_1d(np.asarray(beta2, dtype=np.float64)) / _FINE_STRUCTURE**2
        lo = self.beta2_range()[0]
        if y.size and y.min() * _FINE_STRUCTURE**2 < lo:
            raise OutOfRangeError(
                (float(y.min() * _FINE_STRUCTURE**2), float(y.max() * _FINE_STRUCTURE**2)),
                (lo, 1.0),
                source="water Barkas correction (beta^2)",
            )
        out: FloatArray = 0.2 * barkas_l1(1, y, b=self.b_H, gamma=self.gamma) + 0.8 * barkas_l1(
            8, y / 8.0, b=self.b_O, gamma=self.gamma
        )
        return out

    def describe(self) -> str:
        return f"ARB 1973 (b_H={self.b_H:g}, b_O={self.b_O:g}, gamma={self.gamma:g})"


__all__ = [
    "ARB_1972",
    "ARB_1973",
    "ARB_Q",
    "B_VALUES",
    "ICRU49_B_WATER",
    "ICRU49_GAMMA",
    "WaterBarkasCorrection",
    "arb_function",
    "barkas_l1",
    "f_from_kappa",
]
