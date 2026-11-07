"""Natural cubic spline of MSTAR 3.12 (routines MSSPL1 / MSSPL2).

Python port of the MSTAR subroutine package by Andreas Schinner and Helmut Paul
(program system MORESTAR, V 3.12, 24-05-2004), released by the authors as freeware
("you can freely use, copy, modify, and/or distribute the source code. However,
please do not remove the authors' names from the source code comments").

This file is under the MSTAR notice in ``THIRD_PARTY_NOTICES.md`` (AGENTS.md rule 3),
not pyRadBio's MIT licence.

MSTAR interpolates the alpha stopping powers with a *natural* cubic spline in
**linear** energy and stopping power (no logarithms). The operations follow the
FORTRAN line by line, in the same order, so the result matches the original
program to floating-point rounding.
"""

from __future__ import annotations

from collections.abc import Sequence


def msspl1(x: Sequence[float], y: Sequence[float]) -> list[float]:
    """MSSPL1: second derivatives of the natural spline through ``(x, y)``.

    Requires ``x[0] < x[1] < ... < x[n-1]`` and ``n > 2`` (unchecked, as in MSTAR).
    """
    n = len(x)
    ypp = [0.0] * n
    ws = [0.0] * n
    for i in range(1, n - 1):
        hi = x[i] - x[i - 1]
        hi1 = x[i + 1] - x[i]
        si = hi / (hi1 + hi)
        ai = 2.0 / si
        bi = 1.0 / si - 1.0
        ci = 6.0 * ((y[i + 1] - y[i]) / hi1 - (y[i] - y[i - 1]) / hi) / hi
        ypp[i] = -bi / (ypp[i - 1] + ai)
        ws[i] = (ci - ws[i - 1]) / (ypp[i - 1] + ai)
    ypp[n - 1] = 0.0
    for i in range(n - 2, -1, -1):
        ypp[i] = ypp[i] * ypp[i + 1] + ws[i]
    return ypp


def msspl2(x: Sequence[float], y: Sequence[float], ypp: Sequence[float], xs: float) -> float:
    """MSSPL2: the spline value at ``xs`` (``x[0] <= xs <= x[-1]``, unchecked)."""
    il, ir = 0, len(x) - 1
    while ir - il > 1:
        i = (il + ir) // 2
        if x[i] <= xs:
            il = i
        else:
            ir = i
    h = x[ir] - x[il]
    a = (x[ir] - xs) / h
    b = (xs - x[il]) / h
    c = a * (a * a - 1.0) * h * h / 6.0
    d = b * (b * b - 1.0) * h * h / 6.0
    return a * y[il] + b * y[ir] + c * ypp[il] + d * ypp[ir]


__all__ = ["msspl1", "msspl2"]
