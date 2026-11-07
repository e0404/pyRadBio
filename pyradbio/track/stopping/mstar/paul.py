"""Paul's stopping-power ratio and MSTAR's top level (routines MSPAUL, MSTAR1).

Python port of the MSTAR subroutine package by Andreas Schinner and Helmut Paul
(program system MORESTAR, V 3.12, 24-05-2004), freeware; the authors' names are
kept as they ask. Physics: H. Paul and A. Schinner, Nucl. Instrum. Methods Phys.
Res. B 179 (2001) 299 and B 195 (2002) 166 ("An empirical approach to the stopping
power of solids and gases for ions from 3Li to 18Ar", parts I and II).

This file is under the MSTAR notice in ``THIRD_PARTY_NOTICES.md`` (AGENTS.md rule 3),
not pyRadBio's MIT licence.

MSTAR obtains the electronic stopping power of an ion ``Z1`` (3-18) at kinetic
energy ``E`` per nucleon from the alpha stopping power at ``4 E`` (total alpha
energy) times Paul's fitted ratio (a cumulative Weibull function of ``log10 E``):

    S(Z1, E) = S_alpha(4 E) * FOUT(Z1, E) * Z1^2 / 4                  (MSTAR1)

**Scope: liquid water only** (target ID 276). The general parts of MSPAUL
(modes 'c', 'g' for Z2 > 2, 'h') are ported in full; of mode 'd''s target-dependent
correction factors only the branch water selects (effective Z2 = 7, "C and Mylar
targets") is ported -- the other branches belong to targets we do not support.

**Floating-point fidelity.** FORTRAN 77 constants written without a ``D`` exponent
are REAL*4; MSTAR has several (two coefficients of mode 'c'/'d', every mode 'h'
coefficient, the effective-Z table). They are rounded to single precision here
(:func:`_r4`) exactly as the original does, and every expression keeps the
FORTRAN's operation order.
"""

from __future__ import annotations

import math
import struct

#: MSTAR's identification number of liquid water (ICRU / ASTAR target ID).
WATER_LIQUID_ID = 276

#: Modes of MSTAR1 (see data/stopping/mstar/PROVENANCE.md).
MODES = ("a", "b", "c", "d", "g", "h")

#: Mode 'h' (Z2-independent gas fits) exists for these ions only (MSPAUL).
MODE_H_IONS = (3, 4, 5, 6, 7, 8, 9, 10, 11, 16, 17, 18)


class MSTARModeError(ValueError):
    """A mode MSTAR cannot apply to this ion/target (its return code 4)."""


def _r4(value: float) -> float:
    """Round to IEEE single precision: a FORTRAN REAL*4 constant used in DOUBLE context."""
    return float(struct.unpack("f", struct.pack("f", value))[0])


#: Effective target Z of liquid water: MSID2Z's DZTAB entry for ID 276 (REAL*4).
WATER_LIQUID_DZ2 = _r4(7.216742)

#: Mode 'h' coefficients (a, b, c) per ion, MSPAUL (REAL*4 constants in the source).
_MODE_H = {
    3: (0.59251, -0.71211, 1.91956),
    4: (0.35030, -0.28291, 2.92675),
    5: (0.32192, -0.13151, 2.97854),
    6: (0.27061, -0.01952, 2.80761),
    7: (0.21054, 0.01528, 3.99705),
    8: (0.16969, 0.1075, 3.75755),
    9: (0.12192, 0.08955, 3.59459),
    10: (0.09358, 0.08692, 3.39734),
    11: (0.06353, 0.21303, 4.29082),
    16: (0.05698, 0.34491, 3.01333),
    17: (0.05163, 0.37948, 2.81286),
    18: (0.048, 0.38887, 2.84076),
}


def _coefficients(mode: str, z1: int, z2: int) -> tuple[float, float, float]:
    """MSPAUL's parameters a, b, c for ``mode`` (after 'a'/'b' resolution)."""
    x = float(z1)
    lx = math.log(x)
    if mode in ("c", "d"):
        # All solid targets, Coef(a,b,c)A2s.prn, 22 Oct 02
        a = (
            152.3281067
            - 509.175005 * x**0.5
            + 764.0597161 * x
            - 666.456225 * x**1.5
            + 369.6135044 * x**2.0
            - 134.506826 * x**2.5
            + 32.02328376 * x**3.0
            - 4.80247840 * x**3.5
            + 0.411498619 * x**4.0
            - 0.015353889 * x**4.5
        )
        b = (
            -1.60299361
            + 2.594689067 * x**0.5
            - 1.73972608 * x
            + 0.601973528 * x**1.5
            - 0.10683182 * x**2.0
            + 0.0077231543 * x**2.5
        ) / (
            1.0
            - 1.44605189 * x**0.5
            + 0.871321669 * x
            - 0.24591206 * x**1.5
            + 0.027726119 * x**2.0
            - 0.00042121782 * x**2.5
        )
        c = 1.0 / (0.15607719 - _r4(0.0023161329) * x**2.0 * lx + _r4(0.0016853943) * x**2.5)
        return a, b, c
    if mode == "g" and z2 > 2:
        # All gaseous targets except H, He; FitCoef2g(a,b,c)R.prn, 28 Jun 01
        a = (
            38.150347
            - 5.0998443 * x
            - 151.25405 / x
            + 0.3682134 * x**2
            + 301.77474 / x**2
            - 1.3542411e-2 * x**3
            - 225.08009 / x**3
            + 1.9888596e-4 * x**4
        )
        b = -0.1983923 + 3.2653672e-2 * x - 12.281832 * math.exp(-x)
        c = math.exp(-1.3058678 - 0.5460869 * lx**2 + 2.3894315 * lx)
        return a, b, c
    if mode == "h" and z1 in _MODE_H:
        a, b, c = _MODE_H[z1]
        return _r4(a), _r4(b), _r4(c)
    raise MSTARModeError(f"MSTAR mode {mode!r} is not available for Z1={z1}, Z2={z2}")


def mspaul(mode: str, z1: int, energy_MeV_u: float, z2: float = WATER_LIQUID_DZ2) -> float:
    """MSPAUL: Paul's ratio FOUT for ion ``z1`` at ``energy_MeV_u`` in a target of
    effective charge ``z2`` (liquid water by default).

    :raises MSTARModeError: for a mode the original rejects (its RC=4).
    """
    iz2 = int(z2 + 0.5)
    if mode == "h" and iz2 < 3:
        raise MSTARModeError("mode 'h' is blocked for Z2 < 3")
    a, b, c = _coefficients(mode, z1, iz2)
    x = math.log10(energy_MeV_u)
    if mode in ("c", "d"):
        d1 = 17.18567 - 0.65734 * float(z1)
        fout = 1.01 - a
        tp = (x + c * math.log(2.0) ** (1 / d1) - b) / c
        tp = max(tp, 0.0)
        fout = fout * (1.0 - math.exp(-(tp**d1)))
        fout = a + fout
        fout = (1.0 - 0.0041466419 - 0.0010098198 * math.exp(-x)) * fout
        factor = 1.0
        if mode == "d":
            if 4 < iz2 < 10:
                # C and Mylar targets (CMylrC3.prn, 11 Nov 02): the branch water uses
                factor = (
                    1.0 + 0.0012171149 - 0.04021975 / (1.0 + ((x + 2.0388341) / 0.48193661) ** 2.0)
                )
            else:
                raise MSTARModeError(
                    f"mode 'd' correction for Z2={iz2} is not ported (liquid water only)"
                )
        return fout * factor
    d1 = 9.1963 - 0.29643 * float(z1)
    fout = 1.01 - a
    tp = (x + c * math.log(2.0) ** (1 / d1) - b) / c
    tp = max(tp, 0.0)
    fout = fout * (1.0 - math.exp(-(tp**d1)))
    return a + fout


def resolve_mode(mode: str, target_is_gas: bool = False) -> tuple[str, bool]:
    """MSTAR1's mode handling: 'a' -> c|g, 'b' -> d|h preferred; returns (mode, prefer_dh)."""
    mode = mode.lower()
    if mode not in MODES:
        raise ValueError(f"mstar_mode must be one of {MODES}, got {mode!r}")
    if mode in ("a", "b"):
        prefer = mode == "b"
        if target_is_gas:
            return ("h" if prefer else "g"), prefer
        return ("d" if prefer else "c"), prefer
    return mode, False


def mstar1(alpha_stopping: float, mode: str, z1: int, energy_MeV_u: float) -> tuple[float, str]:
    """MSTAR1 for liquid water: the ion's stopping power and the mode actually used.

    ``alpha_stopping`` is the alpha electronic stopping power at ``4 * energy_MeV_u``
    (MSSEAL); the result is in its units. As in MSTAR1, mode 'b' falls back from
    'd'/'h' to 'c'/'g' where MSPAUL rejects the special fit (its return code 4), and
    the product is formed in the original's order, ``STE*FOUT*Z1**2/4``.
    """
    if not 3 <= z1 <= 18:
        raise ValueError(f"MSTAR covers ions Z1 = 3-18, got {z1}")
    used, prefer = resolve_mode(mode)
    try:
        fout = mspaul(used, z1, energy_MeV_u)
    except MSTARModeError:
        if not prefer:
            raise
        used = {"d": "c", "h": "g"}[used]
        fout = mspaul(used, z1, energy_MeV_u)
    return alpha_stopping * fout * float(z1) ** 2 / 4.0, used


__all__ = [
    "MODES",
    "MODE_H_IONS",
    "WATER_LIQUID_DZ2",
    "WATER_LIQUID_ID",
    "MSTARModeError",
    "mspaul",
    "mstar1",
    "resolve_mode",
]
