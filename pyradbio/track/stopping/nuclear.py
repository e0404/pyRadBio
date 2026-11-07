"""Nuclear stopping of ions in liquid water, the ICRU 49 way.

ICRU Report 49 (1993), sec. 4.3, with the corrections of ICRU Report 90 (2014), App. A.3,
footnote 35. For a projectile of charge ``z``, mass ``M`` [u] and kinetic energy ``T``
[keV] on target atoms ``(Z, M_t, A)``::

    T_hat = 32.536 T / [a(z, Z) (1 + M/M_t) z Z]                                 (4.14)
    S_nuc/rho = 5105.3 z Z / [a(z, Z) (1 + M_t/M) A] S_hat(T_hat)   [MeV cm^2/g] (4.15)

``S_hat`` is ICRU 49 Table 4.1, tabulated for two interatomic potentials:

* ``"zbl"`` -- the universal Ziegler-Biersack-Littmark potential with
  ``a(z, Z) = z^0.23 + Z^0.23``. ICRU 49 used it for alphas; ICRU 90 footnote 35 corrects
  ICRU 49's misprinted ``a`` and applies it to "particles heavier than protons" (C ions);
* ``"moliere"`` -- Moliere's Thomas-Fermi potential with ``a = Z^(1/3)``, used by ICRU
  49 for protons, classically below ``Z_t/10`` MeV of each target atom and with
  Moliere's (1947) quantum-mechanical cross section above (ICRU 49 sec. 4.3). Above the
  switch we use the first Born approximation of that cross section, which is analytic for
  Moliere's three-exponential screening (eq. 4.6): ``F(q) = 2 alpha_S sum_i a_i /
  (q^2 + (b_i/r_s)^2)``, ``alpha_S = z Z_t / (137 beta)``, and per target atom
  ``S_nuc = (pi hbar^2 / M_t) int_0^{q_max} q^3 |F(q)|^2 dq``. Kinematics as ICRU's: the
  velocity is nonrelativistic (``v^2 = 2T/M``, as ``T_hat ~ T``); ``q_max = 2 p`` uses the
  relativistic centre-of-mass momentum (without it the result drifts to -8% by 10 GeV).
  ``moliere_quantum=False`` keeps the classical Table 4.1 at all energies.
  MEASURED vs PSTAR's nuclear column, max deviation per band: < 0.1 MeV
  0.10%; 0.1-0.8 MeV -2.7%; 0.8-3 MeV +5.0% (oxygen's ``alpha_S`` = 1.4-0.7, where Born
  is not valid and Moliere's full theory -- not implemented -- would be needed); 3-10
  MeV -1.4%; above 10 MeV a flat -2.4% (classical-only: +5% at 1-100 MeV, -19% at
  10 GeV). Nuclear stopping is < 0.1% of the total at 1 MeV, so the +5% band is
  <= 4e-5 of S_tot.

Water is a Bragg-additive mixture of H and O by mass, with mean atomic masses from the
shipped NIST isotope table (:func:`~pyradbio.track.stopping.masses.mean_atomic_mass_u`);
the projectile mass is the isotope's atomic mass. Measured (ZBL): ASTAR /
ICRU 90 alpha and ICRU 90 C-12 nuclear columns reproduced to <= 0.07% everywhere.

``proton_potential`` selects the protons' potential (default ``"zbl"``: one universal
potential for every ion, the analytic engines' choice); every heavier ion uses ZBL.
:data:`ICRU49_PROTON_POTENTIAL` is ICRU 49's own choice.

``S_hat`` ships as our regridded copy of Table 4.1 (``data/icru49_reduced_nuclear.csv``,
cubic in log-log, ``T_hat`` = 1e-5 to 1e8). Below 1e-5 the model raises; above 1e8,
where the table's ``S_hat * T_hat`` is constant to 0.12% over its last decade, it is
continued as ``S_hat ~ 1/T_hat`` (ASSUMPTION; nuclear stopping is ~1e-7 of the total
there).
"""

from __future__ import annotations

import csv
from dataclasses import dataclass
from functools import cache
from importlib import resources
from typing import Literal, get_args

import numpy as np
import numpy.typing as npt

from pyradbio.core.constants import (
    ATOMIC_MASS_UNIT_MEV,
    AVOGADRO_PER_MOL,
    BOHR_RADIUS_CM,
    ELECTRON_MASS_MEV,
    HARTREE_MEV,
)
from pyradbio.track.stopping.base import OutOfRangeError
from pyradbio.track.stopping.grid import RegriddedTable
from pyradbio.track.stopping.masses import atomic_mass_u, mean_atomic_mass_u

Potential = Literal["zbl", "moliere"]
FloatArray = npt.NDArray[np.float64]

ICRU49_NUCLEAR_REFERENCE = (
    "ICRU Report 49 (1993), sec. 4.3, eqs. (4.14), (4.15), Table 4.1; ICRU Report 90 (2014), "
    "App. A.3, footnote 35"
)
SHIPPED_STEM = "icru49_reduced_nuclear"
#: ICRU 49 used Moliere's potential for protons.
ICRU49_PROTON_POTENTIAL: Potential = "moliere"
#: Constants of ICRU 49 eqs. (4.14) and (4.15) (T in keV; result in MeV cm^2/g).
T_HAT_PER_KEV = 32.536
S_NUC_COEFFICIENT = 5105.3
#: Liquid water: (Z, atoms per molecule).
_WATER = ((1, 2), (8, 1))
#: Moliere's Thomas-Fermi fit (Moliere 1947 eq. 7,1; ICRU 49 eq. 4.6): (a_i, b_i).
MOLIERE_SCREENING = ((0.10, 6.0), (0.55, 1.2), (0.35, 0.3))
#: ICRU 49 eq. (4.5): r_s = 0.88534 r_Bohr Z^(-1/3) for protons.
THOMAS_FERMI_CONSTANT = 0.88534
_INV_ALPHA = 137.035999084
_ME_U = ELECTRON_MASS_MEV / ATOMIC_MASS_UNIT_MEV


def _born_q_integral(alpha_s: FloatArray, lam: tuple[float, ...], q_max: FloatArray) -> FloatArray:
    """``int_0^q_max q^3 |F|^2 dq`` for ``F = 2 alpha_s sum_i a_i / (q^2 + lam_i^2)``."""
    u = q_max * q_max
    tot = np.zeros_like(u)
    weights = [a for a, _ in MOLIERE_SCREENING]
    for ai, li in zip(weights, lam, strict=True):
        for aj, lj in zip(weights, lam, strict=True):
            a2, b2 = li * li, lj * lj
            if a2 == b2:
                val = 0.5 * (np.log((u + a2) / a2) + a2 / (u + a2) - 1.0)
            else:
                val = 0.5 * (b2 * np.log((u + b2) / b2) - a2 * np.log((u + a2) / a2)) / (b2 - a2)
            tot = tot + ai * aj * val
    out: FloatArray = 4.0 * alpha_s**2 * tot
    return out


def moliere_born_nuclear_stopping(
    Z: int, A: int, target_Z: int, target_mass_u: float, energy_MeV_u: npt.ArrayLike
) -> FloatArray:
    """Nuclear stopping [MeV cm^2 per g of the target element], Born + Moliere screening.

    Atomic units throughout (lengths in Bohr radii, ``hbar = m_e = 1``); see the module
    docstring for the formula and the kinematics.
    """
    e = np.atleast_1d(np.asarray(energy_MeV_u, dtype=np.float64))
    t_mev = e * A
    m_u = atomic_mass_u(Z, A) - Z * _ME_U  # bare projectile
    mu_me = m_u * target_mass_u / (m_u + target_mass_u) / _ME_U
    beta_nr = np.sqrt(2.0 * t_mev / (m_u * ATOMIC_MASS_UNIT_MEV))
    gamma = 1.0 + t_mev / (m_u * ATOMIC_MASS_UNIT_MEV)
    beta = np.sqrt(1.0 - 1.0 / gamma**2)
    alpha_s = Z * target_Z / (_INV_ALPHA * beta_nr)
    r_s = THOMAS_FERMI_CONSTANT * target_Z ** (-1.0 / 3.0)
    lam = tuple(b / r_s for _, b in MOLIERE_SCREENING)
    q_max = 2.0 * mu_me * gamma * beta * _INV_ALPHA
    per_atom_hartree_a0 = np.pi / (target_mass_u / _ME_U) * _born_q_integral(alpha_s, lam, q_max)
    out: FloatArray = (
        per_atom_hartree_a0 * HARTREE_MEV * BOHR_RADIUS_CM**2 * AVOGADRO_PER_MOL / target_mass_u
    )
    return out


@cache
def _reduced(potential: Potential) -> RegriddedTable:
    text = resources.files(__package__).joinpath("data").joinpath(f"{SHIPPED_STEM}.csv")
    lines = text.read_text(encoding="utf-8").splitlines()
    rows = list(csv.DictReader(line for line in lines if not line.startswith("#")))
    t_hat = np.array([float(r["T_hat"]) for r in rows])
    return RegriddedTable(t_hat, np.array([float(r[f"S_{potential}"]) for r in rows]))


def reduced_nuclear_stopping(t_hat: npt.ArrayLike, potential: Potential = "zbl") -> FloatArray:
    """ICRU 49 Table 4.1, ``S_hat(T_hat)`` (``S_hat ~ 1/T_hat`` beyond its top end)."""
    if potential not in get_args(Potential):
        raise ValueError(f"potential must be one of {get_args(Potential)}, got {potential!r}")
    t = np.atleast_1d(np.asarray(t_hat, dtype=np.float64))
    table = _reduced(potential)
    lo, hi = table.energy_range
    if t.size and t.min() < lo:
        raise OutOfRangeError(
            (float(t.min()), float(t.max())), (lo, np.inf), source="ICRU 49 T_hat"
        )
    inside = t <= hi
    out = np.empty_like(t)
    out[inside] = table(t[inside])
    out[~inside] = float(table(np.array([hi]))[0]) * hi / t[~inside]
    return out


def screening_a(z: int, Z: int, potential: Potential) -> float:
    """``a(z, Z)``: ``z^0.23 + Z^0.23`` (ZBL, ICRU 90 fn. 35) or ``Z^(1/3)`` (Moliere)."""
    if potential == "zbl":
        return float(z**0.23 + Z**0.23)
    return float(Z ** (1.0 / 3.0))


@dataclass(frozen=True)
class WaterNuclearStopping:
    """Mass nuclear stopping power of liquid water [MeV cm^2/g] (module docstring)."""

    proton_potential: Potential = "zbl"
    moliere_quantum: bool = True

    def __post_init__(self) -> None:
        if self.proton_potential not in get_args(Potential):
            raise ValueError(f"proton_potential must be one of {get_args(Potential)}")

    def potential(self, Z: int) -> Potential:
        return self.proton_potential if Z == 1 else "zbl"

    def mass_stopping_power(self, Z: int, A: int, energy_MeV_u: npt.ArrayLike) -> FloatArray:
        """Ion ``(Z, A)`` at kinetic energy per nucleon ``energy_MeV_u``."""
        e = np.atleast_1d(np.asarray(energy_MeV_u, dtype=np.float64))
        t_kev = e * A * 1e3
        m = atomic_mass_u(Z, A)
        potential = self.potential(Z)
        weights = {zt: n * mean_atomic_mass_u(zt) for zt, n in _WATER}
        total = sum(weights.values())
        out = np.zeros_like(e)
        for zt, _ in _WATER:
            mt = mean_atomic_mass_u(zt)
            a = screening_a(Z, zt, potential)
            t_hat = T_HAT_PER_KEV * t_kev / (a * (1.0 + m / mt) * Z * zt)
            s = S_NUC_COEFFICIENT * Z * zt / (a * (1.0 + mt / m) * mt)
            quantum = np.zeros_like(e, dtype=bool)
            if Z == 1 and potential == "moliere" and self.moliere_quantum:
                quantum = t_kev * 1e-3 >= zt / 10.0  # ICRU 49 sec. 4.3: above Z_t/10 MeV
            part = np.empty_like(e)
            if (~quantum).any():
                part[~quantum] = s * reduced_nuclear_stopping(t_hat[~quantum], potential)
            if quantum.any():
                part[quantum] = moliere_born_nuclear_stopping(Z, A, zt, mt, e[quantum])
            out += weights[zt] / total * part
        return out

    def describe(self) -> str:
        quantum = (
            ", Born above Z_t/10 MeV"
            if (self.proton_potential == "moliere" and self.moliere_quantum)
            else ""
        )
        return (
            f"ICRU 49 eqs. 4.14-4.15 + Table 4.1 (protons: {self.proton_potential}{quantum}, "
            "others: zbl)"
        )


__all__ = [
    "ICRU49_NUCLEAR_REFERENCE",
    "ICRU49_PROTON_POTENTIAL",
    "MOLIERE_SCREENING",
    "SHIPPED_STEM",
    "S_NUC_COEFFICIENT",
    "THOMAS_FERMI_CONSTANT",
    "T_HAT_PER_KEV",
    "Potential",
    "WaterNuclearStopping",
    "moliere_born_nuclear_stopping",
    "reduced_nuclear_stopping",
    "screening_a",
]
