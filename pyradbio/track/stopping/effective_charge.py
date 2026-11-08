"""Effective projectile charge, for scaling stopping powers between ions.

An ion moving slower than its own orbital electrons carries some of them along, so
its stopping power scales with an *effective* charge ``z_eff <= Z`` rather than
with ``Z``. [`EffectiveChargeModel`][] is the pluggable interface;
[`BarkasEffectiveCharge`][] (the default) is Barkas's empirical form

    z_eff = Z * (1 - exp(-125 * beta * Z^(-2/3)))

(W.H. Barkas, *Nuclear Research Emulsions*, Vol. I, Academic Press, New York, 1963),
the form used across the track-structure literature and by Katz, MCDS and the
lineal-energy engines here. [`barkas_effective_charge`][] is the scalar function
those modules have always called (re-exported as
``pyradbio.track.stopping_power.effective_charge``).

For **stopping powers** of partially stripped ions (``Bethe(effective_charge=...)``),
Barkas's form -- a mean *charge state* -- under-predicts: close collisions see the nucleus
inside the electron cloud, so the stopping charge is larger. Further models,
each measured with Bethe (shell, Barkas, Bloch, Ahlen) for Z 3-18 at
4.7-300 MeV/u, worst |S/S_ref - 1| vs MSTAR / the ICRU 73 errata / ICRU 90 carbon:

* [`HubertEffectiveCharge`][] -- F. Hubert, R. Bimbot and H. Gauvin, At. Data Nucl.
  Data Tables 46, 1 (1990), eqs. (2)-(9): an empirical stopping-charge fit (3-90 MeV/u,
  O to U in Be to U) relative to **alpha particles**, ``S = (gamma Z/2)^2 S_alpha`` at the
  same velocity (``reference_Z = 2``). Water is outside their solids; ``target_Z = 8``
  (oxygen, ASSUMPTION; their separate carbon fit is worse). 5.0% / 4.6% / 1.1%.
* [`BarkasEffectiveCharge`][] with [`FITTED_BARKAS_COEFFICIENT`][] -- Barkas's form
  with the velocity coefficient fitted to MSTAR for Z 3-18 (CALIBRATION, our own):
  3.4% / 5.4% / 1.3% (MSTAR is the fit target, so only the other two are checks).
* [`BrandtKitagawaEffectiveCharge`][] -- W. Brandt and M. Kitagawa, Phys. Rev. B 25,
  5631 (1982), eq. (23) with their Table II ionization and eq. (15) ion size, for the
  water valence plasma: 12.1% / 16.9% / 7.6% -- the Thomas-Fermi stripping of Table II
  leaves too many bound electrons at these velocities.
* Barkas (original): 13.7% / 18.3% / 4.0%; bare charge: 22.8% / 16.2% / 0.04%.
"""

from __future__ import annotations

import csv
import math
from dataclasses import dataclass
from functools import cache
from importlib import resources
from typing import ClassVar, Protocol

import numpy as np
import numpy.typing as npt
from scipy.interpolate import PchipInterpolator

from pyradbio.core.constants import ATOMIC_MASS_UNIT_MEV, HARTREE_MEV

BARKAS_COEFFICIENT = 125.0
"""Barkas (1963) velocity coefficient in ``exp(-125 beta Z^(-2/3))``."""
FITTED_BARKAS_COEFFICIENT = 162.0
"""Barkas's form refitted for stopping powers in water: the coefficient minimising the rms
deviation of Bethe (shell, Barkas, Bloch, Ahlen) from MSTAR, Z 3-18, 4.7-300 MeV/u
(CALIBRATION).
"""
HUBERT_1990 = "F. Hubert, R. Bimbot and H. Gauvin, At. Data Nucl. Data Tables 46, 1 (1990)"
"""Hubert, Bimbot & Gauvin (1990) eqs. (3)-(9): D(Z2) = d0 + d1 exp(d2 Z2), B, C, and
x2..x4 = a + b ln Z2.
"""
BRANDT_KITAGAWA_1982 = "W. Brandt and M. Kitagawa, Phys. Rev. B 25, 5631 (1982)"
BRANDT_KITAGAWA_TABLE = "brandt1982_ionization.csv"
_INV_ALPHA = 137.035999084


def barkas_effective_charge(z: int, beta: float) -> float:
    """Barkas effective charge ``z_eff = Z*(1 - exp(-125*beta*Z^(-2/3)))``.

    Captures electron pick-up at low velocity (Barkas 1963; as used in the
    track-structure literature). Reduces to ``Z`` at high velocity.
    """
    return z * (1.0 - math.exp(-BARKAS_COEFFICIENT * beta * z ** (-2.0 / 3.0)))


class EffectiveChargeModel(Protocol):
    """``z_eff(Z, beta)`` for an ion of atomic number ``Z`` at speed ``beta = v/c``.

    ``reference_Z`` is the ion the model scales from: ``None`` -- the bare ion itself
    (``S = (z_eff/Z)^2 S_bare``); ``2`` -- alpha particles (``S = (z_eff/2)^2 S_alpha``).
    """

    @property
    def name(self) -> str: ...

    @property
    def reference_Z(self) -> int | None: ...

    def z_eff(self, Z: int, beta: npt.NDArray[np.float64]) -> npt.NDArray[np.float64]: ...


@dataclass(frozen=True)
class BarkasEffectiveCharge:
    """Barkas (1963) effective charge, vectorised over ``beta`` (``coefficient`` 125)."""

    coefficient: float = BARKAS_COEFFICIENT
    reference_Z: ClassVar[int | None] = None

    @property
    def name(self) -> str:
        if self.coefficient == BARKAS_COEFFICIENT:
            return "barkas"
        return f"barkas (coefficient {self.coefficient:g})"

    def z_eff(self, Z: int, beta: npt.NDArray[np.float64]) -> npt.NDArray[np.float64]:
        b = np.asarray(beta, dtype=np.float64)
        out: npt.NDArray[np.float64] = Z * (1.0 - np.exp(-self.coefficient * b * Z ** (-2.0 / 3.0)))
        return out


def _energy_per_u(beta: npt.NDArray[np.float64]) -> npt.NDArray[np.float64]:
    """Kinetic energy per atomic mass unit [MeV/u] at speed ``beta`` (Hubert's E/A)."""
    b = np.asarray(beta, dtype=np.float64)
    out: npt.NDArray[np.float64] = (1.0 / np.sqrt(1.0 - b * b) - 1.0) * ATOMIC_MASS_UNIT_MEV
    return out


@dataclass(frozen=True)
class HubertEffectiveCharge:
    """Hubert, Bimbot & Gauvin (1990) eqs. (2)-(9), relative to alpha particles.

    ``gamma = 1 - x1 exp(-x2 (E/A)^x3 Z^-x4)`` with ``x1 = D + B exp(-C Z)``; ``z_eff =
    gamma Z``. ``target_Z`` is the stopping medium's atomic number in their eqs. (3)-(9)
    (default 8: oxygen stands in for water, ASSUMPTION). Fitted for 3-90 MeV/u and
    tabulated by them for 2.5-500 MeV/u; above, ``gamma`` is 1 to < 1e-5.
    """

    target_Z: float = 8.0
    reference_Z: ClassVar[int | None] = 2

    @property
    def name(self) -> str:
        return f"hubert1990 (Z2 = {self.target_Z:g})"

    def z_eff(self, Z: int, beta: npt.NDArray[np.float64]) -> npt.NDArray[np.float64]:
        z2 = self.target_Z
        d = 1.164 + 0.2319 * math.exp(-0.004302 * z2)
        x1 = d + 1.658 * math.exp(-0.05170 * Z)
        x2 = 8.144 + 0.09876 * math.log(z2)
        x3 = 0.3140 + 0.01072 * math.log(z2)
        x4 = 0.5218 + 0.02521 * math.log(z2)
        e = _energy_per_u(beta)
        out: npt.NDArray[np.float64] = Z * (1.0 - x1 * np.exp(-x2 * e**x3 * Z ** (-x4)))
        return out


@cache
def _bk_ionization() -> tuple[PchipInterpolator, float, float, float]:
    """``ln(1 - q)`` vs ``ln y_r`` (PCHIP) and the power-law tail of ``1 - q``."""
    text = resources.files(__package__).joinpath("data").joinpath(BRANDT_KITAGAWA_TABLE)
    lines = text.read_text(encoding="utf-8").splitlines()
    rows = list(csv.DictReader(line for line in lines if not line.startswith("#")))
    y = np.array([float(r["y_r"]) for r in rows])
    q = np.array([float(r["q"]) for r in rows])
    slope = math.log((1 - q[-1]) / (1 - q[-2])) / math.log(y[-1] / y[-2])
    return PchipInterpolator(np.log(y), np.log(1.0 - q)), float(y[0]), float(y[-1]), slope


def brandt_kitagawa_ionization(y_r: npt.ArrayLike) -> npt.NDArray[np.float64]:
    """Mean ionization ``q(y_r)`` of Brandt & Kitagawa's Table II (b = 1.33).

    Above the table (``y_r > 5.932``) ``1 - q`` continues as the power law of its last two
    rows (ASSUMPTION); below its first row (0.024) it raises.
    """
    spline, lo, hi, slope = _bk_ionization()
    y = np.atleast_1d(np.asarray(y_r, dtype=np.float64))
    if y.size and y.min() < lo:
        raise ValueError(f"y_r below Brandt & Kitagawa's Table II ({lo:g})")
    out = np.empty_like(y)
    inside = y <= hi
    out[inside] = 1.0 - np.exp(spline(np.log(y[inside])))
    one_minus_q_hi = float(np.exp(spline(math.log(hi))))
    out[~inside] = 1.0 - one_minus_q_hi * (y[~inside] / hi) ** slope
    return out


@dataclass(frozen=True)
class BrandtKitagawaEffectiveCharge:
    """Brandt & Kitagawa (1982) eq. (23): effective charge fraction at high velocity.

    ``zeta^2 = q^2 + (1-q)/(2L) [(1-q)(1/((k+ L)^2+1) - 1/((k- L)^2+1)) + (1+q) ln(...)]``
    with ``L = ln(k+/k-)``, ``k+-`` of eq. (20) for the plasmon-pole dielectric function
    (eq. 18), the ion size ``Lambda`` of eq. (15) (``lambda = 4/7``, ``a = 0.240``) and
    ``q`` from Table II at ``y_r = v_1 / (Z^(2/3) v_0)`` (``v_r ~ v_1`` for ``v_1 >> v_F``,
    ASSUMPTION). Water's valence plasma: ``plasma_energy_eV`` (PDG: 21.47 eV) gives ``r_s``
    via ``omega_p = 3^(1/2) r_s^(-3/2)``; ``band_gap_eV`` (``omega_g``) defaults to 0
    (insensitive: 0 vs ``Omega_0 = I`` changes zeta^2 by < 1%).
    """

    plasma_energy_eV: float = 21.47
    band_gap_eV: float = 0.0
    reference_Z: ClassVar[int | None] = None

    @property
    def name(self) -> str:
        return "brandt-kitagawa1982"

    def z_eff(self, Z: int, beta: npt.NDArray[np.float64]) -> npt.NDArray[np.float64]:
        hartree_ev = HARTREE_MEV * 1e6
        wp = self.plasma_energy_eV / hartree_ev
        r_s = (math.sqrt(3.0) / wp) ** (2.0 / 3.0)
        k_f = 1.919 / r_s
        b2 = 0.6 * k_f**2
        om2 = wp**2 + (self.band_gap_eV / hartree_ev) ** 2
        v1 = np.asarray(beta, dtype=np.float64) * _INV_ALPHA
        q = brandt_kitagawa_ionization(v1 / Z ** (2.0 / 3.0))
        n_frac = 1.0 - q
        lam = 2.0 * 0.240 * n_frac ** (2.0 / 3.0) / (Z ** (1.0 / 3.0) * (1.0 - n_frac / 7.0))
        u = v1**2 - b2
        root = np.sqrt(u**2 - om2)
        k_plus, k_minus = np.sqrt(2.0 * (u + root)), np.sqrt(2.0 * (u - root))
        ell = np.log(k_plus / k_minus)
        xp, xm = (k_plus * lam) ** 2 + 1.0, (k_minus * lam) ** 2 + 1.0
        zeta2 = q**2 + (1.0 - q) / (2.0 * ell) * (
            (1.0 - q) * (1.0 / xp - 1.0 / xm) + (1.0 + q) * np.log(xp / xm)
        )
        out: npt.NDArray[np.float64] = Z * np.sqrt(zeta2)
        return out


__all__ = [
    "BARKAS_COEFFICIENT",
    "BRANDT_KITAGAWA_1982",
    "FITTED_BARKAS_COEFFICIENT",
    "HUBERT_1990",
    "BarkasEffectiveCharge",
    "BrandtKitagawaEffectiveCharge",
    "EffectiveChargeModel",
    "HubertEffectiveCharge",
    "barkas_effective_charge",
    "brandt_kitagawa_ionization",
]
