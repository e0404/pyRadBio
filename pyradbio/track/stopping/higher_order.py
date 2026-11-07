"""Higher-order corrections to the Bethe stopping logarithm: Bloch, Mott, Ahlen, LS.

All three add a term to the Bethe bracket ``L`` (multiplied by ``z^2``), with
``eta = Z alpha / beta``:

* **Bloch** (F. Bloch, Ann. Phys. 16, 285 (1933)), in the form of J. Lindhard and
  A.H. Sorensen, Phys. Rev. A 53, 2443 (1996), eq. (20):
  ``Delta L = sum_l [(l+1)/((l+1)^2 + eta^2) - 1/(l+1)] = psi(1) - Re psi(1 + i eta)``.
* **Mott**, first order: ``Delta L = Z alpha pi beta / 2`` -- the "linear Mott correction
  Z1 alpha pi v/2c" Lindhard & Sorensen quote (p. 2449; p. 2450: "for moderate charge
  numbers, the stopping is approximately obtained by adding the nonrelativistic
  Delta L_NR and the linear Mott correction").
* **Ahlen** (S.P. Ahlen, Rev. Mod. Phys. 52, 121 (1980), eqs. 4.22, 4.23): the
  close-collision Mott correction ``Delta L = G(Z, beta) / 2``, a fit to the exact Mott
  cross section (Doggett & Spencer) good to 1% for ``|Z|/beta < 100``; ICRU 90 (App. A.3)
  names it as BEST's relativistic Mott correction for C ions. Used with Bloch, in place of
  the linear Mott term. ``w_0`` -- the mean square-root ionization potential -- is set to
  0, which Ahlen states causes "negligible error ... for all but the heaviest absorbers".
  ``cos(chi)`` is Ahlen's Table XII (``|Z alpha/beta|`` = 0-2), shipped as our interpolated
  copy (``data/ahlen1980_cos_chi.csv``); outside 0-2 it raises.
* **Lindhard-Sorensen** (LS eqs. (33), (34), (41)), the exact relativistic
  generalisation of Bloch's correction for a **point nucleus**: it replaces Bloch and
  Mott together. Finite nuclear size (LS Sec. VI) is not included; LS show it matters
  once ``gamma A^(1/3) / 160`` approaches 1 (p. 2451), far above therapy energies.
  The sum over the Dirac quantum number ``kappa`` converges like ``1/kappa_max``; it is
  evaluated at two cut-offs and Richardson-extrapolated.

Phase shifts enter only through ``sin^2`` of their differences, so they are needed
mod pi; ``arg Gamma`` is taken as the imaginary part of :func:`scipy.special.loggamma`.
"""

from __future__ import annotations

import csv
import math
from functools import cache
from importlib import resources

import numpy as np
import numpy.typing as npt
from scipy.interpolate import CubicSpline
from scipy.special import digamma, loggamma

#: Fine-structure constant alpha (CODATA 2018).
FINE_STRUCTURE = 1.0 / 137.035999084

#: Default cut-off of the kappa sum (the extrapolation uses it and half of it).
DEFAULT_KAPPA_MAX = 4000


def bloch_correction(Z: float, beta: npt.NDArray[np.float64]) -> npt.NDArray[np.float64]:
    """Bloch's correction ``psi(1) - Re psi(1 + i eta)`` (LS eq. 20), ``eta = Z alpha/beta``."""
    eta = Z * FINE_STRUCTURE / np.asarray(beta, dtype=np.float64)
    out: npt.NDArray[np.float64] = digamma(1.0) - np.real(digamma(1.0 + 1j * eta))
    return out


def mott_correction(Z: float, beta: npt.NDArray[np.float64]) -> npt.NDArray[np.float64]:
    """Linear Mott correction ``Z alpha pi beta / 2`` (as quoted by LS, p. 2449)."""
    out: npt.NDArray[np.float64] = Z * FINE_STRUCTURE * math.pi * np.asarray(beta) / 2.0
    return out


AHLEN_1980 = "S.P. Ahlen, Rev. Mod. Phys. 52, 121 (1980), eqs. (4.22), (4.23), Table XII"
AHLEN_COS_CHI_TABLE = "ahlen1980_cos_chi.csv"


@cache
def _cos_chi() -> CubicSpline:
    text = resources.files(__package__).joinpath("data").joinpath(AHLEN_COS_CHI_TABLE)
    lines = text.read_text(encoding="utf-8").splitlines()
    rows = list(csv.DictReader(line for line in lines if not line.startswith("#")))
    y = np.array([float(r["abs_z1_alpha_over_beta"]) for r in rows])
    return CubicSpline(y, np.array([float(r["cos_chi"]) for r in rows]))


def ahlen_cos_chi(y: npt.ArrayLike) -> npt.NDArray[np.float64]:
    """Ahlen's Table XII, ``cos(chi)`` at ``y = |Z alpha / beta|`` (0 <= y <= 2)."""
    ya = np.asarray(y, dtype=np.float64)
    spline = _cos_chi()
    lo, hi = float(spline.x[0]), float(spline.x[-1])
    if ya.size and (ya.min() < lo or ya.max() > hi):
        raise ValueError(f"|Z alpha/beta| outside Ahlen's Table XII ({lo:g}-{hi:g})")
    out: npt.NDArray[np.float64] = spline(ya)
    return out


def ahlen_mott_correction(Z: float, beta: npt.NDArray[np.float64]) -> npt.NDArray[np.float64]:
    """``G(Z, beta) / 2`` of Ahlen's eq. (4.23) with ``w_0 = 0`` (module docstring)."""
    b = np.asarray(beta, dtype=np.float64)
    za = abs(Z) * FINE_STRUCTURE
    cos_chi = ahlen_cos_chi(za / b)
    g = (
        np.sign(Z) * za * b * (1.725 + 0.52 * math.pi * cos_chi)
        + za**2 * (3.246 - 0.451 * b**2)
        + np.sign(Z) * za**3 * (1.522 * b + 0.987 / b)
        + za**4 * (4.569 - 0.494 * b**2 - 2.696 / b**2)
        + np.sign(Z) * za**5 * (1.254 * b + 0.222 / b - 1.170 / b**3)
    )
    out: npt.NDArray[np.float64] = 0.5 * g
    return out


def ls_phase_shift(
    kappa: float | npt.NDArray[np.float64], eta: float, gamma: float, alpha_z: float
) -> npt.NDArray[np.float64]:
    """Relativistic Coulomb phase shift ``delta_kappa`` (LS eqs. 33, 34), mod pi.

    ``delta_k = xi_k - arg Gamma(s_k + 1 + i eta) - pi s_k / 2 + pi l / 2`` with
    ``s_k = sqrt(k^2 - (alpha Z)^2)``, ``exp(2 i xi_k) = (k - i eta/gamma)/(s_k - i eta)``
    and ``l = k`` for ``k > 0``, ``-k - 1`` for ``k < 0`` (LS eq. 31).
    """
    k = np.asarray(kappa, dtype=np.float64)
    s = np.sqrt(k * k - alpha_z * alpha_z)
    xi = 0.5 * np.angle((k - 1j * eta / gamma) / (s - 1j * eta))
    l_orbital = np.where(k > 0, k, -k - 1.0)
    out: npt.NDArray[np.float64] = (
        xi - np.imag(loggamma(s + 1.0 + 1j * eta)) - 0.5 * math.pi * s + 0.5 * math.pi * l_orbital
    )
    return out


def ls_phase_shift_difference(
    kappa: npt.NDArray[np.float64], eta: float, gamma: float, alpha_z: float
) -> npt.NDArray[np.float64]:
    """``delta_k - delta_(k-1)`` in LS's closed form, eq. (37), mod pi.

    Built only from small quantities, unlike the difference of two eq.-(33) phase
    shifts, which subtracts the large ``pi l / 2`` terms and loses precision exactly
    where the sum divides by ``eta^2`` (small charges, large ``kappa``).
    """
    k = np.asarray(kappa, dtype=np.float64)
    s_k = np.sqrt(k * k - alpha_z * alpha_z)
    s_k1 = np.sqrt((k - 1.0) ** 2 - alpha_z * alpha_z)
    ratio = eta / gamma
    out: npt.NDArray[np.float64] = (
        0.5 * np.arctan(eta * (s_k1 - s_k) / (s_k * s_k1 + eta * eta))
        + 0.5 * np.arctan(ratio / (k * (k - 1.0) + ratio * ratio))
        - np.imag(loggamma(s_k + 1.0 + 1j * eta))
        + np.imag(loggamma(s_k1 + 1.0 + 1j * eta))
        - 0.5 * math.pi * (s_k - s_k1 - np.sign(k))
    )
    return out


def _ls_sum(z: float, gamma: float, kappa_max: int) -> float:
    beta = math.sqrt(1.0 - 1.0 / (gamma * gamma))
    eta = z * FINE_STRUCTURE / beta
    alpha_z = z * FINE_STRUCTURE
    k = np.concatenate([np.arange(-kappa_max, 0), np.arange(2, kappa_max + 1)]).astype(float)
    d = ls_phase_shift_difference(k, eta, gamma, alpha_z)
    terms = np.abs(k) / (eta * eta) * (k - 1.0) / (2.0 * k - 1.0) * np.sin(d) ** 2
    # kappa = 1 has (kappa - 1) = 0 and contributes only its -1/(2|kappa|) = -1/2.
    first = float(np.sum(terms - 0.5 / np.abs(k))) - 0.5
    kp = np.arange(1, kappa_max + 1, dtype=float)
    ratio = eta / gamma
    spin_flip = float(np.sum(kp / (4.0 * kp * kp - 1.0) / (kp * kp + ratio * ratio))) / gamma**2
    return first + spin_flip + 0.5 * beta * beta


def lindhard_sorensen_correction(
    Z: float,
    gamma: npt.NDArray[np.float64],
    *,
    kappa_max: int = DEFAULT_KAPPA_MAX,
    extrapolate: bool = True,
) -> npt.NDArray[np.float64]:
    """LS eq. (41), point nucleus: ``Delta L`` relative to first-order perturbation theory.

    :raises ValueError: if ``alpha Z >= 1`` (no real ``s_k`` for ``|k| = 1``).
    """
    if Z * FINE_STRUCTURE >= 1.0:
        raise ValueError(f"point-nucleus Dirac phase shifts need alpha Z < 1, got Z = {Z}")
    out = np.empty(np.shape(gamma), dtype=np.float64)
    for i, g in enumerate(np.atleast_1d(np.asarray(gamma, dtype=np.float64))):
        full = _ls_sum(Z, float(g), kappa_max)
        if extrapolate:  # error ~ c / kappa_max (measured): Richardson with kappa_max / 2
            full = 2.0 * full - _ls_sum(Z, float(g), kappa_max // 2)
        out.flat[i] = full
    return out


__all__ = [
    "AHLEN_1980",
    "DEFAULT_KAPPA_MAX",
    "FINE_STRUCTURE",
    "ahlen_cos_chi",
    "ahlen_mott_correction",
    "bloch_correction",
    "lindhard_sorensen_correction",
    "ls_phase_shift",
    "ls_phase_shift_difference",
    "mott_correction",
]
