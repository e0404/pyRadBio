"""Shell corrections from hydrogenic K and L shells (Walske), assembled for liquid water.

The Bethe stopping number per target electron is reduced by ``C/Z`` (ICRU 90 eq. 4.18).
ICRU 49 (sec. 2.4.1, "Model 1") takes the K-shell corrections from M.C. Walske, Phys.
Rev. 88, 1283 (1952) and the L-shell corrections from M.C. Walske, Phys. Rev. 101, 940
(1956), refined by Bichsel (1967) and Khandelwal (1968) -- the two refinements are not
accessible to us, so this module computes Walske's own corrections, exactly, and leaves
the free parameters as options.

Walske's quantities, in shell units (energy ``Z_eff^2 Ry``, ``Q = q^2`` in
``2 m Z_eff^2 Ry``), with ``phi_W(Q) = W |F_W(Q)|^2 / Q`` (1952, after eq. 6):

* stopping number of the shell (1952 eq. 5)::

      B(theta, eta) = int_{W_min}^inf dW int_{W^2/4eta}^inf dQ/Q phi_W(Q),  W_min = theta/n^2

* ``B = S ln eta + T - C`` (1952 eq. 4): ``S`` and ``T`` are the large-``eta`` constants,
  ``C -> 0`` as ``eta -> inf``. ``S = n^2 + int_{W_min}^inf phi_W(0) dW`` (the Bethe
  ridge carries the sum rule ``int phi_W(Q) dW = n^2`` at large ``Q``);
* ``K``: 2 electrons, ``|F_W,K|^2`` from 1952 eq. 9 (``W = k^2 + 1``); ``L``: 8 electrons,
  1956 eq. 5 (``W = k^2 + 1/4``). Below the ionisation threshold the continuum formulae
  are continued to ``k^2 < 0`` with unit normalisation, which reproduces the discrete
  transitions (1956, after eq. 6; Walske's "unoccupied discrete states"); ``tan^-1`` is
  taken in the first and second quadrants (1952 footnote 11);
* ``C`` for ``eta >= 3`` from Walske's exact representation (1952 p. 1285, including the
  ``eta^-5`` term he drops in eq. 8), for ``eta < 3`` as ``S ln eta + T - B``. The two
  agree to ~1e-5 at ``eta = 3``; below it the representation degrades (it relies on the
  sum rule over the continued spectrum, which fails at small ``Q``).

Checked against Walske (tests): ``S``, ``T`` of 1952 eq. 19 and 1956 eq. 11 to all printed
digits; ``C_L`` of 1956 Table III ("exact calculation") to <= 0.05% for ``eta_L >= 3.5``
and 0.04% at ``eta_L = 1``; ``C_K`` of 1952 Table I to 0.2% at ``eta_K = 20`` and 1.6% at
``eta_K = 10`` (Walske's own two routes to ``B_K`` differ by 0.1% there). Against G.S.
Khandelwal, Nucl. Phys. A116, 97 (1968): ``B_K`` of his Table 3 to a median 1.8e-5 (he
states four figures), ``S_L``, ``T_L`` of Table 2 to 3e-4; his ``S_K``, ``T_K`` (Table 1)
sit 4e-4 above Walske's, which ours equal. The hydrogen atom's ``1/eta^2`` coefficient is
25/3 (Khandelwal eq. 18; our exact atom: 8.33), not the 22/3 of Walske's eq. 17 -- so
Walske's ``V_K`` (eq. 19) are 1.000 low; ``C`` itself is computed, not taken from either
series.

**Water** (:class:`WaterShellCorrection`): ``C = 2 C_H + C_K,O + (n_L/8) C_L,O`` per
molecule, ``Z = 10``. Every choice below is an ASSUMPTION or a calibration, not a value
from ICRU 49:

* H: the hydrogen atom, exactly (``Z_eff = 1``, its true discrete lines plus continuum);
* O K shell: ``Z_eff = Z - 0.3`` (Walske 1956 sec. III), ``theta_K`` default 0.7;
* O L shell: ``Z_eff = Z - 4.15`` (Walske 1956 footnote 8, after Slater), Walske's
  8-electron result scaled by ``n_L/8`` with ``n_L = 6`` (ASSUMPTION), ``theta_L``
  default 0.5. Walske warns his L shell is unreliable for ``Z < 30``.
* ``theta_K = 0.7, theta_L = 0.5`` are CALIBRATED: with Bloch and the
  ARB 1973 Barkas term they bring Bethe within +-0.13% of PSTAR (ICRU 49, I = 75 eV) for
  protons 0.5-1000 MeV, and reproduce the digitised ``C/Z`` of ICRU 90 Fig. 4.1 (BEST,
  protons in water) to 2.5% rms from 0.5 to 50 MeV -- as Bichsel fitted Model 1's
  parameters to measured stopping powers (ICRU 49 sec. 2.4.1).

``evaluation="interpolate"`` (default) reads a table of ``C(eta)`` computed by this
module at the default ``theta`` (``data/shell_corrections_water.csv``, written by
``scripts/stopping/build_tables.py``); other ``theta`` build the same table on first use
(seconds). ``"direct"`` integrates at every call.
"""

from __future__ import annotations

import csv
import math
from dataclasses import dataclass
from functools import cache
from importlib import resources
from typing import Literal, get_args

import numpy as np
import numpy.typing as npt
from scipy.interpolate import CubicSpline

from pyradbio.track.stopping.base import OutOfRangeError

Shell = Literal["K", "L"]
Evaluation = Literal["interpolate", "direct"]
FloatArray = npt.NDArray[np.float64]

WALSKE_1952 = "M.C. Walske, Phys. Rev. 88, 1283 (1952)"
WALSKE_1956 = "M.C. Walske, Phys. Rev. 101, 940 (1956)"

#: threshold ``a`` (``W = k^2 + a``), ``tan^-1`` factor ``s``, ``n^2``, electrons.
_SHELL = {"K": (1.0, 2.0, 1.0, 2), "L": (0.25, 1.0, 4.0, 8)}
#: Effective-charge screening (Walske 1956 sec. III and footnote 8).
SCREENING = {"K": 0.3, "L": 4.15}

_GL_X, _GL_W = np.polynomial.legendre.leggauss(10)
_ETA_SWITCH = 3.0
_N_LINES = 400  # hydrogen: explicit lines n = 2.._N_LINES, continued formula above
_Q_TOP = 1e30
N_Q_PANELS = 30


def _expfac(k2: FloatArray, x: FloatArray, s: float) -> FloatArray:
    """normalisation x exp(-(2/k) tan^-1(s k / x)), continued to k^2 <= 0 (unit norm)."""
    out = np.empty(np.broadcast(k2, x).shape)
    k2b, xb = np.broadcast_arrays(k2, x)
    pos, neg = k2b > 1e-14, k2b < -1e-14
    zero = ~(pos | neg)
    k = np.sqrt(k2b[pos])
    out[pos] = np.exp(-(2.0 / k) * np.arctan2(s * k, xb[pos])) / -np.expm1(-2.0 * np.pi / k)
    kap = np.sqrt(-k2b[neg])
    out[neg] = np.exp(-(2.0 / kap) * np.arctanh(s * kap / xb[neg]))
    out[zero] = np.exp(-2.0 * s / xb[zero])
    return out


def _phi(shell: Shell, W: FloatArray, Q: FloatArray) -> FloatArray:
    """phi_W(Q) = W |F_W(Q)|^2 / Q (Walske 1952 eq. 9; 1956 eq. 5 times W/Q)."""
    a, s, _, _ = _SHELL[shell]
    k2 = W - a
    e = _expfac(k2, Q - k2 + a, s)
    if shell == "K":
        den = ((Q - k2 + 1.0) ** 2 + 4.0 * k2) ** 3
        out: FloatArray = 2.0**7 * W * (Q + k2 / 3.0 + 1.0 / 3.0) * e / den
        return out
    den = ((Q + k2 + 0.25) ** 2 - 4.0 * Q * k2) ** 4
    poly = (
        Q**3
        + (-11.0 / 12.0 - 5.0 / 3.0 * k2) * Q**2
        + (65.0 / 48.0 + 1.5 * k2 + k2**2 / 3.0) * Q
        + (5.0 / 64.0 + 23.0 / 48.0 * k2 + 0.75 * k2**2 + k2**3 / 3.0)
    )
    out = 2.0**4 * W * e * poly / den
    return out


def _qint(shell: Shell, W: FloatArray, lo: FloatArray, hi: FloatArray, *, sub0: bool) -> FloatArray:
    """int_{lo}^{hi} dQ/Q [phi_W(Q) - phi_W(0) if sub0], vectorised over W.

    ``lo = 0`` is allowed with ``sub0``. Gauss-Legendre in ln Q on panels: 60 uniform
    ones plus points that resolve the Bethe ridge (``Q ~ k^2``, width ~ ``2k``).
    """
    a = _SHELL[shell][0]
    W = np.asarray(W, dtype=np.float64)
    ln_hi = np.log(hi)
    ln_lo = np.where(lo > 0.0, np.log(np.maximum(lo, 1e-300)), np.minimum(ln_hi, -6.0) - 34.0)
    k2 = W - a
    ridge = k2 > 0.5
    k = np.sqrt(np.where(ridge, k2, 1.0))
    t = np.linspace(0.0, 1.0, N_Q_PANELS + 1)
    pts = [ln_lo[:, None] + (ln_hi - ln_lo)[:, None] * t[None, :]]
    for c in (0.0, 1.0, -1.0, 3.0, -3.0, 8.0, -8.0, 20.0, -20.0, 50.0, -50.0):
        q = np.where(ridge, k2 + c * 2.0 * k, np.exp(ln_lo))
        pts.append(np.log(np.maximum(q, 1e-300))[:, None])
    edges = np.sort(np.clip(np.concatenate(pts, axis=1), ln_lo[:, None], ln_hi[:, None]), axis=1)
    half = 0.5 * np.diff(edges, axis=1)
    mid = 0.5 * (edges[:, 1:] + edges[:, :-1])
    nodes = mid[:, :, None] + half[:, :, None] * _GL_X
    f = _phi(shell, W[:, None, None], np.exp(nodes))
    if sub0:
        f = f - _phi(shell, W, np.zeros_like(W))[:, None, None]
    out: FloatArray = np.einsum("ijk,ijk->i", f, half[:, :, None] * _GL_W)
    return np.where(ln_hi > ln_lo, out, 0.0)


def _w_nodes(
    lo: float, hi: float, breaks: tuple[float, ...], width: float = 0.25
) -> tuple[FloatArray, FloatArray]:
    """Gauss-Legendre nodes and dW weights on [e^lo, e^hi], panels <= ``width`` in ln W."""
    pts = np.unique(np.clip(np.r_[np.arange(lo, hi, width), hi, np.log(breaks)], lo, hi))
    half = 0.5 * np.diff(pts)
    u = (0.5 * (pts[1:] + pts[:-1]))[:, None] + half[:, None] * _GL_X
    w = np.exp(u)
    return w.ravel(), (half[:, None] * _GL_W * w).ravel()


@dataclass(frozen=True)
class _Spectrum:
    """Allowed transitions of one shell: discrete lines plus a continued continuum."""

    shell: Shell
    w_min: float
    lines: tuple[FloatArray, FloatArray] | None = None  # (W_n, dW/dn) for exact hydrogen

    def allowed(self, eta: float) -> tuple[FloatArray, FloatArray]:
        a = _SHELL[self.shell][0]
        top = math.log(max(1e4, 4e4 * eta))
        w, dw = _w_nodes(math.log(self.w_min), top, (4.0 * eta, a))
        if self.lines is not None:
            w, dw = np.r_[self.lines[0], w], np.r_[self.lines[1], dw]
        return w, dw

    def forbidden(self) -> tuple[FloatArray, FloatArray]:
        if self.lines is not None:  # the hydrogen atom: every transition is allowed
            return np.empty(0), np.empty(0)
        return _w_nodes(math.log(self.w_min) - 25.0, math.log(self.w_min), (), 2.5)

    def S(self) -> float:
        n2 = _SHELL[self.shell][2]
        w, dw = self.allowed(1e4)
        return n2 + float(np.dot(dw, _phi(self.shell, w, np.zeros_like(w))))

    def B(self, eta: float) -> float:
        w, dw = self.allowed(eta)
        return float(
            np.dot(dw, _qint(self.shell, w, w * w / (4 * eta), np.full_like(w, _Q_TOP), sub0=False))
        )

    def minus_C(self, eta: float) -> float:
        """Walske 1952 p. 1285, exact (the last term is the one eq. 8 drops)."""
        e4 = 4.0 * eta
        w, dw = self.allowed(eta)
        qm = w * w / e4
        top = np.full_like(w, _Q_TOP)
        low = w <= e4
        f = np.where(
            low,
            _qint(self.shell, w, np.full_like(w, e4), top, sub0=False)
            - _qint(self.shell, w, np.zeros_like(w), np.minimum(qm, e4), sub0=True),
            _qint(self.shell, w, np.maximum(qm, e4), top, sub0=False)
            - _qint(self.shell, w, np.zeros_like(w), np.full_like(w, e4), sub0=True)
            + _phi(self.shell, w, np.zeros_like(w)) * 2.0 * np.log(np.maximum(w, e4) / e4),
        )
        wf, dwf = self.forbidden()
        extra = _qint(self.shell, wf, np.full_like(wf, e4), np.full_like(wf, _Q_TOP), sub0=False)
        return float(np.dot(dw, f) + np.dot(dwf, extra))


@cache
def _spectrum(shell: Shell, theta: float | None) -> _Spectrum:
    if theta is None:  # the hydrogen atom
        n = np.arange(2, _N_LINES + 1, dtype=np.float64)
        lines = (1.0 - 1.0 / n**2, 2.0 / n**3)
        return _Spectrum("K", 1.0 - 1.0 / (_N_LINES + 0.5) ** 2, lines)
    if not 0.0 < theta <= 1.0:
        raise ValueError(f"theta must be in (0, 1], got {theta!r}")
    return _Spectrum(shell, theta / _SHELL[shell][2])


@cache
def _S_T(shell: Shell, theta: float | None) -> tuple[float, float]:
    sp = _spectrum(shell, theta)
    s = sp.S()
    eta = 1000.0
    return s, sp.B(eta) - sp.minus_C(eta) - s * math.log(eta)


def _C(shell: Shell, theta: float | None, eta: float) -> float:
    sp = _spectrum(shell, theta)
    if eta >= _ETA_SWITCH:
        return -sp.minus_C(eta)
    s, t = _S_T(shell, theta)
    return s * math.log(eta) + t - sp.B(eta)


def walske_asymptotic_constants(shell: Shell, theta: float) -> tuple[float, float]:
    """``(S, T)`` of ``B = S ln eta + T - C`` for Walske's K (2 e) or L (8 e) shell."""
    _check_shell(shell)
    return _S_T(shell, float(theta))


def walske_stopping_number(shell: Shell, theta: float, eta: npt.ArrayLike) -> FloatArray:
    """Walske's ``B(theta, eta)`` (1952 eq. 5) of the K (2 e) or L (8 e) shell."""
    _check_shell(shell)
    sp = _spectrum(shell, float(theta))
    return np.array([sp.B(float(e)) for e in np.atleast_1d(np.asarray(eta, dtype=float))])


def walske_shell_correction(shell: Shell, theta: float, eta: npt.ArrayLike) -> FloatArray:
    """Walske's ``C(theta, eta)`` of the K (2 electrons) or L (8 electrons) shell."""
    _check_shell(shell)
    return np.array([_C(shell, float(theta), float(e)) for e in _positive(eta)])


def hydrogen_shell_correction(eta: npt.ArrayLike) -> FloatArray:
    """``C(eta)`` of the hydrogen atom (one electron), ``eta = m v^2 / 2 Ry``.

    Walske's K-shell formula with ``Z_eff = 1`` and hydrogen's true discrete lines;
    half of Brown/Walske's two-electron ``C(W_min = 0, eta)`` (1952 eqs. 14, 17).
    """
    return np.array([0.5 * _C("K", None, float(e)) for e in _positive(eta)])


def _positive(eta: npt.ArrayLike) -> FloatArray:
    e = np.atleast_1d(np.asarray(eta, dtype=np.float64))
    if (e <= 0).any():
        raise ValueError("eta must be > 0")
    return e


def _check_shell(shell: str) -> None:
    if shell not in get_args(Shell):
        raise ValueError(f"shell must be one of {get_args(Shell)}, got {shell!r}")


# --- water ---------------------------------------------------------------------------------

#: Default (calibrated) oxygen thetas, see the module docstring.
DEFAULT_THETA_K = 0.7
DEFAULT_THETA_L = 0.5
#: eta grid of the tabulated corrections: 20 nodes per decade over [0.02, 1e5].
ETA_GRID: FloatArray = np.geomspace(
    0.02, 1e5, round(20 * math.log10(1e5 / 0.02)) + 1, dtype=np.float64
)
SHIPPED_TABLE = "shell_corrections_water.csv"
_FINE_STRUCTURE = 1.0 / 137.035999084
_Z_O = 8


def shell_table(theta_K: float, theta_L: float) -> dict[str, FloatArray]:
    """``C(eta)`` of the H atom, the O K shell and Walske's L shell on :data:`ETA_GRID`."""
    return {
        "eta": ETA_GRID.copy(),
        "C_H": hydrogen_shell_correction(ETA_GRID),
        "C_K": walske_shell_correction("K", theta_K, ETA_GRID),
        "C_L": walske_shell_correction("L", theta_L, ETA_GRID),
    }


@cache
def _shipped_table() -> tuple[dict[str, str], dict[str, FloatArray]]:
    text = resources.files(__package__).joinpath("data").joinpath(SHIPPED_TABLE)
    lines = text.read_text(encoding="utf-8").splitlines()
    meta = dict(line[3:].split(" = ", 1) for line in lines if line.startswith("#@ "))
    rows = list(csv.DictReader(line for line in lines if not line.startswith("#")))
    return meta, {k: np.array([float(r[k]) for r in rows]) for k in rows[0]}


@cache
def _splines(theta_K: float, theta_L: float) -> dict[str, CubicSpline]:
    meta, shipped = _shipped_table()
    table = dict(shipped)  # C_H does not depend on theta
    if float(meta["theta_K"]) != theta_K:
        table["C_K"] = walske_shell_correction("K", theta_K, table["eta"])
    if float(meta["theta_L"]) != theta_L:
        table["C_L"] = walske_shell_correction("L", theta_L, table["eta"])
    x = np.log(table["eta"])
    return {k: CubicSpline(x, table[k]) for k in ("C_H", "C_K", "C_L")}


@dataclass(frozen=True)
class WaterShellCorrection:
    """``C/Z`` of liquid water as a sum of hydrogenic shells (module docstring)."""

    theta_K: float = DEFAULT_THETA_K
    theta_L: float = DEFAULT_THETA_L
    l_shell_electrons: int = 6
    evaluation: Evaluation = "interpolate"

    def __post_init__(self) -> None:
        if self.evaluation not in get_args(Evaluation):
            raise ValueError(f"evaluation must be one of {get_args(Evaluation)}")
        for name in ("theta_K", "theta_L"):
            if not 0.0 < getattr(self, name) <= 1.0:
                raise ValueError(f"{name} must be in (0, 1]")
        if not 0 <= self.l_shell_electrons <= 8:
            raise ValueError("l_shell_electrons must be in 0..8")

    @property
    def eta_range(self) -> tuple[float, float]:
        """``eta`` of every component must lie here (interpolation: the table's range)."""
        return float(ETA_GRID[0]), float(ETA_GRID[-1])

    def etas(self, beta2: FloatArray) -> dict[str, FloatArray]:
        """``eta = m v^2 / (2 Z_eff^2 Ry) = beta^2 / (alpha Z_eff)^2`` per component."""
        y = np.asarray(beta2, dtype=np.float64) / _FINE_STRUCTURE**2
        return {
            "C_H": y,
            "C_K": y / (_Z_O - SCREENING["K"]) ** 2,
            "C_L": y / (_Z_O - SCREENING["L"]) ** 2,
        }

    def beta2_range(self) -> tuple[float, float]:
        lo, hi = self.eta_range
        a2 = _FINE_STRUCTURE**2
        return lo * a2 * (_Z_O - SCREENING["K"]) ** 2, min(1.0, hi * a2)

    def components(self, beta2: npt.ArrayLike) -> dict[str, FloatArray]:
        b2 = np.atleast_1d(np.asarray(beta2, dtype=np.float64))
        lo, hi = self.beta2_range()
        if b2.size and (b2.min() < lo or b2.max() > hi):
            raise OutOfRangeError(
                (float(b2.min()), float(b2.max())),
                (lo, hi),
                source="water shell correction (beta^2)",
            )
        etas = self.etas(b2)
        if self.evaluation == "direct":
            return {
                "C_H": hydrogen_shell_correction(etas["C_H"]),
                "C_K": walske_shell_correction("K", self.theta_K, etas["C_K"]),
                "C_L": walske_shell_correction("L", self.theta_L, etas["C_L"]),
            }
        sp = _splines(self.theta_K, self.theta_L)
        return {k: np.asarray(sp[k](np.log(etas[k])), dtype=np.float64) for k in sp}

    def c_over_z(self, beta2: npt.ArrayLike) -> FloatArray:
        """``C/Z`` per electron of H2O (Z = 10): (2 C_H + C_K + (n_L/8) C_L) / 10."""
        c = self.components(beta2)
        out: FloatArray = (
            2.0 * c["C_H"] + c["C_K"] + (self.l_shell_electrons / 8.0) * c["C_L"]
        ) / 10.0
        return out

    def describe(self) -> str:
        return (
            f"Walske hydrogenic shells (H atom; O K theta={self.theta_K:g}, "
            f"O L theta={self.theta_L:g} x {self.l_shell_electrons}/8), {self.evaluation}"
        )


__all__ = [
    "DEFAULT_THETA_K",
    "DEFAULT_THETA_L",
    "ETA_GRID",
    "SCREENING",
    "WALSKE_1952",
    "WALSKE_1956",
    "WaterShellCorrection",
    "hydrogen_shell_correction",
    "shell_table",
    "walske_asymptotic_constants",
    "walske_shell_correction",
    "walske_stopping_number",
]
