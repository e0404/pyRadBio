"""Matsuya et al. 2025 (IMK): digitized Figs. 2, 4 and 7, and reconstruction helpers.

Matsuya Y, Saga R, Wang Y & Sato T, *Med. Phys.* **52**(10):e70040 (2025),
doi:10.1002/mp.70040. Model equations and parameters:
``docs/references/IMK_Matsuya2025.md``.

Contents
========
* Digitized curves and their fits: [`get_fig2_curve`][] (survival, sec. 3.1.1),
  [`get_fig4_curve`][] (micronuclei, sec. 3.1.2) and [`get_fig7_curve`][]
  (dose rate, secs. 3.1.3-3.1.4), each described in its section below.
* Values recovered from them: [`RECOVERED_Z_STAR_1D_GY`][], [`ANCHORS`][]
  (the four carbon fit points, secs. 3.1.1 and 4.1), [`recovered_z_star_1D`][],
  [`recovered_h`][], [`recovered_z_star_1D_from_mn`][] and
  [`paper_rbe_alpha`][] (eq. 6 on the fig. 2 lines).
* Reconstruction helpers: [`microdosimetry_routes`][] with [`PAPER_ROUTE`][]
  (the four ``z*_1D`` routes for one table-1 cell line, sec. 2.1),
  [`fig7_schedule`][], [`fig7_protocol`][], [`relative_mn`][] and
  [`dose_rate_shape_fit`][] (fig. 7, secs. 3.1.3-3.1.4).

Fig. 2 -- cell survival
=======================
Two panels, each with three calculated (model) curves and the
corresponding experimental points:

* **(a) HSG** -- 200 kVp X-rays, and 290 MeV/n carbon at LET 13 and 100 keV/um.
* **(b) SCC VII** -- X-rays, and carbon at LET 46 and 80 keV/um.

Hand-digitized from the published PDF (its figures are raster, not vector, so
the vector-extraction route used elsewhere in this package does not apply).
Experimental points carry ``survival_lo``/``survival_hi`` read off the error
bars; the bar ends are roughly geometric about the centre because the axis is
logarithmic.

Why this panel is worth digitizing
----------------------------------
The paper computes every ``y*`` with PHITS and prints none of them, which
otherwise leaves ``alpha(photon)`` -- the denominator of every RBE -- unavailable
(see [`IMKModel`][]). But the *calculated* curves are
plain LQ, so fitting one and inverting the model's own eq. (1) with the printed
``alpha0``/``beta0`` recovers the ``z*_1D`` that produced it:

```text
z*_1D = (alpha_fitted - alpha0) / beta0
```

[`recovered_z_star_1D`][] does this. The results (see
[`RECOVERED_Z_STAR_1D_GY`][]) are self-validating in three ways:

1. every fitted ``beta`` lands on the printed ``beta0`` -- all six curves within
   0.5%, against HSG's 0.0467 and SCC's 0.0298 -- which checks the
   digitization's calibration *and* the reading ``beta = F beta0`` with
   ``F ~ 1`` for acute delivery;
2. the two panels are independent digitizations of different cell lines, yet
   recover the same photon ``y*`` to 5.6% (3.62 vs 3.41 keV/um) at the same
   domain size;
3. the recovered photon ``y*`` of ~3.5 keV/um is a physically sensible value for
   a kV X-ray beam in a 0.84 um site, which the model never forces.

Two traps this closes
---------------------
* **The paper's HSG photon fit is not Furusawa's.** Its own X-ray points give
  ``alpha = 0.197``, ``beta = 0.0454``, and its line 0.1991/0.0465; Furusawa
  et al. 2000 report ``alpha_x = 0.313``, ``beta_x = 0.061`` for HSG at 200 kVp.
  A factor 1.6. Do not substitute the Furusawa photon LQ -- nor this
  package's mMKM HSG defaults, which are in the same family -- as this paper's
  reference.
* **In fig. 2a the two experimental carbon series pair with the opposite
  calculated line**, by a factor 8-10 in RMS log-residual (0.107/0.082 swapped
  versus 0.821/0.842 as labelled). They are stored here under the LET they
  actually match; [`series_pairing_residuals`][] reproduces the evidence, and
  ``tests/benchmark/test_matsuya2025.py`` pins it. Whether the swap is in the
  digitization or in the figure's own symbol legend is not settled here.

Caveat on ``radiation="x-rays"``: panel (a)'s photon arm is labelled 200 kVp in
the figure, panel (b)'s is labelled only "X-rays", so the SCC photon energy is
recorded as unspecified rather than assumed.

Fig. 4 -- micronuclei
=====================
The same three HSG qualities as fig. 2a (X-rays, carbon at LET 13 and
100 keV/um), plotted as micronuclei per binucleated cell on **linear** axes, so
its error bars are symmetric (only the centre and the bar top were traced; the
lower end is reconstructed). Its legend states the table-1 parameters.

This panel is a **zero-free-parameter prediction**, and that is the whole reason
to hold it: eq. (3) says ``F_MN = h <w>``, so once fig. 2a has given the paper's
own ``alpha``/``beta`` for these exact three radiations, the fig. 4 curves are
forced to be ``h x`` those. Fitting them measures ``h`` instead of assuming it,
and it comes out at **0.275 -- table 1's cell-line-independent value, not table
2's HSG-specific 0.245** (which is 11% off, an order of magnitude outside the
0.6% residual the table-1 value leaves). [`recovered_h`][] does this.

Fitting fig. 4 on its own then re-derives the paper's unprinted PHITS inputs a
second time, from a different figure: 1.039 / 2.905 / 16.420 Gy against fig.
2a's 1.045 / 2.923 / 16.184 ([`RECOVERED_Z_STAR_1D_GY`][]), i.e. 0.6%, 0.6%
and 1.5%. Two independent hand digitizations of two different panels agreeing
to that level is the strongest available check on the recovery in
[`recovered_z_star_1D`][].

Unlike fig. 2a, **fig. 4's experimental points are not swapped**: its LET-100
series sits on its own calculated line (RMS relative residual 0.11, against 1.6
and 2.4 for the other two). Its X-ray and LET-13 series are *not* separable this
way -- the two calculated curves differ by only ~30% at 5 Gy, well inside the
scatter -- so fig. 4 confirms no swap for carbon-100 and is simply silent on the
other pair.

One digitization floor to know about: the y axis resolves ~0.0011 MN per
binucleated cell per pixel, and the three lowest traced points sit at exactly
one, two and four times that. [`FIG4_MIN_FIT_DOSE_GY`][] keeps them out of
every fit; they are still stored, because the file is the digitization.

Fig. 7 -- the dose-rate axis
===========================
DU145 (a) and WI-38 (b), 4 Gy total delivered at 1 to 190 Gy/h: the model's mean
line, its 95% CI (from the paper's MCMC parameter uncertainty -- context only,
we have no route to reproduce it) and four measured points each. The x axis is
logarithmic and was traced linearly, the traced value being the exponent; the
stored ``dose_rate_Gy_per_h`` is ``10**x``.

**The y quantity is relative, not absolute.** Sec. 3.1.3: "*the relative MN
frequencies were normalized using those at high dose rate (i.e., at 60 Gy/h)*".
So the curve equals 1 at 60 Gy/h by construction -- the digitized experimental
point there reads 1.0002 (DU145) and 0.9979 (WI-38), which doubles as a check on
the y calibration. Reading these as absolute MN counts is the panel's one trap,
and it is a tempting one, because both curves happen to top out near 1.

**What this panel actually validates.** Eq. (1) makes the relative frequency
affine in the Lea-Catcheside factor, so [`dose_rate_shape_fit`][] fits
``A + B F(rate)`` with ``F`` supplied by the caller and tests the *shape* alone.
On DU145 that lands within **0.0019** of a 58-point curve spanning 0.45 --
and halving or doubling the printed ``a+c = 2.10/h`` degrades it 25-fold, to
0.048 and 0.053. That is a sharp, published-model-line test of
[`lea_catcheside_g`][], not a soft one.

**WI-38 is not a test, and must not be reported as one.** Its whole effect is
0.028 in y, and the same fit gives 0.0041 / 0.0039 / 0.0042 at the printed rate,
half it and double it -- i.e. the residual is *insensitive to the parameter the
panel is nominally about*. It confirms "no dose-rate dependence" and nothing
more. For the same reason its ``alpha`` is unidentifiable here: a one-parameter
fit of the photon ``z*_1D`` bottoms out at 0.0 Gy and barely moves out to 11 Gy.

**Neither panel pins the photon reference.** DU145's best-fit photon ``z*_1D``
is 0.736 Gy, but the RMS only doubles outside [0.255, 1.299] Gy -- consistent
with fig. 2's 0.99-1.05 and with nothing else useful. That is why
[`imk`][pyradbio.models.mkm.imk] still leaves DU145 and WI-38 without a photon
reference: this panel does not supply one, however tempting the number looks.

Delivery: the paper's line is **continuous**, not its sec. 3.1.4 fractionated
protocol. It makes almost no difference here -- ``F`` differs by <0.1% at all
four measured rates -- but the line is smooth in rate, and continuous is what
reproduces it.
"""

from __future__ import annotations

import csv
import math
from collections import defaultdict
from collections.abc import Callable
from dataclasses import dataclass
from functools import lru_cache
from importlib import resources
from typing import TYPE_CHECKING

import numpy as np
import numpy.typing as npt

from pyradbio.core.lq import LQParameters
from pyradbio.core.protraction import DoseSchedule

if TYPE_CHECKING:  # the engines are imported where they are used, not by the loaders
    from pyradbio.models.mkm.imk import IMKModel
    from pyradbio.track.stopping_power import StoppingPower

MATSUYA2025_DOI = "10.1002/mp.70040"

_DATA_DIR = "data"
_FIG2_FILE = "matsuya2025_fig2_survival.csv"
_FIG4_FILE = "matsuya2025_fig4_micronuclei.csv"
_FIG7_FILE = "matsuya2025_fig7_dose_rate.csv"

FIG4_MIN_FIT_DOSE_GY: float = 0.05
"""Doses below this are dropped from every fig. 4 fit. The panel's y axis
resolves ~0.0011 MN/BN-cell per pixel and its three lowest traced points sit
at one, two and four times that, so their *relative* value is quantization,
not signal -- the worst of them is a factor 4 off its own curve.
"""

RECOVERED_Z_STAR_1D_GY: dict[tuple[str, float], float] = {
    ("HSG", 0.0): 1.045,
    ("HSG", 13.0): 2.923,
    ("HSG", 100.0): 16.184,
    ("SCC", 0.0): 0.987,
    ("SCC", 46.0): 7.007,
    ("SCC", 80.0): 10.825,
}
"""``z*_1D`` [Gy] recovered from each calculated curve by [`recovered_z_star_1D`][]
with the printed table-1 ``alpha0``/``beta0``, fitted over the full traced
range. Keyed by ``(cell_line, let_keV_um)``, photons at ``0.0``.
**These are the paper's own PHITS inputs, not a fit of ours.**
"""

ANCHORS: tuple[tuple[str, float], ...] = (
    ("HSG", 13.0),
    ("SCC", 46.0),
    ("SCC", 80.0),
    ("HSG", 100.0),
)
"""The paper's four carbon fit points (secs. 3.1.1, 4.1), in LET order: the
``(cell_line, let_keV_um)`` keys of [`RECOVERED_Z_STAR_1D_GY`][] that are
not photons -- fig. 2a's HSG at 13 and 100 keV/um, fig. 2b's SCC at 46 and 80.
"""


@dataclass(frozen=True, slots=True)
class Fig2Curve:
    """One digitized series from Fig. 2.

    ``series`` is ``"cal"`` (the paper's model curve) or ``"exp"`` (its measured
    points, with error-bar ends in ``survival_lo``/``survival_hi``).
    ``let_keV_um`` is ``0.0`` for the photon arms.
    """

    cell_line: str  # "HSG" | "SCC"
    panel: str  # "a" | "b"
    series: str  # "cal" | "exp"
    radiation: str  # "12C" | "x-rays"
    let_keV_um: float
    dose_Gy: tuple[float, ...]
    survival: tuple[float, ...]
    survival_lo: tuple[float, ...] = ()
    survival_hi: tuple[float, ...] = ()
    doi: str = MATSUYA2025_DOI

    @property
    def dose(self) -> npt.NDArray[np.float64]:
        return np.asarray(self.dose_Gy, dtype=np.float64)

    @property
    def s(self) -> npt.NDArray[np.float64]:
        return np.asarray(self.survival, dtype=np.float64)

    @property
    def label(self) -> str:
        return "X-rays" if self.radiation == "x-rays" else f"C {self.let_keV_um:.0f} keV/um"

    def error_bars(self) -> tuple[npt.NDArray[np.float64], npt.NDArray[np.float64]]:
        """Lower/upper *offsets* from the centre, as matplotlib's ``yerr`` wants."""
        if self.series != "exp":
            raise ValueError("only the experimental series carry error bars")
        lo = np.asarray(self.survival_lo, dtype=np.float64)
        hi = np.asarray(self.survival_hi, dtype=np.float64)
        return self.s - lo, hi - self.s

    def lq_fit(
        self, min_log_kill: float = 0.0, max_log_kill: float = math.inf
    ) -> tuple[float, float]:
        """Least-squares ``(alpha, beta)`` of ``-ln S = alpha D + beta D^2``.

        **Fitted over the full traced range by default**, because a calculated
        curve *is* exactly LQ: every digitized point is signal, and the optional
        window (on ``-ln S``, not on dose) is there for sensitivity checks only.
        Measured: over the full range all six calculated curves return ``beta``
        within 0.5% of the printed ``beta0``, while a ``[0.1, 6]`` window --
        which looks reasonable, and is what the LEM reconstructions here use for
        *simulated* curves -- puts HSG's LET-100 line 4.9% off, because it
        discards the high-dose arm that determines ``beta`` in the first place.
        """
        dose, s = self.dose, self.s
        keep = (dose > 1.0e-3) & (s > 0.0)
        dose, effect = dose[keep], -np.log(s[keep])
        window = (effect >= min_log_kill) & (effect <= max_log_kill)
        dose, effect = dose[window], effect[window]
        if dose.size < 2:
            raise ValueError(f"{self.cell_line} {self.label} {self.series}: too few points to fit")
        design = np.vstack([dose, dose**2]).T
        (alpha, beta), *_ = np.linalg.lstsq(design, effect, rcond=None)
        return float(alpha), float(beta)


@lru_cache(maxsize=1)
def load_fig2_curves() -> tuple[Fig2Curve, ...]:
    """All twelve digitized Fig. 2 series (2 cell lines x 3 radiations x 2 series)."""
    text = (
        resources.files(__package__)
        .joinpath(_DATA_DIR)
        .joinpath(_FIG2_FILE)
        .read_text(encoding="utf-8")
    )
    grouped: dict[tuple[str, str, float], list[tuple[float, float, float, float]]] = defaultdict(
        list
    )
    meta: dict[tuple[str, str, float], tuple[str, str]] = {}
    for row in csv.DictReader(_data_lines(text)):
        key = (row["cell_line"], row["series"], float(row["let_keV_um"]))
        lo = float(row["survival_lo"]) if row["survival_lo"] else float("nan")
        hi = float(row["survival_hi"]) if row["survival_hi"] else float("nan")
        grouped[key].append((float(row["dose_Gy"]), float(row["survival"]), lo, hi))
        meta[key] = (row["panel"], row["radiation"])
    curves = []
    for (cell_line, series, let), points in sorted(grouped.items()):
        points.sort()
        panel, radiation = meta[(cell_line, series, let)]
        dose = tuple(row[0] for row in points)
        surv = tuple(row[1] for row in points)
        bar_lo = tuple(row[2] for row in points)
        bar_hi = tuple(row[3] for row in points)
        curves.append(
            Fig2Curve(
                cell_line=cell_line,
                panel=panel,
                series=series,
                radiation=radiation,
                let_keV_um=let,
                dose_Gy=dose,
                survival=surv,
                survival_lo=() if series == "cal" else bar_lo,
                survival_hi=() if series == "cal" else bar_hi,
            )
        )
    return tuple(curves)


def _data_lines(text: str) -> list[str]:
    """CSV lines without the ``#`` citation header the data files carry."""
    return [line for line in text.splitlines() if not line.startswith("#")]


def get_fig2_curve(*, cell_line: str, series: str, let_keV_um: float) -> Fig2Curve:
    """The single series matching all three keys (photons are ``let_keV_um=0.0``)."""
    for curve in load_fig2_curves():
        if (
            curve.cell_line == cell_line
            and curve.series == series
            and curve.let_keV_um == let_keV_um
        ):
            return curve
    raise KeyError(f"no Fig. 2 series for {cell_line!r} {series!r} at LET {let_keV_um}")


def recovered_z_star_1D(
    curve: Fig2Curve, *, alpha0_Gy_inv: float, beta0_Gy2: float
) -> tuple[float, float, float]:
    """``(alpha, beta, z*_1D)`` for one calculated curve, inverting eq. (1).

    ``z*_1D = (alpha - alpha0)/beta0`` is the saturation-corrected dose-mean
    specific energy the paper fed the model -- i.e. its PHITS ``y*`` divided by
    ``rho pi r_d^2``. Only meaningful for ``series="cal"``; applying it to the
    experimental points would fold their scatter into a quantity that is a model
    *input*.
    """
    if curve.series != "cal":
        raise ValueError("z*_1D can only be recovered from a calculated curve")
    alpha, beta = curve.lq_fit()
    return alpha, beta, (alpha - alpha0_Gy_inv) / beta0_Gy2


def series_pairing_residuals(cell_line: str) -> dict[tuple[float, float], float]:
    """RMS of ``log10(S_exp / S_cal)`` for every (experiment, model curve) pair.

    The diagnostic behind the fig. 2a swap documented in the module docstring:
    keys are ``(exp LET, cal LET)`` and a *small* off-diagonal value means the
    experimental series belongs to the other curve. Evaluated at the measured
    doses, so it does not depend on either LQ fit.
    """
    curves = [c for c in load_fig2_curves() if c.cell_line == cell_line]
    out: dict[tuple[float, float], float] = {}
    for exp in (c for c in curves if c.series == "exp"):
        for cal in (c for c in curves if c.series == "cal"):
            alpha, beta = cal.lq_fit()
            predicted = np.exp(-(alpha * exp.dose + beta * exp.dose**2))
            out[(exp.let_keV_um, cal.let_keV_um)] = float(
                np.sqrt(np.mean(np.log10(exp.s / predicted) ** 2))
            )
    return out


@dataclass(frozen=True, slots=True)
class Fig4Curve:
    """One digitized series from Fig. 4 -- micronuclei per binucleated cell.

    ``series`` is ``"cal"`` (the paper's model curve) or ``"exp"``. The panel's
    axes are linear and its bars symmetric, so ``mn_lo`` is the reflection of the
    traced ``mn_hi`` about the centre rather than a second reading.
    """

    cell_line: str  # "HSG"
    series: str  # "cal" | "exp"
    radiation: str  # "12C" | "x-rays"
    let_keV_um: float
    dose_Gy: tuple[float, ...]
    mn_per_bn_cell: tuple[float, ...]
    mn_lo: tuple[float, ...] = ()
    mn_hi: tuple[float, ...] = ()
    doi: str = MATSUYA2025_DOI

    @property
    def dose(self) -> npt.NDArray[np.float64]:
        return np.asarray(self.dose_Gy, dtype=np.float64)

    @property
    def mn(self) -> npt.NDArray[np.float64]:
        return np.asarray(self.mn_per_bn_cell, dtype=np.float64)

    @property
    def label(self) -> str:
        return "X-rays" if self.radiation == "x-rays" else f"C {self.let_keV_um:.0f} keV/um"

    def error_bars(self) -> tuple[npt.NDArray[np.float64], npt.NDArray[np.float64]]:
        """Lower/upper *offsets* from the centre, as matplotlib's ``yerr`` wants."""
        if self.series != "exp":
            raise ValueError("only the experimental series carry error bars")
        lo = np.asarray(self.mn_lo, dtype=np.float64)
        hi = np.asarray(self.mn_hi, dtype=np.float64)
        return self.mn - lo, hi - self.mn

    def lq_fit(self, min_dose_Gy: float = FIG4_MIN_FIT_DOSE_GY) -> tuple[float, float]:
        """Least-squares ``(alpha_m, beta_m)`` of ``F_MN = alpha_m D + beta_m D^2``.

        Eq. (3)'s micronucleus LQ, i.e. the survival LQ times ``h``. The default
        ``min_dose_Gy`` is [`FIG4_MIN_FIT_DOSE_GY`][]; see its note for why the
        cut is on dose here and on ``-ln S`` in [`Fig2Curve.lq_fit`][].
        """
        dose, mn = self.dose, self.mn
        keep = dose > min_dose_Gy
        dose, mn = dose[keep], mn[keep]
        if dose.size < 2:
            raise ValueError(f"{self.cell_line} {self.label} {self.series}: too few points to fit")
        design = np.vstack([dose, dose**2]).T
        (alpha_m, beta_m), *_ = np.linalg.lstsq(design, mn, rcond=None)
        return float(alpha_m), float(beta_m)


@lru_cache(maxsize=1)
def load_fig4_curves() -> tuple[Fig4Curve, ...]:
    """All six digitized Fig. 4 series (3 radiations x {calculated, measured})."""
    text = (
        resources.files(__package__)
        .joinpath(_DATA_DIR)
        .joinpath(_FIG4_FILE)
        .read_text(encoding="utf-8")
    )
    grouped: dict[tuple[str, str, float], list[tuple[float, float, float, float]]] = defaultdict(
        list
    )
    meta: dict[tuple[str, str, float], str] = {}
    for row in csv.DictReader(_data_lines(text)):
        key = (row["cell_line"], row["series"], float(row["let_keV_um"]))
        lo = float(row["mn_lo"]) if row["mn_lo"] else float("nan")
        hi = float(row["mn_hi"]) if row["mn_hi"] else float("nan")
        grouped[key].append((float(row["dose_Gy"]), float(row["mn_per_bn_cell"]), lo, hi))
        meta[key] = row["radiation"]
    curves = []
    for (cell_line, series, let), points in sorted(grouped.items()):
        points.sort()
        dose = tuple(row[0] for row in points)
        mn = tuple(row[1] for row in points)
        bar_lo = tuple(row[2] for row in points)
        bar_hi = tuple(row[3] for row in points)
        curves.append(
            Fig4Curve(
                cell_line=cell_line,
                series=series,
                radiation=meta[(cell_line, series, let)],
                let_keV_um=let,
                dose_Gy=dose,
                mn_per_bn_cell=mn,
                mn_lo=() if series == "cal" else bar_lo,
                mn_hi=() if series == "cal" else bar_hi,
            )
        )
    return tuple(curves)


def get_fig4_curve(*, series: str, let_keV_um: float, cell_line: str = "HSG") -> Fig4Curve:
    """The single Fig. 4 series matching all three keys (photons are ``0.0``)."""
    for curve in load_fig4_curves():
        if (
            curve.cell_line == cell_line
            and curve.series == series
            and curve.let_keV_um == let_keV_um
        ):
            return curve
    raise KeyError(f"no Fig. 4 series for {cell_line!r} {series!r} at LET {let_keV_um}")


def recovered_h(curve: Fig4Curve, *, beta0_Gy2: float) -> tuple[float, float, float]:
    """``(alpha_m, beta_m, h)`` for one calculated Fig. 4 curve, inverting eq. (3).

    ``h = beta_m/beta0`` reads the micronucleus probability off the *quadratic*
    term, which eq. (1) fixes at ``beta0`` for every radiation quality -- so this
    needs no knowledge of the curve's ``z*_1D`` and is a genuine measurement of
    ``h``, not a restatement of the fit. All three curves return 0.275.
    """
    if curve.series != "cal":
        raise ValueError("h can only be recovered from a calculated curve")
    alpha_m, beta_m = curve.lq_fit()
    return alpha_m, beta_m, beta_m / beta0_Gy2


def recovered_z_star_1D_from_mn(
    curve: Fig4Curve, *, alpha0_Gy_inv: float, beta0_Gy2: float
) -> tuple[float, float]:
    """``(h, z*_1D)`` from a calculated Fig. 4 curve alone.

    Undoes eq. (3) with the ``h`` the same curve implies, then inverts eq. (1)
    exactly as [`recovered_z_star_1D`][] does for fig. 2 -- giving a wholly
    independent reading of the paper's unprinted PHITS inputs from a different
    panel. See the module docstring for how closely the two agree.
    """
    alpha_m, _beta_m, h = recovered_h(curve, beta0_Gy2=beta0_Gy2)
    return h, (alpha_m / h - alpha0_Gy_inv) / beta0_Gy2


@dataclass(frozen=True, slots=True)
class Fig7Curve:
    """One digitized series from Fig. 7 -- MN frequency vs dose rate, 4 Gy total.

    ``series`` is ``"cal"`` (the model's mean line), ``"cal_ci_lo"`` /
    ``"cal_ci_hi"`` (its 95% CI, from the paper's MCMC parameter uncertainty --
    we have no route to reproduce those, they are context) or ``"exp"``.

    ``relative_mn`` is **normalised to the 60 Gy/h arm** (sec. 3.1.3: "*the
    relative MN frequencies were normalized using those at high dose rate*"), so
    it is dimensionless and equals 1 there by construction -- not an absolute MN
    count. Reading it as absolute is the panel's one trap.
    """

    cell_line: str  # "DU145" | "WI-38"
    panel: str  # "a" | "b"
    series: str  # "cal" | "cal_ci_lo" | "cal_ci_hi" | "exp"
    dose_rate_Gy_per_h: tuple[float, ...]
    relative_mn: tuple[float, ...]
    mn_lo: tuple[float, ...] = ()
    mn_hi: tuple[float, ...] = ()
    total_dose_Gy: float = 4.0
    normalisation_rate_Gy_per_h: float = 60.0
    doi: str = MATSUYA2025_DOI

    @property
    def rate(self) -> npt.NDArray[np.float64]:
        return np.asarray(self.dose_rate_Gy_per_h, dtype=np.float64)

    @property
    def mn(self) -> npt.NDArray[np.float64]:
        return np.asarray(self.relative_mn, dtype=np.float64)

    def error_bars(self) -> tuple[npt.NDArray[np.float64], npt.NDArray[np.float64]]:
        """Lower/upper *offsets* from the centre, as matplotlib's ``yerr`` wants."""
        if self.series != "exp":
            raise ValueError("only the experimental series carry error bars")
        lo = np.asarray(self.mn_lo, dtype=np.float64)
        hi = np.asarray(self.mn_hi, dtype=np.float64)
        return self.mn - lo, hi - self.mn


@lru_cache(maxsize=1)
def load_fig7_curves() -> tuple[Fig7Curve, ...]:
    """All eight digitized Fig. 7 series (2 cell lines x {mean, 2 CI bounds, data})."""
    text = (
        resources.files(__package__)
        .joinpath(_DATA_DIR)
        .joinpath(_FIG7_FILE)
        .read_text(encoding="utf-8")
    )
    grouped: dict[tuple[str, str], list[tuple[float, float, float, float]]] = defaultdict(list)
    meta: dict[tuple[str, str], str] = {}
    for row in csv.DictReader(_data_lines(text)):
        key = (row["cell_line"], row["series"])
        lo = float(row["mn_lo"]) if row["mn_lo"] else float("nan")
        hi = float(row["mn_hi"]) if row["mn_hi"] else float("nan")
        grouped[key].append((float(row["dose_rate_Gy_per_h"]), float(row["relative_mn"]), lo, hi))
        meta[key] = row["panel"]
    curves = []
    for (cell_line, series), points in sorted(grouped.items()):
        points.sort()
        bar_lo = tuple(row[2] for row in points)
        bar_hi = tuple(row[3] for row in points)
        curves.append(
            Fig7Curve(
                cell_line=cell_line,
                panel=meta[(cell_line, series)],
                series=series,
                dose_rate_Gy_per_h=tuple(row[0] for row in points),
                relative_mn=tuple(row[1] for row in points),
                mn_lo=bar_lo if series == "exp" else (),
                mn_hi=bar_hi if series == "exp" else (),
            )
        )
    return tuple(curves)


def get_fig7_curve(*, cell_line: str, series: str) -> Fig7Curve:
    """The single Fig. 7 series for this cell line and series kind."""
    for curve in load_fig7_curves():
        if curve.cell_line == cell_line and curve.series == series:
            return curve
    raise KeyError(f"no Fig. 7 series for {cell_line!r} {series!r}")


def dose_rate_shape_fit(
    curve: Fig7Curve, time_factor: Callable[[float], float]
) -> tuple[float, float, float]:
    """``(A, B, max|residual|)`` of ``relative_MN = A + B F(rate)``.

    Eq. (1) makes the relative MN frequency affine in the Lea-Catcheside factor
    -- ``A`` carries the (rate-independent) linear term and ``B`` the quadratic
    one -- so a two-parameter fit with ``F`` supplied by the caller tests the
    *shape* alone, with no reference to ``alpha`` or to the normalisation. Pass
    ``time_factor`` as a callable of the dose rate in Gy/h, e.g.:

    ```text
    model.time_factor(DoseSchedule.continuous(4.0, duration_h=4.0 / rate))
    ```

    See the module docstring for what the residual is worth on each panel: it
    discriminates sharply on DU145 and not at all on WI-38.
    """
    if curve.series != "cal":
        raise ValueError("the shape fit applies to the calculated mean line")
    rate = curve.rate
    f = np.array([time_factor(float(r)) for r in rate])
    design = np.vstack([np.ones_like(f), f]).T
    (a_fit, b_fit), *_ = np.linalg.lstsq(design, curve.mn, rcond=None)
    residual = design @ np.array([a_fit, b_fit]) - curve.mn
    return float(a_fit), float(b_fit), float(np.abs(residual).max())


# --------------------------------------------------------------------------
# Paper RBE_alpha from the fig. 2 lines
# --------------------------------------------------------------------------
def paper_rbe_alpha(cell_line: str, let_keV_um: float) -> float:
    """Eq. (6)'s ``RBE_alpha = alpha*/alpha(photon)`` from the paper's own fig. 2 lines.

    Both ``alpha`` come from [`Fig2Curve.lq_fit`][] of the calculated curves
    (carbon at ``let_keV_um`` and the same panel's X-ray arm), so this is the
    paper's model, not a measurement and not ours.
    """
    ion_line = get_fig2_curve(cell_line=cell_line, series="cal", let_keV_um=let_keV_um)
    photon_line = get_fig2_curve(cell_line=cell_line, series="cal", let_keV_um=0.0)
    return ion_line.lq_fit()[0] / photon_line.lq_fit()[0]


# --------------------------------------------------------------------------
# Microdosimetry routes for one table-1 cell line
# --------------------------------------------------------------------------
PAPER_ROUTE = "sato-AMF"
"""The route that follows *this paper's* own method. Two independent statements
pick it out: the domain is a **sphere** (sec. 2.1, "the shape is generally
assumed to be a simple sphere with a diameter of < 1.0 um"), and ``y*`` comes
from the PHITS ``[t-sed]`` tally (sec. 3.1.1), which evaluates **Sato's
analytical microdosimetric function** -- what
[`SatoLinealEnergyEngine`][pyradbio.micro.sato.SatoLinealEnergyEngine] implements, on a spherical
site. The MKM-family default (``"impact/cyl"``) is Inaniwa's *disk* convention,
right for the SMK/mMKM papers and not for this one. Faithful is not best: it is
not the closest-scoring route at mid LET (``docs/references/IMK_Matsuya2025.md``).
"""

_PRESETS = {
    "hsg": "hsg_matsuya2025",
    "scc": "scc_matsuya2025",
    "wi38": "wi38_matsuya2025",
    "du145": "du145_matsuya2025",
}
"""Normalised cell-line name -> the [`IMKModel`][]
table-1 preset constructor.
"""


def microdosimetry_routes(
    cell_line: str = "HSG", *, stopping_power: StoppingPower
) -> dict[str, IMKModel]:
    """One table-1 cell line, scored through four microdosimetry routes.

    Keys, in plotting order:

    * ``"impact/cyl"`` -- the MKM-family default: the analytic impact-parameter
      profile over a disk. No ``d(y)`` is built.
    * ``"impact/sph"`` -- the same over a sphere, the domain shape this paper
      states (sec. 2.1).
    * ``"sato-AMF"`` -- Sato's AMF single-event ``d(y)`` on a sphere of the
      domain radius: [`PAPER_ROUTE`][].
    * ``"analytic"`` -- the Xapsos-Kellerer chord/straggling ``d(y)``,
      restricted, with the delta-ray tail folded in, on the same sphere.

    ``cell_line`` is ``"HSG"``, ``"SCC"``, ``"WI-38"`` or ``"DU145"`` (any case,
    hyphen optional). Every route carries the preset's photon reference, so HSG
    and SCC are RBE-ready as built. The sphere's diameter is twice the preset's
    table-1 domain radius: 0.84 um for HSG/SCC, 1.0 um for WI-38/DU145 (the
    script this came from used 0.84 um for every cell line).

    ``stopping_power`` is required and drives the two ``d(y)`` engines only: the
    caller chooses it. The impact-parameter routes keep the model's own default
    stopping power, so passing anything other than
    [`default_stopping_power`][pyradbio.track.stopping_power.default_stopping_power] mixes two
    stopping powers across the four routes.
    """
    from pyradbio.micro.lineal_energy import AnalyticSpectrumEngine, SphericalSite
    from pyradbio.micro.sato import SatoLinealEnergyEngine
    from pyradbio.models.mkm.imk import IMKModel

    try:
        preset = _PRESETS[cell_line.lower().replace("-", "")]
    except KeyError:
        raise KeyError(f"no Matsuya 2025 table-1 cell line {cell_line!r}") from None
    build = getattr(IMKModel, preset)
    sp = stopping_power
    default = build()
    site = SphericalSite(diameter_um=2.0 * default.domain_radius_um)
    return {
        "impact/cyl": default,
        "impact/sph": build(site_geometry="sphere"),
        "sato-AMF": build(engine=SatoLinealEnergyEngine(site=site, stopping_power=sp)),
        "analytic": build(
            engine=AnalyticSpectrumEngine(sp, site, restricted=True, include_delta_tail=True)
        ),
    }


# --------------------------------------------------------------------------
# Fig. 7 delivery and its relative MN frequency
# --------------------------------------------------------------------------
FIG7_DOSE_GY: float = 4.0
"""Fig. 7's total dose [Gy] and the rate its y axis is normalised at [Gy/h]
(sec. 3.1.3).
"""
FIG7_NORM_RATE_GY_PER_H: float = 60.0


def fig7_schedule(rate_Gy_per_h: float) -> DoseSchedule:
    """Fig. 7's delivery: 4 Gy **continuous** at ``rate_Gy_per_h``.

    Continuous is what reproduces the paper's line (see the module docstring);
    [`fig7_protocol`][] is the sec. 3.1.4 laboratory protocol. Raises
    `ValueError` unless ``rate_Gy_per_h > 0``.
    """
    if not rate_Gy_per_h > 0.0:
        raise ValueError(f"fig. 7 dose rate must be > 0 Gy/h, got {rate_Gy_per_h!r}")
    return DoseSchedule.continuous(FIG7_DOSE_GY, duration_h=FIG7_DOSE_GY / rate_Gy_per_h)


def fig7_protocol(rate_Gy_per_min: float) -> DoseSchedule:
    """Sec. 3.1.4: 4 Gy total, delivered at 1.0 Gy/min in fractions.

    At ``rate_Gy_per_min >= 1`` it is one continuous block at 1 Gy/min. Below,
    the protracted arms are **fractionated**, not continuous: ``rate_Gy_per_min``
    Gy per fraction delivered at 1 Gy/min, one fraction per minute, which is what
    makes their *average* rate 0.5, 0.1 and 0.05 Gy/min (the paper's three arms).

    A sub-1 rate must split the 4 Gy into a whole number of fractions (to 1e-9
    relative), so the schedule really delivers 4 Gy: ``0.3`` (13.3 fractions) is
    rejected rather than rounded to 3.9 Gy. Raises `ValueError` for that
    and for ``rate_Gy_per_min <= 0``.
    """
    if not rate_Gy_per_min > 0.0:
        raise ValueError(f"fig. 7 protocol rate must be > 0 Gy/min, got {rate_Gy_per_min!r}")
    if rate_Gy_per_min >= 1.0:
        return DoseSchedule.continuous(FIG7_DOSE_GY, duration_h=FIG7_DOSE_GY / 60.0)
    n_exact = FIG7_DOSE_GY / rate_Gy_per_min
    n_fractions = round(n_exact)
    if abs(n_exact - n_fractions) > 1e-9 * n_exact:
        raise ValueError(
            f"{rate_Gy_per_min!r} Gy/fraction does not split {FIG7_DOSE_GY} Gy into whole "
            f"fractions ({n_exact:.4g}); sec. 3.1.4 uses 0.5, 0.1 and 0.05 Gy/min"
        )
    return DoseSchedule.fractionated(
        rate_Gy_per_min,
        n_fractions=n_fractions,
        interval_h=1.0 / 60.0,
        fraction_duration_h=rate_Gy_per_min / 60.0,
    )


def relative_mn(model: IMKModel, rate_Gy_per_h: float, z_photon_Gy: float = 1.0) -> float:
    """Fig. 7's y: the MN frequency at this rate over the one at 60 Gy/h.

    Eq. (1) + eq. (3) under [`fig7_schedule`][], normalised as sec. 3.1.3
    specifies. ``h`` cancels in the ratio, exactly as in eq. (6), and the photon
    ``z*_1D`` barely enters -- so this is essentially eq. (1)'s time factor and the
    printed ``alpha0``/``beta0``, with nothing fitted.
    """

    def lq(rate: float) -> LQParameters:
        return model.alpha_beta_from_z_star_1D(z_photon_Gy, schedule=fig7_schedule(rate))

    return model.micronucleus_frequency(
        FIG7_DOSE_GY, lq(rate_Gy_per_h)
    ) / model.micronucleus_frequency(FIG7_DOSE_GY, lq(FIG7_NORM_RATE_GY_PER_H))
