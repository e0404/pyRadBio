"""Digitized benchmark curves from Scholz & Kraft 1996.

Source paper (the original amorphous track-structure / Local Effect Model):

Scholz, M. & Kraft, G. 1996, "Track structure and the calculation of
biological effects of heavy charged particles", Adv. Space Res. 18(1/2), 5-14,
doi:10.1016/0273-1177(95)00784-C.

The bundled numeric curves for Figs. 2-4 are **hand-digitized** from the paper's
figures, and are accurate enough for quantitative comparison (not just visual
regression). Fig. 2 carries the 7 experimental ion data sets plus the paper's
model ``sigma(LET)`` drawn as six LET-band segments (``curve1``..``curve6``).

Two findings worth knowing when using this data:

* **Fig. 2 uranium exceeds the nuclear size on purpose.** ``sigma`` runs to
  ~100-118 um^2 against a 50 um^2 effective nucleus, because ``sigma`` integrates
  over impact parameters that miss the nucleus (a large delta-ray penumbra still
  inactivates it). This is physics, not a digitization error.
* **The paper's ``sigma_inact`` is a final-slope quantity.** Scholz & Kraft state
  that it is determined from the final slope of the survival curve; their 1994
  Fig. 4 caption fixes the evaluation at 1% survival. It must not be converted to
  the initial slope with the single-track relation ``alpha = sigma/(A_eff*D_1)``
  except where the ion survival curve is exponential. Both Figs. 3 and 4 are
  full-curve observables: build
  ``LEM1.scholz1997_traversal_conditioned_survival`` once per point and read the
  slope and the iso-survival doses straight off it. The sampled-array helpers
  [`estimate_final_slope_cross_section`][] and
  [`estimate_iso_survival_rbe`][] remain for curves that only exist as
  ``(dose, survival)`` pairs.

Each curve lives in its own CSV under ``data/`` named
``scholz_kraft1996_figure<N>_<description>.csv`` with a four-line header
(metadata keys, metadata values, x/y quantities, x/y units) then the ``x,y``
data rows; the ``series``/``particle`` metadata records which figure line or
symbol set each point came from.

Each figure carries the paper's own cell/target inputs (see
[`FIGURE_INPUTS`][]) so the reconstruction script can configure LEM I with
exactly the parameters used in that panel.
"""

from __future__ import annotations

import csv
import math
import os
from collections.abc import Callable, Mapping, Sequence
from dataclasses import dataclass
from functools import lru_cache
from importlib import resources
from types import MappingProxyType
from typing import TYPE_CHECKING, Any

import numpy as np
import numpy.typing as npt

from pyradbio.benchmark.checkpoint import CsvCheckpoint
from pyradbio.core.cell import CellTarget
from pyradbio.core.constants import LEM_D1_PREFACTOR, WATER_DENSITY_G_CM3
from pyradbio.core.lq import dose_for_survival
from pyradbio.core.photon import PhotonParameters
from pyradbio.core.radiation import Ion

if TYPE_CHECKING:
    from pyradbio.models.lem.base import TraversalConditionedSurvival
    from pyradbio.models.lem.lem1 import LEM1
    from pyradbio.track.stopping_power import StoppingPower

SCHOLZ_KRAFT1996_DOI = "10.1016/0273-1177(95)00784-C"
# CSV compatibility marker for the final-slope observable. Version 3 reads the
# slope analytically off a traversal-conditioned Poisson mixture instead of
# differentiating per-dose Monte-Carlo samples, and drops the v2 branch that
# picked between two estimators depending on whether the samples happened to
# come out monotone.
FINAL_SLOPE_ESTIMATOR_VERSION = 3

_DATA_DIR = "data"
_CURVE_PREFIX = "scholz_kraft1996_figure"


@dataclass(frozen=True, slots=True)
class PaperCellInputs:
    """Cell/target inputs a paper panel used (LEM I photon + geometry)."""

    alpha_x: float  # Gy^-1 (paper's "a")
    beta_x: float  # Gy^-2 (paper's "b")
    D_t: float  # transition dose [Gy]
    A_nucl_um2: float  # effective geometric nuclear cross-section [um^2]

    def photon(self) -> PhotonParameters:
        """Return the LEM I photon reference ``(alpha_x, beta_x, D_t)``, no cluster.

        Scholz & Kraft 1996 (doi:10.1016/0273-1177(95)00784-C), figure captions.
        Equal to
        ``PhotonParameters(alpha_x=inputs.alpha_x, beta_x=inputs.beta_x, D_t=inputs.D_t)``.
        """
        return PhotonParameters(alpha_x=self.alpha_x, beta_x=self.beta_x, D_t=self.D_t)

    def cell_target(self) -> CellTarget:
        """Return a disk nucleus of area ``A_nucl_um2``, with ``effective_size_factor=1.0``.

        The radius is ``math.sqrt(A_nucl_um2 / math.pi)`` (the disk of the paper's
        effective area), in exactly that expression order, so it is bit-identical to
        the existing hand constructions. The factor 1.0 is passed
        explicitly because Scholz & Kraft's 50 um^2 already *is* the effective
        cross-section (against a measured mean of 75 um^2; see
        [`pyradbio.core.cell.CellTarget.effective_nucleus_radius_um`][]), so it must
        not be shrunk a second time.
        """
        return CellTarget(
            nucleus_radius_um=math.sqrt(self.A_nucl_um2 / math.pi),
            effective_size_factor=1.0,
        )


@dataclass(frozen=True, slots=True)
class FinalSlopeEstimate:
    """Final survival-curve slope and inactivation cross-section at one survival.

    Scholz & Kraft 1996 determine ``sigma_inact`` from the final survival-curve
    slope; the preceding Scholz & Kraft 1994 Fig. 4 caption specifies the 1%
    survival level. ``slope_Gy_inv`` is ``d[-ln S(D)]/dD`` at that level and
    ``sigma_um2`` converts the dose slope to a fluence slope using
    ``D = 0.16 LET Phi / rho`` for ``Phi`` in um^-2.
    """

    survival_level: float
    dose_Gy: float
    slope_Gy_inv: float
    sigma_um2: float


@dataclass(frozen=True, slots=True)
class IsoSurvivalEstimate:
    """Ion dose and RBE evaluated at a specified surviving fraction."""

    survival_level: float
    dose_Gy: float
    rbe: float


# Per-figure inputs taken from the Scholz & Kraft 1996 figure captions/text.
FIGURE_INPUTS: dict[str, PaperCellInputs] = {
    # Fig. 1A V79 protons: two labs. Folkard set (caption).
    "fig1a_survival_protons_v79": PaperCellInputs(0.109, 0.0283, 17.0, 50.0),
    # Fig. 1B T1 alpha particles (caption).
    "fig1b_survival_alpha_t1": PaperCellInputs(0.255, 0.0263, 20.0, 90.0),
    # Figs. 2-4 share the V79 sigma-LET inputs (Fig. 2 caption).
    "fig2_sigma_vs_let_v79": PaperCellInputs(0.136, 0.040, 12.0, 50.0),
    "fig3_sigma_vs_energy_v79": PaperCellInputs(0.136, 0.040, 12.0, 50.0),
    "fig4_rbe_vs_let": PaperCellInputs(0.136, 0.040, 12.0, 50.0),
}

# Fig. 1A Belli parameter set does not fit the one-per-figure map above; expose
# it explicitly for the reconstruction script.
FIG1A_BELLI_INPUTS = PaperCellInputs(0.136, 0.0421, 12.0, 50.0)


def _fit_log_kill_at_survival(
    doses_Gy: Sequence[float] | npt.NDArray[np.float64],
    survivals: Sequence[float] | npt.NDArray[np.float64],
    *,
    survival_level: float,
    fit_points: int = 5,
) -> tuple[float, float]:
    """Return local ``(dose, d[-ln S]/dD)`` at a bracketed survival level.

    The sampled ``-ln S`` is first projected onto the physical non-decreasing
    dose response, then a single local quadratic is fitted. Projection is a
    no-op on a smooth curve, so exact input still yields the exact derivative;
    on a noisy curve it removes the reversals that would otherwise be
    differentiated. Deliberately *one* estimator: selecting between a quadratic
    and a straight tail based on whether the samples happen to be monotone makes
    the choice a function of Monte-Carlo noise, so two streams of the same point
    could report different quantities and the difference between them would be
    read as precision.
    """
    doses = np.asarray(doses_Gy, dtype=np.float64)
    survival = np.asarray(survivals, dtype=np.float64)
    if doses.ndim != 1 or survival.ndim != 1 or doses.size != survival.size:
        raise ValueError("doses_Gy and survivals must be one-dimensional and equally sized")
    if doses.size < 3:
        raise ValueError("at least three survival-curve points are required")
    if not np.all(np.isfinite(doses)) or np.any(doses < 0.0) or np.any(np.diff(doses) <= 0.0):
        raise ValueError("doses_Gy must be finite, non-negative, and strictly increasing")
    if not np.all(np.isfinite(survival)) or np.any(survival <= 0.0) or np.any(survival > 1.0):
        raise ValueError("survivals must be finite and in (0, 1]")
    if not 0.0 < survival_level < 1.0:
        raise ValueError("survival_level must be in (0, 1)")
    if fit_points < 3:
        raise ValueError("fit_points must be >= 3")

    log_kill = _isotonic_log_kill(-np.log(survival))
    target_log_kill = -math.log(survival_level)
    signed = log_kill - target_log_kill
    # Only accept the physical direction: increasing dose crosses from lower to
    # higher log-kill. A noisy MC curve can also cross in the reverse direction.
    crossings = np.flatnonzero((signed[:-1] <= 0.0) & (signed[1:] >= 0.0))
    if crossings.size == 0:
        raise ValueError(f"survival curve must bracket survival_level={survival_level:g}")

    # Use the crossing whose two samples lie closest to the target. This remains
    # well-defined if small Monte-Carlo fluctuations create multiple crossings.
    crossing = int(min(crossings, key=lambda i: abs(float(signed[i])) + abs(float(signed[i + 1]))))
    count = min(fit_points, doses.size)
    start = max(0, min(crossing - (count - 2) // 2, doses.size - count))
    stop = start + count
    x = doses[start:stop]
    y = log_kill[start:stop]

    origin = float(0.5 * (doses[crossing] + doses[crossing + 1]))
    coefficients = np.polynomial.polynomial.polyfit(x - origin, y, deg=2)
    roots = np.polynomial.Polynomial(
        (coefficients[0] - target_log_kill, coefficients[1], coefficients[2])
    ).roots()
    # Accept a root anywhere in the fit window, not only inside the two samples
    # that straddle the target: after the monotone projection the straddling
    # pair can be a pooled plateau, and the fitted curve then crosses just
    # outside it. Extrapolation beyond the window stays excluded.
    bracket_lo = float(x[0] - origin)
    bracket_hi = float(x[-1] - origin)
    root_tolerance = 1.0e-10 * max(1.0, abs(bracket_lo), abs(bracket_hi))
    real_roots = [
        float(root.real)
        for root in roots
        if abs(float(root.imag)) <= 1.0e-10
        and bracket_lo - root_tolerance <= float(root.real) <= bracket_hi + root_tolerance
    ]
    if real_roots:
        offset = min(max(min(real_roots, key=abs), bracket_lo), bracket_hi)
        dose_at_level = origin + offset
        slope = float(coefficients[1] + 2.0 * coefficients[2] * offset)
    else:
        slope = math.nan

    # A quadratic through noisy neighboring MC points can turn locally downward
    # or miss the actual bracket even though the directed two-point crossing is
    # valid. In that case use its positive secant rather than extrapolating or
    # rejecting the whole batch point. Exact LQ data retain the quadratic path.
    if not math.isfinite(slope) or slope <= 0.0:
        delta_log_kill = float(log_kill[crossing + 1] - log_kill[crossing])
        delta_dose = float(doses[crossing + 1] - doses[crossing])
        if delta_log_kill <= 0.0:
            raise ValueError("directed survival crossing must have a positive local slope")
        fraction = float((target_log_kill - log_kill[crossing]) / delta_log_kill)
        dose_at_level = float(doses[crossing] + fraction * delta_dose)
        slope = delta_log_kill / delta_dose
    return dose_at_level, slope


def _isotonic_log_kill(log_kill: npt.NDArray[np.float64]) -> npt.NDArray[np.float64]:
    """Project sampled log-kill onto the physical non-decreasing dose response."""
    levels: list[float] = []
    weights: list[int] = []
    for value in log_kill:
        levels.append(float(value))
        weights.append(1)
        while len(levels) >= 2 and levels[-2] > levels[-1]:
            combined_weight = weights[-2] + weights[-1]
            combined_level = (levels[-2] * weights[-2] + levels[-1] * weights[-1]) / combined_weight
            levels[-2:] = [combined_level]
            weights[-2:] = [combined_weight]

    projected = np.empty_like(log_kill)
    start = 0
    for level, weight in zip(levels, weights, strict=True):
        projected[start : start + weight] = level
        start += weight
    return projected


def photon_dose_for_survival(
    survival_level: float,
    *,
    alpha_Gy_inv: float,
    beta_Gy_inv2: float,
    D_t_Gy: float | None = None,
) -> float:
    """Iso-survival dose of the LEM photon reference curve.

    LQ below ``D_t``, straight exponential tail above it (Scholz et al. 1997
    eq. 1). ``D_t = None`` gives the plain LQ inversion.
    """
    return PhotonParameters(alpha_Gy_inv, beta_Gy_inv2, D_t_Gy).dose_for_survival(survival_level)


def estimate_final_slope_cross_section(
    doses_Gy: Sequence[float] | npt.NDArray[np.float64],
    survivals: Sequence[float] | npt.NDArray[np.float64],
    *,
    let_keV_um: float,
    survival_level: float = 0.01,
    density_g_cm3: float = WATER_DENSITY_G_CM3,
    fit_points: int = 9,
) -> FinalSlopeEstimate:
    """Estimate the 1996 final-slope cross-section from a *sampled* survival curve.

    Compatibility path for curves that only exist as ``(dose, survival)`` pairs.
    The dose and derivative come from a monotone-projected local quadratic (see
    `_fit_log_kill_at_survival`). The input must bracket
    ``survival_level``; extrapolation would hide an insufficient dose range.

    Prefer ``LEM1.scholz1997_traversal_conditioned_survival``. Its curve is an
    analytic Poisson mixture, so it needs no differentiation of samples at all --
    which is the dominant error here, because at 1 % survival the sampled slope
    is being read off exactly where a per-dose Monte Carlo is noisiest.

    Reference: Scholz & Kraft 1996,
    doi:10.1016/0273-1177(95)00784-C, discussion of Figs. 2-3; evaluation at 1%
    is stated explicitly in Scholz & Kraft 1994 Fig. 4.
    """
    if not math.isfinite(let_keV_um) or let_keV_um <= 0.0:
        raise ValueError("let_keV_um must be finite and > 0")
    if not math.isfinite(density_g_cm3) or density_g_cm3 <= 0.0:
        raise ValueError("density_g_cm3 must be finite and > 0")

    dose_at_level, slope = _fit_log_kill_at_survival(
        doses_Gy,
        survivals,
        survival_level=survival_level,
        fit_points=fit_points,
    )

    sigma = LEM_D1_PREFACTOR * let_keV_um * slope / density_g_cm3
    return FinalSlopeEstimate(
        survival_level=survival_level,
        dose_Gy=dose_at_level,
        slope_Gy_inv=slope,
        sigma_um2=sigma,
    )


def estimate_iso_survival_rbe(
    doses_Gy: Sequence[float] | npt.NDArray[np.float64],
    survivals: Sequence[float] | npt.NDArray[np.float64],
    *,
    survival_level: float,
    reference_alpha_Gy_inv: float,
    reference_beta_Gy_inv2: float,
    reference_D_t_Gy: float | None = None,
    fit_points: int = 5,
) -> IsoSurvivalEstimate:
    """Estimate Fig. 4 RBE from a full ion survival curve at fixed survival.

    The ion dose is obtained from a monotone-projected local quadratic fit to
    ``-ln S`` around a bracketed crossing; ``RBE = D_reference / D_ion``. No ion
    LQ surrogate is introduced.

    Pass ``reference_D_t_Gy`` whenever the photon reference is LEM's shouldered
    curve (LQ below ``D_t``, exponential above). Beyond ``D_t`` the plain LQ root
    is smaller than the real iso-survival dose, which would silently inflate the
    RBE. For the 1996 V79 inputs the 1 % photon dose is 9.2 Gy against
    ``D_t = 12`` Gy, so the two agree there -- but that is a property of those
    numbers, not of the method.

    Reference: Scholz & Kraft 1996 Fig. 4,
    doi:10.1016/0273-1177(95)00784-C.
    """
    ion_dose, _slope = _fit_log_kill_at_survival(
        doses_Gy,
        survivals,
        survival_level=survival_level,
        fit_points=fit_points,
    )
    reference_dose = photon_dose_for_survival(
        survival_level,
        alpha_Gy_inv=reference_alpha_Gy_inv,
        beta_Gy_inv2=reference_beta_Gy_inv2,
        D_t_Gy=reference_D_t_Gy,
    )
    return IsoSurvivalEstimate(
        survival_level=survival_level,
        dose_Gy=ion_dose,
        rbe=reference_dose / ion_dose,
    )


def iso_survival_rbe_from_final_slope(
    sigma_um2: Sequence[float] | npt.NDArray[np.float64],
    let_keV_um: Sequence[float] | npt.NDArray[np.float64],
    *,
    reference_dose_Gy: float,
    survival_level: float = 0.01,
    density_g_cm3: float = WATER_DENSITY_G_CM3,
) -> npt.NDArray[np.float64]:
    r"""RBE at ``survival_level`` implied by a final-slope cross-section.

    Lets Fig. 2 and Fig. 3, which plot ``sigma``, be compared against Fig. 4,
    which plots RBE. Inverting [`estimate_final_slope_cross_section`][] gives
    the log-kill slope exactly,

    ``slope = sigma * density / (LEM_D1_PREFACTOR * LET)``

    and then ``RBE = D_reference / D_ion`` with
    ``D_ion = -ln(S) / slope``.

    **That last step assumes the ion curve is a pure exponential through the
    origin**, i.e. that it has no shoulder left at ``survival_level``. ``sigma``
    is a *slope*; turning a slope into a *dose* needs the intercept, which the
    figures do not carry. Writing the real curve as
    ``-ln S = slope * D - c``, the construction returns ``D_ion`` short by
    ``c / slope``, so it **over-estimates RBE by a factor ``1 + c / -ln(S)``**.

    That factor is not a worry at high LET and not negligible at low LET.
    Measured against the converged 1996 checkpoint at ``S = 0.01``, where both
    the true iso-survival dose and the slope are known per point:

    ==============  ==========  ============
    ion             LET keV/um  over-estimate
    ==============  ==========  ============
    carbon 1-5      282-778     -0.2 to +0.9%
    carbon 8-12     141-196     +2.4 to +6.4%
    carbon 20       93          +12.5%
    helium 1-3      48-121      +0.2 to +1.9%
    helium 8        22          +11.3%
    helium 20       10          +26.4%
    ==============  ==========  ============

    So read a derived RBE as an **upper bound that tightens with LET**, and do
    not attribute a low-LET gap against Fig. 4 to the paper being inconsistent
    -- most of it is this construction. The physics is the same statement in
    reverse: at low LET the ion survival curve keeps a shoulder down to 1%
    survival, which is exactly when a final slope stops determining a dose.

    Reference: Scholz & Kraft 1996, doi:10.1016/0273-1177(95)00784-C, Figs. 2-4.
    """
    if not 0.0 < survival_level < 1.0:
        raise ValueError(f"survival_level must be in (0, 1), got {survival_level!r}")
    if not math.isfinite(reference_dose_Gy) or reference_dose_Gy <= 0.0:
        raise ValueError("reference_dose_Gy must be finite and > 0")
    sigma = np.asarray(sigma_um2, dtype=np.float64)
    let = np.asarray(let_keV_um, dtype=np.float64)
    if sigma.shape != let.shape:
        raise ValueError(f"sigma and LET must align, got {sigma.shape} and {let.shape}")
    if np.any(let <= 0.0) or np.any(sigma <= 0.0):
        raise ValueError("sigma and LET must be > 0")

    slope = sigma * density_g_cm3 / (LEM_D1_PREFACTOR * let)
    ion_dose = -math.log(survival_level) / slope
    return np.asarray(reference_dose_Gy / ion_dose, dtype=np.float64)


@dataclass(frozen=True, slots=True)
class ScholzKraft1996Curve:
    """One digitized reference curve (a single figure line or symbol set)."""

    figure: str
    particle: str
    series: str  # "calc" (paper model line) or "data" (experimental symbols)
    survival_level: float | None  # Fig. 4 iso-survival level (e.g. 0.80), else None
    energy_MeV_u: float | None  # per-energy series, else None
    x_quantity: str
    x_unit: str
    y_quantity: str
    y_unit: str
    x: tuple[float, ...]
    y: tuple[float, ...]
    source_page: int
    note: str
    doi: str = SCHOLZ_KRAFT1996_DOI


@dataclass(frozen=True, slots=True)
class CurveResidualPoint:
    """One model point compared with an interpolated paper calculation line."""

    x: float
    model_y: float
    paper_y: float
    model_error: float | None
    relative_residual: float
    model_relative_uncertainty: float | None


@dataclass(frozen=True, slots=True)
class CurveComparison:
    """Residual summary over the model points supported by a paper curve.

    ``points_outside_support`` is reported explicitly because extrapolating a
    digitized line would turn stopping-power/range differences into artificial
    residuals. Model uncertainties are divided by the interpolated paper value;
    they remain a Monte-Carlo precision diagnostic, not a formal confidence
    interval for the model-paper difference.
    """

    points: tuple[CurveResidualPoint, ...]
    points_outside_support: int
    mean_ratio: float
    rms_relative_error: float
    max_abs_relative_error: float
    max_model_relative_uncertainty: float | None


def compare_points_to_curve(
    curve: ScholzKraft1996Curve,
    *,
    x: Sequence[float] | npt.NDArray[np.float64],
    model_y: Sequence[float] | npt.NDArray[np.float64],
    model_error: Sequence[float] | npt.NDArray[np.float64] | None = None,
    log_x: bool = False,
) -> CurveComparison:
    """Compare model points with a digitized paper line by interpolation.

    Points outside the digitized curve's x support are counted and excluded;
    the function never extrapolates. Set ``log_x`` when the source figure uses a
    logarithmic x axis (Scholz & Kraft 1996 Fig. 4), so interpolation follows
    the geometry of the plotted paper line.
    """
    x_array = np.asarray(x, dtype=float)
    model_array = np.asarray(model_y, dtype=float)
    if x_array.ndim != 1 or model_array.ndim != 1 or x_array.size != model_array.size:
        raise ValueError("x and model_y must be one-dimensional arrays of the same length")
    if model_error is None:
        error_array: npt.NDArray[np.float64] | None = None
    else:
        error_array = np.asarray(model_error, dtype=float)
        if error_array.ndim != 1 or error_array.size != x_array.size:
            raise ValueError("model_error must have the same length as x and model_y")
        if np.any(error_array < 0.0) or not np.all(np.isfinite(error_array)):
            raise ValueError("model_error must contain finite non-negative values")
    if not np.all(np.isfinite(x_array)) or not np.all(np.isfinite(model_array)):
        raise ValueError("x and model_y must contain finite values")

    paper_x = np.asarray(curve.x, dtype=float)
    paper_y = np.asarray(curve.y, dtype=float)
    if log_x:
        if np.any(paper_x <= 0.0) or np.any(x_array <= 0.0):
            raise ValueError("log-x comparison requires positive x values")
        interpolation_x = np.log(paper_x)
        query_x = np.log(x_array)
    else:
        interpolation_x = paper_x
        query_x = x_array

    supported = (x_array >= paper_x[0]) & (x_array <= paper_x[-1])
    reference = np.interp(query_x[supported], interpolation_x, paper_y)
    if np.any(reference == 0.0):
        raise ValueError("paper curve contains a zero-valued interpolation point")

    supported_indices = np.flatnonzero(supported)
    points: list[CurveResidualPoint] = []
    for output_index, input_index in enumerate(supported_indices):
        paper_value = float(reference[output_index])
        model_value = float(model_array[input_index])
        error = None if error_array is None else float(error_array[input_index])
        points.append(
            CurveResidualPoint(
                x=float(x_array[input_index]),
                model_y=model_value,
                paper_y=paper_value,
                model_error=error,
                relative_residual=model_value / paper_value - 1.0,
                model_relative_uncertainty=None if error is None else error / abs(paper_value),
            )
        )

    if not points:
        raise ValueError("no model points lie inside the digitized paper curve support")
    residuals = np.asarray([point.relative_residual for point in points])
    uncertainties = [
        point.model_relative_uncertainty
        for point in points
        if point.model_relative_uncertainty is not None
    ]
    return CurveComparison(
        points=tuple(points),
        points_outside_support=int(x_array.size - len(points)),
        mean_ratio=float(np.mean(residuals + 1.0)),
        rms_relative_error=float(np.sqrt(np.mean(residuals**2))),
        max_abs_relative_error=float(np.max(np.abs(residuals))),
        max_model_relative_uncertainty=max(uncertainties, default=None),
    )


def _optional_float(value: str) -> float | None:
    stripped = value.strip()
    return None if stripped == "" else float(stripped)


def _data_files(prefix: str) -> list[tuple[str, str]]:
    """Return ``(filename, text)`` for every ``data/`` CSV starting with ``prefix``."""
    data_dir = resources.files(__package__).joinpath(_DATA_DIR)
    out: list[tuple[str, str]] = []
    for entry in data_dir.iterdir():
        if entry.name.startswith(prefix) and entry.name.endswith(".csv"):
            out.append((entry.name, entry.read_text(encoding="utf-8")))
    return out


@lru_cache(maxsize=1)
def load_curves() -> tuple[ScholzKraft1996Curve, ...]:
    """Return every digitized Scholz & Kraft 1996 benchmark curve (one per file)."""
    curves: list[ScholzKraft1996Curve] = []
    for _name, text in _data_files(_CURVE_PREFIX):
        rows = list(csv.reader(text.splitlines()))
        meta = dict(zip(rows[0], rows[1], strict=True))
        x_quantity, y_quantity = rows[2][0], rows[2][1]
        x_unit, y_unit = rows[3][0], rows[3][1]
        points = sorted((float(r[0]), float(r[1])) for r in rows[4:] if r and r[0].strip())
        curves.append(
            ScholzKraft1996Curve(
                figure=meta["figure"],
                particle=meta["particle"],
                series=meta["series"],
                survival_level=_optional_float(meta["survival_level"]),
                energy_MeV_u=_optional_float(meta["energy_MeV_u"]),
                x_quantity=x_quantity,
                x_unit=x_unit,
                y_quantity=y_quantity,
                y_unit=y_unit,
                x=tuple(x for x, _ in points),
                y=tuple(y for _, y in points),
                source_page=int(meta["source_page"]),
                note=meta["note"],
                doi=meta.get("doi", SCHOLZ_KRAFT1996_DOI),
            )
        )
    return tuple(
        sorted(
            curves,
            key=lambda c: (
                c.figure,
                c.particle,
                c.series,
                c.survival_level if c.survival_level is not None else -1.0,
                c.energy_MeV_u if c.energy_MeV_u is not None else -1.0,
            ),
        )
    )


def iter_curves(
    *,
    figure: str | None = None,
    particle: str | None = None,
    series: str | None = None,
    survival_level: float | None = None,
    energy_MeV_u: float | None = None,
) -> tuple[ScholzKraft1996Curve, ...]:
    """Filter digitized Scholz & Kraft 1996 curves.

    ``series="calc"`` selects the paper's own model/calculation lines;
    ``series="data"`` selects the experimental symbols redrawn in the figures.
    """
    curves = load_curves()
    return tuple(
        c
        for c in curves
        if (figure is None or c.figure == figure)
        and (particle is None or c.particle == particle)
        and (series is None or c.series == series)
        and (survival_level is None or c.survival_level == survival_level)
        and (energy_MeV_u is None or c.energy_MeV_u == energy_MeV_u)
    )


def get_curve(
    figure: str,
    *,
    particle: str | None = None,
    series: str | None = None,
    survival_level: float | None = None,
    energy_MeV_u: float | None = None,
) -> ScholzKraft1996Curve:
    """Return exactly one Scholz & Kraft 1996 reference curve.

    Raises ``LookupError`` if the filters select zero or multiple curves.
    """
    matches = iter_curves(
        figure=figure,
        particle=particle,
        series=series,
        survival_level=survival_level,
        energy_MeV_u=energy_MeV_u,
    )
    if len(matches) != 1:
        raise LookupError(f"expected exactly one Scholz & Kraft 1996 curve, got {len(matches)}")
    return matches[0]


# --- the offline full-LEM observables of Figs. 2-4 ----------------------------------
#
# The generator is scripts/lem1_mc_reference.py ``sk1996-final-slope``; the
# checkpoint is validation/data/sk1996_mc_observables.csv.

OBSERVABLE_ESTIMATOR_VERSION = FINAL_SLOPE_ESTIMATOR_VERSION
"""CSV compatibility marker of ``sk1996_mc_observables.csv`` rows; equal to
[`FINAL_SLOPE_ESTIMATOR_VERSION`][] (the slope is read analytically off the
traversal-conditioned Poisson mixture).
"""

SPECIES: Mapping[str, tuple[int, int]] = MappingProxyType(
    {
        "proton": (1, 1),
        "helium": (2, 4),
        "carbon": (6, 12),
        "neon": (10, 20),
        "argon": (18, 40),
        "krypton": (36, 84),
        "xenon": (54, 132),
        "uranium": (92, 238),
    }
)
"""The ions of Figs. 2-4, ``name -> (Z, A)``."""
SIGMA_PARTICLES: tuple[str, ...] = ("carbon", "argon", "uranium")
"""Fig. 3's C/Ar/U branches (sigma) and Fig. 4's p/He/C branches (RBE)."""
RBE_PARTICLES: tuple[str, ...] = ("proton", "helium", "carbon")
ENERGIES_MEV_U: tuple[float, ...] = (1.0, 2.0, 3.0, 5.0, 8.0, 12.0, 16.0, 20.0)
"""1-20 MeV/u, the interval of the 1996 Fig. 4 caption (the 1994 sibling figure
stops p/He at 10 MeV/u; 1-20 is deliberate, since this reconstructs the 1996 panel).
"""
MC_POINTS: tuple[tuple[str, float], ...] = tuple(
    (particle, energy)
    for particle in sorted(set(SIGMA_PARTICLES) | set(RBE_PARTICLES))
    for energy in ENERGIES_MEV_U
)
"""The default point catalogue of the offline batch."""
SURVIVAL_LEVELS: tuple[float, ...] = (0.80, 0.20, 0.01)
LEVEL_TAG: Mapping[float, str] = MappingProxyType({0.80: "80pct", 0.20: "20pct", 0.01: "1pct"})
DOSE_HEADROOM = 1.6
"""The traversal grid must span the 1% dose: it is sized from the cheap two-track LQ
estimate with this headroom, then doubled on demand (at most
``MAX_DOSE_EXPANSIONS`` times), so an under-sized grid fails loudly rather than
extrapolating.
"""
MAX_DOSE_EXPANSIONS = 8
SURVIVAL_BRACKET_MARGIN = 0.5
"""The pilot must reach comfortably PAST the deepest level: the bracket is checked on
the pilot but the observables are read off the pooled curve, and pooling moves S by
a fraction of a percent (e.g. an argon 16 MeV/u pilot that stopped at 0.010067).
"""
DEFAULT_MAX_MEAN_TRAVERSALS = 40_000.0
"""The p, 20 MeV/u 1%-dose sample has about 13,200 tracks inside R_eff + r_max; the
guard applies to lambda at the top of the grid (headroom plus Poisson padding).
"""
DEFAULT_BATCHES = 16
"""Adaptive allocation: batches are pooled, so a pilot is never thrown away."""
DEFAULT_REALIZATIONS_PER_BATCH = 25
DEFAULT_MAX_REALIZATIONS = 20_000
DEFAULT_TARGET_RELATIVE_ERROR = 0.03
MAX_ALLOCATION_GROWTH = 4.0
MAX_POISSON_TAIL_MASS = 1.0e-6
"""Poisson weight allowed beyond the traversal grid before a point is rejected."""

MC_OBSERVABLE_FIELDS: tuple[str, ...] = (
    "particle",
    "estimator_version",
    "stopping_power",
    "energy_MeV_u",
    "let_keV_um",
    "dose_1pct_Gy",
    "slope_1pct_Gy_inv",
    "sigma_1pct_um2",
    "sigma_1pct_err_um2",
    "sigma_low_dose_um2",
    "dose_80pct_Gy",
    "rbe_80pct",
    "rbe_80pct_err",
    "dose_20pct_Gy",
    "rbe_20pct",
    "rbe_20pct_err",
    "rbe_1pct",
    "rbe_1pct_err",
    "n_realizations",
    "max_relative_error",
    "target_relative_error",
    "precision_converged",
)
"""Columns of ``sk1996_mc_observables.csv``, in file order."""
MC_OBSERVABLE_TEXT_COLUMNS = frozenset({"particle", "stopping_power"})
"""Columns of ``sk1996_mc_observables.csv`` that are labels, not measurements."""


def v79_lem1_models(
    stopping_power: StoppingPower,
    inputs: PaperCellInputs | None = None,
) -> tuple[LEM1, LEM1]:
    """``(full, bracket)`` LEM I models of the 1996 V79 panels (Figs. 2-4).

    Both use ``inputs`` (default [`FIGURE_INPUTS`][] ``["fig3_sigma_vs_energy_v79"]``;
    Figs. 2-4 share one input set) through ``inputs.photon()`` and
    ``inputs.cell_target()`` and the same ``stopping_power`` object. ``full`` carries
    ``beta_model="scholz1997_montecarlo"`` and runs the traversal-conditioned Monte
    Carlo; ``bracket`` carries ``beta_model="scholz1997_two_track"``, the second-order
    fluence (cluster) expansion of the same stochastic model. The bracket only sizes
    the traversal grid, but sizing it badly is not free: ``slope_matching`` is far too
    steep at high LET and guessed 9.9 Gy for U at 1 MeV/u where the true 1% dose is
    200 Gy (five discarded sweeps), while the two-track expansion is within 5% of the
    finished curve over most of the grid and costs about a second. It is also the
    deterministic line the reconstruction draws in Figs. 2-4.

    For a ``ns1970|bethe`` choice string pass
    ``pyradbio.benchmark.stopping.select_stopping_power(choice)``.
    """
    # Imported here: the LEM stack (and numba) is seconds of import time that the
    # digitized curves and the checkpoint loaders of this module never need.
    from pyradbio.models.lem.lem1 import LEM1

    inputs = FIGURE_INPUTS["fig3_sigma_vs_energy_v79"] if inputs is None else inputs
    photon = inputs.photon()
    cell = inputs.cell_target()
    full = LEM1(photon, cell, stopping_power=stopping_power, beta_model="scholz1997_montecarlo")
    bracket = LEM1(photon, cell, stopping_power=stopping_power, beta_model="scholz1997_two_track")
    return full, bracket


def _build_curve(
    full: LEM1,
    bracket: LEM1,
    ion: Ion,
    *,
    n_batches: int,
    realizations_per_batch: int,
    seed: int,
    max_mean_traversals: float,
) -> TraversalConditionedSurvival:
    """Build a conditioned curve whose traversal grid spans the 1% crossing.

    The cheap ion LQ only sizes the traversal grid; nothing downstream uses it.
    An under-sized grid is detected and doubled, so a poor guess costs time, not
    accuracy.
    """
    lq = bracket.alpha_beta(ion)
    max_dose = DOSE_HEADROOM * dose_for_survival(min(SURVIVAL_LEVELS), lq.alpha, lq.beta)
    for _attempt in range(MAX_DOSE_EXPANSIONS + 1):
        curve = full.scholz1997_traversal_conditioned_survival(
            ion,
            max_dose_Gy=max_dose,
            n_batches=n_batches,
            realizations_per_batch=realizations_per_batch,
            seed=seed,
            max_mean_traversals=max_mean_traversals,
        )
        if curve.survival(max_dose) <= SURVIVAL_BRACKET_MARGIN * min(SURVIVAL_LEVELS):
            return curve
        max_dose *= 2.0
    raise ValueError(
        f"survival did not reach {SURVIVAL_BRACKET_MARGIN * min(SURVIVAL_LEVELS):g} "
        f"through D={max_dose / 2:g} Gy for {ion}"
    )


def _observables(
    curve: TraversalConditionedSurvival,
    *,
    let_keV_um: float,
    photon: PhotonParameters,
) -> dict[str, tuple[float, float]]:
    """Return ``{name: (value, jackknife standard error)}`` for one curve.

    Every observable is a functional of the same ``g`` vector, so the jackknife
    replicates cost nothing beyond re-evaluating closed-form expressions.
    """
    results: dict[str, tuple[float, float]] = {}

    def dose_at(level: float, g: npt.NDArray[np.float64]) -> float:
        return curve.dose_for_survival(level, g=g)

    results["dose_1pct_Gy"] = curve.jackknife(lambda g: dose_at(0.01, g))
    results["slope_1pct_Gy_inv"] = curve.jackknife(
        lambda g: curve.log_kill_slope_Gy_inv(dose_at(0.01, g), g=g)
    )
    results["sigma_1pct_um2"] = curve.jackknife(
        lambda g: 0.16 * let_keV_um * curve.log_kill_slope_Gy_inv(dose_at(0.01, g), g=g)
    )
    for level in SURVIVAL_LEVELS:
        tag = LEVEL_TAG[level]
        reference_dose = photon.dose_for_survival(level)

        def level_dose(g: npt.NDArray[np.float64], lv: float = level) -> float:
            return dose_at(lv, g)

        def level_rbe(
            g: npt.NDArray[np.float64], lv: float = level, ref: float = reference_dose
        ) -> float:
            return ref / dose_at(lv, g)

        results[f"dose_{tag}_Gy"] = curve.jackknife(level_dose)
        results[f"rbe_{tag}"] = curve.jackknife(level_rbe)
    return results


def _maximum_relative_error(values: dict[str, tuple[float, float]], particle: str) -> float:
    """Worst relative error among the observables this particle actually plots."""
    names: list[str] = []
    # Fig. 2 contains every configured ion except protons; Fig. 3 uses C/Ar/U.
    if particle != "proton":
        names.append("sigma_1pct_um2")
    # Fig. 4 contains only p/He/C. Do not spend histories converging unused RBE
    # columns for Ar/U or optional Ne/Kr/Xe sigma points.
    if particle in RBE_PARTICLES:
        names.extend(f"rbe_{tag}" for tag in LEVEL_TAG.values())
    if not names:
        raise ValueError(f"no benchmark observables configured for particle={particle!r}")
    errors = []
    for name in names:
        value, error = values[name]
        errors.append(abs(error) / abs(value) if value != 0.0 else math.inf)
    return max(errors)


def estimate_final_slope_point(
    full: LEM1,
    bracket: LEM1,
    particle: str,
    energy_MeV_u: float,
    *,
    n_batches: int,
    realizations_per_batch: int,
    max_realizations: int,
    target_relative_error: float,
    seed: int,
    max_mean_traversals: float,
) -> dict[str, Any]:
    """One ``sk1996_mc_observables.csv`` row: pool batches until every observable is precise.

    Builds ``LEM1.scholz1997_traversal_conditioned_survival`` once per point:
    ``g_k = E[exp(-N) | k tracks]`` does not depend on dose, so one sweep determines
    the whole curve as a Poisson mixture, monotone by construction and analytically
    differentiable. The final slope at 1% survival,
    ``sigma_1% = 0.16 * LET * d[-ln S]/dD |_{S=0.01}`` (Scholz & Kraft 1996; the 1%
    level from their 1994 Fig. 4 caption), and the iso-survival doses and
    ``RBE = D_photon / D_ion`` at 80/20/1% (Fig. 4) are read off exactly -- no
    fitted window, no ion LQ surrogate, and an exact zero-traversal term, which
    carries most of ``S`` at 1% while occurring in well under 1% of histories.

    Uncertainties are a delete-one jackknife over the independent batches. Extra
    batches are *pooled* into the curve (at most ``MAX_ALLOCATION_GROWTH`` times the
    histories per round, up to ``max_realizations``), never restarted, until the worst
    relative error of the observables this particle plots meets
    ``target_relative_error``; ``precision_converged`` records whether it did.

    Raises:
        ValueError: on a non-positive target, a grid that cannot reach the 1%
            bracket, pooling that moves the curve off its bracket, or a Poisson tail
            heavier than ``MAX_POISSON_TAIL_MASS`` beyond the traversal grid.
    """
    if not math.isfinite(target_relative_error) or target_relative_error <= 0.0:
        raise ValueError("target_relative_error must be finite and > 0")

    z, a = SPECIES[particle]
    ion = Ion(z, a, energy_MeV_u)
    let_keV_um = full.stopping_power.let_keV_um(ion)

    curve = _build_curve(
        full,
        bracket,
        ion,
        n_batches=n_batches,
        realizations_per_batch=realizations_per_batch,
        seed=seed,
        max_mean_traversals=max_mean_traversals,
    )
    round_index = 0
    while True:
        values = _observables(curve, let_keV_um=let_keV_um, photon=full.photon)
        relative_error = _maximum_relative_error(values, particle)
        if relative_error <= target_relative_error or curve.n_realizations >= max_realizations:
            break

        # Standard errors scale as 1/sqrt(N); cap one jump at 4x so a noisy pilot
        # cannot allocate the whole ceiling at once. Unlike a restart, the extra
        # batches ADD to the curve, so nothing already computed is discarded.
        required = (relative_error / target_relative_error) ** 2
        target_total = min(
            max_realizations,
            math.ceil(curve.n_realizations * min(required, MAX_ALLOCATION_GROWTH)),
        )
        extra = max(target_total - curve.n_realizations, realizations_per_batch)
        extra_batches = max(2, math.ceil(extra / realizations_per_batch))
        round_index += 1
        curve = curve.pooled_with(
            full.scholz1997_traversal_conditioned_survival(
                ion,
                max_dose_Gy=curve.max_supported_dose_Gy,
                n_batches=extra_batches,
                realizations_per_batch=realizations_per_batch,
                seed=seed + 7919 * round_index,
                max_mean_traversals=max_mean_traversals,
            )
        )
        if curve.survival(curve.max_supported_dose_Gy) > min(SURVIVAL_LEVELS):
            raise ValueError(
                f"pooling moved {particle} {energy_MeV_u} MeV/u off its bracket; "
                f"raise SURVIVAL_BRACKET_MARGIN (currently {SURVIVAL_BRACKET_MARGIN:g})"
            )

    tail = curve.poisson_tail_mass(values["dose_1pct_Gy"][0])
    if tail > MAX_POISSON_TAIL_MASS:
        raise ValueError(
            f"traversal grid too short for {particle} {energy_MeV_u} MeV/u: "
            f"Poisson tail mass {tail:.2e} beyond k_max={curve.k_max}"
        )

    result: dict[str, Any] = {
        "particle": particle,
        "estimator_version": OBSERVABLE_ESTIMATOR_VERSION,
        "stopping_power": full.stopping_power.name,  # type: ignore[attr-defined]
        "energy_MeV_u": energy_MeV_u,
        "let_keV_um": let_keV_um,
        "sigma_low_dose_um2": full.inactivation_cross_section_um2(ion),
        "n_realizations": curve.n_realizations,
        "max_relative_error": relative_error,
        "target_relative_error": target_relative_error,
        "precision_converged": int(relative_error <= target_relative_error),
    }
    for name in ("dose_1pct_Gy", "slope_1pct_Gy_inv"):
        result[name] = values[name][0]
    result["sigma_1pct_um2"], result["sigma_1pct_err_um2"] = values["sigma_1pct_um2"]
    for level in SURVIVAL_LEVELS:
        tag = LEVEL_TAG[level]
        result[f"dose_{tag}_Gy"] = values[f"dose_{tag}_Gy"][0]
        result[f"rbe_{tag}"], result[f"rbe_{tag}_err"] = values[f"rbe_{tag}"]
    return result


def checkpoint_reuse_reason(
    row: Mapping[str, Any],
    *,
    target_relative_error: float,
    max_realizations: int,
    stopping_power: str,
) -> str | None:
    """Return why a checkpoint row can be reused (``"target met"`` / ``"ceiling reached"``), if any.

    A row from an older estimator, or from a different stopping power, is never
    reused: the definitions differ, so mixing them inside one panel would compare two
    different quantities against a single paper line. LET enters both the x axis and
    the cross-section conversion, so a mixed checkpoint is not merely imprecise -- it
    is incoherent.
    """
    if int(float(row.get("estimator_version", -1))) != OBSERVABLE_ESTIMATOR_VERSION:
        return None
    if str(row.get("stopping_power", "")) != stopping_power:
        return None
    if "max_relative_error" not in row:
        return None
    if float(row["max_relative_error"]) <= target_relative_error:
        return "target met"
    if int(float(row["n_realizations"])) >= max_realizations:
        return "ceiling reached"
    return None


def load_mc_observables(path: str | os.PathLike[str]) -> dict[str, list[dict[str, Any]]]:
    """Read ``sk1996_mc_observables.csv`` for plotting, grouped by particle.

    Each row keeps every non-empty column except ``particle``: the text columns
    ([`MC_OBSERVABLE_TEXT_COLUMNS`][]) as ``str``, the rest as ``float``; rows are
    sorted by energy. A missing file reads as ``{}``; provenance header lines are
    skipped.
    """
    checkpoint = CsvCheckpoint(path, text_columns=MC_OBSERVABLE_TEXT_COLUMNS)
    grouped: dict[str, list[dict[str, Any]]] = {}
    for key in checkpoint.keys():
        for row in checkpoint.rows(key):
            grouped.setdefault(row["particle"], []).append(
                {
                    k: (v if k in MC_OBSERVABLE_TEXT_COLUMNS else float(v))
                    for k, v in row.items()
                    if k != "particle" and v not in (None, "")
                }
            )
    for rows in grouped.values():
        rows.sort(key=lambda r: float(r["energy_MeV_u"]))
    return grouped


ComparisonKey = tuple[str, str, float | None]
"""Comparison keys of [`compare_mc_to_paper`][]: ``(panel, particle, level)``."""


def compare_mc_to_paper(
    mc_results: Mapping[str, Sequence[Mapping[str, Any]]],
    *,
    sigma_particles: Sequence[str] = SIGMA_PARTICLES,
    rbe_particles: Sequence[str] = RBE_PARTICLES,
    levels: Sequence[float] = SURVIVAL_LEVELS,
) -> dict[ComparisonKey, CurveComparison]:
    """Residuals of the offline MC rows against the paper's own calculated lines.

    * ``("fig3", p, None)``: ``sigma_1pct_um2`` (with its jackknife error) against
      the Fig. 3 calc line of ``p``, interpolated in energy;
    * ``("fig3_sigma0", p, None)``: the low-dose ``sigma_low_dose_um2`` against the
      same line (the observable the paper does *not* plot, for contrast);
    * ``("fig4", p, level)``: ``rbe_<tag>`` against the Fig. 4 calc line at
      ``level``, interpolated in log LET.

    [`compare_points_to_curve`][] never extrapolates: an MC point outside the
    digitized x range is counted in ``points_outside_support``. The jackknife error
    is an MC allocation diagnostic, not a confidence interval on the residual.
    Raises ``ValueError`` (from [`compare_points_to_curve`][]) when a particle has
    no point inside the line's support.

    The printing stays in the script.
    """
    comparisons: dict[ComparisonKey, CurveComparison] = {}
    for particle in sigma_particles:
        curve = next(
            iter(iter_curves(figure="fig3_sigma_vs_energy_v79", particle=particle, series="calc"))
        )
        rows = mc_results.get(particle, [])
        comparisons[("fig3", particle, None)] = compare_points_to_curve(
            curve,
            x=[row["energy_MeV_u"] for row in rows],
            model_y=[row["sigma_1pct_um2"] for row in rows],
            model_error=[row["sigma_1pct_err_um2"] for row in rows],
        )
        comparisons[("fig3_sigma0", particle, None)] = compare_points_to_curve(
            curve,
            x=[row["energy_MeV_u"] for row in rows],
            model_y=[row["sigma_low_dose_um2"] for row in rows],
        )
    for particle in rbe_particles:
        rows = mc_results.get(particle, [])
        for level in levels:
            tag = LEVEL_TAG[level]
            curve = next(
                iter(
                    iter_curves(
                        figure="fig4_rbe_vs_let",
                        particle=particle,
                        series="calc",
                        survival_level=level,
                    )
                )
            )
            comparisons[("fig4", particle, level)] = compare_points_to_curve(
                curve,
                x=[row["let_keV_um"] for row in rows],
                model_y=[row[f"rbe_{tag}"] for row in rows],
                model_error=[row[f"rbe_{tag}_err"] for row in rows],
                log_x=True,
            )
    return comparisons


def paper_sigma_derived_rbe_1pct(
    let_keV_um: Callable[[Ion], float],
    *,
    reference_dose_Gy: float,
) -> dict[str, tuple[npt.NDArray[np.float64], npt.NDArray[np.float64]]]:
    """RBE at 1% survival implied by the paper's OWN sigma curves, ``{particle: (LET, RBE)}``.

    ``sigma`` and RBE live in different figures, so the paper never plots them
    together; this closes the loop at S = 1%, where both describe the same depth of
    the survival curve. Carbon comes from Fig. 3 (sigma vs energy, so it needs a
    LET(E) mapping: ``let_keV_um(Ion(6, 12, E))``); helium from Fig. 2 ``curve1``
    (sigma vs LET directly, so no stopping power enters). ``curve1`` is unlabelled in
    the digitization but is helium: it is the only calc curve overlapping the helium
    LET range at all (10.7-66.7 keV/um against the helium points' 20.4-58.2), and the
    model's own helium sigma(1%) tracks it to 2-5% at LET 32-48.

    The values are **upper bounds that tighten with LET**
    ([`iso_survival_rbe_from_final_slope`][]): a slope needs an intercept to give a
    dose. Each pair is sorted by LET; a missing curve is skipped.
    """
    sources = (
        ("carbon", "fig3_sigma_vs_energy_v79", "carbon"),
        ("helium", "fig2_sigma_vs_let_v79", "curve1"),
    )
    derived: dict[str, tuple[npt.NDArray[np.float64], npt.NDArray[np.float64]]] = {}
    for particle, figure, series in sources:
        curves = list(iter_curves(figure=figure, particle=series, series="calc"))
        if not curves:
            continue
        curve = curves[0]
        x = np.asarray(curve.x, dtype=float)
        sigma = np.asarray(curve.y, dtype=float)
        # Fig. 3's abscissa is energy; Fig. 2's is already LET.
        z, a = SPECIES[particle]
        let = (
            np.array([let_keV_um(Ion(z, a, float(e))) for e in x])
            if figure.startswith("fig3")
            else x
        )
        rbe = iso_survival_rbe_from_final_slope(sigma, let, reference_dose_Gy=reference_dose_Gy)
        order = np.argsort(let)
        derived[particle] = (let[order], rbe[order])
    return derived


# --- numeric fingerprint of the committed checkpoint -----------

MC_OBSERVABLES_FINGERPRINT_POINT: tuple[str, float] = ("carbon", 8.0)
"""The ``sk1996_mc_observables.csv`` fingerprint point and its reduced settings."""
MC_OBSERVABLES_FINGERPRINT_SETTINGS: Mapping[str, float] = MappingProxyType(
    {
        "n_batches": 4,
        "realizations_per_batch": 5,
        "max_realizations": 20,
        "target_relative_error": 0.03,
        "seed": 1996,
        "max_mean_traversals": DEFAULT_MAX_MEAN_TRAVERSALS,
    }
)


def mc_observables_fingerprint(stopping_power: StoppingPower) -> list[dict[str, Any]]:
    """Recompute the ``sk1996_mc_observables.csv`` fingerprint (one tiny point).

    One [`estimate_final_slope_point`][] run at
    [`MC_OBSERVABLES_FINGERPRINT_POINT`][] (carbon 8 MeV/u) with
    [`MC_OBSERVABLES_FINGERPRINT_SETTINGS`][] (20 histories) on
    [`v79_lem1_models`][] ``(stopping_power)``. The traversal-conditioned Monte
    Carlo is numpy-only, so there is no engine choice. Every column is kept.
    """
    particle, energy = MC_OBSERVABLES_FINGERPRINT_POINT
    settings = dict(MC_OBSERVABLES_FINGERPRINT_SETTINGS)
    full, bracket = v79_lem1_models(stopping_power)
    row = estimate_final_slope_point(
        full,
        bracket,
        particle,
        energy,
        n_batches=int(settings["n_batches"]),
        realizations_per_batch=int(settings["realizations_per_batch"]),
        max_realizations=int(settings["max_realizations"]),
        target_relative_error=float(settings["target_relative_error"]),
        seed=int(settings["seed"]),
        max_mean_traversals=float(settings["max_mean_traversals"]),
    )
    return [{**row, **settings}]


__all__ = [
    "DEFAULT_BATCHES",
    "DEFAULT_MAX_MEAN_TRAVERSALS",
    "DEFAULT_MAX_REALIZATIONS",
    "DEFAULT_REALIZATIONS_PER_BATCH",
    "DEFAULT_TARGET_RELATIVE_ERROR",
    "DOSE_HEADROOM",
    "ENERGIES_MEV_U",
    "FIG1A_BELLI_INPUTS",
    "FIGURE_INPUTS",
    "FINAL_SLOPE_ESTIMATOR_VERSION",
    "LEVEL_TAG",
    "MAX_ALLOCATION_GROWTH",
    "MAX_DOSE_EXPANSIONS",
    "MAX_POISSON_TAIL_MASS",
    "MC_OBSERVABLES_FINGERPRINT_POINT",
    "MC_OBSERVABLES_FINGERPRINT_SETTINGS",
    "MC_OBSERVABLE_FIELDS",
    "MC_OBSERVABLE_TEXT_COLUMNS",
    "MC_POINTS",
    "OBSERVABLE_ESTIMATOR_VERSION",
    "RBE_PARTICLES",
    "SCHOLZ_KRAFT1996_DOI",
    "SIGMA_PARTICLES",
    "SPECIES",
    "SURVIVAL_BRACKET_MARGIN",
    "SURVIVAL_LEVELS",
    "ComparisonKey",
    "CurveComparison",
    "CurveResidualPoint",
    "FinalSlopeEstimate",
    "IsoSurvivalEstimate",
    "PaperCellInputs",
    "ScholzKraft1996Curve",
    "checkpoint_reuse_reason",
    "compare_mc_to_paper",
    "compare_points_to_curve",
    "estimate_final_slope_cross_section",
    "estimate_final_slope_point",
    "estimate_iso_survival_rbe",
    "get_curve",
    "iter_curves",
    "load_curves",
    "load_mc_observables",
    "mc_observables_fingerprint",
    "paper_sigma_derived_rbe_1pct",
    "photon_dose_for_survival",
    "v79_lem1_models",
]
