"""Digitized LEM I benchmark curves from Scholz et al. 1997.

The source paper compares the original stochastic Local Effect Model (labelled
``Exact`` in the figures) with the fast central-traversal / specific-energy
approximation used for treatment-planning-style calculations:

Scholz, Kellerer, Kraft-Weyrather & Kraft 1997,
doi:10.1007/s004110050055.

The bundled numeric curves are **hand-digitized** from the paper's figures, and
are accurate enough for quantitative comparison (not just visual regression).
Over the plotted ranges every panel sits well below ``D_t = 30 Gy``, so the LQ
form is exact there and two figures are **derived rather than digitized**:

* **Figs. 2 and 4** are straight lines (``-ln S/D = alpha + beta D`` and
  ``alpha_z + 2 beta_z z``), stored as least-squares fits of the digitized points
  — two endpoints per curve, R^2 >= 0.998. So the Fig. 2 endpoints *are*
  ``(alpha, alpha + 30 beta)`` and the Fig. 4 endpoints ``(alpha_z, ...)``.
* **Fig. 5 is derived**, not digitized: ``S = exp(-(alpha D + beta D^2))``. The
  ``exact`` curves take ``(alpha, beta)`` from the Fig. 2 fit; the
  ``approximation`` curves take ``alpha`` from the Fig. 1 approximation branch
  (the overkill-saturated central slope ``(1-e^-N_1)/D_1`` — *not* the Fig. 4
  intercept ``alpha_z = N_1/D_1``, which the survival's Poisson averaging
  saturates away) and ``beta`` from the Fig. 3 approximation branch. Each curve
  runs to its ``S = 0.01`` crossing.

Note the Fig. 4 carbon panels: the paper labels the *exact* panel 400 MeV/u, but
its ``alpha_z`` matches 200 MeV/u (its approximation partner), so it is stored as
200 — a paper typo. The Fig. 2/Fig. 5 carbon curves genuinely are 400 MeV/u.

Each curve lives in its own CSV under ``data/`` named
``scholz1997_figure<N>_<description>.csv`` with a four-line header::

    <metadata keys>          e.g. figure,particle,calculation,energy_MeV_u,doi,source_page,note
    <metadata values>
    <x quantity>,<y quantity>   e.g. energy,alpha
    <x unit>,<y unit>           e.g. MeV/u,Gy^-1
    <x>,<y> data rows ...

Identity/provenance live in the metadata block; the filename is human-facing.
"""

from __future__ import annotations

import csv
import math
import os
from collections.abc import Mapping
from dataclasses import dataclass, field
from functools import lru_cache
from importlib import resources
from types import MappingProxyType
from typing import TYPE_CHECKING, Any

import numpy as np
import numpy.typing as npt

from pyradbio.benchmark.checkpoint import CsvCheckpoint
from pyradbio.core.cell import CellTarget
from pyradbio.core.lq import LQParameters, fit_lq_linearised
from pyradbio.core.photon import PhotonParameters
from pyradbio.core.radiation import Ion

if TYPE_CHECKING:
    from pyradbio.models.lem.engine import Engine
    from pyradbio.models.lem.lem1 import LEM1, SpecificEnergySurvival
    from pyradbio.track.stopping_power import StoppingPower

SCHOLZ1997_DOI = "10.1007/s004110050055"

_DATA_DIR = "data"
_CURVE_PREFIX = "scholz1997_figure"


@dataclass(frozen=True, slots=True)
class PaperCellInputs:
    """Cell/target inputs a Scholz 1997 panel used (LEM I photon + geometry).

    All Figs. 1-5 in the paper use the same CHO reference cell, so this doubles
    as the single :data:`CHO_INPUTS` record; it mirrors the per-figure inputs map
    of the Scholz & Kraft 1996 benchmark for parity between the two examples.
    """

    alpha_x: float  # Gy^-1
    beta_x: float  # Gy^-2
    D_t: float  # transition dose [Gy]
    nucleus_radius_um: float  # geometric nuclear radius [um]
    effective_size_factor: float = 1.0  # paper uses the geometric size directly

    def photon(self) -> PhotonParameters:
        """Return the LEM I photon reference ``(alpha_x, beta_x, D_t)``, no cluster.

        Scholz et al. 1997 (doi:10.1007/s004110050055), Fig. 1 caption. Equal to
        ``PhotonParameters(alpha_x=inputs.alpha_x, beta_x=inputs.beta_x, D_t=inputs.D_t)``.
        """
        return PhotonParameters(alpha_x=self.alpha_x, beta_x=self.beta_x, D_t=self.D_t)

    def cell_target(self) -> CellTarget:
        """Return the nucleus geometry, passing ``effective_size_factor`` on explicitly.

        Scholz et al. 1997 (doi:10.1007/s004110050055), Fig. 1 caption. Equal to
        ``CellTarget(nucleus_radius_um=inputs.nucleus_radius_um,
        effective_size_factor=inputs.effective_size_factor)``; the factor is forwarded
        rather than left to ``CellTarget``'s default so the paper's convention (its
        5 um radius used as-is) stays visible at the construction.
        """
        return CellTarget(
            nucleus_radius_um=self.nucleus_radius_um,
            effective_size_factor=self.effective_size_factor,
        )


# CHO reference cell used throughout Scholz et al. 1997 (Fig. 1 caption / text:
# alpha_x=0.18 Gy^-1, beta_x=0.028 Gy^-2, D_t=30 Gy, 10 um nuclear diameter).
CHO_INPUTS = PaperCellInputs(
    alpha_x=0.18,
    beta_x=0.028,
    D_t=30.0,
    nucleus_radius_um=5.0,
    effective_size_factor=1.0,
)


@dataclass(frozen=True, slots=True)
class Scholz1997Curve:
    """One digitized reference curve (one plotted line, from one CSV file)."""

    figure: str
    particle: str
    calculation: str
    energy_MeV_u: float | None
    x_quantity: str
    x_unit: str
    y_quantity: str
    y_unit: str
    x: tuple[float, ...]
    y: tuple[float, ...]
    source_page: int
    note: str
    doi: str = SCHOLZ1997_DOI


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


def _parse_curve(
    text: str,
) -> tuple[dict[str, str], tuple[str, str], tuple[str, str], list[tuple[float, float]]]:
    """Parse the four-line header + data of one curve file."""
    rows = list(csv.reader(text.splitlines()))
    meta = dict(zip(rows[0], rows[1], strict=True))
    quantities = (rows[2][0], rows[2][1])
    units = (rows[3][0], rows[3][1])
    points = sorted((float(r[0]), float(r[1])) for r in rows[4:] if r and r[0].strip())
    return meta, quantities, units, points


@lru_cache(maxsize=1)
def load_lem1_curves() -> tuple[Scholz1997Curve, ...]:
    """Return every digitized Scholz 1997 LEM I benchmark curve (one per file)."""
    curves: list[Scholz1997Curve] = []
    for _name, text in _data_files(_CURVE_PREFIX):
        meta, (x_quantity, y_quantity), (x_unit, y_unit), points = _parse_curve(text)
        curves.append(
            Scholz1997Curve(
                figure=meta["figure"],
                particle=meta["particle"],
                calculation=meta["calculation"],
                energy_MeV_u=_optional_float(meta["energy_MeV_u"]),
                x_quantity=x_quantity,
                x_unit=x_unit,
                y_quantity=y_quantity,
                y_unit=y_unit,
                x=tuple(x for x, _ in points),
                y=tuple(y for _, y in points),
                source_page=int(meta["source_page"]),
                note=meta["note"],
                doi=meta.get("doi", SCHOLZ1997_DOI),
            )
        )
    return tuple(
        sorted(
            curves,
            key=lambda curve: (
                curve.figure,
                curve.particle,
                curve.calculation,
                curve.energy_MeV_u if curve.energy_MeV_u is not None else -1.0,
            ),
        )
    )


def iter_lem1_curves(
    *,
    figure: str | None = None,
    particle: str | None = None,
    calculation: str | None = None,
    energy_MeV_u: float | None = None,
) -> tuple[Scholz1997Curve, ...]:
    """Filter digitized Scholz 1997 LEM I curves.

    ``calculation="exact"`` denotes the paper's explicit/full stochastic model
    path; ``calculation="approximation"`` denotes the fast approximation path.
    """
    curves = load_lem1_curves()
    return tuple(
        curve
        for curve in curves
        if (figure is None or curve.figure == figure)
        and (particle is None or curve.particle == particle)
        and (calculation is None or curve.calculation == calculation)
        and (energy_MeV_u is None or curve.energy_MeV_u == energy_MeV_u)
    )


def get_lem1_curve(
    figure: str,
    *,
    particle: str | None = None,
    calculation: str | None = None,
    energy_MeV_u: float | None = None,
) -> Scholz1997Curve:
    """Return exactly one Scholz 1997 LEM I reference curve.

    Raises ``LookupError`` if the filters select zero or multiple curves.
    """
    matches = iter_lem1_curves(
        figure=figure,
        particle=particle,
        calculation=calculation,
        energy_MeV_u=energy_MeV_u,
    )
    if len(matches) != 1:
        raise LookupError(f"expected exactly one Scholz 1997 curve, got {len(matches)}")
    return matches[0]


# --- LEM I models for the CHO panels ------------------------------------------------


def cho_lem1(stopping_power: StoppingPower | None = None, **model_kw: Any) -> LEM1:
    """``LEM1`` for the Scholz 1997 CHO cell (:data:`CHO_INPUTS`), all figures.

    ``CHO_INPUTS.photon()`` and ``CHO_INPUTS.cell_target()`` supply the photon
    reference and the geometry; ``model_kw`` are further ``LEM1`` fields
    (``beta_model``, ``initial_slope_model``, ...). ``stopping_power=None`` keeps
    the ``LEM1`` default (:func:`~pyradbio.track.stopping_power.default_stopping_power`),
    which is what the ``z-slope`` generator uses; the reconstructions pass the
    paper's ``ns1970`` choice explicitly.
    """
    # Imported here: the LEM stack (and numba) is seconds of import time that the
    # digitized curves and the checkpoint loaders of this module never need.
    from pyradbio.models.lem.lem1 import LEM1

    if stopping_power is not None:
        model_kw["stopping_power"] = stopping_power
    return LEM1(photon=CHO_INPUTS.photon(), cell=CHO_INPUTS.cell_target(), **model_kw)


#: Reduced (still <1%-converged) quadrature for ``LEM1.scholz1997_two_track_beta`` /
#: ``scholz1997_explicit_lq``: the two-track overlap integral is the heavy part of the
#: Fig. 3 and Fig. 2/5 panels, and refining every axis 4x moves it by <0.08%.
#: Shared by ``validation/scholz1997_lem1.py`` and ``examples/lem1_variants.py``.
TWO_TRACK_FAST_QUADRATURE: Mapping[str, int] = MappingProxyType(
    {"radial_nodes": 300, "separation_nodes": 50, "angular_nodes": 200, "impact_nodes": 60}
)


# --- the stochastic-LEM Monte-Carlo reference (Figs. 1 and 3) -----------------------
#
# The generator is scripts/lem1_mc_reference.py ``alpha-beta``; the checkpoint it
# writes is validation/data/mc_results.csv.

#: The two species the Scholz 1997 panels show, ``name -> (Z, A)``.
SPECIES: Mapping[str, tuple[int, int]] = MappingProxyType({"proton": (1, 1), "carbon": (6, 12)})

#: Survival doses of the alpha/beta fit (Scholz 1997 eq. 8 range) [Gy].
MC_FIT_DOSES_GY: npt.NDArray[np.float64] = np.array([0.5, 1.0, 1.5, 2.0, 2.5, 3.0])
MC_FIT_DOSES_GY.flags.writeable = False

#: Independent streams per point. The delete-one jackknife over them has
#: ``MC_N_STREAMS - 1`` degrees of freedom; the history budget is split across the
#: streams, not added to.
MC_N_STREAMS = 8

#: Far tracks become one compound-Poisson background draw beyond this cut [um]. 10 um
#: is safe for the high-energy points (far-field fluctuation 0.25-0.34%) but not below
#: about carbon 40 MeV/u (4.5%), where the split is worth little anyway, so it is only
#: applied from :data:`KELLERER_B_CUT_MIN_NBAR` traversals up.
KELLERER_B_CUT_UM = 10.0
KELLERER_B_CUT_MIN_NBAR = 2000.0

#: ``error_estimator_version`` of the rows :func:`montecarlo_lq_jackknife` writes.
#: v3 is the delete-one jackknife over :data:`MC_N_STREAMS` streams with weighted
#: (Kellerer) sampling, so no checkpoint mixes estimators. Rows of an older version
#: are recomputed, never carried forward.
MC_ERROR_ESTIMATOR_VERSION = 3

#: Columns of ``mc_results.csv``, in file order.
MC_REFERENCE_FIELDS: tuple[str, ...] = (
    "particle",
    "energy_MeV_u",
    "error_estimator_version",
    "alpha",
    "alpha_err",
    "beta",
    "beta_err",
    "n_realizations",
    "n_streams",
    "n_bar",
    "n_field_points",
    "alpha_cross_section",
    "alpha_rel_dev",
    "two_track_beta",
    "sampling",
    "stopping_power",
)
#: Provenance columns of ``mc_results.csv`` that are text, not numbers. ``sampling``
#: is always ``"kellerer"`` (the only sampler); the column keeps the rows
#: self-describing. ``stopping_power`` is the ``.name`` of the LET
#: source a row was computed at: a row at another LET is a different point (13% in
#: alpha at carbon 1 MeV/u between ``ns1970+bethe`` and analytic Bethe).
MC_REFERENCE_TEXT_COLUMNS = frozenset({"particle", "sampling", "stopping_power"})


def montecarlo_lq_jackknife(
    model: LEM1,
    particle: str,
    energy: float,
    n_total: int,
    engine: Engine,
    seed: int,
) -> dict[str, float | str]:
    """One Scholz 1997 reference point: ``MC_N_STREAMS`` streams, pooled fit, jackknife error.

    Runs ``LEM1.scholz1997_montecarlo_survival`` (the paper's "exact" path, Kellerer
    weighted sampling, ``n_field_points = MC_KELLERER_FIELD_POINTS``) on
    ``n_total // MC_N_STREAMS`` realizations per stream at :data:`MC_FIT_DOSES_GY`,
    seeds ``seed + stream``. The fit (Scholz 1997 eq. 8, ``-ln S/D = alpha + beta D``,
    :func:`~pyradbio.core.lq.fit_lq_linearised`) is nonlinear in the survivals, so
    the streams are pooled *in survival* and fitted once -- averaging per-stream fits
    would return a different, biased quantity. The error is a delete-one jackknife
    over the streams. Neither coefficient is clamped.

    ``alpha_rel_dev`` is the row's own validity gate: ``LEM1.initial_slope`` is the
    *exact* deterministic low-dose slope, so any MC disagreement with it is estimator
    error, not physics (within 0.63% across the committed catalogue); a row where it
    opens up should not have its ``beta`` believed either. ``beta`` is the quadratic
    coefficient of a least-squares fit over 0.5-3 Gy, which projects part of the cubic
    and higher cluster terms onto ``D^2``; it is not the window-free
    ``scholz1997_two_track_beta`` it is stored next to (measured at proton 7 MeV/u:
    ratio 0.995 / 0.986 / 0.971 for windows 0.25-1.5 / 0.5-3.0 / 1.0-6.0 Gy).

    The near/far split (``b_cut_um = KELLERER_B_CUT_UM``) is applied when the mean
    traversal count at 3 Gy reaches :data:`KELLERER_B_CUT_MIN_NBAR`. The field-point
    budget is fixed rather than scaled with ``n_bar``: weighted-sampling precision is
    flat in the track count (relative SD 0.48/0.55/0.81% over a 36x range).

    The alpha/beta fit is :func:`~pyradbio.core.lq.fit_lq_linearised` of ``-ln S``.

    Returns:
        One row of ``mc_results.csv`` (:data:`MC_REFERENCE_FIELDS`), numbers as floats.
    """
    from pyradbio.models.lem.base import MC_KELLERER_FIELD_POINTS

    z, a = SPECIES[particle]
    ion = Ion(z, a, energy)
    n_per_stream = max(n_total // MC_N_STREAMS, 1)
    n_bar = model.mean_traversals(ion, float(MC_FIT_DOSES_GY[-1]))
    kw: dict[str, object] = {"engine": engine}
    if n_bar >= KELLERER_B_CUT_MIN_NBAR:
        kw["b_cut_um"] = KELLERER_B_CUT_UM

    survivals = np.vstack(
        [
            model.scholz1997_montecarlo_survival(
                ion,
                MC_FIT_DOSES_GY,
                n_realizations=n_per_stream,
                n_field_points=MC_KELLERER_FIELD_POINTS,
                seed=seed + stream,
                **kw,  # type: ignore[arg-type]
            )
            for stream in range(MC_N_STREAMS)
        ]
    )

    alpha, beta = fit_lq_linearised(MC_FIT_DOSES_GY, -np.log(survivals.mean(axis=0)))
    replicates = np.array(
        [
            fit_lq_linearised(
                MC_FIT_DOSES_GY, -np.log(np.delete(survivals, i, axis=0).mean(axis=0))
            )
            for i in range(MC_N_STREAMS)
        ]
    )
    scale = (MC_N_STREAMS - 1) / MC_N_STREAMS
    errors = np.sqrt(scale * ((replicates - replicates.mean(axis=0)) ** 2).sum(axis=0))
    exact_alpha = model.initial_slope(ion)

    return {
        "particle": particle,
        "energy_MeV_u": energy,
        "error_estimator_version": float(MC_ERROR_ESTIMATOR_VERSION),
        "alpha": alpha,
        "alpha_err": float(errors[0]),
        "beta": beta,
        "beta_err": float(errors[1]),
        "n_realizations": float(MC_N_STREAMS * n_per_stream),
        "n_streams": float(MC_N_STREAMS),
        "n_bar": n_bar,
        "n_field_points": float(MC_KELLERER_FIELD_POINTS),
        "alpha_cross_section": exact_alpha,
        "alpha_rel_dev": alpha / exact_alpha - 1.0,
        "two_track_beta": model.scholz1997_two_track_beta(ion),
        "sampling": "kellerer",
        "stopping_power": model.stopping_power.name,  # type: ignore[attr-defined]
    }


@dataclass(frozen=True, slots=True)
class MCReference:
    """The ``mc_results.csv`` rows a reconstruction overlays (Figs. 1 and 3).

    ``rows`` maps particle -> rows sorted by energy, each row every numeric column
    as ``float`` (the text columns :data:`MC_REFERENCE_TEXT_COLUMNS` are left out).
    ``dropped`` counts the rows skipped because they were computed at another LET:
    ``{stored stopping_power name: count}``.
    """

    rows: dict[str, list[dict[str, float]]] = field(default_factory=dict)
    dropped: dict[str, int] = field(default_factory=dict)


def load_mc_reference(
    path: str | os.PathLike[str], *, stopping_power_name: str | None = None
) -> MCReference:
    """Read ``mc_results.csv`` for plotting; a missing file reads as empty.

    Rows whose ``stopping_power`` differs from ``stopping_power_name`` are dropped
    and counted (``None`` keeps every row): the stars are overlaid on curves the
    reconstruction computes itself, so a row from another stopping power is a
    different point, not a noisier one. Provenance header lines (``# key: value``,
    :mod:`pyradbio.benchmark.checkpoint`) are skipped.

    The dropped counts are returned, not printed; the caller reports them.
    """
    checkpoint = CsvCheckpoint(path, text_columns=MC_REFERENCE_TEXT_COLUMNS)
    reference = MCReference()
    for key in checkpoint.keys():
        for row in checkpoint.rows(key):
            stored = row.get("stopping_power")
            if stopping_power_name is not None and stored is not None:
                if stored != stopping_power_name:
                    reference.dropped[stored] = reference.dropped.get(stored, 0) + 1
                    continue
            reference.rows.setdefault(row["particle"], []).append(
                {k: float(v) for k, v in row.items() if k not in MC_REFERENCE_TEXT_COLUMNS}
            )
    for rows in reference.rows.values():
        rows.sort(key=lambda r: r["energy_MeV_u"])
    return reference


# --- the z-conditioned Monte Carlo (Fig. 4 exact panels) ----------------------------
#
# The generator is scripts/lem1_mc_reference.py ``z-slope``; the checkpoint is
# validation/data/sk1997_z_slope.csv.

#: The (particle, energy) points Scholz 1997 Fig. 4 draws.
Z_SLOPE_POINTS: tuple[tuple[str, float], ...] = tuple(
    (particle, energy) for particle in SPECIES for energy in (1.0, 8.0, 50.0, 200.0)
)
Z_SLOPE_Z_MAX_GY = 30.0
Z_SLOPE_N_BINS = 12
#: Many fluences, few realizations each: each fluence deposits a narrow bump in z, so
#: coverage -- not history count -- limits the measured *shape*. At a fixed 48 000
#: histories, 120 means instead of 12 drop the chi-square per degree of freedom of a
#: quadratic fit from 64 to 7, for free.
Z_SLOPE_N_POISSON_MEANS = 120
Z_SLOPE_REALIZATIONS_PER_MEAN = 1200
Z_SLOPE_N_FIELD_POINTS = 400
#: Far tracks enter as a Gaussian background with closed-form moments; without it,
#: proton 200 MeV/u would have to enumerate 2.3e8 traversals per realization.
Z_SLOPE_B_CUT_UM = 10.0

#: Columns of ``sk1997_z_slope.csv``, in file order (several rows per point).
Z_SLOPE_FIELDS: tuple[str, ...] = (
    "particle",
    "energy_MeV_u",
    "z_Gy",
    "log_kill",
    "slope_Gy_inv",
    "counts",
    "realizations_per_mean",
    "n_field_points",
    "b_cut_um",
    "log_kill_err",
    "alpha_z",
    "beta_z",
)
Z_SLOPE_TEXT_COLUMNS = frozenset({"particle"})


@dataclass(frozen=True, slots=True)
class ZSlopePoint:
    """One z-conditioned point: its checkpoint rows, the binned curve and ``(alpha_z, beta_z)``."""

    rows: list[dict[str, object]]
    curve: SpecificEnergySurvival
    z_lq: LQParameters


def z_slope_point(
    model: LEM1,
    particle: str,
    energy: float,
    *,
    seed: int,
    engine: Engine,
    z_max_Gy: float = Z_SLOPE_Z_MAX_GY,
    n_bins: int = Z_SLOPE_N_BINS,
    n_poisson_means: int = Z_SLOPE_N_POISSON_MEANS,
    realizations_per_mean: int = Z_SLOPE_REALIZATIONS_PER_MEAN,
    n_field_points: int = Z_SLOPE_N_FIELD_POINTS,
    b_cut_um: float = Z_SLOPE_B_CUT_UM,
    **survival_kw: Any,
) -> ZSlopePoint:
    """Exact ``S(z)`` at one Fig. 4 point and its ``sk1997_z_slope.csv`` rows.

    Calls ``LEM1.scholz1997_specific_energy_survival`` (``survival_kw`` reaches it,
    e.g. ``min_bin_count`` for a small fingerprint run) and builds one row per ``z``
    bin with the central-difference slope ``d[-ln S]/dz`` and the fast path's
    ``(alpha_z, beta_z)`` (``scholz1997_specific_energy_lq``). ``z`` is a per-cell
    stochastic quantity, so ``S(z)`` is neither the dose-domain ``S(D)`` nor
    recoverable from the traversal-conditioned survival.
    """
    z, a = SPECIES[particle]
    ion = Ion(z, a, energy)
    curve = model.scholz1997_specific_energy_survival(
        ion,
        z_max_Gy=z_max_Gy,
        n_bins=n_bins,
        n_poisson_means=n_poisson_means,
        realizations_per_mean=realizations_per_mean,
        n_field_points=n_field_points,
        b_cut_um=b_cut_um,
        seed=seed,
        engine=engine,
        **survival_kw,
    )
    z_lq = model.scholz1997_specific_energy_lq(ion)
    slope = curve.slope_Gy_inv()
    rows: list[dict[str, object]] = [
        {
            "particle": particle,
            "energy_MeV_u": energy,
            "z_Gy": float(zi),
            "log_kill": float(lk),
            "log_kill_err": float(le),
            "slope_Gy_inv": float(sl),
            "counts": int(ct),
            "realizations_per_mean": realizations_per_mean,
            "n_field_points": n_field_points,
            "b_cut_um": b_cut_um,
            "alpha_z": z_lq.alpha,
            "beta_z": z_lq.beta,
        }
        for zi, lk, le, sl, ct in zip(
            curve.z_Gy, curve.log_kill, curve.log_kill_err, slope, curve.counts, strict=True
        )
    ]
    return ZSlopePoint(rows=rows, curve=curve, z_lq=z_lq)


def z_traversals(model: LEM1, ion: Ion, z_Gy: float) -> float:
    """Mean traversals in the ``b_max = R_eff + r_max`` disc giving specific energy ``z_Gy``.

    The cost driver of :func:`z_slope_point`: 23 for carbon at 1 MeV/u, 2.3e8 for
    protons at 200 MeV/u, where the track is 200 um wide.
    """
    r_eff = model.cell.effective_nucleus_radius_um()
    b_max = r_eff + model.resolved_radial_dose.r_max_um(ion)
    return z_Gy * math.pi * b_max**2 / (math.pi * r_eff**2 * model.mean_traversal_dose_Gy(ion))


def load_z_slope_reference(
    path: str | os.PathLike[str],
) -> dict[tuple[str, float], tuple[npt.NDArray[np.float64], npt.NDArray[np.float64]]]:
    """Read ``sk1997_z_slope.csv`` as ``{(particle, energy): (z_Gy, slope_Gy_inv)}``.

    Each curve is sorted by ``z``; a missing file reads as ``{}``. Provenance header
    lines are skipped.
    """
    checkpoint = CsvCheckpoint(path, text_columns=Z_SLOPE_TEXT_COLUMNS)
    out: dict[tuple[str, float], tuple[npt.NDArray[np.float64], npt.NDArray[np.float64]]] = {}
    for key in checkpoint.keys():
        pairs = sorted(
            (float(row["z_Gy"]), float(row["slope_Gy_inv"])) for row in checkpoint.rows(key)
        )
        out[(str(key[0]), float(key[1]))] = (
            np.array([z for z, _ in pairs]),
            np.array([sl for _, sl in pairs]),
        )
    return out


# --- numeric fingerprints of the committed checkpoints ---------

#: The ``mc_results.csv`` fingerprint points: (particle, energy, n_total, seed). Carbon
#: 6 MeV/u (n_bar 8.8) runs the plain weighted sampler; carbon 70 MeV/u (n_bar 9341,
#: above :data:`KELLERER_B_CUT_MIN_NBAR`) runs the near/far split that produced the
#: high-energy rows, so a change confined to either branch is caught.
MC_REFERENCE_FINGERPRINT_POINTS: tuple[tuple[str, float, int, int], ...] = (
    ("carbon", 6.0, 16, 1),
    ("carbon", 70.0, 8, 1),
)
#: The ``sk1997_z_slope.csv`` fingerprint point and its reduced settings.
Z_SLOPE_FINGERPRINT_POINT: tuple[str, float] = ("carbon", 8.0)
Z_SLOPE_FINGERPRINT_SETTINGS: Mapping[str, int] = MappingProxyType(
    {"n_bins": 6, "n_poisson_means": 12, "realizations_per_mean": 20, "min_bin_count": 5, "seed": 7}
)


def mc_reference_fingerprint(stopping_power: StoppingPower) -> list[dict[str, Any]]:
    """Recompute the ``mc_results.csv`` fingerprint: two tiny :func:`montecarlo_lq_jackknife`.

    :data:`MC_REFERENCE_FINGERPRINT_POINTS` (carbon 6 MeV/u with 16 histories and
    carbon 70 MeV/u with 8, seed 1) on
    ``cho_lem1(stopping_power, beta_model="scholz1997_montecarlo")`` with
    ``engine="numpy"``; every column of each row is kept. A change in any of them
    means the code no longer computes what the committed rows were computed with.
    """
    model = cho_lem1(stopping_power, beta_model="scholz1997_montecarlo")
    return [
        {
            **montecarlo_lq_jackknife(model, particle, energy, n_total, "numpy", seed),
            "n_total": n_total,
            "seed": seed,
            "engine": "numpy",
        }
        for particle, energy, n_total, seed in MC_REFERENCE_FINGERPRINT_POINTS
    ]


def z_slope_fingerprint(stopping_power: StoppingPower | None = None) -> list[dict[str, Any]]:
    """Recompute the ``sk1997_z_slope.csv`` fingerprint: one reduced :func:`z_slope_point`.

    :data:`Z_SLOPE_FINGERPRINT_POINT` (carbon 8 MeV/u) with
    :data:`Z_SLOPE_FINGERPRINT_SETTINGS` and ``engine="numpy"`` on
    ``cho_lem1(stopping_power, beta_model="scholz1997_montecarlo")``; ``None`` keeps
    the ``LEM1`` default stopping power, like the generator.
    """
    particle, energy = Z_SLOPE_FINGERPRINT_POINT
    settings = dict(Z_SLOPE_FINGERPRINT_SETTINGS)
    model = cho_lem1(stopping_power, beta_model="scholz1997_montecarlo")
    point = z_slope_point(model, particle, energy, engine="numpy", **settings)
    columns = ("z_Gy", "log_kill", "log_kill_err", "slope_Gy_inv", "counts")
    return [
        {
            "particle": particle,
            "energy_MeV_u": energy,
            **settings,
            "engine": "numpy",
            "alpha_z": point.z_lq.alpha,
            "beta_z": point.z_lq.beta,
            **{column: [row[column] for row in point.rows] for column in columns},
        }
    ]


__all__ = [
    "CHO_INPUTS",
    "KELLERER_B_CUT_MIN_NBAR",
    "KELLERER_B_CUT_UM",
    "MC_ERROR_ESTIMATOR_VERSION",
    "MC_FIT_DOSES_GY",
    "MC_N_STREAMS",
    "MC_REFERENCE_FIELDS",
    "MC_REFERENCE_FINGERPRINT_POINTS",
    "MC_REFERENCE_TEXT_COLUMNS",
    "SCHOLZ1997_DOI",
    "SPECIES",
    "TWO_TRACK_FAST_QUADRATURE",
    "Z_SLOPE_B_CUT_UM",
    "Z_SLOPE_FIELDS",
    "Z_SLOPE_FINGERPRINT_POINT",
    "Z_SLOPE_FINGERPRINT_SETTINGS",
    "Z_SLOPE_N_BINS",
    "Z_SLOPE_N_FIELD_POINTS",
    "Z_SLOPE_N_POISSON_MEANS",
    "Z_SLOPE_POINTS",
    "Z_SLOPE_REALIZATIONS_PER_MEAN",
    "Z_SLOPE_TEXT_COLUMNS",
    "Z_SLOPE_Z_MAX_GY",
    "MCReference",
    "PaperCellInputs",
    "Scholz1997Curve",
    "ZSlopePoint",
    "cho_lem1",
    "get_lem1_curve",
    "iter_lem1_curves",
    "load_lem1_curves",
    "load_mc_reference",
    "load_z_slope_reference",
    "mc_reference_fingerprint",
    "montecarlo_lq_jackknife",
    "z_slope_fingerprint",
    "z_slope_point",
    "z_traversals",
]
