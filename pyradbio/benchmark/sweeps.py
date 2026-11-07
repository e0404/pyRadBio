"""Energy and LET sweeps of a model's LQ parameters, with the RBE reductions on top.

The validation scripts and tests all need the same loop: "build an ion per grid
point, call ``alpha_beta``, record ``(LET, alpha, beta)``, maybe sort by LET, maybe
turn failures into NaN". The two sweeps below are that loop:

- :func:`sweep_energies` -- one ion per kinetic energy (failures become NaN by
  default; ``e_max_MeV_u`` caps the energy, ``sort_by_let`` orders the rows).
- :func:`sweep_lets` -- one ion per *LET*, the energy found by a bracketed root
  solve (NaN where the LET is unreachable; ``xtol`` sets the root tolerance).

Both return a :class:`LetSweep`, whose RBE reductions are two *named* methods
(no dispatch on the argument type), because the scripts use two different photon
references and mixing them up is a silent 14% error (see
:meth:`LetSweep.rbe10_shouldered`).

Printing and plotting stay in the scripts.
"""

from __future__ import annotations

import math
from dataclasses import dataclass
from typing import Literal, Protocol

import numpy as np
import numpy.typing as npt

from pyradbio.core.lq import LQParameters, dose_for_survival, rbe
from pyradbio.core.photon import PhotonParameters
from pyradbio.core.radiation import Ion
from pyradbio.track.stopping_power import StoppingPower, energy_for_let_MeV_u

__all__ = [
    "AlphaBetaModel",
    "LetSweep",
    "sweep_energies",
    "sweep_lets",
]

FloatArray = npt.NDArray[np.float64]


class AlphaBetaModel(Protocol):
    """Anything with a per-ion ``alpha_beta`` -- every :class:`BiophysicalModel`."""

    def alpha_beta(self, ion: Ion, /) -> LQParameters: ...


@dataclass(frozen=True, eq=False)
class LetSweep:
    """One ion species swept over energy (or LET): aligned 1-D float arrays.

    ``energy_MeV_u`` is the kinetic energy evaluated, ``let_keV_um`` the stopping
    power's LET of that ion, ``alpha``/``beta`` the model's LQ parameters. A row
    that could not be evaluated holds NaN in the affected columns (see the sweep
    functions for which). Equality is identity (``eq=False``): array fields have no
    useful ``==``.
    """

    energy_MeV_u: FloatArray
    let_keV_um: FloatArray
    alpha: FloatArray
    beta: FloatArray

    def __post_init__(self) -> None:
        shapes = {
            np.shape(self.energy_MeV_u),
            np.shape(self.let_keV_um),
            np.shape(self.alpha),
            np.shape(self.beta),
        }
        if len(shapes) != 1 or len(next(iter(shapes))) != 1:
            raise ValueError(f"LetSweep columns must be 1-D and equally long, got {shapes}")

    def rbe_alpha(self, alpha_x: float) -> FloatArray:
        """``RBE_alpha = alpha / alpha_x`` per row (NaN rows stay NaN).

        Element-wise float64 division, the same IEEE operation as a scalar
        ``lq.alpha / alpha_x``.
        """
        out: FloatArray = self.alpha / alpha_x
        return out

    def rbe10_lq(self, alpha_x: float, beta_x: float) -> FloatArray:
        """RBE at 10% survival against a *plain* LQ photon curve ``(alpha_x, beta_x)``.

        ``rbe(0.10, alpha, beta, reference=LQParameters(alpha_x, beta_x))`` per row,
        as in ``examples/lem2_cluster_effect.py`` ``carbon_curve(model, "rbe10")``
        (and, with the same expression order, ``tests/models/lem/test_lem4.py``
        ``_rbe10``): ``D10_x / D10_ion`` with both doses from the LQ inversion
        :func:`pyradbio.core.lq.dose_for_survival`. Correct only while the photon
        ``D10`` lies below the photon ``D_t``; otherwise use
        :meth:`rbe10_shouldered`. NaN rows stay NaN.
        """
        reference = LQParameters(alpha=alpha_x, beta=beta_x)
        out = np.full(self.alpha.shape, np.nan)
        for i, (a, b) in enumerate(zip(self.alpha, self.beta, strict=True)):
            if math.isnan(a) or math.isnan(b):
                continue
            out[i] = rbe(0.10, float(a), float(b), reference=reference)
        return out

    def rbe10_shouldered(self, photon: PhotonParameters) -> FloatArray:
        """RBE at 10% survival against the *shouldered* photon curve of ``photon``.

        ``photon.dose_for_survival(0.1) / dose_for_survival(0.1, alpha, beta)`` per
        row. The photon ``D10`` comes off the LEM photon curve, which turns exponential above
        ``D_t`` (Scholz & Kraft 1996, doi:10.1016/0273-1177(95)00784-C); inverting
        the LQ branch there instead (:meth:`rbe10_lq`) under-states ``D10_x``
        whenever ``D10_x > D_t`` -- 14.5% for the Elsasser 2010 V79 panel. NaN rows
        stay NaN.
        """
        d10_x = photon.dose_for_survival(0.1)
        out = np.full(self.alpha.shape, np.nan)
        for i, (a, b) in enumerate(zip(self.alpha, self.beta, strict=True)):
            if math.isnan(a) or math.isnan(b):
                continue
            out[i] = d10_x / dose_for_survival(0.1, float(a), float(b))
        return out


def _as_grid(values: npt.ArrayLike, name: str) -> FloatArray:
    """A fresh 1-D float64 copy of a user grid (never aliases the caller's array)."""
    grid: FloatArray = np.array(values, dtype=float)
    if grid.ndim != 1:
        raise ValueError(f"{name} must be 1-D, got shape {grid.shape}")
    return grid


def sweep_energies(
    model: AlphaBetaModel,
    Z: int,
    A: int,
    energies: npt.ArrayLike,
    *,
    stopping_power: StoppingPower,
    on_error: Literal["nan", "raise"] = "nan",
    errors: tuple[type[Exception], ...] = (ValueError, ZeroDivisionError),
    e_max_MeV_u: float = math.inf,
    sort_by_let: bool = False,
) -> LetSweep:
    """``alpha_beta`` of ion ``(Z, A)`` at each kinetic energy in ``energies`` [MeV/u].

    Per row, in grid order: the ion is ``Ion(Z, A, float(E))``; its LET is
    ``stopping_power.let_keV_um(ion)``; then, unless ``E > e_max_MeV_u`` (the
    route ceiling of ``examples/lem4_trip98_rbe_reproduction.py``, strict: a point
    exactly at the ceiling is evaluated; rows above it are never evaluated and
    keep NaN ``alpha``/``beta``), ``model.alpha_beta(ion)``.

    Error policy:

    - ``on_error="nan"`` (default): an exception of a type in ``errors`` from
      ``alpha_beta`` leaves that row's ``alpha``/``beta`` NaN; one from the LET
      call leaves its ``let_keV_um`` NaN; one from building the ion (``E <= 0``)
      leaves the whole row but ``energy_MeV_u`` NaN, and the model is not
      called. The default ``errors`` are the
      ``(ValueError, ZeroDivisionError)`` of ``examples/lem1_trip98_rbe_reproduction.py``
      (below ~0.4 MeV/u a LEM track has ``r_max <= r_min`` and raises);
      ``lem4_trip98`` adds ``RuntimeError``. Any other exception propagates.
    - ``on_error="raise"``: nothing is caught (the plain loops of lem2/lem3 and
      the Elsasser 2007 tests); ``errors`` is then ignored.

    A loop with no failing point gives the same bits under either policy.

    The LET column is always evaluated, also for call sites whose loop never
    computed a LET (``lem1_trip98``/``lem4_trip98`` ``rbe_alpha``,
    ``model_comparison_alpha_beta``): its errors fall under the same policy, so a
    stopping power raising a type outside ``errors`` (e.g. a ``KeyError`` for an
    ion a table does not cover) aborts a sweep the original loop finished.

    ``sort_by_let=True`` reorders all four columns by ascending LET (a stable
    sort, NaN LETs last). That is the ``pts.sort()`` of lem2/lem3/``test_lem4`` and
    the ``np.argsort(lets)`` of ``test_elsasser2007`` for any grid with distinct
    LETs; they could differ only on an exact LET tie, where the tuple sort would
    break the tie by value.

    Evaluating ``alpha_beta`` for a model whose scripts called a bare
    ``initial_slope`` gives the same ``alpha`` (LEM ``_alpha_beta_ion`` *is*
    ``initial_slope`` plus a beta model) at the extra cost of ``beta`` -- and a
    caught failure of the beta model blanks that row's ``alpha`` as well. A call
    site that sweeps a *specific* bare slope method (``lem1_trip98`` passes
    ``lem.lethal_central_initial_slope`` as well as ``initial_slope``) needs a
    model whose ``initial_slope_model`` selects that method (e.g.
    ``initial_slope_model="lethal_central"``), or a small adapter exposing it as
    ``alpha_beta``; ``initial_slope`` dispatches to the same method, so the bits
    are unchanged.
    """
    if on_error not in ("nan", "raise"):
        raise ValueError(f"on_error must be 'nan' or 'raise', got {on_error!r}")
    caught: tuple[type[Exception], ...] = errors if on_error == "nan" else ()

    energy = _as_grid(energies, "energies")
    let = np.full(energy.shape, np.nan)
    alpha = np.full(energy.shape, np.nan)
    beta = np.full(energy.shape, np.nan)
    for i, e in enumerate(energy):
        try:
            ion = Ion(Z=Z, A=A, energy_MeV_u=float(e))
        except caught:
            continue  # lem1/lem4_trip98 build the Ion inside their ``try``
        try:
            let[i] = stopping_power.let_keV_um(ion)
        except caught:
            pass
        if e > e_max_MeV_u:
            continue
        try:
            lq = model.alpha_beta(ion)
        except caught:
            continue
        alpha[i] = lq.alpha
        beta[i] = lq.beta

    if sort_by_let:
        order = np.argsort(let, kind="stable")
        energy, let, alpha, beta = energy[order], let[order], alpha[order], beta[order]
    return LetSweep(energy_MeV_u=energy, let_keV_um=let, alpha=alpha, beta=beta)


def sweep_lets(
    model: AlphaBetaModel,
    Z: int,
    A: int,
    lets: npt.ArrayLike,
    *,
    stopping_power: StoppingPower,
    energy_bounds_MeV_u: tuple[float, float],
    xtol: float | None = None,
    on_error: Literal["nan", "raise"] = "raise",
) -> LetSweep:
    """``alpha_beta`` of ion ``(Z, A)`` at the energy giving each LET in ``lets``.

    Per row: the energy is the root of ``LET(E) = let`` bracketed on exactly
    ``energy_bounds_MeV_u`` (``energy_for_let_MeV_u(branch="bracket")``, ``xtol``
    forwarded only when given), so the caller's bracket picks the branch of the non-monotone
    ``LET(E)``; then ``ion = Ion(Z, A, E)`` and ``model.alpha_beta(ion)``.

    A LET is *unreachable* on the bracket when ``LET(lo) - let`` and
    ``LET(hi) - let`` have the same sign -- e.g. above the ion's peak LET, below
    ``LET(hi)`` on a bracket on the falling branch, or (for a bracket spanning the
    peak) any LET above ``max(LET(lo), LET(hi))``, even though ``LET(E) = let``
    has roots there. Under ``on_error="raise"`` (default; the ``branch="bracket"``
    contract) scipy's ``ValueError`` propagates, as from the bare ``brentq`` of
    lem3 ``carbon_energy_for_let``, imk ``carbon_at_let`` and ``test_lem4``
    ``_carbon_energy_for_let``; under ``on_error="nan"`` its row is all NaN, as
    in ``examples/lem4_reconstruction.py`` ``sweep`` (which therefore passes
    ``on_error="nan"`` explicitly). Errors from the model propagate under either
    policy. A misconfigured bracket (not ``0 < lo < hi < inf``) is a
    ``ValueError`` before any row is evaluated.

    ``let_keV_um`` holds the stopping power's LET *at the found energy* (what
    ``tests/models/lem/test_lem4.py`` ``_rbe_alpha_curve`` sorts by), which agrees
    with the requested LET to the root tolerance; the requested grid is the
    caller's ``lets``. Rows stay in the order of ``lets``.
    """
    if on_error not in ("nan", "raise"):
        raise ValueError(f"on_error must be 'nan' or 'raise', got {on_error!r}")
    lo, hi = (float(b) for b in energy_bounds_MeV_u)
    if not (0.0 < lo < hi < math.inf):
        raise ValueError(
            f"energy_bounds_MeV_u must satisfy 0 < lo < hi < inf, got {energy_bounds_MeV_u!r}"
        )
    target = _as_grid(lets, "lets")
    energy = np.full(target.shape, np.nan)
    let = np.full(target.shape, np.nan)
    alpha = np.full(target.shape, np.nan)
    beta = np.full(target.shape, np.nan)
    for i, wanted in enumerate(target):
        try:
            e = energy_for_let_MeV_u(
                stopping_power,
                Z,
                A,
                float(wanted),
                branch="bracket",
                energy_bounds_MeV_u=energy_bounds_MeV_u,
                xtol=xtol,
            )
        except ValueError:
            if on_error == "raise":
                raise
            continue
        ion = Ion(Z, A, e)
        lq = model.alpha_beta(ion)
        energy[i] = e
        let[i] = stopping_power.let_keV_um(ion)
        alpha[i] = lq.alpha
        beta[i] = lq.beta
    return LetSweep(energy_MeV_u=energy, let_keV_um=let, alpha=alpha, beta=beta)
