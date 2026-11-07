"""Digitized OER benchmark curves from Inaniwa et al. 2021 (OSMK, figs. 5-7).

Source paper (the original OSMK):

Inaniwa, T., Kanematsu, N., Shinoto, M., Koto, M. & Yamada, S. 2021, "Adaptation
of stochastic microdosimetric kinetic model to hypoxia for hypo-fractionated
multi-ion therapy treatment planning", Phys. Med. Biol. 66(20):205007,
doi:10.1088/1361-6560/ac29cc.

All three of the paper's **oxygen** figures are digitized here -- the oxygen
enhancement ratio at 10% survival against dose-averaged LET, in every case the
ratio of the ``pO2`` dose to the aerobic (160 mmHg) dose:

* **fig. 5** (HSG, p. 10) and **fig. 6** (V79, p. 11): anoxic (0 mmHg), helium,
  carbon and neon, measured by Furusawa et al. 2000 / 2012. Six series each.
* **fig. 7** (CHO, p. 11): carbon, nitrogen, oxygen and silicon at **four**
  oxygen pressures -- 0, 1.14, 3.8 and 15.2 mmHg -- measured by Tinganelli et al.
  2015. Eighteen series.

Fig. 7 is the more searching test of the two kinds, and not only because it adds
species. Figs. 5-6 use only the two endpoint pressures, so they constrain
``Rmax`` but say nothing about the *shape* of ``R(pO2)`` in between; fig. 7 is
the only published data that exercises the Alper & Howard-Flanders curve at
intermediate pressures, and hence the only test of the assumed ``K = 3 mmHg``.

Why these figures
-----------------
They are the only published check on the whole oxygen chain end to end --
``z_d,D`` -> ``Rmax(z_d,D)`` (eq. 14) -> ``R(pO2)`` (eq. 13) -> the LQ pair
(eqs. 17-18) -> eq.-(24) survival -> a dose ratio. Everything upstream is already
pinned: the SMK microdosimetry against Inaniwa 2018 fig. 4
(:mod:`pyradbio.benchmark.inaniwa2018`) and the table-1 parameters against the
paper itself. The OER is what nothing else tests.

The LET axis: 15 in figs. 5-6, 10 in fig. 7
-------------------------------------------
**Figs. 5 and 6 start at 15.0 keV/um, not at 10. Fig. 7 does start at 10.** In
all three Origin drew only two x labels, ``100`` and ``1000``, and left the
left-hand frame edge unlabelled, so reading them all as clean 10-to-1000 decade
pairs is the natural mistake -- and on figs. 5-6 it compresses every LET by up to
a factor 1.5 at the left edge while leaving the right edge exact, which no
eyeball check on the plot will catch. That the three panels are *not* all alike
is the trap: two of them share an axis and the third does not.

The tick marks settle each panel without assuming anything. All three carry 17
interior ticks (2..9 of one decade, the major, then 2..9 of the next). Fitting
``x = x0 + d*log10(v)`` over those slots gives a decade width of 315.9 px in
fig. 5, 316.2 in fig. 6 and 287.9 in fig. 7, in every case to within 0.5 px on
every tick, against frames 576/577 px wide. So figs. 5-6 span 1.823 decades and
start at ``1000 * 10**-1.823 = 15.0``, while fig. 7 spans 2.001 and starts at
9.98. Equivalently, and easier to check by eye: in figs. 5-6 the first interior
tick is the one labelled 20 and it sits 39 px from the left edge where a decade
is 316 px; at a left edge of 10 it would sit 95 px in.

Three independent confirmations, none used in the calibration:

* figs. 5-6 then reproduce the endpoints of the LET range the paper *states* it
  fitted (:data:`FITTED_LET_RANGE_KEV_UM`, sec 3.2) -- helium's lowest measured
  point lands at 18.53 keV/um against a quoted 18.5, and neon's highest at 654.8
  against a quoted 654;
* fig. 7's measured LETs land on the values the paper quotes in *words*, in
  fig. 4's six survival-panel labels: 51.1 (C at 0 mmHg), 29.8 / 68.0 / 124.5
  (C at 1.14), 124.5 (C at 3.8) and 307.6 keV/um (Si at 15.2). All five within
  0.3%, and none of them read off any axis;
* an independent automated colour-mask trace of figs. 5-6, calibrated independently
  from the same ticks, agrees with what is shipped here to 0.5% in LET and 0.005
  in OER (median ratio 1.000-1.004 over all six fig.-6 series).

Digitization method
-------------------
Unlike Inaniwa 2018, this PDF draws its figures as **raster images**, so there
is no content stream to read. All three panels as shipped are digitized **by
hand**. They replaced an automated colour-mask trace of figs. 5-6, which
survives only as the cross-check above: the hand pass finds points the automated
one missed (one to three per measured series) and, more importantly, does not
stop early where curves overlap -- it carries the V79 helium model curve to 220
keV/um where the trace gave up at 89.

Fig. 7's measured series carry **two digitized points per datum**: the mean and
the top of the 1-sigma error bar, at the same LET. :attr:`OERCurve.oer` holds the
mean and :attr:`OERCurve.oer_upper` the bar, so :attr:`OERCurve.sigma` is their
difference. Figs. 5-6 are drawn without error bars and have none.

The automated pass had four traps worth recording, since anything re-tracing
these panels will meet them (the extraction script and its verification overlays
are described in ``docs/references/OSMK_Inaniwa2021_2023.md``):

* each species' curve and its markers share one colour -- separated by the
  **distance transform**, since the curves are ~2 px thick (distance <= 1.5)
  while the markers are 7x7 squares/circles (3.0-3.6) and 9x9 diamonds
  (3.6-5.0); marker centres are the distance maxima, NMS'd at 5 px;
* markers **overlap** in the crowded high-LET cluster -- which the distance
  transform resolves for free, where an area heuristic does not: the largest
  blue cluster in fig. 5 is 245 px of mask, and dividing by one marker's area
  suggests six markers where a pixel-level look shows two;
* the in-plot cell label ("HSG"/"V79") is the same black as the helium series,
  and the **legend** reuses all three colours -- both are cut by coordinates,
  and both regions are checked to contain no data marker;
* the axis anchor is read from the **outward** tick marks, never assumed, as
  above.

Fig. 7 is harder again, and is why it was left until last: there **colour encodes
pO2 and line style encodes species**, so its four curves per colour are
near-degenerate over much of the range, its in-plot pO2 annotations are drawn in
the same four data colours, and its markers carry error bars.

Two limitations are inherent to the figures rather than to any method, and both
are why :func:`reconstruction_lets` exists:

* the model curves are drawn in the order He, C, Ne, so where they converge the
  **helium curve is overdrawn** and simply is not in the image. Figs. 5-6's
  helium curves stop near 220 keV/um where carbon and neon paint over them,
  against 770 and 986 for those two. :attr:`OERCurve.let` reports what was
  recovered.
* the curves are drawn well past the LET their **ion can physically reach**:
  helium tops out near its Bragg peak, ~178 keV/um with
  :class:`~pyradbio.track.stopping_power.SimpleBetheStoppingPower`, and neon
  cannot get *below* ~22 keV/um at any energy. Outside that window ``LET ->
  energy`` inversion is undefined, in both directions.

Reconstruction helpers
----------------------
:func:`model_oer` drives a model from nothing but a series' ion species, a list
of LETs (inverted for energy with
:func:`~pyradbio.track.stopping_power.energy_for_let_MeV_u`) and the series'
hypoxic pO2; :func:`oer_ratios` pools ``ours / paper`` over every species and
pressure of one figure; :func:`with_scaled_z_domain` is the ``z_d,D`` rescaling
diagnostic -- eq. (14) reads ``z_d,D`` only through ``(z_d,D/z_R)^gamma``, so
scaling it is the whole of the "is our microdosimetry biased?" question. They
are shared by ``validation/osmk_inaniwa2021.py`` and the golden tests.
"""

from __future__ import annotations

import csv
from dataclasses import dataclass, fields
from importlib import resources
from typing import Literal

import numpy as np
import numpy.typing as npt

from pyradbio.core.radiation import Ion
from pyradbio.models.mkm.osmk import OSMK2021, OSMK2023
from pyradbio.models.mkm.spectrum_models import SiteSpecificEnergies
from pyradbio.track.stopping_power import StoppingPower, energy_for_let_MeV_u

INANIWA2021_DOI = "10.1088/1361-6560/ac29cc"
_DATA_DIR = "data"
_PREFIX = "inaniwa2021"

#: Figure key -> (cell line, source page in the PDF).
FIGURES: dict[str, tuple[str, int]] = {
    "fig5_hsg": ("HSG", 10),
    "fig6_v79": ("V79", 11),
    "fig7_cho": ("CHO", 11),
}

#: Species label -> (Z, A) across all three figures.
SPECIES: dict[str, tuple[int, int]] = {
    "He": (2, 4),
    "C": (6, 12),
    "N": (7, 14),
    "O": (8, 16),
    "Ne": (10, 20),
    "Si": (14, 28),
}

#: Figure -> hypoxic pO2 [mmHg] -> the species drawn at that pressure. Figs. 5-6
#: are anoxic only; fig. 7 is the one that varies pO2.
CONTENTS: dict[str, dict[float, tuple[str, ...]]] = {
    "fig5_hsg": {0.0: ("He", "C", "Ne")},
    "fig6_v79": {0.0: ("He", "C", "Ne")},
    "fig7_cho": {
        0.0: ("C", "N", "O", "Si"),
        1.14: ("C",),
        3.8: ("C", "Si"),
        15.2: ("C", "Si"),
    },
}

#: The reference arm every OER in these figures is formed against [mmHg]
#: (paper sec 4.2). The hypoxic arm is :attr:`OERCurve.pO2_mmHg`.
OER_REFERENCE_PO2_MMHG = 160.0

#: Survival level the OER is defined at (the ratio of 10%-survival doses).
OER_SURVIVAL_FRACTION = 0.1

#: Cell line -> LET range [keV/um] its underlying data, and hence the fit, spans.
#: Outside it both OSMK formulations extrapolate, and they diverge sharply below
#: it; see the reference summary. HSG/V79 are Furusawa et al. 2000 / 2012 and the
#: paper states the range in sec 3.2. CHO is Tinganelli et al. 2015, whose LET
#: range the paper does *not* state, so this one is read off fig. 7's own
#: measured points rather than quoted.
FITTED_LET_RANGE_KEV_UM: dict[str, tuple[float, float]] = {
    "HSG": (18.5, 654.0),
    "V79": (18.5, 654.0),
    "CHO": (29.8, 316.9),
}


def _data_lines(text: str) -> list[str]:
    """CSV lines without the ``#`` citation header the data files carry."""
    return [line for line in text.splitlines() if not line.startswith("#")]


def _po2_tag(pO2_mmHg: float) -> str:
    """``3.8 -> '3p8mmHg'`` -- the pO2 part of a data file's name."""
    return f"{pO2_mmHg:g}".replace(".", "p") + "mmHg"


@dataclass(frozen=True, slots=True)
class OERCurve:
    """One digitized series: OER at 10% survival against dose-averaged LET."""

    figure: str  # "fig5_hsg" | "fig6_v79" | "fig7_cho"
    cell_line: str  # "HSG" | "V79" | "CHO"
    species: str  # a key of SPECIES
    series: str  # "exp" (measured points) | "osmk" (the paper's model curve)
    let_keV_um: tuple[float, ...]
    oer: tuple[float, ...]
    source_page: int
    #: The *hypoxic* arm of the ratio [mmHg]; the reference arm is always 160.
    pO2_mmHg: float = 0.0
    #: Top of the 1-sigma error bar, where the figure draws one (fig. 7 only).
    oer_upper: tuple[float, ...] = ()
    doi: str = INANIWA2021_DOI

    @property
    def Z(self) -> int:
        return SPECIES[self.species][0]

    @property
    def A(self) -> int:
        return SPECIES[self.species][1]

    @property
    def let(self) -> npt.NDArray[np.float64]:
        """Dose-averaged LET [keV/um] as an array."""
        return np.asarray(self.let_keV_um, dtype=np.float64)

    @property
    def values(self) -> npt.NDArray[np.float64]:
        """OER values as an array."""
        return np.asarray(self.oer, dtype=np.float64)

    @property
    def sigma(self) -> npt.NDArray[np.float64]:
        """1-sigma error bar per point, or zeros where the figure draws none."""
        if not self.oer_upper:
            return np.zeros_like(self.values)
        return np.asarray(self.oer_upper, dtype=np.float64) - self.values

    def interp(self, let_keV_um: npt.NDArray[np.float64] | float) -> npt.NDArray[np.float64]:
        """Log-LET linear interpolation of the digitized series."""
        x = np.asarray(let_keV_um, dtype=np.float64)
        out: npt.NDArray[np.float64] = np.interp(np.log(x), np.log(self.let), self.values)
        return out

    def covers(self, let_keV_um: npt.NDArray[np.float64] | float) -> npt.NDArray[np.bool_]:
        """Where ``let_keV_um`` lies inside this series' digitized LET span.

        Needed because the helium model curve is overdrawn at high LET (see the
        module docstring), so it does not run the full width of its figure.
        """
        x = np.asarray(let_keV_um, dtype=np.float64)
        inside: npt.NDArray[np.bool_] = (x >= self.let[0]) & (x <= self.let[-1])
        return inside


def load_oer_curves() -> tuple[OERCurve, ...]:
    """Every digitized series: 6 from fig. 5, 6 from fig. 6 and 18 from fig. 7."""
    data_dir = resources.files(__package__).joinpath(_DATA_DIR)
    curves: list[OERCurve] = []
    for figure, (cell_line, page) in FIGURES.items():
        for pO2, species_here in CONTENTS[figure].items():
            for species in species_here:
                for series in ("exp", "osmk"):
                    name = f"{_PREFIX}_{figure}_{species}_{series}_{_po2_tag(pO2)}.csv"
                    text = data_dir.joinpath(name).read_text(encoding="utf-8")
                    rows = list(csv.DictReader(_data_lines(text)))
                    rows.sort(key=lambda r: float(r["let_keV_um"]))
                    curves.append(
                        OERCurve(
                            figure=figure,
                            cell_line=cell_line,
                            species=species,
                            series=series,
                            let_keV_um=tuple(float(r["let_keV_um"]) for r in rows),
                            oer=tuple(float(r["oer"]) for r in rows),
                            oer_upper=tuple(float(r["oer_upper"]) for r in rows)
                            if rows and "oer_upper" in rows[0]
                            else (),
                            source_page=page,
                            pO2_mmHg=pO2,
                        )
                    )
    return tuple(curves)


def get_oer_curve(cell_line: str, species: str, series: str, pO2_mmHg: float = 0.0) -> OERCurve:
    """The series for one ``cell_line``, ``species``, ``series`` and hypoxic ``pO2``.

    ``series`` is ``"exp"`` for the measured points or ``"osmk"`` for the paper's
    own model curve. ``pO2_mmHg`` only varies on fig. 7 (CHO); HSG and V79 are
    anoxic only.
    """
    for curve in load_oer_curves():
        if (curve.cell_line, curve.species, curve.series, curve.pO2_mmHg) == (
            cell_line,
            species,
            series,
            pO2_mmHg,
        ):
            return curve
    raise KeyError(
        f"no OER curve for cell_line={cell_line!r} species={species!r} "
        f"series={series!r} pO2_mmHg={pO2_mmHg!r}; cell lines "
        f"{sorted({c for c, _ in FIGURES.values()})}, series 'exp' or 'osmk', and "
        f"per figure {CONTENTS}"
    )


def oer_curves_for(cell_line: str, series: str, pO2_mmHg: float = 0.0) -> tuple[OERCurve, ...]:
    """Every species' series of one kind for one cell line at one hypoxic ``pO2``."""
    return tuple(
        c
        for c in load_oer_curves()
        if c.cell_line == cell_line and c.series == series and c.pO2_mmHg == pO2_mmHg
    )


def pressures(cell_line: str) -> tuple[float, ...]:
    """The hypoxic pO2 values [mmHg] this cell line's figure carries."""
    for figure, (cell, _page) in FIGURES.items():
        if cell == cell_line:
            return tuple(sorted(CONTENTS[figure]))
    raise KeyError(f"no figure for cell line {cell_line!r}")


def reconstruction_lets(
    curve: OERCurve,
    stopping_power: StoppingPower,
    n: int = 18,
    energy_bounds_MeV_u: tuple[float, float] = (0.1, 1000.0),
) -> npt.NDArray[np.float64]:
    """Log-spaced LETs [keV/um] at which ``curve`` can fairly be compared to a model.

    The intersection of three separate limits, all of which matter:

    * the **fitted** LET range for this cell line,
      :data:`FITTED_LET_RANGE_KEV_UM` -- outside it the model extrapolates and
      the two OSMK formulations diverge sharply;
    * the LET span actually **digitized** for this series, which for helium stops
      early because its curve is overdrawn (see the module docstring);
    * the LET this ion can physically **attain**, which bounds it from *both*
      sides. The figures draw every species' curve clear across the panel, but a
      helium ion tops out near its Bragg peak -- ~178 keV/um with
      :class:`~pyradbio.track.stopping_power.SimpleBetheStoppingPower` -- so the
      right-hand end of the drawn helium curve corresponds to no helium beam at
      all; and symmetrically a neon ion cannot get below ~22 keV/um however fast
      it is made to go. ``LET -> energy`` inversion is undefined outside that
      window in either direction.

    Uses the same ``stopping_power`` the caller will invert with, so the cut is
    consistent with what :func:`~pyradbio.track.stopping_power.energy_for_let_MeV_u`
    will accept.
    """
    lo_e, hi_e = energy_bounds_MeV_u
    grid = np.geomspace(lo_e, hi_e, 400)
    lets = [stopping_power.let_keV_um(Ion(curve.Z, curve.A, float(e))) for e in grid]
    peak_let, floor_let = max(lets), min(lets)
    fitted = FITTED_LET_RANGE_KEV_UM[curve.cell_line]
    lo = max(fitted[0], curve.let[0], 1.02 * floor_let)
    hi = min(fitted[1], curve.let[-1], 0.98 * peak_let)
    if not lo < hi:
        raise ValueError(
            f"no comparable LET range for {curve.cell_line} {curve.species} "
            f"{curve.series} at {curve.pO2_mmHg} mmHg: fitted {fitted}, digitized "
            f"({curve.let[0]:.1f}, {curve.let[-1]:.1f}), attainable "
            f"({floor_let:.1f}, {peak_let:.1f})"
        )
    out: npt.NDArray[np.float64] = np.geomspace(lo, hi, n)
    return out


def measured_points(cell_line: str, pO2_mmHg: float = 0.0) -> dict[str, OERCurve]:
    """``{species: measured OER points}`` for one cell line at one hypoxic ``pO2``."""
    return {c.species: c for c in oer_curves_for(cell_line, "exp", pO2_mmHg)}


def model_curves(cell_line: str, pO2_mmHg: float = 0.0) -> dict[str, OERCurve]:
    """``{species: the paper's own OSMK curve}`` for one cell line and ``pO2``.

    These are **OSMK 2021** curves: figs. 5-7 predate the 2023 revision, so they
    are ground truth for :class:`~pyradbio.models.mkm.osmk.OSMK2021` only. The
    2023 formulation is a separate refit and is not expected to lie on them.
    """
    return {c.species: c for c in oer_curves_for(cell_line, "osmk", pO2_mmHg)}


# ---- reconstruction helpers ---------------------------------------------------


def model_oer(
    model: OSMK2021 | OSMK2023,
    curve: OERCurve,
    lets: npt.ArrayLike,
    stopping_power: StoppingPower,
    *,
    unreachable: Literal["nan", "raise"] = "nan",
) -> npt.NDArray[np.float64]:
    """``model``'s OER at each of ``lets`` [keV/um], for ``curve``'s ion and pO2.

    Each LET is inverted for a kinetic energy with
    :func:`~pyradbio.track.stopping_power.energy_for_let_MeV_u` (its default
    high-energy branch) on ``stopping_power``, and the OER is
    :meth:`~pyradbio.models.mkm.osmk.OSMK2021.oer_at_survival` at
    ``curve.pO2_mmHg`` against the 160 mmHg default reference, at 10% survival.

    Parameters
    ----------
    model:
        The OSMK to evaluate (2021 or 2023 formulation).
    curve:
        The series whose ion (``Z``, ``A``) and hypoxic ``pO2_mmHg`` to use.
    lets:
        Dose-averaged LETs [keV/um]; :func:`reconstruction_lets` gives the range
        that can fairly be compared.
    stopping_power:
        The LET source the inversion uses -- the caller's choice, *not* the
        model's own ``stopping_power`` field.
    unreachable:
        What a LET the ion cannot attain (the inversion raises ``ValueError``)
        gives: ``"nan"`` (the default), or ``"raise"`` to propagate the error.
        Only the inversion is guarded; an error inside the model propagates.

    Returns
    -------
    numpy.ndarray
        One OER per LET.
    """
    out: list[float] = []
    for let in np.asarray(lets, dtype=np.float64):
        try:
            energy = energy_for_let_MeV_u(stopping_power, curve.Z, curve.A, float(let))
        except ValueError:  # above the ion's Bragg-peak LET, or below its floor
            if unreachable == "raise":
                raise
            out.append(np.nan)
            continue
        out.append(model.oer_at_survival(Ion(curve.Z, curve.A, energy), pO2_mmHg=curve.pO2_mmHg))
    return np.asarray(out, dtype=float)


def oer_ratios(
    model: OSMK2021 | OSMK2023,
    cell_line: str,
    stopping_power: StoppingPower,
    n: int = 18,
    *,
    unreachable: Literal["nan", "raise"] = "nan",
) -> npt.NDArray[np.float64]:
    """``ours / paper`` OER ratios pooled over one figure.

    Every hypoxic pressure of ``cell_line``'s figure (:func:`pressures`), every
    species drawn there (:func:`model_curves`, the paper's own OSMK-2021 curves),
    each sampled at ``n`` LETs from :func:`reconstruction_lets`, in that order.
    ``unreachable`` is passed to :func:`model_oer`. Under ``"nan"`` every NaN
    ratio is dropped from the pool; under ``"raise"`` nothing is dropped, so a
    NaN from anywhere else (the model, the paper curve) stays in the pool and
    turns its median/RMS into NaN instead of vanishing silently.
    """
    out: list[float] = []
    for po2 in pressures(cell_line):
        for curve in model_curves(cell_line, po2).values():
            lets = reconstruction_lets(curve, stopping_power, n)
            ours = model_oer(model, curve, lets, stopping_power, unreachable=unreachable)
            r = ours / curve.interp(lets)
            out += list(r[~np.isnan(r)] if unreachable == "nan" else r)
    return np.array(out)


@dataclass(frozen=True, slots=True)
class _ZDomainScaledOSMK2021(OSMK2021):
    """OSMK 2021 whose *uncorrected* ``z_d,D`` is multiplied by ``z_domain_scale``.

    Built only through :func:`with_scaled_z_domain`. ``z*_d,D`` and ``z_n,D`` are
    untouched, and so is :meth:`specific_energies` itself -- only the
    ``specific_energies_at`` seam the OSMK algebra reads is rescaled -- so the
    microdosimetry cache serves it the same entries as the unscaled model.
    """

    z_domain_scale: float = 1.0

    def specific_energies_at(self, ion: Ion, pO2_mmHg: float) -> SiteSpecificEnergies:
        # Explicit base call: zero-argument super() breaks in a slots dataclass.
        z = OSMK2021.specific_energies_at(self, ion, pO2_mmHg)
        return SiteSpecificEnergies(
            z_1D_Gy=z.z_1D_Gy * self.z_domain_scale,
            z_star_1D_Gy=z.z_star_1D_Gy,
            z_nucleus_Gy=z.z_nucleus_Gy,
        )


def with_scaled_z_domain(model: OSMK2021, factor: float) -> OSMK2021:
    """A copy of ``model`` with its uncorrected domain ``z_d,D`` scaled by ``factor``.

    The rescaling diagnostic of the reconstruction. Eq. (14) reads ``z_d,D`` only
    through ``(z_d,D/z_R)^gamma``, so "our ``z_d,D`` is x% low" and "the published
    ``z_R`` is x% high" are the same statement from inside figs. 5-7, and scanning
    ``factor`` is the whole diagnostic. Only ``z_d,D`` moves -- that is what a bias
    in our microdosimetry would move. ``z*_d,D`` (which drives ``alpha``) and
    ``z_n,D`` are left alone, and the optional ``z_F`` moments are not carried.

    Every ``init`` field of :class:`~pyradbio.models.mkm.osmk.OSMK2021` is copied
    from ``model`` onto a private subclass, so the override of a user subclass of
    ``model`` is not kept. Scaling an already scaled model composes (the factors
    multiply), and ``factor = 1`` reproduces ``model``'s results exactly. The
    returned model is an ordinary frozen value: unlike a class-level factor,
    building a second one never moves the first.

    Raises
    ------
    TypeError
        If ``model`` is not an :class:`~pyradbio.models.mkm.osmk.OSMK2021` (the
        2023 formulation rescores ``z_d,D`` at a pO2-dependent site, so a single
        scale has no meaning there).
    """
    if not isinstance(model, OSMK2021):
        raise TypeError(
            f"with_scaled_z_domain needs an OSMK2021, got {type(model).__name__}: "
            "OSMK2023 rescores z_d,D at a pO2-dependent site"
        )
    values = {f.name: getattr(model, f.name) for f in fields(OSMK2021) if f.init}
    if isinstance(model, _ZDomainScaledOSMK2021):
        factor = factor * model.z_domain_scale
    return _ZDomainScaledOSMK2021(**values, z_domain_scale=factor)
