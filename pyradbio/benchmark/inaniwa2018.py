"""Digitized benchmark curves from Inaniwa & Kanematsu 2018 (SMK, Fig. 4).

Source paper (the SMK treatment-planning adaptation):

Inaniwa, T. & Kanematsu, N. 2018, "Adaptation of stochastic microdosimetric
kinetic model for charged-particle therapy treatment planning", Phys. Med. Biol.
63(9):095011, doi:10.1088/1361-6560/aabede.

Fig. 4 (p. 11) is the paper's **microdosimetry** figure: the three dose-averaged
specific energies per event that drive the whole SMK, for mono-energetic ions
with ``Z_p = 1-10`` against kinetic energy ``T``:

* panel (a) ``z_d,D``      -- domain, uncorrected      (paper eq. 11)
* panel (b) ``z*_d,D``     -- domain, saturation-corrected (eq. 12)
* panel (c) ``z_n,D``      -- cell nucleus             (eq. 7)

All three panels and all ten ion species are digitized here -- complete coverage
of Fig. 4, 30 series.

Why this figure and not the survival curves
-------------------------------------------
Figs. 2-3 (measured vs modelled survival) are the paper's headline, but they mix
track structure, geometry, and biology. Fig. 4 is the *intermediate* quantity: it
isolates our radial-dose + site-geometry pipeline from the effect model entirely.
Since the paper's table-1 parameters are pinned exactly (see
:func:`~pyradbio.models.mkm.spectrum_models.StochasticMKM.hsg_inaniwa2018`), any
disagreement here is unambiguously *our* microdosimetry -- which makes a later
survival-curve reconstruction diagnosable instead of merely suggestive.

Digitization method
-------------------
The source PDF draws Fig. 4 as **vector paths**, so this is not a raster trace:
the curve polylines were read directly out of the PDF content stream with PyMuPDF
(``page.get_drawings()``), giving exact figure coordinates. Species are separated
by stroke colour, which the figure assigns in the legend order
``Z_p = 1..10 -> black, red, green, blue, cyan, magenta, yellow, olive, navy,
purple``. Axis calibration comes from the panel frames and major tick marks, also
read as vector geometry (the axis *labels* are outlined glyphs, not text, so they
cannot be read as strings):

* x: log, one decade = 0.270427 of the frame width, ``T = 1 MeV/u`` at frac
  0.1888 -- the four decade ticks are consistent to 3e-4 of the width;
* y: linear per panel, full scale 1200 / 52.49 / 1.4496 Gy at the frame top.

Two traps, both of which silently corrupt a naive extraction:

* the **legend swatches** carry the same ten stroke colours as the curves; they
  are excluded by path length (a swatch is a single 20-unit horizontal segment,
  a curve is a ~3000-item polyline spanning the full frame);
* the ``Z_p = 8`` curve in panel (c) is stroked at a **different line width**
  (0.161) from every other curve in the figure (0.484), so filtering on width
  alone drops exactly one of the thirty series.

The digitization self-validates: the paper states ``z_n,D`` runs "about three
orders of magnitude" below ``z_d,D``, and while the track is contained in both
sites ``z ~ 1/r^2`` fixes that ratio at the geometric
``(R_n/r_d)^2 = (8.1/0.28)^2 = 837``. The extracted curves give 834 at 0.3 MeV/u
for every ion -- an independent check on both the colour assignment and the two
panels' separately calibrated y-axes. (Above ~1 MeV/u the ratio *falls*, reaching
~600 by 20 MeV/u, because ``r_max`` outgrows the 0.28 um domain while the 8.1 um
nucleus still collects the penumbra. That is physics, not a trace artifact: our
own model reproduces the same fall-off. The paper's "three orders of magnitude"
is a contained-track statement.)

Resolution floor
----------------
Panels (a) and (c) are drawn on linear axes sized for the *heaviest* ions, so the
light-ion curves lie within a hair of the axis. Below roughly 2% of a panel's
full scale the digitized value is not meaningfully distinguishable from zero --
e.g. protons in panel (a) sit under 1 Gy against a 1200 Gy scale above ~50 MeV/u.
Use :func:`resolvable` (or :attr:`Fig4Curve.resolvable_mask`) before comparing;
this is a limitation of the published figure, not of the extraction. Panel (b) is
well resolved across its whole range and is the panel worth trusting hardest.

Curves are stored thinned onto a 140-point log-``T`` grid per species; the raw
extraction carried ~3000 points per curve, far beyond the figure's own precision.
"""

from __future__ import annotations

import csv
from collections import defaultdict
from collections.abc import Callable
from dataclasses import dataclass
from functools import lru_cache
from importlib import resources

import numpy as np
import numpy.typing as npt

from pyradbio.core.elements import default_mass_number
from pyradbio.core.radiation import Ion
from pyradbio.models.mkm.spectrum_models import ModifiedMKM, StochasticMKM
from pyradbio.track.stopping_power import StoppingPower, energy_for_let_MeV_u

INANIWA2018_DOI = "10.1088/1361-6560/aabede"
_DATA_DIR = "data"
_CURVE_PREFIX = "inaniwa2018_fig4"
_SOURCE_PAGE = 11

#: Panel -> (quantity name, y-axis full scale at the frame top [Gy]).
PANELS: dict[str, tuple[str, float]] = {
    "a": ("z_domain", 1200.0),
    "b": ("z_star_domain", 50.0 / 0.95262),
    "c": ("z_nucleus", 1.25 / 0.86227),
}

#: Fraction of a panel's full scale below which the digitized value is unreliable.
RESOLUTION_FLOOR = 0.02

#: Parameters Fig. 4 was computed with (paper sec 4.2.1; table 1, HSG SMK column).
FIG4_PARAMETERS = {"domain_radius_um": 0.28, "nucleus_radius_um": 8.1, "z0_Gy": 66.0}


@dataclass(frozen=True, slots=True)
class Fig4Curve:
    """One digitized Fig. 4 species curve: a specific energy against ``T``."""

    panel: str  # "a" | "b" | "c"
    quantity: str  # "z_domain" | "z_star_domain" | "z_nucleus"
    Z_p: int
    T_MeV_u: tuple[float, ...]
    z_Gy: tuple[float, ...]
    full_scale_Gy: float
    source_page: int = _SOURCE_PAGE
    doi: str = INANIWA2018_DOI

    @property
    def T(self) -> npt.NDArray[np.float64]:
        """Kinetic energies [MeV/u] as an array."""
        return np.asarray(self.T_MeV_u, dtype=np.float64)

    @property
    def z(self) -> npt.NDArray[np.float64]:
        """Specific energies [Gy] as an array."""
        return np.asarray(self.z_Gy, dtype=np.float64)

    @property
    def peak(self) -> tuple[float, float]:
        """``(T, z)`` at the curve maximum -- the paper's "inflection point"."""
        k = int(np.argmax(self.z))
        return float(self.T[k]), float(self.z[k])

    def interp(self, T_MeV_u: npt.NDArray[np.float64] | float) -> npt.NDArray[np.float64]:
        """Log-``T`` linear interpolation of the digitized curve [Gy]."""
        t = np.asarray(T_MeV_u, dtype=np.float64)
        out: npt.NDArray[np.float64] = np.interp(np.log(t), np.log(self.T), self.z)
        return out

    def resolvable_mask(self, T_MeV_u: npt.NDArray[np.float64] | float) -> npt.NDArray[np.bool_]:
        """Where the digitized curve sits above this panel's resolution floor."""
        return resolvable(self.interp(T_MeV_u), self.panel)


def _data_lines(text: str) -> list[str]:
    """CSV lines without the ``#`` citation header the data files carry."""
    return [line for line in text.splitlines() if not line.startswith("#")]


def resolvable(z_Gy: npt.NDArray[np.float64] | float, panel: str) -> npt.NDArray[np.bool_]:
    """Whether ``z_Gy`` is far enough off a panel's axis to be trusted.

    Panels (a) and (c) are scaled for the heaviest ions, so light-ion curves ride
    the axis; see the module docstring.
    """
    if panel not in PANELS:
        raise ValueError(f"panel must be one of {sorted(PANELS)}, got {panel!r}")
    floor = RESOLUTION_FLOOR * PANELS[panel][1]
    mask: npt.NDArray[np.bool_] = np.asarray(z_Gy, dtype=np.float64) >= floor
    return mask


@lru_cache(maxsize=1)
def load_fig4_curves() -> tuple[Fig4Curve, ...]:
    """Return all 30 digitized Fig. 4 curves (3 quantities x ``Z_p`` 1-10)."""
    data_dir = resources.files(__package__).joinpath(_DATA_DIR)
    curves: list[Fig4Curve] = []
    for panel, (quantity, full_scale) in PANELS.items():
        text = data_dir.joinpath(f"{_CURVE_PREFIX}{panel}_{quantity}.csv").read_text(
            encoding="utf-8"
        )
        by_z: dict[int, list[tuple[float, float]]] = defaultdict(list)
        for row in csv.DictReader(_data_lines(text)):
            by_z[int(row["Z_p"])].append((float(row["T_MeV_u"]), float(row["z_Gy"])))
        for z_p, points in sorted(by_z.items()):
            points.sort()
            curves.append(
                Fig4Curve(
                    panel=panel,
                    quantity=quantity,
                    Z_p=z_p,
                    T_MeV_u=tuple(t for t, _ in points),
                    z_Gy=tuple(v for _, v in points),
                    full_scale_Gy=full_scale,
                )
            )
    return tuple(curves)


def get_fig4_curve(quantity: str, Z_p: int) -> Fig4Curve:
    """The Fig. 4 curve for one ``quantity`` and atomic number.

    ``quantity`` is ``"z_domain"``, ``"z_star_domain"`` or ``"z_nucleus"``.
    """
    matches = [c for c in load_fig4_curves() if c.quantity == quantity and c.Z_p == Z_p]
    if not matches:
        known = sorted({c.quantity for c in load_fig4_curves()})
        raise KeyError(
            f"no Fig. 4 curve for quantity={quantity!r} Z_p={Z_p!r}; quantities: {known}"
        )
    return matches[0]


#: Fig. 4 quantity -> the :class:`~pyradbio.models.mkm.spectrum_models.SiteSpecificEnergies`
#: field our model reports it as.
FIG4_FIELDS: dict[str, str] = {
    "z_domain": "z_1D_Gy",
    "z_star_domain": "z_star_1D_Gy",
    "z_nucleus": "z_nucleus_Gy",
}


def fig4_model_values(
    model: StochasticMKM, Z_p: int, T_MeV_u: npt.NDArray[np.float64]
) -> dict[str, npt.NDArray[np.float64]]:
    """Our three Fig. 4 quantities for ion ``Z_p`` at each kinetic energy [Gy].

    One :meth:`~pyradbio.models.mkm.spectrum_models.StochasticMKM.specific_energies`
    call per energy, for the ion ``Ion(Z_p, default_mass_number(Z_p), T)``; the
    keys are those of :data:`FIG4_FIELDS`. Pass the model the figure was computed
    with, ``StochasticMKM.hsg_inaniwa2018()`` (:data:`FIG4_PARAMETERS`).
    """
    A = default_mass_number(Z_p)
    se = [model.specific_energies(Ion(Z_p, A, float(t))) for t in T_MeV_u]
    return {
        quantity: np.array([getattr(s, field) for s in se])
        for quantity, field in FIG4_FIELDS.items()
    }


def fig4_comparison(
    model: StochasticMKM, quantity: str, Z_p: int, T_MeV_u: npt.NDArray[np.float64]
) -> tuple[npt.NDArray[np.float64], npt.NDArray[np.float64], npt.NDArray[np.bool_]]:
    """``(ours, paper, mask)`` for one Fig. 4 series on the grid ``T_MeV_u``.

    ``paper`` is the digitized curve interpolated onto ``T_MeV_u``
    (:meth:`Fig4Curve.interp`); ``mask`` keeps the points where it is
    :func:`resolvable` and ``ours > 0``. The shape
    :func:`~pyradbio.benchmark.metrics.ratio_summary` takes
    (``ratio_summary(ours, paper, mask=mask)``).
    """
    curve = get_fig4_curve(quantity, Z_p)
    paper = curve.interp(T_MeV_u)
    ours = fig4_model_values(model, Z_p, T_MeV_u)[quantity]
    mask = resolvable(paper, curve.panel) & (ours > 0.0)
    return ours, paper, mask


def fig4_ratios(
    model: StochasticMKM, quantity: str, Z_p: int, T_MeV_u: npt.NDArray[np.float64]
) -> npt.NDArray[np.float64]:
    """``ours / paper`` for one Fig. 4 series, where the paper's curve is resolvable.

    The masked ratios of :func:`fig4_comparison`, in energy order. Pooling these
    over ``Z_p = 1-10`` is the reconstruction score of the module docstring.
    """
    ours, paper, mask = fig4_comparison(model, quantity, Z_p, T_MeV_u)
    ratios: npt.NDArray[np.float64] = ours[mask] / paper[mask]
    return ratios


#: The factor by which our uncorrected dose-mean domain specific energy ``z_d,D``
#: runs *low* against Fig. 4(a): ``1 / 0.933 ~ 1.07``, where 0.933 is the median
#: of :func:`fig4_ratios` pooled over ``Z_p = 1-10`` (134 resolvable points on the
#: 24-point log grid 0.3-500 MeV/u) with ``SimpleBetheStoppingPower`` on the model
#: (docs/references/SMK_Inaniwa2018.md, "Fig. 4 reconstruction").
#: With the library-default stopping power on the model the same median is 0.965,
#: so the number is tied to that stopping power. A *diagnostic* only -- the OSMK
#: (Inaniwa 2021) reconstruction rescales ``z_d,D`` by it to ask how much of its
#: residual this one measured bias explains. Never apply it to a model by default.
Z_D_BIAS = 1.07


# ==========================================================================
# Figs. 2 (HSG) and 3 (V79): survival curves
# ==========================================================================
#: Atomic number per ion label used in the survival panels.
ION_Z = {"He": 2, "C": 6, "Ne": 10}

_SURVIVAL_FILE = "inaniwa2018_survival.csv"


@dataclass(frozen=True, slots=True)
class SurvivalCurve:
    """One digitized survival series from Fig. 2 (HSG) or Fig. 3 (V79).

    ``series`` is ``"exp"`` (the Furusawa et al. 2000 measurements the paper
    redraws as dots), ``"smk"`` (the paper's modified-SMK solid curve) or ``"mk"``
    (its MK dashed curve).
    """

    cell_line: str  # "HSG" | "V79"
    ion: str  # "He" | "C" | "Ne"
    A: int
    let_keV_um: float
    panel: str  # "a".."i"
    series: str  # "exp" | "smk" | "mk"
    dose_Gy: tuple[float, ...]
    survival: tuple[float, ...]
    doi: str = INANIWA2018_DOI

    @property
    def Z(self) -> int:
        return ION_Z[self.ion]

    @property
    def dose(self) -> npt.NDArray[np.float64]:
        return np.asarray(self.dose_Gy, dtype=np.float64)

    @property
    def s(self) -> npt.NDArray[np.float64]:
        return np.asarray(self.survival, dtype=np.float64)

    def dose_at_survival(self, survival_fraction: float = 0.1) -> float:
        """Dose giving ``survival_fraction`` [Gy], by interpolation in ``-log10 S``.

        ``D_10`` is the standard scalar summary of a survival curve and the right
        thing to compare across model and experiment: it is insensitive to the
        digitization noise that a pointwise ratio would amplify at low ``S``.
        """
        return dose_at_survival(self.dose, self.s, survival_fraction)


@lru_cache(maxsize=1)
def load_survival_curves() -> tuple[SurvivalCurve, ...]:
    """Return all 54 digitized survival series (2 cell lines x 9 panels x 3 series)."""
    text = (
        resources.files(__package__)
        .joinpath(_DATA_DIR)
        .joinpath(_SURVIVAL_FILE)
        .read_text(encoding="utf-8")
    )
    grouped: dict[tuple[str, str, str], list[tuple[float, float]]] = defaultdict(list)
    meta: dict[tuple[str, str, str], tuple[str, int, float]] = {}
    for row in csv.DictReader(_data_lines(text)):
        key = (row["cell_line"], row["panel"], row["series"])
        grouped[key].append((float(row["dose_Gy"]), float(row["survival"])))
        meta[key] = (row["ion"], int(row["A"]), float(row["let_keV_um"]))
    curves = []
    for (cell_line, panel, series), points in sorted(grouped.items()):
        points.sort()
        ion, A, let = meta[(cell_line, panel, series)]
        curves.append(
            SurvivalCurve(
                cell_line=cell_line,
                ion=ion,
                A=A,
                let_keV_um=let,
                panel=panel,
                series=series,
                dose_Gy=tuple(d for d, _ in points),
                survival=tuple(s for _, s in points),
            )
        )
    return tuple(curves)


def get_survival_curve(cell_line: str, panel: str, series: str) -> SurvivalCurve:
    """One survival series, e.g. ``("HSG", "f", "smk")``."""
    for c in load_survival_curves():
        if (c.cell_line, c.panel, c.series) == (cell_line, panel, series):
            return c
    raise KeyError(f"no survival curve for {cell_line!r} panel {panel!r} series {series!r}")


def survival_panels(cell_line: str) -> tuple[SurvivalCurve, ...]:
    """The nine ``smk`` curves of one cell line, in panel order -- the panel index."""
    return tuple(
        c for c in load_survival_curves() if c.cell_line == cell_line and c.series == "smk"
    )


#: Cell line -> ``(SMK factory, MK factory)``: the two table-1 columns behind
#: Fig. 2 (HSG) and Fig. 3 (V79). Both factories take model-field overrides, e.g.
#: ``stopping_power=...``.
CELL_MODELS: dict[str, tuple[Callable[..., StochasticMKM], Callable[..., ModifiedMKM]]] = {
    "HSG": (StochasticMKM.hsg_inaniwa2018, ModifiedMKM.hsg_mk_inaniwa2018),
    "V79": (StochasticMKM.v79_inaniwa2018, ModifiedMKM.v79_mk_inaniwa2018),
}


@dataclass(frozen=True, slots=True, eq=False)
class PanelReconstruction:
    """Our survival curves for one Fig. 2/3 panel, from its quoted LET alone.

    ``smk`` / ``mk`` are ``None`` when that model was not requested.
    """

    ion: Ion
    smk: npt.NDArray[np.float64] | None
    mk: npt.NDArray[np.float64] | None


def reconstruct_panel(
    curve: SurvivalCurve,
    dose_Gy: npt.NDArray[np.float64],
    *,
    stopping_power: StoppingPower,
    smk: StochasticMKM | None = None,
    mk: ModifiedMKM | None = None,
) -> PanelReconstruction:
    """Evaluate the paper's SMK and/or MK model for one survival panel.

    Nothing is fitted. doi:10.1088/1361-6560/aabede sec 3.2: "we assumed a
    monoenergetic spectrum of 3He-, 12C- and 20Ne-ion beams with reported LETs",
    so the panel's quoted LET is inverted with ``stopping_power``
    (:func:`~pyradbio.track.stopping_power.energy_for_let_MeV_u`, default
    high-energy root -- the physical one for the degraded beams of the experiments).

    * SMK (``smk``): the three specific energies of that ion, then eqs. (15),
      (16), (24) via ``survival_from_specific_energies``.
    * MK (``mk``): ``alpha_beta_from_z_star_1D`` of its own ``z*_1D``, then the
      plain LQ ``exp(-(alpha D + beta D^2))``.

    ``stopping_power`` drives only the LET inversion; each model scores its track
    with its own stopping power, so pass models built with the same one when the
    two must agree.
    """
    energy = energy_for_let_MeV_u(stopping_power, curve.Z, curve.A, curve.let_keV_um)
    ion = Ion(curve.Z, curve.A, energy)
    s_smk = None
    if smk is not None:
        z = smk.specific_energies(ion)
        s_smk = smk.survival_from_specific_energies(
            dose_Gy, z_star_1D_Gy=z.z_star_1D_Gy, z_1D_Gy=z.z_1D_Gy, z_nucleus_Gy=z.z_nucleus_Gy
        )
    s_mk = None
    if mk is not None:
        lq = mk.alpha_beta_from_z_star_1D(mk.specific_energies(ion).z_star_1D_Gy)
        s_mk = np.exp(-(lq.alpha * dose_Gy + lq.beta * dose_Gy**2))
    return PanelReconstruction(ion=ion, smk=s_smk, mk=s_mk)


def dose_at_survival(
    dose_Gy: npt.NDArray[np.float64],
    survival: npt.NDArray[np.float64],
    survival_fraction: float = 0.1,
) -> float:
    """Dose [Gy] at which a tabulated curve reaches ``survival_fraction``.

    Linear interpolation in ``-log10 S`` (survival clipped to ``[1e-12, 1]``), the
    rule :meth:`SurvivalCurve.dose_at_survival` applies to the digitized curves,
    so our curves and the paper's are summarised identically. ``-log10 S`` must
    rise along ``dose_Gy`` (``np.interp`` needs increasing abscissae).
    """
    target = -np.log10(survival_fraction)
    y = -np.log10(np.clip(survival, 1.0e-12, 1.0))
    return float(np.interp(target, y, dose_Gy))


def delta2_per_point(cell_line: str, series: str, panels: str) -> float:
    """The paper's eq.-(25) goodness metric on the digitized points.

    ``Delta^2/n = mean[log10(S_exp) - log10(S_cal)]^2`` -- exactly the quantity
    table 1 reports, evaluated against the curves the figures draw (``series``
    ``"smk"`` or ``"mk"``) rather than against our own model, so it scores *the
    paper's* two models. ``panels`` is a string of panel letters, e.g. ``"f"`` or
    ``"abcdefghi"``; the points of all of them are pooled into one mean. Table 1
    fitted ~300 points per cell line, the figures draw ~50, so only the ordering
    of the two models is comparable with table 1, not the absolute values.
    """
    d2: list[float] = []
    for panel in panels:
        exp = get_survival_curve(cell_line, panel, "exp")
        model = get_survival_curve(cell_line, panel, series)
        predicted = np.interp(exp.dose, model.dose, np.log10(model.s))
        d2 += list((np.log10(exp.s) - predicted) ** 2)
    return float(np.mean(d2))
