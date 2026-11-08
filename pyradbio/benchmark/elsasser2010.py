"""Digitized benchmark curves from Elsasser et al. 2010 (LEM IV, Fig. 4a/4b).

Source paper (the primary LEM IV / DSB-cluster paper):

Elsasser, T., Weyrather, W.K., Friedrich, T., Durante, M., Iancu, G., Kraemer,
M., Kragl, G., Brons, S., Winter, M., Weber, K.J. & Scholz, M. 2010,
"Quantification of the Relative Biological Effectiveness for Ion Beam
Radiotherapy: Direct Experimental Comparison of Proton and Carbon Ion Beams
and a Novel Approach for Treatment Planning", Int. J. Radiat. Oncol. Biol.
Phys. 78(4):1177-1183, doi:10.1016/j.ijrobp.2010.05.014.

Fig. 4 (p. 1181) is the paper's monoenergetic RBE(LET) figure -- the only one
giving clean single-ion-species curves rather than a depth-dose/SOBP
comparison, and the direct analogue of ``LEM4``'s
``alpha_beta(ion).alpha / photon.alpha_x`` output. Panel (a) is V79 cells,
protons + helium (Belli et al. 1998, ``RBE_alpha`` only); panel (b) is HSG
cells, carbon + helium (Furusawa et al. 2000, ``RBE_alpha`` and ``RBE_10``).
Both panels, all ion species, model lines *and* experimental data are
digitized here -- complete coverage of Fig. 4:

**Fig. 4a (V79, protons + helium), RBE_alpha only:**

* **Model**: proton (solid line) and helium (dashed line) are both digitized
  in full. **LEM4's own proton
  comparison against this line is only reliable up to LET~25 keV/um**: above
  that the *paper's own* model line levels off near ~6.5 while pyRadBio's
  single-particle model keeps rising past 9 by LET 40 (at proton energies
  still nominally within ``SimpleBetheStoppingPower``'s ~0.5 MeV/u validity
  floor) -- a genuine deep-overkill model
  limitation, not a digitization gap; the golden test clips its comparisons
  to LET<=25 accordingly.
* **Experimental data**: proton (triangles, 11 points with error bars -- see
  below) and helium (squares, 5 points, LET 20-79 keV/um, no error bars in
  the source figure) are both digitized. The proton markers carry drawn error
  bars; ``y_err`` is the half-height of the drawn bar and the point sits on the
  marker centre.

**Fig. 4b (HSG, carbon + helium), RBE_alpha and RBE_10:**

* **Model and experimental data for both ions**, all four curves per ion (8
  series total). The panel draws *two* ion species (He, C) but only *one*
  legend entry each for "Model RBE_alpha"/"Model RBE_10" in the original --
  the two ions' RBE_alpha model lines are visually coincident below
  ~60 keV/um and only *diverge* above it (helium continues rising past
  RBE_alpha~10 by ~120 keV/um, while carbon turns over and peaks at ~5.8 near
  160-180 keV/um before declining -- a real difference in the two ions' track
  structure at matched LET, not an artist's shortcut), but both are digitized
  here as separate series.

Each curve lives in its own CSV under ``data/`` named
``elsasser2010_figure4<a|b>_<cell>_<particle>_<series>_<quantity>.csv`` with
the same four-line header convention as [`pyradbio.benchmark.elsasser2007`][]
(metadata keys, metadata values, x/y quantities, x/y units), then ``x,y``
data rows.

Digitization method: all 12 series are extracted from the vector paths of the
published PDF, with the axis calibration measured from the tick segments rather
than anchored by eye. The number of points per series is that of its CSV.

**The Fig. 4b y axis runs 0 to 11, not 0 to 10.** Both panels draw 12
unit-spaced left-axis ticks over 147 pt, majors every 2 labelled 0 2 4 6 8 10 --
and an UNLABELLED tick at 11 on the frame edge. Anchoring the top of the axis at
the labelled 10 would put every Fig. 4b series a factor 10/9 too high.

**Fig. 4b has no error bars.** The helium experimental points at neighbouring
LETs (18.5 and 18.6, 23.6 and 24.0, 32.4 and 33.0, two at 46.0, 53.8 and 54.0,
70.3 and 71.0) are pairs of DISTINCT Furusawa measurements, not error-bar ends,
and must not be merged. The panel draws 13 filled squares and 12 open squares;
all are present, and ``y_err`` is ``None``.

**Fig. 4a uses the same 0..11 axis**; its helium data and both model lines
reproduce the vectors to 0.1%. Its proton points *do* carry drawn
error bars: they sit on the drawn marker centres, with ``y_err`` the half-height
of the drawn bar (averaging each marker with a bar endpoint would put them ~7%
high). The carbon RBE_alpha data are the direct digitization of Fig. 4b's own
filled circles (``fig4b_hsg_rbe_vs_let``, ``carbon``, ``data``, ``rbe_alpha``),
not [`pyradbio.benchmark.elsasser2007`][] Fig. 11's ``alpha`` series divided
by ``alpha_x``.

See ``docs/references/LEM_IV_Elsasser2010.md``.
"""

from __future__ import annotations

import csv
from dataclasses import dataclass
from functools import lru_cache
from importlib import resources

from pyradbio.core.cell import CellTarget
from pyradbio.core.photon import PhotonParameters

ELSASSER2010_DOI = "10.1016/j.ijrobp.2010.05.014"

_DATA_DIR = "data"
_CURVE_PREFIX = "elsasser2010_figure"


@dataclass(frozen=True, slots=True)
class PaperCellInputs:
    """Cell/target inputs Fig. 4b used (LEM IV photon + geometry, p. 1181 caption).

    ``effective_size_factor = 1.0``: the paper quotes a nucleus radius to be
    used as-is, the same convention as
    [`pyradbio.benchmark.elsasser2007.PaperCellInputs`][] and
    [`pyradbio.benchmark.elsasser2008.SuzukiCell`][].
    ``CellTarget`` defaults to 1.0; the factor is still passed explicitly so the
    convention stays visible at the construction.
    """

    alpha_x: float  # Gy^-1
    beta_x: float  # Gy^-2
    D_t: float  # Gy
    nucleus_radius_um: float
    effective_size_factor: float = 1.0

    def photon(self) -> PhotonParameters:
        """Return the LEM IV photon reference ``(alpha_x, beta_x, D_t)``, no cluster.

        Elsasser et al. 2010 (doi:10.1016/j.ijrobp.2010.05.014), Fig. 4 caption,
        p. 1181. Equal to
        ``PhotonParameters(alpha_x=inp.alpha_x, beta_x=inp.beta_x, D_t=inp.D_t)``.
        """
        return PhotonParameters(alpha_x=self.alpha_x, beta_x=self.beta_x, D_t=self.D_t)

    def cell_target(self) -> CellTarget:
        """Return the nucleus geometry, passing ``effective_size_factor`` on explicitly.

        Elsasser et al. 2010 (doi:10.1016/j.ijrobp.2010.05.014), Fig. 4 caption.
        Equal to
        ``CellTarget(inp.nucleus_radius_um, effective_size_factor=inp.effective_size_factor)``.
        """
        return CellTarget(
            nucleus_radius_um=self.nucleus_radius_um,
            effective_size_factor=self.effective_size_factor,
        )


# Fig. 4a caption (p. 1181): "alpha_gamma = 0.129 Gy^-1, beta_gamma = 0.049
# Gy^-2, threshold dose D_t = 3 Gy, radius of cell nucleus r_N = 5 um" (V79,
# protons + helium, Belli et al. 1998).
# Fig. 4b caption (p. 1181): "alpha_gamma = 0.313 Gy^-1, beta_gamma = 0.0615
# Gy^-2, threshold dose D_t = 7.5 Gy, radius of cell nucleus r_N = 5 um" (HSG,
# carbon + helium, Furusawa et al. 2000).
FIGURE_INPUTS: dict[str, PaperCellInputs] = {
    "fig4a_v79": PaperCellInputs(0.129, 0.049, 3.0, 5.0),
    "fig4b_hsg": PaperCellInputs(0.313, 0.0615, 7.5, 5.0),
}


@dataclass(frozen=True, slots=True)
class Elsasser2010Curve:
    """One digitized reference curve (a single figure line or symbol set)."""

    figure: str
    cell_line: str
    particle: str
    series: str  # "calc" (paper model line) or "data" (experimental symbols)
    variant: str  # e.g. "model_rbe_alpha", "model_rbe_10", "rbe_10"
    x_quantity: str
    x_unit: str
    y_quantity: str
    y_unit: str
    x: tuple[float, ...]
    y: tuple[float, ...]
    source_page: int
    note: str
    doi: str = ELSASSER2010_DOI
    y_err: tuple[float, ...] | None = None  # symmetric 1-sigma, when the figure drew bars


def _data_files(prefix: str) -> list[tuple[str, str]]:
    """Return ``(filename, text)`` for every ``data/`` CSV starting with ``prefix``."""
    data_dir = resources.files(__package__).joinpath(_DATA_DIR)
    out: list[tuple[str, str]] = []
    for entry in data_dir.iterdir():
        if entry.name.startswith(prefix) and entry.name.endswith(".csv"):
            out.append((entry.name, entry.read_text(encoding="utf-8")))
    return out


@lru_cache(maxsize=1)
def load_curves() -> tuple[Elsasser2010Curve, ...]:
    """Return every digitized Elsasser et al. 2010 benchmark curve (one per file)."""
    curves: list[Elsasser2010Curve] = []
    for _name, text in _data_files(_CURVE_PREFIX):
        rows = list(csv.reader(text.splitlines()))
        meta = dict(zip(rows[0], rows[1], strict=True))
        x_quantity, y_quantity = rows[2][0], rows[2][1]
        x_unit, y_unit = rows[3][0], rows[3][1]
        # Data rows are ``x,y`` or, where the figure drew error bars, ``x,y,y_err``.
        data = [r for r in rows[4:] if r and r[0].strip()]
        has_err = any(len(r) > 2 and r[2].strip() for r in data)
        points = sorted((float(r[0]), float(r[1]), float(r[2]) if has_err else 0.0) for r in data)
        curves.append(
            Elsasser2010Curve(
                figure=meta["figure"],
                cell_line=meta["cell_line"],
                particle=meta["particle"],
                series=meta["series"],
                variant=meta["variant"],
                x_quantity=x_quantity,
                x_unit=x_unit,
                y_quantity=y_quantity,
                y_unit=y_unit,
                x=tuple(x for x, _, _ in points),
                y=tuple(y for _, y, _ in points),
                source_page=int(meta["source_page"]),
                note=meta["note"],
                doi=meta.get("doi", ELSASSER2010_DOI),
                y_err=tuple(e for _, _, e in points) if has_err else None,
            )
        )
    return tuple(sorted(curves, key=lambda c: (c.figure, c.cell_line, c.series, c.variant)))


def iter_curves(
    *,
    figure: str | None = None,
    cell_line: str | None = None,
    particle: str | None = None,
    series: str | None = None,
    variant: str | None = None,
) -> tuple[Elsasser2010Curve, ...]:
    """Filter digitized Elsasser et al. 2010 curves.

    ``series="calc"`` selects the paper's own model lines; ``series="data"``
    selects the experimental symbols redrawn in the figure.
    """
    curves = load_curves()
    return tuple(
        c
        for c in curves
        if (figure is None or c.figure == figure)
        and (cell_line is None or c.cell_line == cell_line)
        and (particle is None or c.particle == particle)
        and (series is None or c.series == series)
        and (variant is None or c.variant == variant)
    )


def get_curve(
    figure: str,
    *,
    cell_line: str | None = None,
    particle: str | None = None,
    series: str | None = None,
    variant: str | None = None,
) -> Elsasser2010Curve:
    """Return exactly one Elsasser et al. 2010 reference curve.

    Raises ``LookupError`` if the filters select zero or multiple curves.
    """
    matches = iter_curves(
        figure=figure, cell_line=cell_line, particle=particle, series=series, variant=variant
    )
    if len(matches) != 1:
        raise LookupError(f"expected exactly one Elsasser et al. 2010 curve, got {len(matches)}")
    return matches[0]


__all__ = [
    "ELSASSER2010_DOI",
    "FIGURE_INPUTS",
    "Elsasser2010Curve",
    "PaperCellInputs",
    "get_curve",
    "iter_curves",
    "load_curves",
]
