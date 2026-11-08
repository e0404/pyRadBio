"""Digitized benchmark curves from Friedrich et al. 2012 (LEM IV proceedings note).

Source paper (the proceedings follow-up to the primary LEM IV paper,
``docs/references/LEM_IV_Elsasser2010.md``):

Friedrich, T., Scholz, U., Elsasser, T., Durante, M. & Scholz, M. 2012,
"Calculation of the biological effects of ion beams based on the microscopic
spatial damage distribution pattern", Int. J. Radiat. Biol. 88(1-2):103-107,
doi:10.3109/09553002.2011.611213.

**Fig. 3 (p. 106)** is the paper's own quantitative RBE(LET) comparison: LEM
IV's single-particle model against Furusawa et al. (2000) experimental data,
V79 Chinese hamster cells, carbon ions, ``RBE_alpha`` and ``RBE_10``. It is
the direct point-by-point analogue of what
``tests.models.lem.test_lem4::test_golden_lem4_v79_carbon_rbe_alpha_shape``
only pins the *shape* of; this module lets the golden tests compare
point-by-point against the paper's own digitized curves, the same standard
the LEM I-III and Elsasser et al. 2010 reconstructions hold.

Fig. 3 caption (p. 106): "Input photon parameters for calculation:
alpha_gamma = 0.184 Gy^-1, beta_gamma = 0.02 Gy^-2, D_t = 17 Gy, Radius of the
cell nucleus R_Nucleus = 4.7 um" -- identical to the ``V79``/``V79_CELL``
fixture already used in ``test_lem4.py`` (independently transcribed from the
paper before this digitization existed).

All four Fig. 3 series (model/data x RBE_alpha/RBE_10) are hand-digitized in
full. Unlike
[`pyradbio.benchmark.elsasser2010`][]'s Fig. 4, the experimental series here
have **no error-bar-endpoint duplication**: LET values that are close together
(e.g. 40.04 and 40.57 keV/um) carry substantially different RBE (1.54 vs
3.37) -- real point-to-point scatter in the underlying Furusawa et al. (2000)
replicate measurements, not a digitization-tool artifact from tracing error
bars (checked systematically, as for the Elsasser 2010 error bars).

**Fig. 4 (p. 107)** compares the single-particle Kraemer-Scholz slope-matching
approximation against the full intertrack-cDSB Monte-Carlo simulation
(``LEM4(beta_model="intertrack_mc")``, see ``models/lem/lem4.py`` and
``tests/models/lem/test_lem4_intertrack.py``) for a "hypothetical" cell
(caption, p. 107: alpha_gamma=0.25 Gy^-1, beta_gamma=0.05 Gy^-2, D_t=9.5 Gy,
R_Nucleus=5 um), both ``alpha_I(LET)`` and ``beta_I(LET)``.

**Alpha-axis calibration.** The ``alpha_I`` panel of Fig. 4 has a very different
y range from its ``beta_I`` panel (roughly ``0.25-2`` vs ``0-0.05``); the alpha
traces are calibrated to ``[0, 1.5]``. Reading them off the ``[0, 0.06]`` scale
instead would put them a constant factor ``1.5/0.06 = 25.0`` too low across the
whole LET range (10-130 keV/um), the signature of an axis-calibration mistake
rather than digitization noise. With the correct scale both traces agree to
<0.03 RMS with pyRadBio's own single-particle ``alpha_I`` (see the golden test).

Each curve lives in its own CSV under ``data/``, named
``friedrich2012_figure3_v79_carbon_<series>_<quantity>.csv`` (Fig. 3) or
``friedrich2012_figure4_hypothetical_carbon_<series>_<quantity>.csv`` (Fig. 4),
with a four-line header (metadata keys, metadata values, x/y quantities, x/y
units), then ``x,y`` data rows -- the same convention as
[`pyradbio.benchmark.elsasser2010`][], minus the optional ``y_err`` column
(not needed here). Fig. 4's ``series`` field takes the values ``"approx"``
(the single-particle Kraemer-Scholz approximation) and ``"full_simulation"``
(the intertrack Monte-Carlo) -- a model-vs-model comparison, unlike Fig. 3's
``"calc"``/``"data"`` (model-vs-experiment); its ``variant`` field is
``"alpha"`` or ``"beta"``.
"""

from __future__ import annotations

import csv
import math
import os
from collections.abc import Iterable, Mapping, Sequence
from dataclasses import dataclass
from functools import lru_cache
from importlib import resources
from typing import TYPE_CHECKING, Any

import numpy as np
import numpy.typing as npt

from pyradbio.core.cell import CellTarget
from pyradbio.core.photon import PhotonParameters
from pyradbio.core.radiation import Ion

if TYPE_CHECKING:
    from pyradbio.models.lem.engine import Engine
    from pyradbio.models.lem.lem4 import LEM4
    from pyradbio.track.stopping_power import StoppingPower

FRIEDRICH2012_DOI = "10.3109/09553002.2011.611213"

_DATA_DIR = "data"
_CURVE_PREFIX = "friedrich2012_figure"


@dataclass(frozen=True, slots=True)
class PaperCellInputs:
    """Cell/target inputs Fig. 3 used (LEM IV photon + geometry, p. 106 caption).

    ``effective_size_factor = 1.0``: the paper quotes a nucleus radius to be
    used as-is, the same convention as
    [`pyradbio.benchmark.elsasser2010.PaperCellInputs`][].
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

        Friedrich et al. 2012 (doi:10.3109/09553002.2011.611213), Fig. 3 (p. 106) and
        Fig. 4 (p. 107) captions. Equal to
        ``PhotonParameters(alpha_x=inp.alpha_x, beta_x=inp.beta_x, D_t=inp.D_t)``.
        """
        return PhotonParameters(alpha_x=self.alpha_x, beta_x=self.beta_x, D_t=self.D_t)

    def cell_target(self) -> CellTarget:
        """Return the nucleus geometry, passing ``effective_size_factor`` on explicitly.

        Friedrich et al. 2012 (doi:10.3109/09553002.2011.611213), Fig. 3 and Fig. 4
        captions. Equal to
        ``CellTarget(inp.nucleus_radius_um, effective_size_factor=inp.effective_size_factor)``.
        """
        return CellTarget(
            nucleus_radius_um=self.nucleus_radius_um,
            effective_size_factor=self.effective_size_factor,
        )


# Fig. 3 caption (p. 106): "alpha_gamma = 0.184 Gy^-1, beta_gamma = 0.02
# Gy^-2, Dt = 17 Gy, Radius of the cell nucleus R_Nucleus = 4.7 um" (V79,
# carbon, Furusawa et al. 2000).
# Fig. 4 caption (p. 107): "hypothetical" cell, alpha_gamma = 0.25 Gy^-1,
# beta_gamma = 0.05 Gy^-2, Dt = 9.5 Gy, R_Nucleus = 5 um -- also
# tests/models/lem/test_lem4_intertrack.py's FIG4/FIG4_CELL fixture.
FIGURE_INPUTS: dict[str, PaperCellInputs] = {
    "fig3_v79": PaperCellInputs(0.184, 0.02, 17.0, 4.7),
    "fig4_hypothetical": PaperCellInputs(0.25, 0.05, 9.5, 5.0),
}


@dataclass(frozen=True, slots=True)
class Friedrich2012Curve:
    """One digitized reference curve (a single figure line or symbol set)."""

    figure: str
    cell_line: str
    particle: str
    series: str  # "calc" (paper model line) or "data" (experimental symbols)
    variant: str  # "rbe_alpha" or "rbe_10"
    x_quantity: str
    x_unit: str
    y_quantity: str
    y_unit: str
    x: tuple[float, ...]
    y: tuple[float, ...]
    source_page: int
    note: str
    doi: str = FRIEDRICH2012_DOI


def _data_files(prefix: str) -> list[tuple[str, str]]:
    """Return ``(filename, text)`` for every ``data/`` CSV starting with ``prefix``."""
    data_dir = resources.files(__package__).joinpath(_DATA_DIR)
    out: list[tuple[str, str]] = []
    for entry in data_dir.iterdir():
        if entry.name.startswith(prefix) and entry.name.endswith(".csv"):
            out.append((entry.name, entry.read_text(encoding="utf-8")))
    return out


@lru_cache(maxsize=1)
def load_curves() -> tuple[Friedrich2012Curve, ...]:
    """Return every digitized Friedrich et al. 2012 benchmark curve (one per file)."""
    curves: list[Friedrich2012Curve] = []
    for _name, text in _data_files(_CURVE_PREFIX):
        rows = list(csv.reader(text.splitlines()))
        meta = dict(zip(rows[0], rows[1], strict=True))
        x_quantity, y_quantity = rows[2][0], rows[2][1]
        x_unit, y_unit = rows[3][0], rows[3][1]
        data = [r for r in rows[4:] if r and r[0].strip()]
        points = sorted((float(r[0]), float(r[1])) for r in data)
        curves.append(
            Friedrich2012Curve(
                figure=meta["figure"],
                cell_line=meta["cell_line"],
                particle=meta["particle"],
                series=meta["series"],
                variant=meta["variant"],
                x_quantity=x_quantity,
                x_unit=x_unit,
                y_quantity=y_quantity,
                y_unit=y_unit,
                x=tuple(x for x, _ in points),
                y=tuple(y for _, y in points),
                source_page=int(meta["source_page"]),
                note=meta["note"],
                doi=meta.get("doi", FRIEDRICH2012_DOI),
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
) -> tuple[Friedrich2012Curve, ...]:
    """Filter digitized Friedrich et al. 2012 curves.

    ``series="calc"`` selects the paper's own model line; ``series="data"``
    selects the Furusawa et al. (2000) experimental symbols redrawn in the
    figure.
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
) -> Friedrich2012Curve:
    """Return exactly one Friedrich et al. 2012 reference curve.

    Raises ``LookupError`` if the filters select zero or multiple curves.
    """
    matches = iter_curves(
        figure=figure, cell_line=cell_line, particle=particle, series=series, variant=variant
    )
    if len(matches) != 1:
        raise LookupError(f"expected exactly one Friedrich et al. 2012 curve, got {len(matches)}")
    return matches[0]


# --- Fig. 4: pyRadBio's explicit-track Monte Carlo (offline, checkpointed) -----
#
# ``LEM4.local_dose_mc_alpha_beta`` reads alpha_I AND beta_I off one simulated
# survival curve under the exact <exp(-E)> convention -- the Scholz 1997 program
# Elsasser 2010 invokes for the macroscopic coefficients. The 12 production LETs
# cost ~2-3 h (numba box-average kernel), so the run is offline:
# ``scripts/lem4_friedrich2012_mc.py`` writes one CSV row per (LET, seed) into
# ``validation/data/lem4_friedrich2012_mc.csv``, and
# ``validation/lem4_elsasser2010_friedrich2012.py`` overlays it on its Fig. 4
# panels. Both build the model and invert the LET through the functions below, and
# the fast test ``tests/models/lem/test_lem4_mc_fingerprint.py`` recomputes the
# numeric fingerprint the CSV header records.

FIG4_MC_LETS: tuple[float, ...] = (
    15.0,
    18.0,
    21.0,
    24.0,
    28.0,
    32.0,
    36.0,
    40.0,
    55.0,
    70.0,
    85.0,
    100.0,
)
"""Carbon LETs [keV/um] of the production run.

Bounded below by cost, not by choice: low LET means high energy, the simulated
domain is ``R_eff + r_max`` with ``r_max`` growing like ``E^1.7``, and the impact
cut of ``local_dose_mc_impact_cut="auto"`` is floored at ``0.15 R_sim``. LET 13
(a 915 um domain) is left out.
"""

FIG4_MC_REALIZATIONS = 400
"""Realisations per seed of the production run."""

FIG4_MC_SEEDS = 8
"""Independent seeds ``0 .. FIG4_MC_SEEDS - 1``; the error bar is the scatter of their mean."""

FIG4_MC_ENERGY_BOUNDS_MEV_U: tuple[float, float] = (0.4, 600.0)
"""Carbon energy bracket [MeV/u] of the LET inversion (the high-energy branch)."""

FIG4_MC_FINGERPRINT_POINTS: tuple[tuple[float, int, int], ...] = ((40.0, 8, 0),)
"""``(LET, n_realizations, seed)`` of the fingerprint (``engine="numpy"``).

One point, so the fast test stays at a few seconds: LET 40 runs the
track/background split that every production point below LET ~45 depends on, and
its near tracks go through the exact engine the LET 55-100 points use. About 6 s:
the cost is the NumPy box-averaged profile, not the realisations.
"""

FIG4_MC_FIELDS: tuple[str, ...] = (
    "let_keV_um",
    "seed",
    "energy_MeV_u",
    "n_realizations",
    "alpha",
    "beta",
    "wall_s",
)
"""Columns of ``lem4_friedrich2012_mc.csv``: one row per ``(let_keV_um, seed)``."""

FIG4_MC_KEY_COLUMNS: tuple[str, ...] = ("let_keV_um", "seed")
"""Point key of the checkpoint ([`pyradbio.benchmark.checkpoint.CsvCheckpoint`][])."""

FIG4_MC_GENERATOR = "lem4_friedrich2012_mc"
"""The ``scripts/`` generator of the CSV (``assert_fingerprint(..., generator=...)``)."""

FIG4_MC_CSV_NAME = "lem4_friedrich2012_mc.csv"
"""File name of the checkpoint under ``validation/data/`` (or the generator's ``--out``)."""


def fig4_mc_model(*, engine: Engine = "auto", stopping_power: StoppingPower | None = None) -> LEM4:
    """The Fig. 4 "hypothetical" cell's LEM IV for the explicit-track Monte Carlo.

    ``LEM4(photon, cell, stopping_power, engine, local_dose_mc_impact_cut="auto")``
    on ``FIGURE_INPUTS["fig4_hypothetical"]`` (Friedrich et al. 2012,
    doi:10.3109/09553002.2011.611213, Fig. 4 caption, p. 107). The impact cut replaces the
    far tracks by their mean below ~LET 40, which brings LET 15-32 into reach; it
    engages only where the domain is large, so LET 55-100 run the exact engine.
    ``engine`` selects only the box-averaged profile's kernel (numba or NumPy, equal
    to ~1e-14 relative); the track sampling is NumPy either way.
    ``stopping_power=None`` uses
    [`default_stopping_power`][pyradbio.track.stopping_power.default_stopping_power].
    """
    from pyradbio.models.lem.lem4 import LEM4
    from pyradbio.track.stopping_power import default_stopping_power

    inputs = FIGURE_INPUTS["fig4_hypothetical"]
    return LEM4(
        photon=inputs.photon(),
        cell=inputs.cell_target(),
        stopping_power=default_stopping_power() if stopping_power is None else stopping_power,
        engine=engine,
        local_dose_mc_impact_cut="auto",
    )


def fig4_mc_ion(let_keV_um: float, stopping_power: StoppingPower) -> Ion:
    """Carbon-12 at the energy whose water LET is ``let_keV_um`` (high-energy branch).

    ``energy_for_let_MeV_u(branch="bracket")`` on [`FIG4_MC_ENERGY_BOUNDS_MEV_U`][]
    with scipy's default tolerances, bit-identical to the example's
    ``ion_for_let(6, 12, let)``. Raises ``ValueError`` if the LET is unreachable.
    """
    from pyradbio.track.stopping_power import energy_for_let_MeV_u

    energy = energy_for_let_MeV_u(
        stopping_power,
        6,
        12,
        float(let_keV_um),
        branch="bracket",
        energy_bounds_MeV_u=FIG4_MC_ENERGY_BOUNDS_MEV_U,
    )
    return Ion(6, 12, energy)


def fig4_mc_seed_statistics(pairs: npt.ArrayLike) -> tuple[float, float, float, float]:
    """``(alpha, alpha_err, beta, beta_err)`` from per-seed ``(alpha, beta)`` rows.

    The mean over seeds and its standard error ``std(ddof=1) / sqrt(n)``. Seeds are
    independent by construction, so the plain scatter of the mean needs no
    jackknife. Neither coefficient is clamped: in deep overkill a fitted ``beta``
    scatters through zero, and the bar must say so. The arithmetic and its order are
    those of the generator, so statistics recomputed from the CSV are bit-identical
    to the ones it printed.
    """
    rows = np.asarray(pairs, dtype=np.float64)
    if rows.ndim != 2 or rows.shape[1] != 2 or rows.shape[0] < 2:
        raise ValueError(f"need >= 2 seed rows of (alpha, beta), got shape {rows.shape}")
    out: list[float] = []
    for col in (rows[:, 0], rows[:, 1]):
        out.append(float(col.mean()))
        out.append(float(col.std(ddof=1) / math.sqrt(col.size)))
    return out[0], out[1], out[2], out[3]


def fig4_mc_fingerprint(
    points: Iterable[Sequence[float]] = FIG4_MC_FINGERPRINT_POINTS,
    *,
    stopping_power: StoppingPower | None = None,
) -> list[dict[str, Any]]:
    """Recompute the fingerprint: small NumPy MC runs at fixed seeds.

    One entry per ``(LET, n_realizations, seed)`` of ``points``, ``{"let_keV_um",
    "energy_MeV_u", "n_realizations", "seed", "engine": "numpy", "alpha",
    "beta"}``, from [`fig4_mc_model`][] with ``engine="numpy"`` and
    ``local_dose_mc_alpha_beta(ion, doses_Gy=model.fit_doses_Gy(), ...)``: the
    production call at a small sample size. The energy is part of the entry, so a
    change of stopping power fails the check too. Compare with
    [`pyradbio.benchmark.checkpoint.assert_fingerprint`][] (``rtol`` 1e-9).
    """
    model = fig4_mc_model(engine="numpy", stopping_power=stopping_power)
    doses = model.fit_doses_Gy()
    out: list[dict[str, Any]] = []
    for let, n_realizations, seed in points:
        ion = fig4_mc_ion(float(let), model.stopping_power)
        alpha, beta = model.local_dose_mc_alpha_beta(
            ion, doses_Gy=doses, n_realizations=int(n_realizations), seed=int(seed)
        )
        out.append(
            {
                "let_keV_um": float(let),
                "energy_MeV_u": ion.energy_MeV_u,
                "n_realizations": int(n_realizations),
                "seed": int(seed),
                "engine": "numpy",
                "alpha": float(alpha),
                "beta": float(beta),
            }
        )
    return out


@dataclass(frozen=True, eq=False)
class Fig4MonteCarlo:
    """The Fig. 4 explicit-track MC read back from its CSV, one entry per complete LET.

    ``let_keV_um`` ascending; ``alpha``/``beta`` the seed means and
    ``alpha_err``/``beta_err`` their standard errors
    ([`fig4_mc_seed_statistics`][]); ``seeds`` and ``n_realizations`` as recorded
    per LET. A LET with fewer than the required seeds (a run interrupted mid-point)
    is left out and listed in ``incomplete_lets``. ``header`` is the provenance
    header ([`pyradbio.benchmark.checkpoint.read_header`][]). Equality is
    identity (``eq=False``): array fields have no useful ``==``.
    """

    let_keV_um: npt.NDArray[np.float64]
    alpha: npt.NDArray[np.float64]
    alpha_err: npt.NDArray[np.float64]
    beta: npt.NDArray[np.float64]
    beta_err: npt.NDArray[np.float64]
    seeds: tuple[tuple[int, ...], ...]
    n_realizations: tuple[int, ...]
    incomplete_lets: tuple[float, ...]
    header: Mapping[str, Any]


def load_fig4_mc(path: str | os.PathLike[str], *, n_seeds: int | None = None) -> Fig4MonteCarlo:
    """Read ``lem4_friedrich2012_mc.csv`` and reduce it to per-LET seed statistics.

    A LET counts as complete with ``n_seeds`` seeds (default: the header's
    ``n_seeds``, else [`FIG4_MC_SEEDS`][]). Raises ``FileNotFoundError`` if the
    file does not exist and ``ValueError`` if one LET mixes realisation counts.
    """
    from pyradbio.benchmark.checkpoint import CsvCheckpoint

    if not os.path.exists(path):
        raise FileNotFoundError(path)
    ckpt = CsvCheckpoint(path, key_columns=FIG4_MC_KEY_COLUMNS, text_columns=())
    header = dict(ckpt.header)
    need = int(header.get("n_seeds", FIG4_MC_SEEDS)) if n_seeds is None else int(n_seeds)
    by_let: dict[float, list[dict[str, Any]]] = {}
    for record in ckpt.iter_records():
        by_let.setdefault(float(record["let_keV_um"]), []).append(record)
    lets: list[float] = []
    stats: list[tuple[float, float, float, float]] = []
    seeds: list[tuple[int, ...]] = []
    n_real: list[int] = []
    incomplete: list[float] = []
    for let in sorted(by_let):
        records = sorted(by_let[let], key=lambda r: float(r["seed"]))
        if len(records) < need:
            incomplete.append(let)
            continue
        realizations = {int(r["n_realizations"]) for r in records}
        if len(realizations) != 1:
            raise ValueError(f"{path}: LET {let} mixes n_realizations {sorted(realizations)}")
        lets.append(let)
        stats.append(fig4_mc_seed_statistics([[r["alpha"], r["beta"]] for r in records]))
        seeds.append(tuple(int(r["seed"]) for r in records))
        n_real.append(realizations.pop())
    columns = np.array(stats, dtype=np.float64).reshape(len(stats), 4)
    return Fig4MonteCarlo(
        let_keV_um=np.array(lets, dtype=np.float64),
        alpha=columns[:, 0].copy(),
        alpha_err=columns[:, 1].copy(),
        beta=columns[:, 2].copy(),
        beta_err=columns[:, 3].copy(),
        seeds=tuple(seeds),
        n_realizations=tuple(n_real),
        incomplete_lets=tuple(incomplete),
        header=header,
    )


__all__ = [
    "FIG4_MC_CSV_NAME",
    "FIG4_MC_ENERGY_BOUNDS_MEV_U",
    "FIG4_MC_FIELDS",
    "FIG4_MC_FINGERPRINT_POINTS",
    "FIG4_MC_GENERATOR",
    "FIG4_MC_KEY_COLUMNS",
    "FIG4_MC_LETS",
    "FIG4_MC_REALIZATIONS",
    "FIG4_MC_SEEDS",
    "FIGURE_INPUTS",
    "FRIEDRICH2012_DOI",
    "Fig4MonteCarlo",
    "Friedrich2012Curve",
    "PaperCellInputs",
    "fig4_mc_fingerprint",
    "fig4_mc_ion",
    "fig4_mc_model",
    "fig4_mc_seed_statistics",
    "get_curve",
    "iter_curves",
    "load_curves",
    "load_fig4_mc",
]
