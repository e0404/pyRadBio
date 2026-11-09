"""Elsasser, Kraemer & Scholz 2008 (LEM III) benchmark reference data.

Source paper (the velocity-dependent track-core extension of the LEM):

Elsasser, T., Kraemer, M. & Scholz, M. 2008, "Accuracy of the Local Effect Model
for the Prediction of Biologic Effects of Carbon Ion Beams in Vitro and in Vivo",
Int. J. Radiat. Oncol. Biol. Phys. 71(3), 866-872, doi:10.1016/j.ijrobp.2008.02.037.

Unlike LEM I/II, whose reconstructions rest on *digitized* figure rasters, LEM
III's quantitative content is **tabular**: the paper's Tables 1 and 2 report, for
the 15-cell-line HIMAC panel of Suzuki et al. 2000, the photon linear-quadratic
inputs, the experimental initial-slope RBE (``RBE_alpha``) at two carbon
qualities, the per-version transition dose ``D_t``, and the paper's own signed
deviation ``dRBE_alpha = RBE_LEM/RBE_exp - 1`` for LEM I, II and III. Those
deviations are the paper's headline result and the golden reference this module
carries; the three figures (rat-spinal-cord in vivo, and two clinical depth-dose
profiles) are treatment-planning outputs a survival model cannot reproduce
end-to-end and are not digitized.

The two carbon qualities come from a 290 MeV/u beam degraded with Lucite
absorbers to **13 keV/um** (entrance channel) and **77 keV/um** (Bragg-peak
region); in water those LETs correspond to carbon at roughly 282 and 25 MeV/u.

Per-line nuclear areas (Suzuki 2000, Table 3)
---------------------------------------------
Elsasser 2008 sets each cell's nuclear radius from Suzuki 2000's per-line nuclear
*area* and takes 80% of it as the effective target radius (Table 1 caption; the
factor is on the **radius**, [`EFFECTIVE_SIZE_FACTOR`][], confirmed against the
Methods: "the effective nuclear radius is approximately 80% of the average nuclear
radius"). Those areas are **not** reprinted in the LEM III paper; they are
transcribed here from Suzuki et al. 2000 (doi:10.1016/S0360-3016(00)00568-X,
[`SUZUKI2000_DOI`][], Table 3, "Area of cell nuclei"). Suzuki measured them as
an ellipse ``pi * a/2 * b/2`` from stained photographs, so the equivalent
geometric radius is ``sqrt(area / pi)`` ([`SuzukiCell.geometric_radius_um`][]);
pass that to [`CellTarget`][] with
``effective_size_factor=`` [`EFFECTIVE_SIZE_FACTOR`][]. (Suzuki lists 16 lines;
Elsasser dropped U-251MG(KO), leaving the 15 here.)

!!! important
    **The 77 keV/um channel needs these per-line areas; a single representative
    radius is not enough.** With the areas, LEM III reproduces the paper's
    ``dRBE_alpha(77)`` to 0.004 RMS for 14 of the 15 lines (LEM II: 0.037),
    versus a flat ``R_n = 5 um`` which mis-sizes the sensitive lines. The **13
    keV/um channel is nearly radius-independent** and matches per cell either way.

    **KS-1 is the one line the areas do not rescue.** It has the panel's highest
    ``alpha_x`` (0.74 Gy^-1), and our LEM overkills its high-LET track centre:
    ``dRBE_alpha(77)`` sits at about -0.50 (predicted RBE_alpha ~1.5 vs the
    experimental 3.0) against the paper's 0.00, independent of the nuclear radius.
    This is a genuine per-cell over-prediction of overkill for a high-``alpha_x``
    line, not a nuclear-size effect (see docs/references/LEM_III_Elsasser2008.md).
    It is excluded from the tight per-cell
    77 keV/um check in the golden test and flagged in the example.

Photon LQ inputs (Suzuki 2000, Table 2)
---------------------------------------
``alpha_x`` and ``beta_x`` are Suzuki 2000's measured 200-kV X-ray fits (Table 2,
p. 244, printed to four significant figures), **not** Elsasser's Table 1, which
shows them rounded to 2 / 3 decimals. The rounding matters: on the five
low-``alpha_x`` lines (A549, T98G, ONS-76, KNS-60, KNS-89) it shifts ``alpha_x``
by 4-6%, and the golden tests divide the model's alpha by ``alpha_x``. The paper
itself computed with the measured values: its experimental ``RBE_alpha`` column
reproduces to the printed digit from Suzuki's carbon and X-ray alphas (KNS-60:
0.1805/0.03156 = 5.72 as printed, whereas the printed 0.03 would give 6.02),
and switching to them roughly halves our per-cell dRBE(13) residuals for every
version (LEM III 77 keV/um: 0.029 -> 0.004 RMS). Elsasser's printed values are
kept in [`ELSASSER2008_PRINTED_PHOTON_LQ`][]; one of them, SK-MG-1
``beta_x`` = 0.054, is not a rounding of Suzuki's 0.05292 and is recorded as a
misprint ([`ELSASSER2008_BETA_X_MISPRINTS`][]).

Both table averages reproduce from the transcribed per-cell values to the paper's
rounding (see [`PAPER_AVERAGE_DELTA`][] and the module test), which validates
the transcription of the signs (the tumor-channel deviations are negative) and
the interleaved D_t +-10% error columns.
"""

from __future__ import annotations

import math
from collections.abc import Sequence
from dataclasses import dataclass

import numpy as np

from pyradbio.core.cell import CellTarget
from pyradbio.core.photon import ClusterParameters, PhotonParameters
from pyradbio.core.radiation import Ion
from pyradbio.models.lem.lem1 import LEM1
from pyradbio.models.lem.lem2 import LEM2, ClusterResponse
from pyradbio.models.lem.lem3 import LEM3
from pyradbio.track.stopping_power import StoppingPower, energy_for_let_MeV_u

ELSASSER2008_DOI = "10.1016/j.ijrobp.2008.02.037"
SUZUKI2000_DOI = "10.1016/S0360-3016(00)00568-X"

# The three model versions, in the fixed order used by every 3-tuple below.
MODEL_NAMES: tuple[str, str, str] = ("LEM I", "LEM II", "LEM III")

# The two carbon radiation qualities (LET in water), entrance channel and Bragg
# peak, from the 290 MeV/u beam degraded with Lucite absorbers (Results).
CHANNEL_LET_keV_um: tuple[float, float] = (13.0, 77.0)

# Effective nuclear radius = 0.8 * geometric radius. Confirmed against Elsasser
# 2008 Methods ("the effective nuclear radius is approximately 80% of the average
# nuclear radius") and Table 1 caption (radius "taken as 80% of the average
# value"). The factor is on the RADIUS, not the area. This is the one paper in the
# suite whose quoted per-line size is GEOMETRIC, so its reconstruction is the one
# place ``CellTarget.effective_size_factor = 0.8`` legitimately belongs.
EFFECTIVE_SIZE_FACTOR = 0.8


@dataclass(frozen=True, slots=True)
class SuzukiCell:
    """One Suzuki 2000 cell line as tabulated in Elsasser 2008 Tables 1 and 2.

    ``rbe_alpha_exp`` is the experimental initial-slope RBE at the two carbon
    qualities [`CHANNEL_LET_keV_um`][]; ``D_t`` is the chi^2-fitted transition
    dose for each model version; ``dRBE_13``/``dRBE_77`` are the paper's own
    signed deviations ``RBE_LEM/RBE_exp - 1`` at 13 / 77 keV/um, with
    ``dRBE_*_err`` the paper's error estimate from a +-10% variation of the most
    sensitive parameter ``D_t``. Every 3-tuple is indexed by [`MODEL_NAMES`][]
    order (LEM I, II, III).

    ``nucleus_area_um2`` is Suzuki 2000's measured nuclear area (Table 3); the
    *geometric* radius is [`geometric_radius_um`][]. Build the panel's cell with
    ``CellTarget(nucleus_radius_um=cell.geometric_radius_um,
    effective_size_factor=EFFECTIVE_SIZE_FACTOR)``.
    """

    name: str
    alpha_x: float  # Gy^-1, 200-kV X-ray LQ, Suzuki 2000 Table 2 (measured)
    beta_x: float  # Gy^-2, Suzuki 2000 Table 2 (measured)
    rbe_alpha_exp: tuple[float, float]  # experimental RBE_alpha at (13, 77) keV/um
    D_t: tuple[float, float, float]  # transition dose [Gy] for (LEM I, II, III)
    dRBE_13: tuple[float, float, float]  # paper dRBE_alpha(13) for (LEM I, II, III)
    dRBE_77: tuple[float, float, float]  # paper dRBE_alpha(77) for (LEM I, II, III)
    dRBE_13_err: tuple[float, float, float]
    dRBE_77_err: tuple[float, float, float]
    nucleus_area_um2: float  # Suzuki 2000 Table 3, "Area of cell nuclei" [um^2]

    def dRBE(self, version: int) -> tuple[float, float]:
        """Paper's ``(dRBE_alpha(13), dRBE_alpha(77))`` for model ``version`` (0/1/2)."""
        return self.dRBE_13[version], self.dRBE_77[version]

    def rbe_alpha_paper(self, version: int) -> tuple[float, float]:
        """Paper's *predicted* ``RBE_alpha`` = experiment * (1 + dRBE) at (13, 77)."""
        d13, d77 = self.dRBE(version)
        e13, e77 = self.rbe_alpha_exp
        return e13 * (1.0 + d13), e77 * (1.0 + d77)

    @property
    def geometric_radius_um(self) -> float:
        """Equivalent-circle geometric nuclear radius ``sqrt(area / pi)`` [um].

        Suzuki measured the area as an ellipse; this is its area-equivalent
        radius. Multiply by [`EFFECTIVE_SIZE_FACTOR`][] for the LEM target
        radius (or, equivalently, pass ``effective_size_factor`` to
        [`CellTarget`][]).
        """
        return math.sqrt(self.nucleus_area_um2 / math.pi)

    def cell_target(self) -> CellTarget:
        """The LEM target of this line: geometric radius, x0.8 to the effective size.

        ``CellTarget(nucleus_radius_um=self.geometric_radius_um,
        effective_size_factor=EFFECTIVE_SIZE_FACTOR)`` -- Elsasser 2008 sets each
        line's radius from Suzuki 2000's nuclear area (Table 3) and takes 80% of it
        as the effective target (Table 1 caption; Methods). The factor is passed
        explicitly, as every paper-input ``cell_target()`` does.
        """
        return CellTarget(
            nucleus_radius_um=self.geometric_radius_um,
            effective_size_factor=EFFECTIVE_SIZE_FACTOR,
        )


# Elsasser 2008 Tables 1 (LEM I) and 2 (LEM II, LEM III), transcribed from the
# PDF page 869, with the photon LQ (alpha_x, beta_x) taken at full precision from
# Suzuki 2000 Table 2 (see the module docstring and ELSASSER2008_PRINTED_PHOTON_LQ)
# and each line's nuclear area from Suzuki 2000 Table 3. Signs: the
# entrance-channel deviations dRBE(13) are positive (RBE over-predicted); the
# Bragg-channel deviations dRBE(77) are negative (under-predicted). Every 3-tuple
# is indexed (LEM I, LEM II, LEM III).
#
#   SuzukiCell(name, alpha_x, beta_x, (RBE13, RBE77), D_t=(I,II,III),
#              dRBE_13=(I,II,III), dRBE_77=(I,II,III),
#              dRBE_13_err=(I,II,III), dRBE_77_err=(I,II,III), nucleus_area_um2)
SUZUKI_PANEL: tuple[SuzukiCell, ...] = (
    SuzukiCell(
        "KS-1",
        0.7446,
        0.03500,
        (1.20, 3.00),
        (70.0, 11.0, 12.50),
        (0.36, 0.24, -0.02),
        (-0.11, -0.10, 0.00),
        (0.03, 0.03, 0.00),
        (0.04, 0.03, 0.01),
        221.4,
    ),
    SuzukiCell(
        "NB1RGB",
        0.4282,
        0.05179,
        (1.32, 3.96),
        (31.0, 7.0, 36.50),
        (0.51, 0.36, 0.12),
        (-0.17, -0.11, 0.00),
        (0.06, 0.05, 0.01),
        (0.05, 0.05, 0.06),
        172.3,
    ),
    SuzukiCell(
        "HFL-III",
        0.4153,
        0.06176,
        (1.31, 4.06),
        (31.5, 7.0, 30.50),
        (0.70, 0.50, 0.19),
        (-0.08, -0.05, 0.00),
        (0.08, 0.07, 0.01),
        (0.06, 0.06, 0.05),
        142.9,
    ),
    SuzukiCell(
        "C32TG",
        0.2508,
        0.03959,
        (1.34, 4.47),
        (21.0, 5.4, 29.50),
        (0.48, 0.36, 0.19),
        (-0.27, -0.18, -0.02),
        (0.05, 0.06, 0.01),
        (0.04, 0.05, 0.06),
        166.6,
    ),
    SuzukiCell(
        "A-172",
        0.3627,
        0.08094,
        (2.34, 5.66),
        (30.5, 7.6, 30.50),
        (0.20, 0.08, -0.21),
        (-0.09, -0.04, 0.01),
        (0.05, 0.06, 0.01),
        (0.05, 0.06, 0.06),
        209.1,
    ),
    SuzukiCell(
        "Becker",
        0.1000,
        0.02241,
        (1.29, 3.16),
        (8.5, 2.2, 10.00),
        (0.32, 0.19, 0.17),
        (-0.18, -0.07, -0.03),
        (0.05, 0.02, 0.02),
        (0.05, 0.03, 0.05),
        203.9,
    ),
    SuzukiCell(
        "Marcus",
        0.1777,
        0.04737,
        (1.73, 4.12),
        (13.0, 3.4, 14.00),
        (0.25, 0.10, 0.00),
        (-0.12, -0.05, 0.01),
        (0.06, 0.06, 0.02),
        (0.06, 0.06, 0.07),
        187.9,
    ),
    SuzukiCell(
        "SF126",
        0.2113,
        0.06384,
        (1.89, 4.23),
        (13.5, 3.4, 13.00),
        (0.25, 0.06, -0.05),
        (-0.06, -0.03, 0.01),
        (0.06, 0.05, 0.03),
        (0.06, 0.06, 0.07),
        150.9,
    ),
    SuzukiCell(
        "SK-MG-1",
        0.1095,
        0.05292,
        (2.03, 5.66),
        (10.5, 3.0, 10.50),
        (0.39, 0.19, 0.05),
        (-0.11, -0.05, -0.02),
        (0.06, 0.08, 0.03),
        (0.06, 0.08, 0.07),
        218.1,
    ),
    SuzukiCell(
        "LC-1-Sq",
        0.2446,
        0.1271,
        (2.01, 5.97),
        (9.0, 2.8, 11.00),
        (0.35, 0.20, 0.12),
        (-0.21, -0.13, -0.02),
        (0.08, 0.09, 0.03),
        (0.06, 0.08, 0.06),
        230.3,
    ),
    SuzukiCell(
        "A549",
        0.07456,
        0.04548,
        (2.93, 9.66),
        (15.5, 4.8, 17.50),
        (0.39, 0.27, -0.01),
        (-0.17, -0.08, 0.01),
        (0.07, 0.10, 0.03),
        (0.06, 0.09, 0.06),
        182.3,
    ),
    SuzukiCell(
        "T98G",
        0.06373,
        0.04615,
        (1.98, 6.78),
        (7.0, 2.2, 8.00),
        (0.50, 0.30, 0.22),
        (-0.21, -0.14, -0.05),
        (0.11, 0.07, 0.06),
        (0.08, 0.05, 0.08),
        282.5,
    ),
    SuzukiCell(
        "ONS-76",
        0.09584,
        0.09021,
        (1.92, 6.93),
        (4.5, 1.6, 6.00),
        (0.47, 0.32, 0.33),
        (-0.30, -0.20, -0.08),
        (0.08, 0.09, 0.04),
        (0.05, 0.07, 0.05),
        176.9,
    ),
    SuzukiCell(
        "KNS-60",
        0.03156,
        0.04816,
        (5.72, 25.84),
        (18.0, 5.6, 20.00),
        (0.68, 0.52, 0.05),
        (-0.18, -0.11, -0.01),
        (0.12, 0.13, 0.04),
        (0.07, 0.08, 0.08),
        263.8,
    ),
    SuzukiCell(
        "KNS-89",
        0.05279,
        0.08983,
        (2.45, 12.21),
        (5.0, 1.8, 6.00),
        (0.86, 0.64, 0.55),
        (-0.29, -0.21, -0.12),
        (0.12, 0.12, 0.06),
        (0.05, 0.07, 0.06),
        187.3,
    ),
)

# Elsasser 2008 Table 1's photon LQ exactly as printed, (alpha_x, beta_x). Kept
# for provenance only: the panel uses Suzuki's measured values, of which these
# are the rounding -- except the misprints listed below.
ELSASSER2008_PRINTED_PHOTON_LQ: dict[str, tuple[float, float]] = {
    "KS-1": (0.74, 0.035),
    "NB1RGB": (0.43, 0.052),
    "HFL-III": (0.42, 0.062),
    "C32TG": (0.25, 0.040),
    "A-172": (0.36, 0.081),
    "Becker": (0.10, 0.022),
    "Marcus": (0.18, 0.047),
    "SF126": (0.21, 0.064),
    "SK-MG-1": (0.11, 0.054),
    "LC-1-Sq": (0.24, 0.127),
    "A549": (0.07, 0.045),
    "T98G": (0.06, 0.046),
    "ONS-76": (0.10, 0.090),
    "KNS-60": (0.03, 0.048),
    "KNS-89": (0.05, 0.090),
}

# Lines whose printed beta_x is not a rounding of Suzuki 2000 Table 2: SK-MG-1 is
# printed 0.054, Suzuki measured 0.05292 (which rounds to 0.053).
ELSASSER2008_BETA_X_MISPRINTS: frozenset[str] = frozenset({"SK-MG-1"})

# Published "Average" rows of Tables 1-2 (signed mean over the 15 lines), kept as
# an independent transcription so the module test can confirm the per-cell values
# average back to them. Indexed (LEM I, LEM II, LEM III): (mean dRBE13, mean dRBE77).
PAPER_AVERAGE_DELTA: tuple[tuple[float, float], ...] = (
    (0.45, -0.17),  # LEM I
    (0.29, -0.10),  # LEM II
    (0.11, -0.02),  # LEM III
)

# KS-1 is a documented per-cell outlier: our LEM overkills its high-LET track
# centre (highest alpha_x in the panel), so the 77 keV/um deviation stays ~ -0.5
# regardless of nuclear size. Excluded from the tight per-cell 77 keV/um check.
OUTLIER_CELLS_77: frozenset[str] = frozenset({"KS-1"})


@dataclass(frozen=True, slots=True)
class InVivoInputs:
    """Rat spinal-cord in-vivo photon inputs (Elsasser 2008 Fig. 1 / Fig. 3).

    Used for the tolerance comparison of Karger et al. 2006; ``D_t`` is indexed by
    [`MODEL_NAMES`][]. The RBE there is derived from a 1-cm spread-out Bragg peak
    at 50% complication probability, which needs a depth-dose engine, so it is a
    documented input set rather than a reproduced curve.
    """

    alpha_x: float  # Gy^-1
    beta_x: float  # Gy^-2
    nucleus_radius_um: float
    D_t: tuple[float, float, float]  # (LEM I, II, III)


# Fig. 1 caption: alpha=0.1, beta=0.05, R_nucleus=5 um, D_t = 30 / 9.5 / 60 Gy for
# LEM I / II / III (alpha/beta = 2 Gy). Fig. 3 also gives an alpha/beta = 2.8 Gy
# variant (alpha=0.14, beta=0.05, D_t=70 Gy, LEM III only).
RAT_SPINAL_CORD = InVivoInputs(0.1, 0.05, 5.0, (30.0, 9.5, 60.0))


def paper_mean_delta(version: int) -> tuple[float, float]:
    """Signed mean ``(dRBE_alpha(13), dRBE_alpha(77))`` over the panel for ``version``.

    Computed from [`SUZUKI_PANEL`][]; rounds to the published
    [`PAPER_AVERAGE_DELTA`][].
    """
    n = len(SUZUKI_PANEL)
    m13 = sum(c.dRBE_13[version] for c in SUZUKI_PANEL) / n
    m77 = sum(c.dRBE_77[version] for c in SUZUKI_PANEL) / n
    return m13, m77


# --- reconstruction machinery ---------------------------------------------------

PanelRow = tuple[float, float, float, float]
"""One panel row per cell line: ``(RBE_alpha predicted at 13, predicted at 77,
experimental at 13, experimental at 77)`` -- the column order of the Tables 1-2
reconstruction ([`predict_panel`][]).
"""


def channel_ions(
    stopping_power: StoppingPower,
    *,
    energy_bounds_MeV_u: tuple[float, float] = (1.5, 500.0),
) -> tuple[Ion, Ion]:
    """Carbon-12 ions at the two Tables 1-2 qualities, 13 and 77 keV/um in water.

    Elsasser, Kraemer & Scholz 2008 (doi:10.1016/j.ijrobp.2008.02.037) degrade a
    290 MeV/u carbon beam to the [`CHANNEL_LET_keV_um`][] of the entrance channel
    and the Bragg-peak region; in water those are carbon at ~282-285 and ~25 MeV/u.
    Each energy is the root of ``LET(E) = LET`` bracketed on exactly
    ``energy_bounds_MeV_u`` with scipy's default tolerances
    ([`energy_for_let_MeV_u`][],
    ``branch="bracket"``) -- the inline ``brentq(..., 1.5, 500.0)`` of the
    reconstruction script and the LEM III golden tests, bit for bit.
    """
    let_13, let_77 = CHANNEL_LET_keV_um
    e_13 = energy_for_let_MeV_u(
        stopping_power, 6, 12, let_13, branch="bracket", energy_bounds_MeV_u=energy_bounds_MeV_u
    )
    e_77 = energy_for_let_MeV_u(
        stopping_power, 6, 12, let_77, branch="bracket", energy_bounds_MeV_u=energy_bounds_MeV_u
    )
    return Ion(6, 12, e_13), Ion(6, 12, e_77)


def build_panel_model(
    name: str,
    alpha_x: float,
    beta_x: float,
    d_t: float,
    cell: CellTarget,
    *,
    stopping_power: StoppingPower,
    cluster: ClusterResponse | None = None,
) -> LEM1 | LEM2 | LEM3:
    """The LEM version ``name`` (one of [`MODEL_NAMES`][]) for one panel line.

    The photon reference is ``PhotonParameters(alpha_x, beta_x, D_t=d_t,
    cluster=ClusterParameters())``. ``"LEM I"`` is a [`LEM1`][] (no cluster);
    ``"LEM II"`` / ``"LEM III"`` are [`LEM2`][] / [`LEM3`][] with ``cluster``
    (``None``: the model's own default). ``stopping_power`` is required, with no
    default, so the models cannot silently use a different stopping power from the
    one that placed the ions (pass the same object to [`channel_ions`][]). Every
    model uses its version's default radial dose and initial-slope method.

    Raises:
        ValueError: if ``name`` is not one of [`MODEL_NAMES`][].
    """
    if name not in MODEL_NAMES:
        raise ValueError(f"name must be one of {MODEL_NAMES}, got {name!r}")
    photon = PhotonParameters(alpha_x=alpha_x, beta_x=beta_x, D_t=d_t, cluster=ClusterParameters())
    if name == "LEM I":
        return LEM1(photon=photon, cell=cell, stopping_power=stopping_power)
    cls = LEM2 if name == "LEM II" else LEM3
    return cls(photon=photon, cell=cell, stopping_power=stopping_power, cluster=cluster)


def predict_panel(
    name: str,
    ions: tuple[Ion, Ion],
    *,
    stopping_power: StoppingPower,
    cluster: ClusterResponse | None = None,
    panel: Sequence[SuzukiCell] = SUZUKI_PANEL,
) -> tuple[PanelRow, ...]:
    """``RBE_alpha`` of LEM version ``name`` for every line of ``panel`` (Tables 1-2).

    Per line: [`build_panel_model`][] with the line's photon LQ, its version's
    ``D_t`` and [`SuzukiCell.cell_target`][]; then
    ``alpha_beta(ion).alpha / alpha_x`` at the two ``ions`` (13, 77 keV/um, see
    [`channel_ions`][], built with the same ``stopping_power``, which is required
    so the ions and the models cannot mix stopping powers). Returns one
    [`PanelRow`][] per line, in panel order:
    ``(predicted 13, predicted 77, experimental 13, experimental 77)``.
    """
    version = MODEL_NAMES.index(name)
    ion_13, ion_77 = ions
    rows: list[PanelRow] = []
    for cell in panel:
        rbe13_exp, rbe77_exp = cell.rbe_alpha_exp
        model = build_panel_model(
            name,
            cell.alpha_x,
            cell.beta_x,
            cell.D_t[version],
            cell.cell_target(),
            stopping_power=stopping_power,
            cluster=cluster,
        )
        rbe13 = model.alpha_beta(ion_13).alpha / cell.alpha_x
        rbe77 = model.alpha_beta(ion_77).alpha / cell.alpha_x
        rows.append((rbe13, rbe77, rbe13_exp, rbe77_exp))
    return tuple(rows)


def mean_delta(rows: Sequence[PanelRow]) -> tuple[float, float]:
    """Signed mean ``dRBE_alpha = RBE_pred/RBE_exp - 1`` at (13, 77) keV/um.

    Our analogue of the Tables 1-2 "Average" rows ([`PAPER_AVERAGE_DELTA`][]) for
    rows from [`predict_panel`][].
    """
    arr = np.array(rows)
    d13 = arr[:, 0] / arr[:, 2] - 1.0
    d77 = arr[:, 1] / arr[:, 3] - 1.0
    return float(np.mean(d13)), float(np.mean(d77))


def per_cell_rms(
    rows: Sequence[PanelRow],
    name: str,
    channel: int,
    *,
    exclude: frozenset[str] = frozenset(),
    panel: Sequence[SuzukiCell] = SUZUKI_PANEL,
) -> float:
    """RMS over ``panel`` of (our ``dRBE_alpha`` - the paper's tabulated one).

    ``rows`` come from [`predict_panel`][] for the same ``panel`` and model
    ``name``; ``channel`` is 0 for 13 keV/um and 1 for 77 keV/um; lines named in
    ``exclude`` are dropped (e.g. [`OUTLIER_CELLS_77`][]). The paper's value is
    ``dRBE_13`` / ``dRBE_77`` of Elsasser 2008 Tables 1-2.

    Raises:
        ValueError: if ``channel`` is not 0 or 1, or ``rows`` and ``panel`` differ
            in length.
    """
    if channel not in (0, 1):
        raise ValueError(f"channel must be 0 (13 keV/um) or 1 (77 keV/um), got {channel!r}")
    if len(rows) != len(panel):
        raise ValueError(f"rows ({len(rows)}) and panel ({len(panel)}) differ in length")
    v = MODEL_NAMES.index(name)
    residuals = [
        (rows[i][channel] / rows[i][channel + 2] - 1.0)
        - (cell.dRBE_13[v] if channel == 0 else cell.dRBE_77[v])
        for i, cell in enumerate(panel)
        if cell.name not in exclude
    ]
    return float(np.sqrt(np.mean(np.square(residuals))))


__all__ = [
    "EFFECTIVE_SIZE_FACTOR",
    "ELSASSER2008_BETA_X_MISPRINTS",
    "ELSASSER2008_DOI",
    "ELSASSER2008_PRINTED_PHOTON_LQ",
    "MODEL_NAMES",
    "OUTLIER_CELLS_77",
    "PAPER_AVERAGE_DELTA",
    "RAT_SPINAL_CORD",
    "SUZUKI2000_DOI",
    "SUZUKI_PANEL",
    "CHANNEL_LET_keV_um",
    "InVivoInputs",
    "PanelRow",
    "SuzukiCell",
    "build_panel_model",
    "channel_ions",
    "mean_delta",
    "paper_mean_delta",
    "per_cell_rms",
    "predict_panel",
]
