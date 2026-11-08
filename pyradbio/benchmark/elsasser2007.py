"""Digitized benchmark curves from Elsasser & Scholz 2007 (LEM II).

Source paper (the LEM II cluster extension + refined radial dose):

Elsasser, T. & Scholz, M. 2007, "Cluster Effects within the Local Effect Model",
Radiat. Res. 167(3), 319-329, doi:10.1667/RR0467.1.

The bundled numeric curves are **hand-digitized** from the paper's embedded
figure rasters and are accurate enough for quantitative comparison.

Fig. 2 has been digitized **twice, independently** -- by hand and by an automated
raster trace -- which agree to 0.9% RMS (1.6% max), peaking at 14.79 vs 14.74.
The hand digitization is the bundled one. For the auto-traced Fig. 9 the axis
calibration residuals are ~0.7 px in LET and <0.003 RBE units.

Which panels are digitized, and which are *derived*
----------------------------------------------------

* **Fig. 2** (``eta(D)``) is digitized. It is the paper's own model output
  (eq. 9 evaluated from fixed constants), so it doubles as an independent check
  on [`ClusterModel`][pyradbio.models.lem.lem2.ClusterModel] -- see the warning below.
* **Fig. 3** (modified photon dose-effect) is **derived, not digitized**: it is
  exactly [`photon_log_kill`][pyradbio.models.lem.lem2.LEM2.photon_log_kill], i.e. the
  shouldered LQ below ``D_t`` and ``-ln S_x(D_t) + s_max * (eta(D)*D - D_t)``
  above it (eq. 10), with ``s_max = alpha_x + 2*beta_x*D_t``. Every ingredient is
  fixed by Fig. 2's ``eta`` plus the photon inputs, so digitizing it would add no
  information -- confirmed by digitizing it anyway: the derivation reproduces the
  dashed (``eta = 1``) tails to 1.3-1.6% RMS and the clustered ones to 2.7%.
  It is drawn for **CHO**, and its caption's ``D_t = 30`` Gy is a paper error
  (the panel uses CHO's old ``D_t = 40`` Gy); see [`FIG3_D_T_OLD_GY`][].

  Note that ``eta`` **multiplies the dose** in eq. 10; it is not integrated over
  it. Reading eq. 10 as an enhanced local slope (``s_max * integral eta dD'``)
  misses Fig. 3 by 22.7% RMS.
* **Fig. 4** (radial dose vs *experimental* data) is digitized: the paper's only
  absolute, measured check on
  [`ScholzKraftRadialDose`][], and its only panel
  outside carbon. Panel a is 2.57 MeV/u ``16O`` in **tissue-equivalent gas**
  (Varma, Baum & Kuehner 1977); panel b is 377 MeV/u ``20Ne`` in **water** (Varma
  & Baum 1980). Axes are ``r`` in **cm** and dose in **Gy**.

!!! important
    **Fig. 4's "modified" line is the UN-convolved profile** -- the ``r_min``
    reduction to 0.3 nm alone, *without* the sigma = 4 nm radical-diffusion
    Gaussian that Fig. 5's ``D~(r)`` shows and that ``lem2()`` applies. The raster
    settles it: the curve plateaus flat to a **sharp knee** at ~3e-8 cm and is a
    straight ``1/r^2`` line after it, where a Gaussian would round the knee and
    drop the plateau ~50x. Numerically, against panel a our
    ``radical_diffusion=False`` variant scores relRMS(log10) = **0.021** and the
    convolved ``lem2()`` **0.874**. So compare Fig. 4 to
    ``ScholzKraftRadialDose(r_min_nm=LEM2_RMIN_NM, radical_diffusion=False)``;
    using ``lem2()`` here is apples-to-oranges and looks like a 50x core deficit.

  Agreement is essentially exact: RMS(log10) is 0.013-0.078 across the four model
  lines. There is a small positive offset (median ratio O modified 1.05, O original
  1.10, Ne modified 1.12, Ne original 1.14), but it is **within digitization
  uncertainty on these axes, not a physics signal**: the residual is a flat vertical
  offset (its slope vs log10 r is ~-0.003, i.e. no radius trend over 3-7 decades),
  it is only 0.02-0.06 decades on a ~15-decade panel, and on a 1/r^2 tail a 2-6%
  radius-calibration slip already accounts for +4..+12% in dose (D ~ 1/r^2 doubles
  any x error into y). The neon lines are also traced sparsely (5-7 points on a
  straight log-log line), so their offset is just the y-anchor of the trace.
  Consequently Fig. 4 *validates* the absolute radial dose -- and therefore rules
  it out as the cause of the HSG scale offset in Fig. 11.
* **Fig. 8** (``RBE_alpha`` for V79 *and* CHO) is digitized: four model lines plus
  both Weyrather 1999 data sets, the only ones here with error bars (``y_err``).
  Drawing two cell lines from one code path makes it the paper's sharpest test of
  the nuclear geometry -- the lines differ mainly by ``r_nucleus`` (3.8 vs 4.7 um)
  -- and that makes it sensitive to the ``effective_size_factor`` convention (see
  [`PaperCellInputs`][]). Its V79 lines are Fig. 6's ``lem_old``/``lem_new``
  re-drawn and, traced independently, agree with them to 0.3% / 1.5% relRMS.
* **Fig. 9** (RBE at 10% survival, V79 carbon) is digitized: both model lines
  plus the Weyrather 1999 experimental squares (``series="data"``).

  Like Fig. 2 it has been digitized **twice, independently** -- a raster
  auto-trace and a hand trace -- agreeing to 1.3% (modified), 1.9%
  (original) and 1.2% (data) RMS, which is what bounds this panel's digitization
  error. The bundled curves are the hand trace, because the auto-trace was
  **non-monotone past the RBE maximum**: its ``original_lem`` line rose +1.2% at
  428 keV/um before falling again, and it stopped 23 keV/um short of the panel.
  Past the maximum a LEM curve can only thin down, so a rising tail is an
  artifact; ``test_fig9_model_lines_fall_monotonically_past_the_rbe_maximum``
  pins that. The two traces are indistinguishable below ~370 keV/um.
* **Figs. 10 & 12** (survival curves; ``x`` = dose in Gy, ``y`` = surviving
  fraction) are the reconstruction's only test of the *full dose-response* -- i.e.
  of ``beta`` and the ``D_t`` refit, where every other panel is ``alpha``-flavoured.
  Fig. 10 is V79 carbon at four energies (LETs 339 / 153.5 / 32.5 / 13.7 keV/um for
  4.2 / 11 / 76.9 / 266.4 MeV/u); the modified LEM beats the original at every one
  (e.g. relRMS in ln S drops 16.3% -> 5.5% at 4.2 MeV/u), and the 4.2 MeV/u panel
  is where our stopping power is weakest (-9%).

  **Fig. 12 (XRS) is the cleanest isolated test of the cluster ``eta`` in the whole
  paper.** XRS is repair-deficient with a **pure-exponential** photon response
  (``alpha_x = 1.7`` Gy^-1, ``beta_x = 0`` -- no shoulder), so there is nothing for
  an amorphous-track LEM to exploit: without ``eta``, LEM I only *loses* ground to
  overkill saturation (``alpha`` ~ 1.4 < the photon 1.7 at 11 MeV/u, i.e. RBE < 1).
  The cluster ``eta`` making the core response non-linear is the entire reason
  ``alpha`` climbs back above the photon value. Its x-ray reference column has
  ln-slope -1.71 (= ``alpha_x``, confirming the pure exponential). Despite our
  ``eta`` peak being ~11% low, our XRS survival matches the paper's model lines to
  0.7-9.3% relRMS (11 MeV/u ``alpha`` 2.60 vs the digitized 2.61) -- independent
  confirmation that the ``eta`` rule mismatch is nearly RBE-neutral.
  XRS needs its own ``r_nucleus = 5.5`` um (a fourth cell line); see
  ``FIGURE_INPUTS["fig12_xrs"]``.
* **Fig. 11** (initial slope ``alpha`` vs LET, HSG) is digitized: it is the only
  panel with a **second ion species** (helium as well as carbon, hence
  ``particle``) and a **third cell line**, so it is the one place the
  reconstruction leaves carbon-on-V79/CHO. Note its y-axis is ``alpha`` in Gy^-1,
  *not* ``RBE_alpha``. No error bars in the panel.

  Its LET axis is easy to misread: the ticks run ``2,3..9, 10^2, 2,3..9, 10^3``,
  so the leftmost labelled tick is **2e1 = 20** and the panel spans ~15..1000
  keV/um -- not 2..1000. Anchoring it at 2 compresses the axis
  (1.78..333): LEM II scores relRMS 29-58% against the compressed axis vs 7-14%
  against the correct one, and the carbon peak lands at 70 keV/um where the model
  puts it at ~196.

!!! warning
    **Our ``ClusterModel`` does not reproduce the digitized Fig. 2.** Writing
    ``q(mu)`` for the clustered DSBs per stagger-interval and
    ``mu = alpha_SSB*D*h/G``, eq. 9 gives

        eta(D) - 1 = (alpha_SSB/alpha_DSB) * q(mu)/mu = 41.667 * q(mu)/mu

    so the genome size and stagger only set the ``mu``<->``D`` scale and **cannot**
    change the peak *height*: ``max[q/mu]`` identifies the clustering rule. The
    digitized Fig. 2 peaks at ``eta = 14.79`` (D = 4.4e5 Gy; the caption says
    "about 5e5 Gy"), requiring ``max[q/mu] = 0.3310``. Our fixed-``h``-bin rule
    gives 0.2984 -> ``eta_peak = 13.43`` at the fitted genome size 8.6e9 (the
    default 6.0e9 gives ``ClusterModel.eta_max`` 13.40, i.e. 0.2976; the spread is
    Monte-Carlo noise). Agreement is within 4% below ~3e4 Gy but the
    tail is -32% at 1e6 Gy and -46% at 3e6 Gy. The two independent digitizations
    bracket the requirement at 0.3297-0.3310, so the ~11% shortfall of our rule is
    far outside digitization error. Note also that
    ``ClusterParameters.genome_size_bp = 6.0e9`` is **our assumption**: the paper
    only says "only the genomic size is required" and never states a number.

    **Fig. 2 is not reachable from the paper's own stated rule, for any genome
    size.** The prose (p. 321) requires two SSBs *on opposite strands* within
    ``h = 25`` bp, with at most one DSB per 25-bp interval ("any additional strand
    break within a 25-bp interval including two SSBs or a DSB [...] is discarded").
    A direct Monte Carlo of that literal algorithm gives ``max[q/mu] = 0.2304``, i.e.
    ``eta_peak = 10.6`` -- further from Fig. 2 than our own rule. Fitting each
    candidate's one free parameter (``G``) to the whole digitized curve ranks them:

    ===========================================  =========  ========  =====  =======
    rule                                         max[q/mu]  eta_peak  rms    G fit
    ===========================================  =========  ========  =====  =======
    any-strand + 25-bp discard                   0.3183     14.20     0.64   1.28e10
    fixed bins >= 2 SSBs (**ours**)              0.2984     13.43     1.53   8.6e9
    paper's literal rule (opposite + discard)    0.2304     10.58     2.43   8.8e9
    opposite-strand, no exclusion                0.3339     14.91     3.32   4.7e9
    ===========================================  =========  ========  =====  =======

    The closest rule needs ``G = 1.28e10`` bp (over twice a mammalian genome) *and*
    discards the opposite-strand physics the paper states. Meanwhile Fig. 2's peak
    is within 0.7% of **exactly 1/3**, which is the ceiling of opposite-strand
    pairing with *no* exclusion (each DSB consumes 3 SSBs on average, since half of
    the neighbouring SSBs are on the same strand). A curve that peaks and then falls
    cannot attain its own no-exclusion asymptote, so Fig. 2 appears **inconsistent
    with the paper's own prose** -- a third paper error alongside the Fig. 3 caption
    ([`FIG3_D_T_OLD_GY`][]), not a bug of ours.

    **The same group's later paper settles it.** Friedrich, Durante & Scholz 2015
    ("Simulation of DSB yield for high LET radiation", doi:10.1093/rpd/ncv147)
    recompute this exact quantity from the *same* ``alpha_SSB = 1250``,
    ``alpha_DSB = 30``, ``t = 25`` bp, with a closed form and a **stricter** rule
    than either of the above -- it drops SSBs shadowed by a real DSB, takes the
    opposite-strand factor explicitly, and forbids one SSB serving two DSBs
    ([`Friedrich2015ClusterModel`][pyradbio.models.lem.lem2.Friedrich2015ClusterModel]). Its peak is
    **13.51**, i.e. on top of our 13.40 and 8.6% *below* Fig. 2's 14.79. Over the
    doses an ion track can actually produce -- ``<= 1e5`` Gy in a carbon track
    centre, per that paper's own Discussion -- the two rules agree to **3.0% RMS**
    and both sit 0.5-4.7% under the digitized Fig. 2. They part company only above
    ~1e6 Gy (+37% vs -42% against Fig. 2, 140% apart from each other), which is
    exactly the regime the 2015 paper says is undetermined: "it is not clear how to
    exactly implement the conversion of candidates for DSB into actual DSB [...] the
    high dose part around the maximum of eta [...] is affected, while the course of
    the curve at lower doses remains rather unchanged."

    So the honest reading is: **Fig. 2's peak is the outlier**, our ``eta`` is
    within a few percent of an independent implementation by the paper's own group
    everywhere it matters, and the disagreement lives in a dose regime that is both
    unreachable by a track and explicitly not pinned down by its authors.

    **This mismatch is, however, NOT a LEM II RBE deficit.** Substituting the
    paper's own digitized ``eta`` here for [`ClusterModel`][pyradbio.models.lem.lem2.ClusterModel]
    moves Fig. 9's peak RBE only 2.43 -> 2.48 and Fig. 6's peak ``RBE_alpha`` 7.73 -> 7.89: a
    *perfect* ``eta`` buys ~1.5 RBE points of a ~12% shortfall. Do not expect fixing the clustering
    rule to move the RBE panels.

    What does move them is the nuclear radius: an ``effective_size_factor`` of 0.8
    shrinks the paper's quoted nucleus radius a second time and produces that ~12%
    RBE shortfall (see [`PaperCellInputs`][]). Configure these panels through
    ``FIGURE_INPUTS`` and pass ``effective_size_factor`` on, and Fig. 6's six
    variants land within +-2.4% of the paper.

Each curve lives in its own CSV under ``data/`` named
``elsasser2007_figure<N>_<description>.csv`` with a four-line header (metadata
keys, metadata values, x/y quantities, x/y units) then the ``x,y`` data rows;
the ``series``/``variant`` metadata records which figure line or symbol set each
point came from (``series="calc"`` for the paper's model lines, ``"data"`` for
experimental symbols).

Each figure carries the paper's own cell/target inputs (see [`FIGURE_INPUTS`][])
so the reconstruction script can configure LEM II with exactly the parameters
used in that panel.
"""

from __future__ import annotations

import csv
from collections.abc import Sequence
from dataclasses import dataclass, field
from functools import lru_cache
from importlib import resources
from typing import Literal

import numpy as np
import numpy.typing as npt

from pyradbio.benchmark.sweeps import sweep_energies
from pyradbio.core.cell import CellTarget
from pyradbio.core.photon import PhotonParameters
from pyradbio.core.radiation import Ion
from pyradbio.models.lem.base import InitialSlopeModel
from pyradbio.models.lem.lem1 import LEM1
from pyradbio.track.scholz_kraft import ScholzKraftRadialDose
from pyradbio.track.stopping_power import StoppingPower

ELSASSER2007_DOI = "10.1667/RR0467.1"

_DATA_DIR = "data"
_CURVE_PREFIX = "elsasser2007_figure"


@dataclass(frozen=True, slots=True)
class PaperCellInputs:
    """Cell/target inputs a paper panel used (LEM II photon + geometry).

    ``D_t_old`` is the original LEM I transition dose; ``D_t_new`` is the LEM II
    refit after adding the cluster extension and the refined radial dose. The
    refit is load-bearing, not cosmetic: combining both modifications *without*
    reducing ``D_t`` gives unrealistically high RBE (Fig. 6, thin dashed line).

    ``effective_size_factor`` is **1.0 and must be passed on to the**
    [`CellTarget`][]: the paper quotes a nucleus radius to
    be used as-is. ``CellTarget``'s default is 1.0 (a 0.8 would shrink the radius to
    the chromatin-bearing fraction), and the factor is still passed explicitly so
    the convention stays visible at the construction. Letting a 0.8 through shrinks
    the area by ``0.8^2 = 0.64``, inflating ``D_1 = LET/(rho*A)`` by 1.56x, so
    overkill sets in too early and every RBE_alpha peak lands ~1.25x low in LET and
    ~10-20% low in height. That single factor accounts for what would look like a
    LEM II "cluster deficit": with 0.8, Fig. 6's six variants sit at -2 to -20%
    bias; with ``1.0`` they are all within +-2.4%, and Fig. 8's V79/CHO lines
    best-fit at the paper's own stated 3.8 / 4.7 um. Same convention as
    [`PaperCellInputs`][pyradbio.benchmark.scholz1997.PaperCellInputs].
    """

    alpha_x: float  # Gy^-1
    beta_x: float  # Gy^-2
    D_t_old: float  # LEM I transition dose [Gy]
    D_t_new: float  # LEM II refit transition dose [Gy]
    nucleus_radius_um: float
    chi2_fit_let_keV_um: tuple[float, ...]  # LETs the D_t chi^2 fit used
    effective_size_factor: float = 1.0  # paper uses the quoted radius directly

    def photon(self, which: Literal["old", "new"]) -> PhotonParameters:
        """Return the photon reference at the ``"old"`` or ``"new"`` transition dose.

        ``"old"`` takes ``D_t_old`` (the original LEM I fit, Elsasser & Scholz 2007,
        doi:10.1667/RR0467.1, Fig. 6 "LEM old"); ``"new"`` takes ``D_t_new`` (the LEM
        II refit). No cluster parameters are attached: the LEM II model takes its
        cluster model as a separate argument. Equal to
        ``PhotonParameters(alpha_x=inp.alpha_x, beta_x=inp.beta_x, D_t=inp.D_t_old)``
        (or ``D_t_new``). Fig. 3's ``FIG3_D_T_OLD_GY``/``FIG3_D_T_NEW_GY`` and Fig. 7's
        stagger variants use other transition doses and stay hand-built.

        Raises:
            ValueError: if ``which`` is neither ``"old"`` nor ``"new"``.
        """
        if which == "old":
            d_t = self.D_t_old
        elif which == "new":
            d_t = self.D_t_new
        else:
            raise ValueError(f"which must be 'old' or 'new', got {which!r}")
        return PhotonParameters(alpha_x=self.alpha_x, beta_x=self.beta_x, D_t=d_t)

    def cell_target(self) -> CellTarget:
        """Return the nucleus geometry, passing ``effective_size_factor`` on explicitly.

        Elsasser & Scholz 2007 (doi:10.1667/RR0467.1), figure captions. Equal to
        ``CellTarget(nucleus_radius_um=inp.nucleus_radius_um,
        effective_size_factor=inp.effective_size_factor)``; the factor is forwarded so
        the paper's convention (its radius used as-is) stays visible.
        """
        return CellTarget(
            nucleus_radius_um=self.nucleus_radius_um,
            effective_size_factor=self.effective_size_factor,
        )


# Per-figure inputs from the Elsasser & Scholz 2007 figure captions/text.
FIGURE_INPUTS: dict[str, PaperCellInputs] = {
    # Fig. 6 caption: alpha=0.093, beta=0.026, D_t=7.5 Gy (solid lines) and
    # D_t=30 Gy (all other lines); V79 r_nucleus=3.8 um (p. 323).
    # D_t fitted at 32 and 153 keV/um for V79 (p. 325).
    "fig6_rbealpha_vs_let": PaperCellInputs(0.093, 0.026, 30.0, 7.5, 3.8, (32.0, 153.0)),
    "fig7_stagger_vs_let": PaperCellInputs(0.093, 0.026, 30.0, 7.5, 3.8, (32.0, 153.0)),
    "fig9_rbe10_vs_let": PaperCellInputs(0.093, 0.026, 30.0, 7.5, 3.8, (32.0, 153.0)),
    # Fig. 10 is V79 carbon survival at four energies -- same cell as Figs. 6/9.
    "fig10_survival_v79": PaperCellInputs(0.093, 0.026, 30.0, 7.5, 3.8, (32.0, 153.0)),
    # Fig. 12 caption + p.326: XRS repair-deficient cells, r_nucleus = 5.5 um, with a
    # PURE-EXPONENTIAL photon curve (single parameter alpha_x = 1.7 Gy^-1, beta_x = 0
    # -> NO shoulder). With beta_x = 0 the tail slope s_max = alpha_x for any D_t, so
    # D_t is a formality here (30 nominal, no refit); the whole ion RBE comes from the
    # cluster eta, making Fig. 12 the cleanest isolated test of the cluster extension.
    "fig12_xrs": PaperCellInputs(1.7, 0.0, 30.0, 30.0, 5.5, (32.0, 153.0)),
    # Fig. 8 caption: CHO alpha=0.228, beta=0.02, D_t 40 -> 9.5 Gy,
    # r_nucleus=4.7 um; fitted at 32 and 275 keV/um.
    "fig8_cho": PaperCellInputs(0.228, 0.020, 40.0, 9.5, 4.7, (32.0, 275.0)),
    # Fig. 8 also draws V79 (same caption): alpha=0.093, beta=0.026, D_t 30 -> 7.5.
    # Its two model lines reproduce Fig. 6's `lem_old`/`lem_new` to 0.3% / 1.5%
    # relRMS -- an independent check on both digitizations, since the panels were
    # traced separately.
    "fig8_v79": PaperCellInputs(0.093, 0.026, 30.0, 7.5, 3.8, (32.0, 153.0)),
    # Fig. 11 caption: HSG alpha=0.313, beta=0.0615, D_t 30 -> 6.0 Gy,
    # r_nucleus=5.0 um; fitted at 30.6 and 137 keV/um (p. 326).
    "fig11_hsg": PaperCellInputs(0.313, 0.0615, 30.0, 6.0, 5.0, (30.6, 137.0)),
}

# Fig. 3 is drawn for **CHO**, and its caption's "D_t = 30 Gy" (panel a) is a
# **paper error**: the panel actually uses CHO's old D_t of 40 Gy. Both facts are
# forced by the digitized curves, not inferred from the caption:
#
#   * the sub-D_t branches fit N = alpha*D + beta*D^2 with alpha = 0.238,
#     beta = 0.0207 -- CHO (0.228 / 0.020) within 4.5%. V79 (0.093 / 0.026) is off
#     by +156%, far outside digitization error.
#   * the dashed tails are straight (R^2 = 0.9999) with slope
#     s_max = 1.877 +/- 0.010, and s_max = alpha + 2*beta*D_t gives D_t = 39.5 Gy.
#     D_t = 30 would need s_max = 1.481 (-21%, ~40 sigma). Forcing D_t = 30 and
#     letting the LQ absorb the measured slope demands alpha = 0.053 and wrecks the
#     sub-D_t fit (41% RMS, 16x the free fit); forcing D_t = 40 recovers
#     alpha = 0.245, beta = 0.0204 at 3.8% RMS.
#
# Deriving the dashed tail at D_t = 40 reproduces it to 1.3-1.6% RMS, versus 20%
# at the caption's 30.
#
#   * and the paper says so itself. The **Fig. 8 caption** (p. 325) gives CHO
#     "D_t = 40 Gy (old) and D_t = 9.5 Gy (modified)", against V79's "D_t = 30 Gy
#     (old)" in the same sentence. So 30 Gy is V79's number, printed under a CHO
#     panel -- the Fig. 3 caption borrowed the wrong cell line's value. This is
#     external corroboration, not another fit, and it is what settles it.
#
# Redoing the whole panel with the paper's own digitized eta (so the D_t is the
# only free thing) ranks them unambiguously: on panel a's cluster branch
# D_t = 40 gives 3.6% RMS against 23.0% at 30.
FIG3_CELL_LINE = "CHO"
FIG3_D_T_OLD_GY = 40.0  # NOT the caption's 30 Gy; see above
# Panel b's 10 Gy is an *illustration*, not CHO's refit: Fig. 8 puts CHO's modified
# D_t at 9.5 Gy, and 9.5 is the WORST of the three against the digitized panel
# (15.2% RMS, vs 12.9% at the caption's 10 and 8.7% at 11). Since the panel does not
# claim to be the CHO fit and 11 is not the caption's number either, the caption's
# value is kept -- but do not read Fig. 3b as pinning a cell line's D_t.
FIG3_D_T_NEW_GY = 10.0

# Fig. 7 stagger sizes [bp] and their transition doses [Gy]. The h=10 bp curve is
# shown twice: at the shared D_t=7.5 Gy and at D_t=11 Gy, the latter chosen so
# that its low-LET RBE agrees with the original LEM (p. 325).
FIG7_STAGGER_VARIANTS: tuple[tuple[float, float], ...] = (
    (10.0, 7.5),
    (20.0, 7.5),
    (25.0, 7.5),  # the LEM II default ("LEM new")
    (35.0, 7.5),
    (10.0, 11.0),
)


@dataclass(frozen=True, slots=True)
class Elsasser2007Curve:
    """One digitized reference curve (a single figure line or symbol set)."""

    figure: str
    cell_line: str
    series: str  # "calc" (paper model line) or "data" (experimental symbols)
    variant: str  # e.g. "modified_lem", "original_lem", "cluster_h25"
    energy_MeV_u: float | None
    x_quantity: str
    x_unit: str
    y_quantity: str
    y_unit: str
    x: tuple[float, ...]
    y: tuple[float, ...]
    source_page: int
    note: str
    doi: str = ELSASSER2007_DOI
    y_err: tuple[float, ...] | None = None  # symmetric 1-sigma, when the figure drew bars
    particle: str = ""  # "helium"/"carbon"; "" where the panel has one (carbon) or no ion


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
def load_curves() -> tuple[Elsasser2007Curve, ...]:
    """Return every digitized Elsasser & Scholz 2007 benchmark curve (one per file)."""
    curves: list[Elsasser2007Curve] = []
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
            Elsasser2007Curve(
                figure=meta["figure"],
                cell_line=meta["cell_line"],
                series=meta["series"],
                variant=meta["variant"],
                energy_MeV_u=_optional_float(meta["energy_MeV_u"]),
                x_quantity=x_quantity,
                x_unit=x_unit,
                y_quantity=y_quantity,
                y_unit=y_unit,
                x=tuple(x for x, _, _ in points),
                y=tuple(y for _, y, _ in points),
                y_err=tuple(e for _, _, e in points) if has_err else None,
                particle=meta.get("particle", ""),
                source_page=int(meta["source_page"]),
                note=meta["note"],
                doi=meta.get("doi", ELSASSER2007_DOI),
            )
        )
    return tuple(
        sorted(
            curves,
            key=lambda c: (
                c.figure,
                c.cell_line,
                c.series,
                c.variant,
                c.energy_MeV_u if c.energy_MeV_u is not None else -1.0,
            ),
        )
    )


def iter_curves(
    *,
    figure: str | None = None,
    cell_line: str | None = None,
    particle: str | None = None,
    series: str | None = None,
    variant: str | None = None,
    energy_MeV_u: float | None = None,
) -> tuple[Elsasser2007Curve, ...]:
    """Filter digitized Elsasser & Scholz 2007 curves.

    ``series="calc"`` selects the paper's own model lines; ``series="data"``
    selects the experimental symbols redrawn in the figures.
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
        and (energy_MeV_u is None or c.energy_MeV_u == energy_MeV_u)
    )


def get_curve(
    figure: str,
    *,
    cell_line: str | None = None,
    particle: str | None = None,
    series: str | None = None,
    variant: str | None = None,
    energy_MeV_u: float | None = None,
) -> Elsasser2007Curve:
    """Return exactly one Elsasser & Scholz 2007 reference curve.

    Raises ``LookupError`` if the filters select zero or multiple curves.
    """
    matches = iter_curves(
        figure=figure,
        cell_line=cell_line,
        particle=particle,
        series=series,
        variant=variant,
        energy_MeV_u=energy_MeV_u,
    )
    if len(matches) != 1:
        raise LookupError(f"expected exactly one Elsasser & Scholz 2007 curve, got {len(matches)}")
    return matches[0]


# --- reconstruction machinery ---------------------------------------------------


@dataclass(frozen=True, eq=False)
class DigitizedEtaCluster:
    """A cluster response whose ``eta(D)`` is the paper's *digitized* Fig. 2.

    Swapped in for [`ClusterModel`][pyradbio.models.lem.lem2.ClusterModel], it turns Fig. 3
    into a test of eq. 10 **alone** (Elsasser & Scholz 2007, doi:10.1667/RR0467.1):
    the photon inputs and ``eta`` are then both the paper's own, so any residual is
    the formula's fault rather than our clustering rule's. Implements
    [`ClusterResponse`][pyradbio.models.lem.lem2.ClusterResponse], so it can be passed as
    ``LEM2(cluster=...)``.

    ``eta`` is 1 at and below the first digitized dose and, above it, linear in
    ``log10 D`` between the digitized points (flat past the last one) -- the
    interpolation the reconstruction and ``tests/benchmark/test_elsasser2007.py``
    used inline. ``eta_array`` is the same rule element-wise. ``kernel_eta_table``
    hands the compiled MC kernel the digitized nodes; the kernel clamps to the
    first digitized ``eta`` (~1.0, not exactly 1) below the first node.
    Equality is identity (``eq=False``): array fields have no useful ``==``.
    """

    dose_Gy: npt.NDArray[np.float64]
    eta_values: npt.NDArray[np.float64]
    _log10_dose: npt.NDArray[np.float64] = field(init=False, repr=False)

    def __post_init__(self) -> None:
        dose = np.array(self.dose_Gy, dtype=float)
        eta = np.array(self.eta_values, dtype=float)
        if dose.ndim != 1 or dose.shape != eta.shape or len(dose) < 2:
            raise ValueError("dose_Gy and eta_values must be equal-length 1-D arrays (>= 2)")
        if dose[0] <= 0.0 or np.any(np.diff(dose) <= 0.0):
            raise ValueError("dose_Gy must be positive and strictly increasing")
        object.__setattr__(self, "dose_Gy", dose)
        object.__setattr__(self, "eta_values", eta)
        object.__setattr__(self, "_log10_dose", np.log10(dose))

    def eta(self, dose_Gy: float) -> float:
        """Digitized ``eta(D)``: 1 at/below the first point, log-dose interpolation above."""
        if dose_Gy <= self.dose_Gy[0]:
            return 1.0
        return float(np.interp(np.log10(dose_Gy), self._log10_dose, self.eta_values))

    def eta_array(self, dose_Gy: npt.NDArray[np.float64]) -> npt.NDArray[np.float64]:
        """Vectorised [`eta`][] (same floor, same interpolation)."""
        dose = np.asarray(dose_Gy, dtype=np.float64)
        with np.errstate(divide="ignore", invalid="ignore"):
            interpolated = np.interp(np.log10(dose), self._log10_dose, self.eta_values)
        return np.asarray(np.where(dose <= self.dose_Gy[0], 1.0, interpolated), dtype=np.float64)

    def kernel_eta_table(self) -> tuple[npt.NDArray[np.float64], npt.NDArray[np.float64]]:
        """``(log10 dose, eta)`` nodes for the compiled Monte-Carlo kernel."""
        return self._log10_dose, self.eta_values

    @property
    def eta_max(self) -> float:
        """Peak of the digitized curve (14.79 at ~4.4e5 Gy for the bundled Fig. 2)."""
        return float(self.eta_values.max())


def digitized_eta_cluster() -> DigitizedEtaCluster:
    """The bundled Fig. 2 ``eta(D)`` digitization as a LEM II cluster response.

    ``DigitizedEtaCluster(dose_Gy=fig2.x, eta_values=fig2.y)`` with
    ``fig2 = get_curve("fig2_eta_vs_dose")`` (Elsasser & Scholz 2007,
    doi:10.1667/RR0467.1, Fig. 2). Feeding it to ``LEM2(cluster=...)`` and calling
    [`photon_log_kill`][pyradbio.models.lem.lem2.LEM2.photon_log_kill] derives Fig. 3 from the
    paper's own ``eta``.
    """
    fig2 = get_curve("fig2_eta_vs_dose")
    return DigitizedEtaCluster(dose_Gy=np.array(fig2.x), eta_values=np.array(fig2.y))


METHOD_BANDS_KEV_UM: tuple[tuple[float, float], ...] = (
    (13.0, 30.0),
    (30.0, 70.0),
    (70.0, 150.0),
    (150.0, 300.0),
    (300.0, 600.0),
)
"""LET bands [keV/um] of the eq. (8) method diagnostic, low LET (largest
``r_max/R_eff``) first.
"""

_METHOD_MODES: tuple[InitialSlopeModel, InitialSlopeModel] = (
    "lethal_central_saturating",
    "cross_section",
)
"""The two initial-slope methods the diagnostic compares, in [`MethodBand`][] order."""


@dataclass(frozen=True)
class MethodBand:
    """One LET band of [`paper_method_bands`][].

    ``n`` reference points fall in ``[let_lo_keV_um, let_hi_keV_um)``;
    ``r_max_over_r_eff`` is the carbon track's ``r_max`` at the band's geometric-mean
    LET over the effective nuclear radius; ``bias_eq8`` / ``bias_exact`` are the mean
    ``ours / reference - 1`` of ``RBE_alpha`` under the paper's eq. (8)
    (``lethal_central_saturating``) and the exact impact-parameter integral
    (``cross_section``).
    """

    let_lo_keV_um: float
    let_hi_keV_um: float
    n: int
    r_max_over_r_eff: float
    bias_eq8: float
    bias_exact: float


def _interp_log_let(
    let: npt.NDArray[np.float64], value: npt.NDArray[np.float64], x: npt.NDArray[np.float64]
) -> npt.NDArray[np.float64]:
    """A model curve on the reference LETs (log-LET interpolation, NaN outside)."""
    out: npt.NDArray[np.float64] = np.interp(
        np.log(x), np.log(let), value, left=np.nan, right=np.nan
    )
    return out


def paper_method_bands(
    photon: PhotonParameters,
    cell: CellTarget,
    radial_dose: ScholzKraftRadialDose,
    *,
    stopping_power: StoppingPower,
    energies_MeV_u: npt.ArrayLike,
    reference: Elsasser2007Curve | None = None,
    bands: Sequence[tuple[float, float]] = METHOD_BANDS_KEV_UM,
) -> tuple[MethodBand, ...]:
    """Does the paper's stated eq. (8) reproduce its own ``RBE_alpha`` curves?

    Elsasser & Scholz 2007 (doi:10.1667/RR0467.1, p. 323) state one method for every
    ``RBE_alpha`` curve from Fig. 6 on: a single traversal through the nucleus
    centre, ``alpha_I = (1 - S_1)/D_1`` (eq. 8). That drops every track that misses
    the nucleus while still depositing dose in it, so its cost grows with
    ``r_max/R_eff``. If the paper's lines had been made with eq. (8), our eq.-(8)
    residual would be flat in ``r_max/R_eff``; if they were made with the full
    impact-parameter average, the eq.-(8) residual grows with it and the exact one
    stays flat.

    Two carbon LEM I models (``photon``, ``cell``, ``radial_dose``,
    ``stopping_power``; ``initial_slope_model`` ``"lethal_central_saturating"`` and
    ``"cross_section"``) are swept over ``energies_MeV_u`` and sorted by LET
    ([`sweep_energies`][]); their
    ``RBE_alpha = alpha / photon.alpha_x`` is interpolated in log LET onto the
    ``reference`` points (default: Fig. 6 ``lem_old``, the simplest line in the
    paper: no clusters, no convolution, LQ photon input), NaN outside the swept LET
    range. Each band of ``bands`` holding at least two reference points gives one
    [`MethodBand`][]; bands with fewer are skipped. The expression order is the
    reconstruction script's, so its printed table is unchanged.
    """
    ref = (
        reference if reference is not None else get_curve("fig6_rbealpha_vs_let", variant="lem_old")
    )
    rx, ry = np.array(ref.x), np.array(ref.y)
    curves = {
        mode: sweep_energies(
            LEM1(
                photon=photon,
                cell=cell,
                stopping_power=stopping_power,
                radial_dose=radial_dose,
                initial_slope_model=mode,
            ),
            6,
            12,
            energies_MeV_u,
            stopping_power=stopping_power,
            on_error="raise",
            sort_by_let=True,
        )
        for mode in _METHOD_MODES
    }
    let_sorted = curves[_METHOD_MODES[0]].let_keV_um
    energy_sorted = curves[_METHOD_MODES[0]].energy_MeV_u
    out: list[MethodBand] = []
    for lo, hi in bands:
        m = (rx >= lo) & (rx < hi)
        if m.sum() < 2:
            continue
        mid = float(np.exp(np.mean(np.log(rx[m]))))
        e_mid = float(np.interp(mid, let_sorted, energy_sorted))
        ratio = radial_dose.r_max_um(Ion(Z=6, A=12, energy_MeV_u=e_mid)) / (
            cell.effective_nucleus_radius_um()
        )
        bias = [
            float(
                np.mean(
                    _interp_log_let(
                        curves[mode].let_keV_um, curves[mode].rbe_alpha(photon.alpha_x), rx[m]
                    )
                    / ry[m]
                    - 1
                )
            )
            for mode in _METHOD_MODES
        ]
        out.append(MethodBand(lo, hi, int(m.sum()), ratio, bias[0], bias[1]))
    return tuple(out)


__all__ = [
    "ELSASSER2007_DOI",
    "FIG3_CELL_LINE",
    "FIG3_D_T_NEW_GY",
    "FIG3_D_T_OLD_GY",
    "FIG7_STAGGER_VARIANTS",
    "FIGURE_INPUTS",
    "METHOD_BANDS_KEV_UM",
    "DigitizedEtaCluster",
    "Elsasser2007Curve",
    "MethodBand",
    "PaperCellInputs",
    "digitized_eta_cluster",
    "get_curve",
    "iter_curves",
    "load_curves",
    "paper_method_bands",
]
