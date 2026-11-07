# ---
# jupyter:
#   jupytext:
#     formats: ipynb,py:percent
#     text_representation:
#       extension: .py
#       format_name: percent
#       format_version: '1.3'
#   kernelspec:
#     display_name: Python 3
#     language: python
#     name: python3
# ---

# %% [markdown]
# # LEM II: cluster effects and the paper figure set
#
# Elsasser & Scholz 2007 (doi:10.1667/RR0467.1) introduced the LEM II
# refinements: a high-local-dose SSB cluster enhancement of the photon
# dose-effect curve, and a refined radial dose profile with a 0.3 nm core plus
# radical diffusion. This script reconstructs the paper's panels and reports
# residuals against **hand-digitized reference data**
# (`pyradbio.benchmark.elsasser2007`, whose docstring carries the per-panel
# provenance; the findings are in `docs/references/LEM_II_Elsasser2007.md`).
#
# | panel | status |
# |---|---|
# | Fig. 2 — damage enhancement `eta(D)` | digitized; **model disagrees** (Fig. 2 is the outlier) |
# | Fig. 3 — modified photon dose-effect | digitized; **derived** two ways (paper eta / our eta) |
# | Fig. 4 — radial dose vs **experiment** (O + Ne) | digitized; absolute Gy vs r in cm |
# | Fig. 5 — old/new radial dose profiles | model only (arbitrary units) |
# | Fig. 6 — `RBE_alpha` vs LET per modification | digitized (all 6 modification curves) |
# | Fig. 7 — stagger-size sensitivity | digitized (5 stagger variants + both LEM references) |
# | Fig. 8 — `RBE_alpha` for V79 **and** CHO | digitized (4 lines + both data sets w/ error bars) |
# | Fig. 9 — RBE at 10% survival | digitized (2 lines + V79 data) |
# | Fig. 10 — V79 carbon survival, 4 energies | digitized (4 panels overlaid; β test) |
# | Fig. 11 — initial slope `alpha`, HSG, **He + C** | digitized (4 lines + both data sets) |
# | Fig. 12 — XRS survival (isolated cluster test) | digitized; pure-exp photon, RBE = all `eta` |
#
# **Fig. 3 needs no digitization**: it *is* `LEM2.photon_log_kill` (eq. 10, `eta`
# **multiplies** the dose), fixed by Fig. 2's `eta` plus the photon inputs; the
# panel is **CHO** with `D_t = 40` Gy, not the caption's 30.
#
# **Known discrepancy (Fig. 2).** `eta - 1 = (a_SSB/a_DSB) * q(mu)/mu` with
# `mu = a_SSB*D*h/G`, so genome size and stagger cannot change the peak *height*:
# the paper peaks at 14.79, our fixed-`h`-bin rule at 13.40, and Friedrich 2015's
# independent closed form at 13.51 — Fig. 2 is the outlier. Fig. 2 was digitized
# twice independently (by hand and by raster auto-trace, agreeing to 0.9% RMS), so
# the mismatch is not a digitization artifact.
#
# Quick mode (`PYRADBIO_QUICK=1`) thins the energy grids and the genome-size
# scan; only the full run's numbers are the reference.

# %%
from __future__ import annotations

from collections.abc import Callable
from typing import Any

import matplotlib.colors as mcolors
import matplotlib.pyplot as plt
import numpy as np
import numpy.typing as npt
from _common import pick, quick_mode

from pyradbio import CellTarget, Ion, PhotonParameters
from pyradbio.benchmark import elsasser2007 as es
from pyradbio.benchmark.metrics import ShapeResiduals, ratio_summary, shape_residuals
from pyradbio.benchmark.sweeps import LetSweep, sweep_energies
from pyradbio.core.lq import survival
from pyradbio.core.photon import ClusterParameters
from pyradbio.models.lem.base import InitialSlopeModel
from pyradbio.models.lem.lem1 import LEM1
from pyradbio.models.lem.lem2 import LEM2, ClusterModel, ClusterResponse, Friedrich2015ClusterModel
from pyradbio.plotting import figures_dir, save_figure
from pyradbio.track.scholz_kraft import LEM2_RMIN_NM, R_MAX_GAMMA_UM, ScholzKraftRadialDose
from pyradbio.track.stopping_power import default_stopping_power, describe_stopping_power

FIGURES = figures_dir(globals().get("__file__"))
if quick_mode():
    print("QUICK MODE (PYRADBIO_QUICK=1): 12/20-point energy grids, 2 genome sizes")

#: Carbon energy grid of every RBE panel, and the wider Fig. 11 grid (helium
#: reaches a given LET at a much lower energy than carbon).
ENERGIES_MEV_U = np.geomspace(1.5, 350.0, pick(48, 12))
FIG11_ENERGIES_MEV_U = np.geomspace(0.8, 400.0, pick(90, 20))
GENOME_SIZES_BP = pick((3.0e9, 5.4e9, 6.0e9, 1.2e10), (3.0e9, 1.2e10))

sp = default_stopping_power()
STOPPING_LABEL = describe_stopping_power(sp)
print(f"stopping power: {STOPPING_LABEL}")

# %% [markdown]
# ## Paper inputs and models
#
# V79 from the Fig. 6 caption: `alpha_x = 0.093 Gy^-1`, `beta_x = 0.026 Gy^-2`,
# `r_nucleus = 3.8 um`; *both* solid lines (thick "LEM new", thin "old LEM,
# reduced `D_t`") use the refit `D_t = 7.5 Gy`, all others 30 Gy.
#
# The paper's "original" LEM is the original *physics* (r_min = 10 nm, no
# radical diffusion, no clusters) with THIS paper's track extent,
# `r_max = 0.062 E^1.7` (p. 320), not `ScholzKraftRadialDose.lem1()`'s legacy 0.05.
# The stated value is used on the paper's authority: which gamma "fits" flips with
# the initial-slope method (docs: "gamma: use the stated 0.062").
#
# Every initial slope, LEM I and LEM II alike, uses the paper's stated procedure
# (p. 323): one traversal through the nucleus centre, eq. (8)
# `alpha_I = (1 - S_1)/D_1` = `lethal_central_saturating`. A reconstruction runs
# the paper's method before it says anything about the paper's parameters; set
# `INITIAL_SLOPE_MODEL = "cross_section"` (the exact impact-parameter integral) to
# see what the approximation costs.

# %%
V79 = es.FIGURE_INPUTS["fig6_rbealpha_vs_let"]
cell = V79.cell_target()
photon_old = V79.photon("old")
photon_new = V79.photon("new")

old_radial = ScholzKraftRadialDose(r_min_nm=10.0, stopping_power=sp, r_max_gamma_um=R_MAX_GAMMA_UM)
new_radial = ScholzKraftRadialDose.lem2(sp)
cluster = ClusterModel()
INITIAL_SLOPE_MODEL: InitialSlopeModel = "lethal_central_saturating"


def lem1(
    photon: PhotonParameters, target: CellTarget, radial: ScholzKraftRadialDose = old_radial
) -> LEM1:
    """LEM I with the paper's gamma = 0.062 track and the paper's slope method."""
    return LEM1(
        photon=photon,
        cell=target,
        stopping_power=sp,
        radial_dose=radial,
        initial_slope_model=INITIAL_SLOPE_MODEL,
    )


def lem2(photon: PhotonParameters, target: CellTarget, **kw: Any) -> LEM2:
    """LEM II (default radial dose unless ``radial_dose`` is given) with ``cluster``."""
    kw.setdefault("cluster", cluster)
    return LEM2(
        photon=photon,
        cell=target,
        stopping_power=sp,
        initial_slope_model=INITIAL_SLOPE_MODEL,
        **kw,
    )


# The five Fig. 6 modifications plus "LEM new"; only the two solid lines carry the
# refit D_t.
lem_old = lem1(photon_old, cell)
lem_reduced_dt = lem1(photon_new, cell)
lem_clusters = lem2(photon_old, cell, radial_dose=old_radial)
lem_radial = lem1(photon_old, cell, radial=new_radial)
lem_radial_clusters = lem2(photon_old, cell)
lem_new = lem2(photon_new, cell)

# %% [markdown]
# ## Residual / shape diagnostics
#
# pyRadBio is sampled at the paper's own x-values: a *point* residual (relRMS,
# bias) plus a *shape* diagnostic (`benchmark.metrics.shape_residuals`; `shape`
# near 0 = a uniform scale offset, near 1 = the shapes differ). Residuals are
# *relative*, since Fig. 3 spans eight decades of lethal events. Model curves are
# carbon (or Fig. 11's helium) energy sweeps converted to LET and interpolated in
# log LET; LET falls monotonically with energy here, so a curve is single-valued.


# %%
def compare(
    px: npt.ArrayLike, py: npt.ArrayLike, model_fn: Callable[[np.ndarray], np.ndarray]
) -> ShapeResiduals:
    """Point + shape residuals of ``model_fn`` against one digitized paper curve."""
    px = np.asarray(px, dtype=float)
    py = np.asarray(py, dtype=float)
    return shape_residuals(py, model_fn(px))


def print_compare(title: str, stats: ShapeResiduals) -> None:
    print(
        f"  {title:32s} n={stats.n:3d}  relRMS={stats.rel_rms:6.1%}  "
        f"bias={stats.rel_bias:+.1%}  max|rel|={stats.max_rel:.1%}  "
        f"scale k={stats.scale_k:.3f}  shape={stats.shape_fraction:.2f}"
    )


def carbon_sweep(model: LEM1 | LEM2) -> LetSweep:
    """Carbon over ``ENERGIES_MEV_U``, sorted by LET."""
    return sweep_energies(
        model, 6, 12, ENERGIES_MEV_U, stopping_power=sp, on_error="raise", sort_by_let=True
    )


def rbe_alpha_curve(model: LEM1 | LEM2) -> tuple[np.ndarray, np.ndarray]:
    """``(LET, RBE_alpha)`` of carbon, sorted by LET."""
    sweep = carbon_sweep(model)
    return sweep.let_keV_um, sweep.rbe_alpha(model.photon.alpha_x)


def alpha_vs_let(model: LEM1 | LEM2, z: int, a: int) -> tuple[np.ndarray, np.ndarray]:
    """``(LET, alpha)`` of any ion over the Fig. 11 grid, sorted by LET."""
    sweep = sweep_energies(
        model, z, a, FIG11_ENERGIES_MEV_U, stopping_power=sp, on_error="raise", sort_by_let=True
    )
    return sweep.let_keV_um, sweep.alpha


def interp_on(let: np.ndarray, value: np.ndarray) -> Callable[[np.ndarray], np.ndarray]:
    """Interpolate a model curve onto the paper's LET values (log-LET spacing)."""

    def fn(x: np.ndarray) -> np.ndarray:
        return np.interp(np.log(x), np.log(let), value, left=np.nan, right=np.nan)

    return fn


def curve_xy(curve: es.Elsasser2007Curve) -> tuple[np.ndarray, np.ndarray]:
    return np.array(curve.x), np.array(curve.y)


# %% [markdown]
# ## Fig. 10 caption cross-check: stopping power
#
# The Fig. 10 caption pins four carbon energy/LET pairs, which validates our
# stopping power *independently of LEM*.

# %%
print("\n=== Fig. 10 caption (E -> LET) cross-check ===")
for energy, paper_let in ((4.2, 339.0), (11.0, 153.5), (76.9, 32.5), (266.4, 13.7)):
    ours = sp.let_keV_um(Ion(Z=6, A=12, energy_MeV_u=energy))
    print(
        f"  {energy:6.1f} MeV/u: paper {paper_let:6.1f}  ours {ours:6.1f} keV/um "
        f"({(ours - paper_let) / paper_let:+.1%})"
    )

# %% [markdown]
# ## Fig. 2: eta(D), ours vs the paper vs Friedrich 2015
#
# Friedrich, Durante & Scholz 2015 (doi:10.1093/rpd/ncv147) recompute the same
# quantity from the same alpha_SSB = 1250, alpha_DSB = 30, t = 25 bp with a closed
# form and a stricter coincidence rule (it drops SSBs shadowed by a real DSB,
# requires opposite strands and stops one SSB serving two DSBs; see
# `Friedrich2015ClusterModel`). That third opinion is what decides whether "our
# eta is low" is our defect or Fig. 2's. Its own genome size is 5.4e9 bp (moves
# the dose axis, not the peak height). A track centre never exceeds ~1e5 Gy (its
# Discussion), so only the rising part of eta(D) is ever sampled by LEM.

# %%
eta_x, eta_paper = curve_xy(es.get_curve("fig2_eta_vs_dose"))
eta_model_fn = lambda d: np.array([cluster.eta(float(v)) for v in d])  # noqa: E731
eta_stats = compare(eta_x, eta_paper, eta_model_fn)
doses = np.logspace(0, 7, 400)
eta_dense = eta_model_fn(doses)

eta_f2015 = Friedrich2015ClusterModel(params=ClusterParameters(genome_size_bp=5.4e9))
eta_f2015_fn = lambda d: np.array([eta_f2015.eta(float(v)) for v in d])  # noqa: E731
eta_f2015_dense = eta_f2015_fn(doses)
eta_f2015_stats = compare(eta_x, eta_paper, eta_f2015_fn)
ETA_BANDS = ((1e2, 1e4, "onset"), (1e4, 1e5, "therapy"), (1e5, 1e6, "peak"), (1e6, 1e8, "tail"))
eta_band_stats = []
for lo, hi, band in ETA_BANDS:
    m = (eta_x >= lo) & (eta_x < hi)
    if m.sum() < 2:
        continue
    ours_b, f_b, ref_b = eta_model_fn(eta_x[m]), eta_f2015_fn(eta_x[m]), eta_paper[m]
    eta_band_stats.append(
        (
            band,
            lo,
            hi,
            int(m.sum()),
            float(np.mean(ours_b / ref_b - 1)),
            float(np.mean(f_b / ref_b - 1)),
            float(np.mean(f_b / ours_b - 1)),
        )
    )

# %% [markdown]
# ## Fig. 3: the derived photon dose-effect (CHO)
#
# Two derivations per panel: our `ClusterModel` eta, and the paper's digitized
# eta (`es.digitized_eta_cluster()`), which turns the panel into a test of eq. 10
# alone. The difference between them is exactly the eta gap. Each digitized
# branch has its own reference: `lq_shoulder` is the plain LQ (to the *old*
# `D_t` = 40 Gy in both panels), `original` the un-enhanced LEM I tail, and only
# `cluster` exercises eq. 10.

# %%
CHO = es.FIGURE_INPUTS["fig8_cho"]
fig3_cell = CHO.cell_target()
photon_fig3_old = PhotonParameters(alpha_x=CHO.alpha_x, beta_x=CHO.beta_x, D_t=es.FIG3_D_T_OLD_GY)
photon_fig3_new = PhotonParameters(alpha_x=CHO.alpha_x, beta_x=CHO.beta_x, D_t=es.FIG3_D_T_NEW_GY)
paper_cluster = es.digitized_eta_cluster()


def fig3_model(photon: PhotonParameters, response: ClusterResponse) -> LEM2:
    return LEM2(photon=photon, cell=fig3_cell, stopping_power=sp, cluster=response)


lem_fig3_a_ours = fig3_model(photon_fig3_old, cluster)
lem_fig3_b_ours = fig3_model(photon_fig3_new, cluster)
lem_fig3_a_paper = fig3_model(photon_fig3_old, paper_cluster)
lem_fig3_b_paper = fig3_model(photon_fig3_new, paper_cluster)


def photon_tail(model: LEM2, d: np.ndarray) -> np.ndarray:
    return np.array([model.photon_log_kill(float(v)) for v in d])


fig3_doses = np.logspace(0, 6, 300)
fig3_lq = CHO.alpha_x * fig3_doses + CHO.beta_x * fig3_doses**2
fig3_old_tail = np.array([photon_fig3_old.log_kill(float(d)) for d in fig3_doses])
fig3_a_ours = photon_tail(lem_fig3_a_ours, fig3_doses)
fig3_a_paper = photon_tail(lem_fig3_a_paper, fig3_doses)
fig3_b_ours = photon_tail(lem_fig3_b_ours, fig3_doses)
fig3_b_paper = photon_tail(lem_fig3_b_paper, fig3_doses)
fig3_ref = {
    (panel, variant): es.get_curve(f"fig3{panel}_dose_effect", variant=variant)
    for panel in ("a", "b")
    for variant in ("lq_shoulder", "cluster", "original")
}


def fig3_stats(panel: str, variant: str, model: LEM2) -> ShapeResiduals:
    """Residuals of one digitized Fig. 3 branch against the matching derivation."""
    curve = fig3_ref[(panel, variant)]
    if variant == "lq_shoulder":
        fn: Callable[[np.ndarray], np.ndarray] = lambda d: CHO.alpha_x * d + CHO.beta_x * d**2  # noqa: E731
    elif variant == "original":
        fn = lambda d: np.array([photon_fig3_old.log_kill(float(v)) for v in d])  # noqa: E731
    else:
        fn = lambda d: photon_tail(model, d)  # noqa: E731
    return compare(np.array(curve.x), np.array(curve.y), fn)


# %% [markdown]
# ## Figs. 6-7: RBE_alpha per modification and per stagger size
#
# Fig. 7 varies only the stagger size `h`, the cluster extension's single free
# geometric parameter; its `lem_old` is Fig. 6's original LEM repeated. Each model
# is keyed on the benchmark's `variant`, so it pairs with its digitized curve.

# %%
FIG6_CURVES = {
    "lem_old": ("LEM old", lem_old, "0.25", {"ls": "--", "lw": 2.0}),
    "reduced_dt": ("old LEM, reduced $D_t$", lem_reduced_dt, "C4", {"ls": "-", "lw": 1.0}),
    "clusters": ("LEM+clusters", lem_clusters, "C2", {"ls": "-.", "lw": 1.2}),
    "radial": ("LEM+D(r)", lem_radial, "C1", {"ls": ":", "lw": 1.6}),
    "radial_clusters": (
        "LEM+D(r)+clusters",
        lem_radial_clusters,
        "C0",
        {"ls": (0, (4, 2)), "lw": 1.0},
    ),
    "lem_new": ("LEM new", lem_new, "C3", {"ls": "-", "lw": 2.4}),
}
fig6_sweeps = {v: carbon_sweep(m) for v, (_, m, _c, _s) in FIG6_CURVES.items()}
fig6 = {
    v: (fig6_sweeps[v].let_keV_um, fig6_sweeps[v].rbe_alpha(m.photon.alpha_x))
    for v, (_, m, _c, _s) in FIG6_CURVES.items()
}
fig6_ref = {v: es.get_curve("fig6_rbealpha_vs_let", variant=v) for v in FIG6_CURVES}
fig6_stats = {v: compare(*curve_xy(fig6_ref[v]), interp_on(*fig6[v])) for v in FIG6_CURVES}

FIG7_VARIANTS = {
    "cluster_h10": (10.0, 7.5, "$h$ = 10 bp", {"ls": "--", "lw": 1.0}),
    "cluster_h20": (20.0, 7.5, "$h$ = 20 bp", {"ls": "-.", "lw": 1.2}),
    "lem_new": (25.0, 7.5, "$h$ = 25 bp (LEM new)", {"ls": "-", "lw": 2.4}),
    "cluster_h35": (35.0, 7.5, "$h$ = 35 bp", {"ls": ":", "lw": 1.4}),
    "cluster_h10_dt11": (10.0, 11.0, "$h$ = 10 bp, $D_t$ = 11 Gy", {"ls": (0, (5, 2)), "lw": 1.8}),
}
fig7 = {}
for variant, (stagger_bp, d_t, _label, _style) in FIG7_VARIANTS.items():
    model = lem2(
        PhotonParameters(alpha_x=V79.alpha_x, beta_x=V79.beta_x, D_t=d_t),
        cell,
        cluster=ClusterModel(params=ClusterParameters(stagger_bp=stagger_bp)),
    )
    fig7[variant] = rbe_alpha_curve(model)
fig7["lem_old"] = fig6["lem_old"]
fig7_ref = {v: es.get_curve("fig7_stagger_vs_let", variant=v) for v in fig7}
fig7_stats = {v: compare(*curve_xy(fig7_ref[v]), interp_on(*fig7[v])) for v in fig7}

# %% [markdown]
# ## Method diagnostic and genome size
#
# Does the paper's eq. (8) reproduce the paper's curves? Eq. (8) drops every track
# that misses the nucleus while still depositing dose in it, a cost governed by
# `r_max/R_eff` (~84 in the 13-30 keV/um band down to 0.1 at 300-600). Measured on Fig. 6's
# `lem_old`, the simplest line in the paper (`es.paper_method_bands`).
#
# The genome size is the one cluster input the paper never states ("only the
# genomic size is required", p. 321); we default to 6.0e9 bp, the same group's
# later convention (Friedrich 2012 `S_G = 6e9`; Friedrich 2015 uses 5.4e9). It
# cancels from Fig. 2's peak height but rescales the dose axis, which moves RBE.
# It is scored on the LEM II / LEM I *ratio* (cancels stopping power, r_max,
# effective area and slope method to first order), only over the LET range both
# digitized lines cover.

# %%
method_bands = es.paper_method_bands(
    photon_old,
    cell,
    old_radial,
    stopping_power=sp,
    energies_MeV_u=ENERGIES_MEV_U,
    reference=fig6_ref["lem_old"],
)

_g_lo = max(min(fig6_ref["lem_new"].x), min(fig6_ref["lem_old"].x))
_g_hi = min(max(fig6_ref["lem_new"].x), max(fig6_ref["lem_old"].x))
_g_px = np.geomspace(_g_lo, _g_hi, 40)
_g_paper_ratio = interp_on(*curve_xy(fig6_ref["lem_new"]))(_g_px) / interp_on(
    *curve_xy(fig6_ref["lem_old"])
)(_g_px)
_g_ours_old = interp_on(*fig6["lem_old"])(_g_px)
genome_bands = []
for _g in GENOME_SIZES_BP:
    _g_model = lem2(
        photon_new, cell, cluster=ClusterModel(params=ClusterParameters(genome_size_bp=_g))
    )
    _g_ratio = interp_on(*rbe_alpha_curve(_g_model))(_g_px) / _g_ours_old
    _ok = np.isfinite(_g_ratio) & np.isfinite(_g_paper_ratio)
    genome_bands.append(
        (
            _g,
            _g_model.resolved_cluster.eta_max,
            ratio_summary(_g_ratio, _g_paper_ratio, mask=_ok).mean_rel,
        )
    )

# %% [markdown]
# ## Figs. 8, 9, 11: two more cell lines, RBE at 10%, and helium
#
# Fig. 8 draws V79 *and* CHO from one code path, so it pins the nuclear geometry
# (radii 3.8 vs 4.7 um, used as-is: `cell_target()` passes
# `effective_size_factor = 1.0`). Its V79 lines are Fig. 6's, traced separately.
# Fig. 9 reuses the Fig. 6 `lem_old`/`lem_new` sweeps (RBE at 10% against the plain
# V79 LQ). Fig. 11 (HSG, 5.0 um) is the one panel with a second ion; Furusawa 2000
# used 3He, and alpha(LET) is identical for 3He and 4He (a given LET fixes the
# velocity). Its data x is Furusawa's *dose-averaged* LET (p. 326).

# %%
fig8_models: dict[tuple[str, str], LEM1 | LEM2] = {}
for line, inp in (("V79", es.FIGURE_INPUTS["fig8_v79"]), ("CHO", es.FIGURE_INPUTS["fig8_cho"])):
    fig8_models[line, "original_lem"] = lem1(inp.photon("old"), inp.cell_target())
    fig8_models[line, "modified_lem"] = lem2(inp.photon("new"), inp.cell_target())
fig8 = {key: rbe_alpha_curve(m) for key, m in fig8_models.items()}
fig8_ref = {
    key: es.get_curve("fig8_rbealpha_vs_let", cell_line=key[0], series="calc", variant=key[1])
    for key in fig8_models
}
fig8_stats = {key: compare(*curve_xy(fig8_ref[key]), interp_on(*fig8[key])) for key in fig8_models}
fig8_data = {
    line: es.get_curve("fig8_rbealpha_vs_let", cell_line=line, series="data")
    for line in ("V79", "CHO")
}
fig8_vs_fig6 = {
    f6: compare(*curve_xy(fig6_ref[f6]), interp_on(*curve_xy(fig8_ref["V79", f8])))
    for f6, f8 in (("lem_old", "original_lem"), ("lem_new", "modified_lem"))
}

fig9_mod_ref = es.get_curve("fig9_rbe10_vs_let", series="calc", variant="modified_lem")
fig9_orig_ref = es.get_curve("fig9_rbe10_vs_let", series="calc", variant="original_lem")
fig9_data = es.get_curve("fig9_rbe10_vs_let", series="data")
let_new, rbe10_new = (
    fig6_sweeps["lem_new"].let_keV_um,
    fig6_sweeps["lem_new"].rbe10_lq(V79.alpha_x, V79.beta_x),
)
let_old, rbe10_old = (
    fig6_sweeps["lem_old"].let_keV_um,
    fig6_sweeps["lem_old"].rbe10_lq(V79.alpha_x, V79.beta_x),
)
fig9_new_stats = compare(*curve_xy(fig9_mod_ref), interp_on(let_new, rbe10_new))
fig9_old_stats = compare(*curve_xy(fig9_orig_ref), interp_on(let_old, rbe10_old))

HSG = es.FIGURE_INPUTS["fig11_hsg"]
fig11_models: dict[str, LEM1 | LEM2] = {
    "original_lem": lem1(HSG.photon("old"), HSG.cell_target()),
    "modified_lem": lem2(HSG.photon("new"), HSG.cell_target()),
}
FIG11_IONS = {"helium": (2, 3), "carbon": (6, 12)}
fig11 = {
    (particle, variant): alpha_vs_let(model, *zna)
    for particle, zna in FIG11_IONS.items()
    for variant, model in fig11_models.items()
}
fig11_ref = {
    key: es.get_curve("fig11_alpha_vs_let", particle=key[0], series="calc", variant=key[1])
    for key in fig11
}
fig11_stats = {key: compare(*curve_xy(fig11_ref[key]), interp_on(*fig11[key])) for key in fig11}
fig11_data = {
    particle: es.get_curve("fig11_alpha_vs_let", particle=particle, series="data")
    for particle in FIG11_IONS
}
# The paper's Fig. 11 claim is about agreement with the Furusawa *data*: score both
# our models and the paper's own lines against the experimental points.
fig11_vs_data = {
    (particle, variant): compare(
        *curve_xy(fig11_data[particle]), interp_on(*fig11[particle, variant])
    )
    for particle in FIG11_IONS
    for variant in ("original_lem", "modified_lem")
}
fig11_paper_vs_data = {
    (particle, variant): compare(
        *curve_xy(fig11_data[particle]), interp_on(*curve_xy(fig11_ref[particle, variant]))
    )
    for particle in FIG11_IONS
    for variant in ("original_lem", "modified_lem")
}

# %% [markdown]
# ## Figs. 10 and 12: survival curves (the only test of beta)
#
# Fig. 10: V79 carbon at four energies (four panels in the paper, overlaid here).
# Fig. 12: XRS, repair-deficient with a PURE-EXPONENTIAL photon (alpha_x = 1.7,
# beta_x = 0, r = 5.5 um), so the whole ion RBE comes from the cluster eta — the
# cleanest isolated test of it in the paper.

# %%
FIG10_ENERGIES = {"4.2": 4.2, "11": 11.0, "76.9": 76.9, "266.4": 266.4}
FIG12_ENERGIES = {"4.2": 4.2, "11": 11.0, "76.9": 76.9}
fig10_models = {"modified_lem": lem_new, "original_lem": lem_old}
XRS = es.FIGURE_INPUTS["fig12_xrs"]
xrs_model = lem2(XRS.photon("new"), XRS.cell_target())


def survival_curve(model: LEM1 | LEM2, energy: float, doses: np.ndarray) -> np.ndarray:
    """LQ survival S(D) for a carbon ion, using the model's own alpha/beta."""
    lq = model.alpha_beta(Ion(Z=6, A=12, energy_MeV_u=energy))
    return np.array([survival(float(d), lq.alpha, lq.beta) for d in doses])


def survival_lnrms(model: LEM1 | LEM2, energy: float, ref_curve: es.Elsasser2007Curve) -> float:
    """relRMS of ln S over the digitized curve (survival spans decades)."""
    d, s = np.array(ref_curve.x), np.array(ref_curve.y)
    m = (d > 0.05) & (s > 1e-6) & (s < 0.999)
    ours = survival_curve(model, energy, d[m])
    rel = (np.log(ours) - np.log(s[m])) / np.log(s[m])
    return float(np.sqrt(np.mean(rel**2)))


fig10_stats = {
    (label, variant): survival_lnrms(
        model,
        energy,
        es.get_curve("fig10_survival_v79", series="calc", variant=variant, energy_MeV_u=energy),
    )
    for label, energy in FIG10_ENERGIES.items()
    for variant, model in fig10_models.items()
}
fig12_stats = {
    label: survival_lnrms(
        xrs_model,
        energy,
        es.get_curve(
            "fig12_survival_xrs", series="calc", variant="modified_lem", energy_MeV_u=energy
        ),
    )
    for label, energy in FIG12_ENERGIES.items()
}

# %% [markdown]
# ## Fig. 4: radial dose vs experiment, in absolute units
#
# The paper's only measured, absolute check on the track structure, and its only
# panel outside carbon: 2.57 MeV/u 16O in tissue-equivalent gas (Varma 1977) and
# 377 MeV/u 20Ne in water (Varma & Baum 1980); r in cm, dose in Gy. Its "modified"
# line is the UN-convolved profile (the r_min reduction alone, no sigma = 4 nm
# Gaussian), so both are drawn: the convolved core is ~50x lower, knee rounded.
# Compared in log10(dose), since the panel spans 7+ decades.

# %%
FIG4_IONS = {
    "oxygen": Ion(Z=8, A=16, energy_MeV_u=2.57),
    "neon": Ion(Z=10, A=20, energy_MeV_u=377.0),
}
uncon_radial = ScholzKraftRadialDose(
    r_min_nm=LEM2_RMIN_NM, radical_diffusion=False, stopping_power=sp
)
FIG4_MODELS = {"original": old_radial, "modified": uncon_radial}


def radial_cm_Gy(model: ScholzKraftRadialDose, ion: Ion, r_cm: np.ndarray) -> np.ndarray:
    """``D(r)`` in Gy on the paper's own radii (cm -> um)."""
    return model.local_doses_Gy(np.asarray(r_cm) * 1e4, ion)


fig4_ref = {
    (particle, variant): es.get_curve(
        "fig4_radial_dose_vs_exp", particle=particle, series="calc", variant=variant
    )
    for particle in FIG4_IONS
    for variant in FIG4_MODELS
}
fig4_data = {
    particle: es.get_curve("fig4_radial_dose_vs_exp", particle=particle, series="data")
    for particle in FIG4_IONS
}


def fig4_logratio(particle: str, variant: str, model: ScholzKraftRadialDose) -> dict:
    """Compare in log10(dose)."""
    ref = fig4_ref[particle, variant]
    ours = radial_cm_Gy(model, FIG4_IONS[particle], np.array(ref.x))
    m = (np.array(ref.y) > 0) & (ours > 0)
    d = np.log10(ours[m]) - np.log10(np.array(ref.y)[m])
    return {
        "rms_log10": float(np.sqrt(np.mean(d**2))),
        "median_ratio": ratio_summary(ours, np.array(ref.y), mask=m).median,
        "n": int(m.sum()),
    }


fig4_stats = {
    (particle, variant): fig4_logratio(particle, variant, model)
    for particle in FIG4_IONS
    for variant, model in FIG4_MODELS.items()
}
fig4_convolved_stats = {
    particle: fig4_logratio(particle, "modified", new_radial) for particle in FIG4_IONS
}

# %% [markdown]
# ## Draw the paper-style panels
#
# Fig. 5 draws both energies at a COMMON (arbitrary) LET: the plateau of eq. 6 is
# `lambda*LET/r_min^2`, so at a common LET the 1 / 100 MeV/u ratio is
# `lambda(1 MeV)/lambda(100 MeV)` (energy enters only through r_max in
# `lambda = 1/(pi*(1 + 2 ln(r_max/r_min)))`). D(r) is linear in LET, so dividing
# each curve by its own LET reproduces that presentation; the arbitrary scale is
# anchored to the paper's 1 MeV/u modified curve, ~3e4 at r = 1 nm. The evidence for
# the common-LET reading: the paper's original-D(r) plateaus read ~4700 (1 MeV/u)
# and ~1000 (100 MeV/u), a ratio of ~4.7, against the model's 4.4 at a common LET
# and ~109 at each ion's own LET (the carbon LET ratio dominates).

# %%
fig, axes = plt.subplots(6, 2, figsize=(12.5, 27.5))
(
    ax_eta,
    ax_photon,
    ax_exp_radial,
    ax_radial,
    ax_mods,
    ax_stagger,
    ax_fig8,
    ax_rbe10,
    ax_hsg,
    ax_surv10,
    ax_surv12,
    ax_blank,
) = axes.ravel()
ax_blank.axis("off")

# --- Fig. 2 (the shaded band is the dose range an ion track ever samples) -------
ax_eta.plot(
    eta_x, eta_paper, "o", ms=3.0, mfc="white", mec="black", mew=0.7, label="paper (digitized)"
)
ax_eta.plot(doses, eta_dense, color="C3", lw=1.8, label="pyRadBio ClusterModel")
ax_eta.plot(
    doses,
    eta_f2015_dense,
    color="C0",
    lw=1.4,
    ls="--",
    label="Friedrich 2015 (closed form)",
)
ax_eta.axvspan(1.0e2, 1.0e5, color="0.85", alpha=0.35, zorder=0)
ax_eta.axhline(eta_paper.max(), color="0.7", ls=":", lw=0.8)
ax_eta.axhline(eta_dense.max(), color="C3", ls=":", lw=0.8)
ax_eta.annotate(
    f"peak: paper {eta_paper.max():.2f}\nours {eta_dense.max():.2f}\n"
    f"F2015 {eta_f2015_dense.max():.2f}",
    xy=(1.4e6, eta_paper.max()),
    fontsize=7,
    va="top",
)
ax_eta.annotate("sampled by\nion tracks", xy=(3.0e2, 14.6), fontsize=6.5, color="0.35", va="top")
ax_eta.set_xscale("log")
ax_eta.set_xlim(1.0e2, 1.0e7)
ax_eta.set_ylim(0.0, 16.0)
ax_eta.set_xlabel("Dose (Gy)")
ax_eta.set_ylabel(r"Damage enhancement $\eta$")
ax_eta.set_title("Fig. 2: cluster enhancement (Fig. 2 is the outlier)")
ax_eta.legend(frameon=False, fontsize=8)
ax_eta.grid(True, which="both", ls=":", lw=0.45)

# --- Fig. 3: markers = paper (colour = panel, marker = branch); solid = derived
# with the paper's eta, dashed = with ours ------------------------------------------
FIG3_PANEL_COLOUR = {"a": "C0", "b": "C3"}
FIG3_BRANCH_MARKER = {"lq_shoulder": "^", "cluster": "o", "original": "s"}
FIG3_PANEL_DT = {"a": es.FIG3_D_T_OLD_GY, "b": es.FIG3_D_T_NEW_GY}
for (panel, variant), curve in fig3_ref.items():
    ax_photon.plot(
        curve.x,
        curve.y,
        FIG3_BRANCH_MARKER[variant],
        ms=3.0,
        mfc="none",
        mec=FIG3_PANEL_COLOUR[panel],
        mew=0.7,
        label=f"paper {panel} ($D_t$={FIG3_PANEL_DT[panel]:g}): {variant}",
        zorder=4,
    )
ax_photon.plot(fig3_doses, fig3_lq, color="0.1", ls=":", lw=1.0, label="unbroken LQ")
ax_photon.plot(fig3_doses, fig3_old_tail, color="0.45", lw=1.0, label=r"original tail, $D_t=40$")
ax_photon.plot(fig3_doses, fig3_a_paper, color="C0", lw=2.0, label=r"a: eq.10, paper $\eta$")
ax_photon.plot(fig3_doses, fig3_a_ours, color="C0", ls="--", lw=1.4, label=r"a: eq.10, our $\eta$")
ax_photon.plot(fig3_doses, fig3_b_paper, color="C3", lw=2.0, label=r"b: eq.10, paper $\eta$")
ax_photon.plot(fig3_doses, fig3_b_ours, color="C3", ls="--", lw=1.4, label=r"b: eq.10, our $\eta$")
ax_photon.set_xscale("log")
ax_photon.set_yscale("log")
ax_photon.set_xlim(1.0, 1.0e6)
ax_photon.set_ylim(1.0e-1, 1.0e8)
ax_photon.set_xlabel("Dose (Gy)")
ax_photon.set_ylabel("Average number of lethal events")
ax_photon.set_title("Fig. 3: photon tail, CHO (derived vs digitized)")
ax_photon.legend(frameon=False, fontsize=6, ncols=2)
ax_photon.grid(True, which="both", ls=":", lw=0.45)

# --- Fig. 4 (each model clipped at its OWN r_max: gamma 0.05 vs 0.062) -----------
FIG4_COLOURS = {"oxygen": "C0", "neon": "C3"}
FIG4_LABEL = {
    "oxygen": "$^{16}$O 2.57 MeV/u (tissue-equiv.)",
    "neon": "$^{20}$Ne 377 MeV/u (water)",
}
_r_dense_cm = np.logspace(-8.1, -1.0, 400)
for particle, colour in FIG4_COLOURS.items():
    ion = FIG4_IONS[particle]
    for variant, model in FIG4_MODELS.items():
        r_cm = _r_dense_cm[_r_dense_cm * 1e4 <= model.r_max_um(ion)]
        ref = fig4_ref[particle, variant]
        ax_exp_radial.plot(ref.x, ref.y, color="0.6", lw=2.6, alpha=0.55, zorder=1)
        ax_exp_radial.plot(
            r_cm,
            radial_cm_Gy(model, ion, r_cm),
            color=colour,
            ls="-" if variant == "modified" else "--",
            lw=1.6 if variant == "modified" else 1.0,
            label=FIG4_LABEL[particle] if variant == "modified" else None,
            zorder=3,
        )
    # The convolved D~(r) that LEM II actually uses -- NOT what Fig. 4 draws.
    r_cm = _r_dense_cm[_r_dense_cm * 1e4 <= new_radial.r_max_um(ion)]
    ax_exp_radial.plot(
        r_cm, radial_cm_Gy(new_radial, ion, r_cm), color=colour, ls=":", lw=1.2, alpha=0.8, zorder=2
    )
    data = fig4_data[particle]
    ax_exp_radial.plot(
        data.x, data.y, "o", ms=4.0, mfc="white", mec=colour, mew=0.9, lw=0, zorder=5
    )
ax_exp_radial.plot([], [], color="0.6", lw=2.6, alpha=0.55, label="paper's lines")
ax_exp_radial.plot([], [], "o", ms=4.0, mfc="white", mec="0.3", lw=0, label="experiment")
ax_exp_radial.plot([], [], color="0.3", ls=":", lw=1.2, label=r"our $\tilde{D}(r)$ (convolved)")
ax_exp_radial.set_xscale("log")
ax_exp_radial.set_yscale("log")
ax_exp_radial.set_xlim(1e-8, 2e-1)
ax_exp_radial.set_ylim(1e-8, 1e8)
ax_exp_radial.set_xlabel("$r$ (cm)")
ax_exp_radial.set_ylabel("Dose (Gy)")
ax_exp_radial.set_title("Fig. 4: radial dose vs experiment (absolute)")
ax_exp_radial.legend(frameon=False, fontsize=6.0, loc="lower left")
ax_exp_radial.grid(True, which="major", ls=":", lw=0.45)

# --- Fig. 5 ----------------------------------------------------------------------
r_um = np.logspace(np.log10(0.001), np.log10(0.05), 160)


def radial_per_let(model: ScholzKraftRadialDose, ion: Ion) -> np.ndarray:
    """``D(r)`` at a common LET, i.e. dose per unit LET (arbitrary units)."""
    return model.local_doses_Gy(r_um, ion) / sp.let_keV_um(ion)


ION_1, ION_100 = Ion(Z=6, A=12, energy_MeV_u=1.0), Ion(Z=6, A=12, energy_MeV_u=100.0)
fig5_norm = radial_per_let(new_radial, ION_1)[0] / 3.0e4
for label, ion, colour in (("1 MeV/u", ION_1, "C0"), ("100 MeV/u", ION_100, "C1")):
    ax_radial.plot(
        r_um,
        radial_per_let(old_radial, ion) / fig5_norm,
        color=colour,
        ls="--",
        lw=1.0,
        label=f"{label}, original $D(r)$",
    )
    ax_radial.plot(
        r_um,
        radial_per_let(new_radial, ion) / fig5_norm,
        color=colour,
        lw=2.2,
        label=rf"{label}, modified $\tilde{{D}}(r)$",
    )
fig5_plateau_ratio = radial_per_let(old_radial, ION_1)[0] / radial_per_let(old_radial, ION_100)[0]
ax_radial.set_xscale("log")
ax_radial.set_yscale("log")
ax_radial.set_xticks([0.001, 0.002, 0.005, 0.01, 0.02, 0.05])
ax_radial.set_xticks([], minor=True)
ax_radial.set_xticklabels(["0.001", "0.002", "0.005", "0.01", "0.02", "0.05"])
ax_radial.set_xlim(0.001, 0.05)
ax_radial.set_ylim(30.0, 4.0e4)
ax_radial.set_xlabel(r"Radius ($\mu$m)")
ax_radial.set_ylabel("Dose (arbitrary units)")
ax_radial.set_title("Fig. 5: radial dose profile (model only)")
ax_radial.legend(frameon=False, fontsize=7)
ax_radial.grid(True, which="both", ls=":", lw=0.45)

# --- Fig. 6 ----------------------------------------------------------------------
for variant, (label, _m, colour, style) in FIG6_CURVES.items():
    ref = fig6_ref[variant]
    ax_mods.plot(ref.x, ref.y, "o", ms=2.2, mfc="none", mec=colour, mew=0.6, zorder=3)
    ax_mods.plot(*fig6[variant], color=colour, label=label, **style)
ax_mods.plot([], [], "o", ms=3, mfc="none", mec="0.3", label="paper (digitized)")
ax_mods.set_xscale("log")
ax_mods.set_xlim(5.0, 550.0)
ax_mods.set_ylim(1.0, 20.0)
ax_mods.set_xlabel(r"LET (keV/$\mu$m)")
ax_mods.set_ylabel(r"RBE$_\alpha$")
ax_mods.set_title("Fig. 6: separated modifications (lines = ours, circles = paper)")
ax_mods.legend(frameon=False, fontsize=7)
ax_mods.grid(True, which="both", ls=":", lw=0.45)

# --- Fig. 7: paper black, ours red, one line style per stagger size --------------
FIG7_STYLES = {v: (label, style) for v, (_h, _d, label, style) in FIG7_VARIANTS.items()}
FIG7_STYLES["lem_old"] = ("LEM old (reference)", {"ls": "-", "lw": 1.0})
for variant, (label, style) in FIG7_STYLES.items():
    paper_colour = "0.55" if variant == "lem_old" else "0.1"
    ours_colour = "0.75" if variant == "lem_old" else "C3"
    ax_stagger.plot(fig7_ref[variant].x, fig7_ref[variant].y, color=paper_colour, zorder=1, **style)
    ax_stagger.plot(*fig7[variant], color=ours_colour, zorder=2, label=label, **style)
ax_stagger.plot([], [], color="0.1", lw=1.2, label="black = paper")
ax_stagger.plot([], [], color="C3", lw=1.2, label="red = pyRadBio")
ax_stagger.set_xscale("log")
ax_stagger.set_xlim(10.0, 550.0)
ax_stagger.set_ylim(1.0, 12.0)
ax_stagger.set_xlabel(r"LET (keV/$\mu$m)")
ax_stagger.set_ylabel(r"RBE$_\alpha$")
ax_stagger.set_title("Fig. 7: stagger size $h$ (black = paper, red = ours)")
ax_stagger.legend(frameon=False, fontsize=6.5, ncol=2)
ax_stagger.grid(True, which="both", ls=":", lw=0.45)

# --- Fig. 8 ----------------------------------------------------------------------
FIG8_COLOURS = {"V79": "C3", "CHO": "C0"}
for line in ("V79", "CHO"):
    colour = FIG8_COLOURS[line]
    for variant, style in (
        ("original_lem", {"ls": "--", "lw": 1.0}),
        ("modified_lem", {"ls": "-", "lw": 2.0}),
    ):
        ref = fig8_ref[line, variant]
        ax_fig8.plot(
            ref.x,
            ref.y,
            color="0.55" if variant == "original_lem" else "0.15",
            lw=1.0,
            ls=style["ls"],
            zorder=1,
        )
        ax_fig8.plot(
            *fig8[line, variant],
            color=colour,
            label=f"{line}, {'modified' if variant == 'modified_lem' else 'original'} LEM",
            **style,
        )
    data = fig8_data[line]
    ax_fig8.errorbar(
        data.x,
        data.y,
        yerr=data.y_err,
        fmt="o" if line == "V79" else "s",
        ms=4.5,
        mfc="white",
        mec=colour,
        ecolor=colour,
        elinewidth=0.8,
        capsize=2.5,
        lw=0,
        label=f"{line} exp. (Weyrather 1999)",
        zorder=5,
    )
ax_fig8.set_xscale("log")
ax_fig8.set_xlim(10.0, 550.0)
ax_fig8.set_xlabel(r"LET (keV/$\mu$m)")
ax_fig8.set_ylabel(r"RBE$_\alpha$")
ax_fig8.set_title("Fig. 8: two cell lines (grey = paper)")
ax_fig8.legend(frameon=False, fontsize=6.5, ncol=2)
ax_fig8.grid(True, which="both", ls=":", lw=0.45)

# --- Fig. 9 ----------------------------------------------------------------------
ax_rbe10.plot(
    fig9_orig_ref.x, fig9_orig_ref.y, color="0.55", ls="--", lw=1.2, label="paper, original LEM"
)
ax_rbe10.plot(fig9_mod_ref.x, fig9_mod_ref.y, color="0.15", lw=1.2, label="paper, modified LEM")
ax_rbe10.plot(let_old, rbe10_old, color="C0", ls="--", lw=1.8, label="pyRadBio LEM I")
ax_rbe10.plot(let_new, rbe10_new, color="C3", lw=2.0, label="pyRadBio LEM II")
ax_rbe10.errorbar(
    fig9_data.x,
    fig9_data.y,
    fmt="s",
    ms=5,
    mfc="white",
    mec="black",
    mew=0.9,
    label="V79 exp. (Weyrather 1999)",
    zorder=5,
)
ax_rbe10.set_xscale("log")
ax_rbe10.set_xlim(10.0, 550.0)
ax_rbe10.set_ylim(0.0, 4.2)
ax_rbe10.set_xlabel(r"LET (keV/$\mu$m)")
ax_rbe10.set_ylabel("RBE")
ax_rbe10.set_title("Fig. 9: RBE at 10% survival")
ax_rbe10.legend(frameon=False, fontsize=7)
ax_rbe10.grid(True, which="both", ls=":", lw=0.45)

# --- Fig. 11 ---------------------------------------------------------------------
FIG11_COLOURS = {"helium": "C2", "carbon": "C3"}
FIG11_MARKERS = {"helium": "o", "carbon": "s"}
FIG11_SYMBOL = {"helium": "He", "carbon": "C"}
for particle, colour in FIG11_COLOURS.items():
    for variant, style in (
        ("original_lem", {"ls": "--", "lw": 1.0}),
        ("modified_lem", {"ls": "-", "lw": 2.0}),
    ):
        ref = fig11_ref[particle, variant]
        ax_hsg.plot(ref.x, ref.y, color="0.55", lw=1.0, ls=style["ls"], zorder=1)
        ax_hsg.plot(
            *fig11[particle, variant],
            color=colour,
            label=f"{FIG11_SYMBOL[particle]}, {variant.split('_')[0]} LEM",
            **style,
        )
    data = fig11_data[particle]
    ax_hsg.plot(
        data.x,
        data.y,
        FIG11_MARKERS[particle],
        ms=4.5,
        mfc="white",
        mec=colour,
        mew=0.9,
        lw=0,
        label=f"{FIG11_SYMBOL[particle]} exp. (Furusawa 2000)",
        zorder=5,
    )
ax_hsg.set_xscale("log")
ax_hsg.set_xlim(15.0, 1000.0)
# Label the paper's own ticks: default log minor labels collide over 1.8 decades.
ax_hsg.xaxis.set_major_locator(plt.FixedLocator([20, 30, 50, 100, 200, 300, 500, 1000]))
ax_hsg.xaxis.set_minor_locator(plt.NullLocator())
ax_hsg.xaxis.set_major_formatter(plt.FuncFormatter(lambda v, _pos: f"{v:.0f}"))
ax_hsg.set_ylim(0.0, 3.2)
ax_hsg.set_xlabel(r"LET (keV/$\mu$m)")
ax_hsg.set_ylabel(r"$\alpha$ (Gy$^{-1}$)")
ax_hsg.set_title("Fig. 11: HSG, He + C (grey = paper)")
ax_hsg.legend(frameon=False, fontsize=6.5, ncol=2)
ax_hsg.grid(True, which="both", ls=":", lw=0.45)

# --- Fig. 10: paper lines are thin greyed twins of each energy's colour ----------
FIG10_COLOURS = {"4.2": "C3", "11": "C1", "76.9": "C2", "266.4": "C0"}
_surv_doses = np.linspace(0.0, 8.0, 200)


def _greyed(colour: str, mix: float = 0.55) -> tuple[float, float, float]:
    """Blend a colour toward grey so the paper reference reads as a faded twin."""
    rgb = np.array(mcolors.to_rgb(colour))
    return tuple(rgb * (1.0 - mix) + 0.5 * mix)


for label, energy in FIG10_ENERGIES.items():
    colour = FIG10_COLOURS[label]
    for variant, paper_ls in (("modified_lem", ":"), ("original_lem", "-.")):
        paper = es.get_curve(
            "fig10_survival_v79", series="calc", variant=variant, energy_MeV_u=energy
        )
        ax_surv10.plot(paper.x, paper.y, color=_greyed(colour), lw=0.8, ls=paper_ls, zorder=2)
    data = es.get_curve("fig10_survival_v79", series="data", energy_MeV_u=energy)
    ax_surv10.plot(data.x, data.y, "o", ms=3.5, mfc="white", mec=colour, mew=0.8, lw=0, zorder=5)
    ax_surv10.plot(
        _surv_doses,
        survival_curve(lem_new, energy, _surv_doses),
        color=colour,
        lw=1.8,
        label=f"{label} MeV/u",
        zorder=3,
    )
    ax_surv10.plot(
        _surv_doses,
        survival_curve(lem_old, energy, _surv_doses),
        color=colour,
        ls="--",
        lw=0.9,
        zorder=3,
    )
ax_surv10.plot([], [], color="0.5", lw=0.8, ls=":", label="paper, modified LEM")
ax_surv10.plot([], [], color="0.5", lw=0.8, ls="-.", label="paper, original LEM")
ax_surv10.set_yscale("log")
ax_surv10.set_xlim(0.0, 8.0)
ax_surv10.set_ylim(1e-2, 1.1)
ax_surv10.set_xlabel("dose (Gy)")
ax_surv10.set_ylabel("surviving fraction")
ax_surv10.set_title("Fig. 10: V79 carbon survival (solid=LEM II, dashed=LEM I)")
ax_surv10.legend(frameon=False, fontsize=6.5, title="carbon energy", title_fontsize=6.5)
ax_surv10.grid(True, which="both", ls=":", lw=0.45)

# --- Fig. 12 ---------------------------------------------------------------------
xray = es.get_curve("fig12_survival_xrs", series="calc", variant="xray")
ax_surv12.plot(xray.x, xray.y, color="0.4", lw=1.4, ls="-.", label=r"X-ray ($\alpha_x=1.7$)")
for label, energy in FIG12_ENERGIES.items():
    colour = FIG10_COLOURS[label]
    data = es.get_curve("fig12_survival_xrs", series="data", energy_MeV_u=energy)
    ax_surv12.plot(data.x, data.y, "s", ms=3.5, mfc="white", mec=colour, mew=0.8, lw=0, zorder=5)
    ref = es.get_curve(
        "fig12_survival_xrs", series="calc", variant="modified_lem", energy_MeV_u=energy
    )
    ax_surv12.plot(ref.x, ref.y, color="0.6", lw=2.4, alpha=0.55, zorder=1)
    ax_surv12.plot(
        _surv_doses,
        survival_curve(xrs_model, energy, _surv_doses),
        color=colour,
        lw=1.8,
        label=f"{label} MeV/u",
    )
ax_surv12.set_yscale("log")
ax_surv12.set_xlim(0.0, 3.0)
ax_surv12.set_ylim(1e-2, 1.1)
ax_surv12.set_xlabel("dose (Gy)")
ax_surv12.set_ylabel("surviving fraction")
ax_surv12.set_title("Fig. 12: XRS survival — RBE is all cluster $\\eta$ (grey=paper)")
ax_surv12.legend(frameon=False, fontsize=6.5)
ax_surv12.grid(True, which="both", ls=":", lw=0.45)

fig.suptitle(f"Elsasser & Scholz 2007 (LEM II) reconstruction — {STOPPING_LABEL}", y=0.997)
fig.tight_layout()
print()
save_figure(fig, "lem2_cluster_effect.png", directory=FIGURES, dpi=150)

# %% [markdown]
# ## Reconstruction summary

# %%
print(f"\n=== Elsasser & Scholz 2007 reconstruction summary ({STOPPING_LABEL}) ===")
print(f"initial slope (ALL models, LEM I and LEM II): {INITIAL_SLOPE_MODEL}", end="")
if INITIAL_SLOPE_MODEL == "lethal_central_saturating":
    print(" = the paper's eq. (8)")
else:
    print(" -- NOT the paper's eq. (8); every alpha below moves with it")

print("\n--- Fig. 2: damage enhancement eta(D) ---")
print_compare("eta(D) vs paper", eta_stats)
print(
    f"  peak: paper {eta_paper.max():.2f} at {eta_x[np.argmax(eta_paper)]:.3g} Gy;  "
    f"ours {eta_dense.max():.2f} at {doses[np.argmax(eta_dense)]:.3g} Gy"
)
print("  NOTE: shape ~1 => a genuine rule mismatch, not a scale offset.")
print("        eta-1 = (a_SSB/a_DSB)*q(mu)/mu, so the genome size cannot fix the peak height.")
print_compare("eta(D) Friedrich 2015", eta_f2015_stats)
print(
    f"  peak: Friedrich 2015 {eta_f2015_dense.max():.2f} at "
    f"{doses[np.argmax(eta_f2015_dense)]:.3g} Gy"
)
print("  Friedrich, Durante & Scholz 2015 (doi:10.1093/rpd/ncv147) recompute eta from")
print("  the SAME a_SSB=1250, a_DSB=30, t=25 bp with a closed form and a stricter rule.")
print("  It lands on OUR peak, not the figure's -- so Fig. 2 is the outlier, not us:")
print(
    f"    ours {eta_dense.max():.2f}   Friedrich 2015 {eta_f2015_dense.max():.2f}   "
    f"Fig. 2 {eta_paper.max():.2f}"
)
print("  bias vs the digitized Fig. 2, by dose band (a track centre stays <= 1e5 Gy,")
print("  so only the first two bands are ever sampled by LEM):")
print(f"    {'band':9s} {'range [Gy]':>17s} {'n':>4s}  {'ours':>7s} {'F2015':>7s} {'F/ours':>8s}")
for band, lo, hi, n, b_ours, b_f, b_rel in eta_band_stats:
    print(f"    {band:9s} {lo:7.0e}-{hi:<8.0e} {n:4d}  {b_ours:+6.1%} {b_f:+7.1%} {b_rel:+8.1%}")
print("  The two rules agree to a few percent everywhere a track can reach and diverge")
print("  only past ~1e6 Gy -- which is precisely where the 2015 paper says its own rule")
print("  is undetermined ('not clear how to exactly implement the conversion of")
print("  candidates ... the high dose part around the maximum is affected, while the")
print("  course of the curve at lower doses remains rather unchanged').")

print("\n--- Fig. 3: photon dose-effect (CHO), derived vs digitized ---")
print("  Deriving with the paper's OWN eta tests eq. 10 alone; deriving with our")
print("  ClusterModel eta adds the clustering rule. The difference isolates the bug.")
for panel, paper_model, our_model in (
    ("a", lem_fig3_a_paper, lem_fig3_a_ours),
    ("b", lem_fig3_b_paper, lem_fig3_b_ours),
):
    # The LQ shoulder and the un-enhanced tail carry no eta: report them once.
    print_compare(f"{panel}: LQ shoulder", fig3_stats(panel, "lq_shoulder", paper_model))
    print_compare(f"{panel}: original tail (eta=1)", fig3_stats(panel, "original", paper_model))
    print_compare(f"{panel}: cluster  [paper eta]", fig3_stats(panel, "cluster", paper_model))
    print_compare(f"{panel}: cluster  [our eta]", fig3_stats(panel, "cluster", our_model))
print(f"  panel a used D_t = {es.FIG3_D_T_OLD_GY} Gy (the caption's 30 Gy is a paper error).")

# %% [markdown]
# The paper corroborates the 40 Gy itself: the Fig. 8 caption (p. 325) gives CHO
# "D_t = 40 Gy (old) and D_t = 9.5 Gy (modified)" against V79's "D_t = 30 Gy (old)"
# in the same sentence, so Fig. 3's 30 is V79's number under a CHO panel.
# Re-deriving with the paper's own eta ranks them 3.6% RMS at 40 vs 23.0% at 30 on
# panel a's cluster branch. Panel b's `D_t = 10` Gy (`es.FIG3_D_T_NEW_GY`) is an
# ILLUSTRATION, not CHO's refit: Fig. 8 puts CHO's modified D_t at 9.5 Gy, and 9.5
# is the worst of the three against the digitized panel (15.2% RMS, vs 12.9% at 10
# and 8.7% at 11).

# %%
print("\n--- Fig. 4: radial dose vs experiment (absolute Gy, r in cm) ---")
print("  The paper's only measured, absolute check on the track structure, and its")
print("  only panel outside carbon. Compared in log10(dose): the panel spans 7+ decades.")
for (particle, variant), stats in fig4_stats.items():
    print(
        f"    {particle:7s} {variant:9s} n={stats['n']:3d}  "
        f"RMS(log10)={stats['rms_log10']:.3f}  median ours/paper={stats['median_ratio']:.2f}"
    )
print("  Fig. 4's 'modified' line is the UN-convolved profile (r_min = 0.3 nm only).")
print("  Scoring our convolved D~(r) -- what LEM II uses -- against it instead:")
for particle, stats in fig4_convolved_stats.items():
    print(
        f"    {particle:7s} modified  RMS(log10)={stats['rms_log10']:.3f}"
        f"   <- apples-to-oranges (the sigma=4 nm core is ~50x lower)"
    )

# %% [markdown]
# Our absolute dose runs a few % to ~14% high; panel a's medium is
# tissue-equivalent gas, while our model assumes water.

# %%
print("\n--- Fig. 5: radial dose profile ---")
print(
    f"  original-D(r) plateau ratio (1 / 100 MeV/u) at a common LET: "
    f"{fig5_plateau_ratio:.2f}  (paper reads ~4.7)"
)
print("  NOTE: at each ion's OWN LET this ratio would be ~109 -- the paper's Fig. 5")
print("        draws both energies at one arbitrary LET, so only r_max varies.")

print("\n--- Fig. 6: RBE_alpha vs LET, all six model modifications ---")
print(f"  {'variant':18s} {'paper peak':>16s} {'our peak':>16s}   residual")
for variant, (label, _m, _c, _s) in FIG6_CURVES.items():
    px, py = curve_xy(fig6_ref[variant])
    let, value = fig6[variant]
    st = fig6_stats[variant]
    print(
        f"  {label:18s} {py.max():6.2f} @ {px[np.argmax(py)]:5.0f}  {value.max():6.2f} @ "
        f"{let[np.argmax(value)]:5.0f}   relRMS={st.rel_rms:5.1%} bias={st.rel_bias:+6.1%}"
    )
ratio_old = fig6["lem_old"][1].max() / fig6["lem_old"][1].min()
ratio_new = fig6["lem_new"][1].max() / fig6["lem_new"][1].min()
paper_old = max(fig6_ref["lem_old"].y) / min(fig6_ref["lem_old"].y)
paper_new = max(fig6_ref["lem_new"].y) / min(fig6_ref["lem_new"].y)
print("\n  the paper's key claim -- max/min RBE_alpha must RISE from old to new:")
print(
    f"    paper: {paper_old:.2f} -> {paper_new:.2f}      ours: {ratio_old:.2f} -> {ratio_new:.2f}"
)
print("  the no-refit case must overshoot ('extremely high and unrealistic'):")
print(
    f"    LEM+D(r)+clusters peak: paper {max(fig6_ref['radial_clusters'].y):.2f}, "
    f"ours {fig6['radial_clusters'][1].max():.2f}   (LEM new: paper "
    f"{max(fig6_ref['lem_new'].y):.2f}, ours {fig6['lem_new'][1].max():.2f})"
)

print("\n--- Method check: do the paper's curves match the paper's stated eq. (8)? ---")
print("  Eq. (8) scores one traversal through the nucleus CENTRE and drops every track")
print("  that misses the nucleus but still deposits dose in it. What that costs is set")
print("  by r_max/R_eff, which sweeps three decades across a single RBE_alpha panel.")
print("  Measured on Fig. 6's `lem_old`, the simplest line in the paper:")
print(f"    {'LET [keV/um]':>16s} {'n':>3s} {'r_max/R_eff':>12s} {'eq.(8)':>9s} {'exact':>9s}")
for b in method_bands:
    print(
        f"    {b.let_lo_keV_um:7.0f}-{b.let_hi_keV_um:<8.0f} {b.n:3d} {b.r_max_over_r_eff:12.1f} "
        f"{b.bias_eq8:+8.1%} {b.bias_exact:+8.1%}"
    )
print("  The eq. (8) residual is MONOTONE in r_max/R_eff -- negligible where the track")
print("  fits inside the nucleus, -10% where the track is 83x its radius -- which is the")
print("  signature of the dropped tracks, and it is the shape you would see if the")
print("  paper's lines had NOT been made with eq. (8). The exact impact-parameter")
print("  integral is flat to ~1% across the same three decades.")
print("  So the figures appear to have been produced with the full average, even though")
print("  the text describes the b=0 shortcut -- consistent with eq. (8) being introduced")
print("  under 'Approximation ... For treatment planning'. Everything below runs the")
print("  paper's STATED method anyway; read the residuals with this LET trend in mind.")

print("\n--- Genome size: unstated by the paper, and a lever only on RBE ---")
print("  eta - 1 = (a_SSB/a_DSB) q(mu)/mu with mu = a_SSB D h/G, so G cancels from the")
print("  peak HEIGHT and only rescales the dose axis. But a track's local-dose spectrum")
print("  is fixed, so sliding eta along that axis changes how much of the track is")
print("  enhanced. Scored on the LEM II / LEM I ratio, which cancels the stopping power,")
print("  r_max, the effective area and the slope method to first order:")
print(f"    {'L_gen [bp]':>12s} {'eta_max':>9s} {'ratio bias vs Fig. 6':>21s}")
for _g, _emax, _gb in genome_bands:
    print(f"    {_g:12.2e} {_emax:9.2f} {_gb:+20.1%}")

# %% [markdown]
# eta_max is flat across a 4x span of L_gen, so no genome size can ever reach
# Fig. 2's peak of 14.79 -- that is a clustering-RULE difference. But the RBE ratio
# moves ~2.4% per 10%, which makes this a tempting knob. DO NOT FIT IT: a raw-bias
# fit absorbs the cluster-free LEM I offset into a cluster parameter. The ratio
# fit puts four of the five panels at 5.2-6.2e9 (V79 5.4/5.2, HSG 5.8/6.2) with CHO
# alone at 7.9e9 -- so 6.0e9 is close to a consensus, not a compromise. The raw-bias
# fit is the wrong one for a further reason: it splits by cell line (4.5e9 V79/CHO
# vs 7.4-8.4e9 HSG), whereas the HSG ratio fit gives ~5.1e9 and EVERY panel wants a
# smaller genome -- a uniform offset laundered through a cluster parameter is easier
# to mistake for a good fit, not harder. 6.0e9 stands unfitted; the +-4% is budget.

# %%
print("\n--- Fig. 7: stagger size h (the cluster extension's one free geometric knob) ---")
print(f"  {'variant':24s} {'paper max/min':>14s} {'our max/min':>12s}   residual")
for variant, (label, _style) in FIG7_STYLES.items():
    ref, (let, value), st = fig7_ref[variant], fig7[variant], fig7_stats[variant]
    print(
        f"  {label:24s} {max(ref.y) / min(ref.y):13.2f}  {value.max() / value.min():11.2f}   "
        f"relRMS={st.rel_rms:5.1%} bias={st.rel_bias:+6.1%}"
    )
print("  the paper's claim (p. 325): max/min RBE_alpha RISES with h, and h=10 bp at")
print("  D_t=11 Gy still shows a significant enhancement near the RBE maximum. Both")
print("  hold. But note our max/min runs HIGH against every one of the paper's own")
print("  ratios, and by a roughly constant amount -- the ordering is reproduced, the")
print("  dynamic range is not. That is the low-LET end, not the peak: eq. (8) scores")
print("  only tracks through the nucleus CENTRE, and the tracks it drops matter most")
print("  where r_max is largest, i.e. at the lowest LET, which is exactly the")
print("  denominator of max/min. See INITIAL_SLOPE_MODEL.")


print("\n--- Fig. 8: RBE_alpha for two cell lines (V79 + CHO) ---")
print("  The only panel drawing both lines from one code path, so it pins the nuclear")
print("  geometry: V79 and CHO differ mainly by r_nucleus (3.8 vs 4.7 um), and both are")
print("  the paper's own values used as-is (effective_size_factor = 1.0).")
for line in ("V79", "CHO"):
    for variant in ("original_lem", "modified_lem"):
        print_compare(f"{line} {variant:13s}", fig8_stats[line, variant])
        our_let, ours = fig8[line, variant]
        ref = fig8_ref[line, variant]
        print(
            f"      peak: paper {max(ref.y):5.2f} @ {ref.x[int(np.argmax(ref.y))]:3.0f}"
            f"   ours {ours.max():5.2f} @ {our_let[int(np.argmax(ours))]:3.0f} keV/um"
        )
print("\n  cross-figure consistency (Fig. 8's V79 lines ARE Fig. 6's, traced separately):")
for f6, stats in fig8_vs_fig6.items():
    print_compare(f"fig8 vs fig6 {f6}", stats)

print("\n--- Fig. 9: RBE at 10% survival ---")
print_compare("LEM II vs paper modified LEM", fig9_new_stats)
print_compare("LEM I  vs paper original LEM", fig9_old_stats)
print(
    f"  peak RBE: paper modified {max(fig9_mod_ref.y):.2f}, ours {rbe10_new.max():.2f}; "
    f"paper original {max(fig9_orig_ref.y):.2f}, ours {rbe10_old.max():.2f}"
)

print("\n--- Fig. 10: V79 carbon survival curves (beta / full dose-response) ---")
print("  The only panel testing beta and the D_t refit; the rest are alpha-flavoured.")
print("  relRMS of ln(S) over each digitized curve (modified vs original LEM):")
for label, energy in FIG10_ENERGIES.items():
    let = sp.let_keV_um(Ion(Z=6, A=12, energy_MeV_u=energy))
    print(
        f"    {label:>6} MeV/u  LET {let:5.0f}"
        f"   modified {fig10_stats[label, 'modified_lem']:5.1%}"
        f"   original {fig10_stats[label, 'original_lem']:5.1%}"
    )
print("  Modified beats original at every energy; 4.2 MeV/u is where our SP is weakest.")

print("\n--- Fig. 12: XRS survival -- RBE is entirely the cluster eta ---")
print("  XRS is repair-deficient: pure-exponential photon (alpha_x=1.7, beta_x=0), so a")
print("  linear local response would give RBE=1. All ion RBE comes from eta making the")
print("  core non-linear -- the cleanest isolated test of the cluster rule in the paper.")
for label, energy in FIG12_ENERGIES.items():
    a = xrs_model.alpha_beta(Ion(Z=6, A=12, energy_MeV_u=energy)).alpha
    print(
        f"    {label:>6} MeV/u  our alpha {a:.2f} (x-ray 1.70)"
        f"  relRMS(lnS) {fig12_stats[label]:5.1%}"
    )

# %% [markdown]
# Despite our eta peak being ~11% low, XRS survival matches to a few % -- fresh
# confirmation that the eta rule mismatch is nearly RBE-neutral. XRS needs r = 5.5 um.

# %%
print("\n--- Fig. 11: initial slope alpha, HSG, helium AND carbon ---")
print("  The only panel off the carbon/V79/CHO axis: a third cell line (r=5.0 um)")
print("  and a second ion. y is alpha in Gy^-1, not RBE_alpha.")
for particle in FIG11_IONS:
    for variant in ("original_lem", "modified_lem"):
        print_compare(f"{particle} {variant:13s}", fig11_stats[particle, variant])
print(
    "\n  the paper's Fig. 11 claim is about the DATA: 'the extended model describes the\n"
    "  initial slope better than the original LEM; particularly for helium ions the\n"
    "  improvement is convincing'. relRMS of each model against the Furusawa points:"
)
print(f"    {'':10s} {'original -> modified':>24}   {'(papers own lines)':>24}")
for particle in FIG11_IONS:
    ours = (
        fig11_vs_data[particle, "original_lem"].rel_rms,
        fig11_vs_data[particle, "modified_lem"].rel_rms,
    )
    theirs = (
        fig11_paper_vs_data[particle, "original_lem"].rel_rms,
        fig11_paper_vs_data[particle, "modified_lem"].rel_rms,
    )
    print(
        f"    {particle:10s} {ours[0]:9.1%} -> {ours[1]:9.1%}   "
        f"{theirs[0]:9.1%} -> {theirs[1]:9.1%}"
    )
print(
    "  Both reproduce the claim: modified >> original, helium most. The absolute\n"
    "  numbers are large because relRMS is dominated by the lowest-alpha points,\n"
    "  where the paper's OWN lines are ~2x above the data -- a mismatch below\n"
    "  30 keV/um that the paper itself reports as unexplained (p. 327). The data x\n"
    "  is also Furusawa's DOSE-AVERAGED LET (p. 326), not a track-segment LET."
)
_c_mod = fig11_stats["carbon", "modified_lem"]
print(
    f"  Our HSG carbon sits {_c_mod.rel_bias:+.0%} vs the paper's own line, i.e. in\n"
    "  family with V79 and CHO. It would read +9% if the panel's lowest y tick\n"
    "  (alpha = 0.25) were anchored as alpha = 0; the six Fig. 11 CSVs use the\n"
    "  correct anchoring; see docs/references/LEM_II_Elsasser2007.md."
)
