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
# # LEM IV: reproducing Elsasser et al. 2010 (Fig. 4) and Friedrich et al. 2012 (Figs. 3-4)
#
# LEM IV has two source papers with digitized reconstruction data, both in
# `pyradbio.benchmark.elsasser2010` and `pyradbio.benchmark.friedrich2012` (hand-digitized;
# provenance and the two digitization notes -- Elsasser
# 2010's mistracked "He Exp" trace, Friedrich 2012's Fig. 4 alpha_I axis -- are in those
# modules' docstrings):
# - **Elsasser et al. 2010** (doi:10.1016/j.ijrobp.2010.05.014), the primary paper: Fig. 4
#   (p. 1181), monoenergetic RBE(LET) for V79 (protons + helium) and HSG (carbon + helium).
# - **Friedrich et al. 2012** (doi:10.3109/09553002.2011.611213), the proceedings
#   follow-up: Fig. 3 (p. 106), V79 carbon, and Fig. 4 (p. 107), the single-particle
#   approximation vs the full simulation for a "hypothetical" cell.
#
# Five panels:
# 1. **Elsasser 2010 Fig. 4a**, V79 p + He. Reliable to LET ~25 keV/um; above it the
#    single-particle model enters a deep-overkill regime the data do not follow.
# 2. **Elsasser 2010 Fig. 4b**, HSG C + He with the default `D_eq` effect model. The
#    dotted RBE$_{10}$ is the **paper's own method**: Elsasser 2010 derives its
#    coefficients "by the method of the original LEM (13)" (Scholz 1997, the full
#    simulation, `beta_model="local_dose"`), not by the Kraemer-Scholz approximation our
#    default `beta_model="slope_matching"` implements (a Friedrich 2012 device, right for
#    Panel 3). On the corrected digitization (the y axis runs 0-11, see
#    `docs/references/LEM_IV_Elsasser2010.md`) the RBE$_\alpha$ bias is under 1% on both
#    ions, and the full simulation *adds* a few points of RBE$_{10}$ bias; the script
#    prints how many.
# 3. **Friedrich 2012 Fig. 3**, V79 carbon RBE$_\alpha$ + RBE$_{10}$, model + Furusawa 2000.
# 4. **Friedrich 2012 Fig. 4, alpha_I**, and 5. **beta_I**: the single-particle
#    approximation, two deterministic full simulations (local dose, discrete DSB) and,
#    when its checkpoint exists, pyRadBio's explicit-track Monte Carlo.
#
# **The explicit-track MC is not run here.** `scripts/lem4_friedrich2012_mc.py` (~2 h)
# writes `validation/data/lem4_friedrich2012_mc.csv`; this script overlays it when the
# file exists and prints a one-line note when it does not. It samples real track
# configurations and reads alpha_I *and* beta_I off one simulated survival curve under
# the exact `<exp(-E)>` convention, the Scholz 1997 prescription. Two other MC scorings
# (GLOBLE, and the integer-count `photon_equivalent` one) are not drawn; see
# `docs/references/LEM_IV_Friedrich2012.md`.
#
# **Panels 1-3 pin `legacy_trip98_cluster_index_cap=0.95`**, as their golden tests do. The
# cap is a constant tuned on the TRiP98 carbon curves and in no paper. Uncapped, the
# giant-loop lattice registration of the mean field puts a spurious local RBE maximum at
# LET ~430-440 (`docs/references/LEM_IV_OPEN_CHOICES.md`, entries 1.1 and 3.1). It binds
# only on the two carbon tails (Fig. 3 from LET 397, Fig. 4b from LET 419), so every other
# series is identical either way.
#
# Golden tests: `test_golden_lem4_reproduces_elsasser2010_v79_figure4a`,
# `..._elsasser2010_hsg_figure4b`, `..._friedrich2012_v79_carbon_figure3` and
# `..._friedrich2012_fig4_approximation`.

# %%
from __future__ import annotations

from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
from _common import pick, quick_mode

from pyradbio import Ion
from pyradbio.benchmark import elsasser2010 as es
from pyradbio.benchmark import friedrich2012 as f12
from pyradbio.benchmark.metrics import ratio_summary
from pyradbio.benchmark.sweeps import sweep_lets
from pyradbio.models.lem.lem4 import LEM4
from pyradbio.plotting import figures_dir, save_figure
from pyradbio.track.stopping_power import (
    default_stopping_power,
    describe_stopping_power,
    energy_for_let_MeV_u,
)

QUICK = quick_mode()
FIGURES = figures_dir(globals().get("__file__"))
_HERE = globals().get("__file__")
MC_CSV = (Path(_HERE).resolve().parent if _HERE else Path.cwd()) / "data" / f12.FIG4_MC_CSV_NAME

sp = default_stopping_power()
print(f"stopping power: {describe_stopping_power(sp)}")
if QUICK:
    print("quick mode (PYRADBIO_QUICK=1): coarse grids, no full simulations; not reference values")

# Panels 4-5 overlay two deterministic full simulations on the approximation. The
# LOCAL-DOSE CONVOLUTION is the recommended one: the local dose is a compound-Poisson sum
# over the track field, so its density follows from an exact characteristic function and
# there is no seed noise; it recovers beta_x at low LET and falls with LET. ~0.6 s per LET
# with the numba kernel (the `fast` extra), ~9 s with NumPy. The DISCRETE-DSB engine
# (~1 s per LET) is the stochastic <exp(-E)> reference for its analytic overkill.
LOCAL_DOSE_LETS = np.geomspace(10.0, 150.0, 50)
RUN_LOCAL_DOSE = pick(True, False)
LOCAL_DOSE_ENGINE = "auto"  # numba box-average kernel when the ``fast`` extra is present
RUN_DISCRETE = pick(True, False)
# With the MC checkpoint present, the (a)/(b) comparisons cost two local-dose fits per
# checkpoint LET (~1 s each with numba, ~9 s with NumPy); quick mode takes every 6th.
MC_COMPARE_EVERY = pick(1, 6)
LET_BOUNDS_MEV_U = f12.FIG4_MC_ENERGY_BOUNDS_MEV_U  # carbon/helium/proton bracket (0.4, 600)


def ion_for_let(z: int, a: int, let_keV_um: float) -> Ion | None:
    """Ion ``(z, a)`` whose water LET equals ``let_keV_um``, or ``None`` if unreachable."""
    try:
        e = energy_for_let_MeV_u(
            sp, z, a, let_keV_um, branch="bracket", energy_bounds_MeV_u=LET_BOUNDS_MEV_U
        )
    except ValueError:
        return None
    return Ion(z, a, e)


def sweep(
    lem: LEM4, z: int, a: int, alpha_x: float, let_grid: np.ndarray
) -> tuple[np.ndarray, np.ndarray]:
    """``(RBE_alpha, RBE_10)`` over ``let_grid``, NaN where LET is unreachable.

    RBE_10 takes the photon D10 off the *shouldered* curve (``rbe10_shouldered``):
    inverting the LQ branch above D_t would deflate it whenever D_10 > D_t (14.5% for
    the Elsasser 2010 V79 cell). The panels here all sit below their D_t.
    """
    s = sweep_lets(
        lem, z, a, let_grid, stopping_power=sp, energy_bounds_MeV_u=LET_BOUNDS_MEV_U, on_error="nan"
    )
    return s.rbe_alpha(alpha_x), s.rbe10_shouldered(lem.photon)


def alpha_beta_sweep(
    lem: LEM4, z: int, a: int, let_grid: np.ndarray
) -> tuple[np.ndarray, np.ndarray]:
    """Raw ``(alpha_I, beta_I)`` over ``let_grid``, NaN where LET is unreachable."""
    s = sweep_lets(
        lem, z, a, let_grid, stopping_power=sp, energy_bounds_MeV_u=LET_BOUNDS_MEV_U, on_error="nan"
    )
    return s.alpha, s.beta


def local_dose_full_sweep(lem: LEM4, lets: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
    """Carbon ``(alpha_I, beta_I)`` from ONE low-dose LQ fit of the local-dose response
    (:meth:`LEM4.local_dose_alpha_beta`), ``beta_I`` clamped to >= 0 as plotted.

    With ``local_dose_overkill=True`` the single-track Poisson overkill is kept
    analytically (``<exp(-E)> = sum_n P(n) exp(-E(n D_1))``); the plain convolution's
    ``exp(-<E>)`` misses it, which by LET ~130 lifts alpha_I ~20% over the paper and
    drives beta_I through zero."""
    alpha = np.full_like(lets, np.nan)
    beta = np.full_like(lets, np.nan)
    for i, let in enumerate(lets):
        ion = ion_for_let(6, 12, float(let))
        if ion is None:
            continue
        a, b = lem.local_dose_alpha_beta(ion)
        alpha[i] = a
        beta[i] = max(b, 0.0)
    return alpha, beta


def discrete_overkill_sweep(lem: LEM4, lets: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
    """Carbon ``(alpha_I, beta_I)`` from the DISCRETE-DSB engine with the stochastic
    overkill ``S = <exp(-E)> = sum_n P(n) exp(-E_n)``, ``E_n`` the pattern's effect at
    exactly ``n`` traversals (:meth:`LEM4.discrete_dsb_alpha_beta`); neither clamped."""
    alpha = np.full_like(lets, np.nan)
    beta = np.full_like(lets, np.nan)
    for i, let in enumerate(lets):
        ion = ion_for_let(6, 12, float(let))
        if ion is None:
            continue
        alpha[i], beta[i] = lem.discrete_dsb_alpha_beta(
            ion, convention="per_realisation", survival="stochastic"
        )
    return alpha, beta


def rms_vs_curve(let_valid: np.ndarray, model_valid: np.ndarray, xs: object, ys: object) -> float:
    """RMS of ``model_valid(LET)`` interpolated onto the reference points ``(xs, ys)``."""
    in_range = [(x, y) for x, y in zip(xs, ys, strict=True) if let_valid[0] <= x <= let_valid[-1]]
    model_at = np.interp([x for x, _ in in_range], let_valid, model_valid)
    resid = model_at - np.array([y for _, y in in_range])
    return float(np.sqrt(np.mean(resid**2)))


def rel_vs_curve(
    let_valid: np.ndarray, model_valid: np.ndarray, xs: object, ys: object
) -> tuple[float, float]:
    """``(relRMS %, bias %)`` -- the *sign* is what separates a scale offset from scatter."""
    in_range = [(x, y) for x, y in zip(xs, ys, strict=True) if let_valid[0] <= x <= let_valid[-1]]
    model_at = np.interp([x for x, _ in in_range], let_valid, model_valid)
    ref = np.array([y for _, y in in_range])
    summary = ratio_summary(model_at, ref, mask=ref != 0)
    return summary.rms_rel * 100.0, summary.mean_rel * 100.0


# Five panels: three on top, two (wider) below.
fig = plt.figure(figsize=(17.5, 10.6))
gs = fig.add_gridspec(2, 6)
ax_a = fig.add_subplot(gs[0, 0:2])
ax_b = fig.add_subplot(gs[0, 2:4])
ax_d = fig.add_subplot(gs[0, 4:6])
ax_e = fig.add_subplot(gs[1, 0:3])
ax_f = fig.add_subplot(gs[1, 3:6])

# %% [markdown]
# ## Panel 1: Elsasser 2010 Fig. 4a -- V79, protons + helium

# %%
V79_ES = es.FIGURE_INPUTS["fig4a_v79"]
photon_a = V79_ES.photon()
cell_a = V79_ES.cell_target()
# Capped like this panel's golden test; it binds nowhere here (max C_I 0.275 for
# protons, 0.733 for helium).
lem_a = LEM4(photon=photon_a, cell=cell_a, legacy_trip98_cluster_index_cap=0.95)

let_p = np.geomspace(4.0, 40.0, pick(40, 10))
let_he = np.geomspace(15.0, 110.0, pick(30, 8))
rbe_p, _ = sweep(lem_a, 1, 1, V79_ES.alpha_x, let_p)
rbe_he, _ = sweep(lem_a, 2, 4, V79_ES.alpha_x, let_he)

p_model = es.get_curve("fig4a_v79_rbealpha_vs_let", particle="proton", series="calc")
he_model = es.get_curve("fig4a_v79_rbealpha_vs_let", particle="helium", series="calc")
p_data = es.get_curve("fig4a_v79_rbealpha_vs_let", particle="proton", series="data")
he_data = es.get_curve("fig4a_v79_rbealpha_vs_let", particle="helium", series="data")

ax_a.plot(let_p, rbe_p, "-", color="C3", lw=2.0, label=r"pyRadBio RBE$_\alpha$ (p)")
ax_a.plot(let_he, rbe_he, "-", color="C0", lw=2.0, label=r"pyRadBio RBE$_\alpha$ (He)")
ax_a.plot(p_model.x, p_model.y, "-", color="0.35", lw=1.2, label="paper p (Fig. 4a)")
ax_a.plot(he_model.x, he_model.y, "--", color="0.35", lw=1.2, label="paper He (Fig. 4a)")
ax_a.errorbar(
    p_data.x,
    p_data.y,
    yerr=p_data.y_err,
    fmt="^",
    mfc="none",
    mec="C3",
    mew=1.1,
    ms=5,
    ecolor="C3",
    elinewidth=0.8,
    capsize=2,
    label="Belli 1998 p data",
)
ax_a.plot(
    he_data.x, he_data.y, "s", mfc="none", mec="C0", mew=1.1, ms=5, label="Belli 1998 He data"
)
ax_a.set_xscale("log")
ax_a.set_xlabel("LET (keV/µm)")
ax_a.set_ylabel(r"RBE$_\alpha$")
ax_a.set_title("Elsasser 2010 Fig. 4a: V79, p + He")
ax_a.legend(frameon=False, fontsize=7, loc="upper left")
ax_a.grid(True, which="both", ls=":", lw=0.45)
ax_a.set_ylim(0.0, 11.5)

# %% [markdown]
# ## Panel 2: Elsasser 2010 Fig. 4b -- HSG carbon + helium, default `D_eq` model

# %%
HSG = es.FIGURE_INPUTS["fig4b_hsg"]
photon_b = HSG.photon()
cell_b = HSG.cell_target()
# effect_model="photon_equivalent" (default); capped like this panel's golden test --
# it binds on the carbon tail from LET 419 and nowhere on helium (max C_I 0.854).
lem_deq = LEM4(photon=photon_b, cell=cell_b, legacy_trip98_cluster_index_cap=0.95)

LET_GRID_C = np.geomspace(6.0, 900.0, pick(60, 12))
LET_GRID_HE = np.geomspace(4.0, 130.0, pick(60, 12))
rbe_alpha_deq, rbe10_deq = sweep(lem_deq, 6, 12, HSG.alpha_x, LET_GRID_C)
rbe_alpha_deq_he, rbe10_deq_he = sweep(lem_deq, 2, 4, HSG.alpha_x, LET_GRID_HE)

# The paper's own method for this panel is the full simulation ("the method of the
# original LEM (13)"), so only RBE_10 is overlaid: RBE_alpha depends on initial_slope
# alone and is beta_model-independent. The bias it moves is computed below, never
# quoted: it depends on the LET grid and the stopping power.
rbe10_full = rbe10_full_he = None
if RUN_LOCAL_DOSE:
    lem_full = LEM4(
        photon=photon_b,
        cell=cell_b,
        beta_model="local_dose",
        engine=LOCAL_DOSE_ENGINE,
        legacy_trip98_cluster_index_cap=0.95,  # same panel, same cap as lem_deq
    )
    grid_c = np.geomspace(LET_GRID_C[0], LET_GRID_C[-1], 24)
    grid_he = np.geomspace(LET_GRID_HE[0], LET_GRID_HE[-1], 24)
    rbe10_full = (grid_c, sweep(lem_full, 6, 12, HSG.alpha_x, grid_c)[1])
    rbe10_full_he = (grid_he, sweep(lem_full, 2, 4, HSG.alpha_x, grid_he)[1])

paper_alpha = es.get_curve(
    "fig4b_hsg_rbe_vs_let", particle="carbon", series="calc", variant="model_rbe_alpha"
)
paper_rbe10 = es.get_curve(
    "fig4b_hsg_rbe_vs_let", particle="carbon", series="calc", variant="model_rbe_10"
)
data_alpha = es.get_curve(
    "fig4b_hsg_rbe_vs_let", particle="carbon", series="data", variant="rbe_alpha"
)
data_rbe10 = es.get_curve(
    "fig4b_hsg_rbe_vs_let", particle="carbon", series="data", variant="rbe_10"
)
paper_alpha_he = es.get_curve(
    "fig4b_hsg_rbe_vs_let", particle="helium", series="calc", variant="model_rbe_alpha"
)
paper_rbe10_he = es.get_curve(
    "fig4b_hsg_rbe_vs_let", particle="helium", series="calc", variant="model_rbe_10"
)
data_alpha_he = es.get_curve(
    "fig4b_hsg_rbe_vs_let", particle="helium", series="data", variant="rbe_alpha"
)
data_rbe10_he = es.get_curve(
    "fig4b_hsg_rbe_vs_let", particle="helium", series="data", variant="rbe_10"
)

ax_b.plot(LET_GRID_C, rbe_alpha_deq, "-", color="C3", lw=2.0, label=r"pyRadBio RBE$_\alpha$ (C)")
ax_b.plot(LET_GRID_C, rbe10_deq, "--", color="C3", lw=2.0, label=r"pyRadBio RBE$_{10}$ (C)")
ax_b.plot(
    LET_GRID_HE, rbe_alpha_deq_he, "-", color="C0", lw=2.0, label=r"pyRadBio RBE$_\alpha$ (He)"
)
ax_b.plot(LET_GRID_HE, rbe10_deq_he, "--", color="C0", lw=2.0, label=r"pyRadBio RBE$_{10}$ (He)")
if rbe10_full is not None and rbe10_full_he is not None:
    ax_b.plot(
        *rbe10_full,
        ":",
        color="C3",
        lw=2.2,
        label=r"RBE$_{10}$, full sim. (paper's own method)",
    )
    ax_b.plot(*rbe10_full_he, ":", color="C0", lw=2.2)
ax_b.plot(paper_alpha.x, paper_alpha.y, "-", color="0.35", lw=1.0)
ax_b.plot(paper_rbe10.x, paper_rbe10.y, "--", color="0.35", lw=1.0)
ax_b.plot(paper_alpha_he.x, paper_alpha_he.y, "-", color="0.35", lw=1.0, label="paper (Fig. 4b)")
ax_b.plot(paper_rbe10_he.x, paper_rbe10_he.y, "--", color="0.35", lw=1.0)
ax_b.plot(
    data_alpha.x, data_alpha.y, "o", mfc="none", mec="C3", mew=1.0, ms=4.5, label="Furusawa C"
)
ax_b.plot(data_rbe10.x, data_rbe10.y, "o", mfc="C3", mec="C3", ms=3.5)
ax_b.errorbar(
    data_alpha_he.x,
    data_alpha_he.y,
    yerr=data_alpha_he.y_err,
    fmt="s",
    mfc="none",
    mec="C0",
    mew=1.0,
    ms=4.5,
    ecolor="C0",
    elinewidth=0.8,
    capsize=2,
    label="Furusawa He",
)
ax_b.errorbar(
    data_rbe10_he.x,
    data_rbe10_he.y,
    yerr=data_rbe10_he.y_err,
    fmt="s",
    mfc="C0",
    mec="C0",
    ms=3.5,
    ecolor="C0",
    elinewidth=0.8,
    capsize=2,
)
ax_b.set_xscale("log")
ax_b.set_xlabel("LET (keV/µm)")
ax_b.set_ylabel("RBE")
ax_b.set_title(r"Elsasser 2010 Fig. 4b: HSG, C + He ($D_{eq}$, default)")
ax_b.legend(frameon=False, fontsize=7, loc="upper right")
ax_b.grid(True, which="both", ls=":", lw=0.45)
ax_b.set_ylim(0.0, 11.5)

# %% [markdown]
# ## Panel 3: Friedrich 2012 Fig. 3 -- V79, carbon

# %%
V79_F12 = f12.FIGURE_INPUTS["fig3_v79"]
photon_f3 = V79_F12.photon()
cell_f3 = V79_F12.cell_target()
# Capped like this panel's golden test; it binds on the carbon tail from LET 397.
lem_f3 = LEM4(photon=photon_f3, cell=cell_f3, legacy_trip98_cluster_index_cap=0.95)

let_f3 = np.geomspace(8.0, 800.0, pick(80, 16))
rbe_alpha_f3, rbe10_f3 = sweep(lem_f3, 6, 12, V79_F12.alpha_x, let_f3)

f3_model_alpha = f12.get_curve("fig3_v79_rbe_vs_let", series="calc", variant="rbe_alpha")
f3_model_rbe10 = f12.get_curve("fig3_v79_rbe_vs_let", series="calc", variant="rbe_10")
f3_data_alpha = f12.get_curve("fig3_v79_rbe_vs_let", series="data", variant="rbe_alpha")
f3_data_rbe10 = f12.get_curve("fig3_v79_rbe_vs_let", series="data", variant="rbe_10")

ax_d.plot(let_f3, rbe_alpha_f3, "-", color="C3", lw=2.0, label=r"pyRadBio RBE$_\alpha$")
ax_d.plot(let_f3, rbe10_f3, "--", color="C3", lw=2.0, label=r"pyRadBio RBE$_{10}$")
ax_d.plot(f3_model_alpha.x, f3_model_alpha.y, "-", color="0.35", lw=1.2, label="paper (Fig. 3)")
ax_d.plot(f3_model_rbe10.x, f3_model_rbe10.y, "--", color="0.35", lw=1.2)
ax_d.plot(
    f3_data_alpha.x,
    f3_data_alpha.y,
    "o",
    mfc="none",
    mec="C3",
    mew=1.0,
    ms=4.5,
    label=r"Furusawa 2000 RBE$_\alpha$",
)
ax_d.plot(
    f3_data_rbe10.x,
    f3_data_rbe10.y,
    "s",
    mfc="none",
    mec="C0",
    mew=1.0,
    ms=4.5,
    label=r"Furusawa 2000 RBE$_{10}$",
)
ax_d.set_xscale("log")
ax_d.set_xlabel("LET (keV/µm)")
ax_d.set_ylabel("RBE")
ax_d.set_title("Friedrich 2012 Fig. 3: V79, carbon")
ax_d.legend(frameon=False, fontsize=7, loc="upper right")
ax_d.grid(True, which="both", ls=":", lw=0.45)
ax_d.set_ylim(0.0, 11.0)

# %% [markdown]
# ## Panels 4-5: Friedrich 2012 Fig. 4 -- "hypothetical" cell, alpha_I and beta_I
#
# One colour scheme on both panels: **blue** is the single-particle **approximation**
# (line = pyRadBio, open markers = paper), **red** the recommended **local-dose full
# simulation**, **green** the **discrete-DSB full simulation**; black markers are the
# paper's own full simulation. With the MC checkpoint present, **purple** is the
# explicit-track MC and **orange** the mean-effect (`exp(-<E>)`) convolution at the MC
# LETs.
#
# Both deterministic full simulations apply the **single-track overkill** (a cell can be
# killed only once per track), at different levels: the local-dose engine
# (`local_dose_overkill=True`, the "B2" correction) keeps the Poisson track-*number*
# fluctuation analytically, while the discrete engine keeps the full pattern including the
# intertrack correlations. Both pull alpha_I back onto the paper's curve (RMS 0.16 ->
# 0.06). The local-dose beta is clamped to >= 0: its fit dips marginally negative past
# ~150 keV/um, where the intertrack variance it omits supplies the deep-overkill convexity
# the discrete engine keeps. This is the LEM analogue of the modified-MKM
# saturation-corrected specific energy, keyed to the track's derived `N_1` rather than a
# fitted `z_0`. The MC samples the same exact `<exp(-E)>` instead of approximating it,
# which is what makes B2 checkable: it over-corrects (see
# `docs/references/LEM_IV_Friedrich2012.md`).

# %%
HYP = f12.FIGURE_INPUTS["fig4_hypothetical"]
photon_f4 = HYP.photon()
cell_f4 = HYP.cell_target()
lem_f4 = LEM4(photon=photon_f4, cell=cell_f4)

let_f4 = np.geomspace(9.0, 140.0, pick(60, 12))
alpha_f4, beta_f4 = alpha_beta_sweep(lem_f4, 6, 12, let_f4)

# Both full simulations give alpha_I and beta_I from one low-dose LQ fit per LET, on the
# same window (LEM4.fit_doses_Gy).
alpha_ld_ovk = beta_ld_ovk = alpha_disc = beta_disc = None
if RUN_LOCAL_DOSE:
    lem_f4_ld = LEM4(
        photon=photon_f4,
        cell=cell_f4,
        stopping_power=sp,
        engine=LOCAL_DOSE_ENGINE,
        local_dose_overkill=True,
    )
    print("\nFriedrich 2012 Fig. 4, full simulation with single-track overkill (the cell can be")
    print("killed only once per track). Two engines: 'local dose' is the fast analytic overkill")
    print(f"(~0.6 s/pt, engine={LOCAL_DOSE_ENGINE!r}); 'discrete' is the stochastic <exp(-E)>")
    print("reference (~1 s/pt). Set RUN_LOCAL_DOSE/RUN_DISCRETE=False to skip:")
    alpha_ld_ovk, beta_ld_ovk = local_dose_full_sweep(lem_f4_ld, LOCAL_DOSE_LETS)
    if RUN_DISCRETE:
        lem_f4_disc = LEM4(
            photon=photon_f4,
            cell=cell_f4,
            stopping_power=sp,
            beta_model="discrete_dsb",
            engine=LOCAL_DOSE_ENGINE,
        )
        alpha_disc, beta_disc = discrete_overkill_sweep(lem_f4_disc, LOCAL_DOSE_LETS)
    for k, let in enumerate(LOCAL_DOSE_LETS):
        disc = (
            f"   discrete: alpha={alpha_disc[k]:.4f} beta={beta_disc[k]:+.4f}"
            if alpha_disc is not None
            else ""
        )
        print(
            f"  LET {let:6.1f}  local dose: alpha={alpha_ld_ovk[k]:.4f} "
            f"beta={beta_ld_ovk[k]:+.4f}{disc}"
        )

f4_approx_alpha = f12.get_curve("fig4_hypothetical_vs_let", series="approx", variant="alpha")
f4_approx_beta = f12.get_curve("fig4_hypothetical_vs_let", series="approx", variant="beta")
f4_full_alpha = f12.get_curve("fig4_hypothetical_vs_let", series="full_simulation", variant="alpha")
f4_full_beta = f12.get_curve("fig4_hypothetical_vs_let", series="full_simulation", variant="beta")

ax_e.plot(let_f4, alpha_f4, "-", color="C0", lw=2.0, label=r"pyRadBio $\alpha_I$ (approx.)")
ax_e.plot(
    f4_approx_alpha.x,
    f4_approx_alpha.y,
    "o",
    mfc="none",
    mec="C0",
    mew=1.0,
    ms=4.5,
    label="paper (approx.)",
)
ax_e.plot(f4_full_alpha.x, f4_full_alpha.y, "x", color="0.15", ms=5.5, label="paper (full sim.)")
if alpha_ld_ovk is not None:
    ax_e.plot(
        LOCAL_DOSE_LETS,
        alpha_ld_ovk,
        "o-",
        color="C3",
        lw=1.8,
        ms=4,
        label="pyRadBio full sim. (local dose)",
    )
if alpha_disc is not None:
    ax_e.plot(
        LOCAL_DOSE_LETS,
        alpha_disc,
        "^--",
        color="C2",
        lw=1.6,
        ms=4,
        label="pyRadBio full sim. (discrete)",
    )
ax_e.set_xscale("log")
ax_e.set_xlabel("LET (keV/µm)")
ax_e.set_ylabel(r"$\alpha_I$ (Gy$^{-1}$)")
ax_e.set_title(r"Friedrich 2012 Fig. 4: hypothetical cell, $\alpha_I$")
ax_e.legend(frameon=False, fontsize=7, loc="upper left")
ax_e.grid(True, which="both", ls=":", lw=0.45)

ax_f.plot(let_f4, beta_f4, "-", color="C0", lw=2.0, label=r"pyRadBio $\beta_I$ (approx.)")
ax_f.plot(
    f4_approx_beta.x,
    f4_approx_beta.y,
    "s",
    mfc="none",
    mec="C0",
    mew=1.0,
    ms=4.5,
    label="paper (approx.)",
)
ax_f.plot(f4_full_beta.x, f4_full_beta.y, "+", color="0.15", ms=6, label="paper (full sim.)")

if beta_ld_ovk is not None:
    ax_f.plot(
        LOCAL_DOSE_LETS,
        beta_ld_ovk,
        "o-",
        color="C3",
        lw=1.8,
        ms=4,
        label=r"pyRadBio full sim. (local dose, B2 $\approx\langle e^{-E}\rangle$)",
    )
if beta_disc is not None:
    ax_f.plot(
        LOCAL_DOSE_LETS,
        beta_disc,
        "^--",
        color="C2",
        lw=1.6,
        ms=4,
        label="pyRadBio full sim. (discrete)",
    )

# %% [markdown]
# ### The explicit-track MC, read from its checkpoint
#
# `validation/data/lem4_friedrich2012_mc.csv` (written by `scripts/lem4_friedrich2012_mc.py`)
# holds alpha_I and beta_I per LET and seed, each pair off ONE simulated survival curve,
# all on the SAME fit window as every other curve on Panels 4-5: a fitted beta is a
# window projection, so mixing windows on one panel compares different estimators. The
# bars are the plain scatter of the seed mean; beta is a small curvature on a dominant
# linear term (at LET 100, three seeds of 400 gave 0.0049 / 0.0071 / 0.0371), and the
# deep-overkill points are left unclamped so the bars mean what they say.
#
# The mean-effect convolution is plotted next to it because the beta AMPLITUDE is a
# survival-convention artefact while the peak LOCATION is invariant: over LET 15-40 the
# deterministic `exp(-<E>)` curve reproduces Fig. 4's rise (RMS 1.9%, +3.1% against the
# paper's +3.5%, peak at LET 28 against 24), so the panel shows the bracket, not one edge
# (`docs/references/LEM_IV_Friedrich2012.md`, `LEM_IV_OPEN_CHOICES.md` entry 2.1).

# %%
if MC_CSV.exists():
    mc = f12.load_fig4_mc(MC_CSV)
    MC_LETS = mc.let_keV_um
    alpha_mc, alpha_mc_err, beta_mc, beta_mc_err = mc.alpha, mc.alpha_err, mc.beta, mc.beta_err
    lem_f4_mc = f12.fig4_mc_model(engine=LOCAL_DOSE_ENGINE, stopping_power=sp)
    mc_doses = lem_f4_mc.fit_doses_Gy()
    print(
        f"\nFriedrich 2012 Fig. 4: explicit-track MC, alpha_I AND beta_I off one simulated\n"
        f"survival curve (exact <exp(-E)>, the Scholz 1997 prescription), window "
        f"{mc_doses[0]:.2f}-{mc_doses[-1]:.2f} Gy,\n{mc.header.get('n_seeds', '?')} seeds x "
        f"{mc.header.get('n_realizations', '?')} realisations (from {MC_CSV.name}, "
        f"{mc.header.get('engine', '?')}):"
    )
    for let, am, ae, bm, be in zip(
        MC_LETS, alpha_mc, alpha_mc_err, beta_mc, beta_mc_err, strict=True
    ):
        print(f"    LET {let:5.0f}: alpha={am:.4f}+-{ae:.4f}  beta={bm:+.4f}+-{be:.4f}")
    if mc.incomplete_lets:
        print(f"    (incomplete in the checkpoint, not shown: LET {list(mc.incomplete_lets)})")

    # Two comparisons the reader should see rather than take on trust.
    lem_f4_b2 = LEM4(
        photon=photon_f4,
        cell=cell_f4,
        stopping_power=sp,
        engine=LOCAL_DOSE_ENGINE,
        local_dose_overkill=True,
    )
    sel = slice(None, None, MC_COMPARE_EVERY)  # every checkpoint LET in full mode
    cmp_lets = MC_LETS[sel]
    mc_ions = [ion_for_let(6, 12, float(let)) for let in cmp_lets]
    det_mean = np.array([lem_f4_mc.local_dose_beta(i, doses_Gy=mc_doses) for i in mc_ions])
    det_b2 = np.array([lem_f4_b2.local_dose_beta(i, doses_Gy=mc_doses) for i in mc_ions])
    reported_alpha = np.array([lem_f4_mc.initial_slope(i) for i in mc_ions])
    print(
        "\n  (a) alpha: the MC fits it to the same curve it fits beta to, so it is a check on\n"
        "      whether the reported single-particle alpha_I and the simulated one agree --\n"
        "      the alpha/beta split Elsasser 2010's 'method of the original LEM' does not make."
    )
    for let, am, ae, ar in zip(
        cmp_lets, alpha_mc[sel], alpha_mc_err[sel], reported_alpha, strict=True
    ):
        print(
            f"      LET {let:5.0f}  MC {am:.4f}+-{ae:.4f}   reported {ar:.4f}   "
            f"({100.0 * (am / ar - 1.0):+.1f}%)"
        )
    print(
        "\n  (b) beta: the analytic B2 overkill correction (the plotted 'local dose' curve)\n"
        "      approximates the same <exp(-E)> the MC computes exactly -- by evaluating the\n"
        "      mean-effect curve at n*D_1. It OVER-corrects, so the MC sits above it."
    )
    for let, bm, be, dm, db in zip(
        cmp_lets, beta_mc[sel], beta_mc_err[sel], det_mean, det_b2, strict=True
    ):
        print(
            f"      LET {let:5.0f}  mean effect {dm:+.4f}   B2 {db:+.4f}   "
            f"MC exact {bm:+.4f}+-{be:.4f}"
        )
    ax_f.plot(
        cmp_lets,
        det_mean,
        "d:",
        color="C1",
        lw=1.6,
        ms=4.5,
        label=r"pyRadBio full sim. (mean effect, $e^{-\langle E\rangle}$)",
    )
    ax_f.errorbar(
        MC_LETS,
        beta_mc,
        yerr=beta_mc_err,
        fmt="v",
        color="C4",
        lw=1.2,
        ms=5,
        capsize=2.5,
        label=r"pyRadBio MC (explicit tracks, $\langle e^{-E}\rangle$)",
    )
    ax_e.errorbar(
        MC_LETS,
        alpha_mc,
        yerr=alpha_mc_err,
        fmt="v",
        color="C4",
        lw=1.2,
        ms=5,
        capsize=2.5,
        label="pyRadBio MC (explicit tracks)",
    )
else:
    print(
        f"\nFriedrich 2012 Fig. 4 explicit-track MC: {MC_CSV.name} not generated yet, "
        "overlay skipped (run scripts/lem4_friedrich2012_mc.py)"
    )

ax_f.plot(let_f4, beta_f4, "-", color="C0", lw=2.0)  # redraw approx over MC for clarity
ax_f.set_xscale("log")
ax_f.set_xlabel("LET (keV/µm)")
ax_f.set_ylabel(r"$\beta_I$ (Gy$^{-2}$)")
ax_f.set_title(r"Friedrich 2012 Fig. 4: hypothetical cell, $\beta_I$")
# Everything here lives below 0.06; the lower bound leaves room for the unclamped
# deep-overkill MC points, whose error bars genuinely cross zero.
ax_f.set_ylim(-0.012, 0.062)
ax_f.legend(frameon=False, fontsize=7, loc="upper right")
ax_f.grid(True, which="both", ls=":", lw=0.45)

fig.suptitle("LEM IV vs Elsasser et al. 2010 and Friedrich et al. 2012", y=1.0)
fig.tight_layout()
print()
save_figure(fig, "lem4_reconstruction.png", directory=FIGURES, dpi=150, bbox_inches="tight")

# %% [markdown]
# ## RMS against the digitized curves
#
# The same comparisons as the four golden tests named at the top.

# %%
print("\nElsasser 2010 Fig. 4a (V79):")
print("  (proton RMS below include LET>25 keV/um, where LEM4 enters deep overkill --")
print("   see the module docstring; the golden test clips its comparison to LET<=25)")
valid_p = ~np.isnan(rbe_p)
print(
    "  proton RBE_alpha vs paper model line: RMS ="
    f" {rms_vs_curve(let_p[valid_p], rbe_p[valid_p], p_model.x, p_model.y):.3f}"
)
print(
    "  proton RBE_alpha vs Belli 1998 data:  RMS ="
    f" {rms_vs_curve(let_p[valid_p], rbe_p[valid_p], p_data.x, p_data.y):.3f}"
)
valid_he = ~np.isnan(rbe_he)
print(
    "  helium RBE_alpha vs paper model line: RMS ="
    f" {rms_vs_curve(let_he[valid_he], rbe_he[valid_he], he_model.x, he_model.y):.3f}"
)
print(
    "  helium RBE_alpha vs Belli 1998 data:  RMS ="
    f" {rms_vs_curve(let_he[valid_he], rbe_he[valid_he], he_data.x, he_data.y):.3f}"
)

print("\nElsasser 2010 Fig. 4b (HSG), by effect model:")
particle_grids = {
    "carbon": (
        LET_GRID_C,
        rbe_alpha_deq,
        rbe10_deq,
        paper_alpha,
        paper_rbe10,
        data_alpha,
        data_rbe10,
    ),
    "helium": (
        LET_GRID_HE,
        rbe_alpha_deq_he,
        rbe10_deq_he,
        paper_alpha_he,
        paper_rbe10_he,
        data_alpha_he,
        data_rbe10_he,
    ),
}
for particle, (let_grid, ra, r10, p_alpha, p_r10, d_alpha, d_r10) in particle_grids.items():
    valid = ~np.isnan(ra)
    let_valid = let_grid[valid]
    print(f"  D_eq (default), {particle}:")
    for label, curve, model_valid in (
        ("    RBE_alpha vs paper model line", p_alpha, ra[valid]),
        ("    RBE_10    vs paper model line", p_r10, r10[valid]),
        ("    RBE_alpha vs Furusawa data    ", d_alpha, ra[valid]),
        ("    RBE_10    vs Furusawa data    ", d_r10, r10[valid]),
    ):
        rms = rms_vs_curve(let_valid, model_valid, curve.x, curve.y)
        print(f"{label}: RMS = {rms:.3f} (peak {max(curve.y):.2f})")

# The alpha/beta split of the HSG residual: RBE_alpha depends only on initial_slope and is
# identical for every beta_model, so the split separates the two cleanly. Quote the BIAS,
# not just the RMS: a near-pure one-signed bias is a scale offset, not scatter.
print("\n  The HSG alpha/beta split (vs the paper's own model lines, signed):")
for particle, (let_grid, ra, r10, p_alpha, p_r10, full) in {
    "carbon": (LET_GRID_C, rbe_alpha_deq, rbe10_deq, paper_alpha, paper_rbe10, rbe10_full),
    "helium": (
        LET_GRID_HE,
        rbe_alpha_deq_he,
        rbe10_deq_he,
        paper_alpha_he,
        paper_rbe10_he,
        rbe10_full_he,
    ),
}.items():
    valid = ~np.isnan(ra)
    lv = let_grid[valid]
    a_rms, a_bias = rel_vs_curve(lv, ra[valid], p_alpha.x, p_alpha.y)
    b_rms, b_bias = rel_vs_curve(lv, r10[valid], p_r10.x, p_r10.y)
    print(
        f"    {particle:7} RBE_alpha  relRMS {a_rms:5.2f}%  bias {a_bias:+6.2f}%"
        "   <- beta_model-independent"
    )
    print(
        f"    {particle:7} RBE_10     relRMS {b_rms:5.2f}%  bias {b_bias:+6.2f}%   (approximation)"
    )
    if full is not None:
        fv = ~np.isnan(full[1])
        f_rms, f_bias = rel_vs_curve(full[0][fv], full[1][fv], p_r10.x, p_r10.y)
        moved = abs(b_bias) - abs(f_bias)
        print(
            f"    {particle:7} RBE_10     relRMS {f_rms:5.2f}%  bias {f_bias:+6.2f}%"
            f"   (full sim. = the paper's own method; "
            f"{abs(moved):.1f} pp of |bias| {'removed' if moved > 0 else 'ADDED'})"
        )

# %% [markdown]
# On the Fig. 4b reference the RBE$_\alpha$ bias is under 1% on both ions, so
# there is no alpha deficit to explain. What remains is a small RBE$_{10}$ bias, and the
# full simulation makes it *larger* (it adds about 3 pp), i.e. `slope_matching` fits
# Fig. 4b's RBE$_{10}$ better than `local_dose`. The textual case for the full
# simulation stands -- the paper's "method of the original LEM (13)" -- but
# which engine is *faithful* and which one *fits* are separate questions: see
# `docs/references/LEM_IV_OPEN_CHOICES.md`, "Why RMS cannot pick the engine".

# %%
print("\nFriedrich 2012 Fig. 3 (V79 carbon):")
valid_f3 = ~np.isnan(rbe_alpha_f3)
let_valid_f3 = let_f3[valid_f3]
for label, curve, model_valid in (
    ("  RBE_alpha vs paper model line", f3_model_alpha, rbe_alpha_f3[valid_f3]),
    ("  RBE_10    vs paper model line", f3_model_rbe10, rbe10_f3[valid_f3]),
    ("  RBE_alpha vs Furusawa data    ", f3_data_alpha, rbe_alpha_f3[valid_f3]),
    ("  RBE_10    vs Furusawa data    ", f3_data_rbe10, rbe10_f3[valid_f3]),
):
    rms = rms_vs_curve(let_valid_f3, model_valid, curve.x, curve.y)
    print(f"{label}: RMS = {rms:.3f} (peak {max(curve.y):.2f})")

print("\nFriedrich 2012 Fig. 4 (hypothetical cell, single-particle approximation):")
valid_f4 = ~np.isnan(alpha_f4)
let_valid_f4 = let_f4[valid_f4]
print(
    "  alpha_I vs paper approximation: RMS ="
    f" {rms_vs_curve(let_valid_f4, alpha_f4[valid_f4], f4_approx_alpha.x, f4_approx_alpha.y):.4f}"
)
print(
    "  beta_I  vs paper approximation: RMS ="
    f" {rms_vs_curve(let_valid_f4, beta_f4[valid_f4], f4_approx_beta.x, f4_approx_beta.y):.5f}"
)

print("\nDone. Figure written to", FIGURES)
