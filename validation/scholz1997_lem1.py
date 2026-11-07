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
# # LEM I reconstruction: Scholz et al. 1997 (the model and its approximation)
#
# Scholz, Kellerer, Kraft-Weyrather & Kraft 1997, *Radiat. Environ. Biophys.*
# 36:59-66, doi:10.1007/s004110050055, derive the fast **approximation** of the 1996
# stochastic LEM used for treatment planning: initial slopes from *central*
# traversals only, and a linear-in-`z` rise of the survival-curve slope that fixes
# `beta` from `alpha` and the photon transition dose. The paper keeps two paths
# apart, and so does every panel here:
#
# * **exact** -- the full spatially explicit stochastic LEM, LQ over the displayed
#   dose range (all well below `D_t = 30 Gy`);
# * **approximation** -- the central-traversal / specific-energy fast path.
#
# The reference curves are **hand-digitized** (`pyradbio.benchmark.scholz1997`), so
# the comparison is quantitative. Two panels are *derived* rather than digitized,
# because the paper's own relations fix them exactly over the plotted range:
# Fig. 2 (`-ln S/D = alpha + beta D`) and Fig. 4 (`alpha_z + 2 beta_z z`) are
# straight lines stored as least-squares endpoint fits (R^2 >= 0.998), and Fig. 5's
# survival is `S = exp(-(alpha D + beta D^2))` -- exact curves from the Fig. 2 fit,
# approximation curves from the Fig. 1 approximation alpha (the overkill-saturated
# central slope) and the Fig. 3 approximation beta. Residuals are therefore not
# digitization noise. The 1996 paper is `validation/scholz_kraft1996_lem1.py`.
#
# ### Which method draws which line
#
# "Exact" in the *paper's* sense means its stochastic model; pyRadBio reconstructs
# it with a deterministic expansion in most panels and with the Monte Carlo where
# one is affordable.
#
# | panel | line | method | what it is |
# |---|---|---|---|
# | Fig. 1 | solid | `initial_slope` | exact impact-parameter cross-section |
# | Fig. 1 | dotted | `scholz1997_central_initial_slope` | eqs. (5)-(6) + missed-track halo |
# | Fig. 1 | stars | MC checkpoint `alpha` | full stochastic model, fitted over 0.5-3 Gy |
# | Fig. 3 | thick | `scholz1997_two_track_beta` | exact O(phi^2) cluster coefficient |
# | Fig. 3 | dashed | `beta_model="scholz1997_stochastic"` | z-domain fast path |
# | Fig. 3 | stars | MC checkpoint `beta` | full stochastic model, fitted over 0.5-3 Gy |
# | Fig. 4 left | solid | `scholz1997_specific_energy_survival` | z-conditioned MC |
# | Fig. 4 right | solid | `scholz1997_specific_energy_slope` | `alpha_z + 2 beta_z z`, eqs. 10-11 |
# | Figs. 2 & 5 | thick | `scholz1997_explicit_lq` | cross-section alpha + two-track beta |
# | Figs. 2 & 5 | dotted | `scholz1997_approximate_log_kills` | Poisson fast simulation |
#
# ### Which one is best, and where
#
# * **alpha**: `initial_slope` is exact and cheap -- always prefer it; the central
#   approximation is shown only because Fig. 1 has both.
# * **beta**: `scholz1997_two_track_beta` is the faithful reconstruction; the
#   z-domain `scholz1997_stochastic` is crude and mis-states the low-energy carbon
#   curvature. Do **not** use `beta_model="slope_matching"` (the `LEM1` default) for
#   these figures: it is a high-dose asymptotic identity that breaks the geometric
#   cross-section ceiling in the 1996 panels.
# * **two-track vs the MC stars** are *different quantities*: the exact O(phi^2)
#   cluster coefficient against a least-squares coefficient over 0.5-3 Gy, which
#   projects part of the cubic and higher terms onto `D^2`. They converge as the fit
#   window approaches the origin (proton 7 MeV/u: ratio 0.995 / 0.986 / 0.971 for
#   windows 0.25-1.5 / 0.5-3.0 / 1.0-6.0 Gy), so a percent-level gap between the
#   thick line and the stars is that projection, not an error in either.
#
# `LEM_STOCHASTIC` carries `beta_model="scholz1997_stochastic"`, but only the Fig. 3
# dashed line reads it; every other panel calls the method it wants explicitly.

# %%
from __future__ import annotations

import math
import os
from collections.abc import Callable
from functools import cache
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
from _common import pick, quick_mode
from matplotlib.lines import Line2D

from pyradbio import Ion
from pyradbio.benchmark.metrics import ShapeResiduals, shape_residuals
from pyradbio.benchmark.scholz1997 import (
    TWO_TRACK_FAST_QUADRATURE,
    Scholz1997Curve,
    cho_lem1,
    iter_lem1_curves,
    load_mc_reference,
    load_z_slope_reference,
)
from pyradbio.benchmark.stopping import select_stopping_power, stopping_power_label
from pyradbio.core.lq import LQParameters
from pyradbio.models.lem.lem1 import LEM1
from pyradbio.plotting import figures_dir, save_figure
from pyradbio.track.northcliffe_schilling import (
    ENERGY_RANGE_MEV_U,
    BetheOnly,
    NorthcliffeSchillingWithBetheTail,
)

FIGURES = figures_dir(globals().get("__file__"))
DATA = (Path(__file__).resolve().parent if "__file__" in globals() else Path.cwd()) / "data"
if quick_mode():
    print("quick mode (PYRADBIO_QUICK=1): coarse energy sweeps; numbers are not for quoting")

# %% [markdown]
# ## Stopping power
#
# Selected by `LEM1_STOPPING` exactly as in `validation/scholz_kraft1996_lem1.py`
# (`ns1970` default: the paper's Northcliffe & Schilling 1970 table to 12 MeV/u,
# analytic Bethe above, with the fallbacks counted and printed at the end; `bethe`;
# `pymkm[:SOURCE]`; `libdedx[:PROGRAM]`). The label is stamped into every title.

# %%
LEM1_STOPPING = os.environ.get("LEM1_STOPPING", "ns1970")
try:
    STOPPING_POWER = select_stopping_power(LEM1_STOPPING)
    STOPPING_POWER_LABEL = stopping_power_label(LEM1_STOPPING)
except ImportError:
    print(f"{LEM1_STOPPING!r} not importable; falling back to analytic Bethe")
    STOPPING_POWER, STOPPING_POWER_LABEL = BetheOnly(), "analytic Bethe (fallback)"

SPECIES = {"proton": (1, 1, "C0"), "carbon": (6, 12, "C1")}
# The z-domain (fast Scholz 1997) model for the paper's CHO cell; also the parameter
# carrier every other panel calls its explicit method on.
LEM_STOCHASTIC = cho_lem1(
    STOPPING_POWER, beta_model="scholz1997_stochastic", initial_slope_model="cross_section"
)
PHOTON = LEM_STOCHASTIC.photon


def scholz1997_ion(particle: str, energy_MeV_u: float) -> Ion:
    z, a, _ = SPECIES[particle]
    return Ion(z, a, energy_MeV_u)


# %% [markdown]
# ## Residual / shape diagnostics
#
# Every paper curve is compared at the paper's own x values with a *point* residual
# (RMS, bias) and a *shape* diagnostic: the best single factor `k` aligning model to
# paper, and `shape_fraction` = the RMS left after that scaling over the RMS before.
# Near 0 means a pure scale offset (stopping power, normalisation); near 1 means the
# shapes genuinely differ.


# %%
def compare(paper: Scholz1997Curve, model_fn: Callable[[np.ndarray], np.ndarray]) -> ShapeResiduals:
    """``shape_residuals`` of ``model_fn`` at the paper's own x values."""
    return shape_residuals(paper.y, model_fn(np.asarray(paper.x, dtype=float)))


def print_compare(title: str, stats: ShapeResiduals) -> None:
    print(
        f"  {title:34s} n={stats.n:2d}  RMS={stats.rms:.4g}  "
        f"bias={stats.rel_bias:+.1%}  scale k={stats.scale_k:.3f}  "
        f"shape={stats.shape_fraction:.2f}"
    )


# %% [markdown]
# ## Monte-Carlo overlay data
#
# The full stochastic LEM (`beta_model="scholz1997_montecarlo"`) is too slow to run
# inline, so `scripts/lem1_mc_reference.py alpha-beta` precomputes it into the
# tracked `validation/data/mc_results.csv`: the survival-fit `alpha` and `beta` over
# 0.5-3 Gy with delete-one jackknife errors over 8 streams, for 38 proton (1-200
# MeV/u) and carbon (1-400 MeV/u) points, all with weighted (Kellerer) sampling.
# They are the stars on Figs. 1 and 3. A row computed at another LET than this run
# plots is a different point, not a noisier one, so it is dropped loudly (rows
# computed with analytic Bethe while the curves use the 1970 table below 12 MeV/u
# would misplace every star under ~4 MeV/u: 13% in alpha at carbon 1 MeV/u).

# %%
_MC = load_mc_reference(
    DATA / "mc_results.csv", stopping_power_name=getattr(STOPPING_POWER, "name", None)
)
for _stored, _count in sorted(_MC.dropped.items()):
    print(
        f"  dropped {_count} MC rows computed at LET={_stored!r}; "
        f"this run plots {getattr(STOPPING_POWER, 'name', None)!r}. "
        "Rerun scripts/lem1_mc_reference.py alpha-beta to restore them."
    )
MC_RESULTS = _MC.rows


def mc_arrays(particle: str, key: str) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    rows = MC_RESULTS.get(particle, [])
    return (
        np.array([r["energy_MeV_u"] for r in rows]),
        np.array([r[key] for r in rows]),
        np.array([r[f"{key}_err"] for r in rows]),
    )


# %% [markdown]
# ## Fig. 1 -- initial slope alpha vs energy (protons, carbon)

# %%
ENERGY_SWEEP = np.geomspace(1.0, 400.0, pick(60, 15))


def alpha_exact(model: LEM1, particle: str, energies: np.ndarray) -> np.ndarray:
    return np.array([model.initial_slope(scholz1997_ion(particle, float(e))) for e in energies])


def alpha_central(model: LEM1, particle: str, energies: np.ndarray) -> np.ndarray:
    return np.array(
        [
            model.scholz1997_central_initial_slope(scholz1997_ion(particle, float(e)))
            for e in energies
        ]
    )


fig1, ax = plt.subplots(figsize=(7.6, 5.4))
for particle, (_z, _a, color) in SPECIES.items():
    ax.plot(
        ENERGY_SWEEP,
        alpha_exact(LEM_STOCHASTIC, particle, ENERGY_SWEEP),
        color=color,
        lw=2.0,
        label=f"{particle} pyRadBio (cross-section)",
    )
    ax.plot(
        ENERGY_SWEEP,
        alpha_central(LEM_STOCHASTIC, particle, ENERGY_SWEEP),
        color=color,
        lw=1.5,
        ls=":",
        label=f"{particle} pyRadBio (central approx.)",
    )
    for calc, marker in (("exact", "o"), ("approximation", "s")):
        for ref in iter_lem1_curves(
            figure="fig1_alpha_vs_energy", particle=particle, calculation=calc
        ):
            ax.plot(ref.x, ref.y, color=color, ls="None", marker=marker, ms=6, mfc="white")
    if MC_RESULTS.get(particle):
        e_mc, a_mc, a_err = mc_arrays(particle, "alpha")
        ax.errorbar(
            e_mc, a_mc, yerr=a_err, color=color, ls="None", marker="*", ms=12, capsize=3, zorder=6
        )

ax.set_xscale("log")
ax.set_xlim(0.8, 500)
ax.set_ylim(0.0, 2.0)
ax.set_xlabel("Energy (MeV/u)")
ax.set_ylabel(r"Initial slope $\alpha$ (Gy$^{-1}$)")
ax.set_title(f"Scholz 1997 Fig. 1: alpha vs energy (CHO)\nstopping power: {STOPPING_POWER_LABEL}")
ax.grid(True, which="both", ls=":", lw=0.5)
role_handles = [
    Line2D([0], [0], color="0.3", lw=2.0, label="line: pyRadBio cross-section"),
    Line2D([0], [0], color="0.3", lw=1.5, ls=":", label="dotted: pyRadBio central"),
    Line2D([0], [0], color="0.3", ls="None", marker="o", mfc="white", label="o: paper exact"),
    Line2D([0], [0], color="0.3", ls="None", marker="s", mfc="white", label="s: paper approx."),
]
if MC_RESULTS:
    role_handles.append(
        Line2D([0], [0], color="0.3", ls="None", marker="*", ms=12, label="*: stochastic MC")
    )
particle_handles1 = [
    Line2D([0], [0], color=SPECIES[p][2], lw=2.0, label=p.capitalize()) for p in SPECIES
]
leg_role1 = ax.legend(handles=role_handles, frameon=False, fontsize=8, loc="upper right")
ax.add_artist(leg_role1)
ax.legend(handles=particle_handles1, frameon=False, fontsize=9, loc="lower left")
fig1.tight_layout()
save_figure(fig1, "sk1997_fig1_alpha_vs_energy.png", directory=FIGURES, dpi=130)

# %% [markdown]
# ## Fig. 3 -- dose-domain beta vs energy
#
# * **two-track overlap** (thick) -- the deterministic reconstruction of the paper's
#   *exact* stochastic beta: the survival expanded to second order in fluence, i.e.
#   the pairwise non-additive overlap of two tracks' local doses. It recovers the
#   photon `beta_x` at low LET by construction. Evaluated on a coarser energy grid
#   with the reduced, still <1%-converged quadrature `TWO_TRACK_FAST_QUADRATURE`.
# * **z-domain approximation** (dashed) -- `beta_z * exp(-2 N_1)`
#   (`beta_model="scholz1997_stochastic"`), which mis-states the low-energy carbon
#   curvature.
# * **stochastic MC** (stars) -- the explicit-LEM Monte-Carlo ground truth with
#   jackknife errors; it tracks the two-track curve.

# %%
fig3b, ax = plt.subplots(figsize=(7.6, 5.4))
TWO_TRACK_ENERGIES = np.geomspace(2.0, 400.0, pick(16, 4))


def beta_curve(model: LEM1, particle: str, energies: np.ndarray) -> np.ndarray:
    return np.array([model.alpha_beta(scholz1997_ion(particle, float(e))).beta for e in energies])


def two_track_beta_curve(particle: str, energies: np.ndarray) -> np.ndarray:
    return np.array([explicit_lq(particle, float(e)).beta for e in energies])


@cache
def explicit_lq(particle: str, energy_MeV_u: float) -> LQParameters:
    """Cached explicit-model LQ coefficients used consistently across panels."""
    return LEM_STOCHASTIC.scholz1997_explicit_lq(
        scholz1997_ion(particle, energy_MeV_u), **TWO_TRACK_FAST_QUADRATURE
    )


for particle, (_z, _a, color) in SPECIES.items():
    ax.plot(
        TWO_TRACK_ENERGIES,
        two_track_beta_curve(particle, TWO_TRACK_ENERGIES),
        color=color,
        lw=2.4,
        label=f"{particle} pyRadBio (two-track exact)",
    )
    ax.plot(
        ENERGY_SWEEP,
        beta_curve(LEM_STOCHASTIC, particle, ENERGY_SWEEP),
        color=color,
        lw=1.5,
        ls="--",
        label=f"{particle} pyRadBio (z-domain approx.)",
    )
    for calc, marker in (("exact", "o"), ("approximation", "s")):
        for ref in iter_lem1_curves(
            figure="fig3_beta_vs_energy", particle=particle, calculation=calc
        ):
            ax.plot(ref.x, ref.y, color=color, ls="None", marker=marker, ms=6, mfc="white")
    if MC_RESULTS.get(particle):
        e_mc, b_mc, b_err = mc_arrays(particle, "beta")
        ax.errorbar(
            e_mc, b_mc, yerr=b_err, color=color, ls="None", marker="*", ms=12, capsize=3, zorder=6
        )

ax.set_xscale("log")
ax.set_xlim(0.8, 500)
ax.set_ylim(0.0, 0.032)
ax.set_xlabel("Energy (MeV/u)")
ax.set_ylabel(r"$\beta$ (Gy$^{-2}$)")
ax.set_title(f"Scholz 1997 Fig. 3: beta vs energy (CHO)\nstopping power: {STOPPING_POWER_LABEL}")
ax.grid(True, which="both", ls=":", lw=0.5)
ax.axhline(PHOTON.beta_x, color="0.6", lw=1.0, ls=":")
fig3_handles = [
    Line2D([0], [0], color="0.3", lw=2.4, label="two-track exact"),
    Line2D([0], [0], color="0.3", lw=1.5, ls="--", label="z-domain approx."),
    Line2D([0], [0], color="0.3", ls="None", marker="o", mfc="white", label="paper exact"),
    Line2D([0], [0], color="0.3", ls="None", marker="s", mfc="white", label="paper approx."),
    Line2D([0], [0], color="0.6", lw=1.0, ls=":", label=rf"photon $\beta_x$={PHOTON.beta_x:g}"),
]
if MC_RESULTS:
    fig3_handles.append(
        Line2D([0], [0], color="0.3", ls="None", marker="*", ms=12, label="stochastic MC")
    )
particle_handles3 = [
    Line2D([0], [0], color=SPECIES[p][2], lw=2.0, label=p.capitalize()) for p in SPECIES
]
leg_role3 = ax.legend(handles=fig3_handles, frameon=False, fontsize=8, loc="lower right")
ax.add_artist(leg_role3)
ax.legend(handles=particle_handles3, frameon=False, fontsize=9, loc="upper left")
fig3b.tight_layout()
save_figure(fig3b, "sk1997_fig3_beta_vs_energy.png", directory=FIGURES, dpi=130)

# %% [markdown]
# ## Fig. 4 -- specific-energy slope vs z
#
# The approximation panels draw the straight lines `alpha_z + 2 beta_z z`
# (`scholz1997_specific_energy_slope`) that all approach `s_max` near `z_t = D_t`:
# `alpha_z = N_1/Delta z` is the central-traversal slope (eq. 10) and
# `beta_z = (s_max - alpha_z)/(2 D_t)` slope matching (eq. 11).
#
# The exact panels use `LEM1.scholz1997_specific_energy_survival`. It needs a solver
# of its own because `z` is a per-cell stochastic quantity: `S(z)` is neither the
# dose-domain `S(D)` (which averages over the specific energies one dose produces)
# nor recoverable from `scholz1997_traversal_conditioned_survival` (which conditions
# on traversal *count*, and `z` at fixed count still varies with the impact
# parameters). Each realization gives `z = sum_i d_bar(b_i)` and `exp(-N_leth)` from
# one track configuration, and binning by `z` gives `S(z)`.
#
# At carbon 1 MeV/u exact and approximate agree within ~2% (`beta_z = 0`, both
# flat), but by 8 MeV/u the slope-matched `beta_z` makes the approximation rise far
# too fast: the exact slope is 0.97 of it at z = 1.5 Gy and 0.85 by z = 23 Gy. The
# approximation is not conservative in the same direction everywhere.
#
# Two caveats when reading the left panels against the paper:
#
# * **The paper's `z` is not ours.** Its exact panels approximate the increments of
#   specific energy "in terms of LET", so their `z` is a deterministic multiple of
#   the traversal count; ours is the real `sum_i d_bar(b_i)`, halo included. On the
#   same realizations the two agree in the mean to 0.1% and their slope curves
#   differ by a few percent, so this is not what separates the curves -- nor why the
#   paper's lines are straight: ours are near-straight too (R^2 = 0.95-0.99 wherever
#   `beta_z > 0`), a property of the explicit model.
# * **The paper reference is a two-point line** (linear to R^2 >= 0.998), so its
#   low-`z` end is an extrapolation of a possibly curved original and the least
#   trustworthy part of the comparison. That is where ours part company most: ~5-15%
#   below the paper's line, worst below z ~ 5 Gy, within a few percent by mid-range.
#
# The exact curves are precomputed by `scripts/lem1_mc_reference.py z-slope` into
# `validation/data/sk1997_z_slope.csv` (the cost scales with the traversals needed
# to reach `z_max` over the whole `b_max` disc, 23 for carbon at 1 MeV/u but 2.3e8
# for protons at 200; about 7 min for all eight points). Its rows predate the
# fingerprint header and keep the `LEM1` default stopping power of their generation.

# %%
Z_SWEEP = np.linspace(0.0, 30.0, 31)
ENERGY_COLORS = {1.0: "C0", 8.0: "C1", 50.0: "C2", 200.0: "C3", 400.0: "C4"}


def slope_vs_z(model: LEM1, particle: str, energy: float, zs: np.ndarray) -> np.ndarray:
    ion = scholz1997_ion(particle, energy)
    return np.array([model.scholz1997_specific_energy_slope(ion, float(z)) for z in zs])


Z_SLOPE = load_z_slope_reference(DATA / "sk1997_z_slope.csv")
if not Z_SLOPE:
    print("  (no validation/data/sk1997_z_slope.csv; run scripts/lem1_mc_reference.py z-slope)")

fig4b, axes4 = plt.subplots(2, 2, figsize=(11, 7.6), sharex=True, sharey=True)
fig4_energies: set[float] = set()
for row, particle in enumerate(SPECIES):
    energies = sorted(
        {
            ref.energy_MeV_u
            for ref in iter_lem1_curves(figure="fig4_slope_vs_specific_energy", particle=particle)
            if ref.energy_MeV_u is not None
        }
    )
    fig4_energies.update(energies)
    for col, calculation in enumerate(("exact", "approximation")):
        ax4 = axes4[row, col]
        for energy in energies:
            color = ENERGY_COLORS.get(energy, "0.4")
            if calculation == "approximation":
                ax4.plot(
                    Z_SWEEP,
                    slope_vs_z(LEM_STOCHASTIC, particle, energy, Z_SWEEP),
                    color=color,
                    lw=2.0,
                )
            elif (particle, energy) in Z_SLOPE:
                _z, _slope = Z_SLOPE[(particle, energy)]
                ax4.plot(_z, _slope, color=color, lw=2.0)
            for ref in iter_lem1_curves(
                figure="fig4_slope_vs_specific_energy",
                particle=particle,
                calculation=calculation,
                energy_MeV_u=energy,
            ):
                style = "--" if calculation == "exact" else ":"
                ax4.plot(ref.x, ref.y, color=color, lw=1.1, ls=style, alpha=0.8)
        path_title = "exact (z-conditioned MC)" if calculation == "exact" else "approximation"
        ax4.set_title(f"{particle.capitalize()}s — {path_title}")
        ax4.set_xlim(0, 30)
        ax4.set_ylim(0, 2.15)
        ax4.grid(True, ls=":", lw=0.5)
        ax4.axhline(PHOTON.max_slope(), color="0.6", lw=1.0, ls=":")
        if row == 1:
            ax4.set_xlabel("Specific energy z (Gy)")
    axes4[row, 0].set_ylabel(r"Slope $\alpha_z + 2\beta_z z$ (Gy$^{-1}$)")

energy_handles4 = [
    Line2D([0], [0], color=ENERGY_COLORS.get(e, "0.4"), lw=2.0, label=f"{e:g} MeV/u")
    for e in sorted(fig4_energies)
]
role4 = [
    Line2D([0], [0], color="0.3", lw=2.0, label="solid: pyRadBio (left exact MC, right approx.)"),
    Line2D([0], [0], color="0.3", lw=1.1, ls="--", label="dashed: paper exact"),
    Line2D([0], [0], color="0.3", lw=1.1, ls=":", label="dotted: paper approx."),
    Line2D([0], [0], color="0.6", lw=1.0, ls=":", label=rf"$s_{{\max}}$={PHOTON.max_slope():g}"),
]
fig4b.legend(
    handles=energy_handles4,
    loc="upper center",
    ncol=len(energy_handles4),
    frameon=False,
    bbox_to_anchor=(0.5, 0.93),
)
fig4b.legend(
    handles=role4, loc="upper center", ncol=len(role4), frameon=False, bbox_to_anchor=(0.5, 0.875)
)
fig4b.suptitle(
    f"Scholz 1997 Fig. 4: specific-energy slope (CHO); stopping power: {STOPPING_POWER_LABEL}",
    y=0.995,
)
fig4b.tight_layout(rect=(0, 0, 1, 0.82))
save_figure(fig4b, "sk1997_fig4_slope_vs_specific_energy.png", directory=FIGURES, dpi=130)

# %% [markdown]
# ## Figs. 2 & 5 -- Douglas-Fowler slope and survival vs dose


# %%
def approximate_lns_over_dose(model: LEM1, ion: Ion, doses: np.ndarray) -> np.ndarray:
    log_kills = model.scholz1997_approximate_log_kills(ion, doses)
    alpha0 = model.scholz1997_central_initial_slope(ion)
    return np.array(
        [alpha0 if d == 0.0 else lk / float(d) for d, lk in zip(doses, log_kills, strict=True)]
    )


def approximate_survival(model: LEM1, ion: Ion, doses: np.ndarray) -> np.ndarray:
    return np.exp(-np.array(model.scholz1997_approximate_log_kills(ion, doses)))


def explicit_lns_over_dose(particle: str, energy: float, doses: np.ndarray) -> np.ndarray:
    lq = explicit_lq(particle, energy)
    return lq.alpha + lq.beta * doses


def explicit_survival(particle: str, energy: float, doses: np.ndarray) -> np.ndarray:
    lq = explicit_lq(particle, energy)
    return np.exp(-(lq.alpha * doses + lq.beta * doses**2))


DOSE_LNS = np.linspace(0.0, 30.0, 61)
DOSE_SURV = np.linspace(0.0, 10.0, 81)

fig25, axes25 = plt.subplots(2, 2, figsize=(11, 8.4))
for col, particle in enumerate(SPECIES):
    ax_lns, ax_surv = axes25[0, col], axes25[1, col]
    energies = sorted(
        {
            ref.energy_MeV_u
            for ref in iter_lem1_curves(figure="fig5_survival_vs_dose", particle=particle)
            if ref.energy_MeV_u is not None
        }
    )
    for energy in energies:
        color = ENERGY_COLORS.get(energy, "0.4")
        ion = scholz1997_ion(particle, energy)
        ax_lns.plot(
            DOSE_LNS, explicit_lns_over_dose(particle, energy, DOSE_LNS), color=color, lw=2.2
        )
        ax_lns.plot(
            DOSE_LNS,
            approximate_lns_over_dose(LEM_STOCHASTIC, ion, DOSE_LNS),
            color=color,
            lw=1.5,
            ls="-.",
        )
        for ref in iter_lem1_curves(
            figure="fig2_exact_ln_survival_over_dose",
            particle=particle,
            calculation="exact",
            energy_MeV_u=energy,
        ):
            ax_lns.plot(ref.x, ref.y, color=color, lw=1.1, ls="--", alpha=0.7)
        ax_surv.plot(DOSE_SURV, explicit_survival(particle, energy, DOSE_SURV), color=color, lw=2.2)
        ax_surv.plot(
            DOSE_SURV,
            approximate_survival(LEM_STOCHASTIC, ion, DOSE_SURV),
            color=color,
            lw=1.5,
            ls="-.",
        )
        for calc, style in (("exact", "--"), ("approximation", ":")):
            for ref in iter_lem1_curves(
                figure="fig5_survival_vs_dose",
                particle=particle,
                calculation=calc,
                energy_MeV_u=energy,
            ):
                ax_surv.plot(ref.x, ref.y, color=color, lw=1.1, ls=style, alpha=0.7)
    ax_lns.set_title(f"{particle.capitalize()}s: Fig. 2 -ln(S)/D")
    ax_lns.set_xlim(0, 30)
    ax_lns.set_ylim(0, 2.0)
    ax_lns.set_ylabel(r"$-\ln(S)/D$ (Gy$^{-1}$)")
    ax_lns.set_xlabel("Dose (Gy)")
    ax_lns.grid(True, ls=":", lw=0.5)
    ax_surv.set_title(f"{particle.capitalize()}s: Fig. 5 survival")
    ax_surv.set_xlim(0, 10)
    ax_surv.set_ylim(0.01, 1.0)
    ax_surv.set_yscale("log")
    ax_surv.set_ylabel("Survival")
    ax_surv.set_xlabel("Dose (Gy)")
    ax_surv.grid(True, which="both", ls=":", lw=0.5)

energy_handles = [
    Line2D([0], [0], color=c, lw=2.0, label=f"{e:g} MeV/u") for e, c in ENERGY_COLORS.items()
]
role25 = [
    Line2D([0], [0], color="0.3", lw=2.2, label="solid: pyRadBio explicit LQ"),
    Line2D([0], [0], color="0.3", lw=1.5, ls="-.", label="dash-dot: pyRadBio approx."),
    Line2D([0], [0], color="0.3", lw=1.1, ls="--", label="dashed: paper exact"),
    Line2D([0], [0], color="0.3", lw=1.1, ls=":", label="dotted: paper approx."),
]
fig25.legend(
    handles=energy_handles, loc="upper center", ncol=5, frameon=False, bbox_to_anchor=(0.5, 0.96)
)
fig25.legend(handles=role25, loc="upper center", ncol=4, frameon=False, bbox_to_anchor=(0.5, 0.935))
fig25.suptitle(f"Scholz 1997 Figs. 2 & 5 (CHO); stopping power: {STOPPING_POWER_LABEL}", y=0.995)
fig25.tight_layout(rect=(0, 0, 1, 0.91))
save_figure(fig25, "sk1997_fig2_fig5_dose.png", directory=FIGURES, dpi=130)

# %% [markdown]
# ## Quantitative summary and shape investigation
#
# The two-track exact beta targets the paper's *exact* Fig. 3 curve; the z-domain
# approximation is closer to the paper's *approximation* curve. Fig. 5 compares each
# pyRadBio path with the matching paper path in `ln S`. The last block checks the
# cross-panel identity: every Fig. 2 zero-dose intercept must equal the matching
# Fig. 1 alpha (full impact parameter for the explicit path, central traversal for
# the approximation).

# %%
print(f"\n=== Scholz 1997 reconstruction summary ({STOPPING_POWER_LABEL}) ===")

print("\nFig. 1 alpha(E):")
for particle, (_z, _a, _c) in SPECIES.items():
    fine = np.geomspace(1.0, 400.0, pick(400, 60))
    a_model = alpha_exact(LEM_STOCHASTIC, particle, fine)
    e_peak = float(fine[int(np.argmax(a_model))])
    print(
        f"  {particle}: pyRadBio alpha peaks at E={e_peak:.1f} MeV/u, alpha_max={a_model.max():.3f}"
    )
    for calc in ("exact", "approximation"):
        ref = iter_lem1_curves(figure="fig1_alpha_vs_energy", particle=particle, calculation=calc)
        if not ref:
            continue
        model_fn = alpha_central if calc == "approximation" else alpha_exact
        stats = compare(ref[0], lambda x, p=particle, f=model_fn: f(LEM_STOCHASTIC, p, x))
        print_compare(f"vs paper {calc}", stats)

print("\nFig. 3 beta(E):")
for particle in SPECIES:
    exact_ref = iter_lem1_curves(
        figure="fig3_beta_vs_energy", particle=particle, calculation="exact"
    )
    if exact_ref:
        stats = compare(exact_ref[0], lambda x, p=particle: two_track_beta_curve(p, x))
        print_compare(f"{particle} two-track vs paper exact", stats)
    for calc in ("exact", "approximation"):
        ref = iter_lem1_curves(figure="fig3_beta_vs_energy", particle=particle, calculation=calc)
        if not ref:
            continue
        stats = compare(ref[0], lambda x, p=particle: beta_curve(LEM_STOCHASTIC, p, x))
        print_compare(f"{particle} z-domain vs paper {calc}", stats)

print("\nFig. 5 survival [matching pyRadBio path vs paper path], compared in ln S:")
for particle in SPECIES:
    for calc in ("exact", "approximation"):
        refs = iter_lem1_curves(figure="fig5_survival_vs_dose", particle=particle, calculation=calc)
        for ref in refs:
            energy = ref.energy_MeV_u or 0.0
            ion = scholz1997_ion(particle, energy)

            if calc == "exact":
                lq = explicit_lq(particle, energy)

                def model_lnS(x: np.ndarray, lq: LQParameters = lq) -> np.ndarray:
                    return -(lq.alpha * x + lq.beta * x**2)

            else:

                def model_lnS(x: np.ndarray, ion: Ion = ion) -> np.ndarray:
                    return -np.array(LEM_STOCHASTIC.scholz1997_approximate_log_kills(ion, x))

            paper_lnS = Scholz1997Curve(
                figure=ref.figure,
                particle=ref.particle,
                calculation=ref.calculation,
                energy_MeV_u=ref.energy_MeV_u,
                x_quantity=ref.x_quantity,
                x_unit=ref.x_unit,
                y_quantity="ln_survival",
                y_unit="",
                x=ref.x,
                y=tuple(math.log(max(v, 1e-6)) for v in ref.y),
                source_page=ref.source_page,
                note=ref.note,
            )
            stats = compare(paper_lnS, model_lnS)
            print_compare(f"{particle} {ref.energy_MeV_u:g} MeV/u vs {calc}", stats)

print("\nCross-panel consistency (Fig. 2 intercept == matching Fig. 1 alpha):")
for particle in SPECIES:
    for energy in (1.0, 50.0):
        ion = scholz1997_ion(particle, energy)
        approximate_intercept = float(
            approximate_lns_over_dose(LEM_STOCHASTIC, ion, np.array([0.0]))[0]
        )
        central = LEM_STOCHASTIC.scholz1997_central_initial_slope(ion)
        explicit_intercept = float(explicit_lns_over_dose(particle, energy, np.array([0.0]))[0])
        full = LEM_STOCHASTIC.initial_slope(ion)
        print(
            f"  {particle} {energy:g} MeV/u: explicit={explicit_intercept:.4f}/"
            f"{full:.4f} match={math.isclose(explicit_intercept, full, rel_tol=1e-9)}; "
            f"approx={approximate_intercept:.4f}/{central:.4f} "
            f"match={math.isclose(approximate_intercept, central, rel_tol=1e-9)}"
        )

# The tally the stopping-power section promises: how often the 1970 table answered
# and at which energies analytic Bethe took over.
if isinstance(STOPPING_POWER, NorthcliffeSchillingWithBetheTail):
    _tail = sorted(STOPPING_POWER.fallback_energies)
    print(
        f"\nStopping-power seam: {STOPPING_POWER.table_hits} LET lookups inside the "
        f"{ENERGY_RANGE_MEV_U[0]:g}-{ENERGY_RANGE_MEV_U[1]:g} MeV/u table; analytic Bethe "
        f"at {len(_tail)} distinct energies outside it"
        + (f" ({_tail[0]:g}-{_tail[-1]:g} MeV/u)" if _tail else "")
    )

print("\nDone. Figures written to", FIGURES)
