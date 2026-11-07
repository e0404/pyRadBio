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
# # LEM I vs TRiP98: initial-slope RBE reproduction
#
# GSI's treatment-planning code TRiP98 ships LEM I reference tables (`*.rbe`)
# giving `RBE_alpha = alpha_ion / alpha_x` versus energy per nucleon for the
# standard ion set (H..Ne), under a photon `(alpha_x, beta_x, D_t, R_nucleus)`
# set fixed by each file's header. This script drives pyRadBio's **deterministic**
# LEM I initial-slope paths (no Monte-Carlo) against one such table and overlays
# the curves per ion.
#
# The point of difference from the `tests/benchmark/test_reference_tables.py`
# goldens is the **stopping power**: the TRiP folder ships the exact dE/dx it used
# -- an old (2000) FLUKA H2O table, `I = 77.3 eV`, under `DEDX/`
# (`TripDedxTable.stopping_power()`). Feeding *that* table into pyRadBio removes
# the stopping-power variable, so any residual is the LEM I biophysics (track
# structure + initial-slope estimator), not an LET mismatch. The FLUKA table only
# tabulates light ions (Z<=6), so the faithful comparison runs protons through
# carbon.
#
# Three deterministic pyRadBio paths are shown:
#
# * **cross-section** (`initial_slope`, default) -- `alpha = sigma/(A_eff D_1)`
#   from the amorphous inactivation cross-section. Includes indirect traversals;
#   *thins down* (overkill) at high LET rather than saturating.
# * **lethal_central** (`initial_slope_model="lethal_central"`) -- the Kraemer &
#   Scholz 2006 rapidLEM formula `alpha = min(N_1/D_1, s_max)`; central traversal
#   only, saturates at the photon `s_max` (the low-energy `RBE_alpha` plateau).
# * **lethal_full** (`initial_slope_model="lethal_full"`) -- the impact-parameter
#   integral of the *unsaturated* lethal events `nu(b)` (vs the inactivation
#   probability `1-exp(-nu(b))` of the cross-section, i.e. without the overkill
#   term). It keeps the cross-section's indirect traversals *and* saturates at
#   `s_max`, so it tracks TRiP across the whole range and every tissue --
#   reconciling the `lethal_central` (low-E) / cross-section (high-E) crossover
#   that shows up e.g. in `normal10.rbe`. It reproduces TRiP because rapidLEM makes
#   the same overkill-free simplification; it is NOT the exact LEM slope
#   (`cross_section` keeps the overkill term and matches the explicit model / MC).
#
# The data is external (`PYRADBIO_TRIP98_DIR`; see `pyradbio.datasets`); the
# script exits cleanly with a message if it is not configured. Quick mode
# (`PYRADBIO_QUICK=1`) scores protons and carbon only.

# %%
from __future__ import annotations

import math
import os
from collections.abc import Callable
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
from _common import pick, quick_mode, require_data
from matplotlib.lines import Line2D

from pyradbio import Ion
from pyradbio.benchmark import reference_tables as rt
from pyradbio.datasets import TRIP98
from pyradbio.models.lem.lem1 import LEM1
from pyradbio.plotting import figures_dir, save_figure

FIGURES = figures_dir(globals().get("__file__"))

# Which TRiP LEM I table (its header sets alpha_x/beta_x/D_t/R_nucleus) and which
# FLUKA dE/dx file to use. normal02.rbe = alpha_x 0.1, beta_x 0.05, D_t 30, R_n 5.
RBE_FILE = os.environ.get("TRIP98_RBE_FILE", "normal10.rbe")
DEDX_FILE = os.environ.get("TRIP98_DEDX_FILE", "dEdxFLUKAxTRiP.dedx")

ELEMENT = {1: "H", 2: "He", 3: "Li", 4: "Be", 5: "B", 6: "C"}
# Cost knob: quick mode scores the lightest and the heaviest ion only.
ION_Z = pick(tuple(ELEMENT), (1, 6))
if quick_mode():
    print(f"[quick mode] ions Z={ION_Z} only; numbers are not for quoting")

# %%
rbe_table = require_data(rt.load_trip_lem1, RBE_FILE, dataset=TRIP98)
dedx_table = require_data(rt.load_trip_dedx, DEDX_FILE, dataset=TRIP98)

stopping_power = dedx_table.stopping_power()
print(
    f"RBE table : {RBE_FILE}  alpha_x={rbe_table.alpha_x} beta_x={rbe_table.beta_x} "
    f"D_t={rbe_table.D_t} R_n={rbe_table.r_nucleus_um}"
)
print(
    f"dE/dx     : {DEDX_FILE}  material={dedx_table.material} rho={dedx_table.density_g_cm3} "
    f"(ions Z<=6)"
)

# Ions present in BOTH the RBE table and the FLUKA dE/dx table (protons->carbon).
IONS = [(Z, A) for (Z, A) in sorted(rbe_table.curves) if stopping_power.covers(Z, A) and Z in ION_Z]
print("ions:", ", ".join(f"{ELEMENT[Z]}-{A}" for Z, A in IONS))

# %% [markdown]
# ## Model construction and the three deterministic RBE_alpha paths

# %%
photon = rbe_table.photon_parameters()
cell = rbe_table.cell_target()
lem = LEM1(photon=photon, cell=cell, stopping_power=stopping_power)
lem_full = LEM1(
    photon=photon, cell=cell, stopping_power=stopping_power, initial_slope_model="lethal_full"
)

S_MAX_RBE = photon.max_slope() / photon.alpha_x  # low-energy plateau RBE_alpha


def rbe_alpha(alpha_fn: Callable[[Ion], float], Z: int, A: int, energies: np.ndarray) -> np.ndarray:
    """RBE_alpha = alpha_ion/alpha_x of ion ``(Z, A)`` from ``alpha_fn(ion)``, NaN where undefined.

    Below ~0.4 MeV/u the LEM I track model has no delta-ray range (r_max<r_min)
    and ``alpha_fn`` raises; those points are returned as NaN so the curve simply
    starts where the model is defined.
    """
    out = np.full(energies.shape, np.nan)
    for i, e in enumerate(energies):
        try:
            out[i] = alpha_fn(Ion(Z, A, float(e))) / photon.alpha_x
        except (ValueError, ZeroDivisionError):
            pass
    return out


PATHS: dict[str, Callable[[Ion], float]] = {
    "cross-section": lem.initial_slope,
    "lethal_central": lem.lethal_central_initial_slope,
    "lethal_full": lem_full.initial_slope,
}
# Each path is evaluated once per ion; the figure and the agreement table share it.
PRED = {
    (Z, A): {
        name: rbe_alpha(fn, Z, A, np.array(rbe_table.curve(Z, A).energies_MeV_u))
        for name, fn in PATHS.items()
    }
    for Z, A in IONS
}

# %% [markdown]
# ## Overlay: pyRadBio deterministic paths vs TRiP98, per ion

# %%
STYLE = {
    "cross-section": {"color": "C0", "lw": 1.6},
    "lethal_central": {"color": "C1", "lw": 1.6, "ls": "--"},
    "lethal_full": {"color": "C2", "lw": 2.4},
}
ncols = 3
nrows = math.ceil(len(IONS) / ncols)
fig, axes = plt.subplots(nrows, ncols, figsize=(4.2 * ncols, 3.4 * nrows), squeeze=False)
for ax, (Z, A) in zip(axes.flat, IONS, strict=False):
    curve = rbe_table.curve(Z, A)
    e = np.array(curve.energies_MeV_u)
    ax.plot(
        e, curve.rbe_alpha, ls="None", marker="o", ms=4, mfc="white", color="0.3", label="TRiP98"
    )
    for name, style in STYLE.items():
        ax.plot(e, PRED[(Z, A)][name], label=name, **style)
    ax.axhline(S_MAX_RBE, color="0.6", lw=0.8, ls=":")
    ax.set_xscale("log")
    ax.set_title(f"{ELEMENT[Z]}-{A}")
    ax.set_xlabel("Energy (MeV/u)")
    ax.grid(True, which="both", ls=":", lw=0.4)
axes.flat[0].set_ylabel(r"$RBE_\alpha = \alpha_{ion}/\alpha_x$")
for ax in axes.flat[len(IONS) :]:
    ax.set_visible(False)
handles = [
    Line2D([0], [0], color="0.3", ls="None", marker="o", mfc="white", label="TRiP98"),
    *(Line2D([0], [0], label=name, **style) for name, style in STYLE.items()),
    Line2D([0], [0], color="0.6", lw=0.8, ls=":", label=rf"$s_{{max}}/\alpha_x$={S_MAX_RBE:g}"),
]
fig.legend(handles=handles, loc="upper center", ncol=5, frameon=False, bbox_to_anchor=(0.5, 1.0))
fig.suptitle(
    f"LEM I initial-slope RBE vs TRiP98 ({RBE_FILE}); stopping power: TRiP FLUKA dE/dx", y=1.03
)
fig.tight_layout()
save_figure(
    fig,
    f"lem1_trip98_rbe_{Path(RBE_FILE).stem}.png",
    directory=FIGURES,
    dpi=130,
    bbox_inches="tight",
)

# %% [markdown]
# ## Quantitative agreement (median |relative error| in RBE_alpha)
#
# Reported over two energy windows: the full tabulated range where the LEM I
# track model is defined (`E >= 1 MeV/u`), and the therapeutic window
# (20-350 MeV/u). `lethal_full` should reproduce TRiP tightly *everywhere and for
# every tissue*; `lethal_central` matches at low energy / strongly-shouldered
# tissue but underestimates at high energy; the cross-section path is the mirror
# image (good at high energy, thins down through the low-energy plateau).

# %%
WINDOWS = {"full (>=1 MeV/u)": (1.0, 1.0e9), "therapeutic (20-350)": (20.0, 350.0)}
print(f"\n=== RBE_alpha agreement vs TRiP98 ({RBE_FILE}) ===")
for Z, A in IONS:
    curve = rbe_table.curve(Z, A)
    ref = np.array(curve.rbe_alpha)
    scores = {
        name: rt.windowed_median_errors(curve.energies_MeV_u, pred, ref, WINDOWS)
        for name, pred in PRED[(Z, A)].items()
    }
    print(f"\n{ELEMENT[Z]}-{A}:")
    for label in WINDOWS:
        for name, by_window in scores.items():
            score = by_window[label]
            if score is None:
                continue
            err, n = score
            print(f"  {label:22s} {name:14s} median|rel err|={err:.3%}  (n={n})")

print("\nDone. Figure written to", FIGURES)
