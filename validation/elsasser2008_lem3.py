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
# # LEM III: the velocity-dependent track core
#
# Elsasser, Kraemer & Scholz 2008 (doi:10.1016/j.ijrobp.2008.02.037) extend
# LEM II with a single change: the inner (high-dose) core of the amorphous track
# grows with the projectile velocity,
#
# $$ r_\mathrm{min}(\beta) = \beta\, r_c, \qquad \beta = v/c, \qquad r_c = 40\ \mathrm{nm} $$
#
# (eq. 6, fit to rat spinal-cord tolerance). A wider core spreads the same LET
# over a larger area, so the local-dose peak — and the ion RBE — drops fastest
# for the *fast, low-LET* ions of the entrance channel, exactly where LEM I and
# LEM II over-predicted.
#
# This script reproduces the paper's central in-vitro result (Tables 1-2):
# RBE$_\alpha$ for the 15-cell-line panel of Suzuki et al. 2000 at the entrance
# channel (13 keV/µm) and the Bragg-peak region (77 keV/µm), and how the
# systematic deviation from experiment shrinks from LEM I to LEM II to LEM III.
# The panel (photon LQ, experimental RBE$_\alpha$, per-version $D_t$, each line's
# Suzuki 2000 nuclear area, the paper's own $\Delta$RBE$_\alpha$ and "Average" rows)
# and the reconstruction machinery live in `pyradbio.benchmark.elsasser2008`.
#
# Quick mode (`PYRADBIO_QUICK=1`) scores every third panel line and a 5-point
# LET curve; only the full run's numbers are the reference.

# %%
from __future__ import annotations

import matplotlib.pyplot as plt
import numpy as np
from _common import pick, quick_mode

from pyradbio import CellTarget
from pyradbio.benchmark import elsasser2008 as es
from pyradbio.benchmark.sweeps import sweep_energies
from pyradbio.models.lem.lem2 import ClusterModel
from pyradbio.plotting import figures_dir, save_figure
from pyradbio.track.stopping_power import default_stopping_power, describe_stopping_power

FIGURES = figures_dir(globals().get("__file__"))
if quick_mode():
    print("QUICK MODE (PYRADBIO_QUICK=1): 5 of 15 panel lines, 5-point LET curve")

PANEL = pick(es.SUZUKI_PANEL, es.SUZUKI_PANEL[::3])
CURVE_ENERGIES = pick(
    [300.0, 200.0, 120.0, 70.0, 40.0, 25.0, 15.0, 9.0, 5.0, 3.0, 1.8],
    [300.0, 70.0, 25.0, 9.0, 3.0],
)
MODELS = es.MODEL_NAMES  # ("LEM I", "LEM II", "LEM III")
PAPER_MEAN_DELTA = {name: es.PAPER_AVERAGE_DELTA[v] for v, name in enumerate(MODELS)}

# %% [markdown]
# ## The two carbon qualities
#
# The paper degrades a 290 MeV/u carbon beam to 13 and 77 keV/µm; in water those
# LETs are carbon at ~285 and ~25 MeV/u (`es.channel_ions`).

# %%
sp = default_stopping_power()
ION_13, ION_77 = es.channel_ions(sp)
print(f"stopping power: {describe_stopping_power(sp)}")
print(f"13 keV/um -> {ION_13.energy_MeV_u:.0f} MeV/u (beta={ION_13.beta():.2f})")
print(f"77 keV/um -> {ION_77.energy_MeV_u:.0f} MeV/u (beta={ION_77.beta():.2f})")

# %% [markdown]
# ## Predicted RBE$_\alpha$ for the panel
#
# Each line gets its own nuclear radius from Suzuki 2000's nuclear area
# ($\sqrt{A/\pi}$, x0.8 to the effective target: `SuzukiCell.cell_target()`). That
# per-line size is essential for the 77 keV/µm channel; the 13 keV/µm channel is
# nearly radius-independent. One `ClusterModel` (its $\eta(D)$ table depends only
# on the default cluster parameters) is shared by every LEM II and LEM III model.
# KS-1 (the panel's highest $\alpha_x$) is the one 77 keV/µm outlier: our LEM
# overkills it, so the per-cell RMS is reported with and without it.

# %%
SHARED_CLUSTER = ClusterModel()
IONS = (ION_13, ION_77)
predicted = {
    name: es.predict_panel(name, IONS, stopping_power=sp, cluster=SHARED_CLUSTER, panel=PANEL)
    for name in MODELS
}

print("\n            mean dRBE_alpha   (paper)")
for name in MODELS:
    d13, d77 = es.mean_delta(predicted[name])
    p13, p77 = PAPER_MEAN_DELTA[name]
    print(f"  {name:8s} 13 keV/um {d13:+.2f} ({p13:+.2f})   77 keV/um {d77:+.2f} ({p77:+.2f})")

print("\n  per-cell dRBE_alpha RMS vs paper Tables 1-2:")
print("            13 keV/um   77 keV/um (excl. KS-1)")
for name in MODELS:
    rms13 = es.per_cell_rms(predicted[name], name, 0, panel=PANEL)
    rms77 = es.per_cell_rms(predicted[name], name, 1, panel=PANEL)
    rms77x = es.per_cell_rms(predicted[name], name, 1, exclude=es.OUTLIER_CELLS_77, panel=PANEL)
    print(f"    {name:8s} {rms13:.3f}       {rms77:.3f}  ({rms77x:.3f})")

# %% [markdown]
# ## RBE$_\alpha$ vs LET for the in-vivo inputs
#
# The rat spinal-cord parameters of the paper's in-vivo comparison
# ($\alpha_x=0.1$, $\beta_x=0.05$, $R_n=5$ µm, $D_t$ from the Fig. 1 caption; the
# radius is already an effective size, so it is used as-is). The velocity core
# pulls the low-LET (entrance-channel) branch down the hardest.

# %%
RSC = es.RAT_SPINAL_CORD
INVIVO_CELL = CellTarget(nucleus_radius_um=RSC.nucleus_radius_um)
curves = {}
for v, name in enumerate(MODELS):
    model = es.build_panel_model(
        name,
        RSC.alpha_x,
        RSC.beta_x,
        RSC.D_t[v],
        INVIVO_CELL,
        stopping_power=sp,
        cluster=SHARED_CLUSTER,
    )
    sweep = sweep_energies(
        model, 6, 12, CURVE_ENERGIES, stopping_power=sp, on_error="raise", sort_by_let=True
    )
    curves[name] = (sweep.let_keV_um, sweep.rbe_alpha(RSC.alpha_x))

# %% [markdown]
# ## Figure
#
# Crosses are the paper's own predictions, RBE$_\alpha$,exp x (1 + $\Delta$RBE$_\alpha$)
# (`SuzukiCell.rbe_alpha_paper`): they show whether the reconstruction reproduces
# not just experiment but the paper's *deviation pattern* off the diagonal.

# %%
fig, axes = plt.subplots(2, 2, figsize=(11.5, 9.5))
ax_13, ax_77, ax_bar, ax_let = axes.ravel()
colors = {"LEM I": "0.55", "LEM II": "C1", "LEM III": "C0"}
markers = {"LEM I": "s", "LEM II": "^", "LEM III": "o"}
exp_rbe = np.array([c.rbe_alpha_exp for c in PANEL])  # (n, 2): (13, 77)

for ax, col, title in (
    (ax_13, 0, "13 keV/µm (entrance channel)"),
    (ax_77, 1, "77 keV/µm (Bragg peak)"),
):
    hi = 0.0
    for name in MODELS:
        rows = np.array(predicted[name])
        exp, pred = rows[:, col + 2], rows[:, col]
        hi = max(hi, exp.max(), pred.max())
        ax.plot(
            exp, pred, markers[name], mfc="none", mec=colors[name], mew=1.2, ls="none", label=name
        )
    for v, name in enumerate(MODELS):
        paper_pred = np.array([c.rbe_alpha_paper(v)[col] for c in PANEL])
        hi = max(hi, float(paper_pred.max()))
        ax.plot(
            exp_rbe[:, col],
            paper_pred,
            "x",
            color=colors[name],
            mew=1.4,
            ms=6,
            ls="none",
            label=f"{name} (paper)",
        )
    lim = [0.9, hi * 1.08]
    ax.plot(lim, lim, "k:", lw=1.0, label="ideal (pred = exp)")
    ax.set_xlim(lim)
    ax.set_ylim(lim)
    ax.set_xlabel(r"experimental RBE$_\alpha$")
    ax.set_ylabel(r"predicted RBE$_\alpha$")
    ax.set_title(title)
    ax.legend(frameon=False, fontsize=7, ncol=2)
    ax.grid(True, ls=":", lw=0.45)

x = np.arange(len(MODELS))
width = 0.35
deltas = [es.mean_delta(predicted[n]) for n in MODELS]
ax_bar.bar(x - width / 2, [d[0] for d in deltas], width, color="C0", label="pyRadBio 13 keV/µm")
ax_bar.bar(x + width / 2, [d[1] for d in deltas], width, color="C1", label="pyRadBio 77 keV/µm")
paper_d13 = [PAPER_MEAN_DELTA[n][0] for n in MODELS]
paper_d77 = [PAPER_MEAN_DELTA[n][1] for n in MODELS]
ax_bar.plot(x - width / 2, paper_d13, "k_", ms=16, mew=2.0, label="paper (Tables 1-2)")
ax_bar.plot(x + width / 2, paper_d77, "k_", ms=16, mew=2.0)
ax_bar.axhline(0.0, color="k", lw=0.8)
ax_bar.set_xticks(x)
ax_bar.set_xticklabels(MODELS)
ax_bar.set_ylabel(r"signed mean $\Delta$RBE$_\alpha$")
ax_bar.set_title("Systematic deviation shrinks I -> II -> III")
ax_bar.legend(frameon=False, fontsize=8)
ax_bar.grid(True, axis="y", ls=":", lw=0.45)

for name in MODELS:
    let, rbe = curves[name]
    ax_let.plot(let, rbe, color=colors[name], lw=2.0, marker=markers[name], ms=4, label=name)
for let_mark, tag in ((13.0, "entrance"), (77.0, "Bragg peak")):
    ax_let.axvline(let_mark, color="0.7", ls="--", lw=0.9)
    ax_let.text(let_mark, 0.6, f" {tag}", rotation=90, va="bottom", fontsize=7, color="0.4")
ax_let.set_xscale("log")
ax_let.set_xlabel("LET (keV/µm)")
ax_let.set_ylabel(r"RBE$_\alpha$")
ax_let.set_title(r"RBE$_\alpha$(LET), rat spinal-cord inputs")
ax_let.legend(frameon=False, fontsize=8)
ax_let.grid(True, which="both", ls=":", lw=0.45)

fig.suptitle("Elsasser, Kraemer & Scholz 2008 — LEM III (velocity-dependent track core)", y=0.99)
fig.tight_layout()
print()
save_figure(fig, "lem3_velocity_dependent_track_core.png", directory=FIGURES, dpi=150)
