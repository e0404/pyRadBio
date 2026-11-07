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
# # Validating a model against PIDE (à la Pfuhl et al. 2022)
#
# Reproduces the core analyses of Pfuhl, Friedrich & Scholz, *"Comprehensive
# comparison of local effect model IV predictions with the particle irradiation
# data ensemble"* (Med. Phys. **49**(1):714-726, 2022,
# [doi:10.1002/mp.15343](https://doi.org/10.1002/mp.15343)) on pyRadBio's
# benchmark harness (`pyradbio.benchmark.harness.run_benchmark`). For every PIDE
# cell-survival experiment the paper (a) computes the **measured** RBE from the
# fitted photon/ion LQ parameters and (b) **predicts** it from the photon LQ
# parameters + ion species with LEM IV, then studies the deviation
# systematically.
#
# We focus on the **initial-slope RBE**, `RBE_alpha = alpha_ion / alpha_x`, which
# pyRadBio delivers directly from any model's `alpha_beta`, plus the `beta` ratio
# `beta_ion / beta_x`. The four panels mirror the paper:
#
# * **(a)** predicted-vs-measured `RBE_alpha` scatter for carbon, split at
#   `LET = 150 keV/um`, with a linear fit **through the origin** and its slope /
#   `R^2` against the 1:1 line (paper Fig. 2a). Slope > 1 means the model
#   *underestimates* the RBE.
# * **(b)** `RBE_alpha` vs LET for carbon, measured and predicted, with Blackman
#   **running averages** to guide the eye (paper Fig. 3 / 6b top).
# * **(c)** the relative model deviation `log10(RBE_LEM / RBE_PIDE)` vs LET, one
#   running average per ion: the `RBE_alpha` analogue of paper Fig. 4c (which
#   shows `RBE_10`), whose signature is an underestimation at low LET that grows
#   with atomic number.
# * **(d)** the `beta_ion / beta_x` ratio vs LET for carbon (paper Fig. 6b bottom).
#
# **Scope.** The harness evaluates H-Ne (`max_z = 10`, which keeps benchmark point sets
# comparable), so we reproduce the study for H, He, C
# and Ne; the paper additionally covers Ar and Fe. We apply the loose PIDE
# filters (`alpha_x, beta_x, alpha_ion > 0`, a known energy >= 0.5 MeV/u) rather
# than the paper's full down-to-survival-level selection, so point counts differ
# from the paper's 571 experiments (its Table 1). Each ion is thinned to
# `MAX_POINTS_PER_ION` points evenly spaced in LET (`limit_per_ion`).
#
# **Data.** PIDE is not redistributable and never bundled: set
# `PYRADBIO_PIDE_DIR` (or drop a file in `data/pide/`). Without it the script
# stops with a message.
#
# **Runtime.** `MODEL_NAME = "lem4"` matches the paper but costs ~0.2 s/point; the
# per-ion cap keeps it to about a minute. Switch to `"lem1"`/`"lem3"`/`"mkm"` (and
# raise the cap) to explore other models -- LEM I, for contrast, *over*-predicts
# at low LET. `PYRADBIO_QUICK=1` evaluates 5 points per ion.
#
# This file is a [jupytext](https://jupytext.readthedocs.io) percent script.

# %%
from __future__ import annotations

import matplotlib.pyplot as plt
import numpy as np
from _common import pick, quick_mode, require_data, require_extra

from pyradbio.benchmark.harness import ModelSpec, build_model_spec, run_benchmark
from pyradbio.benchmark.pide import find_pide_file, load_pide
from pyradbio.core.elements import z_to_symbol
from pyradbio.datasets import PIDE
from pyradbio.models.mkm.model import MKMModel
from pyradbio.plotting import figures_dir, save_figure

FIGURES = figures_dir(globals().get("__file__"))

# %%
# --- configuration ---------------------------------------------------------
MODEL_NAME = "lem4"  # the Pfuhl paper's model; try "lem1"/"lem3"/"mkm"
R_N_UM = 5.0  # nuclear radius; the LEM default used throughout the paper
ION_ZS = (1, 2, 6, 10)  # H, He, C, Ne -- the Z<=10 subset the harness evaluates
CARBON_Z = 6
LET_SPLIT = 150.0  # keV/um; the paper's overkill-region boundary (Fig. 2)
MAX_POINTS_PER_ION = pick(40, 5)  # per-ion cap, evenly spaced in LET (bounds runtime)
MIN_ENERGY_MEV_U = 0.5  # below this the track/stopping-power tables are unreliable
if quick_mode():
    print(f"QUICK MODE: {MAX_POINTS_PER_ION} points per ion (PYRADBIO_QUICK=1)")

# Pfuhl 2022 Table 2, RBE_alpha row: the average deviation RBE_LEM/RBE_PIDE of LEM IV
# per ion (H = 1H+2H, He = 3He+4He). Checked against the paper's open-access full
# text (Europe PMC PMC13655482, CC BY): Table 2, doi:10.1002/mp.15343. The page
# within pp. 714-726 is not recorded (checked in the JATS XML, which carries no
# page breaks).
PFUHL2022_TABLE2_RBE_ALPHA = {"H": 0.93, "He": 1.14, "C": 0.82, "Ne": 0.83}

# %% [markdown]
# ## Load PIDE and evaluate the model
#
# The `_fit` LQ columns are used (PIDE's own consistent refits, the same set the
# paper uses), via the loader's default `variant="fit_then_paper"`. The photon
# reference and the transition dose come from each experiment: the paper sets
# `D_t = 4 Gy + 1.1 alpha_x/beta_x` (its eq. 3; the same rule as Pfuhl et al. 2020,
# Radiat. Res. 193:130, doi:10.1667/RR15456.1, eq. 5). The harness builds one model
# per unique photon pair.

# %%
PIDE_PATH = require_data(find_pide_file, dataset=PIDE)
if PIDE_PATH.suffix.lower() == ".xlsx":
    require_extra("openpyxl", "tables")  # the workbook reader
entries = load_pide(PIDE_PATH)
print(f"Loaded {len(entries)} PIDE entries from {PIDE_PATH}")


def transition_dose(alpha_x: float, beta_x: float) -> float:
    """Pfuhl 2022 eq. (3): the empirical LQL threshold dose from the photon alpha/beta."""
    return 4.0 + 1.1 * alpha_x / beta_x


if MODEL_NAME == "mkm":  # photon-dependent here: alpha0/beta0 from the row (not the harness's)
    spec = ModelSpec(
        build=lambda ax, bx: MKMModel(
            domain_radius_um=0.44, nucleus_radius_um=R_N_UM, alpha0=ax, beta0=bx
        )
    )
else:
    spec = build_model_spec(MODEL_NAME, D_t=transition_dose, nucleus_radius_um=R_N_UM)

# The harness applies the rest of the loose filter (known energy >= MIN_ENERGY_MEV_U,
# alpha_x > 0, alpha_ion > 0); beta_x > 0 is needed by D_t and the beta ratio.
candidates = [e for e in entries if e.Z in ION_ZS and e.beta_x > 0.0]
result = run_benchmark(
    candidates,
    MODEL_NAME,
    spec=spec,
    min_energy_MeV_u=MIN_ENERGY_MEV_U,
    limit_per_ion=MAX_POINTS_PER_ION,
    stratify="let",
)
print("Selected per ion:", {z_to_symbol(z): n for z, n in result.attempted_by_ion})

# Per-point records: (Z, LET, measured RBE_alpha, predicted RBE_alpha,
#                     measured beta ratio, predicted beta ratio).
records = [
    (
        p.Z,
        p.let_keV_um,
        p.measured_rbe_alpha,
        p.predicted_rbe_alpha,
        p.measured_beta / p.beta_x,  # beta fields: filled by run_benchmark
        p.predicted_beta / p.beta_x,
    )
    for p in result.points
]
rec = np.array(records, dtype=float).reshape(-1, 6)  # (-1, 6) keeps shape when empty
print(f"Evaluated {len(rec)} points with model '{MODEL_NAME}'.")

# %% [markdown]
# ## Analysis helpers: fit-through-origin and Blackman running average
#
# The paper's running average (its Sec. 2.4) weights each point by a Blackman
# window one decade of LET wide *and* by its inverse squared RBE error; PIDE's LQ
# columns carry no errors, so here the window alone sets the weights.


# %%
def fit_through_origin(x: np.ndarray, y: np.ndarray) -> tuple[float, float]:
    """Least-squares slope of ``y = s*x`` (through origin) and its ``R^2``."""
    slope = float(np.sum(x * y) / np.sum(x * x))
    ss_res = float(np.sum((y - slope * x) ** 2))
    ss_tot = float(np.sum((y - y.mean()) ** 2))
    r2 = 1.0 - ss_res / ss_tot if ss_tot > 0 else float("nan")
    return slope, r2


def running_average(
    let: np.ndarray, y: np.ndarray, n_grid: int = 40, window_decades: float = 1.0
) -> tuple[np.ndarray, np.ndarray]:
    """Blackman-weighted running mean of ``y`` over a log-LET window (Pfuhl Sec. 2.4)."""
    logs = np.log10(let)
    grid = np.linspace(logs.min(), logs.max(), n_grid)
    out = np.full(n_grid, np.nan)
    half = window_decades / 2.0
    for i, g in enumerate(grid):
        d = np.abs(logs - g)
        m = d <= half
        if m.sum() >= 3:
            xx = (logs[m] - g) / half  # in [-1, 1]
            w = 0.42 + 0.5 * np.cos(np.pi * xx) + 0.08 * np.cos(2.0 * np.pi * xx)
            out[i] = float(np.sum(w * y[m]) / np.sum(w))
    return 10.0**grid, out


# %% [markdown]
# ## Figure: four panels mirroring Pfuhl Figs. 2, 3, 4c, 6b

# %%
fig, axes = plt.subplots(2, 2, figsize=(11, 9))
ax_scatter, ax_rbe_let, ax_dev, ax_beta = axes.ravel()

carbon = rec[rec[:, 0] == CARBON_Z]
SCATTER = {"s": 12, "alpha": 0.35, "edgecolors": "none"}

# (a) predicted vs measured RBE_alpha for carbon, split by LET.
if len(carbon):
    low = carbon[carbon[:, 1] <= LET_SPLIT]
    high = carbon[carbon[:, 1] > LET_SPLIT]
    lim = max(carbon[:, 2].max(), carbon[:, 3].max()) * 1.05
    ax_scatter.plot([0, lim], [0, lim], "k--", lw=1, label="1:1")
    for sub, colour, tag in (
        (low, "tab:blue", f"LET <= {LET_SPLIT:g}"),
        (high, "tab:red", f"LET > {LET_SPLIT:g}"),
    ):
        if len(sub) >= 2:
            ax_scatter.scatter(
                sub[:, 3], sub[:, 2], s=16, alpha=0.6, color=colour, edgecolors="none"
            )
            slope, r2 = fit_through_origin(sub[:, 3], sub[:, 2])
            xs = np.array([0.0, lim])
            ax_scatter.plot(
                xs, slope * xs, color=colour, lw=1.5, label=f"{tag}: slope={slope:.2f}, R²={r2:.2f}"
            )
    ax_scatter.set_xlim(0, lim)
    ax_scatter.set_ylim(0, lim)
ax_scatter.set_xlabel(f"predicted RBE$_\\alpha$ ({MODEL_NAME})")
ax_scatter.set_ylabel("measured RBE$_\\alpha$ (PIDE)")
ax_scatter.set_title("(a) carbon: predicted vs measured RBE$_\\alpha$")
ax_scatter.legend(fontsize=7, loc="upper left")


def measured_vs_predicted(ax, meas_col: int, pred_col: int) -> None:
    """Carbon points (measured blue, predicted red) and their running averages."""
    if not len(carbon):
        return
    ax.scatter(carbon[:, 1], carbon[:, meas_col], color="tab:blue", label="measured", **SCATTER)
    ax.scatter(carbon[:, 1], carbon[:, pred_col], color="tab:red", label="predicted", **SCATTER)
    gx, gy = running_average(carbon[:, 1], carbon[:, meas_col])
    ax.plot(gx, gy, color="tab:blue", lw=2, label="measured (avg)")
    gx, gy = running_average(carbon[:, 1], carbon[:, pred_col])
    ax.plot(gx, gy, color="tab:red", lw=2, label="predicted (avg)")


# (b) RBE_alpha vs LET for carbon: measured & predicted + running averages.
measured_vs_predicted(ax_rbe_let, 2, 3)
ax_rbe_let.set_xscale("log")
ax_rbe_let.set_xlabel("LET [keV/um]")
ax_rbe_let.set_ylabel("RBE$_\\alpha$")
ax_rbe_let.set_title("(b) carbon: RBE$_\\alpha$ vs LET")
ax_rbe_let.legend(fontsize=7)

# (c) model deviation log10(pred/meas) vs LET, one running average per ion.
n_ion_curves = 0
for z in ION_ZS:
    sub = rec[rec[:, 0] == z]
    if len(sub) < 3:
        continue
    dev = np.log10(sub[:, 3] / sub[:, 2])
    gx, gy = running_average(sub[:, 1], dev)
    ax_dev.plot(gx, gy, lw=2, label=z_to_symbol(z))
    n_ion_curves += 1
ax_dev.axhline(0.0, color="k", lw=1, ls=":")
ax_dev.set_xscale("log")
ax_dev.set_xlabel("LET [keV/um]")
ax_dev.set_ylabel(r"log$_{10}$(RBE$_\alpha$ model / PIDE)")
ax_dev.set_title("(c) relative deviation vs LET, per ion")
if n_ion_curves:
    ax_dev.legend(fontsize=8, title="ion")

# (d) beta_ion/beta_x vs LET for carbon.
measured_vs_predicted(ax_beta, 4, 5)
ax_beta.axhline(0.0, color="k", lw=1, ls=":")
ax_beta.set_xscale("log")
ax_beta.set_xlabel("LET [keV/um]")
ax_beta.set_ylabel(r"$\beta_{ion} / \beta_x$")
ax_beta.set_title("(d) carbon: beta ratio vs LET")
ax_beta.legend(fontsize=7)

fig.suptitle(f"PIDE validation of {MODEL_NAME} (R_n={R_N_UM} um)", fontsize=13)
fig.tight_layout(rect=(0, 0, 1, 0.97))
save_figure(fig, f"pide_model_validation_{MODEL_NAME}.png", directory=FIGURES, dpi=130)

# %% [markdown]
# ## Table-2-style summary: mean RBE ratio per ion
#
# The paper's Table 2 lists the average deviation of `RBE_LEM / RBE_PIDE` per
# ion: the mean of `log10(RBE_LEM / RBE_PIDE)` taken over the **running
# averages** on a logarithmic LET scale (paper Sec. 3, with Fig. 4d), so sparse
# and dense LET regions count equally. A value below 1 is a net
# *underestimation*. We report the geometric mean over the **points** instead,
# which weights densely measured LET regions more (the paper cautions against
# reading much into such single-number averages), plus the mean absolute log
# deviation (spread; lower is better) -- the quantity the benchmark harness
# aggregates.

# %%
print(f"\n{MODEL_NAME}: RBE_alpha model/PIDE agreement by ion")
print(f"{'ion':>4} {'n':>4} {'geo-mean ratio':>15} {'mean |log dev|':>15}")
for z in ION_ZS:
    sub = rec[rec[:, 0] == z]
    if not len(sub):
        continue
    logdev = np.log10(sub[:, 3] / sub[:, 2])
    geo_mean = 10.0 ** float(logdev.mean())
    mald = float(np.abs(logdev).mean())
    print(f"{z_to_symbol(z):>4} {len(sub):>4} {geo_mean:>15.3f} {mald:>15.4f}")

table2 = ", ".join(f"{ion} {value:.2f}" for ion, value in PFUHL2022_TABLE2_RBE_ALPHA.items())
print(
    f"\nExpected (Pfuhl 2022, Table 2, RBE_alpha, LEM IV): {table2} "
    "-- a net underestimation for H, C and Ne, an overestimation for He.\n"
    "The paper's 'underestimation growing with Z' is a low-LET trend (its Fig. 4c), "
    "not a property of these averages."
)
