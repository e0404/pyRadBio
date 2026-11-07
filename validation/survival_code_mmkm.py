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
# # Modified MKM vs the "Survival" code's rapid mMKM (HSG)
#
# pyRadBio implements the MKM family itself (pyMKM is an optional cross-check). A
# good sanity check is to reproduce the output of an *independent* implementation
# of the same model: the open-source **Survival** toolkit (Manganaro et al. 2018,
# doi:10.1088/1361-6560/aab697) exports per-ion $\alpha(E)$ for **modified MKM**
# (rapid mMKM, Inaniwa 2010) for the HSG cell line (`mMKM_HSG_SurvivalCode.json`,
# external: `PYRADBIO_SURVIVAL_DIR`). It is a reference *model output*, not
# experimental data, so tight agreement is expected where pyRadBio implements the
# same approximation; `tests/benchmark/test_reference_tables.py` pins it as a
# golden (median |rel err| < 12% per ion, 10-400 MeV/u).
#
# The LEM comparisons live elsewhere: LEM I-III against Survival in
# `survival_code_lem.py`, LEM I and LEM IV against the TRiP98 tables in
# `trip98_lem1_rbe.py` / `trip98_lem4_rbe.py`.
# A LEM I $\alpha(E)$ comparison against the Survival rapidLEM HSG table
# (`LEMI_Scholz2006_HSG_SurvivalCode.json`; `LEM1` with `ClusterParameters()` on
# the table's alphaX/betaX/Dt/rNucleus; every 3rd tabulated energy over 1-1000
# MeV/u, then the 10-400 MeV/u window) gives a median |rel err| on $\alpha$ of
# 1H 6.0%, 4He 3.3%, 12C 2.6%; `test_golden_lem1_matches_survival_rapidlem_hsg`
# in `tests/benchmark/test_reference_tables.py` bounds the same comparison at
# < 12% per ion (on its own 20-point subsample of the window).
#
# The figure is written to `benchmark_reference_models.png`.
#
# `MKMModel` runs on its default native backend; `PYRADBIO_USE_PYMKM=1` swaps in
# pyMKM (optional `mkm` extra) as a cross-check. Quick mode (`PYRADBIO_QUICK=1`)
# evaluates every 15th tabulated energy instead of every 5th.

# %%
from __future__ import annotations

import os

import matplotlib.pyplot as plt
import numpy as np
from _common import pick, quick_mode, require_data, require_extra

from pyradbio import Ion
from pyradbio.benchmark import reference_tables as rt
from pyradbio.datasets import SURVIVAL
from pyradbio.models.mkm.model import MKMModel
from pyradbio.plotting import figures_dir, save_figure

FIGURES = figures_dir(globals().get("__file__"))

# Set PYRADBIO_USE_PYMKM=1 to run on the pyMKM backend instead of the native one.
USE_PYMKM = os.environ.get("PYRADBIO_USE_PYMKM") == "1"
# Ions to display (Z, A, label, colour); the reference table covers H-Ne. ASCII
# labels keep console printing portable (Windows cp1252).
IONS = [(1, 1, "1H", "C0"), (2, 4, "4He", "C1"), (6, 12, "12C", "C3")]
# Cost knob: the stride over the tabulated 1-1000 MeV/u energies.
STRIDE = pick(5, 15)
if quick_mode():
    print(f"[quick mode] every {STRIDE}th tabulated energy; numbers are not for quoting")

# %% [markdown]
# ## mMKM α(E) vs the Survival rapid mMKM (HSG)

# %%
mmkm_json = require_data(rt.load_mmkm_survival_hsg, dataset=SURVIVAL)
q = mmkm_json.parameters
geometry = {
    "domain_radius_um": float(q["rDomain"]),
    "nucleus_radius_um": float(q["rNucleus"]),
    "beta0": float(q["betaX"]),
}
if USE_PYMKM:
    require_extra("pymkm", "mkm")
    from pyradbio.models.mkm.pymkm_backend import PyMKMBackend

    mkm = MKMModel(alpha0=float(q["alpha0"]), backend=PyMKMBackend(**geometry))
else:
    mkm = MKMModel(alpha0=float(q["alpha0"]), **geometry)
mmkm_curves: dict[tuple[int, int], tuple[np.ndarray, np.ndarray, np.ndarray]] = {}
print("mMKM vs rapid mMKM  -  median |rel err| on alpha, 10-400 MeV/u")
for Z, A, label, _ in IONS:
    c = mmkm_json.curve(Z, A)
    idx = rt.window_indices(c.energies_MeV_u, 1.0, 1000.0)[::STRIDE]
    e = np.array([c.energies_MeV_u[i] for i in idx])
    ref = np.array([c.alpha[i] for i in idx])
    pred = np.array([mkm.alpha_beta(Ion(Z, A, float(x))).alpha for x in e])
    mmkm_curves[(Z, A)] = (e, ref, pred)
    w = rt.window_indices(e.tolist(), 10.0, 400.0)
    print(f"  {label:>4s} (Z{Z}A{A}): {rt.median_abs_relative_error(pred[w], ref[w]):6.1%}")

# %% [markdown]
# ## Figure
#
# Lines are pyRadBio; open markers are the reference implementation.

# %%
fig, ax_mmkm = plt.subplots(figsize=(6.0, 4.75))
for Z, A, label, colour in IONS:
    e, ref, pred = mmkm_curves[(Z, A)]
    ax_mmkm.plot(e, pred, color=colour, lw=1.8, label=f"{label} pyRadBio")
    ax_mmkm.plot(e, ref, "o", mfc="none", mec=colour, ms=4, mew=1.0)
ax_mmkm.set_xscale("log")
ax_mmkm.set_yscale("log")
ax_mmkm.legend(frameon=False, fontsize=8)
ax_mmkm.set_xlabel("E (MeV/u)")
ax_mmkm.set_ylabel(r"$\alpha$ (Gy$^{-1}$)")
ax_mmkm.set_title("mMKM vs survival rapid mMKM (HSG)")
ax_mmkm.grid(True, which="both", ls=":", lw=0.45)
fig.tight_layout()
print()
save_figure(fig, "benchmark_reference_models.png", directory=FIGURES, dpi=150)
