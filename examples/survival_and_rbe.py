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
# # Cell-survival curves, D10, and RBE at iso-survival
#
# Compares a low-LET photon reference with a higher-LET ion using the
# linear-quadratic model $-\ln S(D) = \alpha D + \beta D^2$, marks the 10%
# survival dose (D10) of each, and reports the RBE at 10% survival.
#
# This file is a [jupytext](https://jupytext.readthedocs.io) percent script: run
# it directly with `python`, or pair/open it as a notebook (see
# `examples/README.md`).

# %%
import matplotlib.pyplot as plt
import numpy as np

from pyradbio import LQParameters, dose_for_survival, rbe, survival
from pyradbio.plotting import figures_dir, save_figure

FIGURES = figures_dir(globals().get("__file__"))

# %% [markdown]
# Pick an LQ parameterisation for the reference photon beam and an illustrative
# high-LET ion.

# %%
photon = LQParameters(alpha=0.313, beta=0.0615)  # HSG-like photon reference
ion = LQParameters(alpha=0.75, beta=0.05)  # illustrative high-LET ion

# %% [markdown]
# Evaluate the survival curves and the iso-survival metrics.

# %%
doses = np.linspace(0.0, 8.0, 400)
s_photon = [survival(float(d), photon.alpha, photon.beta) for d in doses]
s_ion = [survival(float(d), ion.alpha, ion.beta) for d in doses]

d10_photon = dose_for_survival(0.10, photon.alpha, photon.beta)
d10_ion = dose_for_survival(0.10, ion.alpha, ion.beta)
rbe10 = rbe(0.10, ion.alpha, ion.beta, reference=photon)

print(f"Photon D10 = {d10_photon:.2f} Gy")
print(f"Ion    D10 = {d10_ion:.2f} Gy")
print(f"RBE at 10% survival = {rbe10:.2f}")

# %% [markdown]
# Plot the curves and annotate each D10.

# %%
fig, ax = plt.subplots(figsize=(6, 4.5))
ax.semilogy(doses, s_photon, label=f"photon (α={photon.alpha}, β={photon.beta})")
ax.semilogy(doses, s_ion, label=f"ion (α={ion.alpha}, β={ion.beta})")
ax.axhline(0.10, color="grey", ls=":", lw=1)
# stagger the two D10 labels so they don't collide on the 10% line
for d10, color, offset in ((d10_ion, "C1", (0, -18)), (d10_photon, "C0", (6, 10))):
    ax.plot([d10], [0.10], "o", color=color)
    ax.annotate(
        f"D10={d10:.2f} Gy", (d10, 0.10), textcoords="offset points", xytext=offset, color=color
    )
ax.set_xlabel("Dose [Gy]")
ax.set_ylabel("Surviving fraction")
ax.set_ylim(1e-3, 1.0)
ax.set_title(f"LQ survival — RBE(10%) = {rbe10:.2f}")
ax.legend()
fig.tight_layout()

save_figure(fig, "survival_and_rbe.png", directory=FIGURES, dpi=130)
