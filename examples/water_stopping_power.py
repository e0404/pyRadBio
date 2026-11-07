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
# # Linear energy transfer (LET) in water versus energy
#
# Uses the library's default water stopping power (`default_stopping_power()`:
# ICRU 90's procedure for Z 1-18, Hubert alpha scaling for Z 19-92 from
# 2.5 MeV/u, analytic Bethe anchored to them elsewhere). The curves show the
# expected $\sim 1/\beta^2$ rise toward lower energy and the $\sim Z^2$ ordering
# between species (protons < helium < carbon at equal energy per nucleon). The
# overlay of pyMKM's bundled
# tables lives in `validation/pymkm_crosscheck.py` (it needs the `mkm` extra).
#
# This file is a [jupytext](https://jupytext.readthedocs.io) percent script: run
# it directly with `python`, or pair/open it as a notebook (see
# `examples/README.md`).

# %%
import matplotlib.pyplot as plt
import numpy as np

from pyradbio import Ion
from pyradbio.plotting import figures_dir, save_figure
from pyradbio.track.stopping_power import default_stopping_power, describe_stopping_power

FIGURES = figures_dir(globals().get("__file__"))

# (name, Z, A)
IONS = [("proton", 1, 1), ("helium-4", 2, 4), ("carbon-12", 6, 12)]

# %% [markdown]
# Evaluate the LET over a therapy-relevant energy range for each ion.

# %%
sp = default_stopping_power()
sp_name = describe_stopping_power(sp)
energies = np.logspace(np.log10(0.1), np.log10(400.0), 200)  # MeV/u

fig, ax = plt.subplots(figsize=(10, 4.5))
for name, z, a in IONS:
    let = sp.lets_keV_um(z, a, energies)
    ax.loglog(energies, let, label=f"{name} ({sp_name})", linestyle="-")
    print(f"{name:>10s}: LET(100 MeV/u) = {sp.let_keV_um(Ion(z, a, 100.0)):8.3f} keV/um")

ax.set_xlabel("Energy [MeV/u]")
ax.set_ylabel("LET in water [keV/µm]")
ax.set_title("Stopping Powers in water")
ax.legend(loc="center left", bbox_to_anchor=(1, 0.5))
ax.grid(True, which="both", ls=":", lw=0.5)
fig.tight_layout()

save_figure(fig, "water_stopping_power.png", directory=FIGURES, dpi=130)
