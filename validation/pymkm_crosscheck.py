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
# # Cross-check against pyMKM: water stopping power and the Kiefer-Chatterjee track
#
# pyMKM (Magro et al., Computation 2025) is an independent implementation of the
# MKM family and of the amorphous tracks underneath it. pyRadBio uses it as a
# cross-check, never as a default, through the adapter
# `pyradbio.models.mkm.pymkm_backend` (the only module that imports pyMKM; needs
# the `mkm` extra). This script compares two inputs the MKM family rests on:
#
# 1. **Water stopping power.** The library default (`default_stopping_power()`)
#    against pyMKM's three bundled tables (Geant4 11.3.0, FLUKA 2020.0,
#    MSTAR 3.12), for protons, helium and carbon.
# 2. **Kiefer-Chatterjee radial dose** $D(r)$ (Chatterjee & Schaefer 1976), the
#    track behind the MKM/SMK `z*`: `track.kiefer_chatterjee` against pyMKM's
#    `ParticleTrack`, both fed the same LET. The two share the same physics;
#    residuals come only from pyMKM's slightly older elementary-charge /
#    atomic-mass constants.
#
# This file is a [jupytext](https://jupytext.readthedocs.io) percent script.
# `PYRADBIO_QUICK=1` runs it on coarser grids.

# %%
import matplotlib.pyplot as plt
import numpy as np
from _common import pick, quick_mode, require_extra

from pyradbio import Ion
from pyradbio.plotting import figures_dir, save_figure
from pyradbio.track.kiefer_chatterjee import KieferChatterjeeRadialDose
from pyradbio.track.stopping_power import (
    SimpleBetheStoppingPower,
    StoppingPower,
    default_stopping_power,
    describe_stopping_power,
)

require_extra("pymkm", "mkm")
from pyradbio.models.mkm.pymkm_backend import (  # noqa: E402  (needs the mkm extra)
    PyMKMStoppingPower,
    pymkm_kiefer_chatterjee_dose,
)

FIGURES = figures_dir(globals().get("__file__"))
N_ENERGIES = pick(200, 40)  # stopping-power curve points per ion
N_RADII = pick(200, 40)  # radial-dose points per ion
if quick_mode():
    print("QUICK MODE: coarse energy and radius grids (PYRADBIO_QUICK=1)")

# %% [markdown]
# ## 1. Water stopping power: library default vs pyMKM's tables
#
# pyMKM's tables cover Z = 1-10 over 0.1-1000 MeV/u.

# %%
IONS = [("proton", 1, 1), ("helium-4", 2, 4), ("carbon-12", 6, 12)]
PYMKM_SOURCES = ("geant4_11_3_0", "fluka_2020_0", "mstar_3_12")

default_sp = default_stopping_power()
sptables: dict[str, StoppingPower] = {describe_stopping_power(default_sp): default_sp}
for source in PYMKM_SOURCES:
    sptables[f"pyMKM_{source}"] = PyMKMStoppingPower(source=source)  # type: ignore[arg-type]
linestyle_map = {name: ["-", "--", "-.", ":"][i % 4] for i, name in enumerate(sptables)}

energies = np.logspace(np.log10(0.1), np.log10(400.0), N_ENERGIES)  # MeV/u
fig, ax = plt.subplots(figsize=(13, 4.5))  # room for the long default-table label
for name, z, a in IONS:
    for sp_name, table in sptables.items():
        let = table.lets_keV_um(z, a, energies)
        ax.loglog(energies, let, label=f"{name} ({sp_name})", linestyle=linestyle_map[sp_name])
    let_100 = default_sp.let_keV_um(Ion(z, a, 100.0))  # the default table, by name
    print(f"{name:>10s}: LET(100 MeV/u) = {let_100:8.3f} keV/um")

ax.set_xlabel("Energy [MeV/u]")
ax.set_ylabel("LET in water [keV/µm]")
ax.set_title("Stopping Powers in water")
ax.legend(loc="center left", bbox_to_anchor=(1, 0.5), fontsize=8)
ax.grid(True, which="both", ls=":", lw=0.5)
fig.tight_layout()
save_figure(fig, "water_stopping_power.png", directory=FIGURES, dpi=130)

# %% [markdown]
# ## 2. Kiefer-Chatterjee radial dose: pyRadBio vs pyMKM
#
# Both sides get the LET from the dependency-free analytic Bethe stopping power,
# pinned explicitly so the comparison isolates the track model. Each profile is
# evaluated over its own core-to-penumbra radius span; the pyRadBio
# normalisation is confirmed by integrating the radial energy back to a LET.

# %%
sp = SimpleBetheStoppingPower()
model = KieferChatterjeeRadialDose(stopping_power=sp)
ions = [
    Ion(Z=1, A=1, energy_MeV_u=100.0),
    Ion(Z=6, A=12, energy_MeV_u=100.0),
    Ion(Z=10, A=20, energy_MeV_u=200.0),
]

fig, ax = plt.subplots(figsize=(6.5, 4.5))
colors = plt.rcParams["axes.prop_cycle"].by_key()["color"]
for ion, color in zip(ions, colors, strict=False):
    radii = np.geomspace(0.1 * model.r_min_um(ion), 0.99 * model.r_max_um(ion), N_RADII)
    ours = np.array([model.local_dose_Gy(float(r), ion) for r in radii])
    theirs = pymkm_kiefer_chatterjee_dose(ion, radii, sp.let_keV_um(ion))

    label = f"Z={ion.Z} at {ion.energy_MeV_u:.0f} MeV/u"
    ax.loglog(radii * 1e3, ours, color=color, label=label)  # nm on the x axis
    ax.loglog(radii * 1e3, theirs, color=color, ls="--", lw=1.0, alpha=0.8)

    recovered = model.radial_integral_LET_keV_um(ion)
    max_rel = float(np.max(np.abs(ours - theirs) / theirs))
    print(
        f"{label:>22s}: LET(in)={sp.let_keV_um(ion):7.2f}  LET(integral)={recovered:7.2f} keV/um"
        f"   max rel. dev. vs pyMKM = {max_rel:.2%}"
    )

ax.plot([], [], color="gray", label="pyRadBio")
ax.plot([], [], color="gray", ls="--", label="pyMKM")
ax.set_xlabel("Radius from track [nm]")
ax.set_ylabel("Local dose D(r) [Gy]")
ax.set_title("Kiefer-Chatterjee radial dose — pyRadBio vs pyMKM")
ax.legend(fontsize=8)
ax.grid(True, which="both", ls=":", lw=0.5)
fig.tight_layout()
save_figure(fig, "kiefer_chatterjee_vs_pymkm.png", directory=FIGURES, dpi=130)
