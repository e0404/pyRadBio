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
# # MKM: z*_1D and α versus energy
#
# The modified microdosimetric-kinetic model (Inaniwa et al. 2010,
# [doi:10.1088/0031-9155/55/22/008](https://doi.org/10.1088/0031-9155/55/22/008))
# makes the radiation-quality dependence of survival enter through a single
# quantity: the saturation-corrected dose-mean specific energy of a subcellular
# domain, $z^{*}_{1D}$. The LQ slope is then
# $\alpha_{\mathrm{MKM}} = \alpha_0 + \beta\, z^{*}_{1D}$ with a
# radiation-independent $\beta$.
#
# `MKMModel` owns that biophysics and delegates the $z^{*}_{1D}$ computation to a
# pluggable backend; the default is the pyMKM-free `NativeMKMBackend`. Its
# $z^{*}_{1D}$ is cross-checked against pyMKM's by
# `pytest -m requires_pymkm tests/models/test_native_mkm_backend.py` (needs the
# `mkm` extra; carbon across energies, the sphere-vs-disk overshoot, the
# stopping-power choice). Part 1 sweeps a carbon
# beam in energy and plots $z^{*}_{1D}$ and $\alpha$; part 2 plots $z^{*}_{1D}$
# for every ion from hydrogen to neon, the Inaniwa 2010 Fig. 5 family.
#
# This file is a [jupytext](https://jupytext.readthedocs.io) percent script: run
# it directly with `python`, or pair/open it as a notebook (see
# `examples/README.md`).

# %%
import matplotlib.pyplot as plt
import numpy as np

from pyradbio.core.radiation import Ion
from pyradbio.models.mkm import MKMModel, NativeMKMBackend
from pyradbio.plotting import figures_dir, save_figure

FIGURES = figures_dir(globals().get("__file__"))

# %% [markdown]
# ## Part 1: carbon, α and z*_1D
#
# Aerobic HSG geometry and parameters from Inaniwa 2010, Sec. 2.2
# ($r_d = 0.32\,\mu m$, $R_n = 3.9\,\mu m$, $\alpha_0 = 0.172\,\mathrm{Gy^{-1}}$,
# $\beta = 0.0615\,\mathrm{Gy^{-2}}$). The model builds its default native
# backend from this geometry. The carbon-12 grid runs log-spaced from near the
# Bragg peak to plateau energies (30 points).

# %%
model = MKMModel(domain_radius_um=0.32, nucleus_radius_um=3.9, alpha0=0.172, beta0=0.0615)
energies_MeV_u = np.geomspace(1.5, 450.0, 30).tolist()

# alpha_beta_z_star_table returns dependency-free rows with z*_1D, alpha and beta values.
table = model.alpha_beta_z_star_table([6], energies_MeV_u)
rows = sorted(table.rows, key=lambda row: row.energy_MeV_u)
# Show a downsampled view so the printout stays readable.
print("energy_MeV_u  let_keV_um  z_star_1D_Gy  alpha   beta")
for row in rows[::4]:
    let = float("nan") if row.let_keV_um is None else row.let_keV_um
    print(
        f"{row.energy_MeV_u:12.3f}  {let:10.3f}  {row.z_star_1D_Gy:13.3f}  "
        f"{row.alpha:5.3f}  {row.beta:5.4f}"
    )

# %% [markdown]
# $z^{*}_{1D}$ (and therefore $\alpha$) falls with energy as the LET drops -- the
# Inaniwa 2010 Fig. 5 shape. We plot $\alpha$ against LET, the natural
# radiation-quality axis.

# %%
fig, ax1 = plt.subplots(figsize=(6, 4.5))
lets = [row.let_keV_um for row in rows if row.let_keV_um is not None]
alphas = [row.alpha for row in rows if row.let_keV_um is not None]
z_stars = [row.z_star_1D_Gy for row in rows if row.let_keV_um is not None]
ax1.plot(lets, alphas, "-", color="C0", label="α_MKM")
ax1.set_xscale("log")
ax1.set_xlabel("LET [keV/µm]")
ax1.set_ylabel("α [Gy⁻¹]", color="C0")
ax1.tick_params(axis="y", labelcolor="C0")

ax2 = ax1.twinx()
ax2.plot(lets, z_stars, "--", color="C1", label="z*_1D")
ax2.set_ylabel("z*_1D [Gy]", color="C1")
ax2.tick_params(axis="y", labelcolor="C1")

ax1.set_title("MKM (HSG): α and z*_1D vs LET for $^{12}$C (native backend)")
fig.tight_layout()
save_figure(fig, "mkm_alpha_beta_vs_energy.png", directory=FIGURES, dpi=130)

# %% [markdown]
# Sanity check the cited single value: at a clinical carbon energy the model
# returns a positive, physically sensible α, and β is exactly the input β₀.

# %%
ab = model.alpha_beta(Ion(Z=6, A=12, energy_MeV_u=11.0))
print(f"alpha={ab.alpha:.3f} Gy^-1, beta={ab.beta:.4f} Gy^-2 at 11 MeV/u carbon")

# %% [markdown]
# ## Part 2: z*_1D for hydrogen to neon (Inaniwa 2010 Fig. 5)
#
# Each ion's $z^{*}_{1D}$ **peaks** at a characteristic energy per nucleon and
# falls off at both higher energy (LET decreasing) and very low energy (the
# saturation correction -- "overkill" -- turning the curve over). The peak shifts
# to **higher** energy per nucleon for heavier ions (helium's peak sits at the
# low-energy end of the grid, neon's well above 10 MeV/u). Same HSG domain
# geometry; we query the backend directly because this part is about
# $z^{*}_{1D}$ itself, not the LQ slope (ten ions x 30 log-spaced energies).

# %%
backend = NativeMKMBackend(domain_radius_um=0.32, nucleus_radius_um=3.9, beta0=0.0615)

# Z -> symbol for every ion from hydrogen (Z=1) to neon (Z=10).
ions = {1: "H", 2: "He", 3: "Li", 4: "Be", 5: "B", 6: "C", 7: "N", 8: "O", 9: "F", 10: "Ne"}
energies_MeV_u = np.geomspace(0.1, 800.0, 30).tolist()

table = backend.z_star_1D_table_Gy(list(ions), energies_MeV_u)

# Report each ion's peak z*_1D and the energy at which it occurs.
for z, name in ions.items():
    sub = [row for row in table.rows if row.ion_Z == z]
    peak = max(sub, key=lambda row: row.z_star_1D_Gy)
    zp, ep = peak.z_star_1D_Gy, peak.energy_MeV_u
    print(f"{name:>3}: peak z*_1D = {zp:6.2f} Gy at {ep:6.2f} MeV/u")

# %%
fig, ax = plt.subplots(figsize=(6, 4.5))
colors = plt.cm.viridis(np.linspace(0.0, 0.95, len(ions)))
for color, (z, name) in zip(colors, ions.items(), strict=True):
    sub = sorted(
        (row for row in table.rows if row.ion_Z == z),
        key=lambda row: row.energy_MeV_u,
    )
    ax.plot(
        [row.energy_MeV_u for row in sub],
        [row.z_star_1D_Gy for row in sub],
        color=color,
        label=name,
    )

ax.set_xscale("log")
ax.set_xlabel("Energy [MeV/u]")
ax.set_ylabel("z*_1D [Gy]")
ax.set_title("MKM (HSG): z*_1D vs energy per nucleon")
ax.legend(title="Ion", ncol=2)
fig.tight_layout()
save_figure(fig, "mkm_zstar_vs_energy.png", directory=FIGURES, dpi=130)
