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
# # LQ parameters of a mixed field: dose-weighted mixing and wide energy bins
#
# A model's `alpha_beta` accepts a whole `RadiationField`, not just one ion; the
# models that cannot mix (e.g. Katz, NanOx) raise instead. The field is a list of
# `SpectrumComponent`s -- each a mono-energetic ion or a finite **energy bin** --
# given by dose or by fluence. The base class turns the field into dose-weighted
# ion nodes (`core.mixing.dose_quadrature`) and mixes the per-ion LQ parameters
# with the Zaider-Rossi / Kanai rule (Pfuhl et al. 2020, Radiat. Res. 193:130,
# doi:10.1667/RR15456.1, eqs. 9-11):
#
# $$\bar\alpha = \sum_i d_i\,\alpha_i,\qquad
#   \sqrt{\bar\beta} = \sum_i d_i \sqrt{\beta_i},\qquad d_i = D_i / D .$$
#
# A bin is not evaluated at its centre: its dose density $\varphi(E)\,S(E)$ and
# the model's $\alpha(E)$ both vary across it, and the integral of their product
# is not the product at the centre (AGENTS.md, "Numerics"). Each bin is
# integrated on a log sub-grid, 16 nodes by default. The model here is LEM I
# with the CHO inputs of Scholz et al. 1997 (`benchmark.scholz1997.CHO_INPUTS`).
#
# This file is a [jupytext](https://jupytext.readthedocs.io) percent script: run
# it directly with `python`, or pair/open it as a notebook (see
# `examples/README.md`).

# %%
import math

import matplotlib.pyplot as plt
import numpy as np

from pyradbio import Ion, RadiationField, SpectrumComponent
from pyradbio.benchmark.scholz1997 import CHO_INPUTS
from pyradbio.core.mixing import dose_average, dose_quadrature
from pyradbio.models.lem.lem1 import LEM1
from pyradbio.plotting import figures_dir, save_figure

FIGURES = figures_dir(globals().get("__file__"))

lem = LEM1(photon=CHO_INPUTS.photon(), cell=CHO_INPUTS.cell_target())

# %% [markdown]
# ## 1. Two mono-energetic components, dose-weighted
#
# A slow carbon ion (10 MeV/u, near the track end) and a fast one (270 MeV/u,
# entrance channel) share a fixed total dose. $\bar\alpha$ is linear in the dose
# fraction; $\bar\beta$ is not, because it is $\sqrt\beta$ that mixes. A plain
# dose average of $\beta$ (dashed) is *not* the rule.

# %%
TOTAL_DOSE_GY = 2.0
high_let, low_let = Ion(6, 12, 10.0), Ion(6, 12, 270.0)
lq_high, lq_low = lem.alpha_beta(high_let), lem.alpha_beta(low_let)

fractions = np.linspace(0.0, 1.0, 11)  # dose fraction in the high-LET component
mixed = []
for f in fractions:
    doses = [f * TOTAL_DOSE_GY, (1.0 - f) * TOTAL_DOSE_GY]
    field = RadiationField(
        [
            SpectrumComponent(high_let, dose_Gy=doses[0]),
            SpectrumComponent(low_let, dose_Gy=doses[1]),
        ]
    )
    mixed.append(lem.alpha_beta(field))
alphas = [lq.alpha for lq in mixed]
betas = [lq.beta for lq in mixed]
naive_betas = [
    dose_average([lq_high.beta, lq_low.beta], [f, 1.0 - f]) for f in fractions
]  # the wrong rule, for contrast

print(f"pure low-LET : alpha={alphas[0]:.3f}, beta={betas[0]:.4f}")
print(f"50/50 mix    : alpha={alphas[5]:.3f}, beta={betas[5]:.4f}")
print(f"pure high-LET: alpha={alphas[-1]:.3f}, beta={betas[-1]:.4f}")

# %% [markdown]
# ## 2. One wide energy bin
#
# A fluence-specified carbon bin spanning 1-30 MeV/u, with its representative
# ion at the log-centre (5.5 MeV/u). LEM I's $\alpha(E)$ peaks right there, but
# a uniform fluence puts most of the dose ($\propto \varphi S$, i.e. $E\,S(E)$ per
# unit $\ln E$) in the upper half of the bin, where $\alpha$ falls again. The
# bin-centre value therefore overshoots the dose-weighted mean (the error is
# printed below).

# %%
E_LO, E_HI = 1.0, 30.0
centre_ion = Ion(6, 12, math.sqrt(E_LO * E_HI))
carbon_bin = SpectrumComponent(
    centre_ion, fluence=1.0e7, energy_lo_MeV_u=E_LO, energy_hi_MeV_u=E_HI
)
bin_field = RadiationField([carbon_bin])

binned = lem.alpha_beta(bin_field)  # 16 log sub-nodes per bin (the default)
centre = lem.alpha_beta(centre_ion)  # what a bin-centre rule would return
let_of = lem.stopping_power.let_keV_um
node_energies = [ion.energy_MeV_u for ion, _dose in dose_quadrature(bin_field, let_of)]
curve_ions = [Ion(6, 12, float(e)) for e in np.geomspace(E_LO, E_HI, 20)]
curve_energies = [ion.energy_MeV_u for ion in curve_ions]
curve_alpha = [lem.alpha_beta(ion).alpha for ion in curve_ions]
dose_per_ln_e = np.array([ion.energy_MeV_u * let_of(ion) for ion in curve_ions])

# %% [markdown]
# ## 3. Convergence in the number of sub-nodes
#
# The relative change of $\bar\alpha$ against a 64-node reference. By the default
# 16 nodes it is ~1e-4 and the convergence has stalled (going from 8 to 16 nodes
# gains less than a factor of two); the bin-centre rule (dotted) is three orders
# of magnitude worse.

# %%
SUB_NODES = (2, 3, 4, 6, 8, 16)
reference = lem.alpha_beta(bin_field, sub_nodes=64)
errors = [
    abs(lem.alpha_beta(bin_field, sub_nodes=n).alpha / reference.alpha - 1.0) for n in SUB_NODES
]
centre_error = abs(centre.alpha / reference.alpha - 1.0)

print(f"\ncarbon bin {E_LO:g}-{E_HI:g} MeV/u, fluence-specified")
print(f"  sub-grid (16 nodes) : alpha={binned.alpha:.4f}, beta={binned.beta:.5f}")
print(f"  reference (64 nodes): alpha={reference.alpha:.4f}, beta={reference.beta:.5f}")
e_c = centre_ion.energy_MeV_u
print(f"  bin centre {e_c:.2f} MeV/u: alpha={centre.alpha:.4f}, beta={centre.beta:.5f}")
print(f"  bin-centre alpha error: {centre.alpha / reference.alpha - 1.0:+.1%}")

# %%
fig, (ax1, ax2, ax3) = plt.subplots(1, 3, figsize=(15.5, 4.5))

ax1.plot(fractions, alphas, "C0", label=r"$\bar\alpha$")
ax1.set_xlabel("Dose fraction in the 10 MeV/u component")
ax1.set_ylabel(r"$\bar\alpha$ [Gy$^{-1}$]", color="C0")
ax1.tick_params(axis="y", labelcolor="C0")
ax1b = ax1.twinx()
ax1b.plot(fractions, betas, "C1", label=r"$\bar\beta$ ($\sqrt{\beta}$ mixed)")
ax1b.plot(fractions, naive_betas, "C1--", lw=1, label=r"plain dose average of $\beta$")
ax1b.set_ylabel(r"$\bar\beta$ [Gy$^{-2}$]", color="C1")
ax1b.tick_params(axis="y", labelcolor="C1")
ax1b.legend(fontsize=8, loc="lower center")
ax1.set_title("1. Two carbon components, 2 Gy total")

ax2.semilogx(curve_energies, curve_alpha, "k", lw=1.5, label=r"$\alpha(E)$, mono-energetic")
ax2.axhline(binned.alpha, color="C2", label=r"bin $\bar\alpha$ (sub-grid)")
ax2.axhline(centre.alpha, color="C3", ls=":", label=r"$\alpha$ at the bin centre")
ax2.set_xlabel("Carbon energy [MeV/u]")
ax2.set_ylabel(r"$\alpha$ [Gy$^{-1}$]")
ax2.plot(node_energies, [0.47] * len(node_energies), "|", color="0.3", ms=12, label="16 sub-nodes")
ax2.plot([], [], color="0.6", ls="--", label=r"dose per unit $\ln E$ (rel.)")
ax2.set_ylim(0.44, 1.02)
ax2.legend(fontsize=7, loc="lower center", bbox_to_anchor=(0.5, 0.09), ncol=2)
ax2b = ax2.twinx()  # the dose density only needs a shape, so this axis has no ticks
ax2b.semilogx(curve_energies, dose_per_ln_e / dose_per_ln_e.max(), color="0.6", ls="--")
ax2b.set_ylim(-1.0, 1.05)
ax2b.set_yticks([])
ax2.set_title(f"2. One bin, {E_LO:g}-{E_HI:g} MeV/u")

ax3.loglog(SUB_NODES, errors, "C2-o", label="sub-grid quadrature")
ax3.axhline(centre_error, color="C3", ls=":", label="bin centre")
ax3.axvline(16, color="0.6", ls="--", lw=1)
ax3.set_xticks(SUB_NODES, [str(n) for n in SUB_NODES])
ax3.minorticks_off()
ax3.set_xlabel("sub-nodes per bin")
ax3.set_ylabel(r"$|\bar\alpha / \bar\alpha_{64} - 1|$")
ax3.legend(fontsize=8)
ax3.set_title("3. Convergence (default: 16)")

fig.suptitle("Mixed-field LQ parameters, LEM I (CHO, Scholz et al. 1997)")
fig.tight_layout()
save_figure(fig, "mixed_field_parameters.png", directory=FIGURES, dpi=130)
