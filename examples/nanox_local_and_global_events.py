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
# # NanOx: local lethal events, global events, and the boundary of what is public
#
# **NanOx** (NANodosimetry and OXidative stress): Cunha et al. 2017,
# doi:10.1088/1361-6560/aa54c9; the formalism of Alcocer-Ávila et al. 2023,
# doi:10.3389/fphy.2023.1011062; their hadrontherapy implementation (2022),
# doi:10.3389/fphy.2022.1011063; the ELLF of Monini et al. 2020,
# doi:10.1667/RR15463.1. Survival factorises into $S = S_L \times S_{NL}$:
# **local lethal events** in 20 nm × 10 nm water cylinders, scored with the
# stochastic **restricted specific energy** $z$ (no Poisson law, so no
# "non-Poissonian correction"), and **global events** — oxidative-stress
# accumulation over the nucleus, the only non-local class implemented.
#
# The forward $\alpha(\mathrm{LET})$ needs the nanometric spectrum $dP/dz$ and the
# relative chemical effectiveness RCE, both from the group's non-public LQD /
# PHYCHEML / CHEM chain and figure-only in the papers, so pyRadBio ships the
# assembly layer and refuses the ion route rather than substitute an
# amorphous-track spectrum (panel 5 measures why). The printed $\beta_G =
# \beta_r/\eta^2$ chain is asserted in `tests/models/test_nanox.py`; the 2023
# Table 5 $\langle E_p/E_c\rangle$ is `validation/nanox_alcocer_avila2023.py`. See
# `docs/references/NanOx.md`.

# %%
from __future__ import annotations

import textwrap

import matplotlib.pyplot as plt
import numpy as np

from pyradbio.core.photon import PhotonParameters
from pyradbio.core.radiation import Ion
from pyradbio.models.nanox import (
    ETA_RESTRICTED_FRACTION,
    LocalTargetSpectrum,
    NanOxModel,
    NanOxNanodosimetryRequired,
)
from pyradbio.plotting import figures_dir, save_figure

FIGURES = figures_dir(globals().get("__file__"))

CELL_LINES = {
    "V79": (NanOxModel.v79, "#1f77b4"),
    "CHO-K1": (NanOxModel.cho_k1, "#d62728"),
    "HSG": (NanOxModel.hsg, "#2ca02c"),
}

fig, axes = plt.subplots(2, 3, figsize=(16.5, 9.5))
ax1, ax2, ax3, ax4, ax5, unused = axes.ravel()
unused.remove()  # the beta_G chain is asserted in tests/models/test_nanox.py

# %% [markdown]
# ## Panel 1 — the effective local lethal function
#
# $F(z) = \frac{h}{2}\left[1 + \mathrm{erf}\left(\frac{z-z_0}{\sigma}\right)\right]$,
# the three-parameter summary of an ELLF Monini et al. 2020 fitted with **no prior
# shape** (ramp basis functions against measured $\alpha$), which *found* a
# threshold and a saturation. $F$ is not a probability but $-N\ln(1-f)$, an
# effective count over the $N$ targets, so it saturates near $10^5$. The cell
# lines differ far more in **sharpness** than in threshold: HSG's $\sigma = 549$
# Gy against V79's 8117 Gy, a factor of 15.

# %%
z = np.logspace(2, 5.3, 600)
for name, (build, colour) in CELL_LINES.items():
    ellf = build().cell_line.ellf
    ax1.plot(z, ellf.values(z) / ellf.h, color=colour, lw=2, label=name)
    ax1.axvline(ellf.z0_Gy, color=colour, ls=":", lw=1, alpha=0.6)
ax1.set_xscale("log")
ax1.set_xlabel(r"restricted specific energy $z$ in a 20 nm target [Gy]")
ax1.set_ylabel(r"$F(z)\,/\,h$")
ax1.set_title("1. Effective local lethal function\n(dotted: each threshold $z_0$)")
ax1.legend(frameon=False)
ax1.grid(alpha=0.3)

# %% [markdown]
# ## Panel 2 — the core carries the physics in 0.05% of the volume
#
# The track is split into a **core** (a 200 nm square parallelepiped around the
# path) and a **penumbra** of fast $\delta$ electrons, treated *as if it were the
# reference photon radiation* (Approximation 2). $r_c = V_c/V_s \approx 5\times
# 10^{-4}$ for a real nucleus, yet all the radiation-quality dependence lives there.

# %%
diam = np.linspace(6.0, 20.0, 200)
r_c = [NanOxModel.v79(sensitive_volume_diameter_um=float(d)).core_fraction() for d in diam]
ax2.plot(diam, np.array(r_c) * 1e4, color="#7f7f7f", lw=2)
for name, (build, colour) in CELL_LINES.items():
    m = build()
    d_s, r_c_m = m.sensitive_volume_diameter_um, m.core_fraction()
    ax2.plot(d_s, r_c_m * 1e4, "o", color=colour, ms=9, label=f"{name} ({r_c_m:.1e})")
ax2.set_xlabel(r"sensitive-volume diameter $d_s$ [$\mu$m]")
ax2.set_ylabel(r"core volume fraction $r_c\ [\times 10^{-4}]$")
ax2.set_title(
    "2. The track core is ~0.05% of the nucleus\n(and carries all the quality dependence)"
)
ax2.legend(frameon=False, fontsize=8, loc="lower left")
ax2.grid(alpha=0.3)

# %% [markdown]
# ## Panel 3 — the shoulder is *entirely* a global-event effect
#
# Local lethal events obey single-hit single-target statistics, so averaging over
# a Poisson number of impacts gives $S_L(D) = \exp(-\zeta D/(a\,\mathrm{LET})) =
# e^{-\alpha D}$ — a **straight line** at any LET (2023 eq. 52), NanOx's sharpest
# formal departure from the LEM. Every curvature comes from $S_G = \exp(-\beta_G
# \tilde Z^2)$, and dense tracks let primary radicals recombine, so RCE $= G/G_r$
# **falls with LET** and the shoulder flattens. RCE is swept illustratively (the
# published curve, 2023 Fig. 5, is not tabulated); the curves use the papers'
# mean-field limit, eq. 33 (valid for $\sigma_Z \ll \langle Z\rangle$).

# %%
V79_PHOTON = PhotonParameters(alpha_x=0.112, beta_x=0.0259)  # a mixed pair: docs NanOx.md
model = NanOxModel.v79(photon=V79_PHOTON).with_beta_G_from_photon()
dose = np.linspace(0.0, 8.0, 200)
label = r"photons (RCE=1): recovers $e^{-\alpha_x D-\beta_x D^2}$"
ax3.semilogy(dose, [model.photon_survival(float(d)) for d in dose], "k-", lw=2.5, label=label)


def segment(alpha_local: float, rce: float) -> list[float]:
    return [
        model.survival_track_segment(dose_Gy=float(d), alpha_local_per_Gy=alpha_local, rce=rce)
        for d in dose
    ]


label = "local lethal events only: a pure exponential"
ax3.semilogy(dose, segment(0.112, 0.0), "k--", lw=2, label=label)
for rce, colour in zip([0.85, 0.55, 0.25], ["#1f77b4", "#ff7f0e", "#d62728"], strict=True):
    label = rf"ion, RCE={rce:.2f} ($\beta$={model.effective_beta_per_Gy2(rce):.4f})"
    ax3.semilogy(dose, segment(0.40, rce), color=colour, lw=1.8, label=label)
ax3.set_xlabel("dose [Gy]")
ax3.set_ylabel("survival fraction")
ax3.set_ylim(1e-4, 1.2)
ax3.set_title("3. Local events give a straight line;\nthe shoulder is all oxidative stress")
ax3.legend(frameon=False, fontsize=7.5)
ax3.grid(alpha=0.3, which="both")

# %% [markdown]
# ## Panel 4 — the inactivation cross-section is independent of the surface of influence
#
# The single-impact survival $S_1$ is defined relative to a chosen surface of
# influence $\Sigma$ and tends to 1 as $\Sigma$ grows; $\zeta = \Sigma(1-S_1)$ is
# intrinsic (2023 eqs. 55–57). So $\alpha = \zeta/(a\,\mathrm{LET})$ turns a
# *published* $\alpha$ into NanOx's own cross-section without $\Sigma$ — how the
# representative datasets are built (2022 eq. 45).

# %%
surfaces = np.logspace(np.log10(80.0), 4.0, 200)
zeta_true = 30.0
s1 = [model.single_impact_survival(zeta_um2=zeta_true, surface_um2=float(s)) for s in surfaces]
zeta_back = [
    model.inactivation_cross_section_from_single_impact(v, surface_um2=float(s))
    for v, s in zip(s1, surfaces, strict=True)
]
ax4.semilogx(surfaces, s1, color="#d62728", lw=2, label=r"$S_1$ (depends on $\Sigma$)")
ax4.set_xlabel(r"surface of influence $\Sigma$ [$\mu$m$^2$]")
ax4.set_ylabel(r"$S_1$", color="#d62728")
ax4.tick_params(axis="y", labelcolor="#d62728")
ax4.set_ylim(0.0, 1.05)
ax4.grid(alpha=0.3)
ax4b = ax4.twinx()
ax4b.semilogx(surfaces, zeta_back, color="#1f77b4", lw=2.5, label=r"$\zeta=\Sigma(1-S_1)$")
ax4b.axhline(zeta_true, color="#1f77b4", ls=":", lw=1)
ax4b.set_ylabel(r"$\zeta$ [$\mu$m$^2$]", color="#1f77b4")
ax4b.tick_params(axis="y", labelcolor="#1f77b4")
ax4b.set_ylim(0.0, 2 * zeta_true)
spread = max(zeta_back) - min(zeta_back)
ax4.set_title(
    "4. $\\zeta$ is intrinsic, $S_1$ is not\n"
    f"(2023 eqs. 55-57; $\\zeta$ spread over $10^2$ in $\\Sigma$: {spread:.1e})"
)
ax4b.legend(frameon=False, fontsize=8, loc="center left")

# %% [markdown]
# ## Panel 5 — why the missing spectrum cannot simply be substituted
#
# $\alpha_c$ is the mean of $F$ over the *hit* nanotargets divided by their mean
# specific energy (2022 eq. 26); with $F$'s threshold near $2\times10^4$ Gy it is
# set by how much of the spectrum sits above it. Log-normal spectra
# (`LocalTargetSpectrum.log_normal`, an illustration, not NanOx physics) of fixed
# median but varying width move $\alpha_c$ by more than a **decade**. A mean-field
# or amorphous-track description gets the mean right and the fluctuations wrong,
# so it cannot get $\alpha_c$ even approximately right — precisely NanOx's
# criticism of the LEM — and `alpha_beta()` raises instead of guessing.

# %%
median_z = 8.0e3
widths = np.linspace(0.15, 2.4, 40)
alphas = [model.alpha_core(LocalTargetSpectrum.log_normal(median_z, w)) for w in widths]
ax5.semilogy(widths, alphas, color="#9467bd", lw=2.5)
ax5.set_xlabel(r"log-normal width $\sigma_{\ln z}$ of the nanotarget spectrum")
ax5.set_ylabel(r"$\alpha_c$ [Gy$^{-1}$]  (eq. 26)")
ax5.set_title(
    f"5. Same median $z$, {max(alphas) / min(alphas):.0f}x spread in $\\alpha_c$\n"
    "-> the fluctuations ARE the model; they cannot be faked"
)
ax5.grid(alpha=0.3, which="both")
text = f"median $z$ fixed at\n{median_z:.0f} Gy throughout"
ax5.annotate(text, xy=(0.05, 0.9), xycoords="axes fraction", fontsize=8, va="top")

fig.suptitle(
    "NanOx (Cunha 2017; Monini 2020; Alcocer-Avila 2022, 2023): "
    "local lethal events, global events, and the documented scope boundary",
    fontsize=13,
)
fig.tight_layout(rect=(0, 0, 1, 0.96))
save_figure(fig, "nanox_local_and_global_events.png", directory=FIGURES, dpi=140)

# %% [markdown]
# ## The boundary, demonstrated
#
# The ion route raises and says exactly what is missing and why: a documented
# boundary beats a complete implementation built on invented numbers.

# %%
try:
    NanOxModel.v79(photon=V79_PHOTON).alpha_beta(Ion(Z=6, A=12, energy_MeV_u=100.0))
except NanOxNanodosimetryRequired as exc:
    print("\nalpha_beta(carbon) raises NanOxNanodosimetryRequired:\n")
    print(textwrap.fill(str(exc), width=78, initial_indent="   ", subsequent_indent="   "))

print("\n--- what IS reproduced from the papers, with no free parameters ---")
print(f"eta (restricted fraction)          : {ETA_RESTRICTED_FRACTION}")
m = NanOxModel.v79()
print(f"\nV79 sigma_s                        : {m.geometric_cross_section_um2():.2f} um^2")
print(f"V79 core fraction r_c              : {m.core_fraction():.3e}")
print(f"V79 ELLF F(0)/h                    : {m.cell_line.ellf(0.0) / m.cell_line.ellf.h:.2e}")
print(
    f"Cunha 2017 d_s=9.78 um -> sigma_s  : "
    f"{NanOxModel.v79(sensitive_volume_diameter_um=9.78).geometric_cross_section_um2():.1f} um^2"
    "  (paper prints 75)"
)
