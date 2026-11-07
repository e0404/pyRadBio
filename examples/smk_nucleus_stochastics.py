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
# # SMK: what nucleus-level stochastics change
#
# The **stochastic microdosimetric kinetic model** (SMK; Inaniwa & Kanematsu
# 2018, doi:10.1088/1361-6560/aabede) adds to the modified MKM's domain-level
# saturation the **nucleus**-level fluctuation of the specific energy about `D`:
#
# 1. **`β` becomes quality dependent**: `β_SMK = (z*_1D / z_1D) β0` falls with
#    LET, where mMKM holds `β = β_ref` flat.
# 2. **Survival stops being LQ**: `S(D) = exp(-(αD + βD²)) · (1 + γ(D)·D)` with
#    `γ ∝ z̄_n`, lifting survival at high dose.
#
# The price (panel 3): eq. (24) is a *truncated* expansion, so once `γ(D)·D`
# nears 1 it extrapolates. With the paper's HSG parameters that ceiling sits at
# ~3–18 Gy for carbon, the order of the `D ≲ 10 Gy` the paper works at; only the
# 10 MeV/u point falls slightly below the `S = 0.1` iso-survival dose. The model
# reports it via `survival_valid_to_Gy`.
#
# Parameters: table 1 HSG via `StochasticMKM.hsg_inaniwa2018()` — `R_n = 8.1 µm`
# (measured) and `z0 = 66.0 Gy` (a free fit); see `docs/references/SMK_Inaniwa2018.md`.
# The sites are the paper's **cylinders** (sec 3.2, `site_geometry="cylinder"`, as
# in `NativeMKMBackend` and pyMKM); panel 4 shows a sphere moving `z*_1D` by -9%
# to +29% over 1-100 MeV/u (+39% by 400 MeV/u, not plotted).

# %%
import matplotlib.pyplot as plt
import numpy as np

from pyradbio.core.lq import dose_for_survival
from pyradbio.core.radiation import Ion
from pyradbio.models.mkm.spectrum_models import ModifiedMKM, StochasticMKM
from pyradbio.plotting import figures_dir, save_figure
from pyradbio.track.stopping_power import SimpleBetheStoppingPower

FIGURES = figures_dir(globals().get("__file__"))

ENERGIES = [1.0, 10.0, 100.0]  # MeV/u -- LET *decreases* along this list

smk = StochasticMKM.hsg_inaniwa2018()  # table 1: r_d 0.28, R_n 8.1, z0 66.0 Gy
smk_sphere = StochasticMKM.hsg_inaniwa2018(site_geometry="sphere")  # for panel 4
R_D, R_N = smk.domain_radius_um, smk.nucleus_radius_um
# mMKM for contrast, on its own (Kase-geometric) footing at the same domain size
mmkm = ModifiedMKM(domain_radius_um=R_D, nucleus_radius_um=4.3)
sp = SimpleBetheStoppingPower()

ions = {e: Ion(6, 12, e) for e in ENERGIES}
lets = {e: sp.let_keV_um(ions[e]) for e in ENERGIES}

# %% [markdown]
# ## The numbers behind the panels

# %%
print(
    f"{'E [MeV/u]':>10} {'LET':>8} {'z*_1D':>8} {'z_1D':>9} {'z_n':>7} "
    f"{'alpha':>7} {'beta_SMK':>9} {'beta_mMKM':>10} {'gD=1 at':>9}"
)
rows = {}
for e in ENERGIES:
    z = smk.specific_energies(ions[e])
    lq_s = smk.alpha_beta(ions[e])
    lq_m = mmkm.alpha_beta(ions[e])
    valid = smk.survival_valid_to_Gy(ions[e])
    rows[e] = (lq_s, lq_m, valid, z)
    print(
        f"{e:>10.0f} {lets[e]:>8.1f} {z.z_star_1D_Gy:>8.2f} {z.z_1D_Gy:>9.2f} "
        f"{z.z_nucleus_Gy:>7.3f} {lq_s.alpha:>7.3f} {lq_s.beta:>9.4f} "
        f"{lq_m.beta:>10.4f} {valid:>9.1f}"
    )

d_ref_10 = dose_for_survival(0.1, smk.alpha_ref, smk.beta_ref)
print(f"\nreference (photon) dose at S = 0.1: {d_ref_10:.2f} Gy")

# %% [markdown]
# ## Panels

# %%
fig, axes = plt.subplots(1, 4, figsize=(20.0, 4.6))

# --- 1. survival curves: SMK vs its LQ core (z_n = 0 switches gamma off), at 10 MeV/u
ax = axes[0]
doses = np.linspace(0.0, 12.0, 200)
e_show = 10.0
lq_s, lq_m, valid, z = rows[e_show]
core = smk.survival_from_specific_energies(
    doses, z_star_1D_Gy=z.z_star_1D_Gy, z_1D_Gy=z.z_1D_Gy, z_nucleus_Gy=0.0
)
ax.semilogy(doses, core, "--", color="0.45", label="SMK LQ core (γ = 0)")
ax.semilogy(doses, smk.survival(ions[e_show], doses), "-", color="C3", lw=2, label="SMK, eq. (24)")
ax.semilogy(doses, mmkm.survival(ions[e_show], doses), "-", color="C0", label="mMKM (LQ)")
ax.axvline(valid, color="C3", ls=":", lw=1.2)
ax.text(valid, 2e-5, f"  γD = 1\n  ({valid:.1f} Gy)", color="C3", fontsize=8, va="bottom")
ax.set(
    xlabel="dose [Gy]",
    ylabel="surviving fraction",
    title=f"$^{{12}}$C at {e_show:.0f} MeV/u ({lets[e_show]:.0f} keV/µm)",
    ylim=(1e-5, 1.2),
)
ax.legend(fontsize=8)

# --- 2. beta vs LET --------------------------------------------------------
ax = axes[1]
let_axis = [lets[e] for e in ENERGIES]
ax.plot(let_axis, [rows[e][0].beta for e in ENERGIES], "o-", color="C3", label="SMK")
ax.plot(let_axis, [rows[e][1].beta for e in ENERGIES], "s-", color="C0", label="mMKM")
ax.axhline(smk.beta_ref, color="0.6", ls="--", lw=1, label=r"$\beta_{ref}$")
ax.set(
    xscale="log",
    xlabel="LET [keV/µm]",
    ylabel=r"$\beta$ [Gy$^{-2}$]",
    title=r"SMK makes $\beta$ quality-dependent",
)
ax.legend(fontsize=8)

# --- 3. the expansion validity ceiling vs LET ------------------------------
ax = axes[2]
ax.plot(
    let_axis, [rows[e][2] for e in ENERGIES], "o-", color="C3", label=r"dose where $\gamma D = 1$"
)
ax.axhline(d_ref_10, color="0.4", ls="--", lw=1.2, label=f"$D_{{ref}}(S=0.1)$ = {d_ref_10:.1f} Gy")
ax.fill_between([min(let_axis), max(let_axis)], 0, d_ref_10, color="C3", alpha=0.08)
ax.set(
    xscale="log",
    yscale="log",
    xlabel="LET [keV/µm]",
    ylabel="dose [Gy]",
    title="where eq. (4) stops being a\ncontrolled expansion",
)
ax.legend(fontsize=8)
ax.text(
    0.5,
    0.06,
    "below the line → outside the expansion at $S=0.1$",
    transform=ax.transAxes,
    ha="center",
    fontsize=8,
    color="C3",
)

# --- 4. what the site geometry costs ---------------------------------------
ax = axes[3]
cyl = [smk.specific_energies(ions[e]).z_star_1D_Gy for e in ENERGIES]
sph = [smk_sphere.specific_energies(ions[e]).z_star_1D_Gy for e in ENERGIES]
ax.plot(let_axis, cyl, "o-", color="C3", label="cylinder (paper, default)")
ax.plot(let_axis, sph, "s--", color="C7", label="sphere")
for x, c, sv in zip(let_axis, cyl, sph, strict=True):
    ax.annotate(f"{sv / c - 1:+.0%}", (x, sv), fontsize=8, ha="center", va="bottom", color="0.35")
ax.set(
    xscale="log",
    yscale="log",
    xlabel="LET [keV/µm]",
    ylabel=r"$z^*_{1D}$ [Gy]",
    title="site geometry is not cosmetic",
)
ax.legend(fontsize=8)

fig.tight_layout()
print()
save_figure(fig, "smk_nucleus_stochastics.png", directory=FIGURES, dpi=150)
