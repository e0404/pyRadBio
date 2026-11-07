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
# # OSMK: hypoxia, and why high LET beats it
#
# The **oxygen-effect-incorporated SMK** extends the SMK to oxygen-deficient
# cells. Two published formulations, both shipped, both here:
#
# * **OSMK 2021** — Inaniwa, Kanematsu, Shinoto, Koto & Yamada,
#   doi:10.1088/1361-6560/ac29cc
# * **OSMK 2023** — Inaniwa & Kanematsu, doi:10.1093/jrr/rrad049, the
#   "event-by-event" revision
#
# The mechanism is one substitution. Hypoxic cells resist radiation because some
# initial DNA damage is *chemically repaired* (~1 µs) rather than fixed by
# molecular oxygen, so the **sub-lethal** lesion rate drops by a relative
# radioresistance `R(pO₂) ≥ 1` while the **lethal** rate — clustered damage,
# chemically irreparable — does not. Since SMK's `α_S ∝ k_d` and `β₀ ∝ k_d²`:
#
# ```
# α'(pO₂) = α_L + α_S/R + z*_d,D β₀ /R²
# β'(pO₂) = (β₀/R²) (z*_d,D / z_d,D)
# ```
#
# and survival is *unchanged* SMK eq. (24) with that pair substituted in. So OSMK
# is `StochasticMKM` plus `R`, which is exactly how it is implemented.
#
# What the two papers disagree about is **why the OER falls with LET**:
#
# 1. **2021** makes `Rmax` a function of the beam, `Rmax(z̄_d,D)` — dense tracks
#    make chemical repair ineffective. The microdosimetry is untouched by pO₂.
# 2. **2023** makes `Rmax` a cell constant and instead shrinks the *site*:
#    `r_d' = r_d/f_rd(pO₂)`, `z₀' = z₀·f_z0(pO₂)`, then rescores `z̄_d,D` there.
#    Under anoxia the HSG domain goes 0.29 → 0.097 µm.
#
# The 2023 paper's own verdict is that this buys ≤ 1.9 %pt over an SOBP target,
# so neither supersedes the other. Panel 4 shows how little they differ here.
#
# **Two traps**, both pinned by tests (`docs/references/OSMK_Inaniwa2021_2023.md`):
#
# * The three MKM-family fits of HSG are **independent refits**. `z₀` is 66.0 Gy
#   (SMK 2018) / 88.0 (OSMK 2021) / 60.0 (OSMK 2023); `r_d` is 0.28 / 0.23 / 0.29.
#   Never build an OSMK from SMK parameters plus oxygen parameters.
# * The experimental "aerobic" condition of 160 mmHg is **not** `R = 1` — it is
#   `R ≈ 1.014`. `pO2_mmHg` defaults to 160 (the papers' reference); pass
#   `math.inf` for the strict SMK limit.
#
# No external data, no `mkm` extra. The paper-figure reconstruction (figs. 5-7)
# is `validation/osmk_inaniwa2021.py`.
#
# This file is a [jupytext](https://jupytext.readthedocs.io) percent script.

# %%
import math

import matplotlib.pyplot as plt
import numpy as np
from matplotlib.ticker import ScalarFormatter

from pyradbio.core.oxygen import AEROBIC_PO2_MMHG
from pyradbio.core.radiation import Ion
from pyradbio.models.mkm.osmk import OSMK2021, OSMK2023
from pyradbio.plotting import figures_dir, save_figure
from pyradbio.track.stopping_power import SimpleBetheStoppingPower

FIGURES = figures_dir(globals().get("__file__"))

osmk21 = OSMK2021.hsg_inaniwa2021()  # table 1: r_d 0.23, z0 88.0, z_R 28.0, R_m 2.9
osmk23 = OSMK2023.hsg_inaniwa2023()  # table 1: r_d 0.29, z0 60.0, Rmax 4.46
sp = SimpleBetheStoppingPower()

# He / C / Ne spanning the 18.5-654 keV/um range the models were fitted to
# (Furusawa et al. 2000, 2012), LET-ordered because panels 2 and 4 plot against LET.
HE, C, NE = ((2, 4, (10.0, 3.0)), (6, 12, (60.0, 30.0, 12.0, 6.0)), (10, 20, (30.0, 10.0, 4.0)))
LET_IONS = sorted((Ion(z, a, e) for z, a, es in (HE, C, NE) for e in es), key=sp.let_keV_um)
LETS = np.array([sp.let_keV_um(i) for i in LET_IONS])
SPECIES = np.array([i.Z for i in LET_IONS])
# The anoxic OER of every ion, computed once: the table below prints it, panel 2 plots it.
OER21 = np.array([osmk21.oer_at_survival(i, pO2_mmHg=0.0) for i in LET_IONS])
OER23 = np.array([osmk23.oer_at_survival(i, pO2_mmHg=0.0) for i in LET_IONS])

# %% [markdown]
# ## The numbers behind the panels

# %%
carbon = Ion(6, 12, 30.0)  # ~67 keV/um
print(f"carbon {carbon.energy_MeV_u:g} MeV/u -> LET {sp.let_keV_um(carbon):.1f} keV/um\n")

print("R(pO2) and the LQ pair, OSMK 2023 (HSG):")
print(f"  {'pO2':>8} {'R':>7} {'alpha':>8} {'beta':>8} {'D_10':>7}")
for p in (0.0, 1.0, 5.0, 20.0, AEROBIC_PO2_MMHG, math.inf):
    z = osmk23.specific_energies_at(carbon, p)
    r = osmk23.relative_radioresistance(p, z_1D_Gy=z.z_1D_Gy)
    lq = osmk23.alpha_beta(carbon, pO2_mmHg=p)
    d10 = osmk23.dose_for_survival(carbon, 0.1, pO2_mmHg=p)
    print(f"  {p:>8g} {r:7.3f} {lq.alpha:8.4f} {lq.beta:8.4f} {d10:7.3f}")

print("\nOSMK 2023 hypoxic site geometry (HSG, r_d = 0.29 um, z0 = 60 Gy):")
for p in (0.0, 1.0, 5.0, 20.0, AEROBIC_PO2_MMHG):
    r_d, z0 = osmk23.hypoxic_geometry(p)
    print(f"  pO2 {p:>6g} mmHg -> r_d' = {r_d:.4f} um, z0' = {z0:7.1f} Gy")

print("\nAnoxic OER at 10% survival (vs 160 mmHg):")
print(f"  {'ion':>6} {'LET':>8} {'OER 2021':>9} {'OER 2023':>9}")
for ion, let, o21, o23 in zip(LET_IONS, LETS, OER21, OER23, strict=True):
    print(f"  Z={ion.Z:<4d} {let:8.1f} {o21:9.2f} {o23:9.2f}")

# %% [markdown]
# ## Figure


# %%
def _let_ticks(ax: plt.Axes) -> None:
    """Readable decade ticks on a log LET axis (the default minor labels collide)."""
    ax.set_xticks([20, 50, 100, 200, 500, 1000])
    ax.xaxis.set_major_formatter(ScalarFormatter())
    ax.xaxis.set_minor_formatter(plt.NullFormatter())


fig, axes = plt.subplots(2, 2, figsize=(11, 8.5))
axes = axes.ravel()
doses = np.linspace(0.0, 8.0, 200)
PRESSURES = [0.0, 1.0, 5.0, 20.0, AEROBIC_PO2_MMHG]

# --- 1. survival vs dose at graded pO2 --------------------------------------
ax = axes[0]
for p, c in zip(PRESSURES, plt.cm.viridis(np.linspace(0, 0.85, len(PRESSURES))), strict=True):
    ax.plot(doses, osmk23.survival(carbon, doses, pO2_mmHg=p), color=c, label=f"{p:g} mmHg")
ax.axhline(0.1, color="0.7", lw=0.8, ls=":")
ax.set(
    yscale="log",
    ylim=(1e-4, 1.2),
    xlabel="dose [Gy]",
    ylabel="survival fraction",
    title=f"OSMK 2023, HSG, $^{{12}}$C {carbon.energy_MeV_u:g} MeV/u\n"
    f"({sp.let_keV_um(carbon):.0f} keV/µm)",
)
ax.legend(fontsize=8, title="pO$_2$", title_fontsize=8)

# --- 2. the OER-vs-LET curve (the papers' figs. 5-6) ------------------------
ax = axes[1]
markers = {2: "o", 6: "s", 10: "^"}
for z, marker in markers.items():
    sel = SPECIES == z
    ax.plot(LETS[sel], OER21[sel], marker + "-", color="C0", label=f"OSMK 2021, Z={z}")
    ax.plot(LETS[sel], OER23[sel], marker + "--", color="C1", label=f"OSMK 2023, Z={z}")
ax.axhline(1.0, color="0.7", lw=0.8)
ax.axvline(200.0, color="0.85", lw=0.8, ls=":")
ax.set(
    xscale="log",
    xlabel="LET [keV/µm]",
    ylabel="OER at 10% survival (anoxic / 160 mmHg)",
    ylim=(0.9, 4.0),
    title="the OER falls from ~3 to unity above ~200 keV/µm\n"
    "— and depends slightly on particle type at equal LET",
)
ax.legend(fontsize=7, ncol=2)
_let_ticks(ax)

# --- 3. OER vs pO2 at three LETs -------------------------------------------
ax = axes[2]
po2_axis = np.geomspace(0.05, 200.0, 60)
for ion, c in zip(
    [Ion(2, 4, 10.0), Ion(6, 12, 30.0), Ion(6, 12, 6.0)], ("C2", "C3", "C4"), strict=True
):
    ax.plot(
        po2_axis,
        [osmk23.oer_at_survival(ion, pO2_mmHg=p) for p in po2_axis],
        color=c,
        label=f"Z={ion.Z}, {sp.let_keV_um(ion):.0f} keV/µm",
    )
ax.axvline(3.0, color="0.85", lw=0.8, ls=":")  # K = 3 mmHg
ax.axhline(1.0, color="0.7", lw=0.8)
ax.set(
    xscale="log",
    xlabel="pO$_2$ [mmHg]",
    ylabel="OER at 10% survival",
    title="OSMK 2023: the OER is spent by ~20 mmHg\n(dotted: the assumed K = 3 mmHg)",
)
ax.legend(fontsize=8)

# --- 4. do the two formulations actually disagree? -------------------------
ax = axes[3]
for p, c in zip((0.0, 1.0, 5.0), ("C0", "C1", "C2"), strict=True):
    ratio = np.array(
        [
            float(osmk23.survival(i, 4.0, pO2_mmHg=p) / osmk21.survival(i, 4.0, pO2_mmHg=p))
            for i in LET_IONS
        ]
    )
    for z, marker in markers.items():  # per species, as in panel 2 -- they interleave in LET
        sel = SPECIES == z
        ax.plot(LETS[sel], ratio[sel], marker + "-", color=c, label=f"{p:g} mmHg, Z={z}")
ax.axhline(1.0, color="0.7", lw=0.8)
ax.set(
    xscale="log",
    xlabel="LET [keV/µm]",
    ylabel="S(4 Gy) ratio, 2023 / 2021",
    title="2023 vs 2021 at 4 Gy — same cell line, different fits\n"
    "(so this is refit + method, not the method alone)",
)
ax.legend(fontsize=7, ncol=3)
_let_ticks(ax)

fig.tight_layout()
print()
save_figure(fig, "osmk_oxygen_effect.png", directory=FIGURES, dpi=150)
