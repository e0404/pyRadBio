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
# # Amorphous-track radial dose profiles
#
# Plots the local dose $D(r)$ around an ion track for every track-structure model
# in pyRadBio, side by side:
#
# * **LEM I** — a wide fixed 10 nm core (Scholz & Kraft 1996), so the same LET is
#   spread over a large plateau: the *lowest* core dose of the set.
# * **LEM II** — a tiny 0.3 nm core rounded by a 4 nm radical-diffusion Gaussian
#   (Elsasser & Scholz 2007): the *highest* LEM core dose.
# * **LEM III** — a velocity-dependent core $r_c = \beta\, r_{c0}$, $r_{c0}=40$ nm
#   (Elsasser et al. 2008), plus diffusion; sits between LEM I and II.
# * **Kiefer-Chatterjee** — the MKM/SMK track (Chatterjee & Schaefer 1976): the
#   smallest core $r_c = 11.6\,\beta$ nm and hence by far the highest peak dose,
#   with a separately-normalised $K_p=\kappa(Z_\mathrm{eff}/\beta)^2$ penumbra.
#
# All four share the $\sim 1/r^2$ penumbra and are normalised so the radial energy
# integral reproduces the LET; the models differ almost entirely in the **core**
# (small $r$). Only the outermost tail falls below a typical transition dose
# $D_t$, which is why the ion initial slope is dominated by the linear-response
# core.
#
# This file is a [jupytext](https://jupytext.readthedocs.io) percent script: run
# it directly with `python`, or pair/open it as a notebook (see
# `examples/README.md`).

# %%
import matplotlib.pyplot as plt
import numpy as np

from pyradbio import Ion
from pyradbio.plotting import figures_dir, save_figure
from pyradbio.track.base import RadialDoseModel
from pyradbio.track.kiefer_chatterjee import KieferChatterjeeRadialDose
from pyradbio.track.scholz_kraft import ScholzKraftRadialDose
from pyradbio.track.stopping_power import default_stopping_power, describe_stopping_power

FIGURES = figures_dir(globals().get("__file__"))

# %% [markdown]
# One stopping power feeds every model (so they all see the same LET): the
# library default, `default_stopping_power()`.

# %%
sp = default_stopping_power()
SOURCE_LABEL = describe_stopping_power(sp)
print(f"stopping power: {SOURCE_LABEL}")

# (label, model, colour, linestyle)
MODELS: list[tuple[str, RadialDoseModel, str, str]] = [
    ("LEM I", ScholzKraftRadialDose.lem1(sp), "C0", "-"),
    ("LEM II", ScholzKraftRadialDose.lem2(sp), "C1", "-"),
    ("LEM III", ScholzKraftRadialDose.lem3(sp), "C2", "-"),
    ("Kiefer-Chatterjee", KieferChatterjeeRadialDose(stopping_power=sp), "C3", "--"),
]

# %% [markdown]
# Normalisation check: every model fixes its core amplitude by energy
# conservation, so integrating the radial energy back must reproduce the LET.

# %%
for label, model, _c, _ls in MODELS:
    ion = Ion(6, 12, 43.0)
    recovered = model.radial_integral_LET_keV_um(ion, nodes=2048)
    print(
        f"  {label:>18s}: integral LET = {recovered:7.2f} keV/um  (target {sp.let_keV_um(ion):.2f})"
    )

# %% [markdown]
# ## $D(r)$ across representative ions and energies
#
# Two protons (the low-energy entrance channel, where the LEM I shoulder is most
# sensitive to the core) and two carbon ions (Bragg-peak and plateau LET).

# %%
CASES = [
    (1, 1, 1.0, "proton, 1 MeV/u"),
    (1, 1, 4.0, "proton, 4 MeV/u"),
    (6, 12, 10.0, "carbon, 10 MeV/u"),
    (6, 12, 100.0, "carbon, 100 MeV/u"),
]

fig, axes = plt.subplots(2, 2, figsize=(11, 8.5))
dmax = 0.0
for ax, (z, a, energy, title) in zip(axes.ravel(), CASES, strict=True):
    ion = Ion(z, a, energy)
    for label, model, colour, ls in MODELS:
        r_min = model.r_min_um(ion)
        r_max = model.r_max_um(ion)
        radii = np.logspace(np.log10(min(r_min, 1.0e-4) * 0.3), np.log10(r_max), 500)
        dose = np.array([model.local_dose_Gy(float(r), ion) for r in radii])
        dose = np.where(dose > 0.0, dose, np.nan)
        dmax = max(dmax, np.nanmax(dose))
        ax.loglog(
            radii * 1.0e3,  # nm on the x axis
            dose,
            color=colour,
            ls=ls,
            lw=1.8,
            label=f"{label}  ($r_c$={r_min * 1e3:.2g} nm, $r_\\max$={r_max * 1e3:.0f} nm)",
        )
    ax.axhline(30.0, color="0.6", ls=":", lw=1.0)  # a typical CHO transition dose
    ax.set_title(f"{title}   LET = {sp.let_keV_um(ion):.1f} keV/um", fontsize=10)
    ax.set_xlabel("radius from track [nm]")
    ax.set_ylabel("local dose $D(r)$ [Gy]")
    ax.grid(True, which="both", ls=":", lw=0.4)
    ax.legend(fontsize=7, loc="lower left")

for ax in axes.ravel():
    ax.set_ylim(1.0e-1, 3.0 * dmax)
axes.ravel()[0].text(0.03, 40.0, "$D_t\\approx30$ Gy (CHO)", fontsize=7, color="0.5")
fig.suptitle(
    f"Amorphous-track radial dose: LEM I / II / III vs Kiefer-Chatterjee\n"
    f"stopping power: {SOURCE_LABEL}",
    fontsize=12,
)
fig.tight_layout()

save_figure(fig, "radial_dose_profiles.png", directory=FIGURES, dpi=130)

# %% [markdown]
# The models agree in the $1/r^2$ penumbra and cross within ~5-10 nm; they differ
# almost entirely in the core. LEM I's uniquely wide, flat, low core is the same
# knob that controls the low-energy proton initial-slope shoulder — a peakier core
# (LEM II/III, Kiefer-Chatterjee) concentrates the dose and shifts where the
# single-track lethality saturates into overkill.
