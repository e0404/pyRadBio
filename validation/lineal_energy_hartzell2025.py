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
# # Lineal-energy spectra: pyRadBio engines vs the Hartzell 2025 AMF / nBio figures
#
# Validates pyRadBio's single-event lineal-energy engines against **several ion
# species** using dose-weighted spectra `y·d(y)` digitized from Hartzell et al.
# 2025 (Phys. Med. Biol. 70 105010,
# [doi:10.1088/1361-6560/adcfec](https://doi.org/10.1088/1361-6560/adcfec)). The
# proton-only companion is `lineal_energy_decunha.py` (DeCunha 2025 Geant4-DNA
# library, external data).
#
# The paper benchmarks an **analytical microdosimetric function (AMF)** — a
# parameterization of PHITS track-structure data, the same *kind* of analytic
# model pyRadBio implements — against **TOPAS-nBio** Geant4-DNA track-structure
# simulations (the Monte-Carlo reference). Two figures are bundled as digitized
# curves (`pyradbio.benchmark.hartzell2025`), so no external data is needed:
#
# * **Figure 4** — `1H 4He 7Li 12C 16O 20Ne` at 1, 10, 100 MeV/u in a
#   **0.3 µm radius** sphere (→ `SphericalSite(diameter_um=0.6)`).
# * **Figure 7** — 1 MeV/u carbon across five site radii, 0.0015–0.5 µm.
#
# Figure 4 is hand-digitized (WebPlotDigitizer); Figure 7 is a raster split (see
# the module docstring) — neither is an exact table, but the primary peaks are
# faithful. What to look for: pyRadBio's **analytic** (restricted straggling) and
# **amorphous** (Kiefer-Chatterjee `D(r)` + delta-ray Monte Carlo) engines place
# the crosser peak close to the AMF (the analytic reference) and a little above
# the nBio track-structure peak, tracking its shift with ion, energy and site
# size.
#
# This file is a [jupytext](https://jupytext.readthedocs.io) percent script.
# `PYRADBIO_QUICK=1` runs the delta-ray Monte Carlo with 20k instead of 120k
# samples.

# %%
import matplotlib.pyplot as plt
from _common import pick, proxy_legend, quick_mode

from pyradbio import Ion
from pyradbio.benchmark.hartzell2025 import get_amf_curve, iter_amf_curves
from pyradbio.micro import (
    AmorphousTrackEngine,
    DeltaRayMonteCarlo,
    LinealEnergyCalculator,
    SphericalSite,
)
from pyradbio.plotting import figures_dir, save_figure
from pyradbio.track.kiefer_chatterjee import KieferChatterjeeRadialDose
from pyradbio.track.stopping_power import SimpleBetheStoppingPower

FIGURES = figures_dir(globals().get("__file__"))
N_DELTA_SAMPLES = pick(120_000, 20_000)  # delta-ray Monte-Carlo samples per spectrum
if quick_mode():
    print(f"QUICK MODE: {N_DELTA_SAMPLES} delta-ray samples (PYRADBIO_QUICK=1)")

# %% [markdown]
# The engines use the analytic Bethe water stopping power, pinned explicitly (no
# external data; it spans every ion here). The amorphous engine uses the
# Kiefer-Chatterjee radial dose with the faithful stochastic delta-ray layer
# (soft core + partial deposition), as in the DeCunha comparison.


# %%
def build_engines(radius_um: float) -> tuple[LinealEnergyCalculator, AmorphousTrackEngine]:
    """An analytic and an amorphous engine on a sphere of the given radius."""
    site = SphericalSite(diameter_um=2.0 * radius_um)
    sp = SimpleBetheStoppingPower()
    analytic = LinealEnergyCalculator(stopping_power=sp, site=site, restricted=True)
    amorphous = AmorphousTrackEngine(
        KieferChatterjeeRadialDose(sp),
        site,
        stopping_power=sp,
        stochastic=DeltaRayMonteCarlo(n_samples=N_DELTA_SAMPLES, seed=0),
    )
    return analytic, amorphous


# %% [markdown]
# ## Figure 4 — six ions across 1/10/100 MeV/u, 0.3 µm radius
#
# The multi-ion breadth (1H-20Ne) the protons-only DeCunha library cannot give.
# Colour = energy; per ion panel: pyRadBio's analytic engine (solid),
# hand-digitized AMF (dashed) and nBio (dotted). As the ion slows (100 → 1 MeV/u)
# the peak marches up in lineal energy by ~two decades. The analytic engine
# tracks the **AMF** peak closely at every ion and energy (both place the crosser
# peak near ~9/8·LET); the **nBio** peak sits a little lower, the stopping-power
# offset the paper attributes to nBio's LET running below the ICRU value. (An
# auto-digitization under-reads the sharp light-ion peaks, which makes the model
# look like it overshoots — the by-hand curves avoid that artifact.)

# %%
ions_fig4 = [
    ("hydrogen", "H", 1, 1),
    ("helium", "He", 2, 4),
    ("lithium", "Li", 3, 7),
    ("carbon", "C", 6, 12),
    ("oxygen", "O", 8, 16),
    ("neon", "Ne", 10, 20),
]
energies_fig4 = [1.0, 10.0, 100.0]
ECOL = {1.0: "C3", 10.0: "C0", 100.0: "C2"}  # colour == energy
analytic03, amorphous03 = build_engines(0.3)

fig, axes = plt.subplots(2, 3, figsize=(14.0, 7.6), sharex=True)
peak_rows = []
for ax, (name, symbol, Z, A) in zip(axes.ravel(), ions_fig4, strict=True):
    for E in energies_fig4:
        amf = get_amf_curve(figure="fig4", ion=name, method="AMF", energy_MeV_u=E)
        nbio = get_amf_curve(figure="fig4", ion=name, method="nBio", energy_MeV_u=E)
        ana = analytic03.spectrum(Ion(Z=Z, A=A, energy_MeV_u=E))
        ax.plot(ana.y_keV_um, ana.yd, color=ECOL[E], lw=1.9)  # pyRadBio == solid
        ax.plot(amf.y, amf.yd, color=ECOL[E], lw=1.3, ls="--")
        ax.plot(nbio.y, nbio.yd, color=ECOL[E], lw=1.3, ls=":")
        peak_rows.append((name, E, amf.peak_y_keV_um, nbio.peak_y_keV_um, ana.peak_y_keV_um))
    ax.set_xscale("log")
    ax.set_xlim(1.0e-1, 2.0e4)
    ax.set_title(f"{A}{symbol}")
    ax.grid(True, which="both", ls=":", lw=0.4)
axes[0, 0].set_ylabel(r"$y\,d(y)$")
axes[1, 0].set_ylabel(r"$y\,d(y)$")
for ax in axes[1]:
    ax.set_xlabel("y [keV/um]")

# Two small legends: colour == energy, line style == data source.
legend_kw = {"frameon": False, "fontsize": 8, "loc": "upper left"}
proxy_legend(
    axes[0, 0],
    [{"color": ECOL[E], "lw": 2.2, "label": f"{E:.0f} MeV/u"} for E in energies_fig4],
    **legend_kw,
)
proxy_legend(
    axes[0, 1],
    [
        {"color": "0.3", "lw": 1.9, "ls": "-", "label": "pyRadBio analytic"},
        {"color": "0.3", "lw": 1.3, "ls": "--", "label": "AMF (digitized)"},
        {"color": "0.3", "lw": 1.3, "ls": ":", "label": "nBio (digitized)"},
    ],
    **legend_kw,
)
fig.suptitle(
    "Dose-weighted lineal-energy spectra, 0.3 um radius — pyRadBio vs Hartzell 2025 Fig. 4"
)
fig.tight_layout()
save_figure(fig, "lineal_energy_hartzell2025_ions.png", directory=FIGURES, dpi=140)

print(f"\n{'ion':>8} {'E':>5} {'AMF':>7} {'nBio':>7} {'analytic':>9}   peak y [keV/um]")
for name, E, a, n, an in peak_rows:
    print(f"{name:>8} {E:5.0f} {a:7.0f} {n:7.0f} {an:9.0f}")

# %% [markdown]
# ## Figure 4 — carbon energy dependence
#
# One ion, three energies: as the carbon slows (100 → 1 MeV/u) the LET rises and
# the peak marches up in lineal energy by nearly two decades.

# %%
fig, axes = plt.subplots(1, 3, figsize=(13.5, 4.0), sharey=True)
for ax, E in zip(axes, [1.0, 10.0, 100.0], strict=True):
    ion = Ion(Z=6, A=12, energy_MeV_u=E)
    amf = get_amf_curve(figure="fig4", ion="carbon", method="AMF", energy_MeV_u=E)
    nbio = get_amf_curve(figure="fig4", ion="carbon", method="nBio", energy_MeV_u=E)
    ana = analytic03.spectrum(ion)
    amo = amorphous03.single_event_spectrum(ion)
    ax.plot(ana.y_keV_um, ana.yd, color="C3", lw=1.9, label="pyRadBio analytic")
    ax.plot(amo.y_keV_um, amo.yd, color="C1", lw=1.9, label="pyRadBio amorphous")
    ax.plot(amf.y, amf.yd, color="C0", ls="--", lw=1.6, label="AMF (digitized)")
    ax.plot(nbio.y, nbio.yd, color="C2", ls=":", lw=1.8, label="nBio (digitized)")
    ax.set_xscale("log")
    ax.set_xlim(1.0e-1, 5.0e3)
    ax.set_xlabel("y [keV/um]")
    ax.set_title(f"12C  {E:.0f} MeV/u")
    ax.grid(True, which="both", ls=":", lw=0.4)
axes[0].set_ylabel(r"$y\,d(y)$")
axes[0].legend(frameon=False, fontsize=8)
fig.suptitle("Carbon lineal-energy spectra, 0.3 um radius — pyRadBio vs Hartzell 2025 Fig. 4")
fig.tight_layout()
save_figure(fig, "lineal_energy_hartzell2025_carbon_energy.png", directory=FIGURES, dpi=140)

# %% [markdown]
# ## Figure 7 — carbon site-size dependence
#
# 1 MeV/u carbon in shrinking spheres (0.5 → 0.0015 µm radius). As the site
# shrinks below the delta-ray range the crosser peak broadens and a low-`y`
# delta-ray shoulder emerges (clearest in the nBio curve at 1.5 nm). The engines
# use the same site radius as each panel.

# %%
radii = [0.5, 0.3, 0.05, 0.015, 0.0015]
fig, axes = plt.subplots(1, len(radii), figsize=(3.0 * len(radii), 3.8), sharex=True)
for ax, r in zip(axes, radii, strict=True):
    ion = Ion(Z=6, A=12, energy_MeV_u=1.0)
    amf = get_amf_curve(figure="fig7", ion="carbon", method="AMF", site_radius_um=r)
    nbio = get_amf_curve(figure="fig7", ion="carbon", method="nBio", site_radius_um=r)
    analytic, amorphous = build_engines(r)
    ana, amo = analytic.spectrum(ion), amorphous.single_event_spectrum(ion)
    ax.plot(ana.y_keV_um, ana.yd, color="C3", lw=1.8, label="pyRadBio analytic")
    ax.plot(amo.y_keV_um, amo.yd, color="C1", lw=1.8, label="pyRadBio amorphous")
    ax.plot(amf.y, amf.yd, color="C0", ls="--", lw=1.5, label="AMF (digitized)")
    ax.plot(nbio.y, nbio.yd, color="C2", ls=":", lw=1.6, label="nBio (digitized)")
    ax.set_xscale("log")
    ax.set_xlim(1.0e-1, 1.0e4)
    ax.set_xlabel("y [keV/um]")
    ax.set_title(f"rD = {r} um")
    ax.grid(True, which="both", ls=":", lw=0.4)
axes[0].set_ylabel(r"$y\,d(y)$")
axes[0].legend(frameon=False, fontsize=7)
fig.suptitle("1 MeV/u carbon vs site radius — pyRadBio vs Hartzell 2025 Fig. 7")
fig.tight_layout()
save_figure(fig, "lineal_energy_hartzell2025_site_size.png", directory=FIGURES, dpi=140)

# %% [markdown]
# `iter_amf_curves()` filters the bundled digitized set (by figure, ion, method,
# energy, site radius).

# %%
print(
    f"\n{len(iter_amf_curves())} digitized curves bundled "
    f"({len(iter_amf_curves(figure='fig4'))} from Fig. 4, "
    f"{len(iter_amf_curves(figure='fig7'))} from Fig. 7)"
)
