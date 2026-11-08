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
# # Proton lineal-energy spectra: pyRadBio engines vs the DeCunha 2025 library
#
# Overlays pyRadBio's two single-event engines -- **analytic** (Xapsos-Kellerer
# straggling) and **amorphous** (Kiefer-Chatterjee radial dose + delta-ray Monte
# Carlo) -- on the **Geant4-DNA proton library of DeCunha et al. 2025** (Med.
# Phys. 52(5), 3471-3480,
# [doi:10.1002/mp.17561](https://doi.org/10.1002/mp.17561)). That library scores
# lineal energy `y` in a **1 µm diameter water sphere**, so both models use
# `SphericalSite(diameter_um=1.0)`. It is protons only; heavier ions are covered
# by `lineal_energy_hartzell2025.py` (bundled digitized curves).
#
# **Data.** The library is large external data and is not bundled: set
# `PYRADBIO_LINEAL_ENERGY_DECUNHA_DIR` (see `.env.example`) to the folder holding
# its scoring-geometry subfolders (`1mm_linear`, `1mm_logarithmic`, `5um_linear`,
# `5um_logarithmic`). Without it the script stops with a message.
#
# **Geometry.** We compare against the **`5um`** scoring geometry (a 1 µm sphere
# inside a 5 µm cube; `GEOMETRY` below), not the `1mm` one, on purpose. In the
# 1 mm cube a therapeutic proton slows down and even *stops* inside the scored
# volume (a 5.5 MeV proton has a ~0.43 mm range), so the sphere sees a degraded /
# Bragg-peak spectrum with a heavy stopping-proton tail that a single
# monoenergetic track model was never meant to reproduce. The 5 µm cube is thin
# compared with every proton range here, so the beam stays monoenergetic across
# it and the comparison isolates the *track-structure* physics the engines model.
#
# **What to look for** -- two approximate models vs the MC truth. The
# **analytic** engine reproduces `y_F` and the crosser peak at intermediate LET
# but omits the low-`y` delta-ray tail. Its dose-mean `y_D` otherwise runs away at
# low LET (a thin 1 µm chord has an unbounded Bohr-straggling variance), so we
# enable `restricted=True`: delta-rays whose range exceeds the site escape it, so
# both the mean deposit (restricted LET, pulling `y_F` below LET) and the
# straggling variance (capping the tail) drop to their restricted values,
# collapsing `y_F` and `y_D` back onto the library. The **amorphous** engine
# supplies a physical delta-ray tail and restricted-LET behaviour and lands its
# moments in the right range, though the simple amorphous radial dose still keeps
# it from matching the MC exactly. Neither replaces the tabulated library — that
# is the reference here.
#
# This file is a [jupytext](https://jupytext.readthedocs.io) percent script.
# `PYRADBIO_QUICK=1` runs the delta-ray Monte Carlo with 20k instead of 300k
# samples.

# %%
import matplotlib.pyplot as plt
from _common import pick, quick_mode, require_data

from pyradbio import Ion
from pyradbio.datasets import LINEAL_ENERGY_DECUNHA, dataset_dir
from pyradbio.micro import (
    AmorphousTrackEngine,
    AnalyticSpectrumEngine,
    DeCunhaLinealEnergyLibrary,
    DeltaRayMonteCarlo,
    LinealEnergyCalculator,
    SphericalSite,
)
from pyradbio.plotting import figures_dir, save_figure
from pyradbio.track.kiefer_chatterjee import KieferChatterjeeRadialDose
from pyradbio.track.stopping_power import default_stopping_power, describe_stopping_power

FIGURES = figures_dir(globals().get("__file__"))

# DeCunha scoring geometry: "<cube>_<binning>", cube 1mm|5um, binning linear|logarithmic.
GEOMETRY = "5um_linear"
N_DELTA_SAMPLES = pick(300_000, 20_000)  # delta-ray Monte-Carlo samples per spectrum
if quick_mode():
    print(f"QUICK MODE: {N_DELTA_SAMPLES} delta-ray samples (PYRADBIO_QUICK=1)")


def load_library(geometry: str) -> DeCunhaLinealEnergyLibrary:
    """The library's ``geometry`` subfolder under ``$PYRADBIO_LINEAL_ENERGY_DECUNHA_DIR``."""
    folder = dataset_dir(LINEAL_ENERGY_DECUNHA) / geometry
    if not folder.is_dir():
        raise FileNotFoundError(f"no {geometry!r} geometry folder in {folder.parent}")
    return DeCunhaLinealEnergyLibrary.from_directory(folder)


library = require_data(load_library, GEOMETRY, dataset=LINEAL_ENERGY_DECUNHA)
print(f"library: {library.directory}")
print(f"  {len(library.energies_MeV)} energies, environment = {library.environment}")

# %% [markdown]
# Both engines on the same 1 µm sphere with the library's default stopping power.
# The analytic engine leaves `include_delta_tail` off on purpose: its delta-ray
# toucher tail is a deterministic-mean approximation tuned for the MKM regime
# (sub-µm domains, high-LET heavy ions). For these low-LET protons in a 1 µm
# sphere delta-ray events are rare and discrete, not a smooth mean, and the tail
# would bolt a spurious low-`y` plateau onto the crosser peak.
#
# The amorphous engine's `DeltaRayMonteCarlo` defaults are the faithful Vavilov
# configuration -- `partial_deposition` (each delta-ray deposits only its in-site
# path length) plus `soft_core` (a Gaussian near-track "soft" core that sharpens
# the crosser peak, with the soft/hard boundary from the radial-dose model's core
# radius). Kiefer-Chatterjee is used because its explicit dense core carries a
# real fraction of the near-track energy; a diffuse `1/r^2` model (Scholz-Kraft)
# starves the soft core and shows a dip near the soft/hard seam.

# %%
sp = default_stopping_power()
sp_label = describe_stopping_power(sp)

site = SphericalSite(diameter_um=1.0)
spec = AnalyticSpectrumEngine(
    stopping_power=sp, site=site, restricted=True, include_delta_tail=False
)
calc = LinealEnergyCalculator(stopping_power=sp, site=site, engine=spec)
amorphous = AmorphousTrackEngine(
    KieferChatterjeeRadialDose(sp),
    site,
    stopping_power=sp,
    stochastic=DeltaRayMonteCarlo(n_samples=N_DELTA_SAMPLES, seed=0),
)
print(f"analytic engine: 1 um sphere, restricted energy loss, LET from {sp_label}")
print("amorphous engine: 1 um sphere, KC radial dose + delta MC (soft core + partial deposition)")

# %% [markdown]
# A few proton energies spanning the therapeutic range (the library's nearest
# available energy is used for each); dose-mean `y_D` of the library and both
# models.

# %%
target_energies_MeV = [5.5, 10.5, 30.5, 100.5]
rows = []
for target in target_energies_MeV:
    tab = library.nearest(target)
    ion = tab.meta["ion"] or Ion(Z=1, A=1, energy_MeV_u=tab.meta["energy_MeV"])
    rows.append((tab, calc.spectrum(ion), amorphous.single_event_spectrum(ion)))

header = f"{'E [MeV]':>8} {'LET':>7} {'yD_lib':>7} {'yD_ana':>7} {'yD_amo':>7}"
print(f"\n{header}")
for tab, ana, amo in rows:
    let = sp.let_keV_um(tab.meta["ion"])
    print(f"{tab.meta['energy_MeV']:8.1f} {let:7.2f} {tab.y_D:7.2f} {ana.y_D:7.2f} {amo.y_D:7.2f}")

# %% [markdown]
# The dose-weighted spectra `y·d(y)` (equal area = equal dose): the MC library as
# filled steps, the two pyRadBio models as lines. The analytic engine matches the
# crosser peak but has no delta-ray tail; the amorphous engine (with the delta-ray
# Monte Carlo) supplies a tail and a physical low-`y` floor.

# %%
n = len(rows)
fig, axes = plt.subplots(1, n, figsize=(3.6 * n, 3.8), sharey=True)
axes = axes if n > 1 else [axes]

for ax, (tab, ana, amo) in zip(axes, rows, strict=True):
    lib_spec = tab.to_spectrum()
    ax.fill_between(
        lib_spec.y_keV_um, lib_spec.yd, step="mid", alpha=0.35, color="C0", label="library (MC)"
    )
    ax.plot(ana.y_keV_um, ana.yd, color="C3", ls="--", lw=1.8, label="analytic")
    ax.plot(amo.y_keV_um, amo.yd, color="C2", ls=":", lw=1.8, label="amorphous")
    ax.set_xscale("log")
    ax.set_xlim(1.0e-1, 3.0e2)
    ax.set_xlabel("y [keV/um]")
    ax.set_title(f"{tab.meta['energy_MeV']:.1f} MeV proton")
    ax.grid(True, which="both", ls=":", lw=0.4)
axes[0].set_ylabel(r"$y\,d(y)$")
axes[0].legend(frameon=False, fontsize=8)

fig.suptitle(f"Proton lineal-energy spectra, 1 um sphere - models vs DeCunha 2025 ({sp_label})")
fig.tight_layout()
save_figure(fig, "lineal_energy_library_comparison.png", directory=FIGURES, dpi=140)
