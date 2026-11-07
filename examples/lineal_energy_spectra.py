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
# # Analytic lineal-energy spectra
#
# The stochastic analogue of LET is the **lineal energy** $y = \varepsilon /
# \bar{\ell}$: the energy $\varepsilon$ deposited in a small site by a single
# event, divided by the site's mean chord length $\bar{\ell}$. This example
# computes single-event spectra $f_1(y)$ and their dose-weighted form $d_1(y) =
# y f_1(y) / \bar{y}_F$ for a few ions in a micrometre water sphere, using
# pyRadBio's analytic (Xapsos-Kellerer) engine.
#
# The engine needs only a stopping-power lookup plus an energy-loss straggling
# model — no radial-dose model. For a crosser with chord $\ell$ the mean
# deposited energy is $\mathrm{LET}\cdot\ell$; the deposit fluctuates by Bohr
# straggling, modelled with a positive-definite log-normal (Xapsos et al. 1994,
# Radiat. Res. 137, 152). Two exact checks fall out and are worth watching in
# the printout: the frequency mean $\bar{y}_F \to \mathrm{LET}$, and the
# dose-mean $\bar{y}_D$ rises above the geometric value $\tfrac{9}{8}\mathrm{LET}$
# as straggling grows (heavier ions, smaller sites).
#
# This file is a [jupytext](https://jupytext.readthedocs.io) percent script: run
# it directly with `python`, or pair/open it as a notebook (see
# `examples/README.md`).

# %%
import matplotlib.pyplot as plt

from pyradbio import Ion
from pyradbio.micro import (
    AmorphousTrackEngine,
    AnalyticSpectrumEngine,
    CylindricalSite,
    DeltaRayMonteCarlo,
    LinealEnergyCalculator,
    SphericalSite,
)
from pyradbio.plotting import figures_dir, save_figure
from pyradbio.track.kiefer_chatterjee import KieferChatterjeeRadialDose
from pyradbio.track.stopping_power import default_stopping_power, describe_stopping_power

FIGURES = figures_dir(globals().get("__file__"))

# %% [markdown]
# The library's default stopping power; a 1 µm sphere is a typical TEPC-like
# microdosimetric site. Four ions span three decades of LET: a fast proton, a
# helium ion, and carbon at a plateau-like and a slower, higher-LET energy.

# %%
sp = default_stopping_power()
source_label = describe_stopping_power(sp)

site = SphericalSite(diameter_um=1.0)
calc = LinealEnergyCalculator(stopping_power=sp, site=site)  # log-normal straggling
print(
    f"stopping power: {source_label}; site: 1.0 um sphere (l_bar = {site.mean_chord_um():.3f} um)"
)

setups = {
    "proton, 100 MeV/u": Ion(Z=1, A=1, energy_MeV_u=100.0),
    "helium-4, 100 MeV/u": Ion(Z=2, A=4, energy_MeV_u=100.0),
    "carbon-12, 200 MeV/u": Ion(Z=6, A=12, energy_MeV_u=200.0),
    "carbon-12, 40 MeV/u": Ion(Z=6, A=12, energy_MeV_u=40.0),
}
spectra = {label: calc.spectrum(ion) for label, ion in setups.items()}

# %% [markdown]
# `y_F` tracks LET closely (the analytic engine is mean-preserving); `y_D`/`y_F > 1`
# reflects the chord-length and straggling spread, and the saturation-corrected
# `y*` (Kase 2006, `y0 = 150 keV/um`) caps the high-`y` tail as the MKM uses it.
# The right panel plots $y\,d_1(y)$ against $\log y$, the conventional
# presentation in which equal areas represent equal dose.

# %%
print(f"{'setup':>22s} {'LET':>8s} {'y_F':>8s} {'y_D':>8s} {'y_D/y_F':>8s} {'y*':>8s}")
for label, spec in spectra.items():
    let = spec.meta["let_keV_um"]
    print(
        f"{label:>22s} {let:8.2f} {spec.y_F:8.2f} {spec.y_D:8.2f} "
        f"{spec.y_D / spec.y_F:8.3f} {spec.y_star(150.0):8.2f}"
    )

fig, (ax_f, ax_d) = plt.subplots(1, 2, figsize=(12.0, 4.6))
for label, spec in spectra.items():
    ax_f.plot(spec.y_keV_um, spec.f, label=label, lw=1.6)
    ax_d.plot(spec.y_keV_um, spec.yd, label=label, lw=1.6)  # equal-area-in-dose

ax_f.set_yscale("log")
ax_f.set_ylim(1.0e-6, 1.0e0)
ax_f.set_ylabel(r"$f_1(y)$")
ax_f.set_title("Single-event frequency spectrum")
ax_d.set_ylabel(r"$y\,d_1(y)$")
ax_d.set_title("Dose-weighted spectrum (equal area = equal dose)")
for ax in (ax_f, ax_d):
    ax.set_xscale("log")
    ax.set_xlim(1.0e-1, 1.0e3)
    ax.set_xlabel("Lineal energy y [keV/um]")
    ax.grid(True, which="both", ls=":", lw=0.45)
    ax.legend(frameon=False, fontsize=8)

fig.suptitle(f"Analytic lineal-energy spectra, 1 um water sphere — {source_label}")
fig.tight_layout()
save_figure(fig, "lineal_energy_spectra.png", directory=FIGURES, dpi=140)

# %% [markdown]
# **Site size.** For a fixed carbon ion, shrinking the sphere increases the
# relative straggling, so the spectrum broadens and `y_D` climbs above the
# geometric `9/8 · LET`, while `y_F` stays pinned at the LET.

# %%
carbon = Ion(Z=6, A=12, energy_MeV_u=100.0)
let_c = sp.let_keV_um(carbon)
print(f"\ncarbon-12 at 100 MeV/u: LET = {let_c:.2f} keV/um; geometric y_D = {1.125 * let_c:.2f}")
print(f"{'diameter [um]':>14s} {'y_F':>8s} {'y_D':>8s}")
for diameter in (0.1, 0.3, 1.0, 3.0):
    spec = LinealEnergyCalculator(sp, SphericalSite(diameter)).spectrum(carbon)
    print(f"{diameter:14.2f} {spec.y_F:8.2f} {spec.y_D:8.2f}")

# %% [markdown]
# ## Site *shape*: sphere vs cylinder (TEPC-like)
#
# The engine is geometry-agnostic — any site providing a chord-length
# distribution works. A `CylindricalSite` (right circular cylinder) is the
# TEPC-detector analogue; its chord distribution has no closed form, so it is
# obtained from isotropic random secants (validated against Cauchy's exact
# `l_bar = 4V/S`). A "unit" cylinder (d = h = 1 µm) has the same mean chord as a
# 1 µm sphere (`4V/S = 2/3 um` for both), so at equal LET its `y_F` is the same
# (both track the LET) but the spectrum shape and `y_D` differ.

# %%
shapes = {
    "sphere, d=1 um": SphericalSite(diameter_um=1.0),
    "cylinder, d=h=1 um": CylindricalSite(diameter_um=1.0, height_um=1.0),
    "cylinder, d=1 h=0.5 um": CylindricalSite(diameter_um=1.0, height_um=0.5),
}
shape_spectra = {
    label: LinealEnergyCalculator(sp, geom).spectrum(carbon) for label, geom in shapes.items()
}
print(f"\n{'site':>26s} {'l_bar [um]':>11s} {'y_F':>8s} {'y_D':>8s}")
for label, geom in shapes.items():
    spec = shape_spectra[label]
    print(f"{label:>26s} {geom.mean_chord_um():11.3f} {spec.y_F:8.2f} {spec.y_D:8.2f}")

fig2, ax = plt.subplots(figsize=(6.4, 4.6))
for label, spec in shape_spectra.items():
    ax.plot(spec.y_keV_um, spec.yd, label=label, lw=1.6)
ax.set_xscale("log")
ax.set_xlim(1.0e0, 5.0e2)
ax.set_xlabel("Lineal energy y [keV/um]")
ax.set_ylabel(r"$y\,d_1(y)$")
ax.set_title("Site shape: carbon-12, 100 MeV/u")
ax.grid(True, which="both", ls=":", lw=0.45)
ax.legend(frameon=False, fontsize=8)
fig2.tight_layout()
save_figure(fig2, "lineal_energy_site_shape.png", directory=FIGURES, dpi=140)

# %% [markdown]
# ## A second engine + the stochastic delta-ray layer
#
# `AmorphousTrackEngine` builds the spectrum from a **radial-dose model** `D(r)`
# (here Kiefer-Chatterjee, for its more physical penumbra; any `RadialDoseModel`
# works) via the impact-parameter method. On its own it is *deterministic* —
# `eps(b)` is the mean energy at impact parameter `b`. A `DeltaRayMonteCarlo`
# resamples each deposit as a compound Poisson sum of single-collision energy
# losses, the microscopic origin of both straggling and delta-ray discreteness.
#
# - **Left, large site** (20 µm sphere, 1 MeV proton): the deterministic
#   amorphous engine reduces to the geometric chord spectrum and coincides with
#   the analytic engine's `straggling="none"` mode — a mutual validation.
# - **Right, small site** (1 µm sphere, 10 MeV proton): the deterministic
#   amorphous mean over-populates the low-`y` **toucher tail** (it smears discrete
#   delta-rays into continuous tiny deposits), pushing `y_F` far too low. The
#   stochastic layer turns those far touchers into *non-events* (Poisson zeros)
#   and broadens the crosser peak, lifting `y_F` back to a physical value. The
#   analytic engine is drawn with its default log-normal straggling here.

# %%
rdd = KieferChatterjeeRadialDose(sp)
delta_mc = DeltaRayMonteCarlo(n_samples=300_000, seed=0)


def amorphous(site: SphericalSite, ion: Ion, stochastic: DeltaRayMonteCarlo | None = None):
    """Single-event spectrum of the amorphous engine, with or without the delta-ray MC."""
    engine = AmorphousTrackEngine(rdd, site, stopping_power=sp, stochastic=stochastic)
    return engine.single_event_spectrum(ion)


big_site, proton_slow = SphericalSite(diameter_um=20.0), Ion(Z=1, A=1, energy_MeV_u=1.0)
geo_analytic = AnalyticSpectrumEngine(sp, big_site, straggling="none").single_event_spectrum(
    proton_slow
)
geo_amorph = amorphous(big_site, proton_slow)
geo_amorph_mc = amorphous(big_site, proton_slow, delta_mc)

small_site, proton = SphericalSite(diameter_um=1.0), Ion(Z=1, A=1, energy_MeV_u=10.0)
small_analytic = AnalyticSpectrumEngine(sp, small_site).single_event_spectrum(proton)
small_amorph_det = amorphous(small_site, proton)
small_amorph_mc = amorphous(small_site, proton, delta_mc)

print(f"\n{'case':>28s} {'engine':>18s} {'y_F':>8s} {'y_D':>8s}")
for label, spec in [
    ("20 um, 1 MeV p", geo_analytic),
    ("", geo_amorph),
    ("1 um, 10 MeV p", small_amorph_det),
    ("", small_amorph_mc),
]:
    engine = spec.meta.get("engine", spec.meta.get("straggling", "analytic"))
    print(f"{label:>28s} {engine:>18} {spec.y_F:8.2f} {spec.y_D:8.2f}")

DET = {"color": "C3", "ls": "--", "lw": 1.8}
fig3, (axg, axs) = plt.subplots(1, 2, figsize=(11.5, 4.4))
panels = [
    (axg, "Large site (20 um), 1 MeV proton - geometric limit", "none", geo_analytic,
     geo_amorph, geo_amorph_mc),
    (axs, "Small site (1 um), 10 MeV proton - stochastic layer", "log-normal", small_analytic,
     small_amorph_det, small_amorph_mc),
]  # fmt: skip
for ax, title, straggling, analytic, det, mc in panels:
    ax.plot(analytic.y_keV_um, analytic.yd, label=f"analytic ({straggling})", color="C0", lw=2.2)
    ax.plot(det.y_keV_um, det.yd, label="amorphous (deterministic)", **DET)
    ax.plot(mc.y_keV_um, mc.yd, label="amorphous + delta-ray MC", color="C2", lw=2.2)
    ax.set_title(title)
    ax.set_xscale("log")
    ax.set_xlabel("Lineal energy y [keV/um]")
    ax.set_ylabel(r"$y\,d_1(y)$")
    ax.grid(True, which="both", ls=":", lw=0.45)
    ax.legend(frameon=False, fontsize=8)
fig3.suptitle("Amorphous track engine: deterministic mean and stochastic delta-ray layer")
fig3.tight_layout()
save_figure(fig3, "lineal_energy_two_engines.png", directory=FIGURES, dpi=140)
