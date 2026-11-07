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
# # MCF MKM & mMKM: RBE(LET) from digitized lineal-energy spectra
#
# The companion to `validation/lineal_energy_hartzell2025.py`: that one validates
# the *spectra* `d(y)`; this one feeds those spectra to the biophysical models
# that turn `d(y)` into cell survival. Two microdosimetric-kinetic models
# (`pyradbio.models.mkm.spectrum_models`) consume the **same** dose-weighted single-event
# spectra and predict LQ parameters and RBE for the HSG cell line:
#
# * **MCF MKM** (Parisi, Beltran & Furutani 2022) — both `α` and `β` vary with
#   radiation quality through the saturation (overkill) correction `c(y)`.
# * **mMKM** (Kase et al. 2006) — `α` follows the saturation-corrected
#   `y*_D`; `β = β_ref` is radiation-independent. The natural cross-check.
#
# We drive both from the bundled Hartzell 2025 **AMF** spectra (six ions
# `1H 4He 7Li 12C 16O 20Ne` at 1/10/100 MeV/u, 0.3 µm sphere) and plot
# `RBE₁₀` against dose-averaged LET. What to look for: the carbon RBE rises with
# LET to a peak near ~100–200 keV/µm then **overkill-drops** at 1 MeV/u (the
# highest LET), and MCF `β` collapses at high LET where mMKM holds `β` fixed —
# so the two models agree on the rising limb and diverge in the overkill tail.
#
# This file is a [jupytext](https://jupytext.readthedocs.io) percent script.

# %%
import matplotlib.pyplot as plt

from pyradbio.benchmark.hartzell2025 import get_amf_curve, iter_amf_curves
from pyradbio.models.mkm.spectrum_models import MCFMKMModel, ModifiedMKM
from pyradbio.plotting import figures_dir, save_figure
from pyradbio.track.stopping_power import SimpleBetheStoppingPower

FIGURES = figures_dir(globals().get("__file__"))

IONS = ["hydrogen", "helium", "lithium", "carbon", "oxygen", "neon"]
ENERGIES = [1.0, 10.0, 100.0]

mcf = MCFMKMModel()
mmkm = ModifiedMKM()
sp = SimpleBetheStoppingPower()

# %% [markdown]
# ## Table: α, β and RBE₁₀ per (ion, energy)
#
# `α`/`β` come straight from the single-event spectrum `d(y)` (the digitized
# `y·d(y)` with the leading `y` divided out, `curve.dose_spectrum()`); `RBE₁₀`
# is the iso-survival ratio at `S = 0.1` versus the 6 MV photon reference
# (`α_ref = 0.217`, `β_ref = 0.0615`). Each point is computed once and reused by
# the figure below.

# %%
print(
    f"{'ion':>9} {'E':>6} {'LET':>7} | "
    f"{'MCF a':>7} {'MCF b':>7} {'RBE10':>6} | {'mMKM a':>7} {'mMKM b':>7} {'RBE10':>6}"
)
print("-" * 78)
points: dict[str, list[tuple[float, float, float]]] = {}
for ion_name in IONS:
    for energy in ENERGIES:
        curve = get_amf_curve(figure="fig4", ion=ion_name, method="AMF", energy_MeV_u=energy)
        y, d = curve.dose_spectrum()
        let = sp.let_keV_um(curve.as_ion())
        lq_m = mcf.alpha_beta_from_dose_spectrum(y, d)
        lq_k = mmkm.alpha_beta_from_dose_spectrum(y, d)
        rbe_m = mcf.rbe_at_survival_from_dose_spectrum(y, d)
        rbe_k = mmkm.rbe_at_survival_from_dose_spectrum(y, d)
        points.setdefault(ion_name, []).append((let, rbe_m, rbe_k))
        print(
            f"{ion_name:>9} {energy:>6.0f} {let:>7.1f} | "
            f"{lq_m.alpha:>7.3f} {lq_m.beta:>7.4f} {rbe_m:>6.2f} | "
            f"{lq_k.alpha:>7.3f} {lq_k.beta:>7.4f} {rbe_k:>6.2f}"
        )

# %% [markdown]
# ## RBE₁₀ vs LET

# %%
fig, ax = plt.subplots(figsize=(6.5, 4.5))
for ion_name in IONS:
    pts = sorted(points[ion_name])
    let = [p[0] for p in pts]
    ax.plot(let, [p[1] for p in pts], "-o", lw=1.6, ms=5, label=f"{ion_name} (MCF)")
    ax.plot(
        let, [p[2] for p in pts], "--s", lw=1.2, ms=4, alpha=0.7, color=ax.lines[-1].get_color()
    )

ax.set_xscale("log")
ax.set_xlabel("dose-averaged LET [keV/um]")
ax.set_ylabel(r"$\mathrm{RBE}_{10}$")
ax.set_title("HSG RBE10 vs LET — MCF MKM (solid) vs mMKM (dashed)\nHartzell 2025 AMF spectra")
ax.grid(True, which="both", ls=":", lw=0.4)
ax.legend(frameon=False, fontsize=7, ncol=2)
fig.tight_layout()
print()
save_figure(fig, "mcf_mkm_rbe_vs_let.png", directory=FIGURES, dpi=140)

# %% [markdown]
# The two models track each other on the low-LET rising limb and split in the
# high-LET overkill tail (MCF drops faster because its `β` saturates while the
# mMKM `β` is frozen at `β_ref`). `iter_amf_curves()` filters the bundled set:

# %%
print(f"{len(iter_amf_curves(figure='fig4', method='AMF'))} AMF Fig. 4 curves drove this example")
