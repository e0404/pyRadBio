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
# # LEM I reconstruction: Scholz & Kraft 1996
#
# Reproduces the original amorphous-track Local Effect Model paper -- Scholz &
# Kraft 1996, *Adv. Space Res.* 18(1/2):5-14, doi:10.1016/0273-1177(95)00784-C --
# from pyRadBio's `LEM1`, against curves **hand-digitized** from its Figs. 2-4
# (`pyradbio.benchmark.scholz_kraft1996`): the inactivation cross-section `sigma`
# vs LET and vs energy, and RBE vs LET. The comparison is quantitative; residual
# disagreement is not digitization noise. The companion Scholz et al. 1997 paper
# (the fast approximation) is `validation/scholz1997_lem1.py`; the ladder of LEM I
# alpha/beta variants is `examples/lem1_variants.py`.
#
# The panels keep paper-defined observables distinct. In particular the 1996
# `sigma_inact` is a **final-slope quantity at 1% survival**, not the low-dose
# single-track cross-section (`docs/references/LEM_I_ScholzKraft.md`, "Core
# equations"). The paper's Fig. 5 is intentionally not reconstructed: the paper
# labels it a schematic (hypothetical fixed-LET survival of mammalian vs yeast
# cells), and its yeast shoulder needs the spatially explicit track-overlap MC that
# the fast LQ/z-domain paths do not model.

# %%
from __future__ import annotations

import os
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
from _common import pick, quick_mode
from matplotlib.lines import Line2D

from pyradbio import Ion
from pyradbio.benchmark.metrics import ratio_summary
from pyradbio.benchmark.scholz_kraft1996 import (
    FIGURE_INPUTS,
    FINAL_SLOPE_ESTIMATOR_VERSION,
    LEVEL_TAG,
    RBE_PARTICLES,
    SIGMA_PARTICLES,
    compare_mc_to_paper,
    iter_curves,
    load_mc_observables,
    paper_sigma_derived_rbe_1pct,
    v79_lem1_models,
)
from pyradbio.benchmark.stopping import select_stopping_power, stopping_power_label
from pyradbio.core.lq import dose_for_survival
from pyradbio.plotting import figures_dir, save_figure
from pyradbio.track.northcliffe_schilling import (
    ENERGY_RANGE_MEV_U,
    BetheOnly,
    NorthcliffeSchillingWithBetheTail,
)
from pyradbio.track.stopping_power import SimpleBetheStoppingPower

FIGURES = figures_dir(globals().get("__file__"))
DATA = (Path(__file__).resolve().parent if "__file__" in globals() else Path.cwd()) / "data"
if quick_mode():
    print(
        "quick mode (PYRADBIO_QUICK=1): 3-point energy sweeps, Fig. 2 lines for C/Ar/U only; "
        "numbers are not for quoting"
    )

# %% [markdown]
# ## Stopping power
#
# Stopping power places every point on the LET axis and normalises the radial
# dose. The paper normalises to "the tables of Hubert et al. and Northcliffe and
# Schilling"; only Northcliffe & Schilling tabulates water, and pyRadBio ships its
# water column. Select with `LEM1_STOPPING` (`pyradbio.benchmark.stopping`):
#
# * `ns1970` (default) -- the paper's own table. Its grid stops at 12 MeV/u while
#   these panels run to 20, so above 12 MeV/u it falls back to analytic Bethe and
#   *counts* the fallbacks; the tally is printed at the end of the summary. The seam
#   is deliberate and visible: extrapolating a 1970 table past its last point would
#   be invisible on a log-log figure.
# * `bethe` -- dependency-free analytic Bethe everywhere (validated to ~1.4% RMS
#   against pyMKM's Geant4 table for Z <= 10); within 1% of ns1970 for light ions at
#   high energy but ~15% above it for Kr/Xe at 12 MeV/u. It is a paper-era proxy, not
#   the library default (`default_stopping_power()`).
# * `pymkm[:SOURCE]` -- pyMKM's tabulated LET (`mkm` extra; Z <= 10).
# * `libdedx[:PROGRAM]` -- reference-grade *updated* stopping power (optional GPL
#   package); ~15-40% higher heavy-ion LET than the 1980 tables.
#
# An ion outside the selected table's Z coverage falls back to analytic Bethe (with
# a one-time note).

# %%
LEM1_STOPPING = os.environ.get("LEM1_STOPPING", "ns1970")
_BETHE = SimpleBetheStoppingPower()

try:
    STOPPING_POWER = select_stopping_power(LEM1_STOPPING)
    STOPPING_POWER_LABEL = stopping_power_label(LEM1_STOPPING)
except ImportError:
    print(f"{LEM1_STOPPING!r} not importable; falling back to analytic Bethe")
    STOPPING_POWER, STOPPING_POWER_LABEL = BetheOnly(), "analytic Bethe (fallback)"
print(f"stopping power: {STOPPING_POWER_LABEL}")

_HEAVY_FALLBACK_NOTED = False


def let_of(ion: Ion) -> float:
    """LET for ``ion``, falling back to Bethe if the chosen table lacks its Z."""
    global _HEAVY_FALLBACK_NOTED
    try:
        return STOPPING_POWER.let_keV_um(ion)
    except (KeyError, ValueError, RuntimeError, NotImplementedError):
        if not _HEAVY_FALLBACK_NOTED:
            print(
                f"  (note: {STOPPING_POWER_LABEL} lacks Z>{ion.z - 1}; using Bethe for heavy ions)"
            )
            _HEAVY_FALLBACK_NOTED = True
        return _BETHE.let_keV_um(ion)


# %% [markdown]
# ## The offline full-MC checkpoint
#
# The paper obtains `sigma_inact` from the final slope at 1% survival. The correct
# pyRadBio values come from traversal-conditioned stochastic survival curves,
# computed offline by `scripts/lem1_mc_reference.py sk1996-final-slope` into the
# tracked `validation/data/sk1996_mc_observables.csv`; the same curves give Fig. 4's
# iso-survival RBE without an ion-LQ surrogate. Every row must come from the current
# estimator (rows of another final-slope definition are a different quantity) and
# from this run's stopping power: LET enters both the x axis and the sigma
# conversion, so a checkpoint at another LET is incoherent with these axes, not
# merely imprecise. A mismatch warns rather than crashes.

# %%
SK1996_MC_RESULTS = load_mc_observables(DATA / "sk1996_mc_observables.csv")
SK1996_SIGMA_ESTIMATOR_CURRENT = bool(SK1996_MC_RESULTS) and all(
    int(row.get("estimator_version", -1)) == FINAL_SLOPE_ESTIMATOR_VERSION
    for rows in SK1996_MC_RESULTS.values()
    for row in rows
)
SK1996_ACTIVE_STOPPING_NAME = getattr(STOPPING_POWER, "name", None)
SK1996_CHECKPOINT_STOPPING = {
    str(row.get("stopping_power", "unknown")) for rows in SK1996_MC_RESULTS.values() for row in rows
}
if SK1996_ACTIVE_STOPPING_NAME is not None and SK1996_CHECKPOINT_STOPPING - {
    SK1996_ACTIVE_STOPPING_NAME
}:
    print(
        f"  (warning: MC checkpoint used stopping power "
        f"{sorted(SK1996_CHECKPOINT_STOPPING)} but this run uses "
        f"{SK1996_ACTIVE_STOPPING_NAME!r}; rerun scripts/lem1_mc_reference.py "
        "sk1996-final-slope)"
    )
SK1996_MC_SIGMA_LABEL = (
    "pyRadBio full MC"
    if SK1996_SIGMA_ESTIMATOR_CURRENT
    else "pyRadBio full MC (legacy slope; rerun batch)"
)

# %% [markdown]
# ## The deterministic line: the second-order cluster expansion
#
# A solid deterministic curve stays visible in every panel, labelled separately
# from the full-MC points. It is the **second-order fluence (cluster) expansion** of
# the same stochastic model -- `alpha_I` at first order plus the two-track overlap
# `beta` at second (`beta_model="scholz1997_two_track"`) -- so it approximates the
# paper's observable rather than substituting another parameterisation. Figs. 2-3
# also keep the dash-dot `sigma_0` (zero-dose) curve as a reference; it is not the
# paper observable.
#
# It replaced a slope-matching LQ surrogate (the `LEM1` default
# `beta_I = (s_max - alpha_I)/(2 D_t)`), which was not merely imprecise but
# unphysical at high LET: a high-dose asymptotic identity hands the least shouldered
# ions the full photon shoulder, and the resulting `sigma` exceeded the geometric
# ceiling `pi (R + r_max)^2` at 84 of Fig. 2's 126 grid points, up to 40x for
# uranium. Against the converged 40-point MC checkpoint the expansion holds
# `sigma_1%` within 5% at 37/40 points and every RBE within 3% at 111/120 (particle,
# energy, level) combinations. The exceptions are uranium at 16 and 20 MeV/u
# (-14%, -25% on sigma), where the pair term truncates while overlap persists --
# the open high-energy uranium finding (`docs/references/LEM_I_ScholzKraft.md`).
# Figs. 2, 3 and 4 share one input set (Fig. 2 caption), so one model draws them all.

# %%
ION_A = {1: 1, 2: 4, 6: 12, 10: 20, 18: 40, 36: 84, 54: 132, 92: 238}
_, lem_v79 = v79_lem1_models(STOPPING_POWER, FIGURE_INPUTS["fig2_sigma_vs_let_v79"])
a_eff_v79 = FIGURE_INPUTS["fig2_sigma_vs_let_v79"].A_nucl_um2
SK1996_APPROX_ENERGIES = np.geomspace(1.0, 20.0, pick(18, 3))
SK1996_APPROX_SIGMA_CACHE: dict[tuple[int, float], float] = {}
SK1996_SIGMA0_CACHE: dict[tuple[int, float], float] = {}
# One label for the deterministic line in Figs. 2-4: the truncated expansion of the
# stochastic model the MC points sample, not a fit.
SK1996_EXPANSION_LABEL = "pyRadBio 2nd-order cluster expansion"


def sk1996_ion(z: int, energy_MeV_u: float) -> Ion:
    return Ion(z, ION_A[z], energy_MeV_u)


def sk1996_approx_sigma(z: int, energy_MeV_u: float) -> float:
    key = (z, energy_MeV_u)
    if key not in SK1996_APPROX_SIGMA_CACHE:
        SK1996_APPROX_SIGMA_CACHE[key] = lem_v79.approximate_inactivation_cross_section_um2(
            sk1996_ion(z, energy_MeV_u), survival_level=0.01
        )
    return SK1996_APPROX_SIGMA_CACHE[key]


def sk1996_sigma0(z: int, energy_MeV_u: float) -> float:
    key = (z, energy_MeV_u)
    if key not in SK1996_SIGMA0_CACHE:
        SK1996_SIGMA0_CACHE[key] = lem_v79.inactivation_cross_section_um2(
            sk1996_ion(z, energy_MeV_u)
        )
    return SK1996_SIGMA0_CACHE[key]


# %% [markdown]
# ## Fig. 2 -- inactivation cross-section vs LET (V79)
#
# One coloured `sigma(LET)` trace per ion over the paper's 1-20 MeV/u, with the
# paper's experimental symbols in the same colour. `sigma` rises from the photon
# limit toward the effective nuclear size (50 um^2) but does **not** cap there: the
# heaviest ions overshoot to ~100 um^2 (U), because `sigma` integrates over impact
# parameters that miss the nucleus and a large delta-ray penumbra still inactivates
# the cell. Each trace turns back near the ion's Bragg-peak LET. (Quick mode draws
# the expansion lines only for the Fig. 3 ions, whose values Fig. 3 reuses.)

# %%
FIG2_IONS = {
    "helium": (2, "He", "#1f77b4", "o"),
    "carbon": (6, "C", "#ff7f0e", "s"),
    "neon": (10, "Ne", "#2ca02c", "^"),
    "argon": (18, "Ar", "#d62728", "v"),
    "krypton": (36, "Kr", "#9467bd", "D"),
    "xenon": (54, "Xe", "#8c564b", "P"),
    "uranium": (92, "U", "#17becf", "X"),
}
# The cost knob: each expansion point is a two-track quadrature (~2.5 s).
FIG2_LINE_IONS = pick(tuple(FIG2_IONS), ("carbon", "argon", "uranium"))
fig2, ax = plt.subplots(figsize=(8.0, 5.6))
for particle, (z, _sym, colour, marker) in FIG2_IONS.items():
    if particle in FIG2_LINE_IONS:
        ions = [sk1996_ion(z, float(energy)) for energy in SK1996_APPROX_ENERGIES]
        approx_lets = np.array([let_of(ion) for ion in ions])
        approx_sigma = np.array(
            [sk1996_approx_sigma(z, float(energy)) for energy in SK1996_APPROX_ENERGIES]
        )
        sigma0 = np.array([sk1996_sigma0(z, float(energy)) for energy in SK1996_APPROX_ENERGIES])
        approx_order = np.argsort(approx_lets)
        ax.plot(
            approx_lets[approx_order],
            approx_sigma[approx_order],
            color=colour,
            lw=2.0,
            alpha=0.9,
            zorder=4,
        )
        ax.plot(
            approx_lets[approx_order],
            sigma0[approx_order],
            color=colour,
            lw=1.4,
            ls="-.",
            alpha=0.65,
            zorder=3,
        )
    rows = SK1996_MC_RESULTS.get(particle, [])
    if rows:
        lets = np.array([row["let_keV_um"] for row in rows])
        sig = np.array([row["sigma_1pct_um2"] for row in rows])
        err = np.array([row["sigma_1pct_err_um2"] for row in rows])
        order = np.argsort(lets)
        ax.errorbar(
            lets[order],
            sig[order],
            yerr=err[order],
            color=colour,
            ls="None",
            marker="o",
            ms=4,
            capsize=2,
            zorder=6,
        )
    for c in iter_curves(figure="fig2_sigma_vs_let_v79", particle=particle, series="data"):
        ax.plot(c.x, c.y, ls="None", marker=marker, ms=8, mfc="none", mec=colour, mew=1.4)

# The paper draws its model sigma(LET) as several per-ion/LET-band segments
# (digitized as curve1..curve6); all of them form the grey paper-model line.
for model_line in iter_curves(figure="fig2_sigma_vs_let_v79", series="calc"):
    ax.plot(model_line.x, model_line.y, color="0.35", lw=1.6, ls="--", zorder=1)
ax.axhline(a_eff_v79, color="0.55", lw=1.0, ls=":")

species_handles = [
    Line2D([0], [0], marker=m, ms=6, color=col, label=sym) for _z, sym, col, m in FIG2_IONS.values()
]
role_handles = [
    Line2D([0], [0], color="0.3", lw=2.0, label=f"{SK1996_EXPANSION_LABEL} at 1%"),
    Line2D([0], [0], color="0.3", lw=1.4, ls="-.", label="pyRadBio sigma_0: zero-dose reference"),
    Line2D([0], [0], ls="None", marker="o", ms=8, mfc="none", mec="0.3", label="open: paper data"),
    Line2D([0], [0], color="0.35", lw=1.6, ls="--", label="paper model line"),
    Line2D([0], [0], color="0.55", lw=1.0, ls=":", label=f"effective size {a_eff_v79:g} um$^2$"),
]
if SK1996_MC_RESULTS:
    role_handles.insert(
        1,
        Line2D([0], [0], color="0.3", ls="None", marker="o", ms=4, label=SK1996_MC_SIGMA_LABEL),
    )
else:
    role_handles.insert(
        1,
        Line2D(
            [0],
            [0],
            color="none",
            label="pyRadBio sigma at 1%: run scripts/lem1_mc_reference.py",
        ),
    )
legend1 = ax.legend(handles=species_handles, frameon=False, fontsize=9, loc="upper left", ncol=2)
ax.add_artist(legend1)
ax.legend(handles=role_handles, frameon=False, fontsize=8, loc="lower right")
ax.set_xscale("log")
ax.set_yscale("log")
ax.set_xlim(10, 3.0e4)
ax.set_ylim(1, 130)
ax.set_xlabel("LET (keV/um)")
ax.set_ylabel(r"$\sigma_{\mathrm{inact}}$ (um$^2$)")
ax.set_title(
    f"Scholz & Kraft 1996 Fig. 2: sigma vs LET (V79)\nstopping power: {STOPPING_POWER_LABEL}"
)
ax.grid(True, which="both", ls=":", lw=0.5)
fig2.tight_layout()
save_figure(fig2, "sk1996_fig2_sigma_vs_let.png", directory=FIGURES, dpi=130)

# %% [markdown]
# ## Fig. 3 -- inactivation cross-section vs specific energy (C, Ar, U)
#
# Falling / flat / rising branches, driven by the LET decrease against the growth of
# the track radius.

# %%
FIG3_IONS = {"carbon": (6, "C0"), "argon": (18, "C1"), "uranium": (92, "C2")}

fig3, ax = plt.subplots(figsize=(7.2, 5.4))
for particle, (z, color) in FIG3_IONS.items():
    approx_sigma = np.array(
        [sk1996_approx_sigma(z, float(energy)) for energy in SK1996_APPROX_ENERGIES]
    )
    sigma0 = np.array([sk1996_sigma0(z, float(energy)) for energy in SK1996_APPROX_ENERGIES])
    ax.plot(SK1996_APPROX_ENERGIES, approx_sigma, color=color, lw=2.2)
    ax.plot(SK1996_APPROX_ENERGIES, sigma0, color=color, lw=1.4, ls="-.", alpha=0.65)
    rows = SK1996_MC_RESULTS.get(particle, [])
    if rows:
        energy = np.array([row["energy_MeV_u"] for row in rows])
        sig = np.array([row["sigma_1pct_um2"] for row in rows])
        err = np.array([row["sigma_1pct_err_um2"] for row in rows])
        ax.errorbar(energy, sig, yerr=err, color=color, ls="None", marker="o", ms=4, capsize=2)
    for c in iter_curves(figure="fig3_sigma_vs_energy_v79", particle=particle, series="calc"):
        ax.plot(c.x, c.y, color=color, lw=1.3, ls="--", alpha=0.8)
    for c in iter_curves(figure="fig3_sigma_vs_energy_v79", particle=particle, series="data"):
        ax.plot(c.x, c.y, color=color, ls="None", marker="s", ms=5, mfc="white")

ax.set_xlim(0, 20)
ax.set_ylim(0, 120)
ax.set_xlabel("Energy (MeV/u)")
ax.set_ylabel(r"$\sigma_{\mathrm{inact}}$ (um$^2$)")
ax.set_title(
    f"Scholz & Kraft 1996 Fig. 3: sigma vs energy (V79)\nstopping power: {STOPPING_POWER_LABEL}"
)
ax.grid(True, ls=":", lw=0.5)
fig3_handles = [
    Line2D([0], [0], color=color, lw=2.0, label=particle)
    for particle, (_z, color) in FIG3_IONS.items()
]
fig3_handles.extend(
    [
        Line2D([0], [0], color="0.3", lw=2.2, label=SK1996_EXPANSION_LABEL),
        Line2D(
            [0], [0], color="0.3", lw=1.4, ls="-.", label="pyRadBio sigma_0: zero-dose reference"
        ),
        Line2D(
            [0],
            [0],
            color="0.3" if SK1996_MC_RESULTS else "none",
            ls="None",
            marker="o" if SK1996_MC_RESULTS else "None",
            ms=4,
            label=(
                SK1996_MC_SIGMA_LABEL
                if SK1996_MC_RESULTS
                else "full MC: run scripts/lem1_mc_reference.py"
            ),
        ),
        Line2D([0], [0], color="0.3", lw=1.3, ls="--", label="paper model line"),
        Line2D([0], [0], color="0.3", ls="None", marker="s", mfc="white", label="paper data"),
    ]
)
ax.legend(handles=fig3_handles, frameon=False, fontsize=9, title="sigma at 1% survival")
fig3.tight_layout()
save_figure(fig3, "sk1996_fig3_sigma_vs_energy.png", directory=FIGURES, dpi=130)

# %% [markdown]
# ## Fig. 4 -- model RBE vs LET at 80/20/1 % survival
#
# The Z-dependent shift of the RBE maximum (~30 / 90 / 300 keV/um for p / He / C).
# The markers use ion doses read directly off the full stochastic survival curves;
# the solid lines are the cluster expansion, with the photon reference taken from
# the shouldered->exponential inversion (`PhotonParameters.dose_for_survival`) like
# the MC points. RBE is a dose ratio, so the geometric bound does not apply here,
# but an inflated ion `beta` still shows: at 80% survival RBE -> `alpha_I/alpha_x`
# and any beta model is fine (within 3%), while at 1% slope matching reached +46%
# for carbon at 1 MeV/u (RBE 1.26 against the model's own 0.86), flipping the sign
# of the overkill crossing that is the point of the 1% series. The cluster expansion
# tracks every p/He/C checkpoint point at all three levels within 4.4% (worst:
# carbon 5 MeV/u at 80%; the 1% series within 1.7%).
#
# **The dotted lines check the paper against itself.** `sigma` and RBE live in
# different figures; at S = 1% both describe the same survival depth, so the paper's
# own sigma curves imply an RBE(1%): carbon from Fig. 3 (needs our LET(E)), helium
# from Fig. 2 `curve1` (already LET; see `paper_sigma_derived_rbe_1pct`). They are
# **upper bounds** -- a slope needs an intercept to give a dose -- whose over-estimate
# is under 1% for carbon above LET 280 but +12.5% at carbon 20 MeV/u and +26% at
# helium 20 MeV/u, the residual shoulder at 1% (`iso_survival_rbe_from_final_slope`).
# Result: derived sits **3-8% above** Fig. 4 for carbon (RMS ~6% over LET 165-595)
# and **1-6% above** for helium (RMS 4.1% over LET 17-67). For helium the
# construction's own bias over that span (+2 to +18%) exceeds the gap, so Figs. 2
# and 4 are consistent within the method's resolution. For carbon the mid-LET excess
# (+7% near LET 250-390, where the bias is only ~1%) is the residual of the known
# Fig. 3 vs Fig. 4 disagreement. Only carbon's line depends on our LET(E), so a
# LET-scale error would show in carbon alone.

# %%
FIG4_IONS = {"proton": (1, "C0"), "helium": (2, "C1"), "carbon": (6, "C2")}
FIG4_LEVELS = (0.80, 0.20, 0.01)
LEVEL_ALPHA = {0.80: 1.0, 0.20: 0.7, 0.01: 0.45}
HAS_SK1996_RBE = any("rbe_80pct" in row for rows in SK1996_MC_RESULTS.values() for row in rows)
FIG4_APPROX_CURVES: dict[tuple[str, float], tuple[np.ndarray, np.ndarray]] = {}

fig4, ax = plt.subplots(figsize=(7.6, 5.6))
for particle, (z, color) in FIG4_IONS.items():
    approx_ions = [sk1996_ion(z, float(energy)) for energy in SK1996_APPROX_ENERGIES]
    approx_lets = np.array([let_of(ion) for ion in approx_ions])
    approx_lq = [lem_v79.alpha_beta(ion) for ion in approx_ions]
    approx_order = np.argsort(approx_lets)
    rows = SK1996_MC_RESULTS.get(particle, [])
    for level in FIG4_LEVELS:
        reference_dose = lem_v79.photon.dose_for_survival(level)
        approx_rbe = np.array(
            [reference_dose / dose_for_survival(level, lq.alpha, lq.beta) for lq in approx_lq]
        )
        sorted_lets = approx_lets[approx_order]
        sorted_rbe = approx_rbe[approx_order]
        FIG4_APPROX_CURVES[(particle, level)] = (sorted_lets, sorted_rbe)
        ax.plot(sorted_lets, sorted_rbe, color=color, lw=2.0, alpha=LEVEL_ALPHA[level])
        tag = LEVEL_TAG[level]
        level_rows = [row for row in rows if f"rbe_{tag}" in row]
        if level_rows:
            lets = np.array([row["let_keV_um"] for row in level_rows])
            rbes = np.array([row[f"rbe_{tag}"] for row in level_rows])
            errors = np.array([row[f"rbe_{tag}_err"] for row in level_rows])
            order = np.argsort(lets)
            ax.errorbar(
                lets[order],
                rbes[order],
                yerr=errors[order],
                color=color,
                ls="None",
                marker="o",
                ms=4,
                capsize=2,
                alpha=LEVEL_ALPHA[level],
            )
        for c in iter_curves(
            figure="fig4_rbe_vs_let", particle=particle, series="calc", survival_level=level
        ):
            ax.plot(c.x, c.y, color=color, lw=1.2, ls="--", alpha=LEVEL_ALPHA[level])

FIG4_DERIVED = paper_sigma_derived_rbe_1pct(
    let_of, reference_dose_Gy=lem_v79.photon.dose_for_survival(0.01)
)
for _particle, (_let, _rbe) in FIG4_DERIVED.items():
    ax.plot(_let, _rbe, color=FIG4_IONS[_particle][1], lw=1.4, ls=":", alpha=LEVEL_ALPHA[0.01])

handles = [plt.Line2D([0], [0], color=c, lw=2.0, label=p) for p, (_z, c) in FIG4_IONS.items()]
handles.extend(
    [
        plt.Line2D([0], [0], color="0.3", lw=2.0, label=SK1996_EXPANSION_LABEL),
        plt.Line2D(
            [0],
            [0],
            color="0.3" if HAS_SK1996_RBE else "none",
            ls="None",
            marker="o" if HAS_SK1996_RBE else "None",
            ms=4,
            label=(
                "pyRadBio full MC"
                if HAS_SK1996_RBE
                else "pyRadBio: run scripts/lem1_mc_reference.py"
            ),
        ),
        plt.Line2D([0], [0], color="0.3", lw=1.2, ls="--", label="paper model line"),
        plt.Line2D(
            [0],
            [0],
            color="0.3",
            lw=1.4,
            ls=":",
            label="paper sigma -> RBE(1%), upper bound (C: Fig. 3, He: Fig. 2)",
        ),
    ]
)
handles.extend(
    plt.Line2D([0], [0], color="0.3", lw=2.0, alpha=LEVEL_ALPHA[level], label=f"S={level:g}")
    for level in FIG4_LEVELS
)
ax.set_xscale("log")
ax.set_xlim(2, 2.0e3)
ax.set_ylim(0, 6.5)
ax.set_xlabel("LET (keV/um)")
ax.set_ylabel("RBE")
ax.set_title(
    "Scholz & Kraft 1996 Fig. 4: RBE vs LET (80/20/1 %; fading = lower survival)"
    f"\nstopping power: {STOPPING_POWER_LABEL}"
)
ax.grid(True, which="both", ls=":", lw=0.5)
ax.legend(handles=handles, frameon=False, fontsize=9)
# Cross-figure consistency, printed so it is a number and not an impression.
for _particle, (_let_d, _rbe_d) in FIG4_DERIVED.items():
    _f4 = list(
        iter_curves(
            figure="fig4_rbe_vs_let", particle=_particle, series="calc", survival_level=0.01
        )
    )
    if not _f4:
        continue
    _fx, _fy = np.asarray(_f4[0].x), np.asarray(_f4[0].y)
    _lo = max(_let_d.min(), _fx.min())
    _hi = min(_let_d.max(), _fx.max())
    _g = np.geomspace(_lo, _hi, 24)
    _ratio = ratio_summary(np.interp(_g, _let_d, _rbe_d), np.interp(_g, _fx, _fy))
    _src = "Fig. 3 sigma(E)" if _particle == "carbon" else "Fig. 2 curve1 sigma(LET)"
    print(
        f"  {_particle:7s} {_src} -> RBE(1%) vs Fig. 4: "
        f"ratio {_ratio.min_ratio:.3f}-{_ratio.max_ratio:.3f} "
        f"(RMS {100 * _ratio.rms_rel:.1f}%) "
        f"over LET {_lo:.0f}-{_hi:.0f} keV/um"
    )
fig4.tight_layout()
save_figure(fig4, "sk1996_fig4_rbe_vs_let.png", directory=FIGURES, dpi=130)

# %% [markdown]
# ## Quantitative summary
#
# The residual tables compare MC checkpoint rows only with the digitized part of the
# paper's own calculated line (`compare_mc_to_paper`): an MC point outside that x
# range is counted as unsupported, never extrapolated. Fig. 3 is interpolated in
# energy, Fig. 4 along its logarithmic LET axis. The jackknife error is an MC
# allocation diagnostic, not a confidence interval on the paper-model residual. The
# Fig. 4 proton RBE_80 deficit (ratio ~0.92) is not a field-point sampling artifact:
# those RBEs move under 1% across a 9x resolution change.

# %%
print(f"\n=== Scholz & Kraft 1996 reconstruction summary ({STOPPING_POWER_LABEL}) ===")

argon_three = next(
    (row for row in SK1996_MC_RESULTS.get("argon", []) if row["energy_MeV_u"] == 3.0),
    None,
)
if argon_three is not None:
    print(
        "Fig. 2 final-slope sigma at Ar 3 MeV/u = "
        f"{argon_three['sigma_1pct_um2']:5.1f} um^2 (paper saturation ~ {a_eff_v79:g})"
    )
else:
    approx_argon_three = lem_v79.approximate_inactivation_cross_section_um2(
        sk1996_ion(18, 3.0), survival_level=0.01
    )
    print(
        "Fig. 2 sigma at Ar 3 MeV/u: "
        f"cluster expansion={approx_argon_three:5.1f} um^2; run MC batch for full result"
    )

print("\nFig. 4 RBE maxima at 80 % survival (paper: p~30, He~90, C~300 keV/um):")
for particle in FIG4_IONS:
    rows = [row for row in SK1996_MC_RESULTS.get(particle, []) if "rbe_80pct" in row]
    if rows:
        peak = max(rows, key=lambda row: row["rbe_80pct"])
        print(
            f"  {particle:7s}: RBE_max = {peak['rbe_80pct']:4.2f} "
            f"at LET = {peak['let_keV_um']:6.1f} keV/um"
        )
    else:
        approx_let, approx_rbe = FIG4_APPROX_CURVES[(particle, 0.80)]
        peak_index = int(np.argmax(approx_rbe))
        print(
            f"  {particle:7s}: expansion RBE_max = {approx_rbe[peak_index]:4.2f} "
            f"at LET = {approx_let[peak_index]:6.1f} keV/um; MC pending"
        )

SK1996_MC_COMPARISONS = compare_mc_to_paper(SK1996_MC_RESULTS) if SK1996_MC_RESULTS else {}
if SK1996_MC_RESULTS:
    precision_rows = [
        (particle, row)
        for particle, rows in SK1996_MC_RESULTS.items()
        for row in rows
        if "precision_converged" in row
    ]
    if precision_rows:
        unconverged = [
            (particle, row) for particle, row in precision_rows if row["precision_converged"] < 0.5
        ]
        print(
            f"\nAdaptive MC precision: {len(precision_rows) - len(unconverged)}/"
            f"{len(precision_rows)} points reached their requested target"
        )
        for particle, row in unconverged:
            print(
                f"  {particle:7s} E={row['energy_MeV_u']:4.1f} MeV/u: "
                f"{100.0 * row['max_relative_error']:5.1f}% maximum relative jackknife error "
                f"after {int(row['n_realizations'])} total histories"
            )

    print("\nFig. 3 full-MC sigma_1% / paper calculation line:")
    if not SK1996_SIGMA_ESTIMATOR_CURRENT:
        print(
            f"  WARNING: CSV predates final-slope estimator v{FINAL_SLOPE_ESTIMATOR_VERSION}; "
            "sigma_1% is retained as a legacy reference until the batch is rerun"
        )
    print(
        "  ion       n  outside  mean ratio  RMS residual  max residual  max MC error  sigma_0 RMS"
    )
    for particle in SIGMA_PARTICLES:
        comparison = SK1996_MC_COMPARISONS[("fig3", particle, None)]
        sigma0_comparison = SK1996_MC_COMPARISONS[("fig3_sigma0", particle, None)]
        worst = max(comparison.points, key=lambda point: abs(point.relative_residual))
        print(
            f"  {particle:7s} {len(comparison.points):2d} {comparison.points_outside_support:8d} "
            f"{comparison.mean_ratio:10.3f} {100.0 * comparison.rms_relative_error:11.1f}% "
            f"{100.0 * comparison.max_abs_relative_error:11.1f}% "
            f"{100.0 * (comparison.max_model_relative_uncertainty or 0.0):10.1f}%"
            f" {100.0 * sigma0_comparison.rms_relative_error:11.1f}%"
            f"  (worst E={worst.x:g} MeV/u, ratio={worst.model_y / worst.paper_y:.2f})"
        )

    print("\nFig. 4 full-MC RBE / paper calculation line:")
    print("  ion       S     n  outside  mean ratio  RMS residual  max residual  max MC error")
    for particle in RBE_PARTICLES:
        for level in FIG4_LEVELS:
            comparison = SK1996_MC_COMPARISONS[("fig4", particle, level)]
            print(
                f"  {particle:7s} {level:4.2f} {len(comparison.points):2d} "
                f"{comparison.points_outside_support:8d} {comparison.mean_ratio:10.3f} "
                f"{100.0 * comparison.rms_relative_error:11.1f}% "
                f"{100.0 * comparison.max_abs_relative_error:11.1f}% "
                f"{100.0 * (comparison.max_model_relative_uncertainty or 0.0):10.1f}%"
            )

# The tally the stopping-power section promises: how often the 1970 table answered
# and at which energies analytic Bethe took over.
if isinstance(STOPPING_POWER, NorthcliffeSchillingWithBetheTail):
    _tail = sorted(STOPPING_POWER.fallback_energies)
    print(
        f"\nStopping-power seam: {STOPPING_POWER.table_hits} LET lookups inside the "
        f"{ENERGY_RANGE_MEV_U[0]:g}-{ENERGY_RANGE_MEV_U[1]:g} MeV/u table; analytic Bethe "
        f"at {len(_tail)} distinct energies outside it"
        + (f" ({_tail[0]:g}-{_tail[-1]:g} MeV/u)" if _tail else "")
    )
