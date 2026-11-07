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
# # LEM I variants: the approximation ladder for alpha and beta
#
# "LEM I" is a family: the explicit Local Effect Model (Scholz & Kraft 1994/1996)
# plus successively faster approximations built for treatment planning. pyRadBio
# exposes the choices on **two independent axes**:
#
# * **alpha** (`initial_slope_model`):
#     - `cross_section` (default): the EXACT low-dose slope, the full
#       impact-parameter integral of `1-exp(-nu(b))`; it keeps intra-track overkill
#       (thins down at very high LET) and recovers `alpha_x` at low LET.
#     - `lethal_central_saturating`: the KS2006 eq. (9) central slope
#       `(1-e^-N_1)/D_1` from one *central* traversal. It thins down at high LET but
#       under-shoots `alpha_x` at high energy because it misses the large-`b`
#       traversals (0.113 vs `alpha_x` = 0.18 for protons at 400 MeV/u).
#       `LEM1.scholz1997_central_initial_slope` adds back the missed tracks'
#       photon-rate contribution `alpha_x (1-f)`, which is what Scholz 1997 Fig. 1
#       plots; the two are identical below ~15 MeV/u, where `f = 1`.
#     - `lethal_central` / `lethal_full`: rapidLEM/TRiP (Kramer & Scholz 2006) with
#       overkill *dropped*; `lethal_full` keeps the full impact integral and
#       reproduces the TRiP98 RBE tables.
# * **beta** (`beta_model`): `scholz1997_montecarlo` (the exact stochastic beta),
#   `scholz1997_two_track` (its deterministic, MC-validated reconstruction),
#   `scholz1997_stochastic` (the 1997 z-domain `beta_z e^-2N_1`), `rapid_lem`
#   (KS2006 eq. (10), `f^2 beta_z` with `f = (1-e^-N_1)/N_1`) and `slope_matching`
#   (`(s_max - alpha)/(2 D_t)`, fast, the TRiP default).
#
# **Pair alpha and beta from the same tier.** Mixing an exact alpha with a fast beta
# is not a model anyone published. The historically real bundles:
#
# | Bundle                | alpha                       | beta                    | Origin        |
# |-----------------------|-----------------------------|-------------------------|---------------|
# | Explicit LEM (exact)  | `cross_section`             | `scholz1997_two_track`  | Scholz-Kr 96  |
# | Scholz 1997 (fast)    | `scholz1997_central`        | `scholz1997_stochastic` | Scholz 1997   |
# | rapidLEM central 2006 | `lethal_central_saturating` | `rapid_lem`             | Kramer-Sch 06 |
# | rapidLEM/TRiP98 full  | `lethal_full`               | `slope_matching`        | Kramer-Sch 06 |
#
# The Scholz 1997 and KS2006-central bundles share their alpha below ~15 MeV/u and
# differ in how beta is derived. The Monte-Carlo points that validate the exact
# bundle are drawn in `validation/scholz1997_lem1.py` (Figs. 1 and 3).
#
# **Caveat on the pyRadBio default.** `LEM1()` defaults to `cross_section` alpha +
# `slope_matching` beta, an *incoherent* mix: slope matching assumes the capped
# rapidLEM alpha, so fed the exact alpha it over-steepens carbon beta and blows up
# below ~5 MeV/u. Prefer a coherent bundle for quantitative work. Cell: the Scholz
# 1997 CHO inputs; analytic Bethe stopping power keeps the example dependency-free.

# %%
from __future__ import annotations

from collections.abc import Callable

import matplotlib.pyplot as plt
import numpy as np
import numpy.typing as npt
from matplotlib.lines import Line2D

from pyradbio import Ion
from pyradbio.benchmark.scholz1997 import TWO_TRACK_FAST_QUADRATURE, cho_lem1
from pyradbio.plotting import figures_dir, save_figure
from pyradbio.track.stopping_power import SimpleBetheStoppingPower

FIGURES = figures_dir(globals().get("__file__"))
SPECIES = {"proton": (1, 1, "C0"), "carbon": (6, 12, "C1")}
SP = SimpleBetheStoppingPower()

# One base model exposes every alpha/beta method regardless of its own flags; it is
# also the pyRadBio default (cross_section + slope_matching).
_base = cho_lem1(SP)
_tt = cho_lem1(SP, beta_model="scholz1997_two_track")
_zd = cho_lem1(SP, beta_model="scholz1997_stochastic")
_sm_full = cho_lem1(SP, initial_slope_model="lethal_full", beta_model="slope_matching")
photon = _base.photon
print(
    f"CHO: alpha_x={photon.alpha_x} beta_x={photon.beta_x} D_t={photon.D_t} "
    f"s_max={photon.max_slope():.3f}"
)


def _two_track_beta(ion: Ion) -> float:
    # The heavy quadrature, at reduced (still <1%-converged) node counts.
    return _tt.scholz1997_two_track_beta(ion, **TWO_TRACK_FAST_QUADRATURE)


BUNDLES: list[tuple[str, str, Callable[[Ion], float], Callable[[Ion], float]]] = [
    ("Explicit LEM (exact)", "0.15", _base.initial_slope, _two_track_beta),
    (
        "Scholz 1997 (fast)",
        "C0",
        _base.scholz1997_central_initial_slope,
        lambda i: _zd.alpha_beta(i).beta,
    ),
    (
        "rapidLEM central (KS2006)",
        "C2",
        _base.lethal_central_saturating_initial_slope,
        _base.rapid_lem_beta,
    ),
    (
        "rapidLEM / TRiP98 (full)",
        "C1",
        _base.lethal_full_initial_slope,
        lambda i: _sm_full.alpha_beta(i).beta,
    ),
]
ENERGIES = np.geomspace(1.5, 400.0, 20)


def curve(
    fn: Callable[[Ion], float], Z: int, A: int, energies: npt.ArrayLike = ENERGIES
) -> np.ndarray:
    return np.array([fn(Ion(Z, A, float(e))) for e in np.asarray(energies)])


# %% [markdown]
# ## alpha(E) and beta(E) per bundle, for proton and carbon

# %%
fig, axes = plt.subplots(2, 2, figsize=(11.5, 8), sharex=True)
for col, (particle, (Z, A, _c)) in enumerate(SPECIES.items()):
    ax_a, ax_b = axes[0, col], axes[1, col]
    for label, color, alpha_fn, beta_fn in BUNDLES:
        # The KS2006-central bundle shares the Scholz-1997 alpha below ~15 MeV/u; draw
        # its alpha dotted so the coincident lines stay distinguishable.
        alpha_style = ":" if label.startswith("rapidLEM central") else "-"
        ax_a.plot(ENERGIES, curve(alpha_fn, Z, A), color=color, lw=2.0, ls=alpha_style, label=label)
        ax_b.plot(ENERGIES, curve(beta_fn, Z, A), color=color, lw=2.0, label=label)
    ax_b.plot(
        ENERGIES,
        curve(lambda i: _base.alpha_beta(i).beta, Z, A),
        color="C3",
        lw=1.4,
        ls="--",
        label="pyRadBio default (incoherent)",
    )
    ax_a.set_title(particle.capitalize())
    ax_a.axhline(photon.max_slope(), color="0.6", lw=0.8, ls=":")
    ax_b.axhline(photon.beta_x, color="0.6", lw=0.8, ls=":")
    for ax in (ax_a, ax_b):
        ax.set_xscale("log")
        ax.grid(True, which="both", ls=":", lw=0.4)
    ax_b.set_xlabel("Energy (MeV/u)")
    ax_b.set_ylim(0.0, 0.032)
axes[0, 0].set_ylabel(r"$\alpha$ (Gy$^{-1}$)")
axes[1, 0].set_ylabel(r"$\beta$ (Gy$^{-2}$)")
handles = [Line2D([0], [0], color=c, lw=2.0, label=lbl) for lbl, c, _, _ in BUNDLES]
handles.append(Line2D([0], [0], color="C3", lw=1.4, ls="--", label="pyRadBio default (incoherent)"))
fig.legend(handles=handles, loc="upper center", ncol=3, frameon=False, bbox_to_anchor=(0.5, 1.0))
fig.suptitle(
    "LEM I variants (CHO): alpha (top) and beta (bottom); historically-coherent bundles", y=0.93
)
fig.tight_layout(rect=(0, 0, 1, 0.9))
save_figure(fig, "lem1_variants.png", directory=FIGURES, dpi=130, bbox_inches="tight")

# %%
print("\n=== carbon: alpha / beta by bundle (Bethe SP, CHO) ===")
for e in (3.0, 8.0, 50.0, 200.0):
    ion = Ion(6, 12, e)
    print(f"  E={e:6.1f} MeV/u")
    for lbl, _c, alpha_fn, beta_fn in BUNDLES:
        print(f"    {lbl:27s} alpha={alpha_fn(ion):.3f}  beta={beta_fn(ion):.4f}")

# %% [markdown]
# ## beta: slope matching vs the exact two-track (and the KS2006 f^2 ansatz)
#
# TRiP98 / rapidLEM derive the ion `beta` from `alpha` by **slope matching**, the
# assumption that the ion LQ curve reaches the photon maximum slope
# `s_max = alpha_x + 2 beta_x D_t` at `D_t`: `beta_I = (s_max - alpha_I)/(2 D_t)`.
# It depends on which alpha it is fed: the exact `cross_section` alpha thins down at
# high LET, so `s_max - alpha` is large and the inferred curvature large; the
# rapidLEM `lethal_full` alpha caps at `s_max`, so `beta -> 0`.
#
# Against the two-track beta (protons barely overkill and agree to a few %): at
# therapeutic mid energy (~8-50 MeV/u) both slope-matching betas over-steepen carbon
# by ~2x -- sparse carbon tracks overlap little, so the true curvature is much
# smaller -- while `rapid_lem`, a suppression of the intrinsic `beta_z`, tracks it
# far better. The alpha model only governs the very-low-energy limit (<~5 MeV/u):
# the exact alpha makes slope matching blow up, `lethal_full` sends `beta -> 0`; the
# rapidLEM matched pair does not fix the mid-energy over-steepening. For quantitative
# carbon beta use `scholz1997_two_track` (or the MC). The medians below skip
# near-zero two-track values, where a relative error is meaningless.


# %%
def median_rel_err(pred: np.ndarray, ref: np.ndarray, floor: float = 1e-3) -> float:
    mask = ref > floor
    return (
        float(np.median(np.abs(pred[mask] - ref[mask]) / ref[mask])) if mask.any() else float("nan")
    )


def beta_rows(Z: int, A: int, energies: np.ndarray) -> list[np.ndarray]:
    """two-track, rapid_lem, slope-match (cross_section alpha), slope-match (lethal_full)."""
    return [
        curve(_two_track_beta, Z, A, energies),
        curve(_base.rapid_lem_beta, Z, A, energies),
        curve(lambda i: _base.alpha_beta(i).beta, Z, A, energies),
        curve(lambda i: _sm_full.alpha_beta(i).beta, Z, A, energies),
    ]


print("\n=== fast beta vs two-track beta (median |rel err|, beta>1e-3) ===")
ENERGIES_B = np.geomspace(2.0, 400.0, 16)
for particle, (Z, A, _c) in SPECIES.items():
    b_tt, b_rl, b_ex, b_rp = beta_rows(Z, A, ENERGIES_B)
    print(
        f"  {particle:7s}  rapid_lem={median_rel_err(b_rl, b_tt):6.1%}   "
        f"SM/cross_section={median_rel_err(b_ex, b_tt):6.1%}   "
        f"SM/lethal_full={median_rel_err(b_rp, b_tt):6.1%}"
    )

print("\nLow-energy carbon spot check (two-track ~ 0, slope-match/exact over-steepens):")
carbon_E = np.array([2.0, 4.0, 8.0, 20.0])
print("  E(MeV/u)  two-track   rapid_lem   SM/cross_section   SM/lethal_full")
for e, t, rl, x, r in zip(carbon_E, *beta_rows(6, 12, carbon_E), strict=True):
    print(f"  {e:7.0f}  {t:9.5f}  {rl:10.5f}   {x:16.5f}   {r:14.5f}")

print("\nDone. Figures written to", FIGURES)
