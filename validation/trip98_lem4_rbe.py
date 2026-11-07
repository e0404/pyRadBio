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
# # LEM IV vs TRiP98: initial-slope RBE reproduction
#
# GSI's treatment-planning code TRiP98 ships **LEM IV** reference tables (`*.rbe`)
# giving `RBE_alpha = alpha_ion / alpha_x` versus energy per nucleon for the
# standard ion set (H..Ne), under a photon `(alpha_x, beta_x, D_t, R_nucleus)` set
# fixed by each file's header. This script drives pyRadBio's LEM IV DSB-pattern /
# cluster-index model against one such table, route by route -- the LEM IV
# analogue of `trip98_lem1_rbe.py`. As there, the stopping power is **TRiP's own
# FLUKA dE/dx** (`DEDX/`, an old 2000 H2O table, `I = 77.3 eV`), so any residual is
# the LEM IV biophysics, not an LET mismatch; the table stops at carbon (Z<=6).
#
# **What to expect.** pyRadBio implements the LEM IV *single-particle* model:
# discrete 540 nm giant-loop subvolume scoring (cluster index `C_I` ->
# photon-equivalent dose `D_eq` -> damage-count scaling `kappa` -> `alpha_I`, the
# Elsasser 2010 / Pfuhl 2020 chain the TRiP98 tables themselves use), the
# Friedrich-2015 SSB->DSB yield enhancement `eta`, and the `r_c = 6.5 nm` velocity
# core (Pfuhl 2020). It reproduces the *shape* of `RBE_alpha(LET)` -- a rise to a
# single interior maximum, then an overkill drop -- and largely the TRiP98
# magnitude: the high `RBE_alpha` of the low-energy overkill region (carbon at
# ~600 keV/um) and, via the escaped-energy (indirect cross-section) term, the
# entrance-channel `RBE_alpha ~ 1` at high energy.
#
# **The low-energy band.** Below 2.206 MeV/u (the same energy for every ion, since
# `r_max = 0.062 E^1.7`) the whole track fits inside one giant loop, `C_I`
# saturates (1 - 9e-11 for carbon at 1 MeV/u) and inverting `C_g(D_eq) = C_I`
# cancels. `LEM4.mean_field_deq` instead matches the loop's Poisson mean `m`, the
# same Pfuhl eq. 3 done stably, so the red curve runs all the way down -- and
# overshoots TRiP98 sharply there (carbon most; under the legacy 0.95 cap it falls
# below instead; pass `legacy_trip98_cluster_index_cap=0.95` to see the capped
# curves). Most of that gap is an **extraction mismatch**: TRiP's tables are LQ
# fits to its full simulation (`-iter 50000`), where overkill makes `beta`
# negative and lifts `alpha`. The blue curve
# (`initial_slope_model="full_simulation"`) runs that protocol on our damage model
# and cuts the 0.1-1 MeV/u error roughly in half; it is drawn only up to
# `FULLSIM_MAX_MEV_U` because its cost grows as `(R_eff + r_max)^2` and it buys
# nothing at high energy. Below ~0.5 MeV/u the residual is bounded rather than
# solved: carbon stops *inside* the 10 um nucleus there. Measurements:
# `docs/references/LEM_IV_Friedrich2012.md`, "What actually fixes it: match `m`,
# not `C_I`", "What this costs against TRiP98", "TRiP's alpha is FITTED, not the
# D -> 0 slope" and "What exactly saturates, and the cap reconciliation".
#
# **The survival convention is set explicitly: `survival_convention="mean_effect"`,
# NOT the library default.** LEM's stated formula averages survival, `S = <e^-E>`,
# giving `alpha_I = (1-e^-N_1)/D_1 <= 1/D_1` (Pfuhl 2021 eq. 2.32); TRiP98's tables
# behave like `S = e^-<E>`, i.e. the unsaturated `alpha_I = N_1/D_1`. Both cell
# files share one nucleus radius, so the cell-to-cell ratio isolates the convention:
# from H to Ne the `<e^-E>` ratio misses TRiP's by the saturation factor predicted
# from `N_1` alone while `e^-<E>` matches everywhere; re-tested on the fitted route, where
# `mean_survival` buys a partial sub-MeV "cure" by imposing a saturation TRiP does
# not have -- never pick the convention on RMS (`docs/references/LEM_IV_OPEN_CHOICES.md`
# "2.1 [FORK] Survival convention"; `LEM_IV_Friedrich2012.md` "The survival
# convention is NOT the explanation either"). It stays a trade, so the
# default is unchanged: `mean_effect` has no overkill ceiling, is not the published
# formula and makes the light ions worse, while for Z>=3 it is a uniform win (1-10
# MeV/u median error 6.9-11.8% -> 2.0-3.1%). Above ~10 MeV/u the two are identical
# (`N_1 -> 0`), and they never differ by more than ~12%, because the one-column
# damage-area pinning caps `N_1` near 0.37 for every ion.
#
# Unlike LEM I, LEM IV exposes no rapidLEM slope (`lethal_central` / `lethal_full`
# raise for its DSB-pattern override); the curves are *extraction protocols* over
# one shared `_damage_pattern`. The data is external (`PYRADBIO_TRIP98_DIR`); the
# script exits cleanly without it. Quick mode (`PYRADBIO_QUICK=1`) runs carbon
# only, with low route ceilings and 200 MC realisations, and bypasses the cache.

# %%
from __future__ import annotations

import dataclasses
import hashlib
import math
import os
import time
from dataclasses import dataclass
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
from _common import pick, quick_mode, require_data
from matplotlib.lines import Line2D

import pyradbio
from pyradbio import Ion
from pyradbio.benchmark import reference_tables as rt
from pyradbio.datasets import TRIP98
from pyradbio.models.lem.lem4 import LEM4
from pyradbio.plotting import figures_dir, save_figure

FIGURES = figures_dir(globals().get("__file__"))

# Which TRiP LEM IV table (its header sets alpha_x/beta_x/D_t/R_nucleus) and which
# FLUKA dE/dx file to use. Two LEM IV tables ship: chordoma and spinal cord.
RBE_FILE = os.environ.get("TRIP98_LEM4_RBE_FILE", "_spinal-cord_lemIV.rbe")
DEDX_FILE = os.environ.get("TRIP98_DEDX_FILE", "dEdxFLUKAxTRiP.dedx")
# Per-route energy ceilings [MeV/u]: a route is scored only up to its own ceiling
# and is NaN (absent) above it. Seconds per point, carbon, spinal-cord table, idle
# machine (a busy one once read 7x slower):
#
#   route                          0.2    1.0    10     100    1000 MeV/u
#   analytic, photon-equiv, mean   0.53   0.49   0.31   0.35   0.37
#   analytic, photon-equiv, disc   0.96   1.02   4.43  10.46  10.86   <- capped at 20
#   analytic, photon-equiv, d.MC   4.97   3.42   1.51   1.14   0.75   <- cheapest high
#   analytic, GLOBLE               0.11   0.13   0.17   0.15   0.16
#   fitted, cluster index          3.82   3.93   4.78    --      --   <- capped at 10
#   fitted, photon effect          3.97   4.53   5.03    --      --
#   fitted, GLOBLE                 4.26   4.07   5.05    --      --
#
# Only the routes whose cost *rises* with energy are capped; the sampled cluster
# index gets cheaper (fewer straddling realisations as the track outgrows the
# lattice). Environment variables override every knob (full and quick mode).
FULLSIM_MAX_MEV_U = float(os.environ.get("TRIP98_FULLSIM_MAX_MEV_U", pick("10.0", "0.13")))
DISCRETE_MAX_MEV_U = float(os.environ.get("TRIP98_DISCRETE_MAX_MEV_U", pick("20.0", "0.5")))
# Realisations for cluster_index_model="discrete_mc". 16000 (the library default) is
# what the published scoring used; 4000 keeps this overlay at ~2x the seed noise,
# invisible at this figure's scale.
DISCRETE_MC_REALISATIONS = int(
    os.environ.get("TRIP98_DISCRETE_MC_REALISATIONS", pick("4000", "200"))
)
# The full overlay is hours cold, so its curves are cached (below); quick mode never
# reads or writes that cache, so a quick run cannot overwrite full-mode entries.
USE_CACHE = pick(True, False)

ELEMENT = {1: "H", 2: "He", 3: "Li", 4: "Be", 5: "B", 6: "C"}
ION_Z = pick(tuple(ELEMENT), (6,))
if quick_mode():
    print(
        f"[quick mode] ions Z={ION_Z}, fitted <= {FULLSIM_MAX_MEV_U:g} MeV/u, discrete <= "
        f"{DISCRETE_MAX_MEV_U:g} MeV/u, {DISCRETE_MC_REALISATIONS} MC realisations, no cache; "
        "numbers are not for quoting"
    )

# %%
rbe_table = require_data(rt.load_trip_lem4, RBE_FILE, dataset=TRIP98)
dedx_table = require_data(rt.load_trip_dedx, DEDX_FILE, dataset=TRIP98)

stopping_power = dedx_table.stopping_power()
print(
    f"RBE table : {RBE_FILE}  alpha_x={rbe_table.alpha_x} beta_x={rbe_table.beta_x} "
    f"D_t={rbe_table.D_t} R_n={rbe_table.r_nucleus_um}"
)
print(
    f"dE/dx     : {DEDX_FILE}  material={dedx_table.material} rho={dedx_table.density_g_cm3} "
    f"(ions Z<=6)"
)

# Ions present in BOTH the RBE table and the FLUKA dE/dx table (protons->carbon).
IONS = [(Z, A) for (Z, A) in sorted(rbe_table.curves) if stopping_power.covers(Z, A) and Z in ION_Z]
print("ions:", ", ".join(f"{ELEMENT[Z]}-{A}" for Z, A in IONS))

# %% [markdown]
# ## The routes to `RBE_alpha` drawn here
#
# Two independent axes; of the eight combinations, the **five** that earn their
# place are drawn, all on the photon-equivalent chain the TRiP98 tables use.
#
# - **Extraction** (`initial_slope_model`): `cross_section`, the analytic `D -> 0`
#   slope, or `full_simulation`, an LQ fit to a simulated survival curve over a
#   finite dose window (TRiP98's own protocol; Pfuhl 2021 ch. 2.3.2).
# - **Single-traversal effect.** Analytic routes: `cluster_index_model` picks the
#   `C_I` estimator -- `mean_field` ratio-of-areas, deterministic `discrete`
#   average of per-realisation ratios, or sampled `discrete_mc`. Fitted routes:
#   `local_dose_kernel` on the sampled field -- `cluster_index` (ratio of averages
#   `sum P_cDSB / sum P_dmg` per realisation) or `discrete_cluster_index` (average
#   of ratios `E[N_cDSB/N_dmg]`).
#
# **The two fitted curves are a measurement, not a duplicate** (dashed and dotted,
# dotted on top). Sub-MeV, Z>=3 they part and the discrete convention is *worse*
# (two saturated loop columns give `A/B ~ 1` while the ratio of averages is still
# diluted by penumbra loops). Over 1-10 MeV/u they part too once the fit window is
# `D_t`-relative: the concentration argument bounds the gap in the near-unity
# *ratio*, but what propagates is the gap in `C_I = 1 - u`, ~10x larger, so the two
# are not "identical within one point" (numbers in
# `docs/references/LEM_IV_Friedrich2012.md`, "The LQ fit window"). Forming the ratio
# per realisation does not fix the bias; on a
# *single-traversal* pattern (the analytic pair, ~30x apart sub-MeV) the convention
# matters far more than on a whole-nucleus one.
#
# **Measured and dropped: the three GLOBLE-family routes.** Over both cell files,
# H to C, `analytic, GLOBLE`, `fitted, photon effect` and `fitted, GLOBLE` are all
# several times further from the tables over 1-10 MeV/u than the analytic
# photon-equivalent routes, and `analytic, GLOBLE` is also far off sub-MeV. GLOBLE
# alone is strongly cell-dependent (protons sub-MeV), and it is
# rejected as a *replacement* (same `_damage_pattern`, same
# `RBE_alpha ~ 1/LET` degeneracy) -- consistent with Friedrich 2012 (Radiat. Res.
# 178(5):385-394) never applying it to ions. Add `Route`s with
# `effect_model="globle"` / `local_dose_kernel=...` to draw them. **Not curves:**
# `survival_convention` (a convention, pinned above);
# `local_dose_overkill` (bit-identical to the default on the `cross_section` path at
# all three probe energies); `discrete_dsb_lethal_events` / `local_dose_lethal_events`
# (they return `N(D)` on a dose grid, cross-checks rather than extraction protocols).

# %%
photon = rbe_table.photon_parameters()
cell = rbe_table.cell_target()
# One field drives both extraction routes, so the routes never disagree about the
# convention while disagreeing about the method.
_COMMON = dict(
    photon=photon,
    cell=cell,
    stopping_power=stopping_power,
    survival_convention="mean_effect",
)


@dataclass(frozen=True)
class Route:
    """One way of computing ``RBE_alpha``, with its plot style and energy ceiling."""

    label: str
    colour: str
    linestyle: str
    model: LEM4
    e_max: float = float("inf")


ROUTES: list[Route] = [
    # -- analytic D->0 slope: the effect model, then the cluster-index estimator ----
    Route(
        "analytic, photon-equiv, mean field",
        "C3",
        "-",
        LEM4(**_COMMON, effect_model="photon_equivalent"),  # type: ignore[arg-type]
    ),
    Route(
        "analytic, photon-equiv, discrete",
        "C1",
        "-",
        LEM4(**_COMMON, effect_model="photon_equivalent", cluster_index_model="discrete"),  # type: ignore[arg-type]
        DISCRETE_MAX_MEV_U,
    ),
    Route(
        "analytic, photon-equiv, discrete MC",
        "C2",
        "-",
        LEM4(  # type: ignore[arg-type]
            **_COMMON,
            effect_model="photon_equivalent",
            cluster_index_model="discrete_mc",
            cluster_index_mc_realisations=DISCRETE_MC_REALISATIONS,
        ),
    ),
    # -- fitted slope (TRiP's protocol), over the same photon-equivalent chain ------
    Route(
        "fitted, cluster index",
        "C0",
        "--",
        LEM4(**_COMMON, initial_slope_model="full_simulation", local_dose_kernel="cluster_index"),  # type: ignore[arg-type]
        FULLSIM_MAX_MEV_U,
    ),
    # Dotted and drawn ON TOP of the dashed one, so the reader sees where they agree.
    Route(
        "fitted, discrete cluster index",
        "C5",
        ":",
        LEM4(  # type: ignore[arg-type]
            **_COMMON,
            initial_slope_model="full_simulation",
            local_dose_kernel="discrete_cluster_index",
        ),
        FULLSIM_MAX_MEV_U,
    ),
]


def rbe_alpha_curve(route: Route, Z: int, A: int, energies: np.ndarray) -> np.ndarray:
    """``RBE_alpha`` for one route, NaN above its ceiling and where undefined.

    At very low energy the track model can have no delta-ray range
    (``r_max < r_min``), and an estimator's cluster index can fall outside the
    ``C_g`` bracket; those points raise and become NaN, so each curve simply
    starts where its own route is defined rather than aborting the overlay.
    """
    out = np.full(energies.shape, np.nan)
    for i, e in enumerate(energies):
        if e > route.e_max:
            continue
        try:
            out[i] = route.model.initial_slope(Ion(Z, A, float(e))) / photon.alpha_x
        except (ValueError, ZeroDivisionError, RuntimeError):
            pass
    return out


# Every curve is computed once and cached to disk (hours cold, seconds warm). Delete
# the .npz, or set TRIP98_NO_CACHE=1, to force a recompute.
CACHE = FIGURES / f"lem4_trip98_routes_{Path(RBE_FILE).stem.lstrip('_')}.npz"
_cached: dict[str, np.ndarray] = {}
if USE_CACHE and CACHE.exists() and not os.environ.get("TRIP98_NO_CACHE"):
    with np.load(CACHE, allow_pickle=False) as z:
        _cached = {k: z[k] for k in z.files}


def _library_fingerprint() -> str:
    """Digest of the library sources that can change a number in this figure.

    The key below is otherwise built from the routes' *fields*, which cannot see a
    change to an algorithm: a rewrite of
    `analytic_discrete_cluster_index` that touches no field would redraw every
    `discrete` route from cache with stale numbers. Hashing the
    sources makes a library edit invalidate the cache automatically -- the safe
    direction to be wrong in.
    """
    digest = hashlib.sha256()
    root = Path(pyradbio.__file__).resolve().parent
    for sub in ("models/lem", "core", "track"):
        for path in sorted((root / sub).glob("*.py")):
            digest.update(path.read_bytes())
    # The TRiP dE/dx adapter feeds every LET in these curves.
    digest.update((root / "benchmark" / "reference_tables.py").read_bytes())
    return digest.hexdigest()[:12]


_LIB_FINGERPRINT = _library_fingerprint()


def _cache_key(route: Route) -> str:
    """Everything that can change this route's numbers, as one string."""
    fields = dataclasses.asdict(route.model)
    fields.pop("stopping_power", None)  # not hashable, and fixed by DEDX_FILE
    return "|".join(
        [route.label, f"emax={route.e_max:g}", DEDX_FILE, f"lib={_LIB_FINGERPRINT}"]
        + [f"{k}={v}" for k, v in sorted(fields.items()) if not isinstance(v, (dict, list))]
    )


PRED: dict[str, dict[tuple[int, int], np.ndarray]] = {}
# Entries of other routes/ceilings computed on this library survive a narrower run
# (e.g. reduced TRIP98_* knobs); entries of an older library are dropped.
_to_save: dict[str, np.ndarray] = {
    k: v for k, v in _cached.items() if f"|lib={_LIB_FINGERPRINT}" in k
}
for _route in ROUTES:
    _t0 = time.perf_counter()
    _key = _cache_key(_route)
    _hit = _cached.get(_key)
    if _hit is not None and _hit.shape[0] == len(IONS):
        PRED[_route.label] = {ion: _hit[i] for i, ion in enumerate(IONS)}
        _how = "cached"
    else:
        PRED[_route.label] = {
            (Z, A): rbe_alpha_curve(_route, Z, A, np.array(rbe_table.curve(Z, A).energies_MeV_u))
            for Z, A in IONS
        }
        _how = f"{time.perf_counter() - _t0:.1f} s"
    _to_save[_key] = np.array([PRED[_route.label][ion] for ion in IONS])
    _n = sum(int(np.isfinite(v).sum()) for v in PRED[_route.label].values())
    print(f"  {_route.label:38s} {_n:4d} points  {_how:>10s}", flush=True)
if USE_CACHE:
    np.savez_compressed(CACHE, **_to_save)

# %% [markdown]
# ## Overlay: every route vs TRiP98, per ion
#
# Log-log axes: TRiP's `RBE_alpha` spans ~1 (fast, low LET) to tens (slow, high
# LET). **Solid = analytic `D -> 0` slope, dashed/dotted = fitted to a simulated
# curve**; each curve stops at its own ceiling. All routes converge above ~20 MeV/u,
# where every effect model is linear and the extraction protocols agree by
# construction, and fan out below ~3 MeV/u, where the effect model -- not the
# shared track structure -- decides the answer.

# %%
ncols = 3
nrows = math.ceil(len(IONS) / ncols)
fig, axes = plt.subplots(nrows, ncols, figsize=(4.4 * ncols, 3.5 * nrows), squeeze=False)
for ax, (Z, A) in zip(axes.flat, IONS, strict=False):
    curve = rbe_table.curve(Z, A)
    e = np.array(curve.energies_MeV_u)
    ax.plot(
        e, curve.rbe_alpha, ls="None", marker="o", ms=4, mfc="white", color="0.3", label="TRiP98"
    )
    for route in ROUTES:
        ax.plot(e, PRED[route.label][(Z, A)], color=route.colour, lw=1.8, ls=route.linestyle)
    ax.set_xscale("log")
    ax.set_yscale("log")
    ax.set_title(f"{ELEMENT[Z]}-{A}")
    ax.set_xlabel("Energy (MeV/u)")
    ax.grid(True, which="both", ls=":", lw=0.4)
axes.flat[0].set_ylabel(r"$RBE_\alpha = \alpha_{ion}/\alpha_x$")
for ax in axes.flat[len(IONS) :]:
    ax.set_visible(False)
handles = [
    Line2D([0], [0], color="0.3", ls="None", marker="o", mfc="white", label="TRiP98 LEM IV"),
    *(
        Line2D(
            [0],
            [0],
            color=r.colour,
            lw=1.8,
            ls=r.linestyle,
            label=r.label + ("" if math.isinf(r.e_max) else f" (E<={r.e_max:g})"),
        )
        for r in ROUTES
    ),
]
# The legend sits between the title block and the axes, so both anchors follow the
# number of legend rows (adding a route once slid the legend into the subtitle).
_legend_rows = math.ceil((len(ROUTES) + 1) / 3)
fig.legend(handles=handles, loc="upper center", ncol=3, frameon=False, bbox_to_anchor=(0.5, 1.0))
fig.suptitle(
    f"LEM IV: {len(ROUTES)} routes to RBE_alpha vs TRiP98 ({RBE_FILE}); "
    "stopping power: TRiP FLUKA dE/dx.\n"
    "Solid = analytic D->0 slope, dashed = fitted to a simulated survival curve "
    "(TRiP's own protocol).",
    y=1.04 + 0.035 * _legend_rows,
)
fig.tight_layout()
_stem = Path(RBE_FILE).stem.lstrip("_")
save_figure(fig, f"lem4_trip98_rbe_{_stem}.png", directory=FIGURES, dpi=130, bbox_inches="tight")

# %% [markdown]
# ## The LEM IV mechanism, for carbon
#
# One traversal maps to a photon-equivalent dose `D_eq` by matching the track's
# cluster index `C_I = N_cDSB / (N_iDSB + N_cDSB)` to the photon `C_gamma(D)`. Both
# rise with LET (a denser track clusters more giant loops), so `D_eq` and `kappa`
# grow, driving the single-track effect `N_1 = kappa * (-ln S_g(D_eq))` and hence
# `RBE_alpha`.

# %%
carbon = rbe_table.curve(6, 12)
ce = np.array(carbon.energies_MeV_u)
lem = ROUTES[0].model  # the default route: analytic slope, photon-equivalent, mean field
lets, c_index, d_eq = [], [], []
for energy in ce:
    ion = Ion(6, 12, float(energy))
    try:
        ci = lem.cluster_index(ion)
        # mean_field_deq, not photon_equivalent_dose(ci): below 2.206 MeV/u the two
        # are the same equation but only the former is free of cancellation, so this
        # panel shows the D_eq the RBE path actually used.
        de = lem.mean_field_deq(ion)
    except (ValueError, ZeroDivisionError, RuntimeError):
        continue  # keep the three lists in lockstep: append only on full success
    lets.append(stopping_power.let_keV_um(ion))
    c_index.append(ci)
    d_eq.append(de)

fig2, (axc, axd) = plt.subplots(1, 2, figsize=(9.0, 3.6))
axc.plot(lets, c_index, color="C0", marker=".", lw=1.6)
axc.set_xscale("log")
axc.set_xlabel("LET (keV/um)")
axc.set_ylabel(r"cluster index $C_I$")
axc.set_title("Ion cluster index vs LET (C-12)")
axc.grid(True, which="both", ls=":", lw=0.4)
axd.plot(lets, d_eq, color="C1", marker=".", lw=1.6)
axd.set_xscale("log")
axd.set_yscale("log")
axd.set_xlabel("LET (keV/um)")
axd.set_ylabel(r"photon-equivalent dose $D_{eq}$ (Gy)")
axd.set_title("Photon-equivalent dose vs LET (C-12)")
axd.grid(True, which="both", ls=":", lw=0.4)
fig2.suptitle(f"LEM IV single-particle diagnostics for carbon ({RBE_FILE})")
fig2.tight_layout()
save_figure(
    fig2, f"lem4_trip98_diagnostics_{_stem}.png", directory=FIGURES, dpi=130, bbox_inches="tight"
)

# %% [markdown]
# ## Quantitative agreement (median |relative error| in RBE_alpha)
#
# One row per route, one column per energy window, so routes are ranked *within* a
# band: no route wins everywhere, and an aggregate would be dominated by the sub-MeV
# points. `n` is per route -- each is scored only where it is defined, so a route
# with a low ceiling is compared on fewer points.

# %%
WINDOWS = {
    "sub-MeV (<=1)": (0.0, 1.0),
    "low-E (1-10)": (1.0, 10.0),
    "full (>=1 MeV/u)": (1.0, 1.0e9),
    "therapeutic (20-350)": (20.0, 350.0),
    "high-E tail (>=100)": (100.0, 1.0e9),
}
print(f"\n=== RBE_alpha agreement vs TRiP98 LEM IV ({RBE_FILE}) ===")
print("    median |relative error|, per route and energy window (n = points scored)")
_W = max(len(r.label) for r in ROUTES)
for Z, A in IONS:
    curve = rbe_table.curve(Z, A)
    ref = np.array(curve.rbe_alpha)
    print(f"\n{ELEMENT[Z]}-{A}:")
    print(f"  {'route':{_W}s} " + " ".join(f"{lab:>21s}" for lab in WINDOWS))
    for route in ROUTES:
        scores = rt.windowed_median_errors(
            curve.energies_MeV_u, PRED[route.label][(Z, A)], ref, WINDOWS
        )
        cells = [
            f"{'--':>21s}" if score is None else f"{score[0]:14.2%} (n={score[1]:2d})"
            for score in scores.values()
        ]
        print(f"  {route.label:{_W}s} " + " ".join(cells))

print("\nDone. Figures written to", FIGURES)
