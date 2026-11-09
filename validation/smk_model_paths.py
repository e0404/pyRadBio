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
# # SMK: the same model, four ways
#
# `StochasticMKM` is not one calculation. Three independent switches decide what
# runs:
#
# * **microdosimetry route** — impact-parameter (default) or an injected
#   `SpectrumEngine`: how an ion becomes specific energies at all;
# * **`domain_treatment`** — `"inaniwa_moments"` (default) or `"sato_moments"`:
#   whether the `alpha_0` term carries Sato's `z*_d,F / z_d,F` factor;
# * **`nucleus_treatment`** — `"taylor"` (default) or `"numerical"`: the
#   second-order expansion, or Sato's eq. (7) integral itself.
#
# All are *published* readings of one lineage: Inaniwa & Kanematsu 2018
# (doi:10.1088/1361-6560/aabede) is the treatment-planning adaptation of Sato &
# Furusawa 2012 (doi:10.1667/RR2842.1). There is no "correct" setting, only a
# choice, and this script measures what it costs (HSG, table-1 parameters; the
# shipped defaults are unchanged). Headline:
#
# * `nucleus_treatment`: a **factor of 15 on survival** at 10 Gy, at most
#   **5.7% on `RBE_10`** — a high-dose effect and nothing else;
# * `domain_treatment`: **16% on `RBE_10`** at 650 keV/um, 0.03% at 11 keV/um — a
#   high-LET effect at every dose;
# * the **microdosimetry route**: the largest on `RBE_10`, **-9% to +15%**,
#   changing sign with LET.

# %%
import matplotlib.pyplot as plt
import numpy as np
from _common import pick, quick_mode

from pyradbio.core.radiation import Ion
from pyradbio.micro.lineal_energy import AmorphousTrackEngine, SphericalSite
from pyradbio.models.mkm.spectrum_models import StochasticMKM
from pyradbio.plotting import figures_dir, save_figure
from pyradbio.track.kiefer_chatterjee import KieferChatterjeeRadialDose
from pyradbio.track.stopping_power import SimpleBetheStoppingPower

FIGURES = figures_dir(globals().get("__file__"))
if quick_mode():
    print("QUICK MODE (PYRADBIO_QUICK=1): coarse grids; numbers are not the quoted ones")

STOPPING = SimpleBetheStoppingPower()
# The quick-mode energy subset every panel's LET scan falls back to [MeV/u].
QUICK_ENERGIES = [1.0, 10.0, 100.0]


def carbon(energy_MeV_u: float) -> Ion:
    return Ion(Z=6, A=12, energy_MeV_u=energy_MeV_u)


def let(ion: Ion) -> float:
    return STOPPING.let_keV_um(ion)


def engine(radius_um: float) -> AmorphousTrackEngine:
    """A deterministic amorphous d(y) engine scored in a sphere of ``radius_um``."""
    return AmorphousTrackEngine(
        radial_dose=KieferChatterjeeRadialDose(stopping_power=STOPPING),
        site=SphericalSite(diameter_um=2.0 * radius_um),
        stopping_power=STOPPING,
    )


def sato_dose_mean_z_n(model: StochasticMKM, ion: Ion) -> float:
    """Dose-mean single-event nucleus specific energy from Sato's own density."""
    z, mass = model.nucleus_single_event_density(ion)
    return float(np.sum(z**2 * mass) / np.sum(z * mass))


# All models share the paper's own HSG fit (Inaniwa & Kanematsu 2018 table 1):
# alpha0 = 0.174 /Gy, beta0 = 0.0568 /Gy^2, r_d = 0.28 um, R_n = 8.1 um, z0 = 66 Gy.
TAYLOR = StochasticMKM.hsg_inaniwa2018()
NUMERICAL = StochasticMKM.hsg_inaniwa2018(nucleus_treatment="numerical")
R_D, R_N = TAYLOR.domain_radius_um, TAYLOR.nucleus_radius_um

fig, axes = plt.subplots(2, 3, figsize=(16.5, 9.5))

# %% [markdown]
# ## 1. `nucleus_treatment`: expansion vs. integral
#
# `"taylor"` expands `S_n(z_n)` to second order about `z_n = D` (the `(1 + γD)`
# factor of Inaniwa eq. 24); `"numerical"` evaluates Sato eq. (7),
# `S(D) = ∫ S_n(z_n) f_n(z_n; D) dz_n`, over the compound-Poisson density of
# `pyradbio.micro.multi_event`. The integral is a valid survival at any dose; the
# expansion is not, and says so via `survival_valid_to_Gy` (dotted verticals). The
# two agree while the ceiling is well ahead, and the expansion runs increasingly
# *low* once it is not.

# %%
DOSE = np.linspace(0.05, 12.0, pick(60, 12))

ax = axes[0, 0]
for i, e in enumerate([1.0, 10.0, 100.0]):
    ion = carbon(e)
    colour = f"C{i}"
    ax.plot(DOSE, np.asarray(TAYLOR.survival(ion, DOSE)), "--", color=colour, lw=1.4)
    ax.plot(
        DOSE,
        np.asarray(NUMERICAL.survival(ion, DOSE)),
        "-",
        color=colour,
        lw=1.9,
        label=f"{e:g} MeV/u ({let(ion):.0f} keV/um)",
    )
    ax.axvline(TAYLOR.survival_valid_to_Gy(ion), color=colour, ls=":", lw=1.0)
ax.plot([], [], "k--", lw=1.4, label='"taylor" (default)')
ax.plot([], [], "k-", lw=1.9, label='"numerical"')
ax.plot([], [], "k:", lw=1.0, label="self-reported ceiling")
ax.set_yscale("log")
ax.set_ylim(1e-8, 1.4)
ax.set_xlabel("dose [Gy]")
ax.set_ylabel("survival $S(D)$")
ax.set_title("1. nucleus_treatment: $^{12}$C on HSG")
ax.legend(fontsize=7.5, loc="lower left")
ax.grid(alpha=0.3)

# %% [markdown]
# ## 2. Is the self-reported ceiling any good?
#
# `survival_valid_to_Gy` reports where `γD` reaches 1. Against `D / D_valid` the
# numerical/Taylor ratio collapses a **60x range of LET** onto one curve below the
# ceiling: 1.12–1.16x low at half the ceiling, 1.03–1.05x at a quarter. Above it
# the collapse breaks, *worst at low LET*, where the integral picks up the left
# tail at survival levels near `1e-20` — mathematics, not anything measurable.
# Treat the ceiling as an accuracy contour below 1 and as no bound above it.

# %%
ax = axes[0, 1]
FRACTIONS = np.linspace(0.1, 1.6, pick(24, 6))
for e in pick([1.0, 3.0, 10.0, 30.0, 100.0], QUICK_ENERGIES):
    ion = carbon(e)
    doses = FRACTIONS * TAYLOR.survival_valid_to_Gy(ion)
    ratio = np.asarray(NUMERICAL.survival(ion, doses)) / np.asarray(TAYLOR.survival(ion, doses))
    ax.plot(FRACTIONS, ratio, "-o", ms=2.6, lw=1.4, label=f"{let(ion):.0f} keV/um")
ax.axvline(1.0, color="k", ls=":", lw=1.2)
ax.axhline(1.0, color="k", lw=0.8)
ax.text(1.03, 1.4, "self-reported\nceiling", fontsize=7.5, va="bottom")
ax.set_yscale("log")
ax.set_ylim(0.9, 30.0)
ax.set_xlabel(r"$D\ /\ D_\mathrm{valid}$")
ax.set_ylabel(r"$S_\mathrm{numerical}\ /\ S_\mathrm{taylor}$")
ax.set_title("2. the ceiling collapses a 60x LET range")
ax.legend(fontsize=7.5, title="LET", title_fontsize=7.5)
ax.grid(alpha=0.3)

# %% [markdown]
# ## 3. The switch changes *two* things, not one
#
# Neither paper documents this trap, because neither offers the choice.
# `"taylor"` takes `z_n` from the microdosimetry route (impact-parameter dose mean
# over the nucleus disk by default). `"numerical"` needs a whole density and builds
# it as Sato does, from **LET and the sphere geometry** with a Fermi straggling
# width (PHITS' microdosimetric function is valid only "from 1 nm to 1 um"). The two
# `z_n` differ by up to 1.9x (panel 5), so the split is:
#
# * **(i) which `z_n`** — the expansion on Sato's dose mean: 1.05–1.23x at 10 Gy;
# * **(ii) expansion vs. integral** — same `z_n` on both sides: up to **14.5x**.
#
# (ii) peaks at 162 keV/um, exactly where `alpha` peaks (2.17 /Gy, panel 4): the
# steeper `S(z_n)`, the more the answer rides on the low tail of `f_n(z_n; D)`,
# where an expansion about `z_n = D` is blindest; overkill reconverges them. A large
# change on flipping the switch is almost always (ii), but only this split shows it.

# %%
ax = axes[0, 2]
DECOMP_ENERGIES = pick([1.0, 2.0, 3.0, 5.0, 10.0, 30.0, 100.0, 400.0], QUICK_ENERGIES)
for dose, marker in ((6.0, "o"), (10.0, "s")):
    lets, factor_i, factor_ii = [], [], []
    for e in DECOMP_ENERGIES:
        ion = carbon(e)
        z = TAYLOR.specific_energies(ion)
        shared = {"z_star_1D_Gy": z.z_star_1D_Gy, "z_1D_Gy": z.z_1D_Gy}
        shipped = float(
            TAYLOR.survival_from_specific_energies(dose, z_nucleus_Gy=z.z_nucleus_Gy, **shared)
        )
        matched = float(
            TAYLOR.survival_from_specific_energies(
                dose, z_nucleus_Gy=sato_dose_mean_z_n(NUMERICAL, ion), **shared
            )
        )
        exact = float(np.asarray(NUMERICAL.survival(ion, dose)))
        lets.append(let(ion))
        factor_i.append(matched / shipped)
        factor_ii.append(exact / matched)
    ax.plot(
        lets,
        factor_i,
        marker + "--",
        color="C3",
        ms=4,
        lw=1.3,
        label=f"(i) which $z_n$, {dose:g} Gy",
    )
    ax.plot(
        lets,
        factor_ii,
        marker + "-",
        color="C0",
        ms=4,
        lw=1.6,
        label=f"(ii) expansion vs integral, {dose:g} Gy",
    )
ax.axhline(1.0, color="k", lw=0.8)
ax.set_xscale("log")
ax.set_yscale("log")
ax.set_xlabel("LET [keV/um]")
ax.set_ylabel("survival ratio")
ax.set_title("3. flipping nucleus_treatment moves two things")
ax.legend(fontsize=7)
ax.grid(alpha=0.3)

# %% [markdown]
# ## 4. `domain_treatment`: Sato's frequency factor on `alpha_0`
#
# Both readings give the **identical `beta`**; Sato's eq. (33) multiplies the
# `alpha_0` term by `z*_d,F / z_d,F`, which Inaniwa takes as 1. The ratio is <= 1
# and falls with LET, so `"sato_moments"` lowers `alpha` at high LET (18% at
# 650 keV/um, 0.06% at 11 keV/um). A *frequency* mean needs a real `d(y)`: the
# impact-parameter route has no deposition threshold, so its frequency mean rides
# on the `b_max` cutoff, and `"sato_moments"` refuses to run there:

# %%
try:
    StochasticMKM.hsg_inaniwa2018(domain_treatment="sato_moments").alpha_beta(carbon(10.0))
except ValueError as exc:
    print(f"as designed: {exc}")

# %%
DOM_KWARGS = {"engine": engine(R_D), "nucleus_engine": engine(R_N)}
DOM_INANIWA = StochasticMKM.hsg_inaniwa2018(**DOM_KWARGS)
DOM_SATO = StochasticMKM.hsg_inaniwa2018(domain_treatment="sato_moments", **DOM_KWARGS)

ax = axes[1, 0]
DOM_ENERGIES = pick(
    [1.0, 1.5, 2.0, 3.0, 5.0, 10.0, 20.0, 30.0, 60.0, 100.0, 200.0, 400.0], QUICK_ENERGIES
)
lets, alpha_inaniwa, alpha_sato, ratios = [], [], [], []
for e in DOM_ENERGIES:
    ion = carbon(e)
    lets.append(let(ion))
    alpha_inaniwa.append(DOM_INANIWA.alpha_beta(ion).alpha)
    alpha_sato.append(DOM_SATO.alpha_beta(ion).alpha)
    ratios.append(StochasticMKM.z_star_F_Gy_ratio(DOM_SATO.specific_energies(ion)))
ax.plot(lets, alpha_inaniwa, "o-", color="C0", ms=4, label='"inaniwa_moments" (default)')
ax.plot(lets, alpha_sato, "s-", color="C3", ms=4, label='"sato_moments"')
ax.set_xscale("log")
ax.set_xlabel("LET [keV/um]")
ax.set_ylabel(r"$\alpha$ [Gy$^{-1}$]")
ax.set_title(r"4. domain_treatment: identical $\beta$, lower $\alpha$")
ax.legend(fontsize=7.5, loc="upper left")
ax.grid(alpha=0.3)
twin = ax.twinx()
twin.plot(lets, ratios, "^:", color="C2", ms=4, lw=1.2)
twin.set_ylabel(r"$z^*_{d,F}\,/\,z_{d,F}$  (green)", color="C2")
twin.tick_params(axis="y", labelcolor="C2")
twin.set_ylim(0.0, 1.05)

# %% [markdown]
# ## 5. The microdosimetry route: three different `z_n`
#
# * **impact-parameter** (default): Kiefer–Chatterjee radial dose over the site
#   **disk** (MKM cylinder of radius `r`, length `2r`, track along the axis);
# * **injected `SpectrumEngine`**: a real single-event `d(y)` in a **sphere**;
# * **Sato's Fermi form**: `z_n` from LET and the sphere chord geometry, used only
#   by `"numerical"`.
#
# They agree to ~10% at high LET and diverge to **1.9x** at 400 MeV/u: at low LET
# δ-rays carry energy outside an 8.1 um nucleus, which the two track-structure
# routes count and Sato's `L x chord` form (the full LET along the chord) does not.
# The model being reproduced usually dictates this axis; it is here because it is
# the one people forget is a choice at all.

# %%
ax = axes[1, 1]
ROUTE_ENERGIES = pick(
    [1.0, 2.0, 3.0, 5.0, 10.0, 20.0, 30.0, 60.0, 100.0, 200.0, 400.0], QUICK_ENERGIES
)
IMPACT_SPHERE = StochasticMKM.hsg_inaniwa2018(site_geometry="sphere")
lets, z_disk, z_sphere, z_engine, z_fermi = [], [], [], [], []
for e in ROUTE_ENERGIES:
    ion = carbon(e)
    lets.append(let(ion))
    z_disk.append(TAYLOR.specific_energies(ion).z_nucleus_Gy)
    z_sphere.append(IMPACT_SPHERE.specific_energies(ion).z_nucleus_Gy)
    z_engine.append(DOM_INANIWA.specific_energies(ion).z_nucleus_Gy)
    z_fermi.append(sato_dose_mean_z_n(NUMERICAL, ion))
ax.plot(lets, z_disk, "o-", ms=4, label="impact parameter, disk (default)")
ax.plot(lets, z_sphere, "v-", ms=4, label="impact parameter, sphere")
ax.plot(lets, z_engine, "s-", ms=4, label="injected $d(y)$ engine, sphere")
ax.plot(lets, z_fermi, "^-", ms=4, label='Sato Fermi form (used by "numerical")')
ax.set_xscale("log")
ax.set_yscale("log")
ax.set_xlabel("LET [keV/um]")
ax.set_ylabel(r"$\bar z_n$ [Gy]")
ax.set_title("5. three routes to the same symbol")
ax.legend(fontsize=7)
ax.grid(alpha=0.3)

# %% [markdown]
# ## 6. What survives into `RBE_10`
#
# `RBE_10` inverts each curve at `S = 0.1` (2–4 Gy for carbon), below the
# expansion's ceiling nearly everywhere, so a factor of 15 on the survival curve
# collapses to a few percent:
#
# | axis | `RBE_10` shift | where it peaks |
# |---|---|---|
# | `nucleus_treatment` | -0.2% … -5.7% | highest LET |
# | `domain_treatment` | -0.03% … **-15.8%** | highest LET |
# | microdosimetry route | **-9.1% … +15.5%** | changes sign near 100 keV/um |
#
# The largest is the axis with no name in either paper, selected implicitly by
# whether an `engine` is injected; its sign flip is the disk-vs-sphere crossover of
# panel 5. Judge `nucleus_treatment` at the delivered dose (panel 1 at 6–10 Gy is a
# factor, not a percent); judge the other two anywhere. `rbe_at_survival` honours
# both treatment axes (regression tests in `tests/models/test_smk.py`).

# %%
ax = axes[1, 2]
RBE_ENERGIES = pick(
    [1.0, 2.0, 3.0, 5.0, 10.0, 20.0, 30.0, 60.0, 100.0, 200.0, 400.0], QUICK_ENERGIES
)
lets = [let(carbon(e)) for e in RBE_ENERGIES]
for model, style, label in (
    (TAYLOR, "o-", "taylor + inaniwa (shipped default)"),
    (NUMERICAL, "s--", "numerical + inaniwa"),
    (DOM_INANIWA, "v-", "taylor + inaniwa, d(y) route"),
    (DOM_SATO, "^:", "taylor + sato, d(y) route"),
):
    ax.plot(
        lets,
        [model.rbe_at_survival(carbon(e), 0.1) for e in RBE_ENERGIES],
        style,
        ms=4,
        lw=1.5,
        label=label,
    )
ax.set_xscale("log")
ax.set_xlabel("LET [keV/um]")
ax.set_ylabel("RBE$_{10}$")
ax.set_title("6. RBE$_{10}$ hides exactly one of the three")
ax.legend(fontsize=7.5)
ax.grid(alpha=0.3)

fig.suptitle(
    "SMK model paths: microdosimetry route x domain_treatment x nucleus_treatment "
    "(HSG, Inaniwa & Kanematsu 2018 table 1)",
    fontsize=12,
)
fig.tight_layout(rect=(0, 0, 1, 0.97))
save_figure(fig, "smk_model_paths.png", directory=FIGURES, dpi=150)

# %% [markdown]
# ## Choosing
#
# * **Reproducing Inaniwa & Kanematsu 2018, or a planning system**: the defaults
#   (fast, and `RBE_10` is insensitive to the alternatives).
# * **Reproducing Sato & Furusawa 2012**: `domain_treatment="sato_moments"` *and*
#   `nucleus_treatment="numerical"` with an injected engine; the numerical route
#   scores 0.5% / 2.2% RMS log-ratio (10 / 200 keV/um) against the paper's fig. A1
#   (`pyradbio.benchmark.sato2012`).
# * **Above ~4 Gy per fraction at high LET**: `"numerical"`, or at least keep the
#   dose under half of `survival_valid_to_Gy` (panel 2).
# * **Below ~2 Gy**: `nucleus_treatment` is worth <1% (the other two axes are not
#   affected by that argument); pick it on speed (the
#   numerical route is one FFT pair per dose point, ~10–100x the Taylor cost —
#   milliseconds either way for a survival curve; it matters only inside an
#   optimiser).
