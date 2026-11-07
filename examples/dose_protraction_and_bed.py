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
# # The time axis: protraction, the Lea-Catcheside $G$, BED and EQD2
#
# Everything else in pyRadBio computes an *acute* $\alpha$/$\beta$ pair. Spread
# the same dose over time and sublethal lesions are repaired before they can
# pair up, so only the quadratic term is lost:
#
# $$-\ln S(D) = \alpha D + G\,\beta D^2$$
#
# $G$ is a property of the **timing alone** — not of the cell and not of the
# radiation quality — which is why `pyradbio.core.protraction` computes it from
# a `DoseSchedule` that carries no biology at all. This example shows the two
# classical limits, what they do to a survival curve, and why the fraction-size
# isoeffect follows directly.
#
# This file is a [jupytext](https://jupytext.readthedocs.io) percent script: run
# it directly with `python`, or pair/open it as a notebook (see
# `examples/README.md`).

# %%
import matplotlib.pyplot as plt
import numpy as np

from pyradbio import DoseSchedule, bed, eqd2, lea_catcheside_g, survival
from pyradbio.plotting import figures_dir, save_figure

FIGURES = figures_dir(globals().get("__file__"))

# %% [markdown]
# A repair half-time has to be chosen explicitly — the library ships no default:
# it is the single most influential number here, and the literature spans
# roughly 0.2–5 h. 1.5 h is a common working value for late-responding tissue.

# %%
T_HALF_H = 1.5
ALPHA, BETA = 0.3, 0.03  # alpha/beta = 10 Gy

# %% [markdown]
# ## 1. Fractionation: $G$ between the two bounds
#
# $n$ equal fractions run from $G = 1$ (no repair, i.e. one acute dose) down to
# $G = 1/n$ (complete repair between fractions). A daily schedule is essentially
# at the lower bound; the interesting region is the few hours in between, which
# is where twice-a-day schedules live.

# %%
intervals_h = np.geomspace(0.05, 48.0, 200)
fig, axes = plt.subplots(2, 2, figsize=(11, 8.5))

ax = axes[0, 0]
for n, color in ((2, "C0"), (5, "C1"), (30, "C2")):
    g = [
        lea_catcheside_g(
            DoseSchedule.fractionated(2.0, n_fractions=n, interval_h=float(t)),
            repair_half_time_h=T_HALF_H,
        )
        for t in intervals_h
    ]
    ax.semilogx(intervals_h, g, color=color, label=f"n = {n}")
    ax.axhline(1.0 / n, color=color, ls=":", lw=1)
ax.axvline(6.0, color="grey", ls="--", lw=1)
ax.annotate("6 h\n(BID)", (6.0, 0.75), fontsize=8, color="grey", ha="center")
ax.set_xlabel("Interval between fractions [h]")
ax.set_ylabel("Lea-Catcheside $G$")
ax.set_title(f"Fractionation ($T_{{1/2}}$ = {T_HALF_H} h)\ndotted lines: the $1/n$ bound")
ax.set_ylim(0.0, 1.05)
ax.legend()

# %% [markdown]
# ## 2. Continuous irradiation
#
# One block at constant dose rate gives $G = 2(x - 1 + e^{-x})/x^2$ with
# $x = \lambda T$ — the low-dose-rate / brachytherapy form, which falls onto its
# $2/(\lambda T)$ asymptote once delivery is slow compared with repair. The
# library evaluates the *general* double integral; this closed form is not coded
# separately, it just drops out.

# %%
durations_h = np.geomspace(0.01, 500.0, 300)
g_continuous = [
    lea_catcheside_g(
        DoseSchedule.continuous(10.0, duration_h=float(t)), repair_half_time_h=T_HALF_H
    )
    for t in durations_h
]
lam = np.log(2.0) / T_HALF_H

ax = axes[0, 1]
ax.loglog(durations_h, g_continuous, "C3", label="continuous, exact")
ax.loglog(durations_h, 2.0 / (lam * durations_h), "k:", lw=1, label=r"$2/(\lambda T)$ asymptote")
ax.set_xlabel("Delivery time $T$ [h]")
ax.set_ylabel("Lea-Catcheside $G$")
ax.set_title("Continuous irradiation")
ax.set_ylim(1e-3, 2.0)
ax.legend()

# %% [markdown]
# ## 3. What it does to a survival curve
#
# The same total dose, delivered four ways. Only the shoulder moves: at $D \to 0$
# every curve has the same initial slope $\alpha$, because protraction cannot
# touch the linear term.

# %%
doses = np.linspace(0.0, 12.0, 300)
schedules = {
    "acute (single dose)": DoseSchedule.acute,
    "5 fractions, 6 h apart": lambda d: DoseSchedule.fractionated(
        d / 5.0, n_fractions=5, interval_h=6.0
    ),
    "5 fractions, daily": lambda d: DoseSchedule.fractionated(
        d / 5.0, n_fractions=5, interval_h=24.0
    ),
    "continuous over 24 h": lambda d: DoseSchedule.continuous(d, duration_h=24.0),
}

ax = axes[1, 0]
for (label, build), style in zip(schedules.items(), ("-", "--", "-.", ":"), strict=True):
    # G is dose-independent for a fixed schedule shape, so it is computed once.
    g = lea_catcheside_g(build(10.0), repair_half_time_h=T_HALF_H)
    s = [survival(float(d), ALPHA, BETA, g=g) for d in doses]
    ax.semilogy(doses, s, style, label=f"{label}  ($G$ = {g:.3f})")
    print(f"{label:<24s} G = {g:.4f}")
ax.set_xlabel("Total dose [Gy]")
ax.set_ylabel("Surviving fraction")
ax.set_ylim(1e-4, 1.0)
ax.set_title(
    rf"Survival, $\alpha$={ALPHA}, $\beta$={BETA} ($\alpha/\beta$ = {ALPHA / BETA:.0f} Gy)"
)
ax.legend(fontsize=8)

# %% [markdown]
# ## 4. The fraction-size isoeffect
#
# Take schedules that all deliver the same **tumour** effect (EQD2 = 60 Gy at
# $\alpha/\beta$ = 10) and ask what each costs in late-responding tissue
# ($\alpha/\beta$ = 3). Fewer, larger fractions cost steeply more — the whole
# content of Withers, Thames & Peters (1983), and the reason hypofractionation
# is an argument rather than a free lunch.

# %%
TUMOUR_AB, LATE_AB = 10.0, 3.0
TARGET_EQD2 = 60.0

fraction_numbers = np.arange(1, 36)
total_doses, late_cost = [], []
for n in fraction_numbers:
    g = 1.0 / n  # complete repair between fractions
    # Solve eqd2(D, alpha/beta = 10, g = 1/n) = 60 for D: a quadratic in D.
    target_bed = TARGET_EQD2 * (1.0 + 2.0 / TUMOUR_AB)
    dose = (-1.0 + np.sqrt(1.0 + 4.0 * g / TUMOUR_AB * target_bed)) / (2.0 * g / TUMOUR_AB)
    total_doses.append(dose)
    late_cost.append(eqd2(dose, alpha_beta_ratio_Gy=LATE_AB, g=g))

ax = axes[1, 1]
late_label = rf"late tissue ($\alpha/\beta$ = {LATE_AB:.0f})"
ax.plot(fraction_numbers, late_cost, "C4-o", ms=3, label=late_label)
tumour_label = rf"tumour ($\alpha/\beta$ = {TUMOUR_AB:.0f}), fixed"
ax.axhline(TARGET_EQD2, color="C2", ls="--", label=tumour_label)
ax.set_xlabel("Number of fractions")
ax.set_ylabel("EQD2 [Gy]")
ax.set_title("Isoeffective in the tumour — what it costs late")
ax.legend(fontsize=8)

for n in (5, 30):
    i = int(n) - 1
    print(
        f"{n:>2d} x {total_doses[i] / n:.2f} Gy = {total_doses[i]:.1f} Gy   "
        f"tumour EQD2 {eqd2(total_doses[i], alpha_beta_ratio_Gy=TUMOUR_AB, g=1.0 / n):.1f} Gy, "
        f"late EQD2 {late_cost[i]:.1f} Gy"
    )

# %% [markdown]
# BED is the same statement in the units the clinic uses: the effect divided by
# $\alpha$, additive across phases of a treatment.

# %%
tumour_bed = bed(60.0, alpha_beta_ratio_Gy=TUMOUR_AB, g=1.0 / 30.0)
late_bed = bed(60.0, alpha_beta_ratio_Gy=LATE_AB, g=1.0 / 30.0)
print(f"\n60 Gy in 30 x 2 Gy: BED = {tumour_bed:.1f} Gy{TUMOUR_AB:.0f}")
print(f"                    BED = {late_bed:.1f} Gy{LATE_AB:.0f}")

fig.tight_layout()
save_figure(fig, "dose_protraction_and_bed.png", directory=FIGURES, dpi=130)
