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
# # Phenomenological RBE models: the $\mathrm{RBE}_{\min}$ / $\mathrm{RBE}_{\max}$ family
#
# Regression fits to in-vitro survival data that predict the LQ pair from
# $\mathrm{LET}_d$ and $(\alpha/\beta)_x$ alone — no track structure — on one
# skeleton (Carabe-Fernandez et al. 2007): $\mathrm{RBE}_{\max} = \alpha_{\rm
# ion}/\alpha_x$ ($D\to 0$), $\mathrm{RBE}_{\min} = \sqrt{\beta_{\rm ion}/\beta_x}$
# ($D\to\infty$). They are the baseline the mechanistic models are worth judging
# against; provenance is uneven, see `docs/references/Phenomenological_RBE_models.md`.

# %%
import warnings

import matplotlib.pyplot as plt
import numpy as np

from pyradbio import PhotonParameters
from pyradbio.models import phenomenological as ph
from pyradbio.plotting import figures_dir, save_figure

FIGURES = figures_dir(globals().get("__file__"))

# Two photon references; only the RATIO enters the models, both give back survival.
LATE = PhotonParameters(alpha_x=0.10, beta_x=0.0333)  # (a/b)_x =  3 Gy, late-responding
EARLY = PhotonParameters(alpha_x=0.30, beta_x=0.0300)  # (a/b)_x = 10 Gy, early/tumour
PHOTONS = ((LATE, "-"), (EARLY, "--"))

# WilkensOelfke's alpha_0 is a fitted V79 intercept, so it gets its own panel (7).
PROTON_MODELS = {
    "Wedenberg 2013": ph.Wedenberg,
    "McNamara 2015": ph.McNamara,
    "Carabe 2012": ph.Carabe,
    "Frese 2011 (LSM)": ph.LinearScaling,
}
COLOURS = dict(zip(PROTON_MODELS, ("tab:pink", "tab:blue", "tab:green", "tab:orange"), strict=True))

# 0.3 (Wilkens/Frese floor) to 20 keV/um (McNamara ceiling): inside every stated domain.
LET_FITTED = np.linspace(0.3, 20.0, 120)
DOSE = 2.0
LET_LABEL = "$\\mathrm{LET}_d$ [keV/µm]"
RBE_LABEL = f"RBE at {DOSE:g} Gy"


def tissue(alpha_x, beta_x):
    return PhotonParameters(alpha_x=alpha_x, beta_x=beta_x)


def arrow_in(colour):
    return {"color": colour, "arrowprops": {"arrowstyle": "->", "color": colour, "lw": 0.8}}


def finish(ax, xlabel, ylabel, title, **legend):
    ax.set_xlabel(xlabel)
    ax.set_ylabel(ylabel)
    ax.set_title(title, fontsize=10)
    ax.legend(**({"fontsize": 8} | legend))
    ax.grid(alpha=0.3)


def sweep(ax, cases, x, rbe):
    for label, factory in PROTON_MODELS.items():
        for case, style in cases:
            y = [rbe(factory, case, xi) for xi in x]
            name = label if style == "-" else None
            ax.plot(x, y, style, color=COLOURS[label], lw=1.8, label=name)


fig, axes = plt.subplots(2, 4, figsize=(22.0, 9.0))

# %% [markdown]
# ## 1–3. RBE vs $\mathrm{LET}_d$, vs $(\alpha/\beta)_x$, vs dose
#
# RBE rises roughly linearly with LET, faster for low $(\alpha/\beta)_x$ — the one
# thing the literature agrees on; the spread is the honest uncertainty. Frese
# depends on $\alpha_x$, not the ratio, so in panel 2 (fixed $\alpha_x = 0.1$) its
# rise is only the LQ response at 2 Gy. Carabe's $\mathrm{RBE}_{\min}$ can exceed
# its $\mathrm{RBE}_{\max}$, so its RBE **rises** with dose (panel 3) — which is
# why the base class assumes no ordering of the two limits.

# %%
ax = axes[0, 0]
sweep(ax, PHOTONS, LET_FITTED, lambda f, p, let: f(photon=p).rbe_at_dose_from_let(let, DOSE))
ax.axhline(1.1, color="grey", ls=":", lw=1.2)
ax.axvline(2.5, color="grey", ls=":", lw=1.2)
text = "clinical 1.1 @ SOBP plateau\n$\\mathrm{LET}_d \\approx 2.5$ keV/µm"
ax.annotate(text, xy=(2.5, 1.1), xytext=(8.5, 0.98), fontsize=8, **arrow_in("grey"))
finish(ax, LET_LABEL, RBE_LABEL, "RBE vs LET\nsolid $(\\alpha/\\beta)_x$=3 Gy, dashed 10 Gy")

ax = axes[0, 1]
sweep(
    ax,
    ((2.5, "-"), (10.0, "--")),
    np.linspace(0.5, 20.0, 120),
    lambda f, let, r: f(photon=tissue(0.1, 0.1 / r)).rbe_at_dose_from_let(let, DOSE),
)
text = "note: Frese depends on $\\alpha_x$ (held at 0.1 here), not on $(\\alpha/\\beta)_x$"
ax.text(0.5, 0.03, text, transform=ax.transAxes, ha="center", fontsize=7, color="tab:orange")
title = "RBE vs the tissue parameter\nsolid $\\mathrm{LET}_d$=2.5, dashed 10 keV/µm"
finish(ax, "$(\\alpha/\\beta)_x$ [Gy]", RBE_LABEL, title)

ax = axes[0, 2]
doses = np.linspace(0.25, 12.0, 120)
sweep(ax, PHOTONS, doses, lambda f, p, d: f(photon=p).rbe_at_dose_from_let(4.0, d))
carabe_early = ph.Carabe(photon=EARLY).rbe_min_max(4.0)
text = (
    f"Carabe, $(\\alpha/\\beta)_x$=10 Gy:\n$\\mathrm{{RBE}}_{{\\min}}$={carabe_early.rbe_min:.2f}"
    f" > $\\mathrm{{RBE}}_{{\\max}}$={carabe_early.rbe_max:.2f}"
    "\n$\\Rightarrow$ RBE *rises* with dose"
)
xy = (9.0, ph.Carabe(photon=EARLY).rbe_at_dose_from_let(4.0, 9.0))
ax.annotate(text, xy=xy, xytext=(3.2, 1.30), fontsize=8, **arrow_in("tab:green"))
title = "RBE vs dose at $\\mathrm{LET}_d$=4 keV/µm\nsolid 3 Gy, dashed 10 Gy"
finish(ax, "proton dose [Gy]", "RBE", title)

# %% [markdown]
# ## 4–7. The two limits, helium, the LQ pair, and one form with two calibrations
#
# Panel 4 is all a model contributes. Panel 5: only Mairani's helium fit turns
# over (at $L = 2/p_2 \approx 132$ keV/µm); the proton fits, drawn **outside
# their fitted range on purpose** (warnings silenced here only), keep climbing.
# Panel 6: Wedenberg keeps $\beta = \beta_x$, McNamara's $\beta$ falls with LET
# and Carabe's rises — fitted to overlapping data, they disagree on the sign.
# Panel 7: Wilkens & Oelfke 2004 and Frese et al. 2011 are both $\alpha_p =
# \alpha_0 + \lambda L$, $\beta_p := \beta_x$. Wilkens fits both coefficients to V79
# data; Frese anchors $\alpha_0 = \alpha_x - 0.5\,\lambda$ (RBE = 1 at the 0.5 keV/µm
# entrance LET, for any tissue) and sets $\lambda$ = 0.008 so a *plan* averages 1.1.

# %%
ax = axes[0, 3]
for label, factory in PROTON_MODELS.items():
    limits = [factory(photon=LATE).rbe_min_max(let) for let in LET_FITTED]
    for key, style in (("max", "-"), ("min", ":")):
        y = [getattr(x, f"rbe_{key}") for x in limits]
        name = f"{label} $\\mathrm{{RBE}}_{{\\{key}}}$"
        ax.plot(LET_FITTED, y, style, color=COLOURS[label], lw=1.8, label=name)
ax.axhline(1.0, color="grey", lw=1.0)
title = "The two limits, $(\\alpha/\\beta)_x$ = 3 Gy\nWedenberg fixes $\\mathrm{RBE}_{\\min}$=1; "
title += f"Carabe starts at {ph.carabe.CARABE_RBE_MAX_INTERCEPT:g}"
finish(ax, LET_LABEL, "RBE limit", title, fontsize=7, ncol=2)

ax = axes[1, 0]
let_wide = np.linspace(0.1, 250.0, 200)
turnover = 2.0 / ph.mairani.MAIRANI_P2
with warnings.catch_warnings():
    warnings.simplefilter("ignore")  # the whole point of this panel is extrapolation
    helium = [ph.MairaniHelium(photon=LATE).rbe_min_max(let).rbe_max for let in let_wide]
    ax.plot(let_wide, helium, color="tab:red", lw=2.4, label="Mairani 2016 (helium)")
    for label, factory in PROTON_MODELS.items():
        model = factory(photon=LATE)
        lets, rbe_max = [], []
        for let in let_wide:
            try:
                rbe_max.append(model.rbe_min_max(let).rbe_max)
            except ValueError:
                break  # RBE_min has gone negative -- the fit is no longer a model
            lets.append(let)
        name = f"{label} (extrapolated)"
        ax.plot(lets, rbe_max, color=COLOURS[label], lw=1.4, alpha=0.85, label=name)
        if len(lets) < len(let_wide):
            ax.plot(lets[-1], rbe_max[-1], "x", color=COLOURS[label], ms=9, mew=2.2)
ax.axvline(turnover, color="tab:red", ls=":", lw=1.2)
ax.axvspan(0.0, 20.0, color="grey", alpha=0.12)
text = "McNamara fitted\nrange ends"
ax.text(23, 0.42, text, fontsize=7.5, color="grey", va="top", transform=ax.get_xaxis_transform())
ax.text(turnover + 4, 2.0, "$L=2/p_2$: helium turns over", fontsize=7.5, color="tab:red")
text = "x  = model refuses to continue:\n      $\\mathrm{RBE}_{\\min}$ has gone negative"
ax.text(0.02, 0.60, text, fontsize=7.5, color="tab:blue", transform=ax.transAxes)
title = "Only the helium fit rolls over\n(proton curves far outside their fits)"
finish(ax, LET_LABEL, "$\\mathrm{RBE}_{\\max}$", title, fontsize=7.5)

ax = axes[1, 1]
twin = ax.twinx()
for label, factory in PROTON_MODELS.items():
    lq = [factory(photon=LATE).alpha_beta_from_let(let) for let in LET_FITTED]
    ax.plot(LET_FITTED, [p.alpha for p in lq], "-", color=COLOURS[label], lw=1.8, label=label)
    twin.plot(LET_FITTED, [p.beta for p in lq], ":", color=COLOURS[label], lw=1.8)
ax.axhline(LATE.alpha_x, color="grey", lw=1.0)
twin.axhline(LATE.beta_x, color="grey", lw=1.0)
twin.set_ylabel("$\\beta$ [Gy$^{-2}$]  (dotted)")
title = "The LQ pair itself, $(\\alpha/\\beta)_x$ = 3 Gy\n"
title += "grey lines = the photon $\\alpha_x$, $\\beta_x$"
finish(ax, LET_LABEL, "$\\alpha$ [Gy$^{-1}$]  (solid)", title, loc="upper left")

ax = axes[1, 2]
let_w = np.linspace(0.3, 30.0, 160)
wilkens = ph.WilkensOelfke.v79_wilkens2004()
name = f"Wilkens 2004, V79 fit\n($\\alpha_0$={ph.wilkens.WILKENS_ALPHA0_GY:g}, "
name += f"$\\lambda$={ph.wilkens.WILKENS_LAMBDA_UM_KEV_GY:g})"
y = [wilkens.rbe_at_dose_from_let(let, DOSE) for let in let_w]
ax.plot(let_w, y, color="tab:red", lw=2.2, label=name)
for alpha_x, beta_x, name in (  # Frese et al. 2011 table 3
    (0.1124, 0.0112, "nasopharyngeal T1/T2"),
    (0.0407, 0.0203, "spinal cord"),
    (0.0265, 0.0088, "inner ears"),
):
    frese = ph.LinearScaling(photon=tissue(alpha_x, beta_x))
    y = [frese.rbe_at_dose_from_let(let, DOSE) for let in let_w]
    ax.plot(let_w, y, lw=1.5, ls="--", label=f"Frese 2011, {name}\n($\\alpha_x$={alpha_x:.4f})")
entrance = ph.frese.FRESE_ENTRANCE_LET_KEV_UM
ax.axvline(entrance, color="grey", ls=":", lw=1.2)
ax.axhline(1.0, color="grey", lw=1.0)
text = "every Frese curve is\nRBE = 1 here, by construction"
ax.annotate(text, xy=(entrance, 1.0), xytext=(6.0, 1.15), fontsize=7.5, **arrow_in("grey"))
title = "The same form, two calibrations\n"
title += "$\\alpha_p = \\alpha_0 + \\lambda L$, $\\beta_p := \\beta_x$"
finish(ax, LET_LABEL, RBE_LABEL, title, fontsize=6.5)

axes[1, 3].remove()  # Frese table 4 is asserted in tests/models/test_phenomenological_rbe.py
title = "Phenomenological LET-based RBE models — "
title += "the $\\mathrm{RBE}_{\\min}$/$\\mathrm{RBE}_{\\max}$ family"
fig.suptitle(title, fontsize=13)
fig.tight_layout(rect=(0, 0, 1, 0.97))
save_figure(fig, "phenomenological_rbe_models.png", directory=FIGURES, dpi=140)

# %%
print(f"RBE at {DOSE:g} Gy, LET_d = 2.5 keV/um (SOBP plateau), vs the clinical 1.1:")
for label, factory in PROTON_MODELS.items():
    late = factory(photon=LATE).rbe_at_dose_from_let(2.5, DOSE)
    early = factory(photon=EARLY).rbe_at_dose_from_let(2.5, DOSE)
    print(f"  {label:<16}  (a/b)_x=3 Gy: {late:.3f}    (a/b)_x=10 Gy: {early:.3f}")

print("\nSpread between models at the distal edge (LET_d = 10 keV/um, 2 Gy):")
for photon, name in ((LATE, "3 Gy"), (EARLY, "10 Gy")):
    v = [f(photon=photon).rbe_at_dose_from_let(10.0, DOSE) for f in PROTON_MODELS.values()]
    spread = 100 * (max(v) / min(v) - 1)
    print(f"  (a/b)_x={name:<6} {min(v):.3f} - {max(v):.3f}  ({spread:.1f}% spread)")

print("\nWhat each model does to beta (beta/beta_x, (a/b)_x = 3 Gy) -- note the SIGNS differ:")
for label, factory in PROTON_MODELS.items():
    model = factory(photon=LATE)
    lo = model.alpha_beta_from_let(0.3).beta / LATE.beta_x
    hi = model.alpha_beta_from_let(20.0).beta / LATE.beta_x
    trend = "flat" if abs(hi - lo) < 1e-9 else ("falls" if hi < lo else "rises")
    print(f"  {label:<16} LET_d 0.3 -> 20 keV/um: {lo:.3f} -> {hi:.3f}   ({trend})")
