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
# # RMF: DSB induction and binary misrepair
#
# The **Repair-Misrepair-Fixation** model — Carlson, Stewart, Semenenko &
# Sandison (2008), Radiat. Res. **169**:447,
# [doi:10.1667/RR1046.1](https://doi.org/10.1667/RR1046.1), in the formulation of
# Frese, Yu, Stewart & Carlson (2012), IJROBP **83**:442,
# [doi:10.1016/j.ijrobp.2011.06.1983](https://doi.org/10.1016/j.ijrobp.2011.06.1983).
#
# The one family here whose *biology* is not amorphous-track: it counts **DNA
# double-strand breaks**. A single DSB may be misrepaired or fixed (one-track,
# linear in dose); two DSBs may interact into a lethal exchange (**binary
# misrepair**), within one track (**intratrack**, still linear) or between tracks
# (quadratic). Time-integrating that scheme gives
#
# $$\alpha(q) = \underbrace{\theta\,\Sigma(q)}_{\text{one-track}}
#   + \underbrace{\kappa\,\bar z_F(q)\,\Sigma^2(q)}_{\text{intratrack pairs}},
#   \qquad \beta(q) = \frac{\kappa}{2}\,\Sigma^2(q)$$
#
# with $\Sigma$ the DSB yield (Gy⁻¹ Gbp⁻¹), $\bar z_F = 0.204\,\mathrm{LET}/d^2$ in a
# site of **diameter** $d = 5\,\mu$m (the LEM/MKM `R_n = 5 um` is a radius), and
# $\theta$, $\kappa$ two tissue parameters that are **not free**: eqs. (5)–(6)
# invert at the reference radiation (Frese Table 1, asserted in
# `tests/models/test_rmf.py`). Eq. (9) makes $\mathrm{RBE}_{\min} =
# \sqrt{\beta/\beta_x} = \Sigma/\Sigma_x$ exactly. See
# `docs/references/RMF_Carlson2008_Frese2012.md`.

# %%
import warnings

import matplotlib.pyplot as plt
import numpy as np

from pyradbio.core.photon import PhotonParameters
from pyradbio.core.radiation import Ion
from pyradbio.micro import MCDSYields
from pyradbio.models.rmf import CO60_REFERENCE, X_RAY_200KVP_REFERENCE, RMFModel
from pyradbio.plotting import figures_dir, save_figure
from pyradbio.track.stopping_power import SimpleBetheStoppingPower

FIGURES = figures_dir(globals().get("__file__"))

# V79 from Frese Table 1 (200 kVp reference). Its sweeps are plotted against the
# analytic-Bethe LET, so the model carries that stopping power.
V79 = RMFModel.v79_furusawa2000(stopping_power=SimpleBetheStoppingPower())
IONS = (("$^{1}$H", 1, 1), ("$^{4}$He", 2, 4), ("$^{12}$C", 6, 12), ("$^{20}$Ne", 10, 20))
ENERGIES = np.logspace(np.log10(1.5), np.log10(500.0), 140)  # MeV/u


def sweep(Z: int, A: int) -> dict[str, np.ndarray]:
    """LET, Sigma and the two terms of eq. (5) along the energy sweep.

    The sweep deliberately runs past the 150 keV/um ceiling to *show* that region,
    so the model's warning is silenced here and the panels mark the boundary.
    """
    with warnings.catch_warnings():
        warnings.simplefilter("ignore", UserWarning)
        terms = [V79.alpha_terms(Ion(Z=Z, A=A, energy_MeV_u=float(e))) for e in ENERGIES]
    return {
        "let": np.array([t.let_keV_um for t in terms]),
        "sigma": np.array([t.dsb_per_Gy_per_Gbp for t in terms]),
        "one_track": np.array([t.one_track for t in terms]),
        "intratrack": np.array([t.intratrack for t in terms]),
    }


SWEEPS = {name: sweep(Z, A) for name, Z, A in IONS}

# %% [markdown]
# ## The figure
#
# Panel 1: the DSB yield does **not** collapse onto LET — $\Sigma$ is a function
# of $Z_{\rm eff}^2/\beta^2$. Panel 2 splits eq. (5): the one-track term grows only
# as fast as $\Sigma$ (a factor ~3, saturating at 27.4 Gy⁻¹Gbp⁻¹), the intratrack
# term as $\mathrm{LET}\times\Sigma^2$ — at low LET a cell dies of accumulated
# independent damage, at high LET of pairs of breaks from *the same track*. For
# carbon in V79 the two are equal at **214 keV/µm**, *above* Frese's 150 keV/µm
# ceiling: across RMF's stated domain the pairing it is named for is a
# correction, dominant only where the paper says the model over-predicts.
# Panel 3: both $\Sigma$ and $\bar z_F$ rise monotonically, so nothing in eqs.
# (5)–(6) can produce the measured peak near 100–250 keV/µm — no saturation or
# overkill mechanism (the LEM's local-dose saturation, the MKM's saturation
# correction), a limit Frese states plainly. Panels 4–5: RBE vs dose (eq. 16) for
# the six Table 1 tissues, and eq. (17)'s structural claim that RBE depends on
# $(\alpha/\beta)_x$ but not on $\alpha_x$.

# %%
fig, axes = plt.subplots(2, 3, figsize=(16.5, 9.5))
colors = plt.cm.plasma(np.linspace(0.05, 0.75, len(IONS)))
CEILING = V79.let_ceiling_keV_um
NOTE = {"fontsize": 7, "color": "0.3"}


def finish(ax, xlabel, ylabel, title, *, log=("x",), **legend):
    for axis in log:
        getattr(ax, f"set_{axis}scale")("log")
    ax.set_xlabel(xlabel)
    ax.set_ylabel(ylabel)
    ax.set_title(title)
    ax.legend(**({"fontsize": 8} | legend))
    ax.grid(alpha=0.3)


ax = axes[0, 0]
for (name, _Z, _A), c in zip(IONS, colors, strict=True):
    ax.plot(SWEEPS[name]["let"], SWEEPS[name]["sigma"], color=c, label=name)
sigma_x = V79.reference_radiation.dsb_per_Gy_per_Gbp
ax.axhline(sigma_x, color="0.4", ls="--", lw=1.2)
ax.text(0.5, sigma_x * 1.02, r"$\Sigma_x$ = 8.68 (200 kVp reference)", fontsize=7.5, color="0.35")
ax.axvline(CEILING, color="firebrick", ls=":", lw=1.2)
title = r"1. MCDS DSB yield — and $\mathrm{RBE}_{\min} = \Sigma/\Sigma_x$"
finish(ax, r"LET [keV/$\mu$m]", r"$\Sigma$  [Gy$^{-1}$ Gbp$^{-1}$]", title, loc="lower right")
text = "the curves do NOT collapse onto LET:\n"
text += r"$\Sigma$ is a function of $Z_{\rm eff}^2/\beta^2$," + "\nnot of stopping power"
ax.text(0.03, 0.80, text, transform=ax.transAxes, **NOTE)

ax = axes[0, 1]
s = SWEEPS["$^{12}$C"]
ax.plot(s["let"], s["one_track"], color="C0", lw=2, label=r"one-track  $\theta\Sigma$")
label = r"intratrack pairs  $\kappa \bar z_F \Sigma^2$"
ax.plot(s["let"], s["intratrack"], color="C3", lw=2, label=label)
ax.plot(s["let"], s["one_track"] + s["intratrack"], "k-", lw=1.2, label=r"total $\alpha$")
# np.interp needs ascending x; LET falls as energy rises, so sort on the difference.
diff = s["intratrack"] - s["one_track"]
order = np.argsort(diff)
crossover = float(np.interp(0.0, diff[order], s["let"][order]))
ax.axvline(crossover, color="0.5", ls="--", lw=1)
text = f"crossover\n{crossover:.0f} keV/$\\mu$m"
ax.text(crossover * 1.1, 0.02, text, fontsize=7.5, color="0.3")
ax.axvline(CEILING, color="firebrick", ls=":", lw=1.2)
title = r"2. Eq. (5): the LET effect IS the intratrack term ($^{12}$C)"
ylabel = r"contribution to $\alpha$ [Gy$^{-1}$]"
finish(ax, r"LET [keV/$\mu$m]", ylabel, title, log=("x", "y"), loc="upper left")

ax = axes[0, 2]
for (name, _Z, _A), c in zip(IONS, colors, strict=True):
    s = SWEEPS[name]
    ax.plot(s["let"], s["one_track"] + s["intratrack"], color=c, label=name)
ax.axvline(CEILING, color="firebrick", ls=":", lw=1.2)
title = "3. No RBE turnover — the model's known limit"
finish(ax, r"LET [keV/$\mu$m]", r"$\alpha$ [Gy$^{-1}$]", title, log=("x", "y"), loc="upper left")
text = "dotted line: 150 keV/$\\mu$m.\nFrese reports the measured parameters peak at\n"
text += "100-250 keV/$\\mu$m, while RMF keeps rising, because\n"
text += r"$\Sigma$ and $\bar z_F$ both keep rising. Not a bug:" + "\na stated limit of the model."
ax.text(0.30, 0.10, text, transform=ax.transAxes, fontsize=7, color="firebrick")

carbon = Ion(Z=6, A=12, energy_MeV_u=50.0)
doses = np.linspace(0.25, 8.0, 60)


def rbe_curve(model):
    return [model.rbe_at_dose(carbon, float(d)) for d in doses]


ax = axes[1, 0]
for name, factory in RMFModel.published_presets().items():
    model = factory()
    ratio = model.photon.alpha_x / model.photon.beta_x
    style = "-" if model.reference_radiation is CO60_REFERENCE else "--"
    ax.plot(doses, rbe_curve(model), style, label=f"{name}  ($\\alpha/\\beta)_x$={ratio:.1f}")
title = r"4. Eq. (16): RBE vs dose, all six Table 1 tissues"
finish(ax, "dose [Gy]", "RBE", title, log=(), fontsize=7)
text = "solid = Co-60 reference, dashed = 200 kVp.\n"
text += r"Within each group RBE falls monotonically with $(\alpha/\beta)_x$;"
text += "\nthe two groups are offset because $\\Sigma_x$ differs (8.27 vs 8.68)."
ax.text(0.30, 0.78, text, transform=ax.transAxes, fontsize=6.5, color="0.3")

ax = axes[1, 1]
for scale, style, width in ((1.0, "-", 3.0), (3.0, "--", 1.8), (0.4, ":", 1.5)):
    photon = PhotonParameters(alpha_x=0.1 * scale, beta_x=0.05 * scale)
    model = RMFModel(photon=photon, reference_radiation=CO60_REFERENCE)
    label = rf"$\alpha_x$={0.1 * scale:.2f}, $\beta_x$={0.05 * scale:.3f}"
    ax.plot(doses, rbe_curve(model), style, lw=width, label=label)
for ratio, c in ((2.0, "C3"), (10.0, "C2")):
    photon = PhotonParameters(alpha_x=0.2, beta_x=0.2 / ratio)
    model = RMFModel(photon=photon, reference_radiation=CO60_REFERENCE)
    ax.plot(doses, rbe_curve(model), color=c, lw=1.4, label=rf"$(\alpha/\beta)_x$ = {ratio:g} Gy")
title = r"5. Eq. (17): RBE ignores $\alpha_x$, only the ratio matters"
finish(ax, "dose [Gy]", "RBE", title, log=(), fontsize=7)
text = "the three black/grey curves are EXACTLY coincident:\n"
text += r"scaling $\alpha_x$ and $\beta_x$ together leaves RBE"
text += "\nuntouched. Changing the ratio does not."
ax.text(0.28, 0.70, text, transform=ax.transAxes, fontsize=6.5, color="0.3")

axes[1, 2].remove()  # Frese 2012 Table 1 is asserted in tests/models/test_rmf.py
title = "RMF — Repair-Misrepair-Fixation (Carlson et al. 2008; Frese et al. 2012), V79 unless noted"
fig.suptitle(title, fontsize=12)
fig.tight_layout(rect=(0, 0, 1, 0.97))
save_figure(fig, "rmf_dsb_and_binary_misrepair.png", directory=FIGURES, dpi=140)

# %% [markdown]
# ## The reference radiations
#
# $\theta$ and $\kappa$ come from inverting eqs. (5)–(6) at the reference
# radiation, so its DSB yield $\Sigma_x$ is the one input every tissue shares.

# %%
print(
    "\n  Both reference radiations are carried as printed: "
    f"Co-60 Sigma_x = {CO60_REFERENCE.dsb_per_Gy_per_Gbp} at "
    f"{CO60_REFERENCE.let_keV_um} keV/um, 200 kVp "
    f"{X_RAY_200KVP_REFERENCE.dsb_per_Gy_per_Gbp} at "
    f"{X_RAY_200KVP_REFERENCE.let_keV_um} keV/um."
)
print(
    "  Neither follows from the MCDS closed form -- both were simulated with a\n"
    "  secondary-electron spectrum. They sit just above its x -> 0 limit of "
    f"{MCDSYields().dsb_per_Gy_per_Gbp(0.0):.2f}."
)
