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
# # Katz 1971: ion-kill and gamma-kill
#
# The oldest model in pyRadBio and the ancestor of every amorphous-track model in
# it — Katz, Ackerson, Homayoonfar & Sharma (1971), Radiat. Res. **47**(2):402–425,
# [doi:10.2307/3573247](https://doi.org/10.2307/3573247). A cell is four numbers,
# $\sigma_0$, $E_0$, $\kappa$, $m$, and killing happens in **two modes**:
# **ion-kill** (one traversing ion kills outright, with probability $P$ and
# cross-section $\sigma = \sigma_0 P$) and **gamma-kill** (the dose that did not
# act that way pools across ions and kills through the gamma-ray multi-target
# statistic). $P$ is also the fraction of the dose routed to ion-kill, so the
# surviving fraction is a product (eq. 10):
#
# $$\frac{N}{N_0} = e^{-\sigma F}\Big(1 - \big[1 - e^{-(1-P)E/E_0}\big]^m\Big),
#   \qquad E = FL, \qquad
#   P = \Big(1 - e^{-z^{*2}/\kappa\beta^2}\Big)^m$$
#
# with the Barkas effective charge $z^* = Z[1 - e^{-125 Z^{-2/3}\beta}]$ (eq. 12).
# **It is not an LQ model and has no $\beta$** (its shoulder is the exponent $m$,
# so low-dose gamma-kill goes as $D^m$); `alpha_beta()` reports eq. (14)'s
# extrapolated radiosensitivity $k$ with $\beta = 0$ exactly. **Only the
# grain-count regime is implemented**, so $\sigma$ saturates at $\sigma_0$. See
# `docs/references/Katz1971.md`; Table I's own model line is asserted in
# `tests/models/test_katz.py`.

# %%
import matplotlib.pyplot as plt
import numpy as np

from pyradbio.core.radiation import Ion
from pyradbio.models.katz import KatzCellularModel
from pyradbio.plotting import figures_dir, save_figure
from pyradbio.track.stopping_power import SimpleBetheStoppingPower

FIGURES = figures_dir(globals().get("__file__"))

# Bethe covers every Z swept here; the tabulated sources stop at Z = 10.
SP = SimpleBetheStoppingPower()

# Katz's mammalian fit (Fig. 14, Skarsgard et al. 1967). Its sigma_0 = 5.0e-7
# cm^2 = 50 um^2 is the nuclear cross-section the LEM family later adopts for V79.
HAMSTER = KatzCellularModel.chinese_hamster_skarsgard1967()

IONS = (
    ("$^{1}$H", 1, 1),
    ("$^{4}$He", 2, 4),
    ("$^{12}$C", 6, 12),
    ("$^{20}$Ne", 10, 20),
    ("$^{40}$Ar", 18, 40),
)
ENERGIES = np.logspace(np.log10(1.0), np.log10(1000.0), 160)  # MeV/u


def sweep(model: KatzCellularModel, Z: int, A: int) -> dict[str, np.ndarray]:
    """LET, P and eq. (14)'s k = ion-kill + gamma-kill along the energy sweep."""
    let, terms = [], []
    for e in ENERGIES:
        ion = Ion(Z=Z, A=A, energy_MeV_u=float(e))
        let.append(SP.let_keV_um(ion))
        terms.append(model.kill_terms_per_Gy(Z, ion.beta(), let_keV_um=let[-1]))
    return {
        "let": np.array(let),
        "P": np.array([t.ion_kill_probability for t in terms]),
        "k": np.array([t.total for t in terms]),
    }


SWEEPS = {name: sweep(HAMSTER, Z, A) for name, Z, A in IONS}

# %% [markdown]
# ## The figure
#
# Panel 3 is the headline. The five $k$-vs-LET curves **do not overlap**: two
# ions at the same LET have different $z^{*2}/\beta^2$, hence different $P$ and
# a different split between the modes. In Katz's words, "the parameters
# describing the bombarding ion and those describing the cellular system are not
# separable", so any model keyed on LET alone — `models/phenomenological`, the
# `LET_d` route of the PIDE harness — cannot express it. The turnover is
# **overkill**: once $P\to1$ the gamma-kill term $(1-P)/E_0$ vanishes and $k$
# collapses to $\sigma_0/L$, so argon at ~1270 keV/µm falls *below* the low-LET
# floor. The RBE maximum falls out of the two-mode structure, not imposed.
# Panel 5: the oxygen effect lives principally in $E_0$ (and $\kappa$); $\sigma_0$
# and $m$ are identical within each published anoxic/aerobic pair (p. 419).
# Every curve runs on our analytic Bethe LET, not Table I's: for He at
# $\beta = 0.145$ that is 18.2 against the printed 19.1 keV/µm. Fed the paper's
# own $(\beta, L)$, $k$ lands on Table I (asserted in the tests), so any offset
# between these curves and the table is stopping power, not model.

# %%
fig, axes = plt.subplots(2, 3, figsize=(16.5, 9.5))
colors = plt.cm.viridis(np.linspace(0.05, 0.85, len(IONS)))
LET_LABEL = r"LET [keV/$\mu$m]"


def finish(ax, xlabel, ylabel, title, **legend):
    ax.set_xlabel(xlabel)
    ax.set_ylabel(ylabel)
    ax.set_title(title)
    ax.legend(**({"fontsize": 8} | legend))
    ax.grid(alpha=0.3)


ax = axes[0, 0]
for (name, _Z, _A), c in zip(IONS, colors, strict=True):
    ax.plot(SWEEPS[name]["let"], SWEEPS[name]["P"], color=c, label=name)
ax.set_xscale("log")
ylabel = r"$P = \sigma/\sigma_0$  (ion-kill fraction)"
finish(ax, LET_LABEL, ylabel, "1. Eq. (8): how the dose splits between the two modes")
text = "below: nearly all dose\nis gamma-kill"
ax.text(0.03, 0.55, text, transform=ax.transAxes, fontsize=7.5, color="0.35")

ax = axes[0, 1]
doses = np.linspace(0.0, 12.0, 200)
photon = [HAMSTER.photon_survival(float(d)) for d in doses]
ax.semilogy(doses, photon, "k--", lw=2, label=r"photons ($P\to0$, eq. 3)")
for (name, Z, A), c in zip(IONS, colors, strict=True):
    ion = Ion(Z=Z, A=A, energy_MeV_u=10.0)
    line = SP.let_keV_um(ion)
    survival = [HAMSTER.survival(ion, float(d), let_keV_um=line) for d in doses]
    ax.semilogy(doses, survival, color=c, label=f"{name}, {line:.0f} keV/$\\mu$m")
ax.set_ylim(1e-4, 1.5)
title = "2. Eq. (10): the shoulder is eaten as ion-kill takes over"
finish(ax, "dose [Gy]", "surviving fraction", title, fontsize=7.5)

ax = axes[0, 2]
for (name, _Z, _A), c in zip(IONS, colors, strict=True):
    ax.plot(SWEEPS[name]["let"], SWEEPS[name]["k"], color=c, label=name)
ax.axhline(1.0 / HAMSTER.E0_Gy, color="0.4", ls=":", lw=1.2)
ax.text(1.2, 0.16, "dotted: $1/E_0$, the gamma-kill floor", fontsize=7.5, color="0.35")
ax.set_xscale("log")
ax.set_ylim(0.0, 1.85)
finish(ax, LET_LABEL, r"$k$ [Gy$^{-1}$]  (eq. 14)", "3. Overkill — and the curves do NOT overlap")

ax = axes[1, 0]
x = np.logspace(-2, 2.2, 300)  # z*^2 / (kappa beta^2)
for m_value, style in ((2.0, "-"), (2.5, "--"), (3.0, "-."), (4.0, ":")):
    ax.plot(x, (1.0 - np.exp(-x)) ** m_value, style, label=f"$m$ = {m_value:g}")
ax.axhline(1.0, color="0.6", lw=0.8)
ax.set_xscale("log")
title = "4. Eq. (8) saturates — the grain-count ceiling"
finish(ax, r"$z^{*2}/(\kappa\beta^2)$", r"$\sigma/\sigma_0$", title)
text = "beyond saturation the real model\nenters the TRACK-WIDTH regime\n"
text += r"($\sigma/\sigma_0\approx2$ for Ar on kidney)," + "\nwhich needs eq. (4) — not implemented"
ax.text(0.04, 0.72, text, transform=ax.transAxes, fontsize=7, color="firebrick")

ax = axes[1, 1]
PAIRS = (
    (
        "T-1 kidney (Todd 1966)",
        KatzCellularModel.t1_kidney_todd1966_anoxic(),
        KatzCellularModel.t1_kidney_todd1966_aerobic(),
        "tab:blue",
    ),
    (
        "T-1 kidney (Barendsen 1966)",
        KatzCellularModel.t1_kidney_barendsen1966_n2(),
        KatzCellularModel.t1_kidney_barendsen1966_o2(),
        "tab:red",
    ),
)
for label, anoxic, aerobic, c in PAIRS:
    for model, ls, state in ((anoxic, "--", "anoxic"), (aerobic, "-", "aerobic")):
        survival = [model.photon_survival(float(d)) for d in doses]
        ax.semilogy(doses, survival, color=c, ls=ls, label=f"{label} — {state}")
ax.set_ylim(1e-4, 1.5)
ax.set_xlabel("dose [Gy]")
ax.set_ylabel("surviving fraction (photons)")
ax.set_title(r"5. The oxygen effect shows up principally in $E_0$")
text = "$\\sigma_0$ and $m$ are IDENTICAL within each pair;\n"
text += "$E_0$ and $\\kappa$ carry the whole effect (p. 419)"
ax.text(0.30, 0.06, text, transform=ax.transAxes, fontsize=7, color="0.3")
ax.legend(fontsize=7)
ax.grid(alpha=0.3)

axes[1, 2].remove()  # Katz 1971 Table I is asserted in tests/models/test_katz.py
title = "Katz cellular track-structure theory (Katz et al. 1971, doi:10.2307/3573247) — "
fig.suptitle(title + "Chinese hamster fit unless noted", fontsize=12)
fig.tight_layout(rect=(0, 0, 1, 0.97))
save_figure(fig, "katz_ion_kill_and_gamma_kill.png", directory=FIGURES, dpi=140)

# %% [markdown]
# ## RBE at 10% survival, with no external photon reference
#
# Katz needs no borrowed $(\alpha_x, \beta_x)$: $E_0$ and $m$ **are** the
# gamma-ray parameters, fitted jointly with the ion data, so both doses of the
# RBE come from the same four numbers — the photon one from eq. (3), the $P\to0$
# limit of eq. (10).

# %%
d10_photon = HAMSTER.photon_dose_for_survival(0.1)
print(f"photon D10 = {d10_photon:.2f} Gy  (Katz's own gamma-ray curve)\n")
print(f"{'ion':>10} {'E [MeV/u]':>10} {'LET':>9} {'P':>9} {'D10 [Gy]':>9} {'RBE_10':>7}")
for name, Z, A in IONS:
    for energy in (100.0, 10.0):
        ion = Ion(Z=Z, A=A, energy_MeV_u=energy)
        line = SP.let_keV_um(ion)
        d10 = HAMSTER.dose_for_survival(ion, 0.1, let_keV_um=line)
        rbe10 = HAMSTER.rbe_at_survival(ion, 0.1, let_keV_um=line)
        print(
            f"{name.replace('$', '').replace('^{', '').replace('}', ''):>10} {energy:>10.0f} "
            f"{line:>9.1f} {HAMSTER.ion_kill_probability(Z, ion.beta()):>9.3f} "
            f"{d10:>9.2f} {rbe10:>7.2f}"
        )
