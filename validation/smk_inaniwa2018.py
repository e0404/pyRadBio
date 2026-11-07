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
# # SMK reconstruction: Inaniwa & Kanematsu 2018, Figs. 4, 2 and 3
#
# doi:10.1088/1361-6560/aabede. Part 1 reconstructs **Fig. 4**, the three
# dose-averaged specific energies per event that drive the SMK, for `Z_p = 1–10`:
#
# | panel | quantity | paper eq. | role |
# | --- | --- | --- | --- |
# | (a) | `z̄_d,D` | 11 | domain, uncorrected — sets `β_SMK` via the ratio |
# | (b) | `z̄*_d,D` | 12 | domain, saturation-corrected — sets `α_SMK` |
# | (c) | `z̄_n,D` | 7 | cell nucleus — sets the non-LQ `γ` term |
#
# Fig. 4 takes the effect model out of the loop and the table-1 parameters are
# pinned exactly, so every deviation is **our microdosimetry**. That is why it comes
# before Figs. 2–3, which fold track structure, geometry and biology into one number:
# with Fig. 4 settled, a survival residual is diagnosable rather than merely
# suggestive (the lesson of the LEM reconstructions). All 30 series were
# read from the PDF as vector paths; provenance, axis calibration and the two
# extraction traps are in `pyradbio.benchmark.inaniwa2018`, the findings in
# `docs/references/SMK_Inaniwa2018.md` ("Fig. 4 reconstruction").
#
# What to look for: (b) is unbiased (median ratio ~1.00, peaks at ~1 / 12 /
# 40 MeV/u for He / C / Ne as the paper states); (a) and (c) run a few percent
# low, consistently (track-core dominated, where our Kiefer–Chatterjee radial dose
# and stopping power differ most; the saturation flattens that region out of
# (b)); and the domain/nucleus ratio falls from the geometric `(R_n/r_d)² = 837`
# as δ-rays outgrow the 0.28 µm domain but not the 8.1 µm nucleus.
#
# The model scores its track with the library-default stopping power while the
# Part 2 LET inversion uses `SimpleBetheStoppingPower`; that mixed state is kept
# on purpose for now.

# %%
import matplotlib.pyplot as plt
import numpy as np
from _common import pick, quick_mode

from pyradbio.benchmark.inaniwa2018 import (
    CELL_MODELS,
    PANELS,
    delta2_per_point,
    dose_at_survival,
    fig4_comparison,
    fig4_model_values,
    get_fig4_curve,
    get_survival_curve,
    reconstruct_panel,
    survival_panels,
)
from pyradbio.benchmark.metrics import ratio_summary
from pyradbio.models.mkm.spectrum_models import StochasticMKM
from pyradbio.plotting import figures_dir, save_figure
from pyradbio.track.stopping_power import SimpleBetheStoppingPower

FIGURES = figures_dir(globals().get("__file__"))
if quick_mode():
    print("QUICK MODE (PYRADBIO_QUICK=1): coarse grids; numbers are not the quoted ones")

# Fig. 4 was computed with the table-1 HSG column: r_d 0.28 um, R_n 8.1 um,
# z0 66.0 Gy -- exactly `hsg_inaniwa2018()`.
model = StochasticMKM.hsg_inaniwa2018()
SHOWN = [1, 2, 6, 10]  # H, He, C, Ne -- the species the paper discusses by name
COLOURS = {1: "0.25", 2: "C3", 6: "C0", 10: "C2"}
LABELS = {1: "$^{1}$H", 2: "$^{4}$He", 6: "$^{12}$C", 10: "$^{20}$Ne"}
T = np.logspace(np.log10(0.3), np.log10(500.0), pick(60, 15))

ours = {z: fig4_model_values(model, z, T) for z in SHOWN}

# %% [markdown]
# ## Agreement, pooled over all ten species
#
# Restricted per panel to where the published curve is resolvable: (a) and (c)
# are scaled for the heaviest ions, so light-ion curves ride the axis (protons in
# (a) sit under 1 Gy against a 1200 Gy full scale).

# %%
grid = np.logspace(np.log10(0.3), np.log10(500.0), pick(24, 8))
print(f"{'quantity':>15} {'n':>5} {'median ratio':>14} {'RMS log-ratio':>15} {'worst dev':>11}")
for quantity in ("z_domain", "z_star_domain", "z_nucleus"):
    series = [fig4_comparison(model, quantity, z, grid) for z in range(1, 11)]
    ours_q, paper_q, mask_q = (np.concatenate(column) for column in zip(*series, strict=True))
    s = ratio_summary(ours_q, paper_q, mask=mask_q)
    assert s.n == np.count_nonzero(mask_q)  # NaN-free: the nan-aware reductions = plain ones
    print(f"{quantity:>15} {s.n:>5} {s.median:>14.3f} {s.rms_log:>15.3f} {s.max_abs_rel:>10.1%}")

# %% [markdown]
# ## Panels

# %%
fig, axes = plt.subplots(1, 4, figsize=(20.5, 4.7))

for ax, (quantity, panel, title, ylabel) in zip(
    axes[:3],
    [
        ("z_domain", "a", r"(a)  $\bar{z}_{d,D}$   (eq. 11)", r"$\bar{z}_{d,D}$ [Gy]"),
        ("z_star_domain", "b", r"(b)  $\bar{z}^*_{d,D}$   (eq. 12)", r"$\bar{z}^*_{d,D}$ [Gy]"),
        ("z_nucleus", "c", r"(c)  $\bar{z}_{n,D}$   (eq. 7)", r"$\bar{z}_{n,D}$ [Gy]"),
    ],
    strict=True,
):
    for z in SHOWN:
        curve = get_fig4_curve(quantity, z)
        ax.plot(curve.T, curve.z, "-", color=COLOURS[z], lw=3, alpha=0.30)
        ax.plot(T, ours[z][quantity], "--", color=COLOURS[z], lw=1.5, label=LABELS[z])
    floor = 0.02 * PANELS[panel][1]
    ax.axhspan(0, floor, color="0.85", zorder=0)
    ax.set(
        xscale="log",
        yscale="log",
        xlabel="kinetic energy $T$ [MeV/u]",
        ylabel=ylabel,
        title=title,
        ylim=(floor * 0.3, None),
    )
    ax.legend(fontsize=8, title="thick = paper\ndashed = pyRadBio", title_fontsize=7)

# --- 4. the two-site geometry ratio ---------------------------------------
ax = axes[3]
geometric = (8.1 / 0.28) ** 2
for z in (6, 10):
    a, c = get_fig4_curve("z_domain", z), get_fig4_curve("z_nucleus", z)
    Tg = np.logspace(np.log10(0.3), np.log10(50.0), 40)
    ax.plot(Tg, a.interp(Tg) / c.interp(Tg), "-", color=COLOURS[z], lw=3, alpha=0.30)
    ax.plot(
        T[T <= 50],
        (ours[z]["z_domain"] / ours[z]["z_nucleus"])[T <= 50],
        "--",
        color=COLOURS[z],
        lw=1.5,
        label=LABELS[z],
    )
ax.axhline(geometric, color="0.4", ls=":", lw=1.2)
ax.text(0.35, geometric * 1.02, r"$(R_n/r_d)^2 = 837$", fontsize=8, color="0.35")
ax.set(
    xscale="log",
    xlabel="kinetic energy $T$ [MeV/u]",
    ylabel=r"$\bar{z}_{d,D} \; / \; \bar{z}_{n,D}$",
    title="two-site geometry:\n$\\delta$-rays outgrow the domain",
)
ax.legend(fontsize=8)

fig.tight_layout()
print()
save_figure(fig, "smk_reconstruction.png", directory=FIGURES, dpi=150)

# %% [markdown]
# # Part 2 — Figs. 2 and 3: survival curves against experiment
#
# Each panel of Figs. 2 (HSG) and 3 (V79) is one ion at one LET with three series:
# the Furusawa et al. 2000 **measurements** (dots), the paper's **modified SMK**
# (solid) and its **MK** model (dashed). Nothing is fitted:
# `reconstruct_panel` inverts the quoted LET for the monoenergetic energy the
# paper assumed (high-energy root), scores the three specific energies and
# evaluates eqs. (15), (16), (24) with the table-1 parameters.

# %%
CELLS = {cell: (smk(), mk()) for cell, (smk, mk) in CELL_MODELS.items()}
DOSE = np.linspace(0.01, 12.0, 600)
sp = SimpleBetheStoppingPower()

recon = {
    (cell_line, curve.panel): reconstruct_panel(
        curve, DOSE, stopping_power=sp, smk=CELLS[cell_line][0], mk=CELLS[cell_line][1]
    )
    for cell_line in CELLS
    for curve in survival_panels(cell_line)
}

print(
    f"{'cell':>5} {'panel':>6} {'ion':>4} {'LET':>7} {'E MeV/u':>9} "
    f"{'D10 SMK ours/paper':>21} {'D10 MK ours/paper':>20}"
)
for cell_line in CELLS:
    for curve in survival_panels(cell_line):
        r = recon[cell_line, curve.panel]
        d10_smk, d10_mk = dose_at_survival(DOSE, r.smk), dose_at_survival(DOSE, r.mk)
        paper_mk = get_survival_curve(cell_line, curve.panel, "mk")
        print(
            f"{cell_line:>5} {curve.panel:>6} {curve.ion:>4} {curve.let_keV_um:7.1f} "
            f"{r.ion.energy_MeV_u:9.2f} "
            f"{d10_smk:8.2f}/{curve.dose_at_survival():5.2f} "
            f"{d10_smk / curve.dose_at_survival():5.3f} "
            f"{d10_mk:8.2f}/{paper_mk.dose_at_survival():5.2f} "
            f"{d10_mk / paper_mk.dose_at_survival():5.3f}"
        )

# %% [markdown]
# ## The paper's own goodness-of-fit metric
#
# Eq. (25), `Δ²/n = mean[log10(S_exp) − log10(S_cal)]²`, is what table 1 reports:
# 0.0229 (SMK) vs 0.0452 (MK) for HSG, 0.0430 vs 0.0575 for V79. Recomputed on the
# points the figures draw — a subset of the ~300 the paper fitted, hence smaller
# absolute values — the ordering holds, and it is the **highest-LET panels** that
# carry it. At low LET the two models coincide and the difference is noise.

# %%
print(f"\n{'cell':>5} {'panel':>6} {'LET':>7} {'SMK':>9} {'MK':>9}   ratio MK/SMK")
for cell_line in CELLS:
    panels = survival_panels(cell_line)
    for curve in panels:
        smk_d2 = delta2_per_point(cell_line, "smk", curve.panel)
        mk_d2 = delta2_per_point(cell_line, "mk", curve.panel)
        print(
            f"{cell_line:>5} {curve.panel:>6} {curve.let_keV_um:7.1f} "
            f"{smk_d2:9.4f} {mk_d2:9.4f}   {mk_d2 / smk_d2:8.1f}x"
        )
    letters = "".join(c.panel for c in panels)
    ps, pm = delta2_per_point(cell_line, "smk", letters), delta2_per_point(cell_line, "mk", letters)
    print(f"{cell_line:>5} {'ALL':>6} {'':>7} {ps:9.4f} {pm:9.4f}   {pm / ps:8.1f}x")

# %% [markdown]
# ## Panels: HSG (Fig. 2)

# %%
fig, axes = plt.subplots(3, 3, figsize=(15, 12))
for ax, curve in zip(axes.ravel(), survival_panels("HSG"), strict=True):
    exp = get_survival_curve("HSG", curve.panel, "exp")
    paper_mk = get_survival_curve("HSG", curve.panel, "mk")
    r = recon["HSG", curve.panel]
    ax.plot(curve.dose, curve.s, "-", color="0.55", lw=4, alpha=0.55, label="paper SMK")
    ax.plot(paper_mk.dose, paper_mk.s, "-", color="C3", lw=4, alpha=0.30, label="paper MK")
    ax.plot(DOSE, r.smk, "k-", lw=1.6, label="pyRadBio SMK")
    ax.plot(DOSE, r.mk, "--", color="C3", lw=1.4, label="pyRadBio MK")
    ax.plot(exp.dose, exp.s, "ko", ms=6, label="Furusawa 2000")
    ax.set(
        yscale="log",
        ylim=(1e-4, 1.4),
        xlim=(0, 10),
        title=f"({curve.panel})  {curve.ion} {curve.let_keV_um:g} keV/µm  "
        f"({r.ion.energy_MeV_u:.1f} MeV/u)",
    )
    if curve.panel == "a":
        ax.legend(fontsize=7, loc="lower left")
    if curve.panel in "ghi":
        ax.set_xlabel("dose [Gy]")
    if curve.panel in "adg":
        ax.set_ylabel("surviving fraction")
fig.suptitle(
    "HSG — Inaniwa & Kanematsu 2018 Fig. 2 reconstructed "
    "(thick = paper, thin = pyRadBio from the quoted LET alone)"
)
fig.tight_layout()
print()
save_figure(fig, "smk_reconstruction_survival.png", directory=FIGURES, dpi=140)
