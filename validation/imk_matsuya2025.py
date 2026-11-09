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
# # IMK (Matsuya et al. 2025): figs. 2, 4 and 7 reconstructed, without PHITS
#
# The **integrated microdosimetric-kinetic model** -- Matsuya, Saga, Wang & Sato
# 2025, doi:10.1002/mp.70040 -- is, in one line:
#
# ```
# -ln S = (α₀ + z*₁D β₀) D + F β₀ D²,   z*₁D = y*/(ρ π r_d²)
# ```
#
# The linear term is the modified MKM verbatim and `F` is the Lea-Catcheside
# time factor written from the repair rate `(a+c)`, so `IMKModel` subclasses
# `ModifiedMKM` and overrides only `β`, taking `F` from `core/protraction.py`.
#
# Every `y*` in the paper comes from the PHITS `[t-sed]` tally and none is
# printed, the photon one included. Two things make a reconstruction possible
# anyway: inverting eq. (1) on fig. 2's calculated LQ curves recovers the
# paper's own `z*₁D` (`benchmark/matsuya2025.py`), and `[t-sed]` evaluates
# Sato's analytical microdosimetric function, which this package ships. Panels
# 1, 2, 5, 6 and 8 are therefore scored against the paper's own numbers. The
# four microdosimetry routes (`microdosimetry_routes`) are `impact/cyl` (the
# MKM-family disk default), `impact/sph` (the sphere this paper states,
# sec. 2.1), `sato-AMF` (**the paper's own method**, drawn solid; faithful, not
# best) and `analytic` (Xapsos-Kellerer, restricted, delta tail folded in), all
# on the printed table-1 parameters. The experimental points in panel 6 are
# context, not a score: scoring against data would confound our microdosimetry
# with the paper's fit. Background and the full findings:
# `docs/references/IMK_Matsuya2025.md`.

# %%
import matplotlib.pyplot as plt
import numpy as np
from _common import pick, quick_mode

from pyradbio.benchmark.elsasser2010 import get_curve
from pyradbio.benchmark.matsuya2025 import (
    ANCHORS,
    PAPER_ROUTE,
    RECOVERED_Z_STAR_1D_GY,
    dose_rate_shape_fit,
    fig7_protocol,
    fig7_schedule,
    get_fig2_curve,
    get_fig4_curve,
    get_fig7_curve,
    microdosimetry_routes,
    paper_rbe_alpha,
    recovered_h,
    recovered_z_star_1D_from_mn,
    relative_mn,
)
from pyradbio.benchmark.metrics import ratio_summary
from pyradbio.core.protraction import DoseSchedule
from pyradbio.core.radiation import Ion
from pyradbio.models.mkm.imk import IMKModel
from pyradbio.plotting import figures_dir, save_figure
from pyradbio.track.stopping_power import SimpleBetheStoppingPower, energy_for_let_MeV_u

FIGURES = figures_dir(globals().get("__file__"))
if quick_mode():
    print("QUICK MODE (PYRADBIO_QUICK=1): 6 LET points instead of 30 -- numbers not for quoting")

SP = SimpleBetheStoppingPower()

# The four LETs the paper fits its carbon data at (secs. 3.1.1, 4.1).
PAPER_LETS = tuple(let for _, let in ANCHORS)


def carbon_at_let(let_keV_um: float) -> Ion:
    """The 12C energy whose LET in water is ``let_keV_um`` (bracket 0.6-600 MeV/u)."""
    energy = energy_for_let_MeV_u(
        SP, 6, 12, let_keV_um, branch="bracket", energy_bounds_MeV_u=(0.6, 600.0), xtol=1e-8
    )
    return Ion(Z=6, A=12, energy_MeV_u=energy)


# Each cell line's four routes, built once (the photon reference rides along).
ROUTES_BY_CELL = {cell: microdosimetry_routes(cell, stopping_power=SP) for cell in ("HSG", "SCC")}
ROUTES = ROUTES_BY_CELL["HSG"]
COLOURS = dict(zip(ROUTES, ("C0", "C1", "C2", "C3"), strict=True))


def style(name: str, base_width: float = 1.3) -> dict[str, object]:
    """Solid and heavier for the paper's own method, dashed and light otherwise."""
    if name == PAPER_ROUTE:
        return {"linestyle": "-", "lw": base_width + 0.9, "zorder": 4}
    return {"linestyle": "--", "lw": base_width - 0.2, "zorder": 3}


def route_label(name: str) -> str:
    return f"{name} (the paper's method)" if name == PAPER_ROUTE else name


LETS = np.geomspace(10.0, 500.0, pick(30, 6))
IONS = [carbon_at_let(let) for let in LETS]

# %% [markdown]
# ## The route spread, measured
#
# `z*₁D` is the whole radiation-quality content of `α`. The routes disagree
# **at low LET**, where saturation is irrelevant and the difference is pure
# geometry and straggling, not at high LET, where the `y₀ = 150 keV/µm`
# overkill correction pulls them together. (The printed "at LET 13" ratio is
# the first grid point, LET 10.)

# %%
z_star = {
    name: np.array([m.specific_energies(i).z_star_1D_Gy for i in IONS])
    for name, m in ROUTES.items()
}

reference = z_star["impact/cyl"]
print("z*_1D spread vs the family default (impact/cyl):")
for name, values in z_star.items():
    spread = ratio_summary(values, reference)
    print(
        f"  {name:>11}  x{spread.min_ratio:.2f}-{spread.max_ratio:.2f}"
        f"  (at LET 13: x{values[0] / reference[0]:.2f})"
    )

print("\nalpha [1/Gy] at the paper's four fitted LETs:")
paper_ions = [carbon_at_let(let) for let in PAPER_LETS]
header = "".join(f"{name:>12}" for name in ROUTES)
print(f"{'LET':>6} {'E[MeV/u]':>9}{header}")
for let, ion in zip(PAPER_LETS, paper_ions, strict=True):
    row = "".join(f"{m.alpha_beta(ion).alpha:12.3f}" for m in ROUTES.values())
    print(f"{let:6.0f} {ion.energy_MeV_u:9.1f}{row}")

# %%
fig, axes = plt.subplots(2, 4, figsize=(21.5, 10.0))


def reconstruction_panel(ax, cell: str, lets: tuple[float, ...], max_dose_Gy: float) -> None:
    """One Fig. 2 panel: the paper's curves and points, and ours on top.

    The photon arm is drawn but not predicted -- it *is* the recovered
    reference, so plotting our version of it would only redraw the paper's line.
    """
    grid = np.linspace(0.05, max_dose_Gy, 160)
    cell_routes = ROUTES_BY_CELL[cell]
    photon = get_fig2_curve(cell_line=cell, series="cal", let_keV_um=0.0)
    ax.semilogy(photon.dose, photon.s, "-", color="0.55", lw=1.4, label="paper, X-rays")
    pts = get_fig2_curve(cell_line=cell, series="exp", let_keV_um=0.0)
    lo, hi = pts.error_bars()
    ax.errorbar(pts.dose, pts.s, yerr=[lo, hi], fmt="o", ms=4, color="0.55", capsize=2, lw=0.8)
    for i, let in enumerate(lets):
        cal = get_fig2_curve(cell_line=cell, series="cal", let_keV_um=let)
        ax.semilogy(cal.dose, cal.s, "k-", lw=1.5, label="paper, carbon" if i == 0 else None)
        pts = get_fig2_curve(cell_line=cell, series="exp", let_keV_um=let)
        lo, hi = pts.error_bars()
        ax.errorbar(pts.dose, pts.s, yerr=[lo, hi], fmt="ks", ms=4, capsize=2, lw=0.8)
        ion = carbon_at_let(let)
        for name, model in cell_routes.items():
            ax.semilogy(
                grid,
                model.survival(ion, np.asarray(grid, dtype=float)),
                color=COLOURS[name],
                label=route_label(name) if i == 0 else None,
                **style(name, 1.1),
            )
        ax.annotate(
            f"{let:.0f} keV/µm",
            xy=(cal.dose[len(cal.dose) // 3], cal.s[len(cal.dose) // 3]),
            fontsize=7,
            color="k",
        )
    ax.set(xlim=(0, max_dose_Gy), ylim=(1e-4, 1.4), xlabel="dose [Gy]", ylabel="surviving fraction")
    ax.legend(fontsize=7, loc="lower left")


# --- panel 1: fig. 2a, HSG --------------------------------------------------
reconstruction_panel(axes[0, 0], "HSG", (13.0, 100.0), 12.0)
axes[0, 0].set_title(
    "Fig. 2a reconstruction (HSG)\nblack = the paper, dashed = our four routes on its parameters"
)

# --- panel 2: fig. 2b, SCC --------------------------------------------------
reconstruction_panel(axes[0, 1], "SCC", (46.0, 80.0), 16.0)
axes[0, 1].set_title(
    "Fig. 2b reconstruction (SCC VII)\nthe X-ray arm is the recovered reference, not a prediction"
)


# --- panel 3: z*_1D vs LET, per route, with the paper's anchors -------------
ax = axes[0, 2]
for name, values in z_star.items():
    ax.plot(LETS, values, color=COLOURS[name], label=route_label(name), **style(name))
for let in PAPER_LETS:
    ax.axvline(let, color="0.85", lw=0.8, zorder=0)
paper_z = [RECOVERED_Z_STAR_1D_GY[anchor] for anchor in ANCHORS]
ax.plot(PAPER_LETS, paper_z, "k*", ms=12, zorder=5, label="the paper's own (from fig. 2)")
ax.set(
    xscale="log",
    yscale="log",
    xlabel="LET [keV/µm]",
    ylabel="z*₁D  [Gy]",
    title="The quantity the paper imports from PHITS\n"
    "stars: its own values, recovered by inverting eq. (1)",
)
ax.legend(fontsize=8)

# --- panel 4: the spread, as a ratio ----------------------------------------
ax = axes[0, 3]
for name, values in z_star.items():
    ax.plot(LETS, values / reference, color=COLOURS[name], label=route_label(name), **style(name))
ax.axhline(1.0, color="0.7", lw=0.8)
ax.set(
    xscale="log",
    xlabel="LET [keV/µm]",
    ylabel="z*₁D  /  z*₁D (impact/cyl)",
    title="Route spread is a LOW-LET effect\n"
    "the y₀ = 150 keV/µm saturation closes it again above ~100",
)
ax.legend(fontsize=8)

# --- panel 5: our routes against the paper's four recovered anchors ---------
ax = axes[1, 0]
for name, model in ROUTES.items():
    ours = [model.specific_energies(carbon_at_let(let)).z_star_1D_Gy for let in PAPER_LETS]
    ax.plot(
        PAPER_LETS,
        [o / p - 1.0 for o, p in zip(ours, paper_z, strict=True)],
        marker="o",
        ms=7 if name == PAPER_ROUTE else 5,
        color=COLOURS[name],
        label=route_label(name),
        **style(name),
    )
ax.axhline(0.0, color="k", lw=1.0)
for cell, let in ANCHORS:
    ax.annotate(cell, xy=(let, -0.42), fontsize=7, ha="center", color="0.4")
ax.set(
    xscale="log",
    ylim=(-0.45, 0.65),
    xlabel="LET [keV/µm]",
    ylabel="our z*₁D / the paper's  -  1",
    title="Scored against the paper's OWN z*₁D\nrecovered from fig. 2 by inverting its eq. (1)",
)
ax.legend(fontsize=8)

# --- panel 6: RBE_alpha vs LET, on the RECOVERED photon reference -----------
# The HSG preset carries the fig. 2a photon reference (RECOVERED_Z_STAR_1D_GY
# at LET 0), so every route model scores eq. (6) as built.
ax = axes[1, 1]
Z_PHOTON_HSG = RECOVERED_Z_STAR_1D_GY[("HSG", 0.0)]
for name, model in ROUTES.items():
    ax.plot(
        LETS,
        [model.rbe_mn(model.alpha_beta(i)) for i in IONS],
        color=COLOURS[name],
        label=route_label(name),
        **style(name),
    )
for let in (13.0, 100.0):
    ax.plot(
        let,
        paper_rbe_alpha("HSG", let),
        "k*",
        ms=11,
        zorder=5,
        label="the paper's own lines" if let == 13.0 else None,
    )
data = get_curve(
    figure="fig4b_hsg_rbe_vs_let",
    cell_line="HSG",
    particle="carbon",
    series="data",
    variant="rbe_alpha",
)
ax.plot(data.x, data.y, "k.", ms=5, alpha=0.4, label="Furusawa 2000 HSG (context)")
ax.set(
    xscale="log",
    xlabel="LET [keV/µm]",
    ylabel="RBE_α = α*/α(photon)  [eq. 6]",
    title="Eq. (6), on the photon arm recovered from fig. 2a\n"
    f"z*₁D = {Z_PHOTON_HSG:.3f} Gy (y* = 3.6 keV/µm)",
)
ax.legend(fontsize=8)

# --- panel 7: the time factor, per cell line --------------------------------
ax = axes[1, 2]
durations_h = np.geomspace(1e-2, 100.0, 120)
for name, model, colour in (
    ("HSG / SCC  (a+c = 2.19 /h)", IMKModel.hsg_matsuya2025(), "C0"),
    ("DU145  (2.10 /h)", IMKModel.du145_matsuya2025(), "C1"),
    ("WI-38  (0.371 /h)", IMKModel.wi38_matsuya2025(), "C2"),
):
    ax.plot(
        durations_h * 60.0,
        [model.time_factor(DoseSchedule.continuous(4.0, duration_h=t)) for t in durations_h],
        "-",
        color=colour,
        label=name,
    )
ax.axvspan(4.0, 80.0, color="0.9", zorder=0)
ax.text(16.0, 0.06, "the paper's own\ndose-rate protocols", fontsize=7, ha="center")
ax.set(
    xscale="log",
    xlabel="dose-delivery time T [min]",
    ylabel="F  (Lea-Catcheside)",
    title="F = 2[x + e⁻ˣ - 1]/x², x = (a+c)T\n"
    "the repair rate is a CELL property; the schedule is not",
)
ax.legend(fontsize=8)

# --- panel 8: fig. 7, the dose-rate panel (secs. 3.1.3-3.1.4) ---------------
# The paper's line is continuous delivery (fig7_schedule); the sec. 3.1.4
# laboratory protocol is fractionated (fig7_protocol), and the gap is printed.
ax = axes[1, 3]
rate_grid = np.geomspace(1.0, 190.0, 140)

print("\nFig. 7 -- dose-rate dependence of the MN frequency (4 Gy, relative to 60 Gy/h):")
for cell_line, build_cell, colour in (
    ("DU145", IMKModel.du145_matsuya2025, "C1"),
    ("WI-38", IMKModel.wi38_matsuya2025, "C2"),
):
    model = build_cell()
    mean = get_fig7_curve(cell_line=cell_line, series="cal")
    band_hi = get_fig7_curve(cell_line=cell_line, series="cal_ci_hi")
    band_lo = get_fig7_curve(cell_line=cell_line, series="cal_ci_lo")
    ax.fill_between(
        band_hi.rate,
        np.interp(band_hi.rate, band_lo.rate, band_lo.mn),
        band_hi.mn,
        color=colour,
        alpha=0.13,
        lw=0,
        zorder=0,
    )
    ax.plot(
        mean.rate,
        mean.mn,
        "-",
        color=colour,
        lw=3.0,
        alpha=0.45,
        label=f"paper, {cell_line} (95% CI shaded)",
    )
    points = get_fig7_curve(cell_line=cell_line, series="exp")
    bar_lo, bar_hi = points.error_bars()
    ax.errorbar(
        points.rate,
        points.mn,
        yerr=[bar_lo, bar_hi],
        fmt="o",
        ms=4,
        color=colour,
        capsize=2,
        lw=0.8,
        zorder=4,
    )
    ours = [relative_mn(model, r) for r in rate_grid]
    ax.plot(
        rate_grid,
        ours,
        "--",
        color="k",
        lw=1.2,
        zorder=5,
        label="ours (eq. 1 time factor)" if cell_line == "DU145" else None,
    )
    # how well the paper's own line is reproduced, and what the panel can decide
    worst = max(
        abs(relative_mn(model, r) - v)
        for r, v in zip(mean.dose_rate_Gy_per_h, mean.relative_mn, strict=True)
    )
    residuals = {}
    for factor in (0.5, 1.0, 2.0):
        wrong = build_cell(repair_rate_per_h=factor * model.repair_rate_per_h)
        *_, residuals[factor] = dose_rate_shape_fit(
            mean, lambda r, m=wrong: m.time_factor(fig7_schedule(r))
        )
    span = max(mean.relative_mn) - min(mean.relative_mn)
    print(
        f"  {cell_line:6} (a+c = {model.repair_rate_per_h:5.3f}/h)  "
        f"panel span {span:.3f}   worst |ours - paper| {worst:.4f}"
    )
    print(
        f"           A + B·F shape residual   x0.5: {residuals[0.5]:.4f}   "
        f"printed: {residuals[1.0]:.4f}   x2: {residuals[2.0]:.4f}"
    )
    gaps = []
    for rate_Gy_per_min in (1.0, 0.5, 0.1, 0.05):
        rate_Gy_per_h = 60.0 * rate_Gy_per_min
        continuous = fig7_schedule(rate_Gy_per_h)
        gap = (
            model.time_factor(fig7_protocol(rate_Gy_per_min)) / model.time_factor(continuous) - 1.0
        )
        gaps.append(f"{rate_Gy_per_h:4.0f} Gy/h {abs(gap) * 100:.2f}%")
    print("           continuous vs the sec. 3.1.4 fractionated protocol: " + "  ".join(gaps))

ax.set(
    xscale="log",
    xlabel="dose rate [Gy/h]",
    ylabel="MN frequency, relative to 60 Gy/h",
    ylim=(0.0, 1.75),
    title="Fig. 7 reconstruction - 4 Gy, eq. (3)\n"
    "the DU145 arm is a real test of F; the WI-38 arm is not",
)
ax.set_xticks([1, 3, 10, 30, 100])
ax.set_xticklabels(["1", "3", "10", "30", "100"])
ax.minorticks_off()
ax.legend(fontsize=7, loc="lower right")

fig.tight_layout()
print()
save_figure(fig, "imk_microdosimetry_routes.png", directory=FIGURES, dpi=150)

# %% [markdown]
# ## Fig. 4: the one panel the model predicts with nothing left to choose
#
# Eq. (3), `F_MN = h ⟨w⟩ = h (−ln S)`, makes the micronucleus dose-response the
# survival LQ scaled by `h`: fig. 4 is a **zero-free-parameter** prediction from
# fig. 2a's `α`/`β` and table 1's `h`. Fitting it measures `h` (table 1's 0.275
# or table 2's HSG 0.245?), reads `h` off the quadratic term without knowing
# `z*₁D` (`β_m = h β₀`), and, divided back out, re-derives the unprinted PHITS
# inputs from a second, separately digitized figure.

# %%
fig4_lets = (0.0, 13.0, 100.0)
hsg = IMKModel.hsg_matsuya2025()
fig2a_lq = {
    let: get_fig2_curve(cell_line="HSG", series="cal", let_keV_um=let).lq_fit() for let in fig4_lets
}

print("\nFig. 4 -- h and z*_1D recovered from the micronucleus panel alone:")
print(f"{'LET':>6}{'alpha_m':>10}{'beta_m':>10}{'h':>9}{'z*_1D':>9}{'fig. 2a z*_1D':>15}")
for let in fig4_lets:
    curve = get_fig4_curve(series="cal", let_keV_um=let)
    alpha_m, beta_m, h = recovered_h(curve, beta0_Gy2=hsg.beta0)
    _, z_star_mn = recovered_z_star_1D_from_mn(curve, alpha0_Gy_inv=hsg.alpha0, beta0_Gy2=hsg.beta0)
    print(
        f"{let:6.0f}{alpha_m:10.4f}{beta_m:10.5f}{h:9.4f}{z_star_mn:9.3f}"
        f"{RECOVERED_Z_STAR_1D_GY[('HSG', let)]:15.3f}"
    )
print("  table 1 h = 0.275 (used); table 2 HSG-specific h = 0.245 (not used)")

# %%
fig_mn, mn_axes = plt.subplots(1, 2, figsize=(12.5, 5.0))

# --- left: the panel itself, and our routes' version of it ------------------
ax = mn_axes[0]
grid_mn = np.linspace(0.05, 12.0, 200)
for let in fig4_lets:
    cal = get_fig4_curve(series="cal", let_keV_um=let)
    colour = "0.55" if let == 0.0 else "k"
    ax.plot(cal.dose, cal.mn, "-", color=colour, lw=1.5, zorder=4)
    pts = get_fig4_curve(series="exp", let_keV_um=let)
    lo, hi = pts.error_bars()
    ax.errorbar(
        pts.dose,
        pts.mn,
        yerr=[lo, hi],
        fmt="o" if let == 0.0 else "s",
        ms=4,
        color=colour,
        capsize=2,
        lw=0.8,
        zorder=4,
    )
    # eq. (3) applied to the fig. 2a LQ: no fitting, and it must land on the line
    alpha, beta = fig2a_lq[let]
    ax.plot(
        grid_mn,
        hsg.mn_probability * (alpha * grid_mn + beta * grid_mn**2),
        ":",
        color="C3",
        lw=2.4,
        zorder=5,
        label="h × (fig. 2a LQ), h = 0.275" if let == 0.0 else None,
    )
    ax.annotate(
        "X-rays" if let == 0.0 else f"C {let:.0f} keV/µm",
        xy=(cal.dose[len(cal.dose) // 2], cal.mn[len(cal.dose) // 2]),
        fontsize=7,
        color=colour,
    )
for let in (13.0, 100.0):
    ion = carbon_at_let(let)
    for name, model in ROUTES.items():
        lq = model.alpha_beta(ion)
        ax.plot(
            grid_mn,
            [model.micronucleus_frequency(d, lq) for d in grid_mn],
            color=COLOURS[name],
            label=route_label(name) if let == 13.0 else None,
            **style(name, 1.1),
        )
ax.set(
    xlim=(0, 12.0),
    ylim=(0, 3.0),
    xlabel="dose [Gy]",
    ylabel="MN per binucleated cell",
    title="Fig. 4 reconstruction (HSG)\nblack = the paper; dotted red = its own fig. 2a curves × h",
)
ax.legend(fontsize=7, loc="upper left")

# --- right: h, measured pointwise -------------------------------------------
ax = mn_axes[1]
for let, colour in zip(fig4_lets, ("0.55", "C0", "C1"), strict=True):
    cal = get_fig4_curve(series="cal", let_keV_um=let)
    alpha, beta = fig2a_lq[let]
    doses = np.array([d for d in cal.dose if d > 0.05])
    ratio = np.array([f for d, f in zip(cal.dose, cal.mn, strict=True) if d > 0.05]) / (
        alpha * doses + beta * doses**2
    )
    ax.plot(
        doses,
        ratio,
        "-",
        color=colour,
        lw=1.6,
        label="X-rays" if let == 0.0 else f"C {let:.0f} keV/µm",
    )
ax.axhline(0.275, color="k", lw=1.2, label="table 1: h = 0.275 (all cell lines)")
ax.axhline(0.245, color="C3", lw=1.2, ls="--", label="table 2: h = 0.245 (HSG)")
ax.set(
    xlim=(0, 12.0),
    ylim=(0.20, 0.33),
    xlabel="dose [Gy]",
    ylabel="F_MN(D)  /  (−ln S)(D)",
    title="h, read off the figure rather than assumed\n"
    "flat in dose ⇒ eq. (3) holds; the level picks the table",
)
ax.legend(fontsize=8, loc="lower right")

fig_mn.tight_layout()
save_figure(fig_mn, "imk_micronuclei.png", directory=FIGURES, dpi=150)

# %% [markdown]
# ## What this says
#
# The photon reference is recovered (fig. 2a: `z*₁D = 1.045 Gy`, `y* = 3.6
# keV/µm`; fig. 2b independently 3.4), and it is not Furusawa's photon fit; the
# faithful `sato-AMF` route is not the best-scoring one (panel 5); the paper's
# four anchors are beam-specific, not one curve in LET; DU145's fig. 7 arm is a
# sharp test of `F` (shape residual 0.0019, 25x worse at half or double
# `a+c`), WI-38's is not, and neither hands back a photon reference; fig. 4
# measures `h = 0.275` (table 1, not table 2) and re-recovers fig. 2a's `z*₁D`
# to 0.6-1.5%. Each of these is written up in `docs/references/IMK_Matsuya2025.md`
# ("What fig. 2 gives back", "What fig. 4 settles", "What fig. 7 validates").
# Four observations live only here:
#
# 1. **The route spread is a low-LET effect** (panel 4): above ~100 keV/µm the
#    `y₀ = 150` saturation flattens it; below, the routes differ by geometry and
#    straggling. The domain shape alone (disk vs sphere) is 1.38x at the
#    lowest LET.
# 2. **Two curves are not smooth.** The `sato-AMF` ratio dips near LET 25 and
#    the `analytic` one kinks around 20-40: engine artifacts (energy
#    interpolation of the bundled Sato coefficients; the restricted/delta-tail
#    switchover), not physics. Read them as a band.
# 3. **At high LET the shape goes wrong, and not because of the route.** Past
#    ~150 keV/µm our `RBE_α` keeps climbing where the Furusawa points turn
#    over -- the MKM family's overkill behaviour, inherited untouched; the
#    paper's fig. 5 stops before that.
# 4. **Fig. 7's WI-38 arm is inside the CI too, for a dull reason.** Our forward
#    prediction from printed parameters sits inside the paper's 95% CI at every
#    node of *both* arms, but WI-38's whole effect is ~0.03 in y because
#    `α₀ = 0.59 Gy⁻¹` leaves little for `F`, which multiplies `β` only.
#
# Before fig. 2 gave it back, the photon `y_D` could only be bracketed: 2.5
# keV/µm (the MKM-family megavoltage default) to 4.53 keV/µm (250 kVp X-rays,
# Matsuya et al. 2015, J. Radiat. Res. 56:90, sec. 3); at these `y` the
# saturation is negligible, so `y* ≈ y_D`.
#
# Not reconstructed: fig. 5 (`RBE_MN` vs LET for C, Ne, Si, Ar, Fe; Si/Ar/Fe lie
# beyond the bundled Sato AMF coefficients) and fig. 8 (a separate parameter
# set re-fitted to the MN data).
