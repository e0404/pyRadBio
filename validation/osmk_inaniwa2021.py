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
# # OSMK reconstruction: Inaniwa et al. 2021, figs. 5-7
#
# The paper's three **oxygen** figures — the oxygen enhancement ratio at 10%
# survival against dose-averaged LET, in every case the hypoxic dose over the
# aerobic (160 mmHg) one:
#
# * **figs. 5-6**: He/C/Ne on HSG and V79, anoxic (0 mmHg);
# * **fig. 7**: C/N/O/Si on CHO at **four** pressures — 0, 1.14, 3.8, 15.2 mmHg.
#
# All thirty series are digitized in `pyradbio.benchmark.inaniwa2021`; no
# external data.
#
# **Watch the LET axis — the three panels are not alike.** Figs. 5-6 run from
# 15 keV/µm, not 10; fig. 7 really does start at 10. Origin labels only 100 and
# 1000 and leaves the left frame edge bare in all three, and on figs. 5-6
# anchoring it at 10 compresses LET by up to ×1.5 at the left while leaving the
# right edge exact — invisible on the plot, and it moves the HSG residual below
# from +3% to +15%. The tick marks give each panel to within 0.5 px; see the
# benchmark module.
#
# This is the end-to-end test of the whole oxygen chain, and the only one there
# is. Everything upstream is separately pinned — the SMK microdosimetry against
# Inaniwa 2018 fig. 4, the table-1 parameters against the paper — so what is
# exercised here is exactly
#
# ```
# z̄_d,D  →  Rmax(z̄_d,D)  (eq. 14)  →  R(pO₂)  (eq. 13)
#        →  α', β'  (eqs. 17-18)   →  S(D)  (eq. 24)   →  D₁₀ ratio
# ```
#
# starting from nothing but the ion species, the quoted LET (inverted for energy
# with `energy_for_let_MeV_u`) and the oxygen pressure.
#
# **These are OSMK-2021 curves.** Figs. 5-7 predate the 2023 revision, so they
# are ground truth for `OSMK2021` only; `OSMK2023` is a separate refit of the
# same data and is drawn for context, not scored against them.
#
# ## What it shows
#
# 1. **CHO (fig. 7) reconstructs at 0.4% RMS with the published parameters
#    untouched**, over four ion species and four oxygen pressures. It is also the
#    only test anywhere of `R(pO₂)` at *intermediate* pressures, and hence of the
#    assumed `K = 3 mmHg`: figs. 5-6 use only the two endpoints of the Alper &
#    Howard-Flanders curve. The 1.14 / 3.8 / 15.2 mmHg arms come in at
#    0.4 / 0.3 / 0.2%.
# 2. **HSG reconstructs well too**: median 1.03, RMS 3.1% in log, and the
#    residual is a near-*uniform* offset rather than a shape error. That is
#    consistent with an already-measured bias — the fig.-4 reconstruction found
#    our uncorrected `z̄_d,D` runs ~7% low, and correcting by exactly that
#    independently-obtained factor drops HSG to **1.4% RMS**. Panel 4.
# 3. **V79 does not**: median 1.07, RMS 9.3%, up to 18% — a *mid-LET bulge* with
#    near-zero error at both ends, i.e. the eq.-(14) sigmoid sits at the wrong
#    LET rather than the wrong height.
# 4. **Panel 5 says why that is not our fault.** Eq. (14) reads `z̄_d,D` only
#    through `(z̄_d,D/z_R)^γ`, so scanning that one ratio is the whole diagnostic
#    — and the correction each cell line needs is **not monotone in the domain
#    radius**: ×1.07 at 0.23 µm, ×1.25 at 0.29, ×1.00 at 0.35. An `r_d`-dependent
#    bias in our microdosimetry is the only thing that could make the requirement
#    differ between cell lines while the code stays fixed, and this rules it out.
#    Also ruled out: a transcription error (table 1 re-read from the rendered PDF
#    page), the eq.-(24) expansion ceiling (anoxic `D₁₀` reaches at most 45% of
#    it), and the digitization. Open; tracked in `ROADMAP.md`.
#
# **Runtime.** Full mode (the default, and the only mode whose numbers are
# quoted) samples 18 LETs per HSG/V79 series and 8 per CHO series, and scans 13
# rescaling factors in panel 5. `PYRADBIO_QUICK=1` thins all three for a smoke
# run with the same figure name.
#
# This file is a [jupytext](https://jupytext.readthedocs.io) percent script.

# %%
import matplotlib.pyplot as plt
import numpy as np
from _common import pick, quick_mode
from matplotlib.colors import to_rgb
from matplotlib.lines import Line2D
from matplotlib.ticker import ScalarFormatter

from pyradbio.benchmark.inaniwa2018 import Z_D_BIAS  # measured in validation/smk_inaniwa2018.py
from pyradbio.benchmark.inaniwa2021 import (
    measured_points,
    model_curves,
    model_oer,
    oer_ratios,
    pressures,
    reconstruction_lets,
    with_scaled_z_domain,
)
from pyradbio.benchmark.metrics import ratio_summary
from pyradbio.models.mkm.osmk import OSMK2021, OSMK2023
from pyradbio.plotting import figures_dir, save_figure
from pyradbio.track.stopping_power import SimpleBetheStoppingPower

FIGURES = figures_dir(globals().get("__file__"))
if quick_mode():
    print("QUICK MODE (PYRADBIO_QUICK=1): thinned LET sampling and scan -- numbers not quotable")

#: The LET -> energy inversion keeps the analytic Bethe stopping power the
#: reconstruction was scored with (the MKM-family models themselves use the
#: library default; that mixed state is kept here).
SP = SimpleBetheStoppingPower()
#: LETs per series in the tables, panel 3 and the scan table. CHO carries 9
#: species-pressure combinations, so it is sampled less densely.
SAMPLES = pick({"HSG": 18, "V79": 18, "CHO": 8}, {"HSG": 5, "V79": 5, "CHO": 3})
#: LETs per series in panels 1, 2 and 4.
PANEL_SAMPLES = pick(18, 5)
#: Panel 5: rescaling factors scanned, and LETs per series at each (13 factors x
#: 3 cell lines, so each is kept cheap).
SCAN_FACTORS = np.linspace(0.95, 1.40, pick(13, 4))
SCAN_SAMPLES = pick(5, 3)

COLOURS = {"He": "C0", "C": "C3", "N": "C1", "O": "C4", "Ne": "C2", "Si": "C5"}
#: Fig. 7 carries two categorical dimensions, so it needs two visual channels --
#: and it is clearest to encode them the way the paper itself does: pO2 by
#: colour, ion by line style.
PO2_COLOURS = {0.0: "black", 1.14: "tab:green", 3.8: "tab:blue", 15.2: "tab:red"}
ION_STYLES = {"C": "-", "N": "--", "O": ":", "Si": "-."}
#: ...and the paper's own marker shapes, so the measured points carry the ion too
ION_MARKERS = {"C": "o", "N": "D", "O": "^", "Si": "v"}


def tinted(colour: str, weight: float = 0.38, toward: str = "0.90") -> tuple[float, ...]:
    """``colour`` mixed toward light grey, for the paper's own curves.

    Washed out enough to sit behind ours, but keeping the hue -- in fig. 7 the
    colour *is* the oxygen pressure, so a plain grey would throw that away.
    """
    c, g = np.array(to_rgb(colour)), np.array(to_rgb(toward))
    return tuple(weight * c + (1.0 - weight) * g)


#: The microdosimetry (``z_d,D``, ``z*_d,D``, ``z_n,D``) depends only on the site
#: geometry and the ion, never on the oxygen parameters scanned below, so the
#: library's spectrum-MKM microdosimetry cache scores each (site, ion) once. The
#: panel-5 scan asks for the same (site, ion) hundreds of times over; uncached,
#: the original script took about twenty minutes.
CELLS: dict[str, tuple[OSMK2021, OSMK2023]] = {
    "HSG": (OSMK2021.hsg_inaniwa2021(), OSMK2023.hsg_inaniwa2023()),
    "V79": (OSMK2021.v79_inaniwa2021(), OSMK2023.v79_inaniwa2023()),
    "CHO": (OSMK2021.cho_inaniwa2021(), OSMK2023.cho_inaniwa2023()),
}


# %% [markdown]
# ## The numbers

# %%
for cell, (m21, m23) in CELLS.items():
    print(f"=== {cell} (r_d = {m21.domain_radius_um} um, z_R = {m21.z_R_Gy} Gy) ===")
    print(f"  {'pO2':>5} {'sp':>3} {'LET range':>13} {'median':>7} {'RMSlog':>7} {'max|d|':>7}")
    # The OSMK-2021 pool, built inline so each series is evaluated only once. It is
    # oer_ratios(m21, cell, SP, SAMPLES[cell]); that helper's series order and NaN
    # policy are pinned by TestReconstructionHelpers (tests/benchmark/test_inaniwa2021.py).
    pooled: list[float] = []
    for po2 in pressures(cell):
        for species, curve in model_curves(cell, po2).items():
            lets = reconstruction_lets(curve, SP, SAMPLES[cell])
            ours, paper = model_oer(m21, curve, lets, SP), curve.interp(lets)
            row = ratio_summary(ours, paper)
            print(
                f"  {po2:5.2f} {species:>3} {lets[0]:5.0f}-{lets[-1]:6.0f} "
                f"{row.median:7.3f} {100 * row.rms_log:6.1f}% {100 * row.max_abs_rel:6.1f}%"
            )
            r21 = ours / paper
            pooled += list(r21[~np.isnan(r21)])
    a = np.array(pooled)
    b = oer_ratios(m23, cell, SP, SAMPLES[cell])
    s21, s23 = ratio_summary(a, np.ones_like(a)), ratio_summary(b, np.ones_like(b))
    print(
        f"  OSMK 2021 overall: median {s21.median:.3f}  "
        f"RMSlog {100 * s21.rms_log:.1f}%  "
        f"max {100 * s21.max_abs_rel:.1f}%"
    )
    print(
        f"  OSMK 2023 overall: median {s23.median:.3f}  "
        f"RMSlog {100 * s23.rms_log:.1f}%"
        "   (a separate refit -- not this figure's ground truth)\n"
    )

# %% [markdown]
# ## The diagnostic: which cell line needs its `z̄_d,D / z_R` moved, and by how much
#
# Eq. (14) reads `z̄_d,D` only through `(z̄_d,D/z_R)^γ`, so *"our `z̄_d,D` is x%
# low"* and *"the published `z_R` is x% high"* are the same statement from inside
# these figures — they cannot be separated here. Scanning that one ratio is
# therefore the whole diagnostic, and the minimum for each cell line is the
# correction that cell would need.


# %%
def rms_log_percent(model: OSMK2021, cell: str, n: int) -> float:
    """RMS log residual [%] of ``model`` against one figure, pooled."""
    return float(100 * np.sqrt(np.mean(np.log(oer_ratios(model, cell, SP, n)) ** 2)))


print("z̄_d,D / z_R rescaling scan (OSMK 2021, RMS log vs the paper's own curves):")
print(f"  {'factor':>7} " + " ".join(f"{c:>18}" for c in CELLS))
print(
    f"  {'':>7} "
    + " ".join(f"{'r_d=' + str(m.domain_radius_um) + ' um':>18}" for m, _ in CELLS.values())
)
for factor in (1.00, Z_D_BIAS, 1.15, 1.25):
    row = {
        cell: rms_log_percent(with_scaled_z_domain(m21, factor), cell, SAMPLES[cell])
        for cell, (m21, _m23) in CELLS.items()
    }
    print(f"  x{factor:.2f}   " + " ".join(f"{row[c]:17.1f}%" for c in CELLS))
print(
    "\n  The minima sit near x1.07 (HSG, r_d 0.23 um), x1.25 (V79, 0.29) and x1.00"
    "\n  (CHO, 0.35) -- not monotone in r_d, and worst in the middle. So the"
    "\n  correction V79 needs is not a domain-size effect in our microdosimetry."
    "\n  (Panel 5 locates each minimum on a coarse grid; the values above come from"
    "\n  refitting eq. 14 -- see docs/references/OSMK_Inaniwa2021_2023.md.)"
)

# %% [markdown]
# ## Figure

# %%
fig, axes = plt.subplots(2, 3, figsize=(16.5, 8.5))
axes = axes.ravel()
PANEL_OF = {"HSG": 5, "V79": 6, "CHO": 7}


def let_ticks(ax: plt.Axes) -> None:
    ax.set_xticks([20, 50, 100, 200, 500, 1000])
    ax.xaxis.set_major_formatter(ScalarFormatter())
    ax.xaxis.set_minor_formatter(plt.NullFormatter())


# --- 1-2. the two anoxic figures reconstructed ------------------------------
for panel, cell in enumerate(("HSG", "V79")):
    m21 = CELLS[cell][0]
    ax = axes[panel]
    for species, curve in model_curves(cell).items():
        colour = COLOURS[species]
        ax.plot(curve.let, curve.values, "--", color=tinted(colour), lw=2.0, zorder=1)
        pts = measured_points(cell)[species]
        ax.plot(pts.let, pts.values, "o", color=colour, ms=3.5, alpha=0.55, mew=0, zorder=2)
        lets = reconstruction_lets(curve, SP, PANEL_SAMPLES)
        ours = model_oer(m21, curve, lets, SP)
        ax.plot(lets, ours, "-", color=colour, lw=1.8, label=species, zorder=3)
    ax.axhline(1.0, color="0.85", lw=0.8, zorder=0)
    ax.set(
        xscale="log",
        ylim=(0.9, 3.8),
        xlabel="dose-averaged LET [keV/µm]",
        ylabel="OER at 10% survival",
        title=f"{cell} — paper fig. {PANEL_OF[cell]}  (anoxic)\n"
        "dashed+dots: paper curve & measured;  solid: ours",
    )
    ax.legend(fontsize=8, title="ion", title_fontsize=8)
    let_ticks(ax)

# --- 3. fig. 7: CHO across the whole pO2 axis -------------------------------
# Two categorical dimensions at once, encoded as the paper itself does it:
# colour is the oxygen pressure, line style is the ion. The paper's own curves
# go underneath as wide pale grey, ours on top thin and saturated, so the fit is
# read as "does the coloured line sit in the grey band".
ax = axes[2]
m21 = CELLS["CHO"][0]
for po2 in pressures("CHO"):
    colour = PO2_COLOURS[po2]
    for species, curve in model_curves("CHO", po2).items():
        style = ION_STYLES[species]
        ax.plot(curve.let, curve.values, style, color=tinted(colour), lw=3.0, zorder=1)
        lets = reconstruction_lets(curve, SP, SAMPLES["CHO"])
        ax.plot(lets, model_oer(m21, curve, lets, SP), style, color=colour, lw=1.5, zorder=3)
    for species, p in measured_points("CHO", po2).items():
        ax.errorbar(
            p.let,
            p.values,
            yerr=p.sigma,
            fmt=ION_MARKERS[species],
            color=colour,
            ms=3.5,
            lw=0.9,
            capsize=2,
            zorder=4,
        )
ax.axhline(1.0, color="0.85", lw=0.8, zorder=0)
ax.set(
    xscale="log",
    ylim=(0.9, 3.8),
    xlabel="dose-averaged LET [keV/µm]",
    ylabel="OER at 10% survival",
    title="CHO — paper fig. 7  (C, N, O, Si × four pO₂)\n"
    "pale: paper, saturated: ours — 0.4% RMS untouched",
)
po2_legend = ax.legend(
    [Line2D([], [], color=PO2_COLOURS[p], lw=1.6) for p in pressures("CHO")],
    [f"{p:g} mmHg" for p in pressures("CHO")],
    fontsize=7,
    title="pO₂ (colour)",
    title_fontsize=7,
    loc="upper right",
    bbox_to_anchor=(1.0, 1.0),
)
ax.add_artist(po2_legend)
ax.legend(
    [
        Line2D([], [], color="0.35", ls=ION_STYLES[s], marker=ION_MARKERS[s], ms=4)
        for s in ION_STYLES
    ],
    list(ION_STYLES),
    fontsize=7,
    title="ion (style + marker)",
    title_fontsize=7,
    loc="upper right",
    bbox_to_anchor=(1.0, 0.60),
)
let_ticks(ax)

# --- 4. the residual, and what the z_d,D bias explains ----------------------
ax = axes[3]
for cell, marker in (("HSG", "o"), ("V79", "s")):
    m21 = CELLS[cell][0]
    for factor, style, alpha in ((1.00, "-", 1.0), (Z_D_BIAS, ":", 0.75)):
        model = with_scaled_z_domain(m21, factor)
        lets_all, rat_all = [], []
        for _species, curve in model_curves(cell).items():
            lets = reconstruction_lets(curve, SP, PANEL_SAMPLES)
            lets_all += list(lets)
            rat_all += list(model_oer(model, curve, lets, SP) / curve.interp(lets))
        order = np.argsort(lets_all)
        ax.plot(
            np.array(lets_all)[order],
            100 * (np.array(rat_all)[order] - 1),
            style,
            marker=marker if factor == 1.0 else None,
            ms=2.5,
            lw=1.3,
            alpha=alpha,
            color="C0" if cell == "HSG" else "C3",
            label=f"{cell}" + ("" if factor == 1.0 else f"  (z̄$_{{d,D}}$ ×{factor:.2f})"),
        )
ax.axhline(0.0, color="0.75", lw=0.8)
ax.set(
    xscale="log",
    xlabel="dose-averaged LET [keV/µm]",
    ylabel="our OER / paper OER − 1  [%]",
    title="HSG is a uniform offset the measured z̄$_{d,D}$ bias explains;\n"
    "V79 is a mid-LET bulge it does not",
)
ax.legend(fontsize=7)
let_ticks(ax)

# --- 5. THE diagnostic: where each cell line's z/z_R scan bottoms out -------
ax = axes[4]
CELL_COLOUR = {"HSG": "C0", "V79": "C3", "CHO": "C2"}
for cell, (m21, _m23) in CELLS.items():
    rms = [
        rms_log_percent(with_scaled_z_domain(m21, float(f)), cell, SCAN_SAMPLES)
        for f in SCAN_FACTORS
    ]
    best = SCAN_FACTORS[int(np.argmin(rms))]
    ax.plot(
        SCAN_FACTORS,
        rms,
        "-",
        color=CELL_COLOUR[cell],
        label=f"{cell}  ($r_d$={m21.domain_radius_um:g} µm)  min at ×{best:.2f}",
    )
    ax.plot([best], [min(rms)], "o", color=CELL_COLOUR[cell], ms=5)
ax.axvline(Z_D_BIAS, color="0.6", ls=":", lw=1.0)
ax.text(Z_D_BIAS, ax.get_ylim()[1] * 0.95, " measured z̄$_{d,D}$ bias", fontsize=7, color="0.4")
ax.set(
    xlabel=r"scale applied to $\bar z_{d,D}/z_R$  (eq. 14 sees only this ratio)",
    ylabel="RMS log residual vs the paper [%]",
    title="The correction each cell line needs is NOT monotone in $r_d$\n"
    "— so it is not a domain-size effect in our $\\bar z_{d,D}$",
)
ax.legend(fontsize=7)

# --- 6. why V79 is the more sensitive of the three: the eq.-(14) sigmoid ----
ax = axes[5]
z = np.geomspace(0.5, 3000.0, 300)
for cell, colour in CELL_COLOUR.items():
    m = CELLS[cell][0]
    ax.plot(
        z,
        [m.max_radioresistance(float(zi)) for zi in z],
        color=colour,
        label=f"{cell}: $z_R$={m.z_R_Gy:g}, $\\gamma$={m.gamma_exponent:g}, $R_m$={m.R_m:g}",
    )
    ax.axvline(m.z_R_Gy, color=colour, ls=":", lw=0.8)
ax.set(
    xscale="log",
    xlabel=r"$\bar z_{d,D}$ [Gy]",
    ylabel=r"$R_{max}(\bar z_{d,D})$  (eq. 14)",
    title="V79's sigmoid is the steepest ($\\gamma$ 1.6) and shortest\n"
    "($R_m$ 2.5), so a given $\\bar z$ error costs it most",
)
ax.legend(fontsize=7)

fig.tight_layout()
print()
save_figure(fig, "osmk_reconstruction.png", directory=FIGURES, dpi=150)
