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
# # NanOx track core vs penumbra: Alcocer-Ávila et al. 2023 Table 5
#
# Alcocer-Ávila, Monini, Cunha, Testa & Beuve 2023, *Front. Phys.* **11**:1011062,
# [doi:10.3389/fphy.2023.1011062](https://doi.org/10.3389/fphy.2023.1011062),
# sec. 3.3 and Table 5 (p. 12): the average ratio $\langle E_p/E_c\rangle$ of the
# energy deposited in the track **penumbra** to that deposited in the **core**,
# for H, He and C ions at 1–150 MeV/u. Each value is the mean over 1,000 runs of
# the group's LQD Monte Carlo, with the core a parallelepiped of 200 nm square
# cross-section around the path. It is the quantitative side of
# `examples/nanox_local_and_global_events.py` panel 2: the core is ~5e-4 of the
# nucleus volume yet carries all the radiation-quality dependence, while the
# penumbra is treated as the reference photon radiation (2022 Approximation 2).
#
# pyRadBio computes no $\langle E_p/E_c\rangle$ of its own — the core/penumbra
# energy split comes from the non-public LQD Monte Carlo — so this is a plotted
# reference, not a model comparison. No external data and no quick-mode knobs:
# quick and full mode are the same run.

# %%
import matplotlib.pyplot as plt
import numpy as np
from _common import quick_mode

from pyradbio.plotting import figures_dir, save_figure

FIGURES = figures_dir(globals().get("__file__"))
if quick_mode():
    print("quick mode (PYRADBIO_QUICK=1): this script has no cost knobs; same run as full mode")

# Alcocer-Avila et al. 2023, doi:10.3389/fphy.2023.1011062, Table 5 (p. 12):
# (energy [MeV/u], <E_p/E_c>, standard deviation) per ion.
EP_OVER_EC = {
    "H": [
        (1, 0.003, 4.7e-5),
        (2, 0.056, 4.0e-4),
        (10, 0.359, 3.0e-3),
        (50, 0.621, 1.5e-2),
        (75, 0.640, 2.0e-2),
        (100, 0.472, 1.1e-2),
        (150, 0.374, 1.2e-2),
    ],
    "He": [
        (1, 0.003, 2.4e-5),
        (2, 0.056, 1.9e-4),
        (10, 0.350, 2.1e-3),
        (50, 0.607, 7.3e-3),
        (75, 0.633, 1.0e-2),
        (100, 0.485, 6.4e-3),
        (150, 0.381, 7.1e-3),
    ],
    "C": [
        (1, 0.003, 4.5e-5),
        (2, 0.055, 6.4e-4),
        (10, 0.355, 7.1e-4),
        (50, 0.606, 1.7e-3),
        (75, 0.632, 2.4e-3),
        (100, 0.475, 1.6e-3),
        (150, 0.385, 1.9e-3),
    ],
}

# %% [markdown]
# ## The printed ratio, and how far it depends on the ion
#
# The paper draws two conclusions from the table (sec. 3.3): the ratio "depends
# essentially on the ion's energy, but not its type", and it peaks between 50 and
# 100 MeV/u. The species spread at each energy is $(\max - \min)/\mathrm{mean}$
# over H, He, C.

# %%
energies = [row[0] for row in EP_OVER_EC["H"]]
table = np.array([[row[1] for row in rows] for rows in EP_OVER_EC.values()])
print("Alcocer-Avila et al. 2023 Table 5, <E_p/E_c>")
print(f"{'E [MeV/u]':>10} " + " ".join(f"{ion:>6}" for ion in EP_OVER_EC) + "   species spread")
for j, energy in enumerate(energies):
    column = table[:, j]
    spread = 100.0 * (column.max() - column.min()) / column.mean()
    print(f"{energy:>10} " + " ".join(f"{v:>6.3f}" for v in column) + f"   {spread:6.1f}%")
peaks = {ion: energies[int(np.argmax(row))] for ion, row in zip(EP_OVER_EC, table, strict=True)}
print("peak energy per ion [MeV/u]: " + ", ".join(f"{ion} {e}" for ion, e in peaks.items()))

fig, ax = plt.subplots(figsize=(6.4, 4.6))
for (ion, rows), marker in zip(EP_OVER_EC.items(), ["s", "^", "v"], strict=True):
    e, ratio, sd = (np.array(col) for col in zip(*rows, strict=True))
    ax.errorbar(e, ratio, yerr=sd, fmt=marker + "--", ms=6, lw=1.2, capsize=3, label=ion)
ax.set_xscale("log")
ax.set_xlabel("energy [MeV/u]")
ax.set_ylabel(r"$\langle E_p/E_c\rangle$")
ax.set_title("Alcocer-Ávila et al. 2023 Table 5 (printed, ±1 SD)")
ax.legend(frameon=False, ncol=3, loc="lower center")
ax.grid(alpha=0.3)
fig.tight_layout()
save_figure(fig, "nanox_alcocer_avila2023.png", directory=FIGURES, dpi=140)
