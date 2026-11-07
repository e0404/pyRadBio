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
# # Inter-model comparison: alpha(E) and beta(E) for a low- and a high-LET ion
#
# The **LEM family (I-IV)**, the **MKM family (MCF MKM, mMKM)** and **RMF** all map
# a radiation field to the same LQ `(alpha, beta)`. Here they sit side by side on
# **one shared photon reference**, for a **low-LET** (proton) and a **high-LET**
# (carbon) ion, so the *model* differences — not parameter differences — are what
# you see. Fully self-contained (no external data, no pyMKM extra).
#
# The shared reference is a documented V79 set (Scholz & Kraft / Elsässer &
# Scholz 2007): `alpha_x = 0.18 Gy^-1`, `beta_x = 0.028 Gy^-2`, `D_t = 30 Gy`,
# `R_n = 5 um`.
#
# * **LEM I-IV** take `(alpha_x, beta_x, D_t)` and `R_n` directly: the same
#   calibration with progressively refined track/cluster physics, isolated at a
#   common shoulder. The literature *re-fits* `D_t` **downward** for the
#   cluster-extended versions (LEM II/III: `30 -> 7.5 Gy` for V79), so both are
#   shown — shared `D_t = 30 Gy` solid, the refit `7.5 Gy` dashed.
# * **MCF MKM & mMKM** turn a single-event lineal-energy spectrum `d(y)` (here from
#   pyRadBio's analytic engine on a 0.3 um site) into `(alpha, beta)`. Given the
#   same reference as `(alpha_ref, beta_ref)`, they derive the intrinsic `alpha0`
#   via the Hartzell/Kase relation `alpha0 = alpha_ref - beta0 * z_d(y_D,ref)`
#   (Parisi 2022 eq 14) — the proper conversion, not a crude `alpha0 = alpha_x`.
#   They keep their native domain/nucleus geometry (`R_n` differs slightly).

# %%
from __future__ import annotations

import warnings

import matplotlib.pyplot as plt
import numpy as np

from pyradbio.core.cell import CellTarget
from pyradbio.core.photon import ClusterParameters, PhotonParameters
from pyradbio.core.radiation import Ion
from pyradbio.micro import SphericalSite
from pyradbio.micro.lineal_energy import AnalyticSpectrumEngine
from pyradbio.models.lem.lem1 import LEM1
from pyradbio.models.lem.lem2 import LEM2
from pyradbio.models.lem.lem3 import LEM3
from pyradbio.models.lem.lem4 import LEM4
from pyradbio.models.mkm.spectrum_models import MCFMKMModel, ModifiedMKM
from pyradbio.models.rmf import X_RAY_200KVP_REFERENCE, RMFModel
from pyradbio.plotting import figures_dir, save_figure
from pyradbio.track.stopping_power import SimpleBetheStoppingPower

FIGURES = figures_dir(globals().get("__file__"))

# Shared V79 photon reference (Scholz & Kraft / Elsasser & Scholz 2007).
ALPHA_X, BETA_X, D_T, D_T_REFIT, R_N = 0.18, 0.028, 30.0, 7.5, 5.0

# Low-LET (proton) and high-LET (carbon) probes; ASCII labels print portably.
IONS = [(1, 1, "1H (low-LET)"), (6, 12, "12C (high-LET)")]
ENERGIES = np.logspace(0.0, np.log10(500.0), 14)  # 1 .. 500 MeV/u

# %%
# LEM II/III additionally consume ClusterParameters (the SSB->DSB stagger machinery).
photon = PhotonParameters(alpha_x=ALPHA_X, beta_x=BETA_X, D_t=D_T)
photon_cluster = PhotonParameters(
    alpha_x=ALPHA_X, beta_x=BETA_X, D_t=D_T, cluster=ClusterParameters()
)
photon_cluster_refit = PhotonParameters(
    alpha_x=ALPHA_X, beta_x=BETA_X, D_t=D_T_REFIT, cluster=ClusterParameters()
)
cell = CellTarget(nucleus_radius_um=R_N, effective_size_factor=1.0)

engine = AnalyticSpectrumEngine(
    stopping_power=SimpleBetheStoppingPower(),
    site=SphericalSite(diameter_um=0.6),  # 0.3 um radius, matching Hartzell 2025
    restricted=True,
)

# (label, colour, linestyle, model)
MODELS: list[tuple[str, str, str, object]] = [
    ("LEM I", "C0", "-", LEM1(photon=photon, cell=cell)),
    (r"LEM II ($D_t$=30)", "C1", "-", LEM2(photon=photon_cluster, cell=cell)),
    (r"LEM II ($D_t$=7.5)", "C1", "--", LEM2(photon=photon_cluster_refit, cell=cell)),
    (r"LEM III ($D_t$=30)", "C2", "-", LEM3(photon=photon_cluster, cell=cell)),
    (r"LEM III ($D_t$=7.5)", "C2", "--", LEM3(photon=photon_cluster_refit, cell=cell)),
    # legacy_trip98_cluster_index_cap: the comparison spans energies below 2.206
    # MeV/u, where C_I saturates and the uncapped default -- correct but unfitted --
    # runs several times above the TRiP98 tables it is being shown against.
    ("LEM IV", "C3", "-", LEM4(photon=photon, cell=cell, legacy_trip98_cluster_index_cap=0.95)),
    ("MCF MKM", "C4", "-", MCFMKMModel(alpha_ref=ALPHA_X, beta_ref=BETA_X, engine=engine)),
    ("mMKM", "0.25", "-", ModifiedMKM(alpha_ref=ALPHA_X, beta_ref=BETA_X, engine=engine)),
    # RMF is the odd one out here, deliberately: it is the only entry whose
    # biology does not descend from the amorphous-track lineage. It consumes the
    # same photon pair, but turns it into DSB-processing parameters rather than
    # a local-dose or specific-energy calibration. Its site size is the 5 um
    # nucleus DIAMETER both its papers specify -- not the R_N radius above.
    (
        "RMF",
        "C5",
        "-",
        RMFModel(
            photon=PhotonParameters(alpha_x=ALPHA_X, beta_x=BETA_X),
            reference_radiation=X_RAY_200KVP_REFERENCE,
        ),
    ),
]


# %%
def curves(model: object, Z: int, A: int) -> tuple[np.ndarray, np.ndarray]:
    """(alpha, beta) arrays over the energy grid for one ion.

    The grid reaches 1 MeV/u, where carbon is well past the 150 keV/um ceiling
    RMF states for itself; that warning is silenced here because the point of
    the panel is to *show* the divergence, and the markdown below says so.
    """
    with warnings.catch_warnings():
        warnings.simplefilter("ignore", UserWarning)
        lqs = [model.alpha_beta(Ion(Z, A, float(e))) for e in ENERGIES]  # type: ignore[attr-defined]
    return (
        np.array([lq.alpha for lq in lqs]),
        np.array([lq.beta for lq in lqs]),
    )


data = {(label, Z): curves(model, Z, A) for label, _c, _ls, model in MODELS for Z, A, _ in IONS}

# %% [markdown]
# ## Figure
#
# Columns are the two ions; the top row is alpha(E), the bottom row beta(E).
# beta uses a linear axis so the high-LET **overkill collapse** (LEM II/IV and
# MCF-MKM beta falling toward zero as the carbon track saturates, while mMKM
# holds beta = beta_ref) stays visible.
#
# **RMF is the only curve without a peak.** Every amorphous-track model here turns
# over in carbon alpha around 5-15 MeV/u through a saturation mechanism
# (local-dose saturation in the LEM family, the saturation-corrected specific
# energy in the MKM family). RMF has none, so its alpha and beta rise
# monotonically as the ion slows: at 1 MeV/u carbon its beta reaches ~0.26 Gy^-2
# where every other model is below 0.03. That is the model, not the
# implementation -- Frese et al. state the form holds only "up to at least 100 to
# 150 keV/um", and `curves()` silences the warning so the divergence is visible.
# It is what an *independent lineage* buys and costs: low-LET behaviour from DSB
# yields rather than a dose calibration, and no high-LET overkill.

# %%
fig, axes = plt.subplots(2, 2, figsize=(12.0, 9.0), sharex=True)

for col, (Z, _A, ion_label) in enumerate(IONS):
    ax_a, ax_b = axes[0, col], axes[1, col]
    for label, colour, ls, _model in MODELS:
        alpha, beta = data[(label, Z)]
        ax_a.plot(ENERGIES, alpha, color=colour, lw=1.7, ls=ls, label=label)
        ax_b.plot(ENERGIES, beta, color=colour, lw=1.7, ls=ls, label=label)
    ax_a.set_title(ion_label)
    ax_a.set_ylabel(r"$\alpha$ (Gy$^{-1}$)")
    ax_a.set_xscale("log")
    ax_a.set_yscale("log")
    ax_a.grid(True, which="both", ls=":", lw=0.45)
    ax_b.set_ylabel(r"$\beta$ (Gy$^{-2}$)")
    ax_b.set_xscale("log")
    ax_b.set_xlabel("E (MeV/u)")
    ax_b.grid(True, which="both", ls=":", lw=0.45)
    ax_b.set_ylim(bottom=0.0)

axes[0, 0].legend(frameon=False, fontsize=7.5, ncol=2)
fig.suptitle(
    "Inter-model comparison on a shared V79 reference "
    f"($\\alpha_x$={ALPHA_X}, $\\beta_x$={BETA_X}, $D_t$={D_T}, $R_n$={R_N})",
    y=0.99,
)
fig.tight_layout()
save_figure(fig, "model_comparison_alpha_beta.png", directory=FIGURES, dpi=150)
