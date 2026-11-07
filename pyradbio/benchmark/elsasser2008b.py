"""Elsasser, Cunrath, Kraemer & Scholz 2008 (NJP) track-structure benchmark.

Source paper:

Elsasser, T., Cunrath, R., Kraemer, M. & Scholz, M. 2008, "Impact of track
structure calculations on biological treatment planning in ion radiotherapy",
New J. Phys. **10** 075005, doi:10.1088/1367-2630/10/7/075005.

**The module name follows Friedrich et al. 2012**, which cites this paper as
"Elsasser et al. 2008b" for the LEM's radial dose and the *other* 2008 paper
(Elsasser, Kraemer & Scholz, IJROBP 71 866-872 = LEM III, this package's
:mod:`~pyradbio.benchmark.elsasser2008`) as "2008a" for the model version. They
are easy to confuse and the distinction matters: 2008b does **not** define a LEM
version, it *surveys* five amorphous track-structure models.

Why this reference is unusually valuable
----------------------------------------
Every other LEM reconstruction in this package rests on **digitized** figures.
This one does not: the paper's Table 1 prints ``RBE_alpha`` to three significant
figures with an explicit uncertainty, for five track-structure models at two LET
values. It is therefore the cleanest model-line cross-check available anywhere in
the corpus -- no tracing, no axis calibration, no envelope splitting.

The track-structure menu (sec. 3.1.5)
-------------------------------------
The paper labels the variants and maps them onto LEM versions:

========  =====================================  =========================
label     track core                             used by
========  =====================================  =========================
``CR10``  constant ``r_min = 10 nm``             LEM I (diffusion implicit)
``CR03``  constant ``r_min = 0.3 nm``            LEM II (cluster version)
``EDR11`` ``r_min = beta*r_c``, ``r_c = 11.6``   comparison with KC / MKM
``EDR40`` ``r_min = beta*r_c``, ``r_c = 40 nm``  LEM III
``KC``    Kiefer-Chatterjee                      comparison only
``MC``    Monte Carlo delta-electron transport   comparison only
========  =====================================  =========================

All analytical variants share ``r_max = gamma E^delta`` with ``gamma = 0.062``,
``delta = 1.7`` (sec. 3.1.1), and all except ``CR10`` are convolved with the
``sigma = 4 nm`` radical-diffusion Gaussian (sec. 3.3); ``CR10`` folds diffusion
into its large core instead. ``CR10`` is the *original* LEM and carries **no**
cluster enhancement ``eta`` (p. 13: "the original version of the LEM
underestimates the experimental data only because in that model version cluster
effects, represented by the factor eta in equation (7), were not taken into
account").

``r_c`` provenance, also from sec. 3.1.2: Mozumder's adiabatic-invariance
argument gives ``r_c = 9.3 nm`` for liquid water (``n = 1.33``,
``omega_0 = 17 eV/hbar``), Chatterjee et al. ``11.6 nm``; **LEM III's 40 nm is
phenomenological**, a chi^2 fit to rat spinal-cord tolerance data.

The core / ``D_t`` degeneracy
-----------------------------
Section 4 is explicit that the track core and the transition dose trade off:
"the LEM gives good results for all track structure models by choosing the
threshold parameter Dt appropriately". Hence :data:`TRACK_MODELS` carries a
*per-model* ``D_t`` -- 6 Gy for ``CR03`` against 23 for ``EDR40`` and 30 for
``CR10``, all fitting the same HSG data. **Never vary a core radius without
re-fitting ``D_t``**, or the comparison measures the degeneracy rather than the
core.

Initial-slope method
--------------------
Sec. 4: "we apply the approximation described in [10] for a rapid calculation of
cell inactivation", where [10] is Scholz, Kellerer, Kraft-Weyrather & Kraft 1997
(Radiat. Environ. Biophys. 36 59-66). That is the inactivation *cross-section*
formulation, ``sigma = int (1 - exp(-nu(b))) 2 pi b db`` -- pyRadBio's
``initial_slope_model="cross_section"`` -- **not** the ``b = 0`` central-traversal
shortcut of Elsasser & Scholz 2007 eq. (8). The distinction is worth 18-30% at
13 keV/um and is pinned by ``test_elsasser2008b.py``.
"""

from __future__ import annotations

from dataclasses import dataclass

ELSASSER2008B_DOI = "10.1088/1367-2630/10/7/075005"

#: HSG photon inputs used throughout secs. 4-5 (Fig. 5/6 captions, from ref. [21]
#: = Elsasser & Scholz 2007, "shown to give the best fit (based on chi^2-analysis)").
HSG_ALPHA_X: float = 0.313  # Gy^-1
HSG_BETA_X: float = 0.0615  # Gy^-2
HSG_NUCLEUS_RADIUS_UM: float = 5.0

#: The two carbon qualities of Table 1: entrance channel and spread-out Bragg peak.
TABLE1_LET_KEV_UM: tuple[float, float] = (13.0, 77.0)

#: Shared delta-electron range parameterisation (sec. 3.1.1, after eq. 8).
R_MAX_GAMMA_UM: float = 0.062
R_MAX_DELTA: float = 1.7
#: Radical-diffusion width (sec. 3.3, refs. [30, 31]).
RADICAL_SIGMA_NM: float = 4.0


@dataclass(frozen=True, slots=True)
class TrackModel:
    """One row of Table 1: a track-structure variant with its fitted ``D_t``."""

    label: str
    lem_version: int | None  # 1/2/3, or None where the variant maps to no LEM version
    description: str
    d_t_Gy: float  # fitted per model (Fig. 6 caption) -- degenerate with the core
    rbe_alpha: tuple[float, float]  # at TABLE1_LET_KEV_UM
    rbe_alpha_uncertainty: tuple[float, float]  # from varying D_t by +-10%

    def rbe_at(self, let_keV_um: float) -> tuple[float, float]:
        """``(value, uncertainty)`` at 13 or 77 keV/um."""
        try:
            index = TABLE1_LET_KEV_UM.index(let_keV_um)
        except ValueError:  # pragma: no cover - guarded by the caller
            raise KeyError(f"Table 1 has no entry at {let_keV_um} keV/um") from None
        return self.rbe_alpha[index], self.rbe_alpha_uncertainty[index]


#: Table 1 (p. 14) plus the Fig. 6 caption's per-model ``D_t``. ``KC`` and ``MC``
#: are carried for completeness; pyRadBio implements no Kiefer-Chatterjee or
#: delta-electron-transport radial dose, and the paper itself calls their fitted
#: ``D_t`` values (0.01 and 1.6 Gy) physically unrealistic.
TRACK_MODELS: tuple[TrackModel, ...] = (
    TrackModel(
        label="CR03",
        lem_version=2,
        description="constant core r_min = 0.3 nm + sigma = 4 nm diffusion (LEM II)",
        d_t_Gy=6.0,
        rbe_alpha=(2.12, 4.22),
        rbe_alpha_uncertainty=(0.10, 0.23),
    ),
    TrackModel(
        label="CR10",
        lem_version=1,
        description="constant core r_min = 10 nm, diffusion folded in, no cluster eta (LEM I)",
        d_t_Gy=30.0,
        rbe_alpha=(2.60, 4.50),
        rbe_alpha_uncertainty=(0.11, 0.25),
    ),
    TrackModel(
        label="EDR40",
        lem_version=3,
        description="velocity core r_min = beta*40 nm + sigma = 4 nm diffusion (LEM III)",
        d_t_Gy=23.0,
        rbe_alpha=(1.72, 4.35),
        rbe_alpha_uncertainty=(0.03, 0.25),
    ),
    TrackModel(
        label="MC",
        lem_version=None,
        description="Monte-Carlo delta-electron transport; D_t = 1.6 Gy called unrealistic",
        d_t_Gy=1.6,
        rbe_alpha=(2.18, 4.64),
        rbe_alpha_uncertainty=(0.08, 0.15),
    ),
    TrackModel(
        label="KC",
        lem_version=None,
        description="Kiefer-Chatterjee, r_c = 11.6 nm; D_t = 0.01 Gy called unrealistic",
        d_t_Gy=0.01,
        rbe_alpha=(1.68, 5.24),
        rbe_alpha_uncertainty=(0.01, 0.01),
    ),
)


def get_track_model(label: str) -> TrackModel:
    """Look up one Table 1 row by its paper label (``"CR03"``, ``"EDR40"``, ...)."""
    for model in TRACK_MODELS:
        if model.label == label:
            return model
    raise KeyError(f"unknown track-structure model {label!r}")


#: Models pyRadBio can actually build (the analytical LEM cores).
IMPLEMENTED_LABELS: tuple[str, ...] = ("CR03", "CR10", "EDR40")
