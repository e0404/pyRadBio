"""Peeler 2016 (PhD dissertation) -- the only cubic-in-LET member of the family."""

from __future__ import annotations

from dataclasses import dataclass
from typing import ClassVar

from pyradbio.models.phenomenological.base import RBEMinMax, RBEMinMaxModel

PEELER_RBE_MAX_INTERCEPT = 0.75
"""``RBE_max`` intercept [1] and cubic slope [Gy (keV/um)^-3]."""
PEELER_RBE_MAX_SLOPE = 0.00143

PEELER_RBE_MIN_INTERCEPT = 1.24
"""``RBE_min`` intercept [1] and cubic slope [Gy^-1 (keV/um)^-3]."""
PEELER_RBE_MIN_SLOPE = 0.00074


@dataclass(frozen=True, slots=True)
class Peeler(RBEMinMaxModel):
    """Peeler 2016, PhD dissertation, eqs. (30) and (31) -- *not* the RO paper.

    Rorvik et al. 2018's table 1 attributes the PLR model to "(Peeler 2016)",
    single-author, which its reference list resolves to the dissertation
    *Assessing the Potential Clinical Impact of Variable Biological Effectiveness
    in Proton Radiotherapy*. Its reference list separately carries Peeler,
    Mirkovic, Titt et al. 2016, *Clinical evidence of variable proton biological
    effectiveness in pediatric patients treated for ependymoma*, Radiother. Oncol.
    121:395-401 (doi:10.1016/j.radonc.2016.11.001) -- a clinical-outcomes study of
    imaging changes after treatment, which fits no RBE_max/RBE_min to the 48
    in-vitro data points table 1 credits this model with. The dissertation is the
    source; it has no DOI, hence [`reference_citation`][].

    As reformulated by Rorvik et al. 2018 appendix B:

    ```text
    RBE_max(LET_d, r) = 0.75 + 0.00143 LET_d^3 / r
    RBE_min(LET_d, r) = 1.24 + 0.00074 r LET_d^3          with r = (alpha/beta)_x
    ```

    Fitted to MD Anderson clonogenic-survival experiments on **two human lung
    cancer lines, H460 and H1437**, at **12 dose-averaged LET values spanning
    0.94-19.4 keV/um** (its table 3). Their photon parameters (table 4) are
    ``alpha_x = 0.36``, ``beta_x = 0.071`` -> ``(alpha/beta)_x = 5.1 Gy`` for H460
    and ``alpha_x = 0.08``, ``beta_x = 0.031`` -> ``2.6 Gy`` for H1437, which is
    the tissue range the fit actually covers.

    **Cubic in LET**, which the paper adopts deliberately: the other models it
    compares against "agree well ... up to an LET value of approximately
    10 keV/um", and the cube is what lets it follow the measured non-linear rise
    beyond that. Extrapolating a cubic past ~19 keV/um still diverges quickly, so
    the domain is declared and the base class warns -- never clamps.

    Note also that ``RBE_min`` scales **with** ``r`` while ``RBE_max`` scales with
    ``1/r`` -- the paper says so explicitly for each -- so the two limits move in
    opposite directions with tissue. That is unusual in this family, and the
    reason Rorvik's table A.2 records PLR as failing all three of Paganetti's
    qualitative RBE expectations.

    Like [`Carabe`][], it does **not**
    reduce to RBE = 1 at zero LET (``RBE_max(0) = 0.75``, ``RBE_min(0) = 1.24``),
    so "RBE -> 1 as LET -> 0" is not an acceptance test for it.
    """

    model_id: ClassVar[str] = "PLR"
    reference_doi: ClassVar[str] = ""
    """A dissertation, so no DOI exists to cite; see [`reference_citation`][]."""
    reference_citation: ClassVar[str] = (
        "Peeler C R 2016, Assessing the Potential Clinical Impact of Variable "
        "Biological Effectiveness in Proton Radiotherapy (PhD dissertation); "
        "as reformulated in Rorvik et al. 2018, doi:10.1088/1361-6560/aad9db"
    )
    radiation_modes: ClassVar[tuple[str, ...]] = ("protons",)
    # From the dissertation itself, NOT from Rorvik's table A.1, whose two range
    # columns this package first read the wrong way round: the LET_d span is
    # 0.9-19.0 (12 measured values, 0.94-19.4) and the (alpha/beta)_x span is
    # 2.6-5.1 (the two cell lines of its table 4).
    valid_alpha_beta_ratio_Gy: ClassVar[tuple[float, float] | None] = (2.6, 5.1)
    valid_let_keV_um: ClassVar[tuple[float, float] | None] = (0.9, 19.0)

    def _limits(self, let_keV_um: float) -> RBEMinMax:
        ratio = self.alpha_beta_ratio_Gy
        cube = let_keV_um**3
        return RBEMinMax(
            rbe_min=PEELER_RBE_MIN_INTERCEPT + PEELER_RBE_MIN_SLOPE * ratio * cube,
            rbe_max=PEELER_RBE_MAX_INTERCEPT + PEELER_RBE_MAX_SLOPE * cube / ratio,
        )
