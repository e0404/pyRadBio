"""Wilkens & Oelfke 2004 -- linear ``alpha(LET)``, beta pinned to the photon beta."""

from __future__ import annotations

from dataclasses import dataclass
from typing import ClassVar

from pyradbio.core.photon import PhotonParameters
from pyradbio.models.phenomenological.base import RBEMinMax, RBEMinMaxModel

# Wilkens & Oelfke 2004 fig. 1(a): a linear fit of eq. (4) to published V79
# in-vitro proton data, with the uncertainties the paper estimates in sec. 3.
WILKENS_ALPHA0_GY = 0.1  # +- 0.02 Gy^-1
WILKENS_LAMBDA_UM_KEV_GY = 0.02  # +- 0.002 um keV^-1 Gy^-1

# Sec. 2.2.3: V79 relative to 60Co, from Tilly et al. 1999. The paper is explicit
# that the survival in the proton beam depends only on (alpha_0, lambda, beta_p,
# D, L) -- alpha_x and beta_x enter the *RBE* alone.
WILKENS_V79_ALPHA_X = 0.112  # Gy^-1
WILKENS_V79_BETA_X = 0.0298  # Gy^-2


@dataclass(frozen=True, slots=True)
class WilkensOelfke(RBEMinMaxModel):
    """Wilkens & Oelfke 2004, Phys. Med. Biol. 49:2811, doi:10.1088/0031-9155/49/13/004.

    The oldest member of the family, and the only one whose linear term is
    written directly in ``alpha`` rather than in ``RBE_max``::

        alpha_p(L) = alpha_0 + lambda * L        (eq. 4)
        beta_p     := beta_x                     (eq. 5)

    which on this base class is ``RBE_max = (alpha_0 + lambda L)/alpha_x`` and
    ``RBE_min = 1``. Eq. (5) is a *definition*, argued from the observation that
    the published ``beta_p`` data show no clear LET trend and that "beta_p has not
    much influence at high LET, especially at therapeutic levels for the dose per
    fraction".

    **``alpha_0`` is not ``alpha_x``.** It is a fitted intercept of the ion
    response, so this model does **not** reduce to the photon reference at zero
    LET: with the V79 defaults ``RBE_max(0) = 0.1/0.112 = 0.89``. Carabe behaves
    the same way. That is why ``alpha0_Gy`` is a model parameter here rather than
    something derived from ``photon`` -- see :class:`.LinearScaling`, which is
    exactly this model with ``alpha_0`` *tied* to ``alpha_x`` instead.

    **Validity: dose-averaged LET 0.3-30 keV/um** (sec. 2.1), covering therapeutic
    proton beams of ~70-250 MeV. The upper bound matters more than most in this
    family: the paper's own fig. 1(a) shows ``alpha_p`` *decreasing* above about
    30 keV/um, so beyond it the linear form has the wrong sign of slope, not
    merely an unfitted one.

    The paper's sec. 2.3 endorses the reduction this package performs for mixed
    fields -- below 30 keV/um, computing ``alpha_p`` from the *dose-averaged* LET
    is equivalent to the Zaider-Rossi dose-averaging of ``alpha`` and ``sqrt(beta)``.
    """

    alpha0_Gy: float = WILKENS_ALPHA0_GY
    lambda_um_keV_Gy: float = WILKENS_LAMBDA_UM_KEV_GY

    model_id: ClassVar[str] = "WIL"
    reference_doi: ClassVar[str] = "10.1088/0031-9155/49/13/004"
    radiation_modes: ClassVar[tuple[str, ...]] = ("protons",)
    valid_let_keV_um: ClassVar[tuple[float, float] | None] = (0.3, 30.0)

    @classmethod
    def v79_wilkens2004(cls) -> WilkensOelfke:
        """The paper's own worked case: V79 cells against a 60Co reference.

        ``alpha_0``/``lambda`` from the fig. 1(a) fit, ``alpha_x``/``beta_x`` from
        Tilly et al. 1999 as quoted in sec. 2.2.3.
        """
        return cls(photon=PhotonParameters(alpha_x=WILKENS_V79_ALPHA_X, beta_x=WILKENS_V79_BETA_X))

    def alpha_ion_Gy(self, let_keV_um: float) -> float:
        """``alpha_p(L) = alpha_0 + lambda L`` [Gy^-1] -- the paper's eq. (4) itself."""
        return self.alpha0_Gy + self.lambda_um_keV_Gy * let_keV_um

    def _limits(self, let_keV_um: float) -> RBEMinMax:
        return RBEMinMax(
            rbe_min=1.0,
            rbe_max=self.alpha_ion_Gy(let_keV_um) / self.photon.alpha_x,
        )
