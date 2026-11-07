"""Frese et al. 2011 -- the "linear scaling model": Wilkens' form, re-anchored."""

from __future__ import annotations

from dataclasses import dataclass
from typing import ClassVar

from pyradbio.core.photon import PhotonParameters
from pyradbio.models.phenomenological.base import RBEMinMax, RBEMinMaxModel

# Frese et al. 2011. lambda is the four-patient average of the per-patient
# lambda_1.1 in table 4 (0.0074, 0.0070, 0.0086, 0.0091 -> 0.008025, printed as
# 0.008); the entrance LET is the ~0.5 keV/um of table 2's beams. Table 4 (and
# eq. (8) reproducing each row) is asserted in tests/models/test_phenomenological_rbe.py
# (FRESE_TABLE4); the four rows span 0.0070-0.0091, i.e. +-13% on the one
# parameter that sets the model's whole LET sensitivity.
FRESE_LAMBDA_UM_KEV_GY = 0.008  # um / (keV Gy)
FRESE_ENTRANCE_LET_KEV_UM = 0.5  # keV/um


@dataclass(frozen=True, slots=True)
class LinearScaling(RBEMinMaxModel):
    """Frese et al. 2011, Int. J. Radiat. Oncol. Biol. Phys. 79:80,
    doi:10.1016/j.ijrobp.2009.10.022.

    **Not a new functional form.** Its eq. (3) *is* Wilkens & Oelfke's eq. (4)
    (which it cites as ref. 8): ``alpha_p = alpha_0 + lambda*LET`` with
    ``beta_p = beta_x``, so ``RBE_min = 1`` here too. What Frese contributes is a
    **calibration** that removes ``alpha_0`` as a free parameter and ties it to
    the tissue::

        alpha_0 = alpha_x - 0.5 keV/um * lambda        (eq. 6)

    The argument is clinical, not biological: the RBE is taken to be 1.0 in the
    entrance channel, where the dose-averaged LET is about 0.5 keV/um. So unlike
    :class:`.WilkensOelfke`, **this model does reduce to the photon reference** --
    not at zero LET, but at the entrance LET, exactly by construction.

    ``lambda`` is then fixed by requiring the mean RBE over the target to come out
    at the clinical 1.1 (eq. 8, exposed as :meth:`lambda_for_target_rbe`); the
    default 0.008 um/(keV Gy) is the four-patient average from the paper's table 4.
    One ``lambda`` is used for all tissues, the paper's own words being that
    "evidence for different lambda values in different tissues is scarce".

    **Two consequences worth knowing before using it.**

    * It is **fitted for head-and-neck IMPT plans**, at 2 Gy per fraction, and
      ``lambda`` is dose-per-fraction dependent -- the paper says it "has to be
      adjusted for other fractionation doses to keep the mean RBE at 1.1". The
      default is not a general-purpose constant.
    * Because ``alpha_0`` is tied to ``alpha_x`` and ``lambda`` is not, the LET
      sensitivity of the RBE scales as ``lambda/alpha_x``: radioresistant tissues
      (small ``alpha_x``) respond *more* strongly to LET. That is the paper's main
      clinical finding, and it falls straight out of the parameterisation.

    Setting ``lambda_um_keV_Gy = 0`` reproduces the paper's treatment of
    non-specific normal tissue: a flat RBE of 1 from the model.
    """

    lambda_um_keV_Gy: float = FRESE_LAMBDA_UM_KEV_GY
    entrance_let_keV_um: float = FRESE_ENTRANCE_LET_KEV_UM

    model_id: ClassVar[str] = "LSM"
    reference_doi: ClassVar[str] = "10.1016/j.ijrobp.2009.10.022"
    radiation_modes: ClassVar[tuple[str, ...]] = ("protons",)
    # Inherited from the Wilkens form it reuses (Wilkens & Oelfke 2004 sec. 2.1).
    valid_let_keV_um: ClassVar[tuple[float, float] | None] = (0.3, 30.0)

    @property
    def alpha0_Gy(self) -> float:
        """``alpha_0 = alpha_x - L_entrance * lambda`` [Gy^-1] (eq. 6)."""
        return self.photon.alpha_x - self.entrance_let_keV_um * self.lambda_um_keV_Gy

    def alpha_ion_Gy(self, let_keV_um: float) -> float:
        """``alpha_p(L) = alpha_0 + lambda L`` [Gy^-1] (eq. 3)."""
        return self.alpha0_Gy + self.lambda_um_keV_Gy * let_keV_um

    @staticmethod
    def lambda_for_target_rbe(
        *,
        target_rbe: float,
        photon: PhotonParameters,
        mean_let_keV_um: float,
        mean_dose_Gy: float,
        entrance_let_keV_um: float = FRESE_ENTRANCE_LET_KEV_UM,
    ) -> float:
        """Frese et al. 2011 eq. (8): the ``lambda`` giving ``target_rbe`` on average.

        ``lambda = [alpha_x (RBE - 1) + beta_x D (RBE^2 - 1)] / (L - L_entrance)``

        with ``L`` and ``D`` the mean dose-averaged LET and mean fraction dose over
        the target of a physically optimised plan. This is how the paper's default
        0.008 was obtained (RBE = 1.1, four patients), so it is a *planning*
        calibration: the result depends on the plan and on the dose per fraction,
        and is not a property of the tissue alone.
        """
        denominator = mean_let_keV_um - entrance_let_keV_um
        if denominator <= 0.0:
            raise ValueError(
                "mean_let_keV_um must exceed the entrance LET "
                f"({entrance_let_keV_um} keV/um); got {mean_let_keV_um}"
            )
        numerator = photon.alpha_x * (target_rbe - 1.0) + photon.beta_x * mean_dose_Gy * (
            target_rbe * target_rbe - 1.0
        )
        return numerator / denominator

    def _limits(self, let_keV_um: float) -> RBEMinMax:
        return RBEMinMax(
            rbe_min=1.0,
            rbe_max=self.alpha_ion_Gy(let_keV_um) / self.photon.alpha_x,
        )
