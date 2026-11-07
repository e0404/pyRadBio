"""Chen & Ahmad 2012 -- a saturating rise in ``alpha`` on a V79-fitted intercept."""

from __future__ import annotations

import math
from dataclasses import dataclass
from typing import ClassVar

from pyradbio.models.phenomenological.base import RBEMinMax, RBEMinMaxModel

#: Ion-response intercept ``alpha_0`` [Gy^-1] -- the paper's eq. (8) fit to V79
#: data. Like Wilkens' ``alpha_0``, this is a property of *that cell line*, not of
#: the caller's tissue; see the class docstring.
CHEN_ALPHA0_GY = 0.1

#: Curvature of the saturating numerator, ``lambda_1`` [(keV/um)^-2].
CHEN_LAMBDA1 = 0.0013

#: Slope of the denominator, ``lambda_2`` [(keV/um)^-1 Gy].
CHEN_LAMBDA2 = 0.045


@dataclass(frozen=True, slots=True)
class Chen(RBEMinMaxModel):
    """Chen & Ahmad 2012, Radiat. Prot. Dosimetry 149:116-123, doi:10.1093/rpd/ncr218.

    The paper's eq. (8) is a statement about the ion ``alpha``, not about RBE::

        alpha_p(LET_d) = alpha_0 + [1 - exp(-lambda_1 LET_d^2)] / (lambda_2 LET_d)
        beta_p         = beta_x                                        (eq. 9)

    with ``alpha_0 = 0.1 Gy^-1``, ``lambda_1 = 0.0013 (keV/um)^-2`` and
    ``lambda_2 = 0.045 (keV/um)^-1 Gy``, fitted to V79 survival data (its fig. 2).
    On this package's skeleton that gives ``RBE_max = alpha_p/alpha_x`` and
    ``RBE_min = 1``.

    **This is a Wilkens-type model: it keys on ``alpha_x``, not on
    ``(alpha/beta)_x``.** ``alpha_0`` is an ion-response intercept belonging to the
    V79 fit, so pairing it with an arbitrary photon reference is the same caveat
    that applies to
    :class:`~pyradbio.models.phenomenological.wilkens.WilkensOelfke`.

    .. warning::
       **Rorvik et al. 2018's appendix-B reformulation of this model is wrong by a
       factor of 10, and is not what this class implements.** It renders eq. (8) as
       ``RBE_max = 1 + [1 - exp(-0.0013 L^2)] / (0.045 L)``, i.e. it adds the
       LET-dependent term to 1 without dividing it by ``alpha_x``. Under the
       appendix's own stated assumption ``alpha_x = alpha_0 = 0.1 Gy^-1`` the
       correct denominator constant would be ``0.0045``, not ``0.045``.

       The consequence is not subtle: at ``LET_d = 30 keV/um`` on the paper's own
       V79 worked case (``alpha_x = 0.1``, ``beta_x = 0.03``), eq. (8) gives
       ``RBE_10 = 2.23`` while the reformulation gives **1.10** -- the
       entrance-plateau clinical value, at a LET where Chen's fig. 6 plots
       measured V79 RBE well above 2. Pinned by
       ``test_chen_primary_disagrees_with_the_rorvik_reformulation``.

       The paper's fig. 6 caption states ``beta_x = 0.3 Gy^-2`` for its V79 worked
       example, which with ``alpha_x = 0.1`` would give ``(alpha/beta)_x = 0.33 Gy``.
       Rorvik reads it as ``0.03`` (``(alpha/beta)_x = 3.33 Gy``), which is the
       physically sensible value for V79. Nothing here depends on it -- ``beta_x``
       comes from the caller -- but do not propagate the printed 0.3.

    The shape is a saturating rise: the bracket approaches 1 while the denominator
    keeps growing, so ``alpha_p`` peaks and falls back towards ``alpha_0`` -- the
    paper's stated motivation, since Wilkens' linear form "was unable to predict
    the decrease of RBE with the increase of LET".
    """

    model_id: ClassVar[str] = "CHE"
    reference_doi: ClassVar[str] = "10.1093/rpd/ncr218"
    radiation_modes: ClassVar[tuple[str, ...]] = ("protons",)
    # Rorvik table A.1 for the V79 database this was fitted on.
    valid_alpha_beta_ratio_Gy: ClassVar[tuple[float, float] | None] = (2.2, 2.8)
    valid_let_keV_um: ClassVar[tuple[float, float] | None] = (7.7, 37.8)

    def alpha_ion_Gy(self, let_keV_um: float) -> float:
        """``alpha_p(LET_d)`` [Gy^-1] -- the paper's eq. (8) itself."""
        # The quotient is 0/0 at LET = 0. Expanding the numerator,
        # (1 - exp(-l1 L^2)) / (l2 L) -> l1 L / l2 -> 0, so alpha_p -> alpha_0.
        if let_keV_um == 0.0:
            return CHEN_ALPHA0_GY
        numerator = 1.0 - math.exp(-CHEN_LAMBDA1 * let_keV_um * let_keV_um)
        return CHEN_ALPHA0_GY + numerator / (CHEN_LAMBDA2 * let_keV_um)

    def _limits(self, let_keV_um: float) -> RBEMinMax:
        return RBEMinMax(
            rbe_min=1.0,
            rbe_max=self.alpha_ion_Gy(let_keV_um) / self.photon.alpha_x,
        )
