"""Unkelbach et al. 2016 -- an LET-optimisation ansatz, not a fit to survival data."""

from __future__ import annotations

from dataclasses import dataclass
from typing import ClassVar

from pyradbio.models.phenomenological.base import RBEMinMax, RBEMinMaxModel

#: RBE slope in LET_d [(keV/um)^-1], deduced from RBE = 1.1 at the SOBP's mean
#: LET_d of 2.5 keV/um. Rorvik et al. 2018 appendix B.
UNKELBACH_SLOPE = 0.04

#: The paper's own alternative values, obtained by linearising McNamara et al.
#: 2015 at 1.67 Gy per fraction: ``c ~ 0.02`` for ``(alpha/beta)_x = 10 Gy`` and
#: ``c ~ 0.05`` for ``(alpha/beta)_x = 2 Gy``. Quoted as a sensitivity range, not
#: as fitted constants -- pass them via ``slope_um_keV`` if wanted.
UNKELBACH_SLOPE_MCNAMARA_HIGH_ALPHA_BETA = 0.02
UNKELBACH_SLOPE_MCNAMARA_LOW_ALPHA_BETA = 0.05


@dataclass(frozen=True, slots=True)
class Unkelbach(RBEMinMaxModel):
    """Unkelbach et al. 2016, Int. J. Radiat. Oncol. Biol. Phys. 96:1097,
    doi:10.1016/j.ijrobp.2016.08.038.

    ::

        RBE_max(LET_d) = RBE_min(LET_d) = 1 + 0.04 LET_d

    **This is not a biological model.** Unkelbach et al. developed an *LET
    optimisation method*, not an RBE fit; the formula above is what their
    premises imply -- that RBE is linear in ``LET_d`` and averages the clinical
    1.1 over a typical SOBP, whose mean ``LET_d`` is 2.5 keV/um. Rorvik et al.
    2018 include it, with Frese et al. 2011, as one of two "plan-based" models
    for comparison, and this package ships it on the same footing as
    :class:`~pyradbio.models.phenomenological.frese.LinearScaling` for that
    reason.

    **Setting ``RBE_max = RBE_min`` is the whole content of the reformulation**,
    and the primary explains why it is exact rather than an approximation: the
    paper's eq. (1) is ``S = exp(-alpha d)``, a **pure exponential with no
    quadratic term at all**, so there is no shoulder for dose to erode. When the
    two limits coincide, the
    family's RBE-versus-dose curve is flat, so this model's RBE is **independent
    of dose**. Every other member of the family predicts an RBE that changes with
    fraction size. It is also independent of tissue: no ``alpha_x``,
    ``(alpha/beta)_x`` or fractionation enters.

    No fitted domain is declared, because there is no database behind it to
    bound. That is not the same as being valid everywhere.

    .. note::
       Rorvik et al. 2018 describes a "McMahon configuration ... the coefficient
       of 0.04 is increased to 0.055". **No such value, and no mention of McMahon,
       appears in Unkelbach et al.** What the paper does give as alternatives are
       ``c ~ 0.02`` for ``(alpha/beta)_x = 10 Gy`` and ``c ~ 0.05`` for
       ``(alpha/beta)_x = 2 Gy``, both obtained by linearising **McNamara** et al.
       2015 at 1.67 Gy per fraction -- offered as a sensitivity range, not as
       fitted constants. The 0.055 variant is therefore not shipped; use
       ``slope_um_keV`` to set any of these explicitly.
    """

    slope_um_keV: float = UNKELBACH_SLOPE

    model_id: ClassVar[str] = "UNK"
    reference_doi: ClassVar[str] = "10.1016/j.ijrobp.2016.08.038"
    radiation_modes: ClassVar[tuple[str, ...]] = ("protons",)

    def _limits(self, let_keV_um: float) -> RBEMinMax:
        rbe = 1.0 + self.slope_um_keV * let_keV_um
        return RBEMinMax(rbe_min=rbe, rbe_max=rbe)
