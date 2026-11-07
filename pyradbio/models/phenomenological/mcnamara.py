"""McNamara, Schuemann & Paganetti 2015 -- fitted to all published in-vitro data."""

from __future__ import annotations

import math
from dataclasses import dataclass
from typing import ClassVar

from pyradbio.models.phenomenological.base import RBEMinMax, RBEMinMaxModel

# McNamara et al. 2015 sec. 3.1, with the standard errors the paper prints.
# NOTE p0: the paper gives 0.99064 (SE 0.014125), confirmed against the paper.
# Some implementations carry 0.999064 instead; if a comparison against another
# code shows a constant offset of 0.008424 in RBE_max, that is the cause.
MCNAMARA_P0 = 0.99064  # SE 0.014125
MCNAMARA_P1 = 0.35605  # SE 0.015038 [Gy um / keV]
MCNAMARA_P2 = 1.1012  # SE 0.0059972
MCNAMARA_P3 = -0.0038703  # SE 0.00091303 [um / (keV Gy^0.5)]


@dataclass(frozen=True, slots=True)
class McNamara(RBEMinMaxModel):
    """McNamara et al. 2015, Phys. Med. Biol. 60:8399, doi:10.1088/0031-9155/60/21/8399.

    A non-linear regression through 287 in-vitro survival data points collected
    by Paganetti 2014 -- the largest fit of the family::

        RBE_max = p0 + p1 * LET_d / (alpha/beta)_x
        RBE_min = p2 + p3 * sqrt((alpha/beta)_x) * LET_d

    ``RBE_max`` follows Carabe et al. 2012; the ``sqrt((alpha/beta)_x)`` in
    ``RBE_min`` is the paper's own departure from it, after Jones 2015.

    **Validity.** The fit used only ``LET_d < 20 keV/um`` and
    ``(alpha/beta)_x < 30 Gy``, at proton doses of 1-10 Gy, and the paper is
    explicit that the LET cut "corresponds to the clinically relevant LET_d
    range, where the RBE is approximately linear with respect to LET_d". Outside
    that box :meth:`rbe_min_max` warns rather than clamping.

    Note ``RBE_min`` falls with LET (p3 < 0) and will go negative for a large
    enough ``sqrt((alpha/beta)_x) * LET_d`` -- at (alpha/beta)_x = 10 Gy that is
    LET_d ~ 90 keV/um, far outside the fitted range but reachable by a caller who
    ignores the warning. ``RBEMinMax`` rejects a negative limit outright rather
    than letting ``beta = RBE_min**2 beta_x`` quietly come back positive.
    """

    model_id: ClassVar[str] = "MCN"
    reference_doi: ClassVar[str] = "10.1088/0031-9155/60/21/8399"
    radiation_modes: ClassVar[tuple[str, ...]] = ("protons",)
    valid_let_keV_um: ClassVar[tuple[float, float] | None] = (0.0, 20.0)
    valid_alpha_beta_ratio_Gy: ClassVar[tuple[float, float] | None] = (0.0, 30.0)

    def _limits(self, let_keV_um: float) -> RBEMinMax:
        ratio = self.alpha_beta_ratio_Gy
        return RBEMinMax(
            rbe_min=MCNAMARA_P2 + MCNAMARA_P3 * math.sqrt(ratio) * let_keV_um,
            rbe_max=MCNAMARA_P0 + MCNAMARA_P1 * let_keV_um / ratio,
        )
