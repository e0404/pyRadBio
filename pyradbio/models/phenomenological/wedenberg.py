"""Wedenberg, Lind & Hardemark 2013 -- linear alpha, LET-independent beta."""

from __future__ import annotations

from dataclasses import dataclass
from typing import ClassVar

from pyradbio.models.phenomenological.base import RBEMinMax, RBEMinMaxModel

WEDENBERG_Q = 0.434
"""Slope of ``alpha/alpha_x`` against ``LET_d/(alpha/beta)_x`` [Gy um / keV].
Wedenberg et al. 2013 Results: the least-squares fit of eq. (2) to the
experimental ``alpha/alpha_phot`` ratios gives **q = 0.434, 95% CI
(0.366, 0.513)** -- verified against the primary paper.
"""

WEDENBERG_Q_RESTRICTED_20_KEV = 0.378
"""Rorvik et al. 2018 (doi:10.1088/1361-6560/aad9db) appendix reports a refit of
the same model on a database restricted to <= 20 keV/um, giving a *reduced*
slope of 0.378. Not the default -- the 0.434 above is Wedenberg's own value --
but useful if comparing against work that used the restricted fit.
"""


@dataclass(frozen=True, slots=True)
class Wedenberg(RBEMinMaxModel):
    """Wedenberg et al. 2013, Acta Oncol. 52:580, doi:10.3109/0284186X.2012.705892.

    The model assumes ``alpha`` rises linearly with ``LET_d`` at a rate set by the
    tissue, and that ``beta`` does not depend on LET at all:

    ```text
    RBE_max = 1 + q * LET_d / (alpha/beta)_x        with q = 0.434
    RBE_min = 1
    ```

    The paper's title states the claim the fit rests on: "the tissue specific
    parameter alpha/beta of photons is a predictor for the sensitivity to LET
    changes". ``RBE_min = 1`` is a genuine modelling statement -- no significant
    relation between ``beta`` and LET was found (95% CI on that slope,
    -0.01 to 0.05, includes zero) -- not a missing parameter, and the base class
    supports it as the natural special case it is.

    Fitted on 10 cell lines / 24 RBE values over **LET 6-30 keV/um**, and the
    paper is explicit that eq. (1) "is not valid for LETs higher than 30 keV/um
    since it does not account for the decrease at higher LETs".

    At ``LET_d = 0`` the model collapses exactly onto the photon reference, which
    is the sanity anchor for the whole family.
    """

    model_id: ClassVar[str] = "WED"
    reference_doi: ClassVar[str] = "10.3109/0284186X.2012.705892"
    radiation_modes: ClassVar[tuple[str, ...]] = ("protons",)
    # The paper states this outright, immediately after its eq. (1): "Equation 1
    # is not valid for LETs higher than 30 keV/mum since it does not account for
    # the decrease at higher LETs." The fit used LET 6-30 keV/um.
    valid_let_keV_um: ClassVar[tuple[float, float] | None] = (0.0, 30.0)

    def _limits(self, let_keV_um: float) -> RBEMinMax:
        return RBEMinMax(
            rbe_min=1.0,
            rbe_max=1.0 + WEDENBERG_Q * let_keV_um / self.alpha_beta_ratio_Gy,
        )
