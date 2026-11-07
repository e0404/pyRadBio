"""Rorvik et al. 2017 -- the unweighted linear fit of the RØR model pair."""

from __future__ import annotations

from dataclasses import dataclass
from typing import ClassVar

from pyradbio.models.phenomenological.base import RBEMinMax, RBEMinMaxModel

#: Slope of ``RBE_max - 1`` against ``LET_d/(alpha/beta)_x`` [Gy um / keV], from
#: the linear *unweighted* regression. Rorvik et al. 2018 appendix B.
RORVIK_UNWEIGHTED_Q = 0.645


@dataclass(frozen=True, slots=True)
class RorvikUnweighted(RBEMinMaxModel):
    """Rorvik et al. 2017 (RORU), Med. Phys. 44:2586-2594, doi:10.1002/mp.12216.

    Same functional form as Wedenberg et al. 2013 and Tilly et al. 2005::

        RBE_max = 1 + q LET_d / (alpha/beta)_x        with q = 0.645
        RBE_min = 1

    fitted to 85 data points over ``(alpha/beta)_x`` 1.2-18.4 Gy and ``LET_d``
    0.4-37.8 keV/um -- the widest tissue range of any linear member of the family,
    which is what makes its slope worth having beside Wedenberg's 0.434.

    **Only the unweighted arm is implemented.** The companion RØRW model replaces
    this linear slope with a quartic *weighting function* integrated over a LET
    **spectrum** ``d(L)``, not evaluated at a scalar ``LET_d``. It is therefore
    not expressible on this base class, and evaluating its quartic at ``LET_d``
    would be a different model wearing its name. See
    ``docs/references/Phenomenological_RBE_models.md``.

    Rorvik et al. 2018's own survey puts this model among the highest RBE
    predictors of the eleven, alongside WIL and JON -- unsurprising given ``q``
    is ~50% above Wedenberg's. The 2017 primary makes the same point directly:
    the difference from Wedenberg "is most likely explained by the use of
    different databases, with our model derived from a larger set of data points
    and wider range of LET_d".

    .. note::
       The primary's discussion says the LET_d-based simplification would have a
       "linear coefficient of LET_d ... 0.635 Gy(keV/um)^-1", while both its
       table A.2 (unweighted linear fit) and its appendix A.4 print **0.645**.
       0.645 is right and 0.635 is a slip in that sentence: for a *linear*
       weighting function the reduction is exact, not an approximation, because
       ``integral L d(L) dL`` **is** ``LET_d`` by definition -- so the spectrum
       coefficient and the LET_d coefficient are necessarily the same number.
    """

    model_id: ClassVar[str] = "RORU"
    reference_doi: ClassVar[str] = "10.1002/mp.12216"
    radiation_modes: ClassVar[tuple[str, ...]] = ("protons",)
    valid_alpha_beta_ratio_Gy: ClassVar[tuple[float, float] | None] = (1.2, 18.4)
    valid_let_keV_um: ClassVar[tuple[float, float] | None] = (0.4, 37.8)

    def _limits(self, let_keV_um: float) -> RBEMinMax:
        return RBEMinMax(
            rbe_min=1.0,
            rbe_max=1.0 + RORVIK_UNWEIGHTED_Q * let_keV_um / self.alpha_beta_ratio_Gy,
        )
