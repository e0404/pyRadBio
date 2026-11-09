"""Mairani et al. 2016 -- the data-driven helium-ion RBE parameterisation."""

from __future__ import annotations

import math
from dataclasses import dataclass
from typing import ClassVar

from pyradbio.models.phenomenological.base import RBEMinMax, RBEMinMaxModel

# Mairani et al. 2016, table 4, the **fQE** row -- verified against the primary
# paper. fQE is one of ten candidate weighting functions the paper fits and the
# one it selects: lowest RMSE (1.665) and lowest jackknife control variable
# (delta^2 = 24.9) of the set. Units are the paper's table-4 note: k0 in [Gy]^-1,
# the degree-n polynomial coefficient in [Gy][keV um^-1]^-n, and the exponent
# coefficient in [keV um^-1]^-1.
MAIRANI_P0 = 1.36938e-1  # k0 [Gy^-1]
MAIRANI_P1 = 9.73154e-3  # k1 [Gy (keV/um)^-2]
MAIRANI_P2 = 1.51998e-2  # k2 [(keV/um)^-1]


@dataclass(frozen=True, slots=True)
class MairaniHelium(RBEMinMaxModel):
    """Mairani et al. 2016, Phys. Med. Biol. 61:888, doi:10.1088/0031-9155/61/2/888.

    The only non-proton member of the family, and the only one whose ``RBE_max``
    is non-monotone in LET. The paper's general form is
    ``RBE_alpha = 1 + [k0 + (alpha/beta)_ph^-1] f(L)``; we use its selected
    ``f_QE(L) = (k1 L^2) exp(-k2 L)``:

    ```text
    f_QE    = p1 * LET**2 * exp(-p2 * LET)
    RBE_max = 1 + (p0 + 1/(alpha/beta)_x) * f_QE
    RBE_min = 1
    ```

    ``f_QE`` peaks at ``LET = 2/p2`` ~ 132 keV/um and decays beyond it, which is
    the model's built-in overkill roll-over -- the qualitative feature the linear
    proton parameterisations cannot express, and the reason this one extrapolates
    far more gracefully to high LET.

    Like Wedenberg it fixes ``RBE_min = 1`` -- reportedly because fitted
    parameters gained nothing over a constant 1 -- so it reduces to the photon
    reference at ``LET = 0``.

    **Helium only.** Applying a helium fit to protons or carbon is rejected
    rather than extrapolated -- the LET dependence here is specific to the helium
    track structure the fit was built on.

    **Beware other Mairani parameterisations.** The paper fits ten candidate
    ``f(L)`` forms (fL, fQ, fLQ, fLQC, fLE, fQE, fLQE, fLE2, fQE2, fLQE2), each
    with its own coefficients in table 4, and later work uses different ones --
    Rorvik et al. 2018, for instance, parameterises a mixed proton+helium field
    using a revised ``f_LE2``. Coefficients from one form are meaningless in
    another, so check which ``f`` a quoted set belongs to before comparing.
    """

    model_id: ClassVar[str] = "HEL"
    reference_doi: ClassVar[str] = "10.1088/0031-9155/61/2/888"
    radiation_modes: ClassVar[tuple[str, ...]] = ("helium",)

    def _limits(self, let_keV_um: float) -> RBEMinMax:
        f_qe = MAIRANI_P1 * let_keV_um * let_keV_um * math.exp(-MAIRANI_P2 * let_keV_um)
        return RBEMinMax(
            rbe_min=1.0,
            rbe_max=1.0 + (MAIRANI_P0 + 1.0 / self.alpha_beta_ratio_Gy) * f_qe,
        )
