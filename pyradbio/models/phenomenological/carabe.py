"""Carabe et al. 2012 -- both RBE limits linear in LET_d."""

from __future__ import annotations

from dataclasses import dataclass
from typing import ClassVar

from pyradbio.models.phenomenological.base import RBEMinMax, RBEMinMaxModel

# Carabe et al. 2012. The slopes are a coefficient times a common scale factor
# (2.686), kept factored here as other implementations write them, which is why
# the composed slopes are not round numbers.
# Not read from the 2012 paper's own text (no PDF in the library), but
# TRIANGULATED by three primary sources that have been read in this repo, which
# is why there is no `# UNVERIFIED` marker here:
#   * Rorvik et al. 2018 appendix (doi:10.1088/1361-6560/aad9db) prints these
#     coefficients term-for-term, factored 2.686 included, under "the Carabe
#     (CAR) model";
#   * McNamara et al. 2015 sec. 2.3 describes the form, and its reference list
#     resolves "Carabe et al 2012" to Phys. Med. Biol. 57:1159 -- i.e. despite
#     its title being about *range uncertainty*, that paper is where the fit
#     lives;
#   * Wedenberg et al. 2013 (Discussion) describes how it was fitted.
# The formalism is separately verified against Carabe-Fernandez et al. 2007.
# See docs/references/Phenomenological_RBE_models.md.
CARABE_RBE_MAX_INTERCEPT = 0.843
CARABE_RBE_MIN_INTERCEPT = 1.09
CARABE_SCALE = 2.686  # common LET scale factor [Gy um / keV]
CARABE_RBE_MAX_SLOPE = 0.154 * CARABE_SCALE  # = 0.41364
CARABE_RBE_MIN_SLOPE = 0.006 * CARABE_SCALE  # = 0.016116


@dataclass(frozen=True, slots=True)
class Carabe(RBEMinMaxModel):
    """Carabe et al. 2012, Phys. Med. Biol. 57:1159, doi:10.1088/0031-9155/57/5/1159.

    Both asymptotes are linear in ``LET_d`` with a ``1/(alpha/beta)_x`` slope:

    ```text
    RBE_max = 0.843 + 0.154*2.686 * LET_d / (alpha/beta)_x
    RBE_min = 1.09  + 0.006*2.686 * LET_d / (alpha/beta)_x
    ```

    Two features set this model apart from the others in the package and are
    pinned by test:

    * ``RBE_max(0) = 0.843 < 1`` -- unlike Wedenberg and Mairani it does **not**
      reduce to the photon reference at zero LET. That is a property of the fit,
      not an error, and it is why an "RBE = 1 at LET = 0" assertion cannot be a
      shared acceptance criterion for the family.
    * ``RBE_min = 1.09 + ... > 1`` and can exceed ``RBE_max`` at high
      ``(alpha/beta)_x`` and low LET, so this model can predict RBE *rising* with
      dose. McNamara et al. 2015 sec. 3.2 discusses exactly this behaviour and
      notes no other model in the comparison shows it.

    The formalism originates in Carabe-Fernandez, Dale & Jones 2007
    (doi:10.1080/09553000601087176), which introduced ``RBE_min``; the
    coefficients here are the 2012 re-fit.

    **Why it is the outlier at high ``(alpha/beta)_x``.** Wedenberg et al. 2013's
    Discussion records how the fit was made: "their approach for parameter
    determination differs from the one used in this study in that they fit their
    model only to experimental data from **V79 cells**. [...] we fit to
    experimental data of a range of different (alpha/beta)_phot ratios and LET
    values simultaneously". So the ``1/(alpha/beta)_x`` tissue scaling here is an
    *extrapolation away from a single cell line*, not something fitted across
    tissues -- which is the most likely reason this model separates from the
    others as ``(alpha/beta)_x`` moves away from V79's ~5 Gy (McNamara 2015
    sec. 3.2 measures the divergence at up to 14% at 2 keV/um).
    """

    model_id: ClassVar[str] = "CAR"
    reference_doi: ClassVar[str] = "10.1088/0031-9155/57/5/1159"
    radiation_modes: ClassVar[tuple[str, ...]] = ("protons",)

    def _limits(self, let_keV_um: float) -> RBEMinMax:
        ratio = self.alpha_beta_ratio_Gy
        return RBEMinMax(
            rbe_min=CARABE_RBE_MIN_INTERCEPT + CARABE_RBE_MIN_SLOPE * let_keV_um / ratio,
            rbe_max=CARABE_RBE_MAX_INTERCEPT + CARABE_RBE_MAX_SLOPE * let_keV_um / ratio,
        )
