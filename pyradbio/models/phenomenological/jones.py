"""Jones 2015b -- saturating ion-response limits, keyed on alpha_x and beta_x."""

from __future__ import annotations

import math
from dataclasses import dataclass
from typing import ClassVar

from pyradbio.models.phenomenological.base import RBEMinMax, RBEMinMaxModel

JONES_CONTROL_LET_KEV_UM = 0.22
"""LET_d of the low-LET control radiation [keV/um]: the model's anchor, where
both limits are exactly 1. Implied by Rorvik's reformulation; the paper itself
states only the turnover LET below.
"""

JONES_TURNOVER_LET_KEV_UM = 30.5
"""LET_d of the LET-RBE turnover point, ``LET_U`` [keV/um] -- "both assumed to
rise linearly with LET to a maximum at 30.5 keV/um" (sec. 2.1).
"""

JONES_Q = 10.62
"""``alpha_U`` saturation, eq. (1): ``Q`` is the initial gradient [1] and ``j``
the saturation coefficient [Gy], from least-squares fitting (fig. 8).
"""
JONES_J_GY = 3.92

JONES_R = 2.5
"""``beta_U`` saturation, eq. (2): ``R`` [Gy^-1] and ``u`` [Gy^2]. ``R/u = 0.1``
is the paper's stated "maximum ceiling value of 0.1 Gy^-2 for beta_U".
"""
JONES_U_GY2 = 25.0


@dataclass(frozen=True, slots=True)
class Jones(RBEMinMaxModel):
    """Jones 2015b, Cancers 7:460-480, doi:10.3390/cancers7010460.

    Both LQ parameters "are assumed to rise linearly with LET to a maximum at
    30.5 keV/um and to a level determined by the low LET (control radiation)
    alpha and beta parameters" (sec. 2.1). Writing ``s`` for the fraction of the
    way from the control LET to that turnover point:

    ```text
    s = (LET_d - 0.22) / (30.5 - 0.22)

    RBE_max(LET_d, alpha_x) = 1 + s (alpha_U/alpha_x - 1)
    RBE_min(LET_d, beta_x)  = sqrt(1 + s (beta_U/beta_x - 1))
    ```

    where the *saturated* values at the turnover point come from the appendix:

    ```text
    alpha_U = (Q/j) (1 - exp(-j alpha_x))    Q = 10.62,  j = 3.92 Gy   eq. (1)
    beta_U  = (R/u) (1 - exp(-u beta_x))     R = 2.5,    u = 25 Gy^2   eq. (2)
    ```

    Both saturate with the tissue's own radiosensitivity, so a radioresistant
    tissue gains proportionally more. ``R/u = 0.1 Gy^-2`` is exactly the "maximum
    ceiling value" the paper states for ``beta_U``.

    **This model keys on ``alpha_x`` and ``beta_x`` separately, not on their
    ratio** -- like [`WilkensOelfke`][],
    [`LinearScaling`][] and
    [`Chen`][]. It is also the only model
    here anchored at a **non-zero** LET: at 0.22 keV/um both limits are 1.

    !!! note
        **Two departures from Rorvik et al. 2018's appendix B**, which is where this
        package first took the model from and which is not the primary:

        * ``alpha_U``: Rorvik prints a prefactor of ``2.696``; the paper's own
          ``Q/j = 10.62/3.92 = 2.7092`` is used here (0.5% higher).
        * ``beta_U``: Rorvik prints ``0.06 (1 - exp(-50 beta_x))``. The paper's
          eq. (2) is ``0.1 (1 - exp(-25 beta_x))``, and its stated ceiling of
          0.1 Gy^-2 confirms ``R/u = 0.1`` rather than 0.06. This is a genuine
          disagreement about the model, not a rounding difference.

        The paper's units are recoverable from its own derivation rather than from
        its printed superscripts: eq. (1) integrates
        ``d(alpha_U)/d(alpha_L) = Q - j alpha_U``, whose left side is dimensionless,
        so ``Q`` is dimensionless and ``j`` carries Gy -- and likewise ``u`` carries
        Gy^2. Rorvik prints these as Gy^-1 and Gy^-2, which cannot be right. The
        numbers were never in doubt; the dimensional argument settles the units.
    """

    model_id: ClassVar[str] = "JON"
    reference_doi: ClassVar[str] = "10.3390/cancers7010460"
    radiation_modes: ClassVar[tuple[str, ...]] = ("protons",)
    # Rorvik table A.1: the proton database spans LET_d 0.5-33.0 keV/um. The
    # (alpha/beta)_x column is open-ended above 0.5 Gy, so no ratio bound is set
    # -- and the model does not use the ratio anyway.
    valid_let_keV_um: ClassVar[tuple[float, float] | None] = (0.5, 33.0)

    @property
    def _let_scale_keV_um(self) -> float:
        """``LET_U - LET_control`` [keV/um] = 30.28, the model's LET denominator."""
        return JONES_TURNOVER_LET_KEV_UM - JONES_CONTROL_LET_KEV_UM

    @property
    def alpha_saturated_Gy(self) -> float:
        """``alpha_U`` [Gy^-1] at the turnover point -- the paper's eq. (1)."""
        return (JONES_Q / JONES_J_GY) * (1.0 - math.exp(-JONES_J_GY * self.photon.alpha_x))

    @property
    def beta_saturated_Gy2(self) -> float:
        """``beta_U`` [Gy^-2] at the turnover point -- the paper's eq. (2)."""
        return (JONES_R / JONES_U_GY2) * (1.0 - math.exp(-JONES_U_GY2 * self.photon.beta_x))

    def _limits(self, let_keV_um: float) -> RBEMinMax:
        scale = (let_keV_um - JONES_CONTROL_LET_KEV_UM) / self._let_scale_keV_um
        radicand = 1.0 + scale * (self.beta_saturated_Gy2 / self.photon.beta_x - 1.0)
        if radicand < 0.0:
            # beta_U < beta_x makes the radicand *fall* with LET, so a tissue that
            # is already more beta-sensitive than the saturation ceiling drives it
            # negative. A bare sqrt would raise "math domain error" naming neither
            # the model nor the input.
            raise ValueError(
                f"JON: RBE_min is imaginary at LET_d = {let_keV_um:g} keV/um for "
                f"beta_x = {self.photon.beta_x:g} Gy^-2 (radicand {radicand:g} < 0). "
                "The model saturates towards beta_U = "
                f"{self.beta_saturated_Gy2:g} Gy^-2, which is below this beta_x, so "
                "its RBE_min decreases with LET and leaves the real line"
            )
        return RBEMinMax(
            rbe_min=math.sqrt(radicand),
            rbe_max=1.0 + scale * (self.alpha_saturated_Gy / self.photon.alpha_x - 1.0),
        )
