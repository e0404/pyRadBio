"""Tilly et al. 2005 -- Wedenberg's functional form, fitted in two tissue bands.

Two classes rather than one with a swappable slope: the paper fits **two
databases**, one at low ``(alpha/beta)_x`` and one at high, and they carry
different fitted domains as well as different slopes. A single class could hold
only one ``valid_alpha_beta_ratio_Gy``.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import ClassVar

from pyradbio.models.phenomenological.base import RBEMinMax, RBEMinMaxModel

TILLY2_Q = 0.309
"""Slope of ``RBE_max - 1`` against ``LET_d/(alpha/beta)_x`` [Gy um / keV] for the
low-``(alpha/beta)_x`` database (around 2 Gy).

**Rorvik et al. 2018 appendix B is the only numeric source for this**, and
reading Tilly et al. 2005 does not change that: the primary states only that
"the alpha/alpha_ref values were empirically fitted as linear functions of the
LET where the slope is determined by the alpha_ref/beta_ref ratio, as shown in
figure 2" -- it prints no slope. Rorvik presumably read them off that figure.
"""

TILLY10_Q = 0.550964
"""The same slope for the high-``(alpha/beta)_x`` database (around 10 Gy); same
provenance caveat as [`TILLY2_Q`][].
"""

_DOI = "10.1088/0031-9155/50/12/003"


@dataclass(frozen=True, slots=True)
class _TillyBase(RBEMinMaxModel):
    """``RBE_max = 1 + q LET_d / (alpha/beta)_x``, ``RBE_min = 1``.

    Algebraically identical to Wedenberg et al. 2013 eq. (1) -- and predating it
    -- differing only in the fitted ``q``. Kept as its own class rather than a
    Wedenberg preset because the two are separate publications with separate
    databases, and provenance is the thing this package is careful about.
    """

    reference_doi: ClassVar[str] = _DOI
    radiation_modes: ClassVar[tuple[str, ...]] = ("protons",)

    q_Gy_um_keV: ClassVar[float]
    """Subclass supplies the fitted slope."""

    def _limits(self, let_keV_um: float) -> RBEMinMax:
        return RBEMinMax(
            rbe_min=1.0,
            rbe_max=1.0 + self.q_Gy_um_keV * let_keV_um / self.alpha_beta_ratio_Gy,
        )


@dataclass(frozen=True, slots=True)
class Tilly2(_TillyBase):
    """Tilly et al. 2005, Phys. Med. Biol. 50:2765-2777 -- the low ``(alpha/beta)_x`` fit.

    ``q = 0.309`` Gy (keV/um)^-1, fitted to 7 data points over ``(alpha/beta)_x``
    2.7-2.8 Gy and ``LET_d`` 7.7-20.0 keV/um (Rorvik et al. 2018 table A.1). The
    primary's fig. 2(a) identifies the two cell lines behind that band: V79-379A
    at ``alpha_ref/beta_ref = 2.7 Gy`` (Folkard et al. 1996) and V79-753B at
    2.8 Gy (Belli et al. 1993, 1998).

    Seven points is the smallest database in the family, and the band it covers
    is narrow enough that the ``1/(alpha/beta)_x`` scaling is essentially
    untested by this fit -- which is why the paper reports a second one
    ([`Tilly10`][]) rather than one model across both.

    The ``RBE_min = 1`` is the primary's own assumption, and it comes with the
    LET bound encoded above: "the beta values were assumed to be equal to the
    low-LET reference value up to a LET of 20 eV/nm" (sec. 2.2.3).
    """

    model_id: ClassVar[str] = "TIL2"
    q_Gy_um_keV: ClassVar[float] = TILLY2_Q
    valid_alpha_beta_ratio_Gy: ClassVar[tuple[float, float] | None] = (2.7, 2.8)
    valid_let_keV_um: ClassVar[tuple[float, float] | None] = (7.7, 20.0)


@dataclass(frozen=True, slots=True)
class Tilly10(_TillyBase):
    """Tilly et al. 2005 -- the high ``(alpha/beta)_x`` fit.

    ``q = 0.550964`` Gy (keV/um)^-1, fitted to 4 data points over
    ``(alpha/beta)_x`` 7.7-15 Gy and ``LET_d`` 7.7-19.7 keV/um. The primary's
    fig. 2(b) names these as SQ20B at ``alpha_ref/beta_ref = 7.6 Gy`` (Belli et
    al. 2000) and C3H10T1/2 at 15 Gy (Bettega et al. 1998) -- so Rorvik's lower
    bound of 7.7 is 7.6 in the paper. Kept at Rorvik's value, since 0.1 Gy on a
    band edge changes nothing and the slopes come from the same table.

    **The two slopes differ by 1.8x**, which is the whole point of the split: on
    a model whose only tissue dependence is ``1/(alpha/beta)_x``, needing a
    different ``q`` per band is the fit reporting that this scaling does not
    carry across the range. Rorvik's table A.2 records the consequence -- TIL is
    the one model in the survey marked "no, not in between TIL2/TIL10" for
    whether RBE behaves monotonically in the tissue parameter.
    """

    model_id: ClassVar[str] = "TIL10"
    q_Gy_um_keV: ClassVar[float] = TILLY10_Q
    valid_alpha_beta_ratio_Gy: ClassVar[tuple[float, float] | None] = (7.7, 15.0)
    valid_let_keV_um: ClassVar[tuple[float, float] | None] = (7.7, 19.7)
