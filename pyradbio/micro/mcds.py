"""MCDS analytic DNA-damage yields (Semenenko & Stewart 2004/2006).

*Fast Monte Carlo simulation of DNA damage formed by electrons and light ions*,
Phys. Med. Biol. **51**(7):1693-1706, doi:10.1088/0031-9155/51/7/004 (the 2006
revision); the algorithm itself is Semenenko & Stewart 2004, Radiat. Res.
**161**(4):451-457, doi:10.1667/RR3140.

**No Monte Carlo is run here.** The 2006 paper's sec. 2.2 fits the published
track-structure simulations of Nikjoo et al. and Friedland et al. to a closed
form, and that closed form is what the RMF model consumes:

```text
(3)  f(x) = u + (v - u) x / (x + w),        x = Z_eff^2 / beta^2
```

with ``f(1) ~ f(0) = u`` and ``f(inf) = v``, so ``u`` is the low-quality (photon
/ fast-ion) limit, ``v`` the saturation value and ``w`` the half-saturation
point. The fitted constants are printed in sec. 2.2 (yields) and eq. (5)
(``n_seg``).

The quality parameter is Katz's:

```text
(1)  Z_eff = Z [1 - exp(-125 beta Z^{-2/3})]     (Barkas 1963)
(2)  beta  = sqrt(1 - 1/(1 + T/m_0 c^2)^2)
```

and the paper cites **Katz (1970)** for using ``Z^2/beta^2`` as the variable
against which to interpolate radiation quality -- the same scaling
[`pyradbio.models.katz`][] uses, and the same effective charge
[`pyradbio.track.stopping_power.effective_charge`][] already provides, which
is delegated to rather than duplicated.

**Trends.** DSB yield *rises* with ``x`` (48.9 -> 164.5 Gy^-1 cell^-1) while SSB
yield *falls* (1134 -> 291): denser tracks convert single-strand breaks into
double-strand breaks rather than making more lesions overall.

**Validity.** The fit is stated as appropriate for ``1 <= x <= 3200``. Outside
that this module **warns and evaluates**; it never clamps, so a caller can count
out-of-domain points rather than silently believing them.
"""

from __future__ import annotations

import math
import warnings
from dataclasses import dataclass

from pyradbio.core.radiation import Ion
from pyradbio.track.stopping_power import effective_charge

MCDS_DOMAIN_MIN: float = 1.0
"""Fit domain of eq. (3), sec. 2.3: "most appropriate for 1 <= Z_eff^2/beta^2
<= 3200". At the upper limit T ~ 0.105 MeV (protons) and ~2 MeV (alphas).
"""
MCDS_DOMAIN_MAX: float = 3200.0

GBP_PER_CELL: float = 6.0
"""Gbp of DNA per cell implied by the paper's own dual-unit yields: it prints
48.9/8.15, 164.5/27.4, 1134/189 and 291/48.5, all of which give 6.0. That is
a diploid mammalian genome, and it is the only conversion between the two
unit systems used in this lineage (Carlson 2008 works per cell, Frese 2012
per Gbp).
"""


def effective_charge_barkas(z: int, beta: float) -> float:
    """MCDS eq. (1) ``Z_eff = Z[1 - exp(-125 beta Z^{-2/3})]`` (Barkas 1963).

    Identical to [`pyradbio.track.stopping_power.effective_charge`][] and to
    Katz et al. 1971 eq. (12); delegated so the three cannot drift apart.
    """
    return effective_charge(z, beta)


def quality_parameter(z: int, beta: float, *, use_effective_charge: bool = True) -> float:
    """The radiation-quality variable ``x = Z_eff^2 / beta^2``.

    Sec. 2.3 writes it as "``Z_eff^2/beta^2`` (or ``Z^2/beta^2`` for
    electrons)", so ``use_effective_charge=False`` selects the electron
    convention, where screening does not apply.
    """
    if not 0.0 < beta < 1.0:
        raise ValueError(f"beta must be in (0, 1), got {beta!r}")
    charge = effective_charge_barkas(z, beta) if use_effective_charge else float(z)
    return charge * charge / (beta * beta)


def _interpolate(x: float, u: float, v: float, w: float) -> float:
    """Eq. (3) ``f(x) = u + (v - u) x / (x + w)``."""
    if x < 0.0:
        raise ValueError(f"x must be >= 0, got {x!r}")
    if math.isinf(x):
        return v
    return u + (v - u) * x / (x + w)


@dataclass(frozen=True, slots=True)
class MCDSYields:
    """Closed-form MCDS damage yields as a function of track quality.

    Every constant below is printed in Semenenko & Stewart 2006; none is fitted
    here. Yields are per Gy, either per cell (the paper's and Carlson 2008's
    unit) or per Gbp (Frese 2012's).
    """

    # -- eq. (3) constants for DSBs, sec. 2.2 [Gy^-1 cell^-1] --------------
    dsb_u_per_Gy_per_cell: float = 48.9
    dsb_v_per_Gy_per_cell: float = 164.5
    dsb_w: float = 759.0
    # -- eq. (3) constants for SSBs, sec. 2.2 [Gy^-1 cell^-1] --------------
    ssb_u_per_Gy_per_cell: float = 1134.0
    ssb_v_per_Gy_per_cell: float = 291.0
    ssb_w: float = 1615.0
    # -- eq. (5): the MCDS algorithm's own inputs --------------------------
    sigma_Sb_per_Gy_per_cell: float = 1300.0
    """Strand breaks per Gy per cell, fixed at its approximate optimum."""
    base_damage_to_strand_break_ratio: float = 3.0
    """Base damages per strand break, carried over from the 2004 paper."""
    n_min_bp: int = 9
    """Minimum undamaged bp separating two clusters."""
    n_seg_u_bp: float = 149200.0
    """eq. (3) constants for the DNA segment length [bp Gy^-1 cell^-1]."""
    n_seg_v_bp: float = 25600.0
    n_seg_w: float = 267.0

    # ---- domain ---------------------------------------------------------
    def _check(self, x: float, check_domain: bool) -> None:
        if check_domain and not MCDS_DOMAIN_MIN <= x <= MCDS_DOMAIN_MAX:
            warnings.warn(
                f"x = Z_eff^2/beta^2 = {x:.3g} is outside the MCDS fit domain "
                f"[{MCDS_DOMAIN_MIN:g}, {MCDS_DOMAIN_MAX:g}] "
                "(Semenenko & Stewart 2006 sec. 2.3); evaluated, not clamped",
                UserWarning,
                stacklevel=3,
            )

    # ---- DSBs -----------------------------------------------------------
    def dsb_per_Gy_per_cell(self, x: float, *, check_domain: bool = False) -> float:
        """DSB yield [Gy^-1 cell^-1] at track quality ``x`` (eq. 3)."""
        self._check(x, check_domain)
        return _interpolate(x, self.dsb_u_per_Gy_per_cell, self.dsb_v_per_Gy_per_cell, self.dsb_w)

    def dsb_per_Gy_per_Gbp(self, x: float, *, check_domain: bool = False) -> float:
        """DSB yield [Gy^-1 Gbp^-1] -- the unit Frese et al. 2012 uses."""
        return self.dsb_per_Gy_per_cell(x, check_domain=check_domain) / GBP_PER_CELL

    def dsb_yield_limits_per_Gy_per_cell(self) -> tuple[float, float]:
        """The printed range ``(u, v)`` the yield can take, for any ``x``."""
        return (self.dsb_u_per_Gy_per_cell, self.dsb_v_per_Gy_per_cell)

    # ---- SSBs -----------------------------------------------------------
    def ssb_per_Gy_per_cell(self, x: float, *, check_domain: bool = False) -> float:
        """SSB yield [Gy^-1 cell^-1] at track quality ``x`` (eq. 3).

        Note ``u > v`` here: SSB yield *falls* as tracks get denser.
        """
        self._check(x, check_domain)
        return _interpolate(x, self.ssb_u_per_Gy_per_cell, self.ssb_v_per_Gy_per_cell, self.ssb_w)

    def ssb_per_Gy_per_Gbp(self, x: float, *, check_domain: bool = False) -> float:
        """SSB yield [Gy^-1 Gbp^-1]."""
        return self.ssb_per_Gy_per_cell(x, check_domain=check_domain) / GBP_PER_CELL

    # ---- algorithm input ------------------------------------------------
    def n_seg_bp(self, x: float, *, check_domain: bool = False) -> float:
        """DNA segment length [bp Gy^-1 cell^-1] for the MCDS algorithm (eq. 5).

        Printed as ``149200 - 123600 x/(x + 267)``, which is eq. (3) with
        ``v = 25600``; both forms are the same function.
        """
        self._check(x, check_domain)
        return _interpolate(x, self.n_seg_u_bp, self.n_seg_v_bp, self.n_seg_w)

    # ---- ion convenience ------------------------------------------------
    def dsb_per_Gy_per_Gbp_for_ion(self, ion: Ion, *, check_domain: bool = False) -> float:
        """DSB yield [Gy^-1 Gbp^-1] for ``ion``, via ``x = Z_eff^2/beta^2``."""
        x = quality_parameter(ion.Z, ion.beta())
        return self.dsb_per_Gy_per_Gbp(x, check_domain=check_domain)

    def dsb_per_Gy_per_cell_for_ion(self, ion: Ion, *, check_domain: bool = False) -> float:
        """DSB yield [Gy^-1 cell^-1] for ``ion``."""
        x = quality_parameter(ion.Z, ion.beta())
        return self.dsb_per_Gy_per_cell(x, check_domain=check_domain)


__all__ = [
    "GBP_PER_CELL",
    "MCDS_DOMAIN_MAX",
    "MCDS_DOMAIN_MIN",
    "MCDSYields",
    "effective_charge_barkas",
    "quality_parameter",
]
