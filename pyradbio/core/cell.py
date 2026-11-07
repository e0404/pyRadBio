"""Cell / target geometry.

The LEM uses an *effective* nuclear cross-section ~80% of the geometric one
(Scholz & Kraft 1996 'geometrical structure'; Elsasser et al. 2008 sec.).
"""

from __future__ import annotations

from dataclasses import dataclass


@dataclass(frozen=True, slots=True)
class CellTarget:
    nucleus_radius_um: float
    domain_radius_um: float | None = None  # MKM domain radius r_d
    effective_size_factor: float = 1.0  # effective/geometric nuclear radius (LEM)
    height_um: float | None = None  # cylinder height; defaults to 2*radius if None

    def __post_init__(self) -> None:
        if self.nucleus_radius_um <= 0.0:
            raise ValueError("nucleus_radius_um must be > 0")
        if self.domain_radius_um is not None and self.domain_radius_um <= 0.0:
            raise ValueError("domain_radius_um must be > 0 when given")
        if not 0.0 < self.effective_size_factor <= 1.0:
            raise ValueError("effective_size_factor must be in (0, 1]")
        if self.height_um is not None and self.height_um <= 0.0:
            raise ValueError("height_um must be > 0 when given")

    def effective_nucleus_radius_um(self) -> float:
        """Effective nuclear radius = ``effective_size_factor`` x geometric.

        ``effective_size_factor`` converts a **measured/geometric** nuclear size
        into the **effective** one the LEM's hit statistics need. Scholz & Kraft
        1996 (doi:10.1016/0273-1177(95)00784-C, "geometrical structure of the
        critical targets") explain why the two differ: the survival weighting
        ``P(0) = sum_i h(a_i) exp(-a_i*F)`` favours small nuclei, so "an effective
        geometrical nuclear cross section has to be used to represent a size
        distribution, which is in general smaller than the arithmetic mean value".
        For asynchronous Chinese hamster cells they measured an average nuclear
        size of 75 um^2 but an effective size of only 50 um^2 -- a factor 0.67 in
        *area*, i.e. **0.82 in radius**, which is where the ~0.8 comes from.

        The *procedure* originates with Todd, Wood, Walker & Weiss 1985 (Radiat.
        Res. Suppl. 8, 104, S5-S12), which Elsasser & Scholz 2007 p. 320 cites for
        exactly this ("for the calculation procedure to determine the effective
        size, we refer to ref. 17"). Its eq. (3),
        ``sigma_bar = -ln[sum_i x_i exp(-sigma_i f)]/f``, is a fluence-weighted
        average applied to a measured nuclear-area histogram; for T-1E cells it
        "reduce[d] a typical arithmetic mean nuclear area of 200 to 120 um^2 at a
        typical alpha-particle fluence" (0.60 in area = 0.78 in radius).

        **The reduction is fluence-dependent and vanishes at the initial slope.**
        As ``f -> 0``, eq. (3) tends to ``sum_i x_i sigma_i`` -- the arithmetic
        mean -- so an ``alpha``-only calculation has no business applying it. Use
        the factor for finite-fluence/survival work, not for RBE_alpha panels; see
        ``docs/references/LEM_II_Elsasser2007.md``. LEM III
        (Elsasser et al. 2008) reuses it to turn Suzuki 2000's geometric areas into
        effective ones. Its Methods confirm the factor is on the **radius**: "the
        effective nuclear radius is approximately 80% of the average nuclear radius"
        (and Table 1 caption: the radius was "taken as 80% of the average value").

        **The default is 1.0, and that is deliberate.** Published LEM inputs almost
        always quote the *effective* size already -- Scholz & Kraft's own 50 um^2 is
        the effective value, and Elsasser & Scholz 2007's 3.8 um (V79) / 4.7 um
        (CHO) are used as-is. Applying 0.8 to such a number shrinks it a second
        time: the area falls by 0.64, ``D_1 = LET/(rho*A)`` rises by 1.56x, overkill
        starts too early, and every ``RBE_alpha`` peak lands ~1.25x low in LET and
        10-20% low in height. Applied silently, it mimics a LEM II
        cluster deficit (see ``benchmark/elsasser2007.py``). Pass 0.8 explicitly
        only when the radius you hold is genuinely geometric.
        """
        return self.effective_size_factor * self.nucleus_radius_um

    def cylinder_height_um(self) -> float:
        """Nuclear-cylinder height; defaults to the geometric diameter."""
        if self.height_um is not None:
            return self.height_um
        return 2.0 * self.nucleus_radius_um
