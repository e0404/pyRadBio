"""The time axis: dose protraction and the Lea-Catcheside ``G`` factor.

Model-agnostic, like :mod:`pyradbio.core.oxygen`. The LQ model as used
everywhere else in this package describes an *acute* dose: the quadratic term
counts pairs of sublethal lesions that meet, and every pair is available
because the whole dose arrives at once. Spread the same dose over time and
some of those lesions are repaired before their partner appears, so the
quadratic term is reduced by a factor ``G <= 1``::

    -ln S(D) = alpha D + G beta D^2

``G`` is a property of the *temporal pattern* of the irradiation alone. It does
not depend on the total dose, on the cell's ``alpha``/``beta``, or on the
radiation quality -- only on how the dose is distributed in time relative to
the sublethal-damage repair half-time. That is why it lives here rather than
inside any one model, and why :class:`DoseSchedule` carries no biology.

Definition
----------
Lea & Catcheside's generalised time factor (Lea & Catcheside 1942, *J. Genet.*
44:216; in the modern LQ notation of Kellerer & Rossi's theory of dual
radiation action, and reviewed by Brenner 2008)::

    G = (2 / D^2) * int dt Ddot(t) int_{-inf}^{t} dt' Ddot(t') exp(-lam (t - t'))

with ``Ddot`` the dose rate [Gy/h], ``D = int Ddot dt`` the total dose and
``lam = ln2 / T_half`` the first-order repair rate constant [1/h].

Exact evaluation
----------------
This module represents an irradiation as a sequence of non-overlapping blocks,
each delivered at a **constant dose rate**. That class covers every schedule of
practical interest -- a single acute dose, continuous low-dose-rate or
brachytherapy delivery, a fractionated course, a split course, an arbitrary
mixture -- and for it the double integral has an elementary closed form, so no
quadrature is needed. Writing ``D_i`` for the dose of block ``i``,
``T_i`` for its duration, ``x_i = lam T_i``, and ``g_ij >= 0`` for the gap
between the end of an earlier block ``j`` and the start of a later block ``i``::

    G = (2 / D^2) [ sum_i D_i^2 f(x_i)
                    + sum_{i > j} D_i D_j h(x_i) h(x_j) exp(-lam g_ij) ]

    f(x) = (x - 1 + e^-x) / x^2      (a block interacting with itself)
    h(x) = (1 - e^-x) / x            (a block seen by a later one)

Both ``f`` and ``h`` are written so that no cancellation occurs as ``x -> 0``
(:func:`_self_overlap`, :func:`_block_weight`); ``f(0) = 1/2`` and ``h(0) = 1``.
The known special cases drop out of the same expression and are *not*
implemented separately:

- one instantaneous block: ``G = 1`` exactly, the acute LQ reference;
- one block of duration ``T``: ``G = 2 (x - 1 + e^-x) / x^2``, the continuous /
  low-dose-rate form, tending to ``2 / (lam T)`` for slow delivery. This is
  **Thames 1985 eq. (8)** verbatim -- his ``g(mu t) = 2[mu t - 1 +
  exp(-mu t)]/(mu t)^2`` -- and the same function underlies Dale 1985's ``K``;
- ``n`` equal instantaneous fractions spaced by ``tau``, ``phi = e^(-lam tau)``:
  ``G = 1/n + (2/n^2) sum_{k=1}^{n-1} (n - k) phi^k``, the incomplete-repair
  sum, running from ``1`` (no repair) to ``1/n`` (complete repair). This is
  **Thames 1985 eq. (7)** rearranged: he writes ``-ln s_n = n(alpha x + beta
  x^2) + n beta x^2 h_n(phi)`` with ``h_n(phi) = (2/n)[phi/(1-phi)][n -
  (1-phi^n)/(1-phi)]``, and with ``D = n x`` that is exactly ``G = (1 + h_n)/n``
  -- including the closed form of the sum, which is what this module evaluates.
  His stated limits ``h_n(0) = 0`` and ``h_n(1) = n - 1`` are our ``1/n`` and
  ``1`` bounds.

The tests in ``tests/core/test_protraction.py`` check the general routine
against both published closed forms and against brute-force quadrature of the
definition above.

Units
-----
Doses in Gy, times in **hours** (the unit sublethal-damage repair half-times are
tabulated in). ``lam = ln2 / T_half`` is Thames' own convention (his sec. 4.1,
"``mu = ln 2 / T_1/2``"). There is deliberately no default repair half-time: it
is a tissue property spanning roughly 0.2-5 h in the literature, it is the
single most influential parameter here, and picking a number for the caller
would be exactly the "silent physics" ``AGENTS.md`` forbids.

Where this stops being valid
----------------------------
Both source papers bound their own applicability, and neither bound is checked
here -- they are the caller's to respect:

- **``G`` corrects the quadratic term only.** Thames 1985 sec. 2.2, point (1):
  "incomplete repair affects only the beta-component of killing, and not the
  alpha-component". That is a *derivation* result, not an approximation, and it
  is why ``g`` multiplies ``beta`` alone in :mod:`pyradbio.core.lq`.
- **Dose per fraction.** The whole construction assumes the LQ form is adequate
  at the per-fraction dose. Thames sec. 4.2(b) puts that at roughly ``x <= 4 Gy``
  for the cell line he analyses, beyond which the LQ approximation "may not be
  strictly valid (without a cubic term)".
- **Dose rate.** For continuous exposure the model requires ``v << mu / beta``
  (Thames eq. 42) -- i.e. repair must be able to compete with the delivery of
  new damage.
- **Repair is first-order and cannot saturate** (Thames sec. 4.2(a)), and is
  described by a *single* exponential. Real repair is at least bi-exponential.
  Note ``G`` is **linear in the repair kernel**, so a multi-exponential kernel
  ``sum_i w_i exp(-lam_i t)`` needs no new machinery here: it is
  ``sum_i w_i * lea_catcheside_g(schedule, T_half_i)``.
- Proliferation/repopulation is ignored throughout, as in both papers.

References
----------
- Lea, D. E. & Catcheside, D. G. (1942) *J. Genet.* 44:216-245 -- the time factor.
- Withers, H. R., Thames, H. D. & Peters, L. J. (1983)
  doi:10.1016/s0167-8140(83)80021-8 -- fraction-size isoeffect, the EQD2 lineage.
- Jones, B., Dale, R. G., Deehan, C., Hopkins, K. I. & Morgan, D. A. L. (2001)
  doi:10.1053/clon.2001.9221 -- BED in clinical use.
- Dale, R. G. (1985) *Br. J. Radiol.* 58:515-528,
  doi:10.1259/0007-1285-58-690-515 -- the "relative effectiveness" ``RE`` and
  "extrapolated response dose" ``ERD`` (eqs. 5-7), which is the quantity now
  called BED, and the reciprocal-dose test for protracted irradiation (eq. 15).
  **Read and verified**; its table II is reproduced as a golden in
  ``tests/core/test_protraction.py``.
- Thames, H. D. (1985) *Int. J. Radiat. Biol.* 47:319-339,
  doi:10.1080/09553008514550461 -- the incomplete-repair model. Eq. (7) is the
  fractionated case and eq. (8) the continuous case, both confirmed verbatim
  against this implementation. **Read and verified.**
"""

from __future__ import annotations

import math
from dataclasses import dataclass
from itertools import pairwise

#: Repair rate constant per unit half-time: ``lam = LN2 / T_half``.
LN2: float = math.log(2.0)

#: Below this ``x = lam T`` the closed forms are evaluated by their Taylor
#: series. Chosen where the series truncation error (~x^3/120 for ``f``) and the
#: cancellation error of the direct form (~eps/x) are both far below 1e-15.
_SERIES_CUTOFF: float = 1e-4


def _self_overlap(x: float) -> float:
    """``f(x) = (x - 1 + e^-x) / x^2``, the weight of a block against itself.

    ``f(0) = 1/2`` (an instantaneous block, which sees all of its own dose).
    The direct form cancels catastrophically for small ``x`` -- both ``x`` and
    ``1 - e^-x`` are O(x) while their difference is O(x^2) -- so the series is
    used there.
    """
    if x < _SERIES_CUTOFF:
        return 0.5 - x / 6.0 + x * x / 24.0
    return (x + math.expm1(-x)) / (x * x)


def _block_weight(x: float) -> float:
    """``h(x) = (1 - e^-x) / x``, the weight a block carries into later blocks.

    ``h(0) = 1``. Evaluated with :func:`math.expm1`, which is exact to full
    relative precision as ``x -> 0``, so no series branch is needed.
    """
    if x == 0.0:
        return 1.0
    return -math.expm1(-x) / x


@dataclass(frozen=True, slots=True)
class DoseBlock:
    """One irradiation delivered at a constant dose rate.

    Parameters
    ----------
    dose_Gy
        Dose delivered by this block, ``>= 0``.
    start_h
        Time at which the block starts [h]. Only differences matter; the origin
        is arbitrary.
    duration_h
        Duration of the block [h], ``>= 0``. ``0.0`` means instantaneous, i.e.
        short compared with the repair half-time -- the usual idealisation of a
        single treatment fraction.
    """

    dose_Gy: float
    start_h: float
    duration_h: float = 0.0

    def __post_init__(self) -> None:
        if not math.isfinite(self.dose_Gy) or self.dose_Gy < 0.0:
            raise ValueError(f"dose_Gy must be finite and >= 0, got {self.dose_Gy!r}")
        if not math.isfinite(self.start_h):
            raise ValueError(f"start_h must be finite, got {self.start_h!r}")
        if not math.isfinite(self.duration_h) or self.duration_h < 0.0:
            raise ValueError(f"duration_h must be finite and >= 0, got {self.duration_h!r}")

    @property
    def end_h(self) -> float:
        """Time at which the block finishes [h]."""
        return self.start_h + self.duration_h


@dataclass(frozen=True, slots=True)
class DoseSchedule:
    """A complete irradiation: non-overlapping :class:`DoseBlock` s in time.

    Blocks are sorted by start time on construction and must not overlap, which
    is what lets :func:`lea_catcheside_g` split the double integral into a
    self-term per block plus one ordered cross-term per pair.
    """

    blocks: tuple[DoseBlock, ...]

    def __post_init__(self) -> None:
        if not self.blocks:
            raise ValueError("a schedule needs at least one block")
        ordered = tuple(sorted(self.blocks, key=lambda b: b.start_h))
        for earlier, later in pairwise(ordered):
            if later.start_h < earlier.end_h:
                raise ValueError(
                    "dose blocks must not overlap in time, got a block starting at "
                    f"{later.start_h!r} h while another runs to {earlier.end_h!r} h"
                )
        object.__setattr__(self, "blocks", ordered)

    @property
    def total_dose_Gy(self) -> float:
        """Total dose delivered by the schedule [Gy]."""
        return math.fsum(block.dose_Gy for block in self.blocks)

    @property
    def duration_h(self) -> float:
        """Elapsed time from the start of the first block to the end of the last [h]."""
        return self.blocks[-1].end_h - self.blocks[0].start_h

    @classmethod
    def acute(cls, dose_Gy: float) -> DoseSchedule:
        """A single instantaneous dose -- the acute LQ reference, ``G = 1``."""
        return cls((DoseBlock(dose_Gy=dose_Gy, start_h=0.0, duration_h=0.0),))

    @classmethod
    def continuous(cls, dose_Gy: float, *, duration_h: float) -> DoseSchedule:
        """``dose_Gy`` delivered at a constant rate over ``duration_h``."""
        return cls((DoseBlock(dose_Gy=dose_Gy, start_h=0.0, duration_h=duration_h),))

    @classmethod
    def at_dose_rate(cls, dose_rate_Gy_per_h: float, *, duration_h: float) -> DoseSchedule:
        """Constant-dose-rate irradiation given the rate rather than the dose."""
        if not math.isfinite(dose_rate_Gy_per_h) or dose_rate_Gy_per_h < 0.0:
            raise ValueError(
                f"dose_rate_Gy_per_h must be finite and >= 0, got {dose_rate_Gy_per_h!r}"
            )
        return cls.continuous(dose_rate_Gy_per_h * duration_h, duration_h=duration_h)

    @classmethod
    def fractionated(
        cls,
        dose_per_fraction_Gy: float,
        *,
        n_fractions: int,
        interval_h: float,
        fraction_duration_h: float = 0.0,
    ) -> DoseSchedule:
        """``n_fractions`` equal fractions, ``interval_h`` apart start-to-start.

        ``fraction_duration_h`` is the delivery time of one fraction; the
        default ``0.0`` treats a fraction as instantaneous, which is the usual
        assumption for a few minutes of beam-on against a repair half-time of
        hours. ``interval_h`` is measured start-to-start, so it must be at least
        ``fraction_duration_h``.
        """
        if n_fractions < 1:
            raise ValueError(f"n_fractions must be >= 1, got {n_fractions!r}")
        if not math.isfinite(interval_h) or interval_h < 0.0:
            raise ValueError(f"interval_h must be finite and >= 0, got {interval_h!r}")
        if n_fractions > 1 and interval_h < fraction_duration_h:
            raise ValueError(
                f"interval_h ({interval_h!r}) is measured start-to-start and cannot be "
                f"shorter than fraction_duration_h ({fraction_duration_h!r})"
            )
        return cls(
            tuple(
                DoseBlock(
                    dose_Gy=dose_per_fraction_Gy,
                    start_h=i * interval_h,
                    duration_h=fraction_duration_h,
                )
                for i in range(n_fractions)
            )
        )


def repair_rate_per_h(repair_half_time_h: float) -> float:
    """Sublethal-damage repair rate constant ``lam = ln2 / T_half`` [1/h].

    ``repair_half_time_h = inf`` means no repair and returns ``0.0``.
    """
    if math.isnan(repair_half_time_h) or repair_half_time_h <= 0.0:
        raise ValueError(f"repair_half_time_h must be > 0, got {repair_half_time_h!r}")
    if math.isinf(repair_half_time_h):
        return 0.0
    return LN2 / repair_half_time_h


def repair_half_time_h_from_rate(repair_rate_per_h: float) -> float:
    """Inverse of :func:`repair_rate_per_h`: ``T_half = ln2 / lam`` [h].

    For models that state the repair *rate* rather than the half-time -- the IMK
    model's ``(a+c)`` (Matsuya et al. 2025 eq. 1) is the case in hand. A rate of
    ``0.0`` means no repair and returns ``inf``, for which ``G = 1``.
    """
    if math.isnan(repair_rate_per_h) or repair_rate_per_h < 0.0:
        raise ValueError(f"repair_rate_per_h must be >= 0, got {repair_rate_per_h!r}")
    if repair_rate_per_h == 0.0:
        return math.inf
    return LN2 / repair_rate_per_h


def lea_catcheside_g(schedule: DoseSchedule, *, repair_half_time_h: float) -> float:
    """Lea-Catcheside dose-protraction factor ``G`` in ``(0, 1]``.

    Multiplies the LQ quadratic term: ``-ln S = alpha D + G beta D^2``. See the
    module docstring for the definition and the closed form evaluated here.

    Parameters
    ----------
    schedule
        How the dose is distributed in time.
    repair_half_time_h
        Half-time of sublethal-damage repair [h], ``> 0``. ``math.inf`` means no
        repair, for which ``G = 1``.

    Returns
    -------
    float
        ``G = 1`` for a single acute dose (exactly), falling toward ``0`` as the
        irradiation is spread out relative to ``repair_half_time_h``.
    """
    lam = repair_rate_per_h(repair_half_time_h)
    total_dose = schedule.total_dose_Gy
    if total_dose <= 0.0:
        raise ValueError("G is undefined for a schedule that delivers no dose")

    blocks = schedule.blocks
    weights = [_block_weight(lam * block.duration_h) for block in blocks]

    terms: list[float] = [
        block.dose_Gy * block.dose_Gy * _self_overlap(lam * block.duration_h) for block in blocks
    ]
    for i, later in enumerate(blocks):
        for j in range(i):
            earlier = blocks[j]
            gap_h = later.start_h - earlier.end_h  # >= 0 by DoseSchedule's ordering
            terms.append(
                later.dose_Gy * earlier.dose_Gy * weights[i] * weights[j] * math.exp(-lam * gap_h)
            )

    return 2.0 * math.fsum(terms) / (total_dose * total_dose)
