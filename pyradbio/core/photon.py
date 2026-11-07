"""Photon reference parameters and shouldered->exponential survival.

The LEM family needs a photon dose-response with a transition dose ``D_t``
above which the curve becomes exponential with slope ``s_max = alpha + 2*beta*D_t``
(Scholz et al. 1997 eq. 1; Elsasser & Scholz 2007 eq. 7).
"""

from __future__ import annotations

import math
from dataclasses import dataclass


@dataclass(frozen=True, slots=True)
class ClusterParameters:
    """LEM II SSB->DSB clustering inputs (Elsasser & Scholz 2007).

    ``genome_size_bp`` is the one input Elsasser & Scholz 2007 does *not* state.
    It says only "by randomly choosing the damaged base [...] only the genomic
    size is required" (p. 321) and never gives a number. The 6.0e9 bp default is
    the same group's later convention -- Friedrich, Durante & Scholz 2012 builds
    its loop count from ``S_G = 6e9 bp`` (see
    ``docs/references/LEM_IV_Friedrich2012.md``) -- and is the round diploid
    mammalian value. Friedrich 2015 instead uses 5.4e9.

    It does **not** set the size of the enhancement. Rearranging eq. (9),
    ``eta - 1 = (a_SSB/a_DSB) * q(mu)/mu`` with ``mu = a_SSB*D*h/L_gen``, so both
    ``L_gen`` and ``h`` cancel from the *peak height*: that is fixed by the
    coincidence rule and the SSB/DSB yield ratio alone. What ``L_gen`` does set
    is where ``eta`` rises on the dose axis, exactly proportionally, and since a
    track's local-dose spectrum is fixed that lands on ion RBE with real force:
    +-10% in ``L_gen`` moves the LEM II / LEM I ratio by ~2.4% in the opposite
    direction (measured on Figs. 6/8/11, see ``docs/references/``).
    """

    ssb_per_gy_cell: float = 1250.0  # SSB/cell/Gy
    dsb_per_gy_cell: float = 30.0  # DSB/cell/Gy
    stagger_bp: float = 25.0  # max SSB separation forming a DSB [base pairs]
    genome_size_bp: float = 6.0e9  # diploid mammalian genome; NOT stated by Elsasser 2007


@dataclass(frozen=True, slots=True)
class PhotonParameters:
    """Photon LQ reference for a cell line / tissue endpoint."""

    alpha_x: float  # Gy^-1
    beta_x: float  # Gy^-2
    D_t: float | None = None  # transition dose [Gy]; required by LEM
    cluster: ClusterParameters | None = None  # LEM II/IV

    def __post_init__(self) -> None:
        if self.alpha_x < 0.0 or self.beta_x < 0.0:
            raise ValueError("alpha_x, beta_x must be >= 0")
        if self.D_t is not None and self.D_t <= 0.0:
            raise ValueError("D_t must be > 0 when given")

    def max_slope(self) -> float:
        """Slope of the exponential tail ``s_max = alpha_x + 2*beta_x*D_t``.

        Scholz et al. 1997 eq. 1 / Elsasser & Scholz 2007 eq. 7. Requires D_t.
        """
        if self.D_t is None:
            raise ValueError("max_slope requires a transition dose D_t")
        return self.alpha_x + 2.0 * self.beta_x * self.D_t

    def log_kill(self, dose_Gy: float) -> float:
        """``-ln S_x(D)``: LQ below D_t, linear (exponential survival) above."""
        if dose_Gy < 0.0:
            raise ValueError("dose must be >= 0")
        if self.D_t is None or dose_Gy <= self.D_t:
            return self.alpha_x * dose_Gy + self.beta_x * dose_Gy * dose_Gy
        d_t = self.D_t
        lk_t = self.alpha_x * d_t + self.beta_x * d_t * d_t
        return lk_t + self.max_slope() * (dose_Gy - d_t)

    def reference_survival(self, dose_Gy: float) -> float:
        """S_x(D): LQ below D_t, exponential tail above (Eq. 7, LEM II)."""
        return math.exp(-self.log_kill(dose_Gy))

    def dose_for_survival(self, survival_fraction: float) -> float:
        """Invert the shouldered->exponential reference curve for ``D(S)``.

        Below ``D_t`` this is the plain LQ inversion. Above it the curve is the
        straight exponential tail, so inverting the LQ branch there would return
        a dose that is too small -- silently inflating any iso-survival RBE that
        uses this as its reference. ``D_t = None`` keeps the pure LQ behaviour.
        """
        if not 0.0 < survival_fraction <= 1.0:
            raise ValueError(f"survival fraction must be in (0, 1], got {survival_fraction!r}")
        return self.dose_for_log_kill(-math.log(survival_fraction))

    def dose_for_log_kill(self, log_kill: float) -> float:
        """The photon dose whose log-kill ``-ln S_x`` equals ``log_kill`` [Gy].

        The inverse of :meth:`log_kill`, and the form iso-effect RBE needs at
        high dose: going through ``S = exp(-log_kill)`` underflows to zero
        above a few hundred units of log-kill, which this never forms.
        """
        if not math.isfinite(log_kill) or log_kill < 0.0:
            raise ValueError(f"log_kill must be finite and >= 0, got {log_kill!r}")
        target = log_kill
        if self.D_t is not None and target > self.log_kill(self.D_t):
            return self.D_t + (target - self.log_kill(self.D_t)) / self.max_slope()
        denom = self.alpha_x + math.sqrt(self.alpha_x**2 + 4.0 * self.beta_x * target)
        if denom == 0.0:
            raise ValueError("alpha_x and beta_x cannot both be zero")
        return 2.0 * target / denom
