"""IMK -- the integrated microdosimetric-kinetic model (Matsuya et al. 2025).

Matsuya Y, Saga R, Wang Y & Sato T, *Med. Phys.* **52**(10):e70040 (2025),
doi:10.1002/mp.70040 (open access). Equations, both parameter tables and the
sourcing caveats: ``docs/references/IMK_Matsuya2025.md``.

The acronym is **integrated** MK -- coined in Matsuya et al. 2018, *Sci. Rep.*
**8**:4849 -- and the lineage is Matsuya & Date's, unrelated to the Sato-NIRS
line that produced the SMK in [`spectrum_models`][].

The idea in one paragraph
-------------------------
IMK is the MK model with an explicit **repair channel**. Potentially lethal
lesions (PLLs, read as DSBs) either become lethal lesions (LLs) at rates ``a``
(singly) and ``b_d`` (pairwise) or are repaired at rate ``c``; solving those
rate equations gives eq. (1):

```text
-ln S = <w> = (alpha0 + z*_1D beta0) D + F beta0 D^2,  z*_1D = y*/(rho pi r_d^2)
```

The linear term is the **modified MKM verbatim**, so this class subclasses
[`ModifiedMKM`][] and overrides only
``beta``. The factor ``F`` is the Lea-Catcheside time factor, which the paper
writes in terms of the summed rate constant ``(a+c)`` rather than the repair
half-time [`protraction`][pyradbio.core.protraction] uses; ``F`` for a single continuous
block is the paper's printed ``2[x + e^-x - 1]/x^2`` with ``x = (a+c) T``.

This paper's own new content is the **micronucleus** channel: an LL forms an MN
with probability ``h``, so ``F_MN = h <w>`` (eq. 3) -- the same LQ curve scaled
by ``h`` -- and consequently ``RBE_MN = alpha*/alpha(photon)`` (eq. 6) with
``h`` cancelling out.

Timing is an argument, not a field
----------------------------------
``schedule`` is passed per call, exactly as ``pO2_mmHg`` is in
[`osmk`][] and ``g`` is in [`lq`][pyradbio.core.lq]: the
irradiation is *environment*, not a property of the cell. The cell property is
the repair rate ``(a+c)``, which **is** a field here.

The photon reference
--------------------
``alpha(photon)`` in this model is eq. (1) evaluated at the photon quality, so
it needs ``z*_1D`` of the reference radiation -- which the paper obtains from
PHITS and never prints. It is, however, *recoverable*: its fig. 2 X-ray curve is
plain LQ, so inverting eq. (1) on it returns the value that drew it. The
[`RECOVERED_PHOTON_Z_STAR_1D_GY`][] values come from that inversion
([`pyradbio.benchmark.matsuya2025`][]), and **the HSG and SCC published sets
carry them**, so those two are ready for RBE as constructed.

A model built by hand still **requires** ``reference_z_star_1D_Gy`` and raises
without it, as do the WI-38 and DU145 sets, which have no fig.-2 panel to
recover from. That is deliberate: the alternative -- falling back to
``alpha_ref`` -- would silently substitute a *foreign* photon fit, and for HSG
the obvious candidate is out by a factor 1.6 (this paper's X-ray line gives
``alpha = 0.199``; Furusawa's published HSG 200 kVp is 0.313). It is the mixing
trap already recorded for this family in ``OSMK_Inaniwa2021_2023.md``. A photon
``d(y)`` can also be scored directly through
[`IMKModel.reference_from_dose_spectrum`][].
"""

from __future__ import annotations

import math
from dataclasses import dataclass
from typing import Any, overload

import numpy as np
import numpy.typing as npt

from pyradbio.core.lq import LQParameters, log_kill
from pyradbio.core.lq import dose_for_survival as _lq_dose_for_survival
from pyradbio.core.mixing import DEFAULT_SUB_NODES, Spacing
from pyradbio.core.photon import PhotonParameters
from pyradbio.core.protraction import (
    DoseSchedule,
    lea_catcheside_g,
    repair_half_time_h_from_rate,
)
from pyradbio.core.radiation import Ion, RadiationField
from pyradbio.models.base import PhotonReferenceError, apply_preset, lq_survival_curve
from pyradbio.models.mkm.spectrum_models import ModifiedMKM

Y0_KEV_UM: float = 150.0
"""Saturation parameter of the domain, sec. 4.1 prose ("*The rd and y0 values
for the HSG cells were 0.42 um and 150 keV/um*"). Fixed, not fitted, and the
same for every cell line in the paper.
"""

_H_MATSUYA2025_TABLE2: dict[str, float] = {
    "CHO": 0.365,
    "HSG": 0.245,
    "U2OS": 0.261,
    "MCF-12A": 0.229,
    "M5": 0.196,
    "V79": 0.197,
    "SCC VII": 0.281,
}
"""Table 2 -- ``h`` fitted per cell line, with the radiation qualities behind
each fit. The paper's *predictions* use the cell-line-independent table-1
value (0.275) instead; these are offered as evidence that ``h`` *can* be
cell-line dependent. Do not mix a table-2 ``h`` into a table-1 parameter set.
"""


RECOVERED_PHOTON_Z_STAR_1D_GY: dict[str, float] = {"HSG": 1.045, "SCC": 0.987}
"""Photon ``z*_1D`` [Gy] recovered from the fig. 2 X-ray curves by inverting
eq. (1) -- see [`pyradbio.benchmark.matsuya2025`][], which owns the
digitization and the checks that make these trustworthy. **Not printed in the
paper**; recovered, and equal to ``y* = 3.62`` (HSG) / ``3.41`` (SCC) keV/um.
Duplicated here rather than imported so that ``models`` keeps no dependency on
``benchmark``; a test pins the two against each other.
"""


def matsuya2025_h_values() -> dict[str, float]:
    """Table 2: the per-cell-line micronucleus probability ``h``."""
    return dict(_H_MATSUYA2025_TABLE2)


def _z0_Gy_for_y0(y0_keV_um: float, *, domain_radius_um: float, density_g_cm3: float) -> float:
    """Convert a fixed ``y0`` [keV/um] into the ``z0_Gy`` field the family takes.

    `y0_keV_um`
    returns ``z0_Gy * rho pi r_d^2``; this is that relation inverted, so the
    published constructors can state the paper's ``y0`` directly instead of the
    geometric default derived from ``R_n`` and ``beta``.
    """
    from pyradbio.core.constants import KEVUM_PER_GY_UM2_GCM3

    scale = KEVUM_PER_GY_UM2_GCM3 * density_g_cm3 * math.pi * domain_radius_um**2
    return y0_keV_um / scale


def _table1(
    *,
    alpha0: float,
    beta0: float,
    rate_per_h: float,
    domain_radius_um: float,
    photon_z_star_1D_Gy: float | None = None,
) -> IMKModel:
    """One column of table 1, as constructor keywords.

    ``photon_z_star_1D_Gy`` is *not* from table 1 -- it is recovered from the
    fig. 2 X-ray curve, and only HSG and SCC have one.
    """
    return IMKModel(
        alpha0_Gy_inv=alpha0,
        beta0_Gy2=beta0,
        repair_rate_per_h=rate_per_h,
        domain_radius_um=domain_radius_um,
        mn_probability=0.275,
        z0_Gy=_z0_Gy_for_y0(Y0_KEV_UM, domain_radius_um=domain_radius_um, density_g_cm3=1.0),
        reference_z_star_1D_Gy=photon_z_star_1D_Gy,
    )


@dataclass(frozen=True, slots=True)
class IMKModel(ModifiedMKM):
    """Integrated MK model -- mMKM's ``alpha`` plus a repair-driven ``beta``.

    Parameters beyond [`ModifiedMKM`][]:

    ``repair_rate_per_h``
        ``(a+c)`` [h^-1], the summed PLL->LL and PLL-repair rate constants of
        eq. (1). ``0.0`` means no sublethal repair, i.e. ``F = 1`` at any
        delivery time. Table 1 gives 2.19 h^-1 for HSG and SCC, which is a
        repair half-time of 19 min.
    ``mn_probability``
        ``h``, the probability that an LL forms a micronucleus (eq. 3).
    ``reference_z_star_1D_Gy``
        ``z*_1D`` of the photon reference. Required by every RBE route; see the
        module docstring for why there is no default.
    """

    repair_rate_per_h: float = 0.0
    mn_probability: float = 0.275
    reference_z_star_1D_Gy: float | None = None

    def __post_init__(self) -> None:
        ModifiedMKM.__post_init__(self)
        if not math.isfinite(self.repair_rate_per_h) or self.repair_rate_per_h < 0.0:
            raise ValueError(
                f"repair_rate_per_h (a+c) must be finite and >= 0, got {self.repair_rate_per_h!r}"
            )
        if not math.isfinite(self.mn_probability) or not 0.0 <= self.mn_probability <= 1.0:
            raise ValueError(
                f"mn_probability (h) must be a probability in [0, 1], got {self.mn_probability!r}"
            )

    # ---- published parameter sets (table 1) ----------------------------
    @classmethod
    def hsg_matsuya2025(cls, **overrides: Any) -> IMKModel:
        """HSG cells, table 1: ``alpha0 = 0.150``, ``beta0 = 0.0467``, ``a+c = 2.19``.

        Carries the fig.-2a-recovered photon reference, so RBE works as built.
        """
        return apply_preset(cls, _HSG_MATSUYA2025, overrides)

    @classmethod
    def scc_matsuya2025(cls, **overrides: Any) -> IMKModel:
        """SCC VII cells, table 1: ``alpha0 = 0.0825``, ``beta0 = 0.0298``.

        Carries the fig.-2b-recovered photon reference, so RBE works as built.
        """
        return apply_preset(cls, _SCC_MATSUYA2025, overrides)

    @classmethod
    def wi38_matsuya2025(cls, **overrides: Any) -> IMKModel:
        """WI-38 fibroblasts, table 1. Fitted to 150 kVp X-rays only (sec. 3.1.1).

        No fig.-2 panel, so no photon reference: RBE needs one passing in.
        """
        return apply_preset(cls, _WI38_MATSUYA2025, overrides)

    @classmethod
    def du145_matsuya2025(cls, **overrides: Any) -> IMKModel:
        """DU145 prostate carcinoma, table 1. Fitted to 150 kVp X-rays only.

        No fig.-2 panel, so no photon reference: RBE needs one passing in.
        """
        return apply_preset(cls, _DU145_MATSUYA2025, overrides)

    # ---- eq. (1) -------------------------------------------------------
    def time_factor(self, schedule: DoseSchedule | None = None) -> float:
        """Lea-Catcheside factor ``F`` of eq. (1) for ``schedule`` (``None`` = acute).

        For one continuous block of duration ``T`` this is the paper's printed
        ``F = 2[x + e^-x - 1]/x^2`` with ``x = (a+c) T``; the general
        block route in [`protraction`][pyradbio.core.protraction] is used so that
        fractionated and split-course deliveries -- including the paper's own
        dose-rate protocols (sec. 3.1.4) -- are covered by the same call.
        """
        if schedule is None:
            return 1.0
        return lea_catcheside_g(
            schedule, repair_half_time_h=repair_half_time_h_from_rate(self.repair_rate_per_h)
        )

    def alpha_beta_from_z_star_1D(
        self, z_star_1D_Gy: float, *, schedule: DoseSchedule | None = None
    ) -> LQParameters:
        """Eq. (1): ``alpha = alpha0 + beta0 z*_1D``, ``beta = F beta0``."""
        acute = ModifiedMKM.alpha_beta_from_z_star_1D(self, z_star_1D_Gy)
        return LQParameters(alpha=acute.alpha, beta=self.time_factor(schedule) * self.beta0)

    def alpha_beta_from_dose_spectrum(
        self,
        y_keV_um: npt.NDArray[np.float64],
        d_of_y: npt.NDArray[np.float64],
        *,
        schedule: DoseSchedule | None = None,
    ) -> LQParameters:
        """Eq. (1) from a ``(y, d(y))`` pair, with ``y*`` from eq. (2)."""
        z_star = self.y_star_D(y_keV_um, d_of_y) / self._z_domain_scale_keV_um_per_Gy
        return self.alpha_beta_from_z_star_1D(z_star, schedule=schedule)

    # ---- the BiophysicalModel interface, plus a keyword-only schedule ----
    def alpha_beta(
        self,
        field: RadiationField | Ion,
        *,
        schedule: DoseSchedule | None = None,
        sub_nodes: int = DEFAULT_SUB_NODES,
        spacing: Spacing = "log",
    ) -> LQParameters:
        """Eq. (1): ``alpha = alpha0 + beta0 z*_1D``, ``beta = F beta0`` for ``schedule``.

        ``schedule=None`` is an acute dose (``F = 1``); ``z*_1D`` is scored in
        ``site_geometry``. Only ``beta`` depends on the schedule.
        """
        acute = ModifiedMKM.alpha_beta(self, field, sub_nodes=sub_nodes, spacing=spacing)
        if schedule is None:
            return acute
        return LQParameters(alpha=acute.alpha, beta=self.time_factor(schedule) * self.beta0)

    @overload
    def survival(
        self,
        field: Ion | RadiationField,
        dose_Gy: float,
        *,
        schedule: DoseSchedule | None = ...,
    ) -> float: ...

    @overload
    def survival(
        self,
        field: Ion | RadiationField,
        dose_Gy: npt.NDArray[np.float64],
        *,
        schedule: DoseSchedule | None = ...,
    ) -> npt.NDArray[np.float64]: ...

    def survival(
        self,
        field: Ion | RadiationField,
        dose_Gy: float | npt.NDArray[np.float64],
        *,
        schedule: DoseSchedule | None = None,
    ) -> float | npt.NDArray[np.float64]:
        """``S(D) = exp(-<w>)`` -- eq. (1), with ``schedule`` setting ``F`` in ``beta``."""
        return lq_survival_curve(dose_Gy, self.alpha_beta(field, schedule=schedule))

    def dose_for_survival(
        self,
        field: Ion | RadiationField,
        survival_fraction: float,
        *,
        schedule: DoseSchedule | None = None,
    ) -> float:
        """Dose reaching ``survival_fraction`` under ``schedule`` [Gy]."""
        lq = self.alpha_beta(field, schedule=schedule)
        return _lq_dose_for_survival(survival_fraction, lq.alpha, lq.beta)

    def rbe_at_survival(
        self,
        field: Ion | RadiationField,
        survival_fraction: float,
        *,
        schedule: DoseSchedule | None = None,
    ) -> float:
        """Eq. (5): ``RBE_SF = D_SF(photon) / D_SF(ion)``, generalised to any ``S``.

        The photon arm is [`reference`][], acute. The paper scores eq. (5)
        under the acute assumption ``beta ~= beta0`` (sec. 2.4); passing a
        ``schedule`` protracts the ion arm only -- a defensible generalisation,
        but not what the paper reports.
        """
        lq = self.alpha_beta(field, schedule=schedule)
        return self._rbe_at_survival_from_lq(lq, survival_fraction)

    def rbe_at_dose(
        self,
        field: Ion | RadiationField,
        dose_Gy: float,
        *,
        schedule: DoseSchedule | None = None,
    ) -> float:
        """Iso-effect RBE at an ion dose under ``schedule`` (acute photon reference)."""
        if dose_Gy <= 0.0:
            raise ValueError(f"dose must be > 0 Gy, got {dose_Gy!r}")
        lq = self.alpha_beta(field, schedule=schedule)
        effect = log_kill(dose_Gy, lq.alpha, lq.beta)
        return self._reference_dose_for_log_kill(effect) / dose_Gy

    # ---- eq. (3): micronuclei ------------------------------------------
    def micronucleus_lq(self, lq: LQParameters) -> LQParameters:
        """Eq. (3): ``alpha_m = h alpha``, ``beta_m = h beta``.

        The MN dose-response is the survival LQ scaled by ``h`` -- which is why
        ``RBE_MN`` (eq. 6) cannot depend on ``h``.
        """
        return LQParameters(
            alpha=self.mn_probability * lq.alpha, beta=self.mn_probability * lq.beta
        )

    def micronucleus_frequency(self, dose_Gy: float, lq: LQParameters) -> float:
        """Eq. (3): ``F_MN = h <w>`` [micronuclei per binucleated cell].

        ``<w> = -ln S`` is the mean number of lethal lesions per nucleus (eq. 1).
        """
        return self.mn_probability * log_kill(dose_Gy, lq.alpha, lq.beta)

    # ---- eqs. (4)-(6): RBE ---------------------------------------------
    def reference(self) -> PhotonParameters:
        """The photon arm: eq. (1) at ``reference_z_star_1D_Gy``, acutely.

        Raises rather than falling back to ``alpha_ref``/``beta_ref``: in IMK
        the photon reference is the *model's own* prediction at the photon
        quality, and substituting a foreign LQ fit silently changes every RBE.
        """
        if self.reference_z_star_1D_Gy is None:
            raise PhotonReferenceError(
                "IMK's photon reference is eq. (1) at the photon quality, so it needs "
                "reference_z_star_1D_Gy (the paper scores it with PHITS and never prints "
                "it); pass it, or build it with reference_from_dose_spectrum()"
            )
        return PhotonParameters(
            alpha_x=self.alpha0 + self.beta0 * self.reference_z_star_1D_Gy, beta_x=self.beta0
        )

    def reference_from_dose_spectrum(
        self,
        y_keV_um: npt.NDArray[np.float64],
        d_of_y: npt.NDArray[np.float64],
    ) -> LQParameters:
        """The photon reference from a photon ``d(y)``, scored by this same model."""
        return self.alpha_beta_from_dose_spectrum(y_keV_um, d_of_y)

    def d10(self, lq: LQParameters) -> float:
        """Eq. (4): the dose giving 10% survival [Gy].

        Algebraically the paper's ``D10 = [sqrt(alpha^2 + 4 beta S*) - alpha]/(2 beta)``;
        evaluated through [`dose_for_survival`][pyradbio.core.lq.dose_for_survival], whose
        rationalised root is stable as ``beta -> 0``.
        """
        return _lq_dose_for_survival(0.1, lq.alpha, lq.beta)

    def rbe_mn(self, lq: LQParameters) -> float:
        """Eq. (6): ``RBE_MN = alpha_m*/alpha_m(photon) = alpha*/alpha(photon)``.

        ``h`` cancels, so this needs no micronucleus parameter -- the paper's
        central claim is that this equals the RBE for cell survival. It is a
        ratio of ``alpha`` alone, i.e. ``RBE_max``, the ``D -> 0`` limit, and
        not comparable with [`rbe_at_survival`][] at 10% survival.
        """
        return lq.alpha / self.reference().alpha_x


# Matsuya et al. 2025 table 1. Uncertainties (MCMC, 1 sigma) are in the digest;
# note DU145's alpha0 and WI-38's beta0 carry ~50-80% relative uncertainty.
_HSG_MATSUYA2025 = _table1(
    alpha0=0.150,
    beta0=0.0467,
    rate_per_h=2.19,
    domain_radius_um=0.42,
    photon_z_star_1D_Gy=RECOVERED_PHOTON_Z_STAR_1D_GY["HSG"],
)
_SCC_MATSUYA2025 = _table1(
    alpha0=0.0825,
    beta0=0.0298,
    rate_per_h=2.19,
    domain_radius_um=0.42,
    photon_z_star_1D_Gy=RECOVERED_PHOTON_Z_STAR_1D_GY["SCC"],
)
_WI38_MATSUYA2025 = _table1(alpha0=0.590, beta0=0.0163, rate_per_h=0.371, domain_radius_um=0.50)
_DU145_MATSUYA2025 = _table1(alpha0=0.0774, beta0=0.0380, rate_per_h=2.10, domain_radius_um=0.50)
