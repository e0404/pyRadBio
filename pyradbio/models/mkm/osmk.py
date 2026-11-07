"""OSMK -- oxygen-effect-incorporated SMK (Inaniwa & Kanematsu 2021 and 2023).

Two published formulations, both here:

* :class:`OSMK2021` -- Inaniwa, Kanematsu, Shinoto, Koto & Yamada,
  *Phys. Med. Biol.* **66** (2021) 205007, doi:10.1088/1361-6560/ac29cc.
* :class:`OSMK2023` -- Inaniwa & Kanematsu, *J. Radiat. Res.* **64** (2023)
  685-692, doi:10.1093/jrr/rrad049, the "event-by-event" revision.

Equations, both parameter tables and the sourcing caveats:
``docs/references/OSMK_Inaniwa2021_2023.md``.

The idea in one paragraph
-------------------------
Hypoxic cells resist radiation because some of the initial DNA damage is
chemically repaired (within ~1 us) instead of being fixed by molecular oxygen.
OSMK expresses that as a **relative radioresistance** ``R(pO2) >= 1``
(:func:`~pyradbio.core.oxygen.relative_radioresistance`) which divides the
*sub-lethal* lesion production rate, ``k_d -> k_d/R``, and leaves the *lethal*
rate ``lambda_d`` alone -- clustered damage cannot be chemically repaired. Since
SMK's ``alpha_S ∝ k_d`` and ``beta_0 ∝ k_d^2``, that single substitution is the
whole model::

    alpha'_SMK(pO2) = alpha_L + alpha_S R^-1 + z*_d,D beta_0 R^-2       (17)/(11)
    beta'_SMK(pO2)  = beta_0 R^-2 (z*_d,D / z_d,D)                      (18)/(12)

and survival is *unchanged* SMK eq. (24) with the primed pair substituted in
(2021 eq. 16 = 2023 eq. 10). ``z_n,D`` is not touched by oxygen in either paper.
So :class:`~pyradbio.models.mkm.spectrum_models.StochasticMKM` supplies
everything except ``R``, and these classes are thin subclasses of it.

Where the two papers differ
---------------------------
Only in how the measured *fall of the OER with LET* is produced:

* **2021** makes ``Rmax`` a function of radiation quality, ``Rmax(z_d,D)``
  (eq. 14), decreasing from ``R_m`` towards 1 as the track gets denser. The
  microdosimetry is untouched by ``pO2``.
* **2023** makes ``Rmax`` a plain cell constant and instead shrinks the *site*:
  ``r_d' = r_d/f_rd(pO2)`` and ``z_0' = z_0 f_z0(pO2)`` (eqs. 6-9), then rescores
  ``z_d,D'`` and ``z*_d,D'`` at that geometry. This is the "event-by-event"
  content -- and it makes the domain ~3x smaller under anoxia, so a
  :class:`~pyradbio.micro.lineal_energy.SpectrumEngine` bound to ``r_d`` is
  **refused** rather than silently reused.

The 2023 paper's own conclusion is that the difference is marginal for
therapeutic beams (<= 1.9 %pt over an SOBP target), so neither supersedes the
other; both are offered.

Parameters
----------
Published fits: :meth:`OSMK2021.hsg_inaniwa2021` and siblings (HSG / V79 / CHO in
both papers). **The three MKM-family fits of HSG are independent and must not be
mixed**: ``z0`` is 66.0 Gy (SMK 2018) / 88.0 (OSMK 2021) / 60.0 (OSMK 2023), and
``r_d`` is 0.28 / 0.23 / 0.29 um. Constructing an OSMK from SMK parameters plus
oxygen parameters is wrong in every direction.

Note also that the experimental "aerobic" condition of 160 mmHg is **not** the
``R = 1`` limit: ``R(160) ~ 1.014``. ``pO2_mmHg`` defaults to 160 (the papers'
own reference); pass ``math.inf`` for the strict SMK limit.
"""

from __future__ import annotations

import math
from dataclasses import dataclass, fields, replace
from typing import Any, overload

import numpy as np
import numpy.typing as npt

from pyradbio.core.lq import LQParameters
from pyradbio.core.mixing import DEFAULT_SUB_NODES, Spacing
from pyradbio.core.oxygen import (
    AEROBIC_PO2_MMHG,
    K_HALF_EFFECT_MMHG,
    relative_radioresistance,
)
from pyradbio.core.radiation import Ion, RadiationField
from pyradbio.models.base import apply_preset
from pyradbio.models.mkm.spectrum_models import SiteSpecificEnergies, StochasticMKM


@dataclass(frozen=True, slots=True)
class _OSMKModel(StochasticMKM):
    """Shared OSMK machinery: the ``alpha'``/``beta'`` algebra and the OER.

    Subclasses supply :meth:`max_radioresistance` (how ``Rmax`` depends -- or does
    not -- on radiation quality) and, for the 2023 formulation, the ``pO2``
    -dependent site geometry via :meth:`specific_energies_at`.
    """

    alpha_L_Gy_inv: float = 0.0
    alpha_S_Gy_inv: float = 0.0
    k_mmHg: float = K_HALF_EFFECT_MMHG

    def __post_init__(self) -> None:
        """Refuse the SMK treatment axes: neither is derived for this model.

        OSMK replaces SMK's ``alpha`` outright (its ``alpha0`` splits into the
        oxygen-independent ``alpha_L`` and the ``1/R``-carrying ``alpha_S``), so
        Sato's eq.-(33) frequency factor has no published place in it; and the
        oxygen-corrected survival is written against the eq.-(24) expansion.
        Inheriting the fields silently would make both no-ops here -- the exact
        defect the SMK entry-point tests exist to prevent -- so they are refused.
        """
        StochasticMKM.__post_init__(self)
        # ``slots=True`` turns the class attributes into descriptors, so the
        # declared defaults have to come from the dataclass fields.
        defaults = {f.name: f.default for f in fields(StochasticMKM)}
        for axis in ("domain_treatment", "nucleus_treatment"):
            value = getattr(self, axis)
            if value != defaults[axis]:
                raise ValueError(
                    f"OSMK does not implement {axis}={value!r}: it carries its own "
                    "published alpha' / beta' form, and neither SMK treatment axis "
                    "has been derived for it. Use StochasticMKM for those axes."
                )

    # ---- the alpha0 split ----------------------------------------------
    @property
    def alpha0(self) -> float:
        """``alpha_L + alpha_S`` -- SMK's ``alpha0``, split into its two channels.

        ``alpha_L ≡ p lambda_d`` is the *lethal* channel and is oxygen-independent;
        ``alpha_S ≡ p k_d G_alpha`` is the *sub-lethal* one and carries the ``1/R``.
        Both papers fit ``alpha_L = 0`` for HSG and V79 (0.01 for CHO in 2021),
        which they read as confirming Hawkins 2009's ``lambda_d << k_d``.
        """
        return self.alpha_L_Gy_inv + self.alpha_S_Gy_inv

    # ---- R(pO2) --------------------------------------------------------
    def max_radioresistance(self, z_1D_Gy: float) -> float:
        """``Rmax`` for a beam of dose-mean domain specific energy ``z_1D_Gy``."""
        raise NotImplementedError  # pragma: no cover -- implemented by subclasses

    def relative_radioresistance(self, pO2_mmHg: float, *, z_1D_Gy: float) -> float:
        """``R(pO2) = (pO2 + K)/(pO2 + K/Rmax)`` (2021 eq. 13 = 2023 eq. 1).

        ``z_1D_Gy`` is used only by :class:`OSMK2021`, where ``Rmax`` depends on
        radiation quality; :class:`OSMK2023` ignores it.
        """
        return float(
            relative_radioresistance(
                pO2_mmHg, k_mmHg=self.k_mmHg, r_max=self.max_radioresistance(z_1D_Gy)
            )
        )

    # ---- eqs. (17)-(18) / (11)-(12) ------------------------------------
    def alpha_beta_from_specific_energies_at(
        self,
        *,
        pO2_mmHg: float,
        z_star_1D_Gy: float,
        z_1D_Gy: float,
    ) -> LQParameters:
        """The oxygen-corrected LQ pair from the microdosimetric quantities.

        ``alpha' = alpha_L + alpha_S/R + z*_d,D beta_0/R^2`` and
        ``beta' = (beta_0/R^2)(z*_d,D/z_d,D)``. For :class:`OSMK2023` the two
        specific energies must already be the ones scored at the *hypoxic*
        geometry (:meth:`specific_energies_at` does that).

        This is the seam an external engine (pyMKM's ``z_bar_star_domain`` /
        ``z_bar_domain``) can drive directly.
        """
        b0 = self.beta0
        if z_1D_Gy <= 0.0:
            # No energy in the domain -> no quality dependence, and no oxygen
            # dependence either: R multiplies quantities that are all zero.
            return LQParameters(alpha=max(self.alpha0, 0.0), beta=b0)
        r = self.relative_radioresistance(pO2_mmHg, z_1D_Gy=z_1D_Gy)
        alpha = self.alpha_L_Gy_inv + self.alpha_S_Gy_inv / r + z_star_1D_Gy * b0 / (r * r)
        beta = b0 / (r * r) * (z_star_1D_Gy / z_1D_Gy)
        return LQParameters(alpha=max(alpha, 0.0), beta=max(beta, 0.0))

    # ---- ion -> specific energies, at a given pO2 -----------------------
    def specific_energies_at(self, ion: Ion, pO2_mmHg: float) -> SiteSpecificEnergies:
        """Dose-mean specific energies for ``ion`` under ``pO2``.

        The 2021 formulation ignores ``pO2`` here (its microdosimetry is
        oxygen-independent); :class:`OSMK2023` overrides this to rescore at the
        hypoxic domain radius and saturation parameter.
        """
        _validate_pO2(pO2_mmHg)
        return self.specific_energies(ion)

    # ---- the public API: BiophysicalModel's, plus a keyword-only pO2 -----
    def alpha_beta(
        self,
        field: RadiationField | Ion,
        *,
        pO2_mmHg: float = AEROBIC_PO2_MMHG,
        sub_nodes: int = DEFAULT_SUB_NODES,
        spacing: Spacing = "log",
    ) -> LQParameters:
        """Oxygen-corrected ``(alpha, beta)`` for one ion at ``pO2``.

        ``pO2_mmHg`` defaults to the papers' experimental aerobic reference,
        160 mmHg -- where ``R`` is ~1.4% *above* unity, not equal to it. Pass
        ``math.inf`` for the strict ``R = 1`` limit, which reproduces
        :class:`~pyradbio.models.mkm.spectrum_models.StochasticMKM` exactly.
        Mixed fields are refused, as for the whole spectrum-MKM family.
        """
        lq, _z_n = self._lq_and_z_nucleus(self._require_ion(field), pO2_mmHg)
        return lq

    def _alpha_beta_ion(self, ion: Ion) -> LQParameters:
        """The aerobic (160 mmHg) pair -- what the un-keyworded interface returns."""
        return self.alpha_beta(ion)

    @overload
    def survival(
        self, field: Ion | RadiationField, dose_Gy: float, *, pO2_mmHg: float = ...
    ) -> float: ...

    @overload
    def survival(
        self,
        field: Ion | RadiationField,
        dose_Gy: npt.NDArray[np.float64],
        *,
        pO2_mmHg: float = ...,
    ) -> npt.NDArray[np.float64]: ...

    def survival(
        self,
        field: Ion | RadiationField,
        dose_Gy: float | npt.NDArray[np.float64],
        *,
        pO2_mmHg: float = AEROBIC_PO2_MMHG,
    ) -> float | npt.NDArray[np.float64]:
        """SMK eq. (24) with the oxygen-corrected pair (2021 eq. 16 = 2023 eq. 10)."""
        lq, z_n = self._lq_and_z_nucleus(self._require_ion(field), pO2_mmHg)
        curve = self._survival(dose_Gy, lq, z_n)
        return float(curve) if np.ndim(dose_Gy) == 0 else curve

    def _lq_and_z_nucleus(self, ion: Ion, pO2_mmHg: float) -> tuple[LQParameters, float]:
        z = self.specific_energies_at(ion, pO2_mmHg)
        lq = self.alpha_beta_from_specific_energies_at(
            pO2_mmHg=pO2_mmHg, z_star_1D_Gy=z.z_star_1D_Gy, z_1D_Gy=z.z_1D_Gy
        )
        return lq, z.z_nucleus_Gy

    def dose_for_survival(
        self,
        field: Ion | RadiationField,
        survival_fraction: float,
        *,
        pO2_mmHg: float = AEROBIC_PO2_MMHG,
    ) -> float:
        """Dose reaching ``survival_fraction`` under ``pO2`` [Gy] (eq. 24 inverted)."""
        lq, z_n = self._lq_and_z_nucleus(self._require_ion(field), pO2_mmHg)
        return self._dose_for_survival_smk(survival_fraction, lq, z_n)

    def _log_kill(
        self,
        field: Ion | RadiationField,
        dose_Gy: float,
        *,
        pO2_mmHg: float = AEROBIC_PO2_MMHG,
    ) -> float:
        lq, z_n = self._lq_and_z_nucleus(self._require_ion(field), pO2_mmHg)
        return self._taylor_log_kill(dose_Gy, lq, z_n)

    def oer_at_survival(
        self,
        ion: Ion,
        pO2_mmHg: float,
        survival_fraction: float = 0.1,
        reference_pO2_mmHg: float = AEROBIC_PO2_MMHG,
    ) -> float:
        """Oxygen enhancement ratio ``D(pO2) / D(reference pO2)`` at iso-survival.

        The papers benchmark ``OER`` at 10% survival against the aerobic
        (160 mmHg) arm -- 2021 sec 4.2, figs. 5-7 -- which is the default here.
        Expect ~3 at low LET falling towards unity above ~200 keV/um.
        """
        d_hypoxic = self.dose_for_survival(ion, survival_fraction, pO2_mmHg=pO2_mmHg)
        d_reference = self.dose_for_survival(ion, survival_fraction, pO2_mmHg=reference_pO2_mmHg)
        if d_reference <= 0.0:
            raise ValueError("reference dose is zero: check the reference oxygen condition")
        return d_hypoxic / d_reference

    def rbe_at_survival(
        self,
        field: Ion | RadiationField,
        survival_fraction: float,
        *,
        pO2_mmHg: float = AEROBIC_PO2_MMHG,
    ) -> float:
        """``RBE_S`` at ``pO2``, against the (aerobic, LQ) photon reference.

        Note the reference arm is *not* oxygen-corrected: ``alpha_ref``/``beta_ref``
        describe photons under the reference oxygenation, so this is an RBE at
        ``pO2``, not the combined ``RBE/OER`` isoeffective quantity of the paper's
        eq. (20) -- form that by dividing by :meth:`oer_at_survival` if wanted.
        """
        lq, z_n = self._lq_and_z_nucleus(self._require_ion(field), pO2_mmHg)
        return self._rbe_numeric(survival_fraction, lq, z_n)

    def rbe_at_dose(
        self,
        field: Ion | RadiationField,
        dose_Gy: float,
        *,
        pO2_mmHg: float = AEROBIC_PO2_MMHG,
    ) -> float:
        """Iso-effect RBE at an ion dose under ``pO2`` (aerobic photon reference)."""
        if dose_Gy <= 0.0:
            raise ValueError(f"dose must be > 0 Gy, got {dose_Gy!r}")
        effect = self._log_kill(field, dose_Gy, pO2_mmHg=pO2_mmHg)
        return self._reference_dose_for_log_kill(effect) / dose_Gy

    def survival_valid_to_Gy(self, ion: Ion, *, pO2_mmHg: float = AEROBIC_PO2_MMHG) -> float:
        """Where the eq.-(24) expansion stops being controlled, at this ``pO2``."""
        lq, z_n = self._lq_and_z_nucleus(ion, pO2_mmHg)
        return self._valid_to_Gy(lq, z_n)


def _validate_pO2(pO2_mmHg: float) -> None:
    if pO2_mmHg < 0.0 or math.isnan(pO2_mmHg):
        raise ValueError(f"pO2_mmHg must be >= 0, got {pO2_mmHg!r}")


@dataclass(frozen=True, slots=True)
class OSMK2021(_OSMKModel):
    """OSMK 2021 -- the original (Inaniwa et al., doi:10.1088/1361-6560/ac29cc).

    The LET dependence of the OER lives entirely in ``Rmax``, made a function of
    the beam's dose-mean domain specific energy (eq. 14)::

        Rmax(z_d,D) = [(z_d,D/z_R)^g + R_m] / [(z_d,D/z_R)^g + 1]

    which falls from ``R_m`` (sparse tracks -- chemical repair is effective) to 1
    (dense tracks -- clustered damage cannot be repaired, so oxygen stops
    mattering). The microdosimetry is **not** modified by ``pO2``, which makes
    this the cheaper of the two formulations: one extra scalar per beam.

    Its approximation, admitted in the paper's own discussion: ``R`` is evaluated
    from the *dose-averaged* ``z_d,D`` outside the rate equations, so a beam with a
    wide spread of per-event ``z_d`` (an SOBP) gets one representative quality.
    :class:`OSMK2023` is the fix; the 2023 paper finds it changes little.
    """

    z_R_Gy: float = 1.0
    gamma_exponent: float = 1.0
    R_m: float = 1.0

    @classmethod
    def hsg_inaniwa2021(cls, **overrides: Any) -> OSMK2021:
        """HSG cells, 2021 table 1: ``alpha_S = 0.21``, ``beta_0 = 0.043``,
        ``r_d = 0.23 um``, ``z0 = 88.0 Gy``, ``z_R = 28.0 Gy``, ``g = 1.30``,
        ``R_m = 2.9`` (``r^2 = 0.947``)."""
        return apply_preset(cls, _HSG_OSMK2021, overrides)

    @classmethod
    def v79_inaniwa2021(cls, **overrides: Any) -> OSMK2021:
        """V79 cells, 2021 table 1: ``alpha_S = 0.10``, ``beta_0 = 0.026``,
        ``r_d = 0.29 um``, ``z0 = 98.0 Gy``, ``z_R = 29.0 Gy``, ``g = 1.60``,
        ``R_m = 2.5`` (``r^2 = 0.934``)."""
        return apply_preset(cls, _V79_OSMK2021, overrides)

    @classmethod
    def cho_inaniwa2021(cls, **overrides: Any) -> OSMK2021:
        """CHO cells, 2021 table 1: the only fit with ``alpha_L != 0`` (0.01), and
        the only one calibrated against *graded* hypoxia -- Tinganelli et al. 2015
        at pO2 = 0, 1.14, 3.8, 15.2 and 160 mmHg (``r^2 = 0.988``)."""
        return apply_preset(cls, _CHO_OSMK2021, overrides)

    def max_radioresistance(self, z_1D_Gy: float) -> float:
        """Eq. (14). Bounded by ``R_m`` below and 1 above; ``= (1 + R_m)/2`` at ``z_R``."""
        if z_1D_Gy <= 0.0:
            return self.R_m
        ratio = (z_1D_Gy / self.z_R_Gy) ** self.gamma_exponent
        return float((ratio + self.R_m) / (ratio + 1.0))


@dataclass(frozen=True, slots=True)
class OSMK2023(_OSMKModel):
    """OSMK 2023 -- the event-by-event revision (doi:10.1093/jrr/rrad049).

    ``Rmax`` becomes a plain cell constant (:attr:`R_max`); the LET dependence
    moves into the *site geometry*, which shrinks as the cell becomes hypoxic::

        r_d' = r_d / f_rd(pO2)                                              (6)
        z_0' = z_0 * f_z0(pO2)                                              (7)
        f_rd =  1 + ((f_rd_max - 1)/(Rmax - 1)) (R - 1)                     (8)
        f_z0 = [1 + ((f_z0_max - 1)/(Rmax - 1)) (R - 1)]^2                  (9)

    **``f_z0`` is squared and ``f_rd`` is not** -- eq. (9) as printed. ``z_d,D``
    and ``z*_d,D`` are then rescored at ``(r_d', z_0')``, which is what makes the
    chemical repair act per energy-deposition event rather than on one
    dose-averaged quality.

    Rationale (2023 Materials & Methods): Hawkins 2001 found radiosensitisation
    enlarges the domain, so radioresistance should shrink it (following Bopp et
    al. 2016); and with fewer primary lesions the overkill saturation sets in
    later, so ``z_0`` rises.

    Consequences worth knowing
    --------------------------
    * The domain is ~3x smaller under anoxia (HSG: 0.29 -> 0.097 um). Any injected
      :class:`~pyradbio.micro.lineal_energy.SpectrumEngine` is bound to one site
      size, so hypoxic evaluation with an injected engine **raises** instead of
      silently scoring the wrong volume. The default impact-parameter route has
      no such problem.
    * ``V79`` has ``alpha_L = alpha_S = 0`` in this table, so its entire ``alpha``
      is the ``beta_0 z*_d,D R^-2`` channel. Code assuming ``alpha0 > 0`` breaks.
    * pyMKM rounds ``r_d'`` and ``z_0'`` (to 2-3 decimals) before rescoring, which
      at HSG anoxia is a 3.4% radius error. We do not; exact agreement with pyMKM
      on this path is therefore not expected. See the reference summary, gotcha 5.
    """

    f_rd_max: float = 1.0
    f_z0_max: float = 1.0
    R_max: float = 1.0

    @classmethod
    def hsg_inaniwa2023(cls, **overrides: Any) -> OSMK2023:
        """HSG cells, 2023 table 1: ``alpha_S = 0.0797``, ``beta_0 = 0.0677``,
        ``r_d = 0.29 um``, ``z0 = 60.0 Gy``, ``f_rd_max = 3.00``,
        ``f_z0_max = 3.53``, ``Rmax = 4.46`` (``r^2 = 0.949``)."""
        return apply_preset(cls, _HSG_OSMK2023, overrides)

    @classmethod
    def v79_inaniwa2023(cls, **overrides: Any) -> OSMK2023:
        """V79 cells, 2023 table 1. Note ``alpha_L = alpha_S = 0``: the fitted
        ``alpha`` is generated purely by ``beta_0 z*_d,D`` (``r^2 = 0.934``)."""
        return apply_preset(cls, _V79_OSMK2023, overrides)

    @classmethod
    def cho_inaniwa2023(cls, **overrides: Any) -> OSMK2023:
        """CHO cells, 2023 table 1: ``alpha_S = 0.132``, ``beta_0 = 0.0591``,
        ``r_d = 0.41 um``, ``z0 = 37.0 Gy``, ``f_rd_max = 2.69``,
        ``f_z0_max = 3.23``, ``Rmax = 3.51`` (``r^2 = 0.979``)."""
        return apply_preset(cls, _CHO_OSMK2023, overrides)

    def max_radioresistance(self, z_1D_Gy: float) -> float:
        """A cell constant here -- no radiation-quality dependence (2023 sec 2.2)."""
        return self.R_max

    def scaling_factors(self, pO2_mmHg: float) -> tuple[float, float]:
        """``(f_rd, f_z0)`` of eqs. (8)-(9). Both are 1 in the ``R -> 1`` limit.

        At ``pO2 = 0`` they reach ``f_rd_max`` and ``f_z0_max**2`` respectively --
        the square is on ``f_z0`` only.
        """
        _validate_pO2(pO2_mmHg)
        if self.R_max <= 1.0:  # no oxygen effect at all -> no geometry change
            return 1.0, 1.0
        r = self.relative_radioresistance(pO2_mmHg, z_1D_Gy=0.0)
        span = (r - 1.0) / (self.R_max - 1.0)
        f_rd = 1.0 + (self.f_rd_max - 1.0) * span
        f_z0 = (1.0 + (self.f_z0_max - 1.0) * span) ** 2
        return f_rd, f_z0

    def hypoxic_geometry(self, pO2_mmHg: float) -> tuple[float, float]:
        """``(r_d', z_0')`` under ``pO2`` [um, Gy] -- eqs. (6)-(7)."""
        f_rd, f_z0 = self.scaling_factors(pO2_mmHg)
        return self.domain_radius_um / f_rd, self.z0_effective_Gy * f_z0

    def specific_energies_at(self, ion: Ion, pO2_mmHg: float) -> SiteSpecificEnergies:
        """Rescore the domain at ``(r_d', z_0')``; the nucleus is unchanged.

        Raises when an engine is injected *and* the geometry actually moves: a
        :class:`~pyradbio.micro.lineal_energy.SpectrumEngine` is constructed for
        one site size, so reusing it at ``r_d'`` would score the wrong volume.
        """
        _validate_pO2(pO2_mmHg)
        r_d, z0 = self.hypoxic_geometry(pO2_mmHg)
        if r_d == self.domain_radius_um and z0 == self.z0_effective_Gy:
            return self.specific_energies(ion)
        if self.engine is not None:
            raise ValueError(
                "OSMK 2023 rescores the domain at r_d' = r_d / f_rd(pO2) "
                f"({self.domain_radius_um:.4g} -> {r_d:.4g} um at pO2 = {pO2_mmHg:g} mmHg), "
                "but the injected engine is bound to a fixed site size. Drop the "
                "engine to use the impact-parameter route, size a new engine to "
                "r_d' yourself, or use OSMK2021 (its microdosimetry is "
                "oxygen-independent)."
            )
        hypoxic = replace(self, domain_radius_um=r_d, z0_Gy=z0)
        return hypoxic.specific_energies(ion)


# ---- published parameter sets, as instances (applied with dataclasses.replace) --
# Inaniwa, Kanematsu, Shinoto, Koto & Yamada 2021 (doi:10.1088/1361-6560/ac29cc)
# table 1. r_n = 8.1 um and K = 3 mmHg are assumed for all three cell lines.
_HSG_OSMK2021 = OSMK2021(
    alpha_L_Gy_inv=0.00,
    alpha_S_Gy_inv=0.21,
    beta0_Gy2=0.043,
    domain_radius_um=0.23,
    nucleus_radius_um=8.1,
    z0_Gy=88.0,
    z_R_Gy=28.0,
    gamma_exponent=1.30,
    R_m=2.9,
)
_V79_OSMK2021 = OSMK2021(
    alpha_L_Gy_inv=0.00,
    alpha_S_Gy_inv=0.10,
    beta0_Gy2=0.026,
    domain_radius_um=0.29,
    nucleus_radius_um=8.1,
    z0_Gy=98.0,
    z_R_Gy=29.0,
    gamma_exponent=1.60,
    R_m=2.5,
)
_CHO_OSMK2021 = OSMK2021(
    alpha_L_Gy_inv=0.01,
    alpha_S_Gy_inv=0.17,
    beta0_Gy2=0.045,
    domain_radius_um=0.35,
    nucleus_radius_um=8.1,
    z0_Gy=46.0,
    z_R_Gy=9.7,
    gamma_exponent=1.31,
    R_m=3.1,
)

# Inaniwa & Kanematsu 2023 (doi:10.1093/jrr/rrad049) table 1. Same r_n and K.
_HSG_OSMK2023 = OSMK2023(
    alpha_L_Gy_inv=0.0,
    alpha_S_Gy_inv=0.0797,
    beta0_Gy2=0.0677,
    domain_radius_um=0.29,
    nucleus_radius_um=8.1,
    z0_Gy=60.0,
    f_rd_max=3.00,
    f_z0_max=3.53,
    R_max=4.46,
)
_V79_OSMK2023 = OSMK2023(
    alpha_L_Gy_inv=0.0,
    alpha_S_Gy_inv=0.0,  # yes, zero -- see the class docstring
    beta0_Gy2=0.0374,
    domain_radius_um=0.30,
    nucleus_radius_um=8.1,
    z0_Gy=73.0,
    f_rd_max=2.50,
    f_z0_max=2.96,
    R_max=3.41,
)
_CHO_OSMK2023 = OSMK2023(
    alpha_L_Gy_inv=0.0,
    alpha_S_Gy_inv=0.132,
    beta0_Gy2=0.0591,
    domain_radius_um=0.41,
    nucleus_radius_um=8.1,
    z0_Gy=37.0,
    f_rd_max=2.69,
    f_z0_max=3.23,
    R_max=3.51,
)
