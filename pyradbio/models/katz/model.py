"""Katz cellular track-structure model (Katz, Ackerson, Homayoonfar & Sharma 1971).

*Inactivation of Cells by Heavy Ion Bombardment*, Radiat. Res. **47**(2):402-425,
doi:10.2307/3573247. The ancestor of every amorphous-track model in this package:
cell killing is attributed to the dose deposited by delta rays, and a cell is
described by exactly **four fitted parameters** -- ``sigma_0``, ``E_0``,
``kappa`` and ``m``.

Equations, numbered as in the paper::

    (3)   P     = (1 - e^{-E/E_0})^m                   m-target 1-hit site response
    (4)   S     = integral 2 pi t (1 - e^{-E/E_0})^m dt   ion-kill cross section
    (8)   P     = sigma/sigma_0 = (1 - e^{-z*^2/(kappa beta^2)})^m
    (9)   N'    = N_0 e^{-sigma F}                     ion-kill survivors
    (10)  N/N_0 = e^{-sigma F} (1 - [1 - e^{-(1-P)E/E_0}]^m)
    (11)  E     = F L
    (12)  z*    = Z [1 - e^{-125 Z^{-2/3} beta}]       Barkas effective charge
    (14)  k     = sigma_0 P / L + (1-P) / E_0          extrapolated radiosensitivity
    (15)  sigma_ext = k L

**Two inactivation modes.** A cell traversed by a single ion may be killed
outright (*ion-kill*, eq. 9), while the dose that does not act that way pools
across ions and kills through the same multi-target statistic as gamma rays
(*gamma-kill*). ``P`` is both the single-ion kill probability and the fraction of
the dose routed to ion-kill; ``1-P`` goes to gamma-kill. The survivors of ion-kill
are the initial population for gamma-kill, which is why eq. (10) is a product.

**Only the grain-count regime is implemented.** Eq. (8) is the closed form Katz
gives for it, and it is the regime the four-parameter fits were made in ("the
observation of an approach to saturation in the cross sections measured for cells
with increasing LET of the bombarding ion implies that these bombardments lie in
the grain-count regime", p. 410). The *track-width* regime -- where ``sigma``
exceeds ``sigma_0``, e.g. Katz's ``sigma/sigma_0 ~ 2`` for argon on kidney cells
(p. 419) -- requires eq. (4), whose site-averaged dose ``E(t)`` the paper supplies
only as **Figs. 2-3**, numerically computed by the method of Butts & Katz 1967
(Radiat. Res. 30:855). No closed form is printed, so that regime is deliberately
out of scope; see ``docs/references/Katz1971.md`` and ``ROADMAP.md``.

**Units.** The paper is cgs: ``sigma_0`` in cm^2, ``E_0`` in erg/cm^3, ``k`` in
rad^-1. Parameters are stored exactly as printed and converted at the boundary to
this package's conventions (um^2, Gy, Gy^-1).
"""

from __future__ import annotations

import math
from collections.abc import Callable
from dataclasses import dataclass, field, replace
from typing import ClassVar, overload

import numpy as np
import numpy.typing as npt

from pyradbio.core.constants import FLUENCE_LET_TO_DOSE_GY, WATER_DENSITY_G_CM3
from pyradbio.core.lq import LQParameters
from pyradbio.core.mixing import Spacing
from pyradbio.core.photon import PhotonParameters
from pyradbio.core.radiation import Ion, RadiationField, dose_to_fluence
from pyradbio.models.base import (
    BiophysicalModel,
    ModelCapabilities,
    PhotonReferenceError,
    invert_survival_curve,
)
from pyradbio.track.stopping_power import (
    StoppingPower,
    default_stopping_power,
    effective_charge,
)

#: Absorbed dose unit conversion. 1 Gy = 1 J/kg = 1e7 erg / 1e3 g = 1e4 erg/g,
#: and for water (rho = 1 g/cm^3) 1 erg/g = 1 erg/cm^3. Katz tabulates ``E_0``
#: in erg/cm^3 throughout (e.g. Chinese hamster E_0 = 1.95e4 = 1.95 Gy).
ERG_CM3_PER_GY: float = 1.0e4

#: Above this log-kill ``S = exp(-log_kill)`` underflows double precision, and
#: eq. (3)'s inverse is its exponential asymptote ``E_0 (log_kill + ln m)``.
_KATZ_ASYMPTOTIC_LOG_KILL: float = 700.0

#: 1 cm^2 = 1e8 um^2. Katz tabulates ``sigma_0`` in cm^2 (Chinese hamster
#: 5.0e-7 cm^2 = 50 um^2, the classic nuclear cross-section).
UM2_PER_CM2: float = 1.0e8


@dataclass(frozen=True, slots=True)
class KatzKillTerms:
    """Eq. (14) split into its two modes [Gy^-1]; :meth:`KatzCellularModel.kill_terms_per_Gy`.

    ``ion_kill = sigma_0 P / L`` (per unit dose, via ``F = D/L``) and
    ``gamma_kill = (1-P)/E_0``. Their sum is the extrapolated radiosensitivity
    ``k`` of :meth:`KatzCellularModel.radiosensitivity_per_Gy`, bit for bit.
    """

    #: ``P = sigma/sigma_0`` (eq. 8): the fraction of the dose routed to ion-kill.
    ion_kill_probability: float
    #: ``sigma_0 P / L`` [Gy^-1].
    ion_kill: float
    #: ``(1-P) / E_0`` [Gy^-1].
    gamma_kill: float

    @property
    def total(self) -> float:
        """``k = ion_kill + gamma_kill`` [Gy^-1] (eq. 14)."""
        return self.ion_kill + self.gamma_kill


@dataclass(frozen=True, slots=True)
class KatzCellularModel(BiophysicalModel):
    """Katz's four-parameter cellular survival model, grain-count regime.

    ``sigma0_cm2``, ``E0_erg_cm3``, ``kappa`` and ``m`` are stored in the paper's
    own cgs units so a preset can be checked against a printed figure panel
    without arithmetic. Use :attr:`sigma0_um2` / :attr:`E0_Gy` for repo units.

    Every equation can be called with an explicit ``let_keV_um``; the routes
    that take a bare :class:`~pyradbio.core.radiation.Ion` fall back to
    :attr:`stopping_power`.
    """

    #: Saturation ion-kill cross section [cm^2] (eq. 8).
    sigma0_cm2: float
    #: Extrapolated gamma-ray D-37 dose [erg/cm^3] (eq. 3).
    E0_erg_cm3: float
    #: Dimensionless track-structure parameter, "related to the size of the
    #: sensitive element" (eq. 8); larger kappa means a less sensitive cell.
    kappa: float
    #: Gamma-ray extrapolation number (eq. 3). ``m = 1`` is a pure 1-hit system.
    m: float
    #: Supplies LET for the ``Ion`` routes. Defaults to the library default
    #: (:func:`~pyradbio.track.stopping_power.default_stopping_power`, analytic
    #: Bethe), as for every other family. NB Katz's own fits were
    #: made against the 1960s ``L`` tables in his Table I, not against this.
    stopping_power: StoppingPower = field(default_factory=default_stopping_power)
    density_g_cm3: float = WATER_DENSITY_G_CM3
    #: Provenance: the Radiat. Res. figure whose panel prints these four values.
    reference_figure: str = ""
    #: The survival data the fit was made against (a citation, not a dependency).
    data_source: str = ""
    reference_doi: str = "10.2307/3573247"

    #: Not LQ (multi-target shoulder) and no mixed fields: eq. (10) is a product
    #: of two non-LQ modes, not a dose average.
    capabilities: ClassVar[ModelCapabilities] = ModelCapabilities(mixed_field=False, lq_exact=False)
    #: Eq. (4) and the track-width regime are not implemented -- see module docs.
    track_width_regime_implemented: ClassVar[bool] = False

    def __post_init__(self) -> None:
        if self.sigma0_cm2 <= 0.0:
            raise ValueError(f"sigma0_cm2 must be > 0, got {self.sigma0_cm2!r}")
        if self.E0_erg_cm3 <= 0.0:
            raise ValueError(f"E0_erg_cm3 must be > 0, got {self.E0_erg_cm3!r}")
        if self.kappa <= 0.0:
            raise ValueError(f"kappa must be > 0, got {self.kappa!r}")
        if self.m < 1.0:
            raise ValueError(f"m must be >= 1 (m=1 is the 1-hit limit), got {self.m!r}")

    # ---- unit views -----------------------------------------------------
    @property
    def sigma0_um2(self) -> float:
        """Saturation cross section in this package's units [um^2]."""
        return self.sigma0_cm2 * UM2_PER_CM2

    @property
    def E0_Gy(self) -> float:
        """Extrapolated gamma-ray D-37 dose [Gy] (= ``E0_erg_cm3`` / 1e4)."""
        return self.E0_erg_cm3 / ERG_CM3_PER_GY

    # ---- eq. (12) -------------------------------------------------------
    @staticmethod
    def effective_charge(z: int, beta: float) -> float:
        """Eq. (12) ``z* = Z[1 - exp(-125 Z^{-2/3} beta)]``.

        Identical to :func:`pyradbio.track.stopping_power.effective_charge`, the
        Barkas form this package already ships -- Katz eq. (12) *is* that
        formula, so it is delegated rather than duplicated.
        """
        return effective_charge(z, beta)

    # ---- eq. (8) --------------------------------------------------------
    def ion_kill_probability(self, z: int, beta: float) -> float:
        """Eq. (8) ``P = sigma/sigma_0 = [1 - exp(-z*^2/(kappa beta^2))]^m``.

        Both the probability that one traversing ion kills the cell and the
        fraction of the deposited dose routed to the ion-kill mode.
        """
        if not 0.0 < beta < 1.0:
            raise ValueError(f"beta must be in (0, 1), got {beta!r}")
        z_star = self.effective_charge(z, beta)
        x = z_star * z_star / (self.kappa * beta * beta)
        return float((-math.expm1(-x)) ** self.m)

    def inactivation_cross_section_um2(self, z: int, beta: float) -> float:
        """Ion-kill cross section ``sigma = sigma_0 P`` [um^2] (eq. 8).

        Saturates at ``sigma_0`` by construction: this is the grain-count regime
        only (see :attr:`track_width_regime_implemented`).
        """
        return self.sigma0_um2 * self.ion_kill_probability(z, beta)

    # ---- eqs. (9)-(11) --------------------------------------------------
    def _let(self, ion: Ion, let_keV_um: float | None) -> float:
        if let_keV_um is None:
            return self.stopping_power.let_keV_um(ion)
        if let_keV_um <= 0.0:
            raise ValueError(f"let_keV_um must be > 0, got {let_keV_um!r}")
        return let_keV_um

    def ion_kill_survival(
        self, ion: Ion, dose_Gy: float, *, let_keV_um: float | None = None
    ) -> float:
        """Eq. (9) ``N'/N_0 = exp(-sigma F)`` with ``F`` from eq. (11)."""
        if dose_Gy < 0.0:
            raise ValueError(f"dose must be >= 0, got {dose_Gy!r}")
        if dose_Gy == 0.0:
            return 1.0
        let = self._let(ion, let_keV_um)
        fluence_cm2 = dose_to_fluence(dose_Gy, let, self.density_g_cm3)
        sigma_cm2 = self.sigma0_cm2 * self.ion_kill_probability(ion.Z, ion.beta())
        return math.exp(-sigma_cm2 * fluence_cm2)

    def photon_survival(self, dose_Gy: float) -> float:
        """Eq. (3) on the whole dose: ``1 - [1 - exp(-D/E_0)]^m``.

        **The model carries its own photon reference.** ``E_0`` and ``m`` *are*
        the gamma-ray extrapolated D-37 dose and extrapolation number, fitted
        jointly with the ion data, so Katz needs no external ``(alpha_x, beta_x)``
        the way the LEM and MKM families do -- this is the ``P -> 0`` limit of
        eq. (10). Use it for the photon (reference) dose -- the numerator -- of an
        RBE that stays inside the published parameter set.

        Note it is a multi-target curve, not an LQ one: it has a shoulder of
        extrapolation number ``m`` and an asymptotic slope of ``1/E_0``.
        """
        return math.exp(-self._photon_log_kill(dose_Gy))

    def _photon_log_kill(self, dose_Gy: float) -> float:
        """Stable ``-ln S_gamma`` of eq. (3), doi:10.2307/3573247.

        Use log1p/expm1 for the nested multi-target differences; when the
        exponential argument exceeds 700, the correction to ``x - ln(m)``
        is below double precision for the model's fitted extrapolation numbers.
        """
        if dose_Gy < 0.0:
            raise ValueError(f"dose must be >= 0, got {dose_Gy!r}")
        x = dose_Gy / self.E0_Gy
        if x == 0.0:
            return 0.0
        if x > _KATZ_ASYMPTOTIC_LOG_KILL:
            return x - math.log(self.m)
        log_hit = math.log(-math.expm1(-x)) if x < 1.0 else math.log1p(-math.exp(-x))
        return -math.log(-math.expm1(self.m * log_hit))

    def gamma_kill_survival(
        self, ion: Ion, dose_Gy: float, *, let_keV_um: float | None = None
    ) -> float:
        """The bracket of eq. (10): ``1 - [1 - exp(-(1-P) E/E_0)]^m``.

        Only the ``(1-P)`` fraction of the dose acts here; the functional form is
        eq. (3), the same multi-target statistic the model assumes for gamma rays.
        """
        if dose_Gy < 0.0:
            raise ValueError(f"dose must be >= 0, got {dose_Gy!r}")
        self._let(ion, let_keV_um)  # validated for a consistent error surface
        p = self.ion_kill_probability(ion.Z, ion.beta())
        gamma_dose = (1.0 - p) * dose_Gy
        return self.photon_survival(gamma_dose)

    @overload
    def survival(
        self,
        field: Ion | RadiationField,
        dose_Gy: float,
        *,
        let_keV_um: float | None = ...,
    ) -> float: ...

    @overload
    def survival(
        self,
        field: Ion | RadiationField,
        dose_Gy: npt.NDArray[np.float64],
        *,
        let_keV_um: float | None = ...,
    ) -> npt.NDArray[np.float64]: ...

    def survival(
        self,
        field: Ion | RadiationField,
        dose_Gy: float | npt.NDArray[np.float64],
        *,
        let_keV_um: float | None = None,
    ) -> float | npt.NDArray[np.float64]:
        """Eq. (10): surviving fraction under a fluence ``F = E/L`` of one ion.

        The product of the two modes, because the ion-kill survivors are taken as
        the initial population for gamma-kill. Accepts a scalar or an array of
        doses; mixed fields are refused (see :attr:`capabilities`).
        """
        ion = self._require_ion(field)
        if np.ndim(dose_Gy) == 0:
            return self._survival_one(ion, float(dose_Gy), let_keV_um)
        doses = np.asarray(dose_Gy, dtype=np.float64)
        flat = [self._survival_one(ion, float(d), let_keV_um) for d in doses.ravel()]
        return np.asarray(flat, dtype=np.float64).reshape(doses.shape)

    def _survival_one(self, ion: Ion, dose_Gy: float, let_keV_um: float | None) -> float:
        return math.exp(-self._log_kill(ion, dose_Gy, let_keV_um=let_keV_um))

    @staticmethod
    def _require_ion(field: Ion | RadiationField) -> Ion:
        if isinstance(field, Ion):
            return field
        raise NotImplementedError(
            "Katz mixed-field survival is not the LQ dose-average: eq. (10) "
            "is a product of two non-LQ modes. Evaluate single ions."
        )

    def dose_for_survival(
        self,
        field: Ion | RadiationField,
        survival_fraction: float,
        *,
        let_keV_um: float | None = None,
    ) -> float:
        """Dose [Gy] at which eq. (10) reaches ``survival_fraction`` (numerical root).

        Eq. (10) has no closed-form inverse, so this solves on the curve itself.
        """
        ion = self._require_ion(field)
        return invert_survival_curve(
            survival_fraction,
            lambda dose: self._survival_one(ion, dose, let_keV_um),
            start_Gy=self.E0_Gy,
        )

    def _log_kill(
        self, field: Ion | RadiationField, dose_Gy: float, *, let_keV_um: float | None = None
    ) -> float:
        """Eq. (10): ``sigma F - ln S_gamma((1-P)D)``, without forming survival."""
        ion = self._require_ion(field)
        let = self._let(ion, let_keV_um)
        fluence = dose_to_fluence(dose_Gy, let, self.density_g_cm3)
        p = self.ion_kill_probability(ion.Z, ion.beta())
        return self.sigma0_cm2 * p * fluence + self._photon_log_kill((1.0 - p) * dose_Gy)

    def rbe_at_survival(
        self,
        field: Ion | RadiationField,
        survival_fraction: float,
        *,
        let_keV_um: float | None = None,
    ) -> float:
        """Iso-survival RBE of eq. (10), using the same LET override as survival."""
        if not 0.0 < survival_fraction <= 1.0:
            raise ValueError(f"survival fraction must be in (0, 1], got {survival_fraction!r}")
        d_ref = self.photon_dose_for_survival(survival_fraction)
        return d_ref / self.dose_for_survival(field, survival_fraction, let_keV_um=let_keV_um)

    def rbe_at_dose(
        self, field: Ion | RadiationField, dose_Gy: float, *, let_keV_um: float | None = None
    ) -> float:
        """Iso-effect RBE of eq. (10) on log-kill, with an optional measured LET."""
        if dose_Gy <= 0.0:
            raise ValueError(f"dose must be > 0 Gy, got {dose_Gy!r}")
        effect = self._log_kill(field, dose_Gy, let_keV_um=let_keV_um)
        return self._reference_dose_for_log_kill(effect) / dose_Gy

    # ---- photon reference: eq. (3), not LQ -------------------------------
    def reference(self) -> PhotonParameters:
        """Katz has no **LQ** photon reference: it raises.

        The model carries its own, multi-target, gamma-ray curve
        (:meth:`photon_survival`, eq. 3 with ``E_0`` and ``m``), which RBE
        uses directly -- see :meth:`_reference_dose_for_log_kill`. There is no
        ``(alpha_x, beta_x)`` pair to return without inventing one.
        """
        raise PhotonReferenceError(
            "Katz's photon reference is the multi-target curve of eq. (3) "
            "(photon_survival, from E_0 and m), not an LQ pair; rbe_at_survival and "
            "rbe_at_dose use it directly"
        )

    def photon_dose_for_survival(self, survival_fraction: float) -> float:
        """Gamma-ray dose [Gy] at which eq. (3) reaches ``survival_fraction``.

        The closed-form inverse of :meth:`photon_survival`,
        ``D = -E_0 ln(1 - (1 - S)^{1/m})`` -- the numerator (reference dose) of an
        RBE that stays inside the published four-parameter set (the photon dose of
        :meth:`rbe_at_survival`).
        """
        if not 0.0 < survival_fraction <= 1.0:
            raise ValueError(f"survival fraction must be in (0, 1], got {survival_fraction!r}")
        return self._reference_dose_for_log_kill(-math.log(survival_fraction))

    def _reference_dose_for_log_kill(self, log_kill: float) -> float:
        """The gamma-ray dose with ``-ln S = log_kill`` on eq. (3) [Gy].

        Inverts ``S = 1 - [1 - exp(-D/E_0)]^m`` in closed form,
        ``D = -E_0 ln(1 - (1 - S)^{1/m})``, evaluated through ``expm1``/``log1p``
        so it stays accurate at both ``S -> 1`` and ``S -> 0``. Only where ``S``
        underflows is the exponential asymptote ``D = E_0 (log_kill + ln m)`` used.
        """
        if not math.isfinite(log_kill) or log_kill < 0.0:
            raise ValueError(f"log_kill must be finite and >= 0, got {log_kill!r}")
        if log_kill == 0.0:
            return 0.0
        if log_kill > _KATZ_ASYMPTOTIC_LOG_KILL:
            return self.E0_Gy * (log_kill + math.log(self.m))
        # ln(1 - S), accurate at both ends: expm1 near S = 1, log1p near S = 0.
        if log_kill < 1.0:
            log_one_minus_s = math.log(-math.expm1(-log_kill))
        else:
            log_one_minus_s = math.log1p(-math.exp(-log_kill))
        hit = -math.expm1(log_one_minus_s / self.m)  # 1 - (1 - S)^(1/m)
        return -self.E0_Gy * math.log(hit)

    # ---- eqs. (14)-(15) -------------------------------------------------
    def _um2_per_Gy_keV_um(self) -> float:
        """Cross-section per unit ``k*L`` [um^2 / (Gy^-1 keV/um)] (eq. 15)."""
        return FLUENCE_LET_TO_DOSE_GY * UM2_PER_CM2 / self.density_g_cm3

    def radiosensitivity_per_Gy(self, z: int, beta: float, *, let_keV_um: float) -> float:
        """Eq. (14) ``k = sigma_0 P / L + (1-P)/E_0`` [Gy^-1].

        The *extrapolated* radiosensitivity: the reciprocal dose increment that
        reduces survival by ``1/e`` on the exponential (high-dose) part of the
        curve, obtained by expanding the binomial of eq. (10). It is **not** the
        initial slope -- see :meth:`alpha_beta`.

        The first term is ion-kill (per unit dose, via ``F = D/L``), the second
        gamma-kill. They swap rank across the LET range, which is the model's
        central claim.
        """
        return self.kill_terms_per_Gy(z, beta, let_keV_um=let_keV_um).total

    def kill_terms_per_Gy(self, z: int, beta: float, *, let_keV_um: float) -> KatzKillTerms:
        """Eq. (14) term by term: ion-kill ``sigma_0 P / L`` and gamma-kill ``(1-P)/E_0``.

        The split :meth:`radiosensitivity_per_Gy` sums. At low LET almost all of
        ``k`` is gamma-kill; once ``P -> 1`` the gamma-kill term vanishes and ``k``
        collapses to ``sigma_0 / L``, which falls with LET (overkill).
        """
        if let_keV_um <= 0.0:
            raise ValueError(f"let_keV_um must be > 0, got {let_keV_um!r}")
        p = self.ion_kill_probability(z, beta)
        fluence_per_Gy = dose_to_fluence(1.0, let_keV_um, self.density_g_cm3)
        return KatzKillTerms(
            ion_kill_probability=p,
            ion_kill=self.sigma0_cm2 * p * fluence_per_Gy,
            gamma_kill=(1.0 - p) / self.E0_Gy,
        )

    def extrapolated_cross_section_um2(self, z: int, beta: float, *, let_keV_um: float) -> float:
        """Eq. (15) ``sigma_ext = k L`` [um^2] -- the quantity many authors quote."""
        k = self.radiosensitivity_per_Gy(z, beta, let_keV_um=let_keV_um)
        return k * let_keV_um * self._um2_per_Gy_keV_um()

    # ---- BiophysicalModel seam -----------------------------------------
    def _alpha_beta_ion(self, ion: Ion) -> LQParameters:
        """Map Katz onto LQ parameters. **Katz is not an LQ model.**

        ``alpha`` is eq. (14)'s extrapolated radiosensitivity ``k`` -- the
        quantity Katz himself tabulates (Table I, ``k`` th./exp.) and compares
        against the radiosensitivities experimentalists report, which makes it
        the only defensible LQ analogue here.

        ``beta`` is identically **zero**, and that is exact rather than an
        approximation: the model's shoulder comes from the multi-target
        exponent ``m``, so at low dose the gamma-kill term behaves as ``D^m``
        (``m = 2.5`` for most cell lines), not as ``D^2``. There is no quadratic
        coefficient to report. Consequently the true ``D -> 0`` slope is the
        ion-kill term alone, which at low LET is near zero -- use
        :meth:`survival` for anything dose-resolved rather than reading a
        survival curve back out of these parameters.
        """
        let = self._let(ion, None)
        k = self.radiosensitivity_per_Gy(ion.Z, ion.beta(), let_keV_um=let)
        return LQParameters(alpha=k, beta=0.0)

    def _alpha_beta_field(
        self, field: RadiationField, *, sub_nodes: int, spacing: Spacing
    ) -> LQParameters:
        raise NotImplementedError(
            "Katz mixed-field survival is not the LQ dose-average: eq. (10) "
            "is a product of two non-LQ modes. Evaluate single ions."
        )

    def with_stopping_power(self, stopping_power: StoppingPower) -> KatzCellularModel:
        """A copy of this model bound to ``stopping_power`` (presets ship unbound)."""
        return replace(self, stopping_power=stopping_power)

    # ---- published parameter sets (Katz et al. 1971, Figs. 6-15) --------
    # Every value below was read from the figure panel named in
    # ``reference_figure``; all ten also appear in Katz's own IAEA-SM-145/9
    # Table I, which agrees except for one typo (see the Todd entries).

    @classmethod
    def bacterial_spores_powers1968_n2(cls) -> KatzCellularModel:
        """Bacterial spores, N2 (Fig. 6)."""
        return cls(
            sigma0_cm2=2.01e-9,
            E0_erg_cm3=5.9e6,
            kappa=950.0,
            m=4.0,
            reference_figure="Fig. 6",
            data_source="Powers, Lyman & Tobias 1968",
        )

    @classmethod
    def bacterial_spores_powers1968_o2(cls) -> KatzCellularModel:
        """Bacterial spores, O2 (Fig. 7)."""
        return cls(
            sigma0_cm2=2.01e-9,
            E0_erg_cm3=4.7e6,
            kappa=800.0,
            m=4.0,
            reference_figure="Fig. 7",
            data_source="Powers, Lyman & Tobias 1968",
        )

    @classmethod
    def bacterial_spores_powers1968_h2s(cls) -> KatzCellularModel:
        """Bacterial spores, H2S post-treatment (Fig. 8)."""
        return cls(
            sigma0_cm2=2.01e-9,
            E0_erg_cm3=1.1e7,
            kappa=1100.0,
            m=4.0,
            reference_figure="Fig. 8",
            data_source="Powers, Lyman & Tobias 1968",
        )

    @classmethod
    def haploid_yeast_sayeg1959(cls) -> KatzCellularModel:
        """Haploid yeast (Fig. 9). The only preset with ``m = 2``."""
        return cls(
            sigma0_cm2=1.33e-8,
            E0_erg_cm3=4.0e5,
            kappa=1800.0,
            m=2.0,
            reference_figure="Fig. 9",
            data_source="Sayeg, Birge, Beam & Tobias 1959",
        )

    @classmethod
    def t1_kidney_todd1966_anoxic(cls) -> KatzCellularModel:
        """T-1 human kidney cells, anoxic (Fig. 10).

        ``sigma_0`` is 6.7e-7 cm^2 as printed in the Radiat. Res. figure panel.
        Katz's IAEA-SM-145/9 Table I prints ``6.70x10^27 cm^2`` for this row,
        which is impossible; the figure settles it.
        """
        return cls(
            sigma0_cm2=6.7e-7,
            E0_erg_cm3=2.9e4,
            kappa=1300.0,
            m=2.5,
            reference_figure="Fig. 10",
            data_source="Todd 1966",
        )

    @classmethod
    def t1_kidney_todd1966_aerobic(cls) -> KatzCellularModel:
        """T-1 human kidney cells, aerobic (Fig. 11).

        Katz notes (p. 419) that argon bombardment of these cells reaches the
        **track-width** regime, which this implementation does not cover.
        """
        return cls(
            sigma0_cm2=6.7e-7,
            E0_erg_cm3=1.7e4,
            kappa=1000.0,
            m=2.5,
            reference_figure="Fig. 11",
            data_source="Todd 1966",
        )

    @classmethod
    def t1_kidney_barendsen1966_n2(cls) -> KatzCellularModel:
        """T-1 human kidney cells, N2 (Fig. 12)."""
        return cls(
            sigma0_cm2=5.4e-7,
            E0_erg_cm3=4.6e4,
            kappa=1900.0,
            m=2.5,
            reference_figure="Fig. 12",
            data_source="Barendsen et al. 1966",
        )

    @classmethod
    def t1_kidney_barendsen1966_o2(cls) -> KatzCellularModel:
        """T-1 human kidney cells, O2 (Fig. 13)."""
        return cls(
            sigma0_cm2=5.4e-7,
            E0_erg_cm3=1.8e4,
            kappa=1400.0,
            m=2.5,
            reference_figure="Fig. 13",
            data_source="Barendsen et al. 1966",
        )

    @classmethod
    def chinese_hamster_skarsgard1967(cls) -> KatzCellularModel:
        """Mammalian (Chinese hamster) cells (Fig. 14).

        ``sigma_0 = 5.0e-7 cm^2 = 50 um^2`` -- the same effective nuclear
        cross-section Scholz & Kraft later use for V79 (see
        :meth:`pyradbio.core.cell.CellTarget.effective_nucleus_radius_um`).
        """
        return cls(
            sigma0_cm2=5.0e-7,
            E0_erg_cm3=1.95e4,
            kappa=1400.0,
            m=2.5,
            reference_figure="Fig. 14",
            data_source="Skarsgard et al. 1967",
        )

    @classmethod
    def hela_deering1962(cls) -> KatzCellularModel:
        """HeLa cells (Fig. 15). The only preset with ``m = 3``."""
        return cls(
            sigma0_cm2=5.6e-7,
            E0_erg_cm3=1.4e4,
            kappa=750.0,
            m=3.0,
            reference_figure="Fig. 15",
            data_source="Deering & Rice 1962",
        )

    @classmethod
    def published_presets(cls) -> dict[str, Callable[[], KatzCellularModel]]:
        """Every parameter set Katz et al. 1971 prints, keyed by short name."""
        return {
            "spores_n2": cls.bacterial_spores_powers1968_n2,
            "spores_o2": cls.bacterial_spores_powers1968_o2,
            "spores_h2s": cls.bacterial_spores_powers1968_h2s,
            "yeast": cls.haploid_yeast_sayeg1959,
            "t1_todd_anoxic": cls.t1_kidney_todd1966_anoxic,
            "t1_todd_aerobic": cls.t1_kidney_todd1966_aerobic,
            "t1_barendsen_n2": cls.t1_kidney_barendsen1966_n2,
            "t1_barendsen_o2": cls.t1_kidney_barendsen1966_o2,
            "chinese_hamster": cls.chinese_hamster_skarsgard1967,
            "hela": cls.hela_deering1962,
        }


__all__ = ["ERG_CM3_PER_GY", "UM2_PER_CM2", "KatzCellularModel", "KatzKillTerms"]
