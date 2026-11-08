"""Repair-Misrepair-Fixation (RMF) model.

Carlson, Stewart, Semenenko & Sandison 2008, *Radiat. Res.* **169**(4):447-459,
doi:10.1667/RR1046.1 (the model); Frese, Yu, Stewart & Carlson 2012,
*Int. J. Radiat. Oncol. Biol. Phys.* **83**(1):442-450,
doi:10.1016/j.ijrobp.2011.06.1983 (the proton/carbon RBE application, whose
formulation this module implements).

RMF links reproductive cell death to **DSB induction and processing**. A cell may
die because a DSB is misrepaired or fixed (a one-track, linear effect), or
because two DSBs interact pairwise to form a lethal exchange -- *binary
misrepair* -- which happens both within one track (intratrack, still linear in
dose) and between tracks (intertrack, quadratic). Time-integrating that kinetic
scheme gives the LQ parameters:

```text
(5)  alpha(q) = theta Sigma(q) + kappa z_F(q) Sigma^2(q)
(6)  beta(q)  = (kappa/2) Sigma^2(q)
(7)  z_F(q)   = 0.204 LET / d^2                          [Gy]
```

with ``Sigma`` the DSB yield [Gy^-1 Gbp^-1] from [`pyradbio.micro.mcds`][],
``z_F`` the frequency-mean specific energy in a site of diameter ``d``, and
``theta`` [Gbp] / ``kappa`` [Gbp^2] two **tissue** parameters that are
independent of radiation quality.

**Frese eq. (5) is Carlson eq. (9) with ``f_R = 1``.** Carlson's general form
carries a rejoinable fraction, ``alpha = [1 - f_R(1-theta)]Sigma + kappa z_F
(f_R Sigma)^2``; his own analysis concludes that "all classes of DSB are
potentially rejoinable", and Frese drops it. This module implements the Frese
form, and [`RMFModel.carlson_alpha`][] exposes the general one.

**The two tissue parameters are not free here.** Given a photon reference
``(alpha_x, beta_x)`` measured with a radiation whose own ``Sigma_x`` and
``z_F,x`` are known, eqs. (5)-(6) invert:

```text
kappa = 2 beta_x / Sigma_x^2
theta = (alpha_x - kappa z_F,x Sigma_x^2) / Sigma_x
```

which is how Frese Table 1's ``theta``/``kappa`` columns are produced. Equivalent
reference-relative forms are the paper's eqs. (8)-(9):

```text
(8)  alpha(q) = (Sigma(q)/Sigma_x)
                x [alpha_x + (2 beta_x/Sigma_x)(Sigma(q) z_F(q) - Sigma_x z_F,x)]
(9)  beta(q)  = (Sigma(q)/Sigma_x)^2 beta_x
```

Eq. (9) says something worth stating plainly: ``sqrt(beta/beta_x) = Sigma/Sigma_x``,
i.e. the model's ``RBE_min`` **is** the DSB-yield ratio. RMF therefore lands on
the same ``RBE_min``/``RBE_max`` skeleton as
[`pyradbio.models.phenomenological`][], but with mechanistically computed limits
rather than fitted ones. It is not a subclass of that family because its limits
are functions of ``(Z, beta)``, not of a scalar ``LET_d``.

**Site size trap.** ``d = 5 um`` in eq. (7) is a nucleus **diameter** (Frese: "we
use a diameter of 5 mm [um] in Eq. 7"; Carlson: "an assumed cell nucleus 5 m in
diameter"). The LEM/MKM default ``R_n = 5 um`` elsewhere in this package is a
*radius*: the two differ by a factor of 2 in ``d`` and 4 in ``z_F``.

**Validity.** Frese states the form holds "for particles with a stopping power up
to at least 100 to 150 keV/um", and reports that beyond ~150 keV/um the model
over-predicts measured ``alpha`` and ``beta`` because both ``Sigma`` and ``z_F``
keep rising where the data turn over. Past that this module warns; it never
clamps.
"""

from __future__ import annotations

import warnings
from collections.abc import Callable
from dataclasses import dataclass, field
from typing import Any, ClassVar

from pyradbio.core.lq import LQParameters
from pyradbio.core.photon import PhotonParameters
from pyradbio.core.radiation import Ion
from pyradbio.micro.mcds import MCDSYields
from pyradbio.models.base import BiophysicalModel, ModelCapabilities, apply_preset
from pyradbio.track.stopping_power import StoppingPower, default_stopping_power

Z_F_PREFACTOR: float = 0.204
"""Eq. (7) prefactor: ``z_F = 0.204 LET / d^2`` for a spherical water site, with
LET in keV/um, ``d`` in um and ``z_F`` in Gy (Carlson eq. 2, from ICRU
microdosimetry; assumes straight tracks and the continuous-slowing-down
approximation, so it is poor where multiple scattering or delta-ray escape
matter).
"""

RMF_LET_CEILING_KEV_UM: float = 150.0
"""Frese: the RMF form is stated to hold "up to at least 100 to 150 keV/um", and
above ~150 the paper reports it over-predicts both alpha and beta.
"""


@dataclass(frozen=True, slots=True)
class RMFReferenceRadiation:
    """The low-LET radiation a tissue's ``(alpha_x, beta_x)`` was measured with.

    ``Sigma_x`` is **not** obtainable from the MCDS closed form: Frese simulated
    both values with a secondary-electron spectrum (from MCNP for the 200 kVp
    entry). They are carried as printed.
    """

    label: str
    dsb_per_Gy_per_Gbp: float
    """DSB yield of the reference radiation [Gy^-1 Gbp^-1]."""
    let_keV_um: float
    """Its (unrestricted) LET [keV/um], used only for ``z_F,x``."""

    def __post_init__(self) -> None:
        if self.dsb_per_Gy_per_Gbp <= 0.0:
            raise ValueError(f"dsb_per_Gy_per_Gbp must be > 0, got {self.dsb_per_Gy_per_Gbp!r}")
        if self.let_keV_um <= 0.0:
            raise ValueError(f"let_keV_um must be > 0, got {self.let_keV_um!r}")


CO60_REFERENCE = RMFReferenceRadiation(label="Co-60", dsb_per_Gy_per_Gbp=8.27, let_keV_um=0.36)
"""Frese et al. 2012: "we simulated the reference DSB yield Sigma_x = 8.27
Gy^-1 Gbp^-1 with a reference stopping power of 0.36 keV/um", including a
secondary electron spectrum. Used for the in-vivo/clinical parameter sets.
"""

X_RAY_200KVP_REFERENCE = RMFReferenceRadiation(
    label="200 kVp X-rays", dsb_per_Gy_per_Gbp=8.68, let_keV_um=1.53
)
"""Frese et al. 2012: "reference DSB yield Sigma_200 kVp = 8.68 Gy^-1 Gbp^-1 for
200 kVp X-rays was simulated using MCDS with a secondary electron spectrum
obtained from MCNP. The frequency-mean specific energy was calculated for
S_200 kVp = 1.53 keV/um." Used for the Furusawa in-vitro cell lines.
"""


@dataclass(frozen=True, slots=True)
class RMFAlphaTerms:
    """Eq. (5) split into its two terms, with the inputs they were built from.

    ``alpha = one_track + intratrack``: the one-track term ``theta Sigma`` and the
    intratrack binary-misrepair term ``kappa z_F Sigma^2``. The second carries the
    model's whole LET dependence beyond the DSB yield; returned by
    [`RMFModel.alpha_terms`][].
    """

    let_keV_um: float
    """The ion's LET from the model's stopping power [keV/um]."""
    dsb_per_Gy_per_Gbp: float
    """DSB yield ``Sigma(q)`` [Gy^-1 Gbp^-1]."""
    one_track: float
    """``theta Sigma`` [Gy^-1] -- a single DSB misrepaired or fixed."""
    intratrack: float
    """``kappa z_F Sigma^2`` [Gy^-1] -- two DSBs of the same track interacting."""

    @property
    def alpha(self) -> float:
        """``alpha = one_track + intratrack`` [Gy^-1] (eq. 5)."""
        return self.one_track + self.intratrack


@dataclass(frozen=True, slots=True)
class RMFModel(BiophysicalModel):
    """RMF in the Frese et al. 2012 formulation.

    Carries a photon reference and the radiation it was measured with; the two
    tissue parameters ``theta`` and ``kappa`` are derived from those, so the
    model has **no free biological parameters** beyond the site diameter.
    """

    capabilities: ClassVar[ModelCapabilities] = ModelCapabilities()

    photon: PhotonParameters
    reference_radiation: RMFReferenceRadiation = CO60_REFERENCE
    site_diameter_um: float = 5.0
    """Nucleus **diameter** [um] in eq. (7). NB: a diameter, not a radius."""
    yields: MCDSYields = field(default_factory=MCDSYields)
    stopping_power: StoppingPower = field(default_factory=default_stopping_power)
    let_ceiling_keV_um: float = RMF_LET_CEILING_KEV_UM
    """Warn above the stopping power Frese states the form holds to."""

    def __post_init__(self) -> None:
        if self.site_diameter_um <= 0.0:
            raise ValueError(f"site_diameter_um must be > 0, got {self.site_diameter_um!r}")

    # ---- eq. (7) --------------------------------------------------------
    def z_F_Gy(self, let_keV_um: float) -> float:
        """Eq. (7) ``z_F = 0.204 LET / d^2`` [Gy], ``d`` the site diameter [um]."""
        if let_keV_um <= 0.0:
            raise ValueError(f"let_keV_um must be > 0, got {let_keV_um!r}")
        return Z_F_PREFACTOR * let_keV_um / (self.site_diameter_um * self.site_diameter_um)

    # ---- tissue parameters, eqs. (5)-(6) inverted -----------------------
    @property
    def kappa_per_Gbp2(self) -> float:
        """``kappa = 2 beta_x / Sigma_x^2`` [Gbp^2] -- eq. (6) at the reference."""
        sigma_x = self.reference_radiation.dsb_per_Gy_per_Gbp
        return 2.0 * self.photon.beta_x / (sigma_x * sigma_x)

    @property
    def theta_per_Gbp(self) -> float:
        """``theta = (alpha_x - kappa z_F,x Sigma_x^2)/Sigma_x`` [Gbp] -- eq. (5)."""
        sigma_x = self.reference_radiation.dsb_per_Gy_per_Gbp
        z_f_x = self.z_F_Gy(self.reference_radiation.let_keV_um)
        return (self.photon.alpha_x - self.kappa_per_Gbp2 * z_f_x * sigma_x * sigma_x) / sigma_x

    # ---- radiation quality ----------------------------------------------
    def let_keV_um(self, ion: Ion) -> float:
        """LET of ``ion`` from [`stopping_power`][] [keV/um]."""
        return self.stopping_power.let_keV_um(ion)

    def dsb_per_Gy_per_Gbp(self, ion: Ion) -> float:
        """DSB yield ``Sigma(q)`` [Gy^-1 Gbp^-1] from MCDS (eq. 3 there)."""
        return self.yields.dsb_per_Gy_per_Gbp_for_ion(ion)

    def rbe_dsb(self, ion: Ion) -> float:
        """``Sigma(q)/Sigma_x`` -- the DSB-yield RBE, and the model's ``RBE_min``.

        Exact, from eq. (9): ``sqrt(beta(q)/beta_x) = Sigma(q)/Sigma_x``.
        """
        return self.dsb_per_Gy_per_Gbp(ion) / self.reference_radiation.dsb_per_Gy_per_Gbp

    # ---- LQ parameters, eqs. (5)-(6) ------------------------------------
    def carlson_alpha(self, ion: Ion, *, rejoinable_fraction: float = 1.0) -> float:
        """Carlson eq. (9), the general form with a rejoinable fraction ``f_R``.

        ``f_R = 1`` reproduces Frese eq. (5) exactly and is the default, because
        Carlson's own analysis concludes that no class of DSB is intrinsically
        unrejoinable. Exposed so that conclusion stays checkable rather than
        being silently baked in.
        """
        if not 0.0 < rejoinable_fraction <= 1.0:
            raise ValueError(f"rejoinable_fraction must be in (0, 1], got {rejoinable_fraction!r}")
        sigma = self.dsb_per_Gy_per_Gbp(ion)
        z_f = self.z_F_Gy(self.let_keV_um(ion))
        f_r = rejoinable_fraction
        linear = (1.0 - f_r * (1.0 - self.theta_per_Gbp)) * sigma
        return linear + self.kappa_per_Gbp2 * z_f * (f_r * sigma) ** 2

    def _warn_above_ceiling(self, let: float, stacklevel: int) -> None:
        if let > self.let_ceiling_keV_um:
            warnings.warn(
                f"LET {let:.1f} keV/um is above the {self.let_ceiling_keV_um:g} keV/um "
                "ceiling Frese et al. 2012 states for the RMF form; beyond it the "
                "model over-predicts alpha and beta because Sigma and z_F keep "
                "rising where the data turn over (evaluated, not clamped)",
                UserWarning,
                stacklevel=stacklevel + 1,
            )

    def _terms(self, ion: Ion, let: float) -> RMFAlphaTerms:
        sigma = self.dsb_per_Gy_per_Gbp(ion)
        z_f = self.z_F_Gy(let)
        return RMFAlphaTerms(
            let_keV_um=let,
            dsb_per_Gy_per_Gbp=sigma,
            one_track=self.theta_per_Gbp * sigma,
            intratrack=self.kappa_per_Gbp2 * z_f * sigma * sigma,
        )

    def alpha_terms(self, ion: Ion) -> RMFAlphaTerms:
        """Eq. (5) term by term: one-track ``theta Sigma`` and intratrack ``kappa z_F Sigma^2``.

        Their sum is [`alpha_beta`][pyradbio.models.rmf.model.RMFModel.alpha_beta]'s ``alpha`` for
        the same ion, bit for bit. Warns above [`let_ceiling_keV_um`][], like
        [`alpha_beta`][pyradbio.models.rmf.model.RMFModel.alpha_beta] (evaluated, not clamped). The
        LET is [`stopping_power`][]'s, so pass a model built with the stopping power the terms
        should be plotted against.
        """
        let = self.let_keV_um(ion)
        self._warn_above_ceiling(let, stacklevel=2)
        return self._terms(ion, let)

    def _alpha_beta_ion(self, ion: Ion) -> LQParameters:
        let = self.let_keV_um(ion)
        self._warn_above_ceiling(let, stacklevel=3)
        terms = self._terms(ion, let)
        alpha = terms.one_track + terms.intratrack
        sigma = terms.dsb_per_Gy_per_Gbp
        beta = 0.5 * self.kappa_per_Gbp2 * sigma * sigma
        return LQParameters(alpha=alpha, beta=beta)

    def _dose_weight_stopping_power(self, ion: Ion) -> float:
        """Mixed fields (eq. 10) weight fluence by this model's own LET.

        The mixing itself is the inherited [`BiophysicalModel.alpha_beta`][]
        template -- dose-average ``alpha`` and ``sqrt(beta)`` (Zaider-Rossi /
        Kanai) -- with binned components integrated over their energy range.
        """
        return self.let_keV_um(ion)

    # ---- RBE ------------------------------------------------------------
    def reference(self) -> PhotonParameters:
        """The photon reference [`photon`][].

        RBE is the inherited [`rbe_at_dose`][pyradbio.models.base.BiophysicalModel.rbe_at_dose]
        -- Frese eq. (16), the photon dose giving the same effect divided by the
        ion dose -- and [`rbe_at_survival`][pyradbio.models.base.BiophysicalModel.rbe_at_survival].
        Frese eq. (17) shows the former depends on ``(alpha/beta)_x`` but **not**
        on ``alpha_x``: a structural property of the model, not an approximation.
        """
        return self.photon

    # ---- published parameter sets (Frese et al. 2012, Table 1) ----------
    @classmethod
    def t1_furusawa2000(cls, **overrides: Any) -> RMFModel:
        """Human kidney T1 cells; Furusawa et al., 200 kVp reference."""
        return apply_preset(cls, _T1_FURUSAWA2000, overrides)

    @classmethod
    def hsg_furusawa2000(cls, **overrides: Any) -> RMFModel:
        """Human salivary gland (HSG) cells; Furusawa et al., 200 kVp reference."""
        return apply_preset(cls, _HSG_FURUSAWA2000, overrides)

    @classmethod
    def v79_furusawa2000(cls, **overrides: Any) -> RMFModel:
        """V79 hamster cells; Furusawa et al., 200 kVp reference."""
        return apply_preset(cls, _V79_FURUSAWA2000, overrides)

    @classmethod
    def chordoma_kraemer_scholz2000(cls, **overrides: Any) -> RMFModel:
        """Chordoma (in vivo); Kraemer & Scholz, Co-60 reference."""
        return apply_preset(cls, _CHORDOMA_KRAEMER_SCHOLZ2000, overrides)

    @classmethod
    def prostate_carlson2011(cls, **overrides: Any) -> RMFModel:
        """Prostate tumour (in vivo); Carlson et al., Co-60 reference.

        NB Table 1's printed ``theta/kappa`` for this row is 12.79, while its own
        ``alpha_x``/``beta_x`` give 12.90 -- a 0.8% internal rounding
        inconsistency, the only one in the table.
        """
        return apply_preset(cls, _PROSTATE_CARLSON2011, overrides)

    @classmethod
    def head_and_neck(cls, **overrides: Any) -> RMFModel:
        """Head-and-neck tumour (in vivo); Girinsky et al. / Stuschke & Thames."""
        return apply_preset(cls, _HEAD_AND_NECK, overrides)

    @classmethod
    def published_presets(cls) -> dict[str, Callable[[], RMFModel]]:
        """Every parameter set Frese et al. 2012 Table 1 prints."""
        return {
            "t1": cls.t1_furusawa2000,
            "hsg": cls.hsg_furusawa2000,
            "v79": cls.v79_furusawa2000,
            "chordoma": cls.chordoma_kraemer_scholz2000,
            "prostate": cls.prostate_carlson2011,
            "head_and_neck": cls.head_and_neck,
        }


# ---- Frese et al. 2012 Table 1, as instances (applied with dataclasses.replace) --
_T1_FURUSAWA2000 = RMFModel(
    photon=PhotonParameters(alpha_x=0.0305, beta_x=0.0585),
    reference_radiation=X_RAY_200KVP_REFERENCE,
)
_HSG_FURUSAWA2000 = RMFModel(
    photon=PhotonParameters(alpha_x=0.313, beta_x=0.0615),
    reference_radiation=X_RAY_200KVP_REFERENCE,
)
_V79_FURUSAWA2000 = RMFModel(
    photon=PhotonParameters(alpha_x=0.184, beta_x=0.020), reference_radiation=X_RAY_200KVP_REFERENCE
)
_CHORDOMA_KRAEMER_SCHOLZ2000 = RMFModel(
    photon=PhotonParameters(alpha_x=0.1, beta_x=0.050), reference_radiation=CO60_REFERENCE
)
_PROSTATE_CARLSON2011 = RMFModel(
    photon=PhotonParameters(alpha_x=0.15, beta_x=0.048), reference_radiation=CO60_REFERENCE
)
_HEAD_AND_NECK = RMFModel(
    photon=PhotonParameters(alpha_x=0.25, beta_x=0.025), reference_radiation=CO60_REFERENCE
)


__all__ = [
    "CO60_REFERENCE",
    "RMF_LET_CEILING_KEV_UM",
    "X_RAY_200KVP_REFERENCE",
    "Z_F_PREFACTOR",
    "RMFAlphaTerms",
    "RMFModel",
    "RMFReferenceRadiation",
]
