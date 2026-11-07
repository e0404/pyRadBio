"""The Bethe stopping-power formula for liquid water, with switchable corrections.

Mass electronic stopping power (S. Navas et al. (Particle Data Group), Review of Particle
Physics, Phys. Rev. D 110, 030001 (2024), ch. 34, eqs. (34.4), (34.5); CC BY 4.0):

    S/rho = K z^2 (Z/A) / beta^2 * [ 1/2 ln(2 m_e c^2 beta^2 gamma^2 W_max / I^2)
                                     - beta^2 - delta(beta gamma)/2 ]
    W_max = 2 m_e c^2 beta^2 gamma^2 / (1 + 2 gamma m_e/M + (m_e/M)^2)

with ``K = 0.307075 MeV cm^2/mol`` and ``(Z/A) = 0.55509`` for water (PDG), the bare
projectile charge ``z = Z`` and the true nuclear mass ``M``
(:mod:`~pyradbio.track.stopping.masses`). The bare charge OVERESTIMATES partially stripped
ions: measured at 4.7-10 MeV/u (with shell, Barkas, Bloch, Ahlen) vs the ICRU 73
errata / MSTAR, Si +9% / +17%, Ar +16% / +23%; C within 2% (ICRU 90 carbon: 0.04%).
Hence ``effective_charge`` (default ``"hubert"``) for Z >= 3;
protons and alphas are bare in this range. ``"hubert"`` -- Hubert, Bimbot & Gauvin (1990),
``S = (z_eff/2)^2 S_alpha`` at the same velocity, the alpha evaluated with this instance's
own options; ``"fitted"`` -- Barkas's form with our MSTAR-fitted coefficient; ``"brandt_kitagawa"``
-- Brandt & Kitagawa (1982) eq. (23); ``"barkas"`` -- the original Barkas form; ``None`` --
bare; or any :class:`~pyradbio.track.stopping.effective_charge.EffectiveChargeModel`. The
non-alpha models scale ``S = (z_eff/Z)^2 S_bare``. Measured accuracy: see
:mod:`~pyradbio.track.stopping.effective_charge`. The alpha-referenced Hubert form trades
the ion's own Z-dependent higher-order terms (Barkas, Bloch, Mott/LS) for the alpha's,
as Hubert et al. fitted it: ICRU 90 carbon 1.1% (bare 0.04%) -- which is why
:class:`~pyradbio.track.stopping.icru90_procedure.ICRU90Procedure` keeps bare charge.
PDG states the formula holds for ``0.1 <~ beta*gamma <~ 1000`` "with an accuracy of a
few percent"; those are the default range limits (options ``min_beta_gamma`` and
``max_beta_gamma``); outside them :class:`~pyradbio.track.stopping.base.OutOfRangeError`
is raised.

Options:

* ``i_value_eV`` -- mean excitation energy, default 78 eV (ICRU 90).
* ``density_effect`` -- ``"tabulated"`` (default): NIST's delta(beta*gamma) for water at
  exactly the chosen I -- ESTAR at 75 eV, ICRU 90 table A.3 at 78 eV;
  ``"sternheimer"``: PDG eq. (34.7) with PDG's coefficients for water, which belong to
  I = 75 eV (their Cbar = 2 ln(I/hbar omega_p) + 1 identifies it) and are refused for
  other I; ``"sternheimer_peierls"``: the general recipe of R.M. Sternheimer and R.F.
  Peierls, Phys. Rev. B 3, 3681 (1971), for any I; ``"none"``; or any
  :class:`DensityEffectModel`.
* ``shell`` -- subtract ``C/Z``: ``True`` or a
  :class:`~pyradbio.track.stopping.shell.WaterShellCorrection` (Walske's hydrogenic K and
  L shells for O plus the exact hydrogen atom, ICRU 49 "Model 1" in spirit; its two
  oxygen ``theta`` are calibrated, see :mod:`~pyradbio.track.stopping.shell`).
* ``barkas`` -- add ``z L1``: ``True`` or a
  :class:`~pyradbio.track.stopping.barkas.WaterBarkasCorrection` (Ashley, Ritchie &
  Brandt 1973 through ICRU 49 eq. 2.8, ``gamma = 1`` by default -- see
  :mod:`~pyradbio.track.stopping.barkas`). It needs ``x_O >= 1`` (beta^2 >= 8 alpha^2,
  ~0.2 MeV/u) and raises below.
* Measured for protons, ``shell=True, barkas=True, bloch=True``: within
  +-0.13% of PSTAR (ICRU 49, ``i_value_eV=75``) and +-0.15% of ICRU 90 (78 eV) from 0.5
  to 1000 MeV (the tests pin it); without shell and Barkas, up to +2.6% below 5 MeV.
* ``bloch``, ``mott`` -- Bloch's correction and the linear Mott term, added to the
  bracket; ``ahlen`` -- Ahlen's (1980) close-collision Mott correction ``G/2``, used with
  ``bloch`` in place of ``mott`` (ICRU 90's choice for C; measured with Bloch:
  ICRU 90 carbon to 0.058% over 30-833 MeV/u, vs 0.059% with LS); ``lindhard_sorensen``
  -- J. Lindhard and A.H. Sorensen, Phys. Rev. A 53, 2443 (1996), point nucleus, which
  **replaces** Bloch and Mott (combining them raises). See
  :mod:`~pyradbio.track.stopping.higher_order`. With ``lindhard_sorensen=True`` the
  energy range ends where ``gamma A^(1/3) / 160 = 0.1`` (ASSUMPTION: one tenth of LS's
  estimate, p. 2451, of where finite nuclear size -- not implemented -- starts to matter).

``quantity="total"`` adds ``nuclear``
(:class:`~pyradbio.track.stopping.nuclear.WaterNuclearStopping`; default: ZBL for
every ion, protons included -- the analytic engines' choice).
"""

from __future__ import annotations

import csv
import math
from dataclasses import dataclass, field
from functools import cache
from importlib import resources
from typing import ClassVar, Literal, Protocol, get_args

import numpy as np
import numpy.typing as npt
from scipy.interpolate import PchipInterpolator

from pyradbio.core.constants import (
    BETHE_K_MEV_CM2_MOL,
    ELECTRON_MASS_MEV,
    WATER_DENSITY_G_CM3,
    WATER_Z_OVER_A,
)
from pyradbio.track.stopping.barkas import WaterBarkasCorrection
from pyradbio.track.stopping.base import (
    IonSourceInfo,
    MassStoppingPowerModel,
    OptionValue,
    OutOfRangeError,
)
from pyradbio.track.stopping.effective_charge import (
    FITTED_BARKAS_COEFFICIENT,
    BarkasEffectiveCharge,
    BrandtKitagawaEffectiveCharge,
    EffectiveChargeModel,
    HubertEffectiveCharge,
)
from pyradbio.track.stopping.higher_order import (
    ahlen_mott_correction,
    bloch_correction,
    lindhard_sorensen_correction,
    mott_correction,
)
from pyradbio.track.stopping.masses import nuclear_mass_MeV
from pyradbio.track.stopping.nuclear import WaterNuclearStopping
from pyradbio.track.stopping.shell import WaterShellCorrection

DensityEffectName = Literal["tabulated", "sternheimer", "sternheimer_peierls", "none"]

BETHE_REFERENCE = (
    "Bethe formula, S. Navas et al. (Particle Data Group), Phys. Rev. D 110, 030001 (2024), "
    "eqs. (34.4), (34.5)"
)
#: PDG's stated validity of eq. (34.5) in beta*gamma.
PDG_BETA_GAMMA_RANGE = (0.1, 1000.0)
#: With Lindhard-Sorensen, stop at gamma A^(1/3) / 160 = this (point nucleus; ASSUMPTION).
LS_FINITE_SIZE_FRACTION = 0.1
_LN10 = math.log(10.0)


class DensityEffectModel(Protocol):
    """delta(beta*gamma) of liquid water."""

    @property
    def name(self) -> str: ...

    @property
    def i_value_eV(self) -> float | None:
        """The I-value the model is valid for (``None``: any)."""

    @property
    def beta_gamma_range(self) -> tuple[float, float]: ...

    def delta(self, beta_gamma: npt.NDArray[np.float64]) -> npt.NDArray[np.float64]: ...


@dataclass(frozen=True)
class NoDensityEffect:
    """delta = 0."""

    name: ClassVar[str] = "none"
    i_value_eV: ClassVar[float | None] = None
    beta_gamma_range: ClassVar[tuple[float, float]] = (0.0, math.inf)

    def delta(self, beta_gamma: npt.NDArray[np.float64]) -> npt.NDArray[np.float64]:
        return np.zeros_like(np.asarray(beta_gamma, dtype=np.float64))


@cache
def _shipped_rows(name: str) -> tuple[dict[str, str], ...]:
    text = resources.files(__package__).joinpath("data").joinpath(name).read_text(encoding="utf-8")
    return tuple(csv.DictReader(line for line in text.splitlines() if not line.startswith("#")))


@dataclass(frozen=True)
class TabulatedDensityEffect:
    """NIST's delta(beta*gamma) for water at one I-value, PCHIP in log10(beta*gamma).

    Water is an insulator, so delta = 0 below its onset (PDG eq. 34.7, x < x0); a table
    whose first entry is 0 therefore returns 0 below its first node -- that is the
    physics, not an extrapolation. Above its last node it raises.
    """

    source: str
    i_value_eV: float
    density_g_cm3: float
    beta_gamma: tuple[float, ...]
    values: tuple[float, ...]
    _interpolant: PchipInterpolator = field(init=False, repr=False, compare=False)

    def __post_init__(self) -> None:
        object.__setattr__(
            self, "_interpolant", PchipInterpolator(np.log10(self.beta_gamma), self.values)
        )

    @property
    def name(self) -> str:
        return f"tabulated ({self.source})"

    @property
    def beta_gamma_range(self) -> tuple[float, float]:
        lo = 0.0 if self.values[0] == 0.0 else self.beta_gamma[0]
        return lo, self.beta_gamma[-1]

    @classmethod
    def for_i_value(cls, i_value_eV: float) -> TabulatedDensityEffect:
        """ESTAR (75 eV) or ICRU 90 table A.3 (78 eV)."""
        rows = _shipped_rows("density_effect_water_liquid.csv")
        chosen = [r for r in rows if float(r["i_value_eV"]) == i_value_eV]
        if not chosen:
            available = sorted({float(r["i_value_eV"]) for r in rows})
            raise ValueError(
                f"no tabulated density effect for water at I = {i_value_eV:g} eV "
                f"(available: {available}); use density_effect='none' or another model"
            )
        return cls(
            source=chosen[0]["source"],
            i_value_eV=i_value_eV,
            density_g_cm3=float(chosen[0]["density_g_cm3"]),
            beta_gamma=tuple(float(r["beta_gamma"]) for r in chosen),
            values=tuple(float(r["delta"]) for r in chosen),
        )

    def delta(self, beta_gamma: npt.NDArray[np.float64]) -> npt.NDArray[np.float64]:
        bg = np.asarray(beta_gamma, dtype=np.float64)
        lo, hi = self.beta_gamma_range
        if bg.size and (bg.min() < lo or bg.max() > hi):
            raise OutOfRangeError((float(bg.min()), float(bg.max())), (lo, hi), source=self.name)
        below = bg < self.beta_gamma[0]  # only reachable when the table starts at 0
        out = np.zeros_like(bg)
        out[~below] = self._interpolant(np.log10(bg[~below]))
        return out


@dataclass(frozen=True)
class SternheimerDensityEffect:
    """PDG eq. (34.7) for an insulator, ``x = log10(beta*gamma)``::

    delta = 2 ln10 x - Cbar                     x >= x1
          = 2 ln10 x - Cbar + a (x1 - x)^k      x0 <= x < x1
          = 0                                    x < x0      (delta0 = 0 for water)
    """

    coefficients: dict[str, float]
    i_value_eV: float
    name: ClassVar[str] = "sternheimer (PDG eq. 34.7)"
    beta_gamma_range: ClassVar[tuple[float, float]] = (0.0, math.inf)

    @classmethod
    def water(cls) -> SternheimerDensityEffect:
        """PDG's coefficients for liquid water (CC BY 4.0), valid for I = 75 eV."""
        (row,) = _shipped_rows("sternheimer_water_liquid.csv")
        coefficients = {k: float(v) for k, v in row.items() if k != "i_value_eV"}
        if coefficients["delta0"] != 0.0:
            raise ValueError("water is an insulator; delta0 must be 0")
        return cls(coefficients, float(row["i_value_eV"]))

    def delta(self, beta_gamma: npt.NDArray[np.float64]) -> npt.NDArray[np.float64]:
        c = self.coefficients
        x = np.log10(np.asarray(beta_gamma, dtype=np.float64))
        high = 2.0 * _LN10 * x - c["Cbar"]
        mid = high + c["a"] * np.clip(c["x1"] - x, 0.0, None) ** c["k"]
        out: npt.NDArray[np.float64] = np.where(
            x >= c["x1"], high, np.where(x >= c["x0"], mid, 0.0)
        )
        return out


@dataclass(frozen=True)
class SternheimerPeierlsDensityEffect:
    """General density effect of R.M. Sternheimer and R.F. Peierls, Phys. Rev. B 3, 3681
    (1971), for condensed materials (solids and liquids), any I::

        X = log10(beta*gamma) = log10(p / m0 c)
        delta = 4.606 X - Cbar + a (X1 - X)^m     X0 < X < X1      (eq. 1)
              = 4.606 X - Cbar                    X > X1           (eq. 2)
              = 0                                 X < X0           (eq. 6, delta(X0) = 0)
        Cbar = 2 ln(I / h nu_p) + 1,  h nu_p = 28.8 (rho Z/A)^(1/2) eV      (eqs. 3, 5)
        X_a = Cbar / 4.606,  a = 4.606 (X_a - X0) / (X1 - X0)^m             (eqs. 8, 9)
        I < 100 eV:  X1 = 2.0, m = 3.0, X0 = 0.2 (Cbar < 3.681) or 0.326 Cbar - 1.0
        I >= 100 eV: X1 = 3.0, m = 3.0, X0 = 0.2 (Cbar < 5.215) or 0.326 Cbar - 1.5
                                                                     (eqs. 33-36)

    "4.606 X is simply 2 ln(p/m0 c)" (their text after eq. 2), so 4.606 is taken as
    2 ln 10. The authors expect errors in dE/dx below 2%, typically about 1%.
    """

    i_value_eV: float
    density_g_cm3: float = WATER_DENSITY_G_CM3
    z_over_a: float = WATER_Z_OVER_A
    beta_gamma_range: ClassVar[tuple[float, float]] = (0.0, math.inf)

    @property
    def name(self) -> str:
        return "sternheimer-peierls (1971)"

    @property
    def parameters(self) -> dict[str, float]:
        hv_p = 28.8 * math.sqrt(self.density_g_cm3 * self.z_over_a)
        cbar = 2.0 * math.log(self.i_value_eV / hv_p) + 1.0
        if self.i_value_eV < 100.0:
            x1, x0 = 2.0, (0.2 if cbar < 3.681 else 0.326 * cbar - 1.0)
        else:
            x1, x0 = 3.0, (0.2 if cbar < 5.215 else 0.326 * cbar - 1.5)
        m = 3.0
        x_a = cbar / (2.0 * _LN10)
        a = 2.0 * _LN10 * (x_a - x0) / (x1 - x0) ** m
        return {
            "I_eV": self.i_value_eV,
            "plasma_energy_eV": hv_p,
            "Cbar": cbar,
            "x0": x0,
            "x1": x1,
            "m": m,
            "a": a,
        }

    def delta(self, beta_gamma: npt.NDArray[np.float64]) -> npt.NDArray[np.float64]:
        p = self.parameters
        x = np.log10(np.asarray(beta_gamma, dtype=np.float64))
        high = 2.0 * _LN10 * x - p["Cbar"]
        mid = high + p["a"] * np.clip(p["x1"] - x, 0.0, None) ** p["m"]
        out: npt.NDArray[np.float64] = np.where(x >= p["x1"], high, np.where(x > p["x0"], mid, 0.0))
        return out


EffectiveChargeName = Literal["hubert", "fitted", "brandt_kitagawa", "barkas"]


def _effective_charge(
    choice: EffectiveChargeName | EffectiveChargeModel | None,
) -> EffectiveChargeModel | None:
    if choice is None or not isinstance(choice, str):
        return choice
    models: dict[str, EffectiveChargeModel] = {
        "hubert": HubertEffectiveCharge(),
        "fitted": BarkasEffectiveCharge(coefficient=FITTED_BARKAS_COEFFICIENT),
        "brandt_kitagawa": BrandtKitagawaEffectiveCharge(),
        "barkas": BarkasEffectiveCharge(),
    }
    if choice not in models:
        raise ValueError(
            f"effective_charge must be one of {get_args(EffectiveChargeName)}, None or a model"
        )
    return models[choice]


def _density_effect(
    choice: DensityEffectName | DensityEffectModel, i_value_eV: float, density_g_cm3: float
) -> DensityEffectModel:
    if not isinstance(choice, str):
        model = choice
    elif choice == "tabulated":
        try:
            model = TabulatedDensityEffect.for_i_value(i_value_eV)
        except ValueError as exc:
            raise ValueError(f"density_effect='tabulated': {exc}") from None
    elif choice == "sternheimer":
        model = SternheimerDensityEffect.water()
    elif choice == "sternheimer_peierls":
        model = SternheimerPeierlsDensityEffect(i_value_eV, density_g_cm3)
    elif choice == "none":
        model = NoDensityEffect()
    else:
        raise ValueError(f"density_effect must be one of {get_args(DensityEffectName)} or a model")
    if model.i_value_eV is not None and model.i_value_eV != i_value_eV:
        raise ValueError(
            f"density effect {model.name!r} is valid for I = {model.i_value_eV:g} eV only, "
            f"not {i_value_eV:g} eV"
        )
    return model


class Bethe(MassStoppingPowerModel):
    """Bethe electronic stopping power of liquid water, Z 1-18 (module docstring)."""

    name: ClassVar[str] = "bethe"

    def __init__(
        self,
        *,
        i_value_eV: float = 78.0,
        density_effect: DensityEffectName | DensityEffectModel = "tabulated",
        shell: bool | WaterShellCorrection = False,
        barkas: bool | WaterBarkasCorrection = False,
        bloch: bool = False,
        mott: bool = False,
        lindhard_sorensen: bool = False,
        ahlen: bool = False,
        effective_charge: EffectiveChargeName | EffectiveChargeModel | None = "hubert",
        min_beta_gamma: float = PDG_BETA_GAMMA_RANGE[0],
        max_beta_gamma: float = PDG_BETA_GAMMA_RANGE[1],
        quantity: Literal["electronic", "total"] = "electronic",
        nuclear: WaterNuclearStopping | None = None,
        density_g_cm3: float = WATER_DENSITY_G_CM3,
    ) -> None:
        if not i_value_eV > 0.0:
            raise ValueError(f"i_value_eV must be > 0, got {i_value_eV!r}")
        if quantity not in ("electronic", "total"):
            raise ValueError(f"quantity must be 'electronic' or 'total', got {quantity!r}")
        self.quantity = quantity
        self.nuclear = nuclear or WaterNuclearStopping()
        if not 0.0 < min_beta_gamma < max_beta_gamma:
            raise ValueError("need 0 < min_beta_gamma < max_beta_gamma")
        if lindhard_sorensen and (bloch or mott or ahlen):
            raise ValueError("lindhard_sorensen replaces bloch and mott; do not combine them")
        if ahlen and mott:
            raise ValueError("ahlen replaces the linear mott term; do not combine them")
        self.i_value_eV = float(i_value_eV)
        self.effective_charge = _effective_charge(effective_charge)
        self.density_effect = _density_effect(density_effect, self.i_value_eV, density_g_cm3)
        self.shell = WaterShellCorrection() if shell is True else (shell or None)
        self.barkas = WaterBarkasCorrection() if barkas is True else (barkas or None)
        self.corrections = {
            "shell": self.shell is not None,
            "barkas": self.barkas is not None,
            "bloch": bloch,
            "mott": mott,
            "lindhard_sorensen": lindhard_sorensen,
            "ahlen": ahlen,
        }
        lo_d, hi_d = self.density_effect.beta_gamma_range
        lo, hi = max(min_beta_gamma, lo_d), min(max_beta_gamma, hi_d)
        for corr in (self.shell, self.barkas):
            if corr is not None:
                b2_lo, b2_hi = corr.beta2_range()
                lo = max(lo, math.sqrt(b2_lo / (1.0 - b2_lo)))
                if b2_hi < 1.0:
                    hi = min(hi, math.sqrt(b2_hi / (1.0 - b2_hi)))
        if not lo < hi:
            raise ValueError("the corrections leave an empty beta*gamma range")
        self.beta_gamma_range = (lo, hi)
        self.density_g_cm3 = density_g_cm3

    def supported_Z(self) -> tuple[int, ...]:
        return tuple(range(1, 19))

    def energy_range(self, Z: int, A: int) -> tuple[float, float]:
        self._require_supported(Z)
        per_nucleon = nuclear_mass_MeV(Z, A) / A
        lo, hi = self.beta_gamma_range
        gamma_hi = math.sqrt(1.0 + hi * hi)
        if self.corrections["lindhard_sorensen"]:
            gamma_hi = min(gamma_hi, LS_FINITE_SIZE_FRACTION * 160.0 / A ** (1.0 / 3.0))
        return per_nucleon * (math.sqrt(1.0 + lo * lo) - 1.0), per_nucleon * (gamma_hi - 1.0)

    def _evaluate(
        self, Z: int, A: int, energy_MeV_u: npt.NDArray[np.float64]
    ) -> npt.NDArray[np.float64]:
        model = self.effective_charge
        if model is None or Z < 3:
            out = self._bare(Z, A, energy_MeV_u)
        else:
            mass = nuclear_mass_MeV(Z, A)
            gamma = 1.0 + energy_MeV_u * A / mass
            beta = np.sqrt(1.0 - 1.0 / (gamma * gamma))
            z_eff = model.z_eff(Z, beta)
            if model.reference_Z == 2:
                e_alpha = (gamma - 1.0) * nuclear_mass_MeV(2, 4) / 4.0
                out = (z_eff / 2.0) ** 2 * self._bare(2, 4, e_alpha)
            else:
                out = (z_eff / Z) ** 2 * self._bare(Z, A, energy_MeV_u)
        if self.quantity == "total":
            out = out + self.nuclear.mass_stopping_power(Z, A, energy_MeV_u)
        return out

    def _bare(
        self, Z: int, A: int, energy_MeV_u: npt.NDArray[np.float64]
    ) -> npt.NDArray[np.float64]:
        """Electronic stopping of the bare ion (charge Z) with this instance's corrections."""
        mass = nuclear_mass_MeV(Z, A)
        gamma = 1.0 + energy_MeV_u * A / mass
        beta2 = 1.0 - 1.0 / (gamma * gamma)
        bg2 = beta2 * gamma * gamma
        ratio = ELECTRON_MASS_MEV / mass
        w_max = 2.0 * ELECTRON_MASS_MEV * bg2 / (1.0 + 2.0 * gamma * ratio + ratio * ratio)
        i_mev = self.i_value_eV * 1e-6
        log_term = 0.5 * np.log(2.0 * ELECTRON_MASS_MEV * bg2 * w_max / (i_mev * i_mev))
        delta = self.density_effect.delta(np.sqrt(bg2))
        bracket = log_term - beta2 - 0.5 * delta
        beta = np.sqrt(beta2)
        if self.shell is not None:
            bracket = bracket - self.shell.c_over_z(beta2)
        if self.barkas is not None:
            bracket = bracket + float(Z) * self.barkas.l1(beta2)
        if self.corrections["bloch"]:
            bracket = bracket + bloch_correction(Z, beta)
        if self.corrections["mott"]:
            bracket = bracket + mott_correction(Z, beta)
        if self.corrections["lindhard_sorensen"]:
            bracket = bracket + lindhard_sorensen_correction(Z, gamma)
        if self.corrections["ahlen"]:
            bracket = bracket + ahlen_mott_correction(Z, beta)
        out: npt.NDArray[np.float64] = (
            BETHE_K_MEV_CM2_MOL * WATER_Z_OVER_A * float(Z * Z) / beta2 * bracket
        )
        return out

    def metadata(self, Z: int, A: int) -> IonSourceInfo:
        lo, hi = self.energy_range(Z, A)
        options: dict[str, OptionValue] = {
            "quantity": self.quantity,
            "nuclear": None if self.quantity == "electronic" else self.nuclear.describe(),
            "density_effect": self.density_effect.name,
            "projectile_charge": (
                "bare (z = Z)"
                if self.effective_charge is None or Z < 3
                else f"effective: {self.effective_charge.name}"
            ),
            "beta_gamma_range": list(self.beta_gamma_range),  # type: ignore[dict-item]
            **self.corrections,
        }
        if self.shell is not None:
            options["shell"] = self.shell.describe()
        if self.barkas is not None:
            options["barkas"] = self.barkas.describe()
        return IonSourceInfo(
            Z=Z,
            A=A,
            source="Bethe",
            reference=BETHE_REFERENCE,
            i_value_eV=self.i_value_eV,
            energy_range_MeV_u=(lo, hi),
            edition="PDG 2024",
            options=options,
        )


__all__ = [
    "BETHE_REFERENCE",
    "LS_FINITE_SIZE_FRACTION",
    "PDG_BETA_GAMMA_RANGE",
    "Bethe",
    "DensityEffectModel",
    "DensityEffectName",
    "EffectiveChargeName",
    "NoDensityEffect",
    "SternheimerDensityEffect",
    "SternheimerPeierlsDensityEffect",
    "TabulatedDensityEffect",
]
