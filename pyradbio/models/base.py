"""Common model interface.

[`BiophysicalModel`][] is the one interface every model family implements.
A model supplies its per-ion physics (`_alpha_beta_ion`)
and, where it differs from the LQ default, its mixing rule; the base class
expands a [`RadiationField`][] into dose-weighted
``(ion, dose)`` nodes with [`dose_quadrature`][] --
integrating each binned component over a log sub-grid, never at its bin centre
(AGENTS.md, "Spectral / intra-bin integration") -- and dose-averages the rule's
quantities over them.

On top of ``alpha_beta`` the base class defines the survival and RBE interface,
with LQ defaults that non-LQ models override:

==========================================  ========================================
``alpha_beta(field)``                       LQ parameters of an ion or a mixed field
``survival(field, dose_Gy)``                surviving fraction (scalar or array dose)
``dose_for_survival(field, S)``             the dose reaching survival ``S``
``reference()``                             the photon reference (raises if undefined)
``rbe_at_survival(field, S)``               iso-survival RBE, ``D_ref(S) / D(S)``
``rbe_at_dose(field, dose_Gy)``             iso-effect RBE at a given ion dose
``alpha_beta_table(ions, energies)``        an [`AlphaBetaTable`][]
==========================================  ========================================

Model-specific axes -- SMK's nucleus stochastics, OSMK's ``pO2``, IMK's dose
schedule -- extend these with keyword-only arguments; they never change the
positional signature. [`BiophysicalModel.capabilities`][] declares which
routes a family implements, so callers can branch without provoking errors.
"""

from __future__ import annotations

import math
from abc import ABC, abstractmethod
from collections.abc import Callable, Sequence
from dataclasses import dataclass, replace
from typing import TYPE_CHECKING, Any, ClassVar, TypeVar, overload

import numpy as np
import numpy.typing as npt
from scipy.optimize import brentq

from pyradbio.core.lq import LQParameters
from pyradbio.core.lq import dose_for_survival as _lq_dose_for_survival
from pyradbio.core.lq import log_kill as _lq_log_kill
from pyradbio.core.lq import survival as _lq_survival
from pyradbio.core.mixing import DEFAULT_SUB_NODES, Spacing, dose_average, dose_quadrature
from pyradbio.core.photon import PhotonParameters
from pyradbio.core.radiation import Ion, RadiationField

if TYPE_CHECKING:
    from pyradbio.tables.alpha_beta import AlphaBetaTable


class PhotonReferenceError(ValueError):
    """Raised by [`BiophysicalModel.reference`][] when a model has no photon reference.

    A ``ValueError`` because the cause is configuration (no reference was
    supplied, or the model does not define one), not a bug.
    """


@dataclass(frozen=True, slots=True)
class ModelCapabilities:
    """What a model family implements -- declared, so callers can branch on it.

    A statement about the *implementation*, not a per-instance switch, so it is
    a ``ClassVar`` on every family (``capabilities``). The benchmark harness
    reads it (and [`LETModel`][]) instead of keeping its own flags.
    """

    ion_route: bool = True
    """``alpha_beta(Ion)`` is implemented (False only for an assembly-only model)."""
    mixed_field: bool = True
    """``alpha_beta(RadiationField)`` mixes a field; False means it raises."""
    lq_exact: bool = True
    """``survival`` is exactly ``exp(-alpha D - beta D^2)`` of ``alpha_beta``. False
    for models whose survival is not LQ (SMK's ``1 + gamma D`` factor, Katz's
    multi-target shoulder); their ``alpha_beta`` is then a descriptor, not the curve.
    """
    let_route: bool = False
    """``alpha_beta_from_let`` is implemented -- the model is a [`LETModel`][]."""
    spectrum_route: bool = False
    """``alpha_beta_from_spectrum`` / ``alpha_beta_from_dose_spectrum`` take a
    lineal-energy spectrum ``d(y)`` directly.
    """


_P = TypeVar("_P")


def apply_preset(cls: type[_P], preset: _P, overrides: dict[str, Any]) -> _P:
    """Return ``replace(preset, **overrides)``, refusing a preset of another class.

    Preset classmethods are inherited, so e.g. ``IMKModel.hsg_mk_inaniwa2018()``
    would otherwise quietly return the parent's ``ModifiedMKM`` -- or, built as
    ``cls(...)``, an IMK carrying the parent's fitted parameters. Both are wrong,
    so a preset only applies to the exact class it was fitted for.
    """
    if type(preset) is not cls:
        raise TypeError(
            f"this preset is a {type(preset).__name__} parameter set; call it on "
            f"{type(preset).__name__}, not on {cls.__name__}"
        )
    return replace(preset, **overrides)  # type: ignore[type-var]


def check_choice(name: str, value: object, choices: tuple[object, ...]) -> None:
    """Raise ``ValueError`` unless ``value`` is one of ``choices``.

    ``choices`` is normally ``typing.get_args`` of the field's ``Literal``, so the
    annotation is the single source of the allowed values.
    """
    if value not in choices:
        raise ValueError(f"{name} must be one of {choices!r}, got {value!r}")


def invert_survival_curve(
    survival_fraction: float, curve: Callable[[float], float], *, start_Gy: float = 1.0
) -> float:
    """Dose at which a monotonically falling survival ``curve`` reaches ``survival_fraction``.

    For survival curves without a closed-form inverse (SMK's eq. 24, Katz's
    multi-target product). Doubles an upper bracket from ``start_Gy`` until the
    curve drops below the target, then solves on the curve itself.
    """
    if not 0.0 < survival_fraction <= 1.0:
        raise ValueError(f"survival fraction must be in (0, 1], got {survival_fraction!r}")

    def excess(dose: float) -> float:
        return curve(dose) - survival_fraction

    hi = start_Gy
    for _ in range(60):  # expand until the curve drops below the target
        if excess(hi) < 0.0:
            break
        hi *= 2.0
    else:  # pragma: no cover -- only reachable for a non-decreasing curve
        raise ValueError("survival never reaches the target: check the model or dose range")
    return float(brentq(excess, 0.0, hi, xtol=1.0e-12, rtol=1.0e-14))


class BiophysicalModel(ABC):
    """A model maps a radiation field + cell to LQ parameters, survival and RBE.

    Subclasses implement `_alpha_beta_ion` and declare
    [`capabilities`][]. The mixing rule is the pair `_mixing_quantities`
    / `_lq_from_mixed`: the per-ion quantities that are dose-averaged, and
    how the averaged quantities map back to LQ parameters. The default is the
    Zaider-Rossi / Kanai rule (dose-average ``alpha`` and ``sqrt(beta)``); the
    MKM mixes on ``z*_1D`` and the LET-based phenomenological models on the LET.

    Survival and RBE default to the LQ algebra on [`alpha_beta`][] and the
    photon reference of [`reference`][]. A model whose survival is not LQ
    (``capabilities.lq_exact`` False) overrides [`survival`][],
    [`dose_for_survival`][] and `_log_kill`; a model whose photon
    reference is not LQ overrides `_reference_dose_for_log_kill`.
    """

    capabilities: ClassVar[ModelCapabilities]
    """What this family implements; see [`ModelCapabilities`][]."""

    # ---- LQ parameters ------------------------------------------------------
    def alpha_beta(
        self,
        field: RadiationField | Ion,
        *,
        sub_nodes: int = DEFAULT_SUB_NODES,
        spacing: Spacing = "log",
    ) -> LQParameters:
        """LQ parameters for a single ion or a mixed field.

        An [`Ion`][] goes straight to the model's per-ion physics. A
        [`RadiationField`][] is expanded by
        [`dose_quadrature`][]: each mono-energetic
        component is one node, each binned component ``sub_nodes`` nodes on a
        ``spacing`` sub-grid with dose weights ``phi(E) S(E)`` (``S`` from
        `_dose_weight_stopping_power`). The mixing quantities are
        dose-averaged over all nodes, with each distinct ion evaluated once.
        ``sub_nodes``/``spacing`` have no effect on mono-energetic components.
        """
        if isinstance(field, Ion):
            return self._alpha_beta_ion(field)
        return self._alpha_beta_field(field, sub_nodes=sub_nodes, spacing=spacing)

    @abstractmethod
    def _alpha_beta_ion(self, ion: Ion) -> LQParameters:
        """LQ parameters for one mono-energetic ion -- the per-model physics."""

    def _alpha_beta_field(
        self, field: RadiationField, *, sub_nodes: int, spacing: Spacing
    ) -> LQParameters:
        """LQ parameters for a field. Override only to refuse mixed fields."""
        return self._lq_from_mixed(
            self._mixed_quantities(field, sub_nodes=sub_nodes, spacing=spacing)
        )

    def _mixed_quantities(
        self, field: RadiationField, *, sub_nodes: int, spacing: Spacing
    ) -> tuple[float, ...]:
        """Dose-average `_mixing_quantities` over the field's quadrature nodes."""
        nodes = dose_quadrature(
            field, self._dose_weight_stopping_power, sub_nodes=sub_nodes, spacing=spacing
        )
        cache: dict[Ion, tuple[float, ...]] = {}
        rows = []
        for ion, _dose in nodes:
            if ion not in cache:
                cache[ion] = self._mixing_quantities(ion)
            rows.append(cache[ion])
        doses = [dose for _ion, dose in nodes]
        return tuple(dose_average(column, doses) for column in zip(*rows, strict=True))

    def _mixing_quantities(self, ion: Ion) -> tuple[float, ...]:
        """Per-ion quantities that mix linearly in dose. Default ``(alpha, sqrt(beta))``."""
        lq = self._alpha_beta_ion(ion)
        return (lq.alpha, math.sqrt(lq.beta))

    def _lq_from_mixed(self, mixed: tuple[float, ...]) -> LQParameters:
        """Map dose-averaged `_mixing_quantities` back to LQ parameters."""
        alpha, sqrt_beta = mixed
        return LQParameters(alpha=alpha, beta=sqrt_beta * sqrt_beta)

    def _dose_weight_stopping_power(self, ion: Ion) -> float:
        """LET [keV/um] that converts fluence to dose weights in a mixed field.

        Only called for fluence-specified components. The default refuses, so
        a model without a stopping power still mixes dose-specified fields.
        """
        raise NotImplementedError(
            f"{type(self).__name__} has no stopping power to convert fluence to "
            "dose; specify mixed-field components by dose_Gy instead of fluence."
        )

    # ---- survival -------------------------------------------------------------
    @overload
    def survival(self, field: Ion | RadiationField, dose_Gy: float) -> float: ...

    @overload
    def survival(
        self, field: Ion | RadiationField, dose_Gy: npt.NDArray[np.float64]
    ) -> npt.NDArray[np.float64]: ...

    def survival(
        self, field: Ion | RadiationField, dose_Gy: float | npt.NDArray[np.float64]
    ) -> float | npt.NDArray[np.float64]:
        """Surviving fraction after ``dose_Gy`` of ``field`` (scalar or array dose).

        LQ default: ``exp(-alpha D - beta D^2)`` with [`alpha_beta`][].
        Models whose survival is not LQ override this.
        """
        return lq_survival_curve(dose_Gy, self.alpha_beta(field))

    def dose_for_survival(self, field: Ion | RadiationField, survival_fraction: float) -> float:
        """Dose [Gy] of ``field`` that reaches ``survival_fraction``.

        LQ default: the closed-form root of [`pyradbio.core.lq.dose_for_survival`][].
        """
        lq = self.alpha_beta(field)
        return _lq_dose_for_survival(survival_fraction, lq.alpha, lq.beta)

    def _log_kill(self, field: Ion | RadiationField, dose_Gy: float) -> float:
        """``-ln S`` after ``dose_Gy``, formed without exponentiating (no underflow).

        LQ default. A model overriding [`survival`][] overrides this too.
        """
        lq = self.alpha_beta(field)
        return _lq_log_kill(dose_Gy, lq.alpha, lq.beta)

    # ---- photon reference and RBE ---------------------------------------------
    def reference(self) -> PhotonParameters:
        """The photon reference RBE is computed against.

        The default raises [`PhotonReferenceError`][]; families that carry a
        reference (``photon``, ``alpha_ref``/``beta_ref``, IMK's
        ``reference_z_star_1D_Gy``) return it here as
        [`PhotonParameters`][], whose ``D_t`` (if any)
        shapes the reference curve exactly as the model's own physics does.
        """
        raise PhotonReferenceError(
            f"{type(self).__name__} carries no photon reference, so RBE is undefined; "
            "supply one where the model accepts it"
        )

    def _reference_dose_for_log_kill(self, log_kill: float) -> float:
        """Reference dose [Gy] producing log-kill ``log_kill``; from [`reference`][]."""
        return self.reference().dose_for_log_kill(log_kill)

    def rbe_at_survival(self, field: Ion | RadiationField, survival_fraction: float) -> float:
        """Iso-survival RBE ``D_ref(S) / D(S)`` (``RBE_10`` at ``S = 0.1``)."""
        if not 0.0 < survival_fraction <= 1.0:
            raise ValueError(f"survival fraction must be in (0, 1], got {survival_fraction!r}")
        d_ref = self._reference_dose_for_log_kill(-math.log(survival_fraction))
        return d_ref / self.dose_for_survival(field, survival_fraction)

    def rbe_at_dose(self, field: Ion | RadiationField, dose_Gy: float) -> float:
        """RBE at a given **ion** dose: ``D_ref / D`` for the same effect ``-ln S``.

        Formed on the log-kill, never on ``S``, so it stays finite at doses where
        ``S`` underflows -- the ``D -> inf`` limit of an LQ model with a pure LQ
        reference is ``sqrt(beta / beta_x)`` exactly.
        """
        if dose_Gy <= 0.0:
            raise ValueError(f"dose must be > 0 Gy, got {dose_Gy!r}")
        return self._reference_dose_for_log_kill(self._log_kill(field, dose_Gy)) / dose_Gy

    def _rbe_at_survival_from_lq(self, lq: LQParameters, survival_fraction: float) -> float:
        """Iso-survival RBE of an explicit LQ pair against [`reference`][].

        The shared arithmetic of every LQ route (ion, LET, spectrum), so they
        cannot drift apart.
        """
        if not 0.0 < survival_fraction <= 1.0:
            raise ValueError(f"survival fraction must be in (0, 1], got {survival_fraction!r}")
        d_ref = self._reference_dose_for_log_kill(-math.log(survival_fraction))
        return d_ref / _lq_dose_for_survival(survival_fraction, lq.alpha, lq.beta)

    # ---- tables ---------------------------------------------------------------
    def alpha_beta_table(
        self, ions: Sequence[int], energies_MeV_u: Sequence[float]
    ) -> AlphaBetaTable:
        """Build an (ion, energy) -> (alpha, beta) table (see [`pyradbio.tables`][]).

        ``ions`` are atomic numbers ``Z``; each is resolved to its default
        isotope ([`pyradbio.core.elements.default_mass_number`][]) before
        delegating to [`pyradbio.tables.alpha_beta.build_alpha_beta_table`][],
        which evaluates this model over the grid. Every model returns the same
        [`AlphaBetaTable`][]; model-specific
        columns live under their own names (e.g.
        [`alpha_beta_z_star_table`][pyradbio.models.mkm.model.MKMModel.alpha_beta_z_star_table]).
        Imported lazily to avoid an import cycle with the tables layer.
        """
        from pyradbio.core.elements import default_mass_number
        from pyradbio.tables.alpha_beta import build_alpha_beta_table

        resolved = [Ion(Z=z, A=default_mass_number(z), energy_MeV_u=1.0) for z in ions]
        return build_alpha_beta_table(self, resolved, energies_MeV_u)


class LETModel(BiophysicalModel):
    """A model whose natural input is a dose-averaged LET rather than an ion.

    The phenomenological RBE family is exactly this: closed forms in ``LET_d``
    with no track structure to run, which is what a treatment-planning system
    holds. Subclasses implement [`alpha_beta_from_let`][]; the LET-route
    survival and RBE below mirror the ion-route interface with the LET in place
    of the field (``survival(ion, D)`` <-> ``survival_from_let(let, D)``).
    """

    @abstractmethod
    def alpha_beta_from_let(self, let_keV_um: float) -> LQParameters:
        """LQ parameters at a dose-averaged LET [keV/um]."""

    @overload
    def survival_from_let(self, let_keV_um: float, dose_Gy: float) -> float: ...

    @overload
    def survival_from_let(
        self, let_keV_um: float, dose_Gy: npt.NDArray[np.float64]
    ) -> npt.NDArray[np.float64]: ...

    def survival_from_let(
        self, let_keV_um: float, dose_Gy: float | npt.NDArray[np.float64]
    ) -> float | npt.NDArray[np.float64]:
        """LQ survival at a dose-averaged LET (scalar or array dose)."""
        return lq_survival_curve(dose_Gy, self.alpha_beta_from_let(let_keV_um))

    def dose_for_survival_from_let(self, let_keV_um: float, survival_fraction: float) -> float:
        """Dose [Gy] reaching ``survival_fraction`` at a dose-averaged LET."""
        lq = self.alpha_beta_from_let(let_keV_um)
        return _lq_dose_for_survival(survival_fraction, lq.alpha, lq.beta)

    def rbe_at_survival_from_let(self, let_keV_um: float, survival_fraction: float) -> float:
        """Iso-survival RBE ``D_ref(S) / D(S)`` at a dose-averaged LET."""
        return self._rbe_at_survival_from_lq(
            self.alpha_beta_from_let(let_keV_um), survival_fraction
        )

    def rbe_at_dose_from_let(self, let_keV_um: float, dose_Gy: float) -> float:
        """RBE at a given ion dose and dose-averaged LET.

        See [`rbe_at_dose`][pyradbio.models.base.LETModel.rbe_at_dose].
        """
        if dose_Gy <= 0.0:
            raise ValueError(f"dose must be > 0 Gy, got {dose_Gy!r}")
        lq = self.alpha_beta_from_let(let_keV_um)
        effect = _lq_log_kill(dose_Gy, lq.alpha, lq.beta)
        return self._reference_dose_for_log_kill(effect) / dose_Gy


@overload
def lq_survival_curve(dose_Gy: float, lq: LQParameters) -> float: ...


@overload
def lq_survival_curve(
    dose_Gy: npt.NDArray[np.float64], lq: LQParameters
) -> npt.NDArray[np.float64]: ...


def lq_survival_curve(
    dose_Gy: float | npt.NDArray[np.float64], lq: LQParameters
) -> float | npt.NDArray[np.float64]:
    """``exp(-alpha D - beta D^2)`` for a scalar dose (a float) or an array (an array)."""
    if np.ndim(dose_Gy) == 0:
        return _lq_survival(float(dose_Gy), lq.alpha, lq.beta)
    dose = np.asarray(dose_Gy, dtype=np.float64)
    if np.any(dose < 0.0):
        raise ValueError("dose must be >= 0")
    curve: npt.NDArray[np.float64] = np.exp(-(lq.alpha * dose + lq.beta * dose * dose))
    return curve
