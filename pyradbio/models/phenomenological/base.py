"""The RBEmin/RBEmax skeleton shared by the phenomenological LET-based models.

Carabe-Fernandez, Dale & Jones 2007 (doi:10.1080/09553000601087176) introduced
writing an ion's LQ parameters as the photon ones scaled by the two asymptotic
limits of the RBE-vs-dose curve:

```text
RBE_max = alpha_ion / alpha_x         (the D -> 0 limit)
RBE_min = sqrt(beta_ion / beta_x)     (the D -> inf limit)
```

which inverts to the mapping every model in this package uses:

```text
alpha_ion = RBE_max * alpha_x
beta_ion  = RBE_min**2 * beta_x
```

Everything else -- survival, RBE at any dose, mixed fields -- then follows from
the LQ algebra that ``pyradbio.core`` already owns. A published model therefore
contributes exactly two closed forms in ``LET_d`` and ``(alpha/beta)_x``, and
nothing else.

Every model is an [`LETModel`][]. The LET route --
``alpha_beta_from_let(let)``, ``survival_from_let(let, D)``,
``rbe_at_dose_from_let(let, D)``, ``rbe_at_survival_from_let(let, S)`` and
``rbe_at_reference_dose_from_let(let, d_L)`` -- needs no stopping power; the
ion/field interface of [`BiophysicalModel`][pyradbio.models.base.BiophysicalModel]
(``survival(ion, D)``, ``rbe_at_dose(ion, D)``, ...) reduces the field to its
``LET_d`` with ``stopping_power`` first.

Coefficient provenance differs per model and is uneven -- read
``docs/references/Phenomenological_RBE_models.md`` before trusting one.
"""

from __future__ import annotations

import warnings
from abc import abstractmethod
from collections.abc import Sequence
from dataclasses import dataclass
from typing import ClassVar

from pyradbio.core.lq import LQParameters
from pyradbio.core.mixing import DEFAULT_SUB_NODES, Spacing
from pyradbio.core.photon import PhotonParameters
from pyradbio.core.radiation import Ion, RadiationField
from pyradbio.models.base import LETModel, ModelCapabilities
from pyradbio.track.stopping_power import StoppingPower

# Z of the ions each model was fitted for, keyed by the usual mode names.
_MODE_Z: dict[str, int] = {"protons": 1, "helium": 2, "carbon": 6}


@dataclass(frozen=True, slots=True)
class RBEMinMax:
    """The two asymptotic RBE limits at one LET.

    ``rbe_min`` is **not** constrained to be <= ``rbe_max``: for Carabe et al.
    2012 at high ``(alpha/beta)_x`` and low LET the D -> inf limit genuinely sits
    above the D -> 0 one, which is why that model predicts RBE *rising* with dose
    (McNamara et al. 2015, sec. 3.2). Ordering must not be assumed anywhere.
    """

    rbe_min: float
    rbe_max: float

    def __post_init__(self) -> None:
        if self.rbe_min < 0.0:
            raise ValueError(f"rbe_min must be >= 0, got {self.rbe_min!r}")
        if self.rbe_max < 0.0:
            raise ValueError(f"rbe_max must be >= 0, got {self.rbe_max!r}")


@dataclass(frozen=True, slots=True)
class RBEMinMaxModel(LETModel):
    """Base class for a LET-based RBE parameterisation.

    Subclasses implement [`rbe_min_max`][] and declare their provenance and
    fitted validity domain as class attributes. The photon reference is model
    *state* (unlike the ``pO2``/``g`` axes, which are irradiation environment):
    these models are parameterised **by** the tissue through ``(alpha/beta)_x``,
    so a model instance is meaningless without one.

    ``stopping_power`` is only needed for the [`Ion`][] / [`RadiationField`][]
    entry points, which must convert a field to a dose-averaged LET first. Working
    directly from a LET map -- what a treatment-planning system actually has --
    needs no stopping power at all, so it stays optional.
    """

    capabilities: ClassVar[ModelCapabilities] = ModelCapabilities(let_route=True)
    """A [`LETModel`][]: closed forms in ``LET_d``. The
    ion and field routes need a ``stopping_power`` to reduce to ``LET_d``.
    """

    photon: PhotonParameters
    stopping_power: StoppingPower | None = None

    model_id: ClassVar[str]
    """Short identifier, matching the usual three-letter tag for the model."""
    reference_doi: ClassVar[str]
    """DOI of the paper the coefficients come from."""
    reference_citation: ClassVar[str] = ""
    """Full citation, for a source that has no DOI. Only a handful of published
    models live outside the DOI system -- ``Peeler`` is a PhD dissertation --
    and putting its title in ``reference_doi`` would make that field lie.
    Exactly one of the two must be set; [`reference_label`][] picks.
    """
    radiation_modes: ClassVar[tuple[str, ...]] = ("protons",)
    """Radiation modes the fit covers; other ions are rejected, not extrapolated."""
    valid_let_keV_um: ClassVar[tuple[float, float] | None] = None
    """Fitted LET_d range [keV/um], or None if the paper states none."""
    valid_alpha_beta_ratio_Gy: ClassVar[tuple[float, float] | None] = None
    """Fitted (alpha/beta)_x range [Gy], or None if the paper states none."""

    @classmethod
    def reference_label(cls) -> str:
        """How this model cites itself in a warning: a DOI, or a full citation."""
        return f"doi:{cls.reference_doi}" if cls.reference_doi else cls.reference_citation

    def __post_init__(self) -> None:
        if not self.reference_doi and not self.reference_citation:
            raise ValueError(
                f"{type(self).__name__} must declare a reference_doi or, for a source "
                "outside the DOI system, a reference_citation"
            )
        if self.photon.beta_x <= 0.0:
            raise ValueError(
                "beta_x must be > 0: these models are parameterised by the tissue "
                f"ratio (alpha/beta)_x, which is undefined for {self.photon!r}"
            )

    # -- the tissue parameter ------------------------------------------------

    @property
    def alpha_beta_ratio_Gy(self) -> float:
        """``(alpha/beta)_x`` [Gy] -- the models' one tissue-specific input."""
        return self.photon.alpha_x / self.photon.beta_x

    # -- what a subclass must supply ----------------------------------------

    @abstractmethod
    def _limits(self, let_keV_um: float) -> RBEMinMax:
        """The model's own two closed forms. Called only via [`rbe_min_max`][]."""

    def rbe_min_max(self, let_keV_um: float) -> RBEMinMax:
        """The two asymptotic RBE limits at ``let_keV_um`` (dose-averaged LET).

        A template method: validation and the fitted-domain check happen here, so
        no subclass can forget them, and `_limits` stays a pure statement of
        the paper's algebra.
        """
        self._check_domain(let_keV_um)
        return self._limits(let_keV_um)

    # -- shared machinery ----------------------------------------------------

    def _check_domain(self, let_keV_um: float) -> None:
        """Warn (never clamp) when asked outside the paper's fitted range.

        Clamping would silently manufacture a value the paper does not support;
        warning leaves the caller's number intact and visible.
        """
        if let_keV_um < 0.0:
            raise ValueError(f"LET must be >= 0 keV/um, got {let_keV_um!r}")
        if self.valid_let_keV_um is not None:
            lo, hi = self.valid_let_keV_um
            if not lo <= let_keV_um <= hi:
                warnings.warn(
                    f"{self.model_id}: LET_d = {let_keV_um:g} keV/um is outside the "
                    f"fitted range [{lo:g}, {hi:g}] keV/um ({self.reference_label()}); "
                    "the value returned is an extrapolation",
                    UserWarning,
                    stacklevel=3,
                )
        if self.valid_alpha_beta_ratio_Gy is not None:
            lo, hi = self.valid_alpha_beta_ratio_Gy
            ratio = self.alpha_beta_ratio_Gy
            if not lo <= ratio <= hi:
                warnings.warn(
                    f"{self.model_id}: (alpha/beta)_x = {ratio:g} Gy is outside the "
                    f"fitted range [{lo:g}, {hi:g}] Gy ({self.reference_label()}); "
                    "the value returned is an extrapolation",
                    UserWarning,
                    stacklevel=3,
                )

    def alpha_beta_from_let(self, let_keV_um: float) -> LQParameters:
        """LQ parameters at a dose-averaged LET, via ``alpha = RBE_max*alpha_x``.

        This is the model's natural entry point: a TPS holds a LET map, not an
        ion spectrum.
        """
        limits = self.rbe_min_max(let_keV_um)
        return LQParameters(
            alpha=limits.rbe_max * self.photon.alpha_x,
            beta=limits.rbe_min * limits.rbe_min * self.photon.beta_x,
        )

    def reference(self) -> PhotonParameters:
        """The photon reference [`photon`][] these fits are parameterised by."""
        return self.photon

    def rbe_at_dose(self, field: Ion | RadiationField, dose_Gy: float) -> float:
        """RBE at a given ion dose of ``field``, through its ``LET_d``.

        These models are fits in one ``LET_d``, so a field is reduced to its
        dose-averaged LET first ([`let_d_keV_um`][]) and
        [`rbe_at_dose_from_let`][] is evaluated there.
        """
        return self.rbe_at_dose_from_let(self.let_d_keV_um(field), dose_Gy)

    def rbe_at_dose_from_let(self, let_keV_um: float, dose_Gy: float) -> float:
        """RBE at a given **ion** dose: ``D_photon(S) / D_ion`` with ``S = S(D_ion)``.

        For a pure LQ reference this is the family's published closed form
        (McNamara et al. 2015, doi:10.1088/0031-9155/60/21/8399, eq. 2,
        with ``r = (alpha/beta)_x``):

        ```text
        RBE = [sqrt(r^2 + 4 r RBE_max D + 4 RBE_min^2 D^2) - r] / (2 D)
        ```

        Evaluate on log-kill through the shared LET route to avoid survival
        underflow. If the caller supplies ``photon.D_t``, invert that reference's
        exponential tail, consistently with iso-survival RBE. That tail is an
        explicit reference extension, not part of the published LQ fit; the
        high-dose ``RBE_min`` limit applies only when ``D_t is None``.
        """
        return LETModel.rbe_at_dose_from_let(self, let_keV_um, dose_Gy)

    def rbe_at_reference_dose_from_let(self, let_keV_um: float, reference_dose_Gy: float) -> float:
        """RBE at a given **reference (photon)** dose per fraction ``d_L``.

        With a pure LQ reference, Carabe-Fernandez, Dale & Jones 2007 eq. (5) --
        the form the family's originating paper uses, and the one tabulated tissue data plugs into
        directly, since ``(alpha/beta)`` and dose per fraction are what clinical
        practice records:

        ```text
        RBE = [r RBE_max + sqrt(r^2 RBE_max^2 + 4 d_L RBE_min^2 (r + d_L))]
              / (2 (r + d_L))
        ```

        **This is a different function from** [`rbe_at_dose_from_let`][], which takes the
        *ion* dose. Both are called "RBE at 2 Gy" in the literature and they are
        not equal at the same numeric argument -- they agree only on matching
        pairs, ``d_L = RBE * D_ion``. Keeping both named explicitly is cheaper
        than debugging a silent factor later; a test pins the correspondence.

        For a pure LQ reference, ``d_L -> 0`` gives ``RBE_max`` and
        ``d_L -> inf`` gives ``RBE_min``. A supplied ``photon.D_t`` instead
        shapes the reference effect, just as in [`rbe_at_dose_from_let`][].
        """
        if reference_dose_Gy <= 0.0:
            raise ValueError(f"reference dose must be > 0 Gy, got {reference_dose_Gy!r}")
        effect = self.reference().log_kill(reference_dose_Gy)
        lq = self.alpha_beta_from_let(let_keV_um)
        ion_curve = PhotonParameters(alpha_x=lq.alpha, beta_x=lq.beta)
        return reference_dose_Gy / ion_curve.dose_for_log_kill(effect)

    # -- field entry points --------------------------------------------------

    def _stopping_power(self) -> StoppingPower:
        if self.stopping_power is None:
            raise ValueError(
                f"{self.model_id} needs a stopping_power to turn a radiation field into "
                "a dose-averaged LET; pass one, or call alpha_beta_from_let() directly"
            )
        return self.stopping_power

    @classmethod
    def applicable_z(cls) -> frozenset[int]:
        """Atomic numbers this fit covers, from its declared radiation modes.

        Each of these papers fits one ion species; other ions are rejected rather
        than extrapolated. Exposed because callers that do not go through
        [`alpha_beta`][pyradbio.models.phenomenological.base.RBEMinMaxModel.alpha_beta] -- a
        benchmark feeding LET directly, say -- still need to know what the fit covers.
        """
        return frozenset(_MODE_Z[mode] for mode in cls.radiation_modes)

    def _check_mode(self, ions: Sequence[Ion]) -> None:
        allowed = self.applicable_z()
        for ion in ions:
            if ion.Z not in allowed:
                modes = "/".join(self.radiation_modes)
                raise ValueError(
                    f"{self.model_id} was fitted for {modes} only ({self.reference_label()}); "
                    f"got Z={ion.Z}"
                )

    def let_d_keV_um(
        self,
        field: RadiationField | Ion,
        *,
        sub_nodes: int = DEFAULT_SUB_NODES,
        spacing: Spacing = "log",
    ) -> float:
        """Dose-averaged LET of a field: ``sum(d_i L_i)/sum(d_i)``.

        The model is evaluated **after** this reduction, never per component and
        then averaged in ``alpha``. That ordering is the definition of these
        models -- they are fits to RBE as a function of one ``LET_d`` -- and it
        matters because ``RBE_max`` is non-linear in LET for McNamara (through
        ``RBE_min``) and Mairani. A binned component contributes
        ``int phi(E) S(E)^2 dE / int phi(E) S(E) dE`` over its bin, integrated
        on a ``sub_nodes`` sub-grid (see [`pyradbio.core.mixing.dose_quadrature`][]).
        """
        if isinstance(field, Ion):
            stopping_power = self._stopping_power()
            self._check_mode([field])
            return stopping_power.let_keV_um(field)
        self._stopping_power()  # a field always needs one: fail before any work
        self._check_mode([component.ion for component in field.components])
        (let_d,) = self._mixed_quantities(field, sub_nodes=sub_nodes, spacing=spacing)
        return let_d

    # -- BiophysicalModel hooks: these models mix on LET_d --------------------

    def _alpha_beta_ion(self, ion: Ion) -> LQParameters:
        """LQ parameters for one ion, via its LET."""
        return self.alpha_beta_from_let(self.let_d_keV_um(ion))

    def _mixing_quantities(self, ion: Ion) -> tuple[float, ...]:
        return (self.let_d_keV_um(ion),)

    def _lq_from_mixed(self, mixed: tuple[float, ...]) -> LQParameters:
        (let_d,) = mixed
        return self.alpha_beta_from_let(let_d)

    def _dose_weight_stopping_power(self, ion: Ion) -> float:
        return self._stopping_power().let_keV_um(ion)
