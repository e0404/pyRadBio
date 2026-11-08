"""Model-vs-measurement benchmark harness.

Evaluates a pyRadBio model's predicted initial-slope ``alpha`` (and the derived
``RBE_alpha = alpha_ion / alpha_x``) against measured values, point by point,
and rolls the residuals up into the aggregate metrics from
[`pyradbio.benchmark.metrics`][].

The measurements come from any [`ReferenceDataset`][]
-- PIDE, RadPhysBio, or a user's own CSV (see [`pyradbio.benchmark.dataset`][]);
the harness reads only the source-agnostic
[`ReferenceRecord`][], so nothing below is tied
to one database. The prose in this module still says "PIDE" where it describes
that database's particular scope, because the caveats were written against it.

For the LEM family each PIDE record carries its own matched photon reference
(``alpha_x``/``beta_x``), so the model is (re)built per unique photon pair with a
shared, documented transition dose ``D_t`` and nuclear radius -- neither of which
PIDE tabulates. The MKM family is photon-independent and uses a single
``(alpha0, beta0, r_d, R_n)`` set. Both are coarse global choices (PIDE spans many
cell lines with different geometry / D_t), so the absolute numbers are indicative,
not a per-cell-line fit; the value is *relative* model ranking on the same points.

The phenomenological family adds nothing of its own: those models take the tissue
``(alpha/beta)_x`` and nothing else, and PIDE supplies it per row. **What a PIDE
comparison shows for them is narrower than it looks** -- McNamara et al. 2015 is
fitted to substantially what PIDE aggregates, so agreement there validates our
implementation of the published parameterisation rather than the model (partly so
for Wedenberg and Carabe, subsets of the same corpus). ``WilkensOelfke`` is a
further special case: its ``alpha_0`` is a V79 ion-response intercept, not a
function of the row's ``alpha_x``, so its prediction is the same V79 fit whatever
cell line the row describes.

Two evaluation routes exist, selected per model by [`ModelSpec`][]: the
track-structure families take an [`Ion`][] and derive LET themselves, while the
phenomenological family is a closed form in ``LET_d`` and is fed PIDE's own
tabulated LET.

Points the configured model cannot evaluate -- ions heavier than ``max_z``,
energies below ``min_energy_MeV_u``, or any model error -- are skipped and
counted, so a benchmark over the whole database still runs. Points
outside the *published fit's* own scope (wrong ion species, or a missing
``beta_x``) are counted separately, because that is a statement about the paper
rather than about the run.
"""

from __future__ import annotations

import warnings
from collections.abc import Callable, Iterable, Iterator, Sequence
from dataclasses import dataclass
from typing import TYPE_CHECKING, Literal, TypeVar

import numpy as np

from pyradbio.benchmark import metrics
from pyradbio.benchmark.dataset import ReferenceDataset, ReferenceRecord
from pyradbio.core.cell import CellTarget
from pyradbio.core.elements import default_mass_number
from pyradbio.core.lq import LQParameters
from pyradbio.core.photon import PhotonParameters
from pyradbio.core.radiation import Ion
from pyradbio.models.base import BiophysicalModel, LETModel

if TYPE_CHECKING:
    from pyradbio.models.phenomenological.base import RBEMinMaxModel

_T = TypeVar("_T")

LEM_MODELS = ("lem1", "lem2", "lem3", "lem4")
"""model name -> loose default D_t [Gy] is *not* attempted; D_t is a single shared
knob (see [`run_benchmark`][]). LEM class registry lives in the factory.
"""

MKM_MODELS = ("mkm", "mmkm", "smk", "mcf", "osmk2021", "osmk2023", "imk")
"""Benchmark name -> the MKM-family class and the published preset it runs under.
Each is a single cell line's fitted parameter set applied to every PIDE row --
the same coarse global choice the generic ``mkm`` entry already documents.
"""

RMF_MODELS = ("rmf",)
"""Benchmark name -> class in [`pyradbio.models.phenomenological`][]. These carry
no free parameters beyond the photon reference, which PIDE supplies per row, so
the registry adds nothing of its own: each is the published fit as printed.
The RMF model. A single entry, not a set of presets: its two tissue
parameters are *derived* from the row's own photon pair (Frese et al. 2012
eqs. 8-9), which PIDE supplies, so a preset would override the very thing the
model keys on.
"""

KATZ_MODELS = ("katz_hamster", "katz_hela", "katz_t1_todd", "katz_t1_barendsen")
"""Benchmark name -> a published Katz et al. 1971 cell-parameter set. Only the
**aerobic mammalian** fits are registered: PIDE aggregates aerobic mammalian
in-vitro survival, so the anoxic/N2 pairs, the bacterial spores and the
haploid yeast in the same table have no comparable rows. All ten remain
available on ``KatzCellularModel.published_presets()``.
"""

PHENOMENOLOGICAL_MODELS = (
    "wilkens",
    "carabe",
    "frese",
    "wedenberg",
    "mcnamara",
    "mairani",
    "tilly2",
    "tilly10",
    "chen",
    "jones",
    "roru",
    "peeler",
    "unkelbach",
)


@dataclass(frozen=True, slots=True)
class PointResult:
    """One evaluated PIDE point: model prediction vs measurement."""

    cell_line: str
    Z: int
    A: int
    energy_MeV_u: float | None
    """``None`` for a LET-based evaluation of a record whose source gives no
    usable target energy (see ``ReferenceRecord.energy_is_nominal``).
    """
    let_keV_um: float
    alpha_x: float
    predicted_alpha: float
    measured_alpha: float
    beta_x: float | None = None
    """The quadratic terms, filled by [`run_benchmark`][] (``None`` on a
    hand-built point): the row's photon ``beta_x``, the model's ``beta`` and the
    measured ``beta_ion`` [Gy^-2]. The scores never read them -- ``beta`` is
    a far noisier measurement than ``alpha`` -- but the per-point ratios
    ``beta / beta_x`` are what Pfuhl et al. 2022 (Fig. 6) plot next to RBE_alpha.
    """
    predicted_beta: float | None = None
    measured_beta: float | None = None

    @property
    def predicted_rbe_alpha(self) -> float:
        return self.predicted_alpha / self.alpha_x

    @property
    def measured_rbe_alpha(self) -> float:
        return self.measured_alpha / self.alpha_x

    @property
    def log_residual_alpha(self) -> float:
        """``log10(pred) - log10(meas)`` for this point (used in reports)."""
        import math

        return math.log10(self.predicted_alpha) - math.log10(self.measured_alpha)


@dataclass(frozen=True, slots=True)
class BenchmarkResult:
    """Aggregate outcome of a benchmark run over a set of reference points."""

    model_name: str
    points: tuple[PointResult, ...]
    n_skipped: int
    n_out_of_scope: int = 0
    """Points the model's own fit does not cover (wrong ion species, or no
    ``beta_x`` for a model parameterised by ``(alpha/beta)_x``). Kept apart
    from [`n_skipped`][] because a scope limit is a property of the
    published fit, not a failure to evaluate.
    """
    n_warnings: int = 0
    """``UserWarning``s raised by the model during evaluation -- in practice the
    out-of-fitted-domain warnings of the phenomenological family. They are
    captured rather than emitted: one per point would drown a full-PIDE run.
    Deliberately **not** every warning: pyMKM raises a scipy
    ``DeprecationWarning`` on each call, and counting those as domain warnings
    reported 1742 of them for a model that has no stated domain at all.
    """
    dataset_name: str = "PIDE"
    """Which reference dataset the points came from, and what its loader says a
    reader must know before believing a score against it (rendered by the
    report). Defaults describe the original PIDE-only harness.
    """
    dataset_caveats: tuple[str, ...] = ()
    attempted_by_ion: tuple[tuple[int, int], ...] = ()
    """``(Z, n)`` in ascending ``Z``: how many points of each ion passed every
    filter (and the ``limit_per_ion`` cap) and were handed to the model. A
    count above that ion's evaluated points means the model failed on the rest.
    """

    @property
    def n_points(self) -> int:
        return len(self.points)

    def _pred_meas(self) -> tuple[list[float], list[float]]:
        return (
            [p.predicted_alpha for p in self.points],
            [p.measured_alpha for p in self.points],
        )

    @property
    def smape_alpha(self) -> float:
        pred, meas = self._pred_meas()
        return metrics.smape(pred, meas)

    @property
    def mean_log_error_alpha(self) -> float:
        pred, meas = self._pred_meas()
        return metrics.mean_log_error(pred, meas)

    @property
    def mean_abs_log_error_alpha(self) -> float:
        pred, meas = self._pred_meas()
        return metrics.mean_absolute_log_error(pred, meas)

    def by_ion(self) -> dict[int, list[PointResult]]:
        """Points grouped by atomic number ``Z`` (for stratified residuals)."""
        groups: dict[int, list[PointResult]] = {}
        for p in self.points:
            groups.setdefault(p.Z, []).append(p)
        return groups


ModelFactory = Callable[[float, float], BiophysicalModel]
"""A factory maps a PIDE entry's photon reference to a ready-to-evaluate model."""

TransitionDoseRule = Callable[[float, float], float]
"""A per-row LEM transition dose: ``(alpha_x, beta_x) -> D_t`` [Gy], e.g. Pfuhl et
al. 2022 eq. (3) ``D_t = 4 Gy + 1.1 alpha_x/beta_x``. Passed as ``D_t`` to
[`build_model_spec`][] in place of one shared value.
"""

Stratify = Literal["let"]
"""How [`run_benchmark`][] thins each ion to ``limit_per_ion`` points."""


@dataclass(frozen=True, slots=True)
class ModelSpec:
    """How to build a benchmarkable model *and* how it must be evaluated.

    ``model_type`` selects the evaluation route: [`let_based`][] is derived
    from it (``issubclass(model_type, LETModel)``, equivalently
    ``capabilities.let_route``). Track-structure models (LEM, MKM)
    take an [`Ion`][] and derive LET internally; the phenomenological models
    ([`LETModel`][]) take a LET directly, and are fed
    **PIDE's own tabulated LET** rather than one
    recomputed from ``(Z, E)``. That is the faithful choice -- the published fits
    were made against the experiments' quoted LET -- and it also keeps our
    stopping-power tables out of a comparison that is not about them. The cost is
    that the two families do not share an x-axis definition; say so when
    reporting.

    ``applicable_z`` and ``needs_beta_x`` record what the *published fit* covers,
    so points outside it are tallied separately (see
    [`BenchmarkResult.n_out_of_scope`][]) instead of looking like errors.
    """

    build: ModelFactory
    model_type: type[BiophysicalModel] | None = None
    """The class ``build`` returns. Only used to route; ``None`` (the default)
    is the ion route.
    """
    applicable_z: frozenset[int] | None = None
    """Atomic numbers the fit covers, or ``None`` for "no species restriction"."""
    needs_beta_x: bool = False
    """True when the model is parameterised by ``(alpha/beta)_x`` and so cannot
    be evaluated on a PIDE row whose ``beta_x`` is missing or zero.
    """
    cache_by_ion: bool = False
    """Memoise predictions on ``(photon pair, Z, A, energy)``. PIDE repeats the
    same exposure across cell lines -- 892 evaluable rows hold only 520
    distinct ions -- which is worth ~40% of the run time for the spectrum MKM
    models, where one point costs tens of milliseconds. Safe only for the
    stateless models it is switched on for.
    """

    @property
    def let_based(self) -> bool:
        """Evaluated from the record's LET: the model is an [`LETModel`][]."""
        return self.model_type is not None and issubclass(self.model_type, LETModel)


_FAMILY_CAVEATS: dict[str, str] = {
    "lem": (
        "D_t = 30 Gy and R_n = 5 um are a single global choice across every cell "
        "line in PIDE; neither is tabulated there. The absolute residual is "
        "therefore indicative, and only the ranking on shared points is meant to "
        "carry weight."
    ),
    "mkm": (
        "Runs at one published cell line's fitted parameter set, applied to every "
        "PIDE row. The reference y_D it calibrates against belongs to a specific "
        "photon quality, while PIDE rows span 200 kVp, 60Co and MV; recalibrating "
        "per row would be a refit rather than a benchmark."
    ),
    "rmf": (
        "The reference radiation is fixed at Frese et al. 2012's Co-60 set "
        "(Sigma_x = 8.27 Gy^-1 Gbp^-1 at 0.36 keV/um), because PIDE records a "
        "row's alpha_x/beta_x but not the photon quality they were measured "
        "with. Rows measured with 200 kVp X-rays have Sigma_x = 8.68, a 5% "
        "difference that enters kappa through Sigma_x^2 and so moves beta ~10%."
    ),
    "katz": (
        "**alpha here is not an LQ initial slope.** Katz's model is not an LQ "
        "model at all: its shoulder comes from the multi-target exponent m, so "
        "the low-dose term goes as D^m, not D^2, and there is no quadratic "
        "coefficient. The reported alpha is eq. (14)'s *extrapolated* "
        "radiosensitivity k -- the high-dose slope Katz himself tabulates and "
        "compares against measured radiosensitivities. Against PIDE's fitted LQ "
        "alpha it will read high wherever the curve has a shoulder. Read this as "
        "a check on the implementation, not as a ranking against the LQ models."
    ),
    "phenomenological": (
        "**A PIDE comparison here shows less than it looks.** McNamara et al. 2015 "
        "is fitted to substantially what PIDE aggregates, so agreement validates "
        "this package's implementation of the published parameterisation, not the "
        "model's predictive correctness. The same holds, less totally, for "
        "Wedenberg and Carabe, which were fitted to subsets of the same corpus."
    ),
}
"""Model name -> what a reader must know before believing its score. Registered
here rather than in the report so that adding a model and stating its caveat
are one edit. [`model_caveats`][] merges the family note with the per-model
ones; [`pyradbio.benchmark.report`][] renders the result.
"""

_SECOND_HAND_COEFFICIENTS = frozenset({"tilly2", "tilly10"})
"""Models whose coefficients are still not fully attributable to their own
paper. Only Tilly remains: its primary states that the slopes were fitted
"as shown in figure 2" but prints no number, so Rorvik et al. 2018's
appendix B is the only numeric source even with the primary in hand.
"""

_SECOND_HAND_CAVEAT = (
    "The fitted slope is **not printed by its own paper**, which states only "
    "that the slopes were obtained by fitting 'as shown in figure 2'. The value "
    "used here comes from the appendix-B reformulation of Rorvik et al. 2018 "
    "(doi:10.1088/1361-6560/aad9db), presumably read off that figure."
)

_MODEL_CAVEATS_RMF: tuple[str, ...] = (
    "Frese et al. 2012 state the form holds 'up to at least 100 to 150 keV/um' "
    "and report that beyond ~150 it over-predicts both alpha and beta, because "
    "Sigma and z_F keep rising where the measured parameters turn over. Points "
    "above that are evaluated, warned about and counted, never clamped -- so "
    "read the out-of-domain warning count before the high-LET residual.",
    "The DSB yields are the MCDS closed form (Semenenko & Stewart 2006), whose "
    "own fit domain is 1 <= Z_eff^2/beta^2 <= 3200.",
    "The site size is a 5 um nucleus DIAMETER, as both primaries specify. It is "
    "not the 5 um RADIUS the LEM entries above use; the two are not comparable "
    "geometry choices.",
    "Alone among the mechanistic families here, RMF's tissue parameters are "
    "fully determined by the row's own (alpha_x, beta_x) -- nothing is a global "
    "harness choice except the reference radiation and the site diameter.",
)

_KATZ_SHARED_CAVEATS: tuple[str, ...] = (
    "Only the **grain-count regime** (eq. 8) is implemented, so sigma saturates "
    "at sigma_0. The track-width regime -- where Katz reports sigma/sigma_0 ~ 2 "
    "for argon on kidney cells -- needs eq. (4), whose site-averaged dose the "
    "paper prints only as Figs. 2-3 (computed by the method of Butts & Katz "
    "1967). Points above roughly z*^2 = 90 are therefore under-predicted.",
    "The fit is one 1960s cell line's four parameters applied to every PIDE row, "
    "the same coarse global choice the LEM and MKM family notes already carry.",
    "Katz fitted against the L values tabulated in his own Table I; this run "
    "derives LET from the package's default stopping power instead, so the two "
    "do not share an x-axis definition.",
)

_MODEL_CAVEATS: dict[str, tuple[str, ...]] = {
    "rmf": _MODEL_CAVEATS_RMF,
    "katz_hamster": (
        *_KATZ_SHARED_CAVEATS,
        "Chinese hamster cells (Fig. 14), fitted to Skarsgard et al. 1967. Its "
        "sigma_0 = 50 um^2 is the same effective nuclear cross-section the LEM "
        "family later adopts for V79.",
    ),
    "katz_hela": (
        *_KATZ_SHARED_CAVEATS,
        "HeLa cells (Fig. 15), fitted to Deering & Rice 1962. The only preset "
        "with m = 3, so it carries the broadest shoulder in the set.",
    ),
    "katz_t1_todd": (
        *_KATZ_SHARED_CAVEATS,
        "T-1 human kidney, aerobic (Fig. 11), fitted to Todd 1966. Katz states "
        "these cells reach the track-width regime under argon, which is exactly "
        "the regime not implemented here.",
    ),
    "katz_t1_barendsen": (
        *_KATZ_SHARED_CAVEATS,
        "T-1 human kidney, O2 (Fig. 13), fitted to Barendsen et al. 1966 -- the "
        "same cell line as katz_t1_todd from a different laboratory's data. Katz "
        "warns (p. 421) against reading significance into differences smaller "
        "than the gap between these two fits.",
    ),
    "wilkens": (
        "alpha_0 = 0.1 Gy^-1 is a V79 ion-response intercept, not a function of "
        "the row's alpha_x, so this model predicts the same V79 curve whatever "
        "cell line the row describes. Its residual is dominated by cell-line "
        "mismatch rather than by the parameterisation.",
    ),
    "frese": (
        "lambda = 0.008 is a planning constant -- a four-patient head-and-neck "
        "average at 2 Gy per fraction -- which the paper says must be re-derived "
        "for other fraction sizes. It is not a general-purpose fit to in-vitro "
        "data, which is what PIDE holds.",
    ),
    "mcnamara": (
        "Fitted domain LET_d < 20 keV/um and (alpha/beta)_x < 30 Gy; points "
        "outside it are evaluated, warned about and counted, never clamped.",
    ),
    "imk": (
        "Only the targeted channel of the integrated MK model is implemented; the "
        "non-targeted (bystander) half of Matsuya et al. is not.",
    ),
    "osmk2021": ("Evaluated at the papers' aerobic reference pO2 = 160 mmHg.",),
    "osmk2023": ("Evaluated at the papers' aerobic reference pO2 = 160 mmHg.",),
    "peeler": (
        "Cubic in LET_d, fitted to 12 LET_d values over 0.9-19.0 keV/um on two "
        "human lung lines (H460, H1437) spanning (alpha/beta)_x 2.6-5.1 Gy. A "
        "cubic extrapolates violently past that: read the out-of-domain warning "
        "count before the residual.",
        "Its two limits move in opposite directions with tissue: RBE_max scales "
        "with 1/(alpha/beta)_x but RBE_min scales with (alpha/beta)_x.",
    ),
    "unkelbach": (
        "This is **not a biological model**: it is the RBE implied by an LET "
        "optimisation method, whose c = 0.04 um/keV is set to give RBE = 1.1 at "
        "the 2.5 keV/um mean LET_d of a typical SOBP. Included for comparison, "
        "as Rorvik et al. do.",
        "RBE_max = RBE_min, so alone in this family its RBE does not vary with "
        "dose; it is also independent of tissue. That is exact, not an "
        "approximation: the paper's survival model is a pure exponential with "
        "no quadratic term at all.",
    ),
    "chen": (
        "Keys on alpha_x rather than on (alpha/beta)_x: its alpha_0 = 0.1 Gy^-1 "
        "is a V79 ion-response intercept, exactly as in Wilkens.",
        "Implemented from the paper's eq. (8), NOT from Rorvik et al. 2018's "
        "appendix-B reformulation, which omits the 1/alpha_x on the LET term and "
        "so under-predicts by a factor of ~10 in that term (RBE_10 = 1.10 vs "
        "2.23 at LET_d = 30 keV/um on the paper's own V79 worked case).",
    ),
    "jones": (
        "Keys on alpha_x and beta_x separately rather than on their ratio (as "
        "Wilkens, Frese and Chen do), so a sweep in (alpha/beta)_x is not its "
        "tissue axis.",
        "Built on the paper's eqs. (1) and (2), which differ from Rorvik et al. "
        "2018's appendix on both saturation constants: alpha_U uses Q/j = "
        "10.62/3.92 = 2.7092 (not 2.696) and beta_U uses R/u = 2.5/25 = 0.1 (not "
        "0.06 with u = 50), the paper's stated ceiling for beta_U being 0.1.",
    ),
    "tilly2": (
        "One of two fits to two (alpha/beta)_x bands; this is the low band "
        "(2.7-2.8 Gy, 7 data points, cell lines V79-379A and V79-753B). Its "
        "slope is 1.8x below tilly10's, which is the fit reporting that its own "
        "1/(alpha/beta)_x scaling does not carry across the range.",
    ),
    "tilly10": (
        "The high-(alpha/beta)_x band (7.7-15 Gy, 4 data points) of the same "
        "paper as tilly2; see that model's note on why the two differ.",
    ),
    "roru": (
        "Only the unweighted linear arm of the RØR pair. The companion RØRW "
        "integrates a quartic weighting function over a LET *spectrum*, which a "
        "scalar LET_d cannot express, so it is deliberately not implemented.",
    ),
}

BENCHMARK_SCOPE_NOTE = (
    "This scores a model against **measured survival data**, not against a paper's "
    "own published model line. It ranks models on a shared point set; it is not a "
    "reconstruction check, and a residual here must not be used to choose between "
    "implementations or parameter variants."
)
"""True of every run this harness produces, whatever the model or dataset."""


def model_caveats(model_name: str) -> tuple[str, ...]:
    """Everything a reader must know before believing ``model_name``'s score."""
    if model_name in LEM_MODELS:
        family = _FAMILY_CAVEATS["lem"]
    elif model_name in MKM_MODELS:
        family = _FAMILY_CAVEATS["mkm"]
    elif model_name in RMF_MODELS:
        family = _FAMILY_CAVEATS["rmf"]
    elif model_name in KATZ_MODELS:
        family = _FAMILY_CAVEATS["katz"]
    elif model_name in PHENOMENOLOGICAL_MODELS:
        family = _FAMILY_CAVEATS["phenomenological"]
    else:
        return _MODEL_CAVEATS.get(model_name, ())
    provenance = (_SECOND_HAND_CAVEAT,) if model_name in _SECOND_HAND_COEFFICIENTS else ()
    return (family, *provenance, *_MODEL_CAVEATS.get(model_name, ()))


def _lem_factory(
    model_name: str, *, D_t: float | TransitionDoseRule, nucleus_radius_um: float
) -> ModelFactory:
    """Build LEM models on demand, one per unique ``(alpha_x, beta_x)`` photon pair.

    ``D_t`` is one shared value or a [`TransitionDoseRule`][] evaluated on the
    photon pair of the first row that builds each model.
    """
    from pyradbio.models.lem.lem1 import LEM1
    from pyradbio.models.lem.lem2 import LEM2
    from pyradbio.models.lem.lem3 import LEM3
    from pyradbio.models.lem.lem4 import LEM4

    classes: dict[str, type[BiophysicalModel]] = {
        "lem1": LEM1,
        "lem2": LEM2,
        "lem3": LEM3,
        "lem4": LEM4,
    }
    cls = classes[model_name]
    cache: dict[tuple[float, float], BiophysicalModel] = {}

    def factory(alpha_x: float, beta_x: float) -> BiophysicalModel:
        key = (round(alpha_x, 6), round(beta_x, 6))
        model = cache.get(key)
        if model is None:
            d_t = D_t(alpha_x, beta_x) if callable(D_t) else D_t
            photon = PhotonParameters(alpha_x=alpha_x, beta_x=beta_x, D_t=d_t)
            cell = CellTarget(nucleus_radius_um=nucleus_radius_um)
            model = cls(photon=photon, cell=cell)  # type: ignore[call-arg]
            cache[key] = model
        return model

    return factory


def _mkm_factory(
    *, alpha0: float, beta0: float, domain_radius_um: float, nucleus_radius_um: float
) -> ModelFactory:
    """A single photon-independent MKM model, reused for every point.

    Runs on [`MKMModel`][pyradbio.models.mkm.model.MKMModel]'s default, pyMKM-free
    ``NativeMKMBackend``.
    """
    from pyradbio.models.mkm.model import MKMModel

    model: BiophysicalModel = MKMModel(
        domain_radius_um=domain_radius_um,
        nucleus_radius_um=nucleus_radius_um,
        alpha0=alpha0,
        beta0=beta0,
    )

    def factory(_alpha_x: float, _beta_x: float) -> BiophysicalModel:
        return model

    return factory


def _preset_mkm_factory(model_name: str) -> ModelFactory:
    """A single photon-independent MKM-family model, at its published preset.

    **No harness geometry is applied.** These parameter sets are measured or
    fitted together -- SMK's ``R_n = 8.1 um`` (Inaniwa & Kanematsu 2018) against
    mMKM's 4.3, MCF MKM's 4.5 -- so letting the LEM-ish ``R_n = 5 um`` default
    leak in would silently produce a model nobody published. Overrides are
    refused rather than ignored; build a [`ModelSpec`][] directly to vary one.

    One caveat carries over from the family's own reference notes: the reference
    ``y_D`` these models calibrate against belongs to a specific photon quality,
    and PIDE rows span 200 kVp, 60Co and MV. The preset's value is kept for every
    row, because per-row recalibration would be a refit rather than a benchmark.
    """
    from pyradbio.models.mkm import (
        OSMK2021,
        OSMK2023,
        IMKModel,
        MCFMKMModel,
        ModifiedMKM,
        StochasticMKM,
    )

    presets: dict[str, Callable[[], BiophysicalModel]] = {
        "mmkm": ModifiedMKM.hsg_mk_inaniwa2018,
        "smk": StochasticMKM.hsg_inaniwa2018,
        "mcf": MCFMKMModel,
        "osmk2021": OSMK2021.hsg_inaniwa2021,
        "osmk2023": OSMK2023.hsg_inaniwa2023,
        "imk": IMKModel.hsg_matsuya2025,
    }
    model = presets[model_name]()

    def factory(_alpha_x: float, _beta_x: float) -> BiophysicalModel:
        return model

    return factory


def _rmf_factory() -> ModelFactory:
    """Build RMF models on demand, one per unique ``(alpha_x, beta_x)`` photon pair.

    Like LEM this is photon-dependent, but for a different reason: RMF's two
    tissue parameters ``theta`` and ``kappa`` are *derived* from the reference
    pair by inverting Frese et al. 2012 eqs. (5)-(6), so the row's own
    ``(alpha_x, beta_x)`` is the whole tissue axis. Nothing here is a global
    harness choice except the reference radiation, which PIDE does not record.
    """
    from pyradbio.models.rmf import CO60_REFERENCE, RMFModel

    cache: dict[tuple[float, float], BiophysicalModel] = {}

    def factory(alpha_x: float, beta_x: float) -> BiophysicalModel:
        key = (round(alpha_x, 6), round(beta_x, 6))
        model = cache.get(key)
        if model is None:
            model = RMFModel(
                photon=PhotonParameters(alpha_x=alpha_x, beta_x=beta_x),
                reference_radiation=CO60_REFERENCE,
            )
            cache[key] = model
        return model

    return factory


def _katz_factory(model_name: str) -> ModelFactory:
    """A single photon-independent Katz model at one published parameter set.

    Katz is photon-independent in the same sense the MKM family is: the four
    parameters ``(sigma_0, E_0, kappa, m)`` are fitted together against one cell
    line's ion *and* gamma data, so no part of them is re-derived from a PIDE
    row's ``alpha_x``. As with the MKM presets, the harness contributes no
    geometry of its own and refuses overrides rather than ignoring them.
    """
    from pyradbio.models.katz import KatzCellularModel

    presets: dict[str, Callable[[], KatzCellularModel]] = {
        "katz_hamster": KatzCellularModel.chinese_hamster_skarsgard1967,
        "katz_hela": KatzCellularModel.hela_deering1962,
        "katz_t1_todd": KatzCellularModel.t1_kidney_todd1966_aerobic,
        "katz_t1_barendsen": KatzCellularModel.t1_kidney_barendsen1966_o2,
    }
    model: BiophysicalModel = presets[model_name]()

    def factory(_alpha_x: float, _beta_x: float) -> BiophysicalModel:
        return model

    return factory


def _phenomenological_factory(
    model_name: str,
) -> tuple[ModelFactory, type[RBEMinMaxModel]]:
    """Build one phenomenological model per unique ``(alpha_x, beta_x)`` photon pair.

    Unlike LEM, nothing here is a harness choice: these models take the tissue's
    ``(alpha/beta)_x`` and nothing else, and PIDE gives that per row.
    """
    from pyradbio.models.phenomenological import (
        Carabe,
        Chen,
        Jones,
        LinearScaling,
        MairaniHelium,
        McNamara,
        Peeler,
        RorvikUnweighted,
        Tilly2,
        Tilly10,
        Unkelbach,
        Wedenberg,
        WilkensOelfke,
    )

    classes: dict[str, type[RBEMinMaxModel]] = {
        "wilkens": WilkensOelfke,
        "carabe": Carabe,
        "frese": LinearScaling,
        "wedenberg": Wedenberg,
        "mcnamara": McNamara,
        "mairani": MairaniHelium,
        "tilly2": Tilly2,
        "tilly10": Tilly10,
        "chen": Chen,
        "jones": Jones,
        "roru": RorvikUnweighted,
        "peeler": Peeler,
        "unkelbach": Unkelbach,
    }
    cls = classes[model_name]
    cache: dict[tuple[float, float], BiophysicalModel] = {}

    def factory(alpha_x: float, beta_x: float) -> BiophysicalModel:
        key = (round(alpha_x, 6), round(beta_x, 6))
        model = cache.get(key)
        if model is None:
            model = cls(photon=PhotonParameters(alpha_x=alpha_x, beta_x=beta_x))
            cache[key] = model
        return model

    return factory, cls


def build_model_spec(
    model_name: str,
    *,
    D_t: float | TransitionDoseRule | None = None,
    nucleus_radius_um: float | None = None,
    domain_radius_um: float | None = None,
    alpha0: float | None = None,
    beta0: float | None = None,
) -> ModelSpec:
    """Return a [`ModelSpec`][] for ``model_name`` with documented defaults.

    LEM defaults: ``D_t = 30 Gy`` and ``R_n = 5 um`` are representative V79/HSG
    values (Elsasser & Scholz 2007; Friedrich 2012) used as a single global choice
    across cell lines. The generic ``mkm`` entry defaults
    ``(alpha0, beta0, r_d, R_n) = (0.1, 0.05, 0.44 um, 5 um)``, order-of-magnitude
    HSG values (Inaniwa 2010). ``D_t`` may instead be a [`TransitionDoseRule`][]
    ``(alpha_x, beta_x) -> D_t`` evaluated per photon pair, as Pfuhl et al. 2022
    eq. (3) prescribes for LEM IV.

    The remaining models carry a **published parameter set** instead -- the
    phenomenological fits, which need only the photon reference PIDE supplies per
    row, and the MKM-family presets, whose geometry is fitted or measured as a
    unit. Those refuse the knobs above rather than ignoring them.
    """
    if model_name in LEM_MODELS:
        return ModelSpec(
            build=_lem_factory(
                model_name,
                D_t=30.0 if D_t is None else D_t,
                nucleus_radius_um=5.0 if nucleus_radius_um is None else nucleus_radius_um,
            )
        )
    if model_name == "mkm":
        return ModelSpec(
            build=_mkm_factory(
                alpha0=0.1 if alpha0 is None else alpha0,
                beta0=0.05 if beta0 is None else beta0,
                domain_radius_um=0.44 if domain_radius_um is None else domain_radius_um,
                nucleus_radius_um=5.0 if nucleus_radius_um is None else nucleus_radius_um,
            ),
            cache_by_ion=True,
        )

    supplied = {
        name
        for name, value in (
            ("D_t", D_t),
            ("nucleus_radius_um", nucleus_radius_um),
            ("domain_radius_um", domain_radius_um),
            ("alpha0", alpha0),
            ("beta0", beta0),
        )
        if value is not None
    }
    if (
        model_name in MKM_MODELS
        or model_name in PHENOMENOLOGICAL_MODELS
        or model_name in KATZ_MODELS
        or model_name in RMF_MODELS
    ):
        if supplied:
            raise ValueError(
                f"{model_name} runs at its published parameter set; "
                f"{', '.join(sorted(supplied))} cannot be overridden here "
                "(build a ModelSpec directly to vary one)"
            )
    if model_name in MKM_MODELS:
        return ModelSpec(build=_preset_mkm_factory(model_name), cache_by_ion=True)
    if model_name in KATZ_MODELS:
        # Not an LETModel: Katz takes an Ion and derives LET itself, because
        # eq. (8) needs (z*, beta) and not only a scalar LET.
        return ModelSpec(build=_katz_factory(model_name), cache_by_ion=True)
    if model_name in RMF_MODELS:
        # Ion route for the same reason as Katz: the DSB yield keys on
        # Z_eff^2/beta^2. ``needs_beta_x`` because kappa = 2 beta_x / Sigma_x^2
        # is undefined without it.
        return ModelSpec(build=_rmf_factory(), needs_beta_x=True, cache_by_ion=True)
    if model_name in PHENOMENOLOGICAL_MODELS:
        factory, cls = _phenomenological_factory(model_name)
        return ModelSpec(
            build=factory,
            model_type=cls,
            applicable_z=cls.applicable_z(),
            needs_beta_x=True,
        )
    raise ValueError(f"unknown model {model_name!r}; choose from {benchmark_models()}")


def benchmark_models() -> tuple[str, ...]:
    """Every model name [`build_model_spec`][] accepts, in registry order."""
    return (*LEM_MODELS, *MKM_MODELS, *RMF_MODELS, *KATZ_MODELS, *PHENOMENOLOGICAL_MODELS)


def build_model_factory(
    model_name: str, *, D_t: float | TransitionDoseRule | None = None, **kwargs: float | None
) -> ModelFactory:
    """The [`ModelFactory`][] of ``model_name``'s spec (see [`build_model_spec`][]).

    Kept for callers that only need to *build* the model; anything that evaluates
    one wants the whole [`ModelSpec`][], since that carries the evaluation
    route and the fit's scope.
    """
    return build_model_spec(model_name, D_t=D_t, **kwargs).build


@dataclass(frozen=True, slots=True)
class _Candidate:
    """A record that passed every pre-evaluation filter, with its resolved ``A``."""

    entry: ReferenceRecord
    A: int


@dataclass(slots=True)
class _Tally:
    skipped: int = 0
    out_of_scope: int = 0


def _candidates(
    entries: Iterable[ReferenceRecord],
    spec: ModelSpec,
    *,
    max_z: int,
    min_energy_MeV_u: float,
    tally: _Tally,
) -> Iterator[_Candidate]:
    """Yield the records the model will be asked about, tallying the rest.

    Lazy, so a ``limit`` that stops the caller early also stops the tally.
    """
    for entry in entries:
        if spec.applicable_z is not None and entry.Z not in spec.applicable_z:
            tally.out_of_scope += 1
            continue
        if spec.needs_beta_x and entry.beta_x <= 0.0:
            tally.out_of_scope += 1
            continue
        energy = entry.energy_MeV_u
        # A nominal energy never gates a LET model; a missing one does not
        # when the record makes energy optional. A known target energy
        # always gates, so every model sees the same known-energy points.
        energy_gated = not (spec.let_based and entry.energy_is_nominal)
        energy_optional = spec.let_based and not entry.require_energy_for_let
        if energy_gated and (
            (energy is None and not energy_optional)
            or (energy is not None and energy < min_energy_MeV_u)
        ):
            tally.skipped += 1
            continue
        if entry.Z > max_z or entry.alpha_ion <= 0.0 or entry.alpha_x <= 0.0:
            tally.skipped += 1
            continue
        a = entry.A if entry.A is not None else default_mass_number(entry.Z)
        yield _Candidate(entry=entry, A=a)


def stratified_by_let(
    items: Sequence[_T], cap: int, *, let_keV_um: Callable[[_T], float]
) -> list[_T]:
    """At most ``cap`` of ``items``, evenly spaced in LET rank, in ascending LET.

    The picks are the ranks ``round(linspace(0, n - 1, cap))`` (half to even, as
    numpy rounds) in the LET-sorted order -- a stable sort, so ties keep their
    input order. Duplicate ranks collapse, so a cap close to ``n`` can return
    fewer than ``cap`` items; ``n <= cap`` items are all returned, LET-sorted.
    This is the thinning [`run_benchmark`][] applies per ion for
    ``limit_per_ion``.
    """
    if cap < 1:
        raise ValueError(f"cap must be >= 1, got {cap!r}")
    order = sorted(range(len(items)), key=lambda i: let_keV_um(items[i]))
    n = len(items)
    if n <= cap:
        return [items[i] for i in order]
    picks = np.linspace(0, n - 1, cap).round().astype(int)
    return [items[order[int(p)]] for p in sorted(set(picks.tolist()))]


def run_benchmark(
    entries: Sequence[ReferenceRecord] | ReferenceDataset,
    model_name: str = "lem1",
    *,
    spec: ModelSpec | None = None,
    factory: ModelFactory | None = None,
    max_z: int = 10,
    min_energy_MeV_u: float = 0.5,
    limit: int | None = None,
    limit_per_ion: int | None = None,
    stratify: Stratify = "let",
    D_t: float | TransitionDoseRule | None = None,
    **factory_kwargs: float | None,
) -> BenchmarkResult:
    """Evaluate ``model_name`` over ``entries`` and aggregate the residuals.

    Only entries with a known energy, ``Z <= max_z``, energy
    ``>= min_energy_MeV_u``, and a positive measured ``alpha_ion`` are
    evaluated; everything else (including any per-point model error) is skipped
    and tallied in [`BenchmarkResult.n_skipped`][]. The energy conditions are
    waived for a LET-based spec on records marked ``energy_is_nominal``. On
    records with ``require_energy_for_let=False`` (as set by the custom-CSV
    loader) a LET-based spec waives only the *missing*-energy condition; a known
    energy below ``min_energy_MeV_u`` is still skipped. A LET-based model never
    reads the energy. Points that the model's own
    published fit does not cover are tallied in
    [`BenchmarkResult.n_out_of_scope`][] instead -- a proton-only fit meeting a
    carbon point is a scope limit, not a failure.

    By default the energy filters apply to LET-based models too, so that PIDE
    models see the **same** candidate set and the ranking stays like-for-like.
    CSV records opt out of the missing-energy check only, because their energy
    column is optional. ``limit`` caps the
    number of *evaluated* points (useful for a quick slice).
    ``spec`` overrides ``model_name`` + ``D_t`` + ``factory_kwargs`` (both passed to
    [`build_model_spec`][], so ``D_t`` may be a [`TransitionDoseRule`][]);
    a bare ``factory`` is shorthand for an ion-based spec with no scope limit.

    ``limit_per_ion`` instead thins every ion (``Z``) to at most that many of the
    points that passed the filters above, chosen by ``stratify``. ``"let"``, the
    only rule, keeps points evenly spaced in LET rank ([`stratified_by_let`][]),
    so a bounded run still spans each ion's whole LET range. The dropped points
    are neither skipped nor out of scope -- they are simply not asked -- and the
    run then evaluates, and returns, its points grouped by ascending ``Z`` and in
    ascending LET within an ion. Without it the input order is kept.
    [`BenchmarkResult.attempted_by_ion`][] counts what each ion handed the model.

    Records marked ``energy_is_nominal`` are accepted only by LET-based specs.
    An energy-based run raises ``ValueError`` before evaluating any points if
    any record carries a nominal beam energy; nominal energies are not valid
    inputs for the ion response at the target.

    ``entries`` may be a bare sequence of records (the result is then labelled
    with the PIDE defaults, for back-compatibility) or a
    [`ReferenceDataset`][], whose name and caveats
    are carried into the result and so into the report.

    ``max_z`` defaults to 10 (hydrogen-neon, the light-ion range the families
    are validated on) so that benchmark point sets stay
    comparable between runs; it is not a stopping-power limit. The default
    stopping power
    ([`default_stopping_power`][pyradbio.track.stopping_power.default_stopping_power]) has no table
    edge in ``Z``: analytic Bethe covers everything outside its ICRU 90 and Hubert branches.
    """
    if stratify != "let":
        raise ValueError(f"stratify must be 'let', got {stratify!r}")
    if limit_per_ion is not None and limit_per_ion < 1:
        raise ValueError(f"limit_per_ion must be >= 1, got {limit_per_ion!r}")
    dataset_name = "PIDE"
    dataset_caveats: tuple[str, ...] = ()
    if isinstance(entries, ReferenceDataset):
        dataset_name = entries.name
        dataset_caveats = entries.caveats
        entries = entries.records
    if spec is None:
        spec = (
            ModelSpec(build=factory)
            if factory is not None
            else build_model_spec(model_name, D_t=D_t, **factory_kwargs)
        )

    if not spec.let_based and any(entry.energy_is_nominal for entry in entries):
        raise ValueError(
            "Cannot evaluate an energy-based model with nominal beam energies. "
            "Use a LET-based model, or supply independently verified target energies."
        )

    tally = _Tally()
    stream: Iterator[_Candidate] = _candidates(
        entries, spec, max_z=max_z, min_energy_MeV_u=min_energy_MeV_u, tally=tally
    )
    if limit_per_ion is not None:
        groups: dict[int, list[_Candidate]] = {}
        for found in stream:
            groups.setdefault(found.entry.Z, []).append(found)
        stream = iter(
            [
                kept
                for z in sorted(groups)
                for kept in stratified_by_let(
                    groups[z], limit_per_ion, let_keV_um=lambda c: c.entry.let_keV_um
                )
            ]
        )

    points: list[PointResult] = []
    attempted: dict[int, int] = {}
    cache: dict[tuple[float | int | None, ...], LQParameters] = {}
    with warnings.catch_warnings(record=True) as caught:
        # "always" defeats the once-per-location registry, so the tally is per
        # point rather than per call site.
        warnings.simplefilter("always")
        # The limit is checked before the next candidate is drawn, so records
        # after the last evaluated point are not tallied (as before the
        # candidate stream existed).
        while limit is None or len(points) < limit:
            candidate = next(stream, None)
            if candidate is None:
                break
            entry, a = candidate.entry, candidate.A
            energy = entry.energy_MeV_u
            attempted[entry.Z] = attempted.get(entry.Z, 0) + 1
            # Keyed on whatever the chosen route actually reads: a LET-based model
            # is blind to (Z, A, E), and two rows can share an ion at different
            # tabulated LET.
            photon_key = (round(entry.alpha_x, 6), round(entry.beta_x, 6))
            key = (
                (*photon_key, entry.let_keV_um)
                if spec.let_based
                else (*photon_key, entry.Z, a, energy)
            )
            try:
                cached = cache.get(key) if spec.cache_by_ion else None
                if cached is not None:
                    lq = cached
                else:
                    model = spec.build(entry.alpha_x, entry.beta_x)
                    if spec.let_based:
                        if not isinstance(model, LETModel):
                            raise TypeError(
                                f"{type(model).__name__} is not an LETModel, but the spec "
                                f"routes on LET (model_type={spec.model_type!r})"
                            )
                        lq = model.alpha_beta_from_let(entry.let_keV_um)
                    else:
                        assert energy is not None  # gated above for the ion route
                        lq = model.alpha_beta(Ion(Z=entry.Z, A=a, energy_MeV_u=energy))
                    if spec.cache_by_ion:
                        cache[key] = lq
                predicted = lq.alpha
            except (ValueError, KeyError, ZeroDivisionError, NotImplementedError):
                # NotImplementedError: a route the model declares unimplemented
                # (e.g. a refused mixed field, or NanOx's
                # NanOxNanodosimetryRequired) is "not covered", not a crash.
                tally.skipped += 1
                continue
            if predicted <= 0.0:
                tally.skipped += 1
                continue
            points.append(
                PointResult(
                    cell_line=entry.cell_line,
                    Z=entry.Z,
                    A=a,
                    energy_MeV_u=None if entry.energy_is_nominal else energy,
                    let_keV_um=entry.let_keV_um,
                    alpha_x=entry.alpha_x,
                    predicted_alpha=predicted,
                    measured_alpha=entry.alpha_ion,
                    beta_x=entry.beta_x,
                    predicted_beta=lq.beta,
                    measured_beta=entry.beta_ion,
                )
            )
        n_warnings = sum(1 for w in caught if issubclass(w.category, UserWarning))
    return BenchmarkResult(
        model_name=model_name,
        points=tuple(points),
        n_skipped=tally.skipped,
        n_out_of_scope=tally.out_of_scope,
        n_warnings=n_warnings,
        dataset_name=dataset_name,
        dataset_caveats=dataset_caveats,
        attempted_by_ion=tuple(sorted(attempted.items())),
    )
