"""Spectrum-driven MKM family: LQ parameters / RBE from a lineal-energy spectrum.

Microdosimetric-kinetic models that consume a single-event dose-weighted
lineal-energy spectrum ``d(y)`` (our [`Spectrum`][]
or a digitized/library curve) and return ion LQ parameters, hence RBE. They share
`_SpectrumMKModel` -- cell/reference calibration, engine plumbing, RBE --
and differ only in how radiation quality enters ``alpha``/``beta``:

* [`ModifiedMKM`][] -- the **modified MKM** (Kase et al. 2006,
  doi:10.1667/RR0536.1) as used by Hartzell 2025 eqs 10-12: ``alpha`` depends on
  the saturation-corrected dose-mean lineal energy ``y*_D``; ``beta = beta_ref``
  is radiation-independent. The clinical baseline and the natural cross-check.
* [`MCFMKMModel`][] -- the **Mayo Clinic Florida MKM** (Parisi, Beltran &
  Furutani 2022, doi:10.3390/ijms232012491 and doi:10.1088/1361-6560/ac7375).
  Both ``alpha`` *and* ``beta`` depend on radiation quality through a
  saturation (non-Poisson, overkill) correction ``c(y)``.
* [`StochasticMKM`][] -- the **SMK** (Inaniwa & Kanematsu 2018,
  doi:10.1088/1361-6560/aabede). Adds *nucleus*-level stochastics on top of the
  domain-level saturation, so ``beta`` becomes quality-dependent and survival
  picks up a non-LQ dose-dependent correction. Unlike the other two it needs the
  spectrum scored at **two** site sizes (domain *and* nucleus). See
  ``docs/references/SMK_Inaniwa2018.md``.

These are engine-agnostic (any ``d(y)`` source) and pyMKM-free; pyMKM remains the
cross-check oracle where the two overlap -- see ``AGENTS.md``.

Formalism (single event, domain radius ``r_d``, nucleus ``R_n``)
---------------------------------------------------------------
The single-event specific energy of a target of radius ``r`` crossed by a track
of lineal energy ``y`` is ``z(y) = y / (rho * pi * r^2)``; the keV/um -> Gy
conversion uses the repo's ``KEVUM_PER_GY_UM2_GCM3`` (see
[`specific_energy_Gy`][]). That relation is geometry-agnostic between the two
conventions in play: a sphere with ``l_bar = 4r/3``, ``V = 4 pi r^3/3`` and an
axis-parallel cylinder of radius ``r`` and length ``2r`` both give it.

The *distribution* is where geometry bites. The MKM scores a **disk/cylinder**
(Inaniwa 2010; Inaniwa & Kanematsu 2018 sec 3.2), and a spherical site overshoots
``z*_1D`` by up to ~32% at 400 MeV/u. `_SpectrumMKModel.specific_energies`
owns that choice via ``site_geometry`` (default ``"cylinder"``) for models driven
by an *ion*; a supplied ``d(y)`` carries its own geometry and is left alone.

**MCF MKM** (Parisi 2022, eqs 3-5, 14; transcribed from the machine-readable
Europe PMC full text -- the Hartzell 2025 OCR of these is garbled):

    X(y) = (alpha0 + beta0 * z_d(y)) * z_n(y) + beta0 * z_n(y)^2
    c(y) = (1 - exp(-X(y))) / X(y)                         # -> 1 as y -> 0
    alpha = integral (alpha0 + beta0 * z_d(y)) * c(y) * d(y) dy
    beta  = beta0 * ( integral c(y) * d(y) dy )^2
    alpha0 = alpha_ref - beta0 * z_d(y_D,ref) ,   beta0 = beta_ref

The photon limit (a reference-like spectrum: ``c -> 1`` and ``y_D -> y_D,ref``)
recovers ``alpha -> alpha_ref`` and ``beta -> beta_ref``.

**mMKM** (Kase 2006 / Hartzell eqs 10-12):

    y0    = KEVUM * rho * pi * r_d * R_n^2 / sqrt(beta_ref * (r_d^2 + R_n^2))
    y*_D  = y0^2 * integral (1 - exp(-(y/y0)^2)) * d(y) / y  dy
    alpha = alpha_ref + beta_ref * (y*_D - y*_D,ref) / (KEVUM * rho * pi * r_d^2)
    beta  = beta_ref

**SMK** (Inaniwa & Kanematsu 2018, the paper's own eq. numbers; see
``docs/references/SMK_Inaniwa2018.md``):

    alpha_SMK = alpha0 + z*_1D * beta0                                (15)
    beta_SMK  = beta0 * (z*_1D / z_1D)                                (16)
    gamma(D)  = z_n * (0.5 * (alpha_SMK + 2 beta_SMK D)^2 - beta_SMK)
    S(D)      = exp(-(alpha_SMK D + beta_SMK D^2)) * (1 + gamma(D) D)  (24)

with ``z_1D``/``z*_1D`` the (saturation-corrected) dose-mean single-event domain
specific energy (eqs. 11-12) and ``z_n`` its nucleus counterpart (eq. 7) -- the
second site size. Note ``z0`` is a *free fit parameter* in SMK, not the Kase
geometric relation mMKM uses.

Numerics
--------
The integrals are ``integral h(y) d(y) dy`` evaluated with Simpson quadrature on
the spectrum's own ``y`` grid, after renormalising ``d(y)`` to unit area (the
physics assumes ``integral d(y) dy = 1``; digitized/truncated spectra are not).
This is appropriate for a *finely sampled* ``d(y)`` (the analytic engine's dense
grid, or a hand-digitized smooth curve). It is **not** appropriate to feed a
coarse log-binned count histogram through its geometric bin centres -- integrate
such a [`TabulatedLinealEnergySpectrum`][pyradbio.micro.lineal_energy.TabulatedLinealEnergySpectrum]
per-bin instead (see AGENTS.md -> Numerics).
"""

from __future__ import annotations

import math
from collections.abc import Callable
from dataclasses import dataclass, fields, is_dataclass
from functools import cache, lru_cache
from typing import Any, ClassVar, Literal, NamedTuple, get_args, overload

import numpy as np
import numpy.typing as npt
from scipy.integrate import simpson
from scipy.optimize import brentq
from scipy.special import logsumexp

from pyradbio.core.constants import KEVUM_PER_GY_UM2_GCM3, WATER_DENSITY_G_CM3
from pyradbio.core.lq import LQParameters, dose_for_survival
from pyradbio.core.mixing import Spacing
from pyradbio.core.photon import PhotonParameters
from pyradbio.core.radiation import Ion, RadiationField
from pyradbio.micro.impact_parameter import specific_energy_impact_profile
from pyradbio.micro.lineal_energy import (
    AmorphousTrackEngine,
    Spectrum,
    SpectrumEngine,
    SphericalSite,
)
from pyradbio.micro.multi_event import multi_event_density, single_event_density
from pyradbio.models.base import (
    BiophysicalModel,
    ModelCapabilities,
    apply_preset,
    check_choice,
    invert_survival_curve,
)
from pyradbio.track.base import RadialDoseModel
from pyradbio.track.kiefer_chatterjee import KieferChatterjeeRadialDose
from pyradbio.track.northcliffe_schilling import BetheOnly
from pyradbio.track.stopping_power import (
    DefaultStoppingPower,
    SimpleBetheStoppingPower,
    StoppingPower,
    TabularStoppingPower,
    default_stopping_power,
)

# Below this single-event lethal-lesion expectation the c(y) = (1-e^-X)/X ratio
# is evaluated by its Taylor limit 1 - X/2 to avoid 0/0 cancellation noise.
_SMALL_X: float = 1.0e-8

SiteGeometry = Literal["cylinder", "sphere"]
"""How an *ion* is turned into specific energies -- see
`_SpectrumMKModel.specific_energies`. ``"cylinder"`` is the MKM's own disk
geometry (Inaniwa & Kanematsu 2018 sec 3.2); ``"sphere"`` is for comparison.
"""


@cache
def _default_micro_engine(domain_radius_um: float, density_g_cm3: float) -> AmorphousTrackEngine:
    """The MKM family's default single-event spectrum engine.

    The **deterministic amorphous** (Kiefer-Chatterjee radial dose, impact-parameter
    method) engine, scored in the MKM *domain* sphere (diameter ``2 * r_d``). It is
    dependency-light (no pyMKM, no external table), fast and reproducible, and
    reproduces the Hartzell 2025 analytical-microdosimetric-function ``d(y)`` for the
    heavy-ion / high-LET regime the MKM targets to a few percent (see
    ``docs/references/Microdosimetry.md``, "Amorphous-track engine"). Cached per
    ``(domain_radius_um, density)`` since it is stateless. For light ions / low LET,
    inject an engine carrying a
    [`DeltaRayMonteCarlo`][pyradbio.micro.lineal_energy.DeltaRayMonteCarlo] straggling layer, which
    lifts the (otherwise under-estimated) low-LET ``y_D``.
    """
    sp = default_stopping_power(density_g_cm3=density_g_cm3)
    return AmorphousTrackEngine(
        radial_dose=KieferChatterjeeRadialDose(stopping_power=sp),
        site=SphericalSite(diameter_um=2.0 * domain_radius_um),
        stopping_power=sp,
        density_g_cm3=density_g_cm3,
    )


def specific_energy_Gy(
    y_keV_um: npt.NDArray[np.float64] | float,
    radius_um: float,
    density_g_cm3: float = WATER_DENSITY_G_CM3,
) -> npt.NDArray[np.float64]:
    """Single-event specific energy ``z(y) = y / (rho * pi * r^2)`` [Gy].

    For a spherical target of radius ``radius_um`` crossed by a track of lineal
    energy ``y`` [keV/um]. ``y[keV/um] = KEVUM_PER_GY_UM2_GCM3 * rho * z[Gy] *
    pi r^2`` (the same conversion the ``micro`` layer uses), so
    ``z[Gy] = y / (KEVUM_PER_GY_UM2_GCM3 * rho * pi * r^2)``.
    """
    if radius_um <= 0.0:
        raise ValueError("radius_um must be > 0")
    z: npt.NDArray[np.float64] = np.asarray(y_keV_um, dtype=np.float64) / (
        KEVUM_PER_GY_UM2_GCM3 * density_g_cm3 * math.pi * radius_um * radius_um
    )
    return z


def _normalized_dose_spectrum(
    y_keV_um: npt.NDArray[np.float64], d_of_y: npt.NDArray[np.float64]
) -> tuple[npt.NDArray[np.float64], npt.NDArray[np.float64]]:
    """Return ``(y, d)`` sorted on ``y`` with ``d`` renormalised to unit area."""
    y = np.asarray(y_keV_um, dtype=np.float64)
    d = np.asarray(d_of_y, dtype=np.float64)
    if y.ndim != 1 or d.ndim != 1 or y.size != d.size:
        raise ValueError("y and d(y) must be one-dimensional arrays of equal length")
    if y.size < 2:
        raise ValueError("at least two spectral points are required to integrate")
    order = np.argsort(y)
    y, d = y[order], d[order]
    if np.any(y <= 0.0):
        raise ValueError("lineal energy y must be positive")
    area = float(simpson(d, x=y))
    if area <= 0.0:
        raise ValueError("d(y) must have positive area")
    return y, d / area


def _dose_spectrum_of(
    spectrum: Spectrum,
) -> tuple[npt.NDArray[np.float64], npt.NDArray[np.float64]]:
    """Extract the ``(y, d(y))`` dose-weighted spectrum from a [`Spectrum`][]."""
    return spectrum.y_keV_um, spectrum.d


# --- Kase 2006 / Inaniwa 2010 saturation (shared by mMKM and the native MKM
#     backend) --------------------------------------------------------------
def saturation_y0_keV_um(
    *,
    domain_radius_um: float,
    nucleus_radius_um: float,
    beta_Gy2: float,
    density_g_cm3: float = WATER_DENSITY_G_CM3,
) -> float:
    """Saturation parameter ``y0`` [keV/um] (Hartzell eq 12 / Kase 2006).

    Depends only on the domain/nucleus geometry and ``beta`` -- not on the
    spectrum -- so both [`ModifiedMKM`][] and a native MKM backend share it.
    """
    r_d, r_n = domain_radius_um, nucleus_radius_um
    return (
        KEVUM_PER_GY_UM2_GCM3
        * density_g_cm3
        * math.pi
        * r_d
        * r_n
        * r_n
        / math.sqrt(beta_Gy2 * (r_d * r_d + r_n * r_n))
    )


def _y_star_D_keV_um(
    y_keV_um: npt.NDArray[np.float64], d_of_y: npt.NDArray[np.float64], y0_keV_um: float
) -> float:
    """Saturation-corrected dose-mean lineal energy ``y*_D`` [keV/um] (Kase eq 11).

    ``y*_D = y0^2 * integral (1 - exp(-(y/y0)^2)) d(y)/y dy`` for a unit-area
    ``d(y)`` (renormalised here).
    """
    y, d = _normalized_dose_spectrum(y_keV_um, d_of_y)
    integrand = (1.0 - np.exp(-((y / y0_keV_um) ** 2))) * d / y
    return float(y0_keV_um * y0_keV_um * simpson(integrand, x=y))


def _y_frequency_means_keV_um(
    y_keV_um: npt.NDArray[np.float64], d_of_y: npt.NDArray[np.float64], y0_keV_um: float
) -> tuple[float, float]:
    """``(y_F, y*_F)`` -- plain and saturation-corrected **frequency**-mean ``y``.

    Sato & Furusawa 2012 eq. (30), ``z*_d,F = integral z'_d f_d,1(z_d) dz_d``, with
    ``z'_d = z0 sqrt(1 - exp[-(z_d/z0)^2])`` (their eq. 16). Note the **first**
    power of the saturated variable, where the dose mean `_y_star_D_keV_um`
    carries its square -- that is the whole difference between the two means.

    Expressed on the dose-weighted ``d(y)`` so digitized spectra work, using
    ``f(y) proportional to d(y)/y``: for unit-area ``d``, ``1/y_F = integral d/y dy``
    and ``y*_F = y_F integral y'(y) d(y)/y dy``.
    """
    y, d = _normalized_dose_spectrum(y_keV_um, d_of_y)
    inverse_y_F = float(simpson(d / y, x=y))
    if inverse_y_F <= 0.0:
        raise ValueError("d(y)/y must have positive area to define a frequency mean")
    y_F = 1.0 / inverse_y_F
    y_saturated = y0_keV_um * np.sqrt(-np.expm1(-((y / y0_keV_um) ** 2)))
    y_star_F = y_F * float(simpson(y_saturated * d / y, x=y))
    return y_F, y_star_F


def z_star_1D_from_dose_spectrum(
    y_keV_um: npt.NDArray[np.float64],
    d_of_y: npt.NDArray[np.float64],
    *,
    domain_radius_um: float,
    nucleus_radius_um: float,
    beta_Gy2: float,
    density_g_cm3: float = WATER_DENSITY_G_CM3,
) -> float:
    """Saturation-corrected dose-mean domain specific energy ``z*_1D`` [Gy].

    The Inaniwa 2010 MKM quantity (pyMKM's ``z_bar_star_domain``) computed from a
    single-event dose-weighted lineal-energy spectrum ``d(y)`` and the MKM
    geometry -- ``z*_1D = y*_D / (KEVUM * rho * pi * r_d^2)``. This is the seam a
    native (pyMKM-free) [`MKMBackend`][] uses.
    """
    y0 = saturation_y0_keV_um(
        domain_radius_um=domain_radius_um,
        nucleus_radius_um=nucleus_radius_um,
        beta_Gy2=beta_Gy2,
        density_g_cm3=density_g_cm3,
    )
    y_star = _y_star_D_keV_um(y_keV_um, d_of_y, y0)
    scale = KEVUM_PER_GY_UM2_GCM3 * density_g_cm3 * math.pi * domain_radius_um**2
    return y_star / scale


@dataclass(frozen=True, slots=True)
class SiteSpecificEnergies:
    """Dose-mean single-event specific energies for one ion, at both site sizes.

    The moment-level interface between a microdosimetry engine and the models that
    need only moments (mMKM, SMK). ``z_star_1D_Gy``/``z_1D_Gy`` are the
    (saturation-corrected and plain) dose-mean specific energies per event in the
    **domain**; ``z_nucleus_Gy`` is its **nucleus** counterpart -- Inaniwa &
    Kanematsu 2018 eqs. (11), (12) and (7) respectively.
    """

    z_1D_Gy: float
    z_star_1D_Gy: float
    z_nucleus_Gy: float
    z_F_Gy: float | None = None
    """Frequency means, plain and saturation-corrected (Sato & Furusawa 2012
    eq. 30). ``None`` when the route that produced these moments cannot define
    them -- specifically the analytic impact-parameter route, which carries no
    minimum-energy-deposition threshold and so has no event definition to
    frequency-average over. Only ``domain_treatment="sato_moments"`` needs them.
    """
    z_star_F_Gy: float | None = None


# ---- the impact-parameter route, memoized across model instances -----------
# Node counts of the impact-parameter route (see ``specific_energies``). They are
# part of the cache key, so changing one can never serve a stale entry.
_IMPACT_RADIAL_NODES: int = 700
_IMPACT_CORE_NODES: int = 600
_IMPACT_PENUMBRA_NODES: int = 600

_MICRODOSIMETRY_CACHE_SIZE: int = 4096
"""Bound on the number of ``(site, ion)`` entries the route cache keeps."""
_MICRODOSIMETRY_CACHE_ENABLED: bool = True
"""Switch for tests and timing runs; the cache is on in normal use."""

_CACHEABLE_STOPPING_POWERS: frozenset[type] = frozenset(
    {DefaultStoppingPower, SimpleBetheStoppingPower, TabularStoppingPower, BetheOnly}
)
"""Stopping powers whose LET is a pure function of the ion (exact types: a subclass
may add state, so it is not trusted -- a new stateless subclass must be listed
here or it silently bypasses the cache). ``DefaultStoppingPower`` memoizes its
anchor factors, which does not change any value; ``BetheOnly`` is
``SimpleBetheStoppingPower`` plus a class-level ``name``. Stateful sources such as
[`NorthcliffeSchillingWithBetheTail`][pyradbio.track.northcliffe_schilling.NorthcliffeSchillingWithBetheTail]
(it tallies table hits and fallbacks) must see every call, so they bypass.
"""
_CACHEABLE_RADIAL_DOSES: frozenset[type] = frozenset({KieferChatterjeeRadialDose})
"""Radial-dose models that are pure functions of the ion (and of their own stopping
power, which must itself be cacheable).
"""


class _MicrodosimetryKey(NamedTuple):
    """Everything the impact-parameter route of ``specific_energies`` reads.

    ``z0_Gy`` is the saturation parameter in force (``z0_effective_Gy``), so the
    fields it is derived from (``z0_Gy``, ``beta_ref``, the radii, the density)
    enter through its value. ``radial_dose``/``stopping_power`` are the model's
    fields as given (``None`` = the library default at ``density_g_cm3``); stopping
    powers compare by identity, so separately built instances do not share entries.

    The numeric values are kept exactly as the model holds them (no ``float()``),
    so the route sees the same objects as an uncached evaluation. Because a numpy
    ``float32`` compares and hashes equal to the ``float`` of the same value yet
    evaluates differently, ``value_types`` records their types, keeping such
    twins in separate entries; ``radial_dose_value_types`` does the same for the
    fields of a value-hashing (dataclass) radial dose. The ion's field types are
    added per call (see `_impact_parameter_moments_finite`).

    ``resolvers`` are the module-level [`default_stopping_power`][] and
    [`KieferChatterjeeRadialDose`][] that resolve ``None`` at call time, so a
    monkeypatched default is a different key rather than a stale hit.
    """

    domain_radius_um: float
    nucleus_radius_um: float
    z0_Gy: float
    site_geometry: str
    density_g_cm3: float
    radial_dose: RadialDoseModel | None
    stopping_power: StoppingPower | None
    radial_nodes: int
    core_nodes: int
    penumbra_nodes: int
    value_types: tuple[type, ...]
    radial_dose_value_types: tuple[type, ...]
    resolvers: tuple[object, object]


def _field_value_types(obj: object) -> tuple[type, ...]:
    """Types of a dataclass instance's field values (``()`` for anything else)."""
    if not is_dataclass(obj) or isinstance(obj, type):
        return ()
    return tuple(type(getattr(obj, f.name)) for f in fields(obj))


def _resolve_stopping_power(
    stopping_power: StoppingPower | None, density_g_cm3: float
) -> StoppingPower:
    """``stopping_power``, else [`default_stopping_power`][] at ``density_g_cm3``."""
    if stopping_power is not None:
        return stopping_power
    return default_stopping_power(density_g_cm3=density_g_cm3)


def _resolve_radial_dose(
    radial_dose: RadialDoseModel | None,
    stopping_power: StoppingPower | None,
    density_g_cm3: float,
) -> RadialDoseModel:
    """``radial_dose``, else Kiefer-Chatterjee on the resolved stopping power."""
    if radial_dose is not None:
        return radial_dose
    return KieferChatterjeeRadialDose(
        stopping_power=_resolve_stopping_power(stopping_power, density_g_cm3)
    )


def _impact_parameter_moments(key: _MicrodosimetryKey, ion: Ion) -> SiteSpecificEnergies:
    """The analytic impact-parameter route of `_SpectrumMKModel.specific_energies`.

    A pure function of ``(key, ion)``, which is what makes it safe to memoize.
    """
    radial_dose = _resolve_radial_dose(key.radial_dose, key.stopping_power, key.density_g_cm3)
    z0 = key.z0_Gy
    # High node counts: at high energy r_max spans many decades, so the domain
    # core and the far penumbra both need resolving for z*_1D to converge.
    domain = specific_energy_impact_profile(
        radial_dose,
        ion,
        region_radius_um=key.domain_radius_um,
        geometry=key.site_geometry,
        radial_nodes=key.radial_nodes,
        core_nodes=key.core_nodes,
        penumbra_nodes=key.penumbra_nodes,
        density_g_cm3=key.density_g_cm3,
    )
    # Same high node counts for the nucleus: at the default (300/400/400) the
    # dose-mean carries ~2% jitter in T around 0.5-1 MeV/u, which is visible
    # in z_n and therefore in the SMK gamma term. Converged by ~700.
    nucleus = specific_energy_impact_profile(
        radial_dose,
        ion,
        region_radius_um=key.nucleus_radius_um,
        geometry=key.site_geometry,
        radial_nodes=key.radial_nodes,
        core_nodes=key.core_nodes,
        penumbra_nodes=key.penumbra_nodes,
        density_g_cm3=key.density_g_cm3,
    )
    return SiteSpecificEnergies(
        z_1D_Gy=domain.dose_mean_Gy(),
        z_star_1D_Gy=domain.saturation_corrected_dose_mean_Gy(z0),
        z_nucleus_Gy=nucleus.dose_mean_Gy(),
    )


class _NonFiniteMoments(Exception):
    """Carries a non-finite route result past `functools.lru_cache` uncached."""

    def __init__(self, moments: SiteSpecificEnergies) -> None:
        super().__init__("non-finite impact-parameter moments")
        self.moments = moments


def _impact_parameter_moments_finite(
    key: _MicrodosimetryKey, ion: Ion, ion_value_types: tuple[type, ...]
) -> SiteSpecificEnergies:
    """`_impact_parameter_moments`, as the memoized target.

    ``ion_value_types`` is not read: it only splits the entries of an ``Ion``
    and its numpy-scalar twin (equal and equal-hashing, yet evaluated
    differently). A non-finite result (a degenerate input such as ``z0 = 0``)
    is raised rather than returned, because `functools.lru_cache` does not
    cache exceptions -- so every such call re-evaluates and re-warns exactly as
    an uncached call would.
    """
    out = _impact_parameter_moments(key, ion)
    if not (
        math.isfinite(out.z_1D_Gy)
        and math.isfinite(out.z_star_1D_Gy)
        and math.isfinite(out.z_nucleus_Gy)
    ):
        raise _NonFiniteMoments(out)
    return out


_impact_parameter_moments_cached = lru_cache(maxsize=_MICRODOSIMETRY_CACHE_SIZE)(
    _impact_parameter_moments_finite
)


class MicrodosimetryCacheInfo(NamedTuple):
    """Statistics of the impact-parameter microdosimetry cache.

    Same fields as `functools.lru_cache`'s ``cache_info()``.
    """

    hits: int
    misses: int
    maxsize: int | None
    currsize: int


def clear_microdosimetry_cache() -> None:
    """Empty the spectrum-MKM family's impact-parameter microdosimetry cache.

    `_SpectrumMKModel.specific_energies` memoizes its analytic
    impact-parameter route (no injected engine) across model instances, keyed on
    the fields that route reads -- so the many instances of a parameter scan that
    share one site geometry (e.g. the oxygen scan in
    ``validation/osmk_inaniwa2021.py``) score each ``(site, ion)`` once. The
    entries hold references to the models' ``radial_dose``/``stopping_power``
    objects; clearing releases them. Call this after mutating one of those objects
    in place (stopping powers are keyed by identity, so the cache cannot see e.g.
    a changed ``SimpleBetheStoppingPower.density_g_cm3`` or edited
    ``TabularStoppingPower`` arrays) or after patching a module global the route
    reaches indirectly (e.g. the stopping power a ``KieferChatterjeeRadialDose``
    with ``stopping_power=None`` would look up).
    """
    _impact_parameter_moments_cached.cache_clear()


def microdosimetry_cache_info() -> MicrodosimetryCacheInfo:
    """``(hits, misses, maxsize, currsize)`` of the impact-parameter cache.

    See [`clear_microdosimetry_cache`][] for what is cached.
    """
    return MicrodosimetryCacheInfo(*_impact_parameter_moments_cached.cache_info())


def _cacheable_stopping_power(stopping_power: object) -> bool:
    return stopping_power is None or type(stopping_power) in _CACHEABLE_STOPPING_POWERS


def _cacheable_radial_dose(radial_dose: object) -> bool:
    if radial_dose is None:
        return True
    if type(radial_dose) not in _CACHEABLE_RADIAL_DOSES:
        return False
    return _cacheable_stopping_power(getattr(radial_dose, "stopping_power", None))


@dataclass(frozen=True, slots=True)
class _SpectrumMKModel(BiophysicalModel):
    """Shared cell/reference parameters, engine plumbing and RBE for the family.

    Subclasses implement `alpha_beta_from_dose_spectrum`; everything else
    (spectrum extraction, ion -> spectrum via an injected engine, the
    ``alpha0``/``beta0`` reference calibration, RBE) is shared.

    ``site_geometry`` selects how an *ion* (rather than a supplied ``d(y)``) is
    turned into specific energies -- see `specific_energies`.
    """

    capabilities: ClassVar[ModelCapabilities] = ModelCapabilities(
        mixed_field=False, spectrum_route=True
    )
    """Ion and spectrum routes; mixed fields are a follow-up (see ``_alpha_beta_field``)."""

    nucleus_radius_um: float
    domain_radius_um: float = 0.28
    alpha_ref: float = 0.217  # Gy^-1  (HSG, 6 MV photons; Hartzell 2025 sec 2.3)
    beta_ref: float = 0.0615  # Gy^-2
    yD_ref_keV_um: float = 2.5
    density_g_cm3: float = WATER_DENSITY_G_CM3
    engine: SpectrumEngine | None = None
    z0_Gy: float | None = None  # explicit saturation parameter; see y0_keV_um
    site_geometry: SiteGeometry = "cylinder"
    radial_dose: RadialDoseModel | None = None
    alpha0_Gy_inv: float | None = None  # fitted alpha0; see the alpha0 property
    beta0_Gy2: float | None = None  # fitted beta0; see the beta0 property
    stopping_power: StoppingPower | None = None
    """Where LET comes from. Defaults to [`default_stopping_power`][] at this
    model's density. Explicit because more than one thing needs it -- the
    default radial dose and (for the numerical nucleus route) the single-event
    nucleus density -- and they must not be able to disagree.
    """

    def __post_init__(self) -> None:
        check_choice("site_geometry", self.site_geometry, get_args(SiteGeometry))

    # ---- reference calibration (shared by the whole family) -------------
    @property
    def beta0(self) -> float:
        """Intrinsic ``beta0`` [Gy^-2] -- the fitted value if given, else ``beta_ref``.

        Distinct from ``beta_ref`` in general: the MKM papers fit ``beta0`` to the
        *ion* data while ``beta_ref`` describes the photon reference used for RBE.
        For HSG, Inaniwa & Kanematsu 2018 give ``beta0 = 0.0568 Gy^-2`` against a
        200 kVp reference ``beta_X = 0.0593 Gy^-2`` (Kanai et al 2006).
        """
        return self.beta_ref if self.beta0_Gy2 is None else self.beta0_Gy2

    @property
    def z_d_ref_Gy(self) -> float:
        """Domain specific energy of the *reference* radiation, ``z_d(y_D,ref)`` [Gy]."""
        return float(
            specific_energy_Gy(self.yD_ref_keV_um, self.domain_radius_um, self.density_g_cm3)
        )

    @property
    def alpha0(self) -> float:
        """Intrinsic ``alpha0`` [Gy^-1] -- the fitted value if given, else calibrated.

        Calibrated form: ``alpha0 = alpha_ref - beta0 * z_d(y_D,ref)``, the
        ``z -> 0`` intercept implied by the photon reference (MCF MKM eq. 14; the
        same calibration anchors the SMK ``alpha_SMK = alpha0 + beta0 z*_1D`` so
        that a reference-like spectrum returns ``alpha_ref``).

        MKM papers usually quote ``alpha0``/``beta0`` directly instead -- e.g.
        Inaniwa & Kanematsu 2018 table 1 -- so ``alpha0_Gy_inv`` takes the fitted
        number as-is. See [`StochasticMKM.hsg_inaniwa2018`][] and
        [`ModifiedMKM.hsg_mk_inaniwa2018`][].
        """
        if self.alpha0_Gy_inv is not None:
            return self.alpha0_Gy_inv
        return self.alpha_ref - self.beta0 * self.z_d_ref_Gy

    @property
    def _z_domain_scale_keV_um_per_Gy(self) -> float:
        """``KEVUM * rho * pi * r_d^2`` -- the ``y[keV/um] -> z_d[Gy]`` divisor."""
        return KEVUM_PER_GY_UM2_GCM3 * self.density_g_cm3 * math.pi * self.domain_radius_um**2

    # ---- Kase/Inaniwa domain saturation (shared by mMKM and SMK) --------
    @property
    def y0_keV_um(self) -> float:
        """Saturation parameter ``y0`` [keV/um] (Hartzell eq 12 / Kase 2006).

        Derived from the domain/nucleus geometry and ``beta`` unless ``z0_Gy`` is
        set, in which case that value is used directly (converted to keV/um). The
        override matters: the Kase relation ties ``z0`` to ``R_n``, but SMK fits
        ``z0`` as a **free** parameter -- for HSG, Inaniwa & Kanematsu 2018 table 1
        gives ``z0 = 66.0 Gy`` where the geometric relation would give ~121 Gy.
        """
        if self.z0_Gy is not None:
            return self.z0_Gy * self._z_domain_scale_keV_um_per_Gy
        return saturation_y0_keV_um(
            domain_radius_um=self.domain_radius_um,
            nucleus_radius_um=self.nucleus_radius_um,
            beta_Gy2=self.beta_ref,
            density_g_cm3=self.density_g_cm3,
        )

    def y_star_D(self, y_keV_um: npt.NDArray[np.float64], d_of_y: npt.NDArray[np.float64]) -> float:
        """Saturation-corrected dose-mean lineal energy ``y*_D`` [keV/um] (eq 11).

        ``y*_D = y0^2 * integral (1 - exp(-(y/y0)^2)) d(y)/y dy`` for a unit-area
        ``d(y)`` -- identical to [`Spectrum.y_star`][] (verified in tests) but
        expressed on the dose-weighted spectrum so digitized ``d(y)`` curves work.
        """
        return _y_star_D_keV_um(y_keV_um, d_of_y, self.y0_keV_um)

    def z_star_1D_Gy(
        self, y_keV_um: npt.NDArray[np.float64], d_of_y: npt.NDArray[np.float64]
    ) -> float:
        """Saturation-corrected dose-mean domain specific energy ``z*_1D`` [Gy].

        ``z*_1D = y*_D / (KEVUM * rho * pi * r_d^2)`` -- the Inaniwa 2010 eqs 3-4
        MKM quantity. With ``y0`` (eq 12) equal to Kase's ``z0`` scaled by the
        same factor, this integral is identical to pyMKM's ``z_bar_star_domain``
        for a matched single-event spectrum and geometry (cross-checked in tests).
        """
        return self.y_star_D(y_keV_um, d_of_y) / self._z_domain_scale_keV_um_per_Gy

    def z_1D_Gy(self, y_keV_um: npt.NDArray[np.float64], d_of_y: npt.NDArray[np.float64]) -> float:
        """**Un**corrected dose-mean domain specific energy ``z_1D = y_D / scale`` [Gy].

        The ``y0 -> infinity`` (no-saturation) counterpart of `z_star_1D_Gy`.
        SMK needs both: their *ratio* is the quality dependence of ``beta``
        ([`StochasticMKM`][], eq. 2).
        """
        y, d = _normalized_dose_spectrum(y_keV_um, d_of_y)
        y_D = float(simpson(y * d, x=y))
        return y_D / self._z_domain_scale_keV_um_per_Gy

    def z_F_Gy(self, y_keV_um: npt.NDArray[np.float64], d_of_y: npt.NDArray[np.float64]) -> float:
        """**Un**corrected *frequency*-mean domain specific energy ``z_d,F`` [Gy].

        The frequency mean weights every event equally, where the dose mean
        weights by the energy deposited; see `_y_frequency_means_keV_um`.
        Used by Sato & Furusawa 2012 eq. (33).
        """
        y_F, _ = _y_frequency_means_keV_um(y_keV_um, d_of_y, self.y0_keV_um)
        return y_F / self._z_domain_scale_keV_um_per_Gy

    def z_star_F_Gy(
        self, y_keV_um: npt.NDArray[np.float64], d_of_y: npt.NDArray[np.float64]
    ) -> float:
        """Saturation-corrected frequency-mean domain specific energy ``z*_d,F`` [Gy].

        Sato & Furusawa 2012 eq. (30). Its ratio to `z_F_Gy` is the factor
        that paper applies to the ``alpha0`` term and Inaniwa & Kanematsu 2018
        takes as 1 -- see [`StochasticMKM`][]'s ``domain_treatment``.
        """
        _, y_star_F = _y_frequency_means_keV_um(y_keV_um, d_of_y, self.y0_keV_um)
        return y_star_F / self._z_domain_scale_keV_um_per_Gy

    def z_nucleus_Gy(
        self, y_keV_um: npt.NDArray[np.float64], d_of_y: npt.NDArray[np.float64]
    ) -> float:
        """Dose-mean single-event specific energy in the **nucleus** ``z_n`` [Gy].

        ``z_n = y_D / (KEVUM * rho * pi * R_n^2)`` from a spectrum scored *in the
        nucleus* -- note the caller must pass the nucleus-scored ``d(y)``, not the
        domain one. Used only by [`StochasticMKM`][] (eq. 3).
        """
        y, d = _normalized_dose_spectrum(y_keV_um, d_of_y)
        y_D = float(simpson(y * d, x=y))
        return y_D / (
            KEVUM_PER_GY_UM2_GCM3 * self.density_g_cm3 * math.pi * self.nucleus_radius_um**2
        )

    # ---- primary API: a spectrum in, LQ out ----------------------------
    def alpha_beta_from_dose_spectrum(
        self, y_keV_um: npt.NDArray[np.float64], d_of_y: npt.NDArray[np.float64]
    ) -> LQParameters:
        raise NotImplementedError  # pragma: no cover -- implemented by subclasses

    def alpha_beta_from_spectrum(self, spectrum: Spectrum) -> LQParameters:
        """LQ parameters from a [`Spectrum`][] (its dose-weighted ``d(y)``)."""
        y, d = _dose_spectrum_of(spectrum)
        return self.alpha_beta_from_dose_spectrum(y, d)

    def _alpha_beta_ion(self, ion: Ion) -> LQParameters:
        """LQ parameters for one ion.

        Default: build the single-event ``d(y)`` and use the spectral formulae.
        Models whose ``alpha``/``beta`` depend on ``d(y)`` only through its
        *moments* (mMKM, SMK) override this to go through `specific_energies`,
        which honours ``site_geometry``.
        """
        return self.alpha_beta_from_spectrum(self._single_event_spectrum(ion))

    def _require_ion(self, field: Ion | RadiationField) -> Ion:
        """The ion of a single-ion call; mixed fields are refused like `alpha_beta`."""
        if isinstance(field, Ion):
            return field
        raise NotImplementedError(
            f"{type(self).__name__} does not mix fields yet; evaluate each Ion component, "
            "or pass a combined d(y) to the spectrum entry points."
        )

    def _alpha_beta_field(
        self, field: RadiationField, *, sub_nodes: int, spacing: Spacing
    ) -> LQParameters:
        """Mixed [`RadiationField`][] support is a follow-up.

        The field's combined ``d(y)`` is the natural input; per-component
        dose-averaging of ``alpha``/``sqrt(beta)`` is the alternative (Hartzell
        eq 5).
        """
        raise NotImplementedError(
            "mixed-field RBE is a follow-up; call alpha_beta_from_spectrum with a "
            "combined d(y), or alpha_beta on each Ion component."
        )

    def _single_event_spectrum(self, ion: Ion) -> Spectrum:
        """Single-event ``d(y)`` for ``ion``, scored in the **domain**.

        Falls back to the deterministic amorphous engine (`_default_micro_engine`)
        built from this model's ``domain_radius_um`` when no engine is injected, so
        ``MCFMKMModel(...).alpha_beta(ion)`` works out of the box without pyMKM or an
        external microdosimetry table.
        """
        return self._spectrum_at_site(ion, self.domain_radius_um, self.engine)

    # ---- ion -> specific energies (the geometry seam) -------------------
    @property
    def z0_effective_Gy(self) -> float:
        """The saturation parameter actually in force, in Gy (see `y0_keV_um`)."""
        return self.y0_keV_um / self._z_domain_scale_keV_um_per_Gy

    @property
    def _uses_injected_engine(self) -> bool:
        """Whether `specific_energies` must defer to a caller-supplied engine."""
        return self.engine is not None

    def _stopping_power(self) -> StoppingPower:
        """The model's LET source; [`default_stopping_power`][] by default."""
        return _resolve_stopping_power(self.stopping_power, self.density_g_cm3)

    def _microdosimetry_key(self) -> _MicrodosimetryKey:
        """The inputs of the impact-parameter route, as the cache key.

        Exactly the fields the route reads, not the whole model, so instances that
        differ only in cell/oxygen parameters share entries. Completeness is tested
        by perturbing every dataclass field and comparing what is served against
        an evaluation that reads the model's fields directly.
        """
        z0 = self.z0_effective_Gy
        return _MicrodosimetryKey(
            domain_radius_um=self.domain_radius_um,
            nucleus_radius_um=self.nucleus_radius_um,
            z0_Gy=z0,
            site_geometry=self.site_geometry,
            density_g_cm3=self.density_g_cm3,
            radial_dose=self.radial_dose,
            stopping_power=self.stopping_power,
            radial_nodes=_IMPACT_RADIAL_NODES,
            core_nodes=_IMPACT_CORE_NODES,
            penumbra_nodes=_IMPACT_PENUMBRA_NODES,
            value_types=(
                type(self.domain_radius_um),
                type(self.nucleus_radius_um),
                type(z0),
                type(self.density_g_cm3),
            ),
            radial_dose_value_types=_field_value_types(self.radial_dose),
            resolvers=(default_stopping_power, KieferChatterjeeRadialDose),
        )

    def _microdosimetry_cacheable(self) -> bool:
        """Whether the impact-parameter route may be served from the cache.

        No when an engine is injected (that is a different route), or when the
        stopping power / radial dose is not of a known stateless type -- a stateful
        source (e.g. one that tallies its calls) must see every evaluation.
        """
        return (
            not self._uses_injected_engine
            and _cacheable_stopping_power(self.stopping_power)
            and _cacheable_radial_dose(self.radial_dose)
        )

    def specific_energies(self, ion: Ion) -> SiteSpecificEnergies:
        """Dose-mean specific energies for ``ion`` at the domain and nucleus sizes.

        Two routes, and which one runs is decided by whether an ``engine`` is
        injected -- because a [`SpectrumEngine`][]
        already fixes its own site geometry, and silently overriding it would be
        exactly the kind of hidden physics ``AGENTS.md`` forbids:

        * **no engine (default)** -- the analytic **impact-parameter** route
          ([`specific_energy_impact_profile`][])
          in ``site_geometry``. ``"cylinder"`` is the default and the correct MKM
          convention: Inaniwa & Kanematsu 2018 sec 3.2 (and Inaniwa 2010 before it)
          take the domain and nucleus to be **cylinders of radius ``r`` and length
          ``2r`` with tracks parallel to the axis**, which makes ``z_1`` the
          disk area-mean of the radial dose. This is also pyMKM's ``SpecificEnergy``
          geometry, so ``z*_1D`` reproduces ``z_bar_star_domain``. ``"sphere"`` is
          selectable for comparison but **overshoots ``z*_1D`` by ~25-40% at high
          energy** -- see ``docs/references/SMK_Inaniwa2018.md`` and
          [`pyradbio.micro.impact_parameter`][].
        * **engine injected** -- the single-event ``d(y)`` route, scored at each
          site size by that engine, with the moments taken as in the ``d(y)``
          formulae. Use this to drive the models from a measured / digitized /
          Monte-Carlo spectrum.

        Supplying ``d(y)`` directly to ``alpha_beta_from_dose_spectrum`` bypasses
        this entirely -- there the caller owns the geometry.

        The impact-parameter route is memoized across instances in a bounded
        module-level LRU keyed on the fields it reads (`_microdosimetry_key`),
        so a scan over cell or oxygen parameters at one site geometry scores each
        ion once. It is bypassed for an injected engine, an unhashable key, and a
        stopping power / radial dose not known to be stateless, and a non-finite
        result is never stored. The key records the numeric types of the model's,
        the radial dose's and the ion's fields, so a numpy ``float32`` twin of a
        ``float`` input gets its own entry, and the cached value is bit-identical
        to a fresh evaluation. Stopping powers are keyed by identity: after
        mutating one in place, or patching a module global outside this module
        that the route reaches, call [`clear_microdosimetry_cache`][].
        """
        if self._uses_injected_engine:
            return self._specific_energies_from_spectra(ion)

        key = self._microdosimetry_key()
        if _MICRODOSIMETRY_CACHE_ENABLED and self._microdosimetry_cacheable():
            try:
                hash((key, ion))
            except TypeError:  # an unhashable radial dose or stopping power
                pass
            else:
                try:
                    return _impact_parameter_moments_cached(key, ion, _field_value_types(ion))
                except _NonFiniteMoments as non_finite:
                    return non_finite.moments
        return _impact_parameter_moments(key, ion)

    def _specific_energies_from_spectra(self, ion: Ion) -> SiteSpecificEnergies:
        """`specific_energies` via the injected spectrum engine(s)."""
        domain = self._spectrum_at_site(ion, self.domain_radius_um, self.engine)
        nucleus = self._nucleus_spectrum(ion)
        return SiteSpecificEnergies(
            z_1D_Gy=self.z_1D_Gy(domain.y_keV_um, domain.d),
            z_star_1D_Gy=self.z_star_1D_Gy(domain.y_keV_um, domain.d),
            z_nucleus_Gy=self.z_nucleus_Gy(nucleus.y_keV_um, nucleus.d),
            z_F_Gy=self.z_F_Gy(domain.y_keV_um, domain.d),
            z_star_F_Gy=self.z_star_F_Gy(domain.y_keV_um, domain.d),
        )

    def _nucleus_spectrum(self, ion: Ion) -> Spectrum:
        """Nucleus-scored ``d(y)``. Only SMK needs it; it overrides to use its own engine."""
        return self._spectrum_at_site(ion, self.nucleus_radius_um, None)

    def _spectrum_at_site(
        self, ion: Ion, site_radius_um: float, engine: SpectrumEngine | None
    ) -> Spectrum:
        """Single-event ``d(y)`` for ``ion`` scored in a sphere of ``site_radius_um``.

        A [`SpectrumEngine`][] is bound to one site
        size at construction, so an injected ``engine`` is used as-is and the caller
        is responsible for having sized it to ``site_radius_um``. With no engine we
        build (and cache) the deterministic amorphous default at that radius. SMK is
        the reason this is parameterised: it scores the **nucleus** as well as the
        domain ([`StochasticMKM`][]).
        """
        chosen = engine or _default_micro_engine(site_radius_um, self.density_g_cm3)
        return chosen.single_event_spectrum(ion)

    # ---- RBE -----------------------------------------------------------
    def reference(self) -> PhotonParameters:
        """The photon reference ``(alpha_ref, beta_ref)`` as plain LQ (no ``D_t``)."""
        return PhotonParameters(alpha_x=self.alpha_ref, beta_x=self.beta_ref)

    def rbe_at_survival_from_spectrum(
        self, spectrum: Spectrum, survival_fraction: float = 0.1
    ) -> float:
        """Iso-survival RBE of a [`Spectrum`][] vs the photon reference.

        ``RBE_S = D_ref(S) / D_ion(S)`` (Hartzell eq 7). The ion-driven form is
        the inherited [`rbe_at_survival`][pyradbio.models.base.BiophysicalModel.rbe_at_survival].
        """
        lq = self.alpha_beta_from_spectrum(spectrum)
        return self._rbe_at_survival_from_lq(lq, survival_fraction)

    def rbe_at_survival_from_dose_spectrum(
        self,
        y_keV_um: npt.NDArray[np.float64],
        d_of_y: npt.NDArray[np.float64],
        survival_fraction: float = 0.1,
    ) -> float:
        """``RBE_S`` from a raw ``(y, d(y))`` pair (e.g. a digitized curve)."""
        lq = self.alpha_beta_from_dose_spectrum(y_keV_um, d_of_y)
        return self._rbe_at_survival_from_lq(lq, survival_fraction)


@dataclass(frozen=True, slots=True)
class MCFMKMModel(_SpectrumMKModel):
    """Mayo Clinic Florida MKM (Parisi et al. 2022).

    Defaults are the HSG cell line under a 6 MV photon reference (Hartzell 2025
    sec 2.3). ``alpha`` and ``beta`` both vary with radiation quality via the
    saturation correction ``c(y)``.

    With no ``engine`` injected, ``alpha_beta(ion)`` uses the **deterministic
    amorphous** default (`_default_micro_engine`) -- dependency-light, no
    pyMKM/PHITS, and accurate for the heavy-ion / high-LET regime. For light ions
    / low LET inject an engine with a ``DeltaRayMonteCarlo`` straggling layer (it
    lifts the otherwise under-estimated low-LET ``y_D``).

    **Site geometry.** Unlike mMKM and SMK this model stays on the ``d(y)`` route
    for ``alpha_beta(ion)`` and so keeps the *spherical* sites Parisi derives it on
    (``z = y l_bar / (rho V)`` with ``l_bar = 4r/3``). That is the model's own
    convention, not an oversight: its ``c(y)`` couples the domain and nucleus
    specific energies **per event** through the shared lineal energy
    (``z_n/z_d = (r_d/R_n)^2``), which the impact-parameter route -- where the two
    sites have independent ``z(b)`` profiles -- does not reproduce. Re-deriving MCF
    on disk geometry would be a change to the published model, so it is tracked in
    ``ROADMAP.md`` rather than done silently here. ``site_geometry``
    therefore has no effect on [`MCFMKMModel`][].

    Parisi sizes the subnuclear domain ``r_d`` (and picks ``R_n``) *a priori* per
    cell line from its DNA content (Parisi eqs 15-17). That geometry is general
    biophysics, so it lives in [`pyradbio.core.chromatin`][]
    ([`domain_radius_from_genome`][]) rather than here;
    pass its result as ``domain_radius_um`` to reproduce the paper's per-cell-line
    parameters. Note ``yD_ref_keV_um`` must also match the reference radiation
    (~2.3 keV/um for 60Co/6 MV, ~4.3 for 200 kVp X-rays; Parisi sec 2.5.3).
    """

    nucleus_radius_um: float = 4.5

    def saturation_correction(
        self, y_keV_um: npt.NDArray[np.float64] | float
    ) -> npt.NDArray[np.float64]:
        """The per-event saturation (overkill) factor ``c(y) = (1 - e^-X)/X``."""
        y = np.asarray(y_keV_um, dtype=np.float64)
        z_d = specific_energy_Gy(y, self.domain_radius_um, self.density_g_cm3)
        z_n = specific_energy_Gy(y, self.nucleus_radius_um, self.density_g_cm3)
        a0, b0 = self.alpha0, self.beta0
        big_x = (a0 + b0 * z_d) * z_n + b0 * z_n * z_n
        safe_x = np.where(big_x > 0.0, big_x, 1.0)
        c: npt.NDArray[np.float64] = np.where(
            big_x > _SMALL_X, -np.expm1(-big_x) / safe_x, 1.0 - 0.5 * big_x
        )
        return c

    def alpha_beta_from_dose_spectrum(
        self, y_keV_um: npt.NDArray[np.float64], d_of_y: npt.NDArray[np.float64]
    ) -> LQParameters:
        y, d = _normalized_dose_spectrum(y_keV_um, d_of_y)
        z_d = specific_energy_Gy(y, self.domain_radius_um, self.density_g_cm3)
        c = self.saturation_correction(y)
        alpha = float(simpson((self.alpha0 + self.beta0 * z_d) * c * d, x=y))
        beta = self.beta0 * float(simpson(c * d, x=y)) ** 2
        return LQParameters(alpha=max(alpha, 0.0), beta=max(beta, 0.0))


@dataclass(frozen=True, slots=True)
class ModifiedMKM(_SpectrumMKModel):
    """modified MKM (Kase et al. 2006), Hartzell 2025 eqs 10-12.

    ``beta = beta_ref`` (radiation-independent); ``alpha`` follows the
    saturation-corrected dose-mean lineal energy ``y*_D``. ``R_n = 4.3 um`` by
    default (chosen in Hartzell 2025 to align mMKM with the MCF MKM).
    ``yStar_D_ref_keV_um`` is the reference ``y*_D``; if unset it defaults to
    ``y_D,ref`` (its low-LET limit, where saturation is negligible).
    """

    nucleus_radius_um: float = 4.3
    yStar_D_ref_keV_um: float | None = None

    # ---- published parameter sets --------------------------------------
    @classmethod
    def hsg_mk_inaniwa2018(cls, **overrides: Any) -> ModifiedMKM:
        """HSG cells, **MK model** column of Inaniwa & Kanematsu 2018 table 1.

        ``alpha0 = 0.150 Gy^-1``, ``beta0 = 0.0593 Gy^-2``, ``r_d = 0.29 um``,
        ``R_n = 3.9 um``, ``z0 = 55.0 Gy`` -- the parameters behind the *dashed*
        curves of the paper's figs. 2-3, fitted to the same Furusawa et al. 2000
        data as the SMK column. Note ``R_n`` here is the MKM's effective radius,
        entangled with the saturation, **not** the 8.1 um measured nucleus the SMK
        column uses (see [`StochasticMKM.hsg_inaniwa2018`][]).
        """
        return apply_preset(cls, _HSG_MK_INANIWA2018, overrides)

    @classmethod
    def v79_mk_inaniwa2018(cls, **overrides: Any) -> ModifiedMKM:
        """V79 cells, **MK model** column of Inaniwa & Kanematsu 2018 table 1.

        ``alpha0 = 0.136 Gy^-1``, ``beta0 = 0.0200 Gy^-2``, ``r_d = 0.24 um``,
        ``R_n = 3.6 um``, ``z0 = 105.8 Gy``.
        """
        return apply_preset(cls, _V79_MK_INANIWA2018, overrides)

    @property
    def _ystar_ref_keV_um(self) -> float:
        return self.yD_ref_keV_um if self.yStar_D_ref_keV_um is None else self.yStar_D_ref_keV_um

    def alpha_beta_from_dose_spectrum(
        self, y_keV_um: npt.NDArray[np.float64], d_of_y: npt.NDArray[np.float64]
    ) -> LQParameters:
        scale = self._z_domain_scale_keV_um_per_Gy
        return self.alpha_beta_from_z_star_1D(self.y_star_D(y_keV_um, d_of_y) / scale)

    def _alpha_beta_ion(self, ion: Ion) -> LQParameters:
        """``alpha`` from ``z*_1D`` in ``site_geometry`` (cylinder by default)."""
        return self.alpha_beta_from_z_star_1D(self.specific_energies(ion).z_star_1D_Gy)

    def alpha_beta_from_z_star_1D(self, z_star_1D_Gy: float) -> LQParameters:
        """``alpha = alpha0 + beta0 z*_1D``; ``beta = beta0``.

        The moment-level form of [`alpha_beta_from_dose_spectrum`][], so an
        external ``z*_1D`` (a backend, a pyMKM table) can drive mMKM directly.

        Identical to the Hartzell eq.-10 form ``alpha_ref + beta_ref (z*_1D -
        z*_1D,ref)`` whenever ``alpha0`` comes from the photon calibration, since
        ``alpha0 = alpha_ref - beta0 z*_1D,ref`` by construction; writing it this
        way lets a *fitted* ``alpha0_Gy_inv`` (as the MKM papers quote) drive the
        model too.
        """
        alpha = self.alpha0 + self.beta0 * z_star_1D_Gy
        return LQParameters(alpha=max(alpha, 0.0), beta=self.beta0)


DomainTreatment = Literal["inaniwa_moments", "sato_moments"]
"""How the *domain* stochastics are treated in the SMK family. All three are
published readings of the same model lineage -- Inaniwa & Kanematsu 2018 is the
treatment-planning adaptation of Sato & Furusawa 2012, not a separate model.

* ``"inaniwa_moments"`` -- Inaniwa & Kanematsu 2018 eqs. (1)-(2). The default,
  and what this package has always shipped.
* ``"sato_moments"`` -- Sato & Furusawa 2012 eq. (33). Identical ``beta``; the
  ``alpha0`` term additionally carries ``z*_d,F / z_d,F`` (their eq. 30), which
  Inaniwa takes as 1. The ratio is <= 1 and falls with LET, so this lowers
  ``alpha`` at high LET and agrees at low LET.
"""

NucleusTreatment = Literal["taylor", "numerical"]
"""How the *nucleus* stochastics are evaluated.

* ``"taylor"`` -- Inaniwa & Kanematsu 2018's second-order expansion of
  ``S_n(z_n)`` about ``z_n = D``, i.e. the ``(1 + gamma D)`` factor. The
  default and what this package has always shipped. Fast and closed form, but
  ``1 + gamma D`` grows without bound, so it stops being a survival curve at
  large ``z_n D`` -- see ``expansion_validity_limit``.
* ``"numerical"`` -- Sato & Furusawa 2012 eq. (7) itself,
  ``S(D) = integral S_n(z_n) f_n(z_n; D) dz_n``, evaluated on the multi-event
  density from [`pyradbio.micro.multi_event`][]. A weighted average of
  ``exp(-...)`` terms, so it is in ``(0, 1]`` by construction at any dose and
  needs no validity ceiling.
"""


@dataclass(frozen=True, slots=True)
class StochasticMKM(_SpectrumMKModel):
    """SMK -- stochastic microdosimetric kinetic model (Inaniwa & Kanematsu 2018).

    doi:10.1088/1361-6560/aabede. Equations, derivation and sourcing caveats:
    ``docs/references/SMK_Inaniwa2018.md``.

    The MK model evaluates survival at the *mean* nucleus specific energy and so
    over-predicts the effect exactly where particle therapy lives -- high LET (few,
    large events per nucleus) and high dose per fraction. mMKM patches this with a
    saturation correction on the **domain** only. SMK additionally carries the
    **nucleus**-level stochastics, treating the accumulated nucleus specific energy
    as a random variable about the macroscopic dose ``D``:

    ```text
    alpha_SMK = alpha0 + beta0 * z*_1D                                  (1)
    beta_SMK  = (z*_1D / z_1D) * beta0                                  (2)
    gamma(D)  = z_n * (0.5 * (alpha_SMK + 2 beta_SMK D)^2 - beta_SMK)   (3)
    S(D)      = exp(-(alpha_SMK D + beta_SMK D^2)) * (1 + gamma(D) D)   (4)
    ```

    Two consequences distinguish it from mMKM: ``beta`` becomes quality-dependent
    (eq. 2 -- it *falls* with LET, since saturation makes ``z*_1D < z_1D``), and
    **survival is no longer LQ** (eq. 4). ``alpha_SMK`` is unchanged from mMKM.

    Two site sizes
    --------------
    Eq. (3) needs ``z_n``, the dose-mean single-event specific energy in the
    **nucleus**, so SMK consumes the single-event spectrum scored at *both* ``r_d``
    and ``R_n`` -- the one structural difference from its siblings. With no engines
    injected, two default amorphous engines are built (one per site size). Inject
    ``engine`` for the domain and ``nucleus_engine`` for the nucleus; a
    [`SpectrumEngine`][] is bound to one site size,
    so these cannot be the same object.

    Parameter sets
    --------------
    Published fits are available as [`hsg_inaniwa2018`][] / [`v79_inaniwa2018`][]
    (paper table 1). Two things differ from the mMKM conventions and are easy to
    get wrong:

    * ``z0`` is a **free fit parameter** in SMK (``z0_Gy``), not the Kase geometric
      relation of ``r_d``/``R_n``/``beta0``. For HSG the paper fits 66.0 Gy where
      the geometric relation gives ~121 Gy.
    * ``R_n = 8.1 um`` is the *measured* mean nucleus radius of sixteen human cell
      lines (Suzuki et al 2000) and is used **only** to derive ``z_n``. It is not
      the ~3.9 um effective radius the MK-model column of table 1 carries.

    Validity
    --------
    Eq. (24) is a truncated second-order expansion valid "assuming the number of
    energy-deposition events to a cell nucleus is large" (paper sec 3.1): ``1 +
    gamma D`` grows without bound, so at large ``z_n * D`` the raw expression stops
    being a survival curve. Results are clipped to ``[0, 1]`` and
    [`survival_valid_to_Gy`][] reports where the correction reaches
    ``expansion_validity_limit`` (default 1.0, i.e. a 100% correction); beyond that
    the curve is extrapolation. Nothing here is silently rescaled. With the paper's
    own HSG parameters that ceiling is ~4-18 Gy for carbon -- the same order as the
    ``D <~ 10 Gy`` range the paper works in, so check it per case.
    """

    capabilities: ClassVar[ModelCapabilities] = ModelCapabilities(
        mixed_field=False, lq_exact=False, spectrum_route=True
    )
    """Survival is eq. (24), not LQ: ``alpha_beta`` is the LQ core only."""

    nucleus_radius_um: float = 8.1
    nucleus_engine: SpectrumEngine | None = None
    expansion_validity_limit: float = 1.0
    domain_treatment: DomainTreatment = "inaniwa_moments"
    nucleus_treatment: NucleusTreatment = "taylor"
    nucleus_density_nodes: int = 4096
    """Grid resolution for the ``"numerical"`` nucleus route's single-event density."""

    def __post_init__(self) -> None:
        _SpectrumMKModel.__post_init__(self)
        check_choice("domain_treatment", self.domain_treatment, get_args(DomainTreatment))
        check_choice("nucleus_treatment", self.nucleus_treatment, get_args(NucleusTreatment))

    # ---- published parameter sets --------------------------------------
    @classmethod
    def hsg_inaniwa2018(cls, **overrides: Any) -> StochasticMKM:
        """HSG cells, modified-SMK column of Inaniwa & Kanematsu 2018 table 1.

        ``alpha0 = 0.174 Gy^-1``, ``beta0 = 0.0568 Gy^-2``, ``r_d = 0.28 um``,
        ``R_n = 8.1 um``, ``z0 = 66.0 Gy``, fitted to the Furusawa et al. 2000
        3He/12C/20Ne survival data. The photon reference is 200 kVp X-rays,
        ``alpha_X = 0.331 Gy^-1`` / ``beta_X = 0.0593 Gy^-2`` (Kanai et al 2006,
        used for the eq.-26 biological dose).
        """
        return apply_preset(cls, _HSG_INANIWA2018, overrides)

    @classmethod
    def v79_inaniwa2018(cls, **overrides: Any) -> StochasticMKM:
        """V79 cells, modified-SMK column of Inaniwa & Kanematsu 2018 table 1.

        ``alpha0 = 0.103 Gy^-1``, ``beta0 = 0.0210 Gy^-2``, ``r_d = 0.23 um``,
        ``R_n = 8.1 um``, ``z0 = 122.0 Gy``. ``alpha_ref``/``beta_ref`` are left at
        the class defaults -- the paper does not quote a V79 photon reference for
        eq. 26, so set them explicitly before asking for an RBE.
        """
        return apply_preset(cls, _V79_INANIWA2018, overrides)

    # ---- eqs. (1)-(2): the LQ core -------------------------------------
    def alpha_beta_from_specific_energies(
        self,
        *,
        z_star_1D_Gy: float,
        z_1D_Gy: float,
        alpha0: float | None = None,
        alpha0_saturation_ratio: float = 1.0,
    ) -> LQParameters:
        """Eqs. (1)-(2) from the microdosimetric quantities directly.

        The seam that lets an external engine (e.g. pyMKM's ``z_bar_star_domain`` /
        ``z_bar_domain``) drive the SMK algebra without going through our spectra.
        ``alpha0`` defaults to this model's reference calibration.

        ``alpha0_saturation_ratio`` is Sato & Furusawa 2012's ``z*_d,F / z_d,F``
        factor on the ``alpha0`` term (their eq. 33). It defaults to ``1.0``, the
        Inaniwa & Kanematsu 2018 reading, so callers driving this seam directly are
        unaffected; [`alpha_beta_from_dose_spectrum`][] supplies it from the
        spectrum when ``domain_treatment="sato_moments"``.
        """
        a0 = self.alpha0 if alpha0 is None else alpha0
        if z_1D_Gy <= 0.0:
            # No energy deposited in the domain -> no quality dependence to apply.
            return LQParameters(alpha=max(a0, 0.0), beta=self.beta0)
        alpha = a0 * alpha0_saturation_ratio + self.beta0 * z_star_1D_Gy
        beta = (z_star_1D_Gy / z_1D_Gy) * self.beta0
        return LQParameters(alpha=max(alpha, 0.0), beta=max(beta, 0.0))

    def alpha_beta_from_dose_spectrum(
        self, y_keV_um: npt.NDArray[np.float64], d_of_y: npt.NDArray[np.float64]
    ) -> LQParameters:
        """The SMK **LQ core** ``(alpha_SMK, beta_SMK)`` from the *domain* spectrum.

        Eqs. (1)-(2) need only the domain-scored ``d(y)``. This deliberately omits
        the non-LQ ``(1 + gamma D)`` factor of eq. (4), which needs the nucleus
        spectrum -- use [`survival_from_spectra`][] for the actual survival curve.
        """
        ratio = 1.0
        if self.domain_treatment == "sato_moments":
            z_F = self.z_F_Gy(y_keV_um, d_of_y)
            ratio = self.z_star_F_Gy(y_keV_um, d_of_y) / z_F if z_F > 0.0 else 1.0
        return self.alpha_beta_from_specific_energies(
            z_star_1D_Gy=self.z_star_1D_Gy(y_keV_um, d_of_y),
            z_1D_Gy=self.z_1D_Gy(y_keV_um, d_of_y),
            alpha0_saturation_ratio=ratio,
        )

    # ---- eqs. (3)-(4): the stochastic correction -----------------------
    def gamma_Gy_inv(
        self,
        dose_Gy: npt.NDArray[np.float64] | float,
        z_nucleus_Gy: float,
        lq: LQParameters,
    ) -> npt.NDArray[np.float64]:
        """Eq. (3): the nucleus-stochastics coefficient ``gamma(D)`` [Gy^-1].

        ``gamma = z_n * (0.5 (alpha + 2 beta D)^2 - beta)``, the second-order
        cumulant term for a compound-Poisson nucleus specific energy with variance
        ``Var(z_n) = z_n * D`` (derivation in the reference summary). Vanishes as
        ``z_n -> 0``, i.e. in the photon / low-LET limit.
        """
        d = np.asarray(dose_Gy, dtype=np.float64)
        slope = lq.alpha + 2.0 * lq.beta * d
        gamma: npt.NDArray[np.float64] = z_nucleus_Gy * (0.5 * slope * slope - lq.beta)
        return gamma

    def survival_from_specific_energies(
        self,
        dose_Gy: npt.NDArray[np.float64] | float,
        *,
        z_star_1D_Gy: float,
        z_1D_Gy: float,
        z_nucleus_Gy: float,
        alpha0: float | None = None,
        alpha0_saturation_ratio: float = 1.0,
    ) -> npt.NDArray[np.float64]:
        """Eq. (4) driven by microdosimetric quantities (the external-engine seam)."""
        lq = self.alpha_beta_from_specific_energies(
            z_star_1D_Gy=z_star_1D_Gy,
            z_1D_Gy=z_1D_Gy,
            alpha0=alpha0,
            alpha0_saturation_ratio=alpha0_saturation_ratio,
        )
        return self._survival(dose_Gy, lq, z_nucleus_Gy)

    def _survival(
        self,
        dose_Gy: npt.NDArray[np.float64] | float,
        lq: LQParameters,
        z_nucleus_Gy: float,
    ) -> npt.NDArray[np.float64]:
        d = np.asarray(dose_Gy, dtype=np.float64)
        gamma = self.gamma_Gy_inv(d, z_nucleus_Gy, lq)
        s = np.exp(-(lq.alpha * d + lq.beta * d * d)) * (1.0 + gamma * d)
        clipped: npt.NDArray[np.float64] = np.clip(s, 0.0, 1.0)
        return clipped

    def survival_from_spectra(
        self,
        dose_Gy: npt.NDArray[np.float64] | float,
        domain_spectrum: Spectrum,
        nucleus_spectrum: Spectrum,
    ) -> npt.NDArray[np.float64]:
        """Eq. (4) survival ``S(D)`` from the domain- and nucleus-scored spectra."""
        self._require_taylor("survival_from_spectra")
        lq = self.alpha_beta_from_spectrum(domain_spectrum)
        z_n = self.z_nucleus_Gy(nucleus_spectrum.y_keV_um, nucleus_spectrum.d)
        return self._survival(dose_Gy, lq, z_n)

    def survival_from_nucleus_density(
        self,
        dose_Gy: npt.NDArray[np.float64] | float,
        *,
        lq: LQParameters,
        z_Gy: npt.NDArray[np.float64],
        single_event_mass: npt.NDArray[np.float64],
    ) -> npt.NDArray[np.float64]:
        """Sato & Furusawa 2012 eq. (7) directly: average ``S_n`` over ``f_n(z_n; D)``.

        ``S(D) = integral exp(-(alpha z_n + beta z_n^2)) f_n(z_n; D) dz_n`` with the
        multi-event density built by
        [`pyradbio.micro.multi_event.multi_event_density`][]. The seam that lets a
        caller supply any single-event nucleus density -- notably the paper's own
        LET-based Fermi form, which is what its fig. A1 is specified in terms of.

        Unlike the ``"taylor"`` route this is a weighted average of survivals, each
        in ``(0, 1]``, so the result is a valid survival at any dose with no
        clipping and no validity ceiling.
        """
        requested = np.asarray(dose_Gy, dtype=np.float64)
        flat = np.atleast_1d(requested).ravel()
        out = np.empty(flat.shape, dtype=np.float64)
        for i, d in enumerate(flat):
            z, mass = multi_event_density(z_Gy, single_event_mass, dose_Gy=float(d))
            s_n = np.exp(-(lq.alpha * z + lq.beta * z * z))
            out[i] = float(np.sum(s_n * mass))
        # Reshape to the caller's shape -- including 0-d for a scalar dose, which
        # is what the "taylor" route returns and what float() accepts.
        reshaped: npt.NDArray[np.float64] = out.reshape(requested.shape)
        return reshaped

    def nucleus_single_event_density(
        self, ion: Ion
    ) -> tuple[npt.NDArray[np.float64], npt.NDArray[np.float64]]:
        """``f_n,1(z_n)`` for one traversal of the nucleus by ``ion``.

        Sato & Furusawa 2012 eqs. (23)-(25). Note the paper does **not** score this
        with a microdosimetry engine and says why: the PHITS microdosimetric
        function "can be applied to a water sphere of a radius ranging from 1 nm to
        1 um, which is much smaller than the typical size of cell nuclei". It builds
        ``f_n,1`` from the particle's LET and the sphere geometry instead, which is
        what this reproduces. For a mono-energetic ion the paper's LET distribution
        ``F_L(L)`` collapses to a single LET.
        """
        let_keV_um = self._stopping_power().let_keV_um(ion)
        return single_event_density(
            let_keV_um=let_keV_um,
            radius_um=self.nucleus_radius_um,
            nodes=self.nucleus_density_nodes,
        )

    @overload
    def survival(self, field: Ion | RadiationField, dose_Gy: float) -> float: ...

    @overload
    def survival(
        self, field: Ion | RadiationField, dose_Gy: npt.NDArray[np.float64]
    ) -> npt.NDArray[np.float64]: ...

    def survival(
        self, field: Ion | RadiationField, dose_Gy: float | npt.NDArray[np.float64]
    ) -> float | npt.NDArray[np.float64]:
        """Eq. (24) survival for one ion, scoring both site sizes (scalar or array dose).

        Goes through `_SpectrumMKModel.specific_energies`, so it honours
        ``site_geometry`` (cylinder by default -- the paper's own geometry), and
        ``nucleus_treatment`` decides which survival curve is evaluated.
        """
        curve = self._survival_curve(self._require_ion(field), dose_Gy)
        return float(curve) if np.ndim(dose_Gy) == 0 else curve

    def _log_kill(self, field: Ion | RadiationField, dose_Gy: float) -> float:
        """Effect of the selected nucleus treatment, without exponentiating it.

        Taylor: Inaniwa 2018 eq. (24), doi:10.1088/1361-6560/aabede.
        Numerical: log-sum-exp of Sato 2012 eq. (7), doi:10.1667/RR2842.1,
        using the same multi-event masses as [`survival`][]. Log arithmetic
        prevents survival underflow; it does not improve the FFT tail accuracy
        or extend the physical validity of the Taylor expansion.
        """
        ion = self._require_ion(field)
        z = self.specific_energies(ion)
        lq = self.alpha_beta_from_specific_energies(
            z_star_1D_Gy=z.z_star_1D_Gy,
            z_1D_Gy=z.z_1D_Gy,
            alpha0_saturation_ratio=self._alpha0_ratio(z),
        )
        if self.nucleus_treatment == "numerical":
            grid, single_mass = self.nucleus_single_event_density(ion)
            grid, mass = multi_event_density(grid, single_mass, dose_Gy=dose_Gy)
            positive = mass > 0.0
            grid = grid[positive]
            log_terms = np.log(mass[positive]) - (lq.alpha * grid + lq.beta * grid * grid)
            return max(0.0, -float(logsumexp(log_terms)))
        return self._taylor_log_kill(dose_Gy, lq, z.z_nucleus_Gy)

    def _taylor_log_kill(self, dose_Gy: float, lq: LQParameters, z_nucleus_Gy: float) -> float:
        """Eq. (24): ``alpha D + beta D^2 - log1p(gamma D)``.

        Match `_survival`'s clipping: a nonpositive correction gives
        zero survival (infinite effect), and survival above one gives zero
        effect. Neither case legitimises extrapolating the Taylor expansion.
        """
        if dose_Gy < 0.0:
            raise ValueError(f"dose must be >= 0, got {dose_Gy!r}")
        correction = float(self.gamma_Gy_inv(dose_Gy, z_nucleus_Gy, lq)) * dose_Gy
        if correction <= -1.0:
            return math.inf
        effect = lq.alpha * dose_Gy + lq.beta * dose_Gy * dose_Gy
        return max(0.0, effect - math.log1p(correction))

    def _survival_curve(
        self, ion: Ion, dose_Gy: npt.NDArray[np.float64] | float
    ) -> npt.NDArray[np.float64]:
        """Eq. (24) (or Sato eq. 7 under ``"numerical"``) on the ion's moments."""
        z = self.specific_energies(ion)
        if self.nucleus_treatment == "numerical":
            lq = self.alpha_beta_from_specific_energies(
                z_star_1D_Gy=z.z_star_1D_Gy,
                z_1D_Gy=z.z_1D_Gy,
                alpha0_saturation_ratio=self._alpha0_ratio(z),
            )
            z_grid, mass = self.nucleus_single_event_density(ion)
            return self.survival_from_nucleus_density(
                dose_Gy, lq=lq, z_Gy=z_grid, single_event_mass=mass
            )
        return self.survival_from_specific_energies(
            dose_Gy,
            z_star_1D_Gy=z.z_star_1D_Gy,
            z_1D_Gy=z.z_1D_Gy,
            z_nucleus_Gy=z.z_nucleus_Gy,
            alpha0_saturation_ratio=self._alpha0_ratio(z),
        )

    def _alpha0_ratio(self, z: SiteSpecificEnergies) -> float:
        """Sato eq. (33)'s ``z*_d,F / z_d,F`` factor from a moment bundle.

        ``1.0`` under the default ``inaniwa_moments``. Under ``sato_moments`` the
        frequency means must be present: the analytic impact-parameter route
        leaves them ``None`` because it has no event definition to average over,
        and its frequency mean therefore rides on the conventional ``b_max``
        cutoff (measured: ``z*_F/z_F`` drifts 8% over a 20x cut of ``b_max``,
        where ``z*_D/z_D`` is flat to four digits). Returning a cutoff-dependent
        ``alpha`` would be silent physics, so this raises instead.
        """
        if self.domain_treatment != "sato_moments":
            return 1.0
        if z.z_F_Gy is None or z.z_star_F_Gy is None:
            raise ValueError(
                'domain_treatment="sato_moments" needs the frequency means '
                "(Sato & Furusawa 2012 eq. 30), which the analytic "
                "impact-parameter route cannot define: it carries no minimum "
                "energy-deposition threshold, so its frequency mean depends on the "
                "conventional b_max cutoff. Inject a SpectrumEngine (which scores a "
                "real d(y)) or use domain_treatment='inaniwa_moments'."
            )
        return self.z_star_F_Gy_ratio(z)

    @staticmethod
    def z_star_F_Gy_ratio(z: SiteSpecificEnergies) -> float:
        """``z*_d,F / z_d,F`` from a moment bundle; ``1.0`` if ``z_d,F`` vanishes."""
        if z.z_F_Gy is None or z.z_star_F_Gy is None or z.z_F_Gy <= 0.0:
            return 1.0
        return z.z_star_F_Gy / z.z_F_Gy

    def _alpha_beta_ion(self, ion: Ion) -> LQParameters:
        """Eqs. (15)-(16) from the ion, in ``site_geometry`` -- the LQ core only.

        The non-LQ ``(1 + gamma D)`` factor is not in these parameters; use
        [`survival`][] / [`dose_for_survival`][] for the actual curve.
        """
        z = self.specific_energies(ion)
        return self.alpha_beta_from_specific_energies(
            z_star_1D_Gy=z.z_star_1D_Gy,
            z_1D_Gy=z.z_1D_Gy,
            alpha0_saturation_ratio=self._alpha0_ratio(z),
        )

    def _nucleus_spectrum(self, ion: Ion) -> Spectrum:
        """Nucleus-scored ``d(y)`` from [`nucleus_engine`][] (or the default engine)."""
        return self._spectrum_at_site(ion, self.nucleus_radius_um, self.nucleus_engine)

    @property
    def _uses_injected_engine(self) -> bool:
        """Either site's engine being injected puts the whole model on the d(y) route.

        Otherwise a lone ``nucleus_engine`` would be silently ignored, with the
        nucleus scored by the impact-parameter route instead.
        """
        return self.engine is not None or self.nucleus_engine is not None

    def _require_taylor(self, entry_point: str) -> None:
        """Refuse a ``d(y)``-driven entry point under the numerical nucleus route.

        Sato's eq. (7) integrates over the single-event **specific-energy** density
        ``f_n,1(z_n)``, which a nucleus-scored ``d(y)`` does not carry: ``d(y)`` is
        dose-weighted and normalised, so the event *rate* -- the Poisson lambda the
        multi-event density is built from -- has been divided out of it. Quietly
        evaluating the Taylor form instead would return a number computed by a
        route the caller did not select, so this raises. Use
        [`survival`][] /
        [`rbe_at_survival`][pyradbio.models.mkm.spectrum_models.StochasticMKM.rbe_at_survival]
        (which build the density from the ion), or pass the density yourself to
        [`survival_from_nucleus_density`][].
        """
        if self.nucleus_treatment == "numerical":
            raise ValueError(
                f"{entry_point} is driven by a nucleus d(y), which cannot supply the "
                'single-event specific-energy density that nucleus_treatment="numerical" '
                "integrates over (Sato & Furusawa 2012 eq. 7). Call the ion-driven "
                "entry point instead, pass the density to survival_from_nucleus_density, "
                'or use nucleus_treatment="taylor".'
            )

    def survival_valid_to_Gy_from_spectra(
        self, domain_spectrum: Spectrum, nucleus_spectrum: Spectrum
    ) -> float:
        """Largest dose at which ``gamma D <= expansion_validity_limit`` [Gy].

        Beyond this the truncated expansion of eq. (4) is extrapolation rather than
        a controlled approximation. Returns ``inf`` when ``z_n`` is negligible, and
        under ``nucleus_treatment="numerical"``, which does no expansion at all.
        """
        if self.nucleus_treatment == "numerical":
            return math.inf
        lq = self.alpha_beta_from_spectrum(domain_spectrum)
        z_n = self.z_nucleus_Gy(nucleus_spectrum.y_keV_um, nucleus_spectrum.d)
        return self._valid_to_Gy(lq, z_n)

    def _valid_to_Gy(self, lq: LQParameters, z_n: float) -> float:
        if z_n <= 0.0:
            return math.inf

        def excess(dose: float) -> float:
            return float(self.gamma_Gy_inv(dose, z_n, lq) * dose) - self.expansion_validity_limit

        hi = 1.0
        for _ in range(60):  # ``gamma D`` grows with dose; find a bracket
            if excess(hi) > 0.0:
                break
            hi *= 2.0
        else:
            return math.inf  # the correction never reaches the limit at any real dose
        return float(brentq(excess, 0.0, hi, xtol=1.0e-9, rtol=1.0e-12))

    # ---- RBE (numeric: eq. 24 is not LQ) --------------------------------
    def rbe_at_survival_from_spectrum(
        self,
        spectrum: Spectrum,
        survival_fraction: float = 0.1,
        *,
        nucleus_spectrum: Spectrum | None = None,
    ) -> float:
        """Iso-survival RBE from spectra, inverting eq. (4) numerically.

        Widens the family's signature with the required ``nucleus_spectrum``: the
        LQ closed form in [`pyradbio.core.lq`][] does **not** apply to SMK, and
        silently dropping the ``(1 + gamma D)`` factor would be exactly the kind of
        hidden physics ``AGENTS.md`` forbids -- so this raises rather than guesses.
        """
        if nucleus_spectrum is None:
            raise ValueError(
                "SMK survival is not LQ: it needs the nucleus-scored spectrum too. "
                "Pass nucleus_spectrum=..., or use rbe_at_survival(ion, S)."
            )
        self._require_taylor("rbe_at_survival_from_spectrum")
        lq = self.alpha_beta_from_spectrum(spectrum)
        z_n = self.z_nucleus_Gy(nucleus_spectrum.y_keV_um, nucleus_spectrum.d)
        return self._rbe_numeric(survival_fraction, lq, z_n)

    def rbe_at_survival_from_dose_spectrum(
        self,
        y_keV_um: npt.NDArray[np.float64],
        d_of_y: npt.NDArray[np.float64],
        survival_fraction: float = 0.1,
        *,
        nucleus_dose_spectrum: tuple[npt.NDArray[np.float64], npt.NDArray[np.float64]]
        | None = None,
    ) -> float:
        """``RBE_S`` from raw ``(y, d(y))`` pairs for the domain and the nucleus."""
        if nucleus_dose_spectrum is None:
            raise ValueError(
                "SMK survival is not LQ: it needs the nucleus-scored spectrum too. "
                "Pass nucleus_dose_spectrum=(y, d)."
            )
        self._require_taylor("rbe_at_survival_from_dose_spectrum")
        lq = self.alpha_beta_from_dose_spectrum(y_keV_um, d_of_y)
        z_n = self.z_nucleus_Gy(*nucleus_dose_spectrum)
        return self._rbe_numeric(survival_fraction, lq, z_n)

    def dose_for_survival(self, field: Ion | RadiationField, survival_fraction: float) -> float:
        """Dose reaching ``survival_fraction`` for one ion [Gy], on the active paths.

        The single place both treatment axes are resolved for an ion, so the
        inherited [`rbe_at_survival`][pyradbio.models.base.BiophysicalModel.rbe_at_survival]
        cannot drift away from [`survival`][]. That is not free:
        ``domain_treatment`` enters through the eq.-(33) factor on ``alpha0``, and
        ``nucleus_treatment`` decides which survival curve is inverted. Getting
        either wrong is silent -- the RBE is a perfectly plausible number either way.
        """
        ion = self._require_ion(field)
        z = self.specific_energies(ion)
        lq = self.alpha_beta_from_specific_energies(
            z_star_1D_Gy=z.z_star_1D_Gy,
            z_1D_Gy=z.z_1D_Gy,
            alpha0_saturation_ratio=self._alpha0_ratio(z),
        )
        if self.nucleus_treatment == "numerical":
            z_grid, mass = self.nucleus_single_event_density(ion)

            def curve(dose: float) -> float:
                return float(
                    self.survival_from_nucleus_density(
                        dose, lq=lq, z_Gy=z_grid, single_event_mass=mass
                    )
                )

            return self._invert_survival_curve(survival_fraction, curve)
        return self._dose_for_survival_smk(survival_fraction, lq, z.z_nucleus_Gy)

    def survival_valid_to_Gy(self, ion: Ion) -> float:
        """Where the eq.-(24) expansion stops being controlled for one ion [Gy].

        See [`survival_valid_to_Gy_from_spectra`][]; this scores the ion in
        ``site_geometry``.

        ``inf`` under ``nucleus_treatment="numerical"``: that route averages
        ``S_n`` over ``f_n(z_n; D)`` rather than expanding it, so every value is a
        weighted mean of terms in ``(0, 1]`` and there is no expansion to outgrow.
        """
        if self.nucleus_treatment == "numerical":
            return math.inf
        z = self.specific_energies(ion)
        lq = self.alpha_beta_from_specific_energies(
            z_star_1D_Gy=z.z_star_1D_Gy,
            z_1D_Gy=z.z_1D_Gy,
            alpha0_saturation_ratio=self._alpha0_ratio(z),
        )
        return self._valid_to_Gy(lq, z.z_nucleus_Gy)

    def _dose_for_survival_smk(
        self, survival_fraction: float, lq: LQParameters, z_n: float
    ) -> float:
        """Invert eq. (24) numerically: the dose reaching ``survival_fraction`` [Gy].

        Eq. (24) is **not** LQ, so the closed form in [`pyradbio.core.lq`][] does
        not apply; this is a bracketed root solve on the curve itself. Shared by
        the RBE (`_rbe_numeric`) and, in [`osmk`][],
        by the OER -- which needs the ion arm at two oxygen conditions.
        """
        return self._invert_survival_curve(
            survival_fraction, lambda dose: float(self._survival(dose, lq, z_n))
        )

    def _invert_survival_curve(
        self, survival_fraction: float, curve: Callable[[float], float]
    ) -> float:
        """Dose at which ``curve`` reaches ``survival_fraction`` [Gy].

        Neither SMK survival form is LQ, so the closed form in
        [`pyradbio.core.lq`][] does not apply to either -- this brackets and
        solves on the curve itself. Taking the curve as an argument is what lets
        the ``"numerical"`` nucleus route be inverted by the same code as the
        ``"taylor"`` one; before this existed, every RBE went through the Taylor
        form regardless of ``nucleus_treatment``.
        """
        if not 0.0 < survival_fraction <= 1.0:
            raise ValueError(f"survival fraction must be in (0, 1], got {survival_fraction!r}")
        start = max(dose_for_survival(survival_fraction, self.alpha_ref, self.beta_ref), 1.0)
        return invert_survival_curve(survival_fraction, curve, start_Gy=start)

    def _rbe_numeric(self, survival_fraction: float, lq: LQParameters, z_n: float) -> float:
        """``RBE_S = D_ref(S) / D_SMK(S)`` with ``D_SMK`` from a bracketed root solve.

        The reference arm stays the LQ photon curve (its ``z_n`` is negligible), so
        only the ion arm needs inverting.
        """
        d_ref = dose_for_survival(survival_fraction, self.alpha_ref, self.beta_ref)
        return d_ref / self._dose_for_survival_smk(survival_fraction, lq, z_n)


# ---- published parameter sets, as instances (applied with dataclasses.replace) --
# Inaniwa & Kanematsu 2018 (doi:10.1088/1361-6560/aabede) table 1, "Modified-SMK
# model" columns: least-squares fits to the Furusawa et al. 2000 3He/12C/20Ne
# survival data. Note z0 is a *free* fit parameter here, not the Kase relation of
# r_d/R_n/beta0, and R_n = 8.1 um is the measured mean nucleus radius of sixteen
# human cell lines (Suzuki et al 2000), used only to derive z_n,D.
_HSG_INANIWA2018 = StochasticMKM(
    alpha0_Gy_inv=0.174,  # Gy^-1
    beta0_Gy2=0.0568,  # Gy^-2
    domain_radius_um=0.28,
    nucleus_radius_um=8.1,
    z0_Gy=66.0,
    # 200 kVp X-ray reference for the eq.-26 biological dose (Kanai et al 2006)
    alpha_ref=0.331,  # Gy^-1
    beta_ref=0.0593,  # Gy^-2
)
# ... and the "MK model" columns of the same table, for the paper's own MK curves.
_HSG_MK_INANIWA2018 = ModifiedMKM(
    alpha0_Gy_inv=0.150,
    beta0_Gy2=0.0593,
    domain_radius_um=0.29,
    nucleus_radius_um=3.9,
    z0_Gy=55.0,
    alpha_ref=0.331,
    beta_ref=0.0593,
)
_V79_MK_INANIWA2018 = ModifiedMKM(
    alpha0_Gy_inv=0.136,
    beta0_Gy2=0.0200,
    domain_radius_um=0.24,
    nucleus_radius_um=3.6,
    z0_Gy=105.8,
)
_V79_INANIWA2018 = StochasticMKM(
    alpha0_Gy_inv=0.103,  # Gy^-1
    beta0_Gy2=0.0210,  # Gy^-2
    domain_radius_um=0.23,
    nucleus_radius_um=8.1,
    z0_Gy=122.0,
)
