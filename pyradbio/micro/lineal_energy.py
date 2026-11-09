"""Single-event lineal-energy spectra (microdosimetry).

This module computes the single-event lineal-energy spectrum ``f1(y)`` and its
dose-weighted form ``d1(y) = y f1(y) / y_F`` for an ion crossing a small site,
together with the microdosimetric moments ``y_F``, ``y_D`` and the
saturation-corrected ``y*``.

Two engines share the [`SpectrumEngine`][] contract:

* [`AnalyticSpectrumEngine`][] -- the *analytic / statistical* (Xapsos-
  Kellerer) route: needs only a stopping-power lookup plus an energy-loss
  straggling model and a site geometry, **no radial-dose model**. Fast and
  self-contained; captures crosser straggling but not the delta-ray tail.
* [`AmorphousTrackEngine`][] -- the *amorphous track-structure* route: takes
  any [`RadialDoseModel`][] ``D(r)`` and builds the
  spectrum by the impact-parameter method. Resolves the spatial track structure
  (delta-ray tail, restricted-LET behaviour) but is deterministic (no
  straggling). The two engines cross-check in the crosser-dominated regime.

Physics
-------
For a convex site of mean chord length ``l_bar`` a single ion *crosser* with
chord ``l`` deposits, in the track-segment approximation, a mean energy
``eps_bar = LET * l`` (Kellerer 1985, "Fundamentals of microdosimetry"). The
energy actually deposited fluctuates about this mean by energy-loss straggling
with variance given by Bohr's formula (see ``BOHR_STRAGGLING_MEV2_CM2_MOL``).
Xapsos et al. (1994, Radiat. Res. 137, 152) model the conditional energy-loss
distribution as **log-normal**, matched to the Bohr mean and variance; this is
positive-definite and reproduces the Vavilov skew better than a Gaussian in
small sites. A ``"gaussian"`` option is also provided but is only valid for
large sites / weak straggling: when the mean energy of a short chord is not
large compared with its straggling width the Gaussian leaks probability to
negative (unphysical) energies, biasing ``y_F`` high -- exactly the failure the
log-normal was introduced to avoid, so log-normal is the default.

The single-event spectrum is the chord-weighted fold

    f1(y) = l_bar * integral_0^d  f_chord(l) * p_eps(y * l_bar | l)  dl ,

where ``eps = y * l_bar`` and ``p_eps`` is the (log-normal or Gaussian)
conditional energy distribution. Because straggling is mean-preserving,
``y_F = LET`` exactly for any straggling model, while ``y_D`` grows with the
straggling variance -- both used as tests.

Assumptions / limitations (honest, documented, and revisited in future work)
----------------------------------------------------------------------------
* By default unrestricted LET is used for the *mean* deposited energy, so
  ``y_F = LET`` regardless of straggling. Enabling ``restricted`` switches both
  the mean and the variance to their restricted (delta-ray-escape) values for
  sites smaller than the delta-ray range, driven by a single cut ``T_cut`` with
  range ``R(T_cut) = site diameter`` (the delta-ray range power law
  ``R(T) = r_max (T/T_max)^delta``, delta = 1.7, shared with
  [`DeltaRayMonteCarlo`][] and the Scholz-Kraft track):

  - *mean*: the restricted LET ``L_Delta = LET - Delta_L``, where ``Delta_L`` is
    the standard Berger-Seltzer restricted-stopping-power offset -- the energy
    carried out of the site by delta-rays with ``T`` in ``[T_cut, T_max]``,
    ``Delta_L = (K/2)(Z/A) rho (z_eff^2/beta^2)[ln(T_max/T_cut) -
    beta^2 (1 - T_cut/T_max)]``. This lowers ``y_F`` below LET, as a small site
    sees in a Monte-Carlo library.
  - *variance*: the Bohr variance ``Omega^2 = c*l`` (which implicitly integrates
    the single-collision spectrum up to the kinematic ``T_max``) is scaled by
    ``T_cut/T_max``, taming the otherwise-unbounded ``y_D`` of a thin, low-LET
    chord (a variance-restricted, Vavilov-like treatment -- it bounds the second
    moment without changing the log-normal shape).

  Both are inactive for large sites (``r_max <= d``), recovering the unrestricted
  ``y_F = LET`` and full Bohr variance. Note the range power law overestimates
  local containment near ``r_max ~ d`` (a delta-ray of range ``d`` still deposits
  much of its energy outside the site), so the restriction is mild there.
* By default only the *crosser* (direct) component is modelled. Setting
  ``include_delta_tail=True`` (spherical sites only) folds in an analytic
  delta-ray "toucher" tail -- the low-``y`` events from tracks whose core passes
  *outside* the site but whose delta-rays deposit energy inside it (see
  `AnalyticSpectrumEngine._delta_tail_component`). For a fully stochastic
  treatment use [`AmorphousTrackEngine`][] with a [`DeltaRayMonteCarlo`][]
  instead.
* Bohr straggling is used in its non-relativistic form; the mild relativistic
  growth of the variance is neglected (sub-percent at therapy energies).
"""

from __future__ import annotations

import math
from collections.abc import Mapping
from dataclasses import dataclass, field
from typing import Any, Literal, Protocol

import numpy as np
import numpy.typing as npt
from scipy.integrate import simpson
from scipy.ndimage import gaussian_filter1d

from pyradbio.core.constants import (
    ATOMIC_MASS_UNIT_MEV,
    BETHE_K_MEV_CM2_MOL,
    BOHR_STRAGGLING_MEV2_CM2_MOL,
    ELECTRON_MASS_MEV,
    KEVUM_PER_GY_UM2_GCM3,
    WATER_DENSITY_G_CM3,
    WATER_MEAN_EXCITATION_EV,
    WATER_Z_OVER_A,
)
from pyradbio.core.radiation import Ion
from pyradbio.track.base import RadialDoseModel
from pyradbio.track.scholz_kraft import LEM1_R_MAX_GAMMA_UM, R_MAX_DELTA
from pyradbio.track.stopping_power import StoppingPower, effective_charge

StragglingModel = Literal["lognormal", "gaussian", "none"]

# MeV^2 -> keV^2 and cm -> um path-length conversion folded into one factor:
#   Omega^2[keV^2] = (Omega^2[MeV^2]) * 1e6, with s[cm] = l[um] * 1e-4, hence
#   coefficient * l[um] * 1e-4 * 1e6 = coefficient * l[um] * 1e2.
_STRAGGLING_UM_TO_KEV2: float = 1.0e2

# Mass-stopping-power unit reduction: 1 MeV/cm = 1e3 keV / 1e4 um = 0.1 keV/um.
_MEV_CM_TO_KEV_UM: float = 0.1

# Electron (delta-ray) range in liquid water, R[um] = A * T[keV]^EXP, a power-law
# fit to the NIST ESTAR CSDA ranges over ~0.1-300 keV (reproduces the tabulated
# 1/10/50/100 keV values to <2%). Cross-checked against two independent published
# electron range-energy relations, which agree to ~7% over 1-100 keV (see
# test_delta_ray_range_matches_cole_and_estar):
#   * Cole 1969, Radiat. Res. 38, 7-33 (p.30): R = 0.0431(T+0.367)^1.77 - 0.007
#     [um], the measured 5%-transmission / practical range in unit-density matter;
#   * Waligorski, Hamm & Katz 1986, Nucl. Tracks Radiat. Meas. 11, 309
#     (doi:10.1016/1359-0189(86)90074-6), r = 6e-6 * T^1.667 g/cm^2 (T > 1 keV).
# The practical range and the CSDA path length coincide to within these tolerances
# at keV energies, so this is both the delta-ray path length and (to first order)
# the straight-line depth over which its energy is deposited.
DELTA_RANGE_A_UM: float = 0.0457
DELTA_RANGE_EXP: float = 1.748

# Local-containment factor for delta-ray energy deposition: R_dep = R_csda / this.
# It is the ratio of the CSDA range to the *radius containing most of the deposited
# energy* around a point electron source (an X90-type dose-point-kernel radius),
# which is smaller than the range because the tortuous keV-electron path folds the
# deposition back toward its origin. 2.0 sits at the aggressive-but-defensible end
# of that ratio for keV electrons in water.
#
# NOTE: this is a mean-field modelling parameter, NOT the lever for
# the residual high-energy / low-LET y_D undershoot. That undershoot was traced to
# a distribution-SHAPE mismatch (the amorphous + compound-Poisson spectrum is too
# peaked -- peak too high and missing the mid-y single-delta shoulder the reference
# has), and the electron range above is confirmed correct by three independent
# sources, so shortening R_dep is unphysical and makes the benchmark worse. The
# residual is inherent to the amorphous mean-field approach and needs full
# track-structure Monte Carlo. See docs/references/Microdosimetry.md.
DELTA_DETOUR_FACTOR: float = 2.0


@dataclass(frozen=True, slots=True)
class SphericalSite:
    """A spherical microdosimetric site of a given diameter [um].

    The chord-length distribution of a sphere under uniform isotropic randomness
    is ``f(l) = 2 l / d^2`` on ``0 <= l <= d`` with mean chord ``l_bar = 2 d / 3``
    (Kellerer 1985, "Fundamentals of microdosimetry"). These are exact and make
    the geometric limit (``y_F -> LET``) a closed-form check.
    """

    diameter_um: float

    def __post_init__(self) -> None:
        if self.diameter_um <= 0.0:
            raise ValueError("diameter_um must be > 0")

    def mean_chord_um(self) -> float:
        """Mean chord length ``l_bar = 2 d / 3`` [um]."""
        return 2.0 * self.diameter_um / 3.0

    def max_chord_um(self) -> float:
        """Longest chord (the diameter) [um]."""
        return self.diameter_um

    def chord_pdf(self, ell_um: npt.NDArray[np.float64]) -> npt.NDArray[np.float64]:
        """Chord-length pdf ``f(l) = 2 l / d^2`` (0 outside ``[0, d]``)."""
        d = self.diameter_um
        pdf = 2.0 * ell_um / (d * d)
        return np.where((ell_um >= 0.0) & (ell_um <= d), pdf, 0.0)

    def radius_um(self) -> float:
        """Site radius ``R_s = d / 2`` [um]."""
        return 0.5 * self.diameter_um

    def _chord_weight(
        self,
        s_grid_um: npt.NDArray[np.float64],
        impact_um: npt.NDArray[np.float64],
        angular_nodes: int,
    ) -> npt.NDArray[np.float64]:
        """Track-parallel chord weight ``W(s, b)`` [um], shape ``(n_b, n_s)``.

        For a straight track at perpendicular distance ``b`` from the centre, this
        is the azimuthal integral of the spherical chord ``2 sqrt(R^2 - ...)`` at
        radial distance ``s`` from the track axis (Kellerer impact-parameter
        geometry); ``integral_V f(r_perp) dV = integral_s f(s) s W(s, b) ds``.
        """
        radius = self.radius_um()
        phi = np.linspace(0.0, 2.0 * math.pi, angular_nodes)
        b = np.asarray(impact_um, dtype=np.float64)[:, None, None]
        s = np.asarray(s_grid_um, dtype=np.float64)[None, :, None]
        arg = radius * radius - b * b - s * s - 2.0 * b * s * np.cos(phi[None, None, :])
        height = 2.0 * np.sqrt(np.clip(arg, 0.0, None))
        weight: npt.NDArray[np.float64] = simpson(height, x=phi, axis=2)
        return weight

    def volume_integral_over_site(
        self,
        values_s: npt.NDArray[np.float64],
        s_grid_um: npt.NDArray[np.float64],
        impact_um: npt.NDArray[np.float64],
        *,
        angular_nodes: int = 64,
    ) -> npt.NDArray[np.float64]:
        """``integral_V f(r_perp) dV`` [values.um^3] for a track at impact ``b``.

        General impact-parameter integral of an arbitrary radially-symmetric field
        ``f`` (given as ``values_s`` sampled on ``s_grid_um``, the distance from
        the track axis) over the spherical site. This is the geometry primitive
        shared by the mean-dose / specific-energy path (``f = D(r)``) and, e.g., a
        LEM local-effect path (``f = -ln S_x(D(r))``); ``impact_um`` may be an
        array, matched in the leading axis of the result.
        """
        weight = self._chord_weight(s_grid_um, impact_um, angular_nodes)
        integrand = np.asarray(values_s, dtype=np.float64)[None, :] * s_grid_um[None, :] * weight
        integral: npt.NDArray[np.float64] = simpson(integrand, x=s_grid_um, axis=1)
        return integral

    def mean_energy_deposited_keV(
        self,
        dose_s_Gy: npt.NDArray[np.float64],
        s_grid_um: npt.NDArray[np.float64],
        impact_um: npt.NDArray[np.float64],
        *,
        angular_nodes: int = 64,
        density_g_cm3: float = WATER_DENSITY_G_CM3,
    ) -> npt.NDArray[np.float64]:
        """Mean energy ``eps(b)`` [keV] deposited by a track at impact parameter ``b``.

        ``eps(b) = C * integral_V D(r_perp) dV`` with ``C = KEVUM_PER_GY_UM2_GCM3
        * rho`` (Kellerer impact-parameter method) -- a thin wrapper over
        [`volume_integral_over_site`][] with ``f = D``. ``dose_s_Gy`` is the
        radial dose sampled once on ``s_grid_um`` and reused across all impact
        parameters; ``impact_um`` may be an array.
        """
        volume_integral = self.volume_integral_over_site(
            dose_s_Gy, s_grid_um, impact_um, angular_nodes=angular_nodes
        )
        eps: npt.NDArray[np.float64] = KEVUM_PER_GY_UM2_GCM3 * density_g_cm3 * volume_integral
        return eps


_BIG: float = 1.0e30  # sentinel for an empty t-interval (lo > hi -> zero overlap)


def _line_cylinder_chord_lengths(
    origins: npt.NDArray[np.float64],
    directions: npt.NDArray[np.float64],
    radius_um: float,
    half_height_um: float,
) -> npt.NDArray[np.float64]:
    """Chord length of each unit-direction line through a finite cylinder.

    Cylinder: axis along z, radius ``radius_um``, ``|z| <= half_height_um``,
    centred at the origin. Each chord is the overlap of the infinite-cylinder
    t-interval (a quadratic in the ray parameter) and the end-cap slab
    t-interval; length equals the overlap since the directions are unit vectors.
    Non-intersecting lines return 0.
    """
    px, py, pz = origins[:, 0], origins[:, 1], origins[:, 2]
    ux, uy, uz = directions[:, 0], directions[:, 1], directions[:, 2]

    # Infinite-cylinder interval from a*t^2 + b*t + c = 0 (radial constraint).
    a = ux * ux + uy * uy
    b = 2.0 * (px * ux + py * uy)
    c = px * px + py * py - radius_um * radius_um
    parallel = a < 1.0e-12  # line parallel to the axis
    disc = b * b - 4.0 * a * c
    sq = np.sqrt(np.maximum(disc, 0.0))
    safe_a = np.where(parallel, 1.0, a)
    t1 = (-b - sq) / (2.0 * safe_a)
    t2 = (-b + sq) / (2.0 * safe_a)
    radial_hit = (~parallel) & (disc > 0.0)
    # Parallel & inside the radius -> radial constraint is the whole line.
    parallel_inside = parallel & (c <= 0.0)
    cyl_lo = np.where(radial_hit, np.minimum(t1, t2), np.where(parallel_inside, -_BIG, _BIG))
    cyl_hi = np.where(radial_hit, np.maximum(t1, t2), np.where(parallel_inside, _BIG, -_BIG))

    # End-cap slab interval (|z| <= half_height).
    flat = np.abs(uz) < 1.0e-12
    safe_uz = np.where(flat, 1.0, uz)
    ta = (-half_height_um - pz) / safe_uz
    tb = (half_height_um - pz) / safe_uz
    slab_inside = flat & (np.abs(pz) <= half_height_um)
    cap_lo = np.where(flat, np.where(slab_inside, -_BIG, _BIG), np.minimum(ta, tb))
    cap_hi = np.where(flat, np.where(slab_inside, _BIG, -_BIG), np.maximum(ta, tb))

    overlap = np.minimum(cyl_hi, cap_hi) - np.maximum(cyl_lo, cap_lo)
    lengths: npt.NDArray[np.float64] = np.maximum(overlap, 0.0)
    return lengths


def _sample_cylinder_chords(
    diameter_um: float, height_um: float, n_samples: int, seed: int
) -> npt.NDArray[np.float64]:
    """Sample mu-randomness chord lengths of a finite cylinder [um].

    Uniform isotropic random lines are generated through the site's bounding
    sphere (isotropic direction, impact point uniform in the perpendicular disk
    of radius ``R_b``); those that intersect the cylinder yield mu-randomness
    chords -- the correct single-event (uniform isotropic random line) statistic.
    Deterministic given ``seed``.
    """
    rng = np.random.default_rng(seed)
    radius = 0.5 * diameter_um
    half_height = 0.5 * height_um
    r_bound = 0.5 * math.sqrt(diameter_um * diameter_um + height_um * height_um)

    # Isotropic directions on the unit sphere.
    uz = 2.0 * rng.random(n_samples) - 1.0
    phi = 2.0 * math.pi * rng.random(n_samples)
    sin_theta = np.sqrt(np.maximum(0.0, 1.0 - uz * uz))
    directions = np.stack([sin_theta * np.cos(phi), sin_theta * np.sin(phi), uz], axis=1)

    # Orthonormal basis perpendicular to each direction (helper avoids degeneracy).
    helper = np.where(
        (np.abs(uz) < 0.9)[:, None],
        np.array([0.0, 0.0, 1.0]),
        np.array([1.0, 0.0, 0.0]),
    )
    e1 = np.cross(directions, helper)
    e1 /= np.linalg.norm(e1, axis=1, keepdims=True)
    e2 = np.cross(directions, e1)

    # Impact point uniform in the disk of radius r_bound perpendicular to U.
    rho = r_bound * np.sqrt(rng.random(n_samples))
    psi = 2.0 * math.pi * rng.random(n_samples)
    origins = e1 * (rho * np.cos(psi))[:, None] + e2 * (rho * np.sin(psi))[:, None]

    lengths = _line_cylinder_chord_lengths(origins, directions, radius, half_height)
    return lengths[lengths > 0.0]


@dataclass(frozen=True, slots=True)
class CylindricalSite:
    """A right-circular-cylinder microdosimetric site (TEPC-like).

    Unlike the sphere, a finite cylinder has no simple closed-form chord-length
    distribution, so the mu-randomness (uniform isotropic random line) chord
    distribution is estimated once by sampling isotropic random secants and
    binned into a density (deterministic for a fixed ``seed``). Two exact checks
    anchor it: the mean chord is Cauchy's ``l_bar = 4 V / S = 2 d h / (d + 2 h)``
    and the longest chord is the diagonal ``sqrt(d^2 + h^2)``.
    """

    diameter_um: float
    height_um: float
    n_samples: int = 1_000_000
    n_bins: int = 200
    seed: int = 0
    _bin_centres: npt.NDArray[np.float64] = field(init=False, repr=False, compare=False)
    _bin_density: npt.NDArray[np.float64] = field(init=False, repr=False, compare=False)

    def __post_init__(self) -> None:
        if self.diameter_um <= 0.0 or self.height_um <= 0.0:
            raise ValueError("diameter_um and height_um must be > 0")
        if self.n_samples < 1000 or self.n_bins < 8:
            raise ValueError("n_samples >= 1000 and n_bins >= 8 are required")
        chords = _sample_cylinder_chords(
            self.diameter_um, self.height_um, self.n_samples, self.seed
        )
        density, edges = np.histogram(
            chords, bins=self.n_bins, range=(0.0, self.max_chord_um()), density=True
        )
        centres = 0.5 * (edges[:-1] + edges[1:])
        object.__setattr__(self, "_bin_centres", centres.astype(np.float64))
        object.__setattr__(self, "_bin_density", density.astype(np.float64))

    def mean_chord_um(self) -> float:
        """Cauchy mean chord ``l_bar = 4 V / S = 2 d h / (d + 2 h)`` [um]."""
        d, h = self.diameter_um, self.height_um
        return 2.0 * d * h / (d + 2.0 * h)

    def max_chord_um(self) -> float:
        """Longest chord (the cylinder diagonal) ``sqrt(d^2 + h^2)`` [um]."""
        return math.sqrt(self.diameter_um**2 + self.height_um**2)

    def chord_pdf(self, ell_um: npt.NDArray[np.float64]) -> npt.NDArray[np.float64]:
        """Sampled mu-randomness chord-length density (0 outside ``[0, max]``)."""
        pdf: npt.NDArray[np.float64] = np.interp(
            ell_um, self._bin_centres, self._bin_density, left=0.0, right=0.0
        )
        return np.where((ell_um >= 0.0) & (ell_um <= self.max_chord_um()), pdf, 0.0)


class SiteGeometry(Protocol):
    """A microdosimetric target: mean chord, longest chord, and chord pdf."""

    def mean_chord_um(self) -> float: ...
    def max_chord_um(self) -> float: ...
    def chord_pdf(self, ell_um: npt.NDArray[np.float64]) -> npt.NDArray[np.float64]: ...


@dataclass(frozen=True, slots=True)
class Spectrum:
    """A single-event lineal-energy spectrum and its moments.

    ``f`` is ``f1(y)`` normalised so ``integral f(y) dy = 1`` on ``y``. The
    dose-weighted spectrum ``d1(y) = y f(y) / y_F`` is exposed as [`d`][].
    ``meta`` records the ion, LET, site and model so a spectrum is
    self-describing.

    ``y_F_override`` / ``y_D_override`` let a caller supply pre-computed moments
    instead of the Simpson quadrature on ``(y, f)``. This is used when the
    spectrum comes from a coarse, non-uniformly binned histogram (see
    [`TabulatedLinealEnergySpectrum.to_spectrum`][]): Simpson's non-uniform
    weights are not positive there and can drive ``integral f dy`` -- and hence
    ``y_F`` -- negative, which would flip the whole dose-weighted spectrum below
    zero. The exact count-weighted moments are supplied instead.
    """

    y_keV_um: npt.NDArray[np.float64]
    f: npt.NDArray[np.float64]
    meta: Mapping[str, Any] = field(default_factory=dict)
    y_F_override: float | None = field(default=None, repr=False)
    y_D_override: float | None = field(default=None, repr=False)

    def _moment(self, power: int) -> float:
        return float(simpson(self.f * self.y_keV_um**power, x=self.y_keV_um))

    @property
    def norm(self) -> float:
        """``integral f(y) dy`` (should be ~1; deviation flags grid truncation)."""
        return self._moment(0)

    @property
    def y_F(self) -> float:
        """Frequency-mean lineal energy ``y_F = integral y f dy`` [keV/um]."""
        if self.y_F_override is not None:
            return self.y_F_override
        return self._moment(1) / self._moment(0)

    @property
    def y_D(self) -> float:
        """Dose-mean lineal energy ``y_D = integral y^2 f dy / integral y f dy``."""
        if self.y_D_override is not None:
            return self.y_D_override
        return self._moment(2) / self._moment(1)

    @property
    def d(self) -> npt.NDArray[np.float64]:
        """Dose-weighted spectrum ``d1(y) = y f(y) / y_F``."""
        result: npt.NDArray[np.float64] = self.y_keV_um * self.f / self.y_F
        return result

    @property
    def yd(self) -> npt.NDArray[np.float64]:
        """``y d1(y)`` on [`y_keV_um`][] -- the microdosimetry plotting convention.

        Against ``log y`` equal areas carry equal dose, so this is the curve
        spectra are drawn and compared as (e.g. the digitized
        [`pyradbio.micro.hartzell2025.Hartzell2025Curve.yd`][]).
        Dimensionless; ``integral yd d(ln y) = 1`` for a normalised spectrum.
        """
        result: npt.NDArray[np.float64] = self.y_keV_um * self.d
        return result

    @property
    def peak_y_keV_um(self) -> float:
        """Lineal energy [keV/um] at the maximum of [`yd`][] (the primary peak).

        A grid value, not an interpolated maximum: the first ``y`` where ``yd`` is
        largest (the same rule as
        [`pyradbio.micro.hartzell2025.Hartzell2025Curve.peak_y_keV_um`][]).
        """
        return float(self.y_keV_um[np.argmax(self.yd)])

    def y_star(self, y0_keV_um: float) -> float:
        """Saturation-corrected ``y*`` (Kase et al. 2006, doi:10.1667/RR0536.1).

        ``y* = y0^2 * integral (1 - exp(-(y/y0)^2)) f dy / integral y f dy``.
        Used by the MKM to cap the local-dose saturation of ``y``.
        """
        if y0_keV_um <= 0.0:
            raise ValueError("y0_keV_um must be > 0")
        y = self.y_keV_um
        numerator = y0_keV_um**2 * simpson((1.0 - np.exp(-((y / y0_keV_um) ** 2))) * self.f, x=y)
        return float(numerator / self._moment(1))


@dataclass(frozen=True, slots=True)
class TabulatedLinealEnergySpectrum:
    """A measured / tabulated single-event spectrum from a binned count histogram.

    Holds the raw histogram as published by microdosimetry libraries and Monte
    Carlo codes: bin *edges* ``y_edges_keV_um`` (length ``n + 1``) and the
    unnormalised count spectrum ``counts`` (length ``n``, ``N(y)``). Bins are
    represented by their geometric centre ``sqrt(lo * hi)`` (the natural choice
    for logarithmically spaced bins). [`to_spectrum`][] converts to a
    normalised [`Spectrum`][] so the moment machinery (``y_F``, ``y_D``,
    ``y*``, dose-weighting) is shared with the analytic engine.
    """

    y_edges_keV_um: npt.NDArray[np.float64]
    counts: npt.NDArray[np.float64]
    meta: Mapping[str, Any] = field(default_factory=dict)

    def __post_init__(self) -> None:
        if self.y_edges_keV_um.ndim != 1 or self.counts.ndim != 1:
            raise ValueError("y_edges_keV_um and counts must be one-dimensional")
        if self.y_edges_keV_um.size != self.counts.size + 1:
            raise ValueError("y_edges_keV_um must have exactly one more element than counts")
        if self.counts.size < 1:
            raise ValueError("at least one bin is required")
        if np.any(np.diff(self.y_edges_keV_um) <= 0.0):
            raise ValueError("y_edges_keV_um must be strictly increasing")
        if np.any(self.y_edges_keV_um <= 0.0):
            raise ValueError("lineal-energy bin edges must be positive")
        if np.any(self.counts < 0.0):
            raise ValueError("counts must be non-negative")

    @property
    def bin_centres_keV_um(self) -> npt.NDArray[np.float64]:
        """Geometric bin centres ``sqrt(lo * hi)`` [keV/um] (log-bin convention)."""
        centres: npt.NDArray[np.float64] = np.sqrt(
            self.y_edges_keV_um[:-1] * self.y_edges_keV_um[1:]
        )
        return centres

    @property
    def bin_widths_keV_um(self) -> npt.NDArray[np.float64]:
        """Bin widths ``dy`` [keV/um]."""
        widths: npt.NDArray[np.float64] = np.diff(self.y_edges_keV_um)
        return widths

    @property
    def total_counts(self) -> float:
        """Sum of all counts (the histogram normalisation)."""
        return float(self.counts.sum())

    def frequency_density(self) -> npt.NDArray[np.float64]:
        """Normalised frequency spectrum ``f1(y) = N_i / (sum(N) * dy_i)``."""
        density: npt.NDArray[np.float64] = self.counts / (
            self.total_counts * self.bin_widths_keV_um
        )
        return density

    def to_spectrum(self) -> Spectrum:
        """Convert to a normalised [`Spectrum`][] on the geometric bin centres.

        Handy for plotting/dose-weighting and interoperability with the analytic
        engine. The moments ([`y_F`][], [`y_D`][]) are computed directly from
        the count histogram, not by quadrature on this density -- the exact
        estimator, and robust for coarse binning -- and passed through as the
        [`Spectrum`][] moment overrides so the dose-weighted ``d`` normalises
        by the same (positive) ``y_F``. Quadrature on a coarse, non-uniform grid
        can otherwise return a negative ``y_F`` and flip ``d`` below zero.
        """
        return Spectrum(
            y_keV_um=self.bin_centres_keV_um,
            f=self.frequency_density(),
            meta=dict(self.meta),
            y_F_override=self.y_F,
            y_D_override=self.y_D,
        )

    @property
    def y_F(self) -> float:
        """Frequency-mean lineal energy ``sum(N y) / sum(N)`` [keV/um]."""
        y = self.bin_centres_keV_um
        return float((self.counts * y).sum() / self.counts.sum())

    @property
    def y_D(self) -> float:
        """Dose-mean lineal energy ``sum(N y^2) / sum(N y)`` [keV/um]."""
        y = self.bin_centres_keV_um
        return float((self.counts * y * y).sum() / (self.counts * y).sum())


class SpectrumEngine(Protocol):
    """Produces a single-event [`Spectrum`][] for an ion (engine facade)."""

    def single_event_spectrum(
        self, ion: Ion, y_grid_keV_um: npt.NDArray[np.float64] | None = None
    ) -> Spectrum: ...


@dataclass(frozen=True, slots=True)
class AnalyticSpectrumEngine:
    """Analytic-statistical single-event spectrum (Xapsos-Kellerer route).

    Folds the site chord-length distribution with an energy-loss straggling
    kernel (see the module docstring). Requires only a [`StoppingPower`][]
    and a [`SiteGeometry`][].

    Set ``restricted=True`` to switch the mean deposit and the straggling
    variance to their restricted (delta-ray-escape) values for sites smaller than
    the delta-ray range (see the module docstring): ``y_F`` drops below LET and
    the runaway ``y_D`` of thin, low-LET chords collapses toward a Monte-Carlo
    library, while the large-site limit is left untouched.

    Set ``include_delta_tail=True`` (spherical sites only) to fold an analytic
    delta-ray "toucher" tail into ``f(y)`` alongside the crosser term, with the
    correct relative event frequency (see `_delta_tail_component`). This
    adds the low-``y`` density the crosser-only engine lacks, pulling ``y_F`` and
    ``y_D`` down toward the track-structure / Sato-AMF values at small sites; it
    combines naturally with ``restricted`` (the energy leaving a restricted
    crosser via delta-rays is exactly what the touchers deposit in neighbours).
    The tail is a deterministic-mean approximation tuned for the MKM regime
    (sub-um domains, high-LET heavy ions); for low-LET ions in larger sites use
    [`AmorphousTrackEngine`][] with a [`DeltaRayMonteCarlo`][] instead (see
    `_delta_tail_component` for the regime caveat).
    """

    stopping_power: StoppingPower
    site: SiteGeometry
    straggling: StragglingModel = "lognormal"
    restricted: bool = False
    include_delta_tail: bool = False
    density_g_cm3: float = WATER_DENSITY_G_CM3
    chord_nodes: int = 400
    y_nodes: int = 400
    # Delta-ray toucher tail (include_delta_tail): the 1/r^2 penumbra core radius
    # (Scholz-Kraft LEM I ~10 nm) and the quadrature node counts for its radial
    # dose and impact-parameter integrals.
    delta_tail_r_min_nm: float = 10.0
    delta_tail_radial_nodes: int = 200
    delta_tail_impact_nodes: int = 400

    # ---- restricted energy loss ----------------------------------------
    def _restricted_fraction(self, ion: Ion) -> float:
        """Delta-ray energy-cut ratio ``T_cut / T_max`` [dimensionless].

        Delta-rays whose range exceeds the site carry energy out of it. Cutting
        the single-collision spectrum at ``T_cut`` with range
        ``R(T_cut) = site diameter`` gives ``T_cut/T_max = (d/r_max)^(1/delta)``
        -- the delta-ray range power law ``R(T) = r_max (T/T_max)^delta``
        (delta = 1.7) shared with [`DeltaRayMonteCarlo`][] and the
        Scholz-Kraft track, with ``r_max = gamma_LEM1 * E^delta``. This single
        ratio drives both the restricted LET ([`restricted_let_keV_um`][]) and
        the restricted variance. For a large site (``r_max <= d``) it is 1, so
        both restrictions are inactive.
        """
        d = self.site.max_chord_um()
        r_max = LEM1_R_MAX_GAMMA_UM * ion.energy_MeV_u**R_MAX_DELTA
        if r_max <= d:
            return 1.0
        return float((d / r_max) ** (1.0 / R_MAX_DELTA))

    def restricted_let_keV_um(self, ion: Ion, let_keV_um: float) -> float:
        """Restricted LET ``L_Delta`` [keV/um] for the site (mean deposit).

        Subtracts the Berger-Seltzer restricted-stopping-power offset -- the mean
        energy carried out of the site by delta-rays with ``T`` in
        ``[T_cut, T_max]`` (``T_cut`` from `_restricted_fraction`):

            Delta_L = (K/2)(Z/A) rho (z_eff^2/beta^2)
                      [ln(T_max/T_cut) - beta^2 (1 - T_cut/T_max)]

        (integral of ``T`` over the ``(1/T^2)(1 - beta^2 T/T_max)`` collision
        spectrum from ``T_cut`` to ``T_max``, ICRU 37). Returns the unrestricted
        LET when the cut is inactive, and is clamped positive if the (approximate)
        offset would exceed the LET.
        """
        frac = self._restricted_fraction(ion)
        if frac >= 1.0:
            return let_keV_um
        beta2 = ion.beta() ** 2
        z_eff = effective_charge(ion.Z, ion.beta())
        bracket = -math.log(frac) - beta2 * (1.0 - frac)  # ln(T_max/T_cut) - beta^2(1-T_cut/T_max)
        # (K/2)(Z/A) rho (z_eff^2/beta^2) * bracket, MeV/cm -> keV/um (x 0.1).
        delta_L_keV_um = (
            0.5
            * BETHE_K_MEV_CM2_MOL
            * WATER_Z_OVER_A
            * self.density_g_cm3
            * (z_eff * z_eff / beta2)
            * bracket
            * _MEV_CM_TO_KEV_UM
        )
        return max(let_keV_um - delta_L_keV_um, 1.0e-6 * let_keV_um)

    # ---- straggling ----------------------------------------------------
    def straggling_variance_keV2(
        self, ion: Ion, ell_um: npt.NDArray[np.float64]
    ) -> npt.NDArray[np.float64]:
        """Energy-loss variance ``Omega^2(l)`` [keV^2] for path ``l`` [um].

        The full Bohr variance ``c*l``, scaled by ``T_cut/T_max`` (see
        `_restricted_fraction`) when ``restricted`` is set. The scaling is a
        constant per unit length, so the variance stays linear in ``l`` and the
        log-normal remains mean-preserving about the (restricted) mean.
        """
        z_eff = effective_charge(ion.Z, ion.beta())
        coeff = (
            BOHR_STRAGGLING_MEV2_CM2_MOL
            * z_eff
            * z_eff
            * WATER_Z_OVER_A
            * self.density_g_cm3
            * _STRAGGLING_UM_TO_KEV2
        )
        if self.restricted:
            coeff *= self._restricted_fraction(ion)
        return coeff * ell_um

    # ---- default grids -------------------------------------------------
    def _default_y_grid(self, let_keV_um: float, ion: Ion) -> npt.NDArray[np.float64]:
        """Log grid spanning the crosser range and the straggling tails.

        With no straggling the spectrum is a triangle cut off exactly at
        ``y_cut``, so the top node is snapped there (Simpson on the
        piecewise-linear ``f`` is then essentially exact). With straggling the
        log-normal is skewed and, at low LET / small sites, very broad (large
        coefficient of variation), so a fixed number of straggling widths is not
        enough: the edges are widened multiplicatively by the log-normal sigma at
        a representative chord (``l_bar``) so both tails carry their moments.
        """
        l_bar = self.site.mean_chord_um()
        d = self.site.max_chord_um()
        y_mean_max = let_keV_um * d / l_bar  # mean y at the longest chord
        if self.straggling == "none":
            y_hi = y_mean_max
            y_lo = min(1.0e-2, 1.0e-3 * y_hi)
        else:
            var = float(self.straggling_variance_keV2(ion, np.array([l_bar]))[0])
            mean = let_keV_um * l_bar
            sigma_ln = math.sqrt(math.log1p(var / (mean * mean)))  # log-normal width
            # Multiplicative tails: exp(+/- 5 sigma) around the mean-y range, so
            # broad (high-cv) spectra keep both their low- and high-y mass.
            y_hi = y_mean_max * math.exp(5.0 * sigma_ln)
            y_lo = min(1.0e-2, let_keV_um * math.exp(-5.0 * sigma_ln))
        grid: npt.NDArray[np.float64] = np.logspace(
            math.log10(y_lo), math.log10(y_hi), self.y_nodes, dtype=np.float64
        )
        return grid

    # ---- conditional energy distribution -------------------------------
    @staticmethod
    def _lognormal_pdf(
        eps: npt.NDArray[np.float64], mean: npt.NDArray[np.float64], var: npt.NDArray[np.float64]
    ) -> npt.NDArray[np.float64]:
        """Log-normal pdf p(eps) matched to given ``mean`` and ``var`` (Xapsos)."""
        s2 = np.log1p(var / (mean * mean))
        mu = np.log(mean) - 0.5 * s2
        safe_eps = np.maximum(eps, 1.0e-300)
        pdf: npt.NDArray[np.float64] = np.exp(-((np.log(safe_eps) - mu) ** 2) / (2.0 * s2)) / (
            safe_eps * np.sqrt(2.0 * math.pi * s2)
        )
        return np.where(eps > 0.0, pdf, 0.0)

    @staticmethod
    def _gaussian_pdf(
        eps: npt.NDArray[np.float64], mean: npt.NDArray[np.float64], var: npt.NDArray[np.float64]
    ) -> npt.NDArray[np.float64]:
        pdf: npt.NDArray[np.float64] = np.exp(-((eps - mean) ** 2) / (2.0 * var)) / np.sqrt(
            2.0 * math.pi * var
        )
        return pdf

    # ---- engine --------------------------------------------------------
    def single_event_spectrum(
        self, ion: Ion, y_grid_keV_um: npt.NDArray[np.float64] | None = None
    ) -> Spectrum:
        if self.include_delta_tail and not isinstance(self.site, SphericalSite):
            raise ValueError(
                "include_delta_tail=True requires a SphericalSite (the toucher tail "
                "uses the spherical impact-parameter integral)"
            )
        let_inf = self.stopping_power.let_keV_um(ion)
        # The mean deposit uses restricted LET when enabled (delta-rays escape a
        # small site); everything downstream keys off this effective LET so y_F
        # follows it. The variance is restricted independently inside
        # straggling_variance_keV2 (both driven by the same T_cut/T_max).
        let = self.restricted_let_keV_um(ion, let_inf) if self.restricted else let_inf
        l_bar = self.site.mean_chord_um()
        d = self.site.max_chord_um()
        y = (
            self._default_y_grid(let, ion)
            if y_grid_keV_um is None
            else np.asarray(y_grid_keV_um, dtype=np.float64)
        )

        if self.straggling == "none":
            f = self._no_straggling_spectrum(let, l_bar, y)
        else:
            f = self._straggling_spectrum(ion, let, l_bar, d, y)

        # Normalise on the grid so moments are consistent (records the raw norm).
        raw_norm = float(simpson(f, x=y))
        if raw_norm > 0.0:
            f = f / raw_norm

        meta: dict[str, Any] = {
            "ion": ion,
            "let_keV_um": let,
            "let_unrestricted_keV_um": let_inf,
            "restricted": self.restricted,
            "delta_tail": self.include_delta_tail,
            "toucher_event_fraction": 0.0,
            "mean_chord_um": l_bar,
            "site_diameter_um": d,
            "straggling": self.straggling,
            "grid_norm_before_renorm": raw_norm,
        }

        if self.include_delta_tail:
            # Fold the delta-ray toucher tail into the (already normalised) crosser
            # f(y) with its correct event-frequency weight. The tail is built from
            # the *unrestricted* LET (the full track penumbra) regardless of the
            # crosser restriction: the energy a restricted crosser loses to escaping
            # delta-rays is what the neighbouring touchers deposit.
            tail = self._delta_tail_component(ion, let_inf, l_bar, y)
            if tail is not None:
                f_touch, w_cross, w_touch = tail
                f = w_cross * f + w_touch * f_touch
                meta["toucher_event_fraction"] = w_touch

        return Spectrum(y_keV_um=y, f=f, meta=meta)

    def _delta_tail_component(
        self, ion: Ion, let_keV_um: float, l_bar: float, y: npt.NDArray[np.float64]
    ) -> tuple[npt.NDArray[np.float64], float, float] | None:
        """Delta-ray "toucher" component ``f_touch(y)`` and its event weights.

        Models tracks whose core passes *outside* the spherical site (impact
        parameter ``b > R_s``) but whose delta-rays deposit energy inside it. The
        delta-ray penumbra is the Scholz-Kraft ``1/r^2`` amorphous dose normalised
        to the (unrestricted) LET over ``[r_min, r_max]``, with
        ``r_max = gamma_LEM1 * E^delta`` the maximum delta-ray range (the same
        power law the restricted-LET path uses). The mean toucher deposit
        ``eps(b)`` is that dose integrated over the site by the impact-parameter
        method ([`SphericalSite.mean_energy_deposited_keV`][]).

        Each impact annulus is weighted by its area (impact pdf ``2 b / b_max^2``)
        *and* by the Poisson probability that a delta-ray event actually occurs
        there, ``1 - exp(-lambda(b))`` with ``lambda(b) = eps(b) / <w>`` the mean
        number of delta-ray collisions (``<w>`` the mean single-collision energy of
        the ``1/T^2`` spectrum on ``[T_min, T_max]``). This is the same
        importance weight [`DeltaRayMonteCarlo`][] samples from: without it the
        deterministic mean assigns a non-zero ``eps`` to *every* far track, wildly
        over-counting toucher events for low-LET ions (where ``lambda << 1`` and
        most far tracks in fact deposit nothing). Touchers whose mean deposit is
        below one quantum ``T_min = I`` (the water mean excitation energy) are also
        dropped, giving a physical low-``y`` floor at ``T_min / l_bar``.

        Regime of validity: this is a deterministic-mean approximation tuned for
        the MKM regime -- small (sub-um) domains and high-LET heavy ions, where it
        reproduces the Hartzell 2025 AMF ``y_D`` to a few percent. For low-LET ions
        in larger sites (e.g. protons in a 1 um sphere) it is coarse -- delta-ray
        events are then rare and discrete, not a smooth mean, and the faithful tool
        is [`AmorphousTrackEngine`][] with a [`DeltaRayMonteCarlo`][].

        Returns ``(f_touch, w_cross, w_touch)`` with ``f_touch`` normalised on
        ``y`` and the crosser / toucher event-frequency weights (summing to 1), or
        ``None`` when the delta-ray range fits inside the site (``r_max <= R_s``,
        no touchers) so the crosser-only spectrum is returned unchanged.
        """
        site = self.site
        assert isinstance(site, SphericalSite)  # guaranteed by single_event_spectrum
        r_s = site.radius_um()
        r_min = self.delta_tail_r_min_nm * 1.0e-3
        r_max = LEM1_R_MAX_GAMMA_UM * ion.energy_MeV_u**R_MAX_DELTA
        if r_max <= r_s:
            return None

        # Scholz-Kraft 1/r^2 penumbra normalised to LET (energy conservation):
        #   KEVUM * rho * integral D(r) 2 pi r dr = LET, shape = pi(1 + 2 ln(r_max/r_min)).
        shape = math.pi * (1.0 + 2.0 * math.log(r_max / r_min))
        lam = 1.0 / (KEVUM_PER_GY_UM2_GCM3 * self.density_g_cm3 * shape)
        s_grid = np.logspace(
            math.log10(r_min * 1.0e-3),
            math.log10(r_max),
            self.delta_tail_radial_nodes,
            dtype=np.float64,
        )
        dose_s: npt.NDArray[np.float64] = lam * let_keV_um / np.maximum(s_grid, r_min) ** 2

        b_max = r_s + r_max
        b_grid = np.geomspace(r_s, b_max, self.delta_tail_impact_nodes, dtype=np.float64)
        eps = site.mean_energy_deposited_keV(
            dose_s, s_grid, b_grid, density_g_cm3=self.density_g_cm3
        )
        # eps(b) is smooth and monotone decreasing, but the (memory-bounded)
        # impact grid is comparable in size to the y grid, so histogramming it
        # directly aliases into a sawtooth. Densify the monotone y(b) curve by
        # 1-D interpolation (cheap; no extra volume integrals) so each y bin
        # receives many samples and the tail is smooth.
        n_fine = max(self.delta_tail_impact_nodes, 20 * y.size)
        b_fine = np.geomspace(r_s, b_max, n_fine, dtype=np.float64)
        eps_fine = np.interp(b_fine, b_grid, eps)
        y_b = eps_fine / l_bar

        # Drop touchers whose mean deposit is below one delta-ray quantum: a
        # physical low-y floor, and it stops the far-toucher (eps -> 0) tail from
        # over-populating the frequency spectrum.
        t_min_keV = WATER_MEAN_EXCITATION_EV * 1.0e-3
        event = y_b >= t_min_keV / l_bar
        if not bool(event.any()):
            return None

        # Impact-annulus area weight (2 pi b db -> b_hi^2 - b_lo^2) times the
        # Poisson event probability 1 - exp(-lambda(b)), lambda = eps(b)/<w>. <w>
        # is the mean single-collision energy of the 1/T^2 spectrum on
        # [T_min, T_max]; the event weight suppresses the far low-LET touchers the
        # deterministic mean would otherwise over-count (the DeltaRayMonteCarlo
        # importance weight, applied analytically here).
        t_max_keV = max_delta_ray_energy_keV(ion)
        w_mean_keV = math.log(t_max_keV / t_min_keV) / (1.0 / t_min_keV - 1.0 / t_max_keV)
        p_event = -np.expm1(-eps_fine / w_mean_keV)
        b_edges = np.empty(b_fine.size + 1, dtype=np.float64)
        b_edges[1:-1] = np.sqrt(b_fine[:-1] * b_fine[1:])
        b_edges[0] = r_s
        b_edges[-1] = b_max
        annulus = np.clip(b_edges[1:] ** 2 - b_edges[:-1] ** 2, 0.0, None)
        w = np.where(event, annulus * p_event, 0.0)
        toucher_area = float(np.sum(w))
        if toucher_area <= 0.0:
            return None

        # Histogram y(b) into a density on the shared y grid (log-bin edges).
        y_edges = np.empty(y.size + 1, dtype=np.float64)
        y_edges[1:-1] = np.sqrt(y[:-1] * y[1:])
        y_edges[0] = y[0] * y[0] / y_edges[1]
        y_edges[-1] = y[-1] * y[-1] / y_edges[-2]
        counts, _ = np.histogram(y_b, bins=y_edges, weights=w)
        f_touch = counts / np.diff(y_edges)
        norm = float(simpson(f_touch, x=y))
        if norm <= 0.0:
            return None
        f_touch = f_touch / norm

        # Event-frequency weights: crosser cross-section pi R_s^2, toucher annulus
        # pi * toucher_area; return the fractions (they sum to 1).
        total_area = r_s * r_s + toucher_area
        return f_touch, r_s * r_s / total_area, toucher_area / total_area

    def _no_straggling_spectrum(
        self, let: float, l_bar: float, y: npt.NDArray[np.float64]
    ) -> npt.NDArray[np.float64]:
        """Geometric limit: eps = LET*l, so f1(y) = f_chord(l(y)) * l_bar / LET."""
        d = self.site.max_chord_um()
        ell = y * l_bar / let  # chord that yields this y
        # Evaluate the pdf at min(l, d) so a top node landing on l == d (up to
        # rounding) keeps the triangle tip, then zero out y beyond the cutoff.
        pdf = self.site.chord_pdf(np.minimum(ell, d))
        f: npt.NDArray[np.float64] = np.where(ell <= d * (1.0 + 1.0e-12), pdf * l_bar / let, 0.0)
        return f

    def _straggling_spectrum(
        self, ion: Ion, let: float, l_bar: float, d: float, y: npt.NDArray[np.float64]
    ) -> npt.NDArray[np.float64]:
        ell = np.linspace(d / self.chord_nodes, d, self.chord_nodes, dtype=np.float64)
        f_chord = self.site.chord_pdf(ell)
        mean = let * ell  # keV
        var = np.maximum(self.straggling_variance_keV2(ion, ell), 1.0e-30)
        eps = y[:, None] * l_bar  # (y, 1) -> deposited energy per y point [keV]

        pdf = (
            self._lognormal_pdf(eps, mean[None, :], var[None, :])
            if self.straggling == "lognormal"
            else self._gaussian_pdf(eps, mean[None, :], var[None, :])
        )
        integrand = f_chord[None, :] * pdf  # (y, chord)
        f: npt.NDArray[np.float64] = l_bar * simpson(integrand, x=ell, axis=1)
        return f


def max_delta_ray_energy_keV(ion: Ion) -> float:
    """Maximum secondary-electron (delta-ray) kinetic energy ``T_max`` [keV].

    Relativistic head-on limit ``T_max = 2 m_e c^2 beta^2 gamma^2 /
    (1 + 2 gamma m_e/M + (m_e/M)^2)`` with ``M`` the ion rest mass (PDG /
    ICRU 16); it sets the upper edge of the single-collision energy-loss
    spectrum.
    """
    beta = ion.beta()
    gamma = 1.0 / math.sqrt(1.0 - beta * beta)
    mass_ratio = ELECTRON_MASS_MEV / (ion.A * ATOMIC_MASS_UNIT_MEV)
    t_max_MeV = (2.0 * ELECTRON_MASS_MEV * beta * beta * gamma * gamma) / (
        1.0 + 2.0 * gamma * mass_ratio + mass_ratio * mass_ratio
    )
    return t_max_MeV * 1.0e3


@dataclass(frozen=True, slots=True)
class DeltaRayMonteCarlo:
    """Stochastic delta-ray / collision-statistics layer (design doc section 7).

    Post-processes a deterministic amorphous profile ``eps(b)`` (the *mean* energy
    imparted at impact parameter ``b``) into a stochastic single-event spectrum by
    resampling each deposit as a **compound Poisson** sum of single-collision
    energy losses -- the Kellerer-Rossi collision-spectrum picture, which is the
    microscopic origin of *both* energy-loss straggling and delta-ray discreteness
    (Kellerer 1985, "Fundamentals of microdosimetry"; ICRU 36):

        energy = sum_{i=1}^{N} w_i ,   N ~ Poisson(eps(b) / <w>) ,
        w_i ~ single-collision spectrum  p(T) proportional to 1/T^2  on
              [T_min, T_max(ion)] (Rutherford / first-Born delta-ray spectrum).

    ``T_max`` is the kinematic maximum ([`max_delta_ray_energy_keV`][]) and
    ``T_min`` defaults to the water mean excitation energy ``I = 78 eV`` (the
    natural lower edge of the ionisation spectrum). The mean is preserved
    (``E[energy] = eps``), so:

    * many collisions (crossers, ``eps >> <w>``): the sum tends to a Gaussian ->
      the straggling-broadened peak the deterministic engine lacks;
    * ``<< 1`` collision (far touchers, ``eps << <w>``): most realisations deposit
      **zero** -- these are *non-events* and are dropped, curing the deterministic
      engine's over-populated low-``y`` toucher tail; the smallest non-zero deposit
      is one quantum ``~T_min``, giving a physical low-``y`` floor ``T_min / l_bar``.

    Sampling is by importance: impact parameters are drawn from the event
    cross-section ``2*pi*b*(1 - exp(-lambda(b)))`` so every sample is an
    energy-deposition event (crucial when ``r_max >> R_s``, where a uniform-disk
    sample would land almost entirely on empty far touchers and leave the spectrum
    shot-noise dominated). The ``event_fraction`` in ``meta`` is the analytic ratio
    of that cross-section to the whole disk. The histogram is lightly smoothed in
    log-``y`` (``smoothing_sigma_bins``, integral-preserving) to damp the residual
    shot noise of the sparse ``1/T^2`` high-``y`` tail. Deterministic for a fixed
    ``seed``.

    Delta-ray transport is handled one of two ways:

    * default (``partial_deposition=False``): the collision spectrum is *cut* at
      ``T_cut`` with ``R(T_cut) = site diameter`` (`_restricted_t_max_keV`),
      and each surviving collision deposits its full ``T`` locally. Simple, but a
      sharp cut whose surviving quanta (up to ``T_cut``) still deposit their whole
      energy in one point, which slightly over-populates the high-``y`` tail.
    * ``partial_deposition=True``: the full kinematic ``[T_min, T_max]`` spectrum
      is kept, but each collision deposits only the energy its delta-ray loses
      while crossing the site -- ``w(T) = T * phi(T)`` with ``phi`` the fraction of
      the delta's range contained in a representative in-site path ``l_bar``
      (`_deposit_fraction`). A short-range delta (``R(T) <= l_bar``) still
      deposits its full ``T``; a long-range one deposits only the ``LET_e * path``
      it drops before exiting. The compound-Poisson mean is renormalised to the
      amorphous ``eps(b)`` via the deposit mean ``<w>``, so the local dose is
      preserved while the high-``y`` tail is physically softened. This is the more
      faithful model; the restriction ``T_cut`` is then unnecessary.

    Remaining approximation: a delta's energy is imparted where the amorphous
    ``eps(b)`` places it (no explicit delta transport in space), and a single
    representative path ``l_bar`` is used for the containment fraction at every
    ``b``.

    Soft-core (Vavilov-style) two-component model
    ---------------------------------------------
    With ``soft_core=True`` (**default**) the deposit is split into a **soft core**
    and a **hard delta-ray tail** at a boundary ``T_0`` (equivalently a radius
    ``r_0 = R(T_0)``), which the [`AmorphousTrackEngine`][] uses to split the
    radial-dose integral so the split is impact-parameter dependent (far touchers
    get no soft core):

    * soft core (``T < T_0``, near-track dose ``r < r_0``): the many small
      energy transfers are deposited as a **Gaussian** of mean ``eps_soft(b)`` and
      variance ``kappa_soft * eps_soft(b)``. This is the central-limit core the
      single-component model lacks; separating the hard tail out of it sharpens the
      crosser peak.
    * hard tail (``T > T_0``): the compound-Poisson ``1/T^2`` deltas as above,
      always with partial deposition (they are long-range by construction).

    ``T_0`` is set by [`soft_cut_keV`][] (a fixed cut, default 1 keV) or, when
    [`soft_cut_from_core_radius`][] is set (**default**) and the radial-dose model
    exposes a core radius ``r_min_um``, from that core radius via the range law.

    [`soft_variance`][] picks ``kappa_soft`` (variance per unit soft energy):

    * ``"collision"`` (**default**): the soft ``1/T^2`` moments ``<T^2>/<T>``.
      Broader; empirically matches the DeCunha proton peaks best (their peaks carry
      real width the sharper core lacks).
    * ``"bohr"``: the Bohr energy-loss variance restricted to ``T_0``, divided by
      the soft stopping power (`AmorphousTrackEngine._soft_variance_coeff_keV`).
      More rigorous -- it includes the many sub-``I`` soft collisions the ``1/T^2``
      form omits -- and so is narrower; it tends to *over*-sharpen the peak.

    Model sensitivity (a feature, and a caveat): the split works well only when the
    radial-dose model concentrates a real fraction of energy in its core, so the
    soft Gaussian bridges to the hard tail. Kiefer-Chatterjee (explicit dense core,
    ~60% of near-track energy) is smooth; Scholz-Kraft's diffuse ``1/r^2`` puts only
    ~5-7% within ``r_0``, starving the soft core and leaving a dip near the
    soft/hard seam ``~T_0/l_bar``. Prefer a dense-core radial model with soft_core.

    The hard-tail extent in ``y d(y)`` is governed by the delta-ray energy-deposition
    range in partial deposition (`_deposition_range_um`): the electron CSDA
    range scaled by the detour factor for scattering, sampled against a per-delta
    in-site chord. A *contained* delta deposits its full energy (the library's
    high-``y`` shoulder); a longer-range one deposits only the ``LET_e * chord`` it
    drops in transit -- no inflated single-point lump, and no plateau.
    """

    n_samples: int = 200_000
    t_min_keV: float = WATER_MEAN_EXCITATION_EV * 1.0e-3
    # Delta-ray range-energy exponent R(T) ~ T^delta; delta = 1.7 shared with the
    # Scholz-Kraft track model's r_max = gamma * E^delta (Scholz & Kraft 1996).
    delta_range_exponent: float = R_MAX_DELTA
    partial_deposition: bool = True
    soft_core: bool = True
    soft_cut_keV: float = 1.0
    soft_cut_from_core_radius: bool = True
    soft_variance: Literal["bohr", "collision"] = "collision"
    gaussian_threshold: float = 30.0
    y_nodes: int = 200
    smoothing_sigma_bins: float = 1.0
    seed: int = 0

    @staticmethod
    def _moments_interval(t_lo_keV: float, t_hi_keV: float) -> tuple[float, float]:
        """First and second moments of ``p(T) ~ 1/T^2`` on ``[t_lo, t_hi]``."""
        norm = 1.0 / t_lo_keV - 1.0 / t_hi_keV
        m1 = math.log(t_hi_keV / t_lo_keV) / norm
        m2 = (t_hi_keV - t_lo_keV) / norm
        return m1, m2

    def _collision_moments(self, t_max_keV: float) -> tuple[float, float]:
        """First and second moments of ``p(T) ~ 1/T^2`` on ``[T_min, T_max]``."""
        return self._moments_interval(self.t_min_keV, t_max_keV)

    @staticmethod
    def _deposition_range_um(t_keV: npt.NDArray[np.float64]) -> npt.NDArray[np.float64]:
        """Straight-line energy-deposition range ``R_dep = R_csda / detour`` [um].

        The delta-ray's energy is imparted within this straight-line extent of its
        origin (scattering-aware): the ESTAR CSDA range scaled down by the electron
        detour factor. This is the range that governs *local* containment, not the
        (longer) CSDA path length.
        """
        r_csda = DELTA_RANGE_A_UM * t_keV**DELTA_RANGE_EXP
        r_dep: npt.NDArray[np.float64] = r_csda / DELTA_DETOUR_FACTOR
        return r_dep

    def _deposit_fraction(
        self, t_keV: npt.NDArray[np.float64], s_eff_um: float | npt.NDArray[np.float64]
    ) -> npt.NDArray[np.float64]:
        """Fraction ``phi(T)`` of a delta-ray's energy deposited within ``s_eff`` [um].

        With the scattering-aware energy-deposition range ``R_dep(T)``
        (`_deposition_range_um`) and the residual-range relation
        ``T(r) = (r/A_dep)^(1/EXP)``,

            phi(T) = 1                                   if R_dep(T) <= s_eff
                   = 1 - (1 - s_eff/R_dep(T))^(1/EXP)      otherwise .

        A delta whose deposition extent fits the in-site path is *contained*
        (``phi = 1``, the high-``y`` shoulder); a longer-range one deposits only the
        ``LET_e * s_eff`` it drops over that path. ``s_eff`` may be an array (one
        sampled in-site chord per delta).
        """
        r_of_t = self._deposition_range_um(t_keV)
        residual = np.clip(1.0 - s_eff_um / np.maximum(r_of_t, 1.0e-300), 0.0, 1.0)
        frac: npt.NDArray[np.float64] = np.where(
            r_of_t <= s_eff_um, 1.0, 1.0 - residual ** (1.0 / DELTA_RANGE_EXP)
        )
        return frac

    @staticmethod
    def _sphere_diameter_from_l_bar(l_bar_um: float) -> float:
        """Sphere diameter from its mean chord ``l_bar = 2 d / 3`` (spherical site)."""
        return 1.5 * l_bar_um

    def _sample_in_site_path_um(
        self, n: int, l_bar_um: float, rng: np.random.Generator
    ) -> npt.NDArray[np.float64]:
        """Sample per-delta in-site path lengths from the sphere chord pdf ``2 s / d^2``.

        A delta-ray's local deposit depends on how far it travels inside the site,
        which varies event to event; sampling the chord (mean ``l_bar``) rather than
        using it as a fixed path lets deltas that happen to be *contained* deposit
        their full energy (the high-``y`` shoulder), which a fixed ``l_bar`` misses.
        Inverse-CDF of ``2 s / d^2``: ``s = d sqrt(u)``.
        """
        d = self._sphere_diameter_from_l_bar(l_bar_um)
        s: npt.NDArray[np.float64] = d * np.sqrt(rng.random(n))
        return s

    def _deposit_moments(
        self, t_max_kin_keV: float, l_bar_um: float, nodes: int = 2048, s_nodes: int = 96
    ) -> tuple[float, float]:
        """Deposit moments ``<w>`` and ``<w^2>`` for ``w(T) = T * phi(T, s)``.

        Averages over the full kinematic ``p(T) ~ 1/T^2`` on ``[T_min, T_max]`` *and*
        the in-site chord distribution ``2 s / d^2`` (partial-deposition mode).
        ``<w>`` renormalises the compound-Poisson rate so the mean local deposit
        stays ``eps(b)``; ``<w^2>`` sets the Gaussian (large-lambda) variance.
        """
        a = self.t_min_keV
        b = t_max_kin_keV
        t = np.logspace(math.log10(a), math.log10(b), nodes, dtype=np.float64)
        d = self._sphere_diameter_from_l_bar(l_bar_um)
        s = np.linspace(d / s_nodes, d, s_nodes, dtype=np.float64)
        chord_pdf = 2.0 * s / (d * d)  # integrates to 1 on [0, d]
        r_of_t = self._deposition_range_um(t)[:, None]  # (nT, 1)
        residual = np.clip(1.0 - s[None, :] / np.maximum(r_of_t, 1.0e-300), 0.0, 1.0)
        phi = np.where(r_of_t <= s[None, :], 1.0, 1.0 - residual ** (1.0 / DELTA_RANGE_EXP))
        phi_mean = simpson(phi * chord_pdf[None, :], x=s, axis=1)  # E_s[phi](T)
        phi2_mean = simpson(phi * phi * chord_pdf[None, :], x=s, axis=1)  # E_s[phi^2](T)
        p_t = (1.0 / (t * t)) / (1.0 / a - 1.0 / b)  # normalised 1/T^2 pdf
        w_mean = float(simpson(t * phi_mean * p_t, x=t))
        w2_mean = float(simpson(t * t * phi2_mean * p_t, x=t))
        return w_mean, w2_mean

    def _restricted_t_max_keV(self, ion: Ion, r_max_um: float, site_diameter_um: float) -> float:
        """Upper edge of the *local* single-collision spectrum [keV].

        A delta-ray of energy ``T`` has range ``R(T) = r_max (T/T_max)^delta`` (the
        same power law the track model uses for the maximum delta range); one whose
        range exceeds the site mostly escapes it, so it cannot deposit its full
        energy *locally*. Restricting the collision spectrum to ``T_cut`` with
        ``R(T_cut) = site diameter`` removes the unphysical high-``y`` tail while
        preserving the mean (the escaping energy is redistributed into more
        frequent smaller local deposits). For a large site (``r_max <= diameter``)
        the cut is inactive and the full kinematic ``T_max`` is used.
        """
        t_max_kin = max_delta_ray_energy_keV(ion)
        if r_max_um <= site_diameter_um:
            return t_max_kin
        t_cut = t_max_kin * (site_diameter_um / r_max_um) ** (1.0 / self.delta_range_exponent)
        return float(min(t_max_kin, t_cut))  # spectrum() guards degenerate t_cut <= T_min

    def spectrum(
        self,
        ion: Ion,
        eps_keV: npt.NDArray[np.float64],
        b_grid_um: npt.NDArray[np.float64],
        b_max_um: float,
        l_bar_um: float,
        r_max_um: float,
        y_grid_keV_um: npt.NDArray[np.float64] | None = None,
        *,
        eps_soft_keV: npt.NDArray[np.float64] | None = None,
        soft_cut_keV: float | None = None,
        soft_var_coeff_keV: float | None = None,
    ) -> Spectrum:
        """Stochastic spectrum from the deterministic ``eps(b)`` profile.

        ``eps_soft_keV`` (the near-track / soft part of ``eps(b)``) and
        ``soft_cut_keV`` (``T_0``) enable the two-component soft-core model; the
        [`AmorphousTrackEngine`][] supplies them by splitting the radial-dose
        integral at ``r_0 = R(T_0)``. ``soft_var_coeff_keV`` is the Bohr soft-core
        variance-per-mean ``kappa_soft`` (``soft_variance="bohr"``); when absent
        the ``1/T^2`` moments are used.
        """
        if (
            self.soft_core
            and eps_soft_keV is not None
            and soft_cut_keV is not None
            and soft_cut_keV > self.t_min_keV
        ):
            return self._soft_core_spectrum(
                ion,
                eps_keV,
                eps_soft_keV,
                soft_cut_keV,
                b_grid_um,
                b_max_um,
                l_bar_um,
                r_max_um,
                y_grid_keV_um,
                soft_var_coeff_keV,
            )
        site_diameter = 2.0 * (b_max_um - r_max_um)  # R_s = b_max - r_max, d = 2 R_s
        if self.partial_deposition:
            # Keep the full kinematic spectrum; each collision deposits only the
            # in-site fraction w(T) = T*phi(T). <w>/<w^2> replace the collision
            # moments (mean local deposit is renormalised to eps(b) via <w>).
            t_max = max_delta_ray_energy_keV(ion)
            m1, m2 = self._deposit_moments(t_max, l_bar_um)
        else:
            t_max = self._restricted_t_max_keV(ion, r_max_um, site_diameter)
            m1, m2 = self._collision_moments(t_max)
        if t_max <= self.t_min_keV:
            raise ValueError(
                f"T_max ({t_max:.3g} keV) <= T_min ({self.t_min_keV:.3g} keV); "
                "ion too slow (or site too small) for a delta-ray collision spectrum"
            )
        rng = np.random.default_rng(self.seed)

        # The event probability at impact parameter b is 1 - exp(-lambda(b)) with
        # lambda(b) = eps(b)/<w> the mean number of collisions. Importance-sample b
        # from the event cross-section 2*pi*b*(1 - exp(-lambda)) so that *every*
        # sample is an energy-deposition event -- otherwise, when r_max >> R_s, the
        # far touchers (which deposit nothing) would waste almost all the samples
        # and leave the spectrum shot-noise dominated. ``event_fraction`` is the
        # analytic ratio of that cross-section to the whole disk (no MC noise).
        lam_grid = eps_keV / m1
        p_event = -np.expm1(-lam_grid)  # 1 - exp(-lambda), stable as lambda -> 0
        weight = b_grid_um * p_event
        event_fraction = float(2.0 * np.trapezoid(weight, b_grid_um) / (b_max_um * b_max_um))

        b = self._sample_impact(b_grid_um, weight, rng)
        lam = np.interp(b, b_grid_um, lam_grid)
        energy = self._sample_energies(lam, m1, m2, t_max, rng, l_bar_um)

        y_samples = energy[energy > 0.0] / l_bar_um
        f, y = self._density(y_samples, y_grid_keV_um)

        meta: dict[str, Any] = {
            "engine": "amorphous+delta_mc",
            "stochastic": "delta_ray_mc",
            "partial_deposition": self.partial_deposition,
            "t_max_keV": t_max,
            "t_min_keV": self.t_min_keV,
            "mean_collision_keV": m1,
            "event_fraction": event_fraction,
            "n_samples": self.n_samples,
        }
        return Spectrum(y_keV_um=y, f=f, meta=meta)

    def _soft_core_spectrum(
        self,
        ion: Ion,
        eps_keV: npt.NDArray[np.float64],
        eps_soft_keV: npt.NDArray[np.float64],
        soft_cut_keV: float,
        b_grid_um: npt.NDArray[np.float64],
        b_max_um: float,
        l_bar_um: float,
        r_max_um: float,
        y_grid_keV_um: npt.NDArray[np.float64] | None,
        soft_var_coeff_keV: float | None = None,
    ) -> Spectrum:
        """Two-component (Vavilov) spectrum: Gaussian soft core + hard delta tail."""
        t_max = max_delta_ray_energy_keV(ion)  # hard tail uses partial deposition
        t_0 = soft_cut_keV
        eps_soft = np.clip(eps_soft_keV, 0.0, None)
        eps_hard = np.clip(eps_keV - eps_soft, 0.0, None)
        m1_soft, m2_soft = self._moments_interval(self.t_min_keV, t_0)
        m1_hard, _ = self._moments_interval(t_0, t_max)
        # Soft-core variance-per-mean: the Bohr-restricted value (from the engine,
        # includes the many sub-I soft collisions the 1/T^2 form omits, so it is
        # narrower) or, as a fallback, the 1/T^2 soft moments.
        kappa_soft = soft_var_coeff_keV if soft_var_coeff_keV is not None else m2_soft / m1_soft

        # Event weight from the combined mean quantum count (soft + hard), so both
        # crossers (soft-dominated) and touchers (hard-only) are importance-sampled.
        lam_eff = eps_soft / m1_soft + eps_hard / m1_hard
        p_event = -np.expm1(-lam_eff)
        weight = b_grid_um * p_event
        event_fraction = float(2.0 * np.trapezoid(weight, b_grid_um) / (b_max_um * b_max_um))

        rng = np.random.default_rng(self.seed)
        b = self._sample_impact(b_grid_um, weight, rng)
        eps_soft_b = np.interp(b, b_grid_um, eps_soft)
        eps_hard_b = np.interp(b, b_grid_um, eps_hard)
        energy = self._sample_two_component(
            eps_soft_b, eps_hard_b, kappa_soft, m1_hard, t_0, t_max, l_bar_um, rng
        )

        y_samples = energy[energy > 0.0] / l_bar_um
        f, y = self._density(y_samples, y_grid_keV_um)
        meta: dict[str, Any] = {
            "engine": "amorphous+delta_mc",
            "stochastic": "delta_ray_mc",
            "partial_deposition": self.partial_deposition,
            "soft_core": True,
            "soft_cut_keV": t_0,
            "t_max_keV": t_max,
            "t_min_keV": self.t_min_keV,
            "mean_collision_keV": m1_hard,
            "event_fraction": event_fraction,
            "n_samples": self.n_samples,
        }
        return Spectrum(y_keV_um=y, f=f, meta=meta)

    def _sample_two_component(
        self,
        eps_soft_b: npt.NDArray[np.float64],
        eps_hard_b: npt.NDArray[np.float64],
        kappa_soft_keV: float,
        m1_hard: float,
        t_0_keV: float,
        t_max_keV: float,
        s_eff_um: float,
        rng: np.random.Generator,
    ) -> npt.NDArray[np.float64]:
        """Soft Gaussian core + hard partial-deposition delta tail, per sampled ``b``."""
        n = eps_soft_b.shape[0]
        # Soft core: central-limit Gaussian, mean eps_soft, variance kappa_soft*eps_soft
        # (kappa_soft = variance per unit mean deposit; compound-Poisson lambda*m2 = eps*m2/m1).
        var_soft = np.where(eps_soft_b > 0.0, eps_soft_b * kappa_soft_keV, 0.0)
        soft = np.clip(eps_soft_b + np.sqrt(var_soft) * rng.standard_normal(n), 0.0, None)

        # Hard tail: (non-truncated) compound-Poisson 1/T^2 deltas on [T_0, T_max],
        # each depositing only its in-site fraction (they are long-range by design).
        hard: npt.NDArray[np.float64] = np.zeros(n, dtype=np.float64)
        counts = rng.poisson(eps_hard_b / m1_hard)
        total = int(counts.sum())
        if total > 0:
            u = rng.random(total)
            inv_lo, inv_hi = 1.0 / t_0_keV, 1.0 / t_max_keV
            losses = 1.0 / (inv_lo - u * (inv_lo - inv_hi))
            s = self._sample_in_site_path_um(total, s_eff_um, rng)
            losses = losses * self._deposit_fraction(losses, s)
            idx = np.repeat(np.arange(n), counts)
            hard += np.bincount(idx, weights=losses, minlength=n)
        energy: npt.NDArray[np.float64] = soft + hard
        return energy

    def _sample_impact(
        self,
        b_grid_um: npt.NDArray[np.float64],
        weight: npt.NDArray[np.float64],
        rng: np.random.Generator,
    ) -> npt.NDArray[np.float64]:
        """Inverse-CDF sample of impact parameters from the event weight ``w(b)``.

        Integrates ``w(b)`` with the **trapezoid rule on the actual node spacing**
        (not a bare ``cumsum``, which silently assumes a uniform grid) so sampling
        is correct on the non-uniform, core-resolving impact grid too.
        """
        cdf = np.empty_like(b_grid_um)
        cdf[0] = 0.0
        np.cumsum(0.5 * (weight[1:] + weight[:-1]) * np.diff(b_grid_um), out=cdf[1:])
        total = float(cdf[-1])
        if total <= 0.0:
            return np.full(self.n_samples, b_grid_um[0], dtype=np.float64)
        cdf = cdf / total
        u = rng.random(self.n_samples)
        sampled: npt.NDArray[np.float64] = np.interp(u, cdf, b_grid_um)
        return sampled

    def _sample_energies(
        self,
        lam: npt.NDArray[np.float64],
        m1: float,
        m2: float,
        t_max_keV: float,
        rng: np.random.Generator,
        l_bar_um: float,
    ) -> npt.NDArray[np.float64]:
        """Deposit given an event: zero-truncated compound Poisson (Gaussian for big lam)."""
        n = lam.shape[0]
        energy: npt.NDArray[np.float64] = np.zeros(n, dtype=np.float64)
        big = lam > self.gaussian_threshold

        # Small lambda: number of collisions is a zero-truncated Poisson (an event
        # deposits at least one quantum), then sum single-collision 1/T^2 losses.
        counts = np.zeros(n, dtype=np.int64)
        small = ~big
        if np.any(small):
            counts[small] = self._zero_truncated_poisson(lam[small], rng)
        total = int(counts.sum())
        if total > 0:
            u = rng.random(total)
            inv_lo, inv_hi = 1.0 / self.t_min_keV, 1.0 / t_max_keV
            losses = 1.0 / (inv_lo - u * (inv_lo - inv_hi))  # inverse-CDF of 1/T^2
            if self.partial_deposition:
                # Deposit only the in-site fraction of each delta-ray's energy, over
                # a per-delta sampled in-site chord (contained deltas deposit fully).
                s = self._sample_in_site_path_um(total, l_bar_um, rng)
                losses = losses * self._deposit_fraction(losses, s)
            idx = np.repeat(np.arange(n), counts)
            energy += np.bincount(idx, weights=losses, minlength=n)

        # Large lambda: N >= 1 almost surely, so the sum tends to a Gaussian with
        # mean lam*m1 = eps(b) and variance lam*m2 (clipped at zero).
        if np.any(big):
            gaussian = lam * m1 + np.sqrt(lam * m2) * rng.standard_normal(n)
            energy = np.where(big, np.clip(gaussian, 0.0, None), energy).astype(np.float64)
        return energy

    @staticmethod
    def _zero_truncated_poisson(
        lam: npt.NDArray[np.float64], rng: np.random.Generator, max_iter: int = 32
    ) -> npt.NDArray[np.int64]:
        """Poisson counts conditioned on ``N >= 1`` (resample zeros; ``N=1`` as lam->0)."""
        counts: npt.NDArray[np.int64] = rng.poisson(lam)
        for _ in range(max_iter):
            zero = counts == 0
            if not bool(zero.any()):
                break
            counts[zero] = rng.poisson(lam[zero])
        counts[counts == 0] = 1  # asymptotically exact: ZTP -> 1 as lambda -> 0
        return counts

    def _density(
        self, y_samples: npt.NDArray[np.float64], y_grid_keV_um: npt.NDArray[np.float64] | None
    ) -> tuple[npt.NDArray[np.float64], npt.NDArray[np.float64]]:
        """Histogram the samples into a normalised density on log-spaced bins."""
        if y_samples.size == 0:
            raise ValueError("no energy-deposition events sampled; increase n_samples")
        if y_grid_keV_um is None:
            y_lo = max(1.0e-3, 0.5 * float(np.min(y_samples)))
            y_hi = float(np.max(y_samples))
            edges = np.logspace(math.log10(y_lo), math.log10(y_hi), self.y_nodes + 1)
        else:
            centres = np.asarray(y_grid_keV_um, dtype=np.float64)
            log_c = np.log(centres)
            mid = np.sqrt(centres[:-1] * centres[1:])
            edges = np.concatenate(
                [[centres[0] * math.exp(-0.5 * (log_c[1] - log_c[0]))], mid, [centres[-1]]]
            )
        raw_counts, edges = np.histogram(y_samples, bins=edges)
        centres = np.sqrt(edges[:-1] * edges[1:])
        widths = np.diff(edges)
        # Light Gaussian smoothing in log-y (the bins are log-spaced) to damp the
        # shot noise of the sparse 1/T^2 high-y tail; it is integral-preserving, so
        # the moments are unchanged to within the smoothing scale.
        counts = raw_counts.astype(np.float64)
        if self.smoothing_sigma_bins > 0.0:
            counts = gaussian_filter1d(counts, self.smoothing_sigma_bins, mode="nearest")
        in_range = counts.sum()
        f = counts / (in_range * widths) if in_range > 0 else counts
        return f.astype(np.float64), centres.astype(np.float64)


@dataclass(frozen=True, slots=True)
class AmorphousTrackEngine:
    """Single-event spectrum from an amorphous radial-dose (track-structure) model.

    Engine A of the lineal-energy design: it takes any
    [`RadialDoseModel`][] ``D(r)`` (Scholz-Kraft LEM I/II/
    III, or a future Kiefer-Chatterjee / Waligorski-Katz model) and a spherical
    site, and builds the spectrum by the impact-parameter method. For a track at
    perpendicular distance ``b`` the mean deposited energy ``eps(b)`` is ``D(r)``
    integrated over the site (see [`SphericalSite.mean_energy_deposited_keV`][]);
    since impact points are uniform in the plane, ``b`` has pdf ``2 b / b_max^2``
    out to ``b_max = R_s + r_max`` and ``eps(b)`` is monotone decreasing, so

        F(y) = P(Y <= y) = 1 - (b(y) / b_max)^2 ,   f1(y) = dF/dy
             = (2 b / b_max^2) / |dy/db| ,           y = eps(b) / l_bar .

    Complementary to [`AnalyticSpectrumEngine`][]: it resolves the spatial
    track structure explicitly, so it *includes* the low-``y`` delta-ray tail from
    "toucher" tracks (``b > R_s``) and automatically uses only the energy deposited
    inside the site (restricted-LET behaviour for sites smaller than ``r_max``).
    It is **deterministic**, though: ``eps(b)`` is the amorphous mean, with no
    event-to-event energy-loss straggling. In the large-site limit
    (``R_s >> r_max``) it reduces to the geometric chord spectrum with
    ``y_F -> LET`` and ``y_D -> (9/8) LET`` -- the cross-check against the analytic
    engine's ``straggling="none"`` mode.

    Limitations (documented, future stochastic layer):

    * Because the mean radial dose smears discrete delta-rays into a *continuous*
      deposit, distant toucher tracks (``b`` up to ``b_max``) all register a tiny
      non-zero ``eps`` -- so the deterministic low-``y`` tail is **over-populated**
      and ``y_F`` (frequency-weighted) is biased low for sites much smaller than
      ``r_max``. Passing a [`DeltaRayMonteCarlo`][] as ``stochastic`` resamples
      ``eps(b)`` with collision statistics, which both adds straggling to the
      crosser peak and cures the over-populated tail (see that class). Without it
      the dose-mean ``y_D`` is the robust moment.
    * The result is a single spherical geometry; other amorphous-capable sites
      would provide their own ``eps(b)`` (e.g. cylinder with orientation
      sampling).
    """

    radial_dose: RadialDoseModel
    site: SphericalSite
    stopping_power: StoppingPower | None = None
    stochastic: DeltaRayMonteCarlo | None = None
    density_g_cm3: float = WATER_DENSITY_G_CM3
    impact_nodes: int = 400
    radial_nodes: int = 200
    angular_nodes: int = 64
    y_nodes: int = 400

    def _soft_split_radius(self, ion: Ion, r_max_um: float) -> tuple[float, float]:
        """Soft/hard boundary ``(r_0 [um], T_0 [keV])`` for the two-component model.

        Uses the radial-dose core radius ``r_min_um`` when the stochastic layer
        asks for it (``soft_cut_from_core_radius``); otherwise the fixed
        ``soft_cut_keV``. The two are kept consistent through the delta-ray range
        law ``R(T) = r_max (T/T_max)^delta``.
        """
        assert self.stochastic is not None
        mc = self.stochastic
        t_max_kin = max_delta_ray_energy_keV(ion)
        delta = mc.delta_range_exponent
        if mc.soft_cut_from_core_radius:
            r_0 = min(max(self.radial_dose.r_min_um(ion), 0.0), r_max_um)
            t_0 = t_max_kin * (r_0 / r_max_um) ** (1.0 / delta)
        else:
            t_0 = mc.soft_cut_keV
            r_0 = r_max_um * (t_0 / t_max_kin) ** delta
        return r_0, t_0

    def _soft_variance_coeff_keV(self, ion: Ion, let_keV_um: float, t_0_keV: float) -> float:
        """Bohr soft-core variance-per-mean ``kappa_soft = Omega^2_soft / LET_soft`` [keV].

        The soft core's variance is ``kappa_soft * eps_soft(b)``. ``Omega^2_soft`` is
        the Bohr energy-loss variance restricted to ``T < T_0`` (``Omega^2_Bohr *
        T_0/T_max``); ``LET_soft = LET - Delta_L(T_0)`` is the soft stopping power,
        with ``Delta_L`` the Berger-Seltzer offset (energy to hard deltas
        ``T > T_0``). Using the true Bohr straggling -- which includes the many
        sub-``I`` soft collisions -- gives a narrower core than the ``1/T^2``
        moments taken down to ``T_min = I``.
        """
        beta = ion.beta()
        beta2 = beta * beta
        z_eff = effective_charge(ion.Z, beta)
        t_max = max_delta_ray_energy_keV(ion)
        omega2_soft = (
            BOHR_STRAGGLING_MEV2_CM2_MOL
            * z_eff
            * z_eff
            * WATER_Z_OVER_A
            * self.density_g_cm3
            * _STRAGGLING_UM_TO_KEV2
            * (t_0_keV / t_max)
        )
        delta_L = (
            0.5
            * BETHE_K_MEV_CM2_MOL
            * WATER_Z_OVER_A
            * self.density_g_cm3
            * (z_eff * z_eff / beta2)
            * (math.log(t_max / t_0_keV) - beta2 * (1.0 - t_0_keV / t_max))
            * _MEV_CM_TO_KEV_UM
        )
        let_soft = max(let_keV_um - delta_L, 1.0e-6 * let_keV_um)
        return float(omega2_soft / let_soft)

    def _impact_profile(
        self, ion: Ion
    ) -> tuple[
        npt.NDArray[np.float64],
        npt.NDArray[np.float64],
        npt.NDArray[np.float64] | None,
        float | None,
        float,
        float,
        float,
    ]:
        """Deterministic profile: ``(b_grid, eps, eps_soft, T_0, b_max, l_bar, LET)``.

        ``eps_soft`` and ``T_0`` are ``None`` unless the stochastic layer requests
        the soft-core split, in which case ``eps_soft(b)`` is the near-track
        (``r < r_0``) part of ``eps(b)``.
        """
        r_max = self.radial_dose.r_max_um(ion)
        r_min = self.radial_dose.r_min_um(ion)
        b_max = self.site.radius_um() + r_max
        l_bar = self.site.mean_chord_um()

        # Radial dose sampled once on a log grid over the whole track (core + tail).
        s_grid = np.logspace(
            math.log10(r_min * 1.0e-3), math.log10(r_max), self.radial_nodes, dtype=np.float64
        )
        dose_s = np.array(
            [self.radial_dose.local_dose_Gy(float(s), ion) for s in s_grid], dtype=np.float64
        )
        # LET recovered from the same profile (the model's normalisation), for meta.
        let = float(
            KEVUM_PER_GY_UM2_GCM3
            * self.density_g_cm3
            * simpson(dose_s * 2.0 * math.pi * s_grid, x=s_grid)
        )
        # eps(b) on a *core-resolving* impact-parameter grid. A uniform grid with
        # spacing b_max/N misses the crosser core (b < R_s) entirely once
        # r_max >> R_s (high-energy ions), collapsing the spectrum to the toucher
        # tail. Split the nodes geometrically: half across the core [b_min, R_s]
        # and half across the tail [R_s, b_max], so both are always resolved.
        r_s = self.site.radius_um()
        b_min = min(b_max, r_s) * 1.0e-4
        if b_min < r_s < b_max:
            n_core = self.impact_nodes // 2
            b_grid = np.concatenate(
                [
                    np.geomspace(b_min, r_s, n_core, endpoint=False),
                    np.geomspace(r_s, b_max, self.impact_nodes - n_core),
                ]
            )
        else:
            b_grid = np.geomspace(b_min, b_max, self.impact_nodes, dtype=np.float64)
        eps = self.site.mean_energy_deposited_keV(
            dose_s,
            s_grid,
            b_grid,
            angular_nodes=self.angular_nodes,
            density_g_cm3=self.density_g_cm3,
        )

        eps_soft: npt.NDArray[np.float64] | None = None
        t_0: float | None = None
        if getattr(self.stochastic, "soft_core", False):
            r_0, t_0 = self._soft_split_radius(ion, r_max)
            # eps_soft(b) is the same site integral restricted to the near-track
            # core r < r_0 (the integral is linear in D, so mask the dose profile).
            dose_soft = np.where(s_grid < r_0, dose_s, 0.0)
            eps_soft = self.site.mean_energy_deposited_keV(
                dose_soft,
                s_grid,
                b_grid,
                angular_nodes=self.angular_nodes,
                density_g_cm3=self.density_g_cm3,
            )
        return b_grid, eps, eps_soft, t_0, b_max, l_bar, let

    def single_event_spectrum(
        self, ion: Ion, y_grid_keV_um: npt.NDArray[np.float64] | None = None
    ) -> Spectrum:
        b_grid, eps, eps_soft, t_0, b_max, l_bar, let = self._impact_profile(ion)

        meta: dict[str, Any] = {
            "ion": ion,
            "let_keV_um": let,
            "mean_chord_um": l_bar,
            "site_diameter_um": self.site.diameter_um,
            "b_max_um": b_max,
            "engine": "amorphous",
            "radial_dose": type(self.radial_dose).__name__,
        }

        if self.stochastic is not None:
            r_max = b_max - self.site.radius_um()
            soft_var_coeff: float | None = None
            if t_0 is not None and getattr(self.stochastic, "soft_variance", "") == "bohr":
                soft_var_coeff = self._soft_variance_coeff_keV(ion, let, t_0)
            stochastic = self.stochastic.spectrum(
                ion,
                eps,
                b_grid,
                b_max,
                l_bar,
                r_max,
                y_grid_keV_um,
                eps_soft_keV=eps_soft,
                soft_cut_keV=t_0,
                soft_var_coeff_keV=soft_var_coeff,
            )
            return Spectrum(
                y_keV_um=stochastic.y_keV_um, f=stochastic.f, meta={**meta, **stochastic.meta}
            )

        y_b = eps / l_bar
        f, y = self._spectrum_from_epsilon(y_b, b_grid, b_max, y_grid_keV_um)
        # Exact impact-parameter moments (binning-independent): the frequency pdf
        # of the impact b is 2b/b_max^2, so y_F/y_D are integrals over b, immune to
        # the histogram's discretisation. Supplied as overrides so the reported
        # moments stay exact while f(y) carries the (binned) shape.
        y_f, y_d = self._exact_moments(y_b, b_grid, b_max)
        return Spectrum(y_keV_um=y, f=f, meta=meta, y_F_override=y_f, y_D_override=y_d)

    @staticmethod
    def _exact_moments(
        y_b: npt.NDArray[np.float64], b_grid: npt.NDArray[np.float64], b_max: float
    ) -> tuple[float | None, float | None]:
        """Exact ``(y_F, y_D)`` from the impact-parameter integral (pdf ``2b/b_max^2``)."""
        w = 2.0 * b_grid / (b_max * b_max)
        m0 = float(simpson(w, x=b_grid))
        m1 = float(simpson(y_b * w, x=b_grid))
        m2 = float(simpson(y_b * y_b * w, x=b_grid))
        if m0 <= 0.0 or m1 <= 0.0:
            return None, None
        return m1 / m0, m2 / m1

    def _spectrum_from_epsilon(
        self,
        y_b: npt.NDArray[np.float64],
        b_grid: npt.NDArray[np.float64],
        b_max: float,
        y_grid_keV_um: npt.NDArray[np.float64] | None,
    ) -> tuple[npt.NDArray[np.float64], npt.NDArray[np.float64]]:
        """Bin the monotone-*ish* ``y(b)`` into a normalised density ``f1(y)``.

        Each impact-parameter node carries the frequency weight of its annulus
        (impact density ``2b/b_max^2``, i.e. weight ``(b_hi^2 - b_lo^2)/b_max^2``
        between mid-point edges); the weights are **histogrammed** into y-bins.
        This is what a Monte-Carlo over impact points would produce: it conserves
        probability exactly, needs no ``1/|dy/db|`` change of variables (whose
        integrable singularity at the track edge ``y -> 0`` otherwise collapses the
        moments), and -- unlike a CDF inversion -- tolerates the small
        non-monotonicity of the amorphous ``eps(b)`` near the core/tail transition.
        """
        n = b_grid.size
        b_edges = np.empty(n + 1, dtype=np.float64)
        b_edges[1:-1] = np.sqrt(b_grid[:-1] * b_grid[1:])  # geometric mid-points
        b_edges[0] = b_grid[0] * b_grid[0] / b_edges[1]  # mirror the innermost edge
        b_edges[-1] = b_max
        w = np.clip((b_edges[1:] ** 2 - b_edges[:-1] ** 2) / (b_max * b_max), 0.0, None)

        pos = y_b > 0.0
        if int(pos.sum()) < 2:
            y = np.asarray(y_grid_keV_um, dtype=np.float64) if y_grid_keV_um is not None else b_grid
            return np.zeros_like(y), y
        yb, wb = y_b[pos], w[pos]

        if y_grid_keV_um is None:
            y_hi = float(yb.max())
            y_lo = max(y_hi * 1.0e-6, float(yb.min()))
            y_edges = np.logspace(math.log10(y_lo), math.log10(y_hi), self.y_nodes + 1)
            y = np.sqrt(y_edges[:-1] * y_edges[1:])  # geometric bin centres
        else:
            y = np.asarray(y_grid_keV_um, dtype=np.float64)
            y_edges = np.empty(y.size + 1, dtype=np.float64)
            y_edges[1:-1] = np.sqrt(y[:-1] * y[1:])
            y_edges[0] = y[0] * y[0] / y_edges[1]
            y_edges[-1] = y[-1] * y[-1] / y_edges[-2]

        counts, _ = np.histogram(yb, bins=y_edges, weights=wb)
        f = counts / np.diff(y_edges)
        norm = float(simpson(f, x=y))
        if norm > 0.0:
            f = f / norm
        return f, y


@dataclass(frozen=True, slots=True)
class LinealEnergyCalculator:
    """Facade: assemble an ion + stopping power + site + engine, get spectra.

    By default it uses the [`AnalyticSpectrumEngine`][] with log-normal
    straggling. Swap ``straggling`` or inject a different ``engine`` (any
    [`SpectrumEngine`][]) to change the physics without touching callers.
    """

    stopping_power: StoppingPower
    site: SiteGeometry
    straggling: StragglingModel = "lognormal"
    restricted: bool = False
    engine: SpectrumEngine | None = None

    def _engine(self) -> SpectrumEngine:
        if self.engine is not None:
            return self.engine
        return AnalyticSpectrumEngine(
            stopping_power=self.stopping_power,
            site=self.site,
            straggling=self.straggling,
            restricted=self.restricted,
        )

    def spectrum(self, ion: Ion, y_grid_keV_um: npt.NDArray[np.float64] | None = None) -> Spectrum:
        """Single-event lineal-energy spectrum for ``ion``."""
        return self._engine().single_event_spectrum(ion, y_grid_keV_um)
