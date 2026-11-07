"""Analytic impact-parameter integration of an amorphous track over a target.

For a straight ion track passing a target at perpendicular distance ``b`` (the
impact parameter), the mean specific energy deposited is ``z_1(b)``, obtained by
integrating any :class:`~pyradbio.track.base.RadialDoseModel` ``D(r)`` over the
target. Impact points are uniform in the plane, so an event's frequency weight is
``2 b / b_max^2`` out to ``b_max = R + r_max``.

From ``z_1(b)`` the single-event moments follow in closed form -- with **no**
lineal-energy spectrum, energy-loss straggling model, or delta-ray Monte Carlo:

* frequency mean ``z_F = <z_1>``,
* dose mean ``z_D = <z_1^2> / <z_1>``,
* the MKM saturation-corrected dose-mean ``z*_1D`` (Kase 2006 / Inaniwa 2010),

plus a generic :meth:`SpecificEnergyProfile.average` for any per-event functional.
This is track-model-agnostic (Kiefer-Chatterjee, Scholz-Kraft LEM I/II/III, ...).

Two target geometries are provided:

* **cylindrical / disk** (default) -- the MKM **domain**: a disk of the region
  radius perpendicular to the track, ``z_1 = (1/pi R^2) integral_disk D dA``. This
  is exactly pyMKM's ``SpecificEnergy`` geometry ("a water cylinder perpendicular
  to the ion path"), so :func:`specific_energy_impact_profile` reproduces
  ``z_bar_star_domain`` to a few percent. It is also the LEM cell-nucleus geometry:
  :func:`nuclear_functional_impact_profile` integrates an *arbitrary* functional
  of the local dose (e.g. the LEM local log-kill ``-ln S_x(D(r))``) over the disk
  and reduces it to the single-track nuclear effect ``nu(b)`` and the inactivation
  cross-section ``sigma``. The disk arc-angle weight matches
  ``models.lem.base._angular_fraction_inside``.
* **spherical** -- a spherical site, ``z_1 = (1/V) integral_V D dV`` (kept for
  comparison; **not** the MKM geometry -- it overshoots ``z*_1D`` at high energy).
"""

from __future__ import annotations

import math
from collections.abc import Callable
from dataclasses import dataclass

import numpy as np
import numpy.typing as npt
from scipy.integrate import simpson

from pyradbio.core.constants import KEVUM_PER_GY_UM2_GCM3, WATER_DENSITY_G_CM3
from pyradbio.core.radiation import Ion
from pyradbio.micro.lineal_energy import SphericalSite
from pyradbio.track.base import RadialDoseModel


def impact_parameter_grid(
    radius_um: float, r_max_um: float, *, core_nodes: int = 400, penumbra_nodes: int = 400
) -> npt.NDArray[np.float64]:
    """Impact-parameter grid resolving the crosser core and the penumbra.

    A single linear grid to ``b_max = radius + r_max`` under-samples the crosser
    core ``b <~ radius`` (which dominates the dose-mean) when ``r_max >> radius``
    at high energy. We use a fine linear grid over ``[0, 2 radius]`` plus a log
    grid over ``[2 radius, b_max]``.
    """
    b_max = radius_um + r_max_um
    core_hi = min(2.0 * radius_um, b_max)
    core = np.linspace(b_max * 1.0e-9, core_hi, core_nodes, dtype=np.float64)
    if b_max > core_hi:
        penumbra = np.logspace(
            math.log10(core_hi), math.log10(b_max), penumbra_nodes, dtype=np.float64
        )
        grid: npt.NDArray[np.float64] = np.unique(np.concatenate([core, penumbra]))
        return grid
    return core


@dataclass(frozen=True, slots=True)
class SpecificEnergyProfile:
    """Single-event specific energy ``z_1(b)`` of a track crossing the target.

    ``b_grid_um`` are impact parameters in ``[0, b_max_um]``; ``specific_energy_Gy``
    is the target-mean specific energy ``z_1(b)`` [Gy] (built for either a
    cylindrical/disk target -- the MKM domain, matching pyMKM -- or a sphere).
    Moments use the uniform-plane frequency weight ``2 b / b_max^2``; the dose-mean
    and saturation moments are ratios in which ``b_max`` cancels, so they are
    insensitive to the (conventional) ``b_max`` cutoff, while the frequency mean is not.
    """

    b_grid_um: npt.NDArray[np.float64]
    specific_energy_Gy: npt.NDArray[np.float64]
    b_max_um: float
    let_keV_um: float

    def _b(self) -> npt.NDArray[np.float64]:
        return self.b_grid_um

    def average(self, func: Callable[[npt.NDArray[np.float64]], npt.NDArray[np.float64]]) -> float:
        """Frequency average ``<func(z_1)>_F = integral func(z_1) 2b db / b_max^2``."""
        b = self._b()
        integrand = func(self.specific_energy_Gy) * 2.0 * b
        return float(simpson(integrand, x=b) / (self.b_max_um * self.b_max_um))

    def frequency_mean_Gy(self) -> float:
        """Frequency-mean single-event specific energy ``z_F = <z_1>`` [Gy]."""
        return self.average(lambda z: z)

    def dose_mean_Gy(self) -> float:
        """Dose-mean single-event specific energy ``z_D = <z_1^2>/<z_1>`` [Gy]."""
        b = self._b()
        z = self.specific_energy_Gy
        num = float(simpson(z * z * b, x=b))
        den = float(simpson(z * b, x=b))
        return num / den if den > 0.0 else 0.0

    def saturation_corrected_dose_mean_Gy(self, z0_Gy: float) -> float:
        """MKM saturation-corrected dose-mean ``z*_1D`` [Gy] (Kase 2006 eq).

        ``z*_1D = z0^2 * integral (1 - exp(-(z_1/z0)^2)) b db / integral z_1 b db``
        -- the dose-mean of the saturated single-event specific energy. ``z0`` is
        the saturation specific energy (e.g. ``specific_energy_Gy(y0, r_d)`` for
        the Kase ``y0``).
        """
        b = self._b()
        z = self.specific_energy_Gy
        # -expm1(-x) = 1 - exp(-x), accurate for small x (no cancellation).
        num = float(simpson(-np.expm1(-((z / z0_Gy) ** 2)) * b, x=b))
        den = float(simpson(z * b, x=b))
        return z0_Gy * z0_Gy * num / den if den > 0.0 else 0.0


def specific_energy_impact_profile(
    radial_dose: RadialDoseModel,
    ion: Ion,
    *,
    region_radius_um: float,
    geometry: str = "cylinder",
    radial_nodes: int = 300,
    core_nodes: int = 400,
    penumbra_nodes: int = 400,
    density_g_cm3: float = WATER_DENSITY_G_CM3,
    angular_nodes: int = 64,
) -> SpecificEnergyProfile:
    """Build the ``z_1(b)`` impact-parameter profile for ``ion`` in the target.

    Evaluates ``radial_dose`` once on a log ``s``-grid over ``[r_min, r_max]`` and
    integrates it over the target for every impact parameter (any
    :class:`~pyradbio.track.base.RadialDoseModel`).

    ``geometry`` selects the target:

    * ``"cylinder"`` (default) -- the MKM domain: a disk of radius
      ``region_radius_um`` perpendicular to the track, ``z_1 = (1/pi R^2) integral_disk
      D dA``. This is what pyMKM's ``SpecificEnergy`` uses, so it reproduces
      ``z_bar_star_domain``.
    * ``"sphere"`` -- a spherical site of radius ``region_radius_um``,
      ``z_1 = (1/V) integral_V D dV`` (kept for comparison; **not** the MKM geometry).
    """
    r_max = radial_dose.r_max_um(ion)
    r_min = radial_dose.r_min_um(ion)
    radius = region_radius_um

    s_grid = np.logspace(
        math.log10(r_min * 1.0e-3), math.log10(r_max), radial_nodes, dtype=np.float64
    )
    dose_s = np.array([radial_dose.local_dose_Gy(float(s), ion) for s in s_grid], dtype=np.float64)
    let = float(
        KEVUM_PER_GY_UM2_GCM3 * density_g_cm3 * simpson(dose_s * 2.0 * math.pi * s_grid, x=s_grid)
    )
    b_grid = impact_parameter_grid(
        radius, r_max, core_nodes=core_nodes, penumbra_nodes=penumbra_nodes
    )

    if geometry == "cylinder":
        area_integral = disk_area_integral_over_site(dose_s, s_grid, b_grid, radius_um=radius)
        z_1 = area_integral / (math.pi * radius * radius)  # disk area-mean [Gy]
    elif geometry == "sphere":
        volume_integral = SphericalSite(diameter_um=2.0 * radius).volume_integral_over_site(
            dose_s, s_grid, b_grid, angular_nodes=angular_nodes
        )
        z_1 = volume_integral / ((4.0 / 3.0) * math.pi * radius**3)  # sphere volume-mean [Gy]
    else:
        raise ValueError(f"geometry must be 'cylinder' or 'sphere', got {geometry!r}")

    return SpecificEnergyProfile(
        b_grid_um=b_grid,
        specific_energy_Gy=z_1,
        b_max_um=radius + r_max,
        let_keV_um=let,
    )


# --------------------------------------------------------------------------
# Cylindrical / disk geometry (LEM nucleus; track parallel to the axis)
# --------------------------------------------------------------------------
def disk_arc_weight(
    s_grid_um: npt.NDArray[np.float64], impact_um: npt.NDArray[np.float64], radius_um: float
) -> npt.NDArray[np.float64]:
    """Arc-angle weight ``phi_in(s, b)`` [rad], shape ``(n_b, n_s)``.

    The angular measure (0..2*pi) of the circle of radius ``s`` about a track at
    perpendicular distance ``b`` from the disk centre that lies inside the disk of
    radius ``radius_um``: ``phi_in = 2 arccos(clip((s^2 + b^2 - R^2)/(2 s b)))``,
    with the concentric ``b = 0`` limit (fully inside for ``s <= R``). Matches
    ``models.lem.base._angular_fraction_inside``; ``integral_disk f dA =
    integral_s f(s) s phi_in(s, b) ds``.
    """
    b = np.asarray(impact_um, dtype=np.float64)[:, None]
    s = np.asarray(s_grid_um, dtype=np.float64)[None, :]
    with np.errstate(divide="ignore", invalid="ignore"):
        cos_psi = np.clip((s * s + b * b - radius_um * radius_um) / (2.0 * s * b), -1.0, 1.0)
    weight = 2.0 * np.arccos(cos_psi)
    concentric = np.where(s <= radius_um, 2.0 * math.pi, 0.0)
    return np.where(b == 0.0, concentric, weight)


def disk_area_integral_over_site(
    values_s: npt.NDArray[np.float64],
    s_grid_um: npt.NDArray[np.float64],
    impact_um: npt.NDArray[np.float64],
    *,
    radius_um: float,
) -> npt.NDArray[np.float64]:
    """``integral_disk f(r_perp) dA`` [values.um^2] for a track at impact ``b``.

    General impact-parameter integral of a radially-symmetric field ``f`` (given as
    ``values_s`` on ``s_grid_um``) over a disk of radius ``radius_um`` -- the
    transverse cross-section of a LEM nucleus for a track parallel to the axis (the
    height cancels in the area-mean ``.../ (pi R^2)``). The disk analogue of
    :meth:`SphericalSite.volume_integral_over_site`.
    """
    weight = disk_arc_weight(s_grid_um, impact_um, radius_um)
    integrand = np.asarray(values_s, dtype=np.float64)[None, :] * s_grid_um[None, :] * weight
    integral: npt.NDArray[np.float64] = simpson(integrand, x=s_grid_um, axis=1)
    return integral


@dataclass(frozen=True, slots=True)
class NuclearImpactProfile:
    """Single-track nuclear effect ``nu(b)`` of a functional over a LEM disk.

    ``value_b`` is the disk-area-mean of some functional of the local dose (e.g.
    the LEM local log-kill ``-ln S_x(D(r))``) for a track at impact parameter
    ``b``. The inactivation cross-section reduces it to a lethal-event area.
    """

    b_grid_um: npt.NDArray[np.float64]
    value_b: npt.NDArray[np.float64]
    b_max_um: float
    radius_um: float
    let_keV_um: float

    def average(self, func: Callable[[npt.NDArray[np.float64]], npt.NDArray[np.float64]]) -> float:
        """Frequency average ``<func(value)>_F = integral func 2b db / b_max^2``."""
        b = self.b_grid_um
        return float(simpson(func(self.value_b) * 2.0 * b, x=b) / (self.b_max_um * self.b_max_um))

    def central_value(self) -> float:
        """The near-central-traversal value ``nu(b -> 0)`` (smallest ``b``)."""
        return float(self.value_b[0])

    def inactivation_cross_section_um2(self) -> float:
        """``sigma = integral (1 - exp(-nu(b))) 2 pi b db`` [um^2] (Scholz-Kraft).

        The lethal-event inactivation cross-section (Campbell's theorem on the
        non-overlapping single-track kills); ``-ln<S> = Phi * sigma`` in the
        low-dose limit. Matches ``models.lem.base.inactivation_cross_section_um2``.
        """
        b = self.b_grid_um
        integrand = -np.expm1(-self.value_b) * 2.0 * math.pi * b
        return float(simpson(integrand, x=b))

    def lethal_events_um2(self) -> float:
        """``integral nu(b) 2 pi b db`` [um^2] -- the *unsaturated* lethal-event area.

        The mean lethal-event yield per unit fluence, dropping the ``1-exp(-nu)``
        overkill term of :meth:`inactivation_cross_section_um2`. Matches the
        ``lethal_full`` numerator in ``models.lem.base.lethal_full_initial_slope``.
        """
        b = self.b_grid_um
        return float(simpson(self.value_b * 2.0 * math.pi * b, x=b))


def nuclear_functional_impact_profile(
    radial_dose: RadialDoseModel,
    ion: Ion,
    *,
    radius_um: float,
    transform: Callable[[npt.NDArray[np.float64]], npt.NDArray[np.float64]],
    radial_nodes: int = 400,
    core_nodes: int = 400,
    penumbra_nodes: int = 400,
    density_g_cm3: float = WATER_DENSITY_G_CM3,
) -> NuclearImpactProfile:
    """Impact-parameter profile of ``transform(D(r))`` area-averaged over a disk.

    For each impact parameter ``b``, ``value_b`` is the nuclear-disk area-mean of
    ``transform`` applied to the local dose -- e.g. ``transform = photon.log_kill``
    gives the LEM single-track nuclear log-kill ``nu(b)``. Track-model-agnostic;
    the disk radius is the (effective) nuclear radius.
    """
    r_max = radial_dose.r_max_um(ion)
    r_min = radial_dose.r_min_um(ion)

    s_grid = np.logspace(
        math.log10(r_min * 1.0e-3), math.log10(r_max), radial_nodes, dtype=np.float64
    )
    dose_s = np.array([radial_dose.local_dose_Gy(float(s), ion) for s in s_grid], dtype=np.float64)
    let = float(
        KEVUM_PER_GY_UM2_GCM3 * density_g_cm3 * simpson(dose_s * 2.0 * math.pi * s_grid, x=s_grid)
    )
    value_s = np.asarray(transform(dose_s), dtype=np.float64)

    b_grid = impact_parameter_grid(
        radius_um, r_max, core_nodes=core_nodes, penumbra_nodes=penumbra_nodes
    )
    area_integral = disk_area_integral_over_site(value_s, s_grid, b_grid, radius_um=radius_um)
    value_b = area_integral / (math.pi * radius_um * radius_um)  # disk area-mean

    return NuclearImpactProfile(
        b_grid_um=b_grid,
        value_b=value_b,
        b_max_um=radius_um + r_max,
        radius_um=radius_um,
        let_keV_um=let,
    )
