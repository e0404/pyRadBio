"""Local-scale objects of NanOx: the nanometric target, its restricted
specific-energy spectrum, and the effective local lethal function.

Monini, Cunha, Chollier, Testa & Beuve 2020, *Radiat. Res.* **193**(4):331,
doi:10.1667/RR15463.1 -- the paper that determines the ELLF.
"""

from __future__ import annotations

from collections.abc import Sequence
from dataclasses import dataclass

import numpy as np
from numpy.typing import NDArray
from scipy.special import erf


@dataclass(frozen=True, slots=True)
class LocalTarget:
    """A nanometric target, modelled as a cylinder (2023 Simplification 4).

    ``diameter_nm`` is a **diameter** and ``length_nm`` a length. Alcocer-Avila
    et al. 2023 states "cylinders with length Lt = 10 nm and diameter dt =
    20 nm"; the 2022 figure captions describe the same object as "10 nm radius
    and length". The size is deliberately independent of radiation quality and
    cell line -- it is meant to cover the extension of a DSB (up to ~6 nm, or
    20 base pairs) plus the ~4 nm over which reactive species can still damage
    the site.
    """

    diameter_nm: float = 20.0
    length_nm: float = 10.0

    def __post_init__(self) -> None:
        if self.diameter_nm <= 0.0:
            raise ValueError(f"diameter_nm must be > 0, got {self.diameter_nm!r}")
        if self.length_nm <= 0.0:
            raise ValueError(f"length_nm must be > 0, got {self.length_nm!r}")

    def radius_nm(self) -> float:
        """Half the diameter. Present because both conventions appear in the
        NanOx corpus and confusing them is a factor of two."""
        return 0.5 * self.diameter_nm

    def volume_nm3(self) -> float:
        return float(np.pi * self.radius_nm() ** 2 * self.length_nm)


DEFAULT_LOCAL_TARGET = LocalTarget()
"""The one local-target geometry used in every published NanOx calculation."""


@dataclass(frozen=True, slots=True)
class LocalTargetSpectrum:
    """``dP/dz`` in the nanometric targets, as a discrete probability mass.

    This is the quantity Alcocer-Avila et al. 2022 eq. (26) integrates against
    the ELLF. It is produced by the group's **LQD** Monte Carlo track-structure
    code and is *not* published in tabular form anywhere in the NanOx corpus --
    see ``docs/references/NanOx.md``. This container exists so that a caller who
    has such a spectrum (from LQD, Geant4-DNA, or their own simulation) can feed
    it in; this package does not manufacture one.

    ``z_Gy`` are bin representatives and ``probability`` the mass in each bin,
    summing to 1, which is the natural shape of a Monte-Carlo histogram.
    ``mean_hit_z_Gy`` is ``<z>_hit``, the mean restricted specific energy over
    the targets that *received* energy -- eq. (26)'s denominator, and not
    recoverable from a normalised spectrum alone.

    The integral is evaluated as ``sum_i p_i F(z_i)``. Supply a grid fine enough
    that ``F`` is close to linear across a bin: ``F`` is an error function whose
    rise has width ``sigma``, so bins much narrower than ``sigma`` are needed
    near the threshold (see AGENTS.md -> Numerics on evaluating products at bin
    representatives).
    """

    z_Gy: tuple[float, ...]
    probability: tuple[float, ...]
    mean_hit_z_Gy: float

    def __post_init__(self) -> None:
        if len(self.z_Gy) != len(self.probability):
            raise ValueError(
                f"z_Gy and probability must have equal length, "
                f"got {len(self.z_Gy)} and {len(self.probability)}"
            )
        if len(self.z_Gy) == 0:
            raise ValueError("spectrum must have at least one bin")
        if any(z < 0.0 for z in self.z_Gy):
            raise ValueError("z_Gy must be >= 0")
        if any(p < 0.0 for p in self.probability):
            raise ValueError("probability must be >= 0")
        total = float(sum(self.probability))
        if abs(total - 1.0) > 1e-6:
            raise ValueError(f"probability must be normalised to 1, got {total!r}")
        if self.mean_hit_z_Gy <= 0.0:
            raise ValueError(f"mean_hit_z_Gy must be > 0, got {self.mean_hit_z_Gy!r}")

    @classmethod
    def delta(cls, z_Gy: float, mean_hit_z_Gy: float) -> LocalTargetSpectrum:
        """A single-valued spectrum: every hit target receives exactly ``z_Gy``.

        Not physical -- the real distributions span decades (2022 Fig. 4) -- but
        it is the analytic limit in which eq. (42) reduces to ``F(z)``, which
        makes it the right fixture for testing the integrator.
        """
        return cls(z_Gy=(float(z_Gy),), probability=(1.0,), mean_hit_z_Gy=float(mean_hit_z_Gy))

    @classmethod
    def from_density(
        cls,
        z_Gy: Sequence[float] | NDArray[np.float64],
        dP_dz: Sequence[float] | NDArray[np.float64],
        mean_hit_z_Gy: float,
    ) -> LocalTargetSpectrum:
        """Build from a probability *density* [Gy^-1], as the papers write it.

        The density is converted to bin masses by the trapezoid rule and
        renormalised, so a slightly off-normalised simulation output is
        accepted.
        """
        z = np.asarray(z_Gy, dtype=float)
        f = np.asarray(dP_dz, dtype=float)
        if z.size < 2:
            raise ValueError("from_density needs at least two nodes")
        if np.any(np.diff(z) <= 0.0):
            raise ValueError("z_Gy must be strictly increasing")
        # Trapezoid weights: each node carries half of each adjacent interval.
        widths = np.zeros_like(z)
        widths[:-1] += 0.5 * np.diff(z)
        widths[1:] += 0.5 * np.diff(z)
        mass = f * widths
        total = float(mass.sum())
        if total <= 0.0:
            raise ValueError("dP_dz integrates to zero; nothing to normalise")
        return cls(
            z_Gy=tuple(float(v) for v in z),
            probability=tuple(float(v) for v in mass / total),
            mean_hit_z_Gy=float(mean_hit_z_Gy),
        )

    @classmethod
    def log_normal(
        cls,
        median_z_Gy: float,
        sigma_ln: float,
        *,
        n_points: int = 900,
        n_sigma: float = 4.0,
    ) -> LocalTargetSpectrum:
        """A log-normal ``dP/dz`` of median ``median_z_Gy`` and width ``sigma_ln`` (in ``ln z``).

        **An illustration, not NanOx physics**: the real spectrum is the LQD
        Monte-Carlo output this package does not have (see the class docstring).
        It is useful for showing how strongly eq. (26) depends on the *width* of
        the spectrum at a fixed median.

        The grid is deterministic: ``n_points`` log-spaced nodes spanning
        ``log10(median) +- n_sigma * sigma_ln / 2.303`` decades, i.e. about
        ``+-n_sigma`` standard deviations in ``ln z`` (the divisor 2.303 is part of
        the grid definition, not an exact ``ln 10``). The density
        ``exp(-ln(z/median)^2 / (2 sigma_ln^2)) / (z sigma_ln sqrt(2 pi))`` goes
        through [`from_density`][], and ``mean_hit_z_Gy`` is the trapezoid mean
        ``int z p dz / int p dz`` on the same grid.
        """
        if median_z_Gy <= 0.0:
            raise ValueError(f"median_z_Gy must be > 0, got {median_z_Gy!r}")
        if sigma_ln <= 0.0:
            raise ValueError(f"sigma_ln must be > 0, got {sigma_ln!r}")
        if n_points < 2:
            raise ValueError(f"n_points must be >= 2, got {n_points!r}")
        median = float(median_z_Gy)
        w = float(sigma_ln)
        half_width_decades = n_sigma * w / 2.303
        grid = np.logspace(
            np.log10(median) - half_width_decades, np.log10(median) + half_width_decades, n_points
        )
        norm = grid * w * np.sqrt(2 * np.pi)
        dens = np.exp(-((np.log(grid / median)) ** 2) / (2 * w**2)) / norm
        mean_hit = float(np.trapezoid(grid * dens, grid) / np.trapezoid(dens, grid))
        return cls.from_density(grid, dens, mean_hit_z_Gy=mean_hit)


@dataclass(frozen=True, slots=True)
class EffectiveLocalLethalFunction:
    """The ELLF ``F(z)``: Monini et al. 2020 eq. (28).

    ```text
    F(z) = (h/2) [ 1 + erf( (z - z0) / sigma ) ]
    ```

    ``F`` is **not** a probability. It is ``-N ln(1 - f(z))`` (2022 eq. 7), the
    effective number of local lethal events summed over the ``N`` targets of a
    sensitive volume, so it saturates at ``h`` -- of order 1e5 for real cell
    lines -- and not at 1. There is no normalisation and no truncation in the
    published form.

    The three parameters carry the two features Monini et al. 2020 found by
    fitting ``F`` with no prior shape assumption (a linear combination of ramp
    basis functions, their eqs. 21-22): a **threshold** ``z0`` and a
    **saturation** ``h``, with ``sigma`` setting how sharply it rises. The
    threshold is what produces the overkill turnover in ``alpha(LET)``.
    """

    h: float
    """Height of the response (saturation value of F). Dimensionless."""
    z0_Gy: float
    """Restricted specific-energy threshold [Gy]. ``F(z0) = h/2``."""
    sigma_Gy: float
    """Extent of the increase [Gy]."""

    def __post_init__(self) -> None:
        if self.h < 0.0:
            raise ValueError(f"h must be >= 0, got {self.h!r}")
        if self.z0_Gy < 0.0:
            raise ValueError(f"z0_Gy must be >= 0, got {self.z0_Gy!r}")
        if self.sigma_Gy <= 0.0:
            raise ValueError(f"sigma_Gy must be > 0, got {self.sigma_Gy!r}")

    def __call__(self, z_Gy: float) -> float:
        """Eq. (28), for one value of the restricted specific energy."""
        return 0.5 * self.h * (1.0 + float(erf((z_Gy - self.z0_Gy) / self.sigma_Gy)))

    def values(self, z_Gy: NDArray[np.float64]) -> NDArray[np.float64]:
        """Eq. (28), vectorised over a grid."""
        z = np.asarray(z_Gy, dtype=float)
        result: NDArray[np.float64] = 0.5 * self.h * (1.0 + erf((z - self.z0_Gy) / self.sigma_Gy))
        return result
