"""Multi-event specific-energy densities ``f(z; D)`` for a microscopic site.

Sato & Furusawa 2012, *Radiat. Res.* 178:341-356, doi:10.1667/RR2842.1, eqs.
(24)-(28). The single-event density ``f_1(z)`` says how much specific energy one
traversal deposits in a site; the **multi-event** density ``f(z; D)`` says how
that accumulates at absorbed dose ``D``, when the number of traversals is Poisson
distributed. It is the quantity the stochastic MK models integrate survival
against:

```text
S(D) = integral S_n(z_n) f_n(z_n; D) dz_n            (the paper's eq. 7)
```

Model-agnostic and site-agnostic: nothing here knows about domains, nuclei or
cell survival. [`pyradbio.models.mkm.spectrum_models`][] supplies the biology.

Compound Poisson, in one transform
----------------------------------
The paper builds ``f_k`` by iterating the convolution ``f_k = f_1 * f_{k-1}``
(its eq. 26) and then Poisson-averages over ``k`` (eq. 27), which it reports
needing "a little more than 100" iterations, and it switches to a precomputed
database once ``lambda > 50`` because the iteration count becomes impractical.

That is unnecessary. The Poisson-weighted sum of repeated convolutions is a
**compound Poisson** distribution, whose transform is closed form:

```text
sum_k  P(lambda; k) F_1(w)^k  =  exp(lambda (F_1(w) - 1))
```

so the whole sum costs one forward and one inverse FFT **at any ``lambda``** --
no iteration to truncate, no separate high-dose branch, and the ``k = 0`` term
(the ``exp(-lambda)`` of nuclei missed entirely) falls out on its own rather than
being special-cased. [`multi_event_density`][] does that.

Densities are carried as **probability masses on a uniform grid**, not as
densities, because that is what makes the convolution a plain product and keeps
the point mass at ``z = 0`` exact.

Aliasing is the one thing to respect: a circular convolution wraps, so the grid
must extend past where ``f(z; D)`` has any mass. [`multi_event_density`][]
therefore **grows** the grid to ``D + 12 sqrt(D z_D)``, sized from the
compound-Poisson mean and variance -- both closed form -- rather than guessed,
and rounds up to a power of two so the transform stays cheap.

References
----------
- Sato, T. & Furusawa, Y. (2012) doi:10.1667/RR2842.1 -- eqs. (24)-(28).
- Olko, P. & Booz, J. (1990) *Radiat. Environ. Biophys.* 28:1 -- the Fermi form
  of the single-event density the paper cites for eq. (24).
"""

from __future__ import annotations

import math

import numpy as np
import numpy.typing as npt

_JOULE_PER_KEV: float = 1.602176634e-16
"""Joules per keV."""

_WATER_DENSITY_KG_M3: float = 1000.0
"""Liquid-water density [kg/m^3], the site material assumed throughout."""

DEFAULT_FERMI_WIDTH_FRACTION: float = 0.1
"""Sato & Furusawa 2012's own choice for the Fermi width: ``c = L*eta/10``. They
report the straggling it represents is "not so large ... and insignificant in
the final results of the calculation", having checked it against a PHITS
simulation carrying delta rays above 1 keV.

**Measured here, that wording undersells it.** At LET 200 keV/um in a 6.2 um
sphere this width puts **15% of the single-event mass above ``z_max``** -- the
specific energy of a *central* traversal, the most a chord can geometrically
deliver -- because the Fermi form's tail is an unbounded exponential. It also
lifts the frequency mean from ``0.667`` to ``0.709`` of ``z_max`` (+6.4%) and
the dose mean from ``0.750`` to ``0.820`` (+9.4%). The paper's claim is about
the final *survival*, not about the moments, and both readings can be true at
once. Since ``z_D`` sets the multi-event variance, and the variance is the
whole stochastic correction, this is the first thing to vary if the numerical
route ever disagrees with the paper. Pass ``0.0`` for the sharp geometric
limit, where the means are exactly ``2/3`` and ``3/4`` of ``z_max``.
"""

_GRID_SIGMA_HEADROOM: float = 12.0
"""How many standard deviations of headroom the multi-event grid carries beyond
the mean before the circular convolution could alias into itself. 12 sigma
puts the wrapped mass below 1e-30 for a Gaussian-like tail; the compound
Poisson is more skewed than Gaussian at small ``lambda``, which is exactly
where the distribution is instead compactly supported, so this is ample.
"""


def let_to_central_specific_energy_Gy(let_keV_um: float, *, radius_um: float) -> float:
    """``eta``: specific energy [Gy] from a **central** traversal, per LET.

    Sato & Furusawa 2012 eq. (25). A central chord of a sphere of radius ``r`` is
    ``2r`` long, so the deposited energy is ``L * 2r`` and the specific energy is
    that over the sphere's mass -- giving ``eta ~ L / r^2``.

    Parameters
    ----------
    let_keV_um
        Unrestricted LET [keV/um], ``>= 0``.
    radius_um
        Site radius [um], ``> 0``.
    """
    if not math.isfinite(let_keV_um) or let_keV_um < 0.0:
        raise ValueError(f"let_keV_um must be finite and >= 0, got {let_keV_um!r}")
    if not math.isfinite(radius_um) or radius_um <= 0.0:
        raise ValueError(f"radius_um must be finite and > 0, got {radius_um!r}")
    energy_J = let_keV_um * 2.0 * radius_um * _JOULE_PER_KEV
    mass_kg = (4.0 / 3.0) * math.pi * (radius_um * 1e-6) ** 3 * _WATER_DENSITY_KG_M3
    return energy_J / mass_kg


def single_event_density(
    *,
    let_keV_um: float,
    radius_um: float,
    fermi_width_fraction: float = DEFAULT_FERMI_WIDTH_FRACTION,
    nodes: int = 4096,
) -> tuple[npt.NDArray[np.float64], npt.NDArray[np.float64]]:
    """Single-event specific-energy density for one traversal of a sphere.

    Sato & Furusawa 2012 eq. (24): "the Fermi function of ``z_n`` multiplied by
    ``z_n``":

    ```text
    f_1(z) ~ z / (exp[(z - L eta)/c] + 1)
    ```

    The ``z`` prefactor is not a fudge -- a sphere's chord-length distribution
    under uniform-isotropic incidence is ``f(l) ~ l``, and ``z`` is proportional
    to chord length, so ``f(z) ~ z`` on ``[0, L eta]`` is the exact geometric
    result. The Fermi denominator rounds the edge at ``z = L eta`` to represent
    energy-loss straggling; with ``c -> 0`` it becomes the sharp
    ``f(z) = 2z / (L eta)^2``, whose frequency and dose means are exactly
    ``(2/3) L eta`` and ``(3/4) L eta``.

    Parameters
    ----------
    let_keV_um
        LET of the traversing particle [keV/um].
    radius_um
        Site radius [um].
    fermi_width_fraction
        ``c = fermi_width_fraction * L eta``. The paper uses ``1/10``; pass
        ``0.0`` for the sharp geometric limit.
    nodes
        Grid points. The grid is uniform in ``z`` (required by
        [`multi_event_density`][]) and spans zero to a few ``c`` beyond
        ``L eta``.

    Returns
    -------
    (z_Gy, probability_mass)
        A uniform ``z`` grid starting at 0 and the probability **mass** in each
        cell, summing to 1.
    """
    if not math.isfinite(fermi_width_fraction) or fermi_width_fraction < 0.0:
        raise ValueError(
            f"fermi_width_fraction must be finite and >= 0, got {fermi_width_fraction!r}"
        )
    if nodes < 16:
        raise ValueError(f"nodes must be >= 16, got {nodes!r}")

    z_max = let_to_central_specific_energy_Gy(let_keV_um, radius_um=radius_um)
    if z_max <= 0.0:
        raise ValueError("a zero-LET traversal deposits nothing; f_1 is undefined")

    width = fermi_width_fraction * z_max
    # Enough headroom for the Fermi tail: it decays like exp(-(z - z_max)/c).
    upper = z_max + (10.0 * width if width > 0.0 else z_max / nodes)
    z = np.linspace(0.0, upper, nodes)

    if width > 0.0:
        density = z / (np.exp((z - z_max) / width) + 1.0)
    else:
        density = np.where(z <= z_max, z, 0.0)

    total = float(density.sum())
    if total <= 0.0:
        raise ValueError("the single-event density integrated to zero")
    mass: npt.NDArray[np.float64] = np.asarray(density / total, dtype=np.float64)
    return np.asarray(z, dtype=np.float64), mass


def multi_event_density(
    z_Gy: npt.NDArray[np.float64],
    single_event_mass: npt.NDArray[np.float64],
    *,
    dose_Gy: float,
) -> tuple[npt.NDArray[np.float64], npt.NDArray[np.float64]]:
    """Multi-event density ``f(z; D)`` from a single-event density.

    Sato & Furusawa 2012 eqs. (26)-(28), evaluated as a compound Poisson in the
    transform domain (see the module docstring) rather than by iterating the
    convolution:

    ```text
    lambda = D / z_F                                            (eq. 28)
    f(z; D) = IFFT[ exp(lambda (FFT[f_1] - 1)) ]                (eqs. 26-27)
    ```

    The mean of the result is ``D`` exactly and its variance is ``D * z_D``, both
    of which the tests assert rather than assume.

    Parameters
    ----------
    z_Gy
        Uniform grid starting at 0, as returned by [`single_event_density`][].
    single_event_mass
        Probability mass per cell of the single-event density, summing to 1.
    dose_Gy
        Absorbed dose ``D`` [Gy], ``>= 0``. ``0.0`` returns all mass at ``z = 0``.

    Returns
    -------
    (z_Gy, probability_mass)
        The grid is extended as far as the distribution needs; cell 0 carries the
        ``exp(-lambda)`` point mass of sites missed entirely.
    """
    z = np.asarray(z_Gy, dtype=np.float64)
    p1 = np.asarray(single_event_mass, dtype=np.float64)
    if z.ndim != 1 or p1.ndim != 1 or z.size != p1.size:
        raise ValueError("z and the single-event mass must be 1-D arrays of equal length")
    if z.size < 2:
        raise ValueError("at least two grid points are required")
    if not math.isfinite(dose_Gy) or dose_Gy < 0.0:
        raise ValueError(f"dose_Gy must be finite and >= 0, got {dose_Gy!r}")

    step = float(z[1] - z[0])
    if step <= 0.0 or not np.allclose(np.diff(z), step, rtol=1e-9, atol=0.0):
        raise ValueError("z must be a uniform grid (the convolution is a product on one)")
    if z[0] != 0.0:
        raise ValueError("z must start at 0 so the no-hit point mass lands in cell 0")

    z_F = float(np.sum(z * p1))
    if z_F <= 0.0:
        raise ValueError("the single-event density has zero mean; lambda is undefined")

    if dose_Gy == 0.0:
        out = np.zeros_like(p1)
        out[0] = 1.0
        return z, out

    lam = dose_Gy / z_F
    z_D = float(np.sum(z * z * p1) / z_F)
    # Size the grid from the closed-form mean and variance, not by guessing.
    span = dose_Gy + _GRID_SIGMA_HEADROOM * math.sqrt(max(dose_Gy * z_D, 0.0))
    span = max(span, float(z[-1]))
    n = math.ceil(span / step) + 1
    n = 1 << (n - 1).bit_length()  # a power of two keeps the FFT cheap

    padded = np.zeros(n, dtype=np.float64)
    padded[: p1.size] = p1
    transform = np.fft.rfft(padded)
    # exp(lambda (F - 1)): the compound-Poisson characteristic function. The
    # -lambda is real, so this never overflows however large lambda gets.
    result = np.fft.irfft(np.exp(lam * (transform - 1.0)), n=n)
    result = np.clip(result, 0.0, None)
    total = float(result.sum())
    if total <= 0.0:
        raise ValueError("the multi-event density integrated to zero")
    result /= total
    return np.arange(n, dtype=np.float64) * step, result
