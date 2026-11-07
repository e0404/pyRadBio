"""Mixed-field LQ mixing and spectral integration.

CRITICAL NUMERICS (see AGENTS.md). When integrating a model quantity over a
fluence spectrum, integrate the *product* stopping_power(E)*quantity(E) on a
fine logarithmic sub-grid per energy bin (~16 nodes/bin), never at bin centres:
the intra-bin covariance between stopping power and the looked-up quantity is
the numerically significant term.

Models consume this through :func:`dose_quadrature`, which expands every
component of a :class:`~pyradbio.core.radiation.RadiationField` into
``(ion, dose weight)`` nodes; :class:`~pyradbio.models.base.BiophysicalModel`
then dose-averages its per-family mixing quantity over those nodes.
"""

from __future__ import annotations

import dataclasses
import math
from collections.abc import Callable, Sequence
from typing import Literal, overload

import numpy as np

from pyradbio.core.radiation import Ion, RadiationField, SpectrumComponent, fluence_to_dose

#: Sub-grid nodes per energy bin (AGENTS.md: "~16 nodes/bin", log spacing).
DEFAULT_SUB_NODES = 16

Spacing = Literal["log", "linear"]


def dose_average(values: Sequence[float], doses: Sequence[float]) -> float:
    """Dose-averaged scalar: ``sum(d_i v_i)/sum(d_i)``. Used for alpha and z*.

    The dose-mean of a per-component quantity (Zaider-Rossi/Kanai convention,
    Kase et al. 2006). ``alpha_bar = sum d_i alpha_i / sum d_i``.
    """
    if len(values) != len(doses):
        raise ValueError("values and doses must have equal length")
    if not values:
        raise ValueError("need at least one component")
    total = math.fsum(doses)
    if total <= 0.0:
        raise ValueError("total dose must be positive")
    return math.fsum(d * v for d, v in zip(doses, values, strict=True)) / total


def dose_average_sqrt(betas: Sequence[float], doses: Sequence[float]) -> float:
    """Combine beta via dose-averaged sqrt: ``(sum d_i sqrt(b_i)/sum d_i)**2``.

    Zaider-Rossi / Kanai convention: it is ``sqrt(beta)`` that is dose-averaged,
    so ``sqrt(beta_bar) = sum d_i sqrt(b_i) / sum d_i`` (see AGENTS.md numerics).
    """
    if len(betas) != len(doses):
        raise ValueError("betas and doses must have equal length")
    if not betas:
        raise ValueError("need at least one component")
    if any(b < 0.0 for b in betas):
        raise ValueError("beta values must be >= 0")
    root = dose_average([math.sqrt(b) for b in betas], doses)
    return root * root


def _sub_grid(energy_lo: float, energy_hi: float, sub_nodes: int, spacing: Spacing) -> list[float]:
    """Return ``sub_nodes`` ordered nodes spanning [energy_lo, energy_hi]."""
    if energy_lo <= 0.0 or energy_hi <= 0.0:
        raise ValueError("energies must be positive")
    if energy_hi <= energy_lo:
        raise ValueError("energy_hi must exceed energy_lo")
    if sub_nodes < 2:
        raise ValueError("need at least 2 sub-grid nodes")
    n = sub_nodes
    if spacing == "log":
        ratio = energy_hi / energy_lo
        nodes = [energy_lo * ratio ** (i / (n - 1)) for i in range(n)]
    elif spacing == "linear":
        step = (energy_hi - energy_lo) / (n - 1)
        nodes = [energy_lo + i * step for i in range(n)]
    else:
        raise ValueError(f"unknown spacing {spacing!r}")
    # Hit the bin edges exactly, so adjacent bins share their edge node bit-for-bit.
    nodes[0], nodes[-1] = energy_lo, energy_hi
    return nodes


def _quadrature_weights(nodes: Sequence[float], spacing: Spacing) -> list[float]:
    """Composite-Simpson weights ``w_j`` with ``int f(E) dE ~ sum_j w_j f(E_j)``.

    Integrate the three Lagrange basis polynomials on each pair of intervals,
    accumulating their weights in linear memory. For an even number of nodes,
    integrate the final interval using the last three nodes (the Cartwright
    correction used by SciPy >= 1.11). Two nodes use the trapezoid rule.
    On a log grid the rule runs in ``u = ln E`` and the Jacobian ``dE = E du``
    is folded into the weight.
    """
    if spacing == "log":
        x = np.log(np.asarray(nodes, dtype=float))
        jacobian = np.asarray(nodes, dtype=float)
    else:
        x = np.asarray(nodes, dtype=float)
        jacobian = np.ones(len(nodes))
    unit = np.zeros(len(nodes))
    steps = np.diff(x)
    if len(nodes) == 2:
        unit[:] = steps[0] / 2.0
    else:
        h0, h1 = steps[:-1:2], steps[1::2]
        span = h0 + h1
        unit[:-2:2] += span * (2.0 - h1 / h0) / 6.0
        unit[1:-1:2] += span**3 / (6.0 * h0 * h1)
        unit[2::2] += span * (2.0 - h0 / h1) / 6.0
        if len(nodes) % 2 == 0:
            previous, last = steps[-2:]
            span_last = previous + last
            unit[-3] -= last**3 / (6.0 * previous * span_last)
            unit[-2] += last * (last + 3.0 * previous) / (6.0 * previous)
            unit[-1] += last * (2.0 * last + 3.0 * previous) / (6.0 * span_last)
    return [float(w) for w in unit * jacobian]


def _integrate_product(
    product_fn: Callable[[float], float],
    energy_lo: float,
    energy_hi: float,
    *,
    sub_nodes: int = DEFAULT_SUB_NODES,
    spacing: Spacing = "log",
) -> float:
    """Integrate one already-formed product over an energy bin."""
    nodes = _sub_grid(energy_lo, energy_hi, sub_nodes, spacing)
    weights = _quadrature_weights(nodes, spacing)
    return math.fsum(w * product_fn(e) for w, e in zip(weights, nodes, strict=True))


def _default_fluence_density(component: SpectrumComponent, energy_MeV_u: float) -> float:
    """Uniform differential fluence over a finite bin.

    Only called for binned, fluence-specified components (see
    ``_binned_component_integral``), so both ``fluence`` and the bin bounds are
    guaranteed present.
    """
    assert component.fluence is not None
    bounds = component.energy_bounds_MeV_u()
    assert bounds is not None
    lo, hi = bounds
    return component.fluence / (hi - lo)


def _mono_component_integral(
    quantity_fn: Callable[[float], float],
    stopping_power_fn: Callable[[float], float],
    component: SpectrumComponent,
) -> float:
    energy = component.ion.energy_MeV_u
    quantity = quantity_fn(energy)
    if component.dose_Gy is not None:
        return component.dose_Gy * quantity
    assert component.fluence is not None
    dose = fluence_to_dose(component.fluence, stopping_power_fn(energy))
    return dose * quantity


def _binned_component_integral(
    quantity_fn: Callable[[float], float],
    stopping_power_fn: Callable[[float], float],
    component: SpectrumComponent,
    fluence_density_fn: Callable[[SpectrumComponent, float], float] | None,
    *,
    sub_nodes: int,
    spacing: Spacing,
) -> float:
    bounds = component.energy_bounds_MeV_u()
    if bounds is None:
        return _mono_component_integral(quantity_fn, stopping_power_fn, component)
    lo, hi = bounds
    if component.dose_Gy is not None:
        dose_density = component.dose_Gy / (hi - lo)

        def product_from_dose(energy: float) -> float:
            return dose_density * quantity_fn(energy)

        return _integrate_product(product_from_dose, lo, hi, sub_nodes=sub_nodes, spacing=spacing)

    density = fluence_density_fn or _default_fluence_density

    def product_from_fluence(energy: float) -> float:
        fluence_density = density(component, energy)
        dose_density = fluence_to_dose(fluence_density, stopping_power_fn(energy))
        return dose_density * quantity_fn(energy)

    return _integrate_product(product_from_fluence, lo, hi, sub_nodes=sub_nodes, spacing=spacing)


@overload
def spectral_integral(
    quantity_fn: Callable[[float], float],
    stopping_power_fn: Callable[[float], float],
    energy_lo: float,
    energy_hi: float,
    /,
    *,
    sub_nodes: int = DEFAULT_SUB_NODES,
    spacing: Spacing = "log",
) -> float: ...


@overload
def spectral_integral(
    quantity_fn: Callable[[float], float],
    spectrum: RadiationField,
    /,
    *,
    stopping_power_fn: Callable[[float], float],
    fluence_density_fn: Callable[[SpectrumComponent, float], float] | None = None,
    sub_nodes: int = DEFAULT_SUB_NODES,
    spacing: Spacing = "log",
) -> float: ...


def spectral_integral(
    quantity_fn: Callable[[float], float],
    spectrum_or_stopping_power_fn: RadiationField | Callable[[float], float],
    energy_lo: float | None = None,
    energy_hi: float | None = None,
    *,
    stopping_power_fn: Callable[[float], float] | None = None,
    fluence_density_fn: Callable[[SpectrumComponent, float], float] | None = None,
    sub_nodes: int = DEFAULT_SUB_NODES,
    spacing: Spacing = "log",
) -> float:
    """Integrate a quantity over either one bin or a full radiation spectrum.

    Low-level compatibility form ``spectral_integral(q, S, E_lo, E_hi)`` returns
    ``integral S(E)*q(E) dE`` over one bin. High-level spectrum form
    ``spectral_integral(q, field, stopping_power_fn=S)`` sums
    ``dose_density(E)*q(E)`` over all components. Fluence-specified bins use
    ``dose_density = phi(E)*S(E)`` converted to Gy; dose-specified bins use
    uniform dose density over the bin.
    """
    if isinstance(spectrum_or_stopping_power_fn, RadiationField):
        if energy_lo is not None or energy_hi is not None:
            raise ValueError("energy bounds are read from the RadiationField components")
        if stopping_power_fn is None:
            raise ValueError("stopping_power_fn is required for RadiationField integration")
        return math.fsum(
            _binned_component_integral(
                quantity_fn,
                stopping_power_fn,
                component,
                fluence_density_fn,
                sub_nodes=sub_nodes,
                spacing=spacing,
            )
            for component in spectrum_or_stopping_power_fn.components
        )

    if energy_lo is None or energy_hi is None:
        raise ValueError("energy_lo and energy_hi are required for one-bin integration")
    raw_stopping_power_fn = spectrum_or_stopping_power_fn
    return _integrate_product(
        lambda energy: raw_stopping_power_fn(energy) * quantity_fn(energy),
        energy_lo,
        energy_hi,
        sub_nodes=sub_nodes,
        spacing=spacing,
    )


def spectral_dose_average(
    quantity_fn: Callable[[float], float],
    spectrum: RadiationField,
    *,
    stopping_power_fn: Callable[[float], float],
    fluence_density_fn: Callable[[SpectrumComponent, float], float] | None = None,
    sub_nodes: int = DEFAULT_SUB_NODES,
    spacing: Spacing = "log",
) -> float:
    """Dose-average ``quantity_fn(E)`` over a binned radiation field."""
    numerator = spectral_integral(
        quantity_fn,
        spectrum,
        stopping_power_fn=stopping_power_fn,
        fluence_density_fn=fluence_density_fn,
        sub_nodes=sub_nodes,
        spacing=spacing,
    )
    denominator = spectral_integral(
        lambda _energy: 1.0,
        spectrum,
        stopping_power_fn=stopping_power_fn,
        fluence_density_fn=fluence_density_fn,
        sub_nodes=sub_nodes,
        spacing=spacing,
    )
    if denominator <= 0.0:
        raise ValueError("total spectral dose must be positive")
    return numerator / denominator


def dose_quadrature(
    field: RadiationField,
    stopping_power: Callable[[Ion], float],
    *,
    sub_nodes: int = DEFAULT_SUB_NODES,
    spacing: Spacing = "log",
    fluence_density_fn: Callable[[SpectrumComponent, float], float] | None = None,
) -> list[tuple[Ion, float]]:
    """Expand a field into ``(ion, dose weight [Gy])`` quadrature nodes.

    The one place a mixed field is discretised for the models. A mono-energetic
    component is a single node at its own ion with its absorbed dose (exactly
    :meth:`RadiationField.dose_weights`). A binned component becomes
    ``sub_nodes`` ions on a sub-grid spanning its bin, edges included, each
    weighted by ``w_j dD/dE(E_j)`` with ``w_j`` the composite-Simpson weight, so
    ``sum_j D_j q(E_j)`` is the bin's product integral ``int dD/dE q(E) dE`` --
    the same integral :func:`spectral_integral` evaluates -- for *any* per-ion
    quantity ``q``. Dose-averaging a quantity over these nodes is therefore the
    sub-grid integration AGENTS.md requires, never a bin-centre evaluation.
    A dose-specified bin (uniform dose density) is normalised so its node
    weights sum to its ``dose_Gy`` exactly.

    ``stopping_power`` [keV/um] is only consulted for fluence-specified
    components. Its absolute scale cancels only when all components are
    fluence-specified; mixing fluence and dose requires physical Gy weights.
    Use one consistent stopping-power source across components of different Z.
    """
    nodes: list[tuple[Ion, float]] = []
    for component in field.components:
        bounds = component.energy_bounds_MeV_u()
        if bounds is None:
            nodes.append((component.ion, component.absorbed_dose_Gy(stopping_power)))
            continue
        lo, hi = bounds
        energies = _sub_grid(lo, hi, sub_nodes, spacing)
        weights = _quadrature_weights(energies, spacing)
        if component.dose_Gy is not None:
            # Uniform dose density: the bin's total is known exactly, so share it
            # out in proportion to the weights instead of carrying the quadrature
            # error of integrating a constant on a log grid into the mix.
            total = math.fsum(weights)
            nodes.extend(
                (
                    dataclasses.replace(component.ion, energy_MeV_u=energy),
                    component.dose_Gy * w / total,
                )
                for energy, w in zip(energies, weights, strict=True)
            )
            continue
        density = fluence_density_fn or _default_fluence_density
        for energy, weight in zip(energies, weights, strict=True):
            ion = dataclasses.replace(component.ion, energy_MeV_u=energy)
            dose_density = fluence_to_dose(density(component, energy), stopping_power(ion))
            nodes.append((ion, weight * dose_density))
    return nodes
