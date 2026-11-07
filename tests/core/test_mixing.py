"""Mixing and the critical intra-bin covariance test."""

from __future__ import annotations

import math
import tracemalloc

import numpy as np
import pytest
from scipy.integrate import simpson

from pyradbio.core import mixing
from pyradbio.core.radiation import Ion, RadiationField, SpectrumComponent, fluence_to_dose


def test_dose_average_identity_single_component() -> None:
    assert mixing.dose_average([0.7], [1.0]) == pytest.approx(0.7)


def test_dose_average_sqrt_convention() -> None:
    # Two equal-dose components -> ( (sqrt b1 + sqrt b2)/2 )^2
    b1, b2 = 0.04, 0.09
    expected = ((math.sqrt(b1) + math.sqrt(b2)) / 2.0) ** 2
    assert mixing.dose_average_sqrt([b1, b2], [1.0, 1.0]) == pytest.approx(expected)


@pytest.mark.golden
def test_subgrid_captures_stopping_power_covariance() -> None:
    """The single most important numerics test (see AGENTS.md).

    Over a wide energy bin where stopping power S(E) and the looked-up quantity
    q(E) both vary steeply, integrating the PRODUCT on a ~16-node log sub-grid
    must match a dense reference to <0.5%, whereas a bin-centre evaluation does
    not. We assert the sub-grid result here; a companion check should confirm
    the bin-centre approximation deviates substantially.
    """

    # Steeply varying, positively-correlated S and q on [1, 300] MeV/u.
    def S(E: float) -> float:
        return 1.0 / E  # ~ stopping power falling with energy

    def q(E: float) -> float:
        return 1.0 / E  # correlated quantity

    lo, hi = 1.0, 300.0

    # Dense trapezoidal reference of S(E)*q(E) = 1/E^2 -> integral = 1/lo - 1/hi
    reference = (1.0 / lo) - (1.0 / hi)

    sub = mixing.spectral_integral(q, S, lo, hi, sub_nodes=16, spacing="log")
    assert sub == pytest.approx(reference, rel=5e-3)


def test_bin_centre_approximation_deviates_substantially() -> None:
    """Companion to the covariance test: the naive bin-centre rule is wrong.

    Evaluating S(E)*q(E) at the (geometric) bin centre and multiplying by the
    bin width ignores the intra-bin covariance and is off by a large, documented
    amount here (>30%), whereas the sub-grid integrator is within 0.5%.
    """

    def S(E: float) -> float:
        return 1.0 / E

    def q(E: float) -> float:
        return 1.0 / E

    lo, hi = 1.0, 300.0
    reference = (1.0 / lo) - (1.0 / hi)

    centre = 0.5 * (lo + hi)  # arithmetic bin centre (the naive rule)
    bin_centre_value = S(centre) * q(centre) * (hi - lo)
    rel_error = abs(bin_centre_value - reference) / reference
    assert rel_error > 0.30  # naive rule is badly wrong

    sub = mixing.spectral_integral(q, S, lo, hi, sub_nodes=16, spacing="log")
    assert abs(sub - reference) / reference < 5e-3


def test_linear_spacing_integrates() -> None:
    """Linear sub-grid path integrates a smooth product correctly."""

    def S(E: float) -> float:
        return 2.0

    def q(E: float) -> float:
        return 3.0

    # constant product 6 over [0, 4] -> 24
    result = mixing.spectral_integral(q, S, 1.0, 5.0, sub_nodes=8, spacing="linear")
    assert result == pytest.approx(6.0 * 4.0)


def test_spectral_integral_uses_binned_fluence_field() -> None:
    """Binned spectra integrate phi(E)*S(E)*q(E), not just centre values."""

    def stopping_power(E: float) -> float:
        return 1.0 / E

    def quantity(E: float) -> float:
        return 1.0 / E

    lo, hi = 1.0, 300.0
    total_fluence = 2.0e8
    ion = Ion(Z=6, A=12, energy_MeV_u=(lo * hi) ** 0.5)
    field = RadiationField(
        [SpectrumComponent(ion, fluence=total_fluence, energy_lo_MeV_u=lo, energy_hi_MeV_u=hi)]
    )

    # Default bin model is uniform differential fluence over the support.
    phi = total_fluence / (hi - lo)
    expected = fluence_to_dose(phi * ((1.0 / lo) - (1.0 / hi)), let_keV_um=1.0)
    got = mixing.spectral_integral(quantity, field, stopping_power_fn=stopping_power)
    assert got == pytest.approx(expected, rel=5e-3)


def test_spectral_dose_average_divides_by_binned_dose_weight() -> None:
    """The high-level spectral average is the product integral over dose integral."""

    def stopping_power(E: float) -> float:
        return 1.0 / E

    def quantity(E: float) -> float:
        return 1.0 / E

    lo, hi = 1.0, 300.0
    ion = Ion(Z=6, A=12, energy_MeV_u=(lo * hi) ** 0.5)
    field = RadiationField(
        [SpectrumComponent(ion, fluence=1.0e8, energy_lo_MeV_u=lo, energy_hi_MeV_u=hi)]
    )

    numerator = (1.0 / lo) - (1.0 / hi)
    denominator = math.log(hi / lo)
    expected = numerator / denominator
    assert mixing.spectral_dose_average(quantity, field, stopping_power_fn=stopping_power) == (
        pytest.approx(expected, rel=5e-3)
    )


def test_dose_average_input_validation() -> None:
    with pytest.raises(ValueError):
        mixing.dose_average([1.0, 2.0], [1.0])
    with pytest.raises(ValueError):
        mixing.dose_average([], [])
    with pytest.raises(ValueError):
        mixing.dose_average([1.0], [0.0])
    with pytest.raises(ValueError):
        mixing.dose_average_sqrt([-1.0], [1.0])
    with pytest.raises(ValueError):
        mixing.dose_average_sqrt([1.0, 2.0], [1.0])
    with pytest.raises(ValueError):
        mixing.dose_average_sqrt([], [])


def test_spectral_integral_validation() -> None:
    f = lambda E: 1.0  # noqa: E731
    with pytest.raises(ValueError):
        mixing.spectral_integral(f, f, -1.0, 10.0)
    with pytest.raises(ValueError):
        mixing.spectral_integral(f, f, 10.0, 1.0)
    with pytest.raises(ValueError):
        mixing.spectral_integral(f, f, 1.0, 10.0, sub_nodes=1)
    with pytest.raises(ValueError):
        mixing.spectral_integral(f, f, 1.0, 10.0, spacing="quadratic")  # type: ignore[arg-type]
    with pytest.raises(ValueError):
        mixing.spectral_integral(
            f, RadiationField([SpectrumComponent(Ion(1, 1, 1.0), fluence=1.0)])
        )


# --- dose_quadrature: the (ion, dose-weight) nodes every model mixes over -------


def _let_one_over_e(ion: Ion) -> float:
    return 100.0 / ion.energy_MeV_u


def test_dose_quadrature_mono_component_is_one_node_at_its_absorbed_dose() -> None:
    ion = Ion(6, 12, 10.0)
    field = RadiationField(
        [SpectrumComponent(ion, dose_Gy=2.0), SpectrumComponent(Ion(1, 1, 5.0), fluence=1.0e8)]
    )
    nodes = mixing.dose_quadrature(field, _let_one_over_e)
    assert [n[0] for n in nodes] == [ion, Ion(1, 1, 5.0)]
    assert [n[1] for n in nodes] == field.dose_weights(_let_one_over_e)


def test_dose_quadrature_binned_component_matches_spectral_integral() -> None:
    """sum_j w_j q(E_j) is exactly the sub-grid product integral of the bin."""
    lo, hi = 1.0, 300.0
    comp = SpectrumComponent(
        Ion(6, 12, (lo * hi) ** 0.5), fluence=1.0e8, energy_lo_MeV_u=lo, energy_hi_MeV_u=hi
    )
    field = RadiationField([comp])
    nodes = mixing.dose_quadrature(field, _let_one_over_e, sub_nodes=16)
    assert len(nodes) == 16
    assert nodes[0][0] == Ion(6, 12, lo)
    assert nodes[-1][0] == Ion(6, 12, hi)  # bin edges are hit exactly
    assert all(w > 0.0 for _, w in nodes)

    def q(e: float) -> float:
        return e**-0.5

    got = math.fsum(w * q(ion.energy_MeV_u) for ion, w in nodes)
    expected = mixing.spectral_integral(q, field, stopping_power_fn=lambda e: 100.0 / e)
    assert got == pytest.approx(expected, rel=1e-12)
    total = math.fsum(w for _, w in nodes)
    assert total == pytest.approx(fluence_to_dose(1.0e8 / (hi - lo), 100.0) * math.log(hi / lo))


def test_dose_quadrature_dose_specified_bin_sums_to_its_dose() -> None:
    comp = SpectrumComponent(Ion(6, 12, 2.0), dose_Gy=3.0, energy_lo_MeV_u=1.0, energy_hi_MeV_u=4.0)
    nodes = mixing.dose_quadrature(RadiationField([comp]), _let_one_over_e, sub_nodes=9)
    assert math.fsum(w for _, w in nodes) == pytest.approx(3.0, rel=1e-12)


@pytest.mark.parametrize("spacing", ["log", "linear"])
@pytest.mark.parametrize("sub_nodes", [2, 3, 15])
def test_quadrature_matches_sampled_simpson(spacing: mixing.Spacing, sub_nodes: int) -> None:
    """Match sampled Simpson integration and its two-node trapezoid fallback."""
    component = SpectrumComponent(
        Ion(6, 12, 10.0), fluence=1.0e8, energy_lo_MeV_u=1.0, energy_hi_MeV_u=30.0
    )
    nodes = mixing.dose_quadrature(
        RadiationField([component]), _let_one_over_e, sub_nodes=sub_nodes, spacing=spacing
    )
    energies = np.array([ion.energy_MeV_u for ion, _ in nodes])
    quantity = np.sin(energies) + 2.0
    integrand = fluence_to_dose(1.0e8 / 29.0, 100.0 / energies) * quantity
    if spacing == "log":
        expected = simpson(integrand * energies, x=np.log(energies))
    else:
        expected = simpson(integrand, x=energies)
    actual = math.fsum(dose * q for (_, dose), q in zip(nodes, quantity, strict=True))
    assert actual == pytest.approx(expected, rel=1e-12)


@pytest.mark.parametrize("spacing", ["log", "linear"])
@pytest.mark.parametrize("sub_nodes", [4, 16])
def test_even_quadrature_integrates_quadratic_exactly(
    spacing: mixing.Spacing, sub_nodes: int
) -> None:
    """The last-interval correction integrates quadratics, including on SciPy 1.10.

    That version's default even-grid rule differs, so use an analytic oracle.
    For log spacing the Jacobian turns q(E) = log(E)^2/E into u^2.
    """
    if spacing == "log":
        quantity = lambda e: math.log(e) ** 2 / e  # noqa: E731
        expected = math.log(30.0) ** 3 / 3.0
    else:
        quantity = lambda e: e**2  # noqa: E731
        expected = (30.0**3 - 1.0) / 3.0
    actual = mixing.spectral_integral(
        quantity, lambda e: 1.0, 1.0, 30.0, sub_nodes=sub_nodes, spacing=spacing
    )
    assert actual == pytest.approx(expected, rel=1e-12)


def test_dense_spectral_integral_has_bounded_memory() -> None:
    """Refining a convergence grid must not allocate a quadratic-size matrix."""
    tracemalloc.start()
    try:
        result = mixing.spectral_integral(
            lambda e: 1.0 / e, lambda e: 1.0 / e, 1.0, 300.0, sub_nodes=2049
        )
        _, peak_bytes = tracemalloc.get_traced_memory()
    finally:
        tracemalloc.stop()
    assert result == pytest.approx(1.0 - 1.0 / 300.0, rel=1e-10)
    # A linear working set needs well under 1 MiB; allow ample platform headroom.
    assert peak_bytes < 16 * 1024**2
