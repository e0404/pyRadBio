"""Radiation fields and fluence<->dose conversion."""

from __future__ import annotations

import pytest

from pyradbio.core.radiation import (
    Ion,
    RadiationField,
    SpectrumComponent,
    dose_to_fluence,
    fluence_to_dose,
    relativistic_beta,
)


def _let(ion: Ion) -> float:
    """Toy LET lookup [keV/um] used in tests."""
    return 10.0 * ion.Z


def test_fluence_dose_roundtrip() -> None:
    let = 50.0  # keV/um
    fluence = 1.0e8  # cm^-2
    dose = fluence_to_dose(fluence, let)
    assert dose_to_fluence(dose, let) == pytest.approx(fluence, rel=1e-12)


def test_dose_to_fluence_requires_positive_let() -> None:
    with pytest.raises(ValueError):
        dose_to_fluence(1.0, 0.0)


def test_mono_field_dose() -> None:
    ion = Ion(Z=6, A=12, energy_MeV_u=200.0)
    field = RadiationField.mono(ion, dose_Gy=2.0)
    assert not field.is_mixed
    assert field.total_dose_Gy(_let) == pytest.approx(2.0)
    assert field.dose_weights(_let) == [pytest.approx(2.0)]


def test_fluence_component_converted_to_dose() -> None:
    ion = Ion(Z=2, A=4, energy_MeV_u=100.0)
    fluence = 1.0e9
    comp = SpectrumComponent(ion, fluence=fluence)
    expected = fluence_to_dose(fluence, _let(ion))
    assert comp.absorbed_dose_Gy(_let) == pytest.approx(expected)


def test_fluence_component_can_carry_energy_bin_support() -> None:
    ion = Ion(Z=6, A=12, energy_MeV_u=10.0)
    comp = SpectrumComponent(ion, fluence=1.0e8, energy_lo_MeV_u=1.0, energy_hi_MeV_u=100.0)
    assert comp.energy_bounds_MeV_u() == (pytest.approx(1.0), pytest.approx(100.0))
    assert comp.is_energy_bin


def test_mixed_field_weights() -> None:
    c12 = Ion(Z=6, A=12, energy_MeV_u=200.0)
    he4 = Ion(Z=2, A=4, energy_MeV_u=100.0)
    field = RadiationField(
        [SpectrumComponent(c12, dose_Gy=1.0), SpectrumComponent(he4, fluence=1.0e9)]
    )
    assert field.is_mixed
    weights = field.dose_weights(_let)
    assert weights[0] == pytest.approx(1.0)
    assert weights[1] == pytest.approx(fluence_to_dose(1.0e9, _let(he4)))
    assert field.total_dose_Gy(_let) == pytest.approx(sum(weights))


def test_component_requires_exactly_one_weight() -> None:
    ion = Ion(Z=1, A=1, energy_MeV_u=150.0)
    with pytest.raises(ValueError):
        SpectrumComponent(ion)  # neither
    with pytest.raises(ValueError):
        SpectrumComponent(ion, fluence=1.0, dose_Gy=1.0)  # both


def test_relativistic_beta() -> None:
    # Monotonic increasing with energy, in (0, 1).
    b_low = relativistic_beta(10.0)
    b_high = relativistic_beta(400.0)
    assert 0.0 < b_low < b_high < 1.0
    # 100 MeV/u -> beta ~ 0.43 (gamma = 1 + 100/931.494).
    assert relativistic_beta(100.0) == pytest.approx(0.4295, abs=2e-3)
    assert Ion(6, 12, 100.0).beta() == pytest.approx(relativistic_beta(100.0))


def test_relativistic_beta_requires_positive_energy() -> None:
    with pytest.raises(ValueError):
        relativistic_beta(0.0)


def test_ion_validation() -> None:
    with pytest.raises(ValueError):
        Ion(Z=0, A=1, energy_MeV_u=100.0)
    with pytest.raises(ValueError):
        Ion(Z=1, A=1, energy_MeV_u=0.0)


def test_empty_field_rejected() -> None:
    with pytest.raises(ValueError):
        RadiationField([])


def test_negative_weights_rejected() -> None:
    ion = Ion(Z=1, A=1, energy_MeV_u=150.0)
    with pytest.raises(ValueError):
        SpectrumComponent(ion, fluence=-1.0)
    with pytest.raises(ValueError):
        SpectrumComponent(ion, dose_Gy=-1.0)


@pytest.mark.parametrize(
    "kwargs",
    [
        {"energy_lo_MeV_u": 1.0},
        {"energy_hi_MeV_u": 2.0},
        {"energy_lo_MeV_u": 0.0, "energy_hi_MeV_u": 2.0},
        {"energy_lo_MeV_u": 2.0, "energy_hi_MeV_u": 1.0},
        {"energy_lo_MeV_u": 101.0, "energy_hi_MeV_u": 200.0},
    ],
)
def test_energy_bin_validation(kwargs: dict[str, float]) -> None:
    ion = Ion(Z=1, A=1, energy_MeV_u=100.0)
    with pytest.raises(ValueError):
        SpectrumComponent(ion, fluence=1.0, **kwargs)
