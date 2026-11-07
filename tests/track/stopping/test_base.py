"""The common interface of the water stopping-power classes (track/stopping/base.py)."""

from __future__ import annotations

import math
from collections.abc import Mapping

import numpy as np
import numpy.typing as npt
import pytest

from pyradbio.core.constants import ATOMIC_MASS_UNIT_MEV, MEV_PER_CM_TO_KEV_PER_UM
from pyradbio.core.radiation import Ion, relativistic_beta
from pyradbio.track.stopping.base import (
    LIQUID_WATER,
    IonSourceInfo,
    MassStoppingPowerModel,
    OutOfRangeError,
    UnsupportedIonError,
    beta_from_energy_per_nucleon,
    energy_per_nucleon_MeV_u,
    total_kinetic_energy_MeV,
)
from pyradbio.track.stopping_power import VectorizedStoppingPower


class _PowerLaw(MassStoppingPowerModel):
    """A synthetic model: S = Z^2 * 100 / E on [0.1, 100] MeV/u, carbon and protons only."""

    name = "power-law"

    def __init__(self, density_g_cm3: float = 1.0) -> None:
        self.density_g_cm3 = density_g_cm3

    def supported_Z(self) -> tuple[int, ...]:
        return (1, 6)

    def energy_range(self, Z: int, A: int) -> tuple[float, float]:
        self._require_supported(Z)
        return (0.1, 100.0)

    def metadata(self, Z: int, A: int) -> IonSourceInfo:
        lo, hi = self.energy_range(Z, A)
        return IonSourceInfo(
            Z=Z,
            A=A,
            source="synthetic",
            reference="none",
            i_value_eV=75.0,
            energy_range_MeV_u=(lo, hi),
            options={"quantity": "electronic"},
        )

    def _evaluate(
        self, Z: int, A: int, energy_MeV_u: npt.NDArray[np.float64]
    ) -> npt.NDArray[np.float64]:
        return Z * Z * 100.0 / energy_MeV_u


def test_mass_stopping_power_returns_an_array_for_scalars_and_sequences() -> None:
    model = _PowerLaw()
    assert model.mass_stopping_power(6, 12, 10.0).shape == (1,)
    values = model.mass_stopping_power(6, 12, [1.0, 10.0])
    np.testing.assert_allclose(values, [3600.0, 360.0])


@pytest.mark.parametrize("energy", [0.0999, 100.01, [1.0, 200.0]])
def test_out_of_range_raises_instead_of_extrapolating(energy: float | list[float]) -> None:
    with pytest.raises(OutOfRangeError) as info:
        _PowerLaw().mass_stopping_power(6, 12, energy)
    assert info.value.valid_range_MeV_u == (0.1, 100.0)
    assert info.value.Z == 6
    assert isinstance(info.value, ValueError)  # callers catching ValueError still work


def test_range_endpoints_themselves_are_in_range() -> None:
    np.testing.assert_allclose(_PowerLaw().mass_stopping_power(1, 1, [0.1, 100.0]), [1000.0, 1.0])


@pytest.mark.parametrize("energy", [float("nan"), -1.0, 0.0])
def test_non_physical_energies_are_rejected(energy: float) -> None:
    with pytest.raises(ValueError, match="positive"):
        _PowerLaw().mass_stopping_power(1, 1, energy)


def test_unsupported_ion_raises() -> None:
    with pytest.raises(UnsupportedIonError, match="Z=2"):
        _PowerLaw().mass_stopping_power(2, 4, 10.0)


def test_let_adapter_satisfies_the_existing_stopping_power_protocol() -> None:
    model: VectorizedStoppingPower = _PowerLaw(density_g_cm3=1.0)
    expected = 3600.0 * MEV_PER_CM_TO_KEV_PER_UM
    assert model.let_keV_um(Ion(6, 12, 1.0)) == pytest.approx(expected)
    np.testing.assert_allclose(model.lets_keV_um(6, 12, [1.0, 10.0]), [expected, expected / 10])


def test_let_adapter_uses_the_density() -> None:
    assert _PowerLaw(density_g_cm3=2.0).let_keV_um(Ion(1, 1, 1.0)) == pytest.approx(20.0)


def test_metadata_records_source_i_value_range_and_options() -> None:
    info = _PowerLaw().metadata(6, 12)
    assert info.material == LIQUID_WATER
    assert info.energy_range_MeV_u == (0.1, 100.0)
    assert info.options["quantity"] == "electronic"
    assert IonSourceInfo.from_dict(info.to_dict()) == info


def test_ion_source_info_options_are_read_only() -> None:
    info = _PowerLaw().metadata(1, 1)
    assert isinstance(info.options, Mapping)
    with pytest.raises(TypeError):
        info.options["quantity"] = "total"  # type: ignore[index]


def test_ion_source_info_rejects_an_inverted_range() -> None:
    with pytest.raises(ValueError, match="energy_range"):
        IonSourceInfo(
            Z=1,
            A=1,
            source="x",
            reference="x",
            i_value_eV=75.0,
            energy_range_MeV_u=(10.0, 1.0),
        )


# --- the energy convention (kinetic energy per nucleon, E/A) ---------------------------


def test_energy_per_nucleon_is_total_kinetic_energy_over_mass_number() -> None:
    # NIST ASTAR quotes the total kinetic energy of a He-4 ion in MeV.
    assert energy_per_nucleon_MeV_u(8.0, 4) == 2.0
    assert total_kinetic_energy_MeV(2.0, 4) == 8.0


def test_a_proton_in_MeV_per_nucleon_is_plain_MeV() -> None:
    assert energy_per_nucleon_MeV_u(100.0, 1) == 100.0


def test_beta_uses_the_true_ion_mass() -> None:
    # With the mass set to A atomic mass units it reduces to the package's
    # per-nucleon relativistic_beta (which assumes m = A u).
    for e_u, a in ((0.5, 1), (100.0, 12), (1000.0, 4)):
        assert beta_from_energy_per_nucleon(e_u, a, a * ATOMIC_MASS_UNIT_MEV) == pytest.approx(
            relativistic_beta(e_u), rel=1e-14
        )
    # A heavier ion at the same E/A is slower.
    light = beta_from_energy_per_nucleon(10.0, 1, ATOMIC_MASS_UNIT_MEV)
    heavy = beta_from_energy_per_nucleon(10.0, 1, 1.01 * ATOMIC_MASS_UNIT_MEV)
    assert heavy < light


def test_beta_matches_the_relativistic_definition() -> None:
    t, m = 250.0, 3000.0
    gamma = 1.0 + t / m
    assert beta_from_energy_per_nucleon(t, 1, m) == pytest.approx(
        math.sqrt(1.0 - 1.0 / gamma**2), rel=1e-15
    )


@pytest.mark.parametrize(("e", "a", "m"), [(0.0, 1, 938.0), (1.0, 0, 938.0), (1.0, 1, 0.0)])
def test_beta_rejects_non_physical_input(e: float, a: int, m: float) -> None:
    with pytest.raises(ValueError):
        beta_from_energy_per_nucleon(e, a, m)


def test_mass_stopping_power_rejects_a_two_dimensional_energy_array() -> None:
    with pytest.raises(ValueError, match="one-dimensional"):
        _PowerLaw().mass_stopping_power(1, 1, np.ones((2, 2)))


@pytest.mark.parametrize("convert", [energy_per_nucleon_MeV_u, total_kinetic_energy_MeV])
def test_energy_conversions_reject_a_zero_mass_number(convert: object) -> None:
    with pytest.raises(ValueError, match="mass number"):
        convert(1.0, 0)  # type: ignore[operator]


def test_settings_are_the_resolved_options_as_json() -> None:
    """``settings()``: what a table header records about a source."""
    import json

    from pyradbio.track.stopping import ICRU, XSTAR, Bethe, ICRU90Procedure
    from pyradbio.track.stopping.hubert import HubertAlphaScaling
    from pyradbio.track.stopping_power import SimpleBetheStoppingPower, default_stopping_power

    icru = ICRU(edition="2009").settings()
    assert icru["name"] == "icru" and icru["edition"] == "2009"
    assert not any(key.startswith("_") for key in icru)  # caches stay out
    assert icru["nuclear"] == ICRU().nuclear.describe()
    bethe = Bethe(shell=True).settings()
    assert bethe["effective_charge"] == {"name": "hubert1990 (Z2 = 8)", "target_Z": 8.0}
    assert bethe["i_value_eV"] == 78.0
    assert bethe["corrections"]["shell"] is True  # type: ignore[index]
    hubert = HubertAlphaScaling().settings()
    assert hubert["alpha"]["name"] == ICRU90Procedure.name  # type: ignore[index]
    for source in (
        XSTAR(),
        ICRU90Procedure(),
        SimpleBetheStoppingPower(),
        default_stopping_power(),
    ):
        json.dumps(source.settings())  # type: ignore[attr-defined]


def test_settings_record_the_parameters_of_parameterized_collaborators() -> None:
    """Regression: BrandtKitagawaEffectiveCharge must serialize its parameters, not only its
    name, because band_gap_eV = 0 and 78 give different carbon alphas (1.33432 vs 1.33373)."""
    from pyradbio.track.stopping import Bethe
    from pyradbio.track.stopping.effective_charge import (
        BarkasEffectiveCharge,
        BrandtKitagawaEffectiveCharge,
        HubertEffectiveCharge,
    )

    def charge(model: object) -> dict[str, object]:
        out = Bethe(effective_charge=model).settings()["effective_charge"]  # type: ignore[arg-type]
        assert isinstance(out, dict)
        return out

    bk = charge(BrandtKitagawaEffectiveCharge(plasma_energy_eV=20.0, band_gap_eV=78.0))
    assert bk == {"name": "brandt-kitagawa1982", "plasma_energy_eV": 20.0, "band_gap_eV": 78.0}
    assert charge(BarkasEffectiveCharge(coefficient=162.0))["coefficient"] == 162.0
    assert charge(HubertEffectiveCharge(target_Z=7.0))["target_Z"] == 7.0
    density = Bethe(i_value_eV=75.0).settings()["density_effect"]
    assert isinstance(density, dict) and density["i_value_eV"] == 75.0
    assert "values" not in density  # tables stay out; their source names them
