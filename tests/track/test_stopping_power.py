"""Water stopping-power sources (analytic Bethe, tables, adapters)."""

from __future__ import annotations

import sys

import pytest

from pyradbio.core.radiation import Ion
from pyradbio.models.mkm.pymkm_backend import PyMKMStoppingPower, from_pymkm_source
from pyradbio.track.stopping_power import (
    DefaultStoppingPower,
    LibdedxStoppingPower,
    MassStoppingPowerSource,
    SimpleBetheStoppingPower,
    TabularStoppingPower,
    default_stopping_power,
    effective_charge,
)
from tests.conftest import requires_pymkm


def test_proton_100mev_let_ballpark() -> None:
    """100 MeV proton in water: PSTAR collision LET ~ 0.73 keV/um.

    Reference: NIST PSTAR liquid water, 100 MeV -> 7.289 MeV cm^2/g.
    The Bethe approximation should land within a few percent.
    """
    sp = SimpleBetheStoppingPower()
    let = sp.let_keV_um(Ion(Z=1, A=1, energy_MeV_u=100.0))
    assert let == pytest.approx(0.729, rel=0.05)


def test_z_squared_scaling_at_high_energy() -> None:
    """Fully stripped ions scale ~ Z^2 at equal energy/u (carbon vs proton)."""
    sp = SimpleBetheStoppingPower()
    e = 100.0
    let_p = sp.let_keV_um(Ion(1, 1, e))
    let_c = sp.let_keV_um(Ion(6, 12, e))
    assert let_c / let_p == pytest.approx(36.0, rel=0.05)


def test_let_decreases_with_energy_in_therapy_range() -> None:
    sp = SimpleBetheStoppingPower()
    lets = [sp.let_keV_um(Ion(6, 12, e)) for e in (50.0, 100.0, 200.0, 350.0)]
    assert all(lets[i] > lets[i + 1] for i in range(len(lets) - 1))


def test_effective_charge_limits() -> None:
    # high velocity -> approaches Z
    assert effective_charge(6, beta=0.9) == pytest.approx(6.0, abs=0.05)
    # low velocity -> reduced below Z
    assert effective_charge(6, beta=0.02) < 6.0


def test_bethe_invalid_at_very_low_energy() -> None:
    """Below ~0.035 MeV/u the Bethe logarithm turns non-positive and is invalid."""
    sp = SimpleBetheStoppingPower()
    with pytest.raises(ValueError):
        sp.let_keV_um(Ion(6, 12, 0.01))


@requires_pymkm
def test_bethe_matches_reference_tables_within_a_few_percent() -> None:
    """SimpleBethe tracks pyMKM's Geant4 11.3 water table to <=~4% for Z<=10.

    This pins the analytic model's accuracy ceiling over the ion-therapy range
    (1-200 MeV/u; below 1 MeV/u it is up to ~30% low). The effective-charge Bethe
    form is already near-optimal here: the standard shell and Bloch corrections
    both *regress* this agreement (the empirical effective charge already absorbs
    those close-collision effects), so they are deliberately not applied. Heavy
    ions (Z>10) have no reference table here and remain the model's main
    uncertainty -- use a libdedx-backed source there.
    """
    ref = PyMKMStoppingPower("geant4_11_3_0")
    bethe = SimpleBetheStoppingPower()
    worst = 0.0
    for z, a in [(1, 1), (2, 4), (6, 12), (8, 16), (10, 20)]:
        for energy in (1.0, 2.0, 3.0, 5.0, 10.0, 20.0, 50.0, 100.0, 200.0):
            ion = Ion(z, a, energy)
            rel = abs(bethe.let_keV_um(ion) - ref.let_keV_um(ion)) / ref.let_keV_um(ion)
            worst = max(worst, rel)
    assert worst < 0.04


def test_mass_stopping_power_source_converts_units() -> None:
    """Wrapping a mass-SP callable [MeV cm^2/g] yields LET [keV/um].

    LET = mass_SP * density * 0.1 (1 MeV/cm = 0.1 keV/um). A 5 MeV cm^2/g source
    in unit-density water is therefore 0.5 keV/um, independent of the ion.
    """
    calls: list[tuple[int, int, float]] = []

    def fake_mass_sp(z: int, a: int, energy_MeV_u: float) -> float:
        calls.append((z, a, energy_MeV_u))
        return 5.0

    sp = MassStoppingPowerSource(fake_mass_sp)
    assert sp.let_keV_um(Ion(6, 12, 100.0)) == pytest.approx(0.5)
    assert calls == [(6, 12, 100.0)]


def test_libdedx_missing_raises_helpful_importerror(monkeypatch: pytest.MonkeyPatch) -> None:
    """Without the optional libdedx package, construction fails with guidance."""
    monkeypatch.setitem(sys.modules, "libdedx", None)
    with pytest.raises(ImportError, match="libdedx"):
        LibdedxStoppingPower()


def test_libdedx_stopping_power_matches_reference() -> None:
    """When libdedx is importable, LET is reference-grade and keyed by Z.

    100 MeV proton in water -> PSTAR ~0.729 keV/um; libdEdx returns mass stopping
    power [MeV cm^2/g], converted here to keV/um. Skipped unless libdedx is
    installed in the running environment (it is a GPL runtime opt-in).
    """
    pytest.importorskip("libdedx")
    sp = LibdedxStoppingPower()
    assert sp.let_keV_um(Ion(1, 1, 100.0)) == pytest.approx(0.729, rel=0.02)
    # Reaches heavy ions the bundled tables/Bethe cannot validate (Z=92).
    assert sp.let_keV_um(Ion(92, 238, 10.0)) > 1000.0
    # Vectorized lookup agrees with scalar and is Z-keyed (A ignored).
    vec = sp.lets_keV_um(6, 12, [1.0, 10.0])
    assert vec[1] == pytest.approx(sp.let_keV_um(Ion(6, 12, 10.0)))


def test_water_stopping_power_alias_is_gone() -> None:
    """No deprecation aliases before the first release: one name per class."""
    import pyradbio.track.stopping_power as sp_module

    assert not hasattr(sp_module, "WaterStoppingPower")


def test_default_stopping_power_is_icru90_procedure_with_bethe_fallback() -> None:
    """Every model default goes through this factory: ICRU 90's
    procedure for Z 1-18 inside its range, the analytic Bethe beyond -- anchored to the
    adjacent branch at its edge, bare where there is none."""
    from pyradbio.track.stopping import ICRU90Procedure

    sp = default_stopping_power()
    assert isinstance(sp, DefaultStoppingPower)
    assert default_stopping_power() is not sp  # a fresh instance per call
    carbon = Ion(6, 12, 50.0)
    assert sp.source(carbon) == "icru90_procedure"
    assert sp.let_keV_um(carbon) == pytest.approx(ICRU90Procedure().let_keV_um(carbon), rel=1e-12)
    for ion in (Ion(26, 56, 1.0), Ion(1, 1, 2e6)):  # Z > 18 below 2.5 MeV/u; above the range
        assert sp.source(ion) == "simple_bethe_anchored"
    for ion in (Ion(93, 237, 10.0), Ion(26, 99, 10.0)):  # no shipped mass: no branch
        assert sp.source(ion) == "simple_bethe"
        assert sp.let_keV_um(ion) == SimpleBetheStoppingPower().let_keV_um(ion)


def test_default_stopping_power_switches_per_energy_within_one_call() -> None:
    sp = default_stopping_power()
    hi = sp.procedure.energy_range(1, 1)[1]
    lets = sp.lets_keV_um(1, 1, [100.0, hi, hi * 2])
    assert lets[0] == pytest.approx(sp.procedure.let_keV_um(Ion(1, 1, 100.0)), rel=1e-12)
    bethe = SimpleBetheStoppingPower()
    anchor = lets[1] / bethe.let_keV_um(Ion(1, 1, hi))
    assert lets[2] == pytest.approx(anchor * bethe.let_keV_um(Ion(1, 1, hi * 2)), rel=1e-12)


def test_default_stopping_power_scales_with_density() -> None:
    dense = default_stopping_power(density_g_cm3=1.1)
    unit = default_stopping_power()
    ion = Ion(6, 12, 50.0)
    assert dense.let_keV_um(ion) == pytest.approx(1.1 * unit.let_keV_um(ion), rel=1e-12)


def test_tabular_stopping_power_interpolates_single_and_many() -> None:
    sp = TabularStoppingPower.from_arrays({6: ([1.0, 10.0, 100.0], [100.0, 10.0, 1.0])})

    assert sp.let_keV_um(Ion(6, 12, 10.0)) == pytest.approx(10.0)
    assert sp.let_keV_um(Ion(6, 12, 10.0**0.5)) == pytest.approx(10.0**1.5)

    lets = sp.lets_keV_um(Z=6, A=12, energies_MeV_u=[1.0, 10.0, 100.0])
    assert lets.tolist() == pytest.approx([100.0, 10.0, 1.0])


def test_tabular_stopping_power_rejects_unknown_ion_and_extrapolation() -> None:
    sp = TabularStoppingPower.from_arrays({6: ([1.0, 10.0, 100.0], [100.0, 10.0, 1.0])})

    with pytest.raises(KeyError, match="Z=2"):
        sp.let_keV_um(Ion(2, 4, 10.0))
    with pytest.raises(ValueError, match="outside tabulated range"):
        sp.let_keV_um(Ion(6, 12, 0.5))


def test_from_pymkm_source_returns_adapter_when_backend_available(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    monkeypatch.setattr(
        PyMKMStoppingPower,
        "_load_pymkm_table_set",
        staticmethod(lambda source: {6: {"energy": [1.0, 10.0], "let": [1000.0, 100.0]}}),
    )

    sp = from_pymkm_source("fluka_2020_0")

    assert isinstance(sp, PyMKMStoppingPower)
    assert sp.let_keV_um(Ion(6, 12, 10.0)) == pytest.approx(10.0)


def test_from_pymkm_source_errors_when_pymkm_missing(monkeypatch: pytest.MonkeyPatch) -> None:
    def _missing_table_set(source: str) -> object:
        raise ImportError("missing pyMKM")

    monkeypatch.setattr(
        PyMKMStoppingPower,
        "_load_pymkm_table_set",
        staticmethod(_missing_table_set),
    )

    with pytest.raises(ImportError, match="missing pyMKM"):
        from_pymkm_source("fluka_2020_0")


def test_pymkm_stopping_power_reuses_backend_and_let_cache(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """Repeated LET lookups interpolate one loaded pyMKM table set."""

    calls: list[tuple[str, object]] = []

    def _fake_table_set(source: str) -> object:
        calls.append(("load", source))
        return {6: {"energy": [1.0, 10.0, 100.0], "let": [1000.0, 100.0, 10.0]}}

    monkeypatch.setattr(
        PyMKMStoppingPower,
        "_load_pymkm_table_set",
        staticmethod(_fake_table_set),
    )

    sp = PyMKMStoppingPower("fluka_2020_0")
    low = Ion(6, 12, 10.0)
    high = Ion(6, 12, 100.0)

    assert sp.let_keV_um(low) == pytest.approx(10.0)
    assert sp.let_keV_um(low) == pytest.approx(10.0)
    assert sp.let_keV_um(high) == pytest.approx(1.0)
    assert sp.lets_keV_um(Z=6, A=12, energies_MeV_u=[10.0, 100.0]).tolist() == pytest.approx(
        [10.0, 1.0]
    )

    assert calls == [("load", "fluka_2020_0")]
