"""MKM model (default native backend) and the pluggable backend seam."""

from __future__ import annotations

import importlib.util

import pytest

from pyradbio.core.radiation import Ion
from pyradbio.models.mkm import MKMBackend, MKMModel, NativeMKMBackend
from pyradbio.models.mkm.base import BackendZStarRow, BackendZStarTable

requires_pymkm = pytest.mark.skipif(
    importlib.util.find_spec("pymkm") is None,
    reason="pyMKM not installed ([mkm] extra)",
)


class _FixedZStarBackend:
    """Fake ``MKMBackend`` returning a known ``z*_1D``.

    Lets us test the MKM biophysics and the backend seam without pyMKM and
    without inventing an energy -> z* mapping (z* is supplied directly).
    """

    def __init__(self, z_star_1D_Gy: float) -> None:
        self._z = z_star_1D_Gy

    def z_star_1D_Gy(self, ion: Ion) -> float:
        return self._z

    def z_star_1D_table_Gy(self, ions: object, energies_MeV_u: object) -> object:
        raise NotImplementedError

    def let_keV_um(self, ion: Ion) -> float:
        raise NotImplementedError("this fake backend cannot resolve LET")


class _FixedZStarTableBackend:
    """Fake backend returning the protocol table type without pandas."""

    def z_star_1D_Gy(self, ion: Ion) -> float:
        return 3.43

    def let_keV_um(self, ion: Ion) -> float:
        return 100.0

    def z_star_1D_table_Gy(self, ions: object, energies_MeV_u: object) -> BackendZStarTable:
        return BackendZStarTable(
            [
                BackendZStarRow(ion_Z=6, energy_MeV_u=11.0, let_keV_um=100.0, z_star_1D_Gy=3.43),
                BackendZStarRow(ion_Z=6, energy_MeV_u=100.0, let_keV_um=20.0, z_star_1D_Gy=1.0),
            ]
        )


class _FakeColumn:
    def __init__(self, values: list[float]) -> None:
        self._values = values
        self.iloc = self

    def __getitem__(self, index: int) -> float:
        return self._values[index]

    def to_numpy(self) -> list[float]:
        return self._values


class _FakePymkmData:
    def __init__(self) -> None:
        self._columns = {
            "energy": _FakeColumn([11.0]),
            "let": _FakeColumn([42.0]),
            "z_bar_star_domain": _FakeColumn([3.43]),
            "z_bar_domain": _FakeColumn([99.0]),
        }

    def __getitem__(self, key: str) -> _FakeColumn:
        return self._columns[key]


class _FakePymkmTable:
    def __init__(self) -> None:
        self.table = {
            "Carbon": {
                "stopping_power_info": {"atomic_number": 6},
                "data": _FakePymkmData(),
            }
        }


def test_default_backend_errors_without_pymkm(monkeypatch: pytest.MonkeyPatch) -> None:
    """Without pyMKM installed, the default backend must raise a helpful error."""
    import builtins

    from pyradbio.models.mkm.pymkm_backend import PyMKMBackend

    real_import = builtins.__import__

    def _no_pymkm(name: str, *args: object, **kwargs: object) -> object:
        if name == "pymkm" or name.startswith("pymkm."):
            raise ImportError("No module named 'pymkm'")
        return real_import(name, *args, **kwargs)

    monkeypatch.setattr(builtins, "__import__", _no_pymkm)
    backend = PyMKMBackend()
    with pytest.raises(ImportError, match=r"pyradbio\[mkm\]"):
        backend.z_star_1D_Gy(Ion(Z=6, A=12, energy_MeV_u=11.0))


def test_default_backend_table_errors_without_pymkm_before_pandas(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """The default table path should not require pandas before it checks pyMKM."""
    import builtins

    from pyradbio.models.mkm.pymkm_backend import PyMKMBackend

    real_import = builtins.__import__

    def _guard_optional_imports(name: str, *args: object, **kwargs: object) -> object:
        if name == "pandas" or name.startswith("pandas."):
            raise AssertionError("pandas must not be imported before the pyMKM guard")
        if name == "pymkm" or name.startswith("pymkm."):
            raise ImportError("No module named 'pymkm'")
        return real_import(name, *args, **kwargs)

    monkeypatch.setattr(builtins, "__import__", _guard_optional_imports)
    backend = PyMKMBackend()
    with pytest.raises(ImportError, match=r"pyradbio\[mkm\]"):
        backend.z_star_1D_table_Gy([6], [11.0])


def test_backend_is_pluggable() -> None:
    """A custom MKMBackend can be injected, so MKM is not pyMKM-locked."""
    backend = _FixedZStarBackend(3.43)
    assert isinstance(backend, MKMBackend)  # structural conformance
    model = MKMModel(alpha0=0.0708, beta0=0.0615, backend=backend)
    assert model.backend is backend


def test_alpha_beta_table_uses_protocol_table_not_pandas() -> None:
    """The MKM table seam accepts its own table rows, not pandas-only objects."""
    model = MKMModel(alpha0=0.0708, beta0=0.0615, backend=_FixedZStarTableBackend())
    table = model.alpha_beta_z_star_table([6], [11.0, 100.0])

    assert len(table.rows) == 2
    assert table.rows[0].alpha == pytest.approx(0.281745)
    assert table.rows[0].beta == pytest.approx(0.0615)
    assert table.rows[0].z_star_1D_Gy == pytest.approx(3.43)


def test_alpha_beta_table_is_the_shared_alpha_beta_table_type() -> None:
    """MKMModel no longer overrides ``alpha_beta_table``: every model returns one type."""
    from pyradbio.tables.alpha_beta import AlphaBetaTable

    model = MKMModel(alpha0=0.0708, beta0=0.0615, backend=_FixedZStarTableBackend())
    table = model.alpha_beta_table([6], [11.0, 100.0])

    assert isinstance(table, AlphaBetaTable)
    assert [(row.Z, row.A) for row in table.rows] == [(6, 12), (6, 12)]
    assert table.rows[0].alpha == pytest.approx(0.0708 + 0.0615 * 3.43)


@pytest.mark.golden
def test_golden_inaniwa_alpha_from_zstar() -> None:
    """alpha = alpha0 + beta*z*_1D (Inaniwa 2010, Sec. 2.5).

    With the cited z*_1D=3.43 Gy at 7.3 mmWEL, alpha0=0.0708 Gy^-1 and
    beta=0.0615 Gy^-2  ->  alpha = 0.282 Gy^-1. An injected backend supplies
    z*_1D so the biophysics is checked independently of any engine.
    """
    model = MKMModel(alpha0=0.0708, beta0=0.0615, backend=_FixedZStarBackend(3.43))
    ab = model.alpha_beta(Ion(Z=6, A=12, energy_MeV_u=11.0))
    assert ab.alpha == pytest.approx(0.282, abs=1e-3)
    assert ab.beta == pytest.approx(0.0615, abs=1e-4)


def test_pymkm_backend_reads_z_bar_star_domain_column() -> None:
    """The adapter must use pyMKM's saturation-corrected z_bar_star_domain column."""
    from pyradbio.models.mkm.pymkm_backend import PyMKMBackend

    class BackendWithFakeTable(PyMKMBackend):
        def _compute_table(self, ions: object, energies_MeV_u: object) -> _FakePymkmTable:
            return _FakePymkmTable()

    backend = BackendWithFakeTable()
    assert backend.z_star_1D_Gy(Ion(Z=6, A=12, energy_MeV_u=11.0)) == pytest.approx(3.43)
    table = backend.z_star_1D_table_Gy([6], [11.0])
    assert table.rows[0].z_star_1D_Gy == pytest.approx(3.43)
    # pyMKM's "let" column is MeV/cm (42 MeV/cm = 4.2 keV/um).
    assert table.rows[0].let_keV_um == pytest.approx(4.2)
    assert backend.let_keV_um(Ion(Z=6, A=12, energy_MeV_u=11.0)) == pytest.approx(4.2)


def test_default_backend_is_native_and_built_once() -> None:
    """The default backend is the pyMKM-free NativeMKMBackend, built from the geometry."""
    model = MKMModel(domain_radius_um=0.30, nucleus_radius_um=4.0, alpha0=0.172, beta0=0.06)
    backend = model.backend
    assert isinstance(backend, NativeMKMBackend)
    assert (backend.domain_radius_um, backend.nucleus_radius_um, backend.beta0) == (
        0.30,
        4.0,
        0.06,
    )
    ab = model.alpha_beta(Ion(Z=6, A=12, energy_MeV_u=11.0))
    assert ab.alpha > 0.0
    assert ab.beta == pytest.approx(0.06)
    assert model.backend is backend  # not rebuilt per call


def test_default_geometry_is_inaniwa_2010_hsg() -> None:
    model = MKMModel()
    assert (model.domain_radius_um, model.nucleus_radius_um, model.beta0) == (0.32, 3.9, 0.0615)


def test_injected_backend_owns_the_geometry() -> None:
    """Unset model geometry is taken from the backend, not silently left at defaults."""
    backend = NativeMKMBackend(domain_radius_um=0.28, nucleus_radius_um=4.3, beta0=0.05)
    model = MKMModel(alpha0=0.1, backend=backend)
    assert (model.domain_radius_um, model.nucleus_radius_um, model.beta0) == (0.28, 4.3, 0.05)
    assert model.alpha_beta_z_star_table([6], [100.0]).rows[0].beta == pytest.approx(0.05)


@pytest.mark.parametrize(
    "field", [{"domain_radius_um": 0.32}, {"nucleus_radius_um": 3.9}, {"beta0": 0.0615}]
)
def test_conflicting_geometry_with_injected_backend_raises(field: dict[str, float]) -> None:
    backend = NativeMKMBackend(domain_radius_um=0.28, nucleus_radius_um=4.3, beta0=0.05)
    with pytest.raises(ValueError, match="conflicts with the injected backend"):
        MKMModel(backend=backend, **field)


def test_matching_geometry_with_injected_backend_is_accepted() -> None:
    backend = NativeMKMBackend(domain_radius_um=0.28, nucleus_radius_um=4.3, beta0=0.05)
    model = MKMModel(domain_radius_um=0.28, nucleus_radius_um=4.3, beta0=0.05, backend=backend)
    assert model.backend is backend


def test_mkm_model_is_frozen_and_resolves_its_backend_once() -> None:
    """Frozen like every other model; ``stopping_power_source`` is gone (inject a backend)."""
    import dataclasses

    model = MKMModel()
    assert isinstance(model.backend, NativeMKMBackend)
    assert model.domain_radius_um == pytest.approx(0.32)
    with pytest.raises(dataclasses.FrozenInstanceError):
        model.alpha0 = 0.2  # type: ignore[misc]
    with pytest.raises(TypeError, match="stopping_power_source"):
        MKMModel(stopping_power_source="geant4_11_3_0")  # type: ignore[call-arg]


@requires_pymkm
def test_injected_pymkm_backend_runs() -> None:
    """pyMKM, injected explicitly, yields LQ parameters for a carbon ion."""
    from pyradbio.models.mkm.pymkm_backend import PyMKMBackend

    model = MKMModel(alpha0=0.172, backend=PyMKMBackend())
    ab = model.alpha_beta(Ion(Z=6, A=12, energy_MeV_u=11.0))
    assert ab.alpha > 0.0
    assert ab.beta == pytest.approx(0.0615)


@requires_pymkm
def test_zstar_monotonic_with_energy() -> None:
    """z*_1D decreases as energy rises (LET falls) -- Inaniwa 2010 Fig. 5 shape.

    Lower energy carbon deposits more local dose per track, so the
    saturation-corrected dose-mean specific energy is larger.
    """
    from pyradbio.models.mkm.pymkm_backend import PyMKMBackend

    backend = PyMKMBackend(domain_radius_um=0.32, nucleus_radius_um=3.9, beta0=0.0615)
    z_low = backend.z_star_1D_Gy(Ion(Z=6, A=12, energy_MeV_u=11.0))
    z_mid = backend.z_star_1D_Gy(Ion(Z=6, A=12, energy_MeV_u=100.0))
    z_high = backend.z_star_1D_Gy(Ion(Z=6, A=12, energy_MeV_u=400.0))
    assert z_low > z_mid > z_high > 0.0


def test_mixed_field_dose_averages_zstar() -> None:
    """A mixed field dose-averages z*_1D (Inaniwa 2010, eq. 8).

    Two equal-dose carbon components must yield an alpha between the two
    single-ion alphas (the dose-average of their z*_1D).
    """
    from pyradbio.core.radiation import RadiationField, SpectrumComponent

    model = MKMModel(domain_radius_um=0.32, nucleus_radius_um=3.9, alpha0=0.172, beta0=0.0615)
    low = Ion(Z=6, A=12, energy_MeV_u=11.0)
    high = Ion(Z=6, A=12, energy_MeV_u=400.0)
    a_low = model.alpha_beta(low).alpha
    a_high = model.alpha_beta(high).alpha
    field = RadiationField(
        [SpectrumComponent(low, dose_Gy=1.0), SpectrumComponent(high, dose_Gy=1.0)]
    )
    a_mix = model.alpha_beta(field).alpha
    assert a_high < a_mix < a_low
    # equal doses -> arithmetic mean of the two alphas (both share beta0)
    assert a_mix == pytest.approx(0.5 * (a_low + a_high), rel=1e-9)


def test_mixed_field_by_fluence_uses_backend_let() -> None:
    """Fluence-specified components are weighted by dose via the backend LET.

    Equal *fluence* of a low- and high-energy carbon weights the low-energy
    (higher-LET) component more, so the mixed alpha sits closer to the
    low-energy alpha than the equal-dose midpoint.
    """
    from pyradbio.core.radiation import RadiationField, SpectrumComponent

    model = MKMModel(domain_radius_um=0.32, nucleus_radius_um=3.9, alpha0=0.172, beta0=0.0615)
    low = Ion(Z=6, A=12, energy_MeV_u=11.0)
    high = Ion(Z=6, A=12, energy_MeV_u=400.0)
    a_low = model.alpha_beta(low).alpha
    a_high = model.alpha_beta(high).alpha
    field = RadiationField(
        [SpectrumComponent(low, fluence=1.0e7), SpectrumComponent(high, fluence=1.0e7)]
    )
    a_mix = model.alpha_beta(field).alpha
    assert a_high < a_mix < a_low
    assert a_mix > 0.5 * (a_low + a_high)  # high-LET (low E) dominates the dose


def test_mixed_field_by_fluence_rejected_without_let() -> None:
    """A backend whose ``let_keV_um`` cannot answer stops fluence weighting."""
    from pyradbio.core.radiation import RadiationField, SpectrumComponent

    model = MKMModel(alpha0=0.0708, beta0=0.0615, backend=_FixedZStarBackend(3.43))
    field = RadiationField(
        [
            SpectrumComponent(Ion(Z=6, A=12, energy_MeV_u=11.0), fluence=1.0e7),
            SpectrumComponent(Ion(Z=6, A=12, energy_MeV_u=400.0), fluence=1.0e7),
        ]
    )
    with pytest.raises(NotImplementedError, match="cannot resolve LET"):
        model.alpha_beta(field)


def test_alpha_beta_table_adds_lq_columns() -> None:
    """alpha_beta_z_star_table augments the backend z* table with alpha/beta columns."""
    model = MKMModel(domain_radius_um=0.32, nucleus_radius_um=3.9, alpha0=0.172, beta0=0.0615)
    table = model.alpha_beta_z_star_table([6], [11.0, 100.0, 400.0])
    assert [row.ion_Z for row in table.rows] == [6, 6, 6]
    assert all(row.beta == pytest.approx(0.0615) for row in table.rows)
    for row in table.rows:
        assert row.alpha == pytest.approx(0.172 + 0.0615 * row.z_star_1D_Gy)
    # z* decreases with energy (Inaniwa Fig. 5 shape)
    rows_by_energy = sorted(table.rows, key=lambda row: row.energy_MeV_u)
    z_values = [row.z_star_1D_Gy for row in rows_by_energy]
    assert all(z_values[i] > z_values[i + 1] for i in range(len(z_values) - 1))
