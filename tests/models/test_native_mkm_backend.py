"""Native (pyMKM-free) MKM backend: Kiefer-Chatterjee impact-parameter z*_1D."""

from __future__ import annotations

import numpy as np
import pytest

from pyradbio.core.radiation import Ion
from pyradbio.micro import SphericalSite
from pyradbio.micro.lineal_energy import AnalyticSpectrumEngine
from pyradbio.models.mkm import MKMBackend, MKMModel, NativeMKMBackend
from pyradbio.track.stopping_power import SimpleBetheStoppingPower
from tests.conftest import requires_pymkm


def test_native_backend_satisfies_protocol() -> None:
    assert isinstance(NativeMKMBackend(), MKMBackend)


def test_native_zstar_positive_and_monotonic_with_energy() -> None:
    """z*_1D falls with energy over 11->400 MeV/u (Inaniwa 2010 Fig. 5 shape)."""
    backend = NativeMKMBackend()  # default analytic impact-parameter method
    z_low = backend.z_star_1D_Gy(Ion(6, 12, 11.0))
    z_mid = backend.z_star_1D_Gy(Ion(6, 12, 100.0))
    z_high = backend.z_star_1D_Gy(Ion(6, 12, 400.0))
    assert z_low > z_mid > z_high > 0.0


def test_native_zstar_table_rows() -> None:
    backend = NativeMKMBackend()
    table = backend.z_star_1D_table_Gy([1, 6], [11.0, 100.0])
    assert {(r.ion_Z, r.energy_MeV_u) for r in table.rows} == {
        (1, 11.0),
        (1, 100.0),
        (6, 11.0),
        (6, 100.0),
    }
    assert all(r.z_star_1D_Gy > 0.0 and r.let_keV_um and r.let_keV_um > 0.0 for r in table.rows)


def test_default_backend_handles_high_energy_light_ions() -> None:
    """Regression: the analytic method does not fail to sample (the MC engine did).

    High-energy helium/oxygen crashed the delta-ray Monte-Carlo path ("no
    energy-deposition events sampled"); the deterministic impact-parameter method
    must return a finite positive z*_1D.
    """
    backend = NativeMKMBackend()
    for ion in (Ion(2, 4, 400.0), Ion(8, 16, 400.0)):
        z = backend.z_star_1D_Gy(ion)
        assert np.isfinite(z) and z > 0.0


def test_mkm_model_runs_pymkm_free_with_native_backend() -> None:
    """MKMModel produces alpha = alpha0 + beta0*z*_1D through the native seam."""
    backend = NativeMKMBackend()
    model = MKMModel(alpha0=0.172, beta0=0.0615, backend=backend)
    ion = Ion(6, 12, 11.0)
    lq = model.alpha_beta(ion)
    assert lq.beta == pytest.approx(0.0615)
    assert lq.alpha == pytest.approx(0.172 + 0.0615 * backend.z_star_1D_Gy(ion))


def test_injected_spectrum_engine_path() -> None:
    """An injected SpectrumEngine switches to the spectrum-based z*_1D path."""
    sp = SimpleBetheStoppingPower()
    engine = AnalyticSpectrumEngine(stopping_power=sp, site=SphericalSite(0.64), restricted=True)
    backend = NativeMKMBackend(engine=engine)
    assert backend.z_star_1D_Gy(Ion(6, 12, 11.0)) > 0.0


@requires_pymkm
def test_native_zstar_tracks_pymkm_across_energies() -> None:
    """Native impact-parameter z*_1D tracks pyMKM tightly across the whole range.

    With the correct MKM domain-disk geometry (pyMKM's cylinder-perpendicular-to-track),
    the matched saturation parameter and square-root dose-average, the only residual
    is the amorphous track model -- a few percent for carbon over 3-400 MeV/u.
    """
    from pyradbio.models.mkm.pymkm_backend import PyMKMBackend

    native = NativeMKMBackend()
    pymkm = PyMKMBackend()
    errors = []
    for energy in (3.0, 11.0, 30.0, 100.0, 400.0):
        ion = Ion(6, 12, energy)
        zn, zp = native.z_star_1D_Gy(ion), pymkm.z_star_1D_Gy(ion)
        errors.append(abs(zn - zp) / zp)
        assert zn == pytest.approx(zp, rel=0.10)  # each point within 10%
    assert float(np.median(errors)) < 0.05


@requires_pymkm
def test_sphere_geometry_overshoots_pymkm() -> None:
    """The (wrong) spherical domain overshoots z*_1D vs the disk that matches pyMKM.

    Documents that the geometry -- not the Kiefer-Chatterjee track -- drives the
    high-energy discrepancy: at 100 MeV/u carbon the sphere is >~20% high.
    """
    from pyradbio.micro import specific_energy_impact_profile
    from pyradbio.models.mkm.pymkm_backend import PyMKMBackend
    from pyradbio.models.mkm.spectrum_models import saturation_y0_keV_um, specific_energy_Gy
    from pyradbio.track.kiefer_chatterjee import KieferChatterjeeRadialDose
    from pyradbio.track.stopping_power import SimpleBetheStoppingPower

    r_d, r_n, beta0 = 0.32, 3.9, 0.0615
    ion = Ion(6, 12, 100.0)
    z0 = float(
        specific_energy_Gy(
            saturation_y0_keV_um(domain_radius_um=r_d, nucleus_radius_um=r_n, beta_Gy2=beta0), r_d
        )
    )
    rd = KieferChatterjeeRadialDose(SimpleBetheStoppingPower())
    sphere = specific_energy_impact_profile(
        rd, ion, region_radius_um=r_d, geometry="sphere"
    ).saturation_corrected_dose_mean_Gy(z0)
    disk = specific_energy_impact_profile(
        rd, ion, region_radius_um=r_d, geometry="cylinder"
    ).saturation_corrected_dose_mean_Gy(z0)
    z_pymkm = PyMKMBackend().z_star_1D_Gy(ion)
    assert disk == pytest.approx(z_pymkm, rel=0.10)
    assert sphere > 1.2 * z_pymkm


@requires_pymkm
def test_stopping_power_choice_barely_moves_zstar() -> None:
    """Using pyMKM's own mstar table (vs the Bethe default) barely moves z*_1D.

    SimpleBethe already matches the mstar LET closely, so the two native z*_1D
    agree within ~5% -- the stopping power was never the pyMKM discrepancy (the
    geometry was; see test_sphere_geometry_overshoots_pymkm).
    """
    from pyradbio.models.mkm.pymkm_backend import PyMKMStoppingPower
    from pyradbio.track.kiefer_chatterjee import KieferChatterjeeRadialDose

    mstar = PyMKMStoppingPower("mstar_3_12")
    default_backend = NativeMKMBackend()
    mstar_backend = NativeMKMBackend(
        stopping_power=mstar, radial_dose=KieferChatterjeeRadialDose(mstar)
    )
    for energy in (11.0, 100.0, 400.0):
        ion = Ion(6, 12, energy)
        assert mstar_backend.z_star_1D_Gy(ion) == pytest.approx(
            default_backend.z_star_1D_Gy(ion), rel=0.05
        )


@requires_pymkm
def test_pymkm_backend_reports_let_in_keV_um() -> None:
    """pyMKM tabulates LET in MeV/cm; the backend must hand it out in keV/um.

    Reference: the same mstar_3_12 table read through ``PyMKMStoppingPower``,
    which converts it, and analytic Bethe (18.15 keV/um for He at 10 MeV/u).
    A backend passing MeV/cm through reports these values 10x too high.
    """
    from pyradbio.models.mkm.pymkm_backend import PyMKMBackend, PyMKMStoppingPower

    backend = PyMKMBackend()
    mstar = PyMKMStoppingPower("mstar_3_12")
    ions = [Ion(2, 4, 10.0), Ion(6, 12, 100.0)]
    for ion in ions:
        assert backend.let_keV_um(ion) == pytest.approx(mstar.let_keV_um(ion), rel=1e-6)
        assert backend.let_keV_um(ion) == pytest.approx(
            SimpleBetheStoppingPower().let_keV_um(ion), rel=0.05
        )
    rows = backend.z_star_1D_table_Gy([2, 6], [10.0, 100.0]).rows
    for row in rows:
        ion = Ion(row.ion_Z, 4 if row.ion_Z == 2 else 12, row.energy_MeV_u)
        assert row.let_keV_um == pytest.approx(mstar.let_keV_um(ion), rel=1e-6)


@requires_pymkm
def test_pymkm_backend_converts_fluence_to_the_right_dose() -> None:
    """A fluence-specified component gets its true dose, not 10x it.

    ``MKMModel`` uses the backend's LET to weight fluence-specified components
    against dose-specified ones in a mixed field, so a unit slip there skews alpha.
    """
    from pyradbio.core.radiation import RadiationField, SpectrumComponent, fluence_to_dose
    from pyradbio.models.mkm.pymkm_backend import PyMKMBackend, PyMKMStoppingPower

    ion = Ion(6, 12, 100.0)
    field = RadiationField([SpectrumComponent(ion, fluence=1e7)])
    expected = fluence_to_dose(1e7, PyMKMStoppingPower("mstar_3_12").let_keV_um(ion))
    assert field.total_dose_Gy(PyMKMBackend().let_keV_um) == pytest.approx(expected, rel=1e-6)
