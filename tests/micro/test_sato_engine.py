"""Acceptance tests for the native Sato-AMF engine.

These pin ``SatoLinealEnergyEngine`` (see ``docs/references/Microdosimetry.md``,
"Sato AMF engine"). The comparison anchors are the bundled digitized Hartzell 2025
AMF curves (``pyradbio.benchmark.hartzell2025``), which the native engine must
reproduce.
"""

from __future__ import annotations

import numpy as np
import pytest

from pyradbio.benchmark.hartzell2025 import get_amf_curve
from pyradbio.core.radiation import Ion
from pyradbio.micro import SphericalSite
from pyradbio.micro.sato import SatoAmfCoefficients, SatoLinealEnergyEngine
from pyradbio.track.stopping_power import SimpleBetheStoppingPower


def _engine(site_radius_um: float) -> SatoLinealEnergyEngine:
    return SatoLinealEnergyEngine(
        site=SphericalSite(diameter_um=2 * site_radius_um),
        stopping_power=SimpleBetheStoppingPower(),
    )


def _hartzell_yd(ion_name: str, energy_MeV_u: float) -> float:
    curve = get_amf_curve(figure="fig4", ion=ion_name, method="AMF", energy_MeV_u=energy_MeV_u)
    y = np.asarray(curve.y)
    d = np.asarray(curve.yd) / y
    d = d / np.trapezoid(d, y)
    return float(np.trapezoid(y * d, y))


def test_carbon_spectrum_is_physical() -> None:
    """The engine returns a normalised spectrum with positive finite moments."""
    spec = _engine(0.3).single_event_spectrum(Ion(6, 12, 10.0))
    assert spec.y_F > 0.0 and np.isfinite(spec.y_F)
    assert spec.y_D >= spec.y_F  # dose-mean >= frequency-mean
    assert np.trapezoid(spec.f, spec.y_keV_um) == pytest.approx(1.0, rel=1e-2)


def test_carbon_peak_matches_hartzell_amf() -> None:
    """The y*d(y) peak position reproduces the digitized Hartzell 2025 AMF curve.

    Hartzell 2025 Fig. 4, 12C at 10 MeV/u, 0.3 um radius (doi:10.1088/1361-6560/adcfec).
    """
    ref = get_amf_curve(figure="fig4", ion="carbon", method="AMF", energy_MeV_u=10.0)
    spec = _engine(ref.site_radius_um).single_event_spectrum(Ion(6, 12, 10.0))
    yd = spec.y_keV_um * spec.d
    engine_peak_y = float(spec.y_keV_um[int(np.argmax(yd))])
    assert engine_peak_y == pytest.approx(ref.peak_y_keV_um, rel=0.25)


def test_delta_ray_tail_present() -> None:
    """Unlike the crosser-only Xapsos engine, the AMF has low-y delta-ray density.

    Assert non-negligible d(y) at least a decade below the y*d(y) peak.
    """
    spec = _engine(0.3).single_event_spectrum(Ion(6, 12, 10.0))
    yd = spec.y_keV_um * spec.d
    peak_y = float(spec.y_keV_um[int(np.argmax(yd))])
    low = spec.y_keV_um < 0.1 * peak_y
    assert low.any()
    assert float(np.trapezoid(spec.d[low], spec.y_keV_um[low])) > 1e-3


@pytest.mark.parametrize("energy", [1.0, 10.0, 100.0])
@pytest.mark.parametrize(("name", "Z", "A"), [("carbon", 6, 12), ("neon", 10, 20)])
def test_sato_engine_yd_tracks_hartzell_amf(name: str, Z: int, A: int, energy: float) -> None:
    """The auto-lookup engine reproduces the Hartzell AMF y_D for the heavy ions.

    The coefficients are fit to the bundled Hartzell 2025 Fig-4 curves (0.3 um
    radius) with a y_D-matching term, so the engine's dose-mean lineal energy
    tracks the reference across the therapy range (doi:10.1088/1361-6560/adcfec).
    """
    spec = _engine(0.3).single_event_spectrum(Ion(Z, A, energy))
    assert spec.y_D == pytest.approx(_hartzell_yd(name, energy), rel=0.2)


def test_sato_engine_matches_explicit_coefficients() -> None:
    """Passing coefficients explicitly bypasses the auto-lookup and still builds."""
    coeff = SatoAmfCoefficients(B=0.12, C=0.86, mu1=5.0, mu2=1e-3, j1=2.0, j2=3.1)
    spec = SatoLinealEnergyEngine(
        site=SphericalSite(0.6), stopping_power=SimpleBetheStoppingPower(), coefficients=coeff
    ).single_event_spectrum(Ion(6, 12, 10.0))
    assert spec.norm == pytest.approx(1.0, rel=1e-2)
    assert spec.y_D >= spec.y_F > 0.0
    assert spec.meta["coefficients"] is coeff


def test_sato_engine_lowers_mcf_mkm_carbon_rbe_vs_xapsos() -> None:
    """Payoff: as an MCF-MKM micro engine the Sato AMF lowers the (hot) V79-carbon
    RBE_alpha relative to the crosser-only Xapsos engine.

    The Xapsos crosser-only ``y_D`` runs hot for carbon; the Sato AMF tracks the
    Hartzell reference, so feeding it into ``MCFMKMModel`` reduces the RBE_alpha
    over-prediction (Parisi 2022 V79 config, r_d=0.26, yD_ref=4.3).
    """
    from pyradbio.micro.lineal_energy import AnalyticSpectrumEngine
    from pyradbio.models.mkm.spectrum_models import MCFMKMModel

    sp = SimpleBetheStoppingPower()
    r_d = 0.26
    site = SphericalSite(2 * r_d)
    sato = SatoLinealEnergyEngine(site=site, stopping_power=sp)
    xapsos = AnalyticSpectrumEngine(sp, site, restricted=True)

    def model(engine: object) -> MCFMKMModel:
        return MCFMKMModel(
            alpha_ref=0.19,
            beta_ref=0.019,
            nucleus_radius_um=4.0,
            domain_radius_um=r_d,
            yD_ref_keV_um=4.3,
            engine=engine,
        )

    for energy in (10.0, 30.0, 100.0):
        ion = Ion(6, 12, energy)
        assert model(sato).alpha_beta(ion).alpha < model(xapsos).alpha_beta(ion).alpha


def test_sato_engine_guards_site_and_uncovered_ions() -> None:
    """Non-spherical sites are rejected; ions outside the Hartzell set raise."""
    from pyradbio.micro import CylindricalSite

    sp = SimpleBetheStoppingPower()
    coeff = SatoAmfCoefficients(B=0.1, C=0.8, mu1=0.01, mu2=0.001, j1=2.0, j2=10.0)
    with pytest.raises(ValueError):
        SatoLinealEnergyEngine(
            site=CylindricalSite(1.0, 1.0, seed=0), stopping_power=sp, coefficients=coeff
        ).single_event_spectrum(Ion(6, 12, 10.0))
    # Auto-lookup covers only the six Hartzell ions; boron (Z=5) is not fit.
    with pytest.raises(NotImplementedError):
        SatoLinealEnergyEngine(site=SphericalSite(0.6), stopping_power=sp).single_event_spectrum(
            Ion(5, 11, 10.0)
        )
