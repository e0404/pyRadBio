"""``Spectrum.yd`` and ``Spectrum.peak_y_keV_um`` (the dose-weighted plotting view).

``yd = y d1(y)`` is how every lineal-energy script draws a spectrum, and its
maximum is the "primary peak" the Hartzell 2025 comparison tabulates. Both were
re-derived inline in the scripts before they became properties.
"""

from __future__ import annotations

import numpy as np
import pytest
from scipy.integrate import simpson

from pyradbio.core.radiation import Ion
from pyradbio.micro import LinealEnergyCalculator, SphericalSite
from pyradbio.micro.hartzell2025 import Hartzell2025Curve, iter_amf_curves
from pyradbio.micro.lineal_energy import Spectrum, TabulatedLinealEnergySpectrum
from pyradbio.track.stopping_power import SimpleBetheStoppingPower


def _analytic_carbon() -> Spectrum:
    calc = LinealEnergyCalculator(SimpleBetheStoppingPower(), SphericalSite(diameter_um=1.0))
    return calc.spectrum(Ion(6, 12, 100.0))


def test_yd_is_y_times_d_bit_for_bit() -> None:
    """The same expression the scripts used (``spec.y_keV_um * spec.d``)."""
    spec = _analytic_carbon()
    np.testing.assert_array_equal(spec.yd, spec.y_keV_um * spec.d)
    assert spec.yd.shape == spec.y_keV_um.shape


def test_yd_has_unit_area_on_a_log_axis() -> None:
    """``integral y d(y) d(ln y) = integral d(y) dy = 1``: equal area = equal dose."""
    spec = _analytic_carbon()
    area = simpson(spec.yd, x=np.log(spec.y_keV_um))
    assert area == pytest.approx(1.0, rel=2e-3)


def test_peak_is_the_grid_maximum_of_yd() -> None:
    spec = _analytic_carbon()
    peak = spec.peak_y_keV_um
    assert isinstance(peak, float)
    assert peak in set(spec.y_keV_um.tolist())
    assert spec.yd[spec.y_keV_um == peak][0] == spec.yd.max()
    # A crosser-dominated carbon spectrum peaks near its LET.
    let = SimpleBetheStoppingPower().let_keV_um(Ion(6, 12, 100.0))
    assert 0.5 * let < peak < 2.0 * let


def test_peak_of_a_known_shape() -> None:
    """A log-normal ``y d(y)`` centred on 10 keV/um peaks at the grid point nearest 10."""
    y = np.geomspace(0.1, 1.0e3, 401)  # 10 keV/um is exactly node 200
    yd_shape = np.exp(-0.5 * (np.log(y / 10.0) / 0.4) ** 2)
    spec = Spectrum(y_keV_um=y, f=yd_shape / (y * y))  # f ~ yd / y^2 up to a constant
    assert spec.peak_y_keV_um == pytest.approx(10.0, rel=1e-12)


def test_ties_take_the_first_maximum() -> None:
    """Same rule as ``Hartzell2025Curve.peak_y_keV_um`` (``max`` over the indices)."""
    y = np.array([1.0, 2.0, 4.0, 8.0])
    spec = Spectrum(y_keV_um=y, f=np.array([1.0, 0.25, 0.0625, 0.0]) / y)
    # y*d is proportional to y*f*y = [1, 1, 1, 0] -> a three-way tie, first wins.
    assert spec.peak_y_keV_um == 1.0


@pytest.mark.parametrize("curve", iter_amf_curves(figure="fig4")[:6], ids=lambda c: c.ion)
def test_peak_rule_matches_the_digitized_curves(curve: Hartzell2025Curve) -> None:
    """A Spectrum built from a digitized ``y d(y)`` reports that curve's peak."""
    y = np.asarray(curve.y, dtype=float)
    yd_values = np.asarray(curve.yd, dtype=float)
    spec = Spectrum(y_keV_um=y, f=yd_values / (y * y))
    assert spec.peak_y_keV_um == curve.peak_y_keV_um


def test_tabulated_spectrum_uses_its_count_moments() -> None:
    """``to_spectrum`` carries the exact count ``y_F``, which ``yd`` normalises by."""
    edges = np.geomspace(0.5, 50.0, 21)
    centres = np.sqrt(edges[:-1] * edges[1:])
    counts = np.exp(-0.5 * (np.log(centres / 5.0) / 0.5) ** 2)
    tab = TabulatedLinealEnergySpectrum(y_edges_keV_um=edges, counts=counts)
    spec = tab.to_spectrum()
    np.testing.assert_array_equal(spec.yd, spec.y_keV_um * (spec.y_keV_um * spec.f / tab.y_F))
    assert spec.peak_y_keV_um == float(centres[np.argmax(centres * centres * spec.f)])
