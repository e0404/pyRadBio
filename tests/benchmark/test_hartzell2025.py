"""Hartzell et al. 2025 digitized AMF / nBio benchmark curves."""

from __future__ import annotations

import numpy as np
import pytest

from pyradbio.benchmark import hartzell2025


def test_load_counts_and_split() -> None:
    """The bundle is Figure 4 (6 ions H-Ne x 3 E x 2 methods) plus Figure 7."""
    curves = hartzell2025.load_amf_curves()
    assert len(curves) == 46
    fig4 = hartzell2025.iter_amf_curves(figure="fig4")
    fig7 = hartzell2025.iter_amf_curves(figure="fig7")
    assert len(fig4) == 6 * 3 * 2
    assert len(fig7) == 5 * 2
    assert {c.method for c in curves} == {"AMF", "nBio"}
    assert {c.ion for c in fig4} == {"hydrogen", "helium", "lithium", "carbon", "oxygen", "neon"}


def test_curve_fields_and_units() -> None:
    """A single curve exposes provenance, arrays, and the ion it describes."""
    c = hartzell2025.get_amf_curve(figure="fig4", ion="carbon", method="AMF", energy_MeV_u=1.0)
    assert c.doi == "10.1088/1361-6560/adcfec"
    assert c.site_radius_um == pytest.approx(0.3)
    assert (c.Z, c.A) == (6, 12)
    assert len(c.y) == len(c.yd) >= 10
    assert np.all(np.diff(c.y) > 0.0)  # sorted ascending in lineal energy
    assert min(c.yd) >= 0.0
    ion = c.as_ion()
    assert (ion.Z, ion.A) == (6, 12)
    assert ion.energy_MeV_u == pytest.approx(1.0)


def test_get_amf_curve_is_unique_or_raises() -> None:
    """Under-/over-specified selectors raise; a full selector returns one curve."""
    with pytest.raises(LookupError):
        # figure 4 has three energies for carbon AMF -> ambiguous without energy
        hartzell2025.get_amf_curve(figure="fig4", ion="carbon", method="AMF")
    one = hartzell2025.get_amf_curve(figure="fig7", ion="carbon", method="nBio", site_radius_um=0.3)
    assert one.figure == "fig7"


def test_peak_shifts_down_with_energy() -> None:
    """Higher energy -> lower LET -> the primary peak moves to lower lineal energy."""
    peaks = [
        hartzell2025.get_amf_curve(
            figure="fig4", ion="carbon", method="AMF", energy_MeV_u=E
        ).peak_y_keV_um
        for E in (1.0, 10.0, 100.0)
    ]
    assert peaks[0] > peaks[1] > peaks[2]


def test_peak_grows_with_atomic_number() -> None:
    """At fixed 1 MeV/u the peak lineal energy rises from hydrogen to neon (LET order)."""
    order = ["hydrogen", "helium", "lithium", "carbon", "oxygen", "neon"]
    peaks = [
        hartzell2025.get_amf_curve(
            figure="fig4", ion=ion, method="AMF", energy_MeV_u=1.0
        ).peak_y_keV_um
        for ion in order
    ]
    assert peaks == sorted(peaks)
