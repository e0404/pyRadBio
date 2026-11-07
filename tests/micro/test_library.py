"""Loaders for external lineal-energy spectrum libraries (CSV)."""

from __future__ import annotations

from pathlib import Path

import numpy as np
import pytest

from pyradbio.micro import (
    DeCunhaLinealEnergyLibrary,
    TabulatedLinealEnergySpectrum,
    load_decunha_lineal_energy_csv,
)

FIXTURES = Path(__file__).parent / "fixtures" / "1mm_synthetic"
FILE_10 = FIXTURES / "Proton_10.500000_MeV.csv"


def test_load_csv_shapes_and_metadata() -> None:
    """A DeCunha-2025-format file parses into edges, counts and provenance."""
    spec = load_decunha_lineal_energy_csv(FILE_10)
    assert spec.y_edges_keV_um.shape == (7,)
    assert spec.counts.shape == (6,)
    assert spec.counts.tolist() == [5.0, 50.0, 200.0, 120.0, 40.0, 8.0]
    assert spec.y_edges_keV_um[0] == 0.1
    assert spec.y_edges_keV_um[-1] == 100.0
    # Column 3, first row: counts per primary particle.
    assert spec.meta["counts_per_primary"] == pytest.approx(931.591)
    assert spec.meta["energy_MeV"] == pytest.approx(10.5)
    assert spec.meta["species"] == "proton"
    assert spec.meta["environment"] == "1mm"
    assert spec.meta["doi"] == "10.1002/mp.17561"


def test_load_csv_maps_projectile_to_ion() -> None:
    spec = load_decunha_lineal_energy_csv(FILE_10)
    ion = spec.meta["ion"]
    assert (ion.Z, ion.A) == (1, 1)
    assert ion.energy_MeV_u == pytest.approx(10.5)  # A = 1 -> MeV/u == MeV


def test_moments_match_direct_histogram_estimate() -> None:
    """to_spectrum() moments equal the count-weighted geometric-centre estimate."""
    spec = load_decunha_lineal_energy_csv(FILE_10)
    centres = np.sqrt(spec.y_edges_keV_um[:-1] * spec.y_edges_keV_um[1:])
    counts = spec.counts
    yF_ref = float((counts * centres).sum() / counts.sum())
    yD_ref = float((counts * centres**2).sum() / (counts * centres).sum())
    assert spec.y_F == pytest.approx(yF_ref, rel=1e-3)
    assert spec.y_D == pytest.approx(yD_ref, rel=1e-3)
    # The frequency density integrates to 1 (sum of f_i * dy_i) and y_D > y_F.
    density = spec.frequency_density()
    assert float((density * spec.bin_widths_keV_um).sum()) == pytest.approx(1.0)
    assert spec.y_D > spec.y_F


def test_to_spectrum_dose_weight_nonnegative_on_coarse_histogram() -> None:
    """Coarse, non-uniform bins must not drive the dose-weighted spectrum negative.

    Regression: ``Spectrum.d`` normalises by ``y_F``. When ``y_F`` was computed by
    Simpson quadrature on the geometric bin centres, the non-uniform-spacing
    weights of a coarse log-binned histogram made ``integral f dy`` (and thus
    ``y_F``) negative, flipping ``d`` below zero. ``to_spectrum`` carries the
    exact count-weighted ``y_F``/``y_D`` so ``d`` stays positive.
    """
    spec = load_decunha_lineal_energy_csv(FILE_10).to_spectrum()
    assert spec.y_F > 0.0
    assert np.all(spec.d >= 0.0)


def test_bad_filename_rejected(tmp_path: Path) -> None:
    bad = tmp_path / "not_a_spectrum.csv"
    bad.write_text("0.1,1\n1.0,\n")
    with pytest.raises(ValueError):
        load_decunha_lineal_energy_csv(bad)


def test_edges_counts_length_mismatch_rejected() -> None:
    with pytest.raises(ValueError):
        TabulatedLinealEnergySpectrum(
            y_edges_keV_um=np.array([1.0, 2.0, 3.0]),  # 3 edges
            counts=np.array([1.0]),  # needs 2 counts
        )


def test_library_indexes_and_loads() -> None:
    lib = DeCunhaLinealEnergyLibrary.from_directory(FIXTURES)
    assert lib.environment == "1mm"
    assert lib.energies_MeV.tolist() == [10.5, 100.5]

    exact = lib.load(10.5)
    assert exact.meta["energy_MeV"] == pytest.approx(10.5)
    assert exact.meta["environment"] == "1mm"

    # Lower energy (higher LET) has the larger dose-mean lineal energy.
    hi_let = lib.load(10.5)
    lo_let = lib.load(100.5)
    assert hi_let.y_D > lo_let.y_D


def test_library_nearest_and_missing() -> None:
    lib = DeCunhaLinealEnergyLibrary.from_directory(FIXTURES)
    assert lib.nearest(12.0).meta["energy_MeV"] == pytest.approx(10.5)
    assert lib.nearest(90.0).meta["energy_MeV"] == pytest.approx(100.5)
    with pytest.raises(KeyError):
        lib.load(55.0)  # no such file
    assert [s.meta["energy_MeV"] for s in lib.load_many([9.0, 200.0])] == [10.5, 100.5]


def test_library_empty_directory_rejected(tmp_path: Path) -> None:
    with pytest.raises(FileNotFoundError):
        DeCunhaLinealEnergyLibrary.from_directory(tmp_path)
