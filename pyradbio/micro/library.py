"""Loaders for external lineal-energy spectrum libraries (CSV).

Reads binned single-event lineal-energy spectra ``N(y)`` from CSV files into
[`TabulatedLinealEnergySpectrum`][] objects,
so measured / Monte-Carlo libraries can be compared against the analytic engine
through the shared [`Spectrum`][] moments.

Supported format -- DeCunha et al. 2025 proton library (Med. Phys. 52(5),
3471-3480, doi:10.1002/mp.17561; Geant4-DNA, liquid water, 1 um diameter target
sphere):

* column 1: the lower edge of each lineal-energy bin [keV/um], plus a final row
  giving the upper edge of the last bin (so there are ``n + 1`` rows for ``n``
  bins);
* column 2: the unnormalised ``N(y)`` count spectrum (``n`` values);
* column 3: a single value on the first row -- the number of counts in the
  spectrum per primary particle.

File names encode the projectile and energy, e.g. ``Proton_10.500000_MeV.csv``.
The library folder name encodes the surrounding scoring cube (``1mm`` = a dose
voxel, ``5um`` ~ a TEPC / cell-culture thickness); the microdosimetric target
itself is a 1 um diameter sphere in both cases.
"""

from __future__ import annotations

import re
from collections.abc import Iterable
from dataclasses import dataclass
from pathlib import Path
from typing import Any

import numpy as np
import numpy.typing as npt

from pyradbio.core.radiation import Ion
from pyradbio.micro.lineal_energy import TabulatedLinealEnergySpectrum

# Projectile name -> (Z, A). Extend as further species are added to a library.
_SPECIES_ZA: dict[str, tuple[int, int]] = {
    "proton": (1, 1),
    "helium": (2, 4),
    "carbon": (6, 12),
    "oxygen": (8, 16),
}

_FILENAME_RE = re.compile(r"^(?P<species>[A-Za-z]+)_(?P<energy>[0-9]+(?:\.[0-9]+)?)_MeV$")

# Provenance shared by every spectrum loaded from the DeCunha 2025 library.
_DECUNHA2025_META: dict[str, Any] = {
    "source": "DeCunha et al. 2025, Med. Phys. 52(5), 3471-3480",
    "doi": "10.1002/mp.17561",
    "code": "Geant4-DNA",
    "material": "liquid water",
    "target_diameter_um": 1.0,  # 1 um diameter target sphere (paper sec. 2)
}


def _parse_filename(stem: str) -> tuple[str, float]:
    match = _FILENAME_RE.match(stem)
    if match is None:
        raise ValueError(
            f"unrecognised spectrum file name {stem!r}; expected '<Species>_<energy>_MeV'"
        )
    return match["species"].lower(), float(match["energy"])


def load_decunha_lineal_energy_csv(
    path: str | Path, *, environment: str | None = None
) -> TabulatedLinealEnergySpectrum:
    """Load one DeCunha-2025-format lineal-energy CSV file.

    ``environment`` labels the surrounding scoring geometry (e.g. ``"1mm"`` or
    ``"5um"``); when omitted it is inferred from the parent folder name.
    """
    path = Path(path)
    species, energy_MeV = _parse_filename(path.stem)

    # Columns 0 (bin edges) and 1 (counts); the final row has no count value, so
    # genfromtxt yields NaN there and the counts are the leading n entries.
    table = np.genfromtxt(path, delimiter=",", usecols=(0, 1), dtype=np.float64)
    edges = np.ascontiguousarray(table[:, 0], dtype=np.float64)
    counts = np.ascontiguousarray(table[:-1, 1], dtype=np.float64)
    if np.any(np.isnan(counts)):
        raise ValueError(f"missing count value(s) in {path.name}")

    counts_per_primary = _read_counts_per_primary(path)
    if environment is None:
        environment = path.parent.name.split("_", 1)[0]

    za = _SPECIES_ZA.get(species)
    meta: dict[str, Any] = {
        **_DECUNHA2025_META,
        "species": species,
        "energy_MeV": energy_MeV,
        "ion": Ion(Z=za[0], A=za[1], energy_MeV_u=energy_MeV / za[1]) if za else None,
        "counts_per_primary": counts_per_primary,
        "environment": environment,
        "path": str(path),
    }
    return TabulatedLinealEnergySpectrum(y_edges_keV_um=edges, counts=counts, meta=meta)


def _read_counts_per_primary(path: Path) -> float | None:
    """Read the per-primary count value in column 3 of the first data row."""
    with path.open() as handle:
        for line in handle:
            fields = line.strip().split(",")
            if fields and fields[0]:
                return float(fields[2]) if len(fields) > 2 and fields[2] else None
    return None


@dataclass(frozen=True, slots=True)
class DeCunhaLinealEnergyLibrary:
    """An indexed folder of DeCunha-2025 per-energy lineal-energy CSV files.

    Indexes file names once; individual spectra are read on demand via
    [`load`][] (exact energy) or [`nearest`][] (closest available energy).
    Each spectrum is a source-agnostic
    [`TabulatedLinealEnergySpectrum`][].
    """

    directory: Path
    environment: str
    _paths_by_energy: dict[float, Path]

    @classmethod
    def from_directory(
        cls, directory: str | Path, *, environment: str | None = None
    ) -> DeCunhaLinealEnergyLibrary:
        directory = Path(directory)
        if not directory.is_dir():
            raise NotADirectoryError(f"not a directory: {directory}")
        paths: dict[float, Path] = {}
        for path in directory.glob("*.csv"):
            try:
                _species, energy_MeV = _parse_filename(path.stem)
            except ValueError:
                continue
            paths[energy_MeV] = path
        if not paths:
            raise FileNotFoundError(f"no '<Species>_<energy>_MeV.csv' files in {directory}")
        env = environment if environment is not None else directory.name.split("_", 1)[0]
        return cls(directory=directory, environment=env, _paths_by_energy=paths)

    @property
    def energies_MeV(self) -> npt.NDArray[np.float64]:
        """Sorted array of available projectile energies [MeV]."""
        return np.array(sorted(self._paths_by_energy), dtype=np.float64)

    def load(self, energy_MeV: float, *, rtol: float = 1e-6) -> TabulatedLinealEnergySpectrum:
        """Load the spectrum at ``energy_MeV`` (must match an available file)."""
        for available, path in self._paths_by_energy.items():
            if abs(available - energy_MeV) <= rtol * max(available, energy_MeV):
                return load_decunha_lineal_energy_csv(path, environment=self.environment)
        raise KeyError(
            f"no spectrum at {energy_MeV} MeV in {self.directory}; "
            f"nearest available is {self._nearest_energy(energy_MeV)} MeV"
        )

    def nearest(self, energy_MeV: float) -> TabulatedLinealEnergySpectrum:
        """Load the spectrum whose energy is closest to ``energy_MeV``."""
        path = self._paths_by_energy[self._nearest_energy(energy_MeV)]
        return load_decunha_lineal_energy_csv(path, environment=self.environment)

    def load_many(self, energies_MeV: Iterable[float]) -> list[TabulatedLinealEnergySpectrum]:
        """Load the nearest-energy spectrum for each requested energy."""
        return [self.nearest(energy) for energy in energies_MeV]

    def _nearest_energy(self, energy_MeV: float) -> float:
        energies = self.energies_MeV
        return float(energies[int(np.argmin(np.abs(energies - energy_MeV)))])
