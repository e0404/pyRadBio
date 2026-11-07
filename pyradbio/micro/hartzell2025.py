"""Digitized AMF / TOPAS-nBio microdosimetric spectra from Hartzell et al. 2025.

Source paper -- an analytical-microdosimetric-function (AMF) extension of TOPAS
benchmarked against TOPAS-nBio Geant4-DNA track-structure simulations:

Hartzell, Parisi, Sato, Beltran & Furutani 2025, Phys. Med. Biol. 70 105010,
doi:10.1088/1361-6560/adcfec.

The bundled curves are dose-weighted single-event lineal-energy spectra
``y d(y)`` versus lineal energy ``y`` in a spherical water site, digitized from
PyMuPDF-extracted raster figures in the user's local PDF. Two figures are
covered:

* **Figure 4** -- six monoenergetic ions (``1H 4He 7Li 12C 16O 20Ne``) at 1, 10
  and 100 MeV u^-1 in a **0.3 um radius** site, one curve per
  ``(ion, energy, method)`` where ``method`` is ``AMF`` or ``nBio``. (The paper
  also shows 40Ar and 56Fe; those are not bundled here.)
* **Figure 7** -- 1 MeV u^-1 carbon across five site radii
  (``0.0015-0.5 um``), one curve per ``(radius, method)``.

These are intended as visual / regression benchmark data, not exact tabulated
physics. The two methods are separated by curve colour (AMF blue, nBio green).
Figure 4 was digitized by hand (WebPlotDigitizer) from a high-resolution copy of
the figure, tracing each of the six curves per panel individually. Figure 7 was
digitized programmatically (two-colour raster split). The main peaks are
faithful; the faintest features (the light-ion 100 MeV/u dash-dot bumps) remain
the least certain.

Each curve lives in its own CSV under ``data/`` named
``hartzell2025_figure<N>_<ion>_<method>[_<energy>MeVu|_rD<radius>um].csv`` with a
four-line header::

    <metadata keys>   figure,ion,Z,A,method,energy_MeV_u,site_radius_um,doi,source_page,note
    <metadata values>
    lineal_energy,yd_y
    keV/um,dimensionless
    <y>,<yd(y)> data rows ...

Identity/provenance live in the metadata block; the filename is human-facing.
"""

from __future__ import annotations

import csv
from dataclasses import dataclass
from functools import lru_cache
from importlib import resources

import numpy as np
import numpy.typing as npt

from pyradbio.core.radiation import Ion

HARTZELL2025_DOI = "10.1088/1361-6560/adcfec"

_DATA_DIR = "data"
_CURVE_PREFIX = "hartzell2025_figure"


@dataclass(frozen=True, slots=True)
class Hartzell2025Curve:
    """One digitized dose-weighted lineal-energy spectrum (one CSV file).

    ``method`` is ``"AMF"`` (the analytical microdosimetric function) or
    ``"nBio"`` (the TOPAS-nBio track-structure reference). ``y`` is lineal
    energy [keV/um]; ``yd`` is the dose-weighted spectrum ``y d(y)``
    [dimensionless], directly comparable to ``spectrum.y_keV_um * spectrum.d``.
    """

    figure: str
    ion: str
    Z: int
    A: int
    method: str
    energy_MeV_u: float
    site_radius_um: float
    y: tuple[float, ...]
    yd: tuple[float, ...]
    source_page: int
    note: str
    doi: str = HARTZELL2025_DOI

    def as_ion(self) -> Ion:
        """Build the :class:`~pyradbio.core.radiation.Ion` this curve describes."""
        return Ion(Z=self.Z, A=self.A, energy_MeV_u=self.energy_MeV_u)

    def dose_spectrum(
        self, *, normalized: bool = False
    ) -> tuple[npt.NDArray[np.float64], npt.NDArray[np.float64]]:
        """``(y, d(y))``: the dose distribution, with the digitized leading ``y`` divided out.

        The CSV stores the dose-weighted ``y d(y)``, which is what the paper plots;
        the models (``alpha_beta_from_dose_spectrum``) and the dose-mean
        ``y_D = integral y d(y) dy`` want ``d(y)`` itself. ``normalized=True``
        also divides by ``np.trapezoid(d, y)`` -- unit area on the digitized grid,
        i.e. the normalisation every ``y_D`` reference in this package applies.
        """
        y = np.asarray(self.y, dtype=np.float64)
        d = np.asarray(self.yd, dtype=np.float64) / y
        if normalized:
            d = d / np.trapezoid(d, y)
        return y, d

    @property
    def peak_y_keV_um(self) -> float:
        """Lineal energy [keV/um] at the maximum of ``y d(y)`` (the primary peak)."""
        return self.y[max(range(len(self.yd)), key=self.yd.__getitem__)]


def _parse_curve(text: str) -> tuple[dict[str, str], list[tuple[float, float]]]:
    """Parse the four-line header + data rows of one curve file."""
    rows = list(csv.reader(text.splitlines()))
    meta = dict(zip(rows[0], rows[1], strict=True))
    points = sorted((float(r[0]), float(r[1])) for r in rows[4:] if r and r[0].strip())
    return meta, points


def _data_files(prefix: str) -> list[str]:
    """Return the text of every ``data/`` CSV whose name starts with ``prefix``."""
    data_dir = resources.files(__package__).joinpath(_DATA_DIR)
    return [
        entry.read_text(encoding="utf-8")
        for entry in data_dir.iterdir()
        if entry.name.startswith(prefix) and entry.name.endswith(".csv")
    ]


@lru_cache(maxsize=1)
def load_amf_curves() -> tuple[Hartzell2025Curve, ...]:
    """Return every digitized Hartzell 2025 AMF / nBio benchmark curve."""
    curves: list[Hartzell2025Curve] = []
    for text in _data_files(_CURVE_PREFIX):
        meta, points = _parse_curve(text)
        curves.append(
            Hartzell2025Curve(
                figure=meta["figure"],
                ion=meta["ion"],
                Z=int(meta["Z"]),
                A=int(meta["A"]),
                method=meta["method"],
                energy_MeV_u=float(meta["energy_MeV_u"]),
                site_radius_um=float(meta["site_radius_um"]),
                y=tuple(y for y, _ in points),
                yd=tuple(v for _, v in points),
                source_page=int(meta["source_page"]),
                note=meta["note"],
                doi=meta.get("doi", HARTZELL2025_DOI),
            )
        )
    return tuple(
        sorted(
            curves,
            key=lambda c: (c.figure, c.ion, c.method, c.energy_MeV_u, c.site_radius_um),
        )
    )


def iter_amf_curves(
    *,
    figure: str | None = None,
    ion: str | None = None,
    method: str | None = None,
    energy_MeV_u: float | None = None,
    site_radius_um: float | None = None,
) -> tuple[Hartzell2025Curve, ...]:
    """Filter digitized Hartzell 2025 curves by any combination of fields."""
    curves = load_amf_curves()
    return tuple(
        c
        for c in curves
        if (figure is None or c.figure == figure)
        and (ion is None or c.ion == ion)
        and (method is None or c.method == method)
        and (energy_MeV_u is None or c.energy_MeV_u == energy_MeV_u)
        and (site_radius_um is None or c.site_radius_um == site_radius_um)
    )


def get_amf_curve(
    *,
    figure: str,
    ion: str,
    method: str,
    energy_MeV_u: float | None = None,
    site_radius_um: float | None = None,
) -> Hartzell2025Curve:
    """Return exactly one Hartzell 2025 curve; raise ``LookupError`` otherwise."""
    matches = iter_amf_curves(
        figure=figure,
        ion=ion,
        method=method,
        energy_MeV_u=energy_MeV_u,
        site_radius_um=site_radius_um,
    )
    if len(matches) != 1:
        raise LookupError(f"expected exactly one Hartzell 2025 curve, got {len(matches)}")
    return matches[0]


__all__ = [
    "HARTZELL2025_DOI",
    "Hartzell2025Curve",
    "get_amf_curve",
    "iter_amf_curves",
    "load_amf_curves",
]
