"""Hartzell et al. 2025 AMF / TOPAS-nBio spectra (benchmark-layer alias).

The digitized curves and their loader live in [`pyradbio.micro.hartzell2025`][],
because the micro layer fits its Sato-AMF coefficients to them
([`pyradbio.micro.sato_coefficients`][]) and must not import the benchmark
layer. This module re-exports that API unchanged for benchmark code.

Hartzell, Parisi, Sato, Beltran & Furutani 2025, Phys. Med. Biol. 70 105010,
doi:10.1088/1361-6560/adcfec.
"""

from __future__ import annotations

from pyradbio.micro.hartzell2025 import (
    HARTZELL2025_DOI,
    Hartzell2025Curve,
    get_amf_curve,
    iter_amf_curves,
    load_amf_curves,
)

__all__ = [
    "HARTZELL2025_DOI",
    "Hartzell2025Curve",
    "get_amf_curve",
    "iter_amf_curves",
    "load_amf_curves",
]
