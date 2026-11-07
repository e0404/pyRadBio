"""Pluggable MKM computation backend.

The MKM *biophysics* -- ``alpha = alpha0 + beta * z*_1D`` (Inaniwa 2010) -- is
deliberately separated from the *engine* that computes the saturation-corrected
dose-mean specific energy ``z*_1D``. pyRadBio ships the pyMKM-free default
engine (:class:`~pyradbio.models.mkm.native_backend.NativeMKMBackend`) and a
pyMKM-backed one for validation
(:class:`~pyradbio.models.mkm.pymkm_backend.PyMKMBackend`, optional ``mkm``
extra), but **any** object satisfying :class:`MKMBackend` can be handed to
:class:`~pyradbio.models.mkm.model.MKMModel` without changing the model or its
callers.

A backend owns the microdosimetry/track-structure computation and is configured
with the domain geometry it needs (domain radius, nucleus radius, saturation
parameters). It returns ``z*_1D`` in Gy; the model turns that into LQ
parameters.
"""

from __future__ import annotations

from collections.abc import Sequence
from dataclasses import dataclass
from typing import Protocol, runtime_checkable

from pyradbio.core.radiation import Ion


@dataclass(frozen=True, slots=True)
class BackendZStarRow:
    """One backend-computed z*_1D table row.

    This intentionally lives in the MKM layer rather than in pandas so the MKM
    layer remains usable without the ``tables`` extra; ``pyradbio.tables`` adapts
    these rows to DataFrames/Parquet.
    """

    ion_Z: int
    energy_MeV_u: float
    let_keV_um: float | None
    z_star_1D_Gy: float


@dataclass(frozen=True, slots=True)
class BackendZStarTable:
    """Small dependency-free table of MKM backend z*_1D rows."""

    rows: tuple[BackendZStarRow, ...]

    def __init__(self, rows: Sequence[BackendZStarRow]) -> None:
        object.__setattr__(self, "rows", tuple(rows))


@dataclass(frozen=True, slots=True)
class MKMAlphaBetaTableRow:
    """One MKM alpha/beta table row, augmented from a z*_1D row."""

    ion_Z: int
    energy_MeV_u: float
    let_keV_um: float | None
    z_star_1D_Gy: float
    alpha: float
    beta: float


@dataclass(frozen=True, slots=True)
class MKMAlphaBetaTable:
    """Dependency-free MKM alpha/beta table returned by ``MKMModel``."""

    rows: tuple[MKMAlphaBetaTableRow, ...]

    def __init__(self, rows: Sequence[MKMAlphaBetaTableRow]) -> None:
        object.__setattr__(self, "rows", tuple(rows))


@runtime_checkable
class MKMBackend(Protocol):
    """Engine contract for computing MKM specific-energy quantities.

    Implementations are self-contained: geometry and saturation parameters are
    fixed at construction so these methods depend only on the radiation field.
    """

    def z_star_1D_Gy(self, ion: Ion) -> float:
        """Saturation-corrected dose-mean specific energy ``z*_1D`` (Gy) for one ion.

        See Inaniwa 2010, eqs. (5)-(8): this is the quantity that enters
        ``alpha = alpha0 + beta * z*_1D``.
        """
        ...

    def z_star_1D_table_Gy(
        self, ions: Sequence[int], energies_MeV_u: Sequence[float]
    ) -> BackendZStarTable:
        """``z*_1D`` over an (ion Z, energy) grid as a ``BackendZStarTable`` of typed rows."""
        ...

    def let_keV_um(self, ion: Ion) -> float:
        """Unrestricted LET [keV/um] the engine uses for this ion.

        :class:`~pyradbio.models.mkm.model.MKMModel` needs it to turn
        fluence-specified mixed-field components into dose weights, and it must be
        the engine's own LET so the dose weights and ``z*_1D`` agree.
        """
        ...
