"""z*_1D tables, delegated to MKMModel's backend (NativeMKMBackend by default).

The in-memory :class:`ZStarTable` mirrors pyMKM's classic ``.txt`` payload:
per-fragment ``(energy, z*_1D)`` rows plus the MKM header parameters
(``CellType``, ``Alpha_0``, ``Beta``, domain/nucleus radii). It round-trips
through :func:`pyradbio.tables.io.read_pymkm_txt` /
:func:`pyradbio.tables.io.write_pymkm_txt`.
"""

from __future__ import annotations

from collections.abc import Sequence
from dataclasses import dataclass
from typing import TYPE_CHECKING

if TYPE_CHECKING:
    from pyradbio.models.mkm.base import MKMBackend

#: Column order of the per-fragment data rows in the pyMKM classic .txt format.
ZSTAR_COLUMNS: tuple[str, ...] = ("Z", "energy_MeV_u", "z_star_1D_Gy")


@dataclass(frozen=True, slots=True)
class ZStarRow:
    """One ``(ion, energy) -> z*_1D`` entry.

    ``energy_MeV_u`` is in MeV/u; ``z_star_1D_Gy`` is the saturation-corrected
    dose-mean specific energy of the domain (``z_bar_star_domain`` in pyMKM), Gy.
    """

    Z: int
    energy_MeV_u: float
    z_star_1D_Gy: float


@dataclass(frozen=True, slots=True)
class ZStarTable:
    """A z*_1D table with the MKM header parameters, per pyMKM's classic format."""

    rows: tuple[ZStarRow, ...]
    cell_type: str
    alpha0_Gy: float
    beta_Gy2: float
    domain_radius_um: float
    nucleus_radius_um: float


def build_zstar_table(
    ions: Sequence[int],
    energies_MeV_u: Sequence[float],
    *,
    domain_radius_um: float,
    nucleus_radius_um: float,
    alpha0_Gy: float,
    beta0_Gy2: float,
    cell_type: str = "cell",
    backend: MKMBackend | None = None,
) -> ZStarTable:
    """Build a z*_1D table over an (ion Z, energy) grid via the MKM backend.

    Delegates z* to :class:`~pyradbio.models.mkm.model.MKMModel`, whose default
    engine is the pyMKM-free
    :class:`~pyradbio.models.mkm.native_backend.NativeMKMBackend` built from the
    given geometry. Inject ``backend`` to use another engine (e.g.
    ``PyMKMBackend`` for a pyMKM cross-check) or a fake; its geometry must then
    agree with the arguments. The result carries the MKM header
    parameters and round-trips through :func:`pyradbio.tables.io.write_pymkm_txt`.
    """
    from pyradbio.models.mkm.model import MKMModel

    model = MKMModel(
        domain_radius_um=domain_radius_um,
        nucleus_radius_um=nucleus_radius_um,
        alpha0=alpha0_Gy,
        beta0=beta0_Gy2,
        backend=backend,
    )
    ab = model.alpha_beta_z_star_table(ions, energies_MeV_u)
    rows = tuple(
        ZStarRow(Z=row.ion_Z, energy_MeV_u=row.energy_MeV_u, z_star_1D_Gy=row.z_star_1D_Gy)
        for row in ab.rows
    )
    return ZStarTable(
        rows=rows,
        cell_type=cell_type,
        alpha0_Gy=alpha0_Gy,
        beta_Gy2=beta0_Gy2,
        domain_radius_um=domain_radius_um,
        nucleus_radius_um=nucleus_radius_um,
    )
