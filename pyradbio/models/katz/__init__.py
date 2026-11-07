"""Katz cellular track-structure theory (Katz et al. 1971, doi:10.2307/3573247).

The oldest model in this package and the ancestor of the LEM and MKM families:
cell killing follows from the delta-ray dose around an ion's path, and a cell is
summarised by four fitted parameters (``sigma_0``, ``E_0``, ``kappa``, ``m``).

Only the **grain-count regime** is implemented -- the closed form of eq. (8),
which is the regime the published fits were made in. The track-width regime
needs eq. (4)'s numerically tabulated site dose (Butts & Katz 1967) and is out
of scope; see ``docs/references/Katz1971.md``.
"""

from pyradbio.models.katz.model import (
    ERG_CM3_PER_GY,
    UM2_PER_CM2,
    KatzCellularModel,
    KatzKillTerms,
)

__all__ = ["ERG_CM3_PER_GY", "UM2_PER_CM2", "KatzCellularModel", "KatzKillTerms"]
