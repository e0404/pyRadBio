"""Radial-dose model protocol."""

from __future__ import annotations

from typing import Protocol

from pyradbio.core.radiation import Ion


class RadialDoseModel(Protocol):
    def local_dose_Gy(self, r_um: float, ion: Ion) -> float:
        """Local dose D(r) at radius r from the ion trajectory."""
        ...

    def r_max_um(self, ion: Ion) -> float:
        """Maximum track radius from the most energetic delta-electrons."""
        ...

    def r_min_um(self, ion: Ion) -> float:
        """Inner core radius (LEM-version dependent; may vary with velocity)."""
        ...

    def r_support_um(self, ion: Ion) -> float:
        """Radius past which ``local_dose_Gy`` is zero -- the integration domain.

        Distinct from :meth:`r_max_um`: a model that transports dose outward
        after depositing it (LEM II/III/IV radical diffusion) has dose beyond the
        delta-electron range, and integrating only to ``r_max_um`` silently
        discards it. Models with a hard cut-off return ``r_max_um``. Consumers
        integrating ``D(r)`` must use this, never ``r_max_um``.
        """
        ...
