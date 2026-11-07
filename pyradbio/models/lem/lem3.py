"""LEM III: velocity-dependent inner track radius.

Elsasser et al. 2008 (doi:10.1016/j.ijrobp.2008.02.037). LEM III *is* LEM II
(cluster extension + radical-diffusion radial dose) with one further change: the
inner track core is no longer a fixed ``r_min`` but grows with the projectile
velocity,

    r_min(beta) = beta * r_c,   beta = v/c,   r_c = 40 nm            (eq. 6)

(fit to the rat spinal-cord data; cf. Mozumder's 9.3 nm). Physically, at higher
specific energy the primary ionisation/excitation events spread further from the
trajectory, so the high-dose track centre widens. Because a wider core spreads
the same LET over a larger area, the local dose peak -- and hence the ion RBE --
drops fastest for the fast, low-LET ions of the entrance channel, which is
exactly where LEM I/II over-predicted the RBE (Elsasser 2008, Results/Fig. 1).

Everything else is inherited from :class:`~pyradbio.models.lem.lem2.LEM2`: the
high-dose cluster enhancement ``eta(D)`` on the photon tail and the mixed-field
LQ mixing. Only the radial-dose model differs (``ScholzKraftRadialDose.lem3``),
so this is a one-line override; the transition dose ``D_t`` is refit per cell
line (Elsasser 2008 Table 2).
"""

from __future__ import annotations

from dataclasses import dataclass

from pyradbio.models.lem.lem2 import LEM2
from pyradbio.track.base import RadialDoseModel
from pyradbio.track.scholz_kraft import ScholzKraftRadialDose


@dataclass(frozen=True, slots=True)
class LEM3(LEM2):
    """LEM III: LEM II plus the velocity-dependent track core ``r_min(beta)=beta*r_c``.

    Inherits LEM II's cluster-enhanced photon tail and its default
    :class:`~pyradbio.models.lem.lem2.ClusterModel`; overrides only the radial
    dose to use the velocity-dependent core (``r_c = 40 nm``, Elsasser 2008
    eq. 6). Supply the LEM III refit ``D_t`` on the photon parameters.
    """

    def _radial_dose_model(self) -> RadialDoseModel:
        return ScholzKraftRadialDose.lem3(stopping_power=self.stopping_power)
