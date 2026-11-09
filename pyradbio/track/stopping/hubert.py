"""Heavy-ion stopping by Hubert, Bimbot & Gauvin's (1990) alpha scaling, for any Z >= 3.

F. Hubert, R. Bimbot and H. Gauvin, At. Data Nucl. Data Tables 46, 1 (1990), eq. (1):

    S(Z, v) = (gamma Z / 2)^2 S_alpha(v)

with their effective-charge fraction ``gamma`` (eqs. 2-9,
[`HubertEffectiveCharge`][]) and the alpha
stopping power at the same velocity. Their alphas came from Ziegler's fits; ours from
[`ICRU90Procedure`][], i.e. ICRU 90's alpha
table (``alpha`` is an option). Hubert et al. fitted ``gamma`` to ~600 measurements, O to U
in solids at 3-90 MeV/u, and tabulate 2.5-500 MeV/u; the lower limit 2.5 MeV/u (where the
alpha reference is fully stripped) is this model's energy range start. Above 500 MeV/u
``gamma`` is 1 to < 1e-5 and the model is ``(Z/2)^2 S_alpha``: the ion's own Z-dependent
higher-order terms (Mott, Lindhard-Sorensen) are then the alpha's, an open limitation for
very heavy ions at GeV/u.

Water is outside Hubert's solids; oxygen's ``Z2 = 8`` is used (ASSUMPTION). Measured
for Z 3-18 in water against MSTAR / the ICRU 73 errata, worst per band at
2.5-10 / 10-50 / 50-300 MeV/u: Li -2.5/-1.2/-1.0%, C -2.3/-1.9/-1.2%, O +3.4/-1.9/-1.3%,
Ne -1.9/-2.0/-1.3%, Si +7.2/-1.4/-1.4%, Ar +6.6/-1.5/-1.5% vs MSTAR; within 4.7% of ICRU
73 at all of them; ICRU 90 carbon 1.2% over 2.5-300 MeV/u. For Z > 18 no water reference
is held; Hubert et al. report ~5% agreement with Ar, Kr, Xe, Pb and U data in solids.

It is the default stopping power's branch for Z > 18
([`default_stopping_power`][pyradbio.track.stopping_power.default_stopping_power]).
"""

from __future__ import annotations

from typing import ClassVar

import numpy as np
import numpy.typing as npt

from pyradbio.core.constants import WATER_DENSITY_G_CM3
from pyradbio.track.stopping.base import IonSourceInfo, MassStoppingPowerModel, OptionValue
from pyradbio.track.stopping.effective_charge import HUBERT_1990, HubertEffectiveCharge
from pyradbio.track.stopping.masses import nuclear_mass_MeV

HUBERT_MIN_ENERGY_MEV_U = 2.5
"""Hubert et al.'s lower tabulation limit, where helium is fully stripped [MeV/u]."""
HUBERT_MAX_Z = 92
"""Heaviest ion with shipped isotope masses."""


class HubertAlphaScaling(MassStoppingPowerModel):
    """``S = (gamma Z/2)^2 S_alpha`` at equal velocity, Z 3-92 (module docstring)."""

    name: ClassVar[str] = "hubert1990"

    def __init__(
        self,
        *,
        alpha: MassStoppingPowerModel | None = None,
        effective_charge: HubertEffectiveCharge | None = None,
        min_energy_MeV_u: float = HUBERT_MIN_ENERGY_MEV_U,
        density_g_cm3: float = WATER_DENSITY_G_CM3,
    ) -> None:
        if alpha is None:
            from pyradbio.track.stopping.icru90_procedure import ICRU90Procedure

            alpha = ICRU90Procedure(density_g_cm3=density_g_cm3)
        if not min_energy_MeV_u > 0.0:
            raise ValueError("min_energy_MeV_u must be > 0")
        self.alpha = alpha
        self.effective_charge = effective_charge or HubertEffectiveCharge()
        self.min_energy_MeV_u = float(min_energy_MeV_u)
        self.density_g_cm3 = density_g_cm3

    def supported_Z(self) -> tuple[int, ...]:
        return tuple(range(3, HUBERT_MAX_Z + 1))

    @staticmethod
    def _gamma(Z: int, A: int, energy: npt.NDArray[np.float64]) -> npt.NDArray[np.float64]:
        out: npt.NDArray[np.float64] = 1.0 + energy * A / nuclear_mass_MeV(Z, A)
        return out

    def energy_range(self, Z: int, A: int) -> tuple[float, float]:
        """``min_energy_MeV_u`` and the alpha source's range, both at equal velocity."""
        self._require_supported(Z)
        alpha_lo, alpha_hi = self.alpha.energy_range(2, 4)

        def at_equal_velocity(e_alpha: float) -> float:
            gamma = 1.0 + e_alpha * 4 / nuclear_mass_MeV(2, 4)
            return (gamma - 1.0) * nuclear_mass_MeV(Z, A) / A

        return max(self.min_energy_MeV_u, at_equal_velocity(alpha_lo)), at_equal_velocity(alpha_hi)

    def _evaluate(
        self, Z: int, A: int, energy_MeV_u: npt.NDArray[np.float64]
    ) -> npt.NDArray[np.float64]:
        gamma = self._gamma(Z, A, energy_MeV_u)
        beta = np.sqrt(1.0 - 1.0 / (gamma * gamma))
        e_alpha = (gamma - 1.0) * nuclear_mass_MeV(2, 4) / 4.0
        # energy_range() maps the alpha range at equal velocity, so in-range ion energies
        # land inside it up to the rounding of the round trip; clip only that.
        lo, hi = self.alpha.energy_range(2, 4)
        if e_alpha.min() < lo * (1 - 1e-12) or e_alpha.max() > hi * (1 + 1e-12):  # pragma: no cover
            raise AssertionError("alpha energy outside the alpha source's range")
        s_alpha = self.alpha.mass_stopping_power(2, 4, np.clip(e_alpha, lo, hi))
        out: npt.NDArray[np.float64] = (self.effective_charge.z_eff(Z, beta) / 2.0) ** 2 * s_alpha
        return out

    def metadata(self, Z: int, A: int) -> IonSourceInfo:
        lo, hi = self.energy_range(Z, A)
        alpha_info = self.alpha.metadata(2, 4)
        options: dict[str, OptionValue] = {
            "quantity": "electronic",
            "effective_charge": self.effective_charge.name,
            "alpha_source": alpha_info.source,
        }
        return IonSourceInfo(
            Z=Z,
            A=A,
            source="Hubert alpha scaling (pyRadBio)",
            reference=f"{HUBERT_1990}, eqs. (1)-(9)",
            i_value_eV=alpha_info.i_value_eV,
            energy_range_MeV_u=(lo, hi),
            edition="Hubert 1990",
            options=options,
        )


__all__ = ["HUBERT_MAX_Z", "HUBERT_MIN_ENERGY_MEV_U", "HubertAlphaScaling"]
