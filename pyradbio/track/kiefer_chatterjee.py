"""Kiefer-Chatterjee amorphous radial-dose model.

The Kiefer-Chatterjee track structure (Chatterjee & Schaefer 1976,
*Radiat. Environ. Biophys.* 13, 215, doi:10.1007/BF01330766; Kiefer & Straaten
1986, *Phys. Med. Biol.* 31, 1201, doi:10.1088/0031-9155/31/11/002) is the
amorphous-track model underlying the MKM/SMK z* computation. It splits the
radial dose into a constant "core" and a ``1/r^2`` "penumbra" of delta-ray
dose:

```text
D(r) = (LET_E / (pi*rho*Rc^2)) - 2*Kp*ln(Rp/Rc)/Rc^2   r < Rc   (core)
     = Kp / r^2                                          Rc <= r < Rp (penumbra)
     = 0                                                 r >= Rp
```

with
  - ``Rc = Rc0 * beta``      velocity-dependent core radius (Rc0 = 11.6 nm),
  - ``Rp = gamma * E^delta`` penumbra (delta-ray) radius (gamma = 0.0616
    um*(MeV/u)^-delta, delta = 1.7), floored at ``Rc``,
  - ``Kp = kappa * (Zeff/beta)^2`` penumbra amplitude [Gy*um^2], with
    ``Zeff`` the Barkas effective charge and ``kappa = 1.25e-4 Gy*um^2``,
  - ``LET_E`` the energy deposited per unit length.

The core amplitude is fixed by *energy conservation*: the core term subtracts
exactly the penumbra's radially-integrated energy so that
``integral(D(r)*2*pi*r dr)`` reproduces the LET (see
`radial_integral_LET_keV_um`). Expressed in pyRadBio units this is

    D_core = LET[keV/um] / (pi * KEVUM_PER_GY_UM2_GCM3 * rho * Rc^2)
             - 2*Kp*ln(Rp/Rc)/Rc^2 .

Numerical constants (``Rc0``, ``gamma``, ``kappa``) match the reference
implementation in **pyMKM** (``pymkm.physics.particle_track.ParticleTrack``,
model ``"Kiefer-Chatterjee"``), against which
``tests/track/test_kiefer_chatterjee.py`` validates this module. pyMKM is the
MKM backend; this standalone port lets the LEM/track layer use the
Kiefer-Chatterjee profile without importing pyMKM.
"""

from __future__ import annotations

import math
from dataclasses import dataclass

import numpy as np
from scipy.integrate import simpson

from pyradbio.core.constants import KEVUM_PER_GY_UM2_GCM3, WATER_DENSITY_G_CM3
from pyradbio.core.radiation import Ion
from pyradbio.track.stopping_power import StoppingPower, default_stopping_power, effective_charge

# Core radius Rc = KC_RC0_NM * beta [nm] (pyMKM ParticleTrack: Rc0 = 11.6e-3 um).
KC_RC0_NM: float = 11.6
# Penumbra radius Rp = gamma * E^delta [um] (pyMKM KC: gamma = 0.0616, delta = 1.7).
KC_R_MAX_GAMMA_UM: float = 0.0616
KC_R_MAX_DELTA: float = 1.7
# Penumbra amplitude Kp = kappa * (Zeff/beta)^2 [Gy*um^2] (pyMKM KC: 1.25e-4).
KC_KP_KAPPA_GY_UM2: float = 1.25e-4

_NM_TO_UM: float = 1.0e-3


@dataclass(frozen=True, slots=True)
class KieferChatterjeeRadialDose:
    """Kiefer-Chatterjee radial dose ``D(r)`` for an ion track in water.

    Implements the [`RadialDoseModel`][pyradbio.track.base.RadialDoseModel] protocol.
    Parameters default to the pyMKM ``"Kiefer-Chatterjee"`` values; a
    [`StoppingPower`][] supplies the LET, by default (``stopping_power=None``)
    [`default_stopping_power`][pyradbio.track.stopping_power.default_stopping_power] at
    ``density_g_cm3``.
    """

    stopping_power: StoppingPower | None = None
    density_g_cm3: float = WATER_DENSITY_G_CM3
    rc0_nm: float = KC_RC0_NM
    r_max_gamma_um: float = KC_R_MAX_GAMMA_UM
    kp_kappa_Gy_um2: float = KC_KP_KAPPA_GY_UM2

    # ---- geometry -------------------------------------------------------
    def r_min_um(self, ion: Ion) -> float:
        """Core radius ``Rc = Rc0 * beta`` [um] (velocity-dependent)."""
        return self.rc0_nm * _NM_TO_UM * ion.beta()

    def r_max_um(self, ion: Ion) -> float:
        """Penumbra radius ``Rp = max(gamma * E^delta, Rc)`` [um].

        The maximum delta-electron range; floored at the core radius ``Rc`` so
        very slow ions (whose energetic range drops below the core) still have a
        well-defined track (pyMKM ``_compute_penumbra_radius``).
        """
        rp_energy = self.r_max_gamma_um * ion.energy_MeV_u**KC_R_MAX_DELTA
        return float(max(rp_energy, self.r_min_um(ion)))

    def r_support_um(self, ion: Ion) -> float:
        """The profile is cut hard at ``Rp``, so the support is ``r_max_um``.

        Kiefer-Chatterjee deposits its dose where it lands -- there is no
        diffusion step to carry any past the penumbra radius.
        """
        return self.r_max_um(ion)

    # ---- track parameters ----------------------------------------------
    def _kp_Gy_um2(self, ion: Ion) -> float:
        """Penumbra amplitude ``Kp = kappa * (Zeff/beta)^2`` [Gy*um^2].

        ``Zeff`` is the Barkas effective charge (electron pick-up at low
        velocity); at high velocity ``Zeff -> Z``.
        """
        beta = ion.beta()
        z_eff = effective_charge(ion.Z, beta)
        return self.kp_kappa_Gy_um2 * (z_eff / beta) ** 2

    def _let(self, ion: Ion) -> float:
        if self.stopping_power is None:
            return default_stopping_power(self.density_g_cm3).let_keV_um(ion)
        return self.stopping_power.let_keV_um(ion)

    def _core_dose_Gy(self, ion: Ion, let_keV_um: float) -> float:
        """Constant core dose ``r < Rc`` (fixed by energy conservation)."""
        r_c = self.r_min_um(ion)
        r_p = self.r_max_um(ion)
        kp = self._kp_Gy_um2(ion)
        let_term = let_keV_um / (math.pi * KEVUM_PER_GY_UM2_GCM3 * self.density_g_cm3 * r_c * r_c)
        penumbra_term = 2.0 * kp * math.log(r_p / r_c) / (r_c * r_c)
        return let_term - penumbra_term

    # ---- local dose -----------------------------------------------------
    def local_dose_Gy(self, r_um: float, ion: Ion) -> float:
        """Local dose ``D(r)`` at radius ``r`` [um] from the trajectory [Gy]."""
        if r_um < 0.0:
            raise ValueError("r_um must be >= 0")
        r_c = self.r_min_um(ion)
        r_p = self.r_max_um(ion)
        let = self._let(ion)
        if r_um >= r_p:
            return 0.0
        if r_um < r_c:
            return self._core_dose_Gy(ion, let)
        return self._kp_Gy_um2(ion) / (r_um * r_um)

    # ---- validation helper ---------------------------------------------
    def radial_integral_LET_keV_um(self, ion: Ion, *, nodes: int = 4096) -> float:
        """Numerically integrate the energy profile back to a LET [keV/um].

        Returns ``KEVUM_PER_GY_UM2_GCM3 * rho * integral(D(r)*2*pi*r dr)``,
        which by construction reproduces ``stopping_power.let_keV_um(ion)``.
        """
        r_c = self.r_min_um(ion)
        r_p = self.r_max_um(ion)
        # Integrate the constant core and the 1/r^2 penumbra on separate grids:
        # a single grid spanning the core->penumbra step would place all but one
        # node inside the flat core and mis-resolve the discontinuity.
        r_core = np.linspace(0.0, r_c, max(nodes // 4, 2))
        d_core = np.array([self.local_dose_Gy(float(ri), ion) for ri in r_core])
        core = float(simpson(d_core * 2.0 * math.pi * r_core, x=r_core))

        tail = 0.0
        if r_p > r_c:
            r_tail = np.logspace(math.log10(r_c), math.log10(r_p), nodes)
            d_tail = np.array([self.local_dose_Gy(float(ri), ion) for ri in r_tail])
            tail = float(simpson(d_tail * 2.0 * math.pi * r_tail, x=r_tail))

        return KEVUM_PER_GY_UM2_GCM3 * self.density_g_cm3 * (core + tail)
