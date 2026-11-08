"""Scholz-Kraft amorphous 1/r^2 radial dose, parameterised per LEM version.

Common form (Scholz & Kraft 1996 eq. 9; Elsasser & Scholz 2007 eq. 6):
    D(r) = lambda*LET/r_min^2      r < r_min
         = lambda*LET/r^2          r_min <= r <= r_max
         = 0                       r > r_max
with r_max = gamma * E^delta (delta=1.7; gamma=0.05 for LEM I and 0.062 for
LEM II/III) and lambda fixed so the radial integral reproduces LET (energy conservation,
``KEVUM_PER_GY_UM2_GCM3 * rho * integral(D*2*pi*r dr) = LET``).

LEM variants:
  - LEM I  : r_min ~ 10 nm, no radical-diffusion convolution.
  - LEM II : r_min = 0.3 nm + 2-D Gaussian radical-diffusion convolution,
             sigma = 4 nm, via the modified-Bessel I0 form (Elsasser 2007 eq.12).
  - LEM III: velocity-dependent core r_min(beta) = beta * r_c, r_c = 40 nm
             (Elsasser et al. 2008 eq. 6).
"""

from __future__ import annotations

import math
from dataclasses import dataclass

import numpy as np
import numpy.typing as npt
from scipy.integrate import simpson, trapezoid
from scipy.special import i0e

from pyradbio.core.constants import KEVUM_PER_GY_UM2_GCM3, WATER_DENSITY_G_CM3
from pyradbio.core.radiation import Ion, relativistic_beta
from pyradbio.track.stopping_power import StoppingPower

# um * (MeV/u)^-delta. LEM I: gamma=0.05 (Scholz & Kraft 1996 eq. 10, "gamma=0.05;
# delta=1.7"). LEM II onward: gamma=0.062 (Elsasser & Scholz 2007, after eq. 6).
LEM1_R_MAX_GAMMA_UM: float = 0.05
R_MAX_GAMMA_UM: float = 0.062
R_MAX_DELTA: float = 1.7
LEM2_SIGMA_RADICAL_NM: float = 4.0
LEM2_RMIN_NM: float = 0.3
LEM3_RC_NM: float = 40.0

_NM_TO_UM: float = 1.0e-3
# Nodes for the radial-diffusion convolution integral (log-spaced in r').
_CONV_NODES: int = 512
# Gaussian widths past r_max at which the convolved profile is treated as zero
# (see ``ScholzKraftRadialDose.r_support_um``).
_SUPPORT_SIGMA: float = 8.0


@dataclass(frozen=True, slots=True)
class ScholzKraftRadialDose:
    r_min_nm: float = 10.0  # LEM I default
    radical_diffusion: bool = False  # LEM II/III
    velocity_dependent_core: bool = False  # LEM III: r_min(beta)=beta*r_c
    stopping_power: StoppingPower | None = None
    density_g_cm3: float = WATER_DENSITY_G_CM3
    sigma_nm: float = LEM2_SIGMA_RADICAL_NM
    r_c_nm: float = LEM3_RC_NM
    r_max_gamma_um: float = R_MAX_GAMMA_UM

    # ---- LEM-version constructors --------------------------------------
    @classmethod
    def lem1(cls, stopping_power: StoppingPower | None = None) -> ScholzKraftRadialDose:
        """LEM I: ``r_min ~ 10 nm`` core, no radical diffusion."""
        return cls(
            r_min_nm=10.0,
            radical_diffusion=False,
            stopping_power=stopping_power,
            r_max_gamma_um=LEM1_R_MAX_GAMMA_UM,
        )

    @classmethod
    def lem2(cls, stopping_power: StoppingPower | None = None) -> ScholzKraftRadialDose:
        """LEM II: ``r_min = 0.3 nm`` core + 2-D Gaussian radical diffusion."""
        return cls(r_min_nm=LEM2_RMIN_NM, radical_diffusion=True, stopping_power=stopping_power)

    @classmethod
    def lem3(cls, stopping_power: StoppingPower | None = None) -> ScholzKraftRadialDose:
        """LEM III: velocity-dependent core ``r_min = beta*r_c`` + radical diffusion."""
        return cls(
            radical_diffusion=True,
            velocity_dependent_core=True,
            stopping_power=stopping_power,
        )

    # ---- geometry -------------------------------------------------------
    def r_max_um(self, ion: Ion) -> float:
        """Maximum delta-electron range ``r_max = gamma * E^delta`` [um]."""
        return float(self.r_max_gamma_um * ion.energy_MeV_u**R_MAX_DELTA)

    def r_min_um(self, ion: Ion) -> float:
        """Inner core radius [um]; velocity-dependent for LEM III."""
        if self.velocity_dependent_core:
            return relativistic_beta(ion.energy_MeV_u) * self.r_c_nm * _NM_TO_UM
        return self.r_min_nm * _NM_TO_UM

    def r_support_um(self, ion: Ion) -> float:
        """Outer support of the profile [um] -- the radius past which ``D(r)`` is 0.

        This is ``r_max`` for an unconvolved track, but **not** for LEM II/III/IV:
        radical diffusion transports dose *outward* past the delta-electron range,
        so the convolved profile extends to ``r_max + 8 sigma``. Consumers must
        evaluate and integrate over this radius, not ``r_max`` -- that is the
        difference between conserving the track's energy and discarding it.

        The discarded tail is negligible whenever ``r_max >> sigma``, which is why
        truncating at ``r_max`` passes unnoticed at therapeutic energies (2.6e-5 of
        ``r_max``; see [`AnalyticRadialProfile`][pyradbio.models.lem.base.AnalyticRadialProfile]).
        It stops being negligible when the track is only a few nm wide. For carbon,
        [`radial_integral_LET_keV_um`][] recovers 99.5% of the LET at 1 MeV/u but
        **4.6%** at 0.1 MeV/u -- and 100.00% at every energy once the domain runs
        to here. 8 sigma is where that integral converges; 6 sigma is also the
        half-width of the convolution's own peak-refinement grid.
        """
        r_max = self.r_max_um(ion)
        if not self.radical_diffusion:
            return r_max
        return r_max + _SUPPORT_SIGMA * self.sigma_nm * _NM_TO_UM

    # ---- normalisation --------------------------------------------------
    def _lambda(self, ion: Ion) -> float:
        """Normalisation so the energy integral reproduces LET (see module doc).

        ``integral(D*2*pi*r dr) = lambda*LET*pi*(1 + 2*ln(r_max/r_min))`` must
        equal ``LET / (KEVUM_PER_GY_UM2_GCM3 * rho)``.
        """
        r_min = self.r_min_um(ion)
        r_max = self.r_max_um(ion)
        if r_max <= r_min:
            raise ValueError(
                f"r_max ({r_max:.3g} um) <= r_min ({r_min:.3g} um) at "
                f"E={ion.energy_MeV_u} MeV/u; energy too low for this track model"
            )
        shape = math.pi * (1.0 + 2.0 * math.log(r_max / r_min))
        return 1.0 / (KEVUM_PER_GY_UM2_GCM3 * self.density_g_cm3 * shape)

    def _let(self, ion: Ion) -> float:
        if self.stopping_power is None:
            raise ValueError("a StoppingPower must be provided to compute local dose")
        return self.stopping_power.let_keV_um(ion)

    # ---- local dose -----------------------------------------------------
    def _base_dose_Gy(self, r_um: float, ion: Ion, let_keV_um: float) -> float:
        """Unconvolved Scholz-Kraft profile at radius ``r_um``."""
        r_min = self.r_min_um(ion)
        r_max = self.r_max_um(ion)
        if r_um > r_max:
            return 0.0
        r_eff = max(r_um, r_min)
        return self._lambda(ion) * let_keV_um / (r_eff * r_eff)

    def _convolution_nodes(
        self, r_um: float, r_floor: float, r_max: float, sigma: float
    ) -> npt.NDArray[np.float64]:
        """Integration nodes for the convolution: a log grid over the whole
        profile (core + 1/r^2 tail) merged with a fine linear grid spanning a
        few sigma around the kernel peak ``r' ~ r`` (which is far narrower than
        the log spacing at large r and would otherwise be under-sampled).
        """
        log_grid = np.logspace(
            math.log10(r_floor), math.log10(r_max), _CONV_NODES, dtype=np.float64
        )
        peak_lo = max(r_floor, r_um - 6.0 * sigma)
        peak_hi = min(r_max, r_um + 6.0 * sigma)
        if peak_hi <= peak_lo:
            # ``r_um`` is more than 6 sigma outside the source domain, so the peak
            # grid would run *backwards* from beyond ``r_max`` -- np.unique sorts it
            # and the nodes land outside the source. Harmless in value (the kernel
            # is e^-18 there) but it is exactly the sort of thing that stops being
            # harmless later; the numba kernel already guards this the same way.
            return log_grid
        peak_grid = np.linspace(peak_lo, peak_hi, 129)
        nodes: npt.NDArray[np.float64] = np.unique(np.concatenate([log_grid, peak_grid]))
        return nodes

    def _convolved_dose_Gy(self, r_um: float, ion: Ion, let_keV_um: float) -> float:
        """Radial dose smeared by 2-D Gaussian radical diffusion (Bessel-I0 form).

        ``D_c(r) = integral_0^rmax D(r') * (r'/sigma^2) *
                   exp(-(r-r')^2/(2 sigma^2)) * I0(r r'/sigma^2) dr'``
        using the exponentially scaled ``i0e`` for numerical stability.

        **There is no elementary closed form, but there are two usable series**,
        and they are complementary. Writing ``lambda = r^2/sigma^2``, the I0 form
        says ``|r e + sigma Z|^2/sigma^2`` is a noncentral chi-square with 2 dof,
        so ``D_c`` is one of its inverse moments:

        ```text
        D_c(r) = D_core [1 - Q_1(r/sigma, r_min/sigma)]
               + (k/2 sigma^2) e^-lambda/2 sum_j (lambda/2)^j / (j!)^2
                                             [Gamma(j,u_min) - Gamma(j,u_max)]
        ```

        with ``u = r'^2/(2 sigma^2)``, ``Q_1`` the Marcum Q, and the ``j = 0``
        term ``Gamma(0,u) = E_1(u)`` carrying the logarithmic divergence of
        ``int dr'/r'`` -- which the core ``r_min > 0`` is what regularises.
        Verified against this method to 2e-5 over ``r = 1..100 nm``. It needs
        ``~lambda/2`` terms, so it overflows by ``r ~ 150 nm`` and is only usable
        near the core.

        Far from it, the small-sigma (heat-kernel) expansion is elementary. In
        2-D ``grad^2 r^-m = m^2 r^-m-2``, so ``grad^2n r^-2 = 4^n (n!)^2
        r^-2-2n`` and

        ```text
        D_c(r) = (k/r^2) sum_n n! x^n,      x = 2 sigma^2 / r^2
        ```

        asymptotic, with optimal truncation at ``n ~ 1/x``. Two terms hold to
        ~1e-5 from ``r = 50 nm`` outward and ~3e-6 beyond 100 nm -- at the
        numerical floor of the quadrature below. That is what
        [`AnalyticRadialProfile`][pyradbio.models.lem.base.AnalyticRadialProfile] evaluates, and
        it is why the LEM II/III Monte Carlo does not have to call this method
        per sample. It also explains the observed switch radii: ``2 sigma^2/r^2 =
        1e-3`` gives 179 nm and ``1e-4`` gives 566 nm, against 175 and 550
        measured, and both are independent of ion and energy because only
        ``sigma`` enters.
        """
        sigma = self.sigma_nm * _NM_TO_UM
        r_min = self.r_min_um(ion)
        r_max = self.r_max_um(ion)
        lam_let = self._lambda(ion) * let_keV_um
        r_floor = min(r_min, sigma) * 1.0e-3
        rp = self._convolution_nodes(r_um, r_floor, r_max, sigma)
        base = lam_let / np.maximum(rp, r_min) ** 2
        s2 = sigma * sigma
        kernel = (rp / s2) * np.exp(-((r_um - rp) ** 2) / (2.0 * s2)) * i0e(r_um * rp / s2)
        # trapezoid, NOT simpson: ``rp`` merges a log grid with a linear grid a few
        # sigma wide, so neighbouring intervals differ by orders of magnitude (44 um
        # against 0.05 nm at r ~ 1000 um). Non-uniform Simpson fits a parabola across
        # such a pair and the result is catastrophic cancellation -- at r = 1642 um it
        # returned -8.9e-9 Gy where the true value is +6.39e-9. Trapezoid is accurate
        # to 5e-5 there. Same trap, same fix, as ``LEMBase._impact_log_kill``.
        return float(trapezoid(base * kernel, x=rp))

    def local_dose_Gy(self, r_um: float, ion: Ion) -> float:
        """Local dose D(r) at radius r [um] from the ion trajectory [Gy]."""
        if r_um < 0.0:
            raise ValueError("r_um must be >= 0")
        let = self._let(ion)
        if self.radical_diffusion:
            return self._convolved_dose_Gy(r_um, ion, let)
        return self._base_dose_Gy(r_um, ion, let)

    def local_doses_Gy(self, r_um: npt.ArrayLike, ion: Ion) -> npt.NDArray[np.float64]:
        """Local dose ``D(r)`` [Gy] at every radius in ``r_um`` [um]; same shape.

        A loop over the scalar [`local_dose_Gy`][] (``float`` of each element), so
        every value is bit-identical to the scalar call -- the radical-diffusion
        convolution is a per-radius quadrature and is not vectorised here. Replaces
        the ``np.array([model.local_dose_Gy(float(r), ion) for r in radii])`` loops of
        the Elsasser & Scholz 2007 (doi:10.1667/RR0467.1) Fig. 4/5 reconstruction.

        Raises:
            ValueError: if any radius is negative (from [`local_dose_Gy`][]).
        """
        radii = np.asarray(r_um, dtype=np.float64)
        doses = np.array([self.local_dose_Gy(float(r), ion) for r in radii.ravel()], dtype=float)
        return doses.reshape(radii.shape)

    # ---- validation helper ---------------------------------------------
    def radial_integral_LET_keV_um(self, ion: Ion, *, nodes: int = 4096) -> float:
        """Numerically integrate the energy profile back to a LET [keV/um].

        Returns ``KEVUM_PER_GY_UM2_GCM3 * rho * integral(D(r)*2*pi*r dr)``, which
        by construction must reproduce ``stopping_power.let_keV_um(ion)``.

        The domain runs to [`r_support_um`][], not ``r_max``: with radical
        diffusion those differ, and integrating to ``r_max`` returns 4.6% of the
        LET for a 0.1 MeV/u carbon ion.
        """
        r_top = self.r_support_um(ion)
        r_min = self.r_min_um(ion)
        r = np.logspace(math.log10(r_min * 1.0e-3), math.log10(r_top), nodes)
        dose = np.array([self.local_dose_Gy(float(ri), ion) for ri in r])
        integral = float(simpson(dose * 2.0 * math.pi * r, x=r))
        return KEVUM_PER_GY_UM2_GCM3 * self.density_g_cm3 * integral
