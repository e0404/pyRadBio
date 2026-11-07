"""Sato Analytical Microdosimetric Function (AMF) engine.

Makes the MKM family (modified-MKM / MCF-MKM / SMK) self-sufficient in
single-event lineal-energy spectra ``d(y)`` -- reproducing the PHITS ``[T-SED]``
spectra natively, without PHITS or any external Monte-Carlo. Sources, coverage
and test pins: ``docs/references/Microdosimetry.md``, "Sato AMF engine".

The **Sato AMF** (Sato, Watanabe & Niita 2006, doi:10.1093/rpd/ncl407, Eq 7; same
form in Sato 2012, doi:10.3109/09553002.2011.611216, Eq 1 -- the PHITS version)
is a fixed functional form in **energy imparted ``eps``** [keV] (Sato 2012 states
``eps*f1(eps)`` is the density and ``y = eps/l_bar``, ``z = eps/m``; the 2006
paper writes the same with the symbol ``z``):

    eps*f1(eps) = A*eps^2 / (exp{B[eps - C*(dE/dx)*phi_s]} + 1)         # primary peak
                + sum_{k=1,2} (mu_k*eps/w_i)/(j_k-1)*((j_k-1)/j_k)^(eps/w_i)  # delta rays
                + sum_{i=1..6} P_i*delta(eps - eps_pi)                  # excitation peaks
                + P7/(sqrt(2pi)*Gamma)*exp{-(eps - eps_p7)^2/(2*Gamma^2)}  # Auger peak

The functional form and its normalisation are implemented here
(:func:`amf_epsilon_density`, :func:`_build_amf_density`): ``A`` is derived from
``B``/``C`` (Sato 2006 Eq 8), ``w_i = 10.9 eV``, and the discrete/Auger peaks are
omitted (sub-keV; negligible for the MKM's dose-weighted moments).

The per-condition parameters ``B, C, mu_k, j_k`` are **fit outputs** -- Sato 2006
(p.4): *"the values of B and C are determined by the least-squares fitting"*. In
PHITS they live in ``tsed.dat`` (ships with PHITS; not in any paper); here they are
obtained by fitting the form to the bundled Hartzell 2025 digitized AMF curves
(``pyradbio.micro.hartzell2025``); the acceptance tests
(``tests/micro/test_sato_engine.py``) hold the fitted ``y_D`` for H-Ne at
1/10/100 MeV/u within 20% of Hartzell. :class:`SatoLinealEnergyEngine` builds a
:class:`Spectrum` from explicit :class:`SatoAmfCoefficients`;
:func:`pyradbio.micro.sato_coefficients.sato_coefficients_for` supplies them per
ion and energy.
"""

from __future__ import annotations

import math
from dataclasses import dataclass

import numpy as np
import numpy.typing as npt
from scipy.integrate import simpson

from pyradbio.core.constants import WATER_DENSITY_G_CM3
from pyradbio.core.radiation import Ion
from pyradbio.micro.lineal_energy import SiteGeometry, Spectrum, SphericalSite
from pyradbio.track.stopping_power import StoppingPower

#: Fixed Sato-2006 excitation/Auger peak energies [keV/um-equivalent eV] and the
#: energy per commonest ionisation. These are the ONLY closed-form pieces of the
#: fitted set; everything else (B, j_k, mu_k, P_i) is a per-condition fit output.
_PEAK_ENERGIES_EV = (10.9, 13.5, 17.0, 21.4, 26.3, 533.0)
_AUGER_PEAK_EV = 700.0
_AUGER_WIDTH_EV = 100.0
_W_ION_EV = 10.9  # average energy per commonest ionisation (Sato 2006)


@dataclass(frozen=True, slots=True)
class SatoAmfCoefficients:
    """The per-(ion, energy, site) Sato-AMF fit parameters (Sato 2006 Eq 7).

    ``B`` and ``C`` (the primary-peak width and the core-fraction / peak-position
    parameter), the two delta-ray moments ``j1``/``j2`` and weights ``mu1``/``mu2``
    are **least-squares fit outputs**: Sato 2006 (p.4) states *"the values of B and
    C are determined by the least-squares fitting of the Monte-Carlo results"*.
    ``C`` also carries the physical meaning of the core energy fraction (ratio of
    energy directly deposited by the primary to the total), so the normalisation
    ``A`` is *derived* from ``B``/``C`` via Sato 2006 Eq 8. In PHITS these live in
    ``tsed.dat``; here they are obtained by fitting the form to the bundled Hartzell
    2025 digitized AMF curves (record provenance; do not invent -- AGENTS.md ->
    "No silent physics").

    The discrete excitation / Auger peak heights ``P_1..P_7`` are omitted: they sit
    at sub-keV energy imparted (``z_pi = 10.9..533 eV``) and are negligible for the
    dose-weighted moments (``y_D``/``y*``) the MKM consumes (Sato 2006: they have
    "trivial influence on the estimation of z*f1(z)dz").
    """

    B: float
    C: float
    mu1: float
    mu2: float
    j1: float
    j2: float


def _primary_shape(
    eps_keV: npt.NDArray[np.float64], let_keV_um: float, site_diameter_um: float, C: float, B: float
) -> npt.NDArray[np.float64]:
    """Unnormalised primary-ion term ``eps^2 / (exp{B[eps - C*L*phi_s]} + 1)``.

    The Fermi-function-shaped core peak (Sato 2006 Eq 7, first term), in energy
    imparted ``eps`` [keV]; its roll-off sits near ``eps0 = C*L*phi_s`` (the core
    energy over a chord ``phi_s``).
    """
    arg = np.clip(B * (eps_keV - C * let_keV_um * site_diameter_um), -500.0, 500.0)
    shape: npt.NDArray[np.float64] = eps_keV * eps_keV / (np.exp(arg) + 1.0)
    return shape


def _primary_norm_A(let_keV_um: float, site_diameter_um: float, C: float, B: float) -> float:
    """Primary normalisation ``A = C / integral eps^2/(exp{B[eps-C*L*phi_s]}+1) deps``.

    Sato 2006 Eq 8: fixes the primary term to carry the core energy fraction ``C``.
    """
    eps0 = C * let_keV_um * site_diameter_um
    hi = max(10.0 * eps0, 1.0)
    grid = np.logspace(np.log10(hi) - 8.0, np.log10(hi), 4000, dtype=np.float64)
    integral = float(simpson(_primary_shape(grid, let_keV_um, site_diameter_um, C, B), x=grid))
    return C / integral if integral > 0.0 else 0.0


def _delta_shape(eps_keV: npt.NDArray[np.float64], mu: float, j: float) -> npt.NDArray[np.float64]:
    """One delta-ray term ``(mu*eps/w_i)/(j-1) * ((j-1)/j)^(eps/w_i)`` (Sato 2006 Eq 7)."""
    x = eps_keV / (_W_ION_EV * 1.0e-3)  # eps/w_i, w_i in keV
    term: npt.NDArray[np.float64] = (mu * x) / (j - 1.0) * ((j - 1.0) / j) ** x
    return term


def amf_epsilon_density(
    eps_keV: npt.NDArray[np.float64],
    let_keV_um: float,
    site_diameter_um: float,
    coeff: SatoAmfCoefficients,
) -> npt.NDArray[np.float64]:
    """Sato AMF ``eps*f1(eps)`` [arb.] on ``eps_keV`` (primary + two delta terms).

    Sato 2006 Eq 7 / Sato 2012 Eq 1 (the discrete/Auger peaks are omitted -- see
    :class:`SatoAmfCoefficients`). Units: ``eps`` is energy imparted [keV].
    """
    A = _primary_norm_A(let_keV_um, site_diameter_um, coeff.C, coeff.B)
    primary = A * _primary_shape(eps_keV, let_keV_um, site_diameter_um, coeff.C, coeff.B)
    d1 = _delta_shape(eps_keV, coeff.mu1, coeff.j1)
    d2 = _delta_shape(eps_keV, coeff.mu2, coeff.j2)
    total: npt.NDArray[np.float64] = primary + d1 + d2
    return total


@dataclass(frozen=True, slots=True)
class SatoLinealEnergyEngine:
    """Native Sato-AMF single-event spectrum engine (``SpectrumEngine``).

    Produces a :class:`Spectrum` for an :class:`Ion` in ``site`` (a spherical water
    target, diameter 1-2000 nm) from the Sato AMF (Sato 2006 Eq 7 / 2012 Eq 1),
    with LET supplied by ``stopping_power``. Drop-in for any MKM model via the
    ``SpectrumEngine`` seam: ``MCFMKMModel(engine=SatoLinealEnergyEngine(...))``.

    The per-condition Sato coefficients (``B, C, mu_k, j_k``) are fit outputs. Pass
    them explicitly via ``coefficients`` for a single condition, or leave it
    ``None`` (default) to look them up automatically from the bundled Hartzell 2025
    AMF fits (:func:`pyradbio.micro.sato_coefficients.sato_coefficients_for`,
    interpolated in energy) -- self-contained, no PHITS. The automatic path covers
    the six Hartzell ions (H, He, Li, C, O, Ne); other ions raise.
    """

    site: SiteGeometry
    stopping_power: StoppingPower
    density_g_cm3: float = WATER_DENSITY_G_CM3
    coefficients: SatoAmfCoefficients | None = None
    y_nodes: int = 400

    def single_event_spectrum(
        self, ion: Ion, y_grid_keV_um: npt.NDArray[np.float64] | None = None
    ) -> Spectrum:
        """Single-event lineal-energy spectrum from the Sato AMF."""
        if not isinstance(self.site, SphericalSite):
            raise ValueError("SatoLinealEnergyEngine requires a SphericalSite")
        coeff = self.coefficients
        if coeff is None:
            # Lazy import keeps the core module free of the benchmark/scipy-fit
            # dependency; the table is fit once and cached.
            from pyradbio.micro.sato_coefficients import sato_coefficients_for

            coeff = sato_coefficients_for(ion)
        let = self.stopping_power.let_keV_um(ion)
        d = self.site.diameter_um
        l_bar = self.site.mean_chord_um()
        y = (
            self._default_y_grid(let, l_bar)
            if y_grid_keV_um is None
            else np.asarray(y_grid_keV_um, dtype=np.float64)
        )
        f = _build_amf_density(let, d, l_bar, y, coeff)
        meta = {
            "ion": ion,
            "let_keV_um": let,
            "site_diameter_um": d,
            "mean_chord_um": l_bar,
            "engine": "sato_amf",
            "coefficients": coeff,
        }
        return Spectrum(y_keV_um=y, f=f, meta=meta)

    def _default_y_grid(self, let_keV_um: float, l_bar_um: float) -> npt.NDArray[np.float64]:
        """Log-y grid from the one-quantum floor up to the primary core peak.

        Lower edge is the smallest physical lineal energy -- a single ionisation
        ``eps = w_i`` gives ``y = w_i / l_bar``. This floor matters: below it the
        AMF delta-ray term (fit only where the reference data reach, ``y >~ 0.1``)
        is unconstrained and, when a fitted ``j_k`` rails toward 1, produces a
        spurious sub-quantum spike. Upper edge a few times the geometric core
        deposit ``~2 L d / l_bar``.
        """
        assert isinstance(self.site, SphericalSite)
        y_hi = 2.0 * let_keV_um * self.site.diameter_um / l_bar_um
        y_lo = (_W_ION_EV * 1.0e-3) / l_bar_um
        grid: npt.NDArray[np.float64] = np.logspace(
            math.log10(y_lo), math.log10(max(y_hi, 10.0 * y_lo)), self.y_nodes, dtype=np.float64
        )
        return grid


def _build_amf_density(
    let_keV_um: float,
    site_diameter_um: float,
    mean_chord_um: float,
    y_grid_keV_um: npt.NDArray[np.float64],
    coeff: SatoAmfCoefficients,
) -> npt.NDArray[np.float64]:
    """Sato-AMF frequency density ``f(y)`` on ``y_grid_keV_um`` (normalised to 1).

    The AMF ``eps*f1(eps)`` (Sato 2012 Eq 1, ``eps = l_bar * y``) is the
    *dose-weighted* distribution in energy imparted -- with ``eps`` linear in ``y``
    it has the same shape as the ``y*d(y)`` curves it is fit to, i.e.
    ``eps*f1(eps) ~ y*d(y)``. Hence the dose density is ``d(y) ~ [eps*f1]/y`` and the
    frequency density is ``f(y) = d(y)*y_F/y ~ [eps*f1]/y^2``. Normalised so
    ``integral f dy = 1``; the :class:`Spectrum` machinery then gives ``y_F``,
    ``y_D`` and ``d(y)`` self-consistently.
    """
    y = np.asarray(y_grid_keV_um, dtype=np.float64)
    eps = mean_chord_um * y
    v = amf_epsilon_density(eps, let_keV_um, site_diameter_um, coeff)
    safe_y = np.maximum(y, 1.0e-300)
    f = np.where(y > 0.0, v / (safe_y * safe_y), 0.0)
    norm = float(simpson(f, x=y))
    if norm > 0.0:
        f = f / norm
    return f
