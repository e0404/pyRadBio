"""Sato-AMF coefficients fit to the bundled Hartzell 2025 AMF curves.

The self-contained stand-in for PHITS's ``tsed.dat``: the per-(ion, energy) Sato
Eq-7 coefficients (``B, C, mu_k, j_k``) are obtained by least-squares fitting the
functional form (:mod:`pyradbio.micro.sato`) to the digitized Hartzell 2025
**Fig-4** AMF curves (``pyradbio.micro.hartzell2025``) -- derived from bundled
reference data, not invented (AGENTS.md -> "No silent physics"). For a requested
ion the coefficients are interpolated in energy.

Coverage / caveats
------------------
* Ions: the six Fig-4 species -- ``1H 4He 7Li 12C 16O 20Ne`` (Z in 1,2,3,6,8,10).
  Other Z raise ``NotImplementedError`` (nearest-Z extrapolation is future work).
* Energy: interpolated in ``log E`` between the fitted 1/10/100 MeV/u anchors and
  clamped outside; exact at the anchors.
* Site: fit at the Fig-4 0.3 um radius. ``C`` is (approximately) the site-
  independent core energy fraction, so the primary peak still scales correctly
  with the engine's actual site diameter via the ``C*L*phi_s`` term; explicit
  site-size interpolation (Fig-7 carbon series) is not yet modelled.
* LET from :func:`~pyradbio.track.stopping_power.default_stopping_power` (the
  library default); build the engine with the same stopping power
  for a consistent peak position.
"""

from __future__ import annotations

import math
from functools import lru_cache

import numpy as np
import numpy.typing as npt
from scipy.optimize import least_squares

from pyradbio.core.radiation import Ion
from pyradbio.micro.hartzell2025 import iter_amf_curves
from pyradbio.micro.sato import SatoAmfCoefficients, amf_epsilon_density
from pyradbio.track.stopping_power import default_stopping_power

_W_ION_KEV = 10.9e-3  # average energy per commonest ionisation (Sato 2006), keV
# Multistart seeds for the fitted core fraction C (Sato 2006: B and C are fit).
_C_SEEDS = (0.5, 0.7, 0.85)
# Positive parameters spanning decades are interpolated in log space; C in [0,1]
# is interpolated linearly.
_LOG_PARAMS = ("B", "mu1", "mu2", "j1", "j2")


def _model_yd(
    coeff: SatoAmfCoefficients, let_keV_um: float, site_diameter_um: float, l_bar_um: float
) -> float:
    """Dose-mean ``y_D`` the engine will report for ``coeff`` (``= int m / int m/y``).

    Computed on the engine's grid convention (one-quantum floor to a few times the
    core peak), so the fit optimises the *reported* moment, not just the shape in
    the data window.
    """
    y_hi = 2.0 * let_keV_um * site_diameter_um / l_bar_um
    y_lo = _W_ION_KEV / l_bar_um
    y = np.logspace(math.log10(y_lo), math.log10(max(y_hi, 10.0 * y_lo)), 600, dtype=np.float64)
    m = amf_epsilon_density(l_bar_um * y, let_keV_um, site_diameter_um, coeff)
    num = float(np.trapezoid(m, y))
    den = float(np.trapezoid(m / y, y))
    return num / den if den > 0.0 else 0.0


def _fit_one_curve(
    let_keV_um: float,
    site_diameter_um: float,
    y_keV_um: npt.NDArray[np.float64],
    yd_ref: npt.NDArray[np.float64],
    ref_yd_moment: float,
) -> SatoAmfCoefficients:
    """Least-squares fit ``B, C, mu_k, j_k`` of the Sato form to one ``y*d(y)`` curve.

    The residual has two parts: (a) the log-space shape mismatch over the digitized
    ``y*d(y)`` window, and (b) a term pinning the model's reported ``y_D`` (computed
    on the engine grid, ``int m / int m/y``) to the reference ``ref_yd_moment``.
    Part (b) is essential: the digitized curves only reach ``y ~ 0.1``, so the
    shape term alone leaves the sub-data delta-ray extrapolation -- which the
    ``f = m/y^2`` engine amplifies -- unconstrained, and the reported ``y_D`` can
    collapse. ``A`` is derived from ``B, C`` (Sato 2006 Eq 8); ``scale`` absorbs
    the digitized curve's arbitrary units.
    """
    l_bar = 2.0 * site_diameter_um / 3.0
    eps = y_keV_um * l_bar
    n = y_keV_um.size
    yd_weight = math.sqrt(n)  # scale the single y_D residual to bite against the curve

    def resid(p: npt.NDArray[np.float64]) -> npt.NDArray[np.float64]:
        B, C, mu1, mu2, j1, j2, scale = p
        coeff = SatoAmfCoefficients(B=B, C=C, mu1=mu1, mu2=mu2, j1=j1, j2=j2)
        model = scale * amf_epsilon_density(eps, let_keV_um, site_diameter_um, coeff)
        shape = np.log(np.maximum(model, 1e-12)) - np.log(np.maximum(yd_ref, 1e-12))
        yd_pen = yd_weight * math.log(
            max(_model_yd(coeff, let_keV_um, site_diameter_um, l_bar), 1e-6) / ref_yd_moment
        )
        return np.concatenate([shape, [yd_pen]])

    lb = [1e-4, 0.05, 0.0, 0.0, 1.05, 1.05, 1e-8]
    ub = [1e3, 1.0, 100.0, 100.0, 100.0, 2000.0, 1e8]
    best_x: npt.NDArray[np.float64] | None = None
    best_cost = np.inf
    for cg in _C_SEEDS:
        p0 = [5.0 / (cg * let_keV_um * site_diameter_um), cg, 0.01, 0.001, 2.0, 20.0, 1.0]
        fit = least_squares(resid, p0, bounds=(lb, ub), max_nfev=8000)
        if fit.cost < best_cost:
            best_cost = fit.cost
            best_x = fit.x
    assert best_x is not None
    B, C, mu1, mu2, j1, j2, _ = best_x
    return SatoAmfCoefficients(B=B, C=C, mu1=mu1, mu2=mu2, j1=j1, j2=j2)


@lru_cache(maxsize=1)
def _fig4_table() -> dict[int, tuple[tuple[float, SatoAmfCoefficients], ...]]:
    """Fit every Hartzell Fig-4 AMF curve; return ``{Z: ((E, coeff), ...)}`` sorted by E."""
    sp = default_stopping_power()
    by_z: dict[int, list[tuple[float, SatoAmfCoefficients]]] = {}
    for curve in iter_amf_curves(figure="fig4", method="AMF"):
        ion = Ion(Z=curve.Z, A=curve.A, energy_MeV_u=curve.energy_MeV_u)
        let = sp.let_keV_um(ion)
        d = 2.0 * curve.site_radius_um
        y, dref = curve.dose_spectrum(normalized=True)
        ref_yd = float(np.trapezoid(y * dref, y))  # reference dose-mean y_D
        coeff = _fit_one_curve(let, d, y, y * dref, ref_yd)
        by_z.setdefault(curve.Z, []).append((curve.energy_MeV_u, coeff))
    return {z: tuple(sorted(entries, key=lambda t: t[0])) for z, entries in by_z.items()}


def sato_coefficients_for(ion: Ion) -> SatoAmfCoefficients:
    """Interpolated Sato coefficients for ``ion`` from the Hartzell Fig-4 fits.

    Interpolates each coefficient in ``log E`` (log-space for the positive
    parameters, linear for ``C``), clamped outside the 1-100 MeV/u anchors.
    Raises ``NotImplementedError`` for ions outside the six covered species.
    """
    table = _fig4_table()
    if ion.Z not in table:
        raise NotImplementedError(
            f"no bundled Hartzell AMF fit for Z={ion.Z}; covered Z: {sorted(table)}"
        )
    entries = table[ion.Z]
    log_e = np.array([math.log(e) for e, _ in entries])
    target = math.log(ion.energy_MeV_u)

    def interp(attr: str) -> float:
        vals = np.array([getattr(c, attr) for _, c in entries])
        if attr in _LOG_PARAMS:
            return float(np.exp(np.interp(target, log_e, np.log(vals))))
        return float(np.interp(target, log_e, vals))

    return SatoAmfCoefficients(
        B=interp("B"),
        C=interp("C"),
        mu1=interp("mu1"),
        mu2=interp("mu2"),
        j1=interp("j1"),
        j2=interp("j2"),
    )


__all__ = ["sato_coefficients_for"]
