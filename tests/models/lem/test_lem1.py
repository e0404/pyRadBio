"""LEM I."""

from __future__ import annotations

import dataclasses
import itertools
import math
import warnings

import numpy as np
import pytest

from pyradbio.core.cell import CellTarget
from pyradbio.core.constants import LEM_D1_PREFACTOR, WATER_DENSITY_G_CM3
from pyradbio.core.lq import dose_for_survival
from pyradbio.core.photon import PhotonParameters
from pyradbio.core.radiation import Ion, RadiationField, SpectrumComponent
from pyradbio.models.lem.base import NuclearDoseTable
from pyradbio.models.lem.lem1 import LEM1, SpecificEnergySurvival
from pyradbio.track.kiefer_chatterjee import KieferChatterjeeRadialDose
from pyradbio.track.scholz_kraft import ScholzKraftRadialDose
from pyradbio.track.stopping_power import SimpleBetheStoppingPower

# V79 photon inputs and geometry (Elsasser & Scholz 2007, Fig. 6 'LEM old').
V79 = PhotonParameters(alpha_x=0.093, beta_x=0.026, D_t=30.0)
V79_CELL = CellTarget(nucleus_radius_um=3.8)
CHO_1997 = PhotonParameters(alpha_x=0.18, beta_x=0.028, D_t=30.0)
CHO_1997_CELL = CellTarget(nucleus_radius_um=5.0, effective_size_factor=1.0)


def _carbon_rbe_alpha_vs_let(lem: LEM1, energies_MeV_u: list[float]) -> list[tuple[float, float]]:
    """Return (LET, RBE_alpha) for carbon-12 at the given energies, LET-sorted."""
    sp = SimpleBetheStoppingPower()
    out = []
    for energy in energies_MeV_u:
        ion = Ion(6, 12, energy)
        let = sp.let_keV_um(ion)
        rbe_alpha = lem.alpha_beta(ion).alpha / lem.photon.alpha_x
        out.append((let, rbe_alpha))
    return sorted(out)


@pytest.mark.golden
def test_golden_lem1_rbe_let_has_single_maximum_v79() -> None:
    """LEM I RBE_alpha(LET) for V79 carbon peaks once near ~150-200 keV/um.

    Reference: Elsasser & Scholz 2007, Fig. 6 ('LEM old'); doi:10.1667/RR0467.1.
    V79 photon inputs alpha_x=0.093, beta_x=0.026, D_t=30 Gy, R_n=3.8 um. The
    curve rises from ~1 at low LET to a single maximum in the carbon overkill
    region, then falls; the peak position is the figure's characteristic
    LEM-I/'old' carbon maximum.

    The digitized Fig. 6 'LEM old' line peaks at **7.78 at 228 keV/um**
    (``benchmark/data/elsasser2007_figure6_lem_old.csv``); this grid's coarse
    energy sampling resolves that to the neighbouring point at 239 keV/um.

    A 140-200 keV/um peak is **not the paper's curve**: it results from
    ``CellTarget`` shrinking R_n by a 0.8 ``effective_size_factor`` (the default is
    1.0), which puts the peak at 6.87@150. See ``pyradbio/core/cell.py``.
    """
    lem = LEM1(photon=V79, cell=V79_CELL)
    energies = [300.0, 150.0, 80.0, 40.0, 25.0, 16.0, 11.0, 8.0, 6.0, 4.5, 3.5, 2.5, 1.8, 1.2]
    curve = _carbon_rbe_alpha_vs_let(lem, energies)
    lets = [let for let, _ in curve]
    rbe = [r for _, r in curve]

    # Exactly one interior maximum: differences flip sign once (+...+ -...-).
    diffs = np.diff(rbe)
    sign_changes = int(np.sum(np.diff(np.sign(diffs)) != 0))
    assert sign_changes == 1, f"expected a single maximum, got {sign_changes} turning points"

    peak = int(np.argmax(rbe))
    # Paper: 7.78 at 228 keV/um; bounds allow this grid's quantization (193/239).
    assert 200.0 <= lets[peak] <= 260.0, f"peak at {lets[peak]:.0f} keV/um (paper: 228)"
    assert 7.0 <= rbe[peak] <= 8.5, f"peak RBE_alpha = {rbe[peak]:.2f} (paper: 7.78)"
    # Low-LET branch approaches the photon response (RBE_alpha -> 1).
    assert rbe[0] < rbe[peak]
    assert lets[0] < 20.0 and 2.0 < rbe[0] < 3.5


@pytest.mark.golden
def test_golden_lem1_initial_slope_matches_scholz1997_fig1_shape() -> None:
    """LEM I initial slope reproduces Scholz 1997 Fig. 1.

    Reference: Scholz et al. 1997, Fig. 1 (doi:10.1007/s004110050055). CHO
    inputs are alpha_x=0.18 Gy^-1, beta_x=0.028 Gy^-2, D_t=30 Gy, and a 10 um
    nuclear diameter. The figure's approximation peaks for carbon near
    5-10 MeV/u, gives low-energy proton alphas near 1.7 Gy^-1, and -- crucially --
    decays back toward the photon value alpha_x=0.18 at low LET (high energy),
    where the impact-parameter integral's indirect traversals dominate.
    """
    lem = LEM1(photon=CHO_1997, cell=CHO_1997_CELL)

    proton = {e: lem.alpha_beta(Ion(1, 1, e)).alpha for e in (1.0, 10.0, 100.0)}
    carbon = {e: lem.alpha_beta(Ion(6, 12, e)).alpha for e in (1.0, 5.0, 10.0, 100.0)}

    assert proton[1.0] == pytest.approx(1.7, rel=0.15)
    assert proton[10.0] == pytest.approx(0.6, rel=0.15)
    assert proton[100.0] == pytest.approx(0.25, rel=0.18)
    assert carbon[1.0] == pytest.approx(0.7, rel=0.2)
    assert carbon[5.0] == pytest.approx(0.95, rel=0.18)
    assert carbon[10.0] == pytest.approx(0.9, rel=0.18)
    assert carbon[100.0] == pytest.approx(0.5, rel=0.2)
    assert carbon[5.0] > carbon[1.0]
    assert carbon[5.0] > carbon[100.0]
    # Low-LET protons approach -- but never drop below -- the photon limit.
    assert lem.photon.alpha_x <= proton[100.0] < proton[10.0]


def test_approximate_inactivation_cross_section_at_one_percent_uses_lq_slope() -> None:
    """Approximate sigma_1% uses the ion-LQ slope at 1% survival.

    The observable follows Scholz & Kraft 1996 Figs. 2-3
    (doi:10.1016/0273-1177(95)00784-C), while the ion-LQ curve is explicitly an
    approximation rather than the paper's full stochastic calculation.
    """
    lem = LEM1(photon=V79, cell=V79_CELL, beta_model="slope_matching")
    ion = Ion(6, 12, 5.0)
    lq = lem.alpha_beta(ion)
    dose_1pct = dose_for_survival(0.01, lq.alpha, lq.beta)
    expected = (
        LEM_D1_PREFACTOR
        * lem.stopping_power.let_keV_um(ion)
        * (lq.alpha + 2.0 * lq.beta * dose_1pct)
        / WATER_DENSITY_G_CM3
    )

    assert lem.approximate_inactivation_cross_section_um2(
        ion, survival_level=0.01
    ) == pytest.approx(expected, rel=1e-12)


def test_approximate_inactivation_cross_section_tends_to_exact_sigma_at_zero_dose() -> None:
    lem = LEM1(photon=V79, cell=V79_CELL)
    ion = Ion(6, 12, 5.0)

    sigma_near_zero = lem.approximate_inactivation_cross_section_um2(ion, survival_level=1.0)

    assert sigma_near_zero == pytest.approx(lem.inactivation_cross_section_um2(ion), rel=1e-12)


@pytest.mark.parametrize("survival_level", [0.0, -0.1, 1.1])
def test_approximate_inactivation_cross_section_rejects_invalid_survival(
    survival_level: float,
) -> None:
    lem = LEM1(photon=V79, cell=V79_CELL)

    with pytest.raises(ValueError, match="survival"):
        lem.approximate_inactivation_cross_section_um2(
            Ion(6, 12, 5.0), survival_level=survival_level
        )


@pytest.mark.slow
def test_initial_slope_radial_matches_explicit_2d_nuclear_integral() -> None:
    """The fast radial N_1 equals an explicit 2-D nuclear-disk integral.

    ``single_traversal_log_kill`` reduces the 2-D nuclear cross-section integral
    of ``-ln S_x(D(r))`` to a 1-D radial quadrature by azimuthal symmetry. Here
    a brute-force Cartesian integral over the effective nuclear disk confirms
    that reduction (Scholz & Kraft 1996 eqs. 5-6) for one V79 carbon case.
    """
    lem = LEM1(photon=V79, cell=V79_CELL)
    ion = Ion(6, 12, 10.0)
    radial = lem._radial()
    r_eff = lem.cell.effective_nucleus_radius_um()

    n = 1600
    xs = np.linspace(-r_eff, r_eff, n)
    dx = float(xs[1] - xs[0])
    total = 0.0
    for x in xs:
        y_lim = math.sqrt(max(r_eff * r_eff - x * x, 0.0))
        ys = np.arange(-y_lim, y_lim, dx)
        for y in ys:
            r = math.hypot(x, float(y))
            total += lem.photon.log_kill(radial.local_dose_Gy(r, ion))
    n1_2d = total * dx * dx / (math.pi * r_eff * r_eff)

    n1_radial = lem.single_traversal_log_kill(ion)
    assert n1_radial == pytest.approx(n1_2d, rel=0.01)


@pytest.mark.slow
def test_impact_log_kill_matches_brute_2d_offset_integral() -> None:
    """nu(b>0) (chord-angle 1-D reduction) equals an explicit 2-D nuclear integral.

    ``_impact_log_kill`` reduces the 2-D integral of ``-ln S_x(D)`` over the
    nuclear disk for a track offset by impact parameter ``b`` to a 1-D radial
    quadrature via the chord half-angle. A brute-force Cartesian integral over the
    disk (track axis at ``(b, 0)``) confirms that reduction for an indirect
    traversal whose track centre lies *outside* the nucleus (b > R_eff).
    """
    lem = LEM1(photon=V79, cell=V79_CELL)
    ion = Ion(6, 12, 50.0)
    radial = lem._radial()
    r_eff = lem.cell.effective_nucleus_radius_um()
    r_max = radial.r_max_um(ion)
    b = 1.6 * r_eff  # indirect: track misses the nucleus, only its tail reaches it
    profile = lem._radial_logkill_profile(ion)
    nu_fast = lem._impact_log_kill(b, profile, r_eff, r_max)

    n = 1400
    xs = np.linspace(-r_eff, r_eff, n)
    dx = float(xs[1] - xs[0])
    total = 0.0
    for x in xs:
        y_lim = math.sqrt(max(r_eff * r_eff - x * x, 0.0))
        ys = np.arange(-y_lim, y_lim, dx)
        for y in ys:
            s = math.hypot(x - b, float(y))
            total += lem.photon.log_kill(max(radial.local_dose_Gy(s, ion), 0.0))
    nu_2d = total * dx * dx / (math.pi * r_eff * r_eff)
    assert nu_fast == pytest.approx(nu_2d, rel=0.02)


def test_initial_slope_recovers_photon_alpha_at_low_let() -> None:
    """alpha_ion -> alpha_x at low LET (high energy).

    The defining LEM consistency check: at vanishing LET the inhomogeneous track
    dose is everywhere small, the local response is linear, and the impact-
    parameter integral (direct + indirect traversals) must return the photon
    initial slope. A central-traversal-only slope would drop toward zero here.
    """
    lem = LEM1(photon=CHO_1997, cell=CHO_1997_CELL)
    for energy in (1.0e3, 5.0e3, 2.0e4):
        alpha = lem.initial_slope(Ion(1, 1, energy))
        assert alpha == pytest.approx(lem.photon.alpha_x, rel=0.08)
        assert alpha >= lem.photon.alpha_x * 0.97  # approaches from above


def test_inactivation_cross_section_saturates_at_nuclear_area() -> None:
    """sigma -> A_eff (overkill) at high LET, so alpha -> 1/D_1.

    Scholz & Kraft 1996 Fig. 2/3: the inactivation cross-section saturates at the
    effective geometric nuclear area once a single traversal kills with near
    certainty. There alpha is capped by the single-traversal rate 1/D_1.
    """
    lem = LEM1(photon=V79, cell=V79_CELL)
    r_eff = lem.cell.effective_nucleus_radius_um()
    area = math.pi * r_eff * r_eff
    ion = Ion(6, 12, 1.5)  # ~390 keV/um, deep in the overkill region
    sigma = lem.inactivation_cross_section_um2(ion)
    # sigma settles at ~A_eff (and may slightly exceed it via indirect kills,
    # the same effect that lifts heavy-ion cross-sections above the nuclear area
    # in Scholz & Kraft 1996 Fig. 3), so alpha tracks the single-traversal rate.
    assert sigma == pytest.approx(area, rel=0.1)
    assert lem.initial_slope(ion) == pytest.approx(1.0 / lem.mean_traversal_dose_Gy(ion), rel=0.1)


def test_alpha_beta_returns_sane_lq_over_let() -> None:
    """LEM1.alpha_beta gives sane alpha(LET), beta(LET): alpha rises with LET on
    the sub-overkill branch, beta is the bounded slope-matched approximation
    anti-correlated with alpha, and the proton branch crosses the photon-
    equivalent point (RBE_alpha = 1)."""
    lem = LEM1(photon=V79, cell=V79_CELL)

    # Carbon alpha rises with LET below the overkill peak (~160 keV/um).
    a200 = lem.alpha_beta(Ion(6, 12, 200.0))
    a50 = lem.alpha_beta(Ion(6, 12, 50.0))
    a20 = lem.alpha_beta(Ion(6, 12, 20.0))
    assert a200.alpha < a50.alpha < a20.alpha

    # beta = (s_max - alpha)/(2 D_t): non-negative, bounded by the alpha->0
    # maximum, and decreasing as alpha grows.
    beta_max = V79.max_slope() / (2.0 * V79.D_t)
    for lq in (a200, a50, a20):
        assert 0.0 <= lq.beta <= beta_max
    assert a20.beta < a50.beta < a200.beta

    # RBE_alpha >= 1 always (the inhomogeneous track dose can only raise the
    # convex log-kill above the photon mean; Jensen). The light proton branch
    # decays toward the photon-equivalent RBE_alpha = 1 at low LET (high energy),
    # approaching it from above rather than dropping below.
    rbe_proton_hi, rbe_proton_lo = (
        lem.alpha_beta(Ion(1, 1, e)).alpha / V79.alpha_x for e in (800.0, 50.0)
    )
    assert 1.0 <= rbe_proton_hi < rbe_proton_lo
    assert rbe_proton_hi == pytest.approx(1.0, abs=0.15)


@pytest.mark.golden
def test_golden_lem1_scholz1997_stochastic_beta_matches_fig3_shape() -> None:
    """LEM I optional stochastic beta approximation follows Scholz 1997 Fig. 3.

    Reference: Scholz et al. 1997, Fig. 3 and eqs. (9)-(11)
    (doi:10.1007/s004110050055). The approximate calculation works in nucleus
    specific energy z, with beta_z = (s_max - alpha_z)/(2 D_t), then averages
    survival over Poisson-distributed traversals; the resulting dose beta is
    ``beta_z * exp(-2 N_1)``, rising with energy and suppressed for dense
    low-energy carbon tracks. This is a fast/crude estimate -- a stochastic
    Monte-Carlo of the full spatial model confirms the deterministic
    `LEM1.scholz1997_two_track_beta` is the quantitatively faithful path.
    """
    lem = LEM1(photon=CHO_1997, cell=CHO_1997_CELL, beta_model="scholz1997_stochastic")

    proton = {e: lem.alpha_beta(Ion(1, 1, e)).beta for e in (1.0, 5.0, 10.0, 100.0)}
    carbon = {e: lem.alpha_beta(Ion(6, 12, e)).beta for e in (1.0, 5.0, 10.0, 100.0)}

    assert proton[1.0] == pytest.approx(0.001, abs=0.001)
    assert proton[5.0] == pytest.approx(0.016, rel=0.25)
    assert proton[10.0] == pytest.approx(0.021, rel=0.18)
    assert proton[100.0] == pytest.approx(0.028, rel=0.12)
    assert carbon[1.0] == pytest.approx(0.0, abs=0.001)
    assert carbon[5.0] == pytest.approx(0.003, abs=0.002)
    assert carbon[10.0] == pytest.approx(0.006, rel=0.35)
    assert carbon[100.0] == pytest.approx(0.020, rel=0.25)
    assert proton[1.0] < proton[5.0] < proton[10.0] < proton[100.0]
    assert carbon[1.0] < carbon[5.0] < carbon[10.0] < carbon[100.0]


def test_scholz1997_central_initial_slope_matches_eq6() -> None:
    """The Fig. 1 central-traversal approximation is exposed explicitly.

    Reference: Scholz et al. 1997, eqs. (5)-(6) and Fig. 1
    (doi:10.1007/s004110050055). With only central traversals,
    alpha=(1-S_1)/D_1.
    """
    lem = LEM1(photon=CHO_1997, cell=CHO_1997_CELL)
    ion = Ion(6, 12, 10.0)

    expected = (1.0 - lem.single_traversal_survival(ion)) / lem.mean_traversal_dose_Gy(ion)

    assert lem.scholz1997_central_initial_slope(ion) == pytest.approx(expected)
    assert lem.scholz1997_central_initial_slope(ion) < lem.initial_slope(ion)


def test_lethal_central_saturating_matches_ks2006_eq9() -> None:
    """alpha_D = (1 - exp(-N_1))/D_1 (Krämer & Scholz 2006 eq 9), no s_max cap.

    The central-traversal (1 - exp) saturating slope: thins down below the uncapped
    lethal_full N_1/D_1 and the lethal_central s_max plateau at high LET, and (being
    central-only) sits below the faithful cross_section slope.
    """
    lem = LEM1(photon=CHO_1997, cell=CHO_1997_CELL, initial_slope_model="lethal_central_saturating")
    ion = Ion(6, 12, 3.0)  # high-LET carbon (near the LET maximum)

    n_1 = lem.single_traversal_log_kill(ion)
    d_1 = lem.mean_traversal_dose_Gy(ion)
    expected = (1.0 - math.exp(-n_1)) / d_1
    assert lem.initial_slope(ion) == pytest.approx(expected)
    # identical to the Scholz-1997 central slope (same formula).
    assert lem.initial_slope(ion) == pytest.approx(lem.scholz1997_central_initial_slope(ion))

    # (1 - exp(-N_1)) <= N_1, so it sits below the uncapped lethal_full N_1/D_1,
    # the lethal_central s_max plateau, and the faithful cross_section slope.
    full = LEM1(photon=CHO_1997, cell=CHO_1997_CELL, initial_slope_model="lethal_full")
    plateau = LEM1(photon=CHO_1997, cell=CHO_1997_CELL, initial_slope_model="lethal_central")
    cross = LEM1(photon=CHO_1997, cell=CHO_1997_CELL, initial_slope_model="cross_section")
    assert lem.initial_slope(ion) < full.initial_slope(ion)
    assert lem.initial_slope(ion) < plateau.initial_slope(ion)  # thins down below the s_max cap
    assert lem.initial_slope(ion) < cross.initial_slope(ion)  # central-only under-shoots


def test_rapid_lem_beta_matches_ks2006_eq10() -> None:
    """beta_D = f^2 * beta_z with f = (1 - exp(-N_1))/N_1 (Krämer & Scholz 2006 eq 10).

    The f^2 suppression of the z-domain beta_z: f in (0, 1) with beta_D < beta_z at
    high LET, and beta_D -> beta_z as f -> 1 at low LET.
    """
    lem = LEM1(photon=CHO_1997, cell=CHO_1997_CELL, beta_model="rapid_lem")
    ion = Ion(6, 12, 3.0)

    n_1 = lem.single_traversal_log_kill(ion)
    beta_z = lem.scholz1997_specific_energy_lq(ion).beta
    f = (1.0 - math.exp(-n_1)) / n_1
    assert lem.rapid_lem_beta(ion) == pytest.approx(f * f * beta_z)
    assert 0.0 < f < 1.0
    assert lem.rapid_lem_beta(ion) < beta_z  # f^2 suppression at high LET
    assert lem.alpha_beta(ion).beta == pytest.approx(lem.rapid_lem_beta(ion))

    # low-LET limit: f -> 1, so beta_D -> beta_z (the z-domain slope-matched beta).
    hi = Ion(1, 1, 1.0e4)
    n_1_hi = lem.single_traversal_log_kill(hi)
    f_hi = (1.0 - math.exp(-n_1_hi)) / n_1_hi
    assert f_hi == pytest.approx(1.0, abs=0.05)
    assert lem.rapid_lem_beta(hi) == pytest.approx(
        lem.scholz1997_specific_energy_lq(hi).beta, rel=0.1
    )


def test_scholz1997_specific_energy_lq_reaches_photon_max_slope_at_dt() -> None:
    """The z-domain approximation follows Scholz 1997 eqs. (9)-(11)."""
    lem = LEM1(photon=CHO_1997, cell=CHO_1997_CELL)
    ion = Ion(1, 1, 50.0)

    z_lq = lem.scholz1997_specific_energy_lq(ion)

    assert z_lq.alpha == pytest.approx(
        lem.single_traversal_log_kill(ion) / lem.mean_traversal_dose_Gy(ion)
    )
    assert z_lq.beta == pytest.approx((lem.photon.max_slope() - z_lq.alpha) / 60.0)
    assert lem.scholz1997_specific_energy_slope(ion, 0.0) == pytest.approx(z_lq.alpha)
    assert lem.scholz1997_specific_energy_slope(ion, 30.0) == pytest.approx(lem.photon.max_slope())


def test_scholz1997_approximate_survival_is_poisson_average() -> None:
    """Fast-path survival averages the z-domain LQ relation over traversals."""
    lem = LEM1(photon=CHO_1997, cell=CHO_1997_CELL)
    ion = Ion(1, 1, 8.0)

    assert lem.scholz1997_approximate_survival(ion, 0.0) == pytest.approx(1.0)
    assert (
        lem.scholz1997_approximate_survival(ion, 4.0)
        < lem.scholz1997_approximate_survival(ion, 2.0)
        < 1.0
    )

    small_dose = 1.0e-7
    assert lem.scholz1997_approximate_log_kill(ion, small_dose) / small_dose == pytest.approx(
        lem.scholz1997_central_initial_slope(ion), rel=1e-4
    )


def test_scholz1997_approximate_survival_handles_high_traversal_counts() -> None:
    """The Poisson sum stays stable for high-energy ions with tiny Delta z."""
    lem = LEM1(photon=CHO_1997, cell=CHO_1997_CELL)
    ion = Ion(1, 1, 400.0)

    survival = lem.scholz1997_approximate_survival(ion, 10.0)

    assert 0.0 < survival < 1.0
    assert math.isfinite(lem.scholz1997_approximate_log_kill(ion, 10.0))


def test_scholz1997_bulk_log_kill_matches_scalar_path() -> None:
    """Bulk evaluation reuses one ion's fast-path parameters without changing results."""
    lem = LEM1(photon=CHO_1997, cell=CHO_1997_CELL)
    ion = Ion(6, 12, 50.0)
    doses = (0.0, 0.5, 2.0, 6.0)

    bulk = lem.scholz1997_approximate_log_kills(ion, doses)

    assert bulk == pytest.approx([lem.scholz1997_approximate_log_kill(ion, dose) for dose in doses])


def test_lem1_default_beta_model_is_slope_matching() -> None:
    """The Scholz 1997 stochastic beta model is opt-in."""
    default = LEM1(photon=CHO_1997, cell=CHO_1997_CELL)
    explicit = LEM1(photon=CHO_1997, cell=CHO_1997_CELL, beta_model="slope_matching")
    ion = Ion(6, 12, 10.0)

    assert default.alpha_beta(ion).beta == pytest.approx(explicit.alpha_beta(ion).beta)
    assert default.alpha_beta(ion).beta != pytest.approx(
        LEM1(
            photon=CHO_1997,
            cell=CHO_1997_CELL,
            beta_model="scholz1997_stochastic",
        )
        .alpha_beta(ion)
        .beta
    )


def test_lem1_rejects_unknown_beta_model() -> None:
    """LEM I beta strategy names are explicit to avoid silent physics."""
    with pytest.raises(ValueError, match="beta_model"):
        LEM1(photon=CHO_1997, cell=CHO_1997_CELL, beta_model="unknown")


# ---- TRiP98 / rapidLEM initial-slope compatibility mode -----------------
def test_lethal_central_initial_slope_is_n1_over_d1_capped_at_smax() -> None:
    """lethal_central: alpha_I = min(N_1/D_1, s_max) (Kraemer & Scholz 2006 / TRiP98)."""
    lem = LEM1(photon=V79, cell=V79_CELL, initial_slope_model="lethal_central")
    for energy in (5.0, 50.0, 300.0):
        ion = Ion(6, 12, energy)
        expected = min(
            lem.single_traversal_log_kill(ion) / lem.mean_traversal_dose_Gy(ion),
            V79.max_slope(),
        )
        assert lem.lethal_central_initial_slope(ion) == pytest.approx(expected)
        assert lem.alpha_beta(ion).alpha == pytest.approx(expected)


def test_lethal_central_saturates_at_smax_where_default_thins_down() -> None:
    """lethal_central holds alpha_I at s_max at high LET; the default thins down.

    The two initial-slope models agree in the therapeutic window but diverge in
    the overkill region: TRiP's rapidLEM saturates the initial slope at the photon
    maximum slope s_max = alpha_x + 2 beta_x D_t (RBE_alpha plateau at
    s_max/alpha_x), while the amorphous inactivation cross-section produces
    overkill thindown (alpha_I turns over and decreases).
    """
    default = LEM1(photon=V79, cell=V79_CELL)
    trip = LEM1(photon=V79, cell=V79_CELL, initial_slope_model="lethal_central")
    s_max = V79.max_slope()

    # Never exceeds the photon max slope, and reaches it at very high LET (low E).
    dense = Ion(6, 12, 1.2)  # deep overkill region
    assert trip.initial_slope(dense) == pytest.approx(s_max, rel=1e-3)
    assert default.initial_slope(dense) < 0.6 * s_max  # cross-section thins down

    # Therapeutic window: the two models agree closely.
    mid = Ion(6, 12, 100.0)
    assert trip.initial_slope(mid) == pytest.approx(default.initial_slope(mid), rel=0.1)


def test_lethal_central_never_exceeds_photon_max_slope() -> None:
    """The s_max cap holds across the whole LET range for several ions."""
    trip = LEM1(photon=CHO_1997, cell=CHO_1997_CELL, initial_slope_model="lethal_central")
    s_max = CHO_1997.max_slope()
    for ion in (Ion(1, 1, 2.0), Ion(2, 4, 5.0), Ion(6, 12, 3.0), Ion(6, 12, 200.0)):
        assert trip.initial_slope(ion) <= s_max + 1e-12


def test_lem1_default_initial_slope_model_is_cross_section() -> None:
    """The TRiP-rapid mode is opt-in; the default is the inactivation cross-section."""
    default = LEM1(photon=V79, cell=V79_CELL)
    explicit = LEM1(photon=V79, cell=V79_CELL, initial_slope_model="cross_section")
    ion = Ion(6, 12, 10.0)
    assert default.initial_slope(ion) == pytest.approx(explicit.initial_slope(ion))
    # In the overkill region the two models differ substantially.
    assert default.initial_slope(Ion(6, 12, 1.5)) != pytest.approx(
        LEM1(photon=V79, cell=V79_CELL, initial_slope_model="lethal_central").initial_slope(
            Ion(6, 12, 1.5)
        ),
        rel=0.2,
    )


def test_lem1_rejects_unknown_initial_slope_model() -> None:
    """Initial-slope model names are explicit to avoid silent physics."""
    with pytest.raises(ValueError, match="initial_slope_model"):
        LEM1(photon=V79, cell=V79_CELL, initial_slope_model="unknown")  # type: ignore[arg-type]


def test_mixed_field_dose_averages_components() -> None:
    """A mixed field dose-averages alpha and sqrt(beta) across components, and a
    one-component field reduces to the single-ion result."""
    lem = LEM1(photon=V79, cell=V79_CELL)
    ion_a = Ion(6, 12, 10.0)
    ion_b = Ion(6, 12, 100.0)
    lq_a = lem.alpha_beta(ion_a)
    lq_b = lem.alpha_beta(ion_b)

    # Identity: a single dose-weighted component equals the bare ion.
    mono = RadiationField([SpectrumComponent(ion_a, dose_Gy=2.0)])
    lq_mono = lem.alpha_beta(mono)
    assert lq_mono.alpha == pytest.approx(lq_a.alpha)
    assert lq_mono.beta == pytest.approx(lq_a.beta)

    # Equal-dose mix lands between the two single-ion alphas (dose average).
    mix = RadiationField(
        [SpectrumComponent(ion_a, dose_Gy=1.0), SpectrumComponent(ion_b, dose_Gy=1.0)]
    )
    lq_mix = lem.alpha_beta(mix)
    assert min(lq_a.alpha, lq_b.alpha) < lq_mix.alpha < max(lq_a.alpha, lq_b.alpha)
    assert lq_mix.alpha == pytest.approx(0.5 * (lq_a.alpha + lq_b.alpha))
    root = 0.5 * (math.sqrt(lq_a.beta) + math.sqrt(lq_b.beta))
    assert lq_mix.beta == pytest.approx(root * root)


def test_requires_transition_dose() -> None:
    """LEM needs a photon transition dose D_t (shouldered->exponential ref)."""
    with pytest.raises(ValueError, match="D_t"):
        LEM1(photon=PhotonParameters(alpha_x=0.093, beta_x=0.026), cell=V79_CELL)


# --- traversal-conditioned (Rao-Blackwellised) stochastic survival ------------
# Scholz & Kraft 1994/1996: S(D) = sum_k Poisson(k; lambda(D)) g_k, with
# g_k = E[exp(-N_leth) | exactly k traversals] independent of dose.

_TC_LEM = LEM1(
    photon=PhotonParameters(alpha_x=0.136, beta_x=0.040, D_t=12.0),
    cell=CellTarget(nucleus_radius_um=math.sqrt(50.0 / math.pi), effective_size_factor=1.0),
    stopping_power=SimpleBetheStoppingPower(),
)
_TC_ION = Ion(18, 40, 3.0)  # argon, few traversals at 1% survival -> cheap


def _tc_curve(**kwargs: object) -> object:
    defaults = dict(
        max_dose_Gy=45.0,
        n_batches=4,
        realizations_per_batch=6,
        n_field_points=400,
        seed=11,
        max_mean_traversals=20_000.0,
    )
    defaults.update(kwargs)  # type: ignore[arg-type]
    return _TC_LEM.scholz1997_traversal_conditioned_survival(_TC_ION, **defaults)  # type: ignore[arg-type]


def test_traversal_conditioned_zero_traversal_term_is_exact() -> None:
    """g_0 = 1 by construction -- the term the unconditioned MC samples worst.

    At 1% survival this term carries most of S while occurring in well under 1%
    of unconditioned histories, so making it exact is the whole point.
    """
    curve = _tc_curve()

    assert curve.g[0] == 1.0
    assert curve.survival(0.0) == pytest.approx(1.0)


def test_traversal_conditioned_curve_is_monotone_by_construction() -> None:
    """A Poisson mixture of a non-increasing g_k cannot reverse direction."""
    curve = _tc_curve()
    doses = np.linspace(0.0, 45.0, 200)

    survivals = np.array([curve.survival(float(d)) for d in doses])
    assert np.all(np.diff(survivals) <= 0.0)


def test_traversal_conditioned_slope_matches_a_finite_difference() -> None:
    """The analytic d[-ln S]/dD must agree with differencing the same curve.

    This is what replaces fitting a local quadratic to noisy samples: the
    derivative is a closed-form linear functional of the same g vector.
    """
    curve = _tc_curve()
    dose = curve.dose_for_survival(0.01)
    step = 1.0e-4 * dose

    numeric = (-math.log(curve.survival(dose + step)) + math.log(curve.survival(dose - step))) / (
        2.0 * step
    )
    assert curve.log_kill_slope_Gy_inv(dose) == pytest.approx(numeric, rel=1e-6)


def test_traversal_conditioned_dose_for_survival_inverts_the_curve() -> None:
    curve = _tc_curve()

    for level in (0.80, 0.20, 0.01):
        assert curve.survival(curve.dose_for_survival(level)) == pytest.approx(level, rel=1e-9)


def test_traversal_conditioned_poisson_grid_covers_the_requested_range() -> None:
    """Truncating the traversal grid must not silently bias S at the top dose."""
    curve = _tc_curve()

    assert curve.poisson_tail_mass(45.0) < 1.0e-9


def test_traversal_conditioned_pooling_adds_histories() -> None:
    """Extra batches ADD to a curve; an adaptive allocator never restarts it."""
    first = _tc_curve(seed=1)
    second = _tc_curve(seed=2)

    pooled = first.pooled_with(second)

    assert pooled.n_realizations == first.n_realizations + second.n_realizations
    assert pooled.g == pytest.approx(0.5 * (first.g + second.g))


def test_traversal_conditioned_pooling_rejects_a_different_grid() -> None:
    with pytest.raises(ValueError, match="k_max mismatch"):
        _tc_curve(max_dose_Gy=45.0).pooled_with(_tc_curve(max_dose_Gy=90.0))


def test_traversal_conditioned_requires_two_batches_for_a_jackknife() -> None:
    with pytest.raises(ValueError, match="n_batches must be >= 2"):
        _tc_curve(n_batches=1)


def test_traversal_conditioned_refuses_to_extrapolate_past_its_grid() -> None:
    with pytest.raises(ValueError, match="does not reach"):
        _tc_curve(max_dose_Gy=2.0).dose_for_survival(0.01)


def test_traversal_conditioned_jackknife_reports_a_finite_error() -> None:
    curve = _tc_curve()

    value, error = curve.jackknife(lambda g: curve.dose_for_survival(0.01, g=g))
    assert value > 0.0
    assert 0.0 < error < 0.5 * value


@pytest.mark.slow
def test_traversal_conditioned_agrees_with_the_per_dose_monte_carlo() -> None:
    """The conditioning is a variance reduction, not a different model.

    Both estimators target the same S(D); the conditioned one removes the
    traversal-count fluctuation analytically instead of sampling it. Compared in
    log-kill, which is the scale the paper's final slope is read on -- a fixed
    relative tolerance on S itself would be dominated by the reference's own
    noise at 1% survival, where it is carried by its rare zero-traversal
    histories.

    Measured over four seeds each, the conditioned curve runs ~1-3% below the
    per-dose estimator across 5-30 Gy, i.e. inside ~1.5 combined standard errors
    at these history counts. The tolerance below is ~3 sigma for this
    configuration; a systematic drift past it means the shared field-point
    sampling has stopped being unbiased for short prefixes.
    """
    curve = _TC_LEM.scholz1997_traversal_conditioned_survival(
        _TC_ION,
        max_dose_Gy=35.0,
        n_batches=16,
        realizations_per_batch=50,
        seed=5,
        max_mean_traversals=20_000.0,
    )
    doses = np.array([5.0, 15.0, 30.0])
    reference = _TC_LEM.scholz1997_montecarlo_survival(
        _TC_ION, doses, n_realizations=20_000, seed=5, engine="numpy"
    )

    for dose, expected in zip(doses, reference, strict=True):
        assert -math.log(curve.survival(float(dose))) == pytest.approx(-math.log(expected), abs=0.2)


# --- geometric ceiling on any inactivation cross-section ---------------------
#
# A track can only affect a cell whose nucleus centre lies within R + r_max of
# it, so sigma <= pi (R + r_max)^2 regardless of the biology. This caught a real
# bug: the Scholz & Kraft 1996 reconstruction plotted an LQ surrogate built from
# beta_model="slope_matching", which breached this at 84 of Fig. 2's 126 grid
# points, by up to 40x (U at 1 MeV/u came out at 1993 um^2 against a 50 um^2
# nucleus). See docs/references/LEM_I_ScholzKraft.md.

SK1996_V79 = PhotonParameters(alpha_x=0.136, beta_x=0.040, D_t=12.0)
SK1996_CELL = CellTarget(nucleus_radius_um=math.sqrt(50.0 / math.pi), effective_size_factor=1.0)
SK1996_ION_A = {2: 4, 6: 12, 10: 20, 18: 40, 36: 84, 54: 132, 92: 238}


def _sk1996_lem(beta_model: str) -> LEM1:
    return LEM1(
        photon=SK1996_V79,
        cell=SK1996_CELL,
        stopping_power=SimpleBetheStoppingPower(),
        beta_model=beta_model,
    )


def _geometric_ceiling_um2(lem: LEM1, ion: Ion) -> float:
    """``pi (R + r_max)^2`` -- the disc outside which the track cannot reach."""
    r_eff = lem.cell.effective_nucleus_radius_um()
    return math.pi * (r_eff + lem._radial().r_max_um(ion)) ** 2


@pytest.mark.parametrize("z", [2, 6, 10, 18, 36, 54, 92])
@pytest.mark.parametrize("energy_MeV_u", [1.0, 2.0, 5.0, 10.0, 20.0])
def test_cluster_expansion_sigma_respects_the_geometric_ceiling(
    z: int, energy_MeV_u: float
) -> None:
    """The reconstruction's plotted sigma_1% never exceeds pi (R + r_max)^2."""
    lem = _sk1996_lem("scholz1997_two_track")
    ion = Ion(z, SK1996_ION_A[z], energy_MeV_u)
    sigma = lem.approximate_inactivation_cross_section_um2(ion, survival_level=0.01)
    ceiling = _geometric_ceiling_um2(lem, ion)
    assert 0.0 < sigma <= ceiling, f"sigma={sigma:.1f} um^2 exceeds ceiling {ceiling:.1f} um^2"


@pytest.mark.parametrize("z", [2, 6, 10, 18, 36, 54, 92])
@pytest.mark.parametrize("energy_MeV_u", [1.0, 2.0, 5.0, 10.0, 20.0])
def test_zero_dose_sigma_respects_the_geometric_ceiling(z: int, energy_MeV_u: float) -> None:
    """The exact zero-dose cross-section obeys the same bound, as it must."""
    lem = _sk1996_lem("scholz1997_two_track")
    ion = Ion(z, SK1996_ION_A[z], energy_MeV_u)
    sigma = lem.inactivation_cross_section_um2(ion)
    assert 0.0 < sigma <= _geometric_ceiling_um2(lem, ion)


def test_slope_matching_sigma_breaches_the_ceiling_at_high_let() -> None:
    """Pin the reason the reconstruction does not use ``beta_model='slope_matching'``.

    ``beta_I = (s_max - alpha_I)/(2 D_t)`` is a high-dose asymptotic identity. At
    high LET ``sigma_0`` saturates near the nuclear area while LET keeps growing,
    so ``alpha_I -> 0`` and ``beta_I -> s_max/(2 D_t)``: the least-shouldered ions
    get essentially the full photon shoulder, and ``sigma_approx`` grows without
    bound in LET. This is a *documented defect*, not a target to fix by loosening
    -- if it ever starts passing, the beta model changed and the reconstruction's
    accuracy statements need re-measuring.
    """
    lem = _sk1996_lem("slope_matching")
    ion = Ion(92, 238, 1.0)
    sigma = lem.approximate_inactivation_cross_section_um2(ion, survival_level=0.01)
    ceiling = _geometric_ceiling_um2(lem, ion)
    assert sigma > 30.0 * ceiling


# --- saturation geometry: where the two-track pair integral is complete ------
#
# h(d) = [alpha_x d + beta_x d^2] - beta_x (d - D_t)^2 1[d > D_t] exactly. The
# first piece is a quadratic, and a quadratic has no inclusion-exclusion excess
# beyond second order, so ALL higher-order track interplay comes from the second
# piece -- and only because its indicator depends on which tracks are present.
# Where the nucleus stays in the quadratic branch, nu is exactly a pair sum.


def _inclusion_exclusion_excess(response, doses: tuple[float, ...]) -> float:
    """``psi_n``: the n-th inclusion-exclusion excess of ``response`` over subsets."""
    n = len(doses)
    return sum(
        (-1) ** (n - r) * response(sum(subset))
        for r in range(1, n + 1)
        for subset in itertools.combinations(doses, r)
    )


@pytest.mark.parametrize("n", [3, 4, 5])
def test_quadratic_branch_has_no_cluster_terms_beyond_second_order(n: int) -> None:
    """The fact the whole saturation-split argument rests on.

    A degree-2 polynomial has an identically vanishing n-th inclusion-exclusion
    excess for n >= 3. If this ever fails the reasoning behind
    `LEMBase.saturated_area_fraction` -- and the claim that the two-track
    integral is geometrically complete in the unsaturated branch -- is void.
    """
    photon = SK1996_V79
    doses = tuple(1.0 + 0.7 * i for i in range(n))

    def quadratic(dose: float) -> float:
        return photon.alpha_x * dose + photon.beta_x * dose * dose

    assert _inclusion_exclusion_excess(quadratic, doses) == pytest.approx(0.0, abs=1e-12)


def test_second_order_excess_of_the_quadratic_branch_is_the_cross_term() -> None:
    """And at n = 2 it is exactly ``2 beta_x D_1 D_2`` -- the pair term itself."""
    photon = SK1996_V79

    def quadratic(dose: float) -> float:
        return photon.alpha_x * dose + photon.beta_x * dose * dose

    d1, d2 = 1.0, 1.7
    assert _inclusion_exclusion_excess(quadratic, (d1, d2)) == pytest.approx(
        2.0 * photon.beta_x * d1 * d2
    )


@pytest.mark.parametrize("n", [3, 4, 5])
def test_saturated_branch_generates_the_non_decaying_alternating_term(n: int) -> None:
    """The other half: above ``D_t`` the excess neither vanishes nor decays.

    ``psi_n = (-1)^n beta_x D_t^2`` for every ``n >= 2`` once all subsets
    saturate, so truncating the cluster series order by order is not an
    improvement scheme there -- see docs/references/LEM_I_ScholzKraft.md.
    """
    photon = SK1996_V79
    assert photon.D_t is not None
    doses = tuple(40.0 + 17.0 * i for i in range(n))
    expected = (-1) ** n * photon.beta_x * photon.D_t**2
    assert _inclusion_exclusion_excess(photon.log_kill, doses) == pytest.approx(expected)


def test_saturation_radius_is_where_the_track_dose_equals_d_t() -> None:
    lem = _sk1996_lem("scholz1997_two_track")
    ion = Ion(92, 238, 20.0)
    r_sat = lem.saturation_radius_um(ion)
    assert 0.0 < r_sat < lem._radial().r_max_um(ion)
    assert lem._radial().local_dose_Gy(r_sat, ion) == pytest.approx(lem.photon.D_t)


def test_saturation_radius_is_zero_when_the_core_never_reaches_d_t() -> None:
    """A high transition dose puts the whole track in the quadratic branch."""
    lem = LEM1(
        photon=PhotonParameters(alpha_x=0.136, beta_x=0.040, D_t=1.0e6),
        cell=SK1996_CELL,
        stopping_power=SimpleBetheStoppingPower(),
    )
    assert lem.saturation_radius_um(Ion(92, 238, 20.0)) == 0.0
    assert lem.saturated_area_fraction(Ion(92, 238, 20.0), 50.0) == 0.0


def test_saturated_fraction_stays_inside_its_markov_upper_bound() -> None:
    """Single-track discs are a lower bound; ``D / D_t`` is the cheap upper one."""
    lem = _sk1996_lem("scholz1997_two_track")
    assert lem.photon.D_t is not None
    for z, energy, dose in ((6, 20.0, 7.5), (18, 5.0, 24.5), (92, 20.0, 62.0)):
        ion = Ion(z, SK1996_ION_A[z], energy)
        fraction = lem.saturated_area_fraction(ion, dose)
        assert 0.0 <= fraction <= min(1.0, dose / lem.photon.D_t) + 1.0e-12


def test_sk1996_grid_is_quadratic_branch_except_high_energy_uranium() -> None:
    """The certificate: which Fig. 2-4 points the pair integral is complete for.

    A single carbon or argon track saturates under 4% of the nucleus at every
    energy in the paper's range, so ``nu`` there is a pair sum and
    ``beta_model='scholz1997_two_track'`` is missing no 3-body geometry. Uranium
    at 20 MeV/u saturates 20.6% on its own and is the one genuine transition
    case -- the same corner as the open final-slope finding.
    """
    lem = _sk1996_lem("scholz1997_two_track")
    r_eff = lem.cell.effective_nucleus_radius_um()
    nucleus_area = math.pi * r_eff * r_eff

    def single_track_fraction(z: int, energy: float) -> float:
        r_sat = lem.saturation_radius_um(Ion(z, SK1996_ION_A[z], energy))
        return math.pi * min(r_sat, r_eff) ** 2 / nucleus_area

    for z in (6, 18):
        for energy in (1.0, 5.0, 20.0):
            assert single_track_fraction(z, energy) < 0.04

    assert single_track_fraction(92, 20.0) == pytest.approx(0.206, abs=0.01)
    assert single_track_fraction(92, 1.0) < 0.01


# --- analytic radial-dose fast path -----------------------------------------
#
# The MC kernel spends a sqrt, a log, a binary search and an exp per (field point
# x track) to evaluate D(r). Both amorphous track models in the package give it
# in closed form -- constant core, 1/r^2, cut-off -- which makes the same
# estimator ~8x faster. The fast path is selected by MEASURING the sampled
# profile, so a model with any other D(r) falls back on its own.


def test_scholz_kraft_profile_is_detected_as_analytic() -> None:
    lem = _sk1996_lem("scholz1997_two_track")
    ion = Ion(6, 12, 40.0)
    profile = lem._analytic_profile(ion)
    assert profile is not None
    radial = lem._radial()
    assert profile.r_min_um == pytest.approx(radial.r_min_um(ion))
    assert profile.r_max_um == pytest.approx(radial.r_max_um(ion))
    # Scholz & Kraft continue the 1/r^2 law into the core, so it is continuous.
    assert profile.core_dose_Gy == pytest.approx(profile.k_Gy_um2 / profile.r_min_um**2)


def test_kiefer_chatterjee_is_detected_with_an_independent_core() -> None:
    """The core is a free parameter for a reason: KC's is not ``k / r_min^2``."""
    lem = LEM1(
        photon=SK1996_V79,
        cell=SK1996_CELL,
        stopping_power=SimpleBetheStoppingPower(),
        radial_dose=KieferChatterjeeRadialDose(stopping_power=SimpleBetheStoppingPower()),
    )
    profile = lem._analytic_profile(Ion(6, 12, 40.0))
    assert profile is not None
    assert profile.core_dose_Gy != pytest.approx(profile.k_Gy_um2 / profile.r_min_um**2, rel=0.5)


def test_radical_diffusion_profile_measures_its_own_sigma() -> None:
    """A convolved D(r) is covered *outside a radius*, with sigma measured.

    The 2-D Gaussian convolution of ``k/r^2`` has no elementary closed form, but
    its small-sigma expansion does: ``D = (k/r^2)(1 + x + 2x^2)``, ``x = 2
    sigma^2/r^2``. What is asserted here is that ``sigma`` comes out of the
    *sampled profile* -- the code is never told the track model's value -- since
    that is what lets a future convolved model pick the fast path up untold.
    """
    lem = LEM1(
        photon=SK1996_V79,
        cell=SK1996_CELL,
        radial_dose=ScholzKraftRadialDose.lem2(stopping_power=SimpleBetheStoppingPower()),
    )
    ion = Ion(6, 12, 40.0)
    profile = lem._analytic_profile(ion)
    assert profile is not None
    sigma_nm = math.sqrt(profile.two_sigma_sq_um2 / 2.0) * 1e3
    assert sigma_nm == pytest.approx(lem._radial().sigma_nm, rel=0.02)
    # The closed form is a *band*, not the whole profile.
    assert profile.r_switch_um > lem._radial().r_min_um(ion)
    assert profile.kernel_r_edge_um < profile.r_max_um


def test_radical_diffusion_closed_form_matches_the_sampled_profile() -> None:
    """Inside its declared band the expansion must hit the convolution it replaces.

    1e-4 is not the expansion's accuracy but the *sampled* profile's: the
    trapezoid convolution carries ~1e-5 of its own noise, so a tighter assertion
    would be fitting that noise rather than testing the formula.
    """
    lem = LEM1(
        photon=SK1996_V79,
        cell=SK1996_CELL,
        radial_dose=ScholzKraftRadialDose.lem2(stopping_power=SimpleBetheStoppingPower()),
    )
    ion = Ion(6, 12, 40.0)
    profile = lem._analytic_profile(ion)
    assert profile is not None
    r, dose, _, _ = lem._two_track_dose_profile(ion, 600)
    band = profile.valid(r)
    assert profile.dose_Gy(r[band]) == pytest.approx(dose[band], rel=1e-4)

    # And the band has to be worth having. Counting *nodes* would be the wrong
    # measure -- the profile grid starts at r_min/1000, so most of its nodes sit
    # in the core -- but what the kernel cares about is how much of the disc it
    # covers, since the table is consulted for the rest. The two exclusions have
    # very different sizes and are asserted apart so a regression in either is
    # legible: the core is negligible, while the cut-off annulus is ~12 sigma /
    # r_max and so shrinks as the track widens with energy.
    core_fraction = (profile.r_switch_um / profile.r_max_um) ** 2
    edge_fraction = 1.0 - (profile.kernel_r_edge_um / profile.r_max_um) ** 2
    assert core_fraction < 1e-4
    assert edge_fraction < 1e-2


def test_a_profile_that_is_neither_form_is_rejected() -> None:
    """The guard still has to say no. A 1/r^3 tail is not a convolved 1/r^2.

    Without this the closed form would be fitted to anything: ``2 sigma^2`` is a
    free parameter, so some value always exists -- it is the *residual* check
    that has to do the rejecting.
    """
    lem = _sk1996_lem("scholz1997_two_track")
    ion = Ion(6, 12, 40.0)
    r, dose, _r_lo, r_max = lem._two_track_dose_profile(ion, 512)
    r_min = lem._radial().r_min_um(ion)
    bogus = np.where(r < r_min, dose[0], dose[0] * (r_min / np.maximum(r, r_min)) ** 3)
    assert LEM1._convolved_analytic_profile(r, bogus, float(r_min), float(r_max)) is None


def test_analytic_profile_reproduces_the_interpolated_dose() -> None:
    """Away from the ``r_min`` corner the two dose functions must agree exactly."""
    lem = _sk1996_lem("scholz1997_two_track")
    ion = Ion(6, 12, 40.0)
    profile = lem._analytic_profile(ion)
    assert profile is not None
    r, dose, r_lo, r_max = lem._two_track_dose_profile(ion, 512)
    r_min = lem._radial().r_min_um(ion)
    dist = np.concatenate(
        [
            np.geomspace(r_lo, r_min * 0.9, 200),  # flat core
            np.geomspace(r_min * 1.1, r_max * 0.999, 400),  # 1/r^2 tail
            np.geomspace(r_max * 1.001, r_max * 3.0, 20),  # past the cut-off
        ]
    )
    interpolated = lem._two_track_dose_at(dist, r, dose, r_lo, r_max)
    analytic = lem._two_track_dose_at(dist, r, dose, r_lo, r_max, profile)
    assert analytic == pytest.approx(interpolated, rel=1e-9)


def test_analytic_and_interpolated_monte_carlo_agree_on_one_rng_stream() -> None:
    """Same seed, same NumPy engine, only the dose evaluation swapped.

    The two dose evaluations differ only inside the single profile-grid interval
    straddling ``r_min`` -- 2e-7 of the nuclear area, pinned at rel 1e-9 by the
    test above.

    That difference does not stay confined to the dose, which is why the bound is
    1e-4 rather than 1e-12. Weighted
    sampling builds ``nuclear_dose_table`` from the same evaluation, so swapping
    it also perturbs ``d_bar(b)`` and the pre-inverted radial CDF -- the sample
    *positions* move, not just the dose read at them. Measured 7e-6. That is
    still four orders tighter than the ~10% two independent streams would differ
    by at these 60 realizations, so it remains a sharp statement that the streams
    are shared and only perturbed.
    """
    lem = _sk1996_lem("scholz1997_montecarlo")
    ion = Ion(6, 12, 20.0)
    doses = np.array([2.0, 6.0])
    kwargs = {
        "n_realizations": 60,
        "seed": 5,
        "engine": "numpy",
    }
    with_analytic = lem.scholz1997_montecarlo_survival(ion, doses, **kwargs)  # type: ignore[arg-type]
    original = LEM1._analytic_profile
    try:
        LEM1._analytic_profile = lambda self, ion: None  # type: ignore[method-assign,assignment]
        without = lem.scholz1997_montecarlo_survival(ion, doses, **kwargs)  # type: ignore[arg-type]
    finally:
        LEM1._analytic_profile = original  # type: ignore[method-assign]
    assert with_analytic == pytest.approx(without, rel=1e-4)


# --- Kellerer weighted sampling (Kellerer 1985 section VI.A.2) ---------------
def _kellerer_model() -> LEM1:
    return LEM1(photon=CHO_1997, cell=CHO_1997_CELL, beta_model="scholz1997_montecarlo")


def test_initial_slope_respects_the_photon_floor_where_the_track_is_dilute() -> None:
    """``alpha_I >= alpha_x`` in the *unsaturated* regime, on both routes.

    **The floor is not universal.** The tempting argument -- the photon log-kill is convex below
    ``D_t``, so
    Jensen gives ``<h(D)> >= alpha_x <D>`` -- only bounds the *lethal-event*
    average. The cross-section then applies the concave ``1 - exp(-N(b))``, which
    destroys the bound: at extreme LET the exact route legitimately falls far
    below ``alpha_x`` (xenon at 1 MeV/u sits at 0.32 ``alpha_x``, uranium at
    0.20), and that overkill thindown is a feature the model exists to reproduce.

    What *is* required is the dilute limit: where ``N(b) << 1`` the saturation is
    inactive, the track field over the nucleus approaches homogeneity, and the ion
    response must approach the photon one **from above**. Protons and carbon from
    10 MeV/u up are in that regime.
    """
    lem = LEM1(photon=CHO_1997, cell=CHO_1997_CELL, beta_model="scholz1997_two_track")
    for energy in (10.0, 50.0, 100.0, 200.0, 400.0):
        for ion in (Ion(1, 1, energy), Ion(6, 12, energy)):
            for name, alpha in (
                ("exact", lem.initial_slope(ion)),
                ("approximation", lem.scholz1997_central_initial_slope(ion)),
            ):
                assert alpha >= CHO_1997.alpha_x, (
                    f"Z={ion.Z} {energy} MeV/u {name}: "
                    f"alpha={alpha:.4f} below alpha_x={CHO_1997.alpha_x}"
                )


def test_overkill_thindown_takes_alpha_below_the_photon_slope() -> None:
    """The other side of the same coin: at extreme LET the floor must NOT hold.

    Guards the fix for the approximation branch against being "improved" into a
    clamp. ``1 - exp(-N_1)`` saturates at 1 while ``D_1`` keeps growing with LET,
    so alpha thins down as ``1/LET`` -- the experimentally observed behaviour, and
    the reason the halo correction is written to vanish identically (``f = 1``)
    wherever the track fits inside the nucleus.
    """
    lem = LEM1(photon=CHO_1997, cell=CHO_1997_CELL, beta_model="scholz1997_two_track")
    for ion in (Ion(54, 131, 1.0), Ion(92, 238, 1.0)):
        assert lem.traversal_energy_fraction(ion) == 1.0
        assert lem.initial_slope(ion) < CHO_1997.alpha_x
        assert lem.scholz1997_central_initial_slope(ion) == pytest.approx(
            lem.lethal_central_saturating_initial_slope(ion)
        )


def test_approximation_recovers_the_cross_section_once_the_halo_is_restored() -> None:
    """The two routes must converge where the track field is dilute.

    They are computing the same nearly-photon response there, so a gap is a
    bookkeeping error rather than an approximation cost. The bare
    central-traversal form ran 42% low at proton 400 MeV/u because it scored only
    tracks hitting the nucleus while charging them for the whole LET; restoring
    the missed tracks at the photon rate closes it to well under a percent.
    """
    lem = LEM1(photon=CHO_1997, cell=CHO_1997_CELL, beta_model="scholz1997_two_track")
    for energy in (20.0, 100.0, 400.0):
        for ion in (Ion(1, 1, energy), Ion(6, 12, energy)):
            assert lem.scholz1997_central_initial_slope(ion) == pytest.approx(
                lem.initial_slope(ion), rel=0.01
            ), f"Z={ion.Z} {energy} MeV/u"


def test_two_track_beta_warns_instead_of_silently_clamping() -> None:
    """A negative O(phi^2) cluster term must be reported, not returned as 0.0.

    Above ``D_t`` the n-track cluster terms alternate in sign with roughly
    constant magnitude, so truncating at second order is not an improvement
    scheme for a deeply saturated track: the raw two-track beta goes *negative*
    rather than small. Clamping it to zero keeps a physically-signed number, but
    a caller cannot tell the clamped floor apart from a genuine "no curvature"
    result -- and the difference matters, because a sigma taken at 1% survival
    then silently equals the low-dose sigma.

    Measured on the Scholz & Kraft 1996 V79 inputs, where uranium reaches
    ~12 000 keV/um: it fires for uranium from 12 MeV/u up. Carbon at the same
    energy is nowhere near saturation and must stay quiet.
    """
    v79 = PhotonParameters(alpha_x=0.136, beta_x=0.04, D_t=12.0)
    cell = CellTarget(nucleus_radius_um=math.sqrt(50.0 / math.pi))
    lem = LEM1(photon=v79, cell=cell, beta_model="scholz1997_two_track")
    fast = {
        "radial_nodes": 300,
        "separation_nodes": 50,
        "angular_nodes": 200,
        "impact_nodes": 60,
    }

    with pytest.warns(UserWarning, match="clamped to 0"):
        clamped = lem.scholz1997_two_track_beta(Ion(92, 238, 16.0), **fast)
    assert clamped == 0.0

    with warnings.catch_warnings():
        warnings.simplefilter("error")  # any warning here fails the test
        quiet = lem.scholz1997_two_track_beta(Ion(6, 12, 16.0), **fast)
    assert quiet > 0.0


def test_traversal_energy_fraction_is_exactly_one_inside_the_nucleus() -> None:
    """``f`` must be *exactly* 1 when the track fits, not 1 +/- quadrature error.

    It multiplies ``alpha_x`` in the halo term, so a 0.1% quadrature drift there
    would perturb every low-energy alpha the reconstruction has been validated on.
    Computing it as a ratio of two integrals over the same profile makes the
    r_max <= R_eff case identically 1 by construction.
    """
    lem = LEM1(photon=CHO_1997, cell=CHO_1997_CELL, beta_model="scholz1997_two_track")
    for ion in (Ion(6, 12, 1.0), Ion(6, 12, 15.0), Ion(1, 1, 8.0), Ion(92, 238, 1.0)):
        assert lem.traversal_energy_fraction(ion) == 1.0

    # And strictly decreasing once r_max outgrows the nucleus (r_max ~ E^1.7).
    fractions = [lem.traversal_energy_fraction(Ion(1, 1, e)) for e in (20.0, 100.0, 200.0, 400.0)]
    assert all(0.0 < b < a < 1.0 for a, b in itertools.pairwise(fractions))


def test_nuclear_dose_table_integrates_to_the_macroscopic_dose() -> None:
    """``d_bar(b)`` summed over a Poisson field must return the macroscopic dose.

    The nucleus-average dose from one track at impact parameter ``b``, integrated
    against the fluence over all ``b``, is by definition the absorbed dose. This
    pins the table's normalisation independently of any survival calculation.

    **Swept over energy, because a single energy cannot catch the failure this
    guards.** The ``b`` grid has to stay uniform for the compiled kernel to index
    it arithmetically, so its near-field resolution is set by
    ``b_max = r_eff + r_max`` while ``r_max`` grows as ``E^1.7``. A fixed node
    count silently coarsens: at carbon 400 MeV/u 512 nodes leave 1.9
    points across the nucleus, the integral falls to 0.85 D, and the Monte-Carlo
    alpha comes out 20% below the exact cross-section -- while carbon 40 stays
    at 1.001.
    """
    lem = _kellerer_model()
    r_eff = lem.cell.effective_nucleus_radius_um()
    dose_Gy = 3.0
    for z, a, energy in ((6, 12, 40.0), (6, 12, 100.0), (6, 12, 400.0), (1, 1, 200.0)):
        ion = Ion(z, a, energy)
        table = lem.nuclear_dose_table(ion)
        fluence = dose_Gy / (math.pi * r_eff**2 * lem.mean_traversal_dose_Gy(ion))
        macroscopic = fluence * np.trapezoid(
            table.mean_dose_Gy * 2.0 * np.pi * table.b_grid, table.b_grid
        )
        assert macroscopic == pytest.approx(dose_Gy, rel=5e-3), (
            f"dose-table normalisation broken at Z={z} {energy} MeV/u "
            f"({table.b_grid.size} b nodes, "
            f"{r_eff / (table.b_grid[1] - table.b_grid[0]):.1f} across the nucleus)"
        )


def test_nuclear_arc_half_angle_limits() -> None:
    """A track at the centre sees the full circle inside, none outside."""
    from pyradbio.models.lem.base import _nuclear_arc_half_angle

    r_eff = 5.0
    b = np.array([0.0, 0.0, 20.0])
    r = np.array([1.0, 9.0, 1.0])  # inside / outside / far track, no overlap
    got = _nuclear_arc_half_angle(b, r, r_eff)
    assert got == pytest.approx([math.pi, 0.0, 0.0])


def test_kellerer_field_integral_matches_the_exact_nuclear_integral() -> None:
    """The weighted field integral reproduces the exact deterministic nu(b).

    For tracks far enough
    apart that their doses never overlap inside the nucleus, the nuclear integral
    is exactly ``sum_i nu(b_i)`` with ``nu`` the tested deterministic quadrature
    ``_impact_log_kill``. So the target is a number, not another Monte Carlo.

    Separation is what buys exactness here, and it costs no coverage: the
    sampler machinery under test -- dose-proportional source selection, the
    pre-inverted radial draw, the in-nucleus arc azimuth -- is identical either
    way. Overlap is covered end-to-end by the initial-slope test below.
    """
    lem = _kellerer_model()
    ion = Ion(6, 12, 20.0)
    r, dose, r_lo, r_max = lem._two_track_dose_profile(ion, 2048)
    analytic = lem._analytic_profile(ion)
    r_eff = lem.cell.effective_nucleus_radius_um()
    profile = lem._radial_logkill_profile(ion)
    table = lem.nuclear_dose_table(ion)

    # Impact parameters spread past r_eff + r_max/2 so no two tracks can both
    # reach the same point inside the nucleus.
    angles = np.linspace(0.0, 2.0 * math.pi, 5, endpoint=False)
    b = r_eff + r_max * 0.6
    centers = np.stack([b * np.cos(angles), b * np.sin(angles)], axis=1)
    exact = sum(lem._impact_log_kill(b, profile, r_eff, r_max) for _ in angles)
    assert exact > 0.0

    weighted = float(
        np.mean(
            [
                lem.kellerer_field_integral(
                    centers,
                    np.random.default_rng(10 + k),
                    table,
                    r,
                    dose,
                    r_lo,
                    r_max,
                    20_000,
                    analytic,
                )
                for k in range(4)
            ]
        )
    )
    assert weighted == pytest.approx(exact, rel=0.02)


def test_kellerer_field_integral_handles_degenerate_input() -> None:
    lem = _kellerer_model()
    ion = Ion(6, 12, 20.0)
    r, dose, r_lo, r_max = lem._two_track_dose_profile(ion, 512)
    table = lem.nuclear_dose_table(ion)
    rng = np.random.default_rng(0)
    empty = np.zeros((0, 2))
    assert lem.kellerer_field_integral(empty, rng, table, r, dose, r_lo, r_max, 100) == 0.0
    one = np.array([[0.0, 0.0]])
    assert lem.kellerer_field_integral(one, rng, table, r, dose, r_lo, r_max, 0) == 0.0


def test_b_cut_split_does_not_shift_the_survival_curve() -> None:
    """``b_cut_um`` is a cost knob, not a physics knob; S(D) must not move.

    What is worth pinning at the survival level is that resolving every track and replacing the far
    field by
    its compound-Poisson mean give the same curve.

    The tolerance is set by the history budget, not by how well the two agree:
    independent runs at ``N`` realizations differ by a binomial
    ``sqrt(2 S (1-S)/N)``, ~6% of ``S`` here. The estimator itself is pinned far
    more tightly against exact truth by
    `test_kellerer_field_integral_matches_the_exact_nuclear_integral` and
    `test_kellerer_accumulate_numba_reproduces_the_exact_initial_slope`.
    """
    lem = _kellerer_model()
    ion = Ion(6, 12, 20.0)
    doses = np.array([1.0, 2.0])
    resolved = lem.scholz1997_montecarlo_survival(
        ion, doses, n_realizations=2000, n_field_points=3000, seed=5, engine="numpy"
    )
    split = lem.scholz1997_montecarlo_survival(
        ion,
        doses,
        n_realizations=2000,
        n_field_points=3000,
        seed=5,
        b_cut_um=10.0,
        engine="numpy",
    )
    assert split == pytest.approx(resolved, rel=0.15)


def test_near_far_split_does_not_shift_the_field_integral() -> None:
    """b_cut only replaces distant tracks by their mean; nu must not move."""
    lem = _kellerer_model()
    ion = Ion(6, 12, 40.0)
    r, dose, r_lo, r_max = lem._two_track_dose_profile(ion, 2048)
    analytic = lem._analytic_profile(ion)
    table = lem.nuclear_dose_table(ion)
    b_max = lem.cell.effective_nucleus_radius_um() + r_max

    rng = np.random.default_rng(11)
    n = 900
    rad = b_max * np.sqrt(rng.random(n))
    ang = 2.0 * math.pi * rng.random(n)
    centers = np.stack([rad * np.cos(ang), rad * np.sin(ang)], axis=1)

    def run(b_cut: float | None) -> float:
        return float(
            np.mean(
                [
                    lem.kellerer_field_integral(
                        centers,
                        np.random.default_rng(50 + k),
                        table,
                        r,
                        dose,
                        r_lo,
                        r_max,
                        20_000,
                        analytic,
                        b_cut,
                    )
                    for k in range(4)
                ]
            )
        )

    assert run(10.0) == pytest.approx(run(None), rel=0.02)


def test_kellerer_numba_kernel_matches_the_numpy_reference() -> None:
    """The compiled ratio sum must reproduce the NumPy loop exactly.

    Unlike ``mc_accumulate``, this kernel is deterministic for a fixed seed: the
    sampling stays in NumPy and only the (order-independent up to floating-point
    reassociation) ratio sum is compiled, so the two agree to round-off.
    """
    pytest.importorskip("numba")
    lem = _kellerer_model()
    ion = Ion(6, 12, 40.0)
    r, dose, r_lo, r_max = lem._two_track_dose_profile(ion, 2048)
    analytic = lem._analytic_profile(ion)
    table = lem.nuclear_dose_table(ion)
    b_max = lem.cell.effective_nucleus_radius_um() + r_max

    rng = np.random.default_rng(99)
    n = 600
    rad = b_max * np.sqrt(rng.random(n))
    ang = 2.0 * math.pi * rng.random(n)
    centers = np.stack([rad * np.cos(ang), rad * np.sin(ang)], axis=1)

    kwargs = dict(analytic=analytic, b_cut_um=10.0)
    by_numpy = lem.kellerer_field_integral(
        centers,
        np.random.default_rng(4),
        table,
        r,
        dose,
        r_lo,
        r_max,
        5000,
        engine="numpy",
        **kwargs,
    )
    by_numba = lem.kellerer_field_integral(
        centers,
        np.random.default_rng(4),
        table,
        r,
        dose,
        r_lo,
        r_max,
        5000,
        engine="numba",
        **kwargs,
    )
    assert by_numba == pytest.approx(by_numpy, rel=1e-9)


def test_survival_runs_past_the_old_traversal_ceiling() -> None:
    """No ``n_bar`` ceiling applies to the weighted sampler.

    Carbon 40 has ``n_bar ~ 916`` at 2.5 Gy, above the 800 of
    ``_MC_MAX_MEAN_TRAVERSALS``. A uniform sampler would thin to under one sample
    per track there and return ``beta > 2 beta_x``; weighted sampling has no such
    failure mode, and the constant applies only to the traversal-conditioned
    estimator.
    """
    lem = _kellerer_model()
    ion = Ion(6, 12, 40.0)
    survival = lem.scholz1997_montecarlo_survival(
        ion,
        [2.5],
        n_realizations=8,
        n_field_points=500,
        b_cut_um=10.0,
    )
    assert 0.0 < float(survival[0]) < 1.0


def test_prefix_field_integral_numba_matches_numpy() -> None:
    """The compiled prefix kernel must reproduce the NumPy block exactly.

    It carries a running dose and log-kill instead of materialising the
    (points, tracks) block and cumulative-summing it, so this pins that the
    reassociation costs nothing: they agree to round-off, not statistically.
    """
    pytest.importorskip("numba")
    lem = LEM1(photon=V79, cell=V79_CELL, beta_model="scholz1997_montecarlo")
    ion = Ion(6, 12, 5.0)
    r, dose, r_lo, r_max = lem._two_track_dose_profile(ion, 2048)
    analytic = lem._analytic_profile(ion)
    r_eff = lem.cell.effective_nucleus_radius_um()
    b_max = r_eff + r_max
    profile = lem._radial_logkill_profile(ion)
    b_grid = np.linspace(0.0, b_max, 512, dtype=np.float64)
    nu_grid = np.array(
        [lem._impact_log_kill(float(b), profile, r_eff, r_max) for b in b_grid],
        dtype=np.float64,
    )
    rng = np.random.default_rng(5)
    n = 120
    rad = b_max * np.sqrt(rng.random(n))
    ang = 2.0 * math.pi * rng.random(n)
    centers = np.stack([rad * np.cos(ang), rad * np.sin(ang)], axis=1)

    args = (centers, None, r, dose, r_lo, r_max, 6000, 0.7, b_grid, nu_grid, analytic)
    by_numpy = lem._montecarlo_prefix_field_integrals(
        *args[:1], np.random.default_rng(3), *args[2:], "numpy"
    )
    by_numba = lem._montecarlo_prefix_field_integrals(
        *args[:1], np.random.default_rng(3), *args[2:], "numba"
    )
    assert by_numba == pytest.approx(by_numpy, rel=1e-12)


def test_kellerer_accumulate_numba_reproduces_the_exact_initial_slope() -> None:
    """The fused kernel's sampler geometry, checked against deterministic truth.

    Unlike ``kellerer_ratio_sum``, ``kellerer_accumulate`` does its own sampling,
    so it cannot be compared to the NumPy path point-for-point -- numba seeds one
    generator per thread and the streams differ. What *can* be pinned exactly is
    the answer: at low dose ``-ln S / D`` converges on ``LEM1.initial_slope``,
    which is the exact deterministic nuclear integral. That number is sensitive
    to every piece the kernel newly implements -- the dose-proportional source
    selection, the pre-inverted radial draw, the arc azimuth and the uniform-grid
    index arithmetic -- so a geometry bug cannot hide behind it.

    The 3% band is the measured spread: at these settings the estimate moves by
    up to 0.9% between independent runs, and 0.25 Gy still carries a little
    ``beta D`` curvature on top.
    """
    pytest.importorskip("numba")
    lem = _kellerer_model()
    ion = Ion(6, 12, 20.0)
    dose = 0.25
    survival = lem.scholz1997_montecarlo_survival(
        ion,
        [dose],
        n_realizations=20_000,
        n_field_points=1500,
        seed=3,
        engine="numba",
    )
    alpha_mc = -math.log(float(survival[0])) / dose
    assert alpha_mc == pytest.approx(lem.initial_slope(ion), rel=0.03)


def test_kellerer_accumulate_numba_agrees_with_the_numpy_loop() -> None:
    """Compiled and interpreted weighted accumulators must estimate the same S(D).

    Independent streams, so the tolerance is a history budget: at ``N = 4000``
    two runs differ by a binomial ``sqrt(2 S (1-S) / N)``, which is ~3% of ``S``
    at ``S ~ 0.3``. Anything structurally wrong in the fused kernel moves ``S``
    far further than that -- the deterministic pin is the initial-slope test
    above.
    """
    pytest.importorskip("numba")
    lem = _kellerer_model()
    ion = Ion(6, 12, 20.0)
    doses = [1.5]
    kwargs = dict(
        n_realizations=4000,
        n_field_points=2000,
        seed=7,
    )
    by_numpy = lem.scholz1997_montecarlo_survival(ion, doses, engine="numpy", **kwargs)  # type: ignore[arg-type]
    by_numba = lem.scholz1997_montecarlo_survival(ion, doses, engine="numba", **kwargs)  # type: ignore[arg-type]
    assert by_numba == pytest.approx(by_numpy, rel=0.08)


def test_kellerer_accumulate_falls_back_on_a_non_uniform_impact_grid() -> None:
    """A hand-built table with unequal ``b`` spacing must take the NumPy path.

    The kernel indexes ``b_grid`` arithmetically, which is only valid for the
    ``linspace`` grid ``nuclear_dose_table`` builds. The routing is observable
    without touching the physics: the NumPy loop consumes the caller's seeded
    ``Generator`` and so repeats exactly, while the kernel draws from numba's own
    unseeded per-thread state and cannot. (The perturbed table is not physically
    consistent -- its dose column now belongs to different impact parameters --
    but this test asserts only where the call goes.)
    """
    pytest.importorskip("numba")
    lem = _kellerer_model()
    ion = Ion(6, 12, 20.0)
    table = lem.nuclear_dose_table(ion)
    r, dose, r_lo, r_max = lem._two_track_dose_profile(ion, 2048)
    analytic = lem._analytic_profile(ion)
    b_max = lem.cell.effective_nucleus_radius_um() + r_max

    def accumulate(tbl: NuclearDoseTable) -> tuple[float, float]:
        return lem._mc_accumulate_kellerer(
            400,
            60.0,
            b_max,
            tbl,
            r,
            dose,
            r_lo,
            r_max,
            600,
            np.random.default_rng(21),
            analytic,
            None,
            "numba",
        )

    skewed = dataclasses.replace(
        table, b_grid=np.linspace(0.0, math.sqrt(b_max), table.b_grid.size) ** 2
    )
    assert accumulate(skewed) == accumulate(skewed)  # seeded NumPy path: exact repeat
    assert accumulate(table) != accumulate(table)  # compiled path: fresh stream


def test_iso_survival_rbe_from_final_slope_inverts_the_cross_section() -> None:
    """The sigma -> RBE construction must invert the sigma definition exactly.

    ``estimate_final_slope_cross_section`` maps a log-kill slope to ``sigma``;
    this maps ``sigma`` back to a dose and hence an RBE. Feeding it a sigma built
    from a known slope must return the RBE that slope implies, so the only
    approximation left is the documented one -- that the ion curve has no
    shoulder at the survival level.
    """
    from pyradbio.benchmark.scholz_kraft1996 import iso_survival_rbe_from_final_slope

    let = np.array([50.0, 200.0, 800.0])
    slope = np.array([0.9, 0.65, 0.38])  # d(-ln S)/dD at 1%, Gy^-1
    sigma = LEM_D1_PREFACTOR * let * slope / WATER_DENSITY_G_CM3
    reference_dose = 9.164  # V79 photon dose at 1% survival

    rbe = iso_survival_rbe_from_final_slope(sigma, let, reference_dose_Gy=reference_dose)
    expected = reference_dose / (-math.log(0.01) / slope)
    assert rbe == pytest.approx(expected, rel=1e-12)


def test_iso_survival_rbe_from_final_slope_overestimates_by_the_shoulder() -> None:
    """A shouldered curve must come back high by exactly ``1 + c / -ln(S)``.

    This pins the docstring's error model rather than a magnitude: writing the
    ion curve as ``-ln S = slope * D - c``, the construction misses the intercept
    ``c`` and so returns a dose short by ``c / slope``. Anything else means the
    quoted per-ion over-estimates are describing the wrong mechanism.
    """
    from pyradbio.benchmark.scholz_kraft1996 import iso_survival_rbe_from_final_slope

    let, slope, offset = 20.0, 0.75, 1.2  # c = 1.2 -> a real shoulder at 1%
    sigma = LEM_D1_PREFACTOR * let * slope / WATER_DENSITY_G_CM3
    reference_dose = 9.164

    true_dose = (-math.log(0.01) + offset) / slope  # what the shouldered curve needs
    derived = float(
        iso_survival_rbe_from_final_slope([sigma], [let], reference_dose_Gy=reference_dose)[0]
    )
    assert derived / (reference_dose / true_dose) == pytest.approx(
        1.0 + offset / -math.log(0.01), rel=1e-12
    )


def test_specific_energy_survival_recovers_the_z_domain_slope_at_low_energy() -> None:
    """At carbon 1 MeV/u the exact z-conditioned slope must match the fast path.

    There ``beta_z = (s_max - alpha_z)/(2 D_t)`` clips to zero, so the paper's
    approximation is the flat line ``alpha_z``. A flat exact slope is therefore
    the model's own answer and not a coincidence, which makes it the one place
    the two paths can be compared without the approximation's error confounding
    it -- and so the sharpest available check that the z binning, the
    ``z = sum_i d_bar(b_i)`` accounting and the survival average all line up.
    """
    lem = _kellerer_model()
    ion = Ion(6, 12, 1.0)
    z_lq = lem.scholz1997_specific_energy_lq(ion)
    assert z_lq.beta == 0.0  # the premise above

    # n_poisson_means well below the default: the default is tuned for shape
    # fidelity in a batch, and this test only needs the value to be right.
    # Pinned to numpy: the numba kernel does not repeat for a seed, and at these
    # settings its worst bin crosses the 5% bound below in about 1 run in 20.
    curve = lem.scholz1997_specific_energy_survival(
        ion,
        z_max_Gy=30.0,
        n_poisson_means=12,
        realizations_per_mean=900,
        b_cut_um=10.0,
        seed=5,
        engine="numpy",
    )
    assert curve.z_Gy.size >= 6
    assert np.all(curve.counts >= 60)
    assert np.all(curve.log_kill_err > 0.0)
    assert np.all(curve.log_kill_err < 0.05 * curve.log_kill)  # errors must be small
    assert np.all(np.diff(curve.z_Gy) > 0.0)
    assert curve.slope_Gy_inv() == pytest.approx(z_lq.alpha, rel=0.05)


def test_specific_energy_survival_rejects_an_unreachable_binning() -> None:
    """Too few histories per bin must raise rather than return a sparse curve.

    Silently dropping bins would leave a curve whose ``np.gradient`` spans
    whatever survived, which reads as a real slope.
    """
    lem = _kellerer_model()
    with pytest.raises(ValueError, match="min_bin_count"):
        lem.scholz1997_specific_energy_survival(
            Ion(6, 12, 1.0),
            z_max_Gy=30.0,
            n_poisson_means=2,
            realizations_per_mean=20,
            min_bin_count=500,
            seed=1,
        )


def test_specific_energy_survival_numpy_and_numba_agree() -> None:
    """The compiled and interpreted z samplers must estimate the same S(z).

    ``_z_samples`` grew a NumPy branch alongside the kernel, and the two build
    the far-field background and the near-track z sum independently. Independent
    RNG streams, so this is a statistical check; the deterministic pin is
    `test_specific_energy_survival_recovers_the_z_domain_slope_at_low_energy`.
    """
    pytest.importorskip("numba")
    lem = _kellerer_model()
    ion = Ion(6, 12, 8.0)
    kwargs = dict(z_max_Gy=30.0, n_bins=6, realizations_per_mean=700, b_cut_um=10.0, seed=4)
    by_numpy = lem.scholz1997_specific_energy_survival(ion, engine="numpy", **kwargs)  # type: ignore[arg-type]
    by_numba = lem.scholz1997_specific_energy_survival(ion, engine="numba", **kwargs)  # type: ignore[arg-type]

    assert by_numba.z_Gy == pytest.approx(by_numpy.z_Gy, rel=0.05)
    assert by_numba.log_kill == pytest.approx(by_numpy.log_kill, rel=0.08)


def test_specific_energy_straightness_detects_a_planted_cubic() -> None:
    """``straightness`` must flag curvature that is really there, and only that.

    Scholz 1997 Fig. 4 draws straight lines, so "is our slope straight" is a
    load-bearing question and the test for it has to be able to fail. A quadratic
    ``-ln S(z)`` means a straight slope; adding a cubic term far above the error
    bars must show up, and removing it must not.
    """
    z = np.linspace(2.0, 30.0, 14)
    err = np.full_like(z, 0.01)
    quadratic = 0.6 * z + 0.02 * z**2

    straight = SpecificEnergySurvival(
        z_Gy=z, log_kill=quadratic, log_kill_err=err, counts=np.full(z.size, 500, dtype=np.int64)
    )
    cubic_sigma, chi2 = straight.straightness()
    assert abs(cubic_sigma) < 3.0
    assert chi2 < 3.0

    bent = SpecificEnergySurvival(
        z_Gy=z,
        log_kill=quadratic + 1.0e-3 * z**3,
        log_kill_err=err,
        counts=np.full(z.size, 500, dtype=np.int64),
    )
    assert abs(bent.straightness()[0]) > 10.0
