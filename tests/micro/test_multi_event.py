"""Multi-event specific-energy densities ``f(z; D)`` (Sato & Furusawa 2012).

Written before the implementation. The independent references are analytic:

* a sphere's chord-length distribution under uniform-isotropic incidence is
  ``f(l) ~ l``, so the single-event specific-energy density is ``f(z) = 2z/z_max^2``
  on ``[0, z_max]`` -- whose frequency and dose means are exactly ``(2/3) z_max``
  and ``(3/4) z_max``. That pins the single-event side with no free constants.
* the multi-event density is a **compound Poisson** sum, so its mean is ``D``
  exactly (the paper's eq. 8) and its variance is ``D * z_D`` -- both closed form,
  and the variance identity is the same one the shipped Taylor ``gamma`` rests on.
* and a direct Monte-Carlo draw of the same compound Poisson, which shares no
  algebra with the transform the implementation uses.
"""

from __future__ import annotations

import math

import numpy as np
import pytest

from pyradbio.micro import multi_event as me

R_N_UM = 6.2  # Sato & Furusawa 2012 fig. A1 / table 1 (HSG)


# --------------------------------------------------------------------------
# eq. (25): LET -> specific energy for a central traversal
# --------------------------------------------------------------------------
def _eta_reference_Gy_per_keV_um(radius_um: float) -> float:
    """``eta`` from first principles: energy over mass for a central chord.

    A central chord of a sphere is ``2r`` long, so ``E = L * 2r`` [keV] and
    ``m = (4/3) pi r^3 rho``. Written out here in SI rather than reusing the
    paper's pre-collapsed constant.
    """
    joule_per_keV = 1.602176634e-16
    energy_J = 2.0 * radius_um * joule_per_keV  # per (keV/um) of LET
    mass_kg = (4.0 / 3.0) * math.pi * (radius_um * 1e-6) ** 3 * 1000.0
    return energy_J / mass_kg


def test_central_specific_energy_matches_energy_over_mass() -> None:
    got = me.let_to_central_specific_energy_Gy(1.0, radius_um=R_N_UM)
    assert got == pytest.approx(_eta_reference_Gy_per_keV_um(R_N_UM), rel=1e-9)


def test_central_specific_energy_scales_as_let_over_radius_squared() -> None:
    base = me.let_to_central_specific_energy_Gy(100.0, radius_um=R_N_UM)
    assert me.let_to_central_specific_energy_Gy(200.0, radius_um=R_N_UM) == pytest.approx(
        2.0 * base, rel=1e-12
    )
    assert me.let_to_central_specific_energy_Gy(100.0, radius_um=2.0 * R_N_UM) == pytest.approx(
        base / 4.0, rel=1e-12
    )


@pytest.mark.parametrize("bad", [0.0, -1.0])
def test_central_specific_energy_rejects_a_bad_radius(bad: float) -> None:
    with pytest.raises(ValueError):
        me.let_to_central_specific_energy_Gy(100.0, radius_um=bad)


# --------------------------------------------------------------------------
# eq. (24): the Fermi-smoothed sphere chord density
# --------------------------------------------------------------------------
def test_single_event_density_normalises_to_one() -> None:
    _z, p = me.single_event_density(let_keV_um=200.0, radius_um=R_N_UM)
    assert float(np.sum(p)) == pytest.approx(1.0, rel=1e-9)


def test_single_event_density_recovers_the_sphere_chord_limit() -> None:
    """With the Fermi width switched off, ``f(z) = 2z/z_max^2`` exactly.

    Its frequency mean is ``(2/3) z_max`` and its dose mean ``(3/4) z_max`` --
    the classical sphere results, which fix both moments with no free constant.
    """
    z_max = me.let_to_central_specific_energy_Gy(200.0, radius_um=R_N_UM)
    z, p = me.single_event_density(let_keV_um=200.0, radius_um=R_N_UM, fermi_width_fraction=0.0)
    z_F = float(np.sum(z * p))
    z_D = float(np.sum(z * z * p) / z_F)
    assert z_F == pytest.approx(2.0 / 3.0 * z_max, rel=2e-3)
    assert z_D == pytest.approx(3.0 / 4.0 * z_max, rel=2e-3)


def test_fermi_smearing_costs_more_than_the_paper_implies() -> None:
    """``c = L*eta/10`` is not a cosmetic rounding of the edge at ``z_max``.

    Measured at LET 200 keV/um, ``r_n = 6.2 um``: **15% of the single-event mass
    sits above ``z_max``**, the specific energy of a *central* traversal, which is
    the largest a chord can geometrically deliver. The Fermi form has an
    unbounded exponential tail, so that mass is straggling by construction, not a
    numerical artefact -- but 15% is worth knowing before calling it small.

    Sato & Furusawa 2012 describe the straggling as "not so large ... and
    insignificant in the final results of the calculation". That claim is about
    the final **survival**, checked against PHITS with delta rays above 1 keV --
    it is not a claim that the moments are unchanged, and they are not (see
    `test_fermi_smearing_shifts_the_means_by_several_percent`).
    """
    z_max = me.let_to_central_specific_energy_Gy(200.0, radius_um=R_N_UM)
    z, p = me.single_event_density(let_keV_um=200.0, radius_um=R_N_UM)
    beyond = float(np.sum(p[z > z_max]))
    assert beyond == pytest.approx(0.150, abs=0.01)


def test_fermi_smearing_shifts_the_means_by_several_percent() -> None:
    """The straggling lifts both means; pinned so a change of ``c`` is visible.

    ``z_F`` goes 0.667 -> 0.709 of ``z_max`` (+6.4%) and ``z_D`` 0.750 -> 0.820
    (+9.4%). Since ``z_D`` is what drives the multi-event variance, and the
    variance is what the whole stochastic correction rests on, this is the number
    to look at first if the numerical route ever disagrees with the paper.
    """
    z_max = me.let_to_central_specific_energy_Gy(200.0, radius_um=R_N_UM)

    def means(fraction: float) -> tuple[float, float]:
        z, p = me.single_event_density(
            let_keV_um=200.0, radius_um=R_N_UM, fermi_width_fraction=fraction
        )
        z_F = float(np.sum(z * p))
        return z_F / z_max, float(np.sum(z * z * p) / z_F) / z_max

    sharp_F, sharp_D = means(0.0)
    soft_F, soft_D = means(me.DEFAULT_FERMI_WIDTH_FRACTION)
    assert (sharp_F, sharp_D) == pytest.approx((2.0 / 3.0, 3.0 / 4.0), rel=2e-3)
    assert soft_F == pytest.approx(0.709, abs=0.005)
    assert soft_D == pytest.approx(0.820, abs=0.005)


# --------------------------------------------------------------------------
# eqs. (26)-(28): the compound-Poisson multi-event density
# --------------------------------------------------------------------------
def _single_event(let_keV_um: float = 200.0) -> tuple[np.ndarray, np.ndarray]:
    return me.single_event_density(let_keV_um=let_keV_um, radius_um=R_N_UM)


def test_multi_event_density_normalises_to_one() -> None:
    z1, p1 = _single_event()
    _z, p = me.multi_event_density(z1, p1, dose_Gy=2.0)
    assert float(np.sum(p)) == pytest.approx(1.0, rel=1e-9)


def test_multi_event_mean_is_the_absorbed_dose() -> None:
    """Eq. (8): ``integral z_n f_n(z_n, D) dz_n = D``, exactly."""
    z1, p1 = _single_event()
    for dose in (0.3, 2.0, 11.4):
        z, p = me.multi_event_density(z1, p1, dose_Gy=dose)
        assert float(np.sum(z * p)) == pytest.approx(dose, rel=1e-6)


def test_multi_event_variance_is_dose_times_the_dose_mean() -> None:
    """Compound Poisson: ``Var(z_n) = D * z_D``.

    This is the same identity the shipped Taylor ``gamma`` is built on, so the
    numerical route must reproduce it or the two are not the same model.
    """
    z1, p1 = _single_event()
    z_F = float(np.sum(z1 * p1))
    z_D = float(np.sum(z1 * z1 * p1) / z_F)
    for dose in (0.5, 2.0, 8.0):
        z, p = me.multi_event_density(z1, p1, dose_Gy=dose)
        mean = float(np.sum(z * p))
        var = float(np.sum((z - mean) ** 2 * p))
        assert var == pytest.approx(dose * z_D, rel=5e-3)


def test_multi_event_keeps_the_non_hit_mass_at_zero() -> None:
    """A fraction ``exp(-lambda)`` of nuclei are missed entirely."""
    z1, p1 = _single_event()
    z_F = float(np.sum(z1 * p1))
    dose = 0.05
    lam = dose / z_F
    _z, p = me.multi_event_density(z1, p1, dose_Gy=dose)
    assert p[0] == pytest.approx(math.exp(-lam), rel=1e-3)


def test_multi_event_matches_a_monte_carlo_of_the_same_compound_poisson() -> None:
    """Independent check: sample the sum, do not transform it."""
    z1, p1 = _single_event()
    z_F = float(np.sum(z1 * p1))
    dose = 3.0
    lam = dose / z_F

    rng = np.random.default_rng(20260827)
    n = 400_000
    counts = rng.poisson(lam, size=n)
    draws = rng.choice(z1, size=(n, int(counts.max()) + 1), p=p1)
    mask = np.arange(draws.shape[1])[None, :] < counts[:, None]
    sampled = (draws * mask).sum(axis=1)

    z, p = me.multi_event_density(z1, p1, dose_Gy=dose)
    mean_exact = float(np.sum(z * p))
    var_exact = float(np.sum((z - mean_exact) ** 2 * p))
    sem = math.sqrt(var_exact / n)
    assert abs(float(sampled.mean()) - mean_exact) < 4.0 * sem
    assert float(sampled.var()) == pytest.approx(var_exact, rel=0.02)


def test_multi_event_becomes_gaussian_at_many_events() -> None:
    """Large ``lambda``: skewness falls off like ``1/sqrt(lambda)``."""
    z1, p1 = _single_event()

    def skew(dose: float) -> float:
        z, p = me.multi_event_density(z1, p1, dose_Gy=dose)
        mean = float(np.sum(z * p))
        var = float(np.sum((z - mean) ** 2 * p))
        return float(np.sum((z - mean) ** 3 * p)) / var**1.5

    assert abs(skew(50.0)) < abs(skew(5.0)) < abs(skew(0.5))


def test_multi_event_survives_a_large_lambda() -> None:
    """The paper switches to a database above lambda = 50; the transform does not.

    ``exp(lambda (F - 1))`` is closed form in the transform domain, so there is no
    iterated convolution to truncate and no separate high-dose branch.
    """
    z1, p1 = _single_event(let_keV_um=10.0)
    z, p = me.multi_event_density(z1, p1, dose_Gy=100.0)
    assert float(np.sum(p)) == pytest.approx(1.0, rel=1e-6)
    assert float(np.sum(z * p)) == pytest.approx(100.0, rel=1e-4)


def test_multi_event_rejects_a_negative_dose() -> None:
    z1, p1 = _single_event()
    with pytest.raises(ValueError):
        me.multi_event_density(z1, p1, dose_Gy=-1.0)


def test_multi_event_at_zero_dose_is_all_at_zero() -> None:
    z1, p1 = _single_event()
    _z, p = me.multi_event_density(z1, p1, dose_Gy=0.0)
    assert p[0] == pytest.approx(1.0, rel=1e-12)
    assert float(np.sum(p[1:])) == pytest.approx(0.0, abs=1e-12)


# --------------------------------------------------------------------------
# Validation -- every guard is reachable and says what is wrong
# --------------------------------------------------------------------------
@pytest.mark.parametrize("bad", [-1.0, math.nan])
def test_single_event_rejects_a_bad_let(bad: float) -> None:
    with pytest.raises(ValueError, match="let_keV_um"):
        me.single_event_density(let_keV_um=bad, radius_um=R_N_UM)


def test_single_event_rejects_zero_let() -> None:
    """A zero-LET traversal deposits nothing, so there is no density to build."""
    with pytest.raises(ValueError, match="deposits nothing"):
        me.single_event_density(let_keV_um=0.0, radius_um=R_N_UM)


@pytest.mark.parametrize("bad", [-0.1, math.nan])
def test_single_event_rejects_a_bad_fermi_width(bad: float) -> None:
    with pytest.raises(ValueError, match="fermi_width_fraction"):
        me.single_event_density(let_keV_um=200.0, radius_um=R_N_UM, fermi_width_fraction=bad)


def test_single_event_rejects_too_few_nodes() -> None:
    with pytest.raises(ValueError, match="nodes"):
        me.single_event_density(let_keV_um=200.0, radius_um=R_N_UM, nodes=4)


def test_multi_event_rejects_mismatched_arrays() -> None:
    with pytest.raises(ValueError, match="equal length"):
        me.multi_event_density(np.array([0.0, 1.0, 2.0]), np.array([0.5, 0.5]), dose_Gy=1.0)


def test_multi_event_rejects_a_single_point() -> None:
    with pytest.raises(ValueError, match="at least two"):
        me.multi_event_density(np.array([0.0]), np.array([1.0]), dose_Gy=1.0)


def test_multi_event_rejects_a_non_uniform_grid() -> None:
    """The convolution is a product only on a uniform grid."""
    z = np.array([0.0, 1.0, 3.0, 7.0])
    p = np.full(4, 0.25)
    with pytest.raises(ValueError, match="uniform grid"):
        me.multi_event_density(z, p, dose_Gy=1.0)


def test_multi_event_rejects_a_grid_not_starting_at_zero() -> None:
    """Cell 0 must be z = 0 or the no-hit point mass lands in the wrong bin."""
    z = np.array([0.5, 1.0, 1.5, 2.0])
    p = np.full(4, 0.25)
    with pytest.raises(ValueError, match="start at 0"):
        me.multi_event_density(z, p, dose_Gy=1.0)


def test_multi_event_rejects_a_zero_mean_single_event_density() -> None:
    """All mass at z = 0 means lambda is undefined."""
    z = np.linspace(0.0, 1.0, 8)
    p = np.zeros(8)
    p[0] = 1.0
    with pytest.raises(ValueError, match="zero mean"):
        me.multi_event_density(z, p, dose_Gy=1.0)
