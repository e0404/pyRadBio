"""IMK -- integrated microdosimetric-kinetic model (Matsuya et al. 2025).

Physics acceptance for `pyradbio.models.mkm.imk.IMKModel`
(doi:10.1002/mp.70040; equations restated in
``docs/references/IMK_Matsuya2025.md``):

* ``alpha`` is *identical* to the mMKM ``alpha`` -- IMK's eq. (1) linear term is
  ``alpha0 + z*_1D beta0``, unchanged from the modified MKM;
* ``beta = F beta0`` with ``F`` the Lea-Catcheside time factor written in terms
  of the repair-rate constant ``(a+c)`` instead of a half-time, so an acute
  irradiation collapses the whole model onto mMKM exactly;
* ``F`` for one continuous block reproduces the paper's printed closed form
  ``2[(a+c)T + e^-((a+c)T) - 1]/((a+c)T)^2`` -- computed here from the general
  `pyradbio.core.protraction` route, so the two derivations cross-check;
* the micronucleus channel (eq. 3) is the survival LQ scaled by ``h``, and
  ``RBE_MN`` (eq. 6) is therefore independent of ``h``;
* eq. (4)'s ``D10`` is the LQ inversion the core already owns.
"""

from __future__ import annotations

import math
from itertools import pairwise

import numpy as np
import pytest

from pyradbio.core.lq import LQParameters, dose_for_survival, log_kill, rbe, survival
from pyradbio.core.photon import PhotonParameters
from pyradbio.core.protraction import (
    DoseSchedule,
    lea_catcheside_g,
    repair_half_time_h_from_rate,
    repair_rate_per_h,
)
from pyradbio.core.radiation import Ion
from pyradbio.models.mkm.imk import (
    RECOVERED_PHOTON_Z_STAR_1D_GY,
    IMKModel,
    matsuya2025_h_values,
)
from pyradbio.models.mkm.spectrum_models import ModifiedMKM

# Matsuya et al. 2025, Med. Phys. 52:e70040, table 1 (HSG column).
HSG_ALPHA0 = 0.150
HSG_BETA0 = 0.0467
HSG_A_PLUS_C = 2.19
HSG_RD_UM = 0.42
H_GLOBAL = 0.275
Y0_KEV_UM = 150.0


def hsg() -> IMKModel:
    return IMKModel.hsg_matsuya2025()


# --- published parameters (table 1, sec. 4.1) --------------------------------


def test_hsg_parameters_match_table_1() -> None:
    model = hsg()
    assert model.alpha0 == pytest.approx(HSG_ALPHA0)
    assert model.beta0 == pytest.approx(HSG_BETA0)
    assert model.repair_rate_per_h == pytest.approx(HSG_A_PLUS_C)
    assert model.domain_radius_um == pytest.approx(HSG_RD_UM)
    assert model.mn_probability == pytest.approx(H_GLOBAL)


def test_scc_shares_hsg_geometry_and_repair_rate() -> None:
    """Table 1: SCC differs from HSG only in ``alpha0``/``beta0``."""
    scc = IMKModel.scc_matsuya2025()
    assert scc.alpha0 == pytest.approx(0.0825)
    assert scc.beta0 == pytest.approx(0.0298)
    assert scc.repair_rate_per_h == pytest.approx(2.19)
    assert scc.domain_radius_um == pytest.approx(HSG_RD_UM)


def test_x_ray_only_cell_lines_use_the_larger_domain() -> None:
    """Table 1: WI-38 and DU145 were fitted with ``r_d = 0.50 um``."""
    for model, alpha0, beta0, rate in (
        (IMKModel.wi38_matsuya2025(), 0.590, 0.0163, 0.371),
        (IMKModel.du145_matsuya2025(), 0.0774, 0.0380, 2.10),
    ):
        assert model.alpha0 == pytest.approx(alpha0)
        assert model.beta0 == pytest.approx(beta0)
        assert model.repair_rate_per_h == pytest.approx(rate)
        assert model.domain_radius_um == pytest.approx(0.50)


def test_saturation_parameter_is_the_papers_150_keV_um() -> None:
    """Sec. 4.1 prose: ``y0 = 150 keV/um``, *not* the geometric default."""
    assert hsg().y0_keV_um == pytest.approx(Y0_KEV_UM)
    assert IMKModel.scc_matsuya2025().y0_keV_um == pytest.approx(Y0_KEV_UM)


def test_table_2_per_cell_line_h_values() -> None:
    """Table 2 -- offered by the paper, not used for its own predictions."""
    h = matsuya2025_h_values()
    assert h["CHO"] == pytest.approx(0.365)
    assert h["HSG"] == pytest.approx(0.245)
    assert h["SCC VII"] == pytest.approx(0.281)
    assert min(h.values()) == pytest.approx(0.196)
    assert max(h.values()) == pytest.approx(0.365)
    # The table-1 global fit is deliberately a different number from table 2's
    # per-line HSG fit; mixing them is the documented trap.
    assert h["HSG"] != pytest.approx(H_GLOBAL)


# --- eq. (1): alpha is mMKM's, beta carries the time factor ------------------


def test_alpha_is_identical_to_modified_mkm() -> None:
    """Eq. (1) linear term == mMKM's ``alpha0 + beta0 z*_1D``."""
    model = hsg()
    mmkm = ModifiedMKM(
        alpha0_Gy_inv=model.alpha0,
        beta0_Gy2=model.beta0,
        domain_radius_um=model.domain_radius_um,
        nucleus_radius_um=model.nucleus_radius_um,
        z0_Gy=model.z0_Gy,
    )
    for z_star in (0.0, 1.0, 12.5, 80.0):
        assert model.alpha_beta_from_z_star_1D(z_star).alpha == pytest.approx(
            mmkm.alpha_beta_from_z_star_1D(z_star).alpha
        )


def test_acute_irradiation_collapses_onto_modified_mkm() -> None:
    """``F -> 1`` for an instantaneous dose, so IMK == mMKM acutely."""
    model = hsg()
    assert model.time_factor(DoseSchedule.acute(4.0)) == pytest.approx(1.0)
    assert model.alpha_beta_from_z_star_1D(10.0).beta == pytest.approx(model.beta0)
    assert model.alpha_beta_from_z_star_1D(
        10.0, schedule=DoseSchedule.acute(4.0)
    ) == model.alpha_beta_from_z_star_1D(10.0)


def test_alpha_zero_at_vanishing_specific_energy() -> None:
    """``z*_1D -> 0`` leaves the intrinsic ``alpha0`` (table 1)."""
    assert hsg().alpha_beta_from_z_star_1D(0.0).alpha == pytest.approx(HSG_ALPHA0)


@pytest.mark.parametrize("duration_h", [0.05, 0.5, 1.0, 4.0, 24.0])
def test_time_factor_matches_the_printed_closed_form(duration_h: float) -> None:
    """Eq. (1): ``F = 2[x + e^-x - 1]/x^2`` with ``x = (a+c) T``.

    The model computes ``F`` through the general block route in
    `pyradbio.core.protraction`; this is the paper's own expression,
    written out independently.
    """
    model = hsg()
    x = HSG_A_PLUS_C * duration_h
    expected = 2.0 * (x + math.exp(-x) - 1.0) / x**2
    schedule = DoseSchedule.continuous(4.0, duration_h=duration_h)
    assert model.time_factor(schedule) == pytest.approx(expected, rel=1e-12)


def test_time_factor_golden_at_unit_x() -> None:
    """``x = 1`` gives ``F = 2/e`` exactly."""
    model = IMKModel.hsg_matsuya2025(repair_rate_per_h=1.0)
    f = model.time_factor(DoseSchedule.continuous(2.0, duration_h=1.0))
    assert f == pytest.approx(2.0 / math.e, rel=1e-12)
    assert f == pytest.approx(0.735758882342885, rel=1e-12)


def test_time_factor_falls_with_slower_delivery() -> None:
    model = hsg()
    factors = [
        model.time_factor(DoseSchedule.continuous(4.0, duration_h=t))
        for t in (0.01, 0.1, 1.0, 10.0, 100.0)
    ]
    assert all(b < a for a, b in pairwise(factors))
    assert factors[0] == pytest.approx(1.0, abs=0.02)
    assert factors[-1] < 0.02


def test_beta_is_the_time_factor_times_beta0() -> None:
    model = hsg()
    schedule = DoseSchedule.continuous(4.0, duration_h=2.0)
    lq = model.alpha_beta_from_z_star_1D(5.0, schedule=schedule)
    assert lq.beta == pytest.approx(model.time_factor(schedule) * model.beta0)
    assert lq.alpha == pytest.approx(model.alpha_beta_from_z_star_1D(5.0).alpha)


def test_repair_rate_and_half_time_are_inverses() -> None:
    """The paper parameterises by ``(a+c)``; ``protraction`` by ``T_half``."""
    assert repair_half_time_h_from_rate(HSG_A_PLUS_C) == pytest.approx(math.log(2.0) / HSG_A_PLUS_C)
    assert repair_rate_per_h(repair_half_time_h_from_rate(HSG_A_PLUS_C)) == pytest.approx(
        HSG_A_PLUS_C
    )
    assert repair_half_time_h_from_rate(0.0) == math.inf


def test_time_factor_equals_lea_catcheside_at_the_matching_half_time() -> None:
    model = hsg()
    schedule = DoseSchedule.fractionated(2.0, n_fractions=3, interval_h=6.0)
    expected = lea_catcheside_g(
        schedule, repair_half_time_h=repair_half_time_h_from_rate(HSG_A_PLUS_C)
    )
    assert model.time_factor(schedule) == pytest.approx(expected, rel=1e-12)


def test_no_repair_leaves_the_quadratic_term_untouched() -> None:
    """``(a+c) = 0`` means no sublethal repair, hence ``F = 1`` at any ``T``."""
    model = IMKModel.hsg_matsuya2025(repair_rate_per_h=0.0)
    assert model.time_factor(DoseSchedule.continuous(4.0, duration_h=48.0)) == pytest.approx(1.0)


# --- eq. (3): micronucleus formation ----------------------------------------


def test_micronucleus_lq_is_the_survival_lq_scaled_by_h() -> None:
    """Eq. (3): ``alpha_m = h alpha``, ``beta_m = h beta``."""
    model = hsg()
    schedule = DoseSchedule.continuous(4.0, duration_h=3.0)
    lq = model.alpha_beta_from_z_star_1D(7.0, schedule=schedule)
    mn = model.micronucleus_lq(lq)
    assert mn.alpha == pytest.approx(H_GLOBAL * lq.alpha)
    assert mn.beta == pytest.approx(H_GLOBAL * lq.beta)


@pytest.mark.parametrize("dose_Gy", [0.5, 2.0, 4.0, 8.0])
def test_micronucleus_frequency_is_h_times_the_log_kill(dose_Gy: float) -> None:
    """Eq. (3): ``F_MN = h <w>`` and ``<w> = -ln S``."""
    model = hsg()
    lq = model.alpha_beta_from_z_star_1D(3.0)
    freq = model.micronucleus_frequency(dose_Gy, lq)
    assert freq == pytest.approx(H_GLOBAL * log_kill(dose_Gy, lq.alpha, lq.beta))
    assert freq / (-math.log(survival(dose_Gy, lq.alpha, lq.beta))) == pytest.approx(H_GLOBAL)


def test_micronucleus_frequency_rises_with_dose_and_falls_with_protraction() -> None:
    model = hsg()
    acute = model.alpha_beta_from_z_star_1D(3.0)
    slow = model.alpha_beta_from_z_star_1D(
        3.0, schedule=DoseSchedule.continuous(4.0, duration_h=8.0)
    )
    freqs = [model.micronucleus_frequency(d, acute) for d in (1.0, 2.0, 4.0, 8.0)]
    assert all(b > a for a, b in pairwise(freqs))
    assert model.micronucleus_frequency(4.0, slow) < model.micronucleus_frequency(4.0, acute)


def test_dose_rate_experiment_protocol_reduces_micronuclei() -> None:
    """Sec. 3.1.4: 4 Gy total, acute at 1 Gy/min vs 0.05 Gy/Fr every 57 s.

    The paper's own in vitro regimens, expressed as schedules. The ordering is
    the measured one (fig. 6): slower average dose rate, fewer micronuclei.
    """
    model = IMKModel.du145_matsuya2025()
    lq_of = lambda s: model.alpha_beta_from_z_star_1D(2.0, schedule=s)  # noqa: E731
    acute = DoseSchedule.continuous(4.0, duration_h=4.0 / 60.0)
    protocols = [
        acute,
        DoseSchedule.fractionated(
            0.5, n_fractions=8, interval_h=1.0 / 60.0, fraction_duration_h=0.5 / 60.0
        ),
        DoseSchedule.fractionated(
            0.1, n_fractions=40, interval_h=1.0 / 60.0, fraction_duration_h=0.1 / 60.0
        ),
        DoseSchedule.fractionated(
            0.05, n_fractions=80, interval_h=1.0 / 60.0, fraction_duration_h=0.05 / 60.0
        ),
    ]
    freqs = [model.micronucleus_frequency(4.0, lq_of(p)) for p in protocols]
    assert all(b < a for a, b in pairwise(freqs))


def test_micronucleus_probability_is_validated() -> None:
    for bad in (-0.1, 1.5, math.nan):
        with pytest.raises(ValueError, match="mn_probability"):
            IMKModel.hsg_matsuya2025(mn_probability=bad)


def test_repair_rate_is_validated() -> None:
    for bad in (-1.0, math.nan):
        with pytest.raises(ValueError, match="repair_rate_per_h"):
            IMKModel.hsg_matsuya2025(repair_rate_per_h=bad)


# --- eqs. (4)-(6): RBE ------------------------------------------------------


def test_d10_is_the_lq_inversion() -> None:
    """Eq. (4) is ``dose_for_survival`` at ``S = 0.1``, algebraically."""
    model = hsg()
    lq = model.alpha_beta_from_z_star_1D(9.0)
    s_star = -math.log(0.1)
    printed = (math.sqrt(lq.alpha**2 + 4.0 * lq.beta * s_star) - lq.alpha) / (2.0 * lq.beta)
    assert model.d10(lq) == pytest.approx(printed, rel=1e-12)
    assert model.d10(lq) == pytest.approx(dose_for_survival(0.1, lq.alpha, lq.beta), rel=1e-12)


def test_rbe_mn_is_the_ratio_of_alphas() -> None:
    """Eq. (6): ``RBE_MN = alpha*/alpha(photon)``."""
    model = IMKModel.hsg_matsuya2025(reference_z_star_1D_Gy=0.8)
    lq = model.alpha_beta_from_z_star_1D(20.0)
    assert model.rbe_mn(lq) == pytest.approx(lq.alpha / model.reference().alpha_x)
    assert model.rbe_mn(lq) > 1.0


def test_rbe_mn_does_not_depend_on_h() -> None:
    """``h`` cancels in eq. (6) -- the paper's central claim."""
    base = IMKModel.hsg_matsuya2025(reference_z_star_1D_Gy=0.8)
    other = IMKModel.hsg_matsuya2025(reference_z_star_1D_Gy=0.8, mn_probability=0.196)
    lq_base = base.alpha_beta_from_z_star_1D(20.0)
    lq_other = other.alpha_beta_from_z_star_1D(20.0)
    assert base.rbe_mn(lq_base) == pytest.approx(other.rbe_mn(lq_other), rel=1e-12)


def test_rbe_mn_equals_the_zero_dose_limit_of_the_survival_rbe() -> None:
    """Eq. (6) is ``RBE_max``: the ``D -> 0`` limit of the iso-survival RBE."""
    model = IMKModel.hsg_matsuya2025(reference_z_star_1D_Gy=0.8)
    lq = model.alpha_beta_from_z_star_1D(20.0)
    ref = model.reference()
    reference = LQParameters(alpha=ref.alpha_x, beta=ref.beta_x)
    limit = [rbe(s, lq.alpha, lq.beta, reference=reference) for s in (0.5, 0.9, 0.99, 0.9999)]
    # The approach to the limit is first order in the log-kill, so S = 0.99 is
    # still 1.3% short of it; S = 0.9999 is not.
    assert limit[-1] == pytest.approx(model.rbe_mn(lq), rel=1e-3)
    assert limit[-2] < model.rbe_mn(lq)
    assert all(b > a for a, b in pairwise(limit))


def test_photon_reference_is_the_model_evaluated_at_the_photon_quality() -> None:
    """The paper's ``alpha(photon)`` is eq. (1) itself, not a foreign LQ fit."""
    model = IMKModel.hsg_matsuya2025(reference_z_star_1D_Gy=0.8)
    assert model.reference() == PhotonParameters(
        alpha_x=HSG_ALPHA0 + HSG_BETA0 * 0.8, beta_x=HSG_BETA0
    )


def test_hsg_and_scc_carry_the_recovered_photon_reference() -> None:
    """Their fig.-2 X-ray curves give it, so RBE works straight off the constructor."""
    for model, expected in (
        (hsg(), RECOVERED_PHOTON_Z_STAR_1D_GY["HSG"]),
        (IMKModel.scc_matsuya2025(), RECOVERED_PHOTON_Z_STAR_1D_GY["SCC"]),
    ):
        assert model.reference_z_star_1D_Gy == pytest.approx(expected)
        assert model.rbe_mn(model.alpha_beta_from_z_star_1D(20.0)) > 1.0
        assert model.reference().alpha_x == pytest.approx(model.alpha0 + model.beta0 * expected)


def test_rbe_requires_the_photon_reference_where_none_was_recovered() -> None:
    """No fig.-2 panel for WI-38/DU145, and none for a hand-built model.

    Falling back to ``alpha_ref`` would substitute a foreign photon fit -- for
    HSG the obvious candidate (Furusawa's 0.313) is 1.6x this paper's own 0.199.
    """
    for model in (
        IMKModel.wi38_matsuya2025(),
        IMKModel.du145_matsuya2025(),
        IMKModel.hsg_matsuya2025(reference_z_star_1D_Gy=None),
    ):
        lq = model.alpha_beta_from_z_star_1D(20.0)
        with pytest.raises(ValueError, match="reference_z_star_1D_Gy"):
            model.rbe_mn(lq)


def test_reference_can_be_built_from_a_photon_dose_spectrum() -> None:
    """The moment-level shortcut agrees with scoring a photon ``d(y)``."""
    model = hsg()
    y = np.geomspace(0.1, 30.0, 400)
    d_of_y = np.exp(-((np.log(y / 2.0)) ** 2) / 0.5)
    reference = model.reference_from_dose_spectrum(y, d_of_y)
    z_star = model.y_star_D(y, d_of_y) / model._z_domain_scale_keV_um_per_Gy
    assert reference.alpha == pytest.approx(HSG_ALPHA0 + HSG_BETA0 * z_star)
    assert reference.beta == pytest.approx(HSG_BETA0)
    assert IMKModel.hsg_matsuya2025(
        reference_z_star_1D_Gy=z_star
    ).reference().alpha_x == pytest.approx(reference.alpha)


# --- the ion route ----------------------------------------------------------


def test_rbe_rises_with_let_over_the_papers_carbon_range() -> None:
    """Fig. 2/5 qualitative behaviour on our own ``d(y)``: RBE grows with LET.

    Not a reconstruction of the paper's numbers -- its ``y*`` comes from PHITS
    (sec. 3.1.1) and its photon reference is unpublished. What is testable is
    that the shipped ion route is monotone over the 13-100 keV/um carbon band
    the paper fits.
    """
    model = IMKModel.hsg_matsuya2025(reference_z_star_1D_Gy=0.8)
    energies = (290.0, 100.0, 30.0, 12.0)
    lqs = [model.alpha_beta(Ion(Z=6, A=12, energy_MeV_u=e)) for e in energies]
    assert all(b.alpha > a.alpha for a, b in pairwise(lqs))
    assert all(model.rbe_mn(lq) > 1.0 for lq in lqs[1:])


def test_ion_route_accepts_a_schedule() -> None:
    model = hsg()
    ion = Ion(Z=6, A=12, energy_MeV_u=100.0)
    schedule = DoseSchedule.continuous(4.0, duration_h=6.0)
    acute_lq = model.alpha_beta(ion)
    slow_lq = model.alpha_beta(ion, schedule=schedule)
    assert slow_lq.alpha == pytest.approx(acute_lq.alpha)
    assert slow_lq.beta == pytest.approx(model.time_factor(schedule) * model.beta0)
    assert slow_lq.beta < acute_lq.beta
