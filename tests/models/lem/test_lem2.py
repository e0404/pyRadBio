"""LEM II (cluster extension + refined radial dose).

All reference values are from Elsasser & Scholz 2007 (doi:10.1667/RR0467.1).
"""

from __future__ import annotations

import numpy as np
import pytest

from pyradbio.core.cell import CellTarget
from pyradbio.core.photon import ClusterParameters, PhotonParameters
from pyradbio.core.radiation import Ion, RadiationField, SpectrumComponent
from pyradbio.models.lem.lem1 import LEM1
from pyradbio.models.lem.lem2 import LEM2, ClusterModel, Friedrich2015ClusterModel

# V79 photon inputs and geometry (Elsasser & Scholz 2007, Fig. 6).
#   alpha_x=0.093, beta_x=0.026 Gy^-2, R_n=3.8 um.
#   D_t = 30 Gy (LEM old) -> 7.5 Gy (LEM new, refit after the cluster extension).
V79_OLD = PhotonParameters(alpha_x=0.093, beta_x=0.026, D_t=30.0)
V79_NEW = PhotonParameters(alpha_x=0.093, beta_x=0.026, D_t=7.5)
V79_CELL = CellTarget(nucleus_radius_um=3.8)

# Carbon-12 energies [MeV/u] spanning the entrance channel through the overkill
# region (LET ~ 12 -> 250 keV/um), used to trace the RBE_alpha(LET) curve.
CARBON_ENERGIES = [
    300.0,
    200.0,
    150.0,
    100.0,
    70.0,
    40.0,
    25.0,
    16.0,
    11.0,
    8.0,
    6.0,
    4.0,
    2.5,
    1.5,
]


def test_lem2_instantiates_with_default_components() -> None:
    """LEM2 builds its default radial-dose and cluster components."""
    lem = LEM2(photon=V79_NEW, cell=V79_CELL)

    assert isinstance(lem.resolved_cluster, ClusterModel)
    assert lem.resolved_radial_dose is not None
    assert lem.cluster is None
    assert lem.radial_dose is None


def _rbe_alpha_curve(lem: LEM1 | LEM2, energies: list[float]) -> list[float]:
    """LET-sorted RBE_alpha = alpha_ion/alpha_x for carbon-12 at ``energies``."""
    from pyradbio.track.stopping_power import SimpleBetheStoppingPower

    sp = SimpleBetheStoppingPower()
    pairs = [
        (sp.let_keV_um(Ion(6, 12, e)), lem.alpha_beta(Ion(6, 12, e)).alpha / lem.photon.alpha_x)
        for e in energies
    ]
    return [rbe for _, rbe in sorted(pairs)]


@pytest.mark.golden
def test_golden_cluster_eta_onset_and_maximum() -> None:
    """Damage enhancement eta(D): onset ~10^3 Gy, maximum ~5x10^5 Gy.

    Reference: Elsasser & Scholz 2007, Fig. 2 (doi:10.1667/RR0467.1) — "For doses
    higher than 10^3 Gy, clustered SSBs contribute significantly to the total
    number of DSBs. Due to saturation a maximum of eta is reached at about
    5x10^5 Gy." The figure tops out near eta ~ 15.
    """
    cm = ClusterModel()  # default SSB=1250, DSB=30, h=25 bp, genome=6e9

    # Below the onset eta is ~ 1 (the measurable low-dose curve is unchanged).
    assert cm.eta(1.0e2) == pytest.approx(1.0, abs=0.05)
    # Onset: clustered SSBs start to contribute above ~10^3 Gy.
    assert 1.02 < cm.eta(1.0e3) < 1.5
    # Rises steeply through the 10^4-10^5 Gy decade.
    assert cm.eta(1.0e4) > 1.5
    assert cm.eta(1.0e5) > 5.0

    doses = np.logspace(3.0, 7.0, 600)
    eta = np.array([cm.eta(float(d)) for d in doses])
    peak = int(np.argmax(eta))
    assert 2.0e5 <= doses[peak] <= 8.0e5, f"eta peaks at {doses[peak]:.2g} Gy"
    assert 10.0 <= eta[peak] <= 16.0, f"eta_max = {eta[peak]:.1f}"
    # Saturation: eta falls back toward 1 once direct DSBs again dominate.
    assert cm.eta(1.0e7) < 0.3 * eta[peak]


def test_default_cluster_model_is_shared_across_instances() -> None:
    """LEM II/III instances that don't supply a cluster model share one memoised
    eta table (it depends only on ClusterParameters, so it is built once).
    """
    a = LEM2(photon=V79_NEW, cell=V79_CELL)
    b = LEM2(photon=PhotonParameters(alpha_x=0.05, beta_x=0.01, D_t=6.0), cell=CellTarget(5.0))
    assert a.resolved_cluster is b.resolved_cluster  # same default parameters -> same object

    # A different fusion window is a distinct table (not the shared default).
    custom = ClusterModel(params=ClusterParameters(stagger_bp=35.0))
    c = LEM2(photon=V79_NEW, cell=V79_CELL, cluster=custom)
    assert c.cluster is custom
    assert c.resolved_cluster is custom
    assert c.resolved_cluster is not a.resolved_cluster


def test_cluster_eta_is_deterministic() -> None:
    """Same seed -> identical eta table (reproducible Monte-Carlo)."""
    a, b = ClusterModel(rng_seed=7), ClusterModel(rng_seed=7)
    for d in (1.0e3, 1.0e4, 1.0e5, 5.0e5):
        assert a.eta(d) == b.eta(d)


def test_eta_is_unity_below_the_cluster_onset() -> None:
    """No clustered damage at or below the table floor, nor at LEM's D_t scale.

    Continuity of eq. 10 at ``D_t`` relies on ``eta(D_t) == 1``: the tail is
    ``s_max*(eta(D)*D - D_t)``, which reduces to ``s_max*(D - D_t)`` only while
    ``eta == 1``. Realistic ``D_t`` values are tens of Gy, far below the ~1e3 Gy
    onset.
    """
    cm = ClusterModel()
    assert cm.eta(0.5) == 1.0
    for d_t in (6.0, 7.5, 9.5, 30.0, 40.0):
        assert cm.eta(d_t) == pytest.approx(1.0, abs=1.0e-4)


def test_photon_log_kill_scales_the_tail_dose_by_eta() -> None:
    """Eq. 10 MULTIPLIES eta into the dose; it does not integrate it.

    At high dose the tail must approach ``s_max*eta(D)*D``, i.e. the un-enhanced
    straight tail times ``eta(D)`` -- the relation that identifies the form
    against the paper's Fig. 3 (solid/dashed ratio == eta(D)).
    """
    photon = PhotonParameters(alpha_x=0.228, beta_x=0.020, D_t=40.0)
    model = LEM2(photon=photon, cell=CellTarget(nucleus_radius_um=4.7))
    cm = model._cluster()

    for dose in (1.0e5, 4.0e5, 1.0e6):
        expected = photon.log_kill(40.0) + photon.max_slope() * (cm.eta(dose) * dose - 40.0)
        assert model._photon_log_kill(dose) == pytest.approx(expected, rel=1e-12)

    # Ratio to the un-enhanced (eta=1) tail approaches eta(D) at high dose.
    dose = 1.0e6
    plain = photon.log_kill(40.0) + photon.max_slope() * (dose - 40.0)
    assert model._photon_log_kill(dose) / plain == pytest.approx(cm.eta(dose), rel=0.02)


def test_photon_log_kill_is_continuous_at_dt() -> None:
    """eta(D_t) == 1 makes eq. 10 join the LQ shoulder without a jump."""
    photon = PhotonParameters(alpha_x=0.228, beta_x=0.020, D_t=40.0)
    model = LEM2(photon=photon, cell=CellTarget(nucleus_radius_um=4.7))

    below = model._photon_log_kill(40.0 - 1.0e-6)
    above = model._photon_log_kill(40.0 + 1.0e-6)
    assert above == pytest.approx(below, rel=1e-6)


@pytest.mark.golden
def test_golden_lem2_raises_rbe_max_to_min_ratio_v79() -> None:
    """LEM II raises the RBE-max/RBE-min ratio vs LEM I for V79 carbon.

    Reference: Elsasser & Scholz 2007, Fig. 6 (doi:10.1667/RR0467.1). The paper's
    "key feature of the cluster extension" is that "the ratio of the highest RBEs
    at the maximum to the lowest RBE for low-LET particles is increased": the
    cluster damage lifts the high-LET maximum (high local doses reach the cluster
    regime) while the refit-lowered D_t pulls the low-LET entrance RBE down, where
    local doses never reach the cluster onset.
    """
    lem_old = LEM1(photon=V79_OLD, cell=V79_CELL)
    lem_new = LEM2(photon=V79_NEW, cell=V79_CELL)

    rbe_old = _rbe_alpha_curve(lem_old, CARBON_ENERGIES)
    rbe_new = _rbe_alpha_curve(lem_new, CARBON_ENERGIES)
    ratio_old = max(rbe_old) / min(rbe_old)
    ratio_new = max(rbe_new) / min(rbe_new)

    assert ratio_new > ratio_old, f"LEM II ratio {ratio_new:.2f} !> LEM I {ratio_old:.2f}"
    # The lift comes from the maximum (overkill region), not the low-LET minimum:
    # the entrance-channel RBE drops with the reduced D_t.
    assert min(rbe_new) < min(rbe_old)


@pytest.mark.golden
def test_golden_lem2_without_dt_reduction_is_unrealistically_high() -> None:
    """Both modifications without reducing D_t give unrealistically high RBE.

    Reference: Elsasser & Scholz 2007, Fig. 6 thin dashed line (doi:10.1667/
    RR0467.1): "Combining both modifications [refined radial dose + cluster
    extension] yields extremely high and unrealistic RBE values ... Hence we must
    decrease the threshold parameter D_t (from 30 Gy to 7.5 Gy)." Keeping the old
    D_t=30 with the cluster extension must overshoot LEM I's maximum markedly.
    """
    lem_old = LEM1(photon=V79_OLD, cell=V79_CELL)
    lem_both_no_refit = LEM2(photon=V79_OLD, cell=V79_CELL)  # cluster + new radial, D_t=30

    peak_old = max(_rbe_alpha_curve(lem_old, CARBON_ENERGIES))
    peak_both = max(_rbe_alpha_curve(lem_both_no_refit, CARBON_ENERGIES))
    assert peak_both > 1.5 * peak_old, f"peak {peak_both:.1f} not >> LEM I {peak_old:.1f}"


@pytest.mark.golden
def test_golden_lem2_stagger_size_orders_rbe_ratio() -> None:
    """RBE-max/min ratio grows with the stagger size h (Fig. 7 ordering).

    Reference: Elsasser & Scholz 2007, Fig. 7 (doi:10.1667/RR0467.1): at fixed
    D_t=7.5 Gy, "the ratio of the highest to lowest RBE values is increased for a
    stagger size of 35 bp and decreased for 20 bp and 10 bp" relative to the
    chosen h=25 bp. A larger fusion window clusters more SSBs into DSBs at high
    local dose, lifting the maximum.
    """
    energies = [300.0, 150.0, 90.0, 50.0, 25.0, 12.0, 3.0]
    ratios = []
    for h in (10.0, 20.0, 25.0, 35.0):
        cluster = ClusterModel(params=ClusterParameters(stagger_bp=h))
        lem = LEM2(photon=V79_NEW, cell=V79_CELL, cluster=cluster)
        curve = _rbe_alpha_curve(lem, energies)
        ratios.append(max(curve) / min(curve))

    assert ratios == sorted(ratios), f"ratio not monotonic in h: {ratios}"
    assert ratios[-1] > ratios[0]  # h=35 strictly above h=10


def test_photon_tail_enhanced_above_dt_only() -> None:
    """LEM II's local log-kill matches the photon curve below D_t and exceeds it
    above (the cluster enhancement applies only to the exponential tail)."""
    lem = LEM2(photon=V79_NEW, cell=V79_CELL)
    d_t = V79_NEW.D_t
    assert d_t is not None

    # Below / at D_t: identical to the plain shouldered LQ photon response.
    for d in (1.0, 5.0, d_t):
        assert lem._photon_log_kill(d) == pytest.approx(V79_NEW.log_kill(d))

    # Just above D_t the enhancement is still negligible (eta ~ 1 below the
    # onset), but at high local dose it strictly exceeds the bare linear tail.
    assert lem._photon_log_kill(2.0 * d_t) == pytest.approx(V79_NEW.log_kill(2.0 * d_t), rel=1e-3)
    high = 1.0e4
    assert lem._photon_log_kill(high) > V79_NEW.log_kill(high)


def test_lem2_lethal_central_keeps_cluster_enhancement() -> None:
    """``lethal_central`` caps at the *enhanced* ``s_max * eta_max``, not ``s_max``.

    ``N_1`` carries the cluster enhancement ``eta(D)`` (via ``_photon_log_kill``),
    and the local tail slope is ``s_max * eta(D)`` (Elsasser 2007 eq. 10), so
    ``N_1/D_1 <= s_max * eta_max``. Capping at the un-enhanced ``s_max`` would
    silently drop the enhancement that the ``cross_section`` default and
    ``lethal_full`` keep; the enhanced cap makes all three modes retain it above
    ``s_max``. For LEM I (eta = 1) the cap is unchanged.
    """
    cm = ClusterModel()
    cs = LEM2(photon=V79_NEW, cell=V79_CELL, cluster=cm, initial_slope_model="cross_section")
    lc = LEM2(photon=V79_NEW, cell=V79_CELL, cluster=cm, initial_slope_model="lethal_central")
    lf = LEM2(photon=V79_NEW, cell=V79_CELL, cluster=cm, initial_slope_model="lethal_full")
    s_max = V79_NEW.max_slope()
    s_max_enh = s_max * cm.eta_max

    ion = Ion(6, 12, 11.0)  # high-LET carbon: enhanced N_1/D_1 exceeds s_max
    n1_over_d1 = lc.single_traversal_log_kill(ion) / lc.mean_traversal_dose_Gy(ion)
    assert n1_over_d1 > s_max, "test ion no longer exposes the un-enhanced cap"
    assert cm.eta_max > 1.0
    assert n1_over_d1 <= s_max_enh * (1.0 + 1e-9)  # the enhanced cap is a valid bound

    # lethal_central keeps the enhanced slope (= N_1/D_1 here, below the cap).
    assert lc.initial_slope(ion) == pytest.approx(n1_over_d1)
    assert lc.initial_slope(ion) > s_max
    # ... consistent with the cross_section default and lethal_full: all three
    # modes retain the enhancement above the un-enhanced s_max. Their ordering
    # among themselves is not asserted -- under eq. 10 the two overkill-free
    # modes sit within ~0.02% of each other for this ion, so any ordering here
    # would be numerical noise rather than a modelling claim.
    assert cs.initial_slope(ion) > s_max
    assert lf.initial_slope(ion) > s_max


def test_mixed_field_dose_averages_components() -> None:
    """A one-component field reduces to the single ion; an equal-dose mix
    dose-averages alpha and dose-averages sqrt(beta) across components."""
    import math

    lem = LEM2(photon=V79_NEW, cell=V79_CELL)
    ion_a, ion_b = Ion(6, 12, 12.0), Ion(6, 12, 120.0)
    lq_a, lq_b = lem.alpha_beta(ion_a), lem.alpha_beta(ion_b)

    mono = RadiationField([SpectrumComponent(ion_a, dose_Gy=2.0)])
    assert lem.alpha_beta(mono).alpha == pytest.approx(lq_a.alpha)

    mix = RadiationField(
        [SpectrumComponent(ion_a, dose_Gy=1.0), SpectrumComponent(ion_b, dose_Gy=1.0)]
    )
    lq_mix = lem.alpha_beta(mix)
    assert lq_mix.alpha == pytest.approx(0.5 * (lq_a.alpha + lq_b.alpha))
    root = 0.5 * (math.sqrt(lq_a.beta) + math.sqrt(lq_b.beta))
    assert lq_mix.beta == pytest.approx(root * root)


# --- Friedrich, Durante & Scholz 2015 closed-form eta ---------------------------

_F2015 = ClusterParameters(genome_size_bp=5.4e9)  # the 2015 paper's own genome size


@pytest.mark.golden
def test_golden_friedrich2015_reproduces_its_own_quoted_alpha_over_beta() -> None:
    """Friedrich 2015 eq. (10) + its stated alpha_DSB/beta_DSB ~ 8300 Gy.

    The paper gives only one number tying its algebra to a value, so this is the
    single available end-to-end check on our transcription of eqs. (2)-(8): the
    quadratic DSB coefficient is beta = alpha_SSB^2 t / (2 L_gen), and the paper
    states alpha_DSB/beta ~ 8300 Gy "i.e. eta = 2" there. Both must hold together
    -- the ratio alone would pass on eq. (10) while eq. (7) was wrong.
    """
    model = Friedrich2015ClusterModel(params=_F2015)
    balance_dose = _F2015.dsb_per_gy_cell / model.beta_dsb_per_gy2()

    assert balance_dose == pytest.approx(8300.0, rel=0.01)
    assert model.eta(balance_dose) == pytest.approx(2.0, rel=0.05)


@pytest.mark.golden
def test_golden_friedrich2015_agrees_with_our_rule_where_tracks_can_reach() -> None:
    """The independent-rule cross-check that exonerates our ``ClusterModel``.

    Same alpha_SSB/alpha_DSB/t, a stricter coincidence rule, closed form instead
    of Monte-Carlo -- and it lands on our peak (13.51 vs 13.40), not the digitized
    Fig. 2's 14.79. Below 1e5 Gy, the most a carbon track centre produces (2015
    paper, Discussion), the two rules track each other to a few percent. Above
    ~1e6 Gy they diverge hard, which is exactly where that paper says its own
    implementation is not determined -- so the divergence is not pinned.
    """
    ours, theirs = ClusterModel(), Friedrich2015ClusterModel(params=_F2015)

    assert theirs.eta_max == pytest.approx(ours.eta_max, rel=0.02)

    sampled = np.geomspace(1.0e2, 1.0e5, 40)
    ratio = np.array([theirs.eta(float(d)) / ours.eta(float(d)) for d in sampled])
    assert np.sqrt(np.mean((ratio - 1.0) ** 2)) < 0.05
    assert np.max(np.abs(ratio - 1.0)) < 0.10

    # Both start at no enhancement and stay >= 1: eta is a damage *enhancement*.
    assert theirs.eta(0.0) == 1.0
    assert all(theirs.eta(float(d)) >= 1.0 for d in np.geomspace(1.0, 1.0e9, 60))


def test_friedrich2015_peak_height_is_independent_of_genome_size() -> None:
    """``L_gen`` rescales the dose axis and cancels from the peak height.

    It enters only through rho_SSB and rho_DSB, which are both proportional to
    D/L_gen, so changing it slides the curve along log-dose without reshaping it.
    That is why our 6.0e9 default and the 2015 paper's 5.4e9 can be compared at
    all -- and why no genome size could ever reconcile either with Fig. 2's peak.
    """
    small = Friedrich2015ClusterModel(params=ClusterParameters(genome_size_bp=3.0e9))
    large = Friedrich2015ClusterModel(params=ClusterParameters(genome_size_bp=1.2e10))

    assert small.eta_max == pytest.approx(large.eta_max, rel=1e-3)
    # ... and the shift is exactly the genome-size ratio: eta(D) = eta(4D) at 4x L_gen.
    for dose in (1.0e3, 1.0e4, 1.0e5):
        assert large.eta(4.0 * dose) == pytest.approx(small.eta(dose), rel=1e-9)


def test_cluster_model_peak_height_is_independent_of_genome_size() -> None:
    """The shipped MC rule obeys the same invariance as the closed form.

    ``eta - 1 = (a_SSB/a_DSB) * q(mu)/mu`` with ``mu = a_SSB*D*h/L_gen``, so
    ``L_gen`` cancels from the peak height and only rescales the dose axis. This
    is what makes the genome size useless as a knob on Fig. 2 (whose peak of
    14.79 is out of reach for *any* ``L_gen``) while still being a real lever on
    ion RBE, where the local-dose spectrum is fixed and sliding ``eta`` along the
    dose axis changes how much of the track is enhanced.

    The algebra is exact; every deviation below is Monte-Carlo sampling noise,
    which is why the dose-axis check starts at 1e4 Gy. Near the onset the two
    realisations differ by ~1% on ``eta`` -- small in absolute terms but ~6% of
    the enhancement ``eta - 1``, since that is only ~0.2 there.
    """
    small = ClusterModel(params=ClusterParameters(genome_size_bp=3.0e9))
    large = ClusterModel(params=ClusterParameters(genome_size_bp=1.2e10))

    assert small.eta_max == pytest.approx(large.eta_max, rel=5e-3)
    for dose in (1.0e4, 1.0e5, 1.0e6):
        assert large.eta(4.0 * dose) == pytest.approx(small.eta(dose), rel=1e-2)


def test_public_photon_log_kill_is_the_private_hook() -> None:
    """``photon_log_kill`` (public) returns exactly the eq.-10 hook LEMBase calls."""
    photon = PhotonParameters(alpha_x=0.228, beta_x=0.020, D_t=40.0)
    model = LEM2(photon=photon, cell=CellTarget(nucleus_radius_um=4.7))
    for dose in (0.5, 10.0, 40.0, 40.0 + 1e-9, 1.0e3, 1.0e5, 4.0e5, 1.0e7):
        assert model.photon_log_kill(dose) == model._photon_log_kill(dose)
