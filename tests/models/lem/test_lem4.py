"""LEM IV (DSB-pattern / cluster-index).

Reference values from Friedrich et al. 2012 (doi:10.3109/09553002.2011.611213),
Figs. 2-4 and the DSB-pattern equations (p.104-105). The golden tests pin the
*shape* of the V79 carbon comparison (Fig. 3): RBE_alpha rises from the photon
limit toward a single interior maximum in the intermediate-high LET range and
drops in the overkill region, with RBE_10 below RBE_alpha throughout, and the
cluster index / photon-equivalent dose rising with LET. See
docs/references/LEM_IV_Friedrich2012.md for the (documented) magnitude limit of
the single-particle continuous approximation vs the paper's full Monte-Carlo.
"""

from __future__ import annotations

import dataclasses
import itertools
import math
import warnings
from typing import Protocol

import numpy as np
import pytest

from pyradbio.benchmark import elsasser2010 as es10
from pyradbio.benchmark import friedrich2012 as f12
from pyradbio.benchmark.sweeps import sweep_energies
from pyradbio.core.cell import CellTarget
from pyradbio.core.lq import dose_for_survival
from pyradbio.core.photon import ClusterParameters, PhotonParameters
from pyradbio.core.radiation import Ion, RadiationField, SpectrumComponent
from pyradbio.models.lem.lem4 import (
    _CLUSTER_INDEX_TRIP98_LEGACY,
    _DISCRETE_N_IMPACT,
    DSB_PER_GY_CELL,
    DSB_PER_GY_PER_LOOP,
    LEM4,
    SUBVOLUME_SIDE_NM,
)
from pyradbio.track.scholz_kraft import ScholzKraftRadialDose
from pyradbio.track.stopping_power import SimpleBetheStoppingPower, energy_for_let_MeV_u

_SP = SimpleBetheStoppingPower()

# V79 photon inputs and geometry (Friedrich 2012, Fig. 3 caption). LEM IV does
# not consume photon.cluster (see test_lem4_warns_on_unused_photon_cluster), so
# the fixture leaves it unset.
V79 = PhotonParameters(alpha_x=0.184, beta_x=0.02, D_t=17.0)
V79_CELL = CellTarget(nucleus_radius_um=4.7)


def _carbon_energy_for_let(let_keV_um: float) -> float:
    """Carbon-12 energy [MeV/u] whose water LET equals ``let_keV_um``."""
    return energy_for_let_MeV_u(
        _SP, 6, 12, let_keV_um, branch="bracket", energy_bounds_MeV_u=(1.6, 600.0)
    )


# Carbon-12 sampled across the entrance channel through the overkill region.
_CARBON_LETS = [16.0, 20.0, 30.0, 45.0, 65.0, 90.0, 120.0, 160.0, 220.0, 320.0, 420.0]
CARBON_IONS = [Ion(6, 12, _carbon_energy_for_let(let)) for let in _CARBON_LETS]


def _rbe_alpha_curve(lem: LEM4, ions: list[Ion]) -> tuple[list[float], list[float]]:
    """LET-sorted ``(LET, RBE_alpha)`` for the given carbon ions."""
    assert all((ion.Z, ion.A) == (6, 12) for ion in ions)
    curve = sweep_energies(
        lem,
        6,
        12,
        [ion.energy_MeV_u for ion in ions],
        stopping_power=_SP,
        on_error="raise",
        sort_by_let=True,
    )
    return curve.let_keV_um.tolist(), curve.rbe_alpha(lem.photon.alpha_x).tolist()


def _rbe10(lem: LEM4, ion: Ion) -> float:
    """RBE at 10% survival = D10_photon / D10_ion (LQ inversion)."""
    d10_x = dose_for_survival(0.1, lem.photon.alpha_x, lem.photon.beta_x)
    lq = lem.alpha_beta(ion)
    return d10_x / dose_for_survival(0.1, lq.alpha, lq.beta)


# ---- construction / physics wiring -------------------------------------
def test_lem4_uses_lem3_radial_dose() -> None:
    """LEM IV builds on the LEM III velocity-dependent-core radial dose."""
    lem = LEM4(photon=V79, cell=V79_CELL)
    radial = lem.resolved_radial_dose
    assert isinstance(radial, ScholzKraftRadialDose)
    assert radial.velocity_dependent_core
    assert radial.radical_diffusion


def test_dsb_yield_per_subvolume_matches_elsasser2010_density() -> None:
    """kappa_dsb = 30 DSB/Gy/cell * l_DSB^3 / V (Elsasser 2010 eq. 1 density).

    l_DSB=540 nm, V=500 um^3 -> ~0.00945 /Gy (~the giant-loop 0.01 estimate).
    """
    assert DSB_PER_GY_PER_LOOP == pytest.approx(30.0 * 0.540**3 / 500.0)
    assert DSB_PER_GY_PER_LOOP == pytest.approx(0.00945, abs=1e-4)


def test_lem4_warns_on_unused_photon_cluster() -> None:
    """LEM IV ignores photon.cluster (a LEM II/III mechanism) and says so.

    LEM IV replaces the SSB->DSB cluster-enhanced tail with the DSB-pattern /
    cluster-index formulation, so a ClusterParameters on the photon object has no
    effect. It must warn rather than silently drop it.
    """
    photon = PhotonParameters(alpha_x=0.184, beta_x=0.02, D_t=17.0, cluster=ClusterParameters())
    with pytest.warns(UserWarning, match="LEM IV does not use photon.cluster"):
        LEM4(photon=photon, cell=V79_CELL)

    # No warning when cluster is absent (the documented fixture path).
    import warnings

    with warnings.catch_warnings():
        warnings.simplefilter("error")
        LEM4(photon=PhotonParameters(alpha_x=0.184, beta_x=0.02, D_t=17.0), cell=V79_CELL)


def test_lem4_rejects_lethal_central_initial_slope() -> None:
    """The TRiP-rapid N_1/D_1 slope is undefined for LEM IV's DSB-pattern override.

    A faithful TRiP98 LEM IV reproduction would need a cross-section rework that is
    not implemented, so the flag must fail loudly rather than
    silently apply a meaningless formula -- at construction, not at first use.
    """
    with pytest.raises(ValueError, match="lethal_central"):
        LEM4(photon=V79, cell=V79_CELL, initial_slope_model="lethal_central")


def test_lem4_numba_engine_matches_numpy() -> None:
    """The compiled parallel kernel agrees with the NumPy path to <0.5% on RBE_alpha."""
    from pyradbio.models.lem._lem4_kernel import HAVE_NUMBA

    if not HAVE_NUMBA:
        pytest.skip("numba (fast extra) not installed")
    lnp = LEM4(photon=V79, cell=V79_CELL, engine="numpy")
    lnb = LEM4(photon=V79, cell=V79_CELL, engine="numba")
    assert lnp._use_numba() is False
    assert lnb._use_numba() is True
    for let in (30.0, 90.0, 200.0):
        ion = Ion(6, 12, _carbon_energy_for_let(let))
        assert lnb.alpha_beta(ion).alpha == pytest.approx(lnp.alpha_beta(ion).alpha, rel=5e-3)


def test_lem4_engine_validation() -> None:
    """Bad engine name is rejected; numba needs the built-in Scholz-Kraft dose."""
    from pyradbio.models.lem._lem4_kernel import HAVE_NUMBA

    with pytest.raises(ValueError, match="engine"):
        LEM4(photon=V79, cell=V79_CELL, engine="fast")  # type: ignore[arg-type]
    if HAVE_NUMBA:
        # A custom radial dose without radical diffusion is numpy-only.
        custom = ScholzKraftRadialDose.lem1(stopping_power=_SP)
        lem = LEM4(photon=V79, cell=V79_CELL, radial_dose=custom, engine="numba")
        with pytest.raises(ValueError, match="Scholz-Kraft radical-diffusion"):
            lem.alpha_beta(Ion(6, 12, 100.0))
        # ...but engine="auto" silently falls back to numpy for it.
        auto = LEM4(photon=V79, cell=V79_CELL, radial_dose=custom, engine="auto")
        assert auto._use_numba() is False


def test_lem4_overkill_region_has_no_jumping() -> None:
    """The deep-overkill rise (very high LET / low energy) is smooth.

    As the delta-ray range drops below the 540 nm subvolume the track collapses
    into one loop column and ``C_I -> 1``; without the legacy TRiP98 cap the raw
    RBE_alpha jumps ~40% between adjacent energies. A radioresistant cell (small
    alpha_x) amplifies it most.

    This is a statement about the *legacy* configuration, so it is pinned to it.
    The uncapped default has no such jump -- below 2.206 MeV/u it reaches ``D_eq``
    by matching the single loop's Poisson mean (`LEM4.mean_field_deq`), which
    is smooth in energy because ``m`` is -- but it lands several times above the
    TRiP98 tables, so the smoothness is not by itself evidence of correctness.
    """
    photon = PhotonParameters(alpha_x=0.02, beta_x=0.005, D_t=20.0)
    lem = LEM4(photon=photon, cell=CellTarget(5.0), legacy_trip98_cluster_index_cap=0.95)
    energies = np.geomspace(1.2, 12.0, 24)
    rbe = np.array([lem.alpha_beta(Ion(6, 12, float(e))).alpha / photon.alpha_x for e in energies])
    rising = rbe[: int(np.argmax(rbe)) + 1]  # deep-overkill up to the interior peak
    ratios = rising[1:] / rising[:-1]
    assert ratios.min() > 0.95, f"overkill jumping: min rise ratio {ratios.min():.3f} in {rising}"


# ---- GLOBLE per-loop effect model (Friedrich, Durante & Scholz 2012) ----
def test_lem4_effect_model_validation() -> None:
    """``effect_model`` must be a known value and ``globle_n_loops`` positive."""
    with pytest.raises(ValueError, match="effect_model"):
        LEM4(photon=V79, cell=V79_CELL, effect_model="dose")  # type: ignore[arg-type]
    with pytest.raises(ValueError, match="globle_n_loops"):
        LEM4(photon=V79, cell=V79_CELL, effect_model="globle", globle_n_loops=0.0)
    # default is the photon-equivalent-dose mapping (the current model, unchanged).
    assert LEM4(photon=V79, cell=V79_CELL).effect_model == "photon_equivalent"


def test_lem4_globle_lethalities_match_friedrich2012_figure2() -> None:
    """GLOBLE per-loop lethalities reproduce the paper's Fig. 2 values.

    Reference: Friedrich, Durante & Scholz 2012 (doi:10.1667/RR2964.1), eqs. 8-9 and
    Fig. 2, which use eps_iDSB = 0.0036 and eps_cDSB = 0.18 with a_DSB = 30/Gy and
    N_L = 3000 loops. Those correspond to photon LQ alpha = eps_i*a_DSB = 0.108 Gy^-1
    and beta = (eps_c*a_DSB^2/2 - a_DSB*alpha)/N_L = 0.0259 Gy^-2 (inverting eq. 9).
    """
    alpha = 0.0036 * DSB_PER_GY_CELL  # 0.108
    beta = (0.18 * DSB_PER_GY_CELL**2 / 2.0 - DSB_PER_GY_CELL * alpha) / 3000.0
    photon = PhotonParameters(alpha_x=alpha, beta_x=beta, D_t=15.0)
    lem = LEM4(photon=photon, cell=V79_CELL, effect_model="globle", globle_n_loops=3000.0)
    eps_i, eps_c = lem._globle_lethalities()
    assert eps_i == pytest.approx(0.0036, rel=1e-9)
    assert eps_c == pytest.approx(0.18, rel=1e-3)


def test_lem4_globle_photon_effect_recovers_lq_at_low_dose() -> None:
    """The GLOBLE photon effect n_i*eps_i + n_c*eps_c -> alpha*D + beta*D^2 at low dose.

    Reference: Friedrich 2012 eqs. 2-3, 8-9. The Poisson loop counts n_i, n_c at dose D
    (Y_loop = a_DSB/N_L per Gy per loop) folded with the per-loop lethalities must
    reproduce the photon LQ in the linear-quadratic regime.
    """
    alpha, beta, n_loops = 0.15, 0.03, 3000.0
    photon = PhotonParameters(alpha_x=alpha, beta_x=beta, D_t=15.0)
    lem = LEM4(photon=photon, cell=V79_CELL, effect_model="globle", globle_n_loops=n_loops)
    eps_i, eps_c = lem._globle_lethalities()
    y_loop = DSB_PER_GY_CELL / n_loops
    # LQ agreement holds in the low-dose limit; GLOBLE deliberately curves below LQ
    # at higher dose (the loop-saturation feature, Friedrich 2012 Fig. 4), e.g. ~0.4%
    # low by 2 Gy, so this pins only the LQ-valid regime.
    for d in (0.25, 0.5, 1.0):
        n_i = n_loops * (y_loop * d) * np.exp(-y_loop * d)
        n_c = n_loops * (1.0 - np.exp(-y_loop * d) * (1.0 + y_loop * d))
        effect = n_i * eps_i + n_c * eps_c
        assert effect == pytest.approx(alpha * d + beta * d * d, rel=2e-3)


def test_lem4_globle_entrance_channel_photon_limit() -> None:
    """GLOBLE RBE_alpha ~ 1 at low LET (high energy), like the photon-equivalent model."""
    lem = LEM4(photon=V79, cell=V79_CELL, effect_model="globle")
    for e in (150.0, 400.0, 1000.0):  # ~19, ~11, ~8 keV/um for carbon
        rbe = lem.alpha_beta(Ion(6, 12, e)).alpha / V79.alpha_x
        assert 0.9 < rbe < 1.4, f"GLOBLE entrance-channel RBE_alpha={rbe:.2f} at {e:.0f} MeV/u"


@pytest.mark.parametrize("effect_model", ["globle", "photon_equivalent"])
def test_lem4_low_let_ion_is_marginally_more_effective_than_photons(effect_model: str) -> None:
    """A low-LET ion (~10 keV/um) is marginally *more* effective than photons, never less.

    ``RBE_alpha -> 1`` from *above* as LET falls, so ``alpha_x <= alpha_ion``. It catches
    the ``C_I -> 0`` degeneracy class where a per-region ``D_eq`` score floors at zero
    effect and makes the ion *less* effective than photons. The bounded
    GLOBLE and the ``D_eq`` photon-equivalent effect models both hold it; the
    single-particle default path has a documented low-LET dip below 1 and is intentionally
    excluded (see ``test_lem4_high_energy_rbe_alpha_near_one``).
    """
    lem = LEM4(photon=V79, cell=V79_CELL, effect_model=effect_model)
    for let in (10.0, 12.0, 15.0):
        rbe = lem.alpha_beta(Ion(6, 12, _carbon_energy_for_let(let))).alpha / V79.alpha_x
        assert 1.0 <= rbe < 1.4, f"{effect_model} RBE_alpha={rbe:.3f} at LET {let:.0f} keV/um"


def test_lem4_globle_deep_overkill_smooth_and_finite_without_cap() -> None:
    """GLOBLE is bounded: the deep-overkill rise is smooth AND finite with no cap.

    The photon-equivalent model needs ``_CLUSTER_INDEX_MAX`` to tame the singular
    D_eq inversion as C_I -> 1; the GLOBLE per-loop effect is bounded by construction
    (E = n_i eps_i + n_c eps_c), so a radioresistant cell shows a smooth monotone
    overkill rise to the interior peak with no jumps and no divergence at E = 1 MeV/u.
    """
    photon = PhotonParameters(alpha_x=0.02, beta_x=0.005, D_t=20.0)
    lem = LEM4(photon=photon, cell=CellTarget(5.0), effect_model="globle")
    energies = np.geomspace(1.0, 12.0, 24)
    rbe = np.array([lem.alpha_beta(Ion(6, 12, float(e))).alpha / photon.alpha_x for e in energies])
    assert np.all(np.isfinite(rbe)) and np.all(rbe > 0.0)
    rising = rbe[: int(np.argmax(rbe)) + 1]
    ratios = rising[1:] / rising[:-1]
    assert ratios.min() > 0.95, f"GLOBLE overkill jumping: min rise ratio {ratios.min():.3f}"


def test_lem4_globle_n_loops_is_configurable() -> None:
    """globle_n_loops tunes the clustered-DSB (high-LET) lethality; low LET is unchanged.

    eps_c = 2(N_L beta + a_DSB alpha)/a_DSB^2 grows with N_L, so a denser (high-LET)
    track's RBE_alpha rises with globle_n_loops, while the entrance channel (iDSB /
    alpha-channel dominated) is insensitive to it.
    """
    hi_let = Ion(6, 12, _carbon_energy_for_let(200.0))
    lo_let = Ion(6, 12, 400.0)  # entrance channel, ~11 keV/um (few cDSB)
    lem_2000 = LEM4(photon=V79, cell=V79_CELL, effect_model="globle", globle_n_loops=2000.0)
    lem_3000 = LEM4(photon=V79, cell=V79_CELL, effect_model="globle", globle_n_loops=3000.0)
    hi_2000, hi_3000 = lem_2000.alpha_beta(hi_let).alpha, lem_3000.alpha_beta(hi_let).alpha
    lo_2000, lo_3000 = lem_2000.alpha_beta(lo_let).alpha, lem_3000.alpha_beta(lo_let).alpha
    # higher N_L -> larger clustered-DSB lethality -> higher high-LET RBE_alpha
    assert hi_3000 > hi_2000
    # the entrance channel (iDSB / alpha-channel dominated) is far less sensitive
    assert abs(lo_3000 - lo_2000) / lo_2000 < 0.3 * abs(hi_3000 - hi_2000) / hi_2000


@pytest.mark.golden
def test_golden_lem4_globle_v79_carbon_peak() -> None:
    """GLOBLE V79 carbon RBE_alpha: single interior maximum, magnitude near the paper.

    Reference: Elsasser et al. 2010 (doi:10.1016/j.ijrobp.2010.05.014, Fig. 4a) for the
    RBE_alpha(LET) target shape, and the GLOBLE effect of Friedrich, Durante & Scholz 2012
    (Radiat. Res. 178(5):385-394, doi:10.1667/RR2964.1) for the bounded per-loop effect
    itself. The bounded effect reproduces the RBE_alpha(LET) shape -- ~1 in the entrance
    channel to a single interior maximum (paper ~6.5) and an overkill drop -- without the
    D_eq cap.

    ``globle_n_loops`` (N_L) is the dominant lever: the paper's illustration constant
    N_L=3000 (= S_G/S_L, used to draw its single example curves) gives a peak of 8.52
    (+31%), whereas its own *empirically best-fit* N_L=2000 (Fig. 5B: "best agreement
    with the average values for the large data ensemble" of >150 photon survival
    curves) is the default. The paper never applies GLOBLE to ions
    itself -- it flags that as future work -- so there is no published ion-RBE N_L to
    defer to; N_L=2000 is the better-grounded default (see ``GLOBLE_N_LOOPS`` in
    ``lem4.py``) and brings the peak to ~7.0 (+8%), within the golden bound below.
    """
    lem = LEM4(photon=V79, cell=V79_CELL, effect_model="globle")
    _lets, rbe = _rbe_alpha_curve(lem, CARBON_IONS)
    i_peak = int(np.argmax(rbe))
    assert 0 < i_peak < len(rbe) - 1, "GLOBLE RBE_alpha maximum is not interior"
    assert rbe[0] == pytest.approx(1.0, abs=0.4), f"GLOBLE low-LET RBE_alpha={rbe[0]:.2f}"
    assert 5.5 < rbe[i_peak] < 7.5, f"GLOBLE RBE_alpha max = {rbe[i_peak]:.2f} not near paper ~6.5"
    assert rbe[-1] < rbe[i_peak], "no overkill drop at the highest LET"


def test_lem4_high_energy_rbe_alpha_near_one() -> None:
    """Entrance-channel RBE_alpha ~ 1 at high energy / low LET (escaped-energy term).

    When r_max >> R_eff most delta-ray energy escapes the nucleus while D_1 counts
    the full LET, so the central traversal alone gives RBE_alpha < 1 (~0.5-0.9). The
    escaped-energy indirect term (alpha_x * (1 - D_1^in/D_1)) restores the photon
    limit, tracking the TRiP98 LEM IV entrance channel.
    """
    lem = LEM4(photon=V79, cell=V79_CELL)
    for e in (150.0, 400.0, 1000.0):  # ~19, ~11, ~8 keV/um for carbon
        rbe = lem.alpha_beta(Ion(6, 12, e)).alpha / V79.alpha_x
        assert 0.9 < rbe < 1.4, f"entrance-channel RBE_alpha={rbe:.2f} at {e:.0f} MeV/u"


def test_photon_cluster_index_monotone_zero_to_one() -> None:
    """C_g(D) grows monotonically from 0 to 1 (Poisson >=2 / >=1 fraction)."""
    doses = np.logspace(-1, 5, 50)
    cg = [LEM4.photon_cluster_index(float(d)) for d in doses]
    assert LEM4.photon_cluster_index(0.0) == 0.0
    assert cg == sorted(cg)
    assert cg[0] < 0.01  # negligible clustering at low dose
    assert 0.0 < LEM4.photon_cluster_index(50.0) < 1.0  # a genuine interior fraction
    assert cg[-1] == pytest.approx(1.0, abs=1e-3)  # saturates at high dose


def test_photon_equivalent_dose_round_trips() -> None:
    """D_eq(C) inverts C_g(D): C_g(D_eq(C)) == C, and D_eq is monotone in C."""
    lem = LEM4(photon=V79, cell=V79_CELL)
    cs = [0.02, 0.05, 0.1, 0.2, 0.4]
    deqs = [lem.photon_equivalent_dose(c) for c in cs]
    for c, d_eq in zip(cs, deqs, strict=True):
        assert lem.photon_cluster_index(d_eq) == pytest.approx(c, rel=1e-6)
    assert deqs == sorted(deqs)  # larger cluster index -> larger equivalent dose


def test_photon_equivalent_dose_rejects_out_of_range() -> None:
    """Cluster index must be a proper fraction in (0, 1)."""
    lem = LEM4(photon=V79, cell=V79_CELL)
    for bad in (0.0, 1.0, -0.1, 1.5):
        with pytest.raises(ValueError):
            lem.photon_equivalent_dose(bad)


def test_single_traversal_is_a_cut_out_of_the_photon_pattern() -> None:
    """The scaling factor kappa < 1: one traversal damages only a 'cut-out'.

    Friedrich 2012 (Fig. 2a discussion): a single traversal produces a small
    fraction of the damaged loops that the photon-equivalent dose produces over
    the whole nucleus, so kappa = N_total,Ion / N_total,g(D_eq) < 1.
    """
    lem = LEM4(photon=V79, cell=V79_CELL)
    ion = Ion(6, 12, _carbon_energy_for_let(120.0))
    kappa = lem.scaling_factor(ion)
    assert 0.0 < kappa < 1.0
    # N_1 = kappa * (eta-enhanced) photon effect at D_eq is consistent with the helper.
    c_i = lem.cluster_index(ion)
    d_eq = lem.photon_equivalent_dose(c_i)
    assert lem.single_particle_log_kill(ion) == pytest.approx(kappa * lem._photon_effect(d_eq))


# ---- golden shape reproductions (Friedrich 2012 Figs. 2-3) --------------
@pytest.mark.golden
def test_golden_lem4_v79_carbon_rbe_alpha_shape() -> None:
    """RBE_alpha(LET) for V79 carbon: single interior maximum near ~150 keV/um.

    Reference: Friedrich et al. 2012, Fig. 3 (doi:10.3109/09553002.2011.611213);
    photon inputs alpha_g=0.184, beta_g=0.02, D_t=17 Gy, R_n=4.7 um. The model
    RBE_alpha rises from ~1 in the entrance channel to a single maximum in the
    intermediate-high LET range, then drops in the overkill region.

    The peak *position* (~160-220 keV/um) and the overall shape are reproduced,
    and the discrete 540 nm subvolume scoring + Friedrich-2015 eta yield brings the
    absolute peak (~5) close to the paper's ~6.5 (LEM IV itself under-predicts V79,
    Pfuhl 2022) -- see docs/references/LEM_IV_Friedrich2012.md.
    """
    lem = LEM4(photon=V79, cell=V79_CELL)
    lets, rbe = _rbe_alpha_curve(lem, CARBON_IONS)

    i_peak = int(np.argmax(rbe))
    # Single interior maximum: strictly rising up to the peak, falling after.
    assert 0 < i_peak < len(rbe) - 1, "RBE_alpha maximum is not interior"
    assert rbe[: i_peak + 1] == sorted(rbe[: i_peak + 1]), "not monotone rising to the peak"
    assert rbe[i_peak:] == sorted(rbe[i_peak:], reverse=True), "not monotone falling after the peak"

    # Peak position in the intermediate-high LET band (paper ~150-200 keV/um).
    assert 100.0 <= lets[i_peak] <= 260.0, f"RBE_alpha peaks at {lets[i_peak]:.0f} keV/um"
    # Photon limit toward low LET; a pronounced maximum approaching the paper's ~6.5;
    # overkill drop at high LET.
    assert rbe[0] == pytest.approx(1.0, abs=0.4), f"low-LET RBE_alpha={rbe[0]:.2f}"
    assert 4.0 < rbe[i_peak] < 7.0, f"RBE_alpha max = {rbe[i_peak]:.2f} not near the paper ~6.5"
    assert rbe[-1] < rbe[i_peak], "no overkill drop at the highest LET"


@pytest.mark.golden
def test_golden_lem4_rbe10_below_rbe_alpha() -> None:
    """RBE_10 lies below RBE_alpha across the LET range (Fig. 3 dashed < solid).

    Reference: Friedrich et al. 2012, Fig. 3: the RBE for 10% survival (dashed)
    is below the initial-slope RBE_alpha (solid), because the ion curve keeps a
    finite quadratic shoulder (beta_I > 0) so higher-dose survival is relatively
    less enhanced. This holds where RBE_alpha > 1 (the therapeutic range the paper
    shows); below ~30 keV/um the single-particle approximation dips RBE_alpha below
    1 (documented low-LET limitation), where the relation need not hold.
    """
    lem = LEM4(photon=V79, cell=V79_CELL)
    lets, rbe_alpha = _rbe_alpha_curve(lem, CARBON_IONS)
    rbe10 = [_rbe10(lem, ion) for ion in sorted(CARBON_IONS, key=_SP.let_keV_um)]

    compared = 0
    for let, ra, r10 in zip(lets, rbe_alpha, rbe10, strict=True):
        if ra <= 1.0:  # low-LET single-particle dip; relation not expected there
            continue
        assert r10 < ra, f"RBE_10 {r10:.2f} !< RBE_alpha {ra:.2f} at {let:.0f} keV/um"
        compared += 1
    assert compared >= 5, "too few LET points above the low-LET dip to test"
    assert max(rbe10) > 1.0  # a real enhancement remains at 10% survival
    i10 = int(np.argmax(rbe10))
    # The maximum sits at the 220 grid point (it would fall at ~160 if CellTarget
    # shrank R_n by a 0.8 effective_size_factor), so the upper bound leaves room.
    assert 80.0 <= lets[i10] <= 250.0, f"RBE_10 peaks at {lets[i10]:.0f} keV/um"


@pytest.mark.golden
def test_golden_lem4_reproduces_friedrich2012_v79_carbon_figure3() -> None:
    """LEM IV's V79 carbon RBE_alpha/RBE_10(LET) reproduce Friedrich et al.
    2012's own digitized Fig. 3 (p. 106) model line and experimental data,
    point-by-point -- the quantitative counterpart to
    ``test_golden_lem4_v79_carbon_rbe_alpha_shape``/``..._rbe10_below_rbe_alpha``
    above, which only pin the *shape* against the paper's qualitative
    description.

    Reference: Friedrich, T. et al. 2012 (doi:10.3109/09553002.2011.611213),
    Fig. 3 -- V79 cells, carbon ions, Furusawa et al. (2000) experimental data
    -- digitized in ``pyradbio.benchmark.friedrich2012`` (hand-digitized).
    The model-line reproduction is close (RMS ~2-3% of peak): the paper's own
    RBE_alpha peaks at 6.67 near LET 186 keV/um, pyRadBio reaches 6.60 there
    (-1%); the point-by-point digitized comparison shows near-quantitative
    agreement.
    """
    # Fig. 3 runs into deep overkill (its lowest LET points are ~0.4 MeV/u carbon),
    # where C_I saturates. The uncapped default is well posed there (it matches the
    # loop's Poisson mean instead) but runs far above the published line, so
    # reproducing Fig. 3 REQUIRES the legacy TRiP98-tuned cap. Pinning it here
    # records exactly that: the match is contingent on that constant, not derived.
    lem = LEM4(photon=V79, cell=V79_CELL, legacy_trip98_cluster_index_cap=0.95)

    model_alpha = f12.get_curve("fig3_v79_rbe_vs_let", series="calc", variant="rbe_alpha")
    rms_model_alpha, n_model_alpha = _rms_alpha_vs_curve(lem, 6, 12, V79.alpha_x, model_alpha)
    assert n_model_alpha >= 35, f"too few comparable model RBE_alpha points ({n_model_alpha})"
    assert rms_model_alpha < 0.2, f"RBE_alpha RMS vs Fig. 3 model line = {rms_model_alpha:.3f}"

    model_rbe10 = f12.get_curve("fig3_v79_rbe_vs_let", series="calc", variant="rbe_10")
    rms_model_rbe10, n_model_rbe10 = _rms_rbe10_vs_curve(
        lem, 6, 12, V79.alpha_x, V79.beta_x, model_rbe10
    )
    assert n_model_rbe10 >= 30, f"too few comparable model RBE_10 points ({n_model_rbe10})"
    assert rms_model_rbe10 < 0.1, f"RBE_10 RMS vs Fig. 3 model line = {rms_model_rbe10:.3f}"

    data_alpha = f12.get_curve("fig3_v79_rbe_vs_let", series="data", variant="rbe_alpha")
    rms_data_alpha, n_data_alpha = _rms_alpha_vs_curve(lem, 6, 12, V79.alpha_x, data_alpha)
    assert n_data_alpha >= 18, f"too few comparable data RBE_alpha points ({n_data_alpha})"
    assert rms_data_alpha < 1.2, (
        f"RBE_alpha RMS vs Furusawa 2000 experimental data = {rms_data_alpha:.3f}"
    )

    data_rbe10 = f12.get_curve("fig3_v79_rbe_vs_let", series="data", variant="rbe_10")
    rms_data_rbe10, n_data_rbe10 = _rms_rbe10_vs_curve(
        lem, 6, 12, V79.alpha_x, V79.beta_x, data_rbe10
    )
    assert n_data_rbe10 >= 18, f"too few comparable data RBE_10 points ({n_data_rbe10})"
    assert rms_data_rbe10 < 0.85, (
        f"RBE_10 RMS vs Furusawa 2000 experimental data = {rms_data_rbe10:.3f}"
    )


@pytest.mark.golden
def test_golden_lem4_cluster_index_and_deq_rise_with_let() -> None:
    """Cluster index and photon-equivalent dose increase with LET.

    Reference: Friedrich et al. 2012, Fig. 2b: a denser track (higher LET)
    clusters more loops, so C_I rises with LET; since C_g(D) is monotone, the
    matched photon-equivalent dose D_eq rises with C_I as well.
    """
    lem = LEM4(photon=V79, cell=V79_CELL)
    ions = sorted(CARBON_IONS, key=_SP.let_keV_um)
    c_i = [lem.cluster_index(ion) for ion in ions]
    d_eq = [lem.photon_equivalent_dose(c) for c in c_i]

    assert c_i == sorted(c_i), f"cluster index not monotone in LET: {c_i}"
    assert d_eq == sorted(d_eq), "photon-equivalent dose not monotone in cluster index"
    assert 0.0 < c_i[0] < c_i[-1] < 1.0


@pytest.mark.golden
def test_golden_lem4_discrete_cluster_index_matches_pfuhl() -> None:
    """Discrete 3D single-traversal cluster index reproduces the Pfuhl Table 3 anchors.

    Reference: Pfuhl, Friedrich & Scholz 2019/2020 (Radiat. Res. 193:130,
    doi:10.1667/RR15456.1, Table 3): 12C 2.5 MeV/u (LET ~437) C_I ~0.80 and 20Ne
    31 MeV/u (LET ~183) C_I ~0.19. The discrete average-of-ratios reproduces them, while
    the mean-field `cluster_index` ratio-of-averages over-clusters (~0.99 / ~0.31) --
    this asserts the discrete value both matches Pfuhl and sits well below the mean-field.
    Seed-pinned; MC scatter is < 0.004 over seeds at this realisation count.
    """
    lem = LEM4(photon=V79, cell=V79_CELL)
    carbon, neon = Ion(6, 12, 2.5), Ion(10, 20, 31.0)

    c_carbon = lem.discrete_cluster_index(carbon, n_realisations=4000, seed=1)
    c_neon = lem.discrete_cluster_index(neon, n_realisations=4000, seed=1)

    assert 0.70 <= c_carbon <= 0.82, f"12C discrete C_I={c_carbon:.3f} not near Pfuhl 0.80"
    assert 0.16 <= c_neon <= 0.23, f"20Ne discrete C_I={c_neon:.3f} not near Pfuhl 0.19"
    # the mean-field areal index over-clusters -- the whole reason this method exists
    assert c_carbon < lem.cluster_index(carbon) - 0.15
    assert c_neon < lem.cluster_index(neon) - 0.08


def _ion_for_let(z: int, a: int, let: float) -> Ion | None:
    """Ion ``(z, a)`` at ``let``, or ``None`` outside the reachable energy range."""
    try:
        e = energy_for_let_MeV_u(_SP, z, a, let, branch="bracket", energy_bounds_MeV_u=(0.4, 600.0))
    except ValueError:
        return None
    return Ion(z, a, e)


class _RbeCurve(Protocol):
    """Structural type for digitized ``(LET, RBE)`` benchmark curves.

    Both `pyradbio.benchmark.elsasser2010.Elsasser2010Curve` and
    `pyradbio.benchmark.friedrich2012.Friedrich2012Curve` satisfy this.
    Declared as read-only properties (not plain attributes) so frozen
    dataclasses -- whose fields mypy treats as non-writable -- can match.
    """

    @property
    def x(self) -> tuple[float, ...]: ...
    @property
    def y(self) -> tuple[float, ...]: ...


def _rms_alpha_vs_curve(
    lem: LEM4, z: int, a: int, alpha_x: float, curve: _RbeCurve
) -> tuple[float, int]:
    """RMS of ``lem``'s RBE_alpha against a digitized ``(LET, RBE_alpha)`` curve."""
    residuals = [
        lem.alpha_beta(ion).alpha / alpha_x - rbe
        for let, rbe in zip(curve.x, curve.y, strict=True)
        if (ion := _ion_for_let(z, a, let)) is not None
    ]
    return float(np.sqrt(np.mean(np.square(residuals)))), len(residuals)


def _rms_rbe10_vs_curve(
    lem: LEM4, z: int, a: int, alpha_x: float, beta_x: float, curve: _RbeCurve
) -> tuple[float, int]:
    """RMS of ``lem``'s RBE_10 against a digitized ``(LET, RBE_10)`` curve."""
    d10_x = dose_for_survival(0.1, alpha_x, beta_x)
    residuals = []
    for let, rbe10 in zip(curve.x, curve.y, strict=True):
        ion = _ion_for_let(z, a, let)
        if ion is None:
            continue
        lq = lem.alpha_beta(ion)
        residuals.append(d10_x / dose_for_survival(0.1, lq.alpha, lq.beta) - rbe10)
    return float(np.sqrt(np.mean(np.square(residuals)))), len(residuals)


@pytest.mark.golden
def test_golden_lem4_reproduces_elsasser2010_v79_figure4a() -> None:
    """LEM IV's V79 proton and helium RBE_alpha(LET) reproduce Elsasser et al.
    2010's own digitized Fig. 4a model lines and experimental data.

    Reference: Elsasser et al. 2010, Fig. 4a (p. 1181) -- V79 cells, protons +
    helium (Belli et al. 1998) -- digitized in ``pyradbio.benchmark.elsasser2010``
    (hand-digitized in full; see that module's docstring). All four proton/helium x
    model/data comparisons are restricted to LET<=25 keV/um: above that our
    single-particle model enters a deep-overkill regime and diverges sharply
    from the paper's own proton model line (which itself levels off near ~6.5
    while ours keeps rising past 9 by LET 40, at proton energies that are
    still nominally within ``SimpleBetheStoppingPower``'s valid range) -- the
    same kind of low-energy/deep-overkill limitation already documented for
    carbon (``ROADMAP.md``), evident here for protons at their
    much lower absolute LET scale.
    """
    v79 = es10.FIGURE_INPUTS["fig4a_v79"]
    photon = PhotonParameters(alpha_x=v79.alpha_x, beta_x=v79.beta_x, D_t=v79.D_t)
    cell = CellTarget(v79.nucleus_radius_um, effective_size_factor=v79.effective_size_factor)
    # Legacy-capped for the same reason as the Fig. 4b golden below: the digitized
    # series reach below 2.206 MeV/u, where C_I saturates, and matching the
    # published line there is contingent on the TRiP98-tuned cap.
    lem = LEM4(photon=photon, cell=cell, legacy_trip98_cluster_index_cap=0.95)

    def clip(curve: es10.Elsasser2010Curve, let_max: float) -> es10.Elsasser2010Curve:
        xy = [(x, y) for x, y in zip(curve.x, curve.y, strict=True) if x <= let_max]
        return dataclasses.replace(curve, x=tuple(x for x, _ in xy), y=tuple(y for _, y in xy))

    # Points digitized twice at (nearly) the same LET are an error-bar top/bottom
    # pair, already consolidated to one point per measurement (with y_err) by the
    # loader -- so this LET<=25 cut leaves fewer, but each is a real measurement.
    proton_model = clip(
        es10.get_curve("fig4a_v79_rbealpha_vs_let", particle="proton", series="calc"), 25.0
    )
    rms_p_model, n_p_model = _rms_alpha_vs_curve(lem, 1, 1, v79.alpha_x, proton_model)
    assert n_p_model >= 15, f"too few comparable proton model points ({n_p_model})"
    assert rms_p_model < 0.35, f"proton RBE_alpha RMS vs Fig. 4a model line = {rms_p_model:.3f}"

    proton_data = clip(
        es10.get_curve("fig4a_v79_rbealpha_vs_let", particle="proton", series="data"), 25.0
    )
    rms_p_data, n_p_data = _rms_alpha_vs_curve(lem, 1, 1, v79.alpha_x, proton_data)
    assert n_p_data >= 6, f"too few comparable proton data points ({n_p_data})"
    # The bound is 0.65 (measured RMS 0.62); the proton points are digitized at the
    # drawn marker centres. The model-line comparisons are the ones that matter,
    # since a reconstruction is scored against model lines.
    assert rms_p_data < 0.65, (
        f"proton RBE_alpha RMS vs Belli 1998 data (LET<=25) = {rms_p_data:.3f}"
    )

    helium_model = es10.get_curve("fig4a_v79_rbealpha_vs_let", particle="helium", series="calc")
    rms_he_model, n_he_model = _rms_alpha_vs_curve(lem, 2, 4, v79.alpha_x, helium_model)
    assert n_he_model >= 8, f"too few comparable helium model points ({n_he_model})"
    assert rms_he_model < 0.8, f"helium RBE_alpha RMS vs Fig. 4a model line = {rms_he_model:.3f}"

    # Helium's experimental series is sparse (5 points, Belli et al. 1998) and
    # spans LET 20-79 keV/um -- well above the proton LET<=25 cap, so it is not
    # clipped; it is a hand digitization (see the module docstring).
    helium_data = es10.get_curve("fig4a_v79_rbealpha_vs_let", particle="helium", series="data")
    rms_he_data, n_he_data = _rms_alpha_vs_curve(lem, 2, 4, v79.alpha_x, helium_data)
    assert n_he_data >= 4, f"too few comparable helium data points ({n_he_data})"
    assert rms_he_data < 0.85, f"helium RBE_alpha RMS vs Belli 1998 data = {rms_he_data:.3f}"


@pytest.mark.golden
@pytest.mark.parametrize("particle,z,a", [("carbon", 6, 12), ("helium", 2, 4)])
def test_golden_lem4_reproduces_elsasser2010_hsg_figure4b(particle: str, z: int, a: int) -> None:
    """LEM IV's HSG RBE_alpha(LET)/RBE_10(LET) reproduce Elsasser et al. 2010's
    own digitized Fig. 4b model lines and experimental data, for both ions.

    Reference: Elsasser et al. 2010 (doi:10.1016/j.ijrobp.2010.05.014), Fig. 4b
    (p. 1181), the primary LEM IV paper's own monoenergetic HSG carbon+helium
    RBE(LET) panel -- digitized in ``pyradbio.benchmark.elsasser2010`` (hand-
    digitized in full). Unlike
    ``test_golden_lem4_v79_carbon_rbe_alpha_shape`` (which only pins the *shape*
    against the Friedrich 2012 proceedings note's qualitative description), this
    compares point-by-point against a hand-digitized reference curve, the same
    standard the LEM I-III reconstructions hold (see e.g.
    ``test_golden_lem3_reproduces_paper_per_cell_bragg_deviation``).
    """
    hsg = es10.FIGURE_INPUTS["fig4b_hsg"]
    photon = PhotonParameters(alpha_x=hsg.alpha_x, beta_x=hsg.beta_x, D_t=hsg.D_t)
    cell = CellTarget(hsg.nucleus_radius_um, effective_size_factor=hsg.effective_size_factor)
    # Legacy-capped for the same reason as the Friedrich Fig. 3 golden: the helium
    # series runs past the point where the whole profile fits inside one giant loop
    # and the single-central-traversal mean field has no cluster index to give.
    # Reproducing the published line there is contingent on the TRiP98-tuned cap.
    lem = LEM4(photon=photon, cell=cell, legacy_trip98_cluster_index_cap=0.95)

    alpha_model = es10.get_curve(
        "fig4b_hsg_rbe_vs_let", particle=particle, series="calc", variant="model_rbe_alpha"
    )
    rms_alpha, n_alpha = _rms_alpha_vs_curve(lem, z, a, hsg.alpha_x, alpha_model)
    assert n_alpha >= 120, f"too few comparable RBE_alpha points ({n_alpha})"
    # Measured RMS is 0.12 (carbon) and 0.18 (helium); the bound is kept tight so
    # it protects. See the y-axis notes in the elsasser2010 module docstring.
    assert rms_alpha < 0.25, f"{particle} RBE_alpha RMS vs Fig. 4b model line = {rms_alpha:.3f}"

    rbe10_model = es10.get_curve(
        "fig4b_hsg_rbe_vs_let", particle=particle, series="calc", variant="model_rbe_10"
    )
    rms_rbe10, n_rbe10 = _rms_rbe10_vs_curve(lem, z, a, hsg.alpha_x, hsg.beta_x, rbe10_model)
    assert n_rbe10 >= 120, f"too few comparable RBE_10 points ({n_rbe10})"
    # Measured RMS is 0.05 (carbon) and 0.15 (helium).
    assert rms_rbe10 < 0.22, f"{particle} RBE_10 RMS vs Fig. 4b model line = {rms_rbe10:.3f}"

    # Helium carries 13 RBE_alpha / 12 RBE_10 points, carbon 16 / 16 -- every
    # marker the panel draws. Fig. 4b has no error bars, and pairs of DISTINCT
    # Furusawa measurements at neighbouring LETs (18.5 and 18.6, 23.6 and 24.0, ...)
    # must not be merged into one midpoint each. See the elsasser2010 module docstring.
    alpha_data = es10.get_curve(
        "fig4b_hsg_rbe_vs_let", particle=particle, series="data", variant="rbe_alpha"
    )
    rms_alpha_exp, n_alpha_exp = _rms_alpha_vs_curve(lem, z, a, hsg.alpha_x, alpha_data)
    assert n_alpha_exp >= 13, f"too few comparable RBE_alpha data points ({n_alpha_exp})"
    assert rms_alpha_exp < 0.95, (
        f"{particle} RBE_alpha RMS vs Furusawa 2000 experimental data = {rms_alpha_exp:.3f}"
    )

    rbe10_data = es10.get_curve(
        "fig4b_hsg_rbe_vs_let", particle=particle, series="data", variant="rbe_10"
    )
    rms_rbe10_exp, n_rbe10_exp = _rms_rbe10_vs_curve(lem, z, a, hsg.alpha_x, hsg.beta_x, rbe10_data)
    assert n_rbe10_exp >= 12, f"too few comparable RBE_10 data points ({n_rbe10_exp})"
    assert rms_rbe10_exp < 0.45, (
        f"{particle} RBE_10 RMS vs Furusawa 2000 experimental data = {rms_rbe10_exp:.3f}"
    )


# ---- mixed-field mixing is inherited unchanged --------------------------
def test_lem4_mixed_field_dose_averages_components() -> None:
    """Mixed-field mixing (dose-average alpha, dose-average sqrt(beta)) inherited."""
    import math

    lem = LEM4(photon=V79, cell=V79_CELL)
    ion_a, ion_b = Ion(6, 12, 30.0), Ion(6, 12, 200.0)
    lq_a, lq_b = lem.alpha_beta(ion_a), lem.alpha_beta(ion_b)

    mix = RadiationField(
        [SpectrumComponent(ion_a, dose_Gy=1.0), SpectrumComponent(ion_b, dose_Gy=1.0)]
    )
    lq_mix = lem.alpha_beta(mix)
    assert lq_mix.alpha == pytest.approx(0.5 * (lq_a.alpha + lq_b.alpha))
    root = 0.5 * (math.sqrt(lq_a.beta) + math.sqrt(lq_b.beta))
    assert lq_mix.beta == pytest.approx(root * root)


@pytest.mark.golden
@pytest.mark.xfail(
    strict=True,
    reason=(
        "The deterministic estimator does NOT reproduce the Pfuhl anchors; "
        "agreement with them is an artefact of the lattice registration defect. It "
        "holds only at R_n = 4.7 um, where a pinned lattice phase happens to "
        "straddle the origin; at R_n = 5.0 um the pinned lattice gives 0.930 / "
        "0.316 against the paper's 0.80 / 0.19. With the registration marginalised "
        "it gives 0.851 / 0.268 at every radius -- carbon inside the band below, "
        "neon outside it. The bounds are deliberately not widened to go green. "
        "The anchors are covered by test_golden_lem4_discrete_cluster_index_matches_pfuhl, whose "
        "sampled estimator still lands on them (0.745 / 0.194)."
    ),
)
def test_golden_lem4_analytic_cluster_index_matches_pfuhl_without_sampling() -> None:
    """The *deterministic* discrete cluster index hits the same Pfuhl Table 3 anchors.

    Reference: Pfuhl, Friedrich & Scholz 2019/2020 (Radiat. Res. 193:130,
    doi:10.1667/RR15456.1, Table 3): 12C 2.5 MeV/u C_I ~0.80, 20Ne 31 MeV/u ~0.19.

    Same average-of-ratios quantity as `LEM4.discrete_cluster_index`, but
    evaluated by quadrature instead of sampled, so it carries no seed and no
    realisation noise. Conditioned on the track position the loop occupancies are
    independent Poisson, so with ``1/B = int_0^1 t^(B-1) dt`` the expectation
    closes into a single 1-D integral of the occupancy generating function.

    Deterministic matters here because this is the quantity a future attempt at
    routing the RBE path through a *physical* cluster index would need, and a
    seeded 4000-realisation MC cannot sit inside `alpha_beta`. (That routing was
    tried and rejected -- see the module notes and "Three cluster-index estimators,
    three energy bands" in docs/references/LEM_IV_Friedrich2012.md.)

    **This does not hold -- see the xfail reason.** It is kept, strict, so that if
    a future change makes the deterministic estimator land on the anchors we are
    told.
    """
    lem = LEM4(photon=V79, cell=V79_CELL)
    carbon, neon = Ion(6, 12, 2.5), Ion(10, 20, 31.0)

    c_carbon = lem.analytic_discrete_cluster_index(carbon)
    c_neon = lem.analytic_discrete_cluster_index(neon)

    assert 0.70 <= c_carbon <= 0.88, f"12C analytic C_I={c_carbon:.3f} not near Pfuhl 0.80"
    assert 0.15 <= c_neon <= 0.23, f"20Ne analytic C_I={c_neon:.3f} not near Pfuhl 0.19"
    # It must agree with the sampled estimator it replaces, and both must sit well
    # below the mean-field areal index.
    assert c_carbon == pytest.approx(lem.discrete_cluster_index(carbon, seed=1), abs=0.10)
    assert c_neon == pytest.approx(lem.discrete_cluster_index(neon, seed=1), abs=0.05)
    assert c_carbon < lem.cluster_index(carbon) - 0.10
    assert c_neon < lem.cluster_index(neon) - 0.08


def test_analytic_cluster_index_is_deterministic() -> None:
    """No seed, no scatter: two calls return the identical float."""
    lem = LEM4(photon=V79, cell=V79_CELL)
    ion = Ion(10, 20, 31.0)
    assert lem.analytic_discrete_cluster_index(ion) == lem.analytic_discrete_cluster_index(ion)


@pytest.mark.parametrize("energy", [0.1, 0.126, 0.2, 0.3])
def test_sub_mev_tracks_are_scored_not_dropped(energy: float) -> None:
    """A track only a few nm wide still has a well-defined cluster index.

    Radical diffusion smears the profile over ``~sigma = 4 nm``, but the scoring
    domain must not stop at the delta-ray range ``r_max`` -- 1.2 nm for carbon at
    0.1 MeV/u. The whole track would then fall *outside* the domain: every subvolume
    scores ``m = 0``, ``N_cDSB`` comes out exactly ``0``, and ``cluster_index``
    raises ``ValueError: cluster_index must be in (0, 1), got 0.0``. Energies
    just above (0.2 MeV/u) would not raise but would lose 62% of the track's energy,
    which shows up as a ~25% dip in ``RBE_alpha`` against both its neighbours.

    TRiP98 tabulates these energies because a distal field edge reaches them, so
    the model has to produce a number here.
    """
    lem = LEM4(photon=V79, cell=V79_CELL, stopping_power=_SP)
    ion = Ion(6, 12, energy)

    # a deeply overkilled track damages at least its own 540 nm column
    n_dmg, n_cdsb, in_nuc = lem._damage_pattern(ion)
    assert n_dmg >= (SUBVOLUME_SIDE_NM * 1e-3) ** 2 * 0.99, f"N_dmg={n_dmg:.4g} below one column"
    assert in_nuc > 0.0, "the track deposits no dose inside the nucleus"
    assert n_cdsb > 0.0

    # The whole profile fits in one giant loop, so C_I saturates -- but the pattern
    # is still one Poisson loop with a finite mean, and D_eq matches that mean
    # instead of inverting the saturated index. The default path must therefore
    # return a number here, which is what TRiP98 tabulating these energies requires.
    assert lem.pattern_is_single_loop(ion)
    m = lem.single_loop_dsb_mean(ion)
    assert m > 0.0 and math.isfinite(m)
    assert math.isclose(
        lem.mean_field_deq(ion), lem.photon_equivalent_dose_from_dsb_mean(m), rel_tol=1e-12
    )
    alpha_mf = lem.initial_slope(ion)
    assert math.isfinite(alpha_mf) and alpha_mf > 0.0

    # ...as do the estimators that resolve 1 - C_I by marginalising over the lattice.
    mc = dataclasses.replace(
        lem, cluster_index_model="discrete_mc", cluster_index_mc_realisations=400
    )
    assert 0.0 < mc.cluster_index(ion) <= 1.0
    alpha = mc.initial_slope(ion)
    assert math.isfinite(alpha) and alpha > 0.0


def test_the_two_deq_routes_agree_across_the_single_loop_seam() -> None:
    """No step where ``mean_field_deq`` switches from matching ``m`` to inverting ``C_g``.

    The two are the same equation -- ``C_g`` is strictly monotone in the photon
    loop's Poisson mean alone, so matching ``C_I`` and matching ``m`` coincide
    wherever the ion pattern is a single loop. They must therefore join without a
    step at the 2.206 MeV/u boundary, and a step would mean one of the two is not
    solving what it claims to.

    Guards the seam specifically, straddling it by 0.5e-3 MeV/u. Do not widen the
    bracket to make this pass: ``RBE_alpha`` really does fall by half between 2.3
    and 2.5 MeV/u as penumbra loops appear and ``C_I`` comes off saturation, so a
    loose bracket would test nothing.
    """
    lem = LEM4(photon=V79, cell=V79_CELL, stopping_power=_SP)
    below, above = Ion(6, 12, 2.2055), Ion(6, 12, 2.2065)
    assert lem.pattern_is_single_loop(below)
    assert not lem.pattern_is_single_loop(above)

    d_below, d_above = lem.mean_field_deq(below), lem.mean_field_deq(above)
    assert d_below == pytest.approx(d_above, rel=2e-3), (
        f"D_eq steps across the seam: {d_below:.4f} -> {d_above:.4f}"
    )
    a_below = lem.initial_slope(below)
    assert a_below == pytest.approx(lem.initial_slope(above), rel=2e-3)


def test_rbe_alpha_is_smooth_through_the_sub_mev_band() -> None:
    """No kink where ``r_max`` crosses ``sigma``.

    The truncation defect was energy-dependent -- it discarded 95% of the track
    at 0.1 MeV/u but only 2% at 0.5 -- so it did not merely bias this band, it
    put a non-monotone notch in it. ``RBE_alpha`` here is a smooth shallow
    minimum, so consecutive ratios stay near 1.

    Scored under ``legacy_trip98_cluster_index_cap`` because that is the only
    configuration whose *mean field* returns a number across this whole band: the
    single-central-traversal pattern is degenerate here (see
    `LEM4.mean_field_pattern_is_degenerate`). The defect this test guards is
    in the radial scoring domain, which is upstream of that and shared by every
    configuration, so the legacy cap is the right lens for it.
    """
    lem = LEM4(photon=V79, cell=V79_CELL, stopping_power=_SP, legacy_trip98_cluster_index_cap=0.95)
    energies = [0.126, 0.158, 0.2, 0.251, 0.316, 0.398, 0.501]
    rbe = [lem.initial_slope(Ion(6, 12, e)) / V79.alpha_x for e in energies]

    steps = [b / a for a, b in itertools.pairwise(rbe)]
    assert max(steps) < 1.10 and min(steps) > 0.90, (
        f"RBE_alpha jumps through the sub-MeV band: {dict(zip(energies, rbe, strict=True))}"
    )


# ---- full-simulation initial slope (the TRiP98 extraction protocol) -----
def test_full_simulation_initial_slope_is_deterministic() -> None:
    """Averaging fixed seeds, so ``alpha_beta`` stays a usable table generator.

    A stochastic alpha would make every regenerated RBE table differ from the
    last one; the seeds are fixed in ``_FULLSIM_SEEDS`` for exactly that reason.
    """
    lem = LEM4(photon=V79, cell=V79_CELL, stopping_power=_SP, initial_slope_model="full_simulation")
    ion = Ion(6, 12, 2.0)
    assert lem.initial_slope(ion) == lem.initial_slope(ion)


def test_full_simulation_lifts_alpha_where_the_response_saturates() -> None:
    """The fitted slope exceeds the ``D -> 0`` slope in the overkill region.

    Under overkill each track is already lethal, so ``-ln S(D)`` is *concave*: the
    LQ fit absorbs that concavity and lifts ``alpha`` above the true initial slope.
    That is the whole mechanism by which TRiP98's tabulated ``RBE_alpha`` sits above
    our analytic one below ~1 MeV/u, and it has to switch off at low LET, where the
    response is convex and the two agree closely.

    Measured against the LEGACY-capped analytic slope. That matters: the lift is a
    *ratio*, and the 0.95 cap held the analytic denominator down. Without it the
    denominator is the raw (near-saturated) mean field, the ratio drops below 1,
    and the statement being tested here -- about the extraction protocol -- would
    be swamped by the cap's effect on the reference. See
    docs/references/LEM_IV_Friedrich2012.md.
    """
    kw = dict(photon=V79, cell=V79_CELL, stopping_power=_SP, legacy_trip98_cluster_index_cap=0.95)
    slope = LEM4(**kw)
    full = LEM4(**kw, initial_slope_model="full_simulation")

    overkill = Ion(6, 12, 1.0)  # deep overkill: LET ~ 620 keV/um
    lift = full.initial_slope(overkill) / slope.initial_slope(overkill)
    assert lift > 1.05, f"no overkill lift from the fitted slope: {lift:.3f}"

    # The curvature it exploits, asserted on the response rather than on the sign of a
    # fitted beta. That distinction is load-bearing: the sign of a fitted
    # ``beta`` depends on the window (a 0.4-3 Gy window stops inside the
    # single-track region and gives ``beta < 0``). Over the D_t-relative window the response is
    # concave-then-convex -- E/D falls from 0.704 at 1.3 Gy to 0.545 at 5.4 Gy, then
    # climbs to 0.783 at 14.4 Gy as tracks start to overlap -- so the LQ fit averages
    # two opposite curvatures and returns a small POSITIVE beta (+0.0075). Neither
    # sign is a physical curvature here; the response simply is not LQ at LET 620
    # under any window. What is real, and what actually lifts alpha, is the concavity
    # over the lower half.
    doses = full.fit_doses_Gy()
    _effect, survival, _d = full._local_dose_mc_curves(overkill, doses, n_realizations=400, seed=0)
    specific = -np.log(np.clip(survival, 1e-300, 1.0)) / doses
    lower = specific[: len(doses) // 2]
    assert lower[-1] < lower[0], f"overkill concavity gone from the lower window: {lower}"

    # ...and alpha survives the window choice even though beta does not: 0.6312 on an
    # absolute 0.4-3.0 Gy window against 0.6172 on the D_t-relative one, a 2.2% move, both well
    # above the
    # analytic 0.5468. The linear term is what the TRiP98 low-energy reproduction
    # rests on, and it is the robust half.
    alpha_old, _b = full.local_dose_mc_alpha_beta(
        overkill, doses_Gy=(0.4, 0.8, 1.2, 1.6, 2.0, 2.5, 3.0), seed=0
    )
    alpha_new, _b2 = full.local_dose_mc_alpha_beta(overkill, seed=0)
    assert alpha_new == pytest.approx(alpha_old, rel=0.10), (
        f"fitted alpha moved with the window: {alpha_old:.4f} -> {alpha_new:.4f}"
    )

    # 20 MeV/u (LET ~ 93) is as far from saturation as this route can be run
    # cheaply -- r_max ~ E^1.7 makes the simulated domain, and the cost, explode.
    entrance = Ion(6, 12, 20.0)
    near = full.initial_slope(entrance) / slope.initial_slope(entrance)
    assert 0.85 < near < 1.15, f"fitted and analytic slopes disagree off overkill: {near:.3f}"


def test_full_simulation_warns_before_a_runaway_domain() -> None:
    """It must not silently hang: cost grows as ``(R_eff + r_max)^2``, ``r_max ~ E^1.7``.

    62 s for a single 50 MeV/u carbon ion against 1.8 s at 1 MeV/u, and rising --
    at therapeutic energies ``r_max`` reaches hundreds of microns. The warning also
    points at the cheaper route, which is the *correct* one that far from overkill.

    The trigger is the domain-to-nucleus ratio, not the energy, so a small nucleus
    reproduces a therapeutic-energy blow-up at an energy cheap enough to test.
    """
    small = CellTarget(nucleus_radius_um=1.2)  # still several 540 nm subvolumes across
    lem = LEM4(photon=V79, cell=small, stopping_power=_SP, initial_slope_model="full_simulation")
    ion = Ion(6, 12, 25.0)  # r_max ~ 13 um against a 1.2 um nucleus: a 12x domain

    with pytest.warns(UserWarning, match="cross_section"):
        assert lem.initial_slope(ion) > 0.0

    # ...and it stays quiet where the domain is the nucleus' own size.
    with warnings.catch_warnings():
        warnings.simplefilter("error")
        LEM4(
            photon=V79, cell=V79_CELL, stopping_power=_SP, initial_slope_model="full_simulation"
        ).initial_slope(Ion(6, 12, 1.0))


def test_full_simulation_lift_is_cell_dependent() -> None:
    """How far the fit lifts ``alpha`` depends on the cell, not just the track.

    This is the property that separates this route from every track-level
    correction, and it is why it can address a residual that differs between two
    TRiP98 cell files sharing an identical ``rnucleus``: the fit window sits at a
    fixed *dose*, so how much saturation curvature falls inside it depends on the
    cell's own photon parameters. V79 lifts ~1.1x at 1 MeV/u; the radioresistant
    late-responding parameterisation of TRiP's spinal-cord file (a 60x smaller
    ``alpha_x``) lifts ~1.4x on the identical track.

    Legacy-capped for the same reason as
    ``test_full_simulation_lifts_alpha_where_the_response_saturates``: the lift is
    a ratio whose denominator the cap controls.
    """
    resistant = PhotonParameters(alpha_x=0.003, beta_x=0.0015, D_t=22.0)
    ion = Ion(6, 12, 1.0)

    def lift(photon: PhotonParameters, cell: CellTarget) -> float:
        kw = dict(
            photon=photon,
            cell=cell,
            stopping_power=_SP,
            legacy_trip98_cluster_index_cap=0.95,
        )
        return LEM4(**kw, initial_slope_model="full_simulation").initial_slope(ion) / LEM4(
            **kw
        ).initial_slope(ion)

    assert lift(resistant, CellTarget(5.0)) > lift(V79, V79_CELL) + 0.15


def test_full_simulation_is_rejected_by_the_local_dose_lem_versions() -> None:
    """LEM I/II/III have no full simulation to fit, and must say so, not fall back.

    The dispatch in ``LEMBase.initial_slope`` reaches its ``cross_section`` default
    by falling through, so an unhandled model would silently return the wrong
    quantity; the value is therefore refused at construction.
    """
    from pyradbio.models.lem.lem1 import LEM1

    with pytest.raises(ValueError, match="full_simulation"):
        LEM1(photon=V79, cell=V79_CELL, initial_slope_model="full_simulation")


# ---- survival convention (<e^-E> vs e^-<E>) -----------------------------
def test_survival_convention_validation() -> None:
    """Unknown convention is rejected at construction, not silently ignored."""
    with pytest.raises(ValueError, match="survival_convention"):
        LEM4(photon=V79, cell=V79_CELL, survival_convention="linear")  # type: ignore[arg-type]


def test_mean_effect_drops_the_poisson_saturation_factor() -> None:
    """``mean_effect`` is exactly ``mean_survival`` divided by ``(1-e^-N_1)/N_1``.

    Both keep the escaped-energy (indirect) term, which is already linear, so the
    two differ only in the direct term. Checking against the factor computed from
    ``N_1`` -- rather than against a hard-coded number -- is what makes this a test
    of the formula and not of one operating point.
    """
    # Legacy-capped so 1 MeV/u carbon stays in range: the claim is about the
    # survival convention, which sits downstream of whatever supplies C_I.
    kw = dict(photon=V79, cell=V79_CELL, stopping_power=_SP, legacy_trip98_cluster_index_cap=0.95)
    sat = LEM4(**kw)
    lin = LEM4(**kw, survival_convention="mean_effect")

    for energy in (1.0, 2.5, 10.0, 100.0):
        ion = Ion(6, 12, energy)
        n_1, in_nuc = sat._single_traversal_effect(ion)
        d_1 = sat.mean_traversal_dose_Gy(ion)
        r_eff = sat.cell.effective_nucleus_radius_um()
        indirect = V79.alpha_x * max(1.0 - in_nuc / (np.pi * r_eff * r_eff * d_1), 0.0)
        expected = n_1 / d_1 + indirect
        assert lin.initial_slope(ion) == pytest.approx(expected, rel=1e-12)
        # mean_effect never kills less: 1 - e^-N <= N for all N >= 0
        assert lin.initial_slope(ion) >= sat.initial_slope(ion)


def test_mean_effect_loses_the_overkill_ceiling() -> None:
    """The reason the saturated convention stays the default.

    ``mean_survival`` bounds ``alpha_I <= 1/D_1 + alpha_x``: one traversal cannot
    kill a cell more than once, however lethal it is. ``mean_effect`` has no such
    bound and will run past it wherever ``N_1`` is large. That does not bite on the
    TRiP98 tables -- ``N_1`` is capped near 0.37 there by the one-column damage-area
    pinning -- but it is a real property of the option, so it is pinned here.

    Reaching ``N_1 > 1`` takes more than a radioresistant cell: ``N_1 = kappa *
    E_gamma(D_eq)`` and ``kappa`` is one damaged column over the nuclear area, so a
    *small* nucleus is what raises it. TRiP's chordoma photon curve on a 2.5 um
    nucleus gives ``N_1 = 1.43``; the same curve on its own 5 um nucleus gives 0.36.
    """
    photon = PhotonParameters(alpha_x=0.0081, beta_x=0.0033, D_t=30.0)
    # Legacy-capped: 20Ne at 2 MeV/u is inside the degenerate band (the whole
    # profile fits in one giant loop), so the uncapped mean field has no C_I to
    # give. The claim under test is about the survival convention, which is
    # downstream of C_I and unaffected by which value supplies it.
    kw = dict(
        photon=photon,
        cell=CellTarget(2.5),
        stopping_power=_SP,
        legacy_trip98_cluster_index_cap=0.95,
    )
    sat, lin = LEM4(**kw), LEM4(**kw, survival_convention="mean_effect")
    ion = Ion(10, 20, 2.0)

    n_1 = sat._single_traversal_effect(ion)[0]
    assert n_1 > 1.0, f"fixture failed to reach saturation: N_1={n_1:.3f}"
    ceiling = 1.0 / sat.mean_traversal_dose_Gy(ion) + photon.alpha_x
    assert sat.initial_slope(ion) <= ceiling
    assert lin.initial_slope(ion) > ceiling


def test_survival_convention_also_drives_the_fitted_slope() -> None:
    """One field, both extraction routes -- they must not disagree about the average.

    ``full_simulation`` fits a simulated curve; the convention chooses whether that
    curve is ``-ln <e^-E>`` or ``<E>``. If the field only reached the analytic
    route, a user selecting ``mean_effect`` would silently keep ``<e^-E>`` here.
    """
    kw = dict(photon=V79, cell=V79_CELL, stopping_power=_SP, initial_slope_model="full_simulation")
    ion = Ion(6, 12, 1.0)
    sat = LEM4(**kw).initial_slope(ion)
    lin = LEM4(**kw, survival_convention="mean_effect").initial_slope(ion)
    assert lin != pytest.approx(sat, rel=1e-6)
    assert lin > sat  # same direction as the analytic route: no saturation to lose


# ---- cluster-index estimator for the RBE path --------------------------
def test_cluster_index_model_validation() -> None:
    """Unknown estimator, and a nonsensical realisation count, are rejected."""
    with pytest.raises(ValueError, match="cluster_index_model"):
        LEM4(photon=V79, cell=V79_CELL, cluster_index_model="sampled")  # type: ignore[arg-type]
    with pytest.raises(ValueError, match="cluster_index_mc_realisations"):
        LEM4(photon=V79, cell=V79_CELL, cluster_index_mc_realisations=0)


def test_discrete_cluster_index_convergence_is_judged_on_one_minus_ci() -> None:
    """The default must converge the UNCLUSTERED fraction, not ``C_I``.

    ``C_I`` is 0.98+ through the overkill band, so agreeing to 1% in it is a 60%
    error in ``1 - C_I`` -- and ``1 - C_I`` is what ``D_eq = C_g^-1(C_I)`` reads.
    ``n_impact = 24`` passes a 1%-in-``C_I`` check while being wrong by
    that 60%. This compares against a 4x refinement rather than the next
    step up, which can agree by coincidence.

    The convergence claim is made at 2.5 MeV/u, not at 0.126. Below ~1 MeV/u the
    estimator does not converge in ``n_impact`` at all and is not supposed to be
    believed: ``1 - C_I`` there is carried by the narrow shell of track positions
    that straddle a loop boundary, and separately the box-averaged profile smears a
    34 nm track over 84 nm, putting it 3-6x above the sampled MC. That band belongs
    to ``discrete_mc``. The second half of this test pins the non-convergence, so
    that "it converges everywhere" cannot be quietly assumed later.
    """
    lem = LEM4(photon=V79, cell=V79_CELL, stopping_power=_SP)
    ion = Ion(6, 12, 2.5)

    default = 1.0 - lem.analytic_discrete_cluster_index(ion)
    refined = 1.0 - lem.analytic_discrete_cluster_index(ion, n_impact=4 * _DISCRETE_N_IMPACT)
    assert default == pytest.approx(refined, rel=0.07), (
        f"1-C_I not converged: {default:.6f} vs {refined:.6f}"
    )
    # ...and 24 nodes are not converged, by far more than the tolerance above.
    stale = 1.0 - lem.analytic_discrete_cluster_index(ion, n_impact=24)
    assert abs(stale / refined - 1.0) > 0.10, (
        f"n_impact=24 no longer misconverges; this test has stopped testing anything "
        f"({stale:.6f} vs {refined:.6f})"
    )
    assert 1.0 - stale == pytest.approx(1.0 - refined, rel=0.05)  # ...yet close in C_I

    # The low-energy band is a different regime, and refining does not rescue it.
    sub_mev = Ion(6, 12, 0.126)  # a 34 nm track against a 540 nm lattice
    coarse = 1.0 - lem.analytic_discrete_cluster_index(sub_mev)
    fine = 1.0 - lem.analytic_discrete_cluster_index(sub_mev, n_impact=4 * _DISCRETE_N_IMPACT)
    assert abs(coarse / fine - 1.0) > 0.05, (
        f"0.126 MeV/u now converges in n_impact ({coarse:.6f} vs {fine:.6f}); if that is "
        "real, the estimator's low-energy warning and cluster_index_model docs are stale"
    )


def test_discrete_cluster_index_model_is_not_clipped_onto_the_mean_field() -> None:
    """The discrete index must keep its own bound, or the option does nothing.

    Through the overkill band the discrete ``C_I`` (~0.98) sits *above* the legacy
    TRiP98 cap (0.95), so clipping it with the mean field's cap collapses the two
    estimators onto the same 0.95 and the RBE is bit-identical. That is what made
    this route look uniformly worthless. Scored against the legacy configuration
    because that is the trap being guarded.
    """
    # A 5.0 um nucleus, as both TRiP98 cell files use -- the discrete index only
    # rises above the mean field's cap for the deeply overkilled tracks there.
    kw = dict(
        photon=PhotonParameters(alpha_x=0.003, beta_x=0.0015, D_t=22.0),
        cell=CellTarget(5.0),
        stopping_power=_SP,
        survival_convention="mean_effect",
        legacy_trip98_cluster_index_cap=_CLUSTER_INDEX_TRIP98_LEGACY,
    )
    mf = LEM4(**kw)
    disc = LEM4(**kw, cluster_index_model="discrete")
    ion = Ion(6, 12, 1.0)

    c_disc = disc.analytic_discrete_cluster_index(ion)
    assert c_disc > _CLUSTER_INDEX_TRIP98_LEGACY, "fixture no longer exercises the trap"
    assert mf.cluster_index(ion) > c_disc, "mean field should over-cluster here"
    # The two RBEs must differ: if they do not, the cap has eaten the difference.
    assert disc.initial_slope(ion) != pytest.approx(mf.initial_slope(ion), rel=1e-6)
    # ...and the discrete one must come out HIGHER, which is worth stating because the
    # sign is counter-intuitive: the discrete index is below the *raw* mean field
    # (0.978 vs 0.99990) but above its capped value (0.95), and it is the capped value
    # the RBE path actually used. The cap was holding the mean field below a less
    # clustered estimator.
    assert disc.initial_slope(ion) > mf.initial_slope(ion)


def test_discrete_mc_cluster_index_model_is_seeded() -> None:
    """``discrete_mc`` is stochastic but reproducible, and it does move the RBE."""
    kw = dict(photon=V79, cell=V79_CELL, stopping_power=_SP, cluster_index_mc_realisations=400)
    ion = Ion(6, 12, 1.0)

    a = LEM4(**kw, cluster_index_model="discrete_mc", cluster_index_mc_seed=0)
    b = LEM4(**kw, cluster_index_model="discrete_mc", cluster_index_mc_seed=0)
    c = LEM4(**kw, cluster_index_model="discrete_mc", cluster_index_mc_seed=7)
    assert a.initial_slope(ion) == pytest.approx(b.initial_slope(ion), rel=1e-12)
    assert a.initial_slope(ion) != pytest.approx(c.initial_slope(ion), rel=1e-9)
    mean_field = LEM4(**kw)
    assert a.initial_slope(ion) != pytest.approx(mean_field.initial_slope(ion), rel=1e-6), (
        "the sampled estimator is not reaching the RBE path"
    )
    # The two differ for a structural reason, not a numerical one: at 1 MeV/u the
    # mean field's C_I has saturated (1 - C_I = 9e-11), so it reaches D_eq by
    # matching the single loop's Poisson mean while the sampled estimator inverts
    # a resolvable index. Same equation, different well-posed route to it.
    assert LEM4(**kw).pattern_is_single_loop(ion)
    assert 1.0 - mean_field.cluster_index(ion) < 1.0e-9


def test_neither_discrete_estimator_is_pinned_to_the_lattice_registration() -> None:
    """``1 - C_I`` must not care where the 540 nm tiling's origin happens to fall.

    Placing every track at ``(b, 0)`` against a lattice whose phase comes from the
    parity of ``subvolume_centres`` puts a loop centre exactly on the origin at
    ``r_eff = 5.0 um`` and straddles it at 4.7 and 5.3. ``1 - C_I`` at 12C
    0.126 MeV/u would then read 0.063 / 0.015 / 0.062 across those radii -- a 6%
    change in nucleus radius moving it 4x -- and 5.0 um is what both TRiP98 cell
    files use, so every score would sit on the singular value.
    ``analytic_discrete_cluster_index`` marginalises the lattice phase along with the
    impact parameter, folding both into one R2 quasi-random sweep at the same cost.

    The bound is looser at 0.126 than at 2.5 MeV/u on purpose: down there ``1 - C_I``
    is carried by the narrow shell of positions straddling a loop boundary, so the
    quadrature holds it to ~12% at the default node count and no refinement fixes
    that (see the estimator's warning). Tightening this bound by raising ``n_impact``
    would be testing the sampler, not the registration.
    """
    for ion, bound in ((Ion(6, 12, 0.126), 1.5), (Ion(6, 12, 2.5), 1.15)):
        analytic = [
            1.0
            - LEM4(
                photon=V79, cell=CellTarget(r), stopping_power=_SP
            ).analytic_discrete_cluster_index(ion)
            for r in (4.7, 5.0, 5.3)
        ]
        assert max(analytic) / min(analytic) < bound, (
            f"deterministic estimator still moves with the nucleus radius at "
            f"{ion.energy_MeV_u} MeV/u: {analytic}"
        )

    sampled = [
        1.0
        - LEM4(photon=V79, cell=CellTarget(r), stopping_power=_SP).discrete_cluster_index(
            Ion(6, 12, 0.126), n_realisations=4000, seed=0
        )
        for r in (4.7, 5.0, 5.3)
    ]
    assert max(sampled) / min(sampled) < 1.3, f"sampled estimator is lattice-pinned too: {sampled}"


def _trip_geometry(**kw: object) -> LEM4:
    """A LEM4 on the TRiP98 cell geometry (``R_n`` = 5.0 um, both cell files)."""
    photon = PhotonParameters(alpha_x=0.003, beta_x=0.0015, D_t=22.0)
    return LEM4(photon=photon, cell=CellTarget(5.0), stopping_power=_SP, **kw)  # type: ignore[arg-type]


def test_trip98_composite_switches_on_the_single_loop_criterion() -> None:
    """``"trip98_composite"`` uses discrete inside the single-loop regime, mean field above.

    The threshold is `LEM4.pattern_is_single_loop` (2.206 MeV/u for every ion,
    since ``r_max = 0.062 E^1.7`` depends on energy per nucleon alone) -- no free
    parameter, and also the accuracy optimum on the TRiP98 tables (the RMS error
    rises for every larger E_c).

    A seam at 10 MeV/u, picked because the two estimators agree there to 1.6%, is
    not used. That agreement is an artefact of the giant-loop registration defect
    and holds only at ``R_n = 5.0 um``; worse, a 10 MeV/u seam routes ``discrete`` through the 1-10
    MeV/u band where it is several
    times worse than the mean field.

    **The seam is discontinuous, and no threshold fixes that.** Both estimators are
    wrong there in opposite directions -- just below the seam the mean field
    overshoots the table and the sampled discrete one undershoots it -- so the step
    is the model's own inconsistency surfacing. This asserts the step is bounded
    and one-signed rather than a smoothness the model does not have.
    """
    comp = _trip_geometry(cluster_index_model="trip98_composite")
    discrete = _trip_geometry(cluster_index_model="discrete")
    mean_field = _trip_geometry(cluster_index_model="mean_field")

    eps = 1.0e-3
    floor = 2.206  # where r_max + 8 sigma first exceeds half a 540 nm loop
    for Z, A in ((1, 1), (2, 4), (6, 12)):
        below = comp.initial_slope(Ion(Z, A, floor - eps))
        above = comp.initial_slope(Ion(Z, A, floor + eps))
        assert 0.15 < below / above < 1.0, (
            f"composite steps by {below / above - 1:+.1%} at the seam for Z={Z}; it "
            "hands over from the lower estimator to the mean field"
        )

    # It really is switching, and on the geometric criterion rather than on an energy
    # constant that could drift out of step with it.
    lo, hi = Ion(6, 12, 1.0), Ion(6, 12, 100.0)
    assert comp.pattern_is_single_loop(lo) and not comp.pattern_is_single_loop(hi)
    assert comp.initial_slope(lo) == pytest.approx(discrete.initial_slope(lo), rel=1e-12)
    assert comp.initial_slope(hi) == pytest.approx(mean_field.initial_slope(hi), rel=1e-12)
    assert abs(comp.initial_slope(lo) / mean_field.initial_slope(lo) - 1.0) > 0.5


def test_the_removed_auto_model_is_rejected_with_a_pointer() -> None:
    """``"auto"`` was removed, not renamed silently: it must not fall through."""
    with pytest.raises(ValueError, match="trip98_composite"):
        _trip_geometry(cluster_index_model="auto")


def test_the_discrete_rbe_path_is_free_of_the_lattice_registration_defect() -> None:
    """The registration fix has to survive to ``RBE_alpha``, not just to ``C_I``.

    With the defect, ``discrete / mean_field`` for 12C reads -19.3% / -0.2% / -20.5% at
    10 MeV/u and -28.6% / +0.1% / -28.7% at 40 MeV/u across ``R_n`` = 4.7 / 5.0 /
    5.3 um: a 6% change in nucleus radius flipping the estimator ranking, with the
    agreement landing exactly on the radius both TRiP98 cell files use. Stopping
    power was irrelevant to it (identical to 0.1%) -- purely the radius, through the
    lattice phase.

    ``discrete_mc`` marginalises the track azimuth per realisation and is
    flat across this scan; it is the independent control.
    """
    photon = PhotonParameters(alpha_x=0.184, beta_x=0.02, D_t=17.0)

    def ratio(radius: float, energy: float, model: str, **kw: object) -> float:
        common = dict(photon=photon, cell=CellTarget(radius), stopping_power=_SP)
        base = LEM4(**common, cluster_index_model="mean_field")  # type: ignore[arg-type]
        other = LEM4(**common, cluster_index_model=model, **kw)  # type: ignore[arg-type]
        ion = Ion(6, 12, energy)
        return other.initial_slope(ion) / base.initial_slope(ion)

    for energy in (10.0, 40.0):
        scan = [ratio(r, energy, "discrete") for r in (4.7, 5.0, 5.3)]
        assert max(scan) - min(scan) < 0.02, (
            f"discrete still moves with the nucleus radius at {energy} MeV/u: {scan}"
        )

    sampled = [
        ratio(r, 40.0, "discrete_mc", cluster_index_mc_realisations=2000) for r in (4.7, 5.0, 5.3)
    ]
    assert max(sampled) - min(sampled) < 0.06, (
        f"discrete_mc is supposed to be flat across the radius scan, got {sampled}"
    )
