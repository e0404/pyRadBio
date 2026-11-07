"""LEM IV intertrack-cDSB Monte-Carlo ``beta_I`` (Friedrich 2012, Fig. 4).

The full-simulation reference engine (``beta_model="intertrack_mc"``) places the
discrete DSBs of many tracks into a shared 540 nm giant-loop grid, so DSBs from
*different* tracks cluster in the same loop (intertrack cDSB). This raises
``beta_I`` above the Kraemer-Scholz slope-matching approximation in the
intermediate-LET band and shifts the ``beta_I(LET)`` drop to higher LET
(Friedrich, Scholz, Elsasser, Durante & Scholz 2012, Int. J. Radiat. Biol.
88(1-2):103-107, doi:10.3109/09553002.2011.611213). ``alpha_I`` is unchanged.

The per-loop lethal effect is the bounded GLOBLE model (Friedrich, Durante &
Scholz 2012, Radiat. Res. 178:385, doi:10.1667/RR2964.1); the photon-limit and
GLOBLE discreteness tests anchor the calibration before the LET trend is trusted.
"""

from __future__ import annotations

import math

import numpy as np
import numpy.typing as npt
import pytest
from scipy.optimize import brentq

from pyradbio.benchmark import friedrich2012 as f12
from pyradbio.core.cell import CellTarget
from pyradbio.core.photon import PhotonParameters
from pyradbio.core.radiation import Ion
from pyradbio.models.lem._intertrack_kernel import (
    HAVE_NUMBA,
    count_loops,
    fit_alpha_beta,
    ion_survival_batches_numpy,
    ion_survival_numpy,
)
from pyradbio.models.lem.lem4 import (
    DSB_PER_GY_CELL,
    LEM4,
    MC_GRID_N_LOOPS,
    SUBVOLUME_SIDE_NM,
    IntertrackSurvival,
    _p_cdsb,
)
from pyradbio.track.stopping_power import SimpleBetheStoppingPower

# Every model and every LET inversion here use the same analytic Bethe, so the
# energies and the LETs the model sees agree (the library default differs).
_SP = SimpleBetheStoppingPower()

# Fig. 4 "hypothetical" photon parameters and geometry (Friedrich 2012, p.105).
FIG4 = PhotonParameters(alpha_x=0.25, beta_x=0.05, D_t=9.5)
FIG4_CELL = CellTarget(nucleus_radius_um=5.0)


def _carbon_energy_for_let(let_keV_um: float) -> float:
    """Carbon-12 energy [MeV/u] whose water LET equals ``let_keV_um``."""
    return float(brentq(lambda e: _SP.let_keV_um(Ion(6, 12, e)) - let_keV_um, 1.6, 600.0))


# ---- construction / wiring ---------------------------------------------
def test_lem4_beta_model_validation() -> None:
    """``beta_model`` must be a known value; the default is slope matching."""
    with pytest.raises(ValueError, match="beta_model"):
        LEM4(photon=FIG4, cell=FIG4_CELL, stopping_power=_SP, beta_model="mc")  # type: ignore[arg-type]
    assert LEM4(photon=FIG4, cell=FIG4_CELL, stopping_power=_SP).beta_model == "slope_matching"


def test_lem4_beta_model_routes_to_intertrack_mc(monkeypatch: pytest.MonkeyPatch) -> None:
    """``beta_model="intertrack_mc"`` dispatches ``_beta_for_ion`` to the MC beta,
    and choosing the beta model does not change ``alpha_I`` (Friedrich 2012)."""
    ion = Ion(6, 12, _carbon_energy_for_let(120.0))
    # Stub the (slow) MC so the wiring check is instant and deterministic.
    monkeypatch.setattr(LEM4, "intertrack_mc_beta", lambda self, i: 0.1234)
    lem_mc = LEM4(photon=FIG4, cell=FIG4_CELL, stopping_power=_SP, beta_model="intertrack_mc")
    lem_sm = LEM4(photon=FIG4, cell=FIG4_CELL, stopping_power=_SP, beta_model="slope_matching")
    lq = lem_mc.alpha_beta(ion)
    assert lq.beta == pytest.approx(0.1234)
    # alpha is the single-particle slope, independent of the beta model.
    assert lq.alpha == pytest.approx(lem_sm.alpha_beta(ion).alpha)


def test_lem4_intertrack_engine_validation() -> None:
    """The intertrack MC ``engine`` argument rejects unknown backends."""
    lem = LEM4(photon=FIG4, cell=FIG4_CELL, stopping_power=_SP)
    ion = Ion(6, 12, _carbon_energy_for_let(60.0))
    with pytest.raises(ValueError, match="engine"):
        lem.intertrack_mc_survival(ion, [1.0, 2.0], engine="fast")  # type: ignore[arg-type]


def test_lem4_mc_effect_model_validation() -> None:
    """``mc_effect_model`` must be 'globle' (default) or 'photon_equivalent'."""
    with pytest.raises(ValueError, match="mc_effect_model"):
        LEM4(photon=FIG4, cell=FIG4_CELL, stopping_power=_SP, mc_effect_model="deq")  # type: ignore[arg-type]
    assert LEM4(photon=FIG4, cell=FIG4_CELL, stopping_power=_SP).mc_effect_model == "globle"


@pytest.mark.slow
def test_lem4_mc_photon_equivalent_recovers_low_let_beta() -> None:
    """The opt-in photon-equivalent MC scoring recovers ~beta_x at low LET.

    ``mc_effect_model="photon_equivalent"`` scores the discrete pattern through the
    cluster-index -> ``D_eq`` -> ``S_gamma(D_eq)`` mapping. At low LET (near the photon
    limit) it must still return ~``beta_x``. (It does *not* reproduce the Fig. 4 high-LET
    collapse -- it sits well above the paper at every LET measured; that is a documented,
    ruled-out hypothesis, not tested here. Whether it additionally *rises* through
    mid-LET is unsettled: those numbers came off the LET-dependent adaptive window,
    which manufactured exactly such a rise in the GLOBLE path, and the PE path has not
    been measured over a common window.)
    """
    lem = LEM4(
        photon=FIG4,
        cell=FIG4_CELL,
        stopping_power=_SP,
        beta_model="intertrack_mc",
        mc_effect_model="photon_equivalent",
    )
    ion = Ion(6, 12, _carbon_energy_for_let(30.0))
    doses = lem._adaptive_doses(ion, 9)
    betas = [
        fit_alpha_beta(doses, lem.intertrack_mc_survival(ion, doses, n_realizations=800, seed=s))[1]
        for s in range(3)
    ]
    beta = float(np.mean(betas))
    assert beta == pytest.approx(FIG4.beta_x, rel=0.15), f"low-LET PE beta={beta:.4f}"


def test_count_loops_isolated_and_clustered() -> None:
    """``count_loops`` splits cubes into iDSB (exactly one) and cDSB (>=2)."""
    # cubes: (0,0,0)x3 -> cDSB; (1,0,0)x1 -> iDSB; (0,1,0)x2 -> cDSB; (5,5,5)x1 -> iDSB
    ix = np.array([0, 0, 0, 1, 0, 0, 5], dtype=np.int64)
    iy = np.array([0, 0, 0, 0, 1, 1, 5], dtype=np.int64)
    iz = np.array([0, 0, 0, 0, 0, 0, 5], dtype=np.int64)
    n_i, n_c = count_loops(ix, iy, iz)
    assert (n_i, n_c) == (2, 2)
    assert count_loops(np.array([], np.int64), np.array([], np.int64), np.array([], np.int64)) == (
        0,
        0,
    )


# ---- photon-limit calibration anchor (Friedrich 2012 Radiat. Res. 178) --
def test_lem4_intertrack_photon_limit_recovers_lq() -> None:
    """Photon-limit convergence: the uniform-field MC recovers ``(alpha_x, beta_x)``.

    Reference: Friedrich, Durante & Scholz 2012 (doi:10.1667/RR2964.1), the GLOBLE
    effect. Scattering the discrete DSBs of a uniform field over the giant loops and
    fitting ``-ln S(D)/D = alpha + beta D`` must return the input photon LQ (the
    calibration anchor for the intertrack engine). ``beta`` sits a little below
    ``beta_x`` because the GLOBLE effect curves sub-LQ with dose (documented).
    """
    lem = LEM4(photon=FIG4, cell=FIG4_CELL, stopping_power=_SP, effect_model="globle")
    doses = np.linspace(0.5, 4.0, 10)
    survival = lem.intertrack_mc_photon_survival(doses, n_realizations=400, seed=0)
    alpha, beta = fit_alpha_beta(doses, survival)
    assert alpha == pytest.approx(FIG4.alpha_x, rel=0.06)
    assert beta == pytest.approx(FIG4.beta_x, rel=0.2)


def test_lem4_intertrack_globle_discreteness_matches_analytic() -> None:
    """Discrete uniform MC ``N_lethal`` matches the analytic GLOBLE eqs. to <=3%.

    Reference: Friedrich 2012 (doi:10.1667/RR2964.1) Table 1 discreteness check
    (their MC/analytical ratio ~1.00-1.03 at 10-50 Gy). Distributing the integer DSB
    count over the exact ``globle_n_loops`` loops reproduces the analytic
    ``E = n_i eps_i + n_c eps_c`` (eqs. 2-3, Poisson loop counts with the *paper's*
    ``Y_loop = a_DSB/N_L``, no ``eta`` -- GLOBLE's own photon model predates the
    Friedrich 2015 DSB-yield enhancement) to Monte-Carlo noise. Uses
    :attr:`LEM4.globle_n_loops` (not the spatial 540 nm subvolume grid) to match what
    :meth:`LEM4.intertrack_mc_photon_survival` actually places DSBs into.
    """
    photon = PhotonParameters(alpha_x=0.184, beta_x=0.02, D_t=17.0)
    lem = LEM4(photon=photon, cell=FIG4_CELL, stopping_power=_SP, effect_model="globle")
    eps_i, eps_c = lem._globle_lethalities()
    n_loops = round(lem.globle_n_loops)
    y_loop = DSB_PER_GY_CELL / n_loops
    doses = np.array([10.0, 20.0, 30.0, 50.0])
    e_mc = -np.log(lem.intertrack_mc_photon_survival(doses, n_realizations=200, seed=1))
    for d, emc in zip(doses, e_mc, strict=True):
        m = y_loop * d
        n_i = n_loops * m * math.exp(-m)
        n_c = n_loops * float(_p_cdsb(np.array([m]))[0])
        e_an = n_i * eps_i + n_c * eps_c
        assert emc == pytest.approx(e_an, rel=0.03), (
            f"discreteness ratio {emc / e_an:.3f} at {d} Gy"
        )


def test_intertrack_mc_eps_c_calibrated_to_grid_recovers_photon_beta() -> None:
    """The MC's ``eps_c`` uses the geometric grid count, so its photon-limit beta = beta_x.

    The GLOBLE photon (uniform) limit recovers ``beta = beta_x * N_L(eps_c) / N_grid``:
    distributing the DSBs over more loops than ``eps_c`` was calibrated for dilutes the
    clustering and lowers beta by that ratio. The intertrack MC bins into the physical
    540 nm grid (``N_grid = MC_GRID_N_LOOPS = V/l^3 ~ 3175``), so calibrating ``eps_c``
    to that same count recovers ``beta_x`` exactly. Feeding the single-particle
    ``globle_n_loops`` (2000) instead diluted it by ``2000/3175`` -- the ~40% low-LET
    Fig. 4 beta undershoot (Friedrich 2012). This pins the calibration math directly
    (fast, deterministic), independent of the slow MC.
    """
    lem = LEM4(photon=FIG4, cell=FIG4_CELL, stopping_power=_SP)
    a_dsb = DSB_PER_GY_CELL

    # Analytic low-dose photon-limit beta over N_grid loops: (a_dsb^2/N)(eps_c/2 - eps_i).
    eps_i, eps_c = lem._globle_lethalities(n_loops=MC_GRID_N_LOOPS)
    beta_grid = (a_dsb * a_dsb / MC_GRID_N_LOOPS) * (eps_c / 2.0 - eps_i)
    assert beta_grid == pytest.approx(FIG4.beta_x, rel=1e-9)

    # A mismatched calibration (globle_n_loops with the 3175-loop grid) dilutes
    # beta by exactly globle_n_loops / N_grid -- the documented undershoot.
    eps_i0, eps_c0 = lem._globle_lethalities()  # default -> globle_n_loops
    beta_mismatch = (a_dsb * a_dsb / MC_GRID_N_LOOPS) * (eps_c0 / 2.0 - eps_i0)
    assert beta_mismatch == pytest.approx(
        FIG4.beta_x * lem.globle_n_loops / MC_GRID_N_LOOPS, rel=1e-9
    )
    assert beta_mismatch < 0.7 * FIG4.beta_x  # ~0.63 beta_x, the observed ~40% low bias


@pytest.mark.slow
def test_lem4_intertrack_photon_limit_ion_path_recovers_beta_x() -> None:
    """The ion path's own photon limit (low LET / high energy) recovers ~beta_x.

    At low LET the delta-ray range ``r_max >> R_eff``, so the extended-domain tracks
    tile the nucleus near-uniformly and the intertrack MC must return ~``beta_x``. With
    ``eps_c`` calibrated to the grid count (:data:`MC_GRID_N_LOOPS`) it does, within the
    disk-rim (circle-vs-square) dilution ~10%; the previous ``globle_n_loops``
    calibration undershot to ~0.6 ``beta_x`` (Friedrich 2012 Fig. 4 low-LET plateau).
    """
    if not HAVE_NUMBA:
        pytest.skip("numba (fast extra) not installed")
    lem = LEM4(
        photon=FIG4,
        cell=FIG4_CELL,
        stopping_power=_SP,
        beta_model="intertrack_mc",
        effect_model="globle",
    )
    ion = Ion(6, 12, _carbon_energy_for_let(30.0))
    doses = lem._adaptive_doses(ion, 9)
    betas = [
        fit_alpha_beta(
            doses,
            lem.intertrack_mc_survival(ion, doses, n_realizations=2000, seed=s, engine="numba"),
        )[1]
        for s in range(4)
    ]
    beta = float(np.mean(betas))
    assert beta == pytest.approx(FIG4.beta_x, rel=0.15), f"ion-path photon-limit beta={beta:.4f}"


def test_lem4_intertrack_beta_estimator_stable_across_let() -> None:
    """The adaptive-dose estimator returns finite, non-negative beta across LET.

    The fixed 1-9 Gy spike broke at high LET (``alpha*D >> 1`` -> negative alpha);
    the adaptive per-LET dose grid keeps the effect in the LQ region so no LET gives
    garbage. Kept light (few realisations) -- it asserts sanity, not magnitude.
    """
    lem = LEM4(photon=FIG4, cell=FIG4_CELL, stopping_power=_SP, beta_model="intertrack_mc")
    for let in (30.0, 120.0, 300.0):
        beta = lem.intertrack_mc_beta(Ion(6, 12, _carbon_energy_for_let(let)), n_realizations=120)
        assert np.isfinite(beta) and beta >= 0.0


@pytest.mark.slow
def test_lem4_intertrack_numba_matches_numpy() -> None:
    """The compiled parallel kernel agrees with the NumPy reference within MC error.

    The numba path uses per-thread RNG (not bit-reproducible), so this compares the
    seed-independent statistics: the mean effect ``-ln S(D)`` from many realisations
    must agree with the NumPy reference to Monte-Carlo noise.
    """
    if not HAVE_NUMBA:
        pytest.skip("numba (fast extra) not installed")
    lem = LEM4(photon=FIG4, cell=FIG4_CELL, stopping_power=_SP, effect_model="globle")
    ion = Ion(6, 12, _carbon_energy_for_let(55.0))
    doses = lem._adaptive_doses(ion, 9)
    e_np = -np.log(lem.intertrack_mc_survival(ion, doses, n_realizations=3000, engine="numpy"))
    e_nb = -np.log(lem.intertrack_mc_survival(ion, doses, n_realizations=3000, engine="numba"))
    assert np.allclose(e_nb, e_np, rtol=0.06, atol=0.03)


@pytest.mark.slow
def test_lem4_intertrack_pe_numba_matches_numpy() -> None:
    """The photon-equivalent numba kernel agrees with its NumPy reference (MC error).

    Same statistics comparison as the GLOBLE-scoring numba/numpy check above, for the
    ``mc_effect_model="photon_equivalent"`` path (the numba fast path added so low-LET
    points, with their huge domain-extension track counts, stay tractable).
    """
    if not HAVE_NUMBA:
        pytest.skip("numba (fast extra) not installed")
    lem = LEM4(photon=FIG4, cell=FIG4_CELL, stopping_power=_SP, mc_effect_model="photon_equivalent")
    ion = Ion(6, 12, _carbon_energy_for_let(55.0))
    doses = lem._adaptive_doses(ion, 9)
    e_np = -np.log(lem.intertrack_mc_survival(ion, doses, n_realizations=3000, engine="numpy"))
    e_nb = -np.log(lem.intertrack_mc_survival(ion, doses, n_realizations=3000, engine="numba"))
    assert np.allclose(e_nb, e_np, rtol=0.06, atol=0.03)


# ---- Fig. 4 approximation vs digitized data (fast, quantitative) --------
@pytest.mark.golden
def test_golden_lem4_reproduces_friedrich2012_fig4_approximation() -> None:
    """LEM IV's default single-particle approximation reproduces Friedrich et
    al. 2012's own digitized Fig. 4 "approx" alpha_I/beta_I(LET), point-by-
    point, for the hypothetical cell.

    Reference: Friedrich et al. 2012 (doi:10.3109/09553002.2011.611213),
    Fig. 4 (p. 107) -- digitized in ``pyradbio.benchmark.friedrich2012``. The
    alpha_I axis spans [0, 1.5] (scale factor 25.0, not [0, 0.06]); the
    digitized traces use it (see that module's docstring).

    Uses ``LEM4(photon=FIG4, cell=FIG4_CELL)`` -- the *default*
    ``effect_model="photon_equivalent"``/``beta_model="slope_matching"``
    combination, not ``effect_model="globle"`` (which the intertrack-MC path
    below uses internally regardless of ``effect_model`` -- a different,
    GLOBLE-bookkeeping convention that does not match this single-particle
    comparison as tightly).
    """
    lem = LEM4(photon=FIG4, cell=FIG4_CELL, stopping_power=_SP)

    alpha_curve = f12.get_curve("fig4_hypothetical_vs_let", series="approx", variant="alpha")
    alpha_resid = [
        lem.alpha_beta(Ion(6, 12, _carbon_energy_for_let(let))).alpha - a
        for let, a in zip(alpha_curve.x, alpha_curve.y, strict=True)
    ]
    rms_alpha = float(np.sqrt(np.mean(np.square(alpha_resid))))
    assert len(alpha_resid) >= 25, f"too few comparable alpha_I points ({len(alpha_resid)})"
    assert rms_alpha < 0.05, f"alpha_I RMS vs Fig. 4 approximation = {rms_alpha:.4f}"

    beta_curve = f12.get_curve("fig4_hypothetical_vs_let", series="approx", variant="beta")
    beta_resid = [
        lem.alpha_beta(Ion(6, 12, _carbon_energy_for_let(let))).beta - b
        for let, b in zip(beta_curve.x, beta_curve.y, strict=True)
    ]
    rms_beta = float(np.sqrt(np.mean(np.square(beta_resid))))
    assert len(beta_resid) >= 30, f"too few comparable beta_I points ({len(beta_resid)})"
    assert rms_beta < 0.01, f"beta_I RMS vs Fig. 4 approximation = {rms_beta:.4f}"


# ---- Fig. 4 trend (slow reference) -------------------------------------
@pytest.mark.slow
@pytest.mark.golden
def test_lem4_intertrack_beta_shifted_to_higher_let() -> None:
    """Fig. 4: the full-simulation ``beta_I`` drop is shifted to higher LET.

    Reference: Friedrich, Scholz, Elsasser, Durante & Scholz 2012
    (doi:10.3109/09553002.2011.611213), Fig. 4. For carbon with the hypothetical
    photon parameters (alpha_g=0.25, beta_g=0.05, D_t=9.5, R_n=5), the intertrack
    Monte-Carlo ``beta_I`` stays near ``beta_x`` at LET ~70 keV/um where the
    Kraemer-Scholz slope-matching approximation has already collapsed well below
    ``beta_x`` -- "the drop of the beta_I(LET)-curve is significantly shifted to
    higher LET" (paper text).

    The MC's *fitted* ``alpha`` is not cross-checked against the single-particle
    GLOBLE alpha: the MC calibrates ``eps_c`` to the
    geometric grid count (:data:`MC_GRID_N_LOOPS` ~ 3175) while the single-particle
    ``alpha`` uses the empirically-fitted ``globle_n_loops`` (2000), so the two use
    different ``eps_c`` and the MC's fitted linear term runs higher. This does *not*
    change the model's reported ``alpha_I`` (always the single-particle slope; the MC
    only supplies ``beta`` -- see ``test_lem4_beta_model_routes_to_intertrack_mc``).

    The anchor LET is ~70 keV/um because ``GLOBLE_N_LOOPS`` is 2000 (not 3000; see
    ``lem4.py``): a smaller ``N_L`` lowers ``eps_c``, so ``alpha_I`` is smaller at a
    given LET and the slope-matching ``beta`` collapses later -- the "shifted to
    higher LET" LET point shifts with it. 55 keV/um does not isolate the effect (slope-matching
    beta only drops to
    ~0.54 beta_x there, not "collapsed"); 70 keV/um does (~0.28 beta_x) while
    keeping ``beta_mc`` comfortably above the MC noise floor.

    **This only checks the qualitative trend and direction, not the paper's
    own magnitude.** Fig. 4 is digitized
    (``pyradbio.benchmark.friedrich2012``) and the paper's own ``beta_I`` at
    LET~70 is ~0.011 -- well below ``beta_mc`` here. Measured across 16 seeds
    at 600 realizations: ``beta_mc`` at LET 70 is
    0.0611 +- 0.0226 per seed (sem 0.0056), so the gap to both the paper
    (9 sem) and slope matching (10 sem) is real, not seed noise -- and a
    single-seed value is noise-dominated, which is why this test seed-averages.
    A tight point-by-point golden test against the digitized full-simulation
    curve is still not shipped: the *measured* discrepancy -- this engine's
    ``beta`` stays pinned at ``beta_x`` where the paper's collapses (0.0526 +-
    0.0163 at LET 100 against the paper's 0.0010, over a common fit window) --
    is a real, characterised property of the domain-extended intertrack
    scoring, documented in docs/references/LEM_IV_Friedrich2012.md, not
    something more statistics would reconcile.

    The MC's ``beta`` does not *rise* with LET; an apparent rise is the
    LET-dependent :meth:`LEM4._adaptive_doses` window, and over a
    fixed window ``beta(LET)`` is flat (see
    ``test_intertrack_beta_is_flat_at_beta_x_over_a_common_fit_window``).
    """
    lem = LEM4(photon=FIG4, cell=FIG4_CELL, stopping_power=_SP, effect_model="globle")
    ion = Ion(6, 12, _carbon_energy_for_let(70.0))

    # Adaptive dose-response, seed-averaged for a tighter beta estimate. beta is a small
    # curvature on a dominant linear term, so it needs several seeds x many realisations;
    # 4 x 1500 keeps the LET-70 estimate stable (~0.04) well above the slope-matching
    # collapse (~0.003) and the 0.03 floor below.
    doses = lem._adaptive_doses(ion, 9)
    survivals = [
        lem.intertrack_mc_survival(ion, doses, n_realizations=1500, seed=s) for s in (0, 1, 2, 3)
    ]
    alpha_mc, beta_mc = fit_alpha_beta(doses, np.mean(survivals, axis=0))

    alpha_sp = lem._globle_single_particle_slope(ion)
    beta_slope_matching = lem._beta(alpha_sp)

    # The MC's fitted alpha is finite/positive but uses the decoupled grid eps_c, so it
    # runs above the single-particle slope (see docstring) -- not cross-checked here.
    assert np.isfinite(alpha_mc) and alpha_mc > 0.0
    # The slope-matching beta has dropped well below beta_x by this LET...
    assert beta_slope_matching < 0.5 * FIG4.beta_x
    # ...while the intertrack MC beta stays high -- the drop shifted to higher LET.
    assert beta_mc > 2.0 * beta_slope_matching
    assert beta_mc > 0.03


# ---- the beta estimator must not clamp away a measurement -----------------
def test_fit_alpha_beta_keeps_a_negative_curvature_measurement() -> None:
    """A negative fitted ``beta`` is a measurement and must survive the fit.

    Deep overkill genuinely has no quadratic curvature left, so the least-squares
    ``beta`` scatters through zero; silently flooring it at 0 biases the mean
    *upward* and -- far worse for an error bar -- makes the spread look smaller
    than it is. The LEM I Monte-Carlo states this convention explicitly
    (``scripts/lem1_mc_reference.py alpha-beta``: "neither is clamped ...
    kept, so the error bars stay honest"); this pins the LEM IV kernel to it.
    Clamping is the *caller's* job, and it warns -- see
    :meth:`LEM4.intertrack_mc_beta`.
    """
    doses = np.linspace(1.0, 5.0, 9)
    alpha_in, beta_in = 0.5, -0.02
    survival = np.exp(-(alpha_in * doses + beta_in * doses * doses))

    alpha, beta = fit_alpha_beta(doses, survival)

    assert alpha == pytest.approx(alpha_in, rel=1.0e-9)
    assert beta == pytest.approx(beta_in, rel=1.0e-9)


def test_intertrack_mc_beta_warns_when_it_clamps_a_negative_beta(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """The clamp still protects ``LQParameters``, but it may not be silent.

    ``_beta_for_ion`` feeds :class:`~pyradbio.core.lq.LQParameters`, which rejects
    a negative ``beta``, so the public scalar path has to floor it. A floor is not
    a computed value, so it warns -- the same non-silent pattern as
    :meth:`LEM4.local_dose_beta`.
    """
    lem = LEM4(photon=FIG4, cell=FIG4_CELL, stopping_power=_SP, beta_model="intertrack_mc")
    ion = Ion(6, 12, _carbon_energy_for_let(300.0))

    def negative_curvature(
        _self: LEM4, _ion: Ion, doses: npt.NDArray[np.float64], **_kwargs: object
    ) -> npt.NDArray[np.float64]:
        d = np.asarray(doses, dtype=np.float64)
        return np.asarray(np.exp(-(0.5 * d - 0.02 * d * d)), dtype=np.float64)

    monkeypatch.setattr(LEM4, "intertrack_mc_survival", negative_curvature)

    with pytest.warns(UserWarning, match="clamped to 0"):
        beta = lem.intertrack_mc_beta(ion)

    assert beta == 0.0


# ---- batching for the error estimator (workstream A) ----------------------
def _kernel_args(lem: LEM4, ion: Ion) -> dict[str, object]:
    """The per-ion inputs both kernel entry points take."""
    m_total, cdf, cdf_r, d_1, r_eff, r_sim, height = lem._intertrack_setup(ion)
    eps_i, eps_c = lem._globle_lethalities(n_loops=MC_GRID_N_LOOPS)
    return {
        "m_total": m_total,
        "cdf": cdf,
        "cdf_r": cdf_r,
        "eps_i": eps_i,
        "eps_c": eps_c,
        "d_1": d_1,
        "r_eff": r_eff,
        "r_sim": r_sim,
        "height": height,
        "ell": SUBVOLUME_SIDE_NM * 1.0e-3,
    }


def test_intertrack_batches_pool_to_the_single_call_result() -> None:
    """Batching must not perturb the estimator, only partition it.

    The batched kernel consumes the RNG stream in exactly the same order as the
    unbatched one, so pooling ``n_batches x realizations_per_batch`` batch-mean
    *effects* has to reproduce the single ``n_real`` call -- up to floating-point
    summation order, which is the only thing that differs. This is the guard that
    lets the error estimator be added without silently changing any published
    central value.
    """
    lem = LEM4(photon=FIG4, cell=FIG4_CELL, stopping_power=_SP, beta_model="intertrack_mc")
    ion = Ion(6, 12, _carbon_energy_for_let(70.0))
    doses = lem._adaptive_doses(ion, 5)
    args = _kernel_args(lem, ion)

    single = ion_survival_numpy(doses, n_real=12, rng=np.random.default_rng(3), **args)
    batches = ion_survival_batches_numpy(
        doses, n_batches=4, realizations_per_batch=3, rng=np.random.default_rng(3), **args
    )

    assert batches.shape == (4, doses.size)
    pooled = np.exp(-batches.mean(axis=0))
    assert pooled == pytest.approx(single, rel=1.0e-12)


def test_intertrack_batches_are_mean_effects_not_survivals() -> None:
    """The batch quantity has to be the *effect*, because pooling is linear in it.

    Survival is ``exp(-<E>)`` over realisations, so the poolable per-batch
    quantity is the mean effect; averaging per-batch *survivals* would apply
    Jensen at the batch boundary and give a different (biased) curve. Pinning the
    convention, since a jackknife over the wrong one would silently estimate the
    error of a quantity nobody reports.
    """
    lem = LEM4(photon=FIG4, cell=FIG4_CELL, stopping_power=_SP, beta_model="intertrack_mc")
    ion = Ion(6, 12, _carbon_energy_for_let(70.0))
    doses = lem._adaptive_doses(ion, 4)
    args = _kernel_args(lem, ion)

    batches = ion_survival_batches_numpy(
        doses, n_batches=3, realizations_per_batch=4, rng=np.random.default_rng(11), **args
    )

    assert np.all(batches >= 0.0), "effects are non-negative; survivals would be <= 1"
    assert np.all(np.diff(batches, axis=1) >= 0.0), "effect must be monotone in dose"


def test_intertrack_mc_beta_jackknife_reports_a_finite_positive_error() -> None:
    """The jackknife path returns ``(beta, sem)`` with a usable error bar.

    The whole point of workstream A: a single ``intertrack_mc_beta`` call is one
    draw whose spread is 7% of the value at LET 25 and larger than the value by
    LET 130, and nothing in that signature says so.
    """
    lem = LEM4(photon=FIG4, cell=FIG4_CELL, stopping_power=_SP, beta_model="intertrack_mc")
    ion = Ion(6, 12, _carbon_energy_for_let(70.0))

    beta, error = lem.intertrack_mc_beta_jackknife(ion, n_batches=6, realizations_per_batch=25)

    assert math.isfinite(beta)
    assert math.isfinite(error) and error > 0.0


def test_intertrack_mc_beta_jackknife_pools_to_the_scalar_path() -> None:
    """The jackknife's central value *is* the scalar path's value, same seed.

    An error estimator that quietly changed the number it decorates would be
    worse than none. The two differ only where the scalar path applies its
    documented deep-overkill floor, which is why this checks at LET 70 where the
    floor never binds.
    """
    lem = LEM4(photon=FIG4, cell=FIG4_CELL, stopping_power=_SP, beta_model="intertrack_mc")
    ion = Ion(6, 12, _carbon_energy_for_let(70.0))

    scalar = lem.intertrack_mc_beta(ion, n_realizations=150, seed=5)
    pooled, _error = lem.intertrack_mc_beta_jackknife(
        ion, n_batches=6, realizations_per_batch=25, seed=5
    )

    assert pooled == pytest.approx(scalar, rel=1.0e-9)


def test_intertrack_survival_jackknife_does_not_clamp() -> None:
    """A deep-overkill jackknife must report a negative ``beta`` unclamped.

    The container consumes the raw fit precisely so the error bar stays honest;
    clamping here would reintroduce the bias documented in ``fit_alpha_beta``.
    Built from a hand-made effect curve with genuine negative curvature, so the
    assertion does not depend on catching a rare Monte-Carlo draw.
    """
    doses = np.linspace(1.0, 5.0, 9)
    base = 0.5 * doses - 0.02 * doses * doses  # alpha 0.5, beta -0.02
    jitter = np.random.default_rng(0).normal(scale=1.0e-3, size=(6, doses.size))
    curve = IntertrackSurvival(
        batch_effect=base + jitter, doses_Gy=doses, realizations_per_batch=10
    )

    beta, error = curve.jackknife_beta()

    assert beta == pytest.approx(-0.02, abs=2.0e-3)
    assert math.isfinite(error) and error > 0.0


def test_intertrack_survival_pooling_is_additive() -> None:
    """``pooled_with`` concatenates batches, so precision budgeting can *add* work.

    Restarting with a bigger realization count throws away the pilot; the
    allocator in workstream B needs to extend an existing run instead.
    """
    lem = LEM4(photon=FIG4, cell=FIG4_CELL, stopping_power=_SP, beta_model="intertrack_mc")
    ion = Ion(6, 12, _carbon_energy_for_let(70.0))

    first = lem.intertrack_mc_batches(ion, n_batches=3, realizations_per_batch=10, seed=0)
    second = lem.intertrack_mc_batches(ion, n_batches=2, realizations_per_batch=10, seed=1)
    pooled = first.pooled_with(second)

    assert pooled.n_batches == 5
    assert pooled.realizations_per_batch == 10
    assert pooled.batch_effect.shape == (5, first.batch_effect.shape[1])


@pytest.mark.slow
def test_intertrack_jackknife_error_matches_the_brute_force_seed_scatter() -> None:
    """The jackknife must estimate the *right quantity*, checked against ground truth.

    An error estimator is only as good as its own validation, and the cheap way
    to get one wrong is a missing ``sqrt(n)``: the delete-one jackknife over
    batches has to return the standard deviation of a single ``n_realizations``
    estimate, which is directly measurable by re-running independent seeds. Both
    sides are themselves noisy (12 batches, 8 seeds), so this bounds the ratio
    rather than pinning it -- but a factor-2 window still catches a per-batch /
    per-realization mix-up, which would be off by ``sqrt(50) ~ 7``.

    Measured across the Friedrich 2012 LET band (600 realisations):
    jackknife 0.0034 / 0.0047 / 0.0103 / 0.0205 / 0.1132 against seed scatter
    0.0030 / 0.0044 / 0.0081 / 0.0226 / 0.0705 at LET 25 / 40 / 55 / 70 / 100.
    LET 70 is used here because it is cheap and its spread is large enough to
    measure without an enormous seed count.
    """
    lem = LEM4(photon=FIG4, cell=FIG4_CELL, stopping_power=_SP, beta_model="intertrack_mc")
    ion = Ion(6, 12, _carbon_energy_for_let(70.0))
    doses = lem._adaptive_doses(ion, 9)

    per_seed = [
        fit_alpha_beta(doses, lem.intertrack_mc_survival(ion, doses, n_realizations=600, seed=s))[1]
        for s in range(8)
    ]
    brute_force = float(np.std(per_seed, ddof=1))

    _beta, jackknifed = lem.intertrack_mc_beta_jackknife(
        ion, doses_Gy=doses, n_batches=12, realizations_per_batch=50, seed=101
    )

    assert 0.5 < jackknifed / brute_force < 2.0, (
        f"jackknife {jackknifed:.4f} vs seed scatter {brute_force:.4f}"
    )


@pytest.mark.slow
def test_intertrack_beta_is_flat_at_beta_x_over_a_common_fit_window() -> None:
    """Over ONE fixed dose window the MC's ``beta`` is pinned at ``beta_x``, at every LET.

    This is the engine's documented mechanism -- the domain-extension near-miss
    background clusters like a photon field -- measured as such.
    Per-LET numbers fitted on the adaptive window would *rise* with LET,
    contradicting that very mechanism, because :meth:`LEM4._adaptive_doses` shrinks
    ``D_max`` as LET rises (5.6 Gy at LET 25 down to 1.3 Gy at 100) and a narrower,
    lower window reads a higher ``beta`` (a fit-window projection). Refit over a
    common window the trend is gone: 0.0492 / 0.0340 / 0.0537 / 0.0441 / 0.0526 (+- 0.0034 / 0.0061
    / 0.0104
    / 0.0117 / 0.0163) at LET 25 / 40 / 55 / 70 / 100, weighted mean
    0.0463 +- 0.0027 with chi2 5.5 on 4 dof against a constant, and LET 25 -> 100
    flat to 0.2 sigma. The 7% shortfall against ``beta_x`` is the documented
    disk-rim (circle-vs-square) dilution, same as the photon-limit test above.

    Pinned here at LET 55 and 100 only, which cost a few seconds each: the fixed
    window is *cheap* at high LET (``D_max`` sets the track count), which is why
    12000 realisations are affordable where the adaptive window at LET 25 needs
    four minutes for the same precision.
    """
    lem = LEM4(photon=FIG4, cell=FIG4_CELL, stopping_power=_SP, beta_model="intertrack_mc")
    high = Ion(6, 12, _carbon_energy_for_let(100.0))
    # The narrowest adaptive window in the band -- valid (inside the LQ region)
    # at every LET, where a wider one would break the high-LET fit.
    window = lem._adaptive_doses(high, 9)

    beta_55, sem_55 = lem.intertrack_mc_beta_jackknife(
        Ion(6, 12, _carbon_energy_for_let(55.0)),
        doses_Gy=window,
        n_batches=20,
        realizations_per_batch=600,
    )
    beta_100, sem_100 = lem.intertrack_mc_beta_jackknife(
        high, doses_Gy=window, n_batches=20, realizations_per_batch=600
    )

    # Flat: no LET dependence once the window is held fixed.
    spread = math.hypot(sem_55, sem_100)
    assert abs(beta_100 - beta_55) < 3.0 * spread, (
        f"beta moved with LET at a fixed window: {beta_55:.4f} -> {beta_100:.4f}"
    )
    # ... and flat *at beta_x*, not at some other constant.
    for beta, sem, let in ((beta_55, sem_55, 55), (beta_100, sem_100, 100)):
        assert abs(beta - FIG4.beta_x) < 3.0 * sem + 0.1 * FIG4.beta_x, (
            f"LET {let} beta={beta:.4f} is not beta_x within rim dilution"
        )
    # The Fig. 4 collapse is still absent -- the paper's own full simulation has
    # collapsed to ~0.001 by LET 100 while this engine sits at beta_x.
    paper = f12.get_curve("fig4_hypothetical_vs_let", series="full_simulation", variant="beta")
    paper_at_100 = float(np.interp(100.0, paper.x, paper.y))
    assert beta_100 - paper_at_100 > 2.5 * sem_100


# ---- precision budgeting (workstream B) -----------------------------------
def test_intertrack_beta_to_precision_buys_realizations_until_it_hits_the_target() -> None:
    """The allocator adds realisations until the jackknife error meets the target.

    Sizing a Monte-Carlo by a flat realisation count is wrong at both ends here,
    because precision runs *opposite* to cost: at 600 realisations the per-point
    sigma spans 0.0030 (LET 25, ~70 s) to 0.0705 (LET 100, ~1 s). Budgeting by
    target precision spends where spending is cheap.
    """
    lem = LEM4(photon=FIG4, cell=FIG4_CELL, stopping_power=_SP, beta_model="intertrack_mc")
    ion = Ion(6, 12, _carbon_energy_for_let(100.0))
    pilot = 12 * 50

    beta, sem, n_real = lem.intertrack_mc_beta_to_precision(
        ion, target_sem=0.012, n_batches=12, realizations_per_batch=50, max_realizations=40_000
    )

    assert math.isfinite(beta)
    assert sem <= 0.012
    assert n_real > pilot, "should have bought more than the pilot"


def test_intertrack_beta_to_precision_warns_rather_than_silently_capping() -> None:
    """An unreachable target must stop at the cap and *say so*.

    Silent truncation reads as "target met" when it was not. The repo rule is
    that a bounded budget reports what it gave up.
    """
    lem = LEM4(photon=FIG4, cell=FIG4_CELL, stopping_power=_SP, beta_model="intertrack_mc")
    ion = Ion(6, 12, _carbon_energy_for_let(100.0))

    with pytest.warns(UserWarning, match="target_sem"):
        _beta, sem, n_real = lem.intertrack_mc_beta_to_precision(
            ion,
            target_sem=1.0e-6,  # unreachable
            n_batches=12,
            realizations_per_batch=25,
            max_realizations=900,
        )

    assert sem > 1.0e-6
    assert n_real <= 900


def test_intertrack_beta_to_precision_rejects_a_nonsense_target() -> None:
    """A non-positive or non-finite target is a caller bug, not a slow run."""
    lem = LEM4(photon=FIG4, cell=FIG4_CELL, stopping_power=_SP, beta_model="intertrack_mc")
    ion = Ion(6, 12, _carbon_energy_for_let(70.0))
    for bad in (0.0, -0.01, math.inf, math.nan):
        with pytest.raises(ValueError, match="target_sem"):
            lem.intertrack_mc_beta_to_precision(ion, target_sem=bad)
