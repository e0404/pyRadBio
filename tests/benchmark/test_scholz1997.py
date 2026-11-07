"""Scholz et al. 1997 LEM I benchmark reference data."""

from __future__ import annotations

import numpy as np
import pytest

from pyradbio import Ion
from pyradbio.benchmark import scholz1997
from pyradbio.models.lem import _mc_kernel
from pyradbio.models.lem.lem1 import LEM1
from pyradbio.track.stopping_power import SimpleBetheStoppingPower


def _cho_model(beta_model: str = "scholz1997_stochastic") -> LEM1:
    # Pinned to the stopping power this reconstruction was validated with (the library
    # default is ICRU90Procedure: carbon's LET at 10 MeV/u differs by -0.8%).
    # The MC-beta test is statistical and passes with either.
    return scholz1997.cho_lem1(SimpleBetheStoppingPower(), beta_model=beta_model)


# Reduced quadrature for the (otherwise ~1.5 s) two-track beta in fast tests; it
# matches the default-resolution result to <1% but runs in ~0.1 s.
_TWO_TRACK_FAST = scholz1997.TWO_TRACK_FAST_QUADRATURE


def test_cho_inputs_match_paper_caption() -> None:
    """The bundled CHO reference cell matches the Scholz 1997 Fig. 1 caption."""
    inputs = scholz1997.CHO_INPUTS

    assert inputs.alpha_x == pytest.approx(0.18)
    assert inputs.beta_x == pytest.approx(0.028)
    assert inputs.D_t == pytest.approx(30.0)
    assert inputs.nucleus_radius_um == pytest.approx(5.0)  # 10 um diameter
    assert inputs.effective_size_factor == pytest.approx(1.0)


def test_lem1_curve_loader_groups_digitized_points() -> None:
    """Digitized curves are grouped by figure, particle, path, and energy."""
    curves = scholz1997.load_lem1_curves()

    assert len(curves) >= 30
    assert {curve.figure for curve in curves} == {
        "fig1_alpha_vs_energy",
        "fig2_exact_ln_survival_over_dose",
        "fig3_beta_vs_energy",
        "fig4_slope_vs_specific_energy",
        "fig5_survival_vs_dose",
    }
    assert all(len(curve.x) == len(curve.y) >= 2 for curve in curves)
    assert all(curve.doi == scholz1997.SCHOLZ1997_DOI for curve in curves)


def test_lem1_fig1_exact_proton_curve_is_easy_to_select() -> None:
    """The exact/full stochastic Fig. 1 proton alpha curve is queryable."""
    curve = scholz1997.get_lem1_curve(
        "fig1_alpha_vs_energy",
        particle="proton",
        calculation="exact",
    )

    assert curve.x_quantity == "energy"
    assert curve.x_unit == "MeV/u"
    assert curve.y_quantity == "alpha"
    assert curve.y_unit == "Gy^-1"
    assert curve.x[0] < 2.0
    assert curve.x[-1] >= 100.0
    assert max(curve.y) == pytest.approx(1.65, rel=0.08)
    assert curve.y[-1] < 0.35


def test_lem1_fig3_beta_contains_exact_and_approximation_paths() -> None:
    """Fig. 3 keeps exact/full-stochastic and approximation beta curves separate."""
    exact = scholz1997.get_lem1_curve(
        "fig3_beta_vs_energy",
        particle="carbon",
        calculation="exact",
    )
    approx = scholz1997.get_lem1_curve(
        "fig3_beta_vs_energy",
        particle="carbon",
        calculation="approximation",
    )

    assert exact.y_unit == "Gy^-2"
    assert approx.y_unit == "Gy^-2"
    assert approx.y[-1] > exact.y[0]
    assert any(y > 0.02 for y in approx.y)


@pytest.mark.slow
def test_lem1_alpha_reproduces_paper_peak_positions() -> None:
    """LEM I with CHO inputs reproduces the paper's qualitative Fig. 1 shape.

    The proton initial slope falls monotonically over 1-200 MeV/u, while the
    carbon slope peaks at a few MeV/u (Scholz 1997 Fig. 1). The peak positions and
    magnitudes are checked loosely against the digitized reference.
    """
    model = _cho_model()
    energies = np.geomspace(1.0, 200.0, 80)

    proton = np.array([model.initial_slope(Ion(1, 1, float(e))) for e in energies])
    carbon = np.array([model.initial_slope(Ion(6, 12, float(e))) for e in energies])

    # Proton alpha is monotonically decreasing across the therapeutic window.
    assert np.all(np.diff(proton) < 1.0e-3)
    # Carbon alpha peaks in the low-MeV/u range, not at the ends of the sweep.
    peak_energy = float(energies[int(np.argmax(carbon))])
    assert 2.0 < peak_energy < 12.0
    # Peak magnitudes are within ~10% of the digitized paper exact curves.
    paper_carbon = scholz1997.get_lem1_curve(
        "fig1_alpha_vs_energy", particle="carbon", calculation="exact"
    )
    assert carbon.max() == pytest.approx(max(paper_carbon.y), rel=0.10)


@pytest.mark.slow
def test_lem1_approximation_alpha_tracks_the_cross_section_at_high_energy() -> None:
    """The Fig. 1 approximation must converge on the exact slope, not diverge.

    Regression: the approximation must not stay below ``0.9 x`` the cross-section at
    200 MeV/u -- that would encode a bookkeeping bug as intended behaviour. At high
    energy the track field over the nucleus is nearly
    homogeneous, so the two routes are computing the same nearly-photon response
    and *must* agree; a 34% gap was the tell.

    The bare central-traversal form scored only tracks that hit the nucleus while
    charging them for the whole LET, so it converged to ``alpha_x f`` with ``f``
    the in-nucleus energy fraction (0.60 at proton 200 MeV/u).
    :meth:`LEM1.scholz1997_central_initial_slope` restores the missed tracks'
    photon-rate contribution and the two agree to well under a percent, while
    staying identical below ~15 MeV/u where ``f = 1``.
    """
    model = _cho_model()

    low = Ion(1, 1, 2.0)
    assert model.scholz1997_central_initial_slope(low) == pytest.approx(
        model.initial_slope(low), rel=0.02
    )

    for energy in (100.0, 200.0, 400.0):
        high = Ion(1, 1, energy)
        approx = model.scholz1997_central_initial_slope(high)
        exact = model.initial_slope(high)
        assert approx == pytest.approx(exact, rel=0.02), (
            f"proton {energy} MeV/u: approximation {approx:.4f} vs exact {exact:.4f}"
        )
        assert approx >= model.photon.alpha_x


def test_two_track_beta_recovers_photon_limit() -> None:
    """The two-track overlap beta approaches the photon beta_x at low LET.

    At very low LET (high-energy proton) tracks are sparse and near-linear, so the
    pairwise overlap reduces to the photon quadratic term: beta -> beta_x. This is
    the exact analytic limit of the two-track construction and the strongest check
    that its normalisation is right.
    """
    model = _cho_model("scholz1997_two_track")
    beta = model.scholz1997_two_track_beta(Ion(1, 1, 200.0), **_TWO_TRACK_FAST)

    assert beta == pytest.approx(model.photon.beta_x, rel=0.05)


def test_two_track_beta_rises_with_energy_and_stays_below_photon() -> None:
    """Two-track beta rises with energy toward beta_x (Scholz 1997 Fig. 3 shape).

    Higher energy means lower LET, less overlap saturation, and a beta closer to
    the photon limit; carbon beta must increase monotonically over 10-200 MeV/u
    and stay at or below beta_x.
    """
    model = _cho_model("scholz1997_two_track")
    betas = [
        model.scholz1997_two_track_beta(Ion(6, 12, float(e)), **_TWO_TRACK_FAST)
        for e in (10.0, 50.0, 200.0)
    ]

    assert betas[0] < betas[1] < betas[2]
    assert all(0.0 <= b <= model.photon.beta_x * 1.02 for b in betas)


def test_explicit_lq_path_is_distinct_from_fast_approximation_in_fig5() -> None:
    """Fig. 5 exact curves use explicit alpha/beta, not the fast z simulation.

    The reconstruction error this pins is *substitution*: the two paths are
    different models and the explicit one is what Figs. 2 and 5 plot, so
    :meth:`LEM1.scholz1997_explicit_lq` must return the impact-parameter initial
    slope bundled with the two-track curvature, and the fast path must stay
    numerically distinguishable from it.

    **It does not assert that the explicit path is closer to the paper.** With the
    missed-track halo included the two alphas agree to 0.3% (0.2054 explicit,
    0.2060 fast) and the residual is all curvature: over the Fig. 5 dose range the
    z-domain beta lands at 0.02873 against the paper's 0.02860, while the
    two-track beta gives 0.02809, so the *fast* path tracks the paper's exact
    curve about five times better here. That is a real, separately open ~2%
    two-track beta deficit, not a reason to prefer the fast path for Fig. 5.

    Reference: Scholz et al. 1997 Figs. 2 and 5,
    doi:10.1007/s004110050055.
    """
    model = _cho_model("scholz1997_stochastic")
    ion = Ion(1, 1, 200.0)
    explicit = model.scholz1997_explicit_lq(ion, **_TWO_TRACK_FAST)
    paper = scholz1997.get_lem1_curve(
        "fig5_survival_vs_dose",
        particle="proton",
        calculation="exact",
        energy_MeV_u=200.0,
    )
    doses = np.asarray(paper.x)
    paper_log_kill = -np.log(np.asarray(paper.y))
    explicit_log_kill = explicit.alpha * doses + explicit.beta * doses**2
    fast_log_kill = np.asarray(model.scholz1997_approximate_log_kills(ion, doses))

    assert explicit.alpha == pytest.approx(model.initial_slope(ion))
    assert explicit.beta == pytest.approx(model.scholz1997_two_track_beta(ion, **_TWO_TRACK_FAST))

    # Distinct models: substituting one for the other must remain detectable.
    separation = float(np.max(np.abs(explicit_log_kill - fast_log_kill)))
    assert separation > 0.05, f"paths collapsed onto each other (max gap {separation:.4f})"

    # Both must stay in the paper's neighbourhood -- a guard against either path
    # drifting off while the other holds it up.
    for name, log_kill in (("explicit", explicit_log_kill), ("fast", fast_log_kill)):
        rms = float(np.sqrt(np.mean((log_kill - paper_log_kill) ** 2)))
        assert rms < 0.1, f"{name} path RMS {rms:.4f} against the paper's Fig. 5 curve"


@pytest.mark.slow
def test_two_track_beta_matches_paper_fig3_exact() -> None:
    """Two-track beta reproduces the digitized Scholz 1997 Fig. 3 exact curve.

    The paper's "exact" beta is the full stochastic LEM; the two-track overlap is
    its second-order-in-fluence reconstruction. Agreement is checked loosely (the
    reference is eyeball-digitized at ~10-20% and the model omits third- and
    higher-order overlaps).
    """
    model = _cho_model("scholz1997_two_track")
    paper = scholz1997.get_lem1_curve("fig3_beta_vs_energy", particle="carbon", calculation="exact")

    for energy in (50.0, 100.0, 200.0):
        beta = model.alpha_beta(Ion(6, 12, energy)).beta
        reference = float(np.interp(energy, paper.x, paper.y))
        assert beta == pytest.approx(reference, rel=0.25)


def test_montecarlo_runs_at_high_energy() -> None:
    """The stochastic MC has no ``n_bar`` ceiling, and must not grow one.

    Regression: carbon at 100 MeV/u must not raise "intractable". Such a ceiling would
    guard a *uniform* sampler, whose cost is ``O(n_bar^2)`` and whose accuracy collapses
    above ``n_bar ~ 1400``. Weighted sampling has neither problem -- precision is flat
    in the track count, measured to ``n_bar = 336 293`` -- so a guard would block exactly
    the runs the sampler exists to make possible. Carbon 100 has ``n_bar = 38 675``
    at 3 Gy and finishes here in well under a second.
    """
    model = _cho_model("scholz1997_montecarlo")

    beta = model.scholz1997_montecarlo_beta(
        Ion(6, 12, 100.0), n_realizations=50, n_field_points=500, b_cut_um=10.0
    )
    assert 0.0 <= beta < 2.0 * model.photon.beta_x


@pytest.mark.slow
def test_montecarlo_single_traversal_matches_n1() -> None:
    """One central-traversal MC lethal-event integral equals the analytic N_1.

    Validates the weighted (Kellerer) field integral against
    ``single_traversal_log_kill`` (Scholz & Kraft 1994 eq. 6) for a single track
    at the nucleus centre. With one track there is no overlap, so the target is
    an exact deterministic number and this pins the sampler geometry -- the
    dose-proportional radial draw and the in-nucleus arc azimuth -- rather than
    comparing one Monte Carlo against another.
    """
    model = _cho_model("scholz1997_montecarlo")
    ion = Ion(6, 12, 5.0)
    r, dose, r_lo, r_max = model._two_track_dose_profile(ion, 400)
    table = model.nuclear_dose_table(ion)
    centre = np.zeros((1, 2))
    values = [
        model.kellerer_field_integral(
            centre, np.random.default_rng(k), table, r, dose, r_lo, r_max, 8000
        )
        for k in range(12)
    ]

    assert float(np.mean(values)) == pytest.approx(model.single_traversal_log_kill(ion), rel=0.02)


#: Batches x realizations of the MC-beta test: 48 000 realizations in 10 independent batches.
_MC_BATCHES, _MC_PER_BATCH = 10, 4800


@pytest.mark.slow
def test_montecarlo_alpha_and_small_carbon_beta() -> None:
    """The MC reproduces the cross-section alpha and a small low-energy carbon beta.

    The population initial slope must match the deterministic cross-section
    ``initial_slope`` (a strong validation of the whole MC), and the dose-domain
    carbon beta at 10 MeV/u must be small -- consistent with the two-track value and
    clearly below the z-domain curvature ``beta_z`` -- confirming the shallow Fig. 3 slope.

    Statistical form: beta is a small curvature on a dominant linear term, so
    a single run's beta is noise-dominated (12 000 realizations: 0.0005-0.0061 over seeds
    1-5 against a fixed bound of 0.4 beta_z = 0.0052, which sits at the true mean). The
    test pools 10 independent batches and asserts against the batch-means standard
    error sigma_beta. Validated over 10 base seeds x 2 stopping powers (SimpleBethe, the
    library default): sigma_beta = 0.0011, tracking the seed-to-seed spread (0.0017 /
    0.0010) within 0.9-1.5x; worst (beta + 3 sigma) / beta_z 0.68 / 0.71; worst
    |beta - beta_two_track| 2.1 / 1.9 sigma (bound 4 sigma: a 10-batch sigma is itself
    uncertain by ~25%); alpha within 0.82% / 0.57%. Hence the bounds: beta + 3 sigma below
    0.8 beta_z (the worst measured is 0.71), and sigma_beta itself below 0.002.
    """
    model = _cho_model("scholz1997_montecarlo")
    two_track = _cho_model("scholz1997_two_track")
    z_domain = _cho_model("scholz1997_stochastic")
    ion = Ion(6, 12, 10.0)

    doses = np.array([0.5, 1.0, 1.5, 2.0, 2.5, 3.0])
    design = np.vstack([np.ones_like(doses), doses]).T

    def fit(survival: np.ndarray) -> np.ndarray:
        return np.linalg.lstsq(design, -np.log(survival) / doses, rcond=None)[0]

    # engine="numpy": reproducible for a fixed seed (numba's prange reduction is not).
    batches = np.array(
        [
            model.scholz1997_montecarlo_survival(
                ion, doses, n_realizations=_MC_PER_BATCH, seed=1000 + k, engine="numpy"
            )
            for k in range(_MC_BATCHES)
        ]
    )
    alpha_mc, beta_mc = fit(batches.mean(axis=0))
    beta_batches = np.array([fit(b)[1] for b in batches])
    sigma_beta = float(beta_batches.std(ddof=1) / np.sqrt(_MC_BATCHES))

    assert alpha_mc == pytest.approx(model.initial_slope(ion), rel=0.02)
    beta_z = z_domain.scholz1997_specific_energy_lq(ion).beta
    # The error bar itself is bounded, so a noisier MC cannot widen the two bounds below.
    assert sigma_beta < 0.002
    assert beta_mc + 3.0 * sigma_beta < 0.8 * beta_z
    assert abs(beta_mc - two_track.scholz1997_two_track_beta(ion)) < 4.0 * sigma_beta


def test_montecarlo_numba_engine_requires_fast_extra(monkeypatch: pytest.MonkeyPatch) -> None:
    """``engine='numba'`` raises a helpful error when the fast extra is absent.

    The compiled kernel is optional; without numba the survival call must refuse
    ``engine='numba'`` (and point at the alternatives) rather than fail obscurely.
    """
    model = _cho_model("scholz1997_montecarlo")
    monkeypatch.setattr(_mc_kernel, "HAVE_NUMBA", False)

    with pytest.raises(ValueError, match="fast"):
        model.scholz1997_montecarlo_survival(
            Ion(6, 12, 10.0), [1.0], n_realizations=10, engine="numba"
        )


@pytest.mark.slow
def test_montecarlo_engines_agree_within_mc_error() -> None:
    """The numba engine matches the NumPy reference statistically.

    Both accumulate the same estimator; only the RNG and execution differ, so the
    population survival must agree to within a few Monte-Carlo standard errors
    (``~1/sqrt(N)``). This guards the compiled kernel against silently diverging
    from the reference loop.
    """
    model = _cho_model("scholz1997_montecarlo")
    ion = Ion(6, 12, 10.0)
    doses = np.array([1.0, 2.5])
    n = 4000

    # Pin engine="numpy": the default "auto" resolves to numba whenever the fast
    # extra is installed, so this reference was silently the numba kernel and the
    # test compared numba against itself rather than against the NumPy loop.
    s_numpy = model.scholz1997_montecarlo_survival(
        ion, doses, n_realizations=n, seed=7, engine="numpy"
    )
    tol = 6.0 / np.sqrt(n)  # ~0.095; observed differences are ~0.002

    if _mc_kernel.HAVE_NUMBA:
        s_numba = model.scholz1997_montecarlo_survival(
            ion, doses, n_realizations=n, seed=7, engine="numba"
        )
        assert np.max(np.abs(s_numba - s_numpy)) < tol
