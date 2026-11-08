"""LQ least-squares fits in :mod:`pyradbio.core.lq`.

Three kinds of test:

* the fits do what they say (exact recovery of planted LQ data, weighting,
  the intercept switch) and refuse degenerate inputs with a clear message;
* **bit-equality with every script/test copy they replace**: each copy's
  expression is reproduced inline here, in its own order, and compared with
  ``==`` -- not ``approx`` -- because migrating a script must change no printed
  digit;
* **recorded-input regression for the two library copies that delegate**
  (``_localdose.lq_coefficients``, ``_intertrack_kernel.fit_alpha_beta``): the
  inputs below are real LEM IV dose responses (Friedrich 2012 Fig. 4 cell,
  ``fit_doses_Gy()``), and the outputs were recorded from the original *pre-delegation*
  functions, stored as ``float.hex``. OpenBLAS picks CPU-specific kernels, so the
  least squares differs in the last bits between machines and a recorded output
  is matched to ``_RECORDED_REL``; bit-equality is asserted against the frozen
  pre-delegation copy on the running machine.
"""

from __future__ import annotations

import numpy as np
import numpy.typing as npt
import pytest

from pyradbio.core.lq import fit_lq_linearised, fit_lq_weighted
from pyradbio.models.lem import _intertrack_kernel, _localdose

FloatArray = npt.NDArray[np.float64]


#: Cross-machine match of a recorded output (the GitLab runner differs by up to
#: 4e-14 relative; see the module docstring).
_RECORDED_REL = 1e-12


def _fromhex(values: tuple[str, ...]) -> FloatArray:
    return np.array([float.fromhex(v) for v in values], dtype=np.float64)


def _recorded(expected: tuple[str, str]) -> tuple[float, float]:
    return float.fromhex(expected[0]), float.fromhex(expected[1])


# ---------------------------------------------------------------------------
# Recorded real inputs and pre-delegation outputs (see the module docstring).
# Recorded with LEM4(photon=PhotonParameters(0.25, 0.05, D_t=9.5),
# cell=CellTarget(nucleus_radius_um=5.0)) on the default fit_doses_Gy() grid:
#   lq_coefficients:  doses, local_dose_lethal_events(Ion(6, 12, E), doses)
#   fit_alpha_beta:   doses, intertrack_mc_survival(Ion(6, 12, E), doses,
#                     n_realizations=40, seed, engine="numpy")  (E=3 seed 0,
#                     E=20 seed 1) and intertrack_mc_photon_survival(doses,
#                     n_realizations=40, seed=2)
# ---------------------------------------------------------------------------
_RECORDED_DOSES_HEX = (
    "0x1.e666666666667p-2", "0x1.6cccccccccccdp-1", "0x1.e666666666667p-1",
    "0x1.6cccccccccccdp+0", "0x1.e666666666667p+0", "0x1.3000000000000p+1",
    "0x1.851eb851eb852p+1", "0x1.e666666666667p+1", "0x1.3000000000000p+2",
    "0x1.7f0a3d70a3d71p+2", "0x1.c800000000000p+2", "0x1.0266666666666p+3",
)  # fmt: skip
_RECORDED_LQ_COEFFICIENTS = (
    (
        1.0,
        (
            "0x1.95326c15bd2a1p-1", "0x1.37dccbe201c9ap+0", "0x1.abd7c01015de2p+0",
            "0x1.50509e6a206a4p+1", "0x1.d011487235e22p+1", "0x1.2997336f8e710p+2",
            "0x1.8724202368455p+2", "0x1.f3a1eb2d8a59cp+2", "0x1.3e53764ca0dfbp+3",
            "0x1.97fc619385beep+3", "0x1.eb0bda0e1df9ep+3", "0x1.18282d6212b21p+4",
        ),
        ("0x1.bd5da62c5388bp+0", "0x1.043099aad3709p-4"),
    ),
    (
        4.0,
        (
            "0x1.47d557d8645f1p+0", "0x1.ee173404f6f53p+0", "0x1.4a392ef756b7bp+1",
            "0x1.f0759c7c03440p+1", "0x1.4b346e018ab69p+2", "0x1.9e086cea1baf1p+2",
            "0x1.08d8d787a593ap+3", "0x1.4ac09c2595cb2p+3", "0x1.9cd1b21cd9e1cp+3",
            "0x1.037ff49c15cf9p+4", "0x1.3442a873f4bd9p+4", "0x1.5cb4b14f17e26p+4",
        ),
        ("0x1.5bdf959de2a7ep+1", "-0x1.4eca43705b70ap-10"),
    ),
    (
        30.0,
        (
            "0x1.0f6be9f02d51ap-1", "0x1.9944fa77f2f61p-1", "0x1.1239babe86cc3p+0",
            "0x1.9f477b8f156cfp+0", "0x1.174fa15feb031p+1", "0x1.600735575d667p+1",
            "0x1.c7690a8fa0ab9p+1", "0x1.1fc7af5329eafp+2", "0x1.6c2cc1fc946bbp+2",
            "0x1.d150437ce0305p+2", "0x1.181945795224fp+3", "0x1.401b0c3a40145p+3",
        ),
        ("0x1.1d812a828a2a2p+0", "0x1.0aad5ff861b58p-6"),
    ),
    (
        200.0,
        (
            "0x1.48eb038107586p-3", "0x1.ff0d0bf85a632p-3", "0x1.607b5a97b6f5dp-2",
            "0x1.1a07aac0d646bp-1", "0x1.8f97a580d6fb1p-1", "0x1.08763e6bec826p+0",
            "0x1.6ce031b54c0c2p+0", "0x1.edbe89a2cb152p+0", "0x1.51fdca3aa77b5p+1",
            "0x1.da0049a6377dbp+1", "0x1.3492ca86eb681p+2", "0x1.76aa2765095e0p+2",
        ),
        ("0x1.418e4c6a7f547p-2", "0x1.a106c1d381ecfp-5"),
    ),
)  # fmt: skip
_RECORDED_FIT_ALPHA_BETA = (
    (
        3.0,
        (
            "0x1.72765a1442f7bp-4", "0x1.22c17567beaebp-6", "0x1.d92b4f6bd3d06p-8",
            "0x1.c3626d68a3b90p-12", "0x1.6b94ce528e961p-16", "0x1.32f8665ff3e71p-19",
            "0x1.7b025bf2fff9bp-26", "0x1.6e81b43415ce3p-32", "0x1.13549b51ed186p-39",
            "0x1.52219c733dac7p-49", "0x1.da2c438c69313p-60", "0x1.c009bc058ad3ap-68",
        ),
        ("0x1.57c1f2e9c40b6p+2", "0x1.df92216364fb4p-5"),
    ),
    (
        20.0,
        (
            "0x1.82e232fd265d2p-2", "0x1.191075ca78671p-2", "0x1.31d76339e0439p-3",
            "0x1.b7d883a0fa5e9p-5", "0x1.5483f187df379p-6", "0x1.b4a8c8f86058cp-8",
            "0x1.a0a15c58dca33p-10", "0x1.0fd6921d85ca5p-12", "0x1.0f16ea413d2ecp-15",
            "0x1.20604f519b398p-20", "0x1.b77efa41b99d2p-25", "0x1.7de7c3a281106p-29",
        ),
        ("0x1.ed83af13005fdp+0", "0x1.f7496b59bfecdp-5"),
    ),
    (
        "photon",
        (
            "0x1.bc4711bf4d03ep-1", "0x1.a78cd478c88b5p-1", "0x1.88eb27bfdd823p-1",
            "0x1.4b7dd80a17646p-1", "0x1.06d54346f4b22p-1", "0x1.9367fd98e29c7p-2",
            "0x1.1a1cbe3834ba6p-2", "0x1.5f7e15c9497c5p-3", "0x1.a4cb05eb9507bp-4",
            "0x1.1c4b626dd9835p-5", "0x1.de4fe001d18bdp-7", "0x1.bf599137a9c84p-8",
        ),
        ("0x1.081a0814b8fbdp-2", "0x1.868b899c12789p-5"),
    ),
)  # fmt: skip

#: Synthetic inputs that exercise ``fit_alpha_beta``'s ``clip(S, 1e-300, 1)`` --
#: a survival of exactly 0 and one below the floor (lower bound), and a survival
#: above 1 (upper bound: S = 1.02 is clipped to 1, so the third case records the
#: same numbers as the second) -- recorded the same way.
_RECORDED_FIT_ALPHA_BETA_CLIP = (
    (
        (0.5, 1.0, 2.0, 3.0, 4.0),
        (0.3, 0.05, 1e-200, 0.0, 0.0),
        ("-0x1.a6d763371867ep+0", "0x1.ecdb13c5c54f2p+5"),
    ),
    (
        (0.5, 1.0, 2.0, 3.0, 4.0),
        (1.0, 1.0, 0.9, 0.5, 0.2),
        ("-0x1.c47b7ecb14a6ap-4", "0x1.e31b97b23c881p-4"),
    ),
    (
        (0.5, 1.0, 2.0, 3.0, 4.0),
        (1.02, 1.0, 0.9, 0.5, 0.2),
        ("-0x1.c47b7ecb14a6ap-4", "0x1.e31b97b23c881p-4"),
    ),
)


# ---------------------------------------------------------------------------
# Inline reproductions of every copy the two fits replace, in their own order.
# ---------------------------------------------------------------------------
def _script_mc_batch_scholz1997(survival: FloatArray, doses: FloatArray) -> tuple[float, float]:
    """The original ``mc_batch_scholz1997`` script's ``fit_alpha_beta`` (verbatim body).

    Same expression as ``scholz1997_montecarlo_beta`` in ``pyradbio/models/lem/base.py``
    (before its ``max(beta, 0)``) and in ``tests/benchmark/test_scholz1997.py``.
    """
    log_kills = -np.log(survival)
    design = np.vstack([np.ones_like(doses), doses]).T
    coeffs, *_ = np.linalg.lstsq(design, log_kills / doses, rcond=None)
    return float(coeffs[0]), float(coeffs[1])


def _script_survival_code_lq_fit(
    doses: FloatArray, log_kill: FloatArray, sigma: FloatArray
) -> tuple[float, float]:
    """The original Survival-comparison script's ``_lq_fit`` (verbatim body)."""
    w = 1.0 / np.maximum(sigma, 1e-12)
    design = np.column_stack([doses, doses**2]) * w[:, None]
    coeffs, *_ = np.linalg.lstsq(design, log_kill * w, rcond=None)
    return float(coeffs[0]), float(coeffs[1])


def _unweighted_quadratic(dose: FloatArray, effect: FloatArray) -> tuple[float, float]:
    """Plain ``(D, D^2)`` least squares, no weights, no intercept.

    As in ``pyradbio/benchmark/matsuya2025.py`` and
    ``tests/benchmark/test_elsasser2007.py``.
    """
    design = np.vstack([dose, dose**2]).T
    (alpha, beta), *_ = np.linalg.lstsq(design, effect, rcond=None)
    return float(alpha), float(beta)


def _pre_delegation_lq_coefficients(doses: FloatArray, effect: FloatArray) -> tuple[float, float]:
    """Frozen copy of the original ``_localdose.lq_coefficients`` body (verbatim)."""
    doses = np.asarray(doses, dtype=np.float64)
    effect = np.asarray(effect, dtype=np.float64)
    y = effect / doses
    design = np.vstack([np.ones_like(doses), doses]).T
    alpha, beta = np.linalg.lstsq(design, y, rcond=None)[0]
    return float(alpha), float(beta)


def _pre_delegation_fit_alpha_beta(doses: FloatArray, survival: FloatArray) -> tuple[float, float]:
    """Frozen copy of the original ``_intertrack_kernel.fit_alpha_beta`` body (verbatim)."""
    doses = np.asarray(doses, dtype=np.float64)
    survival = np.asarray(survival, dtype=np.float64)
    y = -np.log(np.clip(survival, 1.0e-300, 1.0)) / doses
    design = np.vstack([np.ones_like(doses), doses]).T
    coef = np.linalg.lstsq(design, y, rcond=None)[0]
    return float(coef[0]), float(coef[1])


def _noisy_lq(seed: int, n: int = 6) -> tuple[FloatArray, FloatArray, FloatArray]:
    """Doses, a noisy (positive) LQ log-kill and a per-point sigma, MC-like."""
    rng = np.random.default_rng(seed)
    doses = np.sort(rng.uniform(0.3, 8.0, n))
    sigma = rng.uniform(0.01, 0.2, n)
    log_kill = 0.4 * doses + 0.03 * doses**2 + rng.normal(0.0, sigma)
    return doses, np.abs(log_kill) + 1.0e-3, sigma


_SEEDS = tuple(range(8))


# ---------------------------------------------------------------------------
# fit_lq_linearised
# ---------------------------------------------------------------------------
def test_linearised_recovers_planted_lq() -> None:
    doses = np.array([0.5, 1.0, 1.5, 2.0, 2.5, 3.0])
    alpha, beta = fit_lq_linearised(doses, 0.7 * doses + 0.045 * doses**2)
    assert alpha == pytest.approx(0.7, rel=1e-12)
    assert beta == pytest.approx(0.045, rel=1e-11)


def test_linearised_keeps_a_negative_beta_and_accepts_sequences() -> None:
    """No clamp: a concave response is a measurement (overkill), not an error."""
    doses = [1.0, 2.0, 4.0, 8.0]
    alpha, beta = fit_lq_linearised(doses, [0.3 * d - 0.02 * d * d for d in doses])
    assert isinstance(alpha, float) and isinstance(beta, float)
    assert alpha == pytest.approx(0.3, rel=1e-12)
    assert beta == pytest.approx(-0.02, rel=1e-11)


def test_linearised_two_points_interpolate_exactly() -> None:
    alpha, beta = fit_lq_linearised([1.0, 2.0], [0.35, 0.8])  # E/D = 0.35, 0.40
    assert alpha == pytest.approx(0.3, rel=1e-12)
    assert beta == pytest.approx(0.05, rel=1e-12)


@pytest.mark.parametrize(
    ("doses", "log_kill", "match"),
    [
        ([1.0], [0.3], "at least 2 doses"),
        ([], [], "at least 2 doses"),
        ([1.0, 2.0, 3.0], [0.3, 0.7], "same shape"),
        ([[1.0, 2.0], [3.0, 4.0]], [[0.3, 0.7], [1.2, 1.9]], "1-D"),
        ([0.0, 1.0, 2.0], [0.0, 0.3, 0.7], "doses must be finite and > 0"),
        ([-1.0, 1.0, 2.0], [0.3, 0.3, 0.7], "doses must be finite and > 0"),
        ([np.nan, 1.0, 2.0], [0.3, 0.3, 0.7], "doses must be finite and > 0"),
        ([1.0, 2.0, np.inf], [0.3, 0.7, 1.0], "doses must be finite and > 0"),
        ([1.0, 2.0, 3.0], [0.3, np.nan, 1.0], "log_kill/effect must be finite"),
        ([1.0, 2.0, 3.0], [0.3, np.inf, 1.0], "log_kill/effect must be finite"),
        ([2.0, 2.0, 2.0], [0.3, 0.31, 0.32], "at least 2 distinct doses"),
    ],
)
def test_linearised_rejects_degenerate_inputs(
    doses: list[float], log_kill: list[float], match: str
) -> None:
    with pytest.raises(ValueError, match=match):
        fit_lq_linearised(doses, log_kill)


@pytest.mark.parametrize("seed", _SEEDS)
def test_linearised_is_bit_equal_to_mc_batch_scholz1997(seed: int) -> None:
    """Replaces the script's ``fit_alpha_beta(survival, doses)``: pass ``-ln S``."""
    doses, log_kill, _sigma = _noisy_lq(seed)
    survival = np.exp(-log_kill)
    expected = _script_mc_batch_scholz1997(survival, doses)
    assert fit_lq_linearised(doses, -np.log(survival)) == expected


def test_linearised_is_bit_equal_to_mc_batch_scholz1997_on_its_dose_grid() -> None:
    """The script's own ``FIT_DOSES_GY`` and a jackknife-like survival row."""
    doses = np.array([0.5, 1.0, 1.5, 2.0, 2.5, 3.0])
    rng = np.random.default_rng(1000)
    survival = np.exp(-(1.1 * doses + 0.02 * doses**2)) * rng.uniform(0.97, 1.03, doses.size)
    expected = _script_mc_batch_scholz1997(survival, doses)
    assert fit_lq_linearised(doses, -np.log(survival)) == expected


@pytest.mark.parametrize("seed", _SEEDS)
def test_linearised_is_bit_equal_to_the_pre_delegation_library_copies(seed: int) -> None:
    doses, log_kill, _sigma = _noisy_lq(seed, n=12)
    assert fit_lq_linearised(doses, log_kill) == _pre_delegation_lq_coefficients(doses, log_kill)
    survival = np.exp(-log_kill)
    expected = _pre_delegation_fit_alpha_beta(doses, survival)
    assert fit_lq_linearised(doses, -np.log(np.clip(survival, 1.0e-300, 1.0))) == expected


# ---------------------------------------------------------------------------
# Recorded-input bit-equality of the delegating library copies
# ---------------------------------------------------------------------------
@pytest.mark.parametrize(
    ("label", "effect", "expected"),
    _RECORDED_LQ_COEFFICIENTS,
    ids=[f"E={case[0]}" for case in _RECORDED_LQ_COEFFICIENTS],
)
def test_lq_coefficients_unchanged_on_recorded_lem4_inputs(
    label: float, effect: tuple[str, ...], expected: tuple[str, str]
) -> None:
    doses, values = _fromhex(_RECORDED_DOSES_HEX), _fromhex(effect)
    want = _pre_delegation_lq_coefficients(doses, values)
    assert want == pytest.approx(_recorded(expected), rel=_RECORDED_REL)
    assert _localdose.lq_coefficients(doses, values) == want
    assert fit_lq_linearised(doses, values) == want


@pytest.mark.parametrize(
    ("label", "survival", "expected"),
    _RECORDED_FIT_ALPHA_BETA,
    ids=[f"E={case[0]}" for case in _RECORDED_FIT_ALPHA_BETA],
)
def test_fit_alpha_beta_unchanged_on_recorded_lem4_inputs(
    label: float | str, survival: tuple[str, ...], expected: tuple[str, str]
) -> None:
    doses, values = _fromhex(_RECORDED_DOSES_HEX), _fromhex(survival)
    want = _pre_delegation_fit_alpha_beta(doses, values)
    assert want == pytest.approx(_recorded(expected), rel=_RECORDED_REL)
    assert _intertrack_kernel.fit_alpha_beta(doses, values) == want


@pytest.mark.parametrize(("doses", "survival", "expected"), _RECORDED_FIT_ALPHA_BETA_CLIP)
def test_fit_alpha_beta_unchanged_through_its_survival_clip(
    doses: tuple[float, ...], survival: tuple[float, ...], expected: tuple[str, str]
) -> None:
    want = _pre_delegation_fit_alpha_beta(np.array(doses), np.array(survival))
    assert want == pytest.approx(_recorded(expected), rel=_RECORDED_REL)
    assert _intertrack_kernel.fit_alpha_beta(np.array(doses), np.array(survival)) == want


def test_lq_coefficients_keeps_its_too_few_doses_message() -> None:
    with pytest.raises(ValueError, match="need at least 2 doses for an LQ fit, got 1"):
        _localdose.lq_coefficients(np.array([1.0]), np.array([0.3]))


# ---------------------------------------------------------------------------
# fit_lq_weighted
# ---------------------------------------------------------------------------
def test_weighted_recovers_planted_lq_without_intercept() -> None:
    doses = np.array([1.0, 2.0, 3.0, 4.0, 5.0, 6.0])
    sigma = np.array([0.01, 0.02, 0.05, 0.1, 0.2, 0.4])
    alpha, beta = fit_lq_weighted(doses, 0.25 * doses + 0.06 * doses**2, sigma)
    assert alpha == pytest.approx(0.25, rel=1e-12)
    assert beta == pytest.approx(0.06, rel=1e-11)


def test_weighted_recovers_planted_lq_with_intercept() -> None:
    doses = np.array([0.0, 1.0, 2.0, 3.0, 4.0, 5.0])
    sigma = np.full(doses.size, 0.05)
    alpha, beta, c = fit_lq_weighted(
        doses, 0.1 + 0.25 * doses + 0.06 * doses**2, sigma, intercept=True
    )
    assert alpha == pytest.approx(0.25, rel=1e-12)
    assert beta == pytest.approx(0.06, rel=1e-11)
    assert c == pytest.approx(0.1, rel=1e-11)


def test_intercept_switch_changes_the_model_not_just_the_tuple() -> None:
    """A planted offset biases the through-origin fit; the intercept fit absorbs it."""
    doses = np.array([1.0, 2.0, 3.0, 4.0, 5.0, 6.0])
    sigma = np.full(doses.size, 0.1)
    log_kill = 0.2 + 0.25 * doses + 0.06 * doses**2
    without = fit_lq_weighted(doses, log_kill, sigma)
    with_c = fit_lq_weighted(doses, log_kill, sigma, intercept=True)
    assert len(without) == 2 and len(with_c) == 3
    assert without[0] > 0.3  # the offset leaks into alpha
    assert with_c == pytest.approx((0.25, 0.06, 0.2), rel=1e-10)
    # ...and on pure LQ data the fitted intercept is zero to rounding.
    _a, _b, c = fit_lq_weighted(doses, 0.25 * doses + 0.06 * doses**2, sigma, intercept=True)
    assert c == pytest.approx(0.0, abs=1e-12)


def test_weighted_downweights_an_uncertain_outlier_the_unweighted_fit_follows() -> None:
    doses = np.array([1.0, 2.0, 3.0, 4.0, 5.0, 6.0])
    log_kill = 0.25 * doses + 0.06 * doses**2
    log_kill[-1] += 1.5
    sigma = np.array([0.01, 0.01, 0.01, 0.01, 0.01, 100.0])
    weighted = fit_lq_weighted(doses, log_kill, sigma)
    unweighted = fit_lq_weighted(doses, log_kill, np.ones_like(doses))
    assert weighted == pytest.approx((0.25, 0.06), rel=1e-6)
    assert abs(unweighted[1] - 0.06) > 0.02


def test_weighted_with_uniform_sigma_matches_any_uniform_sigma() -> None:
    """Only relative weights matter: a common sigma scale leaves the fit unchanged."""
    doses, log_kill, _sigma = _noisy_lq(3)
    unit = fit_lq_weighted(doses, log_kill, np.ones_like(doses))
    scaled = fit_lq_weighted(doses, log_kill, np.full(doses.size, 7.3))
    assert scaled == pytest.approx(unit, rel=1e-12)


def test_weighted_and_linearised_are_different_estimators() -> None:
    """Same planted truth, but on noisy data the two estimators disagree (by design)."""
    doses, log_kill, sigma = _noisy_lq(5)
    assert fit_lq_weighted(doses, log_kill, sigma) != fit_lq_linearised(doses, log_kill)


def test_weighted_floors_a_zero_sigma_instead_of_dividing_by_zero() -> None:
    """A zero repeat-to-repeat spread (e.g. every repeat survives) gets weight 1e12."""
    doses = np.array([1.0, 2.0, 3.0, 4.0])
    log_kill = np.array([0.31, 0.62, 1.05, 1.5])
    sigma = np.array([0.0, 0.05, 0.05, 0.05])
    got = fit_lq_weighted(doses, log_kill, sigma)
    assert np.all(np.isfinite(got))
    assert got == _script_survival_code_lq_fit(doses, log_kill, sigma)


@pytest.mark.parametrize("seed", _SEEDS)
def test_weighted_is_bit_equal_to_survival_code_lq_fit(seed: int) -> None:
    doses, log_kill, sigma = _noisy_lq(seed)
    expected = _script_survival_code_lq_fit(doses, log_kill, sigma)
    assert fit_lq_weighted(doses, log_kill, sigma) == expected


def test_weighted_is_bit_equal_to_survival_code_on_its_dose_set() -> None:
    """The script's own path, ``_mc_one_point``.

    ``doses = np.asarray(OUR_MC_DOSES_GY, dtype=float)``, sigma the ddof=1 std
    over the repeats, then one fit per repeat.
    """
    doses = np.asarray((1.0, 2.0, 3.0, 4.0, 5.0, 6.0), dtype=float)
    rng = np.random.default_rng(7)
    log_kill = (0.5 * doses + 0.04 * doses**2) * rng.uniform(0.95, 1.05, (8, doses.size))
    sigma = log_kill.std(axis=0, ddof=1)
    got = np.array([fit_lq_weighted(doses, lk, sigma) for lk in log_kill]).T
    expected = np.array([_script_survival_code_lq_fit(doses, lk, sigma) for lk in log_kill]).T
    assert np.array_equal(got, expected)


@pytest.mark.parametrize("seed", _SEEDS)
def test_unit_sigma_is_bit_equal_to_the_unweighted_quadratic_copies(seed: int) -> None:
    """``1/max(1, 1e-12)`` is exactly 1.0, so unit weights reproduce plain least squares."""
    doses, log_kill, _sigma = _noisy_lq(seed)
    expected = _unweighted_quadratic(doses, log_kill)
    assert fit_lq_weighted(doses, log_kill, np.ones_like(doses)) == expected


@pytest.mark.parametrize(
    ("doses", "log_kill", "sigma", "intercept", "match"),
    [
        ([1.0, 2.0, 3.0], [0.3, 0.7], [0.1, 0.1, 0.1], False, "same shape"),
        ([1.0, 2.0, 3.0], [0.3, 0.7, 1.2], [0.1, 0.1], False, "same shape"),
        ([[1.0, 2.0]], [[0.3, 0.7]], [[0.1, 0.1]], False, "1-D"),
        ([1.0, 2.0], [0.3, 0.7], [-0.1, 0.1], False, "sigma must be finite and >= 0"),
        ([1.0, 2.0], [0.3, 0.7], [np.nan, 0.1], False, "sigma must be finite and >= 0"),
        ([-1.0, 2.0], [0.3, 0.7], [0.1, 0.1], False, "doses must be finite and >= 0"),
        ([np.inf, 2.0], [0.3, 0.7], [0.1, 0.1], False, "doses must be finite and >= 0"),
        ([1.0, 2.0], [np.nan, 0.7], [0.1, 0.1], False, "log_kill/effect must be finite"),
        ([1.0], [0.3], [0.1], False, "at least 2 distinct positive doses"),
        ([0.0, 2.0, 2.0], [0.0, 0.7, 0.71], [0.1, 0.1, 0.1], False, "2 distinct positive"),
        ([1.0, 2.0], [0.3, 0.7], [0.1, 0.1], True, "at least 3 distinct doses"),
        ([1.0, 1.0, 2.0, 2.0], [0.3, 0.3, 0.7, 0.7], [0.1] * 4, True, "3 distinct doses"),
    ],
)
def test_weighted_rejects_degenerate_inputs(
    doses: list[float],
    log_kill: list[float],
    sigma: list[float],
    intercept: bool,
    match: str,
) -> None:
    with pytest.raises(ValueError, match=match):
        fit_lq_weighted(doses, log_kill, sigma, intercept=intercept)
