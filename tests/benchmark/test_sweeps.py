"""Sweep helpers: each test reproduces one script/test loop inline and asserts
the helper returns the *same bits* (``assert_array_equal`` treats NaN == NaN).

The loops are copied from their call sites (cited per test) with the model swapped
for a cheap LEM I where the site's model is expensive: the loop body, not the
model, is what the helper replaces.
"""

from __future__ import annotations

import dataclasses
import math

import numpy as np
import pytest
from scipy.optimize import brentq

from pyradbio import CellTarget, Ion, LQParameters, PhotonParameters, rbe
from pyradbio.benchmark import elsasser2007 as es
from pyradbio.benchmark.sweeps import LetSweep, sweep_energies, sweep_lets
from pyradbio.core.lq import dose_for_survival
from pyradbio.models.lem.lem1 import LEM1
from pyradbio.track.stopping_power import SimpleBetheStoppingPower

SP = SimpleBetheStoppingPower()
V79 = es.FIGURE_INPUTS["fig6_rbealpha_vs_let"]


@pytest.fixture(scope="module")
def lem1() -> LEM1:
    """The lem2_cluster_effect.py 'old LEM' V79 model (Fig. 6 inputs), D_t = 30 Gy."""
    return LEM1(
        photon=PhotonParameters(alpha_x=V79.alpha_x, beta_x=V79.beta_x, D_t=V79.D_t_old),
        cell=CellTarget(
            nucleus_radius_um=V79.nucleus_radius_um,
            effective_size_factor=V79.effective_size_factor,
        ),
        stopping_power=SP,
    )


class _FlakyModel:
    """Cheap stand-in whose alpha_beta raises a chosen error at chosen energies."""

    def __init__(self, failures: dict[float, type[Exception]]) -> None:
        self.failures = failures
        self.calls: list[float] = []

    def alpha_beta(self, ion: Ion) -> LQParameters:
        self.calls.append(ion.energy_MeV_u)
        exc = self.failures.get(ion.energy_MeV_u)
        if exc is not None:
            raise exc("synthetic failure")
        return LQParameters(alpha=1.0 / ion.energy_MeV_u, beta=0.01 + 1e-4 * ion.energy_MeV_u)


# ---- sweep_energies: script call sites ---------------------------------------


def test_lem2_carbon_curve_rbe_alpha_bit_identical(lem1: LEM1) -> None:
    """The original ``lem2_cluster_effect`` script's ``carbon_curve(model, "rbe_alpha")``."""
    energies = np.geomspace(1.5, 350.0, 7)
    pts: list[tuple[float, float]] = []
    for energy in energies:
        ion = Ion(Z=6, A=12, energy_MeV_u=float(energy))
        lq = lem1.alpha_beta(ion)
        value = lq.alpha / lem1.photon.alpha_x
        pts.append((SP.let_keV_um(ion), value))
    pts.sort()
    let_ref = np.array([p[0] for p in pts])
    val_ref = np.array([p[1] for p in pts])

    sweep = sweep_energies(
        lem1, 6, 12, energies, stopping_power=SP, on_error="raise", sort_by_let=True
    )

    np.testing.assert_array_equal(sweep.let_keV_um, let_ref)
    np.testing.assert_array_equal(sweep.rbe_alpha(lem1.photon.alpha_x), val_ref)


def test_lem2_carbon_curve_rbe10_bit_identical(lem1: LEM1) -> None:
    """lem2_cluster_effect.py ``carbon_curve(model, "rbe10")``: plain-LQ RBE at 10%."""
    reference_lq = LQParameters(alpha=V79.alpha_x, beta=V79.beta_x)
    energies = np.geomspace(1.5, 350.0, 7)
    pts: list[tuple[float, float]] = []
    for energy in energies:
        ion = Ion(Z=6, A=12, energy_MeV_u=float(energy))
        lq = lem1.alpha_beta(ion)
        value = rbe(0.10, lq.alpha, lq.beta, reference=reference_lq)
        pts.append((SP.let_keV_um(ion), value))
    pts.sort()

    sweep = sweep_energies(
        lem1, 6, 12, energies, stopping_power=SP, on_error="raise", sort_by_let=True
    )

    np.testing.assert_array_equal(sweep.let_keV_um, np.array([p[0] for p in pts]))
    np.testing.assert_array_equal(
        sweep.rbe10_lq(V79.alpha_x, V79.beta_x), np.array([p[1] for p in pts])
    )


def test_lem3_rbe_alpha_vs_let_bit_identical(lem1: LEM1) -> None:
    """The original ``lem3_velocity_dependent_track_core`` script's ``rbe_alpha_vs_let``:
    a descending Python-list energy grid, ``sorted`` by LET."""
    curve_energies = [300.0, 120.0, 40.0, 15.0, 5.0, 1.8]
    alpha_x = lem1.photon.alpha_x
    pts = sorted(
        (SP.let_keV_um(Ion(6, 12, e)), lem1.alpha_beta(Ion(6, 12, e)).alpha / alpha_x)
        for e in curve_energies
    )

    sweep = sweep_energies(
        lem1, 6, 12, curve_energies, stopping_power=SP, on_error="raise", sort_by_let=True
    )

    np.testing.assert_array_equal(sweep.let_keV_um, np.array([let for let, _ in pts]))
    np.testing.assert_array_equal(sweep.rbe_alpha(alpha_x), np.array([r for _, r in pts]))
    # Falling energy is rising LET above the Bragg peak: the sort keeps the order.
    np.testing.assert_array_equal(sweep.energy_MeV_u, np.array(curve_energies))


def test_elsasser2007_fig6_peak_loop_bit_identical(lem1: LEM1) -> None:
    """tests/benchmark/test_elsasser2007.py -- unsorted lists, argmax by index."""
    lets, rbes = [], []
    for energy in np.geomspace(2.0, 400.0, 6):
        ion = Ion(Z=6, A=12, energy_MeV_u=float(energy))
        lets.append(SP.let_keV_um(ion))
        rbes.append(lem1.alpha_beta(ion).alpha / V79.alpha_x)

    sweep = sweep_energies(
        lem1, 6, 12, np.geomspace(2.0, 400.0, 6), stopping_power=SP, on_error="raise"
    )

    np.testing.assert_array_equal(sweep.let_keV_um, np.array(lets))
    np.testing.assert_array_equal(sweep.rbe_alpha(V79.alpha_x), np.array(rbes))
    assert int(np.argmax(sweep.rbe_alpha(V79.alpha_x))) == int(np.argmax(rbes))


def test_elsasser2007_mode_bias_argsort_bit_identical(lem1: LEM1) -> None:
    """tests/benchmark/test_elsasser2007.py -- LETs and alphas, then argsort."""
    energies = np.geomspace(1.5, 350.0, 6)
    lets = np.array([SP.let_keV_um(Ion(Z=6, A=12, energy_MeV_u=float(e))) for e in energies])
    order = np.argsort(lets)
    alpha = np.array(
        [lem1.alpha_beta(Ion(Z=6, A=12, energy_MeV_u=float(e))).alpha for e in energies]
    )
    rbe_ = alpha / V79.alpha_x

    sweep = sweep_energies(
        lem1, 6, 12, energies, stopping_power=SP, on_error="raise", sort_by_let=True
    )

    np.testing.assert_array_equal(sweep.let_keV_um, lets[order])
    np.testing.assert_array_equal(sweep.rbe_alpha(V79.alpha_x), rbe_[order])
    # Rising energy is falling LET: the sort reverses the rows, energies included.
    np.testing.assert_array_equal(sweep.energy_MeV_u, energies[::-1])


def test_lem1_trip98_rbe_alpha_nan_policy_bit_identical(lem1: LEM1) -> None:
    """The original ``lem1_trip98_rbe_reproduction`` script's ``rbe_alpha``: points
    where the track model is undefined (r_max <= r_min below ~0.4 MeV/u, a
    ValueError) become NaN instead of aborting the curve."""
    energies = np.array([0.1, 0.3, 0.5, 2.0, 20.0, 200.0])
    out = np.full(energies.shape, np.nan)
    for i, e in enumerate(energies):
        try:
            out[i] = lem1.initial_slope(Ion(6, 12, float(e))) / lem1.photon.alpha_x
        except (ValueError, ZeroDivisionError):
            pass
    assert np.isnan(out[:2]).all() and np.isfinite(out[2:]).all()

    sweep = sweep_energies(lem1, 6, 12, energies, stopping_power=SP)

    np.testing.assert_array_equal(sweep.rbe_alpha(lem1.photon.alpha_x), out)
    assert np.isnan(sweep.beta[:2]).all()
    np.testing.assert_array_equal(sweep.energy_MeV_u, energies)


def test_lem4_trip98_ceiling_and_extra_error_bit_identical() -> None:
    """The original ``lem4_trip98_rbe_reproduction`` script's ``rbe_alpha_curve``:
    NaN above the route's ceiling (never evaluated) and on any caught error. The
    ceiling is strict (``e > route.e_max``): a point exactly at it is evaluated."""
    energies = np.array([1.0, 2.0, 3.0, 50.0, 60.0, 80.0])
    e_max = 60.0
    errors = (ValueError, ZeroDivisionError, RuntimeError)
    alpha_x = 0.184
    inline_model = _FlakyModel({2.0: RuntimeError, 3.0: ZeroDivisionError})
    out = np.full(energies.shape, np.nan)
    for i, e in enumerate(energies):
        if e > e_max:
            continue
        try:
            out[i] = inline_model.alpha_beta(Ion(6, 12, float(e))).alpha / alpha_x
        except errors:
            pass

    model = _FlakyModel({2.0: RuntimeError, 3.0: ZeroDivisionError})
    sweep = sweep_energies(
        model, 6, 12, energies, stopping_power=SP, errors=errors, e_max_MeV_u=e_max
    )

    np.testing.assert_array_equal(sweep.rbe_alpha(alpha_x), out)
    assert model.calls == inline_model.calls == [1.0, 2.0, 3.0, 50.0, 60.0]
    assert math.isfinite(sweep.alpha[4]) and math.isnan(sweep.alpha[5])
    # The LET axis is still filled above the ceiling.
    assert np.isfinite(sweep.let_keV_um).all()


def test_unconstructible_ion_is_a_nan_row_under_nan_policy() -> None:
    """lem1_trip98 ``rbe_alpha`` / lem4_trip98 ``rbe_alpha_curve`` build the Ion
    inside their ``try``: a grid point the Ion rejects (E <= 0) is a NaN row, and
    the model is never called for it."""
    energies = np.array([0.0, 1.0, 2.0])
    inline_model = _FlakyModel({})
    out = np.full(energies.shape, np.nan)
    for i, e in enumerate(energies):
        try:
            out[i] = inline_model.alpha_beta(Ion(6, 12, float(e))).alpha / 0.184
        except (ValueError, ZeroDivisionError):
            pass

    model = _FlakyModel({})
    sweep = sweep_energies(model, 6, 12, energies, stopping_power=SP)

    np.testing.assert_array_equal(sweep.rbe_alpha(0.184), out)
    assert model.calls == inline_model.calls == [1.0, 2.0]
    assert math.isnan(sweep.let_keV_um[0]) and math.isnan(sweep.beta[0])
    np.testing.assert_array_equal(sweep.energy_MeV_u, energies)
    with pytest.raises(ValueError, match="energy_MeV_u"):
        sweep_energies(model, 6, 12, energies, stopping_power=SP, on_error="raise")


def test_model_comparison_curves_bit_identical(lem1: LEM1) -> None:
    """The original ``model_comparison_alpha_beta`` script's ``curves``: raw alpha, beta."""
    energies = np.geomspace(1.0, 300.0, 5)
    lqs = [lem1.alpha_beta(Ion(6, 12, float(e))) for e in energies]

    sweep = sweep_energies(lem1, 6, 12, energies, stopping_power=SP, on_error="raise")

    np.testing.assert_array_equal(sweep.alpha, np.array([lq.alpha for lq in lqs]))
    np.testing.assert_array_equal(sweep.beta, np.array([lq.beta for lq in lqs]))


# ---- sweep_energies: error policy ---------------------------------------------


def test_on_error_raise_propagates() -> None:
    model = _FlakyModel({2.0: ValueError})
    with pytest.raises(ValueError, match="synthetic"):
        sweep_energies(model, 6, 12, [1.0, 2.0], stopping_power=SP, on_error="raise")


def test_uncaught_error_type_propagates_under_nan_policy() -> None:
    model = _FlakyModel({2.0: RuntimeError})
    with pytest.raises(RuntimeError):
        sweep_energies(model, 6, 12, [1.0, 2.0], stopping_power=SP)


def test_unknown_on_error_is_rejected() -> None:
    with pytest.raises(ValueError, match="on_error"):
        sweep_energies(_FlakyModel({}), 6, 12, [1.0], stopping_power=SP, on_error="skip")  # type: ignore[arg-type]


def test_let_failure_under_nan_policy_is_nan_only_for_let() -> None:
    class _Sp:
        def let_keV_um(self, ion: Ion) -> float:
            if ion.energy_MeV_u < 1.5:
                raise ValueError("below the table")
            return 100.0 / ion.energy_MeV_u

    sweep = sweep_energies(_FlakyModel({}), 6, 12, [1.0, 2.0], stopping_power=_Sp())
    assert math.isnan(sweep.let_keV_um[0]) and sweep.let_keV_um[1] == 50.0
    np.testing.assert_array_equal(sweep.alpha, [1.0, 0.5])


# ---- sweep_lets ----------------------------------------------------------------


def test_lem4_reconstruction_sweeps_bit_identical(lem1: LEM1) -> None:
    """The original ``lem4_reconstruction`` script's ``ion_for_let`` + ``sweep`` +
    ``alpha_beta_sweep``: brentq on [0.4, 600] MeV/u, NaN where the LET is
    unreachable, RBE_10 against the *shouldered* photon curve."""
    let_grid = np.array([20.0, 80.0, 5000.0, 150.0])  # 5000 keV/um: past carbon's peak

    def ion_for_let(z: int, a: int, let_keV_um: float) -> Ion | None:
        try:
            e = brentq(lambda x: SP.let_keV_um(Ion(z, a, x)) - let_keV_um, 0.4, 600.0)
        except ValueError:
            return None
        return Ion(z, a, float(e))

    alpha_x = V79.alpha_x
    d10_x = lem1.photon.dose_for_survival(0.1)
    rbe_alpha = np.full_like(let_grid, np.nan)
    rbe10 = np.full_like(let_grid, np.nan)
    alpha_i = np.full_like(let_grid, np.nan)
    beta_i = np.full_like(let_grid, np.nan)
    energies = np.full_like(let_grid, np.nan)
    for i, let in enumerate(let_grid):
        ion = ion_for_let(6, 12, float(let))
        if ion is None:
            continue
        lq = lem1.alpha_beta(ion)
        rbe_alpha[i] = lq.alpha / alpha_x
        rbe10[i] = d10_x / dose_for_survival(0.1, lq.alpha, lq.beta)
        alpha_i[i] = lq.alpha
        beta_i[i] = lq.beta
        energies[i] = ion.energy_MeV_u
    assert np.isnan(rbe10[2]) and np.isfinite(rbe10[[0, 1, 3]]).all()

    sweep = sweep_lets(
        lem1,
        6,
        12,
        let_grid,
        stopping_power=SP,
        energy_bounds_MeV_u=(0.4, 600.0),
        on_error="nan",
    )

    np.testing.assert_array_equal(sweep.energy_MeV_u, energies)
    np.testing.assert_array_equal(sweep.alpha, alpha_i)
    np.testing.assert_array_equal(sweep.beta, beta_i)
    np.testing.assert_array_equal(sweep.rbe_alpha(alpha_x), rbe_alpha)
    np.testing.assert_array_equal(sweep.rbe10_shouldered(lem1.photon), rbe10)


def test_lem4_unit_test_helpers_bit_identical(lem1: LEM1) -> None:
    """tests/models/lem/test_lem4.py -- ``_carbon_energy_for_let`` on
    [1.6, 600] MeV/u, ``_rbe_alpha_curve`` (LET-sorted pairs with the LET
    recomputed at the found energy) and ``_rbe10`` (plain-LQ photon D10)."""
    carbon_lets = [120.0, 16.0, 320.0, 45.0]  # not monotone: row order is checked
    ions = [
        Ion(6, 12, float(brentq(lambda e, let=let: SP.let_keV_um(Ion(6, 12, e)) - let, 1.6, 600.0)))
        for let in carbon_lets
    ]
    pairs = [(SP.let_keV_um(ion), lem1.alpha_beta(ion).alpha / lem1.photon.alpha_x) for ion in ions]
    pairs.sort()
    d10_x = dose_for_survival(0.1, lem1.photon.alpha_x, lem1.photon.beta_x)
    rbe10_ref = []
    for ion in ions:
        lq = lem1.alpha_beta(ion)
        rbe10_ref.append(d10_x / dose_for_survival(0.1, lq.alpha, lq.beta))

    sweep = sweep_lets(
        lem1,
        6,
        12,
        carbon_lets,
        stopping_power=SP,
        energy_bounds_MeV_u=(1.6, 600.0),
        on_error="raise",
    )

    # Rows stay in the order of ``lets``; the adopter sorts by LET itself.
    np.testing.assert_array_equal(sweep.energy_MeV_u, [ion.energy_MeV_u for ion in ions])
    np.testing.assert_array_equal(
        sweep.rbe10_lq(lem1.photon.alpha_x, lem1.photon.beta_x), rbe10_ref
    )
    order = np.argsort(sweep.let_keV_um)
    np.testing.assert_array_equal(sweep.let_keV_um[order], [let for let, _ in pairs])
    np.testing.assert_array_equal(
        sweep.rbe_alpha(lem1.photon.alpha_x)[order], [r for _, r in pairs]
    )


def test_sweep_lets_forwards_xtol_like_imk() -> None:
    """The original ``imk_microdosimetry_routes`` script's ``carbon_at_let``: xtol=1e-8."""
    model = _FlakyModel({})
    lets = [13.0, 70.0]
    expected = [
        brentq(
            lambda e, let=let: SP.let_keV_um(Ion(Z=6, A=12, energy_MeV_u=e)) - let,
            0.6,
            600.0,
            xtol=1e-8,
        )
        for let in lets
    ]
    default_tol = [
        brentq(lambda e, let=let: SP.let_keV_um(Ion(6, 12, e)) - let, 0.6, 600.0) for let in lets
    ]
    assert expected != default_tol  # the keyword is observable on these LETs

    sweep = sweep_lets(
        model,
        6,
        12,
        lets,
        stopping_power=SP,
        energy_bounds_MeV_u=(0.6, 600.0),
        xtol=1e-8,
        on_error="raise",
    )
    plain = sweep_lets(model, 6, 12, lets, stopping_power=SP, energy_bounds_MeV_u=(0.6, 600.0))

    np.testing.assert_array_equal(sweep.energy_MeV_u, expected)
    np.testing.assert_array_equal(plain.energy_MeV_u, default_tol)


def test_sweep_lets_model_errors_propagate() -> None:
    """Only the LET inversion maps to NaN; a model error is the caller's to see."""
    model = _FlakyModel({})
    energy = brentq(lambda e: SP.let_keV_um(Ion(6, 12, e)) - 50.0, 0.4, 600.0)
    model.failures[energy] = ValueError
    with pytest.raises(ValueError, match="synthetic"):
        sweep_lets(model, 6, 12, [50.0], stopping_power=SP, energy_bounds_MeV_u=(0.4, 600.0))


def test_sweep_lets_default_policy_raises_on_unreachable_let() -> None:
    """The default is ``on_error="raise"`` -- the ``branch="bracket"`` contract and
    three of the four source sites; only lem4_reconstruction opts into NaN rows."""
    model = _FlakyModel({})
    with pytest.raises(ValueError):
        sweep_lets(model, 6, 12, [5000.0], stopping_power=SP, energy_bounds_MeV_u=(0.4, 600.0))
    assert model.calls == []


def test_sweep_lets_unreachable_let_raises_under_raise_policy() -> None:
    """lem3 ``carbon_energy_for_let``, imk ``carbon_at_let`` and test_lem4
    ``_carbon_energy_for_let`` call brentq bare: no sign change is an error."""
    with pytest.raises(ValueError):
        sweep_lets(
            _FlakyModel({}),
            6,
            12,
            [50.0, 5000.0],
            stopping_power=SP,
            energy_bounds_MeV_u=(0.4, 600.0),
            on_error="raise",
        )


@pytest.mark.parametrize(
    "bounds", [(0.0, 600.0), (600.0, 0.4), (5.0, 5.0), (-1.0, 10.0), (0.4, math.inf)]
)
@pytest.mark.parametrize("on_error", ["nan", "raise"])
def test_sweep_lets_bad_bracket_raises_under_either_policy(
    bounds: tuple[float, float], on_error: str
) -> None:
    """A misconfigured bracket is checked once, before the rows: it never becomes
    an all-NaN sweep."""
    model = _FlakyModel({})
    with pytest.raises(ValueError, match="energy_bounds_MeV_u"):
        sweep_lets(
            model,
            6,
            12,
            [50.0],
            stopping_power=SP,
            energy_bounds_MeV_u=bounds,
            on_error=on_error,  # type: ignore[arg-type]
        )
    assert model.calls == []


def test_sweep_lets_unknown_on_error_is_rejected() -> None:
    with pytest.raises(ValueError, match="on_error"):
        sweep_lets(
            _FlakyModel({}),
            6,
            12,
            [50.0],
            stopping_power=SP,
            energy_bounds_MeV_u=(0.4, 600.0),
            on_error="skip",  # type: ignore[arg-type]
        )


@pytest.mark.parametrize("grid", [[[1.0, 2.0]], [[1.0], [2.0]]])
def test_sweeps_reject_a_2d_grid(grid: list[list[float]]) -> None:
    with pytest.raises(ValueError, match="energies must be 1-D"):
        sweep_energies(_FlakyModel({}), 6, 12, grid, stopping_power=SP)
    with pytest.raises(ValueError, match="lets must be 1-D"):
        sweep_lets(
            _FlakyModel({}), 6, 12, grid, stopping_power=SP, energy_bounds_MeV_u=(0.4, 600.0)
        )


# ---- LetSweep ------------------------------------------------------------------


@pytest.mark.parametrize(
    "columns",
    [
        ([1.0, 2.0], [1.0, 2.0], [1.0, 2.0], [1.0]),  # unequal length
        ([[1.0, 2.0]], [[1.0, 2.0]], [[1.0, 2.0]], [[1.0, 2.0]]),  # 2-D, equal shape
        (1.0, 1.0, 1.0, 1.0),  # 0-D
    ],
)
def test_letsweep_rejects_non_1d_or_unequal_columns(columns: tuple[object, ...]) -> None:
    with pytest.raises(ValueError, match="1-D and equally long"):
        LetSweep(*(np.asarray(c, dtype=float) for c in columns))


def test_letsweep_is_frozen_and_rbe_methods_keep_nan_rows() -> None:
    sweep = LetSweep(
        energy_MeV_u=np.array([10.0, np.nan]),
        let_keV_um=np.array([100.0, np.nan]),
        alpha=np.array([0.5, np.nan]),
        beta=np.array([0.05, np.nan]),
    )
    with pytest.raises(dataclasses.FrozenInstanceError):
        sweep.alpha = np.array([1.0])  # type: ignore[misc]
    photon = PhotonParameters(alpha_x=0.1, beta_x=0.05, D_t=4.0)
    for values in (
        sweep.rbe_alpha(0.1),
        sweep.rbe10_lq(0.1, 0.05),
        sweep.rbe10_shouldered(photon),
    ):
        assert math.isfinite(values[0]) and math.isnan(values[1])
    assert sweep.rbe_alpha(0.1)[0] == 0.5 / 0.1
    assert sweep.rbe10_lq(0.1, 0.05)[0] == rbe(
        0.10, 0.5, 0.05, reference=LQParameters(alpha=0.1, beta=0.05)
    )
    assert sweep.rbe10_shouldered(photon)[0] == photon.dose_for_survival(0.1) / dose_for_survival(
        0.1, 0.5, 0.05
    )
