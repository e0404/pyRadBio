"""Benchmark harness: aggregation logic with a fake model (no PIDE, no LEM)."""

from __future__ import annotations

import math
import warnings

import pytest

from pyradbio.benchmark.harness import (
    BenchmarkResult,
    ModelSpec,
    PointResult,
    run_benchmark,
)
from pyradbio.benchmark.pide import PideEntry
from pyradbio.core.lq import LQParameters
from pyradbio.core.radiation import Ion
from pyradbio.models.base import BiophysicalModel, LETModel


class _ScaleModel(BiophysicalModel):
    """Predicts alpha = scale * energy; ignores photon (for deterministic tests)."""

    def __init__(self, scale: float) -> None:
        self.scale = scale

    def _alpha_beta_ion(self, ion: Ion) -> LQParameters:
        return LQParameters(alpha=self.scale * ion.energy_MeV_u, beta=0.05)


def _entry(**kw: object) -> PideEntry:
    base: dict[str, object] = dict(
        cell_line="X",
        Z=6,
        A=12,
        energy_MeV_u=10.0,
        let_keV_um=30.0,
        alpha_ion=1.0,
        beta_ion=0.04,
        alpha_x=0.2,
        beta_x=0.05,
    )
    base.update(kw)
    return PideEntry(**base)  # type: ignore[arg-type]


def test_run_benchmark_evaluates_and_predicts() -> None:
    # alpha_pred = 0.1 * energy; energy 10 -> 1.0, exactly the measured alpha_ion.
    entries = [_entry(energy_MeV_u=10.0, alpha_ion=1.0)]
    result = run_benchmark(entries, factory=lambda ax, bx: _ScaleModel(0.1))
    assert result.n_points == 1
    p = result.points[0]
    assert p.predicted_alpha == pytest.approx(1.0)
    assert result.mean_abs_log_error_alpha == pytest.approx(0.0)
    assert result.smape_alpha == pytest.approx(0.0)


def test_run_benchmark_skips_uncovered_points() -> None:
    entries = [
        _entry(energy_MeV_u=None),  # no energy
        _entry(Z=26, A=56),  # heavy ion past max_z
        _entry(energy_MeV_u=0.1),  # below min energy
        _entry(alpha_ion=0.0),  # non-positive measured alpha
        _entry(energy_MeV_u=10.0, alpha_ion=1.0),  # the only good one
    ]
    result = run_benchmark(entries, factory=lambda ax, bx: _ScaleModel(0.1))
    assert result.n_points == 1
    assert result.n_skipped == 4


def test_run_benchmark_limit_caps_points() -> None:
    entries = [_entry(alpha_ion=1.0) for _ in range(10)]
    result = run_benchmark(entries, factory=lambda ax, bx: _ScaleModel(0.1), limit=3)
    assert result.n_points == 3


def test_run_benchmark_skips_model_errors() -> None:
    class _Boom(BiophysicalModel):
        def _alpha_beta_ion(self, ion: Ion) -> LQParameters:
            raise ValueError("no stopping power here")

    result = run_benchmark([_entry()], factory=lambda ax, bx: _Boom())
    assert result.n_points == 0
    assert result.n_skipped == 1


def test_point_result_rbe_and_residual() -> None:
    p = PointResult(
        cell_line="V79",
        Z=6,
        A=12,
        energy_MeV_u=100.0,
        let_keV_um=30.0,
        alpha_x=0.2,
        predicted_alpha=0.4,
        measured_alpha=0.5,
    )
    assert p.predicted_rbe_alpha == pytest.approx(2.0)
    assert p.measured_rbe_alpha == pytest.approx(2.5)
    assert p.log_residual_alpha == pytest.approx(math.log10(0.4) - math.log10(0.5))


def test_aggregate_metrics_match_metric_functions() -> None:
    entries = [
        _entry(energy_MeV_u=10.0, alpha_ion=1.0),  # pred 1.0
        _entry(energy_MeV_u=20.0, alpha_ion=1.0),  # pred 2.0
    ]
    result: BenchmarkResult = run_benchmark(entries, factory=lambda ax, bx: _ScaleModel(0.1))
    # mean log error = mean(log10(1/1), log10(2/1)) = 0.5 * log10(2)
    assert result.mean_log_error_alpha == pytest.approx(0.5 * math.log10(2.0))
    grouped = result.by_ion()
    assert set(grouped) == {6}
    assert len(grouped[6]) == 2


# --- LET-based models, model scope and warning capture ---------------------


class _LetModel(LETModel):
    """A LET-parameterised model: alpha = scale * LET, no stopping power needed.

    Stands in for the phenomenological family, whose natural entry point is
    ``alpha_beta_from_let`` -- ``alpha_beta(Ion)`` raises without a stopping power.
    """

    def __init__(self, scale: float, *, warn: bool = False) -> None:
        self.scale = scale
        self.warn = warn

    def _alpha_beta_ion(self, ion: Ion) -> LQParameters:
        raise ValueError("needs a stopping_power")

    def alpha_beta_from_let(self, let_keV_um: float) -> LQParameters:
        if self.warn:
            warnings.warn("outside the fitted range", UserWarning, stacklevel=2)
        return LQParameters(alpha=self.scale * let_keV_um, beta=0.05)


def test_let_based_spec_evaluates_from_the_entry_let() -> None:
    # The ion route raises, so a non-zero score proves the LET route was taken.
    spec = ModelSpec(build=lambda ax, bx: _LetModel(0.01), model_type=_LetModel)
    result = run_benchmark([_entry(let_keV_um=30.0)], spec=spec)
    assert result.n_points == 1
    assert result.points[0].predicted_alpha == pytest.approx(0.3)


def test_ion_based_model_cannot_reach_the_let_route() -> None:
    spec = ModelSpec(build=lambda ax, bx: _LetModel(0.01))
    result = run_benchmark([_entry(let_keV_um=30.0)], spec=spec)
    assert result.n_points == 0
    assert result.n_skipped == 1


def test_applicable_z_is_out_of_scope_not_skipped() -> None:
    """A proton-only fit meeting a carbon point is a scope limit, not a failure."""
    spec = ModelSpec(
        build=lambda ax, bx: _LetModel(0.01), model_type=_LetModel, applicable_z=frozenset({1})
    )
    entries = [_entry(Z=6, A=12), _entry(Z=1, A=1), _entry(Z=2, A=4)]
    result = run_benchmark(entries, spec=spec)
    assert result.n_points == 1
    assert result.n_out_of_scope == 2
    assert result.n_skipped == 0


def test_needs_beta_x_is_out_of_scope() -> None:
    """(alpha/beta)_x is the tissue axis: a point without beta_x has no scope."""
    spec = ModelSpec(build=lambda ax, bx: _LetModel(0.01), model_type=_LetModel, needs_beta_x=True)
    result = run_benchmark([_entry(beta_x=0.0), _entry(beta_x=0.05)], spec=spec)
    assert result.n_points == 1
    assert result.n_out_of_scope == 1


def test_model_warnings_are_counted_and_not_re_emitted() -> None:
    """One out-of-domain warning per point would drown a full-PIDE run."""
    spec = ModelSpec(build=lambda ax, bx: _LetModel(0.01, warn=True), model_type=_LetModel)
    with warnings.catch_warnings():
        warnings.simplefilter("error")  # any escaping warning fails the test
        result = run_benchmark([_entry(), _entry(), _entry()], spec=spec)
    assert result.n_points == 3
    assert result.n_warnings == 3


def test_factory_argument_still_means_the_ion_route() -> None:
    """Back-compat: a bare ``factory=`` is an ion-based model with no scope limit."""
    result = run_benchmark([_entry(energy_MeV_u=10.0)], factory=lambda ax, bx: _ScaleModel(0.1))
    assert result.n_points == 1
    assert result.n_out_of_scope == 0
    assert result.n_warnings == 0


def test_cache_by_ion_evaluates_each_distinct_ion_once() -> None:
    """PIDE repeats (Z, A, E); a spectrum MKM run is ~40 ms a point without this."""
    calls: list[float] = []

    class _Counting(BiophysicalModel):
        def _alpha_beta_ion(self, ion: Ion) -> LQParameters:
            calls.append(ion.energy_MeV_u)
            return LQParameters(alpha=0.1 * ion.energy_MeV_u, beta=0.05)

    model = _Counting()
    spec = ModelSpec(build=lambda ax, bx: model, cache_by_ion=True)
    entries = [_entry(energy_MeV_u=10.0), _entry(energy_MeV_u=10.0), _entry(energy_MeV_u=20.0)]
    result = run_benchmark(entries, spec=spec)
    assert result.n_points == 3
    assert calls == [10.0, 20.0]
    assert [p.predicted_alpha for p in result.points] == pytest.approx([1.0, 1.0, 2.0])


def test_cache_by_ion_is_keyed_on_the_photon_reference_too() -> None:
    """A per-photon-pair model must not serve another pair's cached prediction."""

    class _PhotonScaled(BiophysicalModel):
        def __init__(self, alpha_x: float) -> None:
            self.alpha_x = alpha_x

        def _alpha_beta_ion(self, ion: Ion) -> LQParameters:
            return LQParameters(alpha=self.alpha_x * ion.energy_MeV_u, beta=0.05)

    spec = ModelSpec(build=lambda ax, bx: _PhotonScaled(ax), cache_by_ion=True)
    entries = [_entry(alpha_x=0.2), _entry(alpha_x=0.4)]
    result = run_benchmark(entries, spec=spec)
    assert [p.predicted_alpha for p in result.points] == pytest.approx([2.0, 4.0])


def test_non_positive_prediction_is_skipped() -> None:
    """A model can return alpha <= 0 in a regime it does not cover; log10 cannot."""

    class _Zero(BiophysicalModel):
        def _alpha_beta_ion(self, ion: Ion) -> LQParameters:
            return LQParameters(alpha=0.0, beta=0.05)

    result = run_benchmark([_entry()], factory=lambda ax, bx: _Zero())
    assert result.n_points == 0
    assert result.n_skipped == 1


def test_only_model_warnings_are_tallied() -> None:
    """pyMKM emits a scipy DeprecationWarning per call; that is not a domain warning."""

    class _Noisy(BiophysicalModel):
        def _alpha_beta_ion(self, ion: Ion) -> LQParameters:
            warnings.warn("`trapz` is deprecated", DeprecationWarning, stacklevel=2)
            warnings.warn("outside the fitted range", UserWarning, stacklevel=2)
            return LQParameters(alpha=0.5, beta=0.05)

    result = run_benchmark([_entry(), _entry()], factory=lambda ax, bx: _Noisy())
    assert result.n_points == 2
    assert result.n_warnings == 2  # the two UserWarnings, not the four warnings


def test_run_benchmark_on_a_dataset_carries_its_name_and_caveats() -> None:
    from pyradbio.benchmark.dataset import ReferenceDataset

    ds = ReferenceDataset("Mine", (_entry(energy_MeV_u=10.0, alpha_ion=1.0),), caveats=("c1",))
    result = run_benchmark(ds, factory=lambda ax, bx: _ScaleModel(0.1))
    assert result.n_points == 1
    assert result.dataset_name == "Mine"
    assert result.dataset_caveats == ("c1",)
    bare = run_benchmark(ds.records, factory=lambda ax, bx: _ScaleModel(0.1))
    assert bare.dataset_name == "PIDE" and bare.dataset_caveats == ()


@pytest.mark.parametrize("wrapped", [False, True])
def test_nominal_energy_is_rejected_before_building_an_ion_model(wrapped: bool) -> None:
    from pyradbio.benchmark.dataset import ReferenceDataset

    records = (_entry(energy_is_nominal=True),)
    entries = ReferenceDataset("Nominal", records) if wrapped else records

    def unexpected_build(ax: float, bx: float) -> BiophysicalModel:
        pytest.fail("A nominal beam energy must never reach an ion model")

    with pytest.raises(ValueError, match="nominal beam energ"):
        run_benchmark(entries, factory=unexpected_build)


def test_nominal_energy_can_be_used_as_metadata_for_a_let_model() -> None:
    spec = ModelSpec(build=lambda ax, bx: _LetModel(0.01), model_type=_LetModel)
    result = run_benchmark([_entry(energy_is_nominal=True, let_keV_um=30.0)], spec=spec)
    assert result.n_points == 1
    assert result.points[0].predicted_alpha == pytest.approx(0.3)


def test_let_route_waives_the_energy_gate_only_for_nominal_energies() -> None:
    """On a nominal-energy record a LET-based model needs no energy at all."""
    spec = ModelSpec(build=lambda ax, bx: _LetModel(0.01), model_type=_LetModel)
    nominal = [
        _entry(energy_is_nominal=True, energy_MeV_u=None, let_keV_um=30.0),
        _entry(energy_is_nominal=True, energy_MeV_u=0.1, let_keV_um=30.0),  # below the floor
    ]
    result = run_benchmark(nominal, spec=spec)
    assert result.n_points == 2 and result.n_skipped == 0
    assert all(p.energy_MeV_u is None for p in result.points)
    # A verified (target) energy keeps gating the LET route, so PIDE rankings
    # across routes stay like-for-like.
    verified = [
        _entry(energy_MeV_u=None, let_keV_um=30.0),
        _entry(energy_MeV_u=0.1, let_keV_um=30.0),
        _entry(energy_MeV_u=10.0, let_keV_um=30.0),
    ]
    result = run_benchmark(verified, spec=spec)
    assert result.n_points == 1 and result.n_skipped == 2
    assert result.points[0].energy_MeV_u == pytest.approx(10.0)
