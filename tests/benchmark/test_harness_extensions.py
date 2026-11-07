"""Harness extensions for the PIDE validation script.

* ``build_model_spec(..., D_t=rule)``: a per-photon-pair LEM transition dose
  (Pfuhl et al. 2022 eq. 3 is the use case);
* ``PointResult.beta_x`` / ``predicted_beta`` / ``measured_beta``;
* ``run_benchmark(limit_per_ion=..., stratify="let")`` and ``stratified_by_let``,
  which replace the ``subsample()`` of the original PIDE validation script.
"""

from __future__ import annotations

import numpy as np
import pytest

from pyradbio.benchmark import harness
from pyradbio.benchmark.harness import (
    ModelSpec,
    PointResult,
    build_model_factory,
    build_model_spec,
    run_benchmark,
    stratified_by_let,
)
from pyradbio.benchmark.pide import PideEntry
from pyradbio.core.lq import LQParameters
from pyradbio.core.radiation import Ion
from pyradbio.models.base import BiophysicalModel


class _LetModel(BiophysicalModel):
    """alpha = 0.01 * energy, beta = 0.02; fails on a chosen energy."""

    def __init__(self, fail_at: float | None = None) -> None:
        self.fail_at = fail_at

    def _alpha_beta_ion(self, ion: Ion) -> LQParameters:
        if ion.energy_MeV_u == self.fail_at:
            raise ValueError("outside coverage")
        return LQParameters(alpha=0.01 * ion.energy_MeV_u, beta=0.02)


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


def _old_subsample(items: list[PideEntry], cap: int) -> list[PideEntry]:
    """``subsample`` of the original PIDE validation script, verbatim."""
    if len(items) <= cap:
        return items
    order = sorted(range(len(items)), key=lambda i: items[i].let_keV_um)
    picks = np.linspace(0, len(items) - 1, cap).round().astype(int)
    return [items[order[p]] for p in sorted(set(picks))]


# --- callable D_t -------------------------------------------------------------


def test_callable_transition_dose_is_evaluated_per_photon_pair() -> None:
    calls: list[tuple[float, float]] = []

    def rule(alpha_x: float, beta_x: float) -> float:
        calls.append((alpha_x, beta_x))
        return 4.0 + 1.1 * alpha_x / beta_x

    build = build_model_spec("lem1", D_t=rule, nucleus_radius_um=5.0).build
    m1 = build(0.2, 0.05)
    assert m1.photon.D_t == pytest.approx(4.0 + 1.1 * 0.2 / 0.05)  # type: ignore[attr-defined]
    assert m1.cell.nucleus_radius_um == 5.0  # type: ignore[attr-defined]
    assert build(0.2, 0.05) is m1  # cached on the pair: the rule ran once
    m2 = build(0.3, 0.03)
    assert m2.photon.D_t == pytest.approx(15.0)  # type: ignore[attr-defined]
    assert calls == [(0.2, 0.05), (0.3, 0.03)]


def test_float_transition_dose_is_unchanged() -> None:
    build = build_model_spec("lem4", D_t=12.5).build
    assert build(0.2, 0.05).photon.D_t == 12.5  # type: ignore[attr-defined]
    assert build_model_spec("lem4").build(0.2, 0.05).photon.D_t == 30.0  # type: ignore[attr-defined]


def test_transition_dose_rule_is_refused_by_preset_models() -> None:
    with pytest.raises(ValueError, match="D_t"):
        build_model_spec("smk", D_t=lambda ax, bx: 10.0)


def test_transition_dose_rule_passes_through_the_factory_shorthands(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    def rule(alpha_x: float, beta_x: float) -> float:
        return 4.0 + 1.1 * alpha_x / beta_x

    model = build_model_factory("lem1", D_t=rule, nucleus_radius_um=5.0)(0.2, 0.05)
    assert model.photon.D_t == pytest.approx(8.4)  # type: ignore[attr-defined]

    seen: dict[str, object] = {}

    def fake_spec(model_name: str, **kwargs: object) -> ModelSpec:
        seen.update(kwargs, model_name=model_name)
        return ModelSpec(build=lambda ax, bx: _LetModel())

    monkeypatch.setattr(harness, "build_model_spec", fake_spec)
    run_benchmark([_entry()], "lem4", D_t=rule, nucleus_radius_um=5.0)
    assert seen == {"model_name": "lem4", "D_t": rule, "nucleus_radius_um": 5.0}


# --- beta fields --------------------------------------------------------------


def test_point_results_carry_the_quadratic_terms() -> None:
    result = run_benchmark([_entry(beta_ion=0.07, beta_x=0.04)], factory=lambda ax, bx: _LetModel())
    (p,) = result.points
    assert p.beta_x == 0.04
    assert p.measured_beta == 0.07
    assert p.predicted_beta == 0.02


def test_hand_built_point_results_need_no_beta() -> None:
    p = PointResult(
        cell_line="X",
        Z=6,
        A=12,
        energy_MeV_u=10.0,
        let_keV_um=30.0,
        alpha_x=0.2,
        predicted_alpha=0.4,
        measured_alpha=0.5,
    )
    assert p.beta_x is None and p.predicted_beta is None and p.measured_beta is None


# --- per-ion stratified thinning ----------------------------------------------


def _many(z: int, a: int, n: int, *, seed: int) -> list[PideEntry]:
    rng = np.random.default_rng(seed)
    lets = rng.uniform(1.0, 500.0, n).round(1)  # rounding makes LET ties likely
    energies = rng.uniform(1.0, 300.0, n).round(3)
    return [
        _entry(Z=z, A=a, let_keV_um=float(let), energy_MeV_u=float(e), cell_line=f"c{i}")
        for i, (let, e) in enumerate(zip(lets, energies, strict=True))
    ]


@pytest.mark.parametrize(("n", "cap"), [(97, 40), (40, 40), (12, 40), (41, 40), (300, 7), (5, 1)])
def test_stratified_by_let_matches_the_old_script(n: int, cap: int) -> None:
    items = _many(6, 12, n, seed=n * 31 + cap)
    ours = stratified_by_let(items, cap, let_keV_um=lambda e: e.let_keV_um)
    old = _old_subsample(items, cap)
    if n > cap:
        assert ours == old  # same members, same (LET-sorted) order
    else:
        # the old script kept input order when nothing was dropped; the harness
        # always returns LET order (same members)
        assert ours == sorted(old, key=lambda e: e.let_keV_um)
    lets = [e.let_keV_um for e in ours]
    assert lets == sorted(lets)
    assert len(ours) <= cap


def test_stratified_by_let_spans_the_range_and_rejects_bad_caps() -> None:
    items = _many(1, 1, 50, seed=3)
    picked = stratified_by_let(items, 5, let_keV_um=lambda e: e.let_keV_um)
    all_lets = sorted(e.let_keV_um for e in items)
    assert picked[0].let_keV_um == all_lets[0]
    assert picked[-1].let_keV_um == all_lets[-1]
    with pytest.raises(ValueError):
        stratified_by_let(items, 0, let_keV_um=lambda e: e.let_keV_um)


def test_limit_per_ion_thins_each_ion_and_orders_by_z_then_let() -> None:
    carbon = _many(6, 12, 60, seed=1)
    helium = _many(2, 4, 25, seed=2)
    proton = _many(1, 1, 8, seed=4)
    entries = [*carbon[:30], *helium, *carbon[30:], *proton]
    result = run_benchmark(entries, factory=lambda ax, bx: _LetModel(), limit_per_ion=20)
    expected = [
        *sorted(proton, key=lambda e: e.let_keV_um),
        *_old_subsample(helium, 20),
        *_old_subsample(carbon, 20),
    ]
    assert [(p.Z, p.cell_line) for p in result.points] == [(e.Z, e.cell_line) for e in expected]
    assert result.attempted_by_ion == ((1, 8), (2, 20), (6, 20))
    assert result.n_skipped == 0  # thinned points are not skipped
    assert result.n_out_of_scope == 0


def test_limit_per_ion_thins_after_the_filters() -> None:
    """Records the harness would skip never take a slot of the per-ion cap."""
    good = _many(6, 12, 10, seed=5)
    bad = [_entry(energy_MeV_u=0.1, let_keV_um=float(i)) for i in range(10)]
    result = run_benchmark([*bad, *good], factory=lambda ax, bx: _LetModel(), limit_per_ion=4)
    assert result.n_skipped == 10
    assert result.attempted_by_ion == ((6, 4),)
    assert [p.cell_line for p in result.points] == [e.cell_line for e in _old_subsample(good, 4)]


def test_attempted_counts_include_model_failures() -> None:
    entries = _many(6, 12, 10, seed=6)
    fail_at = entries[3].energy_MeV_u
    result = run_benchmark(
        entries, factory=lambda ax, bx: _LetModel(fail_at=fail_at), limit_per_ion=50
    )
    assert result.attempted_by_ion == ((6, 10),)
    assert result.n_points == 9
    assert result.n_skipped == 1


def test_without_limit_per_ion_the_input_order_is_kept() -> None:
    entries = _many(6, 12, 12, seed=7)
    result = run_benchmark(entries, factory=lambda ax, bx: _LetModel())
    assert [p.cell_line for p in result.points] == [e.cell_line for e in entries]
    assert result.attempted_by_ion == ((6, 12),)


def test_global_limit_still_stops_the_tally() -> None:
    """Records after the last evaluated point are neither evaluated nor tallied."""
    entries = [_entry(), _entry(), _entry(energy_MeV_u=0.1), _entry()]
    result = run_benchmark(entries, factory=lambda ax, bx: _LetModel(), limit=2)
    assert result.n_points == 2
    assert result.n_skipped == 0


def test_bad_thinning_arguments_are_refused() -> None:
    with pytest.raises(ValueError, match="stratify"):
        run_benchmark([_entry()], factory=lambda ax, bx: _LetModel(), stratify="random")  # type: ignore[arg-type]
    with pytest.raises(ValueError, match="limit_per_ion"):
        run_benchmark([_entry()], factory=lambda ax, bx: _LetModel(), limit_per_ion=0)
