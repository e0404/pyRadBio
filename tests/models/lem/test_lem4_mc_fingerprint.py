"""Numeric fingerprint of the committed LEM IV explicit-track MC.

``validation/data/lem4_friedrich2012_mc.csv`` holds hours of Monte Carlo
(``scripts/lem4_friedrich2012_mc.py``, Friedrich et al. 2012,
doi:10.3109/09553002.2011.611213, Fig. 4). Its header records one small NumPy run
at a fixed seed; the fast test here recomputes it, so a change to the code that
produced the CSV fails CI with "regenerate with scripts/lem4_friedrich2012_mc.py"
instead of silently invalidating the overlay. It skips until the CSV exists.

The remaining tests exercise the generator and the fingerprint machinery on tiny
runs: fast ones with the Monte Carlo stubbed out, and one ``@slow`` real run.
"""

from __future__ import annotations

import importlib.util
import math
import sys
from pathlib import Path
from types import ModuleType
from typing import Any

import numpy as np
import numpy.typing as npt
import pytest

from pyradbio.benchmark import friedrich2012 as f12
from pyradbio.benchmark.checkpoint import FINGERPRINT_KEY, assert_fingerprint, read_header
from pyradbio.track.stopping_power import default_stopping_power

REPO = Path(__file__).resolve().parents[3]
CSV = REPO / "validation" / "data" / f12.FIG4_MC_CSV_NAME
GENERATOR = REPO / "scripts" / f"{f12.FIG4_MC_GENERATOR}.py"
NOT_GENERATED = (
    "validation/data/lem4_friedrich2012_mc.csv not generated yet (scripts/lem4_friedrich2012_mc.py)"
)


def _recompute(recorded: Any) -> list[dict[str, Any]]:
    """Recompute the fingerprint at the points the artefact recorded."""
    points = [(p["let_keV_um"], p["n_realizations"], p["seed"]) for p in recorded or []]
    return f12.fig4_mc_fingerprint(points)


def test_committed_lem4_mc_matches_its_fingerprint() -> None:
    if not CSV.exists():
        pytest.skip(NOT_GENERATED)
    recorded = read_header(CSV).get(FINGERPRINT_KEY)
    assert_fingerprint(recorded, _recompute(recorded), generator=f12.FIG4_MC_GENERATOR)


def test_committed_lem4_mc_covers_the_production_grid() -> None:
    if not CSV.exists():
        pytest.skip(NOT_GENERATED)
    mc = f12.load_fig4_mc(CSV)
    assert tuple(mc.let_keV_um) == f12.FIG4_MC_LETS
    assert mc.incomplete_lets == ()
    assert set(mc.n_realizations) == {f12.FIG4_MC_REALIZATIONS}
    assert all(seeds == tuple(range(f12.FIG4_MC_SEEDS)) for seeds in mc.seeds)
    assert np.all(np.isfinite(mc.alpha)) and np.all(mc.alpha_err > 0.0)


# ---- the generator and the fingerprint machinery on tiny runs -------------------


def _load_generator(monkeypatch: pytest.MonkeyPatch) -> ModuleType:
    name = "_lem4_friedrich2012_mc_under_test"
    spec = importlib.util.spec_from_file_location(name, GENERATOR)
    assert spec is not None and spec.loader is not None
    module = importlib.util.module_from_spec(spec)
    monkeypatch.setitem(sys.modules, name, module)
    spec.loader.exec_module(module)
    return module


class _StubModel:
    """Stands in for ``fig4_mc_model``: deterministic (alpha, beta) per (ion, seed)."""

    def __init__(self) -> None:
        self.stopping_power = default_stopping_power()
        self.calls: list[tuple[float, int]] = []

    def fit_doses_Gy(self) -> npt.NDArray[np.float64]:
        return np.array([0.475, 1.0, 8.075])

    def local_dose_mc_alpha_beta(
        self, ion: Any, *, doses_Gy: Any, n_realizations: int, seed: int
    ) -> tuple[float, float]:
        self.calls.append((ion.energy_MeV_u, seed))
        return 0.01 * ion.energy_MeV_u + 0.1 * seed, 0.05 - 0.003 * seed + 1e-17 * n_realizations


@pytest.fixture
def stubbed(monkeypatch: pytest.MonkeyPatch) -> dict[str, Any]:
    state: dict[str, Any] = {"model": _StubModel(), "fingerprint": [{"alpha": 1.0, "beta": 0.5}]}
    monkeypatch.setattr(f12, "fig4_mc_model", lambda **kw: state["model"])
    monkeypatch.setattr(f12, "fig4_mc_fingerprint", lambda *a, **kw: state["fingerprint"])
    return state


def test_generator_writes_header_rows_and_resumes(
    stubbed: dict[str, Any],
    monkeypatch: pytest.MonkeyPatch,
    tmp_path: Path,
    capsys: pytest.CaptureFixture[str],
) -> None:
    gen = _load_generator(monkeypatch)
    argv = ["--lets", "100,55", "--realizations", "3", "--seeds", "2", "--engine", "numpy"]
    assert gen.main([*argv, "--out", str(tmp_path)]) == 0
    out = capsys.readouterr().out
    path = tmp_path / f12.FIG4_MC_CSV_NAME
    assert sorted(p.name for p in tmp_path.iterdir()) == [path.name]  # no lock file left

    header = read_header(path)
    assert header[FINGERPRINT_KEY] == stubbed["fingerprint"]
    assert header["n_seeds"] == "2" and header["n_realizations"] == "3"
    assert header["generator"] == "scripts/lem4_friedrich2012_mc.py"
    mc = f12.load_fig4_mc(path)
    assert tuple(mc.let_keV_um) == (55.0, 100.0)
    assert mc.seeds == ((0, 1), (0, 1))
    # The printed per-LET lines are the statistics the loader recomputes from the CSV.
    for k, let in enumerate(mc.let_keV_um):
        line = (
            f"    LET {let:5.0f}: alpha={mc.alpha[k]:.4f}+-{mc.alpha_err[k]:.4f}  "
            f"beta={mc.beta[k]:+.4f}+-{mc.beta_err[k]:.4f}"
        )
        assert line in out

    # Resume: nothing left to compute, the file is unchanged.
    before = path.read_bytes()
    stubbed["model"].calls.clear()
    assert gen.main([*argv, "--out", str(tmp_path)]) == 0
    assert stubbed["model"].calls == []
    assert path.read_bytes() == before


def test_generator_refuses_to_extend_a_file_the_code_no_longer_reproduces(
    stubbed: dict[str, Any], monkeypatch: pytest.MonkeyPatch, tmp_path: Path
) -> None:
    gen = _load_generator(monkeypatch)
    out = ["--out", str(tmp_path)]
    assert gen.main(["--lets", "100", "--realizations", "3", "--seeds", "2", *out]) == 0
    before = (tmp_path / f12.FIG4_MC_CSV_NAME).read_bytes()

    stubbed["fingerprint"] = [{"alpha": 1.0, "beta": 0.5000001}]  # the code changed
    assert gen.main(["--lets", "100,55", "--realizations", "3", "--seeds", "2", *out]) == 2
    assert (tmp_path / f12.FIG4_MC_CSV_NAME).read_bytes() == before

    # ... and so is a different configuration, unless --fresh replaces the file.
    stubbed["fingerprint"] = [{"alpha": 1.0, "beta": 0.5}]
    assert gen.main(["--lets", "100", "--realizations", "4", "--seeds", "2", *out]) == 2
    assert gen.main(["--lets", "100", "--realizations", "4", "--seeds", "2", "--fresh", *out]) == 0
    assert read_header(tmp_path / f12.FIG4_MC_CSV_NAME)["n_realizations"] == "4"


def test_the_lock_file_goes_with_its_holder_even_on_an_interrupt(
    monkeypatch: pytest.MonkeyPatch, tmp_path: Path
) -> None:
    """An interrupted run leaves no ``.lock`` behind; a refused second run removes nothing."""
    gen = _load_generator(monkeypatch)
    lock = tmp_path / "out.csv.lock"
    with pytest.raises(KeyboardInterrupt):
        with gen._exclusive_lock(lock):
            with pytest.raises(SystemExit, match="another generator"):
                with gen._exclusive_lock(lock):
                    pass  # pragma: no cover - refused
            assert lock.exists()
            raise KeyboardInterrupt
    assert not lock.exists()


def test_seed_statistics_are_the_examples_arithmetic() -> None:
    """``fig4_mc_seed_statistics`` = ``mc_alpha_beta_sweep``'s mean and std(ddof=1)/sqrt(n)."""
    rng = np.random.default_rng(3)
    pairs = np.column_stack([rng.normal(1.0, 0.1, 8), rng.normal(0.02, 0.01, 8)])
    expected = []
    for col in (pairs[:, 0], pairs[:, 1]):
        expected += [float(col.mean()), float(col.std(ddof=1) / math.sqrt(col.size))]
    assert f12.fig4_mc_seed_statistics(pairs) == tuple(expected)
    assert f12.fig4_mc_seed_statistics(pairs.tolist()) == tuple(expected)
    with pytest.raises(ValueError, match="2 seed rows"):
        f12.fig4_mc_seed_statistics([[1.0, 0.1]])


@pytest.mark.slow
def test_a_tiny_real_run_passes_its_own_fingerprint_and_a_perturbed_one_fails(
    monkeypatch: pytest.MonkeyPatch, tmp_path: Path
) -> None:
    """End to end with the real MC (NumPy, one cheap fingerprint point, 2 x 2 realisations)."""
    cheap = ((100.0, 2, 0),)
    original = f12.fig4_mc_fingerprint
    monkeypatch.setattr(f12, "fig4_mc_fingerprint", lambda *a, **kw: original(cheap, **kw))
    gen = _load_generator(monkeypatch)
    argv = ["--lets", "100", "--realizations", "2", "--seeds", "2", "--engine", "numpy"]
    assert gen.main([*argv, "--out", str(tmp_path)]) == 0

    recorded = read_header(tmp_path / f12.FIG4_MC_CSV_NAME)[FINGERPRINT_KEY]
    assert [(p["let_keV_um"], p["n_realizations"], p["seed"]) for p in recorded] == list(cheap)
    recomputed = original([(p["let_keV_um"], p["n_realizations"], p["seed"]) for p in recorded])
    assert_fingerprint(recorded, recomputed, generator=f12.FIG4_MC_GENERATOR)

    perturbed = [{**recorded[0], "beta": recorded[0]["beta"] * (1.0 + 1e-6)}]
    with pytest.raises(AssertionError, match=r"regenerate with scripts/lem4_friedrich2012_mc.py"):
        assert_fingerprint(perturbed, recomputed, generator=f12.FIG4_MC_GENERATOR)
