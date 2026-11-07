"""The Scholz & Kraft 1996 offline full-LEM observables (``benchmark.scholz_kraft1996``).

Unit tests of the offline batch machinery: the default point catalogue,
``estimate_final_slope_point`` on curves with closed-form observables, and
``checkpoint_reuse_reason``. The last test drives ``scripts/lem1_mc_reference.py
sk1996-final-slope`` end to end at a tiny sample size.

Reference: Scholz & Kraft 1996, doi:10.1016/0273-1177(95)00784-C.
"""

from __future__ import annotations

import importlib.util
import math
from pathlib import Path
from types import ModuleType, SimpleNamespace

import numpy as np
import pytest

from pyradbio.benchmark import scholz_kraft1996 as sk
from pyradbio.benchmark.checkpoint import FINGERPRINT_KEY, CsvCheckpoint, read_header
from pyradbio.core.lq import LQParameters
from pyradbio.core.photon import PhotonParameters
from pyradbio.models.lem.base import TraversalConditionedSurvival

_SCRIPT = Path(__file__).resolve().parents[2] / "scripts" / "lem1_mc_reference.py"


def test_scholz_kraft_mc_batch_uses_paper_fig4_energy_interval() -> None:
    """The Fig. 4 p/He/C batch covers 1-20 MeV/u (Scholz & Kraft 1996).

    Reference: doi:10.1016/0273-1177(95)00784-C.
    """
    points = set(sk.MC_POINTS)

    for particle in sk.RBE_PARTICLES:
        energies = sorted(energy for name, energy in points if name == particle)
        assert energies[0] == 1.0
        assert energies[-1] == 20.0


def _exponential_curve(
    *,
    max_dose_Gy: float,
    alpha_Gy_inv: float,
    n_batches: int = 16,
    realizations_per_batch: int = 25,
    batch_spread: float = 0.0,
    seed: int = 0,
) -> TraversalConditionedSurvival:
    """A conditioned curve whose survival is exactly ``exp(-alpha D)``.

    ``g_k = q^k`` collapses the Poisson mixture to ``exp(-lambda (1 - q))``, so
    every observable the batch reads off the curve has a closed form to check
    against. Geometry is chosen so ``dlambda/dD = 1`` and hence ``lambda = D``.
    ``batch_spread`` scatters the per-batch estimates to give the jackknife a
    non-zero standard error, which is what drives the adaptive allocator.
    """
    k_max = math.ceil(max_dose_Gy + 8.0 * math.sqrt(max_dose_Gy)) + 12
    g = (1.0 - alpha_Gy_inv) ** np.arange(k_max + 1, dtype=float)
    rng = np.random.default_rng(seed)
    scatter = 1.0 + batch_spread * rng.standard_normal((n_batches, 1))
    return TraversalConditionedSurvival(
        batch_g=np.clip(g[None, :] * scatter, 0.0, 1.0),
        b_max_um=1.0,
        mean_traversal_dose_Gy=1.0,
        effective_area_um2=math.pi,
        realizations_per_batch=realizations_per_batch,
        max_supported_dose_Gy=max_dose_Gy,
    )


class _FakeBracket:
    def __init__(self, alpha: float = 0.4, beta: float = 0.025) -> None:
        self._lq = LQParameters(alpha, beta)

    def alpha_beta(self, _ion: object) -> LQParameters:
        return self._lq


class _FakeFull:
    """Stands in for LEM1, returning an exactly-known conditioned curve."""

    def __init__(self, alpha_Gy_inv: float, *, batch_spread: float = 0.0) -> None:
        self.alpha_Gy_inv = alpha_Gy_inv
        self.batch_spread = batch_spread
        self.photon = PhotonParameters(alpha_x=0.136, beta_x=0.040, D_t=12.0)
        self.stopping_power = SimpleNamespace(let_keV_um=lambda _ion: 100.0, name="test-let")

    def scholz1997_traversal_conditioned_survival(
        self,
        _ion: object,
        *,
        max_dose_Gy: float,
        n_batches: int,
        realizations_per_batch: int,
        seed: int,
        **_kwargs: object,
    ) -> TraversalConditionedSurvival:
        return _exponential_curve(
            max_dose_Gy=max_dose_Gy,
            alpha_Gy_inv=self.alpha_Gy_inv,
            n_batches=n_batches,
            realizations_per_batch=realizations_per_batch,
            batch_spread=self.batch_spread,
            seed=seed,
        )

    def inactivation_cross_section_um2(self, _ion: object) -> float:
        return 12.0


def test_scholz_kraft_mc_batch_derives_rbe_from_survival_crossings() -> None:
    """Every Fig. 3/4 observable is read off the curve, not off an LQ surrogate."""
    alpha_ion = 0.25
    full = _FakeFull(alpha_ion)

    result = sk.estimate_final_slope_point(
        full,  # type: ignore[arg-type]
        _FakeBracket(),  # type: ignore[arg-type]
        "carbon",
        3.0,
        n_batches=8,
        realizations_per_batch=25,
        max_realizations=2_000,
        target_relative_error=0.03,
        seed=1,
        max_mean_traversals=20_000.0,
    )

    for level, tag in ((0.80, "80pct"), (0.20, "20pct"), (0.01, "1pct")):
        ion_dose = -math.log(level) / alpha_ion
        photon_dose = full.photon.dose_for_survival(level)
        assert result[f"dose_{tag}_Gy"] == pytest.approx(ion_dose, rel=1e-6)
        assert result[f"rbe_{tag}"] == pytest.approx(photon_dose / ion_dose, rel=1e-6)

    # An exponential curve has final slope == initial slope everywhere.
    assert result["slope_1pct_Gy_inv"] == pytest.approx(alpha_ion, rel=1e-6)
    assert result["sigma_1pct_um2"] == pytest.approx(0.16 * 100.0 * alpha_ion, rel=1e-6)


def test_scholz_kraft_mc_batch_expands_until_the_traversal_grid_spans_one_percent() -> None:
    """A poor LQ dose guess must not abort the full stochastic batch."""
    alpha_ion = 0.02  # far shallower than the bracket model's guess

    result = sk.estimate_final_slope_point(
        _FakeFull(alpha_ion),  # type: ignore[arg-type]
        _FakeBracket(alpha=0.4, beta=0.025),  # type: ignore[arg-type]
        "argon",
        1.0,
        n_batches=8,
        realizations_per_batch=25,
        max_realizations=2_000,
        target_relative_error=0.03,
        seed=1,
        max_mean_traversals=20_000.0,
    )

    assert result["dose_1pct_Gy"] == pytest.approx(-math.log(0.01) / alpha_ion, rel=1e-6)


def test_scholz_kraft_mc_batch_pools_extra_batches_instead_of_restarting() -> None:
    """Precise points stop after the pilot; noisy points grow by ADDING batches.

    Restarting a point with a larger history count and the same seed would discard
    the pilot. Pooling means the cost of reaching the target is
    the target's cost, and it is what makes the jackknife error shrink.
    """
    common = {
        "bracket": _FakeBracket(),
        "particle": "carbon",
        "energy_MeV_u": 3.0,
        "n_batches": 8,
        "realizations_per_batch": 25,
        "max_realizations": 20_000,
        "target_relative_error": 0.03,
        "seed": 10,
        "max_mean_traversals": 20_000.0,
    }
    precise = sk.estimate_final_slope_point(full=_FakeFull(0.25), **common)  # type: ignore[arg-type]
    noisy = sk.estimate_final_slope_point(full=_FakeFull(0.25, batch_spread=0.05), **common)  # type: ignore[arg-type]

    assert precise["n_realizations"] == 200  # pilot only: identical batches, zero spread
    assert noisy["n_realizations"] > precise["n_realizations"]
    assert noisy["max_relative_error"] <= 0.03
    assert noisy["precision_converged"] == 1


def test_scholz_kraft_mc_batch_invalidates_superseded_estimator_checkpoints() -> None:
    """Rows from an older final-slope estimator are a different quantity.

    No per-particle exemption: mixing estimator versions inside one panel would
    plot two definitions against a single paper line.
    """
    reuse_reason = sk.checkpoint_reuse_reason
    version = sk.OBSERVABLE_ESTIMATOR_VERSION
    legacy = {"particle": "carbon", "max_relative_error": 0.01, "n_realizations": 40_000}
    current = {**legacy, "estimator_version": version, "stopping_power": "ns1970+bethe"}
    common = {"target_relative_error": 0.03, "max_realizations": 20_000}

    assert reuse_reason(legacy, stopping_power="ns1970+bethe", **common) is None
    assert (
        reuse_reason({**legacy, "particle": "proton"}, stopping_power="ns1970+bethe", **common)
        is None
    )
    assert reuse_reason(current, stopping_power="ns1970+bethe", **common) == "target met"
    # LET enters both the x axis and the sigma conversion, so a row computed with
    # a different stopping power is a different quantity, not a stale one.
    assert reuse_reason(current, stopping_power="bethe", **common) is None
    assert (
        reuse_reason(
            {**current, "max_relative_error": 0.5}, stopping_power="ns1970+bethe", **common
        )
        == "ceiling reached"
    )


# --- the generator, end to end ------------------------------------------------------


def _generator() -> ModuleType:
    spec = importlib.util.spec_from_file_location("_lem1_mc_reference_under_test", _SCRIPT)
    assert spec is not None and spec.loader is not None
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


_TINY = [
    "--only",
    "carbon:8",
    "--batches",
    "4",
    "--realizations-per-batch",
    "5",
    "--realizations",
    "20",
]


def test_sk1996_generator_writes_resumes_and_guards_its_fingerprint(
    tmp_path: Path, capsys: pytest.CaptureFixture[str], monkeypatch: pytest.MonkeyPatch
) -> None:
    """``sk1996-final-slope`` writes a headered checkpoint, reuses it, refuses a stale one.

    The fingerprint itself is recomputed in ``test_lem1_mc_fingerprints.py``; here a
    stand-in keeps the plumbing test cheap.
    """
    fingerprint = [{"particle": "carbon", "value": 1.25}]
    monkeypatch.setattr(sk, "mc_observables_fingerprint", lambda _sp: fingerprint)
    generator = _generator()
    out = tmp_path / "sk1996_mc_observables.csv"

    assert generator.main(["sk1996-final-slope", *_TINY, "--out", str(tmp_path)]) == 0
    checkpoint = CsvCheckpoint(out, text_columns=sk.MC_OBSERVABLE_TEXT_COLUMNS)
    assert checkpoint.keys() == [("carbon", 8.0)]
    header = read_header(out)
    assert header["generator"] == "scripts/lem1_mc_reference.py sk1996-final-slope"
    assert header["stopping_power"] == "ns1970+bethe"
    assert header[FINGERPRINT_KEY] == fingerprint
    row = sk.load_mc_observables(out)["carbon"][0]
    assert row["stopping_power"] == "ns1970+bethe"
    assert row["n_realizations"] == 20.0

    # A second run with a looser target reuses the point instead of recomputing it.
    capsys.readouterr()
    loose = [*_TINY, "--target-relative-error", "10"]
    assert generator.main(["sk1996-final-slope", *loose, "--out", str(tmp_path)]) == 0
    assert "checkpoint target met; skipping" in capsys.readouterr().out

    # A fingerprint the code no longer reproduces blocks merging until --fresh.
    checkpoint = CsvCheckpoint(out, text_columns=sk.MC_OBSERVABLE_TEXT_COLUMNS)
    checkpoint.header = {
        **checkpoint.header,
        FINGERPRINT_KEY: [{"particle": "carbon", "value": 2.0}],
    }
    checkpoint.save()
    with pytest.raises(SystemExit, match="Rerun with --fresh, which discards EVERY"):
        generator.main(["sk1996-final-slope", *loose, "--out", str(tmp_path)])
    assert generator.main(["sk1996-final-slope", *loose, "--fresh", "--out", str(tmp_path)]) == 0
    assert read_header(out)[FINGERPRINT_KEY] == fingerprint
