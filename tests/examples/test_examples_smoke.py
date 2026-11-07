"""Every example runs end-to-end, writes a PNG and stays within the runtime budget.

Discovery-based: a new ``examples/<name>.py`` is picked up automatically, and
``test_examples_set_is_exactly`` makes adding or removing one a deliberate edit.
The duration guard (120 s) applies only locally without
coverage: CI (``$CI`` set) and coverage runs are slower and only smoke-test.
"""

from __future__ import annotations

from pathlib import Path

import pytest

from tests.examples._harness import (
    EXAMPLE_BUDGET_S,
    assert_wrote_png,
    duration_guard_applies,
    run_script,
)

EXAMPLES_DIR = Path(__file__).resolve().parents[2] / "examples"
EXAMPLES = sorted(EXAMPLES_DIR.glob("[!_]*.py"))


@pytest.mark.parametrize("path", [pytest.param(p, id=p.stem) for p in EXAMPLES])
def test_example_runs_and_writes_a_png(
    path: Path, monkeypatch: pytest.MonkeyPatch, tmp_path: Path
) -> None:
    run = run_script(path, monkeypatch, tmp_path)

    assert_wrote_png(run, path)
    if duration_guard_applies():
        assert run.elapsed_s <= EXAMPLE_BUDGET_S, (
            f"{path.name} took {run.elapsed_s:.1f} s; examples must finish within "
            f"{EXAMPLE_BUDGET_S:.0f} s (move the work to validation/ or coarsen the demo)"
        )


#: The agreed gallery: a new example is a deliberate decision, and a
#: reference reconstruction belongs in ``validation/`` instead.
_EXAMPLE_SET = frozenset(
    {
        "dose_protraction_and_bed",
        "katz_ion_kill_and_gamma_kill",
        "lem1_variants",
        "lineal_energy_spectra",
        "mcf_mkm_rbe_vs_let",
        "mixed_field_parameters",
        "mkm_zstar_and_alpha",
        "model_comparison_alpha_beta",
        "nanox_local_and_global_events",
        "osmk_oxygen_effect",
        "phenomenological_rbe_models",
        "radial_dose_profiles",
        "rmf_dsb_and_binary_misrepair",
        "smk_nucleus_stochastics",
        "survival_and_rbe",
        "water_stopping_power",
    }
)


def test_examples_set_is_exactly() -> None:
    """The gallery is exactly the agreed set; this also guards an empty parametrization."""
    assert {p.stem for p in EXAMPLES} == _EXAMPLE_SET, (
        f"examples/ changed; update _EXAMPLE_SET and examples/README.md ({EXAMPLES_DIR})"
    )
