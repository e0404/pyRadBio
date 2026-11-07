"""The two LEM I reconstructions draw the paper panels with their labelled roles (``@slow``).

The scripts are ``validation/scholz_kraft1996_lem1.py`` (Scholz & Kraft 1996,
doi:10.1016/0273-1177(95)00784-C) and ``validation/scholz1997_lem1.py`` (Scholz et al.
1997, doi:10.1007/s004110050055). The legend labels are asserted in full and in quick
mode: quick mode thins the energy sweeps (and draws Fig. 2's expansion lines for the
Fig. 3 ions only), but it must not change the roles a panel's legend names.
The generic quick-mode smoke run of every validation script is
``tests/validation/test_validation_smoke.py``.
"""

from __future__ import annotations

import runpy
from pathlib import Path
from typing import Any

import pytest

pytestmark = pytest.mark.slow

VALIDATION_DIR = Path(__file__).resolve().parents[2] / "validation"


def _run(name: str, quick: bool, monkeypatch: pytest.MonkeyPatch, tmp_path: Path) -> dict[str, Any]:
    matplotlib = pytest.importorskip("matplotlib")
    monkeypatch.setenv("MPLBACKEND", "Agg")
    monkeypatch.setenv("PYRADBIO_FIGURES_DIR", str(tmp_path))
    if quick:
        monkeypatch.setenv("PYRADBIO_QUICK", "1")
    else:
        monkeypatch.delenv("PYRADBIO_QUICK", raising=False)
    monkeypatch.delenv("LEM1_STOPPING", raising=False)
    monkeypatch.syspath_prepend(str(VALIDATION_DIR))
    matplotlib.use("Agg", force=True)
    return runpy.run_path(str(VALIDATION_DIR / f"{name}.py"), run_name="__main__")


def _labels(*legends: Any) -> set[str]:
    return {text.get_text() for legend in legends for text in legend.get_texts()}


@pytest.fixture(params=[False, True], ids=["full", "quick"])
def quick(request: pytest.FixtureRequest) -> bool:
    return bool(request.param)


def test_scholz_kraft1996_panels(
    quick: bool, monkeypatch: pytest.MonkeyPatch, tmp_path: Path
) -> None:
    """Figs. 2-4: the final-slope MC, the cluster expansion and the sigma_0 reference."""
    import matplotlib.pyplot as plt

    try:
        namespace = _run("scholz_kraft1996_lem1", quick, monkeypatch, tmp_path)
        for name in (
            "sk1996_fig2_sigma_vs_let.png",
            "sk1996_fig3_sigma_vs_energy.png",
            "sk1996_fig4_rbe_vs_let.png",
        ):
            assert (tmp_path / name).is_file()
        for figure_name in ("fig2", "fig3"):
            labels = _labels(namespace[figure_name].axes[0].get_legend())
            assert "pyRadBio sigma_0: zero-dose reference" in labels
            assert "pyRadBio full MC" in labels
        assert "pyRadBio 2nd-order cluster expansion at 1%" in _labels(
            namespace["fig2"].axes[0].get_legend()
        )
        fig4_labels = _labels(namespace["fig4"].axes[0].get_legend())
        assert "paper sigma -> RBE(1%), upper bound (C: Fig. 3, He: Fig. 2)" in fig4_labels
        assert {"S=0.8", "S=0.2", "S=0.01"} <= fig4_labels
    finally:
        plt.close("all")


def test_scholz1997_panels(quick: bool, monkeypatch: pytest.MonkeyPatch, tmp_path: Path) -> None:
    """Figs. 1-5: exact vs approximation paths, with the MC and z-slope overlays."""
    import matplotlib.pyplot as plt

    try:
        namespace = _run("scholz1997_lem1", quick, monkeypatch, tmp_path)
        for name in (
            "sk1997_fig1_alpha_vs_energy.png",
            "sk1997_fig3_beta_vs_energy.png",
            "sk1997_fig4_slope_vs_specific_energy.png",
            "sk1997_fig2_fig5_dose.png",
        ):
            assert (tmp_path / name).is_file()
        assert len(namespace["fig4b"].axes) == 4
        assert namespace["Z_SLOPE"], "the z-slope checkpoint was not read"
        fig25_labels = _labels(*namespace["fig25"].legends)
        assert "solid: pyRadBio explicit LQ" in fig25_labels
        assert "dash-dot: pyRadBio approx." in fig25_labels
        assert "*: stochastic MC" in _labels(namespace["leg_role1"])
        assert "stochastic MC" in _labels(namespace["leg_role3"])
    finally:
        plt.close("all")
