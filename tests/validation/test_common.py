"""``validation/_common.py``: the script plumbing shared by the validation scripts.

The module is not part of the wheel; it is loaded from its file the way a
validation script sees it (``validation/`` on ``sys.path``).
"""

from __future__ import annotations

import importlib.util
import sys
from collections.abc import Iterator
from pathlib import Path
from types import ModuleType
from typing import Any

import pytest

from pyradbio.datasets import TRIP98, Dataset, MissingDatasetError

_COMMON_PATH = Path(__file__).resolve().parents[2] / "validation" / "_common.py"


def _load_common() -> ModuleType:
    spec = importlib.util.spec_from_file_location("_validation_common_under_test", _COMMON_PATH)
    assert spec is not None and spec.loader is not None
    module = importlib.util.module_from_spec(spec)
    sys.modules[spec.name] = module
    spec.loader.exec_module(module)
    return module


common = _load_common()


# --- quick mode -----------------------------------------------------------------


@pytest.mark.parametrize("value", ["1", "true", "True", "yes", "on", "2"])
def test_quick_mode_is_on_for_truthy_values(monkeypatch: pytest.MonkeyPatch, value: str) -> None:
    monkeypatch.setenv("PYRADBIO_QUICK", value)
    assert common.quick_mode() is True


@pytest.mark.parametrize("value", ["", "0", "false", "FALSE", "no", "off", " 0 "])
def test_quick_mode_is_off_for_falsy_values(monkeypatch: pytest.MonkeyPatch, value: str) -> None:
    monkeypatch.setenv("PYRADBIO_QUICK", value)
    assert common.quick_mode() is False


def test_quick_mode_is_off_when_unset(monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.delenv("PYRADBIO_QUICK", raising=False)
    assert common.quick_mode() is False


def test_pick_returns_the_full_value_by_default(monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.delenv("PYRADBIO_QUICK", raising=False)
    assert common.pick(300_000, 20_000) == 300_000


def test_pick_returns_the_quick_value_in_quick_mode(monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.setenv("PYRADBIO_QUICK", "1")
    assert common.pick(300_000, 20_000) == 20_000


# --- data and extra gates -------------------------------------------------------


def test_require_data_returns_the_loader_result() -> None:
    def loader(name: str, *, scale: float) -> tuple[str, float]:
        return name, scale

    assert common.require_data(loader, "normal02.rbe", dataset=TRIP98, scale=2.0) == (
        "normal02.rbe",
        2.0,
    )


def test_require_data_turns_a_missing_dataset_into_a_clean_exit() -> None:
    def loader(_name: str) -> None:
        raise MissingDatasetError(TRIP98, None)

    with pytest.raises(common.MissingDataExit) as info:
        common.require_data(loader, "normal02.rbe", dataset=TRIP98)

    assert isinstance(info.value, SystemExit)
    message = str(info.value)
    # The wording the TRiP98 scripts print.
    assert message.startswith("TRiP98 reference data not available (")
    assert message.endswith(
        ").\nSet PYRADBIO_TRIP98_DIR (see pyradbio.datasets / .env.example) to run this script."
    )
    assert isinstance(info.value.__cause__, MissingDatasetError)


def test_require_data_also_catches_a_missing_file_inside_the_dataset() -> None:
    def loader() -> None:
        raise FileNotFoundError("normal02.rbe not found")

    with pytest.raises(common.MissingDataExit, match=r"\(normal02.rbe not found\)"):
        common.require_data(loader, dataset=TRIP98)


def test_require_data_labels_an_unknown_dataset_by_its_key() -> None:
    novel = Dataset("novel", "PYRADBIO_NOVEL_DIR", "A dataset added later.")

    def loader() -> None:
        raise MissingDatasetError(novel, None)

    with pytest.raises(common.MissingDataExit, match=r"^novel reference data not available"):
        common.require_data(loader, dataset=novel)


def test_require_data_does_not_swallow_other_errors() -> None:
    def loader() -> None:
        raise ValueError("malformed table")

    with pytest.raises(ValueError, match="malformed table"):
        common.require_data(loader, dataset=TRIP98)


def test_require_extra_returns_the_imported_module() -> None:
    assert common.require_extra("json", "dev") is sys.modules["json"]


def test_require_extra_turns_a_missing_module_into_a_clean_exit() -> None:
    with pytest.raises(common.MissingExtraExit) as info:
        common.require_extra("pymkm_does_not_exist_xyz", "mkm")

    assert isinstance(info.value, SystemExit)
    assert str(info.value) == (
        "pymkm_does_not_exist_xyz not installed ([mkm] extra): pip install -e '.[mkm]'"
    )


def test_require_extra_names_pymkm_the_way_the_scripts_do(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    # A ``None`` entry makes ``import pymkm`` raise ImportError whether or not it is installed.
    monkeypatch.setitem(sys.modules, "pymkm", None)
    with pytest.raises(common.MissingExtraExit) as info:
        common.require_extra("pymkm", "mkm")
    assert str(info.value) == "pyMKM not installed ([mkm] extra): pip install -e '.[mkm]'"


# --- plot idioms ----------------------------------------------------------------


@pytest.fixture
def ax() -> Iterator[Any]:
    matplotlib = pytest.importorskip("matplotlib")
    matplotlib.use("Agg", force=True)
    import matplotlib.pyplot as plt

    fig, axis = plt.subplots()
    yield axis
    plt.close(fig)


def test_proxy_legend_builds_one_handle_per_entry(ax: Any) -> None:
    legend = common.proxy_legend(
        ax,
        [
            {"color": "0.3", "lw": 2.0, "label": "pyRadBio"},
            {"color": "C1", "ls": "--", "label": "paper model line"},
        ],
        frameon=False,
        loc="upper left",
    )

    assert [t.get_text() for t in legend.get_texts()] == ["pyRadBio", "paper model line"]
    handles = legend.legend_handles if hasattr(legend, "legend_handles") else legend.legendHandles
    assert handles[1].get_linestyle() == "--"
    assert legend.get_frame_on() is False
    assert ax.get_legend() is legend


def test_proxy_legend_does_not_consume_the_property_cycle(ax: Any) -> None:
    common.proxy_legend(ax, [{"color": "k", "label": "proxy"}])
    from matplotlib.colors import same_color

    (line,) = ax.plot([0, 1], [0, 1])
    assert same_color(line.get_color(), "C0")  # the cycle was not advanced


def test_proxy_legend_requires_an_explicit_color(ax: Any) -> None:
    with pytest.raises(ValueError, match="explicit 'color'"):
        common.proxy_legend(ax, [{"color": "k", "label": "a"}, {"ls": "--", "label": "b"}])


def test_proxy_legend_accepts_the_c_alias_for_color(ax: Any) -> None:
    from matplotlib.colors import same_color

    legend = common.proxy_legend(ax, [{"c": "C2", "label": "alias"}])

    handles = legend.legend_handles if hasattr(legend, "legend_handles") else legend.legendHandles
    assert same_color(handles[0].get_color(), "C2")


def test_require_data_names_the_survival_data_as_its_scripts_do() -> None:
    from pyradbio.datasets import DATASETS

    survival = DATASETS["survival"]

    def loader() -> None:
        raise MissingDatasetError(survival, None)

    with pytest.raises(common.MissingDataExit, match=r"^Survival reference data not available"):
        common.require_data(loader, dataset=survival)


def test_plot_reference_overlays_a_line_without_errors(ax: Any) -> None:
    lines = common.plot_reference(ax, [1.0, 2.0], [3.0, 4.0], color="k", ls="none", marker="o")
    assert len(lines) == 1
    assert list(lines[0].get_ydata()) == [3.0, 4.0]
    assert lines[0].get_marker() == "o"


def test_plot_reference_draws_error_bars_when_given(ax: Any) -> None:
    from matplotlib.container import ErrorbarContainer

    container = common.plot_reference(
        ax, [1.0, 2.0], [3.0, 4.0], y_err=[0.1, 0.2], fmt="o", color="k"
    )
    assert isinstance(container, ErrorbarContainer)
    assert container.has_yerr and not container.has_xerr
