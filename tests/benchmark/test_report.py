"""Benchmark report: markdown + plot artefacts."""

from __future__ import annotations

from pathlib import Path

from pyradbio.benchmark.harness import BenchmarkResult, PointResult
from pyradbio.benchmark.report import write_report


def _result() -> BenchmarkResult:
    points = (
        PointResult("V79", 6, 12, 100.0, 30.0, 0.2, 0.40, 0.50),
        PointResult("V79", 6, 12, 11.0, 150.0, 0.2, 0.90, 0.85),
        PointResult("HSG", 2, 4, 20.0, 20.0, 0.31, 0.55, 0.60),
    )
    return BenchmarkResult(model_name="lem1", points=points, n_skipped=2)


def test_write_report_markdown_only(tmp_path: Path) -> None:
    report = write_report(_result(), tmp_path, plots=False)
    assert report == tmp_path / "report.md"
    text = report.read_text(encoding="utf-8")
    assert "PIDE benchmark - lem1" in text
    assert "Points evaluated: **3**" in text
    assert "Points skipped" in text
    # per-ion table lists carbon and helium
    assert "| C |" in text
    assert "| He |" in text
    # no plot references when plots=False
    assert "alpha_scatter.png" not in text


def test_write_report_with_plots(tmp_path: Path) -> None:
    report = write_report(_result(), tmp_path, plots=True)
    assert (tmp_path / "alpha_scatter.png").is_file()
    assert (tmp_path / "residual_vs_let.png").is_file()
    text = report.read_text(encoding="utf-8")
    assert "alpha_scatter.png" in text
    assert "residual_vs_let.png" in text


def test_write_report_empty_result_no_plots(tmp_path: Path) -> None:
    empty = BenchmarkResult(model_name="lem1", points=(), n_skipped=7)
    report = write_report(empty, tmp_path, plots=True)
    text = report.read_text(encoding="utf-8")
    assert "Points evaluated: **0**" in text
    # nothing to plot -> no PNGs, no image links
    assert not (tmp_path / "alpha_scatter.png").exists()
    assert "alpha_scatter.png" not in text


def test_report_states_the_scope_and_warning_tallies(tmp_path: Path) -> None:
    result = BenchmarkResult(
        model_name="mcnamara",
        points=_result().points,
        n_skipped=2,
        n_out_of_scope=1037,
        n_warnings=45,
    )
    text = write_report(result, tmp_path, plots=False).read_text(encoding="utf-8")
    assert "1037" in text
    assert "45" in text
    # a scope limit must not read as a failure
    assert "outside the model's fitted scope" in text


def test_report_carries_the_pide_circularity_caveat(tmp_path: Path) -> None:
    """The one thing that must never be dropped from a phenomenological score."""
    result = BenchmarkResult(model_name="mcnamara", points=_result().points, n_skipped=0)
    text = write_report(result, tmp_path, plots=False).read_text(encoding="utf-8")
    assert "fitted to substantially what PIDE aggregates" in text
    assert "LET_d < 20 keV/um" in text  # its own fitted domain
    # and the standing scope note, on every report
    assert "not against a paper's own" in text


def test_report_caveats_are_per_model(tmp_path: Path) -> None:
    wilkens = BenchmarkResult(model_name="wilkens", points=_result().points, n_skipped=0)
    text = write_report(wilkens, tmp_path, plots=False).read_text(encoding="utf-8")
    assert "V79 ion-response intercept" in text
    lem = BenchmarkResult(model_name="lem1", points=_result().points, n_skipped=0)
    text = write_report(lem, tmp_path / "lem", plots=False).read_text(encoding="utf-8")
    assert "V79 ion-response intercept" not in text
    assert "D_t = 30 Gy" in text


def test_report_is_titled_and_caveated_by_dataset(tmp_path: Path) -> None:
    """A run on another dataset must say so, and carry that dataset's own caveats."""
    result = BenchmarkResult(
        model_name="lem1",
        points=_result().points,
        n_skipped=0,
        dataset_name="RadPhysBio",
        dataset_caveats=("Largely digitized.", "Ion LET partly derived."),
    )
    text = write_report(result, tmp_path, plots=False).read_text(encoding="utf-8")
    assert text.startswith("# RadPhysBio benchmark - lem1")
    assert "### About the RadPhysBio data" in text
    assert "- Largely digitized." in text
    assert "- Ion LET partly derived." in text
    # the standing scope note is still there, before the dataset notes
    assert text.index("not against a paper's own") < text.index("About the RadPhysBio data")
