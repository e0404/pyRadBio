"""Benchmark report generation: markdown + scatter/residual plots.

Renders a :class:`~pyradbio.benchmark.harness.BenchmarkResult` into a small
self-contained report directory: a ``report.md`` with the aggregate metrics and a
per-ion residual table, plus (optionally) two PNGs -- predicted-vs-measured
``alpha`` (log-log scatter) and the log-residual versus LET. Plotting needs the
``plot`` extra (matplotlib); the markdown itself has no optional dependency, so a
report can always be written with ``plots=False``.
"""

from __future__ import annotations

import math
from collections.abc import Callable, Sequence
from pathlib import Path

from pyradbio.benchmark import metrics
from pyradbio.benchmark.harness import (
    BENCHMARK_SCOPE_NOTE,
    BenchmarkResult,
    PointResult,
    model_caveats,
)
from pyradbio.core.elements import z_to_symbol

_SCATTER_PNG = "alpha_scatter.png"
_RESIDUAL_PNG = "residual_vs_let.png"

_Metric = Callable[[Sequence[float], Sequence[float]], float]


def _safe(fn: _Metric, points: Sequence[PointResult], attr: str) -> float | None:
    """Apply a metric to a point attribute, returning ``None`` if it cannot."""
    pred = [float(getattr(p, "predicted_" + attr)) for p in points]
    meas = [float(getattr(p, "measured_" + attr)) for p in points]
    try:
        return fn(pred, meas)
    except ValueError:
        return None


def _fmt(value: float | None) -> str:
    return "n/a" if value is None else f"{value:.4f}"


def _per_ion_table(result: BenchmarkResult) -> str:
    lines = [
        "| Ion | n | mean log err | mean |log err| | SMAPE |",
        "| --- | ---: | ---: | ---: | ---: |",
    ]
    for z in sorted(result.by_ion()):
        pts = result.by_ion()[z]
        pred = [p.predicted_alpha for p in pts]
        meas = [p.measured_alpha for p in pts]
        try:
            mle = metrics.mean_log_error(pred, meas)
            male = metrics.mean_absolute_log_error(pred, meas)
            sm = metrics.smape(pred, meas)
        except ValueError:
            mle = male = sm = math.nan
        sym = z_to_symbol(z) if 1 <= z <= 10 else f"Z{z}"
        lines.append(f"| {sym} | {len(pts)} | {mle:.4f} | {male:.4f} | {sm:.4f} |")
    return "\n".join(lines)


def _caveats(result: BenchmarkResult) -> str:
    """What must be read alongside the numbers: the run, the model, the dataset."""
    notes = [BENCHMARK_SCOPE_NOTE, *model_caveats(result.model_name)]
    lines = ["## Read this with the numbers", "", *(f"- {n}" for n in notes)]
    if result.dataset_caveats:
        lines += [
            "",
            f"### About the {result.dataset_name} data",
            "",
            *(f"- {n}" for n in result.dataset_caveats),
        ]
    return "\n".join(lines)


def _markdown(result: BenchmarkResult, *, with_plots: bool) -> str:
    parts = [
        f"# {result.dataset_name} benchmark - {result.model_name}",
        "",
        f"- Points evaluated: **{result.n_points}**",
        f"- Points skipped (uncovered ion/energy or model error): {result.n_skipped}",
        f"- Points outside the model's fitted scope (ion species, or no beta_x): "
        f"{result.n_out_of_scope}",
        f"- Out-of-fitted-domain warnings raised by the model: {result.n_warnings}",
        "",
        _caveats(result),
        "",
        "## Aggregate metrics (predicted vs measured alpha)",
        "",
        "The initial-slope RBE (`alpha_ion / alpha_x`) gives identical values for all",
        "three metrics -- the shared per-point `alpha_x` cancels in every ratio -- so",
        "only the `alpha` figures are reported.",
        "",
        "| Metric | value |",
        "| --- | ---: |",
        f"| mean log error | {_fmt(_safe(metrics.mean_log_error, result.points, 'alpha'))} |",
        f"| mean \\|log error\\| "
        f"| {_fmt(_safe(metrics.mean_absolute_log_error, result.points, 'alpha'))} |",
        f"| SMAPE | {_fmt(_safe(metrics.smape, result.points, 'alpha'))} |",
        "",
        "## Residuals by ion",
        "",
        _per_ion_table(result),
        "",
    ]
    if with_plots:
        parts += [
            "## Plots",
            "",
            f"![predicted vs measured alpha]({_SCATTER_PNG})",
            "",
            f"![log residual vs LET]({_RESIDUAL_PNG})",
            "",
        ]
    return "\n".join(parts)


def _write_plots(result: BenchmarkResult, out_dir: Path) -> None:
    """Write the scatter + residual PNGs (needs the ``plot`` extra)."""
    try:
        import matplotlib

        matplotlib.use("Agg")
        import matplotlib.pyplot as plt
    except ModuleNotFoundError as exc:  # pragma: no cover - exercised via extra gating
        raise ImportError(
            "Benchmark plots need the 'plot' extra: pip install 'pyradbio[plot]'."
        ) from exc

    pred = [p.predicted_alpha for p in result.points]
    meas = [p.measured_alpha for p in result.points]
    lets = [p.let_keV_um for p in result.points]
    resid = [p.log_residual_alpha for p in result.points]

    fig, ax = plt.subplots(figsize=(4.5, 4.5))
    ax.scatter(meas, pred, s=14, alpha=0.6, edgecolors="none")
    if pred and meas:
        lo = min(min(pred), min(meas))
        hi = max(max(pred), max(meas))
        ax.plot([lo, hi], [lo, hi], "k--", lw=1, label="identity")
    ax.set_xscale("log")
    ax.set_yscale("log")
    ax.set_xlabel("measured alpha [Gy$^{-1}$]")
    ax.set_ylabel("predicted alpha [Gy$^{-1}$]")
    ax.set_title(f"{result.model_name}: predicted vs measured alpha")
    ax.legend(loc="best", fontsize=8)
    fig.tight_layout()
    fig.savefig(out_dir / _SCATTER_PNG, dpi=120)
    plt.close(fig)

    fig, ax = plt.subplots(figsize=(5.0, 3.5))
    ax.scatter(lets, resid, s=14, alpha=0.6, edgecolors="none")
    ax.axhline(0.0, color="k", lw=1)
    ax.set_xscale("log")
    ax.set_xlabel("LET [keV/um]")
    ax.set_ylabel(r"log$_{10}$(pred / meas) alpha")
    ax.set_title(f"{result.model_name}: residual vs LET")
    fig.tight_layout()
    fig.savefig(out_dir / _RESIDUAL_PNG, dpi=120)
    plt.close(fig)


def write_report(result: BenchmarkResult, out_dir: str | Path, *, plots: bool = True) -> Path:
    """Write a markdown report (+ optional plots) for ``result`` under ``out_dir``.

    Returns the path to ``report.md``. Set ``plots=False`` to skip the PNGs (and
    the matplotlib dependency).
    """
    out = Path(out_dir)
    out.mkdir(parents=True, exist_ok=True)
    if plots and result.n_points:
        _write_plots(result, out)
    report = out / "report.md"
    report.write_text(
        _markdown(result, with_plots=plots and bool(result.n_points)), encoding="utf-8"
    )
    return report
