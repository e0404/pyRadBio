"""CLI smoke test for `pyradbio benchmark` on the synthetic fixture."""

from __future__ import annotations

from pathlib import Path

import pytest

from pyradbio.cli import main

FIXTURE = Path(__file__).parent / "fixtures" / "synthetic_pide.xlsx"


def test_benchmark_cli_writes_report(tmp_path, capsys) -> None:  # type: ignore[no-untyped-def]
    out = tmp_path / "report_dir"
    rc = main(
        [
            "benchmark",
            "--pide",
            str(FIXTURE),
            "--model",
            "lem1",
            "--out",
            str(out),
            "--no-plots",
        ]
    )
    assert rc == 0
    assert (out / "report.md").is_file()
    printed = capsys.readouterr().out
    assert "lem1" in printed
    assert "mean |log err|" in printed


def test_benchmark_cli_missing_pide_exits_cleanly(tmp_path) -> None:  # type: ignore[no-untyped-def]
    with pytest.raises(SystemExit):
        main(["benchmark", "--pide", str(tmp_path), "--out", str(tmp_path / "r")])


def test_benchmark_cli_offers_every_registered_model() -> None:
    """The benchmark registry, not `table`'s builders, defines --model here."""
    from pyradbio.benchmark.harness import benchmark_models
    from pyradbio.cli import _build_parser

    action = next(
        a
        for a in _build_parser()._subparsers._group_actions[0].choices["benchmark"]._actions  # type: ignore[union-attr]
        if a.dest == "model"
    )
    assert set(action.choices or ()) == set(benchmark_models())


def test_benchmark_cli_rejects_a_knob_the_model_does_not_take(tmp_path) -> None:  # type: ignore[no-untyped-def]
    """--rn against a preset model must fail loudly, not be ignored."""
    with pytest.raises(SystemExit, match="published parameter set"):
        main(
            [
                "benchmark",
                "--pide",
                str(FIXTURE),
                "--model",
                "smk",
                "--rn",
                "5.0",
                "--out",
                str(tmp_path),
                "--no-plots",
            ]
        )


RPB_FIXTURE = Path(__file__).parent / "fixtures" / "radphysbio_excerpt.csv"


def test_benchmark_cli_runs_on_radphysbio(tmp_path, capsys) -> None:  # type: ignore[no-untyped-def]
    out = tmp_path / "rpb"
    rc = main(
        [
            "benchmark",
            "--dataset",
            "radphysbio",
            "--path",
            str(RPB_FIXTURE),
            "--model",
            "mcnamara",
            "--out",
            str(out),
            "--no-plots",
        ]
    )
    assert rc == 0
    text = (out / "report.md").read_text(encoding="utf-8")
    assert text.startswith("# RadPhysBio benchmark - mcnamara")
    assert "About the RadPhysBio data" in text
    assert "Pairing tally" in text
    assert "on RadPhysBio" in capsys.readouterr().out


def test_benchmark_cli_runs_on_a_custom_csv(tmp_path) -> None:  # type: ignore[no-untyped-def]
    csv_path = tmp_path / "mine.csv"
    csv_path.write_text(
        "cell,ion,energy,let,alpha_ion,beta_ion,alpha_x,beta_x\n"
        "V79,C-12,100,30,0.45,0.045,0.13,0.05\n"
        "V79,H-1,10,5,0.20,0.05,0.13,0.05\n",
        encoding="utf-8",
    )
    out = tmp_path / "csv_report"
    rc = main(
        [
            "benchmark",
            "--dataset",
            "csv",
            "--path",
            str(csv_path),
            "--model",
            "wedenberg",
            "--out",
            str(out),
            "--no-plots",
        ]
    )
    assert rc == 0
    text = (out / "report.md").read_text(encoding="utf-8")
    assert text.startswith("# csv benchmark - wedenberg")
    assert "user-supplied" in text


def test_benchmark_cli_csv_needs_a_path(tmp_path) -> None:  # type: ignore[no-untyped-def]
    with pytest.raises(SystemExit, match="--path"):
        main(["benchmark", "--dataset", "csv", "--out", str(tmp_path)])


def test_benchmark_cli_csv_without_energy(tmp_path: Path) -> None:
    csv_path = tmp_path / "let_only.csv"
    csv_path.write_text(
        "cell,ion,let,alpha_ion,beta_ion,alpha_x,beta_x\nX,H-1,5,0.4,0.04,0.2,0.03\n",
        encoding="utf-8",
    )
    out = tmp_path / "report"
    assert (
        main(
            [
                "benchmark",
                "--dataset",
                "csv",
                "--path",
                str(csv_path),
                "--model",
                "wedenberg",
                "--out",
                str(out),
                "--no-plots",
            ]
        )
        == 0
    )
    assert "Points evaluated: **1**" in (out / "report.md").read_text(encoding="utf-8")


def test_benchmark_cli_offers_every_registered_dataset() -> None:
    from pyradbio.benchmark.dataset import dataset_names
    from pyradbio.cli import _build_parser

    action = next(
        a
        for a in _build_parser()._subparsers._group_actions[0].choices["benchmark"]._actions  # type: ignore[union-attr]
        if a.dest == "dataset"
    )
    assert set(action.choices or ()) == set(dataset_names())


def test_radphysbio_cli_rejects_an_energy_based_model_without_writing_a_report(tmp_path) -> None:  # type: ignore[no-untyped-def]
    out = tmp_path / "unsupported"
    with pytest.raises(SystemExit, match="nominal beam energ"):
        main(
            [
                "benchmark",
                "--dataset",
                "radphysbio",
                "--path",
                str(RPB_FIXTURE),
                "--model",
                "lem1",
                "--out",
                str(out),
                "--no-plots",
            ]
        )
    assert not out.exists()
