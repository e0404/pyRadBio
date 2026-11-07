"""Source-agnostic reference records, the custom-CSV loader and the dataset registry."""

from __future__ import annotations

from pathlib import Path

import pytest

from pyradbio.benchmark import dataset as ds
from pyradbio.benchmark.pide import PideEntry


def test_pide_entry_is_a_reference_record() -> None:
    e = PideEntry("V79", 6, 100.0, 30.0, 0.4, 0.04, 0.1, 0.05, A=12, source="pide:x")
    assert isinstance(e, ds.ReferenceRecord)
    assert e.rbe_alpha == pytest.approx(4.0)


def test_parse_reference_rows_reads_a_minimal_custom_table() -> None:
    headers = [
        "cell",
        "ion",
        "energy",
        "LET",
        "alpha_ion",
        "beta_ion",
        "alpha_x",
        "beta_x",
        "source",
    ]
    rows = [
        ["X", "C-12", "100", "30.0", "0.5", "0.1", "0.2", "0.03", "example"],
        ["X", "C-12", "", "40.0", "0.7", "", "0.2", "0.03", "example"],  # no beta_ion
    ]
    out = ds.parse_reference_rows(headers, rows, dataset="mine")
    assert len(out) == 1
    (r,) = out
    assert (r.Z, r.A) == (6, 12)
    assert r.energy_MeV_u == pytest.approx(100.0)
    assert r.source == "mine:example"


def test_parse_reference_rows_strict_names_the_missing_field() -> None:
    headers = ["cell", "charge", "LET", "alpha_ion", "beta_ion", "alpha_x", "beta_x"]
    with pytest.raises(ValueError, match="beta_x"):
        ds.parse_reference_rows(
            headers, [["X", "6", "30", "0.4", "0.04", "0.1", "N/A"]], strict=True
        )


def test_load_reference_csv_roundtrip(tmp_path: Path) -> None:
    p = tmp_path / "mine.csv"
    p.write_text(
        "cell,ion,energy,let,alpha_ion,beta_ion,alpha_x,beta_x\n"
        "V79,C-12,100,30,0.45,0.045,0.13,0.05\n"
        "V79,He-4,10,20,0.30,0.05,0.13,0.05\n",
        encoding="utf-8",
    )
    got = ds.load_reference_csv(p)
    assert got.name == "csv"
    assert len(got) == 2
    assert {r.Z for r in got.records} == {2, 6}
    assert got.caveats and "user-supplied" in got.caveats[0]


def test_registry_lists_the_bundled_loaders() -> None:
    assert set(ds.dataset_names()) >= {"pide", "radphysbio", "csv"}


def test_registry_unknown_name_and_csv_without_path() -> None:
    with pytest.raises(ValueError, match="unknown dataset"):
        ds.load_dataset("nope")
    with pytest.raises(FileNotFoundError, match="--path"):
        ds.load_dataset("csv")


def test_registry_is_user_extensible(tmp_path: Path) -> None:
    rec = ds.ReferenceRecord("X", 1, 10.0, 5.0, 0.3, 0.03, 0.2, 0.02)
    ds.DATASET_LOADERS["_test_only"] = lambda p: ds.ReferenceDataset("mine", (rec,))
    try:
        assert ds.load_dataset("_test_only").records == (rec,)
    finally:
        del ds.DATASET_LOADERS["_test_only"]


def test_to_float_and_missing_markers() -> None:
    assert ds.to_float("1.5") == 1.5
    assert ds.to_float(3) == 3.0
    assert ds.to_float(True) is None
    assert ds.to_float("N/A") is None
    assert ds.to_float("\u039d/\u0391") is None
    assert ds.clean_str(" \u039d/A ") == ""
    assert ds.clean_str(" HSG ") == "HSG"


@pytest.mark.parametrize("energy", [None, 0.1, 10.0])
@pytest.mark.parametrize("wrapped", [False, True])
def test_custom_csv_energy_is_optional_only_for_let_models(
    tmp_path: Path, energy: float | None, wrapped: bool
) -> None:
    from pyradbio.benchmark.harness import run_benchmark

    path = tmp_path / "optional_energy.csv"
    path.write_text(
        "cell,ion,energy,let,alpha_ion,beta_ion,alpha_x,beta_x\n"
        f"X,H-1,{'' if energy is None else energy},5,0.4,0.04,0.2,0.03\n",
        encoding="utf-8",
    )
    dataset = ds.load_reference_csv(path)
    entries = dataset if wrapped else dataset.records
    result = run_benchmark(entries, "wedenberg")
    # Only a *missing* energy is waived; a known one below the floor still gates,
    # so LET and ion models see the same points wherever the energy is known.
    assert result.n_points == (energy != 0.1)
    assert result.n_skipped == (energy == 0.1)
    if result.points:
        assert result.points[0].energy_MeV_u == energy
    # Omitting an energy must not mislabel it as nominal or bypass ion-route checks.
    assert not dataset.records[0].energy_is_nominal
    ion_result = run_benchmark(entries, "lem1")
    assert ion_result.n_points == (energy == 10.0)
    assert ion_result.n_skipped == (energy != 10.0)


def test_custom_csv_let_route_honours_an_explicit_energy_floor(tmp_path: Path) -> None:
    from pyradbio.benchmark.harness import run_benchmark

    path = tmp_path / "floor.csv"
    path.write_text(
        "cell,ion,energy,let,alpha_ion,beta_ion,alpha_x,beta_x\n"
        "X,H-1,3,5,0.4,0.04,0.2,0.03\n"
        "X,H-1,,6,0.45,0.04,0.2,0.03\n",
        encoding="utf-8",
    )
    result = run_benchmark(ds.load_reference_csv(path), "wedenberg", min_energy_MeV_u=5.0)
    # The 3 MeV/u row falls below the caller's floor; the energy-less row is waived.
    assert result.n_skipped == 1
    assert [p.energy_MeV_u for p in result.points] == [None]
