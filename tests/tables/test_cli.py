"""CLI smoke tests for `pyradbio survival` and `pyradbio table`."""

from __future__ import annotations

import json
import math

import pytest

from pyradbio.cli import main
from pyradbio.tables import io


def test_version(capsys) -> None:  # type: ignore[no-untyped-def]
    from pyradbio import __version__

    assert main(["--version"]) == 0
    assert capsys.readouterr().out.strip() == __version__


def test_survival_prints_surviving_fraction(capsys) -> None:  # type: ignore[no-untyped-def]
    rc = main(["survival", "--alpha", "0.3", "--beta", "0.05", "--dose", "2.0"])
    assert rc == 0
    out = capsys.readouterr().out
    expected = math.exp(-(0.3 * 2.0 + 0.05 * 2.0**2))
    assert f"{expected:.6f}" in out


def test_table_builds_lem1_csv(tmp_path, capsys) -> None:  # type: ignore[no-untyped-def]
    out = tmp_path / "ab.csv"
    rc = main(
        [
            "table",
            "--model",
            "lem1",
            "--alpha-x",
            "0.1",
            "--beta-x",
            "0.05",
            "--dt",
            "30",
            "--rn",
            "5.0",
            "--ions",
            "1,6",
            "--energies",
            "10,100",
            "--out",
            str(out),
        ]
    )
    assert rc == 0
    table = io.read_csv(out)
    assert {r.Z for r in table.rows} == {1, 6}
    assert len(table.rows) == 4  # 2 ions x 2 energies
    assert table.metadata["model"] == "lem1"
    assert "wrote" in capsys.readouterr().out.lower()


@pytest.mark.parametrize("model", ["lem1", "lem2", "lem3", "lem4"])
def test_table_builds_lem_family_csv(tmp_path, model) -> None:  # type: ignore[no-untyped-def]
    out = tmp_path / f"{model}.csv"
    rc = main(
        [
            "table",
            "--model",
            model,
            "--alpha-x",
            "0.184",
            "--beta-x",
            "0.02",
            "--dt",
            "17",
            "--rn",
            "4.7",
            "--ions",
            "6",
            "--energies",
            "50,200",
            "--out",
            str(out),
        ]
    )
    assert rc == 0
    table = io.read_csv(out)
    assert len(table.rows) == 2
    assert table.metadata["model"] == model
    assert all(r.alpha >= 0 and r.beta >= 0 for r in table.rows)


def test_table_rejects_unknown_model() -> None:
    with pytest.raises(SystemExit):
        main(["table", "--model", "nope", "--out", "x.csv"])


def test_table_mkm_requires_its_own_flags() -> None:
    """Selecting mkm without alpha0/beta0/rd exits cleanly (no traceback)."""
    with pytest.raises(SystemExit):
        main(
            [
                "table",
                "--model",
                "mkm",
                "--rn",
                "3.9",
                "--ions",
                "6",
                "--energies",
                "100",
                "--out",
                "x.csv",
            ]
        )


def _lem4_args(out, *extra: str) -> list[str]:
    return [
        "table",
        "--model",
        "lem4",
        "--alpha-x",
        "0.1",
        "--beta-x",
        "0.05",
        "--dt",
        "30",
        "--rn",
        "5.0",
        "--ions",
        "1,6",
        "--energies",
        "10,100",
        "--out",
        str(out),
        *extra,
    ]


def test_table_records_the_resolved_model_config(tmp_path) -> None:  # type: ignore[no-untyped-def]
    """The header must state the protocol, including axes left at their default.

    A LEM IV table is not comparable with another one unless it names its survival
    convention, cluster-index estimator and beta model -- and the whole point is
    that those change the numbers without changing the call, so recording only the
    flags the caller typed would leave the important ones invisible.
    """
    out = tmp_path / "ab.csv"
    assert main(_lem4_args(out)) == 0
    config = io.read_csv(out).metadata["model_config"]
    assert isinstance(config, dict)
    assert config["model"] == "LEM4"
    # Never passed on the command line, yet decisive for the numbers.
    assert config["survival_convention"] == "mean_survival"
    assert config["cluster_index_model"] == "mean_field"
    assert config["beta_model"] == "slope_matching"
    assert config["legacy_trip98_cluster_index_cap"] is None
    photon = config["photon"]
    assert isinstance(photon, dict)
    assert photon["alpha_x"] == 0.1 and photon["D_t"] == 30.0


def test_table_behaviour_flags_reach_the_model(tmp_path) -> None:  # type: ignore[no-untyped-def]
    out = tmp_path / "ab.csv"
    assert (
        main(
            _lem4_args(
                out,
                "--cluster-index-cap",
                "0.95",
                "--survival-convention",
                "mean_effect",
                "--beta-model",
                "rapid_lem",
                "--effective-size-factor",
                "0.8",
            )
        )
        == 0
    )
    config = io.read_csv(out).metadata["model_config"]
    assert isinstance(config, dict)
    assert config["legacy_trip98_cluster_index_cap"] == 0.95
    assert config["survival_convention"] == "mean_effect"
    assert config["beta_model"] == "rapid_lem"
    cell = config["cell"]
    assert isinstance(cell, dict)
    assert cell["effective_size_factor"] == 0.8


def test_table_rejects_a_behaviour_flag_the_model_lacks(tmp_path) -> None:  # type: ignore[no-untyped-def]
    """Silently ignoring it would hand back a table that is not what was asked for."""
    args = [
        "table",
        "--model",
        "lem1",
        "--alpha-x",
        "0.1",
        "--beta-x",
        "0.05",
        "--dt",
        "30",
        "--rn",
        "5.0",
        "--ions",
        "1",
        "--energies",
        "10",
        "--survival-convention",
        "mean_effect",
        "--out",
        str(tmp_path / "ab.csv"),
    ]
    with pytest.raises(SystemExit, match="survival-convention"):
        main(args)


def test_table_mass_numbers_select_the_isotope(tmp_path) -> None:  # type: ignore[no-untyped-def]
    out = tmp_path / "ab.csv"
    assert main(_lem4_args(out, "--mass-numbers", "2,13")) == 0
    assert {(r.Z, r.A) for r in io.read_csv(out).rows} == {(1, 2), (6, 13)}
    with pytest.raises(SystemExit, match="--mass-numbers"):
        main(_lem4_args(tmp_path / "bad.csv", "--mass-numbers", "2"))


@pytest.mark.parametrize(
    ("choice", "class_name"),
    [
        ("default", "DefaultStoppingPower"),
        ("bethe", "SimpleBetheStoppingPower"),
        ("icru90-procedure", "ICRU90Procedure"),
        ("icru2016", "ICRU"),
        ("icru2005", "ICRU"),
        ("xstar", "XSTAR"),
        ("ns1970", "NorthcliffeSchillingWithBetheTail"),
    ],
)
def test_stopping_power_choices_resolve(choice: str, class_name: str) -> None:
    from pyradbio.cli import _resolve_stopping_power

    assert type(_resolve_stopping_power(choice)).__name__ == class_name
    assert _resolve_stopping_power(None) is None


def test_icru_choices_select_the_edition() -> None:
    from pyradbio.cli import _resolve_stopping_power

    assert _resolve_stopping_power("icru2009").edition == "2009"  # type: ignore[union-attr]


def test_table_records_the_stopping_power_settings(tmp_path) -> None:  # type: ignore[no-untyped-def]
    """Regression: the three ICRU editions must not all serialize as ``{"type": "ICRU",
    "name": "icru"}`` although they give different alphas; the header carries the
    resolved settings, edition included."""
    configs = {}
    for choice in ("icru2016", "icru2009", "icru2005"):
        out = tmp_path / f"{choice}.csv"
        assert main(_lem4_args(out, "--stopping-power", choice)) == 0
        sp = io.read_csv(out).metadata["model_config"]["stopping_power"]  # type: ignore[index]
        assert sp["type"] == "ICRU" and sp["settings"]["edition"] == choice[4:]
        assert sp["settings"]["quantity"] == "electronic"
        configs[choice] = json.dumps(sp, sort_keys=True)
    assert len(set(configs.values())) == 3
    out = tmp_path / "default.csv"
    assert main(_lem4_args(out)) == 0
    sp = io.read_csv(out).metadata["model_config"]["stopping_power"]  # type: ignore[index]
    assert sp["type"] == "DefaultStoppingPower"
    branches = sp["settings"]["branches"]
    assert branches["icru90_procedure"]["Z"] == [1, 18]
    assert branches["icru90_procedure"]["i_value_eV"] == 78.0
    assert branches["hubert1990"]["Z"] == [19, 92]
    assert branches["hubert1990"]["effective_charge"]["target_Z"] == 8.0
    assert "anchored" in sp["settings"]["fallback"]["anchoring"]


def test_table_headers_differ_when_an_effective_charge_parameter_differs() -> None:
    """Regression: band_gap_eV 0 vs 78 changes carbon alpha (1.33432 vs 1.33373), so it must
    change the header."""
    from pyradbio.core.cell import CellTarget
    from pyradbio.core.photon import PhotonParameters
    from pyradbio.models.lem.lem1 import LEM1
    from pyradbio.tables.alpha_beta import model_provenance
    from pyradbio.track.stopping import Bethe
    from pyradbio.track.stopping.effective_charge import BrandtKitagawaEffectiveCharge

    headers = {
        json.dumps(
            model_provenance(
                LEM1(
                    photon=PhotonParameters(alpha_x=0.1, beta_x=0.05, D_t=30.0),
                    cell=CellTarget(nucleus_radius_um=5.0),
                    stopping_power=Bethe(
                        effective_charge=BrandtKitagawaEffectiveCharge(band_gap_eV=gap)
                    ),
                )
            ),
            sort_keys=True,
        )
        for gap in (0.0, 78.0)
    }
    assert len(headers) == 2


def test_table_mkm_runs_on_the_native_backend(tmp_path) -> None:  # type: ignore[no-untyped-def]
    """``--model mkm`` needs no pyMKM: it builds MKMModel on NativeMKMBackend."""
    out = tmp_path / "mkm.csv"
    rc = main(
        [
            "table",
            "--model",
            "mkm",
            "--alpha0",
            "0.172",
            "--beta0",
            "0.0615",
            "--rd",
            "0.32",
            "--rn",
            "3.9",
            "--ions",
            "6",
            "--energies",
            "100",
            "--stopping-power",
            "bethe",
            "--out",
            str(out),
        ]
    )
    assert rc == 0
    meta = out.read_text(encoding="utf-8").splitlines()[0]
    assert '"stopping_power": {"type": "SimpleBetheStoppingPower", "settings": {' in meta
    assert '"radial_dose": null' in meta  # NativeMKMBackend fields
