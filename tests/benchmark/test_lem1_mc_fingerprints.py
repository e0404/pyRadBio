"""Numeric fingerprints and loaders of the committed LEM I Monte-Carlo checkpoints.

Every committed MC artefact stores, in its ``# key: value``
provenance header, 1-2 points recomputed at a tiny sample size with
``engine="numpy"`` and fixed seeds. These fast tests recompute them; a failure
means the code no longer computes what the committed rows were computed with, and
the message says to regenerate with ``scripts/lem1_mc_reference.py``.

The second part proves the readers the validation scripts use return exactly what
the readers of the original LEM I reconstruction script (frozen below) returned
on the same data rows, i.e. that the header lines change nothing a reader sees. The
last part runs the ``alpha-beta`` and ``z-slope`` write paths of the generator on a
copy of the committed file at a tiny sample size (``sk1996-final-slope`` has its own
end-to-end test in ``test_scholz_kraft1996_mc.py``).
"""

from __future__ import annotations

import csv
import functools
import importlib.util
import shutil
import warnings
from pathlib import Path
from types import ModuleType
from typing import Any

import numpy as np
import pytest

from pyradbio.benchmark import scholz1997 as s97
from pyradbio.benchmark import scholz_kraft1996 as sk
from pyradbio.benchmark.checkpoint import (
    FINGERPRINT_KEY,
    CsvCheckpoint,
    assert_fingerprint,
    read_header,
)
from pyradbio.benchmark.stopping import select_stopping_power

REPO = Path(__file__).resolve().parents[2]
DATA = REPO / "validation" / "data"
MC_RESULTS = DATA / "mc_results.csv"
SK1996 = DATA / "sk1996_mc_observables.csv"
Z_SLOPE = DATA / "sk1997_z_slope.csv"
GENERATOR = "lem1_mc_reference"


# The generator's --stopping-power choice behind each row's ``stopping_power`` name.
_CHOICE = {"ns1970+bethe": "ns1970", "bethe": "bethe"}


def _recorded(path: Path) -> Any:
    header = read_header(path)
    assert header["generator"].startswith(f"scripts/{GENERATOR}.py ")
    return header.get(FINGERPRINT_KEY)


def _recorded_choice(recorded: Any) -> str:
    """The stopping power the fingerprint was computed at (each entry records it)."""
    (name,) = {entry["stopping_power"] for entry in recorded}
    return _CHOICE[name]


def test_mc_results_fingerprint() -> None:
    """``mc_results.csv`` (``alpha-beta``): carbon 6 and 70 MeV/u, 16 / 8 histories, seed 1.

    Carbon 70 MeV/u is above ``KELLERER_B_CUT_MIN_NBAR``, so the near/far split that
    produced the high-energy rows is gated too.
    """
    recorded = _recorded(MC_RESULTS)
    assert [entry["n_bar"] >= s97.KELLERER_B_CUT_MIN_NBAR for entry in recorded] == [False, True]
    recomputed = s97.mc_reference_fingerprint(select_stopping_power(_recorded_choice(recorded)))
    assert_fingerprint(recorded, recomputed, generator=GENERATOR)


def test_sk1996_mc_observables_fingerprint() -> None:
    """``sk1996_mc_observables.csv`` (``sk1996-final-slope``): carbon 8 MeV/u, 20 histories."""
    recorded = _recorded(SK1996)
    recomputed = sk.mc_observables_fingerprint(select_stopping_power(_recorded_choice(recorded)))
    assert_fingerprint(recorded, recomputed, generator=GENERATOR)


def test_sk1997_z_slope_fingerprint() -> None:
    """``sk1997_z_slope.csv`` (``z-slope``): carbon 8 MeV/u at reduced settings.

    While its data rows are the original ones the header must say they
    predate the fingerprint; a regeneration rewrites that line.
    """
    data_rows = read_header(Z_SLOPE).get("data_rows", "")
    if "the original z-slope batch script" in data_rows:
        assert "PREDATE" in data_rows
    with warnings.catch_warnings():
        warnings.simplefilter("ignore", UserWarning)
        recomputed = s97.z_slope_fingerprint()
    assert_fingerprint(_recorded(Z_SLOPE), recomputed, generator=GENERATOR)


def test_a_changed_fingerprint_names_the_generator() -> None:
    recorded = _recorded(SK1996)
    changed = [dict(recorded[0], sigma_1pct_um2=recorded[0]["sigma_1pct_um2"] * (1 + 1e-6))]
    with pytest.raises(AssertionError, match=r"regenerate with scripts/lem1_mc_reference\.py"):
        assert_fingerprint(recorded, changed, generator=GENERATOR)


# --- frozen readers of the original LEM I reconstruction script ---------------------


def _frozen_sk1996(path: Path) -> dict[str, list[dict[str, object]]]:
    text_columns = frozenset({"particle", "stopping_power"})
    grouped: dict[str, list[dict[str, object]]] = {}
    with path.open(newline="") as handle:
        for row in csv.DictReader(handle):
            grouped.setdefault(row["particle"], []).append(
                {
                    key: (value if key in text_columns else float(value))
                    for key, value in row.items()
                    if key != "particle" and value not in (None, "")
                }
            )
    for rows in grouped.values():
        rows.sort(key=lambda row: float(row["energy_MeV_u"]))  # type: ignore[arg-type]
    return grouped


def _frozen_mc(path: Path, active: str | None) -> tuple[dict[str, Any], dict[str, int]]:
    text_columns = frozenset({"particle", "sampling", "stopping_power"})
    grouped: dict[str, list[dict[str, float]]] = {}
    dropped: dict[str, int] = {}
    with path.open() as handle:
        for row in csv.DictReader(handle):
            stored = row.get("stopping_power")
            if active is not None and stored is not None and stored != active:
                dropped[stored] = dropped.get(stored, 0) + 1
                continue
            grouped.setdefault(row["particle"], []).append(
                {k: float(v) for k, v in row.items() if k not in text_columns}
            )
    for rows in grouped.values():
        rows.sort(key=lambda r: r["energy_MeV_u"])
    return grouped, dropped


def _frozen_z_slope(path: Path) -> dict[tuple[str, float], tuple[np.ndarray, np.ndarray]]:
    grouped: dict[tuple[str, float], list[tuple[float, float]]] = {}
    with path.open() as handle:
        for row in csv.DictReader(handle):
            key = (row["particle"], float(row["energy_MeV_u"]))
            grouped.setdefault(key, []).append((float(row["z_Gy"]), float(row["slope_Gy_inv"])))
    out = {}
    for key, pairs in grouped.items():
        pairs.sort()
        out[key] = (np.array([z for z, _ in pairs]), np.array([sl for _, sl in pairs]))
    return out


def _without_header(path: Path, tmp_path: Path) -> Path:
    """The data rows alone, as the file was before the provenance header."""
    raw = path.read_bytes()
    lines = raw.splitlines(keepends=True)
    assert lines[0].startswith(b"# ")  # the header is there ...
    plain = tmp_path / path.name
    plain.write_bytes(b"".join(line for line in lines if not line.startswith(b"#")))
    return plain


def test_sk1996_loader_matches_the_frozen_reader(tmp_path: Path) -> None:
    expected = _frozen_sk1996(_without_header(SK1996, tmp_path))
    assert sum(len(rows) for rows in expected.values()) == 40
    assert sk.load_mc_observables(SK1996) == expected


@pytest.mark.parametrize("active", ["ns1970+bethe", "bethe", None])
def test_mc_reference_loader_matches_the_frozen_reader(tmp_path: Path, active: str | None) -> None:
    rows, dropped = _frozen_mc(_without_header(MC_RESULTS, tmp_path), active)
    reference = s97.load_mc_reference(MC_RESULTS, stopping_power_name=active)
    assert reference.rows == rows
    assert reference.dropped == dropped
    assert sum(len(r) for r in rows.values()) + sum(dropped.values()) == 38


def test_z_slope_loader_matches_the_frozen_reader(tmp_path: Path) -> None:
    expected = _frozen_z_slope(_without_header(Z_SLOPE, tmp_path))
    loaded = s97.load_z_slope_reference(Z_SLOPE)
    assert loaded.keys() == expected.keys()
    assert len(loaded) == 8
    for key, (z, slope) in expected.items():
        np.testing.assert_array_equal(loaded[key][0], z)
        np.testing.assert_array_equal(loaded[key][1], slope)


def test_loaders_read_a_missing_file_as_empty(tmp_path: Path) -> None:
    missing = tmp_path / "absent.csv"
    assert sk.load_mc_observables(missing) == {}
    assert s97.load_mc_reference(missing) == s97.MCReference()
    assert s97.load_z_slope_reference(missing) == {}


# --- the generator's write paths (alpha-beta, z-slope) --------------------------------


def _generator() -> ModuleType:
    spec = importlib.util.spec_from_file_location(
        "_lem1_mc_reference_write_paths", REPO / "scripts" / "lem1_mc_reference.py"
    )
    assert spec is not None and spec.loader is not None
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


def _data_lines(path: Path) -> list[bytes]:
    lines = path.read_bytes().splitlines(keepends=True)
    return [line for line in lines if not line.startswith(b"#")]


def _stale(recorded: list[dict[str, Any]]) -> list[dict[str, Any]]:
    """The recorded fingerprint with one number nudged: what changed code reproduces."""
    first = dict(recorded[0])
    key = next(k for k, v in first.items() if isinstance(v, float))
    first[key] *= 1.0 + 1e-6
    return [first, *recorded[1:]]


def test_alpha_beta_write_path(tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> None:
    """``alpha-beta --only carbon:6`` on a copy: one row replaced, 37 byte-identical.

    The fingerprint is a stand-in (the real one is recomputed above), so the run costs
    one point at ``--n-total-scale 1e-4`` (30 histories: 8 streams x 3 run). Then a
    stale fingerprint at the same stopping power is refused with the warning that
    ``--fresh`` discards every point, and the file is left as it was.
    """
    out = tmp_path / MC_RESULTS.name
    shutil.copyfile(MC_RESULTS, out)
    recorded = read_header(out)[FINGERPRINT_KEY]
    monkeypatch.setattr(s97, "mc_reference_fingerprint", lambda _sp: recorded)
    generator = _generator()
    argv = ["alpha-beta", "--only", "carbon:6", "--n-total-scale", "1e-4", "--engine", "numpy"]

    assert generator.main([*argv, "--out", str(tmp_path)]) == 0
    before, after = _data_lines(MC_RESULTS), _data_lines(out)
    assert len(after) == len(before) == 39  # column row + 38 points
    changed = [after[i] for i in range(len(after)) if after[i] != before[i]]
    assert [line.split(b",")[:2] for line in changed] == [[b"carbon", b"6.0"]]
    row = CsvCheckpoint(out, text_columns=s97.MC_REFERENCE_TEXT_COLUMNS).record(("carbon", 6.0))
    assert row["n_realizations"] == 24.0  # 8 streams x (30 // 8)
    assert row["stopping_power"] == "ns1970+bethe"
    header = read_header(out)
    assert header["generator"] == "scripts/lem1_mc_reference.py alpha-beta"
    assert header["stopping_power"] == "ns1970+bethe"
    assert header["data_rows"].startswith(
        "mixed: some points regenerated by scripts/lem1_mc_reference.py alpha-beta "
        "(the last run: carbon:6); the others: written by the original LEM I batch script"
    )
    assert header[FINGERPRINT_KEY] == recorded

    stale = _stale(recorded)
    monkeypatch.setattr(s97, "mc_reference_fingerprint", lambda _sp: stale)
    with pytest.raises(SystemExit, match=r"discards EVERY existing point of mc_results\.csv"):
        generator.main([*argv, "--out", str(tmp_path)])
    assert _data_lines(out) == after


def test_z_slope_write_path(tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> None:
    """``z-slope --only carbon:8`` on a copy, at the fingerprint's reduced settings."""
    out = tmp_path / Z_SLOPE.name
    shutil.copyfile(Z_SLOPE, out)
    recorded = read_header(out)[FINGERPRINT_KEY]
    monkeypatch.setattr(s97, "z_slope_fingerprint", lambda: recorded)
    settings = {k: v for k, v in s97.Z_SLOPE_FINGERPRINT_SETTINGS.items() if k != "seed"}
    monkeypatch.setattr(s97, "z_slope_point", functools.partial(s97.z_slope_point, **settings))
    generator = _generator()
    argv = ["z-slope", "--only", "carbon:8", "--engine", "numpy", "--out", str(tmp_path)]

    with warnings.catch_warnings():
        warnings.simplefilter("ignore", UserWarning)
        assert generator.main(argv) == 0
    loaded, committed = s97.load_z_slope_reference(out), s97.load_z_slope_reference(Z_SLOPE)
    assert loaded.keys() == committed.keys()
    assert loaded[("carbon", 8.0)][0].size == len(recorded[0]["z_Gy"])

    def untouched(path: Path) -> list[bytes]:
        return [line for line in _data_lines(path) if not line.startswith(b"carbon,8.0,")]

    assert untouched(out) == untouched(Z_SLOPE)
    header = read_header(out)
    assert header["data_rows"].startswith(
        "mixed: some points regenerated by scripts/lem1_mc_reference.py z-slope "
        "(the last run: carbon:8); the others: PREDATE the fingerprint"
    )
    assert header["stopping_power"].startswith("mixed: LEM1 default")
    assert "the others: the LEM1 default at generation time" in header["stopping_power"]

    monkeypatch.setattr(s97, "z_slope_fingerprint", lambda: _stale(recorded))
    with pytest.raises(SystemExit, match=r"discards EVERY existing point of sk1997_z_slope\.csv"):
        generator.main(argv)


def test_another_stopping_power_is_not_refused() -> None:
    """A fingerprint recorded at another LET does not gate this run (rows are per LET)."""
    generator = _generator()
    recorded = [{"stopping_power": "ns1970+bethe", "alpha": 1.0}]
    assert generator._same_let(recorded, "ns1970+bethe")
    assert not generator._same_let(recorded, "bethe")
    assert generator._same_let([{"alpha_z": 1.0}], None)  # z-slope: no LET entry
