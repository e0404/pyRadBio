"""Checkpoint I/O (``pyradbio.benchmark.checkpoint``): byte-exact round trips.

The batch scripts' checkpoints are a provenance contract: ``scripts/`` writes them,
``validation/`` and the ``benchmark.<paper>`` loaders read them. These tests pin
the shared reader/writer to the files those scripts actually produced, byte for
byte, and pin ``config_key`` to the keys already stored in the Survival-comparison
MC cache. The
three LEM I batch scripts were merged into ``scripts/lem1_mc_reference.py``, which
writes through ``CsvCheckpoint`` itself; the readers and writers of the original
scripts are kept below as a frozen copy, so these comparisons stay real tests.
"""

from __future__ import annotations

import argparse
import csv
import hashlib
import json
import math
import shutil
from dataclasses import replace
from pathlib import Path
from typing import Any, ClassVar

import pytest

from pyradbio import CellTarget, Ion, PhotonParameters
from pyradbio.benchmark.checkpoint import (
    FINGERPRINT_KEY,
    PROVENANCE_KEY,
    CsvCheckpoint,
    JsonPointCache,
    assert_fingerprint,
    config_key,
    parse_point_spec,
    read_header,
)
from pyradbio.models.lem.lem1 import LEM1
from pyradbio.models.lem.lem2 import LEM2
from pyradbio.models.lem.lem3 import LEM3
from pyradbio.track.scholz_kraft import R_MAX_GAMMA_UM, ScholzKraftRadialDose

REPO = Path(__file__).resolve().parents[2]
DATA_DIR = REPO / "validation" / "data"


def _data_file(name: str) -> Path:
    path = DATA_DIR / name
    if not path.is_file():
        pytest.skip(f"{name} not found under {DATA_DIR}")
    return path


def _script(name: str) -> dict[str, Any]:
    """The script's namespace, executed with its ``__main__`` guard closed.

    The three LEM I batch scripts no longer exist as files; their frozen
    readers/writers (:data:`_FROZEN_LEM1_BATCHES`) stand in for them.
    """
    return _FROZEN_LEM1_BATCHES[name]


def _data_only(path: Path, tmp_path: Path) -> Path:
    """A copy of ``path`` without its ``#`` provenance lines (what the old readers read)."""
    lines = path.read_bytes().splitlines(keepends=True)
    plain = tmp_path / "data_only" / path.name
    plain.parent.mkdir(exist_ok=True)
    plain.write_bytes(b"".join(line for line in lines if not line.startswith(b"#")))
    return plain


# --- frozen copy of the original LEM I batch scripts' checkpoint I/O ---------------
# The field names, text columns and the merge/parse/load/write functions of the three
# original scripts (alpha/beta, z-slope, SK1996 final slope), verbatim.

_S97_FIELDNAMES = [
    "particle",
    "energy_MeV_u",
    "error_estimator_version",
    "alpha",
    "alpha_err",
    "beta",
    "beta_err",
    "n_realizations",
    "n_streams",
    "n_bar",
    "n_field_points",
    "alpha_cross_section",
    "alpha_rel_dev",
    "two_track_beta",
    "sampling",
    "stopping_power",
]


def _s97_write_merged(path: Path, rows: dict[tuple[str, float], dict[str, float | str]]) -> None:
    """Write all rows sorted by particle then energy."""
    ordered = sorted(rows.values(), key=lambda r: (r["particle"], r["energy_MeV_u"]))
    with path.open("w", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=_S97_FIELDNAMES)
        writer.writeheader()
        writer.writerows(ordered)


def _parse_only(spec: str) -> set[tuple[str, float]]:
    """Parse ``--only carbon:20,proton:6`` into a filter set."""
    out: set[tuple[str, float]] = set()
    for token in spec.split(","):
        particle, energy = token.split(":")
        out.add((particle.strip(), float(energy)))
    return out


_ZSLOPE_FIELDNAMES = [
    "particle",
    "energy_MeV_u",
    "z_Gy",
    "log_kill",
    "slope_Gy_inv",
    "counts",
    "realizations_per_mean",
    "n_field_points",
    "b_cut_um",
    "log_kill_err",
    "alpha_z",
    "beta_z",
]


def _zslope_load_existing(path: Path) -> dict[tuple[str, float], list[dict[str, str]]]:
    """Existing rows grouped by (particle, energy) so points can be replaced."""
    rows: dict[tuple[str, float], list[dict[str, str]]] = {}
    if not path.exists():
        return rows
    with path.open() as handle:
        for row in csv.DictReader(handle):
            rows.setdefault((row["particle"], float(row["energy_MeV_u"])), []).append(row)
    return rows


def _zslope_write_merged(path: Path, rows: dict[tuple[str, float], list[dict[str, Any]]]) -> None:
    with path.open("w", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=_ZSLOPE_FIELDNAMES)
        writer.writeheader()
        for key in sorted(rows, key=lambda k: (k[0], k[1])):
            writer.writerows(rows[key])


_SK1996_SPECIES = {
    "proton": (1, 1),
    "helium": (2, 4),
    "carbon": (6, 12),
    "neon": (10, 20),
    "argon": (18, 40),
    "krypton": (36, 84),
    "xenon": (54, 132),
    "uranium": (92, 238),
}
_SK1996_FIELDNAMES = [
    "particle",
    "estimator_version",
    "stopping_power",
    "energy_MeV_u",
    "let_keV_um",
    "dose_1pct_Gy",
    "slope_1pct_Gy_inv",
    "sigma_1pct_um2",
    "sigma_1pct_err_um2",
    "sigma_low_dose_um2",
    "dose_80pct_Gy",
    "rbe_80pct",
    "rbe_80pct_err",
    "dose_20pct_Gy",
    "rbe_20pct",
    "rbe_20pct_err",
    "rbe_1pct",
    "rbe_1pct_err",
    "n_realizations",
    "max_relative_error",
    "target_relative_error",
    "precision_converged",
]


def _sk1996_parse_only(spec: str) -> set[tuple[str, float]]:
    """Parse ``carbon:3,argon:8`` into selected particle/energy pairs."""
    selected: set[tuple[str, float]] = set()
    for token in spec.split(","):
        particle, energy = token.split(":")
        particle = particle.strip().lower()
        if particle not in _SK1996_SPECIES:
            raise argparse.ArgumentTypeError(f"unknown particle {particle!r}")
        selected.add((particle, float(energy)))
    return selected


def _sk1996_load_existing(path: Path) -> dict[tuple[str, float], dict[str, Any]]:
    rows: dict[tuple[str, float], dict[str, Any]] = {}
    if not path.exists():
        return rows
    with path.open(newline="") as handle:
        for row in csv.DictReader(handle):
            parsed = {
                key: (value if key in {"particle", "stopping_power"} else float(value))
                for key, value in row.items()
                if value is not None and value != ""
            }
            rows[(str(parsed["particle"]), float(parsed["energy_MeV_u"]))] = parsed
    return rows


def _sk1996_write_results(path: Path, rows: dict[tuple[str, float], dict[str, Any]]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    ordered = sorted(rows.values(), key=lambda row: (str(row["particle"]), row["energy_MeV_u"]))
    with path.open("w", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=_SK1996_FIELDNAMES)
        writer.writeheader()
        writer.writerows(ordered)


_FROZEN_LEM1_BATCHES: dict[str, dict[str, Any]] = {
    "mc_batch_scholz1997": {
        "FIELDNAMES": _S97_FIELDNAMES,
        "TEXT_COLUMNS": frozenset({"particle", "sampling", "stopping_power"}),
        "write_merged": _s97_write_merged,
        "parse_only": _parse_only,
    },
    "mc_batch_scholz1997_zslope": {
        "FIELDNAMES": _ZSLOPE_FIELDNAMES,
        "TEXT_COLUMNS": frozenset({"particle"}),
        "load_existing": _zslope_load_existing,
        "write_merged": _zslope_write_merged,
        "parse_only": _parse_only,
    },
    "mc_batch_scholz_kraft1996": {
        "SPECIES": _SK1996_SPECIES,
        "FIELDNAMES": _SK1996_FIELDNAMES,
        "parse_only": _sk1996_parse_only,
        "load_existing": _sk1996_load_existing,
        "write_results": _sk1996_write_results,
    },
}
# end of the frozen copy


def test_frozen_fieldnames_are_the_generators_columns() -> None:
    """The frozen writers' columns are what ``scripts/lem1_mc_reference.py`` writes."""
    from pyradbio.benchmark import scholz1997 as s97
    from pyradbio.benchmark import scholz_kraft1996 as sk

    assert tuple(_S97_FIELDNAMES) == s97.MC_REFERENCE_FIELDS
    assert tuple(_ZSLOPE_FIELDNAMES) == s97.Z_SLOPE_FIELDS
    assert tuple(_SK1996_FIELDNAMES) == sk.MC_OBSERVABLE_FIELDS
    assert _SK1996_SPECIES == dict(sk.SPECIES)


def _our_mc_cache() -> Path:
    """The Survival-comparison MC cache (``scripts/lem_mc_vs_survival.py`` writes it)."""
    return _data_file("survival_our_mc.json")


# (checkpoint file, generating script, the script's text columns). The text
# columns are copied from each script's own TEXT_COLUMNS / load_existing.
CSV_FILES = (
    ("mc_results.csv", "mc_batch_scholz1997", ("particle", "sampling", "stopping_power")),
    ("sk1996_mc_observables.csv", "mc_batch_scholz_kraft1996", ("particle", "stopping_power")),
    ("sk1997_z_slope.csv", "mc_batch_scholz1997_zslope", ("particle",)),
)


def _copy(src: Path, tmp_path: Path) -> Path:
    dst = tmp_path / src.name
    shutil.copyfile(src, dst)
    return dst


# --------------------------------------------------------------------------- CSV


class TestCsvRoundTrip:
    @pytest.mark.parametrize(("name", "script", "text"), CSV_FILES)
    def test_file_header_round_trip_is_byte_identical(
        self, tmp_path: Path, name: str, script: str, text: tuple[str, ...]
    ) -> None:
        original = _data_file(name)
        copy = _copy(original, tmp_path)
        checkpoint = CsvCheckpoint(copy, text_columns=text)
        checkpoint.save()
        assert copy.read_bytes() == original.read_bytes()

    @pytest.mark.parametrize(("name", "script", "text"), CSV_FILES)
    def test_script_fieldnames_round_trip_is_byte_identical(
        self, tmp_path: Path, name: str, script: str, text: tuple[str, ...]
    ) -> None:
        original = _data_file(name)
        fieldnames = _script(script)["FIELDNAMES"]
        copy = _copy(original, tmp_path)
        CsvCheckpoint(copy, fieldnames, text_columns=text).save()
        assert copy.read_bytes() == original.read_bytes()

    @pytest.mark.parametrize(("name", "script", "text"), CSV_FILES)
    def test_putting_every_point_back_is_byte_identical(
        self, tmp_path: Path, name: str, script: str, text: tuple[str, ...]
    ) -> None:
        """Replacing each point with its own stored rows changes nothing."""
        original = _data_file(name)
        copy = _copy(original, tmp_path)
        checkpoint = CsvCheckpoint(copy, text_columns=text)
        for key in checkpoint.keys():
            checkpoint.put_group(checkpoint.rows(key))
        checkpoint.save()
        assert copy.read_bytes() == original.read_bytes()

    def test_crlf_and_lf_files_both_keep_their_line_endings(self, tmp_path: Path) -> None:
        body = "particle,energy_MeV_u,alpha{nl}carbon,1.0,0.5{nl}proton,2.0,0.25{nl}"
        for nl in ("\r\n", "\n"):
            path = tmp_path / f"f{len(nl)}.csv"
            path.write_bytes(body.format(nl=nl).encode())
            checkpoint = CsvCheckpoint(path)
            assert checkpoint.lineterminator == nl
            checkpoint.save()
            assert path.read_bytes() == body.format(nl=nl).encode()

    def test_new_file_uses_the_csv_module_default_terminator(self, tmp_path: Path) -> None:
        """The scripts' ``csv.DictWriter`` wrote ``\\r\\n`` on every platform."""
        path = tmp_path / "sub" / "new.csv"
        checkpoint = CsvCheckpoint(path, ["particle", "energy_MeV_u", "alpha"])
        assert len(checkpoint) == 0
        checkpoint.put({"particle": "carbon", "energy_MeV_u": 1.0, "alpha": 0.5})
        checkpoint.save()
        assert path.read_bytes() == b"particle,energy_MeV_u,alpha\r\ncarbon,1.0,0.5\r\n"

    def test_save_without_fieldnames_needs_a_file_header(self, tmp_path: Path) -> None:
        checkpoint = CsvCheckpoint(tmp_path / "none.csv")
        with pytest.raises(ValueError, match="fieldnames"):
            checkpoint.save()

    def test_len_counts_points_not_rows(self, tmp_path: Path) -> None:
        """``sk1997_z_slope.csv`` holds several rows per point."""
        original = _data_file("sk1997_z_slope.csv")
        checkpoint = CsvCheckpoint(_copy(original, tmp_path))
        with original.open(newline="") as handle:
            rows = list(csv.DictReader(line for line in handle if not line.startswith("#")))
        assert len(checkpoint) == len({(r["particle"], float(r["energy_MeV_u"])) for r in rows})
        assert len(checkpoint) < len(rows)

    def test_file_without_a_newline_gets_the_csv_default(self, tmp_path: Path) -> None:
        path = tmp_path / "bare.csv"
        path.write_bytes(b"particle,energy_MeV_u")
        checkpoint = CsvCheckpoint(path)
        assert checkpoint.lineterminator == "\r\n"
        assert len(checkpoint) == 0
        checkpoint.save()
        assert path.read_bytes() == b"particle,energy_MeV_u\r\n"

    def test_failed_save_leaves_the_file_intact(self, tmp_path: Path) -> None:
        """The rewrite goes through a temporary file, so a crash mid-save loses nothing."""
        path = tmp_path / "c.csv"
        path.write_bytes(b"particle,energy_MeV_u\r\ncarbon,1.0\r\n")
        checkpoint = CsvCheckpoint(path)
        checkpoint.put({"particle": "argon", "energy_MeV_u": 2.0, "not_a_column": 1})  # sorts first
        with pytest.raises(ValueError, match="not_a_column"):
            checkpoint.save()
        assert path.read_bytes() == b"particle,energy_MeV_u\r\ncarbon,1.0\r\n"
        assert sorted(p.name for p in tmp_path.iterdir()) == ["c.csv"]


class TestCsvDuplicatePoints:
    """A point stored twice: kept as a group, or de-duplicated like the scripts."""

    BODY = "particle,energy_MeV_u,alpha\r\ncarbon,1.0,0.1\r\nproton,1.0,0.3\r\ncarbon,1.0,0.2\r\n"

    def test_default_keeps_every_row(self, tmp_path: Path) -> None:
        path = tmp_path / "d.csv"
        path.write_bytes(self.BODY.encode())
        checkpoint = CsvCheckpoint(path)
        assert [r["alpha"] for r in checkpoint.records(("carbon", 1.0))] == [0.1, 0.2]
        with pytest.raises(ValueError, match="2 rows"):
            checkpoint.record(("carbon", 1.0))

    def test_one_row_per_point_keeps_the_last_row(self, tmp_path: Path) -> None:
        """The dict-based ``load_existing`` of the single-row scripts: last row wins."""
        path = tmp_path / "d.csv"
        path.write_bytes(self.BODY.encode())
        checkpoint = CsvCheckpoint(path, one_row_per_point=True)
        assert checkpoint.record(("carbon", 1.0))["alpha"] == 0.2
        assert len(checkpoint) == 2
        checkpoint.save()
        assert path.read_bytes() == (
            b"particle,energy_MeV_u,alpha\r\ncarbon,1.0,0.2\r\nproton,1.0,0.3\r\n"
        )

    def test_one_row_per_point_rejects_groups(self, tmp_path: Path) -> None:
        checkpoint = CsvCheckpoint(
            tmp_path / "d.csv", ["particle", "energy_MeV_u"], one_row_per_point=True
        )
        row = {"particle": "carbon", "energy_MeV_u": 1.0}
        with pytest.raises(ValueError, match="one row per point"):
            checkpoint.put_group([row, row])
        checkpoint.put_group([row])
        assert len(checkpoint) == 1


class TestCsvWriterMatchesTheScripts:
    """The shared writer emits exactly what each script's own writer emitted."""

    def _parsed_rows(self, path: Path, text: tuple[str, ...]) -> list[dict[str, Any]]:
        # mc_batch_scholz1997.load_existing's parse, minus its model-derived
        # backfills (every committed row already carries those columns).
        with path.open(newline="") as handle:
            return [
                {k: (v if k in text else float(v)) for k, v in row.items() if v != ""}
                for row in csv.DictReader(handle)
            ]

    def test_scholz1997_write_merged(self, tmp_path: Path) -> None:
        original = _data_only(_data_file("mc_results.csv"), tmp_path)
        ns = _script("mc_batch_scholz1997")
        text = tuple(ns["TEXT_COLUMNS"])
        rows = self._parsed_rows(original, text)
        theirs = tmp_path / "theirs.csv"
        ns["write_merged"](theirs, {(r["particle"], r["energy_MeV_u"]): r for r in rows})
        ours = CsvCheckpoint(tmp_path / "ours.csv", ns["FIELDNAMES"], text_columns=text)
        for row in reversed(rows):  # insertion order must not matter
            ours.put(row)
        ours.save()
        assert (tmp_path / "ours.csv").read_bytes() == theirs.read_bytes()
        assert theirs.read_bytes().replace(b"\r\n", b"\n") == original.read_bytes().replace(
            b"\r\n", b"\n"
        )

    def test_sk1996_write_results(self, tmp_path: Path) -> None:
        original = _data_only(_data_file("sk1996_mc_observables.csv"), tmp_path)
        ns = _script("mc_batch_scholz_kraft1996")
        rows = ns["load_existing"](original)
        theirs = tmp_path / "theirs.csv"
        ns["write_results"](theirs, rows)
        ours = CsvCheckpoint(
            tmp_path / "ours.csv", ns["FIELDNAMES"], text_columns=("particle", "stopping_power")
        )
        for row in rows.values():
            ours.put(row)
        ours.save()
        assert (tmp_path / "ours.csv").read_bytes() == theirs.read_bytes()

    def test_sk1996_untouched_rows_keep_their_integer_spelling(self, tmp_path: Path) -> None:
        """The script's own reload turned ``3`` into ``3.0``; the checkpoint does not."""
        original = _data_only(_data_file("sk1996_mc_observables.csv"), tmp_path)
        ns = _script("mc_batch_scholz_kraft1996")
        theirs = tmp_path / "theirs.csv"
        ns["write_results"](theirs, ns["load_existing"](original))
        if theirs.read_bytes() == original.read_bytes():
            pytest.skip("the script's own round trip is byte-exact for this file")
        copy = _copy(original, tmp_path)
        CsvCheckpoint(copy, ns["FIELDNAMES"]).save()
        assert copy.read_bytes() == original.read_bytes()

    def test_zslope_write_merged(self, tmp_path: Path) -> None:
        original = _data_only(_data_file("sk1997_z_slope.csv"), tmp_path)
        ns = _script("mc_batch_scholz1997_zslope")
        existing = ns["load_existing"](original)
        theirs = tmp_path / "theirs.csv"
        ns["write_merged"](theirs, existing)
        ours = CsvCheckpoint(tmp_path / "ours.csv", ns["FIELDNAMES"])
        for group in reversed(list(existing.values())):
            ours.put_group(group)
        ours.save()
        assert (tmp_path / "ours.csv").read_bytes() == theirs.read_bytes()


class TestCsvAccess:
    FIELDS = ("particle", "energy_MeV_u", "alpha", "sampling")

    def _checkpoint(self, tmp_path: Path) -> CsvCheckpoint:
        path = tmp_path / "c.csv"
        path.write_text(
            "particle,energy_MeV_u,alpha,sampling\n"
            "carbon,2.0,0.8,kellerer\n"
            "carbon,1.0,0.6,\n"
            "proton,1.0,1.5,kellerer\n",
            newline="",
        )
        return CsvCheckpoint(path, self.FIELDS, text_columns=("particle", "sampling"))

    def test_keys_are_particle_and_float_energy_sorted(self, tmp_path: Path) -> None:
        checkpoint = self._checkpoint(tmp_path)
        assert checkpoint.keys() == [("carbon", 1.0), ("carbon", 2.0), ("proton", 1.0)]
        assert ("carbon", 2.0) in checkpoint
        assert ("carbon", 3.0) not in checkpoint
        assert len(checkpoint) == 3

    def test_rows_are_the_stored_strings(self, tmp_path: Path) -> None:
        checkpoint = self._checkpoint(tmp_path)
        assert checkpoint.rows(("carbon", 1.0)) == [
            {"particle": "carbon", "energy_MeV_u": "1.0", "alpha": "0.6", "sampling": ""}
        ]

    def test_record_parses_numbers_and_drops_empty_cells(self, tmp_path: Path) -> None:
        """The parse of mc_batch_scholz1997.load_existing / sk1996 load_existing."""
        checkpoint = self._checkpoint(tmp_path)
        assert checkpoint.record(("carbon", 1.0)) == {
            "particle": "carbon",
            "energy_MeV_u": 1.0,
            "alpha": 0.6,
        }
        assert checkpoint.record(("proton", 1.0))["sampling"] == "kellerer"
        assert [r["energy_MeV_u"] for r in checkpoint.iter_records()] == [1.0, 2.0, 1.0]

    def test_record_needs_exactly_one_row(self, tmp_path: Path) -> None:
        checkpoint = self._checkpoint(tmp_path)
        checkpoint.put_group(
            [
                {"particle": "carbon", "energy_MeV_u": 1.0, "alpha": 0.1},
                {"particle": "carbon", "energy_MeV_u": 1.0, "alpha": 0.2},
            ]
        )
        assert [r["alpha"] for r in checkpoint.records(("carbon", 1.0))] == [0.1, 0.2]
        with pytest.raises(ValueError, match="2 rows"):
            checkpoint.record(("carbon", 1.0))

    def test_put_replaces_and_discard_removes(self, tmp_path: Path) -> None:
        checkpoint = self._checkpoint(tmp_path)
        key = checkpoint.put({"particle": "carbon", "energy_MeV_u": 2, "alpha": 0.9})
        assert key == ("carbon", 2.0)
        assert checkpoint.record(key)["alpha"] == 0.9
        checkpoint.discard(key)
        checkpoint.discard(key)  # idempotent, like dict.pop(key, None)
        assert key not in checkpoint

    def test_put_group_rejects_empty_and_mixed_points(self, tmp_path: Path) -> None:
        checkpoint = self._checkpoint(tmp_path)
        with pytest.raises(ValueError, match="at least one"):
            checkpoint.put_group([])
        with pytest.raises(ValueError, match="one point"):
            checkpoint.put_group(
                [
                    {"particle": "carbon", "energy_MeV_u": 1.0},
                    {"particle": "carbon", "energy_MeV_u": 2.0},
                ]
            )

    def test_custom_key_columns(self, tmp_path: Path) -> None:
        path = tmp_path / "k.csv"
        checkpoint = CsvCheckpoint(
            path, ["cell", "let", "x"], key_columns=("cell",), text_columns=("cell",)
        )
        checkpoint.put({"cell": "V79", "let": 10.0, "x": 1})
        assert checkpoint.keys() == [("V79",)]


# ------------------------------------------------------------------ CSV headers


class TestCsvHeader:
    FINGERPRINT: ClassVar[list[dict[str, Any]]] = [
        {"particle": "carbon", "energy_MeV_u": 6.0, "seed": 1, "alpha": 0.1 + 0.2, "beta": -1e-17}
    ]

    def test_header_round_trip_is_byte_identical(self, tmp_path: Path) -> None:
        path = tmp_path / "h.csv"
        checkpoint = CsvCheckpoint(path, ["particle", "energy_MeV_u", "alpha"])
        checkpoint.header["generator"] = "scripts/lem1_mc_reference.py"
        checkpoint.header["stopping_power"] = "ns1970+bethe"
        checkpoint.header[FINGERPRINT_KEY] = self.FINGERPRINT
        checkpoint.put({"particle": "carbon", "energy_MeV_u": 1.0, "alpha": 0.5})
        checkpoint.save()
        first = path.read_bytes()
        assert first.startswith(b"# generator: scripts/lem1_mc_reference.py\r\n")
        assert b"\r\n# fingerprint: [{" in first

        reread = CsvCheckpoint(path)
        assert reread.header == checkpoint.header
        assert reread.record(("carbon", 1.0))["alpha"] == 0.5
        reread.save()
        assert path.read_bytes() == first

    def test_read_header_decodes_the_fingerprint_exactly(self, tmp_path: Path) -> None:
        path = tmp_path / "h.csv"
        checkpoint = CsvCheckpoint(path, ["particle", "energy_MeV_u"])
        checkpoint.header["engine"] = "numpy"
        checkpoint.header[FINGERPRINT_KEY] = self.FINGERPRINT
        checkpoint.save()
        header = read_header(path)
        assert header == {"engine": "numpy", FINGERPRINT_KEY: self.FINGERPRINT}
        assert header[FINGERPRINT_KEY][0]["alpha"] == 0.1 + 0.2  # float repr round trip

    def test_header_lines_are_skipped_by_the_reader(self, tmp_path: Path) -> None:
        path = tmp_path / "h.csv"
        path.write_text("# a: 1\n#b:two words\nparticle,energy_MeV_u\ncarbon,3\n", newline="")
        checkpoint = CsvCheckpoint(path)
        assert checkpoint.header == {"a": "1", "b": "two words"}
        assert checkpoint.fieldnames == ["particle", "energy_MeV_u"]
        assert checkpoint.keys() == [("carbon", 3.0)]

    def test_hand_written_headers_are_canonicalised_on_save(self, tmp_path: Path) -> None:
        """Only the canonical ``# key: value`` form round-trips byte for byte."""
        path = tmp_path / "h.csv"
        path.write_bytes(b"#b:two words\r\n#  c :  x \r\nparticle,energy_MeV_u\r\n")
        CsvCheckpoint(path).save()
        assert path.read_bytes() == b"# b: two words\r\n# c: x\r\nparticle,energy_MeV_u\r\n"

    def test_fingerprint_line_has_sorted_keys(self, tmp_path: Path) -> None:
        path = tmp_path / "h.csv"
        checkpoint = CsvCheckpoint(path, ["particle", "energy_MeV_u"])
        checkpoint.header[FINGERPRINT_KEY] = [{"z": 1, "seed": 2, "alpha": 0.5}]
        checkpoint.save()
        assert path.read_bytes().startswith(
            b'# fingerprint: [{"alpha": 0.5, "seed": 2, "z": 1}]\r\nparticle,'
        )

    def test_duplicate_header_key_raises(self, tmp_path: Path) -> None:
        path = tmp_path / "h.csv"
        path.write_text("# a: 1\n# a: 2\nparticle,energy_MeV_u\n", newline="")
        with pytest.raises(ValueError, match="duplicate"):
            CsvCheckpoint(path)
        with pytest.raises(ValueError, match="duplicate"):
            read_header(path)

    @pytest.mark.parametrize(
        "raw",
        [
            b"# a: 1\n\nparticle,energy_MeV_u\ncarbon,3\n",  # blank line after the header
            b"\xef\xbb\xbfparticle,energy_MeV_u\ncarbon,3\n",  # UTF-8 BOM
            b"cell,let\nV79,3\n",  # no key columns at all
        ],
    )
    def test_column_row_without_key_columns_is_a_clear_error(
        self, tmp_path: Path, raw: bytes
    ) -> None:
        path = tmp_path / "h.csv"
        path.write_bytes(raw)
        with pytest.raises(ValueError, match="key columns"):
            CsvCheckpoint(path)

    def test_committed_checkpoints_carry_a_fingerprinted_header(self, tmp_path: Path) -> None:
        # The three LEM I checkpoints carry a provenance header and a numeric
        # fingerprint; a missing file still reads as no header.
        for name, _, _ in CSV_FILES:
            header = read_header(_data_file(name))
            assert header["generator"].startswith("scripts/lem1_mc_reference.py ")
            assert FINGERPRINT_KEY in header
        assert read_header(tmp_path / "missing.csv") == {}

    def test_malformed_header_line_raises(self, tmp_path: Path) -> None:
        path = tmp_path / "bad.csv"
        path.write_text("# no separator here\nparticle\n", newline="")
        with pytest.raises(ValueError, match="key: value"):
            CsvCheckpoint(path)

    @pytest.mark.parametrize(
        ("key", "value"),
        [("multi", "two\nlines"), ("bad:key", "x"), ("", "x"), ("n", 3)],
    )
    def test_unwritable_header_entries_raise(self, tmp_path: Path, key: str, value: Any) -> None:
        checkpoint = CsvCheckpoint(tmp_path / "x.csv", ["particle", "energy_MeV_u"])
        checkpoint.header[key] = value
        with pytest.raises((ValueError, TypeError)):
            checkpoint.save()


# ------------------------------------------------------------------ point spec


class TestParsePointSpec:
    SPECS = ("carbon:20,proton:6", "carbon:20", " carbon :8.5, proton:1e1", "carbon:20,carbon:20.0")

    @pytest.mark.parametrize("script", ["mc_batch_scholz1997", "mc_batch_scholz1997_zslope"])
    @pytest.mark.parametrize("spec", SPECS)
    def test_matches_unvalidated_parse_only(self, script: str, spec: str) -> None:
        assert parse_point_spec(spec) == _script(script)["parse_only"](spec)

    @pytest.mark.parametrize("spec", (*SPECS, "Carbon:3,ARGON:8"))
    def test_matches_sk1996_validated_parse_only(self, spec: str) -> None:
        ns = _script("mc_batch_scholz_kraft1996")
        assert parse_point_spec(spec, ns["SPECIES"]) == ns["parse_only"](spec)

    def test_unknown_particle_raises_the_scripts_error(self) -> None:
        ns = _script("mc_batch_scholz_kraft1996")
        with pytest.raises(argparse.ArgumentTypeError) as theirs:
            ns["parse_only"]("gold:3")
        with pytest.raises(argparse.ArgumentTypeError) as ours:
            parse_point_spec("gold:3", ns["SPECIES"])
        assert str(ours.value) == str(theirs.value) == "unknown particle 'gold'"

    @pytest.mark.parametrize("spec", ["carbon", "carbon:1:2", "carbon:x", ""])
    def test_malformed_tokens_raise_value_error(self, spec: str) -> None:
        with pytest.raises(ValueError):
            parse_point_spec(spec)

    def test_no_species_keeps_case(self) -> None:
        assert parse_point_spec("Carbon:3") == {("Carbon", 3.0)}


# ------------------------------------------------------------------- JSON cache


class TestJsonPointCache:
    def test_existing_cache_round_trip_is_byte_identical(self, tmp_path: Path) -> None:
        original = _our_mc_cache()
        copy = _copy(original, tmp_path)
        cache = JsonPointCache(copy)
        stored = json.loads(original.read_text())
        assert len(cache) == len(stored) - (PROVENANCE_KEY in stored)
        cache.save()
        assert copy.read_bytes() == original.read_bytes()

    def test_put_of_an_existing_entry_is_byte_identical(self, tmp_path: Path) -> None:
        original = _our_mc_cache()
        copy = _copy(original, tmp_path)
        cache = JsonPointCache(copy)
        key = cache.keys()[0]
        cache.put(key, dict(cache[key]))
        assert copy.read_bytes() == original.read_bytes()

    def test_write_matches_the_survival_script(self, tmp_path: Path) -> None:
        """survival_code_lem_comparison.py:330-332 (write_text, indent=1, sorted)."""
        path = tmp_path / "c.json"
        entry = {"let": 307.7, "alpha": 1.4428860962661978, "model": "LEMII", "Z": 8, "A": 16}
        JsonPointCache(path).put("036f31d4b674d62a", entry)
        expected = tmp_path / "expected.json"
        expected.write_text(json.dumps({"036f31d4b674d62a": entry}, indent=1, sort_keys=True))
        assert path.read_bytes() == expected.read_bytes()

    def test_put_rereads_so_a_sibling_run_is_kept(self, tmp_path: Path) -> None:
        path = tmp_path / "c.json"
        first, second = JsonPointCache(path), JsonPointCache(path)
        first.put("a", {"x": 1.0})
        second.put("b", {"x": 2.0})
        assert sorted(json.loads(path.read_text())) == ["a", "b"]
        assert sorted(second.keys()) == ["a", "b"]

    def test_unreadable_file_is_an_empty_cache(self, tmp_path: Path) -> None:
        """Mirrors ``_load_mc_cache``: missing or corrupt reads as empty."""
        path = tmp_path / "c.json"
        assert len(JsonPointCache(path)) == 0
        path.write_text("{not json")
        cache = JsonPointCache(path)
        assert len(cache) == 0
        assert "a" not in cache
        assert cache.get("a") is None

    def test_put_keeps_memory_when_the_file_turns_unreadable(self, tmp_path: Path) -> None:
        """A sibling caught mid-write must not make put() discard the cache."""
        path = tmp_path / "c.json"
        cache = JsonPointCache(path)
        cache.put("a", {"x": 1.0})
        path.write_text("{truncated")
        cache.put("b", {"x": 2.0})
        assert sorted(json.loads(path.read_text())) == ["a", "b"]

    def test_non_object_json_raises(self, tmp_path: Path) -> None:
        """``_load_mc_cache`` crashed on a JSON list too (``dict()`` raised TypeError)."""
        path = tmp_path / "c.json"
        path.write_text("[1, 2]")
        with pytest.raises(TypeError, match="JSON object"):
            JsonPointCache(path)

    def test_non_mapping_provenance_raises(self, tmp_path: Path) -> None:
        path = tmp_path / "c.json"
        path.write_text(json.dumps({PROVENANCE_KEY: "scripts/x.py", "a": {"x": 1.0}}))
        with pytest.raises(ValueError, match=PROVENANCE_KEY):
            JsonPointCache(path)

    def test_save_leaves_no_temporary_file(self, tmp_path: Path) -> None:
        path = tmp_path / "c.json"
        JsonPointCache(path).put("a", {"x": 1.0})
        assert [p.name for p in tmp_path.iterdir()] == ["c.json"]

    def test_matches_the_paired_mc_checkpoint_writer(self, tmp_path: Path) -> None:
        """``save_checkpoint`` of the original paired-MC script."""
        checkpoint = {
            "LEMI|C|10.0": {"alpha": 0.71, "beta": 0.05, "cells": 4000.0},
            "LEMII|H|1.0": "failed",
        }
        theirs = tmp_path / "theirs.json"

        def save_checkpoint(cp: dict[str, Any]) -> None:  # body copied VERBATIM
            theirs.write_text(json.dumps(cp, indent=1, sort_keys=True))

        save_checkpoint(checkpoint)
        ours = tmp_path / "ours.json"
        cache = JsonPointCache(ours)
        for key, entry in reversed(list(checkpoint.items())):
            cache.put(key, entry)
        assert ours.read_bytes() == theirs.read_bytes()
        reread = JsonPointCache(theirs)
        assert reread["LEMII|H|1.0"] == "failed"
        reread.save()
        assert ours.read_bytes() == theirs.read_bytes()

    @pytest.mark.parametrize("newline", ["\n", "\r\n"])
    def test_existing_line_endings_are_kept(self, tmp_path: Path, newline: str) -> None:
        path = tmp_path / "c.json"
        text = json.dumps({"k": {"x": 1}}, indent=1, sort_keys=True)
        path.write_bytes(text.replace("\n", newline).encode())
        JsonPointCache(path).save()
        assert path.read_bytes() == text.replace("\n", newline).encode()

    def test_provenance_is_a_reserved_top_level_key(self, tmp_path: Path) -> None:
        path = tmp_path / "c.json"
        cache = JsonPointCache(path)
        cache.put("0123456789abcdef", {"alpha": 1.0})
        fingerprint = [{"model": "LEMI", "Z": 6, "E": 10.0, "alpha": 0.7}]
        cache.provenance = {"generator": "scripts/x.py", FINGERPRINT_KEY: fingerprint}
        cache.save()
        raw = json.loads(path.read_text())
        assert raw[PROVENANCE_KEY][FINGERPRINT_KEY] == fingerprint
        # a legacy reader (dict lookup by hash key) is unaffected
        assert raw["0123456789abcdef"] == {"alpha": 1.0}
        reread = JsonPointCache(path)
        assert reread.keys() == ["0123456789abcdef"]
        assert PROVENANCE_KEY not in reread
        assert len(reread) == 1
        assert reread.provenance == cache.provenance
        assert read_header(path) == cache.provenance
        reread.put("fedcba9876543210", {"alpha": 2.0})
        assert JsonPointCache(path).provenance == cache.provenance

    def test_provenance_set_before_put_survives_the_reread(self, tmp_path: Path) -> None:
        path = tmp_path / "c.json"
        cache = JsonPointCache(path)
        cache.provenance = {"engine": "numpy"}
        cache.put("a", {"x": 1.0})
        assert read_header(path) == {"engine": "numpy"}

    def test_reserved_key_cannot_be_an_entry(self, tmp_path: Path) -> None:
        cache = JsonPointCache(tmp_path / "c.json")
        with pytest.raises(ValueError, match=PROVENANCE_KEY):
            cache.put(PROVENANCE_KEY, {})
        with pytest.raises(KeyError):
            cache[PROVENANCE_KEY]

    def test_no_provenance_reads_as_empty_header(self, tmp_path: Path) -> None:
        path = tmp_path / "c.json"
        JsonPointCache(path).put("a", {"x": 1.0})
        assert JsonPointCache(path).provenance is None
        assert read_header(path) == {}
        assert read_header(tmp_path / "absent.json") == {}


# ------------------------------------------------------------------- config_key


def test_config_key_is_the_scripts_hash() -> None:
    payload = {"v": 1, "model": "LEMI", "Z": 6, "E": 10.0, "doses": [1.0, 2.0]}
    expected = hashlib.sha256(json.dumps(payload, sort_keys=True).encode()).hexdigest()[:16]
    assert config_key(payload) == expected
    assert config_key(dict(reversed(list(payload.items())))) == expected
    assert len(expected) == 16


# Copied VERBATIM from the original Survival-comparison script (its constants and its
# cache-key function). That script had no __main__ guard and ran the MC at import, so
# it cannot be loaded; the copy is what pins config_key to the keys the cache holds.
ALPHA_X, BETA_X, R_N = 0.313, 0.0615, 5.0
D_T = {"LEMI": 30.0, "LEMII": 6.0, "LEMIII": 19.0}
MODELS = {"LEMI": LEM1, "LEMII": LEM2, "LEMIII": LEM3}
MATCH_SURVIVAL_R_MAX = True
OUR_MC_DOSES_GY = (1.0, 2.0, 3.0, 4.0, 5.0, 6.0)  # Survival's own MC dose set
OUR_MC_N_REPEAT = 8
OUR_MC_TARGET_REL_ALPHA = 0.003
OUR_MC_TARGET_ABS_BETA = 0.02 * BETA_X
OUR_MC_PILOT_N = 400
OUR_MC_N_MIN = 250
OUR_MC_N_MAX = 200_000
OUR_MC_CACHE_VERSION = 1
OUR_MC_B_CUT_UM = 10.0
OUR_MC_B_CUT_MIN_NBAR = 2000.0
STOPPING: Any = None  # srim_stopping_power(("H", "He", "C", "O")), set by the test


def build(model: str, **kwargs) -> LEM1 | LEM2 | LEM3:
    """A pyRadBio LEM on the HSG parameters, driven by Survival's own LET."""
    photon = PhotonParameters(alpha_x=ALPHA_X, beta_x=BETA_X, D_t=D_T[model])
    if MATCH_SURVIVAL_R_MAX and model == "LEMI" and "radial_dose" not in kwargs:
        kwargs["radial_dose"] = replace(
            ScholzKraftRadialDose.lem1(stopping_power=STOPPING),
            r_max_gamma_um=R_MAX_GAMMA_UM,
        )
    return MODELS[model](photon=photon, cell=CellTarget(R_N), stopping_power=STOPPING, **kwargs)


def _mc_config_key(model: str, Z: int, A: int, e_u: float) -> str:
    """Cache key: everything that can change the number, and nothing else.

    Any of these moving must invalidate the cached point, so they are hashed
    rather than trusted to a version bump someone will forget. The LEM version
    string is *not* enough on its own -- the same `LEMII` at a different `D_t` or
    `r_max` is a different calculation.
    """
    lem = build(model)
    payload = json.dumps(
        {
            "v": OUR_MC_CACHE_VERSION,
            "model": model,
            "Z": Z,
            "A": A,
            "E": e_u,
            "alpha_x": ALPHA_X,
            "beta_x": BETA_X,
            "R_N": R_N,
            "D_t": D_T[model],
            "r_max_gamma": lem._radial().r_max_gamma_um,
            "r_min_um": lem._radial().r_min_um(Ion(Z, A, e_u)),
            "sigma_nm": getattr(lem._radial(), "sigma_nm", 0.0),
            "doses": list(OUR_MC_DOSES_GY),
            "repeats": OUR_MC_N_REPEAT,
            "b_cut": OUR_MC_B_CUT_UM,
            "b_cut_min_nbar": OUR_MC_B_CUT_MIN_NBAR,
            "target_alpha": OUR_MC_TARGET_REL_ALPHA,
            "target_beta": OUR_MC_TARGET_ABS_BETA,
            "n_pilot": OUR_MC_PILOT_N,
            "n_min": OUR_MC_N_MIN,
            "n_max": OUR_MC_N_MAX,
            "let": STOPPING.let_keV_um(Ion(Z, A, e_u)),
        },
        sort_keys=True,
    )
    return hashlib.sha256(payload.encode()).hexdigest()[:16]


def _payload(model: str, Z: int, A: int, e_u: float) -> dict[str, Any]:
    """The same payload as ``_mc_config_key``, handed to ``config_key`` unhashed."""
    lem = build(model)
    radial = lem._radial()
    return {
        "v": OUR_MC_CACHE_VERSION,
        "model": model,
        "Z": Z,
        "A": A,
        "E": e_u,
        "alpha_x": ALPHA_X,
        "beta_x": BETA_X,
        "R_N": R_N,
        "D_t": D_T[model],
        "r_max_gamma": radial.r_max_gamma_um,
        "r_min_um": radial.r_min_um(Ion(Z, A, e_u)),
        "sigma_nm": getattr(radial, "sigma_nm", 0.0),
        "doses": list(OUR_MC_DOSES_GY),
        "repeats": OUR_MC_N_REPEAT,
        "b_cut": OUR_MC_B_CUT_UM,
        "b_cut_min_nbar": OUR_MC_B_CUT_MIN_NBAR,
        "target_alpha": OUR_MC_TARGET_REL_ALPHA,
        "target_beta": OUR_MC_TARGET_ABS_BETA,
        "n_pilot": OUR_MC_PILOT_N,
        "n_min": OUR_MC_N_MIN,
        "n_max": OUR_MC_N_MAX,
        "let": STOPPING.let_keV_um(Ion(Z, A, e_u)),
    }


def test_config_key_reproduces_every_stored_cache_key(monkeypatch: pytest.MonkeyPatch) -> None:
    from pyradbio.track.srim import srim_stopping_power

    cache = JsonPointCache(_our_mc_cache())
    try:
        stopping = srim_stopping_power(("H", "He", "C", "O"))
    except (FileNotFoundError, OSError):
        pytest.skip("Survival's SRIM tables (PYRADBIO_SURVIVAL_SRC_DIR) are not available")
    monkeypatch.setitem(globals(), "STOPPING", stopping)
    assert len(cache) > 0
    for key in cache.keys():
        entry = cache[key]
        point = (entry["model"], entry["Z"], entry["A"], entry["E"])
        assert _mc_config_key(*point) == key  # the verbatim copy is faithful
        assert config_key(_payload(*point)) == key


# ------------------------------------------------------------------ fingerprint


class TestAssertFingerprint:
    RECORDED: ClassVar[list[dict[str, Any]]] = [
        {"particle": "carbon", "energy_MeV_u": 6.0, "seed": 3, "alpha": 0.71, "beta": 0.012},
        {"particle": "proton", "energy_MeV_u": 2.0, "seed": 4, "alpha": 0.33, "beta": 0.04},
    ]

    def test_identical_passes(self) -> None:
        assert_fingerprint(self.RECORDED, json.loads(json.dumps(self.RECORDED)), generator="g")

    def test_within_rtol_passes_and_tuples_match_lists(self) -> None:
        recomputed = [dict(p) for p in self.RECORDED]
        recomputed[0]["alpha"] *= 1.0 + 5e-10
        assert_fingerprint(self.RECORDED, tuple(recomputed), generator="g")

    def test_numpy_scalars_compare_as_numbers(self) -> None:
        np = pytest.importorskip("numpy")
        assert_fingerprint(
            {"a": 1.5, "n": 3}, {"a": np.float64(1.5), "n": np.int64(3)}, generator="g"
        )
        assert_fingerprint({"a": [1.0, 2.0]}, {"a": np.array([1.0, 2.0])}, generator="g")

    def test_drift_raises_with_the_regenerate_hint(self) -> None:
        recomputed = [dict(p) for p in self.RECORDED]
        recomputed[1]["beta"] *= 1.0 + 1e-6
        with pytest.raises(AssertionError) as info:
            assert_fingerprint(self.RECORDED, recomputed, generator="lem1_mc_reference")
        message = str(info.value)
        assert "[1].beta" in message
        assert message.endswith("regenerate with scripts/lem1_mc_reference.py")

    def test_generator_with_suffix_or_subdirectory(self) -> None:
        with pytest.raises(AssertionError, match=r"regenerate with scripts/lem/gen\.py$"):
            assert_fingerprint({"a": 1.0}, {"a": 2.0}, generator="lem/gen.py")

    @pytest.mark.parametrize(
        "recomputed",
        [
            {"a": 1.0},  # missing key
            {"a": 1.0, "b": [1.0], "c": 0},  # extra key
            {"a": 1.0, "b": [1.0, 2.0]},  # length
            {"a": "1.0", "b": [1.0]},  # type
            {"a": 1.0, "b": 1.0},  # list vs scalar
            {"a": True, "b": [1.0]},  # bool is not a number here
            [1.0, [1.0]],  # sequence where a mapping was recorded
        ],
    )
    def test_structure_mismatch_raises(self, recomputed: Any) -> None:
        with pytest.raises(AssertionError, match=r"regenerate with scripts/g\.py"):
            assert_fingerprint({"a": 1.0, "b": [1.0]}, recomputed, generator="g")

    def test_missing_record_raises(self) -> None:
        with pytest.raises(AssertionError, match=r"no fingerprint.*regenerate with scripts/g\.py"):
            assert_fingerprint(None, self.RECORDED, generator="g")

    def test_nan_matches_nan_and_zero_needs_atol(self) -> None:
        assert_fingerprint({"a": math.nan}, {"a": float("nan")}, generator="g")
        with pytest.raises(AssertionError):
            assert_fingerprint({"a": math.nan}, {"a": 0.0}, generator="g")
        with pytest.raises(AssertionError):
            assert_fingerprint({"b": 0.0}, {"b": 1e-300}, generator="g")
        assert_fingerprint({"b": 0.0}, {"b": 1e-300}, generator="g", atol=1e-200)

    @pytest.mark.parametrize(
        ("recorded", "recomputed"),
        [
            (5.0, math.inf),
            (math.inf, 5.0),
            (math.inf, -math.inf),
            (-math.inf, math.inf),
            (1e308, math.inf),
            (math.nan, math.inf),
        ],
    )
    def test_infinity_matches_only_itself(self, recorded: float, recomputed: float) -> None:
        with pytest.raises(AssertionError, match=r"regenerate with scripts/g\.py"):
            assert_fingerprint({"a": recorded}, {"a": recomputed}, generator="g")
        with pytest.raises(AssertionError):
            assert_fingerprint({"a": recorded}, {"a": recomputed}, generator="g", atol=1e300)

    def test_equal_infinities_pass(self) -> None:
        both = {"a": math.inf, "b": -math.inf}
        assert_fingerprint(both, dict(both), generator="g")

    def test_tolerance_is_relative_to_the_larger_magnitude(self) -> None:
        """``|a - b| <= rtol * max(|a|, |b|)``: 0.75 <= 0.5 * 1.75, though > 0.5 * 1.0."""
        assert_fingerprint({"a": 1.0}, {"a": 1.75}, generator="g", rtol=0.5)
        assert_fingerprint({"a": 1.75}, {"a": 1.0}, generator="g", rtol=0.5)
        with pytest.raises(AssertionError):
            assert_fingerprint({"a": 1.0}, {"a": 2.01}, generator="g", rtol=0.5)

    @pytest.mark.parametrize("recorded", [[], {}, ()])
    def test_empty_fingerprint_raises(self, recorded: Any) -> None:
        with pytest.raises(AssertionError, match=r"empty.*regenerate with scripts/g\.py"):
            assert_fingerprint(recorded, recorded, generator="g")

    def test_strings_and_bools_compare_exactly(self) -> None:
        assert_fingerprint({"e": "numpy", "ok": True}, {"e": "numpy", "ok": True}, generator="g")
        with pytest.raises(AssertionError):
            assert_fingerprint({"e": "numpy"}, {"e": "numba"}, generator="g")
