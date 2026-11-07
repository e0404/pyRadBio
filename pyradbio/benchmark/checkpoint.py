"""Checkpoint I/O shared by the offline Monte-Carlo batches and their readers.

The offline MC batches write their results into checkpoints that are resumed
point by point, and the validation scripts, the ``benchmark.<paper>`` loaders and
their unit tests read them back. This module is the single definition of those
file formats, so writer and readers cannot drift apart:

* :class:`CsvCheckpoint` -- the per-point CSVs of ``scripts/lem1_mc_reference.py``:
  ``alpha-beta`` (``mc_results.csv``), ``sk1996-final-slope``
  (``sk1996_mc_observables.csv``) and ``z-slope`` (``sk1997_z_slope.csv``,
  several rows per point), all in ``validation/data/``. Rows are grouped by a point
  key (``(particle, energy_MeV_u)`` by default), a point is replaced as a whole,
  and the file is rewritten sorted by key after every point.
* :func:`parse_point_spec` -- the ``--only carbon:20,proton:6`` CLI parser the
  batches share.
* :class:`JsonPointCache` and :func:`config_key` -- the hash-keyed per-point
  cache ``validation/data/survival_our_mc.json`` that
  ``scripts/lem_mc_vs_survival.py`` writes. The out-of-repo ``checkpoint.json``
  of ``scripts/survival_paired_mc.py`` (``model|ion|E`` keys, ``"failed"``
  string entries) has the same format and round-trips too; that script loops
  over every top-level key, so give its file a provenance header only once it
  reads through :meth:`JsonPointCache.keys`.

**Byte-exact round trip.** Reading a checkpoint and writing it back reproduces
the file byte for byte: rows read from disk keep their original cell strings
(``3`` stays ``3``, not ``3.0``), the
file's own line terminator is reused, and new rows are formatted as
``csv.DictWriter`` formats them (``str()`` per cell, so floats
are their shortest round-trip ``repr``). A provenance header round-trips byte
for byte in its canonical form ``# key: value`` (the form :meth:`CsvCheckpoint.save`
writes); a hand-written ``#key:value`` reads the same but is rewritten
canonically. Every rewrite goes through a temporary file in the same directory
and ``os.replace``, so a run killed mid-save leaves the previous file intact.
Both formats are read and written as UTF-8 (identical to the locale default for
the ASCII files the batches write).

**Provenance header.** Every committed MC artefact is gated by a *numeric
fingerprint*, so a change to the code that produced it fails a fast test
instead of silently invalidating it. A CSV checkpoint may start with
``# key: value`` lines (keys unique); readers skip them and :func:`read_header`
returns them. The value under ``fingerprint`` (:data:`FINGERPRINT_KEY`) is one line of JSON:
1-2 points recomputed at a small sample size with ``engine="numpy"`` and fixed
seeds, e.g. ``[{"particle": "carbon", "energy_MeV_u": 6.0, "n_realizations":
200, "seed": 1, "alpha": ..., "beta": ...}]``. A JSON cache carries the same
mapping under the reserved top-level key ``"_provenance"``
(:data:`PROVENANCE_KEY`): that is the least invasive form, since a legacy reader
looks entries up by their 16-hex-digit hash and never sees it, and rewrites it
untouched. A fast test recomputes the fingerprint and compares it with
:func:`assert_fingerprint`; on mismatch it tells the reader which
``scripts/<generator>.py`` regenerates the artefact. Files without a header
round-trip unchanged and get none added.

No physics lives here.
"""

from __future__ import annotations

import argparse
import csv
import hashlib
import io
import json
import math
import numbers
import os
from collections.abc import Collection, Iterator, Mapping, Sequence
from pathlib import Path
from typing import Any

__all__ = [
    "DEFAULT_KEY_COLUMNS",
    "FINGERPRINT_KEY",
    "PROVENANCE_KEY",
    "CheckpointKey",
    "CsvCheckpoint",
    "JsonPointCache",
    "assert_fingerprint",
    "config_key",
    "parse_point_spec",
    "read_header",
]

CheckpointKey = tuple[Any, ...]
"""A point key: one entry per key column, text columns as ``str``, others ``float``."""

DEFAULT_KEY_COLUMNS: tuple[str, ...] = ("particle", "energy_MeV_u")
"""The point identity every batch checkpoint uses: ``(particle, float(energy))``."""

FINGERPRINT_KEY = "fingerprint"
"""Header key holding the JSON fingerprint (decoded by :func:`read_header`)."""

PROVENANCE_KEY = "_provenance"
"""Reserved top-level key of a :class:`JsonPointCache` holding its header."""

_CSV_DEFAULT_TERMINATOR = "\r\n"  # csv.DictWriter's default, used by every batch script


def _detect_terminator(raw: bytes, default: str) -> str:
    """The line terminator of the first line of ``raw``; ``default`` if it has none."""
    newline = raw.find(b"\n")
    if newline < 0:
        return default
    return "\r\n" if newline > 0 and raw[newline - 1 : newline] == b"\r" else "\n"


def _atomic_write(path: Path, data: bytes) -> None:
    """Write ``data`` to a sibling temporary file, then ``os.replace`` it over ``path``.

    A crash mid-write leaves the previous file untouched. The temporary name
    carries the process id, so concurrent runs sharing a cache never collide.
    """
    path.parent.mkdir(parents=True, exist_ok=True)
    tmp = path.with_name(f"{path.name}.{os.getpid()}.tmp")
    try:
        tmp.write_bytes(data)
        os.replace(tmp, path)
    except BaseException:
        tmp.unlink(missing_ok=True)
        raise


def _parse_header_line(line: str) -> tuple[str, str]:
    body = line.rstrip("\r\n")[1:]
    key, sep, value = body.partition(":")
    if not sep or not key.strip():
        raise ValueError(f"malformed checkpoint header line {line!r}; expected '# key: value'")
    return key.strip(), value.strip()


def _add_header_line(raw: dict[str, str], line: str) -> None:
    key, value = _parse_header_line(line)
    if key in raw:
        raise ValueError(f"duplicate checkpoint header key {key!r}")
    raw[key] = value


def _decode_header(raw: Mapping[str, str]) -> dict[str, Any]:
    header: dict[str, Any] = dict(raw)
    if FINGERPRINT_KEY in header:
        header[FINGERPRINT_KEY] = json.loads(header[FINGERPRINT_KEY])
    return header


def _encode_header_line(key: str, value: Any) -> str:
    if not key or ":" in key or "\n" in key or "\r" in key or key != key.strip():
        raise ValueError(f"invalid checkpoint header key {key!r}")
    if key == FINGERPRINT_KEY:
        text = json.dumps(value, sort_keys=True)
    elif isinstance(value, str):
        text = value
    else:
        raise TypeError(
            f"checkpoint header value for {key!r} must be str (only {FINGERPRINT_KEY!r} "
            f"holds JSON), got {type(value).__name__}"
        )
    if "\n" in text or "\r" in text or text != text.strip():
        raise ValueError(f"checkpoint header value for {key!r} must be one stripped line")
    return f"# {key}: {text}"


class CsvCheckpoint:
    """A resumable per-point CSV checkpoint (read, replace points, rewrite).

    Opening the checkpoint reads ``path`` if it exists (the batches' resume
    step). :meth:`put` / :meth:`put_group` replace one point in memory and
    :meth:`save` rewrites the whole file sorted by point key -- the batches call
    it after every point, and the rewrite is atomic (temporary file plus
    ``os.replace``), so a killed run loses at most the point in flight.

    A point stored in several rows is kept as one group by default (the
    ``sk1997_z_slope.csv`` layout), and :meth:`record` refuses it. The
    single-row scripts instead read their file into a dict, so a duplicated
    point silently kept its *last* row and the next write dropped the others;
    ``one_row_per_point=True`` reproduces that policy.

    The scripts' row filters (estimator version, stopping power) and
    model-derived backfills stay with the scripts: they are paper logic, not
    file format.

    Parameters
    ----------
    path:
        The checkpoint file. It need not exist yet.
    fieldnames:
        The columns written by :meth:`save`, in order (the scripts'
        ``FIELDNAMES``). ``None`` reuses the header row of the existing file,
        which is what a reader that does not know the columns wants.
    key_columns:
        Columns forming the point key; default ``("particle", "energy_MeV_u")``.
    text_columns:
        Columns that stay strings in :meth:`record` (and in the key); every
        other non-empty cell is parsed with ``float``.
    lineterminator:
        ``None`` (default) reuses the existing file's terminator and otherwise
        writes ``"\\r\\n"``, the ``csv`` default every batch script wrote with.
    one_row_per_point:
        Keep only the last row of a point read from disk (dict-based loading)
        and reject multi-row :meth:`put_group` calls.

    Attributes
    ----------
    header:
        The provenance header, ``{key: value}`` in file order. Values are
        ``str``, except :data:`FINGERPRINT_KEY`, which holds decoded JSON. Empty
        for files without a header; :meth:`save` writes none then.
    """

    def __init__(
        self,
        path: str | os.PathLike[str],
        fieldnames: Sequence[str] | None = None,
        *,
        key_columns: Sequence[str] = DEFAULT_KEY_COLUMNS,
        text_columns: Collection[str] = ("particle",),
        lineterminator: str | None = None,
        one_row_per_point: bool = False,
    ) -> None:
        self.path = Path(path)
        self.key_columns = tuple(key_columns)
        self.text_columns = frozenset(text_columns)
        self.one_row_per_point = one_row_per_point
        self.header: dict[str, Any] = {}
        self._groups: dict[CheckpointKey, list[dict[str, Any]]] = {}
        self._file_fieldnames: list[str] | None = None
        detected = _CSV_DEFAULT_TERMINATOR
        if self.path.exists():
            raw = self.path.read_bytes()
            detected = _detect_terminator(raw, _CSV_DEFAULT_TERMINATOR)
            self._read(raw.decode("utf-8"))
        self.lineterminator = lineterminator if lineterminator is not None else detected
        self._fieldnames = list(fieldnames) if fieldnames is not None else None

    @property
    def fieldnames(self) -> list[str] | None:
        """The columns :meth:`save` writes: the given ones, else the file's own."""
        if self._fieldnames is not None:
            return list(self._fieldnames)
        return list(self._file_fieldnames) if self._file_fieldnames is not None else None

    def _read(self, text: str) -> None:
        lines = text.splitlines(keepends=True)
        n_header = 0
        raw_header: dict[str, str] = {}
        while n_header < len(lines) and lines[n_header].startswith("#"):
            _add_header_line(raw_header, lines[n_header])
            n_header += 1
        self.header = _decode_header(raw_header)
        reader = csv.DictReader(io.StringIO("".join(lines[n_header:]), newline=""))
        if reader.fieldnames is not None:
            missing = [c for c in self.key_columns if c not in reader.fieldnames]
            if missing:
                raise ValueError(
                    f"{self.path}: the column row {list(reader.fieldnames)!r} lacks the key "
                    f"columns {missing!r}; it must directly follow the '# key: value' header "
                    "lines (no blank line, no byte-order mark)"
                )
        for row in reader:
            key = self.key_of(row)
            if self.one_row_per_point:
                self._groups[key] = [dict(row)]  # last row wins, like the scripts' dict
            else:
                self._groups.setdefault(key, []).append(dict(row))
        self._file_fieldnames = list(reader.fieldnames) if reader.fieldnames else None

    def key_of(self, row: Mapping[str, Any]) -> CheckpointKey:
        """The point key of ``row``: text key columns as ``str``, others ``float``."""
        return tuple(
            str(row[c]) if c in self.text_columns else float(row[c]) for c in self.key_columns
        )

    def __len__(self) -> int:
        """Number of points (not rows)."""
        return len(self._groups)

    def __contains__(self, key: object) -> bool:
        return key in self._groups

    def keys(self) -> list[CheckpointKey]:
        """All point keys, sorted as :meth:`save` writes them."""
        return sorted(self._groups)

    def rows(self, key: CheckpointKey) -> list[dict[str, Any]]:
        """Copies of the stored rows of one point (cells read from disk are ``str``)."""
        return [dict(row) for row in self._groups[key]]

    def _parse(self, row: Mapping[str, Any]) -> dict[str, Any]:
        return {
            k: (v if isinstance(v, str) else str(v)) if k in self.text_columns else float(v)
            for k, v in row.items()
            if v is not None and v != ""
        }

    def records(self, key: CheckpointKey) -> list[dict[str, Any]]:
        """Parsed rows of one point: text columns ``str``, others ``float``, empty cells dropped."""
        return [self._parse(row) for row in self._groups[key]]

    def record(self, key: CheckpointKey) -> dict[str, Any]:
        """The single parsed row of a one-row-per-point checkpoint."""
        rows = self._groups[key]
        if len(rows) != 1:
            raise ValueError(f"point {key!r} has {len(rows)} rows; use records()")
        return self._parse(rows[0])

    def iter_records(self) -> Iterator[dict[str, Any]]:
        """Every parsed row, points in key order, rows in stored order."""
        for key in self.keys():
            yield from self.records(key)

    def put(self, row: Mapping[str, Any]) -> CheckpointKey:
        """Replace the point of ``row`` with this single row; return its key."""
        return self.put_group([row])

    def put_group(self, rows: Sequence[Mapping[str, Any]]) -> CheckpointKey:
        """Replace one point with several rows (``sk1997_z_slope.csv``); return its key."""
        if not rows:
            raise ValueError("put_group needs at least one row")
        if self.one_row_per_point and len(rows) != 1:
            raise ValueError(f"this checkpoint holds one row per point, got {len(rows)} rows")
        keys = {self.key_of(row) for row in rows}
        if len(keys) != 1:
            raise ValueError(f"put_group rows must all belong to one point, got {sorted(keys)}")
        (key,) = keys
        self._groups[key] = [dict(row) for row in rows]
        return key

    def discard(self, key: CheckpointKey) -> None:
        """Drop a point if present (e.g. a row from a stale estimator version)."""
        self._groups.pop(key, None)

    def save(self) -> None:
        """Rewrite the file: header lines, column header, rows sorted by point key.

        Cells are written by ``csv.DictWriter``; ``mkdir`` covers a fresh output
        directory. The file is replaced atomically, so an error mid-save (e.g. a
        row with a column not in ``fieldnames``) leaves the previous file intact.
        """
        fieldnames = self.fieldnames
        if fieldnames is None:
            raise ValueError(
                f"{self.path}: no fieldnames given and no existing header row to reuse"
            )
        header_lines = [_encode_header_line(k, v) for k, v in self.header.items()]
        handle = io.StringIO(newline="")
        for line in header_lines:
            handle.write(line + self.lineterminator)
        writer = csv.DictWriter(handle, fieldnames=fieldnames, lineterminator=self.lineterminator)
        writer.writeheader()
        for key in self.keys():
            writer.writerows(self._groups[key])
        _atomic_write(self.path, handle.getvalue().encode("utf-8"))


def parse_point_spec(spec: str, species: Collection[str] | None = None) -> set[tuple[str, float]]:
    """Parse a ``--only`` point list such as ``"carbon:20,proton:6"``.

    With ``species=None`` particle names are stripped but kept as typed. With a
    ``species`` collection names are lower-cased and an unknown one raises
    :class:`argparse.ArgumentTypeError`. A token that is not ``name:energy``
    raises ``ValueError`` (argparse reports both as an invalid value).
    """
    out: set[tuple[str, float]] = set()
    for token in spec.split(","):
        particle, energy = token.split(":")
        particle = particle.strip()
        if species is not None:
            particle = particle.lower()
            if particle not in species:
                raise argparse.ArgumentTypeError(f"unknown particle {particle!r}")
        out.add((particle, float(energy)))
    return out


def config_key(payload: Mapping[str, Any]) -> str:
    """The 16-hex-digit cache key of a JSON-serialisable configuration payload.

    ``sha256(json.dumps(payload, sort_keys=True))`` truncated to 16 hex digits.
    The payload should hold everything that can change the
    cached number and nothing else, so any change invalidates the point.
    """
    text = json.dumps(dict(payload), sort_keys=True)
    return hashlib.sha256(text.encode()).hexdigest()[:16]


class JsonPointCache:
    """A hash-keyed JSON cache of MC points (``validation/data/survival_our_mc.json``).

    The file is one JSON object ``{config_key: entry}`` written with
    ``indent=1, sort_keys=True`` and no trailing newline. A missing, unreadable or
    corrupt file reads as an empty cache; valid JSON that is not an object raises
    ``TypeError``. :meth:`put` re-reads the file before writing, so points a sibling
    run added meanwhile are kept; if that re-read fails (a sibling caught mid-write,
    a Windows sharing violation) the entries already in memory are kept instead of being
    discarded. Writes are atomic (temporary file plus ``os.replace``).

    The optional provenance header lives under the reserved top-level key
    :data:`PROVENANCE_KEY` and is exposed as :attr:`provenance`, never as an
    entry; a non-object value there raises ``ValueError``. Line endings of an
    existing file are kept; a new file is written with the platform's, like
    ``Path.write_text``.
    """

    def __init__(self, path: str | os.PathLike[str], *, newline: str | None = None) -> None:
        self.path = Path(path)
        self._newline = newline
        self._entries: dict[str, Any] = {}
        self._provenance: dict[str, Any] | None = None
        self._provenance_set = False
        self.reload()

    def reload(self) -> None:
        """Re-read the file (missing, unreadable or corrupt -> empty)."""
        self._load(keep_on_failure=False)

    def _load(self, *, keep_on_failure: bool) -> None:
        try:
            raw = self.path.read_bytes()
            data = json.loads(raw.decode("utf-8"))
        except (OSError, ValueError):
            if keep_on_failure:
                return
            data = {}
        else:
            if not isinstance(data, dict):
                raise TypeError(
                    f"{self.path}: a point cache must be a JSON object, got {type(data).__name__}"
                )
            if self._newline is None:
                self._newline = _detect_terminator(raw, os.linesep)
        provenance = data.pop(PROVENANCE_KEY, None)
        if provenance is not None and not isinstance(provenance, dict):
            raise ValueError(
                f"{self.path}: {PROVENANCE_KEY!r} must be a JSON object, "
                f"got {type(provenance).__name__}"
            )
        if not self._provenance_set:
            self._provenance = provenance
        self._entries = data

    @property
    def provenance(self) -> dict[str, Any] | None:
        """The provenance header (with :data:`FINGERPRINT_KEY`), or ``None``."""
        return dict(self._provenance) if self._provenance is not None else None

    @provenance.setter
    def provenance(self, value: Mapping[str, Any] | None) -> None:
        self._provenance = dict(value) if value is not None else None
        self._provenance_set = True

    def __len__(self) -> int:
        return len(self._entries)

    def __contains__(self, key: object) -> bool:
        return key in self._entries

    def __getitem__(self, key: str) -> Any:
        return self._entries[key]

    def get(self, key: str, default: Any = None) -> Any:
        """The entry under ``key``, or ``default``."""
        return self._entries.get(key, default)

    def keys(self) -> list[str]:
        """Entry keys in file order (the reserved provenance key excluded)."""
        return list(self._entries)

    def put(self, key: str, entry: Any) -> None:
        """Re-read the file, store ``entry`` under ``key`` and write it back.

        If the re-read fails, the entries already in memory are written instead.
        """
        if key == PROVENANCE_KEY:
            raise ValueError(f"{PROVENANCE_KEY!r} is reserved for the provenance header")
        self._load(keep_on_failure=True)
        self._entries[key] = entry
        self.save()

    def save(self) -> None:
        """Write the cache (``indent=1, sort_keys=True``, no trailing newline)."""
        data: dict[str, Any] = dict(self._entries)
        if self._provenance is not None:
            data[PROVENANCE_KEY] = self._provenance
        text = json.dumps(data, indent=1, sort_keys=True)
        # Path.write_text(newline=None) translates "\n" to os.linesep; JSON
        # escapes newlines inside strings, so every raw "\n" is a line break.
        newline = os.linesep if self._newline is None else self._newline
        if newline:
            text = text.replace("\n", newline)
        _atomic_write(self.path, text.encode("utf-8"))


def read_header(path: str | os.PathLike[str]) -> dict[str, Any]:
    """The provenance header of a checkpoint; ``{}`` if it has none or is missing.

    ``.json`` files are read as a :class:`JsonPointCache` (the
    :data:`PROVENANCE_KEY` object), anything else as a CSV whose leading
    ``# key: value`` lines form the header. In both, :data:`FINGERPRINT_KEY`
    holds decoded JSON.
    """
    path = Path(path)
    if path.suffix.lower() == ".json":
        return JsonPointCache(path).provenance or {}
    if not path.exists():
        return {}
    raw: dict[str, str] = {}
    with path.open(newline="", encoding="utf-8") as handle:
        for line in handle:
            if not line.startswith("#"):
                break
            _add_header_line(raw, line)
    return _decode_header(raw)


def _is_number(value: Any) -> bool:
    return isinstance(value, numbers.Real) and not isinstance(value, bool)


def _fingerprint_mismatch(
    recorded: Any, recomputed: Any, where: str, rtol: float, atol: float
) -> str | None:
    """Describe the first difference between two fingerprint values, if any."""
    if hasattr(recomputed, "tolist") and not _is_number(recomputed):
        recomputed = recomputed.tolist()  # numpy arrays
    label = where or "<root>"
    if isinstance(recorded, Mapping):
        if not isinstance(recomputed, Mapping):
            return f"{label}: recorded a mapping, recomputed {recomputed!r}"
        if set(recorded) != set(recomputed):
            return f"{label}: keys {sorted(recorded)} recorded, {sorted(recomputed)} recomputed"
        for k in recorded:
            path = f"{where}.{k}" if where else str(k)
            found = _fingerprint_mismatch(recorded[k], recomputed[k], path, rtol, atol)
            if found:
                return found
        return None
    if isinstance(recorded, (list, tuple)):
        if not isinstance(recomputed, (list, tuple)) or len(recorded) != len(recomputed):
            return f"{label}: recorded {recorded!r}, recomputed {recomputed!r}"
        for i, (a, b) in enumerate(zip(recorded, recomputed, strict=True)):
            found = _fingerprint_mismatch(a, b, f"{where}[{i}]", rtol, atol)
            if found:
                return found
        return None
    if _is_number(recorded) and _is_number(recomputed):
        a, b = float(recorded), float(recomputed)
        if math.isnan(a) and math.isnan(b):
            return None
        # isclose is |a - b| <= max(rtol * max(|a|, |b|), atol), and an infinity
        # matches only the same infinity (a hand-written form lets inf through).
        if math.isclose(a, b, rel_tol=rtol, abs_tol=atol):
            return None
        return f"{label}: recorded {a!r}, recomputed {b!r} (rtol={rtol:g}, atol={atol:g})"
    if type(recorded) is type(recomputed) and recorded == recomputed:
        return None
    return f"{label}: recorded {recorded!r}, recomputed {recomputed!r}"


def assert_fingerprint(
    recorded: Any,
    recomputed: Any,
    *,
    generator: str,
    rtol: float = 1e-9,
    atol: float = 0.0,
) -> None:
    """Fail unless a recomputed fingerprint matches the one an artefact recorded.

    Each committed MC artefact stores 1-2 points computed at a small sample
    size with ``engine="numpy"`` and fixed seeds, and a fast test recomputes
    them. Numbers compare with a tight relative tolerance, not bit-exactly,
    because NumPy ``Generator`` streams and float rounding are not guaranteed
    identical across versions and platforms; an infinity matches only the same
    infinity and NaN matches NaN. Strings, booleans and the nesting of
    mappings/sequences compare exactly.

    Parameters
    ----------
    recorded:
        The stored fingerprint (``read_header(path).get(FINGERPRINT_KEY)``);
        ``None`` (no fingerprint) and an empty list or mapping (one that
        checks nothing) fail too.
    recomputed:
        The same structure computed now (numpy scalars and arrays accepted).
    generator:
        The generating script under ``scripts/``, e.g. ``"lem1_mc_reference"``
        (``.py`` is appended unless given).
    rtol, atol:
        Pass when ``|a - b| <= max(rtol * max(|a|, |b|), atol)``
        (:func:`math.isclose`).

    Raises
    ------
    AssertionError
        Naming the first differing entry and ending in
        ``"regenerate with scripts/<generator>.py"``.
    """
    script = generator if generator.endswith(".py") else f"{generator}.py"
    hint = f"regenerate with scripts/{script}"
    if recorded is None:
        raise AssertionError(f"artefact has no fingerprint to check; {hint}")
    if isinstance(recorded, (Mapping, list, tuple)) and not recorded:
        raise AssertionError(f"artefact has an empty fingerprint, which checks nothing; {hint}")
    found = _fingerprint_mismatch(recorded, recomputed, "", rtol, atol)
    if found is not None:
        raise AssertionError(f"MC fingerprint changed at {found}; {hint}")
