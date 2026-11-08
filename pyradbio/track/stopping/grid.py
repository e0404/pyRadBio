"""Logarithmic energy grids, log-log regridding and the shipped-table CSV format.

* [`LogEnergyGrid`][] -- a user-defined log-uniform energy grid with exact
  endpoints.
* [`RegriddedTable`][] -- one ion's values on a set of nodes, interpolated in
  ``log E -> log S``: a not-a-knot cubic spline by default, monotone cubic (PCHIP)
  for comparison. Outside its nodes it raises [`OutOfRangeError`][].
* [`leave_one_out_relative_error`][] -- the interpolation-error estimate available
  for a real table, where nothing is known between the source points.
* [`write_table_csv`][] / [`read_table_csv`][] -- the long-format CSV our
  regridded tables ship in, with a provenance header and per-ion metadata.

CSV format ``pyradbio-stopping-table/1``:

```text
# free-text provenance notes, one per line
#@ format = "pyradbio-stopping-table/1"
#@ <key> = <JSON value>                      file-level metadata
#@ ion.<Z>.<A> = <JSON IonSourceInfo>        one line per ion
Z,A,energy_MeV_u,<column>,<column>,...
6,12,0.001,...
```

Values are written with ``repr`` so a file round-trips bit for bit.
"""

from __future__ import annotations

import csv
import io
import json
import math
from collections.abc import Mapping, Sequence
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any, Literal, Protocol

import numpy as np
import numpy.typing as npt
from scipy.interpolate import CubicSpline, PchipInterpolator

from pyradbio.track.stopping.base import IonSourceInfo, OutOfRangeError

Interpolation = Literal["cubic", "pchip"]
"""Interpolants available in log-log space."""

TABLE_FORMAT = "pyradbio-stopping-table/1"
_META_PREFIX = "#@ "
_ION_KEY = "ion."
_INDEX_COLUMNS = ("Z", "A", "energy_MeV_u")


class _ReadableText(Protocol):
    """Anything with ``read_text`` -- a path or an ``importlib.resources`` traversable."""

    def read_text(self, encoding: str | None = ...) -> str: ...


@dataclass(frozen=True)
class LogEnergyGrid:
    """``n_points`` log-uniform energies from ``e_min_MeV_u`` to ``e_max_MeV_u`` inclusive."""

    e_min_MeV_u: float
    e_max_MeV_u: float
    n_points: int

    def __post_init__(self) -> None:
        if not 0.0 < self.e_min_MeV_u < self.e_max_MeV_u:
            raise ValueError(
                "grid needs 0 < e_min_MeV_u < e_max_MeV_u, got "
                f"{(self.e_min_MeV_u, self.e_max_MeV_u)!r}"
            )
        if self.n_points < 2:
            raise ValueError(f"grid needs at least 2 points, got {self.n_points}")

    @classmethod
    def from_points_per_decade(
        cls, e_min_MeV_u: float, e_max_MeV_u: float, points_per_decade: float
    ) -> LogEnergyGrid:
        """The smallest grid with at least ``points_per_decade`` intervals per decade."""
        if points_per_decade <= 0.0:
            raise ValueError(f"points_per_decade must be > 0, got {points_per_decade!r}")
        decades = math.log10(e_max_MeV_u / e_min_MeV_u) if e_min_MeV_u > 0.0 else 0.0
        n_intervals = max(1, math.ceil(decades * points_per_decade - 1e-9))
        return cls(e_min_MeV_u, e_max_MeV_u, n_intervals + 1)

    @property
    def energies(self) -> npt.NDArray[np.float64]:
        """The node energies [MeV/u]; first and last are exactly the endpoints."""
        e = np.geomspace(self.e_min_MeV_u, self.e_max_MeV_u, self.n_points, dtype=np.float64)
        e[0], e[-1] = self.e_min_MeV_u, self.e_max_MeV_u
        return e

    def to_dict(self) -> dict[str, Any]:
        return {
            "e_min_MeV_u": self.e_min_MeV_u,
            "e_max_MeV_u": self.e_max_MeV_u,
            "n_points": self.n_points,
        }

    @classmethod
    def from_dict(cls, data: Mapping[str, Any]) -> LogEnergyGrid:
        return cls(float(data["e_min_MeV_u"]), float(data["e_max_MeV_u"]), int(data["n_points"]))


def _validated_nodes(
    energy: npt.ArrayLike, values: npt.ArrayLike
) -> tuple[npt.NDArray[np.float64], npt.NDArray[np.float64]]:
    e = np.asarray(energy, dtype=np.float64)
    s = np.asarray(values, dtype=np.float64)
    if e.ndim != 1 or s.shape != e.shape:
        raise ValueError("energy and values must be one-dimensional and of equal length")
    if e.size < 2:
        raise ValueError("a table needs at least two nodes")
    if not (np.all(np.isfinite(e)) and np.all(np.isfinite(s))):
        raise ValueError("table nodes must be finite")
    if np.any(e <= 0.0) or np.any(s <= 0.0):
        raise ValueError("log-log interpolation needs positive energies and values")
    if np.any(np.diff(e) <= 0.0):
        raise ValueError("table energies must be strictly increasing")
    return e, s


def _log_log_interpolant(
    e: npt.NDArray[np.float64], s: npt.NDArray[np.float64], method: str
) -> CubicSpline | PchipInterpolator:
    if method == "cubic":
        return CubicSpline(np.log(e), np.log(s), bc_type="not-a-knot")
    if method == "pchip":
        return PchipInterpolator(np.log(e), np.log(s))
    raise ValueError(f"method must be 'cubic' or 'pchip', got {method!r}")


@dataclass(frozen=True, eq=False)
class RegriddedTable:
    """One ion's values on strictly increasing nodes, interpolated in log-log space.

    Calling the table evaluates the interpolant; any energy outside
    ``[energy_MeV_u[0], energy_MeV_u[-1]]`` raises [`OutOfRangeError`][].
    """

    energy_MeV_u: npt.NDArray[np.float64]
    values: npt.NDArray[np.float64]
    method: Interpolation = "cubic"
    _interpolant: CubicSpline | PchipInterpolator = field(init=False, repr=False)

    def __post_init__(self) -> None:
        e, s = _validated_nodes(self.energy_MeV_u, self.values)
        e.setflags(write=False)
        s.setflags(write=False)
        object.__setattr__(self, "energy_MeV_u", e)
        object.__setattr__(self, "values", s)
        object.__setattr__(self, "_interpolant", _log_log_interpolant(e, s, self.method))

    @property
    def energy_range(self) -> tuple[float, float]:
        return float(self.energy_MeV_u[0]), float(self.energy_MeV_u[-1])

    def __call__(self, energy_MeV_u: npt.ArrayLike) -> npt.NDArray[np.float64]:
        e = np.atleast_1d(np.asarray(energy_MeV_u, dtype=np.float64))
        lo, hi = self.energy_range
        if e.size and (e.min() < lo or e.max() > hi):
            raise OutOfRangeError((float(e.min()), float(e.max())), (lo, hi))
        out = np.exp(self._interpolant(np.log(e)))
        # Interpolation is exact at the nodes up to exp(log()) rounding; return the
        # node values themselves there so a table reproduces its own data bit for bit.
        idx = np.searchsorted(self.energy_MeV_u, e)
        idx = np.clip(idx, 0, self.energy_MeV_u.size - 1)
        hit = self.energy_MeV_u[idx] == e
        out[hit] = self.values[idx[hit]]
        return np.asarray(out, dtype=np.float64)

    @classmethod
    def from_source(
        cls,
        source_energy_MeV_u: npt.ArrayLike,
        source_values: npt.ArrayLike,
        grid: LogEnergyGrid,
        method: Interpolation = "cubic",
    ) -> RegriddedTable:
        """Interpolate source nodes onto ``grid``; the result interpolates with ``method`` too.

        :raises OutOfRangeError: if the grid reaches outside the source nodes.
        """
        source = cls(np.asarray(source_energy_MeV_u), np.asarray(source_values), method)
        energies = grid.energies
        return cls(energies, source(energies), method)


def leave_one_out_relative_error(
    energy_MeV_u: npt.ArrayLike, values: npt.ArrayLike, method: Interpolation = "cubic"
) -> npt.NDArray[np.float64]:
    """Relative error ``S_interp / S - 1`` at each interior node with that node removed.

    Removing a node doubles the local spacing, so for a smooth curve this
    over-estimates the error of interpolating the full table: a conservative,
    data-only bound where no truth exists between source points.
    """
    e, s = _validated_nodes(energy_MeV_u, values)
    if e.size < 3:
        raise ValueError("leave-one-out needs at least three nodes")
    errors = np.empty(e.size - 2, dtype=np.float64)
    for k in range(1, e.size - 1):
        keep = np.arange(e.size) != k
        interpolant = _log_log_interpolant(e[keep], s[keep], method)
        errors[k - 1] = math.exp(float(interpolant(math.log(e[k])))) / s[k] - 1.0
    return errors


# --- CSV ---------------------------------------------------------------------------------


@dataclass(frozen=True, eq=False)
class IonTable:
    """One ion's tabulated columns (e.g. electronic and nuclear) on one energy axis."""

    info: IonSourceInfo
    energy_MeV_u: npt.NDArray[np.float64]
    columns: Mapping[str, npt.NDArray[np.float64]]

    def __post_init__(self) -> None:
        e = np.asarray(self.energy_MeV_u, dtype=np.float64)
        columns = {name: np.asarray(v, dtype=np.float64) for name, v in self.columns.items()}
        for name, column in columns.items():
            if column.shape != e.shape:
                raise ValueError(
                    f"column {name!r} has length {column.size}, energy axis has {e.size}"
                )
            if name in _INDEX_COLUMNS:
                raise ValueError(f"column name {name!r} is reserved")
        object.__setattr__(self, "energy_MeV_u", e)
        object.__setattr__(self, "columns", columns)


@dataclass(frozen=True)
class TableFile:
    """A parsed shipped-table file: notes, file-level metadata and per-ion tables."""

    notes: tuple[str, ...]
    metadata: Mapping[str, Any]
    tables: tuple[IonTable, ...]

    def table(self, Z: int, A: int) -> IonTable:
        for t in self.tables:
            if (t.info.Z, t.info.A) == (Z, A):
                return t
        raise KeyError(f"no table for Z={Z} A={A}")


def write_table_csv(
    path: str | Path,
    tables: Sequence[IonTable],
    metadata: Mapping[str, Any],
    notes: Sequence[str] = (),
) -> None:
    """Write ``tables`` in the ``pyradbio-stopping-table/1`` format (see module docstring)."""
    if not tables:
        raise ValueError("nothing to write")
    names = list(tables[0].columns)
    for t in tables:
        if list(t.columns) != names:
            raise ValueError(
                f"all tables must have the same columns {names}, got {list(t.columns)}"
            )
    if "format" in metadata or any(key.startswith(_ION_KEY) for key in metadata):
        raise ValueError(f"metadata keys 'format' and '{_ION_KEY}*' are reserved")

    out = io.StringIO()
    for note in notes:
        out.write(f"# {note}".rstrip() + "\n")
    out.write(f"{_META_PREFIX}format = {json.dumps(TABLE_FORMAT)}\n")
    for key, value in metadata.items():
        out.write(f"{_META_PREFIX}{key} = {json.dumps(value, sort_keys=True)}\n")
    for t in tables:
        key = f"{_ION_KEY}{t.info.Z}.{t.info.A}"
        out.write(f"{_META_PREFIX}{key} = {json.dumps(t.info.to_dict(), sort_keys=True)}\n")
    writer = csv.writer(out, lineterminator="\n")
    writer.writerow([*_INDEX_COLUMNS, *names])
    for t in tables:
        for i, energy in enumerate(t.energy_MeV_u):
            row = [t.info.Z, t.info.A, repr(float(energy))]
            writer.writerow(row + [repr(float(t.columns[name][i])) for name in names])
    Path(path).write_text(out.getvalue(), encoding="utf-8", newline="\n")


def parse_table_csv(text: str) -> TableFile:
    """Parse the text of a ``pyradbio-stopping-table/1`` file."""
    notes: list[str] = []
    meta: dict[str, Any] = {}
    body: list[str] = []
    for line in text.splitlines():
        if line.startswith(_META_PREFIX):
            key, sep, value = line[len(_META_PREFIX) :].partition(" = ")
            if not sep:
                raise ValueError(f"malformed metadata line {line!r}")
            meta[key] = json.loads(value)
        elif line.startswith("#"):
            notes.append(line[2:] if line.startswith("# ") else line[1:])
        elif line.strip():
            body.append(line)
    if meta.pop("format", None) != TABLE_FORMAT:
        raise ValueError(f"not a {TABLE_FORMAT} file (missing or unknown format line)")
    infos = {
        key: IonSourceInfo.from_dict(meta.pop(key))
        for key in list(meta)
        if key.startswith(_ION_KEY)
    }

    reader = csv.reader(body)
    header = next(reader, None)
    if header is None or tuple(header[:3]) != _INDEX_COLUMNS:
        raise ValueError(f"table header must start with {','.join(_INDEX_COLUMNS)}")
    names = header[3:]
    rows: dict[str, list[list[float]]] = {}
    for row in reader:
        rows.setdefault(f"{_ION_KEY}{int(row[0])}.{int(row[1])}", []).append(
            [float(x) for x in row[2:]]
        )
    tables = []
    for key, values in rows.items():
        if key not in infos:
            raise ValueError(f"table body has rows for {key} but no '{_META_PREFIX}{key}' line")
        arr = np.asarray(values, dtype=np.float64)
        columns = {name: arr[:, j + 1] for j, name in enumerate(names)}
        tables.append(IonTable(infos[key], arr[:, 0], columns))
    return TableFile(tuple(notes), meta, tuple(tables))


def read_table_csv(path: str | Path | _ReadableText) -> TableFile:
    """Read a ``pyradbio-stopping-table/1`` file from disk or package resources."""
    source = Path(path) if isinstance(path, str) else path
    return parse_table_csv(source.read_text(encoding="utf-8"))


__all__ = [
    "TABLE_FORMAT",
    "Interpolation",
    "IonTable",
    "LogEnergyGrid",
    "RegriddedTable",
    "TableFile",
    "leave_one_out_relative_error",
    "parse_table_csv",
    "read_table_csv",
    "write_table_csv",
]
