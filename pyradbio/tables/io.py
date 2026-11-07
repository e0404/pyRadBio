"""Serialization: CSV / Parquet and interop with pyMKM .txt tables.

CSV uses only the standard library, so it works without the ``tables`` extra.
Parquet (and any pandas view) needs ``pyarrow`` from that extra and is guarded.
The pyMKM classic ``.txt`` z* format is documented in
``docs/references/pyMKM_API.md`` and round-tripped by
:func:`write_pymkm_txt` / :func:`read_pymkm_txt`.
"""

from __future__ import annotations

import csv
import json
from pathlib import Path
from typing import Any

from pyradbio.core.elements import symbol_to_z, z_to_symbol
from pyradbio.tables.alpha_beta import COLUMNS, AlphaBetaRow, AlphaBetaTable
from pyradbio.tables.zstar import ZStarRow, ZStarTable

# Leading CSV comment line carrying JSON metadata, e.g.
#   # pyradbio-meta: {"model": "mkm", "alpha_x": 0.313}
_META_PREFIX = "# pyradbio-meta: "


def _format_float(value: float) -> str:
    """``repr`` round-trips floats exactly (Python >= 3.1) and stays compact."""
    return repr(float(value))


def write_csv(table: AlphaBetaTable, path: str | Path) -> None:
    """Write ``table`` as CSV with a JSON metadata comment header."""
    path = Path(path)
    with path.open("w", newline="", encoding="utf-8") as fh:
        if table.metadata:
            fh.write(_META_PREFIX + json.dumps(dict(table.metadata)) + "\n")
        writer = csv.writer(fh)
        writer.writerow(COLUMNS)
        for row in table.rows:
            writer.writerow(
                [
                    row.Z,
                    row.A,
                    _format_float(row.energy_MeV_u),
                    _format_float(row.alpha),
                    _format_float(row.beta),
                ]
            )


def read_csv(path: str | Path) -> AlphaBetaTable:
    """Read a table written by :func:`write_csv` back into an AlphaBetaTable."""
    path = Path(path)
    metadata: dict[str, object] = {}
    with path.open("r", newline="", encoding="utf-8") as fh:
        first = fh.readline()
        if first.startswith(_META_PREFIX):
            metadata = json.loads(first[len(_META_PREFIX) :])
        else:
            # No metadata line — rewind so the header row is read below.
            fh.seek(0)
        reader = csv.DictReader(fh)
        rows = tuple(
            AlphaBetaRow(
                Z=int(r["Z"]),
                A=int(r["A"]),
                energy_MeV_u=float(r["energy_MeV_u"]),
                alpha=float(r["alpha"]),
                beta=float(r["beta"]),
            )
            for r in reader
        )
    return AlphaBetaTable(rows=rows, metadata=metadata)


def _require_pyarrow() -> tuple[Any, Any]:
    try:
        import pyarrow as pa
        import pyarrow.parquet as pq
    except ImportError as exc:  # pragma: no cover - exercised via extra gating
        raise ImportError(
            "Parquet I/O needs the 'tables' extra: pip install 'pyradbio[tables]'."
        ) from exc
    return pa, pq


def write_parquet(table: AlphaBetaTable, path: str | Path) -> None:
    """Write ``table`` to Parquet, stashing metadata in the file's schema."""
    pa, pq = _require_pyarrow()
    arr = {
        "Z": [r.Z for r in table.rows],
        "A": [r.A for r in table.rows],
        "energy_MeV_u": [r.energy_MeV_u for r in table.rows],
        "alpha": [r.alpha for r in table.rows],
        "beta": [r.beta for r in table.rows],
    }
    pa_table = pa.table({c: arr[c] for c in COLUMNS})
    if table.metadata:
        meta = {b"pyradbio-meta": json.dumps(dict(table.metadata)).encode("utf-8")}
        pa_table = pa_table.replace_schema_metadata(meta)
    pq.write_table(pa_table, str(path))


def read_parquet(path: str | Path) -> AlphaBetaTable:
    """Read a table written by :func:`write_parquet`."""
    _pa, pq = _require_pyarrow()
    pa_table = pq.read_table(str(path))
    cols = pa_table.to_pydict()
    rows = tuple(
        AlphaBetaRow(
            Z=int(z),
            A=int(a),
            energy_MeV_u=float(e),
            alpha=float(al),
            beta=float(be),
        )
        for z, a, e, al, be in zip(
            cols["Z"], cols["A"], cols["energy_MeV_u"], cols["alpha"], cols["beta"], strict=True
        )
    )
    metadata: dict[str, object] = {}
    schema_meta = pa_table.schema.metadata or {}
    raw = schema_meta.get(b"pyradbio-meta")
    if raw is not None:
        metadata = json.loads(raw.decode("utf-8"))
    return AlphaBetaTable(rows=rows, metadata=metadata)


def write_pymkm_txt(table: ZStarTable, path: str | Path) -> None:
    """Write a z* table in pyMKM's classic ``.txt`` format (MKTable.write_txt).

    Two-space ``CellType`` line, ``.4f`` header parameters, ``.5e`` data columns
    (``energy z*_1D``), fragments grouped by element with a trailing blank line.
    ``newline=""`` keeps line endings LF, matching pyMKM's output byte-for-byte.
    """
    path = Path(path)
    parts: list[str] = [
        f"CellType  {table.cell_type}\n\n",
        f"Parameter Alpha_0 {table.alpha0_Gy:.4f}\n",
        f"Parameter Beta {table.beta_Gy2:.4f}\n",
        f"Parameter DomainRadius {table.domain_radius_um:.4f}\n",
        f"Parameter NucleusRadius {table.nucleus_radius_um:.4f}\n\n",
    ]
    for z in dict.fromkeys(row.Z for row in table.rows):
        parts.append(f"Fragment {z_to_symbol(z)}\n")
        for row in table.rows:
            if row.Z == z:
                parts.append(f"{row.energy_MeV_u:.5e} {row.z_star_1D_Gy:.5e}\n")
        parts.append("\n")
    with path.open("w", encoding="utf-8", newline="") as fh:
        fh.write("".join(parts))


def read_pymkm_txt(path: str | Path) -> ZStarTable:
    """Read a pyMKM classic ``.txt`` z* table into a :class:`ZStarTable`.

    Tolerates the stochastic (SMK) 4-column layout by keeping the
    ``z_bar_star_domain`` column.
    """
    path = Path(path)
    cell_type = ""
    params: dict[str, float] = {}
    rows: list[ZStarRow] = []
    current_z: int | None = None
    for raw in path.read_text(encoding="utf-8").splitlines():
        line = raw.strip()
        if not line:
            continue
        if line.startswith("CellType"):
            cell_type = line.split(None, 1)[1].strip()
        elif line.startswith("Parameter"):
            _, name, value = line.split()
            params[name] = float(value)
        elif line.startswith("Fragment"):
            current_z = symbol_to_z(line.split(None, 1)[1].strip())
        else:
            if current_z is None:
                raise ValueError("Data row before any 'Fragment' line in pyMKM .txt.")
            cols = line.split()
            # classic: [energy, z*]; stochastic: [energy, z_bar, z*, z_nucleus].
            z_star = float(cols[2]) if len(cols) >= 4 else float(cols[1])
            rows.append(ZStarRow(Z=current_z, energy_MeV_u=float(cols[0]), z_star_1D_Gy=z_star))
    return ZStarTable(
        rows=tuple(rows),
        cell_type=cell_type,
        alpha0_Gy=params["Alpha_0"],
        beta_Gy2=params["Beta"],
        domain_radius_um=params["DomainRadius"],
        nucleus_radius_um=params["NucleusRadius"],
    )
