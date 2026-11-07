"""Source-agnostic reference records for the validation harness.

Every measured-survival database this package can score a model against --
PIDE (:mod:`pyradbio.benchmark.pide`), RadPhysBio
(:mod:`pyradbio.benchmark.radphysbio`), or a user's own CSV
(:func:`load_reference_csv`) -- is reduced to the same small record,
:class:`ReferenceRecord`: one ion exposure with its matched photon reference,
both as LQ parameters under ``-ln S = alpha*d + beta*d^2``. The harness in
:mod:`pyradbio.benchmark.harness` reads nothing else, so a new database is a new
loader that returns these records, plus the caveats a reader must know before
believing a score against it (:class:`ReferenceDataset`).

Where a source carries more than the harness reads (PIDE's fit/paper variants,
RadPhysBio's DSB yields and dose rates), the source module keeps its own richer
row type and converts; the record here is the *interface*, not the archive.

The :func:`load_dataset` registry is what the CLI drives (``--dataset``); it is
a plain dict so a user can register a loader for a private dataset without
touching this package.
"""

from __future__ import annotations

import csv
import re
from collections.abc import Callable, Iterable, Mapping, Sequence
from dataclasses import dataclass
from pathlib import Path

from pyradbio.core.elements import SYMBOL_TO_Z

__all__ = [
    "DATASET_LOADERS",
    "ReferenceDataset",
    "ReferenceRecord",
    "clean_str",
    "dataset_names",
    "is_missing",
    "load_dataset",
    "load_reference_csv",
    "parse_ion_token",
    "parse_reference_rows",
    "read_table",
    "to_float",
]


@dataclass(frozen=True, slots=True)
class ReferenceRecord:
    """One ion exposure with its matched photon reference, as LQ parameters.

    All LQ parameters follow ``-ln S = alpha*d + beta*d^2`` (Gy^-1 / Gy^-2).
    ``A`` is the mass number when the source encodes it, else ``None`` (the
    harness then takes the most abundant isotope). ``energy_MeV_u`` is the
    specific kinetic energy at the target, unless ``energy_is_nominal`` is true;
    ``None`` when the source gives only a LET. Nominal beam energies are metadata
    for LET-based evaluation and must never be used to construct a target ion.
    ``source`` names where the row came from (``"pide:<PublicationName>"``,
    ``"radphysbio:11025645"``, ``"csv:row 12"``) so a residual can be traced back.
    """

    cell_line: str
    Z: int
    energy_MeV_u: float | None
    let_keV_um: float
    alpha_ion: float
    beta_ion: float
    alpha_x: float
    beta_x: float
    A: int | None = None
    #: Reference photon quality as a raw string (e.g. ``"200kVp"``, ``"60Co"``,
    #: ``"6MV"``); ``""`` when absent. Kept verbatim -- mapping a quality to a
    #: reference dose-mean lineal energy is a model concern, not the loader's.
    photon_radiation: str = ""
    #: Provenance key, ``"<dataset>:<publication or row>"``.
    source: str = ""
    #: True when the source gives a nominal beam energy rather than a verified
    #: target energy. The harness rejects these records for energy-based models.
    energy_is_nominal: bool = False
    #: Skip records *without* an energy on the LET route too, keeping PIDE model
    #: comparisons on a shared candidate set (the historical default). CSV
    #: loaders set False because energy is optional there. A known energy below
    #: the harness floor gates either way, and the ion route always needs one.
    require_energy_for_let: bool = True

    @property
    def rbe_alpha(self) -> float:
        """Initial-slope RBE ``alpha_ion / alpha_x`` (undefined if ``alpha_x==0``)."""
        return self.alpha_ion / self.alpha_x


@dataclass(frozen=True, slots=True)
class ReferenceDataset:
    """A named set of :class:`ReferenceRecord` plus what a reader must know.

    ``caveats`` are rendered verbatim into every benchmark report against this
    dataset, next to the model's own. A dataset that was largely digitized, or
    whose LET column is partly derived, says so here rather than in a docstring
    nobody reads at report time.
    """

    name: str
    records: tuple[ReferenceRecord, ...]
    citation: str = ""
    caveats: tuple[str, ...] = ()

    def __len__(self) -> int:
        return len(self.records)


# --- shared cell / header helpers -------------------------------------------

# Latin N/A plus the Greek capital/small nu and alpha look-alikes (U+039D, U+03BD,
# U+0391, U+03B1) that the RadPhysBio export mixes in.
_MISSING_RE = re.compile(r"^\s*[Nn\u039d\u03bd]\s*/\s*[Aa\u0391\u03b1]\s*$")


def is_missing(value: object) -> bool:
    """True for ``None``, blank, or an ``N/A`` marker.

    Accepts the Latin ``N/A`` and its Greek look-alikes (a Greek capital nu
    and/or a Greek capital alpha in place of the Latin letters), which the
    RadPhysBio export carries in a few hundred cells.
    """
    if value is None:
        return True
    text = str(value).strip()
    return not text or _MISSING_RE.match(text) is not None


def to_float(value: object) -> float | None:
    """Best-effort float parse; ``None``/blank/``N/A``/non-numeric -> ``None``."""
    if isinstance(value, bool):
        return None
    if isinstance(value, (int, float)):
        return float(value)
    if is_missing(value):
        return None
    try:
        return float(str(value).strip())
    except ValueError:
        return None


def clean_str(value: object) -> str:
    """Trim a text cell; treat blank / ``N/A`` (any spelling) as ``""``."""
    return "" if is_missing(value) else str(value).strip()


def normalise_header(header: str) -> str:
    """Lower-case ``header`` and drop every non-alphanumeric character."""
    return re.sub(r"[^a-z0-9]", "", header.lower())


_ION_RE = re.compile(r"^\s*(\d+)?\s*[- ]?\s*([A-Za-z]+)\s*[- ]?\s*(\d+)?\s*$")


def parse_ion_token(token: str) -> tuple[int | None, int | None]:
    """Parse an ion token like ``"C-12"``/``"12C"``/``"C"`` into ``(Z, A)``.

    Returns ``(None, None)`` when no element symbol is recognised (e.g. a purely
    numeric token, which is ambiguous between Z and A and is deferred to an
    explicit charge column). ``A`` is ``None`` when the token carries no mass
    number.
    """
    match = _ION_RE.match(token)
    if match is None:
        return (None, None)
    lead, symbol, trail = match.groups()
    key = symbol.capitalize()
    if key not in SYMBOL_TO_Z:
        return (None, None)
    mass = lead or trail
    return (SYMBOL_TO_Z[key], int(mass) if mass else None)


# --- file readers ------------------------------------------------------------

Table = tuple[list[str], list[list[object]]]


def _read_xlsx(path: Path) -> Table:
    """Read the first worksheet into ``(headers, rows)`` via openpyxl."""
    try:
        from openpyxl import load_workbook
    except ModuleNotFoundError as exc:  # pragma: no cover - exercised via extra gating
        raise ImportError(
            "Reading an .xlsx needs the 'tables' extra: pip install 'pyradbio[tables]'."
        ) from exc

    wb = load_workbook(path, read_only=True, data_only=True)
    try:
        ws = wb.active
        rows_iter = ws.iter_rows(values_only=True)
        try:
            header = next(rows_iter)
        except StopIteration:
            return ([], [])
        headers = ["" if h is None else str(h) for h in header]
        data = [list(r) for r in rows_iter]
        return (headers, data)
    finally:
        wb.close()


def _read_csv(path: Path) -> Table:
    """Read a delimited export into ``(headers, rows)``.

    Sniffs comma/semicolon/tab; falls back to comma. Rows are *not* padded or
    trimmed -- a loader that knows its schema decides what a ragged row means.
    """
    text = path.read_text(encoding="utf-8-sig")
    sample = text[:4096]
    try:
        dialect: type[csv.Dialect] | csv.Dialect = csv.Sniffer().sniff(sample, delimiters=",;\t")
    except csv.Error:
        dialect = csv.excel
    reader = csv.reader(text.splitlines(), dialect)
    all_rows = [row for row in reader if row]
    if not all_rows:
        return ([], [])
    return (all_rows[0], [list(r) for r in all_rows[1:]])


def read_table(path: str | Path) -> Table:
    """``(headers, rows)`` of a ``.xlsx``/``.xls`` workbook or a delimited text file."""
    p = Path(path)
    if p.suffix.lower() in (".xlsx", ".xls"):
        return _read_xlsx(p)
    return _read_csv(p)


# --- generic column-aliased parser (custom CSVs) -----------------------------

#: Logical field -> accepted *normalised* header aliases, in priority order (see
#: :func:`normalise_header`): the first alias present in the header row wins.
REFERENCE_ALIASES: Mapping[str, tuple[str, ...]] = {
    "cell_line": ("cells", "cell", "cellline", "celltype", "name"),
    "Z": ("charge", "z", "atomicnumber"),
    "ion": ("ion", "particle", "projectile", "species"),
    "energy_MeV_u": ("energy", "e", "emevu", "energymevu", "specificenergy"),
    "let_keV_um": ("let", "letkevum", "letd", "doseaveragedlet"),
    "alpha_ion": ("alphaion", "alphai", "alphal", "alpha", "aipaper", "aifit"),
    "beta_ion": ("betaion", "betai", "betal", "beta", "bipaper", "bifit"),
    "alpha_x": ("alphax", "alpharef", "alphaphoton", "axpaper", "axfit"),
    "beta_x": ("betax", "betaref", "betaphoton", "bxpaper", "bxfit"),
    "photon_radiation": ("photonradiation", "photon", "referenceradiation", "reference"),
    "source": ("publicationname", "source", "publication", "doi", "pmid"),
}

_REQUIRED = ("let_keV_um", "alpha_ion", "beta_ion", "alpha_x", "beta_x")


def parse_reference_rows(
    headers: Sequence[str],
    rows: Iterable[Sequence[object]],
    *,
    strict: bool = False,
    dataset: str = "csv",
) -> list[ReferenceRecord]:
    """Turn a header row + data rows into :class:`ReferenceRecord` records.

    Columns match logical fields via :data:`REFERENCE_ALIASES`, header-tolerant
    (case and punctuation ignored). The ion is resolved from a ``Charge``/``Z``
    column when present, else parsed from an ``Ion`` token (``C-12``, ``12C``,
    ``C``). Rows missing a required numeric field (LET and the four LQ values)
    or whose ion cannot be resolved are skipped -- unless ``strict``, when the
    first such row raises ``ValueError``. ``energy`` is optional.
    """
    norm_to_idx: dict[str, int] = {}
    for idx, head in enumerate(headers):
        norm_to_idx.setdefault(normalise_header(str(head)), idx)
    col: dict[str, int] = {}
    for fld, aliases in REFERENCE_ALIASES.items():
        for alias in aliases:
            if alias in norm_to_idx:
                col[fld] = norm_to_idx[alias]
                break

    def cell(row: Sequence[object], fld: str) -> object:
        idx = col.get(fld)
        return row[idx] if idx is not None and idx < len(row) else None

    out: list[ReferenceRecord] = []
    for row_no, row in enumerate(rows):
        z_val = to_float(cell(row, "Z"))
        z: int | None = round(z_val) if z_val is not None else None
        a: int | None = None
        ion_raw = cell(row, "ion")
        if ion_raw is not None and str(ion_raw).strip():
            z_tok, a = parse_ion_token(str(ion_raw))
            if z is None:
                z = z_tok
        numeric = {f: to_float(cell(row, f)) for f in _REQUIRED}
        missing = [f for f, v in numeric.items() if v is None]
        if z is None or missing:
            if strict:
                reason = "unresolved ion Z" if z is None else f"missing {missing}"
                raise ValueError(f"{dataset} row {row_no}: {reason}")
            continue
        src = clean_str(cell(row, "source"))
        out.append(
            ReferenceRecord(
                cell_line=clean_str(cell(row, "cell_line")),
                Z=z,
                A=a,
                energy_MeV_u=to_float(cell(row, "energy_MeV_u")),
                require_energy_for_let=False,
                let_keV_um=numeric["let_keV_um"],  # type: ignore[arg-type]
                alpha_ion=numeric["alpha_ion"],  # type: ignore[arg-type]
                beta_ion=numeric["beta_ion"],  # type: ignore[arg-type]
                alpha_x=numeric["alpha_x"],  # type: ignore[arg-type]
                beta_x=numeric["beta_x"],  # type: ignore[arg-type]
                photon_radiation=clean_str(cell(row, "photon_radiation")),
                source=f"{dataset}:{src or f'row {row_no}'}",
            )
        )
    return out


def load_reference_csv(
    path: str | Path, *, strict: bool = False, name: str = "csv"
) -> ReferenceDataset:
    """Load a user-supplied ``.csv``/``.xlsx`` of paired ion/photon LQ parameters.

    Expected columns (header-tolerant; see :data:`REFERENCE_ALIASES`): ``cell``,
    ``ion`` (or ``charge``), ``energy`` (MeV/u, optional), ``let`` (keV/um),
    ``alpha_ion``, ``beta_ion``, ``alpha_x``, ``beta_x``, and optionally
    ``photon_radiation`` and ``source``. A missing energy does not exclude a
    row from LET-based models (``require_energy_for_let=False``), but a given
    energy below the harness floor still does; energy-based models require a
    target energy within their supported range. This is the extension point for a
    private dataset: write it in this shape and run ``pyradbio benchmark
    --dataset csv --path FILE``.
    """
    headers, rows = read_table(path)
    records = parse_reference_rows(headers, rows, strict=strict, dataset=name)
    return ReferenceDataset(
        name=name,
        records=tuple(records),
        citation=f"user-supplied file {Path(path).name}",
        caveats=(
            "A user-supplied dataset: this package has not checked its provenance, "
            "its LQ convention (must be -ln S = alpha*d + beta*d^2) or its LET "
            "definition.",
        ),
    )


# --- registry ---------------------------------------------------------------

#: A dataset loader: ``(path or None) -> ReferenceDataset``. ``None`` means "use
#: the source's configured default location".
DatasetLoader = Callable[[str | Path | None], ReferenceDataset]


def _load_pide_dataset(path: str | Path | None) -> ReferenceDataset:
    from pyradbio.benchmark.pide import load_pide_dataset

    return load_pide_dataset(path)


def _load_radphysbio_dataset(path: str | Path | None) -> ReferenceDataset:
    from pyradbio.benchmark.radphysbio import load_radphysbio_dataset

    return load_radphysbio_dataset(path)


def _load_csv_dataset(path: str | Path | None) -> ReferenceDataset:
    if path is None:
        raise FileNotFoundError("--dataset csv needs an explicit --path FILE")
    return load_reference_csv(path)


#: Dataset name -> loader. Extend in user code:
#: ``DATASET_LOADERS["mine"] = lambda p: ReferenceDataset(...)``.
DATASET_LOADERS: dict[str, DatasetLoader] = {
    "pide": _load_pide_dataset,
    "radphysbio": _load_radphysbio_dataset,
    "csv": _load_csv_dataset,
}


def dataset_names() -> tuple[str, ...]:
    """Every registered dataset name, in registration order."""
    return tuple(DATASET_LOADERS)


def load_dataset(name: str, path: str | Path | None = None) -> ReferenceDataset:
    """Load the registered dataset ``name`` (see :data:`DATASET_LOADERS`)."""
    try:
        loader = DATASET_LOADERS[name]
    except KeyError:
        raise ValueError(f"unknown dataset {name!r}; choose from {dataset_names()}") from None
    return loader(path)
