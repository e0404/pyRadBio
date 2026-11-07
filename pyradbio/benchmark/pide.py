"""Loader for a user-supplied PIDE file.

PIDE (Particle Irradiation Data Ensemble, GSI; Friedrich et al. 2013,
doi:10.1093/jrr/rrs114; update Friedrich et al. 2021, doi:10.1093/jrr/rrab034) is
distributed as an Excel workbook (``PIDE3.4.xlsx``) after registration with an
institutional e-mail. It is NOT redistributable and NOT bundled here.

By default the loader reads from the local, git-ignored ``data/pide/`` folder
(see :func:`default_pide_dir`); drop your own PIDE export there. An explicit
path always overrides this: ``--pide PATH`` / ``find_pide_file(path)``. The
``PYRADBIO_PIDE_DIR`` environment variable overrides the default folder.

Tested against the schema documented in ``docs/references/pide_schema.md`` (LQ
formalism ``-ln S = alpha*d + beta*d^2`` for both the ion exposure and its matched
photon reference). The loader matches each logical field against a set of
aliases (:data:`_ALIASES`) rather than a fixed name, so it also reads the synthetic
CI fixture. Each PIDE row already pairs an ion exposure with its photon reference,
so a row maps one-to-one onto a :class:`PideEntry`, which *is* the harness's
:class:`~pyradbio.benchmark.dataset.ReferenceRecord`.
"""

from __future__ import annotations

import os
from collections.abc import Iterable, Mapping, Sequence
from dataclasses import dataclass
from pathlib import Path
from typing import Literal, cast

from pyradbio.benchmark.dataset import (
    ReferenceDataset,
    ReferenceRecord,
    clean_str,
    normalise_header,
    parse_ion_token,
    read_table,
    to_float,
)

__all__ = [
    "DEFAULT_PIDE_DIRNAME",
    "PIDE_CAVEATS",
    "PIDE_CITATION",
    "LQVariant",
    "PideEntry",
    "PideObservation",
    "default_pide_dir",
    "find_pide_file",
    "load_pide",
    "load_pide_dataset",
    "load_pide_observations",
    "parse_ion_token",
    "parse_pide_observations",
    "parse_pide_rows",
]

#: Which LQ-parameter flavour to read from PIDE. The workbook carries both the
#: as-published values (``*_paper``) and PIDE's own consistent refits
#: (``*_fit``); the refits cover more rows (~1010 vs ~545 in 3.4), so the default
#: prefers them and falls back to paper per field. See ``pide_schema.md``.
LQVariant = Literal["fit", "paper", "fit_then_paper", "paper_then_fit"]

#: Default folder, relative to the working directory, searched for PIDE files.
#: Git-ignored except for its README (see .gitignore). Override via the
#: PYRADBIO_PIDE_DIR environment variable.
DEFAULT_PIDE_DIRNAME = "data/pide"

#: Spreadsheet/data suffixes recognised as PIDE files, in preference order.
_PIDE_SUFFIXES = (".xlsx", ".xls", ".dat", ".csv", ".ascii", ".txt")

PIDE_CITATION = (
    "Friedrich T, Scholz U, Elsasser T, Durante M, Scholz M. J Radiat Res 2013;54:494-514 "
    "(doi:10.1093/jrr/rrs114); Friedrich T, Pfuhl T, Scholz M. J Radiat Res 2021;62:645-655 "
    "(doi:10.1093/jrr/rrab034). PIDE 3.4."
)

#: What a reader must know before believing a score against PIDE (rendered into
#: every report). Sources: the PIDE 3.4 information sheet, sections 1-2.
PIDE_CAVEATS: tuple[str, ...] = (
    "PIDE pairs every ion exposure with its own photon reference from the same "
    "publication; alpha_x/beta_x are therefore the experiment's own, not a "
    "generic tissue value.",
    "The default LQ variant is PIDE's own unweighted refit of the raw survival data "
    "(`*_fit`, with the plating efficiency re-fitted as an offset), falling back to "
    "the as-published values (`*_paper`) per field. The two can differ "
    "considerably where the publication used a purely linear fit or fixed the "
    "plating efficiency (information sheet, section 2.ii).",
    "Where a publication gave only LET or only energy, PIDE computed the "
    "counterpart with a stopping-power code (ATIMA); for SOBP rows the energy is "
    "the monoenergetic beam that would give the same LET (section 1, note *).",
)


@dataclass(frozen=True, slots=True)
class PideEntry(ReferenceRecord):
    """One PIDE ion-exposure record with its matched photon reference.

    A :class:`~pyradbio.benchmark.dataset.ReferenceRecord` with PIDE's own
    provenance in ``source`` (``"pide:<PublicationName>"``, e.g.
    ``"pide:furusawa00"``). See ``docs/references/pide_schema.md``.
    """


@dataclass(frozen=True, slots=True)
class PideObservation:
    """One PIDE observation before benchmark completeness filtering.

    Missing LET or LQ quantities stay ``None`` so cross-checks can compare the
    available quantities without selecting only complete experiments. Units
    and LQ convention follow :class:`ReferenceRecord` (doi:10.1093/jrr/rrs114).
    Use :func:`load_pide` for the benchmark's complete :class:`PideEntry` records.
    """

    cell_line: str
    Z: int
    energy_MeV_u: float | None
    let_keV_um: float | None
    alpha_ion: float | None
    beta_ion: float | None
    alpha_x: float | None
    beta_x: float | None
    A: int | None = None
    photon_radiation: str = ""
    source: str = ""


# --- header matching -------------------------------------------------------
#: Non-LQ logical field -> set of accepted *normalised* header aliases.
#: Normalisation (see :func:`normalise_header`) lower-cases and strips
#: non-alphanumerics, so ``alpha_X``/``Alpha X``/``alphaX`` all collapse to
#: ``alphax`` and PIDE's ``Cells``/``Charge``/``Energy`` are matched too.
_ALIASES: Mapping[str, tuple[str, ...]] = {
    "cell_line": ("cells", "cell", "cellline", "celltype", "name"),
    "Z": ("charge", "z", "atomicnumber"),
    "ion": ("ion", "particle", "projectile", "species"),
    "energy_MeV_u": ("energy", "e", "emevu", "specificenergy"),
    "let_keV_um": ("let", "letkevum", "letd", "doseaveragedlet"),
    "photon_radiation": ("photonradiation", "photon", "referenceradiation", "reference"),
    # Priority order matters: PIDE has both ``#Publication`` (a running number)
    # and ``PublicationName`` (``furusawa00``); the name is the provenance key.
    "publication": ("publicationname", "publication", "source"),
}

#: LQ field -> (PIDE stem, generic aliases). The stem yields the PIDE ``*_fit`` /
#: ``*_paper`` pair (e.g. ``ai`` -> ``aifit``/``aipaper``); the generic aliases
#: cover other exports and the synthetic fixture (``alpha_I`` -> ``alphai``).
_LQ_FIELDS: Mapping[str, tuple[str, frozenset[str]]] = {
    "alpha_ion": ("ai", frozenset({"alphai", "alphal", "alphaion", "alpha"})),
    "beta_ion": ("bi", frozenset({"betai", "betal", "betaion", "beta"})),
    "alpha_x": ("ax", frozenset({"alphax", "alpharef", "alphaphoton"})),
    "beta_x": ("bx", frozenset({"betax", "betaref", "betaphoton"})),
}


def _variant_order(variant: LQVariant) -> tuple[str, ...]:
    """Suffix priority (``fit``/``paper``) for the chosen LQ variant."""
    return {
        "fit": ("fit",),
        "paper": ("paper",),
        "fit_then_paper": ("fit", "paper"),
        "paper_then_fit": ("paper", "fit"),
    }[variant]


def _parse_pide_observations(
    headers: Sequence[str],
    rows: Iterable[Sequence[object]],
    *,
    strict: bool = False,
    variant: LQVariant = "fit_then_paper",
    require_complete: bool = False,
) -> list[PideObservation]:
    """Shared header/variant parser with an explicit completeness policy."""
    norm_to_idx: dict[str, int] = {}
    for idx, head in enumerate(headers):
        norm_to_idx.setdefault(normalise_header(str(head)), idx)

    # Single-column fields: aliases are tried in *their* order, so a preferred
    # header wins over a fallback wherever it sits in the row.
    col: dict[str, int] = {}
    for field_name, aliases in _ALIASES.items():
        for alias in aliases:
            if alias in norm_to_idx:
                col[field_name] = norm_to_idx[alias]
                break

    # LQ fields: an *ordered* list of candidate columns (variant-prioritised
    # PIDE stems first, then generic aliases); the first that parses wins.
    suffixes = _variant_order(variant)
    lq_candidates: dict[str, list[int]] = {}
    for field_name, (stem, generic) in _LQ_FIELDS.items():
        ordered: list[int] = []
        for suffix in suffixes:
            stem_idx = norm_to_idx.get(stem + suffix)
            if stem_idx is not None:
                ordered.append(stem_idx)
        ordered += [norm_to_idx[n] for n in generic if n in norm_to_idx]
        lq_candidates[field_name] = ordered

    def cell(row: Sequence[object], field_name: str) -> object:
        idx = col.get(field_name)
        return row[idx] if idx is not None and idx < len(row) else None

    def lq_value(row: Sequence[object], field_name: str) -> float | None:
        for idx in lq_candidates[field_name]:
            if idx < len(row):
                val = to_float(row[idx])
                if val is not None:
                    return val
        return None

    entries: list[PideObservation] = []
    for row_no, row in enumerate(rows):
        # Resolve the ion: explicit charge column wins; else parse the ion token.
        z_val = to_float(cell(row, "Z"))
        z: int | None = round(z_val) if z_val is not None else None
        a: int | None = None
        ion_raw = cell(row, "ion")
        if ion_raw is not None and str(ion_raw).strip():
            z_tok, a = parse_ion_token(str(ion_raw))
            if z is None:
                z = z_tok

        numeric = {f: lq_value(row, f) for f in _LQ_FIELDS}
        numeric["let_keV_um"] = to_float(cell(row, "let_keV_um"))
        missing = [f for f, v in numeric.items() if v is None]
        if z is None or (require_complete and missing):
            if strict:
                reason = "unresolved ion Z" if z is None else f"missing {missing}"
                raise ValueError(f"PIDE row {row_no}: {reason}")
            continue

        publication = clean_str(cell(row, "publication"))
        entries.append(
            PideObservation(
                cell_line=clean_str(cell(row, "cell_line")),
                Z=z,
                A=a,
                energy_MeV_u=to_float(cell(row, "energy_MeV_u")),
                let_keV_um=numeric["let_keV_um"],
                alpha_ion=numeric["alpha_ion"],
                beta_ion=numeric["beta_ion"],
                alpha_x=numeric["alpha_x"],
                beta_x=numeric["beta_x"],
                photon_radiation=clean_str(cell(row, "photon_radiation")),
                source=f"pide:{publication or f'row {row_no}'}",
            )
        )
    return entries


def parse_pide_observations(
    headers: Sequence[str],
    rows: Iterable[Sequence[object]],
    *,
    strict: bool = False,
    variant: LQVariant = "fit_then_paper",
) -> list[PideObservation]:
    """Parse PIDE rows while preserving missing LET/LQ quantities as ``None``.

    Header aliases and per-field fit/paper fallback are the same as in
    :func:`parse_pide_rows`. Only unresolved ions are skipped, or raise with
    ``strict=True``. Missing measurements remain missing even in strict mode.
    """
    return _parse_pide_observations(headers, rows, strict=strict, variant=variant)


def parse_pide_rows(
    headers: Sequence[str],
    rows: Iterable[Sequence[object]],
    *,
    strict: bool = False,
    variant: LQVariant = "fit_then_paper",
) -> list[PideEntry]:
    """Parse complete PIDE benchmark records using ordered header aliases.

    ``variant`` selects fit/paper LQ columns with per-field fallback for the
    ``*_then_*`` modes. Unresolved ions and rows missing LET or any LQ quantity
    are skipped, or raise with ``strict=True``. For comparisons of individual
    quantities, use :func:`parse_pide_observations` to retain partial rows.
    """
    observations = _parse_pide_observations(
        headers, rows, strict=strict, variant=variant, require_complete=True
    )
    return [
        PideEntry(
            cell_line=e.cell_line,
            Z=e.Z,
            A=e.A,
            energy_MeV_u=e.energy_MeV_u,
            # The parser's require_complete policy guarantees these quantities.
            let_keV_um=cast(float, e.let_keV_um),
            alpha_ion=cast(float, e.alpha_ion),
            beta_ion=cast(float, e.beta_ion),
            alpha_x=cast(float, e.alpha_x),
            beta_x=cast(float, e.beta_x),
            photon_radiation=e.photon_radiation,
            source=e.source,
        )
        for e in observations
    ]


def default_pide_dir() -> Path:
    """Folder the loader reads PIDE files from when no explicit path is given.

    The ``PYRADBIO_PIDE_DIR`` environment variable overrides the built-in
    ``data/pide`` default. PIDE is never bundled; this is a local, git-ignored
    convenience folder.
    """
    from pyradbio.datasets import load_project_dotenv

    load_project_dotenv()  # let a local .env supply PYRADBIO_PIDE_DIR
    env = os.environ.get("PYRADBIO_PIDE_DIR")
    return Path(env) if env else Path(DEFAULT_PIDE_DIRNAME)


def find_pide_file(path: str | Path | None = None) -> Path:
    """Resolve a PIDE file. Explicit path wins; otherwise discover in the default folder.

    Resolution order:
    1. ``path`` is a file -> use it.
    2. ``path`` is a directory (or ``None`` -> :func:`default_pide_dir`) ->
       pick the highest-preference PIDE file in it (``.xlsx`` first).

    Raises ``FileNotFoundError`` with guidance if nothing usable is found.
    """
    if path is not None:
        p = Path(path)
        if p.is_file():
            return p
        if not p.is_dir():
            raise FileNotFoundError(f"PIDE path does not exist: {p}")
        search_dir = p
    else:
        search_dir = default_pide_dir()

    if search_dir.is_dir():
        candidates = [f for f in sorted(search_dir.iterdir()) if f.suffix.lower() in _PIDE_SUFFIXES]
        candidates.sort(key=lambda f: _PIDE_SUFFIXES.index(f.suffix.lower()))
        if candidates:
            return candidates[0]

    raise FileNotFoundError(
        f"No PIDE file found in {search_dir}. Download PIDE (registration required) "
        "from the GSI PIDE project page and place it there, set PYRADBIO_PIDE_DIR, "
        "or pass an explicit --pide PATH. PIDE is not redistributable; never commit it."
    )


def load_pide(
    path: str | Path | None = None,
    *,
    strict: bool = False,
    variant: LQVariant = "fit_then_paper",
) -> list[PideEntry]:
    """Parse a PIDE ``.xlsx``/``.csv`` export into typed :class:`PideEntry` records.

    ``path`` may be a file, a directory, or ``None`` (use the default folder);
    see :func:`find_pide_file`. ``variant`` selects PIDE's ``*_fit``/``*_paper``
    LQ columns. Incomplete rows are skipped unless ``strict``. Never writes or
    commits the PIDE file itself.
    """
    headers, rows = read_table(find_pide_file(path))
    return parse_pide_rows(headers, rows, strict=strict, variant=variant)


def load_pide_dataset(
    path: str | Path | None = None,
    *,
    variant: LQVariant = "fit_then_paper",
) -> ReferenceDataset:
    """:func:`load_pide` wrapped as a :class:`~pyradbio.benchmark.dataset.ReferenceDataset`."""
    return ReferenceDataset(
        name="PIDE",
        records=tuple(load_pide(path, variant=variant)),
        citation=PIDE_CITATION,
        caveats=(*PIDE_CAVEATS, f"LQ variant read: `{variant}`."),
    )


def load_pide_observations(
    path: str | Path | None = None,
    *,
    strict: bool = False,
    variant: LQVariant = "fit_then_paper",
) -> list[PideObservation]:
    """Read a local PIDE export without discarding partially populated experiments.

    Use this loader with ``crosscheck.rows_from_pide``. File discovery and LQ
    variants follow :func:`load_pide`; missing measurements remain ``None``.
    Nothing is written. PIDE must remain in the user's untracked data folder.
    """
    headers, rows = read_table(find_pide_file(path))
    return parse_pide_observations(headers, rows, strict=strict, variant=variant)
