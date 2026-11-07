"""Cross-check two reference databases experiment by experiment.

Two literature-mined LQ databases that cite the same publication should hold
the same experiment: same cell line, same ion, same LET, and -- if both
transcribed the printed numbers -- the same ``alpha``/``beta``. This module
finds those shared experiments and classifies how well each LQ quantity
agrees, so a reader can decide whether a database is a *transcription* of the
published values, a *digitization* of the figures (typical error of a few
percent), or something else. It was written to assess RadPhysBio
(:mod:`pyradbio.benchmark.radphysbio`, largely digitized) against PIDE
(:mod:`pyradbio.benchmark.pide`, printed ``*_paper`` values and PIDE's own
``*_fit`` refits), but the matcher and the classifier are generic: they read
only :class:`CrossRow`.

Bibliography bridge
-------------------
PIDE keys its rows by a short publication name; RadPhysBio
keys by PubMed ID. A bridge file ``pide_publications_pmid.csv`` maps publication
names to a PMID and DOI where RadPhysBio cites the same paper. It is built from
PIDE's reference list, so it is **not** shipped with pyRadBio (PIDE is a
registration-gated database): a registered user keeps it beside their PIDE export
(``PYRADBIO_PIDE_DIR``), where :func:`load_pide_publications` reads it, or passes
an explicit path. Without a bridge the cross-check has no PIDE-to-PMID keys.

How the bridge was built: PIDE's reference list was parsed into first-author
surname, year, volume and first page; each entry was matched to the PubMed
records of the PMIDs cited by RadPhysBio on surname + year and then confirmed
on volume + first page (``confidence = exact``), or on volume or page alone
(``probable``; one case, ``raju78``, where PIDE cites Part I and RadPhysBio
the companion Part II paper of the same study). Publications with no
RadPhysBio PMID of that surname + year are ``none``, with empty ``pmid``/``doi``:
they are simply not in RadPhysBio (most are rodent-only studies, which
RadPhysBio excludes by design). DOIs come from the PubMed records only; none
was invented. ``listed_in`` records the public source of each row.

Matching rule
-------------
Two rows are the same experiment when they share the publication key, the
cell line (compared through :func:`cell_key`), the projectile ``Z``, and a
LET within ``let_tol`` (relative, default 3%) -- or, when either side has no
LET, an energy within ``energy_tol``. Two rows that *both* carry a LET and
disagree beyond tolerance are never paired on energy. Within a
(publication, cell, Z) group the assignment is one-to-one and greedy on the
relative LET distance, with the relative ``alpha_ion`` distance as
tie-breaker so that two exposures listed at the *same* LET (repeat
experiments, common in Furusawa 2000) are paired with their own values rather
than crosswise.

Comparison categories (:func:`compare_values`)
----------------------------------------------
``identical``
    equal once both values are rounded (half-even) to the coarser of the two
    printed precisions -- ``0.0305`` vs ``0.03`` and ``0.53697`` vs ``0.537``
    both qualify.
``rounding``
    differ by at most one unit in the last printed digit of the coarser value
    (``0.074`` vs ``0.075``): a transcription of the same printed number, or of
    a value the other side rounded differently.
``sign``
    opposite signs and not within the rounding unit.
``digitization``
    relative difference ``<= digitization_tol`` (default 10%): consistent with
    reading a curve and refitting.
``discrepant``
    relative difference above that.
``missing``
    either side has no value.

The printed precision of a value is the number of decimals in its shortest
round-trip representation (``0.0615`` -> 4, ``1.6998479352116`` -> 13, capped
at 12); an exact ``0`` carries no precision and defers to the other side.
"""

from __future__ import annotations

import csv
import math
from collections.abc import Iterable, Mapping, Sequence
from dataclasses import dataclass, field
from decimal import ROUND_HALF_EVEN, Decimal
from pathlib import Path

from pyradbio.benchmark.cell_identity import CELL_LINE_ALIASES, cell_key
from pyradbio.benchmark.dataset import ReferenceRecord
from pyradbio.benchmark.pide import PideObservation
from pyradbio.benchmark.radphysbio import RadPhysBioEntry

__all__ = [
    "CATEGORIES",
    "CELL_LINE_ALIASES",
    "QUANTITIES",
    "Comparison",
    "CrossCheckResult",
    "CrossRow",
    "MatchedPair",
    "PidePublication",
    "cell_key",
    "compare_values",
    "crosscheck",
    "decimal_places",
    "load_pide_publications",
    "local_bridge_path",
    "match_rows",
    "pmid_by_publication",
    "rows_from_pide",
    "rows_from_radphysbio",
]

#: LQ quantities the driver compares, in report order.
QUANTITIES: tuple[str, ...] = ("alpha_ion", "beta_ion", "alpha_x", "beta_x")

#: Comparison categories, in report order (see the module docstring).
CATEGORIES: tuple[str, ...] = (
    "identical",
    "rounding",
    "digitization",
    "discrepant",
    "sign",
    "missing",
)

# --- bibliography bridge -----------------------------------------------------


@dataclass(frozen=True, slots=True)
class PidePublication:
    """One row of a local ``pide_publications_pmid.csv`` bridge.

    ``pmid``/``doi`` are ``""`` when no RadPhysBio PMID could be matched
    (``confidence == "none"``). ``match_method`` says which bibliographic
    fields agreed. ``listed_in`` names the public source of the row
    (e.g. ``"friedrich2013"``). ``pide_publication_no`` is PIDE's own running
    number, when the bridge carries it.
    """

    publication_name: str
    pmid: str
    doi: str
    year: int | None
    first_author: str
    match_method: str
    confidence: str
    listed_in: str = ""
    pide_publication_no: int | None = None


def _read_bridge(src: Path) -> list[PidePublication]:
    out: list[PidePublication] = []
    with src.open(newline="", encoding="utf-8") as fh:
        for row in csv.DictReader(fh):
            year_txt = (row.get("year") or "").strip()
            no_txt = (row.get("pide_publication_no") or "").strip()
            out.append(
                PidePublication(
                    publication_name=row["publication_name"].strip(),
                    pmid=(row.get("pmid") or "").strip(),
                    doi=(row.get("doi") or "").strip(),
                    year=int(year_txt) if year_txt.isdigit() else None,
                    first_author=(row.get("first_author") or "").strip(),
                    match_method=(row.get("match_method") or "").strip(),
                    confidence=(row.get("confidence") or "").strip(),
                    listed_in=(row.get("listed_in") or "").strip(),
                    pide_publication_no=int(no_txt) if no_txt.isdigit() else None,
                )
            )
    return out


def local_bridge_path() -> Path | None:
    """``<PYRADBIO_PIDE_DIR>/pide_publications_pmid.csv`` if configured and present."""
    from pyradbio.datasets import PIDE, dataset_dir

    folder = dataset_dir(PIDE, required=False)
    if folder is None:
        return None
    candidate = folder / "pide_publications_pmid.csv"
    return candidate if candidate.is_file() else None


def load_pide_publications(path: str | Path | None = None) -> tuple[PidePublication, ...]:
    """The bridge rows: an explicit ``path``, else the local bridge, else none.

    The bridge is never bundled (see the module docstring); without an explicit
    ``path`` this reads :func:`local_bridge_path` and returns ``()`` when no local
    bridge is configured.
    """
    src = Path(path) if path is not None else local_bridge_path()
    return () if src is None else tuple(_read_bridge(src))


def pmid_by_publication(
    publications: Iterable[PidePublication],
    *,
    min_confidence: str = "probable",
) -> dict[str, str]:
    """``{publication key: pmid}`` for the bridged publications.

    Keyed by the PIDE publication name (what
    :func:`~pyradbio.benchmark.pide.load_pide` writes into ``source``) and,
    for rows of a local complete bridge that carry it, by PIDE's running
    number as a string too. ``min_confidence`` is ``"exact"`` or
    ``"probable"`` (the default, which includes the exact ones); rows without
    a PMID are never included.
    """
    accepted = {"exact"} if min_confidence == "exact" else {"exact", "probable"}
    out: dict[str, str] = {}
    for p in publications:
        if p.pmid and p.confidence in accepted:
            out[p.publication_name] = p.pmid
            if p.pide_publication_no is not None:
                out[str(p.pide_publication_no)] = p.pmid
    return out


# --- rows -----------------------------------------------------------------------


@dataclass(frozen=True, slots=True)
class CrossRow:
    """The minimum a matcher needs: one ion exposure with optional LQ values.

    ``publication`` is whatever key both sides share (a PMID for PIDE vs
    RadPhysBio). ``let_keV_um``/``energy_MeV_u`` may be ``None``; the four LQ
    values may be ``None`` where the source has no value (PIDE ``N/A``, a
    RadPhysBio ion row without a photon partner). ``label`` is free-text
    provenance for reports (``"pide:<PublicationName>"``, ``"radphysbio:carbon#12"``).
    """

    publication: str
    cell_line: str
    Z: int
    let_keV_um: float | None
    energy_MeV_u: float | None
    alpha_ion: float | None = None
    beta_ion: float | None = None
    alpha_x: float | None = None
    beta_x: float | None = None
    label: str = ""

    def value(self, quantity: str) -> float | None:
        """The LQ quantity by name (one of :data:`QUANTITIES`)."""
        if quantity not in QUANTITIES:
            raise KeyError(quantity)
        result: float | None = getattr(self, quantity)
        return result


def rows_from_pide(
    entries: Iterable[ReferenceRecord | PideObservation],
    pmid_of: Mapping[str, str],
) -> list[CrossRow]:
    """PIDE records -> :class:`CrossRow` keyed by PMID via the bridge.

    Prefer :func:`~pyradbio.benchmark.pide.load_pide_observations`, which retains
    missing quantities as ``None``. Complete benchmark records are accepted
    for compatibility, but :func:`~pyradbio.benchmark.pide.load_pide` has already
    discarded partial experiments and cannot provide missing-value statistics.
    ``source`` must be ``"pide:<PublicationName or #Publication>"``;
    ``pmid_of`` is :func:`pmid_by_publication`, which carries both keys.
    Records whose publication has no PMID are dropped (they cannot be in
    RadPhysBio).
    """
    out: list[CrossRow] = []
    for e in entries:
        name = e.source.split(":", 1)[1] if ":" in e.source else e.source
        pmid = pmid_of.get(name)
        if not pmid:
            continue
        out.append(
            CrossRow(
                publication=pmid,
                cell_line=e.cell_line,
                Z=e.Z,
                let_keV_um=e.let_keV_um,
                energy_MeV_u=e.energy_MeV_u,
                alpha_ion=e.alpha_ion,
                beta_ion=e.beta_ion,
                alpha_x=e.alpha_x,
                beta_x=e.beta_x,
                label=e.source,
            )
        )
    return out


def rows_from_radphysbio(entries: Iterable[RadPhysBioEntry]) -> list[CrossRow]:
    """RadPhysBio ion rows -> :class:`CrossRow` keyed by PMID.

    Each ion row is paired with the X-ray/gamma row of the same PMID and cell
    line (first by ``#Exp``), the rule of
    :func:`~pyradbio.benchmark.radphysbio.pair_photon_references`; unlike that
    function, ion rows *without* a photon partner are kept with
    ``alpha_x``/``beta_x`` ``None`` so their ion parameters are still compared.
    The export's nominal beam energy is not a target energy, so it is omitted
    from the matching axis. Only LET can match these observations.
    """
    seq = list(entries)
    photons: dict[tuple[str, str], list[RadPhysBioEntry]] = {}
    for e in seq:
        if e.is_photon:
            photons.setdefault((e.pmid, cell_key(e.cell_line)), []).append(e)
    for group in photons.values():
        group.sort(key=lambda p: p.exp_no)
    out: list[CrossRow] = []
    for e in seq:
        if e.is_photon or e.Z is None:
            continue
        partner = photons.get((e.pmid, cell_key(e.cell_line)))
        photon = partner[0] if partner else None
        out.append(
            CrossRow(
                publication=e.pmid,
                cell_line=e.cell_line,
                Z=e.Z,
                let_keV_um=e.let_keV_um,
                energy_MeV_u=None,
                alpha_ion=e.alpha,
                beta_ion=e.beta,
                alpha_x=None if photon is None else photon.alpha,
                beta_x=None if photon is None else photon.beta,
                label=f"radphysbio:{e.radiation_type}#{e.exp_id}",
            )
        )
    return out


# --- value comparison -----------------------------------------------------------

_MAX_DECIMALS = 12


def decimal_places(x: float) -> int | None:
    """Decimals in the shortest round-trip representation of ``x`` (capped at 12).

    ``None`` for an exact zero or a non-finite value: those carry no printed
    precision. ``0.0615 -> 4``, ``1e-05 -> 5``; a float never reports fewer
    than one decimal (``2.0 -> 1``), the conservative reading of an integer-
    looking value whose printed precision is unknown.
    """
    if not math.isfinite(x) or x == 0.0:
        return None
    exp = Decimal(repr(float(x))).as_tuple().exponent
    if not isinstance(exp, int):  # pragma: no cover - NaN/Inf handled above
        return None
    return min(max(0, -exp), _MAX_DECIMALS)


def _quantize(x: float, decimals: int) -> Decimal:
    return Decimal(repr(float(x))).quantize(Decimal(1).scaleb(-decimals), rounding=ROUND_HALF_EVEN)


@dataclass(frozen=True, slots=True)
class Comparison:
    """Outcome of :func:`compare_values` for one quantity."""

    category: str
    #: ``|a - b|``; ``None`` when a side is missing.
    abs_diff: float | None
    #: ``|a - b| / max(|a|, |b|)``; ``None`` when missing or both zero.
    rel_diff: float | None
    #: Decimals the comparison was made at (the coarser side); ``None`` when
    #: neither side carries a precision (both zero) or a side is missing.
    decimals: int | None


def compare_values(
    a: float | None,
    b: float | None,
    *,
    digitization_tol: float = 0.10,
) -> Comparison:
    """Classify the agreement of two values (categories in the module docstring)."""
    if a is None or b is None or not (math.isfinite(a) and math.isfinite(b)):
        return Comparison("missing", None, None, None)
    diff = abs(a - b)
    scale = max(abs(a), abs(b))
    rel = diff / scale if scale > 0.0 else None
    known = [d for d in (decimal_places(a), decimal_places(b)) if d is not None]
    decimals = min(known) if known else None
    if decimals is None:  # both exactly zero
        return Comparison("identical", 0.0, None, None)
    if _quantize(a, decimals) == _quantize(b, decimals):
        return Comparison("identical", diff, rel, decimals)
    unit = 10.0**-decimals
    if diff <= unit * (1.0 + 1e-9):
        return Comparison("rounding", diff, rel, decimals)
    if a * b < 0.0:
        return Comparison("sign", diff, rel, decimals)
    category = "digitization" if rel is not None and rel <= digitization_tol else "discrepant"
    return Comparison(category, diff, rel, decimals)


# --- experiment matching ----------------------------------------------------------


@dataclass(frozen=True, slots=True)
class MatchedPair:
    """One shared experiment and the per-quantity comparison."""

    left: CrossRow
    right: CrossRow
    #: ``"let"`` or ``"energy"``: which axis the pairing was made on.
    match_basis: str
    #: Relative LET difference ``|L_l - L_r| / L_l`` (``None`` if a side lacks LET).
    let_rel_diff: float | None
    comparisons: Mapping[str, Comparison] = field(default_factory=dict)

    def category(self, quantity: str) -> str:
        return self.comparisons[quantity].category


def _rel_close(x: float | None, y: float | None, tol: float) -> float | None:
    """Relative distance ``|x - y| / |x|`` if both present and within ``tol``, else ``None``."""
    if x is None or y is None:
        return None
    scale = abs(x) if x != 0.0 else abs(y)
    if scale == 0.0:
        return 0.0 if y == 0.0 else None
    d = abs(x - y) / scale
    return d if d <= tol else None


def _alpha_gap(a: CrossRow, b: CrossRow) -> float:
    if a.alpha_ion is None or b.alpha_ion is None:
        return 0.0
    scale = max(abs(a.alpha_ion), abs(b.alpha_ion))
    return abs(a.alpha_ion - b.alpha_ion) / scale if scale > 0.0 else 0.0


RawPair = tuple[CrossRow, CrossRow, str, float | None]


def _assign(
    left: Sequence[CrossRow],
    right: Sequence[CrossRow],
    *,
    let_tol: float,
    energy_tol: float,
) -> list[RawPair]:
    """One-to-one greedy assignment inside one (publication, cell, Z) group."""
    candidates: list[tuple[tuple[float, float, int], int, int, str, float | None]] = []
    for i, lrow in enumerate(left):
        for j, rrow in enumerate(right):
            d_let = _rel_close(lrow.let_keV_um, rrow.let_keV_um, let_tol)
            if d_let is not None:
                candidates.append(
                    ((round(d_let, 3), _alpha_gap(lrow, rrow), 0), i, j, "let", d_let)
                )
                continue
            if lrow.let_keV_um is not None and rrow.let_keV_um is not None:
                # Both have a LET and they disagree beyond tolerance: not the
                # same exposure, whatever the energies say.
                continue
            d_e = _rel_close(lrow.energy_MeV_u, rrow.energy_MeV_u, energy_tol)
            if d_e is not None:
                candidates.append(
                    ((round(d_e, 3), _alpha_gap(lrow, rrow), 1), i, j, "energy", None)
                )
    candidates.sort(key=lambda c: c[0])
    used_l: set[int] = set()
    used_r: set[int] = set()
    out: list[RawPair] = []
    for _, i, j, basis, let_rel in candidates:
        if i in used_l or j in used_r:
            continue
        used_l.add(i)
        used_r.add(j)
        out.append((left[i], right[j], basis, let_rel))
    return out


def match_rows(
    left: Sequence[CrossRow],
    right: Sequence[CrossRow],
    *,
    let_tol: float = 0.03,
    energy_tol: float = 0.03,
) -> tuple[list[RawPair], list[CrossRow], list[CrossRow]]:
    """Pair shared experiments (see "Matching rule" in the module docstring).

    Returns ``(pairs, unmatched_left, unmatched_right)`` where each pair is
    ``(left_row, right_row, match_basis, let_rel_diff)``.
    """
    groups_l: dict[tuple[str, str, int], list[CrossRow]] = {}
    groups_r: dict[tuple[str, str, int], list[CrossRow]] = {}
    for r in left:
        groups_l.setdefault((r.publication, cell_key(r.cell_line), r.Z), []).append(r)
    for r in right:
        groups_r.setdefault((r.publication, cell_key(r.cell_line), r.Z), []).append(r)
    pairs: list[RawPair] = []
    matched_l: set[int] = set()
    matched_r: set[int] = set()
    for key, lrows in groups_l.items():
        rrows = groups_r.get(key)
        if not rrows:
            continue
        for lrow, rrow, basis, let_rel in _assign(
            lrows, rrows, let_tol=let_tol, energy_tol=energy_tol
        ):
            pairs.append((lrow, rrow, basis, let_rel))
            matched_l.add(id(lrow))
            matched_r.add(id(rrow))
    unmatched_l = [r for r in left if id(r) not in matched_l]
    unmatched_r = [r for r in right if id(r) not in matched_r]
    return pairs, unmatched_l, unmatched_r


# --- driver --------------------------------------------------------------------------


@dataclass(frozen=True, slots=True)
class CrossCheckResult:
    """Everything :func:`crosscheck` found."""

    pairs: tuple[MatchedPair, ...]
    unmatched_left: tuple[CrossRow, ...]
    unmatched_right: tuple[CrossRow, ...]
    #: ``summary[quantity][category]`` -> count over all pairs.
    summary: Mapping[str, Mapping[str, int]]

    @property
    def n_pairs(self) -> int:
        return len(self.pairs)

    def worst(self, quantity: str, n: int = 15) -> list[MatchedPair]:
        """The ``n`` pairs with the largest relative difference in ``quantity``."""
        scored = [p for p in self.pairs if p.comparisons[quantity].rel_diff is not None]
        scored.sort(key=lambda p: -(p.comparisons[quantity].rel_diff or 0.0))
        return scored[:n]

    def format_summary(self) -> str:
        """A fixed-width text table of the category histogram."""
        head = f"{'quantity':10}" + "".join(f"{c:>13}" for c in CATEGORIES) + f"{'total':>8}"
        lines = [head]
        for q, counts in self.summary.items():
            lines.append(
                f"{q:10}"
                + "".join(f"{counts.get(c, 0):>13}" for c in CATEGORIES)
                + f"{self.n_pairs:>8}"
            )
        return "\n".join(lines)


def crosscheck(
    left: Sequence[CrossRow],
    right: Sequence[CrossRow],
    *,
    let_tol: float = 0.03,
    energy_tol: float = 0.03,
    digitization_tol: float = 0.10,
    quantities: Sequence[str] = QUANTITIES,
) -> CrossCheckResult:
    """Match ``left`` against ``right`` and classify every LQ quantity of every pair.

    ``left`` is the reference side (its LET is the denominator of the relative
    LET difference). ``quantities`` defaults to all four of :data:`QUANTITIES`.
    """
    raw_pairs, un_l, un_r = match_rows(left, right, let_tol=let_tol, energy_tol=energy_tol)
    pairs: list[MatchedPair] = []
    summary: dict[str, dict[str, int]] = {q: dict.fromkeys(CATEGORIES, 0) for q in quantities}
    for lrow, rrow, basis, let_rel in raw_pairs:
        comps = {
            q: compare_values(lrow.value(q), rrow.value(q), digitization_tol=digitization_tol)
            for q in quantities
        }
        for q, c in comps.items():
            summary[q][c.category] += 1
        pairs.append(MatchedPair(lrow, rrow, basis, let_rel, comps))
    return CrossCheckResult(
        pairs=tuple(pairs),
        unmatched_left=tuple(un_l),
        unmatched_right=tuple(un_r),
        summary=summary,
    )
