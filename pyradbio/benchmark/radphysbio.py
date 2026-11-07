"""Loader for the RadPhysBio database (NTUA; Zanni et al. 2024).

RadPhysBio -- Zanni V, Papakonstantinou D, et al., "RadPhysBio: A Radiobiological
Database for the Prediction of Cell Survival upon Exposure to Ionizing Radiation",
*Int. J. Mol. Sci.* 25(9):4729, 2024, doi:10.3390/ijms25094729 (CC BY 4.0;
metadata record doi:10.5281/zenodo.12582567) -- is a literature-mined table of
LQ survival parameters for **human** cell lines under X-rays, gamma-rays,
protons, carbon ions and alpha-particles, browsable and exportable at
https://radbiodb.physics.ntua.gr/radphysbio/. It is open access, so unlike PIDE
it *may* be redistributed; this package still reads it from a user-configured
folder (``PYRADBIO_RADPHYSBIO_DIR``, see :mod:`pyradbio.datasets`) to keep the
package small, and bundles only a short attributed excerpt as a test fixture.

What the export is, and is not
------------------------------
The CSV export of the whole database is **five tables concatenated** -- one per
radiation type, each with its own ``#ExpID`` numbering restarting at 1 -- so
``(TypeofRadiation, #ExpID)`` is the row key, not ``#ExpID`` alone. Each row is
one survival experiment: a cell line, a radiation, and the LQ ``alpha``/``beta``
of that exposure. **Photon references are separate rows**, not paired per ion
row as in PIDE; :func:`pair_photon_references` builds the pairing this package
needs by matching an ion row to the X-ray/gamma row of the *same publication
(PMID) and cell line*. Only about a third of the ion rows have such a partner.

Semantics established from the paper (Methods, section 4) and the site's own
column descriptions:

* ``alpha``/``beta`` are "as given in the corresponding publication, or else
  from fitting to raw data" -- the authors digitized survival curves with
  WebPlotDigitizer and fitted the LQ form in Python, unweighted, from at least
  three points, "without considering extreme values". The export does not flag
  which rows are which. A trailing ``f`` marks a *fitted* value only in the DSB
  column.
* ``Energy (MeV)`` is the **total nominal beam** kinetic energy (a 290 MeV/u
  carbon beam is listed as 3480); :attr:`RadPhysBioEntry.energy_MeV_u` divides
  by the mass number. For photons it is the tube voltage / photon energy in MeV
  (``0.2`` for 200 kVp, ``1.332`` for 60Co).
* ``LET`` is in water, keV/um. **For ions it is partly derived, not measured**:
  the paper states that LET "was calculated from well-known RBE/LET data
  [Jones 2015] using the WebPlotDigitizer software with a maximum estimated
  uncertainty of 20%", and missing LETs were filled from RBE-LET curves, MCDS
  results, or "similar alpha values". Photon LETs are assumed constants (2 for
  ~250 kVp, 0.3 for 6-15 MV, 2.5 below 100 kVp, 1 for gamma-rays).
* ``RBE`` is "Relative Biological Effectiveness" with **no stated survival level
  or reference**; it is kept verbatim and not used by the harness.
* ``IrradiationConditions``: ``m`` monoenergetic, ``s`` spread-out Bragg peak.
  ``CellClass``: ``t`` tumour, ``n`` normal. ``CellCycle``: ``a`` asynchronous
  or the phase.
* The DSB / non-DSB-cluster columns are MCDS (Semenenko & Stewart 2006) yields
  per Gy per Gbp, at the experiment's energy or at 1 keV / 10 keV, for 1% and
  20% O2; ``DSBs/(Gy*Gbp)`` is the value from the publication when given.

Known export defects handled here: two rows whose cell-line name contains a
comma (``SP6,5``, the uveal melanoma line SP6.5) are split across 31 fields and
are re-joined; ``N/A`` appears in Latin and Greek spellings (Greek nu / alpha); the
``Source`` column has stray spaces (``2 3 8Pu``). Nothing is corrected beyond
re-joining; values are kept as exported.
"""

from __future__ import annotations

import re
from collections.abc import Iterable, Mapping, Sequence
from dataclasses import dataclass
from pathlib import Path

from pyradbio.benchmark.cell_identity import cell_key
from pyradbio.benchmark.dataset import (
    ReferenceDataset,
    ReferenceRecord,
    clean_str,
    is_missing,
    normalise_header,
    read_table,
    to_float,
)

__all__ = [
    "ION_SPECIES",
    "KNOWN_DEFECTS",
    "PHOTON_TYPES",
    "RADPHYSBIO_CAVEATS",
    "RADPHYSBIO_CITATION",
    "KnownDefect",
    "MCDSYields",
    "PairingReport",
    "RadPhysBioEntry",
    "duplicate_copies",
    "find_radphysbio_file",
    "load_radphysbio",
    "load_radphysbio_dataset",
    "pair_photon_references",
    "parse_radphysbio_rows",
    "photon_quality_label",
]

RADPHYSBIO_CITATION = (
    "Zanni V, Papakonstantinou D, et al. RadPhysBio: A Radiobiological Database for the "
    "Prediction of Cell Survival upon Exposure to Ionizing Radiation. Int J Mol Sci "
    "2024;25(9):4729 (doi:10.3390/ijms25094729, CC BY 4.0). "
    "https://radbiodb.physics.ntua.gr/radphysbio/"
)

#: What a reader must know before believing a score against RadPhysBio.
RADPHYSBIO_CAVEATS: tuple[str, ...] = (
    "**Largely digitized.** Where a publication printed no LQ parameters, RadPhysBio's "
    "alpha/beta are the authors' own unweighted LQ fits to survival points read off "
    "the figure with WebPlotDigitizer (>= 3 points, extremes dropped). The export "
    "does not say which rows are printed and which are refit.",
    "**Ion LET is partly derived, not measured.** The paper states ion LETs were "
    "read off a generic RBE-LET curve (Jones 2015) with up to 20% uncertainty, and "
    "missing LETs were filled from RBE-LET curves, MCDS output or 'similar alpha "
    "values'. A LET-based model scored on this axis inherits that.",
    "**The ion/photon pairing is this package's, not the database's.** RadPhysBio "
    "lists photon experiments as separate rows; an ion row is paired here with the "
    "X-ray/gamma row of the same PMID and cell line. Roughly two thirds of the ion "
    "rows have no such partner and are not scored.",
    "Energies are total kinetic energy (MeV) in the export and are divided by the "
    "mass number here; the RBE column has no stated survival level and is not used.",
    "Human cell lines only: none of the V79 / CHO / xrs5 rodent series that anchor "
    "the LEM papers are in this database.",
    "**One experiment can appear under several PMIDs.** Verbatim re-listings of a "
    "published table under a later PMID are collapsed to the earliest eligible "
    "copy of the ion row by default (`collapse_duplicates`; the tally says how many, "
    "and how many re-listings went unscored for want of a usable photon pair). Only "
    "the ion row is compared; the kept copy is scored against its own publication's "
    "photon reference. Re-digitizations of the same "
    "figure by a later paper differ in the last digits, are not detectable, and are "
    "still counted as independent points: Furusawa 2000's HSG carbon series is such a "
    "case, listed under five publications.",
    "**The energy column is the nominal beam energy**, not a verified energy at the "
    "target. Records are marked energy_is_nominal; energy-based model runs are "
    "rejected, and LET-based runs neither gate nor label points by it, so every "
    "record with a LET is scored. Cross-checks use LET only for these rows.",
    "Potential transcription and metadata defects are listed in `KNOWN_DEFECTS` and "
    "left as exported; check the original publications before quantitative use.",
)


@dataclass(frozen=True, slots=True)
class KnownDefect:
    """A qualitative transcription or metadata concern for primary-source review.

    Values are **not** corrected by the loader (it returns the export as is);
    this list exists so a user can identify rows needing verification. Restricted
    PIDE values and experiment-level cross-check results are not included here.
    """

    pmid: str
    radiation_type: str
    what: str


#: Qualitative review notes only; no PIDE-derived replacement parameters.
KNOWN_DEFECTS: tuple[KnownDefect, ...] = (
    KnownDefect(
        "9687977",
        "carbon",
        "Tsuboi et al. 1998: verify the scale factors in the original LQ parameter "
        "table before using these rows; the export is left unchanged.",
    ),
    KnownDefect(
        "10924995",
        "X-rays",
        "Suzuki et al. 2000: verify the KNS-60 X-ray beta against the original table "
        "before using it as a photon reference.",
    ),
    KnownDefect(
        "10924995",
        "carbon",
        "Suzuki et al. 2000: verify the HFL-III carbon beta transcription against "
        "the original table before use.",
    ),
    KnownDefect(
        "11025645",
        "carbon",
        "Furusawa et al. 2000: HSG/T1 carbon rows reuse a nominal beam energy "
        "across a LET series. Do not interpret it as target energy. Verify the "
        "irradiation-condition flags and beta values against the original paper.",
    ),
    KnownDefect(
        "18182686, 20726730, 29726401, 27380803",
        "carbon",
        "Re-listings of Furusawa 2000's HSG carbon series under the PMIDs of papers "
        "that re-plotted it (the 29726401 copy is a re-digitization of Inaniwa 2018's "
        "figure); not independent experiments.",
    ),
)

#: ``TypeofRadiation`` value (ASCII-normalised) -> ``(Z, A)`` for the ion tables.
#: RadPhysBio carries no isotope column; carbon is 12C and the alpha-particle
#: table is 4He from radionuclide sources, so these are the only sensible reads.
ION_SPECIES: Mapping[str, tuple[int, int]] = {
    "protons": (1, 1),
    "carbon": (6, 12),
    "alpha-particles": (2, 4),
}

#: The two photon tables.
PHOTON_TYPES: frozenset[str] = frozenset({"X-rays", "gamma-rays"})

_RADIATION_NORMALISE: Mapping[str, str] = {
    "x-rays": "X-rays",
    "xrays": "X-rays",
    "\u03b3-rays": "gamma-rays",  # Greek gamma, as exported
    "gamma-rays": "gamma-rays",
    "gammarays": "gamma-rays",
    "protons": "protons",
    "carbon": "carbon",
    "\u03b1-particles": "alpha-particles",  # Greek alpha, as exported
    "alpha-particles": "alpha-particles",
    "alphaparticles": "alpha-particles",
}


def normalise_radiation_type(raw: str) -> str:
    """ASCII form of the export's ``TypeofRadiation`` (Greek ``gamma``-rays -> ``gamma-rays``)."""
    key = raw.strip().lower()
    return _RADIATION_NORMALISE.get(
        key, _RADIATION_NORMALISE.get(key.replace(" ", ""), raw.strip())
    )


@dataclass(frozen=True, slots=True)
class MCDSYields:
    """MCDS damage yields per Gy per Gbp at 1% and 20% oxygen (``None`` = N/A)."""

    dsb_1pct_O2: float | None
    non_dsb_1pct_O2: float | None
    dsb_20pct_O2: float | None
    non_dsb_20pct_O2: float | None


@dataclass(frozen=True, slots=True)
class RadPhysBioEntry:
    """One RadPhysBio row: a single survival experiment (ion *or* photon).

    Units follow the export: ``energy_MeV`` is total nominal beam energy (photons:
    tube voltage / photon energy in MeV), ``let_keV_um`` in water, ``alpha``
    Gy^-1, ``beta`` Gy^-2, ``dose_rate_Gy_min`` Gy/min. ``radiation_type`` is
    ASCII-normalised (``"X-rays"``, ``"gamma-rays"``, ``"protons"``, ``"carbon"``,
    ``"alpha-particles"``). ``exp_id`` restarts per radiation table, so the row
    key is ``(radiation_type, exp_id)``.
    """

    exp_id: int
    pmid: str
    exp_no: int
    cell_line: str
    tissue: str
    cell_class: str
    cell_cycle: str
    radiation_type: str
    energy_MeV: float | None
    rbe: float | None
    let_keV_um: float | None
    dose_rate_Gy_min: float | None
    alpha: float | None
    beta: float | None
    #: DSB yield from the publication (per Gy per Gbp); ``dsb_from_fit`` marks the
    #: export's trailing ``f`` (a value the authors fitted rather than read).
    dsb_per_Gy_Gbp: float | None
    dsb_from_fit: bool
    non_dsb_clusters_per_Gy_Gbp: float | None
    #: MCDS yields at the experiment's own energy, and at the fixed 1 / 10 keV
    #: electron energies the authors ran as stand-ins.
    mcds_at_energy: MCDSYields
    mcds_1keV: MCDSYields
    mcds_10keV: MCDSYields
    irradiation_conditions: str
    #: Radionuclide / source string as exported (``"60Co"``, ``"241Am"``, ``""``).
    radiation_source: str

    @property
    def is_photon(self) -> bool:
        return self.radiation_type in PHOTON_TYPES

    @property
    def Z(self) -> int | None:
        """Atomic number for an ion row, ``None`` for a photon row."""
        za = ION_SPECIES.get(self.radiation_type)
        return None if za is None else za[0]

    @property
    def A(self) -> int | None:
        """Mass number for an ion row, ``None`` for a photon row."""
        za = ION_SPECIES.get(self.radiation_type)
        return None if za is None else za[1]

    @property
    def energy_MeV_u(self) -> float | None:
        """Nominal specific beam energy ``energy_MeV / A`` for ions, else ``None``."""
        a = self.A
        if a is None or self.energy_MeV is None:
            return None
        return self.energy_MeV / a

    @property
    def key(self) -> tuple[str, int]:
        return (self.radiation_type, self.exp_id)


# --- column mapping --------------------------------------------------------

#: Logical field -> normalised header of the export (see :func:`normalise_header`).
_COLUMNS: Mapping[str, str] = {
    "exp_id": "expid",
    "pmid": "pmid",
    "exp_no": "exp",
    "cell_line": "cellline",
    "tissue": "tissue",
    "cell_class": "cellclass",
    "cell_cycle": "cellcycle",
    "energy": "energymev",
    "rbe": "rbe",
    "let": "letkevm",  # "LET (keV/<mu>m)": the Greek mu is not [a-z0-9] and drops out
    "dose_rate": "doserategymin",
    "alpha": "",  # Greek alpha: resolved by position fallback below
    "beta": "",
    "dsb": "dsbsgygbp",
    "non_dsb": "nondsbclustersgygbp",
    "mcds_1keV_dsb_1": "1kevdsbs1o2",
    "mcds_1keV_other_1": "1kevother1o2",
    "mcds_1keV_dsb_20": "1kevdsbs20o2",
    "mcds_1keV_other_20": "1kevother20o2",
    "mcds_10keV_dsb_1": "10kevdsbs1o2",
    "mcds_10keV_other_1": "10kevother1o2",
    "mcds_10keV_dsb_20": "10kevdsbs20o2",
    "mcds_10keV_other_20": "10kevother20o2",
    "radiation_type": "typeofradiation",
    "conditions": "irradiationconditions",
    "mcds_E_dsb_1": "dsbs1o2",
    "mcds_E_other_1": "other1o2",
    "mcds_E_dsb_20": "dsbs20o2",
    "mcds_E_other_20": "other20o2",
    "source": "source",
}

#: The export's column order (30 fields), used to resolve headers the
#: normaliser cannot (the Greek alpha/beta headers and ``DoseRate``), and to repair rows.
_EXPECTED_HEADERS: tuple[str, ...] = (
    "#ExpID",
    "PMID",
    "#Exp",
    "CellLine",
    "Tissue",
    "CellClass",
    "CellCycle",
    "Energy (MeV)",
    "RBE",
    "LET (keV/\u03bcm)",
    "DoseRate (Gy/min)",
    "\u03b1",
    "\u03b2",
    "DSBs/(Gy*Gbp)",
    "nonDSBClusters/(Gy*Gbp)",
    "1keV_DSBs_1%O2",
    "1keV_Other_1%O2",
    "1keV_DSBs_20%O2",
    "1keV_Other_20%O2",
    "10keV_DSBs_1%O2",
    "10keV_Other_1%O2",
    "10keV_DSBs_20%O2",
    "10keV_Other_20%O2",
    "TypeofRadiation",
    "IrradiationConditions",
    "DSBs_1%O2",
    "Other_1%O2",
    "DSBs_20%O2",
    "Other_20%O2",
    "Source",
)
_N_FIELDS = len(_EXPECTED_HEADERS)

_FIELD_BY_POSITION: Mapping[str, int] = {
    "exp_id": 0,
    "pmid": 1,
    "exp_no": 2,
    "cell_line": 3,
    "tissue": 4,
    "cell_class": 5,
    "cell_cycle": 6,
    "energy": 7,
    "rbe": 8,
    "let": 9,
    "dose_rate": 10,
    "alpha": 11,
    "beta": 12,
    "dsb": 13,
    "non_dsb": 14,
    "mcds_1keV_dsb_1": 15,
    "mcds_1keV_other_1": 16,
    "mcds_1keV_dsb_20": 17,
    "mcds_1keV_other_20": 18,
    "mcds_10keV_dsb_1": 19,
    "mcds_10keV_other_1": 20,
    "mcds_10keV_dsb_20": 21,
    "mcds_10keV_other_20": 22,
    "radiation_type": 23,
    "conditions": 24,
    "mcds_E_dsb_1": 25,
    "mcds_E_other_1": 26,
    "mcds_E_dsb_20": 27,
    "mcds_E_other_20": 28,
    "source": 29,
}

_GREEK_HEADERS: Mapping[str, str] = {"\u03b1": "alpha", "\u03b2": "beta"}


def _column_index(headers: Sequence[str]) -> dict[str, int]:
    """Map each logical field to its column, by header text first, position second."""
    norm: dict[str, int] = {}
    for idx, head in enumerate(headers):
        h = str(head).strip()
        if h in _GREEK_HEADERS:
            norm.setdefault(_GREEK_HEADERS[h], idx)
        norm.setdefault(normalise_header(h), idx)
    idx_of: dict[str, int] = {}
    for field_name, target in _COLUMNS.items():
        if field_name in ("alpha", "beta"):
            found = norm.get(field_name)
        else:
            found = norm.get(target)
        if found is None:
            # The LET header's mu drops out differently depending on the
            # export's encoding; fall back to the canonical position.
            found = _FIELD_BY_POSITION[field_name] if len(headers) == _N_FIELDS else None
        if found is not None:
            idx_of[field_name] = found
    return idx_of


def _repair_row(row: Sequence[object]) -> list[object] | None:
    """Re-join a row whose cell-line name was split on an embedded comma.

    The export has 30 fields; a row with 31 is one whose ``CellLine`` contained
    a comma (``SP6,5``). Anything else off-length is returned as ``None``.
    """
    if len(row) == _N_FIELDS:
        return list(row)
    if len(row) == _N_FIELDS + 1:
        return [*row[:3], f"{row[3]},{row[4]}", *row[5:]]
    return None


_FIT_FLAG_RE = re.compile(r"^\s*([-+0-9.eE]+)\s*f\s*$")


def _dsb_value(raw: object) -> tuple[float | None, bool]:
    """Split the export's fit flag.

    ``"5.39 f"`` -> ``(5.39, True)``; ``"5.39"`` -> ``(5.39, False)``; N/A -> ``(None, False)``.
    """
    if is_missing(raw):
        return (None, False)
    text = str(raw).strip()
    m = _FIT_FLAG_RE.match(text)
    if m:
        return (to_float(m.group(1)), True)
    return (to_float(text), False)


def _to_int(raw: object, default: int = 0) -> int:
    val = to_float(raw)
    return default if val is None else round(val)


def parse_radphysbio_rows(
    headers: Sequence[str],
    rows: Iterable[Sequence[object]],
    *,
    strict: bool = False,
) -> list[RadPhysBioEntry]:
    """Typed :class:`RadPhysBioEntry` records from the export's header + rows.

    Rows with an unexpected field count (other than the 31-field cell-line-comma
    case, which is repaired) are skipped, or raise ``ValueError`` if ``strict``.
    Nothing else is filtered: photon rows, rows without LET or ``alpha`` are all
    kept, because the photon rows are the references and the harness filters
    later with its own, stated rules.
    """
    idx = _column_index(headers)
    required = ("radiation_type", "cell_line", "pmid", "alpha")
    lacking = [f for f in required if f not in idx]
    if lacking:
        raise ValueError(
            f"RadPhysBio export lacks columns for {lacking}; got headers {list(headers)!r}"
        )

    def cell(row: Sequence[object], field_name: str) -> object:
        i = idx.get(field_name)
        return row[i] if i is not None and i < len(row) else None

    def num(row: Sequence[object], field_name: str) -> float | None:
        return to_float(cell(row, field_name))

    def yields(row: Sequence[object], prefix: str) -> MCDSYields:
        return MCDSYields(
            dsb_1pct_O2=num(row, f"{prefix}_dsb_1"),
            non_dsb_1pct_O2=num(row, f"{prefix}_other_1"),
            dsb_20pct_O2=num(row, f"{prefix}_dsb_20"),
            non_dsb_20pct_O2=num(row, f"{prefix}_other_20"),
        )

    out: list[RadPhysBioEntry] = []
    for row_no, raw_row in enumerate(rows):
        row = _repair_row(raw_row)
        if row is None:
            if strict:
                raise ValueError(
                    f"RadPhysBio row {row_no}: {len(raw_row)} fields, expected {_N_FIELDS}"
                )
            continue
        radiation = normalise_radiation_type(clean_str(cell(row, "radiation_type")))
        dsb, dsb_fit = _dsb_value(cell(row, "dsb"))
        out.append(
            RadPhysBioEntry(
                exp_id=_to_int(cell(row, "exp_id"), default=row_no),
                pmid=clean_str(cell(row, "pmid")),
                exp_no=_to_int(cell(row, "exp_no")),
                cell_line=clean_str(cell(row, "cell_line")),
                tissue=clean_str(cell(row, "tissue")),
                cell_class=clean_str(cell(row, "cell_class")),
                cell_cycle=clean_str(cell(row, "cell_cycle")),
                radiation_type=radiation,
                energy_MeV=num(row, "energy"),
                rbe=num(row, "rbe"),
                let_keV_um=num(row, "let"),
                dose_rate_Gy_min=num(row, "dose_rate"),
                alpha=num(row, "alpha"),
                beta=num(row, "beta"),
                dsb_per_Gy_Gbp=dsb,
                dsb_from_fit=dsb_fit,
                non_dsb_clusters_per_Gy_Gbp=num(row, "non_dsb"),
                mcds_at_energy=yields(row, "mcds_E"),
                mcds_1keV=yields(row, "mcds_1keV"),
                mcds_10keV=yields(row, "mcds_10keV"),
                irradiation_conditions=clean_str(cell(row, "conditions")),
                radiation_source=clean_str(cell(row, "source")),
            )
        )
    return out


# --- file discovery ---------------------------------------------------------

_SUFFIXES = (".csv", ".xlsx", ".xls", ".tsv", ".txt")


def find_radphysbio_file(path: str | Path | None = None) -> Path:
    """Resolve the RadPhysBio export: explicit file/dir, else ``PYRADBIO_RADPHYSBIO_DIR``."""
    if path is not None:
        p = Path(path)
        if p.is_file():
            return p
        if not p.is_dir():
            raise FileNotFoundError(f"RadPhysBio path does not exist: {p}")
        search_dir = p
    else:
        from pyradbio.datasets import RADPHYSBIO, dataset_dir

        search_dir = dataset_dir(RADPHYSBIO)  # raises MissingDatasetError with guidance
    candidates = [f for f in sorted(search_dir.iterdir()) if f.suffix.lower() in _SUFFIXES]
    candidates.sort(key=lambda f: _SUFFIXES.index(f.suffix.lower()))
    if not candidates:
        raise FileNotFoundError(
            f"No RadPhysBio export (.csv/.xlsx) found in {search_dir}. Export the database "
            "from https://radbiodb.physics.ntua.gr/radphysbio/ and place it there, or "
            "pass an explicit path."
        )
    return candidates[0]


def load_radphysbio(
    path: str | Path | None = None, *, strict: bool = False
) -> list[RadPhysBioEntry]:
    """Parse a RadPhysBio export (``.csv`` from the site, or ``.xlsx``) into entries."""
    headers, rows = read_table(find_radphysbio_file(path))
    return parse_radphysbio_rows(headers, rows, strict=strict)


# --- ion/photon pairing -----------------------------------------------------


def photon_quality_label(photon: RadPhysBioEntry) -> str:
    """A PIDE-style photon quality string for a photon row.

    ``"200kVp"`` for an X-ray row listed at 0.2 MeV, ``"6MV"`` at 6 MeV, the
    radionuclide (``"60Co"``) for a gamma row with a source, else the radiation
    type. Informational only; the harness does not interpret it.
    """
    if photon.radiation_type == "gamma-rays":
        return photon.radiation_source or "gamma-rays"
    e = photon.energy_MeV
    if e is None:
        return photon.radiation_type
    if e < 1.0:
        return f"{round(e * 1000)}kVp"
    return f"{e:g}MV"


@dataclass(frozen=True, slots=True)
class PairingReport:
    """Tally of how the ion rows were paired with photon references."""

    n_ion_rows: int
    n_paired: int
    #: Ion rows with no photon row of the same PMID + cell line.
    n_unpaired: int
    #: Ion rows whose PMID + cell line has *several* photon rows with different
    #: LQ values (paired with the first by ``#Exp``; flagged, not dropped).
    n_ambiguous: int
    #: Paired rows dropped for the harness: no LET, no alpha, or alpha_x <= 0.
    n_incomplete: int
    #: Eligible ion rows dropped as verbatim copies of an eligible ion row under
    #: an earlier PMID (see ``collapse_duplicates``). Only the ion row is
    #: compared; the retained copy keeps its own publication's photon partner.
    n_duplicate_copies: int = 0
    #: Ion rows that verbatim re-list a row under an earlier PMID but formed no
    #: eligible pair (unpaired, incomplete, or an excluded ambiguous pair).
    #: Informational: already counted in those tallies, not a partition term.
    n_relisted_unscored: int = 0


def _pmid_order(pmid: str) -> tuple[int, str]:
    """Sort key putting the numerically smallest (oldest) PMID first."""
    return (int(pmid) if pmid.isdigit() else 10**12, pmid)


def duplicate_copies(entries: Sequence[RadPhysBioEntry]) -> set[tuple[str, int]]:
    """Keys of ion rows that are verbatim copies of a row under an earlier PMID.

    Two ion rows are copies of one experiment when their normalised cell line,
    ``Z``, LET, ``alpha`` and ``beta`` are all **exactly** equal and they sit
    under different PMIDs -- the signature of a published table re-listed by a
    later paper. The copy under the numerically smallest PMID (the original
    publication, PMIDs being assigned in order) is kept. A re-digitization of
    the same figure by a later paper differs in the last digits and is *not*
    caught; the caveats say so.
    """
    groups: dict[tuple[object, ...], list[RadPhysBioEntry]] = {}
    for e in entries:
        if e.is_photon or e.Z is None:
            continue
        key = (cell_key(e.cell_line), e.Z, e.let_keV_um, e.alpha, e.beta)
        groups.setdefault(key, []).append(e)
    dropped: set[tuple[str, int]] = set()
    for group in groups.values():
        if len({e.pmid for e in group}) < 2:
            continue
        keep_pmid = min((e.pmid for e in group), key=_pmid_order)
        dropped.update(e.key for e in group if e.pmid != keep_pmid)
    return dropped


def pair_photon_references(
    entries: Sequence[RadPhysBioEntry],
    *,
    include_ambiguous: bool = True,
    collapse_duplicates: bool = True,
) -> tuple[list[ReferenceRecord], PairingReport]:
    """Build harness records by pairing each ion row with its publication's photon row.

    An ion row (protons / carbon / alpha-particles) is paired with the X-ray or
    gamma-ray row of the **same PMID and cell line** (cell-line names compared
    case- and punctuation-insensitively). When a publication lists several photon
    experiments for the line (different dose rates, energies, or repeats), the
    first by ``#Exp`` is used and the row is counted ``n_ambiguous``; pass
    ``include_ambiguous=False`` to drop those instead. Ion rows without LET or
    ``alpha``, or whose photon ``alpha`` is not positive, are counted
    ``n_incomplete`` and left out. A missing ``beta`` on either side becomes
    ``0.0`` (the harness treats ``beta_x <= 0`` as "no (alpha/beta)_x").

    With ``collapse_duplicates`` (default), only ion rows whose pairs pass the
    above completeness and ambiguity rules participate in
    :func:`duplicate_copies`, which compares the ion row alone. The earliest
    eligible PMID is retained with its own photon partner; later eligible copies
    are dropped and counted ``n_duplicate_copies`` rather than ``n_paired``. An
    unpaired or excluded original cannot suppress a usable later copy. Re-listings
    that never formed an eligible pair keep their own tally category and are
    additionally counted ``n_relisted_unscored``.

    The record's ``source`` is ``"radphysbio:<PMID>"`` and ``photon_radiation``
    is :func:`photon_quality_label` of the partner. Every record is marked
    ``energy_is_nominal=True``; the harness permits only LET-based models.
    """
    photons: dict[tuple[str, str], list[RadPhysBioEntry]] = {}
    for e in entries:
        if e.is_photon:
            photons.setdefault((e.pmid, cell_key(e.cell_line)), []).append(e)
    for partners in photons.values():
        partners.sort(key=lambda p: p.exp_no)

    records: list[ReferenceRecord] = []
    eligible_entries: list[RadPhysBioEntry] = []
    ambiguous_keys: set[tuple[str, int]] = set()
    n_ion = n_paired = n_unpaired = n_amb = n_incomplete = n_copies = 0
    for e in entries:
        if e.is_photon or e.Z is None:
            continue
        n_ion += 1
        group = photons.get((e.pmid, cell_key(e.cell_line)))
        if not group:
            n_unpaired += 1
            continue
        ambiguous = len({(p.alpha, p.beta) for p in group}) > 1
        if ambiguous:
            n_amb += 1
            ambiguous_keys.add(e.key)
            if not include_ambiguous:
                continue
        photon = group[0]
        n_paired += 1
        if e.let_keV_um is None or e.alpha is None or photon.alpha is None or photon.alpha <= 0.0:
            n_incomplete += 1
            continue
        eligible_entries.append(e)
        records.append(
            ReferenceRecord(
                cell_line=e.cell_line,
                Z=e.Z,
                A=e.A,
                energy_MeV_u=e.energy_MeV_u,
                energy_is_nominal=True,
                let_keV_um=e.let_keV_um,
                alpha_ion=e.alpha,
                beta_ion=e.beta if e.beta is not None else 0.0,
                alpha_x=photon.alpha,
                beta_x=photon.beta if photon.beta is not None else 0.0,
                photon_radiation=photon_quality_label(photon),
                source=f"radphysbio:{e.pmid}",
            )
        )
    if collapse_duplicates:
        copies = duplicate_copies(eligible_entries)
        retained: list[ReferenceRecord] = []
        for entry, record in zip(eligible_entries, records, strict=True):
            if entry.key in copies:
                n_copies += 1
                n_paired -= 1
                n_amb -= entry.key in ambiguous_keys
            else:
                retained.append(record)
        records = retained
    eligible_keys = {e.key for e in eligible_entries}
    n_relisted_unscored = sum(1 for k in duplicate_copies(entries) if k not in eligible_keys)
    report = PairingReport(
        n_ion_rows=n_ion,
        n_paired=n_paired,
        n_unpaired=n_unpaired,
        n_ambiguous=n_amb,
        n_incomplete=n_incomplete,
        n_duplicate_copies=n_copies,
        n_relisted_unscored=n_relisted_unscored,
    )
    return records, report


def load_radphysbio_dataset(
    path: str | Path | None = None,
    *,
    include_ambiguous: bool = True,
    collapse_duplicates: bool = True,
) -> ReferenceDataset:
    """RadPhysBio as a harness :class:`~pyradbio.benchmark.dataset.ReferenceDataset`.

    Loads the export, pairs ion rows with their publication's photon row
    (:func:`pair_photon_references`) and appends the pairing tally to the
    dataset's caveats so every report states how much of the database it scored.
    """
    entries = load_radphysbio(path)
    records, report = pair_photon_references(
        entries, include_ambiguous=include_ambiguous, collapse_duplicates=collapse_duplicates
    )
    tally = (
        f"Pairing tally: {report.n_ion_rows} ion rows, {report.n_duplicate_copies} dropped as "
        f"verbatim copies of an eligible ion row under an earlier PMID, "
        f"{report.n_paired} paired with a "
        f"same-publication photon row ({report.n_ambiguous} of them ambiguous: several "
        f"photon rows, first taken), {report.n_unpaired} without a partner, "
        f"{report.n_incomplete} paired but lacking LET/alpha; {len(records)} records scored. "
        f"{report.n_relisted_unscored} of the rows not scored are verbatim re-listings "
        f"of a row under an earlier PMID."
    )
    return ReferenceDataset(
        name="RadPhysBio",
        records=tuple(records),
        citation=RADPHYSBIO_CITATION,
        caveats=(*RADPHYSBIO_CAVEATS, tally),
    )
