"""Reference-implementation benchmark tables (TRiP RBE + Survival-code JSON).

pyRadBio's LEM/MKM models are cross-checked against tables produced by two
established, independent implementations:

* **TRiP98 RBE tables** (``*.rbe``) — GSI's treatment-planning LEM output. Each
  file has a photon header (``alpha``/``beta``/``cut``=D_t/``rnucleus``) followed
  by one block per projectile ion listing ``RBE_alpha`` (= ``alpha_ion/alpha_x``,
  the initial-slope RBE) versus energy per nucleon. LEM I and LEM IV tables live
  under ``$PYRADBIO_TRIP98_DIR/RBE/{LEM1,LEM4}/`` (not committed; see
  [`pyradbio.datasets`][]).
* **"Survival" code JSON** — per-ion ``alpha``/``beta`` curves from the
  open-source *survival* toolkit (Manganaro et al.), for LEM I (rapidLEM,
  Kraemer-Scholz 2006 approximation) and modified-MKM (rapid mMKM, Inaniwa 2010).
  ``meta.modelParameters`` carries the photon/geometry inputs; ``data`` is a list
  of ``{Z, A, energies, alpha, beta}`` per ion.

These are *reference model outputs*, not experimental data, so they support tight
implementation-agreement benchmarks (see ``tests/benchmark/test_reference_tables.py``).
Loaders here parse and expose the data. A few thin conveniences turn a table into
model inputs -- [`TripRbeTable.photon_parameters`][] / [`TripRbeTable.cell_target`][]
(the header's photon set and nucleus), [`TripDedxTable.stopping_power`][] (TRiP's own
dE/dx as a [`StoppingPower`][pyradbio.track.stopping_power.StoppingPower]) -- and
[`windowed_median_errors`][] scores a curve per energy window; building models
is left to the caller.

The tables are **not** redistributed with pyRadBio: their directories are given
by ``PYRADBIO_TRIP98_DIR`` and ``PYRADBIO_SURVIVAL_DIR`` (via ``.env``; see
[`pyradbio.datasets`][]). A loader raises `FileNotFoundError` when the
data is absent, so data-driven benchmarks skip cleanly on machines without it.
"""

from __future__ import annotations

import bisect
import json
import math
import re
from collections.abc import Mapping, Sequence
from dataclasses import dataclass
from functools import cache, lru_cache
from typing import TYPE_CHECKING, Any, Protocol

import numpy as np
import numpy.typing as npt

from pyradbio.core.cell import CellTarget
from pyradbio.core.elements import SYMBOL_TO_Z
from pyradbio.core.photon import PhotonParameters
from pyradbio.datasets import SURVIVAL, SURVIVAL_SRC, TRIP98, dataset_file

if TYPE_CHECKING:
    from pyradbio.core.radiation import Ion

# Element symbol -> atomic number Z for the ions appearing in the tables
# (hydrogen through neon; TRiP writes the ion as "<A><Symbol>", e.g. "12C").
# One map for the whole package: `pyradbio.core.elements.SYMBOL_TO_Z`.
ELEMENT_Z: Mapping[str, int] = SYMBOL_TO_Z

# TRiP tables live under the PYRADBIO_TRIP98_DIR base as RBE/{LEM1,LEM4}/;
# Survival-code JSON exports live directly under PYRADBIO_SURVIVAL_DIR.
_TRIP_LEM1_SUBDIR = ("RBE", "LEM1")
_TRIP_LEM4_SUBDIR = ("RBE", "LEM4")
_LEM1_SURVIVAL_JSON = "LEMI_Scholz2006_HSG_SurvivalCode.json"
_MMKM_SURVIVAL_JSON = "mMKM_HSG_SurvivalCode.json"
_ETA_DAT = "eta_Elsasser2007.dat"

_PROJECTILE_RE = re.compile(r"^\s*(\d+)\s*([A-Za-z]+)\s*$")


# ---- typed containers ---------------------------------------------------
@dataclass(frozen=True, slots=True)
class ProjectileRbeCurve:
    """One projectile's ``RBE_alpha`` vs energy from a TRiP RBE table."""

    Z: int
    A: int
    energies_MeV_u: tuple[float, ...]
    rbe_alpha: tuple[float, ...]


@dataclass(frozen=True, slots=True)
class TripRbeTable:
    """A parsed TRiP98 ``.rbe`` table: photon header + per-projectile curves."""

    celltype: str
    alpha_x: float  # Gy^-1
    beta_x: float  # Gy^-2
    D_t: float  # transition ("cut") dose [Gy]
    r_nucleus_um: float
    curves: Mapping[tuple[int, int], ProjectileRbeCurve]

    def curve(self, Z: int, A: int) -> ProjectileRbeCurve:
        """The projectile curve for ion ``(Z, A)`` (KeyError if absent)."""
        return self.curves[(Z, A)]

    def photon_parameters(self) -> PhotonParameters:
        """The header's photon reference ``(alpha_x, beta_x, D_t)``, no cluster parameters.

        Equal to ``PhotonParameters(alpha_x=table.alpha_x, beta_x=table.beta_x,
        D_t=table.D_t)``, the construction the TRiP98 scripts and tests spelled out
        (``D_t`` is the header's ``!cut``). LEM I and LEM IV do not consume
        ``photon.cluster`` (LEM IV warns on it); a LEM II/III caller sets it itself.
        """
        return PhotonParameters(alpha_x=self.alpha_x, beta_x=self.beta_x, D_t=self.D_t)

    def cell_target(self) -> CellTarget:
        """The header's nucleus (``!rnucleus``), ``effective_size_factor`` passed explicitly.

        Equal to ``CellTarget(table.r_nucleus_um)``: TRiP's radius is the effective
        nucleus, used as-is, so the factor is ``1.0``; it is forwarded rather than
        left to ``CellTarget``'s default so that convention stays visible.
        """
        return CellTarget(nucleus_radius_um=self.r_nucleus_um, effective_size_factor=1.0)


@dataclass(frozen=True, slots=True)
class DedxCurve:
    """One projectile's mass stopping power (and optional range) vs energy."""

    Z: int
    A: int
    energies_MeV_u: tuple[float, ...]
    mass_stopping_MeV_cm2_g: tuple[float, ...]
    range_g_cm2: tuple[float, ...] | None


@dataclass(frozen=True, slots=True)
class TripDedxTable:
    """A parsed TRiP98 ``.dedx`` table: material header + per-projectile curves.

    This is the exact stopping power TRiP98 used to generate the RBE tables (an
    old FLUKA H2O table, ``I = 77.3 eV``), so feeding it into pyRadBio removes the
    stopping-power variable when benchmarking against the ``.rbe`` curves.
    """

    material: str
    density_g_cm3: float
    curves: Mapping[tuple[int, int], DedxCurve]

    def curve(self, Z: int, A: int) -> DedxCurve:
        """The dE/dx curve for ion ``(Z, A)`` (KeyError if absent)."""
        return self.curves[(Z, A)]

    def stopping_power(self, fallback: _LetSource | None = None) -> TripDedxStoppingPower:
        """This table as a [`StoppingPower`][pyradbio.track.stopping_power.StoppingPower].

        Returns a [`TripDedxStoppingPower`][]; ions the table lacks go to
        ``fallback`` (any object with ``let_keV_um(ion)``) or raise ``KeyError``.
        """
        return TripDedxStoppingPower(self, fallback)


class _LetSource(Protocol):
    """A stopping power that can stand in for an ion the dE/dx table lacks."""

    def let_keV_um(self, ion: Ion) -> float: ...


class TripDedxStoppingPower:
    """A TRiP98 dE/dx table as a [`StoppingPower`][pyradbio.track.stopping_power.StoppingPower].

    It log-log interpolates the tabulated mass stopping power of each projectile
    and converts it to LET (``MeV cm^2/g * rho[g/cm^3] * 0.1 -> keV/um``). Feeding
    TRiP's own table (an old FLUKA H2O table, ``I = 77.3 eV``) into a model removes
    the stopping-power variable from a comparison against TRiP's ``.rbe`` curves,
    so any residual is the biophysics. An ion absent from the table (the bundled
    FLUKA file stops at carbon, Z <= 6) goes to ``fallback`` if one is given and
    otherwise raises ``KeyError``.

    It carries a ``name`` that
    [`describe_stopping_power`][pyradbio.track.stopping_power.describe_stopping_power] maps.
    Construct it with [`TripDedxTable.stopping_power`][]. Instances hash by identity.

    Attributes:
        table: The parsed dE/dx table.
        density_g_cm3: The table's material density.
        fallback: The stopping power used for ions the table lacks, or ``None``.
    """

    name = "trip98-fluka-dedx"
    """[`describe_stopping_power`][pyradbio.track.stopping_power.describe_stopping_power] key."""
    _MEV_PER_CM_TO_KEV_PER_UM = 0.1

    def __init__(self, table: TripDedxTable, fallback: _LetSource | None = None) -> None:
        self.table = table
        self.density_g_cm3 = table.density_g_cm3
        self.fallback = fallback
        self._log_grids: dict[
            tuple[int, int], tuple[npt.NDArray[np.float64], npt.NDArray[np.float64]]
        ] = {
            key: (
                np.log(np.asarray(c.energies_MeV_u)),
                np.log(np.asarray(c.mass_stopping_MeV_cm2_g)),
            )
            for key, c in table.curves.items()
        }

    def covers(self, Z: int, A: int) -> bool:
        """Whether projectile ``(Z, A)`` is tabulated (no fallback involved)."""
        return (Z, A) in self._log_grids

    def let_keV_um(self, ion: Ion) -> float:
        """Unrestricted LET of ``ion`` in the table's material [keV/um]."""
        key = (ion.Z, ion.A)
        grid = self._log_grids.get(key)
        if grid is None:
            if self.fallback is not None:
                return self.fallback.let_keV_um(ion)
            raise KeyError(f"ion Z{ion.Z}A{ion.A} not in the FLUKA dE/dx table (Z<=6 only)")
        log_e, log_s = grid
        mass_sp = math.exp(float(np.interp(math.log(ion.energy_MeV_u), log_e, log_s)))
        return mass_sp * self.density_g_cm3 * self._MEV_PER_CM_TO_KEV_PER_UM


@dataclass(frozen=True, slots=True)
class IonLQCurve:
    """One ion's ``alpha``/``beta`` vs energy from a Survival-code table.

    ``alpha_sigma``/``beta_sigma`` are present only for tables from Survival's
    Monte-Carlo path, where alpha and beta converge at very different rates and
    the uncertainty is load-bearing rather than decorative -- the rapid methods
    are deterministic and carry none.
    """

    Z: int
    A: int
    energies_MeV_u: tuple[float, ...]
    alpha: tuple[float, ...]  # Gy^-1
    beta: tuple[float, ...]  # Gy^-2
    alpha_sigma: tuple[float, ...] | None = None
    beta_sigma: tuple[float, ...] | None = None


@dataclass(frozen=True, slots=True)
class EtaTable:
    """Survival's tabulated LEM II/III damage enhancement ``eta`` vs local dose."""

    dose_Gy: tuple[float, ...]
    eta: tuple[float, ...]

    def eta_at(self, dose_Gy: float) -> float:
        """``eta`` at one local dose, log-log interpolated as Survival does.

        Outside the tabulated ``(1e2, 5e6]`` Gy window this returns ``1.0``:
        below it there is no cluster enhancement, and above it Survival's
        ``interpolatedDamageEnhancement`` also falls back to 1 rather than
        extrapolating.
        """
        lo, hi = self.dose_Gy[0], self.dose_Gy[-1]
        if not lo < dose_Gy <= hi:
            return 1.0
        log_d = math.log10(dose_Gy)
        k = bisect.bisect_left(self.dose_Gy, dose_Gy)
        k = min(max(k, 1), len(self.dose_Gy) - 1)
        d0, d1 = math.log10(self.dose_Gy[k - 1]), math.log10(self.dose_Gy[k])
        e0, e1 = math.log10(self.eta[k - 1]), math.log10(self.eta[k])
        slope = (e1 - e0) / (d1 - d0)
        return float(10.0 ** (slope * (log_d - d0) + e0))

    def peak(self) -> tuple[float, float]:
        """``(dose_Gy, eta)`` at the tabulated maximum of ``eta``."""
        i = max(range(len(self.eta)), key=self.eta.__getitem__)
        return self.dose_Gy[i], self.eta[i]


@dataclass(frozen=True, slots=True)
class SurvivalLQTable:
    """A parsed Survival-code JSON table: model metadata + per-ion LQ curves."""

    model: str
    parameters: Mapping[str, float | str]
    curves: tuple[IonLQCurve, ...]

    def curve(self, Z: int, A: int) -> IonLQCurve:
        """The LQ curve for ion ``(Z, A)`` (KeyError if absent)."""
        for c in self.curves:
            if c.Z == Z and c.A == A:
                return c
        raise KeyError((Z, A))


# ---- parsers ------------------------------------------------------------
def parse_projectile(token: str) -> tuple[int, int]:
    """Parse a TRiP projectile token like ``"12C"`` into ``(Z, A)``.

    The token is ``<mass number><element symbol>``; the element maps to ``Z`` via
    [`ELEMENT_Z`][]. Raises ``ValueError`` for an unknown symbol or bad format.
    """
    match = _PROJECTILE_RE.match(token)
    if match is None:
        raise ValueError(f"unrecognised projectile token {token!r}")
    a = int(match.group(1))
    symbol = match.group(2)
    if symbol not in ELEMENT_Z:
        raise ValueError(f"unknown element symbol {symbol!r} in projectile {token!r}")
    return ELEMENT_Z[symbol], a


def parse_trip_rbe(text: str) -> TripRbeTable:
    """Parse the text of a TRiP98 ``.rbe`` file into a [`TripRbeTable`][].

    The shared photon header (``!alpha``/``!beta``/``!cut``/``!rnucleus``/
    ``!celltype``) precedes one ``!projectile`` block per ion, each followed by a
    ``!rbe`` marker, a ``#`` comment line, and ``energy  RBE`` rows. Duplicate
    projectile blocks keep the first occurrence.
    """
    header: dict[str, str] = {}
    curves: dict[tuple[int, int], ProjectileRbeCurve] = {}
    current: tuple[int, int] | None = None
    energies: list[float] = []
    rbe: list[float] = []

    def _flush() -> None:
        if current is not None and current not in curves and energies:
            curves[current] = ProjectileRbeCurve(
                Z=current[0],
                A=current[1],
                energies_MeV_u=tuple(energies),
                rbe_alpha=tuple(rbe),
            )

    for raw in text.splitlines():
        line = raw.strip()
        if not line or line.startswith("#"):
            continue
        if line.startswith("!"):
            parts = line[1:].split(None, 1)
            key = parts[0].lower()
            value = parts[1].strip() if len(parts) > 1 else ""
            if key == "projectile":
                _flush()
                current = parse_projectile(value)
                energies, rbe = [], []
            elif key in ("alpha", "beta", "cut", "rnucleus", "celltype"):
                header[key] = value
            continue
        e_str, rbe_str = line.split()[:2]
        energies.append(float(e_str))
        rbe.append(float(rbe_str))
    _flush()

    return TripRbeTable(
        celltype=header.get("celltype", ""),
        alpha_x=float(header["alpha"]),
        beta_x=float(header["beta"]),
        D_t=float(header["cut"]),
        r_nucleus_um=float(header["rnucleus"]),
        curves=curves,
    )


def parse_trip_dedx(text: str) -> TripDedxTable:
    """Parse the text of a TRiP98 ``.dedx`` file into a [`TripDedxTable`][].

    A material header (``!material``/``!density``) precedes one ``!projectile``
    block per ion, each followed by a ``!dedx`` marker and
    ``E  dE/dx  [range]`` rows (energy in MeV/u, mass stopping power in
    MeV cm^2/g, optional range in g/cm^2). Duplicate projectile blocks keep the
    first occurrence.
    """
    header: dict[str, str] = {}
    curves: dict[tuple[int, int], DedxCurve] = {}
    current: tuple[int, int] | None = None
    energies: list[float] = []
    stopping: list[float] = []
    ranges: list[float] = []

    def _flush() -> None:
        if current is not None and current not in curves and energies:
            curves[current] = DedxCurve(
                Z=current[0],
                A=current[1],
                energies_MeV_u=tuple(energies),
                mass_stopping_MeV_cm2_g=tuple(stopping),
                range_g_cm2=tuple(ranges) if len(ranges) == len(energies) else None,
            )

    for raw in text.splitlines():
        line = raw.strip()
        if not line or line.startswith("#"):
            continue
        if line.startswith("!"):
            parts = line[1:].split(None, 1)
            key = parts[0].lower()
            value = parts[1].strip() if len(parts) > 1 else ""
            if key == "projectile":
                _flush()
                current = parse_projectile(value)
                energies, stopping, ranges = [], [], []
            elif key in ("material", "composition", "density"):
                header[key] = value
            continue
        cols = line.split()
        energies.append(float(cols[0]))
        stopping.append(float(cols[1]))
        if len(cols) >= 3:
            ranges.append(float(cols[2]))
    _flush()

    return TripDedxTable(
        material=header.get("material", ""),
        density_g_cm3=float(header.get("density", "1")),
        curves=curves,
    )


def parse_eta_table(text: str) -> EtaTable:
    """Parse Survival's ``eta_Elsasser2007.dat`` (two columns: dose [Gy], eta)."""
    doses: list[float] = []
    etas: list[float] = []
    for raw in text.splitlines():
        line = raw.strip()
        if not line or line.startswith("#"):
            continue
        dose_str, eta_str = line.split()[:2]
        doses.append(float(dose_str))
        etas.append(float(eta_str))
    if len(doses) < 2:
        raise ValueError("eta table needs at least two rows")
    return EtaTable(dose_Gy=tuple(doses), eta=tuple(etas))


def parse_survival_lq(text: str) -> SurvivalLQTable:
    """Parse a Survival-code JSON string into a [`SurvivalLQTable`][]."""
    doc = json.loads(text)
    meta = doc["meta"]

    def _optional(entry: Mapping[str, Any], key: str) -> tuple[float, ...] | None:
        values = entry.get(key)
        return None if values is None else tuple(float(x) for x in values)

    curves = tuple(
        IonLQCurve(
            Z=int(entry["Z"]),
            A=int(entry["A"]),
            energies_MeV_u=tuple(float(x) for x in entry["energies"]),
            alpha=tuple(float(x) for x in entry["alpha"]),
            beta=tuple(float(x) for x in entry["beta"]),
            alpha_sigma=_optional(entry, "alpha_sigma"),
            beta_sigma=_optional(entry, "beta_sigma"),
        )
        for entry in doc["data"]
    )
    return SurvivalLQTable(
        model=str(meta["model"]),
        parameters=dict(meta["modelParameters"]),
        curves=curves,
    )


# ---- external-file accessors (resolved via pyradbio.datasets) -----------
@cache
def load_trip_lem1(filename: str = "normal02.rbe") -> TripRbeTable:
    """Load a TRiP98 **LEM I** RBE table (``$PYRADBIO_TRIP98_DIR/RBE/LEM1/``)."""
    path = dataset_file(TRIP98, *_TRIP_LEM1_SUBDIR, filename)
    return parse_trip_rbe(path.read_text(encoding="utf-8"))


@cache
def load_trip_dedx(filename: str = "dEdxFLUKAxTRiP.dedx") -> TripDedxTable:
    """Load the TRiP98 FLUKA ``.dedx`` table (``$PYRADBIO_TRIP98_DIR/DEDX/``).

    This is the stopping power TRiP used for the LEM I RBE tables; only light
    ions (Z<=6, i.e. up to carbon) are tabulated in the bundled FLUKA file.
    """
    path = dataset_file(TRIP98, "DEDX", filename)
    return parse_trip_dedx(path.read_text(encoding="utf-8"))


@cache
def load_trip_lem4(filename: str = "chordoma_lemIV.rbe") -> TripRbeTable:
    """Load a TRiP98 **LEM IV** RBE table (``$PYRADBIO_TRIP98_DIR/RBE/LEM4/``)."""
    path = dataset_file(TRIP98, *_TRIP_LEM4_SUBDIR, filename)
    return parse_trip_rbe(path.read_text(encoding="utf-8"))


@lru_cache(maxsize=1)
def load_lem1_survival_hsg() -> SurvivalLQTable:
    """Load the Survival-code **LEM I** (rapidLEM) HSG LQ table (external)."""
    path = dataset_file(SURVIVAL, _LEM1_SURVIVAL_JSON)
    return parse_survival_lq(path.read_text(encoding="utf-8"))


@lru_cache(maxsize=1)
def load_mmkm_survival_hsg() -> SurvivalLQTable:
    """Load the Survival-code **mMKM** (rapid mMKM) HSG LQ table (external)."""
    path = dataset_file(SURVIVAL, _MMKM_SURVIVAL_JSON)
    return parse_survival_lq(path.read_text(encoding="utf-8"))


@cache
def load_survival_table(filename: str) -> SurvivalLQTable:
    """Load any Survival-code LQ JSON by filename from ``$PYRADBIO_SURVIVAL_DIR``.

    The two historical exports have their own named loaders above; use this for
    tables you generate yourself from a Survival build (see
    ``docs/references/survival_repo_notes.md`` for the run recipe and the
    ``<MODEL>_<calculus>_<cell>_<ref>.json`` naming these follow).
    """
    path = dataset_file(SURVIVAL, filename)
    return parse_survival_lq(path.read_text(encoding="utf-8"))


@lru_cache(maxsize=1)
def load_survival_eta_table() -> EtaTable:
    """Load Survival's tabulated LEM II/III damage enhancement ``eta(D)``.

    ``$PYRADBIO_SURVIVAL_SRC_DIR/data/eta_Elsasser2007.dat`` -- 200 log-spaced
    dose nodes from 1e2 to 5e6 Gy. This is an **independent implementation's**
    eta curve, which makes it a rare direct check on the LEM II cluster rule
    (see ``docs/references/survival_repo_notes.md``): pyRadBio's own rule and
    Survival's table can be compared point-by-point without going through a
    figure digitization.

    Survival consumes it via ``CellLine::interpolatedDamageEnhancement``, which
    interpolates log-log between nodes and returns exactly ``1`` outside
    ``(1e2, 5e6]`` Gy -- [`EtaTable.eta`][] reproduces that.
    """
    path = dataset_file(SURVIVAL_SRC, "data", _ETA_DAT)
    return parse_eta_table(path.read_text(encoding="utf-8"))


# ---- comparison helpers -------------------------------------------------
def window_indices(energies: Sequence[float], lo_MeV_u: float, hi_MeV_u: float) -> list[int]:
    """Indices of ``energies`` lying within ``[lo, hi]`` MeV/u (inclusive)."""
    return [i for i, e in enumerate(energies) if lo_MeV_u <= e <= hi_MeV_u]


def median_abs_relative_error(predicted: Sequence[float], reference: Sequence[float]) -> float:
    """Median of ``|predicted - reference| / |reference|`` over paired samples.

    A robust (outlier-tolerant) cross-implementation agreement metric. Samples
    with a zero reference are skipped; raises ``ValueError`` if none remain. This
    is a benchmark-local helper distinct from the PIDE harness metrics.
    """
    errors = [abs(p - r) / abs(r) for p, r in zip(predicted, reference, strict=True) if r != 0.0]
    if not errors:
        raise ValueError("no samples with a non-zero reference value")
    errors.sort()
    n = len(errors)
    mid = n // 2
    return errors[mid] if n % 2 else 0.5 * (errors[mid - 1] + errors[mid])


def windowed_median_errors(
    energies: Sequence[float],
    predicted: Sequence[float],
    reference: Sequence[float],
    windows: Mapping[str, tuple[float, float]],
) -> dict[str, tuple[float, int] | None]:
    """Median ``|relative error|`` of a predicted curve per energy window.

    For each ``label -> (lo, hi)`` in ``windows`` (MeV/u, inclusive), the
    samples whose energy lies in the window and whose prediction is not NaN (a
    model undefined there) are scored with [`median_abs_relative_error`][].

    Returns:
        ``{label: (error, n_scored)}`` in ``windows`` order, with ``None`` for a
        window that holds no scorable sample. The per-window loop of the TRiP98
        agreement tables (``examples/lem{1,4}_trip98_rbe_reproduction.py``).
    """
    out: dict[str, tuple[float, int] | None] = {}
    for label, (lo, hi) in windows.items():
        pairs = [
            (predicted[i], reference[i])
            for i in window_indices(energies, lo, hi)
            if not math.isnan(predicted[i])
        ]
        if not pairs:
            out[label] = None
            continue
        err = median_abs_relative_error([p for p, _ in pairs], [r for _, r in pairs])
        out[label] = (float(err), len(pairs))
    return out
