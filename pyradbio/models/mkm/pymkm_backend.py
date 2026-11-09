"""pyMKM adapter: an optional MKM engine and pyMKM's stopping-power tables.

pyMKM (BeppeMagro/pymkm, Magro et al., Computation 2025, MIT) implements MKM /
modified-MKM / SMK / OSMK with Kiefer-Chatterjee & Scholz-Kraft track structure.
pyRadBio uses it as an **independent cross-check**, not as a default: the
default MKM engine is [`NativeMKMBackend`][]
and the default stopping power is
[`default_stopping_power`][pyradbio.track.stopping_power.default_stopping_power] (ICRU 90's
procedure for Z 1-18, Hubert alpha scaling for Z 19-92 from 2.5 MeV/u, analytic Bethe anchored to
them elsewhere). This module offers [`PyMKMBackend`][] (an [`MKMBackend`][]) and
[`PyMKMStoppingPower`][] (pyMKM's bundled water tables) to validation code and tests, plus
[`pymkm_kiefer_chatterjee_dose`][] (pyMKM's Kiefer-Chatterjee radial dose, the reference for
[`KieferChatterjeeRadialDose`][pyradbio.track.kiefer_chatterjee.KieferChatterjeeRadialDose]); it
needs the optional ``mkm`` extra.

This module is the *only* place that imports pyMKM -- see
[`pyradbio.models.mkm.base`][].

pyMKM API used (verified against the installed package; see
``docs/references/pyMKM_API.md``):

    from pymkm.mktable import MKTable, MKTableParameters
    from pymkm.io.table_set import StoppingPowerTableSet
    sp = StoppingPowerTableSet.from_default_source(source).filter_by_ions(["6"])
    params = MKTableParameters(domain_radius=r_d, nucleus_radius=R_n, beta0=beta)
    table = MKTable(parameters=params, sp_table_set=sp)
    table.compute(ions=[6], energy=[E], parallel=False)
    # table.table[name]["data"] is a DataFrame with columns
    #   energy [MeV/u], let [MeV/cm], z_bar_star_domain [Gy] (== z*_1D)
"""

from __future__ import annotations

import contextlib
import io
from collections.abc import Iterator, Mapping, Sequence
from dataclasses import dataclass
from typing import Any, Literal

import numpy as np
import numpy.typing as npt

from pyradbio.core.constants import MEV_PER_CM_TO_KEV_PER_UM, WATER_DENSITY_G_CM3
from pyradbio.core.radiation import Ion
from pyradbio.models.mkm.base import BackendZStarRow, BackendZStarTable
from pyradbio.track.stopping_power import (
    StoppingPower,
    StoppingPowerTable,
    TabularStoppingPower,
    let_from_mass_stopping_power,
)

_PYMKM_IMPORT_ERROR = (
    'PyMKMBackend requires the optional pyMKM dependency. Install with: pip install "pyradbio[mkm]"'
)


@contextlib.contextmanager
def _silence_pymkm() -> Iterator[None]:
    """Redirect pyMKM's verbose stdout/stderr into an in-memory buffer.

    pyMKM prints configuration tables and tqdm progress bars containing Unicode
    glyphs (``->``, ``beta``, block characters). On a Windows ``cp1252`` console
    these raise `UnicodeEncodeError`; capturing into a text buffer both
    silences the noise and avoids that crash regardless of console encoding.
    """
    sink = io.StringIO()
    with contextlib.redirect_stdout(sink), contextlib.redirect_stderr(sink):
        yield


@dataclass(slots=True)
class PyMKMBackend:
    """MKM engine delegating to pyMKM (validation). Defaults: Inaniwa 2010 HSG.

    Satisfies [`MKMBackend`][]. Geometry and the
    saturation parameter ``beta0`` are fixed at construction (Inaniwa 2010,
    Sec. 2.2): ``domain_radius_um`` (r_d), ``nucleus_radius_um`` (R_n) and
    ``beta0`` map onto pyMKM's ``MKTableParameters``; ``z_bar_star_domain`` is
    read back as ``z*_1D`` [Gy].
    """

    domain_radius_um: float = 0.32
    nucleus_radius_um: float = 3.9
    beta0: float = 0.0615  # Gy^-2, enters the saturation correction (Inaniwa 2010 eq. 5)
    stopping_power_source: str = "mstar_3_12"

    def _require_pymkm(self) -> None:
        try:
            import pymkm  # noqa: F401
        except ImportError as exc:  # pragma: no cover - exercised via monkeypatch in tests
            raise ImportError(_PYMKM_IMPORT_ERROR) from exc

    def _compute_table(self, ions: Sequence[int], energies_MeV_u: Sequence[float]) -> Any:
        """Run pyMKM and return its ``MKTable`` for the given ions/energies.

        Returns the pyMKM ``MKTable`` whose ``.table`` maps element name ->
        ``{"stopping_power_info": ..., "data": DataFrame}``; rows carry
        ``z_bar_star_domain`` (== ``z*_1D`` [Gy]) per energy.
        """
        self._require_pymkm()
        from pymkm.io.table_set import StoppingPowerTableSet
        from pymkm.mktable import MKTable, MKTableParameters

        z_list = list(dict.fromkeys(int(z) for z in ions))  # de-dup, keep order
        with _silence_pymkm():
            sp = StoppingPowerTableSet.from_default_source(
                self.stopping_power_source
            ).filter_by_ions([str(z) for z in z_list])
            params = MKTableParameters(
                domain_radius=self.domain_radius_um,
                nucleus_radius=self.nucleus_radius_um,
                beta0=self.beta0,
            )
            table = MKTable(parameters=params, sp_table_set=sp)
            # parallel=False: deterministic, avoids Windows multiprocessing-spawn issues.
            table.compute(ions=z_list, energy=list(energies_MeV_u), parallel=False)
        return table

    @staticmethod
    def _entry_for_z(table: Any, atomic_number: int) -> Any:
        """Find the per-ion entry in a pyMKM table by atomic number.

        Keyed by element name in pyMKM; we match on ``atomic_number`` so we
        never have to maintain a Z -> element-name map ourselves.
        """
        for entry in table.table.values():
            if entry["stopping_power_info"]["atomic_number"] == atomic_number:
                return entry
        raise KeyError(f"pyMKM returned no entry for Z={atomic_number}")

    @staticmethod
    def _column_values(data: Any, column: str) -> list[float]:
        """Read a numeric column from pyMKM's DataFrame-like result without pandas."""
        values = data[column]
        to_numpy = getattr(values, "to_numpy", None)
        if callable(to_numpy):
            return [float(value) for value in to_numpy()]
        return [float(value) for value in values]

    @classmethod
    def _first_value(cls, data: Any, column: str) -> float:
        values = data[column]
        iloc = getattr(values, "iloc", None)
        if iloc is not None:
            return float(iloc[0])
        return cls._column_values(data, column)[0]

    def z_star_1D_Gy(self, ion: Ion) -> float:
        table = self._compute_table([ion.Z], [ion.energy_MeV_u])
        data = self._entry_for_z(table, ion.Z)["data"]
        return self._first_value(data, "z_bar_star_domain")

    def let_keV_um(self, ion: Ion) -> float:
        """Unrestricted LET [keV/um] from pyMKM's stopping-power table.

        pyMKM tabulates LET in MeV/cm; it is converted here. Used by the model to
        resolve fluence -> dose weights when mixing fluence-specified fields.
        """
        table = self._compute_table([ion.Z], [ion.energy_MeV_u])
        data = self._entry_for_z(table, ion.Z)["data"]
        return self._first_value(data, "let") * MEV_PER_CM_TO_KEV_PER_UM

    def z_star_1D_table_Gy(
        self, ions: Sequence[int], energies_MeV_u: Sequence[float]
    ) -> BackendZStarTable:
        """``z*_1D`` over an (ion Z, energy) grid as dependency-free rows.

        One row per (ion, energy); ``pyradbio.tables`` owns pandas/Parquet conversion.
        """
        table = self._compute_table(ions, energies_MeV_u)
        rows: list[BackendZStarRow] = []
        for z in dict.fromkeys(int(i) for i in ions):
            data = self._entry_for_z(table, z)["data"]
            energies = self._column_values(data, "energy")
            lets = [
                let * MEV_PER_CM_TO_KEV_PER_UM  # pyMKM tabulates MeV/cm
                for let in self._column_values(data, "let")
            ]
            z_stars = self._column_values(data, "z_bar_star_domain")
            rows.extend(
                BackendZStarRow(
                    ion_Z=z,
                    energy_MeV_u=energy,
                    let_keV_um=let,
                    z_star_1D_Gy=z_star,
                )
                for energy, let, z_star in zip(energies, lets, z_stars, strict=True)
            )
        return BackendZStarTable(rows)


class PyMKMStoppingPower(TabularStoppingPower):
    """pyMKM-backed LET lookup for one of pyMKM's bundled stopping-power sources.

    A validation reference (``fluka_2020_0`` / ``geant4_11_3_0`` / ``mstar_3_12``,
    Z = 1-10, 0.1-1000 MeV/u), not a library default. It lives in the pyMKM
    adapter so that [`pyradbio.track`][] never imports pyMKM or ``models``.
    Constructing it validates that the optional ``mkm`` extra is installed.
    The pyMKM stopping-power table is loaded once and scalar/vector lookups are
    handled by local interpolation; no MKM z* table computation is performed.
    """

    def __init__(self, source: Literal["fluka_2020_0", "geant4_11_3_0", "mstar_3_12"]) -> None:
        self.source = source
        table_set = self._load_pymkm_table_set(source)
        super().__init__(self._tables_from_pymkm_table_set(table_set))

    @staticmethod
    def _load_pymkm_table_set(source: str) -> Any:
        try:
            from pymkm.io.table_set import StoppingPowerTableSet
        except ImportError as exc:  # pragma: no cover - exercised via monkeypatch in tests
            raise ImportError(_PYMKM_IMPORT_ERROR) from exc

        with _silence_pymkm():
            return StoppingPowerTableSet.from_default_source(source)

    @classmethod
    def _tables_from_pymkm_table_set(cls, table_set: Any) -> dict[int, StoppingPowerTable]:
        tables: dict[int, StoppingPowerTable] = {}
        for z, data in cls._iter_table_data(table_set):
            mass_stopping_power = cls._column_values(data, "let")
            linear_let = [
                let_from_mass_stopping_power(value, WATER_DENSITY_G_CM3)
                for value in mass_stopping_power
            ]
            tables[z] = StoppingPowerTable.from_arrays(
                cls._column_values(data, "energy"),
                linear_let,
            )
        if not tables:
            raise ValueError("pyMKM stopping-power table set contained no energy/LET tables")
        return tables

    @classmethod
    def _iter_table_data(cls, obj: Any) -> list[tuple[int, Any]]:
        if hasattr(obj, "to_dict") and callable(obj.to_dict):
            obj = obj.to_dict()

        found: list[tuple[int, Any]] = []

        def visit(value: Any, key_z: int | None = None) -> None:
            if isinstance(value, Mapping):
                z = cls._z_from_mapping(value, key_z)
                if "energy" in value and "let" in value and z is not None:
                    found.append((z, value))
                    return
                data = value.get("data")
                if data is not None and z is not None and cls._has_columns(data, "energy", "let"):
                    found.append((z, data))
                    return
                for key, nested in value.items():
                    next_z = int(key) if isinstance(key, int) else None
                    visit(nested, key_z=next_z)
            elif isinstance(value, Sequence) and not isinstance(value, str | bytes):
                for nested in value:
                    visit(nested, key_z=key_z)

        visit(obj)
        return found

    @staticmethod
    def _z_from_mapping(
        mapping: Mapping[str, Any] | Mapping[Any, Any],
        fallback: int | None,
    ) -> int | None:
        for key in ("atomic_number", "atomicNumber", "Z", "z"):
            value = mapping.get(key)
            if value is not None:
                return int(value)
        info = mapping.get("stopping_power_info")
        if isinstance(info, Mapping):
            return PyMKMStoppingPower._z_from_mapping(info, fallback)
        return fallback

    @staticmethod
    def _has_columns(data: Any, *columns: str) -> bool:
        try:
            for column in columns:
                data[column]
        except (KeyError, TypeError):
            return False
        return True

    @staticmethod
    def _column_values(data: Any, column: str) -> list[float]:
        values = data[column]
        to_numpy = getattr(values, "to_numpy", None)
        if callable(to_numpy):
            return [float(value) for value in to_numpy()]
        return [float(value) for value in values]


def from_pymkm_source(
    source: Literal["fluka_2020_0", "geant4_11_3_0", "mstar_3_12"],
) -> StoppingPower:
    """Adapt a pyMKM StoppingPowerTableSet to the StoppingPower protocol.

    source in {"fluka_2020_0", "geant4_11_3_0", "mstar_3_12"}.
    Raises `ImportError` immediately when the ``mkm`` extra is missing.
    """
    return PyMKMStoppingPower(source)


KEV_PER_UM_TO_MEV_PER_CM: float = 10.0
"""pyMKM's ``ParticleTrack`` takes LET in MeV/cm: 1 keV/um = 10 MeV/cm, the
inverse of [`MEV_PER_CM_TO_KEV_PER_UM`][]. A product
(not a division by 0.1) so the value handed to pyMKM is bit-for-bit what the
scripts and tests passed before this helper existed.
"""


def pymkm_kiefer_chatterjee_dose(
    ion: Ion, radii_um: Sequence[float] | npt.NDArray[np.float64], let_keV_um: float
) -> npt.NDArray[np.float64]:
    """pyMKM's Kiefer-Chatterjee local dose ``D(r)`` [Gy] at ``radii_um`` around ``ion``.

    The cross-check reference for
    [`KieferChatterjeeRadialDose`][pyradbio.track.kiefer_chatterjee.KieferChatterjeeRadialDose]:
    ``pymkm.physics.particle_track.ParticleTrack(model_name="Kiefer-Chatterjee")``
    evaluated with its ``initial_local_dose``. The track model is Chatterjee &
    Schaefer 1976 (doi:10.1007/BF01330766) with the Kiefer & Straaten 1986 penumbra
    (doi:10.1088/0031-9155/31/11/002), as implemented by pyMKM (Magro et al.,
    Computation 2025, doi:10.3390/computation13110264); see the class above for
    pyRadBio's side. The caller supplies the LET in
    **keV/um** -- normally from the same stopping power the pyRadBio model uses, so
    the comparison isolates the track model -- and it is converted to the MeV/cm
    pyMKM ingests here, once: every ``let`` pyMKM reads or tabulates is MeV/cm, and
    passing keV/um would silently hand it a LET ten times too small.

    Parameters
    ----------
    ion:
        The ion; pyMKM reads its atomic number and energy per nucleon [MeV/u].
    radii_um:
        Radial distances from the track centre [um].
    let_keV_um:
        Unrestricted LET [keV/um].

    Returns
    -------
    numpy.ndarray
        Local dose [Gy], one value per radius (flattened to 1-D).

    Raises
    ------
    ImportError
        When the optional ``mkm`` extra is not installed.
    """
    try:
        from pymkm.physics.particle_track import ParticleTrack
    except ImportError as exc:
        raise ImportError(_PYMKM_IMPORT_ERROR) from exc

    with _silence_pymkm():
        track = ParticleTrack(
            model_name="Kiefer-Chatterjee",
            energy=ion.energy_MeV_u,
            atomic_number=ion.Z,
            let=let_keV_um * KEV_PER_UM_TO_MEV_PER_CM,
        )
        dose, _ = track.initial_local_dose(radius=np.asarray(radii_um, dtype=float))
    return np.asarray(dose, dtype=float).ravel()
