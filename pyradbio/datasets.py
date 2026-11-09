"""Locate non-committed external reference datasets via environment / ``.env``.

pyRadBio cross-checks its models against reference data that is **not** stored in
this repository -- TRiP98 RBE / dE-dx tables, "Survival"-code JSON exports, the
PIDE database, and measured / Monte-Carlo lineal-energy libraries. Each data
*type* has its own directory, named by an environment variable, so different
machines can keep the data wherever they like (a shared Nextcloud folder, a
scratch disk, ...).

A local, git-ignored ``.env`` at the project root supplies these variables; copy
``.env.example`` and edit the paths. ``.env`` is loaded lazily on first lookup
(via ``python-dotenv`` when installed) and never overrides variables already set
in the real environment, so CI / explicit exports win. Only self-digitised paper
data is bundled inside the package; see ``AGENTS.md`` and the data sections of
``THIRD_PARTY_NOTICES.md``.
"""

from __future__ import annotations

import os
from dataclasses import dataclass
from pathlib import Path
from typing import Literal, overload

__all__ = [
    "DATASETS",
    "LINEAL_ENERGY_DECUNHA",
    "PIDE",
    "RADPHYSBIO",
    "SURVIVAL",
    "SURVIVAL_SRC",
    "TRIP98",
    "Dataset",
    "MissingDatasetError",
    "dataset_dir",
    "dataset_file",
    "load_project_dotenv",
]

_dotenv_loaded = False


def load_project_dotenv() -> None:
    """Populate ``os.environ`` from a project ``.env`` once, if possible.

    Uses ``python-dotenv`` when available; a missing package or missing file is
    not an error (the variables may also be exported in the real environment).
    Variables already present in the environment are never overridden.
    """
    global _dotenv_loaded
    if _dotenv_loaded:
        return
    _dotenv_loaded = True
    try:
        from dotenv import find_dotenv, load_dotenv
    except ModuleNotFoundError:
        return
    load_dotenv(find_dotenv(usecwd=True), override=False)


@dataclass(frozen=True, slots=True)
class Dataset:
    """An external, non-committed reference dataset located by an env var."""

    key: str
    env_var: str
    description: str


PIDE = Dataset(
    "pide",
    "PYRADBIO_PIDE_DIR",
    "PIDE database (GSI; registration required, not redistributable).",
)
RADPHYSBIO = Dataset(
    "radphysbio",
    "PYRADBIO_RADPHYSBIO_DIR",
    "RadPhysBio database export (NTUA; Zanni et al. 2024, doi:10.3390/ijms25094729, "
    "CC BY 4.0). Export the CSV from https://radbiodb.physics.ntua.gr/radphysbio/ "
    "into this folder.",
)
TRIP98 = Dataset(
    "trip98",
    "PYRADBIO_TRIP98_DIR",
    "TRiP98 reference tables (subfolders DEDX/ and RBE/{LEM1,LEM4}/).",
)
SURVIVAL = Dataset(
    "survival",
    "PYRADBIO_SURVIVAL_DIR",
    '"Survival"-code JSON exports (rapidLEM, rapid mMKM).',
)
SURVIVAL_SRC = Dataset(
    "survival_src",
    "PYRADBIO_SURVIVAL_SRC_DIR",
    "batuff/Survival *source checkout* (GPL-2, not redistributable here). Its "
    "data/ folder holds the SRIM <Ion>_Srim.dat stopping-power tables and "
    "eta_Elsasser2007.dat. Separate from PYRADBIO_SURVIVAL_DIR, which holds "
    "exported result JSON rather than the code and its input tables.",
)
LINEAL_ENERGY_DECUNHA = Dataset(
    "lineal_energy_decunha",
    "PYRADBIO_LINEAL_ENERGY_DECUNHA_DIR",
    "DeCunha 2025 lineal-energy spectrum library (Geant4-DNA, doi:10.1002/mp.17561). "
    "Keep other lineal-energy sources under their own dataset/env var.",
)

DATASETS: dict[str, Dataset] = {
    d.key: d for d in (PIDE, RADPHYSBIO, TRIP98, SURVIVAL, SURVIVAL_SRC, LINEAL_ENERGY_DECUNHA)
}
"""All known external datasets, keyed by [`Dataset.key`][]."""


class MissingDatasetError(FileNotFoundError):
    """An external dataset directory is not configured or does not exist.

    Subclasses `FileNotFoundError` so callers that already skip on missing
    reference data keep working unchanged.
    """

    def __init__(self, dataset: Dataset, resolved: Path | None) -> None:
        if resolved is None:
            msg = (
                f"{dataset.description}\n"
                f"Set {dataset.env_var} to its directory (e.g. in a local .env; "
                "copy .env.example). This data is not bundled with pyRadBio."
            )
        else:
            msg = (
                f"{dataset.description}\n"
                f"{dataset.env_var} points at {resolved}, which does not exist. "
                "Fix the path in your .env."
            )
        super().__init__(msg)
        self.dataset = dataset
        self.resolved = resolved


@overload
def dataset_dir(dataset: Dataset, *, required: Literal[True] = ...) -> Path: ...
@overload
def dataset_dir(dataset: Dataset, *, required: bool) -> Path | None: ...
def dataset_dir(dataset: Dataset, *, required: bool = True) -> Path | None:
    """Directory configured for ``dataset`` via its environment variable.

    Loads ``.env`` on first use. Supports ``~`` expansion and forward slashes on
    all platforms. Returns ``None`` when unset/absent and ``required`` is
    ``False``; otherwise raises [`MissingDatasetError`][].
    """
    load_project_dotenv()
    raw = os.environ.get(dataset.env_var)
    if not raw:
        if required:
            raise MissingDatasetError(dataset, None)
        return None
    path = Path(raw).expanduser()
    if not path.is_dir():
        if required:
            raise MissingDatasetError(dataset, path)
        return None
    return path


def dataset_file(dataset: Dataset, *relative: str) -> Path:
    """Resolve ``<dataset dir>/<relative...>``, requiring the file to exist."""
    base = dataset_dir(dataset)
    path = base.joinpath(*relative)
    if not path.is_file():
        raise FileNotFoundError(
            f"{path} not found under {dataset.env_var}={base}. "
            "Check the layout expected by the loader."
        )
    return path
