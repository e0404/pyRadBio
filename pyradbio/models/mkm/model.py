"""Engine-agnostic MKM model.

:class:`MKMModel` owns the MKM *biophysics* -- the LQ mapping
``alpha = alpha0 + beta * z*_1D`` (Inaniwa 2010) -- and delegates the
``z*_1D`` computation to a pluggable :class:`~pyradbio.models.mkm.base.MKMBackend`.
The default backend is the pyMKM-free
:class:`~pyradbio.models.mkm.native_backend.NativeMKMBackend`; inject another to
swap engines without touching the model:

    MKMModel(...)                            # default: NativeMKMBackend built from params
    MKMModel(backend=PyMKMBackend(...))      # pyMKM (optional extra), for validation
    MKMModel(backend=MyFastBackend())        # any MKMBackend implementation

Geometry has one owner. Without a ``backend`` the model's
``domain_radius_um`` / ``nucleus_radius_um`` / ``beta0`` configure the default
one. With an injected backend those fields default to the backend's own values
(read from its attributes of the same name, when it has them), and setting one to
a different value raises ``ValueError`` instead of being silently ignored.
"""

from __future__ import annotations

from collections.abc import Callable, Sequence
from dataclasses import dataclass
from typing import ClassVar

from pyradbio.core.lq import LQParameters
from pyradbio.core.photon import PhotonParameters
from pyradbio.core.radiation import Ion
from pyradbio.models.base import BiophysicalModel, ModelCapabilities, PhotonReferenceError
from pyradbio.models.mkm.base import MKMAlphaBetaTable, MKMAlphaBetaTableRow, MKMBackend

#: Inaniwa 2010 HSG defaults (doi:10.1088/0031-9155/55/22/008, Table 2).
_DEFAULT_GEOMETRY: dict[str, float] = {
    "domain_radius_um": 0.32,
    "nucleus_radius_um": 3.9,
    "beta0": 0.0615,  # Gy^-2
}


@dataclass(frozen=True, slots=True)
class MKMModel(BiophysicalModel):
    """modified-MKM. Defaults are Inaniwa 2010 HSG.

    Parameters configure both the LQ mapping (``alpha0``, ``beta0``) and -- when
    no explicit ``backend`` is supplied -- the default
    :class:`~pyradbio.models.mkm.native_backend.NativeMKMBackend` built from the
    geometry (``None`` = the Inaniwa 2010 HSG value). With an injected
    ``backend``, ``None`` geometry fields take the backend's values and an
    explicit value that disagrees with the backend raises ``ValueError``.

    To run on pyMKM (the ``mkm`` extra), inject
    ``backend=PyMKMBackend(..., stopping_power_source=...)``. Frozen: the
    geometry and backend are resolved once, in ``__post_init__``.
    """

    capabilities: ClassVar[ModelCapabilities] = ModelCapabilities()

    domain_radius_um: float | None = None
    nucleus_radius_um: float | None = None
    alpha0: float = 0.172  # Gy^-1
    beta0: float | None = None  # Gy^-2
    backend: MKMBackend | None = None
    #: Photon reference for RBE (:meth:`reference`). Optional: the MKM's
    #: ``alpha0``/``beta0`` do not need one, so ``None`` leaves RBE undefined.
    photon: PhotonParameters | None = None

    def __post_init__(self) -> None:
        # Frozen: every resolved value is written once, here, via object.__setattr__.
        if self.backend is not None:
            for name, default in _DEFAULT_GEOMETRY.items():
                mine = getattr(self, name)
                theirs = getattr(self.backend, name, None)
                if mine is None:
                    resolved = default if theirs is None else float(theirs)
                    object.__setattr__(self, name, resolved)
                elif theirs is not None and float(theirs) != float(mine):
                    raise ValueError(
                        f"MKMModel.{name}={mine} conflicts with the injected backend's "
                        f"{name}={theirs}; configure the geometry on the backend only"
                    )
            return
        for name, default in _DEFAULT_GEOMETRY.items():
            if getattr(self, name) is None:
                object.__setattr__(self, name, default)
        from pyradbio.models.mkm.native_backend import NativeMKMBackend

        backend = NativeMKMBackend(
            domain_radius_um=self._geometry("domain_radius_um"),
            nucleus_radius_um=self._geometry("nucleus_radius_um"),
            beta0=self._geometry("beta0"),
        )
        object.__setattr__(self, "backend", backend)

    def reference(self) -> PhotonParameters:
        """The photon reference :attr:`photon`; raises if none was supplied."""
        if self.photon is None:
            raise PhotonReferenceError(
                "MKMModel needs photon=PhotonParameters(...) for RBE: alpha0/beta0 "
                "describe the cell, not the photon reference"
            )
        return self.photon

    def _geometry(self, name: str) -> float:
        value = getattr(self, name)
        assert value is not None  # resolved in __post_init__
        return float(value)

    def _resolve_backend(self) -> MKMBackend:
        """The backend built (or checked) once in ``__post_init__``."""
        assert self.backend is not None  # resolved in __post_init__
        return self.backend

    def _alpha_beta_ion(self, ion: Ion) -> LQParameters:
        """LQ parameters via ``alpha = alpha0 + beta0 * z*_1D`` (Inaniwa 2010, eq. 2).

        ``z*_1D`` comes straight from the backend; ``beta`` is the
        radiation-independent ``beta0`` of the modified MKM.
        """
        return self._lq_from_mixed(self._mixing_quantities(ion))

    def _mixing_quantities(self, ion: Ion) -> tuple[float, ...]:
        """MKM mixes on ``z*_1D`` (Inaniwa 2010, eq. 8 / appendix), not on alpha.

        For a mixed :class:`RadiationField` the inherited
        :meth:`BiophysicalModel.alpha_beta` template dose-averages this
        quantity, integrating binned components over their energy range.
        """
        return (self._resolve_backend().z_star_1D_Gy(ion),)

    def _lq_from_mixed(self, mixed: tuple[float, ...]) -> LQParameters:
        (z_star,) = mixed
        beta0 = self._geometry("beta0")
        return LQParameters(alpha=self.alpha0 + beta0 * z_star, beta=beta0)

    def _dose_weight_stopping_power(self, ion: Ion) -> float:
        return self._stopping_power(self._resolve_backend())(ion)

    def alpha_beta_z_star_table(
        self, ions: Sequence[int], energies_MeV_u: Sequence[float]
    ) -> MKMAlphaBetaTable:
        """An (ion Z, energy) table of ``z*_1D`` *and* the LQ pair, from the backend.

        Uses the backend's batched :meth:`~pyradbio.models.mkm.base.MKMBackend.z_star_1D_table_Gy`
        and augments each row with ``alpha = alpha0 + beta0 * z*_1D`` and
        ``beta = beta0``. Unlike the generic
        :meth:`~pyradbio.models.base.BiophysicalModel.alpha_beta_table` (an
        :class:`~pyradbio.tables.alpha_beta.AlphaBetaTable`, which every model
        returns) the rows keep the MKM-specific ``z*_1D`` and LET columns.
        """
        backend = self._resolve_backend()
        beta0 = self._geometry("beta0")
        z_table = backend.z_star_1D_table_Gy(ions, energies_MeV_u)
        return MKMAlphaBetaTable(
            [
                MKMAlphaBetaTableRow(
                    ion_Z=row.ion_Z,
                    energy_MeV_u=row.energy_MeV_u,
                    let_keV_um=row.let_keV_um,
                    z_star_1D_Gy=row.z_star_1D_Gy,
                    alpha=self.alpha0 + beta0 * row.z_star_1D_Gy,
                    beta=beta0,
                )
                for row in z_table.rows
            ]
        )

    def _stopping_power(self, backend: MKMBackend) -> Callable[[Ion], float]:
        """LET [keV/um] lookup for fluence -> dose conversion in mixed fields.

        Only invoked when a component is specified by fluence rather than dose.
        Every :class:`MKMBackend` provides ``let_keV_um`` (part of the protocol).
        """
        return backend.let_keV_um
