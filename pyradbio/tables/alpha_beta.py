"""Build (ion, energy) -> (alpha, beta) tables for any model.

The canonical in-memory table is a small typed object ([`AlphaBetaTable`][])
that carries only core-Python types, so it can be built, compared and CSV-
serialized without the optional ``tables`` extra. Parquet and pandas views live
in [`pyradbio.tables.io`][] behind that extra.
"""

from __future__ import annotations

from collections.abc import Mapping, Sequence
from dataclasses import dataclass, field, fields, is_dataclass

from pyradbio.core.radiation import Ion
from pyradbio.models.base import BiophysicalModel

COLUMNS: tuple[str, ...] = ("Z", "A", "energy_MeV_u", "alpha", "beta")
"""Column order used by the CSV/Parquet writers, matching [`AlphaBetaRow`][]."""

MODEL_CONFIG_KEY: str = "model_config"
"""Metadata key under which [`model_provenance`][] is stored by
[`build_alpha_beta_table`][].
"""

_COLLABORATOR_ATTRS: tuple[str, ...] = ("source", "name")
"""Attributes worth recording on a non-dataclass collaborator (e.g. a stopping
power), beyond its class name.
"""


@dataclass(frozen=True, slots=True)
class AlphaBetaRow:
    """One ``(ion, energy) -> (alpha, beta)`` entry.

    ``alpha`` is in Gy^-1, ``beta`` in Gy^-2, ``energy_MeV_u`` in MeV/u.
    """

    Z: int
    A: int
    energy_MeV_u: float
    alpha: float
    beta: float


@dataclass(frozen=True, slots=True)
class AlphaBetaTable:
    """A tidy table of LQ parameters over ions and energies.

    ``metadata`` holds JSON-serializable provenance (model name, cell line,
    photon reference ``alpha_x``/``beta_x``, geometry). It round-trips through
    the CSV/Parquet writers in [`pyradbio.tables.io`][].
    """

    rows: tuple[AlphaBetaRow, ...]
    metadata: Mapping[str, object] = field(default_factory=dict)


def _describe(value: object, *, depth: int) -> object:
    """JSON-safe description of one configuration value."""
    if value is None or isinstance(value, bool | int | float | str):
        return value
    if is_dataclass(value) and not isinstance(value, type):
        if depth <= 0:
            return type(value).__name__
        return {
            f.name: _describe(getattr(value, f.name), depth=depth - 1)
            for f in fields(value)
            if f.repr
        }
    if isinstance(value, Mapping):
        return {str(k): _describe(v, depth=depth) for k, v in value.items()}
    if isinstance(value, list | tuple):
        return [_describe(v, depth=depth) for v in value]
    # A collaborator object (stopping power, radial dose, MKM backend...): its
    # class name plus whichever identifying attribute it carries.
    described: dict[str, object] = {"type": type(value).__name__}
    for attr in _COLLABORATOR_ATTRS:
        found = getattr(value, attr, None)
        if isinstance(found, str):
            described[attr] = found
            break
    # Stopping powers resolve their options (ICRU edition, I-value, corrections...),
    # which change the numbers without changing the class.
    settings = getattr(value, "settings", None)
    if callable(settings):
        described["settings"] = _describe(settings(), depth=depth)
    return described


def model_provenance(model: BiophysicalModel, *, depth: int = 2) -> dict[str, object]:
    """The model's **resolved** configuration, as JSON-safe values.

    This is what the model will actually compute with, not the arguments the
    caller happened to pass: every unset field reports its default. That
    distinction is the point. LEM IV in particular has a dozen protocol axes --
    survival convention, cluster-index estimator, beta model, fit window, the
    legacy cluster-index cap -- that change the numbers without changing the
    call, and a table which does not name them cannot be compared with another
    one. See ``docs/references/LEM_IV_OPEN_CHOICES.md``.

    Non-dataclass collaborators (a stopping power, a radial-dose model) are
    recorded by class name plus an identifying attribute such as ``source``,
    since their internals are tables rather than settings.
    """
    described: dict[str, object] = {"model": type(model).__name__}
    if is_dataclass(model) and not isinstance(model, type):
        for f in fields(model):
            if f.repr:
                described[f.name] = _describe(getattr(model, f.name), depth=depth)
    return described


def build_alpha_beta_table(
    model: BiophysicalModel,
    ions: Sequence[Ion],
    energies_MeV_u: Sequence[float],
    *,
    metadata: Mapping[str, object] | None = None,
    record_model_config: bool = True,
) -> AlphaBetaTable:
    """Evaluate ``model`` over the outer product of ``ions`` x ``energies``.

    ``ions`` supply ``(Z, A)`` (their own ``energy_MeV_u`` is ignored — the
    ``energies_MeV_u`` grid is swept instead), so the caller controls the exact
    isotope per species.

    Unless ``record_model_config`` is false, the model's resolved configuration
    ([`model_provenance`][]) is stored under [`MODEL_CONFIG_KEY`][] so the
    written table says which protocol produced it. Explicit ``metadata`` keys
    win on collision.
    """
    rows: list[AlphaBetaRow] = []
    for ion in ions:
        for energy in energies_MeV_u:
            lq = model.alpha_beta(Ion(Z=ion.Z, A=ion.A, energy_MeV_u=energy))
            rows.append(
                AlphaBetaRow(Z=ion.Z, A=ion.A, energy_MeV_u=energy, alpha=lq.alpha, beta=lq.beta)
            )
    merged: dict[str, object] = {}
    if record_model_config:
        merged[MODEL_CONFIG_KEY] = model_provenance(model)
    if metadata:
        merged.update(metadata)
    return AlphaBetaTable(rows=tuple(rows), metadata=merged)
