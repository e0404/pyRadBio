"""build_alpha_beta_table over an injected model (no pyMKM)."""

from __future__ import annotations

from pyradbio.core.lq import LQParameters
from pyradbio.core.radiation import Ion
from pyradbio.models.base import BiophysicalModel
from pyradbio.tables.alpha_beta import AlphaBetaTable, build_alpha_beta_table


class _FakeModel(BiophysicalModel):
    """Records every ion it is asked about; alpha encodes (Z, energy)."""

    def __init__(self) -> None:
        self.seen: list[Ion] = []

    def _alpha_beta_ion(self, ion: Ion) -> LQParameters:
        self.seen.append(ion)
        return LQParameters(alpha=ion.Z + ion.energy_MeV_u / 1000.0, beta=0.05)


def test_builds_ion_by_energy_outer_product() -> None:
    model = _FakeModel()
    ions = (Ion(Z=1, A=1, energy_MeV_u=1.0), Ion(Z=6, A=12, energy_MeV_u=1.0))
    energies = (10.0, 100.0)

    table = build_alpha_beta_table(model, ions, energies, metadata={"model": "fake"})

    assert isinstance(table, AlphaBetaTable)
    assert len(table.rows) == len(ions) * len(energies)
    # Row order: ions outer, energies inner.
    assert [(r.Z, r.energy_MeV_u) for r in table.rows] == [
        (1, 10.0),
        (1, 100.0),
        (6, 10.0),
        (6, 100.0),
    ]
    # The resolved model configuration rides along by default; explicit
    # metadata keeps its own keys.
    assert table.metadata["model"] == "fake"
    assert table.metadata["model_config"]["model"] == "_FakeModel"


def test_sweeps_energy_grid_not_ion_energy() -> None:
    """The ion's own energy is ignored; the energies grid is swept instead."""
    model = _FakeModel()
    ions = (Ion(Z=2, A=4, energy_MeV_u=999.0),)

    table = build_alpha_beta_table(model, ions, (5.0, 50.0))

    assert [ion.energy_MeV_u for ion in model.seen] == [5.0, 50.0]
    assert table.rows[0].A == 4
    # alpha = Z + energy/1000 from the fake model.
    assert table.rows[0].alpha == 2 + 5.0 / 1000.0


def test_empty_metadata_default() -> None:
    """With no metadata and provenance switched off, the header is empty."""
    model = _FakeModel()
    table = build_alpha_beta_table(
        model, (Ion(Z=1, A=1, energy_MeV_u=1.0),), (1.0,), record_model_config=False
    )
    assert table.metadata == {}


def test_model_provenance_reports_defaults_not_just_what_was_passed() -> None:
    """The resolved configuration, so a table names the protocol behind it."""
    from pyradbio.core.cell import CellTarget
    from pyradbio.core.photon import PhotonParameters
    from pyradbio.models.lem.lem4 import LEM4
    from pyradbio.tables.alpha_beta import MODEL_CONFIG_KEY, model_provenance

    lem = LEM4(
        photon=PhotonParameters(alpha_x=0.1, beta_x=0.05, D_t=30.0),
        cell=CellTarget(nucleus_radius_um=5.0),
    )
    config = model_provenance(lem)
    assert config["model"] == "LEM4"
    assert config["survival_convention"] == "mean_survival"  # never passed
    assert config["cluster_index_model"] == "mean_field"
    assert isinstance(config["photon"], dict)
    # A non-dataclass collaborator reports its class, not its internal tables.
    assert isinstance(config["stopping_power"], dict)
    assert "type" in config["stopping_power"]

    import json

    json.dumps(config)  # the CSV header serializes it verbatim

    table = build_alpha_beta_table(lem, (Ion(Z=1, A=1, energy_MeV_u=1.0),), (10.0,))
    assert table.metadata[MODEL_CONFIG_KEY] == config


def test_explicit_metadata_wins_and_recording_can_be_switched_off() -> None:
    from pyradbio.tables.alpha_beta import MODEL_CONFIG_KEY

    model = _FakeModel()
    ions = (Ion(Z=1, A=1, energy_MeV_u=1.0),)
    plain = build_alpha_beta_table(model, ions, (10.0,), record_model_config=False)
    assert plain.metadata == {}

    overridden = build_alpha_beta_table(
        model, ions, (10.0,), metadata={MODEL_CONFIG_KEY: "hand-written"}
    )
    assert overridden.metadata[MODEL_CONFIG_KEY] == "hand-written"
