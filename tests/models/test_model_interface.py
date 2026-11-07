"""The shared model interface: capabilities, survival, RBE and the photon reference.

``BiophysicalModel.capabilities`` is the one place a family states what it
implements; the benchmark harness routes on it (and on ``LETModel``) instead of
keeping its own flags, so it must be declared everywhere and must not lie.

Every family answers ``survival(ion, D)``, ``dose_for_survival(ion, S)``,
``rbe_at_survival(ion, S)``, ``rbe_at_dose(ion, D)`` and ``reference()``; the
model-specific axes (OSMK's pO2, IMK's schedule) are keyword-only extensions.
"""

from __future__ import annotations

import inspect
import math
import sys

import numpy as np
import pytest

import pyradbio.models.katz
import pyradbio.models.lem.lem1
import pyradbio.models.lem.lem2
import pyradbio.models.lem.lem3
import pyradbio.models.lem.lem4
import pyradbio.models.mkm
import pyradbio.models.nanox
import pyradbio.models.phenomenological
import pyradbio.models.rmf  # noqa: F401  (imported for the subclass walk)
from pyradbio.core.cell import CellTarget
from pyradbio.core.lq import LQParameters, dose_for_survival, rbe, survival
from pyradbio.core.photon import PhotonParameters
from pyradbio.core.radiation import Ion, RadiationField, SpectrumComponent
from pyradbio.models.base import (
    BiophysicalModel,
    LETModel,
    ModelCapabilities,
    PhotonReferenceError,
)


def _concrete_models() -> list[type[BiophysicalModel]]:
    found: list[type[BiophysicalModel]] = []
    stack: list[type[BiophysicalModel]] = [BiophysicalModel]
    while stack:
        cls = stack.pop()
        for sub in cls.__subclasses__():
            stack.append(sub)
            # ``dataclass(slots=True)`` replaces the class object, so the
            # pre-slots original can linger in ``__subclasses__`` until it is
            # collected; keep only the class its module actually exports.
            exported = getattr(sys.modules[sub.__module__], sub.__name__, None) is sub
            if sub.__module__.startswith("pyradbio.") and exported:
                if not inspect.isabstract(sub) and not sub.__name__.startswith("_"):
                    found.append(sub)
    return sorted(set(found), key=lambda c: c.__qualname__)


CONCRETE = _concrete_models()


def test_the_walk_finds_every_family() -> None:
    names = {cls.__name__ for cls in CONCRETE}
    for expected in ("LEM1", "LEM4", "MKMModel", "StochasticMKM", "OSMK2023", "IMKModel"):
        assert expected in names
    assert {"RMFModel", "KatzCellularModel", "McNamara"} <= names


@pytest.mark.parametrize("cls", CONCRETE, ids=lambda c: c.__name__)
def test_every_model_declares_its_capabilities(cls: type[BiophysicalModel]) -> None:
    assert isinstance(cls.capabilities, ModelCapabilities)


@pytest.mark.parametrize("cls", CONCRETE, ids=lambda c: c.__name__)
def test_let_route_is_exactly_the_let_model_mixin(cls: type[BiophysicalModel]) -> None:
    assert cls.capabilities.let_route == issubclass(cls, LETModel)


def test_family_capabilities() -> None:
    from pyradbio.models.katz import KatzCellularModel
    from pyradbio.models.lem.lem1 import LEM1
    from pyradbio.models.mkm import (
        OSMK2021,
        IMKModel,
        MCFMKMModel,
        MKMModel,
        ModifiedMKM,
        StochasticMKM,
    )
    from pyradbio.models.nanox import NanOxModel
    from pyradbio.models.phenomenological import McNamara
    from pyradbio.models.rmf import RMFModel

    full = ModelCapabilities()
    assert LEM1.capabilities == full
    assert MKMModel.capabilities == full
    assert RMFModel.capabilities == full
    assert McNamara.capabilities == ModelCapabilities(let_route=True)
    spectrum = ModelCapabilities(mixed_field=False, spectrum_route=True)
    for cls in (MCFMKMModel, ModifiedMKM, IMKModel):
        assert cls.capabilities == spectrum
    smk = ModelCapabilities(mixed_field=False, lq_exact=False, spectrum_route=True)
    assert StochasticMKM.capabilities == smk
    assert OSMK2021.capabilities == smk
    assert KatzCellularModel.capabilities == ModelCapabilities(mixed_field=False, lq_exact=False)
    assert NanOxModel.capabilities.ion_route is False


# ---- the uniform survival / RBE interface -----------------------------------

CARBON = Ion(Z=6, A=12, energy_MeV_u=100.0)
PROTON = Ion(Z=1, A=1, energy_MeV_u=10.0)


def _rmf():  # type: ignore[no-untyped-def]
    from pyradbio.models.rmf import RMFModel

    return RMFModel.v79_furusawa2000()


def test_lq_default_survival_is_the_lq_curve_for_scalars_and_arrays() -> None:
    model = _rmf()
    lq = model.alpha_beta(CARBON)
    assert model.survival(CARBON, 2.0) == survival(2.0, lq.alpha, lq.beta)
    doses = np.array([0.0, 1.0, 4.0])
    curve = model.survival(CARBON, doses)
    assert isinstance(curve, np.ndarray)
    assert curve == pytest.approx([survival(float(d), lq.alpha, lq.beta) for d in doses])


def test_lq_default_dose_for_survival_inverts_survival() -> None:
    model = _rmf()
    dose = model.dose_for_survival(CARBON, 0.1)
    assert model.survival(CARBON, dose) == pytest.approx(0.1, rel=1e-12)


def test_rbe_at_survival_matches_core_lq_rbe_against_reference() -> None:
    model = _rmf()
    lq = model.alpha_beta(CARBON)
    ref = model.reference()
    expected = rbe(0.1, lq.alpha, lq.beta, reference=LQParameters(ref.alpha_x, ref.beta_x))
    assert model.rbe_at_survival(CARBON, 0.1) == pytest.approx(expected, rel=1e-12)


def test_rbe_at_dose_is_iso_effect_and_reaches_rbe_min_without_underflow() -> None:
    model = _rmf()
    lq = model.alpha_beta(CARBON)
    ref = model.reference()
    d = 2.0
    d_ref = d * model.rbe_at_dose(CARBON, d)
    effect = lq.alpha * d + lq.beta * d * d
    assert ref.alpha_x * d_ref + ref.beta_x * d_ref * d_ref == pytest.approx(effect, rel=1e-12)
    # S underflows at 1e4 Gy; the log-kill route does not.
    assert model.rbe_at_dose(CARBON, 1.0e4) == pytest.approx(
        math.sqrt(lq.beta / ref.beta_x), rel=1e-3
    )


def test_mixed_field_survival_uses_the_mixed_lq_pair() -> None:
    model = _rmf()
    field = RadiationField(
        components=(
            SpectrumComponent(ion=CARBON, dose_Gy=1.0),
            SpectrumComponent(ion=PROTON, dose_Gy=1.0),
        )
    )
    lq = model.alpha_beta(field)
    assert model.survival(field, 2.0) == pytest.approx(survival(2.0, lq.alpha, lq.beta))


def test_reference_is_exposed_uniformly() -> None:
    from pyradbio.models.lem.lem1 import LEM1
    from pyradbio.models.mkm import MKMModel, StochasticMKM
    from pyradbio.models.phenomenological import McNamara

    photon = PhotonParameters(alpha_x=0.1, beta_x=0.05, D_t=30.0)
    assert LEM1(photon=photon, cell=CellTarget(nucleus_radius_um=5.0)).reference() is photon
    assert McNamara(photon=photon).reference() is photon
    rmf = _rmf()
    assert rmf.reference() is rmf.photon
    smk = StochasticMKM.hsg_inaniwa2018()
    assert smk.reference() == PhotonParameters(alpha_x=smk.alpha_ref, beta_x=smk.beta_ref)
    with pytest.raises(PhotonReferenceError, match="photon"):
        MKMModel().reference()
    assert MKMModel(photon=photon).reference() is photon


def test_mkm_rbe_needs_a_photon_reference() -> None:
    from pyradbio.models.mkm import MKMModel

    with pytest.raises(PhotonReferenceError):
        MKMModel().rbe_at_survival(CARBON, 0.1)
    photon = PhotonParameters(alpha_x=0.1, beta_x=0.0615)
    assert MKMModel(photon=photon).rbe_at_survival(CARBON, 0.1) > 1.0


def test_imk_without_a_reference_raises_the_shared_error() -> None:
    from pyradbio.models.mkm import IMKModel

    with pytest.raises(PhotonReferenceError, match="reference_z_star_1D_Gy"):
        IMKModel.wi38_matsuya2025().rbe_at_survival(CARBON, 0.1)


def test_imk_schedule_is_a_keyword_on_every_entry_point() -> None:
    from pyradbio.core.protraction import DoseBlock, DoseSchedule
    from pyradbio.models.mkm import IMKModel

    model = IMKModel.hsg_matsuya2025()
    slow = DoseSchedule((DoseBlock(start_h=0.0, duration_h=2.0, dose_Gy=1.0),))
    acute = model.alpha_beta(CARBON)
    protracted = model.alpha_beta(CARBON, schedule=slow)
    assert protracted.alpha == acute.alpha
    assert protracted.beta == pytest.approx(model.time_factor(slow) * acute.beta)
    assert model.survival(CARBON, 4.0, schedule=slow) > model.survival(CARBON, 4.0)
    assert model.dose_for_survival(CARBON, 0.1, schedule=slow) > model.dose_for_survival(
        CARBON, 0.1
    )
    assert model.rbe_at_survival(CARBON, 0.1, schedule=slow) < model.rbe_at_survival(CARBON, 0.1)


def test_smk_rbe_inverts_its_own_non_lq_survival() -> None:
    from pyradbio.models.mkm import StochasticMKM

    smk = StochasticMKM.hsg_inaniwa2018()
    ion = Ion(Z=6, A=12, energy_MeV_u=20.0)
    dose = smk.dose_for_survival(ion, 0.1)
    assert smk.survival(ion, dose) == pytest.approx(0.1, rel=1e-9)
    d_ref = dose_for_survival(0.1, smk.alpha_ref, smk.beta_ref)
    assert smk.rbe_at_survival(ion, 0.1) == pytest.approx(d_ref / dose, rel=1e-12)
    # rbe_at_dose is iso-effect on the SMK curve, not on the LQ core.
    d = 2.0
    d_ref_2 = d * smk.rbe_at_dose(ion, d)
    assert survival(d_ref_2, smk.alpha_ref, smk.beta_ref) == pytest.approx(
        smk.survival(ion, d), rel=1e-9
    )


def test_spectrum_models_refuse_fields_on_every_entry_point() -> None:
    from pyradbio.models.mkm import StochasticMKM

    smk = StochasticMKM.hsg_inaniwa2018()
    field = RadiationField(components=(SpectrumComponent(ion=CARBON, dose_Gy=1.0),))
    for call in (
        lambda: smk.survival(field, 1.0),
        lambda: smk.dose_for_survival(field, 0.1),
        lambda: smk.rbe_at_survival(field, 0.1),
    ):
        with pytest.raises(NotImplementedError):
            call()


def test_osmk_po2_is_a_keyword_on_every_entry_point() -> None:
    from pyradbio.models.mkm import OSMK2021

    osmk = OSMK2021.hsg_inaniwa2021()
    ion = Ion(Z=6, A=12, energy_MeV_u=20.0)
    assert osmk.rbe_at_survival(ion, 0.1, pO2_mmHg=0.5) < osmk.rbe_at_survival(ion, 0.1)
    assert osmk.dose_for_survival(ion, 0.1, pO2_mmHg=0.5) > osmk.dose_for_survival(ion, 0.1)
    assert osmk.rbe_at_dose(ion, 2.0, pO2_mmHg=0.5) < osmk.rbe_at_dose(ion, 2.0)
    assert osmk.alpha_beta(ion, pO2_mmHg=0.5).alpha < osmk.alpha_beta(ion).alpha


def test_let_route_mirrors_the_ion_route() -> None:
    from pyradbio.models.phenomenological import McNamara

    model = McNamara(photon=PhotonParameters(alpha_x=0.1, beta_x=0.05))
    lq = model.alpha_beta_from_let(5.0)
    assert model.survival_from_let(5.0, 2.0) == survival(2.0, lq.alpha, lq.beta)
    assert model.survival_from_let(5.0, np.array([2.0]))[0] == pytest.approx(
        survival(2.0, lq.alpha, lq.beta)
    )
    assert model.dose_for_survival_from_let(5.0, 0.1) == pytest.approx(
        dose_for_survival(0.1, lq.alpha, lq.beta)
    )
    assert model.rbe_at_survival_from_let(5.0, 0.1) == pytest.approx(
        rbe(0.1, lq.alpha, lq.beta, reference=LQParameters(0.1, 0.05))
    )


def test_katz_rbe_uses_its_own_multi_target_photon_curve() -> None:
    from pyradbio.models.katz import KatzCellularModel

    m = KatzCellularModel.chinese_hamster_skarsgard1967()
    with pytest.raises(PhotonReferenceError, match="multi-target"):
        m.reference()
    # The reference inverse undoes eq. (3) across many decades of survival. (At
    # 40 Gy the *forward* 1 - [1 - e^-x]^m has cancelled to ~1e-9, hence rel.)
    for d, rel in ((0.01, 1e-9), (1.0, 1e-9), (5.0, 1e-9), (40.0, 1e-8)):
        s = m.photon_survival(d)
        assert m._reference_dose_for_log_kill(-math.log(s)) == pytest.approx(d, rel=rel)
    assert m._reference_dose_for_log_kill(1000.0) == pytest.approx(
        m.E0_Gy * (1000.0 + math.log(m.m)), rel=1e-12
    )
    ion = Ion(Z=6, A=12, energy_MeV_u=10.0)
    d_ion = m.dose_for_survival(ion, 0.1)
    assert m.survival(ion, d_ion) == pytest.approx(0.1, rel=1e-9)
    d_photon = m._reference_dose_for_log_kill(-math.log(0.1))
    assert m.photon_survival(d_photon) == pytest.approx(0.1, rel=1e-9)
    assert m.rbe_at_survival(ion, 0.1) == pytest.approx(d_photon / d_ion, rel=1e-12)
    assert m.rbe_at_survival(ion, 0.1) > 1.0
    curve = m.survival(ion, np.array([0.5, 1.0]))
    assert curve == pytest.approx([m.survival(ion, 0.5), m.survival(ion, 1.0)])


def test_presets_refuse_a_subclass_they_were_not_fitted_for() -> None:
    """Inherited preset classmethods must not hand out another class's parameters."""
    from pyradbio.models.mkm.imk import IMKModel
    from pyradbio.models.mkm.osmk import OSMK2021
    from pyradbio.models.mkm.spectrum_models import ModifiedMKM, StochasticMKM

    assert type(ModifiedMKM.hsg_mk_inaniwa2018()) is ModifiedMKM
    assert type(StochasticMKM.hsg_inaniwa2018()) is StochasticMKM
    assert type(IMKModel.hsg_matsuya2025()) is IMKModel
    with pytest.raises(TypeError, match="call it on ModifiedMKM"):
        IMKModel.hsg_mk_inaniwa2018()
    with pytest.raises(TypeError, match="call it on StochasticMKM"):
        OSMK2021.hsg_inaniwa2018()
