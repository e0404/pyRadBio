"""Regression checks for configuration copies and consistent survival/RBE routes."""

import math
from dataclasses import replace

import numpy as np
import pytest

from pyradbio.core.cell import CellTarget
from pyradbio.core.photon import ClusterParameters, PhotonParameters
from pyradbio.core.radiation import Ion, RadiationField, SpectrumComponent, dose_to_fluence
from pyradbio.models.katz import KatzCellularModel
from pyradbio.models.lem.lem1 import LEM1
from pyradbio.models.lem.lem2 import LEM2, Friedrich2015ClusterModel
from pyradbio.models.lem.lem3 import LEM3
from pyradbio.models.lem.lem4 import LEM4
from pyradbio.models.mkm import OSMK2021, OSMK2023, StochasticMKM
from pyradbio.models.phenomenological import McNamara
from pyradbio.track.stopping_power import TabularStoppingPower

CARBON = Ion(Z=6, A=12, energy_MeV_u=100.0)
PHOTON = PhotonParameters(alpha_x=0.1, beta_x=0.05, D_t=7.5)
CELL = CellTarget(nucleus_radius_um=5.0)


@pytest.mark.parametrize("cls", [LEM1, LEM2, LEM3, LEM4])
def test_replacing_lem_stopping_power_matches_fresh_construction(cls):
    original = cls(photon=PHOTON, cell=CELL)
    sp = TabularStoppingPower.from_arrays({6: ([1.0, 1000.0], [100.0, 100.0])})
    copied = replace(original, stopping_power=sp)
    fresh = cls(photon=PHOTON, cell=CELL, stopping_power=sp)
    radii = np.geomspace(0.01, 1.0, 12)
    # The track's local dose must follow the new LET, not only the mixing weights.
    assert [copied._radial().local_dose_Gy(r, CARBON) for r in radii] == pytest.approx(
        [fresh._radial().local_dose_Gy(r, CARBON) for r in radii], rel=1e-12
    )
    assert copied._radial() is not original._radial()
    if cls is LEM1:
        assert copied.alpha_beta(CARBON) == fresh.alpha_beta(CARBON)


@pytest.mark.parametrize("cls", [LEM2, LEM3])
def test_replacing_photon_cluster_parameters_rebuilds_default_response(cls):
    original = cls(photon=PHOTON, cell=CELL)
    photon = replace(PHOTON, cluster=ClusterParameters(ssb_per_gy_cell=2500.0))
    copied = replace(original, photon=photon)
    fresh = cls(photon=photon, cell=CELL)
    assert fresh._photon_log_kill(1e5) != original._photon_log_kill(1e5)
    assert copied._photon_log_kill(1e5) == fresh._photon_log_kill(1e5)
    assert copied._cluster() is fresh._cluster()  # defaults remain memoised


def test_lem_copy_preserves_explicit_injections_and_can_reset_them():
    radial = LEM1(photon=PHOTON, cell=CELL)._radial()
    cluster = Friedrich2015ClusterModel()
    original = LEM2(photon=PHOTON, cell=CELL, radial_dose=radial, cluster=cluster)
    photon = replace(PHOTON, cluster=ClusterParameters(ssb_per_gy_cell=2500.0))
    copied = replace(original, photon=photon)
    assert copied._radial() is radial
    assert copied._cluster() is cluster
    reset = replace(copied, radial_dose=None, cluster=None)
    fresh = LEM2(photon=photon, cell=CELL)
    assert reset._radial() is not radial
    assert reset._photon_log_kill(1e5) == fresh._photon_log_kill(1e5)


@pytest.mark.parametrize("dose", [0.5, 5.0, 1000.0])
def test_phenomenological_rbe_routes_share_the_transitioned_reference(dose):
    photon = PhotonParameters(alpha_x=0.1, beta_x=0.05, D_t=2.0)
    ion = Ion(Z=1, A=1, energy_MeV_u=100.0)
    sp = TabularStoppingPower.from_arrays({1: ([1.0, 1000.0], [5.0, 5.0])})
    model = McNamara(photon=photon, stopping_power=sp)
    lq = model.alpha_beta_from_let(5.0)
    effect = lq.alpha * dose + lq.beta * dose**2
    d_ref = photon.dose_for_log_kill(effect)
    expected = d_ref / dose
    assert model.rbe_at_dose_from_let(5.0, dose) == pytest.approx(expected)
    assert model.rbe_at_dose(ion, dose) == pytest.approx(expected)
    assert model.rbe_at_reference_dose_from_let(5.0, d_ref) == pytest.approx(expected)
    field = RadiationField(components=(SpectrumComponent(ion=ion, dose_Gy=1.0),))
    assert model.rbe_at_dose(field, dose) == pytest.approx(expected)
    if dose < 1000.0:
        assert model.rbe_at_survival_from_let(5.0, math.exp(-effect)) == pytest.approx(expected)
        assert model.rbe_at_survival(field, math.exp(-effect)) == pytest.approx(expected)


@pytest.mark.parametrize("dose", [75.0, 2000.0])
def test_katz_rbe_stays_finite_after_gamma_survival_cancellation(dose):
    """Eq. (10)'s high-dose limit, Katz 1971, doi:10.2307/3573247.

    For these proton doses exp(-(1-P)D/E0) < 3e-17, so the multi-target
    exponential asymptote is accurate beyond the asserted tolerance.
    """
    model = KatzCellularModel.chinese_hamster_skarsgard1967()
    ion = Ion(Z=1, A=1, energy_MeV_u=100.0)
    p = model.ion_kill_probability(ion.Z, ion.beta())
    fluence = dose_to_fluence(dose, model.stopping_power.let_keV_um(ion), model.density_g_cm3)
    effect = model.sigma0_cm2 * p * fluence + (1 - p) * dose / model.E0_Gy - math.log(model.m)
    expected = model.E0_Gy * (effect + math.log(model.m)) / dose
    if dose == 75.0:
        assert model.survival(ion, dose) == pytest.approx(math.exp(-effect), rel=1e-12, abs=0.0)
    assert model.rbe_at_dose(ion, dose) == pytest.approx(expected, rel=1e-12)


@pytest.mark.parametrize(
    "factory",
    [
        StochasticMKM.hsg_inaniwa2018,
        OSMK2021.hsg_inaniwa2021,
        OSMK2023.hsg_inaniwa2023,
    ],
)
def test_taylor_mkm_rbe_uses_log_effect_even_when_survival_underflows(factory):
    """Check eq. (24) numerically; this does not extend its physical validity range."""
    model = factory()
    kwargs = {} if type(model) is StochasticMKM else {"pO2_mmHg": 0.5}
    lq = model.alpha_beta(CARBON, **kwargs)
    z = (
        model.specific_energies(CARBON)
        if not kwargs
        else model.specific_energies_at(CARBON, kwargs["pO2_mmHg"])
    )
    for dose in (2.0, 1000.0):
        gamma = z.z_nucleus_Gy * (0.5 * (lq.alpha + 2 * lq.beta * dose) ** 2 - lq.beta)
        effect = lq.alpha * dose + lq.beta * dose**2 - math.log1p(gamma * dose)
        d_ref = dose * model.rbe_at_dose(CARBON, dose, **kwargs)
        assert model.reference().log_kill(d_ref) == pytest.approx(effect, rel=1e-12)
        if dose == 2.0:
            assert effect == pytest.approx(-math.log(model.survival(CARBON, dose, **kwargs)))
        else:
            assert model.survival(CARBON, dose, **kwargs) == 0.0


def test_numerical_smk_rbe_matches_survival():
    model = StochasticMKM.hsg_inaniwa2018(nucleus_treatment="numerical")
    dose = 2.0
    d_ref = dose * model.rbe_at_dose(CARBON, dose)
    assert model.reference().reference_survival(d_ref) == pytest.approx(
        model.survival(CARBON, dose)
    )


def test_numerical_smk_accumulates_small_survivals_in_log_space(monkeypatch):
    # Isolate the model's integration from FFT tail precision: a supplied
    # two-point multi-event distribution with individually underflowing terms.
    import pyradbio.models.mkm.spectrum_models as module

    model = StochasticMKM.hsg_inaniwa2018(nucleus_treatment="numerical")
    grid, mass = np.array([1000.0, 1001.0]), np.array([0.25, 0.75])
    monkeypatch.setattr(module, "multi_event_density", lambda *args, **kwargs: (grid, mass))
    lq = model.alpha_beta(CARBON)
    effects = lq.alpha * grid + lq.beta * grid**2
    expected = effects[0] - math.log(0.25 + 0.75 * math.exp(-(effects[1] - effects[0])))
    assert model.survival(CARBON, 1000.0) == 0.0
    rbe_value = model.rbe_at_dose(CARBON, 1000.0)
    assert model.reference().log_kill(1000.0 * rbe_value) == pytest.approx(expected, rel=1e-12)


@pytest.mark.parametrize("let_value", [20.0, 100.0])
def test_katz_let_override_reaches_both_rbe_routes(let_value):
    model = KatzCellularModel.chinese_hamster_skarsgard1967()
    dose = 2.0
    s = model.survival(CARBON, dose, let_keV_um=let_value)
    d_ref = model._reference_dose_for_log_kill(-math.log(s))
    assert model.rbe_at_dose(CARBON, dose, let_keV_um=let_value) == pytest.approx(d_ref / dose)
    assert model.rbe_at_survival(CARBON, s, let_keV_um=let_value) == pytest.approx(d_ref / dose)
    # The override must be equivalent to injecting the same stopping power.
    sp = TabularStoppingPower.from_arrays({6: ([1.0, 1000.0], [let_value, let_value])})
    bound = model.with_stopping_power(sp)
    assert model.rbe_at_dose(CARBON, dose, let_keV_um=let_value) == pytest.approx(
        bound.rbe_at_dose(CARBON, dose)
    )


@pytest.mark.parametrize("method,value", [("rbe_at_dose", 2.0), ("rbe_at_survival", 0.1)])
def test_katz_rbe_let_override_validates_and_refuses_fields(method, value):
    model = KatzCellularModel.chinese_hamster_skarsgard1967()
    with pytest.raises(ValueError, match="let_keV_um"):
        getattr(model, method)(CARBON, value, let_keV_um=0.0)
    field = RadiationField(components=(SpectrumComponent(ion=CARBON, dose_Gy=1.0),))
    with pytest.raises(NotImplementedError, match="mixed-field"):
        getattr(model, method)(field, value, let_keV_um=100.0)
