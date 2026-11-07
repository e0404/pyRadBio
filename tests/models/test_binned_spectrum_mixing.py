"""Binned spectrum components are integrated over their energy range.

A :class:`~pyradbio.core.radiation.SpectrumComponent` with ``energy_lo_MeV_u`` /
``energy_hi_MeV_u`` is a finite energy bin. Every model that accepts a mixed
field must integrate it on a sub-grid with the dose weight ``phi(E) S(E)``,
not evaluate it at the representative ``component.ion`` energy (AGENTS.md,
"Spectral / intra-bin integration"). The oracle here is an explicit dense
log-grid dose-weighted average written out in the test, independent of
:mod:`pyradbio.core.mixing`:

* LQ families (LEM, RMF): dose-average ``alpha`` and ``sqrt(beta)``;
* MKM: dose-average ``z*_1D``, then ``alpha = alpha0 + beta0 z*``;
* phenomenological: dose-average the LET, then evaluate the model at ``LET_d``.

A uniform fluence density ``phi(E) = Phi/(E_hi - E_lo)`` is the convention for a
fluence-specified bin, so the dose weight is ``S(E)`` up to a constant.
"""

from __future__ import annotations

import math
from collections.abc import Callable
from dataclasses import dataclass, field

import numpy as np
import pytest
from scipy.integrate import simpson

from pyradbio.core.cell import CellTarget
from pyradbio.core.lq import LQParameters
from pyradbio.core.photon import PhotonParameters
from pyradbio.core.radiation import Ion, RadiationField, SpectrumComponent
from pyradbio.models.base import BiophysicalModel
from pyradbio.models.lem.lem1 import LEM1
from pyradbio.models.lem.lem4 import LEM4
from pyradbio.models.mkm.base import BackendZStarTable
from pyradbio.models.mkm.model import MKMModel
from pyradbio.models.phenomenological import McNamara
from pyradbio.models.rmf.model import RMFModel
from pyradbio.track.stopping_power import SimpleBetheStoppingPower

V79 = PhotonParameters(alpha_x=0.093, beta_x=0.026, D_t=30.0)
V79_CELL = CellTarget(nucleus_radius_um=3.8)
PHOTON_AB3 = PhotonParameters(alpha_x=0.15, beta_x=0.05)
SP = SimpleBetheStoppingPower()

# A wide carbon bin over which alpha varies strongly (and non-monotonically for
# LEM: it peaks near 5 MeV/u), and the representative energy at its log-centre.
C_LO, C_HI = 1.0, 30.0
C_MID = math.sqrt(C_LO * C_HI)


def _carbon_bin(lo: float = C_LO, hi: float = C_HI, *, fluence: float = 1.0e7) -> SpectrumComponent:
    return SpectrumComponent(
        Ion(6, 12, math.sqrt(lo * hi)), fluence=fluence, energy_lo_MeV_u=lo, energy_hi_MeV_u=hi
    )


def _reference_dose_average(
    quantity: Callable[[float], float],
    stopping_power: Callable[[float], float],
    lo: float,
    hi: float,
    *,
    nodes: int,
) -> float:
    """Dense reference ``int S(E) q(E) dE / int S(E) dE`` on a log grid (Simpson)."""
    energies = np.geomspace(lo, hi, nodes)
    u = np.log(energies)
    s = np.array([stopping_power(e) for e in energies])
    q = np.array([quantity(e) for e in energies])
    num = simpson(s * q * energies, x=u)
    den = simpson(s * energies, x=u)
    return float(num / den)


# --- the base-class template (no physics) ----------------------------------


@dataclass
class _AnalyticLQ(BiophysicalModel):
    """alpha(E) and beta(E) in closed form; records the energies it is asked for."""

    seen: list[float] = field(default_factory=list)

    def _alpha_beta_ion(self, ion: Ion) -> LQParameters:
        self.seen.append(ion.energy_MeV_u)
        e = ion.energy_MeV_u
        return LQParameters(alpha=1.0 / math.sqrt(e), beta=0.01 + 0.1 / e)

    def _dose_weight_stopping_power(self, ion: Ion) -> float:
        return SP.let_keV_um(ion)


def test_template_wide_bin_matches_dense_oracle_not_bin_centre() -> None:
    model = _AnalyticLQ()
    field = RadiationField([_carbon_bin()])
    lq = model.alpha_beta(field)

    def s(e: float) -> float:
        return SP.let_keV_um(Ion(6, 12, e))

    alpha_ref = _reference_dose_average(lambda e: 1.0 / math.sqrt(e), s, C_LO, C_HI, nodes=801)
    root_ref = _reference_dose_average(
        lambda e: math.sqrt(0.01 + 0.1 / e), s, C_LO, C_HI, nodes=801
    )
    assert lq.alpha == pytest.approx(alpha_ref, rel=1e-4)
    assert lq.beta == pytest.approx(root_ref**2, rel=1e-4)
    # The bin-centre value is off by ~2.6%, two orders above the quadrature error.
    centre = model.alpha_beta(Ion(6, 12, C_MID))
    assert abs(lq.alpha / centre.alpha - 1.0) > 0.01


def test_template_sub_nodes_parameter_and_convergence() -> None:
    model = _AnalyticLQ()
    field = RadiationField([_carbon_bin()])
    model.seen.clear()
    a16 = model.alpha_beta(field).alpha
    assert len(model.seen) == 16  # the default sub-grid
    model.seen.clear()
    a32 = model.alpha_beta(field, sub_nodes=32).alpha
    assert len(model.seen) == 32
    a8 = model.alpha_beta(field, sub_nodes=8).alpha
    # Converged: doubling the default moves the answer far less than halving it.
    assert a16 == pytest.approx(a32, rel=1e-4)
    assert abs(a8 - a32) > abs(a16 - a32)


def test_template_shared_bin_edges_are_evaluated_once() -> None:
    """Adjacent bins share an edge node; the per-call cache evaluates it once."""
    model = _AnalyticLQ()
    field = RadiationField([_carbon_bin(1.0, 4.0), _carbon_bin(4.0, 30.0)])
    model.alpha_beta(field, sub_nodes=8)
    assert len(model.seen) == 15
    assert model.seen.count(4.0) == 1


def test_template_mono_components_are_unchanged() -> None:
    """Unbinned components still use one evaluation at the ion energy."""
    model = _AnalyticLQ()
    a, b = Ion(6, 12, 2.0), Ion(6, 12, 20.0)
    field = RadiationField([SpectrumComponent(a, dose_Gy=1.0), SpectrumComponent(b, dose_Gy=3.0)])
    lq = model.alpha_beta(field)
    la, lb = model.alpha_beta(a), model.alpha_beta(b)
    assert lq.alpha == pytest.approx((1.0 * la.alpha + 3.0 * lb.alpha) / 4.0, rel=1e-15)
    root = (1.0 * math.sqrt(la.beta) + 3.0 * math.sqrt(lb.beta)) / 4.0
    assert lq.beta == pytest.approx(root * root, rel=1e-15)


def test_template_dose_specified_bin_uses_uniform_dose_density() -> None:
    """A bin given by dose (not fluence) spreads that dose uniformly over E."""
    model = _AnalyticLQ()
    comp = SpectrumComponent(
        Ion(6, 12, C_MID), dose_Gy=2.0, energy_lo_MeV_u=C_LO, energy_hi_MeV_u=C_HI
    )
    lq = model.alpha_beta(RadiationField([comp]))
    ref = _reference_dose_average(
        lambda e: 1.0 / math.sqrt(e), lambda e: 1.0, C_LO, C_HI, nodes=801
    )
    assert lq.alpha == pytest.approx(ref, rel=1e-4)


def test_template_narrow_bin_reduces_to_point_value() -> None:
    model = _AnalyticLQ()
    lq = model.alpha_beta(RadiationField([_carbon_bin(5.0, 5.0 * (1.0 + 1e-6))]))
    point = model.alpha_beta(Ion(6, 12, 5.0))
    assert lq.alpha == pytest.approx(point.alpha, rel=1e-6)
    assert lq.beta == pytest.approx(point.beta, rel=1e-6)


# --- LEM ---------------------------------------------------------------------


def test_lem1_wide_bin_integrates_over_the_bin() -> None:
    lem = LEM1(photon=V79, cell=V79_CELL)
    field = RadiationField([_carbon_bin()])
    lq = lem.alpha_beta(field)

    def s(e: float) -> float:
        return lem.stopping_power.let_keV_um(Ion(6, 12, e))

    cache: dict[float, LQParameters] = {}

    def lq_at(e: float) -> LQParameters:
        if e not in cache:
            cache[e] = lem.alpha_beta(Ion(6, 12, e))
        return cache[e]

    alpha_ref = _reference_dose_average(lambda e: lq_at(e).alpha, s, C_LO, C_HI, nodes=41)
    root_ref = _reference_dose_average(lambda e: math.sqrt(lq_at(e).beta), s, C_LO, C_HI, nodes=41)
    assert lq.alpha == pytest.approx(alpha_ref, rel=5e-3)
    assert lq.beta == pytest.approx(root_ref**2, rel=5e-3)
    centre = lem.alpha_beta(Ion(6, 12, C_MID))
    assert abs(lq.alpha / centre.alpha - 1.0) > 0.10


@pytest.mark.parametrize("cls", [LEM1, LEM4])
def test_lem_narrow_bin_reduces_to_point_value(cls: type[LEM1] | type[LEM4]) -> None:
    lem = cls(photon=V79, cell=V79_CELL)
    lq = lem.alpha_beta(RadiationField([_carbon_bin(10.0, 10.0 * (1.0 + 1e-6))]), sub_nodes=4)
    point = lem.alpha_beta(Ion(6, 12, 10.0))
    assert lq.alpha == pytest.approx(point.alpha, rel=1e-5)
    assert lq.beta == pytest.approx(point.beta, rel=1e-5)


# --- RMF -----------------------------------------------------------------------


def test_rmf_wide_bin_integrates_over_the_bin() -> None:
    model = RMFModel(photon=PHOTON_AB3)
    lq = model.alpha_beta(RadiationField([_carbon_bin(3.0, 60.0)]))

    def s(e: float) -> float:
        return model.let_keV_um(Ion(6, 12, e))

    alpha_ref = _reference_dose_average(
        lambda e: model.alpha_beta(Ion(6, 12, e)).alpha, s, 3.0, 60.0, nodes=201
    )
    root_ref = _reference_dose_average(
        lambda e: math.sqrt(model.alpha_beta(Ion(6, 12, e)).beta), s, 3.0, 60.0, nodes=201
    )
    assert lq.alpha == pytest.approx(alpha_ref, rel=1e-3)
    assert lq.beta == pytest.approx(root_ref**2, rel=1e-3)
    centre = model.alpha_beta(Ion(6, 12, math.sqrt(3.0 * 60.0)))
    assert abs(lq.alpha / centre.alpha - 1.0) > 0.05


# --- MKM -----------------------------------------------------------------------


@dataclass
class _AnalyticZStarBackend:
    """An MKMBackend with a closed-form z*_1D(E) and a Bethe LET (no pyMKM)."""

    def z_star_1D_Gy(self, ion: Ion) -> float:
        return 5.0 / ion.energy_MeV_u**0.8

    def z_star_1D_table_Gy(self, ions: object, energies_MeV_u: object) -> BackendZStarTable:
        raise NotImplementedError

    def let_keV_um(self, ion: Ion) -> float:
        return SP.let_keV_um(ion)


def test_mkm_wide_bin_dose_averages_z_star_over_the_bin() -> None:
    model = MKMModel(alpha0=0.172, beta0=0.0615, backend=_AnalyticZStarBackend())
    lq = model.alpha_beta(RadiationField([_carbon_bin()]))

    def s(e: float) -> float:
        return SP.let_keV_um(Ion(6, 12, e))

    z_ref = _reference_dose_average(lambda e: 5.0 / e**0.8, s, C_LO, C_HI, nodes=801)
    assert lq.alpha == pytest.approx(0.172 + 0.0615 * z_ref, rel=1e-4)
    assert lq.beta == 0.0615
    # The bin-centre value is off by ~2.1%, two orders above the quadrature error.
    centre = model.alpha_beta(Ion(6, 12, C_MID))
    assert abs(lq.alpha / centre.alpha - 1.0) > 0.01


def test_mkm_narrow_bin_reduces_to_point_value() -> None:
    model = MKMModel(alpha0=0.172, beta0=0.0615, backend=_AnalyticZStarBackend())
    lq = model.alpha_beta(RadiationField([_carbon_bin(5.0, 5.0 * (1.0 + 1e-6))]))
    assert lq.alpha == pytest.approx(model.alpha_beta(Ion(6, 12, 5.0)).alpha, rel=1e-6)


# --- phenomenological (LET_d) -------------------------------------------------


def _proton_bin(lo: float, hi: float) -> SpectrumComponent:
    return SpectrumComponent(
        Ion(1, 1, math.sqrt(lo * hi)), fluence=1.0e8, energy_lo_MeV_u=lo, energy_hi_MeV_u=hi
    )


def test_phenomenological_let_d_is_integrated_over_the_bin() -> None:
    model = McNamara(photon=PHOTON_AB3, stopping_power=SP)
    field = RadiationField([_proton_bin(3.0, 60.0)])

    def s(e: float) -> float:
        return SP.let_keV_um(Ion(1, 1, e))

    let_ref = _reference_dose_average(s, s, 3.0, 60.0, nodes=801)
    assert model.let_d_keV_um(field) == pytest.approx(let_ref, rel=1e-4)
    lq = model.alpha_beta(field)
    expected = model.alpha_beta_from_let(let_ref)
    assert lq.alpha == pytest.approx(expected.alpha, rel=1e-4)
    assert lq.beta == pytest.approx(expected.beta, rel=1e-4)
    centre_let = s(math.sqrt(3.0 * 60.0))
    assert abs(let_ref / centre_let - 1.0) > 0.10


def test_phenomenological_narrow_bin_reduces_to_point_value() -> None:
    model = McNamara(photon=PHOTON_AB3, stopping_power=SP)
    field = RadiationField([_proton_bin(10.0, 10.0 * (1.0 + 1e-6))])
    point = model.alpha_beta(Ion(1, 1, 10.0))
    assert model.alpha_beta(field).alpha == pytest.approx(point.alpha, rel=1e-6)
