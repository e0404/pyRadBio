"""The impact-parameter microdosimetry cache of the spectrum-MKM family.

``_SpectrumMKModel.specific_energies`` memoizes its analytic impact-parameter route
in a module-level LRU keyed on exactly the fields that route reads. These tests pin
the contract: the cache never changes a number (on/off bit-identical), the key is
complete (every dataclass field either moves the key or leaves the result alone),
the stateful / unhashable / engine-driven cases bypass it, ``OSMK2023``'s
``replace()`` copies share it, and what it hands out cannot be mutated.
"""

from __future__ import annotations

import dataclasses
import math
from collections.abc import Callable, Iterator
from typing import Any

import numpy as np
import pytest

import pyradbio.models.mkm.spectrum_models as sm
from pyradbio.core.radiation import Ion
from pyradbio.micro.impact_parameter import specific_energy_impact_profile
from pyradbio.micro.lineal_energy import AmorphousTrackEngine, SphericalSite
from pyradbio.models.mkm.imk import IMKModel
from pyradbio.models.mkm.osmk import OSMK2021, OSMK2023
from pyradbio.models.mkm.spectrum_models import (
    MCFMKMModel,
    ModifiedMKM,
    SiteSpecificEnergies,
    StochasticMKM,
    clear_microdosimetry_cache,
    microdosimetry_cache_info,
)
from pyradbio.track.kiefer_chatterjee import KieferChatterjeeRadialDose
from pyradbio.track.northcliffe_schilling import BetheOnly, NorthcliffeSchillingWithBetheTail
from pyradbio.track.stopping_power import (
    DefaultStoppingPower,
    SimpleBetheStoppingPower,
    TabularStoppingPower,
    default_stopping_power,
)

CARBON_LOW = Ion(Z=6, A=12, energy_MeV_u=5.0)
CARBON_HIGH = Ion(Z=6, A=12, energy_MeV_u=100.0)
HELIUM = Ion(Z=2, A=4, energy_MeV_u=10.0)
NEON = Ion(Z=10, A=20, energy_MeV_u=30.0)
PROTON = Ion(Z=1, A=1, energy_MeV_u=20.0)


@pytest.fixture(autouse=True)
def _fresh_cache() -> Iterator[None]:
    """Every test starts and ends with an empty cache (no cross-test leakage)."""
    clear_microdosimetry_cache()
    yield
    clear_microdosimetry_cache()


@pytest.fixture
def coarse_nodes(monkeypatch: pytest.MonkeyPatch) -> None:
    """Cheap node counts: the cache contract does not depend on convergence.

    The node counts are part of the key, so patching them keeps the key honest.
    """
    radial, core, penumbra = _COARSE_NODES
    monkeypatch.setattr(sm, "_IMPACT_RADIAL_NODES", radial)
    monkeypatch.setattr(sm, "_IMPACT_CORE_NODES", core)
    monkeypatch.setattr(sm, "_IMPACT_PENUMBRA_NODES", penumbra)


def _uncached(model: Any, ion: Ion, monkeypatch: pytest.MonkeyPatch) -> SiteSpecificEnergies:
    """``model.specific_energies(ion)`` with the cache switched off."""
    with monkeypatch.context() as m:
        m.setattr(sm, "_MICRODOSIMETRY_CACHE_ENABLED", False)
        out: SiteSpecificEnergies = model.specific_energies(ion)
    return out


_FULL_NODES = (700, 600, 600)
_COARSE_NODES = (60, 40, 40)


def _original_route(
    model: Any, ion: Ion, nodes: tuple[int, int, int] = _FULL_NODES
) -> SiteSpecificEnergies:
    """The impact-parameter route exactly as it read before the cache.

    Verbatim copy of the pre-cache no-engine branch of
    ``_SpectrumMKModel.specific_energies`` (there the node counts were the literals
    700/600/600, the ``_FULL_NODES`` default). It reads the model's fields
    directly, never the cache key, so the cached route is pinned to the pre-cache
    arithmetic and not merely to itself -- and an input the key misses shows up
    as a stale value.

    The copy itself was cross-checked once against the last pre-cache version of
    the package: 56 values (MCF, mMKM, SMK disk/sphere/Kase-z0/explicit
    Bethe/shared-default KC, OSMK2021 incl. a ``replace()`` variant, OSMK2023, IMK;
    C 5 and He 10 MeV/u; ``specific_energies``, ``alpha_beta`` and
    ``specific_energies_at`` at pO2 0 and 5) agreed bit for bit, cold and warm.
    """
    radial_nodes, core_nodes, penumbra_nodes = nodes
    if model.radial_dose is not None:
        radial_dose = model.radial_dose
    else:
        sp = (
            model.stopping_power
            if model.stopping_power is not None
            else default_stopping_power(density_g_cm3=model.density_g_cm3)
        )
        radial_dose = KieferChatterjeeRadialDose(stopping_power=sp)
    z0 = model.z0_effective_Gy
    domain = specific_energy_impact_profile(
        radial_dose,
        ion,
        region_radius_um=model.domain_radius_um,
        geometry=model.site_geometry,
        radial_nodes=radial_nodes,
        core_nodes=core_nodes,
        penumbra_nodes=penumbra_nodes,
        density_g_cm3=model.density_g_cm3,
    )
    nucleus = specific_energy_impact_profile(
        radial_dose,
        ion,
        region_radius_um=model.nucleus_radius_um,
        geometry=model.site_geometry,
        radial_nodes=radial_nodes,
        core_nodes=core_nodes,
        penumbra_nodes=penumbra_nodes,
        density_g_cm3=model.density_g_cm3,
    )
    return SiteSpecificEnergies(
        z_1D_Gy=domain.dose_mean_Gy(),
        z_star_1D_Gy=domain.saturation_corrected_dose_mean_Gy(z0),
        z_nucleus_Gy=nucleus.dose_mean_Gy(),
    )


def _assert_bit_identical(a: SiteSpecificEnergies, b: SiteSpecificEnergies) -> None:
    # dataclass equality on floats is exact ``==``; also rule out -0.0 vs 0.0
    assert a == b
    for f in dataclasses.fields(SiteSpecificEnergies):
        x, y = getattr(a, f.name), getattr(b, f.name)
        if x is not None:
            assert math.copysign(1.0, x) == math.copysign(1.0, y)


# --------------------------------------------------------------------------
# (1) cache on vs off: bit-identical
# --------------------------------------------------------------------------
@pytest.mark.parametrize(
    ("model", "ion"),
    [
        # Full node counts cost ~1-2 s per evaluation; the coarse-node test below
        # covers every class and ion in the fast suite.
        pytest.param(
            StochasticMKM.hsg_inaniwa2018(), CARBON_LOW, id="SMK-C5", marks=pytest.mark.slow
        ),
        pytest.param(
            OSMK2021.hsg_inaniwa2021(), HELIUM, id="OSMK2021-He10", marks=pytest.mark.slow
        ),
        pytest.param(
            IMKModel.hsg_matsuya2025(), CARBON_HIGH, id="IMK-C100", marks=pytest.mark.slow
        ),
    ],
)
def test_cache_is_bit_identical_to_the_pre_cache_route(
    model: Any, ion: Ion, monkeypatch: pytest.MonkeyPatch
) -> None:
    """Full node counts: cold, warm and cache-off all equal the pre-cache arithmetic."""
    reference = _original_route(model, ion)
    cold = model.specific_energies(ion)
    warm = model.specific_energies(ion)
    off = _uncached(model, ion, monkeypatch)
    for value in (cold, warm, off):
        _assert_bit_identical(value, reference)
    info = microdosimetry_cache_info()
    assert (info.misses, info.hits) == (1, 1)


@pytest.mark.usefixtures("coarse_nodes")
def test_osmk2023_hypoxic_route_is_bit_identical(monkeypatch: pytest.MonkeyPatch) -> None:
    """The rescored hypoxic geometry, cached, equals the uncached rescoring."""
    model = OSMK2023.hsg_inaniwa2023()
    cold = model.specific_energies_at(NEON, 0.0)
    warm = model.specific_energies_at(NEON, 0.0)
    with monkeypatch.context() as m:
        m.setattr(sm, "_MICRODOSIMETRY_CACHE_ENABLED", False)
        off = model.specific_energies_at(NEON, 0.0)
    r_d, z0 = model.hypoxic_geometry(0.0)
    hypoxic = dataclasses.replace(model, domain_radius_um=r_d, z0_Gy=z0)
    reference = _original_route(hypoxic, NEON, _COARSE_NODES)
    for value in (cold, warm, off):
        _assert_bit_identical(value, reference)


_ALL_MODELS: list[tuple[str, Any]] = [
    ("MCF", MCFMKMModel()),
    ("mMKM", ModifiedMKM.hsg_mk_inaniwa2018()),
    ("SMK", StochasticMKM.hsg_inaniwa2018()),
    ("SMK-sphere", StochasticMKM.v79_inaniwa2018(site_geometry="sphere")),
    ("OSMK2021", OSMK2021.v79_inaniwa2021()),
    ("OSMK2023", OSMK2023.cho_inaniwa2023()),
    ("IMK", IMKModel.du145_matsuya2025()),
    ("SMK-bethe", StochasticMKM.hsg_inaniwa2018(stopping_power=SimpleBetheStoppingPower())),
]


@pytest.mark.parametrize(("name", "model"), _ALL_MODELS, ids=[n for n, _ in _ALL_MODELS])
@pytest.mark.usefixtures("coarse_nodes")
def test_cache_on_off_bit_identical_across_ions(
    name: str, model: Any, monkeypatch: pytest.MonkeyPatch
) -> None:
    _assert_bit_identical(
        model.specific_energies(NEON), _original_route(model, NEON, _COARSE_NODES)
    )
    clear_microdosimetry_cache()
    for ion in (PROTON, HELIUM, CARBON_LOW, CARBON_HIGH, NEON):
        off = _uncached(model, ion, monkeypatch)
        cold = model.specific_energies(ion)
        warm = model.specific_energies(ion)
        _assert_bit_identical(cold, off)
        _assert_bit_identical(warm, off)
        assert warm is cold  # served from the cache
    assert microdosimetry_cache_info().misses == 5


@pytest.mark.usefixtures("coarse_nodes")
def test_shared_geometry_shares_entries_across_instances_and_classes() -> None:
    """The key is the route's inputs, not the model: oxygen fields do not split it."""
    a = OSMK2021.hsg_inaniwa2021()
    b = dataclasses.replace(a, alpha_S_Gy_inv=0.5, z_R_Gy=10.0, R_m=1.7, k_mmHg=5.0)
    smk = StochasticMKM(**{f.name: getattr(a, f.name) for f in dataclasses.fields(StochasticMKM)})
    za = a.specific_energies(CARBON_LOW)
    assert b.specific_energies(CARBON_LOW) is za
    assert smk.specific_energies(CARBON_LOW) is za
    info = microdosimetry_cache_info()
    assert (info.misses, info.hits) == (1, 2)


# --------------------------------------------------------------------------
# (2) field-perturbation completeness
# --------------------------------------------------------------------------
_NONE_REPLACEMENTS: dict[str, Any] = {
    "z0_Gy": 50.0,
    "alpha0_Gy_inv": 0.123,
    "beta0_Gy2": 0.0456,
    "yStar_D_ref_keV_um": 3.3,
    "reference_z_star_1D_Gy": 1.2,
    "radial_dose": KieferChatterjeeRadialDose(stopping_power=SimpleBetheStoppingPower()),
    "stopping_power": SimpleBetheStoppingPower(),
    "engine": AmorphousTrackEngine(
        radial_dose=KieferChatterjeeRadialDose(stopping_power=SimpleBetheStoppingPower()),
        site=SphericalSite(diameter_um=0.56),
        stopping_power=SimpleBetheStoppingPower(),
    ),
    "nucleus_engine": AmorphousTrackEngine(
        radial_dose=KieferChatterjeeRadialDose(stopping_power=SimpleBetheStoppingPower()),
        site=SphericalSite(diameter_um=16.2),
        stopping_power=SimpleBetheStoppingPower(),
    ),
}
_CHOICE_REPLACEMENTS: dict[str, Any] = {
    "site_geometry": "sphere",
    "domain_treatment": "sato_moments",
    "nucleus_treatment": "numerical",
}
# Fields a class refuses to take off their default (OSMK refuses the SMK axes).
_REFUSED: dict[type, set[str]] = {
    OSMK2021: {"domain_treatment", "nucleus_treatment"},
    OSMK2023: {"domain_treatment", "nucleus_treatment"},
}


def _perturbed(name: str, value: Any) -> Any:
    if name in _CHOICE_REPLACEMENTS:
        return _CHOICE_REPLACEMENTS[name]
    if value is None:
        return _NONE_REPLACEMENTS[name]
    if isinstance(value, bool):  # pragma: no cover -- no bool fields today
        return not value
    if isinstance(value, int):
        return value + 1
    if isinstance(value, float):
        return value * 1.1 + 0.01 if value != 0.0 else 0.05
    if name in _NONE_REPLACEMENTS:  # an object-valued field already set
        return _NONE_REPLACEMENTS[name]
    raise AssertionError(f"no perturbation rule for field {name!r} = {value!r}")


_PERTURBATION_BASES: list[tuple[str, Any]] = [
    ("MCF", MCFMKMModel()),
    ("mMKM", ModifiedMKM.hsg_mk_inaniwa2018()),
    ("SMK", StochasticMKM.hsg_inaniwa2018()),
    ("SMK-kase-z0", StochasticMKM.hsg_inaniwa2018(z0_Gy=None)),
    ("OSMK2021", OSMK2021.hsg_inaniwa2021()),
    ("OSMK2023", OSMK2023.hsg_inaniwa2023()),
    ("IMK", IMKModel.hsg_matsuya2025()),
]


@pytest.mark.parametrize(
    ("name", "base"), _PERTURBATION_BASES, ids=[n for n, _ in _PERTURBATION_BASES]
)
@pytest.mark.usefixtures("coarse_nodes")
def test_every_field_moves_the_key_or_leaves_the_result_alone(
    name: str, base: Any, monkeypatch: pytest.MonkeyPatch
) -> None:
    """For EVERY dataclass field: the key changes, the cache is bypassed, or the
    route's output is unchanged.

    The check is against `_original_route`, which reads the perturbed
    model's fields directly (not the key). With the base entry already cached, a
    perturbation the key fails to see is served the base value, and that differs
    from the independent reference whenever the field matters -- so an incomplete
    key fails here rather than passing by construction.
    """
    ion = CARBON_LOW
    assert base._microdosimetry_cacheable()
    base_key = base._microdosimetry_key()
    base_out = base.specific_energies(ion)  # warms the cache with the base entry
    _assert_bit_identical(base_out, _original_route(base, ion, _COARSE_NODES))
    refused: set[str] = set()
    outcome: dict[str, str] = {}
    for f in dataclasses.fields(base):
        new_value = _perturbed(f.name, getattr(base, f.name))
        try:
            changed = dataclasses.replace(base, **{f.name: new_value})
        except ValueError:
            refused.add(f.name)
            continue
        if not changed._microdosimetry_cacheable():
            outcome[f.name] = "bypass"
            continue
        # Served (possibly from the cache) == an independent pre-cache evaluation.
        _assert_bit_identical(
            changed.specific_energies(ion), _original_route(changed, ion, _COARSE_NODES)
        )
        outcome[f.name] = "key" if changed._microdosimetry_key() != base_key else "unchanged"
    assert refused == _REFUSED.get(type(base), set())
    # Sanity: the fields the route reads really do move the key (or bypass).
    for read in ("domain_radius_um", "nucleus_radius_um", "density_g_cm3", "site_geometry"):
        assert outcome[read] == "key", read
    # ... including the inputs of the saturation parameter z0_effective_Gy:
    # z0_Gy always, beta_ref when z0 is derived from the Kase relation.
    assert outcome["z0_Gy"] == "key"
    assert outcome["beta_ref"] == ("key" if base.z0_Gy is None else "unchanged")
    assert outcome["radial_dose"] == "key"
    assert outcome["stopping_power"] == "key"
    assert outcome["engine"] == "bypass"
    if "nucleus_engine" in outcome:
        assert outcome["nucleus_engine"] == "bypass"


@pytest.mark.parametrize(
    "constant", ["_IMPACT_RADIAL_NODES", "_IMPACT_CORE_NODES", "_IMPACT_PENUMBRA_NODES"]
)
def test_node_counts_are_part_of_the_key(constant: str, monkeypatch: pytest.MonkeyPatch) -> None:
    model = StochasticMKM.hsg_inaniwa2018()
    before = model._microdosimetry_key()
    monkeypatch.setattr(sm, constant, getattr(sm, constant) + 1)
    assert model._microdosimetry_key() != before


@pytest.mark.usefixtures("coarse_nodes")
def test_numpy_float32_inputs_stay_bit_identical_and_do_not_share() -> None:
    """Reduced-precision field values reach the route unconverted (as before the
    cache), and a float32 model never shares an entry with its float64 twin."""
    f32 = StochasticMKM.hsg_inaniwa2018(
        domain_radius_um=np.float32(0.28),
        nucleus_radius_um=np.float32(8.1),
        density_g_cm3=np.float32(1.06),
    )
    f64 = dataclasses.replace(
        f32,
        domain_radius_um=float(f32.domain_radius_um),
        nucleus_radius_um=float(f32.nucleus_radius_um),
        density_g_cm3=float(f32.density_g_cm3),
    )
    for model in (f32, f64, f32):
        _assert_bit_identical(
            model.specific_energies(CARBON_LOW),
            _original_route(model, CARBON_LOW, _COARSE_NODES),
        )
    assert f32._microdosimetry_key() != f64._microdosimetry_key()
    info = microdosimetry_cache_info()
    assert (info.misses, info.hits) == (2, 1)


def _assert_twins_never_share(
    a: tuple[Any, Ion], b: tuple[Any, Ion], monkeypatch: pytest.MonkeyPatch
) -> None:
    """Two inputs that compare and hash equal yet evaluate differently (a numpy
    ``float32`` and the ``float`` of the same value) must keep separate entries:
    in either call order, cold and warm, each is served its own cache-off value."""
    refs = [_uncached(model, ion, monkeypatch) for model, ion in (a, b)]
    assert refs[0] != refs[1]  # the twins really do evaluate differently
    for first, second in ((0, 1), (1, 0)):
        clear_microdosimetry_cache()
        for i in (first, second, first, second):
            model, ion = (a, b)[i]
            _assert_bit_identical(model.specific_energies(ion), refs[i])
        info = microdosimetry_cache_info()
        assert (info.misses, info.hits) == (2, 2)


@pytest.mark.usefixtures("coarse_nodes")
def test_float32_ion_energy_never_shares_an_entry(monkeypatch: pytest.MonkeyPatch) -> None:
    """``Ion(6, 12, np.float32(5.0)) == Ion(6, 12, 5.0)`` but the route sees the
    float32 (e.g. ``energy_MeV_u ** delta`` in the Kiefer-Chatterjee r_max)."""
    model = StochasticMKM.hsg_inaniwa2018()
    f32_ion = Ion(Z=6, A=12, energy_MeV_u=np.float32(5.0))  # type: ignore[arg-type]
    assert f32_ion == CARBON_LOW and hash(f32_ion) == hash(CARBON_LOW)
    _assert_twins_never_share((model, CARBON_LOW), (model, f32_ion), monkeypatch)


@pytest.mark.usefixtures("coarse_nodes")
def test_float32_radial_dose_field_never_shares_an_entry(monkeypatch: pytest.MonkeyPatch) -> None:
    """The allowlisted ``KieferChatterjeeRadialDose`` hashes by value, so its float
    fields need the same type guard as the model's own."""
    sp = SimpleBetheStoppingPower()
    rd64 = KieferChatterjeeRadialDose(stopping_power=sp, rc0_nm=40.0)
    rd32 = KieferChatterjeeRadialDose(stopping_power=sp, rc0_nm=np.float32(40.0))  # type: ignore[arg-type]
    assert rd32 == rd64 and hash(rd32) == hash(rd64)
    m64 = StochasticMKM.hsg_inaniwa2018(radial_dose=rd64)
    m32 = StochasticMKM.hsg_inaniwa2018(radial_dose=rd32)
    assert m32._microdosimetry_key() != m64._microdosimetry_key()
    _assert_twins_never_share((m64, CARBON_LOW), (m32, CARBON_LOW), monkeypatch)


@pytest.mark.usefixtures("coarse_nodes")
def test_float32_z0_never_shares_an_entry(monkeypatch: pytest.MonkeyPatch) -> None:
    """A float32 ``z0_Gy`` equal in value to a float one still gets its own key."""
    m32 = StochasticMKM.hsg_inaniwa2018(z0_Gy=np.float32(66.0))  # type: ignore[arg-type]
    m64 = StochasticMKM.hsg_inaniwa2018(z0_Gy=66.0)
    assert m32.z0_effective_Gy == m64.z0_effective_Gy
    assert m32._microdosimetry_key() != m64._microdosimetry_key()
    _assert_twins_never_share((m64, CARBON_LOW), (m32, CARBON_LOW), monkeypatch)


@pytest.mark.usefixtures("coarse_nodes")
def test_non_finite_moments_are_not_cached() -> None:
    """``z0_Gy=0`` makes ``z*_1D`` NaN with RuntimeWarnings. Such a result is not
    cached, so every call warns again (and raises again under ``-W error``)."""
    model = StochasticMKM.hsg_inaniwa2018(z0_Gy=0.0)
    for _ in range(2):
        with pytest.warns(RuntimeWarning):
            z = model.specific_energies(CARBON_LOW)
        assert math.isnan(z.z_star_1D_Gy)
    nan_z0 = StochasticMKM.hsg_inaniwa2018(z0_Gy=math.nan)
    assert math.isnan(nan_z0.specific_energies(CARBON_LOW).z_star_1D_Gy)
    info = microdosimetry_cache_info()
    assert (info.hits, info.currsize) == (0, 0)


@pytest.mark.usefixtures("coarse_nodes")
def test_patched_library_defaults_are_part_of_the_key(monkeypatch: pytest.MonkeyPatch) -> None:
    """Monkeypatching the module's default stopping power is never served stale."""
    model = StochasticMKM.hsg_inaniwa2018()
    default_out = model.specific_energies(CARBON_LOW)
    bethe = SimpleBetheStoppingPower()
    monkeypatch.setattr(sm, "default_stopping_power", lambda density_g_cm3: bethe)
    patched = model.specific_energies(CARBON_LOW)
    assert patched != default_out
    _assert_bit_identical(patched, _uncached(model, CARBON_LOW, monkeypatch))


def test_cache_info_is_a_public_named_tuple() -> None:
    info = microdosimetry_cache_info()
    assert isinstance(info, sm.MicrodosimetryCacheInfo)
    assert info._fields == ("hits", "misses", "maxsize", "currsize")


# --------------------------------------------------------------------------
# (3) bypass cases
# --------------------------------------------------------------------------
class _CountingStoppingPower(SimpleBetheStoppingPower):
    """A stateful subclass of an allowlisted class: must NOT be cached."""

    def __init__(self) -> None:
        super().__init__()
        self.calls = 0

    def let_keV_um(self, ion: Ion) -> float:
        self.calls += 1
        return SimpleBetheStoppingPower.let_keV_um(self, ion)


@dataclasses.dataclass(frozen=True)
class _ForeignRadialDose:
    """A hashable radial dose that is not on the allowlist (a value-hashing wrapper).

    Hashable on purpose: only the allowlist, not the ``TypeError`` guard, may
    keep it out of the cache.
    """

    inner: KieferChatterjeeRadialDose

    def r_max_um(self, ion: Ion) -> float:
        return self.inner.r_max_um(ion)

    def r_min_um(self, ion: Ion) -> float:
        return self.inner.r_min_um(ion)

    def local_dose_Gy(self, r_um: float, ion: Ion) -> float:
        return self.inner.local_dose_Gy(r_um, ion)


class _CountingRadialDose(KieferChatterjeeRadialDose):
    """A stateful subclass of the allowlisted radial dose: must NOT be cached.

    It inherits the frozen dataclass ``__eq__``/``__hash__``, so it hashes fine;
    only the exact-type allowlist check keeps it out of the cache.
    """

    def local_dose_Gy(self, r_um: float, ion: Ion) -> float:
        object.__setattr__(self, "calls", getattr(self, "calls", 0) + 1)
        return KieferChatterjeeRadialDose.local_dose_Gy(self, r_um, ion)


class _UnhashableFloat(float):
    """A float that refuses to hash (a stand-in for any unhashable field value)."""

    __hash__ = None  # type: ignore[assignment]


def _kc(sp: Any) -> KieferChatterjeeRadialDose:
    return KieferChatterjeeRadialDose(stopping_power=sp)


def _bypass_cases() -> list[tuple[str, Any]]:
    engine = _NONE_REPLACEMENTS["engine"]
    nucleus_engine = _NONE_REPLACEMENTS["nucleus_engine"]
    return [
        ("engine", StochasticMKM.hsg_inaniwa2018(engine=engine, nucleus_engine=nucleus_engine)),
        ("nucleus_engine_only", StochasticMKM.hsg_inaniwa2018(nucleus_engine=nucleus_engine)),
        (
            "ns1970_stopping",
            StochasticMKM.hsg_inaniwa2018(stopping_power=NorthcliffeSchillingWithBetheTail()),
        ),
        (
            "kc_with_ns1970",
            StochasticMKM.hsg_inaniwa2018(radial_dose=_kc(NorthcliffeSchillingWithBetheTail())),
        ),
        (
            "stateful_subclass",
            StochasticMKM.hsg_inaniwa2018(stopping_power=_CountingStoppingPower()),
        ),
        (
            "non_allowlisted_radial_dose",
            StochasticMKM.hsg_inaniwa2018(
                radial_dose=_ForeignRadialDose(_kc(SimpleBetheStoppingPower()))
            ),
        ),
        (
            "stateful_radial_dose_subclass",
            StochasticMKM.hsg_inaniwa2018(
                radial_dose=_CountingRadialDose(stopping_power=SimpleBetheStoppingPower())
            ),
        ),
        (
            # allowlisted type, unhashable field: reaches hash() and must bypass
            "unhashable_allowlisted_radial_dose",
            StochasticMKM.hsg_inaniwa2018(
                radial_dose=KieferChatterjeeRadialDose(
                    stopping_power=SimpleBetheStoppingPower(),
                    rc0_nm=_UnhashableFloat(0.4),
                )
            ),
        ),
    ]


@pytest.mark.parametrize(("name", "model"), _bypass_cases(), ids=[n for n, _ in _bypass_cases()])
@pytest.mark.usefixtures("coarse_nodes")
def test_bypass_cases_never_touch_the_cache(name: str, model: Any) -> None:
    if name == "unhashable_allowlisted_radial_dose":
        assert model._microdosimetry_cacheable()  # the hash, not the allowlist, bypasses
        with pytest.raises(TypeError):
            hash(model._microdosimetry_key())
    elif name in ("non_allowlisted_radial_dose", "stateful_radial_dose_subclass"):
        hash(model._microdosimetry_key())  # hashable: the allowlist alone must bypass
    first = model.specific_energies(CARBON_LOW)
    second = model.specific_energies(CARBON_LOW)
    _assert_bit_identical(first, second)
    assert first is not second  # recomputed, not served
    info = microdosimetry_cache_info()
    assert (info.hits, info.misses, info.currsize) == (0, 0, 0)


@pytest.mark.usefixtures("coarse_nodes")
def test_stateful_sources_keep_counting() -> None:
    """The tally of a stateful source must see every call, cache or not."""
    ns = NorthcliffeSchillingWithBetheTail()
    counting = _CountingStoppingPower()
    tallies: list[tuple[Any, Callable[[], int]]] = [
        (ns, lambda: ns.table_hits + len(ns.fallback_energies)),
        (counting, lambda: counting.calls),
    ]
    for sp, tally in tallies:
        model = StochasticMKM.hsg_inaniwa2018(stopping_power=sp)
        model.specific_energies(CARBON_LOW)
        after_first = tally()
        assert after_first > 0
        model.specific_energies(CARBON_LOW)
        assert tally() > after_first
    rd = _CountingRadialDose(stopping_power=SimpleBetheStoppingPower())
    model = StochasticMKM.hsg_inaniwa2018(radial_dose=rd)
    for _ in range(2):
        before = getattr(rd, "calls", 0)
        model.specific_energies(CARBON_LOW)
        assert getattr(rd, "calls", 0) > before


@pytest.mark.parametrize(
    "sp",
    [
        None,
        DefaultStoppingPower(),
        SimpleBetheStoppingPower(),
        TabularStoppingPower.from_arrays({6: ([1.0, 1000.0], [100.0, 10.0])}),
        BetheOnly(),
    ],
    ids=["none", "default", "simple_bethe", "tabular", "bethe_only"],
)
@pytest.mark.usefixtures("coarse_nodes")
def test_stateless_sources_are_cacheable(sp: Any) -> None:
    """Allowlisted, and actually served: the second call of each model is a hit."""
    assert StochasticMKM.hsg_inaniwa2018(radial_dose=_kc(sp))._microdosimetry_cacheable()
    models = [StochasticMKM.hsg_inaniwa2018(stopping_power=sp)]
    if sp is not None:  # a Kiefer-Chatterjee without a stopping power cannot evaluate
        models.append(StochasticMKM.hsg_inaniwa2018(radial_dose=_kc(sp)))
    for model in models:
        assert model._microdosimetry_cacheable()
        clear_microdosimetry_cache()
        first = model.specific_energies(CARBON_LOW)
        assert model.specific_energies(CARBON_LOW) is first
        info = microdosimetry_cache_info()
        assert (info.misses, info.hits) == (1, 1)


@pytest.mark.usefixtures("coarse_nodes")
def test_distinct_stopping_power_instances_miss() -> None:
    """Stopping powers hash by identity: a second instance is a (conservative) miss."""
    a = StochasticMKM.hsg_inaniwa2018(stopping_power=SimpleBetheStoppingPower())
    b = StochasticMKM.hsg_inaniwa2018(stopping_power=SimpleBetheStoppingPower())
    a.specific_energies(CARBON_LOW)
    b.specific_energies(CARBON_LOW)
    assert microdosimetry_cache_info().misses == 2


# --------------------------------------------------------------------------
# (4) OSMK2023's replace() copies hit
# --------------------------------------------------------------------------
@pytest.mark.usefixtures("coarse_nodes")
def test_osmk2023_hypoxic_replace_copies_hit() -> None:
    model = OSMK2023.hsg_inaniwa2023()
    first = model.specific_energies_at(CARBON_LOW, 0.0)
    assert microdosimetry_cache_info().misses == 1
    # the same call builds a fresh replace() copy and must hit
    assert model.specific_energies_at(CARBON_LOW, 0.0) is first
    # an oxygen-parameter variant sharing the hypoxic geometry hits as well
    variant = dataclasses.replace(model, alpha_S_Gy_inv=0.2, k_mmHg=model.k_mmHg)
    assert variant.specific_energies_at(CARBON_LOW, 0.0) is first
    info = microdosimetry_cache_info()
    assert (info.misses, info.hits) == (1, 2)
    # a different pO2 moves the hypoxic geometry, so it is a new entry
    model.specific_energies_at(CARBON_LOW, 3.0)
    assert microdosimetry_cache_info().misses == 2


# --------------------------------------------------------------------------
# (5) mutation safety + clearing
# --------------------------------------------------------------------------
@pytest.mark.usefixtures("coarse_nodes")
def test_cached_values_cannot_be_mutated() -> None:
    model = StochasticMKM.hsg_inaniwa2018()
    z = model.specific_energies(CARBON_LOW)
    with pytest.raises(dataclasses.FrozenInstanceError):
        z.z_1D_Gy = 0.0  # type: ignore[misc]
    for f in dataclasses.fields(z):  # immutable leaves only: no arrays to mutate
        assert getattr(z, f.name) is None or type(getattr(z, f.name)) is float
    assert model.specific_energies(CARBON_LOW) is z
    assert z.z_1D_Gy != 0.0


@pytest.mark.usefixtures("coarse_nodes")
def test_clear_empties_the_cache() -> None:
    model = StochasticMKM.hsg_inaniwa2018()
    first = model.specific_energies(CARBON_LOW)
    assert microdosimetry_cache_info().currsize == 1
    clear_microdosimetry_cache()
    info = microdosimetry_cache_info()
    assert (info.hits, info.misses, info.currsize) == (0, 0, 0)
    again = model.specific_energies(CARBON_LOW)
    assert again is not first
    _assert_bit_identical(again, first)


def test_cache_maxsize_is_4096() -> None:
    assert microdosimetry_cache_info().maxsize == 4096
