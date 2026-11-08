"""describe_stopping_power() and energy_for_let_MeV_u(branch="bracket").

``describe_stopping_power`` gives the label of whichever source is in use, instead of a
hard-coded string such as "analytic Bethe (library default)"; ``branch="bracket"`` is
the plain ``brentq`` LET inversion. The tests below pin ``"bracket"`` against the
equivalent inline ``brentq`` calls (reproduced verbatim, exact float equality) and pin
the ``"high"`` / ``"low"`` branches against a frozen reference copy of their code.
"""

from __future__ import annotations

import importlib
import inspect
import math
import pkgutil
from collections.abc import Callable

import numpy as np
import pytest
from scipy.optimize import brentq

import pyradbio.track
from pyradbio.core.radiation import Ion
from pyradbio.models.mkm.pymkm_backend import PyMKMStoppingPower
from pyradbio.track.northcliffe_schilling import BetheOnly, NorthcliffeSchillingWithBetheTail
from pyradbio.track.stopping import base as stopping_base
from pyradbio.track.stopping.bethe import Bethe
from pyradbio.track.stopping.hubert import HubertAlphaScaling
from pyradbio.track.stopping.icru import ICRU
from pyradbio.track.stopping.icru90_procedure import ICRU90Procedure
from pyradbio.track.stopping.mstar import MSTAR
from pyradbio.track.stopping.xstar import XSTAR
from pyradbio.track.stopping_power import (
    STOPPING_POWER_DESCRIPTIONS,
    DefaultStoppingPower,
    LibdedxStoppingPower,
    MassStoppingPowerSource,
    SimpleBetheStoppingPower,
    StoppingPower,
    TabularStoppingPower,
    default_stopping_power,
    describe_stopping_power,
    energy_for_let_MeV_u,
)
from tests.conftest import requires_pymkm

DEFAULT_TEXT = "ICRU 90 procedure (Z 1-18) + Hubert 1990 + Bethe (library default)"

# ---- describe_stopping_power -------------------------------------------------------


def test_library_default_is_described_as_the_icru90_hubert_bethe_chain() -> None:
    assert describe_stopping_power(default_stopping_power()) == DEFAULT_TEXT
    assert describe_stopping_power(DefaultStoppingPower(density_g_cm3=1.1)) == DEFAULT_TEXT


def test_simple_bethe_is_described_as_analytic_bethe() -> None:
    assert describe_stopping_power(SimpleBetheStoppingPower()) == "analytic Bethe"


def test_bethe_only_reports_its_formula_not_its_cli_name() -> None:
    # BetheOnly's class attribute ``name = "bethe"`` is a CLI/checkpoint key that collides
    # with pyradbio.track.stopping.bethe.Bethe's; its settings() (inherited) say what it
    # computes, so settings()["name"] is read first.
    assert BetheOnly.name == Bethe.name == "bethe"
    assert describe_stopping_power(BetheOnly()) == "analytic Bethe"
    assert describe_stopping_power(Bethe()) != describe_stopping_power(BetheOnly())


@pytest.mark.parametrize(
    ("build", "text"),
    [
        (
            NorthcliffeSchillingWithBetheTail,
            "Northcliffe & Schilling 1970 table + analytic Bethe above 12 MeV/u",
        ),
        (Bethe, "Bethe formula with switchable corrections"),
        (ICRU90Procedure, "ICRU 90 procedure"),
        (HubertAlphaScaling, "Hubert, Bimbot & Gauvin 1990 alpha scaling"),
        (ICRU, "ICRU 49/73/90 tables (edition 2016)"),
        (XSTAR, "NIST PSTAR/ASTAR + MSTAR 3.12 (XSTAR)"),
    ],
)
def test_every_constructible_source_has_its_own_text(
    build: Callable[[], StoppingPower], text: str
) -> None:
    assert describe_stopping_power(build()) == text


def _track_stopping_power_classes() -> dict[type, object]:
    """Every stopping-power class of pyradbio.track, with its class-level ``name`` (or None).

    Imports every submodule first, so a source in a module nothing else imports is
    still found: a MassStoppingPowerModel subclass, or any class with ``let_keV_um``.
    """
    found: dict[type, object] = {}
    for info in pkgutil.walk_packages(pyradbio.track.__path__, "pyradbio.track."):
        module = importlib.import_module(info.name)
        for _, cls in inspect.getmembers(module, inspect.isclass):
            if cls.__module__ != module.__name__:
                continue
            if issubclass(cls, stopping_base.MassStoppingPowerModel) or callable(
                getattr(cls, "let_keV_um", None)
            ):
                found[cls] = cls.__dict__.get("name")
    return found


def test_every_named_stopping_power_class_is_mapped() -> None:
    # MSTAR needs a local msdbs1.d (not shipped), so the class name stands in for an
    # instance; the abstract bases and the unnamed sources (tables, columns, adapters)
    # carry no name of their own and fall back to their type name.
    classes = _track_stopping_power_classes()
    named = {cls: name for cls, name in classes.items() if isinstance(name, str)}
    assert set(named) >= {
        Bethe,
        BetheOnly,
        HubertAlphaScaling,
        ICRU,
        ICRU90Procedure,
        MSTAR,
        NorthcliffeSchillingWithBetheTail,
        XSTAR,
    }
    for cls, name in named.items():
        assert name in STOPPING_POWER_DESCRIPTIONS, cls
    assert STOPPING_POWER_DESCRIPTIONS["mstar"] == "MSTAR 3.12 (Paul & Schinner)"


def test_icru_text_names_its_edition() -> None:
    assert describe_stopping_power(ICRU(edition="2005")) == "ICRU 49/73/90 tables (edition 2005)"
    assert describe_stopping_power(ICRU(edition="2009")) == "ICRU 49/73/90 tables (edition 2009)"


def test_the_description_map_is_read_only() -> None:
    with pytest.raises(TypeError):
        STOPPING_POWER_DESCRIPTIONS["default"] = "x"  # type: ignore[index]


def test_pymkm_tables_are_described_without_constructing_them() -> None:
    # PyMKMStoppingPower has neither a name nor settings(); its type name is the key.
    # object.__new__ skips __init__, so no pyMKM import is needed.
    sp = object.__new__(PyMKMStoppingPower)
    assert describe_stopping_power(sp) == "pyMKM bundled table (validation reference)"
    # a constructed one stores its table source, which the text then names
    sp.source = "geant4_11_3_0"
    assert describe_stopping_power(sp) == (
        "pyMKM bundled table (validation reference): geant4_11_3_0"
    )


@requires_pymkm
def test_a_real_pymkm_table_is_described() -> None:
    sp = PyMKMStoppingPower("fluka_2020_0")
    assert describe_stopping_power(sp) == (
        "pyMKM bundled table (validation reference): fluka_2020_0"
    )


def test_trip98_dedx_adapter_is_described_by_its_type_name() -> None:
    # The examples' TRiP98 FLUKA dE/dx adapter (lem1/lem4_trip98_rbe_reproduction.py)
    # has no name either; a stand-in with the same class name exercises the key.
    TripDedxStoppingPower = type("TripDedxStoppingPower", (), {})
    assert describe_stopping_power(TripDedxStoppingPower()) == "TRiP98 FLUKA dE/dx table"


def test_trip98_dedx_adapter_is_described_by_its_design_name() -> None:
    # The registered key for when the adapter gains an
    # NS1970-style ``name`` attribute; any class name then works.
    Adapter = type("Adapter", (), {"name": "trip98-fluka-dedx"})
    assert describe_stopping_power(Adapter()) == "TRiP98 FLUKA dE/dx table"


def test_unnamed_sources_fall_back_to_their_type_name() -> None:
    table = TabularStoppingPower.from_arrays({6: ([1.0, 10.0], [100.0, 20.0])})
    assert describe_stopping_power(table) == "TabularStoppingPower"
    source = MassStoppingPowerSource(lambda z, a, e: 100.0)
    assert describe_stopping_power(source) == "MassStoppingPowerSource"
    # LibdedxStoppingPower needs the optional libdedx package to construct; it reports
    # no name either, so a bare instance (object.__new__ skips __init__) suffices.
    libdedx = object.__new__(LibdedxStoppingPower)
    assert describe_stopping_power(libdedx) == "LibdedxStoppingPower"


def test_unknown_or_odd_objects_fall_back_to_their_type_name() -> None:
    class Named:
        name = "something-unmapped"

        def let_keV_um(self, ion: Ion) -> float:
            return 1.0

    class OddSettings:
        def settings(self) -> list[str]:
            return ["not", "a", "mapping"]

    class NonStringName:
        name = 42

    class UnhashableSettingsName:
        def settings(self) -> dict[str, object]:
            return {"name": ["x"]}

    class SettingsNeedsArguments:
        name = "simple_bethe"

        def settings(self, verbose: bool) -> dict[str, object]:
            return {"name": "default"}

    class SettingsRaises:
        def settings(self) -> dict[str, object]:
            raise ValueError("no settings for this one")

    assert describe_stopping_power(Named()) == "Named"
    assert describe_stopping_power(OddSettings()) == "OddSettings"
    assert describe_stopping_power(NonStringName()) == "NonStringName"
    assert describe_stopping_power(UnhashableSettingsName()) == "UnhashableSettingsName"
    # a settings() that cannot be called bare is skipped; the ``name`` attribute still counts
    assert describe_stopping_power(SettingsNeedsArguments()) == "analytic Bethe"
    assert describe_stopping_power(SettingsRaises()) == "SettingsRaises"
    assert describe_stopping_power(object()) == "object"


def test_settings_name_wins_over_an_unmapped_attribute_name() -> None:
    class Both:
        name = "unmapped"

        def settings(self) -> dict[str, object]:
            return {"name": "simple_bethe"}

    class AttributeOnly:
        name = "ns1970+bethe"

        def settings(self) -> dict[str, object]:
            return {"name": "unmapped"}

    assert describe_stopping_power(Both()) == "analytic Bethe"
    assert describe_stopping_power(AttributeOnly()) == (
        "Northcliffe & Schilling 1970 table + analytic Bethe above 12 MeV/u"
    )


# ---- energy_for_let_MeV_u(branch="bracket") vs the equivalent inline brentq calls ---


def _same(inline: Callable[[], float], helper: Callable[[], float]) -> None:
    """Exact float equality, or both raise ValueError."""
    try:
        expected = inline()
    except ValueError:
        with pytest.raises(ValueError):
            helper()
        return
    got = helper()
    assert got == expected
    assert type(got) is float


@pytest.mark.parametrize("let", [13.0, 77.0])
def test_bracket_equals_lem3_example_inline_inversion(let: float) -> None:
    # the inline inversion of the LEM III velocity-dependent track-core example
    sp = default_stopping_power()

    def inline() -> float:
        return float(brentq(lambda e: sp.let_keV_um(Ion(6, 12, e)) - let, 1.5, 500.0))

    _same(
        inline,
        lambda: energy_for_let_MeV_u(
            sp, 6, 12, let, branch="bracket", energy_bounds_MeV_u=(1.5, 500.0)
        ),
    )


_LEM4_RECON_POINTS = (
    [(1, 1, float(x)) for x in np.geomspace(4.0, 40.0, 40)[::3]]  # let_p, :464
    + [(2, 4, float(x)) for x in np.geomspace(15.0, 110.0, 30)[::3]]  # let_he, :465
    + [(6, 12, float(x)) for x in np.geomspace(6.0, 900.0, 60)[::4]]  # LET_GRID_C, :515
    + [(2, 4, float(x)) for x in np.geomspace(4.0, 130.0, 60)[::4]]  # LET_GRID_HE, :516
    + [(6, 12, 15.0), (6, 12, 100.0), (6, 12, 900.0)]  # MC_LETS + one beyond the bracket
)


def test_bracket_equals_lem4_reconstruction_example_inline_inversion() -> None:
    # the inline inversion (sp and ion_for_let) of the LEM IV reconstruction example
    sp = default_stopping_power()
    n_unreachable = 0
    for z, a, let in _LEM4_RECON_POINTS:

        def inline(z: int = z, a: int = a, let: float = let) -> float:
            e = brentq(lambda x: sp.let_keV_um(Ion(z, a, x)) - let, 0.4, 600.0)
            return float(e)

        def helper(z: int = z, a: int = a, let: float = let) -> float:
            return energy_for_let_MeV_u(
                sp, z, a, let, branch="bracket", energy_bounds_MeV_u=(0.4, 600.0)
            )

        try:
            inline()
        except ValueError:
            n_unreachable += 1
        _same(inline, helper)
    assert n_unreachable >= 1  # the None path of ion_for_let is exercised


@pytest.mark.parametrize("let", [13.0, 46.0, 80.0, 100.0, 10.0, 500.0])
def test_bracket_equals_imk_example_inline_inversion(let: float) -> None:
    # the inline inversion (SP and carbon_at_let) of the IMK microdosimetry-routes example
    SP = SimpleBetheStoppingPower()

    def inline() -> float:
        energy = brentq(
            lambda e: SP.let_keV_um(Ion(Z=6, A=12, energy_MeV_u=e)) - let,
            0.6,
            600.0,
            xtol=1e-8,
        )
        return float(energy)

    _same(
        inline,
        lambda: energy_for_let_MeV_u(
            SP, 6, 12, let, branch="bracket", energy_bounds_MeV_u=(0.6, 600.0), xtol=1e-8
        ),
    )


@pytest.mark.parametrize("let", [13.0, 77.0])
def test_bracket_equals_test_lem3_inline_inversion(let: float) -> None:
    # tests/models/lem/test_lem3.py
    _SP = SimpleBetheStoppingPower()

    def inline() -> float:
        return float(brentq(lambda e: _SP.let_keV_um(Ion(6, 12, e)) - let, 1.5, 500.0))

    _same(
        inline,
        lambda: energy_for_let_MeV_u(
            _SP, 6, 12, let, branch="bracket", energy_bounds_MeV_u=(1.5, 500.0)
        ),
    )


@pytest.mark.parametrize(
    "let", [16.0, 20.0, 30.0, 45.0, 65.0, 90.0, 120.0, 160.0, 220.0, 320.0, 420.0]
)
def test_bracket_equals_test_lem4_carbon_inline_inversion(let: float) -> None:
    # tests/models/lem/test_lem4.py (_SP, _CARBON_LETS)
    _SP = SimpleBetheStoppingPower()

    def inline() -> float:
        return float(brentq(lambda e: _SP.let_keV_um(Ion(6, 12, e)) - let, 1.6, 600.0))

    _same(
        inline,
        lambda: energy_for_let_MeV_u(
            _SP, 6, 12, let, branch="bracket", energy_bounds_MeV_u=(1.6, 600.0)
        ),
    )


@pytest.mark.parametrize(
    ("z", "a", "let"),
    [(1, 1, 5.0), (1, 1, 30.0), (2, 4, 20.0), (2, 4, 100.0), (6, 12, 50.0), (6, 12, 2000.0)],
)
def test_bracket_equals_test_lem4_ion_for_let_inline_inversion(z: int, a: int, let: float) -> None:
    # tests/models/lem/test_lem4.py (_ion_for_let; ValueError -> None)
    _SP = SimpleBetheStoppingPower()

    def inline() -> float:
        e = brentq(lambda x: _SP.let_keV_um(Ion(z, a, x)) - let, 0.4, 600.0)
        return float(e)

    _same(
        inline,
        lambda: energy_for_let_MeV_u(
            _SP, z, a, let, branch="bracket", energy_bounds_MeV_u=(0.4, 600.0)
        ),
    )


@pytest.mark.parametrize("let_keV_um", [13.0, 46.0, 80.0, 100.0])
def test_bracket_equals_test_matsuya2025_inline_inversion(let_keV_um: float) -> None:
    # tests/benchmark/test_matsuya2025.py (_carbon_at)
    stopping = SimpleBetheStoppingPower()

    def inline() -> float:
        energy = brentq(
            lambda e: stopping.let_keV_um(Ion(Z=6, A=12, energy_MeV_u=e)) - let_keV_um,
            0.6,
            600.0,
            xtol=1e-8,
        )
        return float(energy)

    _same(
        inline,
        lambda: energy_for_let_MeV_u(
            stopping,
            6,
            12,
            let_keV_um,
            branch="bracket",
            energy_bounds_MeV_u=(0.6, 600.0),
            xtol=1e-8,
        ),
    )


def test_bracket_forwards_rtol_and_uses_scipy_defaults_otherwise() -> None:
    sp = SimpleBetheStoppingPower()

    def f(e: float) -> float:
        return sp.let_keV_um(Ion(6, 12, e)) - 50.0

    lo, hi = 1.0, 600.0

    def bracket(xtol: float | None = None, rtol: float | None = None) -> float:
        return energy_for_let_MeV_u(
            sp,
            6,
            12,
            50.0,
            branch="bracket",
            energy_bounds_MeV_u=(lo, hi),
            xtol=xtol,
            rtol=rtol,
        )

    assert bracket(rtol=1e-6) == brentq(f, lo, hi, rtol=1e-6)
    assert bracket(xtol=1e-3) == brentq(f, lo, hi, xtol=1e-3)
    assert bracket(xtol=1e-3, rtol=1e-6) == brentq(f, lo, hi, xtol=1e-3, rtol=1e-6)
    # None means scipy's defaults, not the high/low branches' 1e-10 / 1e-12
    assert bracket() == brentq(f, lo, hi)
    # a loose xtol visibly stops early, so the value is really forwarded
    assert bracket(xtol=1.0) != bracket()


def test_bracket_raises_without_a_sign_change() -> None:
    sp = SimpleBetheStoppingPower()
    # carbon LET at 1.5 / 500 MeV/u is ~ 400 / ~ 11 keV/um, so 5 keV/um is outside it
    with pytest.raises(ValueError, match="no sign change"):
        energy_for_let_MeV_u(sp, 6, 12, 5.0, branch="bracket", energy_bounds_MeV_u=(1.5, 500.0))


def test_bracket_does_not_prescan_or_reevaluate_the_endpoints() -> None:
    # The same number of LET evaluations as the inline brentq call: no 400-point
    # pre-scan, and the endpoint values are reused rather than recomputed (the NS1970
    # hybrid tallies every lookup, so an extra call would show in its counters).
    calls: list[float] = []
    bethe = SimpleBetheStoppingPower()

    class Counting:
        def let_keV_um(self, ion: Ion) -> float:
            calls.append(ion.energy_MeV_u)
            return bethe.let_keV_um(ion)

    counting = Counting()
    brentq(lambda e: counting.let_keV_um(Ion(6, 12, e)) - 50.0, 1.0, 600.0)
    inline_calls = list(calls)
    calls.clear()
    energy_for_let_MeV_u(counting, 6, 12, 50.0, branch="bracket", energy_bounds_MeV_u=(1.0, 600.0))
    assert calls == inline_calls


def test_bracket_endpoint_root_is_returned_like_brentq() -> None:
    sp = SimpleBetheStoppingPower()
    lo, hi = 10.0, 100.0
    let_lo = sp.let_keV_um(Ion(6, 12, lo))
    assert energy_for_let_MeV_u(
        sp, 6, 12, let_lo, branch="bracket", energy_bounds_MeV_u=(lo, hi)
    ) == brentq(lambda e: sp.let_keV_um(Ion(6, 12, e)) - let_lo, lo, hi)


def test_tolerances_are_rejected_on_the_scanning_branches() -> None:
    sp = SimpleBetheStoppingPower()
    with pytest.raises(ValueError, match="bracket"):
        energy_for_let_MeV_u(sp, 6, 12, 50.0, xtol=1e-8)
    with pytest.raises(ValueError, match="bracket"):
        energy_for_let_MeV_u(sp, 6, 12, 50.0, branch="low", rtol=1e-8)


def test_unknown_branch_names_all_three() -> None:
    with pytest.raises(ValueError, match="'bracket'"):
        energy_for_let_MeV_u(SimpleBetheStoppingPower(), 6, 12, 50.0, branch="middle")


def test_bracket_keeps_the_shared_input_validation() -> None:
    sp = SimpleBetheStoppingPower()
    with pytest.raises(ValueError, match="let_keV_um must be > 0"):
        energy_for_let_MeV_u(sp, 6, 12, 0.0, branch="bracket")
    with pytest.raises(ValueError, match="0 < lo < hi"):
        energy_for_let_MeV_u(sp, 6, 12, 50.0, branch="bracket", energy_bounds_MeV_u=(5.0, 1.0))


# ---- "high" / "low" are unchanged ---------------------------------------------------


def _pre_h3_energy_for_let(
    stopping_power: StoppingPower,
    Z: int,
    A: int,
    let_keV_um: float,
    *,
    branch: str = "high",
    energy_bounds_MeV_u: tuple[float, float] = (0.1, 1000.0),
) -> float:
    """Frozen reference copy of the "high" / "low" energy_for_let_MeV_u algorithm."""
    lo, hi = energy_bounds_MeV_u
    if not 0.0 < lo < hi:
        raise ValueError(f"energy_bounds_MeV_u must be 0 < lo < hi, got {energy_bounds_MeV_u!r}")
    if let_keV_um <= 0.0:
        raise ValueError(f"let_keV_um must be > 0, got {let_keV_um!r}")

    grid = np.logspace(math.log10(lo), math.log10(hi), 400)
    lets = np.array([stopping_power.let_keV_um(Ion(Z, A, float(e))) for e in grid])
    peak = int(np.argmax(lets))
    if let_keV_um > lets[peak]:
        raise ValueError(
            f"LET {let_keV_um} keV/um exceeds the peak LET {lets[peak]:.1f} for Z={Z} A={A}"
        )

    def excess(energy: float) -> float:
        return stopping_power.let_keV_um(Ion(Z, A, energy)) - let_keV_um

    a, b = (grid[peak], grid[-1]) if branch == "high" else (grid[0], grid[peak])
    if branch not in ("high", "low"):
        raise ValueError(f"branch must be 'high' or 'low', got {branch!r}")
    if excess(a) * excess(b) > 0.0:
        raise ValueError(f"LET {let_keV_um} keV/um is not bracketed on the {branch} branch")
    return float(brentq(excess, a, b, xtol=1e-10, rtol=1e-12))


# (source, Z, A, LET, branch, value printed by the reference algorithm on the development machine).
_PRE_H3_VALUES = [
    ("bethe", 6, 12, 13.0, "high", 282.3854986099179),
    ("bethe", 6, 12, 77.0, "high", 25.197112682464333),
    ("bethe", 6, 12, 700.0, "low", 0.17210185961063654),
    ("default", 1, 1, 10.0, "high", 3.647815606237633),
    ("default", 2, 4, 50.0, "high", 2.695433095529184),
    ("default", 6, 12, 800.0, "low", 0.1574648171383248),
    ("default", 6, 12, 800.0, "high", 0.5071238603531906),
]


@pytest.mark.parametrize(("source", "z", "a", "let", "branch", "value"), _PRE_H3_VALUES)
def test_high_and_low_branches_are_unchanged(
    source: str, z: int, a: int, let: float, branch: str, value: float
) -> None:
    sp = SimpleBetheStoppingPower() if source == "bethe" else default_stopping_power()
    got = energy_for_let_MeV_u(sp, z, a, let, branch=branch)
    # bit-for-bit against the frozen reference code on this platform ...
    assert got == _pre_h3_energy_for_let(sp, z, a, let, branch=branch)
    # ... and against the recorded values (a libm may differ in the
    # last bit across platforms, so the recorded literal gets a sub-xtol tolerance)
    assert got == pytest.approx(value, rel=1e-12, abs=1e-10)


@pytest.mark.parametrize(
    ("source", "z", "a", "let", "branch", "message"),
    [
        ("default", 2, 4, 200.0, "low", "not bracketed on the low branch"),
        ("bethe", 6, 12, 5000.0, "high", "exceeds the peak LET"),
    ],
)
def test_high_and_low_branch_errors_are_unchanged(
    source: str, z: int, a: int, let: float, branch: str, message: str
) -> None:
    sp = SimpleBetheStoppingPower() if source == "bethe" else default_stopping_power()
    with pytest.raises(ValueError, match=message):
        energy_for_let_MeV_u(sp, z, a, let, branch=branch)
    with pytest.raises(ValueError, match=message):
        _pre_h3_energy_for_let(sp, z, a, let, branch=branch)
