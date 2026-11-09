"""Public Monte-Carlo primitives of `pyradbio.models.lem.base.LEMBase`.

``mean_traversals`` / ``near_traversals`` reproduce the original Scholz 1997
batch-script functions, and ``resolve_mc_engine`` replaces the scripts'
``_mc_kernel.HAVE_NUMBA`` reach-ins. The methods must be bit-identical to them,
so they are compared with ``==`` against a FROZEN copy of the original script's
functions (below). The copy, not the live script, is the reference:
``scripts/lem1_mc_reference.py`` calls the new methods itself, so comparing
against it would compare the method with itself.
"""

from __future__ import annotations

import ast
import inspect
import math
from typing import Any

import pytest

import pyradbio.models.lem as lem_pkg
from pyradbio.benchmark.scholz1997 import CHO_INPUTS
from pyradbio.core.cell import CellTarget
from pyradbio.core.photon import ClusterParameters, PhotonParameters
from pyradbio.core.radiation import Ion
from pyradbio.models.lem import _mc_kernel
from pyradbio.models.lem import base as lem_base
from pyradbio.models.lem.lem1 import LEM1
from pyradbio.models.lem.lem3 import LEM3
from pyradbio.track.northcliffe_schilling import build_stopping_power
from pyradbio.track.stopping_power import SimpleBetheStoppingPower

# --------------------------------------------------------------------------
# Frozen copy of the original Scholz 1997 batch-script implementation: its
# constants and functions, VERBATIM (bodies and
# expression order unchanged; only the docstrings are shortened). Do not "fix"
# or modernise this block -- it is the bit-identity reference.
# --------------------------------------------------------------------------
KELLERER_B_CUT_UM = 10.0
KELLERER_B_CUT_MIN_NBAR = 2000.0


def build_model(stopping_power_choice: str = "ns1970") -> LEM1:
    """The CHO stochastic-LEM model used throughout the Scholz 1997 panels."""
    photon = PhotonParameters(
        alpha_x=CHO_INPUTS.alpha_x, beta_x=CHO_INPUTS.beta_x, D_t=CHO_INPUTS.D_t
    )
    cell = CellTarget(
        nucleus_radius_um=CHO_INPUTS.nucleus_radius_um,
        effective_size_factor=CHO_INPUTS.effective_size_factor,
    )
    return LEM1(
        photon=photon,
        cell=cell,
        stopping_power=build_stopping_power(stopping_power_choice),
        beta_model="scholz1997_montecarlo",
    )


def mean_traversals(model: LEM1, ion: Ion, dose_Gy: float) -> float:
    """``n_bar``: mean tracks inside ``R_eff + r_max`` at ``dose_Gy``."""
    r_eff = model.cell.effective_nucleus_radius_um()
    b_max = r_eff + model._radial().r_max_um(ion)
    fluence = dose_Gy / (math.pi * r_eff**2 * model.mean_traversal_dose_Gy(ion))
    return fluence * math.pi * b_max**2


def near_traversals(model: LEM1, ion: Ion, n_bar: float) -> float:
    """Tracks the weighted sampler actually generates, after the near/far split."""
    if n_bar < KELLERER_B_CUT_MIN_NBAR:
        return n_bar
    r_eff = model.cell.effective_nucleus_radius_um()
    b_max = r_eff + model._radial().r_max_um(ion)
    gen_radius = min(r_eff + KELLERER_B_CUT_UM, b_max)
    return n_bar * (gen_radius / b_max) ** 2


# --------------------------------------------------------------------------
# end of the frozen copy
# --------------------------------------------------------------------------


def _lem3() -> LEM3:
    return LEM3(
        photon=PhotonParameters(alpha_x=0.129, beta_x=0.049, D_t=3.0, cluster=ClusterParameters()),
        cell=CellTarget(nucleus_radius_um=5.0),
        stopping_power=SimpleBetheStoppingPower(),
    )


def _models() -> list[Any]:
    """The script's own CHO model at both of its LET sources, plus a LEM III."""
    return [build_model("ns1970"), build_model("bethe"), _lem3()]


_IONS = (
    Ion(1, 1, 20.0),
    Ion(1, 1, 70.0),
    Ion(6, 12, 20.0),
    Ion(6, 12, 100.0),
)
_DOSES_GY = (0.5, 3.0)


def test_mean_traversals_is_bit_identical_to_the_script() -> None:
    checked = 0
    for model in _models():
        for ion in _IONS:
            for dose in _DOSES_GY:
                assert model.mean_traversals(ion, dose) == mean_traversals(model, ion, dose)
                checked += 1
    assert checked >= 6


def test_near_traversals_is_bit_identical_to_the_script() -> None:
    """The script gates the split on ``n_bar``; the method takes the cut explicitly.

    The script's rule (cut ``KELLERER_B_CUT_UM`` only from
    ``KELLERER_B_CUT_MIN_NBAR`` traversals up) maps to ``b_cut_um=None`` below
    the threshold, which is also what ``scholz1997_montecarlo_survival`` takes
    for "no split". Both branches must be exercised.
    """
    split = unsplit = 0
    for model in _models():
        for ion in _IONS:
            for dose in _DOSES_GY:
                n_bar = mean_traversals(model, ion, dose)
                cut = KELLERER_B_CUT_UM if n_bar >= KELLERER_B_CUT_MIN_NBAR else None
                new = model.near_traversals(ion, n_bar, cut)
                assert new == near_traversals(model, ion, n_bar)
                if cut is None:
                    assert new == n_bar
                    unsplit += 1
                else:
                    split += 1
    assert split >= 6
    assert unsplit >= 2


def test_near_traversals_honours_an_explicit_cut(monkeypatch: pytest.MonkeyPatch) -> None:
    """A non-default cut matches the frozen function run with that cut in its globals."""
    model = _models()[0]
    for b_cut in (2.5, 40.0, 1.0e6):  # the last one exceeds b_max: no split
        monkeypatch.setitem(globals(), "KELLERER_B_CUT_UM", b_cut)
        for ion in (Ion(6, 12, 100.0), Ion(1, 1, 70.0)):
            n_bar = model.mean_traversals(ion, 3.0)
            assert n_bar >= KELLERER_B_CUT_MIN_NBAR
            assert model.near_traversals(ion, n_bar, b_cut) == near_traversals(model, ion, n_bar)
    ion = Ion(1, 1, 70.0)
    n_bar = model.mean_traversals(ion, 3.0)
    assert model.near_traversals(ion, n_bar, 1.0e6) == n_bar


@pytest.mark.parametrize("bad", [0.0, -1.0, math.nan])
def test_near_traversals_rejects_a_non_positive_cut(bad: float) -> None:
    model = _lem3()
    ion = Ion(6, 12, 100.0)
    with pytest.raises(ValueError, match="b_cut_um"):
        model.near_traversals(ion, model.mean_traversals(ion, 3.0), bad)


def test_near_traversals_measures_the_cut_from_the_nucleus_edge() -> None:
    """The proxy exceeds the MC's own near-track Poisson mean by ``((R_eff + b)/b)^2``.

    ``scholz1997_montecarlo_survival`` generates ``n_bar (b_cut / b_max)^2``
    tracks (cut from the nucleus centre); the script's estimate (kept
    bit-identical) counts out to ``R_eff + b_cut`` (cut from the edge). A cut
    taken from the centre would give a ratio of 1 and fail here.
    """
    model = _lem3()
    ion = Ion(6, 12, 100.0)
    n_bar = model.mean_traversals(ion, 3.0)
    r_eff = model.cell.effective_nucleus_radius_um()
    b_max = r_eff + model.resolved_radial_dose.r_max_um(ion)
    b_cut = 10.0
    assert r_eff + b_cut < b_max  # both radii are inside the disc: a real split
    generated = n_bar * (b_cut / b_max) ** 2
    proxy = model.near_traversals(ion, n_bar, b_cut)
    assert proxy > generated
    assert proxy / generated == pytest.approx(((r_eff + b_cut) / b_cut) ** 2, rel=1e-12)


def test_kellerer_field_points_constant_is_the_private_one() -> None:
    assert lem_base.MC_KELLERER_FIELD_POINTS is lem_base._MC_KELLERER_FIELD_POINTS
    assert lem_pkg.MC_KELLERER_FIELD_POINTS is lem_base._MC_KELLERER_FIELD_POINTS
    # ``is`` alone cannot tell a binding from a duplicated literal (CPython shares
    # equal int constants within a module), so check the source binds the name.
    tree = ast.parse(inspect.getsource(lem_base))
    bindings = [
        node.value
        for node in ast.walk(tree)
        if isinstance(node, ast.AnnAssign)
        and isinstance(node.target, ast.Name)
        and node.target.id == "MC_KELLERER_FIELD_POINTS"
    ]
    assert len(bindings) == 1
    assert isinstance(bindings[0], ast.Name)
    assert bindings[0].id == "_MC_KELLERER_FIELD_POINTS"


def test_package_exports() -> None:
    assert lem_pkg.resolve_mc_engine is lem_base.resolve_mc_engine
    assert {"MC_KELLERER_FIELD_POINTS", "resolve_mc_engine"} <= set(lem_pkg.__all__)


def test_resolve_mc_engine_numpy_is_always_available(monkeypatch: pytest.MonkeyPatch) -> None:
    for have in (True, False):
        monkeypatch.setattr(_mc_kernel, "HAVE_NUMBA", have)
        assert lem_base.resolve_mc_engine("numpy") == "numpy"


def test_resolve_mc_engine_with_numba(monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.setattr(_mc_kernel, "HAVE_NUMBA", True)
    assert lem_base.resolve_mc_engine("numba") == "numba"
    assert lem_base.resolve_mc_engine("auto") == "numba"


def test_resolve_mc_engine_without_numba(monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.setattr(_mc_kernel, "HAVE_NUMBA", False)
    assert lem_base.resolve_mc_engine("auto") == "numpy"
    with pytest.raises(ValueError, match="fast"):
        lem_base.resolve_mc_engine("numba")


@pytest.mark.parametrize("bad", ["cuda", "NumPy", "", None, 1])
def test_resolve_mc_engine_rejects_unknown_engines(bad: object) -> None:
    with pytest.raises(ValueError, match="engine"):
        lem_base.resolve_mc_engine(bad)


def test_the_auto_rule_is_written_once() -> None:
    """Every ``"auto"`` site in ``base.py`` shares one definition of the rule.

    The private Monte-Carlo helpers keep their own (lenient) handling of an
    explicit ``"numba"``, but the ``auto -> numba if importable`` choice itself
    must not be restated inline.
    """
    source = inspect.getsource(lem_base)
    assert source.count('"numba" if _mc_kernel.HAVE_NUMBA else "numpy"') == 1


def test_montecarlo_survival_uses_the_shared_resolution() -> None:
    """The public MC entry point rejects an unknown engine instead of running numpy."""
    with pytest.raises(ValueError, match="engine"):
        _lem3().scholz1997_montecarlo_survival(
            Ion(6, 12, 10.0),
            [1.0],
            n_realizations=10,
            engine="cuda",
        )
