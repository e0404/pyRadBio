"""Cross-implementation benchmarks against TRiP98 and Survival-code tables.

pyRadBio's LEM I, LEM IV, and modified-MKM are compared to reference tables from
two independent, established implementations bundled under
``pyradbio/benchmark/data/``:

* TRiP98 RBE tables (GSI treatment planning) — ``RBE_alpha`` vs energy.
* "Survival" code JSON (Manganaro et al.) — ``alpha``/``beta`` vs energy for
  rapidLEM (Kraemer-Scholz 2006) and rapid mMKM (Inaniwa 2010).

Agreement is quantified by the median absolute relative error over an energy
window. LEM I and mMKM reproduce their reference implementations tightly (a few
percent); LEM IV's single-particle approximation only tracks the TRiP LEM IV
qualitatively in the high-energy tail (see the LEM IV test and
docs/references/LEM_IV_Friedrich2012.md for the documented magnitude limit).
"""

from __future__ import annotations

import importlib.util
import math
from itertools import pairwise

import numpy as np
import pytest

from pyradbio.benchmark import reference_tables as rt
from pyradbio.core.cell import CellTarget
from pyradbio.core.photon import ClusterParameters, PhotonParameters
from pyradbio.core.radiation import Ion
from pyradbio.models.lem.lem1 import LEM1
from pyradbio.models.lem.lem4 import LEM4

# find_spec does not import pyMKM (avoids its NumPy-reload side effect at collection).
requires_pymkm = pytest.mark.skipif(
    importlib.util.find_spec("pymkm") is None, reason="pyMKM not installed ([mkm] extra)"
)


def _reference_data_available() -> bool:
    """Whether the (non-committed) TRiP/Survival reference tables are present."""
    try:
        rt.load_trip_lem1("normal02.rbe")
        rt.load_lem1_survival_hsg()
    except (FileNotFoundError, OSError):
        return False
    return True


# The TRiP `.rbe` / Survival `.json` tables under benchmark/data are not committed;
# skip the data-driven benchmarks when they are absent (the parser/metric unit
# tests below need no data and always run).
requires_reference_data = pytest.mark.skipif(
    not _reference_data_available(), reason="TRiP/Survival reference tables not present"
)


def _photon(table: rt.TripRbeTable) -> PhotonParameters:
    # No cluster params: this helper feeds LEM1/LEM4, neither of which consumes
    # photon.cluster (LEM IV warns on it). LEM II/III set it explicitly instead.
    return PhotonParameters(alpha_x=table.alpha_x, beta_x=table.beta_x, D_t=table.D_t)


def _subsample(indices: list[int], max_points: int) -> list[int]:
    """Thin ``indices`` to at most ``max_points`` (keeps benchmarks fast)."""
    if len(indices) <= max_points:
        return indices
    stride = len(indices) // max_points
    return indices[::stride]


# ---- loader / parser unit tests ----------------------------------------
def test_parse_projectile_maps_symbol_to_z() -> None:
    """``<A><Symbol>`` tokens parse to ``(Z, A)``; unknowns raise."""
    assert rt.parse_projectile("12C") == (6, 12)
    assert rt.parse_projectile(" 2H ") == (1, 2)
    assert rt.parse_projectile("20Ne") == (10, 20)
    assert rt.parse_projectile("4He") == (2, 4)
    for bad in ("C12", "12Xx", "carbon"):
        with pytest.raises(ValueError):
            rt.parse_projectile(bad)


@requires_reference_data
def test_load_trip_lem1_header_and_curves() -> None:
    """A TRiP LEM I table exposes its photon header and per-projectile curves."""
    table = rt.load_trip_lem1("normal02.rbe")
    assert table.celltype == "normal02"
    assert (table.alpha_x, table.beta_x, table.D_t, table.r_nucleus_um) == (0.1, 0.05, 30.0, 5.0)

    # Carbon-12 through neon-20 are present; curves are energy-sorted and positive.
    assert (6, 12) in table.curves and (10, 20) in table.curves
    carbon = table.curve(6, 12)
    assert carbon.Z == 6 and carbon.A == 12
    assert len(carbon.energies_MeV_u) == len(carbon.rbe_alpha) > 20
    assert list(carbon.energies_MeV_u) == sorted(carbon.energies_MeV_u)
    assert all(r > 0.0 for r in carbon.rbe_alpha)


def test_parse_trip_dedx_reads_header_and_curves() -> None:
    """A TRiP ``.dedx`` text parses into material header + per-projectile curves."""
    text = (
        "!filetype    dEdx\n"
        "# FLUKA 77.3 eV\n"
        "!material    H2O\n"
        "!density    1\n"
        "!projectile   2H \n"
        "# E/(MeV/u) dE/dx Range\n"
        "!dedx\n"
        "  1.  261.4  0.00245\n"
        "  10.  48.5  0.05\n"
        "!projectile  12C \n"
        "!dedx\n"
        "  1.  9000.  0.0001\n"
        "  10.  1600.  0.0025\n"
    )
    table = rt.parse_trip_dedx(text)

    assert table.material == "H2O"
    assert table.density_g_cm3 == 1.0
    assert set(table.curves) == {(1, 2), (6, 12)}
    carbon = table.curve(6, 12)
    assert carbon.energies_MeV_u == (1.0, 10.0)
    assert carbon.mass_stopping_MeV_cm2_g == (9000.0, 1600.0)
    assert carbon.range_g_cm2 == (0.0001, 0.0025)
    with pytest.raises(KeyError):
        table.curve(10, 20)  # not tabulated in this FLUKA file (Z<=6)


@requires_reference_data
def test_load_trip_dedx_light_ions_present() -> None:
    """The bundled FLUKA dE/dx table exposes light-ion (Z<=6) mass stopping power."""
    table = rt.load_trip_dedx()
    assert table.material == "H2O"
    assert (1, 2) in table.curves and (6, 12) in table.curves
    assert (10, 20) not in table.curves  # FLUKA file stops at carbon
    carbon = table.curve(6, 12)
    assert len(carbon.energies_MeV_u) == len(carbon.mass_stopping_MeV_cm2_g) > 20
    assert all(s > 0.0 for s in carbon.mass_stopping_MeV_cm2_g)


@requires_reference_data
def test_load_survival_json_structure() -> None:
    """The Survival-code LEM I / mMKM JSON parse into model + per-ion LQ curves."""
    lem1 = rt.load_lem1_survival_hsg()
    assert lem1.model == "LEMI"
    assert lem1.parameters["alphaX"] == pytest.approx(0.217)
    assert lem1.parameters["Dt"] == pytest.approx(30.0)
    assert {(c.Z, c.A) for c in lem1.curves} >= {(1, 1), (6, 12)}
    carbon = lem1.curve(6, 12)
    assert len(carbon.energies_MeV_u) == len(carbon.alpha) == len(carbon.beta) == 100
    assert all(a > 0.0 for a in carbon.alpha)

    mmkm = rt.load_mmkm_survival_hsg()
    assert mmkm.model == "mMKM"
    assert mmkm.parameters["alpha0"] == pytest.approx(0.117)
    assert mmkm.parameters["rDomain"] == pytest.approx(0.28)

    with pytest.raises(KeyError):
        lem1.curve(99, 250)  # absent ion


def test_median_abs_relative_error_known_values() -> None:
    """The benchmark metric on hand-checked inputs."""
    assert rt.median_abs_relative_error([1.0, 1.0], [1.0, 1.0]) == 0.0
    # errors 0.1, 0.5 -> median 0.3 (even count averages the two middle values).
    assert rt.median_abs_relative_error([1.1, 1.5], [1.0, 1.0]) == pytest.approx(0.3)
    # zero-reference samples are skipped.
    assert rt.median_abs_relative_error([2.0, 1.2], [0.0, 1.0]) == pytest.approx(0.2)
    with pytest.raises(ValueError):
        rt.median_abs_relative_error([1.0], [0.0])


def test_window_indices_selects_inclusive_range() -> None:
    energies = [1.0, 10.0, 50.0, 100.0, 500.0]
    assert rt.window_indices(energies, 10.0, 100.0) == [1, 2, 3]


# ---- LEM I: tight agreement with reference implementations --------------
@requires_reference_data
@pytest.mark.golden
def test_golden_lem1_matches_survival_rapidlem_hsg() -> None:
    """LEM I alpha(E) matches the Survival-code rapidLEM HSG table to a few %.

    Reference: `LEMI_Scholz2006_HSG_SurvivalCode.json` (rapidLEM, Kraemer-Scholz
    2006 approximation; HSG alpha_x=0.217, beta_x=0.0615, D_t=30 Gy, R_n=5 um).
    pyRadBio's `LEM1` uses the same fast slope-matching beta and the amorphous
    Scholz-Kraft track, so per-ion median |rel err| on alpha stays small across
    protons->carbon over the therapeutic window (10-400 MeV/u).
    """
    table = rt.load_lem1_survival_hsg()
    photon = PhotonParameters(
        alpha_x=float(table.parameters["alphaX"]),
        beta_x=float(table.parameters["betaX"]),
        D_t=float(table.parameters["Dt"]),
        cluster=ClusterParameters(),
    )
    assert photon.alpha_x == 0.217  # sanity: parameters wired from the table
    lem = LEM1(photon=photon, cell=CellTarget(float(table.parameters["rNucleus"])))

    for c in table.curves:
        idx = _subsample(rt.window_indices(c.energies_MeV_u, 10.0, 400.0), 20)
        pred = [lem.alpha_beta(Ion(c.Z, c.A, c.energies_MeV_u[i])).alpha for i in idx]
        ref = [c.alpha[i] for i in idx]
        err = rt.median_abs_relative_error(pred, ref)
        assert err < 0.12, f"LEM I vs rapidLEM ion Z{c.Z}A{c.A}: median |rel err|={err:.3f}"


@requires_reference_data
@pytest.mark.golden
def test_golden_lem1_matches_trip_rbe() -> None:
    """LEM I RBE_alpha(E) matches TRiP98 LEM I to a few % in the therapeutic window.

    Reference: `data/TRIP_RBE/LEM1/normal02.rbe` (alpha_x=0.1, beta_x=0.05,
    D_t=30 Gy, R_n=5 um). RBE_alpha = alpha_ion/alpha_x. Agreement is tight for
    E >= 50 MeV/u; below ~10 MeV/u TRiP's LEM saturates RBE_alpha toward
    s_max/alpha_x while pyRadBio's inactivation cross-section does not, so the
    benchmark uses the therapeutic window (see module docstring).
    """
    table = rt.load_trip_lem1("normal02.rbe")
    lem = LEM1(photon=_photon(table), cell=CellTarget(table.r_nucleus_um))

    for Z, A in ((1, 2), (2, 4), (6, 12), (8, 16), (10, 20)):
        c = table.curve(Z, A)
        idx = _subsample(rt.window_indices(c.energies_MeV_u, 50.0, 430.0), 12)
        pred = [lem.alpha_beta(Ion(Z, A, c.energies_MeV_u[i])).alpha / table.alpha_x for i in idx]
        ref = [c.rbe_alpha[i] for i in idx]
        err = rt.median_abs_relative_error(pred, ref)
        assert err < 0.10, f"LEM I vs TRiP ion Z{Z}A{A}: median |rel err|={err:.3f}"


@requires_reference_data
@pytest.mark.golden
def test_golden_lem1_lethal_central_matches_full_trip_curve() -> None:
    """The lethal_central mode reproduces TRiP98 LEM I over the *whole* energy range.

    Reference: `data/TRIP_RBE/LEM1/normal02.rbe`. With
    `initial_slope_model="lethal_central"` (alpha_I = min(N_1/D_1, s_max), the
    Kraemer-Scholz 2006 / TRiP98 rapidLEM formula), pyRadBio reproduces the TRiP
    RBE_alpha curve to a few % across the full range -- including the low-energy
    `s_max/alpha_x` plateau that the default cross-section (overkill thindown)
    does not, and where the default's median error is much larger.
    """
    table = rt.load_trip_lem1("normal02.rbe")
    cell = CellTarget(table.r_nucleus_um)
    trip = LEM1(photon=_photon(table), cell=cell, initial_slope_model="lethal_central")
    default = LEM1(photon=_photon(table), cell=cell)
    s_max_rbe = (table.alpha_x + 2.0 * table.beta_x * table.D_t) / table.alpha_x  # plateau

    def rbe(
        model: LEM1, Z: int, A: int, indices: list[int], c: rt.ProjectileRbeCurve
    ) -> list[float]:
        return [
            model.alpha_beta(Ion(Z, A, c.energies_MeV_u[i])).alpha / table.alpha_x for i in indices
        ]

    for Z, A in ((2, 4), (6, 12), (10, 20)):
        c = table.curve(Z, A)
        # Full range: lethal_central tracks TRiP to a few % (incl. the plateau) and
        # never overshoots s_max/alpha_x.
        full = _subsample([i for i, e in enumerate(c.energies_MeV_u) if e >= 1.0], 16)
        rbe_trip = rbe(trip, Z, A, full, c)
        ref_full = [c.rbe_alpha[i] for i in full]
        assert rt.median_abs_relative_error(rbe_trip, ref_full) < 0.05, f"lethal_central Z{Z}A{A}"
        assert max(rbe_trip) <= s_max_rbe * 1.001  # never overshoots the plateau

        # Low-energy overkill region (1-10 MeV/u): lethal_central reproduces the
        # plateau where the default (thindown) is far off.
        low = [i for i, e in enumerate(c.energies_MeV_u) if 1.0 <= e <= 10.0]
        ref_low = [c.rbe_alpha[i] for i in low]
        err_trip_low = rt.median_abs_relative_error(rbe(trip, Z, A, low, c), ref_low)
        err_def_low = rt.median_abs_relative_error(rbe(default, Z, A, low, c), ref_low)
        assert err_trip_low < 0.05 < err_def_low, (
            f"Z{Z}A{A} low-E: lethal_central {err_trip_low:.3f} vs default {err_def_low:.3f}"
        )


@requires_reference_data
@pytest.mark.golden
def test_golden_lem1_lethal_full_matches_trip_across_tissues() -> None:
    """The lethal_full mode reproduces TRiP98 LEM I over the whole range AND across tissues.

    `initial_slope_model="lethal_full"` integrates the *unsaturated* lethal
    events `nu(b)` over impact parameter -- the same overkill-free simplification
    as TRiP98/rapidLEM (cap at `s_max`, no thindown), but keeping the indirect
    traversals that `lethal_central`'s central-only form drops. (It is not the exact
    LEM slope; `cross_section` keeps the overkill term and matches the explicit model /
    MC instead.) So it tracks the TRiP tables where each of the other two fails:
    at high energy (`lethal_central` underestimates, missing indirect traversals) and
    through the low-energy plateau (`cross_section` thins down). This shows up on a
    high-alpha_x / high-alpha-beta tissue (`normal10`, plateau `s_max/alpha_x=7`),
    where the `lethal_central`/`cross_section` crossover sits mid-curve; `lethal_full` matches
    the whole curve for both a shouldered (`normal02`) and a shallow (`normal10`)
    tissue. Uses pyRadBio's default stopping power (not TRiP's FLUKA dE/dx), so the
    tolerance absorbs the ~2% LET difference.
    """
    for filename in ("normal02.rbe", "normal10.rbe"):
        table = rt.load_trip_lem1(filename)
        cell = CellTarget(table.r_nucleus_um)
        full_lem = LEM1(photon=_photon(table), cell=cell, initial_slope_model="lethal_full")
        s_max_rbe = (table.alpha_x + 2.0 * table.beta_x * table.D_t) / table.alpha_x

        for Z, A in ((1, 2), (2, 4), (6, 12)):
            c = table.curve(Z, A)
            full = _subsample([i for i, e in enumerate(c.energies_MeV_u) if e >= 1.0], 16)
            pred = [
                full_lem.initial_slope(Ion(Z, A, c.energies_MeV_u[i])) / table.alpha_x for i in full
            ]
            ref = [c.rbe_alpha[i] for i in full]
            err = rt.median_abs_relative_error(pred, ref)
            assert err < 0.025, f"lethal_full {filename} Z{Z}A{A}: median |rel err|={err:.3f}"
            assert max(pred) <= s_max_rbe * 1.001  # never overshoots the plateau

    # On normal10 carbon, lethal_full beats BOTH other paths over the full range
    # (the crossover case): default thins down at low E, lethal_central drops at high E.
    table = rt.load_trip_lem1("normal10.rbe")
    cell = CellTarget(table.r_nucleus_um)
    models = {
        "cross_section": LEM1(photon=_photon(table), cell=cell),
        "lethal_central": LEM1(
            photon=_photon(table), cell=cell, initial_slope_model="lethal_central"
        ),
        "lethal_full": LEM1(photon=_photon(table), cell=cell, initial_slope_model="lethal_full"),
    }
    c = table.curve(6, 12)
    full = [i for i, e in enumerate(c.energies_MeV_u) if e >= 1.0]
    ref = [c.rbe_alpha[i] for i in full]
    errs = {
        name: rt.median_abs_relative_error(
            [m.initial_slope(Ion(6, 12, c.energies_MeV_u[i])) / table.alpha_x for i in full], ref
        )
        for name, m in models.items()
    }
    assert errs["lethal_full"] < errs["cross_section"], errs
    assert errs["lethal_full"] < errs["lethal_central"], errs


# ---- mMKM: tight agreement with the Survival-code rapid mMKM ------------
def _mmkm_survival_errors(backend_factory: object = None) -> dict[tuple[int, int], float]:
    """Per-ion median |rel err| of mMKM alpha vs the Survival rapid mMKM HSG table."""
    from pyradbio.models.mkm.model import MKMModel

    table = rt.load_mmkm_survival_hsg()
    geometry = {
        "domain_radius_um": float(table.parameters["rDomain"]),
        "nucleus_radius_um": float(table.parameters["rNucleus"]),
        "beta0": float(table.parameters["betaX"]),
    }
    alpha0 = float(table.parameters["alpha0"])
    if backend_factory is None:
        mkm = MKMModel(alpha0=alpha0, **geometry)
    else:
        mkm = MKMModel(alpha0=alpha0, backend=backend_factory(**geometry))  # type: ignore[operator]
    errors = {}
    for c in table.curves:
        idx = _subsample(rt.window_indices(c.energies_MeV_u, 10.0, 400.0), 12)
        pred = [mkm.alpha_beta(Ion(c.Z, c.A, c.energies_MeV_u[i])).alpha for i in idx]
        ref = [c.alpha[i] for i in idx]
        errors[(c.Z, c.A)] = rt.median_abs_relative_error(pred, ref)
    return errors


@requires_reference_data
@pytest.mark.golden
def test_golden_mmkm_matches_survival_rapidmmkm_hsg() -> None:
    """mMKM alpha(E) matches the Survival-code rapid mMKM HSG table to a few %.

    Reference: `mMKM_HSG_SurvivalCode.json` (rapid mMKM, Inaniwa 2010; HSG
    alpha0=0.117, beta0=0.0615, R_n=4.5 um, r_d=0.28 um). pyRadBio's `MKMModel`
    on its default native backend computes alpha = alpha0 + beta0*z*_1D from the
    same saturation-corrected dose-mean specific energy, so per-ion median
    |rel err| on alpha is within ~10% over 10-400 MeV/u (protons and carbon agree
    to a few %; the light-ion residual reflects the track-structure and
    stopping-power differences between the two codes).
    """
    for (z, a), err in _mmkm_survival_errors().items():
        assert err < 0.12, f"mMKM vs rapid mMKM ion Z{z}A{a}: median |rel err|={err:.3f}"


@requires_reference_data
@requires_pymkm
@pytest.mark.golden
def test_golden_mmkm_pymkm_backend_matches_survival_rapidmmkm_hsg() -> None:
    """The same golden through an injected pyMKM backend (independent cross-check)."""
    from pyradbio.models.mkm.pymkm_backend import PyMKMBackend

    for (z, a), err in _mmkm_survival_errors(PyMKMBackend).items():
        assert err < 0.12, f"pyMKM mMKM vs rapid mMKM ion Z{z}A{a}: median |rel err|={err:.3f}"


# ---- LEM IV: qualitative tracking only (documented magnitude limit) -----
@requires_reference_data
def test_lem4_tracks_trip_rbe_in_high_energy_tail() -> None:
    """LEM IV tracks TRiP98 LEM IV only qualitatively, in the high-energy tail.

    Reference: `data/TRIP_RBE/LEM4/chordoma_lemIV.rbe`. pyRadBio's single-particle
    LEM IV reproduces the *shape* of the RBE_alpha(LET) curve but under-predicts
    the strong low-energy RBE_alpha peak of the full TRiP Monte-Carlo (which
    scores discrete DSB on the 540 nm grid; see
    docs/references/LEM_IV_Friedrich2012.md). This benchmark therefore only checks
    that the model runs over the whole table and, in the high-energy tail
    (>= 100 MeV/u), both curves decrease with energy and agree within a factor of
    ~3 -- it does not pin the low-energy magnitude.
    """
    table = rt.load_trip_lem4("chordoma_lemIV.rbe")
    # Legacy-capped: the assertion sweeps the *whole* table, including energies
    # below 2.206 MeV/u, where the uncapped default is well posed but far above
    # the table. The claim under test is about the high-energy tail.
    lem = LEM4(
        photon=_photon(table),
        cell=CellTarget(table.r_nucleus_um),
        legacy_trip98_cluster_index_cap=0.95,
    )
    carbon = table.curve(6, 12)

    # Runs across the whole table: finite, positive RBE_alpha everywhere.
    predicted = {
        e: lem.alpha_beta(Ion(6, 12, e)).alpha / table.alpha_x
        for e in carbon.energies_MeV_u
        if e >= 1.0  # below ~1 MeV/u the track model has no delta-ray range
    }
    assert all(0.0 < r < 1.0e3 for r in predicted.values())

    # High-energy tail: correct decreasing trend and same order of magnitude.
    tail_idx = rt.window_indices(carbon.energies_MeV_u, 100.0, 430.0)
    tail_e = [carbon.energies_MeV_u[i] for i in tail_idx]
    ref_tail = [carbon.rbe_alpha[i] for i in tail_idx]
    pred_tail = [predicted[e] for e in tail_e]

    assert ref_tail == sorted(ref_tail, reverse=True), "reference tail not decreasing"
    assert pred_tail == sorted(pred_tail, reverse=True), "pyRadBio tail not decreasing"
    for e, p, r in zip(tail_e, pred_tail, ref_tail, strict=True):
        assert 1.0 / 3.0 < p / r < 3.0, f"E={e:.0f}: RBE ratio {p / r:.2f} outside [1/3, 3]"


# ---- Survival's eta table: a traced figure, not a computed curve ---------
def _eta_available() -> bool:
    try:
        rt.load_survival_eta_table()
    except (FileNotFoundError, OSError):
        return False
    return True


requires_survival_src = pytest.mark.skipif(
    not _eta_available(), reason="Survival source checkout not configured"
)


def test_parse_eta_table_reads_two_columns() -> None:
    table = rt.parse_eta_table("# c\n1.0e2  1.05\n1.0e3  2.5\n1.0e4  9.0\n")
    assert table.dose_Gy == (100.0, 1000.0, 10000.0)
    assert table.peak() == (10000.0, 9.0)
    # log-log interpolation, exactly on a node and midway between two
    assert table.eta_at(1000.0) == pytest.approx(2.5)
    assert table.eta_at(math.sqrt(1e2 * 1e3)) == pytest.approx(math.sqrt(1.05 * 2.5))
    # outside the tabulated window Survival returns 1, not an extrapolation
    assert table.eta_at(50.0) == 1.0
    assert table.eta_at(1.0e9) == 1.0


def test_parse_eta_table_rejects_a_one_row_file() -> None:
    with pytest.raises(ValueError, match="at least two rows"):
        rt.parse_eta_table("1.0e2  1.05\n")


@requires_survival_src
def test_survival_eta_table_is_a_traced_figure_not_either_of_its_own_generators() -> None:
    """Pins the provenance argument in ``docs/references/survival_repo_notes.md``.

    Both routines that could have produced this table return ``>= 1`` by
    construction (``analyticDamageEnhancement`` is ``1 + non-negative``;
    ``damageEnhancement`` is ``(N_DSB + N_2SB)/N_DSB``). The shipped table dips
    below 1 and wobbles on its rising limb, so it is neither -- it is Elsasser &
    Scholz 2007 fig. 2 traced and resampled. That is why it cannot be used as
    independent corroboration of fig. 2 against our own cluster rule.
    """
    table = rt.load_survival_eta_table()
    assert len(table.dose_Gy) == 200
    assert table.dose_Gy[0] == pytest.approx(1.0e2)
    assert table.dose_Gy[-1] == pytest.approx(5.0e6)

    dose_peak, eta_peak = table.peak()
    assert eta_peak == pytest.approx(14.807, rel=1e-3)
    assert dose_peak == pytest.approx(4.571e5, rel=1e-3)

    below_one = [e for e in table.eta if e < 1.0]
    assert below_one, "eta < 1 is impossible for both of Survival's own generators"
    assert min(below_one) == pytest.approx(0.98707, rel=1e-4)

    rising = [e for d, e in zip(table.dose_Gy, table.eta, strict=True) if 1e2 < d < 1e5]
    steps = [b - a for a, b in pairwise(rising)]
    sign_changes = sum(1 for a, b in pairwise(steps) if (a > 0) != (b > 0))
    assert sign_changes > 5, "a curve from a smooth rule rises monotonically here"


# ---- table -> model inputs ------------------------------------------------
_RBE_TEXT = (
    "!celltype normal02\n"
    "!alpha 0.1\n"
    "!beta 0.05\n"
    "!cut 30\n"
    "!rnucleus 5.0\n"
    "!projectile 12C\n"
    "!rbe\n"
    "# E RBE\n"
    "1.0 3.0\n"
    "10.0 2.0\n"
)
_DEDX_TEXT = (
    "!material H2O\n"
    "!density 1.0\n"
    "!projectile 2H\n"
    "!dedx\n"
    "1. 261.4\n"
    "10. 48.5\n"
    "100. 7.3\n"
    "!projectile 12C\n"
    "!dedx\n"
    "1. 9000.\n"
    "10. 1600.\n"
)


def test_trip_rbe_table_photon_parameters_and_cell_target() -> None:
    """The header becomes the photon set and nucleus the TRiP98 scripts spelled out."""
    table = rt.parse_trip_rbe(_RBE_TEXT)
    assert table.photon_parameters() == PhotonParameters(alpha_x=0.1, beta_x=0.05, D_t=30.0)
    assert table.photon_parameters().cluster is None
    cell = table.cell_target()
    assert cell == CellTarget(5.0)
    assert cell.effective_size_factor == 1.0


def test_trip_dedx_stopping_power_interpolates_log_log() -> None:
    """LET = exp(interp(log E, log E_i, log S_i)) * rho * 0.1, the scripts' adapter."""
    table = rt.parse_trip_dedx(_DEDX_TEXT)
    sp = table.stopping_power()
    assert isinstance(sp, rt.TripDedxStoppingPower)
    assert sp.covers(6, 12) and sp.covers(1, 2) and not sp.covers(2, 4)
    # on a node: the tabulated value, converted MeV cm^2/g -> keV/um
    assert sp.let_keV_um(Ion(6, 12, 10.0)) == pytest.approx(160.0, rel=1e-14)
    # between nodes: a power law in E
    expected = math.exp(0.5 * (math.log(9000.0) + math.log(1600.0))) * 0.1
    assert sp.let_keV_um(Ion(6, 12, math.sqrt(10.0))) == pytest.approx(expected, rel=1e-14)
    # an ion the table lacks: KeyError, or the fallback when one is given
    with pytest.raises(KeyError, match="Z<=6 only"):
        sp.let_keV_um(Ion(2, 4, 10.0))

    class _Fallback:
        def let_keV_um(self, ion: Ion) -> float:
            return 42.0

    assert table.stopping_power(fallback=_Fallback()).let_keV_um(Ion(2, 4, 10.0)) == 42.0


def test_trip_dedx_stopping_power_is_described_and_hashes_by_identity() -> None:
    from pyradbio.track.stopping_power import describe_stopping_power

    table = rt.parse_trip_dedx(_DEDX_TEXT)
    sp = table.stopping_power()
    assert sp.name == "trip98-fluka-dedx"
    assert describe_stopping_power(sp) == "TRiP98 FLUKA dE/dx table"
    other = table.stopping_power()
    assert sp != other and len({sp, other}) == 2  # identity, like the scripts' class


@requires_reference_data
def test_trip_dedx_stopping_power_matches_the_scripts_formula_on_the_real_table() -> None:
    """Bit-for-bit the arithmetic of the original examples' TripDedxStoppingPower."""
    table = rt.load_trip_dedx()
    sp = table.stopping_power()
    for (Z, A), curve in table.curves.items():
        log_e = [math.log(e) for e in curve.energies_MeV_u]
        log_s = [math.log(s) for s in curve.mass_stopping_MeV_cm2_g]
        for e in (0.15, 1.0, 7.3, 88.0, 400.0):
            mass_sp = math.exp(float(np.interp(math.log(e), log_e, log_s)))
            assert sp.let_keV_um(Ion(Z, A, e)) == mass_sp * table.density_g_cm3 * 0.1


def test_windowed_median_errors_scores_each_window_and_skips_nan() -> None:
    energies = [0.5, 1.0, 10.0, 100.0]
    reference = [2.0, 2.0, 1.0, 1.0]
    predicted = [float("nan"), 2.2, 1.5, 1.0]
    windows = {"sub": (0.0, 0.6), "low": (1.0, 10.0), "all": (0.0, 1e9)}
    scores = rt.windowed_median_errors(energies, predicted, reference, windows)
    assert list(scores) == ["sub", "low", "all"]
    assert scores["sub"] is None  # only a NaN prediction in the window
    assert scores["low"] == pytest.approx((0.3, 2))  # errors 0.1, 0.5 -> median 0.3
    assert scores["all"] == pytest.approx((0.1, 3))  # errors 0.1, 0.5, 0.0
