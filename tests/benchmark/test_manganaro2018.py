"""The Manganaro et al. 2018 comparison inputs (``pyradbio.benchmark.manganaro2018``).

These tests pin the constants and the cache-key payload to the values the
committed cache was written with, and -- where Survival's SRIM tables are
configured -- re-derive every key of the committed cache
``validation/data/survival_our_mc.json``.
"""

from __future__ import annotations

from pathlib import Path

import numpy as np
import pytest

from pyradbio.benchmark import manganaro2018 as m18
from pyradbio.benchmark.checkpoint import JsonPointCache, config_key
from pyradbio.core.cell import CellTarget
from pyradbio.core.lq import fit_lq_weighted
from pyradbio.core.photon import PhotonParameters
from pyradbio.core.radiation import Ion
from pyradbio.models.lem.lem1 import LEM1
from pyradbio.models.lem.lem2 import LEM2
from pyradbio.models.lem.lem3 import LEM3
from pyradbio.track.scholz_kraft import R_MAX_GAMMA_UM
from pyradbio.track.stopping_power import SimpleBetheStoppingPower

CACHE = Path(__file__).resolve().parents[2] / "validation" / "data" / "survival_our_mc.json"

# The payload keys of _mc_config_key in examples/survival_code_lem_comparison.py.
_OLD_PAYLOAD_KEYS = {
    "v", "model", "Z", "A", "E", "alpha_x", "beta_x", "R_N", "D_t", "r_max_gamma",
    "r_min_um", "sigma_nm", "doses", "repeats", "b_cut", "b_cut_min_nbar",
    "target_alpha", "target_beta", "n_pilot", "n_min", "n_max", "let",
}  # fmt: skip


def test_constants_are_the_scripts_values() -> None:
    """Table 1 HSG LEM row and the paired grid, as the two scripts hard-coded them."""
    assert (m18.HSG_LEM.alpha_x, m18.HSG_LEM.beta_x, m18.HSG_LEM.nucleus_radius_um) == (
        0.313,
        0.0615,
        5.0,
    )
    assert dict(m18.HSG_LEM.D_t) == {"LEMI": 30.0, "LEMII": 6.0, "LEMIII": 19.0}
    assert dict(m18.PAIRED_IONS) == {"H": (1, 1), "He": (2, 4), "C": (6, 12), "O": (8, 16)}
    assert m18.PAIRED_ENERGIES_MEV_U == (3.0, 10.0, 30.0, 100.0, 300.0)
    assert dict(m18.LEM_MODELS) == {"LEMI": LEM1, "LEMII": LEM2, "LEMIII": LEM3}
    assert m18.HSG_LEM.photon("LEMII") == PhotonParameters(alpha_x=0.313, beta_x=0.0615, D_t=6.0)
    assert m18.HSG_LEM.cell_target() == CellTarget(5.0)
    assert (
        m18.survival_table_filename("LEMII", "MonteCarloPaired")
        == "LEMII_MonteCarloPaired_HSG_Manganaro2018.json"
    )


def test_mc_settings_are_the_scripts_values() -> None:
    s = m18.OUR_MC
    assert s.cache_version == 1
    assert s.doses_Gy == (1.0, 2.0, 3.0, 4.0, 5.0, 6.0)
    assert (s.n_repeat, s.n_pilot, s.n_min, s.n_max) == (8, 400, 250, 200_000)
    assert (s.target_rel_alpha, s.target_abs_beta) == (0.003, 0.02 * 0.0615)
    assert (s.b_cut_um, s.b_cut_min_nbar) == (10.0, 2000.0)
    assert s.models() == ["LEMI", "LEMII", "LEMIII"]
    assert s.energies("LEMI", 1) == (1.0, 3.0, 10.0, 30.0, 100.0, 300.0)
    assert s.energies("LEMIII", 8) == (1.0, 3.0, 10.0, 20.0, 30.0, 100.0, 300.0)
    assert m18.OurMcSettings(high_e=False).energies("LEMII", 6) == (1.0, 3.0, 10.0, 20.0)


def test_survival_matched_lem_sets_survival_r_max_for_lem1_only() -> None:
    sp = SimpleBetheStoppingPower()
    lem1 = m18.survival_matched_lem("LEMI", sp)
    assert isinstance(lem1, LEM1)
    assert lem1.resolved_radial_dose.r_max_gamma_um == R_MAX_GAMMA_UM  # type: ignore[attr-defined]
    own = m18.survival_matched_lem("LEMI", sp, match_r_max=False)
    assert own.resolved_radial_dose.r_max_gamma_um != R_MAX_GAMMA_UM  # type: ignore[attr-defined]
    lem2 = m18.survival_matched_lem("LEMII", sp, beta_model="rapid_lem")
    assert isinstance(lem2, LEM2) and lem2.beta_model == "rapid_lem"
    assert lem2.photon.D_t == 6.0 and lem2.cell == CellTarget(5.0)


def test_mc_cache_payload_has_the_old_keys_and_hashes_with_config_key() -> None:
    sp = SimpleBetheStoppingPower()
    lem = m18.survival_matched_lem("LEMII", sp)
    payload = m18.mc_cache_payload("LEMII", 6, 12, 20.0, lem, sp)
    assert set(payload) == _OLD_PAYLOAD_KEYS
    assert payload["v"] == 1 and payload["model"] == "LEMII" and payload["E"] == 20.0
    assert payload["D_t"] == 6.0 and payload["doses"] == [1.0, 2.0, 3.0, 4.0, 5.0, 6.0]
    assert payload["let"] == sp.let_keV_um(Ion(6, 12, 20.0))
    key = m18.mc_cache_key("LEMII", 6, 12, 20.0, lem, sp)
    assert key == config_key(payload) and len(key) == 16


def _srim_available() -> bool:
    try:
        from pyradbio.track.srim import srim_stopping_power

        srim_stopping_power(("H", "He", "C", "O"))
    except (FileNotFoundError, OSError):
        return False
    return True


@pytest.mark.skipif(not _srim_available(), reason="Survival source checkout not configured")
def test_every_committed_cache_key_is_re_derived_from_its_point() -> None:
    """``mc_cache_key`` reproduces the keys the old script wrote, for all 78 points."""
    from pyradbio.track.srim import srim_stopping_power

    sp = srim_stopping_power(("H", "He", "C", "O"))
    lems = {m: m18.survival_matched_lem(m, sp) for m in m18.LEM_MODELS}
    cache = JsonPointCache(CACHE)
    planned = {
        (m, Z, e) for m in m18.OUR_MC.models() for Z, _ in m18.PAIRED_IONS.values()
        for e in m18.OUR_MC.energies(m, Z)
    }  # fmt: skip
    seen = set()
    for key in cache.keys():
        entry = cache[key]
        model, Z, A, e = entry["model"], entry["Z"], entry["A"], entry["E"]
        assert m18.mc_cache_key(model, Z, A, e, lems[model], sp) == key, (model, Z, e)
        seen.add((model, Z, e))
    assert seen == planned


def test_montecarlo_lq_repeats_fits_each_seed_like_survival() -> None:
    """Seeds 0..n-1, -ln S per seed, the inverse-spread weighted fit through the origin."""
    sp = SimpleBetheStoppingPower()
    lem = m18.survival_matched_lem("LEMI", sp)
    ion = Ion(8, 16, 1.0)
    doses = m18.OUR_MC.doses_Gy
    alphas, betas = m18.montecarlo_lq_repeats(
        lem, ion, doses, n_realizations=10, n_repeat=3, engine="numpy"
    )
    log_kill = np.array(
        [
            -np.log(
                lem.scholz1997_montecarlo_survival(
                    ion, doses, n_realizations=10, seed=s, engine="numpy"
                )
            )
            for s in range(3)
        ]
    )
    sigma = log_kill.std(axis=0, ddof=1)
    expected = np.array([fit_lq_weighted(doses, lk, sigma) for lk in log_kill]).T
    np.testing.assert_array_equal(alphas, expected[0])
    np.testing.assert_array_equal(betas, expected[1])


def test_adaptive_point_keeps_the_pilot_when_it_already_meets_the_targets() -> None:
    """A loose target needs no more than the pilot: the pilot's numbers are returned."""
    sp = SimpleBetheStoppingPower()
    lem = m18.survival_matched_lem("LEMI", sp)
    ion = Ion(8, 16, 1.0)
    loose = m18.OurMcSettings(
        n_repeat=3, n_pilot=10, n_min=5, n_max=50, target_rel_alpha=10.0, target_abs_beta=10.0
    )
    row = m18.adaptive_montecarlo_point(lem, ion, sp, loose, engine="numpy")
    alphas, betas = m18.montecarlo_lq_repeats(
        lem, ion, loose.doses_Gy, n_realizations=10, n_repeat=3, engine="numpy"
    )
    # The sized count (clipped up to n_min=5) is recorded even though the pilot's 10
    # realisations stand -- the old script's bookkeeping, which the cache carries.
    assert row["n_real"] == 5.0
    assert row["alpha"] == float(alphas.mean())
    assert row["beta_sigma"] == float(betas.std(ddof=1))
    assert row["let"] == sp.let_keV_um(ion)


# Rows of the original ``our_mc`` implementation, run with SimpleBethe and the numpy
# engine at the shrunken settings below: one point where the pilot stands, one that
# is re-sized and resolves tracks beyond the near/far cut as the background draw.
# Pinning them catches a regression in the sizing or the split.
_OLD_OUR_MC_ROWS = [
    (
        ("LEMI", 8, 16, 1.0),
        {
            "n_pilot": 10,
            "n_min": 5,
            "n_max": 50,
            "n_repeat": 3,
            "target_rel_alpha": 10.0,
            "target_abs_beta": 10.0,
        },
        {
            "let": 992.5341749658006,
            "alpha": -0.6151120986681421,
            "alpha_sigma": 0.15803539878828207,
            "beta": 0.4930286820118906,
            "beta_sigma": 0.023099849027246742,
            "n_real": 5.0,
        },
    ),
    (
        ("LEMI", 6, 12, 10.0),
        {
            "n_pilot": 4,
            "n_min": 2,
            "n_max": 8,
            "n_repeat": 3,
            "b_cut_um": 6.0,
            "b_cut_min_nbar": 0.0,
        },
        {
            "let": 161.98996628130988,
            "alpha": 1.4243467675779649,
            "alpha_sigma": 0.23044422116492222,
            "beta": 0.06453549411301737,
            "beta_sigma": 0.051794316821030716,
            "n_real": 8.0,
        },
    ),
]


@pytest.mark.parametrize(
    ("point", "overrides", "expected"), _OLD_OUR_MC_ROWS, ids=["pilot-stands", "sized-far-field"]
)
def test_adaptive_point_reproduces_the_old_our_mc_rows(
    point: tuple[str, int, int, float], overrides: dict[str, float], expected: dict[str, float]
) -> None:
    """``adaptive_montecarlo_point`` == the original ``our_mc`` row (numpy, seeds 0..2)."""
    from dataclasses import replace

    model, Z, A, energy = point
    sp = SimpleBetheStoppingPower()
    lem = m18.survival_matched_lem(model, sp)
    settings = replace(m18.OUR_MC, **overrides)
    row = m18.adaptive_montecarlo_point(lem, Ion(Z, A, energy), sp, settings, engine="numpy")
    assert set(row) == set(expected)
    for key, value in expected.items():
        assert row[key] == pytest.approx(value, rel=1e-9, abs=0.0), key
