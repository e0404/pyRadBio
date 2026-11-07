"""Barkas correction of Ashley, Ritchie & Brandt via ICRU 49 eq. 2.8 (track/stopping/barkas.py)."""

from __future__ import annotations

import csv
import math
from collections.abc import Callable
from pathlib import Path

import numpy as np
import pytest

from pyradbio.track.stopping.barkas import (
    ARB_Q,
    B_VALUES,
    ICRU49_GAMMA,
    WaterBarkasCorrection,
    arb_function,
    barkas_l1,
    f_from_kappa,
)
from pyradbio.track.stopping.base import OutOfRangeError
from pyradbio.track.stopping.bethe import Bethe

ALPHA = 1 / 137.035999084


def test_shipped_f_reproduces_the_arb_table_at_its_points(
    raw_source_file: Callable[[str, str], Path],
) -> None:
    """Regridded table vs the transcribed ARB 1973 Table I (x >= 1). MEASURED: <= 1.9e-6."""
    path = raw_source_file("arb1973", "arb1973_table1.csv")
    with path.open(encoding="utf-8") as fh:
        rows = list(csv.DictReader(line for line in fh if not line.startswith("#")))
    x = np.array([float(r["x"]) for r in rows])
    keep = x >= 1.0
    for b in B_VALUES:
        kappa = np.array([float(r[f"kappa_b{b:.1f}"]) for r in rows])
        expected = f_from_kappa(kappa[keep], x[keep])
        np.testing.assert_allclose(arb_function(b, x[keep]), expected, rtol=5e-6)


def test_f_is_kappa_times_root_x_times_l() -> None:
    """ARB 1973 eqs. (2), (3): kappa = F / (x^1/2 L), L = ln(x/q), q = 0.18."""
    assert ARB_Q == 0.18
    assert f_from_kappa(0.3090, 10.0)[()] == pytest.approx(
        0.3090 * math.sqrt(10) * math.log(10 / 0.18)
    )


def test_l1_is_icru49_eq_2_8() -> None:
    x = np.array([2.0, 50.0])
    f = arb_function(1.8, x)
    np.testing.assert_allclose(barkas_l1(8, x, gamma=1.29), 1.29 * f / (math.sqrt(8) * x**1.5))


def test_f_continues_linearly_in_ln_x_beyond_the_table() -> None:
    lx = np.log([100.0, 300.0, 1000.0, 1e4])
    f = arb_function(1.8, np.exp(lx))
    slopes = np.diff(f[1:]) / np.diff(lx[1:])
    assert slopes == pytest.approx(slopes[0])
    assert 2.0 < slopes[0] < 3.5  # the printed table's own last-decade slope is ~2.6


def test_water_l1_is_electron_weighted_and_stops_at_x_o_1() -> None:
    w = WaterBarkasCorrection()
    b2 = np.array([0.002, 0.05])
    y = b2 / ALPHA**2
    expected = 0.2 * barkas_l1(1, y) + 0.8 * barkas_l1(8, y / 8)
    np.testing.assert_allclose(w.l1(b2), expected)
    lo, _ = w.beta2_range()
    assert lo == pytest.approx(8 * ALPHA**2)
    w.l1([lo * 1.0001])
    with pytest.raises(OutOfRangeError):
        w.l1([lo * 0.9])


def test_options_and_validation() -> None:
    assert ICRU49_GAMMA == 1.29
    assert "gamma=1.29" in WaterBarkasCorrection(gamma=1.29).describe()
    with pytest.raises(ValueError, match="b must"):
        WaterBarkasCorrection(b_O=1.4)
    with pytest.raises(ValueError, match="gamma"):
        WaterBarkasCorrection(gamma=0.0)
    with pytest.raises(ValueError, match="b must"):
        arb_function(1.7, [2.0])


def test_gamma_one_is_what_the_evaluated_alpha_tables_need(
    raw_source_file: Callable[[str, str], Path],
) -> None:
    """Why gamma defaults to 1 (barkas.py docstring). With the shell thetas fitted to
    protons only, alphas (z L1 twice as large relative to L) discriminate. MEASURED
    ASTAR 0.5-1 MeV/u: gamma 1 -> -0.14%, gamma 1.29 -> +2.4%."""
    with raw_source_file("astar", "astar_water_liquid.csv").open() as fh:
        rows = list(csv.DictReader(line for line in fh if not line.startswith("#")))
    e = np.array([float(r["T_MeV"]) for r in rows]) / 4
    s = np.array([float(r["S_el_MeV_cm2_g"]) for r in rows])
    keep = (e >= 0.5) & (e < 1.0)
    dev = {}
    for gamma in (1.0, ICRU49_GAMMA):
        bethe = Bethe(
            i_value_eV=75.0,
            min_beta_gamma=0.02,
            shell=True,
            barkas=WaterBarkasCorrection(gamma=gamma),
            bloch=True,
        )
        dev[gamma] = np.abs(bethe.mass_stopping_power(2, 4, e[keep]) / s[keep] - 1).max()
    assert dev[1.0] < 3e-3
    assert dev[ICRU49_GAMMA] > 2e-2
