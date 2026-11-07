"""Bloch, Mott and Lindhard-Sorensen corrections to the Bethe stopping logarithm.

Source: J. Lindhard and A.H. Sorensen, "Relativistic theory of stopping for heavy
ions", Phys. Rev. A 53, 2443 (1996), doi:10.1103/PhysRevA.53.2443 (LS below):
eq. (20) (Bloch's correction), the linear Mott correction Z1 alpha pi v / 2c quoted
on p. 2449, and eqs. (33), (34), (37), (41) (point nucleus).
"""

from __future__ import annotations

import csv
import math
from collections.abc import Callable
from pathlib import Path

import numpy as np
import pytest

from pyradbio.track.stopping.higher_order import (
    FINE_STRUCTURE,
    ahlen_cos_chi,
    ahlen_mott_correction,
    bloch_correction,
    lindhard_sorensen_correction,
    ls_phase_shift,
    ls_phase_shift_difference,
    mott_correction,
)


def _beta(gamma: float) -> float:
    return math.sqrt(1.0 - 1.0 / gamma**2)


# --- Bloch -----------------------------------------------------------------------------


def test_bloch_is_ls_eq_20_summed_over_l() -> None:
    """LS eq. (20): Delta L = sum_l [(l+1)/((l+1)^2 + eta^2) - 1/(l+1)], eta = Z alpha/beta."""
    for z, beta in ((1, 0.1), (6, 0.3), (18, 0.5)):
        eta = z * FINE_STRUCTURE / beta
        n = np.arange(1, 2_000_001, dtype=float)
        series = float(np.sum(n / (n * n + eta * eta) - 1.0 / n))
        assert bloch_correction(z, np.array([beta]))[0] == pytest.approx(series, abs=1e-9)


def test_bloch_small_eta_expansion() -> None:
    """LS p. 2446: 'For small values of k ... Delta L ~ -1.202 (k/2)^2' (k/2 = eta)."""
    beta = 0.9
    eta = FINE_STRUCTURE / beta
    assert bloch_correction(1, np.array([beta]))[0] == pytest.approx(-1.202 * eta**2, rel=2e-3)


# --- Mott ------------------------------------------------------------------------------


def test_mott_is_the_linear_term() -> None:
    beta = np.array([0.3, 0.9])
    np.testing.assert_allclose(
        mott_correction(6, beta), 6 * FINE_STRUCTURE * math.pi * beta / 2, rtol=1e-15
    )


# --- Lindhard-Sorensen -----------------------------------------------------------------


@pytest.mark.golden
@pytest.mark.parametrize(
    ("z", "gamma", "printed"),
    [
        (84, 100.0, 1.67),
        (92, 100.0, 1.88),
        (84, 1.5, 1.18),
        (92, 1.5, 1.29),
        (84, 1.2, 0.78),
        (92, 1.2, 0.82),
    ],
)
def test_ls_point_nucleus_reproduces_the_printed_values(
    z: int, gamma: float, printed: float
) -> None:
    """LS p. 2450 (doi:10.1103/PhysRevA.53.2443): Delta L - Delta L_NR for point nuclei,
    'For Z1 = 84 and 92, the value at gamma = 100 is 1.67 and 1.88 ... At gamma = 1.5 our
    values ... are 1.18 and 1.29 ... At gamma = 1.2 our values are 0.78 and 0.82'.
    Two printed decimals: agreement to their rounding (+-0.005, plus 1e-3 slack for
    the half-way case 0.825)."""
    beta = np.array([_beta(gamma)])
    value = lindhard_sorensen_correction(z, np.array([gamma]))[0] - bloch_correction(z, beta)[0]
    assert abs(value - printed) <= 0.006


def test_ls_vanishes_for_vanishing_charge() -> None:
    """LS p. 2449: 'we have produced a Delta L of zero (up to 1 part in 10^6) for Z1 -> 0'.

    It vanishes linearly: for small Z the leading term is the linear Mott term
    (1.0e-6 at Z = 1e-4), which is what remains, to 1e-8.
    """
    g = np.array([2.0])
    beta = np.array([_beta(2.0)])
    value = lindhard_sorensen_correction(1e-4, g)[0]
    assert abs(value) < 2e-6
    assert value == pytest.approx(mott_correction(1e-4, beta)[0], abs=1e-8)


def test_ls_tends_to_bloch_at_low_velocity() -> None:
    """LS Fig. 2: Delta L - Delta L_NR -> 0 for gamma -> 1, like the linear Mott term.

    MEASURED: 0.0248 / 0.0096 / 0.0026 at beta = 0.30 / 0.10 / 0.03 (linear Mott
    0.0210 / 0.0069 / 0.0022).
    """
    diffs = []
    for gamma in (1.05, 1.005, 1.0005):
        beta = np.array([_beta(gamma)])
        ls = lindhard_sorensen_correction(6, np.array([gamma]))[0]
        diffs.append(ls - bloch_correction(6, beta)[0])
        assert diffs[-1] == pytest.approx(mott_correction(6, beta)[0], rel=0.5)
    assert diffs[0] > diffs[1] > diffs[2] > 0.0


def test_closed_form_difference_matches_direct_phase_shifts() -> None:
    """Our eq.-(37) differences agree with differences of eq.-(33) phase shifts (mod pi)."""
    z, gamma = 18.0, 1.5
    beta = _beta(gamma)
    eta, az = FINE_STRUCTURE * z / beta, FINE_STRUCTURE * z
    k = np.array([2.0, 5.0, 40.0, -1.0, -9.0])
    direct = ls_phase_shift(k, eta, gamma, az) - ls_phase_shift(k - 1.0, eta, gamma, az)
    closed = ls_phase_shift_difference(k, eta, gamma, az)
    np.testing.assert_allclose(np.sin(direct) ** 2, np.sin(closed) ** 2, atol=1e-12)


def test_ls_for_moderate_charges_is_bloch_plus_linear_mott() -> None:
    """LS p. 2450: 'for moderate charge numbers, the stopping is approximately obtained
    by adding the nonrelativistic Delta L_NR and the linear Mott correction'."""
    for gamma in (1.1, 1.5, 2.0):
        beta = np.array([_beta(gamma)])
        ls = lindhard_sorensen_correction(6, np.array([gamma]))[0]
        approx = bloch_correction(6, beta)[0] + mott_correction(6, beta)[0]
        assert ls == pytest.approx(approx, abs=0.01)


def test_ls_phase_shift_differences_match_eq_37() -> None:
    """Eq. (37) is LS's closed form of delta_k - delta_(k-1); our phase shifts come from
    eqs. (33), (34). Phase shifts matter mod pi (they enter through sin^2)."""
    from scipy.special import loggamma

    z, gamma = 36.0, 2.0
    beta = _beta(gamma)
    eta, az = FINE_STRUCTURE * z / beta, FINE_STRUCTURE * z
    for k in (2, 3, 7, -1, -4):
        s_k, s_k1 = math.sqrt(k * k - az * az), math.sqrt((k - 1) ** 2 - az * az)
        eq37 = (
            0.5 * math.atan(eta * (s_k1 - s_k) / (s_k * s_k1 + eta * eta))
            + 0.5 * math.atan((eta / gamma) / (k * (k - 1) + (eta / gamma) ** 2))
            - loggamma(s_k + 1 + 1j * eta).imag
            + loggamma(s_k1 + 1 + 1j * eta).imag
            - 0.5 * math.pi * (s_k - s_k1 - k / abs(k))
        )
        ours = ls_phase_shift(float(k), eta, gamma, az) - ls_phase_shift(
            float(k - 1), eta, gamma, az
        )
        assert math.sin(ours) ** 2 == pytest.approx(math.sin(eq37) ** 2, abs=1e-12)


def test_ls_angular_momentum_tail_is_extrapolated() -> None:
    """The kappa sum converges like 1/kappa_max; the default extrapolation is converged."""
    g = np.array([2.0])
    default = lindhard_sorensen_correction(36, g)[0]
    brute = lindhard_sorensen_correction(36, g, kappa_max=200_000, extrapolate=False)[0]
    assert default == pytest.approx(brute, abs=5e-6)


def test_ls_rejects_point_charges_beyond_its_range() -> None:
    with pytest.raises(ValueError, match="alpha Z"):
        lindhard_sorensen_correction(140, np.array([2.0]))


# --- Ahlen 1980 (Rev. Mod. Phys. 52, 121): eq. (4.23), Table XII ------------------------


def test_shipped_cos_chi_reproduces_ahlen_table_xii(
    raw_source_file: Callable[[str, str], Path],
) -> None:
    path = raw_source_file("ahlen1980", "ahlen1980_table12.csv")
    with path.open(encoding="utf-8") as fh:
        rows = list(csv.DictReader(line for line in fh if not line.startswith("#")))
    y = np.array([float(r["abs_z1_alpha_over_beta"]) for r in rows])
    c = np.array([float(r["cos_chi"]) for r in rows])
    np.testing.assert_allclose(ahlen_cos_chi(y), c, rtol=1e-12)


def test_ahlen_g_is_eq_4_23_with_w0_zero() -> None:
    """Independent evaluation of G(Z1, beta)/2 at one point (cos chi from the shipped table)."""
    z, beta = 8, 0.4
    za = z * FINE_STRUCTURE
    cc = float(ahlen_cos_chi(za / beta))
    g = (
        za * beta * (1.725 + 0.52 * math.pi * cc)
        + za**2 * (3.246 - 0.451 * beta**2)
        + za**3 * (1.522 * beta + 0.987 / beta)
        + za**4 * (4.569 - 0.494 * beta**2 - 2.696 / beta**2)
        + za**5 * (1.254 * beta + 0.222 / beta - 1.170 / beta**3)
    )
    assert ahlen_mott_correction(z, np.array([beta]))[0] == pytest.approx(g / 2, rel=1e-12)


def test_ahlen_low_z_limit_is_near_the_linear_mott_term() -> None:
    """For small Z alpha, G/2 -> (Z alpha beta)(1.725 + 0.52 pi)/2: ~7% above pi Z alpha beta/2."""
    beta = np.array([0.5])
    ratio = ahlen_mott_correction(1, beta)[0] / mott_correction(1, beta)[0]
    assert 1.0 < ratio < 1.15


def test_ahlen_distinguishes_the_charge_sign_and_raises_outside_table_xii() -> None:
    beta = np.array([0.3])
    assert ahlen_mott_correction(6, beta)[0] != pytest.approx(ahlen_mott_correction(-6, beta)[0])
    with pytest.raises(ValueError, match="Table XII"):
        ahlen_cos_chi([2.5])
    with pytest.raises(ValueError, match="Table XII"):
        ahlen_mott_correction(18, np.array([0.05]))
