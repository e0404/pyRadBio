"""Walske's hydrogenic K/L shell corrections and the water assembly (track/stopping/shell.py).

Golden values: M.C. Walske, Phys. Rev. 88, 1283 (1952) [K] and Phys. Rev. 101, 940 (1956)
[L], printed tables and asymptotic formulae as cited per test. Walske states his numerics
"accurate to at least one percent" (1956, B_L; 1952 Table II).
"""

from __future__ import annotations

import numpy as np
import pytest

from pyradbio.track.stopping.base import OutOfRangeError
from pyradbio.track.stopping.shell import (
    _S_T,  # private: hydrogen's (S, T)
    DEFAULT_THETA_K,
    DEFAULT_THETA_L,
    WaterShellCorrection,
    hydrogen_shell_correction,
    walske_asymptotic_constants,
    walske_shell_correction,
    walske_stopping_number,
)

# --- the asymptotic constants: B = S ln(eta) + T - C -------------------------------------


@pytest.mark.parametrize(
    ("theta", "S", "T"),
    [  # Walske 1952 eq. (19): B_K(theta, eta) = S ln(eta) + T - U/eta - V/eta^2
        (0.7, 1.8133, 2.4603),
        (0.75, 1.7223, 2.4044),
        (0.8, 1.6457, 2.3462),
        (0.85, 1.5807, 2.2868),
        (0.9, 1.5250, 2.2273),
    ],
)
def test_k_shell_constants_are_walske_eq_19(theta: float, S: float, T: float) -> None:
    s, t = walske_asymptotic_constants("K", theta)
    assert s == pytest.approx(S, abs=5e-5)
    assert t == pytest.approx(T, abs=2e-4)


@pytest.mark.parametrize(
    ("theta", "S", "T"),
    [  # Walske 1956 eq. (11)
        (0.35, 10.0371, 28.1449),
        (0.45, 7.9116, 24.4501),
        (0.55, 6.7451, 21.9061),
        (0.65, 6.0345, 20.0154),
    ],
)
def test_l_shell_constants_are_walske_eq_11(theta: float, S: float, T: float) -> None:
    s, t = walske_asymptotic_constants("L", theta)
    assert s == pytest.approx(S, abs=5e-5)
    assert t == pytest.approx(T, abs=2e-4)


# --- C and B against Walske's tables -------------------------------------------------------


@pytest.mark.parametrize(
    ("theta", "values"),
    [  # Walske 1956 Table III, "C_L from exact calculation", eta_L = 3.5, 10, 25, 100, 1000
        (0.35, (0.4917, 0.1619, 0.06235, 0.01516, 0.001505)),
        (0.45, (0.5950, 0.1988, 0.07719, 0.01888, 0.001877)),
        (0.55, (0.6268, 0.2100, 0.08171, 0.02001, 0.001991)),
        (0.65, (0.6312, 0.2116, 0.08232, 0.02016, 0.002006)),
    ],
)
def test_c_l_is_walske_table_iii(theta: float, values: tuple[float, ...]) -> None:
    """MEASURED: <= 0.04% at eta 3.5-25, <= 0.21% at 100 and 1000."""
    c = walske_shell_correction("L", theta, [3.5, 10.0, 25.0, 100.0, 1000.0])
    np.testing.assert_allclose(c, values, rtol=4e-3)
    np.testing.assert_allclose(c[:3], values[:3], rtol=1e-3)


def test_c_l_at_eta_1_is_walske_table_iii() -> None:
    """MEASURED: -0.23% (theta 0.35) to +0.02%; Walske's B_L itself is good to 1%."""
    c = walske_shell_correction("L", 0.45, [1.0])
    assert c[0] == pytest.approx(2.0298, rel=1e-3)
    assert walske_shell_correction("L", 0.35, [1.0])[0] == pytest.approx(1.6934, rel=5e-3)


@pytest.mark.parametrize(
    ("eta", "values"),
    [  # Walske 1952 Table I, "computed values of C_K", theta = 0.7, 0.8, 0.9
        (20.0, (0.1153, 0.1180, 0.1186)),
        (10.0, (0.2407, 0.2460, 0.2472)),
    ],
)
def test_c_k_is_walske_table_i(eta: float, values: tuple[float, ...]) -> None:
    """MEASURED: -0.11% to -0.16% at eta 20, -1.6% at eta 10. At eta 10 Walske's own two
    routes to B_K (Table II vs S ln eta + T - C) differ by 0.1% of B_K = 0.006 = 2.5% of
    C_K, so 1.6% is within his accuracy; our B_K matches his Table II (next test)."""
    c = [walske_shell_correction("K", th, [eta])[0] for th in (0.7, 0.8, 0.9)]
    np.testing.assert_allclose(c, values, rtol=3e-3 if eta == 20.0 else 2e-2)


def test_b_k_and_b_l_are_walske_tables_ii() -> None:
    """MEASURED: B_K (1952 Table II) within 0.22%, B_L (1956 Table II) within 0.8%."""
    for eta, ref in ((1.5, (2.249, 2.031, 1.857)), (10.0, (6.406, 5.900, 5.496))):
        got = [walske_stopping_number("K", th, eta)[0] for th in (0.7, 0.8, 0.9)]
        np.testing.assert_allclose(got, ref, rtol=3e-3)
    for eta, ref in ((1.0, (26.52, 22.48, 19.80, 17.89)), (2.0, (34.09, 28.74, 25.31, 22.90))):
        got = [walske_stopping_number("L", th, eta)[0] for th in (0.35, 0.45, 0.55, 0.65)]
        np.testing.assert_allclose(got, ref, rtol=1e-2)


def test_hydrogen_atom_asymptotics_are_khandelwal_eq_18() -> None:
    """Two electrons, W_min = 0: B = 2 ln eta + 2.57861 - 2/eta - V/eta^2, so
    eta (C eta - 2) -> V. Walske 1952 eq. (17) prints V = 22/3, Khandelwal 1968 eq. (18)
    V = 25/3; the exact hydrogen atom settles it: MEASURED, V = 8.329 from a fit
    over eta = 100-1000 (8.176 / 8.239 / 8.273 from narrower, lower windows, rising
    toward 25/3). Walske's own V_K (eq. 19) inherit the same -1: Khandelwal's Table 1 V_K
    are Walske's + 1.000 at every theta."""
    s, t = _S_T("K", None)
    assert s == pytest.approx(2.0, abs=1e-9)
    assert t == pytest.approx(2.57861, abs=2e-5)
    eta = np.geomspace(100.0, 1000.0, 14)
    v_fit = np.polyfit(1 / eta, (2 * hydrogen_shell_correction(eta) * eta - 2.0) * eta, 2)[-1]
    assert v_fit == pytest.approx(25 / 3, abs=0.02)
    assert abs(v_fit - 22 / 3) > 0.9


# --- Khandelwal 1968, Nucl. Phys. A116, 97: Tables 1-3 ---------------------------------


@pytest.mark.parametrize(
    ("theta", "S", "T", "U"),
    [  # Table 1 (K shell): S_K, T_K, U_K
        (0.64, 1.9477, 2.5222, 1.9999),
        (0.72, 1.7754, 2.4388, 2.0817),
        (0.85, 1.5811, 2.2872, 2.1292),
        (0.95, 1.4772, 2.1686, 2.1271),
    ],
)
def test_k_shell_constants_are_khandelwal_table_1(
    theta: float, S: float, T: float, U: float
) -> None:
    """MEASURED over all 20 rows: S_K and T_K sit 0.00039 +- 0.00009 and
    0.00035 +- 0.00010 BELOW Khandelwal -- where ours equal Walske's eq. (19) to every
    printed digit, so the offset is his; U_K (fitted from our C at eta = 300-3000) within
    0.0031."""
    s, t = walske_asymptotic_constants("K", theta)
    assert s == pytest.approx(S, abs=1e-3)
    assert t == pytest.approx(T, abs=1e-3)
    eta = np.geomspace(300.0, 3000.0, 8)
    u_fit = np.polyfit(1 / eta, walske_shell_correction("K", theta, eta) * eta, 2)[-1]
    assert u_fit == pytest.approx(U, abs=6e-3)


@pytest.mark.parametrize(
    ("theta", "S", "T"),
    [  # Table 2 (L shell, 8 electrons): S_L, T_L
        (0.24, 15.3343, 35.0669),
        (0.40, 8.8005, 26.1061),
        (0.50, 7.2506, 23.0771),
        (0.66, 5.9792, 19.8546),
    ],
)
def test_l_shell_constants_are_khandelwal_table_2(theta: float, S: float, T: float) -> None:
    """MEASURED over all 26 rows: S_L within 1e-4, T_L within 0.0059 (3e-4 relative)."""
    s, t = walske_asymptotic_constants("L", theta)
    assert s == pytest.approx(S, abs=2e-4)
    assert t == pytest.approx(T, rel=6e-4)


KHANDELWAL_TABLE_3_THETAS = (0.95, 0.85, 0.75, 0.64)
KHANDELWAL_TABLE_3 = {  # eta_K: B_K at the thetas above (Table 3, K-shell stopping number)
    0.005: (1.34782e-08, 3.13778e-08, 8.00602e-08, 2.55370e-07),
    0.02: (9.43891e-06, 1.96439e-05, 4.30763e-05, 1.08909e-04),
    0.1: (6.14216e-03, 9.40534e-03, 1.44371e-02, 2.31795e-02),
    0.5: (3.92181e-01, 4.57610e-01, 5.36554e-01, 6.44677e-01),
    1.0: (1.12850, 1.25257, 1.40036, 1.60140),
    5.0: (4.07599, 4.36172, 4.71316, 5.21181),
    10.0: (5.32590, 5.68353, 6.13015, 6.77535),
}


@pytest.mark.parametrize("eta", sorted(KHANDELWAL_TABLE_3))
def test_b_k_is_khandelwal_table_3(eta: float) -> None:
    """MEASURED over the whole table (580 cells): median 1.8e-5, max 0.22% on
    the 552 cells without a defect; this subset max 0.062%. Khandelwal: B_K "should be good
    to four figures only". Defects found (his values, not OCR -- checked on the page
    images): the eta = 2 row is low by 0.26-0.36% against his own neighbouring rows at
    every theta (ours: 0.002%); at eta = 7, theta <= 0.68 is high by 0.73-0.76% against
    them; eta = 0.01 / theta = 0.68 repeats the theta = 0.70 entry; eta = 0.06 / 0.78
    and eta = 0.1 / 0.76 are isolated outliers (+0.47%, -0.36%)."""
    got = [walske_stopping_number("K", th, eta)[0] for th in KHANDELWAL_TABLE_3_THETAS]
    np.testing.assert_allclose(got, KHANDELWAL_TABLE_3[eta], rtol=1.5e-3)


def test_corrections_vanish_at_high_velocity_and_turn_negative_at_low() -> None:
    """Walske 1956 sec. V: C becomes large and negative at low enough energies."""
    assert walske_shell_correction("K", 0.7, [0.03])[0] < -2.0
    assert 0.0 < walske_shell_correction("K", 0.7, [1e4])[0] < 3e-4


def test_invalid_shell_arguments_raise() -> None:
    with pytest.raises(ValueError, match="shell"):
        walske_shell_correction("M", 0.5, [1.0])  # type: ignore[arg-type]
    with pytest.raises(ValueError, match="theta"):
        walske_shell_correction("K", 1.5, [1.0])
    with pytest.raises(ValueError, match="eta"):
        hydrogen_shell_correction([0.0])


# --- water ------------------------------------------------------------------------------


def test_water_assembly_is_the_documented_sum() -> None:
    w = WaterShellCorrection(evaluation="direct")
    b2 = np.array([0.01])
    c = w.components(b2)
    y = b2 / (1 / 137.035999084) ** 2
    np.testing.assert_allclose(c["C_H"], hydrogen_shell_correction(y))
    np.testing.assert_allclose(c["C_K"], walske_shell_correction("K", DEFAULT_THETA_K, y / 7.7**2))
    np.testing.assert_allclose(c["C_L"], walske_shell_correction("L", DEFAULT_THETA_L, y / 3.85**2))
    expected = (2 * c["C_H"] + c["C_K"] + 0.75 * c["C_L"]) / 10
    np.testing.assert_allclose(w.c_over_z(b2), expected)


def test_shipped_table_interpolates_the_direct_computation() -> None:
    """Interpolation by default. MEASURED: max |dC/Z| = 2.1e-7 over the whole range."""
    w, wd = WaterShellCorrection(), WaterShellCorrection(evaluation="direct")
    b2 = np.geomspace(w.beta2_range()[0] * 1.01, 0.99, 12)
    np.testing.assert_allclose(w.c_over_z(b2), wd.c_over_z(b2), atol=1e-6, rtol=0)


def test_other_thetas_are_computed_on_first_use() -> None:
    b2 = np.array([0.002, 0.02])
    w = WaterShellCorrection(theta_K=0.8)  # only C_K is recomputed (C_H, C_L are shipped)
    wd = WaterShellCorrection(theta_K=0.8, evaluation="direct")
    np.testing.assert_allclose(w.c_over_z(b2), wd.c_over_z(b2), atol=1e-6, rtol=0)
    assert not np.allclose(w.c_over_z(b2), WaterShellCorrection().c_over_z(b2))


def test_water_range_and_options() -> None:
    w = WaterShellCorrection()
    lo, _ = w.beta2_range()
    with pytest.raises(OutOfRangeError):
        w.c_over_z([lo * 0.5])
    with pytest.raises(ValueError, match="theta_L"):
        WaterShellCorrection(theta_L=0.0)
    with pytest.raises(ValueError, match="l_shell_electrons"):
        WaterShellCorrection(l_shell_electrons=9)
    with pytest.raises(ValueError, match="evaluation"):
        WaterShellCorrection(evaluation="exact")  # type: ignore[arg-type]
    assert "theta=0.7" in w.describe()
