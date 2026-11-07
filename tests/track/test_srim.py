"""SRIM ``<Ion>_Srim.dat`` reader (the Survival code's stopping-power source).

Parsing is tested on an inline fixture so it runs everywhere; the tests that
need the real tables skip cleanly when ``PYRADBIO_SURVIVAL_SRC_DIR`` is unset.
"""

from __future__ import annotations

from itertools import pairwise

import pytest

from pyradbio.core.radiation import Ion
from pyradbio.track import srim

# Four rows lifted verbatim from C_Srim.dat (header + data), enough to exercise
# the column layout, the tab/space mix, and both E-notation cases.
_FIXTURE = """   Ion        dE/dx       dE/dx      Projected    Longitudinal   Lateral
  Energy      Elec.      Nuclear        Range      Straggling   Straggling
 (MeV/u)     (MeV/um)   (MeV/um)        (um)          (um)         (um)
----------- ----------- ----------- ------------- ------------  ---------
8.333e-04   5.641E-02\t9.014E-02\t 0.0566       0.0209\t  0.0152
1.000e-03   6.180E-02\t8.543E-02\t 0.0677       0.0242\t  0.0177
1.000e+00   2.000E-01\t1.000E-03\t 1.0000       0.1000\t  0.1000
1.000e+01   5.000E-02\t1.000E-05\t 9.0000       0.5000\t  0.4000
"""


def test_parse_reads_all_six_columns_and_skips_the_header() -> None:
    col = srim.parse_srim_table(_FIXTURE, "C")
    assert col.symbol == "C"
    assert col.Z == 6
    assert col.energy_MeV_u == (8.333e-4, 1.0e-3, 1.0, 10.0)
    assert col.electronic_MeV_um[0] == pytest.approx(5.641e-2)
    assert col.nuclear_MeV_um[0] == pytest.approx(9.014e-2)
    assert col.projected_range_um[-1] == pytest.approx(9.0)
    assert col.longitudinal_straggling_um[-1] == pytest.approx(0.5)
    assert col.lateral_straggling_um[-1] == pytest.approx(0.4)


def test_let_is_electronic_only_by_default_in_keV_per_um() -> None:
    """Survival drops the nuclear column; MeV/um -> keV/um is a factor 1e3."""
    col = srim.parse_srim_table(_FIXTURE, "C")
    assert col.let_keV_um()[0] == pytest.approx(56.41)
    assert col.let_keV_um(include_nuclear=True)[0] == pytest.approx(56.41 + 90.14)


def test_a_short_row_raises_rather_than_being_dropped() -> None:
    broken = _FIXTURE + "2.000e+01   1.0E-02\n"
    with pytest.raises(ValueError, match="expected 6 columns"):
        srim.parse_srim_table(broken, "C")


def test_an_empty_table_raises() -> None:
    with pytest.raises(ValueError, match="no data rows"):
        srim.parse_srim_table("\n".join(_FIXTURE.splitlines()[:4]), "C")


def test_unknown_element_symbol_is_rejected() -> None:
    with pytest.raises(ValueError, match="unknown element symbol"):
        srim.load_srim_column("Xx")


# ---- against the real tables (skipped when the checkout is not configured) --
def _tables_available() -> bool:
    try:
        srim.load_srim_column("C")
    except (FileNotFoundError, OSError):
        return False
    return True


requires_srim = pytest.mark.skipif(
    not _tables_available(),
    reason="Survival source checkout not configured (PYRADBIO_SURVIVAL_SRC_DIR)",
)


@requires_srim
def test_every_shipped_ion_parses_with_a_monotone_energy_grid() -> None:
    for symbol in srim.SRIM_IONS:
        col = srim.load_srim_column(symbol)
        assert len(col.energy_MeV_u) > 100, symbol
        assert all(b > a for a, b in pairwise(col.energy_MeV_u)), symbol
        assert all(v > 0.0 for v in col.electronic_MeV_um), symbol


@requires_srim
def test_srim_carbon_let_peaks_near_the_bragg_peak() -> None:
    """A physical screen on the table, independent of Survival: C peaks <1 MeV/u."""
    col = srim.load_srim_column("C")
    peak = max(range(len(col.energy_MeV_u)), key=lambda i: col.electronic_MeV_um[i])
    assert 0.1 < col.energy_MeV_u[peak] < 1.0


@requires_srim
def test_stopping_power_defaults_match_survivals_own_conventions() -> None:
    """Linear-in-energy interpolation lands exactly on a tabulated node."""
    sp = srim.srim_stopping_power(("C",))
    col = srim.load_srim_column("C")
    i = len(col.energy_MeV_u) // 2
    at_node = sp.let_keV_um(Ion(6, 12, col.energy_MeV_u[i]))
    assert at_node == pytest.approx(col.electronic_MeV_um[i] * 1e3, rel=1e-12)

    # Midway between two nodes the two interpolants must actually differ, else
    # the "match Survival's interpolation" default would be untested.
    mid = 0.5 * (col.energy_MeV_u[i] + col.energy_MeV_u[i + 1])
    lin = sp.let_keV_um(Ion(6, 12, mid))
    log = srim.srim_stopping_power(("C",), match_survival_interpolation=False).let_keV_um(
        Ion(6, 12, mid)
    )
    assert lin != pytest.approx(log, rel=1e-9)


@requires_srim
def test_common_energy_range_is_non_empty_and_covers_the_therapeutic_window() -> None:
    lo, hi = srim.srim_energy_range_MeV_u()
    assert lo < 1.0 < 400.0 < hi
