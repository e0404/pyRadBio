"""The Python port of MSTAR 3.12 (Paul & Schinner) and its use in XSTAR.

The decisive test compares the port with values produced by the ORIGINAL FORTRAN,
compiled by scripts/stopping/mstar_reference.py into the git-ignored data/stopping/
folder; it and every test needing MSTAR's database skip cleanly when absent.
"""

from __future__ import annotations

from collections.abc import Callable
from pathlib import Path

import numpy as np
import pytest
from scipy.interpolate import CubicSpline

import pyradbio.track.stopping.mstar.paul as paul
from pyradbio.track.stopping.base import OutOfRangeError
from pyradbio.track.stopping.masses import most_abundant_mass_number
from pyradbio.track.stopping.mstar import (
    MODE_H_IONS,
    MSTAR,
    AlphaSpline,
    MSTARDatabase,
    MSTARDatabaseError,
    MSTARModeError,
    mstar_stopping,
    resolve_mode,
)
from pyradbio.track.stopping.mstar.spline import msspl1, msspl2
from pyradbio.track.stopping.nuclear import WaterNuclearStopping
from pyradbio.track.stopping.xstar import MSTAR_DATABASE_ENV, XSTAR
from tests.conftest import requires_pymkm

DB = ("mstar", "MSTAR312/Sources/msdbs1.d")
REFERENCE = ("mstar", "reference/mstar_water_reference.txt")


@pytest.fixture
def database(raw_source_file: Callable[[str, str], Path]) -> Path:
    return raw_source_file(*DB)


@pytest.fixture
def alpha(database: Path) -> AlphaSpline:
    return AlphaSpline(MSTARDatabase.from_file(database).target(276))


def _reference(path: Path) -> list[tuple[int, str, str, float, float, int]]:
    rows = []
    for line in path.read_text().splitlines():
        if line.startswith("#"):
            continue
        z, mode_in, mode_used, e, ste, rc = line.split()
        rows.append((int(z), mode_in, mode_used, float(e), float(ste), int(rc)))
    return rows


# --- the port against the original program ---------------------------------------------


def test_port_reproduces_the_original_fortran(
    raw_source_file: Callable[[str, str], Path], alpha: AlphaSpline
) -> None:
    """Every Z 3-18 x mode a/b/c/d/g/h x (database energies and their midpoints).

    MEASURED (25,248 calls; gfortran 13.1 build of the authors' sources):
    24,154 of 24,196 values bit-identical, max relative deviation 2.0e-15; all return
    codes (incl. the 1,052 refused mode-'h' calls for Mg, Al, Si, P) and all
    resolved modes agree. Bound: a few ulp.
    """
    rows = _reference(raw_source_file(*REFERENCE))
    assert len(rows) == 25_248
    worst = 0.0
    for z, mode_in, mode_used, e, ste, rc in rows:
        if rc == 4:
            with pytest.raises(MSTARModeError):
                mstar_stopping(alpha, mode_in, z, e)
            continue
        assert rc == 0
        value, used = mstar_stopping(alpha, mode_in, z, e)
        assert used == mode_used
        worst = max(worst, abs(value / ste - 1.0))
    assert worst < 1e-14


def test_single_precision_constants_are_what_makes_it_exact(
    raw_source_file: Callable[[str, str], Path],
    alpha: AlphaSpline,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """Treating MSTAR's REAL*4 constants as double precision is visibly wrong (~2e-7)."""
    monkeypatch.setattr(paul, "_r4", lambda x: x)
    rows = [r for r in _reference(raw_source_file(*REFERENCE)) if r[2] == "c" and r[5] == 0]
    worst = max(abs(mstar_stopping(alpha, "c", z, e)[0] / ste - 1.0) for z, _, _, e, ste, _ in rows)
    assert worst > 1e-8


@requires_pymkm
def test_port_agrees_with_pymkms_mstar_table_in_mode_b(database: Path) -> None:
    """Independent cross-check, Z 3-10 (pyMKM bundles an MSTAR 3.12 water table).

    MEASURED: mode 'b' agrees to <= 5.2e-6 (pyMKM prints 5 digits); mode 'c' differs
    by up to 5.9e-3, so pyMKM's table is mode 'b'.
    """
    import pymkm

    root = Path(pymkm.__file__).parent / "data" / "defaults" / "mstar_3_12"
    mode_b, mode_c = MSTAR(database, mode="b"), MSTAR(database, mode="c")
    for z in range(3, 11):
        (path,) = root.glob(f"Z{z:02d}_*.txt")
        rows = [line.split() for line in path.read_text().splitlines() if line[:1].isdigit()]
        e = np.array([float(r[0]) for r in rows])
        s = np.array([float(r[1]) for r in rows])  # MeV/cm at 1 g/cm3 = MeV cm2/g
        a = most_abundant_mass_number(z)
        assert np.abs(mode_b.mass_stopping_power(z, a, e) / s - 1.0).max() < 1e-5
        assert np.abs(mode_c.mass_stopping_power(z, a, e) / s - 1.0).max() > 1e-3


# --- building blocks --------------------------------------------------------------------


def test_spline_is_a_natural_cubic_spline() -> None:
    x = [0.5, 1.0, 1.7, 3.0, 4.2, 6.0]
    y = [2.0, 3.5, 3.1, 1.2, 0.9, 1.5]
    ypp = msspl1(x, y)
    ref = CubicSpline(x, y, bc_type="natural")
    for xs in np.linspace(0.5, 6.0, 37):
        assert msspl2(x, y, ypp, float(xs)) == pytest.approx(float(ref(xs)), rel=1e-13)
    assert ypp[0] == ypp[-1] == 0.0


def test_database_parser_accepts_the_layout(tmp_path: Path) -> None:
    path = tmp_path / "db.d"
    path.write_text(
        "# header\n1.0E-03\n2.0E-03\n3.0E-03\n4.0E-03\n"
        "  276 'WATER, LIQUID'\n1.5E-01\n2.5E-01\n3.5E-01\n4.5E-01\n"
        "  1 'HYDROGEN'\n1.0E+00\n2.0E+00\n3.0E+00\n4.0E+00\n"
    )
    db = MSTARDatabase.from_file(path)
    table = db.target(276)
    assert table.name == "WATER, LIQUID"
    assert table.energy_MeV == (1e-3, 2e-3, 3e-3, 4e-3)
    assert table.stopping_MeV_cm2_mg[-1] == 0.45
    with pytest.raises(KeyError):
        db.target(277)


@pytest.mark.parametrize(
    "body",
    [
        "1.0\n2.0\n3.0\n4.0\n 276 'W'\n1.0\n2.0\n",  # too few values
        "1.0\n2.0\n3.0\n4.0\n 276 'W'\n1.0\n2.0\n3.0\n-4.0\n",  # non-positive
        "1.0\n3.0\n2.0\n4.0\n 276 'W'\n1.0\n2.0\n3.0\n4.0\n",  # not increasing
        "1.0\n2.0\n3.0\n4.0\n 0 'W'\n1.0\n2.0\n3.0\n4.0\n",  # bad id
        "1.0\n2.0\n3.0\n4.0\n 276 'W'\n1.0\n2.0\n3.0\n4.0\n 276 'X'\n1.0\n2.0\n3.0\n4.0\n",
        "1.0\n2.0\n3.0\n4.0\n 276 'W'\n1.0\n2.0\n3.0\n4.0\n 1 'H'\n1.0\n",  # unequal
        "1.0\n2.0\n3.0\n4.0\n 276 'W'\n1.0 2.0\n",  # unreadable
        "1.0\n2.0\n3.0\n",  # no targets
    ],
)
def test_database_parser_rejects_corrupt_files(tmp_path: Path, body: str) -> None:
    path = tmp_path / "db.d"
    path.write_text(body)
    with pytest.raises(MSTARDatabaseError):
        MSTARDatabase.from_file(path)


def test_mode_resolution_follows_mstar1() -> None:
    assert resolve_mode("a") == ("c", False)
    assert resolve_mode("B") == ("d", True)
    assert resolve_mode("b", target_is_gas=True) == ("h", True)
    assert resolve_mode("a", target_is_gas=True) == ("g", False)
    assert resolve_mode("g") == ("g", False)
    with pytest.raises(ValueError, match="mstar_mode"):
        resolve_mode("x")


def test_paul_rejects_what_the_original_rejects() -> None:
    with pytest.raises(MSTARModeError):
        paul.mspaul("h", 12, 1.0)  # no mode-'h' fit for Mg
    with pytest.raises(MSTARModeError):
        paul.mspaul("h", 6, 1.0, z2=1.0)  # mode 'h' blocked for H/He targets
    with pytest.raises(MSTARModeError, match="not ported"):
        paul.mspaul("d", 6, 1.0, z2=29.0)  # only water's Z2 branch is ported
    with pytest.raises(ValueError, match="3-18"):
        paul.mstar1(1.0, "b", 19, 1.0)


def test_mode_b_falls_back_to_c_where_d_is_unavailable(monkeypatch: pytest.MonkeyPatch) -> None:
    """MSTAR1 retries 'b' with 'c' after MSPAUL's RC=4 (not reachable for liquid water)."""
    real = paul.mspaul

    def no_d(mode: str, z1: int, e: float, z2: float = paul.WATER_LIQUID_DZ2) -> float:
        if mode == "d":
            raise MSTARModeError("no d")
        return real(mode, z1, e, z2)

    monkeypatch.setattr(paul, "mspaul", no_d)
    value, used = paul.mstar1(1.0, "b", 6, 1.0)
    assert used == "c"
    assert value == real("c", 6, 1.0) * 36.0 / 4.0
    with pytest.raises(MSTARModeError):
        paul.mstar1(1.0, "d", 6, 1.0)


# --- the MSTAR model class -------------------------------------------------------------


def test_mstar_model_range_modes_and_metadata(database: Path) -> None:
    model = MSTAR(database)
    assert model.energy_range(6, 12) == (0.00025, 1000.0)
    assert model.mode_used(6, 1.0) == "d"
    info = model.metadata(6, 12)
    assert (info.source, info.i_value_eV) == ("MSTAR 3.12", 75.0)
    assert info.options["evaluation"] == "direct"
    with pytest.raises(OutOfRangeError):
        model.mass_stopping_power(6, 12, 1000.1)
    with pytest.raises(MSTARModeError):
        MSTAR(database, mode="h").mass_stopping_power(12, 24, 1.0)
    assert 12 not in MODE_H_IONS


def test_mstar_model_velocity_scales_other_isotopes(database: Path) -> None:
    model = MSTAR(database)
    assert model.metadata(6, 13).options["velocity_scaled_from"] == "A=12"
    lo, hi = model.energy_range(6, 13)
    assert hi > 1000.0  # a C-13 nucleon is heavier than a C-12 one: same velocity, higher E/A
    assert model.mass_stopping_power(6, 13, [lo, hi]).shape == (2,)


def test_alpha_spline_raises_outside_the_database(alpha: AlphaSpline) -> None:
    lo, hi = alpha.energy_range_MeV
    alpha(lo)
    with pytest.raises(OutOfRangeError):
        alpha(hi * 1.0001)


# --- XSTAR with MSTAR heavy ions --------------------------------------------------------


def test_xstar_mstar_interpolation_error_at_the_originals_energies(
    raw_source_file: Callable[[str, str], Path], database: Path
) -> None:
    """Shipped tables (port values at 50 nodes/decade) vs the port at the reference
    energies (database grid + midpoints, i.e. off our nodes), modes 'c' and 'd'.

    MEASURED: max 6.8e-6 over Z 3-18 and both modes. Bound ~2x.
    """
    energies = sorted({r[3] for r in _reference(raw_source_file(*REFERENCE))})
    e = np.array(energies)
    worst = 0.0
    for mode in ("a", "b"):
        shipped, direct = XSTAR(mstar_mode=mode), MSTAR(database, mode=mode)
        for z in range(3, 19):
            a = most_abundant_mass_number(z)
            rel = shipped.mass_stopping_power(z, a, e) / direct.mass_stopping_power(z, a, e) - 1
            worst = max(worst, float(np.abs(rel).max()))
    print(f"\nXSTAR(MSTAR) shipped-table interpolation error, max over Z 3-18: {worst:.2e}")
    assert worst < 1.4e-5


def test_xstar_direct_evaluation_is_the_port(database: Path) -> None:
    direct = XSTAR(mstar_evaluation="direct", mstar_database=database)
    e = np.geomspace(0.001, 900.0, 11)
    np.testing.assert_array_equal(
        direct.mass_stopping_power(8, 16, e), MSTAR(database).mass_stopping_power(8, 16, e)
    )
    assert direct.energy_range(8, 16) == (0.00025, 1000.0)
    info = direct.metadata(8, 16)
    assert info.options["mstar_evaluation"] == "direct"
    assert info.options["grid"] is None
    # p and He still come from the shipped PSTAR/ASTAR tables
    np.testing.assert_array_equal(
        direct.mass_stopping_power(1, 1, e), XSTAR().mass_stopping_power(1, 1, e)
    )


@pytest.mark.parametrize("mode", ["g", "h", "b", "c"])
def test_xstar_direct_metadata_comes_from_the_port(database: Path, mode: str) -> None:
    """Regression: with direct gas modes, metadata() must not read the unshipped mstar_g
    table (FileNotFoundError); direct evaluation reports the port's provenance."""
    direct = XSTAR(mstar_evaluation="direct", mstar_mode=mode, mstar_database=database)
    port = MSTAR(database, mode=mode).metadata(6, 12)
    info = direct.metadata(6, 12)
    assert info.source == f"MSTAR 3.12 (mode {direct.mstar_mode_used})"
    assert (info.reference, info.edition, info.i_value_eV) == (
        port.reference,
        port.edition,
        port.i_value_eV,
    )
    assert info.energy_range_MeV_u == direct.energy_range(6, 12)
    assert info.options["mstar_evaluation"] == "direct" and info.options["grid"] is None
    assert info.options["mstar_mode"] == mode
    total = XSTAR(
        mstar_evaluation="direct", mstar_mode=mode, mstar_database=database, quantity="total"
    ).metadata(6, 12)
    assert total.options["nuclear"] == XSTAR(quantity="total").nuclear.describe()


def test_xstar_direct_evaluation_reads_the_environment(
    database: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    monkeypatch.setenv(MSTAR_DATABASE_ENV, str(database))
    assert XSTAR(mstar_evaluation="direct", mstar_mode="g").mass_stopping_power(6, 12, 1.0)[0] > 0


def test_xstar_direct_evaluation_needs_a_database(monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.delenv(MSTAR_DATABASE_ENV, raising=False)
    with pytest.raises(FileNotFoundError, match=r"msdbs1\.d"):
        XSTAR(mstar_evaluation="direct")


def test_xstar_gas_modes_need_direct_evaluation() -> None:
    for mode in ("g", "h"):
        with pytest.raises(ValueError, match="no shipped table"):
            XSTAR(mstar_mode=mode)
    with pytest.raises(ValueError, match="mstar_evaluation"):
        XSTAR(mstar_evaluation="exact")  # type: ignore[arg-type]


def test_xstar_mstar_metadata_and_ranges() -> None:
    xstar = XSTAR()
    info = xstar.metadata(6, 12)
    assert info.source == "MSTAR 3.12 (mode d)"
    assert info.i_value_eV == 75.0
    assert info.options["mstar_mode"] == "b"
    assert info.options["mstar_mode_used"] == "d"
    assert XSTAR(mstar_mode="a").metadata(6, 12).source == "MSTAR 3.12 (mode c)"
    assert xstar.energy_range(6, 12) == (0.00025, 1000.0)
    assert xstar.metadata(6, 13).options["velocity_scaled_from"] == "C-12"


def test_total_for_mstar_ions_adds_the_nuclear_model() -> None:
    e = np.array([0.003, 0.3, 30.0])
    total = XSTAR(quantity="total").mass_stopping_power(6, 12, e)
    el = XSTAR().mass_stopping_power(6, 12, e)
    np.testing.assert_allclose(
        total,
        el + WaterNuclearStopping(proton_potential="moliere").mass_stopping_power(6, 12, e),
        rtol=1e-12,
    )


def test_total_for_direct_mstar_adds_the_nuclear_model(database: Path) -> None:
    e = np.array([0.003, 0.3])
    xs = XSTAR(quantity="total", mstar_evaluation="direct", mstar_database=database)
    el = XSTAR(mstar_evaluation="direct", mstar_database=database).mass_stopping_power(6, 12, e)
    nuc = WaterNuclearStopping(proton_potential="moliere").mass_stopping_power(6, 12, e)
    np.testing.assert_allclose(xs.mass_stopping_power(6, 12, e), el + nuc, rtol=1e-12)


def test_mstar_class_total_defaults_to_zbl(database: Path) -> None:
    e = np.array([0.003, 0.3])
    total = MSTAR(database, quantity="total")
    el = MSTAR(database).mass_stopping_power(8, 16, e)
    nuc = WaterNuclearStopping().mass_stopping_power(8, 16, e)
    np.testing.assert_allclose(total.mass_stopping_power(8, 16, e), el + nuc, rtol=1e-12)
    assert "zbl" in str(total.metadata(8, 16).options["nuclear"])
    with pytest.raises(ValueError, match="quantity"):
        MSTAR(database, quantity="nuclear")  # type: ignore[arg-type]
