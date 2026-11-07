"""Northcliffe & Schilling 1970 water stopping-power table.

Reference: Northcliffe, L.C. & Schilling, R.F. 1970, Nuclear Data Tables A7,
233-463, doi:10.1016/S0092-640X(70)80016-X.

These tests are the standing guard on a table transcribed from a 1970 scan: they
re-run the physical checks that validated the transcription, so a future edit
that corrupts a cell fails here rather than silently changing an LET.
"""

from __future__ import annotations

from itertools import pairwise

import numpy as np
import pytest

from pyradbio.core.radiation import Ion
from pyradbio.track.northcliffe_schilling import (
    ENERGY_RANGE_MEV_U,
    BetheOnly,
    NorthcliffeSchillingWithBetheTail,
    build_stopping_power,
    load_water_columns,
    northcliffe_schilling1970_water,
)

# The source's fixed 38-point grid (its 6-6-6-6-6-6-2 visual row grouping).
GRID_MEV_U = [
    0.0125,
    0.0160,
    0.0200,
    0.0250,
    0.0320,
    0.0400,
    0.0500,
    0.0600,
    0.0700,
    0.0800,
    0.0900,
    0.1000,
    0.1250,
    0.1600,
    0.2000,
    0.2500,
    0.3200,
    0.4000,
    0.5000,
    0.6000,
    0.7000,
    0.8000,
    0.9000,
    1.0000,
    1.2500,
    1.6000,
    2.0000,
    2.5000,
    3.2000,
    4.0000,
    5.0000,
    6.0000,
    7.0000,
    8.0000,
    9.0000,
    10.0000,
    11.0000,
    12.0000,
]

# NIST PSTAR electronic stopping power, liquid water [MeV cm^2/g].
PSTAR_PROTON = {5.0: 79.11, 8.0: 54.55, 10.0: 45.67, 12.0: 39.53}
PSTAR_PROTON_2_MEV_U = 159.8  # where the 1970 compilation starts to drift


def test_table_covers_every_ion_the_source_tabulates() -> None:
    columns = load_water_columns()

    assert sorted(columns) == list(range(1, 104))
    assert {c.source for c in columns.values()} == {"scan", "ocr"}
    # Z=100's PDF page has no text layer at all, so it can only be scan-read.
    assert columns[100].source == "scan"
    for column in columns.values():
        assert list(column.energy_MeV_u) == pytest.approx(GRID_MEV_U)
        assert len(column.mass_stopping_power_MeV_cm2_g) == len(GRID_MEV_U)
        assert all(v > 0.0 for v in column.mass_stopping_power_MeV_cm2_g)


# The lanthanides print a flat, faintly double-humped top: La reads 95.939,
# 95.779, 95.599, 95.668 across 0.8-1.25 MeV/u. That is in the source, not an
# artifact -- La and Ce were read off the scan and Pr was re-checked at 700 dpi.
# So the shape test allows a plateau wobble but nothing larger.
PLATEAU_WOBBLE = 0.005


def test_every_column_rises_to_a_single_bragg_peak() -> None:
    """A stopping-power curve rises, peaks once, and falls.

    This is what caught the OCR digit errors during transcription: a corrupted
    cell breaks monotonicity by far more than the printed plateau wobble.
    """
    for z, column in load_water_columns().items():
        values = np.asarray(column.mass_stopping_power_MeV_cm2_g)
        peak = int(np.argmax(values))
        tolerance = PLATEAU_WOBBLE * values[peak]
        assert 0 < peak < len(values) - 1, f"Z={z} peaks at an endpoint"
        assert np.all(np.diff(values[: peak + 1]) > -tolerance), f"Z={z} dips while rising"
        assert np.all(np.diff(values[peak:]) < tolerance), f"Z={z} climbs after its peak"


def test_no_column_deviates_more_than_real_curvature_in_energy() -> None:
    """Log-log three-point smoothness, the screen that found the OCR errors.

    A mis-read digit shows up here at 0.1-0.4; genuine curvature near the Bragg
    peak tops out at 0.0294 across all 103 columns -- the same bound the columns
    read off the scan reach on their own.
    """
    columns = load_water_columns()
    log_energy = np.log(np.asarray(GRID_MEV_U))
    worst = 0.0
    for column in columns.values():
        ln = np.log(np.asarray(column.mass_stopping_power_MeV_cm2_g))
        for i in range(1, len(ln) - 1):
            weight = (log_energy[i] - log_energy[i - 1]) / (log_energy[i + 1] - log_energy[i - 1])
            predicted = ln[i - 1] * (1.0 - weight) + ln[i + 1] * weight
            worst = max(worst, abs(ln[i] - predicted))
    assert worst < 0.035


def test_peak_position_moves_to_higher_energy_with_z() -> None:
    """Effective-charge screening pushes the Bragg peak up in energy with Z."""
    peaks = {
        z: GRID_MEV_U[int(np.argmax(column.mass_stopping_power_MeV_cm2_g))]
        for z, column in load_water_columns().items()
    }

    assert peaks[1] == 0.25
    assert peaks[6] == 0.5
    assert peaks[92] == 3.2
    assert peaks[103] == 3.2
    ordered = [peaks[z] for z in sorted(peaks)]
    assert all(b >= a for a, b in pairwise(ordered))


def test_proton_column_matches_pstar_above_5_MeV_u() -> None:
    """Anchors the absolute scale against a modern independent reference.

    Only above 5 MeV/u. This is a pre-ICRU-49 compilation: it drifts to ~6% at
    2 MeV/u and to about half the modern value at 0.05 MeV/u. The second
    assertion pins that drift so the table's real validity range stays visible
    instead of being assumed to hold everywhere.
    """
    proton = load_water_columns()[1]
    grid = list(proton.energy_MeV_u)

    for energy, reference in PSTAR_PROTON.items():
        value = proton.mass_stopping_power_MeV_cm2_g[grid.index(energy)]
        assert value == pytest.approx(reference, rel=0.015)

    at_2_MeV_u = proton.mass_stopping_power_MeV_cm2_g[grid.index(2.0)]
    assert at_2_MeV_u / PSTAR_PROTON_2_MEV_U == pytest.approx(1.058, abs=0.01)


# Water-column cells re-read off the page images [MeV cm^2/g], keyed by
# (Z, E [MeV/u]). Printed pages: 1H p. 254, 4He p. 256, 12C p. 264.
PRINTED_LOW_ENERGY_CELLS = {
    (1, 0.25): 669.0,
    (1, 0.32): 654.0,
    (1, 0.5): 553.0,
    (1, 1.0): 308.0,
    (2, 0.32): 2337.0,
    (2, 0.5): 2094.0,
    (2, 1.0): 1209.0,
    (6, 0.5): 10216.0,
    (6, 1.0): 7783.0,
}


def test_low_energy_excess_is_the_printed_table() -> None:
    """The 0.3-1 MeV/u excess over modern data is in the 1970 source itself.

    The proton column runs ~+18% over PSTAR at 1 MeV/u and ~+29% at 0.5 MeV/u,
    and peaks at 0.25 MeV/u instead of ~0.08. That looks like a row offset but
    is not one: these cells match the page images, and the printed water column
    is the Bragg sum of NS1970's *gaseous* H and O columns (their footnote 31,
    p. 250; poor low-energy proton fits accepted in Sec. VIII, p. 249). This
    pins them so a well-meant "correction" towards modern data fails here.
    """
    columns = load_water_columns()
    for (z, energy), printed in PRINTED_LOW_ENERGY_CELLS.items():
        column = columns[z]
        value = column.mass_stopping_power_MeV_cm2_g[list(column.energy_MeV_u).index(energy)]
        assert value == printed, f"Z={z} at {energy} MeV/u"


def test_light_ion_columns_scale_as_z_squared_at_high_energy() -> None:
    """At 12 MeV/u light ions are ~fully stripped, so dE/dx -> Z^2 times the proton.

    A single mis-transcribed digit anywhere in a light-ion column shows up here.
    """
    columns = load_water_columns()
    index = list(columns[1].energy_MeV_u).index(12.0)
    proton = columns[1].mass_stopping_power_MeV_cm2_g[index]

    for z in (2, 3, 4, 5, 6, 7, 8, 9, 10):
        ratio = columns[z].mass_stopping_power_MeV_cm2_g[index] / proton
        assert ratio / z**2 == pytest.approx(1.0, abs=0.05)


def test_effective_charge_screening_grows_with_z() -> None:
    """Heavy ions fall progressively below the bare-Z^2 scaling."""
    columns = load_water_columns()
    index = list(columns[1].energy_MeV_u).index(12.0)
    proton = columns[1].mass_stopping_power_MeV_cm2_g[index]

    screening = [
        columns[z].mass_stopping_power_MeV_cm2_g[index] / proton / z**2
        for z in (10, 14, 18, 26, 36, 54, 92, 103)
    ]
    assert all(b < a for a, b in pairwise(screening))
    assert screening[-2] == pytest.approx(0.36, abs=0.02)  # uranium
    assert screening[-1] == pytest.approx(0.34, abs=0.02)  # lawrencium


def test_stopping_power_adapter_converts_to_let() -> None:
    sp = northcliffe_schilling1970_water()
    column = load_water_columns()[6]
    index = list(column.energy_MeV_u).index(1.0)

    # MeV cm^2/g * 1 g/cm^3 -> MeV/cm -> keV/um is a factor 0.1.
    expected = column.mass_stopping_power_MeV_cm2_g[index] * 0.1
    assert sp.let_keV_um(Ion(6, 12, 1.0)) == pytest.approx(expected, rel=1e-12)


def test_adapter_refuses_to_extrapolate_past_the_source_grid() -> None:
    """Scholz & Kraft 1996 Fig. 4 runs to 20 MeV/u; this table stops at 12.

    Extrapolating a 1970 table half an order of magnitude past its last point
    would be invisible in a figure, so the table raises instead.
    """
    sp = northcliffe_schilling1970_water()

    assert ENERGY_RANGE_MEV_U == (0.0125, 12.0)
    with pytest.raises(ValueError, match="outside tabulated range"):
        sp.let_keV_um(Ion(6, 12, 20.0))
    with pytest.raises(ValueError, match="outside tabulated range"):
        sp.let_keV_um(Ion(1, 1, 0.001))


def test_agrees_with_the_bundled_bethe_model_where_both_are_valid() -> None:
    """A sanity band, not a golden: two independent routes to the same LET."""
    from pyradbio.track.stopping_power import SimpleBetheStoppingPower

    table = northcliffe_schilling1970_water()
    bethe = SimpleBetheStoppingPower()

    for ion in (Ion(1, 1, 10.0), Ion(6, 12, 10.0), Ion(18, 40, 10.0), Ion(92, 238, 10.0)):
        ratio = table.let_keV_um(ion) / bethe.let_keV_um(ion)
        assert 0.85 < ratio < 1.15, f"{ion} ratio {ratio:.3f}"


def test_hybrid_uses_the_table_inside_the_grid_and_bethe_outside() -> None:
    """The seam is at 12 MeV/u, and the hybrid records which side it used.

    Both halves must be the *unmodified* sources: inside the grid it has to be
    the table exactly (not an average, not a blend), and outside it has to be
    Bethe exactly. Anything else would be a third stopping power nobody chose.
    """
    from pyradbio.track.stopping_power import SimpleBetheStoppingPower

    hybrid = NorthcliffeSchillingWithBetheTail()
    table = northcliffe_schilling1970_water()
    bethe = SimpleBetheStoppingPower()

    inside, outside = Ion(6, 12, 8.0), Ion(6, 12, 400.0)
    assert hybrid.let_keV_um(inside) == table.let_keV_um(inside)
    assert hybrid.let_keV_um(outside) == bethe.let_keV_um(outside)

    assert hybrid.table_hits == 1
    assert hybrid.fallback_energies == {400.0}


def test_build_stopping_power_names_the_choice_for_the_checkpoints() -> None:
    """``name`` is what the MC batches stamp into their CSVs.

    Both Scholz batches refuse to reuse a stored row whose ``stopping_power``
    differs from the active one, so these strings are load-bearing: renaming one
    silently invalidates a checkpoint, and returning the same name for two
    different LETs would silently mix them. That mattering is measured, not
    hypothetical -- the two choices differ by 13% in LEM I alpha at carbon
    1 MeV/u, which is visible on the Fig. 1 reconstruction.
    """
    ns, plain = build_stopping_power("ns1970"), build_stopping_power("bethe")

    assert isinstance(ns, NorthcliffeSchillingWithBetheTail)
    assert isinstance(plain, BetheOnly)
    assert ns.name == "ns1970+bethe"
    assert plain.name == "bethe"
    assert ns.name != plain.name

    low = Ion(6, 12, 1.0)
    assert ns.let_keV_um(low) / plain.let_keV_um(low) > 1.1

    with pytest.raises(ValueError, match="unknown stopping power"):
        build_stopping_power("pstar")
