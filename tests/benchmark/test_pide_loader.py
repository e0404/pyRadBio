"""PIDE loader and benchmark harness."""

from __future__ import annotations

from pathlib import Path

import pytest

from pyradbio.benchmark import metrics, pide

FIXTURE = Path(__file__).parent / "fixtures" / "synthetic_pide.xlsx"


def test_metrics_known_values() -> None:
    assert metrics.smape([1.0, 1.0], [1.0, 1.0]) == pytest.approx(0.0)


# --- ion token parsing -----------------------------------------------------
@pytest.mark.parametrize(
    ("token", "expected"),
    [
        ("C-12", (6, 12)),
        ("12C", (6, 12)),
        ("C12", (6, 12)),
        ("C", (6, None)),
        ("He-4", (2, 4)),
        (" H-1 ", (1, 1)),
        ("Zz-99", (None, None)),  # unknown symbol
        ("42", (None, None)),  # purely numeric -> ambiguous, defer to charge column
    ],
)
def test_parse_ion_token(token: str, expected: tuple[int | None, int | None]) -> None:
    assert pide.parse_ion_token(token) == expected


# --- row parsing (format-independent) --------------------------------------
def test_parse_pide_rows_matches_aliased_headers() -> None:
    headers = ["Cell", "Ion", "Energy", "LET", "Alpha_I", "Beta_I", "alpha X", "beta-x"]
    rows = [["V79", "C-12", 100.0, 30.0, 0.45, 0.045, 0.13, 0.05]]
    (entry,) = pide.parse_pide_rows(headers, rows)
    assert entry.cell_line == "V79"
    assert (entry.Z, entry.A) == (6, 12)
    assert entry.energy_MeV_u == pytest.approx(100.0)
    assert entry.let_keV_um == pytest.approx(30.0)
    assert entry.rbe_alpha == pytest.approx(0.45 / 0.13)
    assert entry.photon_radiation == ""  # no PhotonRadiation column -> ""


def test_parse_pide_rows_photon_radiation() -> None:
    headers = ["Cell", "Ion", "LET", "alpha_I", "beta_I", "alpha_X", "beta_X", "PhotonRadiation"]
    rows = [
        ["V79", "C-12", 30.0, 0.4, 0.04, 0.1, 0.05, "200kVp"],
        ["HSG", "C-12", 30.0, 0.4, 0.04, 0.1, 0.05, "N/A"],  # missing marker -> ""
    ]
    kvp, na = pide.parse_pide_rows(headers, rows)
    assert kvp.photon_radiation == "200kVp"
    assert na.photon_radiation == ""  # N/A normalised away


def test_parse_pide_rows_uses_charge_column_when_ion_unknown() -> None:
    headers = ["Cell", "Ion", "Charge", "LET", "alpha_I", "beta_I", "alpha_X", "beta_X"]
    rows = [["X", "Unobtanium", 6, 30.0, 0.4, 0.04, 0.1, 0.05]]
    (entry,) = pide.parse_pide_rows(headers, rows)
    assert entry.Z == 6
    assert entry.A is None


def test_parse_pide_rows_skips_incomplete_and_unresolved() -> None:
    headers = ["Cell", "Ion", "Charge", "LET", "alpha_I", "beta_I", "alpha_X", "beta_X"]
    rows = [
        ["ok", "C-12", 6, 30.0, 0.4, 0.04, 0.1, 0.05],
        ["missing-beta", "C-12", 6, 30.0, 0.4, 0.04, 0.1, None],
        ["no-ion", "Zz", None, 30.0, 0.4, 0.04, 0.1, 0.05],
    ]
    entries = pide.parse_pide_rows(headers, rows)
    assert [e.cell_line for e in entries] == ["ok"]


def test_parse_pide_rows_lq_variant_selection() -> None:
    """The PIDE ``*_fit``/``*_paper`` columns are selected per the variant, with fallback."""
    headers = [
        "Cells",
        "Ion",
        "LET",
        "ai_paper",
        "bi_paper",
        "ax_paper",
        "bx_paper",
        "ai_fit",
        "bi_fit",
        "ax_fit",
        "bx_fit",
    ]
    # Row 1: both flavours present. Row 2: fit is N/A -> fallback to paper.
    rows = [
        ["V79", "C-12", 30.0, 0.40, 0.04, 0.10, 0.05, 0.44, 0.044, 0.11, 0.055],
        ["V79", "C-12", 30.0, 0.40, 0.04, 0.10, 0.05, "N/A", "N/A", "N/A", "N/A"],
    ]
    # "fit" uses only the fit columns: row 2 (all N/A) is incomplete -> skipped.
    fit = pide.parse_pide_rows(headers, rows, variant="fit")
    assert len(fit) == 1
    assert fit[0].alpha_ion == pytest.approx(0.44)

    # "fit_then_paper" falls back per field: row 2 recovers its paper values.
    ftp = pide.parse_pide_rows(headers, rows, variant="fit_then_paper")
    assert len(ftp) == 2
    assert ftp[0].alpha_ion == pytest.approx(0.44)  # fit wins where present
    assert ftp[1].alpha_ion == pytest.approx(0.40)  # fell back to paper

    paper = pide.parse_pide_rows(headers, rows, variant="paper")
    assert paper[0].alpha_ion == pytest.approx(0.40)
    assert paper[0].alpha_x == pytest.approx(0.10)


def test_parse_pide_rows_strict_raises_on_bad_row() -> None:
    headers = ["Cell", "Ion", "Charge", "LET", "alpha_I", "beta_I", "alpha_X", "beta_X"]
    rows = [["missing-beta", "C-12", 6, 30.0, 0.4, 0.04, 0.1, None]]
    with pytest.raises(ValueError):
        pide.parse_pide_rows(headers, rows, strict=True)


# --- end-to-end loading of the synthetic xlsx fixture ----------------------
def test_load_synthetic_fixture() -> None:
    """Loader parses the synthetic PIDE-schema fixture into typed entries."""
    assert FIXTURE.is_file(), "run tests/benchmark/fixtures/make_synthetic_pide.py"
    entries = pide.load_pide(FIXTURE)
    assert entries, "fixture should yield at least one entry"
    assert all(isinstance(e, pide.PideEntry) for e in entries)
    # The two malformed rows (missing beta_X; unknown ion) are dropped.
    assert len(entries) == 5
    cell_lines = {e.cell_line for e in entries}
    assert "Bad" not in cell_lines and "Weird" not in cell_lines
    # The intentionally blank-energy row survives with energy_MeV_u = None.
    assert any(e.energy_MeV_u is None for e in entries)
    assert all(e.energy_MeV_u is None or e.energy_MeV_u > 0 for e in entries)
    # PhotonRadiation is captured verbatim; the fixture's "N/A" row -> "".
    assert {e.photon_radiation for e in entries} == {"200kVp", "60Co", "6MV", ""}


def test_load_pide_missing_file_raises(tmp_path: Path) -> None:
    with pytest.raises(FileNotFoundError):
        pide.load_pide(tmp_path)  # empty dir -> no PIDE file


@pytest.mark.benchmark
def test_end_to_end_against_real_pide(pide_path: str) -> None:
    """Skipped unless --pide <path> is given; runs a real benchmark slice."""
    from pyradbio.benchmark.harness import run_benchmark

    entries = pide.load_pide(pide_path)
    assert entries, "real PIDE file should yield entries"
    # Cap the evaluated points to keep the slice fast.
    result = run_benchmark(entries, model_name="lem1", limit=25)
    # Sanity bounds: some points evaluated, aggregate errors finite and modest.
    assert result.n_points > 0
    assert 0.0 <= result.smape_alpha <= 1.0
    assert result.mean_abs_log_error_alpha < 1.0  # within ~1 decade on average


def test_source_is_the_publication_name_not_the_running_number() -> None:
    """Fabricated values distinguish publication-name priority from column order."""
    headers = [
        "#ExpID",
        "#Publication",
        "PublicationName",
        "Cells",
        "Ion",
        "LET",
        "Energy",
        "ax_paper",
        "bx_paper",
        "ai_paper",
        "bi_paper",
    ]
    rows = [[1, 9001, "synthetic", "Synthetic", "12C", 50.0, 20.0, 0.2, 0.04, 0.8, 0.03]]
    (entry,) = pide.parse_pide_rows(headers, rows)
    assert entry.source == "pide:synthetic"
    assert entry.cell_line == "Synthetic"


def test_partial_observation_strict_mode_keeps_missing_values_but_rejects_unknown_ions() -> None:
    headers = ["Cells", "Ion", "LET", "ai_paper", "bi_paper", "ax_paper", "bx_paper"]
    rows = [["Synthetic", "12C", None, 0.8, None, None, 0.04]]
    (entry,) = pide.parse_pide_observations(headers, rows, variant="paper", strict=True)
    assert entry.A == 12 and entry.Z == 6
    assert entry.let_keV_um is None
    assert entry.alpha_ion == 0.8 and entry.beta_ion is None
    assert entry.alpha_x is None and entry.beta_x == 0.04
    assert entry.source == "pide:row 0"
    with pytest.raises(ValueError, match="missing"):
        pide.parse_pide_rows(headers, rows, variant="paper", strict=True)
    unknown = [["Synthetic", "unknown", None, None, None, None, None]]
    assert pide.parse_pide_observations(headers, unknown) == []
    with pytest.raises(ValueError, match="unresolved ion Z"):
        pide.parse_pide_observations(headers, unknown, strict=True)
