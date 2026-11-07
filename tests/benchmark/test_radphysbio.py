"""RadPhysBio loader (Zanni et al. 2024, doi:10.3390/ijms25094729) on the CC BY fixture excerpt."""

from __future__ import annotations

from pathlib import Path

import pytest

from pyradbio.benchmark import radphysbio as rpb
from pyradbio.benchmark.dataset import ReferenceRecord

FIXTURE = Path(__file__).parent / "fixtures" / "radphysbio_excerpt.csv"


@pytest.fixture(scope="module")
def entries() -> list[rpb.RadPhysBioEntry]:
    return rpb.load_radphysbio(FIXTURE)


def _one(
    entries: list[rpb.RadPhysBioEntry], radiation: str, pmid: str, exp_no: int
) -> rpb.RadPhysBioEntry:
    (e,) = [
        x
        for x in entries
        if x.radiation_type == radiation and x.pmid == pmid and x.exp_no == exp_no
    ]
    return e


# --- parsing ----------------------------------------------------------------


def test_every_fixture_row_is_loaded(entries: list[rpb.RadPhysBioEntry]) -> None:
    n_lines = sum(1 for _ in FIXTURE.open(encoding="utf-8")) - 1
    assert len(entries) == n_lines


def test_radiation_types_are_ascii_normalised(entries: list[rpb.RadPhysBioEntry]) -> None:
    kinds = {e.radiation_type for e in entries}
    assert kinds <= {"X-rays", "gamma-rays", "protons", "carbon", "alpha-particles"}
    assert "gamma-rays" in kinds and "alpha-particles" in kinds


def test_furusawa_hsg_photon_row_matches_the_paper(entries: list[rpb.RadPhysBioEntry]) -> None:
    """Furusawa 2000 HSG 200 kVp: alpha_x = 0.313, beta_x = 0.061 as the paper prints."""
    e = _one(entries, "X-rays", "11025645", 1)
    assert e.cell_line == "HSG"
    assert e.alpha == pytest.approx(0.313)
    assert e.beta == pytest.approx(0.061)
    assert e.is_photon and e.Z is None and e.energy_MeV_u is None
    assert e.let_keV_um == pytest.approx(2.0)  # the database's assumed X-ray LET


def test_carbon_energy_is_total_and_converted_per_nucleon(
    entries: list[rpb.RadPhysBioEntry],
) -> None:
    e = _one(entries, "carbon", "11025645", 3)
    assert (e.Z, e.A) == (6, 12)
    assert e.energy_MeV == pytest.approx(1620.0)
    assert e.energy_MeV_u == pytest.approx(135.0)
    assert e.let_keV_um == pytest.approx(30.3)
    assert e.alpha == pytest.approx(0.359)
    assert e.beta == pytest.approx(0.141)
    assert e.irradiation_conditions == "s"


def test_missing_beta_is_none_not_zero(entries: list[rpb.RadPhysBioEntry]) -> None:
    e = _one(entries, "carbon", "11025645", 1)
    assert e.alpha == pytest.approx(0.161)
    assert e.beta is None


def test_split_cell_line_rows_are_rejoined(entries: list[rpb.RadPhysBioEntry]) -> None:
    """Two rows have 31 fields because 'SP6,5' was split on its comma."""
    sp = [e for e in entries if e.pmid == "26024093" and e.cell_line == "SP6,5"]
    assert len(sp) == 2
    assert {e.exp_no for e in sp} == {3, 4}
    assert {e.tissue for e in sp} == {"Uvea"}
    ion = next(e for e in sp if not e.is_photon)
    assert ion.radiation_type == "carbon"
    assert ion.mcds_at_energy.dsb_1pct_O2 == pytest.approx(6.21)
    assert ion.mcds_at_energy.non_dsb_20pct_O2 == pytest.approx(289.65)
    assert ion.radiation_source == ""


def test_greek_na_markers_read_as_missing() -> None:
    assert rpb.is_missing("\u039d/\u0391")  # Greek capital nu + alpha
    assert rpb.is_missing("\u039d/A")  # mixed
    assert rpb.is_missing("N/A")
    assert not rpb.is_missing("0")


def test_dsb_fit_flag_is_split_off(entries: list[rpb.RadPhysBioEntry]) -> None:
    flagged = [e for e in entries if e.dsb_from_fit]
    assert flagged, "fixture carries an 'f'-flagged DSB value"
    assert all(e.dsb_per_Gy_Gbp is not None for e in flagged)
    assert any(not e.dsb_from_fit and e.dsb_per_Gy_Gbp is None for e in entries)


def test_alpha_source_and_gamma_source_strings(entries: list[rpb.RadPhysBioEntry]) -> None:
    bi = [e for e in entries if e.radiation_type == "alpha-particles" and e.pmid == "17307126"]
    assert bi and all(e.radiation_source == "213Bi" for e in bi)
    assert all((e.Z, e.A) == (2, 4) for e in bi)
    co = _one(entries, "gamma-rays", "10630978", 1)
    assert co.radiation_source == "60Co"
    assert rpb.photon_quality_label(co) == "60Co"


def test_photon_quality_label_for_x_rays(entries: list[rpb.RadPhysBioEntry]) -> None:
    e = _one(entries, "X-rays", "10924995", 1)  # Suzuki 2000, 0.2 MeV -> 200 kVp
    assert rpb.photon_quality_label(e) == "200kVp"


def test_strict_rejects_a_ragged_row() -> None:
    headers = list(rpb._EXPECTED_HEADERS)
    bad = ["1", "1", "1", "X"] + ["N/A"] * 20
    with pytest.raises(ValueError, match="fields"):
        rpb.parse_radphysbio_rows(headers, [bad], strict=True)
    assert rpb.parse_radphysbio_rows(headers, [bad]) == []


def test_missing_required_column_raises() -> None:
    with pytest.raises(ValueError, match="lacks columns"):
        rpb.parse_radphysbio_rows(["foo", "bar"], [["1", "2"]])


# --- pairing ------------------------------------------------------------------


def test_pairing_joins_ion_to_same_publication_photon(entries: list[rpb.RadPhysBioEntry]) -> None:
    records, _ = rpb.pair_photon_references(entries)
    hsg = [r for r in records if r.source == "radphysbio:11025645" and r.cell_line == "HSG"]
    assert len(hsg) == 21
    r = next(x for x in hsg if x.let_keV_um == pytest.approx(30.3))
    assert isinstance(r, ReferenceRecord)
    assert r.alpha_x == pytest.approx(0.313)
    assert r.beta_x == pytest.approx(0.061)
    assert r.alpha_ion == pytest.approx(0.359)
    assert r.Z == 6 and r.A == 12
    assert r.energy_MeV_u == pytest.approx(135.0)
    assert r.energy_is_nominal
    # Furusawa's X-ray row lists no energy, so the label falls back to the type.
    assert r.photon_radiation == "X-rays"
    assert r.rbe_alpha == pytest.approx(0.359 / 0.313)


def test_pairing_report_accounts_for_every_ion_row(entries: list[rpb.RadPhysBioEntry]) -> None:
    records, report = rpb.pair_photon_references(entries)
    n_ion = sum(1 for e in entries if not e.is_photon)
    assert report.n_ion_rows == n_ion
    assert report.n_paired + report.n_unpaired + report.n_duplicate_copies == n_ion
    assert len(records) == report.n_paired - report.n_incomplete
    assert report.n_unpaired > 0  # the 213Bi rows have no photon row in their paper


def test_missing_beta_becomes_zero_in_the_record(entries: list[rpb.RadPhysBioEntry]) -> None:
    records, _ = rpb.pair_photon_references(entries)
    r = next(x for x in records if x.source == "radphysbio:11025645" and x.let_keV_um == 22.5)
    assert r.beta_ion == 0.0


def test_ambiguous_photon_rows_are_flagged_and_droppable(
    entries: list[rpb.RadPhysBioEntry],
) -> None:
    """A publication with several, differing photon rows for the line an ion row uses."""
    _, keep = rpb.pair_photon_references(entries, include_ambiguous=True)
    _, drop = rpb.pair_photon_references(entries, include_ambiguous=False)
    assert keep.n_ambiguous == drop.n_ambiguous > 0
    assert drop.n_paired < keep.n_paired


def test_dataset_wrapper_carries_caveats_and_tally() -> None:
    ds = rpb.load_radphysbio_dataset(FIXTURE)
    assert ds.name == "RadPhysBio"
    assert "10.3390/ijms25094729" in ds.citation
    assert any("digitized" in c for c in ds.caveats)
    assert any("Pairing tally" in c for c in ds.caveats)
    assert len(ds) == len(ds.records) > 0


@pytest.mark.parametrize("cell", ["T\u0395-2", "HSGc-C5"])
def test_pairing_uses_the_same_cell_identity_as_crosscheck(
    entries: list[rpb.RadPhysBioEntry], cell: str
) -> None:
    from dataclasses import replace

    from pyradbio.benchmark.crosscheck import rows_from_radphysbio

    # Fabricated observations; no experiment-specific PIDE values.
    photon = replace(
        next(e for e in entries if e.is_photon),
        pmid="synthetic",
        cell_line="TE-2" if cell.startswith("T") else "HSG",
        alpha=0.2,
        beta=0.04,
    )
    ion = replace(
        next(e for e in entries if e.Z == 6),
        pmid="synthetic",
        cell_line=cell,
        alpha=0.8,
        beta=0.03,
        let_keV_um=50.0,
    )
    records, report = rpb.pair_photon_references([photon, ion])
    assert len(records) == 1
    assert report.n_unpaired == 0
    assert records[0].alpha_x == rows_from_radphysbio([photon, ion])[0].alpha_x


def test_missing_export_gives_guidance(tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> None:
    with pytest.raises(FileNotFoundError, match="No RadPhysBio export"):
        rpb.find_radphysbio_file(tmp_path)
    monkeypatch.delenv("PYRADBIO_RADPHYSBIO_DIR", raising=False)
    monkeypatch.setattr("pyradbio.datasets.load_project_dotenv", lambda: None)
    with pytest.raises(FileNotFoundError, match="PYRADBIO_RADPHYSBIO_DIR"):
        rpb.find_radphysbio_file(None)


def test_known_defects_are_rows_that_exist_and_are_left_as_exported(
    entries: list[rpb.RadPhysBioEntry],
) -> None:
    """The loader documents the Tsuboi/Suzuki/Furusawa slips; it does not silently fix them."""
    assert all(d.pmid and d.radiation_type and d.what for d in rpb.KNOWN_DEFECTS)
    tsuboi = next(d for d in rpb.KNOWN_DEFECTS if d.pmid == "9687977")
    assert "scale factors" in tsuboi.what
    # The Suzuki KNS-60 X-ray beta is in the fixture and still reads 0.022 (uncorrected).
    kns60 = next(
        e for e in entries if e.pmid == "10924995" and e.cell_line == "KNS-60" and e.is_photon
    )
    assert kns60.beta == pytest.approx(0.022)
    assert any(c.startswith("**One experiment can appear") for c in rpb.RADPHYSBIO_CAVEATS)


# --- verbatim copies across PMIDs -----------------------------------------------


def test_verbatim_copies_under_a_later_pmid_are_detected(
    entries: list[rpb.RadPhysBioEntry],
) -> None:
    """PMID 27380803 re-lists Suzuki 2000 (PMID 10924995) rows digit for digit."""
    copies = rpb.duplicate_copies(entries)
    assert copies, "the fixture carries a re-listed table"
    assert all(k[0] == "carbon" for k in copies)
    dropped = [e for e in entries if e.key in copies]
    assert {e.pmid for e in dropped} == {"27380803"}  # the later PMID goes, the original stays
    originals = [e for e in entries if e.pmid == "10924995" and not e.is_photon]
    assert originals and not any(e.key in copies for e in originals)
    a549 = next(e for e in dropped if rpb.cell_key(e.cell_line) == "a549")
    twin = next(
        e
        for e in originals
        if rpb.cell_key(e.cell_line) == "a549" and e.let_keV_um == a549.let_keV_um
    )
    assert (twin.alpha, twin.beta) == (a549.alpha, a549.beta)

    kept, on = rpb.pair_photon_references(entries)
    _, off = rpb.pair_photon_references(entries, collapse_duplicates=False)
    # The excerpt has no photon partners for these two later copies: they are
    # counted as unpaired before duplicate filtering, not as dropped pairs, so
    # pairing collapses nothing here (the synthetic tests below exercise that).
    assert on.n_duplicate_copies == 0
    assert off.n_duplicate_copies == 0
    assert on == off
    # ...but the tally still says they are re-listings, not independent rows.
    assert on.n_relisted_unscored == len(dropped)
    later = {rpb.cell_key(e.cell_line) for e in entries if e.pmid == "27380803" and e.is_photon}
    assert all(rpb.cell_key(e.cell_line) not in later for e in dropped)
    # The later PMID's own, partnered experiments are still scored.
    assert any(r.source == "radphysbio:27380803" for r in kept)


def test_pairing_report_accounts_for_copies_too(entries: list[rpb.RadPhysBioEntry]) -> None:
    _, report = rpb.pair_photon_references(entries)
    assert report.n_paired + report.n_unpaired + report.n_duplicate_copies == report.n_ion_rows
    ds = rpb.load_radphysbio_dataset(FIXTURE)
    tally = next(c for c in ds.caveats if c.startswith("Pairing tally"))
    assert "verbatim copies" in tally
    assert f"{report.n_relisted_unscored} of the rows not scored" in tally


@pytest.mark.parametrize("earlier_reference", ["absent", "incomplete", "ambiguous", "usable"])
def test_duplicates_keep_the_earliest_eligible_photon_pair(
    entries: list[rpb.RadPhysBioEntry], earlier_reference: str
) -> None:
    from dataclasses import replace

    from pyradbio.benchmark.harness import run_benchmark

    # Fabricated observations: the original ion row may lack a usable reference.
    original = replace(
        next(e for e in entries if e.Z == 1),
        pmid="100",
        exp_id=9001,
        cell_line="synthetic",
        let_keV_um=5.0,
        alpha=0.4,
        beta=0.04,
    )
    later = replace(original, pmid="200", exp_id=9002)
    latest = replace(original, pmid="300", exp_id=9003)
    photon = replace(
        next(e for e in entries if e.is_photon),
        pmid="200",
        exp_id=9004,
        cell_line="synthetic",
        alpha=0.2,
        beta=0.03,
    )
    # Reverse publication order to ensure eligibility, then PMID, determines retention.
    observations = [latest, later, original, photon, replace(photon, pmid="300", exp_id=9005)]
    if earlier_reference != "absent":
        observations.append(
            replace(
                photon,
                pmid="100",
                exp_id=9006,
                alpha=None if earlier_reference == "incomplete" else 0.2,
            )
        )
    if earlier_reference == "ambiguous":
        observations.append(replace(photon, pmid="100", exp_id=9007, alpha=0.3))

    records, tally = rpb.pair_photon_references(observations, include_ambiguous=False)
    assert len(records) == 1
    assert records[0].source == (
        "radphysbio:100" if earlier_reference == "usable" else "radphysbio:200"
    )
    assert run_benchmark(records, "wedenberg").n_points == 1
    assert tally.n_duplicate_copies == (2 if earlier_reference == "usable" else 1)
    assert tally.n_unpaired == (earlier_reference == "absent")
    assert tally.n_incomplete == (earlier_reference == "incomplete")
    assert tally.n_ambiguous == (earlier_reference == "ambiguous")
    assert tally.n_paired - tally.n_incomplete == len(records)
    assert tally.n_paired + tally.n_unpaired + tally.n_duplicate_copies + tally.n_ambiguous == 3
    # An unusable *original* is not a re-listing of anything.
    assert tally.n_relisted_unscored == 0


def test_unscored_relistings_are_counted_but_not_collapsed(
    entries: list[rpb.RadPhysBioEntry],
) -> None:
    from dataclasses import replace

    # Fabricated observations: a usable original, one re-listing without a
    # photon partner, and one whose partner has no alpha.
    original = replace(
        next(e for e in entries if e.Z == 1),
        pmid="100",
        exp_id=9201,
        cell_line="synthetic",
        let_keV_um=5.0,
        alpha=0.4,
        beta=0.04,
    )
    photon = replace(
        next(e for e in entries if e.is_photon),
        pmid="100",
        exp_id=9202,
        cell_line="synthetic",
        alpha=0.2,
        beta=0.03,
    )
    observations = [
        original,
        photon,
        replace(original, pmid="200", exp_id=9203),
        replace(original, pmid="300", exp_id=9204),
        replace(photon, pmid="300", exp_id=9205, alpha=None),
    ]
    records, tally = rpb.pair_photon_references(observations)
    assert [r.source for r in records] == ["radphysbio:100"]
    assert (tally.n_unpaired, tally.n_incomplete, tally.n_duplicate_copies) == (1, 1, 0)
    assert tally.n_relisted_unscored == 2


def test_a_dropped_ambiguous_copy_leaves_the_ambiguous_tally(
    entries: list[rpb.RadPhysBioEntry],
) -> None:
    from dataclasses import replace

    # Fabricated observations: the later copy's publication has two photon rows.
    original = replace(
        next(e for e in entries if e.Z == 1),
        pmid="100",
        exp_id=9101,
        cell_line="synthetic",
        let_keV_um=5.0,
        alpha=0.4,
        beta=0.04,
    )
    photon = replace(
        next(e for e in entries if e.is_photon),
        pmid="100",
        exp_id=9102,
        cell_line="synthetic",
        alpha=0.2,
        beta=0.03,
    )
    observations = [
        original,
        photon,
        replace(original, pmid="200", exp_id=9103),
        replace(photon, pmid="200", exp_id=9104),
        replace(photon, pmid="200", exp_id=9105, alpha=0.3),
    ]
    _, off = rpb.pair_photon_references(observations, collapse_duplicates=False)
    assert (off.n_paired, off.n_ambiguous) == (2, 1)
    records, on = rpb.pair_photon_references(observations)
    assert [r.source for r in records] == ["radphysbio:100"]
    assert (on.n_paired, on.n_ambiguous, on.n_duplicate_copies) == (1, 0, 1)
    assert on.n_paired + on.n_unpaired + on.n_duplicate_copies == on.n_ion_rows
