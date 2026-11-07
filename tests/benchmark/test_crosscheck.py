"""Experiment-level cross-check of two reference databases (synthetic rows only).

The one ``@benchmark`` test at the end runs the real PIDE 3.4 vs RadPhysBio
comparison and skips when either dataset is absent.
"""

from __future__ import annotations

from pathlib import Path

import pytest

from pyradbio.benchmark import crosscheck as cc
from pyradbio.benchmark.dataset import ReferenceRecord
from pyradbio.benchmark.radphysbio import MCDSYields, RadPhysBioEntry

# --- cell-line identity -------------------------------------------------------


@pytest.mark.parametrize(
    ("a", "b"),
    [
        ("T1", "T-1"),
        ("A549", "A-549"),
        ("HSG", "HSGc-C5"),
        ("HeLa S3", "HeLaS3"),
        ("LC-1 sq", "LC-1sq"),
        ("U-251MG(KO)", "U-251MGKO"),
        ("AG1522", "AG01522"),
        ("AG1522", "AG1522B"),
        ("U-87", "U-87MG"),
        ("AT1OS/T-n", "AT1OS"),
        ("HEK/293", "HEK"),
        ("TG98G", "T98G"),
        ("T" + chr(0x039A) + "-1", "TK1"),  # Greek capital kappa, as in the RadPhysBio export
        (chr(0x0391) + "-549", "A-549"),  # Greek capital alpha
        (chr(0x03BC) + "2", "mu2"),  # small mu
    ],
)
def test_cell_key_equates_aliases(a: str, b: str) -> None:
    assert cc.cell_key(a) == cc.cell_key(b)


def test_cell_key_keeps_distinct_lines_apart() -> None:
    assert cc.cell_key("A-172") != cc.cell_key("A-549")
    assert cc.cell_key("HFL-I") != cc.cell_key("HFL-III")
    assert cc.cell_key("H1299wtp53") != cc.cell_key("H1299tp53-null")


# --- value comparison ---------------------------------------------------------


@pytest.mark.parametrize(
    ("x", "expected"),
    [(0.0615, 4), (0.537, 3), (2.0, 1), (1e-05, 5), (0.0, None), (float("nan"), None)],
)
def test_decimal_places(x: float, expected: int | None) -> None:
    assert cc.decimal_places(x) == expected


@pytest.mark.parametrize(
    ("a", "b", "category"),
    [
        (0.313, 0.313, "identical"),
        (0.0305, 0.03, "identical"),  # rounds half-even to the coarser 2 decimals
        (0.123456789012345, 0.123, "identical"),  # MADE-UP 15-digit refit vs a 3-decimal print
        (1.895, 1.9, "identical"),
        (0.074, 0.075, "rounding"),  # one unit in the last printed digit
        (0.0615, 0.061, "rounding"),
        (0.512, 0.541, "digitization"),  # 5.4%, beyond one unit of the 3rd decimal
        (0.512, 0.62, "discrepant"),  # 17%
        (0.5, 0.53, "identical"),  # 0.5 carries one decimal: 0.53 rounds to it
        (0.5, 0.58, "rounding"),  # one unit of that decimal
        (1.0, 2.0, "discrepant"),
        (-0.062, 0.062, "sign"),
        (0.15, -0.15, "sign"),
        (0.0, 0.0, "identical"),
        (0.0, 0.05, "discrepant"),
        (None, 0.3, "missing"),
        (0.3, None, "missing"),
        (float("nan"), 0.3, "missing"),
    ],
)
def test_compare_values_categories(a: float | None, b: float | None, category: str) -> None:
    assert cc.compare_values(a, b).category == category


def test_compare_values_reports_differences() -> None:
    c = cc.compare_values(0.512, 0.541)
    assert c.abs_diff == pytest.approx(0.029)
    assert c.rel_diff == pytest.approx(0.029 / 0.541)
    assert c.decimals == 3
    assert cc.compare_values(0.5, 0.6, digitization_tol=0.2).category == "rounding"
    assert cc.compare_values(0.512, 0.601, digitization_tol=0.2).category == "digitization"
    assert cc.compare_values(0.512, 0.601, digitization_tol=0.1).category == "discrepant"


# --- matching -------------------------------------------------------------------


def _row(
    cell: str = "HSG",
    z: int = 6,
    let: float | None = 50.0,
    energy: float | None = None,
    alpha: float | None = 1.0,
    beta: float | None = 0.05,
    alpha_x: float | None = 0.3,
    beta_x: float | None = 0.06,
    pub: str = "P1",
    label: str = "",
) -> cc.CrossRow:
    return cc.CrossRow(pub, cell, z, let, energy, alpha, beta, alpha_x, beta_x, label)


def test_match_on_let_within_tolerance_and_same_cell_and_z() -> None:
    left = [
        _row(let=50.0, label="L0"),
        _row(let=100.0, label="L1"),
        _row(cell="T1", let=50.0, label="L2"),
    ]
    right = [
        _row(let=51.0, label="R0"),  # 2% off L0
        _row(let=110.0, label="R1"),  # 10% off L1: no
        _row(cell="T-1", let=50.0, label="R2"),  # alias of L2
        _row(cell="HSG", z=2, let=50.0, label="R3"),  # wrong Z
        _row(pub="P2", let=50.0, label="R4"),  # wrong publication
    ]
    pairs, un_l, un_r = cc.match_rows(left, right, let_tol=0.03)
    got = {(lrow.label, rrow.label, basis) for lrow, rrow, basis, _ in pairs}
    assert got == {("L0", "R0", "let"), ("L2", "R2", "let")}
    assert [r.label for r in un_l] == ["L1"]
    assert {r.label for r in un_r} == {"R1", "R3", "R4"}
    let_rel = {lrow.label: d for lrow, _, _, d in pairs}
    assert let_rel["L0"] == pytest.approx(0.02)


def test_energy_fallback_only_when_a_let_is_missing() -> None:
    left = [_row(let=None, energy=10.0, label="L0"), _row(let=30.0, energy=20.0, label="L1")]
    right = [
        _row(let=77.0, energy=10.1, label="R0"),  # L0 has no LET: energy decides
        _row(let=60.0, energy=20.0, label="R1"),  # both have LET and disagree: never paired
    ]
    pairs, un_l, un_r = cc.match_rows(left, right)
    assert [(lrow.label, rrow.label, basis, d) for lrow, rrow, basis, d in pairs] == [
        ("L0", "R0", "energy", None)
    ]
    assert [r.label for r in un_l] == ["L1"]
    assert [r.label for r in un_r] == ["R1"]


def test_duplicate_let_pairs_by_alpha_not_crosswise() -> None:
    """Two exposures at the same LET (a repeat experiment) keep their own values."""
    # Fabricated repeats, not parameter pairs copied from either database.
    left = [_row(let=50.0, alpha=1.234, label="L-a"), _row(let=50.0, alpha=1.678, label="L-b")]
    right = [_row(let=50.0, alpha=1.68, label="R-b"), _row(let=50.0, alpha=1.23, label="R-a")]
    pairs, _, _ = cc.match_rows(left, right)
    assert {(lrow.label, rrow.label) for lrow, rrow, _, _ in pairs} == {
        ("L-a", "R-a"),
        ("L-b", "R-b"),
    }


def test_one_to_one_assignment_prefers_closest_let() -> None:
    left = [_row(let=100.0, label="L0")]
    right = [_row(let=102.0, label="R-far"), _row(let=100.5, label="R-near")]
    pairs, _, un_r = cc.match_rows(left, right)
    assert [(lrow.label, rrow.label) for lrow, rrow, _, _ in pairs] == [("L0", "R-near")]
    assert [r.label for r in un_r] == ["R-far"]


# --- driver ------------------------------------------------------------------------


def test_crosscheck_summary_counts_every_quantity() -> None:
    left = [
        _row(let=50.0, alpha=1.137, beta=None, alpha_x=0.313, beta_x=0.0615, label="L0"),
        _row(let=80.0, alpha=1.449, beta=0.05, alpha_x=0.313, beta_x=0.0615, label="L1"),
        _row(let=120.0, alpha=1.0, beta=0.1, alpha_x=0.3, beta_x=0.06, label="L2"),
    ]
    right = [
        _row(
            let=50.0, alpha=1.137, beta=0.02, alpha_x=0.313, beta_x=0.061, label="R0"
        ),  # identical/missing/identical/rounding
        _row(
            let=80.0, alpha=1.52, beta=-0.05, alpha_x=0.35, beta_x=0.0615, label="R1"
        ),  # digit/sign/discrepant/identical
        _row(let=200.0, label="R2"),  # unmatched
    ]
    res = cc.crosscheck(left, right)
    assert res.n_pairs == 2
    assert [r.label for r in res.unmatched_left] == ["L2"]
    assert [r.label for r in res.unmatched_right] == ["R2"]
    assert res.summary["alpha_ion"] == {
        "identical": 1,
        "rounding": 0,
        "digitization": 1,
        "discrepant": 0,
        "sign": 0,
        "missing": 0,
    }
    assert res.summary["beta_ion"]["missing"] == 1
    assert res.summary["beta_ion"]["sign"] == 1
    assert res.summary["alpha_x"]["discrepant"] == 1
    assert res.summary["beta_x"] == {
        "identical": 1,
        "rounding": 1,
        "digitization": 0,
        "discrepant": 0,
        "sign": 0,
        "missing": 0,
    }
    worst = res.worst("alpha_x", n=1)
    assert worst[0].left.label == "L1"
    table = res.format_summary()
    assert table.splitlines()[0].startswith("quantity")
    assert "alpha_ion" in table
    assert res.pairs[0].category("alpha_ion") == "identical"


# --- converters ----------------------------------------------------------------------


def _rpb(
    radiation: str,
    exp_id: int,
    cell: str,
    *,
    pmid: str = "111",
    exp_no: int = 1,
    let: float | None = None,
    energy: float | None = None,
    alpha: float | None = 0.5,
    beta: float | None = 0.05,
) -> RadPhysBioEntry:
    y = MCDSYields(None, None, None, None)
    return RadPhysBioEntry(
        exp_id=exp_id,
        pmid=pmid,
        exp_no=exp_no,
        cell_line=cell,
        tissue="",
        cell_class="t",
        cell_cycle="a",
        radiation_type=radiation,
        energy_MeV=energy,
        rbe=None,
        let_keV_um=let,
        dose_rate_Gy_min=None,
        alpha=alpha,
        beta=beta,
        dsb_per_Gy_Gbp=None,
        dsb_from_fit=False,
        non_dsb_clusters_per_Gy_Gbp=None,
        mcds_at_energy=y,
        mcds_1keV=y,
        mcds_10keV=y,
        irradiation_conditions="m",
        radiation_source="",
    )


def test_rows_from_radphysbio_pairs_photons_and_keeps_orphans() -> None:
    entries = [
        _rpb("X-rays", 1, "HSG", exp_no=2, alpha=0.31, beta=0.06),
        _rpb("X-rays", 2, "HSG", exp_no=1, alpha=0.30, beta=0.05),  # first by #Exp
        _rpb("carbon", 1, "HSG", let=50.0, energy=1620.0, alpha=1.1, beta=0.1),
        _rpb("protons", 1, "T-1", let=10.0, energy=20.0, alpha=0.4, beta=0.04),  # no photon partner
        _rpb("X-rays", 3, "T1", pmid="222", alpha=0.2, beta=0.02),  # other publication
    ]
    rows = cc.rows_from_radphysbio(entries)
    assert [r.label for r in rows] == ["radphysbio:carbon#1", "radphysbio:protons#1"]
    carbon, proton = rows
    assert (carbon.publication, carbon.Z, carbon.let_keV_um) == ("111", 6, 50.0)
    # Nominal beam energy must not be matched to another source's target energy.
    assert carbon.energy_MeV_u is None
    assert (carbon.alpha_ion, carbon.beta_ion, carbon.alpha_x, carbon.beta_x) == (
        1.1,
        0.1,
        0.30,
        0.05,
    )
    assert (proton.Z, proton.alpha_x, proton.beta_x) == (1, None, None)


def test_rows_from_pide_maps_publication_to_pmid_and_drops_unbridged() -> None:
    recs = [
        ReferenceRecord("HSG", 6, 50.0, 30.0, 1.0, 0.05, 0.3, 0.06, A=12, source="pide:furusawa00"),
        ReferenceRecord("V79", 6, 50.0, 30.0, 1.0, 0.05, 0.3, 0.06, A=12, source="pide:wulf85"),
    ]
    rows = cc.rows_from_pide(recs, {"furusawa00": "11025645"})
    assert len(rows) == 1
    # A record whose publication is not bridged is dropped, not mis-keyed.
    recs_no = [ReferenceRecord("HSG", 6, 50.0, 30.0, 1.0, 0.05, 0.3, 0.06, source="pide:nope99")]
    assert cc.rows_from_pide(recs_no, {"furusawa00": "11025645"}) == []
    assert (rows[0].publication, rows[0].label, rows[0].Z) == ("11025645", "pide:furusawa00", 6)


def test_crosscheck_preserves_partial_pide_observations(tmp_path: Path) -> None:
    from pyradbio.benchmark.pide import load_pide, load_pide_observations

    path = tmp_path / "synthetic.csv"
    path.write_text(
        "Cells,Ion,LET,Energy,ai_paper,bi_paper,ax_paper,bx_paper,ai_fit,PublicationName\n"
        "Synthetic,12C,50,10,0.8,N/A,0.2,0.04,0.9,synthetic\n"
        "Synthetic,12C,N/A,20,N/A,0.03,0.2,0.04,0.7,synthetic\n",
        encoding="utf-8",
    )
    assert load_pide(path, variant="paper") == []  # benchmark contract unchanged
    observations = load_pide_observations(path, variant="paper")
    left = cc.rows_from_pide(observations, {"synthetic": "P1"})
    assert len(left) == 2
    right = [
        _row(cell="Synthetic", let=50, alpha=0.8, beta=0.03, alpha_x=0.2, beta_x=0.04),
        _row(cell="Synthetic", let=None, energy=20, alpha=0.7, beta=0.03, alpha_x=0.2, beta_x=0.04),
    ]
    result = cc.crosscheck(left, right)
    assert result.n_pairs == 2
    assert result.summary["alpha_ion"]["identical"] == 1
    assert result.summary["alpha_ion"]["missing"] == 1
    assert result.summary["beta_ion"]["missing"] == 1
    assert result.summary["beta_ion"]["identical"] == 1
    assert result.summary["alpha_x"]["identical"] == 2
    fallback = load_pide_observations(path, variant="paper_then_fit")
    assert fallback[0].alpha_ion == 0.8
    assert fallback[1].alpha_ion == 0.7


def test_radphysbio_nominal_energy_cannot_create_an_energy_only_match() -> None:
    ion = _rpb("carbon", 1, "Synthetic", pmid="P1", let=None, energy=120.0)
    right = cc.rows_from_radphysbio([ion])
    result = cc.crosscheck([_row(cell="Synthetic", let=None, energy=10.0)], right)
    assert result.n_pairs == 0


# --- local bibliography bridge --------------------------------------------------------


def test_no_bridge_without_a_local_file(tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> None:
    """The bridge is never bundled: without a local file there are no rows."""
    monkeypatch.setenv("PYRADBIO_PIDE_DIR", str(tmp_path))  # no bridge here
    assert cc.local_bridge_path() is None
    assert cc.load_pide_publications() == ()
    assert cc.pmid_by_publication(()) == {}


def test_local_bridge_is_read(tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.setenv("PYRADBIO_PIDE_DIR", str(tmp_path))
    (tmp_path / "pide_publications_pmid.csv").write_text(
        "pide_publication_no,publication_name,pmid,doi,year,first_author,"
        "match_method,confidence,listed_in\n"
        # Entirely synthetic rows.
        "9001,foo00,424241,,2000,Foo,author+year+volume+page,exact,synthetic\n"
        "99,extra99,424242,,2015,Extra,author+year+volume+page,exact,synthetic\n"
        "98,none98,,,2015,None,author+year,none,synthetic\n",
        encoding="utf-8",
    )
    assert cc.local_bridge_path() == tmp_path / "pide_publications_pmid.csv"
    pubs = cc.load_pide_publications()
    by_name = {p.publication_name: p for p in pubs}
    assert by_name["extra99"].pmid == "424242"
    assert by_name["foo00"].pide_publication_no == 9001
    bridged = cc.pmid_by_publication(pubs)
    assert bridged["9001"] == "424241" and bridged["99"] == "424242"
    assert bridged["foo00"] == "424241"
    assert "none98" not in bridged
    # An explicit path reads that file alone.
    src = tmp_path / "bridge.csv"
    src.write_text(
        "publication_name,pmid,doi,year,first_author,match_method,confidence,listed_in\n"
        "foo01,123,,2001,Foo,author+year+volume+page,exact,synthetic\n",
        encoding="utf-8",
    )
    (only,) = cc.load_pide_publications(src)
    assert (only.publication_name, only.pmid, only.year, only.pide_publication_no) == (
        "foo01",
        "123",
        2001,
        None,
    )


# --- the real thing --------------------------------------------------------------------


@pytest.mark.benchmark
def test_pide_vs_radphysbio_shared_experiments(pide_path: str) -> None:
    """PIDE 3.4 vs RadPhysBio on their shared publications (skips without the data)."""
    from pyradbio.benchmark.pide import load_pide_observations
    from pyradbio.benchmark.radphysbio import load_radphysbio
    from pyradbio.datasets import RADPHYSBIO, dataset_dir

    if dataset_dir(RADPHYSBIO, required=False) is None:
        pytest.skip("RadPhysBio export not configured (PYRADBIO_RADPHYSBIO_DIR)")
    bridge = cc.pmid_by_publication(cc.load_pide_publications())
    pide_rows = cc.rows_from_pide(load_pide_observations(pide_path, variant="paper"), bridge)
    rpb_rows = cc.rows_from_radphysbio(load_radphysbio())
    res = cc.crosscheck(pide_rows, rpb_rows)
    # Validate the integration without embedding experiment-specific PIDE results.
    # Partial observations change the population compared with the benchmark loader.
    assert res.n_pairs > 0
    for counts in res.summary.values():
        assert sum(counts.values()) == res.n_pairs
    for pair in res.pairs:
        assert pair.left.publication == pair.right.publication
        assert cc.cell_key(pair.left.cell_line) == cc.cell_key(pair.right.cell_line)
        assert pair.left.Z == pair.right.Z
        assert pair.match_basis == "let"
        assert pair.let_rel_diff is not None and pair.let_rel_diff <= 0.03
