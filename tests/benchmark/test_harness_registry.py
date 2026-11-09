"""The benchmark model registry: which models exist and how each is evaluated.

Numbers here pin the *wiring* (route, scope, parameter set), not the biophysics --
each model family has its own test module for that.
"""

from __future__ import annotations

import pytest

from pyradbio.benchmark.harness import (
    benchmark_models,
    build_model_factory,
    build_model_spec,
    model_caveats,
    run_benchmark,
)
from pyradbio.benchmark.pide import PideEntry
from pyradbio.core.cell import CellTarget
from pyradbio.core.photon import PhotonParameters
from pyradbio.core.radiation import Ion
from pyradbio.models.lem.lem1 import LEM1
from pyradbio.models.mkm import (
    OSMK2021,
    OSMK2023,
    IMKModel,
    MCFMKMModel,
    ModifiedMKM,
    StochasticMKM,
)
from pyradbio.models.phenomenological import (
    Carabe,
    LinearScaling,
    MairaniHelium,
    McNamara,
    Wedenberg,
    WilkensOelfke,
)

PHENOMENOLOGICAL = ("wilkens", "carabe", "frese", "wedenberg", "mcnamara", "mairani")


def _entry(**kw: object) -> PideEntry:
    base: dict[str, object] = dict(
        cell_line="X",
        Z=1,
        A=1,
        energy_MeV_u=100.0,
        let_keV_um=10.0,
        alpha_ion=0.5,
        beta_ion=0.04,
        alpha_x=0.2,
        beta_x=0.05,
    )
    base.update(kw)
    return PideEntry(**base)  # type: ignore[arg-type]


def test_every_phenomenological_model_is_registered() -> None:
    assert set(PHENOMENOLOGICAL) <= set(benchmark_models())


@pytest.mark.parametrize("name", PHENOMENOLOGICAL)
def test_phenomenological_models_take_the_let_route(name: str) -> None:
    spec = build_model_spec(name)
    assert spec.let_based
    # (alpha/beta)_x is their tissue axis, so a row without beta_x has no scope.
    assert spec.needs_beta_x


@pytest.mark.parametrize(
    ("name", "expected_z"),
    [
        ("wilkens", frozenset({1})),
        ("carabe", frozenset({1})),
        ("frese", frozenset({1})),
        ("wedenberg", frozenset({1})),
        ("mcnamara", frozenset({1})),
        ("mairani", frozenset({2})),  # helium, not protons
    ],
)
def test_phenomenological_scope_follows_the_papers_radiation_mode(
    name: str, expected_z: frozenset[int]
) -> None:
    assert build_model_spec(name).applicable_z == expected_z


@pytest.mark.parametrize(
    ("name", "cls"),
    [
        ("wilkens", WilkensOelfke),
        ("carabe", Carabe),
        ("frese", LinearScaling),
        ("wedenberg", Wedenberg),
        ("mcnamara", McNamara),
        ("mairani", MairaniHelium),
    ],
)
def test_registered_model_predicts_its_own_closed_form(name: str, cls: type) -> None:
    """The registry must add no parameters of its own: same photon, same number."""
    photon = PhotonParameters(alpha_x=0.2, beta_x=0.05)
    expected = cls(photon=photon).alpha_beta_from_let(10.0).alpha
    z = 2 if name == "mairani" else 1
    result = run_benchmark([_entry(Z=z, A=2 * z, let_keV_um=10.0)], spec=build_model_spec(name))
    assert result.n_points == 1
    assert result.points[0].predicted_alpha == pytest.approx(expected)


def test_phenomenological_model_uses_the_entry_let_not_our_stopping_power() -> None:
    """Two rows at the same energy but different tabulated LET must differ."""
    spec = build_model_spec("wedenberg")
    entries = [_entry(let_keV_um=2.0), _entry(let_keV_um=20.0)]
    result = run_benchmark(entries, spec=spec)
    assert result.n_points == 2
    lo, hi = (p.predicted_alpha for p in result.points)
    assert hi > lo


def test_unknown_model_name_lists_the_registry() -> None:
    with pytest.raises(ValueError, match="unknown model 'nope'"):
        build_model_spec("nope")


# --- the MKM family ---------------------------------------------------------

MKM_FAMILY = ("mkm", "mmkm", "smk", "mcf", "osmk2021", "osmk2023", "imk")


def test_every_mkm_model_is_registered() -> None:
    assert set(MKM_FAMILY) <= set(benchmark_models())


@pytest.mark.parametrize("name", MKM_FAMILY)
def test_mkm_models_take_the_ion_route_with_no_species_limit(name: str) -> None:
    spec = build_model_spec(name)
    assert not spec.let_based
    assert spec.applicable_z is None
    assert not spec.needs_beta_x


@pytest.mark.parametrize(
    ("name", "build"),
    [
        ("mmkm", lambda: ModifiedMKM.hsg_mk_inaniwa2018()),
        ("smk", lambda: StochasticMKM.hsg_inaniwa2018()),
        ("mcf", lambda: MCFMKMModel()),
        ("osmk2021", lambda: OSMK2021.hsg_inaniwa2021()),
        ("osmk2023", lambda: OSMK2023.hsg_inaniwa2023()),
        ("imk", lambda: IMKModel.hsg_matsuya2025()),
    ],
)
def test_registered_mkm_model_is_its_published_preset(name: str, build: object) -> None:
    """Pins the parameter set, so the harness's own geometry defaults cannot leak in.

    SMK's R_n = 8.1 um is measured (Inaniwa & Kanematsu 2018) and mMKM's is 4.3;
    silently substituting the LEM-ish 5.0 would be the classic trap.
    """
    ion = Ion(Z=6, A=12, energy_MeV_u=100.0)
    expected = build().alpha_beta(ion).alpha  # type: ignore[operator]
    result = run_benchmark([_entry(Z=6, A=12, energy_MeV_u=100.0)], spec=build_model_spec(name))
    assert result.n_points == 1
    assert result.points[0].predicted_alpha == pytest.approx(expected)


@pytest.mark.parametrize("name", ("mmkm", "smk", "mcf", "osmk2021", "osmk2023", "imk"))
def test_preset_models_reject_geometry_overrides(name: str) -> None:
    """Silently ignoring --rn would be worse than refusing it."""
    with pytest.raises(ValueError, match="published parameter set"):
        build_model_spec(name, nucleus_radius_um=5.0)


def test_lem_and_mkm_still_take_their_documented_defaults() -> None:
    photon = PhotonParameters(alpha_x=0.2, beta_x=0.05, D_t=30.0)
    expected = LEM1(photon=photon, cell=CellTarget(nucleus_radius_um=5.0))
    ion = Ion(Z=6, A=12, energy_MeV_u=100.0)
    result = run_benchmark([_entry(Z=6, A=12, energy_MeV_u=100.0)], spec=build_model_spec("lem1"))
    assert result.points[0].predicted_alpha == pytest.approx(expected.alpha_beta(ion).alpha)


@pytest.mark.benchmark
@pytest.mark.parametrize("name", benchmark_models())
def test_every_registered_model_runs_on_real_pide(name: str, pide_path: str) -> None:
    """Skipped unless --pide <path> is given. A wiring regression net, not a score.

    Deliberately asserts only that each model evaluates points and stays inside a
    decade on average -- the residual against measured data must not be used to
    choose between models or parameter variants (see AGENTS.md).
    """
    from pyradbio.benchmark.pide import load_pide

    entries = load_pide(pide_path)
    result = run_benchmark(entries, model_name=name, spec=build_model_spec(name), limit=20)
    assert result.n_points > 0
    assert result.mean_abs_log_error_alpha < 1.0


def test_every_registered_model_states_its_caveats() -> None:
    """A model with no caveat registered would ship a score with no health warning."""
    for name in benchmark_models():
        assert model_caveats(name), name


def test_caveats_of_an_unregistered_model_are_empty() -> None:
    assert model_caveats("nope") == ()


def test_build_model_factory_returns_the_specs_factory() -> None:
    photon = PhotonParameters(alpha_x=0.2, beta_x=0.05)
    model = build_model_factory("wedenberg")(0.2, 0.05)
    ion_free = Wedenberg(photon=photon).alpha_beta_from_let(5.0).alpha
    assert model.alpha_beta_from_let(5.0).alpha == pytest.approx(ion_free)  # type: ignore[attr-defined]


# --- the Rorvik-sourced additions ------------------------------------------

SECOND_HAND = ("tilly2", "tilly10")

# Everything else in the Rorvik-introduced group is read from its primary.
PRIMARY_SOURCED = ("chen", "jones", "roru", "peeler", "unkelbach")


@pytest.mark.parametrize("name", SECOND_HAND)
def test_tilly_still_declares_its_second_hand_slope(name: str) -> None:
    """Tilly et al. print no slope, so this one stays attributable to Rorvik."""
    joined = " ".join(model_caveats(name))
    assert "not printed by its own paper" in joined
    assert "10.1088/1361-6560/aad9db" in joined


@pytest.mark.parametrize("name", PRIMARY_SOURCED)
def test_primary_sourced_models_no_longer_claim_second_hand_coefficients(
    name: str,
) -> None:
    """Having read the primary, the weaker-provenance note must not linger."""
    assert "not printed by its own paper" not in " ".join(model_caveats(name))


def test_peeler_caveat_names_its_real_fit_window() -> None:
    """The LET and tissue ranges were transposed once; the caveat must not repeat it."""
    joined = " ".join(model_caveats("peeler"))
    assert "0.9-19.0 keV/um" in joined
    assert "2.6-5.1 Gy" in joined


def test_unkelbach_caveat_says_it_is_not_a_biological_model() -> None:
    joined = " ".join(model_caveats("unkelbach"))
    assert "not a biological model" in joined
    assert "dose" in joined


def test_chen_caveat_warns_off_the_rorvik_reformulation() -> None:
    """Its worst failure mode is someone re-deriving it from the appendix."""
    joined = " ".join(model_caveats("chen"))
    assert "alpha_x" in joined
    assert "factor of ~10" in joined


def test_jones_caveat_records_both_constants_it_diverges_on() -> None:
    joined = " ".join(model_caveats("jones"))
    assert "alpha_x" in joined
    assert "2.7092" in joined and "0.06" in joined


# --- Katz et al. 1971 -------------------------------------------------------

KATZ = ("katz_hamster", "katz_hela", "katz_t1_todd", "katz_t1_barendsen")


def test_every_katz_preset_is_registered() -> None:
    assert set(KATZ) <= set(benchmark_models())


@pytest.mark.parametrize("name", KATZ)
def test_katz_takes_the_ion_route_not_the_let_route(name: str) -> None:
    """Eq. (8) needs (z*, beta), which a scalar LET_d cannot supply."""
    spec = build_model_spec(name)
    assert not spec.let_based
    assert not spec.needs_beta_x
    assert spec.applicable_z is None


@pytest.mark.parametrize("name", KATZ)
def test_katz_presets_refuse_harness_geometry(name: str) -> None:
    """The four parameters are fitted as a unit; a stray R_n must not leak in."""
    with pytest.raises(ValueError, match="published parameter set"):
        build_model_spec(name, nucleus_radius_um=5.0)


@pytest.mark.parametrize("name", KATZ)
def test_katz_caveat_says_alpha_is_not_an_lq_initial_slope(name: str) -> None:
    """The single most misreadable thing about this family's score."""
    joined = " ".join(model_caveats(name))
    assert "not an LQ initial slope" in joined
    assert "grain-count" in joined


@pytest.mark.parametrize(
    ("name", "factory_name"),
    [
        ("katz_hamster", "chinese_hamster_skarsgard1967"),
        ("katz_hela", "hela_deering1962"),
        ("katz_t1_todd", "t1_kidney_todd1966_aerobic"),
        ("katz_t1_barendsen", "t1_kidney_barendsen1966_o2"),
    ],
)
def test_registered_katz_model_is_the_published_preset(name: str, factory_name: str) -> None:
    """The registry must add no parameters of its own."""
    from pyradbio.models.katz import KatzCellularModel

    expected = getattr(KatzCellularModel, factory_name)()
    built = build_model_factory(name)(0.2, 0.05)
    assert built.sigma0_cm2 == expected.sigma0_cm2  # type: ignore[attr-defined]
    assert built.E0_erg_cm3 == expected.E0_erg_cm3  # type: ignore[attr-defined]
    assert built.kappa == expected.kappa  # type: ignore[attr-defined]
    assert built.m == expected.m  # type: ignore[attr-defined]


def test_katz_is_photon_independent() -> None:
    """Two rows with different photon references must get the same model."""
    factory = build_model_factory("katz_hamster")
    assert factory(0.2, 0.05) is factory(0.9, 0.01)


def test_katz_predicts_eq14_on_a_pide_row() -> None:
    """End to end: the benchmark's predicted alpha is eq. (14) at the row's ion."""
    from pyradbio.models.katz import KatzCellularModel

    entry = _entry(Z=6, A=12, energy_MeV_u=100.0)
    result = run_benchmark([entry], spec=build_model_spec("katz_hamster"))
    assert result.n_points == 1
    model = KatzCellularModel.chinese_hamster_skarsgard1967()
    expected = model.alpha_beta(Ion(Z=6, A=12, energy_MeV_u=100.0)).alpha
    assert result.points[0].predicted_alpha == pytest.approx(expected)


# --- RMF (Carlson 2008 / Frese 2012) ---------------------------------------


def test_rmf_is_registered() -> None:
    assert "rmf" in benchmark_models()


def test_rmf_takes_the_ion_route_and_needs_beta_x() -> None:
    """Its DSB yield keys on Z_eff^2/beta^2, and kappa = 2 beta_x/Sigma_x^2."""
    spec = build_model_spec("rmf")
    assert not spec.let_based
    assert spec.needs_beta_x
    assert spec.applicable_z is None


def test_rmf_refuses_harness_geometry() -> None:
    with pytest.raises(ValueError, match="published parameter set"):
        build_model_spec("rmf", nucleus_radius_um=5.0)


def test_rmf_is_photon_dependent_and_cached_per_pair() -> None:
    """Unlike the MKM/Katz presets, the row's own photon pair IS the tissue."""
    factory = build_model_factory("rmf")
    first = factory(0.2, 0.05)
    assert factory(0.2, 0.05) is first
    assert factory(0.9, 0.01) is not first


def test_rmf_derives_its_tissue_parameters_from_the_row() -> None:
    """theta and kappa must come from the row, not from a preset."""
    factory = build_model_factory("rmf")
    model = factory(0.2, 0.05)
    assert model.kappa_per_Gbp2 == pytest.approx(2 * 0.05 / 8.27**2)  # type: ignore[attr-defined]


def test_rmf_caveat_names_the_let_ceiling_and_the_diameter_trap() -> None:
    joined = " ".join(model_caveats("rmf"))
    assert "150 keV" in joined
    assert "DIAMETER" in joined
    assert "Co-60" in joined


def test_rmf_predicts_its_own_closed_form_on_a_pide_row() -> None:
    from pyradbio.core.photon import PhotonParameters
    from pyradbio.models.rmf import CO60_REFERENCE, RMFModel

    entry = _entry(Z=6, A=12, energy_MeV_u=200.0, alpha_x=0.2, beta_x=0.05)
    result = run_benchmark([entry], spec=build_model_spec("rmf"))
    assert result.n_points == 1
    expected = RMFModel(
        photon=PhotonParameters(alpha_x=0.2, beta_x=0.05), reference_radiation=CO60_REFERENCE
    ).alpha_beta(Ion(Z=6, A=12, energy_MeV_u=200.0))
    assert result.points[0].predicted_alpha == pytest.approx(expected.alpha)


def test_rmf_row_without_beta_x_is_out_of_scope_not_an_error() -> None:
    result = run_benchmark([_entry(beta_x=0.0)], spec=build_model_spec("rmf"))
    assert result.n_out_of_scope == 1
    assert result.n_points == 0
