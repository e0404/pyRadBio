"""Katz cellular track-structure model (Katz et al. 1971, doi:10.2307/3573247).

Goldens are the paper's own **model line** -- Table I's ``k (th.)`` and ``P``
columns, which are computed from the fitted parameters -- never its ``(exp.)``
columns, which are the experimentalists' values. See
``docs/references/Katz1971.md``.
"""

from __future__ import annotations

import math

import pytest

from pyradbio.core.radiation import Ion, dose_to_fluence
from pyradbio.models.katz import KatzCellularModel

# ---------------------------------------------------------------------------
# Table I rows: (Z, beta, z*^2/beta^2, L [MeV cm^2/g], P, k_th [rad^-1]).
# Read from the page images of Katz et al. 1971 Table I (pp. 420-422); the OCR
# text layer of this scan is unusable for numerals.
# ---------------------------------------------------------------------------
HAMSTER_TABLE_I = (
    (1, 0.145, 4.74e1, 4.60e1, 2.02e-4, 5.26e-3),
    (2, 0.145, 1.90e2, 1.91e2, 5.70e-3, 6.02e-3),
)


def hamster() -> KatzCellularModel:
    return KatzCellularModel.chinese_hamster_skarsgard1967()


class TestParameters:
    def test_the_four_published_parameters_are_carried_verbatim(self) -> None:
        """Fig. 14's printed panel: sigma_0, E_0, kappa, m."""
        m = hamster()
        assert m.sigma0_cm2 == pytest.approx(5.0e-7)
        assert m.E0_erg_cm3 == pytest.approx(1.95e4)
        assert m.kappa == pytest.approx(1400.0)
        assert m.m == pytest.approx(2.5)

    def test_cgs_parameters_convert_to_repo_units(self) -> None:
        """1 Gy = 1e4 erg/cm^3 in water; 1 cm^2 = 1e8 um^2."""
        m = hamster()
        assert m.E0_Gy == pytest.approx(1.95)
        assert m.sigma0_um2 == pytest.approx(50.0)

    @pytest.mark.parametrize("bad", [{"m": 0.5}, {"kappa": 0.0}, {"sigma0_cm2": -1.0}])
    def test_unphysical_parameters_are_rejected(self, bad: dict[str, float]) -> None:
        kwargs = {"sigma0_cm2": 5e-7, "E0_erg_cm3": 1.95e4, "kappa": 1400.0, "m": 2.5}
        kwargs.update(bad)
        with pytest.raises(ValueError):
            KatzCellularModel(**kwargs)  # type: ignore[arg-type]


class TestEffectiveCharge:
    def test_eq12_is_the_barkas_effective_charge_we_already_ship(self) -> None:
        """Katz eq. (12) ``z* = Z[1-exp(-125 beta Z^-2/3)]`` is our helper."""
        from pyradbio.track.stopping_power import effective_charge

        for z, beta in ((1, 0.145), (2, 0.145), (18, 0.111)):
            assert KatzCellularModel.effective_charge(z, beta) == pytest.approx(
                effective_charge(z, beta)
            )

    @pytest.mark.golden
    @pytest.mark.parametrize(("z", "beta", "z2_over_b2"), [r[:3] for r in HAMSTER_TABLE_I])
    def test_z_star_squared_over_beta_squared_matches_table_i(
        self, z: int, beta: float, z2_over_b2: float
    ) -> None:
        """Table I column ``z^2/beta^2`` (doi:10.2307/3573247)."""
        zs = KatzCellularModel.effective_charge(z, beta)
        assert zs * zs / (beta * beta) == pytest.approx(z2_over_b2, rel=5e-3)


class TestGrainCountRegime:
    @pytest.mark.golden
    @pytest.mark.parametrize(("z", "beta", "_zb", "_L", "p_expected", "_k"), HAMSTER_TABLE_I)
    def test_ion_kill_probability_matches_table_i(
        self, z: int, beta: float, _zb: float, _L: float, p_expected: float, _k: float
    ) -> None:
        """Eq. (8) ``P = sigma/sigma_0 = [1-exp(-z*^2/(kappa beta^2))]^m``."""
        assert hamster().ion_kill_probability(z, beta) == pytest.approx(p_expected, rel=1e-2)

    def test_probability_is_bounded_and_monotonic_in_z(self) -> None:
        m = hamster()
        previous = -1.0
        for z in range(1, 19):
            p = m.ion_kill_probability(z, 0.145)
            assert 0.0 <= p <= 1.0
            assert p > previous
            previous = p

    def test_cross_section_is_sigma0_times_probability(self) -> None:
        """Eq. (8): sigma = sigma_0 * P, capped by sigma_0 in this regime."""
        m = hamster()
        p = m.ion_kill_probability(6, 0.143)
        assert m.inactivation_cross_section_um2(6, 0.143) == pytest.approx(m.sigma0_um2 * p)


class TestRadiosensitivity:
    @pytest.mark.golden
    @pytest.mark.parametrize(("z", "beta", "_zb", "let_mev_cm2_g", "_p", "k_th"), HAMSTER_TABLE_I)
    def test_eq14_reproduces_table_i_theoretical_radiosensitivity(
        self, z: int, beta: float, _zb: float, let_mev_cm2_g: float, _p: float, k_th: float
    ) -> None:
        """Eq. (14) ``k = sigma_0 P/L + (1-P)/E_0`` vs Table I ``k (th.)``.

        Fed the paper's own tabulated ``L`` so the check isolates the model from
        this package's stopping-power tables.
        """
        k = hamster().radiosensitivity_per_Gy(z, beta, let_keV_um=let_mev_cm2_g * 0.1)
        assert k / 100.0 == pytest.approx(k_th, rel=1e-2)

    def test_gamma_kill_dominates_at_low_let_and_ion_kill_at_high(self) -> None:
        """The two terms of eq. (14) swap rank across the LET range."""
        m = hamster()
        low = m.ion_kill_probability(1, 0.145)
        high = m.ion_kill_probability(18, 0.111)
        assert low < 0.01 < 0.5 < high


class TestSurvival:
    def test_eq10_is_the_product_of_ion_kill_and_gamma_kill(self) -> None:
        m = hamster()
        ion = Ion(Z=6, A=12, energy_MeV_u=10.0)
        d = 3.0
        s = m.survival(ion, d, let_keV_um=100.0)
        assert s == pytest.approx(
            m.ion_kill_survival(ion, d, let_keV_um=100.0)
            * m.gamma_kill_survival(ion, d, let_keV_um=100.0)
        )

    def test_survival_is_one_at_zero_dose_and_decreasing(self) -> None:
        m = hamster()
        ion = Ion(Z=6, A=12, energy_MeV_u=10.0)
        assert m.survival(ion, 0.0, let_keV_um=100.0) == pytest.approx(1.0)
        previous = 1.0
        for d in (0.5, 1.0, 2.0, 5.0, 10.0):
            s = m.survival(ion, d, let_keV_um=100.0)
            assert 0.0 < s < previous
            previous = s

    def test_pure_gamma_kill_reduces_to_multi_target_single_hit(self) -> None:
        """With P = 0 eq. (10) must collapse to ``1-(1-e^-D/E0)^m``, eq. (3)."""
        m = KatzCellularModel(sigma0_cm2=1e-30, E0_erg_cm3=1.95e4, kappa=1e12, m=2.5)
        ion = Ion(Z=1, A=1, energy_MeV_u=10.0)
        for d in (1.0, 3.0, 6.0):
            expected = 1.0 - (1.0 - math.exp(-d / 1.95)) ** 2.5
            assert m.survival(ion, d, let_keV_um=4.6) == pytest.approx(expected, rel=1e-6)

    def test_survival_curve_has_a_shoulder_when_m_exceeds_one(self) -> None:
        """m > 1 means zero initial slope in the gamma-kill channel."""
        m = KatzCellularModel(sigma0_cm2=1e-30, E0_erg_cm3=1.95e4, kappa=1e12, m=2.5)
        ion = Ion(Z=1, A=1, energy_MeV_u=10.0)
        tiny = m.survival(ion, 1e-4, let_keV_um=4.6)
        assert tiny == pytest.approx(1.0, abs=1e-8)

    def test_negative_dose_is_rejected(self) -> None:
        with pytest.raises(ValueError):
            hamster().survival(Ion(Z=6, A=12, energy_MeV_u=10.0), -1.0, let_keV_um=100.0)


class TestExtrapolatedCrossSection:
    @pytest.mark.golden
    def test_eq15_matches_table_i_for_the_self_consistent_row(self) -> None:
        """Table I, Chinese hamster Z=2: ``sigma_ext (th.) = 1.84e-8 cm^2``.

        Only this row is used. Table I's Z=1 row prints ``4.02e-9 cm^2``, which
        does **not** follow from its own printed ``k = 5.26e-3 rad^-1`` and
        ``L = 46.0 MeV cm^2/g`` -- eq. (15) gives 3.88e-9, a 3.6% internal
        inconsistency in the paper. See ``docs/references/Katz1971.md``.
        """
        sigma_ext_cm2 = hamster().extrapolated_cross_section_um2(2, 0.145, let_keV_um=19.1) / 1e8
        assert sigma_ext_cm2 == pytest.approx(1.84e-8, rel=1e-2)

    def test_eq15_is_the_dose_to_fluence_inverse_of_eq14(self) -> None:
        """sigma_ext = k L must undo the fluence conversion inside eq. (14)."""
        m = hamster()
        let = 19.1
        k = m.radiosensitivity_per_Gy(2, 0.145, let_keV_um=let)
        fluence_per_Gy = dose_to_fluence(1.0, let, m.density_g_cm3)
        assert m.extrapolated_cross_section_um2(2, 0.145, let_keV_um=let) == pytest.approx(
            k / fluence_per_Gy * 1e8
        )


class TestPresets:
    def test_all_ten_published_parameter_sets_are_available(self) -> None:
        assert len(KatzCellularModel.published_presets()) == 10

    def test_presets_carry_their_figure_and_data_source(self) -> None:
        for name, factory in KatzCellularModel.published_presets().items():
            model = factory()
            assert model.reference_figure, name
            assert model.data_source, name
            assert model.reference_doi == "10.2307/3573247", name

    @pytest.mark.golden
    def test_oxygen_effect_shows_only_in_E0(self) -> None:
        """Paper p. 419: 'E_0 varies, while sigma_0 and m remain constant'."""
        anoxic = KatzCellularModel.t1_kidney_todd1966_anoxic()
        aerobic = KatzCellularModel.t1_kidney_todd1966_aerobic()
        assert anoxic.sigma0_cm2 == aerobic.sigma0_cm2
        assert anoxic.m == aerobic.m
        assert anoxic.E0_erg_cm3 > aerobic.E0_erg_cm3

    @pytest.mark.golden
    def test_todd_sigma0_is_the_corrected_exponent(self) -> None:
        """Figs. 10/11 print 6.7e-7 cm^2; the IAEA companion's table has a typo.

        Katz's own IAEA-SM-145/9 Table I prints this as ``6.70x10^27 cm^2``,
        which is impossible. Both Radiat. Res. figure panels print 6.7x10^-7.
        """
        assert KatzCellularModel.t1_kidney_todd1966_anoxic().sigma0_cm2 == pytest.approx(6.7e-7)
        assert KatzCellularModel.t1_kidney_todd1966_anoxic().sigma0_um2 == pytest.approx(67.0)


class TestLQAdapter:
    def test_alpha_beta_reports_eq14_with_zero_beta(self) -> None:
        """Katz is not an LQ model: its shoulder is D^m, not quadratic."""
        m = hamster()
        ion = Ion(Z=6, A=12, energy_MeV_u=10.0)
        lq = m.alpha_beta(ion)
        assert lq.beta == 0.0
        assert lq.alpha > 0.0

    def test_a_bare_ion_falls_back_to_the_default_stopping_power(self) -> None:
        """Presets ship with the library default, as LEM does."""
        from pyradbio.track.stopping_power import DefaultStoppingPower

        assert isinstance(hamster().stopping_power, DefaultStoppingPower)
        assert hamster().alpha_beta(Ion(Z=6, A=12, energy_MeV_u=10.0)).alpha > 0.0

    def test_an_explicit_let_overrides_the_stopping_power(self) -> None:
        """At fixed *dose* a higher LET means fewer traversals, so more survival.

        ``F = D/L`` (eq. 11), and only the ion-kill term of eq. (10) sees the
        fluence -- gamma-kill depends on ``P`` and ``D`` alone. So raising LET at
        constant dose thins the traversals and lifts survival.
        """
        ion = Ion(Z=6, A=12, energy_MeV_u=10.0)
        m = hamster()
        derived = m.survival(ion, 2.0)
        assert m.survival(ion, 2.0, let_keV_um=1000.0) > derived
        assert m.survival(ion, 2.0, let_keV_um=1.0) < derived

    def test_with_stopping_power_rebinds_and_preserves_provenance(self) -> None:
        from pyradbio.track.stopping_power import SimpleBetheStoppingPower

        replacement = SimpleBetheStoppingPower()
        rebound = hamster().with_stopping_power(replacement)
        assert rebound.stopping_power is replacement
        assert rebound.sigma0_cm2 == hamster().sigma0_cm2
        assert rebound.reference_figure == "Fig. 14"
        assert rebound.data_source == "Skarsgard et al. 1967"


class TestTrackWidthRegimeIsOutOfScope:
    def test_grain_count_saturation_is_documented_not_exceeded(self) -> None:
        """Eq. (8) saturates at sigma_0; eq. (4) is not implemented.

        Katz reports sigma/sigma_0 ~ 2 for argon on kidney cells (p. 419), which
        this grain-count-only implementation cannot produce by construction.
        """
        m = KatzCellularModel.t1_kidney_todd1966_aerobic()
        assert m.inactivation_cross_section_um2(18, 0.111) <= m.sigma0_um2
        assert m.track_width_regime_implemented is False


class TestGuardRails:
    """Every validation branch, so the error surface is pinned not incidental."""

    def test_non_positive_E0_is_rejected(self) -> None:
        with pytest.raises(ValueError, match="E0_erg_cm3"):
            KatzCellularModel(sigma0_cm2=5e-7, E0_erg_cm3=0.0, kappa=1400.0, m=2.5)

    @pytest.mark.parametrize("beta", [0.0, 1.0, -0.1, 1.5])
    def test_beta_outside_the_open_unit_interval_is_rejected(self, beta: float) -> None:
        with pytest.raises(ValueError, match="beta"):
            hamster().ion_kill_probability(6, beta)

    def test_non_positive_explicit_let_is_rejected(self) -> None:
        ion = Ion(Z=6, A=12, energy_MeV_u=10.0)
        with pytest.raises(ValueError, match="let_keV_um"):
            hamster().survival(ion, 1.0, let_keV_um=0.0)

    def test_gamma_kill_rejects_negative_dose(self) -> None:
        ion = Ion(Z=6, A=12, energy_MeV_u=10.0)
        with pytest.raises(ValueError, match="dose"):
            hamster().gamma_kill_survival(ion, -1.0, let_keV_um=100.0)

    def test_radiosensitivity_rejects_non_positive_let(self) -> None:
        with pytest.raises(ValueError, match="let_keV_um"):
            hamster().radiosensitivity_per_Gy(6, 0.145, let_keV_um=-1.0)

    def test_mixed_fields_are_refused_rather_than_dose_averaged(self) -> None:
        """Eq. (10) is a product of two non-LQ modes; LQ mixing does not apply."""
        from pyradbio.core.radiation import RadiationField, SpectrumComponent

        field = RadiationField(
            components=(SpectrumComponent(ion=Ion(Z=6, A=12, energy_MeV_u=10.0), dose_Gy=1.0),)
        )
        with pytest.raises(NotImplementedError, match="mixed-field"):
            hamster().alpha_beta(field)


class TestOverkill:
    """The model's characteristic behaviour, not pinned to any printed number."""

    def test_low_let_radiosensitivity_tends_to_the_gamma_kill_floor(self) -> None:
        """As P -> 0 every equation collapses to eq. (14)'s second term, 1/E_0."""
        m = hamster()
        k = m.radiosensitivity_per_Gy(1, 0.43, let_keV_um=0.72)  # ~100 MeV proton
        assert k == pytest.approx(1.0 / m.E0_Gy, rel=1e-3)

    def test_radiosensitivity_peaks_then_falls_with_let(self) -> None:
        """Overkill: once P saturates, eq. (14) becomes sigma_0/L and decays.

        This is the RBE-vs-LET maximum, and it drops out of the two-mode
        structure rather than being imposed.
        """
        from pyradbio.track.stopping_power import SimpleBetheStoppingPower

        sp = SimpleBetheStoppingPower()
        m = hamster()
        ks = []
        for z, a, e in ((1, 1, 100.0), (2, 4, 10.0), (6, 12, 10.0), (18, 40, 10.0)):
            ion = Ion(Z=z, A=a, energy_MeV_u=e)
            ks.append(m.radiosensitivity_per_Gy(z, ion.beta(), let_keV_um=sp.let_keV_um(ion)))
        assert ks[0] < ks[1] < ks[2], "k must rise while P is still climbing"
        assert ks[3] < ks[0], "argon must fall below the low-LET floor (overkill)"

    def test_saturated_ion_kill_leaves_only_sigma0_over_L(self) -> None:
        """At P = 1 the gamma-kill term is gone: k L -> sigma_0 exactly (eq. 15)."""
        m = hamster()
        ion = Ion(Z=18, A=40, energy_MeV_u=10.0)
        assert m.ion_kill_probability(18, ion.beta()) == pytest.approx(1.0, abs=1e-3)
        assert m.extrapolated_cross_section_um2(
            18, ion.beta(), let_keV_um=1268.64
        ) == pytest.approx(m.sigma0_um2, rel=1e-3)


class TestPhotonResponse:
    """E_0 and m *are* the gamma-ray parameters, so the model is self-contained."""

    def test_photon_survival_is_eq3_on_the_whole_dose(self) -> None:
        m = hamster()
        for d in (0.0, 1.0, 3.0, 8.0):
            expected = 1.0 - (1.0 - math.exp(-d / 1.95)) ** 2.5
            assert m.photon_survival(d) == pytest.approx(expected, rel=1e-9)

    def test_photon_survival_is_the_P_to_zero_limit_of_eq10(self) -> None:
        """A vanishing ion-kill probability must recover the gamma-ray curve."""
        m = KatzCellularModel(sigma0_cm2=1e-30, E0_erg_cm3=1.95e4, kappa=1e12, m=2.5)
        ion = Ion(Z=1, A=1, energy_MeV_u=10.0)
        for d in (1.0, 4.0):
            assert m.survival(ion, d, let_keV_um=4.6) == pytest.approx(
                m.photon_survival(d), rel=1e-6
            )

    def test_photon_survival_has_the_multi_target_shoulder(self) -> None:
        m = hamster()
        assert m.photon_survival(0.0) == pytest.approx(1.0)
        assert m.photon_survival(0.01) == pytest.approx(1.0, abs=1e-4)
        assert m.photon_survival(2.0) < m.photon_survival(1.0)

    def test_photon_final_slope_is_one_over_E0(self) -> None:
        """The multi-target curve's asymptotic slope, i.e. eq. (14) at P = 0."""
        m = hamster()
        hi, lo = m.photon_survival(20.0), m.photon_survival(21.0)
        assert math.log(hi / lo) == pytest.approx(1.0 / m.E0_Gy, rel=1e-3)

    def test_photon_survival_rejects_negative_dose(self) -> None:
        with pytest.raises(ValueError, match="dose"):
            hamster().photon_survival(-1.0)


class TestKillTerms:
    """``kill_terms_per_Gy`` is eq. (14) split into ion-kill and gamma-kill."""

    @pytest.mark.parametrize(
        ("z", "beta", "let"), [(1, 0.145, 4.6), (2, 0.145, 19.1), (18, 0.1, 1200.0)]
    )
    def test_terms_are_eq14_and_sum_to_k_bit_for_bit(self, z: int, beta: float, let: float) -> None:
        m = hamster()
        terms = m.kill_terms_per_Gy(z, beta, let_keV_um=let)
        p = m.ion_kill_probability(z, beta)
        assert terms.ion_kill_probability == p
        assert terms.ion_kill == m.sigma0_cm2 * p * dose_to_fluence(1.0, let, m.density_g_cm3)
        assert terms.gamma_kill == (1.0 - p) / m.E0_Gy
        assert terms.total == m.radiosensitivity_per_Gy(z, beta, let_keV_um=let)

    def test_overkill_leaves_only_sigma0_over_L(self) -> None:
        """Once P -> 1 the gamma-kill term vanishes and k collapses to sigma_0/L."""
        m = hamster()
        terms = m.kill_terms_per_Gy(18, 0.046, let_keV_um=1268.6)  # argon, ~1 MeV/u
        assert terms.ion_kill_probability == pytest.approx(1.0, abs=1e-6)
        assert terms.gamma_kill == pytest.approx(0.0, abs=1e-6)
        assert terms.total < 1.0 / m.E0_Gy  # below the low-LET gamma-kill floor

    def test_rejects_non_positive_let(self) -> None:
        with pytest.raises(ValueError, match="let_keV_um"):
            hamster().kill_terms_per_Gy(6, 0.145, let_keV_um=0.0)


class TestPhotonDoseForSurvival:
    """The closed-form inverse of eq. (3), the photon side of every Katz RBE."""

    @pytest.mark.parametrize("s", [0.9, 0.5, 0.1, 1e-3, 1e-8])
    def test_inverts_photon_survival(self, s: float) -> None:
        m = hamster()
        assert m.photon_survival(m.photon_dose_for_survival(s)) == pytest.approx(s, rel=1e-9)

    def test_is_the_numerator_of_rbe_at_survival(self) -> None:
        m = hamster()
        ion = Ion(Z=6, A=12, energy_MeV_u=10.0)
        rbe = m.rbe_at_survival(ion, 0.1, let_keV_um=162.0)
        d_ion = m.dose_for_survival(ion, 0.1, let_keV_um=162.0)
        assert rbe * d_ion == pytest.approx(m.photon_dose_for_survival(0.1), rel=1e-12)

    def test_full_survival_needs_no_dose(self) -> None:
        assert hamster().photon_dose_for_survival(1.0) == 0.0

    @pytest.mark.parametrize("bad", [0.0, -0.1, 1.5])
    def test_rejects_a_survival_outside_zero_one(self, bad: float) -> None:
        with pytest.raises(ValueError, match="survival fraction"):
            hamster().photon_dose_for_survival(bad)
