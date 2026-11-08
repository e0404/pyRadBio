"""RMF model (Carlson et al. 2008; Frese et al. 2012).

Goldens are the papers' own printed model lines -- Frese Table 1's ``theta`` and
``kappa`` columns, which are *computed* from the printed reference parameters,
and the limits both papers state in words. Never a digitized figure point.
"""

from __future__ import annotations

import math
import warnings

import numpy as np
import pytest

from pyradbio.core.photon import PhotonParameters
from pyradbio.core.radiation import Ion
from pyradbio.models.rmf import CO60_REFERENCE, X_RAY_200KVP_REFERENCE, RMFModel
from pyradbio.track.stopping_power import SimpleBetheStoppingPower

# Frese et al. 2012 Table 1: (name, alpha_x, beta_x, theta [Gbp], kappa [Gbp^2],
# theta/kappa [Gbp^-1], reference radiation).
FRESE_TABLE_1 = (
    ("t1_furusawa2000", 0.0305, 0.0585, "0.0033", "1.6e-3", 2.15, "200 kVp"),
    ("hsg_furusawa2000", 0.313, 0.0615, "0.0359", "1.6e-3", 21.98, "200 kVp"),
    ("v79_furusawa2000", 0.184, 0.020, "0.0211", "5.3e-4", 39.82, "200 kVp"),
    ("chordoma_kraemer_scholz2000", 0.1, 0.050, "0.0121", "1.5e-3", 8.25, "Co-60"),
    ("prostate_carlson2011", 0.15, 0.048, "0.0181", "1.4e-3", 12.79, "Co-60"),
    ("head_and_neck", 0.25, 0.025, "0.0302", "7.3e-4", 41.33, "Co-60"),
)


def rounds_to(computed: float, printed: str) -> bool:
    """True if ``computed`` rounds to ``printed`` at the precision printed.

    Frese Table 1 prints theta and kappa to two or three significant figures, so
    a relative tolerance is the wrong test -- it either passes trivially or
    fails on the row with the fewest digits (T1's theta = 0.0033, where half an
    ulp is 1.5%). Asserting that our value *rounds to* the printed one is the
    statement the table actually supports, and it tightens rather than loosens
    as the printed precision improves.
    """
    digits = len(printed.replace("-", "").replace(".", "").lstrip("0").split("e")[0])
    return float(f"%.{digits}g" % computed) == float(printed)


def v79() -> RMFModel:
    return RMFModel.v79_furusawa2000()


class TestReferenceRadiations:
    @pytest.mark.golden
    def test_the_two_printed_reference_radiations(self) -> None:
        """Frese sec. 'Low-LET reference radiosensitivity parameters'."""
        assert CO60_REFERENCE.dsb_per_Gy_per_Gbp == pytest.approx(8.27)
        assert CO60_REFERENCE.let_keV_um == pytest.approx(0.36)
        assert X_RAY_200KVP_REFERENCE.dsb_per_Gy_per_Gbp == pytest.approx(8.68)
        assert X_RAY_200KVP_REFERENCE.let_keV_um == pytest.approx(1.53)

    def test_reference_yields_are_not_reproducible_from_the_mcds_closed_form(self) -> None:
        """Both were simulated with a secondary-electron spectrum, not eq. (3).

        They sit just above eq. (3)'s ``x -> 0`` limit of 8.15, which is the
        only cross-check available; the numbers themselves are taken as printed.
        """
        from pyradbio.micro import MCDSYields

        limit = MCDSYields().dsb_per_Gy_per_Gbp(0.0)
        for ref in (CO60_REFERENCE, X_RAY_200KVP_REFERENCE):
            assert limit < ref.dsb_per_Gy_per_Gbp < 1.10 * limit


class TestFrequencyMeanSpecificEnergy:
    @pytest.mark.golden
    def test_eq7_uses_a_5_micron_nucleus_DIAMETER(self) -> None:
        """``z_F = 0.204 LET/d^2`` with d = 5 um -- a DIAMETER, not a radius.

        Frese: 'To represent the nucleus of a cell, we use a diameter of 5 mm
        [sic, um] in Eq. 7'; Carlson sec. 'an assumed cell nucleus 5 m in
        diameter'. The LEM/harness default R_n = 5 um is a *radius*, so the two
        differ by a factor of 2 in d and 4 in z_F.
        """
        model = v79()
        assert model.site_diameter_um == pytest.approx(5.0)
        assert model.z_F_Gy(1.53) == pytest.approx(0.204 * 1.53 / 25.0)

    def test_z_F_is_linear_in_let_and_inverse_square_in_diameter(self) -> None:
        model = v79()
        assert model.z_F_Gy(20.0) == pytest.approx(10.0 * model.z_F_Gy(2.0))
        wider = RMFModel.v79_furusawa2000(site_diameter_um=10.0)
        assert wider.z_F_Gy(5.0) == pytest.approx(model.z_F_Gy(5.0) / 4.0)


class TestTissueParameters:
    """Table 1's theta and kappa are *derived* from the printed alpha_x/beta_x."""

    @pytest.mark.golden
    @pytest.mark.parametrize(("name", "ax", "bx", "theta", "kappa", "ratio", "ref"), FRESE_TABLE_1)
    def test_kappa_matches_table_1(
        self,
        name: str,
        ax: float,
        bx: float,
        theta: str,
        kappa: str,
        ratio: float,
        ref: str,
    ) -> None:
        """Eq. (6) inverted: ``kappa = 2 beta_x / Sigma_x^2``."""
        model = getattr(RMFModel, name)()
        assert rounds_to(model.kappa_per_Gbp2, kappa), (
            f"{model.kappa_per_Gbp2!r} does not round to {kappa}"
        )

    @pytest.mark.golden
    @pytest.mark.parametrize(("name", "ax", "bx", "theta", "kappa", "ratio", "ref"), FRESE_TABLE_1)
    def test_theta_matches_table_1(
        self,
        name: str,
        ax: float,
        bx: float,
        theta: str,
        kappa: str,
        ratio: float,
        ref: str,
    ) -> None:
        """Eq. (5) inverted at the reference radiation:

        ``theta = (alpha_x - kappa z_F,x Sigma_x^2) / Sigma_x``.
        """
        model = getattr(RMFModel, name)()
        assert rounds_to(model.theta_per_Gbp, theta), (
            f"{model.theta_per_Gbp!r} does not round to {theta}"
        )

    @pytest.mark.golden
    @pytest.mark.parametrize(("name", "ax", "bx", "theta", "kappa", "ratio", "ref"), FRESE_TABLE_1)
    def test_theta_over_kappa_matches_table_1(
        self,
        name: str,
        ax: float,
        bx: float,
        theta: str,
        kappa: str,
        ratio: float,
        ref: str,
    ) -> None:
        """Table 1's last column. Prostate is the loosest at 0.8% -- see the docs."""
        model = getattr(RMFModel, name)()
        assert model.theta_per_Gbp / model.kappa_per_Gbp2 == pytest.approx(ratio, rel=0.01)

    @pytest.mark.parametrize(("name", "ax", "bx", "theta", "kappa", "ratio", "ref"), FRESE_TABLE_1)
    def test_each_preset_carries_its_printed_photon_reference(
        self,
        name: str,
        ax: float,
        bx: float,
        theta: str,
        kappa: str,
        ratio: float,
        ref: str,
    ) -> None:
        model = getattr(RMFModel, name)()
        assert model.photon.alpha_x == pytest.approx(ax)
        assert model.photon.beta_x == pytest.approx(bx)
        expected = CO60_REFERENCE if ref == "Co-60" else X_RAY_200KVP_REFERENCE
        assert model.reference_radiation is expected

    def test_all_six_table_1_sets_are_available(self) -> None:
        assert len(RMFModel.published_presets()) == 6


class TestRoundTrip:
    """theta and kappa must return the photon parameters they came from."""

    @pytest.mark.parametrize(("name", "ax", "bx", "theta", "kappa", "ratio", "ref"), FRESE_TABLE_1)
    def test_eqs_5_and_6_recover_the_reference_parameters(
        self,
        name: str,
        ax: float,
        bx: float,
        theta: str,
        kappa: str,
        ratio: float,
        ref: str,
    ) -> None:
        model = getattr(RMFModel, name)()
        sigma_x = model.reference_radiation.dsb_per_Gy_per_Gbp
        z_f_x = model.z_F_Gy(model.reference_radiation.let_keV_um)
        alpha = model.theta_per_Gbp * sigma_x + model.kappa_per_Gbp2 * z_f_x * sigma_x**2
        beta = 0.5 * model.kappa_per_Gbp2 * sigma_x**2
        assert alpha == pytest.approx(ax, rel=1e-9)
        assert beta == pytest.approx(bx, rel=1e-9)


class TestRadiationQualityDependence:
    def test_rbe_min_is_the_dsb_yield_ratio(self) -> None:
        """From eq. (9), ``sqrt(beta/beta_x) = Sigma(q)/Sigma_x`` exactly."""
        model = v79()
        ion = Ion(Z=6, A=12, energy_MeV_u=50.0)
        lq = model.alpha_beta(ion)
        assert math.sqrt(lq.beta / model.photon.beta_x) == pytest.approx(model.rbe_dsb(ion))

    def test_rbe_dsb_is_the_yield_ratio_by_construction(self) -> None:
        model = v79()
        ion = Ion(Z=6, A=12, energy_MeV_u=50.0)
        assert model.rbe_dsb(ion) == pytest.approx(
            model.dsb_per_Gy_per_Gbp(ion) / model.reference_radiation.dsb_per_Gy_per_Gbp
        )

    def test_alpha_and_beta_rise_with_let(self) -> None:
        model = v79()
        fast = model.alpha_beta(Ion(Z=6, A=12, energy_MeV_u=300.0))
        slow = model.alpha_beta(Ion(Z=6, A=12, energy_MeV_u=25.0))  # in-domain
        assert slow.alpha > fast.alpha
        assert slow.beta > fast.beta

    def test_a_very_low_let_proton_reproduces_the_reference_parameters(self) -> None:
        """Frese: 'For protons, parameters in the entrance channel equal the
        reference parameters'. A fast proton sits near the reference quality."""
        model = RMFModel.chordoma_kraemer_scholz2000()
        lq = model.alpha_beta(Ion(Z=1, A=1, energy_MeV_u=200.0))
        assert lq.alpha == pytest.approx(model.photon.alpha_x, rel=0.10)
        assert lq.beta == pytest.approx(model.photon.beta_x, rel=0.10)

    def test_carbon_entrance_is_about_fifty_percent_above_the_reference(self) -> None:
        """Frese: for carbon ions the entrance-channel parameters 'are increased
        by ~50%'. Checked loosely -- their entrance is a spectrum, ours one ion.
        """
        model = RMFModel.chordoma_kraemer_scholz2000()
        lq = model.alpha_beta(Ion(Z=6, A=12, energy_MeV_u=270.0))
        assert 1.2 < lq.alpha / model.photon.alpha_x < 1.9


class TestRBE:
    def test_rbe_is_independent_of_alpha_x_at_fixed_ratio(self) -> None:
        """Eq. (17): 'RBE depends on (a/b)_x but is independent of a_x'.

        A genuine structural claim of the model, and a sharp test: scaling
        alpha_x and beta_x together must leave RBE untouched.
        """
        ion = Ion(Z=6, A=12, energy_MeV_u=100.0)
        base = RMFModel(
            photon=PhotonParameters(alpha_x=0.1, beta_x=0.05), reference_radiation=CO60_REFERENCE
        )
        scaled = RMFModel(
            photon=PhotonParameters(alpha_x=0.4, beta_x=0.2), reference_radiation=CO60_REFERENCE
        )
        for dose in (1.0, 2.0, 6.0):
            assert base.rbe_at_dose(ion, dose) == pytest.approx(
                scaled.rbe_at_dose(ion, dose), rel=1e-9
            )

    def test_rbe_does_depend_on_the_alpha_beta_ratio(self) -> None:
        ion = Ion(Z=6, A=12, energy_MeV_u=100.0)
        low = RMFModel(
            photon=PhotonParameters(alpha_x=0.1, beta_x=0.05), reference_radiation=CO60_REFERENCE
        )
        high = RMFModel(
            photon=PhotonParameters(alpha_x=0.3, beta_x=0.01), reference_radiation=CO60_REFERENCE
        )
        assert low.rbe_at_dose(ion, 2.0) != pytest.approx(high.rbe_at_dose(ion, 2.0))

    def test_rbe_matches_the_isoeffect_definition(self) -> None:
        """Eq. (16) must be the dose ratio at equal effect, by construction."""
        from pyradbio.core.lq import log_kill

        model = v79()
        ion = Ion(Z=6, A=12, energy_MeV_u=100.0)
        dose = 2.0
        lq = model.alpha_beta(ion)
        rbe = model.rbe_at_dose(ion, dose)
        ion_effect = log_kill(dose, lq.alpha, lq.beta)
        photon_effect = log_kill(rbe * dose, model.photon.alpha_x, model.photon.beta_x)
        assert photon_effect == pytest.approx(ion_effect, rel=1e-9)

    def test_rbe_falls_with_dose(self) -> None:
        model = v79()
        ion = Ion(Z=6, A=12, energy_MeV_u=100.0)
        assert model.rbe_at_dose(ion, 1.0) > model.rbe_at_dose(ion, 6.0)

    def test_rbe_rejects_non_positive_dose(self) -> None:
        with pytest.raises(ValueError, match="dose"):
            v79().rbe_at_dose(Ion(Z=6, A=12, energy_MeV_u=100.0), 0.0)


class TestCarlsonEquivalence:
    def test_frese_eq5_is_carlson_eq9_with_all_dsbs_rejoinable(self) -> None:
        """Carlson eq. (9) has ``[1 - f_R(1-theta)]Sigma + kappa z_F (f_R Sigma)^2``.

        Setting ``f_R = 1`` -- which Carlson's own analysis concludes ('these
        observations suggest that all classes of DSB are potentially
        rejoinable') -- collapses it to Frese eq. (5).
        """
        model = v79()
        ion = Ion(Z=6, A=12, energy_MeV_u=50.0)
        sigma = model.dsb_per_Gy_per_Gbp(ion)
        z_f = model.z_F_Gy(model.let_keV_um(ion))
        theta, kappa, f_R = model.theta_per_Gbp, model.kappa_per_Gbp2, 1.0
        carlson = (1.0 - f_R * (1.0 - theta)) * sigma + kappa * z_f * (f_R * sigma) ** 2
        frese = theta * sigma + kappa * z_f * sigma**2
        assert carlson == pytest.approx(frese)
        assert model.alpha_beta(ion).alpha == pytest.approx(frese)


class TestDomain:
    def test_beyond_the_stated_stopping_power_warns(self) -> None:
        """Frese: valid 'up to at least 100 to 150 keV/um'. Above that the paper
        itself says the model over-predicts; we warn rather than clamp."""
        model = v79()
        with pytest.warns(UserWarning, match="150 keV"):
            model.alpha_beta(Ion(Z=10, A=20, energy_MeV_u=5.0))

    def test_outside_the_mcds_fit_domain_warns(self) -> None:
        """Regression: mcds.py promises a warning outside ``1 <= x <= 3200``, but RMF
        never asked for the check. A 0.08 MeV proton has x ~ 3800 at ~82 keV/um, so
        only the MCDS domain is exceeded, not the LET ceiling."""
        with pytest.warns(UserWarning, match="MCDS fit domain") as record:
            v79().alpha_beta(Ion(Z=1, A=1, energy_MeV_u=0.08))
        assert not any("ceiling" in str(w.message) for w in record)

    def test_inside_the_domain_is_silent(self, recwarn: pytest.WarningsRecorder) -> None:
        v79().alpha_beta(Ion(Z=6, A=12, energy_MeV_u=200.0))
        assert len(recwarn) == 0


class TestGuardRails:
    def test_mixed_fields_are_dose_averaged_on_alpha_and_sqrt_beta(self) -> None:
        from pyradbio.core.radiation import RadiationField, SpectrumComponent

        model = v79()
        ion = Ion(Z=6, A=12, energy_MeV_u=100.0)
        field = RadiationField(components=(SpectrumComponent(ion=ion, dose_Gy=1.0),))
        single = model.alpha_beta(ion)
        mixed = model.alpha_beta(field)
        assert mixed.alpha == pytest.approx(single.alpha)
        assert mixed.beta == pytest.approx(single.beta)

    def test_non_positive_site_diameter_is_rejected(self) -> None:
        with pytest.raises(ValueError, match="site_diameter_um"):
            RMFModel(
                photon=PhotonParameters(alpha_x=0.1, beta_x=0.05),
                reference_radiation=CO60_REFERENCE,
                site_diameter_um=0.0,
            )

    def test_reference_requires_positive_yield_and_let(self) -> None:
        from pyradbio.models.rmf import RMFReferenceRadiation

        with pytest.raises(ValueError):
            RMFReferenceRadiation(label="bad", dsb_per_Gy_per_Gbp=0.0, let_keV_um=1.0)
        with pytest.raises(ValueError):
            RMFReferenceRadiation(label="bad", dsb_per_Gy_per_Gbp=8.0, let_keV_um=0.0)


class TestCarlsonGeneralForm:
    """Carlson eq. (9) keeps the rejoinable fraction f_R that Frese drops."""

    def test_f_R_of_one_reproduces_the_shipped_alpha(self) -> None:
        model = v79()
        ion = Ion(Z=6, A=12, energy_MeV_u=50.0)
        assert model.carlson_alpha(ion) == pytest.approx(model.alpha_beta(ion).alpha)

    def test_unrejoinable_dsbs_raise_alpha(self) -> None:
        """(1-f_R) of the DSBs become intrinsically lethal, so alpha must rise.

        theta << 1 for every Table 1 tissue, so the ``[1 - f_R(1-theta)]``
        prefactor grows as f_R falls.
        """
        model = v79()
        ion = Ion(Z=6, A=12, energy_MeV_u=50.0)
        assert model.carlson_alpha(ion, rejoinable_fraction=0.5) > model.carlson_alpha(ion)

    @pytest.mark.parametrize("bad", [0.0, -0.1, 1.5])
    def test_rejoinable_fraction_must_be_a_fraction(self, bad: float) -> None:
        with pytest.raises(ValueError, match="rejoinable_fraction"):
            v79().carlson_alpha(Ion(Z=6, A=12, energy_MeV_u=50.0), rejoinable_fraction=bad)

    def test_non_positive_let_is_rejected_by_eq7(self) -> None:
        with pytest.raises(ValueError, match="let_keV_um"):
            v79().z_F_Gy(0.0)


class TestAlphaTerms:
    """``alpha_terms`` splits eq. (5); the split must be the alpha it sums to."""

    @pytest.mark.parametrize(("Z", "A", "energy"), [(1, 1, 10.0), (2, 4, 50.0), (6, 12, 50.0)])
    def test_terms_are_eq5_and_sum_to_alpha_bit_for_bit(
        self, Z: int, A: int, energy: float
    ) -> None:
        model = v79()
        ion = Ion(Z=Z, A=A, energy_MeV_u=energy)
        terms = model.alpha_terms(ion)
        sigma = model.dsb_per_Gy_per_Gbp(ion)
        let = model.let_keV_um(ion)
        assert terms.let_keV_um == let
        assert terms.dsb_per_Gy_per_Gbp == sigma
        assert terms.one_track == model.theta_per_Gbp * sigma
        assert terms.intratrack == model.kappa_per_Gbp2 * model.z_F_Gy(let) * sigma * sigma
        assert terms.alpha == model.alpha_beta(ion).alpha

    def test_terms_follow_the_model_stopping_power(self) -> None:
        sp = SimpleBetheStoppingPower()
        model = RMFModel.v79_furusawa2000(stopping_power=sp)
        ion = Ion(Z=6, A=12, energy_MeV_u=20.0)
        assert model.alpha_terms(ion).let_keV_um == sp.let_keV_um(ion)

    def test_warns_above_the_ceiling_like_alpha_beta(self) -> None:
        with pytest.warns(UserWarning, match="ceiling"):
            v79().alpha_terms(Ion(Z=6, A=12, energy_MeV_u=1.5))

    @pytest.mark.parametrize("method", ["alpha_beta", "alpha_terms"])
    def test_ceiling_warning_points_at_the_caller(self, method: str) -> None:
        """The shared warning helper keeps ``stacklevel`` on the caller's line."""
        ion = Ion(Z=6, A=12, energy_MeV_u=1.5)
        with pytest.warns(UserWarning, match="ceiling") as record:
            getattr(v79(), method)(ion)
        assert record[0].filename == __file__

    def test_intratrack_overtakes_one_track_only_above_the_ceiling(self) -> None:
        """Carbon in V79: across the stated domain the pairing term is a correction.

        The two terms are equal near 214 keV/um on the analytic-Bethe LET
        (``examples/rmf_dsb_and_binary_misrepair.py`` panel 2), above the
        150 keV/um ceiling Frese states for the form.
        """
        model = RMFModel.v79_furusawa2000(stopping_power=SimpleBetheStoppingPower())
        with warnings.catch_warnings():
            warnings.simplefilter("ignore", UserWarning)
            for energy in np.logspace(np.log10(1.5), np.log10(500.0), 60):
                terms = model.alpha_terms(Ion(Z=6, A=12, energy_MeV_u=float(energy)))
                if terms.let_keV_um < 200.0:
                    assert terms.one_track > terms.intratrack, terms
                if terms.let_keV_um > 230.0:
                    assert terms.intratrack > terms.one_track, terms
