"""NanOx (Cunha et al. 2017; Monini et al. 2020; Alcocer-Avila et al. 2022, 2023).

Goldens are the papers' own printed numbers and their own algebra -- the ELLF
parameter tables, which four papers agree on, and the ``beta_G = beta_r/eta^2``
calibration chain, which reproduces every printed ``beta_G`` in the corpus.
Never a digitized figure point: NanOx's nanodosimetric input is figure-only and
is deliberately out of scope (see ``docs/references/NanOx.md``).
"""

from __future__ import annotations

import math
from itertools import pairwise

import numpy as np
import pytest

from pyradbio.core.photon import PhotonParameters
from pyradbio.core.radiation import Ion
from pyradbio.models.nanox import (
    DEFAULT_LOCAL_TARGET,
    ETA_RESTRICTED_FRACTION,
    GY_PER_KEV_UM3,
    EffectiveLocalLethalFunction,
    LocalTarget,
    LocalTargetSpectrum,
    NanOxCellLine,
    NanOxModel,
    NanOxNanodosimetry,
    NanOxNanodosimetryRequired,
)

# ---------------------------------------------------------------------------
# The published parameter sets.
#
# Monini et al. 2020 Table 1 (ELF columns), Monini et al. 2018 Table 1, Monini
# et al. 2019 Table 1 and Alcocer-Avila et al. 2022 Table 1 all print the same
# values. ``beta_r`` is from Monini et al. 2018 Table 5.
#
# (preset, z0 [Gy], sigma [Gy], h, d_s [um], beta_G [Gy^-2], beta_r [Gy^-2])
# ---------------------------------------------------------------------------
PUBLISHED_CELL_LINES = (
    ("v79", 22789.0, 8117.0, 225841.0, 9.8, "0.0405", 0.0259),
    ("cho_k1", 14507.0, 2781.0, 104810.0, 11.8, "0.0625", 0.0400),
    ("hsg", 15654.0, 549.0, 179439.0, 14.0, "0.0961", 0.0615),
)


def rounds_to(computed: float, printed: str) -> bool:
    """True if ``computed`` rounds to ``printed`` at the precision printed.

    The same helper as ``tests/models/test_rmf.py``. The NanOx papers print
    ``beta_G`` to three significant figures, so a relative tolerance either
    passes trivially or has to be loosened to ~1%, which asserts nothing.
    """
    digits = len(printed.replace("-", "").replace(".", "").lstrip("0").split("e")[0])
    return float(f"%.{digits}g" % computed) == float(printed)


def v79() -> NanOxModel:
    return NanOxModel.v79()


class TestEffectiveLocalLethalFunction:
    """Monini et al. 2020 eq. (28), doi:10.1667/RR15463.1."""

    @pytest.mark.golden
    @pytest.mark.parametrize(
        ("preset", "z0", "sigma", "h", "d_s", "beta_G", "beta_r"), PUBLISHED_CELL_LINES
    )
    def test_published_ellf_parameters(
        self,
        preset: str,
        z0: float,
        sigma: float,
        h: float,
        d_s: float,
        beta_G: str,
        beta_r: float,
    ) -> None:
        """Monini 2020 Table 1, ELF columns; identical in three later papers."""
        model = getattr(NanOxModel, preset)()
        assert model.cell_line.ellf.z0_Gy == pytest.approx(z0)
        assert model.cell_line.ellf.sigma_Gy == pytest.approx(sigma)
        assert model.cell_line.ellf.h == pytest.approx(h)

    def test_is_an_error_function_not_a_probability(self) -> None:
        """Eq. (28) has no normalisation and no truncation.

        ``F`` is ``-N ln(1 - f)`` (2022 eq. 7), an *effective number of events*
        summed over N targets, so it saturates at ``h`` (order 1e5), not at 1.
        """
        ellf = EffectiveLocalLethalFunction(h=225841.0, z0_Gy=22789.0, sigma_Gy=8117.0)
        assert ellf(0.0) < 1.0e-3 * ellf.h
        assert ellf(ellf.z0_Gy) == pytest.approx(ellf.h / 2.0)
        assert ellf(1.0e6) == pytest.approx(ellf.h, rel=1e-12)

    def test_threshold_and_saturation(self) -> None:
        """The two features Monini 2020 reports for all three cell lines."""
        ellf = EffectiveLocalLethalFunction(h=1.0, z0_Gy=100.0, sigma_Gy=10.0)
        assert ellf(50.0) < 1e-6
        assert ellf(150.0) == pytest.approx(1.0, abs=1e-6)

    def test_is_monotonically_increasing(self) -> None:
        """Monini 2020: F is built "assuming it increases monotonically"."""
        ellf = EffectiveLocalLethalFunction(h=2.0, z0_Gy=500.0, sigma_Gy=100.0)
        zs = [0.0, 100.0, 300.0, 500.0, 700.0, 1000.0, 5000.0]
        values = [ellf(z) for z in zs]
        assert all(b >= a for a, b in pairwise(values))

    def test_rejects_nonpositive_sigma(self) -> None:
        with pytest.raises(ValueError, match="sigma_Gy"):
            EffectiveLocalLethalFunction(h=1.0, z0_Gy=1.0, sigma_Gy=0.0)

    def test_rejects_negative_height(self) -> None:
        with pytest.raises(ValueError, match="h"):
            EffectiveLocalLethalFunction(h=-1.0, z0_Gy=1.0, sigma_Gy=1.0)


class TestGlobalEventCalibration:
    """``beta_G = beta_r / eta^2`` -- Monini et al. 2017 (AppPolB) eq. (3.2)."""

    @pytest.mark.golden
    @pytest.mark.parametrize(
        ("preset", "z0", "sigma", "h", "d_s", "beta_G", "beta_r"), PUBLISHED_CELL_LINES
    )
    def test_beta_G_reproduces_every_printed_value(
        self,
        preset: str,
        z0: float,
        sigma: float,
        h: float,
        d_s: float,
        beta_G: str,
        beta_r: float,
    ) -> None:
        """The corpus's one closed-form coefficient chain, 3/3 on the later fits.

        Monini et al. 2018 Table 5 prints ``beta_r``; Table 1 of the same paper
        (and of Monini 2019 / Alcocer 2022) prints ``beta_G``. ``eta = 0.8``
        exactly -- not the simulated 0.798-0.802 of Alcocer 2023 Table 4.
        """
        computed = NanOxModel.beta_G_from_photon(beta_r)
        assert rounds_to(computed, beta_G), f"{computed!r} does not round to {beta_G}"

    @pytest.mark.golden
    def test_beta_G_reproduces_the_2017_v79_fit_too(self) -> None:
        """Cunha et al. 2017 sec. 3.2.4: beta_r = 0.0298 (Tilly 1999) -> 0.0466.

        A *fourth* confirmation, on the superseded 2017 parameter set with its
        own photon reference. doi:10.1088/1361-6560/aa54c9.
        """
        assert rounds_to(NanOxModel.beta_G_from_photon(0.0298), "0.0466")

    def test_alpha_local_is_the_photon_alpha_over_eta(self) -> None:
        """Alcocer 2023 eqs. (33) + (35) with ``C_L = 1``, so ``alpha_G = 0``."""
        model = NanOxModel.v79(photon=PhotonParameters(alpha_x=0.112, beta_x=0.0259))
        assert model.alpha_reference_local() == pytest.approx(0.112 / 0.8)

    def test_alpha_G_is_zero_in_this_version(self) -> None:
        """``C_L = 1`` in every published NanOx calculation (2023 eq. 34)."""
        assert v79().cell_line.alpha_G_per_Gy == 0.0

    def test_alpha_local_requires_a_photon_reference(self) -> None:
        with pytest.raises(ValueError, match="photon"):
            v79().alpha_reference_local()


class TestGeometry:
    """Alcocer-Avila et al. 2022 eqs. (14)-(17), doi:10.3389/fphy.2022.1011063."""

    @pytest.mark.golden
    def test_cunha_2017_cross_section_from_the_printed_diameter(self) -> None:
        """Cunha 2017 sec. 3.2.1: d_s = 9.78 um gives sigma_s = 75 um^2.

        The dimensional check that settles ``d_s`` as a DIAMETER, not a radius.
        (The paper prints the unit as "um^-2", a typo.)
        """
        model = NanOxModel.v79(sensitive_volume_diameter_um=9.78)
        assert model.geometric_cross_section_um2() == pytest.approx(75.0, abs=0.2)

    def test_local_target_is_a_diameter_and_a_length(self) -> None:
        """2023 Simplification 4: cylinders, d_t = 20 nm, L_t = 10 nm.

        The 2022 figure captions say "10 nm radius and length" -- the same
        object, stated the other way round.
        """
        assert DEFAULT_LOCAL_TARGET.diameter_nm == pytest.approx(20.0)
        assert DEFAULT_LOCAL_TARGET.length_nm == pytest.approx(10.0)
        assert DEFAULT_LOCAL_TARGET.radius_nm() == pytest.approx(10.0)

    def test_core_and_penumbra_fractions_sum_to_one(self) -> None:
        """Eq. (16): ``r_c + r_p = 1`` by construction."""
        model = v79()
        assert model.core_fraction() + model.penumbra_fraction() == pytest.approx(1.0)

    def test_core_volume_is_a_square_parallelepiped(self) -> None:
        """Eq. (14): ``V_c = L_s e_c^2`` with ``e_c = 200 nm``."""
        model = v79()
        assert model.core_volume_um3() == pytest.approx(1.0 * 0.2**2)

    def test_core_is_a_small_fraction_of_the_nucleus(self) -> None:
        """2022: "the edge of the core volume is very small with respect to the
        radius of the sensitive volume"."""
        assert v79().core_fraction() < 1e-3


class TestImpactStatistics:
    """Alcocer-Avila et al. 2022 eqs. (34)-(35)."""

    def test_mean_impacts_scale_with_dose_over_let(self) -> None:
        model = v79()
        inside, outside = model.mean_impacts(dose_Gy=2.0, let_keV_um=10.0, surface_um2=200.0)
        sigma_s = model.geometric_cross_section_um2()
        assert inside == pytest.approx(2.0 * sigma_s / (GY_PER_KEV_UM3 * 10.0))
        assert outside == pytest.approx(2.0 * (200.0 - sigma_s) / (GY_PER_KEV_UM3 * 10.0))

    def test_surface_of_influence_must_contain_the_nucleus(self) -> None:
        model = v79()
        with pytest.raises(ValueError, match="surface_um2"):
            model.mean_impacts(dose_Gy=1.0, let_keV_um=10.0, surface_um2=1.0)

    def test_unit_conversion_factor_is_the_printed_one(self) -> None:
        """2022 eq. (35): ``a = 0.1602 Gy keV^-1 um^3``."""
        assert GY_PER_KEV_UM3 == pytest.approx(0.1602)


class TestTrackSegmentClosedForms:
    """Alcocer-Avila et al. 2023 sec. 3.5, eqs. (45)-(57)."""

    def test_alpha_from_cross_section_is_the_kanai_relation(self) -> None:
        """Eq. (53): ``alpha = zeta / (a LET)``."""
        model = v79()
        assert model.alpha_from_cross_section(zeta_um2=30.0, let_keV_um=100.0) == pytest.approx(
            30.0 / (GY_PER_KEV_UM3 * 100.0)
        )

    def test_cross_section_inversion_round_trips(self) -> None:
        """Eq. (45) inverted -- the direction that turns a published alpha into
        NanOx's own inactivation cross section."""
        model = v79()
        zeta = model.inactivation_cross_section_um2(alpha_per_Gy=0.5, let_keV_um=60.0)
        assert model.alpha_from_cross_section(zeta, let_keV_um=60.0) == pytest.approx(0.5)

    def test_cross_section_is_independent_of_the_surface_of_influence(self) -> None:
        """Eqs. (54)-(57): the paper's own proof that ``zeta_2 = zeta_1``.

        A real invariant, not a tautology: ``S_1`` is defined relative to the
        surface, so a larger surface must dilute it exactly enough to cancel.
        """
        model = v79()
        zeta = 30.0
        for surface in (100.0, 400.0, 2500.0):
            s1 = model.single_impact_survival(zeta_um2=zeta, surface_um2=surface)
            assert model.inactivation_cross_section_from_single_impact(
                s1, surface_um2=surface
            ) == pytest.approx(zeta)

    def test_local_survival_is_a_pure_exponential(self) -> None:
        """Eq. (52): with only local lethal events, ``S(D) = exp(-alpha D)``.

        This is the structural claim that NanOx's shoulder comes *entirely*
        from global events -- the reason non-local events had to be introduced.
        """
        model = v79()
        alpha = model.alpha_from_cross_section(zeta_um2=25.0, let_keV_um=80.0)
        for dose in (0.5, 2.0, 6.0):
            assert model.survival_local(
                dose_Gy=dose, zeta_um2=25.0, let_keV_um=80.0
            ) == pytest.approx(math.exp(-alpha * dose))

    def test_survival_local_rejects_negative_dose(self) -> None:
        with pytest.raises(ValueError, match="dose_Gy"):
            v79().survival_local(dose_Gy=-1.0, zeta_um2=10.0, let_keV_um=10.0)


class TestGlobalEvents:
    """Alcocer-Avila et al. 2022 eqs. (9)-(11)."""

    def test_chemical_specific_energy_scales_the_restricted_one(self) -> None:
        """Eq. (9): ``Ztilde = RCE * Z``."""
        model = v79()
        assert model.chemical_specific_energy_Gy(z_Gy=3.0, rce=0.6) == pytest.approx(1.8)

    def test_global_survival_is_lq_in_the_chemical_specific_energy(self) -> None:
        """Eq. (11) with ``alpha_G = 0``."""
        model = v79()
        beta_G = model.cell_line.beta_G_per_Gy2
        assert model.survival_global(z_tilde_Gy=2.0) == pytest.approx(math.exp(-beta_G * 4.0))

    def test_reference_radiation_has_rce_one_by_definition(self) -> None:
        """Eq. (10): ``RCE = G/G_r``, so the reference radiation gives 1."""
        model = v79()
        assert model.chemical_specific_energy_Gy(z_Gy=5.0, rce=1.0) == pytest.approx(5.0)


class TestEffectiveNumberOfLocalLethalEvents:
    """Alcocer-Avila et al. 2023 eqs. (42)-(43) and 2022 eq. (26)."""

    def test_enllf_integrates_the_spectrum_against_the_ellf(self) -> None:
        """Eq. (42): ``n* = int (dP/dz) F(z) dz``."""
        model = v79()
        # A delta-like spectrum at a single z: n* must reduce to F(z).
        z = 30000.0
        spectrum = LocalTargetSpectrum.delta(z_Gy=z, mean_hit_z_Gy=z)
        assert model.effective_local_lethal_events(spectrum) == pytest.approx(
            model.cell_line.ellf(z), rel=1e-9
        )

    def test_alpha_core_is_the_ellf_mean_over_the_mean_hit_energy(self) -> None:
        """Eq. (26): ``alpha_c = <F(z)> / <z>_hit``."""
        model = v79()
        z = 30000.0
        spectrum = LocalTargetSpectrum.delta(z_Gy=z, mean_hit_z_Gy=z)
        assert model.alpha_core(spectrum) == pytest.approx(model.cell_line.ellf(z) / z, rel=1e-9)

    def test_a_spectrum_below_threshold_is_negligible_against_saturation(self) -> None:
        """The threshold is what produces the overkill turnover in alpha(LET).

        "Negligible" here is *relative to h*, not absolute. An error function
        has no compact support, so ``F(0) = (h/2)(1 - erf(z0/sigma))`` is small
        but non-zero -- 3.6e-5 of ``h`` for V79. Asserting ``F ~ 0`` absolutely
        would be wrong, since ``h`` is of order 1e5.
        """
        model = v79()
        below = LocalTargetSpectrum.delta(z_Gy=100.0, mean_hit_z_Gy=100.0)
        above = LocalTargetSpectrum.delta(z_Gy=40000.0, mean_hit_z_Gy=40000.0)
        h = model.cell_line.ellf.h
        assert model.effective_local_lethal_events(below) / h < 1e-4
        assert model.effective_local_lethal_events(above) / h > 0.99

    def test_the_ellf_tail_at_zero_is_small_but_not_zero(self) -> None:
        """Pinned because it looks like a bug and is not.

        ``F(0)`` is non-zero for every published parameter set. It never
        matters, because eq. (26) integrates the distribution over targets that
        were *hit*, and the assembly then scales the core term by
        ``r_c ~ 5e-4``; but a reader meeting ``F(0) = 8.1`` for V79 should find
        it recorded rather than filed as a defect.
        """
        ellf = v79().cell_line.ellf
        assert 0.0 < ellf(0.0) / ellf.h < 1e-4

    def test_spectrum_must_be_normalised(self) -> None:
        with pytest.raises(ValueError, match="normalis"):
            LocalTargetSpectrum(
                z_Gy=(0.0, 1.0, 2.0), probability=(0.0, 0.5, 0.0), mean_hit_z_Gy=1.0
            )

    def test_from_density_normalises_and_reproduces_a_narrow_peak(self) -> None:
        """The density constructor is the shape the papers write ``dP/dz`` in."""
        model = v79()
        z = [float(v) for v in range(29000, 31001, 10)]
        centre, width = 30000.0, 200.0
        dens = [math.exp(-0.5 * ((v - centre) / width) ** 2) for v in z]
        spectrum = LocalTargetSpectrum.from_density(z, dens, mean_hit_z_Gy=centre)
        assert sum(spectrum.probability) == pytest.approx(1.0)
        # Narrow compared with the ELLF's sigma = 8117 Gy, so it must land close
        # to the delta-limit answer.
        assert model.effective_local_lethal_events(spectrum) == pytest.approx(
            model.cell_line.ellf(centre), rel=1e-3
        )


class TestNanodosimetryBoundary:
    """The documented scope boundary -- see docs/references/NanOx.md."""

    def test_alpha_beta_raises_without_a_nanodosimetry_backend(self) -> None:
        """The nanometric ``dP/dz`` comes from LQD, which is not public.

        Rather than substitute a different track-structure model, the ion route
        refuses and says why. Same treatment as Katz's track-width regime.
        """
        model = NanOxModel.v79(photon=PhotonParameters(alpha_x=0.112, beta_x=0.0259))
        with pytest.raises(NanOxNanodosimetryRequired, match="LQD"):
            model.alpha_beta(Ion(Z=6, A=12, energy_MeV_u=100.0))

    def test_the_error_names_the_two_missing_quantities(self) -> None:
        model = NanOxModel.v79(photon=PhotonParameters(alpha_x=0.112, beta_x=0.0259))
        with pytest.raises(NanOxNanodosimetryRequired) as excinfo:
            model.alpha_beta(Ion(Z=6, A=12, energy_MeV_u=100.0))
        message = str(excinfo.value)
        assert "dP/dz" in message
        assert "RCE" in message

    def test_the_error_does_not_promise_that_a_backend_is_enough(self) -> None:
        """A backend removes only the *data* obstacle.

        The configuration average (Alcocer-Avila et al. 2022 eqs. 36-43) is not
        implemented either, so :meth:`alpha_beta` still raises with a backend
        supplied (see ``TestBackendSuppliedRoute``). The first message must say
        so, rather than telling the caller a backend "enables this route".
        """
        model = NanOxModel.v79(photon=PhotonParameters(alpha_x=0.112, beta_x=0.0259))
        with pytest.raises(NanOxNanodosimetryRequired) as excinfo:
            model.alpha_beta(Ion(Z=6, A=12, energy_MeV_u=100.0))
        message = str(excinfo.value)
        assert "enable this route" not in message
        assert "eqs. 36-43" in message
        for method in ("alpha_core()", "local_lethal_events()", "survival_global()"):
            assert method in message

    def test_the_boundary_is_advertised_on_the_class(self) -> None:
        assert NanOxModel.capabilities.ion_route is False

    def test_nanox_is_not_a_biophysical_model(self) -> None:
        """No ion route, so it must not be substitutable where alpha_beta is promised."""
        from pyradbio.models.base import BiophysicalModel

        assert not issubclass(NanOxModel, BiophysicalModel)


class TestPublishedConstants:
    def test_eta_is_exactly_four_fifths(self) -> None:
        """Alcocer 2023 sec. 3.1 reports 0.798-0.802 from simulation, but the
        published ``beta_G`` values are all reproduced by 0.8 exactly."""
        assert ETA_RESTRICTED_FRACTION == 0.8

    @pytest.mark.golden
    @pytest.mark.parametrize(
        ("preset", "z0", "sigma", "h", "d_s", "beta_G", "beta_r"), PUBLISHED_CELL_LINES
    )
    def test_published_geometry(
        self,
        preset: str,
        z0: float,
        sigma: float,
        h: float,
        d_s: float,
        beta_G: str,
        beta_r: float,
    ) -> None:
        """Alcocer 2022 Table 1 (diameters) == Monini 2018/2019 (radii) x 2."""
        model = getattr(NanOxModel, preset)()
        assert model.sensitive_volume_diameter_um == pytest.approx(d_s)
        assert model.cell_line.sensitive_volume_length_um == pytest.approx(1.0)

    def test_published_presets_are_registered(self) -> None:
        presets = NanOxModel.published_presets()
        assert set(presets) == {"v79", "cho_k1", "hsg"}
        for build in presets.values():
            assert isinstance(build(), NanOxModel)

    def test_the_2017_v79_fit_is_not_shipped_as_a_preset(self) -> None:
        """Cunha 2017's V79 fit (z0=20200, sigma=6000, h=180000) is superseded
        and used a different photon reference *and* no core/penumbra split.
        Shipping it as a peer preset would invite mixing it with the rest."""
        assert "v79_cunha2017" not in NanOxModel.published_presets()
        assert NanOxModel.v79().cell_line.ellf.z0_Gy != 20200.0


class TestCellLineValidation:
    def test_rejects_nonpositive_diameter(self) -> None:
        with pytest.raises(ValueError, match="sensitive_volume_diameter_um"):
            NanOxCellLine(
                label="bad",
                sensitive_volume_diameter_um=0.0,
                sensitive_volume_length_um=1.0,
                ellf=EffectiveLocalLethalFunction(h=1.0, z0_Gy=1.0, sigma_Gy=1.0),
                beta_G_per_Gy2=0.04,
            )

    def test_rejects_negative_beta_G(self) -> None:
        with pytest.raises(ValueError, match="beta_G_per_Gy2"):
            NanOxCellLine(
                label="bad",
                sensitive_volume_diameter_um=10.0,
                sensitive_volume_length_um=1.0,
                ellf=EffectiveLocalLethalFunction(h=1.0, z0_Gy=1.0, sigma_Gy=1.0),
                beta_G_per_Gy2=-0.01,
            )


class TestCoreAndPenumbraAssembly:
    """Alcocer-Avila et al. 2022 eq. (27), the two-region sum."""

    def _model(self) -> NanOxModel:
        return NanOxModel.v79(photon=PhotonParameters(alpha_x=0.112, beta_x=0.0259))

    def test_assembly_sums_the_two_weighted_contributions(self) -> None:
        model = self._model()
        n_star = model.local_lethal_events(
            alpha_core_per_Gy=0.004,
            core_specific_energy_Gy=5.0e4,
            penumbra_specific_energy_Gy=1.5,
        )
        expected = 0.004 * model.core_fraction() * 5.0e4 + (0.112 / 0.8) * (
            model.penumbra_fraction() * 1.5
        )
        assert n_star == pytest.approx(expected)

    def test_the_penumbra_is_treated_as_reference_radiation(self) -> None:
        """Approximation 2: the penumbra coefficient is alpha_r,L for every ion.

        Its spectra are made by fast delta electrons, like a photon field's.
        """
        model = self._model()
        only_penumbra = model.local_lethal_events(
            alpha_core_per_Gy=0.0,
            core_specific_energy_Gy=0.0,
            penumbra_specific_energy_Gy=2.0,
        )
        assert only_penumbra == pytest.approx(
            model.alpha_reference_local() * model.penumbra_fraction() * 2.0
        )

    def test_survival_to_local_events_is_exp_minus_n_star(self) -> None:
        """Eq. (40) -- exact, with no Poisson assumption anywhere."""
        model = self._model()
        assert model.survival_local_from_events(2.5) == pytest.approx(math.exp(-2.5))
        assert model.survival_local_from_events(0.0) == 1.0

    def test_a_track_missing_the_nucleus_contributes_only_penumbra(self) -> None:
        """2022 Fig. 1B: if the ion misses the nucleus the core term vanishes."""
        model = self._model()
        missed = model.local_lethal_events(
            alpha_core_per_Gy=0.004,
            core_specific_energy_Gy=0.0,
            penumbra_specific_energy_Gy=0.3,
        )
        assert missed > 0.0
        assert missed < model.local_lethal_events(
            alpha_core_per_Gy=0.004,
            core_specific_energy_Gy=5.0e4,
            penumbra_specific_energy_Gy=0.3,
        )


class TestVectorisedEllf:
    def test_values_matches_the_scalar_call(self) -> None:
        ellf = v79().cell_line.ellf
        grid = np.array([0.0, 1.0e4, 22789.0, 5.0e4])
        assert ellf.values(grid) == pytest.approx([ellf(float(z)) for z in grid])


class TestLocalTargetGeometry:
    def test_volume_of_the_published_target(self) -> None:
        """A 20 nm diameter, 10 nm long cylinder."""
        assert DEFAULT_LOCAL_TARGET.volume_nm3() == pytest.approx(math.pi * 100.0 * 10.0)

    def test_rejects_nonpositive_dimensions(self) -> None:
        with pytest.raises(ValueError, match="diameter_nm"):
            LocalTarget(diameter_nm=0.0, length_nm=10.0)
        with pytest.raises(ValueError, match="length_nm"):
            LocalTarget(diameter_nm=20.0, length_nm=-1.0)


class TestSpectrumValidation:
    def test_rejects_mismatched_lengths(self) -> None:
        with pytest.raises(ValueError, match="equal length"):
            LocalTargetSpectrum(z_Gy=(1.0, 2.0), probability=(1.0,), mean_hit_z_Gy=1.0)

    def test_rejects_empty(self) -> None:
        with pytest.raises(ValueError, match="at least one bin"):
            LocalTargetSpectrum(z_Gy=(), probability=(), mean_hit_z_Gy=1.0)

    def test_rejects_negative_z(self) -> None:
        with pytest.raises(ValueError, match="z_Gy must be >= 0"):
            LocalTargetSpectrum(z_Gy=(-1.0,), probability=(1.0,), mean_hit_z_Gy=1.0)

    def test_rejects_negative_probability(self) -> None:
        with pytest.raises(ValueError, match="probability must be >= 0"):
            LocalTargetSpectrum(z_Gy=(1.0, 2.0), probability=(-0.5, 1.5), mean_hit_z_Gy=1.0)

    def test_rejects_nonpositive_mean_hit(self) -> None:
        with pytest.raises(ValueError, match="mean_hit_z_Gy"):
            LocalTargetSpectrum(z_Gy=(1.0,), probability=(1.0,), mean_hit_z_Gy=0.0)

    def test_from_density_rejects_a_single_node(self) -> None:
        with pytest.raises(ValueError, match="at least two nodes"):
            LocalTargetSpectrum.from_density([1.0], [1.0], mean_hit_z_Gy=1.0)

    def test_from_density_rejects_a_non_increasing_grid(self) -> None:
        with pytest.raises(ValueError, match="strictly increasing"):
            LocalTargetSpectrum.from_density([2.0, 1.0], [1.0, 1.0], mean_hit_z_Gy=1.0)

    def test_from_density_rejects_an_all_zero_density(self) -> None:
        with pytest.raises(ValueError, match="integrates to zero"):
            LocalTargetSpectrum.from_density([1.0, 2.0], [0.0, 0.0], mean_hit_z_Gy=1.0)

    def test_ellf_rejects_negative_threshold(self) -> None:
        with pytest.raises(ValueError, match="z0_Gy"):
            EffectiveLocalLethalFunction(h=1.0, z0_Gy=-1.0, sigma_Gy=1.0)


class TestModelValidation:
    def test_rejects_nonpositive_core_edge(self) -> None:
        with pytest.raises(ValueError, match="core_edge_um"):
            NanOxModel.v79(core_edge_um=0.0)

    def test_rejects_eta_outside_the_unit_interval(self) -> None:
        with pytest.raises(ValueError, match="eta"):
            NanOxModel.v79(eta=0.0)
        with pytest.raises(ValueError, match="eta"):
            NanOxModel.v79(eta=1.5)

    def test_rejects_a_core_wider_than_the_nucleus(self) -> None:
        """The core/penumbra split is meaningless if the core fills the volume."""
        with pytest.raises(ValueError, match="smaller than the sensitive-volume"):
            NanOxModel.v79(core_edge_um=50.0)

    def test_beta_G_rejects_a_negative_photon_beta(self) -> None:
        with pytest.raises(ValueError, match="beta_x_per_Gy2"):
            NanOxModel.beta_G_from_photon(-0.01)

    def test_cell_line_rejects_nonpositive_length(self) -> None:
        with pytest.raises(ValueError, match="sensitive_volume_length_um"):
            NanOxCellLine(
                label="bad",
                sensitive_volume_diameter_um=10.0,
                sensitive_volume_length_um=0.0,
                ellf=EffectiveLocalLethalFunction(h=1.0, z0_Gy=1.0, sigma_Gy=1.0),
                beta_G_per_Gy2=0.04,
            )

    def test_cell_line_rejects_negative_alpha_G(self) -> None:
        with pytest.raises(ValueError, match="alpha_G_per_Gy"):
            NanOxCellLine(
                label="bad",
                sensitive_volume_diameter_um=10.0,
                sensitive_volume_length_um=1.0,
                ellf=EffectiveLocalLethalFunction(h=1.0, z0_Gy=1.0, sigma_Gy=1.0),
                beta_G_per_Gy2=0.04,
                alpha_G_per_Gy=-0.1,
            )

    def test_rce_must_be_non_negative(self) -> None:
        with pytest.raises(ValueError, match="rce"):
            v79().chemical_specific_energy_Gy(z_Gy=1.0, rce=-0.1)

    def test_mean_impacts_rejects_bad_dose_and_let(self) -> None:
        model = v79()
        with pytest.raises(ValueError, match="dose_Gy"):
            model.mean_impacts(dose_Gy=-1.0, let_keV_um=10.0, surface_um2=200.0)
        with pytest.raises(ValueError, match="let_keV_um"):
            model.mean_impacts(dose_Gy=1.0, let_keV_um=0.0, surface_um2=200.0)

    def test_alpha_from_cross_section_rejects_bad_inputs(self) -> None:
        model = v79()
        with pytest.raises(ValueError, match="let_keV_um"):
            model.alpha_from_cross_section(zeta_um2=1.0, let_keV_um=0.0)
        with pytest.raises(ValueError, match="zeta_um2"):
            model.alpha_from_cross_section(zeta_um2=-1.0, let_keV_um=10.0)

    def test_cross_section_inversion_rejects_bad_inputs(self) -> None:
        model = v79()
        with pytest.raises(ValueError, match="alpha_per_Gy"):
            model.inactivation_cross_section_um2(alpha_per_Gy=-1.0, let_keV_um=10.0)
        with pytest.raises(ValueError, match="let_keV_um"):
            model.inactivation_cross_section_um2(alpha_per_Gy=1.0, let_keV_um=0.0)

    def test_single_impact_survival_rejects_bad_inputs(self) -> None:
        model = v79()
        with pytest.raises(ValueError, match="surface_um2"):
            model.single_impact_survival(zeta_um2=1.0, surface_um2=0.0)
        with pytest.raises(ValueError, match="zeta_um2"):
            model.single_impact_survival(zeta_um2=500.0, surface_um2=100.0)

    def test_cross_section_from_single_impact_rejects_bad_inputs(self) -> None:
        model = v79()
        with pytest.raises(ValueError, match="surface_um2"):
            model.inactivation_cross_section_from_single_impact(0.5, surface_um2=0.0)
        with pytest.raises(ValueError, match="single_impact_survival"):
            model.inactivation_cross_section_from_single_impact(1.5, surface_um2=100.0)


class _StubNanodosimetry:
    """A minimal backend, standing in for LQD in tests only."""

    def core_target_spectrum(self, ion: Ion) -> LocalTargetSpectrum:
        return LocalTargetSpectrum.delta(z_Gy=3.0e4, mean_hit_z_Gy=3.0e4)

    def relative_chemical_effectiveness(self, ion: Ion) -> float:
        return 0.7


class TestBackendSuppliedRoute:
    """A backend removes the *data* obstacle; the configuration average is a
    separate piece of work, and the error says so rather than pretending."""

    def test_a_supplied_backend_gives_a_different_message(self) -> None:
        model = NanOxModel.v79(
            photon=PhotonParameters(alpha_x=0.112, beta_x=0.0259),
            nanodosimetry=_StubNanodosimetry(),
        )
        with pytest.raises(NanOxNanodosimetryRequired, match=r"eqs\. 36-43"):
            model.alpha_beta(Ion(Z=6, A=12, energy_MeV_u=100.0))

    def test_a_backend_satisfies_the_protocol(self) -> None:
        assert isinstance(_StubNanodosimetry(), NanOxNanodosimetry)

    def test_the_pieces_a_caller_is_pointed_at_do_compose(self) -> None:
        """The error names alpha_core / local_lethal_events / survival_global;
        this checks that route actually works end to end."""
        model = NanOxModel.v79(
            photon=PhotonParameters(alpha_x=0.112, beta_x=0.0259),
            nanodosimetry=_StubNanodosimetry(),
        )
        ion = Ion(Z=6, A=12, energy_MeV_u=100.0)
        backend = _StubNanodosimetry()
        alpha_c = model.alpha_core(backend.core_target_spectrum(ion))
        n_star = model.local_lethal_events(
            alpha_core_per_Gy=alpha_c,
            core_specific_energy_Gy=4.0e4,
            penumbra_specific_energy_Gy=1.0,
        )
        z_tilde = model.chemical_specific_energy_Gy(
            z_Gy=2.0, rce=backend.relative_chemical_effectiveness(ion)
        )
        survival = model.survival_local_from_events(n_star) * model.survival_global(z_tilde)
        assert 0.0 < survival < 1.0


class TestTrackSegmentSurvival:
    """The mean-field survival curve, and the round trip it closes.

    Alcocer-Avila et al. 2023 eqs. (24)-(33): the full model averages over the
    specific-energy distribution, but in the regime where ``sigma_Z << <Z>``
    the paper's own eq. (33) reduces that to the mean, which is what these
    methods evaluate.
    """

    def _v79(self) -> NanOxModel:
        return NanOxModel.v79(photon=PhotonParameters(alpha_x=0.112, beta_x=0.0259))

    @pytest.mark.golden
    def test_photon_survival_reproduces_the_lq_reference_exactly(self) -> None:
        """The calibration chain closes on itself.

        ``alpha_r,L = alpha_x/eta`` acting on ``<Z> = eta D`` gives back
        ``alpha_x D``; ``beta_G = beta_x/eta^2`` acting on
        ``Ztilde = RCE eta D`` with ``RCE = 1`` gives back ``beta_x D^2``. Both
        factors of ``eta`` must cancel, so this is a real test of the two
        published relations together rather than of either alone.
        """
        model = self._v79().with_beta_G_from_photon()
        for dose in (0.5, 2.0, 5.0, 10.0):
            assert model.photon_survival(dose) == pytest.approx(
                math.exp(-0.112 * dose - 0.0259 * dose**2), rel=1e-12
            )

    def test_the_shipped_beta_G_closes_only_to_its_printed_precision(self) -> None:
        """The presets carry the papers' rounded ``beta_G``, not ``beta_r/eta^2``.

        V79 prints 0.0405 where the chain gives 0.040469 -- 0.08% high. So the
        round trip above holds to three significant figures with the shipped
        value and to machine precision only once ``beta_G`` is derived. Pinned
        because the gap is real, tiny, and otherwise looks like a bug.
        """
        shipped = self._v79()
        derived = shipped.with_beta_G_from_photon()
        assert shipped.cell_line.beta_G_per_Gy2 == 0.0405
        assert derived.cell_line.beta_G_per_Gy2 == pytest.approx(0.0259 / 0.64)
        exact = math.exp(-0.112 * 4.0 - 0.0259 * 16.0)
        assert shipped.photon_survival(4.0) == pytest.approx(exact, rel=1e-2)
        assert shipped.photon_survival(4.0) != pytest.approx(exact, rel=1e-9)
        assert derived.photon_survival(4.0) == pytest.approx(exact, rel=1e-12)

    def test_deriving_beta_G_needs_a_photon_reference(self) -> None:
        with pytest.raises(ValueError, match="photon"):
            NanOxModel.v79().with_beta_G_from_photon()

    @pytest.mark.golden
    @pytest.mark.parametrize(
        ("preset", "z0", "sigma", "h", "d_s", "beta_G", "beta_r"), PUBLISHED_CELL_LINES
    )
    def test_photon_survival_closes_for_every_published_cell_line(
        self,
        preset: str,
        z0: float,
        sigma: float,
        h: float,
        d_s: float,
        beta_G: str,
        beta_r: float,
    ) -> None:
        """Uses each line's own recorded ``beta_r``, so the shipped ``beta_G``
        and the shipped provenance have to agree."""
        model = getattr(NanOxModel, preset)(photon=PhotonParameters(alpha_x=0.2, beta_x=beta_r))
        # beta_G is stored rounded to 3 s.f., so the round trip is good to
        # that precision rather than to machine epsilon.
        assert model.photon_survival(3.0) == pytest.approx(
            math.exp(-0.2 * 3.0 - beta_r * 9.0), rel=2e-3
        )

    def test_the_shoulder_comes_entirely_from_global_events(self) -> None:
        """Local events alone are a straight line on a log-survival plot.

        2023 eq. (52). Curvature is the second difference of ``ln S``; it must
        be zero for the local-only curve and negative once global events are
        included.
        """
        model = self._v79()
        doses = [1.0, 2.0, 3.0]
        local = [
            -math.log(model.survival_track_segment(dose_Gy=d, alpha_local_per_Gy=0.2, rce=0.0))
            for d in doses
        ]
        both = [
            -math.log(model.survival_track_segment(dose_Gy=d, alpha_local_per_Gy=0.2, rce=1.0))
            for d in doses
        ]
        assert local[0] - 2 * local[1] + local[2] == pytest.approx(0.0, abs=1e-12)
        assert both[0] - 2 * both[1] + both[2] > 1e-6

    def test_a_lower_rce_straightens_the_curve(self) -> None:
        """High-LET ions produce fewer primary radicals per unit dose, so the
        global-event term weakens and the survival curve loses its shoulder --
        NanOx's mechanism for the LET dependence of beta."""
        model = self._v79()
        curvatures = []
        for rce in (1.0, 0.7, 0.4):
            vals = [
                -math.log(model.survival_track_segment(dose_Gy=d, alpha_local_per_Gy=0.2, rce=rce))
                for d in (1.0, 2.0, 3.0)
            ]
            curvatures.append(vals[0] - 2 * vals[1] + vals[2])
        assert curvatures[0] > curvatures[1] > curvatures[2] > 0.0

    def test_effective_beta_scales_as_rce_squared(self) -> None:
        """``beta = beta_G (RCE eta)^2`` -- the quadratic coefficient the model
        would report for a track-segment irradiation."""
        model = self._v79()
        assert model.effective_beta_per_Gy2(rce=0.5) == pytest.approx(
            model.cell_line.beta_G_per_Gy2 * (0.5 * 0.8) ** 2
        )
        # RCE = 1 must return the photon beta.
        assert model.effective_beta_per_Gy2(rce=1.0) == pytest.approx(0.0259, rel=2e-3)

    def test_photon_survival_requires_a_photon_reference(self) -> None:
        with pytest.raises(ValueError, match="photon"):
            NanOxModel.v79().photon_survival(2.0)

    def test_track_segment_survival_validates_inputs(self) -> None:
        model = self._v79()
        with pytest.raises(ValueError, match="dose_Gy"):
            model.survival_track_segment(dose_Gy=-1.0, alpha_local_per_Gy=0.2)
        with pytest.raises(ValueError, match="alpha_local_per_Gy"):
            model.survival_track_segment(dose_Gy=1.0, alpha_local_per_Gy=-0.2)
        with pytest.raises(ValueError, match="rce"):
            model.survival_track_segment(dose_Gy=1.0, alpha_local_per_Gy=0.2, rce=-0.1)

    def test_effective_beta_validates_rce(self) -> None:
        with pytest.raises(ValueError, match="rce"):
            self._v79().effective_beta_per_Gy2(rce=-1.0)


class TestLogNormalSpectrum:
    """``LocalTargetSpectrum.log_normal``: an illustration with an exact, fixed grid."""

    def test_grid_density_and_mean_are_the_documented_ones(self) -> None:
        median, w = 8.0e3, 0.9
        grid = np.logspace(np.log10(median) - 4 * w / 2.303, np.log10(median) + 4 * w / 2.303, 900)
        dens = np.exp(-((np.log(grid / median)) ** 2) / (2 * w**2)) / (
            grid * w * np.sqrt(2 * np.pi)
        )
        mean_hit = float(np.trapezoid(grid * dens, grid) / np.trapezoid(dens, grid))
        expected = LocalTargetSpectrum.from_density(grid, dens, mean_hit_z_Gy=mean_hit)
        assert LocalTargetSpectrum.log_normal(median, w) == expected

    def test_is_normalised_and_centred_on_the_median(self) -> None:
        spectrum = LocalTargetSpectrum.log_normal(8.0e3, 0.5)
        assert sum(spectrum.probability) == pytest.approx(1.0)
        cdf = np.cumsum(spectrum.probability)
        median = spectrum.z_Gy[int(np.searchsorted(cdf, 0.5))]
        assert median == pytest.approx(8.0e3, rel=0.02)
        # <z> of a log-normal is median * exp(sigma^2 / 2)
        assert spectrum.mean_hit_z_Gy == pytest.approx(8.0e3 * math.exp(0.125), rel=1e-3)

    def test_width_alone_moves_alpha_core_by_more_than_a_decade(self) -> None:
        """The example's panel 5: same median, the fluctuations set alpha_c."""
        model = v79()
        alphas = [
            model.alpha_core(LocalTargetSpectrum.log_normal(8.0e3, float(w)))
            for w in np.linspace(0.15, 2.4, 40)
        ]
        assert max(alphas) / min(alphas) > 10.0

    @pytest.mark.parametrize(
        ("median", "width", "n", "match"),
        [(0.0, 0.5, 900, "median_z_Gy"), (8e3, 0.0, 900, "sigma_ln"), (8e3, 0.5, 1, "n_points")],
    )
    def test_rejects_bad_arguments(self, median: float, width: float, n: int, match: str) -> None:
        with pytest.raises(ValueError, match=match):
            LocalTargetSpectrum.log_normal(median, width, n_points=n)
