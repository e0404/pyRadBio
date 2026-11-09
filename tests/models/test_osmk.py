"""OSMK -- oxygen-effect-incorporated SMK (Inaniwa & Kanematsu 2021 and 2023).

Physics acceptance for `pyradbio.models.mkm.osmk`
(doi:10.1088/1361-6560/ac29cc and doi:10.1093/jrr/rrad049; equations restated in
``docs/references/OSMK_Inaniwa2021_2023.md``):

* both formulations collapse **exactly** onto SMK in the ``R -> 1`` limit, with
  ``alpha0 = alpha_L + alpha_S`` -- the structural claim that OSMK is an
  extension of the shipped SMK and not a parallel model;
* the experimental aerobic condition (160 mmHg) is *not* that limit;
* hypoxia raises survival, and its effect fades with LET -- the OER falls from
  ~3 towards unity above ~200 keV/um (2021 discussion), which is the whole point;
* OSMK 2021's quality dependence sits in ``Rmax(z_d,D)``, bounded by ``R_m`` and 1;
* OSMK 2023's sits in the *geometry*: ``r_d' = r_d/f_rd``, ``z0' = z0 f_z0`` with
  ``f_z0`` squared, so the domain shrinks ~3x under anoxia and the microdosimetry
  must be rescored -- an injected engine bound to ``r_d`` is refused, not reused;
* the algebra matches pyMKM's ``biology/oxygen_effect`` to machine precision.
"""

from __future__ import annotations

import importlib.util
import math
from dataclasses import dataclass
from itertools import pairwise

import numpy as np
import pytest

from pyradbio.core.oxygen import AEROBIC_PO2_MMHG
from pyradbio.core.radiation import Ion
from pyradbio.micro.lineal_energy import AmorphousTrackEngine, SphericalSite
from pyradbio.models.mkm.osmk import OSMK2021, OSMK2023
from pyradbio.models.mkm.spectrum_models import StochasticMKM
from pyradbio.track.kiefer_chatterjee import KieferChatterjeeRadialDose
from pyradbio.track.stopping_power import SimpleBetheStoppingPower

requires_pymkm = pytest.mark.skipif(
    importlib.util.find_spec("pymkm") is None,
    reason="pyMKM not installed ([mkm] extra)",
)

# A mid-LET carbon ion (~75 keV/um) and a high-LET one, for OER-vs-LET checks.
CARBON_MID = Ion(6, 12, 30.0)
CARBON_HIGH = Ion(6, 12, 3.0)


class TestPublishedParameters:
    """Table 1 of each paper, transcribed. These are the model's ground truth."""

    def test_osmk2021_hsg(self) -> None:
        m = OSMK2021.hsg_inaniwa2021()
        assert (m.alpha_L_Gy_inv, m.alpha_S_Gy_inv) == (0.00, 0.21)
        assert m.beta0_Gy2 == 0.043
        assert m.domain_radius_um == 0.23
        assert m.nucleus_radius_um == 8.1
        assert m.z0_Gy == 88.0
        assert (m.z_R_Gy, m.gamma_exponent, m.R_m) == (28.0, 1.30, 2.9)
        assert m.k_mmHg == 3.0

    def test_osmk2021_v79_and_cho(self) -> None:
        v79, cho = OSMK2021.v79_inaniwa2021(), OSMK2021.cho_inaniwa2021()
        assert (v79.z0_Gy, v79.domain_radius_um, v79.R_m) == (98.0, 0.29, 2.5)
        assert (cho.z0_Gy, cho.domain_radius_um, cho.R_m) == (46.0, 0.35, 3.1)
        assert cho.alpha_L_Gy_inv == 0.01  # the only non-zero alpha_L in either paper

    def test_osmk2023_tables(self) -> None:
        hsg, v79, cho = (
            OSMK2023.hsg_inaniwa2023(),
            OSMK2023.v79_inaniwa2023(),
            OSMK2023.cho_inaniwa2023(),
        )
        assert (hsg.alpha_S_Gy_inv, hsg.beta0_Gy2, hsg.domain_radius_um) == (0.0797, 0.0677, 0.29)
        assert (hsg.f_rd_max, hsg.z0_Gy, hsg.f_z0_max, hsg.R_max) == (3.00, 60.0, 3.53, 4.46)
        assert (cho.f_rd_max, cho.z0_Gy, cho.f_z0_max, cho.R_max) == (2.69, 37.0, 3.23, 3.51)
        assert (v79.f_rd_max, v79.z0_Gy, v79.f_z0_max, v79.R_max) == (2.50, 73.0, 2.96, 3.41)
        # V79 2023 has alpha_L = alpha_S = 0: its whole alpha comes from beta0*z*.
        assert v79.alpha_S_Gy_inv == 0.0
        assert v79.alpha0 == 0.0

    def test_alpha0_is_the_sum_of_the_two_channels(self) -> None:
        """SMK's ``alpha0`` is OSMK's ``alpha_L + alpha_S`` -- the split is the new bit."""
        assert OSMK2021.hsg_inaniwa2021().alpha0 == pytest.approx(0.21)  # 0.00 + 0.21
        assert OSMK2021.cho_inaniwa2021().alpha0 == pytest.approx(0.18)  # 0.01 + 0.17
        assert OSMK2023.cho_inaniwa2023().alpha0 == pytest.approx(0.132)  # 0.0 + 0.132

    def test_the_three_hsg_fits_are_genuinely_different(self) -> None:
        """Gotcha 1: SMK-2018 / OSMK-2021 / OSMK-2023 refit HSG independently.

        Mixing an SMK parameter set with an oxygen parameter set is wrong in
        every direction; pin the divergence so nobody "harmonises" them.
        """
        smk, o21, o23 = (
            StochasticMKM.hsg_inaniwa2018(),
            OSMK2021.hsg_inaniwa2021(),
            OSMK2023.hsg_inaniwa2023(),
        )
        assert (smk.z0_Gy, o21.z0_Gy, o23.z0_Gy) == (66.0, 88.0, 60.0)
        assert (smk.domain_radius_um, o21.domain_radius_um, o23.domain_radius_um) == (
            0.28,
            0.23,
            0.29,
        )
        assert (smk.beta0, o21.beta0, o23.beta0) == (0.0568, 0.043, 0.0677)


class TestAerobicLimitReducesToSMK:
    """The structural claim: OSMK *is* SMK once the oxygen correction switches off."""

    @pytest.mark.parametrize("osmk", [OSMK2021.hsg_inaniwa2021(), OSMK2023.hsg_inaniwa2023()])
    def test_alpha_beta_at_infinite_po2_equals_smk(self, osmk: OSMK2021 | OSMK2023) -> None:
        smk = StochasticMKM(
            nucleus_radius_um=osmk.nucleus_radius_um,
            domain_radius_um=osmk.domain_radius_um,
            z0_Gy=osmk.z0_Gy,
            alpha0_Gy_inv=osmk.alpha0,
            beta0_Gy2=osmk.beta0,
        )
        ours = osmk.alpha_beta(CARBON_MID, pO2_mmHg=math.inf)
        theirs = smk.alpha_beta(CARBON_MID)
        assert ours.alpha == pytest.approx(theirs.alpha, rel=1e-12)
        assert ours.beta == pytest.approx(theirs.beta, rel=1e-12)

    @pytest.mark.parametrize("osmk", [OSMK2021.hsg_inaniwa2021(), OSMK2023.hsg_inaniwa2023()])
    def test_survival_at_infinite_po2_equals_smk(self, osmk: OSMK2021 | OSMK2023) -> None:
        smk = StochasticMKM(
            nucleus_radius_um=osmk.nucleus_radius_um,
            domain_radius_um=osmk.domain_radius_um,
            z0_Gy=osmk.z0_Gy,
            alpha0_Gy_inv=osmk.alpha0,
            beta0_Gy2=osmk.beta0,
        )
        doses = np.linspace(0.0, 8.0, 17)
        assert osmk.survival(CARBON_MID, doses, pO2_mmHg=math.inf) == pytest.approx(
            smk.survival(CARBON_MID, doses), rel=1e-12
        )

    def test_experimental_aerobic_is_not_the_infinite_limit(self) -> None:
        """Gotcha 2: at 160 mmHg ``R ~ 1.014``, so the two differ by ~1-3%."""
        osmk = OSMK2023.hsg_inaniwa2023()
        aerobic = osmk.alpha_beta(CARBON_MID, pO2_mmHg=AEROBIC_PO2_MMHG)
        limit = osmk.alpha_beta(CARBON_MID, pO2_mmHg=math.inf)
        assert aerobic.alpha < limit.alpha  # slightly more resistant
        assert aerobic.alpha == pytest.approx(limit.alpha, rel=5e-2)
        assert aerobic.alpha != pytest.approx(limit.alpha, rel=1e-4)

    def test_default_po2_is_the_papers_aerobic_reference(self) -> None:
        osmk = OSMK2021.hsg_inaniwa2021()
        assert osmk.alpha_beta(CARBON_MID).alpha == pytest.approx(
            osmk.alpha_beta(CARBON_MID, pO2_mmHg=AEROBIC_PO2_MMHG).alpha, rel=1e-12
        )


class TestRelativeRadioresistance:
    def test_2023_anoxic_r_is_the_cell_constant_rmax(self) -> None:
        osmk = OSMK2023.hsg_inaniwa2023()
        assert osmk.relative_radioresistance(0.0, z_1D_Gy=1.0) == pytest.approx(4.46, rel=1e-12)
        # ... and it does not depend on radiation quality at all
        assert osmk.relative_radioresistance(5.0, z_1D_Gy=1.0) == pytest.approx(
            osmk.relative_radioresistance(5.0, z_1D_Gy=500.0), rel=1e-12
        )

    def test_2021_rmax_falls_from_Rm_to_unity_with_domain_specific_energy(self) -> None:
        """Eq. (14): ``Rmax(z_d,D)`` is how the 2021 model makes the OER fall with LET."""
        osmk = OSMK2021.hsg_inaniwa2021()
        z = np.geomspace(1e-3, 1e5, 60)
        r_max = np.array([osmk.max_radioresistance(zi) for zi in z])
        assert np.all(np.diff(r_max) < 0.0)
        assert r_max[0] == pytest.approx(osmk.R_m, rel=1e-3)  # low quality -> R_m
        assert r_max[-1] == pytest.approx(1.0, rel=1e-3)  # dense track -> no oxygen effect
        assert np.all((r_max >= 1.0) & (r_max <= osmk.R_m))

    def test_2021_anoxic_r_equals_rmax_of_the_beam(self) -> None:
        osmk = OSMK2021.hsg_inaniwa2021()
        z_d = osmk.specific_energies(CARBON_MID).z_1D_Gy
        assert osmk.relative_radioresistance(0.0, z_1D_Gy=z_d) == pytest.approx(
            osmk.max_radioresistance(z_d), rel=1e-12
        )

    def test_2021_rmax_at_z_equal_zR_is_the_midpoint_of_Rm_and_one(self) -> None:
        """At ``z_d,D = z_R`` the ratio in eq. (14) is ``(1 + R_m)/2`` exactly."""
        osmk = OSMK2021.hsg_inaniwa2021()
        assert osmk.max_radioresistance(osmk.z_R_Gy) == pytest.approx(
            (1.0 + osmk.R_m) / 2.0, rel=1e-12
        )


class TestOSMK2023Geometry:
    def test_anoxic_scaling_factors_match_the_paper(self) -> None:
        """Eqs. (8)-(9) at ``pO2 = 0``: ``f_rd = f_rd_max`` and ``f_z0 = f_z0_max^2``."""
        osmk = OSMK2023.hsg_inaniwa2023()
        f_rd, f_z0 = osmk.scaling_factors(0.0)
        assert f_rd == pytest.approx(3.00, rel=1e-12)
        assert f_z0 == pytest.approx(3.53**2, rel=1e-12)

    def test_anoxic_geometry_shrinks_the_domain_and_lifts_z0(self) -> None:
        osmk = OSMK2023.hsg_inaniwa2023()
        r_d, z0 = osmk.hypoxic_geometry(0.0)
        assert r_d == pytest.approx(0.29 / 3.00, rel=1e-12)  # 0.09667 um
        assert z0 == pytest.approx(60.0 * 3.53**2, rel=1e-12)  # 747.65 Gy

    def test_scaling_factors_are_unity_in_the_aerobic_limit(self) -> None:
        f_rd, f_z0 = OSMK2023.hsg_inaniwa2023().scaling_factors(math.inf)
        assert f_rd == pytest.approx(1.0, rel=1e-12)
        assert f_z0 == pytest.approx(1.0, rel=1e-12)

    def test_geometry_is_monotone_in_po2(self) -> None:
        osmk = OSMK2023.hsg_inaniwa2023()
        radii = [osmk.hypoxic_geometry(p)[0] for p in (0.0, 1.0, 5.0, 20.0, 160.0)]
        assert all(a < b for a, b in pairwise(radii))

    def test_specific_energies_are_rescored_at_the_hypoxic_domain(self) -> None:
        """The event-by-event content of the 2023 revision: ``z_d,D`` really moves."""
        osmk = OSMK2023.hsg_inaniwa2023()
        aerobic = osmk.specific_energies_at(CARBON_MID, math.inf)
        anoxic = osmk.specific_energies_at(CARBON_MID, 0.0)
        # A 3x smaller domain concentrates the same track energy -> larger z_d.
        assert anoxic.z_1D_Gy > 2.0 * aerobic.z_1D_Gy
        # The nucleus is untouched by oxygen in both papers.
        assert anoxic.z_nucleus_Gy == pytest.approx(aerobic.z_nucleus_Gy, rel=1e-12)

    def test_2021_does_not_touch_the_microdosimetry(self) -> None:
        osmk = OSMK2021.hsg_inaniwa2021()
        aerobic = osmk.specific_energies_at(CARBON_MID, math.inf)
        anoxic = osmk.specific_energies_at(CARBON_MID, 0.0)
        assert anoxic.z_1D_Gy == pytest.approx(aerobic.z_1D_Gy, rel=1e-12)
        assert anoxic.z_star_1D_Gy == pytest.approx(aerobic.z_star_1D_Gy, rel=1e-12)

    def test_injected_engine_is_refused_under_hypoxia_not_silently_reused(self) -> None:
        """Gotcha 4: an engine is bound to one site size; ``r_d'`` is a different one."""
        sp = SimpleBetheStoppingPower()
        engine = AmorphousTrackEngine(
            radial_dose=KieferChatterjeeRadialDose(stopping_power=sp),
            site=SphericalSite(diameter_um=2.0 * 0.29),
            stopping_power=sp,
        )
        osmk = OSMK2023.hsg_inaniwa2023(engine=engine)
        with pytest.raises(ValueError, match="engine"):
            osmk.alpha_beta(CARBON_MID, pO2_mmHg=5.0)
        # ... but the aerobic limit needs no rescoring, so it is allowed.
        osmk.alpha_beta(CARBON_MID, pO2_mmHg=math.inf)


class TestHypoxiaRadioresistance:
    @pytest.mark.parametrize("osmk", [OSMK2021.hsg_inaniwa2021(), OSMK2023.hsg_inaniwa2023()])
    def test_hypoxia_raises_survival(self, osmk: OSMK2021 | OSMK2023) -> None:
        doses = np.linspace(0.5, 6.0, 12)
        anoxic = osmk.survival(CARBON_MID, doses, pO2_mmHg=0.0)
        aerobic = osmk.survival(CARBON_MID, doses, pO2_mmHg=AEROBIC_PO2_MMHG)
        assert np.all(anoxic > aerobic)

    @pytest.mark.parametrize("osmk", [OSMK2021.hsg_inaniwa2021(), OSMK2023.hsg_inaniwa2023()])
    def test_survival_is_monotone_in_po2(self, osmk: OSMK2021 | OSMK2023) -> None:
        s = [
            float(osmk.survival(CARBON_MID, 3.0, pO2_mmHg=p)) for p in (0.0, 1.0, 5.0, 20.0, 160.0)
        ]
        assert all(a > b for a, b in pairwise(s))

    @pytest.mark.parametrize("osmk", [OSMK2021.hsg_inaniwa2021(), OSMK2023.hsg_inaniwa2023()])
    def test_alpha_falls_under_hypoxia(self, osmk: OSMK2021 | OSMK2023) -> None:
        a_anoxic = osmk.alpha_beta(CARBON_MID, pO2_mmHg=0.0).alpha
        a_aerobic = osmk.alpha_beta(CARBON_MID, pO2_mmHg=AEROBIC_PO2_MMHG).alpha
        assert a_anoxic < a_aerobic


class TestOER:
    """``OER = D_10(pO2) / D_10(160 mmHg)`` -- the quantity the papers benchmark."""

    @pytest.mark.parametrize("osmk", [OSMK2021.hsg_inaniwa2021(), OSMK2023.hsg_inaniwa2023()])
    def test_aerobic_oer_is_one(self, osmk: OSMK2021 | OSMK2023) -> None:
        assert osmk.oer_at_survival(CARBON_MID, pO2_mmHg=AEROBIC_PO2_MMHG) == pytest.approx(
            1.0, rel=1e-9
        )

    @pytest.mark.parametrize("osmk", [OSMK2021.hsg_inaniwa2021(), OSMK2023.hsg_inaniwa2023()])
    def test_anoxic_oer_exceeds_one_and_falls_with_LET(self, osmk: OSMK2021 | OSMK2023) -> None:
        oer_mid = osmk.oer_at_survival(CARBON_MID, pO2_mmHg=0.0)
        oer_high = osmk.oer_at_survival(CARBON_HIGH, pO2_mmHg=0.0)
        assert oer_mid > 1.0
        assert oer_high < oer_mid  # the OER decreases with increasing LET

    @pytest.mark.parametrize("osmk", [OSMK2021.hsg_inaniwa2021(), OSMK2023.hsg_inaniwa2023()])
    def test_low_LET_anoxic_oer_is_in_the_reported_range(self, osmk: OSMK2021 | OSMK2023) -> None:
        """ "The OER decreased with increasing LET from ~3 to unity" (2021 sec 4.2).

        Evaluated at the *lowest LET the models were fitted to* -- 18.5 keV/um
        helium (Furusawa et al. 2000). Below that both formulations extrapolate,
        and they diverge: at ~2 keV/um the 2023 fit gives OER > 4, outside
        anything reported. Both papers warn the model is untested below Z = 2.
        """
        oer = osmk.oer_at_survival(Ion(2, 4, 10.0), pO2_mmHg=0.0)  # ~18 keV/um
        assert 2.5 < oer < 3.6

    @pytest.mark.parametrize("osmk", [OSMK2021.hsg_inaniwa2021(), OSMK2023.hsg_inaniwa2023()])
    def test_oer_falls_monotonically_across_the_fitted_LET_range(
        self, osmk: OSMK2021 | OSMK2023
    ) -> None:
        """Figs. 5-6 in one assertion: ~3 at 18 keV/um down to ~1 at ~1100.

        The ions span the He/C/Ne set and the 18.5-654 keV/um range of the
        Furusawa data the fits are built on.
        """
        ions = [
            Ion(2, 4, 10.0),
            Ion(2, 4, 3.0),
            Ion(6, 12, 30.0),
            Ion(6, 12, 6.0),
            Ion(10, 20, 4.0),
        ]
        lets = [SimpleBetheStoppingPower().let_keV_um(i) for i in ions]
        assert all(a < b for a, b in pairwise(lets))  # the ions are ordered in LET
        oers = [osmk.oer_at_survival(i, pO2_mmHg=0.0) for i in ions]
        assert all(a > b for a, b in pairwise(oers))
        assert 2.5 < oers[0] < 3.6
        assert oers[-1] < 1.4

    @pytest.mark.parametrize("osmk", [OSMK2021.hsg_inaniwa2021(), OSMK2023.hsg_inaniwa2023()])
    def test_particle_type_matters_at_the_same_LET(self, osmk: OSMK2021 | OSMK2023) -> None:
        """ "a slight dependence on particle type at the same LET" (2021 sec 4.2).

        The mechanism is the same in both formulations and is what
        dose-averaged-LET OER models cannot express: at matched LET the lighter,
        slower ion has the *denser* track (shorter delta-ray range), hence the
        larger ``z_d,D``, hence the smaller oxygen effect.
        """
        sp = SimpleBetheStoppingPower()
        helium, carbon = Ion(2, 4, 2.2), Ion(6, 12, 34.86)  # both ~59 keV/um
        assert sp.let_keV_um(helium) == pytest.approx(sp.let_keV_um(carbon), rel=0.01)
        z_he = osmk.specific_energies_at(helium, 0.0).z_1D_Gy
        z_c = osmk.specific_energies_at(carbon, 0.0).z_1D_Gy
        assert z_he > z_c
        assert osmk.oer_at_survival(helium, pO2_mmHg=0.0) < osmk.oer_at_survival(
            carbon, pO2_mmHg=0.0
        )

    @pytest.mark.parametrize("osmk", [OSMK2021.hsg_inaniwa2021(), OSMK2023.hsg_inaniwa2023()])
    def test_oer_approaches_unity_at_very_high_LET(self, osmk: OSMK2021 | OSMK2023) -> None:
        """Above ~200 keV/um the oxygen effect is essentially gone (2021 discussion)."""
        oer = osmk.oer_at_survival(Ion(10, 20, 2.0), pO2_mmHg=0.0)
        assert 1.0 <= oer < 1.6

    def test_oer_is_monotone_in_po2(self) -> None:
        osmk = OSMK2023.hsg_inaniwa2023()
        oers = [osmk.oer_at_survival(CARBON_MID, pO2_mmHg=p) for p in (0.0, 1.0, 5.0, 20.0)]
        assert all(a > b for a, b in pairwise(oers))


class TestAlgebra:
    def test_alpha_beta_from_specific_energies_matches_the_written_equations(self) -> None:
        """Eqs. (17)-(18) / (11)-(12), spelled out against the implementation."""
        osmk = OSMK2023.hsg_inaniwa2023()
        z_star, z_d = 12.0, 20.0
        r = osmk.relative_radioresistance(5.0, z_1D_Gy=z_d)
        lq = osmk.alpha_beta_from_specific_energies_at(
            pO2_mmHg=5.0, z_star_1D_Gy=z_star, z_1D_Gy=z_d
        )
        b0 = osmk.beta0
        assert lq.alpha == pytest.approx(
            osmk.alpha_L_Gy_inv + osmk.alpha_S_Gy_inv / r + z_star * b0 / r**2, rel=1e-12
        )
        assert lq.beta == pytest.approx(b0 / r**2 * (z_star / z_d), rel=1e-12)

    def test_survival_is_the_smk_eq24_with_the_primed_pair(self) -> None:
        osmk = OSMK2021.hsg_inaniwa2021()
        doses = np.linspace(0.0, 6.0, 13)
        z = osmk.specific_energies_at(CARBON_MID, 5.0)
        lq = osmk.alpha_beta_from_specific_energies_at(
            pO2_mmHg=5.0, z_star_1D_Gy=z.z_star_1D_Gy, z_1D_Gy=z.z_1D_Gy
        )
        gamma = z.z_nucleus_Gy * (0.5 * (lq.alpha + 2 * lq.beta * doses) ** 2 - lq.beta)
        expected = np.exp(-(lq.alpha * doses + lq.beta * doses**2)) * (1 + gamma * doses)
        assert osmk.survival(CARBON_MID, doses, pO2_mmHg=5.0) == pytest.approx(
            np.clip(expected, 0.0, 1.0), rel=1e-12
        )

    def test_negative_po2_rejected(self) -> None:
        with pytest.raises(ValueError, match="pO2"):
            OSMK2021.hsg_inaniwa2021().alpha_beta(CARBON_MID, pO2_mmHg=-1.0)

    def test_zero_domain_energy_falls_back_to_the_intrinsic_pair(self) -> None:
        osmk = OSMK2023.hsg_inaniwa2023()
        lq = osmk.alpha_beta_from_specific_energies_at(
            pO2_mmHg=math.inf, z_star_1D_Gy=0.0, z_1D_Gy=0.0
        )
        assert lq.alpha == pytest.approx(osmk.alpha0)
        assert lq.beta == pytest.approx(osmk.beta0)


@requires_pymkm
class TestPyMKMCrossCheck:
    """pyMKM is the oracle for the MKM family (``AGENTS.md``)."""

    def test_2021_radioresistance_and_lq_match_pymkm(self) -> None:
        from pymkm.biology.oxygen_effect import (
            apply_oxygen_correction_alpha,
            apply_oxygen_correction_beta,
            compute_osmk_radioresistance,
        )

        osmk = OSMK2021.hsg_inaniwa2021()
        z_d, z_star = 25.0, 15.0
        params = _PyMKMParams(
            K=osmk.k_mmHg,
            pO2=5.0,
            zR=osmk.z_R_Gy,
            gamma=osmk.gamma_exponent,
            Rm=osmk.R_m,
            alphaL=osmk.alpha_L_Gy_inv,
            alphaS=osmk.alpha_S_Gy_inv,
            beta0=osmk.beta0,
        )
        r_ref, f_rd, f_z0 = compute_osmk_radioresistance("2021", np.array(z_d), params)
        assert f_rd is None and f_z0 is None  # 2021 has no geometry scaling
        assert osmk.relative_radioresistance(5.0, z_1D_Gy=z_d) == pytest.approx(
            float(r_ref), rel=1e-12
        )

        lq = osmk.alpha_beta_from_specific_energies_at(
            pO2_mmHg=5.0, z_star_1D_Gy=z_star, z_1D_Gy=z_d
        )
        assert lq.alpha == pytest.approx(
            float(apply_oxygen_correction_alpha(np.array(z_star), r_ref, params)), rel=1e-12
        )
        assert lq.beta == pytest.approx(
            float(apply_oxygen_correction_beta(np.array(z_star), np.array(z_d), r_ref, params)),
            rel=1e-12,
        )

    def test_2023_radioresistance_and_scaling_factors_match_pymkm(self) -> None:
        from pymkm.biology.oxygen_effect import compute_osmk_radioresistance

        osmk = OSMK2023.hsg_inaniwa2023()
        params = _PyMKMParams(
            K=osmk.k_mmHg,
            pO2=1.0,
            Rmax=osmk.R_max,
            f_rd_max=osmk.f_rd_max,
            f_z0_max=osmk.f_z0_max,
        )
        r_ref, f_rd_ref, f_z0_ref = compute_osmk_radioresistance("2023", np.array(20.0), params)
        f_rd, f_z0 = osmk.scaling_factors(1.0)
        assert osmk.relative_radioresistance(1.0, z_1D_Gy=20.0) == pytest.approx(
            float(r_ref), rel=1e-12
        )
        assert f_rd == pytest.approx(float(f_rd_ref), rel=1e-12)
        assert f_z0 == pytest.approx(float(f_z0_ref), rel=1e-12)


@dataclass
class _PyMKMParams:
    """Duck-typed stand-in for pyMKM's parameter container (it reads attributes)."""

    K: float = 3.0
    pO2: float = 0.0
    zR: float = 1.0
    gamma: float = 1.0
    Rm: float = 1.0
    Rmax: float = 1.0
    f_rd_max: float = 1.0
    f_z0_max: float = 1.0
    alphaL: float = 0.0
    alphaS: float = 0.0
    beta0: float = 0.0
