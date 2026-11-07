"""MCDS analytic DNA-damage yields (Semenenko & Stewart 2006, doi:10.1088/0031-9155/51/7/004).

Goldens are the paper's own printed fit constants and the limits it states for
them -- never a digitized figure point.
"""

from __future__ import annotations

import math

import pytest

from pyradbio.micro.mcds import (
    GBP_PER_CELL,
    MCDS_DOMAIN_MAX,
    MCDS_DOMAIN_MIN,
    MCDSYields,
    effective_charge_barkas,
    quality_parameter,
)


@pytest.fixture
def mcds() -> MCDSYields:
    return MCDSYields()


class TestQualityParameter:
    """``x = Z_eff^2 / beta^2`` -- MCDS eqs. (1) and (2)."""

    def test_effective_charge_is_the_barkas_form_the_package_already_ships(self) -> None:
        """MCDS eq. (1) is the same Barkas 1963 formula as Katz eq. (12)."""
        from pyradbio.track.stopping_power import effective_charge

        for z, beta in ((1, 0.0150), (2, 0.0328), (6, 0.143), (10, 0.2)):
            assert effective_charge_barkas(z, beta) == pytest.approx(effective_charge(z, beta))

    def test_effective_charge_tends_to_Z_at_high_velocity(self) -> None:
        assert effective_charge_barkas(6, 0.9) == pytest.approx(6.0, rel=1e-9)

    @pytest.mark.golden
    @pytest.mark.parametrize(
        ("z", "a", "t_min_MeV"),
        [(1, 1, 0.105), (2, 4, 2.0)],
    )
    def test_stated_minimum_energies_land_on_the_domain_ceiling(
        self, z: int, a: int, t_min_MeV: float
    ) -> None:
        """Sec. 2.3: T_min 'is ~0.105 MeV and 2 MeV for protons and alpha particles'.

        Those are the energies at which ``x`` reaches the fit's upper limit of
        3200, so reproducing them checks the whole chain -- beta, the Barkas
        effective charge, and the quality parameter -- against a number the
        paper prints for a different purpose.
        """
        from pyradbio.core.radiation import Ion

        ion = Ion(Z=z, A=a, energy_MeV_u=t_min_MeV / a)
        assert quality_parameter(z, ion.beta()) == pytest.approx(MCDS_DOMAIN_MAX, rel=0.01)

    def test_electrons_use_bare_Z_not_the_effective_charge(self) -> None:
        """Sec. 2.3 writes ``Z_eff^2/beta^2`` (or ``Z^2/beta^2`` for electrons)."""
        beta = 0.0177  # ~8e-5 MeV electron
        assert quality_parameter(1, beta, use_effective_charge=False) == pytest.approx(
            1.0 / (beta * beta)
        )

    def test_quality_parameter_rejects_unphysical_beta(self) -> None:
        with pytest.raises(ValueError, match="beta"):
            quality_parameter(1, 0.0)


class TestInterpolationFormula:
    """Eq. (3) ``f(x) = u + (v-u)x/(x+w)``."""

    def test_the_stated_limits_hold(self, mcds: MCDSYields) -> None:
        """Sec. 2.2: 'f(1) ~ f(0) = u and f(inf) = v'."""
        assert mcds.dsb_per_Gy_per_cell(0.0) == pytest.approx(48.9)
        assert mcds.dsb_per_Gy_per_cell(1e12) == pytest.approx(164.5, rel=1e-6)
        assert mcds.dsb_per_Gy_per_cell(1.0) == pytest.approx(48.9, rel=0.02)

    @pytest.mark.golden
    def test_dsb_yield_range_matches_the_printed_range(self, mcds: MCDSYields) -> None:
        """'DSB yields range between 48.9 Gy-1 cell-1 ... and 164.5 Gy-1 cell-1'."""
        assert mcds.dsb_per_Gy_per_cell(0.0) == pytest.approx(48.9)
        assert mcds.dsb_yield_limits_per_Gy_per_cell() == pytest.approx((48.9, 164.5))

    @pytest.mark.golden
    def test_ssb_yield_range_matches_the_printed_range(self, mcds: MCDSYields) -> None:
        """'the number of SSBs is within the range between 291 ... and 1134'."""
        assert mcds.ssb_per_Gy_per_cell(0.0) == pytest.approx(1134.0)
        assert mcds.ssb_per_Gy_per_cell(1e12) == pytest.approx(291.0, rel=1e-6)

    def test_ssb_falls_while_dsb_rises_with_track_quality(self, mcds: MCDSYields) -> None:
        """Clustering converts SSBs into DSBs, so the two run opposite ways."""
        lo, hi = 1.0, 3000.0
        assert mcds.ssb_per_Gy_per_cell(hi) < mcds.ssb_per_Gy_per_cell(lo)
        assert mcds.dsb_per_Gy_per_cell(hi) > mcds.dsb_per_Gy_per_cell(lo)

    def test_dsb_yield_is_monotonic_in_x(self, mcds: MCDSYields) -> None:
        previous = 0.0
        for x in (0.0, 1.0, 10.0, 100.0, 759.0, 3200.0, 1e5):
            y = mcds.dsb_per_Gy_per_cell(x)
            assert y > previous
            previous = y

    def test_w_is_the_half_saturation_point(self, mcds: MCDSYields) -> None:
        """At ``x = w`` eq. (3) is exactly halfway between u and v."""
        mid = mcds.dsb_per_Gy_per_cell(759.0)
        assert mid == pytest.approx(0.5 * (48.9 + 164.5))


class TestUnits:
    @pytest.mark.golden
    def test_cell_and_gbp_yields_are_the_printed_pairs(self, mcds: MCDSYields) -> None:
        """The paper prints both units: 48.9/cell = 8.15/Gbp, 164.5 = 27.4."""
        assert mcds.dsb_per_Gy_per_Gbp(0.0) == pytest.approx(8.15, rel=2e-3)
        assert mcds.dsb_per_Gy_per_Gbp(1e12) == pytest.approx(27.4, rel=2e-3)
        assert mcds.ssb_per_Gy_per_Gbp(0.0) == pytest.approx(189.0, rel=2e-3)
        assert mcds.ssb_per_Gy_per_Gbp(1e12) == pytest.approx(48.5, rel=2e-3)

    def test_the_conversion_is_a_diploid_genome(self) -> None:
        """All four printed pairs imply the same 6.0 Gbp per cell."""
        assert GBP_PER_CELL == pytest.approx(6.0)
        for per_cell, per_gbp in ((48.9, 8.15), (164.5, 27.4), (1134.0, 189.0), (291.0, 48.5)):
            assert per_cell / per_gbp == pytest.approx(GBP_PER_CELL, rel=2e-3)


class TestAlgorithmParameters:
    """Eq. (5): the inputs the paper proposes for the MCDS algorithm itself."""

    @pytest.mark.golden
    def test_the_four_printed_algorithm_parameters(self, mcds: MCDSYields) -> None:
        assert mcds.sigma_Sb_per_Gy_per_cell == pytest.approx(1300.0)
        assert mcds.base_damage_to_strand_break_ratio == pytest.approx(3.0)
        assert mcds.n_min_bp == 9

    @pytest.mark.golden
    def test_n_seg_uses_eq3_with_its_own_constants(self, mcds: MCDSYields) -> None:
        """``n_seg(x) = 149200 - 123600x/(x+267)``, i.e. eq. (3) with v = 25600."""
        assert mcds.n_seg_bp(0.0) == pytest.approx(149200.0)
        assert mcds.n_seg_bp(1e12) == pytest.approx(25600.0, rel=1e-6)
        # the printed closed form, written the other way round
        for x in (1.0, 50.0, 267.0, 3200.0):
            assert mcds.n_seg_bp(x) == pytest.approx(149200.0 - 123600.0 * x / (x + 267.0))


class TestValidityDomain:
    def test_the_domain_is_the_printed_one(self) -> None:
        """'most appropriate for 1 <= Z_eff^2/beta^2 <= 3200'."""
        assert (MCDS_DOMAIN_MIN, MCDS_DOMAIN_MAX) == (1.0, 3200.0)

    def test_out_of_domain_warns_but_still_evaluates(self, mcds: MCDSYields) -> None:
        """Never clamp: warn, evaluate, and let the caller count it."""
        with pytest.warns(UserWarning, match="outside"):
            value = mcds.dsb_per_Gy_per_cell(5000.0, check_domain=True)
        assert value > 0.0

    def test_in_domain_is_silent(self, mcds: MCDSYields, recwarn: pytest.WarningsRecorder) -> None:
        mcds.dsb_per_Gy_per_cell(100.0, check_domain=True)
        assert len(recwarn) == 0

    def test_negative_x_is_rejected(self, mcds: MCDSYields) -> None:
        with pytest.raises(ValueError, match="x must be"):
            mcds.dsb_per_Gy_per_cell(-1.0)


class TestIonRoute:
    def test_yield_for_an_ion_matches_the_explicit_x_route(self, mcds: MCDSYields) -> None:
        from pyradbio.core.radiation import Ion

        ion = Ion(Z=6, A=12, energy_MeV_u=100.0)
        x = quality_parameter(ion.Z, ion.beta())
        assert mcds.dsb_per_Gy_per_Gbp_for_ion(ion) == pytest.approx(mcds.dsb_per_Gy_per_Gbp(x))

    def test_dsb_yield_rises_as_an_ion_slows(self, mcds: MCDSYields) -> None:
        from pyradbio.core.radiation import Ion

        fast = mcds.dsb_per_Gy_per_Gbp_for_ion(Ion(Z=6, A=12, energy_MeV_u=300.0))
        slow = mcds.dsb_per_Gy_per_Gbp_for_ion(Ion(Z=6, A=12, energy_MeV_u=5.0))
        assert slow > fast
        # and stays inside the printed physical range
        assert 8.0 < fast < slow < 27.5

    def test_heavier_ion_gives_more_dsb_at_the_same_speed(self, mcds: MCDSYields) -> None:

        beta = 0.14
        proton = mcds.dsb_per_Gy_per_Gbp(quality_parameter(1, beta))
        carbon = mcds.dsb_per_Gy_per_Gbp(quality_parameter(6, beta))
        assert carbon > proton


class TestPhotonReference:
    @pytest.mark.golden
    def test_frese_reference_yield_is_near_the_low_quality_limit(self, mcds: MCDSYields) -> None:
        """Frese et al. 2012 simulate ``Sigma_x = 8.27 Gy^-1 Gbp^-1`` for their
        reference photon field, just above eq. (3)'s ``x -> 0`` limit of 8.15.

        Not reproducible from eq. (3) alone -- Frese used a secondary-electron
        spectrum -- so this pins the relationship, not the number.
        """
        low_quality_limit = mcds.dsb_per_Gy_per_Gbp(0.0)
        assert low_quality_limit < 8.27
        assert math.isclose(8.27, low_quality_limit, rel_tol=0.02)


class TestSaturationLimit:
    def test_infinite_quality_returns_the_saturation_value_exactly(self, mcds: MCDSYields) -> None:
        """``f(inf) = v``: taken as an exact limit, not a large-x approximation."""
        assert mcds.dsb_per_Gy_per_cell(math.inf) == 164.5
        assert mcds.ssb_per_Gy_per_cell(math.inf) == 291.0
        assert mcds.n_seg_bp(math.inf) == 25600.0

    def test_per_cell_ion_route_is_gbp_route_times_the_genome(self, mcds: MCDSYields) -> None:
        from pyradbio.core.radiation import Ion

        ion = Ion(Z=2, A=4, energy_MeV_u=10.0)
        assert mcds.dsb_per_Gy_per_cell_for_ion(ion) == pytest.approx(
            mcds.dsb_per_Gy_per_Gbp_for_ion(ion) * GBP_PER_CELL
        )
