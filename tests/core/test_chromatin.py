"""Chromatin-based subnuclear-domain sizing (Parisi 2022 eqs 15-17)."""

from __future__ import annotations

import pytest

from pyradbio.core import chromatin


class TestNuclearDnaContent:
    def test_chinese_hamster_asynchronized(self) -> None:
        # gamma * p * xi = 2700 * 2 * 4/3 = 7200 Mbp.
        assert chromatin.nuclear_dna_content_mbp(2700.0, ploidy=2.0) == pytest.approx(7200.0)

    def test_g2_doubles_g1(self) -> None:
        g1 = chromatin.nuclear_dna_content_mbp(3050.0, ploidy=2.0, cell_cycle_factor=1.0)
        g2 = chromatin.nuclear_dna_content_mbp(3050.0, ploidy=2.0, cell_cycle_factor=2.0)
        assert g2 == pytest.approx(2.0 * g1)

    def test_rejects_non_positive(self) -> None:
        with pytest.raises(ValueError):
            chromatin.nuclear_dna_content_mbp(0.0)
        with pytest.raises(ValueError):
            chromatin.nuclear_dna_content_mbp(3050.0, ploidy=-1.0)


class TestPloidyFromChromosomes:
    def test_ratio(self) -> None:
        assert chromatin.ploidy_from_chromosome_count(69.0, 46.0) == pytest.approx(69.0 / 46.0)

    def test_diploid_normal(self) -> None:
        assert chromatin.ploidy_from_chromosome_count(46.0, 46.0) == pytest.approx(1.0)

    def test_rejects_non_positive(self) -> None:
        with pytest.raises(ValueError):
            chromatin.ploidy_from_chromosome_count(0.0, 46.0)


class TestDomainRadius:
    def test_scales_linearly_with_nucleus(self) -> None:
        r1 = chromatin.chromatin_domain_radius_um(4.0, 7200.0)
        r2 = chromatin.chromatin_domain_radius_um(8.0, 7200.0)
        assert r2 == pytest.approx(2.0 * r1)

    def test_shrinks_with_more_dna(self) -> None:
        small = chromatin.chromatin_domain_radius_um(4.0, 7200.0)
        big_genome = chromatin.chromatin_domain_radius_um(4.0, 14400.0)
        assert big_genome < small
        # r_d ~ Gamma^(-1/3): doubling Gamma scales r_d by 2^(-1/3).
        assert big_genome == pytest.approx(small * 2.0 ** (-1.0 / 3.0))

    def test_rejects_non_positive(self) -> None:
        with pytest.raises(ValueError):
            chromatin.chromatin_domain_radius_um(0.0, 7200.0)
        with pytest.raises(ValueError):
            chromatin.chromatin_domain_radius_um(4.0, 0.0)


class TestConvenience:
    def test_matches_composed_pipeline(self) -> None:
        gamma = chromatin.nuclear_dna_content_mbp(2700.0, ploidy=2.0, cell_cycle_factor=4.0 / 3.0)
        direct = chromatin.chromatin_domain_radius_um(4.0, gamma)
        chained = chromatin.domain_radius_from_genome(
            4.0, species="chinese_hamster", ploidy=2.0, cell_cycle="asynchronized"
        )
        assert chained == pytest.approx(direct)

    def test_species_name_normalised(self) -> None:
        a = chromatin.domain_radius_from_genome(4.5, species="Chinese Hamster")
        b = chromatin.domain_radius_from_genome(4.5, species="chinese_hamster")
        assert a == pytest.approx(b)

    def test_requires_exactly_one_genome_source(self) -> None:
        with pytest.raises(ValueError):
            chromatin.domain_radius_from_genome(4.0)  # neither species nor size
        with pytest.raises(ValueError):
            chromatin.domain_radius_from_genome(4.0, species="human", haploid_genome_mbp=3050.0)

    def test_unknown_species_and_cycle_raise(self) -> None:
        with pytest.raises(ValueError):
            chromatin.domain_radius_from_genome(4.0, species="tardigrade")
        with pytest.raises(ValueError):
            chromatin.domain_radius_from_genome(4.0, species="human", cell_cycle="S")


@pytest.mark.golden
class TestParisiTable1:
    """Reproduce the a-priori MCF MKM domain radii of Parisi et al 2022, Table 1.

    Parisi, Beltran & Furutani, Phys. Med. Biol. 67 (2022) 185013,
    doi:10.1088/1361-6560/ac7375, eqs (15)-(16): r_d = R_n (lambda/Gamma)^(1/3)
    with lambda = 2 Mbp, asynchronized cells (xi = 4/3), ploidy 2.
    """

    def test_v79_domain_radius(self) -> None:
        # Chinese hamster, R_n = 4.0 um -> Table 1: r_d = 0.26 um.
        r_d = chromatin.domain_radius_from_genome(
            4.0, species="chinese_hamster", ploidy=2.0, cell_cycle="asynchronized"
        )
        assert r_d == pytest.approx(0.26, abs=3e-3)

    def test_hsg_domain_radius(self) -> None:
        # Human, R_n = 4.5 um -> Table 1: r_d = 0.28 um.
        r_d = chromatin.domain_radius_from_genome(
            4.5, species="human", ploidy=2.0, cell_cycle="asynchronized"
        )
        assert r_d == pytest.approx(0.28, abs=3e-3)
