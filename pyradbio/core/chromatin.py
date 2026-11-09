"""Chromatin-based sizing of subnuclear domains (model-agnostic biophysics).

The radiation-sensitive *subnuclear domains* of the microdosimetric kinetic
models -- and the giant chromatin loops of the LEM IV / GLOBLE picture -- stand
for subnuclear structures (giant chromatin loops, chromosome territories) that
each carry a roughly fixed amount of DNA (~2 Mbp). Treating the nuclear DNA as
spatially homogeneous, a domain is just the sub-volume of the nucleus that holds
that much DNA, so its radius follows purely from the nuclear radius and the
nucleus' total DNA content -- no model parameter, only cell geometry:

    r_d = R_n * (lambda / Gamma)^(1/3)                              (Parisi eq 15)

with ``lambda`` the DNA per domain (~2 Mbp) and ``Gamma`` the nuclear DNA content

    Gamma = gamma * p * xi                                          (Parisi eq 16)

for the haploid genome size ``gamma``, the ploidy ``p``, and a cell-cycle factor
``xi`` (G1 -> 1, G2 -> 2; an asynchronized 2:1 G1:G2 mix averages 4/3). For
aneuploid tumour lines the ploidy may be estimated from the chromosome count

    p = x / x_n.                                                    (Parisi eq 17)

This is general radiobiological geometry, not tied to any single model: it can
size the MCF MKM / MKM domain, the LEM IV giant-loop side length, or any other
chromatin-substructure target. Reference: Parisi, Beltran & Furutani,
*Phys. Med. Biol.* 67 (2022) 185013, doi:10.1088/1361-6560/ac7375, sec 2.5.4;
the ~2 Mbp loop size is Rogakou et al 1998 / Friedrich et al 2012
(doi:10.1667/RR2964.1). Genome sizes: McMahon et al 2017 (Sci. Rep. 7, 10790).
"""

from __future__ import annotations

from collections.abc import Mapping

HAPLOID_GENOME_MBP: Mapping[str, float] = {
    "human": 3050.0,
    "rat": 2750.0,
    "chinese_hamster": 2700.0,
    "mouse": 2650.0,
}
"""Haploid genome size per set of chromosomes [Mbp] (McMahon et al 2017, via
Parisi 2022 sec 2.5.4). Keys are normalised by `_normalise_species`.
"""

CHROMATIN_LOOP_MBP: float = 2.0
"""DNA per radiation-sensitive chromatin substructure ("giant loop") [Mbp]
(Rogakou et al 1998; Friedrich et al 2012; Parisi 2022 eq 15).
"""

CELL_CYCLE_DNA_FACTOR: Mapping[str, float] = {
    "g1": 1.0,
    "g2": 2.0,
    "asynchronized": 4.0 / 3.0,
}
"""Cell-cycle DNA-content factor ``xi`` (Parisi 2022 eq 16 / McMahon et al 2017):
G1 holds one genome copy, G2 two; an asynchronized 2:1 G1:G2 mix averages 4/3.
"""


def _normalise_species(species: str) -> str:
    """Lower-case and unify separators so ``"Chinese hamster"`` maps to a key."""
    return species.strip().lower().replace(" ", "_").replace("-", "_")


def ploidy_from_chromosome_count(mean_chromosomes: float, normal_set_chromosomes: float) -> float:
    """Estimate ploidy ``p = x / x_n`` for an aneuploid population (Parisi eq 17).

    ``mean_chromosomes`` is the mean chromosome number in the (aneuploid) cell
    population and ``normal_set_chromosomes`` the count in a normal set (e.g. 46
    for human). This is an approximation -- mutated chromosomes may carry an
    abnormal amount of DNA. Both counts must be positive.
    """
    if mean_chromosomes <= 0.0 or normal_set_chromosomes <= 0.0:
        raise ValueError("chromosome counts must be > 0")
    return mean_chromosomes / normal_set_chromosomes


def nuclear_dna_content_mbp(
    haploid_genome_mbp: float,
    ploidy: float = 2.0,
    cell_cycle_factor: float = CELL_CYCLE_DNA_FACTOR["asynchronized"],
) -> float:
    """Average nuclear DNA content ``Gamma = gamma * p * xi`` [Mbp] (Parisi eq 16).

    ``haploid_genome_mbp`` is the DNA in one chromosome set (``gamma``),
    ``ploidy`` is ``p`` (2 for a normal diploid cell), and ``cell_cycle_factor``
    is ``xi`` (see [`CELL_CYCLE_DNA_FACTOR`][]; default = asynchronized, 4/3).
    All inputs must be positive.
    """
    if haploid_genome_mbp <= 0.0 or ploidy <= 0.0 or cell_cycle_factor <= 0.0:
        raise ValueError("genome size, ploidy and cell-cycle factor must be > 0")
    return haploid_genome_mbp * ploidy * cell_cycle_factor


def chromatin_domain_radius_um(
    nucleus_radius_um: float,
    nuclear_dna_mbp: float,
    loop_mbp: float = CHROMATIN_LOOP_MBP,
) -> float:
    """Subnuclear-domain radius ``r_d = R_n (lambda/Gamma)^(1/3)`` [um] (Parisi eq 15).

    ``nucleus_radius_um`` is ``R_n``, ``nuclear_dna_mbp`` the total nuclear DNA
    content ``Gamma`` (from [`nuclear_dna_content_mbp`][]), and ``loop_mbp`` the
    DNA per domain ``lambda`` (default [`CHROMATIN_LOOP_MBP`][], 2 Mbp). Assumes
    a homogeneous DNA distribution, so the domain-to-nucleus *volume* ratio equals
    the DNA ratio. All inputs must be positive.
    """
    if nucleus_radius_um <= 0.0 or nuclear_dna_mbp <= 0.0 or loop_mbp <= 0.0:
        raise ValueError("nucleus radius, nuclear DNA and loop size must be > 0")
    # Base is positive (validated above), so the cube root is real.
    return float(nucleus_radius_um * (loop_mbp / nuclear_dna_mbp) ** (1.0 / 3.0))


def domain_radius_from_genome(
    nucleus_radius_um: float,
    *,
    species: str | None = None,
    haploid_genome_mbp: float | None = None,
    ploidy: float = 2.0,
    cell_cycle: str = "asynchronized",
    loop_mbp: float = CHROMATIN_LOOP_MBP,
) -> float:
    """Domain radius ``r_d`` from nuclear radius + genome / ploidy / cell cycle.

    Convenience wrapper chaining [`nuclear_dna_content_mbp`][] (eq 16) and
    [`chromatin_domain_radius_um`][] (eq 15). Provide the haploid genome size
    either by ``species`` (a key of [`HAPLOID_GENOME_MBP`][], separator- and
    case-insensitive) *or* explicitly via ``haploid_genome_mbp`` -- exactly one.
    ``cell_cycle`` is a key of [`CELL_CYCLE_DNA_FACTOR`][] (default asynchronized).
    """
    if (species is None) == (haploid_genome_mbp is None):
        raise ValueError("provide exactly one of species or haploid_genome_mbp")
    if species is not None:
        key = _normalise_species(species)
        if key not in HAPLOID_GENOME_MBP:
            raise ValueError(
                f"unknown species {species!r}; known: {sorted(HAPLOID_GENOME_MBP)} "
                "(or pass haploid_genome_mbp explicitly)"
            )
        gamma = HAPLOID_GENOME_MBP[key]
    else:
        assert haploid_genome_mbp is not None
        gamma = haploid_genome_mbp
    cycle_key = cell_cycle.strip().lower()
    if cycle_key not in CELL_CYCLE_DNA_FACTOR:
        raise ValueError(
            f"unknown cell_cycle {cell_cycle!r}; known: {sorted(CELL_CYCLE_DNA_FACTOR)}"
        )
    nuclear_dna = nuclear_dna_content_mbp(gamma, ploidy, CELL_CYCLE_DNA_FACTOR[cycle_key])
    return chromatin_domain_radius_um(nucleus_radius_um, nuclear_dna, loop_mbp)
