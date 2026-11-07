# Raw water stopping-power sources (not committed)

This folder holds the raw source data behind `pyradbio.track.stopping`. Only this
README and each source's `PROVENANCE.md` are tracked; every downloaded or extracted
file is git-ignored and must never be committed. What the package ships is our own
regridded tables in `pyradbio/track/stopping/data/`, built from these files by
`scripts/stopping/build_tables.py`.

Fetch (or re-fetch) everything with

    python scripts/stopping/fetch_sources.py            # downloads + extracts
    python scripts/stopping/fetch_sources.py --verify   # checksums vs PROVENANCE.md

MSTAR must be downloaded by hand (the IAEA site rejects scripted clients); see
`mstar/PROVENANCE.md`. ICRU 73 currently comes from libdedx's raw files as a
surrogate for the report; `icru73/README.md` defines the format a transcription of
the report itself can replace it with.

| folder | source | used by |
|---|---|---|
| `pstar/` | NIST PSTAR (SRD 124), protons in liquid water, ICRU 49 | `XSTAR`, `ICRU("2005"/"2009")` |
| `astar/` | NIST ASTAR (SRD 124), He-4 in liquid water, ICRU 49 | `XSTAR`, `ICRU("2005"/"2009")` |
| `icru90/` | NIST summary of ICRU Report 90: p, alpha, C in liquid water | `ICRU("2016")`, `ICRU90Procedure` tests |
| `mstar/` | MSTAR 3.12 (Paul & Schinner), IAEA: FORTRAN source + database | the MSTAR port |
| `icru73/` | ICRU Report 73 (2005) and errata (2009), Z 3-18, via libdedx (surrogate) | `ICRU(...)` for Z 3-18 |
| `estar/` | NIST ESTAR (SRD 124), electrons in liquid water: density-effect delta at I = 75 eV | `Bethe(density_effect="tabulated")` |
| `pdg/` | PDG 2024 (CC BY 4.0): Sternheimer coefficients (I = 75 eV) and plasma energy of water | `Bethe(density_effect="sternheimer")` |
| `nist_isotopes/` | NIST Atomic Weights and Isotopic Compositions (masses: AME2012) | ion masses |
| `brandt1982/` | Brandt & Kitagawa 1982, Table II (ion ionization; hand transcription) | `Bethe(effective_charge="brandt_kitagawa")` |
| `icru49/` | ICRU 49 empirical (Varelas-Biersack) coefficients and cut-offs for liquid water; Table 4.1 scaled nuclear stopping (hand transcriptions) | `ICRU90Procedure` for p and alpha; nuclear stopping (`quantity="total"`) |
| `ahlen1980/` | Ahlen 1980, Table XII (cos chi; hand transcription from the page image) | `Bethe(ahlen=True)` |
| `arb1973/` | Ashley, Ritchie & Brandt 1973, Table I (hand transcription, checked against the PDF) | `Bethe(barkas=True)` |

`python scripts/stopping/mstar_reference.py` compiles the original MSTAR FORTRAN
(gfortran) and writes the reference values the port is tested against into
`mstar/reference/`.

Tests that need these files skip cleanly when they are absent.
