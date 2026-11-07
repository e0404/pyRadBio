# ICRU Report 90: protons, alpha particles and carbon ions in liquid water

**Origin.** NIST, "Update to ESTAR, PSTAR, and ASTAR Databases" (17-page PDF addendum
to SRD 124), https://www.nist.gov/document/update-estar-pstar-and-astar-databases,
linked from the SRD 124 contents page as the summary of ICRU Report 90. Fetched by
`scripts/stopping/fetch_sources.py`.

**Retrieved.** 2026-10-13 (UTC timestamp in `FETCHED.json`).

**Citation.** ICRU (2014), *Key Data for Ionizing-Radiation Dosimetry: Measurement
Standards and Applications*, ICRU Report 90, J. ICRU 14 (Oxford University Press,
Oxford); released 2016 (as cited in the addendum). Tables distributed by NIST as an
addendum to M.J. Berger et al., NIST SRD 124, doi:10.18434/T4NC7P.

**Notice.** The values are ICRU Report 90 results distributed by NIST within SRD 124:
NIST Standard Reference Data under 15 U.S.C. 290e (see `../pstar/PROVENANCE.md` for
NIST's statement), and the underlying report is an ICRU publication. Only regridded
tables derived from this source are shipped; the raw values are not redistributed.

**Content and conventions.**

* Used: Table A.9 (protons), A.12 (alpha particles), A.15 (carbon ions), each "in liquid
  water, I = 78 eV, density 0.998 g/cm^3"; and Table A.3 (electrons in liquid water)
  for its density-effect column delta only (49 rows, collision +
  radiative = total on every row). Air and graphite tables are not extracted.
* Columns as printed: T [MeV], electronic, nuclear and total stopping power
  [MeV cm^2/g], CSDA range [g/cm^2], and (p, alpha only) detour factor.
* **T is the total kinetic energy of the ion** (the addendum's symbol list: "T, kinetic
  energy of the charged particle"). Checked: reading carbon's T as total
  energy gives a C/He electronic stopping ratio at equal T/A of 8.9-9.07 over
  10-200 MeV/u, i.e. (6/2)^2 as expected at high velocity; reading it as MeV/u gives
  38-64, which is unphysical. Ranges in MeV/u: protons 0.001-10000, alpha
  0.00025-250 (T/4), carbon 0.001/12-10000/12 (about 833 MeV/u).
* Rows: A.9 57, A.12 49, A.15 57. Every row satisfies electronic + nuclear = total to
  the printed precision.
* **Extraction trap.** `pdftotext -layout` misaligns the energy column of these tables
  by one row (and the electron tables' radiative column). The fetch script groups
  words by their position on the page (pypdf), which reproduces the tables exactly; the
  1 keV proton row matches PSTAR's 1 keV row cell for cell, as expected below 0.1 MeV.

**Files.**

| file | sha256 | bytes |
|---|---|---|
| `nist_icru90_update.pdf` | `e8340e3d3dd154f7a278fc39164162ca0f75f4b7fcdb64da2111776abab1b474` | 1589504 |
| `icru90_electron_water_liquid.csv` | `3cc5157dc9208f20a10966c20177e160f95b4eb0df4f22bcaeb241d064257065` | 3706 |
| `icru90_proton_water_liquid.csv` | `8f9cd53c64625ae7908be920520512dc9eb1f278171cdee767369c7fcd87d582` | 3515 |
| `icru90_alpha_water_liquid.csv` | `a5220d109f220fe92699c8c1f7f861d824d4748390215e1b349deee3275b2374` | 3067 |
| `icru90_carbon_water_liquid.csv` | `9249edd69f5bbe49eeefe5ba0e079c9ea7211ac8c0dcc64dd43649ea22d5894d` | 3107 |
