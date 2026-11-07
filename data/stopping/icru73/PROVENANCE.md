# ICRU Report 73 and its 2009 errata (liquid water, Z 3-18), via libdedx

**Origin.** libdedx (https://github.com/APTG/libdedx), raw regeneration inputs in
`data/raw/`, pinned to commit `5f345aa031d52ac1eed8a71b1e51781c3089bd53`:
`ICRU73.dat` + `icru73Eng.dat` and `ICRU73_NEW.dat` + `icru73_newEng.dat`, fetched from
`https://raw.githubusercontent.com/APTG/libdedx/<commit>/data/raw/`. Converted by
`scripts/stopping/fetch_sources.py --only icru73` into our format (`README.md`).

**Retrieved.** UTC timestamp in `FETCHED.json`.

**Why a surrogate.** libdedx's ICRU 73 data are used in place of the report tables; a
transcription of the report itself can replace the two CSVs (same format, same names).
The libdedx files stay in this git-ignored folder; only our regridded tables derived
from them ship.

**Citation.** ICRU (2005), *Stopping of Ions Heavier than Helium*, ICRU Report 73,
J. ICRU 5 (1), doi:10.1093/jicru/ndi001 (title, journal, volume, issue and year
checked against Crossref; the report's chapters carry separate DOIs,
ndi001 among them). Errata: P. Sigmund, A. Schinner and H. Paul, *Errata and Addenda
for ICRU Report 73, Stopping of Ions Heavier than Helium* (May 14, 2009), a standalone
document that prints no journal, volume or DOI. Both documents were read
(maintainer's Zotero library). Data files as distributed by the libdedx project
(APTG, https://github.com/APTG/libdedx); its README names no authors.

**Notice.** The libdedx repository is licensed GPL-3.0 (GitHub licence metadata,
`COPYING`). The files used are numerical tables transcribed from ICRU Report 73; the
report values themselves are ICRU's. The raw libdedx files are not redistributed;
no libdedx code was copied (its sources were read, not reused, to answer how it
treats nuclear stopping).

**Content.**

* Block format: a header `#<target ID>:<Z>:<n>` then `n` values, on a shared 53-point
  grid 0.025-1000 MeV/u (libdedx's README: its tables use MeV/nucleon). Both grids
  are byte-identical.
* `ICRU73.dat`: 56 targets x Z 3-18 (896 blocks). `ICRU73_NEW.dat`: targets 104 and
  276 only, for Z 1-18 (36 blocks). Liquid water = ICRU/NIST ID 276. Only water,
  Z 3-18 is taken from either file; 848 rows each.
* Values: mass stopping power in MeV cm^2/g; the reports print MeV cm^2/mg (x1000).
  ICRU 73 prints 4 significant digits, the errata 5 -- the files keep them as printed.
* **Verified against the printed reports.** The liquid-water blocks equal
  the printed tables at all 848 points each: `ICRU73.dat` = ICRU 73 pp. 232-233
  (edition `2005`), `ICRU73_NEW.dat` = errata Appendix B (edition `2009`), which
  "replace pp. 232-233 in ICRU 73" (errata sec. 2.2). Spot checks read off the
  rendered ICRU 73 page (Li and C at 0.025 and 1 MeV/u) agree.
* **I-values.** ICRU 73 prints no water I-value. The errata, sec. 2.2: "The
  oscillator-strength spectrum used in Report 73 for liquid water gives an I-value of
  67.2 eV (compared to 75.0 eV in Reports 37 and 49)"; the revised tables use "Liquid
  water, I = 78.0 eV" (errata Table 1; only the valence-shell input of the PASS code
  changed, following Schardt et al. 2008). The errata quote deviations from the original
  of ~1.5% at 1000 MeV/u, < 5% at 1 MeV/u and up to ~15% at 0.025 MeV/u, as measured
  between the two files.
* **Quantity: electronic only.** ICRU 73 sec. 6.1: "This section presents tables of
  electronic mass stopping forces"; the tables come from the PASS code (binary stopping
  theory of Sigmund and Schinner). ICRU 73 has no nuclear-stopping tables.
* **Energy axis.** E/A1 in MeV; ICRU 73 footnote 6: "no distinction will be made between
  energy per nucleon and energy per atomic mass unit because the numerical difference
  amounts to at most 0.25 % for stable isotopes of all elements from lithium upward."
* **H and He in `ICRU73_NEW.dat` are not used.** The errata contain Li-Ar only (their
  appendices A-D: nylon, liquid water, water vapour, liquid-water ranges), so libdedx's
  H/He blocks do not come from them; they lie close to, but not on, ICRU 49 (+0.8% /
  +1.8% vs PSTAR / ASTAR at 0.025 MeV/u). H and He come from PSTAR/ASTAR.
* One non-positive value exists in `ICRU73.dat` (Na in argon, first energy); not water,
  not used.

**Files.**

| file | sha256 | bytes |
|---|---|---|
| `libdedx/ICRU73.dat` | `03c1509f7dc332e03c82f5de23616a0d9a7d7b9dca843f9c8b5b8cad30891b4f` | 483832 |
| `libdedx/icru73Eng.dat` | `8b5b809f5c0cc4f6018f9faa89e9f73c7b9b81a3c588f98bef556712d4c77dd1` | 198 |
| `libdedx/ICRU73_NEW.dat` | `749768c1c2de475c928cc2c50a354903577139c11debd4b5948b035ca3a3d87e` | 21260 |
| `libdedx/icru73_newEng.dat` | `8b5b809f5c0cc4f6018f9faa89e9f73c7b9b81a3c588f98bef556712d4c77dd1` | 198 |
| `icru73_2005_water_liquid.csv` | `9f468418cacf5f41b6a6105b74a3b5890985e8a26958a17175305f9c11a48799` | 14098 |
| `icru73_2009_water_liquid.csv` | `00d4c84d3388abfe7927a86ef8f039fb95b5aea18472e0fdcebede08255dd772` | 14954 |
