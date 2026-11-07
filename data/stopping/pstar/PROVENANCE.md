# PSTAR: protons in liquid water (ICRU 49)

**Origin.** NIST PSTAR, text interface:
`POST https://physics.nist.gov/cgi-bin/Star/ap_table-t.pl` with
`prog=PSTAR`, `matno=276` ("Water, Liquid"), `ShowDefault=on` (NIST's default energy grid).
Fetched by `scripts/stopping/fetch_sources.py`.

**Retrieved.** 2026-10-13 (UTC timestamp in `FETCHED.json`).

**Citation.** M.J. Berger, J.S. Coursey, M.A. Zucker and J. Chang, *ESTAR, PSTAR, and
ASTAR*, NIST Standard Reference Database 124, NISTIR 4999, doi:10.18434/T4NC7P (online
October 1998, last data update July 2017; updated by S.M. Seltzer and P.M. Bergstrom).
The proton tables are those of ICRU Report 49 (ICRU 1993, *Stopping Powers and Ranges
for Protons and Alpha Particles*, Bethesda, MD).

**Notice.** NIST Standard Reference Data are covered by the Standard Reference Data
Act, 15 U.S.C. 290e. NIST's statement
(https://www.nist.gov/open/copyright-fair-use-and-licensing-statements-srd-data-software-and-technical-series-publications):
"Copyright protection on this compilation of data has been secured by the Secretary of
the U.S. Department of Commerce on behalf of the United States in the United States and
all countries that are parties to the Universal Copyright Convention, pursuant to
Section 290(e) of Title 15 of the United States Code." NIST data disclaimer: NIST makes
no warranties and is not liable for errors or omissions in the database
(https://www.nist.gov/copyrights-disclaimers). The maintainer has accepted shipping
*regridded* tables derived from this source; the raw values are not redistributed.

**Content and conventions.**

* Columns as printed: T [MeV], electronic, nuclear and total stopping power
  [MeV cm^2/g], CSDA range and projected range [g/cm^2], detour factor.
* T is the proton's kinetic energy; for a proton this equals MeV per nucleon.
* Mean excitation energy of water: I = 75 eV (ICRU 49). Checked: the online
  values equal the ICRU 90 table (I = 78 eV) at and below 0.1 MeV and lie 0.3-1.2%
  higher from 1 MeV up, as expected for the lower I, so the interactive database still
  serves ICRU 49, not ICRU 90.
* 132 rows, 1 keV to 10 GeV. NIST's introduction states a default grid of 133 energies;
  the returned grid lacks 22.5 MeV, the only decade without its x2.25 point. No
  energy was requested beyond the defaults.
* Every row satisfies electronic + nuclear = total to the printed precision (checked by
  the fetch script).

**Files.** The raw HTML embeds a per-request Cloudflare token, so its checksum changes
on every fetch; it is recorded as retrieved and not verified. The extracted CSV holds
the printed strings unchanged and is stable across fetches (verified on two fetches).

Raw response as retrieved: `pstar_water_liquid.html`, sha256
`1f3d968fe1dec71f6b2b258c4324f2db2b4a367cf86ef89a54b5c91cfabf5d4a`, 13197 bytes.

| file | sha256 | bytes |
|---|---|---|
| `pstar_water_liquid.csv` | `e863613aaadb1de297b7b599784d8fd2c82fc8c5498bcdd65049073643842de9` | 9194 |
