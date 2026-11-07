# ASTAR: alpha particles (He-4) in liquid water (ICRU 49)

**Origin.** NIST ASTAR, text interface:
`POST https://physics.nist.gov/cgi-bin/Star/ap_table-t.pl` with
`prog=ASTAR`, `matno=276` ("Water, Liquid"), `ShowDefault=on` (NIST's default energy grid).
Fetched by `scripts/stopping/fetch_sources.py`.

**Retrieved.** 2026-10-13 (UTC timestamp in `FETCHED.json`).

**Citation.** M.J. Berger, J.S. Coursey, M.A. Zucker and J. Chang, *ESTAR, PSTAR, and
ASTAR*, NIST Standard Reference Database 124, NISTIR 4999, doi:10.18434/T4NC7P (online
October 1998, last data update July 2017). The helium tables are those of ICRU Report 49
(ICRU 1993, *Stopping Powers and Ranges for Protons and Alpha Particles*, Bethesda, MD).

**Notice.** As for PSTAR (see `../pstar/PROVENANCE.md`): NIST Standard Reference Data,
copyright secured under 15 U.S.C. 290e; NIST data disclaimer applies. Only regridded
tables derived from this source are shipped; the raw values are not redistributed.

**Content and conventions.**

* Columns as printed: T [MeV], electronic, nuclear and total stopping power
  [MeV cm^2/g], CSDA range and projected range [g/cm^2], detour factor.
* **T is the total kinetic energy of the He-4 ion** ("alpha particle (helium ion)
  kinetic energy, MeV"); the package's MeV/u is T/4. The grid therefore ends at
  1000 MeV = **250 MeV/u**.
* Mean excitation energy of water: I = 75 eV (ICRU 49).
* 121 rows, 1 keV to 1 GeV. NIST's introduction states 122 default energies; as for
  PSTAR, the returned grid lacks 22.5 MeV.
* Every row satisfies electronic + nuclear = total to the printed precision.

**Files.** Raw HTML changes per request (Cloudflare token), recorded only as retrieved;
the extracted CSV is stable and verified.

Raw response as retrieved: `astar_water_liquid.html`, sha256
`4796c79e2ea83acf5ea18706d08da28b55d44ce6224cde7397d467b519821872`, 12376 bytes.

| file | sha256 | bytes |
|---|---|---|
| `astar_water_liquid.csv` | `a84e54af47088a80675f76efd476514ac7da425b3e6e0596892e1cd784475d99` | 8457 |
