# ESTAR: electrons in liquid water (density-effect parameter delta, I = 75 eV)

**Origin.** NIST ESTAR, text interface:
`POST https://physics.nist.gov/cgi-bin/Star/e_table-t.pl` with `matno=276`
("Water, Liquid"), `ShowDefault=on` (NIST's default energy grid, 10 keV to 1 GeV).
Fetched by `scripts/stopping/fetch_sources.py --only estar`.

**Retrieved.** 2026-10-14 (UTC timestamp in `FETCHED.json`).

**Citation.** M.J. Berger, J.S. Coursey, M.A. Zucker and J. Chang, *ESTAR, PSTAR, and
ASTAR*, NIST Standard Reference Database 124, NISTIR 4999, doi:10.18434/T4NC7P. Per NIST's
ESTAR method page, collision stopping powers use Bethe theory "with a density-effect
correction evaluated according to Sternheimer (1952, 1982)" (R.M. Sternheimer, Phys. Rev.
88, 851 (1952); R.M. Sternheimer, S.M. Seltzer and M.J. Berger, Phys. Rev. B 26, 6067
(1982)), with the ICRU Report 37 I-values.

**Notice.** NIST Standard Reference Data, copyright secured under 15 U.S.C. 290e (see
`../pstar/PROVENANCE.md` for NIST's statement); NIST data disclaimer applies. Only a
regridded delta(beta*gamma) table derived from this source is shipped.

**Use.** Only the column "DENSITY EFFECT DELTA" is used: delta depends on the medium and
on the projectile's beta*gamma alone, so the electron table supplies delta(beta*gamma)
for any projectile (beta*gamma from the electron kinetic energy T as
sqrt(T(T + 2 m_e c^2)) / m_e c^2). The table header states I = 75.0 eV and density
1.0000 g/cm^3, i.e. the ICRU 37 / ICRU 49 water used by PSTAR and ASTAR.

**Content.** Columns as printed: T [MeV], collision, radiative and total stopping power
[MeV cm^2/g], CSDA range [g/cm^2], radiation yield, density-effect delta. 81 rows;
every row satisfies collision + radiative = total to the printed precision.

**Files.** Raw HTML changes per request (Cloudflare token), recorded only as retrieved;
the extracted CSV is stable and verified.

Raw response as retrieved: `estar_water_liquid.html`, sha256
`9098c90f5b76a42bb3f3a1452d8ffb405601c41d194bc2d7bd4acd7271550fcc`, 9667 bytes.

| file | sha256 | bytes |
|---|---|---|
| `estar_water_liquid.csv` | `6029113dba276ef8d1a50ff8c389ab00c5e721741d7baa29d4642df6153de1ef` | 6074 |
