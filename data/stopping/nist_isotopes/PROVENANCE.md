# NIST Atomic Weights and Isotopic Compositions (isotope masses)

**Origin.** NIST, "Atomic Weights and Isotopic Compositions with Relative Atomic Masses",
https://www.nist.gov/pml/atomic-weights-and-isotopic-compositions-relative-atomic-masses;
data from
`GET https://physics.nist.gov/cgi-bin/Compositions/stand_alone.pl?ele=&ascii=ascii2&isotype=all`
(all elements, all isotopes, linearized ASCII). Fetched by
`scripts/stopping/fetch_sources.py`.

**Retrieved.** 2026-10-13 (UTC timestamp in `FETCHED.json`). Database version 4.1.

**Citation.** J.S. Coursey, D.J. Schwab, J.J. Tsai and R.A. Dragoset, *Atomic Weights and
Isotopic Compositions with Relative Atomic Masses*, NIST Physical Measurement
Laboratory, version 4.1. Per that page: atomic weights from J. Meija et al., *Atomic
Weights of the Elements 2013*; isotopic compositions from M. Berglund and M.E. Wieser,
*Isotopic Compositions of the Elements 2009*; **relative atomic masses of the isotopes
from the AME2012 atomic mass evaluation** (G. Audi, M. Wang, A.H. Wapstra, F.G. Kondev,
M. MacCormick, X. Xu and B. Pfeiffer, Chinese Physics C 36 (2012) 1287-1602, and the
companion tables). NIST notes the compilation "does not represent a critical
evaluation by the NIST Physical Measurement Laboratory".

**Notice.** NIST web content: "information presented on NIST sites are considered
public information and may be distributed or copied. Use of appropriate byline/photo/
image credits is requested" (https://www.nist.gov/copyrights-disclaimers), plus the NIST
data disclaimer. The maintainer confirmed that these masses may be
reprinted.

**Use.** Atomic masses for the ions we model; nuclear mass = atomic mass minus
Z electron masses (electron binding neglected). Default mass numbers for Z without a
therapy convention come from the isotopic compositions in the same file.

**Content.** One record per isotope: Z, symbol (D and T listed separately for hydrogen),
A, relative atomic mass (value and uncertainty in the last digits, '#' = estimated,
stripped), isotopic composition (blank where not given). 3352 records.

**Files.** Raw HTML changes per request (Cloudflare token), recorded only as retrieved;
the extracted CSV is stable and verified.

Raw response as retrieved: `nist_isotopic_compositions.html`, sha256
`b64cfcc0b301ff2fe5d4bbaf12839e5b98392a04b438422b2ea70c2c50389dee`, 568794 bytes.

| file | sha256 | bytes |
|---|---|---|
| `nist_isotopic_compositions.csv` | `2e9e796b8b414ad5fb04154e7da7c57ae77b6cd0ac61748683b34b7eca240c37` | 85126 |
