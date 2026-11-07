# Ashley, Ritchie & Brandt (1973), Table I: the Barkas function kappa(b, x)

**Origin.** Table I (printed p. 2403) of the 1973 article cited below: kappa(b, x)
[their eq. (2)] for b = 1.6, 1.8 and 2.0 at 41 values of x = v^2 / (v0^2 Z2) from 0.115
to 100.

**Citation.**

* J.C. Ashley, R.H. Ritchie and W. Brandt, "Z1^3-Dependent Stopping Power and Range
  Contributions", Phys. Rev. A 8, 2402 (1973), doi:10.1103/PhysRevA.8.2402.
* The function F it tabulates: J.C. Ashley, R.H. Ritchie and W. Brandt, "Z1^3 Effect in
  the Stopping Power of Matter for Charged Particles", Phys. Rev. B 5, 2393 (1972),
  doi:10.1103/PhysRevB.5.2393, eqs. (12)-(14).

**Retrieved.** From the maintainer's copy of the article (Zotero item
G9AYC9IV, attachment HAXGEG86). There is nothing to download: `arb1973_table1.csv` is a
**hand transcription** of the printed table (values are the printed strings), made from
the page image. It was cross-checked against the PDF's OCR text layer: 161 of 164 entries
agree; the three differences were settled on a 1200-dpi render of the page -- the OCR
drops kappa(1.8, 1.7) = 0.3201 and kappa(2.0, 0.115) = 0.09511, and misreads
kappa(1.8, 1) = 0.2846 as 0.2848 (a stray mark on the printed 4).

**Notice.** The article is (c) American Physical Society. The transcription is raw source
data and is **not committed**; pyRadBio ships only its own table derived from it (the
function F(b/x^1/2) = kappa x^1/2 L(x) for x >= 1, regridded; see
`scripts/stopping/build_tables.py`).

**Use (ICRU 49 sec. 2.3).** kappa(b, x) = F(b/x^1/2) / (x^1/2 L(x)) with L(x) = ln(x/q),
q = 0.18, for x > 1 (ARB 1973 eqs. 2, 3; "a value q = 0.18 gives good agreement with the
statistical model when x > 1"). Below x = 1 ARB's L(x) contains a statistical
inner-shell correction that is shown only as a figure (ARB 1972 Fig. 3), so F is not
recovered there and the entries with x < 1 are unused. ICRU 49 eq. (2.8) quotes "a
tabulation of the function F_ARB ... in Ashley et al. (1974)", which we do not hold.

**Files.**

| file | sha256 | bytes |
|---|---|---|
| `arb1973_table1.csv` | `61e05d70e5f3f6fcc2aea23bed5044ad8e7286b324165a62b3e8d32f880fa2d1` | 1378 |
