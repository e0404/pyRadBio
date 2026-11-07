# ICRU Report 49 (1993): empirical low-energy coefficients and scaled nuclear stopping

**Origin.** Two hand transcriptions from the report cited below.

1. `icru49_water_empirical.csv`: the rows for liquid water (ID 276) of Table 3.1 (p. 21;
   proton coefficients A1-A5 of eqs. 3.1, 3.2, and the molecular weight), Table 3.5
   (p. 32; alpha coefficients a1-a5 of eq. 3.3) and Table 3.7 (p. 37; the cut-off
   energies T1, T2), plus M_p/u = 1.0073 (p. 23, footnote 7). The equations are on
   pp. 23, 29 and 33 and are implemented in `pyradbio/track/stopping/icru49_empirical.py`.
2. `icru49_table41_reduced_nuclear.csv`: Table 4.1 (p. 46), the scaled nuclear stopping
   power S_hat(T_hat) for the Moliere (Thomas-Fermi) and the ZBL universal potential, all
   104 printed rows (T_hat = 1e-5 to 1e8). The table prints the block T_hat = 1e3 ... 15
   out of order (ICRU 90, footnote 35: "a section of Table 4.1 has been transposed") and
   has no row for T_hat = 20; rows are stored sorted by T_hat. Used with eqs. (4.14),
   (4.15) in `pyradbio/track/stopping/nuclear.py`.

**Citation.**

* ICRU Report 49, Stopping Powers and Ranges for Protons and Alpha Particles
  (International Commission on Radiation Units and Measurements, Bethesda, MD, 1993),
  ISBN 978-0-913394-47-2 (no DOI in the copy held).
* Corrections: ICRU Report 90 (2014), Key Data for Ionizing-Radiation Dosimetry, App. A.3,
  footnote 35 (the transposed section of Table 4.1; a(z, Z) = z^0.23 + Z^0.23 for
  particles heavier than protons).

**Retrieved.** From the maintainer's copy of the report (Zotero item VIFZ9C8Z,
attachment RYYS26GI). Both files hold the printed strings, unchanged.

* File 1: each of the 17 values was matched against the PDF's text layer on its page; A2
  and A5 appear there only with OCR spacing ("4.542E + 00"), and the whole row 276 of
  Table 3.1 was re-read on a 600-dpi render.
* File 2: the 208 values were taken from the PDF's text layer (complete, in a known block
  order; the layer drops several T_hat labels, which were restored from the printed
  layout), then proof-read value by value against 220-dpi renders of the page; all agree.
  A smoothness check of log S_hat vs log T_hat finds no outlier above 1.4e-3.

**Notice.** ICRU reports are copyrighted. The transcriptions are raw source data and are
**not committed**; pyRadBio ships only its own derived tables
(`pyradbio/track/stopping/data/icru49_empirical_water.csv`: the formulae evaluated on its
grid; `icru49_reduced_nuclear.csv`: Table 4.1 regridded), written by
`scripts/stopping/build_tables.py`.

**Checks.**

* The empirical formulae reproduce PSTAR and ASTAR (NIST's ICRU 49 tables, `../pstar/`,
  `../astar/`) below T1 to <= 0.020% (protons, 36 points) and <= 0.049% (alphas, 56
  points) -- the tables' 4-digit printing.
* With Table 4.1, eqs. (4.14), (4.15) and footnote 35, the nuclear columns of ASTAR, ICRU
  90 alpha and ICRU 90 carbon are reproduced to <= 0.07% at every energy; ICRU 49's fit
  (eq. 4.16) agrees with the ZBL column to 0.87% for T_hat < 3 (ICRU 49: "within 1
  percent"), which also confirms the column labels.

**Files.**

| file | sha256 | bytes |
|---|---|---|
| `icru49_water_empirical.csv` | `db74ce9d784c19067985a5819f18267c6280b6c2158d56dce13c662419db247c` | 1205 |
| `icru49_table41_reduced_nuclear.csv` | `e049ef6ce7009eed24b786a9f6c475045dfbbfad45bb83055eb1876c35b89341` | 3241 |
