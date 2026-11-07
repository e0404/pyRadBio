# Northcliffe & Schilling 1970: heavy-ion stopping power in water

Northcliffe, L.C. & Schilling, R.F. 1970, "Range and stopping-power tables for heavy
ions", *Nuclear Data Tables* **A7**, 233-463, doi:10.1016/S0092-640X(70)80016-X.

Scholz & Kraft 1996 normalise the LEM radial dose so its track integral returns "the LET as
given by the tables of Hubert et al. and Northcliffe and Schilling" (text below eq. 9; their
refs. 17/18). This file documents the water column of that table, in
`pyradbio/track/data/northcliffe_schilling1970_water.csv`, loaded by
`pyradbio.track.northcliffe_schilling`. Use it to reproduce the 1996 papers, not as a modern
stopping-power reference.

## Which source has water

* **Northcliffe & Schilling 1970**: representative ions for every 1 <= Z <= 103 in 24 media
  (twelve solid elements, nine gases, polyethylene, Mylar, **water**) at 38 energies,
  0.0125-12 MeV/u.
* **Hubert, Bimbot & Gauvin 1990** (and its 1980 predecessor, the edition Scholz cites):
  36 solid elements only, Z >= 2, E >= 2.5 MeV/u. No water, no protons.

So the paper's own LET in water comes from Northcliffe & Schilling alone.

## Source layout (for re-extraction)

* Table pages start at PDF page 22 and alternate stopping power / range for Z = 1, 2, 3, ...;
  the stopping-power page for Z is 0-based index `21 + 2*(Z-1)`. Page captions OCR badly;
  position is reliable (every stopping-power column has an interior maximum, every range
  column rises monotonically).
* Each page has 78 wide lines: header, 38 rows, header, 38 rows. Block 1 is twelve solid
  elements; block 2 is `H HE N O NE AR KR XE RN MYLAR (CH2)n WATER`, so **water is the last
  data column** (x ~ 471, left of the right-hand energy column at x ~ 509).
* Locate the water column by **x-position, not by "right-most token on the row"**: the scan
  drops single cells (uranium's page loses four on one row), and "right-most" then silently
  promotes a neighbouring medium into the water column.
* Rows are indexed by ordinal position on the fixed 38-point grid (printed energies carry the
  same OCR damage as the data).
* Source units MeV/(mg/cm^2); the CSV stores MeV cm^2/g (x1000).

## Transcription and checks

Values were read off 330-dpi renders and cross-checked against an independent OCR pass. On
the first 16 ions, 573 of 608 cells agreed exactly and every one of the 35 differences was an
OCR error, resolved at 900 dpi (mostly the scan's 4/6 confusion: Fe at 0.5 MeV/u reads
`46.499`, OCR gave `66.699`). Across all 60 scan-read columns the passes agreed on 2074 of
2206 readable cells. Do not trust a single automated pass on this scan.

Physical checks, kept as tests in `tests/track/test_northcliffe_schilling.py`:

1. every column rises to a single Bragg peak and falls;
2. the peak moves monotonically up in energy with Z (0.25 MeV/u for H to 3.2 MeV/u for U);
3. at 12 MeV/u the ratio to the proton column reproduces Z^2 for light ions (1.000, 1.000,
   0.992, 0.984, 0.977, 0.977, 0.982, 0.966, 0.969, 0.960 for H..Ne), then falls with
   screening (Si 0.872, Ar 0.804, Fe 0.689, Kr 0.582, Xe 0.469, U 0.361);
4. the proton column against NIST PSTAR for liquid water.

The lanthanides Z = 57-62 print a flat, faintly double-humped top (La: 95.939, 95.779,
95.599, 95.668 across 0.8-1.25 MeV/u). This is in the source (Pr re-checked at 700 dpi), so
the shape test allows a 0.5% plateau wobble.

## Coverage

All 103 ions. The CSV `source` column records the route: `scan` (60 ions; every ion the audit
flagged, plus Z = 100 whose PDF page has no text layer) and `ocr` (43 ions that passed both
screens; Pr spot-checked at 700 dpi).

## Validity

| E [MeV/u] | NS70 / PSTAR (protons) |
|---|---|
| 12 | 1.012 |
| 10 | 1.007 |
| 8 | 1.008 |
| 5 | 0.999 |
| 2 | 1.058 |
| 1 | 1.181 |
| 0.5 | 1.289 |
| 0.2 | 1.011 |
| 0.1 | 0.690 |
| 0.05 | 0.511 |

Pre-ICRU-49 compilation: excellent above ~5 MeV/u, **too high** from 2 to ~0.25 MeV/u (+6% at
2, +18% at 1, +29% at 0.5 MeV/u), too low below ~0.15 MeV/u (roughly half the modern value at
0.05 MeV/u). The proton Bragg peak sits at 0.25 MeV/u (669 MeV cm^2/g) where modern data put
it near 0.08 MeV/u.

**The 0.3-1 MeV/u excess is in the printed table, not our transcription.**

* The water column was re-read cell for cell off zoomed renders for 1H (printed p. 254, PDF
  index 21), 4He (p. 256, index 23) and 12C (p. 264, index 31): all 3 x 38 values equal the
  CSV. Spot values, MeV cm^2/g: H 669 / 553 / 308 at 0.25 / 0.5 / 1.0 MeV/u; He 2337 at
  0.32; C 10216 at 0.5.
* The water column is the Bragg sum of the printed gas columns: on the 1H page
  `(2.016 H + 16.00 O)/18.02` reproduces the printed water column at all 38 energies to
  within 0.0017 MeV/(mg/cm^2) (rounding). The anomaly is already in the H and O columns.
* The source says why: water, Mylar and polyethylene are built by Bragg additivity (Sec. IV G,
  p. 244) from the *gaseous* H and O curves (footnote 31, p. 250), and the authors knowingly
  gave up the low-energy proton fit (Sec. VIII, p. 249).
* Heavier ions inherit it. NS70/MSTAR 3.12 (pyMKM table) at 0.25 / 0.5 / 1.0 / 5 MeV/u is
  1.12 / 1.35 / 1.19 / 1.01 for H, 1.05 / 1.29 / 1.17 / 1.01 for He, 1.02 / 1.25 / 1.18 / 1.02
  for C (pyMKM's Geant4 and FLUKA tables agree within ~2%).

`test_low_energy_excess_is_the_printed_table` pins these printed cells so a "correction"
towards modern data fails. For modern stopping powers choose a different source; do not edit
this table.

## Known limits

* **The grid stops at 12 MeV/u**, while Scholz & Kraft 1996 Fig. 4 runs to 20 MeV/u.
  `StoppingPowerTable.interpolate` raises rather than extrapolating, so the 16 and 20 MeV/u
  points of that reconstruction need `SimpleBetheStoppingPower` (or a pyMKM source). State
  which was used; do not extend the table by fitting.
* `validation/scholz_kraft1996_lem1.py` defaults to `LEM1_STOPPING=ns1970`: this table where
  valid, analytic Bethe above 12 MeV/u, **counting** the fallbacks and printing the tally. The
  MC batch records the choice in its checkpoint and refuses to reuse rows computed with a
  different stopping power (LET enters both the x axis and the cross-section conversion).
