# Ahlen (1980), Table XII: cos(chi) for the close-collision Mott correction

**Origin.** Table XII (printed p. 150) of the review cited below: cos(chi) against
|Z1 alpha / beta| from 0 to 2.00 (14 rows), "to be used in Eq. (4.23)", the close-collision
Mott correction G(Z1, beta) that ICRU Report 90 (App. A.3) says BEST incorporates for C ions.
Ahlen defines cos(chi) after Doggett & Spencer (1956). The equation itself (eq. 4.23, with the
correction of the Z1^5 term noted in the text) is implemented in
`pyradbio/track/stopping/higher_order.py`.

**Citation.**

* S.P. Ahlen, "Theoretical and experimental aspects of the energy loss of relativistic heavily
  ionizing particles", Rev. Mod. Phys. 52, 121 (1980), doi:10.1103/RevModPhys.52.121,
  eqs. (4.22), (4.23), Table XII.

**Retrieved.** From the maintainer's copy (Zotero item B8LLKD3J, attachment
7FXCCEKP). The PDF's OCR text layer is unusable, but the page images are clear:
`ahlen1980_table12.csv` is a **hand transcription** from a 300-dpi render of p. 150 (values
are the printed strings). Eq. (4.23)'s coefficients were read from a 300-dpi render of p. 149.

**Notice.** The article is (c) American Physical Society. The transcription is raw source
data and is **not committed**; pyRadBio ships only its own interpolated table
(`pyradbio/track/stopping/data/ahlen1980_cos_chi.csv`, written by
`scripts/stopping/build_tables.py`).

**Files.**

| file | sha256 | bytes |
|---|---|---|
| `ahlen1980_table12.csv` | `656ddee1493ddcb830da68425b51174cac2e50c0522adf8387efaeab402e4259` | 498 |
