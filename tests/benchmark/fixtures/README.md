# Benchmark fixtures

`synthetic_pide.xlsx` mimics the PIDE 3.4 column
schema with a handful of fabricated rows so CI can exercise the loader and
metrics WITHOUT the real, registration-gated PIDE data. Never place real PIDE
data here.

`radphysbio_excerpt.csv` is a short, verbatim excerpt (103 rows)
of the RadPhysBio database export -- Zanni V, Papakonstantinou D, et al.,
*Int. J. Mol. Sci.* 25(9):4729, 2024, doi:10.3390/ijms25094729, distributed
under CC BY 4.0 (https://radbiodb.physics.ntua.gr/radphysbio/). It keeps the
export's quirks on purpose: two rows whose cell-line name was split on a comma
(31 fields), Greek-letter `N/A` markers, an `f`-flagged DSB value, and one
publication with several photon rows for one line. It exercises the loader and
the ion/photon pairing without the full file. Redistribution is permitted by the
licence; attribution above.
