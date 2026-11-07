# ICRU 73 data format (liquid water)

`ICRU(edition=...)` builds its ICRU 73 part from one CSV per edition in this folder.
The same format serves both producers:

* `scripts/stopping/fetch_sources.py --only icru73` writes it from libdedx's raw files
  (the current surrogate, see `PROVENANCE.md`);
* a hand transcription of the report tables can be dropped in under the same names,
  replacing the surrogate, without any code change.

## Files

| file | edition | meaning |
|---|---|---|
| `icru73_2005_water_liquid.csv` | `2005` | ICRU Report 73 as originally published |
| `icru73_2009_water_liquid.csv` | `2009` | ICRU Report 73 with the errata applied |

## Layout

UTF-8 text. Comment lines start with `#`; a comment of the form `# key: value` is
metadata. Then one CSV header line and one row per (ion, energy):

    # edition: 2005
    # material: liquid water
    # quantity: electronic
    # energy: kinetic energy per nucleon, MeV/u
    # source: <citation / where the numbers came from; may span several lines>
    Z,energy_MeV_u,S_el_MeV_cm2_g
    3,0.025,<value>
    3,0.03,<value>
    ...

Required metadata keys: `edition` (`2005` or `2009`), `material` (must be
`liquid water`), `quantity` (`electronic`, or `electronic+nuclear` if a fourth column
is given), `energy`, `source` (repeatable; the lines are joined). Further keys (e.g.
`i_value_eV`, `table`, `page`) are kept as metadata and may be added freely.

Columns:

* `Z`: atomic number of the ion, 3-18 (ICRU 73 does not cover H and He; those come
  from ICRU 49 / PSTAR / ASTAR in every edition).
* `energy_MeV_u`: kinetic energy **per nucleon** in MeV/u, as printed.
* `S_el_MeV_cm2_g`: electronic mass stopping power in MeV cm^2/g, as printed.
* optional `S_nuc_MeV_cm2_g`: nuclear mass stopping power, if the source tabulates it.

Rules: numbers are kept exactly as printed (no reformatting); energies strictly
increasing within each ion; every ion has the same energy grid; values positive.
