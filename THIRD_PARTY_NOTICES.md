# Third-party notices

pyRadBio is MIT-licensed (`LICENSE`). The code listed here is derived from third-party
code under its own permissive notice, admitted by `AGENTS.md` rule 3; the notice applies
to those files instead of the MIT licence. The data credited below remain the work of
their authors and are reproduced with attribution. Where our own shipped tables come from
third-party data, the origin and terms are recorded per source in
`data/stopping/<source>/PROVENANCE.md` and summarised in
`docs/references/StoppingPower.md`.

## MSTAR 3.12 (Helmut Paul and Andreas Schinner)

* **Files:** `pyradbio/track/stopping/mstar/` (`__init__.py`, `database.py`, `paul.py`,
  `spline.py`), a Python port of the MSTAR subroutine package (program system MORESTAR,
  V 3.12, 24-05-2004), written by Andreas Schinner and Helmut Paul. The shipped tables
  `pyradbio/track/stopping/data/mstar_c_water_liquid.csv` and `mstar_d_water_liquid.csv`
  hold values computed with this port.
* **Origin:** IAEA Nuclear Data Section, "Electronic Stopping Power of Matter for Ions",
  https://www-nds.iaea.org/stopping/mstar (provenance and checksums
  in `data/stopping/mstar/PROVENANCE.md`).
* **References:** H. Paul and A. Schinner, Nucl. Instrum. Methods Phys. Res. B 179 (2001)
  299; B 195 (2002) 166; Atomic Data Nucl. Data Tables 85 (2003) 377.
* **Notice** (verbatim from the program page):

  > This program and/or the corresponding subroutine package is hereby released as
  > freeware. This means you can freely use, copy, modify, and/or distribute the source
  > code. However, please do not remove the authors' names from the source code comments.
  > The authors cannot be made responsible for any damage caused by the use of their
  > program code. YOU ARE USING THE ENTIRE PROGRAM SYSTEM OR ANY PART OF IT ENTIRELY AT
  > YOUR OWN RISK!

  As the notice asks, every module of the port names the authors in its source comments.

## Data from publications

* **Files:** the digitized figures and printed tables in `pyradbio/benchmark/data/` and
  `pyradbio/micro/data/`. Each file names its source publication and DOI in its header
  (or its loader module, for the per-figure series), and is used only to validate the
  models against the published results.

## RadPhysBio (Zanni et al. 2024)

* **Source:** V. Zanni, D. Papakonstantinou et al., "RadPhysBio: A Radiobiological
  Database for the Prediction of Cell Survival upon Exposure to Ionizing Radiation",
  Int. J. Mol. Sci. 25(9) (2024) 4729, doi:10.3390/ijms25094729; database at
  https://radbiodb.physics.ntua.gr/radphysbio/ (metadata record
  doi:10.5281/zenodo.12582567).
* **Licence:** Creative Commons Attribution 4.0 International (CC BY 4.0),
  https://creativecommons.org/licenses/by/4.0/.
* **Use:** pyRadBio does not ship the database; `pyradbio.benchmark.radphysbio` reads a
  user's own export, and the benchmark harness validates the models against it. The test
  fixture `tests/benchmark/fixtures/radphysbio_excerpt.csv` is an excerpt (a selection of
  rows) of that export.

## Not included

PIDE (GSI) is a registration-gated database and is not redistributed in any form: the
loader reads a user's own export, and nothing derived from it is shipped.
