# RadPhysBio: Zanni et al. 2024 (doi:10.3390/ijms25094729)

Open-access, literature-mined database of human-cell survival experiments under X-rays,
gamma-rays, protons, carbon ions and alpha-particles: a second reference dataset alongside
user-supplied PIDE data.

Zanni V, Papakonstantinou D, et al. *RadPhysBio: A Radiobiological Database for the Prediction
of Cell Survival upon Exposure to Ionizing Radiation.* Int. J. Mol. Sci. 2024;25(9):4729
(CC BY 4.0). Metadata record doi:10.5281/zenodo.12582567. Database and CSV export:
<https://radbiodb.physics.ntua.gr/radphysbio/>.

## Export semantics and loader contract

The site export concatenates five radiation tables, each restarting `#ExpID`, so a row's
identity is `(TypeofRadiation, #ExpID)`. `load_radphysbio()` returns every row as a
`RadPhysBioEntry` (LQ values, LET, energy, dose rate, cell metadata, damage yields).

* `Energy (MeV)` is the total **nominal** beam energy; `energy_MeV_u` divides by the mass
  number (1H, 12C, 4He). It is **not a verified target energy**.
* LQ: `-ln S = alpha d + beta d^2`, alpha in Gy^-1, beta in Gy^-2. Values may be transcribed
  or fitted to digitized survival curves; the export does not say which.
* LET is in water, keV/um. The paper describes partly derived ion LETs (including values
  inferred from RBE-LET curves); a LET-based benchmark inherits that.
* The `RBE` column has no specified survival level or reference; the harness does not use it.
* Photon references are separate rows. Pairing uses the same publication (PMID) and a
  normalised cell-line identity (Greek look-alike letters, punctuation and documented aliases
  handled by `benchmark.cell_identity`).
* Several photon references with different LQ parameters are ambiguous: the first by `#Exp` is
  selected by default; `include_ambiguous=False` excludes them. The pairing report records it.
* The same experiment can be listed under several PMIDs. Ion rows with identical normalised
  cell line, Z, LET, alpha and beta are compared only after photon pairing and
  completeness/ambiguity filtering; the earliest eligible PMID is kept
  (`collapse_duplicates=True`, counted in the pairing report), so an unusable original cannot
  suppress a usable later copy. Only the ion row is compared: the kept copy is scored against
  its own publication's photon reference. Re-listings that never form an eligible pair keep
  their unpaired/incomplete category and count as `n_relisted_unscored`. Re-digitized copies
  differ in the last digits and stay.
* Missing ion or photon beta becomes zero only when building a benchmark record (the raw entry
  keeps `None`); models requiring positive photon beta exclude such records.
* Latin and Greek `N/A` markers are accepted. The two unquoted `SP6,5` cell-name rows are
  repaired; the DSB column's trailing `f` becomes a separate fit flag. Other values stay as
  exported.

The attributed 103-row test fixture is an excerpt of RadPhysBio, not PIDE
(`tests/benchmark/fixtures/README.md`).

## Running a benchmark

Set `PYRADBIO_RADPHYSBIO_DIR` to a folder with the site export, or pass a file:

```bash
pyradbio benchmark --dataset radphysbio --path export.csv --model mcnamara --out benchmark_results/radphysbio/
```

`load_radphysbio_dataset()` pairs the observations and returns a `ReferenceDataset`. Every
paired record has `energy_is_nominal=True`, and **energy-based models are rejected before
evaluation** (also when bare records are passed), so LEM and energy-based MKM routes cannot be
scored on this export. For an energy-based comparison, build separate records from
independently verified target energies; do not clear the nominal flag.

LET-based models remain available: for `energy_is_nominal` records the harness waives its
energy gate (a LET model never reads the energy, and no ion route can run on the same
records), and every paired record with a LET and a positive photon alpha is offered to the
model. Reports carry the dataset caveats and the pairing tally; model scope and incomplete
observations can reduce the evaluated count.

## Known limits

Human cell lines only; incomplete photon coverage; ambiguous photon choices; repeated
experiments under different publications; possible transcription defects. `KNOWN_DEFECTS`
holds qualitative review notes only, with no PIDE-derived replacement parameters; verify
corrections against the original publications before applying them.

## Cross-checking against a local PIDE export

PIDE workbooks, raw traces, publication bridges and any report containing their values stay
outside version control (the default `data/pide/` folder is ignored except its `README.md`).
Never copy experiment-level PIDE values into documentation, fixtures, constants or expected
test results. The PIDE publication-to-PMID bridge (`pide_publications_pmid.csv`) is built from
PIDE's reference list and is therefore not shipped; a registered user keeps it beside their
PIDE export, where `load_pide_publications()` finds it, untracked.

```python
from pyradbio.benchmark import crosscheck as cc
from pyradbio.benchmark.pide import load_pide_observations
from pyradbio.benchmark.radphysbio import load_radphysbio

bridge = cc.pmid_by_publication(cc.load_pide_publications())
left = cc.rows_from_pide(load_pide_observations(variant="paper"), bridge)
right = cc.rows_from_radphysbio(load_radphysbio())
result = cc.crosscheck(left, right)
print(result.format_summary())
```

`load_pide_observations()` keeps missing LET and individual LQ quantities as `None`, so an
experiment contributes what it has; `load_pide()` deliberately requires complete records and
must not be used to assess missing data. The generic matcher uses publication, normalised cell
line, Z and LET within its tolerance; target-energy fallback exists for other sources when LET
is missing, but RadPhysBio's nominal energy is omitted from `CrossRow` to prevent a false match
to a target energy, so its rows match on LET only. The classifier treats each LQ quantity
separately. Experiment-level cross-check results are intentionally not reproduced here.
