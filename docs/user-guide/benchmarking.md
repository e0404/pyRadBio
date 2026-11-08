# Benchmarking against data

pyRadBio can score a model against a database of measured LQ parameters. The benchmark layer
reduces every database to one small record type, runs a model over the records, and writes a
report that states, next to the numbers, what a reader must know before believing them. It
lives in [`pyradbio.benchmark`][pyradbio.benchmark] and is driven from the shell by
`pyradbio benchmark`.

The score is about the initial-slope \( \alpha \) (and the derived
\( \mathrm{RBE}_\alpha = \alpha_\text{ion} / \alpha_x \)) of a model against **measured**
values. It is not a reconstruction check against a paper's own model line.

!!! warning "A benchmark is not a ranking"
    A residual on a shared point set says how a model with a given parameter choice fares on
    that data. It must not be used to choose between implementations or parameter variants,
    and several of the proton fits were themselves calibrated on much of the same corpus. Every
    report prints these notes; read them with the numbers.

## Datasets

Three datasets are registered, selected with `--dataset`:

| Dataset | Source | Bundled? |
| --- | --- | --- |
| `pide` (default) | PIDE, the Particle Irradiation Data Ensemble (GSI) | **No**. Registration-gated and not redistributable |
| `radphysbio` | RadPhysBio (Zanni et al. 2024, doi:10.3390/ijms25094729, CC BY 4.0) | No, but open access: you export it yourself |
| `csv` | your own file | n/a |

### PIDE is yours to supply

PIDE is never part of pyRadBio, and nothing derived from it is shipped in the package or the
documentation. Register with GSI, download the export from the
[PIDE project page](https://www.gsi.de/work/forschung/biophysik/forschungsfelder/radiobiological_modelling/pide_project),
and tell the loader where it is, in any of three ways:

- put it in the git-ignored `data/pide/` folder of your working directory (the default),
- set the `PYRADBIO_PIDE_DIR` environment variable (or put it in a `.env` file; this needs
  `python-dotenv`, in the `benchmark` extra),
- or pass an explicit file or folder with `--path` (the older spelling `--pide PATH` still
  works).

An `.xlsx` export needs the `tables` extra (`pip install "pyradbio[tables]"`); CSV exports
need nothing extra. The loader was tested against PIDE 3.4. `--variant` chooses the LQ
columns: `fit` (PIDE's own refits), `paper` (as published), or `fit_then_paper` (the default)
and `paper_then_fit`, which fall back from one to the other per field.

!!! note "Never commit PIDE"
    Keep the file out of version control. The `data/pide/` folder is git-ignored for exactly
    this reason.

### RadPhysBio

Export the CSV from <https://radbiodb.physics.ntua.gr/radphysbio/> and point
`PYRADBIO_RADPHYSBIO_DIR` (or `--path`) at it. The export is five tables concatenated,
largely digitized from figures, covers human cell lines only, and lists photon references as
separate rows. The loader therefore pairs each ion row with an X-ray or gamma row of the same
publication and cell line, and only a fraction of the ion rows have such a partner. The
report states how many were paired and how many were not. Its beam energies are nominal, so
RadPhysBio can only score LET-based models, and energy-based models are rejected.

### Your own CSV

`--dataset csv --path FILE` reads a user file. Columns are matched header-tolerantly (case
and punctuation are ignored), so `alpha_ion`, `AlphaIon` and `alpha-ion` are the same column.

| Field | Accepted headers (first match wins) | Required |
| --- | --- | --- |
| LET in keV/µm | `let`, `letkevum`, `letd`, `doseaveragedlet` | yes |
| ion α, Gy⁻¹ | `alphaion`, `alphai`, `alphal`, `alpha`, ... | yes |
| ion β, Gy⁻² | `betaion`, `betai`, `betal`, `beta`, ... | yes |
| photon α, Gy⁻¹ | `alphax`, `alpharef`, `alphaphoton`, ... | yes |
| photon β, Gy⁻² | `betax`, `betaref`, `betaphoton`, ... | yes |
| ion | `ion`, `particle`, `projectile`, `species` (tokens like `C-12`, `12C`, `C`) or an atomic-number column `charge` / `z` | yes, one of the two |
| energy in MeV/u | `energy`, `e`, `emevu`, `specificenergy` | optional |
| cell line | `cells`, `cell`, `cellline`, `celltype`, `name` | optional |
| photon quality | `photonradiation`, `photon`, `referenceradiation`, `reference` | optional |
| source label | `publicationname`, `source`, `publication`, `doi`, `pmid` | optional |

All LQ values must follow \( -\ln S = \alpha d + \beta d^2 \). A row with a missing required
value or an unresolvable ion is skipped. Delimiters (comma, semicolon, tab) are detected, and
`.xlsx` works with the `tables` extra. A missing energy does not exclude a row from LET-based
models, whereas the track-structure models need an energy at the target. The loader checks
nothing about the file's provenance, LQ convention or LET definition, and says so in the
report. The authoritative alias table is
`pyradbio.benchmark.dataset.REFERENCE_ALIASES`, and
[`load_reference_csv()`][pyradbio.benchmark.dataset.load_reference_csv] documents the format.

### Adding a dataset

Every dataset is reduced to [`ReferenceRecord`][pyradbio.benchmark.dataset.ReferenceRecord]
objects in a [`ReferenceDataset`][pyradbio.benchmark.dataset.ReferenceDataset], which also
carries the caveats rendered into every report. The registry
`pyradbio.benchmark.dataset.DATASET_LOADERS` is a plain dict, so a private
dataset is one loader function registered in your own code:

```py
from pyradbio.benchmark.dataset import DATASET_LOADERS, ReferenceDataset, load_reference_csv


def load_mine(path):
    base = load_reference_csv(path or "mine.csv")
    return ReferenceDataset(
        name="mine",
        records=base.records,
        citation="my group's survival experiments",
        caveats=("Measured with 6 MV photons as the reference.",),
    )


DATASET_LOADERS["mine"] = load_mine
```

Register it before calling `pyradbio.cli.main` and `--dataset mine` becomes available.

## Running a benchmark

```bash
pyradbio benchmark --model lem4 --out benchmark_results/                    # PIDE from data/pide/
pyradbio benchmark --dataset radphysbio --model mcnamara --out rpb_results/
pyradbio benchmark --dataset csv --path mine.csv --model wedenberg --out mine_results/
pyradbio benchmark --pide /path/to/PIDE3.4.xlsx --model lem4 --out benchmark_results/
```

| Option | Meaning |
| --- | --- |
| `--dataset {pide,radphysbio,csv}` | reference dataset (default `pide`) |
| `--path PATH` | dataset file or folder; defaults to `PYRADBIO_PIDE_DIR` or `PYRADBIO_RADPHYSBIO_DIR`; required for `csv` |
| `--pide PATH` | alias for `--dataset pide --path PATH` |
| `--model NAME` | model to score (default `lem1`) |
| `--out DIR` | report directory (default `benchmark_results`) |
| `--variant` | PIDE LQ columns, `fit`, `paper`, `fit_then_paper` (default) or `paper_then_fit`; PIDE only |
| `--limit N` | cap the number of evaluated points, for a quick slice |
| `--no-plots` | skip the PNGs, so matplotlib is not needed |
| `--dt`, `--rn` | LEM transition dose \( D_t \) in Gy and nucleus radius in µm |
| `--rd`, `--alpha0`, `--beta0` | generic MKM domain radius in µm, \( \alpha_0 \), \( \beta_0 \) |

`--model` accepts LEM I-IV (`lem1` to `lem4`), the MKM family at its published presets
(`mkm`, `mmkm`, `smk`, `mcf`, `osmk2021`, `osmk2023`, `imk`), `rmf`, the Katz parameter
sets (`katz_hamster`, `katz_hela`, `katz_t1_todd`, `katz_t1_barendsen`), and the thirteen
phenomenological RBE fits (`wilkens`, `carabe`, `frese`, `wedenberg`, `mcnamara`, `mairani`,
`tilly2`, `tilly10`, `chen`, `jones`, `roru`, `peeler`, `unkelbach`).
`pyradbio.benchmark.harness.benchmark_models()` lists them. Only the generic `lem*` and
`mkm` models accept the model knobs; the preset models run at their published parameters and
refuse an override rather than ignore it.

A run prints one summary line (points evaluated, skipped and out of the model's fitted scope,
the mean \( |\log_{10} \text{error}| \) of \( \alpha \) and the SMAPE) and writes
`report.md` in the output folder, plus `alpha_scatter.png` and `residual_vs_let.png` unless
you pass `--no-plots` (plots need the `plot` extra). The report holds the point counts, the
caveats, three aggregate metrics (mean log error, mean absolute log error, SMAPE) and a
residual table per ion.

### Which points are evaluated

A point is evaluated only if it has a known energy, an ion up to neon (`max_z=10`), an
energy of at least 0.5 MeV/u and a positive measured \( \alpha \). The filters are the same
for every model, so models see the same candidate set. Anything else, and any per-point
model error, is skipped and counted. A point outside the *published fit's* own scope (a
proton-only fit meeting a carbon point, or a fit that needs \( \beta_x \) meeting a row
without one) is counted separately, because that is a statement about the paper, not the
run. The phenomenological models take the dataset's own tabulated LET; the track-structure
models derive LET from the ion and energy through the stopping power.

### What every report states

The report's "Read this with the numbers" section is assembled from three sources:

- **The scope of the run.** It scores against measured survival data, not against a paper's
  model line; it ranks models on a shared point set and is not a reconstruction check.
- **The model family.** LEM models run at one shared \( D_t \) and nucleus radius (30 Gy and
  5 µm by default) across every cell line, so the absolute residual is indicative and only
  the ranking on shared points is meant to carry weight. The MKM models run at one published
  cell line's parameter set; RMF at a fixed Co-60 photon reference; Katz's reported
  \( \alpha \) is the extrapolated radiosensitivity \( k \), not an LQ initial slope; and the
  phenomenological fits were calibrated on much of the same corpus, so agreement there
  validates the implementation of the published parameterisation, not the model.
  Individual models add their own notes, for instance where a coefficient is not printed in
  its own paper.
- **The dataset.** PIDE's loader states which LQ variant was read and how PIDE derived
  missing LET or energy values. RadPhysBio's states that it is largely digitized, that ion
  LETs are partly derived, and the pairing tally. A custom CSV states that its provenance is
  unchecked.

[`model_caveats()`][pyradbio.benchmark.harness.model_caveats] returns the model-side text for
any model name.

## From Python

The CLI is a thin layer over a few functions, which you can call to score a model you built
yourself.

```py
from pyradbio.benchmark.dataset import load_dataset
from pyradbio.benchmark.harness import build_model_spec, run_benchmark
from pyradbio.benchmark.report import write_report

dataset = load_dataset("csv", "mine.csv")  # or "pide" / "radphysbio", with path=None
spec = build_model_spec("lem1", D_t=30.0, nucleus_radius_um=5.0)
result = run_benchmark(dataset, model_name="lem1", spec=spec, limit_per_ion=20)

print(result.n_points, result.n_skipped, result.n_out_of_scope)
print(result.mean_abs_log_error_alpha, result.smape_alpha)
print(write_report(result, "my_report", plots=False))
```

- [`build_model_spec()`][pyradbio.benchmark.harness.build_model_spec] builds a
  [`ModelSpec`][pyradbio.benchmark.harness.ModelSpec] (how to build the model and how it must
  be evaluated). `D_t` may be a per-row rule `(alpha_x, beta_x) -> D_t` instead of one value.
- [`run_benchmark()`][pyradbio.benchmark.harness.run_benchmark] returns a
  [`BenchmarkResult`][pyradbio.benchmark.harness.BenchmarkResult]. `limit` caps the evaluated
  points, and `limit_per_ion` thins each ion to points evenly spaced in LET rank so a bounded
  run still spans each ion's whole LET range. `max_z` and `min_energy_MeV_u` change the
  filters.
- `result.points` holds one [`PointResult`][pyradbio.benchmark.harness.PointResult] per
  evaluated point, with the prediction, the measurement and the log residual, and
  `result.by_ion()` groups them by atomic number.

### Metrics

[`pyradbio.benchmark.metrics`][pyradbio.benchmark.metrics] holds the aggregates the report
uses, with the PIDE-community conventions: `mean_log_error` (signed bias in log space),
`mean_absolute_log_error` (the headline spread) and `smape` (bounded, tolerates zeros). The log
metrics need strictly positive inputs. Two curve-level summaries serve the reconstruction
scripts: `shape_residuals(reference, predicted)`, which splits a mismatch into a scale and a
shape part, and `ratio_summary(ours, reference)` for the statistics of a ratio of two curves.

### Sweeps

[`sweep_energies()`][pyradbio.benchmark.sweeps.sweep_energies] and
[`sweep_lets()`][pyradbio.benchmark.sweeps.sweep_lets] are the loop every comparison needs:
evaluate a model for one species over an energy grid, or over a LET grid by solving for the
energy, and get a [`LetSweep`][pyradbio.benchmark.sweeps.LetSweep] of aligned arrays. Rows the
model cannot evaluate become NaN rather than aborting the sweep.

```python
from pyradbio.benchmark.sweeps import sweep_energies, sweep_lets
from pyradbio.models.mkm import MKMModel
from pyradbio.track.stopping_power import default_stopping_power

model = MKMModel()
sp = default_stopping_power()

by_energy = sweep_energies(model, 6, 12, [5.0, 20.0, 100.0], stopping_power=sp, sort_by_let=True)
print(by_energy.let_keV_um)
print(by_energy.rbe_alpha(alpha_x=model.alpha0))  # alpha / alpha_x per row

# LET grid: the energy bracket picks the branch of the non-monotone LET(E)
by_let = sweep_lets(model, 6, 12, [20.0, 60.0], stopping_power=sp, energy_bounds_MeV_u=(1.0, 400.0))
print(by_let.energy_MeV_u)
```

`LetSweep` also offers `rbe10_lq` and `rbe10_shouldered`, two RBE-at-10%-survival reductions
against different photon references. They are named methods because mixing the two up is a
silent error of several percent: use `rbe10_shouldered` against an LEM photon curve whose
\( D_{10} \) lies above the transition dose.

## Cross-checking two databases

[`pyradbio.benchmark.crosscheck`][pyradbio.benchmark.crosscheck] compares two literature-mined
LQ databases against each other experiment by experiment. Rows are matched when they share
the publication, the cell line, the ion and a LET within a tolerance (3% by default), and
each LQ quantity of a matched pair is classified as `identical`, `rounding`, `digitization`
(within 10% by default), `discrepant`, `sign` or `missing`. That tells you whether a database
transcribes the printed values or digitizes the figures.
[`rows_from_pide()`][pyradbio.benchmark.crosscheck.rows_from_pide] and
[`rows_from_radphysbio()`][pyradbio.benchmark.crosscheck.rows_from_radphysbio] convert the two
loaders' output, and [`crosscheck()`][pyradbio.benchmark.crosscheck.crosscheck] returns a
[`CrossCheckResult`][pyradbio.benchmark.crosscheck.CrossCheckResult] with
`format_summary()` and `worst(quantity)`.

Matching PIDE publications to RadPhysBio ones needs a bibliography bridge file that is
built from PIDE's reference list. It is therefore not shipped: a registered user keeps it
beside their PIDE export (see
[`load_pide_publications()`][pyradbio.benchmark.crosscheck.load_pide_publications]).

## Paper reconstructions and cross-code comparisons

The rest of `pyradbio.benchmark` is not about data scoring. It holds the hand-digitized
paper figures and tables (one loader module per paper) that the golden tests and the scripts
in
[`validation/`](https://github.com/e0404/pyRadBio/tree/develop/validation) read, and the
loaders for external reference tables from other codes that you supply yourself, configured
through `pyradbio.datasets`. `validation/` reconstructs published figures and compares the
models against TRiP98, the Survival code, pyMKM and published spectra. Those comparisons
exist there, run from a clone of the repository, and are deliberately not reproduced in this
documentation. See [Examples](examples.md#validation-and-scripts) for how to run them.
