# Installation

pyRadBio supports Python 3.10 to 3.14. The core install depends only on NumPy and SciPy; every
heavier dependency is an optional extra, so a plain install stays small and imports quickly.

```bash
pip install pyradbio
```

This is enough for every model, the LQ utilities, the stopping powers and the `pyradbio`
command-line tool. Install the extras below only for the features that need them.

## Optional extras

Install an extra with, for example, `pip install "pyradbio[tables]"`, and combine them with a
comma: `pip install "pyradbio[tables,plot]"`.

| Extra | Installs | Enables |
| --- | --- | --- |
| `tables` | pandas, pyarrow, openpyxl | Parquet I/O and pandas views of the tables (CSV works without it), Excel reading, and the PIDE benchmark harness |
| `benchmark` | python-dotenv | Reading external reference-data locations from a local `.env` file (see below) |
| `fast` | numba | Compiled kernels for the LEM Monte-Carlo and radial-dose integrals |
| `plot` | matplotlib | Running the `examples/` and `validation/` scripts, which plot with matplotlib (the library itself never imports it) |
| `dev` | pytest, hypothesis, ruff, mypy, pre-commit, ... | Running the test suite and the linters when developing pyRadBio |
| `docs` | Zensical, mkdocstrings | Building this documentation site; needs Python 3.11 or newer |
| `all` | `tables`, `benchmark`, `fast`, `plot`, `dev` | Everything above except `docs` and `mkm` |
| `mkm` | pyMKM | The external pyMKM package, as an independent cross-check of the MKM family |

!!! note "`mkm` is not part of `all`"
    pyMKM is an optional validation reference, never a default: `MKMModel` runs on pyRadBio's own
    native \(z^*\) backend, and only the guarded backend in `pyradbio.models.mkm.pymkm_backend`
    imports pyMKM. Install `pyradbio[mkm]` explicitly if you want to cross-check against it. The
    extra also keeps NumPy below 2.4, because pyMKM 1.0.0 still calls `np.trapz`.

## Installing from a clone

To work on pyRadBio itself, or to run the `examples/` and `validation/` scripts, install an
editable copy from the repository:

```bash
git clone https://github.com/e0404/pyRadBio.git
cd pyRadBio
pip install -e ".[all]"
```

The branch workflow, the checks to run before a pull request and the rules for data and numerics
are in [Contributing](../development/contributing.md).

## Checking the installation

```bash
pyradbio --version
python -c "import pyradbio; print(pyradbio.__version__)"
```

Both print the installed version. A version of `0+unknown` means the package is being imported
from a source tree that was never installed; run `pip install -e .` there.

## External reference datasets

Some benchmarks compare models against measured or published data that pyRadBio does **not**
ship. Only self-digitised paper data is bundled with the package; everything below is optional
and you only need it for the benchmark harness and the cross-implementation checks.

### PIDE

The Particle Irradiation Data Ensemble (PIDE, GSI) is registration-gated and not redistributable,
so it is never bundled and must never be committed to a repository. Obtain it from the
[GSI PIDE project page](https://www.gsi.de/work/forschung/biophysik/forschungsfelder/radiobiological_modelling/pide_project)
and make it known to pyRadBio in one of three ways:

- place the export in a `data/pide/` folder in the directory you run pyradbio from (this folder
  is git-ignored in a clone);
- set the `PYRADBIO_PIDE_DIR` environment variable to the folder holding it;
- pass an explicit path, `pyradbio benchmark --pide /path/to/PIDE3.4.xlsx ...`.

See [Benchmarking against data](../user-guide/benchmarking.md) for what the harness reports.

### The `.env` mechanism

Each kind of external data has its own directory, named by an environment variable. For
convenience, pyRadBio reads these from a `.env` file in the working directory (or a parent of it)
the first time it looks one up. This needs python-dotenv, which the `benchmark` and `dev` extras
install. Variables already set in your shell or in CI always win over the file.

Copy [`.env.example`](https://github.com/e0404/pyRadBio/blob/develop/.env.example) to `.env` and
edit the paths:

```bash
PYRADBIO_PIDE_DIR=/path/to/biodata/pide
PYRADBIO_RADPHYSBIO_DIR=/path/to/biodata/RadPhysBio
```

| Variable | Dataset |
| --- | --- |
| `PYRADBIO_PIDE_DIR` | PIDE database (GSI; registration required, not redistributable) |
| `PYRADBIO_RADPHYSBIO_DIR` | RadPhysBio database export (NTUA; Zanni et al. 2024, CC BY 4.0), exported as CSV |
| `PYRADBIO_TRIP98_DIR` | TRiP98 reference tables (subfolders `DEDX/` and `RBE/{LEM1,LEM4}/`) |
| `PYRADBIO_SURVIVAL_DIR` | JSON exports of the third-party "Survival" code (rapidLEM, rapid mMKM) |
| `PYRADBIO_SURVIVAL_SRC_DIR` | A source checkout of the "Survival" code, for its SRIM stopping-power tables |
| `PYRADBIO_LINEAL_ENERGY_DECUNHA_DIR` | DeCunha 2025 lineal-energy spectrum library (Geant4-DNA) |

Paths may use `~` and forward slashes on every platform. If a dataset is needed but its variable
is unset or points at a missing folder, pyRadBio raises a `MissingDatasetError` that names the
variable to set.
