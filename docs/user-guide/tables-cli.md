# Tables and command line

pyRadBio's typical product is a table: LQ parameters \( (\alpha, \beta) \) over a grid of
ions and energies, ready to hand to a treatment-planning code or a spreadsheet. Every model
builds the same typed table object, the table writes to CSV with the core install alone, and
the `pyradbio` command does the same thing from a shell.

## α/β tables from Python

Every model has an
[`alpha_beta_table(ions, energies_MeV_u)`][pyradbio.models.base.BiophysicalModel.alpha_beta_table]
method. `ions` are atomic numbers, each resolved to its most abundant isotope; the result is
an [`AlphaBetaTable`][pyradbio.tables.alpha_beta.AlphaBetaTable] with one
[`AlphaBetaRow`][pyradbio.tables.alpha_beta.AlphaBetaRow] per ion and energy (`Z`, `A`,
`energy_MeV_u`, `alpha` in Gy⁻¹, `beta` in Gy⁻²) and a `metadata` mapping.

For a specific isotope per species, or to record extra metadata, call
[`build_alpha_beta_table()`][pyradbio.tables.alpha_beta.build_alpha_beta_table] directly with
a list of [`Ion`][pyradbio.core.radiation.Ion] objects. Their own energies are ignored; the
energy grid is swept.

```python
from pyradbio.models.mkm import MKMModel
from pyradbio.tables.io import read_csv, write_csv

model = MKMModel()  # native z* backend, default HSG preset
table = model.alpha_beta_table(ions=[1, 6], energies_MeV_u=[10.0, 100.0])

for row in table.rows:
    print(row.Z, row.A, row.energy_MeV_u, row.alpha, row.beta)

write_csv(table, "mkm_table.csv")
again = read_csv("mkm_table.csv")
assert again.rows == table.rows  # floats round-trip exactly
```

### The header records the resolved configuration

`build_alpha_beta_table()` stores the model's **resolved** configuration under the
`model_config` metadata key, not the arguments you happened to pass: every field you left
alone reports its default, and the stopping-power source and its resolved settings are
included. This matters because several models have protocol choices that change the numbers
without changing the call (LEM IV has about a dozen), and a table that does not name them
cannot be compared with another one.

```python
config = table.metadata["model_config"]
print(config["model"])  # 'MKMModel'
print(sorted(config))  # the resolved fields
```

[`model_provenance()`][pyradbio.tables.alpha_beta.model_provenance] returns the same mapping
for any model without building a table. Pass `record_model_config=False` to
`build_alpha_beta_table()` to leave it out.

## Reading and writing

[`pyradbio.tables.io`][pyradbio.tables.io] handles three formats.

| Format | Functions | Needs |
| --- | --- | --- |
| CSV | `write_csv` / `read_csv` | core install (standard library only) |
| Parquet | `write_parquet` / `read_parquet` | the `tables` extra (`pip install "pyradbio[tables]"`) |
| pyMKM `.txt` | `write_pymkm_txt` / `read_pymkm_txt` | core install |

The CSV starts with one comment line, `# pyradbio-meta: {...}`, holding the metadata as JSON,
then a header row `Z,A,energy_MeV_u,alpha,beta` and one row per point. `read_csv` accepts a
file with or without the comment line. Parquet keeps the same JSON in the file's schema
metadata under `pyradbio-meta`, so the provenance travels with either format.

### z\* tables and pyMKM interop

The MKM family also tabulates the saturation-corrected dose-mean specific energy
\( z^*_{1D} \). [`build_zstar_table()`][pyradbio.tables.zstar.build_zstar_table] builds a
[`ZStarTable`][pyradbio.tables.zstar.ZStarTable] through the model's backend, and
`write_pymkm_txt` writes it in pyMKM's classic `.txt` layout (a `CellType` line, header
parameters, then `energy z*_1D` columns grouped by `Fragment`), so a table made here can be
used wherever a pyMKM table is expected. `read_pymkm_txt` reads one back, including the
four-column stochastic layout.

```python
from pyradbio.tables.io import read_pymkm_txt, write_pymkm_txt
from pyradbio.tables.zstar import build_zstar_table

zstar = build_zstar_table(
    ions=[6],
    energies_MeV_u=[10.0, 100.0],
    domain_radius_um=0.44,  # illustrative geometry and photon parameters
    nucleus_radius_um=5.0,
    alpha0_Gy=0.1,
    beta0_Gy2=0.05,
)
write_pymkm_txt(zstar, "zstar_carbon.txt")
print(len(read_pymkm_txt("zstar_carbon.txt").rows))
```

The `.txt` format prints five significant figures, so values read back agree with the
originals to that precision, not exactly. Use the CSV route when you need an exact round trip.
For the MKM-specific columns (`z*_1D` and LET next to `alpha` and `beta`) use
[`MKMModel.alpha_beta_z_star_table()`][pyradbio.models.mkm.model.MKMModel.alpha_beta_z_star_table].

## The `pyradbio` command

The installation provides a `pyradbio` command with three subcommands. `pyradbio --version`
prints the version, and `pyradbio <command> --help` lists every option.

```bash
pyradbio --help
pyradbio table --help
```

### `pyradbio survival`

The LQ surviving fraction for one dose. All three options are required.

```bash
pyradbio survival --alpha 0.3 --beta 0.03 --dose 2.0
```

It prints one line, `S(2 Gy) = ...`, for \( S = \exp(-\alpha D - \beta D^2) \).

### `pyradbio table`

Builds an α/β table for LEM I-IV or the MKM and writes it as CSV, with the resolved model
configuration in the header (see above).

```bash
# LEM I, carbon and protons, on the library default stopping power
pyradbio table --model lem1 --alpha-x 0.18 --beta-x 0.028 --dt 30 --rn 5 \
    --ions 1,6 --energies 10,100 --out lem1_table.csv

# MKM on the native backend
pyradbio table --model mkm --alpha0 0.1 --beta0 0.05 --rd 0.44 --rn 5 \
    --ions 1,6 --energies 10,100 --out mkm_table.csv
```

The values here are illustrative inputs for the sake of the example, not recommended
parameters. Options:

| Option | Meaning |
| --- | --- |
| `--model {lem1,lem2,lem3,lem4,mkm}` | model (required) |
| `--rn RN` | nucleus radius in µm (required) |
| `--alpha-x`, `--beta-x`, `--dt` | photon reference \( \alpha_x \) in Gy⁻¹, \( \beta_x \) in Gy⁻², and transition dose \( D_t \) in Gy; required for `lem*` |
| `--alpha0`, `--beta0`, `--rd` | MKM \( \alpha_0 \), \( \beta_0 \) and domain radius in µm; required for `mkm` |
| `--ions Z,Z,...` | comma-separated atomic numbers (required) |
| `--mass-numbers A,A,...` | mass numbers matching `--ions`; default is the most abundant isotope |
| `--energies E,E,...` | comma-separated energies in MeV/u (required) |
| `--out FILE` | output CSV path (required) |
| `--effective-size-factor` | scale applied to the nucleus radius (default 1.0) |
| `--stopping-power` | LET source, see below |
| `--engine {auto,numpy,numba}` | numerical backend for the LEM integrals (default `auto`: numba when installed) |

The LEM behaviour axes, which apply to the `lem*` models and are rejected by a LEM class that
lacks them:

| Option | Choices |
| --- | --- |
| `--beta-model` | `slope_matching` (default), `rapid_lem`, `scholz1997_stochastic`, `scholz1997_two_track`, `scholz1997_montecarlo`, `intertrack_mc`, `local_dose`, `discrete_dsb` |
| `--initial-slope-model` | `cross_section` (default), `lethal_central`, `lethal_central_saturating`, `lethal_full`, `full_simulation` |
| `--survival-convention` | `mean_survival` (default), `mean_effect` (LEM IV) |
| `--cluster-index-model` | `mean_field` (default), `discrete`, `discrete_mc`, `trip98_composite` (LEM IV) |
| `--cluster-index-cap` | clip the LEM IV cluster index at this value |
| `--local-dose-overkill` | LEM IV single-track overkill correction (off by default) |

Which choices are valid for which LEM version is explained on the
[Local effect model](models/lem.md) page. Some routes are slow, and the Monte-Carlo ones
(`scholz1997_montecarlo`, `intertrack_mc`) are far slower than the default.

`--stopping-power` selects the LET source and takes `default` (the library default, see
[Physics building blocks](physics.md)), `bethe` (analytic Bethe alone), `icru90-procedure`,
`icru2016`, `icru2009`, `icru2005` (the ICRU 90, 73 and 49 tables), `xstar`, `ns1970`
(Northcliffe and Schilling 1970) and `libdedx` (needs the separately installed, GPL libdEdx
package). Leave it out and each model keeps its own default, which is the library default.
The table sources cover Z 1-18 and raise outside their range rather than extrapolate.

### `pyradbio benchmark`

Scores a model against a measured-survival dataset. It is covered on
[Benchmarking against data](benchmarking.md).

## Where the numbers come from

Every CSV from `pyradbio table` carries the resolved model, stopping-power and cell settings
in its first line. To see what a table was computed under:

```py
from pyradbio.tables.io import read_csv

table = read_csv("lem1_table.csv")
print(table.metadata["model"])  # the CLI's own label, e.g. 'lem1'
print(table.metadata["model_config"]["stopping_power"])
```

Keep the CSV with its header intact when you pass a table on; the header is the record of
which protocol produced it.
