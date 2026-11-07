# PIDE 3.4 schema (loader contract)

Column contract of the PIDE workbook as read by `pyradbio.benchmark.pide`. Header strings
verified against a registered `PIDE3.4.xlsx`. PIDE (Particle Irradiation Data Ensemble, GSI;
Friedrich et al., *J. Radiat. Res.* 2013, doi:10.1093/jrr/rrs114; 2021 update
doi:10.1093/jrr/rrab034) is distributed only after registration and **may not be
redistributed**: no PIDE data, and nothing derived from it, is committed. The loader reads the
git-ignored `data/pide/` folder (or `PYRADBIO_PIDE_DIR`, or an explicit `--pide` path).

The workbook has one sheet, `PIDEMasterfile`, with these 24 columns, in order:

```
#ExpID, #Publication, PublicationName, #IonExp, Cells, CellClass, CellOrigin,
CellCycle, DNAcontent, PhotonRadiation, #PhotonExp, Ion, Charge,
IrradiationConditions, LET, Energy, ax_paper, bx_paper, ai_paper, bi_paper,
ax_fit, bx_fit, ai_fit, bi_fit
```

Conventions: `N/A` marks a missing value. `Ion` is written `12C`, `4He`, `1H`.
`PublicationName` is first-author surname + two-digit year (+ a disambiguating letter, e.g.
`cox77n`). `IrradiationConditions` is `m` (monoenergetic) or `s` (SOBP); `CellClass` `n`/`t`;
`CellOrigin` `h`/`r`. The `_paper` quartet is as published, the `_fit` quartet is PIDE's refit
(missing = `N/A`); the loader selects the flavour via `variant` and falls back per field.

The loader is **header-tolerant** (aliases below), so it also reads the synthetic CI fixture
`tests/benchmark/fixtures/synthetic_pide.xlsx` (built by `make_synthetic_pide.py` from the
canonical names in the table below) and other exports.

## Records

A `PideEntry` subclasses the source-agnostic `pyradbio.benchmark.dataset.ReferenceRecord`
(`source = "pide:<PublicationName>"`); `load_pide_dataset()` wraps the rows in a
`ReferenceDataset` carrying PIDE's caveats for the report. The open-access RadPhysBio
database has its own loader (`RadPhysBio.md`).

For cross-checks use `load_pide_observations()`: its `PideObservation` records keep absent LET
and individual LQ parameters as `None`, so a missing beta does not discard an available alpha.
Header aliases and the fit/paper fallback are shared with `load_pide()`, whose complete-record
contract is unchanged. Neither loader writes data or reports.

## LQ convention

PIDE stores `-ln S = alpha d + beta d^2`, the pyRadBio convention, for both the ion exposure
and its matched photon reference. No unit conversion: alpha in Gy^-1, beta in Gy^-2.

## Logical fields

| Field (`PideEntry`) | Unit | Canonical column | Accepted aliases (normalised¹) |
|---|---|---|---|
| `cell_line` | none | `Cell` | `cell`, `cellline`, `celltype`, `name` |
| `Z` | none | `Charge` | `charge`, `z`, `atomicnumber` (or parsed from `Ion`) |
| `A` | none | (from `Ion`) | mass number parsed from the ion token; `None` if absent |
| `energy_MeV_u` | MeV/u | `Energy` | `energy`, `e`, `emevu`, `specificenergy` |
| `let_keV_um` | keV/um | `LET` | `let`, `letkevum`, `letd`, `dose_averaged_let` |
| `alpha_ion` | Gy^-1 | `alpha_I` | `alphai`, `alphal`, `alphaion`, `alpha` |
| `beta_ion` | Gy^-2 | `beta_I` | `betai`, `betal`, `betaion`, `beta` |
| `alpha_x` | Gy^-1 | `alpha_X` | `alphax`, `alpharef`, `alphaphoton` |
| `beta_x` | Gy^-2 | `beta_X` | `betax`, `betaref`, `betaphoton` |
| `photon_radiation` | none | `PhotonRadiation` | `photon`, `referenceradiation`, `reference` |

¹ Normalised = lower-cased with every non-alphanumeric character removed (`alpha_X`,
`alpha X`, `Alpha_X`, `alphaX` all match `alphax`).

The `Ion` column (`C-12`, `12C` or bare `C`) recovers `Z`/`A` when no explicit charge column
is present; a numeric charge column always wins for `Z`.

## Rows the loader skips

* Rows missing any *required* numeric field (`LET`, `alpha_I`, `beta_I`, `alpha_X`,
  `beta_X`). `energy_MeV_u` is optional (`None` when blank).
* Rows whose ion cannot be resolved to a `Z`.

Skips are silent by default; `strict=True` raises.

## Columns not consumed

The workbook carries more (first author / year / DOI, cell origin, tumour vs normal,
cell-cycle phase, oxygenation, dose rate / fractionation, RBE at fixed survival, fit-quality
flags). `PideEntry` starts with the minimal set needed for the alpha benchmark; extend it by
adding fields and aliases here in step.

`photon_radiation` is stored as the raw reference-quality string (e.g. `200kVp`, `250kVp`,
`60Co`, `137Cs`, `6MV`; `N/A` becomes `""`). The loader does not interpret it: mapping a
quality to a reference dose-mean lineal energy y_D,ref (Parisi 2022: ~2.3 keV/um for
`60Co`/MV, ~4.3 for kVp X-rays, feeding the MCF-MKM alpha_0) is a model-side concern.
