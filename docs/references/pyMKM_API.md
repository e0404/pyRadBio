# pyMKM public API (verified against the installed package)

Source: BeppeMagro/pymkm (Magro et al., *Computation* 2025,
doi:10.3390/computation13110264, MIT). Docs: https://beppemagro.github.io/pymkm/

pyMKM is an optional cross-check (`mkm` extra), imported only by
`pyradbio/models/mkm/pymkm_backend.py`. Re-inspect with
`python -c "import pymkm; help(pymkm.mktable)"` after an upgrade; do not rely on
memory.

## Stopping-power tables

```python
from pymkm.io.table_set import StoppingPowerTableSet
sp = StoppingPowerTableSet.from_default_source("mstar_3_12").filter_by_ions(["6"])
```

- `from_default_source(source: str) -> StoppingPowerTableSet`. Bundled sources:
  `"fluka_2020_0"`, `"geant4_11_3_0"`, `"mstar_3_12"`.
- `filter_by_ions(ion_inputs: list[str]) -> StoppingPowerTableSet`: atomic
  number as string (`"6"`), symbols/names also accepted. Ints work in practice
  but the signature is `list[str]`.
- Also: `save`/`load`, `to_json`/`from_json`, `to_dict`/`from_dict`,
  `from_directory`.

## MK table parameters & computation

```python
from pymkm.mktable import MKTable, MKTableParameters
params = MKTableParameters(domain_radius=0.32, nucleus_radius=3.9, beta0=0.0615)
table = MKTable(parameters=params, sp_table_set=sp)
table.compute(ions=[6], energy=[11.0, 100.0], parallel=False)
```

- `MKTableParameters(domain_radius: float [um], nucleus_radius: float [um],
  z0: float|None = None, beta0: float|None = None,
  model_name: str = "Kiefer-Chatterjee", core_radius_type: str = "energy-dependent",
  ..., use_stochastic_model: bool = False, ...)`. `z0` [Gy] is derived from the
  geometry + `beta0` if omitted (Inaniwa 2010 eq. 5).
- `MKTable(parameters, sp_table_set=None)`.
- `MKTable.compute(ions=None, energy=None, parallel=True, number_of_workers=None,
  integration_method="trapz")`; `ions` as `list[int|str]`, `energy` in MeV/u
  (scalar or list). **Use `parallel=False`** for deterministic results and to
  avoid Windows multiprocessing-spawn issues.
- pyMKM prints a config summary (Unicode glyphs) and a tqdm bar; capture
  stdout/stderr to silence it and avoid a Windows `cp1252` `UnicodeEncodeError`.

## Reading results

`MKTable.table` is a `dict` keyed by **element name** (e.g. `"Carbon"`). Each
value:

- `["params"]`: the resolved parameters (incl. derived `z0`).
- `["stopping_power_info"]`: `{"ion_symbol", "mass_number", "atomic_number",
  "source_program", "ionization_potential", ...}`. Match on `atomic_number`.
- `["data"]`: a `pandas.DataFrame`, one row per energy:
  - `energy` [MeV/u]
  - `let`: unrestricted LET from the SP table. **pyMKM tabulates it in MeV/cm**
    (= 10 × keV/um); the adapter converts with `MEV_PER_CM_TO_KEV_PER_UM`
    (`PyMKMStoppingPower.let_keV_um`). `ParticleTrack` likewise takes MeV/cm.
  - `z_bar_star_domain` [Gy]: **this is `z*_1D`**, feeding
    `alpha_MKM = alpha0 + beta*z*_1D`.
  - `z_bar_domain`, `z_bar_nucleus` (None in the classic model run).

Sanity: with `(r_d=0.32, R_n=3.9, beta0=0.0615)` carbon `z_bar_star_domain`
decreases with energy (26.2 @ 11 MeV/u, 7.36 @ 100, 2.78 @ 400), the Inaniwa
2010 fig. 5 shape.

## `.txt` z* table format

`MKTable.write_txt(params={"CellType":..., "Alpha_0":..., "Beta":...}, filename=...)`
(source `pymkm/mktable/core.py::write_txt`, model="classic"). pyRadBio
round-trips it in `tables/io.py` (`write_pymkm_txt`/`read_pymkm_txt`) via
`tables/zstar.ZStarTable`: per-fragment `energy` vs `z_bar_star_domain`.

```
CellType  <CellType>            # note: TWO spaces after the key, then a blank line

Parameter Alpha_0 <a0:.4f>
Parameter Beta <beta:.4f>
Parameter DomainRadius <r_d:.4f>
Parameter NucleusRadius <R_n:.4f>
                                # blank line
Fragment <ElementSymbol>        # e.g. "C" (symbol only, no mass number)
<energy:.5e> <z_bar_star_domain:.5e>
<energy:.5e> <z_bar_star_domain:.5e>
                                # blank line after each fragment block
Fragment <ElementSymbol>
...
```

Header parameters `%.4f`, data `%.5e`, single-space separated; energy in MeV/u,
z* in Gy. Round-trips are exact only for values representable at that
precision. The **stochastic (SMK)** variant writes `Parameter Alpha_ref/Beta_ref/
scale_factor/Alpha0/Beta0` and four columns
`energy z_bar_domain z_bar_star_domain z_bar_nucleus`; `read_pymkm_txt` keeps the
`z_bar_star_domain` column.

## Other modules used as oracles

- `biology/oxygen_effect.py`: OSMK algebra; matches both papers exactly.
- `mktable/compute.py`, `sftable/compute.py`: OSMK-2023 `r_d'`/`z_0'` are
  **rounded** before rescoring; see `OSMK_Inaniwa2021_2023.md` note 5.
- `sftable/compute.py` `model="stochastic"`: SMK oracle (`rel=1e-12`).

Not documented here: the SMK/OSMK stochastic entry points of `MKTable`
(`use_stochastic_model`, `apply_oxygen_effect`, `pO2`); pyRadBio writes only the
classic `.txt`.
