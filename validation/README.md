# pyRadBio validation scripts

Paper reconstructions, cross-implementation checks and comparisons against
external data. Where `examples/` *demonstrates* a capability in seconds, a
validation script *scores* a model against a published model line, another
code or a dataset, and may take minutes.

Each script is a plain `.py` file in [jupytext](https://jupytext.readthedocs.io)
**percent** format (cells delimited by `# %%`), so it also opens as a notebook.
Run one from the repository root:

```bash
pip install -e ".[plot]"          # plus any extra the table below lists
python validation/<script>.py      # full mode: the only mode whose numbers are quoted
PYRADBIO_QUICK=1 python validation/<script>.py   # quick mode: same figures, cheap settings
```

Figures go to `validation/figures/` (git-ignored; override the folder with
`PYRADBIO_FIGURES_DIR`).

## Rules for this folder

| | validation/ |
|---|---|
| Format | jupytext percent; no `main()` |
| Size | no cap; keep the narrative condensed (long-form findings belong in `docs/references/`) |
| Runtime | minutes in full mode, documented in the table below |
| Quick mode | `PYRADBIO_QUICK=1`, read through `_common.quick_mode()`; every cost knob is `pick(full, quick)`; prints one banner line and writes the same PNG names |
| Data | bundled data, external datasets through `pyradbio.datasets` (gated with `_common.require_data`), and the tracked checkpoints in `validation/data/` |
| Imports | numpy, scipy, matplotlib, pyradbio; optional extras only through `_common.require_extra` |
| Figures | `figures_dir(globals().get("__file__"))` + `save_figure(...)` from `pyradbio.plotting` |
| Lint | RUF001-003 ignored (Greek letters in labels are intended) |

`_common.py` holds the script plumbing (quick mode, the missing-data and
missing-extra exits, two plot idioms). It is imported as a sibling module and is
never part of the `pyradbio` wheel. A script that lacks its data or extra stops
with a one-line message, which the smoke test turns into a skip.

Every script here is smoke-tested by `tests/validation/test_validation_smoke.py`
(`@slow`, quick mode): `pytest -m slow tests/validation`.

## Scripts

| script | paper / DOI | data | extras | full / quick runtime | checkpoint read |
|---|---|---|---|---|---|
| `scholz_kraft1996_lem1.py` | Scholz & Kraft 1996, doi:10.1016/0273-1177(95)00784-C, Figs. 2-4 | bundled digitized curves (`benchmark.scholz_kraft1996`) | plot (`LEM1_STOPPING=pymkm` needs `mkm`, `libdedx` needs libdedx) | ~8 min / ~45 s | `sk1996_mc_observables.csv` |
| `scholz1997_lem1.py` | Scholz et al. 1997, doi:10.1007/s004110050055, Figs. 1-5 | bundled digitized curves (`benchmark.scholz1997`) | plot (same `LEM1_STOPPING` options) | ~75 s / ~30 s | `mc_results.csv`, `sk1997_z_slope.csv` |
| `elsasser2007_lem2.py` | Elsasser & Scholz 2007, doi:10.1667/RR0467.1, Figs. 2-4, 6-8, 10 and 12 | bundled digitized curves (`benchmark.elsasser2007`) | plot | ~2 min / ~50 s | none |
| `elsasser2008_lem3.py` | Elsasser, Kraemer & Scholz 2008, doi:10.1016/j.ijrobp.2008.02.037, Tables 1-2 | bundled tables (`benchmark.elsasser2008`) | plot | ~20 s / ~12 s | none |
| `lem4_elsasser2010_friedrich2012.py` | Elsasser et al. 2010, doi:10.1016/j.ijrobp.2010.05.014, Fig. 4; Friedrich et al. 2012, doi:10.3109/09553002.2011.611213, Figs. 3-4 | bundled digitizations | plot (`fast` recommended) | ~4 min / ~10 s | `lem4_friedrich2012_mc.csv` |
| `trip98_lem1_rbe.py` | TRiP98 LEM I `.rbe` tables on TRiP's own FLUKA dE/dx | `PYRADBIO_TRIP98_DIR` | plot | ~25 s / ~12 s | none |
| `trip98_lem4_rbe.py` | TRiP98 LEM IV `.rbe` tables; Elsasser 2010; Pfuhl 2020/2021; Friedrich 2012 | `PYRADBIO_TRIP98_DIR` | plot | ~2.7 h cold, seconds warm / ~40 s | its local route cache `validation/figures/lem4_trip98_routes_*.npz` (git-ignored) |
| `survival_code_lem.py` | the INFN Survival code, Manganaro et al. 2018, doi:10.1088/1361-6560/aab697: LEM I-III on Survival's own SRIM LET | `PYRADBIO_SURVIVAL_DIR`, `PYRADBIO_SURVIVAL_SRC_DIR` | plot | ~5 min / ~45 s | `survival_our_mc.json` |
| `survival_code_mmkm.py` | Manganaro 2018: Survival's rapid mMKM (Inaniwa 2010) and LEM I tables | `PYRADBIO_SURVIVAL_DIR` | plot (`mkm` only with `PYRADBIO_USE_PYMKM=1`) | ~8 s / ~5 s | none |
| `smk_inaniwa2018.py` | Inaniwa & Kanematsu 2018, doi:10.1088/1361-6560/aabede, Figs. 2 and 4 (Fig. 3 numbers in stdout) | bundled vector-path curves and survival CSV (`benchmark.inaniwa2018`) | plot | ~1-2 min / ~20-30 s | none |
| `smk_model_paths.py` | Inaniwa & Kanematsu 2018; Sato & Furusawa 2012, doi:10.1667/RR2842.1: what each SMK path (microdosimetry route, `domain_treatment`, `nucleus_treatment`) costs | none | plot | ~1-2 min / ~20 s | none |
| `osmk_inaniwa2021.py` | Inaniwa et al. 2021, doi:10.1088/1361-6560/ac29cc, Figs. 5-7 (the 2023 refit, doi:10.1093/jrr/rrad049, for context) | bundled digitization (`benchmark.inaniwa2021`) | plot | ~1-2 min / ~20-40 s | none |
| `imk_matsuya2025.py` | Matsuya et al. 2025, doi:10.1002/mp.70040, Figs. 2, 4 and 7 through four microdosimetry routes | bundled digitizations | plot | ~2.5 min / ~45-75 s | none |
| `pymkm_crosscheck.py` | pyMKM (Magro 2025, doi:10.3390/computation13110264): Kiefer-Chatterjee D(r) (Chatterjee & Schaefer 1976, doi:10.1007/BF01330766) and water LET | bundled | plot, `mkm` | ~5 s / ~5 s | none; its `water_stopping_power.png` is the pyMKM overlay, not the example of that name |
| `lineal_energy_hartzell2025.py` | Hartzell 2025, doi:10.1088/1361-6560/adcfec: analytic and amorphous spectra vs AMF / TOPAS-nBio, 1H-20Ne | bundled digitized curves (`benchmark.hartzell2025`) | plot | ~20 s / ~13 s | none |
| `lineal_energy_decunha.py` | DeCunha 2025, doi:10.1002/mp.17561: proton spectra vs the Geant4-DNA library | `PYRADBIO_LINEAL_ENERGY_DECUNHA_DIR` (`GEOMETRY = "5um_linear"` in the script) | plot | ~9 s / ~5 s | none |
| `pide_pfuhl2022.py` | Pfuhl et al. 2022, doi:10.1002/mp.15343: a model against PIDE, Table 2 style per-ion summary | `PYRADBIO_PIDE_DIR` | plot (`tables` for an `.xlsx` export) | ~1 min / ~10-15 s | none |
| `nanox_alcocer_avila2023.py` | Alcocer-Avila et al. 2023, doi:10.3389/fphy.2023.1011062, Table 5 (p. 12) | none | plot | ~2 s / ~2 s | none |

## Checkpoints

`validation/data/` holds **our own** offline Monte-Carlo outputs that the
scripts read, so a fresh clone can draw the figures without re-running hours of
simulation. They are tracked in git (never third-party data, never PIDE-derived
numbers) and regenerated only by the generators in [`scripts/`](../scripts/README.md).

| file | written by (`scripts/` command) | read by |
|---|---|---|
| `mc_results.csv` | `lem1_mc_reference.py alpha-beta` | `scholz1997_lem1.py` |
| `sk1997_z_slope.csv` | `lem1_mc_reference.py z-slope` | `scholz1997_lem1.py` |
| `sk1996_mc_observables.csv` | `lem1_mc_reference.py sk1996-final-slope` | `scholz_kraft1996_lem1.py` |
| `lem4_friedrich2012_mc.csv` | `lem4_friedrich2012_mc.py` | `lem4_elsasser2010_friedrich2012.py` |
| `survival_our_mc.json` | `lem_mc_vs_survival.py` | `survival_code_lem.py` |

Each file carries a provenance header (generator, stopping power, ...) with a
**numeric fingerprint**: a few points (one to three) recomputed at a small sample size
with `engine="numpy"` and fixed seeds. A fast test recomputes them
(`tests/benchmark/test_lem1_mc_fingerprints.py`,
`tests/models/lem/test_lem4_mc_fingerprint.py`,
`tests/benchmark/test_survival_cache_fingerprint.py`) and fails with
"regenerate with scripts/<generator>.py" when the library's numbers move.
