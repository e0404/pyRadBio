# pyRadBio

Radiobiological modelling for ion-beam therapy, centred on the
linear-quadratic (LQ) model

```
-ln S(D) = α D + β D²
```

pyRadBio computes ion- and energy-resolved **α/β tables**, **z\* tables**,
**survival curves**, **RBE**, and the **dose required for a target survival
level**, for single ions and mixed radiation fields. It is built for
research use (experiment planning, model comparison, table generation for
treatment-planning research), MIT-licensed.

## Model coverage

All models are implemented in pyRadBio and map a radiation field to the same
LQ parameters `(alpha, beta)`.

| Family | Models | Package |
| --- | --- | --- |
| LEM | LEM I, II, III, IV (+ GLOBLE) | `pyradbio.models.lem` |
| MKM | MKM, modified MKM, MCF MKM, SMK, OSMK 2021 / 2023, IMK | `pyradbio.models.mkm` |
| Phenomenological | 13 LET-based proton RBE fits (Wilkens & Oelfke, Carabe, Frese, Wedenberg, McNamara, Mairani, ...) | `pyradbio.models.phenomenological` |
| RMF | Repair-Misrepair-Fixation with MCDS DSB yields | `pyradbio.models.rmf` |
| Katz | Cellular track-structure theory (ion kill + gamma kill) | `pyradbio.models.katz` |
| NanOx | Calibration and assembly layer; the ion route raises (its nanodosimetric Monte Carlo is not public) | `pyradbio.models.nanox` |

`MKMModel` computes `z*` through a pluggable backend: pyRadBio's native
engine (the default) or, as an independent cross-check, the external,
MIT-licensed [`pyMKM`](https://github.com/BeppeMagro/pymkm) (optional
`pyradbio[mkm]` extra). Every model runs on pyRadBio's own track structure and
microdosimetry.

By default LET comes from `pyradbio.track.stopping_power.default_stopping_power`:
ICRU 90's procedure for Z 1-18 (within ~0.3% of ICRU 90 for p, alpha and C), Hubert,
Bimbot & Gauvin's 1990 alpha scaling for Z 19-92 from 2.5 MeV/u, and an analytic
Bethe formula anchored to them elsewhere (unvalidated there). The water tables in
`pyradbio.track.stopping` (PSTAR/ASTAR/MSTAR, ICRU 49/73/90, Bethe with switchable
corrections), Northcliffe & Schilling 1970, user SRIM tables, libdEdx, or pyMKM's
tables are explicit opt-ins passed to a model's `stopping_power`; see
`docs/references/StoppingPower.md`.

A shared, model-agnostic layer provides LQ survival/RBE, mixed-field
dose-averaging, photon-reference and cell/target descriptions, and a benchmark
harness against measured survival data. The full inventory, with every
reference, is in `ROADMAP.md`.

## Key references

- **LEM I:** Scholz & Kraft 1996, *Adv. Space Res.* 18, 5–14 — doi:10.1016/0273-1177(95)00784-C
- **LEM II:** Elsässer & Scholz 2007, *Radiat. Res.* 167, 319–329 — doi:10.1667/RR0467.1
- **LEM III:** Elsässer et al. 2008, *Int. J. Radiat. Oncol. Biol. Phys.* 71, 866–872 — doi:10.1016/j.ijrobp.2008.02.037
- **LEM IV:** Elsässer et al. 2010, *Int. J. Radiat. Oncol. Biol. Phys.* 78, 1177–1183 — doi:10.1016/j.ijrobp.2010.05.014; Friedrich et al. 2012, *Int. J. Radiat. Biol.* 88, 103–107 — doi:10.3109/09553002.2011.611213
- **MKM (amorphous track):** Kase et al. 2008, *Phys. Med. Biol.* 53, 37–59 — doi:10.1088/0031-9155/53/1/003
- **modified MKM:** Inaniwa et al. 2010, *Phys. Med. Biol.* 55, 6721 — doi:10.1088/0031-9155/55/22/008
- **MCF MKM:** Parisi et al. 2022 — doi:10.1088/1361-6560/ac7375
- **SMK / OSMK:** Inaniwa & Kanematsu 2018 — doi:10.1088/1361-6560/aabede; Inaniwa et al. 2021 — doi:10.1088/1361-6560/ac29cc; Inaniwa & Kanematsu 2023 — doi:10.1093/jrr/rrad049
- **IMK:** Matsuya et al. 2025, *Med. Phys.* — doi:10.1002/mp.70040
- **Phenomenological RBE:** Rørvik et al. 2018 — doi:10.1088/1361-6560/aad9db (review; each fit's own paper is cited in its module)
- **RMF:** Carlson et al. 2008, *Radiat. Res.* — doi:10.1667/RR1046.1
- **Katz:** Katz et al. 1971, *Radiat. Res.* — doi:10.2307/3573247
- **NanOx:** Cunha et al. 2017 — doi:10.1088/1361-6560/aa54c9; Monini et al. 2020 — doi:10.1667/RR15463.1
- **pyMKM:** Magro et al. 2025, *Computation* 13, 264 — doi:10.3390/computation13110264

## Install

```bash
pip install -e ".[all]"      # dev: core + tables + benchmark + fast + plot + dev tools
pip install pyradbio         # (once published) core only
pip install "pyradbio[mkm]"  # optional: pyMKM, for cross-checking the MKM family
```

## Quick start

Models live in `pyradbio.models.*` and are imported by name, so optional
dependencies load only when you ask for them. The LQ utilities
and the radiation, photon and cell descriptions are top-level.

```python
from pyradbio import CellTarget, Ion, LQParameters, PhotonParameters
from pyradbio import dose_for_survival, rbe, survival
from pyradbio.models.lem.lem1 import LEM1

# V79 photon reference (Scholz & Kraft 1996; Elsässer & Scholz 2007)
photon = PhotonParameters(alpha_x=0.18, beta_x=0.028, D_t=30.0)
lem = LEM1(photon=photon, cell=CellTarget(nucleus_radius_um=5.0))

ab = lem.alpha_beta(Ion(Z=6, A=12, energy_MeV_u=11.0))  # LQParameters(alpha, beta)
s = survival(2.0, ab.alpha, ab.beta)  # surviving fraction at 2 Gy
d10 = dose_for_survival(0.10, ab.alpha, ab.beta)  # dose for 10% survival [Gy]
rbe10 = rbe(0.10, ab.alpha, ab.beta, reference=LQParameters(photon.alpha_x, photon.beta_x))
```

```python
# MKM on the native z* backend (no extra needed); pyMKM is an opt-in cross-check
from pyradbio import Ion
from pyradbio.models.mkm import MKMModel

mkm = MKMModel()  # Inaniwa 2010 HSG defaults, native
ab = mkm.alpha_beta(Ion(Z=6, A=12, energy_MeV_u=100.0))

# from pyradbio.models.mkm.pymkm_backend import PyMKMBackend   # needs pyradbio[mkm]
# check = MKMModel(backend=PyMKMBackend())
```

```python
# LET-based phenomenological RBE (here McNamara et al. 2015)
from pyradbio import PhotonParameters
from pyradbio.models.phenomenological import McNamara

mcn = McNamara(photon=PhotonParameters(alpha_x=0.18, beta_x=0.028))
ab = mcn.alpha_beta_from_let(5.0)  # dose-averaged LET [keV/um]
rbe10 = mcn.rbe_at_survival(0.10, let_keV_um=5.0)
```

The `examples/` folder has runnable demonstrations, one or more per model
family; `validation/` reconstructs published figures and compares against other
codes and external data, reading the Monte-Carlo checkpoints that `scripts/`
generates.

## Benchmark against measured survival data

`pyradbio benchmark` scores a model's initial-slope `alpha` (and `RBE_alpha`)
against a database of measured LQ parameters. Three datasets are built in and
the loader registry is user-extensible (`pyradbio.benchmark.dataset`):

- **PIDE** (default; tested against 3.4) is **not bundled** (registration-gated,
  non-redistributable). Obtain it from the
  [GSI PIDE project page](https://www.gsi.de/work/forschung/biophysik/forschungsfelder/radiobiological_modelling/pide_project)
  and drop the export into the git-ignored `data/pide/` folder (or set
  `PYRADBIO_PIDE_DIR`).
- **RadPhysBio** (Zanni et al. 2024, CC BY 4.0) is open access: export the CSV
  from <https://radbiodb.physics.ntua.gr/radphysbio/> into the folder named by
  `PYRADBIO_RADPHYSBIO_DIR`. Human cell lines only, and largely digitized; every
  report against it says so, together with how many ion rows could be paired
  with a same-publication photon reference. Its nominal beam energies restrict
  these benchmarks to LET-based models; energy-based routes are rejected.
- **csv**: your own file with columns `cell, ion, energy, let, alpha_ion,
  beta_ion, alpha_x, beta_x` (header-tolerant). `energy` may be left empty for
  LET-based models (a given energy is still filtered like any other);
  energy-based models require a supported target energy.

```bash
pyradbio benchmark --model lem4 --out benchmark_results/                    # PIDE from data/pide/
pyradbio benchmark --dataset radphysbio --model mcnamara --out rpb_results/
pyradbio benchmark --dataset csv --path mine.csv --model wedenberg --out mine_results/
pyradbio benchmark --pide /path/to/PIDE3.4.xlsx --model lem4 --out benchmark_results/   # explicit file
```

As an independent check, the LET-based models are validated against RadPhysBio with
the second command above. RadPhysBio is by Zanni et al. 2024 (doi:10.3390/ijms25094729),
licensed CC BY 4.0, and credited in `THIRD_PARTY_NOTICES.md`.

`--model` covers LEM I-IV, the MKM family (`mkm`, `mmkm`, `smk`, `mcf`,
`osmk2021`, `osmk2023`, `imk`) at their published presets, RMF, four Katz
parameter sets, and the thirteen phenomenological RBE fits (`wilkens`, `carabe`,
`frese`, `wedenberg`, `mcnamara`, `mairani`, `tilly2`, `tilly10`, `chen`,
`jones`, `roru`, `peeler`, `unkelbach`). Each report states what must be read
alongside its numbers -- chief among them that this scores a model against
**measured data**, not against a paper's own model line, and that the proton
fits were themselves calibrated on much of the PIDE corpus.

## Status

Research software under active development; the public API may still change.
Each model is checked against its source papers by golden tests pinned to
published values (`pytest -m golden`), and `docs/references/` records every
reconstruction, including where and why a result departs from the paper. See
`ROADMAP.md` for planned and open work and `AGENTS.md` for the
contributor/agent contract.

## License

MIT. See `LICENSE`. One exception: the Python port of MSTAR 3.12
(`pyradbio/track/stopping/mstar/`) is under its authors' freeware notice; see
`THIRD_PARTY_NOTICES.md`.
