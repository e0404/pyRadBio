# pyRadBio

Radiobiological modelling for ion-beam therapy, centred on the linear-quadratic (LQ) model

\[
-\ln S(D) = \alpha D + \beta D^2 .
\]

pyRadBio computes ion- and energy-resolved **\(\alpha/\beta\) tables**, **\(z^*\) tables**,
**survival curves**, **RBE**, and the **dose required for a target survival level**, for single
ions and mixed radiation fields. It is built for research use (experiment planning, model
comparison, table generation for treatment-planning research) and is MIT-licensed.

Every model maps a radiation field and a cell description to the same pair of LQ parameters
\((\alpha, \beta)\) behind one interface, so models can be swapped, compared and benchmarked
without changing the surrounding code. Survival, iso-survival RBE and mixed-field averaging
are then shared LQ algebra. The models run on pyRadBio's own track structure, stopping power
and microdosimetry; the optional external `pyMKM` package is only a cross-check for the MKM
family.

!!! note "Research software"
    The public API may still change. Each model is checked against its source papers by golden
    tests pinned to published values, and the contributor digests in
    [`docs/references/`](https://github.com/e0404/pyRadBio/tree/develop/docs/references) record
    every reconstruction, including where and why a result departs from the paper.

## Model families

| Family | What it is | Documented in |
| --- | --- | --- |
| LEM | Local effect model I, II, III and IV (with GLOBLE): survival from the local dose of ion tracks and the photon response | [Local effect model](user-guide/models/lem.md) |
| MKM | Microdosimetric kinetic models: MKM, modified MKM, MCF MKM, SMK, OSMK 2021/2023 and IMK | [MKM family](user-guide/models/mkm.md) |
| Phenomenological | Thirteen closed-form, LET-based proton (and helium) RBE fits | [Phenomenological proton RBE](user-guide/models/phenomenological.md) |
| RMF | Repair-misrepair-fixation with MCDS double-strand-break yields | [RMF, Katz and NanOx](user-guide/models/other.md) |
| Katz | Cellular track-structure theory (ion kill plus gamma kill) | [RMF, Katz and NanOx](user-guide/models/other.md) |
| NanOx | Calibration and assembly layer; the ion route raises because the nanodosimetric Monte Carlo it needs is not public | [RMF, Katz and NanOx](user-guide/models/other.md) |

The [model overview](user-guide/models/index.md) compares them side by side. Around the models,
a shared layer provides LQ survival and RBE, mixed-field dose averaging, photon-reference and
cell descriptions, stopping powers and a benchmark harness against measured survival data.

## A first calculation

Models live in `pyradbio.models.*` and are imported by name, so optional dependencies load only
when you ask for them. This computes the LQ parameters, survival and iso-survival RBE of a 100
MeV/u carbon beam with the modified MKM on its native backend, using the HSG default cell.

```python
from pyradbio import Ion, PhotonParameters
from pyradbio.models.mkm import MKMModel

# Illustrative photon reference (RBE is formed against it); use your cell line's measured
# alpha_x and beta_x
photon = PhotonParameters(alpha_x=0.172, beta_x=0.0615)
model = MKMModel(photon=photon)

carbon = Ion(Z=6, A=12, energy_MeV_u=100.0)
lq = model.alpha_beta(carbon)  # LQParameters(alpha, beta)
s2 = model.survival(carbon, 2.0)  # surviving fraction after 2 Gy
d10 = model.dose_for_survival(carbon, 0.10)  # dose for 10% survival [Gy]
rbe10 = model.rbe_at_survival(carbon, 0.10)  # RBE at 10% survival
```

The same four calls (`alpha_beta`, `survival`, `dose_for_survival`, `rbe_at_survival`) work on
every model; see [Core concepts](user-guide/concepts.md).

## Where next

- [Installation](getting-started/installation.md): extras, installing from a clone, optional
  external datasets.
- [Quick start](getting-started/quickstart.md): LEM I, MKM, a phenomenological model and a mixed
  field, step by step.
- [Core concepts](user-guide/concepts.md): the LQ layer, photon reference, cell target and the
  common model interface.
- [User guide](user-guide/models/index.md): one page per model family, plus fields, physics
  building blocks, tables and the command line, and benchmarking.
- [Examples](user-guide/examples.md): runnable scripts, one or more per model family.
- [API reference][pyradbio]: every public module, class and function, generated from the
  docstrings.

## Citing

Every release is archived on Zenodo. The concept DOI
[10.5281/zenodo.24046436](https://doi.org/10.5281/zenodo.24046436) always resolves to the latest
version, and the Zenodo page lists the DOI of each release. Cite the release you used (see
[`CITATION.cff`](https://github.com/e0404/pyRadBio/blob/develop/CITATION.cff)), together with the
papers of the models you use. Each model's docstring and the
[reference digests](https://github.com/e0404/pyRadBio/tree/develop/docs/references) give the
primary papers and their DOIs.
