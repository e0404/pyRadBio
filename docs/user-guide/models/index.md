# Models

pyRadBio implements six model families. All of them map a radiation field to the
parameters of the linear-quadratic (LQ) model, \( -\ln S(D) = \alpha D + \beta D^2 \), and
(except [NanOx](other.md), which is an assembly layer) share one interface,
[`BiophysicalModel`][pyradbio.models.base.BiophysicalModel]:

| Method | Returns |
| --- | --- |
| `alpha_beta(field)` | `LQParameters` of an `Ion` or a mixed `RadiationField` |
| `survival(field, dose_Gy)` | surviving fraction (scalar or array dose) |
| `dose_for_survival(field, S)` | dose reaching survival `S` |
| `rbe_at_survival(field, S)`, `rbe_at_dose(field, dose_Gy)` | iso-effect RBE against the photon reference |
| `reference()` | the photon reference (raises if the model has none) |
| `alpha_beta_table(ions, energies)` | an alpha/beta table, see [Tables and command line](../tables-cli.md) |

Models are frozen dataclasses. Import them by name from their own module, so that
optional dependencies are only loaded when you ask for them.

## Which model needs what

What a model implements is declared on the class as
[`ModelCapabilities`][pyradbio.models.base.ModelCapabilities], so you can branch on it
instead of provoking an exception:

```python
from pyradbio.models.mkm import MKMModel, StochasticMKM
from pyradbio.models.phenomenological import McNamara

print(MKMModel.capabilities)
print(StochasticMKM.capabilities)
print(McNamara.capabilities.let_route)
```

The flags mean:

- **ion route**: `alpha_beta(Ion(...))` is implemented. The model derives LET and track
  structure from the ion and its energy itself.
- **LET route**: `alpha_beta_from_let(let_keV_um)` takes a dose-averaged LET directly. This
  is what a treatment-planning system holds, and needs no track structure.
- **spectrum route**: `alpha_beta_from_spectrum` and `alpha_beta_from_dose_spectrum` take a
  single-event lineal-energy spectrum \( d(y) \).
- **mixed field**: `alpha_beta(RadiationField)` mixes the components of a binned field
  (see [Radiation fields and mixing](../fields.md)). Where this is `no`, the call raises:
  evaluate each ion component yourself.
- **LQ exact**: `survival` is exactly \( \exp(-\alpha D - \beta D^2) \) of `alpha_beta`.
  Where this is `no`, `alpha_beta` is a descriptor of the model and `survival` is the
  curve to use (SMK and OSMK add a non-LQ factor, Katz has a multi-target shoulder).

| Family | Class | Routes | Mixed field | LQ exact | Documented on |
| --- | --- | --- | --- | --- | --- |
| LEM | [`LEM1`][pyradbio.models.lem.lem1.LEM1] | ion | yes | yes | [LEM](lem.md) |
| LEM | [`LEM2`][pyradbio.models.lem.lem2.LEM2] | ion | yes | yes | [LEM](lem.md) |
| LEM | [`LEM3`][pyradbio.models.lem.lem3.LEM3] | ion | yes | yes | [LEM](lem.md) |
| LEM | [`LEM4`][pyradbio.models.lem.lem4.LEM4] | ion | yes | yes | [LEM](lem.md) |
| MKM | [`MKMModel`][pyradbio.models.mkm.model.MKMModel] | ion (`z*_1D` from a backend) | yes | yes | [MKM family](mkm.md) |
| MKM | [`ModifiedMKM`][pyradbio.models.mkm.spectrum_models.ModifiedMKM] | ion, spectrum | no | yes | [MKM family](mkm.md) |
| MKM | [`MCFMKMModel`][pyradbio.models.mkm.spectrum_models.MCFMKMModel] | ion, spectrum | no | yes | [MKM family](mkm.md) |
| MKM | [`StochasticMKM`][pyradbio.models.mkm.spectrum_models.StochasticMKM] (SMK) | ion, spectrum | no | no | [MKM family](mkm.md) |
| MKM | [`OSMK2021`][pyradbio.models.mkm.osmk.OSMK2021] | ion, spectrum | no | no | [MKM family](mkm.md) |
| MKM | [`OSMK2023`][pyradbio.models.mkm.osmk.OSMK2023] | ion, spectrum | no | no | [MKM family](mkm.md) |
| MKM | [`IMKModel`][pyradbio.models.mkm.imk.IMKModel] | ion, spectrum | no | yes | [MKM family](mkm.md) |
| Phenomenological | [`WilkensOelfke`][pyradbio.models.phenomenological.wilkens.WilkensOelfke] | LET, ion | yes | yes | [Phenomenological](phenomenological.md) |
| Phenomenological | [`Carabe`][pyradbio.models.phenomenological.carabe.Carabe] | LET, ion | yes | yes | [Phenomenological](phenomenological.md) |
| Phenomenological | [`LinearScaling`][pyradbio.models.phenomenological.frese.LinearScaling] | LET, ion | yes | yes | [Phenomenological](phenomenological.md) |
| Phenomenological | [`Wedenberg`][pyradbio.models.phenomenological.wedenberg.Wedenberg] | LET, ion | yes | yes | [Phenomenological](phenomenological.md) |
| Phenomenological | [`McNamara`][pyradbio.models.phenomenological.mcnamara.McNamara] | LET, ion | yes | yes | [Phenomenological](phenomenological.md) |
| Phenomenological | [`MairaniHelium`][pyradbio.models.phenomenological.mairani.MairaniHelium] | LET, ion | yes | yes | [Phenomenological](phenomenological.md) |
| Phenomenological | [`Tilly2`][pyradbio.models.phenomenological.tilly.Tilly2] | LET, ion | yes | yes | [Phenomenological](phenomenological.md) |
| Phenomenological | [`Tilly10`][pyradbio.models.phenomenological.tilly.Tilly10] | LET, ion | yes | yes | [Phenomenological](phenomenological.md) |
| Phenomenological | [`Chen`][pyradbio.models.phenomenological.chen.Chen] | LET, ion | yes | yes | [Phenomenological](phenomenological.md) |
| Phenomenological | [`Jones`][pyradbio.models.phenomenological.jones.Jones] | LET, ion | yes | yes | [Phenomenological](phenomenological.md) |
| Phenomenological | [`RorvikUnweighted`][pyradbio.models.phenomenological.rorvik.RorvikUnweighted] | LET, ion | yes | yes | [Phenomenological](phenomenological.md) |
| Phenomenological | [`Peeler`][pyradbio.models.phenomenological.peeler.Peeler] | LET, ion | yes | yes | [Phenomenological](phenomenological.md) |
| Phenomenological | [`Unkelbach`][pyradbio.models.phenomenological.unkelbach.Unkelbach] | LET, ion | yes | yes | [Phenomenological](phenomenological.md) |
| RMF | [`RMFModel`][pyradbio.models.rmf.model.RMFModel] | ion | yes | yes | [RMF, Katz and NanOx](other.md) |
| Katz | [`KatzCellularModel`][pyradbio.models.katz.model.KatzCellularModel] | ion (LET as keyword) | no | no | [RMF, Katz and NanOx](other.md) |
| NanOx | [`NanOxModel`][pyradbio.models.nanox.model.NanOxModel] | none (the ion route raises) | no | no | [RMF, Katz and NanOx](other.md) |

For the phenomenological models the ion route reduces the ion to a dose-averaged LET with a
`stopping_power` first; working from a LET map needs none.

## Choosing a model

Start from what you have, not from the model you would like to run.

- **A dose-averaged LET map only** (a typical treatment-planning input): the
  [phenomenological proton models](phenomenological.md) are the only family with a native
  LET route. They are closed-form fits and serve as the baseline the mechanistic models
  are judged against.
- **Ions with energies, and a photon reference with a transition dose \( D_t \)**: the
  [LEM](lem.md). It needs
  [`PhotonParameters`][pyradbio.core.photon.PhotonParameters] with `D_t` and a
  [`CellTarget`][pyradbio.core.cell.CellTarget] (the nucleus radius).
- **Ions with energies, cell parameters from a published MK fit**: the
  [MKM family](mkm.md). The published parameter sets carry the cell geometry and the
  photon reference, so most of them need nothing else. Only `MKMModel` mixes fields;
  the spectrum-based MKMs take one ion at a time.
- **A measured or simulated lineal-energy spectrum**: the spectrum route of the
  [MKM family](mkm.md).
- **DSB yields and binary misrepair, or the oldest track-structure theory**:
  [RMF and Katz](other.md).

Two cautions apply across the whole package:

- Models of the same family are not interchangeable parameter sets. The published fits of
  one cell line in different papers are independent and must not be mixed; each
  constructor is documented with the table it comes from.
- The LQ parameters of an ion model are a statement about that model on that cell line.
  Comparisons between models, and against measured data, are discussed in
  [Benchmarking against data](../benchmarking.md).

## Published presets

Where a paper fixes a full parameter set, the class has a constructor named after the cell
line and the paper, for example `StochasticMKM.hsg_inaniwa2018()`,
`OSMK2021.hsg_inaniwa2021()` or `IMKModel.hsg_matsuya2025()`. Each constructor accepts
keyword overrides, and because models are frozen dataclasses you can also change a field
afterwards with `dataclasses.replace`. Both routes give the same model:

```python
from dataclasses import replace

from pyradbio import Ion
from pyradbio.models.mkm import ModifiedMKM, StochasticMKM

mmkm = ModifiedMKM.hsg_mk_inaniwa2018()
at_construction = ModifiedMKM.hsg_mk_inaniwa2018(site_geometry="sphere")
afterwards = replace(mmkm, site_geometry="sphere")
assert at_construction == afterwards

smk = StochasticMKM.hsg_inaniwa2018()
ab = smk.alpha_beta(Ion(Z=6, A=12, energy_MeV_u=100.0))  # an LQParameters
```

A preset only applies to the class it was fitted for: `IMKModel.hsg_mk_inaniwa2018()`
raises a `TypeError` rather than quietly returning the parent's `ModifiedMKM` parameters.

Presets exist for the MKM family (see [MKM family](mkm.md)), for RMF, Katz and NanOx (see
[RMF, Katz and NanOx](other.md)) and for the phenomenological `WilkensOelfke.v79_wilkens2004()`.
The LEM has none: you supply the photon reference and the cell yourself, see
[LEM](lem.md).

## Where the numbers come from

By default LET comes from
[`default_stopping_power`][pyradbio.track.stopping_power.default_stopping_power]; every
model takes an explicit `stopping_power` where another source is wanted (see
[Physics building blocks](../physics.md)). Each model is checked against its source
papers by golden tests, and the paper digests in
[`docs/references/`](https://github.com/e0404/pyRadBio/tree/develop/docs/references) record every
reconstruction, including where a result departs from the paper. Open work and known
deviations are listed in
[`ROADMAP.md`](https://github.com/e0404/pyRadBio/blob/develop/ROADMAP.md).
