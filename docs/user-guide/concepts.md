# Core concepts

Every model in pyRadBio, from the closed-form LET fits to LEM IV, is built on the same small
set of objects: an LQ layer that turns \((\alpha, \beta)\) into survival, dose and RBE; a photon
reference; a cell description; and one model interface that all families implement. This page
describes those pieces. Units are listed on [Units and conventions](../parameter_conventions.md).

## The LQ layer

The module `pyradbio.core.lq` owns the linear-quadratic algebra. With dose \(D\) in Gy,
\(\alpha\) in Gy\(^{-1}\) and \(\beta\) in Gy\(^{-2}\),

\[
-\ln S(D) = \alpha D + g\,\beta D^2 ,
\]

where \(g\) is the Lea-Catcheside protraction factor. The default \(g = 1\) is an acute dose, so
callers who do not care about time see the plain LQ model. Fractionated or protracted schedules
supply \(g\) from [`DoseSchedule`][pyradbio.core.protraction.DoseSchedule] and
[`lea_catcheside_g`][pyradbio.core.protraction.lea_catcheside_g]; it depends only on the timing
of the irradiation, never on the cell or the radiation quality, which is why it is an argument
rather than a field of the parameters.

The building blocks are plain functions of floats:

| Function | Returns |
| --- | --- |
| [`LQParameters`][pyradbio.core.lq.LQParameters] | A frozen `(alpha, beta)` pair, validated finite and non-negative |
| [`survival`][pyradbio.core.lq.survival] | \(S(D) = \exp(-(\alpha D + g\beta D^2))\) |
| [`log_kill`][pyradbio.core.lq.log_kill] | \(-\ln S(D)\), without exponentiating (no underflow at large doses) |
| [`dose_for_survival`][pyradbio.core.lq.dose_for_survival] | The dose reaching a survival fraction, by the numerically stable root of the LQ quadratic |
| [`rbe`][pyradbio.core.lq.rbe] | Iso-survival RBE against a reference `LQParameters`, with separate `g` and `reference_g` |
| [`bed`][pyradbio.core.lq.bed], [`eqd2`][pyradbio.core.lq.eqd2] | Biologically effective dose and the dose in 2 Gy fractions that is isoeffective |
| [`fit_lq_linearised`][pyradbio.core.lq.fit_lq_linearised], [`fit_lq_weighted`][pyradbio.core.lq.fit_lq_weighted] | Least-squares \((\alpha, \beta)\) read back off a dose response |

```python
from pyradbio import LQParameters, bed, dose_for_survival, eqd2, log_kill, rbe, survival
from pyradbio.core.lq import fit_lq_linearised, fit_lq_weighted

alpha, beta = 0.3, 0.03
s = survival(2.0, alpha, beta)
effect = log_kill(2.0, alpha, beta)  # -ln S
d10 = dose_for_survival(0.10, alpha, beta)

ref = LQParameters(alpha=0.15, beta=0.03)
rbe10 = rbe(0.10, alpha, beta, reference=ref)

# 30 fractions of 2 Gy with complete repair between them: g = 1/30
total = 60.0
bed_10 = bed(total, alpha_beta_ratio_Gy=10.0, g=1 / 30)
eqd2_10 = eqd2(total, alpha_beta_ratio_Gy=10.0, g=1 / 30)
```

The two fit functions read \((\alpha, \beta)\) back from a dose response, which is how the LEM
extracts its LQ parameters from a simulated curve. They are **different estimators** and give
different numbers on noisy data: `fit_lq_linearised` fits \(-\ln S / D = \alpha + \beta D\)
unweighted, `fit_lq_weighted` fits \(-\ln S = \alpha D + \beta D^2\) with inverse-sigma weights.
Pick the one the reference you compare with used. Neither clamps a negative \(\beta\): in
single-track overkill it is a measurement, not noise.

```python
import numpy as np

doses = np.array([0.5, 1.0, 2.0, 4.0, 6.0])
effect = alpha * doses + beta * doses**2
a_fit, b_fit = fit_lq_linearised(doses, effect)
a_w, b_w = fit_lq_weighted(doses, effect, sigma=np.full_like(doses, 0.01))
```

## The photon reference

[`PhotonParameters`][pyradbio.core.photon.PhotonParameters] describes the photon response of a
cell line or tissue endpoint: `alpha_x` and `beta_x`, an optional transition dose `D_t`, and an
optional `cluster` block of LEM II/IV clustering inputs. RBE is always formed against it.

Without `D_t` the reference is the plain LQ curve. With `D_t` it is the **shouldered
exponential** the LEM family is built on (Scholz et al. 1997, eq. 1; Elsasser and Scholz 2007,
eq. 7): LQ up to `D_t`, then a straight exponential tail with slope
\(s_\mathrm{max} = \alpha_x + 2\beta_x D_t\). The LEM needs `D_t` because its ion survival is
built from the photon response of the local dose. For every model, `D_t` also reshapes the
reference used for RBE, so an iso-survival RBE at a survival level reached beyond `D_t` is formed
against the tail, not against the LQ curve extrapolated to high dose.

```python
from pyradbio import PhotonParameters, log_kill

photon = PhotonParameters(alpha_x=0.18, beta_x=0.028, D_t=30.0)
s_max = photon.max_slope()  # alpha_x + 2 beta_x D_t
lk = photon.log_kill(40.0)  # on the exponential tail
d_ref = photon.dose_for_survival(1e-6)  # inverts the shouldered curve, tail included

# The tail lies below the LQ parabola it leaves at D_t
assert lk < log_kill(40.0, photon.alpha_x, photon.beta_x)
```

`dose_for_log_kill` is the inverse of `log_kill` and is what the iso-effect RBE uses, because
going through \(S = e^{-\text{log-kill}}\) underflows to zero at a few hundred units of log-kill.

## The cell target

[`CellTarget`][pyradbio.core.cell.CellTarget] holds the geometry the models need:
`nucleus_radius_um`, and optionally `domain_radius_um` (the MKM domain radius \(r_d\)),
`height_um` (the nuclear cylinder, defaulting to the diameter) and `effective_size_factor`.
That factor converts a measured, geometric nuclear radius into the effective one the LEM hit
statistics need. It defaults to 1.0 on purpose: published LEM inputs usually quote the effective
size already, and applying 0.8 to them shrinks the nucleus a second time. Pass 0.8 only when the
radius you hold is genuinely geometric.

```python
from pyradbio import CellTarget

cell = CellTarget(nucleus_radius_um=5.0)
r_eff = cell.effective_nucleus_radius_um()
```

Not every model takes a `CellTarget`: the MKM family carries its geometry as constructor fields
(`nucleus_radius_um`, `domain_radius_um`), and the LET-based fits need no geometry at all. Each
model page says which inputs it takes.

## One interface for every model

All families implement [`BiophysicalModel`][pyradbio.models.base.BiophysicalModel]. A model
supplies its per-ion physics; the base class provides everything else. The *field* argument is
either an [`Ion`][pyradbio.core.radiation.Ion] or a
[`RadiationField`][pyradbio.core.radiation.RadiationField] (see
[Radiation fields and mixing](fields.md)).

| Method | Returns |
| --- | --- |
| `alpha_beta(field)` | `LQParameters` of an ion or a mixed field |
| `survival(field, dose_Gy)` | Surviving fraction (scalar or array dose) |
| `dose_for_survival(field, S)` | The dose that reaches survival `S` |
| `reference()` | The photon reference, as `PhotonParameters` (raises if the model has none) |
| `rbe_at_survival(field, S)` | Iso-survival RBE, \(D_\mathrm{ref}(S) / D(S)\) (RBE\(_{10}\) at \(S = 0.1\)) |
| `rbe_at_dose(field, dose_Gy)` | Iso-effect RBE at a given ion dose |
| `alpha_beta_table(ions, energies)` | An [`AlphaBetaTable`][pyradbio.tables.alpha_beta.AlphaBetaTable] over an (ion, energy) grid |

Model-specific axes extend these with keyword-only arguments and never change the positional
signature: OSMK takes `pO2_mmHg=`, IMK a dose `schedule=`, Katz a `let_keV_um=`. Code written
against the table above works on every family.

```python
from pyradbio import Ion
from pyradbio.models.mkm import MKMModel

model = MKMModel(photon=PhotonParameters(alpha_x=0.172, beta_x=0.0615))  # illustrative
proton = Ion(Z=1, A=1, energy_MeV_u=100.0)

lq = model.alpha_beta(proton)
s2 = model.survival(proton, 2.0)
d10 = model.dose_for_survival(proton, 0.10)
ref = model.reference()  # PhotonParameters
rbe10 = model.rbe_at_survival(proton, 0.10)
rbe2 = model.rbe_at_dose(proton, 2.0)
table = model.alpha_beta_table(ions=[1, 6], energies_MeV_u=[10.0, 100.0])
```

### Capabilities

Not every family implements every route.
[`ModelCapabilities`][pyradbio.models.base.ModelCapabilities] is a class-level declaration, a
`ClassVar` named `capabilities`, so callers can branch without provoking errors:

| Flag | Meaning |
| --- | --- |
| `ion_route` | `alpha_beta(Ion)` is implemented (false only for an assembly-only model such as NanOx) |
| `mixed_field` | `alpha_beta(RadiationField)` mixes a field; false means it raises |
| `lq_exact` | Survival is exactly \(\exp(-\alpha D - \beta D^2)\) of `alpha_beta`; false for SMK and Katz, whose `alpha_beta` is then a descriptor, not the curve |
| `let_route` | `alpha_beta_from_let` is implemented (the model is a `LETModel`) |
| `spectrum_route` | A lineal-energy spectrum \(d(y)\) can be passed directly |

```python
print(MKMModel.capabilities)
if not MKMModel.capabilities.let_route:
    print("MKMModel has no LET route")
```

### The LET route

The phenomenological family subclasses [`LETModel`][pyradbio.models.base.LETModel], whose natural
input is a dose-averaged LET in keV/\(\mu\)m rather than an ion. It mirrors the ion interface with
the LET in place of the field:

| Ion route | LET route |
| --- | --- |
| `alpha_beta(ion)` | `alpha_beta_from_let(let)` |
| `survival(ion, D)` | `survival_from_let(let, D)` |
| `dose_for_survival(ion, S)` | `dose_for_survival_from_let(let, S)` |
| `rbe_at_survival(ion, S)` | `rbe_at_survival_from_let(let, S)` |
| `rbe_at_dose(ion, D)` | `rbe_at_dose_from_let(let, D)` |

## RBE is formed on the log-kill

RBE in pyRadBio is the ratio of photon dose to ion dose at equal biological effect, and the
effect is carried as the log-kill \(-\ln S\), never as \(S\). That keeps RBE finite at doses where
\(S\) underflows: for an LQ model against a pure LQ reference, the \(D \to \infty\) limit of
`rbe_at_dose` is exactly \(\sqrt{\beta/\beta_x}\), and \(D \to 0\) gives \(\alpha/\alpha_x\).
The same log-kill form lets non-LQ models (Katz, SMK and OSMK) form their RBE from direct log
effects.

!!! warning "RBE definitions are not interchangeable"
    "RBE" at "2 Gy" can mean three different things, and the same numeric argument gives different
    numbers:

    1. **Iso-survival RBE**, `rbe_at_survival(field, S)`: the dose ratio at a survival level such
       as \(S = 0.1\).
    2. **Iso-effect RBE at an ion dose**, `rbe_at_dose(field, D)`: the reference dose producing
       the same effect as the ion dose \(D\), divided by \(D\).
    3. **RBE at a reference (photon) dose per fraction**, `rbe_at_reference_dose_from_let(let, d_L)`
       on the phenomenological family: the form used in the clinical RBE literature, where
       \((\alpha/\beta)_x\) and the photon dose per fraction are what is recorded.

    The second and third agree only on matching pairs, \(d_L = \mathrm{RBE} \cdot D_\mathrm{ion}\).
    Always state which one you are quoting.

An RBE also depends on the reference's own schedule: quoting a fractionated ion treatment against
an acute photon reference mixes the fractionation effect into the RBE. `rbe` therefore takes
separate protraction factors `g` and `reference_g` for the two arms.

## Models are frozen dataclasses

Every model is a frozen dataclass. Defaults that depend on other fields are resolved once, at
construction, so a model never changes after you build it. To vary a parameter, build a new
instance with `dataclasses.replace` instead of mutating:

```python
import dataclasses

from pyradbio.models.mkm import ModifiedMKM

try:
    model.alpha0 = 0.2
except dataclasses.FrozenInstanceError:
    print("models are immutable")

variant = dataclasses.replace(model, alpha0=0.2)
```

### Published presets

A published parameter set (a cell line from a paper) is stored as a model instance, built by a
classmethod such as `ModifiedMKM.hsg_mk_inaniwa2018()` or `IMKModel.hsg_matsuya2025()`.
Any keyword you pass overrides that preset's field, which is `dataclasses.replace(preset,
**overrides)` underneath. A preset applies only to the exact class it was fitted for: calling it
on a subclass raises a `TypeError` instead of quietly returning the parent's parameters.

```python
preset = ModifiedMKM.hsg_mk_inaniwa2018()  # HSG cells, Inaniwa 2018
tweaked = ModifiedMKM.hsg_mk_inaniwa2018(domain_radius_um=0.30)  # override one field
same = dataclasses.replace(preset, domain_radius_um=0.30)
assert tweaked == same
```

The model pages list the presets of each family.

### Validated options

Options that take one of a fixed set of strings are typed as `Literal`s and checked when the model
is constructed, so a typo fails immediately with the list of allowed values instead of deep in a
calculation:

```python
from pyradbio.models.lem.lem1 import LEM1

try:
    LEM1(photon=photon, cell=cell, beta_model="not-a-model")
except ValueError as err:
    print(err)  # lists the allowed beta_model values
```

## Where next

- [Radiation fields and mixing](fields.md): ions, spectrum components and how fields are averaged.
- [Models](models/index.md): the model families and their presets.
- [Units and conventions](../parameter_conventions.md) and [Architecture](../architecture.md).
- [API reference][pyradbio]: every object named here, with its full signature.
