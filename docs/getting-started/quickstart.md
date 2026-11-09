# Quick start

This page walks through the three kinds of model pyRadBio offers (a track-structure model, a
microdosimetric model and a LET-based fit) and a mixed field, using the same few calls each time.
Everything runs with the core install; see [Installation](installation.md) if you have not
installed the package yet.

Models live in `pyradbio.models.*` and are imported by name, so optional dependencies load only
when you ask for them. The LQ utilities and the radiation, photon and cell descriptions are
available from the top-level `pyradbio` package.

## Survival, dose and RBE with LEM I

A model needs a **photon reference** (the LQ parameters of the cell line for photons, plus the
transition dose `D_t` the LEM uses) and a **cell target** (here the nuclear radius). Both are
plain frozen dataclasses.

```python
from pyradbio import CellTarget, Ion, LQParameters, PhotonParameters
from pyradbio import dose_for_survival, rbe, survival
from pyradbio.models.lem.lem1 import LEM1

# V79 photon reference (Scholz & Kraft 1996; Elsasser & Scholz 2007)
photon = PhotonParameters(alpha_x=0.18, beta_x=0.028, D_t=30.0)
lem = LEM1(photon=photon, cell=CellTarget(nucleus_radius_um=5.0))
```

`alpha_beta` returns the model's LQ parameters for one ion at one energy (kinetic energy per
nucleon, MeV/u). The LQ utilities in `pyradbio.core.lq` then turn them into survival, the dose
for a target survival and the iso-survival RBE:

```python
carbon = Ion(Z=6, A=12, energy_MeV_u=11.0)
ab = lem.alpha_beta(carbon)  # LQParameters(alpha, beta)

s = survival(2.0, ab.alpha, ab.beta)  # surviving fraction at 2 Gy
d10 = dose_for_survival(0.10, ab.alpha, ab.beta)  # dose for 10% survival [Gy]
photon_lq = LQParameters(photon.alpha_x, photon.beta_x)
rbe10 = rbe(0.10, ab.alpha, ab.beta, reference=photon_lq)  # RBE at 10% survival
```

You rarely need to do this by hand: the model itself offers the same quantities as methods, with
the photon reference taken from the model. Unlike the pure-LQ `rbe` above, they respect the
shouldered-exponential photon curve defined by `D_t`:

```python
s = lem.survival(carbon, 2.0)
d10 = lem.dose_for_survival(carbon, 0.10)
rbe10 = lem.rbe_at_survival(carbon, 0.10)  # iso-survival RBE, RBE_10
rbe_2gy = lem.rbe_at_dose(carbon, 2.0)  # iso-effect RBE at a 2 Gy ion dose
```

!!! note "Two RBE definitions"
    `rbe_at_survival` compares doses at the same survival level; `rbe_at_dose` compares the
    reference dose that gives the same effect as a given ion dose. They are different functions
    and are not interchangeable. See [Core concepts](../user-guide/concepts.md).

## MKM on the native backend

The microdosimetric kinetic model computes the saturation-corrected specific energy \(z^*_{1D}\)
through a pluggable backend. The default is pyRadBio's own native engine, so no extra is needed;
the defaults are the Inaniwa 2010 HSG cell parameters.

```python
from pyradbio.models.mkm import MKMModel

# Illustrative photon reference; use your cell line's measured alpha_x and beta_x
mkm = MKMModel(photon=PhotonParameters(alpha_x=0.172, beta_x=0.0615))
lq = mkm.alpha_beta(Ion(Z=6, A=12, energy_MeV_u=100.0))
rbe10 = mkm.rbe_at_survival(Ion(Z=6, A=12, energy_MeV_u=100.0), 0.10)
```

Without a `photon` the model still gives \((\alpha, \beta)\) but raises on any RBE call, because
the MKM parameters `alpha0` and `beta0` describe the cell, not the photon reference. The external
pyMKM package can be injected as a different backend for cross-checking; see the
[MKM family](../user-guide/models/mkm.md).

## A LET-based phenomenological model

The phenomenological proton RBE fits are closed forms in the dose-averaged LET, as a treatment
planning system holds it. Their natural input is a LET in keV/\(\mu\)m rather than an ion, so
they add a LET route that mirrors the ion interface method for method:

```python
from pyradbio.models.phenomenological import McNamara

# LET-based proton RBE (here McNamara et al. 2015)
mcn = McNamara(photon=PhotonParameters(alpha_x=0.18, beta_x=0.028))
ab = mcn.alpha_beta_from_let(5.0)  # dose-averaged LET [keV/um]
rbe10 = mcn.rbe_at_survival_from_let(5.0, 0.10)  # RBE at 10% survival
```

The LET route needs no stopping power. See
[Phenomenological proton RBE](../user-guide/models/phenomenological.md) for the other twelve fits
and their validity ranges.

## A mixed field

A `RadiationField` is a list of `SpectrumComponent`s, each an ion with either a fluence or a dose.
Every model accepts one wherever it accepts an `Ion`; the per-ion quantities are averaged over
the dose-weighted components with the model's own mixing rule.

```python
from pyradbio import RadiationField, SpectrumComponent

field = RadiationField(
    [
        SpectrumComponent(Ion(Z=1, A=1, energy_MeV_u=150.0), dose_Gy=1.0),
        SpectrumComponent(Ion(Z=6, A=12, energy_MeV_u=100.0), dose_Gy=1.0),
    ]
)
mixed = mkm.alpha_beta(field)
rbe10_mixed = mkm.rbe_at_survival(field, 0.10)
```

Components may also be given as fluence, or as an energy bin that is integrated properly across
the bin rather than evaluated at its centre. See
[Radiation fields and mixing](../user-guide/fields.md).

## Tables

Every model can tabulate its LQ parameters over a grid of ions (by atomic number) and energies:

```python
table = mkm.alpha_beta_table([1, 6], [10.0, 100.0])
for row in table.rows:
    print(row.Z, row.energy_MeV_u, row.alpha, row.beta)
```

Writing such tables to CSV or Parquet, and the command-line interface, are covered in
[Tables and command line](../user-guide/tables-cli.md).

## Where next

- [Core concepts](../user-guide/concepts.md): the LQ layer, the photon reference and the common
  model interface in detail.
- [Models](../user-guide/models/index.md): one page per family, with the published parameter
  presets.
- [Examples](../user-guide/examples.md): runnable scripts, one or more per model family.
- [API reference][pyradbio]: every public object.
