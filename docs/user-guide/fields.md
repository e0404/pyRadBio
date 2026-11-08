# Radiation fields and mixing

A model's `alpha_beta`, `survival` and RBE methods accept either a single
[`Ion`][pyradbio.core.radiation.Ion] or a
[`RadiationField`][pyradbio.core.radiation.RadiationField], a list of components that may be
mono-energetic or binned in energy. This page describes how a field is built, how models evaluate
it, and which models support it. The units are those of [Units and conventions](../parameter_conventions.md).

## Building a field

- [`Ion`][pyradbio.core.radiation.Ion]`(Z, A, energy_MeV_u)` is a mono-energetic ion; the energy is
  the kinetic energy per nucleon in MeV/u.
- [`SpectrumComponent`][pyradbio.core.radiation.SpectrumComponent] is one component of a field. It
  is given by **exactly one** of `fluence` (particles/cm\( ^2 \)) or `dose_Gy`. Without energy
  bounds it is mono-energetic. With both `energy_lo_MeV_u` and `energy_hi_MeV_u` it is an energy
  **bin**, and `ion.energy_MeV_u` must lie inside the bin. For a bin the ion supplies only the species
  \( (Z, A) \); the energy is replaced by the nodes of a sub-grid.
- [`RadiationField`][pyradbio.core.radiation.RadiationField]`(components)` holds the components;
  `RadiationField.mono(ion, dose_Gy)` is the shortcut for a single ion.

A fluence is converted to dose through the ion's LET,
\( D = 1.602\times10^{-9}\,\Phi\,L/\rho \) with \( D \) in Gy, \( L \) in keV/\( \mu \)m, \( \Phi \) in cm\( ^{-2} \)
and \( \rho \) in g/cm\( ^3 \), by
[`fluence_to_dose`][pyradbio.core.radiation.fluence_to_dose] and its inverse
[`dose_to_fluence`][pyradbio.core.radiation.dose_to_fluence]. Models do this conversion themselves,
with their own stopping power, when a component is specified by fluence.

```python
from pyradbio import Ion, RadiationField, SpectrumComponent
from pyradbio.core.radiation import dose_to_fluence, fluence_to_dose
from pyradbio.track.stopping_power import default_stopping_power

stopping = default_stopping_power()  # the library default stopping power

proton = Ion(Z=1, A=1, energy_MeV_u=150.0)
let = stopping.let_keV_um(proton)  # keV/um
fluence = dose_to_fluence(2.0, let)  # cm^-2 for 2 Gy
print(fluence_to_dose(fluence, let))  # back to 2 Gy

# A mono-energetic two-component field, specified by dose.
mixed = RadiationField(
    [
        SpectrumComponent(Ion(Z=6, A=12, energy_MeV_u=10.0), dose_Gy=0.5),
        SpectrumComponent(Ion(Z=6, A=12, energy_MeV_u=270.0), dose_Gy=1.5),
    ]
)
print(mixed.is_mixed, mixed.total_dose_Gy(stopping.let_keV_um))
```

## How a model evaluates a field

Every model derived from
[`BiophysicalModel`][pyradbio.models.base.BiophysicalModel] shares one implementation of
[`alpha_beta`][pyradbio.models.base.BiophysicalModel.alpha_beta]. There is no per-model mixing loop:

1. [`dose_quadrature`][pyradbio.core.mixing.dose_quadrature] expands the field into
   `(ion, dose weight)` nodes. A mono-energetic component is one node. A binned component becomes
   `sub_nodes` ions (default 16, on a logarithmic sub-grid, edges included), each carrying a
   composite-Simpson weight times the dose density \( \varphi(E)\,S(E) \).
2. The model's per-ion mixing quantities are **dose-averaged** over the nodes, each distinct ion
   being evaluated once.
3. The averaged quantities are mapped back to LQ parameters.

### The default rule

By default the mixing quantities are \( \alpha \) and \( \sqrt\beta \) (the Zaider-Rossi / Kanai
convention):

\[
\bar\alpha = \sum_i d_i\,\alpha_i,
\qquad
\sqrt{\bar\beta} = \sum_i d_i\,\sqrt{\beta_i},
\qquad
d_i = \frac{D_i}{\sum_j D_j}.
\]

It is \( \sqrt\beta \), not \( \beta \), that is dose-averaged. The helpers
[`dose_average`][pyradbio.core.mixing.dose_average] and
[`dose_average_sqrt`][pyradbio.core.mixing.dose_average_sqrt] implement the two averages. The RMF
model uses this rule, so its mixed-field result can be reproduced by hand:

```python
from pyradbio.core.mixing import dose_average, dose_average_sqrt
from pyradbio.models.rmf import RMFModel

rmf = RMFModel.v79_furusawa2000()
p_slow, p_fast = Ion(1, 1, 5.0), Ion(1, 1, 150.0)
field = RadiationField(
    [SpectrumComponent(p_slow, dose_Gy=0.5), SpectrumComponent(p_fast, dose_Gy=1.5)]
)

a, b = rmf.alpha_beta(p_slow), rmf.alpha_beta(p_fast)
doses = [0.5, 1.5]
print(rmf.alpha_beta(field))
print(dose_average([a.alpha, b.alpha], doses), dose_average_sqrt([a.beta, b.beta], doses))
```

### Families that mix on something else

The two hooks `_mixing_quantities` and `_lq_from_mixed` let a family choose the quantity that is
averaged:

| Family | Quantity dose-averaged |
| --- | --- |
| LEM I to IV, RMF | \( \alpha \) and \( \sqrt\beta \) (default) |
| `MKMModel` | the saturation-corrected dose-mean specific energy \( z^*_{1D} \) (then \( \alpha = \alpha_0 + \beta_0 z^*_{1D} \)) |
| [phenomenological models](models/phenomenological.md) | the dose-averaged LET \( \mathrm{LET}_d \), after which the model is evaluated once |

## Binned components: never at the bin centre

Across a wide energy bin the stopping power \( S \), the model quantity (\( z^* \), \( \alpha \))
and hence their product all vary. The product integral
\( \int \varphi(E)\,S(E)\,Q(E)\,\mathrm{d}E \) is **not** the product of the values at the bin centre:
what matters is the covariance between \( S \) and \( Q \) inside the bin. pyRadBio therefore
integrates the *product* on a fine logarithmic sub-grid and never evaluates a bin at its centre. The
default fluence density of a binned component is uniform in energy; a dose-specified bin has a
uniform dose density. See the numerics section of
[AGENTS.md](https://github.com/e0404/pyRadBio/blob/develop/AGENTS.md) for the rule as a project
requirement.

Use the model's `alpha_beta` and let it build the nodes; both `sub_nodes` and `spacing`
(`"log"` or `"linear"`) can be passed. The example below uses the native MKM, which is fast, on a wide carbon bin and compares it with
the bin-centre shortcut that the library avoids:

```python
import math

from pyradbio.core.mixing import dose_quadrature
from pyradbio.models.mkm import MKMModel

mkm = MKMModel()  # Inaniwa 2010 HSG defaults, native backend

e_lo, e_hi = 1.0, 30.0  # MeV/u, a wide bin
centre = Ion(Z=6, A=12, energy_MeV_u=math.sqrt(e_lo * e_hi))
carbon_bin = SpectrumComponent(centre, fluence=1.0e7, energy_lo_MeV_u=e_lo, energy_hi_MeV_u=e_hi)
binned = RadiationField([carbon_bin])

print(mkm.alpha_beta(binned))  # 16 log sub-nodes per bin: the default
print(mkm.alpha_beta(centre))  # the bin-centre value: shown for contrast only

# The nodes a model dose-averages over, with their dose weights [Gy]. (Here the
# weights use the default stopping power; a model uses its own.)
nodes = dose_quadrature(binned, stopping.let_keV_um)
print(len(nodes), nodes[0][0].energy_MeV_u, nodes[-1][0].energy_MeV_u)
print(sum(dose for _ion, dose in nodes))  # dose of the whole bin
```

The dose of a binned component is the sum of its node weights, as above, and that is what
`SpectrumComponent.absorbed_dose_Gy`, `RadiationField.dose_weights` and
`RadiationField.total_dose_Gy` return: `binned.total_dose_Gy(stopping.let_keV_um)` equals the sum
printed above.

The sub-grid converges quickly. With a phenomenological model (instant to evaluate) the effect of
`sub_nodes` is cheap to inspect:

```python
from pyradbio import PhotonParameters
from pyradbio.models.phenomenological import McNamara

photon = PhotonParameters(alpha_x=0.15, beta_x=0.05)
model = McNamara(photon, stopping_power=stopping)

proton_bin = SpectrumComponent(
    Ion(1, 1, 8.0), fluence=1.0e8, energy_lo_MeV_u=2.0, energy_hi_MeV_u=30.0
)
spectrum = RadiationField([proton_bin, SpectrumComponent(Ion(1, 1, 150.0), dose_Gy=1.0)])

for n in (4, 16, 64):
    print(n, model.alpha_beta(spectrum, sub_nodes=n))
```

The absolute scale of the stopping power cancels only when all components are fluence-specified;
mixing fluence-specified and dose-specified components requires physical Gy weights. Use one
stopping-power source for all components, including components of different \( Z \).
To impose a differential fluence other than uniform in energy, call `dose_quadrature` with
`fluence_density_fn`.

## Which models support mixed fields

Support is declared in the class attribute
[`capabilities`][pyradbio.models.base.ModelCapabilities] of each model; check it instead of calling
and catching an exception:

```python
print(MKMModel.capabilities)
print(McNamara.capabilities.mixed_field, McNamara.capabilities.let_route)
```

| Family | `mixed_field` | Note |
| --- | --- | --- |
| LEM I to IV | yes | default rule; LEM evaluations are slow, so a binned field takes correspondingly longer |
| `MKMModel` | yes | mixes on \( z^*_{1D} \) |
| RMF | yes | default rule |
| phenomenological models (13) | yes | reduced to one \( \mathrm{LET}_d \) first |
| Katz | no | eq. (10) is a product of two non-LQ modes, not a dose average; raises `NotImplementedError` |
| NanOx | no | no ion route |
| `ModifiedMKM`, `StochasticMKM`, `MCFMKMModel`, `OSMK2021`, `OSMK2023`, `IMKModel` | no | `spectrum_route=True`: take a lineal-energy spectrum \( d(y) \) directly |

A model with `mixed_field=False` raises `NotImplementedError` for a mixed field. For the
spectrum-based MKM variants the error message points to `alpha_beta_from_spectrum` with a combined
\( d(y) \), or to `alpha_beta` on each ion component; see the [MKM family](models/mkm.md).

## Example script

[`examples/mixed_field_parameters.py`](https://github.com/e0404/pyRadBio/blob/develop/examples/mixed_field_parameters.py)
works through two dose-weighted components, one wide energy bin, and the convergence in the number of
sub-nodes, with LEM I (needs matplotlib).
