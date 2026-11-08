# Local effect model (LEM)

The LEM maps the photon dose response onto the inhomogeneous local dose that an ion track
deposits inside the cell nucleus. pyRadBio implements four versions, all on one shared base
([`LEMBase`][pyradbio.models.lem.base.LEMBase]):

| Class | Version | Papers |
| --- | --- | --- |
| [`LEM1`][pyradbio.models.lem.lem1.LEM1] | LEM I | Scholz & Kraft 1996 (doi:10.1016/0273-1177(95)00784-C); the fast approximation of Scholz et al. 1997 (doi:10.1007/s004110050055) |
| [`LEM2`][pyradbio.models.lem.lem2.LEM2] | LEM II | Elsässer & Scholz 2007 (doi:10.1667/RR0467.1): refined radial dose and the high-dose cluster enhancement \( \eta(D) \) |
| [`LEM3`][pyradbio.models.lem.lem3.LEM3] | LEM III | Elsässer et al. 2008 (doi:10.1016/j.ijrobp.2008.02.037): LEM II with a velocity-dependent track core |
| [`LEM4`][pyradbio.models.lem.lem4.LEM4] | LEM IV | Elsässer et al. 2010 (doi:10.1016/j.ijrobp.2010.05.014) and Friedrich et al. 2012 (doi:10.3109/09553002.2011.611213): DSB-pattern and cluster-index formalism, plus the GLOBLE effect model |

LEM III is LEM II with one different radial-dose model, so `LEM3` subclasses `LEM2`; LEM IV
is deliberately not a pointwise function of the local dose and so has its own routes (see
[below](#lem-iv)).

## Constructing a model

Every version needs a photon reference with a transition dose \( D_t \) (the dose above
which the shouldered photon curve turns exponential) and a cell target. Both are the same
objects across the package:

```python
from pyradbio import CellTarget, Ion, PhotonParameters
from pyradbio.models.lem.lem1 import LEM1
from pyradbio.models.lem.lem2 import LEM2

cell = CellTarget(nucleus_radius_um=5.0)
carbon = Ion(Z=6, A=12, energy_MeV_u=100.0)

# V79 photon reference with the LEM I transition dose
lem1 = LEM1(photon=PhotonParameters(alpha_x=0.18, beta_x=0.028, D_t=30.0), cell=cell)
print(lem1.alpha_beta(carbon))  # LQParameters(alpha, beta)
print(lem1.rbe_at_survival(carbon, 0.10))

# LEM II with the refit (reduced) D_t: the V79 photon curve of Elsasser & Scholz 2007,
# Fig. 6 caption
lem2 = LEM2(photon=PhotonParameters(alpha_x=0.093, beta_x=0.026, D_t=7.5), cell=cell)
print(lem2.alpha_beta(carbon))
```

A few things to know:

- `PhotonParameters` without `D_t` is refused: the LEM has no meaning without it. Use the
  refit `D_t` of each version (the LEM II/III papers reduce it per cell line, for V79 from
  30 Gy to 7.5 Gy); combining LEM II's physics with LEM I's `D_t` overshoots the RBE.
- LET comes from [`default_stopping_power`][pyradbio.track.stopping_power.default_stopping_power];
  pass `stopping_power=` to use another source, and `radial_dose=` to inject a different
  track model. By default each version builds its own
  [`ScholzKraftRadialDose`][pyradbio.track.scholz_kraft.ScholzKraftRadialDose].
- LEM II and III build their cluster enhancement from `photon.cluster`
  ([`ClusterParameters`][pyradbio.core.photon.ClusterParameters]), or take any
  [`ClusterResponse`][pyradbio.models.lem.lem2.ClusterResponse] through `cluster=`, for
  example [`ClusterModel`][pyradbio.models.lem.lem2.ClusterModel] or the closed-form
  [`Friedrich2015ClusterModel`][pyradbio.models.lem.lem2.Friedrich2015ClusterModel].
- Mixed fields work as for every other model with `alpha_beta(RadiationField)`, see
  [Radiation fields and mixing](../fields.md).

## Selecting how alpha and beta are computed

The LEM is a family of approximations of the same underlying model, and pyRadBio exposes
them as two independent choices. They are plain constructor arguments, validated at
construction, so a typo fails immediately. Change one afterwards with
`dataclasses.replace(model, beta_model="rapid_lem")`.

### The initial slope: `initial_slope_model`

| Value | Meaning |
| --- | --- |
| `"cross_section"` (default) | The exact low-dose slope: the full impact-parameter integral of \( 1-e^{-\nu(b)} \). Includes tracks that miss the nucleus, so it recovers \( \alpha_x \) at low LET, and keeps single-track overkill, so it thins down at very high LET. |
| `"lethal_central_saturating"` | rapidLEM's central-traversal slope \( (1-e^{-N_1})/D_1 \) (Krämer & Scholz 2006 eq. 9). It uses one central traversal only, so it undershoots \( \alpha_x \) at high energy. |
| `"lethal_central"` | The TRiP-style plateau variant: \( N_1/D_1 \) capped at the maximum local slope. |
| `"lethal_full"` | The overkill-free impact-parameter integral \( E[N]/D \). Not the exact slope, but the variant that matches the rapidLEM/TRiP98 tables. |
| `"full_simulation"` | LEM IV only: \( \alpha_I \) fitted to a simulated survival curve, the TRiP98 protocol. |

LEM I, II and III accept the first four; LEM IV accepts `"cross_section"` and
`"full_simulation"`.

### The quadratic term: `beta_model`

`LEM1`, `LEM2` and `LEM3` share three values:

| Value | Meaning |
| --- | --- |
| `"slope_matching"` (default) | \( \beta_I = (s_\text{max} - \alpha_I)/(2 D_t) \), with \( s_\text{max} = \alpha_x + 2\beta_x D_t \) (Krämer & Scholz 2006). Cheap. |
| `"rapid_lem"` | rapidLEM's \( \beta_D = f^2 \beta_z \) with \( f = (1-e^{-N_1})/N_1 \). |
| `"scholz1997_two_track"` | The exact \( O(\phi^2) \) pair-overlap coefficient of the explicit model, no slope matching anywhere. About a second per ion. |

`LEM1` additionally accepts the Scholz et al. 1997 reconstructions
`"scholz1997_stochastic"` (the fast specific-energy approximation, indicative only) and
`"scholz1997_montecarlo"` (the full stochastic Monte Carlo, slow).

!!! warning "Pair alpha and beta from the same tier"
    The default `LEM1()` combines the exact `"cross_section"` alpha with `"slope_matching"`
    beta. Slope matching assumes the capped rapidLEM alpha, so with the exact alpha it
    over-steepens the carbon beta at low energy. For quantitative work choose a coherent
    bundle:

    | Bundle | `initial_slope_model` | `beta_model` |
    | --- | --- | --- |
    | Explicit LEM (exact) | `cross_section` | `scholz1997_two_track` |
    | rapidLEM, central traversal | `lethal_central_saturating` | `rapid_lem` |
    | rapidLEM / TRiP98 compatible | `lethal_full` | `slope_matching` |

The last row is the TRiP98/rapidLEM compatibility mode. There is no single switch for it:
it is this combination of the two options above. A compatible, fast model:

```python
from pyradbio.models.lem.lem1 import LEM1

photon = PhotonParameters(alpha_x=0.18, beta_x=0.028, D_t=30.0)
rapid = LEM1(
    photon=photon,
    cell=cell,
    initial_slope_model="lethal_central_saturating",
    beta_model="rapid_lem",
)
print(rapid.alpha_beta(carbon))
```

The exact bundle and the Monte-Carlo routes cost more and are not run on this page:

```py
exact = LEM1(
    photon=photon,
    cell=cell,
    initial_slope_model="cross_section",
    beta_model="scholz1997_two_track",
)
exact.alpha_beta(carbon)  # about a second per ion

# full stochastic reference: Kellerer-weighted sampling, seeded
mc = LEM1(photon=photon, cell=cell, beta_model="scholz1997_montecarlo")
mc.alpha_beta(carbon)
survival = mc.scholz1997_montecarlo_survival(carbon, [0.5, 1.0, 2.0], n_realizations=20000, seed=0)
```

## Monte-Carlo engines and numba

The local-dose Monte Carlo lives on `LEMBase`, so it serves LEM I, II and III alike. Its
methods take an `engine` argument, a literal (`pyradbio.models.lem.engine.Engine`) with
three values:

- `"numpy"`: the vectorised, seed-reproducible reference and the accuracy anchor.
- `"numba"`: the compiled parallel kernel. It needs the `fast` extra
  (`pip install "pyradbio[fast]"`) and raises without it. Its per-thread random numbers
  make it not bit-reproducible for a fixed seed.
- `"auto"`: numba when it is importable, otherwise numpy.

`LEM4` carries the same literal as its `engine` field, which selects the backend of its
DSB-pattern scoring. Use `engine="numpy"` for any tight assertion or regression test.

## LEM IV

LEM IV replaces the pointwise local-dose response by a statistic of the DSB pattern in
540 nm sub-volumes: the single-particle cluster index \( C_I \) is matched to a
photon-equivalent dose, and that sets the lethal events. It uses the velocity-core
radial dose with \( r_c = 6.5 \) nm and the built-in SSB-to-DSB enhancement of Friedrich
et al. 2015, so a `ClusterParameters` on the photon object is ignored (with a warning).

```py
from pyradbio.models.lem.lem4 import LEM4

# V79 reference of Friedrich et al. 2012, Fig. 3
photon4 = PhotonParameters(alpha_x=0.184, beta_x=0.02, D_t=17.0)
lem4 = LEM4(photon=photon4, cell=CellTarget(nucleus_radius_um=4.7))
lem4.alpha_beta(carbon)
```

The options, each a `Literal` validated at construction:

| Field | Values | Selects |
| --- | --- | --- |
| `effect_model` | `"photon_equivalent"` (default), `"globle"` | How a DSB pattern maps to lethal events: the \( D_\text{eq} \) mapping of Elsässer 2010, or the bounded per-loop GLOBLE effect of Friedrich, Durante & Scholz 2012 (doi:10.1667/RR2964.1). |
| `globle_n_loops` | float | GLOBLE's giant-loop count \( N_L \). The paper never applies GLOBLE to ions, so this is a free, configurable parameter. |
| `beta_model` | `"slope_matching"` (default), `"rapid_lem"`, `"intertrack_mc"`, `"local_dose"`, `"discrete_dsb"` | The last three are the Friedrich 2012 Fig. 4 full simulations (`local_dose_beta`, `discrete_dsb_beta`, `intertrack_mc_beta`). `"scholz1997_two_track"` is not defined for LEM IV. |
| `cluster_index_model` | `"mean_field"` (default), `"discrete"`, `"discrete_mc"`, `"trip98_composite"` | Which estimator supplies the single-particle cluster index. None is right everywhere: the mean-field estimator over-clusters at low energy, the discrete one is deterministic, `"discrete_mc"` is sampled and seeded, and `"trip98_composite"` is for reproducing TRiP98 tables only. |
| `local_dose_kernel` | `"cluster_index"` (default), `"discrete_cluster_index"`, `"photon_effect"`, `"globle"` | How the local-dose convolution maps its dose density to lethal events. |
| `survival_convention` | `"mean_survival"` (default), `"mean_effect"` | Whether the single-particle slope averages \( e^{-E} \) or \( E \). |
| `initial_slope_model` | `"cross_section"` (default), `"full_simulation"` | `"full_simulation"` fits \( \alpha_I \) to a simulated survival curve (`full_simulation_initial_slope`). |
| `engine` | `"auto"`, `"numpy"`, `"numba"` | The execution engine, see above. |

Switching to GLOBLE, and choosing a different cluster-index estimator, are plain keyword
arguments:

```py
globle = LEM4(photon=photon4, cell=CellTarget(nucleus_radius_um=4.7), effect_model="globle")
sampled = LEM4(
    photon=photon4,
    cell=CellTarget(nucleus_radius_um=4.7),
    cluster_index_model="discrete_mc",
    cluster_index_mc_seed=0,
)
```

The fitted routes (`local_dose`, `discrete_dsb`, `full_simulation`) read their dose window
from `LEM4.fit_doses_Gy`, a grid given as fractions of the cell's \( D_t \); that window is a
pyRadBio choice that no paper documents, so quote it with any fitted number. Several LEM IV
choices are of this kind; each free parameter, fork and known defect is registered, with
the evidence, in the
[LEM IV open-choices register](https://github.com/e0404/pyRadBio/blob/develop/docs/references/LEM_IV_OPEN_CHOICES.md).

## Accuracy and known deviations

Each version is checked against its source paper, and where a reconstruction departs from a
published figure the departure is documented rather than hidden. The known deviations and
open items are listed in
[`ROADMAP.md`](https://github.com/e0404/pyRadBio/blob/develop/ROADMAP.md) and in the paper
digests in `docs/references/`. Read the relevant digest before trusting a number in a regime
the paper did not cover, in particular LEM IV at low energy.

## Further reading

Paper digests (contributor notes, linked on GitHub):

- [LEM I: Scholz & Kraft 1996 and Scholz et al. 1997](https://github.com/e0404/pyRadBio/blob/develop/docs/references/LEM_I_ScholzKraft.md)
- [rapidLEM: Krämer & Scholz 2006](https://github.com/e0404/pyRadBio/blob/develop/docs/references/rapidLEM_KraemerScholz2006.md)
- [LEM II: Elsässer & Scholz 2007](https://github.com/e0404/pyRadBio/blob/develop/docs/references/LEM_II_Elsasser2007.md)
- [LEM III: Elsässer et al. 2008](https://github.com/e0404/pyRadBio/blob/develop/docs/references/LEM_III_Elsasser2008.md)
- [Track structure: Elsässer et al. 2008b](https://github.com/e0404/pyRadBio/blob/develop/docs/references/LEM_TrackStructure_Elsasser2008b.md)
- [LEM IV: Elsässer et al. 2010](https://github.com/e0404/pyRadBio/blob/develop/docs/references/LEM_IV_Elsasser2010.md)
- [LEM IV and GLOBLE: Friedrich et al. 2012](https://github.com/e0404/pyRadBio/blob/develop/docs/references/LEM_IV_Friedrich2012.md)

Examples (they plot with the `plot` extra):

- [`lem1_variants.py`](https://github.com/e0404/pyRadBio/blob/develop/examples/lem1_variants.py):
  the LEM I approximation ladder on both axes and the coherent bundles.
- [`model_comparison_alpha_beta.py`](https://github.com/e0404/pyRadBio/blob/develop/examples/model_comparison_alpha_beta.py):
  LEM I-IV beside MCF MKM, modified MKM and RMF on one V79 reference.

See also the [Examples](../examples.md) page and the [model overview](index.md).
