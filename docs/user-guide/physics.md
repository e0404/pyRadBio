# Physics building blocks

Every model in pyRadBio sits on the same small set of physics layers: a stopping power
that turns an ion and an energy into a LET, an amorphous radial-dose track, microdosimetric
spectra, and two model-agnostic axes (oxygen and time). This page shows how to use each
layer on its own. The models consume them through typed arguments (a `stopping_power`
field, a radial-dose object), so you can swap any of them without touching model code.

Throughout, an ion is an [`Ion`][pyradbio.core.radiation.Ion] with atomic number `Z`, mass
number `A` and **kinetic energy per nucleon in MeV/u**. LET is the unrestricted electronic
stopping power in liquid water in **keV/µm**. See [Units and conventions](../parameter_conventions.md)
for the rest.

## Stopping power

### The default source

[`default_stopping_power()`][pyradbio.track.stopping_power.default_stopping_power] is the one
factory every model default goes through. It returns a
[`DefaultStoppingPower`][pyradbio.track.stopping_power.DefaultStoppingPower] that combines
three branches:

| Ion | Source |
| --- | --- |
| Z 1-18, inside the procedure's energy range | [`ICRU90Procedure`][pyradbio.track.stopping.icru90_procedure.ICRU90Procedure] (ICRU 90's procedure for water) |
| Z 19-92, from 2.5 MeV/u | [`HubertAlphaScaling`][pyradbio.track.stopping.hubert.HubertAlphaScaling] (Hubert, Bimbot and Gauvin 1990, scaling ICRU 90's alpha curve) |
| everywhere else | analytic [`SimpleBetheStoppingPower`][pyradbio.track.stopping_power.SimpleBetheStoppingPower], anchored to the adjacent branch at its range edge so the curve stays continuous |

The Bethe fallback is unvalidated for Z above 18 (no water reference is held there). The
class docstring and
[`docs/references/StoppingPower.md`](https://github.com/e0404/pyRadBio/blob/develop/docs/references/StoppingPower.md)
give the measured accuracy of each branch and the seams between them.

```python
from pyradbio import Ion
from pyradbio.track.stopping_power import default_stopping_power, describe_stopping_power

sp = default_stopping_power()
print(describe_stopping_power(sp))

carbon = Ion(Z=6, A=12, energy_MeV_u=100.0)
print(sp.let_keV_um(carbon))  # keV/um
print(sp.source(carbon))  # which branch served it

# a whole energy grid for one species, as an array
print(sp.lets_keV_um(6, 12, [10.0, 50.0, 100.0]))
```

`sp.source(ion)` names the serving branch (`"icru90_procedure"`, `"hubert1990"`,
`"simple_bethe_anchored"` or `"simple_bethe"`), which is the quickest way to see when a
result leans on the unvalidated fallback.

!!! note "Use the factory, not a hard-coded source"
    Pass `default_stopping_power()` (or leave a model's `stopping_power` field at its
    default) rather than constructing Bethe yourself. Where a test or reconstruction needs
    plain analytic Bethe, construct
    [`SimpleBetheStoppingPower`][pyradbio.track.stopping_power.SimpleBetheStoppingPower]
    explicitly.

### Other sources

Every other source is an explicit opt-in, passed to a model's `stopping_power` field (or to
`pyradbio table --stopping-power`, see [Tables and command line](tables-cli.md)).

| Source | What it is |
| --- | --- |
| [`ICRU`][pyradbio.track.stopping.icru.ICRU] | The ICRU 49 / 73 / 90 water tables, chosen with `edition="2005"`, `"2009"` or `"2016"` (Z 1-18, no extrapolation) |
| [`XSTAR`][pyradbio.track.stopping.xstar.XSTAR] | NIST PSTAR / ASTAR with MSTAR for Z 3-18 |
| [`MSTAR`][pyradbio.track.stopping.mstar.MSTAR] | A Python port of MSTAR 3.12 (Paul and Schinner), under its authors' freeware notice |
| [`Bethe`][pyradbio.track.stopping.bethe.Bethe] | Bethe (PDG 2024) with switchable effective charge, density effect, shell, Barkas, Bloch, Mott, Ahlen and Lindhard-Sorensen corrections |
| [`WaterNuclearStopping`][pyradbio.track.stopping.nuclear.WaterNuclearStopping] | Nuclear stopping, added to any of the above with `quantity="total"` |
| [`NorthcliffeSchillingWithBetheTail`][pyradbio.track.northcliffe_schilling.NorthcliffeSchillingWithBetheTail] | The Northcliffe and Schilling 1970 water table plus a Bethe tail, for reconstructions that must use the paper-era source |
| `pyradbio.track.srim` | A reader for user-supplied SRIM tables (`srim_stopping_power`) |
| [`LibdedxStoppingPower`][pyradbio.track.stopping_power.LibdedxStoppingPower] | libdEdx, which is GPL and loaded only when you install it yourself |

All of the table classes in `pyradbio.track.stopping` share one interface: they take the
energy per nucleon, raise `OutOfRangeError` rather than extrapolating, and record their
source per ion. For example:

```python
from pyradbio import Ion
from pyradbio.track.stopping import ICRU, ICRU90Procedure

icru90 = ICRU(edition="2016")
print(icru90.let_keV_um(Ion(6, 12, 100.0)))
print(icru90.energy_range(6, 12))  # MeV/u range this table covers

# the procedure behind the library default, on its own
print(ICRU90Procedure().let_keV_um(Ion(6, 12, 100.0)))
```

### Helpers

- [`describe_stopping_power()`][pyradbio.track.stopping_power.describe_stopping_power] returns
  a readable label for any source, for plot titles and log lines. Use it instead of
  hard-coding "analytic Bethe".
- [`energy_for_let_MeV_u()`][pyradbio.track.stopping_power.energy_for_let_MeV_u] inverts
  LET(E). LET is not monotone in energy, so a LET below the Bragg peak has two solutions:
  `branch="high"` (the default) returns the one above the peak, `branch="low"` the one
  below, and `branch="bracket"` runs a plain bracketed root solve on `energy_bounds_MeV_u`.
  It raises `ValueError` for a LET the ion never reaches.
- [`let_from_mass_stopping_power()`][pyradbio.track.stopping_power.let_from_mass_stopping_power]
  converts a mass stopping power in MeV cm²/g (the unit of the published tables) to keV/µm.

```python
from pyradbio.track.stopping_power import (
    default_stopping_power,
    energy_for_let_MeV_u,
    let_from_mass_stopping_power,
)

sp = default_stopping_power()
print(energy_for_let_MeV_u(sp, 6, 12, 50.0))  # MeV/u of a carbon ion at 50 keV/um
print(let_from_mass_stopping_power(10.0))  # 10 MeV cm^2/g in water, in keV/um
```

A runnable plot of LET against energy for several ions is
[`examples/water_stopping_power.py`](https://github.com/e0404/pyRadBio/blob/develop/examples/water_stopping_power.py).

## Radial dose and track structure

The LEM family and the MKM family both integrate an amorphous radial dose profile
\( D(r) \) around the ion path. A radial-dose object implements the
[`RadialDoseModel`][pyradbio.track.base.RadialDoseModel] protocol: `local_dose_Gy(r_um, ion)`
plus the radii `r_min_um`, `r_max_um` and `r_support_um`. Each profile fixes its core
amplitude by energy conservation, so integrating \( D(r) \) over the plane reproduces the
LET of the stopping power it was built with.

[`ScholzKraftRadialDose`][pyradbio.track.scholz_kraft.ScholzKraftRadialDose] is the
\( 1/r^2 \) track of the LEM papers, with the delta-electron range
\( r_\text{max} = \gamma E^{\delta} \) (\( E \) in MeV/u, \( \delta = 1.7 \)). Three
constructors give the LEM-version variants:

- `lem1()`: a fixed 10 nm core, no radical diffusion, \( \gamma = 0.05\ \mu\text{m} \).
- `lem2()`: a 0.3 nm core rounded by a Gaussian radical-diffusion convolution
  (\( \sigma = 4 \) nm), \( \gamma = 0.062\ \mu\text{m} \).
- `lem3()`: a velocity-dependent core \( r_\text{min} = \beta r_c \) plus the same radical
  diffusion.

[`KieferChatterjeeRadialDose`][pyradbio.track.kiefer_chatterjee.KieferChatterjeeRadialDose]
is the Kiefer-Chatterjee track underlying the MKM family: a velocity-dependent constant core
plus a \( 1/r^2 \) penumbra.

```python
from pyradbio import Ion
from pyradbio.track.kiefer_chatterjee import KieferChatterjeeRadialDose
from pyradbio.track.scholz_kraft import ScholzKraftRadialDose
from pyradbio.track.stopping_power import default_stopping_power

sp = default_stopping_power()  # one LET source feeds every track model
ion = Ion(Z=6, A=12, energy_MeV_u=100.0)

models = {
    "LEM I": ScholzKraftRadialDose.lem1(sp),
    "LEM II": ScholzKraftRadialDose.lem2(sp),
    "Kiefer-Chatterjee": KieferChatterjeeRadialDose(stopping_power=sp),
}
for name, track in models.items():
    print(name, track.r_max_um(ion), track.local_dose_Gy(0.01, ion))  # um, Gy at r = 10 nm

# energy conservation: the radial integral returns the LET
print(models["LEM I"].radial_integral_LET_keV_um(ion), sp.let_keV_um(ion))
```

`local_doses_Gy(r_um, ion)` evaluates an array of radii. Use `r_support_um` (not `r_max_um`)
as the outer integration radius: for LEM II and III the radical diffusion carries dose past
the delta-electron range.

A radial-dose object built on its own needs a `stopping_power` (it raises `ValueError` without
one), so pass `default_stopping_power()`. The LEM and MKM models build their track objects
themselves, so you only construct one directly to inspect a profile or to feed the
microdosimetry layer below. See
[`examples/radial_dose_profiles.py`](https://github.com/e0404/pyRadBio/blob/develop/examples/radial_dose_profiles.py)
for all four profiles plotted side by side.

## Microdosimetry

[`pyradbio.micro`][pyradbio.micro] provides single-event lineal-energy spectra
\( f_1(y) \) and their dose-weighted form \( d_1(y) = y f_1(y) / y_F \), with the moments
\( y_F \), \( y_D \) and the saturation-corrected \( y^* \). Each engine returns a
[`Spectrum`][pyradbio.micro.lineal_energy.Spectrum]. The site is a
[`SphericalSite`][pyradbio.micro.lineal_energy.SphericalSite] or a
[`CylindricalSite`][pyradbio.micro.lineal_energy.CylindricalSite] (TEPC-like).

| Engine | Route |
| --- | --- |
| [`AnalyticSpectrumEngine`][pyradbio.micro.lineal_energy.AnalyticSpectrumEngine] | Analytic (Xapsos-Kellerer): a stopping-power lookup, an energy-loss straggling model and the site's chord distribution. No radial-dose model needed; fast and self-contained |
| [`AmorphousTrackEngine`][pyradbio.micro.lineal_energy.AmorphousTrackEngine] | Impact-parameter method over any `RadialDoseModel`: resolves the delta-ray tail but is deterministic (no straggling) |
| [`SatoLinealEnergyEngine`][pyradbio.micro.sato.SatoLinealEnergyEngine] | Sato's analytical microdosimetric function (AMF), the form behind PHITS `[T-SED]`; coefficients come from `pyradbio.micro.sato_coefficients.sato_coefficients_for` for H, He, Li, C, O and Ne |
| [`TabulatedLinealEnergySpectrum`][pyradbio.micro.lineal_energy.TabulatedLinealEnergySpectrum] | A binned count histogram from a measurement or Monte-Carlo library, e.g. `load_decunha_lineal_energy_csv` for the DeCunha 2025 proton library. The digitized AMF and TOPAS-nBio curves of Hartzell 2025 ship in `pyradbio.micro.hartzell2025` |

[`LinealEnergyCalculator`][pyradbio.micro.lineal_energy.LinealEnergyCalculator] is the
facade: give it a stopping power and a site, and it builds the analytic engine (or inject
another engine).

```python
from pyradbio import Ion
from pyradbio.micro import LinealEnergyCalculator, SphericalSite
from pyradbio.track.stopping_power import default_stopping_power

calc = LinealEnergyCalculator(
    stopping_power=default_stopping_power(),
    site=SphericalSite(diameter_um=1.0),
)
spectrum = calc.spectrum(Ion(Z=6, A=12, energy_MeV_u=100.0))

print(spectrum.y_F, spectrum.y_D)  # keV/um; y_F equals the LET by construction
print(spectrum.y_star(150.0))  # saturation-corrected, y0 = 150 keV/um
```

!!! warning "Product integrals are not evaluated at bin centres"
    Quantities such as \( y^* \) and the MKM \( z^*_{1D} \) integrate products of a spectrum
    with a response function. The library does this with proper quadrature, and the
    `Spectrum` moments above use Simpson integration. If you integrate a binned spectrum
    yourself, do not collapse each bin to its centre: that is the most common silent error
    in this field (see `AGENTS.md`).

### Multi-event specific energy

[`pyradbio.micro.multi_event`](https://github.com/e0404/pyRadBio/blob/develop/pyradbio/micro/multi_event.py)
turns a single-event specific-energy density into the multi-event density \( f(z; D) \) at a
dose \( D \), with Poisson-distributed traversals (Sato and Furusawa 2012, eqs. 24-28).
`single_event_density(let_keV_um=..., radius_um=...)` builds \( f_1(z) \) as probability
mass on a uniform grid, and `multi_event_density(z, mass, dose_Gy=...)` returns
\( f(z; D) \) from one forward and one inverse FFT at any dose. This is what the stochastic
MK models integrate survival against.

### Impact-parameter integration

For the MKM family you rarely need a spectrum at all.
[`specific_energy_impact_profile()`][pyradbio.micro.impact_parameter.specific_energy_impact_profile]
integrates any radial-dose model over the target for every impact parameter \( b \) and
returns a [`SpecificEnergyProfile`][pyradbio.micro.impact_parameter.SpecificEnergyProfile]
with the single-event moments in closed form: `frequency_mean_Gy()`, `dose_mean_Gy()` and the
saturation-corrected `saturation_corrected_dose_mean_Gy(z0_Gy)`. The default geometry is the
disk used by the MKM domain; `geometry="sphere"` is kept for comparison.

```python
from pyradbio import Ion
from pyradbio.micro import specific_energy_impact_profile
from pyradbio.track.kiefer_chatterjee import KieferChatterjeeRadialDose
from pyradbio.track.stopping_power import default_stopping_power

ion = Ion(Z=6, A=12, energy_MeV_u=100.0)
profile = specific_energy_impact_profile(
    KieferChatterjeeRadialDose(stopping_power=default_stopping_power()),
    ion,
    region_radius_um=0.44,  # an illustrative domain radius [um]
)
print(profile.let_keV_um, profile.dose_mean_Gy())
```

[`nuclear_functional_impact_profile()`][pyradbio.micro.impact_parameter.nuclear_functional_impact_profile]
does the same for an arbitrary functional of the local dose over the nucleus disk, which is
how the LEM single-track effect and the inactivation cross-section are built. See the
[MKM family](models/mkm.md) page for how the models use these routes, and
[`examples/lineal_energy_spectra.py`](https://github.com/e0404/pyRadBio/blob/develop/examples/lineal_energy_spectra.py)
for the analytic, amorphous-track and Monte-Carlo engines compared.

## The oxygen axis

Oxygenation is an environment variable, like dose, so it is passed to a model as an explicit
`pO2_mmHg` argument rather than stored on the cell or the field.
[`pyradbio.core.oxygen`](https://github.com/e0404/pyRadBio/blob/develop/pyradbio/core/oxygen.py)
holds the model-agnostic piece every OER model in the literature shares, the Alper and
Howard-Flanders hyperbola

\[
R(p\text{O}_2) = \frac{p\text{O}_2 + K}{p\text{O}_2 + K/R_\text{max}} ,
\]

the factor by which the saturation-corrected specific energy must be increased at that
\( p\text{O}_2 \) to reach the aerobic effect. \( R(0) = R_\text{max} \) and
\( R \to 1 \) as \( p\text{O}_2 \to \infty \). The constants
`ANOXIC_PO2_MMHG`, `AEROBIC_PO2_MMHG` (160 mmHg, the experimental aerobic reference, where
\( R \) is slightly above 1), `NORMOXIC_TISSUE_PO2_MMHG` and `K_HALF_EFFECT_MMHG` are
exported.

```python
from pyradbio.core.oxygen import OxygenCondition, relative_radioresistance

# r_max is required and has no default: it is a model fit output, not a constant
print(relative_radioresistance([0.0, 3.0, 160.0], k_mmHg=3.0, r_max=2.0))  # illustrative r_max

hypoxic = OxygenCondition(pO2_mmHg=5.0)
print(hypoxic.relative_radioresistance(r_max=2.0))
```

The `r_max` above is an illustrative argument, not a physical value: in OSMK it comes from
the fitted model. The models that use this axis are described under the
[MKM family](models/mkm.md); the OSMK example is
[`examples/osmk_oxygen_effect.py`](https://github.com/e0404/pyRadBio/blob/develop/examples/osmk_oxygen_effect.py).

## The time axis

The LQ model as used everywhere else in pyRadBio describes an acute dose. For a protracted
or fractionated delivery the quadratic term shrinks by the Lea-Catcheside factor
\( G \le 1 \):

\[
-\ln S(D) = \alpha D + G\,\beta D^2 .
\]

\( G \) depends only on the temporal pattern of the dose and the sublethal-damage repair
half-time, not on the cell, the dose or the radiation quality.
[`pyradbio.core.protraction`](https://github.com/e0404/pyRadBio/blob/develop/pyradbio/core/protraction.py)
therefore carries no biology: a
[`DoseSchedule`][pyradbio.core.protraction.DoseSchedule] is a sequence of non-overlapping
constant-rate [`DoseBlock`][pyradbio.core.protraction.DoseBlock]s (Gy and hours), and
[`lea_catcheside_g()`][pyradbio.core.protraction.lea_catcheside_g] evaluates the exact
closed form, with no quadrature.

Build a schedule with a class method: `DoseSchedule.acute(dose)`,
`DoseSchedule.continuous(dose, duration_h=...)`,
`DoseSchedule.at_dose_rate(rate, duration_h=...)` or
`DoseSchedule.fractionated(d, n_fractions=..., interval_h=..., fraction_duration_h=0.0)`.
There is deliberately **no default repair half-time**: it is the most influential input
(the literature spans roughly 0.2-5 h), so you pass `repair_half_time_h` explicitly.

```python
from pyradbio import DoseSchedule, bed, eqd2, lea_catcheside_g, survival

T_HALF_H = 1.5  # an explicit choice, not a library default

course = DoseSchedule.fractionated(2.0, n_fractions=30, interval_h=24.0)
g_course = lea_catcheside_g(course, repair_half_time_h=T_HALF_H)

brachy = DoseSchedule.continuous(2.0, duration_h=1.0)
g_brachy = lea_catcheside_g(brachy, repair_half_time_h=T_HALF_H)

print(g_course, g_brachy)
print(lea_catcheside_g(DoseSchedule.acute(2.0), repair_half_time_h=T_HALF_H))  # exactly 1
```

The factor enters the LQ helpers as a keyword-only `g` (default `1.0`, the acute case) on
[`survival`][pyradbio.core.lq.survival], [`log_kill`][pyradbio.core.lq.log_kill],
[`dose_for_survival`][pyradbio.core.lq.dose_for_survival] and
[`rbe`][pyradbio.core.lq.rbe], and on [`bed`][pyradbio.core.lq.bed] and
[`eqd2`][pyradbio.core.lq.eqd2]. `rbe` takes a separate `reference_g`, because the test and
reference arms need not share a schedule, and an RBE quoted against an acute photon
reference otherwise mixes the fractionation effect into the RBE.

```python
# an illustrative LQ pair (alpha/beta = 10 Gy), not a fitted tissue
alpha, beta = 0.3, 0.03

print(survival(2.0, alpha, beta))  # acute
print(survival(2.0, alpha, beta, g=g_brachy))  # same dose, delivered over an hour

# fractionation isoeffect: for n fractions with full repair, g = 1/n
print(bed(60.0, alpha_beta_ratio_Gy=10.0, g=1 / 30))
print(eqd2(60.0, alpha_beta_ratio_Gy=10.0, g=1 / 30))
```

`G` corrects the quadratic term only and assumes single-exponential repair; the
[module docstring](https://github.com/e0404/pyRadBio/blob/develop/pyradbio/core/protraction.py)
lists the validity bounds from the source papers. A multi-exponential repair kernel needs no
new machinery, since \( G \) is linear in the kernel:
`sum(w_i * lea_catcheside_g(schedule, repair_half_time_h=T_i))`. The runnable walk-through,
with plots, is
[`examples/dose_protraction_and_bed.py`](https://github.com/e0404/pyRadBio/blob/develop/examples/dose_protraction_and_bed.py).

## Chromatin-based domain sizing

[`pyradbio.core.chromatin`](https://github.com/e0404/pyRadBio/blob/develop/pyradbio/core/chromatin.py)
sizes a subnuclear domain (the MKM domain, the LEM IV giant loop) from cell geometry alone,
following Parisi et al. 2022: the domain is the sub-volume of the nucleus holding about
2 Mbp of DNA,
\( r_d = R_n (\lambda / \Gamma)^{1/3} \), with \( \Gamma \) the nuclear DNA content from the
haploid genome size, the ploidy and the cell-cycle factor.

```python
from pyradbio import domain_radius_from_genome

# nucleus radius 5 um, human genome, defaults: diploid, asynchronized, 2 Mbp loops
print(domain_radius_from_genome(5.0, species="human"))
```

`species` is one of `human`, `rat`, `chinese_hamster` and `mouse`; pass
`haploid_genome_mbp` instead for another species, and `ploidy` and `cell_cycle` (`"g1"`,
`"g2"` or `"asynchronized"`) as needed. The building blocks
`nuclear_dna_content_mbp`, `chromatin_domain_radius_um` and `ploidy_from_chromosome_count`
are exported at the top level too.
